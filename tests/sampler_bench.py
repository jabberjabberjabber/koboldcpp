#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Sampler benchmark for KoboldCpp.
# Compares how much slop different sampler configs (including anti-slop) let through, and what they cost.
# It works in separate passes, each one keeps its output in a file so it can be reused:
#
#   corpus  Generate plain texts from many prompts and add them to a corpus file. Works with any OpenAI compatible
#           API, so a large corpus can be made quickly on a hosted service and kept for later.
#   mine    Find the phrases a corpus repeats across different prompts, and write them to a slop list text file.
#           Review and edit that file by hand, it will contain some ordinary phrases too.
#   run     Run the same prompts and seeds through each sampler config on a running KoboldCpp, counting the slop.
#           Results are added to a json file, so running it again with another model loaded compares across models.
#   show    Print the comparison from a results file without running anything.
#
# Examples:
#   python tests/sampler_bench.py corpus --url https://integrate.api.nvidia.com/v1 --model meta/llama-3.3-70b-instruct \
#       --api-key-env NVIDIA_API_KEY --num-prompts 100 --out corpus.json
#   python tests/sampler_bench.py mine --corpus corpus.json --out slop.txt --top 40
#   python tests/sampler_bench.py run --slop slop.txt --seeds 3 --out bench.json
#   python tests/sampler_bench.py run --slop slop.txt --configs baseline,antislop_bt \
#       --add "bt_t60:antislop=true,antislop_backtrack=12,antislop_threshold=0.6"
#   python tests/sampler_bench.py show --out bench.json --dump antislop_bt
#
# A slop dataset can be used instead of a mined list with --dataset, it is a json list of entries shaped like
#   {"slop_reason_list": ["regex", ...], "prompt": [{"from": "human", "value": "..."}]}
# e.g. Jobaar/NobodyExistsOnTheInternet_full120k-SlopOnly-KTOSloPreferenceShareGPT on huggingface (MIT).
# That one is mostly adult roleplay, use --num-prompts 0 to take only its slop patterns and not its prompts,
# or --exclude-prompts "regex" to leave out prompts you do not want to run.

import argparse
import itertools
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

# phrases counted as slop in every output, and banned by configs that have "antislop": true
# only used when no --slop or --dataset is given
# taken from the default list in the look-ahead sampler PR, names and weights removed
default_slop = [
    "delve", "tapestry", "testament to", "a sense of", "it's worth noting", "it is worth noting",
    "it's important to note", "it is important to note", "in conclusion", "in summary",
    "barely above a whisper", "shivers down", "shiver down", "sends shivers", "sent shivers",
    "maybe, just maybe", "ministrations", "palpable", "a mix of", "a mixture of", "eyes sparkling",
    "eyes glinting", "with mischief", "mischievous glint", "husky voice", "voice barely",
    "can't help but", "couldn't help but", "the air was thick", "hung heavy in the air", "little did",
    "unspoken", "a dance of", "intricate dance", "symphony of", "ever-evolving",
    "ever-changing landscape", "navigate the complexities", "rich tapestry", "embark on",
    "journey of self", "i hope this helps", "feel free to", "great question", "certainly!",
    "absolutely!", "let's dive", "dive into", "stark reminder", "a reminder that",
    "in a world where", "game-changer", "unwavering", "profound", "resonate",
]

# the built in prompts are every genre combined with every situation, in a fixed shuffled order
prompt_genres = ["gothic romance", "high fantasy", "hard science fiction", "noir detective", "cozy mystery", "western",
    "post-apocalyptic", "historical drama", "supernatural horror", "slice of life", "political thriller", "fairy tale"]
prompt_situations = ["two rivals are forced to work together", "someone returns home after many years away",
    "a secret is revealed at the worst possible moment", "a stranger arrives with an unusual request",
    "a mentor says goodbye to their student", "someone must choose between duty and love",
    "a long planned heist goes wrong", "two old friends meet again as enemies",
    "someone discovers they have been lied to", "a leader addresses their people before a battle",
    "a traveller takes shelter from a storm", "someone confesses something they have hidden for years"]

# settings shared by every config, a config overrides these
base_params = {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "rep_pen": 1.0}

# "antislop": true is not sent to the server, it means banned_tokens is set to the slop list
default_configs = {
    "baseline": {},
    "hightemp_minp": {"temperature": 1.2, "top_p": 1.0, "min_p": 0.05},
    "xtc": {"xtc_threshold": 0.1, "xtc_probability": 0.5},
    "dry": {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2},
    "antislop": {"antislop": True},
    "antislop_bt": {"antislop": True, "antislop_backtrack": 12, "antislop_threshold": 0.8},
}

# words that do not make a phrase interesting on their own
stopwords = set(("a an the and or but if then so of to in on at by for with from into onto over under as is was were be "
    "been being are am it its it's he she they them his her their him you your i me my we us our this that these "
    "those there here not no had has have do did does would could should will can may might just very too than up "
    "down out off about what who which when where how all any some each more most only own same said asked like "
    "against across through between around behind before after during toward towards upon within without beneath "
    "above below along among near beside beyond until while because though although one two back away again still "
    "even now never always something nothing everything").split())

def http_json(url, body=None, timeout=600, api_key=""):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as ex:
            if attempt == 4 or not (ex.code == 429 or ex.code >= 500): #only retry rate limits and server errors
                raise
            time.sleep(5 * attempt)

def chat_url(url):
    url = url.rstrip("/")
    return url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")

def parse_add(text):
    # "name:key=value,key=value", values are parsed as json where possible
    name, _, rest = text.partition(":")
    params = {}
    for item in rest.split(","):
        if item.strip() == "":
            continue
        key, _, val = item.partition("=")
        try:
            params[key.strip()] = json.loads(val.strip())
        except json.JSONDecodeError:
            params[key.strip()] = val.strip()
    return name.strip(), params

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip() != "" and not ln.startswith("#")]

def load_json(path, empty):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return empty

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

def regex_to_bans(pattern):
    # the server bans plain strings, so turn a simple slop regex into the strings it can match
    # returns an empty list if the regex is too complex, or too short to ban safely as a substring
    text = pattern.replace("\\.", ".")
    if text.startswith("\\b"):
        text = " " + text[2:] #only match at the start of a word
    text = text.replace("\\b", "")
    parts = []
    for piece in re.split(r"(\([^()]*\)\??)", text):
        if piece.startswith("("):
            optional = piece.endswith("?")
            choices = piece.rstrip("?")[1:-1].split("|")
            if any("\\" in c for c in choices): #something like \w+, we can only skip it
                if not optional:
                    return []
                choices = [""]
            elif optional:
                choices.append("")
            parts.append(choices)
        else:
            parts.append([piece])
    bans = ["".join(combo) for combo in itertools.product(*parts)]
    if any(re.search(r"[\\\[\]*+?{}^$|]", b) for b in bans):
        return []
    return [b for b in bans if len(b.strip()) > 4]

def load_dataset(path, num_prompts, exclude):
    data = load_json(path, [])
    counts = {}
    prompts = []
    for entry in data:
        for pattern in entry.get("slop_reason_list", []):
            counts[pattern] = counts.get(pattern, 0) + 1
        turns = [t["value"] for t in entry.get("prompt", []) if t.get("from") == "human"]
        if len(turns) > 0 and exclude and re.search(exclude, turns[-1], re.IGNORECASE):
            continue
        if len(turns) > 0 and turns[-1] not in prompts:
            prompts.append(turns[-1])
    patterns = sorted(counts, key=lambda k: -counts[k])
    random.Random(0).shuffle(prompts) #same prompts every time
    return patterns, prompts[:num_prompts]

def get_prompts(args, from_end):
    # the corpus pass takes built in prompts from the end of the list and the run pass from the start,
    # so the slop list is not mined from the same prompts it is tested on
    if args.prompts:
        return load_lines(args.prompts)[:args.num_prompts] if args.num_prompts > 0 else load_lines(args.prompts)
    if args.dataset and args.num_prompts > 0:
        return load_dataset(args.dataset, args.num_prompts, args.exclude_prompts)[1]
    built_in = [f"Write a {g} scene in which {s}. About 300 words." for g in prompt_genres for s in prompt_situations]
    random.Random(0).shuffle(built_in)
    count = args.num_prompts if args.num_prompts > 0 else 4
    return built_in[-count:] if from_end else built_in[:count]

def get_slop(args):
    # returns (names, regex patterns, strings to ban), or None if nothing was specified
    if args.dataset:
        patterns = load_dataset(args.dataset, 0, "")[0]
        bans = []
        for pattern in patterns:
            bans += [b for b in regex_to_bans(pattern) if b not in bans]
        return patterns, patterns, bans
    if args.slop:
        bans = load_lines(args.slop)
        return bans, [re.escape(b.lower()) for b in bans], bans
    return None

def phrases(text, nmin, nmax, min_content):
    # yields every phrase of nmin to nmax words that does not cross a sentence, exactly as it is written
    for segment in re.split(r"[.!?;:\n\"“”()\[\]*—–…]+|--| - ", text.lower()):
        words = segment.split()
        content = [0 if w.strip(",'’-") in stopwords else 1 for w in words]
        for n in range(nmin, nmax + 1):
            for i in range(len(words) - n + 1):
                if sum(content[i:i + n]) >= min_content:
                    yield " ".join(words[i:i + n]).strip(" ,'’-")

def mine_slop(texts, top, min_prompts):
    # texts is a list of (prompt, text). returns [(phrase, number of different prompts it appeared under)]
    seen = {}
    for prompt, text in texts:
        for phrase in set(phrases(text, 2, 5, 1)):
            seen.setdefault(phrase, set()).add(prompt)
    all_prompts = " ".join(p.lower() for p, _ in texts)
    found = [(p, len(s)) for p, s in seen.items() if len(s) >= min_prompts and p not in all_prompts]
    found = sorted(found, key=lambda x: (-x[1], -len(x[0])))[:3000]
    # a phrase that is nearly always part of the same longer phrase is dropped in favor of the longer one
    found = [(p, c) for p, c in found if not any(len(q) > len(p) and d >= 0.8 * c and p in q for q, d in found)]
    kept = []
    for phrase, count in found:
        if len(kept) >= top:
            break
        if not any(k in phrase for k, _ in kept): #already covered by a shorter phrase
            kept.append((phrase, count))
    return kept

def measure(text, slop):
    # slop is a list of (name, compiled regex)
    low = text.lower()
    words = re.findall(r"[a-z0-9']+", low)
    hits = {}
    for name, rx in slop:
        n = len(rx.findall(low))
        if n > 0:
            hits[name] = n
    bigrams = list(zip(words, words[1:]))
    return {"words": len(words), "hits": sum(hits.values()), "phrases": hits,
            "distinct2": (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0}

def count_repeats(runs):
    # how many 3 word phrases in these runs also appear under a different prompt, needs no slop list
    seen = {}
    total = 0
    for r in runs:
        for phrase in phrases(r["text"], 3, 3, 2):
            seen.setdefault(phrase, [set(), 0])
            seen[phrase][0].add(r["prompt"])
            seen[phrase][1] += 1
    for prompts, count in seen.values():
        total += count if len(prompts) > 1 else 0
    return total

def run_one(url, prompt, seed, params, bans, max_length):
    body = dict(base_params)
    body.update(params)
    if body.pop("antislop", False):
        body["banned_tokens"] = bans
    body["messages"] = [{"role": "user", "content": prompt}]
    body["max_tokens"] = max_length
    body["seed"] = seed
    start = time.time()
    resp = http_json(chat_url(url), body)
    elapsed = time.time() - start
    text = resp["choices"][0]["message"].get("content", "") or ""
    sampled = resp.get("usage", {}).get("completion_tokens", 0)
    kept = http_json(url + "/api/extra/tokencount", {"prompt": text, "special": False}).get("value", 0)
    # rewound tokens still count towards max_tokens, so the difference is what anti-slop threw away
    return {"text": text, "seconds": round(elapsed, 2), "sampled_tokens": sampled, "kept_tokens": kept}

def print_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    for row in [headers, ["-" * w for w in widths]] + rows:
        print("  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(row, widths))))

def show(results, override, dump):
    runs = results.get("runs", [])
    if len(runs) == 0:
        print("No results to show.")
        return
    models = []
    for r in runs:
        if r["model"] not in models:
            models.append(r["model"])
    per_model = {}
    for model in models:
        # each model is scored with the slop list it was run with, unless one is given now
        stored = results.get("slop", {}).get(model)
        names, patterns = override[:2] if override else ((stored["names"], stored["patterns"]) if stored else (default_slop, [re.escape(b) for b in default_slop]))
        slop = [(n, re.compile(p, re.IGNORECASE)) for n, p in zip(names, patterns)]
        configs = []
        for r in runs:
            if r["model"] == model and r["config"] not in configs:
                configs.append(r["config"])
        print(f"\n=== {model} ({len(slop)} slop patterns) ===")
        rows = []
        per_model[model] = {}
        for name in configs:
            mine = [r for r in runs if r["model"] == model and r["config"] == name]
            ms = [measure(r["text"], slop) for r in mine]
            words = sum(m["words"] for m in ms)
            hits = sum(m["hits"] for m in ms)
            sampled = sum(r["sampled_tokens"] for r in mine)
            secs = sum(r["seconds"] for r in mine)
            wasted = max(0, sampled - sum(r["kept_tokens"] for r in mine))
            found = {}
            for m in ms:
                for k, v in m["phrases"].items():
                    found[k] = found.get(k, 0) + v
            top = sorted(found.items(), key=lambda kv: -kv[1])[:3]
            per_model[model][name] = (1000.0 * hits / words) if words else 0
            rows.append([name, len(mine), words // len(mine), hits, f"{per_model[model][name]:.1f}",
                f"{(1000.0 * count_repeats(mine) / words) if words else 0:.1f}",
                f"{sum(m['distinct2'] for m in ms) / len(ms):.3f}",
                f"{(sampled / secs) if secs else 0:.1f}",
                f"{(100.0 * wasted / sampled) if sampled else 0:.0f}%",
                ", ".join(f"{k} x{v}" for k, v in top)])
        print_table(["config", "runs", "words/run", "slop", "slop/1k", "repeats/1k", "distinct-2", "tok/s", "rewound", "top slop"], rows)
    if len(models) > 1:
        print("\n=== slop per 1k words, by model ===")
        names = []
        for model in models:
            names += [n for n in per_model[model] if n not in names]
        rows = [[n] + [f"{per_model[m][n]:.1f}" if n in per_model[m] else "-" for m in models] for n in names]
        print_table(["config"] + [m[:28] for m in models], rows)
    print("\nslop = slop list phrases found in the output, slop/1k is per 1000 words.")
    print("repeats/1k = 3 word phrases that also appear under a different prompt, per 1000 words. Needs no slop list,")
    print("  so it shows if a sampler removed repetition or only swapped one repeated phrase for another.")
    print("distinct-2 = unique word pairs / all word pairs, higher is more varied.")
    print("rewound = sampled tokens that anti-slop discarded, these still count towards the max length.")
    if dump:
        for r in runs:
            if r["config"] == dump:
                print(f"\n--- {r['model']} | {r['config']} | {r['prompt'][:60]} | seed {r['seed']} ---\n{r['text']}")

def cmd_corpus(args):
    corpus = load_json(args.out, {"texts": []})
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if args.api_key_env and not api_key:
        sys.exit(f"Environment variable {args.api_key_env} is not set.")
    url = args.url.rstrip("/")
    model = args.model
    if not model: #no model given, so this should be a KoboldCpp instance
        try:
            model = http_json(url + "/api/v1/model", timeout=10).get("result", "unknown").replace("koboldcpp/", "")
        except Exception as ex:
            sys.exit(f"Could not get the model name from {url}, use --model to set it: {ex}")
    prompts = get_prompts(args, True)
    todo = [(p, s) for p in prompts for s in range(1001, 1001 + args.seeds)
            if not any(t["model"] == model and t["prompt"] == p and t["seed"] == s for t in corpus["texts"])]
    print(f"Model: {model}\n{len(prompts) * args.seeds - len(todo)} texts already in {args.out}, generating {len(todo)} more")
    for n, (prompt, seed) in enumerate(todo):
        print(f"\r[{n + 1}/{len(todo)}]", end="", flush=True)
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_length,
                "temperature": args.temperature, "top_p": 0.95, "seed": seed}
        resp = http_json(chat_url(url), body, api_key=api_key)
        text = resp["choices"][0]["message"].get("content", "") or ""
        corpus["texts"].append({"model": model, "prompt": prompt, "seed": seed, "text": text})
        save_json(args.out, corpus)
    print(f"\n{len(corpus['texts'])} texts in {args.out}")

def cmd_mine(args):
    corpus = load_json(args.corpus, {"texts": []})
    texts = [(t["prompt"], t["text"]) for t in corpus["texts"] if not args.model or t["model"] == args.model]
    if len(texts) == 0:
        sys.exit(f"No texts found in {args.corpus}")
    found = mine_slop(texts, args.top, args.min_prompts)
    num_prompts = len(set(p for p, _ in texts))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# mined from {len(texts)} texts over {num_prompts} prompts in {args.corpus}, one phrase per line\n")
        f.write("# remove the lines that are ordinary phrases and not slop before using this list\n")
        for phrase, count in found:
            f.write(phrase + "\n")
    print(f"Phrases repeated under at least {args.min_prompts} of {num_prompts} prompts:")
    print_table(["phrase", "prompts"], [[p, c] for p, c in found])
    print(f"\nSaved to {args.out}. Edit it to remove ordinary phrases, then use it with: run --slop {args.out}")

def cmd_run(args):
    results = load_json(args.out, {"runs": []})
    names, patterns, bans = get_slop(args) or (default_slop, [re.escape(b) for b in default_slop], default_slop)
    prompts = get_prompts(args, False)
    configs = dict(default_configs)
    if args.config_file:
        configs.update(load_json(args.config_file, {}))
    for item in args.add:
        name, params = parse_add(item)
        configs[name] = params
    if args.configs:
        wanted = [c.strip() for c in args.configs.split(",") if c.strip() != ""]
        missing = [c for c in wanted if c not in configs]
        if missing:
            sys.exit(f"Unknown config: {', '.join(missing)}. Available: {', '.join(configs)}")
        configs = {c: configs[c] for c in wanted}

    url = args.url.rstrip("/")
    try:
        model = args.label if args.label else http_json(url + "/api/v1/model", timeout=10).get("result", "unknown")
    except Exception as ex:
        sys.exit(f"Could not reach KoboldCpp at {url}: {ex}")
    model = model.replace("koboldcpp/", "")
    results.setdefault("slop", {})[model] = {"names": names, "patterns": patterns}
    total = len(configs) * len(prompts) * args.seeds
    print(f"Model: {model}\nCounting {len(names)} slop patterns, banning {len(bans)} phrases in anti-slop configs")
    print(f"Running {len(configs)} configs x {len(prompts)} prompts x {args.seeds} seeds = {total} generations")

    done = 0
    for name, params in configs.items():
        for prompt in prompts:
            for seed in range(1, args.seeds + 1):
                done += 1
                print(f"\r[{done}/{total}] {name}        ", end="", flush=True)
                run = run_one(url, prompt, seed, params, bans, args.max_length)
                run.update({"model": model, "config": name, "params": params, "prompt": prompt, "seed": seed})
                # a repeated run of the same thing replaces the old one
                results["runs"] = [r for r in results["runs"] if not (r["model"] == model and r["config"] == name and r["prompt"] == prompt and r["seed"] == seed)]
                results["runs"].append(run)
                save_json(args.out, results)
    print(f"\nSaved to {args.out}")
    show(results, None, args.dump)

def cmd_show(args):
    show(load_json(args.out, {"runs": []}), get_slop(args), args.dump)

def main():
    parser = argparse.ArgumentParser(description="Compare samplers (including anti-slop) on a running KoboldCpp instance. See the top of this file for examples.")
    sub = parser.add_subparsers(dest="command", required=True)

    prompt_args = argparse.ArgumentParser(add_help=False)
    prompt_args.add_argument("--prompts", default="", help="Text file with one prompt per line, replaces the built in prompts")
    prompt_args.add_argument("--num-prompts", type=int, default=0, help="Number of prompts to use (default 100 for corpus, 4 for run). With --dataset, 0 takes only its slop patterns")
    prompt_args.add_argument("--exclude-prompts", default="", metavar="REGEX", help="Skip --dataset prompts that match this regex, e.g. to leave out adult content")
    slop_args = argparse.ArgumentParser(add_help=False)
    slop_args.add_argument("--slop", default="", help="Text file with one slop phrase per line, e.g. made by the mine pass")
    slop_args.add_argument("--dataset", default="", help="Json slop dataset to take slop patterns (and prompts) from")
    slop_args.add_argument("--dump", default="", metavar="CONFIG", help="Also print the generated texts for this config")

    p = sub.add_parser("corpus", parents=[prompt_args], help="Generate texts from many prompts and add them to a corpus file")
    p.add_argument("--url", default="http://localhost:5001", help="KoboldCpp address, or the base url of an OpenAI compatible API")
    p.add_argument("--model", default="", help="Model name to request, needed for hosted APIs (default is the model loaded in KoboldCpp)")
    p.add_argument("--api-key-env", default="", metavar="NAME", help="Name of the environment variable that holds the API key, if one is needed")
    p.add_argument("--out", default="sampler_corpus.json", help="Corpus file, new texts are added to it (default sampler_corpus.json)")
    p.add_argument("--seeds", type=int, default=1, help="Number of texts to generate for each prompt (default 1)")
    p.add_argument("--max-length", type=int, default=450, help="Max tokens to generate per text (default 450)")
    p.add_argument("--temperature", type=float, default=0.8, help="Temperature to generate with (default 0.8)")
    p.add_argument("--dataset", default="", help="Json slop dataset to take the prompts from")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("mine", help="Find the phrases a corpus repeats across prompts, and write them to a slop list")
    p.add_argument("--corpus", default="sampler_corpus.json", help="Corpus file made by the corpus pass (default sampler_corpus.json)")
    p.add_argument("--out", default="sampler_slop.txt", help="Slop list text file to write (default sampler_slop.txt)")
    p.add_argument("--top", type=int, default=40, help="Number of phrases to keep (default 40)")
    p.add_argument("--min-prompts", type=int, default=3, help="A phrase must appear under this many different prompts (default 3)")
    p.add_argument("--model", default="", help="Only use the texts from this model (default all)")
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("run", parents=[prompt_args, slop_args], help="Run each sampler config on a running KoboldCpp and compare them")
    p.add_argument("--url", default="http://localhost:5001", help="KoboldCpp address (default http://localhost:5001)")
    p.add_argument("--out", default="sampler_bench.json", help="Results file, new runs are added to it (default sampler_bench.json)")
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds to run for each prompt (default 3)")
    p.add_argument("--max-length", type=int, default=450, help="Max tokens to generate per run (default 450)")
    p.add_argument("--configs", default="", help="Comma separated names of the configs to run (default all)")
    p.add_argument("--add", action="append", default=[], metavar="NAME:KEY=VAL,...", help="Add or replace a config, can be used multiple times. Use antislop=true to ban the slop list.")
    p.add_argument("--config-file", default="", help="Json file of extra configs, as {name: {param: value}}")
    p.add_argument("--label", default="", help="Name to record this model under (default is the loaded model name)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", parents=[slop_args], help="Print the comparison from a results file")
    p.add_argument("--out", default="sampler_bench.json", help="Results file to read (default sampler_bench.json)")
    p.set_defaults(func=cmd_show)

    args = parser.parse_args()
    if args.command == "corpus" and args.num_prompts == 0:
        args.num_prompts = 100
    args.func(args)

if __name__ == "__main__":
    main()
