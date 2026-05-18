import re
import json
import ctypes
from datetime import datetime

from . import state
from .utils import convert_json_to_gbnf, fix_unquoted_keys


def coerce_tool_argtypes(tool_calls: list, tool_list: list) -> list:
    if not tool_calls or not tool_list:
        return tool_calls

    schema_map = {}
    for tool in tool_list:
        try:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                props = func.get("parameters", {}).get("properties", {})
            else:
                name = tool.get("name", "")
                props = tool.get("parameters", {}).get("properties", {})
            if name:
                schema_map[name] = props
        except Exception:
            continue

    def coerce_value(val, prop_type):
        if val is None:
            return val
        try:
            if prop_type == "integer":
                return val if isinstance(val, int) else int(val)
            elif prop_type == "number":
                return val if isinstance(val, (int, float)) else float(val)
            elif prop_type == "boolean":
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    if val.lower() in ("true", "1", "yes"):
                        return True
                    if val.lower() in ("false", "0", "no"):
                        return False
                if isinstance(val, int):
                    return bool(val)
                return val
            elif prop_type == "string":
                return val if isinstance(val, str) else str(val) if val is not None else val
            elif prop_type == "array":
                if isinstance(val, list):
                    return val
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except Exception:
                        return [val]
                if isinstance(val, (set, tuple)):
                    return list(val)
                return [val]
            elif prop_type == "object":
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        return parsed if isinstance(parsed, dict) else val
                    except Exception:
                        return val
                return val
            elif prop_type == "null":
                return None
        except (ValueError, TypeError, AttributeError):
            pass
        return val

    result = []
    for call in tool_calls:
        try:
            if "function" in call:
                name = call["function"].get("name", "")
                arguments = call["function"].get("arguments", {})
            else:
                name = call.get("name", "")
                arguments = call.get("arguments", {})

            props = schema_map.get(name, {})
            if props and isinstance(arguments, dict):
                coerced = {}
                for key, val in arguments.items():
                    prop_schema = props.get(key, {})
                    prop_type = prop_schema.get("type")
                    # handle anyOf/oneOf for nullable types like {"anyOf": [{"type": "string"}, {"type": "null"}]}
                    if prop_type is None:
                        for combiner in ("anyOf", "oneOf"):
                            options = prop_schema.get(combiner, [])
                            for option in options:
                                t = option.get("type")
                                if t and t != "null":
                                    prop_type = t
                                    break
                            if prop_type: # Found a type, stop looking in other combiners
                                break
                    try:
                        coerced[key] = coerce_value(val, prop_type)
                    except Exception:
                        coerced[key] = val
                if "function" in call:
                    call = {**call, "function": {**call["function"], "arguments": coerced}}
                else:
                    call = {**call, "arguments": coerced}
        except Exception:
            pass
        result.append(call)

    return result

def toolcall_to_normalized_json(text,start_tag,end_tag): #convert weird formats into standard tool call json
    text = text.strip()
    def parse_qwen35(text: str) -> str:
        fn_match = re.search(r"<function=(.*?)>", text)
        if not fn_match:
            return text
        fn_name = fn_match.group(1).strip()
        params = {}
        param_blocks = re.findall(r"<parameter=(.*?)>(.*?)</parameter>", text, re.DOTALL)
        for key, value in param_blocks:
            params[key.strip()] = value.strip()
        return json.dumps({"name": fn_name, "arguments": params})
    def parse_glm(text: str) -> str:
        text = text.strip()
        # Extract function name: it's the first thing before any <arg_key>
        fn_match = re.match(r"^\s*([^\<\s]+)", text)
        if not fn_match:
            return text
        fn_name = fn_match.group(1).strip()
        # Extract all key/value pairs
        keys = re.findall(r"<arg_key>(.*?)</arg_key>", text)
        values = re.findall(r"<arg_value>(.*?)</arg_value>", text)
        params = {}
        for i in range(min(len(keys), len(values))):
            params[keys[i].strip()] = values[i].strip()
        return json.dumps({"name": fn_name, "arguments": params})
    def parse_deepseek_r1_sep(text: str) -> str:
        text = re.sub(r'<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>', r'\1',
                    text, flags=re.DOTALL).strip()
        sep = '<｜tool▁sep｜>'
        if sep not in text:
            return text
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        results = []
        for part in parts:
            lines = part.split('\n', 1)
            fn_name = lines[0].strip()
            args_block = lines[1] if len(lines) > 1 else '{}'
            args_block = re.sub(r'^```(?:json)?\s*', '', args_block.strip())
            args_block = re.sub(r'\s*```$', '', args_block.strip())
            try:
                results.append({"name": fn_name, "arguments": json.loads(args_block)})
            except Exception:
                pass
        if not results:
            return text
        return json.dumps(results) if len(results) > 1 else json.dumps(results[0])
    def parse_minimax(text: str) -> str:
        results = []
        for invoke in re.finditer(
            r'<invoke\s+name=["\']?([^"\'>\s]+)["\']?>(.*?)</invoke>',
            text, re.DOTALL
        ):
            fn_name = invoke.group(1).strip()
            params = {}
            for p in re.finditer(
                r'<parameter\s+name=["\']?([^"\'>\s]+)["\']?>(.*?)</parameter>',
                invoke.group(2), re.DOTALL
            ):
                val = p.group(2).strip()
                try:
                    params[p.group(1).strip()] = json.loads(val)
                except Exception:
                    params[p.group(1).strip()] = val
            results.append({"name": fn_name, "arguments": params})
        if not results:
            return text
        return json.dumps(results) if len(results) > 1 else json.dumps(results[0])
    def parse_gemma4(text: str) -> str:
        if text.startswith("call:"):
            text = text[len("call:"):]
        if '<|"|>' in text:
            text = text.replace('<|"|>', '!$$REAL_QUOTE$$!')
            text = text.replace('\\', '\\\\')
            text = text.replace('"', '\\"')
            text = text.replace('!$$REAL_QUOTE$$!','"')
        fn_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\{(.*)\}$', text.strip(), re.DOTALL) # extract fn name
        if not fn_match:
            return text
        fn_name = fn_match.group(1)
        body = fn_match.group(2).strip()
        body = '{' + body + '}'
        if not body:
            return json.dumps({"name": fn_name, "arguments": {}})
        try:   # Try to parse body as JSON object by wrapping it
            args = json.loads(body,strict=False)
            return json.dumps({"name": fn_name, "arguments": args})
        except Exception:
            pass
        normalized = fix_unquoted_keys(body)
        try:
            args = json.loads(normalized,strict=False)
            return json.dumps({"name": fn_name, "arguments": args})
        except Exception:
            pass
        return text

    def parse_gpt_oss(text: str) -> str:
        fn_match = re.search(r'functions\.([a-zA-Z_][a-zA-Z0-9_]*)', text)
        if not fn_match:
            return text
        fn_name = fn_match.group(1).strip()
        msg_split = text.split('<|message|>', 1)
        if len(msg_split) < 2:
            return text
        args_block = msg_split[1].strip()
        try:
            args = json.loads(args_block)
        except Exception:
            return text
        return json.dumps({"name": fn_name, "arguments": args})

    # gemma4 takes precedence, since it can contain valid json fragments
    if end_tag=="<tool_call|>":
        return parse_gemma4(text)

    #if we are already valid JSON, return
    check_ok = extract_json_from_string(text, True)
    if check_ok and len(check_ok)>0:
        return text #is valid JSON or parsable

    if "<arg_key>" in text and "<arg_value>" in text: # handle glm with args
        return parse_glm(text)

    if "<function=" in text: # handle qwen3.5
        return parse_qwen35(text)

    if "<invoke " in text: #minimax
        return parse_minimax(text)

    if '<｜tool▁sep｜>' in text: #deepseek
        return parse_deepseek_r1_sep(text)

    if ' ' not in text and '\n' not in text: # handle glm without args
        return parse_glm(text)

    if 'functions.' in text and "commentary" in start_tag:  # handle GPT-OSS
        return parse_gpt_oss(text)

    return text #fallback

def repack_toolcall_tags(text: str, original_tools:list):
    tool_calls = []
    if not text:
        return tool_calls
    for fmt in state.thinkformats:
        pattern = f"{re.escape(fmt['start'])}.*?{re.escape(fmt['end'])}"
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    text = text.strip()
    found = False
    for start, end, streamhandled in state.tool_call_pairs:
        pattern=""
        if end:
            pattern = re.escape(start) + r"(.*?)" + re.escape(end)
        else:
            pattern = re.escape(start) + r"(.*)$"  # match to end of string
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            found = True
            for match in matches:
                normalizedtc = toolcall_to_normalized_json(match.strip(),start,end)
                sub_tool_calls = extract_json_from_string(normalizedtc)
                tool_calls.extend(sub_tool_calls)
            break
    # fallback ONLY if no tags were found at all
    if not found:
        tool_calls = extract_json_from_string(text,True)
    tool_calls = coerce_tool_argtypes(tool_calls, original_tools)
    return tool_calls

def format_jinja(messages_orig, tools, chat_template_kwargs=None):
    try:
        def strftime_now(format='%Y-%m-%d %H:%M:%S'):
            return datetime.now().strftime(format)
        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        def raise_exception(msg):
            print(f"Warning: Jinja template raised an exception: {msg}")
            return ""
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        # sanitize messages to remove none types
        messages = json.loads(json.dumps(messages_orig))
        for m in messages:
            if m.get("content") is None:
                m["content"] = ""
        # fix image placeholders, erase them and slap a reference onto the turn text message
        mediacount = 1
        for m in messages:
            if isinstance(m.get("content"), list):
                normalized = []
                turn_text = ""
                media_text = ""
                for item in m["content"]:
                    if item.get("type")=="text":
                        turn_text += item.get("text","")
                for item in m["content"]:
                    if item.get("type")=="text":
                        pass
                    elif item.get("type")=="image_url" or item.get("type")=="image":
                        media_text += f"\n(Attached Image {mediacount})\n"
                        mediacount += 1
                    elif item.get("type")=="input_audio":
                        media_text += f"\n(Attached Audio {mediacount})\n"
                        mediacount += 1
                    else:
                        normalized.append(item)
                turn_text = media_text + turn_text
                if turn_text:
                    normalized.append({"type": "text","text": turn_text})
                m["content"] = normalized
        for m in messages: # Fix tool_calls arguments and content if parsable
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except Exception:
                            pass
        jinja_env.globals['strftime_now'] = strftime_now
        jinja_env.globals['raise_exception'] = raise_exception
        jinja_env.filters["tojson"] = tojson
        jinja_compiled_template = jinja_env.from_string(state.cached_chat_template)
        text = None
        messages_for_render = []
        assist_should_prefill = False
        chat_template_kwargs = chat_template_kwargs or {}
        last_assist_msg = ""
        if messages:
            last_assist_msg = messages[-1]["content"]
            assist_should_prefill = (messages and messages[-1]["role"].lower() == "assistant" and last_assist_msg and isinstance(last_assist_msg, str) and len(last_assist_msg.strip())>0) #avoid single character newline or space content
            last_assist_msg = "" if not assist_should_prefill else last_assist_msg
            messages_for_render = messages[:-1] if len(messages) > 1 and assist_should_prefill else messages
        if tools and len(tools)>0:
            text = jinja_compiled_template.render(messages=messages_for_render, tools=tools, add_generation_prompt=True, bos_token="", eos_token="", **chat_template_kwargs)
        else:
            text = jinja_compiled_template.render(messages=messages_for_render, add_generation_prompt=True, bos_token="", eos_token="", **chat_template_kwargs)
        if assist_should_prefill and text and last_assist_msg: # handle prefill continuations
            text = text + last_assist_msg
        return text if text else None
    except Exception as e:
        print(f"Jinja formatting failed: {e}")
        return None

def remove_outer_tags(inputstr):
    try:
        stripped = inputstr.strip()
        match = re.match(r'^<([^\s<>]+)>(.*?)</\1>\s*$', stripped, re.DOTALL) # Try angle brackets first
        if match:
            return match.group(2).strip()
        match = re.match(r'^\[([^\s<>]+)\](.*?)\[/\1]\s*$', stripped, re.DOTALL) # Then try square brackets
        if match:
            return match.group(2).strip()
        return stripped # If no match, return original string
    except Exception:
        return stripped

def normalize_tool_call(obj): # Normalize various tool call formats to OpenAI format
    if "type" in obj and "function" in obj: # Already in OpenAI format
        return obj
    if "name" in obj and ("arguments" in obj or "parameters" in obj):
        args = obj.get("arguments", obj.get("parameters", {}))
        return {
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": args
            }
        }
    if "function" in obj and isinstance(obj["function"], dict):
        func = obj["function"]
        if "name" in func:
            return {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "arguments": func.get("arguments", func.get("parameters", {}))
                }
            }

    return obj

# Used to parse json for openai tool calls
def extract_json_from_string(input_string, check_strict=False):
    parsed_json = None
    input_string = remove_outer_tags(input_string) #if we detected wrapper tags, remove them

    try: # First check if model exported perfect json
        parsed_json = json.loads(input_string)
        if not isinstance(parsed_json, list):
            parsed_json = [parsed_json]
        return parsed_json
    except Exception:
        pass
    try: # Next check if all we need is to add brackets to make it perfect json
        parsed_json = json.loads(f"[{input_string}]")
        return parsed_json
    except Exception:
        pass
    try:
        if not check_strict: #only allow when not strict mode
            # Now use regular expression to match JSON objects or arrays in case part is valid json and part is not
            json_pattern = r'(\{.*?\}|\[.*?\])'  # was json_pattern = r'(\{.*\}|\[.*\])'
            potential_jsons = re.findall(json_pattern, input_string, re.DOTALL)
            for potential_json in potential_jsons:
                try:
                    parsed_json = json.loads(potential_json, strict=False)
                    if not isinstance(parsed_json, list):
                        parsed_json = [parsed_json]
                    return parsed_json
                except Exception:
                    continue
    except Exception:
        pass
    return []

def parse_last_logprobs(lastlogprobs):
    if not lastlogprobs:
        return None
    logprobsdict = {}
    logprobsdict['content'] = []
    logprobsdict['tokens'] = []
    logprobsdict['token_ids'] = []
    logprobsdict['token_logprobs'] = []
    logprobsdict['top_logprobs'] = []
    logprobsdict['text_offset'] = []
    text_offset_counter = 0
    for i in range(lastlogprobs.count):
        lp_content_item = {}
        logprob_item = lastlogprobs.logprob_items[i]
        toptoken = ctypes.string_at(logprob_item.selected_token).decode("UTF-8","ignore")
        logprobsdict['tokens'].append(toptoken)
        logprobsdict['token_ids'].append(logprob_item.selected_token_id)
        lp_content_item['token'] = toptoken
        lp_content_item['token_id'] = logprob_item.selected_token_id
        logprobsdict['token_logprobs'].append(logprob_item.selected_logprob)
        lp_content_item['logprob'] = logprob_item.selected_logprob
        lp_content_item['bytes'] = list(toptoken.encode('utf-8'))
        lp_content_item['top_logprobs'] = []
        logprobsdict['text_offset'].append(text_offset_counter)
        text_offset_counter += len(toptoken)
        tops = {}
        for j in range(min(logprob_item.option_count,state.logprobs_max)):
            tl_item = {}
            tl_item['logprob'] = logprob_item.logprobs[j]
            tokstr = ctypes.string_at(logprob_item.tokens[j]).decode("UTF-8","ignore")
            tops[tokstr] = logprob_item.logprobs[j]
            tl_item['token'] = tokstr
            tl_item['token_id'] = logprob_item.token_ids[j]
            tl_item['bytes'] = list(tokstr.encode('utf-8'))
            lp_content_item['top_logprobs'].append(tl_item)
        logprobsdict['top_logprobs'].append(tops)
        logprobsdict['content'].append(lp_content_item)
    return logprobsdict

def extract_tool_info_from_tool_array(chosen_tool, tools_array):
    found_function = ""
    found_tooljson = None
    try:
        if isinstance(chosen_tool, str):
            found_function = chosen_tool
        elif isinstance(chosen_tool, dict): #if we can match the tool name, we must use that tool, remove all other tools
            found_function = chosen_tool.get('function').get('name')
        #if we find the function in tools, remove all other tools except the one matching the function name
        for tool in tools_array:
            if found_function and tool.get('type') == "function" and tool.get('function').get('name').lower() == found_function.lower():
                found_tooljson = tool
                break
    except Exception:
        # In case of any issues, just revert back to no specified function
        print("Tools parsing not valid - discarded")
        pass
    return found_tooljson

def extract_all_names_from_tool_array(tools_array):
    toolnames = []
    for tool in tools_array:
        try:
            if tool.get('type') == "function" and tool.get('function').get('name'):
                toolnames.append(tool.get('function').get('name'))
        except Exception:
            pass
    return toolnames

def strip_oaicontent_of_media(oaicontent):
    if isinstance(oaicontent, list):
        outarr = []
        for x in oaicontent:
            if not isinstance(x, dict):
                outarr.append({"type": "unknown", "data": "(base64 data attached)"})
                continue
            xtype = x.get("type","data")
            if xtype=="text":
                outarr.append(x)
            else:
                outarr.append({"type":xtype, "data":"(base64 data attached)"})
        return outarr
    return oaicontent

def strip_mcpcontent_of_media(mcpcontentstr):
    try:
        if isinstance(mcpcontentstr, str):
            #we try to strip out the b64 of MCP type tool responses with images for past turns
            mcp_pl = json.loads(mcpcontentstr)
            pl_modified = False
            if isinstance(mcp_pl, dict) and isinstance(mcp_pl.get("content",None),list):
                pl_arr = mcp_pl.get("content",[])
                for idx in range(len(pl_arr)):
                    if pl_arr[idx].get("type","")=="image" and pl_arr[idx].get("data","")!="":
                        pl_arr[idx]["data"] = "(base64 data attached)"
                        pl_modified = True
                if pl_modified:
                    mcpcontentstr = json.dumps(mcp_pl)
    except Exception:
        pass
    return mcpcontentstr

#returns the found JSON of the correct tool to use, or None if no tool is suitable
def determine_tool_json_to_use(genparams, curr_ctx, assistant_message_start, is_followup_tool):
    from .backend import generate
    # tools handling: Check if user is passing a openai tools array, if so add to end of prompt before assistant prompt unless tool_choice has been set to None
    tools_array = genparams.get('tools', [])
    chosen_tool = genparams.get('tool_choice', "auto")
    messages = genparams.get('messages',[])
    toolmem = genparams.get("memory","")

    # first handle auto mode, determine whether a tool is needed
    used_tool_json = None
    if not curr_ctx:
        return None

    # get user's last message and last tool call results
    last_user_message = ""
    tool_call_results = ""

    images_added = [] #sometimes images are needed to make a decision too
    audio_added = []

    if messages:
        images_added, audio_added = sweep_media_from_messages(messages)
        reversed_messages = list(reversed(messages))
        for message in reversed_messages:
            if message["role"] == "user":
                last_user_message = message["content"]
                last_user_message = strip_oaicontent_of_media(last_user_message)
                last_user_message = f"\n\nUser's current request: {last_user_message}"
                break
        tool_call_chunk = []
        for message in reversed_messages:
            if message["role"] == "tool":
                toolrespstr = message["content"]
                # toolrespstr = strip_mcpcontent_of_media(toolrespstr)
                tool_call_chunk.append(toolrespstr)
            else:
                break
        tmp_tool_replies = list(reversed(tool_call_chunk))
        if tmp_tool_replies and len(tmp_tool_replies)>0:
            tool_call_results = f"\n\nTool call responses: {tmp_tool_replies}"

    if tools_array and len(tools_array) > 0 and chosen_tool is not None and chosen_tool!="none":
        should_use_tools = True
        if chosen_tool=="auto" or chosen_tool=="required":
            # note: message string already contains the instruct start tag!
            temptoolnames = extract_all_names_from_tool_array(tools_array)
            tempjson = {}
            if chosen_tool=="required":
                custom_tools_prompt_json_format = "Respond with a JSON object using this structure:\r\n{\r\n    \"tool_name\": \"exact_tool_name_here\"\r\n}\r\n\r\nRules:\r\n- You must pick one of the tools to use, pick the most suitable tool."
                tempjson = {"type":"object","properties":{"tool_name":{"type":"string","enum":temptoolnames}},"required":["tool_name"],"additionalProperties":False}
            else:
                temptoolnames.append("null")
                custom_tools_prompt_json_format = "Respond with a JSON object using this structure:\r\n{\r\n    \"reasoning\": \"Your reasoning here\",\r\n    \"final_decision\": \"yes\" or \"no\",\r\n    \"tool_name\": \"exact_tool_name_here\" or \"null\"\r\n}\r\n\r\nRules:\r\n- Output only the JSON object. Do NOT add anything before or after the json object.\r\n- final_decision must be exactly \"yes\" or \"no\"\r\n- tool_name must be either an exact tool name, or if no tool is required, an empty string: \"\"\r\n- Keep reasoning short, maximum one or two sentences.\r\n- No unnecessary comments"
                tempjson = {"type":"object","properties":{"reasoning":{"type":"string"},"final_decision":{"type":"string","enum":["yes","no","Yes","No","YES","NO"," yes"," no"," Yes"," No"," YES"," NO"]},"tool_name":{"type":"string","enum":temptoolnames}},"required":["reasoning","final_decision","tool_name"],"additionalProperties":False}
            toolquerygrammar = convert_json_to_gbnf(tempjson)

            if not is_followup_tool:
                custom_tools_prompt = "Is calling one of the tools listed above absolutely essential to answer user's current request, or is a tool call optional?"
                custom_tools_prompt_processed = f"{curr_ctx}{last_user_message}\n\n{custom_tools_prompt} {custom_tools_prompt_json_format}{assistant_message_start}"
            else:
                custom_tools_prompt = "Given the tool call response to the user's current request, is another tool call needed to further answer user's message?"
                custom_tools_prompt_processed = f"{curr_ctx}{last_user_message}{tool_call_results}\n\n{custom_tools_prompt} {custom_tools_prompt_json_format}{assistant_message_start}"

            # first, prompt to see if a tool call is needed using the prompt above.
            # the result is a short explanation by the LLM on why a tool call is or is not needed, along with it's final decision at the end.
            temp_poll = {
                "prompt": custom_tools_prompt_processed,
                "memory": toolmem,
                "max_length":300,
                "temperature":0.1,
                "top_k":1,
                "rep_pen":1,
                "ban_eos_token":False,
                "grammar":toolquerygrammar
            }
            if len(images_added)>0:
                temp_poll["images"] = images_added
            if len(audio_added)>0:
                temp_poll["audio"] = audio_added
            temp_poll_result = generate(genparams=temp_poll)
            temp_poll_text = temp_poll_result['text'].strip().rstrip('.')
            temp_poll_data_arr = extract_json_from_string(temp_poll_text)
            temp_poll_data = temp_poll_data_arr[0] if (temp_poll_data_arr and len(temp_poll_data_arr)>0) else None

            if temp_poll_data:
                if chosen_tool!="required" and ("yes" not in temp_poll_data.get("final_decision","").lower() or "null" in temp_poll_data.get("tool_name","").lower()):
                    should_use_tools = False
                elif (chosen_tool=="auto" or chosen_tool=="required") and "null" not in temp_poll_data.get("tool_name","").lower():
                    chosen_tool = temp_poll_data.get("tool_name","").lower().strip()

            if not state.args.quiet:
                print(f"\n[TOOLCALL REASONING]: {temp_poll_text}")

        if should_use_tools:
            #first, try and extract a specific tool if selected
            used_tool_json = extract_tool_info_from_tool_array(chosen_tool, tools_array)
            if used_tool_json: #already found the tool we want, remove all others
                pass
            elif len(tools_array)==1:
                used_tool_json = tools_array[0]
            else: # we have to find the tool we want the old fashioned way
                toolnames = extract_all_names_from_tool_array(tools_array)
                if len(toolnames) == 1:
                    used_tool_json = extract_tool_info_from_tool_array(toolnames[0], tools_array)

    return used_tool_json

def compress_tools_array(tools_array):
    tools_array_filtered = []
    for tool_dict in tools_array:
        tool_data = tool_dict
        if 'function' in tool_dict:
            tool_data = tool_dict['function']
        tool_props = {}
        params = tool_data.get("parameters", {})
        props = params.get("properties", {})
        for prop_name, prop_data in props.items():
            prop_type = prop_data.get("type")
            if prop_type is None and "anyOf" in prop_data:
                for option in prop_data["anyOf"]:
                    option_type = option.get("type")
                    if option_type and option_type != "null":
                        prop_type = option_type
                        break
            if prop_type is None:
                prop_type = "string"
            tool_props[prop_name] = prop_type
        tools_array_filtered.append({
            "name": tool_data['name'],
            "description": tool_data.get("description", ""),
            "properties": tool_props
        })

    return tools_array_filtered

def sweep_media_from_messages(messages_array):
    images = []
    audio = []
    for message in messages_array:
        curr_content = message.get("content", None)
        if isinstance(curr_content, list):
            for item in curr_content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        images.append(url.split(",", 1)[1])
                elif item.get("type") == "input_audio":
                    data = item.get("input_audio", {}).get("data")
                    if data:
                        audio.append(data)
        elif message.get("role", "")=="tool" and isinstance(curr_content, str): #handle mcp returned images
            try:
                mcp_pl = json.loads(curr_content)
                if isinstance(mcp_pl, dict) and isinstance(mcp_pl.get("content",None),list):
                    pl_arr = mcp_pl.get("content",[])
                    if len(pl_arr)>0 and pl_arr[0].get("type","")=="image" and pl_arr[0].get("data","")!="":
                        images.append(pl_arr[0].get("data",""))
            except Exception:
                pass
        imgs_ollama = message.get("images", None)
        if imgs_ollama:
            for img in imgs_ollama:
                images.append(img)
    return images, audio
