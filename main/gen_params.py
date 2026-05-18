import json

from . import state
from .utils import tryparseint, tryparsefloat, replace_last_in_string, convert_json_to_gbnf
from .backend import generate, detokenize_ids
from .tool_call import (
    format_jinja, sweep_media_from_messages, determine_tool_json_to_use,
    compress_tools_array, strip_mcpcontent_of_media, extract_json_from_string,
    repack_toolcall_tags
)


def transform_genparams(genparams, api_format, use_jinja):
    if api_format < 0: #not text gen, do nothing
        return

    jsongrammar = r"""
root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws
arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [1-9] [0-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"""

    used_tool_json = None
    #api format 1=basic,2=kai,3=oai,4=oai-chat,5=interrogate,6=ollama,7=ollamachat,8=oai-responses,9=anthropic-messages
    #alias all nonstandard alternative names for rep pen.
    rp1 = float(genparams.get('repeat_penalty', 1.0))
    rp2 = float(genparams.get('repetition_penalty', 1.0))
    rp3 = float(genparams.get('rep_pen', 1.0))
    rp_max = max(rp1,rp2,rp3)
    genparams["rep_pen"] = rp_max
    if "use_default_badwordsids" in genparams and "ban_eos_token" not in genparams:
        genparams["ban_eos_token"] = genparams.get('use_default_badwordsids', False)

    if api_format==1:
        genparams["prompt"] = genparams.get('text', "")
        genparams["top_k"] = int(genparams.get('top_k', 100))
        genparams["max_length"] = int(genparams.get('max', state.args.defaultgenamt))

    elif api_format==2: #note: kobold api does not support tool calling
        adapter_obj = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
        assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)

    elif api_format==3 or api_format==4 or api_format==7:
        default_adapter = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
        adapter_obj = genparams.get('adapter', default_adapter)
        default_max_tok = (adapter_obj.get("max_length", state.args.defaultgenamt) if (api_format==4 or api_format==7) else state.args.defaultgenamt)
        oaiml = tryparseint(genparams.get('max_tokens', genparams.get('max_completion_tokens', default_max_tok)),default_max_tok)
        genparams["max_length"] = genparams.get('max_length', oaiml)
        if genparams["max_length"] <= 0:
            genparams["max_length"] = default_max_tok
        presence_penalty = genparams.get('presence_penalty', genparams.get('frequency_penalty', 0.0))
        genparams["presence_penalty"] = tryparsefloat(presence_penalty,0.0)
        # openai allows either a string or a list as a stop sequence
        if genparams.get('stop',[]) is not None:
            if isinstance(genparams.get('stop',[]), list):
                genparams["stop_sequence"] = genparams.get('stop', [])
            else:
                genparams["stop_sequence"] = [genparams.get('stop')]

        genparams["sampler_seed"] = tryparseint(genparams.get('seed', -1),-1)
        genparams["mirostat"] = genparams.get('mirostat_mode', 0)

        if api_format==4 or api_format==7: #handle ollama chat here too
            # translate openai chat completion messages format into one big string.
            messages_array = genparams.get('messages', [])
            messages_string = adapter_obj.get("chat_start", "")
            system_message_start = adapter_obj.get("system_start", "\n### Instruction:\n")
            system_message_end = adapter_obj.get("system_end", "")
            user_message_start = adapter_obj.get("user_start", "\n### Instruction:\n")
            user_message_end = adapter_obj.get("user_end", "")
            assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
            assistant_message_end = adapter_obj.get("assistant_end", "")
            assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
            tools_message_start = adapter_obj.get("tools_start", "")
            tools_message_end = adapter_obj.get("tools_end", "")
            images_added = []
            audio_added = []
            continue_assistant_turn = genparams.get('continue_assistant_turn', True)
            latest_turn_was_assistant = False
            latest_turn_was_tool = False

            # handle structured outputs
            respformat = genparams.get('response_format', None)
            if respformat:
                try:
                    rt = respformat.get('type')
                    if rt.lower() == "json_schema":
                        schema = respformat.get('json_schema').get('schema')
                        decoded = convert_json_to_gbnf(schema)
                        if decoded:
                            genparams["grammar"] = decoded
                    elif rt.lower() == "json_object":
                        genparams["grammar"] = jsongrammar
                except Exception:
                    # In case of any issues, just do normal gen
                    print("Structured Output not valid - discarded")
                    pass
            elif 'json_schema' in genparams:
                try:
                    schema = genparams.get('json_schema')
                    decoded = convert_json_to_gbnf(schema)
                    if decoded:
                        genparams["grammar"] = decoded
                except Exception:
                    print("Structured Output (old format) not valid - discarded")
                    pass

            message_index = 0
            attachedimgid = 0
            attachedaudid = 0
            jinja_output = None
            jinjatools = genparams.get('tools', [])
            if use_jinja and state.cached_chat_template:
                copied_jinja_kwargs = dict(state.cached_jinja_kwargs or {})
                if "reasoning_effort" in genparams and genparams["reasoning_effort"] is not None:
                    copied_jinja_kwargs["reasoning_effort"] = genparams["reasoning_effort"]
                jinja_output = format_jinja(messages_array,jinjatools,copied_jinja_kwargs)
            if jinja_output:
                messages_string = jinja_output
                for pair in state.thinkformats:
                    starter = pair['start']
                    if jinja_output.rstrip().endswith(starter): #the prompt template already forced a start think.
                        genparams["already_started_thinking"] = True
                        break
                if jinjatools and len(jinjatools)>0:
                    genparams["using_openai_tools"] = True
                # handle media
                images_added, audio_added = sweep_media_from_messages(messages_array)
            else:
                if jinjatools:
                    # inject the tools list at the top of the context window, even if context has shifted
                    # uses koboldcpp's special memory parameter
                    tools_string = f"{system_message_start}### Available Tools:\n{json.dumps(compress_tools_array(jinjatools), indent=0)}{system_message_end}\n"
                    exist_mem = genparams.get('memory', "")
                    genparams["memory"] = tools_string + exist_mem

                for message in messages_array:
                    message_index += 1
                    latest_turn_was_assistant = False
                    latest_turn_was_tool = False
                    if message['role'] == "system":
                        messages_string += system_message_start
                    elif message['role'] == "user":
                        messages_string += user_message_start
                    elif message['role'] == "assistant":
                        messages_string += assistant_message_start
                        latest_turn_was_assistant = True
                    elif message['role'] == "tool":
                        latest_turn_was_tool = True
                        messages_string += tools_message_start
                        tcid = message.get("tool_call_id","")
                        tcid = ("" if not tcid else f" {tcid}")
                        messages_string += f"\nReceived results of function call{tcid}:\n"

                    # content can be a string or an array of objects
                    curr_content = message.get("content",None)
                    if api_format==7: #ollama handle vision
                        imgs = message.get("images",None)
                        if imgs and len(imgs) > 0:
                            for img in imgs:
                                images_added.append(img)
                    if not curr_content:
                        if "tool_calls" in message:
                            try:
                                nlstart = True
                                for tc in message.get("tool_calls"):
                                    if nlstart:
                                        nlstart = False
                                        messages_string += "\n"
                                    tcid = tc.get("id","")
                                    tcfnname = tc.get("function").get("name")
                                    tcfnargs = tc.get("function").get("arguments","")
                                    tcfnargs = (f" with arguments={tcfnargs}" if tcfnargs else "")
                                    messages_string += f"(Made a function call {tcid} to {tcfnname}{tcfnargs})\n"
                            except Exception:
                                messages_string += "\n(Made a function call)\n"
                        pass  # do nothing
                    elif isinstance(curr_content, str):
                        if latest_turn_was_tool and message_index < len(messages_array):
                            curr_content = strip_mcpcontent_of_media(curr_content)
                        messages_string += curr_content
                    elif isinstance(curr_content, list): #is an array
                        for item in curr_content:
                            if isinstance(item, dict):
                                if item['type']=="text":
                                        messages_string += item['text']
                                elif item['type']=="image_url":
                                    if 'image_url' in item and item['image_url'] and item['image_url']['url'] and item['image_url']['url'].startswith("data:image"):
                                        images_added.append(item['image_url']['url'].split(",", 1)[1])
                                        attachedimgid += 1
                                        messages_string += f"\n(Attached Image {attachedimgid})\n"
                                elif item['type']=="input_audio":
                                    if 'input_audio' in item and item['input_audio'] and item['input_audio']['data']:
                                        audio_added.append(item['input_audio']['data'])
                                        attachedaudid += 1
                                        messages_string += f"\n(Attached Audio {attachedaudid})\n"
                            elif isinstance(item, str):
                                messages_string += item # If item is just a string, append it directly

                    # If last message, add any tools calls after message content and before message end token if any
                    if message_index == len(messages_array):
                        is_followup = (message['role'] == "tool")
                        #small hack: if the current turn is assistant, but its short (e.g. a prefilled name), and the previous turn was tool, consider it followup as well
                        if(not is_followup and message_index>1 and message['role'] == "assistant" and messages_array[message_index-2]['role']=="tool" and message['content'] and len(message['content']) < 100): #100 char limit
                            is_followup = True
                        used_tool_json = determine_tool_json_to_use(genparams, messages_string, assistant_message_start, is_followup)

                        if used_tool_json:
                            toolparamjson = None
                            toolname = None
                            # Set temperature lower automatically if function calling, cannot exceed 0.5
                            genparams["temperature"] = (1.0 if genparams.get("temperature", 0.5) > 1.0 else genparams.get("temperature", 0.5))
                            genparams["using_openai_tools"] = True
                            # Set grammar to llamacpp example grammar to force json response (see https://github.com/ggerganov/llama.cpp/blob/master/grammars/json_arr.gbnf)
                            genparams["grammar"] = jsongrammar
                            try:
                                toolname = used_tool_json.get('function').get('name')
                                toolparamjson = used_tool_json.get('function').get('parameters')
                                bettergrammarjson = {"type":"array","items":{"type":"object","properties":{"id":{"type":"string","enum":["call_001"]},"type":{"type":"string","enum":["function"]},"function":{"type":"object","properties":{"name":{"type":"string"},"arguments":{}},"required":["name","arguments"],"additionalProperties":False}},"required":["id","type","function"],"additionalProperties":False}}
                                bettergrammarjson["items"]["properties"]["function"]["properties"]["arguments"] = toolparamjson
                                decoded = convert_json_to_gbnf(bettergrammarjson)
                                if decoded:
                                    genparams["grammar"] = decoded
                            except Exception:
                                pass
                            tool_json_formatting_instruction = f"\nPlease use the provided schema to fill the parameters to create a function call for {toolname}, in the following format: " + json.dumps([{"id": "call_001", "type": "function", "function": {"name": f"{toolname}", "arguments": {"first property key": "first property value", "second property key": "second property value"}}}], indent=0)
                            messages_string += f"\n\nJSON Schema:\n{used_tool_json}\n\n{tool_json_formatting_instruction}{assistant_message_start}"

                    if message['role'] == "system":
                        messages_string += system_message_end
                    elif message['role'] == "user":
                        messages_string += user_message_end
                    elif message['role'] == "assistant":
                        messages_string += assistant_message_end
                    elif message['role'] == "tool":
                        messages_string += tools_message_end
                messages_string += assistant_message_gen
                if (latest_turn_was_assistant and continue_assistant_turn): #allow continue a prefill, chop off end
                    messages_string = messages_string[:-(len(assistant_message_gen)+len(assistant_message_end))]
            genparams["prompt"] = messages_string
            for pair in state.thinkformats:
                starter = pair['start']
                if messages_string.rstrip().endswith(starter): #the prompt template already forced a start think.
                    genparams["already_started_thinking"] = True
                    break
            if len(images_added)>0:
                genparams["images"] = images_added
            if len(audio_added)>0:
                genparams["audio"] = audio_added
            if len(genparams.get('stop_sequence', []))==0: #only set stop seq if it wont overwrite existing
                genparams["stop_sequence"] = [user_message_start.strip(),assistant_message_start.strip()]
            else:
                genparams["stop_sequence"].append(user_message_start.strip())
                genparams["stop_sequence"].append(assistant_message_start.strip())
            if not used_tool_json and jinjatools and latest_turn_was_tool:
                genparams["stop_sequence"].append("(Made a function call") # qol prevent fake toolcalls
            genparams["trim_stop"] = True


    elif api_format==5:
        firstimg = genparams.get('image', "")
        genparams["images"] = [firstimg]
        genparams["max_length"] = 150
        adapter_obj = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
        user_message_start = adapter_obj.get("user_start", "### Instruction:")
        assistant_message_start = adapter_obj.get("assistant_start", "### Response:")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
        genparams["prompt"] = f"{user_message_start} In one sentence, write a descriptive caption for this image.\n{assistant_message_gen}"

    elif api_format==6:
        detokstr = ""
        tokids = genparams.get('context', [])
        adapter_obj = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
        user_message_start = adapter_obj.get("user_start", "\n\n### Instruction:\n")
        assistant_message_start = adapter_obj.get("assistant_start", "\n\n### Response:\n")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
        try:
            detokstr = detokenize_ids(tokids,True)
        except Exception as e:
            from .utils import utfprint
            utfprint("Ollama Context Error: " + str(e))
        ollamasysprompt = genparams.get('system', "")
        ollamabodyprompt = f"{detokstr}{user_message_start}{genparams.get('prompt', '')}{assistant_message_gen}"
        ollamaopts = genparams.get('options', {})
        if genparams.get('stop',[]) is not None:
            genparams["stop_sequence"] = genparams.get('stop', [])
        if "num_predict" in ollamaopts:
            genparams["max_length"] = ollamaopts.get('num_predict', state.args.defaultgenamt)
        if "num_ctx" in ollamaopts:
            genparams["max_context_length"] = ollamaopts.get('num_ctx', state.maxctx)
        if "temperature" in ollamaopts:
            genparams["temperature"] = ollamaopts.get('temperature', 0.75)
        if "top_k" in ollamaopts:
            genparams["top_k"] = ollamaopts.get('top_k', 100)
        if "top_p" in ollamaopts:
            genparams["top_p"] = ollamaopts.get('top_p', 0.92)
        if "seed" in ollamaopts:
            genparams["sampler_seed"] = tryparseint(ollamaopts.get('seed', -1),-1)
        if "stop" in ollamaopts:
            genparams["stop_sequence"] = ollamaopts.get('stop', [])
        genparams["stop_sequence"].append(user_message_start.strip())
        genparams["stop_sequence"].append(assistant_message_start.strip())
        genparams["trim_stop"] = True
        genparams["ollamasysprompt"] = ollamasysprompt
        genparams["ollamabodyprompt"] = ollamabodyprompt
        genparams["prompt"] = ollamasysprompt + ollamabodyprompt
    elif api_format==8: # OpenAI Responses API, oai-responses
        raw_input = genparams.get('input', '')
        raw_instructions = genparams.get('instructions', '')
        if isinstance(raw_input, str):
            genparams['messages'] = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list): # Convert Responses API input items to chat messages format
            converted = []
            for item in raw_input:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    # content can itself be a list of typed parts
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if part.get("type") == "input_text":
                                parts.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "input_image":
                                img = part.get("image_url", part.get("source", {}))
                                parts.append({"type": "image_url", "image_url": {"url": img}})
                        content = parts
                    converted.append({"role": role, "content": content})
                elif isinstance(item, str):
                    converted.append({"role": "user", "content": item})
            genparams['messages'] = converted
        else:
            genparams['messages'] = []
        if raw_instructions and isinstance(raw_instructions, str):
            genparams['messages'].insert(0, {"role": "system", "content": raw_instructions})
        transform_genparams(genparams, 4, use_jinja) # Delegate to the chat-completions transform by re-running as format 4
        return genparams
    elif api_format==9: # Anthropic Messages API
        genparams["max_length"] = genparams.get("max_tokens", state.args.defaultgenamt)
        sys_prompt = genparams.get("system", "")
        messages = genparams.get("messages", [])
        if sys_prompt:
            if isinstance(sys_prompt, list): # Handle array-style system prompts
                sys_prompt = "".join([s.get("text","") for s in sys_prompt if s.get("type") == "text"])
            messages.insert(0, {"role": "system", "content": sys_prompt})
        genparams["messages"] = messages
        transform_genparams(genparams, 4, use_jinja) # Delegate to oai chat completions
        return genparams

    #final transformations (universal template replace)
    replace_instruct_placeholders = genparams.get('replace_instruct_placeholders', True)
    stop_sequence = (genparams.get('stop_sequence', []) if genparams.get('stop_sequence', []) is not None else [])
    stop_sequence = stop_sequence[:state.stop_token_max]
    if replace_instruct_placeholders:
        prompt = genparams.get('prompt', "")
        memory = genparams.get('memory', "")
        adapter_obj = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
        system_message_start = adapter_obj.get("system_start", "\n### Instruction:\n")
        system_message_end = adapter_obj.get("system_end", "")
        user_message_start = adapter_obj.get("user_start", "\n### Instruction:\n")
        user_message_end = adapter_obj.get("user_end", "")
        assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
        assistant_message_end = adapter_obj.get("assistant_end", "")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
        if isinstance(prompt, str): #needed because comfy SD uses same field name
            if assistant_message_gen and assistant_message_gen!=assistant_message_start: #replace final output tag with unspaced (gen) version if exists
                if "{{[OUTPUT]}}" in prompt:
                    prompt = replace_last_in_string(prompt,"{{[OUTPUT]}}",assistant_message_gen)
                elif "{{[OUTPUT]}}" in memory:
                    memory = replace_last_in_string(memory,"{{[OUTPUT]}}",assistant_message_gen)
                elif assistant_message_start and prompt.rstrip().endswith(assistant_message_start):
                    prompt = replace_last_in_string(prompt, assistant_message_start, assistant_message_gen)
            if "{{[INPUT_END]}}" in prompt or "{{[OUTPUT_END]}}" in prompt:
                prompt = prompt.replace("{{[INPUT]}}", user_message_start)
                prompt = prompt.replace("{{[OUTPUT]}}", assistant_message_start)
                prompt = prompt.replace("{{[SYSTEM]}}", system_message_start)
                prompt = prompt.replace("{{[INPUT_END]}}", user_message_end)
                prompt = prompt.replace("{{[OUTPUT_END]}}", assistant_message_end)
                prompt = prompt.replace("{{[SYSTEM_END]}}", system_message_end)
                memory = memory.replace("{{[INPUT]}}", user_message_start)
                memory = memory.replace("{{[OUTPUT]}}", assistant_message_start)
                memory = memory.replace("{{[SYSTEM]}}", system_message_start)
                memory = memory.replace("{{[INPUT_END]}}", user_message_end)
                memory = memory.replace("{{[OUTPUT_END]}}", assistant_message_end)
                memory = memory.replace("{{[SYSTEM_END]}}", system_message_end)
            else:
                if "{{[INPUT]}}" in memory:
                    memory = memory.replace("{{[INPUT]}}", user_message_start, 1)
                else:
                    prompt = prompt.replace("{{[INPUT]}}", user_message_start, 1)
                prompt = prompt.replace("{{[INPUT]}}", assistant_message_end + user_message_start)
                prompt = prompt.replace("{{[OUTPUT]}}", user_message_end + assistant_message_start)
                prompt = prompt.replace("{{[SYSTEM]}}", system_message_start)
                prompt = prompt.replace("{{[INPUT_END]}}", "")
                prompt = prompt.replace("{{[OUTPUT_END]}}", "")
                prompt = prompt.replace("{{[SYSTEM_END]}}", "")
                memory = memory.replace("{{[INPUT]}}", assistant_message_end + user_message_start)
                memory = memory.replace("{{[OUTPUT]}}", user_message_end + assistant_message_start)
                memory = memory.replace("{{[SYSTEM]}}", system_message_start)
                memory = memory.replace("{{[INPUT_END]}}", "")
                memory = memory.replace("{{[OUTPUT_END]}}", "")
                memory = memory.replace("{{[SYSTEM_END]}}", "")
        for i in range(len(stop_sequence)):
            if stop_sequence[i] == "{{[INPUT]}}":
                stop_sequence[i] = user_message_start.strip()
            elif stop_sequence[i] == "{{[OUTPUT]}}":
                stop_sequence[i] = assistant_message_start.strip()
            elif stop_sequence[i] == "{{[INPUT_END]}}":
                stop_sequence[i] = (user_message_end.strip() if user_message_end.strip()!="" else "")
            elif stop_sequence[i] == "{{[OUTPUT_END]}}":
                stop_sequence[i] = (assistant_message_end.strip() if assistant_message_end.strip()!="" else "")
        stop_sequence = list(filter(None, stop_sequence))
        genparams["prompt"] = prompt
        genparams["memory"] = memory
    genparams["stop_sequence"] = stop_sequence
    return genparams
