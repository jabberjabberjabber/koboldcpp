import os
import sys
import ctypes
import platform
import ssl
import http.server
import asyncio
import socket
import threading
import html
import re
import copy
import hashlib
import urllib.parse
import urllib.request
import time
import json
import gzip
import struct
import base64
import random
import queue
import math
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from . import state
from .utils import (
    utfprint, truncate_long_json, LaunchWebbrowser, get_my_epurl,
    is_port_in_use, is_ipv6_supported, simple_lcg_hash, strip_base64_prefix,
    get_capabilities, scan_directory, get_current_admindir_list,
    string_contains_or_overlaps_sequence_substring, is_incomplete_utf8_sequence,
    bring_terminal_to_foreground, convert_json_to_gbnf, end_trim_to_sentence,
    make_url_request, tryparseint,
)
from .gen_params import transform_genparams
from .tool_call import (
    toolcall_to_normalized_json, repack_toolcall_tags, extract_json_from_string,
    normalize_tool_call, extract_tool_info_from_tool_array,
    extract_all_names_from_tool_array, coerce_tool_argtypes,
    compress_tools_array, sweep_media_from_messages, parse_last_logprobs,
    strip_mcpcontent_of_media, strip_oaicontent_of_media,
    determine_tool_json_to_use, format_jinja, remove_outer_tags,
)
from .backend import generate, continuous_batching_python_eligible, tokenize_ids, detokenize_ids
from .sd import (
    sd_generate, sd_get_info, sd_sdapi_samplers, sd_convdirect_option, sd_quant_option,
    sd_upscale, sd_oai_transform_params, sd_comfyui_tranform_params,
    sanitize_lora_list, sanitize_lora_multipliers, prepare_lora_multipliers,
    prepare_initial_lora_multipliers, prepare_lora_multipliers_backend,
    mk_sdapi_lora_list, extract_loras_from_prompt, lora_map_name_to_path,
    sd_sampler_canonical_name, parse_json_object, gendefaults_parse_meta_field,
)
from .whisper import whisper_generate
from .tts import tts_generate, tts_extract_instruction
from .embeddings import embeddings_generate
from .music import music_generate_codes, music_generate_audio
from .websearch import websearch
from .mcp import MCPHTTPClient, MCPStdioClient, load_mcp_async
from .model_meta import exit_with_error, has_valid_model
from .horde import run_horde_worker
from .config import reload_new_config, reload_from_new_args

class KcppProxyHandler(http.server.BaseHTTPRequestHandler):
    sys_version = "1"
    server_version = "KoboldCppServer"
    protocol_version = "HTTP/1.1"
    HOP_BY_HOP = { "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade" }
    STREAM_CHUNK = 512

    def log_message(self, fmt, *args):
        if state.showdebug:
            print(f"[proxy] {self.address_string()} {fmt % state.args}", flush=True)
        pass

    def wait_for_upstream_ready(self, port, timeout, interval):
        start = time.time()
        while time.time() - start < timeout:
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=5)
                conn.request("GET", "/api/v1/info/version")
                resp = conn.getresponse()
                if resp.status == 200:
                    data = resp.read()
                    try:
                        json.loads(data.decode("utf-8"))
                        return True
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            time.sleep(interval)
        return False  # timeout

    def _handle(self):
        upstream_port = self.server.upstream_port
        length = self.headers.get("Content-Length") #  read request body
        body = None
        if length:
            body = self.rfile.read(int(length))
        headers = {} # forward headers
        for k, v in self.headers.items():
            if k.lower() not in self.HOP_BY_HOP:
                headers[k] = v
        headers["Connection"] = "close"

        #specifically look for generation requests from completions or chat completions to handle hotswap
        is_post = self.command.upper() == "POST"
        is_completions_path = (self.path.endswith('/v1/completions') or self.path.endswith('/v1/completion') or self.path=='/completions')
        is_chat_completions_path = (self.path.endswith('/v1/chat/completions') or self.path=='/chat/completions')

        #any requests to the following endpoints is capable of waking the server
        wake_requests = ["/api/extra/generate/stream","/api/extra/tokencount","/api/v1/generate","/sdapi/v1/interrogate","/v1/completions","/v1/chat/completions","/v1/responses","/completions","/chat/completions","/responses","/api/extra/transcribe","/v1/audio/transcriptions","/api/extra/tts","/v1/audio/speech","/api/extra/embeddings","/v1/embeddings","/api/extra/music/prepare","/api/extra/music/generate","/sdapi/v1/txt2img","/sdapi/v1/img2img","/sdapi/v1/upscale"]
        is_wake_request = self.path in wake_requests

        autoswapEnabled = state.global_memory["autoswapmode"] is not None and state.global_memory["autoswapmode"]
        model_switch_pass = False

        with proxy_reload_lock:
            if is_post and (is_completions_path or is_chat_completions_path or (not autoswapEnabled and is_wake_request)):
                model_name = ""
                if body:
                    try:
                        request_json = json.loads(body.decode("utf-8"))
                        model_name = request_json.get("model")
                    except Exception:
                        pass

                was_auto_unloaded = (state.global_memory["triggered_sleeping"] and state.global_memory["current_model"]=="unload_model")

                is_different_model = False
                # we only need to check if the currently loaded model is different from the requested model. everything else is handled downstream in the stack
                if model_name and model_name != state.global_memory["current_model"]:
                    is_different_model = True

                if is_different_model or was_auto_unloaded:
                    model_switch_pass = True
                    whitelist = get_current_admindir_list() # see if its an allowed swap
                    if was_auto_unloaded and not model_name:
                        model_name = "initial_model"
                    if is_different_model and (model_name in whitelist):
                        state.global_memory["last_active_timestamp"] = datetime.now()
                        state.global_memory["triggered_sleeping"] = False
                        reqbody = json.dumps({"filename":model_name})
                        reqheaders = {
                            'Content-Type': 'application/json',
                            'Content-Length': str(len(reqbody)),
                        }
                        if state.args.adminpassword:
                            reqheaders["Authorization"] = f"Bearer {state.args.adminpassword}"
                        conn = http.client.HTTPConnection('localhost', upstream_port, timeout=state.args.reqtimeout)
                        conn.request("POST", "/api/admin/reload_config", body=reqbody, headers=reqheaders)
                        resp = conn.getresponse()
                        time.sleep(3)
                        state.global_memory["last_active_timestamp"] = datetime.now()
                        state.global_memory["triggered_sleeping"] = False
                        if not self.wait_for_upstream_ready(upstream_port,120,0.5):
                            self.send_error(504, "KoboldCpp model swap reload timed out")
                            return
                        time.sleep(0.1)
            if autoswapEnabled and not model_switch_pass:
                textReqs = ["/api/extra/generate/stream","/api/extra/tokencount","/api/v1/generate","/sdapi/v1/interrogate","/v1/completions","/v1/chat/completions","/v1/responses","/completions","/chat/completions","/responses"]
                sttReqs = ["/api/extra/transcribe","/v1/audio/transcriptions"]
                ttsReqs = ["/api/extra/tts", "/v1/audio/speech"]
                embedReqs = ["/api/extra/embeddings", "/v1/embeddings"]
                musicReqs = ["/api/extra/music/prepare","/api/extra/music/generate"]
                imageReqs = ["/sdapi/v1/txt2img", "/sdapi/v1/img2img", "/sdapi/v1/upscale"] # "/sdapi/v1/sd-models", "/sdapi/v1/options", "/sdapi/v1/samplers"

                swapModeChanged = False
                if any(self.path.endswith(e) for e in textReqs) and (state.global_memory["swapReqType"] is None or state.global_memory["swapReqType"] != "text"):
                    state.global_memory["swapReqType"] = "text"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in sttReqs) and (state.global_memory["swapReqType"] is None or state.global_memory["swapReqType"] != "stt"):
                    state.global_memory["swapReqType"] = "stt"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in ttsReqs) and (state.global_memory["swapReqType"] is None or state.global_memory["swapReqType"] != "tts"):
                    state.global_memory["swapReqType"] = "tts"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in embedReqs) and (state.global_memory["swapReqType"] is None or state.global_memory["swapReqType"] != "embed"):
                    state.global_memory["swapReqType"] = "embed"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in musicReqs) and (state.global_memory["swapReqType"] is None or state.global_memory["swapReqType"] != "music"):
                    state.global_memory["swapReqType"] = "music"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in imageReqs) and (state.global_memory["swapReqType"] is None or state.global_memory["swapReqType"] != "image"):
                    state.global_memory["swapReqType"] = "image"
                    swapModeChanged = True

                if (state.global_memory["swapReqType"] is not None and swapModeChanged):
                    reqbody = json.dumps({"filename":state.global_memory["current_model"], "baseconfig": state.global_memory["base_config"]})
                    reqheaders = {
                        'Content-Type': 'application/json',
                        'Content-Length': str(len(reqbody)),
                    }
                    if state.args.adminpassword:
                        reqheaders["Authorization"] = f"Bearer {state.args.adminpassword}"
                    conn = http.client.HTTPConnection('localhost', upstream_port, timeout=state.args.reqtimeout)
                    conn.request("POST", "/api/admin/reload_config", body=reqbody, headers=reqheaders)
                    resp = conn.getresponse()
                    time.sleep(3)
                    state.global_memory["last_active_timestamp"] = datetime.now()
                    if not self.wait_for_upstream_ready(upstream_port,120,0.5):
                        self.send_error(504, "KoboldCpp model swap reload timed out")
                        return
                    time.sleep(0.1)

        try:  # connect upstream
            conn = http.client.HTTPConnection('localhost', upstream_port, timeout=state.args.reqtimeout)
            conn.request( self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            if state.args.debugmode:
                print(f"OSError has occurred: {e}")
            html_502 = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>502 - KoboldCpp</title>
                <style>
                *,dialog{padding:0;margin:0}*{box-sizing:border-box}body{background-color:#0a0e14;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif;min-height:100vh;display:flex;justify-content:center;align-items:center}dialog{background-color:#1a222e;border:1px solid #3a4a5a;border-radius:4px;width:90%;max-width:550px;box-shadow:0 8px 32px rgba(0,0,0,.6);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)}dialog::backdrop{background-color:rgba(0,0,0,.7)}.dialog-header{background-color:#3a506b;padding:14px 18px;border-bottom:1px solid #2a3a4a}.dialog-header h2{color:#e8e8e8;font-size:15px;font-weight:600;margin:0}.dialog-content{padding:24px 20px;text-align:center}.dialog-content p{color:#d0d0d0;font-size:15px;line-height:1.7;margin:0 0 16px}.dialog-content p:last-child{margin-bottom:0;font-style:italic;color:#a0a0a0}
                </style>
            </head>
            <body>
                <dialog open>
                    <div class="dialog-header"><h2>KoboldCpp is not available.</h2></div>
                    <div class="dialog-content">
                        <p>It may take some time during a model (re)load before it is ready to use.</p>
                        <p>Taking a long time for this message to go away?<br>It may have crashed, check the logs.</p>
                        <p>Your browser should automatically refresh when KoboldCpp is back online.</p>
                    </div>
                </dialog>
            </body>
            <script>
                setInterval(async () => {
                    try {
                        const response = await fetch(window.location.href, { cache: "no-store" });
                        if (response.ok) {
                            window.location.reload();
                        }
                    } catch (err) {
                        // Ignore network errors and try again on next interval
                    }
                }, 2000);
            </script>
            </html>
            """
            self.send_response(502)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_502.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html_502.encode("utf-8"))
            return


        self.send_response(resp.status, resp.reason) # forward response headers
        for k, v in resp.getheaders():
            lk = k.lower()
            if lk in self.HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True

        try:  # stream response
            while True:
                chunk = resp.read(self.STREAM_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
                conn.close()

    # proxy all HTTP methods
    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_PATCH = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle

class KcppProxyHttpServer(http.server.HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, upstream_port):
        self.upstream_port = upstream_port
        super().__init__(server_address, RequestHandlerClass)
    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._worker,args=(request, client_address),daemon=True)
        thread.start()
    def _worker(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        finally:
            self.shutdown_request(request)

def run_router_proxy(proxy_port, upstream_port):
    server = KcppProxyHttpServer(("", proxy_port), KcppProxyHandler, upstream_port)
    if state.args.ssl and state.sslvalid:
        import ssl
        if state.args.nocertify:
            ssl._create_default_https_context = ssl._create_unverified_context
        certpath = os.path.abspath(state.args.ssl[0])
        keypath = os.path.abspath(state.args.ssl[1])
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=certpath, keyfile=keypath)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f"KoboldCpp Proxy starting on port {proxy_port} (SSL/HTTPS), forwarding to port {upstream_port}",flush=True)
    else:
        print(f"KoboldCpp Proxy starting on port {proxy_port}, forwarding to port {upstream_port}",flush=True)
    proxy_thread = threading.Thread(target=server.serve_forever, daemon=True)
    proxy_thread.start()
    return server  # Return the server object in case you need to shut it down later


#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
class KcppServerRequestHandler(http.server.SimpleHTTPRequestHandler):
    sys_version = "1"
    server_version = "KoboldCppServer"

    def __init__(self, addr, port):
        self.addr = addr
        self.port = port

    def __call__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        if state.showdebug:
            super().log_message(format, *args)
        pass

    def extract_formdata_from_file_upload(self, body):
        result = {"file": None, "prompt": None, "language": None}
        try:
            if 'content-type' in self.headers and self.headers['content-type']:
                boundary = self.headers['content-type'].split("=")[1].encode()
                if boundary:
                    fparts = body.split(boundary)
                    for fpart in fparts:
                        detected_upload_filename = re.findall(r'Content-Disposition[^;]*;\s*name=(?:"file"|file)\s*;\s*filename=(?:"([^"]+)"|([^\s";]+))', fpart.decode('utf-8',errors='ignore'),flags=re.IGNORECASE)
                        detected_upload_filename_comfy = re.findall(r'Content-Disposition[^;]*;\s*name=(?:"image"|image)\s*;\s*filename=(?:"([^"]+)"|([^\s";]+))', fpart.decode('utf-8',errors='ignore'),flags=re.IGNORECASE)
                        if detected_upload_filename and len(detected_upload_filename)>0:
                            utfprint(f"Detected uploaded file: {detected_upload_filename[0]}")
                            file_content_start = fpart.find(b'\r\n\r\n') + 4  # Position after headers
                            file_content_end = fpart.rfind(b'\r\n')  # Ending boundary
                            if file_content_start != -1 and file_content_end != -1:
                                if "file" in result and result["file"] is None:
                                    file_data = fpart[file_content_start:file_content_end]
                                    file_data_base64 = base64.b64encode(file_data).decode('utf-8',"ignore")
                                    base64_string = f"data:audio/wav;base64,{file_data_base64}"
                                    result["file"] = base64_string
                        elif detected_upload_filename_comfy and len(detected_upload_filename_comfy)>0:
                            utfprint(f"Detected uploaded image: {detected_upload_filename_comfy[0]}")
                            file_content_start = fpart.find(b'\r\n\r\n') + 4  # Position after headers
                            file_content_end = fpart.rfind(b'\r\n')  # Ending boundary
                            if file_content_start != -1 and file_content_end != -1:
                                if "file" in result and result["file"] is None:
                                    file_data = fpart[file_content_start:file_content_end]
                                    file_data_base64 = base64.b64encode(file_data).decode('utf-8',"ignore")
                                    base64_string = f"{file_data_base64}"
                                    result["file"] = base64_string

                        # Check for fields
                        detected_prompt_field = re.findall(r'Content-Disposition.*name="prompt"\r\n\r\n(.*)\r\n', fpart.decode('utf-8', errors='ignore'))
                        if detected_prompt_field and len(detected_prompt_field)>0:
                            result["prompt"] = detected_prompt_field[0].strip()  # Extract and strip whitespace

                        detected_lang_field = re.findall(r'Content-Disposition.*name="language"\r\n\r\n(.*)\r\n', fpart.decode('utf-8', errors='ignore'))
                        if detected_lang_field and len(detected_lang_field)>0:
                            result["language"] = detected_lang_field[0].strip()  # Extract and strip whitespace

            if not ("file" in result and result["file"]):
                print("Uploaded file not found.")
            return result
        except Exception as e:
            print(f"File Upload Process Error: {e}")
            return result

    def prepare_basic_responses_body(self,resp_id,genparams):

        modelNameToReturn = state.friendlymodelname
        if state.autoswapmode and state.textName is not None:
            modelNameToReturn = state.textName
        ret = {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "completed_at": None,
            "incomplete_details": None,
            "previous_response_id": None,
            "truncation": "disabled",
            "parallel_tool_calls": False,
            "text": {"format": {"type": "text"},"verbosity": "medium"},
            "instructions": genparams.get('instructions', None),
            "model": modelNameToReturn,
            "error": None,
            "metadata": {},
            "tools": genparams.get('tools', []),
            "tool_choice": "auto",
            "background": False,
            "service_tier": "default",
            "safety_identifier": None,
            "prompt_cache_key": None,
            "max_tool_calls": None,
            "store": False,
            "top_p": genparams.get("top_p", 0.92),
            "max_output_tokens":genparams.get("max_length", None),
            "presence_penalty": genparams.get("presence_penalty", 0),
            "frequency_penalty": genparams.get("frequency_penalty", 0),
            "top_logprobs": 0,
            "temperature": genparams.get("temperature", 1),
            "reasoning": {"effort": None, "summary": None},
            "usage": None
        }
        return ret

    async def generate_text(self, genparams, api_format, stream_flag):

        currfinishreason = None
        req_id_suffix = genparams.get('oai_uniqueid',1)
        chatcmpl_id = f"chatcmpl-A{req_id_suffix}"
        cmpl_id = f"cmpl-A{req_id_suffix}"

        def run_blocking():  # api format 1=basic,2=kai,3=oai,4=oai-chat
            # flag instance as non-idle for a while
            washordereq = genparams.get('genkey', '').startswith('HORDEREQ_')
            if not washordereq:
                last_non_horde_req_time = time.time()

            return generate(genparams=genparams,stream_flag=stream_flag)

        genout = {"text": "", "status": -1, "stopreason": -1, "prompt_tokens":0, "completion_tokens": 0, "total_tokens": 0}
        if stream_flag:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor()
            genout = await loop.run_in_executor(executor, run_blocking)
        else:
            genout = run_blocking()

        recvtxt = genout['text']
        prompttokens = genout['prompt_tokens'] if genout['prompt_tokens'] > 0 else 0
        comptokens = genout['completion_tokens'] if genout['completion_tokens'] > 0 else 0
        currfinishreason = "error" if (genout['stopreason'] == -2) else ("length" if (genout['stopreason'] != 1) else "stop")

        # grab logprobs if not streaming
        logprobsdict = None
        if not stream_flag and ("logprobs" in genparams and genparams["logprobs"]):
            lastlogprobs = state.handle.last_logprobs()
            logprobsdict = parse_last_logprobs(lastlogprobs)

        # flag instance as non-idle for a while
        washordereq = genparams.get('genkey', '').startswith('HORDEREQ_')
        if not washordereq:
            last_non_horde_req_time = time.time()

        utfprint("\nOutput: " + recvtxt,1)

        #tool calls resolution
        tool_calls = []
        if api_format == 4 or api_format == 2 or api_format == 8:
            using_openai_tools = genparams.get('using_openai_tools', False)
            if using_openai_tools:
                # first, check and potentially segment multiple tags for multi-tool calls
                tool_calls = repack_toolcall_tags(recvtxt,genparams.get('tools', []))
                if tool_calls and len(tool_calls)>0:
                    flat = []
                    for obj in tool_calls:
                        if isinstance(obj, list):
                            flat.extend(obj)
                        else:
                            flat.append(obj)
                    tool_calls = [normalize_tool_call(obj) for obj in flat]
                    for tc in tool_calls:
                        tcarg = tc.get("function",{}).get("arguments",None)
                        tc["id"] = f"call_{random.randint(10000, 99999)}"
                        if tcarg is not None and not isinstance(tcarg, str):
                            tc["function"]["arguments"] = json.dumps(tcarg)
                    recvtxt = None
                    currfinishreason = "tool_calls"
                    if state.args.debugmode >= 1:
                        print(f"\nDebug ToolCall Response: {json.dumps(tool_calls)}")

        modelNameToReturn = state.friendlymodelname
        if state.autoswapmode and state.textName is not None:
            modelNameToReturn = state.textName

        #handle potential think tags, but only chat completions will return them. the others just drop them
        reasoningtxt = ""
        if api_format==4 or api_format==8 or api_format==9: #chat completions, responses and anthropic messages, but only chat has reasoning returned
            if recvtxt:
                for pair in state.thinkformats:
                    starter = pair['start']
                    ender = pair['end']
                    start_idx = recvtxt.find(starter)
                    end_idx = recvtxt.find(ender, start_idx + len(starter))
                    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                        reasoningtxt = recvtxt[start_idx + len(starter):end_idx]
                        recvtxt = recvtxt[:start_idx] + recvtxt[end_idx + len(ender):]
                        break
                    elif starter not in recvtxt and ender in recvtxt:
                        parts = recvtxt.split(ender, 1)
                        reasoningtxt = parts[0]
                        recvtxt = parts[1]
                        break
        if api_format == 1:
            res = {"data": {"seqs": [recvtxt]}}
        elif api_format == 3:
            res = {"id": cmpl_id, "object": "text_completion", "created": int(time.time()), "model": modelNameToReturn,
                   "usage": {"prompt_tokens": prompttokens, "completion_tokens": comptokens, "total_tokens": (prompttokens+comptokens)},
                   "choices": [{"text": recvtxt, "index": 0, "finish_reason": state.currfinishreason, "logprobs":logprobsdict}]}
        elif api_format == 4: #chat completions
            ccmsg = {"role": "assistant", "content": recvtxt, "tool_calls": tool_calls}
            if reasoningtxt and genparams.get('encapsulate_thinking', True):
                ccmsg["reasoning_content"] = reasoningtxt
            else:
                ccmsg["content"] = reasoningtxt + (recvtxt if recvtxt else "")
            res = {"id": chatcmpl_id, "object": "chat.completion", "created": int(time.time()), "model": modelNameToReturn,
                   "usage": {"prompt_tokens": prompttokens, "completion_tokens": comptokens, "total_tokens": (prompttokens+comptokens)},
                   "choices": [{"index": 0, "message": ccmsg, "finish_reason": state.currfinishreason, "logprobs":logprobsdict}]}
        elif api_format == 5:
            res = {"caption": end_trim_to_sentence(recvtxt)}
        elif api_format == 6:
            oldprompt = genparams.get('ollamabodyprompt', "")
            tokarr = tokenize_ids(oldprompt+recvtxt,False)
            res = {"model": modelNameToReturn,"created_at": str(datetime.now(timezone.utc).isoformat()),"response":recvtxt,"done": True,"done_reason":state.currfinishreason,"context": tokarr,"total_duration": 1,"load_duration": 1,"prompt_eval_count": prompttokens,"prompt_eval_duration": 1,"eval_count": comptokens,"eval_duration": 1}
        elif api_format == 7:
            res = {"model": modelNameToReturn,"created_at": str(datetime.now(timezone.utc).isoformat()),"message":{"role":"assistant","content":recvtxt},"done": True,"done_reason":state.currfinishreason,"total_duration": 1,"load_duration": 1,"prompt_eval_count": prompttokens,"prompt_eval_duration": 1,"eval_count": comptokens,"eval_duration": 1}
        elif api_format == 8: #oai-responses
            resp_id = f"resp-A{genparams.get('oai_uniqueid', 1)}"
            output_item_id = f"msg_0{genparams.get('oai_uniqueid', 1)}"
            output_items = []
            if recvtxt is not None:    # Add text message if there's content
                output_items.append({
                    "type": "message",
                    "id": output_item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": recvtxt, "annotations": [], "logprobs": []}]
                })
            if tool_calls and len(tool_calls) > 0:  # Add function call items if tool calls exist
                for tc in tool_calls:
                    output_items.append({"type": "function_call", "id": tc.get("id", ""), "call_id": tc.get("id", ""), "name": tc.get("function", {}).get("name", ""), "arguments": tc.get("function", {}).get("arguments", "{}"), "status": "completed"})
            res = self.prepare_basic_responses_body(resp_id,genparams)
            res["completed_at"] = int(time.time())
            res["status"] = "completed" if state.currfinishreason != "error" else "failed"
            res["output"] = output_items
            res["usage"] = {"input_tokens": prompttokens, "output_tokens": comptokens, "total_tokens": prompttokens + comptokens, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
        elif api_format == 9: # Anthropic Format
            anthropic_reason = "end_turn" if state.currfinishreason == "stop" else ("max_tokens" if state.currfinishreason == "length" else "stop_sequence")
            res = {
                "id": f"msg_A{req_id_suffix}",
                "type": "message",
                "role": "assistant",
                "model": modelNameToReturn,
                "content": [{"type": "text", "text": recvtxt}],
                "stop_reason": anthropic_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": prompttokens, "output_tokens": comptokens}
            }
        else: #kcpp format
            res = {"results": [{"text": recvtxt, "tool_calls": tool_calls, "finish_reason": state.currfinishreason, "logprobs":logprobsdict, "prompt_tokens": prompttokens, "completion_tokens": comptokens}]}

        try:
            return res
        except Exception as e:
            print(f"Generate: Error while generating: {e}")

    async def send_oai_sse_event(self, data):
        if data and data.strip()=="[DONE]":
            self.wfile.write(f'data: {data.strip()}\n\n'.encode())
        else:
            self.wfile.write(f'data: {data}\n\n'.encode())
        self.wfile.flush()

    async def send_oai_responses_sse_event(self, eventname, data):
        self.wfile.write(f'event: {eventname}\ndata: {data}\n\n'.encode())
        self.wfile.flush()

    async def send_anthropic_sse_event(self, eventname, data):
        self.wfile.write(f'event: {eventname}\ndata: {data}\n\n'.encode())
        self.wfile.flush()

    async def send_kai_sse_event(self, data):
        self.wfile.write('event: message\n'.encode())
        self.wfile.write(f'data: {data}\n\n'.encode())
        self.wfile.flush()

    async def handle_sse_stream(self, genparams, api_format):

        modelNameToReturn = state.friendlymodelname
        if state.autoswapmode and state.textName is not None:
            modelNameToReturn = state.textName

        using_openai_tools = genparams.get('using_openai_tools', False)
        req_id_suffix = genparams.get('oai_uniqueid',1)
        chatcmpl_id = f"chatcmpl-A{req_id_suffix}"
        cmpl_id = f"cmpl-A{req_id_suffix}"
        self.send_response(200)
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers(content_type='text/event-stream')

        # if tools, do not send anything else - OAI tool calls will be handled with fakestreaming!
        # only exception is if we know the exact toolcall tag to segment!
        tool_segment_tag = ""
        for start, end, streamhandled in state.tool_call_pairs:
            if streamhandled and state.cached_chat_template and start in state.cached_chat_template:
                tool_segment_tag = start
                break
        jinjatools = (state.args.jinja and state.args.jinja_tools)
        if api_format == 4 and using_openai_tools:
            if not jinjatools or not tool_segment_tag:
                genparams['sync_toolcall_stream_ineligible'] = True
                return

        think_tag_buf = ""
        encap_in_thinking = False
        if genparams.get('already_started_thinking', False):
            encap_in_thinking = True
        encap_first_loop = True
        thinkpairs = json.loads(json.dumps(state.thinkformats))
        responses_first_loop = True
        anthropic_first_loop = True
        rseq_num = 0
        current_token = 0
        prompttokens = 0
        incomplete_token_buffer = bytearray()
        async_sleep_short = 0.02
        await asyncio.sleep(0.35) #anti race condition, prevent check from overtaking generate
        batch_request_id = genparams.get('_batch_request_id', -1)
        batch_final_result = None

        try:
            tokenReserve = "" #keeps fully formed tokens that we cannot send out yet
            while True:
                if batch_request_id < 0:
                    batch_request_id = genparams.get('_batch_request_id', -1)
                    if genparams.get('_batch_expected', False) and batch_request_id < 0 and not genparams.get('_batch_fallback', False):
                        await asyncio.sleep(async_sleep_short)
                        continue
                using_batch_stream = batch_request_id >= 0
                streamDone = state.handle.batch_generate_has_finished(batch_request_id) if using_batch_stream else state.handle.has_finished() #exit next loop on done
                if streamDone:
                    if using_batch_stream and batch_final_result is None:
                        batch_final_result = state.handle.batch_generate_result(batch_request_id)
                    sr = batch_final_result.stopreason if using_batch_stream else state.handle.get_last_stop_reason()
                    currfinishreason = "error" if sr==-2 else ("length" if (sr!=1) else "stop")
                    prompttokens = batch_final_result.prompt_tokens if using_batch_stream else state.handle.get_last_input_count()
                tokenStr = ""
                streamcount = state.handle.batch_generate_stream_count(batch_request_id) if using_batch_stream else state.handle.get_stream_count()
                while current_token < streamcount:
                    token = state.handle.batch_generate_new_token(batch_request_id, current_token) if using_batch_stream else state.handle.new_token(current_token)

                    if token is None: # Token isnt ready yet, received nullpointer
                        break

                    current_token += 1
                    newbyte = ctypes.string_at(token)
                    incomplete_token_buffer += bytearray(newbyte)
                    tokenSeg = incomplete_token_buffer.decode("UTF-8","ignore")
                    incseq = is_incomplete_utf8_sequence(incomplete_token_buffer)
                    badFragment = (tokenSeg==" " and len(incomplete_token_buffer)>1) or incseq #partial incomplete unicode
                    if tokenSeg!="" and not badFragment:
                        incomplete_token_buffer.clear()
                        tokenStr += tokenSeg

                if tokenStr!="" or streamDone:
                    # split think tag handling
                    tokenStr = think_tag_buf + tokenStr
                    think_tag_buf = ""
                    if not streamDone and genparams.get('encapsulate_thinking', True):
                        tail = ""
                        for pair in thinkpairs:
                            for tag in (pair["start"], pair["end"]):
                                for n in range(1, len(tag)):
                                    if tokenStr.endswith(tag[:n]) and len(tag[:n]) > len(tail):
                                        tail = tag[:n]
                        if tail:
                            think_tag_buf = tail
                            tokenStr = tokenStr[:-len(tail)]
                    # end split think tag handling

                    sseq = genparams.get('stop_sequence', [])
                    trimstop = genparams.get('trim_stop', True)
                    if trimstop and not streamDone and string_contains_or_overlaps_sequence_substring(tokenStr,sseq):
                        tokenReserve += tokenStr
                        await asyncio.sleep(async_sleep_short) #if a stop sequence could trigger soon, do not send output
                    else:
                        if tokenStr!="" or tokenReserve!="":
                            tokenStr = tokenReserve + tokenStr
                            tokenReserve = ""

                            #apply trimming if needed
                            if trimstop:
                                for trim_str in sseq:
                                    sindex = tokenStr.find(trim_str)
                                    if sindex != -1 and trim_str!="":
                                        tokenStr = tokenStr[:sindex]

                        sync_potential_toolcall_splitmatch = ""
                        if tokenStr!="" or streamDone:
                            # Tool boundary detection for tool-capable chat completions.
                            # if triggered, stop real streaming, and let the buffered fakestreaming take over
                            if api_format == 4 and using_openai_tools:
                                tokenStr = tokenReserve + tokenStr
                                tokenReserve = ""
                                if tool_segment_tag in tokenStr:
                                    if not genparams.get("sync_toolcall_potential_triggered",False):
                                        sync_potential_toolcall_splitmatch = tool_segment_tag
                                        genparams['sync_toolcall_potential_triggered'] = True #if tool calls is triggered, rest will be sync fake streaming. we'll buffer it for later

                            need_split_final_msg = True if (state.currfinishreason is not None and streamDone and tokenStr!="") else False

                            # Hack for lcppui reasoning_content for thinking models
                            delta = {'role': 'assistant'}
                            if genparams.get('encapsulate_thinking', True):
                                if encap_in_thinking:
                                    foundend = False
                                    for pair in thinkpairs:
                                        if pair["end"] in tokenStr:
                                            encap_in_thinking = False
                                            foundend = True
                                            out1, out2 = tokenStr.split(pair["end"], 1)
                                            # Swallow any extraneous start tags that appear while already thinking
                                            if pair["start"] in out1:
                                                out1 = out1.replace(pair["start"], "")
                                            if out1:
                                                delta['reasoning_content'] = out1
                                            # Swallow any extraneous end tags that appear after already ending
                                            if pair["end"] in out2:
                                                out2 = out2.replace(pair["end"], "")
                                            if out2:
                                                delta['content'] = out2
                                            break
                                    if not foundend:
                                        # Still thinking - swallow extraneous start tags from THIS pair only
                                        cleaned = tokenStr
                                        if pair["start"] in cleaned:
                                            cleaned = cleaned.replace(pair["start"], "")
                                        delta['reasoning_content'] = cleaned
                                else:
                                    # Not thinking. Let's see if a start tag appears in this chunk.
                                    matched_start = False
                                    for pair in thinkpairs:
                                        # Condition A: The prompt ended exactly with the start tag
                                        if encap_first_loop and genparams.get("prompt", "").endswith(pair["start"]):
                                            encap_in_thinking = True
                                            thinkpairs = [pair] # lock in this pair
                                            delta['reasoning_content'] = tokenStr
                                            matched_start = True
                                            break
                                        # Condition B: The start tag is inside this chunk
                                        elif pair["start"] in tokenStr:
                                            encap_in_thinking = True
                                            thinkpairs = [pair] # lock in this pair
                                            out1, out2 = tokenStr.split(pair["start"], 1)
                                            # Preserve text that came BEFORE the start tag
                                            if out1:
                                                delta['content'] = out1
                                            if out2:
                                                delta['reasoning_content'] = out2
                                            # Edge Case: The end tag is ALSO in this exact same chunk (in out2)
                                            if pair["end"] in out2:
                                                encap_in_thinking = False
                                                out2_think, out2_content = out2.split(pair["end"], 1)
                                                # Overwrite reasoning with the exact thinking part
                                                delta['reasoning_content'] = out2_think
                                                # Append anything after the end tag to the content part
                                                if out2_content:
                                                    delta['content'] = delta.get('content', '') + out2_content
                                            matched_start = True
                                            break
                                    # Condition C: No start tag found, just normal text
                                    # Swallow any extraneous end tags that appear while not thinking
                                    if not matched_start:
                                        cleaned = tokenStr
                                        # Only swallow stray end tags if we've already locked in a pair
                                        if len(thinkpairs) == 1 and thinkpairs[0]["end"] in cleaned:
                                            cleaned = cleaned.replace(thinkpairs[0]["end"], "")
                                        delta['content'] = cleaned
                                encap_first_loop = False
                            else:
                                delta['content'] = tokenStr

                            if genparams.get("sync_toolcall_potential_triggered",False) and delta: # if sync_toolcall_potential_triggered, buffer up the impending content chunk for tools in fakestreaming, in case toolcalls fail
                                ec = genparams.get("sync_toolcall_extra_content","")
                                erc = genparams.get("sync_toolcall_extra_reasoning_content","")
                                ec += delta.get("content","")
                                erc += delta.get("reasoning_content","")
                                if erc and sync_potential_toolcall_splitmatch and sync_potential_toolcall_splitmatch in erc:
                                    parts = erc.split(sync_potential_toolcall_splitmatch,1)
                                    erc = sync_potential_toolcall_splitmatch + parts[1]
                                    delta["reasoning_content"] = parts[0]
                                elif ec and sync_potential_toolcall_splitmatch and sync_potential_toolcall_splitmatch in ec:
                                    parts = ec.split(sync_potential_toolcall_splitmatch,1)
                                    ec = sync_potential_toolcall_splitmatch + parts[1]
                                    delta["content"] = parts[0]
                                genparams['sync_toolcall_extra_content'] = ec
                                genparams['sync_toolcall_extra_reasoning_content'] = erc
                                if not sync_potential_toolcall_splitmatch:
                                    if not streamDone:
                                        await asyncio.sleep(async_sleep_short)
                                        continue
                                    await asyncio.sleep(async_sleep_short)
                                    return

                            if need_split_final_msg: #we need to send one message without the finish reason, then send a finish reason with no msg to follow standards
                                if api_format == 4:  # if oai chat, set format to expected openai streaming response
                                    event_str = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"delta":delta}]})
                                    await self.send_oai_sse_event(event_str)
                                elif api_format == 3:  # non chat completions
                                    event_str = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"text":tokenStr}]})
                                    await self.send_oai_sse_event(event_str)
                                else:
                                    event_str = json.dumps({"token": tokenStr, "finish_reason":None})
                                    await self.send_kai_sse_event(event_str)
                                tokenStr = "" # now the final finish reason can be sent alone
                                if delta and 'role' in delta:
                                    delta = {'role':delta["role"],'content':''}
                            if api_format == 4:  # if oai chat, set format to expected openai streaming response
                                if streamDone and ("logprobs" in genparams and genparams["logprobs"]): # this is a hack that sends an extra message containing ALL the logprobs
                                    lastlogprobs = state.handle.last_logprobs()
                                    logprobsdict = parse_last_logprobs(lastlogprobs)
                                    addonstr = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"delta":{'role':'assistant','content':''},"logprobs":logprobsdict}]})
                                    await self.send_oai_sse_event(addonstr)
                                event_str = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":state.currfinishreason,"delta":delta}]})
                                genparams['sync_toolcall_first_role_sent'] = True
                                await self.send_oai_sse_event(event_str)
                            elif api_format == 3:  # non chat completions
                                if streamDone and ("logprobs" in genparams and genparams["logprobs"]): # this is a hack that sends an extra message containing ALL the logprobs
                                    lastlogprobs = state.handle.last_logprobs()
                                    logprobsdict = parse_last_logprobs(lastlogprobs)
                                    addonstr = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"text":"","logprobs":logprobsdict}]})
                                    await self.send_oai_sse_event(addonstr)
                                event_str = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":state.currfinishreason,"text":tokenStr}]})
                                await self.send_oai_sse_event(event_str)
                            elif api_format == 8: #oai-responses
                                resp_id = f"resp-A{genparams.get('oai_uniqueid', 1)}"
                                item_id = f"msg_0{genparams.get('oai_uniqueid', 1)}"
                                # Send response.created once at the start (only on first iteration)
                                if responses_first_loop:
                                    res = self.prepare_basic_responses_body(resp_id, genparams)
                                    res["status"] = "in_progress"
                                    res["output"] = []
                                    created_event = json.dumps({"type": "response.created", "response": res, "sequence_number":rseq_num})
                                    rseq_num += 1
                                    await self.send_oai_responses_sse_event("response.created",created_event)
                                    # response.output_item.added
                                    item_added = json.dumps({"type": "response.output_item.added", "output_index": 0, "sequence_number":rseq_num, "item": { "type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []}})
                                    rseq_num += 1
                                    await self.send_oai_responses_sse_event("response.output_item.added",item_added)
                                    # content_part.added
                                    part_added = json.dumps({"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "sequence_number":rseq_num, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
                                    rseq_num += 1
                                    await self.send_oai_responses_sse_event("response.content_part.added",part_added)
                                    responses_first_loop = False
                                if tokenStr != "" or streamDone:
                                    if tokenStr != "":
                                        delta_event = json.dumps({"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "sequence_number":rseq_num, "logprobs":[], "content_index": 0, "delta": tokenStr})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.output_text.delta",delta_event)
                                    if streamDone:
                                        # content_part.done, reply full text
                                        await asyncio.sleep(async_sleep_short)
                                        finalraw = state.handle.batch_generate_pending_output(batch_request_id) if using_batch_stream else state.handle.get_pending_output()
                                        finaltxt = finalraw.decode("UTF-8", "ignore")
                                        await asyncio.sleep(async_sleep_short)
                                        done_event = json.dumps({"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "sequence_number":rseq_num, "content_index": 0, "text": finaltxt})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.output_text.done",done_event)
                                        # response.output_item.done
                                        item_done = json.dumps({"type": "response.output_item.done", "output_index": 0, "sequence_number":rseq_num, "item": { "type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": finaltxt, "annotations": [], "logprobs": []}]}})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.output_item.done",item_done)
                                        usage_pp = batch_final_result.prompt_tokens if using_batch_stream else state.handle.get_last_input_count()
                                        usage_gen = current_token
                                        res = self.prepare_basic_responses_body(resp_id,genparams)
                                        res["completed_at"] = int(time.time())
                                        res["status"] = "completed" if state.currfinishreason != "error" else "failed"
                                        res["output"] = [{"type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": finaltxt, "annotations": [], "logprobs": []}]}]
                                        res["usage"] = {"input_tokens": usage_pp,"input_tokens_details":{"cached_tokens":0}, "output_tokens": usage_gen, "output_tokens_details":{"reasoning_tokens":0}, "total_tokens": usage_pp + usage_gen}
                                        completed_event = json.dumps({"type": "response.completed", "response": res, "sequence_number":rseq_num})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.completed",completed_event)
                            elif api_format == 9: # Anthropic Streaming Format
                                if anthropic_first_loop:
                                    start_msg = json.dumps({"type":"message","id":f"msg_A{req_id_suffix}","role":"assistant","model":modelNameToReturn,"usage":{"input_tokens":prompttokens,"output_tokens":0}})
                                    await self.send_anthropic_sse_event("message_start", json.dumps({"type": "message_start", "message": json.loads(start_msg)}))
                                    await self.send_anthropic_sse_event("content_block_start", json.dumps({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}))
                                    anthropic_first_loop = False
                                if tokenStr != "":
                                    await self.send_anthropic_sse_event("content_block_delta", json.dumps({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":tokenStr}}))
                                if streamDone:
                                    anthropic_reason = "end_turn" if state.currfinishreason == "stop" else ("max_tokens" if state.currfinishreason == "length" else "stop_sequence")
                                    await self.send_anthropic_sse_event("content_block_stop", json.dumps({"type":"content_block_stop","index":0}))
                                    await self.send_anthropic_sse_event("message_delta", json.dumps({"type":"message_delta","delta":{"stop_reason":anthropic_reason,"stop_sequence":None},"usage":{"output_tokens":current_token}}))
                                    await self.send_anthropic_sse_event("message_stop", json.dumps({"type":"message_stop"}))
                            else:
                                event_str = json.dumps({"token": tokenStr, "finish_reason":state.currfinishreason})
                                await self.send_kai_sse_event(event_str)
                            tokenStr = ""
                        else:
                            await asyncio.sleep(async_sleep_short)
                else:
                    await asyncio.sleep(async_sleep_short) #this should keep things responsive

                if streamDone:
                    if api_format == 4 or api_format == 3:  # if oai chat, send last [DONE] message consistent with openai format
                        strop = genparams.get("stream_options",None)
                        if (strop and strop.get("include_usage",False)):  # Send a final chunk with usage info, only if requested
                            usage_obj = {"prompt_tokens": prompttokens, "completion_tokens": current_token, "total_tokens": (prompttokens + current_token)}
                            if api_format == 4:
                                usage_str = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[],"usage":usage_obj})
                            else:
                                usage_str = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[],"usage":usage_obj})
                            await self.send_oai_sse_event(usage_str)
                        await self.send_oai_sse_event('[DONE]')
                        await asyncio.sleep(async_sleep_short)
                    break
        except Exception as ex:
            print("Token streaming was interrupted or aborted!")
            print(ex)
            if batch_request_id >= 0:
                state.handle.batch_generate_abort(batch_request_id)
            else:
                state.handle.abort_generate()
            await asyncio.sleep(0.2) #short delay
        finally:
            if batch_request_id >= 0:
                state.handle.batch_generate_release(batch_request_id)
                genparams.pop('_batch_request_id', None)
            genparams.pop('_batch_expected', None)
            genparams.pop('_batch_fallback', None)

        # flush buffers, sleep a bit to make sure all data sent, and then force close the connection
        self.wfile.flush()
        await asyncio.sleep(0.1)
        self.close_connection = True
        await asyncio.sleep(0.05)

    async def monitor_connection(self): #Poll the socket to detect client disconnection during prompt processing
        import select
        loop = asyncio.get_event_loop()
        def check_connection_closed():
            try:
                sock = self.connection
                readable, _, exceptional = select.select([sock], [], [sock], 0)
                if exceptional:
                    return True
                if readable:
                    data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    if len(data) == 0:
                        return True
                return False
            except (OSError, Exception):
                return True  # Treat any error as disconnected
        while True:
            try:
                await asyncio.sleep(0.5)
                disconnected = await loop.run_in_executor(None, check_connection_closed)
                if disconnected:
                    if state.args.debugmode:
                        print("\nClient disconnected unexpectedly, aborting...")
                    state.handle.abort_generate()
                    return
            except Exception:
                return

    async def handle_request(self, genparams, api_format, stream_flag):
        tasks = []
        genparams["oai_uniqueid"] = random.randint(100000, 999999)
        monitor_task = None
        try:
            if stream_flag:
                tasks.append(self.handle_sse_stream(genparams, api_format))
            generate_task = asyncio.create_task(self.generate_text(genparams, api_format, stream_flag))
            tasks.append(generate_task)
            if stream_flag:
                monitor_task = asyncio.create_task(self.monitor_connection())
            await asyncio.gather(*tasks)
            generate_result = generate_task.result()
            return generate_result
        except (BrokenPipeError, ConnectionAbortedError) as cae: # attempt to abort if connection lost
            print("An ongoing connection was aborted or interrupted!")
            print(cae)
            state.handle.abort_generate()
            await asyncio.sleep(0.2) #short delay
        except Exception as e:
            print(e)
        finally:
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    def get_multiplayer_idle_state(self,userid):
        if state.modelbusy.locked() or state.batched_request_runner_count>0:
            return False
        for key, value in state.multiplayer_lastactive.items():
            if key!=userid and time.time()-value<6: #6s to idle
                return False
        return True

    def check_header_password(self, target_password, target_alt_password=None):
        auth_ok = True
        if target_password and target_password !="":
            auth_header = None
            auth_ok = False
            if 'Authorization' in self.headers:
                auth_header = self.headers['Authorization']
            elif 'authorization' in self.headers:
                auth_header = self.headers['authorization']
            if auth_header is not None and auth_header.startswith('Bearer '):
                token = auth_header[len('Bearer '):].strip()
                if token==target_password or (target_alt_password and target_alt_password!="" and token==target_alt_password):
                    auth_ok = True
        return auth_ok

    def secure_endpoint(self): #returns false if auth fails. caller should exit
        #handle password stuff
        auth_ok = self.check_header_password(state.password, state.args.adminpassword)
        if auth_ok is False:
            self.send_response(401)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"detail": {
                    "error": "Unauthorized",
                    "msg": "Authentication key is missing or invalid.",
                    "type": "unauthorized",
                }}).encode())
            return False
        return True

    def noscript_webui(self):
        parsed_url = urllib.parse.urlparse(self.path)
        parsed_dict = urllib.parse.parse_qs(parsed_url.query)
        reply = ""
        status = str(parsed_dict['status'][0]) if 'status' in parsed_dict else "Ready To Generate"
        prompt = str(parsed_dict['prompt'][0]) if 'prompt' in parsed_dict else ""
        chatmsg = str(parsed_dict['chatmsg'][0]) if 'chatmsg' in parsed_dict else ""
        imgprompt = str(parsed_dict['imgprompt'][0]) if 'imgprompt' in parsed_dict else ""
        max_length = int(parsed_dict['max_length'][0]) if 'max_length' in parsed_dict else 100
        temperature = float(parsed_dict['temperature'][0]) if 'temperature' in parsed_dict else 0.75
        top_k = int(parsed_dict['top_k'][0]) if 'top_k' in parsed_dict else 100
        top_p = float(parsed_dict['top_p'][0]) if 'top_p' in parsed_dict else 0.9
        rep_pen = float(parsed_dict['rep_pen'][0]) if 'rep_pen' in parsed_dict else 1.0
        ban_eos_token = int(parsed_dict['ban_eos_token'][0]) if 'ban_eos_token' in parsed_dict else 0
        steps = int(parsed_dict['steps'][0]) if 'steps' in parsed_dict else 25
        cfg = int(parsed_dict['cfg'][0]) if 'cfg' in parsed_dict else 7
        genbtnval = (parsed_dict['generate'][0] if 'generate' in parsed_dict else "")
        gencommand = (genbtnval=="Generate" or genbtnval=="Send")
        chatmode = int(parsed_dict['chatmode'][0]) if 'chatmode' in parsed_dict else 0
        imgmode = int(parsed_dict['imgmode'][0]) if 'imgmode' in parsed_dict else 0
        human_name = str(parsed_dict['human_name'][0]) if 'human_name' in parsed_dict else "User"
        bot_name = str(parsed_dict['bot_name'][0]) if 'bot_name' in parsed_dict else "Assistant"
        stops = []
        prefix = ""
        if chatmode:
            ban_eos_token = False
            prompt = prompt.replace("1HdNl1","\n")
            if chatmsg:
                prompt += f"\n{human_name}: {chatmsg}\n{bot_name}:"
            else:
                gencommand = False
            stops = [f"\n{human_name}:",f"\n{bot_name}:"]
            prefix = f"[This is a chat conversation log between {human_name} and {bot_name}.]\n"
        elif imgmode:
            if imgprompt:
                prompt = imgprompt
                max_length = 1
            else:
                gencommand = False

        if state.modelbusy.locked() or state.batched_request_runner_count>0:
            status = "Model is currently busy, try again later."
        elif gencommand:
            if prompt=="" or max_length<=0:
                status = "Need a valid prompt and length to generate."
            else:
                if max_length>512:
                    max_length = 512
                epurl = get_my_epurl()
                if imgmode and imgprompt:
                    gen_payload = {"prompt":{"3":{"class_type": "KSampler","inputs":{"cfg":cfg,"steps":steps,"latent_image":["5", 0],"positive": ["6", 0]}},"5":{"class_type": "EmptyLatentImage","inputs":{"height":512,"width":512}},"6":{"class_type": "CLIPTextEncode","inputs":{"text":imgprompt}}}}
                    respjson = make_url_request(f'{epurl}/prompt', gen_payload)
                else:
                    gen_payload = {"prompt": prefix+prompt,"max_length": max_length,"temperature": temperature,"top_k": top_k,"top_p": top_p,"rep_pen": rep_pen,"ban_eos_token":ban_eos_token, "stop_sequence":stops}
                    respjson = make_url_request(f'{epurl}/api/v1/generate', gen_payload)
                    reply = html.escape(respjson["results"][0]["text"])
                    if chatmode:
                        reply = " "+reply.strip()
                status = "Generation Completed"

            if "generate" in parsed_dict:
                del parsed_dict["generate"]
            if "chatmsg" in parsed_dict:
                del parsed_dict["chatmsg"]
            if "imgprompt" in parsed_dict:
                del parsed_dict["imgprompt"]
            parsed_dict["prompt"] = prompt + reply
            parsed_dict["status"] = status
            parsed_dict["chatmode"] = ("1" if chatmode else "0")
            parsed_dict["imgmode"] = ("1" if imgmode else "0")
            updated_query_string = urllib.parse.urlencode(parsed_dict, doseq=True)
            updated_path = parsed_url._replace(query=updated_query_string).geturl()
            self.path = updated_path
            time.sleep(0.5) #short delay
            self.send_response(302)
            self.send_header("location", self.path)
            self.end_headers(content_type='text/html')
            return

        imgbtn = '''<form action="/noscript" style="display: inline;">
        <input type="hidden" name="imgmode" value="1">
        <input type="submit" value="Image Mode">
        </form>'''

        bodycontent = f'''<b><u>{"Image Mode" if imgmode else ("Chat Mode" if chatmode else "Story Mode")}</u></b><br>'''
        optionscontent = ""
        if imgmode:
            randimg = f'<img src="view_image{random.randint(100, 999)}.png" width="320" width="320">'
            bodycontent += f'''<p>Generated Image: {prompt if prompt else "None"}</p>
            {randimg if prompt else ""}<br>
            <label>Image Prompt: </label><input type="text" size="40" value="" name="imgprompt">
            <input type="hidden" name="generate" value="Generate" />
            <input type="submit" value="Generate"> (Be patient)'''
        elif chatmode:
            oldconvo = prompt.strip().replace(f"{human_name}:",f"<b>{human_name}:</b>").replace(f"{bot_name}:",f"<b>{bot_name}:</b>").replace("\n","<br>")
            oldconvo += f'''<input type="hidden" name="human_name" value="{human_name}"><input type="hidden" name="bot_name" value="{bot_name}">'''
            newconvo = '''Start a new conversation.<br>
            <label>Your Name: </label> <input type="text" size="10" value="User" name="human_name"><br>
            <label>Bot Name: </label> <input type="text" size="10" value="Assistant" name="bot_name"><br>'''
            clnprompt = prompt.replace("\n","1HdNl1")
            bodycontent += f'''<p>{newconvo if prompt=="" else oldconvo}</p>
            <input type="hidden" name="prompt" value="{clnprompt}">
            <label>Say: </label><input type="text" size="40" value="" name="chatmsg">
            <input type="hidden" name="generate" value="Send" />
            <input type="submit" value="Send"> (Be patient)'''
        else:
            bodycontent += f'''
<textarea name="prompt" cols="60" rows="8" wrap="soft" placeholder="Enter Prompt Here">{prompt}</textarea><br>
<input type="hidden" name="generate" value="Generate" />
<input type="submit" value="Generate"> (Be patient)
'''
        if not imgmode:
            optionscontent = f'''<label>Gen. Amount</label> <input type="text" size="4" value="{max_length}" name="max_length"><br>
            <label>Temperature</label> <input type="text" size="4" value="{temperature}" name="temperature"><br>
            <label>Top-K</label> <input type="text" size="4" value="{top_k}" name="top_k"><br>
            <label>Top-P</label> <input type="text" size="4" value="{top_p}" name="top_p"><br>
            <label>Rep. Pen</label> <input type="text" size="4" value="{rep_pen}" name="rep_pen"><br>
            <label>Prevent EOS</label> <input type="checkbox" name="ban_eos_token" value="1" {"checked" if ban_eos_token else ""}><br>'''
        else:
            optionscontent = f'''<label>Steps</label> <input type="text" size="4" value="{steps}" name="steps"><br>
            <label>Cfg. Scale</label> <input type="text" size="4" value="{cfg}" name="cfg"><br>'''

        caps = get_capabilities()
        finalhtml = f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KoboldCpp NoScript Mode</title></head><body>
<h2>KoboldCpp NoScript Mode</h2>
<div>
<p>KoboldCpp can be used without Javascript enabled, however this is not recommended.
<br>If you have Javascript, please use <a href="/">KoboldAI Lite WebUI</a> instead.</p><hr>
<form action="/noscript">
{bodycontent}
<hr>
<b>{status}</b><br>
<hr>
{optionscontent}
<input type="hidden" name="chatmode" value="{chatmode}">
<input type="hidden" name="imgmode" value="{imgmode}">
</form>
<hr>
<div style="display: inline-block;">
Change Mode<br>
<form action="/noscript" style="display: inline;">
<input type="submit" value="Story Mode">
</form>
<form action="/noscript" style="display: inline;">
<input type="hidden" name="chatmode" value="1">
<input type="submit" value="Chat Mode">
</form>
{imgbtn if ("txt2img" in caps and caps["txt2img"]) else ""}
</div>
</div>
</body></html>'''
        finalhtml = finalhtml.encode('utf-8')
        self.send_response(200)
        self.send_header('content-length', str(len(finalhtml)))
        self.end_headers(content_type='text/html')
        self.wfile.write(finalhtml)

    def do_GET(self):

        clean_path = self.path.split("?")[0] #for cases where we do not want query params
        if clean_path=="/lcpp": #fix for svelte redirect issues, browser path needs to end with slash
            clean_path = "/lcpp/"
            self.send_response(302)
            self.send_header("location", clean_path)
            self.end_headers(content_type='text/html')
            return None

        clean_path = clean_path.rstrip('/')
        response_body = None
        content_type = 'application/json'
        content_encoding = None

        # Check if browser supports gzip
        accept_encoding = self.headers.get('Accept-Encoding', '')
        supports_gzip = 'gzip' in accept_encoding.lower()

        if clean_path!="/lcpp" and clean_path.startswith("/lcpp/"):
            clean_path = clean_path[5:] #adapt lcpp paths to the root

        if clean_path in [""]: # the root url is lite
            content_type = 'text/html'
            if supports_gzip and state.embedded_kailite_gz is not None:
                response_body = state.embedded_kailite_gz
                content_encoding = 'gzip'
            elif state.embedded_kailite is not None:
                response_body = state.embedded_kailite
            else:
                response_body = (f"Embedded KoboldAI Lite is not found.<br>You will have to connect via the main KoboldAI client, or <a href='https://lite.koboldai.net?local=1&port={self.port}'>use this URL</a> to connect.").encode()


        elif clean_path in ["/noscript","noscript"]: #noscript webui
            self.noscript_webui()
            return

        elif clean_path.endswith(('/manifest.json')):
            response_body = (json.dumps({"name":"KoboldAI Lite","short_name":"KoboldAI Lite","description":"Progressive Web App for KoboldAI Lite","start_url":"./","scope":".","display":"standalone","background_color":"#303030","theme_color":"#337ab7","orientation":"portrait-primary","icons":[{"src":"data:image/png;base64,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","type":"image/png","sizes":"150x150"}]}).encode())

        elif clean_path.endswith(('/api/v1/model', '/api/latest/model')):
            auth_ok = self.check_header_password(state.password, state.args.adminpassword)
            modelNameToReturn = state.friendlymodelname
            if state.autoswapmode and state.textName is not None:
                modelNameToReturn = state.textName
            response_body = (json.dumps({'result': (modelNameToReturn if auth_ok else "koboldcpp/protected-model") }).encode())

        elif clean_path.endswith(('/api/v1/config/max_length', '/api/latest/config/max_length')):
            response_body = (json.dumps({"value": state.maxhordelen}).encode())

        elif clean_path.endswith(('/api/v1/config/max_context_length', '/api/latest/config/max_context_length')):
            response_body = (json.dumps({"value": min(state.maxctx,(state.maxctx if state.maxhordectx==0 else state.maxhordectx))}).encode())

        elif clean_path.endswith(('/api/v1/config/soft_prompt', '/api/latest/config/soft_prompt')):
            response_body = (json.dumps({"value":""}).encode())

        elif clean_path.endswith(('/api/v1/config/soft_prompts_list', '/api/latest/config/soft_prompts_list')):
            response_body = (json.dumps({"values": []}).encode())

        elif clean_path.endswith(('/api/v1/info/version', '/api/latest/info/version')):
            response_body = (json.dumps({"result":"1.2.5"}).encode())

        elif clean_path.endswith(('/api/extra/true_max_context_length')): #do not advertise this to horde
            response_body = (json.dumps({"value": state.maxctx}).encode())

        elif clean_path.endswith(('/api/extra/version')):
            caps = get_capabilities()
            response_body = (json.dumps(caps).encode())

        elif clean_path.endswith(('/api/admin/list_options')):  # used by admin to get info about a kcpp instance
            opts = []
            if state.args.admin and state.args.admindir and os.path.exists(state.args.admindir) and self.check_header_password(state.args.adminpassword):
                opts = get_current_admindir_list()
            response_body = (json.dumps(opts).encode())

        elif clean_path.endswith(('/api/extra/perf')):
            lastp = state.handle.get_last_process_time()
            laste = state.handle.get_last_eval_time()
            lastc = state.handle.get_last_token_count()
            lastic = state.handle.get_last_input_count()
            totalgens = state.handle.get_total_gens()
            totalimggens = state.handle.get_total_img_gens()
            totalttsgens = state.handle.get_total_tts_gens()
            totaltranscribegens = state.handle.get_total_transcribe_gens()
            stopreason = state.handle.get_last_stop_reason()
            lastseed = state.handle.get_last_seed()
            lastdraftsuccess = state.handle.get_last_draft_success()
            lastdraftfailed = state.handle.get_last_draft_failed()
            t_pp = float(lastp)*float(lastic)*0.001
            t_gen = float(laste)*float(lastc)*0.001
            s_pp = float(lastic)/t_pp if t_pp>0 else 0
            s_gen = float(lastc)/t_gen if t_gen>0 else 0
            uptime = time.time() - state.start_time
            idletime = time.time() - state.last_req_time
            is_quiet = True if (state.args.quiet and state.args.debugmode != 1) else False
            response_body = json.dumps(
                {
                    "last_process": lastp,
                    "last_eval": laste,
                    "last_token_count": lastc,
                    "last_input_count": lastic,
                    "last_process_time": t_pp,
                    "last_eval_time": t_gen,
                    "last_process_speed": s_pp,
                    "last_eval_speed": s_gen,
                    "last_seed": lastseed,
                    "last_draft_success": lastdraftsuccess,
                    "last_draft_failed": lastdraftfailed,
                    "total_gens": state.totalgens,
                    "stop_reason": stopreason,
                    "total_img_gens": totalimggens,
                    "total_tts_gens": totalttsgens,
                    "total_transcribe_gens": totaltranscribegens,
                    "queue": state.requestsinqueue,
                    "idle": (0 if (state.modelbusy.locked() or state.batched_request_runner_count>0) else 1),
                    "hordeexitcounter": state.exitcounter,
                    "uptime": uptime,
                    "idletime": idletime,
                    "quiet": is_quiet,
                }
            ).encode()

        elif clean_path.endswith('/api/extra/generate/check'):
            if not self.secure_endpoint():
                return
            pendtxtStr = ""
            if state.requestsinqueue==0 and state.totalgens>0 and state.currentusergenkey=="":
                pendtxt = state.handle.get_pending_output()
                pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            response_body = (json.dumps({"results": [{"text": pendtxtStr}]}).encode())

        elif clean_path.endswith('/api/extra/last_logprobs'):
            if not self.secure_endpoint():
                return
            logprobsdict = None
            if state.requestsinqueue==0 and state.totalgens>0 and state.currentusergenkey=="":
                lastlogprobs = state.handle.last_logprobs()
                logprobsdict = parse_last_logprobs(lastlogprobs)
            response_body = (json.dumps({"logprobs":logprobsdict}).encode())

        elif clean_path.endswith('/v1/models') or clean_path=='/models':
            modelNameToReturn = state.friendlymodelname
            if state.autoswapmode and state.textName is not None:
                modelNameToReturn = state.textName

            mlist = [{"id":modelNameToReturn,"object":"model","created":int(time.time()),"owned_by":"koboldcpp","permission":[],"root":"koboldcpp"}]
            if state.args.routermode:
                alist = get_current_admindir_list()
                for itm in alist:
                    mlist.append({"id":itm,"object":"model","created":int(time.time()),"owned_by":"koboldcpp","permission":[],"root":"koboldcpp"})
            response_body = (json.dumps({"object":"list","data":mlist}).encode())

        elif clean_path.endswith('/sdapi/v1/loras'):
            response_body = (json.dumps(mk_sdapi_lora_list(state.imglora_bypath))).encode()

        elif clean_path.endswith('/sdapi/v1/upscalers'):
            if state.args.sdupscaler:
                response_body = (json.dumps([{"name":"ESRGAN_4x","model_name":"ESRGAN_4x","model_path":"upscaler_model.gguf","model_url":None,"scale":4}]).encode())
            else:
                response_body = (json.dumps([]).encode())

        elif clean_path.endswith('/sdapi/v1/sd-models'):
            if state.autoswapmode and state.imageName is not None:
                response_body = (json.dumps([{"title":state.imageName,"model_name":state.imageName,"hash":"8888888888","sha256":"8888888888888888888888888888888888888888888888888888888888888888","filename":state.imageName,"config": None}]).encode())
            elif state.friendlysdmodelname=="inactive" or state.fullsdmodelpath=="":
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([{"title":state.friendlysdmodelname,"model_name":state.friendlysdmodelname,"hash":"8888888888","sha256":"8888888888888888888888888888888888888888888888888888888888888888","filename":state.fullsdmodelpath,"config": None}]).encode())
        elif clean_path.endswith('/sdapi/v1/options'):
            modelNameToReturn = state.friendlysdmodelname
            if state.autoswapmode and state.imageName is not None:
                modelNameToReturn = state.imageName
            response_body = (json.dumps({"samples_format":"png","sd_model_checkpoint":modelNameToReturn}).encode())
        elif clean_path.endswith('/sdapi/v1/samplers'):
            if (state.friendlysdmodelname=="inactive" or state.fullsdmodelpath=="") and not(state.autoswapmode and state.imageName is not None):
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps(sd_sdapi_samplers()).encode())
        elif clean_path.endswith('/sdapi/v1/schedulers'):
            if (state.friendlysdmodelname=="inactive" or state.fullsdmodelpath=="") and not(state.autoswapmode and state.imageName is not None):
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([{"name":name,"label":name} for name in state.cached_sd_info.get('available_schedulers', [])]).encode())
        elif clean_path.endswith('/sdapi/v1/latent-upscale-modes'):
           response_body = (json.dumps([]).encode())
        elif clean_path.endswith('/sdapi/v1/upscalers'):
           response_body = (json.dumps([]).encode())

        #vits compatible
        elif clean_path=='/voice/check':
            response_body = (json.dumps({"id":4,"lang":["en"],"name":"KoboldCppTTS","status":"success"}).encode())
        elif clean_path=='/voice/speakers':
            response_body = (json.dumps({"VITS":[{"id":4,"lang":["en"],"name":"KoboldCppTTS"}]}).encode())
        elif clean_path=='/voice/vits':
            parsed_url = urllib.parse.urlparse(self.path)
            parsed_dict = urllib.parse.parse_qs(parsed_url.query)
            prompt = str(parsed_dict['text'][0]) if 'text' in parsed_dict else ""
            if prompt:
                epurl = get_my_epurl()
                content_type = 'audio/wav'
                response_body = make_url_request(f'{epurl}/api/extra/tts', {"input": prompt})
            pass

        elif clean_path.endswith('/speakers_list') or clean_path.endswith('/api/extra/speakers_list'): #xtts compatible
            response_body = (json.dumps(state.voicelist).encode()) #some random voices for them to enjoy
        elif clean_path.endswith('/speakers'): #xtts compatible
            tmplist = []
            for itm in state.voicelist:
                tmplist.append({"name":itm,"voice_id":itm,"preview_url":""})
            response_body = (json.dumps(tmplist).encode()) #some random voices for them to enjoy
        elif clean_path.endswith('/v1/audio/voices') or clean_path=='/audio/voices':
            response_body = (json.dumps({"status":"ok","voices":state.voicelist}).encode()) #some random voices for them to enjoy
        elif clean_path.endswith('/get_tts_settings'): #xtts compatible
            response_body = (json.dumps({"temperature":0.75,"speed":1,"length_penalty":1,"repetition_penalty":1,"top_p":1,"top_k":4,"enable_text_splitting":True,"stream_chunk_size":100}).encode()) #some random voices for them to enjoy

        elif clean_path.endswith('/api/tags') or clean_path.endswith('/api/ps'): #ollama compatible
            modelNameToReturn = state.friendlymodelname
            if state.autoswapmode and state.textName is not None:
                modelNameToReturn = state.textName
            response_body = (json.dumps({"models":[{"name":"koboldcpp","model":f"{modelNameToReturn}:latest","modified_at":"2024-07-19T15:26:55.6122841+08:00","expires_at": "2055-06-04T19:06:25.5433636+08:00","size":394998579,"size_vram":394998579,"digest":"b5dc5e784f2a3ee1582373093acf69a2f4e2ac1710b253a001712b86a61f88bb","details":{"parent_model":"","format":"gguf","family":"koboldcpp","families":["koboldcpp"],"parameter_size":"128M","quantization_level":"Q4_0"}},{"name":"koboldcpp","model":modelNameToReturn,"modified_at":"2025-01-01T01:00:00.0000000+00:00","expires_at": "2069-01-01T01:00:00.0000000+00:00","size":394998579,"size_vram":394998579,"digest":"b5dc5e784f2a3ee1582373093acf69a2f4e2ac1710b253a001712b86a61f88bb","details":{"parent_model":"","format":"gguf","family":"koboldcpp","families":["koboldcpp"],"parameter_size":"128M","quantization_level":"Q4_0"}}]}).encode())
        elif clean_path.endswith('/api/version'): #ollama compatible, NOT the kcpp version
            response_body = (json.dumps({"version":"0.7.0"}).encode())
        elif clean_path=='/ping':
            response_body = (json.dumps({"status": "healthy"}).encode())

        #comfyui compatible
        elif clean_path=='/system_stats':
            response_body = (json.dumps({"system":{"os":"posix","ram_total":12345678900,"ram_free":12345678900,"comfyui_version":"v0.3.4-3-g7126ecf","python_version":"3.10.12","pytorch_version":"2.5.1","embedded_python":False,"argv":[]},"devices":[{"name":"koboldcpp","type":"cuda","index":0,"vram_total":12345678900,"vram_free":12345678900,"torch_vram_total":12345678900,"torch_vram_free":12345678900}]}).encode())
        elif clean_path=='/object_info':
            modelNameToReturn = state.friendlysdmodelname
            if state.autoswapmode and state.imageName is not None:
                modelNameToReturn = state.imageName
            response_body = (json.dumps({"KSampler":{"input":{"required":{"model":["MODEL",{"tooltip":""}],"seed":["INT",{"default":0,"min":0,"max":512,"tooltip":""}],"steps":["INT",{"default":20,"min":1,"max":512,"tooltip":""}],"cfg":["FLOAT",{"default":8.0,"min":0.0,"max":100.0,"step":0.1,"round":0.01,"tooltip":"512"}],"sampler_name":[["euler"],{"tooltip":""}],"scheduler":[["normal"],{"tooltip":""}],"positive":["CONDITIONING",{"tooltip":""}],"negative":["CONDITIONING",{"tooltip":""}],"latent_image":["LATENT",{"tooltip":""}],"denoise":["FLOAT",{"default":1.0,"min":0.0,"max":1.0,"step":0.01,"tooltip":""}]}},"input_order":{"required":["model","seed","steps","cfg","sampler_name","scheduler","positive","negative","latent_image","denoise"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"KSampler","display_name":"KSampler","description":"KSampler","python_module":"nodes","category":"sampling","output_node":False,"output_tooltips":[""]},"CheckpointLoaderSimple":{"input":{"required":{"ckpt_name":[[modelNameToReturn],{"tooltip":""}]}},"input_order":{"required":["ckpt_name"]},"output":["MODEL","CLIP","VAE"],"output_is_list":[False,False,False],"output_name":["MODEL","CLIP","VAE"],"name":"CheckpointLoaderSimple","display_name":"Load","description":"","python_module":"nodes","category":"loaders","output_node":False,"output_tooltips":["","",""]},"CLIPTextEncode":{"input":{"required":{"text":["STRING",{"multiline":True,"dynamicPrompts":True,"tooltip":""}],"clip":["CLIP",{"tooltip":""}]}},"input_order":{"required":["text","clip"]},"output":["CONDITIONING"],"output_is_list":[False],"output_name":["CONDITIONING"],"name":"CLIPTextEncode","display_name":"CLIP","description":"","python_module":"nodes","category":"conditioning","output_node":False,"output_tooltips":[""]},"CLIPSetLastLayer":{"input":{"required":{"clip":["CLIP"],"stop_at_clip_layer":["INT",{"default":-1,"min":-24,"max":-1,"step":1}]}},"input_order":{"required":["clip","stop_at_clip_layer"]},"output":["CLIP"],"output_is_list":[False],"output_name":["CLIP"],"name":"CLIPSetLastLayer","display_name":"CLIPSLL","description":"","python_module":"nodes","category":"conditioning","output_node":False},"VAEDecode":{"input":{"required":{"samples":["LATENT",{"tooltip":""}],"vae":["VAE",{"tooltip":""}]}},"input_order":{"required":["samples","vae"]},"output":["IMAGE"],"output_is_list":[False],"output_name":["IMAGE"],"name":"VAEDecode","display_name":"VAE","description":"","python_module":"nodes","category":"latent","output_node":False,"output_tooltips":[""]},"VAEEncode":{"input":{"required":{"pixels":["IMAGE"],"vae":["VAE"]}},"input_order":{"required":["pixels","vae"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"VAEEncode","display_name":"VAE","description":"","python_module":"nodes","category":"latent","output_node":False},"VAEEncodeForInpaint":{"input":{"required":{"pixels":["IMAGE"],"vae":["VAE"],"mask":["MASK"],"grow_mask_by":["INT",{"default":6,"min":0,"max":64,"step":1}]}},"input_order":{"required":["pixels","vae","mask","grow_mask_by"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"VAEEncodeForInpaint","display_name":"VAE","description":"","python_module":"nodes","category":"latent/inpaint","output_node":False},"VAELoader":{"input":{"required":{"vae_name":[["kcpp_vae"]]}},"input_order":{"required":["vae_name"]},"output":["VAE"],"output_is_list":[False],"output_name":["VAE"],"name":"VAELoader","display_name":"Load VAE","description":"","python_module":"nodes","category":"loaders","output_node":False},"EmptyLatentImage":{"input":{"required":{"width":["INT",{"default":512,"min":16,"max":16384,"step":8,"tooltip":""}],"height":["INT",{"default":512,"min":16,"max":16384,"step":8,"tooltip":""}],"batch_size":["INT",{"default":1,"min":1,"max":1,"tooltip":""}]}},"input_order":{"required":["width","height","batch_size"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"EmptyLatentImage","display_name":"Empty Latent Image","description":"","python_module":"nodes","category":"latent","output_node":False,"output_tooltips":[""]}}).encode())
        elif clean_path.endswith('/api/models/checkpoints') or clean_path.endswith('/models/checkpoints'): #emulate comfyui, duplication is redundant but added for clarity
            if state.autoswapmode and state.imageName is not None:
                response_body = (json.dumps([state.imageName]).encode())
            elif state.friendlysdmodelname=="inactive" or state.fullsdmodelpath=="":
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([state.friendlysdmodelname]).encode())
        elif clean_path=='/api/models/loras' or clean_path=='/models/loras':
            response_body = (json.dumps([]).encode())
        elif clean_path=='/view' or clean_path=='/view.png' or clean_path=='/api/view' or clean_path.startswith('/view_image'): #emulate comfyui
            content_type = 'image/png'
            response_body = state.lastgeneratedcomfyimg
        elif clean_path=='/history' or clean_path=='/api/history' or clean_path.startswith('/api/history/') or clean_path.startswith('/history/'): #emulate comfyui
            modelNameToReturn = state.friendlysdmodelname
            if state.autoswapmode and state.imageName is not None:
                modelNameToReturn = state.imageName
            imgdone = (False if state.lastgeneratedcomfyimg==b'' else True)
            response_body = (json.dumps({"12345678-0000-0000-0000-000000000001":{"prompt":[0,"12345678-0000-0000-0000-000000000001",{"3":{"class_type":"KSampler","inputs":{"cfg":5.0,"denoise":1.0,"latent_image":["5",0],"model":["4",0],"negative":["7",0],"positive":["6",0],"sampler_name":"euler","scheduler":"normal","seed":1,"steps":20}},"4":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":modelNameToReturn}},"5":{"class_type":"EmptyLatentImage","inputs":{"batch_size":1,"height":512,"width":512}},"6":{"class_type":"CLIPTextEncode","inputs":{"clip":["4",1],"text":"prompt"}},"7":{"class_type":"CLIPTextEncode","inputs":{"clip":["4",1],"text":""}},"8":{"class_type":"VAEDecode","inputs":{"samples":["3",0],"vae":["4",2]}},"9":{"class_type":"SaveImage","inputs":{"filename_prefix":"kliteimg","images":["8",0]}}},{},["9"]],"outputs":{"9":{"images":[{"filename":"kliteimg_00001_.png","subfolder":"","type":"output"}]}},"status":{"status_str":"success","completed":imgdone,"messages":[["execution_start",{"prompt_id":"12345678-0000-0000-0000-000000000001","timestamp":1}],["execution_cached",{"nodes":[],"prompt_id":"12345678-0000-0000-0000-000000000001","timestamp":1}],["execution_success",{"prompt_id":"12345678-0000-0000-0000-000000000001","timestamp":1}]]},"meta":{"9":{"node_id":"9","display_node":"9","parent_node":None,"real_node_id":"9"}}}}).encode())
        elif clean_path=='/ws' and ('Upgrade' in self.headers and self.headers['Upgrade'].lower() == 'websocket' and
            'Sec-WebSocket-Key' in self.headers):
            ws_key = self.headers['Sec-WebSocket-Key']
            ws_accept = base64.b64encode(hashlib.sha1((ws_key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
            self.protocol_version = "HTTP/1.1"
            self.send_response(101) #fake websocket response, Switching Protocols
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.send_header('Sec-WebSocket-Accept', ws_accept)
            self.end_headers()
            try:
                # Send a dummy WebSocket text frame: empty string
                payload = json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": "ffff000012345678ffff000012345678"}}).encode("utf-8")
                header = struct.pack("!BB", 0x81, len(payload))  # FIN + text frame, no mask
                self.connection.sendall(header + payload)
                time.sleep(0.1) #short delay before replying
                # Send close frame with status code 1000 (Normal Closure)
                close_payload = struct.pack("!H", 1000)
                close_frame = struct.pack("!BB", 0x88, len(close_payload)) + close_payload
                self.connection.sendall(close_frame)
                time.sleep(0.1) #short delay before replying
            except Exception as e:
                print(f"WebSocket send error: {e}")
            self.connection.close()
            return
        elif clean_path.endswith(('/.well-known/serviceinfo')):
            response_body = (json.dumps({"version":"0.2","software":{"name":"KoboldCpp","version":state.KcppVersion,"repository":"https://github.com/LostRuins/koboldcpp","homepage":"https://github.com/LostRuins/koboldcpp","logo":"https://raw.githubusercontent.com/LostRuins/koboldcpp/refs/heads/concedo/niko.ico"},"api":{"koboldai":{"name":"KoboldAI API","rel_url":"/api","documentation":"https://lite.koboldai.net/koboldcpp_api","version":state.KcppVersion},"openai":{"name":"OpenAI API","rel_url ":"/v1","documentation":"https://openai.com/documentation/api","version":state.KcppVersion}}}).encode())

        elif clean_path=="/props":
            modelNameToReturn = state.friendlymodelname
            if state.autoswapmode and state.textName is not None:
                modelNameToReturn = state.textName
            mmprojOverride = False
            if state.autoswapmode and state.mmprojName is not None:
                mmprojOverride = True
            response_body = (json.dumps({
                "chat_template": state.cached_chat_template,
                "id": 0,
		        "id_task": -1,
                "total_slots": 1,
                "modalities": {
                    "vision": mmprojOverride or state.has_vision_support,
                    "audio": state.has_audio_support
                },
                "model_path": modelNameToReturn,
                "n_ctx": state.maxctx,
                "default_generation_settings": {
                    "n_ctx": state.maxctx,
                },
            }).encode())

        elif clean_path=="/slots":
            self.send_response(501)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"error":{"code":501,"message":"This server does not support slots endpoint.","type":"not_supported_error"}}).encode())
            return

        elif clean_path=="/api" or clean_path=="/docs":
            content_type = 'text/html'
            if supports_gzip and state.embedded_kcpp_docs_gz is not None:
                response_body = state.embedded_kcpp_docs_gz
                content_encoding = 'gzip'
            elif state.embedded_kcpp_docs is not None:
                response_body = state.embedded_kcpp_docs
            else:
                response_body = ("KoboldCpp API is running!\n\nAPI usage reference can be found at the wiki: https://github.com/LostRuins/koboldcpp/wiki").encode()

        elif clean_path=="/lcpp":
            content_type = 'text/html'
            if supports_gzip and state.embedded_lcpp_ui_gz is not None:
                response_body = state.embedded_lcpp_ui_gz
                content_encoding = 'gzip'
            else:
                response_body = ("Llama.cpp UI is not available. Please use the KoboldAI Lite UI instead.").encode()

        elif clean_path.startswith(("/sdui")):
            content_type = 'text/html'
            if supports_gzip and state.embedded_kcpp_sdui_gz is not None:
                response_body = state.embedded_kcpp_sdui_gz
                content_encoding = 'gzip'
            elif state.embedded_kcpp_sdui is not None:
                response_body = state.embedded_kcpp_sdui
            else:
                response_body = ("KoboldCpp API is running, but KCPP SDUI is not loaded").encode()

        elif clean_path.startswith(("/musicui")):
            content_type = 'text/html'
            if supports_gzip and state.embedded_musicui_gz is not None:
                response_body = state.embedded_musicui_gz
                content_encoding = 'gzip'
            elif state.embedded_musicui is not None:
                response_body = state.embedded_musicui
            else:
                response_body = ("KoboldCpp API is running, but KCPP MusicUI is not loaded").encode()

        elif clean_path=="/v1":
            content_type = 'text/html'
            response_body = ("KoboldCpp OpenAI compatible endpoint is running!<br>For usage reference, see <a href='https://platform.openai.com/docs/api-reference'>https://platform.openai.com/docs/api-reference</a><br>For other endpoints, see <a href='/api'>KoboldCpp API Documentation</a>").encode()

        elif clean_path=="/api/extra/preloadstory":
            if state.preloaded_story is None:
                response_body = (json.dumps({}).encode())
            else:
                response_body = state.preloaded_story
        elif clean_path.endswith(('/api')) or clean_path.endswith(('/api/v1')):
            self.send_response(302)
            self.send_header("location", "/api")
            self.end_headers(content_type='text/html')
            return None

        if response_body is None:
            self.send_response(404)
            self.end_headers(content_type='text/html')
            rp = f"Error: KoboldCpp HTTP Server is running, but this endpoint does not exist. Please check the URL and METHOD.<br><a href=\"/api\">[Read API documentation here]</a><br><br>Current path: {self.path}"
            self.wfile.write(rp.encode())
        else:
            self.send_response(200)
            self.send_header('content-length', str(len(response_body)))
            if content_encoding:
                self.send_header('Content-Encoding', content_encoding)
            self.end_headers(content_type=content_type)
            self.wfile.write(response_body)
        return

    def do_POST(self):
        contlenstr = self.headers['content-length']
        content_length = 0
        body = None
        if contlenstr:
            content_length = int(contlenstr)
            max_pl = int(state.args.maxrequestsize) if state.args.maxrequestsize else 32
            if content_length > (1024*1024*max_pl): #payload size limit
                self.send_response(500)
                self.end_headers(content_type='application/json')
                self.wfile.write(json.dumps({"detail": {
                "msg": f"Payload is too big. Max payload size is {max_pl}MB.",
                "type": "bad_input",
                }}).encode())
                return
            body = self.rfile.read(content_length)
        elif self.headers.get('transfer-encoding', '').lower()=="chunked":
            content_length = 0
            chunklimit = 0  # do not process more than 512 chunks, prevents bad actors
            body = b''
            try:
                while True:
                    chunklimit += 1
                    line = self.rfile.readline().strip()
                    if line:
                        chunk_length = max(0,int(line, 16))
                        content_length += chunk_length
                    if not line or chunklimit > 512 or content_length > (1024*1024*48): #48mb payload limit
                        self.send_response(500)
                        self.end_headers(content_type='application/json')
                        self.wfile.write(json.dumps({"detail": {
                        "msg": "Payload is too big. Max payload size is 48MB.",
                        "type": "bad_input",
                        }}).encode())
                        return
                    if chunk_length != 0:
                        chunk = self.rfile.read(chunk_length)
                        body += chunk
                    self.rfile.readline()
                    if chunk_length == 0:
                        break
            except Exception:
                self.send_response(500)
                self.end_headers(content_type='application/json')
                self.wfile.write(json.dumps({"detail": {
                "msg": "Failed to parse chunked request.",
                "type": "bad_input",
                }}).encode())
                return

        self.path = self.path.rstrip('/')
        response_body = None
        response_code = 200

        if self.path.endswith('/api/extra/tokencount') or self.path.endswith('/api/extra/tokenize'):
            if not self.secure_endpoint():
                return
            try:
                genparams = json.loads(body)
                countprompt = genparams.get('prompt', "")
                tcaddspecial = genparams.get('special', True)
                msgs = genparams.get('messages',[])
                if msgs and len(msgs) > 0 and not countprompt:
                    transform_genparams(genparams,4,state.args.jinja)
                    countprompt = genparams.get('prompt', "")
                countdata = tokenize_ids(countprompt,tcaddspecial)
                response_body = (json.dumps({"value": len(countdata),"ids": countdata, "prompt":countprompt}).encode())

            except Exception as e:
                utfprint("Count Tokens - Body Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"value": -1}).encode())

        elif self.path.endswith('/api/extra/detokenize'):
            if not self.secure_endpoint():
                return
            try:
                genparams = json.loads(body)
                tokids = genparams.get('ids', [])
                addspecial = genparams.get('special', True)
                detokstr = detokenize_ids(tokids,addspecial)
                response_body = (json.dumps({"result": detokstr,"success":True}).encode())
            except Exception as e:
                utfprint("Detokenize Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"result": "","success":False}).encode())

        elif self.path.endswith('/api/extra/json_to_grammar'):
            if not self.secure_endpoint():
                return
            try:
                genparams = json.loads(body)
                schema = genparams.get('schema', None)
                if not schema:
                    schema = genparams
                decoded = convert_json_to_gbnf(schema)
                response_body = (json.dumps({"result": decoded,"success":(True if decoded else False)}).encode())
            except Exception as e:
                utfprint("JSON to Grammar Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"result": "","success":False}).encode())

        elif self.path.endswith('/api/extra/abort'):
            if not self.secure_endpoint():
                return
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception:
                multiuserkey = ""
                pass
            if (multiuserkey=="" and state.requestsinqueue==0) or (multiuserkey!="" and multiuserkey==state.currentusergenkey):
                ag = state.handle.abort_generate()
                time.sleep(0.1) #short delay before replying
                response_body = (json.dumps({"success": ("true" if ag else "false"), "done":"true"}).encode())
                print("\nGeneration Aborted")
            elif (multiuserkey!="" and state.requestsinqueue>0):
                pendingabortkey = multiuserkey
                response_body = (json.dumps({"success": "true", "done":"false"}).encode())
            else:
                response_body = (json.dumps({"success": "false", "done":"false"}).encode())

        elif self.path.endswith('/api/extra/generate/check'):
            if not self.secure_endpoint():
                return
            pendtxtStr = ""
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception:
                multiuserkey = ""

            if state.totalgens>0:
                if (multiuserkey=="" and multiuserkey==state.currentusergenkey and state.requestsinqueue==0) or (multiuserkey!="" and multiuserkey==state.currentusergenkey): #avoid leaking prompts in multiuser
                    pendtxt = state.handle.get_pending_output()
                    pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            response_body = (json.dumps({"results": [{"text": pendtxtStr}]}).encode())

        elif self.path.endswith('/api/extra/last_logprobs'):
            if not self.secure_endpoint():
                return
            logprobsdict = None
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception:
                multiuserkey = ""

            if state.totalgens>0:
                if (multiuserkey=="" and multiuserkey==state.currentusergenkey and state.requestsinqueue==0) or (multiuserkey!="" and multiuserkey==state.currentusergenkey): #avoid leaking prompts in multiuser
                    lastlogprobs = state.handle.last_logprobs()
                    logprobsdict = parse_last_logprobs(lastlogprobs)
            response_body = (json.dumps({"logprobs":logprobsdict}).encode())

        elif self.path.endswith('/api/extra/multiplayer/status'):
            if not self.secure_endpoint():
                return
            if not state.has_multiplayer:
                response_body = (json.dumps({"error":"Multiplayer not enabled!"}).encode())
            else:
                sender = ""
                senderbusy = False
                try:
                    tempbody = json.loads(body)
                    if isinstance(tempbody, dict):
                        sender = tempbody.get('sender', "")
                        senderbusy = tempbody.get('senderbusy', False)
                except Exception:
                    pass
                if sender!="" and senderbusy:
                    state.multiplayer_lastactive[sender] = int(time.time())
                response_body = (json.dumps({"turn_major":state.multiplayer_turn_major,"turn_minor":state.multiplayer_turn_minor,"idle":self.get_multiplayer_idle_state(sender),"data_format":state.multiplayer_dataformat}).encode())

        elif self.path.endswith('/api/extra/data/list'):
            if not self.secure_endpoint():
                return
            if state.savedata_obj is None:
                response_body = (json.dumps([]).encode())
                return
            output = []
            for i in range (state.net_save_slots):
                if str(i) in state.savedata_obj:
                    output.append(state.savedata_obj[str(i)]["title"])
                else:
                    output.append("")
            response_body = (json.dumps(output).encode())

        elif self.path.endswith('/api/extra/data/load'):
            if not self.secure_endpoint():
                return
            if state.savedata_obj is None:
                response_body = (json.dumps({"success":False,"data":None}).encode())
            loadid = -1
            try:
                tempbody = json.loads(body)
                loadid = tryparseint(tempbody.get('slot', 0),0)
            except Exception:
                loadid = -1
            if loadid < 0 or str(loadid) not in state.savedata_obj:
                response_body = (json.dumps({"success":False,"data":None}).encode())
            else:
                response_body = (json.dumps({"success":True,"data":state.savedata_obj[str(loadid)]}).encode())

        elif self.path.endswith('/api/extra/data/save'):
            if not self.secure_endpoint():
                return
            if state.savedata_obj is None:
                response_code = 400
                response_body = (json.dumps({"success":False, "error":"SaveDataFile not enabled!"}).encode())
            else:
                try:
                    incoming_story = json.loads(body) # ensure submitted data is valid json
                    slotid = tryparseint(incoming_story.get('slot', -1),-1)
                    dataformat = incoming_story.get('format', "")
                    title = incoming_story.get('title', "")
                    if not title or title=="":
                        title = "Untitled Save"
                    storybody = incoming_story.get('data', None) #should be a compressed string
                    if slotid >= 0 and slotid < state.net_save_slots:  # we shall provide some fixed network save slots
                        saveneeded = False
                        if storybody and storybody!="":
                            storybody = str(storybody)
                            if len(storybody) > (1024*1024*10): #limit each story to 10mb
                                response_code = 400
                                response_body = (json.dumps({"success":False, "error":"Story is too long!"}).encode())
                            else:
                                state.savedata_obj[str(slotid)] = {"title":title, "format":dataformat, "data":storybody}
                                saveneeded = True
                        else: #erasing existing story
                            if str(slotid) in state.savedata_obj:
                                state.savedata_obj.pop(str(slotid))
                                saveneeded = True
                        if saveneeded:
                            if state.args.savedatafile and os.path.exists(os.path.abspath(state.args.savedatafile)):
                                with open(os.path.abspath(state.args.savedatafile), 'w+', encoding='utf-8', errors='ignore') as f:
                                    json.dump(state.savedata_obj, f)
                                    print(f"Data was saved to slot {slotid}")
                                response_body = (json.dumps({"success":True, "error":""}).encode())
                            else:
                                response_code = 400
                                response_body = (json.dumps({"success":False, "error":"SaveDataFile is missing!"}).encode())
                        else:
                            response_body = (json.dumps({"success":True, "error":""}).encode())
                    else:
                        response_code = 400
                        response_body = (json.dumps({"success":False, "error":"No story submitted or invalid slot!"}).encode())
                except Exception as e:
                    utfprint("Remote Save Story - Body Error: " + str(e))
                    response_code = 400
                    response_body = (json.dumps({"success": False, "error":"Submitted story invalid!"}).encode())

        elif self.path.endswith('/api/extra/multiplayer/getstory'):
            if not self.secure_endpoint():
                return
            if not state.has_multiplayer:
                response_body = ("".encode())
            elif state.multiplayer_story_data_compressed is None:
                response_body = ("".encode())
            else:
                response_body = state.multiplayer_story_data_compressed.encode()

        elif self.path.endswith('/api/extra/multiplayer/setstory'):
            if not self.secure_endpoint():
                return
            if not state.has_multiplayer:
                response_code = 400
                response_body = (json.dumps({"success":False, "error":"Multiplayer not enabled!"}).encode())
            else:
                try:
                    incoming_story = json.loads(body) # ensure submitted data is valid json
                    fullupdate = incoming_story.get('full_update', False)
                    dataformat = incoming_story.get('data_format', "")
                    sender = incoming_story.get('sender', "")
                    storybody = incoming_story.get('data', None) #should be a compressed string
                    if storybody:
                        storybody = str(storybody)
                        if len(storybody) > (1024*1024*3): #limit story to 3mb
                            response_code = 400
                            response_body = (json.dumps({"success":False, "error":"Story is too long!"}).encode())
                        else:
                            multiplayer_story_data_compressed = str(storybody) #save latest story
                            multiplayer_dataformat = dataformat
                            if sender!="":
                                state.multiplayer_lastactive[sender] = int(time.time())
                            if fullupdate:
                                multiplayer_turn_minor = 1
                                state.multiplayer_turn_major += 1
                            else:
                                state.multiplayer_turn_minor += 1
                            response_body = (json.dumps({"success":True,"turn_major":state.multiplayer_turn_major,"turn_minor":state.multiplayer_turn_minor,"idle":self.get_multiplayer_idle_state(sender),"data_format":state.multiplayer_dataformat}).encode())
                    else:
                        response_code = 400
                        response_body = (json.dumps({"success":False, "error":"No story submitted!"}).encode())
                except Exception as e:
                    utfprint("Multiplayer Set Story - Body Error: " + str(e))
                    response_code = 400
                    response_body = (json.dumps({"success": False, "error":"Submitted story invalid!"}).encode())

        elif self.path.startswith(("/api/extra/websearch")):
            if not self.secure_endpoint():
                return
            if state.args.websearch:
                try:
                    tempbody = json.loads(body)
                    searchstr = tempbody.get('q', "")
                    searchres = websearch(searchstr)
                    response_body = (json.dumps(searchres).encode())
                except Exception as e:
                    utfprint("WebSearch Parse Error: " + str(e))
                    response_code = 400
                    response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([]).encode())

        elif self.path.startswith(("/api/admin/reload_config")):
            resp = {"success": False}
            if state.global_memory and state.args.admin and state.args.admindir and os.path.exists(state.args.admindir) and self.check_header_password(state.args.adminpassword):
                targetfile = ""
                baseconfig = ""
                try:
                    tempbody = json.loads(body)
                    if isinstance(tempbody, dict):
                        targetfile = tempbody.get('filename', "")
                        baseconfig = tempbody.get('baseconfig', tempbody.get('overrideconfig', ""))
                except Exception:
                    targetfile = ""
                if targetfile and targetfile!="":
                    if targetfile=="unload_model" or targetfile=="initial_model": #special request to simply unload model or swap back top intial model
                        print("Admin: Received request to unload model")
                        state.global_memory["restart_target"] = targetfile
                        state.global_memory["restart_override_base_config"] = ""
                        resp = {"success": True}
                    else:
                        dirpath = os.path.abspath(state.args.admindir)
                        allowed_files = get_current_admindir_list()
                        # Normalize requested target path
                        targetfilepath = os.path.abspath(os.path.join(dirpath, targetfile))

                        if (targetfile in allowed_files and os.path.commonpath([dirpath, targetfilepath]) == dirpath and os.path.exists(targetfilepath)):
                            state.global_memory["restart_override_base_config"] = "" # Jail enforcement
                            if targetfile and baseconfig:
                                baseconfigfilepath = os.path.abspath(os.path.join(dirpath, baseconfig))
                                if (baseconfig in allowed_files and os.path.commonpath([dirpath, baseconfigfilepath]) == dirpath and os.path.exists(baseconfigfilepath)):
                                    print(f"Admin: Override base config set to {baseconfig}")
                                    state.global_memory["restart_override_base_config"] = baseconfig
                            print(f"Admin: Received request to reload config to {targetfile}")
                            state.global_memory["restart_target"] = targetfile
                            resp = {"success": True}
            response_body = (json.dumps(resp).encode())

        elif self.path.endswith('/set_tts_settings'): #return dummy response
            response_body = (json.dumps({"message": "Settings successfully applied"}).encode())

        elif self.path=="/api/show": #ollama compatible
            response_body = (json.dumps({"parameters":"temperature 1.0","license":"Ollama Emulation. Running on KoboldCpp","modelfile":"KoboldCpp","capabilities":["completion"],"modified_at":"2025-01-01T01:00:00.0000000+00:00","details":{},"model_info":{}}).encode())

        elif self.path=="/mcp": #simple mcp proxy
            if not self.secure_endpoint():
                return
            try:
                tempbody = json.loads(body)
                method = tempbody.get("method","")
                if method == "initialize":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "mcp-koboldcpp", "version": "1.0.0"},
                        },
                    }
                    response_body = (json.dumps(reply).encode())
                elif method == "tools/list":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "result": {"tools": []},
                    }
                    with state.mcp_lock:
                        for conn in state.mcp_connections:
                            currtools = conn["tools"]
                            for tool in currtools:
                                reply["result"]["tools"].append(tool)
                    response_body = (json.dumps(reply).encode())
                elif method == "tools/call":
                    foundtool = False
                    callparams = tempbody.get("params",{})
                    callname = callparams.get("name","")
                    with state.mcp_lock:
                        for conn in state.mcp_connections:
                            currtools = conn["tools"]
                            currclient = conn["client"]
                            for tool in currtools:
                                if currclient and tool.get("name","")!="" and tool.get("name","")==callname:
                                    foundtool = True
                                    mcpresp = currclient.send(tempbody)
                                    response_body = (json.dumps(mcpresp).encode())
                                    break
                    if not foundtool:
                        response_code = 400
                        response_body = (json.dumps({"error": {"code": -32700, "message": "Tool not found"}}).encode())
                else: #probably a notify, send empty response
                    response_body = (json.dumps({}).encode())
            except Exception as e:
                print(f"MCP Call Error: {e}")
                response_code = 400
                response_body = (json.dumps({"error": {"code": -32700, "message": "Parse error"}}).encode())

        elif self.path=="/api/extra/shutdown":
            # if args.singleinstance:
            client_ip = self.client_address[0]
            is_local = client_ip in ('127.0.0.1', '::1', 'localhost')
            if is_local and state.args.singleinstance:
                response_body = (json.dumps({"success": True}).encode())
                self.send_response(response_code)
                self.send_header('content-length', str(len(response_body)))
                self.end_headers(content_type='application/json')
                self.wfile.write(response_body)
                print("\nReceived Shutdown Command! Shutting down...\n")
                time.sleep(1)
                exitcounter = 999
                sys.exit(0)
                return
            else:
                response_body = (json.dumps({"success": False}).encode())

        if response_body is not None:
            self.send_response(response_code)
            self.send_header('content-length', str(len(response_body)))
            self.end_headers(content_type='application/json')
            self.wfile.write(response_body)
            return

        reqblocking = False
        #handle rate limiting
        ratelimiter = int(state.args.ratelimit)
        if ratelimiter > 0:
            client_ip = self.client_address[0]
            lastdone = state.ratelimitlookup.get(client_ip, datetime.min)
            diff = (datetime.now() - lastdone).total_seconds()
            if diff < ratelimiter:
                self.send_response(503)
                self.end_headers(content_type='application/json')
                self.wfile.write(json.dumps({"detail": {
                        "msg": f"You are sending requests too quickly. Please try again in {int(ratelimiter-diff)} seconds.",
                        "type": "service_unavailable",
                    }}).encode())
                return
            state.ratelimitlookup[client_ip] = datetime.now()
        muint = int(state.args.multiuser)
        if muint<=0 and ((state.args.whispermodel and state.args.whispermodel!="") or (state.args.sdmodel and state.args.sdmodel!="") or (state.args.ttsmodel and state.args.ttsmodel!="") or (state.args.embeddingsmodel and state.args.embeddingsmodel!="")):
            muint = 2 # this prevents errors when using voice/img together with text
        if getattr(state.args, "continuous_batching", 0) > 1 and muint <=0:
            muint = state.multiuser_concurrent_limit # multiuser required for batching
        multiuserlimit = ((muint-1) if muint > 1 else state.multiuser_concurrent_limit)
        #backwards compatibility for up to X concurrent requests, use default limit of X if multiuser set to 1
        if muint > 0 and state.requestsinqueue < multiuserlimit:
            reqblocking = True
            state.requestsinqueue += 1
        if not state.modelbusy.acquire(blocking=reqblocking):
            self.send_response(503)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"detail": {
                    "msg": "Server is busy; please try again later.",
                    "type": "service_unavailable",
                }}).encode())
            return
        is_batchable_req = False
        if reqblocking:
            requestsinqueue = (state.requestsinqueue - 1) if state.requestsinqueue > 0 else 0

        # handle endpoints that require mutex locking and handle actual gens
        try:
            sse_stream_flag = False
            api_format = 0 #1=basic,2=kai,3=oai,4=oai-chat,5=interrogate,6=ollama,7=ollamachat,8=oai-responses,9=anthropic-messages
            is_imggen = False
            is_comfyui_imggen = False
            is_oai_imggen = False
            is_img_upscale = False
            is_transcribe = False
            is_tts = False
            is_embeddings = False
            is_music_codes = False
            is_music_audio = False
            response_body = None
            use_jinja = state.args.jinja
            state.global_memory["last_active_timestamp"] = datetime.now()
            state.global_memory["triggered_sleeping"] = False
            if self.path.endswith('/api/admin/check_state'):
                if state.global_memory and state.args.admin and state.args.admindir and os.path.exists(state.args.admindir) and self.check_header_password(state.args.adminpassword):
                    cur_states = []
                    for sl in range(state.savestate_limit): #0,1,2,3
                        oldstate = state.handle.calc_old_state_kv(sl)
                        oldtokencnt = state.handle.calc_old_state_tokencount(sl)
                        cur_states.append({"tokens":oldtokencnt,"size":oldstate})
                    newstate = state.handle.calc_new_state_kv()
                    newtokencnt = state.handle.calc_new_state_tokencount()
                    response_body = (json.dumps({"success": True, "old_states":cur_states, "new_state_size":newstate, "new_tokens":newtokencnt}).encode())
                else:
                    response_body = (json.dumps({"success": False, "old_states":[], "new_state_size":0, "new_tokens":0}).encode())
            elif self.path.endswith('/api/admin/load_state'):
                if state.global_memory and state.savestate_limit>0 and state.args.admin and state.args.admindir and os.path.exists(state.args.admindir) and self.check_header_password(state.args.adminpassword):
                    targetslot = 0
                    try:
                        tempbody = json.loads(body)
                        if isinstance(tempbody, dict):
                            targetslot = tempbody.get('slot', 0)
                    except Exception:
                        pass
                    targetslot = (targetslot if targetslot<state.savestate_limit else 0)
                    result = state.handle.load_state_kv(targetslot)
                    tokencnt = state.handle.calc_new_state_tokencount()
                    response_body = (json.dumps({"success": result, "new_tokens":tokencnt}).encode())
                else:
                    response_body = (json.dumps({"success": False, "new_tokens":0}).encode())
            elif self.path.endswith('/api/admin/save_state'):
                if state.global_memory and state.savestate_limit>0 and state.args.admin and state.args.admindir and os.path.exists(state.args.admindir) and self.check_header_password(state.args.adminpassword):
                    targetslot = 0
                    try:
                        tempbody = json.loads(body)
                        if isinstance(tempbody, dict):
                            targetslot = tempbody.get('slot', 0)
                    except Exception:
                        pass
                    targetslot = (targetslot if targetslot<state.savestate_limit else 0)
                    result = state.handle.save_state_kv(targetslot)
                    tokencnt = state.handle.calc_new_state_tokencount()
                    response_body = (json.dumps({"success": (result>0), "new_state_size":result, "new_tokens":tokencnt}).encode())
                else:
                    response_body = (json.dumps({"success": False, "new_state_size":0, "new_tokens":0}).encode())
            elif self.path.endswith('/api/admin/clear_state'):
                if state.global_memory and state.savestate_limit>0 and state.args.admin and state.args.admindir and os.path.exists(state.args.admindir) and self.check_header_password(state.args.adminpassword):
                    result = state.handle.clear_state_kv()
                    response_body = (json.dumps({"success": result}).encode())
                else:
                    response_body = (json.dumps({"success": False}).encode())
            elif self.path.startswith('/api/upload/image') or self.path.startswith("/upload/image"): #comfyui compatible
                lastuploadedcomfyimg = b''
                formdata = self.extract_formdata_from_file_upload(body)
                if "file" in formdata and formdata["file"]:
                    lastuploadedcomfyimg = formdata["file"]
                response_body = (json.dumps({"name": "kcpp_img2img.jpg", "subfolder": "", "type": "input"}).encode())
            elif self.path.endswith('/request'):
                api_format = 1
            elif self.path.endswith(('/api/v1/generate', '/api/latest/generate')):
                api_format = 2
            elif self.path.endswith('/api/extra/generate/stream'):
                api_format = 2
                sse_stream_flag = True
            elif self.path.endswith('/v1/completions') or self.path.endswith('/v1/completion') or self.path=='/completions':
                api_format = 3
            elif self.path.endswith('/v1/chat/completions') or self.path=='/chat/completions':
                api_format = 4
            elif self.path.endswith('/sdapi/v1/interrogate'):
                mmprojOverride = False
                if state.autoswapmode and state.mmprojName is not None:
                    mmprojOverride = True
                if not mmprojOverride and not state.has_vision_support:
                    self.send_response(503)
                    self.end_headers(content_type='application/json')
                    self.wfile.write(json.dumps({"detail": {
                            "msg": "No Vision model loaded",
                            "type": "service_unavailable",
                        }}).encode())
                    return
                api_format = 5
            elif self.path.endswith('/api/generate'): #ollama
                api_format = 6
            elif self.path.endswith('/api/chat'): #ollama
                api_format = 7
            elif self.path.endswith('/v1/responses') or self.path=='/responses': #oai-responses
                api_format = 8
            elif self.path.endswith('/v1/messages') or self.path=='/messages': #anthropic
                api_format = 9
            elif self.path.endswith('/sdapi/v1/extra-single-image') or self.path.endswith('/sdapi/v1/upscale'):
                is_img_upscale = True
            elif self.path=="/prompt" or self.path=="/images/generations" or self.path.endswith('/v1/images/generations') or self.path.endswith('/sdapi/v1/txt2img') or self.path.endswith('/sdapi/v1/img2img'):
                is_imggen = True
                if self.path=="/prompt":
                    is_comfyui_imggen = True
                elif self.path.endswith('/v1/images/generations') or self.path=="/images/generations":
                    is_oai_imggen = True
            elif self.path.endswith('/api/extra/transcribe') or self.path.endswith('/v1/audio/transcriptions') or self.path=="/audio/transcriptions":
                is_transcribe = True
            elif self.path.endswith('/api/extra/tts') or self.path.endswith('/v1/audio/speech') or self.path=="/audio/speech" or self.path.endswith('/tts_to_audio'):
                is_tts = True
            elif self.path.endswith('/api/extra/embeddings') or self.path.endswith('/v1/embeddings'):
                is_embeddings = True
            elif self.path.endswith('/api/extra/music/prepare'):
                is_music_codes = True
            elif self.path.endswith('/api/extra/music/generate'):
                is_music_audio = True

            if response_body is not None:
                self.send_response(response_code)
                self.send_header('content-length', str(len(response_body)))
                self.end_headers(content_type='application/json')
                self.wfile.write(response_body)
            elif is_imggen or is_img_upscale or is_transcribe or is_tts or is_embeddings or is_music_codes or is_music_audio or api_format > 0:
                last_req_time = time.time()

                if not is_imggen and not is_img_upscale and not self.path.endswith('/tts_to_audio') and api_format!=5:
                    if not self.secure_endpoint():
                        return

                genparams = None
                try:
                    genparams = json.loads(body)
                except Exception:
                    genparams = None
                    if is_transcribe: #fallback handling of file uploads
                        formdata = self.extract_formdata_from_file_upload(body)
                        if "file" in formdata and formdata["file"]:
                            b64wav = formdata["file"]
                            genparams = {"audio_data":b64wav}
                            if "prompt" in formdata and formdata["prompt"]:
                                genparams["prompt"] = formdata["prompt"]
                            if "language" in formdata and formdata["language"]:
                                genparams["language"] = formdata["language"]

                    if not genparams:
                        utfprint("Body Err: " + str(body))
                        self.send_response(500)
                        self.end_headers(content_type='application/json')
                        self.wfile.write(json.dumps({"detail": {
                        "msg": "Error parsing input.",
                        "type": "bad_input",
                        }}).encode())
                        return

                gendefaults = gendefaults_parse_meta_field(state.args.gendefaults or '')
                gen_new_keys = {k: v for k, v in gendefaults.items() if k not in genparams}
                #special handling for some params that should be overwritten if equal to literal string default
                special_fields = ["sampler_name", "scheduler"]
                for field in special_fields:
                    if field in genparams and isinstance(genparams[field], str):
                        genparams[field] = genparams[field].lower()
                special_fields_overwrite = {}
                if not state.args.gendefaultsoverwrite:
                    for field in special_fields:
                        if genparams.get(field, "default") == "default" and field in gendefaults:
                            value = gendefaults.get(field, "default")
                            if isinstance(value, str):
                                value = value.lower()
                            special_fields_overwrite[field] = value
                genparams.update(gendefaults if state.args.gendefaultsoverwrite else gen_new_keys)
                genparams.update(special_fields_overwrite)

                trunc_len = 10000
                if state.args.debugmode >= 1:
                    trunc_len = 40000

                if use_jinja and not state.args.jinja_tools:
                    tmptools = genparams.get('tools', [])
                    if tmptools and len(tmptools) > 0:
                        use_jinja = False # not allowed to use tools with jinja

                # payload modifications for lcpp endpoint. we detect this by the timings_per_token field existing
                if "timings_per_token" in genparams:
                    genparams["continue_assistant_turn"] = True
                    genparams["encapsulate_thinking"] = True

                printablegenparams_raw = truncate_long_json(genparams,trunc_len)
                utfprint("\nInput: " + json.dumps(printablegenparams_raw,ensure_ascii=False),1)

                # transform genparams (only used for text gen) first
                genparams = transform_genparams(genparams, api_format, use_jinja)

                if state.args.debugmode >= 1:
                    printablegenparams = truncate_long_json(genparams,trunc_len)
                    utfprint("\nAdapted Input: " + json.dumps(printablegenparams),1)

                if state.args.foreground:
                    bring_terminal_to_foreground()

                #if it's a non-batchable request and we already have batching ongoing, stall this request
                if state.batched_request_runner_count > 0 and not continuous_batching_python_eligible(genparams, api_format):
                    with state.batched_cond:
                        while state.batched_request_runner_count > 0:
                            state.batched_cond.wait()

                if api_format > 0: #text gen
                    # Check if streaming chat completions, if so, set stream mode to true
                    if (api_format == 4 or api_format == 3 or api_format == 8 or api_format == 9) and "stream" in genparams and genparams["stream"]:
                        sse_stream_flag = True
                    if continuous_batching_python_eligible(genparams, api_format):
                        genparams['_batch_expected'] = True
                        state.modelbusy.release()
                        is_batchable_req = True
                        with state.batched_cond:
                            state.batched_request_runner_count += 1

                    gendat = asyncio.run(self.handle_request(genparams, api_format, sse_stream_flag))

                    try:
                        modelNameToReturn = state.friendlymodelname
                        if state.autoswapmode and state.textName is not None:
                            modelNameToReturn = state.textName
                        # Headers are already sent when streaming
                        if (api_format == 6 or api_format == 7) and genparams.get('stream', True):
                            #ollama fake streaming
                            self.send_response(200)
                            self.send_header("X-Accel-Buffering", "no")
                            self.send_header("cache-control", "no-cache")
                            self.send_header("connection", "keep-alive")
                            self.end_headers(content_type='text/event-stream')
                            if api_format == 6:
                                bodytxt = gendat.get("response","") # extract and erase the AI response from the sync payload.
                                gendat["response"] = ""
                                pl = {"model":modelNameToReturn,"created_at":str(datetime.now(timezone.utc).isoformat()),"response":bodytxt,"done":False}
                                self.wfile.write(f'{json.dumps(pl)}\n'.encode())
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                                self.wfile.write(f'{json.dumps(gendat)}\n'.encode()) # note: gendat already contains done=true and empty response
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                            else:
                                bodytxt = gendat.get("message",{}).get("content","") # extract and erase the AI response from the sync payload.
                                gendat["message"] = {"role":"assistant","content":""}
                                pl = {"model":modelNameToReturn,"created_at":str(datetime.now(timezone.utc).isoformat()),"message":{"role":"assistant","content":bodytxt},"done":False}
                                self.wfile.write(f'{json.dumps(pl)}\n'.encode())
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                                self.wfile.write(f'{json.dumps(gendat)}\n'.encode()) # note: gendat already contains done=true and empty response
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                            self.close_connection = True
                        elif not sse_stream_flag:
                            self.send_response(200)
                            genresp = (json.dumps(gendat).encode())
                            self.send_header('content-length', str(len(genresp)))
                            self.end_headers(content_type='application/json')
                            self.wfile.write(genresp)
                        elif api_format == 4 and genparams.get('using_openai_tools', False): #special case, fake streaming for openai tool calls
                            # we only send content_text and reasoning_text if tools aren't used. they contain the balance of the output after sync_toolcall_potential_triggered was triggered
                            content_text = genparams.get('sync_toolcall_extra_content', "") #populated by the sse call, we don't use gendat['choices'][0]['message'].get('content', None)
                            reasoning_text = genparams.get('sync_toolcall_extra_reasoning_content', "")
                            toolsdata_res = []
                            try:
                                toolsdata_res = gendat['choices'][0]['message']['tool_calls']
                                if toolsdata_res and len(toolsdata_res)>0:
                                    toolsdata_res[0]["index"] = 0 # need to add an index for OWUI
                            except Exception:
                                toolsdata_res = []

                           # Send role chunk first, if needed
                            if genparams.get('sync_toolcall_first_role_sent', False):
                                genparams['sync_toolcall_first_role_sent'] = True
                                chunk_role = json.dumps({
                                    "id": "koboldcpp",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": modelNameToReturn,
                                    "choices": [{"index": 0, "finish_reason": None, "delta": {"role": "assistant"}}]
                                })
                                self.wfile.write(f"data: {chunk_role}\n\n".encode())
                                self.wfile.flush()

                            # if no valid tool splitter, we have to do 100% synchronous
                            if not content_text and not reasoning_text and genparams.get('sync_toolcall_stream_ineligible', False):
                                temp_content = ""
                                temp_reasoning = ""
                                try:
                                    temp_content = gendat['choices'][0]['message'].get('content', None)
                                except Exception:
                                    temp_content = None
                                try:
                                    temp_reasoning = gendat['choices'][0]['message'].get('reasoning_content', None)
                                except Exception:
                                    temp_reasoning = None
                                if temp_content and not temp_reasoning: #fix incorrect reasoning sent as content
                                    thinkstrips = [item["start"] for item in state.thinkformats] #start thinking tags
                                    thinksplitters = [item["end"] for item in state.thinkformats] #end thinking tags
                                    for tsp in thinksplitters:
                                        if tsp in temp_content:
                                            parts = temp_content.split(tsp, 1)
                                            temp_reasoning = parts[0]
                                            temp_content = parts[1]
                                            for ts in thinkstrips:
                                                temp_reasoning = temp_reasoning.replace(ts, "")

                                if temp_reasoning:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": temp_reasoning}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()
                                if temp_content:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"content": temp_content}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()

                            # Send tool calls incrementally in OpenAI format
                            if toolsdata_res and len(toolsdata_res) > 0:
                                for idx, tool_call in enumerate(toolsdata_res):
                                    tc_meta = {
                                        "index": idx,
                                        "id": tool_call.get("id", f"call_{idx}"),
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.get("function", {}).get("name", ""),
                                            "arguments": ""
                                        }
                                    }
                                    chunk_meta = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [tc_meta]}}]
                                    })
                                    self.wfile.write(f"data: {chunk_meta}\n\n".encode())
                                    self.wfile.flush()

                                    args_str = tool_call.get("function", {}).get("arguments", "{}")
                                    if isinstance(args_str, dict):
                                        args_str = json.dumps(args_str)
                                    tc_args = {
                                        "index": idx,
                                        "function": {"arguments": args_str}
                                    }
                                    chunk_args = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [tc_args]}}]
                                    })
                                    self.wfile.write(f"data: {chunk_args}\n\n".encode())
                                    self.wfile.flush()
                            else:
                                # Send remaining buffered content if no tool calls were made
                                if reasoning_text:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": reasoning_text}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()
                                if content_text:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"content": content_text}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()

                            # Final chunk
                            chunk_final = json.dumps({
                                "id": "koboldcpp",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": modelNameToReturn,
                                "choices": [{"index": 0, "finish_reason": "tool_calls" if (len(toolsdata_res) > 0) else state.currfinishreason, "delta": {}}]
                            })
                            self.wfile.write(f"data: {chunk_final}\n\n".encode())
                            self.wfile.write("data: [DONE]\n\n".encode())
                            self.wfile.flush()
                            self.close_connection = True
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Generate: The response could not be sent, maybe connection was terminated?")
                        state.handle.abort_generate()
                        time.sleep(0.2) #short delay
                    return

                elif is_img_upscale: #esrgan upscale
                    try:
                        gen = sd_upscale(genparams)
                        genresp = (json.dumps({"html_info":"<p>Postprocess upscale by: 2.0, Postprocess upscaler: ESRGAN_4x</p>","image":gen}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Upscale Image: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_imggen: #image gen
                    try:
                        if is_comfyui_imggen:
                            lastgeneratedcomfyimg = b''
                            genparams = sd_comfyui_tranform_params(genparams)
                        elif is_oai_imggen:
                            genparams = sd_oai_transform_params(genparams)
                        if not genparams.get('lora'):
                            # process <lora:name:multiplier> syntax
                            prompt, loras = extract_loras_from_prompt(genparams['prompt'])
                            if loras:
                                genparams['prompt'] = prompt
                                genparams['lora'] = lora_map_name_to_path(loras)
                        gen = sd_generate(genparams)
                        gendat = gen["data"]
                        genanim = gen["animated"]
                        gendatextra = gen["data_extra"]
                        geninfo = json.dumps(gen["info"]) # sdapi really expects a stringified JSON
                        genresp = None
                        if is_comfyui_imggen:
                            if gendat:
                                lastgeneratedcomfyimg = base64.b64decode(gendat)
                            else:
                                lastgeneratedcomfyimg = b''
                            genresp = (json.dumps({"prompt_id": "12345678-0000-0000-0000-000000000001","number": 0,"node_errors":{}}).encode())
                        elif is_oai_imggen:
                            genresp = (json.dumps({"created":int(time.time()),"data":[{"b64_json":gendat}],"background":"opaque","output_format":"png","size":"1024x1024","quality":"medium"}).encode())
                        else:
                            genresp = (json.dumps({"images":[gendat],"parameters":{},"info":geninfo,"animated":genanim,"extra_data":gendatextra}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Generate Image: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_transcribe:
                    try:
                        gendat = None
                        if genparams.get("audio_data","") and state.fullwhispermodelpath=="" and state.has_audio_support: #if we have no whisper model but an audio-capable projector, use that instead
                            adapter_obj = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
                            user_message_start = adapter_obj.get("user_start", "### Instruction:")
                            assistant_message_start = adapter_obj.get("assistant_start", "### Response:")
                            assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
                            prompt = f"{user_message_start} Transcribe all speech in the audio.\n{assistant_message_gen}"
                            rawaudio = genparams.get("audio_data","").replace("data:audio/wav;base64,","")
                            temp_poll = {
                                "prompt": prompt,
                                "max_length":300,
                                "temperature":0.1,
                                "top_k":1,
                                "rep_pen":1,
                                "ban_eos_token":False,
                                "audio": [rawaudio]
                            }
                            temp_poll_result = generate(genparams=temp_poll)
                            gendat = temp_poll_result['text']
                        else:
                            gendat = whisper_generate(genparams)
                        genresp = (json.dumps({"text":gendat}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Transcribe: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_tts:
                    try:
                        gendat = tts_generate(genparams)
                        wav_data = b''
                        if gendat:
                            wav_data = base64.b64decode(gendat) # Decode the Base64 string into binary data
                        self.send_response(200)
                        self.send_header('content-length', str(len(wav_data)))  # Set content length
                        self.send_header('Content-Disposition', 'attachment; filename="output.wav"')
                        self.end_headers(content_type='audio/wav')
                        self.wfile.write(wav_data) # Write the binary WAV data to the response
                    except Exception as ex:
                        utfprint(ex,1)
                        print("TTS: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_embeddings:
                    try:
                        modelNameToReturn = state.friendlyembeddingsmodelname
                        if state.autoswapmode and state.embedName is not None:
                            modelNameToReturn = state.embedName
                        gendat = embeddings_generate(genparams)
                        outdatas = []
                        odidx = 0
                        for od in gendat["data"]:
                            if genparams.get("encoding_format", "")=="base64":
                                binary_data = struct.pack('<' + 'f' * len(od), *od)
                                b64_string = base64.b64encode(binary_data).decode('utf-8')
                                outdatas.append({"object":"embedding","index":odidx,"embedding":b64_string})
                            else:
                                outdatas.append({"object":"embedding","index":odidx,"embedding":od})
                            odidx += 1
                        genresp = (json.dumps({"object":"list","data":outdatas,"model":modelNameToReturn,"usage":{"prompt_tokens":gendat["count"],"total_tokens":gendat["count"]}}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Create Embeddings: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_music_codes:
                    try:
                        gendat = music_generate_codes(genparams)
                        genresp = (json.dumps({"error":"music code generation failed"}).encode())
                        if gendat:
                            genresp = gendat.encode()
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Music Gen Codes: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_music_audio:
                    try:
                        gendat = music_generate_audio(genparams)
                        wav_data = b''
                        if gendat:
                            wav_data = base64.b64decode(gendat) # Decode the Base64 string into binary data
                        self.send_response(200)
                        self.send_header('content-length', str(len(wav_data)))  # Set content length
                        self.send_header('Content-Disposition', 'attachment; filename="output.wav"')
                        self.end_headers(content_type='audio/wav')
                        self.wfile.write(wav_data) # Write the binary WAV data to the response
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Music Gen Audio: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return

        finally:
            time.sleep(0.05)
            if is_batchable_req:
                with state.batched_cond:
                    state.batched_request_runner_count -= 1
                    state.batched_cond.notify_all()
            else:
                state.modelbusy.release()

        self.send_response(404)
        self.end_headers(content_type='text/html')


    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers(content_type='text/html')

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers(content_type='text/html')

    def end_headers(self, content_type=None):
        self.send_header('access-control-allow-origin', '*')
        self.send_header('access-control-allow-methods', '*')
        self.send_header('access-control-allow-headers', '*, Accept, Content-Type, Content-Length, Cache-Control, Accept-Encoding, X-CSRF-Token, Client-Agent, X-Fields, Content-Type, Authorization, X-Requested-With, X-HTTP-Method-Override, apikey, genkey')
        self.send_header("cache-control", "no-store")
        if content_type is not None:
            self.send_header('content-type', content_type)
        return super(KcppServerRequestHandler, self).end_headers()

def RunServerMultiThreaded(addr, port, server_handler):
    if is_port_in_use(port):
        print(f"Warning: Port {port} already appears to be in use by another program.")

    ipv4_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ipv4_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ipv6_sock = None
    if is_ipv6_supported():
        ipv6_sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        ipv6_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ipv6_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)

    if state.args.ssl and state.sslvalid and not state.args.routermode: #if routermode, ssl is already offloaded
        import ssl
        certpath = os.path.abspath(state.args.ssl[0])
        keypath = os.path.abspath(state.args.ssl[1])
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=certpath, keyfile=keypath)
        ipv4_sock = context.wrap_socket(ipv4_sock, server_side=True)
        if ipv6_sock:
            ipv6_sock = context.wrap_socket(ipv6_sock, server_side=True)

    numThreads = 24
    try:
        ipv4_sock.bind((addr, port))
        ipv4_sock.listen(numThreads)
    except Exception:
        ipv4_sock = None
        print("IPv4 Socket Failed to Bind.")

    if ipv6_sock:
        try:
            ipv6_sock.bind((addr, port))
            ipv6_sock.listen(numThreads)
        except Exception:
            ipv6_sock = None
            print("IPv6 Socket Failed to Bind. IPv6 will be unavailable.")

    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()

        def run(self):
            handler = server_handler(addr, port)
            with http.server.HTTPServer((addr, port), handler, False) as self.httpd:
                try:
                    if ipv4_sock and ipv6_sock:
                        self.httpd.socket = ipv4_sock if self.i < 16 else ipv6_sock
                    elif ipv6_sock:
                        self.httpd.socket = ipv6_sock
                    elif ipv4_sock:
                        self.httpd.socket = ipv4_sock
                    else:
                        print("ERROR: Both IPv4 and IPv6 cannot bind. Server features will not work.")
                        return

                    self.httpd.server_bind = self.server_close = lambda self: None
                    self.httpd.serve_forever()
                except (KeyboardInterrupt,SystemExit):
                    exitcounter = 999
                    self.httpd.server_close()
                    sys.exit(0)
                finally:
                    exitcounter = 999
                    self.httpd.server_close()
                    os._exit(0)
        def stop(self):
            exitcounter = 999
            self.httpd.server_close()

    threadArr = []
    for i in range(numThreads):
        threadArr.append(Thread(i))
    while 1:
        try:
            time.sleep(10)
        except (KeyboardInterrupt,SystemExit):
            exitcounter = 999
            for i in range(numThreads):
                try:
                    threadArr[i].stop()
                except Exception:
                    continue
            sys.exit(0)

# Based on https://github.com/mathgeniuszach/xdialog/blob/main/xdialog/zenity_dialogs.py - MIT license | - Expanded version by Henk717
