import os
import re
import sys
import json
import ctypes
import socket
import struct
import threading
import subprocess
import platform
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from . import state


def end_trim_to_sentence(input_text):
    enders = ['.', '!', '?', '*', '"', ')', '}', '`', ']', ';', '…']
    last = -1
    for ender in enders:
        last = max(last, input_text.rfind(ender))
    nl = input_text.rfind("\n")
    last = max(last, nl)
    if last > 0:
        return input_text[:last + 1].strip()
    return input_text.strip()

def tryparseint(value,fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value == "true":
            return 1
        if lower_value == "false":
            return 0
    try:
        return int(value)
    except ValueError:
        return fallback

def tryparsefloat(value,fallback):
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback

def replace_last_in_string(text: str, match: str, replacement: str) -> str:
    if match == "":
        return text
    head, sep, tail = text.rpartition(match)
    if sep == "":
        return text  # old not found
    return head + replacement + tail

def is_incomplete_utf8_sequence(byte_seq): #note, this will only flag INCOMPLETE sequences, corrupted ones will be ignored.
    try:
        byte_seq.decode('utf-8')
        return False  # Valid UTF-8
    except UnicodeDecodeError as e:
        if e.reason == 'unexpected end of data':
            return True #incomplete sequence
        return False #invalid sequence, but not incomplete

def strip_base64_prefix(encoded_data):
    if not encoded_data:
        return ""
    if encoded_data.startswith("data:image"):
        encoded_data = encoded_data.split(',', 1)[-1]
    return encoded_data

def fix_unquoted_keys(s: str) -> str:
    """
    Fix JSON with unquoted keys by only quoting identifiers that appear
    in key position (after '{' or ',' at object level, before ':').
    Uses a state machine to track position in the JSON structure.
    """
    result = []
    i = 0
    n = len(s)
    def skip_whitespace():
        nonlocal i
        while i < n and s[i].isspace():
            result.append(s[i])
            i += 1
    def read_string():
        """Read a quoted string, handling escape sequences correctly."""
        nonlocal i
        assert s[i] == '"'
        result.append(s[i])
        i += 1
        while i < n:
            ch = s[i]
            result.append(ch)
            i += 1
            if ch == '\\':
                if i < n:
                    result.append(s[i])
                    i += 1
            elif ch == '"':
                break
    def read_value():
        """Read any JSON value."""
        nonlocal i
        skip_whitespace()
        if i >= n:
            return
        ch = s[i]
        if ch == '{':
            read_object()
        elif ch == '[':
            read_array()
        elif ch == '"':
            read_string()
        else:
            while i < n and s[i] not in ',}]':
                result.append(s[i])
                i += 1
    def read_object():
        nonlocal i
        result.append(s[i])
        i += 1
        skip_whitespace()
        if i < n and s[i] == '}':
            result.append(s[i])
            i += 1
            return
        while i < n:
            skip_whitespace()
            if i < n and s[i] == '"':
                read_string()
            elif i < n and re.match(r'[a-zA-Z_]', s[i]):
                key = []
                while i < n and re.match(r'[a-zA-Z0-9_]', s[i]):
                    key.append(s[i])
                    i += 1
                result.append('"' + ''.join(key) + '"')
            skip_whitespace()
            if i < n and s[i] == ':':
                result.append(s[i])
                i += 1
            read_value()
            skip_whitespace()
            if i >= n or s[i] == '}':
                break
            if s[i] == ',':
                result.append(s[i])
                i += 1
        if i < n and s[i] == '}':
            result.append(s[i])
            i += 1
    def read_array():
        nonlocal i
        result.append(s[i])
        i += 1
        skip_whitespace()
        if i < n and s[i] == ']':
            result.append(s[i])
            i += 1
            return
        while i < n:
            read_value()
            skip_whitespace()
            if i >= n or s[i] == ']':
                break
            if s[i] == ',':
                result.append(s[i])
                i += 1
        if i < n and s[i] == ']':
            result.append(s[i])
            i += 1
    read_value()
    return ''.join(result)

def old_cpu_check(): #return -1 for pass, 0 if has avx2, 1 if has avx, 2 if has nothing
    shouldcheck = ((sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64")) or
                  (os.name == 'nt' and platform.machine().lower() in ("amd64", "x86_64")))
    if not shouldcheck:
        return -1 #doesnt deal with avx at all.
    try:
        retflags = 0
        if sys.platform == "linux":
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                cpuinfo = cpuinfo.lower()
                if 'avx' not in cpuinfo and 'avx2' not in cpuinfo:
                    retflags = 2
                elif 'avx2' not in cpuinfo:
                    retflags = 1
        elif os.name == 'nt':
            basepath = state.getdirpath()
            output = ""
            data = None
            output = subprocess.run([os.path.join(basepath, "simplecpuinfo.exe")], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS, encoding='utf-8', timeout=6).stdout
            data = json.loads(output)
            if data["avx2"]==0 and data["avx"]==0:
                retflags = 2
            elif data["avx2"]==0:
                retflags = 1
        return retflags
    except Exception:
        return -1 #cannot determine

def utfprint(str, importance = 2): #0 = only debugmode, 1 = except quiet, 2 = always print
    if state.args.quiet and importance<2: #quiet overrides debugmode
        return
    if state.args.debugmode < 1:
        if importance==1 and (state.args.debugmode == -1 or state.args.quiet):
            return
        if importance==0:
            return
    maxlen = 40000
    if state.args.debugmode >= 1:
        maxlen = 240000
    try:
        strlength = len(str)
        if strlength > maxlen: #limit max output len
            str = str[:maxlen] + f"... (+{strlength-maxlen} chars)"
    except Exception:
        pass

    try:
        print(str)
    except UnicodeEncodeError:
        # Replace or omit the problematic character
        utf_string = str.encode('ascii', 'ignore').decode('ascii',"ignore")
        utf_string = utf_string.replace('\a', '') #remove bell characters
        print(utf_string)

def bring_terminal_to_foreground():
    if os.name=='nt':
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)
        ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())

def simple_lcg_hash(input_string): #turns any string into a number between 10000 and 99999
    a = 1664525
    c = 1013904223
    m = 89999  # Modulo
    hash_value = 25343
    for char in input_string:
        hash_value = (a * hash_value + ord(char) + c) % m
    hash_value += 10000
    return hash_value

def string_has_overlap(str_a, str_b, maxcheck):
    max_overlap = min(maxcheck, len(str_a), len(str_b))
    for i in range(1, max_overlap + 1):
        if str_a[-i:] == str_b[:i]:
            return True
    return False

def string_contains_or_overlaps_sequence_substring(inputstr, sequences):
    if inputstr=="":
        return False
    for s in sequences:
        if s.strip()=="":
            continue
        if s.strip() in inputstr.strip() or inputstr.strip() in s.strip():
            return True
        if string_has_overlap(inputstr, s, 10):
            return True
    return False

def truncate_long_json(data, max_length):
    def truncate_middle(s, max_length):
        if len(s) <= max_length or max_length < 5:
            return s
        half = (max_length - 3) // 2
        return s[:half] + "..." + s[-half:]

    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                new_data[key] = truncate_middle(value, max_length)
            else:
                new_data[key] = truncate_long_json(value, max_length)
        return new_data
    elif isinstance(data, list):
        return [truncate_long_json(item, max_length) for item in data]
    elif isinstance(data, str):
        return truncate_middle(data, max_length)
    else:
        return data

def convert_json_to_gbnf(json_obj):
    try:
        from json_to_gbnf import SchemaConverter
        prop_order = []
        converter = SchemaConverter(
        prop_order={name: idx for idx, name in enumerate(prop_order)},
        allow_fetch=False,
        dotall=False,
        raw_pattern=False)
        schema = json.loads(json.dumps(json_obj))
        schema = converter.resolve_refs(schema, '')
        converter.visit(schema, '')
        outstr = converter.format_grammar()
        return outstr
    except Exception as e:
        print(f"JSON to GBNF failed: {e}")
        return ""

def get_capabilities():
    has_llm = not (state.friendlymodelname=="inactive") or (state.autoswapmode and state.textName is not None)
    has_txt2img = not (state.friendlysdmodelname=="inactive" or state.fullsdmodelpath=="") or (state.autoswapmode and state.imageName is not None)
    has_password = (state.password!="")
    has_whisper = (state.fullwhispermodelpath!="") or (state.autoswapmode and state.sttName is not None)
    has_search = True if state.args.websearch else False
    has_tts = (state.ttsmodelpath!="") or (state.autoswapmode and state.ttsName is not None)
    has_embeddings = (state.embeddingsmodelpath!="") or (state.autoswapmode and state.embedName is not None)
    has_music = (state.musicdiffusionmodelpath!="" or state.musicllmmodelpath!="") or (state.autoswapmode and state.musicName is not None)
    visionSupport = (state.has_vision_support) or (state.autoswapmode and state.mmprojName is not None) #todo: not always correct
    audioSupport = (state.has_audio_support) #todo: not always correct
    has_guidance = True if state.args.enableguidance else False
    has_jinja = True if state.args.jinja else False
    has_mcp = True if (state.args.mcpfile and state.mcp_connections and len(state.mcp_connections) > 0) else False
    admin_type = (2 if state.args.admin and state.args.admindir and state.args.adminpassword else (1 if state.args.admin and state.args.admindir else 0))
    has_router = True if state.args.routermode else False
    return {"result":"KoboldCpp", "version":state.KcppVersion, "protected":has_password, "llm":has_llm, "txt2img":has_txt2img,"vision":visionSupport,"audio":audioSupport,"transcribe":has_whisper,"multiplayer":state.has_multiplayer,"websearch":has_search,"tts":has_tts, "embeddings":has_embeddings, "music":has_music, "savedata":(state.savedata_obj is not None), "admin": admin_type, "router":has_router, "guidance": has_guidance, "jinja": has_jinja, "mcp":has_mcp}


def scan_directory(dirpath, valid_exts, depth):
    files = []
    for entry in sorted(os.listdir(dirpath)): # Scan top-level directory
        full_path = os.path.join(dirpath, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(valid_exts): # If toplevel file
            files.append(entry)
        elif depth > 0 and os.path.isdir(full_path): #if dir, scan up to 1 level deep
            for subentry in sorted(os.listdir(full_path)):
                sub_full_path = os.path.join(full_path, subentry)
                if os.path.isfile(sub_full_path) and subentry.lower().endswith(valid_exts):
                    rel_path = os.path.join(entry, subentry)
                    files.append(rel_path)
    return files


def get_current_admindir_list():
    opts = []
    if state.args.admin and state.args.admindir:
        dirpath = os.path.abspath(state.args.admindir)
        valid_exts = (".kcpps", ".kcppt", ".gguf")
        opts = scan_directory(dirpath, valid_exts, 1)
        opts.append("initial_model")
        opts.append("unload_model")
    return opts


def is_port_in_use(portNum):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('localhost', portNum)) == 0
    except Exception:
        return True

def is_ipv6_supported():
    try:
        # Attempt to create an IPv6 socket
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.close()
        return True
    except Exception:
        return False

def LaunchWebbrowser(target_url, failedmsg):
    try:
        if os.name == "posix" and "DISPLAY" in os.environ:  # UNIX-like systems
            clean_env = os.environ.copy()
            clean_env.pop("LD_LIBRARY_PATH", None)
            clean_env["PATH"] = "/usr/bin:/bin"
            result = subprocess.run(["/usr/bin/env", "xdg-open", target_url], check=True, env=clean_env)
            if result.returncode == 0:
                return  # fallback successful
        raise RuntimeError("no xdg-open")
    except Exception:
        try:
            import webbrowser as wb
            if wb.open(target_url, autoraise=True):
                return  # If successful, exit the function
            raise RuntimeError("wb.open failed")
        except Exception:
            print(failedmsg)
            print(f"Please manually open your browser to {target_url}")

def get_my_epurl():
    httpsaffix = ("https" if state.sslvalid else "http")
    displayedport = (state.args.port if not state.args.proxy_port else state.args.proxy_port)
    epurl = f"{httpsaffix}://localhost:{displayedport}"
    if state.args.host!="":
        epurl = f"{httpsaffix}://{state.args.host}:{displayedport}"
    return epurl

def print_with_time(txt):
    print(f"{datetime.now().strftime('[%H:%M:%S]')} " + txt, flush=True)

def make_url_request(url, data, method='POST', headers={}, timeout=300):
    try:
        request = None
        ssl_cert_dir = os.environ.get('SSL_CERT_DIR')
        if not ssl_cert_dir and not state.nocertify and os.name != 'nt':
            os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
        if method=='POST':
            json_payload = json.dumps(data).encode('utf-8')
            request = urllib.request.Request(url, data=json_payload, headers=headers, method=method)
            request.add_header('content-type', 'application/json')
        else:
            request = urllib.request.Request(url, headers=headers, method=method)
        response_data = ""
        with urllib.request.urlopen(request,timeout=timeout) as response:
            content_type = response.headers.get('Content-Type', '')
            response_data = response.read()
            if 'audio/wav' in content_type:
                return response_data #raw binary
            else:
                json_response = json.loads(response_data.decode('utf-8',"ignore"))
                return json_response
    except urllib.error.HTTPError as e:
        try:
            errmsg = e.read().decode('utf-8',"ignore")
            print_with_time(f"Error: {e} - {errmsg}")
        except Exception as e:
            print_with_time(f"Error: {e}")
        return None
    except Exception as e:
        print_with_time(f"Error: {e} - {response_data}")
        return None

def sanitize_string(input_string):
    # alphanumeric characters, dots, dashes, and underscores
    sanitized_string = re.sub( r'[^\w\d\.\-_]', '', input_string)
    return sanitized_string
