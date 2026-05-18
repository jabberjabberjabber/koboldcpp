#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KoboldCpp main entry point (split from koboldcpp.py)

import os
import copy
import ctypes
import multiprocessing
import re
import argparse
import platform
import base64
import json
import sys
import time
import asyncio
import threading
import subprocess
import gzip
from datetime import datetime, timezone

from . import state
from .state import (
    KcppVersion, defaultport, multiuser_concurrent_limit,
    default_genlen, default_autofit_padding, default_ttsmaxlen,
    default_visionmaxres, default_vae_tile_threshold, default_reqtimeout,
    swa_padding_default, default_draft_amount,
)
from .config import (
    splitmode_choices, convert_invalid_args, reload_from_new_args,
    reload_new_config, load_config_cli, save_config_cli,
)
from .model_meta import (
    has_valid_model, exit_with_error, delete_old_pyinstaller,
    unpack_to_dir, analyze_gguf_model_wrapper, download_model_from_url,
    downloader_internal, extract_modelfile_params,
)
from .hardware import (
    get_default_threads, fetch_gpu_properties, auto_set_backend_cli,
    autoset_gpu_layers,
)
from .backend import init_library, load_model, generate
from .sd import (
    sd_load_model, sd_get_info, sanitize_lora_list, sanitize_lora_multipliers,
    sd_convdirect_option, sd_convdirect_choices,
)
from .whisper import whisper_load_model
from .tts import tts_load_model
from .embeddings import embeddings_load_model
from .music import music_load_model
from .horde import run_horde_worker
from .mcp import load_mcp_async
from .io_redirect import suppress_stdout, restore_stdout
from .utils import (
    get_capabilities, scan_directory, is_port_in_use,
    LaunchWebbrowser, make_url_request, sanitize_string,
)
from .gen_params import transform_genparams


from .gui import show_gui, show_gui_msgbox, show_gui_yesnobox
from .server import run_router_proxy, RunServerMultiThreaded, KcppServerRequestHandler


# ---------------------------------------------------------------------------
# setuptunnel
# ---------------------------------------------------------------------------

def setuptunnel(global_memory, has_sd, has_music):
    # This script will help setup a cloudflared tunnel for accessing KoboldCpp over the internet
    # It should work out of the box on both linux and windows
    try:
        httpsaffix = ("https" if state.sslvalid else "http")
        ssladd = (" --no-tls-verify" if state.sslvalid else "")
        def run_tunnel():
            tunnelproc = None
            tunneloutput = ""
            tunnelrawlog = ""
            time.sleep(0.2)
            tunnelbinary = ""
            if os.name == 'nt':
                print("Starting Cloudflare Tunnel for Windows, please wait...", flush=True)
                tunnelbinary = "cloudflared.exe"
            elif sys.platform=="darwin":
                print("Starting Cloudflare Tunnel for MacOS, please wait...", flush=True)
                tunnelbinary = "./cloudflared"
            elif sys.platform == "linux" and platform.machine().lower() == "aarch64":
                print("Starting Cloudflare Tunnel for ARM64 Linux, please wait...", flush=True)
                tunnelbinary = "./cloudflared-linux-arm64"
            else:
                print("Starting Cloudflare Tunnel for Linux, please wait...", flush=True)
                tunnelbinary = "./cloudflared-linux-amd64"

            tunnelproc = None
            displayedport = (state.args.port if not state.args.proxy_port else state.args.proxy_port)
            if sys.platform == "linux":
                clean_env = os.environ.copy()
                clean_env.pop("LD_LIBRARY_PATH", None)
                clean_env["PATH"] = "/usr/bin:/bin"
                tunnelproc = subprocess.Popen(f"{tunnelbinary} tunnel --url {httpsaffix}://localhost:{int(displayedport)}{ssladd}", text=True, encoding='utf-8', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=clean_env)
            else:
                tunnelproc = subprocess.Popen(f"{tunnelbinary} tunnel --url {httpsaffix}://localhost:{int(displayedport)}{ssladd}", text=True, encoding='utf-8', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(10)

            def tunnel_reader():
                nonlocal tunnelproc,tunneloutput,tunnelrawlog
                pattern = r'https://[\w\.-]+\.trycloudflare\.com'
                while True:
                    line = tunnelproc.stderr.readline() #cloudflare writes to stderr for some reason
                    tunnelrawlog += line+"\n"
                    if not line:
                        return
                    found = re.findall(pattern, line)
                    for x in found:
                        tunneloutput = x
                        if global_memory and global_memory["load_complete"]:
                            print(f"Your remote Kobold API can be found at {tunneloutput}/api")
                            print(f"Your remote OpenAI Compatible API can be found at {tunneloutput}/v1")
                            print(f"Your remote llama.cpp secondary WebUI at {tunneloutput}/lcpp/")
                            if has_sd:
                                print(f"StableUI is available at {tunneloutput}/sdui/")
                            if has_music:
                                print(f"MusicUI is available at {tunneloutput}/musicui/")
                            print("======\n")
                            print(f"Your remote tunnel is ready, please connect to {tunneloutput}", flush=True)
                        if global_memory:
                            global_memory["tunnel_url"] = tunneloutput
                        return

            tunnel_reader_thread = threading.Thread(target=tunnel_reader)
            tunnel_reader_thread.start()
            time.sleep(5)
            if tunneloutput=="":
                print(f"Error: Could not create cloudflare tunnel!\nMore Info:\n{tunnelrawlog}", flush=True)
            time.sleep(0.5)
            tunnelproc.wait()

        if os.name == 'nt':
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe", "cloudflared.exe", True, 500000)
        elif sys.platform=="darwin":
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz", "cloudflared-darwin-amd64.tgz", True, 500000)
            subprocess.run("tar -xzf cloudflared-darwin-amd64.tgz", shell=True)
            subprocess.run("chmod +x 'cloudflared'", shell=True)
        elif sys.platform == "linux" and platform.machine().lower() == "aarch64":
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64", "cloudflared-linux-arm64", True, 500000)
            subprocess.run("chmod +x 'cloudflared-linux-arm64'", shell=True)
        else:
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "cloudflared-linux-amd64", True, 500000)
            subprocess.run("chmod +x 'cloudflared-linux-amd64'", shell=True)
        print("Attempting to start tunnel thread...", flush=True)
        tunnel_thread = threading.Thread(target=run_tunnel)
        tunnel_thread.start()
    except Exception as ex:
        print("Remote Tunnel Failed!")
        print(str(ex))
        return None


# ---------------------------------------------------------------------------
# register_koboldcpp
# ---------------------------------------------------------------------------

def register_koboldcpp():
    try:
        exe_path = ""
        if getattr(sys, 'frozen', False):
            exe_path = sys.executable
        if os.name == 'nt' and exe_path!="":
            confirmyes = show_gui_yesnobox("Confirm Add File Extensions","Do you want to register KoboldCpp as the default file associations for .gguf, .kcpps, .kcppt and .ggml files?",icon="question")
            if confirmyes == 'yes':
                import winreg
                print(f"Registering file associations to {exe_path}")
                entries = [
                    (r"Software\Classes\KoboldCpp\DefaultIcon", "", f"{exe_path},0"),
                    (r"Software\Classes\KoboldCpp\shell\Open\command", "", f'"{exe_path}" "%1" --singleinstance'),
                    (r"Software\Classes\KoboldCpp\shell\Edit\command", "", f'"{exe_path}" "%1" --singleinstance --showgui'),
                    (r"Software\Classes\.gguf", "", "KoboldCpp"),
                    (r"Software\Classes\.kcpps", "", "KoboldCpp"),
                    (r"Software\Classes\.kcppt", "", "KoboldCpp"),
                    (r"Software\Classes\.ggml", "", "KoboldCpp"),
                ]
                for key_path, value_name, value_data in entries:
                    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, value_data)
                print("KoboldCpp file associations registered successfully.")
        else:
            show_gui_msgbox("Cannot Set File Association","File Associations only available for Windows standalone executables.")
    except Exception as e:
        print(f"Register Extensions: An error occurred: {e}")


# ---------------------------------------------------------------------------
# unregister_koboldcpp
# ---------------------------------------------------------------------------

def unregister_koboldcpp():
    try:
        if os.name == 'nt':
            confirmyes = show_gui_yesnobox("Confirm Remove File Extensions","Do you want to unregister KoboldCpp as the default file associations for .gguf, .kcpps, .kcppt and .ggml files?",icon="question")
            if confirmyes == 'yes':
                import winreg
                keys_to_delete = [
                    r"Software\Classes\KoboldCpp\shell\Edit\command",
                    r"Software\Classes\KoboldCpp\shell\Edit",
                    r"Software\Classes\KoboldCpp\shell\Open\command",
                    r"Software\Classes\KoboldCpp\shell\Open",
                    r"Software\Classes\KoboldCpp\shell",
                    r"Software\Classes\KoboldCpp\DefaultIcon",
                    r"Software\Classes\KoboldCpp",
                    r"Software\Classes\.gguf",
                    r"Software\Classes\.kcpps",
                    r"Software\Classes\.kcppt",
                    r"Software\Classes\.ggml",
                ]
                for key_path in keys_to_delete:
                    try:
                        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
                    except Exception:
                        print(f"Failed to delete registry key: {key_path}")
                print("KoboldCpp file associations unregistered.")
        else:
            show_gui_msgbox("Cannot Set File Association","File Associations only available for Windows standalone executables.")
    except Exception as e:
        print(f"Unregister Extensions: An error occurred: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(launch_args, default_args):
    state.args = launch_args  # note: these are NOT shared with the child processes!

    if (state.args.version) and len(sys.argv) <= 2:
        print(f"{KcppVersion}") # just print version and exit
        return

    if state.args.testmemory:
        fetch_gpu_properties(True, True, testmemory=True)
        return

    #prevent disallowed combos
    if (state.args.nomodel or state.args.benchmark or state.args.launch or state.args.admin) and state.args.cli:
        exit_with_error(1, "Error: --cli cannot be combined with --launch, --nomodel, --admin or --benchmark")

    state.args = convert_invalid_args(state.args)

    temp_hide_print = (state.args.model_param and (state.args.prompt and not state.args.cli) and not state.args.benchmark and not (state.args.debugmode >= 1))

    if not temp_hide_print:
        print(f"***\nWelcome to KoboldCpp - Version {KcppVersion}")
    if state.args.debugmode != 1:
        state.showdebug = False  # not shared with child process!
    if state.args.debugmode >= 1:
        print("Debug Mode is Enabled!")
        state.args.quiet = False  # verbose outputs

    # assign title to terminal on windows
    try:
        if os.name == 'nt':
            windowtitle = f"KoboldCpp {KcppVersion} Terminal"
            os.system(f'title {windowtitle}')
    except Exception:
        pass

    try:
        delete_old_pyinstaller()  #perform some basic cleanup of old temporary directories
    except Exception as e:
        print(f"Error cleaning up orphaned pyinstaller dirs: {e}")

    if state.args.unpack:
        unpack_to_dir(state.args.unpack)
        return

    if state.args.analyze:
        analyze_gguf_model_wrapper(state.args.analyze)
        return

    cfgname = ""
    if state.args.config and len(state.args.config)==1: #handle initial config loading for launch
        cfgname = state.args.config[0] #store first so baseconfig wont overwrite it

    if cfgname: #handle initial config loading for launch
        if isinstance(cfgname, str):
            dlfile = download_model_from_url(cfgname,[".kcpps",".kcppt"])
            if dlfile:
                cfgname = dlfile
        if isinstance(cfgname, str) and os.path.exists(cfgname):
           load_config_cli(cfgname)
        elif state.args.ignoremissing:
            print("Ignoring missing kcpp config file...")
        else:
            state.exitcounter = 999
            exit_with_error(2,"Specified kcpp config file invalid or not found.")
    state.args = convert_invalid_args(state.args)

    #positional handling for kcpps files (drag and drop)
    if state.args.model_param and state.args.model_param!="" and (state.args.model_param.lower().endswith('.kcpps') or state.args.model_param.lower().endswith('.kcppt') or state.args.model_param.lower().endswith('.kcpps?download=true') or state.args.model_param.lower().endswith('.kcppt?download=true')):
        dlfile = download_model_from_url(state.args.model_param,[".kcpps",".kcppt"]) # maybe download from url
        if dlfile:
            state.args.model_param = dlfile
        load_config_cli(state.args.model_param)

    if state.args.exportconfig:
        save_config_cli(state.args.exportconfig,False)
        return
    if state.args.exporttemplate:
        save_config_cli(state.args.exporttemplate,True)
        return

    # show the GUI launcher if a model was not provided
    if state.args.showgui or not has_valid_model():
        #give them a chance to pick a file
        print("For command line arguments, please refer to --help")
        print("***")
        try:
            show_gui()
        except Exception as ex:
            state.exitcounter = 999
            ermsg = "Reason: " + str(ex) + "\nFile selection GUI unsupported.\ncustomtkinter python module required!\n\nYou must use the command line instead, e.g. python ./koboldcpp.py --help"
            show_gui_msgbox("Warning, GUI failed to start",ermsg)
            if state.args.skiplauncher:
                print("Note: In order to use --skiplauncher, you need to specify a model with --model")
            time.sleep(3)
            sys.exit(2)

    if state.args.ssl: #need to duplicate here for the tunnel
        if len(state.args.ssl)==2 and isinstance(state.args.ssl[0], str) and os.path.exists(state.args.ssl[0]) and isinstance(state.args.ssl[1], str) and os.path.exists(state.args.ssl[1]):
            state.sslvalid = True

    state.args.proxy_port = None #normally unused
    if state.args.autoswapmode:
        if not state.args.routermode:
            print("\nWARNING: Autoswap mode requires router, enabling router...")
            state.args.routermode = True
    if state.args.routermode:
        if not state.args.admin:
            print("\nWARNING: Router mode requires admin, enabling admin...")
            state.args.admin = True
        # setup router mode, find a usable high port swap the port
        newport = 15001
        for prt in range(15001,15011):
            if not is_port_in_use(prt):
                newport = prt
                break
        state.args.proxy_port = state.args.port_param
        state.args.port = state.args.port_param = newport
        if state.args.singleinstance and is_port_in_use(state.args.proxy_port):
            try:
                print(f"Warning: Port {state.args.proxy_port} already appears to be in use by another program.")
                print(f"Attempting to request shutdown of previous instance on port {state.args.proxy_port}...")
                shutdownreq = make_url_request(f'http://localhost:{state.args.proxy_port}/api/extra/shutdown',{},timeout=5)
                shutdownok = (shutdownreq and "success" in shutdownreq and shutdownreq["success"] is True)
                time.sleep(2)
                print("Shutdown existing successful!" if shutdownok else "Shutdown existing failed!")
                time.sleep(1)
            except Exception:
                pass
        run_router_proxy(state.args.proxy_port,newport)

    if state.args.admin and not state.args.admindir:
        print("\nWARNING: Admin was set without selecting an admin directory. Selecting current executable directory...")
        autopath = os.path.realpath(__file__)
        if getattr(sys, 'frozen', False):
            autopath = sys.executable
        autopath = os.path.dirname(autopath)
        state.args.admindir = autopath
        print(f"Admin Directory Set: {autopath}\n")

    if not state.args.admin: #run in single process mode
        if state.args.remotetunnel and not state.args.prompt and not state.args.benchmark and not state.args.cli:
            setuptunnel(state.global_memory, True if state.args.sdmodel else False, True if (state.args.musicdiffusion or state.args.musicllm or state.args.ttsmodel) else False)
        kcpp_main_process(state.args,state.global_memory,state.using_gui_launcher)
        if state.global_memory["input_to_exit"]:
            print("===")
            print("Press ENTER key to exit.", flush=True)
            input()
    else:  # manager command queue for admin mode
        with multiprocessing.Manager() as mp_manager:
            state.global_memory = mp_manager.dict({"tunnel_url": "", "restart_target":"", "input_to_exit":False, "load_complete":False, "restart_override_base_config":"", "last_active_timestamp":datetime.now(), "triggered_sleeping":False, "current_model":"initial_model", "base_config":"", "swapReqType": None, "autoswapmode": False})

            if state.args.remotetunnel and not state.args.prompt and not state.args.benchmark and not state.args.cli:
                setuptunnel(state.global_memory, True if state.args.sdmodel else False, True if (state.args.musicdiffusion or state.args.musicllm or state.args.ttsmodel) else False)

            # invoke the main koboldcpp process
            original_args = copy.deepcopy(state.args)

            state.kcpp_instance = multiprocessing.Process(target=kcpp_main_process,kwargs={"launch_args": state.args, "g_memory": state.global_memory, "gui_launcher": state.using_gui_launcher})
            state.kcpp_instance.daemon = True
            state.kcpp_instance.start()

            fault_recovery_mode = False #if a config reload fails, recover back to old settings

            while True: # keep the manager alive
                try:
                    restart_target = ""
                    restart_override_base_config = ""
                    if not state.kcpp_instance or not state.kcpp_instance.is_alive():
                        if fault_recovery_mode:
                            #attempt to recover
                            print("Attempting to recover to safe mode, launching known-good config...")
                            fault_recovery_mode = False
                            state.args = copy.deepcopy(original_args) #restore known good original launcher args
                            if state.kcpp_instance:
                                state.kcpp_instance.terminate()
                                state.kcpp_instance.join(timeout=10)  # Ensure process is stopped
                                state.kcpp_instance = None
                            state.kcpp_instance = multiprocessing.Process(target=kcpp_main_process,kwargs={"launch_args": state.args, "g_memory": state.global_memory, "gui_launcher": False})
                            state.kcpp_instance.daemon = True
                            state.kcpp_instance.start()
                            state.global_memory["restart_target"] = ""
                            state.global_memory["restart_override_base_config"] = ""
                            state.global_memory["swapReqType"] = None
                            time.sleep(3)
                        else:
                            break # kill the program
                    if fault_recovery_mode and state.global_memory["load_complete"]:
                        fault_recovery_mode = False
                    restart_target = state.global_memory["restart_target"]
                    restart_override_base_config = state.global_memory["restart_override_base_config"]
                    last_active = state.global_memory["last_active_timestamp"]
                    if last_active and state.args.adminunloadtimeout>0:
                        curtime = datetime.now()
                        elapsedtime = curtime - last_active
                        time_since_last_active = elapsedtime.total_seconds()
                        if time_since_last_active > state.args.adminunloadtimeout:
                            if state.args.autoswapmode:
                                if state.global_memory["swapReqType"] is not None and state.global_memory["swapReqType"] != "nomodel":
                                    print(f"[Unload Timeout] Inactive for over {time_since_last_active}s, unloading models via autoswap...")
                                    state.global_memory["swapReqType"] = "nomodel"
                                    state.global_memory["triggered_sleeping"] = True
                            elif state.global_memory["current_model"]!="unload_model":
                                print(f"[Unload Timeout] Inactive for over {time_since_last_active}s, unloading models...")
                                restart_target = "unload_model"
                                state.global_memory["triggered_sleeping"] = True
                    if restart_target!="":
                        overridetxt = ("" if not restart_override_base_config else f" with override config {restart_override_base_config}")
                        print(f"Reloading new model/config: {restart_target}{overridetxt}")
                        state.global_memory["restart_target"] = ""
                        state.global_memory["restart_override_base_config"] = ""
                        time.sleep(0.5) #sleep for 0.5s then restart
                        if state.args.admin and state.args.admindir:
                            dirpath = os.path.abspath(state.args.admindir)
                            maintarget_filepath = os.path.abspath(os.path.join(dirpath, restart_target))
                            basecfg_filepath = os.path.abspath(os.path.join(dirpath, restart_override_base_config)) if restart_override_base_config else ""
                            if os.path.commonpath([dirpath, maintarget_filepath]) != dirpath: # Enforce admindir jail
                                print("Security: Invalid restart target path.")
                                continue
                            if basecfg_filepath and os.path.commonpath([dirpath, basecfg_filepath]) != dirpath:
                                print("Security: Invalid override config path.")
                                continue
                            #if override config is not specified, AND baseconfig is, swap it as our override
                            if restart_target!="unload_model" and restart_target!="initial_model" and state.args.baseconfig and not basecfg_filepath:
                                my_basecfg_path = os.path.abspath(state.args.baseconfig)
                                if os.path.exists(my_basecfg_path):
                                    basecfg_filepath = my_basecfg_path
                                    print(f"No override config provided, using baseconfig {state.args.baseconfig}")
                            defaultargs = vars(default_args)
                            if (os.path.exists(maintarget_filepath) or restart_target=="unload_model" or restart_target=="initial_model") and (restart_override_base_config=="" or os.path.exists(basecfg_filepath)):
                                print("Terminating old process...")
                                state.global_memory["load_complete"] = False
                                state.kcpp_instance.terminate()
                                state.kcpp_instance.join(timeout=10)  # Ensure process is stopped
                                state.kcpp_instance = None
                                print("Restarting KoboldCpp...")
                                fault_recovery_mode = True
                                #then, apply the rest of the config stack
                                if restart_target=="unload_model":
                                    reload_from_new_args(defaultargs)
                                    state.args.model_param = None
                                    state.args.model = None
                                    state.args.nomodel = True
                                elif restart_target=="initial_model":
                                    reload_from_new_args(vars(original_args))
                                elif maintarget_filepath.endswith(".gguf") and basecfg_filepath=="":
                                    reload_from_new_args(defaultargs)
                                    state.args.model_param = maintarget_filepath
                                elif maintarget_filepath.endswith(".gguf") and basecfg_filepath!="":
                                    reload_from_new_args(defaultargs)
                                    reload_new_config(basecfg_filepath,vars(state.args),True)
                                    state.args.model_param = maintarget_filepath
                                elif maintarget_filepath and basecfg_filepath and basecfg_filepath!="":
                                    # stuff applied later overwrites stuff applied earlier, if they have the same field names
                                    reload_from_new_args(defaultargs)
                                    reload_new_config(basecfg_filepath,vars(state.args),True)
                                    reload_new_config(maintarget_filepath,vars(state.args),True)
                                else:
                                    reload_from_new_args(defaultargs)
                                    reload_new_config(maintarget_filepath,vars(state.args),True)
                                state.global_memory["autoswapmode"] = state.args.autoswapmode
                                state.kcpp_instance = multiprocessing.Process(target=kcpp_main_process,kwargs={"launch_args": state.args, "g_memory": state.global_memory, "gui_launcher": False})
                                state.kcpp_instance.daemon = True
                                state.kcpp_instance.start()
                                state.global_memory["restart_target"] = ""
                                if (restart_override_base_config and restart_override_base_config!=""):
                                    state.global_memory["base_config"] = restart_override_base_config
                                else:
                                    state.global_memory["base_config"] = ""
                                state.global_memory["restart_override_base_config"] = ""
                                state.global_memory["current_model"] = restart_target
                                time.sleep(3)
                    else:
                        time.sleep(0.2)
                except (KeyboardInterrupt,SystemExit):
                    break
            if state.global_memory["input_to_exit"]:
                print("===")
                print("Press ENTER key to exit.", flush=True)
                input()


# ---------------------------------------------------------------------------
# mk_lora_info
# ---------------------------------------------------------------------------

def mk_lora_info(imgloras, multipliers):
    first_multiplier = multipliers[0] if len(multipliers) > 0 else 1.
    lora_files = []
    lora_dirs = []
    # identify files and dirs
    for i, lora_path in enumerate(imgloras):
        multiplier = multipliers[i] if i < len(multipliers) else first_multiplier
        if os.path.isfile(lora_path):
            lora_files.append(('', lora_path, multiplier))
        elif os.path.isdir(lora_path):
            lora_dirs.append(lora_path)
        elif os.path.exists(lora_path):
            print(f"Unexpected file type for SD LORA model file {lora_path}")
        else:
            print(f"Missing SD LORA model file {lora_path}...")
    # scan all dirs
    for lora_dir in lora_dirs:
        print(f'Scanning {lora_dir} for LoRAs...')
        files = scan_directory(lora_dir, ('.safetensors', '.gguf'), 1)
        print(f'  found {len(files)} files under {lora_dir}')
        for file in files:
            lora_files.append((lora_dir, file, 0.0))
    # dedup and map all files
    unique_lora_names = set()
    lora_fullmap = {}
    for i, (lora_dir, lora_path, multiplier) in enumerate(lora_files):
        if lora_dir:
            # lora_path is relative: we can show it on the interface and accept it
            lora_fullpath = os.path.join(lora_dir, lora_path)
            # NOTE: we are including the relative directory on the short name
            lora_file = lora_path
            preloaded = False
        else:
            lora_fullpath = lora_path
            # we don't know which portion of the path we can show, so omit it
            lora_file = os.path.basename(lora_path)
            preloaded = True
        lora_fullpath = os.path.abspath(lora_fullpath)
        # dedup paths (e.g. preloaded and on directory)
        info = lora_fullmap.get(lora_fullpath)
        if info:
            info["multiplier"] += multiplier
            if multiplier == 0.0 and 'fixed' in info:
                # allow changes if we see this lora again with weight 0
                del info['fixed']
            continue
        lora_name, lora_ext = os.path.splitext(lora_file)
        # ensure unique names
        i = 1
        lora_uname = lora_name
        while lora_uname in unique_lora_names:
            i += 1
            lora_uname = lora_name + '_' + str(i)
        unique_lora_names.add(lora_uname)
        lora_upath = lora_uname + lora_ext
        lora_entry = {
            'fullpath': lora_fullpath,  # where it is on disk
            'name': lora_uname,         # 'name' in api field and <lora:name:multiplier>
            'path': lora_upath,         # 'path' in api field (relative), + extension
            'multiplier': multiplier,   # preload multiplier
        }
        if preloaded:
            lora_entry['preloaded'] = preloaded
        if multiplier != 0.0 and state.imglora_initial_fixed:
            lora_entry['fixed'] = True
        lora_fullmap[lora_fullpath] = lora_entry
    # build the runtime tables
    preloaded_table = []
    lora_path_map = {}
    lora_name_map = {}
    for lora_entry in lora_fullmap.values():
        if not lora_entry.get("fixed"):  # only map LoRAs that can be changed
            lora_path_map[lora_entry["path"]] = lora_entry
            lora_name_map[lora_entry["name"]] = lora_entry["path"]
        if lora_entry.get("preloaded"):
            preloaded_table.append(lora_entry)
    return preloaded_table, lora_path_map, lora_name_map


# ---------------------------------------------------------------------------
# disableSwappedFieldsInConfig
# ---------------------------------------------------------------------------

def disableSwappedFieldsInConfig(args, swapReqType):
    print(f"Swapping to type: {swapReqType}")
    if swapReqType != "text":
        for e in ["model", "model_param", "lora", "mmproj"]:
            setattr(args, e, "")
    if swapReqType != "stt":
        for e in ["whispermodel"]:
            setattr(args, e, "")
    if swapReqType != "tts":
        for e in ["ttsmodel", "ttswavtokenizer"]:
            setattr(args, e, "")
    if swapReqType != "embed":
        for e in ["embeddingsmodel"]:
            setattr(args, e, "")
    if swapReqType != "music":
        for e in ["musicllm", "musicembeddings", "musicdiffusion", "musicvae"]:
            setattr(args, e, "")
    if swapReqType != "image":
        for e in ["sdmodel", "sdt5xxl", "sdclip1", "sdclip2", "sdphotomaker", "sdupscaler", "sdvae", "sdlora"]:
            setattr(args, e, "")


# ---------------------------------------------------------------------------
# kcpp_main_process
# ---------------------------------------------------------------------------

def kcpp_main_process(launch_args, g_memory=None, gui_launcher=False):
    state.start_time = time.time()
    state.args = launch_args
    state.global_memory = g_memory
    state.using_gui_launcher = gui_launcher

    start_server = True

    if state.args.model_param and (state.args.prompt and not state.args.cli) and not state.args.benchmark and not (state.args.debugmode >= 1):
        suppress_stdout()

    state.autoswapmode = False
    state.textName = None
    state.mmprojName = None
    state.sttName = None
    state.ttsName = None
    state.embedName = None
    state.musicName = None
    state.imageName = None
    if state.args.autoswapmode is not None and state.args.autoswapmode:
        state.autoswapmode = True
        state.global_memory["autoswapmode"] = True
        if state.args.model_param and state.args.model_param!="":
            tempName = os.path.basename(os.path.abspath(state.args.model_param))
            tempName = os.path.splitext(tempName)[0]
            state.textName = "koboldcpp/" + sanitize_string(tempName)
            if state.args.mmproj and state.args.mmproj!="": # multimodal vision and audio support is assumed to work with mmproj - this may be incorrect!
                tempName = os.path.basename(os.path.abspath(state.args.mmproj))
                tempName = os.path.splitext(tempName)[0]
                state.mmprojName = sanitize_string(tempName)
        if state.args.whispermodel and state.args.whispermodel!="":
            tempName = os.path.basename(os.path.abspath(state.args.whispermodel))
            tempName = os.path.splitext(tempName)[0]
            state.sttName = sanitize_string(tempName)
        if state.args.ttsmodel and state.args.ttsmodel!="":
            tempName = os.path.basename(os.path.abspath(state.args.ttsmodel))
            tempName = os.path.splitext(tempName)[0]
            state.ttsName = sanitize_string(tempName)
        if state.args.embeddingsmodel and state.args.embeddingsmodel!="":
            tempName = os.path.basename(os.path.abspath(state.args.embeddingsmodel))
            tempName = os.path.splitext(tempName)[0]
            state.embedName = sanitize_string(tempName)
        if state.args.musicdiffusion and state.args.musicdiffusion!="":
            tempName = os.path.basename(os.path.abspath(state.args.musicdiffusion))
            tempName = os.path.splitext(tempName)[0]
            state.musicName = sanitize_string(tempName)
        if state.args.sdmodel and state.args.sdmodel!="":
            tempName = os.path.basename(os.path.abspath(state.args.sdmodel))
            tempName = os.path.splitext(tempName)[0]
            state.imageName = sanitize_string(tempName)
        if state.global_memory["swapReqType"] is not None:
            disableSwappedFieldsInConfig(state.args, state.global_memory["swapReqType"])
        else:
            state.global_memory["swapReqType"] = "nomodel"
            setattr(state.args, "nomodel", True)
            disableSwappedFieldsInConfig(state.args, "nomodel")
    else:
        state.global_memory["autoswapmode"] = False

    if state.args.model_param and (state.args.benchmark or state.args.prompt or state.args.cli):
        start_server = False

    state.args.sdlora = sanitize_lora_list(state.args.sdlora)
    state.args.sdloramult = sanitize_lora_multipliers(state.args.sdloramult)

    #try to read story if provided
    if state.args.preloadstory:
        canload = False
        if isinstance(state.args.preloadstory, str) and os.path.exists(state.args.preloadstory):
            print(f"Preloading saved story {state.args.preloadstory} into server...")
            with open(state.args.preloadstory, mode='rb') as f:
                state.preloaded_story = f.read()
                canload = True
        elif isinstance(state.args.preloadstory, str):
            print("Preloading saved story as JSON into server...")
            try:
                import ast
                parsed = ast.literal_eval(state.args.preloadstory)
                state.preloaded_story = json.dumps(parsed).encode()
                canload = True
            except Exception as ex:
                print(ex)
        elif isinstance(state.args.preloadstory, dict):
            try:
                state.preloaded_story = json.dumps(state.args.preloadstory).encode()
                canload = True
            except Exception as ex:
                print(ex)
        if canload:
            print("Saved story preloaded.")
        else:
            print("Warning: Saved story file invalid or not found. No story will be preloaded into server.")

    # try to read chat completions adapter
    if state.args.chatcompletionsadapter:
        from .sd import parse_json_object
        ccadapter_path = None
        canload = False
        adapt_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kcpp_adapters')
        adapt_dir = adapt_dir if os.path.isdir(adapt_dir) else None
        if isinstance(state.args.chatcompletionsadapter, str) and os.path.exists(state.args.chatcompletionsadapter):
            ccadapter_path = os.path.abspath(state.args.chatcompletionsadapter)
        elif isinstance(state.args.chatcompletionsadapter, str) and adapt_dir:
            filename = state.args.chatcompletionsadapter
            if not filename.endswith(".json"):
                filename += ".json"
            #strip to just the filename
            filename = os.path.basename(filename)
            # Case-insensitive match inside adapt_dir
            matched = None
            if adapt_dir:
                for f in os.listdir(adapt_dir):
                    if f.lower().strip() == filename.lower().strip():
                        matched = os.path.join(adapt_dir, f)
                        break
            if matched and os.path.exists(matched):
                ccadapter_path = os.path.abspath(matched)
        if ccadapter_path:
            print(f"Loading Chat Completions Adapter: {ccadapter_path}")
            with open(ccadapter_path, 'r', encoding='utf-8', errors='replace') as f:
                state.chatcompl_adapter = json.load(f)
                canload = True
        else:
            if isinstance(state.args.chatcompletionsadapter, str) and state.args.chatcompletionsadapter!="":
                try:
                    import ast
                    parsed = ast.literal_eval(state.args.chatcompletionsadapter)
                    state.chatcompl_adapter = json.loads(json.dumps(parsed))
                    canload = True
                except Exception as ex:
                    print(ex)
            elif isinstance(state.args.chatcompletionsadapter, dict):
                try:
                    state.chatcompl_adapter = json.loads(json.dumps(state.args.chatcompletionsadapter))
                    canload = True
                except Exception as ex:
                    print(ex)
        if canload:
            print("Chat Completions Adapter Loaded")
        else:
            print("Warning: Chat Completions Adapter invalid or not found.")
        if (state.chatcompl_adapter is not None and isinstance(state.chatcompl_adapter, list)):
            state.chatcompl_adapter_list = state.chatcompl_adapter
            state.chatcompl_adapter = None

    # handle model downloads if needed
    if state.args.model_param and state.args.model_param!="":
        dlfile = download_model_from_url(state.args.model_param,[".gguf",".bin", ".ggml"],min_file_size=500000,handle_multipart=True)
        if dlfile:
            state.args.model_param = dlfile
        if state.args.model and isinstance(state.args.model, list) and len(state.args.model)>1: #handle multi file downloading
            for extramodel in state.args.model[1:]:
                download_model_from_url(extramodel,[".gguf",".bin", ".ggml"],min_file_size=500000)
    if state.args.sdmodel and state.args.sdmodel!="":
        dlfile = download_model_from_url(state.args.sdmodel,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            state.args.sdmodel = dlfile
    if state.args.sdt5xxl and state.args.sdt5xxl!="":
        dlfile = download_model_from_url(state.args.sdt5xxl,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            state.args.sdt5xxl = dlfile
    if state.args.sdclip1 and state.args.sdclip1!="":
        dlfile = download_model_from_url(state.args.sdclip1,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            state.args.sdclip1 = dlfile
    if state.args.sdclip2 and state.args.sdclip2!="":
        dlfile = download_model_from_url(state.args.sdclip2,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            state.args.sdclip2 = dlfile
    if state.args.sdphotomaker and state.args.sdphotomaker!="":
        dlfile = download_model_from_url(state.args.sdphotomaker,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            state.args.sdphotomaker = dlfile
    if state.args.sdupscaler and state.args.sdupscaler!="":
        dlfile = download_model_from_url(state.args.sdupscaler,[".gguf",".safetensors",".pth"],min_file_size=500000)
        if dlfile:
            state.args.sdupscaler = dlfile
    if state.args.sdvae and state.args.sdvae!="":
        dlfile = download_model_from_url(state.args.sdvae,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            state.args.sdvae = dlfile
    if state.args.sdlora and len(state.args.sdlora)>0:
        for i in range(0,len(state.args.sdlora)):
            dlfile = download_model_from_url(state.args.sdlora[i],[".gguf",".safetensors"],min_file_size=500000)
            if dlfile:
                state.args.sdlora[i] = dlfile
    if state.args.mmproj and state.args.mmproj!="":
        dlfile = download_model_from_url(state.args.mmproj,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.mmproj = dlfile
    if state.args.whispermodel and state.args.whispermodel!="":
        dlfile = download_model_from_url(state.args.whispermodel,[".gguf",".bin"],min_file_size=500000)
        if dlfile:
            state.args.whispermodel = dlfile
    if state.args.draftmodel and state.args.draftmodel!="":
        dlfile = download_model_from_url(state.args.draftmodel,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.draftmodel = dlfile
    if state.args.ttsmodel and state.args.ttsmodel!="":
        dlfile = download_model_from_url(state.args.ttsmodel,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.ttsmodel = dlfile
    if state.args.ttswavtokenizer and state.args.ttswavtokenizer!="":
        dlfile = download_model_from_url(state.args.ttswavtokenizer,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.ttswavtokenizer = dlfile
    if state.args.embeddingsmodel and state.args.embeddingsmodel!="":
        dlfile = download_model_from_url(state.args.embeddingsmodel,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.embeddingsmodel = dlfile
    if state.args.musicllm and state.args.musicllm!="":
        dlfile = download_model_from_url(state.args.musicllm,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.musicllm = dlfile
    if state.args.musicembeddings and state.args.musicembeddings!="":
        dlfile = download_model_from_url(state.args.musicembeddings,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.musicembeddings = dlfile
    if state.args.musicdiffusion and state.args.musicdiffusion!="":
        dlfile = download_model_from_url(state.args.musicdiffusion,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.musicdiffusion = dlfile
    if state.args.musicvae and state.args.musicvae!="":
        dlfile = download_model_from_url(state.args.musicvae,[".gguf"],min_file_size=500000)
        if dlfile:
            state.args.musicvae = dlfile
    if state.args.mcpfile and state.args.mcpfile!="":
        dlfile = download_model_from_url(state.args.mcpfile,[".json"],min_file_size=64)
        if dlfile:
            state.args.mcpfile = dlfile
    if state.args.jinjatemplate and state.args.jinjatemplate!="":
        dlfile = download_model_from_url(state.args.jinjatemplate,[".jinja"],min_file_size=64)
        if dlfile:
            state.args.jinjatemplate = dlfile

    if state.args.jinjatemplate and os.path.exists(state.args.jinjatemplate):
        try:
            print(f"Using custom Jinja template: {state.args.jinjatemplate}")
            with open(state.args.jinjatemplate, 'r', encoding='utf-8', errors='ignore') as f:
                state.preloaded_custom_jinja = f.read()
        except Exception as e:
            print(f"Error loading jinja templat: {e}")
            state.preloaded_custom_jinja = ""

    # sanitize and replace the default vanity name. remember me....
    if state.args.model_param and state.args.model_param!="":
        newmdldisplayname = os.path.basename(state.args.model_param)
        newmdldisplayname = os.path.splitext(newmdldisplayname)[0]
        state.friendlymodelname = "koboldcpp/" + sanitize_string(newmdldisplayname)

    # horde worker settings
    if state.args.hordemodelname and state.args.hordemodelname!="":
        state.friendlymodelname = state.args.hordemodelname
        if state.args.debugmode == 1 or state.args.gendefaults:
            state.friendlymodelname = "debug-" + state.friendlymodelname
        if not state.friendlymodelname.startswith("koboldcpp/"):
            state.friendlymodelname = "koboldcpp/" + state.friendlymodelname
    if (state.args.hordemodelname and state.args.hordemodelname!="") or (state.args.hordeworkername and state.args.hordeworkername!="") or (state.args.hordekey and state.args.hordekey!=""):
        if state.args.debugmode == 0:
            state.args.debugmode = -1
    if state.args.hordegenlen and state.args.hordegenlen > 0:
        state.maxhordelen = int(state.args.hordegenlen)
    if state.args.hordemaxctx and state.args.hordemaxctx >= 0:
        state.maxhordectx = int(state.args.hordemaxctx)

    if state.args.debugmode != 1:
        state.showdebug = False
    else:
        state.showdebug = True

    if state.args.multiplayer:
        state.has_multiplayer = True

    if state.args.savedatafile and isinstance(state.args.savedatafile, str):
        filepath = os.path.abspath(state.args.savedatafile)  # Ensure it's an absolute path
        if not filepath.lower().endswith(".jsondb"):
            filepath += ".jsondb"
            state.args.savedatafile += ".jsondb"
        try:
            with open(filepath, 'r+', encoding='utf-8', errors='ignore') as f:
                loaded = json.load(f)
                state.savedata_obj = loaded
                print(f"Loaded existing savedatafile at '{filepath}'.")
        except FileNotFoundError:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w+', encoding='utf-8', errors='ignore') as f:
                    state.savedata_obj = {}
                    print(f"File '{filepath}' did not exist. Created new savedatafile.")
                    json.dump(state.savedata_obj, f)
            except Exception as e:
                print(f"Failed to create savedatafile '{filepath}': {e}")
        except Exception as e:
            print(f"Failed to access savedatafile '{filepath}': {e}")

    if state.args.highpriority:
        print("Setting process to Higher Priority - Use Caution")
        try:
            import psutil
            os_used = sys.platform
            process = psutil.Process(os.getpid())  # Set high priority for the python script for the CPU
            oldprio = process.nice()
            if os.name == 'nt':  # Windows (either 32-bit or 64-bit)
                process.nice(psutil.REALTIME_PRIORITY_CLASS)
                print("High Priority for Windows Set: " + str(oldprio) + " to " + str(process.nice()))
            elif os_used == "linux":  # linux
                process.nice(psutil.IOPRIO_CLASS_RT)
                print("High Priority for Linux Set: " + str(oldprio) + " to " + str(process.nice()))
            else:  # MAC OS X or other
                process.nice(-18)
                print("High Priority for Other OS Set :" + str(oldprio) + " to " + str(process.nice()))
        except Exception as ex:
             print("Error, Could not change process priority: " + str(ex))

    if state.args.contextsize:
        state.maxctx = state.args.contextsize

    state.args.defaultgenamt = max(64, min(state.args.defaultgenamt, 8192))
    state.args.defaultgenamt = min(state.args.defaultgenamt, state.maxctx / 2)

    #this uses the true port instead of the displayport, because we dont want to shut down a router
    if start_server and state.args.singleinstance and is_port_in_use(state.args.port):
        try:
            print(f"Warning: Port {state.args.port} already appears to be in use by another program.")
            print(f"Attempting to request shutdown of previous instance on port {state.args.port}...")
            shutdownreq = make_url_request(f'http://localhost:{state.args.port}/api/extra/shutdown',{},timeout=5)
            shutdownok = (shutdownreq and "success" in shutdownreq and shutdownreq["success"] is True)
            time.sleep(2)
            print("Shutdown existing successful!" if shutdownok else "Shutdown existing failed!")
            time.sleep(1)
        except Exception:
            pass

    if state.args.nocertify:
        import ssl
        state.nocertify = True
        ssl._create_default_https_context = ssl._create_unverified_context

    if state.args.gpulayers:
        if state.args.autofit:
            state.args.gpulayers = -1
        shouldavoidgpu = False
        if state.args.usecpu and sys.platform!="darwin":
            shouldavoidgpu = True
            if state.args.gpulayers and state.args.gpulayers>0:
                print("WARNING: GPU layers is set, but a GPU backend was not selected! GPU will not be used!")
            state.args.gpulayers = 0
        elif state.args.gpulayers==-1 and sys.platform=="darwin" and state.args.model_param and os.path.exists(state.args.model_param):
            print("MacOS detected: Auto GPU layers set to maximum")
            state.args.gpulayers = 200
        elif not shouldavoidgpu and state.args.model_param and os.path.exists(state.args.model_param):
            if (state.args.usecuda is None) and (state.args.usevulkan is None):
                print("No GPU or CPU backend was selected. Trying to assign one for you automatically...")
                auto_set_backend_cli()
            if state.MaxMemory[0] == 0: #try to get gpu vram for cuda if not picked yet
                fetch_gpu_properties(True,True)
                pass
            if state.args.autofit:
                print("Forced autofit is selected, moecpu and overridetensors will be set automatically.")
                state.args.overridetensors = ""
                state.args.moecpu = 0
            if state.args.gpulayers==-1:
                if (not state.args.usecpu) and ((state.args.usecuda is not None) or (state.args.usevulkan is not None) or sys.platform=="darwin"):
                    if state.MaxMemory[0] > 0:
                        extract_modelfile_params(state.args.model_param,state.args.sdmodel,state.args.whispermodel,state.args.mmproj,state.args.draftmodel,state.args.ttsmodel if state.args.ttsgpu else "",state.args.embeddingsmodel if state.args.embeddingsgpu else "", state.args.musicllm, state.args.musicdiffusion)
                        layeramt = autoset_gpu_layers(state.args.contextsize,state.args.sdquant,state.args.batchsize,state.args.musiclowvram)
                        print(f"Auto Recommended GPU Layers: {layeramt}")
                        state.args.gpulayers = layeramt
                    else:
                        print("Unable to detect VRAM, but autofit may still be used if applicable.")
                        state.args.gpulayers = 0
                    # also enable autofit also if permissible
                    if not state.args.autofit and not state.args.tensor_split and not state.args.overridetensors and not state.args.moecpu:
                        state.args.autofit = True
                        state.args.autofitpadding = default_autofit_padding
                        print("GPU layers is default: Will enable AutoFit for increased estimation accuracy.")
                else:
                    print("No GPU backend found, or could not automatically determine GPU layers. You may prefer to set layers manually.")
                    state.args.gpulayers = 0

    if state.args.threads <= 0:
        state.args.threads = get_default_threads()
        print(f"Auto Set Threads: {state.args.threads}")

    print(f"System: {platform.system()} {platform.version()} {platform.machine()} {platform.processor()}")
    if state.MaxMemory[0]>0:
        print(f"Detected Available GPU Memory: {int(state.MaxMemory[0]/1024/1024)} MB")
    else:
        print("Unable to determine GPU Memory")
    try:
        import psutil
        vmem = psutil.virtual_memory()
        print(f"Detected Available RAM: {int(vmem.available/1024/1024)} MB")
    except Exception:
        print("Unable to determine available RAM")

    init_library() # Note: if blas does not exist and is enabled, program will crash.
    print("==========")
    time.sleep(1)

    if state.args.password and state.args.password!="":
        state.password = state.args.password.strip()

    print(state.args)
    print("==========")

    #handle loading text model
    if state.args.model_param:
        if not os.path.exists(state.args.model_param):
            if state.args.ignoremissing:
                print(f"Ignoring missing model file: {state.args.model_param}")
                state.args.model_param = None
            else:
                state.exitcounter = 999
                exit_with_error(2,f"Cannot find text model file: {state.args.model_param}")

        if state.args.lora and state.args.lora[0]!="":
            if not os.path.exists(state.args.lora[0]):
                if state.args.ignoremissing:
                    print(f"Ignoring missing lora file: {state.args.lora[0]}")
                    state.args.lora = None
                else:
                    state.exitcounter = 999
                    exit_with_error(2,f"Cannot find lora file: {state.args.lora[0]}")
            else:
                state.args.lora[0] = os.path.abspath(state.args.lora[0])
                if len(state.args.lora) > 1:
                    if not os.path.exists(state.args.lora[1]):
                        if state.args.ignoremissing:
                            print(f"Ignoring missing lora base: {state.args.lora[1]}")
                            state.args.lora = None
                        else:
                            state.exitcounter = 999
                            exit_with_error(2,f"Cannot find lora base: {state.args.lora[1]}")

                    else:
                        state.args.lora[1] = os.path.abspath(state.args.lora[1])

        if state.args.mmproj and state.args.mmproj!="":
            if not os.path.exists(state.args.mmproj):
                if state.args.ignoremissing:
                    print(f"Ignoring missing mmproj file: {state.args.mmproj}")
                    state.args.mmproj = None
                else:
                    state.exitcounter = 999
                    exit_with_error(2,f"Cannot find mmproj file: {state.args.mmproj}")
            else:
                state.args.mmproj = os.path.abspath(state.args.mmproj)

        if not state.args.blasthreads or state.args.blasthreads <= 0:
            state.args.blasthreads = state.args.threads

        modelname = os.path.abspath(state.args.model_param)

        # Flush stdout for win32 issue with regards to piping in terminals,
        # especially before handing over to C++ context.
        print(f"Loading Text Model: {modelname}", flush=True)
        if not modelname.endswith(".bin") and not modelname.endswith(".gguf"):
            print("WARNING: Selected Text Model does not seem to be a GGUF file! Are you sure you picked the right file?")
        loadok = load_model(modelname)
        print("Load Text Model OK: " + str(loadok))
        if state.args.mmproj and state.args.mmproj!="": # multimodal vision and audio support is only known at runtime
            state.has_audio_support = state.handle.has_audio_support()
            state.has_vision_support = state.handle.has_vision_support()
        else:
            state.has_audio_support = False
            state.has_vision_support = False

        if not loadok:
            state.exitcounter = 999
            exit_with_error(3,"Could not load text model: " + modelname)

        # The chat completions adapter is a list that needs derivation from chat templates
        # Try to derive chat completions adapter from chat template, now that we have the model loaded
        if state.args.model_param:
            ctbytes = state.handle.get_chat_template()
            state.cached_chat_template = ctypes.string_at(ctbytes).decode("UTF-8","ignore")
            if state.cached_chat_template != "" and (state.chatcompl_adapter_list is not None and isinstance(state.chatcompl_adapter_list, list)):
                for entry in state.chatcompl_adapter_list:
                    if all(s in state.cached_chat_template for s in entry['search']):
                        print(f"Chat completion heuristic: {entry['name']}")
                        state.chatcompl_adapter = entry['adapter']
                        break
            state.cached_jinja_kwargs = None
            try:
                jinjakwargsstr = state.args.jinja_kwargs if state.args.jinja_kwargs else None
                if jinjakwargsstr and isinstance(jinjakwargsstr, str):
                    from .sd import parse_json_object
                    state.cached_jinja_kwargs = parse_json_object(jinjakwargsstr,"jinja_kwargs")
                    state.cached_jinja_kwargs = state.cached_jinja_kwargs if state.cached_jinja_kwargs else None
            except Exception:
                print("Jinja Kwargs not valid JSON dict!")
                pass

            if state.chatcompl_adapter is None:
                print("Chat template heuristics failed to identify chat completions format. Alpaca will be used.")

    #handle loading image model
    if state.args.sdmodel and state.args.sdmodel!="":
        imgmodel = state.args.sdmodel
        if not imgmodel or not os.path.exists(imgmodel):
            if state.args.ignoremissing:
                print(f"Ignoring missing img model file: {imgmodel}")
                state.args.sdmodel = None
            else:
                state.exitcounter = 999
                exit_with_error(2,f"Cannot find image model file: {imgmodel}")
        else:
            imgvae = ""
            imgt5xxl = ""
            imgclip1 = ""
            imgclip2 = ""
            imgphotomaker = ""
            imgupscaler = ""
            state.imglora_preload, state.imglora_bypath, state.imglora_name2path = mk_lora_info(state.args.sdlora, state.args.sdloramult)
            if state.args.sdvae:
                if os.path.exists(state.args.sdvae):
                    imgvae = os.path.abspath(state.args.sdvae)
                else:
                    print("Missing SD VAE model file...")
            if state.args.sdt5xxl:
                if os.path.exists(state.args.sdt5xxl):
                    imgt5xxl = os.path.abspath(state.args.sdt5xxl)
                else:
                    print("Missing SD T5-XXL model file...")
            if state.args.sdclip1:
                if os.path.exists(state.args.sdclip1):
                    imgclip1 = os.path.abspath(state.args.sdclip1)
                else:
                    print("Missing SD Clip-1 model file...")
            if state.args.sdclip2:
                if os.path.exists(state.args.sdclip2):
                    imgclip2 = os.path.abspath(state.args.sdclip2)
                else:
                    print("Missing SD Clip-2 model file...")
            if state.args.sdphotomaker:
                if os.path.exists(state.args.sdphotomaker):
                    imgphotomaker = os.path.abspath(state.args.sdphotomaker)
                else:
                    print("Missing SD Photomaker model file...")
            if state.args.sdupscaler:
                if os.path.exists(state.args.sdupscaler):
                    imgupscaler = os.path.abspath(state.args.sdupscaler)
                else:
                    print("Missing SD Upscaler model file...")

            imgmodel = os.path.abspath(imgmodel)
            state.fullsdmodelpath = imgmodel
            state.friendlysdmodelname = os.path.basename(imgmodel)
            state.friendlysdmodelname = os.path.splitext(state.friendlysdmodelname)[0]
            state.friendlysdmodelname = sanitize_string(state.friendlysdmodelname)
            loadok = sd_load_model(imgmodel,imgvae,imgt5xxl,imgclip1,imgclip2,imgphotomaker,imgupscaler)
            state.cached_sd_info = sd_get_info()
            print("Load Image Model OK: " + str(loadok))
            if not loadok:
                state.exitcounter = 999
                exit_with_error(3,"Could not load image model: " + imgmodel)

    #handle whisper model
    if state.args.whispermodel and state.args.whispermodel!="":
        whispermodel = state.args.whispermodel
        if not whispermodel or not os.path.exists(whispermodel):
            if state.args.ignoremissing:
                print(f"Ignoring missing whisper model file: {whispermodel}")
                state.args.whispermodel = None
            else:
                state.exitcounter = 999
                exit_with_error(2,f"Cannot find whisper model file: {whispermodel}")
        else:
            whispermodel = os.path.abspath(whispermodel)
            state.fullwhispermodelpath = whispermodel
            loadok = whisper_load_model(whispermodel)
            print("Load Whisper Model OK: " + str(loadok))
            if not loadok:
                state.exitcounter = 999
                exit_with_error(3,"Could not load whisper model: " + whispermodel)

    #handle tts model
    if state.args.ttsmodel and state.args.ttsmodel!="":
        if not os.path.exists(state.args.ttsmodel) or (state.args.ttswavtokenizer and state.args.ttswavtokenizer!="" and not os.path.exists(state.args.ttswavtokenizer)):
            if state.args.ignoremissing:
                print("Ignoring missing TTS model files!")
                state.args.ttsmodel = None
                state.args.ttswavtokenizer = None
            else:
                state.exitcounter = 999
                exit_with_error(2,f"Cannot find tts model files: {state.args.ttsmodel} or {state.args.ttswavtokenizer}")
        else:
            ttsmodelpath = state.args.ttsmodel
            ttsmodelpath = os.path.abspath(ttsmodelpath)
            state.ttsmodelpath = ttsmodelpath
            wavtokpath = state.args.ttswavtokenizer
            if wavtokpath:
                wavtokpath = os.path.abspath(wavtokpath)
            loadok = tts_load_model(ttsmodelpath,wavtokpath)
            print("Load TTS Model OK: " + str(loadok))
            if not loadok:
                state.exitcounter = 999
                exit_with_error(3,"Could not load TTS model!")

    #handle embeddings model
    if state.args.embeddingsmodel and state.args.embeddingsmodel!="":
        if not os.path.exists(state.args.embeddingsmodel):
            if state.args.ignoremissing:
                print("Ignoring missing TTS model files!")
                state.args.embeddingsmodel = None
            else:
                state.exitcounter = 999
                exit_with_error(2,f"Cannot find embeddings model files: {state.args.embeddingsmodel}")
        else:
            embeddingsmodelpath = state.args.embeddingsmodel
            embeddingsmodelpath = os.path.abspath(embeddingsmodelpath)
            state.embeddingsmodelpath = embeddingsmodelpath
            loadok = embeddings_load_model(embeddingsmodelpath)
            print("Load Embeddings Model OK: " + str(loadok))
            state.friendlyembeddingsmodelname = os.path.basename(embeddingsmodelpath)
            state.friendlyembeddingsmodelname = os.path.splitext(state.friendlyembeddingsmodelname)[0]
            state.friendlyembeddingsmodelname = sanitize_string(state.friendlyembeddingsmodelname)
            if not loadok:
                state.exitcounter = 999
                exit_with_error(3,"Could not load Embeddings model!")

    #handle music model
    mu_has_llm = True if (state.args.musicllm and state.args.musicllm!="") else False
    mu_has_embed = True if  (state.args.musicembeddings and state.args.musicembeddings!="") else False
    mu_has_diff = True if (state.args.musicdiffusion and state.args.musicdiffusion!="") else False
    mu_has_vae = True if (state.args.musicvae and state.args.musicvae!="") else False
    if mu_has_llm or mu_has_embed or mu_has_diff or mu_has_vae:
        if mu_has_llm and not any([mu_has_embed, mu_has_diff, mu_has_vae]):
            if not os.path.exists(state.args.musicllm):
                if state.args.ignoremissing:
                    print("Ignoring missing Music LLM model file!")
                    state.args.musicllm = None
                else:
                    state.exitcounter = 999
                    exit_with_error(2, "Cannot find Music LLM model file!")
            else:
                musicllmpath = os.path.abspath(state.args.musicllm)
                loadok = music_load_model(musicllmpath, "", "", "")
                print("Load Music LLM Only OK: " + str(loadok))
                if not loadok:
                    state.exitcounter = 999
                    exit_with_error(3, "Could not load Music LLM model!")
        elif mu_has_diff:
            if not (mu_has_embed and mu_has_vae):
                state.exitcounter = 999
                exit_with_error(2,"Invalid config: Music Diffusion requires Music embedding and Music VAE models!")

            paths_to_check = [state.args.musicdiffusion,state.args.musicembeddings,state.args.musicvae]
            if mu_has_llm:
                paths_to_check.append(state.args.musicllm)

            if not all(os.path.exists(p) for p in paths_to_check):
                if state.args.ignoremissing:
                    print("Ignoring missing Music model files!")
                    state.args.musicllm = None
                    state.args.musicembeddings = None
                    state.args.musicdiffusion = None
                    state.args.musicvae = None
                else:
                    state.exitcounter = 999
                    exit_with_error(2,"Cannot find required music diffusion/embedding/VAE model files!")
            else:
                state.musicdiffusionmodelpath = os.path.abspath(state.args.musicdiffusion)
                musicembedpath = os.path.abspath(state.args.musicembeddings)
                musicvaepath = os.path.abspath(state.args.musicvae)
                musicllmpath = os.path.abspath(state.args.musicllm) if mu_has_llm else ""
                loadok = music_load_model(musicllmpath,musicembedpath,state.musicdiffusionmodelpath,musicvaepath)
                print("Load Music Models OK: " + str(loadok))
                if not loadok:
                    state.exitcounter = 999
                    exit_with_error(3, "Could not load Music models!")

    #load embedded lite
    embddir = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), '..', "embd_res")
    try:
        with open(os.path.join(embddir, "klite.embd"), mode='rb') as f:
            state.embedded_kailite = f.read()
            # patch it with extra stuff
            patches = [{"find":"Sorry, KoboldAI Lite requires Javascript to function.","replace":"Sorry, KoboldAI Lite requires Javascript to function.<br>You can use <a class=\"color_blueurl\" href=\"/noscript\">KoboldCpp NoScript mode</a> instead."},
                       {"find":"var localflag = urlParams.get('local');","replace":"var localflag = true;"},
                       {"find":"<p id=\"tempgtloadtxt\">Loading...</p>","replace":"<p id=\"tempgtloadtxt\">Loading...<br>(If load fails, try <a class=\"color_blueurl\" href=\"/noscript\">KoboldCpp NoScript mode</a> instead, or adding /noscript at this url.)</p>"}]
            state.embedded_kailite = state.embedded_kailite.decode("UTF-8","ignore")
            for p in patches:
                state.embedded_kailite = state.embedded_kailite.replace(p["find"], p["replace"])
            state.embedded_kailite = state.embedded_kailite.encode()
            state.embedded_kailite_gz = gzip.compress(state.embedded_kailite)
            print("Embedded KoboldAI Lite loaded.")
    except Exception:
        print("Could not find KoboldAI Lite. Embedded KoboldAI Lite will not be available.")

    try:
        with open(os.path.join(embddir, "kcpp_docs.embd"), mode='rb') as f:
            state.embedded_kcpp_docs = f.read()
            state.embedded_kcpp_docs_gz = gzip.compress(state.embedded_kcpp_docs)
            print("Embedded API docs loaded.")
    except Exception:
        print("Could not find Embedded KoboldCpp API docs.")

    try:
        with open(os.path.join(embddir, "kcpp_sdui.embd"), mode='rb') as f:
            state.embedded_kcpp_sdui = f.read()
            state.embedded_kcpp_sdui_gz = gzip.compress(state.embedded_kcpp_sdui)
            if state.args.sdmodel:
                print("Embedded SDUI loaded.")
    except Exception:
        print("Could not find Embedded SDUI.")

    try:
        with open(os.path.join(embddir, "lcpp.gz.embd"), mode='rb') as f:
            state.embedded_lcpp_ui_gz = f.read()
            print("Llama.cpp UI loaded.")
    except Exception:
        print("Could not find Embedded llama.cpp UI.")

    try:
        with open(os.path.join(embddir, "kcpp_musicui.embd"), mode='rb') as f:
            state.embedded_musicui = f.read()
            state.embedded_musicui_gz = gzip.compress(state.embedded_musicui)
            if state.args.musicllm or state.args.musicdiffusion or state.args.ttsmodel:
                print("Embedded MusicUI loaded.")
    except Exception:
        print("Could not find Embedded MusicUI.")

    # load all TTS audio files
    if state.args.ttsmodel or state.ttsName is not None:
        try:
            state.voicebank = {}
            voicecount = 0
            state.voicelist = []

            try:
                with open(os.path.join(embddir, "qwen3tts_voices_json.embd"), mode='r', encoding='utf-8', errors='ignore') as f:
                    vdict = json.load(f)
                    for key, value in vdict.items():
                        state.voicelist.append(key)
                        state.voicebank[key] = value
            except Exception:
                print("Could not find Embedded Qwen3TTS voices.")

            state.voicelist.append("random")
            state.voicebank["random"] = ""
            state.voicelist.append("instruct")
            state.voicebank["instruct"] = ""

            if state.args.ttsdir and os.path.isdir(state.args.ttsdir):
                for filename in os.listdir(state.args.ttsdir):
                    if filename.lower().endswith((".mp3", ".wav")):
                        full_path = os.path.join(state.args.ttsdir, filename)
                        with open(full_path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                            state.voicebank[filename] = encoded
                            voicecount += 1
                            state.voicelist.append(os.path.basename(filename))
            print(f"Loaded {voicecount} TTS voices.")
        except Exception:
            print("Could not load TTS voices.")

    if state.args.mcpfile and isinstance(state.args.mcpfile, str):
        threading.Thread(target=load_mcp_async, args=(state.args,), daemon=True).start()
        time.sleep(0.2) # short delay to allow get_capabilities to work

    # print enabled modules
    caps = get_capabilities()
    enabledmlist = []
    disabledmlist = []
    apimlist = ["KoboldCppApi"]
    if "llm" in caps and caps["llm"]:
        apimlist.append("OpenAiApi")
        apimlist.append("OllamaApi")
        apimlist.append("AnthropicApi")
    if "txt2img" in caps and caps["txt2img"]:
        apimlist.append("A1111ForgeApi")
        apimlist.append("ComfyUiApi")
    if "transcribe" in caps and caps["transcribe"]:
        apimlist.append("WhisperTranscribeApi")
    if "tts" in caps and caps["tts"]:
        apimlist.append("XttsApi")
        apimlist.append("OpenAiSpeechApi")
    enabledmlist.append("TextGeneration") if "llm" in caps and caps["llm"] else disabledmlist.append("TextGeneration")
    enabledmlist.append("ImageGeneration") if "txt2img" in caps and caps["txt2img"] else disabledmlist.append("ImageGeneration")
    enabledmlist.append("VoiceRecognition") if "transcribe" in caps and caps["transcribe"] else disabledmlist.append("VoiceRecognition")
    enabledmlist.append("MultimodalVision") if "vision" in caps and caps["vision"] else disabledmlist.append("MultimodalVision")
    enabledmlist.append("MultimodalAudio") if "audio" in caps and caps["audio"] else disabledmlist.append("MultimodalAudio")
    enabledmlist.append("NetworkMultiplayer") if "multiplayer" in caps and caps["multiplayer"] else disabledmlist.append("NetworkMultiplayer")
    enabledmlist.append("ApiKeyPassword") if "protected" in caps and caps["protected"] else disabledmlist.append("ApiKeyPassword")
    enabledmlist.append("WebSearchProxy") if "websearch" in caps and caps["websearch"] else disabledmlist.append("WebSearchProxy")
    enabledmlist.append("TextToSpeech") if "tts" in caps and caps["tts"] else disabledmlist.append("TextToSpeech")
    enabledmlist.append("VectorEmbeddings") if "embeddings" in caps and caps["embeddings"] else disabledmlist.append("VectorEmbeddings")
    enabledmlist.append("AdminControl") if "admin" in caps and caps["admin"]!=0 else disabledmlist.append("AdminControl")
    enabledmlist.append("MCPBridge") if "mcp" in caps and caps["mcp"] else disabledmlist.append("MCPBridge")
    enabledmlist.append("MusicGen") if "music" in caps and caps["music"] else disabledmlist.append("MusicGen")
    enabledmlist.append("RouterMode") if "router" in caps and caps["router"] else disabledmlist.append("RouterMode")

    print(f"======\nActive Modules: {' '.join(enabledmlist)}")
    print(f"Inactive Modules: {' '.join(disabledmlist)}")
    if not state.args.cli:
        print(f"Enabled APIs: {' '.join(apimlist)}")

    if state.args.ssl:
        if len(state.args.ssl)==2 and isinstance(state.args.ssl[0], str) and os.path.exists(state.args.ssl[0]) and isinstance(state.args.ssl[1], str) and os.path.exists(state.args.ssl[1]):
            state.sslvalid = True
            print("SSL configuration is valid and will be used.")
        else:
            print("Your SSL configuration is INVALID. SSL will not be used.")
    endpoint_url = ""
    remote_url = ""
    httpsaffix = ("https" if state.sslvalid else "http")
    displayedport = (state.args.port if not state.args.proxy_port else state.args.proxy_port)
    if state.args.host=="":
        endpoint_url = f"{httpsaffix}://localhost:{displayedport}"
    else:
        endpoint_url = f"{httpsaffix}://{state.args.host}:{displayedport}"

    if start_server:
        if not state.args.remotetunnel:
            if displayedport!=11434:
                print("Note: For third party Ollama API Emulation, you should set the port to 11434.")
            else:
                print("Ollama Emulation is now available at port 11434.")
            print(f"Starting Kobold API on port {displayedport} at {endpoint_url}/api/")
            print(f"Starting OpenAI Compatible API on port {displayedport} at {endpoint_url}/v1/")
            print(f"Starting llama.cpp secondary WebUI at {endpoint_url}/lcpp/")
            if state.args.sdmodel:
                print(f"StableUI is available at {endpoint_url}/sdui/")
            if state.args.musicdiffusion or state.args.musicllm or state.args.ttsmodel:
                print(f"MusicUI is available at {endpoint_url}/musicui/")
        elif state.global_memory:
            val = state.global_memory["tunnel_url"]
            if val:
                endpoint_url = val
                remote_url = val
                print(f"Your remote Kobold API can be found at {endpoint_url}/api")
                print(f"Your remote OpenAI Compatible API can be found at {endpoint_url}/v1")
                print(f"Starting llama.cpp secondary WebUI at {endpoint_url}/lcpp/")
                if state.args.sdmodel:
                    print(f"StableUI is available at {endpoint_url}/sdui/")
                if state.args.musicdiffusion or state.args.musicllm or state.args.ttsmodel:
                    print(f"MusicUI is available at {endpoint_url}/musicui/")
            state.global_memory["load_complete"] = True
        if state.args.launch:
            def launch_browser_thread():
                LaunchWebbrowser(endpoint_url,"--launch was set, but could not launch web browser automatically.")
            browser_thread = threading.Timer(2, launch_browser_thread) #2 second delay
            browser_thread.start()

        if state.args.hordekey and state.args.hordekey!="":
            if state.args.hordeworkername and state.args.hordeworkername!="":
                workers_to_use = 1
                if state.args.continuous_batching:
                    workers_to_use = int(state.args.continuous_batching) if hasattr(state.args, "continuous_batching") else 1
                for w in range(0,workers_to_use):
                    wid = (w+1)
                    print(f"Launching horde worker {wid}...")
                    horde_thread = threading.Thread(target=run_horde_worker,args=(state.args,state.args.hordekey,state.args.hordeworkername,wid,workers_to_use))
                    horde_thread.daemon = True
                    horde_thread.start()
            else:
                print("Horde worker could not start. You need to specify a horde worker name with --hordeworkername")

    #if post-ready script specified, execute it
    if state.args.onready:
        def onready_subprocess():
            print("Starting Post-Load subprocess...")
            subprocess.run(state.args.onready[0], shell=True)
        timer_thread = threading.Timer(1, onready_subprocess) #1 second delay
        timer_thread.start()

    if not start_server:
        if state.args.cli:
            print("\n===\nNow running KoboldCpp in Interactive Terminal Chat mode.\nType /quit or /exit to end session.\n")
            lastturns = []
            if state.args.prompt and state.args.prompt!="":
                lastturns.append({"role":"system","content":state.args.prompt})
                print(f"System Prompt:\n{state.args.prompt}\n")
            while True:
                lastuserinput = input("> ")
                if lastuserinput=="/quit" or lastuserinput=="/exit":
                    break
                if not lastuserinput:
                    continue
                lastturns.append({"role":"user","content":lastuserinput})
                payload = {"messages":lastturns,"rep_pen":1.07,"temperature":0.8}
                payload = transform_genparams(payload, 4, False) #to chat completions
                if state.args.debugmode < 1:
                    suppress_stdout()
                genout = generate(genparams=payload)
                if state.args.debugmode < 1:
                    restore_stdout()
                result = (genout["text"] if "text" in genout else "")
                if result:
                    lastturns.append({"role":"assistant","content":result})
                    print(result.strip() + "\n", flush=True)
                else:
                    print("(No Response Received)\n", flush=True)
        else:
            save_to_file = (state.args.benchmark and state.args.benchmark!="stdout" and state.args.benchmark!="")
            benchmaxctx = state.maxctx
            benchlen = state.args.genlimit if state.args.genlimit > 0 else 100
            benchtemp = 0.1
            benchtopk = 1
            benchreppen = 1
            benchbaneos = True
            benchmodel = sanitize_string(os.path.splitext(os.path.basename(modelname))[0])
            benchprompt = ""
            if state.args.prompt:
                benchprompt = state.args.prompt
                benchtopk = 100
                benchreppen = 1.07
                benchtemp = 0.8
                if not state.args.benchmark:
                    benchbaneos = False
            if state.args.benchmark:
                if os.path.exists(state.args.benchmark) and os.path.getsize(state.args.benchmark) > 1000000:
                    print("\nWarning: The benchmark CSV output file you selected exceeds 1MB. This is probably not what you want, did you select the wrong CSV file?\nFor safety, benchmark output will not be saved.")
                    save_to_file = False
                if save_to_file:
                    print(f"\nRunning benchmark (Save to File: {state.args.benchmark})...")
                else:
                    print("\nRunning benchmark (Not Saved)...")
                if benchprompt=="":
                    benchprompt = " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
                    for i in range(0,14): #generate massive prompt
                        benchprompt += benchprompt
            genp = {
                "prompt":benchprompt,
                "max_length":benchlen,
                "max_context_length":benchmaxctx,
                "temperature":benchtemp,
                "top_k":benchtopk,
                "rep_pen":benchreppen,
                "ban_eos_token":benchbaneos
            }
            genout = generate(genparams=genp)
            result = genout['text']
            if state.args.prompt and not state.args.benchmark:
                restore_stdout()
                print(result)
            if state.args.benchmark:
                result = (result[:8] if len(result)>8 else "") if not state.args.prompt else result
                t_pp = float(state.handle.get_last_process_time())*float(benchmaxctx-benchlen)*0.001
                t_gen = float(state.handle.get_last_eval_time())*float(benchlen)*0.001
                s_pp = float(benchmaxctx-benchlen)/t_pp
                s_gen = float(benchlen)/t_gen
                datetimestamp = datetime.now(timezone.utc)
                benchflagstr = f"{str(vars(state.args))}"
                print(f"\nBenchmark Completed - v{KcppVersion} Results:\n======")
                print(f"Flags: {benchflagstr}")
                print(f"Timestamp: {datetimestamp}")
                print(f"Backend: {state.libname}")
                print(f"Layers: {state.args.gpulayers if not state.args.autofit else 'Autofit'}")
                print(f"Model: {benchmodel}")
                print(f"MaxCtx: {benchmaxctx}")
                print(f"GenAmount: {benchlen}\n-----")
                print(f"ProcessingTime: {t_pp:.3f}s")
                print(f"ProcessingSpeed: {s_pp:.2f}T/s")
                print(f"GenerationTime: {t_gen:.3f}s")
                print(f"GenerationSpeed: {s_gen:.2f}T/s")
                print(f"TotalTime: {(t_pp+t_gen):.3f}s")
                print(f"Output: {result}\n-----")
                if save_to_file:
                    try:
                        with open(state.args.benchmark, "a") as file:
                            file.seek(0, 2)
                            if file.tell() == 0: #empty file
                                file.write("Timestamp,Backend,Layers,Model,MaxCtx,GenAmount,ProcessingTime,ProcessingSpeed,GenerationTime,GenerationSpeed,TotalTime,Output,Flags")
                            file.write(f"\n{datetimestamp},{state.libname},{state.args.gpulayers},{benchmodel},{benchmaxctx},{benchlen},{t_pp:.2f},{s_pp:.2f},{t_gen:.2f},{s_gen:.2f},{(t_pp+t_gen):.2f},{result},\"{benchflagstr}\"")
                    except Exception as e:
                        print(f"Error writing benchmark to file: {e}")
                if state.global_memory and state.using_gui_launcher and not save_to_file:
                    state.global_memory["input_to_exit"] = True
                    time.sleep(1)

    if start_server:
        if state.args.remotetunnel:
            if remote_url:
                print(f"======\nYour remote tunnel is ready, please connect to {remote_url}", flush=True)
        else:
            # Flush stdout for previous win32 issue so the client can see output.
            print(f"======\nPlease connect to custom endpoint at {endpoint_url}", flush=True)
        asyncio.run(RunServerMultiThreaded(state.args.host, state.args.port, KcppServerRequestHandler))
    else:
        # Flush stdout for previous win32 issue so the client can see output.
        if not state.args.prompt or state.args.benchmark or state.args.cli:
            print("Server was not started, main function complete. Idling.", flush=True)


# ---------------------------------------------------------------------------
# Argparse setup  (only runs when this module is the entry point)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    multiprocessing.freeze_support()

    def check_range(value_type, min_value, max_value):
        def range_checker(arg: str):
            try:
                f = value_type(arg)
            except ValueError:
                raise argparse.ArgumentTypeError(f'must be a valid {value_type}')
            if f < min_value or f > max_value:
                raise argparse.ArgumentTypeError(f'must be within [{min_value}, {max_value}]')
            return f
        return range_checker

    parser = argparse.ArgumentParser(description=f'KoboldCpp Server - Version {KcppVersion}')
    modelgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    modelgroup.add_argument("--model","-m", metavar=('[filenames]'), help="Model file to load. Accepts multiple values if they are URLs.", type=str, nargs='+', default=[])
    modelgroup.add_argument("model_param", help="Model file to load (positional)", nargs="?")
    portgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    portgroup.add_argument("--port", metavar=('[portnumber]'), help=f"Port to listen on. (Defaults to {defaultport})", default=defaultport, type=int, action='store')
    portgroup.add_argument("port_param", help="Port to listen on (positional)", default=defaultport, nargs="?", type=int, action='store')
    parser.add_argument("--host", metavar=('[ipaddr]'), help="Host IP to listen on. If this flag is not set, all routable interfaces are accepted.", default="")
    parser.add_argument("--launch", help="Launches a web browser when load is completed.", action='store_true')
    parser.add_argument("--config", metavar=('[filename]'), help="Load settings from a .kcpps file. Other arguments will be ignored", type=str, nargs=1)
    parser.add_argument("--threads","-t", metavar=('[threads]'), help="Use a custom number of threads if specified. Otherwise, uses an amount based on CPU cores", type=int, default=get_default_threads())
    compatgroup = parser.add_mutually_exclusive_group()
    compatgroup.add_argument("--usecuda", "--usecublas", "--usehipblas", help="Use CUDA for GPU Acceleration. Requires CUDA. Enter a number afterwards to select and use 1 GPU. Leaving no number will use all GPUs.", nargs='*',metavar=('[main GPU ID]'), choices=['0','1','2','3','all',  'mmq','nommq','normal','lowvram','rowsplit'])
    compatgroup.add_argument("--usevulkan", help="Use Vulkan for GPU Acceleration. Can optionally specify one or more GPU Device ID (e.g. --usevulkan 0), leave blank to autodetect.", metavar=('[Device IDs]'), nargs='*', type=int, default=None)
    compatgroup.add_argument("--usecpu", help="Do not use any GPU acceleration (CPU Only)", action='store_true')
    parser.add_argument("--contextsize","--ctx-size", "-c", help="Controls the memory allocated for maximum context size, only change if you need more RAM for big contexts. (default 8192).",metavar=('[256 to 262144]'), type=check_range(int,256,262144), default=8192)
    parser.add_argument("--gpulayers","--gpu-layers","--n-gpu-layers","-ngl", help="Set number of layers to offload to GPU when using GPU. Requires GPU. Set to -1 to try autodetect, set to 0 to disable GPU offload.",metavar=('[GPU layers]'), nargs='?', const=1, type=int, default=-1)
    parser.add_argument("--tensor_split","--tensorsplit","--tensor-split","-ts", help="For CUDA and Vulkan only, ratio to split tensors across multiple GPUs, space-separated list of proportions, e.g. 7 3", metavar=('[Ratios]'), type=float, nargs='+')
    parser.add_argument("--autofit","--fit","-fit", help="Automatically attempt to fit the model in the best possible way. Overrides everything else. Experimental.", action='store_true')

    #more advanced params
    advparser = parser.add_argument_group('Advanced Commands')
    advparser.add_argument("--version", help="Prints version and exits.", action='store_true')
    advparser.add_argument("--analyze", metavar=('[filename]'), help="Reads the metadata, weight types and tensor names in any GGUF file.", default="")
    advparser.add_argument("--maingpu","--main-gpu","-mg", help="Only used in a multi-gpu setup. Sets the index of the main GPU that will be used.",metavar=('[Device ID]'), type=int, default=-1)
    advparser.add_argument("--batchsize","--blasbatchsize","--batch-size","-b", help="Sets the batch size used in batched processing (default 512). Setting it to -1 disables batched mode, but keeps other benefits like GPU offload.", type=int,choices=[-1,16,32,64,128,256,512,1024,2048,4096], default=512)
    advparser.add_argument("--continuous-batching","--contbatch", help=argparse.SUPPRESS, metavar=('[slots]'), type=check_range(int,0,64), default=0)
    advparser.add_argument("--blasthreads","--batchthreads","--threadsbatch","--threads-batch", help="Use a different number of threads during batching if specified. Otherwise, has the same value as --threads",metavar=('[threads]'), type=int, default=0)
    advparser.add_argument("--splitmode","-sm","--split-mode", help="How to split the model across multiple GPUs", metavar=('[split mode]'), type=str, choices=splitmode_choices, default=splitmode_choices[0])
    advparser.add_argument("--nommq", help="Disables MMQ, only used for cuda backend. This flag may be removed in future.", action='store_true')
    advparser.add_argument("--lora", help="GGUF models only, applies a lora file on top of model.", metavar=('[lora_filename]'), nargs='+')
    advparser.add_argument("--loramult", metavar=('[amount]'), help="Multiplier for the Text LORA model to be applied.", type=float, default=1.0)
    advparser.add_argument("--noshift","--no-context-shift", help="If set, do not attempt to Trim and Shift the GGUF context.", action='store_true')
    advparser.add_argument("--nofastforward", help="If set, do not attempt to fast forward GGUF context (always reprocess). Will also enable noshift", action='store_true')
    advparser.add_argument("--useswa", help="If set, allows Sliding Window Attention (SWA) KV Cache, which saves memory but cannot be used with context shifting.", action='store_true')
    advparser.add_argument("--swapadding", help="How much extra to pad the SWA KV cache, this affects the rewind limit before reprocessing is forced.", type=int, default=swa_padding_default)
    advparser.add_argument("--smartcache", help="Enables intelligent context switching by saving KV cache snapshots to RAM. Requires fast forwarding.", metavar=('limit'), nargs='?', const=1, type=int, default=0)
    advparser.add_argument("--ropeconfig", help="If set, uses customized RoPE scaling from configured frequency scale and frequency base (e.g. --ropeconfig 0.25 10000). Otherwise, uses NTK-Aware scaling set automatically based on context size. For linear rope, simply set the freq-scale and ignore the freq-base",metavar=('[rope-freq-scale]', '[rope-freq-base]'), default=[0.0, 10000.0], type=float, nargs='+')
    advparser.add_argument("--overridenativecontext", help="Overrides the native trained context of the loaded model with a custom value to be used for Rope scaling.",metavar=('[trained context]'), type=int, default=0)
    compatgroup3 = advparser.add_mutually_exclusive_group()
    compatgroup3.add_argument("--usemmap", help="If set, uses mmap to load model.", action='store_true')
    advparser.add_argument("--usemlock","--mlock", help="Enables mlock, preventing the RAM used to load the model from being paged out. Not usually recommended.", action='store_true')
    advparser.add_argument("--noavx2", help="Do not use AVX2 instructions, a slower compatibility mode for older devices.", action='store_true')
    advparser.add_argument("--failsafe", help="Use failsafe mode, extremely old CPU compatibility mode that should work on all devices.", action='store_true')
    advparser.add_argument("--debugmode", help="Shows additional debug info in the terminal. Levels: -1 (Horde-quiet, suppresses non-essential prints; auto-applied when Horde args are set), 0 (default, normal output), 1 (verbose: extra slot/cache info, larger print buffers, retains horde-debug prefix). Passing the flag without a value implies 1.", nargs='?', const=1, type=int, default=0)
    advparser.add_argument("--onready", help="An optional shell command to execute after the model has been loaded.", metavar=('[shell command]'), type=str, default="",nargs=1)
    advparser.add_argument("--benchmark", help="Do not start server, instead run benchmarks. If filename is provided, appends results to provided file.", metavar=('[filename]'), nargs='?', const="stdout", type=str, default=None)
    advparser.add_argument("--prompt","-p", metavar=('[prompt]'), help="Passing a prompt string triggers a direct inference, loading the model, outputs the response to stdout and exits. Can be used alone or with benchmark.", type=str, default="")
    advparser.add_argument("--cli", help="Does not launch KoboldCpp HTTP server. Instead, enables KoboldCpp from the command line, accepting interactive console input and displaying responses to the terminal.", action='store_true')
    advparser.add_argument("--genlimit","--promptlimit", help="Sets the maximum number of generated tokens, it will restrict all generations to this or lower. Also usable with --prompt or --benchmark.",metavar=('[token limit]'), type=int, default=0)
    advparser.add_argument("--multiuser", help="Set maximum number of queued incoming requests allowed.", metavar=('limit'), type=int, nargs='?', const=multiuser_concurrent_limit, default=multiuser_concurrent_limit)
    advparser.add_argument("--multiplayer", help="Hosts a shared multiplayer session that others can join.", action='store_true')
    advparser.add_argument("--websearch", help="Enable the local search engine proxy so Web Searches can be done.", action='store_true')
    advparser.add_argument("--remotetunnel", help="Uses Cloudflare to create a remote tunnel, allowing you to access koboldcpp remotely over the internet even behind a firewall.", action='store_true')
    advparser.add_argument("--highpriority", help="Experimental flag. If set, increases the process CPU priority, potentially speeding up generation. Use caution.", action='store_true')
    advparser.add_argument("--foreground", help="Windows only. Sends the terminal to the foreground every time a new prompt is generated. This helps avoid some idle slowdown issues.", action='store_true')
    advparser.add_argument("--preloadstory", metavar=('[savefile]'), help="Configures a prepared story json save file to be hosted on the server, which frontends (such as KoboldAI Lite) can access over the API.", default="")
    advparser.add_argument("--savedatafile", metavar=('[savefile]'), help="If enabled, creates or opens a persistent database file on the server, that allows users to save and load their data remotely. A new file is created if it does not exist.", default="")
    advparser.add_argument("--quiet", help="Enable quiet mode, which hides generation inputs and outputs in the terminal. Quiet mode is automatically enabled when running a horde worker.", action='store_true')
    advparser.add_argument("--ssl", help="Allows all content to be served over SSL instead. A valid UNENCRYPTED SSL cert and key .pem files must be provided", metavar=('[cert_pem]', '[key_pem]'), nargs='+')
    advparser.add_argument("--nocertify", help="Allows insecure SSL connections. Use this if you have cert errors and need to bypass certificate restrictions.", action='store_true')
    advparser.add_argument("--mmproj", metavar=('[filename]'), help="Select a multimodal projector file for vision models like LLaVA.", default="")
    advparser.add_argument("--mmprojcpu","--no-mmproj-offload", help="Force CLIP for Vision mmproj always on CPU.", action='store_true')
    advparser.add_argument("--visionmaxres", metavar=('[max px]'), help="Clamp MMProj vision maximum allowed resolution. Allowed values are between 512 to 2048 px (default 1024).", type=int, default=default_visionmaxres)
    advparser.add_argument("--visionmintokens","--image-min-tokens", metavar=('[tokens]'), help="Override the minimum tokens for the MMProj embedding (default -1).", type=int, default=-1)
    advparser.add_argument("--visionmaxtokens","--image-max-tokens", metavar=('[tokens]'), help="Override the maximum tokens for the MMProj embedding (default -1).", type=int, default=-1)
    advparser.add_argument("--draftmodel","--model-draft","-md", metavar=('[filename]'), help="Load a small draft model for speculative decoding. It will be fully offloaded. Vocab must match the main model.", default="")
    advparser.add_argument("--draftamount","--draft-max","--draft-n", metavar=('[tokens]'), help="How many tokens to draft per chunk before verifying results", type=int, default=default_draft_amount)
    advparser.add_argument("--draftgpulayers","--gpu-layers-draft","--n-gpu-layers-draft","-ngld", metavar=('[layers]'), help="How many layers to offload to GPU for the draft model (default=full offload)", type=int, default=999)
    advparser.add_argument("--draftgpusplit", help="GPU layer distribution ratio for draft model (default=same as main). Only works if multi-GPUs selected for MAIN model and tensor_split is set!", metavar=('[Ratios]'), type=float, nargs='+')
    advparser.add_argument("--password", metavar=('[API key]'), help="Enter a password required to use this instance. This key will be required for all text endpoints. Image endpoints are not secured. Can also be set with env var KCPP_PASSWORD", default=os.getenv('KCPP_PASSWORD',None))
    advparser.add_argument("--ratelimit", metavar=('[seconds]'), help="If enabled, rate limit generative request by IP address. Each IP can only send a new request once per X seconds.", type=int, default=0)
    advparser.add_argument("--ignoremissing", help="Ignores all missing non-essential files, just skipping them instead.", action='store_true')
    advparser.add_argument("--chatcompletionsadapter", metavar=('[filename]'), help="Select an optional ChatCompletions Adapter JSON file to force custom instruct tags.", default="AutoGuess")
    advparser.add_argument("--jinja", help="Enables using jinja chat template formatting for chat completions endpoint. Other endpoints are unaffected. Tool calls are done without jinja.", action='store_true')
    advparser.add_argument("--jinja_tools","--jinja-tools","--jinjatools", help="Enables using jinja chat template formatting for chat completions endpoint. Other endpoints are unaffected. Tool calls are done with jinja.", action='store_true')
    advparser.add_argument("--jinja_kwargs","--jinja-kwargs","--jinjakwargs","--chat-template-kwargs", metavar=('{"parameter":"value",...}'), help="Set additional fields for Jinja JSON template parser, must be a valid JSON object.", default="")
    advparser.add_argument("--jinjatemplate","--chat-template-file", metavar=('[filename]'), help="Select a custom Jinja chat template, will overwrite model jinja chat template", default="")
    advparser.add_argument("--noflashattention","--no-flash-attn","-nofa", help="Disables flash attention.", action='store_true')
    advparser.add_argument("--lowvram","-nkvo","--no-kv-offload", help="If supported by the backend, do not offload KV to GPU (lowvram mode). Not recommended, will be slow.", action='store_true')
    advparser.add_argument("--quantkv", help="Sets the KV cache data type quantization, options are f16/bf16/q8_0/q5_1/q4_0. Requires Flash Attention for full effect, otherwise only K cache is quantized.",metavar=('[quantization level f16/bf16/q8_0/q5_1/q4_0]'), type=str, choices=["f16","bf16","q8_0","q5_1","q4_0","0","1","2","3"], default="f16")
    advparser.add_argument("--smartcontext", help="Reserving a portion of context to try processing less frequently. Outdated. Not recommended.", action='store_true')
    advparser.add_argument("--unpack", help="Extracts the file contents of the KoboldCpp binary into a target directory.", metavar=('destination'), type=str, default="")
    advparser.add_argument("--exportconfig", help="Exports the current selected arguments as a .kcpps settings file", metavar=('[filename]'), type=str, default="")
    advparser.add_argument("--exporttemplate", help="Exports the current selected arguments as a .kcppt template file", metavar=('[filename]'), type=str, default="")
    advparser.add_argument("--nomodel", help="Allows you to launch the GUI alone, without selecting any model.", action='store_true')
    advparser.add_argument("--moeexperts", metavar=('[num of experts]'), help="How many experts to use for MoE models (default=follow gguf)", type=int, default=-1)
    advparser.add_argument("--moecpu","--n-cpu-moe", "-ncmoe", metavar=('[layers affected]'), help="Keep the Mixture of Experts (MoE) weights of the first N layers in the CPU. If no value is provided, applies to all layers.", nargs='?', const=999, type=int, default=0)
    advparser.add_argument("--defaultgenamt", help="How many tokens to generate by default, if not specified. Must be smaller than context size. Usually, your frontend GUI will override this.", type=check_range(int,64,8192), default=default_genlen)
    advparser.add_argument("--nobostoken", help="Prevents BOS token from being added at the start of any prompt. Usually NOT recommended for most models.", action='store_true')
    advparser.add_argument("--enableguidance", help="Enables the use of Classifier-Free-Guidance, which allows the use of negative prompts. Has performance and memory impact.", action='store_true')
    advparser.add_argument("--maxrequestsize", metavar=('[size in MB]'), help="Specify a max request payload size. Any requests to the server larger than this size will be dropped. Do not change if unsure.", type=int, default=32)
    advparser.add_argument("--overridekv","--override-kv", metavar=('[name=type:value]'), help="Override metadata value by key. Separate multiple values with commas. Format is name=type:value. Types: int, float, bool, str", default="")
    advparser.add_argument("--overridetensors","--override-tensor","-ot", metavar=('[tensor name pattern=buffer type]'), help="Override selected backend for specific tensors matching tensor_name_regex_pattern=buffer_type, same as in llama.cpp.", default="")
    compatgroup2 = parser.add_mutually_exclusive_group()
    compatgroup2.add_argument("--showgui", help="Always show the GUI instead of launching the model right away when loading settings from a .kcpps file.", action='store_true')
    compatgroup2.add_argument("--skiplauncher", help="Doesn't display or use the GUI launcher. Overrides showgui.", action='store_true')
    advparser.add_argument("--singleinstance", help="Allows this KoboldCpp instance to be shut down by any new instance requesting the same port, preventing duplicate servers from clashing on a port.", action='store_true')
    advparser.add_argument("--nopipelineparallel", help="Disable Pipeline Parallelism. Pipeline Parallelism provides faster multigpu speeds but using more memory, only active for multigpu.", action='store_true')
    advparser.add_argument("--gendefaults", metavar=('{"parameter":"value",...}'), help="Sets extra default parameters for some fields in API requests, as a JSON string.", default="")
    advparser.add_argument("--gendefaultsoverwrite", help="Allow the gendefaults parameters to overwrite the original value in API payloads.", action='store_true')
    advparser.add_argument("--mcpfile", metavar=('[mcp json file]'), help="Specify path to mcp.json which contains the Cladue Desktop compatible MCP server config.", default="")
    advparser.add_argument("--device", "-dev", metavar=('<dev1,dev2,..>'), help="Set llama.cpp compatible device selection override. Comma separated. Overrides normal device choices.", default="")
    advparser.add_argument("--downloaddir", metavar=('[directory]'), help="Specify a directory that models will be downloaded to or searched from, if unset uses the working directory.", default="")
    advparser.add_argument("--autofitpadding", metavar=('[padding in MB]'), help="How much spare allowance in MB should autofit reserve? If it's too little, the load might fail.", type=int, default=default_autofit_padding)

    hordeparsergroup = parser.add_argument_group('Horde Worker Commands')
    hordeparsergroup.add_argument("--hordemodelname", metavar=('[name]'), help="Sets your AI Horde display model name.", default="")
    hordeparsergroup.add_argument("--hordeworkername", metavar=('[name]'), help="Sets your AI Horde worker name.", default="")
    hordeparsergroup.add_argument("--hordekey", metavar=('[apikey]'), help="Sets your AI Horde API key.", default="")
    hordeparsergroup.add_argument("--hordemaxctx", metavar=('[amount]'), help="Sets the maximum context length your worker will accept from an AI Horde job. If 0, matches main context limit.", type=int, default=0)
    hordeparsergroup.add_argument("--hordegenlen", metavar=('[amount]'), help="Sets the maximum number of tokens your worker will generate from an AI horde job.", type=int, default=0)

    sdparsergroup = parser.add_argument_group('Image Generation Commands')
    sdparsergroup.add_argument("--sdmodel", metavar=('[filename]'), help="Specify an image generation safetensors or gguf model to enable image generation.", default="")
    sdparsergroup.add_argument("--sdthreads", metavar=('[threads]'), help="Use a different number of threads for image generation if specified. Otherwise, has the same value as --threads.", type=int, default=0)
    sdparsergroup.add_argument("--sdclamped", metavar=('[maxres]'), help="If specified, limit generation steps and image size for shared use. Accepts an extra optional parameter that indicates maximum resolution (eg. 768 clamps to 768x768, min 512px, disabled if 0).", nargs='?', const=512, type=int, default=0)
    sdparsergroup.add_argument("--sdclampedsoft", metavar=('[maxres]'), help="If specified, limit max image size to curb memory usage. Similar to --sdclamped, but less strict, allows trade-offs between width and height (e.g. 640 would allow 640x640, 512x768 and 768x512 images).", type=int, default=0)
    sdparsergroup.add_argument("--sdt5xxl", metavar=('[filename]'), help="Specify a T5-XXL safetensors model. Leave blank if prebaked or unused.", default="")
    sdparsergroup.add_argument("--sdclip1", "--sdclipl", metavar=('[filename]'), help="Specify first safetensors Clip model (SD3 or Flux Clip-L, WAN or QwenImg vision). Leave blank if prebaked or unused.", default="")
    sdparsergroup.add_argument("--sdclip2", "--sdclipg", metavar=('[filename]'), help="Specify second safetensors Clip model (SD3 Clip-G). Leave blank if prebaked or unused.", default="")
    sdparsergroup.add_argument("--sdphotomaker", metavar=('[filename]'), help="PhotoMaker is a model that allows face cloning. Specify a PhotoMaker safetensors model which will be applied replacing img2img. SDXL models only. Leave blank if unused.", default="")
    sdparsergroup.add_argument("--sdupscaler", metavar=('[filename]'), help="You can use ESRGAN as an upscaling model to resize images. Leave blank if unused.", default="")
    sdparsergroup.add_argument("--sdflashattention", help="Enables Flash Attention for image generation.", action='store_true')
    sdparsergroup.add_argument("--sdoffloadcpu", help="Offload image weights in RAM to save VRAM, swap into VRAM when needed.", action='store_true')
    sdparsergroup.add_argument("--sdvaecpu", help="Force VAE to CPU only for image generation.", action='store_true')
    sdparsergroup.add_argument("--sdclipgpu", help="Put CLIP and T5 to GPU for image generation. Otherwise, CLIP will use CPU.", action='store_true')
    sdparsergroup.add_argument("--sdconvdirect", help="Enables Conv2D Direct. May improve performance or reduce memory usage. Might crash if not supported by the backend. Can be 'off' (default) to disable, 'full' to turn it on for all operations, or 'vaeonly' to enable only for the VAE.", type=sd_convdirect_option, choices=sd_convdirect_choices, default=sd_convdirect_choices[0])
    sdparsergroupvae = sdparsergroup.add_mutually_exclusive_group()
    sdparsergroupvae.add_argument("--sdvae", metavar=('[filename]'), help="Specify an image generation safetensors VAE which replaces the one in the model.", default="")
    sdparsergroupvae.add_argument("--sdvaeauto", help="Uses a built-in tiny VAE via TAE SD, which is very fast, and fixed bad VAEs.", action='store_true')
    sdparsergrouplora = sdparsergroup.add_mutually_exclusive_group()
    sdparsergrouplora.add_argument("--sdquant",  metavar=('[quantization level 0/1/2]'), help="If specified, loads the model quantized to save memory. 0=off, 1=q8, 2=q4", type=int, choices=[0,1,2], nargs="?", const=2, default=0)
    sdparsergrouplora.add_argument("--sdlora", metavar=('[filename]'), help="Specify image generation LoRAs safetensors models to be applied. Multiple LoRAs are accepted.", nargs='+')
    sdparsergroup.add_argument("--sdloramult", metavar=('[amounts]'), help="Multipliers for the image LoRA model to be applied.", type=float, nargs='+', default=[1.0])
    sdparsergroup.add_argument("--sdtiledvae", metavar=('[maxres]'), help="Adjust the automatic VAE tiling trigger for images above this size. 0 disables vae tiling.", type=int, default=default_vae_tile_threshold)
    sdparsergroup.add_argument("--sdmaingpu", metavar=('[Device ID]'), help="If specified, Image Generation weights will be placed on the selected GPU index", type=int, default=-1)

    whisperparsergroup = parser.add_argument_group('Whisper Transcription Commands')
    whisperparsergroup.add_argument("--whispermodel", metavar=('[filename]'), help="Specify a Whisper .bin model to enable Speech-To-Text transcription.", default="")

    ttsparsergroup = parser.add_argument_group('TTS Narration Commands')
    ttsparsergroup.add_argument("--ttsmodel", metavar=('[filename]'), help="Specify the TTS Text-To-Speech GGUF model.", default="")
    ttsparsergroup.add_argument("--ttswavtokenizer", metavar=('[filename]'), help="Specify the WavTokenizer GGUF model.", default="")
    ttsparsergroup.add_argument("--ttsgpu", help="Use the GPU for TTS.", action='store_true')
    ttsparsergroup.add_argument("--ttsmaxlen", help="Limit number of audio tokens generated with TTS.",  type=int, default=default_ttsmaxlen)
    ttsparsergroup.add_argument("--ttsthreads", metavar=('[threads]'), help="Use a different number of threads for TTS if specified. Otherwise, has the same value as --threads.", type=int, default=0)
    ttsparsergroup.add_argument("--ttsdir", metavar=('[directory]'), help="Select directory containing voices for voice cloning.", default="")

    musicparsergroup = parser.add_argument_group('Music Gen Commands')
    musicparsergroup.add_argument("--musicllm", metavar=('[filename]'), help="Select music LLM model (e.g acestep-5Hz-lm-0.6B)", default="")
    musicparsergroup.add_argument("--musicembeddings", metavar=('[filename]'), help="Select music embedding model (e.g Qwen3-Embedding-0.6B)", default="")
    musicparsergroup.add_argument("--musicdiffusion", metavar=('[filename]'), help="Select music diffusion (DiT) model (e.g acestep-v15-turbo)", default="")
    musicparsergroup.add_argument("--musicvae", metavar=('[filename]'), help="Select music VAE model", default="")
    musicparsergroup.add_argument("--musiclowvram", help="Unload music models when not in use", action='store_true')

    embeddingsparsergroup = parser.add_argument_group('Embeddings Model Commands')
    embeddingsparsergroup.add_argument("--embeddingsmodel", metavar=('[filename]'), help="Specify an embeddings model to be loaded for generating embedding vectors.", default="")
    embeddingsparsergroup.add_argument("--embeddingsmaxctx", metavar=('[amount]'), help="Overrides the default maximum supported context of an embeddings model (defaults to trained context).", type=int, default=0)
    embeddingsparsergroup.add_argument("--embeddingsgpu", help="Attempts to offload layers of the embeddings model to GPU. Usually not needed.", action='store_true')

    admingroup = parser.add_argument_group('Administration Commands')
    admingroup.add_argument("--admin", help="Enables admin mode, allowing you to unload and reload different configurations or models.", action='store_true')
    admingroup.add_argument("--adminpassword", metavar=('[password]'), help="Require a password to access admin functions. You are strongly advised to use one for publically accessible instances! Can also be set with env var KCPP_ADMINPASSWORD", default=os.getenv('KCPP_ADMINPASSWORD',None))
    admingroup.add_argument("--admindir", metavar=('[directory]'), help="Specify a directory to look for .kcpps configs in, which can be used to swap models.", default="")
    admingroup.add_argument("--adminunloadtimeout", help="Set an idle timeout in seconds after which KoboldCpp will automatically unload the current model.", type=int, default=0)
    admingroup.add_argument("--routermode", help="Router mode uses a reverse proxy router, allowing you to easily hotswap models and configs within a single request. Requires admin mode.", action='store_true')
    admingroup.add_argument("--reqtimeout", metavar=('[seconds]'), help="Timeout in seconds for HTTP requests.", type=int, default=default_reqtimeout)
    admingroup.add_argument("--autoswapmode", help="Autoswap mode builds on router mode to allow switching of model types within the same config automatically. Requires admin mode and router mode. All models desired must be defined within the same config.", action='store_true')
    admingroup.add_argument("--baseconfig", help="Specify a base .kcpps config to apply, if no custom base config is selected during a model swap", default="")

    deprecatedgroup = parser.add_argument_group('Deprecated Commands, DO NOT USE!')
    deprecatedgroup.add_argument("--hordeconfig", help=argparse.SUPPRESS, nargs='+')
    deprecatedgroup.add_argument("--sdconfig", help=argparse.SUPPRESS, nargs='+')
    compatgroup.add_argument("--noblas", help=argparse.SUPPRESS, action='store_true')
    compatgroup3.add_argument("--nommap","--no-mmap", help=argparse.SUPPRESS, action='store_true')
    deprecatedgroup.add_argument("--pipelineparallel", help=argparse.SUPPRESS, action='store_true') #changed to nopipelineparallel
    deprecatedgroup.add_argument("--sdnotile", help=argparse.SUPPRESS, action='store_true') # legacy option, see sdtiledvae
    deprecatedgroup.add_argument("--forceversion", help=argparse.SUPPRESS, action='store_true') #no longer used
    deprecatedgroup.add_argument("--sdgendefaults", help=argparse.SUPPRESS, action='store_true') # legacy option, see gendefaults
    deprecatedgroup.add_argument("--flashattention","--flash-attn","-fa", help=argparse.SUPPRESS, action='store_true') #flash attention now default on

    debuggroup = parser.add_argument_group('Debug Commands')
    debuggroup.add_argument("--testmemory", help=argparse.SUPPRESS, action='store_true')

    main(launch_args=parser.parse_args(), default_args=parser.parse_args([]))
