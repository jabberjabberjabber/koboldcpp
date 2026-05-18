import os
import sys
import re
import json
import copy
import threading
import subprocess
import platform
import time
import gzip
import base64
import math
import struct
import urllib.request
import urllib.parse
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import tkinter.font as tkfont
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None
    tkfont = None

from . import state
from .zenity_gui import (
    zenity, zentk_askopenfilename, zentk_askopenfilenames,
    zentk_askdirectory, zentk_asksaveasfilename,
)
from .config import (
    splitmode_choices_to_int, save_config_dict, convert_args_to_template,
    load_config_cli, save_config_cli, reload_new_config, convert_invalid_args,
)
from .hardware import (
    fetch_gpu_properties, auto_set_backend_cli, autoset_gpu_layers,
    detect_memory_cu, detect_memory_vk, get_default_threads,
)
from .utils import (
    utfprint, LaunchWebbrowser, get_my_epurl, make_url_request,
    sanitize_string, simple_lcg_hash, bring_terminal_to_foreground,
    truncate_long_json, print_with_time, old_cpu_check,
)
from .model_meta import (
    extract_modelfile_params, dump_gguf_metadata, read_gguf_metadata,
    analyze_gguf_model, analyze_gguf_model_wrapper, exit_with_error,
    unpack_to_dir, has_valid_model, downloader_internal,
    download_model_from_url, delete_old_pyinstaller,
)
from .backend import load_model, init_library
from .sd import sd_get_info

def show_gui():
    using_gui_launcher = True

    #check for potential scaling issues
    def get_problematic_scaler():
        if sys.platform != "linux":
            return False
        xdg_curr_desk = os.environ.get("XDG_CURRENT_DESKTOP")
        if xdg_curr_desk and ("KDE" in xdg_curr_desk or "GNOME" in xdg_curr_desk or "Cinnamon" in xdg_curr_desk): # broad spectrum dpi handler
            dpi = 0
            try:
                output = subprocess.check_output(["xrdb", "-query"], text=True).strip()
                if output:
                    for line in output.splitlines():
                        if line.startswith("Xft.dpi:"):
                            dpi = float(line.split(":")[1].strip())
                            break
            except Exception:
                pass
            if dpi > 100:
                return True

        import xml.etree.ElementTree as ET
        from pathlib import Path
        fractional_enabled = False # Check if fractional scaling is enabled
        try:
            features = subprocess.check_output(
                ["gsettings", "get", "org.gnome.mutter", "experimental-features"],
                text=True
            ).strip()
            fractional_enabled = "scale-monitor-framebuffer" in features
        except Exception:
            return False
        xml_path = Path.home() / ".config" / "monitors.xml"
        if not xml_path.exists(): #monitors.xml not found. if we have fractional scaling on gnome, just trigger the fallback
            if fractional_enabled and "GNOME" in os.environ.get("XDG_CURRENT_DESKTOP") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
                return True
            return False
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            configs = root.findall(".//configuration")
            if not configs:
                return False
            logical_confs = [c for c in configs if c.findtext(".//layoutmode") == "logical"]
            physical_confs = [c for c in configs if c.findtext(".//layoutmode") == "physical"]
            if fractional_enabled and logical_confs:
                chosen_conf = logical_confs[-1]
            elif not fractional_enabled and physical_confs:
                chosen_conf = physical_confs[-1]
            else:
                chosen_conf = configs[-1]
            scales = [float(s.text) for s in chosen_conf.findall(".//scale") if s.text]
            if scales:
                return max(scales)>1.0
        except Exception:
            pass
        return False

    corrupt_scaler = get_problematic_scaler()

    # if args received, launch
    if len(sys.argv) != 1 and not state.args.showgui:
        import tkinter as tk
        root = tk.Tk() #we dont want the useless window to be visible, but we want it in taskbar
        root.attributes("-alpha", 0)
        state.args.model_param = zentk_askopenfilename(title="Select ggml model .bin or .gguf file or .kcpps config")
        root.withdraw()
        root.quit()
        if state.args.model_param and state.args.model_param!="" and (state.args.model_param.lower().endswith('.kcpps') or state.args.model_param.lower().endswith('.kcppt') or state.args.model_param.lower().endswith('.kcpps?download=true') or state.args.model_param.lower().endswith('.kcppt?download=true')):
            dlfile = download_model_from_url(state.args.model_param,[".kcpps",".kcppt"]) # maybe download from url
            if dlfile:
                state.args.model_param = dlfile
            load_config_cli(state.args.model_param)
        if not has_valid_model():
            exitcounter = 999
            exit_with_error(2,"No gguf model or kcpps file was selected. Exiting.")
        return

    #dummy line to get darkdetect imported in pyinstaller
    try:
        import darkdetect as darkdt
        darkdt.isDark()
        pass
    except Exception:
        pass

    import customtkinter as ctk
    nextstate = 0 #0=exit, 1=launch
    original_windowwidth = int(590)
    original_windowheight = int(590)
    windowwidth = original_windowwidth
    windowheight = original_windowheight
    ctk.set_appearance_mode("dark")
    ctk.deactivate_automatic_dpi_awareness()
    root = ctk.CTk(fg_color="#2b2b2b")
    if corrupt_scaler:
        print("Adjusting tk scaling to try and fix scaling issues...")
        root.tk.call('tk','scaling', 2.25)
    root.geometry(str(windowwidth) + "x" + str(windowheight))
    root.title(f"KoboldCpp v{state.KcppVersion}")

    gtooltip_box = None
    gtooltip_label = None

    window_reference_width = None
    window_reference_height = None
    previous_event_width = None
    previous_event_height = None
    resizing = False
    resizing_id1 = None
    resizing_id2 = None
    def clearesizing():
        nonlocal resizing, resizing_id1
        resizing = False
        resizing_id1 = None
    def actually_resize(windowwidth,windowheight,lastpos,smallratio):
        root.geometry(str(windowwidth) + "x" + str(windowheight) + str(lastpos))
        ctk.set_widget_scaling(smallratio)
        changerunmode(1,1,1)
        togglerope(1,1,1)
        toggleflashattn(1,1,1)
        togglectxshift(1,1,1)
        togglehorde(1,1,1)
        toggletaesd(1,1,1)
        togglesdlora(1,1,1)
        togglejinja(1,1,1)
        toggleadmin(1,1,1)
        tabbuttonaction(tabnames[curr_tab_idx])
        pass
    def on_resize(event):
        nonlocal resizing, resizing_id1, resizing_id2
        if not event.widget.master and event.widget == root:
            nonlocal window_reference_width, window_reference_height, previous_event_width,previous_event_height
            if resizing:
                previous_event_width = event.width
                previous_event_height = event.height
                return
            if not window_reference_width and not window_reference_height:
                window_reference_width = event.width
                window_reference_height = event.height
                previous_event_width = window_reference_width
                previous_event_height = window_reference_height
            else:
                new_width = event.width
                new_height = event.height
                incr_w = new_width/window_reference_width
                incr_h = new_height/window_reference_height
                smallratio = min(incr_w,incr_h)
                smallratio = round(smallratio,2)
                if new_width != previous_event_width or new_height!=previous_event_height:
                    resizing = True
                    lastpos = root.geometry()
                    lparr = lastpos.split('+', 1)
                    lastpos = ("+"+str(lparr[1])) if (len(lparr)==2) else ""
                    previous_event_width = new_width
                    previous_event_height = new_height
                    windowwidth = math.floor(original_windowwidth*smallratio)
                    windowwidth = max(256, min(1024, windowwidth))
                    windowheight = math.floor(original_windowheight*smallratio)
                    windowheight = max(256, min(1024, windowheight))
                    if resizing_id2:
                        root.after_cancel(resizing_id2)
                        resizing_id2 = None
                    resizing_id2 = root.after(100, lambda: actually_resize(windowwidth,windowheight,lastpos,smallratio))
                    if resizing_id1:
                        root.after_cancel(resizing_id1)
                        resizing_id1 = None
                    resizing_id1 = root.after(5, clearesizing)

    if sys.platform=="darwin":
        root.resizable(False,False)
    else:
        root.resizable(True,True)
        root.bind("<Configure>", on_resize)
    kcpp_exporting_template = False

    # trigger empty tooltip then remove it
    def show_tooltip(event, tooltip_text=None):
        nonlocal gtooltip_box, gtooltip_label
        if not gtooltip_box and not gtooltip_label:
            gtooltip_box = ctk.CTkToplevel(root)
            gtooltip_box.configure(fg_color="#ffffe0")
            gtooltip_box.withdraw()
            gtooltip_box.overrideredirect(True)
            gtooltip_label = ctk.CTkLabel(gtooltip_box, text=tooltip_text, text_color="#000000", fg_color="#ffffe0")
            gtooltip_label.pack(expand=True, ipadx=2, ipady=1)
        else:
            gtooltip_label.configure(text=tooltip_text)

        gtooltip_box.update_idletasks()
        x, y = root.winfo_pointerxy()
        gtooltip_box.wm_geometry(f"+{x + 10}+{y + 10}")
        gtooltip_box.deiconify()

    def hide_tooltip(event):
        nonlocal gtooltip_box
        if gtooltip_box:
            gtooltip_box.withdraw()
    show_tooltip(None,"") #initialize tooltip objects
    hide_tooltip(None)

    default_threads = get_default_threads()

    tabs = ctk.CTkFrame(root, corner_radius = 0, width=windowwidth, height=windowheight-50)
    tabs.grid(row=0, stick="nsew")
    tabnames= ["Quick Launch", "Hardware", "Context", "Loaded Files", "Network", "Horde Worker","Image Gen","Audio","Admin","Extra"]
    navbuttons = {}
    navbuttonframe = ctk.CTkFrame(tabs, width=int(104), height=int(tabs.cget("height")))
    navbuttonframe.grid(row=0, column=0, padx=2,pady=2)
    navbuttonframe.grid_propagate(False)

    tabcontentframe = ctk.CTkFrame(tabs, width=windowwidth - int(navbuttonframe.cget("width")), height=int(tabs.cget("height")),fg_color="transparent")
    tabcontentframe.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    tabcontentframe.grid_propagate(False)

    tabcontent = {}
    # slider data
    batchsize_values = ["-1","16","32","64","128","256","512","1024","2048","4096"]
    batchsize_text = ["Don't Batch","16","32","64","128","256","512","1024","2048","4096"]
    contextsize_text = ["256", "512", "1024", "2048", "3072", "4096", "6144", "8192", "10240", "12288", "14336", "16384", "20480", "24576", "28672", "32768", "40960", "49152", "57344", "65536", "81920", "98304", "114688", "131072","163840","196608","229376","262144"]
    quantkv_text = ["f16","bf16","q8_0","q5_1","q4_0"]

    if not any(state.runopts):
        exitcounter = 999
        exit_with_error(2,"KoboldCPP couldn't locate any backends to use (i.e Default, Vulkan, CUDA).\n\nTo use the program, please run the 'make' command from the directory.","No Backends Available!")

    # Vars - should be in scope to be used by multiple widgets
    gpulayers_var = ctk.StringVar(value="-1")
    threads_var = ctk.StringVar(value=str(default_threads))
    runopts_var = ctk.StringVar()
    gpu_choice_var = ctk.StringVar(value="1")
    autofit_padding_var = ctk.StringVar(value=str(state.default_autofit_padding))

    launchbrowser = ctk.IntVar(value=1)
    highpriority = ctk.IntVar()
    usemmap = ctk.IntVar(value=0)
    usemlock = ctk.IntVar()
    debugmode = ctk.IntVar()
    keepforeground = ctk.IntVar()
    terminalonly = ctk.IntVar()
    pipelineparallel = ctk.IntVar(value=1)
    quietmode = ctk.IntVar(value=0)
    nocertifymode = ctk.IntVar(value=0)

    lowvram_var = ctk.IntVar()
    mmq_var = ctk.IntVar(value=1)
    quantkv_var = ctk.IntVar(value=0)
    blas_threads_var = ctk.StringVar()
    blas_size_var = ctk.IntVar()
    autofit_var = ctk.IntVar()
    tensor_split_str_vars = ctk.StringVar(value="")
    splitmode_var = ctk.StringVar(value=splitmode_choices[0])
    maingpu_var = ctk.StringVar(value="-1")
    deviceoverride_var = ctk.StringVar(value="")

    contextshift_var = ctk.IntVar(value=1)
    fastforward_var = ctk.IntVar(value=1)
    swa_var = ctk.IntVar(value=0)
    swa_padding_var = ctk.StringVar(value=str(state.swa_padding_default))
    smartcache_var = ctk.IntVar(value=0)
    smartcacheslots_var = ctk.StringVar(value=str(state.savestate_limit_default))
    remotetunnel_var = ctk.IntVar(value=0)
    smartcontext_var = ctk.IntVar()
    flashattention_var = ctk.IntVar(value=1)
    context_var = ctk.IntVar()
    customrope_var = ctk.IntVar()
    manualrope_var = ctk.IntVar()
    customrope_scale = ctk.StringVar(value="1.0")
    customrope_base = ctk.StringVar(value="10000")
    customrope_nativectx = ctk.StringVar(value=str(state.default_native_ctx))
    chatcompletionsadapter_var = ctk.StringVar(value="AutoGuess")
    jinjatemplate_var = ctk.StringVar()
    jinja_var = ctk.IntVar(value=0)
    jinja_tools_var = ctk.IntVar(value=0)
    jinja_kwargs_var = ctk.StringVar()
    moeexperts_var = ctk.StringVar(value=str(-1))
    moecpu_var = ctk.StringVar(value=str(0))
    defaultgenamt_var = ctk.StringVar(value=str(state.default_genlen))
    genlimit_var = ctk.StringVar(value=str(0))
    nobostoken_var = ctk.IntVar(value=0)
    override_kv_var = ctk.StringVar(value="")
    override_tensors_var = ctk.StringVar(value="")
    enableguidance_var = ctk.IntVar(value=0)

    model_var = ctk.StringVar()
    lora_var = ctk.StringVar()
    loramult_var = ctk.StringVar(value="1.0")
    preloadstory_var = ctk.StringVar()
    savedatafile_var = ctk.StringVar()
    mcpfile_var = ctk.StringVar()
    mmproj_var = ctk.StringVar()
    mmprojcpu_var = ctk.IntVar(value=0)
    visionmaxres_var = ctk.StringVar(value=str(state.default_visionmaxres))
    vision_min_tokens_var = ctk.StringVar(value="-1")
    vision_max_tokens_var = ctk.StringVar(value="-1")
    draftmodel_var = ctk.StringVar()
    draftamount_var = ctk.StringVar(value=str(state.default_draft_amount))
    draftgpulayers_var = ctk.StringVar(value=str(999))
    draftgpusplit_str_vars = ctk.StringVar(value="")
    nomodel = ctk.IntVar(value=0)
    download_dir_var = ctk.StringVar()

    port_var = ctk.StringVar(value=state.defaultport)
    host_var = ctk.StringVar(value="")
    multiuser_var = ctk.StringVar(value=str(state.multiuser_concurrent_limit))
    multiplayer_var = ctk.IntVar(value=state.has_multiplayer)
    websearch_var = ctk.IntVar(value=0)
    horde_name_var = ctk.StringVar(value="koboldcpp")
    horde_gen_var = ctk.StringVar(value=state.maxhordelen)
    horde_context_var = ctk.StringVar(value=state.maxhordectx)
    horde_apikey_var = ctk.StringVar(value="")
    horde_workername_var = ctk.StringVar(value="")
    usehorde_var = ctk.IntVar()
    ssl_cert_var = ctk.StringVar()
    ssl_key_var = ctk.StringVar()
    password_var = ctk.StringVar()
    maxrequestsize_var = ctk.StringVar(value=str(32))
    ratelimit_var = ctk.StringVar(value=str(0))
    reqtimeout_var = ctk.StringVar(value=str(state.default_reqtimeout))

    sd_model_var = ctk.StringVar()
    sd_lora_var = ctk.StringVar()
    sd_loramult_var = ctk.StringVar(value="1.0")
    sd_vae_var = ctk.StringVar()
    sd_t5xxl_var = ctk.StringVar()
    sd_clip1_var = ctk.StringVar()
    sd_clip2_var = ctk.StringVar()
    sd_photomaker_var = ctk.StringVar()
    sd_upscaler_var = ctk.StringVar()
    sd_flash_attention_var = ctk.IntVar(value=0)
    sd_offload_cpu_var = ctk.IntVar(value=0)
    sd_vae_cpu_var = ctk.IntVar(value=0)
    sd_clip_gpu_var = ctk.IntVar(value=0)
    sd_runtime_loras_var = ctk.IntVar(value=0)
    sd_vaeauto_var = ctk.IntVar(value=0)
    sd_tiled_vae_var = ctk.StringVar(value=str(state.default_vae_tile_threshold))
    sd_convdirect_var = ctk.StringVar(value=str(sd_convdirect_choices[0]))
    sd_clamped_var = ctk.StringVar(value="0")
    sd_clamped_soft_var = ctk.StringVar(value="0")
    sd_threads_var = ctk.StringVar(value=str(default_threads))
    sd_quant_var = ctk.StringVar(value=sd_quant_choices[0])
    sd_main_gpu_var = ctk.StringVar(value="-1")

    gen_defaults_var = ctk.StringVar()
    gen_defaults_overwrite_var = ctk.IntVar(value=0)

    whisper_model_var = ctk.StringVar()
    tts_model_var = ctk.StringVar()
    wavtokenizer_var = ctk.StringVar()
    ttsgpu_var = ctk.IntVar(value=0)
    tts_threads_var = ctk.StringVar(value=str(default_threads))
    ttsmaxlen_var = ctk.StringVar(value=str(state.default_ttsmaxlen))
    tts_dir_var = ctk.StringVar()

    musicllm_var = ctk.StringVar()
    musicembeddings_var = ctk.StringVar()
    musicdiffusion_var = ctk.StringVar()
    musicvae_var = ctk.StringVar()
    musiclowvram_var = ctk.IntVar(value=0)

    embeddings_model_var = ctk.StringVar()
    embeddings_ctx_var = ctk.StringVar(value=str(""))
    embeddings_gpu_var = ctk.IntVar(value=0)

    admin_var = ctk.IntVar(value=0)
    admin_dir_var = ctk.StringVar()
    baseconfig_var = ctk.StringVar()
    admin_password_var = ctk.StringVar()
    singleinstance_var = ctk.IntVar(value=0)
    router_mode_var = ctk.IntVar(value=0)
    autoswap_mode_var = ctk.IntVar(value=0)
    admin_unload_timeout_var = ctk.StringVar(value=str(0))

    nozenity_var = ctk.IntVar(value=0)

    curr_tab_idx = 0

    def tabbuttonaction(name):
        nonlocal curr_tab_idx
        idx = 0
        for t in tabcontent:
            if name == t:
                tabcontent[t].grid(row=0, column=0)
                navbuttons[t].configure(fg_color="#6f727b")
                curr_tab_idx = idx
            else:
                tabcontent[t].grid_remove()
                navbuttons[t].configure(fg_color="transparent")
            idx += 1

    # Dynamically create tabs + buttons based on values of [tabnames]
    for idx, name in enumerate(tabnames):
        tabcontent[name] = ctk.CTkFrame(tabcontentframe, width=int(tabcontentframe.cget("width")), height=int(tabcontentframe.cget("height")), fg_color="transparent")
        tabcontent[name].grid_propagate(False)
        if idx == 0:
            tabcontent[name].grid(row=idx, sticky="nsew")
        ctk.CTkLabel(tabcontent[name], text= name, font=ctk.CTkFont(None, 14, 'bold')).grid(row=0, padx=12, pady = 5, stick='nw')

        navbuttons[name] = ctk.CTkButton(navbuttonframe, text=name, width = 100, corner_radius=0 , command = lambda d=name:tabbuttonaction(d), hover_color="#868a94" )
        navbuttons[name].grid(row=idx)

    tabbuttonaction(tabnames[0])
    # Quick Launch Tab
    quick_tab = tabcontent["Quick Launch"]

    # helper functions
    def makecheckbox(parent, text, variable=None, row=0, column=0, command=None, padx=8,tooltiptxt=""):
        temp = ctk.CTkCheckBox(parent, text=text,variable=variable, onvalue=1, offvalue=0)
        if command is not None and variable is not None:
            variable.trace_add("write", command)
        temp.grid(row=row,column=column, padx=padx, pady=1, stick="nw")
        if tooltiptxt!="":
            temp.bind("<Enter>", lambda event: show_tooltip(event, tooltiptxt))
            temp.bind("<Leave>", hide_tooltip)
        return temp

    def makelabelcombobox(parent, text, variable=None, row=0, width=50, command=None, padx=8,tooltiptxt="", values=[], labelpadx=8):
        label = makelabel(parent, text, row, 0, tooltiptxt, padx=labelpadx)
        combo = ctk.CTkComboBox(parent, variable=variable, width=width, values=values, state="readonly")
        if command is not None and variable is not None:
            variable.trace_add("write", command)
        combo.grid(row=row,column=0, padx=padx, sticky="nw")
        if tooltiptxt!="":
            combo.bind("<Enter>", lambda event: show_tooltip(event, tooltiptxt))
            combo.bind("<Leave>", hide_tooltip)
        return combo, label

    def makelabel(parent, text, row, column=0, tooltiptxt="", columnspan=1, padx=8):
        temp = ctk.CTkLabel(parent, text=text)
        temp.grid(row=row, column=column, padx=padx, pady=1, stick="nw", columnspan=columnspan)
        if tooltiptxt!="":
            temp.bind("<Enter>", lambda event: show_tooltip(event, tooltiptxt))
            temp.bind("<Leave>", hide_tooltip)
        return temp

    def makeslider(parent, label, options, var, row=0, width=160, height=10, set=0, tooltip=""):
        sliderLabel = makelabel(parent, options[set], row + 1, 0, columnspan=2, padx=(width+12))
        titleLabel = makelabel(parent, label, row,0,tooltip)
        from_ = 0
        to = len(options)-1
        def sliderUpdate(a,b,c):
            sliderLabel.configure(text = options[int(var.get())])
        var.trace_add("write", sliderUpdate)
        slider = ctk.CTkSlider(parent, from_=from_, to=to, variable = var, width = width, height=height, border_width=5,number_of_steps=len(options) - 1)
        slider.grid(row=row+1,  column=0, padx = 8, stick="w", columnspan=2)
        slider.set(set)
        return slider, sliderLabel, titleLabel


    def makelabelentry(parent, text, var, row=0, width=50, padx=8, singleline=False, tooltip="", labelpadx=8):
        label = makelabel(parent, text, row, 0, tooltip, padx=labelpadx)
        entry = ctk.CTkEntry(parent, width=width, textvariable=var)
        entry.grid(row=row, column=(0 if singleline else 1), padx=padx, pady=1, sticky="nw")
        return entry, label

    #file dialog types: 0=openfile,1=savefile,2=opendir
    def makefileentry(parent, text, searchtext, var, row=0, width=200, filetypes=[], onchoosefile=None, singlerow=False, singlecol=True, dialog_type=0, tooltiptxt="", multiple=False):
        label = makelabel(parent, text, row,0,tooltiptxt,columnspan=3)
        def getfilename(var, text):
            initialDir = os.path.dirname(var.get())
            initialDir = initialDir if os.path.isdir(initialDir) else None
            fnam = None
            if dialog_type==2:
                fnam = zentk_askdirectory(title=text, mustexist=True, initialdir=initialDir)
            elif dialog_type==1:
                fnam = zentk_asksaveasfilename(title=text, filetypes=filetypes, defaultextension=filetypes, initialdir=initialDir)
                if not fnam:
                    fnam = ""
                else:
                    fnam = str(fnam).strip()
                    fnam = f"{fnam}.jsondb" if ".jsondb" not in fnam.lower() else fnam
            else:
                if multiple:
                    fnam = zentk_askopenfilenames(title=text,filetypes=filetypes, initialdir=initialDir)
                    fnam = "|".join(fnam)
                else:
                    fnam = zentk_askopenfilename(title=text,filetypes=filetypes, initialdir=initialDir)
            if fnam:
                var.set(fnam)
                if onchoosefile:
                    onchoosefile(var.get())
        entry = ctk.CTkEntry(parent, width, textvariable=var)
        button = ctk.CTkButton(parent, 50, text="Browse", command= lambda a=var,b=searchtext:getfilename(a,b))
        if singlerow:
            if singlecol:
                entry.grid(row=row, column=0, padx=((94)+8), pady=2, stick="w")
                button.grid(row=row, column=0, padx=((94)+width+12), pady=2, stick="w")
            else:
                entry.grid(row=row, column=1, padx=8, pady=2, stick="w")
                button.grid(row=row, column=1, padx=(width+12), pady=2, stick="w")
        else:
            if singlecol:
                entry.grid(row=row+1, column=0, columnspan=3, padx=8, pady=2, stick="w")
                button.grid(row=row+1, column=0, columnspan=3, padx=(width+12), pady=2, stick="w")
            else:
                entry.grid(row=row+1, column=0, columnspan=1, padx=8, pady=2, stick="w")
                button.grid(row=row+1, column=1, columnspan=1, padx=8, pady=2, stick="w")
        return label, entry, button

    def model_searcher():
        searchbox1 = None
        searchbox2 = None
        modelsearch1_var = ctk.StringVar(value="")
        modelsearch2_var = ctk.StringVar(value="")
        fileinfotxt_var = ctk.StringVar(value="")
        # Create popup window
        popup = ctk.CTkToplevel(root)
        popup.title("Model File Browser")
        popup.geometry("400x400")
        searchedmodels = []
        searchedsizes = []

        def confirm_search_model_choice():
            nonlocal modelsearch1_var, modelsearch2_var, model_var, fileinfotxt_var
            if modelsearch1_var.get()!="" and modelsearch2_var.get()!="":
                model_var.set(f"https://huggingface.co/{modelsearch1_var.get()}/resolve/main/{modelsearch2_var.get()}")
            popup.destroy()
        def update_search_quant_file_size(a,b,c):
            nonlocal modelsearch1_var, modelsearch2_var, fileinfotxt_var, searchedmodels, searchedsizes, searchbox2
            try:
                selected_index = searchbox2.cget("values").index(modelsearch2_var.get())
                pickedsize = searchedsizes[selected_index]
                fileinfotxt_var.set(f"Size: {round(pickedsize/1024/1024/1024,2)} GB")
            except Exception:
                fileinfotxt_var.set("")
        def fetch_search_quants(a,b,c):
            nonlocal modelsearch1_var, modelsearch2_var, fileinfotxt_var, searchedmodels, searchedsizes
            try:
                if modelsearch1_var.get()=="":
                    return
                searchedmodels = []
                searchedsizes = []
                resp = make_url_request(f"https://huggingface.co/api/models/{modelsearch1_var.get()}/tree/main?recursive=true",None,'GET',{},10)
                for m in resp:
                    if m["type"]=="file" and ".gguf" in m["path"]:
                        if "-of-0" in m["path"] and "00001" not in m["path"]:
                            continue
                        searchedmodels.append(m["path"])
                        searchedsizes.append(m["size"])
                searchbox2.configure(values=searchedmodels)
                if len(searchedmodels)>0:
                    quants = ["q4k","q4_k","q4", "q3", "q5", "q6", "q8"] #autopick priority
                    chosen_model = searchedmodels[0]
                    found_good = False
                    for quant in quants:
                        for filename in searchedmodels:
                            if quant in filename.lower():
                                chosen_model = filename
                                found_good = True
                                break
                        if found_good:
                            break
                    modelsearch2_var.set(chosen_model)
                    update_search_quant_file_size(1,1,1)
                else:
                    modelsearch2_var.set("")
                    fileinfotxt_var.set("")
            except Exception as e:
                modelsearch1_var.set("")
                modelsearch2_var.set("")
                fileinfotxt_var.set("")
                print(f"Error: {e}")
        def fetch_search_models():
            from tkinter import messagebox
            nonlocal searchbox1, searchbox2, modelsearch1_var, modelsearch2_var, fileinfotxt_var
            try:
                modelsearch1_var.set("")
                modelsearch2_var.set("")
                fileinfotxt_var.set("")
                searchbox1.configure(values=[])
                searchbox2.configure(values=[])
                searchedmodels = []
                searchbase = model_search.get()
                if searchbase.strip()=="":
                    return
                urlcode = urllib.parse.urlencode({"search":( "GGUF " + searchbase),"limit":10}, doseq=True)
                urlcode2 = urllib.parse.urlencode({"search":searchbase,"limit":6}, doseq=True)
                resp = make_url_request(f"https://huggingface.co/api/models?{urlcode}",None,'GET',{},10)
                for m in resp:
                    searchedmodels.append(m["id"])
                if len(resp)<=3: #too few results, repeat search without GGUF in the string
                    resp2 = make_url_request(f"https://huggingface.co/api/models?{urlcode2}",None,'GET',{},10)
                    for m in resp2:
                        searchedmodels.append(m["id"])

                if len(searchedmodels)==0:
                    messagebox.showinfo("No Results Found", "Search found no results")
                searchbox1.configure(values=searchedmodels)
                if len(searchedmodels)>0:
                    modelsearch1_var.set(searchedmodels[0])
                else:
                    modelsearch1_var.set("")
            except Exception as e:
                modelsearch1_var.set("")
                modelsearch2_var.set("")
                fileinfotxt_var.set("")
                print(f"Error: {e}")

        ctk.CTkLabel(popup, text="Enter Search String:").pack(pady=(10, 0))
        model_search = ctk.CTkEntry(popup, width=300)
        model_search.pack(pady=5)
        model_search.insert(0, "")

        ctk.CTkButton(popup, text="Search Huggingface", command=fetch_search_models).pack(pady=5)

        ctk.CTkLabel(popup, text="Selected Model:").pack(pady=(10, 0))
        searchbox1 = ctk.CTkComboBox(popup, values=[], width=340, variable=modelsearch1_var, state="readonly")
        searchbox1.pack(pady=5)
        ctk.CTkLabel(popup, text="Selected Quant:").pack(pady=(10, 0))
        searchbox2 = ctk.CTkComboBox(popup, values=[], width=340, variable=modelsearch2_var, state="readonly")
        searchbox2.pack(pady=5)
        modelsearch1_var.trace_add("write", fetch_search_quants)
        modelsearch2_var.trace_add("write", update_search_quant_file_size)
        ctk.CTkLabel(popup, text="", textvariable=fileinfotxt_var, text_color="#ffff00").pack(pady=(10, 0))
        ctk.CTkButton(popup, text="Confirm Selection", command=confirm_search_model_choice).pack(pady=5)

        popup.transient(root)

    # decided to follow yellowrose's and kalomaze's suggestions, this function will automatically try to determine GPU identifiers
    # run in new thread so it doesnt block. does not return anything, instead overwrites specific values and redraws GUI
    def auto_set_backend_gui(manual_select=False):
        if manual_select:
            print("\nA .kcppt template was selected from GUI - automatically selecting your backend...")
            runmode_untouched = True
        fetch_gpu_properties(True,True)
        found_new_backend = False

        # check for avx2 and avx support
        is_oldpc_ver = "Use CPU" not in state.runopts #on oldcpu ver, default lib does not exist
        cpusupport = old_cpu_check() # 0 if has avx2, 1 if has avx, 2 if has nothing
        eligible_cuda = (cpusupport<1 and not is_oldpc_ver) or (cpusupport<2 and is_oldpc_ver)

        #autopick cublas if suitable, requires at least 3.5GB VRAM to auto pick
        #we do not want to autoselect hip/cublas if the user has already changed their desired backend!
        if eligible_cuda and state.exitcounter < 100 and state.MaxMemory[0]>3500000000 and (("Use CUDA" in state.runopts and state.CUDevicesNames[0]!="") or "Use hipBLAS (ROCm)" in state.runopts) and (any(state.CUDevicesNames)) and state.runmode_untouched:
            if "Use CUDA" in state.runopts:
                runopts_var.set("Use CUDA")
                gpu_choice_var.set("1")
                print(f"Auto Selected CUDA Backend (flag={cpusupport})\n")
                found_new_backend = True
            elif "Use hipBLAS (ROCm)" in state.runopts:
                runopts_var.set("Use hipBLAS (ROCm)")
                gpu_choice_var.set("1")
                print(f"Auto Selected HIP Backend (flag={cpusupport})\n")
                found_new_backend = True
        elif state.exitcounter < 100 and (1 in state.VKIsDGPU) and state.runmode_untouched and ("Use Vulkan" in state.runopts or "Use Vulkan (Old CPU)" in state.runopts):
            for i in range(0,len(state.VKIsDGPU)):
                if state.VKIsDGPU[i]==1:
                    if cpusupport<1 and "Use Vulkan" in state.runopts:
                        runopts_var.set("Use Vulkan")
                    else:
                        runopts_var.set("Use Vulkan (Old CPU)")
                    gpu_choice_var.set(str(i+1))
                    print(f"Auto Selected Vulkan Backend (flag={cpusupport})\n")
                    found_new_backend = True
                    break
        else:
            if runopts_var.get()=="Use CPU" and cpusupport==1 and "Use CPU (Old CPU)" in state.runopts:
                runopts_var.set("Use CPU (Old CPU)")
            elif runopts_var.get()=="Use CPU" and cpusupport==2 and "Failsafe Mode (Older CPU)" in state.runopts:
                runopts_var.set("Failsafe Mode (Older CPU)")
        if not found_new_backend:
            print(f"Auto Selected Default Backend (flag={cpusupport})\n")
        changed_gpu_choice_var()

    def on_picked_model_file(filepath):
        if filepath and (filepath.lower().endswith('.kcpps') or filepath.lower().endswith('.kcppt')):
            #load it as a config file instead
            if filepath.lower().endswith('.kcpps'):
                runmode_untouched = False
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                dict = json.load(f)
                import_vars(dict)

    def gui_changed_modelfile(*args):
        if not state.importvars_in_progress:
            filepath = model_var.get()
            sdfilepath = sd_model_var.get()
            whisperfilepath = whisper_model_var.get()
            mmprojfilepath = mmproj_var.get()
            draftmodelpath = draftmodel_var.get()
            ttsmodelpath = tts_model_var.get() if ttsgpu_var.get()==1 else ""
            embdmodelpath = embeddings_model_var.get() if embeddings_gpu_var.get()==1 else ""
            musicllmpath = musicllm_var.get()
            musicditpath = musicdiffusion_var.get()
            extract_modelfile_params(filepath,sdfilepath,whisperfilepath,mmprojfilepath,draftmodelpath,state.ttsmodelpath,embdmodelpath,musicllmpath,musicditpath)
            changed_gpulayers_estimate()
        pass

    def changed_autofit(*args):
        orig_rmu = state.runmode_untouched
        changerunmode(1,1,1)
        runmode_untouched = orig_rmu
        changed_gpulayers_estimate()

    def changed_gpulayers_estimate(*args):
        autoset_gpu_layers(int(contextsize_text[context_var.get()]),sd_quant_option(sd_quant_var.get()),int(batchsize_values[int(blas_size_var.get())]),musiclowvram_var.get()==1)
        max_gpu_layers = (f"{state.modelfile_extracted_meta[1][0]+1}" if (state.modelfile_extracted_meta and state.modelfile_extracted_meta[1] and state.modelfile_extracted_meta[1][0]!=0) else "")
        index = runopts_var.get()
        gpu_be = (index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or index == "Use hipBLAS (ROCm)")
        layercounter_label.grid(row=6, column=0, padx=230, sticky="W")
        quick_layercounter_label.grid(row=6, column=1, padx=75, sticky="W")
        if sys.platform=="darwin" and gpulayers_var.get()=="-1" and max_gpu_layers:
            quick_layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
            layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
        elif gpu_be and gpulayers_var.get()=="-1" and max_gpu_layers:
            quick_layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
            layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
        elif gpu_be and gpulayers_var.get()=="":
            quick_layercounter_label.configure(text="(Set -1 for Auto)")
            layercounter_label.configure(text="(Set -1 for Auto)")
        else:
            layercounter_label.grid_remove()
            quick_layercounter_label.grid_remove()

        if autofit_var.get()==1:
            layercounter_label.grid_remove()
            quick_layercounter_label.grid_remove()

    def changed_gpu_choice_var(*args):
        if state.exitcounter > 100:
            return
        if gpu_choice_var.get()!="All":
            try:
                s = int(gpu_choice_var.get())-1
                v = runopts_var.get()
                if v == "Use Vulkan" or v == "Use Vulkan (Old CPU)" or v == "Use Vulkan (Older CPU)":
                    quick_gpuname_label.configure(text=state.VKDevicesNames[s])
                    gpuname_label.configure(text=state.VKDevicesNames[s])
                else:
                    quick_gpuname_label.configure(text=state.CUDevicesNames[s])
                    gpuname_label.configure(text=state.CUDevicesNames[s])
            except Exception:
                pass
        else:
            quick_gpuname_label.configure(text="(dGPUs only, tensor split sets ratio)")
            gpuname_label.configure(text="(dGPUs only, tensor split sets ratio)")

    gpu_choice_var.trace_add("write", changed_gpu_choice_var)
    gpulayers_var.trace_add("write", changed_gpulayers_estimate)

    def toggleswa(a,b,c):
        if swa_var.get()==1:
            contextshift_var.set(0)
            swa_padding_entry.grid()
            swa_padding_label.grid()
        else:
            swa_padding_entry.grid_remove()
            swa_padding_label.grid_remove()

    def togglesmartcache(a,b,c):
        if smartcache_var.get()==1:
            fastforward_var.set(1)

    def togglefastforward(a,b,c):
        if fastforward_var.get()==0:
            contextshift_var.set(0)
            smartcontext_var.set(0)
            smartcache_var.set(0)

    def togglectxshift(a,b,c):
        if contextshift_var.get()==0:
            smartcontextbox.grid()
        else:
            fastforward_var.set(1)
            swa_var.set(0)
            smartcontextbox.grid_remove()
        qkvslider.grid()
        qkvlabel.grid()
        if flashattention_var.get()==0 and (quantkv_var.get()>1):
            noqkvlabel.grid()
        else:
            noqkvlabel.grid_remove()


    def toggleflashattn(a,b,c):
        qkvslider.grid()
        qkvlabel.grid()
        if flashattention_var.get()==0 and (quantkv_var.get()>1):
            noqkvlabel.grid()
        else:
            noqkvlabel.grid_remove()
        changed_gpulayers_estimate()

    def guibench():
        state.args.benchmark = "stdout"
        launchbrowser.set(0)
        guilaunch()

    def changerunmode(a,b,c):
        runmode_untouched = False
        index = runopts_var.get()
        if index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or index == "Use hipBLAS (ROCm)":
            quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
            gpuname_label.grid(row=3, column=0, padx=230, sticky="W")
            gpu_selector_label.grid(row=3, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_selector_label.grid(row=3, column=0, padx = 8, pady=1, stick="nw")
            CUDA_gpu_selector_box.grid(row=3, column=0, padx=160, pady=1, stick="nw")
            CUDA_quick_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
            maingpu_label.grid(row=8, column=0, padx = 270, pady=1, stick="nw")
            maingpu_entry.grid(row=8, column=0, padx = 340, pady=1, stick="nw")
            lowvram_box.grid(row=4, column=0, padx=8, pady=1, stick="nw")
            splitmode_box.grid(row=4, column=0, padx=230, pady=1, stick="nw")
            splitmode_box_label.grid(row=4, column=0, padx=160, pady=1, stick="nw")
        else:
            quick_gpuname_label.grid_remove()
            gpuname_label.grid_remove()
            gpu_selector_label.grid_remove()
            CUDA_gpu_selector_box.grid_remove()
            quick_gpu_selector_label.grid_remove()
            CUDA_quick_gpu_selector_box.grid_remove()
            maingpu_label.grid_remove()
            maingpu_entry.grid_remove()
            lowvram_box.grid_remove()
            splitmode_box.grid_remove()
            splitmode_box_label.grid_remove()

        if index == "Use CUDA" or index == "Use hipBLAS (ROCm)":
            mmq_box.grid(row=4, column=0, padx=340, pady=1,  stick="nw")
            quick_mmq_box.grid(row=4, column=1, padx=8, pady=1,  stick="nw")
            tensor_split_label.grid(row=8, column=0, padx = 8, pady=1, stick="nw")
            tensor_split_entry.grid(row=8, column=0, padx = 160, pady=1, stick="nw")
        else:
            mmq_box.grid_remove()
            quick_mmq_box.grid_remove()
            tensor_split_label.grid_remove()
            tensor_split_entry.grid_remove()

        if index == "Use Vulkan" or index == "Use Vulkan (Old CPU)":
            tensor_split_label.grid(row=8, column=0, padx = 8, pady=1, stick="nw")
            tensor_split_entry.grid(row=8, column=0, padx = 160, pady=1, stick="nw")

        if index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or index == "Use hipBLAS (ROCm)":
            gpu_layers_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
            gpu_layers_entry.grid(row=6, column=0, padx=160, pady=1, stick="nw")
            quick_gpu_layers_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_layers_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
        elif sys.platform=="darwin":
            gpu_layers_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
            gpu_layers_entry.grid(row=6, column=0, padx=160, pady=1, stick="nw")
            quick_gpu_layers_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_layers_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
        else:
            gpu_layers_label.grid_remove()
            gpu_layers_entry.grid_remove()
            quick_gpu_layers_label.grid_remove()
            quick_gpu_layers_entry.grid_remove()

        if autofit_var.get()==1:
            gpu_layers_label.grid_remove()
            gpu_layers_entry.grid_remove()
            quick_gpu_layers_label.grid_remove()
            quick_gpu_layers_entry.grid_remove()
            autofit_padding_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
            autofit_padding_entry.grid(row=6, column=0, padx=160, pady=1, stick="nw")
            moecpu_box.grid_remove()
            tenos_box.grid_remove()
            moecpu_box_lbl.grid_remove()
            tenos_box_lbl.grid_remove()
        else:
            autofit_padding_label.grid_remove()
            autofit_padding_entry.grid_remove()
            moecpu_box.grid()
            tenos_box.grid()
            moecpu_box_lbl.grid()
            tenos_box_lbl.grid()

        changed_gpulayers_estimate()
        changed_gpu_choice_var()

    # presets selector
    makelabel(quick_tab, "Backend:", 1,0,"Select a backend to use.\nCUDA runs on Nvidia GPUs, and is much faster.\nVulkan works on all GPUs but is somewhat slower.\nOtherwise, runs on CPU only.\nNoAVX2 and Failsafe modes support older PCs.")

    runoptbox = ctk.CTkComboBox(quick_tab, values=state.runopts, width=190,variable=runopts_var, state="readonly")
    runoptbox.grid(row=1, column=1,padx=8, stick="nw")
    runoptbox.set(state.runopts[0]) # Set to first available option

    # gpu options
    quick_gpu_selector_label = makelabel(quick_tab, "GPU ID:", 3,0,"Which GPU ID to load the model with.\nNormally your main GPU is #1, but it can vary for multi GPU setups.",padx=8)
    CUDA_quick_gpu_selector_box = ctk.CTkComboBox(quick_tab, values=state.CUDevices, width=60, variable=gpu_choice_var, state="readonly")
    CUDA_quick_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
    quick_gpuname_label = ctk.CTkLabel(quick_tab, text="")
    quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
    quick_gpuname_label.configure(text_color="#ffff00")
    quick_gpu_layers_entry,quick_gpu_layers_label = makelabelentry(quick_tab,"GPU Layers:", gpulayers_var, 6, 50,tooltip="How many layers to offload onto the GPU.\nUsage varies based on model type and increases with model and context size.\nRequires some trial and error to find the best fit value.\n\nNote: The auto estimation is often inaccurate! Please set layers yourself for best results!")
    quick_gpu_layers_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
    quick_gpu_layers_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
    quick_layercounter_label = ctk.CTkLabel(quick_tab, text="")
    quick_layercounter_label.grid(row=6, column=1, padx=75, sticky="W")
    quick_layercounter_label.configure(text_color="#ffff00")
    quick_mmq_box = makecheckbox(quick_tab,  "Use MMQ", mmq_var, 4,1,tooltiptxt="Enable MMQ mode instead of CuBLAS for prompt processing. Read the wiki. Speed may vary.")

    # quick boxes
    quick_boxes = {
        "Launch Browser": [launchbrowser, "Launches your default browser after model loading is complete"],
        "Use MMAP": [usemmap,  "Use mmap to load models if enabled, model will not be unloadable"],
        "Use ContextShift": [contextshift_var, "Uses Context Shifting to reduce reprocessing.\nRecommended. Check the wiki for more info."],
        "Remote Tunnel": [remotetunnel_var,  "Creates a trycloudflare tunnel.\nAllows you to access koboldcpp from other devices over an internet URL."],
        "Use FlashAttention": [flashattention_var, "Enable flash attention for GGUF models."],
        "Force AutoFit": [autofit_var, "Automatically attempt to fit the model in the best possible way. Overrides everything else.\nNot recommended for multi model setups. Experimental."],
        "Quiet Mode": [quietmode, "Prevents all generation related terminal output from being displayed."]
    }

    for idx, (name, properties) in enumerate(quick_boxes.items()):
        makecheckbox(quick_tab, name, properties[0], int(idx/2) + 20, idx % 2, tooltiptxt=properties[1])

    # context size
    makeslider(quick_tab, "Context Size:", contextsize_text, context_var, 40, width=280, set=7,tooltip="What is the maximum context size to support. Model specific. You cannot exceed it.\nLarger contexts require more memory, and not all models support it.")

    # load model
    makefileentry(quick_tab, "GGUF Text Model:", "Select GGUF or GGML Model File", model_var, 50, 280, onchoosefile=on_picked_model_file,tooltiptxt="Select a GGUF or GGML model file on disk to be loaded.")
    model_var.trace_add("write", gui_changed_modelfile)
    ctk.CTkButton(quick_tab, width=70, text = "HF Search", command = model_searcher ).grid(row=51,column=1, stick="sw", padx=184, pady=2)

    # Hardware Tab
    hardware_tab = tabcontent["Hardware"]

    # presets selector
    makelabel(hardware_tab, "Backend:", 1,0,"Select a backend to use.\nCUDA runs on Nvidia GPUs, and is much faster.\nVulkan works on all GPUs but is somewhat slower.\nOtherwise, runs on CPU only.\nNoAVX2 and Failsafe modes support older PCs.")
    runoptbox = ctk.CTkComboBox(hardware_tab, values=state.runopts,  width=180,variable=runopts_var, state="readonly")
    runoptbox.grid(row=1, column=0,padx=160, stick="nw")
    runoptbox.set(state.runopts[0]) # Set to first available option

    # gpu options
    gpu_selector_label = makelabel(hardware_tab, "GPU ID:", 3,0,"Which GPU ID to load the model with.\nNormally your main GPU is #1, but it can vary for multi GPU setups.")
    CUDA_gpu_selector_box = ctk.CTkComboBox(hardware_tab, values=state.CUDevices, width=60, variable=gpu_choice_var, state="readonly")
    CUDA_gpu_selector_box.grid(row=3, column=0, padx=160, pady=1, stick="nw")
    gpuname_label = ctk.CTkLabel(hardware_tab, text="")
    gpuname_label.grid(row=3, column=0, padx=230, sticky="W")
    gpuname_label.configure(text_color="#ffff00")
    lowvram_box = makecheckbox(hardware_tab,  "No KV offload", lowvram_var, 4,0, tooltiptxt='Avoid offloading KV Cache or scratch buffers to VRAM.\nAllows more layers to fit, but may result in a large speed loss.')
    mmq_box = makecheckbox(hardware_tab,  "Use MMQ", mmq_var, 4,0,padx=340, tooltiptxt="Enable MMQ mode to use finetuned kernels instead of default CuBLAS/HipBLAS for prompt processing.\nRead the wiki. Speed may vary.")
    splitmode_box,splitmode_box_label = makelabelcombobox(hardware_tab, "SplitMode: ", splitmode_var, 4, width=(80), padx=(230), labelpadx=160, tooltiptxt="How to split the model across multiple GPUs. Layer split is default.", values=splitmode_choices)
    gpu_layers_entry,gpu_layers_label = makelabelentry(hardware_tab,"GPU Layers:", gpulayers_var, 6, 50, padx=160,singleline=True,tooltip="How many layers to offload onto the GPU.\nUsage varies based on model type and increases with model and context size.\nRequires some trial and error to find the best fit value.\n\nNote: The auto estimation is often inaccurate! Please set layers yourself for best results!")
    autofit_padding_entry,autofit_padding_label = makelabelentry(hardware_tab,"Autofit Padding (MB):", autofit_padding_var, 6, 50, padx=160,singleline=True,tooltip="How much spare allowance in MB should autofit reserve? If it's too little, the load might fail.")
    layercounter_label = ctk.CTkLabel(hardware_tab, text="")
    layercounter_label.grid(row=6, column=0, padx=230, sticky="W")
    layercounter_label.configure(text_color="#ffff00")
    tensor_split_entry,tensor_split_label = makelabelentry(hardware_tab, "Tensor Split:", tensor_split_str_vars, 8, 80, padx=160, singleline=True, tooltip='When using multiple GPUs this option controls how large tensors should be split across all GPUs.\nUses a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order.\nFor example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1.')
    maingpu_entry,maingpu_label = makelabelentry(hardware_tab, "Main GPU:" , maingpu_var, 8, 50,padx=340,singleline=True,tooltip="Only for multi-gpu, which GPU ID to set as main?\nIf left blank or -1, uses default value.",labelpadx=270)

    # threads
    makelabelentry(hardware_tab, "Threads:" , threads_var, 11, 50, padx=160, singleline=True,tooltip="How many threads to use.\nRecommended value is your CPU core count, defaults are usually OK.")
    # blas thread specifier
    makelabelentry(hardware_tab, "Batch Threads:" , blas_threads_var, 11, 50,padx=340, singleline=True,tooltip="How many threads to use during batched processing.\nIf left blank, uses same value as regular thread count.",labelpadx=240)
    makelabelentry(hardware_tab, "Device Override", deviceoverride_var, 15, 120, padx=(160), singleline=True, tooltip="Set llama.cpp compatible device selection override. Comma separated (e.g. Vulkan0,Vulkan1). Overrides normal device choices.")

    # hardware checkboxes
    hardware_boxes = {
        "Launch Browser": [launchbrowser, "Launches your default browser after model loading is complete"],
        "High Priority": [highpriority, "Increases the koboldcpp process priority.\nMay cause lag or slowdown instead. Not recommended."],
        "Use MMAP": [usemmap, "Use mmap to load models if enabled, model will not be unloadable"],
        "Use mlock": [usemlock, "Enables mlock, preventing the RAM used to load the model from being paged out."],
        "Debug Mode": [debugmode, "Enables debug mode, with extra info printed to the terminal."],
        "Keep Foreground": [keepforeground, "Bring KoboldCpp to the foreground every time there is a new generation."],
        "CLI Terminal Only": [terminalonly, "Does not launch KoboldCpp HTTP server. Instead, enables KoboldCpp from the command line, accepting interactive console input and displaying responses to the terminal."],
        "Pipeline Parallel": [pipelineparallel, "Enable Pipeline Parallelism for faster multigpu speeds but using more memory, only active for multigpu."],
    }

    for idx, (name, properties) in enumerate(hardware_boxes.items()):
        makecheckbox(hardware_tab, name, properties[0], int(idx/2) + 30, 0, padx=(160 if idx % 2 else 8), tooltiptxt=properties[1])

    # blas batch size
    makeslider(hardware_tab, "Batch Size:", batchsize_text, blas_size_var, 16,width=200, set=6,tooltip="How many tokens to process at once per batch.\nLarger values use more memory.")
    blas_size_var.trace_add("write", changed_gpulayers_estimate)

    makecheckbox(hardware_tab, "Use FlashAttention", flashattention_var, 100, command=toggleflashattn,  tooltiptxt="Enable flash attention for GGUF models.")

    makecheckbox(hardware_tab, "Force AutoFit", autofit_var, 100,0,command=changed_autofit,padx=160, tooltiptxt="Automatically attempt to fit the model in the best possible way. Overrides everything else.\nNot recommended for multi model setups. Experimental.")
    ctk.CTkButton(hardware_tab , text = "Run Benchmark", command = guibench ).grid(row=110,column=0, stick="nw", padx= 8, pady=2)


    # Context Tab
    context_tab = tabcontent["Context"]
    # Context checkboxes
    smartcontextbox = makecheckbox(context_tab, "Use SmartContext", smartcontext_var, 3, padx=330,tooltiptxt="Uses SmartContext. Now considered outdated and not recommended.\nCheck the wiki for more info.")
    makecheckbox(context_tab, "Use ContextShift", contextshift_var, 3,padx=180,tooltiptxt="Uses Context Shifting to reduce reprocessing.\nRecommended. Check the wiki for more info.", command=togglectxshift)
    makecheckbox(context_tab, "Use FastForwarding", fastforward_var, 3,tooltiptxt="Use fast forwarding to recycle previous context (always reprocess if disabled).\nRecommended.", command=togglefastforward)
    makecheckbox(context_tab, "Use SWA", swa_var, 4,tooltiptxt="Allows Sliding Window Attention (SWA) KV Cache, which saves memory but cannot be used with context shifting.", command=toggleswa)
    swa_padding_entry,swa_padding_label = makelabelentry(context_tab,"SWA Padding Tokens:", swa_padding_var, 4, 50, padx=300,singleline=True,tooltip="If the SWA is too small, you can expand it with padding, allowing for greater distance context rewinds.",labelpadx=160)
    makecheckbox(context_tab, "Use SmartCache", smartcache_var, 5,tooltiptxt="Enables intelligent context switching by saving KV cache snapshots to RAM. Requires fast forwarding.", command=togglesmartcache)
    makelabelentry(context_tab, "CacheSlots:", smartcacheslots_var, row=5, padx=(300), singleline=True, tooltip="Number of slots for smartcache",labelpadx=(220))

    # context size
    makeslider(context_tab, "Context Size:",contextsize_text, context_var, 18, width=280, set=7,tooltip="What is the maximum context size to support. Model specific. You cannot exceed it.\nLarger contexts require more memory, and not all models support it.")
    context_var.trace_add("write", changed_gpulayers_estimate)
    makelabelentry(context_tab, "Default Gen Amt:", defaultgenamt_var, row=20, padx=(120), singleline=True, tooltip="How many tokens to generate by default, if not specified. Must be smaller than context size. Usually, your frontend GUI will override this.")
    makelabelentry(context_tab, "Prompt Limit:", genlimit_var, row=20, padx=(300), singleline=True, tooltip="If set, restricts max output tokens to this limit regardless of API request. Set to 0 to disable.",labelpadx=(210))
    makelabelentry(context_tab, "Default Params:", gen_defaults_var, row=21, width=200, padx=(110), singleline=True, tooltip='Set default generation parameters for incoming API payloads.\nSpecified as JSON fields: {"KEY1":"VALUE1", "KEY2":"VALUE2"...}')
    makecheckbox(context_tab, "Override", gen_defaults_overwrite_var, row=21,padx=(330), tooltiptxt="Allow the gendefaults parameters to overwrite the original value in API payloads.")

    nativectx_entry, nativectx_label = makelabelentry(context_tab, "Override Native Context:", customrope_nativectx, row=23, padx=(146), singleline=True, tooltip="Overrides the native trained context of the loaded model with a custom value to be used for Rope scaling.")
    customrope_scale_entry, customrope_scale_label = makelabelentry(context_tab, "RoPE Scale:", customrope_scale, row=23, padx=(100), singleline=True, tooltip="For Linear RoPE scaling. RoPE frequency scale.")
    customrope_base_entry, customrope_base_label = makelabelentry(context_tab, "Base:", customrope_base, row=23, padx=(220), singleline=True, tooltip="For NTK Aware Scaling. RoPE frequency base.",labelpadx=(180))
    def togglerope(a,b,c):
        manualropebox.grid_remove()
        nativectx_label.grid_remove()
        nativectx_entry.grid_remove()
        customrope_scale_label.grid_remove()
        customrope_scale_entry.grid_remove()
        customrope_base_label.grid_remove()
        customrope_base_entry.grid_remove()
        if customrope_var.get() == 1:
            manualropebox.grid(row=22, column=0,padx=(200), pady=1, stick="nw")
            if manualrope_var.get() == 1:
                customrope_scale_label.grid(row=23, column=0, padx=8, pady=1, stick="nw")
                customrope_scale_entry.grid(row=23, column=0, padx=(100), pady=1, stick="nw")
                customrope_base_label.grid(row=23, column=0, padx=(180), pady=1, stick="nw")
                customrope_base_entry.grid(row=23, column=0,  padx=(220), pady=1, stick="nw")
            else:
                nativectx_label.grid(row=23, column=0, padx=8, pady=1, stick="nw")
                nativectx_entry.grid(row=23, column=0, padx=(146), pady=1, stick="nw")

    manualropebox = makecheckbox(context_tab, "Manual Rope Scale", variable=manualrope_var, row=22, command=togglerope, padx=(200), tooltiptxt="Set RoPE base and scale manually.")

    makecheckbox(context_tab, "Custom RoPE Config", variable=customrope_var, row=22, command=togglerope,tooltiptxt="Override the default RoPE configuration with custom RoPE scaling.")
    noqkvlabel = makelabel(context_tab,"(Note: QuantKV works best with flash attention)",30,0,"Only K cache can be quantized, and performance can suffer.\nIn some cases, it might even use more VRAM when doing a full offload.",padx=160)
    noqkvlabel.configure(text_color="#ff5555")
    qkvslider,qkvlabel,qkvtitle = makeslider(context_tab, "Quantize KV Cache:", quantkv_text, quantkv_var, 30, set=0,tooltip="Enable quantization of KV cache.\nRequires Flash Attention for full effect, otherwise only K cache is quantized.")
    quantkv_var.trace_add("write", toggleflashattn)
    makecheckbox(context_tab, "No BOS Token", nobostoken_var, 43, tooltiptxt="Prevents BOS token from being added at the start of any prompt. Usually NOT recommended for most models.")
    makecheckbox(context_tab, "Enable Guidance", enableguidance_var, 43,padx=(140), tooltiptxt="Enables the use of Classifier-Free-Guidance, which allows the use of negative prompts. Has performance and memory impact.")
    def togglejinja(a,b,c):
        if jinja_var.get()==1:
            jinjatoolsbox.grid()
            jinjakwargsbox.grid()
            jinjakwargsboxlbl.grid()
        else:
            jinja_tools_var.set(0)
            jinjatoolsbox.grid_remove()
            jinjakwargsbox.grid_remove()
            jinjakwargsboxlbl.grid_remove()
        changed_gpulayers_estimate()
    makecheckbox(context_tab, "Use Jinja", jinja_var, row=45, command=togglejinja, tooltiptxt="Enables using jinja chat template formatting for chat completions endpoint. Other endpoints are unaffected.")
    jinjatoolsbox = makecheckbox(context_tab, "Jinja for Tools", jinja_tools_var, row=45 ,padx=(140), tooltiptxt="Allows jinja even with tool calls. If unchecked, jinja will be disabled when tools are used.")
    jinjakwargsbox,jinjakwargsboxlbl = makelabelentry(context_tab, "Jj.Kwargs:", jinja_kwargs_var, row=45, width=80, padx=(350), singleline=True, tooltip='Set additiona fields for Jinja JSON template parser, must be a valid json object.\nSpecified as JSON fields: {"KEY1":"VALUE1", "KEY2":"VALUE2"...}', labelpadx=285)
    jinja_var.trace_add("write", togglejinja)
    makelabelentry(context_tab, "MoE Experts:", moeexperts_var, row=55, padx=(86), singleline=True, tooltip="Override number of MoE experts.")
    moecpu_box,moecpu_box_lbl = makelabelentry(context_tab, "MoE CPU Layers:", moecpu_var, row=55, padx=(334), singleline=True, tooltip="Force Mixture of Experts (MoE) weights of the first N layers to the CPU.\nSetting it higher than GPU layers has no effect.", labelpadx=(230))
    makelabelentry(context_tab, "Override KV:", override_kv_var, row=57, padx=(86), singleline=True, width=130, tooltip="Override metadata value by key. Separate multiple values with commas. Format is name=type:value. Types: int, float, bool, str")
    tenos_box,tenos_box_lbl = makelabelentry(context_tab, "Override Tensors:", override_tensors_var, row=57, padx=(334), singleline=True, width=130, tooltip="Override selected backend for specific tensors matching tensor_name_regex_pattern=buffer_type, same as in llama.cpp.", labelpadx=(230))

    # Model Tab
    model_tab = tabcontent["Loaded Files"]

    makefileentry(model_tab, "Text Model:", "Select GGUF or GGML Model File", model_var, 1,width=205,singlerow=True, onchoosefile=on_picked_model_file,tooltiptxt="Select a GGUF or GGML model file on disk to be loaded.")
    ctk.CTkButton(model_tab, width=70, text = "HF Search", command = model_searcher ).grid(row=1,column=0, stick="nw", padx=(370), pady=2)
    makefileentry(model_tab, "Text Lora:", "Select Lora File",lora_var, 3,width=160,singlerow=True,tooltiptxt="Select an optional GGML Text LoRA adapter to use.\nLeave blank to skip.")
    makelabelentry(model_tab, "Multiplier: ", loramult_var, 3, 50,padx=(390),singleline=True,tooltip="Scale multiplier for Text LoRA Strength. Default is 1.0", labelpadx=(330))
    makefileentry(model_tab, "Mmproj File:", "Select Audio or Vision mmproj File", mmproj_var, 7,width=280,singlerow=True,tooltiptxt="Select a mmproj file to use for multimodal models for vision and audio recognition.\nLeave blank to skip.")
    makelabelentry(model_tab, "Vision MaxRes:", visionmaxres_var, 9, width=40, padx=(100), singleline=True, tooltip=f"Clamp MMProj vision maximum allowed resolution. Allowed values are between 512 to 2048 px (default {state.default_visionmaxres}).")
    makelabelentry(model_tab, "V.Min/Max Tok:", vision_min_tokens_var, 9, width=36, padx=(244), singleline=True, tooltip="Override the minimum tokens for the MMProj embedding (default -1).", labelpadx=(150))
    makelabelentry(model_tab, "", vision_max_tokens_var, 9, padx=(284),width=36, singleline=True, tooltip="Override the maximum tokens for the MMProj embedding (default -1).", labelpadx=(260))
    makecheckbox(model_tab, "V.Force CPU", mmprojcpu_var, 9, padx=340, tooltiptxt="Force CLIP for Vision mmproj always on CPU.")
    makefileentry(model_tab, "Draft Model:", "Select Speculative Text Model File", draftmodel_var, 11,width=280,singlerow=True,tooltiptxt="Select a draft text model file to use for speculative decoding.\nLeave blank to skip.")
    makelabelentry(model_tab, "Draft Amount: ", draftamount_var, 13, 50,padx=(100),singleline=True,tooltip="How many tokens to draft per chunk before verifying results")
    makelabelentry(model_tab, "Splits: ", draftgpusplit_str_vars, 13, 50,padx=(210),singleline=True,tooltip="Distribution of draft model layers. Leave blank to follow main model's gpu split. Only works if multi-gpu (All) selected in main model.", labelpadx=(160))
    makelabelentry(model_tab, "Layers: ", draftgpulayers_var, 13, 50,padx=(320),singleline=True,tooltip="How many layers to GPU offload for the draft model", labelpadx=(270))
    makefileentry(model_tab, "Embeds Model:", "Select Embeddings Model File", embeddings_model_var, 15, width=130,singlerow=True, filetypes=[("*.gguf","*.gguf")], tooltiptxt="Select an embeddings GGUF model that can be used to generate embedding vectors.")
    makelabelentry(model_tab, "ECtx: ", embeddings_ctx_var, 15, 50,padx=(335),singleline=True,tooltip="If set above 0, limits max context for embedding model to save memory.", labelpadx=(302))
    makecheckbox(model_tab, "GPU", embeddings_gpu_var, 15, 0,padx=(390),tooltiptxt="Uses the GPU for Embeddings.")
    embeddings_gpu_var.trace_add("write", gui_changed_modelfile)
    makefileentry(model_tab, "Preload Story:", "Select Preloaded Story File", preloadstory_var, 17,width=280,singlerow=True,tooltiptxt="Select an optional KoboldAI JSON savefile \nto be served on launch to any client.")
    makefileentry(model_tab, "SaveData File:", "Select or Create New SaveData Database File", savedatafile_var, 19,width=280,filetypes=[("KoboldCpp SaveDB", "*.jsondb")],singlerow=True,dialog_type=1,tooltiptxt="Selecting a file will allow data to be loaded and saved persistently to this KoboldCpp server remotely. File is created if it does not exist.")
    makefileentry(model_tab, "MCP JSON:", "Select a mcp.json configuration file", mcpfile_var, 21,width=280,filetypes=[("MCP JSON", "*.json")],singlerow=True,tooltiptxt="Specify path to mcp.json which contains the Claude Desktop compatible MCP server config.")
    makefileentry(model_tab, "Chat Adapter:", "Select ChatCompletions Adapter File", chatcompletionsadapter_var, 24, width=184, filetypes=[("JSON Adapter", "*.json")], singlerow=True, tooltiptxt="Select an optional ChatCompletions Adapter JSON file to force custom instruct tags.")
    def pickpremadetemplate():
        initialDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kcpp_adapters')
        initialDir = initialDir if os.path.isdir(initialDir) else None
        fnam = zentk_askopenfilename(title="Pick Premade ChatCompletions Adapter",filetypes=[("JSON Adapter", "*.json")], initialdir=initialDir)
        if fnam:
            chatcompletionsadapter_var.set(fnam)
    ctk.CTkButton(model_tab, 64, text="Pick Premade", command=pickpremadetemplate).grid(row=24, column=0, padx=(350), pady=2, stick="nw")
    makefileentry(model_tab, "Jinja Template:", "Select a custom Jinja chat template", jinjatemplate_var, 30, width=280, filetypes=[("Jinja Template", "*.jinja")], singlerow=True, tooltiptxt="Select a custom Jinja chat template, will overwrite model jinja chat template")

    mmproj_var.trace_add("write", gui_changed_modelfile)
    draftmodel_var.trace_add("write", gui_changed_modelfile)
    makefileentry(model_tab, "Download Dir:", "Select directory to store all model downloads", download_dir_var, 35, width=280, singlerow=True, dialog_type=2, tooltiptxt="Specify a directory to store any downloaded models.")
    makecheckbox(model_tab, "Allow Launch Without Models", nomodel, 40, tooltiptxt="Allows running the WebUI with no model loaded.")

    # Network Tab
    network_tab = tabcontent["Network"]

    # interfaces
    makelabelentry(network_tab, "Port: ", port_var, 1, 150,tooltip=f"Select the port to host the KoboldCPP webserver.\n(Defaults to {state.defaultport})")
    makelabelentry(network_tab, "Host: ", host_var, 2, 150,tooltip="Select a specific host interface to bind to.\n(Defaults to all)")

    makecheckbox(network_tab, "Remote Tunnel", remotetunnel_var, 3,tooltiptxt="Creates a trycloudflare tunnel.\nAllows you to access koboldcpp from other devices over an internet URL.")
    makecheckbox(network_tab, "Quiet Mode", quietmode, 4,tooltiptxt="Prevents all generation related terminal output from being displayed.")
    makecheckbox(network_tab, "NoCertify Mode (Insecure)", nocertifymode, 4, 1,tooltiptxt="Allows insecure SSL connections. Use this if you have cert errors and need to bypass certificate restrictions.")
    makecheckbox(network_tab, "Shared Multiplayer", multiplayer_var, 5,tooltiptxt="Hosts a shared multiplayer session that others can join.")
    makecheckbox(network_tab, "Enable WebSearch", websearch_var, 5, 1,tooltiptxt="Enable the local search engine proxy so Web Searches can be done.")

    makefileentry(network_tab, "SSL Cert:", "Select SSL cert.pem file",ssl_cert_var, 7, width=200 ,filetypes=[("Unencrypted Certificate PEM", "*.pem")], singlerow=True, singlecol=False,tooltiptxt="Select your unencrypted .pem SSL certificate file for https.\nCan be generated with OpenSSL.")
    makefileentry(network_tab, "SSL Key:", "Select SSL key.pem file", ssl_key_var, 9, width=200, filetypes=[("Unencrypted Key PEM", "*.pem")], singlerow=True, singlecol=False, tooltiptxt="Select your unencrypted .pem SSL key file for https.\nCan be generated with OpenSSL.")
    makelabelentry(network_tab, "Password: ", password_var, 10, 200,tooltip="Enter a password required to use this instance.\nThis key will be required for all text endpoints.\nImage endpoints are not secured.")

    makelabelentry(network_tab, "Multiuser Queue:", multiuser_var, row=20, width=50, tooltip="Maximum queued incoming requests.")
    makelabelentry(network_tab, "Max Req. Size (MB):", maxrequestsize_var, row=22, width=50, tooltip="Specify a max request payload size. Any requests to the server larger than this size will be dropped. Do not change if unsure.")
    makelabelentry(network_tab, "IP Rate Limiter (s):", ratelimit_var, row=24, width=50, tooltip="Rate limits each IP to allow a new request once per X seconds. Do not change if unsure.")
    makelabelentry(network_tab, "Request Timeout (s):", reqtimeout_var, row=26, width=50, tooltip="Timeout in seconds for HTTP requests")


    # Horde Tab
    horde_tab = tabcontent["Horde Worker"]
    makelabel(horde_tab, "Horde:", 18,0,"Settings for embedded AI Horde worker").grid(pady=10)

    horde_name_entry,  horde_name_label = makelabelentry(horde_tab, "Horde Model Name:", horde_name_var, 20, 180,tooltip="The model name to be displayed on the AI Horde.")
    horde_gen_entry,  horde_gen_label = makelabelentry(horde_tab, "Gen. Length:", horde_gen_var, 21, 50,tooltip="The maximum amount to generate per request that this worker will accept jobs for.")
    horde_context_entry,  horde_context_label = makelabelentry(horde_tab, "Max Context:",horde_context_var, 22, 50,tooltip="The maximum context length that this worker will accept jobs for.\nIf 0, matches main context limit.")
    horde_apikey_entry,  horde_apikey_label = makelabelentry(horde_tab, "API Key (If Embedded Worker):",horde_apikey_var, 23, 180,tooltip="Your AI Horde API Key that you have registered.")
    horde_workername_entry,  horde_workername_label = makelabelentry(horde_tab, "Horde Worker Name:",horde_workername_var, 24, 180,tooltip="Your worker's name to be displayed.")

    def togglehorde(a,b,c):
        horde_items = zip([horde_name_entry, horde_gen_entry, horde_context_entry, horde_apikey_entry, horde_workername_entry],
                          [horde_name_label, horde_gen_label, horde_context_label, horde_apikey_label, horde_workername_label])

        for item, label in horde_items:
            if usehorde_var.get() == 1:
                item.grid()
                label.grid()
            else:
                item.grid_remove()
                label.grid_remove()
        if usehorde_var.get()==1 and (horde_name_var.get()=="koboldcpp" or horde_name_var.get()=="") and model_var.get()!="":
            basefile = os.path.basename(model_var.get())
            horde_name_var.set(sanitize_string(os.path.splitext(basefile)[0]))

    makecheckbox(horde_tab, "Configure for Horde", usehorde_var, 19, command=togglehorde,tooltiptxt="Enable the embedded AI Horde worker.")

    # Image Gen Tab

    images_tab = tabcontent["Image Gen"]
    makefileentry(images_tab, "Image Model:", "Select Image Gen Model File (safetensors/gguf)", sd_model_var, 1, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")], tooltiptxt="Select a .safetensors or .gguf Image Generation model file on disk to be loaded.")

    def togglesdlora(a,b,c):
        if sd_runtime_loras_var.get()==1:
            imglora1.grid_remove()
            imglora2.grid_remove()
            imglora3.grid_remove()
            imglora4.grid_remove()
            imglora5.grid_remove()
            imglora6.grid()
            imglora7.grid()
            imglora8.grid()
        else:
            imglora1.grid()
            imglora2.grid()
            imglora3.grid()
            imglora4.grid()
            imglora5.grid()
            imglora6.grid_remove()
            imglora7.grid_remove()
            imglora8.grid_remove()

    makelabelentry(images_tab, "Clamp Resolution (Hard):", sd_clamped_var, 14, 50, padx=(150),singleline=True,tooltip="Limit generation steps and output image size for shared use.\nSet to 0 to disable, otherwise value is clamped to the max size limit (min 512px).")
    makelabelentry(images_tab, "(Soft):", sd_clamped_soft_var, 14, 50, padx=(250),singleline=True,tooltip="Square image size restriction, to protect the server against memory crashes.\nAllows width-height tradeoffs, eg. 640 allows 640x640 and 512x768\nLeave at 0 for the default value: 832 for SD1.5/SD2, 1024 otherwise.",labelpadx=(210))
    makecheckbox(images_tab, "Runtime LoRAs", sd_runtime_loras_var, 14,command=togglesdlora, padx=(310), tooltiptxt="Allow using LoRAs in a directory dynamically (syntax is <lora:name:weight>)")

    makelabelentry(images_tab, "ImgThreads:" , sd_threads_var, 8, 40,padx=(280),singleline=True,tooltip="How many threads to use during image generation.\nIf left blank, uses same value as threads.",labelpadx=(200))
    makelabelentry(images_tab, "ImgGPU:" , sd_main_gpu_var, 8, 40,padx=394,singleline=True,tooltip="Which GPU ID to use for Image Gen?\nIf left blank or -1, uses default value.",labelpadx=340)
    sd_model_var.trace_add("write", gui_changed_modelfile)
    makelabelcombobox(images_tab, "Compress Weights: ", sd_quant_var, 8, width=(60), padx=(126), labelpadx=8, tooltiptxt="Quantizes the SD model weights to save memory.\nHigher levels save more memory, and cause more quality degradation.", values=sd_quant_choices)
    sd_quant_var.trace_add("write", changed_gpulayers_estimate)

    imglora1,imglora2,imglora3 = makefileentry(images_tab, "Image LoRAs:", "Select SD lora files to load",sd_lora_var, 20, width=160, singlerow=True, filetypes=[("*.safetensors *.gguf", "*.safetensors *.gguf")],tooltiptxt="Select multiple .safetensors or .gguf SD LoRA model files to be loaded. Should be unquantized!", multiple=True)
    imglora4,imglora5 = makelabelentry(images_tab, "Multiplier:" , sd_loramult_var, 20, 50,padx=(390),singleline=True,tooltip="What mutiplier value to apply the SD LoRA with.",labelpadx=(330))
    imglora6,imglora7,imglora8 = makefileentry(images_tab, "LoRA Dir:", "Select directory for runtime lora triggers",sd_lora_var, 20, width=280, singlerow=True, dialog_type=2,tooltiptxt="Select directory containing LoRAs that can be used at runtime.\nSyntax is <lora:name:weight>")

    makefileentry(images_tab, "T5-XXL File:", "Select T5-XXL model file (SD3, Flux, WAN)",sd_t5xxl_var, 24, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select a .safetensors t5xxl file to be loaded.")
    makefileentry(images_tab, "Clip-1 File:", "Select First Clip model file (Clip-L for SD3 or Flux, or other vision encoder)",sd_clip1_var, 26, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select a .safetensors Clip-1 file to be loaded.\nThis is Clip-L for SD3 and Flux, Clip Vision for WAN, and Qwen2.5VL for QwenImage")
    makefileentry(images_tab, "Clip-2 File:", "Select Second Clip model file (Clip-G for SD3)",sd_clip2_var, 28, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select a .safetensors Clip-2 file to be loaded.\nThis is Clip-G for SD3")
    makefileentry(images_tab, "PhotoMaker:", "Select Optional PhotoMaker model file (SDXL)",sd_photomaker_var, 30, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="PhotoMaker is a model that allows face cloning.\nSelect a .safetensors PhotoMaker file to be loaded (SDXL only).")
    makefileentry(images_tab, "Upscaler:", "Select Optional Upscaling model file (ESRGAN)",sd_upscaler_var, 32, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf *.pth","*.safetensors *.gguf *.pth")],tooltiptxt="Select an upscaler model file.\nCurrently only ESRGAN is supported.")


    sdvaeitem1,sdvaeitem2,sdvaeitem3 = makefileentry(images_tab, "Image VAE:", "Select Optional SD VAE file",sd_vae_var, 40, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf", "*.safetensors *.gguf")],tooltiptxt="Select a .safetensors or .gguf SD VAE file to be loaded.")
    def toggletaesd(a,b,c):
        if sd_vaeauto_var.get()==1:
            sdvaeitem1.grid_remove()
            sdvaeitem2.grid_remove()
            sdvaeitem3.grid_remove()
        else:
            if not sdvaeitem1.grid_info() or not sdvaeitem2.grid_info() or not sdvaeitem3.grid_info():
                sdvaeitem1.grid()
                sdvaeitem2.grid()
                sdvaeitem3.grid()
    makecheckbox(images_tab, "Automatic VAE (TAE SD)", sd_vaeauto_var, 42,command=toggletaesd,tooltiptxt="Replace VAE with TAESD. May fix bad VAE.")
    makelabelcombobox(images_tab, "Conv2D Direct:", sd_convdirect_var, row=42, labelpadx=(220), padx=(310), width=90, tooltiptxt="Use Conv2D Direct operation. May save memory or improve performance.\nMight crash if not supported by the backend.\n", values=sd_convdirect_choices)
    makelabelentry(images_tab, "VAE Tiling Threshold:", sd_tiled_vae_var, 44, 50, padx=(144),singleline=True,tooltip="Enable VAE Tiling for images above this size, to save memory.\nSet to 0 to disable VAE tiling.")
    makecheckbox(images_tab, "SD Flash Attention", sd_flash_attention_var, 44,padx=(230), tooltiptxt="Enable Flash Attention for image diffusion. May save memory or improve performance.")
    makecheckbox(images_tab, "Model CPU Offload", sd_offload_cpu_var, 50,padx=8, tooltiptxt="Offload image weights in RAM to save VRAM, swap into VRAM when needed.")
    makecheckbox(images_tab, "VAE on CPU", sd_vae_cpu_var, 50,padx=(160), tooltiptxt="Force VAE to CPU only for image generation.")
    makecheckbox(images_tab, "CLIP on GPU", sd_clip_gpu_var, 50,padx=(280), tooltiptxt="Put CLIP and T5 to GPU for image generation. Otherwise, CLIP will use CPU.")

    # audio tab
    audio_tab = tabcontent["Audio"]
    makefileentry(audio_tab, "Whisper Model (Speech-To-Text):", "Select Whisper .bin Model File", whisper_model_var, 1, width=280, filetypes=[("*.bin","*.bin")], tooltiptxt="Select a Whisper .bin model file on disk to be loaded for Voice Recognition.")
    whisper_model_var.trace_add("write", gui_changed_modelfile)
    makefileentry(audio_tab, "TTS Model (Text-To-Speech):", "Select TTS GGUF Model File", tts_model_var, 3, width=280, filetypes=[("*.gguf","*.gguf")], tooltiptxt="Select a TTS GGUF model file on disk to be loaded for Narration.")
    tts_model_var.trace_add("write", gui_changed_modelfile)
    makelabelentry(audio_tab, "TTS Threads:" , tts_threads_var, 5, 50,padx=100,singleline=True,tooltip="How many threads to use during TTS generation.\nIf left blank, uses same value as threads.")
    makelabelentry(audio_tab, "TTS Max Tokens:" , ttsmaxlen_var, 5, 50,padx=300,singleline=True,tooltip="Max allowed audiotokens to generate per TTS request.", labelpadx=190)
    makecheckbox(audio_tab, "TTS Use GPU", ttsgpu_var, 9, 0,tooltiptxt="Uses the GPU for TTS. Currently only works on certain models (OuteTTS/Q3TTS).")
    ttsgpu_var.trace_add("write", gui_changed_modelfile)
    makefileentry(audio_tab, "WavTokenizer Model (Required for some models):", "Select WavTokenizer GGUF Model File", wavtokenizer_var, 11, width=280, filetypes=[("*.gguf","*.gguf")], tooltiptxt="Select a WavTokenizer GGUF model file on disk to be loaded for Narration.")
    wavtokenizer_var.trace_add("write", gui_changed_modelfile)
    makefileentry(audio_tab, "TTS Voices Dir:", "Select directory containing voices for voice cloning", tts_dir_var, 20, width=280, singlerow=True, dialog_type=2, tooltiptxt="Select directory containing voices for voice cloning")

    makefileentry(audio_tab, "MusicLLM:", "Select music LLM model (e.g acestep-5Hz-lm-0.6B)", musicllm_var, 30, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music LLM model (e.g acestep-5Hz-lm)")
    makefileentry(audio_tab, "MusicEmbeds:", "Select music embedding model (e.g Qwen3-Embedding-0.6B)", musicembeddings_var, 32, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music embedding model (e.g Qwen3-Embedding-0.6B)")
    makefileentry(audio_tab, "MusicDiffuser:", "Select music diffusion (DiT) model (e.g acestep-v15-turbo)", musicdiffusion_var, 34, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music diffusion (DiT) model (e.g acestep-v15-turbo)")
    makefileentry(audio_tab, "MusicVAE:", "Select music VAE model", musicvae_var, 36, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music VAE model")
    makecheckbox(audio_tab, "Music Low VRAM", musiclowvram_var, 38, 0,tooltiptxt="Unload music models when not in use.")

    admin_tab = tabcontent["Admin"]
    def toggleadmin(a,b,c):
        if admin_var.get()==1 and admin_dir_var.get()=="":
            autopath = os.path.realpath(__file__)
            if getattr(sys, 'frozen', False):
                autopath = sys.executable
            autopath = os.path.dirname(autopath)
            admin_dir_var.set(autopath)
        if admin_var.get()==1:
            router_mode_box.grid()
        else:
            router_mode_box.grid_remove()
        togglerouter(1,1,1)

    def togglerouter(a,b,c):
        if router_mode_var.get()==1 and admin_var.get()==1:
            autoswap_mode_box.grid()
        else:
            autoswap_mode_box.grid_remove()

    makecheckbox(admin_tab, "Enable Model Administration", admin_var, 1, 0, command=toggleadmin,tooltiptxt="Enable a admin server, allowing you to remotely relaunch and swap models and configs.")
    makelabelentry(admin_tab, "Admin Password:" , admin_password_var, 3, 150,padx=(120),singleline=True,tooltip="Require a password to access admin functions. You are strongly advised to use one for publically accessible instances!")
    makefileentry(admin_tab, "Config Directory (Required):", "Select directory containing .gguf or .kcpps files to relaunch from", admin_dir_var, 5, width=280, dialog_type=2, tooltiptxt="Specify a directory to look for .kcpps configs in, which can be used to swap models.")
    makefileentry(admin_tab, "Base config .kcpps (For reloading):", "", baseconfig_var, 7, width=280, dialog_type=0, tooltiptxt="Specify a base .kcpps config to apply, if no custom base config is selected during a model swap.")
    makelabelentry(admin_tab, "Auto Unload Timeout:" , admin_unload_timeout_var, 17, 70,padx=(150),singleline=True,tooltip="Set an idle timeout in seconds after which KoboldCpp will automatically unload the current model.")
    makecheckbox(admin_tab, "SingleInstance Mode", singleinstance_var, 19, 0,tooltiptxt="Allows this server to be shut down by another KoboldCpp instance with singleinstance starting on the same port.")
    router_mode_box = makecheckbox(admin_tab, "Router Mode", router_mode_var, 21, 0, command=togglerouter, tooltiptxt="Router mode uses a reverse proxy router, allowing you to easily hotswap models and configs within a single request. Requires admin mode.")
    autoswap_mode_box = makecheckbox(admin_tab, "Autoswap Mode", autoswap_mode_var, 23, 0,tooltiptxt="Autoswap mode builds on router mode to allow switching of model types within the same config automatically. Requires admin mode and router mode. All models desired must be defined within the same config.")

    def kcpp_export_template():
        nonlocal kcpp_exporting_template
        kcpp_exporting_template = True
        export_vars()
        kcpp_exporting_template = False
        savdict = json.loads(json.dumps(state.args.__dict__,indent=2))
        for key in state.deprecated_keys:
            savdict.pop(key, None)  # avoids KeyError if missing
        file_type = [("KoboldCpp LaunchTemplate", "*.kcppt")]
        #remove blacklisted fields
        savdict = convert_args_to_template(savdict)
        filename = zentk_asksaveasfilename(filetypes=file_type, defaultextension=".kcppt")
        if not filename:
            return
        filenamestr = str(filename).strip()
        if not filenamestr.endswith(".kcppt"):
            filenamestr += ".kcppt"
        file = open(filenamestr, 'w')
        file.write(json.dumps(savdict,indent=2))
        file.close()
        pass

    # extra tab
    extra_tab = tabcontent["Extra"]
    makelabel(extra_tab, "Extract KoboldCpp Files", 3, 0,tooltiptxt="Unpack KoboldCpp to a local directory to modify its files. You can also launch via koboldcpp.py for faster startup.")
    ctk.CTkButton(extra_tab , text = "Unpack KoboldCpp To Folder", command = unpack_to_dir ).grid(row=3,column=0, stick="w", padx=(170), pady=2)
    makelabel(extra_tab, "Export as .kcppt template", 4, 0,tooltiptxt="Creates a KoboldCpp launch template for others to use.\nEmbeds JSON files directly into exported file when saving.\nWhen loaded, forces the backend to be automatically determined.\nWarning! Not recommended for beginners!")
    ctk.CTkButton(extra_tab , text = "Generate LaunchTemplate", command = kcpp_export_template ).grid(row=4,column=0, stick="w", padx=(170), pady=2)
    makelabel(extra_tab, "Analyze GGUF Metadata", 6, 0,tooltiptxt="Reads the metadata, weight types and tensor names in any GGUF file.")
    ctk.CTkButton(extra_tab , text = "Analyze GGUF", command = analyze_gguf_model_wrapper ).grid(row=6,column=0, stick="w", padx=(170), pady=2)
    if os.name == 'nt':
        makelabel(extra_tab, "File Extensions Handler", 10, 0,tooltiptxt="Makes KoboldCpp the default handler for .kcpps, .kcppt, .ggml and .gguf files.")
        ctk.CTkButton(extra_tab , text = "Register", width=90, command = register_koboldcpp ).grid(row=10,column=0, stick="w", padx= (170), pady=2)
        ctk.CTkButton(extra_tab , text = "Unregister", width=90, command = unregister_koboldcpp ).grid(row=10,column=0, stick="w", padx= (264), pady=2)
    if sys.platform == "linux":
        def togglezenity(a,b,c):
            zenity_permitted = (nozenity_var.get()==0)
        makecheckbox(extra_tab, "Use Classic FilePicker", nozenity_var, 20, tooltiptxt="Use the classic TKinter file picker instead.")
        nozenity_var.trace_add("write", togglezenity)

    extra_terminal_process = None
    def showtermlogs():
        nonlocal extra_terminal_process
        try:
            if extra_terminal_process and extra_terminal_process.poll() is None:
                print("Error: Secondary terminal already running.")
                return
            if sys.platform == "linux":
                # Create an unnamed pipe, launch a terminal that reads from the read-end FD
                r, w = os.pipe()
                extra_terminal_process = subprocess.Popen(["xterm", "-hold","-e", f"bash -c 'cat <&{r}'"], pass_fds=[r])
                writer = os.fdopen(w, "w", buffering=1)
                redirector = StdoutRedirector(writer)
                sys.stdout = redirector
                print("--- Secondary Linux Terminal Active ---")
            else:
                print("Error: Secondary Terminal Not Supported on this Platform")
        except Exception as e:
            print(f"Spawn Extra Terminal Failed: {e}")
    if sys.platform == "linux":
        makelabel(extra_tab, "Spawn Terminal Logs", 12, 0,tooltiptxt="A simple terminal logger that duplicates the command line output.")
        ctk.CTkButton(extra_tab , text = "Spawn Terminal", command = showtermlogs ).grid(row=12,column=0, stick="w", padx= 170, pady=2)

    # refresh
    runopts_var.trace_add("write", changerunmode)
    changerunmode(1,1,1)
    runmode_untouched = True
    togglerope(1,1,1)
    toggleflashattn(1,1,1)
    togglectxshift(1,1,1)
    togglehorde(1,1,1)
    toggletaesd(1,1,1)
    togglesdlora(1,1,1)
    togglejinja(1,1,1)
    toggleadmin(1,1,1)

    # launch
    def guilaunch():
        if model_var.get() == "" and sd_model_var.get() == "" and whisper_model_var.get() == "" and tts_model_var.get() == "" and embeddings_model_var.get() == "" and musicdiffusion_var.get() == "" and musicllm_var.get() == "" and nomodel.get()!=1:
            # prevent launch without at least one valid model
            givehelp = show_gui_yesnobox("No Models Selected","Error: You need to load at least one AI model to continue.\n\nDo you want help finding a model?")
            if givehelp == 'yes':
                display_help()
            return
        nonlocal nextstate
        nextstate = 1
        root.withdraw()
        root.quit()
        pass

    def export_vars():
        nonlocal kcpp_exporting_template
        state.args.threads =  (get_default_threads() if threads_var.get()=="" else int(threads_var.get()))
        state.args.usemlock   = usemlock.get() == 1
        state.args.debugmode  = debugmode.get()
        state.args.launch     = launchbrowser.get()==1
        state.args.highpriority = highpriority.get()==1
        state.args.usemmap = usemmap.get()==1
        state.args.smartcontext = smartcontext_var.get()==1
        state.args.noflashattention = flashattention_var.get()==0
        state.args.noshift = contextshift_var.get()==0
        state.args.nofastforward = fastforward_var.get()==0
        state.args.useswa = swa_var.get()==1
        state.args.swapadding = int(swa_padding_var.get()) if swa_padding_var.get()!="" else state.swa_padding_default
        state.args.smartcache = (0 if smartcache_var.get()!=1 else int(smartcacheslots_var.get()))
        state.args.remotetunnel = remotetunnel_var.get()==1
        state.args.foreground = keepforeground.get()==1
        state.args.cli = terminalonly.get()==1
        state.args.nopipelineparallel = pipelineparallel.get()==0
        state.args.quiet = quietmode.get()==1
        state.args.nocertify = nocertifymode.get()==1
        state.args.nomodel = nomodel.get()==1
        qkvopt = quantkv_text[quantkv_var.get()].lower() if (quantkv_var.get()>=0 and quantkv_var.get() < len(quantkv_text)) else "f16"
        state.args.quantkv = qkvopt
        state.args.lowvram = lowvram_var.get()==1
        state.args.nommq = mmq_var.get()==0
        state.args.splitmode = splitmode_var.get() if splitmode_var.get() in splitmode_choices else splitmode_choices[0]

        gpuchoiceidx = 0
        state.args.usecpu = False
        state.args.usevulkan = None
        state.args.usecuda = None
        state.args.noavx2 = False
        state.args.failsafe = False
        if gpu_choice_var.get()!="All":
            gpuchoiceidx = int(gpu_choice_var.get())-1
        if runopts_var.get() == "Use CUDA" or runopts_var.get() == "Use hipBLAS (ROCm)":
            if gpu_choice_var.get()=="All":
                state.args.usecuda = ["normal"]
            else:
                state.args.usecuda = ["normal",str(gpuchoiceidx)]
        if runopts_var.get() == "Use Vulkan" or runopts_var.get() == "Use Vulkan (Old CPU)" or runopts_var.get() == "Use Vulkan (Older CPU)":
            if gpu_choice_var.get()=="All":
                state.args.usevulkan = []
            else:
                state.args.usevulkan = [int(gpuchoiceidx)]
            if runopts_var.get() == "Use Vulkan (Old CPU)":
                state.args.noavx2 = True
            elif runopts_var.get() == "Use Vulkan (Older CPU)":
                state.args.noavx2 = True
                state.args.failsafe = True

        state.args.gpulayers = (-1 if gpulayers_var.get()=="" else int(gpulayers_var.get()))
        state.args.autofitpadding = (state.default_autofit_padding if autofit_padding_var.get()=="" else int(autofit_padding_var.get()))
        if runopts_var.get()=="Use CPU":
            state.args.usecpu = True
        if runopts_var.get()=="Use CPU (Old CPU)":
            state.args.noavx2 = True
            state.args.usecpu = True
        if runopts_var.get()=="Failsafe Mode (Older CPU)":
            state.args.noavx2 = True
            state.args.usecpu = True
            state.args.usemmap = False
            state.args.failsafe = True
        state.args.tensor_split = None
        if tensor_split_str_vars.get()!="":
            tssv = tensor_split_str_vars.get()
            if "," in tssv:
                state.args.tensor_split = [float(x) for x in tssv.split(",")]
            else:
                state.args.tensor_split = [float(x) for x in tssv.split(" ")]
        state.args.draftgpusplit = None
        if draftgpusplit_str_vars.get()!="":
            tssv = draftgpusplit_str_vars.get()
            if "," in tssv:
                state.args.draftgpusplit = [float(x) for x in tssv.split(",")]
            else:
                state.args.draftgpusplit = [float(x) for x in tssv.split(" ")]

        state.args.maingpu = -1 if maingpu_var.get()=="" else int(maingpu_var.get())
        state.args.blasthreads = None if blas_threads_var.get()=="" else int(blas_threads_var.get())
        state.args.device = deviceoverride_var.get()
        state.args.batchsize = int(batchsize_values[int(blas_size_var.get())])
        state.args.autofit = autofit_var.get() == 1
        state.args.contextsize = int(contextsize_text[context_var.get()])
        if customrope_var.get()==1:
            if manualrope_var.get()==1:
                state.args.ropeconfig = [float(customrope_scale.get()),float(customrope_base.get())]
                state.args.overridenativecontext = 0
            else:
                state.args.ropeconfig = [0.0, 10000.0]
                state.args.overridenativecontext = int(customrope_nativectx.get())
        else:
            state.args.ropeconfig = [0.0, 10000.0]
            state.args.overridenativecontext = 0
        state.args.moeexperts = int(moeexperts_var.get()) if moeexperts_var.get()!="" else -1
        state.args.moecpu = int(moecpu_var.get()) if moecpu_var.get()!="" else 0
        state.args.defaultgenamt = int(defaultgenamt_var.get()) if defaultgenamt_var.get()!="" else state.default_genlen
        state.args.genlimit = int(genlimit_var.get()) if genlimit_var.get()!="" else 0
        state.args.nobostoken = (nobostoken_var.get()==1)
        state.args.jinja = (jinja_var.get()==1)
        state.args.jinja_tools = (jinja_tools_var.get()==1)
        state.args.jinja_kwargs = jinja_kwargs_var.get()  if jinja_kwargs_var.get() != "" else ""
        state.args.jinjatemplate = jinjatemplate_var.get() if jinjatemplate_var.get() != "" else ""
        state.args.enableguidance = (enableguidance_var.get()==1)
        state.args.overridekv = None if override_kv_var.get() == "" else override_kv_var.get()
        state.args.overridetensors = None if override_tensors_var.get() == "" else override_tensors_var.get()
        state.args.chatcompletionsadapter = "AutoGuess" if chatcompletionsadapter_var.get() == "" else chatcompletionsadapter_var.get()
        try:
            if kcpp_exporting_template and isinstance(state.args.chatcompletionsadapter, str) and state.args.chatcompletionsadapter!="" and os.path.exists(state.args.chatcompletionsadapter):
                print("Embedding chat completions adapter...")   # parse and save embedded preload story
                with open(state.args.chatcompletionsadapter, 'r', encoding='utf-8', errors='ignore') as f:
                    state.args.chatcompletionsadapter = json.load(f)
        except Exception:
            pass

        state.args.model_param = None if model_var.get() == "" else model_var.get()
        state.args.lora = None if lora_var.get() == "" else ([lora_var.get()])
        state.args.loramult = (float(loramult_var.get()) if loramult_var.get()!="" else 1.0)
        pls_str_or_obj = None if preloadstory_var.get() == "" else preloadstory_var.get()
        if pls_str_or_obj and isinstance(pls_str_or_obj,str):
            try:
                temp = json.loads(pls_str_or_obj)
                pls_str_or_obj = temp
            except Exception:
                pass
        state.args.preloadstory = pls_str_or_obj
        state.args.savedatafile = None if savedatafile_var.get() == "" else savedatafile_var.get()
        state.args.mcpfile = None if mcpfile_var.get() == "" else mcpfile_var.get()
        state.args.downloaddir = download_dir_var.get()
        try:
            if kcpp_exporting_template and isinstance(state.args.preloadstory, str) and state.args.preloadstory!="" and os.path.exists(state.args.preloadstory):
                print("Embedding preload story...")   # parse and save embedded preload story
                with open(state.args.preloadstory, 'r', encoding='utf-8', errors='ignore') as f:
                    state.args.preloadstory = json.load(f)
        except Exception:
            pass
        state.args.mmproj = None if mmproj_var.get() == "" else mmproj_var.get()
        state.args.mmprojcpu = (mmprojcpu_var.get()==1)
        state.args.visionmaxres = int(visionmaxres_var.get()) if visionmaxres_var.get()!="" else state.default_visionmaxres
        state.args.visionmintokens = int(vision_min_tokens_var.get()) if vision_min_tokens_var.get()!="" else -1
        state.args.visionmaxtokens = int(vision_max_tokens_var.get()) if vision_max_tokens_var.get()!="" else -1
        state.args.draftmodel = None if draftmodel_var.get() == "" else draftmodel_var.get()
        state.args.draftamount = int(draftamount_var.get()) if draftamount_var.get()!="" else state.default_draft_amount
        state.args.draftgpulayers = int(draftgpulayers_var.get()) if draftgpulayers_var.get()!="" else 999

        state.args.ssl = None if (ssl_cert_var.get() == "" or ssl_key_var.get() == "") else ([ssl_cert_var.get(), ssl_key_var.get()])
        state.args.password = None if (password_var.get() == "") else (password_var.get())

        state.args.port_param = state.defaultport if port_var.get()=="" else int(port_var.get())
        state.args.port = state.args.port_param
        state.args.host = host_var.get()
        state.args.multiuser = int(multiuser_var.get()) if multiuser_var.get()!="" else state.multiuser_concurrent_limit
        state.args.multiplayer = (multiplayer_var.get()==1)
        state.args.websearch = (websearch_var.get()==1)
        state.args.maxrequestsize = int(maxrequestsize_var.get()) if maxrequestsize_var.get()!="" else 32
        state.args.ratelimit = int(ratelimit_var.get()) if ratelimit_var.get()!="" else 0
        state.args.reqtimeout = int(reqtimeout_var.get()) if reqtimeout_var.get()!="" else 0

        if usehorde_var.get() != 0:
            state.args.hordemodelname = horde_name_var.get()
            state.args.hordegenlen = int(horde_gen_var.get())
            state.args.hordemaxctx = int(horde_context_var.get())
            if horde_apikey_var.get()!="" and horde_workername_var.get()!="":
                state.args.hordekey = horde_apikey_var.get()
                state.args.hordeworkername = horde_workername_var.get()

        state.args.sdmodel = sd_model_var.get() if sd_model_var.get() != "" else ""
        state.args.sdflashattention = True if sd_flash_attention_var.get()==1 else False
        state.args.sdoffloadcpu = True if sd_offload_cpu_var.get()==1 else False
        state.args.sdvaecpu = True if sd_vae_cpu_var.get()==1 else False
        state.args.sdclipgpu = True if sd_clip_gpu_var.get()==1 else False
        state.args.sdthreads = (0 if sd_threads_var.get()=="" else int(sd_threads_var.get()))
        state.args.sdclamped = (0 if int(sd_clamped_var.get())<=0 else int(sd_clamped_var.get()))
        state.args.sdclampedsoft = (0 if int(sd_clamped_soft_var.get())<=0 else int(sd_clamped_soft_var.get()))
        state.args.sdtiledvae = (state.default_vae_tile_threshold if sd_tiled_vae_var.get()=="" else int(sd_tiled_vae_var.get()))
        if sd_vaeauto_var.get()==1:
            state.args.sdvaeauto = True
            state.args.sdvae = ""
        else:
            state.args.sdvaeauto = False
            state.args.sdvae = ""
            if sd_vae_var.get() != "":
                state.args.sdvae = sd_vae_var.get()
        state.args.sdconvdirect = sd_convdirect_option(sd_convdirect_var.get())
        state.args.sdt5xxl = sd_t5xxl_var.get() if sd_t5xxl_var.get() != "" else ""
        state.args.sdclip1 = sd_clip1_var.get() if sd_clip1_var.get() != "" else ""
        state.args.sdclip2 = sd_clip2_var.get() if sd_clip2_var.get() != "" else ""
        state.args.sdphotomaker = sd_photomaker_var.get() if sd_photomaker_var.get() != "" else ""
        state.args.sdupscaler = sd_upscaler_var.get()  if sd_upscaler_var.get() != "" else ""
        state.args.sdquant = sd_quant_option(sd_quant_var.get())
        state.args.sdlora = [item.strip() for item in sd_lora_var.get().split("|") if item]
        # XXX the user may have used '|' since it's used for the LoRAs
        state.args.sdloramult = sanitize_lora_multipliers(re.split(r"[ |]+", sd_loramult_var.get()))
        state.args.sdmaingpu = (-1 if sd_main_gpu_var.get()=="" else int(sd_main_gpu_var.get()))
        state.args.gendefaults = gen_defaults_var.get()  if gen_defaults_var.get() != "" else ""
        state.args.gendefaultsoverwrite = (gen_defaults_overwrite_var.get()==1)
        state.args.whispermodel = whisper_model_var.get() if whisper_model_var.get() != "" else ""
        state.args.embeddingsmodel = embeddings_model_var.get()  if embeddings_model_var.get() != "" else ""
        state.args.embeddingsmaxctx = (0 if embeddings_ctx_var.get()=="" else int(embeddings_ctx_var.get()))
        state.args.embeddingsgpu = (embeddings_gpu_var.get()==1)

        state.args.ttsthreads = (0 if tts_threads_var.get()=="" else int(tts_threads_var.get()))
        state.args.ttsmaxlen = (state.default_ttsmaxlen if ttsmaxlen_var.get()=="" else int(ttsmaxlen_var.get()))
        state.args.ttsgpu = (ttsgpu_var.get()==1)
        if tts_model_var.get() != "":
            state.args.ttsmodel = tts_model_var.get()
            state.args.ttswavtokenizer = wavtokenizer_var.get()
            state.args.ttsdir = tts_dir_var.get()
        else:
            state.args.ttsmodel = ""
            state.args.ttswavtokenizer = ""
            state.args.ttsdir = ""

        state.args.musicllm = musicllm_var.get()
        state.args.musicembeddings = musicembeddings_var.get()
        state.args.musicdiffusion = musicdiffusion_var.get()
        state.args.musicvae = musicvae_var.get()
        state.args.musiclowvram = musiclowvram_var.get()==1

        state.args.admin = (admin_var.get()==1 and not state.args.cli)
        state.args.admindir = admin_dir_var.get()
        state.args.adminpassword = admin_password_var.get()
        state.args.singleinstance = (singleinstance_var.get()==1)
        state.args.routermode = (router_mode_var.get()==1 and admin_var.get()==1)
        state.args.autoswapmode = (autoswap_mode_var.get()==1 and router_mode_var.get()==1 and admin_var.get()==1)
        state.args.baseconfig = baseconfig_var.get()
        state.args.adminunloadtimeout = (0 if admin_unload_timeout_var.get()=="" else int(admin_unload_timeout_var.get()))
        state.args.showgui = False #prevent showgui from leaking into configs, its cli only

    def import_vars(mydict):
        importvars_in_progress = True
        mydict = convert_invalid_args(mydict)

        if "threads" in mydict:
            threads_var.set(mydict["threads"])
        usemlock.set(1 if "usemlock" in mydict and mydict["usemlock"] else 0)
        if "debugmode" in mydict:
            debugmode.set(mydict["debugmode"])
        launchbrowser.set(1 if "launch" in mydict and mydict["launch"] else 0)
        highpriority.set(1 if "highpriority" in mydict and mydict["highpriority"] else 0)
        usemmap.set(1 if "usemmap" in mydict and mydict["usemmap"] else 0)
        smartcontext_var.set(1 if "smartcontext" in mydict and mydict["smartcontext"] else 0)
        flashattention_var.set(0 if "noflashattention" in mydict and mydict["noflashattention"] else 1)
        contextshift_var.set(0 if "noshift" in mydict and mydict["noshift"] else 1)
        fastforward_var.set(0 if "nofastforward" in mydict and mydict["nofastforward"] else 1)
        swa_var.set(1 if "useswa" in mydict and mydict["useswa"] else 0)
        swa_padding_var.set(mydict["swapadding"] if ("swapadding" in mydict) else state.swa_padding_default)
        smartcache_var.set(1 if "smartcache" in mydict and mydict["smartcache"] else 0)
        smartcacheslots_var.set(mydict["smartcache"] if ("smartcache" in mydict and mydict["smartcache"] and int(mydict["smartcache"])>1) else state.savestate_limit_default)
        remotetunnel_var.set(1 if "remotetunnel" in mydict and mydict["remotetunnel"] else 0)
        keepforeground.set(1 if "foreground" in mydict and mydict["foreground"] else 0)
        terminalonly.set(1 if "cli" in mydict and mydict["cli"] else 0)
        pipelineparallel.set(0 if "nopipelineparallel" in mydict and mydict["nopipelineparallel"] else 1)
        quietmode.set(1 if "quiet" in mydict and mydict["quiet"] else 0)
        nocertifymode.set(1 if "nocertify" in mydict and mydict["nocertify"] else 0)
        nomodel.set(1 if "nomodel" in mydict and mydict["nomodel"] else 0)
        lowvram_var.set(1 if "lowvram" in mydict and mydict["lowvram"] else 0)
        if "quantkv" in mydict:
            qkvstr = str(mydict["quantkv"]).lower()
            qkvval = 0
            if qkvstr=="bf16" or qkvstr=="3": #migration for old index based values
                qkvval = 1
            elif qkvstr=="q8_0" or qkvstr=="1":
                qkvval = 2
            elif qkvstr=="q5_1":
                qkvval = 3
            elif qkvstr=="q4_0" or qkvstr=="2":
                qkvval = 4
            quantkv_var.set(qkvval)
        if "usecuda" in mydict and mydict["usecuda"]:
            if cublas_option is not None or hipblas_option is not None:
                if cublas_option:
                    runopts_var.set(cublas_option)
                elif hipblas_option:
                    runopts_var.set(hipblas_option)
                gpu_choice_var.set("All")
                for g in range(4):
                    if str(g) in mydict["usecuda"]:
                        gpu_choice_var.set(str(g+1))
                        break
        elif "usevulkan" in mydict and mydict['usevulkan'] is not None:
            if "noavx2" in mydict and mydict["noavx2"]:
                if vulkan_noavx2_option is not None:
                    runopts_var.set(vulkan_noavx2_option)
                    gpu_choice_var.set("All")
                    for opt in range(0,4):
                        if opt in mydict["usevulkan"]:
                            gpu_choice_var.set(str(opt+1))
                            break
            elif "failsafe" in mydict and mydict["failsafe"]:
                if vulkan_failsafe_option is not None:
                    runopts_var.set(vulkan_failsafe_option)
                    gpu_choice_var.set("All")
                    for opt in range(0,4):
                        if opt in mydict["usevulkan"]:
                            gpu_choice_var.set(str(opt+1))
                            break
            else:
                if vulkan_option is not None:
                    runopts_var.set(vulkan_option)
                    gpu_choice_var.set("All")
                    for opt in range(0,4):
                        if opt in mydict["usevulkan"]:
                            gpu_choice_var.set(str(opt+1))
                            break

        elif ("noavx2" in mydict and "usecpu" in mydict and mydict["usecpu"] and mydict["noavx2"]) or ("failsafe" in mydict and mydict["failsafe"]):
            if failsafe_option is not None:
                runopts_var.set(failsafe_option)
        elif "noavx2" in mydict and mydict["noavx2"]:
            if noavx2_option is not None:
                runopts_var.set(noavx2_option)
        elif "usecpu" in mydict and mydict["usecpu"]:
            if default_option is not None:
                runopts_var.set(default_option)
        if "gpulayers" in mydict and mydict["gpulayers"]:
            gpulayers_var.set(mydict["gpulayers"])
        else:
            gpulayers_var.set("0")
        if "maingpu" in mydict:
            maingpu_var.set(mydict["maingpu"])
        else:
            maingpu_var.set("-1")
        if "tensor_split" in mydict and mydict["tensor_split"]:
            tssep = ','.join(map(str, mydict["tensor_split"]))
            tensor_split_str_vars.set(tssep)
        if "draftgpusplit" in mydict and mydict["draftgpusplit"]:
            tssep = ','.join(map(str, mydict["draftgpusplit"]))
            draftgpusplit_str_vars.set(tssep)
        if "blasthreads" in mydict and mydict["blasthreads"]:
            blas_threads_var.set(str(mydict["blasthreads"]))
        else:
            blas_threads_var.set("")
        if "device" in mydict and mydict["device"]:
            deviceoverride_var.set(str(mydict["device"]))
        else:
            deviceoverride_var.set("")
        if "contextsize" in mydict and mydict["contextsize"]:
            context_var.set(contextsize_text.index(str(mydict["contextsize"])))
        if "overridenativecontext" in mydict and mydict["overridenativecontext"]>0:
            customrope_var.set(1)
            manualrope_var.set(0)
            customrope_nativectx.set(str(mydict["overridenativecontext"]))
        elif "ropeconfig" in mydict and mydict["ropeconfig"] and len(mydict["ropeconfig"])>1:
            customrope_nativectx.set(state.default_native_ctx)
            if mydict["ropeconfig"][0]>0:
                customrope_var.set(1)
                manualrope_var.set(1)
                customrope_scale.set(str(mydict["ropeconfig"][0]))
                customrope_base.set(str(mydict["ropeconfig"][1]))
            else:
                customrope_var.set(0)
                manualrope_var.set(0)
        else:
            customrope_nativectx.set(state.default_native_ctx)
            customrope_var.set(0)
            manualrope_var.set(0)
        if "moeexperts" in mydict and mydict["moeexperts"]:
            moeexperts_var.set(mydict["moeexperts"])
        if "moecpu" in mydict and mydict["moecpu"]:
            moecpu_var.set(mydict["moecpu"])
        if "defaultgenamt" in mydict and mydict["defaultgenamt"]:
            defaultgenamt_var.set(mydict["defaultgenamt"])
        if "genlimit" in mydict and mydict["genlimit"]:
            genlimit_var.set(mydict["genlimit"])
        else:
            genlimit_var.set(str(0))
        nobostoken_var.set(mydict["nobostoken"] if ("nobostoken" in mydict) else 0)
        jinja_var.set(mydict["jinja"] if ("jinja" in mydict) else 0)
        jinja_tools_var.set(mydict["jinja_tools"] if ("jinja_tools" in mydict) else 0)
        jinja_kwargs = (mydict["jinja_kwargs"] if ("jinja_kwargs" in mydict and mydict["jinja_kwargs"]) else "")
        if isinstance(jinja_kwargs, type({})):
            jinja_kwargs = json.dumps(jinja_kwargs)
        jinja_kwargs_var.set(jinja_kwargs)
        jinjatemplate_var.set(mydict["jinjatemplate"] if ("jinjatemplate" in mydict and mydict["jinjatemplate"]) else "")

        enableguidance_var.set(mydict["enableguidance"] if ("enableguidance" in mydict) else 0)
        if "overridekv" in mydict and mydict["overridekv"]:
            override_kv_var.set(mydict["overridekv"])
        if "overridetensors" in mydict and mydict["overridetensors"]:
            override_tensors_var.set(mydict["overridetensors"])

        if "batchsize" in mydict and mydict["batchsize"]:
            blas_size_var.set(batchsize_values.index(str(mydict["batchsize"])))

        autofit_var.set(1 if "autofit" in mydict and mydict["autofit"] else 0)
        model_var.set(mydict["model_param"] if ("model_param" in mydict and mydict["model_param"]) else "")

        if "autofitpadding" in mydict and mydict["autofitpadding"]:
            autofit_padding_var.set(mydict["autofitpadding"])
        else:
            autofit_padding_var.set(str(state.default_autofit_padding))

        lora_var.set("")
        if "lora" in mydict and mydict["lora"]:
            if len(mydict["lora"]) > 1:
                lora_var.set(mydict["lora"][0])
            else:
                lora_var.set(mydict["lora"][0])
        loramult_var.set(str(mydict["loramult"]) if ("loramult" in mydict and mydict["loramult"]) else "1.0")

        splitmode_var.set(mydict["splitmode"] if ("splitmode" in mydict and mydict["splitmode"] in splitmode_choices) else splitmode_choices[0])
        mmq_var.set(0 if "nommq" in mydict and mydict["nommq"] else 1)

        mmproj_var.set(mydict["mmproj"] if ("mmproj" in mydict and mydict["mmproj"]) else "")
        mmprojcpu_var.set(1 if ("mmprojcpu" in mydict and mydict["mmprojcpu"]) else 0)
        if "visionmaxres" in mydict and mydict["visionmaxres"]:
            visionmaxres_var.set(mydict["visionmaxres"])
        if "visionmintokens" in mydict and mydict["visionmintokens"]:
            vision_min_tokens_var.set(mydict["visionmintokens"])
        if "visionmaxtokens" in mydict and mydict["visionmaxtokens"]:
            vision_max_tokens_var.set(mydict["visionmaxtokens"])
        draftmodel_var.set(mydict["draftmodel"] if ("draftmodel" in mydict and mydict["draftmodel"]) else "")
        if "draftamount" in mydict:
            draftamount_var.set(mydict["draftamount"])
        if "draftgpulayers" in mydict:
            draftgpulayers_var.set(mydict["draftgpulayers"])

        ssl_cert_var.set("")
        ssl_key_var.set("")
        if "ssl" in mydict and mydict["ssl"]:
            if len(mydict["ssl"]) == 2:
                ssl_cert_var.set(mydict["ssl"][0])
                ssl_key_var.set(mydict["ssl"][1])

        password_var.set(mydict["password"] if ("password" in mydict and mydict["password"]) else "")
        pls_obj = ""
        if ("preloadstory" in mydict and mydict["preloadstory"]):
            pls_obj = mydict["preloadstory"] if not isinstance(mydict["preloadstory"], dict) else json.dumps(mydict["preloadstory"])
        preloadstory_var.set(pls_obj)
        savedatafile_var.set(mydict["savedatafile"] if ("savedatafile" in mydict and mydict["savedatafile"]) else "")
        mcpfile_var.set(mydict["mcpfile"] if ("mcpfile" in mydict and mydict["mcpfile"]) else "")
        chatcompletionsadapter_var.set(mydict["chatcompletionsadapter"] if ("chatcompletionsadapter" in mydict and mydict["chatcompletionsadapter"]) else "")
        port_var.set(mydict["port_param"] if ("port_param" in mydict and mydict["port_param"]) else state.defaultport)
        host_var.set(mydict["host"] if ("host" in mydict and mydict["host"]) else "")
        multiuser_var.set(mydict["multiuser"] if ("multiuser" in mydict and mydict["multiuser"]>1) else state.multiuser_concurrent_limit)
        multiplayer_var.set(mydict["multiplayer"] if ("multiplayer" in mydict) else 0)
        websearch_var.set(mydict["websearch"] if ("websearch" in mydict) else 0)
        download_dir_var.set(mydict["downloaddir"] if ("downloaddir" in mydict and mydict["downloaddir"]) else "")

        horde_name_var.set(mydict["hordemodelname"] if ("hordemodelname" in mydict and mydict["hordemodelname"]) else "koboldcpp")
        horde_context_var.set(mydict["hordemaxctx"] if ("hordemaxctx" in mydict and mydict["hordemaxctx"]) else state.maxhordectx)
        horde_gen_var.set(mydict["hordegenlen"] if ("hordegenlen" in mydict and mydict["hordegenlen"]) else state.maxhordelen)
        horde_apikey_var.set(mydict["hordekey"] if ("hordekey" in mydict and mydict["hordekey"]) else "")
        horde_workername_var.set(mydict["hordeworkername"] if ("hordeworkername" in mydict and mydict["hordeworkername"]) else "")
        usehorde_var.set(1 if ("hordekey" in mydict and mydict["hordekey"]) else 0)
        maxrequestsize_var.set(mydict["maxrequestsize"] if ("maxrequestsize" in mydict and mydict["maxrequestsize"]) else 32)
        ratelimit_var.set(mydict["ratelimit"] if ("ratelimit" in mydict and mydict["ratelimit"]) else 0)
        reqtimeout_var.set(mydict["reqtimeout"] if ("reqtimeout" in mydict and mydict["reqtimeout"]) else 0)

        sd_model_var.set(mydict["sdmodel"] if ("sdmodel" in mydict and mydict["sdmodel"]) else "")
        sd_clamped_var.set(int(mydict["sdclamped"]) if ("sdclamped" in mydict and mydict["sdclamped"]) else 0)
        sd_clamped_soft_var.set(int(mydict["sdclampedsoft"]) if ("sdclampedsoft" in mydict and mydict["sdclampedsoft"]) else 0)
        sd_threads_var.set(str(mydict["sdthreads"]) if ("sdthreads" in mydict and mydict["sdthreads"]) else str(default_threads))
        sd_quant_var.set(sd_quant_choices[(mydict["sdquant"] if ("sdquant" in mydict and mydict["sdquant"]>=0 and mydict["sdquant"]<len(sd_quant_choices)) else 0)])
        sd_flash_attention_var.set(1 if ("sdflashattention" in mydict and mydict["sdflashattention"]) else 0)
        sd_offload_cpu_var.set(1 if ("sdoffloadcpu" in mydict and mydict["sdoffloadcpu"]) else 0)
        sd_vae_cpu_var.set(1 if ("sdvaecpu" in mydict and mydict["sdvaecpu"]) else 0)
        sd_clip_gpu_var.set(1 if ("sdclipgpu" in mydict and mydict["sdclipgpu"]) else 0)
        sd_convdirect_var.set(sd_convdirect_option(mydict.get("sdconvdirect")))
        sd_vae_var.set(mydict["sdvae"] if ("sdvae" in mydict and mydict["sdvae"]) else "")
        sd_t5xxl_var.set(mydict["sdt5xxl"] if ("sdt5xxl" in mydict and mydict["sdt5xxl"]) else "")
        sd_clip1_var.set(mydict["sdclip1"] if ("sdclip1" in mydict and mydict["sdclip1"]) else "")
        sd_clip2_var.set(mydict["sdclip2"] if ("sdclip2" in mydict and mydict["sdclip2"]) else "")
        sd_photomaker_var.set(mydict["sdphotomaker"] if ("sdphotomaker" in mydict and mydict["sdphotomaker"]) else "")
        sd_upscaler_var.set(mydict["sdupscaler"] if ("sdupscaler" in mydict and mydict["sdupscaler"]) else "")
        sd_vaeauto_var.set(1 if ("sdvaeauto" in mydict and mydict["sdvaeauto"]) else 0)
        sd_tiled_vae_var.set(str(mydict["sdtiledvae"]) if ("sdtiledvae" in mydict and mydict["sdtiledvae"]) else str(state.default_vae_tile_threshold))
        sdl_sanitized = sanitize_lora_list(mydict.get('sdlora'))
        sd_lora_var.set("|".join(sdl_sanitized))
        sd_loramult_var.set(" ".join(f"{n:.3f}".rstrip('0').rstrip('.') for n in mydict.get("sdloramult", [])))
        if "sdmaingpu" in mydict:
            sd_main_gpu_var.set(mydict["sdmaingpu"])
        else:
            sd_main_gpu_var.set("-1")
        if sdl_sanitized and len(sdl_sanitized)==1 and os.path.isdir(sdl_sanitized[0]):
            sd_runtime_loras_var.set(1)
        else:
            sd_runtime_loras_var.set(0)

        gendefaults = (mydict["gendefaults"] if ("gendefaults" in mydict and mydict["gendefaults"]) else "")
        if isinstance(gendefaults, type({})):
            gendefaults = json.dumps(gendefaults)
        gen_defaults_var.set(gendefaults)
        gen_defaults_overwrite_var.set(1 if "gendefaultsoverwrite" in mydict and mydict["gendefaultsoverwrite"] else 0)

        whisper_model_var.set(mydict["whispermodel"] if ("whispermodel" in mydict and mydict["whispermodel"]) else "")

        tts_threads_var.set(str(mydict["ttsthreads"]) if ("ttsthreads" in mydict and mydict["ttsthreads"]) else str(default_threads))
        tts_model_var.set(mydict["ttsmodel"] if ("ttsmodel" in mydict and mydict["ttsmodel"]) else "")
        wavtokenizer_var.set(mydict["ttswavtokenizer"] if ("ttswavtokenizer" in mydict and mydict["ttswavtokenizer"]) else "")
        ttsgpu_var.set(mydict["ttsgpu"] if ("ttsgpu" in mydict) else 0)
        ttsmaxlen_var.set(str(mydict["ttsmaxlen"]) if ("ttsmaxlen" in mydict and mydict["ttsmaxlen"]) else str(state.default_ttsmaxlen))
        tts_dir_var.set(mydict["ttsdir"] if ("ttsdir" in mydict and mydict["ttsdir"]) else "")

        musicllm_var.set(mydict["musicllm"] if ("musicllm" in mydict and mydict["musicllm"]) else "")
        musicembeddings_var.set(mydict["musicembeddings"] if ("musicembeddings" in mydict and mydict["musicembeddings"]) else "")
        musicdiffusion_var.set(mydict["musicdiffusion"] if ("musicdiffusion" in mydict and mydict["musicdiffusion"]) else "")
        musicvae_var.set(mydict["musicvae"] if ("musicvae" in mydict and mydict["musicvae"]) else "")
        musiclowvram_var.set(mydict["musiclowvram"] if ("musiclowvram" in mydict) else 0)

        embeddings_model_var.set(mydict["embeddingsmodel"] if ("embeddingsmodel" in mydict and mydict["embeddingsmodel"]) else "")
        embeddings_ctx_var.set(str(mydict["embeddingsmaxctx"]) if ("embeddingsmaxctx" in mydict and mydict["embeddingsmaxctx"]) else "")
        embeddings_gpu_var.set(mydict["embeddingsgpu"] if ("embeddingsgpu" in mydict) else 0)

        admin_var.set(mydict["admin"] if ("admin" in mydict) else 0)
        router_mode_var.set(mydict["routermode"] if ("routermode" in mydict) else 0)
        autoswap_mode_var.set(mydict["autoswapmode"] if ("autoswapmode" in mydict) else 0)
        admin_dir_var.set(mydict["admindir"] if ("admindir" in mydict and mydict["admindir"]) else "")
        baseconfig_var.set(mydict["baseconfig"] if ("baseconfig" in mydict and mydict["baseconfig"]) else "")
        admin_password_var.set(mydict["adminpassword"] if ("adminpassword" in mydict and mydict["adminpassword"]) else "")
        admin_unload_timeout_var.set(mydict["adminunloadtimeout"] if ("adminunloadtimeout" in mydict and mydict["adminunloadtimeout"]) else 0)
        singleinstance_var.set(mydict["singleinstance"] if ("singleinstance" in mydict) else 0)

        importvars_in_progress = False
        gui_changed_modelfile()
        if "istemplate" in mydict and mydict["istemplate"]:
            auto_set_backend_gui(True)

    def save_config_gui():
        nonlocal kcpp_exporting_template
        kcpp_exporting_template = False
        export_vars()
        savdict = json.loads(json.dumps(state.args.__dict__,indent=2))
        for key in state.deprecated_keys:
            savdict.pop(key, None)  # avoids KeyError if missing
        savdict["istemplate"] = False
        file_type = [("KoboldCpp Settings", "*.kcpps")]
        filename = zentk_asksaveasfilename(filetypes=file_type, defaultextension=".kcpps",title="Save kcpps settings config file")
        if not filename:
            return
        save_config_dict(filename, savdict, False)
        pass

    def load_config_gui(): #this is used to populate the GUI with a config file, whereas load_config_cli simply overwrites cli args
        file_type = [("KoboldCpp Settings", "*.kcpps *.kcppt")]
        filename = zentk_askopenfilename(filetypes=file_type, defaultextension=".kcppt", initialdir=None, title="Select kcpps or kcppt settings config file")
        if not filename or filename=="":
            return
        if not os.path.exists(filename) or os.path.getsize(filename)<4 or os.path.getsize(filename)>50000000: #for sanity, check invaid kcpps
            print("The selected config file seems to be invalid.")
            if state.zenity_permitted:
                print("You can try using the legacy filepicker instead (in Extra).")
            return
        runmode_untouched = False
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            dict = json.load(f)
            import_vars(dict)
        pass

    def display_help():
        popup = ctk.CTkToplevel(root)
        popup.title("Help Menu")
        popup.geometry("380x440")
        templatedchoice_var = ctk.StringVar(value="")
        templatecatbox_var = ctk.StringVar(value="Newbie Templates")
        POPULAR_TEMPLATE_LBL = "Popular Templates"
        NEWB_TEMPLATE_LBL = "Newbie Templates"
        POPULAR_TEMPLATE_REPO = "popular-templates"
        NEWB_TEMPLATE_REPO = "newbie-templates"
        newbdesc1 = newbdesc2 = None
        def display_hf():
            popup.destroy()
            model_searcher()
        def fetch_easy_templates(a,b,c):
            nonlocal templatechoicebox, templatecatbox, newbdesc1, newbdesc2
            noobmodels = []
            resp = None
            if templatecatbox_var.get() == POPULAR_TEMPLATE_LBL:
                resp = make_url_request(f"https://huggingface.co/api/models/koboldcpp/{POPULAR_TEMPLATE_REPO}", None, 'GET', {}, 10)
                newbdesc1.pack_forget()
                newbdesc2.pack_forget()
                commdesc.pack(pady=(10, 0))
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt"):
                        noobmodels.append(entry[:-6])
            else:
                resp = make_url_request(f"https://huggingface.co/api/models/koboldcpp/{NEWB_TEMPLATE_REPO}", None, 'GET', {}, 10)
                newbdesc1.pack(pady=(10, 0))
                newbdesc2.pack(pady=(10, 0))
                commdesc.pack_forget()
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt") and "LowSpec" in entry:
                        noobmodels.append(entry[:-6])
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt") and "MidSpec" in entry:
                        noobmodels.append(entry[:-6])
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt") and "HighSpec" in entry:
                        noobmodels.append(entry[:-6])
            templatechoicebox.configure(values=noobmodels)
            if len(noobmodels)>0:
                templatedchoice_var.set(noobmodels[0])
        def load_easy_template():
            nonlocal templatechoicebox, templatecatbox, newbdesc1, newbdesc2
            cat = templatecatbox.get()
            repo = (POPULAR_TEMPLATE_REPO if cat==POPULAR_TEMPLATE_LBL else NEWB_TEMPLATE_REPO)
            fname = f"https://huggingface.co/koboldcpp/{repo}/resolve/main/{templatedchoice_var.get()}.kcppt"
            data = make_url_request(fname,data=None,method="GET")
            if data is not None:
                import_vars(data)
            popup.destroy()
        ctk.CTkLabel(popup, text="Helpful Newbie Resources").pack(pady=(5, 0))
        ctk.CTkButton(popup, text="Read the Wiki", command=display_wiki).pack(pady=5)
        ctk.CTkButton(popup, text="Read Starter Guides", command=display_starter_guides).pack(pady=5)
        ctk.CTkButton(popup, text="Search Model on Hugginface", command=display_hf).pack(pady=5)
        ctk.CTkLabel(popup, text="Or, Pick an Easy Template for Newbies").pack(pady=(12, 0))
        templatecatbox = ctk.CTkComboBox(popup, values=[NEWB_TEMPLATE_LBL,POPULAR_TEMPLATE_LBL], width=280, variable=templatecatbox_var, state="readonly")
        templatecatbox.pack(pady=5)
        templatechoicebox = ctk.CTkComboBox(popup, values=[], width=280, variable=templatedchoice_var, state="readonly")
        templatechoicebox.pack(pady=5)
        ctk.CTkButton(popup, text="Load Template", command=load_easy_template).pack(pady=5)
        newbdesc1 = ctk.CTkLabel(popup, text="LowSpec = Recommend 6GB VRAM\nMidSpec = Recommend 12GB VRAM\nHighSpec = Recommend 24GB VRAM")
        newbdesc1.pack(pady=(10, 0))
        newbdesc2 = ctk.CTkLabel(popup, text="Everything = All Features         Text = Text Generation\nImages = Image Generation         Vision = Image Recognition\nVoice = Speech Generation         Audio = Speech Recognition")
        newbdesc2.pack(pady=(10, 0))
        commdesc = ctk.CTkLabel(popup, text="Templates here are subject to change from time to time.\n\nFound a broken template? Want to contribute one?\nVisit https://huggingface.co/koboldcpp/popular-templates/")
        commdesc.pack_forget()
        templatecatbox_var.trace_add("write", fetch_easy_templates)
        fetch_easy_templates(1,1,1)
        popup.transient(root)

    def display_help_models():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/wiki#what-models-does-koboldcpp-support-what-architectures-are-supported","Cannot launch help in browser.")
    def display_starter_guides():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/wiki#step-by-step-guides","Cannot launch help in browser.")
    def display_wiki():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/wiki#the-koboldcpp-faq-and-knowledgebase","Cannot launch help in browser.")
    def display_updates():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/releases/latest","Cannot launch updates in browser.")

    ctk.CTkButton(tabs , text = "Launch", fg_color="#2f8d3c", hover_color="#2faa3c", command = guilaunch, width=100, height = 35 ).grid(row=1,column=1, stick="se", padx=(25), pady=5)

    ctk.CTkButton(tabs , text = "Update", fg_color="#9900cc", hover_color="#aa11dd", command = display_updates, width=90, height = 35 ).grid(row=1,column=0, stick="sw", padx= 5, pady=5)
    ctk.CTkButton(tabs , text = "Save Config", fg_color="#084a66", hover_color="#085a88", command = save_config_gui, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= 5, pady=5)
    ctk.CTkButton(tabs , text = "Load Config", fg_color="#084a66", hover_color="#085a88", command = load_config_gui, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= (92), pady=5)
    ctk.CTkButton(tabs , text = "Get Help", fg_color="#992222", hover_color="#bb3333", command = display_help, width=70, height = 35 ).grid(row=1,column=1, stick="sw", padx= (180), pady=5)

    # start a thread that tries to get actual gpu names and layer counts
    gpuinfo_thread = threading.Thread(target=auto_set_backend_gui)
    gpuinfo_thread.start() #submit job in new thread so nothing is waiting

    if state.args.showgui:
        if isinstance(state.args, argparse.Namespace):
            mydict = vars(state.args)
            import_vars(mydict)

    # runs main loop until closed or launch clicked
    try:
        root.mainloop()
    except (KeyboardInterrupt,SystemExit):
        exitcounter = 999
        print("Exiting by user request.")
        sys.exit(0)


    if nextstate==0:
        exitcounter = 999
        print("Exiting by user request.")
        sys.exit(0)
    else:
        # processing vars
        kcpp_exporting_template = False
        export_vars()

        if not has_valid_model():
            exitcounter = 999
            print("")
            time.sleep(0.5)
            print("Error: No valid model files were selected. Cannot continue.", flush=True)
            time.sleep(2)
            sys.exit(2)

def show_gui_msgbox(title,message):
    print(title + ": " + message, flush=True)
    try:
        from tkinter import messagebox
        import tkinter as tk
        root2 = tk.Tk()
        root2.attributes("-alpha", 0)
        messagebox.showerror(title=title, message=message)
        root2.withdraw()
        root2.destroy()
    except Exception:
        pass

def show_gui_yesnobox(title,message,icon='error'):
    print(title + ": " + message, flush=True)
    try:
        from tkinter import messagebox
        import tkinter as tk
        root2 = tk.Tk()
        root2.attributes("-alpha", 0)
        result = messagebox.askquestion(title=title, message=message,icon=icon)
        root2.withdraw()
        root2.destroy()
        return result
    except Exception:
        return False
        pass

