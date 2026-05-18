import sys
import json
import argparse

from . import state
from . import hardware
from . import sd as sd_module
from .sd import sanitize_lora_list, sanitize_lora_multipliers, parse_json_object


splitmode_choices = ['layer', 'row', 'tensor']

def splitmode_choices_to_int(value):  # layer=1, row=2, tensor=3
    if value == 'layer':
        return 1
    elif value == 'row':
        return 2
    elif value == 'tensor':
        return 3
    return 1


def convert_invalid_args(args):
    dict = args
    if isinstance(args, argparse.Namespace):
        dict = vars(args)
    if "usecuda" not in dict and "usecublas" in dict and dict["usecublas"]:
        dict["usecuda"] = dict["usecublas"]
    if "usecuda" in dict and dict["usecuda"] and "lowvram" in dict["usecuda"]:
        dict["lowvram"] = True
    if "usecuda" in dict and dict["usecuda"] and "nommq" in dict["usecuda"]:
        dict["nommq"] = True
    if "usecuda" in dict and dict["usecuda"] and "rowsplit" in dict["usecuda"]:
        dict["splitmode"] = "row"
    if "batchsize" not in dict and "blasbatchsize" in dict and dict["blasbatchsize"]:
        dict["batchsize"] = dict["blasbatchsize"]
    if "sdconfig" in dict and dict["sdconfig"] and len(dict["sdconfig"]) > 0:
        dict["sdmodel"] = dict["sdconfig"][0]
        if dict["sdconfig"] and len(dict["sdconfig"]) > 1:
            dict["sdclamped"] = 512
        if dict["sdconfig"] and len(dict["sdconfig"]) > 2:
            dict["sdthreads"] = int(dict["sdconfig"][2])
        if dict["sdconfig"] and len(dict["sdconfig"]) > 3:
            dict["sdquant"] = (2 if dict["sdconfig"][3] == "quant" else 0)
    if "hordeconfig" in dict and dict["hordeconfig"] and dict["hordeconfig"][0] != "":
        dict["hordemodelname"] = dict["hordeconfig"][0]
        if len(dict["hordeconfig"]) > 1:
            dict["hordegenlen"] = int(dict["hordeconfig"][1])
        if len(dict["hordeconfig"]) > 2:
            dict["hordemaxctx"] = int(dict["hordeconfig"][2])
        if len(dict["hordeconfig"]) > 4:
            dict["hordekey"] = dict["hordeconfig"][3]
            dict["hordeworkername"] = dict["hordeconfig"][4]
    if "noblas" in dict and dict["noblas"]:
        dict["usecpu"] = True
    if "failsafe" in dict and dict["failsafe"]:  # failsafe implies noavx2
        dict["noavx2"] = True
    if "skiplauncher" in dict and dict["skiplauncher"]:
        dict["showgui"] = False
    if "useswa" in dict and dict["useswa"]:
        dict["noshift"] = True
    if ("model_param" not in dict or not dict["model_param"]) and ("model" in dict):
        model_value = dict["model"]  # may be null, empty/non-empty string, empty/non empty array
        if isinstance(model_value, str) and model_value:  # Non-empty string
            dict["model_param"] = model_value
        elif isinstance(model_value, list) and model_value:  # Non-empty list
            dict["model_param"] = model_value[0]  # Take the first file in the list
    if ("port_param" in dict and dict["port_param"] and dict["port_param"] != state.defaultport):
        dict["port"] = dict["port_param"]
    if "sdnotile" in dict and "sdtiledvae" not in dict:
        dict["sdtiledvae"] = (0 if (dict["sdnotile"]) else state.default_vae_tile_threshold)  # convert legacy option
    if 'sdquant' in dict and type(dict['sdquant']) is bool:
        dict['sdquant'] = 2 if dict['sdquant'] else 0
    if "sdclipl" in dict and "sdclip1" not in dict:
        dict["sdclip1"] = dict["sdclipl"]
    if "sdclipg" in dict and "sdclip2" not in dict:
        dict["sdclip2"] = dict["sdclipg"]
    if "jinja_tools" in dict and dict["jinja_tools"]:
        dict["jinja"] = True
    if "jinja_kwargs" in dict and dict["jinja_kwargs"]:
        dict["jinja"] = True
    if "sdgendefaults" in dict and "gendefaults" not in dict:
        dict["gendefaults"] = dict["sdgendefaults"]
    if "flashattention" in dict and "noflashattention" not in dict:
        dict["noflashattention"] = not dict["flashattention"]
    if "sdlora" in dict:
        dict["sdlora"] = sanitize_lora_list(dict["sdlora"])
    if "sdloramult" in dict:
        dict["sdloramult"] = sanitize_lora_multipliers(dict["sdloramult"])
    return args


def reload_from_new_args(newargs):
    try:
        state.args.istemplate = False
        newargs = convert_invalid_args(newargs)
        for key, value in newargs.items():  # do not overwrite certain values
            if key not in ["remotetunnel", "showgui", "port", "host", "port_param", "admin", "adminpassword", "password", "adminunloadtimeout", "routermode", "admindir", "ssl", "nocertify", "benchmark", "prompt", "config", "baseconfig", "downloaddir"]:
                setattr(state.args, key, value)
        setattr(state.args, "showgui", False)
        setattr(state.args, "benchmark", False)
        setattr(state.args, "prompt", "")
        setattr(state.args, "config", None)
        setattr(state.args, "launch", None)
        if "istemplate" in newargs and newargs["istemplate"]:
            hardware.auto_set_backend_cli()
    except Exception as e:
        print(f"Reload New Config Failed: {e}")


def reload_new_config(filename, defaultargs, overwrite_blank=False):  # for changing config after launch
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            config = json.load(f)
            for key, value in defaultargs.items():   # Fill missing defaults directly into config
                if key not in config:
                    config[key] = value
                elif overwrite_blank and key in config and config[key] in (None, ""):
                    config[key] = value
            reload_from_new_args(config)
        except Exception as e:
            print(f"Reload New Config Failed: {e}")


def load_config_cli(filename):
    print(f"Loading configuration file {filename}...")
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        config = json.load(f)
        config = convert_invalid_args(config)
        if "onready" in config:
            config["onready"] = ""  # do not allow onready commands from config
        state.args.istemplate = False
        raw_args = (sys.argv[1:])  # a lousy hack to allow for overriding kcpps
        # special: overriding model applies to model_param too
        if "--model" in raw_args:
            raw_args.append("--model_param")
        for key, value in config.items():
            if f"--{key}" in raw_args:
                if key != "config":
                    print(f"Overriding Config Value: {key}")
            else:
                setattr(state.args, key, value)
        if state.args.istemplate:
            print("\nA .kcppt template was selected from CLI...")
            if (state.args.usecuda is None) and (state.args.usevulkan is None):
                print("Automatically selecting your backend...")
                hardware.auto_set_backend_cli()


def convert_args_to_template(savdict):
    savdict["istemplate"] = True
    savdict["gpulayers"] = -1
    savdict["threads"] = -1
    savdict["hordekey"] = ""
    savdict["hordeworkername"] = ""
    savdict["sdthreads"] = 0
    savdict["maingpu"] = -1
    savdict["sdmaingpu"] = -1
    savdict["password"] = None
    savdict["adminpassword"] = None
    savdict["usemmap"] = False
    savdict["usemlock"] = False
    savdict["debugmode"] = 0
    savdict["ssl"] = None
    savdict["usecuda"] = None
    savdict["usevulkan"] = None
    savdict["usecpu"] = None
    savdict["tensor_split"] = None
    savdict["draftgpusplit"] = None
    savdict["config"] = None
    savdict["ttsthreads"] = 0
    savdict["nommq"] = False
    savdict["splitmode"] = splitmode_choices[0]
    return savdict


def save_config_dict(filename, savdict, template):
    filenamestr = str(filename).strip()
    if not filenamestr.endswith(".kcpps") and not template:
        filenamestr += ".kcpps"
    if not filenamestr.endswith(".kcppt") and template:
        filenamestr += ".kcppt"
    do_not_save = {'analyze', 'config', 'exportconfig', 'exporttemplate', 'testmemory', 'unpack', 'version'}
    filtered = {k: v for k, v in savdict.items() if k not in do_not_save}
    if 'gendefaults' in filtered:
        gendefaults = parse_json_object(filtered['gendefaults'], 'gendefaults')
        if isinstance(gendefaults, dict):
            filtered['gendefaults'] = gendefaults
        # keep it as-is if it's a broken string
    with open(filenamestr, 'w') as file:
        file.write(json.dumps(filtered, indent=2))
    return filenamestr


def save_config_cli(filename, template):
    savdict = json.loads(json.dumps(state.args.__dict__))
    if template:
        savdict = convert_args_to_template(savdict)
    else:
        savdict["istemplate"] = False
    if filename is None:
        return
    filenamestr = save_config_dict(filename, savdict, template)
    print(f"\nSaved configuration file as {filenamestr}\nIt can be loaded with --config [filename] in future.")
    pass
