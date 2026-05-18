import ctypes
import json
import re
import random
import argparse
from datetime import datetime

from . import state
from .structs import (
    sd_load_model_inputs, sd_generation_inputs, sd_generation_outputs,
    sd_upscale_inputs, sd_info_outputs,
)
from .backend import set_backend_props
from .utils import tryparseint, tryparsefloat, strip_base64_prefix


def sd_get_info():
    info = state.handle.sd_get_info()
    if info.status == 0:
        try:
            return json.loads(info.data)
        except Exception:
            print("An error occurred while decoding sd metadata info")
    else:
        print("An error occurred while getting sd metadata info")
    return {}

sampler_aliases = [
    # sd.cpp name, UI name, aliases
    ['euler',         'Euler',    'k_euler'],
    ['euler_a',       'Euler A',  'k_euler_a', 'euler a'],
    ['heun',          'Heun',     'k_heun'],
    ['dpm2',          'DPM2',     'k_dpm_2'],
    ['lcm',           'LCM',      'k_lcm'],
    ['dpm++2m',       'DPM++ 2M', 'k_dpmpp_2m', 'dpm++ 2m karras', 'dpm++ 2m'],
    ['ddim_trailing', 'DDIM',     'ddim'],
    ['res_multistep', 'Res Multistep', 'k_res_multistep', 'res multistep'],
    ['res_2s',        'Res 2s',        'k_res_2s', 'res 2s'],
]

def sd_sampler_canonical_name(name):
    available = state.cached_sd_info.get('available_samplers', [])
    alias_map = {}
    for aliases in sampler_aliases:
        for alias in aliases:
            alias_map[alias] = aliases[0]
            alias_map[alias.lower()] = aliases[0]
    cname = alias_map.get(name.lower(), name)
    if cname in available:
        return cname
    return 'default'

def sd_sdapi_samplers():
    result = []
    available = set(state.cached_sd_info.get('available_samplers', []))
    # ensure we only advertise supported samplers
    smap = {}
    for aliases in sampler_aliases:
        if aliases[0] in available:
            smap[aliases[1]] = aliases[0:1] + aliases[2:]
            available.remove(aliases[0])
    for sampler in available:
        if sampler not in smap:
            smap[sampler] = []
    result = [{'name': k, 'aliases': v, 'options':{}}
                  for k, v in smap.items()]
    return result


sd_convdirect_choices = ['off', 'vaeonly', 'full']

def sd_convdirect_option(value):
    if not value:
        value = ''
    value = value.lower()
    if value in ['disabled', 'disable', 'none', 'off', '0', '']:
        return 'off'
    elif value in ['vae', 'vaeonly']:
        return 'vaeonly'
    elif value in ['enabled', 'enable', 'on', 'full']:
        return 'full'
    raise argparse.ArgumentTypeError(f"Invalid sdconvdirect option \"{value}\". Must be one of {sd_convdirect_choices}.")

sd_quant_choices = ['off','q8','q4']

def sd_quant_option(value):
    try:
        lvl = sd_quant_choices.index(value)
        return lvl
    except Exception:
        return 0

def sd_load_model(model_filename,vae_filename,t5xxl_filename,clip1_filename,clip2_filename,photomaker_filename,upscaler_filename):
    inputs = sd_load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    thds = state.args.threads

    if state.args.sdthreads and state.args.sdthreads > 0:
        sdt = int(state.args.sdthreads)
        if sdt > 0:
            thds = sdt

    inputs.threads = thds
    inputs.quant = state.args.sdquant
    inputs.flash_attention = state.args.sdflashattention
    inputs.offload_cpu = state.args.sdoffloadcpu
    inputs.use_mmap = state.args.usemmap
    inputs.vae_cpu = state.args.sdvaecpu
    inputs.clip_cpu = False if state.args.sdclipgpu else True
    sdconvdirect = sd_convdirect_option(state.args.sdconvdirect)
    inputs.diffusion_conv_direct = sdconvdirect == 'full'
    inputs.vae_conv_direct = sdconvdirect in ['vaeonly', 'full']
    inputs.taesd = True if state.args.sdvaeauto else False
    inputs.tiled_vae_threshold = state.args.sdtiledvae
    inputs.vae_filename = vae_filename.encode("UTF-8")
    inputs.t5xxl_filename = t5xxl_filename.encode("UTF-8")
    inputs.clip1_filename = clip1_filename.encode("UTF-8")
    inputs.clip2_filename = clip2_filename.encode("UTF-8")
    inputs.photomaker_filename = photomaker_filename.encode("UTF-8")
    inputs.upscaler_filename = upscaler_filename.encode("UTF-8")

    lora_filenames, lora_multipliers = prepare_initial_lora_multipliers()
    inputs.lora_len = len(lora_filenames)
    inputs.lora_filenames = (ctypes.c_char_p * inputs.lora_len)(*lora_filenames)
    inputs.lora_multipliers = (ctypes.c_float * inputs.lora_len)(*lora_multipliers)
    # auto if no zero-weight lora, dynamic otherwise
    lora_apply_mode = 0 # auto
    if state.imglora_bypath:
        lora_dynamic = 1 << 3 # accept changes at runtime
        lora_cache   = 1 << 4 if state.imglora_cached else 0 # cache the preloaded LoRAs
        lora_apply_mode = lora_dynamic | lora_cache
    inputs.lora_apply_mode = lora_apply_mode

    inputs.img_hard_limit = state.args.sdclamped
    inputs.img_soft_limit = state.args.sdclampedsoft
    inputs = set_backend_props(inputs)
    inputs.kcpp_main_gpu = state.args.sdmaingpu
    ret = state.handle.sd_load_model(inputs)
    return ret

def sd_oai_transform_params(genparams):
    size = genparams.get('size') or ''
    pattern = r'^\D*(\d+)x(\d+)$'
    match = re.fullmatch(pattern, size)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        genparams["width"] = width
        genparams["height"] = height
    return genparams

def sd_comfyui_tranform_params(genparams):
    promptobj = genparams.get('prompt', None)
    if promptobj and isinstance(promptobj, dict):
        for node_id, node_data in promptobj.items():
            class_type = node_data.get("class_type","")
            if class_type == "KSampler" or class_type == "KSamplerAdvanced":
                inp = node_data.get("inputs",{})

                # sampler settings from this node
                genparams["seed"] = inp.get("seed", -1)
                genparams["steps"] = inp.get("steps", 20)
                genparams["cfg_scale"] = inp.get("cfg", 5)
                genparams["sampler_name"] = inp.get("sampler_name", "euler")

                pos = inp.get("positive",[]) #positive prompt node
                neg = inp.get("negative",[]) #negative prompt node
                latentimg = inp.get("latent_image",[]) #image size node

                if latentimg and isinstance(latentimg, list) and len(latentimg) > 0:
                    temp = promptobj.get(str(latentimg[0]), {}) #now, this may be a VAEEncode or EmptyLatentImage
                    nodetype = temp.get("class_type", "") #if its a VAEEncode, it will have pixels
                    temp = temp.get('inputs', {})
                    if nodetype=="VAEEncode" and state.lastuploadedcomfyimg!="": #img2img
                        genparams["init_images"] = [state.lastuploadedcomfyimg]
                    genparams["width"] = temp.get("width", 512)
                    genparams["height"] = temp.get("height", 512)
                if neg and isinstance(neg, list) and len(neg) > 0:
                    temp = promptobj.get(str(neg[0]), {})
                    temp = temp.get('inputs', {})
                    genparams["negative_prompt"] = temp.get("text", "")
                if pos and isinstance(pos, list) and len(pos) > 0:
                    temp = promptobj.get(str(pos[0]), {})
                    temp = temp.get('inputs', {})
                    genparams["prompt"] = temp.get("text", "")
                    break
        if genparams.get("prompt","")=="": #give up, set generic prompt
            genparams["prompt"] = "high quality"
    else:
        print("Warning: ComfyUI Payload Missing!")
    return genparams

# json with top-level dict
def parse_json_object(value, field):
    if not value:
        return None
    broken = False
    if isinstance(value, str):
        retry = False
        try: # Try parsing as-is
            value = json.loads(value)
            retry = False
        except json.JSONDecodeError:
            retry = True

        if retry and ":" in value: # Try wrapping in braces for loose key/value strings
            try:
                value = json.loads(f"{{{value}}}")
                retry = False
            except json.JSONDecodeError:
                retry = True

        if retry and '\\"' in value:  #try handle double escape
            try:
                tmp = json.loads(f"\"{value}\"")
                value = json.loads(tmp)
                retry = False
            except json.JSONDecodeError:
                retry = True

        if retry:
            broken = True
    if isinstance(value, dict):
        return value
    elif broken:
        if value:
            try:
                import ast
                value = ast.literal_eval(value)
                if value and isinstance(value, dict):
                    return value
            except Exception:
                pass
        print(f"Warning: couldn't parse {field} field.")
    else:
        print(f"Warning: {field} field - not a JSON object.")
    return None

def gendefaults_parse_meta_field(value):
    alias_map = {
        'cfg-scale': 'cfg_scale',
        'guidance': 'distilled_guidance',
        'sampler': 'sampler_name',
        'sampling-method': 'sampler_name',
        'timestep-shift': 'shifted_timestep',
        'flow-shift': 'flow_shift',
        'cache-mode': 'cache_mode',
        'cache-options': 'cache_options',
        # match sd.cpp flag
        'cache-option': 'cache_options',
        'cache_option': 'cache_options',
    }
    parsed = parse_json_object(value, 'gendefaults') or {}
    result = {}
    # First pass: apply aliases only if canonical key is not explicitly present
    for key, value in parsed.items():
        canonical = alias_map.get(key, key)
        if canonical not in parsed:
            result[canonical] = value
    result.update(parsed)  # Second pass: explicit keys override aliases
    return result

def sd_upscale(genparams):
    init_images = genparams.get("image", "")
    inputs = sd_upscale_inputs()
    inputs.init_images = init_images.encode("UTF-8")
    inputs.upscaling_resize = tryparseint(genparams.get("upscaling_resize", 2),2) # how many times to upscale
    ret = state.handle.sd_upscale(inputs)
    data_main = ""
    if ret.status==1:
        data_main = ret.data.decode("UTF-8","ignore")
    return data_main

def sanitize_lora_list(sdlora):
    if not sdlora:
        sdlora = []
    elif isinstance(sdlora, str):
        sdlora = [sdlora]
    elif not isinstance(sdlora, list):
        sdlora = []
    return sdlora

def sanitize_lora_multipliers(sdloramult):
    if sdloramult is None:
        sdloramult = [1.0]
    elif not isinstance(sdloramult, list):
        sdloramult = [sdloramult]
    sdloramult = [tryparsefloat(m, 0.) for m in sdloramult]
    return sdloramult

def prepare_initial_lora_multipliers():
    res_paths = []
    res_multipliers = []
    num_loras = len(state.imglora_preload)
    if num_loras > state.lora_filenames_max:
        print(f'Warning: more than {state.lora_filenames_max} preloaded LoRAs, extra ones will be ignored')
        num_loras = state.lora_filenames_max
    for info in state.imglora_preload[:num_loras]:
        res_paths.append(info['fullpath'].encode("UTF-8"))
        res_multipliers.append(info['multiplier'])
    return res_paths, res_multipliers

def prepare_lora_multipliers_backend(request_list, imglora_bypath):
    req_dedup = {}
    for r in request_list:
        if not isinstance(r, dict):
            continue
        path = r.get('path')
        multiplier = tryparsefloat(r.get('multiplier'), 0.)
        if not path or not isinstance(path, str) or not multiplier:
            continue
        info = imglora_bypath.get(path)
        if info:
            fullpath = info["fullpath"]
            req_dedup[fullpath] = req_dedup.get(fullpath, 0.) + multiplier
    res_paths = []
    res_multipliers = []
    for fullpath, multiplier in req_dedup.items():
        if multiplier != 0.0:
            res_paths.append(fullpath.encode("UTF-8"))
            res_multipliers.append(multiplier)
    # enforce lora_filenames_max
    max_requests = state.lora_filenames_max - len(state.imglora_preload)
    if len(res_paths) > max_requests:
        msg_preloaded = ""
        if len(state.imglora_preload) > 0:
            msg_preloaded = f" (including {len(state.imglora_preload)} preloaded)"
        print(f'Warning: more than {state.lora_filenames_max} requested LoRAs{msg_preloaded}, extra ones will be ignored')
        res_paths = res_paths[:max_requests]
        res_multipliers = res_multipliers[:max_requests]
    return res_paths, res_multipliers

def prepare_lora_multipliers(request_list):
    return prepare_lora_multipliers_backend(request_list, state.imglora_bypath)

def mk_sdapi_lora_list(imglora_bypath):
    return [
        {'name': info['name'], 'path': info['path']}
            for info in imglora_bypath.values()
                if not info.get('fixed')
    ]

def extract_loras_from_prompt(prompt):
    pattern = r'<lora:([^:>]+):([^>]+)>'
    lora_data = []
    matches = list(re.finditer(pattern, prompt))
    for match in matches:
        raw_path = match.group(1)
        raw_mul = match.group(2)
        try:
            mul = float(raw_mul)
        except ValueError:
            continue
        is_high_noise = False
        prefix = "|high_noise|"
        if raw_path.startswith(prefix):
            raw_path = raw_path[len(prefix):]
            is_high_noise = True
        item = {'name': raw_path, 'multiplier': mul}
        if is_high_noise:
            item["is_high_noise"] = is_high_noise
        lora_data.append(item)
        prompt = prompt.replace(match.group(0), "", 1)
    return prompt, lora_data

def lora_map_name_to_path(request_list):
    result = []
    for req in request_list:
        out = dict(req)
        name = out.pop('name')
        path = state.imglora_name2path.get(name)
        if not path:
            print(f'LoRA {name} not found')
            continue
        info = state.imglora_bypath.get(path)
        if info:
            out['path'] = info['path']
            result.append(out)
    return result

def sd_generate(genparams):
    job_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    default_adapter = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
    adapter_obj = genparams.get('adapter', default_adapter)
    forced_negprompt = adapter_obj.get("add_sd_negative_prompt", "")
    forced_posprompt = adapter_obj.get("add_sd_prompt", "")
    forced_steplimit = tryparseint(adapter_obj.get("add_sd_step_limit", genparams.get("add_sd_step_limit",80)),80)
    forced_maxcfg = tryparsefloat(adapter_obj.get("add_sd_cfg_limit", genparams.get("add_sd_cfg_limit",25)),25)
    allow_remove_limits = tryparseint(adapter_obj.get("remove_limits", genparams.get("remove_limits",0)),0)

    prompt = genparams.get("prompt", "high quality")
    negative_prompt = genparams.get("negative_prompt", "")
    if forced_negprompt!="":
        if negative_prompt!="":
            negative_prompt += ", " + forced_negprompt
        else:
            negative_prompt = forced_negprompt
    if forced_posprompt!="":
        if prompt!="":
            prompt += ", " + forced_posprompt
        else:
            prompt = forced_posprompt
    init_images_arr = genparams.get("init_images", [])
    init_images = ("" if (not init_images_arr or len(init_images_arr)==0 or not init_images_arr[0]) else init_images_arr[0])
    init_images = strip_base64_prefix(init_images)
    mask = strip_base64_prefix(genparams.get("mask", ""))
    flip_mask = genparams.get("inpainting_mask_invert", 0)
    denoising_strength = tryparsefloat(genparams.get("denoising_strength", 0.6),0.6)
    cfg_scale = tryparsefloat(genparams.get("cfg_scale", 5),5)
    distilled_guidance = tryparsefloat(genparams.get("distilled_guidance", None), None)
    shifted_timestep = tryparseint(genparams.get("shifted_timestep", None), None)
    flow_shift = tryparsefloat(genparams.get("flow_shift", None), None)
    sample_steps = tryparseint(genparams.get("steps", 20),20)
    width = tryparseint(genparams.get("width", 512),512)
    height = tryparseint(genparams.get("height", 512),512)
    seed = tryparseint(genparams.get("seed", -1),-1)
    if seed < 0:
        seed = random.randint(100000, 999999)
    sample_method = (genparams.get("sampler_name") or "default")
    scheduler = (genparams.get("scheduler") or "default").lower()
    clip_skip = tryparseint(genparams.get("clip_skip", -1),-1)
    eta = tryparsefloat(genparams.get("eta", None), None)
    vid_req_frames = tryparseint(genparams.get("frames", 1),1)
    vid_req_frames = 1 if (not vid_req_frames or vid_req_frames < 1) else vid_req_frames
    video_output_type = genparams.get("video_output_type", 0)
    cache_mode = str(genparams.get("cache_mode", ""))
    cache_options = str(genparams.get("cache_options", ""))
    extra_images_arr = genparams.get("extra_images", [])
    extra_images_arr = ([] if not extra_images_arr else extra_images_arr)
    extra_images_arr = [img for img in extra_images_arr if img not in (None, "")]
    extra_images_arr = extra_images_arr[:state.extra_images_max]
    lora_filenames, lora_multipliers = prepare_lora_multipliers(genparams.get("lora", []))

    #clean vars
    cfg_scale = (1 if cfg_scale < 1 else (forced_maxcfg if cfg_scale > forced_maxcfg else cfg_scale))
    if distilled_guidance is not None and (distilled_guidance < 0 or distilled_guidance > 100):
        distilled_guidance = None # fall back to the default
    if shifted_timestep is not None and (shifted_timestep < 0 or shifted_timestep > 1000):
        shifted_timestep = None # fall back to the default
    if flow_shift is not None and flow_shift < 0:
        flow_shift = None # fall back to the default
    sample_steps = (1 if sample_steps < 1 else (forced_steplimit if sample_steps > forced_steplimit else sample_steps))
    vid_req_frames = (1 if vid_req_frames < 1 else (100 if vid_req_frames > 100 else vid_req_frames))

    swap_refimg = (True if tryparseint(genparams.get("send_as_refimg", 0),0) else False)
    if len(extra_images_arr)==0 and swap_refimg and init_images and init_images!="" and not mask:
        extra_images_arr = [init_images]
        init_images = ""

    inputs = sd_generation_inputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.negative_prompt = negative_prompt.encode("UTF-8")
    inputs.init_images = init_images.encode("UTF-8")
    inputs.mask = "".encode("UTF-8") if not mask else mask.encode("UTF-8")
    inputs.extra_images_len = len(extra_images_arr)
    inputs.extra_images = (ctypes.c_char_p * inputs.extra_images_len)()
    for n, estr in enumerate(extra_images_arr):
        extra_image = strip_base64_prefix(estr)
        inputs.extra_images[n] = extra_image.encode("UTF-8")
    inputs.flip_mask = flip_mask
    inputs.cfg_scale = cfg_scale
    if distilled_guidance is not None:
        inputs.distilled_guidance = distilled_guidance
    inputs.denoising_strength = (0 if denoising_strength < 0 else (1 if denoising_strength > 1 else denoising_strength))
    if shifted_timestep is not None:
        inputs.shifted_timestep = shifted_timestep
    if flow_shift is not None:
        inputs.flow_shift = flow_shift
    inputs.sample_steps = sample_steps
    inputs.width = width
    inputs.height = height
    inputs.seed = ((seed + 2**31) % 2**32) - 2**31
    inputs.sample_method = sd_sampler_canonical_name(sample_method).encode("UTF-8")
    inputs.scheduler = scheduler.encode("UTF-8")
    inputs.eta = -1.0 if eta is None else eta
    inputs.clip_skip = clip_skip
    inputs.vid_req_frames = vid_req_frames
    inputs.video_output_type = video_output_type
    inputs.remove_limits = allow_remove_limits
    inputs.circular_x = tryparseint(adapter_obj.get("circular_x", genparams.get("circular_x",0)),0)
    inputs.circular_y = tryparseint(adapter_obj.get("circular_y", genparams.get("circular_y",0)),0)
    inputs.cache_mode = cache_mode.encode("UTF-8")
    inputs.cache_options = cache_options.encode("UTF-8")
    inputs.upscale = (True if tryparseint(genparams.get("enable_hr", 0),0) else False)
    inputs.lora_len = len(lora_filenames)
    inputs.lora_filenames = (ctypes.c_char_p * inputs.lora_len)(*lora_filenames)
    inputs.lora_multipliers = (ctypes.c_float * inputs.lora_len)(*lora_multipliers)

    ret = state.handle.sd_generate(inputs)
    data_main = ""
    data_extra = ""
    info = {}
    animated = False
    if ret.status==1:
        data_main = ret.data.decode("UTF-8","ignore")
        data_extra = ret.data_extra.decode("UTF-8","ignore")
        info = json.loads(ret.info.decode("UTF-8","ignore"))
        animated = True if ret.animated else False
    info["job_timestamp"] = job_timestamp
    return {"animated": animated, "data":data_main, "data_extra":data_extra, "info": info}
