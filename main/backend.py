import os
import ctypes
import platform

from . import state
from . import hardware
from .structs import (
    load_model_inputs, generation_inputs, generation_outputs,
    logit_bias, token_count_outputs, detokenize_inputs,
    sd_load_model_inputs, sd_generation_inputs, sd_generation_outputs,
    sd_upscale_inputs, sd_info_outputs,
    whisper_load_model_inputs, whisper_generation_inputs, whisper_generation_outputs,
    tts_load_model_inputs, tts_generation_inputs, tts_generation_outputs,
    embeddings_load_model_inputs, embeddings_generation_inputs, embeddings_generation_outputs,
    music_load_model_inputs, music_generation_inputs, music_generation_outputs,
    last_logprobs_outputs,
)
from .utils import tryparseint, tryparsefloat, convert_json_to_gbnf, utfprint


def init_library():
    state.libname = state.lib_default

    if not state.args: # debug helper: koboldcpp.py loaded by external script
        pass
    elif state.args.noavx2: #failsafe implies noavx2 always
        if state.args.failsafe and (state.args.usevulkan is not None) and state.file_exists(state.lib_vulkan_failsafe):
            state.libname = state.lib_vulkan_failsafe
        elif (state.args.usevulkan is not None) and state.file_exists(state.lib_vulkan_noavx2):
            state.libname = state.lib_vulkan_noavx2
        elif (state.args.failsafe) and state.file_exists(state.lib_failsafe):
            print("!!! Attempting to use FAILSAFE MODE !!!")
            state.libname = state.lib_failsafe
        elif state.file_exists(state.lib_noavx2):
            state.libname = state.lib_noavx2
    elif (state.args.usecuda is not None):
        if state.file_exists(state.lib_cublas):
            state.libname = state.lib_cublas
        elif state.file_exists(state.lib_hipblas):
            state.libname = state.lib_hipblas
    elif (state.args.usevulkan is not None):
        if state.file_exists(state.lib_vulkan):
            state.libname = state.lib_vulkan
        elif state.file_exists(state.lib_vulkan_noavx2):
            state.libname = state.lib_vulkan_noavx2
    elif state.libname == state.lib_default and not state.file_exists(state.lib_default) and state.file_exists(state.lib_noavx2):
        state.libname = state.lib_noavx2

    print("Initializing dynamic library: " + state.libname)
    dir_path = state.getdirpath()
    abs_path = state.getabspath()

    #add all potential paths
    if os.name=='nt':
        os.add_dll_directory(dir_path)
        os.add_dll_directory(abs_path)
        os.add_dll_directory(os.getcwd())
        if state.libname == state.lib_cublas and "CUDA_PATH" in os.environ:
            newpath = os.path.join(os.environ["CUDA_PATH"], "bin")
            if os.path.exists(newpath):
                os.add_dll_directory(newpath)
        if state.libname == state.lib_hipblas and "HIP_PATH" in os.environ:
            newpath = os.path.join(os.environ["HIP_PATH"], "bin")
            if os.path.exists(newpath):
                os.add_dll_directory(newpath)

    state.handle = ctypes.CDLL(os.path.join(dir_path, state.libname))

    state.handle.load_model.argtypes = [load_model_inputs]
    state.handle.load_model.restype = ctypes.c_bool
    state.handle.generate.argtypes = [generation_inputs]
    state.handle.generate.restype = generation_outputs
    state.handle.new_token.restype = ctypes.c_char_p
    state.handle.new_token.argtypes = [ctypes.c_int]
    state.handle.get_stream_count.restype = ctypes.c_int
    state.handle.has_finished.restype = ctypes.c_bool
    state.handle.batch_generate_enabled.restype = ctypes.c_bool
    state.handle.batch_generate_submit.argtypes = [generation_inputs]
    state.handle.batch_generate_submit.restype = ctypes.c_int
    state.handle.batch_generate_has_finished.argtypes = [ctypes.c_int]
    state.handle.batch_generate_has_finished.restype = ctypes.c_bool
    state.handle.batch_generate_stream_count.argtypes = [ctypes.c_int]
    state.handle.batch_generate_stream_count.restype = ctypes.c_int
    state.handle.batch_generate_new_token.argtypes = [ctypes.c_int, ctypes.c_int]
    state.handle.batch_generate_new_token.restype = ctypes.c_char_p
    state.handle.batch_generate_pending_output.argtypes = [ctypes.c_int]
    state.handle.batch_generate_pending_output.restype = ctypes.c_char_p
    state.handle.batch_generate_result.argtypes = [ctypes.c_int]
    state.handle.batch_generate_result.restype = generation_outputs
    state.handle.batch_generate_abort.argtypes = [ctypes.c_int]
    state.handle.batch_generate_abort.restype = ctypes.c_bool
    state.handle.batch_generate_release.argtypes = [ctypes.c_int]
    state.handle.batch_generate_release.restype = None
    state.handle.has_audio_support.restype = ctypes.c_bool
    state.handle.has_vision_support.restype = ctypes.c_bool
    state.handle.get_last_eval_time.restype = ctypes.c_float
    state.handle.get_last_process_time.restype = ctypes.c_float
    state.handle.get_last_token_count.restype = ctypes.c_int
    state.handle.get_last_input_count.restype = ctypes.c_int
    state.handle.get_last_seed.restype = ctypes.c_int
    state.handle.get_last_draft_success.restype = ctypes.c_int
    state.handle.get_last_draft_failed.restype = ctypes.c_int
    state.handle.get_total_img_gens.restype = ctypes.c_int
    state.handle.get_total_tts_gens.restype = ctypes.c_int
    state.handle.get_total_transcribe_gens.restype = ctypes.c_int
    state.handle.get_total_gens.restype = ctypes.c_int
    state.handle.get_last_stop_reason.restype = ctypes.c_int
    state.handle.abort_generate.restype = ctypes.c_bool
    state.handle.token_count.restype = token_count_outputs
    state.handle.get_pending_output.restype = ctypes.c_char_p
    state.handle.get_chat_template.restype = ctypes.c_char_p
    state.handle.calc_new_state_kv.restype = ctypes.c_size_t
    state.handle.calc_new_state_tokencount.restype = ctypes.c_size_t
    state.handle.calc_old_state_kv.argtypes = [ctypes.c_int]
    state.handle.calc_old_state_kv.restype = ctypes.c_size_t
    state.handle.calc_old_state_tokencount.argtypes = [ctypes.c_int]
    state.handle.calc_old_state_tokencount.restype = ctypes.c_size_t
    state.handle.save_state_kv.argtypes = [ctypes.c_int]
    state.handle.save_state_kv.restype = ctypes.c_size_t
    state.handle.load_state_kv.argtypes = [ctypes.c_int]
    state.handle.load_state_kv.restype = ctypes.c_bool
    state.handle.clear_state_kv.restype = ctypes.c_bool
    state.handle.sd_load_model.argtypes = [sd_load_model_inputs]
    state.handle.sd_load_model.restype = ctypes.c_bool
    state.handle.sd_generate.argtypes = [sd_generation_inputs]
    state.handle.sd_generate.restype = sd_generation_outputs
    state.handle.sd_upscale.argtypes = [sd_upscale_inputs]
    state.handle.sd_upscale.restype = sd_generation_outputs
    state.handle.sd_get_info.argtypes = []
    state.handle.sd_get_info.restype = sd_info_outputs
    state.handle.whisper_load_model.argtypes = [whisper_load_model_inputs]
    state.handle.whisper_load_model.restype = ctypes.c_bool
    state.handle.whisper_generate.argtypes = [whisper_generation_inputs]
    state.handle.whisper_generate.restype = whisper_generation_outputs
    state.handle.tts_load_model.argtypes = [tts_load_model_inputs]
    state.handle.tts_load_model.restype = ctypes.c_bool
    state.handle.tts_generate.argtypes = [tts_generation_inputs]
    state.handle.tts_generate.restype = tts_generation_outputs
    state.handle.embeddings_load_model.argtypes = [embeddings_load_model_inputs]
    state.handle.embeddings_load_model.restype = ctypes.c_bool
    state.handle.embeddings_generate.argtypes = [embeddings_generation_inputs]
    state.handle.embeddings_generate.restype = embeddings_generation_outputs
    state.handle.music_load_model.argtypes = [music_load_model_inputs]
    state.handle.music_load_model.restype = ctypes.c_bool
    state.handle.music_generate.argtypes = [music_generation_inputs]
    state.handle.music_generate.restype = music_generation_outputs
    state.handle.last_logprobs.restype = last_logprobs_outputs
    state.handle.detokenize.argtypes = [detokenize_inputs]
    state.handle.detokenize.restype = ctypes.c_char_p
    state.handle.set_environment_variable.restype = ctypes.c_int
    state.handle.set_environment_variable.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

def set_backend_props(inputs):
    # we must force an explicit tensor split
    # otherwise the default will divide equally and multigpu crap will slow it down badly
    inputs.kcpp_main_gpu = -1
    if(state.args.maingpu is not None and state.args.maingpu>=0):
        inputs.kcpp_main_gpu = state.args.maingpu

    if state.args.usecuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not state.args.tensor_split:
        if (state.args.usecuda and "0" in state.args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
            inputs.kcpp_main_gpu = 0
        elif (state.args.usecuda and "1" in state.args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            os.environ["HIP_VISIBLE_DEVICES"] = "1"
            inputs.kcpp_main_gpu = 0
        elif (state.args.usecuda and "2" in state.args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            os.environ["HIP_VISIBLE_DEVICES"] = "2"
            inputs.kcpp_main_gpu = 0
        elif (state.args.usecuda and "3" in state.args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            os.environ["HIP_VISIBLE_DEVICES"] = "3"
            inputs.kcpp_main_gpu = 0
    else:
        if(state.args.maingpu is None or state.args.maingpu<0):
            if (state.args.usecuda and "0" in state.args.usecuda):
                inputs.kcpp_main_gpu = 0
            elif (state.args.usecuda and "1" in state.args.usecuda):
                inputs.kcpp_main_gpu = 1
            elif (state.args.usecuda and "2" in state.args.usecuda):
                inputs.kcpp_main_gpu = 2
            elif (state.args.usecuda and "3" in state.args.usecuda):
                inputs.kcpp_main_gpu = 3

    if "GGML_VK_VISIBLE_DEVICES" not in os.environ:
        if state.args.usevulkan: # is an empty array if using vulkan without defined gpu
            vulkangpus = ','.join([str(g) for g in state.args.usevulkan])
            state.handle.set_environment_variable("GGML_VK_VISIBLE_DEVICES".encode("UTF-8"),vulkangpus.encode("UTF-8"))

    # set universal flags
    inputs.devices_override = (state.args.device if state.args.device else "").encode("UTF-8")
    inputs.quiet = state.args.quiet
    inputs.debugmode = state.args.debugmode
    inputs.executable_path = (state.getdirpath()+"/").encode("UTF-8")

    return inputs

def load_model(model_filename):
    from .gui import splitmode_choices_to_int
    inputs = load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.max_context_length = state.maxctx #initial value to use for ctx, can be overwritten
    inputs.threads = state.args.threads
    inputs.low_vram = True if state.args.lowvram else False
    inputs.use_mmq = False if state.args.nommq else True
    inputs.splitmode = splitmode_choices_to_int(state.args.splitmode) #layer=1, row=2, tensor=3
    inputs.blasthreads = state.args.blasthreads
    inputs.use_mmap = state.args.usemmap
    inputs.use_mlock = state.args.usemlock
    inputs.lora_filename = "".encode("UTF-8")
    inputs.lora_multiplier = state.args.loramult
    if state.args.lora:
        inputs.lora_filename = state.args.lora[0].encode("UTF-8")

    inputs.draftmodel_filename = state.args.draftmodel.encode("UTF-8") if state.args.draftmodel else "".encode("UTF-8")
    inputs.draft_amount = state.args.draftamount
    inputs.draft_gpulayers = state.args.draftgpulayers
    for n in range(state.tensor_split_max):
        if state.args.draftgpusplit and n < len(state.args.draftgpusplit):
            inputs.draft_gpusplit[n] = float(state.args.draftgpusplit[n])
        else:
            inputs.draft_gpusplit[n] = 0
    inputs.mmproj_filename = state.args.mmproj.encode("UTF-8") if state.args.mmproj else "".encode("UTF-8")
    inputs.mmproj_cpu = (True if state.args.mmprojcpu else False)
    inputs.visionmaxres = (512 if state.args.visionmaxres < 512 else (2048 if state.args.visionmaxres > 2048 else state.args.visionmaxres))
    vmintk = state.args.visionmintokens
    vmaxtk = state.args.visionmaxtokens
    vmintk = -1 if vmintk<-1 else vmintk
    vmaxtk = -1 if vmaxtk<-1 else vmaxtk
    if(vmintk!=-1 or vmaxtk!=-1) and (vmintk==-1 or vmaxtk==-1): #if exactly one of the args is -1
        vmintk = max(vmintk,vmaxtk)
        vmaxtk = max(vmintk,vmaxtk)
    inputs.visionmintokens = vmintk
    inputs.visionmaxtokens = vmaxtk
    inputs.use_smartcontext = state.args.smartcontext
    if getattr(state.args, "continuous_batching", 0) > 1 and not state.args.noshift:
        print("\nWarning: Continuous batching is enabled, so context shifting has been disabled automatically.\n")
        state.args.noshift = True
    inputs.use_contextshift = (0 if state.args.noshift else 1)
    inputs.use_fastforward = (0 if state.args.nofastforward else 1)
    inputs.flash_attention =  (False if state.args.noflashattention else True)
    if state.args.quantkv:
        qkvstr = str(state.args.quantkv).lower()
        qkvval = 0
        if qkvstr=="bf16" or qkvstr=="3": #migration for old index based values
            qkvval = 1
        elif qkvstr=="q8_0" or qkvstr=="1":
            qkvval = 2
        elif qkvstr=="q5_1":
            qkvval = 3
        elif qkvstr=="q4_0" or qkvstr=="2":
            qkvval = 4
        if state.args.noflashattention:
            inputs.quant_k = qkvval
            inputs.quant_v = 0 if qkvval!=1 else qkvval
            if qkvval>1:
                print("\nWarning: Quantized KV was used without flash attention! This is NOT RECOMMENDED!\nOnly K cache can be quantized, and performance can suffer.\nIn some cases, it might even use more VRAM when doing a full offload.\nYou are strongly encouraged to use flash attention if you want to use quantkv.")
        else:
            inputs.quant_k = inputs.quant_v = qkvval
    else:
        inputs.quant_k = inputs.quant_v = 0
    inputs.batchsize = state.args.batchsize
    inputs.autofit = state.args.autofit
    inputs.autofit_tax_mb = int(state.args.autofitpadding) + int(state.calulated_gpu_overhead/(1024*1024))
    inputs.gpulayers = state.args.gpulayers
    if state.args.overridenativecontext and state.args.overridenativecontext>0:
        inputs.overridenativecontext = state.args.overridenativecontext
        inputs.rope_freq_scale = 0
        inputs.rope_freq_base = 10000
    else:
        inputs.overridenativecontext = 0
        inputs.rope_freq_scale = state.args.ropeconfig[0]
        if len(state.args.ropeconfig)>1:
            inputs.rope_freq_base = state.args.ropeconfig[1]
        else:
            inputs.rope_freq_base = 10000

    for n in range(state.tensor_split_max):
        if state.args.tensor_split and n < len(state.args.tensor_split):
            inputs.tensor_split[n] = float(state.args.tensor_split[n])
        else:
            inputs.tensor_split[n] = 0

    inputs.moe_experts = state.args.moeexperts
    inputs.no_bos_token = state.args.nobostoken
    inputs.load_guidance = state.args.enableguidance
    okv = []
    if state.args.overridekv and str(state.args.overridekv).count(",")>0 and str(state.args.overridekv).count("=")>1 and str(state.args.overridekv).count(":")==str(state.args.overridekv).count("="):
        okv = [x.strip() for x in str(state.args.overridekv).split(",")]
        okv = [item for item in okv if item and item.strip()]
    elif state.args.overridekv:
        okv = [state.args.overridekv]
    for n in range(state.overridekv_max):
        if not okv or n >= len(okv):
            inputs.override_kv[n] = "".encode("UTF-8")
        else:
            inputs.override_kv[n] = okv[n].encode("UTF-8")
    inputs.override_tensors = state.args.overridetensors.encode("UTF-8") if state.args.overridetensors else "".encode("UTF-8")
    inputs.moecpu = (200 if state.args.moecpu > 200 else state.args.moecpu)
    inputs.check_slowness = (not state.args.highpriority and os.name == 'nt' and 'Intel' in platform.processor())
    inputs.jinja_template = state.preloaded_custom_jinja.encode("UTF-8")
    inputs.highpriority = state.args.highpriority
    inputs.swa_support = state.args.useswa
    inputs.swa_padding = state.args.swapadding if state.args.useswa else 0
    scint = int(state.args.smartcache)
    inputs.smartcache = False if scint<=0 else True
    sclimit = (state.savestate_limit_default if scint<=1 else scint)
    state.savestate_limit = sclimit
    inputs.smartcacheslots = sclimit
    inputs.pipelineparallel = (not state.args.nopipelineparallel)
    inputs.continuous_batching_slots = int(state.args.continuous_batching) if hasattr(state.args, "continuous_batching") else 0
    inputs = set_backend_props(inputs)
    ret = state.handle.load_model(inputs)
    return ret

def generate(genparams, stream_flag=False):
    import json
    default_adapter = {} if state.chatcompl_adapter is None else state.chatcompl_adapter
    adapter_obj = genparams.get('adapter', default_adapter)

    prompt = genparams.get('prompt', "")
    memory = genparams.get('memory', "")
    negative_prompt = genparams.get('negative_prompt', "")
    guidance_scale = tryparsefloat(genparams.get('guidance_scale', 1.0),1.0)
    images = genparams.get('images', [])
    audio = genparams.get('audio', [])
    max_context_length = tryparseint(genparams.get('max_context_length', state.maxctx),state.maxctx)
    max_length = tryparseint(genparams.get('max_length', state.args.defaultgenamt),state.args.defaultgenamt)
    temperature = tryparsefloat(genparams.get('temperature', adapter_obj.get("temperature", 0.75)),0.75)
    top_k = tryparseint(genparams.get('top_k', adapter_obj.get("top_k", 100)),100)
    top_a = tryparsefloat(genparams.get('top_a', 0.0),0.0)
    top_p = tryparsefloat(genparams.get('top_p', adapter_obj.get("top_p", 0.92)),0.92)
    min_p = tryparsefloat(genparams.get('min_p', adapter_obj.get("min_p", 0.0)),0.0)
    typical_p = tryparsefloat(genparams.get('typical', 1.0),1.0)
    tfs = tryparsefloat(genparams.get('tfs', 1.0),1.0)
    nsigma = tryparsefloat(genparams.get('nsigma', 0.0),0.0)
    rep_pen = tryparsefloat(genparams.get('rep_pen', adapter_obj.get("rep_pen", 1.0)),1.0)
    rep_pen_range = tryparseint(genparams.get('rep_pen_range', 320),320)
    rep_pen_slope = tryparsefloat(genparams.get('rep_pen_slope', 1.0),1.0)
    presence_penalty = tryparsefloat(genparams.get('presence_penalty', 0.0),0.0)
    mirostat = tryparseint(genparams.get('mirostat', 0),0)
    mirostat_tau = tryparsefloat(genparams.get('mirostat_tau', 5.0),5.0)
    mirostat_eta = tryparsefloat(genparams.get('mirostat_eta', 0.1),0.1)
    dry_multiplier = tryparsefloat(genparams.get('dry_multiplier', 0.0),0.0)
    dry_base = tryparsefloat(genparams.get('dry_base', 1.75),1.75)
    dry_allowed_length = tryparseint(genparams.get('dry_allowed_length', 2),2)
    dry_penalty_last_n = tryparseint(genparams.get('dry_penalty_last_n', 320),320)
    dry_sequence_breakers = genparams.get('dry_sequence_breakers', [])
    xtc_threshold = tryparsefloat(genparams.get('xtc_threshold', 0.2),0.2)
    xtc_probability = tryparsefloat(genparams.get('xtc_probability', 0),0)
    sampler_order = genparams.get('sampler_order', [6, 0, 1, 3, 4, 2, 5])
    seed = tryparseint(genparams.get('sampler_seed', -1),-1)
    stop_sequence = genparams.get('stop_sequence', [])
    ban_eos_token = genparams.get('ban_eos_token', False)
    stream_sse = stream_flag
    grammar = genparams.get('grammar', '')

    #translate grammar if its json
    try:
        grammarjson = json.loads(grammar)
        decoded = convert_json_to_gbnf(grammarjson)
        if decoded:
            grammar = decoded
    except Exception:
        pass
    grammar_retain_state = genparams.get('grammar_retain_state', False)
    genkey = genparams.get('genkey', '')
    trimstop = genparams.get('trim_stop', True)
    dynatemp_range = tryparsefloat(genparams.get('dynatemp_range', 0.0),0.0)
    dynatemp_exponent = tryparsefloat(genparams.get('dynatemp_exponent', 1.0),1.0)
    smoothing_factor = tryparsefloat(genparams.get('smoothing_factor', 0.0),0.0)
    smoothing_curve = tryparsefloat(genparams.get('smoothing_curve', 1.0),1.0)
    adaptive_target = tryparsefloat(genparams.get('adaptive_target', -1.0),-1.0)
    adaptive_decay = tryparsefloat(genparams.get('adaptive_decay', 0.9),0.9)
    adaptive_decay = 0.01 if adaptive_decay < 0.01 else (0.99 if adaptive_decay > 0.99 else adaptive_decay)
    if adaptive_target>0 and min_p<=0 and top_p>=1.0: #adaptive p sampler requires a truncation sampler first, force a tiny min-p
        min_p = 0.002
    logit_biases = genparams.get('logit_bias', {})
    render_special = genparams.get('render_special', False)
    banned_strings = genparams.get('banned_strings', []) # SillyTavern uses that name
    banned_tokens = genparams.get('banned_tokens', banned_strings)
    bypass_eos_token = genparams.get('bypass_eos', False)
    tool_call_fix = genparams.get('using_openai_tools', False)
    custom_token_bans = genparams.get('custom_token_bans', '')

    for tok in custom_token_bans.split(','):
        tok = tok.strip()  # Remove leading/trailing whitespace
        if tok.isdigit():
            logit_biases[tok] = state.bias_min_value

    inputs = generation_inputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.memory = memory.encode("UTF-8")
    inputs.negative_prompt = negative_prompt.encode("UTF-8")
    inputs.guidance_scale = guidance_scale

    images = images[-state.images_max:]
    inputs.images_len = len(images)
    inputs.images = (ctypes.c_char_p * inputs.images_len)()
    for n, item in enumerate(images):
        inputs.images[n] = item.encode("UTF-8")
    audio = audio[-state.audio_max:]
    inputs.audio_len = len(audio)
    inputs.audio = (ctypes.c_char_p * inputs.audio_len)()
    for n, item in enumerate(audio):
        inputs.audio[n] = item.encode("UTF-8")

    if max_context_length > state.maxctx:
        if state.showmaxctxwarning:
            print(f"\n!!! ====== !!!\n(Warning! Request max_context_length={max_context_length} exceeds allocated context size of {state.maxctx}. It will be reduced to fit. Consider launching with increased --contextsize to avoid issues. This message will only show once per session.)\n!!! ====== !!!")
            state.showmaxctxwarning = False
        max_context_length = state.maxctx
    min_remain_hardlimit = max(min(max_context_length-4, 16),int(max_context_length*0.2))
    min_remain_softlimit = max(min(max_context_length-4, 16),int(max_context_length*0.4))
    if state.args.genlimit > 0 and max_length > state.args.genlimit:
        max_length = state.args.genlimit
    if max_length >= (max_context_length-min_remain_softlimit):
        print(f"\n!!! ====== !!!\nWarning: You are trying to generate text with max_length ({max_length}) near or exceeding max_context_length limit ({max_context_length}).\nMost of the context will be removed, and your outputs will not be very coherent.\nConsider launching with increased --contextsize to avoid issues.\n!!! ====== !!!")
        if max_length >= (max_context_length-min_remain_hardlimit):
            max_length = max_context_length-min_remain_hardlimit

    reasoning_effort = genparams.get('reasoning_effort', '')
    reasoning_effort = reasoning_effort.strip().lower() if reasoning_effort else ''
    reasoning_budget = -1
    if reasoning_effort == "none":
        reasoning_budget = 0
    elif reasoning_effort == "minimal":
        reasoning_budget = tryparseint(0.1 * max_length,-1)  # 10% of gen amount
    elif reasoning_effort == "low":
        reasoning_budget = tryparseint(0.25 * max_length,-1)  # 25% of gen amount
    elif reasoning_effort == "medium":
        reasoning_budget = tryparseint(0.5 * max_length,-1)  # 50% of gen amount
    else:
        pass #unrestricted

    inputs.max_context_length = max_context_length   # this will resize the context buffer if changed
    inputs.max_length = max_length
    inputs.temperature = temperature
    inputs.top_k = top_k
    inputs.top_a = top_a
    inputs.top_p = top_p
    inputs.min_p = min_p
    inputs.typical_p = typical_p
    inputs.tfs = tfs
    inputs.nsigma = nsigma
    inputs.rep_pen = rep_pen
    inputs.rep_pen_range = rep_pen_range
    inputs.rep_pen_slope = rep_pen_slope
    inputs.presence_penalty = presence_penalty
    inputs.stream_sse = stream_sse
    inputs.dynatemp_range = dynatemp_range
    inputs.dynatemp_exponent = dynatemp_exponent
    inputs.smoothing_factor = smoothing_factor
    inputs.smoothing_curve = smoothing_curve
    inputs.adaptive_target = adaptive_target
    inputs.adaptive_decay = adaptive_decay
    inputs.grammar = grammar.encode("UTF-8")
    inputs.grammar_retain_state = grammar_retain_state
    inputs.allow_eos_token = not ban_eos_token
    inputs.bypass_eos_token = bypass_eos_token
    inputs.tool_call_fix = tool_call_fix
    inputs.render_special = render_special
    if mirostat in (1, 2):
        inputs.mirostat = mirostat
        inputs.mirostat_tau = mirostat_tau
        inputs.mirostat_eta = mirostat_eta
    else:
        inputs.mirostat = inputs.mirostat_tau = inputs.mirostat_eta = 0
    inputs.dry_multiplier = dry_multiplier
    inputs.dry_base = dry_base
    inputs.xtc_threshold = xtc_threshold
    inputs.xtc_probability = xtc_probability
    inputs.dry_allowed_length = dry_allowed_length
    inputs.dry_penalty_last_n = dry_penalty_last_n
    # Handle dry_sequence_breakers being passed as a json-encoded array of
    # strings, rather than as an array of strings itself. This is to support
    # SillyTavern, which passes sequence breakers to Oobabooga that way.
    if dry_multiplier > 0 and isinstance(dry_sequence_breakers, str):
        try:
            dry_sequence_breakers = json.loads(dry_sequence_breakers)
        except ValueError as e:
            print(f"ERROR: dry_sequence_breakers must be an array of strings or a json encoded array of strings. Could not parse '{dry_sequence_breakers}': " + str(e))
            dry_sequence_breakers = []

    if dry_multiplier <= 0 or dry_sequence_breakers is None: # prevent explicitly set to None, retain old behavior
        dry_sequence_breakers = []

    dry_sequence_breakers = dry_sequence_breakers[:state.dry_seq_break_max]
    inputs.dry_sequence_breakers_len = len(dry_sequence_breakers)
    inputs.dry_sequence_breakers = (ctypes.c_char_p * inputs.dry_sequence_breakers_len)()

    for n, breaker in enumerate(dry_sequence_breakers):
        inputs.dry_sequence_breakers[n] = breaker.encode("UTF-8")

    if sampler_order and 0 < len(sampler_order) <= state.sampler_order_max:
        try:
            for i, sampler in enumerate(sampler_order):
                inputs.sampler_order[i] = sampler
            inputs.sampler_len = len(sampler_order)
            if state.showsamplerwarning and inputs.mirostat==0 and inputs.sampler_len>0 and (inputs.sampler_order[0]!=6 or inputs.sampler_order[inputs.sampler_len-1]!=5):
                print("\n(Note: Non-default sampler_order detected. Recommended sampler values are [6,0,1,3,4,2,5]. This message will only show once per session.)")
                state.showsamplerwarning = False
        except TypeError as e:
            print("ERROR: sampler_order must be a list of integers: " + str(e))
    inputs.seed = seed

    inputs.stop_sequence_len = len(stop_sequence)
    inputs.stop_sequence = (ctypes.c_char_p * inputs.stop_sequence_len)()

    for n, sequence in enumerate(stop_sequence):
        if sequence:
            inputs.stop_sequence[n] = sequence.encode("UTF-8")
        else:
            inputs.stop_sequence[n] = "".encode("UTF-8")

    bias_list = []
    try:
        if logit_biases and len(logit_biases) > 0:
            bias_list = [{"key": key, "value": value} for key, value in logit_biases.items()]
    except Exception as ex:
        print(f"Logit bias dictionary is invalid: {ex}")

    bias_list = bias_list[:state.logit_bias_max]
    inputs.logit_biases_len = len(bias_list)
    inputs.logit_biases = (logit_bias * inputs.logit_biases_len)()
    for n, lb in enumerate(bias_list):
        try:
            t_id = int(lb['key'])
            bias = float(lb['value'])
            t_id = -1 if t_id < 0 else t_id
            bias = (state.bias_max_value if bias > state.bias_max_value else (state.bias_min_value if bias < state.bias_min_value else bias))
            inputs.logit_biases[n] = logit_bias(t_id, bias)
        except Exception as ex:
            inputs.logit_biases[n] = logit_bias(-1, 0.0)
            print(f"Skipped unparsable logit bias:{ex}")

    if banned_tokens is None:
        banned_tokens = []
    banned_tokens = banned_tokens[:state.ban_token_max]
    inputs.banned_tokens_len = len(banned_tokens)
    inputs.banned_tokens = (ctypes.c_char_p * inputs.banned_tokens_len)()
    for n, tok in enumerate(banned_tokens):
        inputs.banned_tokens[n] = tok.encode("UTF-8")

    inputs.reasoning_budget = reasoning_budget

    state.currentusergenkey = genkey
    state.totalgens += 1
    #early exit if aborted

    if state.pendingabortkey!="" and state.pendingabortkey==genkey:
        print(f"\nDeferred Abort for GenKey: {state.pendingabortkey}")
        state.pendingabortkey = ""
        return {"text":"","status":-1,"stopreason":-1, "prompt_tokens":0, "completion_tokens": 0, "total_tokens": 0}
    else:
        batch_request_id = -1
        if getattr(state.args, "continuous_batching", 0) > 1:
            try:
                batch_request_id = state.handle.batch_generate_submit(inputs)
            except Exception:
                batch_request_id = -1
        if batch_request_id >= 0:
            genparams['_batch_request_id'] = batch_request_id
            ret = state.handle.batch_generate_result(batch_request_id)
        else:
            genparams['_batch_fallback'] = True
            ret = state.handle.generate(inputs)
        outstr = ""
        if ret.status==1:
            outstr = ret.text.decode("UTF-8","ignore")
        if batch_request_id >= 0 and not stream_flag:
            state.handle.batch_generate_release(batch_request_id)
            genparams.pop('_batch_request_id', None)
            genparams.pop('_batch_expected', None)
            genparams.pop('_batch_fallback', None)
        if trimstop:
            for trim_str in stop_sequence:
                sindex = outstr.find(trim_str)
                if sindex != -1 and trim_str!="":
                    outstr = outstr[:sindex]
        return {"text":outstr,"status":ret.status,"stopreason":ret.stopreason,"prompt_tokens":ret.prompt_tokens, "completion_tokens": ret.completion_tokens}

def continuous_batching_python_eligible(genparams, api_format):
    if getattr(state.args, "continuous_batching", 0) <= 1 or api_format <= 0:
        return False
    model_path = str(getattr(state.args, "model_param", "") or "").lower()
    if model_path and not model_path.endswith(".gguf"):
        utfprint("Batching disabled due to file format",2)
        return False
    if not getattr(state.args, "noshift", False) or getattr(state.args, "smartcontext", False) or getattr(state.args, "draftmodel", "") or getattr(state.args, "enableguidance", False):
        utfprint("Batching disabled due to loaded settings",2)
        return False
    if genparams.get("negative_prompt") or genparams.get("images") or genparams.get("audio"):
        utfprint("Batching disabled due to media",2)
        return False
    if genparams.get("grammar") or genparams.get("grammar_retain_state") or genparams.get("banned_tokens") or genparams.get("banned_strings"):
        utfprint("Batching disabled due to grammar or bans",2)
        return False
    if tryparsefloat(genparams.get("dry_multiplier", 0), 0) or tryparseint(genparams.get("mirostat", 0), 0) or tryparsefloat(genparams.get("xtc_probability", 0), 0) or tryparsefloat(genparams.get("nsigma", 0), 0):
        utfprint("Batching disabled due to samplers set 1",2)
        return False
    if tryparsefloat(genparams.get("smoothing_factor", 0), 0) or tryparsefloat(genparams.get("adaptive_target", -1), -1) > 0 or genparams.get("using_openai_tools", False):
        utfprint("Batching disabled due to samplers set 2",2)
        return False
    if tryparsefloat(genparams.get("top_a", 0), 0) or tryparsefloat(genparams.get("tfs", 1), 1) != 1 or tryparsefloat(genparams.get("dynatemp_range", 0), 0):
        utfprint("Batching disabled due to samplers set 3",2)
        return False
    if genparams.get("sampler_order") and genparams.get("sampler_order") != [6, 0, 1, 3, 4, 2, 5]:
        utfprint("Batching disabled due to sampler order",2)
        return False
    if genparams.get("reasoning_effort"):
        utfprint("Batching disabled due to reasoning",2)
        return False
    return True

def tokenize_ids(countprompt,tcaddspecial):
    rawcountdata = state.handle.token_count(countprompt.encode("UTF-8"),tcaddspecial)
    count = rawcountdata.count
    hardlimit = (2**31) - 1
    countlimit = count if (count>=0 and count<=hardlimit) else 0
    if count > hardlimit:
        utfprint("Warning: TokenCount exceeds max limit.")
    # the above protects the server in case the count limit got corrupted
    countdata = [rawcountdata.ids[i] for i in range(countlimit)]
    return countdata

def detokenize_ids(tokids,addspecial):
    tokidslen = len(tokids)
    detokstr = ""
    if tokidslen > 0 and tokidslen < 65536:
        inputs = detokenize_inputs()
        inputs.count = tokidslen
        inputs.special = addspecial
        inputs.ids = (ctypes.c_int * tokidslen)()
        for i, cid in enumerate(tokids):
            inputs.ids[i] = cid
        detok = state.handle.detokenize(inputs)
        detokstr = ctypes.string_at(detok).decode("UTF-8","ignore")
    return detokstr
