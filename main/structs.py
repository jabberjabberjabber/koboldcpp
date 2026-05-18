import ctypes
from . import state

class logit_bias(ctypes.Structure):
    _fields_ = [("token_id", ctypes.c_int32),
                ("bias", ctypes.c_float)]

class token_count_outputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("ids", ctypes.POINTER(ctypes.c_int))]

class detokenize_inputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("ids", ctypes.POINTER(ctypes.c_int)),
                ("special", ctypes.c_bool)]

class logprob_item(ctypes.Structure):
     _fields_ = [("option_count", ctypes.c_int),
                ("selected_token", ctypes.c_char_p),
                ("selected_logprob", ctypes.c_float),
                ("selected_token_id", ctypes.c_int32),
                ("tokens", ctypes.c_char_p * state.logprobs_max),
                ("token_ids", ctypes.c_int32 * state.logprobs_max),
                ("logprobs", ctypes.POINTER(ctypes.c_float))]

class last_logprobs_outputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("logprob_items", ctypes.POINTER(logprob_item))]

class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("low_vram", ctypes.c_bool),
                ("use_mmq", ctypes.c_bool),
                ("splitmode", ctypes.c_int),
                ("executable_path", ctypes.c_char_p),
                ("model_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("draftmodel_filename", ctypes.c_char_p),
                ("draft_amount", ctypes.c_int),
                ("draft_gpulayers", ctypes.c_int),
                ("draft_gpusplit", ctypes.c_float * state.tensor_split_max),
                ("mmproj_filename", ctypes.c_char_p),
                ("mmproj_cpu", ctypes.c_bool),
                ("visionmaxres", ctypes.c_int),
                ("visionmintokens", ctypes.c_int),
                ("visionmaxtokens", ctypes.c_int),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("use_smartcontext", ctypes.c_bool),
                ("use_contextshift", ctypes.c_bool),
                ("use_fastforward", ctypes.c_bool),
                ("kcpp_main_gpu", ctypes.c_int),
                ("batchsize", ctypes.c_int),
                ("autofit", ctypes.c_bool),
                ("autofit_tax_mb", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("rope_freq_scale", ctypes.c_float),
                ("rope_freq_base", ctypes.c_float),
                ("overridenativecontext", ctypes.c_int),
                ("moe_experts", ctypes.c_int),
                ("moecpu", ctypes.c_int),
                ("no_bos_token", ctypes.c_bool),
                ("load_guidance", ctypes.c_bool),
                ("override_kv", ctypes.c_char_p * state.overridekv_max),
                ("override_tensors", ctypes.c_char_p),
                ("flash_attention", ctypes.c_bool),
                ("tensor_split", ctypes.c_float * state.tensor_split_max),
                ("quant_k", ctypes.c_int),
                ("quant_v", ctypes.c_int),
                ("check_slowness", ctypes.c_bool),
                ("jinja_template", ctypes.c_char_p),
                ("highpriority", ctypes.c_bool),
                ("swa_support", ctypes.c_bool),
                ("swa_padding", ctypes.c_int),
                ("smartcache", ctypes.c_bool),
                ("smartcacheslots", ctypes.c_int),
                ("pipelineparallel", ctypes.c_bool),
                ("lora_multiplier", ctypes.c_float),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int),
                ("continuous_batching_slots", ctypes.c_int)]

class generation_inputs(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("memory", ctypes.c_char_p),
                ("negative_prompt", ctypes.c_char_p),
                ("guidance_scale", ctypes.c_float),
                ("images_len", ctypes.c_int),
                ("images", ctypes.POINTER(ctypes.c_char_p)),
                ("audio_len", ctypes.c_int),
                ("audio", ctypes.POINTER(ctypes.c_char_p)),
                ("max_context_length", ctypes.c_int),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_a", ctypes.c_float),
                ("top_p", ctypes.c_float),
                ("min_p", ctypes.c_float),
                ("typical_p", ctypes.c_float),
                ("tfs", ctypes.c_float),
                ("nsigma", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int),
                ("rep_pen_slope", ctypes.c_float),
                ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("xtc_threshold", ctypes.c_float),
                ("xtc_probability", ctypes.c_float),
                ("sampler_order", ctypes.c_int * state.sampler_order_max),
                ("sampler_len", ctypes.c_int),
                ("allow_eos_token", ctypes.c_bool),
                ("bypass_eos_token", ctypes.c_bool),
                ("tool_call_fix", ctypes.c_bool),
                ("render_special", ctypes.c_bool),
                ("stream_sse", ctypes.c_bool),
                ("grammar", ctypes.c_char_p),
                ("grammar_retain_state", ctypes.c_bool),
                ("dynatemp_range", ctypes.c_float),
                ("dynatemp_exponent", ctypes.c_float),
                ("smoothing_factor", ctypes.c_float),
                ("smoothing_curve", ctypes.c_float),
                ("adaptive_target", ctypes.c_float),
                ("adaptive_decay", ctypes.c_float),
                ("dry_multiplier", ctypes.c_float),
                ("dry_base", ctypes.c_float),
                ("dry_allowed_length", ctypes.c_int),
                ("dry_penalty_last_n", ctypes.c_int),
                ("dry_sequence_breakers_len", ctypes.c_int),
                ("dry_sequence_breakers", ctypes.POINTER(ctypes.c_char_p)),
                ("stop_sequence_len", ctypes.c_int),
                ("stop_sequence", ctypes.POINTER(ctypes.c_char_p)),
                ("logit_biases_len", ctypes.c_int),
                ("logit_biases", ctypes.POINTER(logit_bias)),
                ("banned_tokens_len", ctypes.c_int),
                ("banned_tokens", ctypes.POINTER(ctypes.c_char_p)),
                ("reasoning_budget", ctypes.c_int)]

class generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("stopreason", ctypes.c_int),
                ("prompt_tokens", ctypes.c_int),
                ("completion_tokens", ctypes.c_int),
                ("text", ctypes.c_char_p)]

class sd_load_model_inputs(ctypes.Structure):
    _fields_ = [("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("threads", ctypes.c_int),
                ("quant", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("offload_cpu", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("vae_cpu", ctypes.c_bool),
                ("clip_cpu", ctypes.c_bool),
                ("diffusion_conv_direct", ctypes.c_bool),
                ("vae_conv_direct", ctypes.c_bool),
                ("taesd", ctypes.c_bool),
                ("tiled_vae_threshold", ctypes.c_int),
                ("t5xxl_filename", ctypes.c_char_p),
                ("clip1_filename", ctypes.c_char_p),
                ("clip2_filename", ctypes.c_char_p),
                ("vae_filename", ctypes.c_char_p),
                ("lora_len", ctypes.c_int),
                ("lora_filenames", ctypes.POINTER(ctypes.c_char_p)),
                ("lora_multipliers", ctypes.POINTER(ctypes.c_float)),
                ("lora_apply_mode", ctypes.c_int),
                ("photomaker_filename", ctypes.c_char_p),
                ("upscaler_filename", ctypes.c_char_p),
                ("img_hard_limit", ctypes.c_int),
                ("img_soft_limit", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class sd_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("negative_prompt", ctypes.c_char_p),
                ("init_images", ctypes.c_char_p),
                ("mask", ctypes.c_char_p),
                ("extra_images_len", ctypes.c_int),
                ("extra_images", ctypes.POINTER(ctypes.c_char_p)),
                ("flip_mask", ctypes.c_bool),
                ("denoising_strength", ctypes.c_float),
                ("cfg_scale", ctypes.c_float),
                ("distilled_guidance", ctypes.c_float),
                ("shifted_timestep", ctypes.c_int),
                ("flow_shift", ctypes.c_float),
                ("sample_steps", ctypes.c_int),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int),
                ("seed", ctypes.c_int),
                ("sample_method", ctypes.c_char_p),
                ("scheduler", ctypes.c_char_p),
                ("eta", ctypes.c_float),
                ("clip_skip", ctypes.c_int),
                ("vid_req_frames", ctypes.c_int),
                ("video_output_type", ctypes.c_int),
                ("remove_limits", ctypes.c_bool),
                ("circular_x", ctypes.c_bool),
                ("circular_y", ctypes.c_bool),
                ("cache_mode", ctypes.c_char_p),
                ("cache_options", ctypes.c_char_p),
                ("upscale", ctypes.c_bool),
                ("lora_len", ctypes.c_int),
                ("lora_filenames", ctypes.POINTER(ctypes.c_char_p)),
                ("lora_multipliers", ctypes.POINTER(ctypes.c_float))]

class sd_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("animated", ctypes.c_int),
                ("data", ctypes.c_char_p),
                ("data_extra", ctypes.c_char_p),
                ("info", ctypes.c_char_p)]

class sd_upscale_inputs(ctypes.Structure):
    _fields_ = [("init_images", ctypes.c_char_p),
                ("upscaling_resize", ctypes.c_int)]

class sd_info_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class whisper_load_model_inputs(ctypes.Structure):
    _fields_ = [("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class whisper_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("audio_data", ctypes.c_char_p),
                ("suppress_non_speech", ctypes.c_bool),
                ("langcode", ctypes.c_char_p)]

class whisper_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class tts_load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("ttc_model_filename", ctypes.c_char_p),
                ("cts_model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("ttsmaxlen", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class tts_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("speaker_seed", ctypes.c_int),
                ("audio_seed", ctypes.c_int),
                ("custom_speaker_voice", ctypes.c_char_p),
                ("custom_speaker_text", ctypes.c_char_p),
                ("custom_speaker_data", ctypes.c_char_p),
                ("reference_audio", ctypes.c_char_p),
                ("speaker_instruction", ctypes.c_char_p)]

class tts_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class embeddings_load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("embeddingsmaxctx", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class embeddings_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("truncate", ctypes.c_bool)]

class embeddings_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("count", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class music_load_model_inputs(ctypes.Structure):
    _fields_ = [("musicllm_filename", ctypes.c_char_p),
                ("musicembedding_filename", ctypes.c_char_p),
                ("musicdiffusion_filename", ctypes.c_char_p),
                ("musicvae_filename", ctypes.c_char_p),
                ("lowvram", ctypes.c_bool),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class music_generation_inputs(ctypes.Structure):
    _fields_ = [("is_planner_mode", ctypes.c_bool),
                ("stereo", ctypes.c_bool),
                ("use_mp3", ctypes.c_bool),
                ("gen_codes", ctypes.c_bool),
                ("rewrite_caption", ctypes.c_bool),
                ("input_json", ctypes.c_char_p),
                ("music_reference_audio_data", ctypes.c_char_p)]

class music_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("music_output_json", ctypes.c_char_p),
                ("data", ctypes.c_char_p)]
