import os
import ctypes
import threading
import time
from datetime import datetime

# constants
sampler_order_max = 7
tensor_split_max = 16
images_max = 16
audio_max = 16
bias_min_value = -100.0
bias_max_value = 100.0
logprobs_max = 10
default_draft_amount = 8
default_ttsmaxlen = 4096
default_visionmaxres = 1024
net_save_slots = 12
savestate_limit_default = 5
savestate_limit = 0
default_vae_tile_threshold = 768
default_native_ctx = 16384
default_genlen = 1024
overridekv_max = 16
default_autofit_padding = 1024
lora_filenames_max = 4
multiuser_concurrent_limit = 10
swa_padding_default = 0
default_reqtimeout = 600

# abuse prevention
stop_token_max = 256
ban_token_max = 768
logit_bias_max = 512
dry_seq_break_max = 128
extra_images_max = 4

# global vars
KcppVersion = "1.113.1"
showdebug = True
kcpp_instance = None
global_memory = {"tunnel_url": "", "restart_target":"", "input_to_exit":False, "load_complete":False, "restart_override_base_config":"", "last_active_timestamp":datetime.now(), "triggered_sleeping":False, "current_model":"initial_model", "base_config":"", "swapReqType": None, "autoswapmode": False}
using_gui_launcher = False

handle = None
friendlymodelname = "inactive"
friendlysdmodelname = "inactive"
friendlyembeddingsmodelname = "inactive"
autoswapmode = False
textName = None
sttName = None
ttsName = None
embedName = None
musicName = None
imageName = None
mmprojName = None
lastgeneratedcomfyimg = b''
lastuploadedcomfyimg = b''
fullsdmodelpath = ""
password = ""
fullwhispermodelpath = ""
ttsmodelpath = ""
embeddingsmodelpath = ""
musicllmmodelpath = ""
musicdiffusionmodelpath = ""
imglora_preload = []
imglora_bypath = {}
imglora_name2path = {}
imglora_cached = True
imglora_initial_fixed = True
maxctx = 8192
maxhordectx = 0
maxhordelen = 1024
modelbusy = threading.Lock()
batched_lock = threading.Lock()
batched_cond = threading.Condition(batched_lock)
batched_request_runner_count = 0
requestsinqueue = 0
ratelimitlookup = {}
defaultport = 5001
showsamplerwarning = True
showmaxctxwarning = True
showusedmemwarning = True
showmultigpuwarning = True
session_kudos_earned = 0
session_jobs = 0
session_starttime = None
exitcounter = -1
punishcounter = 0
rewardcounter = 0
totalgens = 0
currentusergenkey = ""
pendingabortkey = ""
args = None
runmode_untouched = True
modelfile_extracted_meta = None
calulated_gpu_overhead = 0
importvars_in_progress = False
has_multiplayer = False
has_audio_support = False
has_vision_support = False
has_whisper = False
cached_chat_template = None
cached_sd_info = {}
cached_jinja_kwargs = None
savedata_obj = None
mcp_connections = []
mcp_lock = threading.Lock()
multiplayer_story_data_compressed = None
multiplayer_turn_major = 1
multiplayer_turn_minor = 1
multiplayer_dataformat = ""
multiplayer_lastactive = {}
websearch_lastquery = ""
websearch_lastresponse = []
preloaded_story = None
chatcompl_adapter = None
chatcompl_adapter_list = None
embedded_kailite = None
embedded_kailite_gz = None
embedded_kcpp_docs = None
embedded_kcpp_docs_gz = None
embedded_kcpp_sdui = None
embedded_kcpp_sdui_gz = None
embedded_lcpp_ui_gz = None
embedded_musicui = None
embedded_musicui_gz = None
preloaded_custom_jinja = ""
voicebank = {}
voicelist = ["kobo","cheery","sleepy","shouty","chatty"]
sslvalid = False
nocertify = False
start_time = time.time()
last_req_time = time.time()
last_non_horde_req_time = time.time()
currfinishreason = None
zenity_recent_dir = os.getcwd()
zenity_permitted = True
thinkformats = [{"start":"<|channel|>analysis<|message|>","end":"<|start|>assistant<|channel|>final<|message|>"},
                {"start":"<think>","end":"</think>"},
                {"start":"<seed:think>","end":"</seed:think>"},
                {"start":"<|channel>thought","end":"<channel|>"}]
tool_call_pairs = [
    ("<tool_call>", "</tool_call>", True),
    ("<seed:tool_call>", "</seed:tool_call>", True),
    ("<|tool_call_begin|>", "<|tool_call_end|>", True),
    ("<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", True),
    ("<minimax:tool_call>", "</minimax:tool_call>", True),
    ("<|tool_call>", "<tool_call|>", True),
    ("<|end|><|start|>assistant<|channel|>commentary to=", "", False),
]
deprecated_keys = {
    "hordeconfig",
    "sdconfig",
    "noblas",
    "nommap",
    "pipelineparallel",
    "sdnotile",
    "forceversion",
    "sdgendefaults",
    "flashattention",
}

saved_stdout = None
saved_stderr = None
saved_stdout_py = None
saved_stderr_py = None
stdout_nullfile = None
stdout_nullfile_py = None

CUDevices = ["1","2","3","4","All"]
CUDevicesNames = ["","","","",""]
VKDevicesNames = ["","","",""]
VKIsDGPU = [0,0,0,0]
MaxMemory = [0]
MaxFreeMemory = [0]

# path helpers (needed at module level to initialize lib_* below)
def getdirpath():
    # Returns the directory containing koboldcpp.py (parent of main/)
    return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

def getabspath():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def file_exists(filename):
    return os.path.exists(os.path.join(getdirpath(), filename))

def pick_existant_file(ntoption, nonntoption):
    precompiled_prefix = "precompiled_"
    ntexist = file_exists(ntoption)
    nonntexist = file_exists(nonntoption)
    precompiled_ntexist = file_exists(precompiled_prefix + ntoption)
    precompiled_nonntexist = file_exists(precompiled_prefix + nonntoption)
    if os.name == 'nt':
        if not ntexist and precompiled_ntexist:
            return (precompiled_prefix + ntoption)
        if nonntexist and not ntexist:
            return nonntoption
        return ntoption
    else:
        if not nonntexist and precompiled_nonntexist:
            return (precompiled_prefix + nonntoption)
        if ntexist and not nonntexist:
            return ntoption
        return nonntoption

lib_default = pick_existant_file("koboldcpp_default.dll","koboldcpp_default.so")
lib_failsafe = pick_existant_file("koboldcpp_failsafe.dll","koboldcpp_failsafe.so")
lib_noavx2 = pick_existant_file("koboldcpp_noavx2.dll","koboldcpp_noavx2.so")
lib_vulkan_failsafe = pick_existant_file("koboldcpp_vulkan_failsafe.dll","koboldcpp_vulkan_failsafe.so")
lib_cublas = pick_existant_file("koboldcpp_cublas.dll","koboldcpp_cublas.so")
lib_hipblas = pick_existant_file("koboldcpp_hipblas.dll","koboldcpp_hipblas.so")
lib_vulkan = pick_existant_file("koboldcpp_vulkan.dll","koboldcpp_vulkan.so")
lib_vulkan_noavx2 = pick_existant_file("koboldcpp_vulkan_noavx2.dll","koboldcpp_vulkan_noavx2.so")
libname = ""
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_vulkan, "Use Vulkan"),
    (lib_noavx2, "Use CPU (Old CPU)"),
    (lib_vulkan_noavx2, "Use Vulkan (Old CPU)"),
    (lib_vulkan_failsafe, "Use Vulkan (Older CPU)"),
    (lib_failsafe, "Failsafe Mode (Older CPU)")]
default_option, cublas_option, hipblas_option, vulkan_option, noavx2_option, vulkan_noavx2_option, vulkan_failsafe_option, failsafe_option = (opt if file_exists(lib) or (os.name == 'nt' and file_exists(opt + ".dll")) else None for lib, opt in lib_option_pairs)
runopts = [opt for lib, opt in lib_option_pairs if file_exists(lib)]
