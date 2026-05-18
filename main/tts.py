import re
import json

from . import state
from .structs import tts_load_model_inputs, tts_generation_inputs, tts_generation_outputs
from .backend import set_backend_props
from .utils import simple_lcg_hash


def tts_load_model(ttc_model_filename,cts_model_filename):
    inputs = tts_load_model_inputs()
    inputs.ttc_model_filename = ttc_model_filename.encode("UTF-8") if ttc_model_filename else "".encode("UTF-8")
    inputs.cts_model_filename = cts_model_filename.encode("UTF-8") if cts_model_filename else "".encode("UTF-8")
    inputs.gpulayers = (999 if state.args.ttsgpu else 0)
    inputs.flash_attention = (False if state.args.noflashattention else True)
    thds = state.args.threads
    if state.args.ttsthreads and state.args.ttsthreads > 0:
        ttst = int(state.args.ttsthreads)
        if ttst > 0:
            thds = ttst
    inputs.threads = thds
    inputs.ttsmaxlen = state.args.ttsmaxlen if state.args.ttsmaxlen < 4096 else 4096
    inputs = set_backend_props(inputs)
    ret = state.handle.tts_load_model(inputs)
    return ret

def tts_prepare_voice_json(jsonstr):
    try:
        if not jsonstr:
            return None
        parsed_json = json.loads(jsonstr)
        txt = parsed_json.get("text","")
        items = parsed_json.get("words",[])
        processed = ""
        if txt=="" or not items or len(items)<1:
            return None
        for item in items:
            word = item.get("word","")
            duration = item.get("duration","")
            codes = item.get("codes",[])
            codestr = ""
            for c in codes:
                codestr += f"<|{c}|>"
            processed += f"{word}<|t_{duration:.2f}|><|code_start|>{codestr}<|code_end|>\n"
        return {"phrase":txt.strip()+".","voice":processed.strip()}
    except Exception:
        return None

def tts_extract_instruction(x):
    match = re.match(r'^\[([^\]]+)\]\s*(.+)$', x, re.DOTALL)
    if match:
        instruction = match.group(1)
        x1 = match.group(2)
        return x1, (instruction if instruction else "")
    return x, ""

def tts_generate(genparams):
    prompt = genparams.get("input", genparams.get("text", ""))
    prompt = prompt.strip()
    voice = 1
    speaker_json = tts_prepare_voice_json(genparams.get("speaker_json","")) #handle custom json voices
    voicestr = genparams.get("voice", genparams.get("speaker_wav", ""))
    oai_voicemap = ["alloy","onyx","echo","nova","shimmer"] # map to kcpp defaults
    voice_mapping = state.voicelist
    normalized_voice = voicestr.strip().lower() if voicestr else ""
    if normalized_voice.endswith(".wav"):
        normalized_voice = normalized_voice[:-4]
    if normalized_voice in voice_mapping:
        voice = voice_mapping.index(normalized_voice) + 1
    elif normalized_voice in oai_voicemap:
        voice = oai_voicemap.index(normalized_voice) + 1
    else:
        voice = simple_lcg_hash(voicestr.strip()) if voicestr else 1
    inputs = tts_generation_inputs()
    inputs.custom_speaker_voice = normalized_voice.encode("UTF-8")
    ttsinstruction = genparams.get("instruction", "")
    # if no instruction provided, extract from text
    if not genparams.get("instruction", ""):
        prompt, ttsinstruction = tts_extract_instruction(prompt)
    inputs.speaker_instruction = ttsinstruction.encode("UTF-8")
    inputs.prompt = prompt.encode("UTF-8")
    inputs.speaker_seed = voice
    aseed = -1
    try:
        aseed = int(genparams.get("seed", -1))
    except Exception:
        aseed = -1
    inputs.audio_seed = aseed
    if speaker_json:
        inputs.custom_speaker_text = speaker_json.get("phrase","").encode("UTF-8")
        inputs.custom_speaker_data = speaker_json.get("voice","").encode("UTF-8")
        inputs.speaker_seed = 100
    else:
        inputs.custom_speaker_text = "".encode("UTF-8")
        inputs.custom_speaker_data = "".encode("UTF-8")
    reference_audio = state.voicebank.get(voicestr,"") #for cloned voices in qwen3tts
    if reference_audio and reference_audio.startswith("data:audio"):
        reference_audio = reference_audio.split(",", 1)[1]
    inputs.reference_audio = reference_audio.encode("UTF-8")
    ret = state.handle.tts_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.data.decode("UTF-8","ignore")
    return outstr
