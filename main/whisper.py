from . import state
from .structs import whisper_load_model_inputs, whisper_generation_inputs, whisper_generation_outputs
from .backend import set_backend_props


def whisper_load_model(model_filename):
    inputs = whisper_load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs = set_backend_props(inputs)
    ret = state.handle.whisper_load_model(inputs)
    return ret

def whisper_generate(genparams):
    prompt = genparams.get("prompt", "")
    audio_data = genparams.get("audio_data", "")
    if audio_data.startswith("data:audio"):
        audio_data = audio_data.split(",", 1)[1]
    inputs = whisper_generation_inputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.audio_data = audio_data.encode("UTF-8")
    lc = genparams.get("langcode", genparams.get("language", "auto"))
    lc = lc.strip().lower() if (lc and lc.strip().lower()!="") else "auto"
    inputs.langcode = lc.encode("UTF-8")
    inputs.suppress_non_speech = genparams.get("suppress_non_speech", False)
    ret = state.handle.whisper_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.data.decode("UTF-8","ignore")
    return outstr
