import json

from . import state
from .structs import music_load_model_inputs, music_generation_inputs, music_generation_outputs
from .backend import set_backend_props


def music_load_model(musicllm,musicembedding,musicdiffusion,musicvae):
    inputs = music_load_model_inputs()
    inputs.musicllm_filename = musicllm.encode("UTF-8")
    inputs.musicembedding_filename = musicembedding.encode("UTF-8")
    inputs.musicdiffusion_filename = musicdiffusion.encode("UTF-8")
    inputs.musicvae_filename = musicvae.encode("UTF-8")
    inputs.lowvram = True if state.args.musiclowvram else False
    inputs = set_backend_props(inputs)
    ret = state.handle.music_load_model(inputs)
    return ret

def music_generate_codes(genparams):
    input_json = json.dumps(genparams)
    inputs = music_generation_inputs()
    inputs.is_planner_mode = True
    inputs.stereo = genparams.get('stereo', True)
    inputs.use_mp3 = genparams.get('use_mp3', False)
    inputs.gen_codes =  genparams.get('gen_codes', False)
    inputs.rewrite_caption =  genparams.get('rewrite_caption', True)
    inputs.input_json = input_json.encode("UTF-8")
    inputs.music_reference_audio_data = "".encode("UTF-8")
    ret = state.handle.music_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.music_output_json.decode("UTF-8","ignore")
        outstr = json.dumps(json.loads(outstr))
    return outstr

def music_generate_audio(genparams):
    input_json = json.dumps(genparams)
    inputs = music_generation_inputs()
    inputs.is_planner_mode = False
    inputs.stereo = genparams.get('stereo', True)
    inputs.use_mp3 = genparams.get('use_mp3', False)
    inputs.gen_codes =  genparams.get('gen_codes', False)
    inputs.rewrite_caption =  genparams.get('rewrite_caption', True)
    inputs.input_json = input_json.encode("UTF-8")
    refaudio = genparams.get('music_reference_audio_data', None)
    inputs.music_reference_audio_data = (refaudio.encode("UTF-8") if (refaudio and refaudio!="") else "".encode("UTF-8"))
    ret = state.handle.music_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.data.decode("UTF-8","ignore")
    return outstr
