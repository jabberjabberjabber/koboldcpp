import json

from . import state
from .structs import embeddings_load_model_inputs, embeddings_generation_inputs, embeddings_generation_outputs
from .backend import set_backend_props


def embeddings_load_model(model_filename):
    inputs = embeddings_load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.gpulayers = (999 if state.args.embeddingsgpu else 0)
    inputs.flash_attention = (False if state.args.noflashattention else True)
    inputs.threads = state.args.threads
    inputs.use_mmap = state.args.usemmap
    inputs.embeddingsmaxctx = (state.args.embeddingsmaxctx if state.args.embeddingsmaxctx else state.args.contextsize) # for us to clamp to contextsize if embeddingsmaxctx unspecified
    inputs = set_backend_props(inputs)
    ret = state.handle.embeddings_load_model(inputs)
    return ret

def embeddings_generate(genparams):
    prompts = []
    if isinstance(genparams.get('input',[]), list):
        prompts = genparams.get('input',[])
    else:
        prompt = genparams.get("input", "")
        if prompt:
            prompts.append(prompt)

    tokarrs = []
    tokcnt = 0
    for prompt in prompts:
        tokarr = []
        tmpcnt = 0
        try:
            inputs = embeddings_generation_inputs()
            inputs.prompt = prompt.encode("UTF-8")
            inputs.truncate = genparams.get('truncate', True)
            ret = state.handle.embeddings_generate(inputs)
            if ret.status==1:
                outstr = ret.data.decode("UTF-8","ignore")
                tokarr = json.loads(outstr) if outstr else []
                tmpcnt = ret.count
        except Exception as e:
            tokarr = []
            tmpcnt = 0
            print(f"Error: {e}")
        tokarrs.append(tokarr)
        tokcnt += tmpcnt
    return {"count":tokcnt, "data":tokarrs}
