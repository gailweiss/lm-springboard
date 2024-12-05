from model.tokenizer import MyTokenizer
from model.model_params import make_mp
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from model.lm import LM


chkpts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + \
         [1000*i for i in range(1,144)]


#### notes ####
# 1. pythia was not trained with a BOS token: see
#    https://github.com/EleutherAI/pythia/issues/123 . 
#    Instead all documents being trained were concatenated with an </endoftext>
#    token between them (registered in the pythia_tokenizer as its bos and eos
#    token). If using this to fine tune, then can arbitrarily decide that
#    target task does start with BOS, but if using this to evaluate its
#    perplexities on different tokens, then whether this BOS should be added or
#    not depends on whether my samples are aligned with actual starts of new
#    sequences


def get_pythia(chkpt_id=0, sizestr="70m", deduped=True, cap_max_seq_len=-1,
               cache=False):
    stepstr = f"step{chkpts[chkpt_id]}"
    modelstr =  f"EleutherAI/pythia-{sizestr}"
    loadstr = modelstr
    if deduped:
        modelstr += "-deduped"
    modelstr += f"/{stepstr}"
    if cache:
        cache_dir = f"../../cached_hf_models/{modelstr}"

    gptneox = GPTNeoXForCausalLM.from_pretrained(loadstr, revision=stepstr)
    pythia_tokenizer = AutoTokenizer.from_pretrained(loadstr, revision=stepstr)

    model_params = make_mp()

    def not_layernorm(param_name):
        # gptneox layernorm names contain: 
        # "layernorm" (in the layers), "layer_norm" (at the end)
        def not_layernorm(param_name):
            if "layernorm" in param_name:
                return False
            if "layer_norm" in param_name:
                return False
            return True
    gptneox.not_layernorm = not_layernorm

    model_params.from_os_pretrained = modelstr
    model_params.max_seq_len = \
        gptneox.gpt_neox.layers[0].attention.rotary_emb.max_position_embeddings
    model_params.layer_architecture = "gptneox-transformer"

    if cap_max_seq_len > 0:
        model_params.max_seq_len = min(model_params.max_seq_len,
                                       cap_max_seq_len)

    model_params.n_layers = len(gptneox.gpt_neox.layers)
    model_params.n_heads = \
        gptneox.gpt_neox.layers[0].attention.num_attention_heads
    model_params.dim = \
        gptneox.gpt_neox.layers[0].input_layernorm.weight.shape[0]
    hidden_dim = gptneox.gpt_neox.layers[0].mlp.dense_h_to_4h.out_features
    model_params.dim_ff_factor = hidden_dim // model_params.dim 
    # as suggested by name above, this should be 4...
    assert model_params.dim_ff_factor == 4
    assert model_params.dim_ff_factor * model_params.dim == hidden_dim
    model_params.tokenizer_source_name = modelstr
    model_params.pos_encoding = modelstr
    model_params.individual_head_params = False

    tokenizer = MyTokenizer(data=None, name=modelstr, no_crop=True)
    lm = LM(tokenizer, gptneox, model_params)
    return lm
