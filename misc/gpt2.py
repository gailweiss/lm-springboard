from model.tokenizer import MyTokenizer
from model.model_params import make_mp
from transformers import AutoModelForCausalLM
from model.lm import LM


#### notes ####
# 1. gpt2 was not trained with a BOS token at each sequence, see:
#    https://github.com/huggingface/transformers/issues/3311 . 
#    If using this to fine tune, then can arbitrarily decide that finetuned
#    task does start with BOS, but if using this to evaluate its perplexities
#    on new samples, then whether this BOS should be added or not depends
#    on whether my samples are aligned with actual starts of new sequences


def get_gpt2(cap_max_seq_len=-1):
    # sometimes i just want to look around it,
    # nothing to do with all the training stuff i have -
    # just some gpt2 poking. nice to have this in the base repository too
    model_params = make_mp()
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

    def not_layernorm(param_name):
        huggingface_gpt2_layernorm_names = ["ln_1", "ln_2", "ln_f"]
        for layernorm_name in huggingface_gpt2_layernorm_names:
            if layernorm_name in param_name:
                return False
        return True
    gpt2.not_layernorm = not_layernorm

    # my code uses these
    model_params.from_os_pretrained = "gpt2"
    model_params.max_seq_len = gpt2.transformer.wpe.weight.shape[0]
    model_params.layer_architecture = "gpt2-transformer"

    if cap_max_seq_len > 0:
        model_params.max_seq_len = min(model_params.max_seq_len,
                                       cap_max_seq_len)

    # noting other attributes in model_params for completeness and potential
    # future use
    model_params.n_layers = len(gpt2.transformer.h)
    model_params.n_heads = gpt2.transformer.h[0].attn.num_heads
    model_params.dim = gpt2.lm_head.in_features
    hidden_dim = gpt2.transformer.h[0].mlp.c_fc.weight.shape[1]
    model_params.dim_ff_factor = hidden_dim // model_params.dim
    assert model_params.dim_ff_factor * model_params.dim == hidden_dim
    model_params.tokenizer_source_name = "gpt2"
    model_params.pos_encoding = "gpt2"
    model_params.individual_head_params = False

    tokenizer = MyTokenizer(data=None, name="gpt2", no_crop=True)
    lm = LM(tokenizer, gpt2, model_params, None)
    return lm
