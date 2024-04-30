from model.tokenizer import MyTokenizer
from model.model_params import ModelParams
from transformers import AutoModelForCausalLM
from model.lm import LM


def get_gpt2():
    # sometimes i just want to look around it,
    # nothing to do with all the training stuff i have -
    # just some gpt2 poking. nice to have this in the base repository too
    model_params = ModelParams()
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

    # my code uses these
    model_params.from_os_pretrained = "gpt2"
    model_params.max_seq_len = gpt2.transformer.wpe.weight.shape[0]
    model_params.layer_architecture = "gpt2-transformer"

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
    lm = LM(tokenizer, gpt2, model_params)
    return lm
