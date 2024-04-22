from model.tokenizer import MyTokenizer
from model.model_params import ModelParams
from transformers import AutoModelForCausalLM
from model.lm import LM


def get_gpt2():
    # sometimes i just want to look around it,
    # nothing to do with all the training stuff i have -
    # just some gpt2 poking. nice to have this in the base repository too
    model_params = ModelParams()
    model_params.layer_architecture = "gpt2-transformer"

    transformer = AutoModelForCausalLM.from_pretrained("gpt2")

    # maximum length it can take
    model_params.max_seq_len = transformer.transformer.wpe.weight.shape[0]

    # my code uses this attribute
    transformer.n_tokens = transformer.transformer.wte.weight.shape[0]

    tokenizer = MyTokenizer(data=None, name="gpt2", no_crop=True)
    lm = LM(tokenizer, transformer, model_params)
    lm.is_from_pretrained = "gpt2"
    return lm
