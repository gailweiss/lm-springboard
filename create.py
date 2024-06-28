from data.dataloader import get_data, LMDataModule, DataParams
from model.tokenizer import MyTokenizer
from model.transformer.transformer import Transformer
from model.model_params import ModelParams
from train.train_params import TrainParams
from model.lm import LM
from gpt2 import get_gpt2
import dataclasses
from train.lora import apply_lora_to_model


def make_tokenizer_and_data(data_params, model_params, train_params,
                            existing_tokenizer=None, verbose=True):
    # the data makes the tokenizer, which in turn is important for the model,
    # because the model needs to know how many tokens its using,
    # so the creation chain has a specific order:

    # 1. need the dataset as it may influence the tokenizer, e.g. if
    # cropping from a loaded huggingface tokenizer, or making a char-
    # level tokenizer
    plain_data = get_data(data_params)

    def as_list(d):
        if isinstance(d, list):
            return d
        else:
            return d["train"] + d["validation"] + d["test"]

    def remove_type_markers(d):
        if isinstance(d[0], tuple):
            return [s for s, t in d]
        else:
            return d

    # 2. make base tokenizer
    if None is existing_tokenizer:
        custom_tokenizer_ntokens = model_params.custom_tokenizer_ntokens
        tokenizer = MyTokenizer(remove_type_markers(as_list(plain_data)),
                                name=model_params.tokenizer_source_name,
                                custom_vocab_size=custom_tokenizer_ntokens,
                                verbose_init=verbose)
    else:
        tokenizer = existing_tokenizer

    # 3. make a data module, with the tokenizer.
    # this one does take the true data, to maintain the train/val/test split
    dataset = LMDataModule(plain_data, tokenizer, data_params, model_params,
                           verbose_init=verbose)

    return tokenizer, dataset


def sync_model_params(requested_model_params, loaded_model_params):
    # these factors are from the loaded model, so write them back:
    for a in ["n_layers", "n_heads", "dim", "dim_ff_factor",
              "tokenizer_source_name", "custom_tokenizer_ntokens",
              "layer_architecture", "from_os_pretrained",
              "individual_head_params", "pos_encoding", "max_seq_len"]:
        setattr(requested_model_params, a, getattr(loaded_model_params, a))


def make_model_and_data(data_params, model_params, train_params,
                        tokenizer=None, verbose=True, skip_data=False):
    if model_params.from_saved or (model_params.from_os_pretrained == "gpt2"):
        if model_params.from_saved:
            from model_explorer import get_model_by_timestamp
            # uses saver to load which uses create for the skeleton, i.e. it's
            # going to be calling this function again, but with the saved
            # model_params as opposed to these ones. instead of implementing
            # getting from timestamp twice, i'll be a bit messy and just import
            # the function here where it won't cause a circular import
            lm = get_model_by_timestamp(model_params.from_saved)[0]
        elif model_params.from_os_pretrained == "gpt2":
            lm = get_gpt2()
        elif model_params.from_os_pretrained:
            raise NotImplementedError("unknown pretrained model requested:" +
                                      f"{model_params.from_os_pretrained}")
        sync_model_params(model_params, lm.model_params)
        if None is not tokenizer:
            # (this better be a call from saver.load_model, passing an already
            # cropped tokenizer)
            lm.tokenizer = tokenizer
        model, tokenizer = lm.decoder, lm.tokenizer
        sync_model_params(model_params, lm.model_params)

    if skip_data:
        dataset = None
        assert None is not tokenizer  # make sure did receive one
    else:
        tokenizer, dataset = make_tokenizer_and_data(
            data_params, model_params, train_params,
            existing_tokenizer=tokenizer, verbose=verbose)

    if not model_params.from_os_pretrained:
        if "transformer" in model_params.layer_architecture:
            model = Transformer(model_params, train_params)
        else:  # hoping to add e.g. RNNs, S6, etc in the future
            raise Exception("unknown layer_architecture:" +
                            f"{model_params.layer_architecture}")

    if not model_params.from_saved:
        lm = LM(tokenizer, model, model_params)

    if train_params.lora_rank > 0:
        apply_lora_to_model(lm, train_params)

    return lm, dataset


def quick_data_grab(dataset_name, tokenizer_source_name="gpt2",
                    existing_tokenizer=None, verbose=False):
    dp = DataParams(dataset_name=dataset_name, debug_crop=500)
    mp = ModelParams(tokenizer_source_name=tokenizer_source_name)
    tp = TrainParams()
    return make_tokenizer_and_data(dp, mp, tp,
                                   existing_tokenizer=existing_tokenizer,
                                   verbose=verbose)
