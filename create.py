from data.dataloader import get_data, LMDataModule, DataParams
from model.tokenizer import MyTokenizer
from model.transformer.transformer import Transformer
from model.model_params import ModelParams
from train.train_params import TrainParams
from model.lm import LM


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


def make_model_and_data(data_params, model_params, train_params,
                        tokenizer=None, verbose=True):
    tokenizer, dataset = make_tokenizer_and_data(data_params,
                                                 model_params,
                                                 train_params,
                                                 existing_tokenizer=tokenizer,
                                                 verbose=verbose)
    if "transformer" in model_params.layer_architecture:
        transformer = Transformer(model_params, train_params)
    else:  # hoping to add e.g. RNNs, S6, etc in the future
        raise Exception("unknown layer_architecture:" +
                        f"{model_params.layer_architecture}")
    lm = LM(tokenizer, transformer, model_params)

    return lm, dataset


def quick_data_grab(dataset_name, existing_tokenizer=None):
    dp = DataParams(dataset_name=dataset_name, debug_crop=500)
    mp = ModelParams()
    tp = TrainParams()
    return make_tokenizer_and_data(dp, mp, tp,
                                   existing_tokenizer=existing_tokenizer)
