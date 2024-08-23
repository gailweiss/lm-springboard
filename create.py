from data.dataloader import get_data, LMDataModule, DataParams, \
                            datamodules_paths
from model.tokenizer import MyTokenizer
from model.transformer.transformer import Transformer
from model.model_params import ModelParams
from train.train_params import TrainParams
from model.lm import LM
from gpt2 import get_gpt2
import dataclasses
from train.lora import apply_lora_to_model
from util import get_timestamp


def make_datamodule(data_params, model_params, verbose=True,
                    keep_datamodule=False):
    # the data makes the tokenizer, with both needed for the data module

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
    custom_tokenizer_ntokens = model_params.custom_tokenizer_ntokens
    tokenizer = MyTokenizer(remove_type_markers(as_list(plain_data)),
                            name=model_params.tokenizer_source_name,
                            custom_vocab_size=custom_tokenizer_ntokens,
                            verbose_init=verbose)

    # 3. make a data module, with the tokenizer.
    # this one does take the true data, to maintain the train/val/test split
    dataset = LMDataModule(plain_data, tokenizer, data_params, model_params,
                           verbose_init=verbose)

    if keep_datamodule:
        # stores in first (i.e., default) datamodules path
        dataset.save_to_folder(f"{datamodules_paths[0]}/{get_timestamp()}")

    return dataset


def make_model(model_params, train_params, tokenizer):    
    if "transformer" in model_params.layer_architecture:
        model = Transformer(model_params, train_params)
    else:  # hoping to add e.g. RNNs, S6, etc in the future
        raise Exception("unknown layer_architecture:" +
                        f"{model_params.layer_architecture}")

    lm = LM(tokenizer, model, model_params)

    if train_params.lora_rank > 0:
        apply_lora_to_model(lm, train_params)

    return lm
