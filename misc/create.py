from data.dataloader import get_data, LMDataModule, datamodules_paths
from data.syntheticdata import SyntheticSamplesIterator
from model.tokenizer import MyTokenizer
from model.transformer.transformer import Transformer
from model.rnn.rnn import RNN
from model.lm import LM
from misc.gpt2 import get_gpt2
import dataclasses
from train.lora import apply_lora_to_model
from misc.util import get_probably_unique
from os.path import join as path_join


def make_datamodule(data_params, model_params, verbose=True,
                    keep_datamodule=False, given_tokenizer=None):
    # the data makes the tokenizer, with both needed for the data module

    # 1. need the dataset as it may influence the tokenizer, e.g. if
    # cropping from a loaded huggingface tokenizer, or making a char-
    # level tokenizer
    plain_data, lang_counters = get_data(data_params)

    def as_iterable(d):
        if isinstance(d, list) or isinstance(d, SyntheticSamplesIterator):
            return d
        else:
            return d["train"] + d["validation"] + d["test"]

    def remove_type_markers(d):
        if isinstance(d[0], tuple):
            return [s for s, t in d]
        else:
            return d

    # 2. make base tokenizer
    tokenizer = given_tokenizer
    if None is tokenizer:
        custom_tokenizer_ntokens = model_params.custom_tokenizer_ntokens
        tokenizer = MyTokenizer(remove_type_markers(as_iterable(plain_data)),
                                name=model_params.tokenizer_source_name,
                                custom_vocab_size=custom_tokenizer_ntokens,
                                verbose_init=verbose,
                                no_crop=not model_params.crop_tokenizer)

    # 3. make a data module, with the tokenizer.
    # this one does take the true data, to maintain the train/val/test split
    dataset = LMDataModule(plain_data, tokenizer, data_params, model_params,
                           verbose_init=verbose, lang_counters=lang_counters)

    if keep_datamodule:
        # stores in first (i.e., default) datamodules path
        dataset.save_to_folder(path_join(
            datamodules_paths[0], data_params.dataset_name,
            get_probably_unique()))

    return dataset


def make_model(model_params, train_params, tokenizer):
    if "transformer" in model_params.layer_architecture:
        model = Transformer(model_params, train_params)
    elif model_params.layer_architecture in ["torch-lstm", "torch-gru"]:
        model = RNN(model_params, train_params)
    else:  # hoping to add e.g. RNNs, S6, etc in the future
        raise Exception("unknown layer_architecture:" +
                        f"{model_params.layer_architecture}")

    lm = LM(tokenizer, model, model_params, train_params)

    if train_params.lora_rank > 0:
        apply_lora_to_model(lm, train_params)

    return lm
