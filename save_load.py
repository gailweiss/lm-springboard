import json
from util import prepare_directory
from pathlib import Path
from os.path import join as path_join
from create import make_model, make_datamodule
from train.trainer import Trainer
from train.train_params import TrainParams
from model.model_params import ModelParams
from model.tokenizer import load_stored_tokenizer_if_exists
from data.dataloader import DataParams, get_existing_datamodule
import glob


final_chkpt = "final"


def save_model(folder_name, pl_trainer, my_trainer, model_params, data_params,
               train_params):
    prepare_directory(folder_name)
    for p, n in [(model_params, "model_params"),
                 (data_params, "data_params"),
                 (train_params, "train_params")]:
        with open(path_join(folder_name, f"{n}.json"), "w") as f:
            json.dump(vars(p), f)
    with open(path_join(folder_name, "train_stats.json"), "w") as f:
        json.dump(my_trainer.logged_stats_dict, f)
    pl_trainer.model.model.tokenizer.save(folder_name)
    pl_trainer.save_checkpoint(path_join(folder_name, "model.model"))


def load_model_info(folder_name):
    if Path(folder_name).exists() is False:
        raise ValueError(f"Folder {folder_name} does not exist")

    res = {"params": {}}
    with open(path_join(folder_name, "model_params.json"), "r") as f:
        res["params"]["model_params"] = ModelParams(**json.load(f))
    with open(path_join(folder_name, "data_params.json"), "r") as f:
        res["params"]["data_params"] = DataParams(**json.load(f))
    with open(path_join(folder_name, "train_params.json"), "r") as f:
        res["params"]["train_params"] = TrainParams(**json.load(f))

    with open(path_join(folder_name, "train_stats.json"), "r") as f:
        res["train_stats"] = json.load(f)
        res["train_stats"]["total_train_samples"] = \
            res["train_stats"].get("n_train_samples",[[0]])[-1][0]
            # if not got, then this is the model at time 0 - no training yet

    return res


def get_datamodule(data_params, model_params,
                            verbose=True, keep_datamodule=False):
    data = get_existing_datamodule(data_params, model_params)
    if None is not data:
        return data

    return make_datamodule(data_params, model_params,
                           verbose=verbose, keep_datamodule=keep_datamodule)


def load_model(folder_name, full=False, verbose=True, with_data=False,
               keep_datamodule=False):

    if not Path(folder_name).exists():
        raise ValueError(f"Folder {folder_name} does not exist")

    res = load_model_info(folder_name)

    res["params"]["train_params"].no_wandb = True  # no wandb in loaded models
    # - or it will try (unsuccessfully) to send validation losses to wandb when
    # doing a validation run

    # get tokenizer, through dataset if asked to get data, else seek stored
    # tokenizer for model
    if with_data:
        res["dataset"] = get_datamodule(res["params"]["data_params"],
                                        res["params"]["model_params"], 
                                        verbose=verbose,
                                        keep_datamodule=keep_datamodule)
        tokenizer = res["dataset"].tokenizer
    else:
        res["dataset"] = None
        tokenizer = load_stored_tokenizer_if_exists(
            res["params"]["model_params"].tokenizer_source_name, folder_name,
            verbose)
    
    # todo: if this happens to be a gpt2 based model, then its fine not to have
    # any stored info on the tokenizer, can load the appropriate one from hf
    # instead. for now not an issue though

    assert None is not tokenizer, "no data, and didnt find saved tokenizer"

    # prepare model to be filled with saved parameters
    lm = make_model(res["params"]["model_params"], 
                    res["params"]["train_params"], 
                    tokenizer)

    # fill model with saved params. don't know if this will affect the
    # passed in parameter lm, so will get it explicitly back from the trainer
    model_trainer = Trainer.load_from_checkpoint(
        path_join(folder_name, "model.model"), model=lm,
        train_params=res["params"]["train_params"])

    res["lm"] = model_trainer.model
    res["lm"].eval()

    return res


