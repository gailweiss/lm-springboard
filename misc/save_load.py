import json
from misc.util import prepare_directory
from misc.gpt2 import get_gpt2
from pathlib import Path
from os.path import join as path_join
from misc.create import make_model, make_datamodule
from train.trainer import Trainer
from train.train_params import make_tp
from model.model_params import make_mp
from model.tokenizer import load_stored_tokenizer_if_exists
from data.dataloader import get_existing_datamodule
from data.data_params import make_dp
import glob
from misc.util import printer_print as print
from dataclasses import asdict


try:
    with open("paths/models-paths.txt", "r") as f:
        models_paths = f.readlines()
        # e.g. ../saved-models, or more complicated if using cloud services
        models_paths = [p.strip("\n") for p in models_paths if not
                        p.startswith("#")]
        # models will be saved specifically in the first path in this file.
        # however, model_explorer.py will use *all* of the paths listed in
        # this file to find models.

except Exception as e:
    print("couldnt find extra models paths")
    models_paths = ["../saved-models"]


final_chkpt = "final"


def save_model(folder_name, pl_trainer, my_trainer, model_params, data_params,
               train_params, just_stats=False):
    prepare_directory(folder_name)
    for p, n in [(model_params, "model_params"),
                 (data_params, "data_params"),
                 (train_params, "train_params")]:
        with open(path_join(folder_name, f"{n}.json"), "w") as f:
            json.dump(vars(p), f)
        # json will turn tuples into lists, which is annoying. but i expect
        # all configs to have only tuples if they have iterables at all,
        # so the dataclass loading function successfully reverts this
    with open(path_join(folder_name, "train_stats.json"), "w") as f:
        json.dump(my_trainer.logged_stats_dict, f)
    if not just_stats:
        pl_trainer.model.model.tokenizer.save(folder_name)
        pl_trainer.save_checkpoint(path_join(folder_name, "model.model"))


def load_model_info(folder_name, with_train_stats=False):
    if not Path(folder_name).exists():
        raise ValueError(f"Folder {folder_name} does not exist")

    res = {"params": {}}
    with open(path_join(folder_name, "model_params.json"), "r") as f:
        res["params"]["model_params"] = make_mp(**json.load(f))
    with open(path_join(folder_name, "data_params.json"), "r") as f:
        res["params"]["data_params"] = make_dp(**json.load(f))
    with open(path_join(folder_name, "train_params.json"), "r") as f:
        res["params"]["train_params"] = make_tp(**json.load(f))

    for pn, pd in res["params"].items():
        for k, v in asdict(pd).items():
            if isinstance(v, list):
                # json turns tuples into lists, but saved configs (i.e.
                # true configs, not config 'lists') don't have lists: correct
                setattr(pd, k, tuple(v))

    if with_train_stats:
        with open(path_join(folder_name, "train_stats.json"), "r") as f:
            res["train_stats"] = json.load(f)
            res["train_stats"]["total_train_samples"] = \
                res["train_stats"].get("n_train_samples", [[0]])[-1][0]
            # if not got, this is the model at time 0 (no training yet)

    return res


def get_datamodule(data_params, model_params, verbose=True,
                   keep_datamodule=False, given_tokenizer=None):
    data = get_existing_datamodule(data_params, model_params)
    if None is not data:
        return data

    return make_datamodule(data_params, model_params, verbose=verbose,
                           keep_datamodule=keep_datamodule,
                           given_tokenizer=given_tokenizer)


def load_model(folder_name, full=False, verbose=True, with_data=False,
               keep_datamodule=False):

    if not Path(folder_name).exists():
        raise ValueError(f"Folder {folder_name} does not exist")

    res = load_model_info(folder_name)

    res["params"]["train_params"].no_wandb = True  # no wandb in loaded models
    # - or it will try (unsuccessfully) to send validation losses to wandb when
    # doing a validation run

    # if working from a pretrained model, get model and tokenizer directly
    if res["params"]["model_params"].from_os_pretrained:
        if res["params"]["model_params"].from_os_pretrained == "gpt2":
            res["lm"] = get_gpt2()
        else:
            e = NotImplementedError(
                "unknown pretrained model requested:" +
                f"{model_params.from_os_pretrained}")
            raise e
        tokenizer = res["lm"].tokenizer
    else:
        tokenizer = None  # will find soon, but not known yet

    # get tokenizer, through dataset if asked to get data, else seek stored
    # tokenizer for model
    if with_data:
        res["dataset"] = get_datamodule(res["params"]["data_params"],
                                        res["params"]["model_params"],
                                        verbose=verbose,
                                        keep_datamodule=keep_datamodule,
                                        given_tokenizer=tokenizer)
        tokenizer = res["dataset"].tokenizer
    else:
        res["dataset"] = None
        if None is tokenizer:
            tokenizer = load_stored_tokenizer_if_exists(
                res["params"]["model_params"].tokenizer_source_name,
                folder_name, verbose)

    assert None is not tokenizer, "no data, and didnt find saved tokenizer"

    # prepare model to be filled with saved parameters
    if "lm" not in res:
        res["lm"] = make_model(res["params"]["model_params"],
                               res["params"]["train_params"],
                               tokenizer)

    # fill model with saved params. don't know if this will affect the
    # passed in parameter lm, so will get it explicitly back from the trainer
    model_trainer = Trainer.load_from_checkpoint(
        path_join(folder_name, "model.model"), model=res["lm"],
        train_params=res["params"]["train_params"])

    res["lm"] = model_trainer.model  # in case it makes a copy or something
    res["lm"].eval()

    return res
