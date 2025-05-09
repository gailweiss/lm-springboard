import json
from misc.util import prepare_directory, convert_all_nested_lists_to_tuples
from misc.gpt2 import get_gpt2
from pathlib import Path
import os.path as path
from misc.create import make_model, make_datamodule
from train.trainer import Trainer
from train.train_params import make_tp
from model.model_params import make_mp
from model.tokenizer import load_stored_tokenizer_if_exists
from data.data_params import make_dp
from eval.eval_params import make_ep
import glob
from misc.util import printer_print as print
from dataclasses import asdict
import torch


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
               train_params, eval_params, just_stats=False):
    prepare_directory(folder_name)
    for p, n in [(model_params, "model_params"),
                 (data_params, "data_params"),
                 (train_params, "train_params"),
                 (eval_params, "eval_params")]:
        with open(path.join(folder_name, f"{n}.json"), "w") as f:
            json.dump(vars(p), f)
        # json will turn tuples into lists, which is annoying. but i expect
        # all configs to have only tuples if they have iterables at all,
        # so the dataclass loading function successfully reverts this
    with open(path.join(folder_name, "train_stats.json"), "w") as f:
        my_trainer.logged_stats_dict["total_train_samples"] = \
            my_trainer.n_train_samples  # this will be slightly more than
            # the last value in the stats_dict["train_samples"] log, as that
            # will have been logged just *before* the last batch was trained on
        json.dump(my_trainer.logged_stats_dict, f)
    if not just_stats:
        pl_trainer.model.model.tokenizer.save(folder_name)
        pl_trainer.save_checkpoint(path.join(folder_name, "model.model"))


def get_train_stats(folder_name, get_lite=True, store_lite=True,
                    batch_size=None):
    # need batch_size so long as still fixing older runs, eventually remove it
    def is_lite_key(k):
        for n in ["||max/", "||min/"]:
            if n in k:
                return False
        for n in ["-max", "-min"]:
            if k.endswith(n):
                return False
        if k in ["stat_syncer", "n_active_params", "avg_lr", "n_epochs"]:
            return False
        return True

    lite_file = path.join(folder_name, "train_stats_lite.json")
    main_file = path.join(folder_name, "train_stats.json")

    if get_lite:
        if path.exists(lite_file):
            with open(lite_file, "r") as f:
                return json.load(f)
            # no fixes needed to add total_train_samples to lite train_stats -
            # they will all be created from fixed loaded train_stats
        # else:
        ts = get_train_stats(folder_name, get_lite=False,
                             batch_size=batch_size)
        ts = {k: v for k, v in ts.items() if is_lite_key(k)}
        if store_lite:
            with open(lite_file, "w") as f:
                json.dump(ts, f)
        return ts
    # else:
    with open(path.join(folder_name, "train_stats.json"), "r") as f:
        ts = json.load(f)
        if "total_train_samples" not in ts:
            correctness_flag = True
            if (None is batch_size) or (batch_size <= 0):
                correctness_flag = False
                batch_size = 0
                # best i am willing to do for now (could probably infer from
                # other stats but dont want to)
            ts["total_train_samples"] = \
                ts.get("n_train_samples", [[0]])[-1][-1] + batch_size
            ts["total_train_samples_correct"] = correctness_flag
            print("total_train_samples computed correctly:", correctness_flag)
        # if not got, this is the model at time 0 (no training yet)
        return ts


def load_model_info(folder_name, with_train_stats=False, verbose=True,
                    get_lite=True, store_lite=True):
    if not Path(folder_name).exists():
        raise ValueError(f"Folder {folder_name} does not exist")

    res = {"params": {}}

    with open(path.join(folder_name, "model_params.json"), "r") as f:
        res["params"]["model_params"] = make_mp(verbose=verbose,
                                                **json.load(f))
    with open(path.join(folder_name, "data_params.json"), "r") as f:
        res["params"]["data_params"] = make_dp(verbose=verbose,
                                               **json.load(f))
    with open(path.join(folder_name, "train_params.json"), "r") as f:
        res["params"]["train_params"] = make_tp(verbose=verbose,
                                                **json.load(f))
    with open(path.join(folder_name, "eval_params.json"), "r") as f:
        res["params"]["eval_params"] = make_ep(verbose=verbose,
                                                **json.load(f))


    for pn, pd in res["params"].items():
        if None is pd:
            if verbose:
                print("failed to load params:", pn, 
                      "--they might be from a different branch.")
            continue
        for k, v in asdict(pd).items():
            setattr(pd, k, convert_all_nested_lists_to_tuples(v))
            # json turns tuples into lists, but saved configs (i.e. true
            # configs, not config 'lists') don't have lists - 
            # correct back to tuples

    if with_train_stats:
        batch_size = res["params"]["train_params"].batch_size if \
                     None is not res["params"]["train_params"] else -1
        res["train_stats"] = get_train_stats(folder_name, get_lite=get_lite,
                                             store_lite=store_lite,
                                             batch_size=batch_size)
    return res


def get_datamodule(data_params, model_params, verbose=True,
                   keep_datamodule=False, given_tokenizer=None):
    
    from misc.model_explorer import find_existing_datamodule
    data = find_existing_datamodule(data_params, model_params)

    if None is not data:
        return data

    return make_datamodule(data_params, model_params, verbose=verbose,
                           keep_datamodule=keep_datamodule,
                           given_tokenizer=given_tokenizer)


def load_model(folder_name, full=False, verbose=True, with_data=False,
               keep_datamodule=False):

    if not Path(folder_name).exists():
        raise ValueError(f"Folder {folder_name} does not exist")

    res = load_model_info(folder_name, verbose=verbose)

    res["params"]["train_params"].no_wandb = True  # no wandb in loaded models
    # - or it will try (unsuccessfully) to send validation losses to wandb when
    # doing a validation run

    # if working from a pretrained model, get model and tokenizer directly
    if res["params"]["model_params"].from_os_pretrained:
        if res["params"]["model_params"].from_os_pretrained == "gpt2":
            res["lm"] = get_gpt2(res["params"]["model_params"].max_seq_len)
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
        path.join(folder_name, "model.model"), model=res["lm"],
        train_params=res["params"]["train_params"])

    res["lm"] = model_trainer.model  # in case it makes a copy or something

    # move model to gpu before returning, else everything will
    # be done on cpu for no reason
    if torch.cuda.is_available():
        res["lm"].cuda()
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        res["lm"].to(mps_device)

    res["lm"].eval()

    return res
