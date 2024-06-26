import json
from util import prepare_directory
from pathlib import Path
from os.path import join as path_join
from create import make_model_and_data
from train.trainer import Trainer
from train.train_params import TrainParams
from model.model_params import ModelParams
from data.dataloader import DataParams
from model.tokenizer import load_stored_tokenizer_if_exists


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


def load_model(folder_name, full=False, verbose=True,
               with_data=False, known_tokenizer=None):
    # folder_name = path_join("../artifacts/models", folder_name)
    # not messing with folder paths anymore, i'll get the full path myself...
    # only pass tokenizer if you know the correct one, it is only to save time
    # by removing the need for this function to make/load it itself!
    if Path(folder_name).exists() is False:
        raise ValueError(f"Folder {folder_name} does not exist")

    with open(path_join(folder_name, "model_params.json"), "r") as f:
        model_params = ModelParams(**json.load(f))
    with open(path_join(folder_name, "data_params.json"), "r") as f:
        data_params = DataParams(**json.load(f))
    with open(path_join(folder_name, "train_params.json"), "r") as f:
        train_params = TrainParams(**json.load(f))

    with open(path_join(folder_name, "train_stats.json"), "r") as f:
        train_stats = json.load(f)

    tokenizer = known_tokenizer
    if None is tokenizer:
        tokenizer = load_stored_tokenizer_if_exists(
            model_params.tokenizer_source_name, folder_name, verbose)

    train_params.no_wandb = True  # no wandb in loaded models - or it will
    # try (unsuccessfully) to send validation losses to wandb when doing a
    # validation run
    lm, dataset = make_model_and_data(data_params, model_params, train_params,
                                      tokenizer=tokenizer, verbose=verbose,
                                      skip_data=not with_data)
    # if with_data==False, will receive None for dataset

    model_trainer = Trainer.load_from_checkpoint(
                                path_join(folder_name, "model.model"),
                                model=lm, train_params=train_params)

    lm = model_trainer.model
    lm.eval()  # return with training off, im basically never trying to retrain
    # a loaded model, at least for now
    if full:
        return (lm, dataset, train_stats, model_params, data_params,
                train_params)
    else:
        return lm, dataset
