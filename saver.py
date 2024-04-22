import json
from util import prepare_directory
from pathlib import Path
from os.path import join as path_join
from create import make_model_and_data
from model.lmtrainer import LMTrainer
from model.model_params import ModelParams
from model.train_params import TrainParams
from data.dataloader import DataParams
from model.tokenizer import MyTokenizer


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
    pl_trainer.model.lm.tokenizer.save(folder_name)
    pl_trainer.save_checkpoint(path_join(folder_name, "model.model"))


def load_model(folder_name, full=False, verbose=True):
    # folder_name = path_join("../artifacts/models", folder_name)
    # not messing with folder paths anymore, i'll get the full path myself...
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

    if model_params.tokenizer_source_name == "custom":
        # then need to load the tokenizer
        tokenizer = MyTokenizer(name=model_params.tokenizer_source_name,
                                from_path=folder_name, verbose_init=verbose)
    else:
        tokenizer = None

    train_params.no_wandb = True  # no wandb in loaded models - or it will
    # try (unsuccessfully) to send validation losses to wandb when doing a 
    # validation run
    lm, dataset = make_model_and_data(data_params, model_params, train_params,
                                      tokenizer=tokenizer, verbose=verbose)

    model_trainer = LMTrainer.load_from_checkpoint(
                                path_join(folder_name, "model.model"),
                                lm=lm, train_params=train_params)

    lm = model_trainer.lm
    lm.eval()  # return with training off, im basically never trying to retrain
    # a loaded model, at least for now
    if full:
        return (lm, dataset, train_stats, model_params, data_params,
                train_params)
    else:
        return lm, dataset
