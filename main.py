import torch
import torch.nn as nn
from data.data_params import make_dp, DataParams
from model.model_params import make_mp, ModelParams
from model.lm import LM
from train.trainer import Trainer
from train.train_params import make_tp, TrainParams
import lightning as pl
import argparse
from dataclasses import asdict
import wandb
from util import get_probably_unique, print_nicely_nested, in_try, \
                 prepare_directory, printer, glob_nosquares
from util import printer_print as print
from save_load import save_model as save_model_
from save_load import final_chkpt
from setup import setup_model_and_data
import shutil
from copy import deepcopy
import ast
from time import process_time, sleep
import os
from os.path import join as path_join
import sys
import numpy as np
import random

# not trying to parallelize the tokenizer, i tokenize everything first and then
# load tokens (not sequences) later. (may want to change this in future, if
# move to training on very big data)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="debug")
parser.add_argument('--task', type=str, default=None)  # e.g. copy, wikitext
parser.add_argument('--wandb-proj-name', type=str, default=None)
parser.add_argument('--save', action='store_true')  # save all including model
parser.add_argument('--save-stats', action='store_true')  # save all but model
# no ablations exist in the base code, but this arg is ready for adding them:
parser.add_argument('--ablate', action='store_true')
# allow blocking wandb from command (but can also do it from config file):
parser.add_argument('--no-wandb', action='store_true')
parser.add_argument('--gpu-id', type=int, default=None)
# for internal debug use (can be passed to get_exception at the bottom here):
parser.add_argument('--return-things', type=bool, default=False)
parser.add_argument('--keep-datamodule', action='store_true')
parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility')


MAIN_PROJ = "base"  # project name for wandb runs
wandb_username = "gail_weiss"


class Namer:
    def __init__(self, args):
        self.args = args

    def set_config_index(self, index):
        self.index = index  # note it might be a tuple

    def set_config_ablation(self, ablation):
        self.ablation = ablation
        # ignored in base code, but may be useful in naming runs and save
        # folders when ablations are added

    def set_config(self, dp, tp, mp):
        self.dp, self.tp, self.mp = dp, tp, mp
        self.identifier = get_probably_unique()

    def wandb_proj_name(self):
        specific = self.args.config if None is self.args.wandb_proj_name \
                    else self.args.wandb_proj_name
        return f"{MAIN_PROJ}-{specific}-{self.dp.dataset_name}"

    def run_name(self):
        model_str = f"L:[{self.mp.n_layers}]-D:[{self.mp.dim}]" +\
                    f"-H:[{self.mp.n_heads}]"
        return None  # until want to overwrite it

    def save_folder_name(self, given_run_name):
        # given run name might have come from wandb so need to receive it
        wn = "" if None is given_run_name else f"{given_run_name}/"
        return f"{self.args.config}/{self.dp.dataset_name}/" +\
               f"{wn}{self.identifier}"


def seed_everything(args_seed, tp):
    seed = args_seed
    if None is seed:
        seed = tp.random_seed # i.e., args_seed is stronger than config, if set
    if None is seed: 
        seed = random.randint(0,2**32 -1 )

    pl.seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def build_full(dp, tp, mp):
    full = {}
    for params, name in [(dp, "dp"), (tp, "tp"), (mp, "mp")]:
        for k, v in asdict(params).items():
            full[name + "." + k] = v
    return full


def read_config(config_filename):
    with open(config_filename, "r") as f:
        lines = [line.split("#")[0].strip() for line in f.readlines()]
        # drop comments, remove edge whitespace (doesn't seem to remove
        # internal whitespace though)
    res = {n: {} for n in ["DataParams", "TrainParams", "ModelParams"]}
    curr_set = ""  # shouldnt get used
    for line in lines:
        if not line:
            continue
        if "=" not in line:
            curr_set = line
        else:
            assert line.count("=") == 1, f"confusing config line: {line}"
            key, val = line.split("=")
            key = key.strip()
            val = ast.literal_eval(val.strip())
            res[curr_set][key] = val
    return res


def get_params(config_filename):
    overwrites = read_config(config_filename)
    dp = make_dp(**overwrites["DataParams"], convert_lists_to_tuples=False)
    tp = make_tp(**overwrites["TrainParams"], convert_lists_to_tuples=False)
    mp = make_mp(**overwrites["ModelParams"], convert_lists_to_tuples=False)
    assert None not in [dp, tp, mp]
    return dp, tp, mp


def train(args, lm, dataset, tp, dp, saving_folder):
    # dp and saving_folder are for saving checkpoints
    tokenizer = lm.tokenizer
    start_time = process_time()
    pltrainer = pl.Trainer(
        logger=False, enable_checkpointing=False,
        # logging and checkpointing off to not make infinite log and checkpoint
        # files, which i dont want
        devices=1 if args.gpu_id is None else [args.gpu_id],  # only run on 1
        # device, else it runs all of main.py n_devices times (????).
        # presumably its for multi-gpu training but i haven't learned how yet
        max_epochs=tp.epochs, val_check_interval=tp.val_check_epoch_frac)

    mytrainer = Trainer(lm, tp, start_time=start_time)
    mytrainer.prepare_saver(dp, saving_folder, save_model_)

    pltrainer.fit(mytrainer, dataset.train_dataloader(tp.batch_size),
                  dataset.val_dataloader(tp.batch_size))
    pltrainer.validate(mytrainer,
                       dataloaders=dataset.val_dataloader(tp.batch_size))
    # when the val_check_interval does not neatly divide 1, pytorch lightning
    # might not run a validation at the end of the last epoch, which will mess
    # up my saved stats and so make it hard to check a loaded model is behaving
    # as expected. so, explicitly run a final validation once training is done.

    return pltrainer


def setup_wandb(args, tp, full_params, namer):
    if not (tp.no_wandb or args.no_wandb):
        run = wandb.init(entity=wandb_username,
                         project=namer.wandb_proj_name(),
                         config=full_params, name=namer.run_name(), dir="..")
        wandb.define_metric("n_train_samples")
        wandb.define_metric("*", step_metric="n_train_samples")
        wandb.log({"n_train_samples": 0})
        run_name = run.name  # if namer sends nothing then wandb makes one
        run_loc = run.dir
        assert run_loc.endswith("/files")
        run_loc = run_loc[:-len("/files")]
    else:
        run, run_loc = None, None
        run_name = namer.run_name()
    return run, run_name, run_loc


def finish_wandb(args, tp, run, run_loc):
    if not (tp.no_wandb or args.no_wandb):
        run.finish()
        current_year = "2024"
        # honestly fine with failing this once a year just to be sure this
        # delete is still fine
        assert run_loc.split("/")[-1].startswith(f'run-{current_year}')
        sleep(10)  # give wandb 10 seconds to actually finish, this is
        # stupid but ugh i guess
        try:
            shutil.rmtree(run_loc)
        except Exception as e:
            print("couldnt delete wandb log at:", run_loc,
                " -- got exception:\n", e)


def show_sample(lm):
    sample = lm.sample(max_seq_len=50, temperature=0.5)
    try:
        print(sample)
    except Exception as e:
        print("could not print this sample - got exception:\n", e)


def save_model(args, saving_folder, pltrainer, dp, tp):
    mytrainer = pltrainer.model
    lm = mytrainer.model
    if args.save or args.save_stats:
        fn = f"{saving_folder}/{final_chkpt}"
        # make sure to use the updated model params after all this
        save_model_(fn, pltrainer, mytrainer, lm.model_params, dp, tp,
                    just_stats=not args.save)
        print(f"saved model {'' if args.save else ' stats'} in:\n", fn)


def run_config(args, dp, tp, mp, namer):
    tp = deepcopy(tp)
    tp.random_seed = seed_everything(args.random_seed, tp)
    full_params = build_full(dp, tp, mp)
    run, run_name, run_loc = setup_wandb(args, tp, full_params, namer)
    saving_folder = f"../saved-models/{namer.save_folder_name(run_name)}"
    if args.save or args.save_stats:
        prepare_directory(saving_folder)
        f = open(path_join(saving_folder, "log.txt"), "w")
        printer.add_output_file(f)

    print("going to train from config: [", args.config,
                  "], using the following parameters:")
    print_nicely_nested(full_params)
    lm, dataset = setup_model_and_data(dp, mp, tp, 
                                      keep_datamodule=args.keep_datamodule)
    
    pltrainer = train(args, lm, dataset, tp, dp, saving_folder)
    show_sample(pltrainer.model.model)
    save_model(args, saving_folder, pltrainer, dp, tp)
    finish_wandb(args, tp, run, run_loc)

    if args.save or args.save_stats:
        f.close()
        printer.remove_output_file(f)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def all_config_variants(params):
    main_config_dict = asdict(params)
    for param_name in main_config_dict:
        if isinstance(main_config_dict[param_name], list):
            res = []
            for v in main_config_dict[param_name]:
                a = deepcopy(params)
                setattr(a, param_name, v)
                res += all_config_variants(a)
            return res
    return [params]


def get_config_filenames(config_name):
    res = glob_nosquares(f"configs/{config_name}-*.txt") +\
           glob_nosquares(f"configs/{config_name}.txt")
    return sorted(res)
    # sorted: in case numbered them and want it to run in that order.
    # note that it will go in lexicographic order so 10 will run
    # before 5 unless its listed 05


def run_all(args, dp, tp, mp, namer):
    # dp, tp, mp have been manipulated on the way here, remake to be sure
    dp = make_dp(**asdict(dp), redo_synth_eval=True)
    tp = make_tp(**asdict(tp))
    mp = make_mp(**asdict(mp))
    namer.set_config_ablation("main")
    namer.set_config(dp, tp, mp)
    if args.no_wandb:
        tp.no_wandb = True  # else will have bugs
    run_config(args, dp, tp, mp, namer)
    if args.ablate:
        pass  # add ablations here


def adjust_args(args):
    # ready for any fixes that might be necessary if you have args that can
    # conflict
    pass


def get_args(arg_bits_list):
    # will take from call if arg_bits_list is None
    args = parser.parse_args(arg_bits_list)
    adjust_args(args)
    return args


def run_main(arg_bits_list=None):
    torch.autograd.set_detect_anomaly(True)
    args = get_args(arg_bits_list)
    namer = Namer(args)
    print("got config name:", args.config)
    config_index = 0
    run_all_args = []
    for filename in get_config_filenames(args.config):
        dp, tp, mp = get_params(filename)
        if None is not args.task:
            dp.dataset_name = args.task
        if not args.save:
            tp.checkpoint_every = 0
        for mpv in all_config_variants(mp):
            for tpv in all_config_variants(tp):
                namer.set_config_index(config_index)
                run_all(args, dp, tpv, mpv, namer)
                config_index += 1


# eg ['--config','debug','--return-things']
def get_exception(arg_bits_list=['--config', 'debug']):
    try:
        run_main(arg_bits_list)
    except Exception as e:
        return e


if __name__ == "__main__":
    run_main()
