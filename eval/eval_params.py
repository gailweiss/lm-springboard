from dataclasses import dataclass
from misc.util import apply_dataclass


@dataclass
class EvalParams:
    val_check_epoch_frac: float = 0.1
    max_sample_tokens: int = 30
    sample_temperature: float = 1
    no_wandb: bool = False
    slower_log_freq: int = 10
    checkpoint_every: int = 0
    early_stop_nsamples: int = -1
    weight_decay: float = 0.01
    random_seed: int = None
    extra_tracking: bool = False
    track_hellaswag: bool = True
    track_sepsquad: bool = False


def make_ep(forgiving=False, takes_extras=False, convert_lists_to_tuples=False,
            verbose=True, **d):
    name_changes = [("hyperparams_log_freq", "slower_log_freq")]
    return apply_dataclass(EvalParams, d, forgiving=forgiving,
                           convert_lists_to_tuples=convert_lists_to_tuples,
                           verbose=verbose, takes_extras=takes_extras,
                           name_changes=name_changes)
    # ready for fixes over time


# val_check_epoch_frac:
#   How often to compute the validation loss during training. If float, treated
#   as fraction of train data trained between each validation check. If int,
#   treated as number of train batches trained between each validation check.
#   Default 1.0 (float) - not the same as 1 (int)! See the docs for
#   val_check_interval in pytorch_lightning
# max_sample_tokens:
#   Yhe maximum length of the sample that will be generated and shown after
#   every validation loop
# sample_temperature:
#   The sampling temperature for the samples that will be generated and shown
#   after every validation loop
# no_wandb:
#   Optionally turn off wandb logging. wandb only runs if this is False and
#   also --no-wandb has not been passed to the main.py call
# slower_log_freq:
#   If wandb is on, how often to record some additional statistics of the
#   training, e.g. the number of trainable model parameters, or "extra"
#   tracked info (see extra_tracking)
# checkpoint_every:
#   How often to save a checkpoint of the model. 0: don't. -1: every validation
#   step. n>0: every n training samples (or closest approximation with
#   available batch size, specifically: save every time n_samples have
#   increased by >=n since last save).
#   Relevant only when also passing --save through the args, otherwise will be
#   overridden to 0.
# early_stop_nsamples:
#   When >0, the number of batches to train before early stopping the training.
# weight_decay:
#   Weight decay to apply to the optimizer.
# random_seed:
#   Random seed for reproducibility. This value can also be overwritten by
#   setting the random seed arg in the main.py script.
# extra_tracking:
#   Track some additional statistics on how training is going, in particular:
#   weight norms, gradient norms, step norms. Will only be logged every
#   slower_log_freq batches
# track_hellaswag, track_sepsquad:
#   Track hellaswag/sepsquad (as relevant) performance at all validation steps