from dataclasses import dataclass
from misc.util import apply_dataclass


@dataclass
class TrainParams:
    batch_size: int = 8
    accumulate_grad_batches: int = 1
    epochs: int = 5
    dropout: float = 0.1
    gradient_clip_val: float = 0.5
    lr: float = 1e-3
    lr_warm_steps: int = 50
    lr_scheduler_type: str = 'Cosine'
    min_lr: float = 1e-7
    scheduler_factor: float = 0.995
    patience: int = 10
    lr_cycle_steps: int = 500
    lora_rank: int = 0
    lora_std: float = 0.02
    early_stop_nsamples: int = -1
    weight_decay: float = 0.01
    random_seed: int = None


def make_tp(forgiving=False, takes_extras=False, convert_lists_to_tuples=False,
            verbose=True, **d):
    return apply_dataclass(TrainParams, d, forgiving=forgiving,
                           convert_lists_to_tuples=convert_lists_to_tuples,
                           verbose=verbose, takes_extras=takes_extras)
    # ready for fixes over time


# batch_size:
#   The batch size used for the training and validation sets
# accumulate_grad_batches:
#   Number of batch gradients to accumulate before every optimizer step, passed
#   to accumulate_grad_batches argument in the pytorch lightning trainer
# epochs:
#   Number of times to train over the full train set. Passed to max_epochs
#   argument in the pytorch lightning trainer
# dropout:
#   The dropout applied to the model during training. Applied specifically to
#   transformerencoderlayers as defined in the default torch implementation.
#   not applied to other parts of the model, e.g. the decoder head.
# gradient_clip_val:
#   Maximum value for gradients during training - greater values are clipped to
#   this.
# lr:
#   Defines the 'main' learning rate, see different scheduler types for
#   details.
# lr_warm_steps:
#   Number of batches for which to warm up the learning rate before moving to
#   the main scheduler. In this phase, the learning rate will increase linearly
#   from 0 to lr
# scheduler_type:
#   The main scheduler that will be used to control the learning rate. Options:
#       "Cosine":
#           Uses torch.optim.lr_scheduler.CosineAnnealingLR, will start from
#           lr and cycle in a nice wave between lr and min_lr. Will cycle every
#           lr_cycle_steps batches.
#       "Cyclic":
#           Uses torch.optim.lr_scheduler.CyclicLR. Like Cosine, but the cycle
#           between high and low lr values is a direct interpolation
#       "Plateau":
#           Uses torch.optim.lr_scheduler.ReduceLROnPlateau. Starts from
#           learning rate lr and reduces the learning rate by scheduler_factor
#           every time the train batch loss ceases improving for patience
#           steps, reducing the learning rate only up to min_lr.
#       "Linear":
#           Uses torch.optim.lr_scheduler.LinearLR. After the warmup,
#           decreases the learning rate linearly at each training step,
#           such that it reaches min_lr exactly at the end of training
# min_lr:
#   The minimum learning rate the scheduler may set in training
# scheduler_factor:
#   Relevant only for "Plateau" scheduler, used as described above
# patience:
#   Relevant only for "Plateau" scheduler, used as described above
# lr_cycle_steps:
#   Relevant only for "Cosine" and "Cyclic" schedulers, used as described above
# lora_rank:
#   Rank of the low-rank adaptation applied when fine tuning. Only applied if
#   lora_rank>0, in this case, will only train the low-rank adaptation.
# lora_std:
#   Standard deviation for initialisation of the low-rank adaptation values.
# early_stop_nsamples:
#   When >0, the number of batches to train before early stopping the training.
# weight_decay:
#   Weight decay to apply to the optimizer.
# random_seed:
#   Random seed for reproducibility. This value can also be overwritten by
#   setting the random seed arg in the main.py script.
