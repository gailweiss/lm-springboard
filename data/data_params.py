from dataclasses import dataclass
from data.syntheticdata import syntheticdatasets
from misc.util import apply_dataclass


@dataclass
class DataParams:
    dataset_name: str = "ptb"
    langs: tuple = ("en", "fra_Latn", "roh_Latn", "deu_Latn", "ita_Latn")
    debug_crop: int = None
    breaking_synthetic_samples_ok: bool = False
    val_pct: int = 10
    test_pct: int = 5
    lines_per_sample: int = 1
    max_seq_len: int = -1
    task_type: str = "?"  # for internal use
    total_train_samples: int = -1  # set internally, for later analysis
    total_samples: int = -1  # set internally, for later analysis


def make_dp(forgiving=False, takes_extras=False, redo_synth_eval=False,
            convert_lists_to_tuples=False, verbose=True, **d):
    # correct old data_params to new attr:
    synth_task_note = "is_synthetic_task"
    if synth_task_note in d:
        d["task_type"] = "synthetic" if d[synth_task_note] else "?"
        del d[synth_task_note]

    if None is not d.get("debug_crop", None):
        d["debug_crop"] = int(d["debug_crop"])

    if "langs" in d:
        d["langs"] = tuple(sorted(list(d["langs"])))
        # canonise for consistent lookup of language combinations

    res = apply_dataclass(DataParams, d, forgiving=forgiving,
                          convert_lists_to_tuples=convert_lists_to_tuples,
                          verbose=verbose, takes_extras=takes_extras)

    if redo_synth_eval or d.get("task_type", "?") == "?":
        res.task_type = synthetic_task_flag(res)
    return res
    # ready for fixes over time


def synthetic_task_flag(data_params):
    if syntheticdatasets.has_dataset(data_params.dataset_name):
        return "synthetic"
    if "fineweb" in data_params.dataset_name and len(data_params.langs) > 1:
        return "multilingual natural"
    return "plain natural"


# dataset_name:
#   Which dataset to load and train on. Current options:
#       'ptb' (Penn Treebank, loads from Huggingface),
#       'wikitext' (Wikitext-103, also loads from Huggingface),
#       {synthetic task} - a task defined and registered in
#           data/syntheticdata.py, e.g. copy
#   You may also add a custom sample set on your own machine as follows:
#   Set some folder on your machine as your local data folder and write its
#   full path in ../../data-path.txt relative to this repository. Store your
#   custom samples, one line per sample, in a file data.txt and place that file
#   in a subfolder {folder-name} of your data folder. You can then set
#   dataset-name={folder-name}, and this code will read it with the function
#   verysimplesamplesreader defined below.
# langs:
#   Which languages to use if loading from a multilingual dataset. Irrelevant
#   for other datasets.
# debug_crop:
#   Number of samples to crop the dataset to, to be used when just doing a
#   quick run to make sure the code doesn't break. The crop is applied before
#   the train/val/test split
# breaking_synthetic_samples_ok:
#   Transformer models - at least transformers taking absolute positional
#   embeddings - can only accept sequences up to a bounded length. The
#   dataloaders here will break data up into chunks of length up to the
#   model's maximum input length. For synthetic tasks, this might be a
#   surprising and unwanted behaviour. So long as this flag is not set, the
#   code here will default to failing if this happens.
# val_pct, test_pct:
#   The % of the total data samples that will go into the validation and test
#   sets, respectively. Should be greater than 0 and sum to less than 100.
#   The splitting of samples to train, val, and test sets according to these
#   fractions happens *before* the samples are broken into smaller chunks
#   according to the model's maximum input length. Relevant only when data has
#   not already been split into train/test/val by some other convention, e.g.
#   by loading an already split dataset such as wikitext-103 from huggingface
# task_type:
#   Value for internal use: leave unset and code will figure it out
# lines_per_sample:
#   Relevant when loading a dataset from the local data folder: every text file
#   read will be broken into multiple samples, with lines_per_sample lines for
#   each sample. Default 1: 1 line per sample. If using -1: will not break up
#   the files.
# max_seq_len:
#   Maximum sequence length (in token) to make the data samples. <=0: matches
#   model max sequence length. Can be relevant when finetuning if have big
#   model but only want to finetune up to certain length
