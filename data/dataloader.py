from torch.utils.data import DataLoader
import lightning as pl
import torch
from copy import deepcopy
from dataclasses import dataclass
import datasets
from data.syntheticdata import syntheticdatasets
from os.path import join as path_join
from model.tokenizer import load_stored_tokenizer_if_exists
from model.model_params import ModelParams
import numpy as np
from util import prepare_directory, glob_nosquares
import json
from util import printer_print as print


try:
    with open("../data/", "r") as f:
        datapath = f.readlines()[0].strip("\n")
        # path to your local data folder,
        # e.g. /Users/yourname/Documents/mydata
except Exception as e:
    print("couldnt read datapath for local datasets")
    datapath = None


try:
    with open("datamodules-paths.txt", "r") as f:
        datamodules_paths = f.readlines()
        # e.g. ../dataloaders, or more complicated if using cloud services
        datamodules_paths = [l.strip("\n") for l in datamodules_paths if not
                             l.startswith("#")]
except Exception as e:
    print("couldnt find extra dataloader paths")
    datamodules_paths = ["../datamodules"]


@dataclass
class DataParams:
    dataset_name: str = "ptb"
    debug_crop: int = None
    breaking_synthetic_samples_ok: bool = False
    val_pct: int = 10
    test_pct: int = 5
    is_synthetic_task: bool = False  # for internal use
    lines_per_sample: int = 1
    max_seq_len: int = -1


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
# is_synthetic_task:
#   Value for internal use: any value set here will be ignored and
#   overwritten by get_data.
# lines_per_sample:
#   Relevant when loading a dataset from the local data folder: every text file
#   read will be broken into multiple samples, with lines_per_sample lines for
#   each sample. Default 1: 1 line per sample. If using -1: will not break up
#   the files.
# max_seq_len:
#   Maximum sequence length (in token) to make the data samples. <=0: matches
#   model max sequence length. Can be relevant when finetuning if have big
#   model but only want to finetune up to certain length


def set_synthetic_task_flag(data_params):
    data_params.is_synthetic_task = \
        syntheticdatasets.has_dataset(data_params.dataset_name)


def get_local_datafolder(n):
    if None is datapath:
        return None
    ps = glob_nosquares(f"{datapath}/*") +\
        glob_nosquares(f"{datapath}/*/*") +\
        glob_nosquares(f"{datapath}/*/*/*")
    dd = f"{datapath}/"
    return next((p for p in ps if dd.join(p.split(dd)[1:]) == n), None)


def get_data(data_params):
    set_synthetic_task_flag(data_params)
    if data_params.dataset_name == "dummy":
        samples = verysimplesamplesreader(".", data_params)
    elif data_params.dataset_name == "wikitext":
        samples = wikitextloader()
    elif data_params.dataset_name == "ptb":
        samples = ptbloader()
    elif data_params.is_synthetic_task:
        samples = syntheticdatasets.get(data_params.dataset_name)
    elif None is not get_local_datafolder(data_params.dataset_name):
        samples = verysimplesamplesreader(
                    get_local_datafolder(data_params.dataset_name),
                    data_params)
    else:
        raise Exception(f"unknown dataset: {data_params.dataset_name}")
    if data_params.debug_crop:
        data_params.debug_crop = int(data_params.debug_crop)
        if isinstance(samples, list):
            samples = samples[:data_params.debug_crop]
        else:
            samples = {n: samples[n][:data_params.debug_crop] for n in samples}
    return samples


def get_existing_datamodule(data_params, model_params):
    def is_match(dpd, mpd):
        # all the attrs that determine the dataset and the tokenizer
        important_model_attrs = ["max_seq_len", "tokenizer_source_name"]
        important_data_attrs = ["dataset_name", "debug_crop", 
                                "breaking_synthetic_samples_ok",
                                "val_pct", "test_pct", "lines_per_sample", 
                                "max_seq_len"]
        for a in important_data_attrs:
            if not dpd[a] == getattr(data_params, a):
                return False
        for a in important_model_attrs:
            if not mpd[a] == getattr(model_params, a):
                return False
        if model_params.tokenizer_source_name == "custom":
            if not (mpd["custom_tokenizer_ntokens"] == 
                    model_params.custom_tokenizer_ntokens):
                return False
        return True

    set_synthetic_task_flag(data_params)
    print("checking for existing datamodule")
    for p in datamodules_paths:
        print("checking inside path:", p)
        print("path contains:", glob_nosquares(f"{p}/*"))
        with_timestamps = glob_nosquares(f"{p}/{data_params.dataset_name}/*")
        for path in with_timestamps:
            print("checking path:", path)
            with open(path_join(path, "model_params.json"), "r") as f:
                mpd = json.load(f)
            with open(path_join(path, "data_params.json"), "r") as f:
                dpd = json.load(f)
            with open(path_join(path, f"dataloader_notes.json"), "r") as f:
                notes = json.load(f)

            if is_match(dpd, mpd):
                print("matched!")
                return LMDataModule(None, None, None, None, from_folder=path)
    print("no match!")
    return None


class ForTorchDataSet:
    def __init__(self, lengths, indices):
        self.lengths = lengths
        self.indices = torch.as_tensor(indices)

    def __getitem__(self, i):
        return self.lengths[i], self.indices[i]

    def __len__(self):
        return len(self.lengths)


class LMDataModule(pl.LightningDataModule):
    def __init__(self, data, tokenizer, data_params, model_params,
                 verbose_init=False, from_folder=None):
        super().__init__()
        self.verbose_init = verbose_init
        if None is not from_folder:
            self.setup_from_folder(from_folder)
            return
        self.data_params = data_params
        self.model_params = model_params
        self.tokenizer = tokenizer
        self.set_max_seq_len()

        if isinstance(data, list):
            self.setup_from_list(data)
        else:  # DatasetDict through huggingface
            self.setup_from_data_dict(data)
        self.prep_for_torch_datasets()

    def set_max_seq_len(self):
        self.max_seq_len = self.model_params.max_seq_len
        if self.data_params.max_seq_len > 0:
            self.max_seq_len = min(self.max_seq_len,
                                   self.data_params.max_seq_len)

    def setup_from_folder(self, path):  
        with open(path_join(path, "model_params.json"), "r") as f:
            self.model_params = ModelParams()
            mp = json.load(f)
            [setattr(self.model_params, a, mp[a]) for a in mp]
            # allows that mp may specify some additional things not in current
            # model_params definition, if made by more specific branch. assumes
            # these additional things only add information, but do not take
            # away or modify anything needed here
        with open(path_join(path, "data_params.json"), "r") as f:
            self.data_params = DataParams()
            dp = json.load(f)
            [setattr(self.data_params, a, dp[a]) for a in dp]
            # as with model_params above
        self.tokenizer = load_stored_tokenizer_if_exists(
            self.model_params.tokenizer_source_name, path, self.verbose_init)
        assert None is not self.tokenizer  
        # can't be loading tokenized data without its tokenizer
        self.set_max_seq_len()

        with open(path_join(path, "dataloader_notes.json"), "r") as f:
            base_attrs = json.load(f)
        [setattr(self, an, base_attrs[an]) for an in base_attrs]

        # load train, test, val samples
        for sn in ["train_samples", "test_samples", "val_samples"]:
            indices = np.load(path_join(path, f"{sn}-indices.npy"))
            lengths = np.load(path_join(path, f"{sn}-lengths.npy"))
            setattr(self, sn, ForTorchDataSet(lengths, indices))

        self.finalise_data()

    def prep_for_torch_datasets(self):
        def arranged(samples):
            lengths = np.array([len(s) for s in samples])
            indices = np.zeros((len(samples), lengths.max()), dtype=int)
            for i, (n, inds) in enumerate(zip(lengths, samples)):
                indices[i, :n] = inds
            return lengths, indices

        for sn in ["train_samples", "test_samples", "val_samples"]:
            lengths, indices = arranged(getattr(self, sn))
            setattr(self, sn, ForTorchDataSet(lengths, indices))

    def save_to_folder(self, path):
        prepare_directory(path)
        # save tokenizer, save self
        self.tokenizer.save(path)
        base_attr_names = ["train_n", "test_n", "val_n"]
        base_attrs = {n: getattr(self,n) for n in base_attr_names}
        with open(path_join(path, f"dataloader_notes.json"), "w") as f:
            json.dump(base_attrs, f)
        with open(path_join(path, f"model_params.json"), "w") as f:
            json.dump(vars(self.model_params), f)
        with open(path_join(path, f"data_params.json"), "w") as f:
            json.dump(vars(self.data_params), f)

        for sn in ["train_samples", "test_samples", "val_samples"]:
            ds = getattr(self, sn)
            for a in ["lengths", "indices"]:
                np.save(path_join(path, f"{sn}-{a}.npy"), getattr(ds, a), 
                        allow_pickle=False)

    def setup_from_data_dict(self, samples):
        train_samples = samples["train"]
        val_samples = samples["validation"]
        test_samples = samples["test"]
        train_n = len(train_samples)
        val_n = len(val_samples)
        test_n = len(test_samples)
        self.setup_from_list(train_samples + val_samples + test_samples,
                             (train_n, val_n, test_n),
                             force_no_split_shuffle=True)

    def compute_n_full_samples_per_set(self, n_samples, sizes):
        if None is not sizes:
            train_n, val_n, test_n = sizes
        else:
            test_n = max(
                int(self.data_params.test_pct * n_samples / 100), 1)
            val_n = max(
                int(self.data_params.val_pct * n_samples / 100), 1)
            train_n = n_samples - (test_n + val_n)
        cond = (test_n > 0) and (val_n > 0) and (train_n > 0)
        assert cond, f"lengths:{test_n},{val_n},{train_n}"
        return train_n, val_n, test_n

    def chunk_long_samples(self, tokenized_data):
        # breaks long samples into samples of length up to
        # self.max_seq_len, so len(res) >= len(tokenized_data)
        res = []
        for s in tokenized_data:
            if (self.data_params.is_synthetic_task and
               not self.data_params.breaking_synthetic_samples_ok):
                msg = f"got synthetic sample longer than allowed: {len(s)}" +\
                      f" tokens (max allowed: {self.max_seq_len})"
                assert len(s) <= self.max_seq_len, msg

            for i in range(0, len(s), self.max_seq_len):
                def get_chunk(lst):
                    return lst[i:i + self.max_seq_len + 1]
                    # +1 to have max_len + 1 tokens, as the first max_len are
                    # input and the last max_len are prediction
                schunk = get_chunk(s)

                # need at least 1 token and its next token
                # to do LM training
                if len(schunk) > 1:
                    res.append(schunk)
        return res

    def print_data_desc(self, overall_list=None):
        def descstr(lst, n):
            lengths = [len(s) for s in lst]
            return f"{n}: {len(lst)} samples, avg: " +\
                   f"{sum(lengths)/len(lengths)} tokens" +\
                   f", max: {max(lengths)} tokens"

        if None is not overall_list:
            print(descstr(overall_list, "overall data"))
        
        named_lists = [(self.train_samples, "train"), (self.val_samples, "val"),
                        (self.test_samples, "test")]
        for lst, n in named_lists:
            print("\t", descstr(lst, n))

    def setup_from_list(self, samples, sizes=None,
                        force_no_split_shuffle=False):
        # force_no_split_shuffle: no shuffle overrides a shuffle before the
        # dataset split, in case i ever want to add one
        n = len(samples)
        data = self.tokenizer(samples)
        train_n, val_n, test_n = self.compute_n_full_samples_per_set(
                                        len(samples), sizes)

        # split without shuffling always for now, and beware of
        # force_no_split_shuffle if want to change later
        self.train_samples = data[:train_n]
        self.val_samples = data[train_n: train_n + val_n]
        self.test_samples = data[train_n + val_n:]
        if self.verbose_init:
            print("=== before chunking ===")
            self.print_data_desc(overall_list=data)

        self.train_samples = self.chunk_long_samples(self.train_samples)
        self.val_samples = self.chunk_long_samples(self.val_samples)
        self.test_samples = self.chunk_long_samples(self.test_samples)

        if self.verbose_init:
            print("=== after chunking ===")
            self.print_data_desc()

        self.finalise_data()

    def finalise_data(self):
        # for showing samples
        self.train_n = len(self.train_samples)
        self.val_n = len(self.val_samples)
        self.test_n = len(self.test_samples)

    def print_dataset_lengths(self):
        print(f"train: {self.train_n}, val: {self.val_n}, test: {self.test_n}")

    def generic_loader(self, d, batch_size, shuffle=False):
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=mycollate)  # ,num_workers=7)
        # >1 workers maybe gives speedup but it seems to me primarily causes
        # problems - latest is complaining of "[Errno 24] Too many open files"
        # when running on my mac for too many epochs (batches dont bother it
        # - epochs do). i dont need this

    def get_sample(self, i):
        datasets = [self.train_samples, self.val_samples, self.test_samples]
        for ds in datasets:
            if i < len(ds):
                n, indices = ds[i]  # length, indices
                return n, indices[:n]
            i -= len(ds)
        n = sum([len(ds) for ds in datasets])
        raise Exception(f"no sample at index {i}, only have {n} samples")

    def get_sample_str(self, i=0):
        n, indices = self.get_sample(i)
        return self.tokenizer.convert_ids_to_nice_string(indices)

    def show_sample(self, i=0):
        n, indices = self.get_sample(i)
        print(f"sample {i}, has {n} tokens:")
        print(self.get_sample_str(i))

    def train_dataloader(self, batch_size):
        return self.generic_loader(self.train_samples, batch_size,
                                   shuffle=True)

    def val_dataloader(self, batch_size):
        return self.generic_loader(self.val_samples, batch_size)

    def test_dataloader(self, batch_size):
        return self.generic_loader(self.test_samples, batch_size)

    def predict_dataloader(self):
        return None  # what are they on about

    def teardown(self, stage: str):
        pass
        # Used to clean-up when the run is finished


def ptbloader():
    d = datasets.load_dataset("ptb_text_only")
    d = {n: d[n]["sentence"] for n in d}
    return d  # see if this works


def wikitextloader():
    d = datasets.load_dataset("wikitext", "wikitext-103-v1")
    d = {n: d[n]["text"] for n in d}

    def regroup_page_lines(lines):
        res = []
        curr = None

        def is_title_line(line):
            return (line.count("=") == 2) and \
                    line.strip().startswith("=") and \
                    line.strip().endswith("=")
        for line in lines:
            if is_title_line(line):
                if None is not curr:
                    res.append(curr)
                curr = line
            elif None is curr:
                continue
            else:
                curr += line
        res.append(curr)
        return res

    return {n: regroup_page_lines(d[n]) for n in d}


def verysimplesamplesreader(path, data_params):
    paths = glob_nosquares(f"{path}/*.txt")
    all_samples = []
    for p in paths:
        with open(p, "r") as f:
            all_lines = f.readlines()
        if data_params.lines_per_sample < 0:
            all_samples.append("".join(all_lines))
        else:
            for i in range(0, len(all_lines), data_params.lines_per_sample):
                all_samples.append("".join(
                    all_lines[i: i + data_params.lines_per_sample]))
    return all_samples


def mycollate(b):
    # b: list of samples. each sample is a tuple of:
    # (int length, tensor of indices).
    dtype = torch.long
    lengths = [s[0] for s in b]
    seqlen = max(lengths)

    batch_indices = torch.zeros((len(b), seqlen), dtype=dtype)

    with_mask = len(set(lengths)) > 1
    mask = torch.ones(batch_indices.shape, dtype=dtype) if with_mask else None

    for i, (n, seq) in enumerate(b):
        batch_indices[i][:n] = seq[:n]
        if with_mask:
            mask[i][:n] = 0

    # indices shape: batch size X seq len
    # mask shape: batch size X seq len, or None. marks pads
    return {"x_indices": batch_indices, "mask": mask}
