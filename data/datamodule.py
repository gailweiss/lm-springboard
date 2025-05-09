from torch.utils.data import DataLoader
import lightning as pl
import torch
from data.syntheticdata import SyntheticSamplesIterator
from data.data_params import make_dp
from os.path import join as path_join
from model.tokenizer import load_stored_tokenizer_if_exists
from model.model_params import make_mp
import numpy as np
from misc.util import prepare_directory
import json
from misc.util import printer_print as print


try:
    with open("paths/datamodules-paths.txt", "r") as f:
        datamodules_paths = f.readlines()
        # e.g. ../dataloaders, or more complicated if using cloud services
        datamodules_paths = [p.strip("\n") for p in datamodules_paths if not
                             p.startswith("#")]
except Exception as e:
    print("couldnt find extra dataloader paths")
    datamodules_paths = ["../datamodules"]


class ForTorchDataSet:
    def __init__(self, lengths, indices):
        self.lengths = lengths
        self.indices = torch.as_tensor(indices)

    def total_tokens(self):
        return sum(self.lengths).item()

    def as_indices_list(self):
        return [s.tolist()[:l] for s, l in zip(self.indices, self.lengths)]

    def __getitem__(self, i):
        return self.lengths[i], self.indices[i]

    def __len__(self):
        return len(self.lengths)


class LMDataModule(pl.LightningDataModule):
    def __init__(self, data, tokenizer, data_params, model_params,
                 verbose_init=False, from_folder=None, skeleton_load=False,
                 lang_counters=None):
        super().__init__()
        self.skeleton_load = skeleton_load
        self.from_path = None
        self.verbose_init = verbose_init
        self.lang_fullsample_counts = lang_counters  # doesnt tell you which
        # sample is which lang, but at least tells you how many there are.
        # note: counts *full* samples - ie before chunking for the max seq len
        if None is not from_folder:
            self.setup_from_folder(from_folder)
        else:
            self.data_params = data_params
            self.model_params = model_params
            self.tokenizer = tokenizer
            self.set_max_seq_len()

            if isinstance(data, list) or \
               isinstance(data, SyntheticSamplesIterator):
                self.setup_from_list(data)
            else:  # DatasetDict through huggingface
                self.setup_from_data_dict(data)
            self.prep_for_torch_datasets()
        if not self.skeleton_load:
            self.data_params.total_train_samples = len(self.train_samples)
            self.data_params.total_samples = \
                sum([len(ds) for ds in [self.train_samples, self.val_samples,
                                        self.test_samples]])
        self.tokenizer_size = self.tokenizer.vocab_size
        # a helpful curio to stick in the notes

    def set_max_seq_len(self):
        self.max_seq_len = self.model_params.max_seq_len
        if self.data_params.max_seq_len > 0:
            self.max_seq_len = min(self.max_seq_len,
                                   self.data_params.max_seq_len)

    def setup_from_folder(self, path):
        self.from_path = path
        with open(path_join(path, "model_params.json"), "r") as f:
            self.model_params = make_mp(**json.load(f), forgiving=True,
                                        takes_extras=True)
            # allows that mp may specify some additional things not in current
            # model_params definition, if made by more specific branch. assumes
            # these additional things only add information, but do not take
            # away or modify anything needed here
        with open(path_join(path, "data_params.json"), "r") as f:
            self.data_params = make_dp(**json.load(f), forgiving=True,
                                       takes_extras=True)
            # as with model_params above
        self.tokenizer = load_stored_tokenizer_if_exists(
            self.model_params.tokenizer_source_name, path, self.verbose_init)
        assert None is not self.tokenizer
        # can't be loading tokenized data without its tokenizer
        self.set_max_seq_len()

        with open(path_join(path, "dataloader_notes.json"), "r") as f:
            base_attrs = json.load(f)
        [setattr(self, an, base_attrs[an]) for an in base_attrs]

        if self.skeleton_load:
            return

        # load train, test, val samples
        for sn in ["train_samples", "test_samples", "val_samples"]:
            indices = np.load(path_join(path, f"{sn}-indices.npy"))
            lengths = np.load(path_join(path, f"{sn}-lengths.npy"))

            # TEMPORARY until stop using old dataloaders which didnt save
            # pads correctly (anything older than 2025.01.14), and had
            # them as -1 instead
            if (indices == -1).sum() > 0:
                indices = torch.as_tensor(indices)
                indices = torch.where(
                    indices == -1, self.tokenizer.pad_token_id, indices)
            setattr(self, sn, ForTorchDataSet(lengths, indices))

        self.finalise_data()

    def prep_for_torch_datasets(self):
        def arranged(samples):
            lengths = np.array([len(s) for s in samples])
            indices = np.zeros((len(samples), lengths.max()), dtype=int)
            for i, (n, inds) in enumerate(zip(lengths, samples)):
                indices[i, :n] = inds
                indices[i, n:] = self.tokenizer.pad_token_id
            return lengths, indices

        for sn in ["train_samples", "test_samples", "val_samples"]:
            lengths, indices = arranged(getattr(self, sn))
            setattr(self, sn, ForTorchDataSet(lengths, indices))

    def save_to_folder(self, path):
        prepare_directory(path)
        # save tokenizer, save self
        self.from_path = path
        self.tokenizer.save(path)
        base_attr_names = ["train_n", "test_n", "val_n",
                           "lang_fullsample_counts", "tokenizer_size"]
        notes = {n: getattr(self, n) for n in base_attr_names}
        with open(path_join(path, f"dataloader_notes.json"), "w") as f:
            json.dump(notes, f)
        with open(path_join(path, f"model_params.json"), "w") as f:
            json.dump(vars(self.model_params), f)
        with open(path_join(path, f"data_params.json"), "w") as f:
            json.dump(vars(self.data_params), f)

        for sn in ["train_samples", "test_samples", "val_samples"]:
            ds = getattr(self, sn)
            for a in ["lengths", "indices"]:
                v = getattr(ds, a)
                if isinstance(v, torch.Tensor):
                    v = v.cpu()  # else numpy wont save
                if None is not v:
                    np.save(path_join(path, f"{sn}-{a}.npy"), v, 
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
            if (self.data_params.task_type == "synthetic" and
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

        named_lists = [(self.train_samples, "train"),
                       (self.val_samples, "val"),
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

    def get_sample(self, i, from_ds="all"):
        orig_i = i
        if from_ds == "all":
            datasets = [self.train_samples, self.val_samples,
                        self.test_samples]
        else:
            datasets = [getattr(self, f"{from_ds}_samples")]
        for ds in datasets:
            if i < len(ds):
                n, indices = ds[i]  # length, indices
                return n, indices[:n]
            i -= len(ds)
        n = sum([len(ds) for ds in datasets])
        raise Exception(f"no sample at index {orig_i}, only have {n} " +\
                        f"samples in dataset {from_ds}")

    def get_sample_str(self, i=0, from_ds="all"):
        n, indices = self.get_sample(i, from_ds=from_ds)
        return self.tokenizer.convert_ids_to_nice_string(indices)

    def show_sample(self, i=0):
        n, indices = self.get_sample(i)
        print(f"sample {i}, has {n} tokens:")
        print(self.get_sample_str(i))

    def train_dataloader(self, batch_size, shuffle=True):
        return self.generic_loader(self.train_samples, batch_size,
                                   shuffle=shuffle)

    def val_dataloader(self, batch_size, shuffle=False):
        return self.generic_loader(self.val_samples, batch_size,
                                   shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return self.generic_loader(self.test_samples, batch_size,
                                   shuffle=shuffle)

    def predict_dataloader(self):
        return None  # what are they on about

    def teardown(self, stage: str):
        pass
        # Used to clean-up when the run is finished


def mycollate(b):
    # b: list of samples. each sample is a tuple of:
    # (int length, tensor of indices). the indices are padded with
    # the tokenizer's pad_token_id 
    dtype = torch.long
    lengths = [s[0] for s in b]
    seqlen = max(lengths)
    example_indices = b[0][1]
    device = example_indices.device

    batch_indices = torch.zeros((len(b), seqlen), dtype=dtype, device=device)
    # nasty nasty bug i dont understand makes the assignment into values in a
    # special branch go very wrong (massive (overflow?) values in row 1, and
    # then all zeros onward) if the zeros are not moved to the expected device
    # beforehand.
    # I am not able to reproduce the bug outside of this setting, so it must
    # also be something about how the values end up here - how they're gathered
    # by torch dataloaders before reaching batch collate.
    # this solves it, and i'm tired of tracking down what it is and just want
    # to not have it

    with_mask = len(set(lengths)) > 1
    mask = torch.ones(batch_indices.shape, dtype=dtype) if with_mask else None

    for i, (n, seq) in enumerate(b):
        batch_indices[i][:n] = seq[:n]
        if with_mask:
            mask[i][:n] = 0
        
    # indices shape: batch size X seq len.
    # values past an individual sequence's length are filled with the
    # tokenizer's pad_token_id
    # mask shape: batch size X seq len, or None. marks pads.
    # mask is 0 where the sequence is active and 1 where it is off (padded)
    if None is not mask:
        mask = mask.to(device=device)
    return {"x_indices": batch_indices.to(device=device), "mask": mask}
