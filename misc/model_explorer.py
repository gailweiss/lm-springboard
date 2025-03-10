from misc.save_load import load_model, load_model_info, final_chkpt, \
                           models_paths
import lightning as pl
from train.trainer import Trainer
import torch
from misc.util import glob_nosquares, pad, same_dict_structure
import sys
from os.path import join as path_join
from misc.util import printer_print as print
import json
import itertools
from copy import deepcopy
from data.dataloader import datamodules_paths, LMDataModule


assert False not in [p.endswith("/saved-models") for p in models_paths]
# "task_name" function in auto_identifiers makes this assumption, so be sure


def identifier2timestamp(identifier):
    if identifier.count("---") == 1:
        return identifier.split("---")[0]
    if identifier.count("---") == 0:
        return identifier
    return -1


shortest_example_m_id = "2024-01-01--00-00-00---0"
example_timestamp = "2024-08-20--12-12-12"


def all_aligned_dashes_digits(s1, s2):
    if not (isinstance(s1, str) and isinstance(s2, str)):
        return False
    if not len(s1) == len(s2):
        return False
    digits = "0123456789"
    for c1, c2 in zip(s1, s2):
        if c1 in digits and c2 in digits:
            continue
        if c1 == "-" and c2 == "-":
            continue
        return False
    return True


def model_id_from_top_of_string(string):
    m_id_start = string[:len(shortest_example_m_id)]
    if not all_aligned_dashes_digits(m_id_start, shortest_example_m_id):
        # this string doesn't open with a model id
        return None
    remaining = list(string[len(shortest_example_m_id):])
    d = remaining.pop(0)
    while d in "0123456789":
        m_id_start += d
        d = remaining.pop(0) if remaining else "!"
    return m_id_start


def is_timestamp(seq):
    return all_aligned_dashes_digits(seq, example_timestamp)


def is_identifier(seq):
    if not isinstance(seq, str):
        return False
    if "---" in seq:
        if seq.count("---") > 1:
            return False
        seq, rand = seq.split("---")
        for d in rand:
            if d not in "0123456789":
                return False
    return is_timestamp(seq)


def auto_identifiers():
    def task_name(path):
        try:
            with open(path_join(path, final_chkpt, "data_params.json"),
                      "r") as f:
                a = json.load(f)
            return a["dataset_name"]
        except Exception as e:
            print(e)
            return None

    def last_folder(path):
        return path.split("/")[-1]

    def last_is_identifier(path):
        return is_identifier(last_folder(path))

    all_paths = []
    for p in models_paths:
        all_paths += glob_nosquares(f"{p}/**", recursive=True)
    all_paths = [p for p in all_paths if last_is_identifier(p)]
    all_tuples = [(task_name(p), last_folder(p), p) for p in all_paths]
    res = {}
    for tn, identifier, p in all_tuples:
        if tn not in res:
            res[tn] = []
        res[tn].append((identifier, p))

    for tn in res:
        res[tn] = sorted(res[tn], key=lambda x: x[0])
    return res


def date_in_range(i, min_date, max_date):
    ts = identifier2timestamp(i)
    if not is_timestamp(ts):
        print("got bad identifier:", i, ". not using")
        return False
    if None is not max_date:
        if not is_timestamp(max_date):
            print("bad max date request:", max_date)
            return False
        if ts > max_date:
            return False
    if None is not min_date:
        if not is_timestamp(min_date):
            print("bad min date request:", min_date)
            return False
        if ts < min_date:
            return False
    return True


def all_identifiers_with_configs(kws, min_date=None, max_date=None, verbose=False):
    # kws: nested dict of configs.
    # level one: data_params, train_params, model_params
    # level two: each config that is being specified in this request,
    # e.g. n_layers
    # values at level 2 can be either a single value (e.g. 4) or a
    # list (specifically a list, not a tuple) of acceptable ones.
    # note this ties into constraints on parameters in other places -
    # lists in the config files are treated as lists of values to run
    # rather than a single value (that is a list) for that parameter
    # min_date/max_date example format: "2024-10-17--00-00-00"
    identifiers_dict = auto_identifiers()
    dataset_names = kws.get("data_params", {}).get("dataset_name", [])
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]
    if not dataset_names:
        dataset_names = list(identifiers_dict.keys())
    res = itertools.chain.from_iterable(
        [identifiers_dict[dn] for dn in dataset_names])

    res = [i for i, path in res if date_in_range(i, min_date, max_date)]

    def get_param(identifier, paramset_name, param_name):
        info = get_info(identifier, verbose=verbose)
        if None is info:
            return []
        if None is info["params"][paramset_name]:
            return []
        d = vars(info["params"][paramset_name])
        return d.get(param_name, [])
        # [] is not a valid param value, for reasons outlined above

    for paramset_name in kws:
        for param_name in kws[paramset_name]:
            target_vals = kws[paramset_name][param_name]
            if not isinstance(target_vals, list):  # not multiple options
                target_vals = [target_vals]  # treat as multiple options
            res = [i for i in res if
                   get_param(i, paramset_name, param_name) in target_vals]

    return res


def ids_with_missing_paramsets(identifiers, verbose=False):
    res = set([i for i in identifiers if None in get_info(i).values()])
    if res and verbose:
        print("following ids have a missing paramset:", file=file)
        for i in sorted(list(res)):
            missing_sets = [psn for psn in get_info(i) if
                            None is get_info(i)[psn]]
            print(f"\n{i}, missing: {missing_sets}. path: {get_full_path(i)}",
                  file=file)
        print("==\n\n", file=file)
    return res


def get_configs_values(identifiers, verbose=False):
    if ids_with_missing_paramsets(identifiers, verbose=verbose):
        return -1

    def params_dicts(model_info):
        return {k: vars(v) for k, v in model_info["params"].items()}

    infos = [(i, params_dicts(get_info(i))) for i in identifiers]

    # check all infos have same structure before reaching weird conclusions
    example_id, example_info = infos[0]
    for i, info in infos:
        if not same_dict_structure(example_info, info, verbose=verbose,
                                   pref=f"in ids {example_id} vs {i}"):
            return -1

    all_vals = deepcopy(example_info)
    for k1 in all_vals:
        for k2 in all_vals[k1]:
            all_vals[k1][k2] = set([all_vals[k1][k2]])
            for i, info in infos[1:]:
                all_vals[k1][k2].add(info[k1][k2])
    return all_vals


def print_config_compare(identifiers, print_padding=30, file=sys.stdout,
                         just_list_differing_keys=False):
    
    all_vals = get_configs_values(identifiers, verbose=True)
    if not isinstance(all_vals, dict):
        print("could not get config value sets for these identifiers")
        return all_vals

    padline = "="*40 + "\n"

    if not just_list_differing_keys:
        print(f"{padline}\tconstant values:\n{padline}", file=file)
        for k1 in all_vals:
            print(f"\n === {k1}: ===\n", file=file)
            for k2 in all_vals[k1]:
                if len(all_vals[k1][k2]) == 1:
                    print(f"{pad(k2, print_padding, 'left')}:\t",
                          list(all_vals[k1][k2])[0], file=file)

    print(f"\n\n{padline}\tvaried values:\n{padline}", file=file)
    if just_list_differing_keys:
        for k1 in all_vals:
            for k2 in all_vals[k1]:
                if len(all_vals[k1][k2]) > 1:
                    print(f"{k1}:\t\t{k2}", file=file)
    else:
        for k1 in all_vals:
            print(f"\n=== {k1}: ===\n", file=file)
            for k2 in all_vals[k1]:
                if len(all_vals[k1][k2]) > 1:
                    print(f"{pad(k2, print_padding, 'left')}:\t",
                          list(all_vals[k1][k2]), file=file)


def checkpoint_ids(identifier):
    all_paths = []
    for p in models_paths:
        all_paths += glob_nosquares(f"{p}/**", recursive=True)
    folder = next(p for p in all_paths if p.endswith(f"/{identifier}"))

    def is_checkpoint_folder(p):
        bits = p.split(f"/{identifier}/")
        if len(bits) != 2:
            return False
        return "/" not in bits[1]

    checkpoint_paths = [p for p in all_paths if is_checkpoint_folder(p)]

    def as_int(c):
        for t in c:
            if t not in "0123456789":
                return c
        return int(c)
    chkpts = [as_int(p.split("/")[-1]) for p in checkpoint_paths]
    return sorted([i for i in chkpts if isinstance(i, int)]) + \
        sorted([i for i in chkpts if not isinstance(i, int)])


def get_datamodule_by_identifier(identifier):
    for p in datamodules_paths:
        paths = glob_nosquares(f"{p}/*/{identifier}")
        if paths:
            assert len(paths) == 1
            path = paths[0]
            print("getting datamodule from:", path)
            return LMDataModule(None, None, None, None, from_folder=path)


get_model_cache = {}


def get_model_by_identifier(identifier, checkpoint=final_chkpt, verbose=True,
                            with_data=True, cache=False):
    cache_identifier = (identifier, checkpoint, with_data)
    if cache:
        if cache_identifier in get_model_cache:
            return get_model_cache[cache_identifier]
        id2 = (identifier, checkpoint, True)
        if id2 in get_model_cache:   # data already loaded, no harm
            return get_model_cache[id2]

    p = get_full_path(identifier, checkpoint=checkpoint, verbose=verbose)
    if None is p:
        if verbose:
            print("did not find path with identifier and checkpoint:",
                  identifier, checkpoint)
        return None
    if verbose:
        print("found model path:", p)

    res = load_model(p, full=True, verbose=verbose, with_data=with_data)

    if verbose:
        if None is not res:
            print("succesfully loaded model")
        else:
            print("failed to load model from path")

    if cache:
        if (identifier, checkpoint, False) in get_model_cache:
            del get_model_cache[(identifier, checkpoint, False)]
        get_model_cache[cache_identifier] = res

    return res


def get_checkpoint_names_by_identifier(identifier):
    paths = []
    for p in models_paths:
        paths += glob_nosquares(f"{p}/**/", recursive=True)
    paths = [p for p in paths if f"/{identifier}/" in p]
    paths = [p for p in paths if p.split("/")[-3] == identifier]
    # chkpt path format: 
    # '{task-and-config specific models folder}/{long model name}/{identifier}/{chkpt}/'
    chkpts = [p.split("/")[-2] for p in paths]
    return sorted(chkpts, key=lambda x:int(x) if not x=="final" else torch.inf)


get_checkpoints_cache = {}


def get_all_checkpoints_by_identifier(identifier, verbose=True, with_data=True,
                                      cache=False):
    cache_identifier = (identifier, with_data)
    if cache:
        if cache_identifier in get_checkpoints_cache:
            return get_checkpoints_cache[cache_identifier]
        id2 = (identifier, True)
        if id2 in get_checkpoints_cache:  # no harm in extra info
            return get_checkpoints_cache[id2]

    p_final = get_full_path(identifier, checkpoint=final_chkpt, verbose=verbose)
    p_containing = p_final[:-len(f"/{final_chkpt}/")]
    paths = glob_nosquares(f"{p_containing}/*/")
    results = {"models": {}}
    for p in paths:
        desc = p.split("/")[-2]
        desc = final_chkpt if desc == final_chkpt else int(desc)
        res = load_model(p, full=True, verbose=verbose,
                         with_data=with_data)
        results["models"][desc] = {a: res[a] for a in ["lm", "train_stats"]}

        if "params" not in results:
            if with_data:
                results["dataset"] = res["dataset"]
            results["params"] = res["params"]

    if cache:
        if (identifier, False) in get_checkpoints_cache:
            del get_checkpoints_cache[(identifier, False)]
        get_checkpoints_cache[cache_identifier] = results
    return results


def clear_chkpts_cache():
    for k in get_checkpoints_cache:
        del get_checkpoints_cache[k]


info_cache = {}


def get_info(identifier, with_train_stats=False, verbose=True,
             dont_cache=False, get_lite=True, store_lite=True):
    # always caches, info is generally small
    cache_id = (identifier, with_train_stats, get_lite, store_lite)
    if cache_id not in info_cache:
        path = get_full_path(identifier, checkpoint=final_chkpt,
                             verbose=verbose)
        if None is path:
            if verbose:
                print("could not get final checkpoint path for identifier:",
                      identifier)
            return None  # don't cache, in case file gets made/copied in soon
        res = load_model_info(path, with_train_stats=with_train_stats,
                              verbose=verbose, get_lite=get_lite,
                              store_lite=True)
        if not dont_cache:
            info_cache[cache_id] = res
    else:
        res = info_cache[cache_id]
    return res


def verify_stable_load(identifier, checkpoint=final_chkpt):
    a1 = get_model_by_identifier(identifier, checkpoint)
    a2 = get_model_by_identifier(identifier, checkpoint)
    m1 = a1["lm"]
    m2 = a2["lm"]

    z1 = list(m1.parameters())
    z2 = list(m2.parameters())

    msg = "two loaded models yielded different number of parameters"
    assert len(z1) == len(z2), msg

    msg = "two loaded models got different values in some parameters"
    assert False not in [torch.equal(p1, p2) for p1, p2 in zip(z1, z2)], msg

    msg = "two loaded models giving different outputs on same input - " +\
          "check models aren't in training mode?"
    assert torch.equal(m1([[1, 2]])["logits"], m2([[1, 2]])["logits"]), msg

    threshold = 1e-4
    val_diff1 = check_validation(a1)
    val_diff2 = check_validation(a2)
    msg = "validation too different from last recorded validation, diffs: " +\
          f"{val_diff1, val_diff2}" +\
          "check data creation/loading isn't randomised?"
    assert (abs(val_diff1) < threshold) and (abs(val_diff2) < threshold), msg

    msg = "checked validations different from each other: " +\
          f"{val_diff1, val_diff2}"
    assert (val_diff1 == val_diff2), msg

    msg = "after checking val loss, two loaded models now giving different " +\
          "outputs on same input - check models haven't switched to " +\
          "training mode (pytorch lightning does this after validation)"
    assert torch.equal(m1([[1, 2]])["logits"], m2([[1, 2]])["logits"]), msg

    print("passed basic load checks")


def compute_validation(lm, dataset, params, sample=True):
    was_training = lm.training
    pltrainer = pl.Trainer(enable_checkpointing=False, logger=False,
                           devices=1, max_epochs=1)
    tp = params["train_params"]
    mytrainer = Trainer(lm, tp, samples_at_validation=sample)
    pltrainer.validate(mytrainer,
                       dataloaders=dataset.val_dataloader(tp.batch_size))
    if not was_training:
        # lightning turns training back on after computing validation loss!
        lm.eval()
    return mytrainer.last_val_loss


def check_validation(loaded_model):
    recorded_val_loss = loaded_model["train_stats"]["val_loss:main"][-1][1]
    curr_val_loss = compute_validation(loaded_model["lm"],
                                       loaded_model["dataset"],
                                       loaded_model["params"], sample=False)
    print("last recorded val loss:", recorded_val_loss,
          ", loaded model val loss:", curr_val_loss,
          ",\ndiff:", recorded_val_loss - curr_val_loss)
    return recorded_val_loss - curr_val_loss




def get_full_path(identifier, checkpoint=final_chkpt, verbose=True):
    paths = []
    for p in models_paths:
        paths += glob_nosquares(f"{p}/**/", recursive=True)
    paths = [p for p in paths if
             (f"/{identifier}/" in p and p.endswith(f"/{checkpoint}/"))]
    if len(paths) == 1:
        return paths[0]
    if len(paths) < 1:
        if verbose:
            print("could not find model folder with:", identifier, checkpoint)
        return None
    if len(paths) > 1:
        if verbose:
            print("found multiple model folders with:", identifier, checkpoint)
            print("\n", "\n".join(paths))
        return None


def have_same_tokenization(lm1, lm2, datamodule, n_checks):
    for i in range(n_checks):
        sample_str = datamodule.get_sample_str(i)
        i1, i2 = lm1.tokenizer(sample_str), lm2.tokenizer(sample_str)
        if i1 != i2:
            return False
    return True


def just_last_stats(model_stats):
    def single_stat(s):
        if not isinstance(s, list):
            return s
        s = s[-1]
        if len(s) == 3:
            return s[1]  # older version: main metric in middle of tuple
        else:
            return s[-1]  # now main metric in last pos
    return {n: single_stat(s) for n, s in model_stats.items()}


def same_characteristics(identifiers, mp=True, tp=True, dp=True,
                         ignorable=None):
    if not identifiers:
        return True
    if None is not ignorable:
        ignorable = set(ignorable)  # in case it gets long and annoying
    infos = [(t, get_info(t)) for t in identifiers]
    to_check = (["model_params"] if mp else []) + \
               (["train_params"] if tp else []) + \
               (["data_params"] if dp else [])
    for pn in to_check:
        tparams = [(t, i["params"][pn]) for (t, i) in infos]
        t0, params0 = tparams[0]
        for k, v in vars(params0).items():
            if None is not ignorable and k in ignorable:
                continue
            for t, p in tparams[1:]:
                p = vars(p)
                if p[k] != v:
                    print(f"mismatch between:\n{get_full_path(t0)} and",
                          f"\n{get_full_path(t)}\n on value {k}:",
                          f"{v} vs {p[k]}")
                    return False
    return True


def find_existing_datamodule_path(data_params, model_params):
    def is_match(dpd, mpd):
        # all the attrs that determine the dataset and the tokenizer
        important_model_attrs = ["max_seq_len", "tokenizer_source_name"]
        important_data_attrs = ["dataset_name", "debug_crop",
                                "breaking_synthetic_samples_ok",
                                "val_pct", "test_pct", "lines_per_sample",
                                "max_seq_len"]

        for a in important_data_attrs:
            if not dpd[a] == getattr(data_params, a):
                print("mismatch on data attr:", a)
                return False
        for a in important_model_attrs:
            if not mpd[a] == getattr(model_params, a):
                print("mismatch on model attr:", a)
                return False
        if model_params.tokenizer_source_name == "custom":
            if not (mpd["custom_tokenizer_ntokens"] ==
                    model_params.custom_tokenizer_ntokens):
                print("mismatch on number of tokens")
                return False

        def same_keys(d,p):
            return sorted(list(d.keys())) == sorted(list(vars(p).keys()))

        if not same_keys(dpd, data_params):
            print("mismatch on data attributes---different branch or commit")
            return False
        # model staying the same not really important so long as the
        # data-relevant attributes (tokenizer, sequence length) are fine
        # if not same_keys(mpd, model_params):
        #     print("mismatch on model attributes---different branch or commit")
        #     return False
        return True

    print("checking for existing datamodule")
    for p in datamodules_paths:
        print("checking inside path:", p)
        print("path contains:", glob_nosquares(f"{p}/*"))
        with_identifiers = glob_nosquares(f"{p}/{data_params.dataset_name}/*")
        for path in with_identifiers:
            print("checking path:", path)
            with open(path_join(path, "model_params.json"), "r") as f:
                mpd = json.load(f)
            with open(path_join(path, "data_params.json"), "r") as f:
                dpd = json.load(f)

            if is_match(dpd, mpd):
                print("matched!")
                return path
    print("no match!")
    return None


def find_existing_datamodule(data_params, model_params):
    path = find_existing_datamodule_path(data_params, model_params)
    if None is not path:
        return LMDataModule(None, None, None, None, from_folder=path)
    else:
        print("no path for these params")
        return None
