import matplotlib.pyplot as plt
from save_load import load_model, load_model_info, final_chkpt, models_paths
import lightning as pl
from train.trainer import Trainer
import torch
from util import prepare_directory, get_timestamp, glob_nosquares
import sys
from os.path import join as path_join
from util import printer_print as print
import json


assert False not in [p.endswith("/saved-models") for p in models_paths]
# "task_name" function in auto_timestamps makes this assumption, so be sure

def auto_timestamps():
    def is_timestamp(sample):
        example = "2024-08-20--12-12-12"
        if not len(sample) == len(example):
            return False
        for s,e in zip(sample, example):
            if e == "-":
                if not s == "-":
                    return False
            elif not s in "0123456789":
                return False
        return True
    
    def task_name(path):
        try:
            with open(path_join(path,final_chkpt,"data_params.json"), "r") as f:
                a = json.load(f)
            return a["dataset_name"]
        except Exception as e:
            print(e)
            return None
    
    def last_folder(path):
        return path.split("/")[-1]
    
    def last_is_timestamp(path):
        return is_timestamp(last_folder(path))
    
    all_paths = []
    for p in models_paths:
        all_paths += glob_nosquares(f"{p}/**", recursive=True)
    all_paths = [p for p in all_paths if last_is_timestamp(p)]
    all_tuples = [(task_name(p), last_folder(p), p) for p in all_paths]
    all_tuples = [t for t in all_tuples if None is not t]
    res = {}
    for tn, ts, p in all_tuples:
        if tn not in res:
            res[tn] = []
        res[tn].append((ts, p))
    return res


def checkpoint_ids(timestamp):
    all_paths = []
    for p in models_paths:
        all_paths += glob_nosquares(f"{p}/**", recursive=True)
    folder = next(p for p in all_paths if p.endswith(f"/{timestamp}"))
    def is_checkpoint_folder(p):
        bits = p.split(f"/{timestamp}/")
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
    return sorted([i for i in chkpts if isinstance(i,int)]) + \
           sorted([i for i in chkpts if not isinstance(i,int)])


get_model_cache = {}


def get_model_by_timestamp(timestamp, checkpoint=final_chkpt, verbose=True,
                           with_data=True, cache=False):
    identifier = (timestamp, checkpoint, with_data)
    if cache:
        if identifier in get_model_cache:
            return get_model_cache[identifier]
        id2 = (timestamp, checkpoint, True)
        if id2 in get_model_cache:   # data already loaded, no harm
            return get_model_cache[id2]

    p = get_full_path(timestamp, checkpoint=checkpoint)
    if None is p:
        if verbose:
            print("did not find path with timestamp:",timestamp)
        return None
    if verbose:
        print("found model path:",p)

    res = load_model(p, full=True, verbose=verbose, with_data=with_data)

    if verbose:
        if None is not res:
            print("succesfully loaded model")
        else:
            print("failed to load model from path")

    if cache:
        if (timestamp, checkpoint, False) in get_model_cache:
            del get_model_cache[(timestamp, checkpoint, False)]
        get_model_cache[identifier] = res

    return res


get_checkpoints_cache = {}


def get_all_checkpoints_by_timestamp(timestamp, verbose=True, with_data=True,
                                     cache=False):
    identifier = (timestamp, with_data)
    if cache: 
        if identifier in get_checkpoints_cache:
            return get_checkpoints_cache[identifier]
        id2 = (timestamp, True)
        if id2 in get_checkpoints_cache:  # no harm in extra info
            return get_checkpoints_cache[id2]

    p_final = get_full_path(timestamp, checkpoint=final_chkpt)
    p_containing = p_final[:-len(f"/{final_chkpt}/")]
    paths = glob_nosquares(f"{p_containing}/*/")
    results = {"models":{}}
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
        if (timestamp, False) in get_checkpoints_cache:
            del get_checkpoints_cache[(timestamp, False)]
        get_checkpoints_cache[identifier] = results
    return results


def clear_chkpts_cache():
    for k in get_checkpoints_cache:
        del get_checkpoints_cache[k]


info_cache = {}


def get_info(timestamp):  # always caches, info is small
    if timestamp in info_cache:
        return info_cache[timestamp]
    path = get_full_path(timestamp, checkpoint=final_chkpt)
    res = load_model_info(path)
    info_cache[timestamp] = res
    return res


def verify_stable_load(timestamp, checkpoint=final_chkpt):
    a1 = get_model_by_timestamp(timestamp, checkpoint)
    a2 = get_model_by_timestamp(timestamp, checkpoint)
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
          f"{val_diff1,val_diff2}" +\
          "check data creation/loading isn't randomised?"
    assert (abs(val_diff1) < threshold) and (abs(val_diff2) < threshold), msg

    msg = "checked validations different from each other: " +\
          f"{val_diff1,val_diff2}"
    assert (val_diff1 == val_diff2), msg

    msg = "after checking val loss, two loaded models now giving different " +\
          "outputs on same input - check models haven't switched to " +\
          "training mode (pytorch lightning does this after validation)"
    assert torch.equal(m1([[1, 2]])["logits"], m2([[1, 2]])["logits"]), msg

    print("passed basic load checks")


def _longest_common_prefix(strings):
    for (i, vals) in enumerate(zip(*strings)):
        if len(set(vals)) > 1:
            return strings[0][:i]
    return strings[0][:i+1]


def _ylabel(metric_names):
    if len(metric_names) == 1:
        if isinstance(metric_names, dict):
            return metric_names[metric_names.keys()[0]]
        else:
            return metric_names[0]
    if isinstance(metric_names, list):
        res = _longest_common_prefix(metric_names)
    if isinstance(metric_names, dict):
        res = _longest_common_prefix(
            [short_name for full_name, short_name in metric_names.items()])
    while res.endswith("/") or res.endswith("|"):
        res = res[:-1]
    return res if res else "metric"


def _plt_title(title, metric_names, timestamps):
    if None is not title:
        return title
    single_timestamp = len(timestamps) == 1
    if len(metric_names) == 1:
        i = 0 if isinstance(metric_names, list) else metric_names.keys()[0]
        return metric_names[i]
    if single_timestamp and isinstance(timestamps, list):
        return get_info(
            timestamps[0])["params"]["data_params"].dataset_name
    if single_timestamp and isinstance(timestamps, dict):
        return timestamps[timestamps.keys()[0]]
    # multi metrics, multi timestamps
    all_ds_names = [get_info(t)["params"]["data_params"].dataset_name for \
                    t in timestamps]
    if len(set(all_ds_names)) == 1:
        return all_ds_names[0]
    raise Exception("given multiple tasks and multiple metrics",
                    "please provide title for plot")


def _line_label(timestamp, metric, timestamps, metric_names, ylabel):
    if isinstance(timestamps, dict):
        # timestamps have been given with labels for plot
        ts_label = timestamps[timestamp]
    else:
        t_info = get_info(timestamp)
        ts_label = t_info["params"]["data_params"].dataset_name
    
    if len(metric_names) == 1:
        return ts_label

    metric_label = metric_names[metric] if \
        isinstance(metric_names, dict) else metric 
    if metric_label.startswith(ylabel):
        metric_label = metric_label[len(ylabel):]
    while metric_label.startswith("/") or metric_label.startswith("|"):
        metric_label = metric_label[1:]
    
    if len(timestamps) == 1:
        return metric_label

    return f"{ts_label}::{metric_label}"


def _plot(x, y, s=0.5, label=None, plot_type="scatter",
          max_x=None, min_x=None, max_y=None, min_y=None):
    def keep(vx, vy):
        return (vx < max_x) and (vx > min_x) and (vy < max_y) and (vy > min_y)
    max_x = max_x if None is not max_x else torch.inf
    min_x = min_x if None is not min_x else -torch.inf
    max_y = max_y if None is not max_y else torch.inf
    min_y = min_y if None is not min_y else -torch.inf
    x, y = list(zip(*[(vx, vy) for vx, vy in zip(x, y) if keep(vx, vy)]))
    if plot_type == "scatter":
        plt.scatter(x, y, s=s, label=label)
    elif plot_type == "line":
        plt.plot(x, y, label=label)


def plot_metrics(timestamps, metric_names, title=None, filename=None,
                 add_to=None, plot_type="scatter", 
                 max_x=None, min_x=None, max_y=None, min_y=None):
    # timestamps can be a dict giving the timestamps special names for 
    # the plot labels, or just an iterable with the timestamps of interest
    # (in which case they will be labeled by their task name)
    if isinstance(timestamps, str):
        timestamps = [timestamps]
    if isinstance(metric_names, str):
        metric_names = [metric_names]
        
    print("printing plots from models from folders:")
    for t in timestamps:
        print(t, get_full_path(t))

    ylabel = _ylabel(metric_names)
    fig, ax = plt.subplots() if None is add_to else add_to
    plt.xlabel("n_train_samples")        
    plt.ylabel(ylabel)
    plt.title(_plt_title(title, metric_names, timestamps))
    
    for t in timestamps:
        for m in metric_names:
            t_info = get_info(t)
            if m not in t_info["train_stats"]:
                continue  # eg if trying to show copy loss on several
                # timestamps but one is just pairs
            d = t_info["train_stats"][m]
            if len(d[0]) == 3:  # older version
                n_train_samples, metric, stat_counter = list(zip(*d))
            else:  # newer version
                stat_syncer, n_train_samples, stat_counter, metric = \
                    list(zip(*d))
            _plot(n_train_samples, metric, plot_type=plot_type, 
                max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y,
                label=_line_label(t, m, timestamps, metric_names, ylabel))

    ax.legend(markerscale=3)
    fig = plt.gcf()
    fig.show()
    if None is not filename:
        fn = f"../metrics/{filename}"
        directory = '/'.join(fn.split('/')[:-1])
        prepare_directory(directory)
        fig.savefig(f"{fn}.png")
        with open(f"{fn}.txt", "w") as f:
            print(f"plot in {fn} made from timestamps:{timestamps}\n",
                  f"and metrics:\n{metric_names}",file=f)
            print("timestamp full paths:\n",file=f)
            for t in timestamps:
                print(t,"\t:",get_full_path(t),"\n",file=f)
    return fig, ax


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
    recorded_val_loss = loaded_model["train_stats"]["stat/val_loss:main"][-1][-1]
    curr_val_loss = compute_validation(loaded_model["lm"], 
                                       loaded_model["dataset"],
                                       loaded_model["params"], sample=False)
    print("last recorded val loss:", recorded_val_loss,
          ", loaded model val loss:", curr_val_loss,
          ",\ndiff:", recorded_val_loss - curr_val_loss)
    return recorded_val_loss - curr_val_loss


def show_lm_attns(timestamp, x, layers=None, heads=None, store=False, 
                  checkpoint=final_chkpt, cache=True):
    
    chkpt = get_model_by_timestamp(timestamp, checkpoint=checkpoint,
        verbose=False, with_data=False, cache=cache)
    task_name = chkpt["params"]["data_params"].dataset_name
    lm = chkpt["lm"]
    nsamples = chkpt["train_stats"]["total_train_samples"]

    # x: input sequence, whether as string or as list of token indices
    res = lm(x, get_attns=True)
    z, attns = res["logits"], res["attns"]
    attns = attns.detach()
    # attns shape: batch size x n layers x n heads x out seq len x in seq len
    # batch size should be 1

    if store:
        folder_name = f"../attentions/{task_name}/{timestamp}/" +\
                      f"heads-at-chkpt/{checkpoint}"
        prepare_directory(folder_name)

    layers = list(range(attns.shape[1])) if None is layers else layers
    heads = list(range(attns.shape[2])) if None is heads else heads
    assert attns.shape[0] == 1  # single x - batch size 1
    attns = attns[0].cpu()  # n layers x n heads x out seq len x in seq len
    for layer in layers:
        for h in heads:
            fig, ax = plt.subplots()
            im = ax.imshow(attns[layer][h])
            cbar = plt.colorbar(im)
            # cbar.ax.set_ylabel("sigmoid value", rotation=-90, va="bottom")
            if isinstance(x, str):
                token_ids = lm.tokenizer(x)
            else:  # list or tensor - tokenizer can handle either
                token_ids = x
            tokens = lm.tokenizer.convert_ids_to_tokens(token_ids)
            ax.set_xticks(range(len(token_ids)), labels=tokens)
            ax.set_yticks(range(len(token_ids)), labels=tokens)

            title = f"attn pattern for head {h} in layer {layer}"
            title += f" after {nsamples} samples"
            plt.title(title)
            plt.xlabel("in dim")
            plt.ylabel("out dim")
            plt.xticks(rotation=-45, ha='left')
            fig = plt.gcf()
            fig.show()
            if store:
                fig.savefig(f"{folder_name}/L[{layer}]-H[{h}]")
    return z, attns, fig  # last fig, but good enough when only requesting one


def show_head_progress(timestamp, x, l, h, cache=True, store=False):
    res = get_all_checkpoints_by_timestamp(timestamp,
        verbose=False, cache=cache, with_data=False)
    
    task_name = res["params"]["data_params"].dataset_name
    alphabet = \
        list(res["models"][final_chkpt]["lm"].tokenizer.get_vocab().keys())

    if store:
        folder_name = f"../attentions/{task_name}/{timestamp}/" +\
                      f"heads-over-time/L[{l}]-H[{h}]"
        prepare_directory(folder_name)

    f = open(f"{folder_name}/notes.txt","w") if store else sys.stdout

    print(f"showing attn patterns for model trained on task: [ {task_name} ].",
          file=f)
    print("\ttask alphabet:", "".join(sorted(alphabet)), file=f)
    print("\tinput sequence:",x,file=f)

    if store:
        f.close()

    models_by_nsamples = {d["train_stats"]["total_train_samples"]: d["lm"] for\
                          n, d in res["models"].items()}

    for nsamples in sorted(list(models_by_nsamples.keys())):
        _, _, fig = show_lm_attns(timestamp, x, layers=[l], heads=[h],
                                  checkpoint=str(nsamples), cache=cache)
        if store:
            fig.savefig(f"{folder_name}/{nsamples}")

def get_full_path(timestamp, checkpoint=final_chkpt):
    paths = []
    for p in models_paths:
        paths += glob_nosquares(f"{p}/**/", recursive=True)
    paths = [p for p in paths if (f"/{timestamp}/" in p and \
                                  p.endswith(f"/{checkpoint}/"))]
    if len(paths) == 1:
        return paths[0]
    if len(paths) < 1:
        print("could not find model folder with:", timestamp, checkpoint)
        return None
    if len(paths) > 1:
        print("found multiple model folders with:", timestamp, checkpoint)
        print(paths)
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


def same_characteristics(timestamps, mp=True, tp=True, dp=True, ignorable=None):
    if not timestamps:
        return True
    infos = [(t, load_model_info(get_full_path(t))) for t in timestamps]
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
                    print(f"mismatch between {get_full_path(t0)} and",
                          f"{get_full_path(t)} on value {k}: {v} vs {p[k]}")
                    return False
    return True
