import matplotlib.pyplot as plt
import saver
import glob
import lightning as pl
from train.trainer import Trainer
import torch
from util import prepare_directory, get_timestamp, glob_nosquares
import sys


get_model_cache = {}


def get_model_by_timestamp(timestamp, verbose=True, with_data=True, cache=False):
    if cache and (timestamp in get_model_cache):
        return get_model_cache[timestamp]

    p = get_full_path(timestamp)
    if None is p:
        return None

    full = saver.load_model(p, full=True, verbose=verbose, with_data=with_data)
    lm, dataset, train_stats = full[:3]
    mp, dp, tp = full[3:]
    params = {"model_params": mp, "data_params": dp, "train_params": tp}
    res = {"lm": lm, "dataset": dataset, "train_stats": train_stats,
           "params": params}
    if cache:
        get_model_cache[timestamp] = res
    return res


get_checkpoints_cache = {}


def get_all_checkpoints_by_timestamp(timestamp, verbose=True, with_data=True,
                                     cache=False):
    identifier = (timestamp, with_data)
    if cache and (identifier in get_checkpoints_cache):
        return get_checkpoints_cache[identifier]

    p_final = get_full_path(timestamp)
    p_containing = p_final[:-len("/final/")]
    paths = glob_nosquares(f"{p_containing}/*/")
    results = {"models":{}}
    for p in paths:
        desc = p.split("/")[-2]
        desc = "final" if desc == "final" else int(desc)
        full = saver.load_model(p, full=True, verbose=verbose,
                                with_data=with_data)
        lm, dataset, train_stats = full[:3]
        train_stats["total_train_samples"] = \
            train_stats.get("n_train_samples",[[0]])[-1][0]
            # if not got, then this is the model at time 0 - no training yet
        results["models"][desc] = {"lm": lm, "train_stats": train_stats}

        if "params" not in results:
            if with_data:
                results["dataset"] = dataset
            results["params"] = {"model_params": full[3],
                                 "data_params": full[4],
                                 "train_params": full[5]}
    if cache:
        get_checkpoints_cache[identifier] = results
    return results


def verify_stable_load(timestamp):
    a1 = get_model_by_timestamp(timestamp)
    a2 = get_model_by_timestamp(timestamp)
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


def get_full_path(timestamp):
    paths = glob.glob("../saved-models/**/", recursive=True)
    paths = [p for p in paths if (timestamp in p and p.endswith("/final/"))]
    if len(paths) == 1:
        return paths[0]
    if len(paths) < 1:
        print("could not find model folder with timestamp:", timestamp)
        return None
    if len(paths) > 1:
        print("found multiple model folders with timestamp:", timestamp)
        print(paths)
        return None


def plot_metric(train_stats, metric_name, folder_name=None):
    d = train_stats[metric_name]
    n_train_samples, metric, stat_indices = list(zip(*d))
    fig, ax = plt.subplots()
    plt.scatter(n_train_samples, metric, s=0.5)
    plt.xlabel("n_train_samples")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    fig = plt.gcf()
    fig.show()
    if None is not folder_name:
        f = f"../metrics/{folder_name}"
        prepare_directory(f)
        fig.savefig(f"{f}/{metric_name}.png")


def list_metrics(train_stats):
    return list(train_stats.keys())


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


def show_lm_attns(timestamp, x, layers=None, heads=None, store=False, 
                  chkpt="final", cache=True):
    chkpts_dict = get_all_checkpoints_by_timestamp(timestamp,
        verbose=False, cache=cache, with_data=False)
    task_name = chkpts_dict["params"]["data_params"].dataset_name
    lm = chkpts_dict["models"][chkpt]["lm"]
    nsamples = chkpts_dict["models"][chkpt]["train_stats"]["total_train_samples"]

    # x: input sequence, whether as string or as list of token indices
    res = lm(x, get_attns=True)
    z, attns = res["logits"], res["attns"]
    attns = attns.detach()
    # attns shape: batch size x n layers x n heads x out seq len x in seq len
    # batch size should be 1

    if store:
        folder_name = f"../attentions/{task_name}/{timestamp}/" +\
                      f"heads-at-chkpt/{chkpt}"
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
    alphabet = list(res["models"][0]["lm"].tokenizer.get_vocab().keys())

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
                                  chkpt=nsamples, cache=cache)
        if store:
            fig.savefig(f"{folder_name}/{nsamples}")
