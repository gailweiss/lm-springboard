import matplotlib.pyplot as plt
import saver
import glob
import lightning as pl
from model.lmtrainer import LMTrainer
import torch
from util import prepare_directory


def get_model_by_timestamp(timestamp, verbose=True):
    p = get_full_path(timestamp)
    if None is p:
        return None

    full = saver.load_model(p, full=True, verbose=verbose)
    lm, dataset, train_stats = full[:3]
    mp, dp, tp = full[3:]
    params = {"model_params": mp, "data_params": dp, "train_params": tp}
    return lm, dataset, train_stats, params


def verify_stable_load(timestamp):
    a1 = get_model_by_timestamp(timestamp)
    a2 = get_model_by_timestamp(timestamp)
    m1, _, _, _ = a1
    m2, _, _, _ = a2

    z1 = list(m1.parameters())
    z2 = list(m2.parameters())

    msg = "two loaded models yielded different number of parameters"
    assert len(z1) == len(z2), msg

    msg = "two loaded models got different values in some parameters"
    assert False not in [torch.equal(p1, p2) for p1, p2 in zip(z1, z2)], msg

    msg = "two loaded models giving different outputs on same input - " +\
          "check models aren't in training mode?"
    assert torch.equal(m1([[1, 2]]), m2([[1, 2]])), msg

    threshold = 1e-4
    val_diff1 = check_validation(*a1)
    val_diff2 = check_validation(*a2)
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
    assert torch.equal(m1([[1, 2]]), m2([[1, 2]])), msg

    print("passed basic load checks")


def get_full_path(timestamp):
    paths = glob.glob("../saved-models/**/", recursive=True)
    paths = [p for p in paths if timestamp in p]
    if len(paths) == 2:
        if paths[0] in paths[1]:
            # timestamp appeared twice in path, remove the sub-path
            paths = paths[1:]
        elif paths[1] in paths[0]:
            # timestamp appeared twice in path, remove the sub-path
            paths = paths[1:]
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
    mytrainer = LMTrainer(lm, tp, samples_at_validation=sample)
    pltrainer.validate(mytrainer,
                       dataloaders=dataset.val_dataloader(tp.batch_size))
    if not was_training:
        # lightning turns training back on after computing validation loss!
        lm.eval()
    return mytrainer.last_val_loss


def check_validation(lm, dataset, train_stats, params):
    recorded_val_loss = train_stats["validation_loss"][-1][1]
    curr_val_loss = compute_validation(lm, dataset, params, sample=False)
    print("last recorded val loss:", recorded_val_loss,
          ", loaded model val loss:", curr_val_loss,
          ",\ndiff:", recorded_val_loss - curr_val_loss)
    return recorded_val_loss - curr_val_loss


def show_lm_attns(lm, x, layers=None, heads=None, folder_name=None):
    # x: input sequence, whether as string or as list of token indices
    z, attns = lm(x, get_attns=True)
    attns = attns.detach()
    # attns shape: batch size x n layers x n heads x in seq len x out seq len
    # batch size should be 1

    if None is not folder_name:
        folder_name = f"../attentions/{folder_name}"
        prepare_directory(folder_name)

    layers = list(range(attns.shape[1])) if None is layers else layers
    heads = list(range(attns.shape[2])) if None is heads else heads
    assert attns.shape[0] == 1  # single x - batch size 1
    attns = attns[0]
    for layer in layers:
        for h in heads:
            fig, ax = plt.subplots()
            im = ax.imshow(attns[layer][h])
            cbar = plt.colorbar(im)
            # cbar.ax.set_ylabel("sigmoid value", rotation=-90, va="bottom")
            if isinstance(x,str):
                token_ids = lm.tokenizer(x)
            else: # list or tensor - tokenizer can handle either
                token_ids = x
            tokens = lm.tokenizer.convert_ids_to_tokens(token_ids)
            ax.set_xticks(range(len(token_ids)), labels=tokens)
            ax.set_yticks(range(len(token_ids)), labels=tokens)

            plt.title(f"attn pattern for head {h} in layer {layer}")
            plt.xlabel("in dim")
            plt.ylabel("out dim")
            plt.xticks(rotation=-45, ha='left')
            fig = plt.gcf()
            fig.show()
            if None is not folder_name:
                fig.savefig(f"{folder_name}/L[{layer}]-H[{h}]")
    return z, attns
