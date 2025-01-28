import matplotlib.pyplot as plt
from misc.save_load import load_model, load_model_info, final_chkpt, \
                           models_paths
import lightning as pl
from train.trainer import Trainer
import torch
from misc.util import prepare_directory, glob_nosquares, pad
import sys
from os.path import join as path_join
from misc.util import printer_print as print
from misc.util import print_nicely_nested
import json
import itertools
from copy import deepcopy
from data.dataloader import datamodules_paths, LMDataModule
from matplotlib.backends.backend_pdf import PdfPages
import math


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


def print_config_compare(identifiers, print_padding=30, file=sys.stdout,
                         just_list_differing_keys=False):
    ids_with_missing_paramsets = \
        set([i for i in identifiers if None in get_info(i).values()])
    if ids_with_missing_paramsets:
        print("following ids have a missing paramset:", file=file)
        for i in sorted(list(ids_with_missing_paramsets)):
            missing_sets = [psn for psn in get_info(i) if
                            None is get_info(i)[psn]]
            print(f"\n{i}, missing: {missing_sets}. path: {get_full_path(i)}",
                  file=file)
        print("==\n\n", file=file)

    def params_dicts(model_info):
        return {k: vars(v) for k, v in model_info["params"].items()}
    infos = [(i, params_dicts(get_info(i))) for i in identifiers if
             i not in ids_with_missing_paramsets]

    # check all infos have same structure before reaching weird conclusions
    example_id, example_info = infos[0]
    for i, info in infos:
        if set(example_info.keys()) != set(info.keys()):
            print("in ids:", example_id, "vs", i, ",", file=file)
            print("have different structures, cant proceed:",
                  example_info.keys(), "vs", info.keys(), file=file)
            return -1
        for k in example_info:
            if set(example_info[k].keys()) != set(info[k].keys()):
                print("in ids:", example_id, "vs", i, ",", file=file)
                print(f"have different structures in {k}, cant proceed:",
                      example_info[k].keys(), "vs", info[k].keys(), file=file)
                return -1

    all_vals = deepcopy(example_info)
    for k1 in all_vals:
        for k2 in all_vals[k1]:
            all_vals[k1][k2] = set([all_vals[k1][k2]])
            for i, info in infos[1:]:
                all_vals[k1][k2].add(info[k1][k2])

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


def get_info(identifier, with_train_stats=False, verbose=True):
    # always caches, info is generally small
    cache_id = (identifier, with_train_stats)
    if cache_id not in info_cache:
        path = get_full_path(identifier, checkpoint=final_chkpt, verbose=verbose)
        if None is path:
            if verbose:
                print("could not get final checkpoint path for identifier:",
                      identifier)
            return None  # don't cache, in case file gets made/copied in soon
        res = load_model_info(path, with_train_stats=with_train_stats,
                              verbose=verbose)
        info_cache[cache_id] = res
    return info_cache[cache_id]


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


def _longest_common_prefix(strings):
    for (i, vals) in enumerate(zip(*strings)):
        if len(set(vals)) > 1:
            return strings[0][:i]
    return strings[0][:i+1]


def _ylabel(metric_names):
    if not metric_names:
        return None
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


def _plt_title(title, metric_names, identifiers):
    if None is not title:
        return title
    single_identifier = len(identifiers) == 1
    if len(metric_names) == 1:
        i = 0 if isinstance(metric_names, list) else metric_names.keys()[0]
        return metric_names[i]
    if single_identifier and isinstance(identifiers, list):
        return get_info(
            identifiers[0])["params"]["data_params"].dataset_name
    if single_identifier and isinstance(identifiers, dict):
        return identifiers[identifiers.keys()[0]]
    # multi metrics, multi identifiers
    all_ds_names = [get_info(t)["params"]["data_params"].dataset_name for
                    t in identifiers]
    if len(set(all_ds_names)) == 1:
        return all_ds_names[0]
    raise Exception("given multiple tasks and multiple metrics",
                    "please provide title for plot")


def _line_label(identifier, metric, identifiers, all_metric_names,
                ylabel, maybe_metric_nicknames):
    if isinstance(identifiers, dict):
        # identifiers have been given with labels for plot
        ts_label = identifiers[identifier]
    else:
        t_info = get_info(identifier)
        ts_label = t_info["params"]["data_params"].dataset_name

    if len(all_metric_names) == 1:
        return ts_label

    if isinstance(maybe_metric_nicknames, dict):
        metric_label = maybe_metric_nicknames[metric]
    else:
        metric_label = metric
        if metric_label.startswith(ylabel):
            metric_label = metric_label[len(ylabel):]
        while metric_label.startswith("/") or metric_label.startswith("|"):
            metric_label = metric_label[1:]

    if len(identifiers) == 1:
        return metric_label

    return f"{ts_label}::{metric_label}"


def _plot(ax, x, y, s=0.5, plot_type="scatter",
          max_x=None, min_x=None, max_y=None, min_y=None,
          extra_kwargs=None, max_points_per_line=None):
    
    def keep(vx, vy):
        return (vx < max_x) and (vx > min_x) and (vy < max_y) and (vy > min_y)
    
    max_x = max_x if None is not max_x else torch.inf
    min_x = min_x if None is not min_x else -torch.inf
    max_y = max_y if None is not max_y else torch.inf
    min_y = min_y if None is not min_y else -torch.inf
    x, y = list(zip(*[(vx, vy) for vx, vy in zip(x, y) if keep(vx, vy)]))

    if None is not max_points_per_line:
        if len(x) > max_points_per_line:
            jump = math.ceil(len(x) / max_points_per_line)
            x = x[::jump]
            y = y[::jump]
            assert len(x) <= max_points_per_line
            assert len(y) == len(x)

    extra_kwargs = {} if None is extra_kwargs else extra_kwargs
    if plot_type == "scatter":
        return ax.scatter(x, y, s=s, **extra_kwargs)
    elif plot_type == "line":
        return ax.plot(x, y, **extra_kwargs)[0]


def get_aligned_vals(train_stats, metric1, metric2, verbose=True):
    d = train_stats[metric1]
    # deprecating, no longer dealing with this:
    # if len(d[0]) == 3:  # older version
    #     n_train_samples, metric, stat_counter = list(zip(*d))    
    stat_syncer1, n_train_samples1, stat_counter1, vals1 = \
        list(zip(*train_stats[metric1]))
    stat_syncer2, n_train_samples2, stat_counter2, vals2 = \
        list(zip(*train_stats[metric2]))
    if stat_syncer1 == stat_syncer2:
        dropped_vals = (None, None)
    else:
        if verbose:
            print("skipping records for alignment of", metric1, "with",
                  metric2)
        d1 = {s: v for s, v in zip(stat_syncer1, vals1)}
        d2 = {s: v for s, v in zip(stat_syncer2, vals2)}
        s1 = set(d1.keys())
        s2 = set(d2.keys())
        diff1 = s1.difference(s2)
        diff2 = s2.difference(s1)
        dropped_vals = (diff1, diff2)
        shared = set(d1.keys()).intersection(set(d2.keys()))
        paired_metrics = [(d1[s], d2[s]) for s in sorted(list(shared))]
        vals1, vals2 = zip(*paired_metrics)
        def dropped_set_str(s):
            if len(s) < 10:
                return str(s)
            return f"[{len(s)} vals]"
        dvss = list(map(dropped_set_str, dropped_vals))
        if verbose:
            print("skipped record points are (respectively):", dvss)
    return vals1, vals2, dropped_vals


def plot_metrics(identifiers, metric_names_ax1, metric_names_ax2=None,
                 title=None, filename=None, colors=None,
                 ylabel_ax1=None, ylabel_ax2=None, x_axis="n_train_samples",
                 add_to=None, plot_type="scatter", stylist=None,
                 max_x=None, min_x=None, max_y=None, min_y=None,
                 legend_markerscale=10, legend_outside=False,
                 add_to_pdf=None, close_at_end=False, verbose=True,
                 skip_show=False, max_points_per_line=None):
    # identifiers can be a dict giving the identifiers special names for
    # the plot labels, or just an iterable with the identifiers of interest
    # (in which case they will be labeled by their task name)

    if None is not add_to_pdf:
        assert isinstance(add_to_pdf, PdfPages)
        # create as: PdfPages(filename)
        # assert here more as a reminder of what this is and how to make it

    if isinstance(identifiers, str):
        identifiers = [identifiers]
    if isinstance(metric_names_ax1, str):
        metric_names_ax1 = [metric_names_ax1]
    if isinstance(metric_names_ax2, str):
        metric_names_ax2 = [metric_names_ax2]
    if None is metric_names_ax2:
        metric_names_ax2 = []
    assert [stylist, colors].count(None) >= 1

    if None is ylabel_ax1:
        ylabel_ax1 = _ylabel(metric_names_ax1)
    if None is ylabel_ax2:
        ylabel_ax2 = _ylabel(metric_names_ax2)

    if None is not add_to:
        fig, ax1, ax2, artists = add_to
    else:
        fig, ax1 = plt.subplots()
        ax2 = None
        artists = []

    if metric_names_ax2 and (None is ax2):
        ax2 = ax1.twinx()

    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(ylabel_ax1)

    if None is not ax2:
        ax2.set_ylabel(ylabel_ax2)
        shared_ylabel = _longest_common_prefix([ylabel_ax1, ylabel_ax2])
    else:
        shared_ylabel = ylabel_ax1

    def names_as_list(names_or_dict):
        return names_or_dict if isinstance(names_or_dict, list) else \
            list(names_or_dict.keys())
    all_metric_names = names_as_list(metric_names_ax1) + \
        names_as_list(metric_names_ax2)
    plt.title(_plt_title(title, all_metric_names, identifiers))

    color_i = 0
    dropped_vals = {i: {} for i in identifiers}
    for ax, metric_names, ylabel in [(ax1, metric_names_ax1, ylabel_ax1),
                                     (ax2, metric_names_ax2, ylabel_ax2)]:
        if not ax:
            continue
        for i in identifiers:
            for m in metric_names:
                t_info = get_info(i, with_train_stats=True)
                if m not in t_info["train_stats"]:
                    continue  # eg if trying to show copy loss on several
                    # identifiers but one is just pairs
                metric, x_vals, dv = get_aligned_vals(
                    t_info["train_stats"], m, x_axis, verbose=verbose)
                dropped_vals[i][(m, x_axis)] = dv
                extra_kwargs = stylist(i, m) if None is not stylist else {}
                if "color" not in extra_kwargs and None is not colors:
                    extra_kwargs["color"] = colors[color_i]
                    color_i += 1
                if "label" not in extra_kwargs:
                    extra_kwargs["label"] = \
                        _line_label(i, m, identifiers, all_metric_names,
                                    shared_ylabel, metric_names)
                if "marker" not in extra_kwargs:
                    extra_kwargs["marker"] = "."
                artists.append(
                    _plot(ax, x_vals, metric, plot_type=plot_type,
                          max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y,
                          extra_kwargs=extra_kwargs,
                          max_points_per_line=max_points_per_line))

    if legend_outside:
        extra_kwargs = {'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
    else:
        extra_kwargs = {}
    ax1.legend(artists, [a.get_label() for a in artists],
               markerscale=legend_markerscale, **extra_kwargs)

    fig = plt.gcf()
    if not skip_show:
        fig.show()
    if None is not filename:
        fn = f"../metrics/{filename}"
        directory = '/'.join(fn.split('/')[:-1])
        prepare_directory(directory)
        fig.savefig(f"{fn}.png", bbox_inches="tight")
        with open(f"{fn}.txt", "w") as f:
            print(f"plot in {fn} made from identifiers:{identifiers}\n",
                  f"and metrics:\n{all_metric_names}",
                  file=f)
            print("identifier full paths:\n", file=f)
            for t in identifiers:
                print(t, "\t:", get_full_path(t), "\n", file=f)
            print("\n\nfor each id and metric, to take only points with",
                  "clear x axis value, dropped values at these stat-syncing",
                  "positions: (format: #metric points dropped,",
                  "#x-axis ({x_axis}) points dropped)\n", file=f)
            print_nicely_nested(dropped_vals, file=f)

    if None is not add_to_pdf:
        add_to_pdf.savefig(fig, bbox_inches="tight")

    if close_at_end:
        plt.close()
    return fig, ax1, ax2, artists


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


def show_lm_attns(identifier, x, layers=None, heads=None, store=False,
                  checkpoint=final_chkpt, cache=True):

    chkpt = get_model_by_identifier(identifier, checkpoint=checkpoint,
                                    verbose=False, with_data=False,
                                    cache=cache)
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
        folder_name = f"../attentions/{task_name}/{identifier}/" +\
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


def show_head_progress(identifier, x, layer, head, cache=True, store=False):
    res = get_all_checkpoints_by_identifier(identifier, verbose=False,
                                            cache=cache, with_data=False)

    task_name = res["params"]["data_params"].dataset_name
    alphabet = \
        list(res["models"][final_chkpt]["lm"].tokenizer.get_vocab().keys())

    if store:
        folder_name = f"../attentions/{task_name}/{identifier}/" +\
                      f"heads-over-time/L[{layer}]-H[{head}]"
        prepare_directory(folder_name)

    f = open(f"{folder_name}/notes.txt", "w") if store else sys.stdout

    print(f"showing attn patterns for model trained on task: [ {task_name} ].",
          file=f)
    print("\ttask alphabet:", "".join(sorted(alphabet)), file=f)
    print("\tinput sequence:", x, file=f)

    if store:
        f.close()

    models_by_nsamples = {d["train_stats"]["total_train_samples"]: d["lm"] for
                          n, d in res["models"].items()}

    for nsamples in sorted(list(models_by_nsamples.keys())):
        _, _, fig = show_lm_attns(identifier, x, layers=[layer], heads=[head],
                                  checkpoint=str(nsamples), cache=cache)
        if store:
            fig.savefig(f"{folder_name}/{nsamples}")


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
