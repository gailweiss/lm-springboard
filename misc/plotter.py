from misc.model_explorer import get_info, get_full_path, \
    get_model_by_identifier, get_all_checkpoints_by_identifier, final_chkpt
import math
import matplotlib.pyplot as plt
from misc.util import prepare_directory, print_nicely_nested
from misc.util import printer_print as print
from matplotlib.backends.backend_pdf import PdfPages
import torch


def _plt_title(title, metric_names, identifiers):
    if None is not title:
        return title
    single_identifier = len(identifiers) == 1
    if len(metric_names) == 1:
        i = 0 if isinstance(metric_names, list) else \
            list(metric_names.keys())[0]
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


def get_line_labeler(identifiers, all_metric_names, shared_ylabel):
    def line_label(identifier, metric, maybe_metric_nicknames):
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
            if metric_label.startswith(shared_ylabel):
                metric_label = metric_label[len(shared_ylabel):]
            while metric_label.startswith("/") or metric_label.startswith("|"):
                metric_label = metric_label[1:]

        if len(identifiers) == 1:
            return metric_label

        return f"{ts_label}::{metric_label}"
    return line_label


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


def get_aligned_vals(train_stats, metrics, verbose=True):
    def get_lists(m):
        a = list(zip(*train_stats[m]))
        if len(a) == 4:
            stat_syncer, n_train_samples, stat_counter, vals = a
        else:  # len(a) == 2, new version
            stat_syncer, vals = a
            stat_counter, n_train_samples = None, None
        return {"stat_syncer": stat_syncer, "n_train_samples": n_train_samples,
                "stat_counter": stat_counter, "vals": vals}
    lists = {m: get_lists(m) for m in metrics}
    dropped_syncs = {}
    sync2vals = {m: {s: v for 
                     s, v in zip(lists[m]["stat_syncer"], lists[m]["vals"])}
                 for m in metrics}
    syncs = set(sync2vals[metrics[0]])
    for m in metrics[1:]:
        syncs = syncs.intersection(set(sync2vals[m].keys()))
    for m in metrics:
        dropped_syncs[m] = set(sync2vals[m].keys()).difference(syncs)
        if verbose and (len(dropped_syncs[m]) > 0):
            print(f"skipping {len(dropped_syncs[m])} records for",
                  f"alignment of {m} in {metrics}")
            if len(dropped_syncs[m]) < 10:
                print(f"(at syncs: {sorted(list(dropped_syncs[m]))})")
    syncs = sorted(list(syncs))
    aligned_vals = {m: [sync2vals[m][s] for s in syncs] for m in metrics}
    return aligned_vals, dropped_syncs


def _ylabel(metric_names):
    if not metric_names:
        return None
    if len(metric_names) == 1:
        if isinstance(metric_names, dict):
            return metric_names[list(metric_names.keys())[0]]
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


def _longest_common_prefix(strings):
    for (i, vals) in enumerate(zip(*strings)):
        if len(set(vals)) > 1:
            return strings[0][:i]
    return strings[0][:i+1]


def _plot_metrics_setup(identifiers, metric_names_ax1, metric_names_ax2,
                        x_axis, title, stylist, ylabel_ax1, ylabel_ax2,
                        add_to_plot, add_to_pdf):
    
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

    if None is ylabel_ax1:
        ylabel_ax1 = _ylabel(metric_names_ax1)
    if None is ylabel_ax2:
        ylabel_ax2 = _ylabel(metric_names_ax2)

    if None is not add_to_plot:
        fig, ax1, ax2, artists = add_to_plot
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

    amy = [(ax1, metric_names_ax1, ylabel_ax1),
           (ax2, metric_names_ax2, ylabel_ax2)]

    line_labeler = get_line_labeler(identifiers, all_metric_names, shared_ylabel)

    return identifiers, amy, fig, ax1, ax2, artists, \
           all_metric_names, line_labeler


def _get_and_plot(ax, identifier, metric_name, metric_names, x_axis, stylist,
                  line_labeler, dropped_syncs_dict, verbose=True,
                  plot_type="scatter", max_x=None, min_x=None, max_y=None,
                  min_y=None, max_points_per_line=None):
    t_info = get_info(identifier, with_train_stats=True)
    if metric_name not in t_info["train_stats"]:
        return  # can happen for example if trying to show copy loss on several
        # identifiers but one is just pairs
    aligned_vals, ds = get_aligned_vals(
        t_info["train_stats"], [metric_name, x_axis], verbose=verbose)
    metric, x_vals = aligned_vals[metric_name], aligned_vals[x_axis]
    dropped_syncs_dict[identifier][(metric_name, x_axis)] = ds
    extra_kwargs = stylist(identifier, metric_name) if None is not stylist \
                   else {}
    if "label" not in extra_kwargs:
        extra_kwargs["label"] = line_labeler(identifier, metric_name,
                                             metric_names)
    if "marker" not in extra_kwargs:
        extra_kwargs["marker"] = "."
    return _plot(ax, x_vals, metric, plot_type=plot_type, max_x=max_x,
                 min_x=min_x, max_y=max_y, min_y=min_y,
                 extra_kwargs=extra_kwargs,
                 max_points_per_line=max_points_per_line)


def complete_plot(ax1, artists, legend_outside, legend_markerscale):
    fig = plt.gcf()
    if legend_outside:
        extra_kwargs = {'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
    else:
        extra_kwargs = {}
    ax1.legend(artists, [a.get_label() for a in artists],
               markerscale=legend_markerscale, **extra_kwargs)
    return fig


def show_and_save_plot(fig, identifiers, all_metric_names, dropped_syncs,
                       skip_show, filename, add_to_pdf, close_at_end):
    if not skip_show:
        fig.show()
    if None is not filename:
        fn = f"../plots/{filename}"
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
            print("\n\nfor each id and metric, to plot only points with",
                  "clear x axis value, dropped values at these stat-syncing",
                  "positions:\n", file=f)
            print_nicely_nested(dropped_syncs, file=f)

    if None is not add_to_pdf:
        add_to_pdf.savefig(fig, bbox_inches="tight")

    if close_at_end:
        plt.close()


def plot_metrics(identifiers, metric_names_ax1, metric_names_ax2=None,
                 title=None, filename=None, ylabel_ax1=None, ylabel_ax2=None,
                 x_axis="n_train_samples", add_to_plot=None,
                 plot_type="scatter", stylist=None, max_x=None, min_x=None,
                 max_y=None, min_y=None, legend_markerscale=10,
                 legend_outside=False, add_to_pdf=None, close_at_end=False,
                 verbose=True, skip_show=False, max_points_per_line=None):
    # identifiers can be a dict giving the identifiers special names for
    # the plot labels, or just an iterable with the identifiers of interest
    # (in which case they will be labeled by their task name). similarly
    # metrics can be given as a dict giving them nicknames

    setup = _plot_metrics_setup(identifiers, metric_names_ax1, metric_names_ax2,
                                x_axis, title, stylist, ylabel_ax1, ylabel_ax2,
                                add_to_plot, add_to_pdf)
    identifiers, amy, fig, ax1, ax2, artists, all_metric_names = setup[:-1]
    line_labeler = setup[-1]

    dropped_syncs = {i: {} for i in identifiers}
    for ax, metric_names, ylabel in amy:
        if not ax:
            continue
        for i in identifiers:
            for m in metric_names:
                artists.append(_get_and_plot(
                    ax, i, m, metric_names, x_axis, stylist, line_labeler,
                    dropped_syncs, verbose=verbose, plot_type=plot_type,
                    max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y,
                    max_points_per_line=max_points_per_line))

    fig = complete_plot(ax1, artists, legend_outside, legend_markerscale)
    show_and_save_plot(fig, identifiers, all_metric_names, dropped_syncs,
                       skip_show, filename, add_to_pdf, close_at_end)

    return fig, ax1, ax2, artists


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
