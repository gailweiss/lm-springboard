import argparse
import os
import misc.model_explorer as me
from misc.plotter import plot_metrics
import misc.util as util
import json
from matplotlib import colormaps
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from copy import copy
import itertools
from misc.plot_scripts_util import plot_tracker, get_ranges, \
    make_extra_plot_kwargs, print_plot_models_and_notes, setup
from tqdm import tqdm
from dataclasses import asdict


parser = argparse.ArgumentParser()
parser.add_argument('--debug-crop', type=int, default=None)
parser.add_argument('--dataset-name', type=str, default="wikitext")
parser.add_argument('--max-seq-len', type=int, default=512)
parser.add_argument('--early-stop-nsamples', type=int, default=30000)
# if want to be v specific about dataset, can use this directly:
parser.add_argument('--d-id', type=str, default=None)
parser.add_argument('--min-date', type=str, default=None)
# e.g. 2024-10-17--00-00-00
parser.add_argument('--max-date', type=str, default=None)
parser.add_argument('--subfolder-note', type=str, default="")
parser.add_argument('--hide-plot-ids', action='store_true')
parser.add_argument('--train-batch-x-start', type=int, default=-1)
parser.add_argument('--train-batch-x-stop', type=int, default=500)
parser.add_argument('--val-x-start', type=int, default=1)
parser.add_argument('--val-x-stop', type=int, default=None)
parser.add_argument('--verbose-plot-metrics', action='store_true')
parser.add_argument('--max-points-per-line', type=int, default=None)
parser.add_argument('--subfolder-substr', default="")
parser.add_argument('--x-axis', default="n_opt_steps")


class Stylist:
    def __init__(self):
        self.is_by_param = False

    def set_by_param(self, p, a, vals):
        self.is_by_param = True
        self.param = (p, a)
        self.n_vals = max(len(vals), 2)
        self.i2val = sorted(list(vals))
        self.val2i = {v: i for i, v in enumerate(self.i2val)}

    def prepare(self, args, all_ids):
        self.is_by_param = False
        self.n_ids = max(len(all_ids), 2)  # max to avoid division by 0 below
        self.i2id = sorted(list(all_ids))
        self.id2i = {mid: i for i, mid in enumerate(self.i2id)}

    def __call__(self, mid, mname):
        cmap = colormaps["nipy_spectral"]
        if self.is_by_param:
            p, a = self.param
            val = asdict(me.get_info(mid)["params"][p])[a]
            color = cmap(self.val2i[val] / (self.n_vals - 1))
            label = f"{a}: {val} -- {mid}"
        else:
            color = cmap(self.id2i[mid] / (self.n_ids - 1))
            label = mid
        return {"markersize": 0, "color": color, "label": label}


stylist = Stylist()


def nice_stat_name(fullstat):
    return fullstat  # consider making nicer


def all_stats_with(m_id, must_contain):
    if "loss" in must_contain:
        must_contain += ["mean"]
    stats = me.get_info(m_id, with_train_stats=True)["train_stats"].keys()
    return [s for s in stats if False not in [n in s for n in must_contain]]


def draw_all(args, all_ids, pdfname):
    with PdfPages(pdfname) as pdf:
        for stat in ["loss", "acc"]:
            for eval_name in get_ranges(args):  # train_batch, val
                extras = make_extra_plot_kwargs(args, eval_name)
                for fullstat in all_stats_with(all_ids[0], [eval_name, stat]):
                    shortstat = nice_stat_name(fullstat)
                    title = plot_tracker.finish_plot_title(
                        all_ids, [fullstat], shortstat)
                    p = plot_metrics(all_ids, [fullstat], ylabel_ax1=shortstat,
                        title=title, stylist=stylist, add_to_pdf=pdf, **extras)
                    plot_tracker.note_last_plots_last_vals(p["last_val"])


def draw_param_grouped_lines(args, all_ids, folder):
    d = me.get_configs_values(all_ids)
    changing_params = []
    for p in d:
        for a in d[p]:
            if len(d[p][a]) > 1:
                changing_params.append(((p, a), sorted(list(d[p][a]))))
    for (p, a), vals in changing_params:
        stylist.set_by_param(p, a, vals)
        pdfname = os.path.join(folder, f"by-{a}.pdf")
        draw_all(args, all_ids, pdfname)


def draw_ungrouped_lines(args, all_ids, folder):
    stylist.prepare(args, all_ids)
    pdfname = os.path.join(folder, "by-models.pdf")
    draw_all(args, all_ids, pdfname)


def run_main():
    args = parser.parse_args()
    desc = {"data_params": {"debug_crop": args.debug_crop,
                            "dataset_name": args.dataset_name},
            "train_params": {"early_stop_nsamples": args.early_stop_nsamples},
            "model_params": {"max_seq_len": args.max_seq_len}}
    hyperparam_examples = ["weight_decay", "lr"]
    # will probably extend with time, for now handled manually in refine loop
    all_ids, folder_path = setup(
        args, desc, hyperparam_examples, None, [stylist], "hyperparams")

    draw_ungrouped_lines(args, all_ids, folder_path)
    draw_param_grouped_lines(args, all_ids, folder_path)
    print_plot_models_and_notes(folder_path)


if __name__ == "__main__":
    matplotlib.use('agg')  # avoid memory leaks
    run_main()
