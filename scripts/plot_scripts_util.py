import os
import misc.model_explorer as me
from tqdm import tqdm
import misc.util as util
import json
from dataclasses import asdict

def prepare_folder_path(args, obtained_ids, min_date, max_date, type_name):
    # make a output folder name, use os.path.join
    sn = args.subfolder_note
    if args.subfolder_substr:
        sn += f"--from[{args.subfolder_substr}]"
    pathbits = ["..", "pdfs", type_name,
        f"{args.dataset_name}-crop[{args.debug_crop}]"]
    pathbits += [sn, f"{min_date}-thru-{max_date}"]
    res = os.path.join(*pathbits)
    util.prepare_directory(res)
    return res


def note_args_and_models(notes_folder, args, all_ids, min_date, max_date):
    with open(os.path.join(notes_folder, "calling-args.txt"), "w") as f:
        print("called plot scripts with args:\n\n", file=f)
        util.print_nicely_nested(vars(args), file=f)
    with open(os.path.join(notes_folder, "main-notes.txt"), "w") as f:
        print("called plot scripts with args:\n\n", file=f)
        util.print_nicely_nested(vars(args), file=f)
        print("\n\ngot trained models with dates ranging from",
              min_date, "to", max_date, "\n\n", file=f)
        for i in sorted(all_ids):
            print(i, file=f)
        print("\n\nmodel full paths:", file=f)
        for i in sorted(all_ids):
            print(me.get_full_path(i), file=f)

    def ps2ds(i, params_dict):
        return {n: asdict(p) for n, p in params_dict.items()}

    params_dicts = {i: ps2ds(i, me.get_info(i, with_train_stats=False)["params"])
                    for i in all_ids}

    with open(os.path.join(notes_folder, "plotted-models.json"), "w") as f:
        json.dump(params_dicts, f)

    with open(os.path.join(notes_folder, "plotted-models.txt"), "w") as f:
        util.print_nicely_nested(params_dicts, file=f)


def all_train_stats(m_id):
    return set(metrics_cache[m_id].keys())


class PlotTracker:
    def __init__(self):
        self.curr_id = -1
        self.id2models = {}
        self.id2metrics = {}
        self.hide_plot_ids = False
        self.id2lastvals = {}

    def finish_plot_title(self, model_ids, metrics, title):
        self.curr_id += 1
        self.id2models[self.curr_id] = model_ids
        self.id2metrics[self.curr_id] = metrics
        if not self.hide_plot_ids:
            title = f"plot #{self.curr_id}\n{title}"
        print("preparing title for plot #", self.curr_id)
        return title

    def note_last_plots_last_vals(self, last_vals):
        assert self.curr_id not in self.id2lastvals
        self.id2lastvals[self.curr_id] = last_vals


plot_tracker = PlotTracker()


def get_ranges(args):
    tbr = {"min": args.train_batch_x_start, "max": args.train_batch_x_stop}
    vr = {"min": args.val_x_start, "max": args.val_x_stop}
    return {"train_batch": tbr, "val": vr}


def make_extra_plot_kwargs(args, eval_name):
    ranges = get_ranges(args)
    r = ranges[eval_name]
    return {"legend_outside": True,
            "legend_markerscale": 1,
            "plot_type": "line",
            "min_x": r["min"],
            "max_x": r["max"],
            "verbose": args.verbose_plot_metrics,
            "close_at_end": True,
            "skip_show": True,
            "max_points_per_line": args.max_points_per_line,
            "no_caching": True,
            "preloaded_metrics": metrics_cache,
            "x_axis": args.x_axis}


def print_plot_models_and_notes(folder_path):
    notes_folder = os.path.join(folder_path, "notes")
    with open(os.path.join(notes_folder, "plot-models.json"), "w") as f:
        json.dump(plot_tracker.id2models, f)
    with open(os.path.join(notes_folder, "plot-metrics.json"), "w") as f:
        json.dump(plot_tracker.id2metrics, f)
    with open(os.path.join(notes_folder, "plot-data.txt"), "w") as f_short, \
         open(os.path.join(notes_folder, "plot-data-full.txt"), 
              "w") as f_full:
        for pid in sorted(list(plot_tracker.id2models.keys())):
            m_ids = sorted(plot_tracker.id2models[pid])
            metrics = sorted(plot_tracker.id2metrics[pid])
            for ff, jldk in [(f_full, False), (f_short, True)]:
                print(f"\n\n===plot #{pid}===\n\n", file=ff)
                print("used metrics:\n\n",
                      "\n".join([f"\t{i}" for i in metrics]),
                      file=ff)
                print("used ids:\n\n", "\n".join([f"\t{i}" for i in m_ids]),
                      file=ff)
                me.print_config_compare(m_ids, file=ff,
                                        just_list_differing_keys=jldk)
    with open(os.path.join(notes_folder, "plot-last-vals.txt"), "w") as f:
        for plot_id in plot_tracker.id2lastvals:
            print(f"\n== plot #{plot_id} , last vals: ==\n", file=f)
            d = plot_tracker.id2lastvals[plot_id]
            # d: dictionary with format {label: (last_sync, last_val)}
            # lowest last val first:
            labels = sorted(list(d.keys()), key=lambda l:d[l][1])
            for l in labels:
                print(util.pad(d[l], 35), l, file=f)

        util.print_nicely_nested(plot_tracker.id2lastvals, file=f)


metrics_cache = {}


def fill_metrics_cache(args, all_ids):
    def keep(m_id, metname):
        if metname == args.x_axis:
            return True  # train samples for x axis
        if "mean" in metname or "acc" in metname:
            return True
        return False

    print("loading relevant metrics")
    for i in tqdm(all_ids):
        metrics_cache[i] = {}
        inf = me.get_info(i, with_train_stats=True, dont_cache=True)
        ts = inf["train_stats"]
        ts = {k: v for k, v in ts.items() if keep(i, k)}
        metrics_cache[i] = ts
    # basically: on one hand loading and caching all train_stats in 
    # memory gets to several gigabytes and slows everything down, so asking
    # plot_metrics to avoid caching. on other hand this will call plot_metrics
    # on many ids many times, so do want to avoid reading those large jsons
    # too many times, so trying to cache what i think will be relevant (which
    # is less than everything, so hopefully lighter than a full cache),
    # hopefully reducing most calls to load


base_diffs = ["random_seed", "max_sample_tokens", "checkpoint_every",
              "val_check_epoch_frac", "total_train_samples", "total_samples"]


def refine_ids_choice(args, all_ids, extra_allowed_differences):
    allowed_differences = base_diffs + extra_allowed_differences
    return me.refine_model_ids_choice(all_ids,
        subfolder_substr=args.subfolder_substr,
        allowed_differences=allowed_differences)


def select_ids(all_ids):
    all_ids = sorted(list(all_ids))
    def print_all_ids():
        print(f"have {len(all_ids)} ids:\n\n")
        for i, mid in enumerate(all_ids):
            print(f"{i}:\t\t{mid}")
    did_nothing = False
    while(not did_nothing):
        print_all_ids()
        did_nothing = True
        i = util.constrained_input("inspect id? int or n: ",
            ["n"] + [str(i) for i in range(len(all_ids))])
        if i != "n":
            did_nothing = False
            i = int(i)
            p = me.get_info(all_ids[i], with_train_stats=False)["params"]
            p = ps2ds(p)
            print(f"{i}:\t\t{all_ids[i]}\n")
            for pn, pd in p.items():
                print(f"\n{pn}:\n")
                util.print_nicely_nested(pd)
        i = util.constrained_input("remove id? int or n: ",
            ["n"] + [str(i) for i in range(len(all_ids))])
        if i != "n":
            did_nothing = False
            i = int(i)
            all_ids = all_ids[:i] + all_ids[i:]
    return all_ids


def setup(args, desc, extra_allowed_differences, ids_filter, stylists,
          type_name):
    plot_tracker.hide_plot_ids = args.hide_plot_ids
    all_ids = me.all_identifiers_with_configs(
        desc, min_date=args.min_date, max_date=args.max_date,
        matching_datamodule_id=args.d_id)
    all_ids = refine_ids_choice(args, all_ids, extra_allowed_differences)
    all_ids = select_ids(all_ids)
    if None is not ids_filter:
        all_ids = ids_filter(args, all_ids)
    print("found", len(all_ids), "relevant models matching the description")

    for s in stylists:
        s.prepare(args, all_ids)
    
    fill_metrics_cache(args, all_ids)

    obtained_times = [me.identifier2timestamp(i) for i in all_ids]
    obtained_times = sorted(obtained_times)

    min_date, max_date = obtained_times[0], obtained_times[-1]

    folder_path = prepare_folder_path(args, all_ids, min_date, max_date,
                                      type_name)
    
    notes_folder = os.path.join(folder_path, "notes")
    util.prepare_directory(notes_folder)
    note_args_and_models(notes_folder, args, all_ids, min_date, max_date)

    return all_ids, folder_path
