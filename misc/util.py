import numpy as np
import random
import itertools
import os
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
from time import time, process_time
import gzip
import subprocess
import multiprocessing
import torch
from datetime import datetime
from functools import reduce
import glob
import math


class GlobalTimingDepth:
    def __init__(self):
        self.d = -1


TheTimingDepth = GlobalTimingDepth()


class Printer:
    # can probably do this with logging module but long docs and just need this
    def __init__(self):
        self.other_files = []

    def add_output_file(self, file):
        self.other_files = list(set(self.other_files + [file]))

    def remove_output_file(self, file):
        self.other_files = [f for f in self.other_files if f is not file]

    def print(self, *args, **kwargs):
        print_files = self.other_files
        print_files = print_files + [kwargs.get("file", sys.stdout)]
        print_files = set(print_files)  # in case of duplicates
        kwargs = {n: v for n, v in kwargs.items() if not n == "file"}
        for f in print_files:
            if not f.closed:
                print(*args, file=f, **kwargs)


printer = Printer()


def printer_print(*args, **kwargs):
    printer.print(*args, **kwargs)


def timed(f):
    def timed_f(*a, _timed_f_silent=False, **kw):
        TheTimingDepth.d += 1
        start = process_time()
        res = f(*a, **kw)
        total = process_time() - start
        if not _timed_f_silent:
            printer_print("TIME:", (" " * 4 * TheTimingDepth.d) + f.__name__,
                          "took:", total, "s", flush=True)
        TheTimingDepth.d -= 1
        return res
    return timed_f


def in_try(f):
    def tried_f(*a, **kw):
        try:
            return f(*a, **kw)
        except Exception as e:
            printer_print(
                f"\n\n\n!!!!!!\n" +
                f"running {f.__name__} hit exception:\n {e}\n\n" +
                "!!!!!!!!!!!!!\n\n\n")
            return e
    return tried_f


def pad(s, width=5, place="center"):
    s = str(s)
    total = width - len(s)
    if place == "left":
        return s + (" " * total)
    if place == "right":
        return (" " * total) + s
    lohalf = int(np.floor(total / 2))
    hihalf = total - lohalf
    return (" " * lohalf) + s + (" " * hihalf)


def binsearch(val, lst):
    # separate the recursion from the main call just so i can time the main one
    # without 100 timing prints
    return _binsearch(val, lst)


def _binsearch(val, lst):
    # returns i such that l[i-1]<=val<l[i], or 0 if l is empty
    TOO_LOW, FOUND, TOO_HIGH = 1, 2, 3

    def check_i(i):
        if lst[i] <= val:
            return TOO_LOW
        if lst[i] > val:
            if i == 0 or lst[i-1] <= val:
                return FOUND
        return TOO_HIGH
    if not lst:
        return 0
    i = int(len(lst) / 2)
    a = check_i(i)
    if a == FOUND:
        return i
    if a == TOO_LOW:
        return i + _binsearch(val, lst[i:])
    return _binsearch(val, lst[:i])


def pick_index_from_distribution(vector):  # all vector values must be >=0
    if isinstance(vector, torch.Tensor):
        assert len(vector.shape) == 1
        rel_indices = (vector > 0).nonzero().view(-1).tolist()
        vector = vector[rel_indices]
        total = vector.sum()
        sums = torch.cumsum(vector, -1)
    else:
        rel_indices = [i for i, v in enumerate(vector) if v > 0]
        vector = [v[i] for i in rel_indices]
        total = np.sum(vector)
        sums = np.cumsum(vector)
    sums = sums.tolist()
    num = random.uniform(0, total)
    return rel_indices[binsearch(num, sums)]


# similar functions, using Path:
# from pathlib import Path
# Path(path).mkdir(exist_ok=True)
# Path(folder_name).exists()
def prepare_directory(path):
    if len(path) == 0:
        return
    if not os.path.exists(path):
        # printer_print("making path:",path)
        os.makedirs(path)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


def get_probably_unique(n_digits=4):
    n = random.randint(0, math.pow(10, n_digits) - 1)
    return f"{get_timestamp()}---{n}"
    # sometimes get jobs scheduled at exact same second,
    # so timestamp not enough for identifiers


def is_all_type(lst, t):
    return False not in [isinstance(v, t) for v in lst]


def print_nicely_nested(d, file=sys.stdout, indent=" " * 8):

    def indents(n):
        return n * indent
    atom_types = [int, str, float, bool]

    def is_atom_type(v):
        for a in atom_types:
            if isinstance(v, a):
                return True
        return None is v

    def is_simple(d):
        if is_atom_type(d):
            return True
        if isinstance(d, list) or isinstance(d, tuple):
            return False not in [is_atom_type(v) for v in d]
        return False

    def print_nested(d, n_indents=0):
        def print_simply(a):
            printer_print(f"{indents(n_indents)}{a}", file=file)
        if is_simple(d):
            print_simply(d)
            return
        if isinstance(d, list) or isinstance(d, tuple):
            for t in atom_types:
                if is_all_type(d, t):
                    print_simply(d)
                    return
            print_simply("[" if isinstance(d, list) else "(")
            print_nested(d[0], n_indents+1)
            for v in d[1:]:
                print_simply(",")
                print_nested(v, n_indents+1)
            print_simply("]" if isinstance(d, list) else ")")
            return
        elif isinstance(d, dict):
            keys = sorted(list(d.keys()))
            for k in keys:
                printed = False
                if is_simple(d[k]):
                    print_simply(f"{k} : {d[k]}")
                    printed = True
                if printed:
                    continue
                print_simply(f"{k} :")
                print_nested(d[k], n_indents + 1)
            return
        else:  # i dont know what this is
            print_simply(d)
    print_nested(d)


def get_parent_module(module, full_param_name):
    names = full_param_name.split(".")[:-1]
    # last one is the parameter name
    return reduce(getattr, names, module)


def get_nested_param(module, full_param_name):
    names = full_param_name.split(".")
    return reduce(getattr, names, module)


def glob_nosquares(p, **kw):
    # glob interprets pair of square brackets as describing range of tokens,
    # so cannot directly glob a filename with square bracket pairs in it.
    # but can turn each [ and ] into the range containing only [ or ], i.e.
    # [[] and []]. use § as mediator in replace operation to avoid unwanted
    # applications in second replace
    p = p.replace("[", "§[§").replace("]", "§]§")
    p = p.replace("§[§", "[[]").replace("§]§", "[]]")
    return glob.glob(p, **kw)


def apply_dataclass(dataclass, given_attrs, forgiving=False,
                    takes_extras=False, convert_lists_to_tuples=True,
                    name_changes=None, verbose=True):

    if convert_lists_to_tuples:
        given_attrs = {n: tuple(v) if isinstance(v, list) else v for
                       n, v in given_attrs.items()}
    if name_changes:
        for old_name, new_name in name_changes:
            if old_name in given_attrs:
                assert new_name not in given_attrs
                given_attrs[new_name] = given_attrs[old_name]
                del given_attrs[old_name]

    allowed = list(vars(dataclass()).keys())
    extra_attrs = {n: v for n, v in given_attrs.items() if n not in allowed}
    expected_attrs = {n: v for n, v in given_attrs.items() if n in allowed}
    if extra_attrs and not forgiving:
        if verbose:
            print("unexpected properties:", extra_attrs, "--not loading params")
        return None
    res = dataclass(**expected_attrs)
    if takes_extras:
        [setattr(res, n, v) for n, v in extra_attrs.items()]
    return res
