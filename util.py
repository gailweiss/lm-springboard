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


class GlobalTimingDepth:
    def __init__(self):
        self.d = -1


TheTimingDepth = GlobalTimingDepth()


def timed(f):
    def timed_f(*a, _timed_f_silent=False, **kw):
        TheTimingDepth.d += 1
        start = process_time()
        res = f(*a, **kw)
        total = process_time() - start
        if not _timed_f_silent:
            print("TIME:", (" " * 4 * TheTimingDepth.d) + f.__name__, "took:",
                  total, "s")
        TheTimingDepth.d -= 1
        return res
    return timed_f


def in_try(f):
    def tried_f(*a, **kw):
        try:
            return f(*a, **kw)
        except Exception as e:
            print(f"\n\n\n!!!!!!\n" +
                  f"running {f.__name__} hit exception:\n {e}\n\n" +
                  "!!!!!!!!!!!!!\n\n\n")
            return e
    return tried_f


def timestamp():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


def pad(s, width=5):
    total = width - len(str(s))
    lohalf = int(np.floor(total / 2))
    hihalf = total - lohalf
    return " " * lohalf + str(s) + " " * hihalf


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
        # print("making path:",path)
        os.makedirs(path)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


def is_all_type(lst, t):
    return False not in [isinstance(v, t) for v in lst]


def print_nicely_nested(d, file=sys.stdout, skip_first=True, indent=" " * 8,
                        hide_braces=True):

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
            print(f"{indents(n_indents)}{a}", file=file)
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
