import torch
import os
import json
from misc.util import timed

try:
    with open("paths/evals-paths.txt", "r") as f:
        path_opts = f.readlines()
        path_opts = [p.strip("\n") for p in path_opts if not
                             p.startswith("#")]
except Exception as e:
    print("couldnt find eval dataset paths")
    path_opts = []


# assumes hellaswag taken from repository: 
# https://github.com/rowanz/hellaswag


def get_hellaswag_subset(lm, subset="validation", path=None):
    if subset == "validation":
        subset = "val"  # hellaswag files use val not validation
    if None is path:
        for p in path_opts:
            if os.path.exists(p):
                path = p
                break
    if None is path:
        raise Exception("Couldn't find hellaswag data, tried:", path_opts)
    if not hasattr(lm, "hellaswag_dl_cache"):
        lm.hellaswag_dl_cache = {}
        # prepare to cache after tokenizing etc specifically for this lm
    if subset in lm.hellaswag_dl_cache:
        return lm.hellaswag_dl_cache[subset]
        # return cached if already tokenized specifically for this lm
    # verify using the hellaswag filenames:
    assert subset in ["train", "val", "test"], subset
    filename = f"{path}/hellaswag_{subset}.jsonl"
    with open(filename, "r") as f:
        qs = [json.loads(line) for line in f]
    qs = qs[:200]  # quick crop
    all_n_opts = [len(s["endings"]) for s in qs]
    assert len(set(all_n_opts)) == 1
    n_opts = all_n_opts[0]
    full_seqs = [(s["ctx"] + " " + s["endings"][i]) for s in qs
                 for i in range(n_opts)]
    # make sure samples are ordered as i assume, i.e., each sample's full
    # options, in order:
    assert full_seqs[0][:20] == full_seqs[1][:20], full_seqs[:5]
    s0 = qs[0]
    assert full_seqs[1] == (s0["ctx"] + " " + s0["endings"][1]), full_seqs[:5]
    y = torch.Tensor([s["label"] for s in qs]).long()
    return full_seqs, y, n_opts
    # original sequences, possible answers, note on num possible answers
    

@timed
def hellaswag_eval(lm, subset="validation", path=None):
    full_seqs_or_dl, y, n_opts = get_hellaswag_subset(lm, subset=subset,
                                                      path=path)
    pp = lm.perplexities(full_seqs_or_dl, per_token=True, dummy_res=0)
    # pp will also provide, in "dl", the dataloader loading the tokenized seqs
    lm.hellaswag_dl_cache[subset] = (pp["dl"], y, n_opts)
    # save time by caching and retrieving same tokenized dl from here on
    ptr = pp["per_token_res"]
    z = ptr.sum(dim=1).view(-1, n_opts).argmin(dim=1)
    # n questions X n answers, perplexity per answer
    acc = (y == z).sum() / len(y)
    return acc.item()
