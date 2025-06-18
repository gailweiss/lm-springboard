from datasets import load_dataset


ds = load_dataset("rajpurkar/squad")


def squadformat(example):
    q = example['question']
    answers = example['answers']['text']
    a = sorted(answers, key=len)[0]
    return example['context'], f"Q: [{q}] A: [{a}]"


def _mysquaddata(frac_ID_QA, file):
    ds = load_dataset("rajpurkar/squad")
    train_pairs = [squadformat(s) for s in ds["train"]]
    val_pairs = [squadformat(s) for s in ds["validation"]]
    n = len(val_pairs) // 2
    val_pairs, test_pairs = val_pairs[:n], val_pairs[n:]
    contexts = {
        "train": set([p[0] for p in train_pairs]),
        "validation": set([p[0] for p in val_pairs]),
        "test": set([p[0] for p in test_pairs])
    }
    # apply set(...) because each context appears multiple times, once with
    # each of its questions
    pre_counts = {dsn: len(cs) for dsn, cs in contexts.items()}
    print("got # contexts:", pre_counts,
          "before removing cross-set duplicates", file=file)
    contexts["validation"] = contexts["validation"].difference(
                             contexts["train"])
    contexts["test"] = contexts["test"].difference(contexts["train"])
    mid_counts = {dsn: len(cs) for dsn, cs in contexts.items()}
    print("got # contexts:", mid_counts,
          "after removing from-train duplicates", file=file)
    msg = "rethink question known/unknown annotation"
    assert (pre_counts == mid_counts), msg
    contexts["test"] = contexts["test"].difference(contexts["validation"])
    post_counts = {dsn: len(cs) for dsn, cs in contexts.items()}
    print("got # contexts:", post_counts, "after removing val copies in test",
          file=file)
    n = int(frac_ID_QA * len(train_pairs))
    QAs = {
        "train": set([p[1] for p in train_pairs[:-(2 * n)]]),
        # unseen questions on seen contexts:
        "validation_known": set([p[1] for p in train_pairs[-(2 * n): -n]]),
        # unseen questions on likely unseen contexts:
        "validation_unknown": set([p[1] for p in val_pairs]),
        # unseen questions on seen contexts:
        "test_known": set([p[1] for p in train_pairs[-n:]]),
          # unseen questions on likely unseen contexts:
        "test_unknown": set([p[1] for p in test_pairs])
    }
    return contexts, QAs

def mysquaddata(frac_ID_QA=0.1, save_path=None):
    if None is not save_path:
        util.prepare_directory(save_path)
        with open(f"{save_path}/notes.txt", "w") as f:
            contexts, QAs = _mysquaddata(frac_ID_QA, f)
            print(f"took {frac_ID_QA} train questions each",
                  f"({2 * frac_ID_QA} total)",
                  "to act as val and test questions with known contexts",
                  file=f)
            for n, d in {"QAs": QAs, "contexts": contexts}.items():
                for k, v in d.items():
                    with open(f"{save_path}/{k}_{n}.txt", "w") as f2:
                        f2.writelines([f"{vv}\n" for vv in v])
                    print(f"{k}_{n} # samples: {len(v)}", file=f)
        return contexts, QAs
    else:
        return _mysquaddata(frac_ID_QA, None)


# use: load, then run mysquaddata with the desired frac and save path