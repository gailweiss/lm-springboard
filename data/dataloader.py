import datasets
from data.syntheticdata import syntheticdatasets, SyntheticSamplesIterator
from misc.util import glob_nosquares
from misc.util import printer_print as print
from tqdm import tqdm
from collections import Counter
from data.support import RawSample, BeforeSubSeqMasker


try:
    with open("../../data-path.txt", "r") as f:
        datapath = f.readlines()[0].strip("\n")
        # path to your local data folder,
        # e.g. /Users/yourname/Documents/mydata
except Exception as e:
    print("couldnt read datapath for local datasets")
    datapath = None


def get_local_datafolder(n):
    if None is datapath:
        return None
    ps = glob_nosquares(f"{datapath}/*") +\
        glob_nosquares(f"{datapath}/*/*") +\
        glob_nosquares(f"{datapath}/*/*/*")
    dd = f"{datapath}/"
    return next((p for p in ps if dd.join(p.split(dd)[1:]) == n), None)


def get_data(data_params):
    lang_counters = None
    if data_params.dataset_name == "dummy":
        samples = verysimplesamplesreader("data", data_params)
    elif data_params.dataset_name == "wikitext":
        samples = wikitextloader()
        # not affected by test/val pct - predetermined
    elif data_params.dataset_name == "ptb":
        samples = ptbloader()
        # not affected by test/val pct - predetermined
    elif data_params.dataset_name.startswith("c4-"):  # eg c4-en 
        samples = c4loader(data_params)
    elif data_params.dataset_name in ["fineweb-ml", "wiki40b"]:
        samples, lang_counters = multiloader(data_params)
    elif data_params.dataset_name == "wikisquad":
        samples = wikisquad(data_params)
        # not affected by test/val pct - predetermined
    elif data_params.task_type == "synthetic":
        samples = syntheticdatasets.get(data_params.dataset_name)
    elif None is not get_local_datafolder(data_params.dataset_name):
        samples = verysimplesamplesreader(
                    get_local_datafolder(data_params.dataset_name),
                    data_params)
    else:
        raise Exception(f"unknown dataset: {data_params.dataset_name}")
    if data_params.debug_crop:
        data_params.debug_crop = int(data_params.debug_crop)

        def apply_crop(it):
            if isinstance(it, SyntheticSamplesIterator):
                return it.cropped(data_params.debug_crop)
            else:
                return it[:data_params.debug_crop]

        if isinstance(samples, list) or \
           isinstance(samples, SyntheticSamplesIterator):
            samples = apply_crop(samples)
        else:
            samples = {n: apply_crop(samples[n]) for n in samples}

    # if samples have not yet been placed in RawSample format, do so now
    # note: SyntheticSamplesIterator makes sure to generate RawSamples
    if isinstance(samples, list) and isinstance(samples[0], str):
        samples = [RawSample(s) for s in samples]
    elif isinstance(samples, dict):
        for dn, dl in samples.items():
            assert isinstance(dl, list), dn
            if isinstance(dl[0], str):
                samples[dn] = [RawSample(s) for s in dl]
    return samples, lang_counters


def ptbloader():
    d = datasets.load_dataset("ptb_text_only")
    d = {n: d[n]["sentence"] for n in d}
    return d  # see if this works


def c4loader(data_params):
    specifier = data_params.dataset_name[len("c4-"):] # e.g., "c4-en" has end "en"
    print(specifier)
    d = datasets.load_dataset("allenai/c4", specifier, streaming=True)
    train_d = iter(d["train"])
    val_d = iter(d["validation"])  # no test, so will make dummy test from this
    assert None is not data_params.debug_crop
    total_load = int(data_params.debug_crop)
    val_frac = data_params.val_pct / 100
    test_frac = data_params.test_pct / 100
    train_frac = 1 - (val_frac + test_frac)
    n_val_fullsamples = max(int(total_load * val_frac), 1)
    n_test_fullsamples = max(int(total_load * test_frac), 1)
    n_train_fullsamples = total_load - (n_val_fullsamples + n_test_fullsamples)
    res = {}
    res["train"] = [next(train_d)["text"] for _ in range(n_train_fullsamples)]
    val_and_test = [next(val_d)["text"] for _ in range(n_val_fullsamples +
                                                       n_test_fullsamples)]
    res["validation"] = val_and_test[:n_val_fullsamples]
    res["test"] = val_and_test[n_val_fullsamples:]
    return res


def wikitextloader():
    d = datasets.load_dataset("wikitext", "wikitext-103-v1")
    d = {n: d[n]["text"] for n in d}

    def regroup_page_lines(lines):
        res = []
        curr = None

        def is_title_line(line):
            return (line.count("=") == 2) and \
                    line.strip().startswith("=") and \
                    line.strip().endswith("=")
        for line in lines:
            if is_title_line(line):
                if None is not curr:
                    res.append(curr)
                curr = line
            elif None is curr:
                continue
            else:
                curr += line
        res.append(curr)
        return res

    return {n: regroup_page_lines(d[n]) for n in d}


def verysimplesamplesreader(path, data_params):
    paths = glob_nosquares(f"{path}/*.txt")
    all_samples = []
    for p in paths:
        print("loading samples from:", p)
        with open(p, "r") as f:
            all_lines = f.readlines()
        if data_params.lines_per_sample < 0:
            all_samples.append("".join(all_lines))
        else:
            for i in range(0, len(all_lines), data_params.lines_per_sample):
                all_samples.append("".join(
                    all_lines[i: i + data_params.lines_per_sample]))
    all_samples = [s.replace("\n","") for s in all_samples]
    print(f"loaded {len(all_samples)} samples overall")
    return all_samples


class MultiLingualLoader:
    def __init__(self, base, langs):
        assert base in ["fineweb-ml", "wiki40b"]
        self.datasets = {}
        self.langs = langs
        for lang in self.langs:
            if base == "wiki40b":
                self.datasets[lang] = datasets.load_dataset(
                    base, name=lang, streaming=True)
            elif base == "fineweb-ml":
                if not lang == "en":
                    self.datasets[lang] = datasets.load_dataset(
                        "HuggingFaceFW/fineweb-2", name=lang, streaming=True)
                else:
                    self.datasets[lang] = datasets.load_dataset(
                        "HuggingFaceFW/fineweb", name="CC-MAIN-2024-18",
                        streaming=True)
            else:
                raise NotImplementedError
        self.iterators = {lang:{"train": iter(self.datasets[lang]["train"])}
                          for lang in self.datasets}
        for lang in self.langs:
            if "test" in self.datasets[lang]:
                self.iterators[lang]["test"] = \
                    iter(self.datasets[lang]["test"])
        self.c = 0
        self.nl = len(self.langs)
        self.ran_out = set()

    def get_next_sample_full(self, split, fallback="train", attempt=0):
        # can consider implementing different proportions for data later,
        # if see dont have enough
        if attempt > self.nl:
            print("all datasets exhausted, cannot continue")
            raise NotImplementedError
        lang = self.langs[self.c % self.nl]
        self.c += 1
        it_d = self.iterators[lang]
        it = it_d.get(split, it_d[fallback])  # if no val/test available,
        # continue iterator from received train, getting samples that *havent
        # been put in own train*
        try:
            res = next(it)
            if "language" not in res:
                res["language"] = lang
            return res
        except StopIteration as e:
            if (lang, split) not in self.ran_out:
                self.ran_out.update([(lang, split)])
                print("ran out of samples for language", lang,
                      "in split", split)
            return self.get_next_sample_full(split, attempt=attempt + 1)

    def get_next_sample_small(self, split, fallback="train"):
        s = self.get_next_sample_full(split, fallback=fallback)
        return (s["text"], f"Lang[{s['language']}]")
        # fineweb "en" samples can also give s["token_count"], the number of
        # tokens they would use in gpt2. but this is not present in the
        # fineweb2 samples, so avoiding here


def multiloader(data_params):
    # interesting note: fineweb samples also have the token count for how many
    # tokens they would use in the gpt2 tokenizer. could be useful if trying to
    # really balance data down the line, for now am ignoring
    langs = data_params.langs
    print(f"\n\ngetting {data_params.dataset_name} langs: {langs}")
    d = MultiLingualLoader(data_params.dataset_name, langs)
    assert None is not data_params.debug_crop
    total_load = int(data_params.debug_crop)
    val_frac = data_params.val_pct / 100
    test_frac = data_params.test_pct / 100
    train_frac = 1 - (val_frac + test_frac)
    n_val_fullsamples = max(int(total_load * val_frac), 1)
    n_test_fullsamples = max(int(total_load * test_frac), 1)
    n_train_fullsamples = total_load - (n_val_fullsamples + n_test_fullsamples)
    fullnumbers = [n_train_fullsamples, n_val_fullsamples, n_test_fullsamples]
    print("working with data_params:", data_params)
    print("so want to get train/val/test amounts:", fullnumbers)
    res = {}
    print(f"loading {data_params.dataset_name} samples - train, val, test")
    res["train"] = [d.get_next_sample_small("train") for _ in
                    tqdm(range(n_train_fullsamples))]
    res["validation"] = [d.get_next_sample_small("validation") for _ in
                         tqdm(range(n_val_fullsamples))]
    res["test"] = [d.get_next_sample_small("test") for _ in
                   tqdm(range(n_test_fullsamples))]

    def print_sample_counts(relation):
        actual_nums = {n: len(res[n]) for n in res}
        actual_nums["total"] = sum(list(actual_nums.values()))
        print(f"\n\n{relation} balancing, got # samples: {actual_nums}")
        print(f"{relation} balancing, val frac is:",
              actual_nums["validation"] / actual_nums["total"])
        print(f"{relation} balancing, test frac is:",
              actual_nums["test"] / actual_nums["total"])

    print_sample_counts("before")
    print("\n\nnow balancing")
    final_lang_counters = assure_validation(data_params, res)
    print_sample_counts("after")

    for dataset_name in res:
        res[dataset_name] = [RawSample(s[0], lang=s[1]) for
                             s in res[dataset_name]]
    return res, final_lang_counters


def assure_validation(data_params, multilingual_samples):
    def count_languages():
        counters = {}
        for dataset_name, samples in multilingual_samples.items():
            # train, test, validation
            counters[dataset_name] = Counter()
            for seq, lang in samples:
                counters[dataset_name][lang] += 1
        return counters

    def lacking_validation(lang, verbose=True):
        n_train = counters["train"][lang]
        n_val = counters["validation"][lang]
        n_test = counters["test"][lang]
        if n_val <= 0:
            if verbose:
                print("\nno val data for lang:", lang)
            return True
        if n_test <= 0:
            if verbose:
                print("\nnote! no test data for lang:", lang)
        if ((n_train * goal_val_v_train) - n_val) > 2:
            if verbose:
                print(lang, "val pct low:", 
                      (n_train, n_val, n_test),
                      n_val / n_train, goal_val_v_train, data_params)
            return True
        if abs((n_train * goal_test_v_train) - n_test) > 2:
            if verbose:
                print("\nnote! test pct off for lang:", lang,
                      "(", (n_train, n_val, n_test),
                      n_test / n_train, goal_test_v_train, data_params, ")")
        return False

    def force_validation(lang):
        train_samples = multilingual_samples["train"]
        val_samples = multilingual_samples["validation"]
        ts_not_lang = [(s, ls) for (s, ls) in train_samples if ls != lang]
        ts_lang = [(s, ls) for (s, ls) in train_samples if ls == lang]
        n_val_missing = int((goal_val_v_train * counters["train"][lang]) -
                            counters["validation"][lang])
        print(f"forcing validation samples in {lang}, taking {n_val_missing}",
              "samples from train to val")
        n_val_missing = max(1, n_val_missing)
        # if here, definitely want to take at least one. really ,this is a
        # corner case when making small datasets to test the code
        val_samples += ts_lang[-n_val_missing:]
        multilingual_samples["train"] = ts_not_lang + \
                                        ts_lang[:-n_val_missing]

    train_pct = 100 - data_params.val_pct - data_params.test_pct
    goal_val_v_train = data_params.val_pct / train_pct
    goal_test_v_train = data_params.test_pct / train_pct
    counters = count_languages()
    print("initial language counters of dataset are:", counters)
    for lang in counters["train"]:
        if lacking_validation(lang):
            force_validation(lang)
            counters = count_languages()
            print("\nnew counters of dataset are:", counters)
            assert not lacking_validation(lang), counters
    counters = count_languages()
    print("\nfinally, language counters are:", counters)
    return counters


def wikisquad(data_params):
    w = wikitextloader()  # dictionary with train, test, val
    w = {dsn: [RawSample(s, note="Source[wiki]") for s in ds]
         for dsn, ds in w.items()}

    qmasker = BeforeSubSeqMasker("] A: [")
    squad_path = f"{datapath}/mysquad"
    # expects here the data created with script eval/sepsquad_prep.py
    contexts = {}
    for n in ["train", "test", "validation"]:
        with open(f"{squad_path}/{n}_contexts.txt", "r") as f:
            contexts[n] = [RawSample(s.replace("\n",""),
                                     note="Source[context]")
                           for s in f.readlines()]
    QAs = {}
    with open(f"{squad_path}/train_QAs.txt", "r") as f:
        QAs["train"] = [RawSample(s.replace("\n",""),
                                  note="Source[known_QA]",
                        target_masker=qmasker)
                        for s in f.readlines()]
    for n in ["test", "validation"]:
        QAs[n] = []
        for u in ["known", "unknown"]:
            # QAs on contexts that are in train (known) vs val/test (unknown)
            with open(f"{squad_path}/{n}_{u}_QAs.txt","r") as f:
                QAs[n] += [RawSample(s.replace("\n",""),
                           note=f"Source[{u}_QA]",
                           target_masker=qmasker)
                           for s in f.readlines()]

    res = w
    for n in res:
        res[n] += contexts[n]
        res[n] += QAs[n]  # train QAs

    return res
