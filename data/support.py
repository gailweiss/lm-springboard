from misc.util import printer_print as print

class NoMask:
    def __init__(self):
        pass
    def prep(self, tokenizer):
        pass
    def __call__(self, indices):
        return None
    def __repr__(self):
        return "nomask"

nomask = NoMask()

class RawSample:
    # for use in creating datasets, before processing in the datamodule
    def __init__(self, seq, lang=None, note=None, target_masker=nomask):
        self.seq = seq
        self.lang = lang
        self.note = note
        self.target_masker = target_masker

    def __repr__(self):
        d = {a: getattr(self, a) for a in ["seq", "lang", "note", "target_masker"]}
        return f"RawSample({d})"


class TokenizedSample:
    # for use in creating datasets, after tokenizing but before final collation
    # in the datamodule
    def __init__(self, indices, target_mask=None):
        self.indices = indices
        self.target_mask = target_mask
        # 1 if off (dont train to predict), 0 if on (do train to predict)

    def __len__(self):
        return len(self.indices)


class BeforeSubSeqMasker:
    def __init__(self, subseq):
        self.subseq = subseq
        self.prepped = False
        self.num_calls = 0

    def prep(self, tokenizer):
        self.tokenizer = tokenizer

    def show_results(self, indices, res):
        if self.num_calls > 4:
            # exit(0)
            return
        self.num_calls += 1
        print("masker showing results")
        print(self.num_calls)
        print("input:", self.tokenizer.convert_ids_to_nice_string(indices))
        mindices = [0 if m else i for i, m in zip(indices, res)]
        print("masked input:",
              self.tokenizer.convert_ids_to_nice_string(mindices))
        print("masked input tokens:",
              self.tokenizer.convert_ids_to_tokens(mindices))

    def __call__(self, indices):
        def get_n_tocontain():
            TOO_LOW, COULD_BE = 1, 2
            def check_i(i):
                s = self.tokenizer.convert_ids_to_nice_string(indices[:i])
                if self.subseq not in s:
                    return TOO_LOW
                if self.subseq in s:
                    return COULD_BE
            if not indices:
                return 0
            if not self.subseq:  # empty subseq
                return 0
            lo, hi = 0, len(indices)  # now lo is always definitely too low
            while lo < hi:
                i = lo + max((hi - lo) // 2, 1)
                a = check_i(i)
                if a == TOO_LOW:
                    lo = i
                if a == COULD_BE:
                    if i == (lo + 1):
                        return i  # know that lo is too low
                    else:
                        hi = i
            return len(indices)  # not found
        len_to_contain = get_n_tocontain()
        res = ([1] * len_to_contain) + ([0] * (len(indices) - len_to_contain))
        # self.show_results(indices, res)
        return res

    def __repr__(self):
        return f"BeforeSubSeqMasker: {self.subseq}"