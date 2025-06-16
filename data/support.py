class NoMask:
    def __init__(self):
        pass
    def prep(self, tokenizer):
        pass
    def __call__(self, indices):
        return None

nomask = NoMask()

class RawSample:
    # for use in creating datasets, before processing in the datamodule
    def __init__(self, seq, lang=None, target_masker=nomask):
        self.seq = seq
        self.lang = lang
        self.target_masker = target_masker


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

    def prep(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, indices):
        def get_index_containing():
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
        len_to_contain = get_index_containing() + 1
        return ([1] * len_to_contain) + ([0] * (len(indices) - len_to_contain))
