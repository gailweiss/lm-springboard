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
        if self.prepped:
            return
        tokens = tokenizer(self.subseq)
        if tokens[0] == tokenizer.bos():
            tokens = tokens[1:]
        if tokens[-1] == tokenizer.eos():
            tokens = tokens[:-1]
        self.subseq_toks = tokens
        self.prepped = True

    def __call__(self, indices):
        ssl = len(self.subseq_toks)
        pos = next((i for i in range(len(indices)) if 
                    indices[i: i + ssl] == self.subseq_toks),
                    len(indices))  # if subseq not found, masks whole seq
        pos = pos + ssl
        return ([1] * pos) + ([0] * (len(indices) - pos))
