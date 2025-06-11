class RawSample:
    # for use in creating datasets, before processing in the datamodule
    def __init__(self, seq, lang=None):
        self.seq = seq
        self.lang = lang


class TokenizedSample:
    # for use in creating datasets, after tokenizing but before final collation
    # in the datamodule
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)
