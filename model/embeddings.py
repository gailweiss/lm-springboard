import math
import torch
import torch.nn as nn


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        # note: original has default dropout=0.1
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # hack around rounding problems
        r = d_model % 2
        d_model += r  # make even for the pe bounds by 2 so cos doesn't end up
        # creating 1 more column than necessary and getting upset

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape: max_len X 1 X d_model

        # get back to actual requested dim after rounding problems
        d_model -= r
        pe = pe[:, :, :d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        embedding = self.pe[:x.size(0), :]
        x = x + embedding
        return self.dropout(x)


class NoEmbedding(nn.Module):
    def __init__(self, *a, **kw):
        super(NoEmbedding, self).__init__()

    def forward(self, x, real_lengths=None):
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_forwards = d_model
        self.forwards_embedding = nn.Embedding(max_len, self.d_forwards)
        nn.init.normal_(self.forwards_embedding.weight, mean=0.0, std=0.02,
                        generator=None)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x, real_lengths=None):
        # x shape: seq len X batch size X hidden dim
        # want to get the positional encodings with same shape
        seq_len = x.shape[0]

        def make_indices_tensor(indices):
            res = torch.LongTensor([indices]).transpose(0, 1)
            res = res.to(device=self.device())
            if next(self.parameters()).is_cuda:
                res = res.cuda()
            return res

        y = self.forwards_embedding(make_indices_tensor(list(range(seq_len))))
        x[:, :, :self.d_forwards] += y

        # figuring out why pytorch broadcasting sorts the dimensions out
        # properly here will break me,
        # but in the meantime it seems to work so idfk
        return self.dropout(x)


class FullEmbedding(nn.Module):
    def __init__(self, d_model, num_tokens, max_len, positional_dropout=0.0,
                 positional_encoding_type='learned', separate_encodings=False):
        super(FullEmbedding, self).__init__()

        position_modules = {'sin': PositionalEncoding,
                            'learned': PositionalEmbedding,
                            'none': NoEmbedding}
        position_module = position_modules[positional_encoding_type]
        positional_encoding = position_module(d_model, positional_dropout,
                                              max_len=max_len)

        word_embedding = nn.Embedding(num_tokens, d_model)
        nn.init.normal_(word_embedding.weight, mean=0.0, std=0.02,
                        generator=None)
        self.separate_encodings = separate_encodings

        self.word = word_embedding
        self.pos = positional_encoding
        self.max_len = max_len

    def forward(self, x, real_lengths=None):
        # x: longtensor of padded 'samples', with shape: batch_size X seq_len.
        # (each value: an int in 0,1,..,num_tokens-1)

        # However, this function was originally written
        # for seq_len X batch_size, so:
        x = x.transpose(0, 1)

        # convert pad inputs to something innocuous for this stage:

        res = self.word(x)
        # res shape: seq len X batch size X d_model
        if self.separate_encodings:
            p = self.pos(torch.zeros(res.shape))
            res = torch.stack((res, p), dim=-1)
            # seq len X batch size X d_model X 2 (2: first is tokens,
            # second is position)
        else:
            res = self.pos(res)
        # and now return to the batch_size X seq_len X embed dim (and if
        # separate encodings: X 2 ) shape expected by our new calling functions
        res = res.transpose(0, 1)
        return res
