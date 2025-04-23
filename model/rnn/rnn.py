import torch
import torch.nn as nn
from misc.util import printer_print as print
from torch.nn.init import constant_


class RNN(nn.Module):
    def __init__(self, model_params, train_params):
        super().__init__()
        self.model_params = model_params
        input_dim = model_params.dim if model_params.rnn_x_dim == -1 \
                 else model_params.rnn_x_dim
        batch_first = True  # i always do batch first
        if model_params.layer_architecture == "torch-lstm":
            self.layers = torch.nn.LSTM(input_dim, model_params.dim,
                num_layers=model_params.n_layers, batch_first=batch_first,
                dropout=train_params.dropout)
        elif model_params.layer_architecture == "torch-gru":
            self.layers = torch.nn.GRU(input_dim, model_params.dim,
                num_layers=model_params.n_layers, batch_first=batch_first,
                dropout=train_params.dropout)
        else:
            raise Exception("unknown layer_architecture:" +
                            f"{model_params.layer_architecture}")
        self.h0 = nn.parameter.Parameter(
            torch.empty((model_params.n_layers, model_params.dim)))
        constant_(self.h0, 0.)
        if model_params.layer_architecture == "torch-lstm":
            self.c0 = nn.parameter.Parameter(
                torch.empty((model_params.n_layers, model_params.dim)))
            constant_(self.c0, 0.)
        self.cache_x = None
        self.cache_state = None

    def not_layernorm(self, param_name):
        return False   # no RNN components are layernorms

    def store_cache(self, x, state):
        # TODO: cache should actually also store the out,
        # and prepend it to the new out in order to give
        # actual output as expected. but because never train
        # with batch size 1 and even if doing so unlikely to
        # be training with extension of previous batch (ie no prefix match),
        # never really hit this issue        
        assert len(x.shape) == 3
        if x.shape[0] == 1:  # batch size 1
            self.cache_x = x
            if isinstance(state, tuple):
                h, c = state
                self.cache_state = (h.detach(), c.detach())
            else:
                h = state
                self.cache_state = h.detach()

    def check_cache(self, x):
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        if batch_size == 1 and None is not self.cache_x:
            cache_l = self.cache_x.shape[1]
            xpref = x[:, :cache_l, :]
            xnew = x[:, cache_l:, :]
            if torch.equal(xpref, self.cache_x):
                return xnew, self.cache_state
        h = self.h0.repeat(batch_size, 1, 1).transpose(0, 1)
        # n layers X batch size X state dim
        state = h
        if self.model_params.layer_architecture == "torch-lstm":
            c = self.c0.repeat(batch_size, 1, 1).transpose(0, 1)
            state = (h, c)
        return x, state

    def forward(self, x, get_attns=False, attn_requests=None,
                embeddings_list=None):
        # attn parameters to be compatible with lm.py expectations, but not
        # actually to be used:
        assert not get_attns, get_attns
        assert None is attn_requests, attn_requests
        # x: batch size X seq len X input dim
        xx, state0 = self.check_cache(x)
        out, staten = self.layers(xx, state0)
        # out shape: batch size X seq len X model dim
        if None is not embeddings_list:
            # list of batch size X seq len X model dim values, from each layer
            embeddings_list.append(out)
            # for an RNN, the embeddings "list" will only contain the top layer
            # embedding. for now, this is all i need anyway, and pytorch
            # doesnt make it convenient to get more than that
        self.store_cache(x, staten)
        return out, None  # None is placeholder for where the attentions
        # would be if this were a transformer - it is here for compatibility
        # with lm.py expectations
