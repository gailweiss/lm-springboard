# adapted from:
# https://pytorch.org/docs/stable/_modules/torch/\
# nn/modules/transformer.html#TransformerEncoderLayer
from typing import Optional, Union, Callable
import torch
import torch.nn as nn
# use local copy of multiheadattention to allow modifications
from model.transformer.multiheadattention import MultiheadAttention
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from copy import deepcopy


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_params, train_params,
                 activation=nn.functional.relu, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True):
        super().__init__()
        self.model_params = model_params
        train_params = deepcopy(train_params)
        self.self_attn = MultiheadAttention(self.model_params,
                                            dropout=train_params.dropout,
                                            bias=bias, batch_first=batch_first)
        # Implementation of Feedforward model
        d_model = model_params.dim
        dim_feedforward = d_model * model_params.dim_ff_factor
        nhead = self.model_params.n_heads
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout_ff_internal = Dropout(train_params.dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout_sa = Dropout(train_params.dropout)
        self.dropout_ff = Dropout(train_params.dropout)
        self.activation = activation

    def not_layernorm(self, param_name):
        return ".norm1." not in param_name and ".norm2." not in param_name

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = nn.functional.relu

    def forward(self, src, src_mask=None, attn_requests=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            attn_res, attn_pattern = self._sa_block(
                self.norm1(x), src_mask, attn_requests=attn_requests)
            x = x + attn_res
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_res, attn_pattern = self._sa_block(
                x, src_mask, attn_requests=attn_requests)
            x = self.norm1(x + attn_res)
            x = self.norm2(x + self._ff_block(x))
        return x, attn_pattern

    # self-attention block
    def _sa_block(self, x, attn_mask=None, attn_requests=None):
        x, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, attn_requests=attn_requests)
        return self.dropout_sa(x), attn_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout_ff_internal(self.activation(
                                                        self.linear1(x))))
        return self.dropout_ff(x)
