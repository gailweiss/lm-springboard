# adapted from:
# https://pytorch.org/docs/stable/_modules/torch/\
# nn/modules/activation.html#MultiheadAttention
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn import functional as F
from model.transformer.torch_f_multi_head_attention_forward import \
        multi_head_attention_forward


class MultiheadAttention(nn.Module):
    def __init__(self, model_params, dropout=0., bias=True,
                 kdim=None, vdim=None, batch_first=False):
        self.model_params = model_params
        embed_dim = model_params.dim
        num_heads = model_params.n_heads
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (self.kdim == embed_dim and
                                    self.vdim == embed_dim)
        self.use_separate_proj_weight = not self._qkv_same_embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        msg = "embed_dim must be divisible by num_heads"
        assert self.head_dim * num_heads == self.embed_dim, msg

        if self.model_params.individual_head_params:
            # biases shape: embed dim (out)
            # q_proj_weight, k_proj_weight, v_proj_weight shapes:
            # embed dim (out) X embed dim (in)
            # (out/in inferred from docs of linear)
            self.Qs = nn.ModuleList(
                        nn.Linear(embed_dim, self.head_dim, bias=bias) for
                        _ in range(self.num_heads))
            self.Ks = nn.ModuleList(
                        nn.Linear(self.kdim, self.head_dim, bias=bias) for
                        _ in range(self.num_heads))
            self.Vs = nn.ModuleList(
                        nn.Linear(self.vdim, self.head_dim, bias=bias) for
                        _ in range(self.num_heads))
        else:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(torch.empty((embed_dim,
                                                            embed_dim)))
                self.k_proj_weight = Parameter(torch.empty((embed_dim,
                                                            self.kdim)))
                self.v_proj_weight = Parameter(torch.empty((embed_dim,
                                                            self.vdim)))
                self.register_parameter('in_proj_weight', None)
            else:
                self.in_proj_weight = Parameter(torch.empty((3 * embed_dim,
                                                             embed_dim)))
                self.register_parameter('q_proj_weight', None)
                self.register_parameter('k_proj_weight', None)
                self.register_parameter('v_proj_weight', None)
            if bias:
                self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
            else:
                self.register_parameter('in_proj_bias', None)

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim,
                                                        bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.model_params.individual_head_params:
            for Ls in [self.Qs, self.Ks, self.Vs]:
                [xavier_uniform_(linear.weight) for linear in Ls]
        else:
            if self._qkv_same_embed_dim:
                xavier_uniform_(self.in_proj_weight)
            else:
                xavier_uniform_(self.q_proj_weight)
                xavier_uniform_(self.k_proj_weight)
                xavier_uniform_(self.v_proj_weight)

        if None is not self.out_proj.bias:
            constant_(self.out_proj.bias, 0.)

            if self.model_params.individual_head_params:
                for Ls in [self.Qs, self.Ks, self.Vs]:
                    [constant_(linear.bias, 0.) for linear in Ls]
            else:
                constant_(self.in_proj_bias, 0.)

    def __setstate__(self, state):
        super().__setstate__(state)

    def forward(self, query, key, value, attn_mask=None, attn_requests=None):
        r"""
        Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input,
            :math:`(L, N, E_q)` when ``batch_first=False`` or
            :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is
            the target sequence length, :math:`N` is the batch size, and
            :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input,
            :math:`(S, N, E_k)` when ``batch_first=False`` or
            :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is
            the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input,
            :math:`(S, N, E_v)` when ``batch_first=False`` or
            :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is
            the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        attn_mask: If specified, a 2D or 3D mask preventing attention to
            certain positions. Must be of shape :math:`(L, S)` or
            :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the
            batch size, :math:`L` is the target sequence length, and :math:`S`
            is the source sequence length. A 2D mask will be broadcasted across
            the batch while a 3D mask allows for a different mask for each
            entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True``
            value indicates that the corresponding position is not allowed to
            attend. For a float mask, the mask values will be added to the
            attention weight.

        Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when
            input is unbatched, :math:`(L, N, E)` when ``batch_first=False`` or
            :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the
            target sequence length, :math:`N` is the batch size, and :math:`E`
            is the embedding dimension ``embed_dim``.
        - **attn_output_weights**  attention weights per head of shape
            :math:`(\text{num\_heads}, L, S)` when input is unbatched or
            :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in
                                     (query, key, value))

        if self.model_params.individual_head_params:
            QKV = (self.Qs, self.Ks, self.Vs)
        else:
            QKV = (self.use_separate_proj_weight, self.in_proj_weight,
                   self.in_proj_bias, self.q_proj_weight, self.k_proj_weight,
                   self.v_proj_weight)
        mhaf_args = (query, key, value, self.num_heads, QKV, self.dropout,
                     self.out_proj.weight, self.out_proj.bias)
        mhaf_kwargs = {"training": self.training,
                       "attn_mask": attn_mask,
                       "model_params": self.model_params,
                       "attn_requests": attn_requests}

        attn_o_w = multi_head_attention_forward(*mhaf_args, **mhaf_kwargs)
        attn_output, attn_output_weights = attn_o_w
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)
        return attn_output, attn_output_weights
