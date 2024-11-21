# adapted from
# https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py
import math
import torch
from torch import Tensor
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.functional import _mha_shape_check, _in_projection_packed, \
                                _in_projection, linear, softmax, dropout
from misc.util import printer_print as print


def multi_head_attention_forward(query, key, value, num_heads, QKV, dropout_p,
                                 out_proj_weight, out_proj_bias, training=True,
                                 attn_mask=None, static_k=None, static_v=None,
                                 model_params=None, attn_requests=None):

    if None is not attn_requests:
        print("successfully received attn request!", attn_requests)
        raise NotImplementedError
    is_batched = _mha_shape_check(query, key, value, None, attn_mask,
                                  num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend
    # that the input is batched, run the computation and before returning
    # squeeze the batch dimension so that the output doesn't carry this
    # temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == model_params.dim

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    msg = f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert head_dim * num_heads == embed_dim, msg
    msg = f"key's sequence and batch dims {key.shape[:2]} do not match " +\
          f"value's {value.shape[:2]}"
    assert key.shape[:2] == value.shape[:2], msg

    if model_params.individual_head_params:
        Q_linears, K_linears, V_linears = QKV
        # taken from torch.nn.functional _in_projection:
        q = concatlinears(query, Q_linears)
        k = concatlinears(key, K_linears)
        v = concatlinears(value, V_linears)
    else:
        use_separate_proj_weight, in_proj_weight, in_proj_bias = QKV[:3]
        q_proj_weight, k_proj_weight, v_proj_weight = QKV[3:]
        if not use_separate_proj_weight:
            mg = "use_separate_proj_weight is False but in_proj_weight is None"
            assert in_proj_weight is not None, mg
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight,
                                            in_proj_bias)
        else:

            def msgf(n):
                return f"use_separate_proj_weight is True but {n} is None"
            assert q_proj_weight is not None, msgf("q_proj_weight")
            assert k_proj_weight is not None, msgf("k_proj_weight")
            assert v_proj_weight is not None, msgf("v_proj_weight")
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(query, key, value, q_proj_weight,
                                     k_proj_weight, v_proj_weight, b_q, b_k,
                                     b_v)

    # the result is these shapes:
    # q, k, v: seq len X bsz X embed_dim

    # prep attention mask
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError("The shape of the 2D attn_mask is " +
                                   f"{attn_mask.shape}, but should be " +
                                   f"{correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is " +
                                   f"{attn_mask.shape}, but should be " +
                                   f"{correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} " +
                               "is not supported")

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.reshape(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections
        # when statics are passed
        msg = f"expecting static_k.size(0) of {bsz * num_heads}, but got " +\
              f"{static_k.size(0)}"
        assert static_k.size(0) == bsz * num_heads, msg
        msg = f"expecting static_k.size(2) of {head_dim}, but got " +\
              f"{static_k.size(2)}"
        assert static_k.size(2) == head_dim, msg
        k = static_k
    if static_v is None:
        v = v.reshape(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections
        # when statics are passed
        msg = f"expecting static_v.size(0) of {bsz * num_heads}, but got " +\
              f"{static_v.size(0)}"
        assert static_v.size(0) == bsz * num_heads, msg
        msg = f"expecting static_v.size(2) of {head_dim}, but got " +\
              f"{static_v.size(2)}"
        assert static_v.size(2) == head_dim, msg
        v = static_v

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    B, Nt, E = q.shape
    q_scaled = q * math.sqrt(1.0 / float(E))

    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled,
                                            k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

    attn_output_weights = softmax(attn_output_weights, dim=-1)

    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz,
                                                                embed_dim)

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                   src_len)

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights


def concatlinears(x, Ls):
    res = [linear(x) for linear in Ls]
    return torch.cat(res, dim=-1)  # cat along the output dim
