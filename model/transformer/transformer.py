import torch
import torch.nn as nn
from model.transformer.transformerencoderlayer import TransformerEncoderLayer
from misc.util import printer_print as print


class Transformer(nn.Module):
    def __init__(self, model_params, train_params):
        super().__init__()
        self.model_params = model_params

        def make_layer():
            batch_first = True  # i always do batch first
            if model_params.layer_architecture == "custom-transformer":
                return TransformerEncoderLayer(self.model_params,
                                               train_params,
                                               batch_first=batch_first)
            elif model_params.layer_architecture == "torch-transformer":
                dim_ff = self.model_params.dim * \
                         self.model_params.dim_ff_factor
                return nn.TransformerEncoderLayer(
                            dropout=train_params.dropout,
                            d_model=self.model_params.dim,
                            nhead=self.model_params.n_heads,
                            dim_feedforward=dim_ff, batch_first=batch_first)
            else:
                raise Exception("unknown layer_architecture:" +
                                f"{model_params.layer_architecture}")
        self.layers = nn.ModuleList([make_layer() for _ in
                                     range(self.model_params.n_layers)])

    def not_layernorm(self, param_name):
        if self.model_params.layer_architecture == "torch-transformer":
            return ".norm1." not in param_name and ".norm2." not in param_name
        if self.model_params.layer_architecture == "custom-transformer":
            return self.layers[0].not_layernorm(param_name)
        return "unknown layer architecture - don't know norm names!"

    def causal_mask(self, length, device, target_type):
        res = nn.Transformer.generate_square_subsequent_mask(length)
        res = res.to(device=device)
        if not torch.is_floating_point(res):
            res = (
                torch.zeros_like(res, dtype=target_type)
                .masked_fill_(res, float("-inf"))
            )
        return res

    def forward(self, x, get_attns=False, attn_requests=None,
                embeddings_list=None):
        # batch size X seq len X embed dim
        mask = self.causal_mask(x.shape[1], x.device, x.dtype)
        attns = []
        for layer in self.layers:
            if self.model_params.layer_architecture == "custom-transformer":
                x, attn = layer(x, src_mask=mask, attn_requests=attn_requests)
                attns.append(attn)
            elif self.model_params.layer_architecture == "torch-transformer":
                if get_attns:
                    x, attn = _layer_forward(layer, x, mask)
                    attns.append(attn)
                else:
                    x = layer(x, src_mask=mask)
            else:
                raise Exception("unknown layer_architecture: " +
                                f"{self.model_params.layer_architecture}")
            if None is not embeddings_list:
                embeddings_list.append(x)
                # accessed outside - lists get mutated
        attns = torch.stack(attns).transpose(0, 1) if attns else None
        # attns shape: batch size, n layers, n heads, seq len, seq len
        # x shape: batch size X seq len X embed dim
        return x, attns

########
# helper functions to get attn patterns for a 'vanilla' or mostly-vanilla
# transformer model
########
# (uses unmodified transformerencoderlayers, even if their internals have
# slightly changed)
# (assumes only a decoding model, and no padding masking)


def _sa_block(layer, x, attn_mask):
    x, attn = layer.self_attn(
        x, x, x, attn_mask=attn_mask, need_weights=True, is_causal=False,
        average_attn_weights=False)
    # attn shape: batch size, n heads, seq len, seq len
    return layer.dropout1(x), attn


def verify_good_forward_sim(layer, x, src_mask, verbose=True):
    if not hasattr(layer, "forward_sim_tests"):
        layer.forward_sim_tests = {}
    if x.shape[0] > 1 and layer.forward_sim_tests.get("batch", False):
        return
    elif x.shape[0] == 1 and layer.forward_sim_tests.get("single", False):
        return

    if layer.training:  # cannot test - dropout will give different results
        if verbose:
            print("not verifying forward simulation - the layer is in " +
                  "training, its dropout will likely cause different results")
        return

    r1 = layer(x, src_mask=src_mask)
    r2, attn = _layer_forward(layer, x, src_mask, skip_test=True)

    msg = "attention getting functions are not equivalent to true layer " +\
          "behaviour - are you using a custom layer? if not, maybe pytorch " +\
          "has changed the layer?"
    assert False not in torch.isclose(r1, r2, atol=1e-6).view(-1), msg
    # would have used torch.equal but it seems am getting some differences up
    # to 4e-7 when trying to run a standard layer. these are really small diffs
    # though and i suspect what's really happening is a difference between
    # using or not using the fast path, so im allowing a small difference here
    if verbose:
        print("successfully verified forward sim on input of shape:", x.shape)
    if x.shape[1] < 4:
        print("not recording this as successful check - too small")
        return
    if x.shape[0] == 1:
        layer.forward_sim_tests["single"] = True
    elif x.shape[0] > 1:
        layer.forward_sim_tests["batch"] = True


def _layer_forward(layer, x, src_mask, skip_test=False):
    if not skip_test:
        verify_good_forward_sim(layer, x, src_mask)
    # no padding mask in my experiments for now, so lets try not to have too
    # much redundant code
    if layer.norm_first:
        attn_res, attn_pattern = _sa_block(layer, layer.norm1(x), src_mask)
        x = x + attn_res
        x = x + layer._ff_block(layer.norm2(x))
    else:
        attn_res, attn_pattern = _sa_block(layer, x, src_mask)
        x = layer.norm1(x + attn_res)
        x = layer.norm2(x + layer._ff_block(x))
    return x, attn_pattern
