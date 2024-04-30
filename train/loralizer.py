import torch.nn as nn
import transformers
import torch
import torch.nn.functional as F
from model.transformer.transformer import Transformer
from copy import deepcopy
import torch.nn.utils.parametrize as parametrize


def get_adaptor(orig_mat, lora_params):
    rank = lora_params.lora_rank
    std = lora_params.lora_std if None is not lora_params.lora_std else 0.02

    def get_mat(shape):
        if len(shape) == 2:
            m = torch.empty(shape)
            nn.init.normal_(m, std=std)
            return nn.Parameter(m)
        elif len(shape) == 1:  # e.g. for LayerNorm
            return nn.Parameter(torch.zeros(shape[0]))

    if len(orig_mat.shape) == 2:
        din, dout = orig_mat.shape
        return get_mat((din, rank)), get_mat((rank, dout))
    elif len(orig_mat.shape) == 1:
        return get_mat(orig_mat.shape)
    else:
        raise Exception("unexpected weight shape for low rank adaptation")

class LoRA(nn.Module):
    def __init__(self, param, lora_params):
        self.pdim = len(param.shape)
        if self.pdim == 1:
            self.a = get_adaptor(param, lora_params)
        elif self.pdim == 2:
            self.a1, self.a2 = get_adaptor(param, lora_params)
        else:
            raise NotImplementedError(param.shape)

    def forward(self, param):
        if self.pdim == 1:
            return param + self.a
        elif self.pdim == 2:
            return param + (self.a1 @ self.a2)


class LoralizedModule(nn.Module):
    def __init__(self, module, lora_params, already_deepcopied=False):
        super().__init__()
        self.module = deepcopy(module) if not already_deepcopied else module
        self.lora_params = lora_params
        self.setup()

    def setup(self):
        assert not hasattr(self.module, "loralized")
        # don't know what kind of mess it would make to apply this twice to the
        # same module, so avoid for now
        def apply_direct_params_lora(module):
            direct_params = [p for p in module.named_parameters() if "." not in p[0]]
            for n,p in direct_params:
                p.requires_grad_(False)
                parametrize.register_parametrization(module, n, LoRA(p, self.lora_params))
        def recursive_module_lora(module):
            apply_direct_params_lora(module)
            for c in module.children():
                recursive_module_lora(c)
        self.loralized = True

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def apply_lora_to_lm(lm, lora_params):
    assert lora_params.lora_rank > 0
    lm.decoder = LoralizedModule(lm.decoder, lora_params)
    if None is not lm.embed:
        lm.embed = LoralizedModule(lm.embed, lora_params)
        lm.de_embedder = LoralizedModule(lm.de_embedder, lora_params)
