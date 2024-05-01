import torch.nn as nn
import transformers
import torch
import torch.nn.functional as F
from model.transformer.transformer import Transformer
from copy import deepcopy
import torch.nn.utils.parametrize as parametrize
from functools import reduce


class LoRA(nn.Module):
    def __init__(self, param, lora_params):
        super().__init__()
        self.pdim = len(param.shape)
        if self.pdim == 1:
            self.a = nn.Parameter(torch.zeros(param.shape))
        elif self.pdim == 2:
            def make_mat(din, dout):
                a = torch.empty((din, dout))
                return nn.Parameter(nn.init.normal_(a,
                                                    std=lora_params.lora_std))
            self.a1 = make_mat(param.shape[0], lora_params.lora_rank)
            self.a2 = make_mat(lora_params.lora_rank, param.shape[1])
        else:
            raise NotImplementedError(param.shape)

    def forward(self, param):
        if self.pdim == 1:
            return param + self.a
        elif self.pdim == 2:
            return param + (self.a1 @ self.a2)


def get_parent_module(module, full_param_name):
    names = full_param_name.split(".")[:-1]
    # last one is the parameter name
    return reduce(getattr, names, module)


def get_nested_param(module, full_param_name):
    names = full_param_name.split(".")
    return reduce(getattr, names, module)


def apply_lora_to_model(model, lora_params):
    assert lora_params.lora_rank > 0
    orig_params = list(model.named_parameters())
    for n, p in orig_params:
        m = get_parent_module(model, n)
        p.requires_grad_(False)
        parametrize.register_parametrization(m, n.split(".")[-1],
                                             LoRA(p, lora_params))
    retie_weights_in_parametrized_model(model)


class Tie(nn.Module):
    def __init__(self, model, nested_param_path):
        super().__init__()
        self.model = [model]  # in a list so it doesnt notice it's a module
        self.nested_param_path = nested_param_path

    def forward(self, param):
        return get_nested_param(self.model[0], self.nested_param_path)


def retie_weights_in_parametrized_model(model):
    def main_params(m):
        return [p for p in m.named_parameters() if "." not in p[0]]

    def get_source_name_in_parametrization(param, parametrization):
        originals = [(n, p) for n, p in parametrization.named_parameters()
                     if n.endswith(".original")]
        return next((n for n, p in originals if param is p), False)

    all_parametrizations = [np for np in model.named_modules() if
                            np[0].endswith("parametrizations")]
    main_modules = [nm for nm in model.named_modules() if
                    "parametrizations" not in nm[0]]
    main_modules_with_params = [nm for nm in main_modules if
                                len(main_params(nm[1])) > 0]
    for mname, module in main_modules_with_params:
        mp = main_params(module)
        for pname, param in mp:
            relevant_parametrizations = \
                [p for p in all_parametrizations if
                 get_source_name_in_parametrization(param, p[1])]
            assert len(relevant_parametrizations) == 1
            pmtzname, pmtz = relevant_parametrizations[0]
            srcname = get_source_name_in_parametrization(param, pmtz)
            originalstr = ".original"
            paramzstr = ".parametrizations"
            assert pmtzname.endswith(paramzstr)
            assert srcname.endswith(originalstr)
            srcname = srcname[: -len(originalstr)]
            pmtzname = pmtzname[: -len(paramzstr)]
            srcpath = f"{pmtzname}.{srcname}"
            parametrize.register_parametrization(module, pname,
                                                 Tie(model, srcpath))
