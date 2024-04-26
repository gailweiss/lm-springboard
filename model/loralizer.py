import torch.nn as nn
import transformers
import torch
import torch.nn.functional as F


def get_adaptor(orig_mat, lora_rank):
    def get_mat(shape):
        if len(shape) == 2:
            m = torch.empty(shape)
            nn.init.normal_(m, std=0.02)
            return nn.Parameter(m)
        elif len(shape) == 1:
            return nn.Parameter(torch.zeros(shape[0]))
    if len(orig_mat.shape) == 2:
        din, dout = orig_mat.shape
        return get_mat((din, lora_rank)), get_mat((lora_rank, dout))
    elif len(orig_mat.shape) == 1:
        return get_mat(orig_mat.shape)
    else:
        raise Exception("unexpected weight shape for low rank adaptation")


class LoralizedModule(nn.Module):
    def __init__(self, *args, **kwargs):
        self.orig_weight = None
        self.orig_bias = None
        super().__init__()

    def set_internal_freezes(self):
        self.orig_weight.requires_grad_(False)
        if self.orig_bias is not None:
            self.orig_bias.requires_grad_(False)


class LoralizedLayerNorm(LoralizedModule):
    def __init__(self, ln):
        super().__init__()
        self.ln = ln
        self.orig_weight = self.ln.weight
        self.orig_bias = self.ln.bias
        self.aw = get_adaptor(self.orig_weight, None)
        self.ab = get_adaptor(self.orig_bias, None)
        self.set_internal_freezes()

    def forward(self, x):
        weight = self.orig_weight + self.aw
        bias = self.orig_bias + self.ab
        return F.layer_norm(x, self.ln.normalized_shape, weight, bias,
                            self.ln.eps)


# adapted from implementation here:
# https://huggingface.co/transformers/v3.1.0/_modules/transformers/
# modeling_utils.html#Conv1D
class LoralizedConv1D(LoralizedModule):
    def __init__(self, c, lora_rank):
        super().__init__()
        self.orig_weight = c.weight
        self.orig_bias = c.bias
        self.nf = c.nf
        self.aw1, self.aw2 = get_adaptor(self.orig_weight, lora_rank)
        self.ab = get_adaptor(self.orig_bias, lora_rank)
        self.set_internal_freezes()

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        w = self.orig_weight + (self.aw1 @ self.aw2)
        b = self.orig_bias + self.ab
        x = torch.addmm(b, x.view(-1, x.size(-1)), w)
        x = x.view(*size_out)
        return x


class LoralizedLinear(LoralizedModule):
    def __init__(self, linear, lora_rank):
        super().__init__()
        self.orig_weight = linear.weight
        self.orig_bias = linear.bias  # might be None
        self.aw1, self.aw2 = get_adaptor(self.orig_weight, lora_rank)

        if None is not self.orig_bias:
            self.ab = get_adaptor(self.orig_bias, lora_rank)

        self.set_internal_freezes()

    def forward(self, x):
        w = self.orig_weight + (self.aw1 @ self.aw2)
        b = self.orig_bias if None is self.orig_bias else \
            (self.orig_bias + self.ab)
        return F.linear(x, w, b)


# taken from:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/
# sparse.html#Embedding ,
# which seems to be what the gpt2 model is using
class LoralizedEmbedding(LoralizedModule):
    def __init__(self, e, lora_rank) -> None:
        super().__init__()
        self.e = e
        self.orig_weight = e.weight
        self.aw1, self.aw2 = get_adaptor(self.e.weight, lora_rank)
        self.set_internal_freezes()

    def forward(self, x):
        weight = self.orig_weight + (self.aw1 @ self.aw2)
        return F.embedding(x, weight, self.e.padding_idx, self.e.max_norm,
                           self.e.norm_type, self.e.scale_grad_by_freq,
                           self.e.sparse)


def loralize_gpt2lmheadmodel(model, lora_rank):
    def loralize_gpt2block(layer):
        layer.ln_1 = LoralizedLayerNorm(layer.ln_1)
        layer.attn.c_attn = LoralizedConv1D(layer.attn.c_attn, lora_rank)
        layer.attn.c_proj = LoralizedConv1D(layer.attn.c_proj, lora_rank)
        layer.ln_2 = LoralizedLayerNorm(layer.ln_2)
        layer.mlp.c_fc = LoralizedConv1D(layer.mlp.c_fc, lora_rank)
        layer.mlp.c_proj = LoralizedConv1D(layer.mlp.c_proj, lora_rank)

    model.transformer.wte = LoralizedEmbedding(model.transformer.wte,
                                               lora_rank)
    model.transformer.wpe = LoralizedEmbedding(model.transformer.wpe,
                                               lora_rank)
    model.transformer.ln_f = LoralizedLayerNorm(model.transformer.ln_f)
    model.lm_head = LoralizedLinear(model.lm_head, lora_rank)
    for layer in model.transformer.h:
        loralize_gpt2block(layer)
    model.loralized = True


class LoralizedGPT2LMHeadModel(LoralizedModule):
    def __init__(self, model, lora_rank):
        super().__init__()
        self.source_model = model
        self.lora_rank = lora_rank
        self.setup()
        self.n_tokens = self.source_model.n_tokens
        # my code uses this somewhere

    def set_internal_freezes(self):
        def _recursive_set_internal_freezes(m):
            if isinstance(m, LoralizedModule):
                m.set_internal_freezes()
            for c in m.children():
                _recursive_set_internal_freezes(c)
        _recursive_set_internal_freezes(self.source_model)

    def setup(self):
        assert not hasattr(self.source_model, "loralized")
        # messy: modifying GPT2LMHeadModel in-place
        # because idk what order they run all the parts in
        loralize_gpt2lmheadmodel(self.source_model, self.lora_rank)
        self.set_internal_freezes()

    def forward(self, *args, **kwargs):
        return self.source_model(*args, **kwargs)


def add_lora(model, lora_rank):
    if isinstance(model, transformers.GPT2LMHeadModel):
        return LoralizedGPT2LMHeadModel(model, lora_rank)
    else:
        raise Exception("haven't implemented loralizing this kind of model")
