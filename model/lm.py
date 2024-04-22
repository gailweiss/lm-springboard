import torch
import torch.nn as nn
from util import pick_index_from_distribution, timed
from model.embeddings import FullEmbedding
from data.dataloader import mycollate, DataLoader
from tqdm import tqdm


class LM(nn.Module):
    def __init__(self, tokenizer, model, model_params):
        super().__init__()
        self.model_params = model_params
        self.decoder = model
        self.n_tokens = tokenizer.vocab_size()
        self.embed = FullEmbedding(
            model_params.dim, self.n_tokens, model_params.max_seq_len,
            positional_encoding_type=model_params.pos_encoding)
        self.de_embedder = nn.Linear(self.model_params.dim, self.n_tokens)
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax(dim=-1)
        self.tested_manual_forward = False
        self.is_from_pretrained = False

    def choose_output_index(self, e, temperature=0):
        if temperature > 0:
            s = self.softmax(e / temperature)
            i = pick_index_from_distribution(s)
        else:
            i = torch.argmax(e).item()
        return i

    def device(self):
        return next(self.parameters()).device

    @timed
    def sample(self, pref="", max_seq_len=100, temperature=1, as_str=True):
        if max_seq_len > self.model_params.max_seq_len:
            print("model max sequence length is",
                  f"{self.model_params.max_seq_len}, requested {max_seq_len}",
                  f"-- cropping to maximum {self.model_params.max_seq_len}")
            max_seq_len = self.model_params.max_seq_len
        if isinstance(pref, str):
            indices = self.tokenizer.tokenize_without_stop(pref)
        else:
            indices = pref
        eos = self.tokenizer.eos()

        def stop(indices):
            if len(indices) >= max_seq_len:
                return True
            if len(indices) <= 1:  # just started
                # check this before checking for eos because sometimes eos
                # and bos are same token, and dont want to stop the sample
                # because of the bos
                return False
            if indices[-1] == eos:
                return True
        while not stop(indices):
            e = self([indices])  # batch size X seq len X vocab size
            next_t = self.choose_output_index(e[0, -1, :],
                                              temperature=temperature)
            indices.append(next_t)

        if as_str:
            return self.tokenizer.convert_ids_to_nice_string(indices)
        else:
            return indices

    def perplexities(self, dl, batch_size=16, before_exp=False,
                     per_token=False, dummy_res=-1):

        def to_perp(v):
            res = torch.exp(torch.Tensor([v]))
            if not isinstance(v, torch.Tensor):
                res = res.item()
            return res

        def cat_with_dim1_pad(xs):
            to_dim2 = max(x.shape[1] for x in xs)

            def pad_with_dummy(x):
                dummy = -1
                x2 = torch.ones((x.shape[0], to_dim2)) * dummy_res
                x2[:, :x.shape[1]] = x
                return x2
            return torch.cat(tuple(pad_with_dummy(x) for x in xs))

        if not isinstance(dl, DataLoader):
            if not dl:
                return None
            if isinstance(dl[0], str):
                dl = self.tokenizer(dl)
            dl = DataLoader(dl, batch_size=batch_size, shuffle=False,
                            collate_fn=mycollate)

        mean_ls = []
        max_l = 0
        min_l = torch.inf
        total_tokens = 0
        per_token_res = []

        for b in tqdm(dl):
            losses, stats = self.batch_perplexities(b, before_exp=True)
            me, ma, mi, n_tokens, mask = stats
            mean_ls.append(me)
            if ma > max_l:
                max_l = ma
            if mi < min_l:
                min_l = mi
            total_tokens += n_tokens
            if per_token:
                ptr = losses if before_exp else torch.exp(losses)
                if None is not mask:
                    ptr = torch.where(mask == 1, dummy_res, ptr)
                per_token_res.append(ptr)
                # ptr shape: batch_size x max_seq_len
        mean_l = sum(mean_ls) / len(mean_ls)
        if not before_exp:
            mean_l, max_l, min_l = map(to_perp, [mean_l, max_l, min_l])
        if per_token_res:
            per_token_res = cat_with_dim1_pad(per_token_res)
        return mean_l, max_l, min_l, total_tokens, per_token_res

    def batch_perplexities(self, batch, before_exp=False):
        indices, mask = batch["x_indices"], batch["mask"]
        if mask is not None:
            indices = indices + (mask * self.tokenizer.pad_token_id)
        x = indices[:, :-1]
        y = indices[:, 1:]  # -> y not contiguous -> y.view(-1) won't work
        # (i.e. need reshape instead of view)
        y = y.to(dtype=torch.long)  # cross entropy loss expects target to be
        # 'long' and will crash without explanation otherwise, so play safe
        z = self(x).detach()  # dont need to grad through these
        loss_fn = nn.CrossEntropyLoss(reduction="none",
                                      ignore_index=self.tokenizer.pad_token_id)
        losses = loss_fn(z.view(-1, self.n_tokens),
                         y.reshape(-1)).view(y.shape)
        res = losses if before_exp else torch.exp(losses)  # perplexity: e^loss
        if None is not mask:
            mask = mask[:, 1:]
            mm = mask.reshape(-1)
            num_masked = torch.where(mm == 1, 1, 0).sum().item()
            num_unmasked = mm.shape[0] - num_masked

            res = torch.where(mask == 1, 0, res)
            # 0 as dummy value useful for computing mean
        else:
            num_unmasked = losses.view(-1).shape[0]
        mean = res.sum() / num_unmasked
        max_p = res.max()
        if None is not mask:
            min_p = torch.where(mask == 1, torch.inf, res).min()
        else:
            min_p = res.min()
        return res, (mean.item(), max_p.item(), min_p.item(),
                     num_unmasked, mask)
        # res contains 0 where masked, as useful dummy value for means

    def forward(self, x, get_attns=False, attn_requests=None):
        # x shape: should be batch size x seq len, possibly padded.
        # but can be: just seq len (in which case will be reshaped to batch
        # size one), or even can be a string (in which case will be tokenized)
        if isinstance(x, str):
            x = self.tokenizer(x)
        if isinstance(x, list):
            x = torch.LongTensor(x).to(device=self.device())
        if len(x.shape) == 1:  # lacks batch dim
            x = x.view(1, -1)
        x = x.to(dtype=torch.long)
        cond = x.shape[1] <= self.model_params.max_seq_len
        msg = f"got input over maximum expected length: {x.shape}, " +\
              f"max len: {self.model_params.max_seq_len}"
        assert cond, msg
        if not self.is_from_pretrained:
            e0 = self.embed(x)
            eL, attns = self.decoder(e0, get_attns=get_attns,
                                     attn_requests=attn_requests)
            res = self.de_embedder(eL)
        elif self.is_from_pretrained == "gpt2":
            r = self.decoder(x)
            res = r.logits
            attns = None  # it seems there should actually be a way to easily
            # get attentions from huggingface gpt2, but a lazy read online only
            # showed how to do it when generating which isnt what i want here
        else:
            raise Exception("lm of unknown type. from pretrained:",
                            self.is_from_pretrained)
        return (res, attns) if get_attns else res
