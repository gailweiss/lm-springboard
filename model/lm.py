import torch
import torch.nn as nn
from misc.util import pick_index_from_distribution, timed
from model.embeddings import FullEmbedding
from data.dataloader import mycollate, DataLoader
from tqdm import tqdm
from misc.util import printer_print as print


class LM(nn.Module):
    def __init__(self, tokenizer, model, model_params):
        super().__init__()
        self.model_params = model_params
        self.decoder = model
        self.n_tokens = tokenizer.vocab_size()
        if True in [n in self.model_params.from_os_pretrained for 
                    n in ["gpt2", "pythia"]]:
            self.embed = None
            self.de_embedder = None
        else:
            assert not self.model_params.from_os_pretrained
            if self.model_params.layer_architecture in ["torch-lstm", "torch-gru"]:
                assert self.model_params.pos_encoding == "none"
                x_dim = model_params.dim if model_params.rnn_x_dim == -1 \
                        else model_params.rnn_x_dim
            else:
                x_dim = model_params.dim
            self.embed = FullEmbedding(
                x_dim, self.n_tokens, model_params.max_seq_len,
                positional_encoding_type=model_params.pos_encoding)
            self.de_embedder = nn.Linear(self.model_params.dim, self.n_tokens)
        self.tokenizer = tokenizer
        self.tested_manual_forward = False
        self.ignore_index = self.tokenizer.pad_token_id
        self.celoss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.ordered_tokens = \
            list(self.tokenizer.convert_ids_to_nice_string([i]) for i in
                 range(self.tokenizer.vocab_size()))

    def in_main_part(self, param_name):
        return "decoder." in param_name

    def not_layernorm(self, param_name):
        return self.decoder.not_layernorm(param_name)

    def device(self):
        return next(self.parameters()).device

    def _sample(self, indices, max_seq_len, temperature, top_k, nucleus):
        was_training = self.training
        self.eval()

        eos = self.tokenizer.eos()

        max_seq_len = min(max_seq_len, self.model_params.max_seq_len)

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
            with torch.no_grad():
                e = self([indices])["logits"]  # batch size X seq len X vocab size
            next_t = choose_output_index(e[0, -1, :], temperature=temperature,
                                         top_k=top_k, nucleus=nucleus)
            indices.append(next_t)

        if was_training:
            self.train()

        return indices

    def next_token_distribution(self, pref=""):
        if isinstance(pref, str):
            indices = self.tokenizer.tokenize_without_stop(pref)
        else:
            indices = pref
        print("using indices:", indices)
        with torch.no_grad():
            e = self([indices])["logits"]  # batch size X seq len X vocab size
        assert e.shape[0] == 1
        # only want to handle one sequence so can zip with vocab nicely
        e = e[0, -1]  # first sequence, last token - vocab size
        
        s = nn.Softmax(dim=-1)(e).tolist()

        pairs = list(zip(s, self.ordered_tokens))
        t2prob = {t:p for p, t in pairs}
        sorted_probs = sorted(pairs, key=lambda x: x[0], reverse=True)
        return {"sorted_probs": sorted_probs, "probs_dict": t2prob}

    @timed
    def sample(self, pref="", max_seq_len=100, temperature=1, as_str=True,
               top_k=None, nucleus=None):
        if max_seq_len > self.model_params.max_seq_len:
            print("model max sequence length is",
                  f"{self.model_params.max_seq_len}, requested {max_seq_len}",
                  f"-- cropping to maximum {self.model_params.max_seq_len}")
            max_seq_len = self.model_params.max_seq_len
        if isinstance(pref, str):
            indices = self.tokenizer.tokenize_without_stop(pref)
        else:
            indices = pref
        indices = self._sample(indices, max_seq_len, temperature, top_k,
                               nucleus)
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
        total_preds = 0
        per_token_res = []

        for b in tqdm(dl):
            losses, stats = self.batch_perplexities(b, before_exp=True)
            mean_ls.append(stats["mean"])
            if stats["max"] > max_l:
                max_l = stats["max"]
            if stats["min"] < min_l:
                min_l = stats["min"]
            total_preds += stats["num_preds"]
            if per_token:
                ptr = losses if before_exp else torch.exp(losses)
                if None is not stats["mask"]:
                    ptr = torch.where(stats["mask"].bool(), dummy_res, ptr)
                    # false (0) - in sequence, true (1) - out of sequence
                per_token_res.append(ptr)
                # ptr shape: batch_size x max_seq_len
        mean_l = sum(mean_ls) / len(mean_ls)
        if not before_exp:
            mean_l, max_l, min_l = map(to_perp, [mean_l, max_l, min_l])
        if per_token_res:
            per_token_res = cat_with_dim1_pad(per_token_res)
        res = {"mean": mean_l, "max": max_l, "min": min_l,
               "total_preds": total_preds, "per_token_res": per_token_res}
        return res

    def get_batch_xyz(self, batch, loss_requests=None):
        loss_requests = {} if None is loss_requests else loss_requests
        indices = batch["x_indices"]
        # padded indices have already been set to self.tokenizer.pad_token_id
        x = indices[:, :-1]
        y = indices[:, 1:]  # -> y not contiguous
        # -> y.view(-1) won't work, need reshape instead
        y = y.to(dtype=torch.long)  # cross entropy loss expects target to have
        # type 'long' and will crash without explanation otherwise, so lets
        # just be safe
        z = self(x)["logits"]
        res = {"x": x, "y": y, "z": z}
        return res

    def get_losses(self, batch, loss_requests=None, accs_too=False):
        loss_requests = {} if None is loss_requests else loss_requests
        a = self.get_batch_xyz(batch, loss_requests=loss_requests)
        x, y, z = a["x"], a["y"], a["z"]
        z, y = z.reshape(-1, z.shape[-1]), y.reshape(-1)
        main_loss = self.celoss(z, y)
        losses = {"main": main_loss}
        if accs_too:
            not_y_mask = y != self.ignore_index  # 1 if in sequence, 0 if out
            z_match = z.argmax(dim=-1) == y
            correct = torch.logical_and(z_match, not_y_mask).sum()
            count = not_y_mask.sum()
            accs = {"main": (correct / count).item()}
        else:
            accs = None
        res = {"loss": losses, "acc": accs, "n_samples": x.shape[0]}
        return res

    def batch_perplexities(self, batch, before_exp=False):
        with torch.no_grad():
            a = self.get_batch_xyz(batch)
            x, y, z = a["x"], a["y"], a["z"]
        z = z.detach()
        loss_fn = nn.CrossEntropyLoss(reduction="none",
                                      ignore_index=self.ignore_index)
        losses = loss_fn(z.reshape(-1, z.shape[-1]),
                         y.reshape(-1)).view(y.shape)
        losses = losses.detach()
        res = losses if before_exp else torch.exp(losses)  # perplexity: e^loss

        mask = batch["mask"]  # 0 if on (in sequence), 1 if off (past sequence)
        if None is not mask:
            mask = mask[:, 1:]  # output positions only
            mm = mask.reshape(-1)
            num_masked = mask.sum().item()
            num_unmasked = mm.shape[0] - num_masked
            # total actual length of input/output sequences
            res = torch.where(mask.bool(), 0, res)
            # 0 as dummy value useful for computing mean and max, below
        else:
            num_unmasked = losses.view(-1).shape[0]
        mean = res.sum() / num_unmasked
        max_p = res.max()  # all losses are >=0, so dummy value (0, above) fine
        if None is not mask:
            min_p = torch.where(mask.bool(), torch.inf, res).min()
        else:
            min_p = res.min()
        stats = {"mean": mean.item(), "max": max_p.item(), "min": min_p.item(),
                 "num_preds": num_unmasked, "mask": mask}
        return res, stats
        # res contains 0 where masked, as useful dummy value for means

    def forward(self, x, get_attns=False, attn_requests=None,
                get_embeddings=False):
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
        if not self.model_params.from_os_pretrained:
            e0 = self.embed(x)  # batch size X seq len X embedding dim
            embeddings_list = [e0] if get_embeddings else None
            eL, attns = self.decoder(e0, get_attns=get_attns,
                                     attn_requests=attn_requests,
                                     embeddings_list=embeddings_list)
            logits = self.de_embedder(eL)
        elif True in [n in self.model_params.from_os_pretrained \
                      for n in ["gpt2", "pythia"]]:
            r = self.decoder(x, output_attentions=get_attns,
                             output_hidden_states=get_embeddings)
            logits = r.logits
            attns = torch.stack(r.attentions).transpose(0, 1) if get_attns\
                else None
            embeddings_list = r.hidden_states  # actually tuple but good enough
        else:
            raise Exception("lm of unknown type. from pretrained:",
                            self.from_os_pretrained)

        embeddings = torch.stack(embeddings_list).transpose(0, 1) if\
            get_embeddings else None
        # embeddings shape:
        # batch size X n layers + 1 (if transformer) or just 1 (if RNN) X
        # seq len X embedding dim
        # (transformers: n layers + 1 because input embeddings.
        #  rnns:         getting all of these is slow in pytorch, so only
        #                giving easily available top layer).
        # logits shape:
        # batch size X seq len X vocab size
        # attns shape:
        # batch size X n layers X n heads X seq len (out) X seq len (in)
        return {"logits": logits, "attns": attns, "embeddings": embeddings}


def choose_output_index(e, temperature=0, top_k=None, nucleus=None):
    if temperature > 0:
        e = filter_to_candidate_tokens(e, top_k, nucleus)
        s = nn.Softmax(dim=-1)(e / temperature)
        i = pick_index_from_distribution(s)
    else:
        i = torch.argmax(e).item()
    return i


def nucleus_threshold(vec, nucleus):
    sorted_vals = torch.sort(vec, descending=True).values
    increasing_sum = nn.Softmax(dim=-1)(sorted_vals).cumsum(dim=-1)
    n = len(vec)
    nucleus_start_point = torch.where(increasing_sum >= nucleus,
                                      torch.arange(n), n).min().item()
    nucleus_threshold = sorted_vals[nucleus_start_point]
    num_chosen = torch.where(vec >= nucleus_threshold, 1, 0).sum().item()
    chosen_ids = torch.where(vec >= nucleus_threshold, torch.arange(n), -1)
    chosen_ids = torch.sort(chosen_ids, descending=True).values[:num_chosen]
    return nucleus_threshold


def filter_to_candidate_tokens(e, top_k, nucleus):
    args = [top_k, nucleus]
    if args.count(None) == len(args):
        return e
    assert args.count(None) >= len(args) - 1
    if None is not top_k:
        assert top_k > 0
        thresh = -torch.kthvalue(-e.view(-1), top_k).values
    if None is not nucleus:
        thresh = nucleus_threshold(e, nucleus)
    e = torch.where(e >= thresh, e, -torch.inf)
    return e
