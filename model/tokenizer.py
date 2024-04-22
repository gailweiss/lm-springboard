from transformers import BertTokenizer, GPT2Tokenizer
from util import timed
import tokenizers
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


@timed
def make_tokenizer(data, vocab_size=30, eos="[EOS]", bos="[BOS]", pad="[PAD]",
                   unk="[UNK]", verbose=False):
    # for now always BPE, not focusing on this
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=unk))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
                                add_prefix_space=False)
    trainer = tokenizers.trainers.BpeTrainer(
                    vocab_size=vocab_size, special_tokens=[unk, pad, bos, eos])
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"{bos} $A {eos}", special_tokens=[(bos, 1), (eos, 2)])
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    if verbose:
        print("\n\n!! made tokenizer with " +
              f"{len(tokenizer.get_vocab())} tokens")
    return tokenizer


def _load_custom_tokenizer(path):
    return PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")


class CharTokenizer:
    def __init__(self, data, verbose_init=False):
        assert isinstance(data, list)
        self.tokens = set()
        for s in data:
            self.tokens.update(list(s))
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.special_tokens = [self.unk_token, self.pad_token, self.bos,
                               self.eos]
        self.tokens.update(self.special_tokens)
        self.id2tok = sorted(list(self.tokens))
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}
        self._pad_token_type_id = self.tok2id[self.pad_token]
        self.eos_token_id = self.tok2id[self.eos]
        self.bos_token_id = self.tok2id[self.bos]
        self.special_token_ids = [self.tok2id[t] for t in self.special_tokens]
        if verbose_init:
            print("made char-level tokenizer with", len(self.tok2id), "tokens")
        self.is_char_tokenizer_with_eos_and_bos = True

    def __call__(self, samples):
        def single(s):
            return [self.bos_token_id] + [self.tok2id[t] for t in s] +\
                   [self.eos_token_id]
        res = single(samples) if isinstance(samples, str) else \
            [single(s) for s in samples]
        return {'input_ids': res}

    def get_vocab(self):
        return self.tok2id

    def convert_ids_to_tokens(self, ids):
        return [self.id2tok[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_token_ids]
        tokens = self.convert_ids_to_tokens(ids)
        return "".join(tokens)


class BertTokenizerLike:
    def __init__(self, data, custom_vocab_size=30,
                 from_path=None, from_gpt2tokenizer=None, verbose_init=False):
        if from_gpt2tokenizer:
            self.internal = from_gpt2tokenizer
            self._pad_token_type_id = self.internal._pad_token_type_id
            self.unk_token_id = self.internal.unk_token_id
        else:
            if from_path:
                self.internal = _load_custom_tokenizer(from_path)
            else:
                self.internal = make_tokenizer(data,
                                               vocab_size=custom_vocab_size,
                                               eos='[EOS]', bos='[BOS]',
                                               pad='[PAD]',
                                               verbose=verbose_init)
            self._pad_token_type_id = self.internal.get_vocab()['[PAD]']

    def __call__(self, samples):
        def single(s):
            # love too save a Tokenizer but load a PreTrainedTokenizerFast,
            # because a tokenizer cannot be loaded but a
            # pretrainedtokenizerfast cannot be saved, and then find out that
            # also these two things have different behaviour.
            if isinstance(self.internal, tokenizers.Tokenizer):
                res = self.internal.encode(s).ids
            else:  # PreTrainedTokenizerFast
                res = self.internal(s)['input_ids']
            if isinstance(self.internal, GPT2Tokenizer):
                res = [self.internal.bos_token_id] + res + \
                      [self.internal.eos_token_id]
            return res
        res = single(samples) if isinstance(samples, str) else \
            [single(s) for s in samples]
        return {'input_ids': res}  # thats how the bert ones do it
        # (though they also send other info that i dont care about rn)

    def get_vocab(self):
        return self.internal.get_vocab()

    def convert_ids_to_tokens(self, ids):
        return self.internal.convert_ids_to_tokens(ids)

    def decode(self, ids, skip_special_tokens=True):
        return self.internal.decode(ids,
                                    skip_special_tokens=skip_special_tokens)


class MyTokenizer:
    def __init__(self, data=None, name="bert-base-uncased",
                 custom_vocab_size=30, from_path=None, verbose_init=False,
                 no_crop=False):
        self.masking_cropped = False
        self.no_crop = no_crop
        self.name = name
        self.from_path = from_path
        self.verbose_init = verbose_init
        if from_path:
            self.tokenizer = BertTokenizerLike(None, from_path=from_path,
                                               verbose_init=verbose_init)
            self.pad_token_id = self.tokenizer._pad_token_type_id

        elif self.name == "char":
            self.tokenizer = CharTokenizer(data, verbose_init=verbose_init)
            self.pad_token_id = self.tokenizer._pad_token_type_id

        elif self.name == "custom":
            self.tokenizer = BertTokenizerLike(
                data, custom_vocab_size=custom_vocab_size,
                verbose_init=verbose_init)
            self.pad_token_id = self.tokenizer._pad_token_type_id

        elif "gpt2" in self.name:
            gpt2tok = GPT2Tokenizer.from_pretrained(self.name)
            self.tokenizer = BertTokenizerLike(data,
                                               from_gpt2tokenizer=gpt2tok,
                                               verbose_init=verbose_init)
            self._crop_from(data)
            self.pad_token_id = self.tokenizer._pad_token_type_id

        elif "bert" in self.name:  # eg bert-base-uncased
            self.tokenizer = BertTokenizer.from_pretrained(name)
            self.tokenizer.add_tokens(["\n"])
            self._crop_from(data)
            self.pad_token_id = self.tokenizer._pad_token_type_id
            # the lmtrainer needs this (the ignored value) for the cross
            # entropy calculation
        else:
            print("\n\n!!unrecognised tokenizer name!:", self.name)

    def save(self, path):
        def full_path(p):
            return f"{p}/tokenizer.json"
        if self.name == "custom":
            self.tokenizer.internal.save(full_path(path))
        elif self.from_path:  # if for some reason making a copy somewhere else
            with open(full_path(self.from_path), "r") as f:
                j = f.readlines()
            with open(full_path(path), "w") as f:
                f.writelines(j)
        else:
            pass  # dont need to save bert/gpt2 types - theyre online

    @timed
    def _crop_from(self, data):
        # data: list of inputs, eg ["hi","i am a sample"]
        if self.no_crop:
            return
        v_orig = self.vocab_size()
        if self.verbose_init:
            print("vocab size before cropping tokenizer:", self.vocab_size())

        actual_used_ids = set()
        jump = 50
        if self.verbose_init:
            print("cropping tokenizer")
        therange = range(0, len(data), jump)
        therange = tqdm(therange) if self.verbose_init else therange
        for i in therange:
            b = data[i: i + jump]
            t = self(b)
            for tok_id in t:
                actual_used_ids.update(tok_id)
        # make sure these get in
        actual_used_ids.update([self.tokenizer.unk_token_id,
                                self.tokenizer._pad_token_type_id])

        self.ids_self2tokenizer = sorted(list(actual_used_ids))
        self.ids_tokenizer2self = {t: s for s, t in
                                   enumerate(self.ids_self2tokenizer)}
        self._unk_id = self.ids_tokenizer2self[self.tokenizer.unk_token_id]
        self._pad_token_type_id = \
            self.ids_tokenizer2self[self.tokenizer._pad_token_type_id]
        self.masking_cropped = True
        if self.verbose_init:
            print("vocab size after cropping tokenizer:", self.vocab_size())
            print("num tokens cropped:", v_orig - self.vocab_size())

    def get_vocab(self):
        r = self.tokenizer.get_vocab()  # gives {tok:id} dict
        if self.masking_cropped:
            rr = {t: i for i, t in r.items()}
            r = {rr[i]: self.ids_tokenizer2self[i] for i in
                 self.ids_self2tokenizer}
        return r

    def eos(self):
        ids = self("")
        assert len(self("")) == 2  # bos, eos
        return self("")[-1]

    def tokenize_without_stop(self, s):
        tokens = self(s)
        check = self("")
        if len(check) == 2:  # then can assume there is an eos and bos
            assert check[0] == tokens[0]
            assert check[-1] == tokens[-1]
            tokens = tokens[:-1]
        return tokens

    def multiids_self2tokenizer(self, ids):
        if not self.masking_cropped:
            return ids
        return [self.ids_self2tokenizer[i] for i in ids]

    def multiids_tokenizer2self(self, ids):
        if not self.masking_cropped:
            return ids
        return [self.ids_tokenizer2self.get(i, self._unk_id) for i in ids]
        # could get an input that the underlying tokenizer does recognise, but
        # we dont, so we send it to unk

    def convert_ids_to_tokens(self, ids):
        ids = self.multiids_self2tokenizer(ids)
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_ids_to_nice_string(self, ids, skip_special_tokens=True):
        ids = self.multiids_self2tokenizer(ids)
        return self.tokenizer.decode(ids,
                                     skip_special_tokens=skip_special_tokens)

    def __call__(self, samples, max_length=None):
        def finalise_single(s):
            res = self.multiids_tokenizer2self(s)
            if max_length:
                res = res[:max_length]
            return res

        internal_res = self.tokenizer(samples)['input_ids']
        if internal_res and isinstance(internal_res[0], int):
            # single sample ('if res' to avoid crashing on no samples)
            return finalise_single(internal_res)
        else:
            return [finalise_single(line) for line in internal_res]

    def vocab_size(self):
        d = self.get_vocab()
        lst = list(d.values())
        assert False not in [i >= 0 for i in lst]
        return max(lst) + 1  # +1 to account for the '0' token
