from transformers import BertTokenizer, GPT2Tokenizer
from util import timed
import tokenizers
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import json


@timed
def make_bpe_tokenizer(data, vocab_size=30, eos="[EOS]", bos="[BOS]",
                       pad="[PAD]", unk="[UNK]", verbose=False):
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


def tokenizer_file(p):
    return f"{p}/tokenizer.json"


def _load_custom_tokenizer(path):
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file(path))


def load_stored_tokenizer_if_exists(source_name, folder_name, verbose):
    if (source_name in ["custom", "char"]) or \
       (True in [n in source_name for n in ["gpt2", "bert"]]):
        # parallel conditions to those of 'save' functions in MyTokenizer below
        return MyTokenizer(name=source_name, from_path=folder_name,
                           verbose_init=verbose)
    return None


class CharTokenizer:
    # made with the BertTokenizer function signatures my functions expect to
    # find
    def __init__(self, data=None, verbose_init=False, from_dict=None):
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.special_tokens = [self.unk_token, self.pad_token, self.bos,
                               self.eos]

        if None is not from_dict:
            id2tok = from_dict["id2tok"]
        else:
            id2tok = self.make_tokens(data)

        self.finish_init(id2tok, verbose_init)

    def make_tokens(self, data):
        tokens = set()
        assert isinstance(data, list)
        for s in data:
            tokens.update(list(s))
        tokens.update(self.special_tokens)
        return sorted(list(tokens))

    def finish_init(self, id2tok, verbose_init):
        self.id2tok = id2tok
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

    def save_dict(self):
        return {"id2tok": self.id2tok}


class BertTokenizerLike:
    def __init__(self, data=None, custom_vocab_size=30,
                 from_path=None, from_gpt2tokenizer=None, verbose_init=False):
        if from_gpt2tokenizer:
            self.internal = from_gpt2tokenizer
            self._pad_token_type_id = self.internal._pad_token_type_id
            self.unk_token_id = self.internal.unk_token_id
        else:
            if from_path:
                self.internal = _load_custom_tokenizer(from_path)
            else:
                self.internal = make_bpe_tokenizer(
                    data, vocab_size=custom_vocab_size, eos='[EOS]',
                    bos='[BOS]', pad='[PAD]', verbose=verbose_init)
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
        if not hasattr(self.internal, "convert_ids_to_tokens"):
            # tokenizers.Tokenizer doesn't have this attribute, but it does
            # have id_to_token
            return [self.internal.id_to_token(i) for i in ids]
        return self.internal.convert_ids_to_tokens(ids)

    def decode(self, ids, skip_special_tokens=True):
        return self.internal.decode(ids,
                                    skip_special_tokens=skip_special_tokens)


def getBertLikeTokenizer(name, data=None, custom_vocab_size=30,
                         verbose_init=False):
    if name == "char":
        return CharTokenizer(data, verbose_init=verbose_init)
    if name == "custom":
        return BertTokenizerLike(data=data,
                                 custom_vocab_size=custom_vocab_size,
                                 verbose_init=verbose_init)
    if "gpt2" in name:
        gpt2tok = GPT2Tokenizer.from_pretrained(name)
        return BertTokenizerLike(from_gpt2tokenizer=gpt2tok,
                                 verbose_init=verbose_init)
    if "bert" in name:
        res = BertTokenizer.from_pretrained(name)
        res.add_tokens(["\n"])  # i prefer the tokenizer to have this
        return res
    print("\n\n!!unrecognised tokenizer name!:", self.name)


class MyTokenizer:
    def __init__(self, data=None, name="bert-base-uncased",
                 custom_vocab_size=30, from_path=None, verbose_init=False,
                 no_crop=False):
        self.masking_cropped = False
        self.verbose_init = verbose_init
        if None is not from_path:
            self.init_from_path(from_path)
        else:
            self.no_crop = no_crop
            self.name = name
            self.is_from_HF = True in [n in name for n in ["gpt2", "bert"]]
            self.tokenizer = getBertLikeTokenizer(
                name, data=data, custom_vocab_size=custom_vocab_size,
                verbose_init=verbose_init)
            if self.is_from_HF and not self.no_crop:
                self.prepare_crop(data)
                self.apply_crop()
        self.pad_token_id = self.tokenizer._pad_token_type_id
        # the trainer needs this (the ignored value) for the cross
        # entropy calculation

    def init_from_path(self, path):
        with open(tokenizer_file(path), "r") as f:
            core = json.load(f)
        if isinstance(core, dict) and core.get("core-dict", False):
            self.name = core["name"]
            self.is_from_HF = core["is_from_HF"]
            self.no_crop = core["no_crop"]
            self.masking_cropped = core["masking_cropped"]
            if self.is_from_HF:
                self.tokenizer = getBertLikeTokenizer(self.name)
                if self.masking_cropped:
                    self.ids_self2tokenizer = core["ids_self2tokenizer"]
                    self.v_orig = core["v_orig"]
                    self.apply_crop()
            elif self.name == "char":
                self.tokenizer = CharTokenizer(
                    from_dict=core["tokenizer-dict"])
        else:  # used custom tokenizer internal save
            self.name = "custom"
            self.tokenizer = BertTokenizerLike(from_path=path,
                                               verbose_init=self.verbose_init)

    def save(self, path):
        if self.name == "custom":
            self.tokenizer.internal.save(tokenizer_file(path))
            return
        elif self.is_from_HF or (self.name == "char"):
            core = {"core-dict": True,
                    "name": self.name,
                    "is_from_HF": self.is_from_HF,
                    "no_crop": self.no_crop,
                    "masking_cropped": self.masking_cropped}
            if self.name == "char":
                core["tokenizer-dict"] = self.tokenizer.save_dict()
                core["core-dict"] = True
            if self.masking_cropped:
                core["ids_self2tokenizer"] = self.ids_self2tokenizer
                core["v_orig"] = self.v_orig
            with open(tokenizer_file(path), "w") as f:
                json.dump(core, f)
        else:
            print("\n\n!!unrecognised tokenizer!:", self.name,
                  "\n - not saved in path", path)

    @timed
    def prepare_crop(self, data):
        # data: list of inputs, eg ["hi","i am a sample"]
        self.v_orig = self.vocab_size()

        actual_used_ids = set()
        jump = 50
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

    def apply_crop(self):
        if self.verbose_init:
            print("vocab size before cropping tokenizer:", self.v_orig)

        self.ids_tokenizer2self = {t: s for s, t in
                                   enumerate(self.ids_self2tokenizer)}
        self._unk_id = self.ids_tokenizer2self[self.tokenizer.unk_token_id]
        self._pad_token_type_id = \
            self.ids_tokenizer2self[self.tokenizer._pad_token_type_id]
        self.masking_cropped = True

        if self.verbose_init:
            print("vocab size after cropping tokenizer:", self.vocab_size())
            print("num tokens cropped:", self.v_orig - self.vocab_size())

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
