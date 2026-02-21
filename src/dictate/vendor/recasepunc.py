"""Vendored inference code from benob/recasepunc (MIT License).

Source: https://github.com/benob/recasepunc
Only the inference-related classes are included - training code (~570 lines) is omitted.
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer

# Label mappings
punctuation = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3, "EXCLAMATION": 4}
punctuation_syms = ["", ",", ".", " ?", " !"]
case = {"LOWER": 0, "UPPER": 1, "CAPITALIZE": 2, "OTHER": 3}

default_config = argparse.Namespace(
    seed=871253,
    lang="en",
    flavor="bert-base-uncased",
    max_length=256,
    batch_size=16,
    device="cuda",
)


class Config(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in default_config.__dict__.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WordpieceTokenizer:
    """WordPiece tokenizer that preserves case information."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr.lower() in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class Model(nn.Module):
    def __init__(self, flavor, device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(flavor)
        size = (
            self.bert.dim
            if hasattr(self.bert, "dim")
            else self.bert.config.pooler_fc_size
            if hasattr(self.bert.config, "pooler_fc_size")
            else self.bert.config.emb_dim
            if hasattr(self.bert.config, "emb_dim")
            else self.bert.config.hidden_size
        )
        self.punc = nn.Linear(size, 5)
        self.case = nn.Linear(size, 4)
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        output = self.bert(x)
        representations = self.dropout(F.gelu(output["last_hidden_state"]))
        punc = self.punc(representations)
        case_out = self.case(representations)
        return punc, case_out


def _init_config(config):
    """Initialize tokenizer and special tokens on the config object."""
    config.tokenizer = BertTokenizer.from_pretrained(config.flavor, do_lower_case=False)
    config.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=config.tokenizer.vocab, unk_token=config.tokenizer.unk_token
    )
    config.pad_token_id = config.tokenizer.pad_token_id
    config.cls_token_id = config.tokenizer.cls_token_id
    config.cls_token = config.tokenizer.cls_token
    config.sep_token_id = config.tokenizer.sep_token_id
    config.sep_token = config.tokenizer.sep_token

    if not torch.cuda.is_available() and config.device == "cuda":
        print("WARNING: reverting to cpu as cuda is not available", file=sys.stderr)
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")


def recase(token, label):
    """Apply case label to a token."""
    if label == case["LOWER"]:
        return token.lower()
    if label == case["CAPITALIZE"]:
        return token.lower().capitalize()
    if label == case["UPPER"]:
        return token.upper()
    return token


class CasePuncPredictor:
    """Loads a recasepunc checkpoint and predicts punctuation + casing."""

    def __init__(self, checkpoint_path, lang="en", flavor="bert-base-uncased", device="cuda"):
        loaded = torch.load(
            checkpoint_path, map_location=device if torch.cuda.is_available() else "cpu"
        )
        if "config" in loaded:
            self.config = Config(**loaded["config"])
        else:
            self.config = Config(lang=lang, flavor=flavor, device=device)
        _init_config(self.config)

        self.model = Model(self.config.flavor, self.config.device)
        self.model.load_state_dict(loaded["model_state_dict"])
        self.model.to(self.config.device)
        self.model.eval()

        self.rev_case = {b: a for a, b in case.items()}
        self.rev_punc = {b: a for a, b in punctuation.items()}

    def tokenize(self, text):
        return (
            [self.config.cls_token]
            + self.config.tokenizer.tokenize(text)
            + [self.config.sep_token]
        )

    def predict(self, tokens, getter=lambda x: x):
        """Yield (token, case_label, punc_label) for each non-special token."""
        max_length = self.config.max_length
        device = self.config.device
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        previous_label = punctuation["PERIOD"]
        for start in range(0, len(tokens), max_length):
            instance = tokens[start : start + max_length]
            if isinstance(getter(instance[0]), str):
                ids = self.config.tokenizer.convert_tokens_to_ids(
                    getter(token) for token in instance
                )
            else:
                ids = [getter(token) for token in instance]
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))
            with torch.no_grad():
                x = torch.tensor([ids]).long().to(device)
                y_scores1, y_scores2 = self.model(x)
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for i, id, token, punc_label, case_label in zip(
                range(len(instance)),
                ids,
                instance,
                y_pred1[0].tolist()[: len(instance)],
                y_pred2[0].tolist()[: len(instance)],
            ):
                if id == self.config.cls_token_id or id == self.config.sep_token_id:
                    continue
                if previous_label is not None and previous_label > 1:
                    if case_label in [case["LOWER"], case["OTHER"]]:
                        case_label = case["CAPITALIZE"]
                if i + start == len(tokens) - 2 and punc_label == punctuation["O"]:
                    punc_label = punctuation["PERIOD"]
                yield (token, self.rev_case[case_label], self.rev_punc[punc_label])
                previous_label = punc_label

    def map_case_label(self, token, case_label):
        if token.endswith("</w>"):
            token = token[:-4]
        if token.startswith("##"):
            token = token[2:]
        return recase(token, case[case_label])

    def map_punc_label(self, token, punc_label):
        if token.endswith("</w>"):
            token = token[:-4]
        if token.startswith("##"):
            token = token[2:]
        return token + punctuation_syms[punctuation[punc_label]]
