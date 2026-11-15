"""Text to batches: tokenize, cut the rare words, encode, hold out, window.

Corpora are plain .txt files in data/. `tinytales` is generated on first use
so the app works with no download; `python -m opengpt fetch` gets TinyStories.
"""

import re
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import tinytales

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TINYSTORIES_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
                   "resolve/main/TinyStories-valid.txt")

TOKEN_RE = re.compile(r"<eot>|[a-z]+'[a-z]+|[a-z]+|[.,!?;:\"]")


def tokenize(text):
    text = text.lower().replace("<|endoftext|>", " <eot> ")
    return TOKEN_RE.findall(text)


def build_vocab(tokens, min_freq=10):
    counts = Counter(tokens)
    words = ["<unk>"] + sorted(w for w, n in counts.items() if n >= min_freq)
    stoi = {w: i for i, w in enumerate(words)}
    return words, stoi


def encode(tokens, stoi):
    unk = stoi["<unk>"]
    return np.array([stoi.get(t, unk) for t in tokens], dtype=np.int64)


def decode(ids, words):
    out = []
    for i in ids:
        w = words[int(i)]
        if out and w in ".,!?;:":
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out)


def split(data, frac=0.9):
    n = int(frac * len(data))
    return data[:n], data[n:]          # train, val (held out)


def get_batch(src, rng, B, T):
    ix = rng.integers(0, len(src) - T, size=B)
    x = np.stack([src[i:i + T] for i in ix])
    y = np.stack([src[i + 1:i + 1 + T] for i in ix])
    return x, y


# ---- corpora ----------------------------------------------------------------------
def corpus_path(name):
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{name}.txt"
    if name == "tinytales" and not path.exists():
        path.write_text(tinytales.generate())
    if not path.exists():
        raise FileNotFoundError(f"no corpus {name!r}: put {name}.txt in {DATA}")
    return path


def list_corpora():
    corpus_path("tinytales")
    return sorted(p.stem for p in DATA.glob("*.txt"))


def fetch_tinystories():
    DATA.mkdir(exist_ok=True)
    path = DATA / "tinystories.txt"
    urllib.request.urlretrieve(TINYSTORIES_URL, path)
    return path


class Dataset:
    def __init__(self, name, min_freq):
        self.name, self.min_freq = name, min_freq
        text = corpus_path(name).read_text()
        tokens = tokenize(text)
        self.words, self.stoi = build_vocab(tokens, min_freq)
        self.data = encode(tokens, self.stoi)
        self.train, self.val = split(self.data)
        self.V = len(self.words)
        counts = np.bincount(self.data, minlength=self.V)
        top = np.argsort(-counts)[:30]
        vc = np.bincount(self.val, minlength=self.V)
        self.stats = {
            "name": name, "min_freq": min_freq, "chars": len(text),
            "tokens": int(len(self.data)), "vocab": self.V,
            "distinct_words": len(set(tokens)),
            "unk_rate": float((self.data == 0).mean()),
            "train_tokens": int(len(self.train)), "val_tokens": int(len(self.val)),
            "top": [[self.words[i], int(counts[i])] for i in top],
            # always guessing the most common held-out word: the bar accuracy must beat
            "baseline_acc": float(vc.max() / max(1, len(self.val))),
            "baseline_word": self.words[int(vc.argmax())],
            "chance_loss": float(np.log(self.V)),
            "sample": text[:600],
        }

    def encode_text(self, text):
        toks = tokenize(text)
        return toks, [int(i) for i in encode(toks, self.stoi)]


@lru_cache(maxsize=4)
def load(name="tinytales", min_freq=1):
    return Dataset(name, int(min_freq))
