"""N-gram language model and the fitness function.

The fitness of a key ``d`` for a ciphertext ``y`` is the log-likelihood of the
decrypted text:  F(d) = sum_i log10 P(g_i),  where g_i are the consecutive
n-grams of ``d(y)`` and P is the smoothed n-gram distribution estimated from
the training corpus. The higher F, the more the text "looks like" language.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .cipher import encode, substitution_table
from .data import DATA_DIR, N_LETTERS, corpus

MODEL_DIR = DATA_DIR / "ngrams"


class NgramModel:
    def __init__(self, n: int, V: int, logp: np.ndarray, ngram_count: int, lang: str = "?"):
        self.n, self.V, self.logp, self.ngram_count, self.lang = n, V, logp, ngram_count, lang
        self.weights = V ** np.arange(n - 1, -1, -1, dtype=np.int64)

    # -- training / storage --------------------------------------------------
    @classmethod
    def train(cls, idx: np.ndarray, n: int, V: int, alpha: float = 0.01, lang: str = "?") -> "NgramModel":
        """Counts the n-grams of an encoded text and applies additive smoothing (alpha)."""
        ind = cls._indices(idx, n, V)
        counts = np.bincount(ind, minlength=V ** n).astype(np.float64)
        logp = np.log10((counts + alpha) / (counts.sum() + alpha * V ** n)).astype(np.float32)
        return cls(n, V, logp, int(counts.sum()), lang)

    @staticmethod
    def _indices(idx: np.ndarray, n: int, V: int) -> np.ndarray:
        L = len(idx) - n + 1
        ind = np.zeros(L, dtype=np.int64)
        for k in range(n):
            ind = ind * V + idx[k:k + L]
        return ind

    def save(self, path: Path) -> None:
        """Atomic write (temporary file + rename), so that parallel processes
        never read a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, n=self.n, V=self.V, logp=self.logp, count=self.ngram_count, lang=self.lang)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "NgramModel":
        z = np.load(path)
        return cls(int(z["n"]), int(z["V"]), z["logp"], int(z["count"]), str(z["lang"]))

    # -- scoring -------------------------------------------------------------
    def score(self, idx: np.ndarray) -> float:
        """Sum of log10 P over all n-grams of the encoded text."""
        return float(self.logp[self._indices(idx, self.n, self.V)].sum())

    def mean_score(self, idx: np.ndarray) -> float:
        return self.score(idx) / max(1, len(idx) - self.n + 1)


def model(lang: str, n: int, keep_spaces: bool = False, alpha: float = 0.01) -> NgramModel:
    """Returns the n-gram model (loaded from the cache or trained on the
    training corpus of the language)."""
    V = N_LETTERS + 1 if keep_spaces else N_LETTERS
    path = MODEL_DIR / f"{lang}_n{n}_V{V}.npz"
    if path.exists():
        return NgramModel.load(path)
    idx = encode(corpus(lang, "train", keep_spaces))
    m = NgramModel.train(idx, n, V, alpha, lang)
    m.save(path)
    return m


class FitnessFunction:
    """Fitness function for a fixed ciphertext: F(d) = score(d(y)).
    The ciphertext is encoded once; every evaluation is a handful of vectorised
    NumPy operations (indexing with the substitution table, computing the
    n-gram indices, summing)."""

    def __init__(self, model: NgramModel, ciphertext_idx: np.ndarray):
        self.model = model
        self.cipher_idx = np.asarray(ciphertext_idx, dtype=np.int64)
        self.n, self.V = model.n, model.V
        L = len(self.cipher_idx) - self.n + 1
        self.windows = [slice(k, k + L) for k in range(self.n)]
        self.evaluations = 0

    def __call__(self, key) -> float:
        self.evaluations += 1
        dec = substitution_table(key)[self.cipher_idx]
        idx = dec[self.windows[0]].copy()
        for window in self.windows[1:]:
            idx *= self.V
            idx += dec[window]
        return float(self.model.logp[idx].sum())


def language_frequencies(lang: str) -> np.ndarray:
    """Letter distribution (unigrams) of the language -- from the n=1 model."""
    m = model(lang, 1)
    p = 10.0 ** m.logp[:N_LETTERS].astype(np.float64)
    return p / p.sum()
