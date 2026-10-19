"""Monoalphabetic substitution cipher and decryption quality metrics.

The encryption key is a permutation ``e`` of {0..25}: plaintext letter ``x``
is replaced by ``e[x]``. In cryptanalysis it is more convenient to work with
the *decryption* key ``d = e^{-1}`` (``d[y]`` = plaintext letter for
ciphertext letter ``y``), and that key is the individual in the genetic
algorithm. The space symbol (index 26) is never encrypted.
"""
from __future__ import annotations

import random

import numpy as np

from .data import ALPHABET, N_LETTERS, SPACE


def encode(text: str) -> np.ndarray:
    """Text (A-Z and spaces) -> array of indices 0..25, space -> 26."""
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("A") \
        if " " not in text else np.array([SPACE if ch == " " else ord(ch) - ord("A") for ch in text],
                                          dtype=np.int64)


def decode(idx: np.ndarray) -> str:
    return "".join(" " if i == SPACE else ALPHABET[i] for i in idx)


def random_key(rng: random.Random) -> list[int]:
    """Random permutation of the 26 letters."""
    return rng.sample(range(N_LETTERS), N_LETTERS)


def invert(key) -> list[int]:
    """Inverse permutation (encryption key <-> decryption key)."""
    inv = [0] * N_LETTERS
    for i, k in enumerate(key):
        inv[k] = i
    return inv


def substitution_table(key) -> np.ndarray:
    """Key extended with the space symbol (26 -> 26) as an index table."""
    table = np.empty(N_LETTERS + 1, dtype=np.int64)
    table[:N_LETTERS] = key
    table[SPACE] = SPACE
    return table


def apply_key(idx: np.ndarray, key) -> np.ndarray:
    return substitution_table(key)[idx]


def encrypt(text: str, enc_key) -> str:
    return decode(apply_key(encode(text), enc_key))


def decrypt(ciphertext: str, dec_key) -> str:
    return decode(apply_key(encode(ciphertext), dec_key))


def letter_accuracy(plain_idx: np.ndarray, decrypted_idx: np.ndarray) -> float:
    """Fraction of letter positions (spaces excluded) where the decrypted
    letter matches the plaintext -- the main success metric of the attack."""
    mask = plain_idx != SPACE
    return float(np.mean(plain_idx[mask] == decrypted_idx[mask]))


def key_accuracy(key, true_key) -> float:
    """Fraction of the 26 substitutions that match the true key."""
    return float(np.mean(np.asarray(key) == np.asarray(true_key)))


def letter_counts(idx: np.ndarray) -> np.ndarray:
    """Letter counts (spaces excluded)."""
    return np.bincount(idx[idx != SPACE], minlength=N_LETTERS)
