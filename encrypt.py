#!/usr/bin/env python3
"""Encrypts a plaintext with the monoalphabetic substitution cipher.

Usage:
  python3 encrypt.py "plain text"                          # random key (seed --seed), letters A-Z only
  python3 encrypt.py --spaces text.txt                     # word spaces are kept visible
  python3 encrypt.py --key RELTPUSCAVIQDNXHOJKMFBYWZG text.txt
  python3 encrypt.py --spaces -o ciphertext.txt text.txt   # ciphertext alone to a file (input for crack.py)

The key is 26 letters: the i-th letter of the key is the image of the i-th
letter of the alphabet (for the key RELT... A -> R, B -> E, C -> L, ...).
The text is normalised first (capital A-Z, Polish letters without diacritics,
no punctuation).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from gasub import cipher, data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", help="file or plain text")
    ap.add_argument("--key", help="26 letters (a permutation of the alphabet); random by default")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random key")
    ap.add_argument("--spaces", action="store_true", help="keep word spaces")
    ap.add_argument("-o", "--output", help="write the ciphertext to a file")
    a = ap.parse_args()

    p = Path(a.text)
    plaintext = data.normalize(p.read_text(encoding="utf-8") if p.exists() else a.text, a.spaces)
    if not plaintext:
        ap.error("empty plaintext")
    if a.key:
        letters = a.key.upper()
        if sorted(letters) != list(data.ALPHABET):
            ap.error("the key must be a permutation of the 26 letters A-Z")
        key = [ord(ch) - ord("A") for ch in letters]
    else:
        key = cipher.random_key(random.Random(a.seed))
    ciphertext = cipher.encrypt(plaintext, key)

    print(f"Encryption key: {''.join(data.ALPHABET[k] for k in key)}")
    print(f"Plaintext ({len(plaintext)} characters):\n{plaintext}\n")
    if a.output:
        Path(a.output).write_text(ciphertext + "\n", encoding="utf-8")
        print(f"Ciphertext (saved to {a.output}):\n{ciphertext}")
    else:
        print(f"Ciphertext:\n{ciphertext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
