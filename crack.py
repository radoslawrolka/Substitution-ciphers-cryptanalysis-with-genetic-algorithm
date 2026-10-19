#!/usr/bin/env python3
"""Breaks a substitution ciphertext (ciphertext-only attack, COA).

Usage:
  python3 crack.py ciphertext.txt                  # text from a file (English by default)
  python3 crack.py --lang pl "GDKZ ..."            # text as an argument, Polish
  python3 crack.py --spaces ciphertext.txt         # ciphertext with visible word spaces
  python3 crack.py --demo --lang en --length 500   # encrypt a random test-book fragment and break it

Attack: GA in the default configuration (``gasub.ga.GAConfig``). A ciphertext
of your own text is produced by ``encrypt.py``.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from gasub import cipher, data, ngrams
from gasub.ga import GAConfig, evolve


def letters(key) -> str:
    return "".join(data.ALPHABET[k] for k in key)


def main() -> int:
    defaults = GAConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ciphertext", nargs="?", help="file or ciphertext")
    ap.add_argument("--lang", default="en", choices=["en", "pl"])
    ap.add_argument("--spaces", action="store_true", help="word spaces in the ciphertext are visible")
    ap.add_argument("--n", type=int, default=4, help="n-gram order (default 4)")
    ap.add_argument("--population", type=int, default=defaults.population)
    ap.add_argument("--generations", type=int, default=defaults.generations)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="encrypt a random test-book fragment and break it")
    ap.add_argument("--length", type=int, default=500, help="fragment length in --demo mode")
    a = ap.parse_args()

    plaintext = None
    if a.demo:
        rng = random.Random(a.seed)
        plaintext, book = data.random_test_fragment(a.lang, a.length, rng, a.spaces)
        enc_key = cipher.random_key(rng)
        text = cipher.encrypt(plaintext, enc_key)
        print(f"Fragment: {book}\nEncryption key: {letters(enc_key)}")
    elif a.ciphertext:
        p = Path(a.ciphertext)
        text = data.normalize(p.read_text(encoding="utf-8") if p.exists() else a.ciphertext, a.spaces)
    else:
        ap.print_help()
        return 1
    if len(text) < a.n:
        print("Ciphertext too short.")
        return 1

    print(f"Ciphertext ({len(text)} characters):\n{text}\n")
    m = ngrams.model(a.lang, a.n, a.spaces)
    fitness = ngrams.FitnessFunction(m, cipher.encode(text))
    r = evolve(fitness, GAConfig(population=a.population, generations=a.generations), seed=a.seed)
    decrypted = cipher.decrypt(text, r.key)
    print(f"GA: {r.generations} generations, {r.evaluations} evaluations, {r.time:.1f} s, F = {r.fitness:.1f}")
    print(f"Decryption key (cipher -> plain): {letters(r.key)}")
    print(f"Corresponding encryption key:     {letters(cipher.invert(r.key))}")
    print(f"\nDecrypted text:\n{decrypted}")
    if plaintext is not None:
        acc = cipher.letter_accuracy(cipher.encode(plaintext), cipher.encode(decrypted))
        print(f"\nLetter accuracy: {acc:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
