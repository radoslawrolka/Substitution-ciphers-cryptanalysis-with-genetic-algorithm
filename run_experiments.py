#!/usr/bin/env python3
"""Project experiments. Results are written to ``results/*.json``.

  main     -- attack attempt per the project criterion: GA, 500 characters
              (letters only), EN and PL, 50 trials per language, with convergence curves
  length   -- effect of the ciphertext length (50-1000 characters), 30 trials per language
  ngrams   -- effect of the n-gram order in the fitness function (1-5), 30 trials per language
  spaces   -- ciphertext with visible word spaces vs letters only, 30 trials per language

Every trial uses a different random test-book fragment and a different random
key (seed = trial number); the GA always runs in the default configuration
(gasub.ga.GAConfig).

Usage:  python3 run_experiments.py [--repeats 30] [--only main length] [--processes 12]
"""
from __future__ import annotations

import argparse

from gasub.experiments import run_experiment
from gasub.ga import GAConfig

DEFAULTS = GAConfig().as_dict()
LANGUAGES = (("en", "EN"), ("pl", "PL"))


def spec(seeds, label: str, lang: str, length: int = 500, spaces: bool = False, n: int = 4,
         curve: bool = False) -> list[dict]:
    return [{"params": DEFAULTS, "lang": lang, "length": length, "spaces": spaces, "n": n,
             "seed": s, "curve": curve, "label": label} for s in seeds]


def both_languages(seeds, label: str, **kw) -> list[dict]:
    out = []
    for lang, tag in LANGUAGES:
        out += spec(seeds, f"{label} ({tag})", lang, **kw)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=30, help="trials per language in the additional experiments")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="names of the experiments to run")
    a = ap.parse_args()
    S = range(a.repeats)
    S_main = range(max(a.repeats, 50))
    selected = set(a.only) if a.only else None

    def wanted(name: str) -> bool:
        return selected is None or name in selected

    if wanted("main"):
        run_experiment("main", both_languages(S_main, "GA", curve=True),
                       "Attack attempt per the project criterion: GA, 500 characters (letters only), EN and PL.",
                       a.processes)
    if wanted("length"):
        specs = []
        for L in (50, 75, 100, 150, 200, 300, 400, 500, 750, 1000):
            specs += both_languages(S, f"L={L}", length=L)
        run_experiment("length", specs, "Effect of the ciphertext length (GA, EN and PL).", a.processes)
    if wanted("ngrams"):
        specs = []
        for n in (1, 2, 3, 4, 5):
            specs += both_languages(S, f"n={n}", n=n)
        run_experiment("ngrams", specs, "Effect of the n-gram order in the fitness function (500 characters, EN and PL).",
                       a.processes)
    if wanted("spaces"):
        run_experiment("spaces", both_languages(S, "letters only", spaces=False) + both_languages(S, "with spaces", spaces=True),
                       "Ciphertext with visible word spaces vs letters only (500 characters).", a.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
