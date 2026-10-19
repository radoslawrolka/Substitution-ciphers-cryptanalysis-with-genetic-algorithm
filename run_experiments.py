#!/usr/bin/env python3
"""Project experiments. Results are written to ``results/*.json``.

  main     -- attack attempt per the project criterion: GA, 500 characters
              (letters only), EN and PL, 50 trials per language, with convergence curves

Every trial uses a different random test-book fragment and a different random
key (seed = trial number); the GA always runs in the default configuration
(gasub.ga.GAConfig).

Usage:  python3 run_experiments.py [--repeats 50] [--only main] [--processes 12]
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
    ap.add_argument("--repeats", type=int, default=50, help="trials per language")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="names of the experiments to run")
    a = ap.parse_args()
    S = range(a.repeats)
    selected = set(a.only) if a.only else None

    def wanted(name: str) -> bool:
        return selected is None or name in selected

    if wanted("main"):
        run_experiment("main", both_languages(S, "GA", curve=True),
                       "Attack attempt per the project criterion: GA, 500 characters (letters only), EN and PL.",
                       a.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
