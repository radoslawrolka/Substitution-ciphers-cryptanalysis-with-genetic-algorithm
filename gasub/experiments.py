"""Attack experiments: task generation (plaintext -> ciphertext), GA runs,
metrics and JSON output.

A *task* is a random fragment of a test-set book encrypted with a random key
(the ``seed`` determines both draws and the GA run, so every trial is fully
reproducible).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cipher, data, ngrams
from .ga import GAConfig, evolve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SUCCESS_THRESHOLD = 0.95     # project criterion: letter accuracy > 95 %

_MODELS: dict = {}           # per-process model cache


def cached_model(lang: str, n: int, spaces: bool) -> ngrams.NgramModel:
    key = (lang, n, spaces)
    if key not in _MODELS:
        _MODELS[key] = ngrams.model(lang, n, spaces)
    return _MODELS[key]


@dataclass
class Task:
    lang: str
    length: int
    spaces: bool
    seed: int
    plaintext: str
    ciphertext: str
    enc_key: list[int]
    book: str

    @property
    def dec_key(self) -> list[int]:
        return cipher.invert(self.enc_key)


def make_task(lang: str, length: int, seed: int, spaces: bool = False) -> Task:
    """Phase 1: a random test-book fragment encrypted with a random key."""
    rng = random.Random(f"{lang}-{length}-{spaces}-{seed}")
    plaintext, book = data.random_test_fragment(lang, length, rng, spaces)
    key = cipher.random_key(rng)
    return Task(lang, length, spaces, seed, plaintext, cipher.encrypt(plaintext, key), key, book)


def evaluate_key(task: Task, key: list[int]) -> dict:
    plain_idx, cipher_idx = cipher.encode(task.plaintext), cipher.encode(task.ciphertext)
    decrypted = cipher.apply_key(cipher_idx, key)
    acc = cipher.letter_accuracy(plain_idx, decrypted)
    return {"accuracy": acc, "key_accuracy": cipher.key_accuracy(key, task.dec_key),
            "success": bool(acc > SUCCESS_THRESHOLD), "decrypted": cipher.decode(decrypted)[:120]}


def run_trial(spec: dict) -> dict:
    """Runs one GA attack trial: ``spec`` = {params, lang, length, spaces, seed, n, curve, label}."""
    task = make_task(spec["lang"], spec["length"], spec["seed"], spec.get("spaces", False))
    n = spec.get("n", 4)
    m = cached_model(task.lang, n, task.spaces)
    cipher_idx = cipher.encode(task.ciphertext)
    fitness = ngrams.FitnessFunction(m, cipher_idx)
    true_fitness = fitness(task.dec_key)        # F* -- fitness of the true key
    fitness.evaluations = 0
    cfg = GAConfig(**spec.get("params", {}))
    t0 = time.perf_counter()
    r = evolve(fitness, cfg, seed=spec["seed"], record_keys=spec.get("curve", False))

    result: dict = {
        "generations": r.generations,
        "history": [(g, round(fm, 2), round(fa, 2)) for g, fm, fa in r.history[::max(1, len(r.history) // 150)]],
        "curve": None, "success_generation": None,
    }
    if r.keys:   # convergence curve: accuracy of the best individual in every generation
        plain_idx = cipher.encode(task.plaintext)
        curve = [cipher.letter_accuracy(plain_idx, cipher.apply_key(cipher_idx, k)) for k in r.keys]
        result["curve"] = [round(v, 4) for v in curve]
        result["success_generation"] = next((g + 1 for g, v in enumerate(curve) if v > SUCCESS_THRESHOLD), None)
    result.update(evaluate_key(task, r.key))
    result.update({
        "label": spec.get("label", "GA"), "method": "ga", "params": cfg.as_dict(),
        "lang": task.lang, "length": task.length, "spaces": task.spaces, "n": n, "seed": task.seed,
        "book": task.book, "time": time.perf_counter() - t0, "evaluations": fitness.evaluations,
        "fitness": float(fitness(r.key)), "true_fitness": float(true_fitness),
        "key": list(map(int, r.key)),
    })
    return result


def run_experiment(name: str, specs: list[dict], description: str = "", processes: int | None = None,
                   path: Path | None = None) -> dict:
    """Runs a list of trials in parallel and writes ``results/<name>.json``."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = path or RESULTS_DIR / f"{name}.json"
    t0 = time.perf_counter()
    processes = processes or max(1, mp.cpu_count() - 2)
    # Train the n-gram models in the main process before the worker pool
    # starts (avoids parallel training and writing of the same file).
    for lang, n, spaces in sorted({(s["lang"], s.get("n", 4), s.get("spaces", False)) for s in specs}):
        cached_model(lang, n, spaces)
    print(f"[{name}] {len(specs)} trials on {processes} processes ...", flush=True)
    if processes == 1:
        trials = [run_trial(s) for s in specs]
    else:
        with mp.get_context("fork").Pool(processes) as pool:
            trials = pool.map(run_trial, specs, chunksize=1)
    result = {"name": name, "description": description, "total_time": time.perf_counter() - t0, "trials": trials}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    success = np.mean([t["success"] for t in trials])
    acc = np.mean([t["accuracy"] for t in trials])
    print(f"[{name}] done in {result['total_time']:.0f} s: mean accuracy {acc:.3f}, "
          f"success {success:.0%}  -> {path.name}", flush=True)
    return result


def summarize(trials: list[dict]) -> dict:
    a = np.array([t["accuracy"] for t in trials])
    return {
        "count": len(trials),
        "acc_mean": float(a.mean()), "acc_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "acc_median": float(np.median(a)), "acc_min": float(a.min()),
        "success_rate": float(np.mean([t["success"] for t in trials])),
        "time_mean": float(np.mean([t["time"] for t in trials])),
        "evaluations_mean": float(np.mean([t["evaluations"] for t in trials])),
        "fitness_ge_true": float(np.mean([t["fitness"] >= t["true_fitness"] - 1e-6 for t in trials])),
    }
