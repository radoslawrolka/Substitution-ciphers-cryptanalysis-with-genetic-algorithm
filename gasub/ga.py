"""Genetic algorithm (DEAP) searching for the decryption key.

Representation: an individual is a permutation of 26 indices (gene ``d[y]``
is the plaintext letter assigned to ciphertext letter ``y``). Permutation-
preserving DEAP operators:
  * PMX crossover (``cxPartialyMatched``): the segment between two random
    points is exchanged between the parents and conflicts are repaired by
    mapping the exchanged genes;
  * swap mutation (``mutShuffleIndexes`` with probability ``indpb`` per gene,
    i.e. ``26*indpb`` swaps on average);
  * tournament selection; elitism (the ``elite`` best pass unchanged).
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from deap import algorithms, base, creator, tools

from .data import N_LETTERS
from .ngrams import FitnessFunction


@dataclass
class GAConfig:
    """Default configuration: population 400 x 150 generations (budget of about
    60 000 evaluations), tournament size 5, p_c = 0.8, p_m = 0.6, two swaps per
    mutated individual on average, elite of 2. The values were chosen in
    preliminary runs on seeds 1000+, disjoint from the evaluation seeds."""
    population: int = 400
    generations: int = 150
    cxpb: float = 0.8           # crossover probability per pair
    mutpb: float = 0.6          # mutation probability per individual
    indpb: float = 2 / 26       # swap probability per gene
    tournament: int = 5
    elite: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GAResult:
    key: list[int]
    fitness: float
    generations: int
    evaluations: int
    time: float
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (generation, max, avg)
    keys: list[list[int]] = field(default_factory=list)                   # best key in every generation


def setup_creator() -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def evolve(fitness: FitnessFunction, cfg: GAConfig, seed: int | None = None,
           record_keys: bool = False) -> GAResult:
    """Evolutionary loop with elitism: selection -> crossover and mutation
    (varAnd) -> evaluation -> elite + offspring. Returns the best key seen."""
    setup_creator()
    rng = random.Random(seed)
    random.seed(seed)            # DEAP operators use the random module
    start = time.perf_counter()

    tb = base.Toolbox()
    tb.register("individual", lambda: creator.Individual(rng.sample(range(N_LETTERS), N_LETTERS)))
    tb.register("evaluate", lambda ind: (fitness(ind),))
    tb.register("mate", tools.cxPartialyMatched)
    tb.register("mutate", tools.mutShuffleIndexes, indpb=cfg.indpb)
    tb.register("select", tools.selTournament, tournsize=cfg.tournament)

    pop = [tb.individual() for _ in range(cfg.population)]
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    hof = tools.HallOfFame(1)
    hof.update(pop)
    history, keys = [], []
    for generation in range(1, cfg.generations + 1):
        elite = [tb.clone(e) for e in tools.selBest(pop, cfg.elite)]
        offspring = tb.select(pop, len(pop) - cfg.elite)
        offspring = algorithms.varAnd(offspring, tb, cfg.cxpb, cfg.mutpb)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)
        pop = elite + offspring
        hof.update(pop)
        f_max = hof[0].fitness.values[0]
        f_avg = float(np.mean([o.fitness.values[0] for o in pop]))
        history.append((generation, f_max, f_avg))
        if record_keys:
            keys.append(list(hof[0]))
    return GAResult(list(hof[0]), hof[0].fitness.values[0], cfg.generations, fitness.evaluations,
                    time.perf_counter() - start, history, keys)
