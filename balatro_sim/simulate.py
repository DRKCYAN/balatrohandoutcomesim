"""Phase 1 trial loop: deal 8 from a shuffled deck, record best hand type.

Seeding contract (the common-random-numbers hook): trial i under base
seed s uses random.Random(f"{s}:{i}"). String seeds hash through SHA-512,
so the shuffle stream is stable across platforms, processes and Python
builds -- required so that later phases can replay identical shuffles
through both arms of a paired comparison (CRN) and so results are
reproducible. Seed per trial, never per run.

Every trial also runs the availability cross-check: best_of() (subset
enumeration) and best_from_availability() (whole-hand counting) are
independent computations of the same quantity and must agree exactly on
duplicate-free decks. Mismatches are counted, not raised, so a full run
always reports.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Optional, Sequence

from .cards import Card
from .evaluator import HandType, availability, best_from_availability, best_of


def trial_rng(seed: int, i: int) -> random.Random:
    """The one place trial randomness comes from."""
    return random.Random(f"{seed}:{i}")


def deal(deck: Sequence[Card], rng: random.Random, k: int = 8) -> list[Card]:
    """Shuffle a copy of the deck, deal the top k."""
    cards = list(deck)
    rng.shuffle(cards)
    return cards[:k]


@dataclass
class Phase1Report:
    n: int
    seed: int
    best_counts: Counter  # HandType -> trials where it was the best playable
    avail_counts: Counter  # availability flag -> trials where it was set
    inconsistencies: int  # trials where best_of() != availability floor

    def p_best(self, t: HandType) -> float:
        return self.best_counts.get(t, 0) / self.n

    def p_avail(self, key: str) -> float:
        return self.avail_counts.get(key, 0) / self.n


def se(p: float, n: int) -> float:
    """Standard error of a Bernoulli(p) mean over n trials."""
    return sqrt(p * (1 - p) / n)


def run_phase1(
    deck: Sequence[Card],
    n: int,
    seed: int = 0,
    progress: Optional[Callable[[int], None]] = None,
) -> Phase1Report:
    """n independent trials: shuffle, deal 8, best playable hand type."""
    if n < 1:
        raise ValueError("need at least one trial")
    best_counts: Counter = Counter()
    avail_counts: Counter = Counter()
    inconsistencies = 0
    for i in range(n):
        hand8 = deal(deck, trial_rng(seed, i))
        t, _ = best_of(hand8)
        av = availability(hand8)
        best_counts[t] += 1
        for key, flag in av.items():
            if flag:
                avail_counts[key] += 1
        if best_from_availability(av) is not t:
            inconsistencies += 1
        if progress is not None and (i + 1) % 10_000 == 0:
            progress(i + 1)
    return Phase1Report(n, seed, best_counts, avail_counts, inconsistencies)
