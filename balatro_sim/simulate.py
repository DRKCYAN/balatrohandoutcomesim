"""Trial loop: shuffle, deal 8, discard/redraw under a policy, best hand type.

Seeding contract (the common-random-numbers hook): trial i under base
seed s uses random.Random(f"{s}:{i}"). String seeds hash through SHA-512,
so the shuffle stream is stable across platforms, processes and Python
builds -- required so that paired comparisons can replay identical
shuffles through both arms (CRN) and so results are reproducible. Seed
per trial, never per run. All trial randomness is the shuffle; policies
are deterministic, so a trial is a pure function of (deck, seed, i,
policy, discards).

Discard mechanics (play_out): the hand is the top 8 of the shuffle; each
discard replaces the chosen cards, in ascending hand-index order, with
the next cards off the top of the remaining deck. Policies must not
depend on hand order, but the rule is fixed so trials are reproducible.

Every trial runs the availability cross-check on the FINAL hand:
best_of() (subset enumeration) and best_from_availability() (whole-hand
counting) are independent computations of the same quantity and must
agree exactly on duplicate-free decks. Mismatches are counted, not
raised, so a full run always reports.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Optional, Sequence

from .cards import Card
from .evaluator import HandType, availability, best_from_availability, best_of
from .policy import NoDiscard, Policy


def trial_rng(seed: int, i: int) -> random.Random:
    """The one place trial randomness comes from."""
    return random.Random(f"{seed}:{i}")


def deal(deck: Sequence[Card], rng: random.Random, k: int = 8) -> list[Card]:
    """Shuffle a copy of the deck, deal the top k."""
    cards = list(deck)
    rng.shuffle(cards)
    return cards[:k]


@dataclass(frozen=True)
class DiscardStep:
    """One discard round, as recorded by play_out(trace=...)."""

    hand_before: tuple[Card, ...]
    discarded_indices: tuple[int, ...]  # ascending
    drawn: tuple[Card, ...]  # replacement cards, in the order they landed


def _validate_discard(idx: tuple[int, ...], hand_len: int) -> None:
    if not 1 <= len(idx) <= 5:
        raise ValueError(f"a discard is 1-5 cards, policy returned {len(idx)}")
    if len(set(idx)) != len(idx):
        raise ValueError(f"policy returned duplicate indices {idx}")
    if any(not 0 <= i < hand_len for i in idx):
        raise ValueError(f"policy returned out-of-range indices {idx}")


def play_out(
    shuffled: Sequence[Card],
    policy: Policy,
    discards: int,
    hand_size: int = 8,
    trace: Optional[list] = None,
) -> tuple[Card, ...]:
    """Deal the top hand_size of an already-shuffled deck, then let the
    policy spend up to `discards` discards. Returns the final hand.

    Does not mutate `shuffled`, so paired arms can share one shuffle.

    If `trace` is a list, one DiscardStep per round is appended to it --
    the replay hook (trace.py). Same loop either way, so a traced replay
    cannot diverge from what the hot path did; the hot path (trace=None)
    allocates nothing extra.
    """
    hand = list(shuffled[:hand_size])
    draw = hand_size  # index of the next card off the top
    left = discards
    while left > 0:
        idx = tuple(policy.discard(tuple(hand), left))
        if not idx:
            break
        _validate_discard(idx, len(hand))
        if draw + len(idx) > len(shuffled):
            raise RuntimeError("deck exhausted during redraw")
        if trace is None:
            for j in sorted(idx):
                hand[j] = shuffled[draw]
                draw += 1
        else:
            before = tuple(hand)
            drawn = []
            for j in sorted(idx):
                hand[j] = shuffled[draw]
                drawn.append(shuffled[draw])
                draw += 1
            trace.append(DiscardStep(before, tuple(sorted(idx)), tuple(drawn)))
        left -= 1
    return tuple(hand)


@dataclass
class DistributionReport:
    n: int
    seed: int
    policy_name: str
    discards: int
    best_counts: Counter  # HandType -> trials where it was the best playable
    avail_counts: Counter  # availability flag -> trials where it was set
    inconsistencies: int  # trials where best_of() != availability floor

    def p_best(self, t: HandType) -> float:
        return self.best_counts.get(t, 0) / self.n

    def p_avail(self, key: str) -> float:
        return self.avail_counts.get(key, 0) / self.n


# Back-compat alias: Phase 1 reports are the zero-discard special case.
Phase1Report = DistributionReport


def se(p: float, n: int) -> float:
    """Standard error of a Bernoulli(p) mean over n trials."""
    return sqrt(p * (1 - p) / n)


def run_distribution(
    deck: Sequence[Card],
    n: int,
    seed: int = 0,
    policy: Optional[Policy] = None,
    discards: int = 0,
    progress: Optional[Callable[[int], None]] = None,
) -> DistributionReport:
    """n independent trials: shuffle, deal 8, play out the policy's
    discards, record the best playable hand type of the final hand."""
    if n < 1:
        raise ValueError("need at least one trial")
    if discards < 0:
        raise ValueError("discards must be >= 0")
    if policy is None:
        policy = NoDiscard()
    best_counts: Counter = Counter()
    avail_counts: Counter = Counter()
    inconsistencies = 0
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        final = play_out(shuffled, policy, discards)
        t, _ = best_of(final)
        av = availability(final)
        best_counts[t] += 1
        for key, flag in av.items():
            if flag:
                avail_counts[key] += 1
        if best_from_availability(av) is not t:
            inconsistencies += 1
        if progress is not None and (i + 1) % 10_000 == 0:
            progress(i + 1)
    return DistributionReport(
        n, seed, policy.name, discards, best_counts, avail_counts, inconsistencies
    )


def run_phase1(
    deck: Sequence[Card],
    n: int,
    seed: int = 0,
    progress: Optional[Callable[[int], None]] = None,
) -> DistributionReport:
    """Phase 1 loop: the zero-discard special case, kept as a stable entry
    point (its results are pinned by tests and by recorded runs)."""
    return run_distribution(deck, n, seed, policy=NoDiscard(), discards=0, progress=progress)
