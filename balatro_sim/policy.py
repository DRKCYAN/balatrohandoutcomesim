"""Discard policies -- PLAN.md's pi, "where intellectual honesty lives".

The simulator measures build and policy jointly, so every policy here is
stated explicitly, held fixed across comparisons, and deliberately simple
enough to reason about. These are heuristics, not optima.

Contract (PLAN.md section 7, extended minimally with discards_left):

    policy.discard(hand, discards_left) -> tuple of indices into hand

  - 1-5 distinct indices to throw away, or () to stop discarding.
  - Called only while discards remain; () ends the loop for the trial.
  - Deterministic and RNG-free. With identical shuffles this keeps
    common-random-number pairing exact: two arms stay in lockstep until
    a policy actually decides differently.
  - The engine (simulate.play_out) validates returns and raises on
    violations rather than silently clamping.

All tie-breaks are by (rank, then suit index) so behaviour is fully
deterministic and pinned by tests.
"""
from __future__ import annotations

from typing import Protocol

from .cards import Card
from .evaluator import HandType, availability, best_from_availability


class Policy(Protocol):
    name: str

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        """Indices to throw away (1-5, distinct), or () to stop."""
        ...


class NoDiscard:
    """Baseline: never discards. Must reproduce Phase 1 trial-for-trial."""

    name = "none"

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        return ()


class BlindDiscard:
    """Validation-only: discards the first k positions regardless of content,
    every time a discard remains.

    Because the decision is content-blind, the final 8 cards are still 8
    fixed positions of a uniformly shuffled deck, i.e. a uniform 8-card
    subset of the 52. Its distribution must therefore match the Phase 1
    exact math -- an end-to-end test of the replace/draw mechanics.
    """

    name = "blind"

    def __init__(self, k: int = 5):
        if not 1 <= k <= 5:
            raise ValueError("k must be 1-5")
        self.k = k

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        return tuple(range(self.k))


class MadeHand:
    """Keep made value, chase the direct improvement. Rules in order:

      a. best available type >= STRAIGHT (straight/flush/full house/quads/
         straight flush): stop.
      b. trips available: keep the highest trips, discard the other 5.
      c. two pair: keep the two highest pairs, discard the other 4
         (three pairs: lowest pair goes).
      d. pair: keep the pair plus the highest kicker, discard the other 5.
      e. high card: keep the 3 highest ranks, discard the lowest 5.

    Deliberately does not chase straights or flushes -- that contrast with
    FlushChaser is the point (PLAN.md section 7 robustness).
    """

    name = "madehand"

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        n = len(hand)
        best = best_from_availability(availability(hand))
        if best >= HandType.STRAIGHT:
            return ()

        by_rank: dict[int, list[int]] = {}
        for i, c in enumerate(hand):
            by_rank.setdefault(c.rank, []).append(i)

        if best is HandType.THREE_OF_A_KIND:
            r = max(r for r, idx in by_rank.items() if len(idx) >= 3)
            keep = set(by_rank[r])
        elif best is HandType.TWO_PAIR:
            pair_ranks = sorted(
                (r for r, idx in by_rank.items() if len(idx) >= 2), reverse=True
            )
            keep = {i for r in pair_ranks[:2] for i in by_rank[r]}
        elif best is HandType.PAIR:
            (r,) = [r for r, idx in by_rank.items() if len(idx) >= 2]
            keep = set(by_rank[r])
            others = [i for i in range(n) if i not in keep]
            keep.add(max(others, key=lambda i: (hand[i].rank, hand[i].suit)))
        else:  # high card: keep the n-5 highest
            order = sorted(range(n), key=lambda i: (hand[i].rank, hand[i].suit))
            return tuple(sorted(order[:5]))

        out = tuple(sorted(i for i in range(n) if i not in keep))
        return out[:5] if len(out) > 5 else out


class FlushChaser:
    """Suit-greedy. Let s* be the most-populated suit (tie: lowest suit
    index, i.e. S > H > D > C priority):

      a. >= 5 of s* (flush made): stop.
      b. else discard up to 5 off-suit cards, lowest ranks first.

    Sacrifices made pairs for flush equity by construction.
    """

    name = "flushchaser"

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        n = len(hand)
        suit_counts = [0, 0, 0, 0]
        for c in hand:
            suit_counts[c.suit] += 1
        s_star = max(range(4), key=lambda s: suit_counts[s])  # first max wins ties
        if suit_counts[s_star] >= 5:
            return ()
        off = [i for i in range(n) if hand[i].suit != s_star]
        off.sort(key=lambda i: (hand[i].rank, hand[i].suit))
        return tuple(sorted(off[:5]))


_REGISTRY = {
    "none": NoDiscard,
    "blind": BlindDiscard,
    "madehand": MadeHand,
    "flushchaser": FlushChaser,
}

POLICY_NAMES = tuple(_REGISTRY)


def get_policy(name: str) -> Policy:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES}") from None
