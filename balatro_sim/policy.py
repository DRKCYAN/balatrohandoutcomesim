"""Discard policies (heuristics, not optima), held fixed across comparisons.
Contract: discard(hand, discards_left) -> a tuple of 1-5 distinct indices
to throw away, or () to stop. Deterministic and RNG-free (CRN stays exact);
tie-breaks by (rank, then suit index).
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
    so the final hand stays a uniform subset that must match the Phase 1
    exact math. Requires hand_size >= k."""

    name = "blind"

    def __init__(self, k: int = 5):
        if not 1 <= k <= 5:
            raise ValueError("k must be 1-5")
        self.k = k

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        return tuple(range(self.k))


class MadeHand:
    """Keep made value, chase the direct improvement: stop at straight or
    better; else keep the best made group (or 3 highest on high card).
    Deliberately does not chase straights/flushes (the FlushChaser contrast).
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
    """Suit-greedy: chase s*, the most-populated suit (tie: lowest index).
    Stop on a made flush, else discard up to 5 off-suit cards, lowest first.
    """

    name = "flushchaser"

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        n = len(hand)
        suit_counts = [0, 0, 0, 0]
        for c in hand:
            suit_counts[c.suit] += 1
        s_star = max(range(4), key=lambda s: suit_counts[s])
        if suit_counts[s_star] >= 5:
            return ()
        off = [i for i in range(n) if hand[i].suit != s_star]
        off.sort(key=lambda i: (hand[i].rank, hand[i].suit))
        return tuple(sorted(off[:5]))


class RankChaser:
    """Chase one rank-family target: stop at target-or-better, else keep the
    top `groups` rank groups of size >= 2 (1 for pair/trips/quads, 2 for two
    pair/full house) and discard up to 5 others, lowest first.
    """

    def __init__(self, target: HandType, name: str):
        self.target = target
        self.name = name
        self.groups = 2 if target in (HandType.TWO_PAIR, HandType.FULL_HOUSE) else 1

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        n = len(hand)
        if best_from_availability(availability(hand)) >= self.target:
            return ()

        by_rank: dict[int, list[int]] = {}
        for i, c in enumerate(hand):
            by_rank.setdefault(c.rank, []).append(i)
        ranked = sorted(
            (r for r, idx in by_rank.items() if len(idx) >= 2),
            key=lambda r: (len(by_rank[r]), r),
            reverse=True,
        )
        keep = {i for r in ranked[: self.groups] for i in by_rank[r]}
        off = [i for i in range(n) if i not in keep]
        off.sort(key=lambda i: (hand[i].rank, hand[i].suit))
        return tuple(sorted(off[:5]))


class HighCardChaser:
    """Chip-max baseline: always discard the 5 lowest by (rank, suit),
    breaking even made hands. An honest dominated baseline.
    """

    name = "highcard"

    def discard(self, hand: tuple[Card, ...], discards_left: int) -> tuple[int, ...]:
        n = len(hand)
        order = sorted(range(n), key=lambda i: (hand[i].rank, hand[i].suit))
        return tuple(sorted(order[:5]))


_REGISTRY = {
    "none": NoDiscard,
    "blind": BlindDiscard,
    "madehand": MadeHand,
    "flushchaser": FlushChaser,
    "pairchaser": lambda: RankChaser(HandType.PAIR, "pairchaser"),
    "twopairchaser": lambda: RankChaser(HandType.TWO_PAIR, "twopairchaser"),
    "tripschaser": lambda: RankChaser(HandType.THREE_OF_A_KIND, "tripschaser"),
    "fullhousechaser": lambda: RankChaser(HandType.FULL_HOUSE, "fullhousechaser"),
    "quadchaser": lambda: RankChaser(HandType.FOUR_OF_A_KIND, "quadchaser"),
    "highcard": HighCardChaser,
}

POLICY_NAMES = tuple(_REGISTRY)


def get_policy(name: str) -> Policy:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES}") from None
