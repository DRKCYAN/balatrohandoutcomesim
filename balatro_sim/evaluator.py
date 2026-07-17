"""Balatro poker-hand classification. Pure, deterministic, zero game state.

evaluate() classifies 1-5 played cards; best_of() finds the best playable
subset by naive enumeration. availability() is an independent whole-hand
implementation, so best_from_availability() must agree with best_of() on
duplicate-free decks (the per-trial cross-check). Secret hands need
duplicates -- vanilla-unreachable but supported for Phase 4.
"""
from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from itertools import combinations
from typing import Sequence

from .cards import CHIP_VALUE, Card


class HandType(IntEnum):
    # Order = Balatro base-score order; comparisons rely on it.
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10
    FIVE_OF_A_KIND = 11
    FLUSH_HOUSE = 12
    FLUSH_FIVE = 13

    @property
    def display(self) -> str:
        return _DISPLAY[self]


_DISPLAY = {
    HandType.HIGH_CARD: "High Card",
    HandType.PAIR: "Pair",
    HandType.TWO_PAIR: "Two Pair",
    HandType.THREE_OF_A_KIND: "Three of a Kind",
    HandType.STRAIGHT: "Straight",
    HandType.FLUSH: "Flush",
    HandType.FULL_HOUSE: "Full House",
    HandType.FOUR_OF_A_KIND: "Four of a Kind",
    HandType.STRAIGHT_FLUSH: "Straight Flush",
    HandType.ROYAL_FLUSH: "Royal Flush",
    HandType.FIVE_OF_A_KIND: "Five of a Kind",
    HandType.FLUSH_HOUSE: "Flush House",
    HandType.FLUSH_FIVE: "Flush Five",
}

HAND_BASE = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.ROYAL_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}

# The 10 rank windows that form a straight: the wheel, then 2-6 .. 10-A.
WINDOWS: tuple[frozenset[int], ...] = (frozenset({14, 2, 3, 4, 5}),) + tuple(
    frozenset(range(lo, lo + 5)) for lo in range(2, 11)
)
ROYAL_RANKS = frozenset({10, 11, 12, 13, 14})
_WHEEL = WINDOWS[0]


def evaluate(cards: Sequence[Card]) -> HandType:
    """Classify 1-5 played cards (duplicates allowed). Checks run in
    descending base-score order, so the highest-value reading wins."""
    n = len(cards)
    if not 1 <= n <= 5:
        raise ValueError(f"a played hand is 1-5 cards, got {n}")

    rank_counts: dict[int, int] = {}
    for c in cards:
        rank_counts[c[0]] = rank_counts.get(c[0], 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)
    top = counts[0]
    second = counts[1] if len(counts) > 1 else 0

    is_flush = n == 5 and len({c[1] for c in cards}) == 1
    is_straight = False
    if n == 5 and len(rank_counts) == 5:
        rs = sorted(rank_counts)
        is_straight = rs[4] - rs[0] == 4 or frozenset(rs) == _WHEEL

    if top == 5:
        return HandType.FLUSH_FIVE if is_flush else HandType.FIVE_OF_A_KIND
    if is_flush and top == 3 and second == 2:
        return HandType.FLUSH_HOUSE
    if is_flush and is_straight:
        if frozenset(rank_counts) == ROYAL_RANKS:
            return HandType.ROYAL_FLUSH
        return HandType.STRAIGHT_FLUSH
    if top == 4:
        return HandType.FOUR_OF_A_KIND
    if top == 3 and second == 2:
        return HandType.FULL_HOUSE
    if is_flush:
        return HandType.FLUSH
    if is_straight:
        return HandType.STRAIGHT
    if top == 3:
        return HandType.THREE_OF_A_KIND
    if top == 2 and second == 2:
        return HandType.TWO_PAIR
    if top == 2:
        return HandType.PAIR
    return HandType.HIGH_CARD


@lru_cache(maxsize=None)
def _subset_indices(n: int) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for k in range(1, min(5, n) + 1):
        out.extend(combinations(range(n), k))
    return tuple(out)


def best_of(cards: Sequence[Card]) -> tuple[HandType, tuple[Card, ...]]:
    """Best playable hand among all 1-5 card subsets. Tiebreak within a type
    is total chip value, then first subset in enumeration order."""
    best_t: HandType | None = None
    best_chips = -1
    best_cards: tuple[Card, ...] = ()
    for idx in _subset_indices(len(cards)):
        subset = tuple(cards[i] for i in idx)
        t = evaluate(subset)
        if best_t is not None and t < best_t:
            continue
        chips = sum(CHIP_VALUE[c[0]] for c in subset)
        if best_t is None or t > best_t or chips > best_chips:
            best_t, best_chips, best_cards = t, chips, subset
    assert best_t is not None
    return best_t, best_cards


def availability(cards: Sequence[Card]) -> dict[str, bool]:
    """Which hand classes exist anywhere in `cards`, by single-pass counting
    independent of evaluate()/best_of() (the cross-check). Flags describe
    duplicate-free decks."""
    rank_counts: dict[int, int] = {}
    suit_ranks: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    suit_counts = [0, 0, 0, 0]
    for c in cards:
        rank_counts[c[0]] = rank_counts.get(c[0], 0) + 1
        suit_counts[c[1]] += 1
        suit_ranks[c[1]].add(c[0])
    ranks = set(rank_counts)
    n_pairs = sum(1 for v in rank_counts.values() if v >= 2)
    has_trips = any(v >= 3 for v in rank_counts.values())
    return {
        "pair": n_pairs >= 1,
        "two_pair": n_pairs >= 2,
        "three_of_a_kind": has_trips,
        "straight": any(w <= ranks for w in WINDOWS),
        "flush": any(s >= 5 for s in suit_counts),
        "full_house": has_trips and n_pairs >= 2,
        "four_of_a_kind": any(v >= 4 for v in rank_counts.values()),
        "straight_flush": any(w <= sr for sr in suit_ranks.values() for w in WINDOWS),
        "royal_flush": any(ROYAL_RANKS <= sr for sr in suit_ranks.values()),
    }


_FLOORS = (
    ("royal_flush", HandType.ROYAL_FLUSH),
    ("straight_flush", HandType.STRAIGHT_FLUSH),
    ("four_of_a_kind", HandType.FOUR_OF_A_KIND),
    ("full_house", HandType.FULL_HOUSE),
    ("flush", HandType.FLUSH),
    ("straight", HandType.STRAIGHT),
    ("three_of_a_kind", HandType.THREE_OF_A_KIND),
    ("two_pair", HandType.TWO_PAIR),
    ("pair", HandType.PAIR),
)


def best_from_availability(avail: dict[str, bool]) -> HandType:
    """Best hand type implied by availability flags. For duplicate-free
    decks this equals best_of()'s type exactly; divergence is a bug."""
    for key, t in _FLOORS:
        if avail[key]:
            return t
    return HandType.HIGH_CARD
