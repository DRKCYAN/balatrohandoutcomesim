"""Core scoring: S = (base_chips(type, level) + scoring-card chips)
* base_mult(type, level). Only the cards forming the hand score (kickers
never do). Levels above 1 add the planet-card increment (LEVEL_INCREMENTS,
from the Balatro wiki -- see docs/VALIDATION.md). best_play() is the
optimal greedy player: the highest-SCORING subset, not the highest-TYPE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .cards import CHIP_VALUE, Card
from .evaluator import HAND_BASE, HandType, evaluate, _subset_indices

Levels = Mapping[HandType, int]

# (+chips, +mult) per level above 1 -- the planet cards.
LEVEL_INCREMENTS: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_OF_A_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.ROYAL_FLUSH: (40, 4),      # shares Neptune
    HandType.FIVE_OF_A_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}


def effective_level(levels: Optional[Levels], t: HandType) -> int:
    """The level that applies to hand type t (Royal reads Straight Flush's)."""
    if levels is None:
        return 1
    key = HandType.STRAIGHT_FLUSH if t is HandType.ROYAL_FLUSH else t
    level = levels.get(key, 1)
    if level < 1:
        raise ValueError(f"level for {key.name} must be >= 1, got {level}")
    return level


def hand_base_at(t: HandType, level: int) -> tuple[int, int]:
    """(chips, mult) for hand type t at the given level. Level 1 == HAND_BASE."""
    base_chips, base_mult = HAND_BASE[t]
    inc_chips, inc_mult = LEVEL_INCREMENTS[t]
    return base_chips + inc_chips * (level - 1), base_mult + inc_mult * (level - 1)


def scoring_cards(hand_type: HandType, played: Sequence[Card]) -> tuple[Card, ...]:
    """The played cards that contribute chip values."""
    if hand_type is HandType.HIGH_CARD:
        return (max(played, key=lambda c: c.rank),)
    rank_counts: dict[int, int] = {}
    for c in played:
        rank_counts[c[0]] = rank_counts.get(c[0], 0) + 1
    if hand_type in (
        HandType.PAIR,
        HandType.THREE_OF_A_KIND,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
    ):
        need = {
            HandType.PAIR: 2,
            HandType.THREE_OF_A_KIND: 3,
            HandType.FOUR_OF_A_KIND: 4,
            HandType.FIVE_OF_A_KIND: 5,
        }[hand_type]
        (r,) = [r for r, n in rank_counts.items() if n == need]
        return tuple(c for c in played if c.rank == r)
    if hand_type is HandType.TWO_PAIR:
        pair_ranks = {r for r, n in rank_counts.items() if n == 2}
        return tuple(c for c in played if c.rank in pair_ranks)
    return tuple(played)


def score(played: Sequence[Card], levels: Optional[Levels] = None) -> int:
    """Score of playing exactly these 1-5 cards."""
    t = evaluate(played)
    chips, mult = hand_base_at(t, effective_level(levels, t))
    chips += sum(CHIP_VALUE[c.rank] for c in scoring_cards(t, played))
    return chips * mult


@dataclass(frozen=True)
class PlayResult:
    """Outcome of one played hand. score is None when levels were absent."""

    hand_type: HandType
    played: tuple[Card, ...]
    score: Optional[int]


def best_play(cards: Sequence[Card], levels: Optional[Levels] = None) -> PlayResult:
    """The highest-scoring play among all 1-5 card subsets (the optimal
    greedy player). Ties: higher hand type, then enumeration order."""
    best_score = -1
    best_t = HandType.HIGH_CARD
    best_cards: tuple[Card, ...] = ()
    for idx in _subset_indices(len(cards)):
        subset = tuple(cards[i] for i in idx)
        t = evaluate(subset)
        chips, mult = hand_base_at(t, effective_level(levels, t))
        chips += sum(CHIP_VALUE[c.rank] for c in scoring_cards(t, subset))
        s = chips * mult
        if s > best_score or (s == best_score and t > best_t):
            best_score, best_t, best_cards = s, t, subset
    return PlayResult(best_t, best_cards, best_score)
