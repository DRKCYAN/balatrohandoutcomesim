"""Core scoring: chips x mult with hand levels. No enhancements (Phase 4),
no jokers (Phase 5), pure integer arithmetic.

    S = (base_chips(type, level) + sum of scoring-card chip values)
        * base_mult(type, level)

Scoring cards: only the cards forming the hand contribute their chip
values (kickers never score; the Splash joker that changes this is Phase
5 material): High Card scores its single highest card; pair-family hands
score the matched-rank cards; every 5-card hand (straight, flush, full
house, straight flush, flush house, flush five) scores all five.

Levels: each hand type has a level (default 1). A level above 1 adds the
hand's planet-card increment per level. Royal Flush shares Straight
Flush's level and increment (one planet, Neptune, levels both).

LEVEL_INCREMENTS source: the Balatro wiki planet-card table
(balatrogame.fandom.com/wiki/Planet_Cards, balatrowiki.org/w/Planet_Cards,
retrieved 2026-07-13), cross-checked against each other. Every value is
additionally listed as "pending in-game confirmation" in
docs/VALIDATION.md -- constants transcribed from wikis are exactly the
kind of input PLAN.md section 9 exists to catch.

best_play() is the optimal greedy player: the highest-SCORING subset,
which is not the highest-TYPE subset -- a junk full house (40+12)x4=208
loses to an ace-high flush (35+50)x4=340 even at level 1. The type-max
best_of() in evaluator.py keeps its capability semantics for the
Phase 1/2 distribution reports; the two are related by tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .cards import CHIP_VALUE, Card
from .evaluator import HAND_BASE, HandType, evaluate, _subset_indices

Levels = Mapping[HandType, int]

# (+chips, +mult) per level above 1 -- the planet cards.
LEVEL_INCREMENTS: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (10, 1),        # Pluto
    HandType.PAIR: (15, 1),             # Mercury
    HandType.TWO_PAIR: (20, 1),         # Uranus
    HandType.THREE_OF_A_KIND: (20, 2),  # Venus
    HandType.STRAIGHT: (30, 3),         # Saturn
    HandType.FLUSH: (15, 2),            # Jupiter
    HandType.FULL_HOUSE: (25, 2),       # Earth
    HandType.FOUR_OF_A_KIND: (30, 3),   # Mars
    HandType.STRAIGHT_FLUSH: (40, 4),   # Neptune
    HandType.ROYAL_FLUSH: (40, 4),      # shares Neptune
    HandType.FIVE_OF_A_KIND: (35, 3),   # Planet X
    HandType.FLUSH_HOUSE: (40, 4),      # Ceres
    HandType.FLUSH_FIVE: (50, 3),       # Eris
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
        # ranks are necessarily distinct (a duplicate would be a pair+)
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
    # every remaining type is a 5-card hand: all played cards score
    return tuple(played)


def score(played: Sequence[Card], levels: Optional[Levels] = None) -> int:
    """Score of playing exactly these 1-5 cards."""
    t = evaluate(played)
    chips, mult = hand_base_at(t, effective_level(levels, t))
    chips += sum(CHIP_VALUE[c.rank] for c in scoring_cards(t, played))
    return chips * mult


@dataclass(frozen=True)
class PlayResult:
    """Outcome of one played hand. score is None when scoring was not
    requested (levels absent) -- type-only statistics still work."""

    hand_type: HandType
    played: tuple[Card, ...]
    score: Optional[int]


def best_play(cards: Sequence[Card], levels: Optional[Levels] = None) -> PlayResult:
    """The highest-scoring play among all 1-5 card subsets (the optimal
    greedy player). Naive enumeration, per the best_of contract: correct
    and fast enough, do not optimize. Ties: higher hand type, then first
    subset in enumeration order.
    """
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
