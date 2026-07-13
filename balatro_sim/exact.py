"""Hand-derived exact probabilities for the zero-discard, 8-card deal.

Phase 1's exit criterion: the Monte Carlo distribution must match math
derivable by hand. This module is deliberately self-contained -- it
imports nothing from the simulator (rank windows, counting, everything
is re-derived here), so agreement between the two is evidence, not
tautology. All probabilities are exact `fractions.Fraction`s.

Sample space: all C(52,8) = 752,538,150 eight-card hands, equally likely.

Three independent techniques, each documented at its function:

  1. Rank-multiset enumeration: partition the 8 cards by rank
     multiplicity, weight each partition by (ways to assign ranks) x
     (ways to choose suits within each rank). Gives every rank-only
     event: pair, two pair, trips, quads, full house.
  2. Closed forms: direct hypergeometric/inclusion-exclusion formulas
     for flush, quads and pair -- the latter two double as independent
     cross-checks of technique 1.
  3. Window inclusion-exclusion for straights and straight flushes.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb, factorial, perm

TOTAL_DEALS = comb(52, 8)

_RANKS = tuple(range(2, 15))  # A = 14, and also plays low in the wheel
_WINDOWS = (frozenset({14, 2, 3, 4, 5}),) + tuple(
    frozenset(range(lo, lo + 5)) for lo in range(2, 11)
)


# ---------------------------------------------------------------- technique 1

def _partitions(total: int = 8, max_part: int = 4, max_parts: int = 13):
    """Non-increasing tuples of per-rank multiplicities summing to `total`.

    max_part=4: at most four copies of a rank in a vanilla deck.
    """
    def rec(remaining: int, cap: int, prefix: tuple[int, ...]):
        if remaining == 0:
            yield prefix
            return
        if len(prefix) == max_parts:
            return
        for part in range(min(cap, remaining), 0, -1):
            yield from rec(remaining - part, part, prefix + (part,))

    yield from rec(total, max_part, ())


def _weight(parts: tuple[int, ...]) -> int:
    """Number of 8-card hands whose rank-multiplicity multiset is `parts`.

    Assign distinct ranks to the k parts: 13!/(13-k)! orderings, divided
    by m_c! for each group of equal parts (they are interchangeable).
    Then choose which suits realise each part: prod C(4, c).
    """
    mult: dict[int, int] = {}
    for p in parts:
        mult[p] = mult.get(p, 0) + 1
    rank_ways = perm(13, len(parts))
    for m in mult.values():
        rank_ways //= factorial(m)
    suit_ways = 1
    for p in parts:
        suit_ways *= comb(4, p)
    return rank_ways * suit_ways


def _rank_event(pred) -> Fraction:
    return Fraction(sum(_weight(p) for p in _partitions() if pred(p)), TOTAL_DEALS)


def sanity_total() -> int:
    """Partition weights must tile the whole sample space: == C(52,8)."""
    return sum(_weight(p) for p in _partitions())


def pair_or_better() -> Fraction:
    return _rank_event(lambda p: p[0] >= 2)


def two_pair_available() -> Fraction:
    """Two distinct ranks each holding >= 2 cards (four of one rank alone
    cannot be played as two pair: any 4-card subset of it is quads)."""
    return _rank_event(lambda p: sum(1 for x in p if x >= 2) >= 2)


def trips_available() -> Fraction:
    return _rank_event(lambda p: p[0] >= 3)


def quads_available() -> Fraction:
    return _rank_event(lambda p: p[0] >= 4)


def full_house_available() -> Fraction:
    """A rank with >= 3 plus a *different* rank with >= 2."""
    return _rank_event(lambda p: p[0] >= 3 and sum(1 for x in p if x >= 2) >= 2)


# ---------------------------------------------------------------- technique 2

def pair_or_better_closed_form() -> Fraction:
    """1 - P(all 8 ranks distinct) = 1 - C(13,8) * 4^8 / C(52,8)."""
    return 1 - Fraction(comb(13, 8) * 4**8, TOTAL_DEALS)


def quads_closed_form() -> Fraction:
    """Inclusion-exclusion over ranks: 13*C(48,4) counts each hand once
    per completed rank; C(13,2) hands complete two ranks (4+4 = 8)."""
    return Fraction(13 * comb(48, 4) - comb(13, 2), TOTAL_DEALS)


def flush_available() -> Fraction:
    """Some suit holds >= 5 of the 8. Two suits can't both do it (5+5 > 8),
    so no inclusion-exclusion: 4 * sum_k C(13,k) C(39,8-k) / C(52,8)."""
    return Fraction(
        4 * sum(comb(13, k) * comb(39, 8 - k) for k in range(5, 9)), TOTAL_DEALS
    )


def royal_flush_available() -> Fraction:
    """All five cards of one suit's 10-J-Q-K-A present: 4 * C(47,3).
    Two suits at once would need 10 cards."""
    return Fraction(4 * comb(47, 3), TOTAL_DEALS)


# ---------------------------------------------------------------- technique 3

@lru_cache(maxsize=None)
def _all_ranks_present(u: int) -> Fraction:
    """P(u specified ranks all appear among the 8 cards), by
    inclusion-exclusion over which of the u ranks are absent."""
    return Fraction(
        sum((-1) ** j * comb(u, j) * comb(52 - 4 * j, 8) for j in range(u + 1)),
        TOTAL_DEALS,
    )


def straight_available() -> Fraction:
    """P(the rank set covers some straight window), inclusion-exclusion
    over the 10 windows; each intersection term is 'all ranks of the
    union present', which depends only on the union's size."""
    total = Fraction(0)
    for r in range(1, 11):
        for sub in combinations(_WINDOWS, r):
            u = len(frozenset().union(*sub))
            total += (-1) ** (r + 1) * _all_ranks_present(u)
    return total


def straight_flush_available() -> Fraction:
    """Inclusion-exclusion over (suit, window) events 'these 5 exact cards
    are in the hand'. Cross-suit intersections need >= 10 cards of 8, so
    they vanish and the sum is 4x the one-suit case. A union of same-suit
    windows with u <= 8 specific cards costs C(52-u, 8-u)."""
    per_suit = 0
    for r in range(1, 11):
        for sub in combinations(_WINDOWS, r):
            u = len(frozenset().union(*sub))
            if u <= 8:
                per_suit += (-1) ** (r + 1) * comb(52 - u, 8 - u)
    return Fraction(4 * per_suit, TOTAL_DEALS)


# ------------------------------------------------------------- best-hand rows

def best_is_high_card() -> Fraction:
    """P(best playable hand is exactly High Card): no pair (8 distinct
    ranks), no straight (rank set misses every window), no flush (every
    suit <= 4).

    For a fixed set of 8 distinct ranks the 4^8 suit assignments are
    uniform and independent of which set it is, so the count factorises:
    (rank sets missing every window) x (assignments with no suit >= 5).
    Only one suit can reach 5 of 8, hence the un-inclusion-excluded 4x.
    """
    no_straight_sets = sum(
        1
        for s in combinations(_RANKS, 8)
        if not any(w <= frozenset(s) for w in _WINDOWS)
    )
    no_flush_assignments = 4**8 - 4 * sum(
        comb(8, k) * 3 ** (8 - k) for k in range(5, 9)
    )
    return Fraction(no_straight_sets * no_flush_assignments, TOTAL_DEALS)


def availability_exact() -> dict[str, Fraction]:
    """Exact counterparts of evaluator.availability(), same keys."""
    return {
        "pair": pair_or_better(),
        "two_pair": two_pair_available(),
        "three_of_a_kind": trips_available(),
        "straight": straight_available(),
        "flush": flush_available(),
        "full_house": full_house_available(),
        "four_of_a_kind": quads_available(),
        "straight_flush": straight_flush_available(),
        "royal_flush": royal_flush_available(),
    }
