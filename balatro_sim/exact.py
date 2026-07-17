"""Hand-derived exact probabilities for the zero-discard, 8-card deal --
Phase 1's exit criterion. Self-contained (imports nothing from the
simulator, so agreement is evidence, not tautology); exact Fractions over
C(52,8) hands via three techniques documented at their functions.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb, factorial, perm

TOTAL_DEALS = comb(52, 8)

_RANKS = tuple(range(2, 15))  # A = 14, also plays low in the wheel
_WINDOWS = (frozenset({14, 2, 3, 4, 5}),) + tuple(
    frozenset(range(lo, lo + 5)) for lo in range(2, 11)
)


def _partitions(total: int = 8, max_part: int = 4, max_parts: int = 13):
    """Non-increasing tuples of per-rank multiplicities summing to `total`
    (max_part=4: at most four copies of a rank in a vanilla deck)."""
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
    """Number of 8-card hands whose rank-multiplicity multiset is `parts`:
    (ways to assign distinct ranks to the parts, perm(13,k) divided by
    m!'s for equal parts) x (suit choices prod C(4, c))."""
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
    """Two distinct ranks each holding >= 2 cards."""
    return _rank_event(lambda p: sum(1 for x in p if x >= 2) >= 2)


def trips_available() -> Fraction:
    return _rank_event(lambda p: p[0] >= 3)


def quads_available() -> Fraction:
    return _rank_event(lambda p: p[0] >= 4)


def full_house_available() -> Fraction:
    """A rank with >= 3 plus a *different* rank with >= 2."""
    return _rank_event(lambda p: p[0] >= 3 and sum(1 for x in p if x >= 2) >= 2)


def pair_or_better_closed_form() -> Fraction:
    """1 - P(all 8 ranks distinct) = 1 - C(13,8) * 4^8 / C(52,8)."""
    return 1 - Fraction(comb(13, 8) * 4**8, TOTAL_DEALS)


def quads_closed_form() -> Fraction:
    """Inclusion-exclusion over ranks: 13*C(48,4) minus C(13,2) (hands
    completing two ranks, 4+4=8)."""
    return Fraction(13 * comb(48, 4) - comb(13, 2), TOTAL_DEALS)


def flush_available() -> Fraction:
    """Some suit holds >= 5 of the 8 (two can't both, 5+5 > 8):
    4 * sum_k C(13,k) C(39,8-k) / C(52,8)."""
    return Fraction(
        4 * sum(comb(13, k) * comb(39, 8 - k) for k in range(5, 9)), TOTAL_DEALS
    )


def royal_flush_available() -> Fraction:
    """All five of one suit's 10-J-Q-K-A present: 4 * C(47,3)."""
    return Fraction(4 * comb(47, 3), TOTAL_DEALS)


@lru_cache(maxsize=None)
def _all_ranks_present(u: int) -> Fraction:
    """P(u specified ranks all appear among the 8 cards), by
    inclusion-exclusion over which are absent."""
    return Fraction(
        sum((-1) ** j * comb(u, j) * comb(52 - 4 * j, 8) for j in range(u + 1)),
        TOTAL_DEALS,
    )


def straight_available() -> Fraction:
    """P(the rank set covers some straight window), inclusion-exclusion
    over the 10 windows (each term depends only on the union's size)."""
    total = Fraction(0)
    for r in range(1, 11):
        for sub in combinations(_WINDOWS, r):
            u = len(frozenset().union(*sub))
            total += (-1) ** (r + 1) * _all_ranks_present(u)
    return total


def straight_flush_available() -> Fraction:
    """Inclusion-exclusion over (suit, window) events. Cross-suit terms
    need >= 10 cards so vanish (sum is 4x the one-suit case); a same-suit
    window union of u <= 8 cards costs C(52-u, 8-u)."""
    per_suit = 0
    for r in range(1, 11):
        for sub in combinations(_WINDOWS, r):
            u = len(frozenset().union(*sub))
            if u <= 8:
                per_suit += (-1) ** (r + 1) * comb(52 - u, 8 - u)
    return Fraction(4 * per_suit, TOTAL_DEALS)


def best_is_high_card() -> Fraction:
    """P(best playable hand is exactly High Card): no pair/straight/flush.
    Factorises over 8 distinct ranks: (rank sets missing every window) x
    (suit assignments with no suit >= 5)."""
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
