"""Hand-derived conditional ground truth for single-discard states:
conditional on a fully specified state (8 seen, 44 unseen), the redraw is
plain hypergeometric sampling. Same discipline as exact.py.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb


def four_flush_completion() -> Fraction:
    """FlushChaser, one discard, initial hand holding exactly 4 of s*.
    Discards the 4 off-suit, draws 4; a flush completes iff any draw is
    one of the 9 unseen s* cards:

        P = 1 - C(35,4)/C(44,4)  ~= 0.61425
    """
    return 1 - Fraction(comb(35, 4), comb(44, 4))


def pair_to_trips_draw() -> Fraction:
    """MadeHand, one discard, initial best exactly a pair. Keeps the pair
    plus one kicker and draws 5; with 2 live outs among the 44 unseen it
    reaches trips iff any out arrives:

        P = 1 - C(42,5)/C(44,5)  ~= 0.21665
    """
    return 1 - Fraction(comb(42, 5), comb(44, 5))
