"""Hand-derived conditional ground truth for single-discard states.

There is no global closed form once a policy reacts to what it sees, but
*conditional on a fully specified state* the redraw is still plain
hypergeometric sampling from the unseen cards. These two states are
common enough to accumulate thousands of conditioned Monte Carlo trials,
and each pins down the draw mechanics plus one policy branch.

Same discipline as exact.py: zero imports from the simulator, exact
Fractions, derivations in the docstrings.

Conditioning is on the state at the moment of the (single) discard:
8 cards seen, 44 unseen, and the unseen suit/rank composition follows
from the hand alone.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb


def four_flush_completion() -> Fraction:
    """FlushChaser, one discard, initial hand holding exactly 4 of the
    chase suit s*.

    The policy discards the 4 off-suit cards and draws 4. Unseen: 44
    cards, of which 13 - 4 = 9 are s*. The final hand keeps no off-suit
    cards, so no other suit can reach 5; a flush completes iff at least
    one of the 4 draws is s*:

        P = 1 - C(35,4)/C(44,4)  ~= 0.61425
    """
    return 1 - Fraction(comb(35, 4), comb(44, 4))


def pair_to_trips_draw() -> Fraction:
    """MadeHand, one discard, initial best exactly a pair.

    The policy keeps the pair plus one kicker and draws 5. The pair's
    rank has exactly 2 copies in hand (a third would have made trips),
    so 2 live outs sit among the 44 unseen. The kept rank reaches three
    of a kind iff at least one out arrives:

        P = 1 - C(42,5)/C(44,5)  ~= 0.21665
    """
    return 1 - Fraction(comb(42, 5), comb(44, 5))
