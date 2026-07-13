"""Exhaustive ground truth: evaluate() over all C(52,5) = 2,598,960 hands.

The canonical 5-card poker frequencies are the strongest available check
of the evaluator -- fully independent, known to the digit. Balatro's
taxonomy matches standard poker on a vanilla deck (wheel counts, no
wraparound), with the royal flush split out of straight flushes.

Slow (~10-30s). Set BALATRO_SKIP_SLOW=1 to skip.
"""
from __future__ import annotations

import os
import unittest
from collections import Counter
from itertools import combinations

from balatro_sim.cards import vanilla_deck
from balatro_sim.evaluator import HandType, evaluate

T = HandType

CANON = {
    T.ROYAL_FLUSH: 4,
    T.STRAIGHT_FLUSH: 36,
    T.FOUR_OF_A_KIND: 624,
    T.FULL_HOUSE: 3_744,
    T.FLUSH: 5_108,
    T.STRAIGHT: 10_200,
    T.THREE_OF_A_KIND: 54_912,
    T.TWO_PAIR: 123_552,
    T.PAIR: 1_098_240,
    T.HIGH_CARD: 1_302_540,
}


@unittest.skipIf(os.environ.get("BALATRO_SKIP_SLOW"), "BALATRO_SKIP_SLOW set")
class TestExhaustiveFiveCard(unittest.TestCase):
    def test_all_five_card_hands_match_canon(self):
        counts: Counter = Counter()
        for combo in combinations(vanilla_deck(), 5):
            counts[evaluate(combo)] += 1
        self.assertEqual(sum(counts.values()), 2_598_960)
        for t, want in CANON.items():
            with self.subTest(hand_type=t.name):
                self.assertEqual(counts.get(t, 0), want)
        for t in (T.FIVE_OF_A_KIND, T.FLUSH_HOUSE, T.FLUSH_FIVE):
            self.assertEqual(counts.get(t, 0), 0)


if __name__ == "__main__":
    unittest.main()
