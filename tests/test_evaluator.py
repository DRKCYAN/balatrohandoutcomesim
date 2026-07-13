"""Unit tests for classification, best_of and the availability cross-check."""
from __future__ import annotations

import random
import unittest

from balatro_sim.cards import Card, card, hand, vanilla_deck
from balatro_sim.evaluator import (
    HandType,
    availability,
    best_from_availability,
    best_of,
    evaluate,
)

T = HandType

# (cards, expected type). Duplicated cards appear only in the
# modified-deck cases at the bottom.
CASES = [
    # high card
    ("AS", T.HIGH_CARD),
    ("2C", T.HIGH_CARD),
    ("AS KH", T.HIGH_CARD),
    ("2S 5D 9C JH KS", T.HIGH_CARD),
    ("2S 3H 4D 5C", T.HIGH_CARD),        # 4-card straight-shape: not a straight
    ("JS QH KD AC 2S", T.HIGH_CARD),     # no wraparound
    ("2H 7H 9H JH", T.HIGH_CARD),        # 4 of one suit: not a flush
    # pair / two pair / trips
    ("AS AH", T.PAIR),
    ("2S 2C 9D", T.PAIR),
    ("KS KH 2C 5D 9H", T.PAIR),
    ("AS AH KS KD", T.TWO_PAIR),
    ("3S 3H 9C 9D KS", T.TWO_PAIR),
    ("7S 7H 7D", T.THREE_OF_A_KIND),
    ("7S 7H 7D KC", T.THREE_OF_A_KIND),
    ("7S 7H 7D KC 2D", T.THREE_OF_A_KIND),
    # straights
    ("AS 2H 3D 4C 5S", T.STRAIGHT),      # wheel
    ("2S 3H 4D 5C 6S", T.STRAIGHT),
    ("9C 10D JH QS KC", T.STRAIGHT),
    ("10S JH QD KC AH", T.STRAIGHT),     # ace-high, mixed suits
    # flush / full house / quads
    ("2H 7H 9H JH KH", T.FLUSH),
    ("3S 3H 3D 9C 9S", T.FULL_HOUSE),
    ("KS KH KD 4C 4D", T.FULL_HOUSE),
    ("5S 5H 5D 5C", T.FOUR_OF_A_KIND),
    ("5S 5H 5D 5C KD", T.FOUR_OF_A_KIND),
    # straight flush / royal
    ("6H 7H 8H 9H 10H", T.STRAIGHT_FLUSH),
    ("AH 2H 3H 4H 5H", T.STRAIGHT_FLUSH),  # wheel SF is not royal
    ("10D JD QD KD AD", T.ROYAL_FLUSH),
    # modified-deck (duplicate card) hands
    ("KS KS KH KH KD", T.FIVE_OF_A_KIND),
    ("KS KS KS KS KS", T.FLUSH_FIVE),
    ("KS KS KS 2S 2S", T.FLUSH_HOUSE),
    ("5S 5S 5S 2S 7S", T.FLUSH),         # flush outranks trips
]


class TestEvaluate(unittest.TestCase):
    def test_classification_table(self):
        for text, want in CASES:
            with self.subTest(hand=text):
                self.assertIs(evaluate(hand(text)), want)

    def test_rejects_bad_sizes(self):
        with self.assertRaises(ValueError):
            evaluate(())
        with self.assertRaises(ValueError):
            evaluate(hand("2S 3S 4S 5S 6S 7S"))


class TestCards(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(card("as"), Card(14, 0))
        self.assertEqual(card("10h"), Card(10, 1))
        self.assertEqual(card("TH"), Card(10, 1))
        self.assertEqual(str(card("10h")), "10H")
        for bad in ("1S", "AX", "S", "11H"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                card(bad)

    def test_vanilla_deck(self):
        deck = vanilla_deck()
        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck)), 52)


class TestBestOf(unittest.TestCase):
    def test_flush_beats_straight(self):
        # hearts flush and a 2-6 straight are both available; flush ranks higher
        t, cards = best_of(hand("2H 5H 9H JH KH 3S 4D 6C"))
        self.assertIs(t, T.FLUSH)
        self.assertIs(evaluate(cards), T.FLUSH)

    def test_quads_beat_flush(self):
        t, _ = best_of(hand("9S 9H 9D 9C 2H 5H 7H JH"))
        self.assertIs(t, T.FOUR_OF_A_KIND)

    def test_royal_found_in_eight(self):
        eight = hand("10D JD QD KD AD 2S 2H 3C")
        t, cards = best_of(eight)
        self.assertIs(t, T.ROYAL_FLUSH)
        self.assertIs(evaluate(cards), T.ROYAL_FLUSH)
        av = availability(eight)
        self.assertTrue(av["royal_flush"] and av["straight_flush"])
        self.assertFalse(av["two_pair"])  # only the 2s pair up
        self.assertIs(best_from_availability(av), T.ROYAL_FLUSH)

    def test_wheel_straight_flush_in_eight(self):
        eight = hand("AH 2H 3H 4H 5H 9S 9C 9D")
        t, _ = best_of(eight)
        self.assertIs(t, T.STRAIGHT_FLUSH)
        av = availability(eight)
        self.assertTrue(av["straight_flush"])
        self.assertFalse(av["royal_flush"])

    def test_full_house_over_flushless_hand(self):
        t, _ = best_of(hand("KS KH KD 4C 4D 2H 5H 7H"))
        self.assertIs(t, T.FULL_HOUSE)

    def test_agrees_with_availability_floor_on_random_hands(self):
        # best_of (subset enumeration) and the availability floor
        # (whole-hand counting) are independent implementations of the
        # same quantity; they must agree exactly on a vanilla deck.
        rng = random.Random(123)
        deck = vanilla_deck()
        for _ in range(300):
            eight = rng.sample(deck, 8)
            t, cards = best_of(eight)
            self.assertIs(evaluate(cards), t)
            self.assertIs(best_from_availability(availability(eight)), t)


if __name__ == "__main__":
    unittest.main()
