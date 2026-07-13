"""Pinned behaviour of every policy rule branch.

Policies are part of the measured object (PLAN.md section 7), so their
decisions are pinned exactly: any change to a policy is a change to what
the simulator measures and must show up here.
"""
from __future__ import annotations

import unittest

from balatro_sim.cards import hand
from balatro_sim.policy import (
    POLICY_NAMES,
    BlindDiscard,
    FlushChaser,
    MadeHand,
    NoDiscard,
    get_policy,
)


class TestNoDiscardAndBlind(unittest.TestCase):
    def test_no_discard_always_stops(self):
        self.assertEqual(NoDiscard().discard(hand("2S 3H 4D 5C 6S 9H JD KC"), 3), ())

    def test_blind_discards_first_k_regardless_of_content(self):
        royal = hand("10D JD QD KD AD 2S 2H 3C")
        self.assertEqual(BlindDiscard().discard(royal, 3), (0, 1, 2, 3, 4))
        self.assertEqual(BlindDiscard(2).discard(royal, 1), (0, 1))

    def test_blind_rejects_bad_k(self):
        for k in (0, 6):
            with self.assertRaises(ValueError):
                BlindDiscard(k)

    def test_registry(self):
        for name in POLICY_NAMES:
            self.assertEqual(get_policy(name).name, name)
        with self.assertRaises(ValueError):
            get_policy("optimal")  # no such thing


class TestMadeHand(unittest.TestCase):
    def setUp(self):
        self.pi = MadeHand()

    def test_stops_on_available_straight(self):
        self.assertEqual(self.pi.discard(hand("2S 3H 4D 5C 6S 9H JD KC"), 3), ())

    def test_stops_on_available_flush(self):
        self.assertEqual(self.pi.discard(hand("2H 5H 7H 9H KH 3S 4D 8C"), 3), ())

    def test_trips_keeps_trips_discards_five(self):
        self.assertEqual(
            self.pi.discard(hand("7S 7H 7D KC QD 2C 3H 4S"), 3), (3, 4, 5, 6, 7)
        )

    def test_two_pair_keeps_both_pairs(self):
        self.assertEqual(
            self.pi.discard(hand("AS AH KS KD 2C 3D 5H 7C"), 3), (4, 5, 6, 7)
        )

    def test_three_pairs_lowest_pair_goes(self):
        self.assertEqual(
            self.pi.discard(hand("AS AH KS KD 2C 2D 5H 7C"), 3), (4, 5, 6, 7)
        )

    def test_pair_keeps_pair_plus_highest_kicker(self):
        self.assertEqual(
            self.pi.discard(hand("9S 9H KD QC 7H 5D 3C 2S"), 3), (3, 4, 5, 6, 7)
        )

    def test_high_card_keeps_three_highest(self):
        self.assertEqual(
            self.pi.discard(hand("AS KH QD 9C 7H 5D 3C 2S"), 3), (3, 4, 5, 6, 7)
        )

    def test_does_not_chase_a_four_straight(self):
        # 2-3-4-5 + 9-J-Q-K: no made straight, no pair -> plain high-card
        # branch (keeps K,Q,J). The non-chasing is the documented contrast
        # with FlushChaser.
        self.assertEqual(
            self.pi.discard(hand("2S 3H 4D 5C 9H JD QC KS"), 3), (0, 1, 2, 3, 4)
        )


class TestFlushChaser(unittest.TestCase):
    def setUp(self):
        self.pi = FlushChaser()

    def test_stops_on_made_flush(self):
        self.assertEqual(self.pi.discard(hand("2H 5H 7H 9H KH 3S 4D 8C"), 3), ())

    def test_four_suited_discards_the_four_off_suit(self):
        self.assertEqual(
            self.pi.discard(hand("2H 5H 9H JH KS QD 3C 7C"), 3), (4, 5, 6, 7)
        )

    def test_suit_tie_breaks_to_lowest_suit_index(self):
        # spades and hearts tied 3-3: chase spades (suit index 0)
        self.assertEqual(
            self.pi.discard(hand("2S 5S 9S 3H 6H 10H 4D 8D"), 3), (3, 4, 5, 6, 7)
        )

    def test_six_off_suit_keeps_the_highest(self):
        # all suits tied 2-2-2-2: chase spades; six off-suit cards but only
        # five may go -- the king of hearts stays
        self.assertEqual(
            self.pi.discard(hand("AS 2S KH 3H QD 4D JC 5C"), 3), (3, 4, 5, 6, 7)
        )


if __name__ == "__main__":
    unittest.main()
