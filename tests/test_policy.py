"""Pinned behaviour of every policy rule branch. Policies are part of the
measured object, so any change to one must show up here."""
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
        # The CLI surface: dropping or renaming a policy is a breaking change.
        self.assertEqual(
            POLICY_NAMES,
            ("none", "blind", "madehand", "flushchaser", "pairchaser",
             "twopairchaser", "tripschaser", "fullhousechaser", "quadchaser",
             "highcard"),
        )
        for name in POLICY_NAMES:
            self.assertEqual(get_policy(name).name, name)
        with self.assertRaises(ValueError):
            get_policy("optimal")


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
        # no made straight, no pair -> high-card branch (keeps K,Q,J)
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
        self.assertEqual(
            self.pi.discard(hand("2S 5S 9S 3H 6H 10H 4D 8D"), 3), (3, 4, 5, 6, 7)
        )

    def test_six_off_suit_keeps_the_highest(self):
        # all suits tied 2-2-2-2: chase spades; only 5 go, the KH stays
        self.assertEqual(
            self.pi.discard(hand("AS 2S KH 3H QD 4D JC 5C"), 3), (3, 4, 5, 6, 7)
        )


class TestRankChasers(unittest.TestCase):
    def test_stops_on_made_target(self):
        self.assertEqual(
            get_policy("pairchaser").discard(hand("9S 9H KD QC 7H 5D 3C 2S"), 3), ()
        )
        self.assertEqual(
            get_policy("tripschaser").discard(hand("7S 7H 7D KC QD 2C 3H 4S"), 3), ()
        )
        self.assertEqual(
            get_policy("quadchaser").discard(hand("7S 7H 7D 7C KD QC 3H 2S"), 3), ()
        )

    def test_stops_on_better_than_target(self):
        self.assertEqual(
            get_policy("pairchaser").discard(hand("2H 5H 7H 9H KH 3S 4D 8C"), 3), ()
        )
        self.assertEqual(
            get_policy("quadchaser").discard(hand("5H 6H 7H 8H 9H 2S 3C 4D"), 3), ()
        )

    def test_quadchaser_breaks_a_made_flush(self):
        # FLUSH < FOUR_OF_A_KIND: keeps the kings, discards 5 lowest others
        self.assertEqual(
            get_policy("quadchaser").discard(hand("2H 5H 7H 9H KH KS 3C 4D"), 3),
            (0, 1, 2, 6, 7),
        )

    def test_group_tie_breaks_to_higher_rank(self):
        # two pairs, one group allowed: aces kept, kings broken (KD spared by the cap)
        self.assertEqual(
            get_policy("quadchaser").discard(hand("AS AH KS KD 2C 3D 5H 7C"), 3),
            (2, 4, 5, 6, 7),
        )

    def test_fullhousechaser_keeps_both_pairs(self):
        self.assertEqual(
            get_policy("fullhousechaser").discard(hand("AS AH KS KD 2C 3D 5H 7C"), 3),
            (4, 5, 6, 7),
        )

    def test_fullhousechaser_three_pairs_lowest_goes(self):
        self.assertEqual(
            get_policy("fullhousechaser").discard(hand("AS AH KS KD 2C 2D 5H 7C"), 3),
            (4, 5, 6, 7),
        )

    def test_fullhousechaser_bare_trips_discards_five(self):
        self.assertEqual(
            get_policy("fullhousechaser").discard(hand("7S 7H 7D KC QD 2C 3H 4S"), 3),
            (3, 4, 5, 6, 7),
        )

    def test_no_group_keeps_three_highest(self):
        # empty keep set: the 5-cap leaves the 3 highest
        self.assertEqual(
            get_policy("pairchaser").discard(hand("AS KH QD 9C 7H 5D 3C 2S"), 3),
            (3, 4, 5, 6, 7),
        )

    def test_twopairchaser_keeps_pair_plus_highest_kicker(self):
        self.assertEqual(
            get_policy("twopairchaser").discard(hand("9S 9H KD QC 7H 5D 3C 2S"), 3),
            (3, 4, 5, 6, 7),
        )

    def test_tripschaser_chases_from_a_pair(self):
        self.assertEqual(
            get_policy("tripschaser").discard(hand("9S 9H KD QC 7H 5D 3C 2S"), 3),
            (3, 4, 5, 6, 7),
        )


class TestHighCardChaser(unittest.TestCase):
    def setUp(self):
        self.pi = get_policy("highcard")

    def test_breaks_a_low_pair(self):
        self.assertEqual(
            self.pi.discard(hand("2S 2H KD QC 7H 5D 3C 9S"), 3), (0, 1, 4, 5, 6)
        )

    def test_breaks_even_a_dealt_royal_flush(self):
        self.assertEqual(
            self.pi.discard(hand("10S JS QS KS AS 2H 3D 4C"), 3), (0, 1, 5, 6, 7)
        )

    def test_never_stands_pat(self):
        self.assertEqual(
            self.pi.discard(hand("AS KS QH JD 10C 9H 8D 7C"), 3), (3, 4, 5, 6, 7)
        )


if __name__ == "__main__":
    unittest.main()
