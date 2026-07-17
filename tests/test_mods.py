"""Phase 4 deck modifications: pinned semantics (selectors match all copies,
matching nothing errors, order matters, apply_all never mutates its input)
plus the engine consequences on duplicate and thinned decks.
"""
from __future__ import annotations

import unittest

from balatro_sim.cards import card, hand, vanilla_deck
from balatro_sim.evaluator import HandType
from balatro_sim.mods import (
    Add,
    Remove,
    Transform,
    apply_all,
    parse_mod,
    parse_mods,
    summarize,
)
from balatro_sim.simulate import run_distribution


class TestRemove(unittest.TestCase):
    def test_exact_card(self):
        deck = apply_all(vanilla_deck(), [parse_mod("remove 2S")])
        self.assertEqual(len(deck), 51)
        self.assertNotIn(card("2S"), deck)

    def test_rank_selector_removes_all_suits(self):
        deck = apply_all(vanilla_deck(), [parse_mod("remove 2 3")])
        self.assertEqual(len(deck), 44)
        self.assertFalse(any(c.rank in (2, 3) for c in deck))

    def test_suit_selector_removes_all_ranks(self):
        deck = apply_all(vanilla_deck(), [parse_mod("remove H")])
        self.assertEqual(len(deck), 39)
        self.assertFalse(any(c.suit == 1 for c in deck))

    def test_selector_hits_every_copy(self):
        deck = apply_all(vanilla_deck(), parse_mods(["add AS AS", "remove AS"]))
        self.assertEqual(len(deck), 51)  # 52 + 2 - 3
        self.assertNotIn(card("AS"), deck)

    def test_no_match_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_all(vanilla_deck(), parse_mods(["remove 2S", "remove 2S"]))
        # within one mod: 'remove 2 2S' -- the rank selector already ate 2S
        with self.assertRaises(ValueError):
            apply_all(vanilla_deck(), [parse_mod("remove 2 2S")])


class TestAdd(unittest.TestCase):
    def test_duplicates_allowed(self):
        deck = apply_all(vanilla_deck(), [parse_mod("add AS AS")])
        self.assertEqual(len(deck), 54)
        self.assertEqual(sum(1 for c in deck if c == card("AS")), 3)

    def test_rank_or_suit_selector_rejected(self):
        for bad in ("add 2", "add H"):
            with self.assertRaises(ValueError):
                parse_mod(bad)


class TestTransform(unittest.TestCase):
    def test_exact_to_exact_creates_a_duplicate(self):
        deck = apply_all(vanilla_deck(), [parse_mod("transform KC>KH")])
        self.assertEqual(len(deck), 52)
        self.assertNotIn(card("KC"), deck)
        self.assertEqual(sum(1 for c in deck if c == card("KH")), 2)

    def test_rank_to_rank_keeps_suit(self):
        deck = apply_all(vanilla_deck(), [parse_mod("transform 7>8")])
        self.assertFalse(any(c.rank == 7 for c in deck))
        self.assertEqual(sum(1 for c in deck if c.rank == 8), 8)
        self.assertEqual(sum(1 for c in deck if c == card("8S")), 2)

    def test_suit_to_suit_keeps_rank(self):
        deck = apply_all(vanilla_deck(), [parse_mod("transform C>H")])
        self.assertEqual(len(deck), 52)
        self.assertFalse(any(c.suit == 3 for c in deck))
        self.assertEqual(sum(1 for c in deck if c.suit == 1), 26)
        self.assertEqual(sum(1 for c in deck if c == card("KH")), 2)

    def test_exact_to_suit_keeps_rank(self):
        deck = apply_all(vanilla_deck(), [parse_mod("transform KC>H")])
        self.assertEqual(sum(1 for c in deck if c == card("KH")), 2)

    def test_no_match_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_all(vanilla_deck(), parse_mods(["remove K", "transform KC>KH"]))

    def test_bad_syntax(self):
        for bad in ("transform KC", "transform KC>", "transform >KH"):
            with self.assertRaises(ValueError):
                parse_mod(bad)


class TestComposition(unittest.TestCase):
    def test_order_matters(self):
        a = apply_all(vanilla_deck(), parse_mods(["transform KC>KH", "remove KH"]))
        b = apply_all(vanilla_deck(), parse_mods(["remove KH", "transform KC>KH"]))
        self.assertEqual(len(a), 50)  # both KH copies removed
        self.assertEqual(len(b), 51)  # KH removed, then KC becomes a fresh KH
        self.assertNotIn(card("KH"), a)
        self.assertIn(card("KH"), b)

    def test_input_deck_is_not_mutated(self):
        deck = vanilla_deck()
        apply_all(deck, parse_mods(["remove 2", "add AS", "transform C>H"]))
        self.assertEqual(sorted(deck), sorted(vanilla_deck()))

    def test_parse_errors(self):
        for bad in ("", "shuffle 2S", "remove", "add", "transform", "remove 1"):
            with self.assertRaises(ValueError):
                parse_mod(bad)

    def test_tokens_are_case_insensitive(self):
        deck = apply_all(vanilla_deck(), [parse_mod("REMOVE 2s h")])
        # '2s' is the exact card 2S (1 gone), 'h' the heart suit (13 gone)
        self.assertEqual(len(deck), 38)
        self.assertNotIn(card("2S"), deck)
        self.assertFalse(any(c.suit == 1 for c in deck))

    def test_summarize(self):
        self.assertEqual(
            summarize(vanilla_deck()), "52 cards, S13 H13 D13 C13"
        )
        dup = apply_all(vanilla_deck(), [parse_mod("add AS")])
        self.assertEqual(
            summarize(dup), "53 cards, S14 H13 D13 C13, duplicates present"
        )


class TestSecretHandsReachable(unittest.TestCase):
    """Duplicate decks make the secret hands playable; the cross-check must
    exempt them while still holding exactly for everything below."""

    def test_five_of_a_kind_no_false_mismatch(self):
        deck = list(hand("AS AH AD AC AS 2H 3D 4C"))
        rep = run_distribution(deck, 30, seed=1)
        self.assertEqual(rep.best_counts[HandType.FIVE_OF_A_KIND], 30)
        self.assertEqual(rep.inconsistencies, 0)

    def test_flush_five_no_false_mismatch(self):
        deck = list(hand("AS AS AS AS AS 2H 3D 4C"))
        rep = run_distribution(deck, 30, seed=1)
        self.assertEqual(rep.best_counts[HandType.FLUSH_FIVE], 30)
        self.assertEqual(rep.inconsistencies, 0)

    def test_flush_house_no_false_mismatch(self):
        deck = list(hand("9H 9H 9H KH KH 2S 3C 4D"))
        rep = run_distribution(deck, 30, seed=1)
        self.assertEqual(rep.best_counts[HandType.FLUSH_HOUSE], 30)
        self.assertEqual(rep.inconsistencies, 0)

    def test_non_secret_check_still_strict_on_duplicate_decks(self):
        # strict equality must hold on every non-secret trial; secret bests are exempt
        deck = apply_all(vanilla_deck(), [parse_mod("add AS KH")])
        rep = run_distribution(deck, 300, seed=2)
        self.assertEqual(rep.inconsistencies, 0)


class TestModifiedDeckDirections(unittest.TestCase):
    """Directional pins with effects far above MC noise (fixed seeds). Note:
    rank-thinning does NOT raise flush availability (suit ops are the flush
    lever); it raises rank-collision density.
    """

    def test_removing_low_ranks_raises_pair_availability(self):
        # 11 ranks instead of 13 in 8 cards: exact 0.888 -> 0.939
        n = 1500
        base = run_distribution(vanilla_deck(), n, seed=7)
        thin = run_distribution(
            apply_all(vanilla_deck(), [parse_mod("remove 2 3")]), n, seed=7
        )
        self.assertGreater(thin.p_avail("pair"), base.p_avail("pair"))
        self.assertEqual(thin.inconsistencies, 0)

    def test_suit_conversion_raises_flush_availability(self):
        n = 1500
        base = run_distribution(vanilla_deck(), n, seed=7)
        merged = run_distribution(
            apply_all(vanilla_deck(), [parse_mod("transform C>H")]), n, seed=7
        )
        self.assertGreater(merged.p_avail("flush"), 3 * base.p_avail("flush"))
        self.assertEqual(merged.inconsistencies, 0)


if __name__ == "__main__":
    unittest.main()
