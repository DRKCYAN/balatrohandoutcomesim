"""Validation gates for the discard/redraw mechanics (Phase 2).

The zero-discard case had global closed-form truth; with a reactive
policy that is gone, so the gates are:

  1. pinned replacement mechanics on a known deck order,
  2. NoDiscard identical to the Phase 1 loop trial-for-trial,
  3. blind (content-ignorant) discarding leaves the final 8 uniform, so
     it must still match the Phase 1 exact math,
  4. conditional single-discard states are plain hypergeometric draws
     and must match exact_discard.py,
  5. more discards monotonically help the chased statistic.
"""
from __future__ import annotations

import unittest
from collections import Counter
from math import sqrt

from balatro_sim import exact, exact_discard
from balatro_sim.cards import hand, vanilla_deck
from balatro_sim.evaluator import (
    HandType,
    availability,
    best_from_availability,
)
from balatro_sim.policy import BlindDiscard, FlushChaser, MadeHand, NoDiscard
from balatro_sim.simulate import play_out, run_distribution, run_phase1, trial_rng

Z_TOL = 4.5


class TestPlayOutMechanics(unittest.TestCase):
    def test_replacement_order_is_pinned(self):
        # unshuffled vanilla deck: 2S..AS then 2H..AH etc.
        deck = vanilla_deck()
        final = play_out(deck, BlindDiscard(5), discards=1)
        self.assertEqual(final, hand("10S JS QS KS AS 7S 8S 9S"))
        final = play_out(deck, BlindDiscard(5), discards=2)
        self.assertEqual(final, hand("2H 3H 4H 5H 6H 7S 8S 9S"))

    def test_does_not_mutate_the_shuffle(self):
        deck = vanilla_deck()
        before = list(deck)
        play_out(deck, BlindDiscard(5), discards=3)
        self.assertEqual(deck, before)

    def test_policy_stop_is_honoured(self):
        first8 = hand("2S 3H 4D 5C 6S 9H JD KC")  # straight available
        rest = [c for c in vanilla_deck() if c not in set(first8)]
        deck = list(first8) + rest
        self.assertEqual(play_out(deck, MadeHand(), discards=3), first8)

    def test_misbehaving_policies_are_rejected(self):
        class Bad:
            name = "bad"

            def __init__(self, ret):
                self.ret = ret

            def discard(self, hand, discards_left):
                return self.ret

        deck = vanilla_deck()
        for ret in [(0, 1, 2, 3, 4, 5), (0, 0), (7, 8), (-1,)]:
            with self.subTest(ret=ret), self.assertRaises(ValueError):
                play_out(deck, Bad(ret), discards=1)

    def test_deck_exhaustion_raises(self):
        with self.assertRaises(RuntimeError):
            play_out(vanilla_deck()[:10], BlindDiscard(5), discards=1)


class TestNoDiscardEqualsPhase1(unittest.TestCase):
    def test_identical_trial_for_trial(self):
        deck = vanilla_deck()
        a = run_distribution(deck, 1500, seed=7, policy=NoDiscard(), discards=3)
        b = run_phase1(deck, 1500, seed=7)
        self.assertEqual(a.best_counts, b.best_counts)
        self.assertEqual(a.avail_counts, b.avail_counts)
        self.assertEqual(a.inconsistencies, 0)
        self.assertEqual(b.inconsistencies, 0)


class TestBlindDiscardInvariance(unittest.TestCase):
    """Content-blind discarding must reproduce the uniform-deal exact math."""

    N = 20_000

    @classmethod
    def setUpClass(cls):
        cls.report = run_distribution(
            vanilla_deck(), cls.N, seed=11, policy=BlindDiscard(5), discards=3
        )

    def test_no_cross_check_mismatches(self):
        self.assertEqual(self.report.inconsistencies, 0)

    def test_availability_still_matches_exact_math(self):
        for key, frac in exact.availability_exact().items():
            p = float(frac)
            tol = Z_TOL * sqrt(p * (1 - p) / self.N)
            with self.subTest(key=key):
                self.assertLessEqual(abs(self.report.p_avail(key) - p), tol)

    def test_best_high_card_still_matches_exact_math(self):
        p = float(exact.best_is_high_card())
        tol = Z_TOL * sqrt(p * (1 - p) / self.N)
        self.assertLessEqual(abs(self.report.p_best(HandType.HIGH_CARD) - p), tol)


class TestConditionalHypergeometrics(unittest.TestCase):
    """Single-discard states vs exact_discard.py, conditioned trial-side."""

    N = 20_000
    SEED = 13

    @classmethod
    def setUpClass(cls):
        deck = vanilla_deck()
        fc, mh = FlushChaser(), MadeHand()
        cls.n_flush = cls.hit_flush = 0
        cls.n_pair = cls.hit_trips = 0
        for i in range(cls.N):
            shuffled = list(deck)
            trial_rng(cls.SEED, i).shuffle(shuffled)
            initial = tuple(shuffled[:8])
            suit_counts = Counter(c.suit for c in initial)
            if max(suit_counts.values()) == 4:
                final = play_out(shuffled, fc, discards=1)
                cls.n_flush += 1
                cls.hit_flush += availability(final)["flush"]
            if best_from_availability(availability(initial)) is HandType.PAIR:
                rank_counts = Counter(c.rank for c in initial)
                (pair_rank,) = [r for r, c in rank_counts.items() if c == 2]
                final = play_out(shuffled, mh, discards=1)
                cls.n_pair += 1
                cls.hit_trips += sum(1 for c in final if c.rank == pair_rank) >= 3

    def _assert_conditional(self, hits, n_cond, exact_p):
        self.assertGreater(n_cond, 500, "conditioning left too few trials")
        p = float(exact_p)
        tol = Z_TOL * sqrt(p * (1 - p) / n_cond)
        self.assertLessEqual(abs(hits / n_cond - p), tol)

    def test_four_flush_completion_matches_hypergeometric(self):
        self._assert_conditional(
            self.hit_flush, self.n_flush, exact_discard.four_flush_completion()
        )

    def test_pair_to_trips_matches_hypergeometric(self):
        self._assert_conditional(
            self.hit_trips, self.n_pair, exact_discard.pair_to_trips_draw()
        )


class TestMonotonicity(unittest.TestCase):
    def test_flush_chaser_flush_rate_rises_with_discards(self):
        deck = vanilla_deck()
        rates = [
            run_distribution(
                deck, 5000, seed=5, policy=FlushChaser(), discards=d
            ).p_avail("flush")
            for d in range(4)
        ]
        for d in range(3):
            # true gaps are enormous (>0.1); 0.02 is far above noise at n=5000
            self.assertGreater(rates[d + 1], rates[d] + 0.02, rates)


if __name__ == "__main__":
    unittest.main()
