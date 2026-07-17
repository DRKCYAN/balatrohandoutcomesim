"""Gates for the configurable held-hand size (--hand-size): pinned
replacement mechanics at a non-8 size, final hand length == hand_size,
blind coherence, monotone availability, CRN pairing, and validation.
"""
from __future__ import annotations

import unittest

from balatro_sim.cards import hand, vanilla_deck
from balatro_sim.evaluator import HandType
from balatro_sim.experiment import at_least, paired_blind_experiment, paired_experiment
from balatro_sim.policy import BlindDiscard, FlushChaser, NoDiscard
from balatro_sim.simulate import (
    play_blind,
    play_out,
    run_blinds,
    run_distribution,
    trial_rng,
)
from balatro_sim.trace import replay_from_shuffled


class TestMechanicsAtNonDefaultHandSize(unittest.TestCase):
    def test_replacement_draw_index_starts_at_hand_size(self):
        # unshuffled vanilla deck, hand_size=10: hand is 2S..JS, so the
        # first blind discard draws QS KS AS 2H 3H (positions 10-14),
        # proving the draw index starts at hand_size (10), not 8.
        final = play_out(vanilla_deck(), BlindDiscard(5), discards=1, hand_size=10)
        self.assertEqual(final, hand("QS KS AS 2H 3H 7S 8S 9S 10S JS"))
        self.assertEqual(len(final), 10)

    def test_final_hand_length_tracks_hand_size(self):
        deck = vanilla_deck()
        for hs in (5, 7, 8, 11):
            shuffled = list(deck)
            trial_rng(3, 0).shuffle(shuffled)
            final = play_out(shuffled, FlushChaser(), discards=3, hand_size=hs)
            with self.subTest(hand_size=hs):
                self.assertEqual(len(final), hs)

    def test_report_carries_hand_size(self):
        rep = run_distribution(
            vanilla_deck(), 200, seed=1, policy=NoDiscard(), discards=0, hand_size=6
        )
        self.assertEqual(rep.hand_size, 6)

    def test_seeding_is_reproducible_at_non_default_size(self):
        a = run_distribution(vanilla_deck(), 300, seed=7, policy=FlushChaser(),
                             discards=3, hand_size=10)
        b = run_distribution(vanilla_deck(), 300, seed=7, policy=FlushChaser(),
                             discards=3, hand_size=10)
        self.assertEqual(a.best_counts, b.best_counts)


class TestBlindHandSizeCoherence(unittest.TestCase):
    def test_run_blinds_reproduces_play_blind_trial_for_trial(self):
        deck = vanilla_deck()
        n, seed, hs = 300, 19, 10
        rep = run_blinds(deck, n, seed, policy=FlushChaser(), hands=4,
                         discards=3, levels={}, hand_size=hs)
        for i in range(n):
            shuffled = list(deck)
            trial_rng(seed, i).shuffle(shuffled)
            direct = play_blind(shuffled, FlushChaser(), hands=4, discards=3,
                                levels={}, hand_size=hs)
            with self.subTest(i=i):
                self.assertEqual(rep.totals[i], direct.total)
        self.assertEqual(rep.hand_size, hs)


class TestMonotonicityInHandSize(unittest.TestCase):
    def test_flush_and_straight_availability_rise_with_hand_size(self):
        deck = vanilla_deck()
        rows = {
            hs: run_distribution(deck, 5000, seed=5, policy=NoDiscard(),
                                 discards=0, hand_size=hs)
            for hs in (5, 8, 11)
        }
        for key in ("flush", "straight"):
            ps = [rows[hs].p_avail(key) for hs in (5, 8, 11)]
            # the true gaps are large; 0.01 is far above noise at n=5000
            self.assertGreater(ps[1], ps[0] + 0.01, (key, ps))
            self.assertGreater(ps[2], ps[1] + 0.01, (key, ps))


class TestCRNSurvivesHandSize(unittest.TestCase):
    def test_self_comparison_is_exactly_zero(self):
        res = paired_experiment(
            vanilla_deck(), 200, seed=5,
            policy_a=FlushChaser(), policy_b=FlushChaser(),
            discards=3, statistic=at_least(HandType.FLUSH), hand_size=10,
        )
        self.assertEqual(res.delta, 0.0)
        self.assertEqual(res.se, 0.0)

    def test_blind_self_comparison_is_exactly_zero(self):
        res = paired_blind_experiment(
            vanilla_deck(), 200, seed=5,
            policy_a=NoDiscard(), policy_b=NoDiscard(),
            blind=600, hands=4, discards=3, levels={}, hand_size=10,
        )
        self.assertEqual(res.delta, 0.0)
        self.assertEqual((res.flips_up, res.flips_down), (0, 0))


class TestTraceAtNonDefaultHandSize(unittest.TestCase):
    def test_initial_hand_has_hand_size_cards(self):
        replay = replay_from_shuffled(vanilla_deck(), 0, NoDiscard(), 0, hand_size=5)
        self.assertEqual(len(replay.initial), 5)
        self.assertEqual(len(replay.final), 5)


class TestHandSizeValidation(unittest.TestCase):
    def test_out_of_range_raises(self):
        deck = vanilla_deck()
        for hs in (0, -1, len(deck) + 1):
            with self.subTest(hand_size=hs):
                with self.assertRaises(ValueError):
                    run_distribution(deck, 10, seed=0, policy=NoDiscard(),
                                     discards=0, hand_size=hs)
                with self.assertRaises(ValueError):
                    run_blinds(deck, 10, seed=0, policy=NoDiscard(),
                               hands=4, discards=3, hand_size=hs)

    def test_play_out_rejects_oversized_hand(self):
        with self.assertRaises(ValueError):
            play_out(vanilla_deck()[:10], NoDiscard(), discards=0, hand_size=11)


if __name__ == "__main__":
    unittest.main()
