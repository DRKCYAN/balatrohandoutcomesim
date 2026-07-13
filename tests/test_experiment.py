"""The paired estimator: CRN guarantees and coherence with the trial loop."""
from __future__ import annotations

import unittest

from balatro_sim.cards import vanilla_deck
from balatro_sim.evaluator import HandType
from balatro_sim.experiment import at_least, paired_experiment
from balatro_sim.policy import FlushChaser, NoDiscard
from balatro_sim.simulate import run_distribution


class TestAtLeast(unittest.TestCase):
    def test_indicator(self):
        stat = at_least(HandType.PAIR)
        self.assertEqual(stat(HandType.HIGH_CARD), 0.0)
        self.assertEqual(stat(HandType.PAIR), 1.0)
        self.assertEqual(stat(HandType.FLUSH), 1.0)


class TestPairedExperiment(unittest.TestCase):
    def test_self_comparison_is_exactly_zero(self):
        # CRN construction: identical shuffles + deterministic policy
        # => identical arms => every D_i is 0, not just their mean.
        res = paired_experiment(
            vanilla_deck(), 400, seed=3,
            policy_a=FlushChaser(), policy_b=FlushChaser(),
            discards=3, statistic=at_least(HandType.FLUSH),
        )
        self.assertEqual(res.delta, 0.0)
        self.assertEqual(res.se, 0.0)
        self.assertEqual((res.flips_up, res.flips_down), (0, 0))
        self.assertEqual(res.p_a, res.p_b)

    def test_flush_chaser_beats_no_discard_on_flushes(self):
        # Plan phrased this as delta(A-B) < 0; convention here is B - A,
        # so the equivalent gate is delta > 0 with the CI excluding zero.
        res = paired_experiment(
            vanilla_deck(), 4000, seed=42,
            policy_a=NoDiscard(), policy_b=FlushChaser(),
            discards=3, statistic=at_least(HandType.FLUSH),
        )
        self.assertGreater(res.p_b, res.p_a)
        self.assertGreater(res.delta, 0.0)
        lo, _ = res.ci95
        self.assertGreater(lo, 0.0, "CI must exclude zero")
        self.assertGreater(res.flips_up, res.flips_down)

    def test_arm_means_cohere_with_run_distribution(self):
        # Same seed => identical trial_rng streams => the paired arms and
        # run_distribution see the exact same final hands, so the arm
        # means must equal the distribution tail probabilities EXACTLY.
        deck = vanilla_deck()
        n, seed = 2000, 9
        res = paired_experiment(
            deck, n, seed,
            policy_a=NoDiscard(), policy_b=FlushChaser(),
            discards=3, statistic=at_least(HandType.FLUSH),
        )
        for policy, p_arm in ((NoDiscard(), res.p_a), (FlushChaser(), res.p_b)):
            rep = run_distribution(deck, n, seed, policy=policy, discards=3)
            tail = sum(c for t, c in rep.best_counts.items() if t >= HandType.FLUSH)
            self.assertEqual(p_arm, tail / n)

    def test_rejects_tiny_n(self):
        with self.assertRaises(ValueError):
            paired_experiment(
                vanilla_deck(), 1, seed=0,
                policy_a=NoDiscard(), policy_b=FlushChaser(),
                discards=3, statistic=at_least(HandType.FLUSH),
            )


if __name__ == "__main__":
    unittest.main()
