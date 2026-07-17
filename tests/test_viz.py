"""Visualization gates: traces cannot diverge from the engine, and the
flip grid's counts cannot diverge from the paired estimator. Chart
rendering is smoke-tested and skipped when matplotlib is absent.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from balatro_sim.cards import vanilla_deck
from balatro_sim.evaluator import HandType
from balatro_sim.experiment import at_least, paired_experiment, paired_samples
from balatro_sim.policy import BlindDiscard, FlushChaser, MadeHand, NoDiscard
from balatro_sim.simulate import play_out, run_distribution, trial_rng
from balatro_sim.trace import (
    render_trace_html,
    replay_from_shuffled,
    replay_trial,
    trace_html,
)

try:
    import matplotlib  # noqa: F401

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


class TestTracedPlayOut(unittest.TestCase):
    def test_traced_final_equals_untraced_final(self):
        deck = vanilla_deck()
        for policy in (MadeHand(), FlushChaser()):
            for i in range(30):
                shuffled = list(deck)
                trial_rng(17, i).shuffle(shuffled)
                steps: list = []
                traced = play_out(shuffled, policy, 3, trace=steps)
                plain = play_out(shuffled, policy, 3)
                self.assertEqual(traced, plain)
                self.assertLessEqual(len(steps), 3)
                for s in steps:
                    self.assertEqual(len(s.hand_before), 8)
                    self.assertEqual(len(s.discarded_indices), len(s.drawn))


class TestReplayPins(unittest.TestCase):
    def test_blind_discard_replay_is_pinned(self):
        # unshuffled vanilla deck: dealt 2S..9S (a 2-6 straight flush is
        # already available); one blind discard of the first five draws
        # 10S JS QS KS AS -- the final hand holds a royal flush.
        replay = replay_from_shuffled(vanilla_deck(), 0, BlindDiscard(5), 1)
        self.assertIs(replay.initial_best, HandType.STRAIGHT_FLUSH)
        self.assertIs(replay.final_best, HandType.ROYAL_FLUSH)
        self.assertEqual(len(replay.steps), 1)
        self.assertEqual(replay.steps[0].discarded_indices, (0, 1, 2, 3, 4))
        self.assertEqual(str(replay.final[0]), "10S")
        self.assertEqual(str(replay.final[4]), "AS")

    def test_html_contains_pinned_content(self):
        replay = replay_from_shuffled(vanilla_deck(), 0, BlindDiscard(5), 1)
        doc = trace_html([replay], "blind", 1, 0)
        for needle in (
            "Royal Flush",
            "Straight Flush",
            "A</span><span class=\"st\">♠",
            "card black dim",
            "card black new best",
            "blind",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, doc)

    def test_render_is_deterministic(self):
        deck = vanilla_deck()
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "a.html")
            p2 = os.path.join(tmp, "b.html")
            render_trace_html(deck, 42, 6, MadeHand(), 3, p1)
            render_trace_html(deck, 42, 6, MadeHand(), 3, p2)
            with open(p1, encoding="utf-8") as f1, open(p2, encoding="utf-8") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_replay_matches_distribution_trial(self):
        deck = vanilla_deck()
        rep = run_distribution(deck, 50, seed=23, policy=FlushChaser(), discards=3)
        from collections import Counter

        replay_counts = Counter(
            replay_trial(deck, 23, i, FlushChaser(), 3).final_best for i in range(50)
        )
        self.assertEqual(replay_counts, rep.best_counts)


class TestPairedSamplesCoherence(unittest.TestCase):
    def test_samples_match_paired_experiment_exactly(self):
        deck = vanilla_deck()
        n, seed = 400, 21
        stat = at_least(HandType.FLUSH)
        samples = paired_samples(deck, n, seed, NoDiscard(), FlushChaser(), 3, stat)
        res = paired_experiment(deck, n, seed, NoDiscard(), FlushChaser(), 3, stat)
        ups = sum(1 for a, b in samples if b > a)
        downs = sum(1 for a, b in samples if b < a)
        self.assertEqual((ups, downs), (res.flips_up, res.flips_down))
        self.assertEqual(sum(a for a, _ in samples) / n, res.p_a)
        self.assertEqual(sum(b for _, b in samples) / n, res.p_b)


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed (optional dependency)")
class TestChartSmoke(unittest.TestCase):
    def _assert_png(self, path):
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 1_000)

    def test_all_charts_render(self):
        from balatro_sim import charts

        deck = vanilla_deck()
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "dist.png")
            charts.distribution_chart(
                [
                    run_distribution(deck, 300, 1, policy=NoDiscard(), discards=0),
                    run_distribution(deck, 300, 1, policy=FlushChaser(), discards=3),
                ],
                p,
            )
            self._assert_png(p)

            p = os.path.join(tmp, "conv.png")
            charts.convergence_chart(
                FlushChaser(), 3, 400, 2, HandType.FLUSH, p, deck=deck
            )
            self._assert_png(p)

            p = os.path.join(tmp, "disc.png")
            charts.discards_curve(
                [MadeHand(), FlushChaser()], 2, 300, 3, HandType.FLUSH, p, deck=deck
            )
            self._assert_png(p)

            p = os.path.join(tmp, "flips.png")
            charts.flip_grid(
                NoDiscard(), FlushChaser(), 3, 225, 4, HandType.FLUSH, p, deck=deck
            )
            self._assert_png(p)


if __name__ == "__main__":
    unittest.main()
