"""Phase 1 exit gate: a fixed-seed 20,000-trial run must match the
hand-derived math (0 mismatches, availability and P(High Card) within
4.5 SE of exact). Also pins the seeding contract.
"""
from __future__ import annotations

import unittest
from math import sqrt

from balatro_sim import exact
from balatro_sim.cards import vanilla_deck
from balatro_sim.evaluator import HandType
from balatro_sim.simulate import run_phase1, trial_rng

N = 20_000
SEED = 42
Z_TOL = 4.5


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_phase1(vanilla_deck(), N, SEED)
        cls.exact_avail = exact.availability_exact()

    def test_no_cross_check_mismatches(self):
        self.assertEqual(self.report.inconsistencies, 0)

    def test_availability_matches_exact_math(self):
        for key, frac in self.exact_avail.items():
            p = float(frac)
            tol = Z_TOL * sqrt(p * (1 - p) / N)
            with self.subTest(key=key, exact=round(p, 6)):
                self.assertLessEqual(abs(self.report.p_avail(key) - p), tol)

    def test_best_high_card_matches_exact_math(self):
        p = float(exact.best_is_high_card())
        tol = Z_TOL * sqrt(p * (1 - p) / N)
        self.assertLessEqual(abs(self.report.p_best(HandType.HIGH_CARD) - p), tol)

    def test_best_counts_cover_all_trials(self):
        self.assertEqual(sum(self.report.best_counts.values()), N)
        for t in (HandType.FIVE_OF_A_KIND, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE):
            self.assertEqual(self.report.best_counts.get(t, 0), 0)

    def test_seeding_is_reproducible(self):
        a = run_phase1(vanilla_deck(), 300, seed=7)
        b = run_phase1(vanilla_deck(), 300, seed=7)
        self.assertEqual(a.best_counts, b.best_counts)
        self.assertEqual(a.avail_counts, b.avail_counts)
        c = run_phase1(vanilla_deck(), 300, seed=8)
        self.assertNotEqual(a.best_counts, c.best_counts)

    def test_trial_rng_stream_is_stable(self):
        # Pin the per-trial stream (string seeds hash via SHA-512, so these
        # hold on any platform); if it moves, every recorded result shifts.
        r = trial_rng(0, 0)
        self.assertEqual([r.randrange(52) for _ in range(6)],
                         [37, 43, 36, 37, 50, 6])
        d = list(range(52))
        trial_rng(42, 0).shuffle(d)
        self.assertEqual(d[:8], [35, 7, 45, 15, 36, 32, 18, 0])
        self.assertNotEqual(trial_rng(0, 1).randrange(2**30),
                            trial_rng(0, 2).randrange(2**30))


if __name__ == "__main__":
    unittest.main()
