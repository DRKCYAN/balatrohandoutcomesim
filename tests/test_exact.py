"""Internal consistency of the exact-math module: the rank-multiset
enumerator and the closed forms must agree to the exact Fraction where
they overlap."""
from __future__ import annotations

import unittest
from math import comb

from balatro_sim import exact


class TestExact(unittest.TestCase):
    def test_partition_weights_tile_the_sample_space(self):
        self.assertEqual(exact.sanity_total(), comb(52, 8))

    def test_pair_enumeration_matches_closed_form(self):
        self.assertEqual(exact.pair_or_better(), exact.pair_or_better_closed_form())

    def test_quads_enumeration_matches_closed_form(self):
        self.assertEqual(exact.quads_available(), exact.quads_closed_form())

    def test_orderings_and_bounds(self):
        av = exact.availability_exact()
        for key, p in av.items():
            with self.subTest(key=key):
                self.assertGreater(p, 0)
                self.assertLess(p, 1)
        self.assertLessEqual(av["royal_flush"], av["straight_flush"])
        self.assertLessEqual(av["straight_flush"], av["straight"])
        self.assertLessEqual(av["straight_flush"], av["flush"])
        self.assertLessEqual(av["full_house"], av["two_pair"])
        self.assertLessEqual(av["two_pair"], av["pair"])
        self.assertLessEqual(av["four_of_a_kind"], av["three_of_a_kind"])
        self.assertLessEqual(av["three_of_a_kind"], av["pair"])
        # pinned exact values (6 dp): regression pins for the three techniques
        self.assertAlmostEqual(float(av["straight"]), 0.098162, places=6)
        self.assertAlmostEqual(float(av["flush"]), 0.069640, places=6)
        self.assertAlmostEqual(float(av["pair"]), 0.887920, places=6)

    def test_best_high_card_below_no_pair(self):
        self.assertLess(exact.best_is_high_card(), 1 - exact.pair_or_better())


if __name__ == "__main__":
    unittest.main()
