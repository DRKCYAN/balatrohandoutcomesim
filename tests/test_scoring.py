"""Scoring gates: hand-computed pins, scoring-card selection, level math,
score-max vs type-max divergence, and coherence across the code paths that
produce scores. Every pinned number is derived by hand in a comment.
"""
from __future__ import annotations

import random
import unittest

from balatro_sim.cards import hand, vanilla_deck
from balatro_sim.evaluator import HAND_BASE, HandType, best_of, evaluate
from balatro_sim.experiment import paired_experiment, score_at_least
from balatro_sim.policy import FlushChaser, NoDiscard
from balatro_sim.scoring import (
    LEVEL_INCREMENTS,
    best_play,
    effective_level,
    hand_base_at,
    score,
    scoring_cards,
)
from balatro_sim.simulate import run_distribution, run_phase1

T = HandType


class TestScorePins(unittest.TestCase):
    def test_pinned_scores(self):
        cases = [
            # (5 + 11) x 1
            ("AS", None, 16),
            # (10 + 10+10) x 2 -- the 7, 4, 2 kickers must add nothing
            ("KS KH 7D 4C 2S", None, 60),
            ("KS KH", None, 60),
            # (35 + 10+10+10+9+7) x 4
            ("KH QH JH 9H 7H", None, 324),
            # Pair L2: (10+15 + 20) x (2+1)
            ("KS KH 7D 4C 2S", {T.PAIR: 2}, 135),
            # (40 + 2+2+2+3+3) x 4
            ("2C 2D 2H 3S 3H", None, 208),
            # (35 + 11+10+10+10+9) x 4
            ("AH KH QH JH 9H", None, 340),
            # wheel straight flush: (100 + 11+2+3+4+5) x 8
            ("AH 2H 3H 4H 5H", None, 1000),
            # royal: (100 + 10+10+10+10+11) x 8
            ("10D JD QD KD AD", None, 1208),
            # royal reads the straight-flush level: L2 = (140 + 51) x 12
            ("10D JD QD KD AD", {T.STRAIGHT_FLUSH: 2}, 2292),
            # five of a kind (modified deck): (120 + 5x10) x 12
            ("KS KS KH KH KD", None, 2040),
        ]
        for cards, levels, want in cases:
            with self.subTest(cards=cards, levels=levels):
                self.assertEqual(score(hand(cards), levels), want)


class TestScoringCards(unittest.TestCase):
    def _sc(self, text):
        h = hand(text)
        return scoring_cards(evaluate(h), h)

    def test_selection(self):
        self.assertEqual(self._sc("KS KH 7D 4C 2S"), hand("KS KH"))
        self.assertEqual(self._sc("AS AH KS KD 2C"), hand("AS AH KS KD"))
        self.assertEqual(self._sc("7S 7H 7D KC QD"), hand("7S 7H 7D"))
        self.assertEqual(self._sc("5S 5H 5D 5C KD"), hand("5S 5H 5D 5C"))
        self.assertEqual(self._sc("2S 5D 9C JH KS"), hand("KS"))
        for text in ("2S 3H 4D 5C 6S", "KH QH JH 9H 7H", "3S 3H 3D 9C 9S"):
            with self.subTest(text=text):
                self.assertEqual(len(self._sc(text)), 5)


class TestLevels(unittest.TestCase):
    def test_level_one_is_hand_base(self):
        for t in HandType:
            self.assertEqual(hand_base_at(t, 1), HAND_BASE[t])

    def test_royal_shares_neptune(self):
        self.assertEqual(
            LEVEL_INCREMENTS[T.ROYAL_FLUSH], LEVEL_INCREMENTS[T.STRAIGHT_FLUSH]
        )
        self.assertEqual(effective_level({T.STRAIGHT_FLUSH: 3}, T.ROYAL_FLUSH), 3)

    def test_effective_level_defaults_and_bounds(self):
        self.assertEqual(effective_level(None, T.PAIR), 1)
        self.assertEqual(effective_level({}, T.PAIR), 1)
        with self.assertRaises(ValueError):
            effective_level({T.PAIR: 0}, T.PAIR)

    def test_score_strictly_increases_with_level(self):
        kk = hand("KS KH")
        scores = [score(kk, {T.PAIR: lvl}) for lvl in (1, 2, 3)]
        self.assertEqual(scores, [60, 135, 240])  # (10+15(l-1)+20) x (2+(l-1))


class TestBestPlay(unittest.TestCase):
    def test_face_flush_beats_junk_full_house(self):
        # hearts: 2H 3H AH KH QH -> flush (35 + 2+3+11+10+10) x 4 = 284
        # full house 222+33 -> (40 + 12) x 4 = 208; type-max still says FH
        eight = hand("2C 2D 2H 3S 3H AH KH QH")
        pr = best_play(eight, {})
        self.assertIs(pr.hand_type, T.FLUSH)
        self.assertEqual(pr.score, 284)
        self.assertIs(best_of(eight)[0], T.FULL_HOUSE)

    def test_leveled_pair_beats_flush(self):
        # Pair L9: (10+15*8 + 20) x (2+8) = 1500 > heart flush (35+33)x4=272
        eight = hand("KS KH 2H 5H 7H 9H JH 3C")
        pr = best_play(eight, {T.PAIR: 9})
        self.assertIs(pr.hand_type, T.PAIR)
        self.assertEqual(pr.score, 1500)

    def test_dominates_type_max_play_on_random_hands(self):
        rng = random.Random(99)
        deck = vanilla_deck()
        for _ in range(100):
            eight = rng.sample(deck, 8)
            pr = best_play(eight, {})
            _, type_max_cards = best_of(eight)
            self.assertGreaterEqual(pr.score, score(type_max_cards, {}))
            self.assertIs(evaluate(pr.played), pr.hand_type)


class TestReportScores(unittest.TestCase):
    def test_scored_run_coherence(self):
        rep = run_distribution(vanilla_deck(), 400, seed=3, levels={})
        self.assertEqual(len(rep.scores), 400)
        self.assertGreaterEqual(min(rep.scores), 7)  # (5+2)x1 floor
        self.assertEqual(rep.p_score_at_least(0), 1.0)
        self.assertEqual(rep.p_score_at_least(10**9), 0.0)
        self.assertEqual(
            rep.p_score_at_least(100),
            sum(1 for s in rep.scores if s >= 100) / 400,
        )
        self.assertEqual(rep.score_percentile(0), min(rep.scores))
        self.assertEqual(rep.score_percentile(100), max(rep.scores))

    def test_unscored_run_refuses_score_queries(self):
        rep = run_phase1(vanilla_deck(), 50, seed=1)
        self.assertIsNone(rep.scores)
        with self.assertRaises(ValueError):
            rep.p_score_at_least(100)
        with self.assertRaises(ValueError):
            rep.score_percentile(50)


class TestExperimentScoreCoherence(unittest.TestCase):
    def test_arm_means_equal_distribution_tails_exactly(self):
        deck = vanilla_deck()
        n, seed, blind = 800, 31, 300
        res = paired_experiment(
            deck, n, seed, NoDiscard(), FlushChaser(), 3,
            statistic=score_at_least(blind), levels={},
        )
        for policy, p_arm in ((NoDiscard(), res.p_a), (FlushChaser(), res.p_b)):
            rep = run_distribution(deck, n, seed, policy=policy, discards=3, levels={})
            self.assertEqual(p_arm, rep.p_score_at_least(blind))

    def test_score_statistic_without_levels_is_an_error(self):
        with self.assertRaises(ValueError):
            paired_experiment(
                vanilla_deck(), 10, 0, NoDiscard(), NoDiscard(), 0,
                statistic=score_at_least(100),
            )


if __name__ == "__main__":
    unittest.main()
