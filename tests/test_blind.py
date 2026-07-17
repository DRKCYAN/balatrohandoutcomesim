"""Blind-trial gates (4 hands, shared discards). Load-bearing: hands=1
equivalence -- a one-hand blind must reproduce the single-hand engine
trial-for-trial for every policy (same replacement rule).
"""
from __future__ import annotations

import unittest

from balatro_sim.cards import vanilla_deck
from balatro_sim.evaluator import HandType
from balatro_sim.experiment import paired_blind_experiment
from balatro_sim.policy import BlindDiscard, FlushChaser, MadeHand, NoDiscard
from balatro_sim.simulate import play_blind, run_blinds, run_distribution

T = HandType


class TestPlayBlindPinnedWalkthrough(unittest.TestCase):
    """Unshuffled vanilla deck (2S..AS, 2H..AH, ...), NoDiscard, level 1.

    hand 1: 2S-9S       -> play 5-9S straight flush: (100+35)x8 = 1080
                           refill 10S,JS,QS,KS,AS
    hand 2: royal spades -> (100+51)x8 = 1208; refill 2H-6H
    hand 3: 2-6H SF      -> (100+20)x8 = 960;  refill 7H-JH
    hand 4: 7-JH SF      -> (100+44)x8 = 1152
    total 1080+1208+960+1152 = 4400
    """

    def test_uncensored_four_hands(self):
        r = play_blind(vanilla_deck(), NoDiscard(), hands=4, discards=3)
        self.assertEqual(r.hand_scores, (1080, 1208, 960, 1152))
        self.assertEqual(r.total, 4400)
        self.assertEqual(r.hand_types, (
            T.STRAIGHT_FLUSH, T.ROYAL_FLUSH, T.STRAIGHT_FLUSH, T.STRAIGHT_FLUSH,
        ))
        self.assertIsNone(r.cleared)
        self.assertEqual((r.hands_used, r.discards_used), (4, 0))

    def test_early_stop_at_clear(self):
        r = play_blind(vanilla_deck(), NoDiscard(), hands=4, discards=3, blind=600)
        self.assertTrue(r.cleared)
        self.assertEqual((r.total, r.hands_used), (1080, 1))

    def test_failed_blind(self):
        r = play_blind(vanilla_deck(), NoDiscard(), hands=4, discards=3, blind=5000)
        self.assertFalse(r.cleared)
        self.assertEqual((r.total, r.hands_used), (4400, 4))


class TestShrinkingDeck(unittest.TestCase):
    def test_twelve_card_deck_walkthrough(self):
        # deck = 2S..KS (12 cards). hand 2S-9S -> 5-9S SF 1080, refill only
        # 10S,JS,QS,KS (deck empty; hand shrinks to 7). Then the best
        # flush 4S,10S,JS,QS,KS = (35+44)x4 = 316, no refill (hand: 2S,3S).
        # Then 3S high card (5+3)x1 = 8, then 2S (5+2)x1 = 7; hand empty
        # with a fifth hand still available -> loop ends at 4 plays.
        r = play_blind(vanilla_deck()[:12], NoDiscard(), hands=5, discards=3)
        self.assertEqual(r.hand_scores, (1080, 316, 8, 7))
        self.assertEqual(r.total, 1411)
        self.assertEqual(r.hands_used, 4)


class TestSingleHandEquivalence(unittest.TestCase):
    def test_one_hand_blind_equals_single_hand_engine(self):
        deck = vanilla_deck()
        n, seed = 300, 17
        for policy in (NoDiscard(), MadeHand(), FlushChaser(), BlindDiscard(5)):
            with self.subTest(policy=policy.name):
                blind_rep = run_blinds(
                    deck, n, seed, policy=policy, hands=1, discards=3, levels={}
                )
                single_rep = run_distribution(
                    deck, n, seed, policy=policy, discards=3, levels={}
                )
                self.assertEqual(blind_rep.totals, single_rep.scores)


class TestResourceAccounting(unittest.TestCase):
    def test_blind_discard_burns_budget_then_is_forced_to_play(self):
        r = play_blind(vanilla_deck(), BlindDiscard(5), hands=4, discards=3)
        self.assertEqual(r.discards_used, 3)
        self.assertEqual(r.hands_used, 4)
        self.assertEqual(len(r.hand_scores), 4)

    def test_zero_discards_allowed(self):
        r = play_blind(vanilla_deck(), FlushChaser(), hands=4, discards=0)
        self.assertEqual(r.discards_used, 0)
        self.assertEqual(r.hands_used, 4)

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            play_blind(vanilla_deck(), NoDiscard(), hands=0)
        with self.assertRaises(ValueError):
            play_blind(vanilla_deck(), NoDiscard(), hands=4, discards=-1)


class TestPairedBlind(unittest.TestCase):
    def test_self_comparison_is_exactly_zero(self):
        res = paired_blind_experiment(
            vanilla_deck(), 200, seed=5,
            policy_a=FlushChaser(), policy_b=FlushChaser(),
            blind=600, hands=4, discards=3, levels={},
        )
        self.assertEqual(res.delta, 0.0)
        self.assertEqual(res.se, 0.0)
        self.assertEqual((res.flips_up, res.flips_down), (0, 0))

    def test_arm_means_equal_run_blinds_exactly(self):
        deck = vanilla_deck()
        n, seed, blind = 400, 21, 600
        res = paired_blind_experiment(
            deck, n, seed, NoDiscard(), FlushChaser(),
            blind=blind, hands=4, discards=3, levels={},
        )
        for policy, p_arm in ((NoDiscard(), res.p_a), (FlushChaser(), res.p_b)):
            rep = run_blinds(
                deck, n, seed, policy=policy, hands=4, discards=3,
                blind=blind, levels={},
            )
            self.assertEqual(p_arm, rep.p_clear)


class TestMonotonicity(unittest.TestCase):
    def test_p_clear_rises_with_hands(self):
        deck = vanilla_deck()
        ps = [
            run_blinds(
                deck, 800, seed=9, policy=FlushChaser(), hands=h,
                discards=3, blind=600, levels={},
            ).p_clear
            for h in (1, 2, 4)
        ]
        self.assertGreater(ps[1], ps[0] + 0.02, ps)
        self.assertGreater(ps[2], ps[1] + 0.02, ps)


if __name__ == "__main__":
    unittest.main()
