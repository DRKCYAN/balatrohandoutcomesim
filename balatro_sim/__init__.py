"""Balatro hand-outcome Monte Carlo simulator. See PLAN.md for the spec."""
from .cards import CHIP_VALUE, Card, card, hand, vanilla_deck
from .evaluator import (
    HAND_BASE,
    HandType,
    availability,
    best_from_availability,
    best_of,
    evaluate,
)
from .experiment import PairedResult, at_least, paired_experiment
from .policy import (
    POLICY_NAMES,
    BlindDiscard,
    FlushChaser,
    MadeHand,
    NoDiscard,
    Policy,
    get_policy,
)
from .simulate import (
    DistributionReport,
    Phase1Report,
    deal,
    play_out,
    run_distribution,
    run_phase1,
    se,
    trial_rng,
)

__all__ = [
    "CHIP_VALUE",
    "Card",
    "card",
    "hand",
    "vanilla_deck",
    "HAND_BASE",
    "HandType",
    "availability",
    "best_from_availability",
    "best_of",
    "evaluate",
    "PairedResult",
    "at_least",
    "paired_experiment",
    "POLICY_NAMES",
    "BlindDiscard",
    "FlushChaser",
    "MadeHand",
    "NoDiscard",
    "Policy",
    "get_policy",
    "DistributionReport",
    "Phase1Report",
    "deal",
    "play_out",
    "run_distribution",
    "run_phase1",
    "se",
    "trial_rng",
]
