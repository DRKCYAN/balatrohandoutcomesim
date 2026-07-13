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
from .experiment import PairedResult, at_least, paired_experiment, paired_samples
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
    DiscardStep,
    DistributionReport,
    Phase1Report,
    deal,
    play_out,
    run_distribution,
    run_phase1,
    se,
    trial_rng,
)
from .trace import TrialReplay, render_trace_html, replay_trial

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
    "paired_samples",
    "POLICY_NAMES",
    "BlindDiscard",
    "FlushChaser",
    "MadeHand",
    "NoDiscard",
    "Policy",
    "get_policy",
    "DiscardStep",
    "DistributionReport",
    "Phase1Report",
    "deal",
    "play_out",
    "run_distribution",
    "run_phase1",
    "se",
    "trial_rng",
    "TrialReplay",
    "render_trace_html",
    "replay_trial",
]
