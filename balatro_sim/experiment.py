"""Paired comparison with common random numbers: both arms replay the same
shuffle per trial, so D_i is nonzero only where one arm flips -- that
collapses Var(delta_hat). Sign convention: delta = p_b - p_a. Statistics
are functions of a PlayResult (at_least on type, score_at_least on score).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Optional, Sequence

from .cards import Card
from .evaluator import HandType, best_of
from .policy import Policy
from .scoring import Levels, PlayResult, best_play
from .simulate import play_blind, play_out, trial_rng

Statistic = Callable[[PlayResult], float]


def at_least(t: HandType) -> Statistic:
    """Indicator: the played hand's type is t or better."""
    return lambda pr: 1.0 if pr.hand_type >= t else 0.0


def score_at_least(blind: float) -> Statistic:
    """Indicator: the played hand scores blind or more. The experiment
    must be run with levels= (else there is no score to threshold)."""

    def stat(pr: PlayResult) -> float:
        if pr.score is None:
            raise ValueError(
                "score_at_least needs a scored run: pass levels= (e.g. {}) "
                "to paired_experiment/paired_samples"
            )
        return 1.0 if pr.score >= blind else 0.0

    return stat


def _play(final: tuple[Card, ...], levels: Optional[Levels]) -> PlayResult:
    if levels is None:
        t, played = best_of(final)
        return PlayResult(t, played, None)
    return best_play(final, levels)


def paired_samples(
    deck: Sequence[Card],
    n: int,
    seed: int,
    policy_a: Policy,
    policy_b: Policy,
    discards: int,
    statistic: Statistic,
    levels: Optional[Levels] = None,
    hand_size: int = 8,
) -> list[tuple[float, float]]:
    """Per-trial (x_a, x_b) pairs under CRN (the raw material for
    visualisations). Same streams as paired_experiment; materialises n pairs."""
    out: list[tuple[float, float]] = []
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        x_a = float(statistic(_play(play_out(shuffled, policy_a, discards, hand_size), levels)))
        x_b = float(statistic(_play(play_out(shuffled, policy_b, discards, hand_size), levels)))
        out.append((x_a, x_b))
    return out


def paired_blind_experiment(
    deck: Sequence[Card],
    n: int,
    seed: int,
    policy_a: Policy,
    policy_b: Policy,
    blind: float,
    hands: int = 4,
    discards: int = 3,
    levels: Optional[Levels] = None,
    hand_size: int = 8,
) -> "PairedResult":
    """Delta P(clear the blind) under CRN -- the headline estimand:
    X = 1[sum of up to `hands` plays reaches `blind`]. Arm means equal
    run_blinds.p_clear exactly at shared seeds."""
    if n < 2:
        raise ValueError("need at least two trials for a sample SE")
    sum_a = sum_b = sum_d = sum_d2 = 0.0
    flips_up = flips_down = 0
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        x_a = 1.0 if play_blind(shuffled, policy_a, hands, discards, levels, blind, hand_size).cleared else 0.0
        x_b = 1.0 if play_blind(shuffled, policy_b, hands, discards, levels, blind, hand_size).cleared else 0.0
        d = x_b - x_a
        sum_a += x_a
        sum_b += x_b
        sum_d += d
        sum_d2 += d * d
        if d > 0:
            flips_up += 1
        elif d < 0:
            flips_down += 1
    delta = sum_d / n
    var_d = max(sum_d2 - n * delta * delta, 0.0) / (n - 1)
    return PairedResult(
        n=n,
        seed=seed,
        name_a=policy_a.name,
        name_b=policy_b.name,
        discards=discards,
        p_a=sum_a / n,
        p_b=sum_b / n,
        delta=delta,
        se=sqrt(var_d / n),
        flips_up=flips_up,
        flips_down=flips_down,
    )


@dataclass
class PairedResult:
    n: int
    seed: int
    name_a: str
    name_b: str
    discards: int
    p_a: float
    p_b: float
    delta: float
    se: float
    flips_up: int
    flips_down: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.delta - 1.96 * self.se, self.delta + 1.96 * self.se)


def paired_experiment(
    deck: Sequence[Card],
    n: int,
    seed: int,
    policy_a: Policy,
    policy_b: Policy,
    discards: int,
    statistic: Statistic,
    levels: Optional[Levels] = None,
    hand_size: int = 8,
) -> PairedResult:
    """Estimate delta = E[stat under B] - E[stat under A] with CRN pairing:
    trial i shuffles once and feeds the identical deck order through both
    arms."""
    if n < 2:
        raise ValueError("need at least two trials for a sample SE")
    sum_a = sum_b = sum_d = sum_d2 = 0.0
    flips_up = flips_down = 0
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        x_a = float(statistic(_play(play_out(shuffled, policy_a, discards, hand_size), levels)))
        x_b = float(statistic(_play(play_out(shuffled, policy_b, discards, hand_size), levels)))
        d = x_b - x_a
        sum_a += x_a
        sum_b += x_b
        sum_d += d
        sum_d2 += d * d
        if d > 0:
            flips_up += 1
        elif d < 0:
            flips_down += 1
    delta = sum_d / n
    var_d = max(sum_d2 - n * delta * delta, 0.0) / (n - 1)
    return PairedResult(
        n=n,
        seed=seed,
        name_a=policy_a.name,
        name_b=policy_b.name,
        discards=discards,
        p_a=sum_a / n,
        p_b=sum_b / n,
        delta=delta,
        se=sqrt(var_d / n),
        flips_up=flips_up,
        flips_down=flips_down,
    )
