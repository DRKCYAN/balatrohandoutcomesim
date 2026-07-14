"""Paired comparison with common random numbers -- PLAN.md section 5.

Both arms of a comparison replay the *same shuffle object* per trial, so
they stay in lockstep until the configurations actually diverge. Garbage
shuffles fail in both arms, great shuffles clear in both; the difference
D_i is nonzero only on the marginal trials one arm flips. That collapses
Var(delta_hat) by the 2*Cov term and is the reason small effects are
detectable at all.

Sign convention: delta = p_b - p_a ("B minus A"; positive means arm B is
better on the statistic). PLAN.md's Delta = p_with - p_without maps to
a = without, b = with.

Statistics are functions of a PlayResult (the played hand's type and,
when levels were supplied, its score) -- the PLAN.md section 10 shape:
joker value will be a delta on P(S >= B) through this same estimator.

  - at_least(t): indicator on the hand type.
  - score_at_least(B): indicator on the score; requires levels=.

Arm semantics: without levels, each arm plays the type-max best_of hand
(Phase 1/2 behaviour, coherence-pinned to run_distribution's counts).
With levels, each arm plays the score-max best_play hand -- the optimal
greedy player -- matching run_distribution's scores exactly.
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
) -> list[tuple[float, float]]:
    """Per-trial (x_a, x_b) pairs under CRN -- the raw material for
    visualisations (flip grids, convergence). Same trial streams as
    paired_experiment, so summaries computed from this list must equal
    its estimates exactly (pinned by tests). Materialises n pairs; for
    plain estimation at large n use paired_experiment, which streams.
    """
    out: list[tuple[float, float]] = []
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        x_a = float(statistic(_play(play_out(shuffled, policy_a, discards), levels)))
        x_b = float(statistic(_play(play_out(shuffled, policy_b, discards), levels)))
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
) -> "PairedResult":
    """Delta P(clear the blind) under CRN -- the project's headline
    estimand (PLAN.md sections 1-2): X = 1[sum of up to `hands` plays
    from the shared shuffle reaches `blind`]. Same pairing guarantees as
    paired_experiment; arm means equal run_blinds.p_clear exactly at
    shared seeds (pinned by tests)."""
    if n < 2:
        raise ValueError("need at least two trials for a sample SE")
    sum_a = sum_b = sum_d = sum_d2 = 0.0
    flips_up = flips_down = 0
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        x_a = 1.0 if play_blind(shuffled, policy_a, hands, discards, levels, blind).cleared else 0.0
        x_b = 1.0 if play_blind(shuffled, policy_b, hands, discards, levels, blind).cleared else 0.0
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
    delta: float  # mean of D_i = X_b - X_a
    se: float  # sample SD of D_i / sqrt(n)
    flips_up: int  # trials with D_i > 0 (B succeeded where A failed)
    flips_down: int  # trials with D_i < 0

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
) -> PairedResult:
    """Estimate delta = E[stat under B] - E[stat under A] with CRN pairing.

    Trial i shuffles once via trial_rng(seed, i) and feeds the identical
    deck order through both arms (play_out does not mutate it).
    """
    if n < 2:
        raise ValueError("need at least two trials for a sample SE")
    sum_a = sum_b = sum_d = sum_d2 = 0.0
    flips_up = flips_down = 0
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        x_a = float(statistic(_play(play_out(shuffled, policy_a, discards), levels)))
        x_b = float(statistic(_play(play_out(shuffled, policy_b, discards), levels)))
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
