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

Phase 2 statistics are indicators over the best hand type (at_least);
Phase 3 will swap in score-threshold indicators without changing the
estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Sequence

from .cards import Card
from .evaluator import HandType, best_of
from .policy import Policy
from .simulate import play_out, trial_rng

Statistic = Callable[[HandType], float]


def at_least(t: HandType) -> Statistic:
    """Indicator: best playable hand type is t or better."""
    return lambda best: 1.0 if best >= t else 0.0


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


def paired_samples(
    deck: Sequence[Card],
    n: int,
    seed: int,
    policy_a: Policy,
    policy_b: Policy,
    discards: int,
    statistic: Statistic,
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
        x_a = float(statistic(best_of(play_out(shuffled, policy_a, discards))[0]))
        x_b = float(statistic(best_of(play_out(shuffled, policy_b, discards))[0]))
        out.append((x_a, x_b))
    return out


def paired_experiment(
    deck: Sequence[Card],
    n: int,
    seed: int,
    policy_a: Policy,
    policy_b: Policy,
    discards: int,
    statistic: Statistic,
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
        x_a = float(statistic(best_of(play_out(shuffled, policy_a, discards))[0]))
        x_b = float(statistic(best_of(play_out(shuffled, policy_b, discards))[0]))
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
