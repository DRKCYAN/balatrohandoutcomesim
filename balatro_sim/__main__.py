"""CLI. Two commands:

Distribution (default):
    python -m balatro_sim --trials 100000 --seed 42
    python -m balatro_sim --policy flushchaser --discards 3 --trials 20000

Paired comparison (common random numbers):
    python -m balatro_sim compare --a none --b flushchaser --stat flush \
        --discards 3 --trials 100000

Output is ASCII-only (Windows console safe). Every distribution run is
self-validating: it reports per-trial cross-check mismatches (must be 0)
and, when the final-hand distribution is provably uniform (policy none
or blind), z-scores against the exact math in exact.py. Policy-shaped
distributions (madehand/flushchaser) have no closed form, so those
columns are omitted rather than faked.
"""
from __future__ import annotations

import argparse
import sys
import time
from math import sqrt

from . import exact
from .cards import vanilla_deck
from .evaluator import HandType
from .experiment import at_least, paired_experiment
from .policy import POLICY_NAMES, get_policy
from .simulate import run_distribution

_AVAIL_ORDER = (
    "pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
    "royal_flush",
)

# Policies whose final hand is still a uniform 8-subset of the deck, so
# the Phase 1 exact math applies (see BlindDiscard's docstring).
_UNIFORM_POLICIES = frozenset({"none", "blind"})

_STAT_TYPES = {
    "pair": HandType.PAIR,
    "two_pair": HandType.TWO_PAIR,
    "three": HandType.THREE_OF_A_KIND,
    "straight": HandType.STRAIGHT,
    "flush": HandType.FLUSH,
    "full_house": HandType.FULL_HOUSE,
    "four": HandType.FOUR_OF_A_KIND,
    "straight_flush": HandType.STRAIGHT_FLUSH,
    "royal": HandType.ROYAL_FLUSH,
}


def _cmd_dist(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="balatro_sim",
        description="Distribution of the best playable hand type in the "
        "final 8 cards (vanilla deck).",
    )
    ap.add_argument("--trials", type=int, default=20_000, help="default 20000")
    ap.add_argument("--seed", type=int, default=42, help="default 42")
    ap.add_argument("--policy", choices=POLICY_NAMES, default="none")
    ap.add_argument("--discards", type=int, default=3,
                    help="discards available to the policy (default 3)")
    args = ap.parse_args(argv)

    deck = vanilla_deck()
    policy = get_policy(args.policy)
    uniform = args.policy in _UNIFORM_POLICIES or args.discards == 0
    if args.policy == "none":
        print("Balatro hand-outcome simulator -- Phase 1")
        print(f"vanilla deck ({len(deck)} cards), deal 8, no discards, best playable hand type")
    else:
        print("Balatro hand-outcome simulator -- distribution under a discard policy")
        print(
            f"vanilla deck ({len(deck)} cards), deal 8, policy={args.policy}, "
            f"discards={args.discards}, best playable hand type"
        )
    print(f"trials={args.trials:,}  seed={args.seed}")
    t0 = time.perf_counter()
    report = run_distribution(
        deck,
        args.trials,
        args.seed,
        policy=policy,
        discards=args.discards,
        progress=lambda i: print(f"  ... {i:,}/{args.trials:,}", flush=True),
    )
    dt = time.perf_counter() - t0
    print(f"done in {dt:.1f}s ({args.trials / dt:,.0f} trials/s)")
    print()

    n = report.n
    exact_best: dict[HandType, object] = {}
    if uniform:
        # Rows of the best-type distribution that reduce to hand-derivable
        # math: best exactly Royal needs royal available; best exactly
        # Straight Flush is (SF available) minus (royal available); High
        # Card is derived directly.
        royal = exact.royal_flush_available()
        exact_best = {
            HandType.ROYAL_FLUSH: royal,
            HandType.STRAIGHT_FLUSH: exact.straight_flush_available() - royal,
            HandType.HIGH_CARD: exact.best_is_high_card(),
        }
    pmax = max(c / n for c in report.best_counts.values())
    print("best hand type        count     p_hat        se      exact        z")
    for t in sorted(HandType, reverse=True):
        c = report.best_counts.get(t, 0)
        if c == 0 and t not in exact_best:
            continue
        p = c / n
        se_hat = sqrt(p * (1 - p) / n)
        if t in exact_best:
            pe = float(exact_best[t])
            z = (p - pe) / sqrt(pe * (1 - pe) / n)
            tail = f"{pe:9.5f}  {z:+7.1f}"
        else:
            tail = f"{'-':>9}  {'-':>7}"
        bar = "#" * round(30 * p / pmax)
        print(f"{t.display:<18} {c:>9}  {p:8.5f}  {se_hat:8.5f}  {tail}  {bar}")
    print()
    print("cross-validation")
    print(f"  best_of() vs availability-floor mismatches: {report.inconsistencies} / {n:,}")
    print()
    if uniform:
        print("  availability of hand classes among the 8 dealt, vs exact math:")
        print("    class                  p_hat      exact        z")
        av_exact = exact.availability_exact()
        for key in _AVAIL_ORDER:
            p = report.avail_counts.get(key, 0) / n
            pe = float(av_exact[key])
            z = (p - pe) / sqrt(pe * (1 - pe) / n)
            print(f"    {key:<18} {p:9.5f}  {pe:9.5f}  {z:+7.1f}")
        print()
        print("  expected signature of a correct simulator: 0 mismatches, all |z| <~ 3")
    else:
        print("  availability of hand classes in the final 8 (policy-shaped;")
        print("  no closed form exists once the policy reacts to what it sees):")
        print("    class                  p_hat")
        for key in _AVAIL_ORDER:
            p = report.avail_counts.get(key, 0) / n
            print(f"    {key:<18} {p:9.5f}")
        print()
        print("  expected signature of a correct simulator: 0 mismatches")
    return 0


def _cmd_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="balatro_sim compare",
        description="Paired policy comparison with common random numbers.",
    )
    ap.add_argument("--a", choices=POLICY_NAMES, required=True, help="baseline arm")
    ap.add_argument("--b", choices=POLICY_NAMES, required=True, help="treatment arm")
    ap.add_argument("--stat", choices=sorted(_STAT_TYPES), default="flush",
                    help="statistic: P(best >= this hand type); default flush")
    ap.add_argument("--trials", type=int, default=20_000, help="default 20000")
    ap.add_argument("--seed", type=int, default=42, help="default 42")
    ap.add_argument("--discards", type=int, default=3, help="default 3")
    args = ap.parse_args(argv)

    deck = vanilla_deck()
    target = _STAT_TYPES[args.stat]
    print("Balatro hand-outcome simulator -- paired comparison (CRN)")
    print(
        f"vanilla deck, discards={args.discards}, "
        f"stat = P(best >= {target.display}), trials={args.trials:,}, seed={args.seed}"
    )
    t0 = time.perf_counter()
    res = paired_experiment(
        deck,
        args.trials,
        args.seed,
        policy_a=get_policy(args.a),
        policy_b=get_policy(args.b),
        discards=args.discards,
        statistic=at_least(target),
    )
    dt = time.perf_counter() - t0
    print(f"done in {dt:.1f}s ({args.trials / dt:,.0f} trials/s)")
    print()
    lo, hi = res.ci95
    print(f"  arm A  {res.name_a:<12} p_A = {res.p_a:.5f}")
    print(f"  arm B  {res.name_b:<12} p_B = {res.p_b:.5f}")
    print(f"  delta (B - A) = {res.delta:+.5f} +/- {res.se:.5f} (SE)")
    print(f"  95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(
        f"  flips: B-only clears {res.flips_up:,}, A-only clears {res.flips_down:,} "
        f"(of {res.n:,} paired trials; the signal lives in the flips)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] == "compare":
        return _cmd_compare(argv[1:])
    return _cmd_dist(argv)


if __name__ == "__main__":
    raise SystemExit(main())
