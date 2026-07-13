"""CLI. Four commands:

Distribution (default):
    python -m balatro_sim --trials 100000 --seed 42
    python -m balatro_sim --policy flushchaser --discards 3 --trials 20000

Paired comparison (common random numbers):
    python -m balatro_sim compare --a none --b flushchaser --stat flush \
        --discards 3 --trials 100000

Trial replay (self-contained HTML, open in any browser):
    python -m balatro_sim trace --policy madehand --discards 3 --trials 12 \
        --out trace.html

Charts (PNG; needs matplotlib, the only optional dependency):
    python -m balatro_sim plot dist --policies none madehand flushchaser
    python -m balatro_sim plot converge --policy flushchaser --stat flush
    python -m balatro_sim plot discards --policies madehand flushchaser
    python -m balatro_sim plot flips --a none --b flushchaser --stat flush

Terminal output is ASCII-only (Windows console safe). Every distribution
run is self-validating: it reports per-trial cross-check mismatches
(must be 0) and, when the final-hand distribution is provably uniform
(policy none or blind), z-scores against the exact math in exact.py.
Policy-shaped distributions (madehand/flushchaser) have no closed form,
so those columns are omitted rather than faked.
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


def _cmd_trace(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="balatro_sim trace",
        description="Replay trials as a self-contained HTML file.",
    )
    ap.add_argument("--policy", choices=POLICY_NAMES, default="madehand")
    ap.add_argument("--discards", type=int, default=3, help="default 3")
    ap.add_argument("--trials", type=int, default=12,
                    help="how many trials to replay (default 12)")
    ap.add_argument("--seed", type=int, default=42, help="default 42")
    ap.add_argument("--out", default="trace.html", help="default trace.html")
    args = ap.parse_args(argv)

    from .trace import render_trace_html

    render_trace_html(
        vanilla_deck(), args.seed, args.trials, get_policy(args.policy),
        args.discards, args.out,
    )
    print(f"wrote {args.out}: trials 0-{args.trials - 1}, policy={args.policy}, "
          f"discards={args.discards}, seed={args.seed}")
    print("open it in a browser; these are the exact trials the statistics count.")
    return 0


def _cmd_plot(argv: list[str]) -> int:
    kinds = ("dist", "converge", "discards", "flips")
    if not argv or argv[0] not in kinds:
        print(f"usage: python -m balatro_sim plot {{{','.join(kinds)}}} [options]")
        return 2
    kind, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"balatro_sim plot {kind}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--discards", type=int, default=3)
    ap.add_argument("--stat", choices=sorted(_STAT_TYPES), default="flush")
    ap.add_argument("--out", default=f"{kind}.png")
    if kind == "dist":
        ap.add_argument("--policies", nargs="+", choices=POLICY_NAMES,
                        default=["none", "madehand", "flushchaser"])
        ap.add_argument("--trials", type=int, default=20_000)
    elif kind == "converge":
        ap.add_argument("--policy", choices=POLICY_NAMES, default="flushchaser")
        ap.add_argument("--trials", type=int, default=20_000)
    elif kind == "discards":
        ap.add_argument("--policies", nargs="+", choices=POLICY_NAMES,
                        default=["madehand", "flushchaser"])
        ap.add_argument("--max-discards", type=int, default=3)
        ap.add_argument("--trials", type=int, default=4_000,
                        help="per point (default 4000)")
    else:  # flips
        ap.add_argument("--a", choices=POLICY_NAMES, required=True)
        ap.add_argument("--b", choices=POLICY_NAMES, required=True)
        ap.add_argument("--trials", type=int, default=2_500)
    args = ap.parse_args(rest)

    from . import charts

    deck = vanilla_deck()
    target = _STAT_TYPES[args.stat]
    print(f"plotting {kind} -> {args.out} ...", flush=True)
    if kind == "dist":
        reports = []
        for name in args.policies:
            print(f"  simulating {name} (n={args.trials:,}) ...", flush=True)
            reports.append(run_distribution(
                deck, args.trials, args.seed,
                policy=get_policy(name), discards=args.discards,
            ))
        charts.distribution_chart(reports, args.out)
    elif kind == "converge":
        charts.convergence_chart(
            get_policy(args.policy), args.discards, args.trials, args.seed,
            target, args.out, deck=deck,
        )
    elif kind == "discards":
        charts.discards_curve(
            [get_policy(p) for p in args.policies], args.max_discards,
            args.trials, args.seed, target, args.out, deck=deck,
        )
    else:
        charts.flip_grid(
            get_policy(args.a), get_policy(args.b), args.discards,
            args.trials, args.seed, target, args.out, deck=deck,
        )
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] == "compare":
        return _cmd_compare(argv[1:])
    if argv and argv[0] == "trace":
        return _cmd_trace(argv[1:])
    if argv and argv[0] == "plot":
        return _cmd_plot(argv[1:])
    return _cmd_dist(argv)


if __name__ == "__main__":
    raise SystemExit(main())
