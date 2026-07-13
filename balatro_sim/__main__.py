"""Phase 1 CLI: best-hand-type distribution for 8 dealt cards, no discards.

    python -m balatro_sim --trials 100000 --seed 42

Output is ASCII-only (Windows console safe). Every run is self-validating:
it reports per-trial cross-check mismatches (must be 0) and z-scores of
the Monte Carlo estimates against the exact math in exact.py.
"""
from __future__ import annotations

import argparse
import time
from math import sqrt

from . import exact
from .cards import vanilla_deck
from .evaluator import HandType
from .simulate import run_phase1

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="balatro_sim",
        description="Phase 1: distribution of the best playable hand type "
        "in 8 dealt cards (vanilla deck, no discards).",
    )
    ap.add_argument("--trials", type=int, default=20_000, help="default 20000")
    ap.add_argument("--seed", type=int, default=42, help="default 42")
    args = ap.parse_args(argv)

    deck = vanilla_deck()
    print("Balatro hand-outcome simulator -- Phase 1")
    print(f"vanilla deck ({len(deck)} cards), deal 8, no discards, best playable hand type")
    print(f"trials={args.trials:,}  seed={args.seed}")
    t0 = time.perf_counter()
    report = run_phase1(
        deck,
        args.trials,
        args.seed,
        progress=lambda i: print(f"  ... {i:,}/{args.trials:,}", flush=True),
    )
    dt = time.perf_counter() - t0
    print(f"done in {dt:.1f}s ({args.trials / dt:,.0f} trials/s)")
    print()

    n = report.n
    # Rows of the best-type distribution that reduce to hand-derivable math:
    # best exactly Royal needs royal available; best exactly Straight Flush
    # is (SF available) minus (royal available); High Card is derived directly.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
