"""CLI. Five commands:

Distribution (default; --blind/--level add the score section):
    python -m balatro_sim --trials 100000 --seed 42
    python -m balatro_sim --policy flushchaser --discards 3 --blind 600

Score a specific hand (make sure u check ts on your xbox):
    python -m balatro_sim score "KS KH 7D 2C 3H" --level pair=2

Full-blind trials -- the real clearing condition (4 hands, 3 shared
discards, continuing deck); compare --stat clear is the paired version:
    python -m balatro_sim blind --policy flushchaser --blind 600
    python -m balatro_sim compare --a madehand --b flushchaser --stat clear --blind 600

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
    python -m balatro_sim plot cdf --policies none flushchaser --blind 600

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
from .cards import CHIP_VALUE, hand as parse_hand, vanilla_deck
from .evaluator import HandType, evaluate
from .experiment import (
    at_least,
    paired_blind_experiment,
    paired_experiment,
    score_at_least,
)
from .policy import POLICY_NAMES, get_policy
from .scoring import best_play, effective_level, hand_base_at, scoring_cards
from .simulate import run_blinds, run_distribution

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

# names accepted by --level TYPE=N (royal shares straight_flush's level)
_LEVEL_TYPES = {
    "high_card": HandType.HIGH_CARD,
    **{k: v for k, v in _STAT_TYPES.items() if k != "royal"},
    "five": HandType.FIVE_OF_A_KIND,
    "flush_house": HandType.FLUSH_HOUSE,
    "flush_five": HandType.FLUSH_FIVE,
}


def _parse_levels(items: list[str] | None) -> dict[HandType, int]:
    """--level pair=2 --level flush=3 -> {PAIR: 2, FLUSH: 3}. Unset types
    are level 1. An empty dict is a valid 'all level 1' scoring config."""
    levels: dict[HandType, int] = {}
    for item in items or []:
        name, _, val = item.partition("=")
        if name == "royal":
            raise SystemExit(
                "royal shares straight_flush's level (one planet, Neptune); "
                "use --level straight_flush=N"
            )
        if name not in _LEVEL_TYPES:
            raise SystemExit(
                f"unknown hand type in --level {item!r}; "
                f"choose from {', '.join(sorted(_LEVEL_TYPES))}"
            )
        if not val.isdigit() or int(val) < 1:
            raise SystemExit(f"--level {item!r}: level must be an integer >= 1")
        levels[_LEVEL_TYPES[name]] = int(val)
    return levels


def _levels_label(levels: dict[HandType, int]) -> str:
    if not levels:
        return "all hands level 1"
    parts = ", ".join(f"{t.display} L{v}" for t, v in sorted(levels.items()))
    return f"{parts}; rest level 1"


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
    ap.add_argument("--blind", type=float, default=None,
                    help="blind requirement: adds the score section with P(S >= blind)")
    ap.add_argument("--level", action="append", metavar="TYPE=N",
                    help="hand level, repeatable (e.g. --level pair=2); implies scoring")
    args = ap.parse_args(argv)

    levels = _parse_levels(args.level)
    scored = args.blind is not None or bool(levels)
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
        levels=levels if scored else None,
    )
    dt = time.perf_counter() - t0
    print(f"done in {dt:.1f}s ({args.trials / dt:,.0f} trials/s)")
    print()

    if scored:
        qs = [(1, "p1"), (5, "p5"), (25, "p25"), (50, "median"),
              (75, "p75"), (95, "p95"), (99, "p99")]
        print(f"score of best play per trial ({_levels_label(levels)})")
        row = f"  min {min(report.scores)}"
        for q, name in qs:
            row += f"   {name} {report.score_percentile(q)}"
        row += f"   max {max(report.scores)}"
        print(row)
        if args.blind is not None:
            p = report.p_score_at_least(args.blind)
            se_p = sqrt(p * (1 - p) / report.n)
            print(f"  P(S >= {args.blind:g}) = {p:.5f} +/- {se_p:.5f} (SE)")
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
    ap.add_argument("--stat", choices=sorted(_STAT_TYPES) + ["score", "clear"],
                    default="flush",
                    help="P(best >= hand type); 'score' = single-hand "
                    "P(S >= --blind); 'clear' = full-blind P(total >= --blind) "
                    "over --hands hands (the real clearing condition). "
                    "default flush")
    ap.add_argument("--blind", type=float, default=None,
                    help="required with --stat score/clear")
    ap.add_argument("--hands", type=int, default=4,
                    help="hands per blind for --stat clear (default 4)")
    ap.add_argument("--level", action="append", metavar="TYPE=N",
                    help="hand level for score/clear stats, repeatable")
    ap.add_argument("--trials", type=int, default=20_000, help="default 20000")
    ap.add_argument("--seed", type=int, default=42, help="default 42")
    ap.add_argument("--discards", type=int, default=3, help="default 3")
    args = ap.parse_args(argv)

    deck = vanilla_deck()
    print("Balatro hand-outcome simulator -- paired comparison (CRN)")
    t0 = time.perf_counter()
    if args.stat == "clear":
        if args.blind is None:
            raise SystemExit("--stat clear needs --blind B")
        levels = _parse_levels(args.level)
        print(
            f"vanilla deck, stat = P(clear {args.blind:g} in {args.hands} hands, "
            f"{args.discards} shared discards) ({_levels_label(levels)}), "
            f"trials={args.trials:,}, seed={args.seed}"
        )
        res = paired_blind_experiment(
            deck, args.trials, args.seed,
            policy_a=get_policy(args.a), policy_b=get_policy(args.b),
            blind=args.blind, hands=args.hands, discards=args.discards,
            levels=levels,
        )
    else:
        if args.stat == "score":
            if args.blind is None:
                raise SystemExit("--stat score needs --blind B")
            levels = _parse_levels(args.level)
            statistic = score_at_least(args.blind)
            stat_desc = f"single-hand P(S >= {args.blind:g}) ({_levels_label(levels)})"
        else:
            target = _STAT_TYPES[args.stat]
            levels = None
            statistic = at_least(target)
            stat_desc = f"P(best >= {target.display})"
        print(
            f"vanilla deck, discards={args.discards}, "
            f"stat = {stat_desc}, trials={args.trials:,}, seed={args.seed}"
        )
        res = paired_experiment(
            deck,
            args.trials,
            args.seed,
            policy_a=get_policy(args.a),
            policy_b=get_policy(args.b),
            discards=args.discards,
            statistic=statistic,
            levels=levels,
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


def _cmd_blind(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="balatro_sim blind",
        description="Full-blind trials: up to --hands plays from a "
        "continuing deck, one shared --discards budget -- the real "
        "clearing condition, unlike the single-hand distribution.",
    )
    ap.add_argument("--policy", choices=POLICY_NAMES, default="none")
    ap.add_argument("--hands", type=int, default=4, help="hands per blind (default 4)")
    ap.add_argument("--discards", type=int, default=3,
                    help="shared discard budget per blind (default 3)")
    ap.add_argument("--blind", type=float, default=None,
                    help="chip requirement; omit for the uncensored total distribution")
    ap.add_argument("--level", action="append", metavar="TYPE=N",
                    help="hand level, repeatable (e.g. --level flush=2)")
    ap.add_argument("--trials", type=int, default=20_000, help="default 20000")
    ap.add_argument("--seed", type=int, default=42, help="default 42")
    args = ap.parse_args(argv)

    levels = _parse_levels(args.level)
    deck = vanilla_deck()
    print("Balatro hand-outcome simulator -- blind trials")
    print(
        f"vanilla deck, policy={args.policy}, hands={args.hands}, "
        f"discards={args.discards} (shared), {_levels_label(levels)}"
    )
    target = "none (uncensored totals)" if args.blind is None else f"{args.blind:g}"
    print(f"blind={target}  trials={args.trials:,}  seed={args.seed}")
    t0 = time.perf_counter()
    report = run_blinds(
        deck, args.trials, args.seed,
        policy=get_policy(args.policy), hands=args.hands,
        discards=args.discards, blind=args.blind,
        levels=levels,
        progress=lambda i: print(f"  ... {i:,}/{args.trials:,}", flush=True),
    )
    dt = time.perf_counter() - t0
    print(f"done in {dt:.1f}s ({args.trials / dt:,.0f} blinds/s)")
    print()
    if args.blind is not None:
        p = report.p_clear
        se_p = sqrt(p * (1 - p) / report.n)
        print(f"  P(clear {args.blind:g}) = {p:.5f} +/- {se_p:.5f} (SE)")
        print("  hands needed to clear:")
        for h in sorted(report.hands_used):
            share = report.hands_used[h] / report.n
            print(f"    {h}: {share:8.2%}")
        print(f"    failed: {1 - p:8.2%}")
        print("  (totals are censored at clear; use no --blind for the full distribution)")
    else:
        qs = [(1, "p1"), (5, "p5"), (25, "p25"), (50, "median"),
              (75, "p75"), (95, "p95"), (99, "p99")]
        print(f"  blind total over {args.hands} hands (uncensored)")
        row = f"  min {min(report.totals)}"
        for q, name in qs:
            row += f"   {name} {report.total_percentile(q)}"
        row += f"   max {max(report.totals)}"
        print(row)
    return 0


def _cmd_score(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="balatro_sim score",
        description="Score a hand exactly as the sim would -- the Xbox "
        "validation workhorse. 1-5 cards: scores that exact play. "
        "6-8 cards: shows the best play the sim would choose.",
    )
    ap.add_argument("cards", help='space-separated cards, e.g. "KS KH 7D 2C 3H"')
    ap.add_argument("--level", action="append", metavar="TYPE=N",
                    help="hand level, repeatable (e.g. --level pair=2)")
    args = ap.parse_args(argv)

    levels = _parse_levels(args.level)
    cards = parse_hand(args.cards)
    if not 1 <= len(cards) <= 8:
        raise SystemExit(f"give 1-8 cards, got {len(cards)}")

    def breakdown(played) -> None:
        t = evaluate(played)
        lvl = effective_level(levels, t)
        base_chips, mult = hand_base_at(t, lvl)
        scoring = scoring_cards(t, played)
        card_chips = [CHIP_VALUE[c.rank] for c in scoring]
        total = (base_chips + sum(card_chips)) * mult
        print(f"  hand type:     {t.display} (level {lvl})")
        print(f"  scoring cards: {' '.join(str(c) for c in scoring)}"
              f"  ({' + '.join(map(str, card_chips))} = {sum(card_chips)} card chips)")
        print(f"  formula:       ({base_chips} base + {sum(card_chips)} cards)"
              f" x {mult} mult")
        print(f"  score:         {total}")

    print(f"levels: {_levels_label(levels)}")
    if len(cards) <= 5:
        print(f"played: {' '.join(str(c) for c in cards)}")
        breakdown(cards)
    else:
        print(f"hand:   {' '.join(str(c) for c in cards)}")
        pr = best_play(cards, levels)
        print(f"best play: {' '.join(str(c) for c in pr.played)}")
        breakdown(pr.played)
    return 0


def _cmd_plot(argv: list[str]) -> int:
    kinds = ("dist", "converge", "discards", "flips", "cdf")
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
    elif kind == "flips":
        ap.add_argument("--a", choices=POLICY_NAMES, required=True)
        ap.add_argument("--b", choices=POLICY_NAMES, required=True)
        ap.add_argument("--trials", type=int, default=2_500)
    else:  # cdf
        ap.add_argument("--policies", nargs="+", choices=POLICY_NAMES,
                        default=["none", "madehand", "flushchaser"])
        ap.add_argument("--blind", type=float, default=None)
        ap.add_argument("--level", action="append", metavar="TYPE=N")
        ap.add_argument("--trials", type=int, default=20_000)
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
    elif kind == "flips":
        charts.flip_grid(
            get_policy(args.a), get_policy(args.b), args.discards,
            args.trials, args.seed, target, args.out, deck=deck,
        )
    else:  # cdf
        charts.score_cdf(
            [get_policy(p) for p in args.policies], args.discards,
            args.trials, args.seed, _parse_levels(args.level), args.blind,
            args.out, deck=deck,
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
    if argv and argv[0] == "score":
        return _cmd_score(argv[1:])
    if argv and argv[0] == "blind":
        return _cmd_blind(argv[1:])
    return _cmd_dist(argv)


if __name__ == "__main__":
    raise SystemExit(main())
