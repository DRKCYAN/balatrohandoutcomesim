"""matplotlib figures for simulation results (PNG, light surface).

matplotlib is the project's only optional dependency: it is imported
inside functions, so the core simulator and the default test run stay
stdlib-only. Install with: pip install matplotlib

Colour discipline (validated with the dataviz palette tooling; worst
adjacent CVD delta-E 47.2):

  - Policies are entities, so each policy owns a fixed categorical slot
    (_POLICY_COLORS) used identically across every chart -- colour
    follows the entity, never its position in this particular figure.
  - The aqua/yellow slots sit below 3:1 contrast on the light surface;
    the required relief is a table view / visible labels, provided by
    the CLI's printed tables (same numbers as every chart) plus legends
    and direct end-labels on the curves.
  - The flip grid encodes *state*, not identity, so it uses the reserved
    status colours (good/critical) plus neutral, with counts written
    into the legend labels -- never colour alone.
"""
from __future__ import annotations

from math import ceil, sqrt
from typing import Optional, Sequence

from .cards import Card, vanilla_deck
from .evaluator import HandType, best_of
from .experiment import at_least, paired_samples
from .policy import Policy
from .simulate import DistributionReport, play_out, run_distribution, trial_rng

# categorical slots (light mode) -- fixed per policy, never cycled
_POLICY_COLORS = {
    "none": "#2a78d6",        # blue
    "madehand": "#1baf7a",    # aqua
    "flushchaser": "#eda100", # yellow
    "blind": "#4a3aa7",       # violet
}
_FALLBACK_COLOR = "#898781"

# chart chrome (light)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"

# status colours (reserved; flip grid only)
_GOOD = "#0ca30c"
_CRITICAL = "#d03b3b"


def _plt():
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError(
            "charts need matplotlib (the project's only optional dependency): "
            "pip install matplotlib"
        ) from None
    matplotlib.use("Agg")  # file output only; no display needed
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans"]
    return plt


def _policy_color(name: str) -> str:
    return _POLICY_COLORS.get(name, _FALLBACK_COLOR)


def _style(fig, ax, grid_axis: str) -> None:
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelcolor=_INK2, labelsize=9)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _finish(fig, ax, title: str, subtitle: str, out_path: str) -> None:
    ax.set_title(title, loc="left", color=_INK, fontsize=12, fontweight="semibold", pad=16)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=_INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=_SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _tail_p(report: DistributionReport, target: HandType) -> float:
    return sum(c for t, c in report.best_counts.items() if t >= target) / report.n


def distribution_chart(reports: Sequence[DistributionReport], out_path: str) -> None:
    """Best-hand-type distribution as grouped horizontal bars with 95% CI
    whiskers, one fixed-colour series per policy."""
    plt = _plt()
    shown = [
        t for t in sorted(HandType, reverse=True)
        if any(r.best_counts.get(t, 0) for r in reports)
    ]
    fig, ax = plt.subplots(figsize=(7.5, 0.52 * len(shown) + 1.8))
    _style(fig, ax, grid_axis="x")
    k = len(reports)
    height = 0.78 / k
    for j, r in enumerate(reports):
        ys, ps, errs = [], [], []
        for row, t in enumerate(shown):
            p = r.p_best(t)
            ys.append(row - 0.39 + height * (j + 0.5))
            ps.append(p)
            errs.append(1.96 * sqrt(p * (1 - p) / r.n))
        ax.barh(
            ys, ps, height=height * 0.92,
            color=_policy_color(r.policy_name),
            edgecolor=_SURFACE, linewidth=0.6,
            label=f"{r.policy_name} (d={r.discards})",
        )
        ax.errorbar(ps, ys, xerr=errs, fmt="none", ecolor=_INK2, elinewidth=1, capsize=2)
    ax.set_yticks(range(len(shown)), [t.display for t in shown])
    ax.invert_yaxis()
    ax.set_xlabel("P(best playable hand type)", color=_INK2, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=_INK2, loc="lower right")
    for h in leg.legend_handles:
        h.set_edgecolor("none")
    n = ", ".join(f"{r.n:,}" for r in reports)
    _finish(fig, ax, "Best playable hand type by policy",
            f"vanilla deck, whiskers = 95% CI, n = {n}, seed = {reports[0].seed}",
            out_path)


def convergence_chart(
    policy: Policy,
    discards: int,
    n: int,
    seed: int,
    target: HandType,
    out_path: str,
    deck: Optional[Sequence[Card]] = None,
) -> None:
    """Cumulative p_hat of best >= target vs trial count (log x) with its
    shrinking 95% band -- the picture of SE ~ 1/sqrt(n)."""
    plt = _plt()
    if n < 20:
        raise ValueError("convergence needs at least 20 trials")
    if deck is None:
        deck = vanilla_deck()
    stat = at_least(target)
    hits = 0
    xs, ps, los, his = [], [], [], []
    checkpoints = sorted({max(20, round(20 * (n / 20) ** (k / 239))) for k in range(240)})
    nexts = iter(checkpoints)
    nxt = next(nexts)
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        hits += stat(best_of(play_out(shuffled, policy, discards))[0])
        if i + 1 == nxt:
            p = hits / (i + 1)
            half = 1.96 * sqrt(p * (1 - p) / (i + 1))
            xs.append(i + 1)
            ps.append(p)
            los.append(p - half)
            his.append(p + half)
            nxt = next(nexts, n + 1)
    color = _policy_color(policy.name)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _style(fig, ax, grid_axis="y")
    ax.fill_between(xs, los, his, color=color, alpha=0.16, linewidth=0)
    ax.plot(xs, ps, color=color, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("trials (log scale)", color=_INK2, fontsize=9)
    ax.set_ylabel(f"P(best ≥ {target.display})", color=_INK2, fontsize=9)
    _finish(fig, ax, f"Convergence — {policy.name}, {discards} discards",
            f"cumulative estimate with 95% band, final p = {ps[-1]:.4f}, "
            f"n = {n:,}, seed = {seed}", out_path)


def discards_curve(
    policies: Sequence[Policy],
    max_discards: int,
    n: int,
    seed: int,
    target: HandType,
    out_path: str,
    deck: Optional[Sequence[Card]] = None,
) -> None:
    """P(best >= target) vs discard count, one fixed-colour line per
    policy, 95% CI whiskers, direct end labels."""
    plt = _plt()
    if deck is None:
        deck = vanilla_deck()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _style(fig, ax, grid_axis="y")
    for policy in policies:
        color = _policy_color(policy.name)
        ds = list(range(max_discards + 1))
        ps, errs = [], []
        for d in ds:
            rep = run_distribution(deck, n, seed, policy=policy, discards=d)
            p = _tail_p(rep, target)
            ps.append(p)
            errs.append(1.96 * sqrt(p * (1 - p) / n))
        ax.errorbar(
            ds, ps, yerr=errs, color=color, linewidth=2,
            marker="o", markersize=7, capsize=3, label=policy.name,
        )
        ax.annotate(
            policy.name, (ds[-1], ps[-1]), xytext=(8, 0),
            textcoords="offset points", color=_INK2, fontsize=9, va="center",
        )
    ax.set_xticks(range(max_discards + 1))
    ax.set_xlim(-0.25, max_discards + 0.9)
    ax.set_xlabel("discards available", color=_INK2, fontsize=9)
    ax.set_ylabel(f"P(best ≥ {target.display})", color=_INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK2, loc="upper left")
    _finish(fig, ax, f"What a discard is worth — P(best ≥ {target.display})",
            f"vanilla deck, whiskers = 95% CI, n = {n:,} per point, seed = {seed}",
            out_path)


def flip_grid(
    policy_a: Policy,
    policy_b: Policy,
    discards: int,
    n: int,
    seed: int,
    target: HandType,
    out_path: str,
    deck: Optional[Sequence[Card]] = None,
) -> None:
    """The CRN picture: n paired trials as a grid (row-major from top
    left). Neutral = both arms agree, green = only B clears, red = only
    A clears. The estimator's whole signal is the green/red imbalance."""
    plt = _plt()
    import numpy as np

    if deck is None:
        deck = vanilla_deck()
    samples = paired_samples(deck, n, seed, policy_a, policy_b, discards, at_least(target))
    ds = [b - a for a, b in samples]
    ups = sum(1 for d in ds if d > 0)
    downs = sum(1 for d in ds if d < 0)
    same = n - ups - downs
    delta = sum(ds) / n
    var_d = max(sum(d * d for d in ds) - n * delta * delta, 0.0) / (n - 1)
    se = sqrt(var_d / n)

    side = ceil(sqrt(n))
    cells = np.full(side * side, 3, dtype=int)  # 3 = padding (surface)
    for i, d in enumerate(ds):
        cells[i] = 0 if d == 0 else (1 if d > 0 else 2)
    grid = cells.reshape(side, side)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap([_GRID, _GOOD, _CRITICAL, _SURFACE])
    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    _style(fig, ax, grid_axis="none")
    ax.pcolormesh(grid, cmap=cmap, vmin=0, vmax=3,
                  edgecolors=_SURFACE, linewidth=0.6)
    ax.invert_yaxis()  # trial 0 at top left, reading order
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=_GRID, label=f"arms agree ({same:,})"),
            Patch(facecolor=_GOOD, label=f"only {policy_b.name} clears ({ups:,})"),
            Patch(facecolor=_CRITICAL, label=f"only {policy_a.name} clears ({downs:,})"),
        ],
        frameon=False, fontsize=9, labelcolor=_INK2,
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
    )
    _finish(
        fig, ax,
        f"Common random numbers — {policy_a.name} vs {policy_b.name}",
        f"P(best ≥ {target.display}), {discards} discards; one cell per paired "
        f"trial, row-major; Δ̂ = {delta:+.4f} ± {se:.4f} (SE), n = {n:,}, seed = {seed}",
        out_path,
    )
