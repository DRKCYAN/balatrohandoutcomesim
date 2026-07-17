"""HTML trial replay: writes one self-contained HTML file (inline CSS, no
JS) showing, per trial, the dealt hand, each discard round, and the final
best play. Trials are seeded, so the replay re-runs the exact trial the
statistics counted (via play_out's trace hook).
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Sequence

from .cards import Card, vanilla_deck
from .evaluator import HandType, best_of
from .policy import Policy
from .simulate import DiscardStep, play_out, trial_rng

_SUIT_GLYPHS = "♠♥♦♣"
_RED_SUITS = (1, 2)
_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}


@dataclass(frozen=True)
class TrialReplay:
    index: int
    initial: tuple[Card, ...]
    steps: tuple[DiscardStep, ...]
    final: tuple[Card, ...]
    initial_best: HandType
    final_best: HandType
    best_cards: tuple[Card, ...]


def replay_from_shuffled(
    shuffled: Sequence[Card], index: int, policy: Policy, discards: int,
    hand_size: int = 8,
) -> TrialReplay:
    """Replay one already-shuffled deck order (the testable core)."""
    steps: list[DiscardStep] = []
    final = play_out(shuffled, policy, discards, hand_size, trace=steps)
    initial = tuple(shuffled[:hand_size])
    initial_best, _ = best_of(initial)
    final_best, best_cards = best_of(final)
    return TrialReplay(
        index, initial, tuple(steps), final, initial_best, final_best, best_cards
    )


def replay_trial(
    deck: Sequence[Card], seed: int, i: int, policy: Policy, discards: int,
    hand_size: int = 8,
) -> TrialReplay:
    """Replay trial i of the (deck, seed) stream -- the same trial
    run_distribution counted."""
    shuffled = list(deck)
    trial_rng(seed, i).shuffle(shuffled)
    return replay_from_shuffled(shuffled, i, policy, discards, hand_size)


_CSS = """
:root { color-scheme: light; }
body { margin: 0; padding: 24px; background: #f9f9f7; color: #0b0b0b;
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
.meta { color: #52514e; font-size: 13px; margin-bottom: 20px; }
.trial { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
         border-radius: 10px; padding: 14px 16px 10px; margin-bottom: 16px; }
.trial h2 { font-size: 14px; margin: 0 0 10px; font-weight: 600; }
.trial h2 .arrow { color: #52514e; font-weight: 400; }
.row { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
.lbl { width: 110px; flex: none; color: #898781; font-size: 12px; }
.card { display: inline-flex; flex-direction: column; justify-content: space-between;
        width: 40px; height: 56px; padding: 3px 4px; box-sizing: border-box;
        background: #ffffff; border: 1px solid #c3c2b7; border-radius: 5px;
        font-size: 13px; font-weight: 600; line-height: 1; }
.card .st { align-self: flex-end; font-size: 15px; }
.card.red { color: #d03b3b; }
.card.black { color: #0b0b0b; }
.card.dim { opacity: 0.32; border-style: dashed; }
.card.new { border: 2px solid #2a78d6; padding: 2px 3px; }
.card.best { border: 2px solid #eda100; padding: 2px 3px;
             box-shadow: 0 0 0 2px rgba(237,161,0,0.25); }
.tag { font-size: 12px; color: #52514e; margin-left: 8px; }
.tag b { color: #0b0b0b; }
.legend { color: #898781; font-size: 12px; margin: 14px 0 0; }
.legend .card { width: 22px; height: 30px; font-size: 9px; padding: 2px; }
.foot { color: #898781; font-size: 12px; margin-top: 16px; }
"""


def _card_html(c: Card, classes: str = "") -> str:
    color = "red" if c.suit in _RED_SUITS else "black"
    rank = _RANK_NAMES.get(c.rank, str(c.rank))
    return (
        f'<span class="card {color}{" " + classes if classes else ""}">'
        f'<span class="rk">{rank}</span>'
        f'<span class="st">{_SUIT_GLYPHS[c.suit]}</span></span>'
    )


def _row_html(label: str, cards: Sequence[Card], dim=(), new=(), best=(), tag: str = "") -> str:
    parts = [f'<div class="row"><span class="lbl">{html.escape(label)}</span>']
    for i, c in enumerate(cards):
        classes = []
        if i in dim:
            classes.append("dim")
        if i in new:
            classes.append("new")
        if i in best:
            classes.append("best")
        parts.append(_card_html(c, " ".join(classes)))
    if tag:
        parts.append(f'<span class="tag">{tag}</span>')
    parts.append("</div>")
    return "".join(parts)


def _best_positions(final: Sequence[Card], best_cards: Sequence[Card]) -> set[int]:
    remaining = list(best_cards)
    pos: set[int] = set()
    for i, c in enumerate(final):
        if c in remaining:
            pos.add(i)
            remaining.remove(c)
    return pos


def _trial_html(r: TrialReplay) -> str:
    if r.initial_best is r.final_best:
        head = r.final_best.display
    else:
        head = f"{r.initial_best.display} → {r.final_best.display}"
    out = [f'<section class="trial"><h2>Trial {r.index} <span class="arrow">{head}</span></h2>']
    best_pos = _best_positions(r.final, r.best_cards)
    if not r.steps:
        out.append(
            _row_html("dealt (stood pat)", r.final, best=best_pos,
                      tag=f"best: <b>{r.final_best.display}</b>")
        )
    else:
        for k, step in enumerate(r.steps):
            label = "dealt" if k == 0 else f"after discard {k}"
            new = set(r.steps[k - 1].discarded_indices) if k > 0 else set()
            out.append(
                _row_html(label, step.hand_before,
                          dim=set(step.discarded_indices), new=new)
            )
        out.append(
            _row_html("final", r.final,
                      new=set(r.steps[-1].discarded_indices), best=best_pos,
                      tag=f"best: <b>{r.final_best.display}</b>")
        )
    out.append("</section>")
    return "".join(out)


def trace_html(
    replays: Sequence[TrialReplay], policy_name: str, discards: int, seed: int,
    hand_size: int = 8,
) -> str:
    """The full document as a string (pure; pinned by tests)."""
    size_note = "" if hand_size == 8 else f"hand_size={hand_size}, "
    body = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>balatro_sim trace: {html.escape(policy_name)}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>Trial replay</h1>",
        f'<div class="meta">policy=<b>{html.escape(policy_name)}</b>, '
        f"discards={discards}, {size_note}seed={seed}, trials 0–{len(replays) - 1}, "
        "vanilla deck</div>",
    ]
    body.extend(_trial_html(r) for r in replays)
    body.append(
        '<div class="legend">dashed/faded = thrown away &nbsp;·&nbsp; '
        'blue border = drawn replacement &nbsp;·&nbsp; '
        "gold border = the cards best_of() would play</div>"
    )
    body.append(
        '<div class="foot">Every trial is reproducible: trial i uses '
        "Random(f\"{seed}:{i}\") -- these are the exact hands the statistics "
        "counted, not illustrations.</div>"
    )
    body.append("</body></html>")
    return "".join(body)


def render_trace_html(
    deck: Sequence[Card] | None,
    seed: int,
    n_trials: int,
    policy: Policy,
    discards: int,
    out_path: str,
    hand_size: int = 8,
) -> None:
    """Replay trials 0..n_trials-1 and write the self-contained HTML file."""
    if deck is None:
        deck = vanilla_deck()
    replays = [
        replay_trial(deck, seed, i, policy, discards, hand_size)
        for i in range(n_trials)
    ]
    doc = trace_html(replays, policy.name, discards, seed, hand_size)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
