# Balatro Hand-Outcome Simulator

Monte Carlo simulator for Balatro whose objective is a **survival probability**
— P(score ≥ blind) — not an expected score. Joker/deck decisions are valued as
differences of clear probabilities (ΔP_win) estimated with common-random-number
paired trials. The full rationale, component contracts, phase roadmap and
validation protocol live in [PLAN.md](PLAN.md); read that first.

Pure Python, stdlib only (3.9+); `matplotlib` is the one *optional* dependency,
needed only for the `plot` command. CLI + printed distributions + static
visual reports by design — no UI.

## Status

- [x] **Phase 1 — hand-type distribution** (evaluator, `best_of`, seeded trial
      loop; validated, see below)
- [x] **Phase 2 — discards + policy π** (two contrasting heuristics, paired
      CRN comparisons; validated, see below)
- [x] **Phase 3 — scoring layer** (chips × mult, hand levels → P(S ≥ B);
      core scope — enhancements land with Phase 4's deck edits. Exit
      criterion is in-game confirmation: see [docs/VALIDATION.md](docs/VALIDATION.md))
- [ ] Phase 4 — deck modifications (tarots as deck edits)
- [ ] Phase 5 — jokers (~10, mechanically diverse)
- [ ] Phase 6 — value function / shop decisions

## Run

```
python -m balatro_sim --trials 100000 --seed 42
python -m balatro_sim --policy flushchaser --discards 3 --trials 20000
python -m balatro_sim --policy flushchaser --discards 3 --blind 600 --level flush=2
python -m balatro_sim score "KS KH 7D 4C 2S" --level pair=2
python -m balatro_sim compare --a none --b flushchaser --stat flush --discards 3 --trials 100000
python -m balatro_sim compare --a madehand --b flushchaser --stat score --blind 600 --trials 100000
```

The first prints the distribution of the best playable hand type in 8 dealt
cards (vanilla deck, no discards), each estimate with its standard error, and
a self-validation block comparing the run against exact math. The second runs
the same distribution under a discard policy. `--blind`/`--level` add the
score section: percentiles of the best-play score per trial plus
P(S ≥ blind) ± SE. `score` prints the exact chips × mult breakdown for a
specific play — the command used by the in-game validation log
([docs/VALIDATION.md](docs/VALIDATION.md)). `compare` is a paired
common-random-numbers comparison: both arms replay identical shuffles, and the
report gives p_A, p_B, Δ̂ ± SE, the 95% CI, and the flip counts where the
signal lives; `--stat score --blind B` compares P(S ≥ B) directly.

Scoring note: the trial plays the highest-**scoring** subset, which is not
always the highest-ranked hand type — even at level 1, a junk full house
(40 + 12) × 4 = 208 loses to an ace-high flush (35 + 50) × 4 = 340. Type
distributions keep their capability semantics; scores model the optimal
greedy player.

### Policies (π)

Deterministic, RNG-free, pinned by tests — the sim measures build and policy
jointly, so π is part of the measured object (PLAN.md §7):

- `none` — never discards (baseline; reproduces Phase 1 trial-for-trial).
- `madehand` — stands pat on straight-or-better; otherwise keeps trips/both
  pairs/pair+top-kicker/3 highest cards and discards the rest. Deliberately
  does not chase draws.
- `flushchaser` — keeps the most-populated suit, discards up to 5 off-suit
  (lowest first) until a flush lands.
- `blind` — discards the first k positions sight-unseen (validation only:
  content-blind discarding leaves the final 8 uniform, so its distribution
  must still match the exact math).

### Visualize

```
python -m balatro_sim trace --policy flushchaser --discards 3 --trials 12 --out trace.html
python -m balatro_sim plot dist --policies none madehand flushchaser --out dist.png
python -m balatro_sim plot converge --policy flushchaser --stat flush --out converge.png
python -m balatro_sim plot discards --policies madehand flushchaser --out discards.png
python -m balatro_sim plot flips --a none --b flushchaser --stat flush --out flips.png
python -m balatro_sim plot cdf --policies none madehand flushchaser --blind 600 --out cdf.png
```

`trace` writes a self-contained HTML replay (no dependencies, open in any
browser): each trial shows the dealt cards, what the policy threw away
(dimmed), what it drew (blue border), and the cards `best_of` would play
(gold border). Because trials are seeded, these are the *exact* hands the
statistics counted, not illustrations.

`plot` (requires `pip install matplotlib`) renders PNGs: the best-hand
distribution by policy with 95% CIs, a convergence curve showing the CI band
shrinking as 1/√n, the value-of-a-discard curve, and the CRN flip grid —
one cell per paired trial, where the green/red imbalance *is* the estimator's
signal.

## Test

```
python -m unittest discover -s tests -t . -v
```

Set `BALATRO_SKIP_SLOW=1` to skip the exhaustive 2.6M-hand evaluator check
(~15 s). `python -m pytest` works too if you have pytest.

## Layout

| Path | Contents |
|---|---|
| `balatro_sim/cards.py` | `Card`, parsing (`hand("AS KH 10D")`), `vanilla_deck()` |
| `balatro_sim/evaluator.py` | `HandType` (Balatro taxonomy incl. secret hands), `evaluate()`, `best_of()` (naive 218-subset enumeration, per contract), `availability()` cross-check |
| `balatro_sim/exact.py` | Hand-derived exact probabilities (Fractions). Self-contained on purpose: shares no code with the simulator, so agreement is evidence. |
| `balatro_sim/policy.py` | The discard policies above, behind a minimal `Policy` protocol (`discard(hand, discards_left) -> indices`). |
| `balatro_sim/scoring.py` | Chips × mult with hand levels (planet increments, wiki-sourced, pending in-game confirmation); `best_play` = highest-scoring subset. |
| `balatro_sim/exact_discard.py` | Conditional hypergeometric ground truth for single-discard states (4-flush completion, pair→trips). Same zero-import discipline as `exact.py`. |
| `balatro_sim/simulate.py` | Seeded trial loop + `play_out` discard mechanics. Per-trial seeding `Random(f"{seed}:{i}")` — SHA-512 string seeds, platform-stable, the CRN hook. |
| `balatro_sim/experiment.py` | `paired_experiment`: both arms share each trial's shuffle (CRN); reports Δ̂, SE, CI and flip counts. |
| `balatro_sim/trace.py` | HTML trial replay (self-contained, zero deps). Steps come from `play_out`'s own trace hook, so replays cannot diverge from what was measured. |
| `balatro_sim/charts.py` | matplotlib PNGs (distribution, convergence, discards curve, CRN flip grid); validated colorblind-safe palette. |
| `balatro_sim/__main__.py` | CLI: distribution + `compare` + `trace` + `plot` |
| `tests/` | Unit cases, exhaustive 5-card canon, exact-math consistency, MC-vs-exact gates, policy pins, discard-mechanics gates, paired-estimator gates |

## How it is validated

Phase 1 (uniform deal):

1. **Exhaustive canon** — `evaluate()` over all C(52,5) = 2,598,960 hands
   reproduces the canonical poker frequencies to the digit (royal 4, straight
   flush 36, quads 624, …).
2. **Exact 8-card math** — `exact.py` derives availability probabilities three
   independent ways (rank-multiset enumeration, closed forms, window
   inclusion-exclusion); overlapping derivations must agree as exact fractions,
   and the MC estimates must land within 4.5 SE of them.
3. **Per-trial cross-check** — every trial, `best_of()` (subset enumeration) is
   compared with an independent whole-hand counting implementation; the two
   must agree exactly. Any mismatch is counted and reported by every CLI run.

Phase 2 (discards), where no global closed form exists:

4. **Blind-discard invariance** — a content-blind policy leaves the final 8 a
   uniform subset, so the full Phase 1 exact-math gate must still pass through
   the discard/redraw machinery.
5. **Conditional hypergeometrics** — in fully specified single-discard states
   (kept exactly 4 suited and drew 4; kept a pair and drew 5) the redraw is a
   plain hypergeometric draw from the 44 unseen cards; conditioned MC trials
   must match `exact_discard.py` within 4.5 SE.
6. **CRN identities** — a policy compared against itself must give Δ̂ = 0 with
   SE = 0 and zero flips *exactly*; paired arm means must equal the
   corresponding distribution-run tail probabilities *exactly* (same seeds,
   same trials).
7. **NoDiscard ≡ Phase 1** trial-for-trial, and monotonicity: FlushChaser's
   flush rate must strictly rise with each added discard.

Phase 3 (scoring), where the ground truth is the game itself:

8. **Hand-computed pins** — every scenario in the test suite carries its
   arithmetic in a comment (pair of kings = (10+20)×2 = 60, royal flush =
   (100+51)×8 = 1208, …), including the score-max ≠ type-max divergence
   cases.
9. **Path coherence** — `score`, `best_play`, `run_distribution(levels=…)`
   and `paired_experiment(levels=…)` must agree exactly on shared seeds.
10. **The in-game log** — [docs/VALIDATION.md](docs/VALIDATION.md): prescribed
    scenarios played on Xbox must match the sim to the chip. This is Phase 3's
    exit criterion and the standing regression suite for every later mechanic.
# balatrohandoutcomesim
