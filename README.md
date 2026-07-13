# Balatro Hand-Outcome Simulator

Monte Carlo simulator for Balatro whose objective is a **survival probability**
— P(score ≥ blind) — not an expected score. Joker/deck decisions are valued as
differences of clear probabilities (ΔP_win) estimated with common-random-number
paired trials. The full rationale, component contracts, phase roadmap and
validation protocol live in [PLAN.md](PLAN.md); read that first.

Pure Python, stdlib only (3.9+). CLI + printed distributions by design — no UI.

## Status

- [x] **Phase 1 — hand-type distribution** (evaluator, `best_of`, seeded trial
      loop; validated, see below)
- [ ] Phase 2 — discards + policy π
- [ ] Phase 3 — scoring layer (chips × mult, hand levels) → P(S ≥ B)
- [ ] Phase 4 — deck modifications (tarots as deck edits)
- [ ] Phase 5 — jokers (~10, mechanically diverse)
- [ ] Phase 6 — value function / shop decisions

## Run

```
python -m balatro_sim --trials 100000 --seed 42
```

Prints the distribution of the best playable hand type in 8 dealt cards
(vanilla deck, no discards), each estimate with its standard error, and a
self-validation block comparing the run against exact math.

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
| `balatro_sim/simulate.py` | Seeded trial loop. Per-trial seeding `Random(f"{seed}:{i}")` — SHA-512 string seeds, platform-stable, the CRN hook for later phases. |
| `balatro_sim/__main__.py` | Phase 1 CLI |
| `tests/` | Unit cases, exhaustive 5-card canon, exact-math consistency, MC-vs-exact gate |

## How Phase 1 is validated

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
# balatrohandoutcomesim
