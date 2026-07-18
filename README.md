If you are an admissions officer or someone checking my resume, read this. If your checking this purely for the love of the game scroll down a bit. Also note that this readme is updated frequently with changes

Balatro is a roguelike deckbuilder where you use poker hands and wild Joker modifiers to hit increasingly massive score requirements. The entire game is a high-stakes balancing act of risk and reward, forcing you to constantly gamble your hard-earned cash on unpredictable booster packs and destructive abilities. You must boldly manipulate your deck and wager on game-breaking synergies to survive, knowing that one wrong bet or bad draw could instantly end your run.

Searching the internet, I saw that there were already score calculators(deterministic) which could tell your max score in a given hand. However for a game like Balatro where risk management and chance play a heavy role, I was surprised to see that there were no stochastic process/calculator to answer this question: given this deck and this policy, what is the distribution of outcomes, and specifically, what is the probability of failing to clear a blind?(Inspiration was when I got done dirty by luck using all discards for a flush which never came). I first thought that I could solve this using simple math but I quickly realized that the simple hypergeometric distribution I was going for wouldn't work due to Balatro having more than 1 discrard. Because of this I decided to use a monte carlo.

Here is the math I used:
1. Treat the run as a random variable. Fix a deck, a play policy(defined as the strategy the user is chasing -- for instance, flushGreedy), a joker set. One trial = shuffle, deal 8, discard and redraw per the policy, play the best hand, score it. That score is a random draw. Do it 500,000 times and you have a distribution instead of a guess.
2. The estimator. Each trial is a coin flip: did I clear the blind or not? That's a Bernoulli variable, so the win rate is just successes ÷ trials. The Law of Large Numbers says that average converges to the true probability.
3. Standard error shrinks as 1/√n — so 4x the trials buys you only 2x the precision. I use Wilson score intervals rather than the textbook normal approximation, because win rates near 0 or 1 break the formula.
4. I check my work using combinatorics. The five-card hand probabilities are exactly computable (a flush is 0.198% — derived by hand, not simulated). If the simulator doesn't reproduce them within the interval given enougn trials, the simulator is wrong. My goal is to scale this project so that it can include more complex situtiations to the point where combinatrics is less efficient than simply running a few 100k to million trials.

I used claude for execution and when stuck, but the logic, variabales, and math is all done by me. I also built an AI agent so that after a simulation is run, it automatically calculates the combinatrics and tells me how far the simulation was off mathematically.

Here are some cool results so far(These results are from before I added jokers and modded cards):
- P(straight even available in 8 cards) is only 0.098(also confirmed by exact math). This is lower than intuition says and surprised me.
- FlushChaser turns 7% flushes into 92%, and multiplies straight/royal flushes ~23× as a side effect.
- MadeHand quietly converts ~52% of hands into full houses.
- Against blind 600 at level 1, MadeHand clears 2.8% vs FlushChaser's 1.9% (Δ̂ = −0.0089 ± 0.0007, n = 100k) — the flush wall tops out at ~340 and never reaches 600; clears come from quads and straight flushes. The 92%-flush policy loses on the objective that matters(This genuinely surprised me especially since this specific simulation has only 1 hand).
- I found that many times even though the mean score was higher for certain policies, the blind clear rate was lower. This is probably because the polciies with the higher means were less consistent but achieved high scores that were useless(quadchaser) while other such as madeHand had lower means but more consistency.

I am also working on using mathplotlib to visualize data.



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
- [x] **Phase 2 — discards + policy** (two contrasting heuristics, paired
      CRN comparisons; validated, see below)
- [x] **Phase 3 — scoring layer** (chips × mult, hand levels → P(S ≥ B);
      core scope — enhancements land with Phase 4's deck edits. Exit
      criterion is in-game confirmation: see [docs/VALIDATION.md](docs/VALIDATION.md))
- [x] **Phase 4 — deck modifications** (fixed configurations only:
      remove/add/transform edits applied once, before the trial loop.
      Modeling how a deck *reaches* a state during a run — stochastic
      Tarot arrival, path-dependent decks — is deliberately out of scope
      here; that second decision layer belongs to Phase 6)
- [ ] Phase 5 — jokers (~10, mechanically diverse) I am looking for existing code about jokers.
- [ ] Phase 6 — value function / shop decisions This is very ambitious and it should be seen as a step 20. 

## Run
Note: you can change discards and hands. I also haven't found a way to code straights and straight flushes in yet so that is currently not a feature. Other poker hands to exist however such as pairchases and quadchaser
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

**Estimand correction (2026-07-14).** PLAN.md §4 defines a *single-played-hand*
trial, but the project's objective (§1–2) is P(clear the **blind**) — the sum
of up to 4 hands from a continuing deck sharing one 3-discard budget. These
are different random variables and can rank policies differently (a policy
whose hands reliably score ~300 one-shots 600 almost never, but sums past it
easily). The `blind` command and `compare --stat clear` estimate the real
clearing condition and are the headline numbers; single-hand `--blind` /
`--stat score` remain as component studies:

```
python -m balatro_sim blind --policy flushchaser --blind 600 --hands 4 --discards 3
python -m balatro_sim compare --a madehand --b flushchaser --stat clear --blind 600
```

### Run resources (hands, discards, hand size)

The three per-blind resources are all run variables. `--hands` (blind trials
only) and `--discards` (everywhere) default to the base game's 4 and 3;
`--hand-size` defaults to 8, the number of cards held in hand. All three are
accepted by every simulating command (dist, `blind`, `compare`, `trace`,
`plot`) and hold fixed across a comparison's arms, so CRN pairing is
unaffected.

```
python -m balatro_sim --hand-size 10 --trials 20000
python -m balatro_sim blind --policy flushchaser --blind 600 --hands 3 --discards 2 --hand-size 9
python -m balatro_sim compare --a none --b flushchaser --stat clear --blind 600 --hand-size 10
```

A larger hand raises availability of every class (more cards seen) — e.g.
P(flush available, no discards) climbs from ≈0.069 at 8 to ≈0.23 at 10. Like
`--mod`, a non-default hand size leaves the closed form behind: the exact-math
columns are derived for **8 dealt from the vanilla 52**, so they switch off for
any other hand size (the per-trial best-vs-availability cross-check still runs).

### Policies (π)

Deterministic, RNG-free, pinned by tests — the sim measures build and policy
jointly, so π is part of the measured object (PLAN.md §7):

- `none` — never discards (baseline; reproduces Phase 1 trial-for-trial).
- `madehand` — stands pat on straight-or-better; otherwise keeps trips/both
  pairs/pair+top-kicker/3 highest cards and discards the rest. Deliberately
  does not chase draws.
- `flushchaser` — keeps the most-populated suit, discards up to 5 off-suit
  (lowest first) until a flush lands.
- `pairchaser` / `twopairchaser` / `tripschaser` / `fullhousechaser` /
  `quadchaser` — chase one rank-family target: stop once the best available
  type is ≥ the target, otherwise keep the largest rank group(s) (two groups
  for two pair / full house, ties to the higher rank) and discard up to 5 of
  the rest, lowest first. Stop-at-target-or-better means a pairchaser stands
  pat on a dealt flush, while a quadchaser will break one to keep chasing —
  the same sacrifice FlushChaser makes with pairs.
- `highcard` — chip-max floor: always discards the 5 lowest cards (it will
  break made hands, even a dealt royal flush, and burns the shared blind
  budget on hand 1). A deliberately dominated baseline.
- `blind` — discards the first k positions sight-unseen (validation only:
  content-blind discarding leaves the final 8 uniform, so its distribution
  must still match the exact math).

### Modified decks (Phase 4)

Every simulating command (dist, `blind`, `compare`, `trace`, `plot`) takes
repeatable `--mod "verb args"` deck edits, applied **in order** before the
trial loop (order matters: convert-then-remove ≠ remove-then-convert):

```
python -m balatro_sim --mod "remove 2 3" --trials 20000
python -m balatro_sim blind --policy flushchaser --blind 600 --mod "remove 2 3"
python -m balatro_sim blind --policy flushchaser --blind 600 --mod "transform C>H"
python -m balatro_sim compare --a madehand --b flushchaser --stat clear --blind 600 --mod "remove 2 3"
```

Three primitives: `remove` (deck thinning), `add` (duplicates legal —
Balatro decks are multisets; this makes Five of a Kind / Flush House /
Flush Five reachable), `transform FROM>TO` (suit conversion `KC>KH` or
`C>H`, rank bumps `7>8`). Selectors are an exact card (`2S`), a rank
(`2`–`10`, `J`…`A`), or a suit (`S H D C`), and match **all** copies;
matching nothing is an error, never a silent no-op. This models a Tarot's
*effect* as a deck edit — it does not model named Tarots, their draw odds,
or how a deck evolves mid-run (Phase 6 scope). `compare --mod` plays both
arms on the same modified deck, so the CRN pairing still holds.

Caveats: the exact-math validation columns apply to the true vanilla 52
only and switch off for modified decks (the per-trial best-vs-availability
cross-check still runs everywhere); secret hands sit above what the
availability flags can see, so trials whose best hand is one are exempt
from that check.

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
(~15 s). `python -m pytest` works too if you have pytest. Every simulation
run also self-validates: 0 cross-check mismatches and small z-scores against
the exact combinatorics are the expected signature of a correct simulator.

