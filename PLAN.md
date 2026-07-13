# Balatro Hand-Outcome Simulator — Project Plan

*Spec as written 2026-07-13. This document is the project's north star; code serves it.*

## 1. Thesis

Balatro is a ruin-constrained capital allocation problem. Money has zero terminal value — it is an intermediate resource whose only purpose is to raise the probability of surviving future blinds. The failure condition is absorbing: you do not lose part of a stack, the run ends.

This makes the correct objective a survival probability, not an expected score, and the correct framework a value function over states, not Kelly sizing.

```
V(gold, jokers, ante) = Pr[win the run from this state]
```

Every shop decision is: take the action maximizing V. Interest, the $25 liquidity floor, and reroll cost decay all fall out of this as consequences, not bolted-on formulas.

Why not Kelly: Kelly maximizes expected log terminal wealth under reinvestment. Balatro has no terminal wealth and no reinvestment. The units don't reconcile — a Kelly-style EV equation ends up with survival probability on one side and dollars on the other, papered over by a fictional "run reward" term.

## 2. Problem

Existing tools (EFHIII's calculator, DivvyCr's preview mod) solve the deterministic problem: given these exact cards and jokers, what does this hand score?

Nothing solves the stochastic problem: given this deck and this policy, what is the distribution of outcomes — and specifically, what is the probability of failing to clear a blind?

## 3. Why Monte Carlo (formal justification)

The zero-discard case is closed-form (hypergeometric). Once discards enter, the player makes sequential, information-dependent decisions: they observe 8 cards and choose discards based on what they see. Analytically this requires summing over all C(52,8) ≈ 7.5×10⁸ initial hands, applying the policy to each, and integrating the resulting conditional subtrees — against a policy function that is not an algebraic object. With three discards it is hopeless.

Monte Carlo replaces integration over an intractable branching structure with sampling from it.

This is structurally an American-option pricing problem: path-dependent payoff with early-exercise decisions. Same reason MC exists in derivatives pricing.

## 4. The estimator

One trial: shuffle deck D → deal 8 → apply policy π (discard/redraw until exhausted) → play best hand → score under joker set J → score S.

Target quantity, for blind requirement B:

```
p = Pr[S >= B]
```

Let Xᵢ = 1[Sᵢ ≥ B], a Bernoulli(p). Estimator:

```
p_hat = (1/n) Σ Xᵢ        SE(p_hat) = sqrt(p(1-p)/n)
```

Consequences:

- Error shrinks as 1/√n — halving error costs 4× the trials.
- Variance is maximal at p = 0.5, vanishing at the extremes.
- Required n is computable in advance. For SE ≤ 0.005 at p ≈ 0.5: n ≥ 0.25/0.005² = 10,000.

So 100k trials is comfortable for a single probability. Trial count is not the bottleneck.

## 5. The real statistical problem: estimating ΔP_win

Joker value is a difference of probabilities:

```
Δ = p_with − p_without
```

Naive approach (two independent runs) gives Var(Δ̂) = Var(p̂₁) + Var(p̂₂). At n = 100k each, SE(Δ̂) ≈ 0.0022. Fine for Δ = 0.02. Fatal for Δ = 0.003 — the CI straddles zero and the joker is undetectable. Most jokers live in this range. This is the central statistical problem of the project.

### Fix: common random numbers (variance reduction)

Run the same shuffle seeds through both arms. Seed per-trial, not per-run (`rng = Random(i)` inside the loop). Define Dᵢ = Xᵢ_with − Xᵢ_without.

```
Var(Δ̂) = [Var(X_w) + Var(X_o) − 2·Cov(X_w, X_o)] / n
```

The two arms are highly positively correlated — garbage shuffles fail in both, great shuffles clear in both. Only marginal, near-threshold hands differ. The covariance term is large and positive, and variance of the difference collapses, often by an order of magnitude.

Most trials have Dᵢ = 0. The signal lives entirely in the trials where the joker flipped a fail into a clear. CRN isolates exactly that and discards the shared noise.

Standard practice in derivatives pricing. Highest-leverage single implementation decision in the project.

## 6. Why the mean is a trap

Balatro scoring is multiplicative (chips × mult, xmult jokers compound). Products of random variables have heavy right tails, so E[S] is dragged upward by rare enormous hands that contribute nothing to survival.

The objective is a threshold-crossing probability, i.e. the CDF at a point — not the first moment. Two builds can share a mean while one clears 90% of blinds and the other 55%.

Report the CDF. Report the left tail. The mean is decoration.

## 7. Component contracts

| Component | Contract and notes |
|---|---|
| `evaluate(five_cards) -> HandType` | Pure, deterministic, zero Balatro logic. Pure poker. Every bug here poisons everything downstream. |
| `best_of(eight_cards) -> (HandType, cards)` | All subsets of size 1–5 = 218 (Balatro allows <5 card plays). Naive is correct and fast. Do not optimize. |
| `policy.discard(hand) -> indices` | Where intellectual honesty lives. The sim measures build and policy jointly. State it explicitly; hold it fixed across comparisons; test whether conclusions are robust to it. |
| `score(hand, cards, jokers) -> int` | Where Balatro's real rules bite (ordering, retriggers, xmult). Consult EFHIII as spec; validate against Xbox. |
| `trial(seed, ...) -> bool` | Seeded — required for CRN. |
| `experiment(n, cfg_A, cfg_B) -> (Δ̂, SE)` | Paired comparison. The object that produces science. |

## 8. Phases

**Phase 1 — Hand-type distribution.** Evaluator + best_of + trial loop. No discards, no scoring. Output: frequency of each best-available hand type.
Exit: matches hand-derived hypergeometric math for the zero-discard case. This is the only chance to validate against ground truth you can derive yourself. Take it.

**Phase 2 — Discards + policy.** Introduce π. First real design decision.

**Phase 3 — Scoring layer.** Chips, mult, hand levels, enhancements. Converts hand-type distribution → score distribution → P(S ≥ B).
Exit: sim score for a hand-constructed scenario matches the real game exactly (validated on Xbox).

**Phase 4 — Deck modifications.** Tarots modeled as deck edits applied before the trial loop, not as drawn cards. Deck is already an input; no new architecture. Answers "how much does removing 8 low cards shift flush odds?"

**Phase 5 — Jokers (limited).** ~10 jokers chosen for mechanical diversity, not power: one flat-chip, one +mult, one xmult, one retrigger, one scaling, one discard-dependent. Prior art hand-writes a function per joker because no clean general abstraction exists; do not expect to find one.
Exit: adding an 11th joker of a new type requires no core-loop changes.

**Phase 6 — Value function / shop.** The destination. Consumes ΔP_win from Phases 1–5. Reroll break-even, the $25 floor, interest opportunity cost.

### Out of scope

- UI. CLI + printed distributions + matplotlib. A slick frontend on thin analysis signals time spent on the easy part.
- All ~150 jokers.
- Recreating EFHIII's calculator. It answers a different question.

## 9. Validation protocol

No automated validation available (game is on Xbox — no mods, no save inspection, no scripted diffing). Therefore validation is manual and expensive, so it must be strategic.

- Maintain a plain-text log: board state, expected score, actual score, date.
- Target cases most likely to expose bugs: retriggers, xmult ordering, enhancement interactions.
- Five carefully chosen scenarios beat fifty lazy ones.
- Re-run the full log against the sim after every mechanic added. This is a hand-built regression suite.

An unvalidated sim is a confident random number generator, and errors compound silently.

## 10. Success metric

A number, an error bar, and an interpretation:

> Under policy π (stated), vanilla deck, Small Blind at Ante 4: joker X raises clear probability from 0.612 to 0.634. Δ̂ = 0.022 ± 0.004 (95% CI, paired, n = 100,000). Community ranks X above Y; we find Y's Δ is larger (0.031) with a thinner left tail. X's mean score is higher, which likely explains the community's perception.

That paragraph is the project. Everything else is scaffolding to produce it.

## 11. First milestone

One ΔP_win. One joker, one deck, one blind. Paired experiment, CRN, with error bars.

Immediate next action: `evaluate(five_cards) -> HandType`.
