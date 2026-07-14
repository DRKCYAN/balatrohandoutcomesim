# In-game validation log (PLAN.md §9)

The game runs on Xbox: no mods, no save inspection, no scripted diffing.
Validation is therefore manual and expensive, so it must be strategic —
five carefully chosen scenarios beat fifty lazy ones. This file is the
hand-built regression suite: **re-run every row after each new mechanic
lands in the simulator** and add rows targeting whatever the new mechanic
could get wrong.

## Protocol

1. Use a fresh run with **no jokers** and **no boss-blind effect active**
   (play scenarios on a Small or Big Blind), vouchers untouched.
2. Hand levels must match the scenario (level 1 = never leveled this run;
   leveled scenarios say which planet to use first).
3. In-game, select **exactly** the listed cards and play them. Record the
   score the hand added.
4. Run the listed command; the sim's prediction must match the game
   **exactly** — a 1-chip difference is a bug, not noise.
5. Fill in `actual`, `date`, `pass`. Any failure: file it, fix the sim,
   re-run the whole log.

## Scenarios (Phase 3: core scoring — base, levels, card chips)

| # | scenario | command | predicted | actual | date | pass |
|---|----------|---------|-----------|--------|------|------|
| 1 | Single ace, level 1 (High Card) | `python -m balatro_sim score "AS"` | 16 | | | |
| 2 | Pair of kings + 3 junk kickers, level 1 — kickers must add nothing | `python -m balatro_sim score "KS KH 7D 4C 2S"` | 60 | | | |
| 3 | Face flush, level 1 | `python -m balatro_sim score "KH QH JH 9H 7H"` | 324 | | | |
| 4 | Same kings pair after ONE Mercury (Pair level 2) | `python -m balatro_sim score "KS KH 7D 4C 2S" --level pair=2` | 135 | | | |
| 5a | Junk full house, level 1 | `python -m balatro_sim score "2C 2D 2H 3S 3H"` | 208 | | | |
| 5b | Ace-high flush, level 1 — must beat 5a's full house (score-max ≠ type-max) | `python -m balatro_sim score "AH KH QH JH 9H"` | 340 | | | |

Worked predictions, for the record:

- #1: (5 base + 11 ace) × 1 = 16
- #2: (10 base + 10 + 10) × 2 = 60 — the 7, 4, 2 do not score
- #3: (35 base + 10+10+10+9+7) × 4 = 324
- #4: Pair L2 = (10+15 base) and (2+1 mult) → (25 + 20) × 3 = 135
- #5a: (40 base + 2+2+2+3+3) × 4 = 208
- #5b: (35 base + 11+10+10+10+9) × 4 = 340

## Constants pending in-game confirmation

`LEVEL_INCREMENTS` in `balatro_sim/scoring.py` was transcribed from the
Balatro wikis (balatrogame.fandom.com and balatrowiki.org, 2026-07-13).
Scenario #4 confirms Mercury. The rest get confirmed the first time a run
naturally levels them — add a log row when that happens:

- [ ] Pluto (High Card +10 / +1)
- [x] Mercury (Pair +15 / +1) — via scenario #4
- [ ] Uranus (Two Pair +20 / +1)
- [ ] Venus (Three of a Kind +20 / +2)
- [ ] Saturn (Straight +30 / +3)
- [ ] Jupiter (Flush +15 / +2)
- [ ] Earth (Full House +25 / +2)
- [ ] Mars (Four of a Kind +30 / +3)
- [ ] Neptune (Straight Flush +40 / +4; Royal Flush shares it)
- [ ] Planet X / Ceres / Eris (secret hands — Phase 4+)

(Scenario #4's checkbox is pre-marked as the *planned* confirmer: tick the
others as they're observed; untick #4's if the game ever disagrees.)
