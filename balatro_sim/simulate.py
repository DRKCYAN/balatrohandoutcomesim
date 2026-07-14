"""Trial loop: shuffle, deal 8, discard/redraw under a policy, best hand type.

Seeding contract (the common-random-numbers hook): trial i under base
seed s uses random.Random(f"{s}:{i}"). String seeds hash through SHA-512,
so the shuffle stream is stable across platforms, processes and Python
builds -- required so that paired comparisons can replay identical
shuffles through both arms (CRN) and so results are reproducible. Seed
per trial, never per run. All trial randomness is the shuffle; policies
are deterministic, so a trial is a pure function of (deck, seed, i,
policy, discards).

Discard mechanics (play_out): the hand is the top 8 of the shuffle; each
discard replaces the chosen cards, in ascending hand-index order, with
the next cards off the top of the remaining deck. Policies must not
depend on hand order, but the rule is fixed so trials are reproducible.

Every trial runs the availability cross-check on the FINAL hand:
best_of() (subset enumeration) and best_from_availability() (whole-hand
counting) are independent computations of the same quantity and must
agree exactly on duplicate-free decks. Mismatches are counted, not
raised, so a full run always reports.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from math import ceil, sqrt
from typing import Callable, Optional, Sequence

from .cards import Card
from .evaluator import HandType, availability, best_from_availability, best_of
from .policy import NoDiscard, Policy
from .scoring import Levels, best_play

# NOTE on estimands (correction, 2026-07-14): PLAN.md section 4 defines a
# single-played-hand trial, but the project's objective (sections 1-2) is
# P(clear the BLIND) -- the sum of up to `hands` plays from a continuing
# deck, sharing one discard budget. Those are different random variables
# and can rank policies differently. play_blind/run_blinds below model
# the real blind; the single-hand path remains as a component study.


def trial_rng(seed: int, i: int) -> random.Random:
    """The one place trial randomness comes from."""
    return random.Random(f"{seed}:{i}")


def deal(deck: Sequence[Card], rng: random.Random, k: int = 8) -> list[Card]:
    """Shuffle a copy of the deck, deal the top k."""
    cards = list(deck)
    rng.shuffle(cards)
    return cards[:k]


@dataclass(frozen=True)
class DiscardStep:
    """One discard round, as recorded by play_out(trace=...)."""

    hand_before: tuple[Card, ...]
    discarded_indices: tuple[int, ...]  # ascending
    drawn: tuple[Card, ...]  # replacement cards, in the order they landed


def _validate_discard(idx: tuple[int, ...], hand_len: int) -> None:
    if not 1 <= len(idx) <= 5:
        raise ValueError(f"a discard is 1-5 cards, policy returned {len(idx)}")
    if len(set(idx)) != len(idx):
        raise ValueError(f"policy returned duplicate indices {idx}")
    if any(not 0 <= i < hand_len for i in idx):
        raise ValueError(f"policy returned out-of-range indices {idx}")


def play_out(
    shuffled: Sequence[Card],
    policy: Policy,
    discards: int,
    hand_size: int = 8,
    trace: Optional[list] = None,
) -> tuple[Card, ...]:
    """Deal the top hand_size of an already-shuffled deck, then let the
    policy spend up to `discards` discards. Returns the final hand.

    Does not mutate `shuffled`, so paired arms can share one shuffle.

    If `trace` is a list, one DiscardStep per round is appended to it --
    the replay hook (trace.py). Same loop either way, so a traced replay
    cannot diverge from what the hot path did; the hot path (trace=None)
    allocates nothing extra.
    """
    hand = list(shuffled[:hand_size])
    draw = hand_size  # index of the next card off the top
    left = discards
    while left > 0:
        idx = tuple(policy.discard(tuple(hand), left))
        if not idx:
            break
        _validate_discard(idx, len(hand))
        if draw + len(idx) > len(shuffled):
            raise RuntimeError("deck exhausted during redraw")
        if trace is None:
            for j in sorted(idx):
                hand[j] = shuffled[draw]
                draw += 1
        else:
            before = tuple(hand)
            drawn = []
            for j in sorted(idx):
                hand[j] = shuffled[draw]
                drawn.append(shuffled[draw])
                draw += 1
            trace.append(DiscardStep(before, tuple(sorted(idx)), tuple(drawn)))
        left -= 1
    return tuple(hand)


@dataclass
class DistributionReport:
    n: int
    seed: int
    policy_name: str
    discards: int
    best_counts: Counter  # HandType -> trials where it was the best playable
    avail_counts: Counter  # availability flag -> trials where it was set
    inconsistencies: int  # trials where best_of() != availability floor
    # per-trial best-play scores, in trial order; None unless the run was
    # given hand levels (scoring is opt-in so type-only runs pay nothing)
    scores: Optional[list[int]] = None

    def p_best(self, t: HandType) -> float:
        return self.best_counts.get(t, 0) / self.n

    def p_avail(self, key: str) -> float:
        return self.avail_counts.get(key, 0) / self.n

    def p_score_at_least(self, blind: float) -> float:
        if self.scores is None:
            raise ValueError("run was not scored; pass levels= to run_distribution")
        return sum(1 for s in self.scores if s >= blind) / self.n

    def score_percentile(self, q: float) -> int:
        """Nearest-rank percentile of the score distribution, q in [0, 100]."""
        if self.scores is None:
            raise ValueError("run was not scored; pass levels= to run_distribution")
        ordered = sorted(self.scores)
        k = max(0, min(self.n - 1, ceil(q / 100 * self.n) - 1))
        return ordered[k]


# Back-compat alias: Phase 1 reports are the zero-discard special case.
Phase1Report = DistributionReport


def se(p: float, n: int) -> float:
    """Standard error of a Bernoulli(p) mean over n trials."""
    return sqrt(p * (1 - p) / n)


def run_distribution(
    deck: Sequence[Card],
    n: int,
    seed: int = 0,
    policy: Optional[Policy] = None,
    discards: int = 0,
    progress: Optional[Callable[[int], None]] = None,
    levels: Optional[Levels] = None,
) -> DistributionReport:
    """n independent trials: shuffle, deal 8, play out the policy's
    discards, record the best playable hand type of the final hand.

    With `levels` (a HandType -> level mapping; pass {} for all level 1)
    each trial additionally records the score of the best PLAY -- the
    highest-scoring subset, which can be a lower TYPE than best_of's
    (a leveled pair can outscore a flush; even at level 1 a junk full
    house loses to a face flush). Type counts keep their capability
    semantics either way.
    """
    if n < 1:
        raise ValueError("need at least one trial")
    if discards < 0:
        raise ValueError("discards must be >= 0")
    if policy is None:
        policy = NoDiscard()
    best_counts: Counter = Counter()
    avail_counts: Counter = Counter()
    inconsistencies = 0
    scores: Optional[list[int]] = None if levels is None else []
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        final = play_out(shuffled, policy, discards)
        t, _ = best_of(final)
        av = availability(final)
        best_counts[t] += 1
        for key, flag in av.items():
            if flag:
                avail_counts[key] += 1
        if best_from_availability(av) is not t:
            inconsistencies += 1
        if scores is not None:
            scores.append(best_play(final, levels).score)
        if progress is not None and (i + 1) % 10_000 == 0:
            progress(i + 1)
    return DistributionReport(
        n, seed, policy.name, discards, best_counts, avail_counts,
        inconsistencies, scores,
    )


def run_phase1(
    deck: Sequence[Card],
    n: int,
    seed: int = 0,
    progress: Optional[Callable[[int], None]] = None,
) -> DistributionReport:
    """Phase 1 loop: the zero-discard special case, kept as a stable entry
    point (its results are pinned by tests and by recorded runs)."""
    return run_distribution(deck, n, seed, policy=NoDiscard(), discards=0, progress=progress)


# ------------------------------------------------------------ blind trials

@dataclass(frozen=True)
class BlindResult:
    """Outcome of one full blind (up to `hands` plays, shared discards)."""

    total: int
    cleared: Optional[bool]  # None when no blind target was set
    hands_used: int
    discards_used: int
    hand_scores: tuple[int, ...]
    hand_types: tuple[HandType, ...]


def _replace_positions(
    hand: list[Card], positions: Sequence[int], shuffled: Sequence[Card], draw: int
) -> tuple[list[Card], int]:
    """Replace hand[positions] (ascending) with the next cards off the
    deck -- the same rule play_out uses, so hands=1 blinds reproduce the
    single-hand engine exactly. Positions the deck can no longer refill
    are removed instead (the hand shrinks; impossible on a vanilla deck,
    real on small Phase-4 decks)."""
    out = list(hand)
    unfilled: list[int] = []
    for j in sorted(positions):
        if draw < len(shuffled):
            out[j] = shuffled[draw]
            draw += 1
        else:
            unfilled.append(j)
    for j in reversed(unfilled):
        del out[j]
    return out, draw


def _positions_of(hand: Sequence[Card], played: Sequence[Card]) -> list[int]:
    """Hand positions of the played cards (multiset-safe for duplicates)."""
    remaining = list(played)
    positions: list[int] = []
    for k, c in enumerate(hand):
        if c in remaining:
            positions.append(k)
            remaining.remove(c)
    return positions


def play_blind(
    shuffled: Sequence[Card],
    policy: Policy,
    hands: int = 4,
    discards: int = 3,
    levels: Optional[Levels] = None,
    blind: Optional[float] = None,
    hand_size: int = 8,
) -> BlindResult:
    """One full blind: deal 8, then each turn the policy either discards
    (non-empty return, budget remaining) or plays best_play(). Played and
    discarded cards leave the deck permanently; replacements come off the
    top. Stops early once total >= blind (accumulation is monotone, so
    P(clear) is unaffected; the reported total is censored at clear).

    Base game: hands=4, discards=3 -- one shared discard budget for the
    whole blind, which is the resource structure the single-hand trial
    got wrong. Does not mutate `shuffled` (CRN-safe). `levels=None`
    scores everything at level 1.
    """
    if hands < 1:
        raise ValueError("a blind allows at least one hand")
    if discards < 0:
        raise ValueError("discards must be >= 0")
    lv: Levels = levels if levels is not None else {}
    hand = list(shuffled[:hand_size])
    draw = len(hand)
    hands_left = hands
    discards_left = discards
    total = 0
    hand_scores: list[int] = []
    hand_types: list[HandType] = []
    while hands_left > 0 and hand:
        idx = ()
        if discards_left > 0:
            idx = tuple(policy.discard(tuple(hand), discards_left))
        if idx:
            _validate_discard(idx, len(hand))
            hand, draw = _replace_positions(hand, idx, shuffled, draw)
            discards_left -= 1
            continue
        pr = best_play(tuple(hand), lv)
        total += pr.score
        hand_scores.append(pr.score)
        hand_types.append(pr.hand_type)
        hands_left -= 1
        hand, draw = _replace_positions(
            hand, _positions_of(hand, pr.played), shuffled, draw
        )
        if blind is not None and total >= blind:
            break
    cleared = None if blind is None else total >= blind
    return BlindResult(
        total, cleared, hands - hands_left, discards - discards_left,
        tuple(hand_scores), tuple(hand_types),
    )


@dataclass
class BlindReport:
    n: int
    seed: int
    policy_name: str
    hands: int
    discards: int
    blind: Optional[float]
    cleared_count: Optional[int]  # None when blind was None
    totals: list[int]  # censored at clear when blind is set
    hands_used: Counter  # hands at stop; failed trials excluded when blind set

    @property
    def p_clear(self) -> float:
        if self.cleared_count is None:
            raise ValueError("run had no blind target; pass blind= to run_blinds")
        return self.cleared_count / self.n

    def total_percentile(self, q: float) -> int:
        ordered = sorted(self.totals)
        k = max(0, min(self.n - 1, ceil(q / 100 * self.n) - 1))
        return ordered[k]


def run_blinds(
    deck: Sequence[Card],
    n: int,
    seed: int = 0,
    policy: Optional[Policy] = None,
    hands: int = 4,
    discards: int = 3,
    blind: Optional[float] = None,
    levels: Optional[Levels] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> BlindReport:
    """n independent blind trials under the standard seeding contract."""
    if n < 1:
        raise ValueError("need at least one trial")
    if policy is None:
        policy = NoDiscard()
    cleared_count: Optional[int] = None if blind is None else 0
    totals: list[int] = []
    hands_used: Counter = Counter()
    for i in range(n):
        shuffled = list(deck)
        trial_rng(seed, i).shuffle(shuffled)
        result = play_blind(shuffled, policy, hands, discards, levels, blind)
        totals.append(result.total)
        if blind is None or result.cleared:
            hands_used[result.hands_used] += 1
        if blind is not None and result.cleared:
            cleared_count += 1
        if progress is not None and (i + 1) % 10_000 == 0:
            progress(i + 1)
    return BlindReport(
        n, seed, policy.name, hands, discards, blind,
        cleared_count, totals, hands_used,
    )
