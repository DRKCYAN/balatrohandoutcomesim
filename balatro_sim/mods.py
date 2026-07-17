"""Phase 4: fixed deck modifications -- pure deck -> deck functions applied
once before the trial loop. Three primitives (Remove/Add/Transform) compose
as an order-sensitive LIST; apply_all() checks the card count after each
step. Selectors are an exact card, a rank, or a suit (case-insensitive, see
parse_selector); matching nothing is an error, never a no-op.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from .cards import _NAME_RANKS, SUITS, Card, card as parse_card

# ("card", Card) | ("rank", int) | ("suit", int)
Selector = tuple[str, Union[Card, int]]

Mod = Union["Remove", "Add", "Transform"]


def parse_selector(tok: str) -> Selector:
    t = tok.strip().upper()
    if len(t) == 1 and t in SUITS:
        return ("suit", SUITS.index(t))
    if t in _NAME_RANKS:
        return ("rank", _NAME_RANKS[t])
    if t.isdigit() and 2 <= int(t) <= 10:
        return ("rank", int(t))
    try:
        return ("card", parse_card(t))
    except ValueError:
        raise ValueError(
            f"bad selector {tok!r}: expected a card ('KH', '10D'), a rank "
            f"('2'-'10', 'J', 'Q', 'K', 'A'), or a suit ('S', 'H', 'D', 'C')"
        ) from None


def _matches(sel: Selector, c: Card) -> bool:
    kind, v = sel
    if kind == "card":
        return c == v
    if kind == "rank":
        return c.rank == v
    return c.suit == v


class Remove:
    """remove SEL ...: delete every matching copy, selector by selector."""

    def __init__(self, *tokens: str):
        if not tokens:
            raise ValueError("remove: needs at least one selector")
        self.selectors = tuple((t, parse_selector(t)) for t in tokens)
        self.text = "remove " + " ".join(tokens)

    def apply(self, deck: Sequence[Card]) -> list[Card]:
        out = list(deck)
        for tok, sel in self.selectors:
            kept = [c for c in out if not _matches(sel, c)]
            if len(kept) == len(out):
                raise ValueError(f"{self.text}: {tok!r} matches nothing in the deck")
            out = kept
        return out


class Add:
    """add CARD ...: append exact cards; repeat a card to add copies."""

    def __init__(self, *tokens: str):
        if not tokens:
            raise ValueError("add: needs at least one card")
        cards = []
        for t in tokens:
            kind, v = parse_selector(t)
            if kind != "card":
                raise ValueError(
                    f"add {t!r}: add takes exact cards only "
                    f"(a bare rank or suit is ambiguous)"
                )
            cards.append(v)
        self.cards = tuple(cards)
        self.text = "add " + " ".join(tokens)

    def apply(self, deck: Sequence[Card]) -> list[Card]:
        return list(deck) + list(self.cards)


class Transform:
    """transform FROM>TO ...: convert every card matching FROM (exact TO
    replaces outright, a rank keeps the suit, a suit keeps the rank)."""

    def __init__(self, *tokens: str):
        if not tokens:
            raise ValueError("transform: needs at least one FROM>TO pair")
        pairs = []
        for t in tokens:
            frm, sep, to = t.partition(">")
            if not sep or not frm or not to:
                raise ValueError(f"transform {t!r}: expected FROM>TO")
            pairs.append((t, parse_selector(frm), parse_selector(to)))
        self.pairs = tuple(pairs)
        self.text = "transform " + " ".join(tokens)

    def apply(self, deck: Sequence[Card]) -> list[Card]:
        out = list(deck)
        for raw, frm, to in self.pairs:
            idx = [i for i, c in enumerate(out) if _matches(frm, c)]
            if not idx:
                raise ValueError(f"{self.text}: {raw!r} matches nothing in the deck")
            for i in idx:
                out[i] = _convert(out[i], to)
        return out


def _convert(c: Card, to: Selector) -> Card:
    kind, v = to
    if kind == "card":
        return v
    if kind == "rank":
        return Card(v, c.suit)
    return Card(c.rank, v)


_VERBS = {"remove": Remove, "add": Add, "transform": Transform}


def parse_mod(text: str) -> Mod:
    """'remove 2 3' | 'add AS AS' | 'transform KC>KH 7D>8D' -> a mod."""
    parts = text.split()
    if not parts:
        raise ValueError("empty mod")
    verb, args = parts[0].lower(), parts[1:]
    if verb not in _VERBS:
        raise ValueError(
            f"unknown mod verb {parts[0]!r}; choose from {', '.join(_VERBS)}"
        )
    return _VERBS[verb](*args)


def parse_mods(texts: Optional[Sequence[str]]) -> list[Mod]:
    return [parse_mod(t) for t in texts or []]


def apply_all(deck: Sequence[Card], mods: Sequence[Mod]) -> list[Card]:
    """Apply modifications in order, checking the card count after every
    step against what the primitive implies (Add: +len(cards), Transform:
    unchanged, Remove: strictly smaller); a mismatch raises. Does not
    mutate `deck`."""
    out = list(deck)
    for m in mods:
        before = len(out)
        nxt = m.apply(out)
        if isinstance(m, Add):
            ok = len(nxt) == before + len(m.cards)
        elif isinstance(m, Transform):
            ok = len(nxt) == before
        else:
            ok = len(nxt) < before
        if not ok:
            raise RuntimeError(
                f"{m.text}: composition check failed ({before} -> {len(nxt)} cards)"
            )
        out = nxt
    return out


def summarize(deck: Sequence[Card]) -> str:
    """Compact composition line for run headers, e.g.
    '44 cards, S11 H11 D11 C11' (+ ', duplicates present')."""
    counts = [0, 0, 0, 0]
    for c in deck:
        counts[c.suit] += 1
    parts = " ".join(f"{SUITS[s]}{counts[s]}" for s in range(4))
    dup = ", duplicates present" if len(set(deck)) != len(deck) else ""
    return f"{len(deck)} cards, {parts}{dup}"
