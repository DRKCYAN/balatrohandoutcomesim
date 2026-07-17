"""Card primitives and deck construction. A Card is a (rank, suit)
NamedTuple: ranks 2..14 (J=11..A=14), suits 0..3 indexing SUITS. Modified
decks are allowed, so don't assume card uniqueness.
"""
from __future__ import annotations

from typing import NamedTuple

SUITS = "SHDC"

_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
_NAME_RANKS = {"J": 11, "Q": 12, "K": 13, "A": 14, "T": 10}

# Chip value of a card when it scores (Balatro: faces 10, ace 11).
CHIP_VALUE = {r: r for r in range(2, 11)} | {11: 10, 12: 10, 13: 10, 14: 11}


class Card(NamedTuple):
    rank: int
    suit: int

    def __str__(self) -> str:
        return f"{_RANK_NAMES.get(self.rank, str(self.rank))}{SUITS[self.suit]}"


def card(text: str) -> Card:
    """Parse 'AS', 'kh', '10D' or 'TD' into a Card. Case-insensitive."""
    t = text.strip().upper()
    if len(t) < 2:
        raise ValueError(f"cannot parse card {text!r}")
    rank_part, suit_part = t[:-1], t[-1]
    if suit_part not in SUITS:
        raise ValueError(f"bad suit in {text!r}")
    if rank_part in _NAME_RANKS:
        rank = _NAME_RANKS[rank_part]
    else:
        try:
            rank = int(rank_part)
        except ValueError:
            raise ValueError(f"bad rank in {text!r}") from None
        if not 2 <= rank <= 10:
            raise ValueError(f"bad rank in {text!r}")
    return Card(rank, SUITS.index(suit_part))


def hand(text: str) -> tuple[Card, ...]:
    """Parse a space-separated card list: 'AS KS QS JS 10S'."""
    return tuple(card(part) for part in text.split())


def vanilla_deck() -> list[Card]:
    """The standard 52-card Balatro starting deck (no modifiers)."""
    return [Card(rank, suit) for suit in range(4) for rank in range(2, 15)]
