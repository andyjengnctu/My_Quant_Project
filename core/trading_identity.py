"""Canonical identity normalization for live Trading domain entities."""
from __future__ import annotations


def normalize_trading_ticker(value: object) -> str:
    """Return the canonical Trading ticker identity or fail fast.

    Trading account, broker orders, scanner rows, protection/indicator plans,
    and progress reconstruction must all compare the same normalized identity.
    Outer whitespace is formatting noise; embedded whitespace is not a legal
    ticker identity and is rejected rather than silently normalized away.
    """

    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("Trading ticker 不可為空")
    if any(ch.isspace() for ch in ticker):
        raise ValueError(f"Trading ticker 不可包含空白: {ticker!r}")
    return ticker


__all__ = ["normalize_trading_ticker"]
