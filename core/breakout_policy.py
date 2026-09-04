"""Derived breakout strategy policy helpers."""

from __future__ import annotations

from config.breakout_policy import (
    BREAKOUT_HIGH_LEN_SEARCH_MAX,
    BREAKOUT_HIGH_LEN_SEARCH_MIN,
    BREAKOUT_HIGH_LEN_SEARCH_STEP,
)


def build_breakout_optimizer_high_len_values() -> tuple[int, ...]:
    low = int(BREAKOUT_HIGH_LEN_SEARCH_MIN)
    high = int(BREAKOUT_HIGH_LEN_SEARCH_MAX)
    step = int(BREAKOUT_HIGH_LEN_SEARCH_STEP)
    if low < 1:
        raise ValueError("BREAKOUT_HIGH_LEN_SEARCH_MIN 必須 >= 1")
    if high < low:
        raise ValueError("BREAKOUT_HIGH_LEN_SEARCH_MAX 必須 >= BREAKOUT_HIGH_LEN_SEARCH_MIN")
    if step < 1:
        raise ValueError("BREAKOUT_HIGH_LEN_SEARCH_STEP 必須 >= 1")
    return tuple(range(low, high + 1, step))


__all__ = ["build_breakout_optimizer_high_len_values"]
