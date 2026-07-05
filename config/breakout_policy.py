"""User-adjustable breakout strategy policy values."""

from __future__ import annotations

BREAKOUT_DEFAULT_HIGH_LEN = 201
BREAKOUT_HIGH_LEN_SEARCH_MIN = 60
BREAKOUT_HIGH_LEN_SEARCH_MAX = 350
BREAKOUT_HIGH_LEN_SEARCH_STEP = 5


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


__all__ = [
    "BREAKOUT_DEFAULT_HIGH_LEN",
    "BREAKOUT_HIGH_LEN_SEARCH_MAX",
    "BREAKOUT_HIGH_LEN_SEARCH_MIN",
    "BREAKOUT_HIGH_LEN_SEARCH_STEP",
    "build_breakout_optimizer_high_len_values",
]
