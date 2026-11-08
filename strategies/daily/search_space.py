"""Registered Daily search semantics, derived from the original Full specs.

Only PARAM-P18 uses the legacy default. New profiles must declare a variant.
Changing any field, bounds, or execution contract changes the scientific hash.
"""
from __future__ import annotations

from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from core.portfolio_entry_policy import (
    DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
)

DAILY_SEARCH_PROFILES = {
    "min_atr_limit_v2": {
        "fields": ("atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail"),
        "execution_contract": DAILY_EXECUTION_ATR_LIMIT,
    },
    "full_common_filters_v2": {
        "fields": (
            "atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail",
            "use_bb", "bb_len", "bb_mult", "use_kc", "kc_len", "kc_mult",
            "use_vol", "vol_long_len", "vol_breakout_mult",
            "use_breakout_ema_filter", "breakout_ema_len",
            "use_breakout_return_filter", "breakout_return_min",
            "use_breakout_false_filter", "breakout_false_filter_atr_pct_min",
        ),
        "execution_contract": DAILY_EXECUTION_FULL_FILTERS,
    },
}


def resolve_daily_search_profile(profile: str | None):
    """None preserves the historical PARAM-P18 contract byte-for-byte."""
    if profile in (None, ""):
        return {
            "fields": ("atr_len", "atr_times_init", "atr_times_trail"),
            "execution_contract": DAILY_EXECUTION_LEGACY,
        }
    try:
        result = DAILY_SEARCH_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(f"Unknown Daily search profile: {profile!r}") from exc
    if len(set(result["fields"])) != len(result["fields"]):
        raise ValueError("Duplicate Daily optimizer dimensions")
    if any(field not in BREAKOUT_OPTIMIZER_SEARCH_SPACE for field in result["fields"]):
        raise ValueError("Daily dimension not in canonical Full search schema")
    return result
