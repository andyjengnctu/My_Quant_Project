"""Registered Daily search semantics, derived from the original Full specs.

Only PARAM-P18 uses the legacy default. New profiles must declare a variant.
Changing any field, bounds, or execution contract changes the scientific hash.
"""
from __future__ import annotations

from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from core.portfolio_entry_policy import (
    DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
    DAILY_EXECUTION_CLOSE_FILTERS,
)

DAILY_SEARCH_PROFILES = {
    "min_close_ema_volume_v1": {
        "fields": (
            "atr_len", "atr_times_init", "atr_times_trail",
            "vol_long_len", "vol_breakout_mult", "breakout_ema_len",
        ),
        "execution_contract": DAILY_EXECUTION_CLOSE_FILTERS,
        "fixed_switches": {
            "use_vol": True,
            "use_breakout_ema_filter": True,
        },
        "fixed_rule_contract": "daily_min_close_ema_volume_always_on_v1",
    },
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
        # Engineering trial-preparation revision, independent of the P20
        # scientific search contract. Isolates old zero-candidate Optuna studies
        # whose signal preparation ran outside the Daily Full runtime context.
        "optimizer_trial_prep_revision": "daily_full_scoped_input_preparation_v2",
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
    switches = dict(result.get("fixed_switches") or {})
    if any(name in result["fields"] or name not in BREAKOUT_OPTIMIZER_SEARCH_SPACE
           or not isinstance(value, bool) for name, value in switches.items()):
        raise ValueError("Daily fixed switches must be distinct, registered bool dimensions")
    for field in result["fields"]:
        enabled_by = BREAKOUT_OPTIMIZER_SEARCH_SPACE[field].get("enabled_by")
        if enabled_by and enabled_by not in result["fields"] and switches.get(enabled_by) is not True:
            raise ValueError(f"Daily conditional dimension requires an enabled parent: {field}")
    return result
