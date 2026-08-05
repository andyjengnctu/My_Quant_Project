"""Breakout Quality策略比較共用rule policy。"""

from __future__ import annotations

from strategies.breakout.schema import BREAKOUT_PARAM_SPECS

ALL_RULE_FILTERS_OFF_OVERRIDES = {
    "use_history_threshold": False,
    "use_breakout_reclaim_reentry": False,
    "use_kc": False,
}

ALL_OFF_INACTIVE_VALUE_OVERRIDES = {
    "breakout_ema_len": BREAKOUT_PARAM_SPECS["breakout_ema_len"]["default"],
    "bb_len": BREAKOUT_PARAM_SPECS["bb_len"]["default"],
    "bb_mult": BREAKOUT_PARAM_SPECS["bb_mult"]["default"],
    "kc_len": BREAKOUT_PARAM_SPECS["kc_len"]["default"],
    "kc_mult": BREAKOUT_PARAM_SPECS["kc_mult"]["default"],
    "vol_long_len": BREAKOUT_PARAM_SPECS["vol_long_len"]["default"],
    "vol_breakout_mult": BREAKOUT_PARAM_SPECS["vol_breakout_mult"]["default"],
    "breakout_return_min": BREAKOUT_PARAM_SPECS["breakout_return_min"]["default"],
    "breakout_false_filter_atr_pct_min": BREAKOUT_PARAM_SPECS[
        "breakout_false_filter_atr_pct_min"
    ]["default"],
    "breakout_reclaim_window_bars": BREAKOUT_PARAM_SPECS[
        "breakout_reclaim_window_bars"
    ]["default"],
    "breakout_reclaim_confirm_atr": BREAKOUT_PARAM_SPECS[
        "breakout_reclaim_confirm_atr"
    ]["default"],
    "min_history_trades": 0,
    "min_history_ev": -1.0,
    "min_history_win_rate": 0.0,
}

__all__ = [
    "ALL_OFF_INACTIVE_VALUE_OVERRIDES",
    "ALL_RULE_FILTERS_OFF_OVERRIDES",
]
