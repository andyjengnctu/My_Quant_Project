import pandas as pd

from core.signal_utils import build_optimizer_signal_feature_cache_for_df
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE


def _resolve_int_range(spec):
    low = int(spec["low"])
    high = int(spec["high"])
    step = int(spec.get("step", 1))
    return range(low, high + 1, step)


def build_optimizer_feature_cache(
    raw_data_cache,
    *,
    high_len_values,
):
    atr_len_values = set(_resolve_int_range(BREAKOUT_OPTIMIZER_SEARCH_SPACE["atr_len"]))
    atr_len_values.update(_resolve_int_range(BREAKOUT_OPTIMIZER_SEARCH_SPACE["kc_len"]))
    bb_len_values = list(_resolve_int_range(BREAKOUT_OPTIMIZER_SEARCH_SPACE["bb_len"]))
    vol_len_values = set(_resolve_int_range(BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_short_len"]))
    vol_len_values.update(range(int(BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_short_len"]["low"]), int(BREAKOUT_OPTIMIZER_SEARCH_SPACE["vol_long_len"]["high"]) + 1))

    resolved_high_len_values = sorted({int(value) for value in high_len_values})
    resolved_atr_len_values = sorted({int(value) for value in atr_len_values})
    resolved_bb_len_values = sorted({int(value) for value in bb_len_values})
    resolved_vol_len_values = sorted({int(value) for value in vol_len_values})

    return {
        ticker: build_optimizer_signal_feature_cache_for_df(
            df,
            atr_lengths=resolved_atr_len_values,
            high_lengths=resolved_high_len_values,
            bb_lengths=resolved_bb_len_values,
            vol_lengths=resolved_vol_len_values,
        )
        for ticker, df in raw_data_cache.items()
    }
