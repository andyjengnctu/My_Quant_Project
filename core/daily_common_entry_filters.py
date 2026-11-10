"""Daily stock-day admission from shared, decision-time filter primitives.

This is an opt-in *candidate* capability, not a Breakout buy signal.  It runs
for declared filtered Daily execution contracts (ATR-limit or close-limit).
Keeping it outside the legacy signal producer preserves original P18 lineage.

ATR/EMA/rolling and ATR-percent calculations reuse their canonical primitives
from ``core.signal_utils`` and ``core.breakout_false_filter``.  No separate
indicator implementation, execution-price adjustment, or trading path exists.
"""
from __future__ import annotations

import numpy as np

from core.breakout_false_filter import build_breakout_false_filter_pass_condition
from core.feature_bank import coerce_feature_bank
from core.portfolio_entry_policy import (
    current_daily_execution_contract, daily_execution_uses_atr_limit,
)
from core.price_utils import adjust_long_buy_limit_array
from core.signal_utils import _rolling_mean_shift1, _rolling_mean_std0, tv_ema


def daily_common_filter_signals(
    df, params, *, atr_main, ticker=None, feature_bank=None,
    use_native_hotpath=False,
):
    """Return filtered Daily eligibility and the declared next-session buy limit.

    All columns are from the completed *information* day.  The result never
    asserts a Breakout event or creates a synthetic standalone trade log.
    """
    if bool(getattr(params, "use_breakout_buy", True)):
        raise ValueError("Filtered Daily cannot require a Breakout buy event")
    if bool(getattr(params, "use_breakout_quality_filter", False)):
        raise ValueError("Filtered Daily cannot use a Breakout-only quality gate")
    close = df["Close"].to_numpy(dtype=np.float64, copy=False)
    opening = df["Open"].to_numpy(dtype=np.float64, copy=False)
    volume = df["Volume"].to_numpy(dtype=np.float64, copy=False)
    atr = np.asarray(atr_main, dtype=np.float64)
    if close.shape != atr.shape:
        raise ValueError("Filtered Daily ATR and information-day prices disagree")
    cache = coerce_feature_bank(feature_bank)
    resolved_ticker = ticker or df.attrs.get("ticker")

    def feature(name, arguments, builder):
        if cache is None or resolved_ticker is None:
            return builder()
        return cache.get_or_compute((resolved_ticker, name, tuple(arguments)), builder)

    # Original Full gates, without the Breakout-new-high / green-candle event.
    eligible = (volume > 0).copy()
    if bool(getattr(params, "use_breakout_ema_filter", True)):
        period = int(getattr(params, "breakout_ema_len", 240))
        ema = feature("ema_close", (period,), lambda: tv_ema(
            close, period, use_native_hotpath=use_native_hotpath,
        ))
        eligible &= close > ema
    if bool(getattr(params, "use_bb", True)):
        period = int(params.bb_len)
        middle, deviation = feature("close_rolling_mean_std0", (period,),
                                    lambda: _rolling_mean_std0(
                                        close, period, use_native_hotpath=use_native_hotpath,
                                    ))
        eligible &= close > middle + float(params.bb_mult) * deviation
    if bool(getattr(params, "use_vol", True)):
        period = int(params.vol_long_len)
        minimum_multiple = float(getattr(params, "vol_breakout_mult", 1.5))
        previous_mean = feature("volume_shift1_rolling_mean", (period,),
                                lambda: _rolling_mean_shift1(
                                    volume, period, use_native_hotpath=use_native_hotpath,
                                ))
        eligible &= (np.isfinite(volume) & np.isfinite(previous_mean)
                     & (volume > 0) & (previous_mean > 0)
                     & (volume > previous_mean * minimum_multiple))
    if bool(getattr(params, "use_breakout_return_filter", False)):
        threshold = float(getattr(params, "breakout_return_min", 0.0))
        daily_return_pass = np.zeros_like(close, dtype=bool)
        if len(close) > 1:
            prior = close[:-1]
            daily_return_pass[1:] = (np.isfinite(close[1:]) & np.isfinite(prior)
                                     & (prior > 0)
                                     & ((close[1:] / prior - 1.0) > threshold))
        eligible &= daily_return_pass
    if bool(getattr(params, "use_breakout_false_filter", False)):
        threshold = float(getattr(params, "breakout_false_filter_atr_pct_min", 0.045))
        eligible &= build_breakout_false_filter_pass_condition(close, atr, threshold)

    # Same admission primitive for ATR-limit and prior-close-limit Daily paths.
    # Price is a separate execution contract; do not silently add atr_buy_tol
    # to an otherwise Min/close-limit experiment.
    use_atr_limit = daily_execution_uses_atr_limit(current_daily_execution_contract())
    raw_limits = close + atr * float(params.atr_buy_tol) if use_atr_limit else close
    limits = np.full_like(close, np.nan)
    valid = eligible & ~np.isnan(raw_limits)
    if np.any(valid):
        limits[valid] = (
            adjust_long_buy_limit_array(raw_limits[valid], ticker=resolved_ticker)
            if use_atr_limit
            else raw_limits[valid]
        )
    return eligible, limits
