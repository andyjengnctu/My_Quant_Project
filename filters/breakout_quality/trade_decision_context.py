"""PIT, scale-invariant strategy conditions for real-path model inputs.

This owner consumes the same verified teacher as the target. It reads no
outcomes, fills, portfolio state or future prices. Missing conditions are
explicit zero-plus-mask inputs and never change sample/score eligibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from core.signal_utils import _rolling_high_shift1
from filters.breakout_quality.trade_path_label import build_signal_cache

CONTEXT_FEATURES = (
    'teacher_available', 'geometry_available',
    'log1p_high_len', 'log1p_atr_len', 'atr_buy_tol',
    'atr_times_init', 'atr_times_trail', 'fixed_risk', 'max_position_cap_pct',
    'atr_to_close', 'prior_breakout_high_to_close', 'new_setup_today',
    'prior_setup_available', 'log1p_sessions_since_setup',
)
CONTEXT_CONTRACT = {
    'contract_id': 'canonical_min_decision_context_v1',
    'features': list(CONTEXT_FEATURES),
    'teacher': 'same_seed_date_effective_published_by_information_close',
    'geometry': 'canonical_atr_and_prior_high_ratios_no_tick_rounded_price',
    'setup_age': 'sessions_since_latest_raw_setup_under_current_effective_policy',
    'setup_age_is_pending_order_state': False,
    'missing': 'zero_with_explicit_teacher_geometry_and_setup_masks',
    'price_legality': 'invariant_under_positive_uniform_ohlc_rescaling',
    'fit_scope': 'unchanged_target_valid_mature_rows',
}


def build_trade_decision_context(*, source, frame, positions, ticker, exchange_calendar):
    """Materialize causal numeric context for every requested information day.

    ``source`` is bound by the canonical same-seed teacher loader. A parameter
    scheduled for the next exchange session is used only if its publication is
    already visible at this information close. Later periods cannot backfill.
    No order/stop rounding is fed to a model from current-vintage adjusted data.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or (len(positions) and
            (np.any(positions < 0) or np.any(positions >= len(frame)))):
        raise ValueError('decision context positions outside canonical price rows')
    result = np.zeros((len(positions), len(CONTEXT_FEATURES)), dtype=np.float32)
    if not len(positions):
        return result
    if source.source_id != 'formal_min_roos_seed' or len(source.schedule) != len(source.publication_dates):
        raise ValueError('decision context requires verified formal Min teacher')
    # Truncate BEFORE indicator construction. Values after the last requested
    # information close cannot influence either inputs or their availability.
    frame = frame.iloc[:int(positions.max()) + 1]
    dates = pd.DatetimeIndex(frame.index).normalize()
    calendar = pd.DatetimeIndex(exchange_calendar).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing or calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('decision context requires ordered canonical sessions')
    for period, published in zip(source.schedule, source.publication_dates):
        if period.source_id != source.source_id or pd.isna(pd.Timestamp(published)):
            raise ValueError('unverified decision context teacher publication')
    feature_index = {name: i for i, name in enumerate(CONTEXT_FEATURES)}
    information_dates = dates.take(positions)
    next_indices = calendar.searchsorted(information_dates, side='right')
    has_next = next_indices < len(calendar)
    next_dates = calendar.take(np.minimum(next_indices, len(calendar) - 1))
    close = frame['Close'].to_numpy(dtype=np.float64)
    high = frame['High'].to_numpy(dtype=np.float64)
    claimed = np.zeros(len(positions), dtype=bool)
    cache = {}
    for period, published in zip(source.schedule, source.publication_dates):
        effective = has_next & (next_dates >= period.start)
        if period.end is not None:
            effective &= next_dates <= period.end
        if np.any(effective & claimed):
            raise ValueError('overlapping decision context teacher periods')
        claimed |= effective
        rows = np.flatnonzero(effective & (information_dates >= pd.Timestamp(published).normalize()))
        if not len(rows):
            continue
        params = period.params
        if period.params_signature not in cache:
            atr, setups, _sell, _rounded_limits = build_signal_cache(frame, params, ticker=ticker)
            breakout = _rolling_high_shift1(high, int(params.high_len))
            # Raw signal history does not assert that a rounded pending order
            # is still valid. Indicators are causal despite batched evaluation.
            previous = np.maximum.accumulate(np.where(setups, np.arange(len(frame)), -1))
            cache[period.params_signature] = (atr, setups, breakout, previous)
        atr, setups, breakout, previous = cache[period.params_signature]
        selected = positions[rows]
        values = {
            'teacher_available': 1., 'log1p_high_len': np.log1p(params.high_len),
            'log1p_atr_len': np.log1p(params.atr_len), 'atr_buy_tol': params.atr_buy_tol,
            'atr_times_init': params.atr_times_init, 'atr_times_trail': params.atr_times_trail,
            'fixed_risk': params.fixed_risk, 'max_position_cap_pct': params.max_position_cap_pct,
            'new_setup_today': setups[selected],
        }
        for name, value in values.items():
            result[rows, feature_index[name]] = value
        geometry = (np.isfinite(close[selected]) & (close[selected] > 0)
                    & np.isfinite(atr[selected]) & np.isfinite(breakout[selected]))
        geometry_rows, geometry_pos = rows[geometry], selected[geometry]
        result[geometry_rows, feature_index['geometry_available']] = 1.
        result[geometry_rows, feature_index['atr_to_close']] = atr[geometry_pos] / close[geometry_pos]
        result[geometry_rows, feature_index['prior_breakout_high_to_close']] = breakout[geometry_pos] / close[geometry_pos] - 1.
        known = previous[selected] >= 0
        result[rows[known], feature_index['prior_setup_available']] = 1.
        result[rows[known], feature_index['log1p_sessions_since_setup']] = np.log1p(selected[known] - previous[selected[known]])
    if not np.isfinite(result).all():
        raise ValueError('nonfinite decision context')
    return result
