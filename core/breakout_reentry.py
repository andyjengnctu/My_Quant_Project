import math

import pandas as pd

from core.exact_accounting import milli_to_price
from core.extended_signals import (
    attach_breakout_quality_rank,
    clone_shadow_position,
    create_signal_tracking_state,
    resolve_breakout_quality_rank,
    resolve_signal_tracking_params,
)
from core.price_utils import adjust_long_buy_limit


BREAKOUT_REENTRY_SOURCE = "reentry"


def is_breakout_reentry_enabled(params):
    return bool(getattr(params, "use_breakout_reclaim_reentry", False))


def _finite_positive(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _positive_int(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _price_from_position(position, field_name, milli_field_name=None):
    if milli_field_name and position.get(milli_field_name) is not None:
        try:
            milli_value = int(position.get(milli_field_name))
        except (TypeError, ValueError):
            milli_value = 0
        if milli_value > 0:
            return milli_to_price(milli_value)
    return _finite_positive(position.get(field_name))


def _last_stop_context_qty(position):
    for ctx in reversed(position.get("_last_exec_contexts", []) or []):
        if ctx.get("event") != "STOP":
            continue
        qty = _positive_int(ctx.get("qty"))
        if qty is not None:
            return qty
    return None


def _resolve_reentry_exit_qty(position, exit_qty):
    explicit_qty = _positive_int(exit_qty)
    if explicit_qty is not None:
        return explicit_qty
    context_qty = _last_stop_context_qty(position)
    if context_qty is not None:
        return context_qty
    for field_name in ("qty_before_exit", "initial_qty"):
        qty = _positive_int(position.get(field_name))
        if qty is not None:
            return qty
    return None


def _sync_price_field_from_milli(state, price_field, milli_field):
    if state.get(milli_field) is None:
        return
    try:
        milli_value = int(state.get(milli_field))
    except (TypeError, ValueError):
        return
    if milli_value > 0:
        state[price_field] = milli_to_price(milli_value)


def _build_parent_management_shadow(position, *, parent_exit_qty):
    if parent_exit_qty is None or parent_exit_qty <= 0:
        return None
    shadow_position = clone_shadow_position(position)
    shadow_position["qty"] = int(parent_exit_qty)
    shadow_position["initial_qty"] = int(parent_exit_qty)
    shadow_position["pending_exit_action"] = None
    shadow_position["pending_exit_trigger_price"] = float("nan")
    shadow_position.pop("_last_exec_contexts", None)

    for price_field, milli_field in (
        ("sl", "sl_milli"),
        ("initial_stop", "initial_stop_milli"),
        ("trailing_stop", "trailing_stop_milli"),
        ("tp_half", "tp_half_milli"),
        ("highest_high_since_entry", "highest_high_since_entry_milli"),
    ):
        _sync_price_field_from_milli(shadow_position, price_field, milli_field)

    return shadow_position


def _resolve_stop_basis(position):
    effective_stop_milli = _positive_int(position.get("sl_milli"))
    trailing_stop_milli = _positive_int(position.get("trailing_stop_milli"))
    initial_stop_milli = _positive_int(position.get("initial_stop_milli"))
    if effective_stop_milli is not None and trailing_stop_milli is not None and effective_stop_milli == trailing_stop_milli:
        return "trailing"
    if effective_stop_milli is not None and initial_stop_milli is not None and effective_stop_milli == initial_stop_milli:
        return "initial"
    return "stop"


def _resolve_confirm_atr(params):
    return float(getattr(params, "breakout_reclaim_confirm_atr", getattr(params, "breakout_reclaim_confirm_r", 0.75)))


def create_breakout_reentry_watch_state(
    position,
    *,
    exit_date,
    params,
    exit_atr=None,
    exit_qty=None,
    quality_rank=None,
):
    if position is None or not is_breakout_reentry_enabled(params):
        return None

    entry_type = str(position.get("entry_type") or "normal")
    if entry_type not in {"normal", "extended", BREAKOUT_REENTRY_SOURCE}:
        return None

    entry_price = _finite_positive(position.get("entry_fill_price") or position.get("pure_buy_price"))
    initial_stop = _price_from_position(position, "initial_stop", "initial_stop_milli")
    exit_stop = _price_from_position(position, "sl", "sl_milli")
    resolved_exit_atr = _finite_positive(exit_atr)
    if entry_price is None or initial_stop is None or exit_stop is None or resolved_exit_atr is None:
        return None

    risk_per_share = entry_price - initial_stop
    if not math.isfinite(risk_per_share) or risk_per_share <= 0.0:
        return None

    confirm_atr = _resolve_confirm_atr(params)
    window_bars = int(getattr(params, "breakout_reclaim_window_bars", 20))
    if confirm_atr <= 0.0 or window_bars <= 0:
        return None

    parent_exit_qty = _resolve_reentry_exit_qty(position, exit_qty)
    parent_shadow_position = _build_parent_management_shadow(position, parent_exit_qty=parent_exit_qty)

    watch_state = {
        "source": BREAKOUT_REENTRY_SOURCE,
        "ticker": str(position.get("ticker") or ""),
        "security_profile": position.get("security_profile"),
        "parent_entry_type": entry_type,
        "parent_entry_trade_date": position.get("entry_trade_date"),
        "parent_exit_date": exit_date,
        "parent_entry_price": float(entry_price),
        "parent_initial_stop": float(initial_stop),
        "parent_risk_per_share": float(risk_per_share),
        "parent_exit_stop_price": float(exit_stop),
        "parent_exit_atr": float(resolved_exit_atr),
        "parent_exit_qty": None if parent_exit_qty is None else int(parent_exit_qty),
        "parent_stop_basis": _resolve_stop_basis(position),
        "confirm_price": float(exit_stop + resolved_exit_atr * confirm_atr),
        "confirm_atr": float(confirm_atr),
        "window_bars": int(window_bars),
        "bars_checked": 0,
        "last_checked_date": None,
        "parent_shadow_position": parent_shadow_position,
        "_params_obj": params,
    }

    inherited_rank = quality_rank
    if inherited_rank is None:
        inherited_rank = position.get("breakout_quality_rank")
    if inherited_rank is None and bool(position.get("use_breakout_quality_ranking", False)):
        raw_score = position.get("breakout_quality_score")
        try:
            parsed_score = float(raw_score)
        except (TypeError, ValueError):
            parsed_score = float("nan")
        score_available = bool(math.isfinite(parsed_score))
        inherited_rank = {
            "score": parsed_score if score_available else None,
            "available": score_available,
            "unavailable_reason": "" if score_available else "missing_score",
            "score_date": position.get("breakout_quality_score_date"),
            "score_source": position.get("breakout_quality_score_source") or "canonical_runtime",
            "shared_group_score": True,
            "filter_id": str(getattr(params, "breakout_quality_filter_id", "") or ""),
        }
    if inherited_rank is not None:
        attach_breakout_quality_rank(watch_state, inherited_rank)
    return watch_state


def _resolve_reentry_watch_params(state, fallback_params):
    return resolve_signal_tracking_params(state, fallback_params)


def _format_date_key(value):
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _watch_state_is_triggered(state, *, close_price):
    confirm_price = _finite_positive((state or {}).get("confirm_price"))
    close_value = _finite_positive(close_price)
    return confirm_price is not None and close_value is not None and close_value >= confirm_price


def _copy_reentry_watch_metadata(signal_state, state):
    for field_name in (
        "parent_entry_trade_date",
        "parent_exit_date",
        "parent_entry_price",
        "parent_initial_stop",
        "parent_risk_per_share",
        "parent_exit_stop_price",
        "parent_exit_atr",
        "parent_exit_qty",
        "parent_stop_basis",
        "confirm_atr",
    ):
        if field_name in state:
            signal_state[field_name] = state.get(field_name)
    signal_state["reentry_confirm_price"] = state.get("confirm_price")
    signal_state["max_qty"] = state.get("parent_exit_qty")

    parent_shadow_position = state.get("parent_shadow_position")
    if parent_shadow_position is not None:
        signal_state["shadow_position"] = clone_shadow_position(parent_shadow_position)
    inherited_rank = resolve_breakout_quality_rank(state)
    if inherited_rank is not None:
        attach_breakout_quality_rank(signal_state, inherited_rank)
    return signal_state


def create_breakout_reentry_signal_state(
    state,
    *,
    close_price,
    atr,
    params,
    ticker=None,
    security_profile=None,
    signal_date=None,
):
    if not isinstance(state, dict):
        return None

    state_params = _resolve_reentry_watch_params(state, params)
    if not is_breakout_reentry_enabled(state_params):
        return None
    if not _watch_state_is_triggered(state, close_price=close_price):
        return None
    if pd.isna(atr):
        return None

    resolved_ticker = ticker or state.get("ticker")
    resolved_security_profile = security_profile or state.get("security_profile")
    raw_limit = float(close_price) + float(atr) * float(getattr(state_params, "atr_buy_tol", 1.5))
    reentry_limit = adjust_long_buy_limit(raw_limit, ticker=resolved_ticker, security_profile=resolved_security_profile)
    signal_state = create_signal_tracking_state(
        reentry_limit,
        atr,
        state_params,
        ticker=resolved_ticker,
        security_profile=resolved_security_profile,
        signal_date=signal_date,
    )
    if signal_state is None:
        return None

    signal_state["source"] = BREAKOUT_REENTRY_SOURCE
    return _copy_reentry_watch_metadata(signal_state, state)


def activate_breakout_reentry_signals_for_day(
    *,
    active_reentry_watchlist,
    active_extended_signals,
    portfolio,
    sold_today,
    all_dfs_fast,
    today,
    params,
):
    if not active_reentry_watchlist:
        return 0

    from core.portfolio_fast_access import get_fast_close, get_fast_dates, get_fast_pos, get_fast_security_profile, get_fast_value

    activated = 0
    for ticker in sorted(list(active_reentry_watchlist.keys())):
        state = active_reentry_watchlist.get(ticker)
        if not isinstance(state, dict):
            active_reentry_watchlist.pop(ticker, None)
            continue

        state_params = _resolve_reentry_watch_params(state, params)
        if not is_breakout_reentry_enabled(state_params):
            active_reentry_watchlist.pop(ticker, None)
            continue

        if ticker in portfolio or ticker in active_extended_signals:
            active_reentry_watchlist.pop(ticker, None)
            continue
        if ticker in sold_today:
            continue

        fast_df = all_dfs_fast.get(ticker)
        if fast_df is None:
            continue

        t_pos = get_fast_pos(fast_df, today)
        if t_pos <= 0:
            continue
        y_pos = t_pos - 1
        dates = get_fast_dates(fast_df)
        signal_date = dates[y_pos]
        signal_date_key = _format_date_key(signal_date)
        if state.get("last_checked_date") == signal_date_key:
            continue

        y_close = get_fast_close(fast_df, pos=y_pos)
        y_atr = get_fast_value(fast_df, "ATR", pos=y_pos)
        state["last_checked_date"] = signal_date_key
        state["bars_checked"] = int(state.get("bars_checked", 0) or 0) + 1

        signal_state = create_breakout_reentry_signal_state(
            state,
            close_price=y_close,
            atr=y_atr,
            params=params,
            ticker=ticker,
            security_profile=get_fast_security_profile(fast_df),
            signal_date=signal_date,
        )
        if signal_state is not None:
            active_extended_signals[ticker] = signal_state
            active_reentry_watchlist.pop(ticker, None)
            activated += 1
            continue

        if int(state.get("bars_checked", 0) or 0) >= int(state.get("window_bars", 0) or 0):
            active_reentry_watchlist.pop(ticker, None)

    return activated
