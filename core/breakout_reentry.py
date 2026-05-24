import math

import pandas as pd

from core.extended_signals import create_signal_tracking_state
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


def create_breakout_reentry_watch_state(position, *, exit_date, params):
    if position is None or not is_breakout_reentry_enabled(params):
        return None

    entry_type = str(position.get("entry_type") or "normal")
    if entry_type not in {"normal", "extended", BREAKOUT_REENTRY_SOURCE}:
        return None

    entry_price = _finite_positive(position.get("entry_fill_price") or position.get("pure_buy_price"))
    initial_stop = _finite_positive(position.get("initial_stop"))
    if entry_price is None or initial_stop is None:
        return None

    risk_per_share = entry_price - initial_stop
    if not math.isfinite(risk_per_share) or risk_per_share <= 0.0:
        return None

    confirm_r = float(getattr(params, "breakout_reclaim_confirm_r", 0.75))
    window_bars = int(getattr(params, "breakout_reclaim_window_bars", 20))
    if confirm_r <= 0.0 or window_bars <= 0:
        return None

    return {
        "source": BREAKOUT_REENTRY_SOURCE,
        "ticker": str(position.get("ticker") or ""),
        "security_profile": position.get("security_profile"),
        "parent_entry_type": entry_type,
        "parent_entry_trade_date": position.get("entry_trade_date"),
        "parent_exit_date": exit_date,
        "parent_entry_price": float(entry_price),
        "parent_initial_stop": float(initial_stop),
        "parent_risk_per_share": float(risk_per_share),
        "confirm_price": float(entry_price + risk_per_share * confirm_r),
        "window_bars": int(window_bars),
        "bars_checked": 0,
        "last_checked_date": None,
    }


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
    if not active_reentry_watchlist or not is_breakout_reentry_enabled(params):
        return 0

    from core.portfolio_fast_data import get_fast_close, get_fast_dates, get_fast_pos, get_fast_security_profile, get_fast_value

    activated = 0
    for ticker in sorted(list(active_reentry_watchlist.keys())):
        state = active_reentry_watchlist.get(ticker)
        if not isinstance(state, dict):
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

        if _watch_state_is_triggered(state, close_price=y_close) and not pd.isna(y_atr):
            raw_limit = float(y_close) + float(y_atr) * float(getattr(params, "atr_buy_tol", 1.5))
            reentry_limit = adjust_long_buy_limit(raw_limit, ticker=ticker, security_profile=get_fast_security_profile(fast_df))
            signal_state = create_signal_tracking_state(
                reentry_limit,
                y_atr,
                params,
                ticker=ticker,
                security_profile=get_fast_security_profile(fast_df),
                signal_date=signal_date,
            )
            if signal_state is not None:
                signal_state["source"] = BREAKOUT_REENTRY_SOURCE
                signal_state["parent_entry_trade_date"] = state.get("parent_entry_trade_date")
                signal_state["parent_exit_date"] = state.get("parent_exit_date")
                signal_state["parent_entry_price"] = state.get("parent_entry_price")
                signal_state["parent_initial_stop"] = state.get("parent_initial_stop")
                signal_state["reentry_confirm_price"] = state.get("confirm_price")
                active_extended_signals[ticker] = signal_state
                active_reentry_watchlist.pop(ticker, None)
                activated += 1
                continue

        if int(state.get("bars_checked", 0) or 0) >= int(state.get("window_bars", 0) or 0):
            active_reentry_watchlist.pop(ticker, None)

    return activated
