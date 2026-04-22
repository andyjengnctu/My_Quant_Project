import copy

import pandas as pd

from core.backtest_core import run_v16_backtest
from core.buy_sort import calc_buy_sort_value
from core.config import BUY_SORT_METHOD
from core.exact_accounting import calc_entry_total_cost
from core.price_utils import calc_reference_candidate_qty
from core.data_utils import get_required_min_rows, resolve_latest_trade_date_from_frame, sanitize_ohlcv_dataframe

from .check_result_utils import is_insufficient_data_error, make_consistency_params


def build_consistency_parity_params(params):
    return make_consistency_params(params)


def make_synthetic_validation_params(base_params, *, tp_percent=None):
    params = make_consistency_params(base_params)
    params.high_len = 3
    params.atr_len = 2
    params.atr_buy_tol = 0.1
    params.atr_times_init = 1.0
    params.atr_times_trail = 1.5
    params.use_bb = False
    params.use_vol = False
    params.use_kc = False
    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0
    if tp_percent is not None:
        params.tp_percent = tp_percent
    return params


def build_scanner_validation_params(base_params):
    return copy.deepcopy(base_params)


def normalize_scanner_result(raw_result):
    if raw_result is None:
        return None

    if isinstance(raw_result, dict):
        return raw_result

    if not isinstance(raw_result, tuple) or len(raw_result) != 7:
        raise TypeError(
            f"scanner 回傳格式異常: type={type(raw_result).__name__}, value={raw_result}"
        )

    status, proj_cost, ev, sort_value, msg, ticker, sanitize_issue = raw_result
    return {
        "status": status,
        "proj_cost": proj_cost,
        "expected_value": ev,
        "sort_value": sort_value,
        "message": msg,
        "ticker": ticker,
        "sanitize_issue": sanitize_issue,
    }


def run_scanner_reference_check(ticker, file_path, params, *, raw_df=None):
    try:
        source_df = pd.read_csv(file_path) if raw_df is None else raw_df
        min_rows_needed = get_required_min_rows(params)
        df, _sanitize_stats = sanitize_ohlcv_dataframe(source_df, ticker, min_rows=min_rows_needed)
        stats = run_v16_backtest(df.copy(), params, ticker=ticker)
        stats["trade_date"] = resolve_latest_trade_date_from_frame(df)
        return stats
    except ValueError as e:
        if is_insufficient_data_error(e):
            return {"scanner_expected_status": "skip_insufficient"}
        raise


def run_scanner_reference_check_on_clean_df(ticker, clean_df, params):
    if clean_df is None or clean_df.empty:
        raise ValueError(f"{ticker}: clean_df 不可為空")
    stats = run_v16_backtest(clean_df, params, ticker=ticker)
    stats["trade_date"] = resolve_latest_trade_date_from_frame(clean_df)
    return stats


def rebuild_scanner_reference_stats_from_single_stats(single_stats, clean_df, params):
    if clean_df is None or clean_df.empty:
        raise ValueError("clean_df 不可為空")
    if not isinstance(single_stats, dict):
        raise TypeError("single_stats 必須為 dict")

    rebuilt = dict(single_stats)
    history_trade_count = int(single_stats.get("history_trade_count", 0) or 0)
    history_ev = float(single_stats.get("history_ev", 0.0) or 0.0)
    history_win_rate = float(single_stats.get("history_win_rate", 0.0) or 0.0)

    min_trades_req = int(getattr(params, "min_history_trades", 0) or 0)
    min_ev_req = float(getattr(params, "min_history_ev", 0.0) or 0.0)
    min_win_rate_req = float(getattr(params, "min_history_win_rate", 0.0) or 0.0)

    if history_trade_count < min_trades_req:
        scanner_expected_value = 0.0
        scanner_history_win_rate = 0.0
        scanner_is_candidate = False
    elif history_trade_count == 0:
        allow_zero_history = (
            (min_trades_req == 0)
            and (min_ev_req <= 0)
            and (min_win_rate_req <= 0)
        )
        scanner_expected_value = 0.0
        scanner_history_win_rate = 0.0
        scanner_is_candidate = bool(allow_zero_history)
    else:
        scanner_expected_value = history_ev
        scanner_history_win_rate = history_win_rate
        scanner_is_candidate = (
            scanner_history_win_rate >= min_win_rate_req
            and scanner_expected_value >= min_ev_req
        )

    rebuilt["expected_value"] = scanner_expected_value
    rebuilt["history_ev"] = scanner_expected_value
    rebuilt["history_win_rate"] = scanner_history_win_rate
    rebuilt["history_trade_count"] = history_trade_count
    rebuilt["is_candidate"] = scanner_is_candidate
    rebuilt["trade_date"] = resolve_latest_trade_date_from_frame(clean_df)
    return rebuilt


def derive_expected_scanner_status(scanner_ref_stats, params, *, ticker=None, trade_date=None):
    if scanner_ref_stats.get("scanner_expected_status") == "skip_insufficient":
        return "skip_insufficient"

    if not scanner_ref_stats or not scanner_ref_stats["is_candidate"]:
        return None

    has_terminal_position = bool(scanner_ref_stats.get("hasOpenPositionAtEnd")) or int(scanner_ref_stats.get("current_position", 0) or 0) > 0

    if scanner_ref_stats["is_setup_today"]:
        if has_terminal_position:
            return None
        proj_qty = calc_reference_candidate_qty(
            scanner_ref_stats["buy_limit"],
            scanner_ref_stats["stop_loss"],
            params,
            ticker=ticker,
            trade_date=trade_date,
        )
        return "buy" if proj_qty > 0 else "candidate"

    extended_candidate_today = scanner_ref_stats.get("extended_candidate_today")
    if extended_candidate_today is not None:
        if not bool(scanner_ref_stats.get("extended_orderable_today", True)):
            return "candidate"
        limit_price = extended_candidate_today.get("limit_price")
        init_sl = extended_candidate_today.get("init_sl")
        if limit_price is None or init_sl is None:
            return "candidate"
        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params, ticker=ticker, trade_date=trade_date)
        return "extended" if proj_qty > 0 else "candidate"

    extended_candidate_tbd_today = scanner_ref_stats.get("extended_candidate_tbd_today")
    if extended_candidate_tbd_today is not None:
        if not bool(scanner_ref_stats.get("extended_tbd_orderable_today", True)):
            return "candidate"
        limit_price = extended_candidate_tbd_today.get("limit_price")
        init_sl = extended_candidate_tbd_today.get("init_sl")
        if limit_price is None or init_sl is None:
            return "candidate"
        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params, ticker=ticker, trade_date=trade_date)
        return "extended_tbd" if proj_qty > 0 else "candidate"

    if has_terminal_position:
        return None

    return "candidate"


def build_expected_scanner_payload(scanner_ref_stats, params, *, ticker=None, trade_date=None):
    status = derive_expected_scanner_status(scanner_ref_stats, params, ticker=ticker, trade_date=trade_date)
    payload = {
        "status": status,
        "expected_value": None,
        "proj_cost": None,
        "sort_value": None,
    }

    if status not in ("buy", "extended", "extended_tbd"):
        return payload

    if status == "buy":
        limit_price = scanner_ref_stats["buy_limit"]
        stop_loss = scanner_ref_stats["stop_loss"]
        proj_qty = calc_reference_candidate_qty(limit_price, stop_loss, params, ticker=ticker, trade_date=trade_date)
    elif status == "extended":
        extended_candidate = scanner_ref_stats.get("extended_candidate_today")
        if extended_candidate is None:
            return payload
        limit_price = extended_candidate["limit_price"]
        init_sl = extended_candidate.get("init_sl")
        if init_sl is None:
            return payload
        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params, ticker=ticker, trade_date=trade_date)
    else:
        extended_candidate_tbd = scanner_ref_stats.get("extended_candidate_tbd_today")
        if extended_candidate_tbd is None:
            return payload
        limit_price = extended_candidate_tbd["limit_price"]
        init_sl = extended_candidate_tbd.get("init_sl")
        if init_sl is None:
            return payload
        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params, ticker=ticker, trade_date=trade_date)

    proj_cost = calc_entry_total_cost(limit_price, proj_qty, params)
    sort_value = calc_buy_sort_value(
        BUY_SORT_METHOD,
        scanner_ref_stats["expected_value"],
        proj_cost,
        scanner_ref_stats["win_rate"] / 100.0,
        scanner_ref_stats["trade_count"],
        scanner_ref_stats.get("asset_growth", 0.0),
    )
    payload.update({
        "expected_value": scanner_ref_stats["expected_value"],
        "proj_cost": proj_cost,
        "sort_value": sort_value,
    })
    return payload
