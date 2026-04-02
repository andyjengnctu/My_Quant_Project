import copy

import pandas as pd

from core.buy_sort import calc_buy_sort_value
from core.config import BUY_SORT_METHOD
from core.backtest_core import calc_entry_price, calc_reference_candidate_qty, run_v16_backtest
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe

from .check_result_utils import is_insufficient_data_error, make_consistency_params


def build_execution_only_params(params):
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
        return run_v16_backtest(df.copy(), params)
    except ValueError as e:
        if is_insufficient_data_error(e):
            return {"scanner_expected_status": "skip_insufficient"}
        raise


def run_scanner_reference_check_on_clean_df(ticker, clean_df, params):
    if clean_df is None or clean_df.empty:
        raise ValueError(f"{ticker}: clean_df 不可為空")
    return run_v16_backtest(clean_df, params)


def derive_expected_scanner_status(scanner_ref_stats, params):
    if scanner_ref_stats.get("scanner_expected_status") == "skip_insufficient":
        return "skip_insufficient"

    if not scanner_ref_stats or not scanner_ref_stats["is_candidate"]:
        return None

    if scanner_ref_stats["is_setup_today"]:
        proj_qty = calc_reference_candidate_qty(
            scanner_ref_stats["buy_limit"],
            scanner_ref_stats["stop_loss"],
            params
        )
        return "buy" if proj_qty > 0 else "candidate"

    extended_candidate_today = scanner_ref_stats.get("extended_candidate_today")
    if extended_candidate_today is not None:
        limit_price = extended_candidate_today.get("limit_price")
        init_sl = extended_candidate_today.get("init_sl")
        if limit_price is None or init_sl is None:
            return "candidate"
        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params)
        return "extended" if proj_qty > 0 else "candidate"

    return "candidate"


def build_expected_scanner_payload(scanner_ref_stats, params):
    status = derive_expected_scanner_status(scanner_ref_stats, params)
    payload = {
        "status": status,
        "expected_value": None,
        "proj_cost": None,
        "sort_value": None,
    }

    if status not in ("buy", "extended"):
        return payload

    if status == "buy":
        limit_price = scanner_ref_stats["buy_limit"]
        stop_loss = scanner_ref_stats["stop_loss"]
        proj_qty = calc_reference_candidate_qty(limit_price, stop_loss, params)
    else:
        extended_candidate = scanner_ref_stats.get("extended_candidate_today")
        if extended_candidate is None:
            return payload
        limit_price = extended_candidate["limit_price"]
        init_sl = extended_candidate.get("init_sl")
        if init_sl is None:
            return payload
        proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params)

    proj_cost = calc_entry_price(limit_price, proj_qty, params) * proj_qty
    sort_value = calc_buy_sort_value(
        BUY_SORT_METHOD,
        scanner_ref_stats["expected_value"],
        proj_cost,
        scanner_ref_stats["win_rate"] / 100.0,
        scanner_ref_stats["trade_count"],
    )
    payload.update({
        "expected_value": scanner_ref_stats["expected_value"],
        "proj_cost": proj_cost,
        "sort_value": sort_value,
    })
    return payload
