import copy

import pandas as pd

from core.v16_buy_sort import calc_buy_sort_value
from core.v16_config import BUY_SORT_METHOD
from core.v16_core import (
    calc_entry_price,
    calc_reference_candidate_qty,
    run_v16_backtest,
)
from core.v16_data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.v16_portfolio_engine import find_sim_start_idx

from .trade_rebuild import rebuild_completed_trades_from_portfolio_trade_log


FLOAT_TOL = 1e-6


def calc_validation_sim_years(sorted_dates, start_year):
    if not sorted_dates:
        return 0.0

    start_idx = find_sim_start_idx(sorted_dates, start_year)
    if start_idx >= len(sorted_dates):
        return 0.0

    first_dt = pd.Timestamp(sorted_dates[start_idx])
    last_dt = pd.Timestamp(sorted_dates[-1])
    span_days = (last_dt - first_dt).days + 1
    if span_days <= 0:
        return 0.0
    return span_days / 365.25

def calc_validation_annual_return_pct(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0

def normalize_ticker_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text[:-2].isdigit() and text.endswith(".0"):
        text = text[:-2]

    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)

    return text

def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))

def make_consistency_params(base_params):
    params = copy.deepcopy(base_params)
    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0
    return params

def add_check(results, module_name, ticker, metric, expected, actual, tol=FLOAT_TOL, note=""):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        diff = abs(float(expected) - float(actual))
        passed = diff <= tol
    else:
        diff = None
        passed = expected == actual

    status = "PASS" if passed else "FAIL"

    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "abs_diff": diff,
        "passed": passed,
        "status": status,
        "note": note
    })

def add_skip_result(results, module_name, ticker, metric, note):
    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": "SKIP",
        "actual": "SKIP",
        "abs_diff": None,
        "passed": True,
        "status": "SKIP",
        "note": note
    })

def add_fail_result(results, module_name, ticker, metric, expected, actual, note):
    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "abs_diff": None,
        "passed": False,
        "status": "FAIL",
        "note": note
    })

def build_execution_only_params(params):
    return make_consistency_params(params)

def normalize_yearly_return_rows(rows):
    normalized = []
    for row in rows or []:
        normalized.append({
            "year": int(row.get("year", 0)),
            "year_return_pct": float(row.get("year_return_pct", 0.0)),
            "is_full_year": bool(row.get("is_full_year", False)),
            "start_date": str(row.get("start_date", "")),
            "end_date": str(row.get("end_date", "")),
        })
    return normalized

def extract_yearly_profile_fields(profile_stats):
    yearly_rows = normalize_yearly_return_rows(profile_stats.get("yearly_return_rows", []))
    bm_yearly_rows = normalize_yearly_return_rows(profile_stats.get("bm_yearly_return_rows", []))
    return {
        "full_year_count": int(profile_stats.get("full_year_count", 0)),
        "min_full_year_return_pct": float(profile_stats.get("min_full_year_return_pct", 0.0)),
        "yearly_return_rows": yearly_rows,
        "bm_full_year_count": int(profile_stats.get("bm_full_year_count", 0)),
        "bm_min_full_year_return_pct": float(profile_stats.get("bm_min_full_year_return_pct", 0.0)),
        "bm_yearly_return_rows": bm_yearly_rows,
    }

def calc_expected_full_year_metrics(yearly_rows):
    full_year_rows = [row for row in (yearly_rows or []) if row["is_full_year"]]
    return {
        "full_year_count": len(full_year_rows),
        "min_full_year_return_pct": float(min((row["year_return_pct"] for row in full_year_rows), default=0.0)),
    }

def summarize_portfolio_trade_output(df_trades):
    portfolio_trade_types = df_trades["Type"].fillna("") if df_trades is not None and len(df_trades) > 0 and "Type" in df_trades.columns else pd.Series(dtype="object")
    portfolio_buy_rows = int(portfolio_trade_types.str.startswith("買進").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_full_exit_rows = int(portfolio_trade_types.isin({"全倉結算(停損)", "全倉結算(指標)", "期末強制結算", "汰弱賣出(Open, T+1再評估買進)"}).sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_half_take_profit_rows = int((portfolio_trade_types == "半倉停利").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_missed_buy_rows = int(portfolio_trade_types.str.startswith("錯失買進").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_missed_sell_rows = int((portfolio_trade_types == "錯失賣出").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_period_closeout_rows = int((portfolio_trade_types == "期末強制結算").sum()) if len(portfolio_trade_types) > 0 else 0
    portfolio_completed_trades = rebuild_completed_trades_from_portfolio_trade_log(df_trades)
    return {
        "portfolio_buy_rows": portfolio_buy_rows,
        "portfolio_full_exit_rows": portfolio_full_exit_rows,
        "portfolio_half_take_profit_rows": portfolio_half_take_profit_rows,
        "portfolio_missed_buy_rows": portfolio_missed_buy_rows,
        "portfolio_missed_sell_rows": portfolio_missed_sell_rows,
        "portfolio_period_closeout_rows": portfolio_period_closeout_rows,
        "portfolio_completed_trades": portfolio_completed_trades,
    }

def build_portfolio_stats_payload(*, module_path, df_trades, total_return, mdd, trade_count, win_rate, pf_ev, pf_payoff, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct, profile_stats):
    payload = {
        "module_path": module_path,
        "total_return": total_return,
        "mdd": mdd,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "final_eq": final_eq,
        "avg_exp": avg_exp,
        "max_exp": max_exp,
        "bm_ret": bm_ret,
        "bm_mdd": bm_mdd,
        "total_missed": total_missed,
        "total_missed_sells": total_missed_sells,
        "r_sq": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_sq": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "normal_trade_count": normal_trade_count,
        "extended_trade_count": extended_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "df_trades": df_trades.copy() if isinstance(df_trades, pd.DataFrame) else pd.DataFrame(),
    }
    payload.update(summarize_portfolio_trade_output(payload["df_trades"]))
    payload.update(extract_yearly_profile_fields(profile_stats or {}))
    return payload

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

def run_scanner_reference_check(ticker, file_path, params):
    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, _sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
        return run_v16_backtest(df.copy(), params)
    except ValueError as e:
        if is_insufficient_data_error(e):
            return {"scanner_expected_status": "skip_insufficient"}
        raise

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
