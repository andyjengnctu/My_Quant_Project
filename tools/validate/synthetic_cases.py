import os
import tempfile

import pandas as pd

from core.v16_config import V16StrategyParams
from core.v16_core import (
    build_cash_capped_entry_plan,
    calc_entry_price,
    calc_reference_candidate_qty,
    can_execute_half_take_profit,
    evaluate_history_candidate_metrics,
    resize_candidate_plan_to_capital,
    run_v16_backtest,
)
from core.v16_data_utils import (
    discover_unique_csv_inputs,
    get_required_min_rows,
    sanitize_ohlcv_dataframe,
)
from core.v16_params_io import build_params_from_mapping, params_to_json_dict
from core.v16_portfolio_engine import run_portfolio_timeline
from core.v16_portfolio_fast_data import pack_prepared_stock_data, prep_stock_data_and_trades

from .checks import (
    add_check,
    add_fail_result,
    add_skip_result,
    build_expected_scanner_payload,
    build_portfolio_stats_payload,
    build_scanner_validation_params,
    make_consistency_params,
    make_synthetic_validation_params,
    normalize_ticker_text,
    run_scanner_reference_check,
)
from .synthetic_fixtures import (
    build_synthetic_competing_candidates_case as fixture_build_synthetic_competing_candidates_case,
    build_synthetic_extended_miss_buy_case as fixture_build_synthetic_extended_miss_buy_case,
    build_synthetic_half_tp_full_year_case as fixture_build_synthetic_half_tp_full_year_case,
    build_synthetic_param_guardrail_case as fixture_build_synthetic_param_guardrail_case,
    build_synthetic_rotation_t_plus_one_case as fixture_build_synthetic_rotation_t_plus_one_case,
    build_synthetic_same_day_sell_block_case as fixture_build_synthetic_same_day_sell_block_case,
    build_synthetic_unexecutable_half_tp_case as fixture_build_synthetic_unexecutable_half_tp_case,
    write_synthetic_csv_bundle,
)
from .tool_adapters import (
    run_debug_trade_log_check,
    run_portfolio_sim_tool_check_for_dir,
    run_scanner_tool_check,
)

def build_synthetic_half_tp_full_year_case(base_params):
    return fixture_build_synthetic_half_tp_full_year_case(base_params, make_synthetic_validation_params)


def build_synthetic_extended_miss_buy_case(base_params):
    return fixture_build_synthetic_extended_miss_buy_case(base_params, make_synthetic_validation_params)


def build_synthetic_competing_candidates_case(base_params):
    return fixture_build_synthetic_competing_candidates_case(base_params, make_synthetic_validation_params)


def build_synthetic_same_day_sell_block_case(base_params):
    return fixture_build_synthetic_same_day_sell_block_case(base_params, make_synthetic_validation_params)


def build_synthetic_unexecutable_half_tp_case(base_params):
    return fixture_build_synthetic_unexecutable_half_tp_case(base_params, make_synthetic_validation_params)


def build_synthetic_rotation_t_plus_one_case(base_params):
    return fixture_build_synthetic_rotation_t_plus_one_case(base_params, make_synthetic_validation_params)


def build_synthetic_param_guardrail_case(base_params):
    return fixture_build_synthetic_param_guardrail_case(base_params, lambda p: params_to_json_dict(make_consistency_params(p)))


def run_portfolio_core_check_for_dir(data_dir, params, *, max_positions, enable_rotation, start_year, benchmark_ticker):
    csv_inputs, _duplicate_issue_lines = discover_unique_csv_inputs(data_dir)
    if not csv_inputs:
        raise ValueError(f"synthetic data_dir 無任何 CSV: {data_dir}")

    all_dfs_fast = {}
    all_trade_logs = {}
    master_dates = set()

    for ticker, file_path in csv_inputs:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, _sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
        prep_df, standalone_logs = prep_stock_data_and_trades(df, params)
        fast_df = pack_prepared_stock_data(prep_df)
        all_dfs_fast[ticker] = fast_df
        all_trade_logs[ticker] = standalone_logs
        master_dates.update(prep_df.index)

    sorted_dates = sorted(master_dates)
    if not sorted_dates:
        raise ValueError(f"{data_dir}: synthetic prep 後沒有任何有效日期")

    benchmark_data = all_dfs_fast.get(benchmark_ticker)
    profile_stats = {}
    result = run_portfolio_timeline(
        all_dfs_fast=all_dfs_fast,
        all_standalone_logs=all_trade_logs,
        sorted_dates=sorted_dates,
        start_year=start_year,
        params=params,
        max_positions=max_positions,
        enable_rotation=enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=profile_stats,
        verbose=False,
    )

    (
        _df_eq,
        _df_tr,
        tot_ret,
        mdd,
        trade_count,
        win_rate,
        pf_ev,
        pf_payoff,
        final_eq,
        avg_exp,
        max_exp,
        bm_ret,
        bm_mdd,
        total_missed,
        total_missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trade_count,
        extended_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
    ) = result

    return build_portfolio_stats_payload(
        module_path="core/v16_portfolio_engine.py",
        df_trades=_df_tr,
        total_return=tot_ret,
        mdd=mdd,
        trade_count=trade_count,
        win_rate=win_rate,
        pf_ev=pf_ev,
        pf_payoff=pf_payoff,
        final_eq=final_eq,
        avg_exp=avg_exp,
        max_exp=max_exp,
        bm_ret=bm_ret,
        bm_mdd=bm_mdd,
        total_missed=total_missed,
        total_missed_sells=total_missed_sells,
        r_sq=r_sq,
        m_win_rate=m_win_rate,
        bm_r_sq=bm_r_sq,
        bm_m_win_rate=bm_m_win_rate,
        normal_trade_count=normal_trade_count,
        extended_trade_count=extended_trade_count,
        annual_trades=annual_trades,
        reserved_buy_fill_rate=reserved_buy_fill_rate,
        annual_return_pct=annual_return_pct,
        bm_annual_return_pct=bm_annual_return_pct,
        profile_stats=profile_stats,
    )

def add_portfolio_stats_equality_checks(results, module_name, ticker, expected_stats, actual_stats):
    metric_names = [
        "total_return", "mdd", "trade_count", "win_rate", "pf_ev", "pf_payoff",
        "final_eq", "avg_exp", "max_exp", "bm_ret", "bm_mdd", "total_missed",
        "total_missed_sells", "r_sq", "m_win_rate", "bm_r_sq", "bm_m_win_rate",
        "normal_trade_count", "extended_trade_count", "annual_trades",
        "reserved_buy_fill_rate", "annual_return_pct", "bm_annual_return_pct",
        "full_year_count", "min_full_year_return_pct", "yearly_return_rows",
        "bm_full_year_count", "bm_min_full_year_return_pct", "bm_yearly_return_rows",
        "portfolio_buy_rows", "portfolio_full_exit_rows", "portfolio_half_take_profit_rows",
        "portfolio_missed_buy_rows", "portfolio_missed_sell_rows", "portfolio_period_closeout_rows",
    ]
    for metric in metric_names:
        add_check(results, module_name, ticker, metric, expected_stats[metric], actual_stats[metric])

    add_check(results, module_name, ticker, "portfolio_completed_trade_count", len(expected_stats["portfolio_completed_trades"]), len(actual_stats["portfolio_completed_trades"]))
    add_check(
        results,
        module_name,
        ticker,
        "portfolio_completed_trade_exit_dates",
        [trade["exit_date"] for trade in expected_stats["portfolio_completed_trades"]],
        [trade["exit_date"] for trade in actual_stats["portfolio_completed_trades"]],
    )
    add_check(
        results,
        module_name,
        ticker,
        "portfolio_completed_trade_pnl_sequence",
        [trade["total_pnl"] for trade in expected_stats["portfolio_completed_trades"]],
        [trade["total_pnl"] for trade in actual_stats["portfolio_completed_trades"]],
    )

def validate_synthetic_half_tp_full_year_case(base_params):
    case = build_synthetic_half_tp_full_year_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        add_portfolio_stats_equality_checks(results, "synthetic_half_tp_full_year", case["case_id"], core_stats, sim_stats)
        add_check(results, "synthetic_half_tp_full_year", case["case_id"], "expected_half_take_profit_rows", 1, sim_stats["portfolio_half_take_profit_rows"])
        add_check(results, "synthetic_half_tp_full_year", case["case_id"], "has_yearly_return_rows", True, bool(sim_stats["yearly_return_rows"]))

        primary_path = os.path.join(temp_dir, f"{case['primary_ticker']}.csv")
        debug_raw_df = pd.read_csv(primary_path)
        debug_input_df, _debug_sanitize_stats = sanitize_ohlcv_dataframe(
            debug_raw_df,
            case["primary_ticker"],
            min_rows=get_required_min_rows(case["params"]),
        )
        debug_df, _debug_module_path = run_debug_trade_log_check(case["primary_ticker"], debug_input_df, case["params"])
        half_rows = int((debug_df["動作"].fillna("") == "半倉停利").sum()) if debug_df is not None and len(debug_df) > 0 else 0
        add_check(results, "synthetic_half_tp_full_year", case["case_id"], "debug_half_take_profit_rows", 1, half_rows)

    summary["half_take_profit_rows"] = 1
    summary["full_year_count"] = True
    return results, summary

def validate_synthetic_extended_miss_buy_case(base_params):
    case = build_synthetic_extended_miss_buy_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        add_portfolio_stats_equality_checks(results, "synthetic_extended_miss_buy", case["case_id"], core_stats, sim_stats)
        add_check(results, "synthetic_extended_miss_buy", case["case_id"], "expected_trade_count", 0, sim_stats["trade_count"])
        add_check(results, "synthetic_extended_miss_buy", case["case_id"], "expected_missed_buy_count", 1, sim_stats["total_missed"])
        add_check(results, "synthetic_extended_miss_buy", case["case_id"], "expected_df_trades_missed_buy_rows", 1, sim_stats["portfolio_missed_buy_rows"])

        primary_ticker = case["primary_ticker"]
        primary_path = os.path.join(temp_dir, f"{primary_ticker}.csv")
        scanner_ref_stats = run_scanner_reference_check(primary_ticker, primary_path, case["params"])
        scanner_result, _scanner_module_path = run_scanner_tool_check(primary_ticker, primary_path, case["params"])
        expected_payload = build_expected_scanner_payload(scanner_ref_stats, case["params"])
        add_check(results, "synthetic_extended_miss_buy", case["case_id"], "scanner_expected_status", "extended", expected_payload["status"])
        add_check(results, "synthetic_extended_miss_buy", case["case_id"], "scanner_tool_status", "extended", None if scanner_result is None else scanner_result["status"])
        add_check(results, "synthetic_extended_miss_buy", case["case_id"], "has_extended_candidate_today", True, bool(scanner_ref_stats.get("extended_candidate_today") is not None))

    summary["extended_candidate"] = True
    summary["missed_buy"] = True
    return results, summary

def validate_synthetic_competing_candidates_case(base_params):
    case = build_synthetic_competing_candidates_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        add_portfolio_stats_equality_checks(results, "synthetic_competing_candidates", case["case_id"], core_stats, sim_stats)

        buy_df = sim_stats["df_trades"]
        buy_rows = buy_df[buy_df["Type"].fillna("").str.startswith("買進")].copy() if not buy_df.empty else pd.DataFrame()
        selected_ticker = buy_rows.iloc[0]["Ticker"] if len(buy_rows) > 0 else None
        rejected_buy_count = int(((buy_df["Ticker"] == "9401") & buy_df["Type"].fillna("").str.startswith("買進")).sum()) if not buy_df.empty else 0
        add_check(results, "synthetic_competing_candidates", case["case_id"], "selected_buy_row_count", 1, len(buy_rows))
        add_check(results, "synthetic_competing_candidates", case["case_id"], "selected_ticker_when_sort_ties", "9402", selected_ticker)
        add_check(results, "synthetic_competing_candidates", case["case_id"], "non_selected_ticker_has_no_buy_row", 0, rejected_buy_count)

    summary["selected_ticker"] = "9402"
    return results, summary

def validate_synthetic_same_day_sell_block_case(base_params):
    case = build_synthetic_same_day_sell_block_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        add_portfolio_stats_equality_checks(results, "synthetic_same_day_sell_block", case["case_id"], core_stats, sim_stats)

        df_trades = sim_stats["df_trades"]
        sell_rows = df_trades[(df_trades["Ticker"] == "9501") & (df_trades["Type"].fillna("").isin(["全倉結算(停損)", "全倉結算(指標)"]))].copy() if not df_trades.empty else pd.DataFrame()
        sell_date = sell_rows.iloc[0]["Date"] if len(sell_rows) > 0 else None
        blocked_same_day_buy = bool(((df_trades["Date"] == sell_date) & (df_trades["Ticker"] == "9502") & df_trades["Type"].fillna("").str.startswith("買進")).any()) if sell_date is not None else None
        later_buy_rows = df_trades[(df_trades["Ticker"] == "9502") & df_trades["Type"].fillna("").str.startswith("買進")].copy() if not df_trades.empty else pd.DataFrame()
        later_buy_date = later_buy_rows.iloc[0]["Date"] if len(later_buy_rows) > 0 else None

        add_check(results, "synthetic_same_day_sell_block", case["case_id"], "has_sell_row", True, sell_date is not None)
        add_check(results, "synthetic_same_day_sell_block", case["case_id"], "same_day_sell_blocks_new_buy", False, blocked_same_day_buy)
        add_check(results, "synthetic_same_day_sell_block", case["case_id"], "later_reentry_is_next_day_or_after", True, (later_buy_date is not None and later_buy_date > sell_date) if sell_date is not None else False)

    summary["same_day_sell_block"] = True
    return results, summary

def validate_synthetic_unexecutable_half_tp_case(base_params):
    case = build_synthetic_unexecutable_half_tp_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, case["params"], max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"], start_year=case["start_year"], benchmark_ticker=case["benchmark_ticker"]
        )
        add_portfolio_stats_equality_checks(results, "synthetic_unexecutable_half_tp", case["case_id"], core_stats, sim_stats)
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "expected_half_take_profit_rows", 0, sim_stats["portfolio_half_take_profit_rows"])

        primary_ticker = case["primary_ticker"]
        primary_path = os.path.join(temp_dir, f"{primary_ticker}.csv")
        debug_raw_df = pd.read_csv(primary_path)
        debug_input_df, _debug_sanitize_stats = sanitize_ohlcv_dataframe(
            debug_raw_df,
            primary_ticker,
            min_rows=get_required_min_rows(case["params"]),
        )
        debug_df, _debug_module_path = run_debug_trade_log_check(primary_ticker, debug_input_df, case["params"])
        half_rows = int((debug_df["動作"].fillna("") == "半倉停利").sum()) if debug_df is not None and len(debug_df) > 0 else 0
        buy_rows = debug_df[debug_df["動作"].fillna("").str.startswith("買進")].copy() if debug_df is not None and len(debug_df) > 0 else pd.DataFrame()
        half_tp_price_is_nan = bool(buy_rows["半倉停利價"].isna().all()) if len(buy_rows) > 0 else False
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "debug_half_take_profit_rows", 0, half_rows)
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "debug_buy_row_half_tp_price_is_nan", True, half_tp_price_is_nan)

    scanner_case = build_synthetic_half_tp_full_year_case(base_params)
    scanner_case["params"].initial_capital = 130.0
    scanner_case["params"].fixed_risk = 1.0
    scanner_case["frames"][scanner_case["primary_ticker"]] = scanner_case["frames"][scanner_case["primary_ticker"]].iloc[:271].copy()

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, scanner_case["frames"])
        scanner_ticker = scanner_case["primary_ticker"]
        scanner_path = os.path.join(temp_dir, f"{scanner_ticker}.csv")
        scanner_ref_stats = run_scanner_reference_check(scanner_ticker, scanner_path, scanner_case["params"])
        scanner_result, _scanner_module_path = run_scanner_tool_check(scanner_ticker, scanner_path, scanner_case["params"])
        proj_qty = calc_reference_candidate_qty(scanner_ref_stats["buy_limit"], scanner_ref_stats["stop_loss"], scanner_case["params"]) if scanner_ref_stats.get("is_setup_today") else 0
        scanner_message = "" if scanner_result is None or scanner_result.get("message") is None else str(scanner_result.get("message"))
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_projected_qty", 1, proj_qty)
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_half_tp_executable", False, can_execute_half_take_profit(proj_qty, scanner_case["params"].tp_percent))
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_tool_status", "buy", None if scanner_result is None else scanner_result["status"])
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_message_marks_unexecutable_half_tp", True, "半倉停利:股數不足" in scanner_message)

    summary["half_take_profit_rows"] = 0
    summary["projected_qty"] = 1
    return results, summary

def validate_synthetic_rotation_t_plus_one_case(base_params):
    case = build_synthetic_rotation_t_plus_one_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    rotation_sell_type = "汰弱賣出(Open, T+1再評估買進)"

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir,
            case["params"],
            max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"],
            start_year=case["start_year"],
            benchmark_ticker=case["benchmark_ticker"],
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir,
            case["params"],
            max_positions=case["max_positions"],
            enable_rotation=case["enable_rotation"],
            start_year=case["start_year"],
            benchmark_ticker=case["benchmark_ticker"],
        )
        add_portfolio_stats_equality_checks(results, "synthetic_rotation_t_plus_one", case["case_id"], core_stats, sim_stats)

        df_trades = sim_stats["df_trades"].copy()
        if df_trades.empty:
            add_fail_result(
                results,
                "synthetic_rotation_t_plus_one",
                case["case_id"],
                "df_trades_exists",
                "non-empty",
                "empty",
                "rotation synthetic case 應產生 trade history。"
            )
            return results, summary

        rotation_rows = df_trades[
            (df_trades["Ticker"] == case["weak_ticker"]) &
            (df_trades["Type"].fillna("") == rotation_sell_type)
        ].copy()
        rotation_sell_date = rotation_rows.iloc[0]["Date"] if len(rotation_rows) > 0 else None

        same_day_reentry = bool(
            (
                (df_trades["Date"] == rotation_sell_date) &
                (df_trades["Ticker"] == case["strong_ticker"]) &
                df_trades["Type"].fillna("").str.startswith("買進")
            ).any()
        ) if rotation_sell_date is not None else False

        post_rotation_df = (
            df_trades[pd.to_datetime(df_trades["Date"]) > pd.to_datetime(rotation_sell_date)].copy()
            if rotation_sell_date is not None else pd.DataFrame()
        )

        extended_miss_rows = post_rotation_df[
            (post_rotation_df["Ticker"] == case["strong_ticker"]) &
            (post_rotation_df["Type"].fillna("") == "錯失買進(延續候選)")
        ].copy()
        extended_miss_date = extended_miss_rows.iloc[0]["Date"] if len(extended_miss_rows) > 0 else None

        delayed_buy_rows = post_rotation_df[
            (post_rotation_df["Ticker"] == case["strong_ticker"]) &
            post_rotation_df["Type"].fillna("").str.startswith("買進")
        ].copy()
        delayed_buy_date = delayed_buy_rows.iloc[0]["Date"] if len(delayed_buy_rows) > 0 else None

        add_check(results, "synthetic_rotation_t_plus_one", case["case_id"], "rotation_sell_row_count", 1, len(rotation_rows))
        add_check(results, "synthetic_rotation_t_plus_one", case["case_id"], "rotation_same_day_reentry_blocked", False, same_day_reentry)
        add_check(results, "synthetic_rotation_t_plus_one", case["case_id"], "rotation_has_post_sell_extended_miss_buy", True, extended_miss_date is not None)
        add_check(
            results,
            "synthetic_rotation_t_plus_one",
            case["case_id"],
            "rotation_extended_miss_occurs_after_sell",
            True,
            (extended_miss_date is not None and rotation_sell_date is not None and extended_miss_date > rotation_sell_date)
        )
        add_check(
            results,
            "synthetic_rotation_t_plus_one",
            case["case_id"],
            "rotation_delayed_buy_occurs_after_sell",
            True,
            (delayed_buy_date is not None and rotation_sell_date is not None and delayed_buy_date > rotation_sell_date)
        )
        add_check(
            results,
            "synthetic_rotation_t_plus_one",
            case["case_id"],
            "rotation_delayed_buy_occurs_after_extended_miss",
            True,
            (delayed_buy_date is not None and extended_miss_date is not None and delayed_buy_date > extended_miss_date)
        )

    summary["rotation_sell"] = True
    summary["delayed_reentry"] = True
    return results, summary

def validate_synthetic_history_ev_threshold_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.min_history_trades = 1
    params.min_history_ev = 0.5
    params.min_history_win_rate = 0.5

    case_id = "SYNTH_HISTORY_EV_THRESHOLD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    is_candidate, expected_value, win_rate, trade_count = evaluate_history_candidate_metrics(
        trade_count=1,
        win_count=1,
        total_r_sum=0.5,
        win_r_sum=0.5,
        loss_r_sum=0.0,
        params=params,
    )

    add_check(results, "synthetic_history_ev_threshold", case_id, "expected_value_equals_threshold", 0.5, expected_value)
    add_check(results, "synthetic_history_ev_threshold", case_id, "win_rate_equals_threshold", 1.0, win_rate)
    add_check(results, "synthetic_history_ev_threshold", case_id, "trade_count_preserved", 1, trade_count)
    add_check(results, "synthetic_history_ev_threshold", case_id, "candidate_allowed_when_ev_equals_threshold", True, is_candidate)

    summary["candidate_allowed"] = bool(is_candidate)
    summary["expected_value"] = expected_value
    return results, summary

def validate_synthetic_proj_cost_cash_capped_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.initial_capital = 1_000_000.0
    params.fixed_risk = 0.01

    case_id = "SYNTH_PROJ_COST_CASH_CAPPED_ORDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    sizing_equity = 1_000_000.0
    available_cash = 50_000.0
    candidate_rows = [
        {"ticker": "9801", "limit_px": 10.0, "init_sl": 9.5, "init_trail": 9.0},
        {"ticker": "9802", "limit_px": 14.0, "init_sl": 13.3, "init_trail": 12.8},
    ]

    estimated_rank_rows = []
    for cand in candidate_rows:
        est_plan = resize_candidate_plan_to_capital(
            {
                "limit_price": cand["limit_px"],
                "init_sl": cand["init_sl"],
                "init_trail": cand["init_trail"],
            },
            sizing_equity,
            params,
        )
        est_reserved_cost = calc_entry_price(est_plan["limit_price"], est_plan["qty"], params) * est_plan["qty"]
        estimated_rank_rows.append({
            "ticker": cand["ticker"],
            "reserved_cost": est_reserved_cost,
        })

    estimated_rank_rows.sort(key=lambda x: (x["reserved_cost"], x["ticker"]), reverse=True)
    stale_top_ticker = estimated_rank_rows[0]["ticker"]

    cash_capped_rank_rows = []
    for cand in candidate_rows:
        cash_capped_plan = build_cash_capped_entry_plan(
            {
                "limit_price": cand["limit_px"],
                "init_sl": cand["init_sl"],
                "init_trail": cand["init_trail"],
            },
            available_cash,
            params,
        )
        if cash_capped_plan is None:
            continue

        cash_capped_rank_rows.append({
            "ticker": cand["ticker"],
            "reserved_cost": cash_capped_plan["reserved_cost"],
            "qty": cash_capped_plan["qty"],
        })

    cash_capped_rank_rows.sort(key=lambda x: (x["reserved_cost"], x["ticker"]), reverse=True)
    cash_capped_top_ticker = cash_capped_rank_rows[0]["ticker"] if cash_capped_rank_rows else None
    cash_capped_top_reserved_cost = cash_capped_rank_rows[0]["reserved_cost"] if cash_capped_rank_rows else None
    cash_capped_top_qty = cash_capped_rank_rows[0]["qty"] if cash_capped_rank_rows else None

    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "stale_proj_cost_top_ticker",
        "9802",
        stale_top_ticker,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "cash_capped_proj_cost_top_ticker",
        "9801",
        cash_capped_top_ticker,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "proj_cost_order_reversal_detected",
        True,
        stale_top_ticker != cash_capped_top_ticker,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "cash_capped_reserved_cost_within_available_cash",
        True,
        cash_capped_top_reserved_cost is not None and cash_capped_top_reserved_cost <= available_cash,
    )
    add_check(
        results,
        "synthetic_proj_cost_cash_capped",
        case_id,
        "cash_capped_qty_positive",
        True,
        cash_capped_top_qty is not None and cash_capped_top_qty > 0,
    )

    summary["stale_top_ticker"] = stale_top_ticker
    summary["cash_capped_top_ticker"] = cash_capped_top_ticker
    summary["available_cash"] = available_cash
    return results, summary

def validate_synthetic_param_guardrail_case(base_params):
    case = build_synthetic_param_guardrail_case(base_params)
    results = []
    summary = {"ticker": case["case_id"], "synthetic": True}

    valid_params = build_params_from_mapping(case["base_payload"])
    add_check(results, "synthetic_param_guardrail", case["case_id"], "valid_payload_loads", True, isinstance(valid_params, V16StrategyParams))

    invalid_cases = [
        ("tp_percent_ge_1_rejected", {**case["base_payload"], "tp_percent": 1.0}, "tp_percent"),
        ("fixed_risk_zero_rejected", {**case["base_payload"], "fixed_risk": 0.0}, "fixed_risk"),
        ("min_history_win_rate_gt_1_rejected", {**case["base_payload"], "min_history_win_rate": 1.1}, "min_history_win_rate"),
        ("vol_long_len_lt_short_rejected", {**case["base_payload"], "vol_short_len": 10, "vol_long_len": 5}, "vol_long_len"),
        ("use_bb_string_type_rejected", {**case["base_payload"], "use_bb": "abc"}, "use_bb"),
    ]

    runtime_valid_payload = {**case["base_payload"], "optimizer_max_workers": 3, "scanner_max_workers": 4}
    runtime_params = build_params_from_mapping(runtime_valid_payload)
    add_check(results, "synthetic_param_guardrail", case["case_id"], "runtime_optimizer_max_workers_loads", 3, getattr(runtime_params, "optimizer_max_workers", None))
    add_check(results, "synthetic_param_guardrail", case["case_id"], "runtime_scanner_max_workers_loads", 4, getattr(runtime_params, "scanner_max_workers", None))

    runtime_invalid_cases = [
        ("optimizer_max_workers_zero_rejected", {**case["base_payload"], "optimizer_max_workers": 0}, "optimizer_max_workers"),
        ("scanner_max_workers_zero_rejected", {**case["base_payload"], "scanner_max_workers": 0}, "scanner_max_workers"),
    ]

    for metric_name, payload, expected_field in invalid_cases + runtime_invalid_cases:
        try:
            build_params_from_mapping(payload)
            add_fail_result(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                metric_name,
                f"ValueError containing {expected_field}",
                "loaded_ok",
                "非法參數不應成功載入。"
            )
        except ValueError as e:
            add_check(results, "synthetic_param_guardrail", case["case_id"], metric_name, True, expected_field in str(e))

    for metric_name, payload, expected_field in invalid_cases:
        try:
            V16StrategyParams(**payload)
            add_fail_result(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                f"direct_dataclass_{metric_name}",
                f"ValueError containing {expected_field}",
                "loaded_ok",
                "直接建立 V16StrategyParams 也不應繞過 guardrail。"
            )
        except ValueError as e:
            add_check(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                f"direct_dataclass_{metric_name}",
                True,
                expected_field in str(e)
            )

    runtime_mutation_params = V16StrategyParams()
    try:
        runtime_mutation_params.optimizer_max_workers = 0
        add_fail_result(
            results,
            "synthetic_param_guardrail",
            case["case_id"],
            "direct_runtime_attr_guardrail",
            "ValueError containing optimizer_max_workers",
            "setattr_ok",
            "runtime worker 設定直接改欄位也不應繞過 guardrail。"
        )
    except ValueError as e:
        add_check(results, "synthetic_param_guardrail", case["case_id"], "direct_runtime_attr_guardrail", True, "optimizer_max_workers" in str(e))

    invalid_direct_setattr_cases = [
        ("direct_setattr_use_bb_string_rejected", "use_bb", "abc", "use_bb"),
        ("direct_setattr_high_len_string_rejected", "high_len", "10", "high_len"),
    ]

    for metric_name, field_name, invalid_value, expected_field in invalid_direct_setattr_cases:
        mutation_target = V16StrategyParams()
        try:
            setattr(mutation_target, field_name, invalid_value)
            add_fail_result(
                results,
                "synthetic_param_guardrail",
                case["case_id"],
                metric_name,
                f"ValueError containing {expected_field}",
                "setattr_ok",
                "直接改 dataclass 欄位不應繞過型別 guardrail。"
            )
        except ValueError as e:
            add_check(results, "synthetic_param_guardrail", case["case_id"], metric_name, True, expected_field in str(e))

    mutation_params = V16StrategyParams()
    try:
        mutation_params.tp_precent = 0.3
        add_fail_result(
            results,
            "synthetic_param_guardrail",
            case["case_id"],
            "unknown_attr_typo_rejected",
            "AttributeError containing tp_precent",
            "setattr_ok",
            "未知屬性 typo 不應靜默掛到 params 物件上。"
        )
    except AttributeError as e:
        add_check(results, "synthetic_param_guardrail", case["case_id"], "unknown_attr_typo_rejected", True, "tp_precent" in str(e))

    default_params_arg = run_v16_backtest.__defaults__[0] if run_v16_backtest.__defaults__ else None
    add_check(results, "synthetic_param_guardrail", case["case_id"], "run_v16_backtest_default_params_is_none", True, default_params_arg is None)

    summary["guardrail_cases"] = (len(invalid_cases) * 2) + len(runtime_invalid_cases) + len(invalid_direct_setattr_cases) + 4
    return results, summary

def run_synthetic_consistency_suite(base_params):
    all_results = []
    summaries = []
    validators = [
        validate_synthetic_half_tp_full_year_case,
        validate_synthetic_extended_miss_buy_case,
        validate_synthetic_competing_candidates_case,
        validate_synthetic_same_day_sell_block_case,
        validate_synthetic_unexecutable_half_tp_case,
        validate_synthetic_rotation_t_plus_one_case,
        validate_synthetic_proj_cost_cash_capped_case,
        validate_synthetic_history_ev_threshold_case,
        validate_synthetic_param_guardrail_case,
    ]

    for validator in validators:
        results, summary = validator(base_params)
        all_results.extend(results)
        summaries.append(summary)

    return all_results, summaries
