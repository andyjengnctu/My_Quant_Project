import os
import tempfile

import pandas as pd

from core.v16_core import calc_reference_candidate_qty, can_execute_half_take_profit
from core.v16_data_utils import get_required_min_rows, sanitize_ohlcv_dataframe

from .checks import add_check, add_fail_result, build_expected_scanner_payload, run_scanner_reference_check
from .synthetic_portfolio_common import (
    add_portfolio_stats_equality_checks,
    build_synthetic_competing_candidates_case,
    build_synthetic_extended_miss_buy_case,
    build_synthetic_half_tp_full_year_case,
    build_synthetic_rotation_t_plus_one_case,
    build_synthetic_same_day_sell_block_case,
    build_synthetic_unexecutable_half_tp_case,
    run_portfolio_core_check_for_dir,
)
from .synthetic_fixtures import write_synthetic_csv_bundle
from .tool_adapters import run_debug_trade_log_check, run_portfolio_sim_tool_check_for_dir, run_scanner_tool_check


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
