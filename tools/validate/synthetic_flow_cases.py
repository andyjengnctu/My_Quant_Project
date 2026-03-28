import os
import tempfile

import pandas as pd

from .checks import add_check, add_fail_result, build_expected_scanner_payload, run_scanner_reference_check
from .synthetic_fixtures import write_synthetic_csv_bundle
from .synthetic_portfolio_common import (
    add_portfolio_stats_equality_checks,
    build_synthetic_competing_candidates_case,
    build_synthetic_extended_miss_buy_case,
    build_synthetic_rotation_t_plus_one_case,
    build_synthetic_same_day_sell_block_case,
    run_portfolio_core_check_for_dir,
)
from .tool_adapters import run_portfolio_sim_tool_check_for_dir, run_scanner_tool_check


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
                "rotation synthetic case 應產生 trade history。",
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
            (extended_miss_date is not None and rotation_sell_date is not None and extended_miss_date > rotation_sell_date),
        )
        add_check(
            results,
            "synthetic_rotation_t_plus_one",
            case["case_id"],
            "rotation_delayed_buy_occurs_after_sell",
            True,
            (delayed_buy_date is not None and rotation_sell_date is not None and delayed_buy_date > rotation_sell_date),
        )
        add_check(
            results,
            "synthetic_rotation_t_plus_one",
            case["case_id"],
            "rotation_delayed_buy_occurs_after_extended_miss",
            True,
            (delayed_buy_date is not None and extended_miss_date is not None and delayed_buy_date > extended_miss_date),
        )

    summary["rotation_sell"] = True
    summary["delayed_reentry"] = True
    return results, summary
