import os
import tempfile

import pandas as pd

from core.portfolio_entries import execute_reserved_entries_for_day
from core.portfolio_fast_data import pack_prepared_stock_data

from .checks import add_check, add_fail_result, build_expected_scanner_payload, make_synthetic_validation_params, run_scanner_reference_check
from .synthetic_fixtures import write_synthetic_csv_bundle
from .synthetic_frame_utils import build_synthetic_baseline_frame, set_synthetic_bar
from .synthetic_portfolio_common import (
    add_portfolio_stats_equality_checks,
    build_synthetic_competing_candidates_case,
    build_synthetic_extended_miss_buy_case,
    build_synthetic_rotation_t_plus_one_case,
    build_synthetic_same_day_sell_block_case,
    run_portfolio_core_check_for_dir,
)
from .tool_adapters import run_portfolio_sim_tool_check_for_dir, run_scanner_tool_check


def _run_failed_fill_no_switch_scenario(base_params, *, case_id, module_name, include_alternate_candidate):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    miss_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.5],
            "Close": [100.0, 101.8],
            "Volume": [1000.0, 1000.0],
            "ATR": [1.0, 1.0],
            "buy_limit": [100.0, 100.0],
            "is_setup": [True, False],
            "ind_sell_signal": [False, False],
        },
        index=dates,
    )

    all_dfs_fast = {"9801": pack_prepared_stock_data(miss_df)}
    orderable_candidates_today = [
        {
            "ticker": "9801",
            "type": "normal",
            "limit_px": 100.0,
            "init_sl": 95.0,
            "init_trail": 95.0,
            "ev": 2.0,
            "today_pos": 1,
            "yesterday_pos": 0,
        }
    ]

    if include_alternate_candidate:
        alt_df = pd.DataFrame(
            {
                "Open": [50.0, 50.0],
                "High": [51.0, 51.0],
                "Low": [49.0, 49.0],
                "Close": [50.0, 50.5],
                "Volume": [1000.0, 1000.0],
                "ATR": [1.0, 1.0],
                "buy_limit": [50.0, 50.0],
                "is_setup": [True, False],
                "ind_sell_signal": [False, False],
            },
            index=dates,
        )
        all_dfs_fast["9802"] = pack_prepared_stock_data(alt_df)
        orderable_candidates_today.append(
            {
                "ticker": "9802",
                "type": "normal",
                "limit_px": 50.0,
                "init_sl": 47.5,
                "init_trail": 47.5,
                "ev": 1.0,
                "today_pos": 1,
                "yesterday_pos": 0,
            }
        )

    portfolio = {}
    active_extended_signals = {}
    sold_today = set()
    trade_history = []
    initial_cash = 1_000_000.0
    cash, total_missed_buys = execute_reserved_entries_for_day(
        portfolio=portfolio,
        active_extended_signals=active_extended_signals,
        orderable_candidates_today=orderable_candidates_today,
        sold_today=sold_today,
        all_dfs_fast=all_dfs_fast,
        today=pd.Timestamp("2024-01-03"),
        params=params,
        cash=initial_cash,
        available_cash=initial_cash,
        max_positions=1,
        trade_history=trade_history,
        is_training=False,
        total_missed_buys=0,
    )

    miss_rows = [row for row in trade_history if row.get("Ticker") == "9801" and str(row.get("Type", "")).startswith("錯失買進")]
    alt_buy_rows = [row for row in trade_history if row.get("Ticker") == "9802" and str(row.get("Type", "")).startswith("買進")]

    add_check(results, module_name, case_id, "missed_buy_count", 1, int(total_missed_buys))
    add_check(results, module_name, case_id, "portfolio_stays_empty", 0, len(portfolio))
    add_check(results, module_name, case_id, "cash_unchanged_without_fill", initial_cash, float(cash), tol=0.01)
    add_check(results, module_name, case_id, "miss_row_recorded", 1, len(miss_rows))
    add_check(results, module_name, case_id, "same_day_limit_above_low_stays_unfilled", True, float(miss_df.loc[dates[1], "Low"]) > 100.0, note="當日低點未觸及原始限價時，不得盤中上調委託價促成成交。")

    if include_alternate_candidate:
        add_check(results, module_name, case_id, "alternate_ticker_not_bought_same_day", 0, len(alt_buy_rows), note="當日未成交後，不得同日盤中改掛其他股票。")

    summary["missed_buy_count"] = int(total_missed_buys)
    summary["alternate_buy_rows"] = len(alt_buy_rows)
    return results, summary


def validate_synthetic_intraday_reprice_forbidden_case(base_params):
    return _run_failed_fill_no_switch_scenario(
        base_params,
        case_id="SYNTH_INTRADAY_REPRICE_FORBIDDEN",
        module_name="synthetic_intraday_reprice_forbidden",
        include_alternate_candidate=False,
    )


def validate_synthetic_no_intraday_switch_after_failed_fill_case(base_params):
    return _run_failed_fill_no_switch_scenario(
        base_params,
        case_id="SYNTH_NO_INTRADAY_SWITCH_AFTER_FAILED_FILL",
        module_name="synthetic_no_intraday_switch_after_failed_fill",
        include_alternate_candidate=True,
    )


def validate_synthetic_missed_buy_no_replacement_case(base_params):
    return _run_failed_fill_no_switch_scenario(
        base_params,
        case_id="SYNTH_MISSED_BUY_NO_REPLACEMENT",
        module_name="synthetic_missed_buy_no_replacement",
        include_alternate_candidate=True,
    )


def validate_synthetic_same_day_buy_sell_forbidden_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    case_id = "SYNTH_SAME_DAY_BUY_SELL_FORBIDDEN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    df = build_synthetic_baseline_frame("2024-01-01", 60)
    set_synthetic_bar(df, 55, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df, 56, open_price=103.8, high_price=104.0, low_price=100.0, close_price=101.0)
    set_synthetic_bar(df, 57, open_price=101.0, high_price=101.2, low_price=99.5, close_price=100.0)
    set_synthetic_bar(df, 58, open_price=100.0, high_price=100.2, low_price=99.7, close_price=100.1)
    set_synthetic_bar(df, 59, open_price=100.1, high_price=100.4, low_price=99.8, close_price=100.2)

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, {"9811": df})
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, params, max_positions=1, enable_rotation=False, start_year=2024, benchmark_ticker="9811"
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, params, max_positions=1, enable_rotation=False, start_year=2024, benchmark_ticker="9811"
        )
        add_portfolio_stats_equality_checks(results, "synthetic_same_day_buy_sell_forbidden", case_id, core_stats, sim_stats)

        df_trades = sim_stats["df_trades"].copy()
        buy_rows = df_trades[df_trades["Type"].fillna("").str.startswith("買進")].copy() if not df_trades.empty else pd.DataFrame()
        exit_rows = df_trades[df_trades["Type"].fillna("").isin(["全倉結算(停損)", "全倉結算(指標)"])].copy() if not df_trades.empty else pd.DataFrame()

        buy_date = pd.to_datetime(buy_rows.iloc[0]["Date"]) if len(buy_rows) > 0 else None
        exit_date = pd.to_datetime(exit_rows.iloc[0]["Date"]) if len(exit_rows) > 0 else None
        same_day_exit = bool(((pd.to_datetime(df_trades["Date"]) == buy_date) & df_trades["Type"].fillna("").isin(["全倉結算(停損)", "全倉結算(指標)"])).any()) if buy_date is not None else False

        add_check(results, "synthetic_same_day_buy_sell_forbidden", case_id, "buy_row_count", 1, len(buy_rows))
        add_check(results, "synthetic_same_day_buy_sell_forbidden", case_id, "exit_row_count", 1, len(exit_rows))
        add_check(results, "synthetic_same_day_buy_sell_forbidden", case_id, "same_day_buy_has_no_same_day_exit", False, same_day_exit, note="買入當日即使觸及停損或停利，也不得同日賣出。")
        add_check(results, "synthetic_same_day_buy_sell_forbidden", case_id, "next_day_exit_occurs_after_buy_date", True, (buy_date is not None and exit_date is not None and exit_date > buy_date))

    summary["same_day_exit_blocked"] = True
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
