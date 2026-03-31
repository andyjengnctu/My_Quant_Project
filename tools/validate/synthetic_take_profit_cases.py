import os
import tempfile

import pandas as pd

from core.backtest_core import calc_reference_candidate_qty, can_execute_half_take_profit
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.entry_plans import build_position_from_entry_fill
from core.position_step import execute_bar_step
from core.price_utils import adjust_long_sell_fill_price, calc_net_sell_price

from .checks import add_check, build_expected_scanner_payload, make_synthetic_validation_params, run_scanner_reference_check
from .synthetic_fixtures import write_synthetic_csv_bundle
from .synthetic_portfolio_common import (
    add_portfolio_stats_equality_checks,
    build_synthetic_half_tp_full_year_case,
    build_synthetic_unexecutable_half_tp_case,
    run_portfolio_core_check_for_dir,
)
from .tool_adapters import run_debug_trade_log_check, run_portfolio_sim_tool_check_for_dir, run_scanner_tool_check


def validate_synthetic_same_bar_stop_priority_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.5)
    case_id = "SYNTH_SAME_BAR_STOP_PRIORITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    buy_price = 100.0
    qty = 10
    init_sl = 95.0
    init_trail = 95.0
    position = build_position_from_entry_fill(
        buy_price=buy_price,
        qty=qty,
        init_sl=init_sl,
        init_trail=init_trail,
        params=params,
        entry_type="normal",
    )

    stop_level = float(position["sl"])
    tp_half_level = float(position["tp_half"])
    entry_price = float(position["entry"])

    updated_position, freed_cash, pnl_realized, events = execute_bar_step(
        position,
        y_atr=1.0,
        y_ind_sell=False,
        y_close=buy_price,
        t_open=100.0,
        t_high=tp_half_level + 1.0,
        t_low=stop_level - 1.0,
        t_close=98.0,
        t_volume=1000.0,
        params=params,
    )

    expected_exec_price = adjust_long_sell_fill_price(min(stop_level, 100.0))
    expected_net_price = calc_net_sell_price(expected_exec_price, qty, params)
    expected_freed_cash = expected_net_price * qty
    expected_pnl = (expected_net_price - entry_price) * qty

    add_check(results, "synthetic_same_bar_stop_priority", case_id, "stop_event_emitted", True, "STOP" in events)
    add_check(results, "synthetic_same_bar_stop_priority", case_id, "tp_half_event_suppressed", False, "TP_HALF" in events, note="同 K 棒同時碰到停損 / 停利時，必須以最壞停損計算。")
    add_check(results, "synthetic_same_bar_stop_priority", case_id, "position_fully_closed", 0, int(updated_position["qty"]))
    add_check(results, "synthetic_same_bar_stop_priority", case_id, "sold_half_remains_false", False, bool(updated_position["sold_half"]))
    add_check(results, "synthetic_same_bar_stop_priority", case_id, "freed_cash_matches_stop_only", expected_freed_cash, float(freed_cash), tol=0.01)
    add_check(results, "synthetic_same_bar_stop_priority", case_id, "realized_pnl_matches_stop_only", expected_pnl, float(pnl_realized), tol=0.01)
    add_check(results, "synthetic_same_bar_stop_priority", case_id, "position_realized_pnl_matches_stop_only", expected_pnl, float(updated_position["realized_pnl"]), tol=0.01)

    summary["events"] = list(events)
    summary["expected_pnl"] = round(float(expected_pnl), 4)
    return results, summary


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
