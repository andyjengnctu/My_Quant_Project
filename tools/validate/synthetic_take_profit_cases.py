import os
import tempfile

import pandas as pd

from core.price_utils import calc_reference_candidate_qty, can_execute_half_take_profit
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.entry_plans import (
    build_cash_capped_entry_plan,
    build_normal_candidate_plan,
    build_position_from_entry_fill,
    execute_pre_market_entry_plan,
)
from core.position_step import execute_bar_step
from core.exact_accounting import build_buy_ledger_from_price, build_sell_ledger_from_price, milli_to_money
from core.price_utils import adjust_long_sell_fill_price, adjust_long_stop_price, calc_net_sell_price
from core.portfolio_fast_data import calc_mark_to_market_equity, pack_prepared_stock_data, prep_stock_data_and_trades
from core.portfolio_engine import run_portfolio_timeline
from .synthetic_frame_utils import build_synthetic_baseline_frame, set_synthetic_bar

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
    original_cost_basis_milli = int(position["remaining_cost_basis_milli"])
    highest_high_since_entry = float(position.get("highest_high_since_entry", buy_price))
    expected_stop_level = max(
        stop_level,
        adjust_long_stop_price(
            highest_high_since_entry - (1.0 * params.atr_times_trail),
            ticker=position.get("ticker"),
        ),
    )

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

    expected_exec_price = adjust_long_sell_fill_price(min(expected_stop_level, 100.0))
    expected_sell_ledger = build_sell_ledger_from_price(expected_exec_price, qty, params)
    expected_freed_cash = milli_to_money(expected_sell_ledger["net_sell_total_milli"])
    expected_pnl = milli_to_money(expected_sell_ledger["net_sell_total_milli"] - original_cost_basis_milli)

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


def validate_synthetic_conservative_executable_exit_interpretation_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.5)
    params.atr_times_init = 2.0
    params.atr_times_trail = 1.0
    case_id = "SYNTH_CONSERVATIVE_EXECUTABLE_EXIT_INTERPRETATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    ambiguous_position = build_position_from_entry_fill(
        buy_price=100.0,
        qty=10,
        init_sl=95.0,
        init_trail=95.0,
        params=params,
        entry_type="normal",
    )

    stop_level = float(ambiguous_position["sl"])
    tp_half_level = float(ambiguous_position["tp_half"])
    original_cost_basis_milli = int(ambiguous_position["remaining_cost_basis_milli"])
    updated_position, freed_cash, pnl_realized, events = execute_bar_step(
        ambiguous_position,
        y_atr=1.0,
        y_ind_sell=False,
        y_close=100.0,
        t_open=90.0,
        t_high=tp_half_level + 1.0,
        t_low=89.0,
        t_close=91.0,
        t_volume=1000.0,
        params=params,
    )
    stop_exec_context = next((ctx for ctx in updated_position.get("_last_exec_contexts", []) if ctx.get("event") == "STOP"), None)
    expected_same_bar_exec_price = adjust_long_sell_fill_price(90.0)
    expected_same_bar_sell_ledger = build_sell_ledger_from_price(expected_same_bar_exec_price, 10, params)
    expected_same_bar_freed_cash = milli_to_money(expected_same_bar_sell_ledger["net_sell_total_milli"])
    expected_same_bar_pnl = milli_to_money(expected_same_bar_sell_ledger["net_sell_total_milli"] - original_cost_basis_milli)

    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "same_bar_ambiguous_exit_prefers_stop", True, "STOP" in events)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "same_bar_ambiguous_exit_suppresses_tp_half", False, "TP_HALF" in events, note="同一事件若同棒同時滿足停損 / 停利，只能採最保守、最不利於績效的停損解讀。")
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "same_bar_stop_executes_at_first_worse_executable_open", 90.0, None if stop_exec_context is None else float(stop_exec_context["exec_price"]))
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "same_bar_stop_freed_cash_uses_worse_executable_open", expected_same_bar_freed_cash, float(freed_cash), tol=0.01)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "same_bar_stop_realized_pnl_uses_worse_executable_open", expected_same_bar_pnl, float(pnl_realized), tol=0.01)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "same_bar_stop_closes_position", 0, int(updated_position["qty"]))

    limit_price = 100.0
    atr = 5.0
    candidate_plan = build_normal_candidate_plan(limit_price, atr, 1_000_000.0, params)
    entry_result = execute_pre_market_entry_plan(
        entry_plan=candidate_plan,
        t_open=98.0,
        t_high=100.0,
        t_low=87.5,
        t_close=89.0,
        t_volume=1000.0,
        y_close=100.0,
        params=params,
        entry_type="normal",
    )
    deferred_stop_position = entry_result.get("position")
    if deferred_stop_position is None:
        raise ValueError("validate_synthetic_conservative_executable_exit_interpretation_case 需要有效成交部位")

    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "entry_day_stop_queues_next_day_open_execution", "STOP", deferred_stop_position.get("pending_exit_action"))

    deferred_original_qty = int(deferred_stop_position["qty"])
    deferred_original_cost_basis_milli = int(deferred_stop_position["remaining_cost_basis_milli"])
    updated_deferred_position, deferred_freed_cash, deferred_pnl_realized, deferred_events = execute_bar_step(
        deferred_stop_position,
        y_atr=atr,
        y_ind_sell=False,
        y_close=89.0,
        t_open=92.0,
        t_high=92.8,
        t_low=92.0,
        t_close=92.5,
        t_volume=1000.0,
        params=params,
        y_high=100.0,
    )
    deferred_stop_context = next((ctx for ctx in updated_deferred_position.get("_last_exec_contexts", []) if ctx.get("event") == "STOP"), None)
    expected_deferred_sell_ledger = build_sell_ledger_from_price(92.0, deferred_original_qty, params)
    expected_deferred_freed_cash = milli_to_money(expected_deferred_sell_ledger["net_sell_total_milli"])
    expected_deferred_pnl = milli_to_money(expected_deferred_sell_ledger["net_sell_total_milli"] - deferred_original_cost_basis_milli)

    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_executes_from_queued_open_without_rehit", True, "STOP" in deferred_events)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_keeps_deferred_marker", True, "DEFERRED_STOP_ON_OPEN" in deferred_events)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_uses_current_bar_first_worse_executable_open", 92.0, None if deferred_stop_context is None else float(deferred_stop_context["exec_price"]))
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_freed_cash_uses_current_bar_execution", expected_deferred_freed_cash, float(deferred_freed_cash), tol=0.01)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_realized_pnl_uses_current_bar_execution", expected_deferred_pnl, float(deferred_pnl_realized), tol=0.01)
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_closes_position_without_rehit", 0, int(updated_deferred_position["qty"]))
    add_check(results, "synthetic_conservative_executable_exit_interpretation", case_id, "next_day_stop_leaves_no_pending_action", None, updated_deferred_position.get("pending_exit_action"))

    summary["same_bar_events"] = list(events)
    summary["same_bar_exec_price"] = None if stop_exec_context is None else float(stop_exec_context["exec_price"])
    summary["next_day_events"] = list(deferred_events)
    summary["deferred_exec_price"] = None if deferred_stop_context is None else float(deferred_stop_context["exec_price"])
    return results, summary


def validate_synthetic_exit_orders_only_for_held_positions_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.5)
    case_id = "SYNTH_EXIT_ORDERS_ONLY_FOR_HELD_POSITIONS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    position = build_position_from_entry_fill(
        buy_price=100.0,
        qty=10,
        init_sl=95.0,
        init_trail=95.0,
        params=params,
        entry_type="normal",
    )
    position["qty"] = 0

    updated_position, freed_cash, pnl_realized, events = execute_bar_step(
        position,
        y_atr=1.0,
        y_ind_sell=True,
        y_close=100.0,
        t_open=90.0,
        t_high=120.0,
        t_low=80.0,
        t_close=85.0,
        t_volume=1000.0,
        params=params,
    )

    add_check(results, "synthetic_exit_orders_only_for_held_positions", case_id, "zero_qty_has_no_events", [], list(events))
    add_check(results, "synthetic_exit_orders_only_for_held_positions", case_id, "zero_qty_has_no_freed_cash", 0.0, float(freed_cash), tol=0.01)
    add_check(results, "synthetic_exit_orders_only_for_held_positions", case_id, "zero_qty_has_no_realized_pnl", 0.0, float(pnl_realized), tol=0.01)
    add_check(results, "synthetic_exit_orders_only_for_held_positions", case_id, "zero_qty_position_stays_zero", 0, int(updated_position["qty"]))
    add_check(results, "synthetic_exit_orders_only_for_held_positions", case_id, "zero_qty_realized_pnl_stays_zero", 0.0, float(updated_position["realized_pnl"]), tol=0.01)

    summary["events"] = list(events)
    return results, summary


def validate_synthetic_fee_tax_net_equity_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    case_id = "SYNTH_FEE_TAX_NET_EQUITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    df = build_synthetic_baseline_frame("2024-01-01", 60)
    set_synthetic_bar(df, 55, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df, 56, open_price=103.8, high_price=104.0, low_price=100.0, close_price=101.0)
    set_synthetic_bar(df, 57, open_price=101.0, high_price=101.2, low_price=99.5, close_price=100.0)
    set_synthetic_bar(df, 58, open_price=100.0, high_price=100.2, low_price=99.7, close_price=100.1)
    set_synthetic_bar(df, 59, open_price=100.1, high_price=100.4, low_price=99.8, close_price=100.2)

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, {"9821": df})
        primary_path = os.path.join(temp_dir, "9821.csv")

        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, params, max_positions=1, enable_rotation=False, start_year=2024, benchmark_ticker="9821"
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, params, max_positions=1, enable_rotation=False, start_year=2024, benchmark_ticker="9821"
        )
        add_portfolio_stats_equality_checks(results, "synthetic_fee_tax_net_equity", case_id, core_stats, sim_stats)

        raw_df = pd.read_csv(primary_path)
        clean_df, _sanitize_stats = sanitize_ohlcv_dataframe(raw_df, "9821", min_rows=get_required_min_rows(params))
        prep_df, standalone_logs = prep_stock_data_and_trades(clean_df, params)
        fast_df = pack_prepared_stock_data(prep_df)
        sorted_dates = list(prep_df.index)
        setup_date = prep_df.index[55]
        entry_date = prep_df.index[56]
        exit_date = prep_df.index[57]

        candidate_plan = build_normal_candidate_plan(
            float(prep_df.loc[setup_date, "buy_limit"]),
            float(prep_df.loc[setup_date, "ATR"]),
            params.initial_capital,
            params,
        )
        entry_plan = build_cash_capped_entry_plan(candidate_plan, params.initial_capital, params)
        if entry_plan is None:
            raise ValueError("synthetic_fee_tax_net_equity_case 應產生有效 entry_plan")

        entry_result = execute_pre_market_entry_plan(
            entry_plan=entry_plan,
            t_open=float(prep_df.loc[entry_date, "Open"]),
            t_high=float(prep_df.loc[entry_date, "High"]),
            t_low=float(prep_df.loc[entry_date, "Low"]),
            t_close=float(prep_df.loc[entry_date, "Close"]),
            t_volume=float(prep_df.loc[entry_date, "Volume"]),
            y_close=float(prep_df.loc[setup_date, "Close"]),
            params=params,
            entry_type="normal",
        )
        if not entry_result["filled"]:
            raise ValueError("synthetic_fee_tax_net_equity_case 應在 entry_date 成交")

        entry_qty = int(entry_plan["qty"])
        entry_cost = milli_to_money(entry_result["position"]["net_buy_total_milli"])
        entry_cash_after_buy = float(params.initial_capital - entry_cost)
        expected_entry_day_equity = calc_mark_to_market_equity(
            entry_cash_after_buy,
            {"9821": dict(entry_result["position"])},
            {"9821": fast_df},
            entry_date,
            params,
        )

        exit_position = dict(entry_result["position"])
        updated_position, freed_cash, pnl_realized, events = execute_bar_step(
            exit_position,
            y_atr=float(prep_df.loc[entry_date, "ATR"]),
            y_ind_sell=bool(prep_df.loc[entry_date, "ind_sell_signal"]),
            y_close=float(prep_df.loc[entry_date, "Close"]),
            t_open=float(prep_df.loc[exit_date, "Open"]),
            t_high=float(prep_df.loc[exit_date, "High"]),
            t_low=float(prep_df.loc[exit_date, "Low"]),
            t_close=float(prep_df.loc[exit_date, "Close"]),
            t_volume=float(prep_df.loc[exit_date, "Volume"]),
            params=params,
        )
        expected_final_eq = float(entry_cash_after_buy + freed_cash)
        expected_total_return = (expected_final_eq - params.initial_capital) / params.initial_capital * 100.0

        all_dfs_fast = {"9821": fast_df}
        all_trade_logs = {"9821": standalone_logs}
        timeline_result = run_portfolio_timeline(
            all_dfs_fast=all_dfs_fast,
            all_standalone_logs=all_trade_logs,
            sorted_dates=sorted_dates,
            start_year=2024,
            params=params,
            max_positions=1,
            enable_rotation=False,
            benchmark_ticker="9821",
            benchmark_data=fast_df,
            is_training=False,
            profile_stats={},
            verbose=False,
        )
        df_equity = timeline_result[0]
        df_trades = timeline_result[1]
        actual_final_eq = float(timeline_result[8])
        actual_total_return = float(timeline_result[2])

        entry_row = df_equity[df_equity["Date"] == entry_date.strftime("%Y-%m-%d")].iloc[0]
        exit_row = df_equity[df_equity["Date"] == exit_date.strftime("%Y-%m-%d")].iloc[0]
        exit_trade_row = df_trades[df_trades["Type"].fillna("").isin(["全倉結算(停損)", "全倉結算(指標)"])].iloc[0]
        actual_entry_cash = float(entry_row["Equity"] - entry_row["Invested_Amount"])

        add_check(results, "synthetic_fee_tax_net_equity", case_id, "entry_qty_positive", True, entry_qty > 0)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "exit_event_is_stop", True, "STOP" in events)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "entry_day_cash_matches_net_entry_cost", entry_cash_after_buy, actual_entry_cash, tol=0.01)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "entry_day_equity_marks_to_net_sell_value", expected_entry_day_equity, float(entry_row["Equity"]), tol=0.01)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "exit_trade_total_pnl_matches_net_realized_pnl", float(updated_position["realized_pnl"]), float(exit_trade_row["該筆總損益"]), tol=0.01)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "exit_day_equity_matches_final_cash", expected_final_eq, float(exit_row["Equity"]), tol=0.01)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "final_equity_matches_net_cash", expected_final_eq, actual_final_eq, tol=0.01)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "total_return_matches_net_final_equity", expected_total_return, actual_total_return, tol=0.0001)
        add_check(results, "synthetic_fee_tax_net_equity", case_id, "pnl_matches_net_cash_delta", params.initial_capital + float(pnl_realized), expected_final_eq, tol=0.01)

    summary["expected_final_eq"] = round(expected_final_eq, 4)
    summary["entry_qty"] = entry_qty
    return results, summary




def validate_synthetic_missed_sell_accounting_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    case_id = "SYNTH_MISSED_SELL_ACCOUNTING"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    df = build_synthetic_baseline_frame("2024-01-01", 60)
    set_synthetic_bar(df, 55, open_price=103.0, high_price=104.5, low_price=102.8, close_price=104.0)
    set_synthetic_bar(df, 56, open_price=103.8, high_price=105.0, low_price=103.4, close_price=104.2, volume=1000)
    set_synthetic_bar(df, 57, open_price=102.5, high_price=103.0, low_price=100.5, close_price=101.5, volume=0)
    set_synthetic_bar(df, 58, open_price=101.4, high_price=101.9, low_price=100.9, close_price=101.1, volume=1000)
    set_synthetic_bar(df, 59, open_price=101.1, high_price=101.3, low_price=100.8, close_price=101.0, volume=1000)

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, {"9851": df})
        core_stats = run_portfolio_core_check_for_dir(
            temp_dir, params, max_positions=1, enable_rotation=False, start_year=2024, benchmark_ticker="9851"
        )
        sim_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir, params, max_positions=1, enable_rotation=False, start_year=2024, benchmark_ticker="9851"
        )
        add_portfolio_stats_equality_checks(results, "synthetic_missed_sell_accounting", case_id, core_stats, sim_stats)

        primary_path = os.path.join(temp_dir, "9851.csv")
        raw_df = pd.read_csv(primary_path)
        debug_input_df, _debug_sanitize_stats = sanitize_ohlcv_dataframe(
            raw_df,
            "9851",
            min_rows=get_required_min_rows(params),
        )
        debug_df, _debug_module_path = run_debug_trade_log_check("9851", debug_input_df, params)
        action_series = debug_df["動作"].fillna("") if debug_df is not None and len(debug_df) > 0 else pd.Series(dtype=str)
        missed_sell_rows = int((action_series == "錯失賣出").sum())
        missed_sell_row = debug_df[action_series == "錯失賣出"].iloc[0] if missed_sell_rows > 0 else None
        portfolio_missed_rows = (
            sim_stats["df_trades"][sim_stats["df_trades"]["Type"].fillna("") == "錯失賣出"].copy()
            if not sim_stats["df_trades"].empty else pd.DataFrame()
        )
        completed_trades = sim_stats["portfolio_completed_trades"]
        completed_trade_pnl = float(completed_trades[0]["total_pnl"]) if completed_trades else None
        final_exit_rows = (
            sim_stats["df_trades"][sim_stats["df_trades"]["Type"].fillna("").isin(["全倉結算(停損)", "全倉結算(指標)"])].copy()
            if not sim_stats["df_trades"].empty else pd.DataFrame()
        )
        final_exit_total_pnl = float(final_exit_rows.iloc[-1]["該筆總損益"]) if len(final_exit_rows) > 0 else None
        missed_sell_total_pnl = float(portfolio_missed_rows.iloc[0]["該筆總損益"]) if len(portfolio_missed_rows) > 0 else None
        missed_sell_note = None if missed_sell_row is None else missed_sell_row.get("備註")

        add_check(results, "synthetic_missed_sell_accounting", case_id, "core_total_missed_sells", 1, int(core_stats["total_missed_sells"]))
        add_check(results, "synthetic_missed_sell_accounting", case_id, "portfolio_total_missed_sells", 1, int(sim_stats["total_missed_sells"]))
        add_check(results, "synthetic_missed_sell_accounting", case_id, "portfolio_missed_sell_rows", 1, int(sim_stats["portfolio_missed_sell_rows"]))
        add_check(results, "synthetic_missed_sell_accounting", case_id, "df_trades_missed_sell_rows", 1, len(portfolio_missed_rows))
        add_check(results, "synthetic_missed_sell_accounting", case_id, "debug_missed_sell_rows", 1, missed_sell_rows)
        add_check(results, "synthetic_missed_sell_accounting", case_id, "missed_sell_note_marks_block_reason", True, missed_sell_note == "零量，當日無法賣出")
        add_check(results, "synthetic_missed_sell_accounting", case_id, "missed_sell_row_carries_pre_exit_realized_pnl", 0.0, missed_sell_total_pnl, tol=0.01)
        add_check(results, "synthetic_missed_sell_accounting", case_id, "eventual_exit_keeps_round_trip_trade_count", 1, int(sim_stats["trade_count"]))
        add_check(results, "synthetic_missed_sell_accounting", case_id, "eventual_exit_round_trip_pnl_matches_completed_trade", completed_trade_pnl, final_exit_total_pnl, tol=0.01)

    summary["missed_sell_rows"] = missed_sell_rows
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


def validate_synthetic_round_trip_pnl_only_on_tail_exit_case(base_params):
    case = build_synthetic_half_tp_full_year_case(base_params)
    results = []
    summary = {"ticker": "SYNTH_ROUND_TRIP_PNL_ONLY_ON_TAIL_EXIT", "synthetic": True}

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
        add_portfolio_stats_equality_checks(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], core_stats, sim_stats)

        df_trades = sim_stats["df_trades"].copy()
        half_rows = df_trades[df_trades["Type"].fillna("") == "半倉停利"].copy() if not df_trades.empty else pd.DataFrame()
        full_exit_rows = df_trades[df_trades["Type"].fillna("").isin(["全倉結算(停損)", "全倉結算(指標)", "期末強制結算", "汰弱賣出(Open, T+1再評估買進)"])].copy() if not df_trades.empty else pd.DataFrame()
        completed_trades = sim_stats["portfolio_completed_trades"]
        completed_trade_pnl = float(completed_trades[0]["total_pnl"]) if completed_trades else None
        half_row_total_pnl = None if half_rows.empty else half_rows.iloc[0]["該筆總損益"]
        final_exit_total_pnl = None if full_exit_rows.empty else float(full_exit_rows.iloc[-1]["該筆總損益"])
        half_exit_count = int(completed_trades[0]["half_exit_count"]) if completed_trades else None

        add_check(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], "half_take_profit_rows", 1, len(half_rows))
        add_check(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], "completed_trade_count_counts_round_trip_only", 1, int(sim_stats["trade_count"]))
        add_check(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], "rebuilt_completed_trade_count", 1, len(completed_trades))
        add_check(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], "completed_trade_records_half_exit_count", 1, half_exit_count)
        add_check(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], "half_take_profit_row_has_no_round_trip_total_pnl", True, pd.isna(half_row_total_pnl), note="半倉停利只視為現金回收，不得在半倉列提前結算完整 Round-Trip PnL。")
        add_check(results, "synthetic_round_trip_pnl_only_on_tail_exit", summary["ticker"], "final_exit_row_carries_round_trip_total_pnl", completed_trade_pnl, final_exit_total_pnl, tol=0.01)

    summary["completed_trade_count"] = len(completed_trades)
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
    scanner_case["params"].scanner_live_capital = 130.0
    scanner_case["params"].fixed_risk = 1.0
    scanner_case["frames"][scanner_case["primary_ticker"]] = scanner_case["frames"][scanner_case["primary_ticker"]].iloc[:271].copy()

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, scanner_case["frames"])
        scanner_ticker = scanner_case["primary_ticker"]
        scanner_path = os.path.join(temp_dir, f"{scanner_ticker}.csv")
        scanner_ref_stats = run_scanner_reference_check(scanner_ticker, scanner_path, scanner_case["params"])
        scanner_result, _scanner_module_path = run_scanner_tool_check(scanner_ticker, scanner_path, scanner_case["params"])
        proj_qty = calc_reference_candidate_qty(scanner_ref_stats["buy_limit"], scanner_ref_stats["stop_loss"], scanner_case["params"], ticker=scanner_ticker, trade_date=scanner_ref_stats.get("trade_date")) if scanner_ref_stats.get("is_setup_today") else 0
        scanner_message = "" if scanner_result is None or scanner_result.get("message") is None else str(scanner_result.get("message"))
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_projected_qty", 1, proj_qty)
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_half_tp_executable", False, can_execute_half_take_profit(proj_qty, scanner_case["params"].tp_percent))
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_tool_status", "buy", None if scanner_result is None else scanner_result["status"])
        add_check(results, "synthetic_unexecutable_half_tp", case["case_id"], "scanner_message_marks_unexecutable_half_tp", True, "半倉停利:股數不足" in scanner_message)

    summary["half_take_profit_rows"] = 0
    summary["projected_qty"] = 1
    return results, summary
