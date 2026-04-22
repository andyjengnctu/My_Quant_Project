from datetime import datetime

import numpy as np
import pandas as pd

from core.backtest_core import run_v16_backtest
from core.entry_plans import build_cash_capped_entry_plan, resize_candidate_plan_to_capital
from core.history_filters import evaluate_history_candidate_metrics
from core.exact_accounting import calc_entry_total_cost
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.portfolio_fast_data import build_normal_setup_index, build_trade_stats_index, get_pit_stats_from_index, pack_prepared_stock_data

from .checks import add_check, make_synthetic_validation_params, run_scanner_reference_check


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
        {"ticker": "9801", "limit_px": 5.0, "init_sl": 4.5, "init_trail": 4.0},
        {"ticker": "9802", "limit_px": 10.0, "init_sl": 9.0, "init_trail": 8.5},
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
        est_reserved_cost = calc_entry_total_cost(est_plan["limit_price"], est_plan["qty"], params)
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


def validate_synthetic_pit_same_day_exit_excluded_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)

    case_id = "SYNTH_PIT_SAME_DAY_EXIT_EXCLUDED"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exit_dates = [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    stats_index = {
        "exit_dates": exit_dates,
        "cum_trade_count": np.array([1, 2], dtype=np.int32),
        "cum_win_count": np.array([1, 1], dtype=np.int32),
        "cum_win_r_sum": np.array([1.0, 1.0], dtype=np.float64),
        "cum_loss_r_sum": np.array([0.0, -1.0], dtype=np.float64),
        "cum_total_r_sum": np.array([1.0, 0.0], dtype=np.float64),
        "cum_pnl_sum": np.array([100.0, 0.0], dtype=np.float64),
    }

    same_day_candidate, same_day_ev, same_day_win_rate, same_day_trade_count, _same_day_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 2),
        params,
    )
    next_day_candidate, next_day_ev, next_day_win_rate, next_day_trade_count, _next_day_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 3),
        params,
    )

    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "same_day_trade_count_excludes_same_day_exit", 0, same_day_trade_count)
    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "same_day_expected_value_excludes_same_day_exit", 0.0, same_day_ev)
    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "same_day_win_rate_excludes_same_day_exit", 0.0, same_day_win_rate)
    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "same_day_candidate_uses_pre_exit_history_only", True, same_day_candidate, note="PIT 統計在 exit_date 當天不得偷看同日剛結束的交易。")

    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "next_day_trade_count_includes_prior_exit", 1, next_day_trade_count)
    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "next_day_expected_value_includes_prior_exit", 1.0, next_day_ev)
    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "next_day_win_rate_includes_prior_exit", 1.0, next_day_win_rate)
    add_check(results, "synthetic_pit_same_day_exit_excluded", case_id, "next_day_candidate_includes_prior_exit", True, next_day_candidate)

    summary["same_day_trade_count"] = int(same_day_trade_count)
    summary["next_day_trade_count"] = int(next_day_trade_count)
    return results, summary


def validate_synthetic_single_backtest_not_gated_by_own_history_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.min_history_trades = 5
    params.min_history_ev = 0.5
    params.min_history_win_rate = 0.8

    case_id = "SYNTH_SINGLE_BACKTEST_NOT_GATED_BY_OWN_HISTORY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 101.0],
            "High": [100.0, 101.0, 102.0],
            "Low": [100.0, 99.0, 100.0],
            "Close": [100.0, 100.0, 101.0],
            "Volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    precomputed_signals = (
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([True, False, False], dtype=bool),
        np.array([False, True, False], dtype=bool),
        np.array([100.0, np.nan, np.nan], dtype=np.float64),
    )

    stats = run_v16_backtest(df, params, precomputed_signals=precomputed_signals)

    add_check(results, "synthetic_single_backtest_not_gated_by_own_history", case_id, "trade_executes_even_when_history_threshold_unmet", 1, int(stats["trade_count"]))
    add_check(results, "synthetic_single_backtest_not_gated_by_own_history", case_id, "history_filter_result_can_still_be_false_after_trade", False, bool(stats["is_candidate"]))
    add_check(results, "synthetic_single_backtest_not_gated_by_own_history", case_id, "position_closed_after_indicator_sell", 0, int(stats["current_position"]))
    add_check(results, "synthetic_single_backtest_not_gated_by_own_history", case_id, "winning_trade_recorded", True, float(stats["asset_growth"]) > 0.0)

    summary["trade_count"] = int(stats["trade_count"])
    summary["is_candidate"] = bool(stats["is_candidate"])
    return results, summary


def validate_synthetic_single_backtest_uses_compounding_capital_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.initial_capital = 1000.0
    params.fixed_risk = 0.1
    params.buy_fee = 0.0
    params.sell_fee = 0.0
    params.tax_rate = 0.0
    params.min_fee = 0.0
    params.atr_buy_tol = 0.0
    params.atr_times_init = 1.0
    params.atr_times_trail = 1.0
    params.use_compounding = True
    params.max_position_cap_pct = 1.0

    case_id = "SYNTH_SINGLE_BACKTEST_COMPOUNDING_CAPITAL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 110.0, 100.0, 110.0, 100.0],
            "High": [100.0, 100.0, 110.0, 100.0, 110.0, 100.0],
            "Low": [100.0, 100.0, 110.0, 100.0, 110.0, 100.0],
            "Close": [100.0, 100.0, 110.0, 100.0, 110.0, 100.0],
            "Volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ]),
    )
    precomputed_signals = (
        np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64),
        np.array([True, False, True, False, False, True], dtype=bool),
        np.array([False, True, False, True, False, False], dtype=bool),
        np.array([100.0, np.nan, 100.0, np.nan, np.nan, 100.0], dtype=np.float64),
    )

    stats = run_v16_backtest(df, params, precomputed_signals=precomputed_signals)

    add_check(results, "synthetic_single_backtest_compounding_capital", case_id, "trade_count_after_two_round_trips", 2, int(stats["trade_count"]))
    add_check(results, "synthetic_single_backtest_compounding_capital", case_id, "asset_growth_uses_compounding_capital_when_flag_true", 21.0, float(stats["asset_growth"]))
    add_check(results, "synthetic_single_backtest_compounding_capital", case_id, "score_uses_compounded_trade_sequence", 10.5, float(stats["score"]))
    add_check(results, "synthetic_single_backtest_compounding_capital", case_id, "final_setup_today_survives_after_compounding_replays", True, bool(stats["is_setup_today"]))

    summary["trade_count"] = int(stats["trade_count"])
    summary["asset_growth"] = float(stats["asset_growth"])
    summary["score"] = float(stats["score"])
    return results, summary



def validate_synthetic_portfolio_history_filter_only_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.min_history_trades = 5
    params.min_history_ev = 0.5
    params.min_history_win_rate = 0.8

    case_id = "SYNTH_PORTFOLIO_HISTORY_FILTER_ONLY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from .synthetic_portfolio_common import build_synthetic_half_tp_full_year_case
    from .synthetic_fixtures import write_synthetic_csv_bundle
    from .tool_adapters import run_scanner_tool_check
    import os
    import tempfile

    case = build_synthetic_half_tp_full_year_case(base_params)
    ticker = case["primary_ticker"]

    with tempfile.TemporaryDirectory() as temp_dir:
        write_synthetic_csv_bundle(temp_dir, case["frames"])
        file_path = os.path.join(temp_dir, f"{ticker}.csv")
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, _sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

        single_stats = run_v16_backtest(df.copy(), params)
        scanner_ref_stats = run_scanner_reference_check(ticker, file_path, params)
        scanner_result, _scanner_module_path = run_scanner_tool_check(
            ticker,
            file_path,
            params,
            precomputed_stats=scanner_ref_stats,
        )

        add_check(results, "synthetic_portfolio_history_filter_only", case_id, "single_backtest_trade_executes_even_when_history_threshold_unmet", True, int(single_stats["trade_count"]) > 0)
        add_check(results, "synthetic_portfolio_history_filter_only", case_id, "single_backtest_history_gate_remains_false", False, bool(single_stats["is_candidate"]))
        add_check(results, "synthetic_portfolio_history_filter_only", case_id, "scanner_reference_candidate_remains_false", False, bool(scanner_ref_stats["is_candidate"]))
        add_check(results, "synthetic_portfolio_history_filter_only", case_id, "scanner_tool_rejects_non_candidate_history_gate", None, None if scanner_result is None else scanner_result.get("status"), note="history filter 僅能作用於投組層 / scanner，不得回頭阻斷單股回測本身。")

    summary["single_trade_count"] = int(single_stats["trade_count"])
    summary["scanner_status"] = None if scanner_result is None else scanner_result.get("status")
    return results, summary




def validate_synthetic_pit_multiple_same_day_exits_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.min_history_trades = 2
    params.min_history_ev = 0.1
    params.min_history_win_rate = 0.5

    case_id = "SYNTH_PIT_MULTIPLE_SAME_DAY_EXITS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    trade_logs = [
        {"exit_date": datetime(2024, 1, 3), "pnl": 120.0, "r_mult": 1.2},
        {"exit_date": datetime(2024, 1, 2), "pnl": 50.0, "r_mult": 0.5},
        {"exit_date": datetime(2024, 1, 2), "pnl": -20.0, "r_mult": -0.2},
    ]
    stats_index = build_trade_stats_index(trade_logs)

    same_day_candidate, same_day_ev, same_day_win_rate, same_day_trade_count, _same_day_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 2),
        params,
    )
    next_day_candidate, next_day_ev, next_day_win_rate, next_day_trade_count, _next_day_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 3),
        params,
    )
    after_all_candidate, after_all_ev, after_all_win_rate, after_all_trade_count, _after_all_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 4),
        params,
    )

    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "same_day_trade_count_excludes_all_same_day_exits", 0, same_day_trade_count)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "same_day_candidate_stays_blocked_without_prior_day_history", False, same_day_candidate)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "same_day_expected_value_excludes_all_same_day_exits", 0.0, same_day_ev)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "same_day_win_rate_excludes_all_same_day_exits", 0.0, same_day_win_rate)

    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "next_day_trade_count_includes_only_prior_day_exits", 2, next_day_trade_count)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "next_day_expected_value_uses_prior_day_batch_only", 0.15, next_day_ev, tol=1e-9)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "next_day_win_rate_uses_prior_day_batch_only", 0.5, next_day_win_rate, tol=1e-9)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "next_day_candidate_uses_prior_day_batch_only", True, next_day_candidate, note="同日多筆 exit 也必須整批排除，僅能在下一交易日一起進入 PIT 歷史。")

    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "after_all_trade_count_includes_later_day_exit", 3, after_all_trade_count)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "after_all_expected_value_includes_full_history", 0.5, after_all_ev, tol=1e-9)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "after_all_win_rate_includes_full_history", 2.0 / 3.0, after_all_win_rate, tol=1e-9)
    add_check(results, "synthetic_pit_multiple_same_day_exits", case_id, "after_all_candidate_includes_full_history", True, after_all_candidate)

    summary["next_day_trade_count"] = int(next_day_trade_count)
    summary["after_all_trade_count"] = int(after_all_trade_count)
    return results, summary

def validate_synthetic_lookahead_prev_day_only_case(base_params):
    params = make_synthetic_validation_params(base_params, tp_percent=0.0)
    params.min_history_trades = 1
    params.min_history_ev = 0.5
    params.min_history_win_rate = 0.5

    case_id = "SYNTH_LOOKAHEAD_PREV_DAY_ONLY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    exit_dates = [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    stats_index = {
        "exit_dates": exit_dates,
        "cum_trade_count": np.array([1, 1], dtype=np.int32),
        "cum_win_count": np.array([1, 1], dtype=np.int32),
        "cum_win_r_sum": np.array([1.0, 1.0], dtype=np.float64),
        "cum_loss_r_sum": np.array([0.0, 0.0], dtype=np.float64),
        "cum_total_r_sum": np.array([1.0, 1.0], dtype=np.float64),
        "cum_pnl_sum": np.array([100.0, 100.0], dtype=np.float64),
    }

    same_day_candidate, same_day_ev, same_day_win_rate, same_day_trade_count, _same_day_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 2),
        params,
    )
    next_day_candidate, next_day_ev, next_day_win_rate, next_day_trade_count, _next_day_asset_growth = get_pit_stats_from_index(
        stats_index,
        datetime(2024, 1, 3),
        params,
    )

    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "same_day_trade_count_uses_previous_day_only", 0, same_day_trade_count)
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "same_day_candidate_blocked_without_prior_history", False, same_day_candidate)
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "same_day_expected_value_excludes_same_day_exit", 0.0, same_day_ev)
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "same_day_win_rate_excludes_same_day_exit", 0.0, same_day_win_rate)
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "next_day_trade_count_includes_prior_exit", 1, next_day_trade_count)
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "next_day_candidate_unlocked_by_prior_exit_only", True, next_day_candidate, note="盤前決策只能讀前一日已完成歷史，不得偷看同日才剛結束的交易。")
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "next_day_expected_value_includes_prior_exit", 1.0, next_day_ev)
    add_check(results, "synthetic_lookahead_prev_day_only", case_id, "next_day_win_rate_includes_prior_exit", 1.0, next_day_win_rate)

    summary["same_day_trade_count"] = int(same_day_trade_count)
    summary["next_day_trade_count"] = int(next_day_trade_count)
    return results, summary


def validate_synthetic_setup_index_prev_day_only_case(base_params):
    _params = make_synthetic_validation_params(base_params, tp_percent=0.0)

    case_id = "SYNTH_SETUP_INDEX_PREV_DAY_ONLY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    packed_df = pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.0],
            "High": [10.2, 10.7, 11.3],
            "Low": [9.8, 10.1, 10.7],
            "Close": [10.0, 10.6, 11.1],
            "Volume": [1000.0, 1000.0, 1000.0],
            "ATR": [1.0, 1.0, 1.0],
            "buy_limit": [10.0, 10.5, 11.0],
            "is_setup": [False, True, False],
            "ind_sell_signal": [False, False, False],
        },
        index=index,
    )
    packed_index = build_normal_setup_index({"2330": pack_prepared_stock_data(packed_df)})

    dict_index = build_normal_setup_index({
        "2330": {
            index[0]: {"is_setup": False},
            index[1]: {"is_setup": True},
            index[2]: {"is_setup": False},
        }
    })

    day_two_entries = packed_index.get(index[1], [])
    day_three_entries = packed_index.get(index[2], [])

    add_check(results, "synthetic_setup_index_prev_day_only", case_id, "same_day_setup_not_exposed_to_same_day_schedule", [], day_two_entries)
    add_check(results, "synthetic_setup_index_prev_day_only", case_id, "next_day_schedule_contains_previous_day_setup_only", [("2330", 1, 2)], day_three_entries)
    add_check(results, "synthetic_setup_index_prev_day_only", case_id, "packed_and_dict_paths_match", dict_index, packed_index)

    summary["scheduled_dates"] = [dt.strftime("%Y-%m-%d") for dt in packed_index.keys()]
    return results, summary
