import os

import pandas as pd

from core.v16_data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.v16_portfolio_engine import run_portfolio_timeline
from core.v16_portfolio_fast_data import pack_prepared_stock_data, prep_stock_data_and_trades

from .checks import add_check, build_portfolio_stats_payload, make_synthetic_validation_params
from .synthetic_fixtures import (
    build_synthetic_competing_candidates_case as fixture_build_synthetic_competing_candidates_case,
    build_synthetic_extended_miss_buy_case as fixture_build_synthetic_extended_miss_buy_case,
    build_synthetic_half_tp_full_year_case as fixture_build_synthetic_half_tp_full_year_case,
    build_synthetic_rotation_t_plus_one_case as fixture_build_synthetic_rotation_t_plus_one_case,
    build_synthetic_same_day_sell_block_case as fixture_build_synthetic_same_day_sell_block_case,
    build_synthetic_unexecutable_half_tp_case as fixture_build_synthetic_unexecutable_half_tp_case,
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
