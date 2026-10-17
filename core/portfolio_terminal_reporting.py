"""Publish canonical portfolio terminal statistics without re-computing them.

Both legacy tuple shapes derive from core.portfolio_run_result; training,
reporting and REUSE consumers keep the same ordering and field semantics.
"""

import time

import pandas as pd

from core.exact_accounting import milli_to_money
from core.fee_rebate import get_fee_rebate_receivable_milli, get_fee_rebate_settled_milli
from core.portfolio_attribution import build_dominant_year_dependency_diagnostics
from core.portfolio_fast_data import build_score_single_stock_profile_fields
from core.portfolio_run_result import PortfolioReportingResult, PortfolioTrainingResult


def publish_portfolio_profile_stats(
    *, profile_stats, fee_rebate_state, timing_enabled, portfolio_start,
    timings, candidate_subprofile, idle_fast_path_days, closed_trades_stats,
    terminal_metrics, period_reports, is_training, active_level_rows,
    capacity_rows, capture_equity_curve, equity_curve,
    benchmark_report_yearly_rows=None,
):
    """Mutate only the existing optional profile sidecar, preserving field order."""
    if profile_stats is None:
        return
    # Read-only reference rows from the existing canonical benchmark producer.
    # Do not alter bm_yearly_return_rows, which historically contains only
    # complete-calendar-year statistics used elsewhere in portfolio research.
    if not is_training:
        profile_stats['bm_report_yearly_return_rows'] = [
            dict(row) for row in (benchmark_report_yearly_rows or ())
        ]
    profile_stats['fee_rebate_settled'] = milli_to_money(get_fee_rebate_settled_milli(fee_rebate_state))
    profile_stats['fee_rebate_receivable_end'] = milli_to_money(get_fee_rebate_receivable_milli(fee_rebate_state))
    profile_stats['portfolio_wall_sec'] = (time.perf_counter() - portfolio_start) if timing_enabled else 0.0
    profile_stats['portfolio_ticker_dates_sec'] = timings['ticker_dates_sec']
    profile_stats['portfolio_build_trade_index_sec'] = timings['build_trade_index_sec']
    profile_stats['portfolio_day_loop_sec'] = timings['day_loop_sec']
    candidate_scan_sec = timings['candidate_scan_sec']
    profile_stats['portfolio_candidate_scan_sec'] = candidate_scan_sec
    candidate_operation_keys = (
        'portfolio_candidate_pit_stats_sec',
        'portfolio_candidate_plan_sec',
        'portfolio_candidate_quality_rank_sec',
        'portfolio_candidate_row_sec',
        'portfolio_candidate_sort_sec',
    )
    candidate_profile_keys = (
        'portfolio_candidate_normal_sec',
        'portfolio_candidate_extended_sec',
        *candidate_operation_keys,
    )
    for key in candidate_profile_keys:
        profile_stats[key] = float((candidate_subprofile or {}).get(key, 0.0))
    candidate_operation_sec = sum(float(profile_stats[key]) for key in candidate_operation_keys)
    profile_stats['portfolio_candidate_other_sec'] = (
        max(0.0, float(candidate_scan_sec) - candidate_operation_sec)
        if candidate_subprofile is not None
        else 0.0
    )
    for key in (
        'day_setup_sec', 'reentry_sec', 'resource_select_sec', 'cleanup_sec',
        'bookkeeping_sec', 'rotation_sec', 'settle_sec', 'buy_sec', 'equity_mark_sec',
    ):
        profile_stats['portfolio_' + key] = timings[key]
    day_accounted_sec = (
        float(candidate_scan_sec)
        + float(timings['day_setup_sec'])
        + float(timings['reentry_sec'])
        + float(timings['resource_select_sec'])
        + float(timings['rotation_sec'])
        + float(timings['settle_sec'])
        + float(timings['buy_sec'])
        + float(timings['cleanup_sec'])
        + float(timings['equity_mark_sec'])
        + float(timings['bookkeeping_sec'])
    )
    profile_stats['portfolio_day_other_sec'] = max(0.0, float(timings['day_loop_sec']) - day_accounted_sec)
    profile_stats['portfolio_closeout_sec'] = timings['closeout_sec']
    profile_stats['curve_stats_sec'] = timings['curve_stats_sec']
    profile_stats['idle_fast_path_days'] = int(idle_fast_path_days)
    profile_stats['dominant_year_dependency_diagnostics'] = build_dominant_year_dependency_diagnostics(closed_trades_stats)
    profile_stats['sim_years'] = terminal_metrics.sim_years
    profile_stats['annual_return_pct'] = terminal_metrics.annual_return_pct
    profile_stats['bm_annual_return_pct'] = terminal_metrics.bm_annual_return_pct
    profile_stats['reserved_buy_fill_rate'] = terminal_metrics.reserved_buy_fill_rate
    profile_stats['portfolio_total_r'] = float(terminal_metrics.portfolio_r_stats.get('total_r', 0.0))
    profile_stats['portfolio_median_r'] = float(terminal_metrics.portfolio_r_stats.get('median_r', 0.0))
    profile_stats['portfolio_avg_r'] = float(terminal_metrics.portfolio_r_stats.get('avg_r', 0.0))
    if not is_training:
        # # (AI註: 一筆一列的扣費後 round-trip 稽核資料；只供報表歸因，不改交易或統計口徑。)
        profile_stats['closed_trade_rows'] = [dict(row) for row in closed_trades_stats]
    profile_stats['breakout_trades'] = int(terminal_metrics.breakout_trade_count)
    profile_stats['reentry_trades'] = int(terminal_metrics.reentry_trade_count)
    profile_stats['normal_trades'] = int(terminal_metrics.normal_trade_count)
    profile_stats['extended_trades'] = int(terminal_metrics.extended_trade_count)
    if terminal_metrics.score_single_stock_trade_stats:
        profile_stats.update(build_score_single_stock_profile_fields(terminal_metrics.score_single_stock_trade_stats))
    profile_stats['filled_buy_count'] = terminal_metrics.filled_buy_count
    if active_level_rows is not None:
        profile_stats['portfolio_active_level_rows'] = list(active_level_rows)
    if capacity_rows is not None:
        profile_stats['portfolio_capacity_rows'] = list(capacity_rows)
    if capture_equity_curve:
        profile_stats['equity_curve'] = list(equity_curve)
    for stats in period_reports:
        profile_stats.update(stats)


def assemble_portfolio_timeline_result(
    *, is_training, equity_curve, trade_history, benchmark_ticker, terminal_metrics,
    total_return, max_drawdown, final_equity_money, max_exp, bm_ret_pct,
    bm_max_drawdown, total_missed_buys, total_missed_sells,
    r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate,
) -> tuple:
    """Use the pre-existing named result contracts, return original plain tuples."""
    metrics = terminal_metrics
    if is_training:
        return tuple(PortfolioTrainingResult(
            total_return, max_drawdown, metrics.trade_count, final_equity_money,
            metrics.avg_exp, max_exp, bm_ret_pct, bm_max_drawdown,
            metrics.win_rate, metrics.pf_ev, metrics.pf_payoff,
            total_missed_buys, total_missed_sells, r_squared, monthly_win_rate,
            bm_r_squared, bm_monthly_win_rate,
            metrics.normal_trade_count, metrics.extended_trade_count,
            metrics.annual_trades, metrics.reserved_buy_fill_rate,
            metrics.annual_return_pct, metrics.bm_annual_return_pct,
        ))

    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_history)
    final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
    return tuple(PortfolioReportingResult(
        df_equity, df_trades, total_return, max_drawdown, metrics.trade_count,
        metrics.win_rate, metrics.pf_ev, metrics.pf_payoff, final_equity_money,
        metrics.avg_exp, max_exp, final_bm_return, bm_max_drawdown,
        total_missed_buys, total_missed_sells, r_squared, monthly_win_rate,
        bm_r_squared, bm_monthly_win_rate,
        metrics.normal_trade_count, metrics.extended_trade_count,
        metrics.annual_trades, metrics.reserved_buy_fill_rate,
        metrics.annual_return_pct, metrics.bm_annual_return_pct,
    ))
