"""Portfolio terminal summaries assembled from existing canonical statistics.

The underlying R, period-return, and annualization formulas remain in
core.portfolio_stats. This module owns only the end-of-run coordination.
"""

from typing import NamedTuple

from core.portfolio_fast_data import summarize_single_stock_trade_stats_from_pit_index
from core.portfolio_stats import (
    build_benchmark_full_month_return_stats,
    build_benchmark_full_quarter_return_stats,
    build_benchmark_full_year_return_stats,
    build_full_month_return_stats,
    build_full_quarter_return_stats,
    build_full_year_return_stats,
    calc_annual_return_pct,
    calc_sim_years,
    summarize_closed_trade_entry_type_counts,
    summarize_closed_trade_r_stats,
)


class PortfolioTerminalMetrics(NamedTuple):
    trade_count: int
    portfolio_r_stats: dict
    normal_trade_count: int
    extended_trade_count: int
    breakout_trade_count: int
    reentry_trade_count: int
    score_single_stock_trade_stats: dict
    win_rate: float
    pf_ev: float
    pf_payoff: float
    avg_exp: float
    sim_years: float
    annual_trades: float
    filled_buy_count: int
    reserved_buy_fill_rate: float
    annual_return_pct: float
    bm_annual_return_pct: float


class PortfolioPeriodReports(NamedTuple):
    yearly_stats: dict
    monthly_stats: dict
    quarterly_stats: dict
    bm_yearly_stats: dict
    bm_monthly_stats: dict
    bm_quarterly_stats: dict


def summarize_portfolio_terminal_metrics(
    *, closed_trades_stats, pit_stats_index, sorted_dates, start_idx,
    profile_requested, active_context_resolver, active_context_ensemble_resolver,
    ev_calc_method_getter, total_exposure, sim_days, portfolio_entry_stats,
    total_missed_buys, initial_capital, final_equity_money, benchmark_start_price,
    bm_ret_pct, benchmark_period_stats,
) -> PortfolioTerminalMetrics:
    trade_count = len(closed_trades_stats)
    portfolio_r_stats = summarize_closed_trade_r_stats(closed_trades_stats)
    entry_type_counts = summarize_closed_trade_entry_type_counts(closed_trades_stats)
    normal_trade_count = int(entry_type_counts.get('normal_trades', 0))
    extended_trade_count = int(entry_type_counts.get('extended_trades', 0))
    breakout_trade_count = int(entry_type_counts.get('breakout_trades', normal_trade_count))
    reentry_trade_count = int(entry_type_counts.get('reentry_trades', 0))
    score_single_stock_trade_stats = {}
    if profile_requested and active_context_resolver is None and active_context_ensemble_resolver is None:
        score_single_stock_trade_stats = summarize_single_stock_trade_stats_from_pit_index(
            pit_stats_index,
            sorted_dates[start_idx:],
        )
    if trade_count > 0:
        wins = [t for t in closed_trades_stats if t['pnl'] > 0]
        losses = [t for t in closed_trades_stats if t['pnl'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / trade_count) * 100
        avg_win_amount = sum(t['pnl'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_amount = abs(sum(t['pnl'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        pf_payoff = (avg_win_amount / avg_loss_amount) if avg_loss_amount > 0 else (99.9 if avg_win_amount > 0 else 0.0)

        avg_win_r = sum(t['r_mult'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_r = abs(sum(t['r_mult'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        if ev_calc_method_getter() == 'B':
            payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
            pf_ev = (win_rate / 100.0 * payoff_for_ev) - (1 - win_rate / 100.0)
        else:
            pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / trade_count
    else:
        win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0

    avg_exp = total_exposure / sim_days if sim_days > 0 else 0.0
    sim_years = calc_sim_years(sorted_dates, start_idx)
    annual_trades = (trade_count / sim_years) if sim_years > 0 else 0.0
    filled_buy_count = int(portfolio_entry_stats.get('filled_buy_count', 0) or 0)
    reserved_buy_fill_rate = (filled_buy_count / (filled_buy_count + total_missed_buys) * 100.0) if (filled_buy_count + total_missed_buys) > 0 else 0.0
    annual_return_pct = calc_annual_return_pct(initial_capital, final_equity_money, sim_years)

    bm_start_value = float(benchmark_start_price) if benchmark_start_price is not None else 0.0
    bm_end_value = bm_start_value * (1.0 + bm_ret_pct / 100.0) if bm_start_value > 0 else 0.0
    if benchmark_period_stats is not None:
        bm_annual_return_pct = float(benchmark_period_stats.get('bm_annual_return_pct', 0.0))
    else:
        bm_annual_return_pct = calc_annual_return_pct(bm_start_value, bm_end_value, sim_years)

    return PortfolioTerminalMetrics(
        trade_count, portfolio_r_stats, normal_trade_count, extended_trade_count,
        breakout_trade_count, reentry_trade_count, score_single_stock_trade_stats,
        win_rate, pf_ev, pf_payoff, avg_exp, sim_years, annual_trades,
        filled_buy_count, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct,
    )


def build_portfolio_period_reports(
    *, sorted_dates,
    year_start_equity, year_end_equity, year_first_sim_date, year_last_sim_date,
    month_start_equity, month_end_equity, month_first_sim_date, month_last_sim_date,
    quarter_start_equity, quarter_end_equity, quarter_first_sim_date, quarter_last_sim_date,
    benchmark_period_stats, benchmark_data,
) -> PortfolioPeriodReports:
    yearly_stats = build_full_year_return_stats(
        sorted_dates=sorted_dates,
        year_start_equity=year_start_equity,
        year_end_equity=year_end_equity,
        year_first_sim_date=year_first_sim_date,
        year_last_sim_date=year_last_sim_date,
    )
    monthly_stats = build_full_month_return_stats(
        sorted_dates=sorted_dates,
        month_start_equity=month_start_equity,
        month_end_equity=month_end_equity,
        month_first_sim_date=month_first_sim_date,
        month_last_sim_date=month_last_sim_date,
    )
    quarterly_stats = build_full_quarter_return_stats(
        sorted_dates=sorted_dates,
        quarter_start_equity=quarter_start_equity,
        quarter_end_equity=quarter_end_equity,
        quarter_first_sim_date=quarter_first_sim_date,
        quarter_last_sim_date=quarter_last_sim_date,
    )
    if benchmark_period_stats is not None:
        full_years = {int(row['year']) for row in yearly_stats['yearly_return_rows'] if row.get('is_full_year')}
        bm_yearly_rows = [
            row for row in benchmark_period_stats.get('bm_yearly_return_rows', [])
            if int(row.get('year', -1)) in full_years
        ]
        bm_yearly_stats = {
            'bm_full_year_count': int(len(bm_yearly_rows)),
            'bm_min_full_year_return_pct': float(min((row['year_return_pct'] for row in bm_yearly_rows), default=0.0)),
            'bm_yearly_return_rows': bm_yearly_rows,
        }
        full_months = {(int(row['year']), int(row['month'])) for row in monthly_stats['monthly_return_rows'] if row.get('is_full_month')}
        bm_monthly_rows = [
            row for row in benchmark_period_stats.get('bm_monthly_return_rows', [])
            if (int(row.get('year', -1)), int(row.get('month', -1))) in full_months
        ]
        bm_monthly_stats = {
            'bm_full_month_count': int(len(bm_monthly_rows)),
            'bm_min_month_return_pct': float(min((row['month_return_pct'] for row in bm_monthly_rows), default=0.0)),
            'bm_monthly_return_rows': bm_monthly_rows,
        }
        full_quarters = {(int(row['year']), int(row['quarter'])) for row in quarterly_stats['quarterly_return_rows'] if row.get('is_full_quarter')}
        bm_quarterly_rows = [
            row for row in benchmark_period_stats.get('bm_quarterly_return_rows', [])
            if (int(row.get('year', -1)), int(row.get('quarter', -1))) in full_quarters
        ]
        bm_quarterly_stats = {
            'bm_full_quarter_count': int(len(bm_quarterly_rows)),
            'bm_min_quarter_return_pct': float(min((row['quarter_return_pct'] for row in bm_quarterly_rows), default=0.0)),
            'bm_quarterly_return_rows': bm_quarterly_rows,
        }
    else:
        bm_yearly_stats = build_benchmark_full_year_return_stats(
            sorted_dates=sorted_dates,
            benchmark_data=benchmark_data,
            yearly_return_rows=yearly_stats['yearly_return_rows'],
        )
        bm_monthly_stats = build_benchmark_full_month_return_stats(
            sorted_dates=sorted_dates,
            benchmark_data=benchmark_data,
            monthly_return_rows=monthly_stats['monthly_return_rows'],
        )
        bm_quarterly_stats = build_benchmark_full_quarter_return_stats(
            sorted_dates=sorted_dates,
            benchmark_data=benchmark_data,
            quarterly_return_rows=quarterly_stats['quarterly_return_rows'],
        )

    return PortfolioPeriodReports(
        yearly_stats, monthly_stats, quarterly_stats,
        bm_yearly_stats, bm_monthly_stats, bm_quarterly_stats,
    )
