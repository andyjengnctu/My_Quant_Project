"""Canonical typed views for the unchanged portfolio timeline tuple contracts.

The timeline owner remains core.portfolio_engine.run_portfolio_timeline.
These views only name its pre-existing ordered return values; they do not
recalculate metrics or alter the trading/accounting semantics.
"""

from collections import namedtuple
from typing import Any, NamedTuple


class PortfolioTrainingResult(NamedTuple):
    """23-position result of run_portfolio_timeline(is_training=True)."""

    total_return_pct: float
    max_drawdown_pct: float
    trade_count: int
    final_equity: float
    average_exposure: float
    maximum_exposure: float
    benchmark_return_pct: float
    benchmark_max_drawdown_pct: float
    win_rate_pct: float
    portfolio_ev: float
    portfolio_payoff: float
    missed_buy_count: int
    missed_sell_count: int
    log_r_squared: float
    monthly_win_rate_pct: float
    benchmark_log_r_squared: float
    benchmark_monthly_win_rate_pct: float
    normal_trade_count: int
    extended_trade_count: int
    annual_trades: float
    reserved_buy_fill_rate: float
    annual_return_pct: float
    benchmark_annual_return_pct: float


class PortfolioReportingResult(NamedTuple):
    """25-position result of run_portfolio_timeline(is_training=False)."""

    equity_curve: Any
    trade_history: Any
    total_return_pct: float
    max_drawdown_pct: float
    trade_count: int
    win_rate_pct: float
    portfolio_ev: float
    portfolio_payoff: float
    final_equity: float
    average_exposure: float
    maximum_exposure: float
    benchmark_return_pct: float
    benchmark_max_drawdown_pct: float
    missed_buy_count: int
    missed_sell_count: int
    log_r_squared: float
    monthly_win_rate_pct: float
    benchmark_log_r_squared: float
    benchmark_monthly_win_rate_pct: float
    normal_trade_count: int
    extended_trade_count: int
    annual_trades: float
    reserved_buy_fill_rate: float
    annual_return_pct: float
    benchmark_annual_return_pct: float


def as_portfolio_training_result(values) -> PortfolioTrainingResult:
    """Name an existing training tuple, including test doubles with the legacy API."""
    if isinstance(values, PortfolioTrainingResult):
        return values
    return PortfolioTrainingResult(*values)


def as_portfolio_reporting_result(values) -> PortfolioReportingResult:
    """Name an existing non-training tuple without changing the legacy output."""
    if isinstance(values, PortfolioReportingResult):
        return values
    return PortfolioReportingResult(*values)


# The replay service appends a diagnostic profile to the unchanged 25-field
# reporting tuple. Its ordered schema must derive from PortfolioReportingResult,
# never from a separately maintained 26-position field list.
PortfolioReplayResult = namedtuple(
    "PortfolioReplayResult",
    (*PortfolioReportingResult._fields, "profile"),
    module=__name__,
)


def as_portfolio_replay_result(values) -> PortfolioReplayResult:
    """Name the existing 26-field replay tuple without reordering values."""
    if isinstance(values, PortfolioReplayResult):
        return values
    return PortfolioReplayResult(*values)


def append_portfolio_replay_profile(report_values, profile) -> tuple:
    """Return the original plain 26-tuple shape, validated against its sole owner."""
    as_portfolio_reporting_result(report_values)
    return (*report_values, profile)
