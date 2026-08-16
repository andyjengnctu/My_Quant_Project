"""Canonical human-report metric metadata shared by strategy renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportMetricSpec:
    key: str
    label: str
    unit: str = ""
    digits: int = 2
    preference: str = "neutral"
    warning_threshold: float | None = None


PORTFOLIO_RESULT_METRICS = (
    ReportMetricSpec("total_return_pct", "報酬", "%", 2, "higher"),
    ReportMetricSpec("max_drawdown_pct", "MDD", "%", 2, "lower"),
    ReportMetricSpec("return_over_max_drawdown", "RoMD", "", 2, "higher"),
    ReportMetricSpec("annual_return_pct", "年化", "%", 2, "higher"),
    ReportMetricSpec("log_r_squared", "Log R²", "", 4, "higher"),
    ReportMetricSpec("monthly_win_rate_pct", "月勝率", "%", 2, "higher"),
    ReportMetricSpec("min_full_year_return_pct", "最差完整年度", "%", 2, "higher"),
    ReportMetricSpec("avg_exposure_pct", "平均曝險", "%", 2, "attention", 5.0),
)

TRADE_RESULT_METRICS = (
    ReportMetricSpec("trade_count", "交易數", "", 0, "neutral"),
    ReportMetricSpec("win_rate_pct", "勝率", "%", 2, "higher"),
    ReportMetricSpec("payoff_ratio", "Payoff", "", 2, "higher"),
    ReportMetricSpec("expected_value_r", "EV", " R", 2, "higher"),
    ReportMetricSpec("portfolio_avg_r", "平均R", " R", 2, "higher"),
    ReportMetricSpec("portfolio_median_r", "中位R", " R", 2, "higher"),
    ReportMetricSpec("reserved_buy_fill_rate_pct", "掛單成交率", "%", 2, "higher"),
)


CORE_STRATEGY_RESULT_METRICS = (
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "total_return_pct"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "max_drawdown_pct"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "return_over_max_drawdown"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "annual_return_pct"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "min_full_year_return_pct"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "log_r_squared"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "monthly_win_rate_pct"),
    next(metric for metric in TRADE_RESULT_METRICS if metric.key == "win_rate_pct"),
    next(metric for metric in TRADE_RESULT_METRICS if metric.key == "payoff_ratio"),
    next(metric for metric in TRADE_RESULT_METRICS if metric.key == "expected_value_r"),
    next(metric for metric in TRADE_RESULT_METRICS if metric.key == "trade_count"),
    next(metric for metric in PORTFOLIO_RESULT_METRICS if metric.key == "avg_exposure_pct"),
)

EXECUTION_CAPACITY_METRICS = (
    ReportMetricSpec("avg_orderable_candidates", "日均可掛候選", "", 2, "neutral"),
    ReportMetricSpec("candidate_supply_gap_days", "候選不足日", " 日", 0, "lower"),
    ReportMetricSpec("underfilled_end_days", "期末未滿倉日", " 日", 0, "lower"),
    ReportMetricSpec("end_position_gap_slot_days", "持股缺口", " 格日", 0, "lower"),
)

# Pair reports keep the same canonical metrics as before, but source their labels,
# units and direction semantics from this shared registry.
PAIR_MAIN_METRICS = (
    *PORTFOLIO_RESULT_METRICS[:6],
    *TRADE_RESULT_METRICS[:4],
    PORTFOLIO_RESULT_METRICS[-1],
    PORTFOLIO_RESULT_METRICS[-2],
    *EXECUTION_CAPACITY_METRICS,
)


__all__ = [
    "ReportMetricSpec",
    "PORTFOLIO_RESULT_METRICS",
    "CORE_STRATEGY_RESULT_METRICS",
    "TRADE_RESULT_METRICS",
    "EXECUTION_CAPACITY_METRICS",
    "PAIR_MAIN_METRICS",
]
