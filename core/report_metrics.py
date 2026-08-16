"""Canonical human-report metric metadata shared by strategy renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ReportMetricSpec:
    key: str
    label: str
    unit: str = ""
    digits: int = 2
    preference: str = "neutral"
    warning_threshold: float | None = None


@dataclass(frozen=True)
class RAnalysisMetricSpec:
    key: str
    label: str
    definition: str
    ideal_direction: str
    unit: str = ""
    digits: int = 2
    format_kind: str = "number"
    preference: str = "neutral"
    signal_mode: str = "relative"


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



R_ACTUAL_TRADE_METRICS = (
    RAnalysisMetricSpec(
        "portfolio_avg_r", "平均R", "實際完成交易 realized R 的平均值", "越高越好",
        unit=" R", digits=2, preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "portfolio_median_r", "中位R", "實際完成交易 realized R 的中位數", "越高越好",
        unit=" R", digits=2, preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "score_coverage", "Coverage", "可掛候選中可取得合法 Score 的比例", "越接近 100% 越好",
        digits=2, format_kind="fraction_pct", preference="higher", signal_mode="coverage",
    ),
    RAnalysisMetricSpec(
        "direct_selection_delta_r", "DL選擇R", "DL 獨有交易總R－同參數 DL-off 獨有交易總R", "> 0，越高越好",
        unit=" R", digits=2, preference="higher", signal_mode="signed",
    ),
)

R_MODEL_PREDICTION_METRICS = (
    RAnalysisMetricSpec(
        "mean_daily_spearman", "Dailyρ", "逐日 Score 與 Target R 橫截面 Spearman 的平均", "> 0，越高越好",
        digits=3, preference="higher", signal_mode="signed",
    ),
    RAnalysisMetricSpec(
        "global_spearman", "Globalρ", "全樣本 Score 與 Target R 的 Spearman", "> 0，越高越好",
        digits=3, preference="higher", signal_mode="signed",
    ),
    RAnalysisMetricSpec(
        "pairwise_concordance", "Pair一致", "Score 與 Target R 成對高低排序的一致率", "> 50%，越高越好",
        digits=2, format_kind="fraction_pct", preference="higher", signal_mode="auc",
    ),
    RAnalysisMetricSpec(
        "top_target_r", "Top-R", "最高 Score 十分位的平均 Target R", "越高越好",
        unit=" R", digits=2, preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "bottom_target_r", "Bottom-R", "最低 Score 十分位的平均 Target R", "越低越好",
        unit=" R", digits=2, preference="lower", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "top_bottom_target_spread_r", "Top-BottomR", "Top-R－Bottom-R", "> 0，越高越好",
        unit=" R", digits=2, preference="higher", signal_mode="signed",
    ),
)

R_SELECTION_TRANSLATION_METRICS = (
    RAnalysisMetricSpec(
        "r_conversion_efficiency", "RCE",
        "同一批有Future Target的exclusive completed trades之 realized mean-R edge ÷ Target mean-R edge；衡量預測品質edge轉成實際交易品質edge的效率",
        "兩側皆有Target-covered exclusive trades且Target mean-R edge > 0時有效；越高越好，100%代表等量轉化",
        digits=2, format_kind="fraction_pct", preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "selected_target_mean_r", "Target mean R", "實際選入候選的平均 Target R", "越高越好",
        unit=" R", digits=2, preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "selected_target_percentile", "Target %ile", "實際選入候選在當日 Target R 的平均百分位", "> 0.5，越高越好",
        digits=3, preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "target_top_k_retention", "Top-K", "實際選入 K 檔與真實 Target Top-K 的平均重疊率", "越高越好",
        digits=2, format_kind="fraction_pct", preference="higher", signal_mode="relative",
    ),
    RAnalysisMetricSpec(
        "target_opportunity_gap_r", "Opp gap", "真實 Target Top-K 平均R－實際選入平均 Target R", "越接近 0 越好",
        unit=" R", digits=2, preference="lower", signal_mode="relative",
    ),
)

R_ANALYSIS_GROUPED_SECTIONS: Tuple[tuple[str, tuple[RAnalysisMetricSpec, ...]], ...] = (
    ("實際交易", R_ACTUAL_TRADE_METRICS),
    ("模型預測", R_MODEL_PREDICTION_METRICS),
    ("選股轉換", R_SELECTION_TRANSLATION_METRICS),
)

R_ANALYSIS_MERGED_METRICS = tuple(
    metric
    for _group_label, group_metrics in R_ANALYSIS_GROUPED_SECTIONS
    for metric in group_metrics
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
    "RAnalysisMetricSpec",
    "PORTFOLIO_RESULT_METRICS",
    "CORE_STRATEGY_RESULT_METRICS",
    "TRADE_RESULT_METRICS",
    "EXECUTION_CAPACITY_METRICS",
    "PAIR_MAIN_METRICS",
    "R_ACTUAL_TRADE_METRICS",
    "R_MODEL_PREDICTION_METRICS",
    "R_SELECTION_TRANSLATION_METRICS",
    "R_ANALYSIS_GROUPED_SECTIONS",
    "R_ANALYSIS_MERGED_METRICS",
]
