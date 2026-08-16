"""Canonical Strategy Compare result normalization and human-readable reporting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from core.console_report import (
    compact_console_enabled,
    console_color_enabled,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_SCORE_RANKING,
    comparison_labels as _comparison_labels,
)
from core.report_metrics import PAIR_MAIN_METRICS
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_NEUTRAL,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    signal_for_delta,
    signal_label,
    markdown_signal,
    terminal_signal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _capacity_summary(profile: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(profile.get("portfolio_capacity_rows") or [])
    if frame.empty:
        return {
            "sim_day_count": 0,
            "avg_orderable_candidates": 0.0,
            "zero_orderable_candidate_days": 0,
            "candidate_supply_gap_days": 0,
            "candidate_supply_gap_slot_days": 0,
            "underfilled_end_days": 0,
            "end_position_gap_slot_days": 0,
            "avg_end_positions": 0.0,
            "full_position_days": 0,
        }
    summary = {
        "sim_day_count": int(len(frame)),
        "avg_orderable_candidates": float(frame["Orderable_Candidates"].mean()),
        "zero_orderable_candidate_days": int((frame["Orderable_Candidates"] == 0).sum()),
        "candidate_supply_gap_days": int((frame["Candidate_Supply_Gap"] > 0).sum()),
        "candidate_supply_gap_slot_days": int(frame["Candidate_Supply_Gap"].sum()),
        "underfilled_end_days": int((frame["End_Position_Gap"] > 0).sum()),
        "end_position_gap_slot_days": int(frame["End_Position_Gap"].sum()),
        "avg_end_positions": float(frame["Post_Execution_Positions"].mean()),
        "full_position_days": int((frame["End_Position_Gap"] == 0).sum()),
    }
    if "Resource_Aware_Mode" in frame.columns:
        mode = frame["Resource_Aware_Mode"].fillna("inactive").astype(str)
        changed = frame.get("Resource_Aware_Changed", pd.Series(False, index=frame.index)).astype(bool)
        promoted = pd.to_numeric(
            frame.get("Resource_Aware_Promoted_PASS", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        baseline_reserved = pd.to_numeric(
            frame.get("Resource_Aware_Baseline_Reserved_Milli", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        selected_reserved = pd.to_numeric(
            frame.get("Resource_Aware_Reserved_Milli", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        baseline_pass_reserved = pd.to_numeric(
            frame.get("Resource_Aware_Baseline_PASS_Reserved_Milli", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        selected_pass_reserved = pd.to_numeric(
            frame.get("Resource_Aware_PASS_Reserved_Milli", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        promoted_score_orders = pd.to_numeric(
            frame.get("Resource_Aware_Promoted_Score_Orders", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        baseline_score_sum = pd.to_numeric(
            frame.get("Resource_Aware_Baseline_Score_Sum", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        selected_score_sum = pd.to_numeric(
            frame.get("Resource_Aware_Score_Sum", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_score_feasible = frame.get(
            "Resource_Aware_Direct_Score_Order_Feasible", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        baseline_selected = pd.to_numeric(
            frame.get("Resource_Aware_Baseline_Selected", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        selected_count = pd.to_numeric(
            frame.get("Resource_Aware_Selected", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        preservation_required = frame.get(
            "Resource_Aware_Preservation_Required", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        selected_preserved = frame.get(
            "Resource_Aware_Selected_Count_Preserved", pd.Series(True, index=frame.index)
        ).fillna(True).astype(bool)
        reserved_preserved = frame.get(
            "Resource_Aware_Reserved_Capital_Preserved", pd.Series(True, index=frame.index)
        ).fillna(True).astype(bool)
        max_dl_eligible = frame.get(
            "Resource_Aware_Max_DL_Eligible", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        max_dl_repair_steps = pd.to_numeric(
            frame.get("Resource_Aware_Max_DL_Repair_Steps", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        max_dl_repair_evaluations = pd.to_numeric(
            frame.get("Resource_Aware_Max_DL_Repair_Evaluations", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        max_dl_fallback = frame.get(
            "Resource_Aware_Max_DL_Fallback", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        max_dl_order_limit = pd.to_numeric(
            frame.get("Resource_Aware_Pre_Market_Order_Limit", pd.Series(float("nan"), index=frame.index)),
            errors="coerce",
        )
        max_dl_seed_fallback = frame.get(
            "Resource_Aware_Max_DL_Seed_Fallback", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        max_dl_ascent_steps = pd.to_numeric(
            frame.get("Resource_Aware_Max_DL_Feasible_Ascent_Steps", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        max_dl_ascent_evaluations = pd.to_numeric(
            frame.get("Resource_Aware_Max_DL_Feasible_Ascent_Evaluations", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        max_dl_ascent_local_optimum = frame.get(
            "Resource_Aware_Max_DL_Feasible_Ascent_Local_Optimum", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        stale_guard_enabled = frame.get(
            "Resource_Aware_Stale_Score_Guard_Enabled", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        stale_guard_triggered = frame.get(
            "Resource_Aware_Stale_Score_Guard_Triggered", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        stale_guard_seed_blocked = frame.get(
            "Resource_Aware_Stale_Score_Guard_Seed_Blocked", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        stale_candidate_count = pd.to_numeric(
            frame.get("Resource_Aware_Stale_Score_Candidate_Count", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        stale_guard_blocked_swaps = pd.to_numeric(
            frame.get("Resource_Aware_Stale_Score_Guard_Blocked_Swaps", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        stale_guard_max_age = pd.to_numeric(
            frame.get("Resource_Aware_Stale_Score_Guard_Max_Age_Days", pd.Series(float("nan"), index=frame.index)),
            errors="coerce",
        )
        selector_elapsed_ns = pd.to_numeric(
            frame.get("Resource_Aware_Selector_Elapsed_Ns", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0).clip(lower=0)
        constrained_certified = frame.get(
            "Resource_Aware_Constrained_Optimality_Certified", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        constrained_states = pd.to_numeric(
            frame.get("Resource_Aware_Constrained_Search_States", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        constrained_pruned = pd.to_numeric(
            frame.get("Resource_Aware_Constrained_Pruned_States", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        constrained_feasible = pd.to_numeric(
            frame.get("Resource_Aware_Constrained_Feasible_Baskets", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        selection_mask = mode == "dl-selection"
        summary.update({
            "resource_aware_dl_selection_days": int(selection_mask.sum()),
            "resource_aware_capital_utilization_days": int((mode == "capital-utilization").sum()),
            "resource_aware_changed_days": int(changed.sum()),
            "resource_aware_promoted_pass_orders": int(promoted.sum()),
            "resource_aware_selected_count_delta": int((selected_count - baseline_selected).sum()),
            "resource_aware_reserved_delta_milli": int((selected_reserved - baseline_reserved).sum()),
            "resource_aware_preservation_required_days": int(preservation_required.sum()),
            "resource_aware_preservation_violation_days": int(
                (preservation_required & (~selected_preserved | ~reserved_preserved)).sum()
            ),
            "resource_aware_pass_reserved_gain_milli": int((selected_pass_reserved - baseline_pass_reserved).sum()),
            "resource_aware_pass_reserved_non_improving_days": int(
                (selection_mask & changed & (selected_pass_reserved <= baseline_pass_reserved)).sum()
            ),
            "resource_aware_promoted_score_orders": int(promoted_score_orders.sum()),
            "resource_aware_selected_score_sum_gain": float((selected_score_sum - baseline_score_sum).sum()),
            "resource_aware_direct_score_order_days": int((selection_mask & direct_score_feasible).sum()),
            "resource_aware_max_dl_eligible_days": int(max_dl_eligible.sum()),
            "resource_aware_max_dl_repair_days": int((max_dl_eligible & (max_dl_repair_steps > 0)).sum()),
            "resource_aware_max_dl_repair_steps": int(max_dl_repair_steps.sum()),
            "resource_aware_max_dl_repair_evaluations": int(max_dl_repair_evaluations.sum()),
            "resource_aware_max_dl_fallback_days": int((max_dl_eligible & max_dl_fallback).sum()),
            "resource_aware_max_dl_order_count_violation_days": int(
                (
                    max_dl_eligible
                    & max_dl_order_limit.notna()
                    & (selected_count != max_dl_order_limit)
                ).sum()
            ),
            "resource_aware_max_dl_seed_fallback_days": int((max_dl_eligible & max_dl_seed_fallback).sum()),
            "resource_aware_max_dl_feasible_ascent_days": int((max_dl_eligible & (max_dl_ascent_steps > 0)).sum()),
            "resource_aware_max_dl_feasible_ascent_steps": int(max_dl_ascent_steps.sum()),
            "resource_aware_max_dl_feasible_ascent_evaluations": int(max_dl_ascent_evaluations.sum()),
            "resource_aware_max_dl_feasible_ascent_local_optimum_days": int((max_dl_eligible & max_dl_ascent_local_optimum).sum()),
            "resource_aware_stale_score_guard_enabled_days": int(stale_guard_enabled.sum()),
            "resource_aware_stale_score_guard_triggered_days": int(stale_guard_triggered.sum()),
            "resource_aware_stale_score_guard_seed_blocked_days": int(stale_guard_seed_blocked.sum()),
            "resource_aware_stale_score_candidate_count": int(stale_candidate_count.sum()),
            "resource_aware_stale_score_guard_blocked_swaps": int(stale_guard_blocked_swaps.sum()),
            "resource_aware_stale_score_guard_max_age_days": (
                None if not bool(stale_guard_max_age.notna().any())
                else int(stale_guard_max_age.dropna().iloc[0])
            ),
            "resource_aware_selector_timing_calls": int((selector_elapsed_ns > 0).sum()),
            "resource_aware_selector_timing_total_ms": float(selector_elapsed_ns.sum() / 1_000_000.0),
            "resource_aware_selector_timing_median_ms": float(selector_elapsed_ns[selector_elapsed_ns > 0].median() / 1_000_000.0) if bool((selector_elapsed_ns > 0).any()) else 0.0,
            "resource_aware_selector_timing_p95_ms": float(selector_elapsed_ns[selector_elapsed_ns > 0].quantile(0.95) / 1_000_000.0) if bool((selector_elapsed_ns > 0).any()) else 0.0,
            "resource_aware_selector_timing_max_ms": float(selector_elapsed_ns.max() / 1_000_000.0),
            "resource_aware_constrained_optimality_certified_days": int(constrained_certified.sum()),
            "resource_aware_constrained_search_states": int(constrained_states.sum()),
            "resource_aware_constrained_pruned_states": int(constrained_pruned.sum()),
            "resource_aware_constrained_feasible_baskets": int(constrained_feasible.sum()),
        })
    return summary

def _to_json_native(value: Any) -> Any:
    """Convert pandas/numpy scalars and timestamps to stable JSON-native values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _to_json_native(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_native(item) for item in value]
    return str(value)

def _scenario_summary(payload: dict[str, Any]) -> dict[str, Any]:
    profile = dict(payload["profile"] or {})
    summary = {
        key: payload[key]
        for key in (
            "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
            "annual_return_pct", "log_r_squared", "monthly_win_rate_pct", "trade_count",
            "win_rate_pct", "payoff_ratio", "expected_value_r", "final_equity",
            "avg_exposure_pct", "max_exposure_pct", "missed_buy_count", "missed_sell_count",
            "reserved_buy_fill_rate_pct", "normal_trade_count", "extended_trade_count",
            "annual_trade_count", "benchmark_return_pct", "benchmark_max_drawdown_pct",
            "benchmark_annual_return_pct",
        )
    }
    summary.update({
        "portfolio_total_r": float(profile.get("portfolio_total_r", 0.0)),
        "portfolio_median_r": float(profile.get("portfolio_median_r", 0.0)),
        "portfolio_avg_r": float(profile.get("portfolio_avg_r", 0.0)),
        "min_full_year_return_pct": float(profile.get("min_full_year_return_pct", 0.0)),
        "min_month_return_pct": float(profile.get("min_month_return_pct", 0.0)),
        "min_quarter_return_pct": float(profile.get("min_quarter_return_pct", 0.0)),
        "full_year_count": int(profile.get("full_year_count", 0)),
    })
    summary.update(_capacity_summary(profile))
    return _to_json_native(summary)

def _delta(quality: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in quality.items():
        base = baseline.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(base, (int, float)) and not isinstance(base, bool):
            out[key] = float(value) - float(base)
    return out

def _assert_shared_benchmark(no_filter: dict[str, Any], quality: dict[str, Any]) -> None:
    for key in ("benchmark_return_pct", "benchmark_max_drawdown_pct", "benchmark_annual_return_pct"):
        if not math.isclose(float(no_filter[key]), float(quality[key]), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"兩組 benchmark 不一致: {key}, no_filter={no_filter[key]}, quality={quality[key]}")
    left_dates = list(pd.to_datetime(no_filter["equity_curve"]["Date"]).dt.strftime("%Y-%m-%d"))
    right_dates = list(pd.to_datetime(quality["equity_curve"]["Date"]).dt.strftime("%Y-%m-%d"))
    if left_dates != right_dates:
        raise ValueError("兩組回測交易日期不一致")

def _normalize_yearly_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    """A clipped final year is not complete merely because it reaches the last available replay date."""

    out = pd.DataFrame(frame).copy()
    if out.empty:
        return out
    required = {"is_full_year", "start_date", "end_date"}
    if not required.issubset(out.columns):
        return out
    start_dates = pd.to_datetime(out["start_date"], errors="raise")
    end_dates = pd.to_datetime(out["end_date"], errors="raise")
    calendar_covered = (start_dates.dt.month == 1) & (end_dates.dt.month == 12)
    out["is_full_year"] = out["is_full_year"].astype(bool) & calendar_covered
    return out

def _yearly_frame(profile: dict[str, Any], scenario: str) -> pd.DataFrame:
    frame = _normalize_yearly_completeness(pd.DataFrame(profile.get("yearly_return_rows") or []))
    if frame.empty:
        return pd.DataFrame(columns=["year", f"{scenario}_return_pct", "is_full_year", "start_date", "end_date"])
    return frame.rename(columns={"year_return_pct": f"{scenario}_return_pct"})

def _build_yearly_comparison(no_filter_profile: dict[str, Any], quality_profile: dict[str, Any]) -> pd.DataFrame:
    left = _yearly_frame(no_filter_profile, "no_filter")
    right = _yearly_frame(quality_profile, "quality_filter")
    keys = [key for key in ("year", "is_full_year", "start_date", "end_date") if key in left.columns and key in right.columns]
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged["delta_pct"] = merged["quality_filter_return_pct"] - merged["no_filter_return_pct"]
    return merged.sort_values("year").reset_index(drop=True)

def _refresh_yearly_summary(summary: dict[str, Any], yearly: pd.DataFrame, *, return_column: str) -> None:
    full = yearly[yearly["is_full_year"].astype(bool)] if not yearly.empty else yearly
    summary["full_year_count"] = int(len(full))
    summary["min_full_year_return_pct"] = float(full[return_column].min()) if not full.empty else 0.0

def _load_existing_comparison_payload(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "strategy_comparison.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取既有策略比較 JSON 失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"既有策略比較 JSON schema 無效: {path}")
    return payload

def _format_metric(value: Any, *, digits: int, unit: str = "", signed: bool = False) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}{unit}"

def _markdown_report(metadata, baseline, quality, delta, yearly, strategy_diagnostics=None) -> str:
    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    rows = [
        (metric.label, metric.key, metric.unit, metric.preference, metric.digits, metric.warning_threshold)
        for metric in PAIR_MAIN_METRICS
    ]
    lines = [
        ("# Breakout Quality Score 排序策略經濟效果對照" if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING else "# Breakout Quality 策略經濟效果對照"), "",
        f"- 期間：`{metadata['comparison_period']['start']}` ～ `{metadata['comparison_period']['end']}`",
        f"- 參數檔：`{project_relative_display_path(metadata['params_path'], project_root=PROJECT_ROOT)}`",
        f"- 參數型態：`{metadata['param_source_kind']}`",
        f"- 參數 selector：`{metadata.get('param_selector')}`",
        f"- Runtime members：`{metadata.get('runtime_member_count_min')}`～`{metadata.get('runtime_member_count_max')}`；min_agree=`{metadata.get('runtime_min_agree')}`",
        f"- 比較設計：`{metadata['comparison_design']}`",
        f"- 歷史 active-param 無前視：`{metadata['lookahead_safe_active_param_schedule']}`",
        f"- Dataset：`{metadata['dataset']}`",
        f"- Score source：`{metadata.get('score_source')}`",
        *(
            [f"- Ranking policy：`{metadata.get('score_ranking_policy')}`"]
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else []
        ),
        f"- Optional entry filters：`{metadata.get('optional_entry_filter_policy')}`",
        f"- Benchmark：`{metadata['benchmark_ticker']}`",
        f"- 唯一差異：`{labels['difference_text']}`",
        (
            f"- Ranking model：`{metadata['filter_id']}` / `{metadata['model_architecture']}` / `{metadata['experiment_profile']}`；"
            "threshold 不作 gate"
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else f"- Filter：`{metadata['filter_id']}` / `{metadata['model_architecture']}` / `{metadata['experiment_profile']}` / threshold `{metadata['threshold']}`"
        ),
        *(
            [f"- 排序鍵：`{' → '.join(metadata.get('score_ranking_order') or [])}`", f"- Ranking 範圍：`{metadata.get('ranking_scope')}`"]
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else []
        ),
        "", "## 主要結果", "", f"| 指標 | No filter | {labels['active_title']} | 差異 | 判讀 |", "|---|---:|---:|---:|:---:|",
    ]
    for label, key, unit, preference, digits, warning_threshold in rows:
        metric_signal = signal_for_delta(
            delta.get(key),
            preference=preference,
            warning_threshold=warning_threshold,
        )
        lines.append(
            f"| {label} | {_format_metric(baseline.get(key), digits=digits, unit=unit)} "
            f"| {_format_metric(quality.get(key), digits=digits, unit=unit)} "
            f"| {markdown_signal(_format_metric(delta.get(key), digits=digits, unit=unit, signed=True), metric_signal)} "
            f"| {markdown_signal(signal_label(metric_signal), metric_signal, bold=True)} |"
        )
    lines += ["", "## 年度報酬", ""]
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        lines += [f"| 年度 | No filter | {labels['active_title']} | 差異 | 判讀 | 完整年度 |", "|---:|---:|---:|---:|:---:|:---:|"]
        for row in yearly.to_dict("records"):
            year_signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            lines.append(
                f"| {int(row['year'])} "
                f"| {_format_metric(row.get('no_filter_return_pct'), digits=2, unit='%')} "
                f"| {_format_metric(row.get(active_yearly_column), digits=2, unit='%')} "
                f"| {markdown_signal(_format_metric(row.get('delta_pct'), digits=2, unit='%', signed=True), year_signal)} "
                f"| {markdown_signal(signal_label(year_signal), year_signal, bold=True)} "
                f"| {'是' if row.get('is_full_year') else '否'} |"
            )
    if strategy_diagnostics:
        left = strategy_diagnostics.get("no_filter") or {}
        right = strategy_diagnostics.get("score_ranking") or {}
        lines += [
            "", "## Selection 選股診斷（Future Target僅於回放後join）", "",
            "| 指標 | Baseline | Score Sort | 差異 | 判讀 |",
            "|---|---:|---:|---:|:---:|",
        ]
        for label, key, digits, preference in (
            ("Orderable Score coverage", "orderable_score_coverage_rate", 4, "higher"),
            ("選中候選 Target percentile", "selected_target_percentile_mean", 4, "higher"),
            ("Target top-k retention", "target_top_k_retention_mean", 4, "higher"),
            ("Target opportunity gap (R)", "target_opportunity_gap_r_mean", 4, "lower"),
            ("選中候選 Target mean (R)", "selected_target_mean_r", 4, "higher"),
        ):
            lv, rv = left.get(key), right.get(key)
            dv = None if lv is None or rv is None else float(rv) - float(lv)
            diag_signal = signal_for_delta(dv, preference=preference)
            lines.append(
                f"| {label} | {_format_metric(lv, digits=digits)} | "
                f"{_format_metric(rv, digits=digits)} | {markdown_signal(_format_metric(dv, digits=digits, signed=True), diag_signal)} "
                f"| {markdown_signal(signal_label(diag_signal), diag_signal, bold=True)} |"
            )
        lines += [
            "",
            "> Future Target未進入候選排序、資金配置或成交決策；上述診斷只在兩組策略回放完成後離線計算。",
        ]
    if not metadata["lookahead_safe_active_param_schedule"]:
        lines += ["", "> 警告：本次使用單一／static 參數，只能視為敏感度診斷，不是無前視 OOS 部署證據。"]
    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        limitation = (
            "> 本報表使用Selection point-in-time Scores與當期歷史active params比較排序機制；"
            "可用於Selection內決定是否進入參數適應，但正式效果仍須由凍結後OOS驗證。"
            if metadata.get("score_source") == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "> 本報表是既有forward period的score-ranking機制比較；不得依結果回頭調整模型或排序規則。"
        )
    else:
        limitation = "> 本報表只驗證目前固定active操作點能否改善策略經濟效果；不得依結果回頭調整threshold、模型或訓練條件。"
    lines += ["", limitation, ""]
    return "\n".join(lines)

def _numeric_delta(delta: dict[str, Any], key: str) -> float | None:
    value = delta.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _compact_metric_rows(
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    specs: tuple[tuple[str, str, str, str], ...],
    *,
    use_color: bool,
) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    integer_keys = {
        "trade_count", "candidate_supply_gap_days", "underfilled_end_days",
        "end_position_gap_slot_days",
    }
    for label, key, unit, preference in specs:
        digits = 0 if key in integer_keys else 4 if key == "log_r_squared" else 2
        signal = signal_for_delta(
            delta.get(key),
            preference=preference,
            warning_threshold=5.0 if key == "avg_exposure_pct" else 0.0,
        )
        rows.append((
            label,
            _format_metric(baseline.get(key), digits=digits, unit=unit),
            _format_metric(quality.get(key), digits=digits, unit=unit),
            terminal_signal(
                _format_metric(delta.get(key), digits=digits, unit=unit, signed=True),
                signal,
                enabled=use_color,
            ),
            terminal_signal(signal_label(signal), signal, enabled=use_color),
        ))
    return rows

def _render_compact_strategy_console_report(
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    yearly: pd.DataFrame,
    strategy_diagnostics: dict[str, Any] | None,
    *,
    use_color: bool,
) -> str:
    """Render the interactive strategy report by decision layer rather than source table."""

    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    active_label = (
        "Score Sort"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else labels["active_title"]
    )
    period = metadata.get("comparison_period") or {}
    title = (
        "Breakout Quality Score 排序策略績效摘要"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else "Breakout Quality 策略績效摘要"
    )

    total_delta = _numeric_delta(delta, "total_return_pct")
    romd_delta = _numeric_delta(delta, "return_over_max_drawdown")
    mdd_delta = _numeric_delta(delta, "max_drawdown_pct")
    if total_delta is not None and romd_delta is not None and total_delta < 0 and romd_delta < 0:
        portfolio_signal = SIGNAL_NEGATIVE
        portfolio_text = "投組層失敗：總報酬與報酬／回撤同步下降。"
    elif total_delta is not None and romd_delta is not None and total_delta > 0 and romd_delta > 0:
        portfolio_signal = SIGNAL_POSITIVE
        portfolio_text = "投組層改善：總報酬與報酬／回撤同步提高。"
    else:
        portfolio_signal = SIGNAL_WARNING
        portfolio_text = "投組層結果混合：報酬與風險指標未同向改善。"

    ev_delta = _numeric_delta(delta, "expected_value_r")
    payoff_delta = _numeric_delta(delta, "payoff_ratio")
    trade_signal = (
        SIGNAL_POSITIVE
        if ev_delta is not None and payoff_delta is not None and ev_delta > 0 and payoff_delta > 0
        else SIGNAL_NEGATIVE
        if ev_delta is not None and payoff_delta is not None and ev_delta < 0 and payoff_delta < 0
        else SIGNAL_WARNING
    )
    trade_text = (
        "交易層：EV與Payoff同步改善，但仍須結合勝率與投入規模判讀。"
        if trade_signal == SIGNAL_POSITIVE
        else "交易層：單筆交易品質未全面改善。"
    )

    exposure_delta = _numeric_delta(delta, "avg_exposure_pct")
    underfilled_delta = _numeric_delta(delta, "underfilled_end_days")
    capital_signal = SIGNAL_WARNING
    capital_text = (
        "資金層：平均曝險下降，但未滿倉日也下降；代表持倉格較滿、每格投入較小。"
        if exposure_delta is not None and exposure_delta < 0
        and underfilled_delta is not None and underfilled_delta < 0
        else "資金層：需分開檢查曝險、候選供給與持倉格使用。"
    )

    diagnostic_specs = (
        ("Orderable Score coverage", "orderable_score_coverage_rate", "higher"),
        ("選中候選 Target percentile", "selected_target_percentile_mean", "higher"),
        ("Target top-k retention", "target_top_k_retention_mean", "higher"),
        ("Target opportunity gap", "target_opportunity_gap_r_mean", "lower"),
        ("選中候選 Target mean", "selected_target_mean_r", "higher"),
    )
    model_signal = SIGNAL_NEUTRAL
    model_text = "模型層：本次沒有可用的事後Target選股診斷。"
    diagnostic_rows: list[tuple[str, str, str, str, str]] = []
    if strategy_diagnostics:
        left = strategy_diagnostics.get("no_filter") or {}
        right = strategy_diagnostics.get("score_ranking") or {}
        positive_count = 0
        available_count = 0
        for label, key, preference in diagnostic_specs:
            left_value, right_value = left.get(key), right.get(key)
            try:
                left_number = float(left_value)
                right_number = float(right_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(left_number) or not math.isfinite(right_number):
                continue
            delta_value = right_number - left_number
            signal = signal_for_delta(delta_value, preference=preference)
            available_count += 1
            positive_count += int(signal == SIGNAL_POSITIVE)
            diagnostic_rows.append((
                label,
                _format_metric(left_number, digits=4),
                _format_metric(right_number, digits=4),
                terminal_signal(
                    _format_metric(delta_value, digits=4, signed=True),
                    signal,
                    enabled=use_color,
                ),
                terminal_signal(signal_label(signal), signal, enabled=use_color),
            ))
        if available_count:
            model_signal = (
                SIGNAL_POSITIVE if positive_count >= max(1, available_count - 1)
                else SIGNAL_WARNING
            )
            model_text = f"模型層：{positive_count}/{available_count}項Selection選股診斷改善。"

    lines = [
        render_title(title),
        render_key_values((
            ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
            ("比較", f"Baseline vs {active_label}"),
            ("Score source", metadata.get("score_source", "-")),
            ("歷史參數無前視", metadata.get("lookahead_safe_active_param_schedule", "-")),
        )),
        render_section("綜合判定", number=1),
        terminal_signal(portfolio_text, portfolio_signal, enabled=use_color),
        terminal_signal(trade_text, trade_signal, enabled=use_color),
        terminal_signal(capital_text, capital_signal, enabled=use_color),
        terminal_signal(model_text, model_signal, enabled=use_color),
    ]
    if total_delta is not None or mdd_delta is not None:
        lines.append(
            "關鍵差異：總報酬 "
            f"{_format_metric(total_delta, digits=2, unit='pp', signed=True)}；"
            "最大回撤 "
            f"{_format_metric(mdd_delta, digits=2, unit='pp', signed=True)}；"
            "平均曝險 "
            f"{_format_metric(exposure_delta, digits=2, unit='pp', signed=True)}。"
        )

    portfolio_specs = (
        ("淨總報酬", "total_return_pct", "%", "higher"),
        ("最大回撤", "max_drawdown_pct", "%", "lower"),
        ("報酬／最大回撤", "return_over_max_drawdown", "", "higher"),
        ("年化報酬", "annual_return_pct", "%", "higher"),
        ("Log R²", "log_r_squared", "", "higher"),
        ("月勝率", "monthly_win_rate_pct", "%", "higher"),
        ("最差完整年度", "min_full_year_return_pct", "%", "higher"),
    )
    lines.extend((
        render_section("投組報酬與風險", number=2),
        "讀法：看最終賺多少、承受多少回撤，以及權益成長是否穩定。",
        render_table(
            ("指標", "Baseline", active_label, "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, portfolio_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    trade_specs = (
        ("交易數", "trade_count", "", "neutral"),
        ("勝率", "win_rate_pct", "%", "higher"),
        ("Payoff", "payoff_ratio", "", "higher"),
        ("EV", "expected_value_r", " R", "higher"),
    )
    lines.extend((
        render_section("單筆交易品質", number=3),
        "讀法：看每筆交易的命中率與盈虧結構；不代表整體資金使用效率。",
        render_table(
            ("指標", "Baseline", active_label, "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, trade_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    capital_specs = (
        ("平均曝險", "avg_exposure_pct", "%", "attention"),
        ("平均每日可掛單候選", "avg_orderable_candidates", "", "neutral"),
        ("候選供給不足日", "candidate_supply_gap_days", " 日", "lower"),
        ("每日結束未滿倉日", "underfilled_end_days", " 日", "lower"),
        ("每日結束持股缺口", "end_position_gap_slot_days", " 格日", "lower"),
    )
    lines.extend((
        render_section("資金使用與持倉容量", number=4),
        "讀法：持有幾檔與投入多少資金是不同概念；持倉格較滿不等於曝險較高。",
        render_table(
            ("指標", "Baseline", active_label, "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, capital_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    lines.append(render_section("年度報酬", number=5))
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        yearly_rows = []
        for row in yearly.to_dict("records"):
            signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            yearly_rows.append((
                int(row["year"]),
                _format_metric(row.get("no_filter_return_pct"), digits=2, unit="%"),
                _format_metric(row.get(active_yearly_column), digits=2, unit="%"),
                terminal_signal(
                    _format_metric(row.get("delta_pct"), digits=2, unit="pp", signed=True),
                    signal,
                    enabled=use_color,
                ),
                terminal_signal(signal_label(signal), signal, enabled=use_color),
            ))
        lines.append(render_table(
            ("年度", "Baseline", active_label, "差異", "判讀"),
            yearly_rows,
            alignments=("right", "right", "right", "right", "left"),
        ))

    next_number = 6
    if diagnostic_rows:
        lines.extend((
            render_section("模型選股方向", number=next_number),
            "讀法：Future Target只在回放完成後加入；這裡衡量選股方向，不是實際策略報酬。",
            render_table(
                ("指標", "Baseline", "Score Sort", "差異", "判讀"),
                diagnostic_rows,
                alignments=("left", "right", "right", "right", "left"),
            ),
        ))
        next_number += 1

    lines.append(render_section("判讀限制", number=next_number))
    if not metadata.get("lookahead_safe_active_param_schedule"):
        lines.append("⚠️ 本次使用單一／static參數，只能視為敏感度診斷。")
    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        lines.append(
            "本結果是Selection內的PIT比較；模型排序、投組績效與資金配置必須分層判讀，"
            "正式採用仍須由凍結後OOS驗證。"
        )
    else:
        lines.append("本結果只驗證固定active操作點，不得依結果回頭調整模型或threshold。")
    return "\n".join(lines)

def _render_strategy_console_report(
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    yearly: pd.DataFrame,
    strategy_diagnostics: dict[str, Any] | None = None,
    *,
    color: bool | None = None,
) -> str:
    """Render the complete readable strategy comparison directly for console."""

    use_color = console_color_enabled() if color is None else bool(color)
    if compact_console_enabled():
        return _render_compact_strategy_console_report(
            metadata,
            baseline,
            quality,
            delta,
            yearly,
            strategy_diagnostics,
            use_color=use_color,
        )
    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    title = (
        "Breakout Quality Score 排序策略經濟效果對照"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else "Breakout Quality 策略經濟效果對照"
    )
    period = metadata.get("comparison_period") or {}
    params_path = project_relative_display_path(
        metadata.get("params_path", "-"), project_root=PROJECT_ROOT
    )
    lines = [
        render_title(title),
        render_key_values(
            (
                ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
                ("參數檔", params_path),
                ("參數型態", metadata.get("param_source_kind", "-")),
                ("參數 selector", metadata.get("param_selector", "-")),
                (
                    "Runtime members",
                    f"{metadata.get('runtime_member_count_min')}～{metadata.get('runtime_member_count_max')}；"
                    f"min_agree={metadata.get('runtime_min_agree')}",
                ),
                ("比較設計", metadata.get("comparison_design", "-")),
                ("歷史 active-param 無前視", metadata.get("lookahead_safe_active_param_schedule", "-")),
                ("Dataset", metadata.get("dataset", "-")),
                ("Score source", metadata.get("score_source", "-")),
                ("Ranking policy", metadata.get("score_ranking_policy", "-")),
                ("Optional entry filters", metadata.get("optional_entry_filter_policy", "-")),
                ("Benchmark", metadata.get("benchmark_ticker", "-")),
                ("唯一差異", labels["difference_text"]),
                *(
                    (
                        ("排序鍵", " → ".join(metadata.get("score_ranking_order") or [])),
                        ("Ranking 範圍", metadata.get("ranking_scope", "-")),
                    )
                    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
                    else ()
                ),
            )
        ),
    ]

    metric_rows = []
    for metric in PAIR_MAIN_METRICS:
        label = metric.label
        key = metric.key
        unit = metric.unit
        digits = metric.digits
        signal = signal_for_delta(
            delta.get(key), preference=metric.preference,
            warning_threshold=metric.warning_threshold,
        )
        delta_text = _format_metric(delta.get(key), digits=digits, unit=unit, signed=True)
        judgment = signal_label(signal)
        metric_rows.append((
            label,
            _format_metric(baseline.get(key), digits=digits, unit=unit),
            _format_metric(quality.get(key), digits=digits, unit=unit),
            terminal_signal(delta_text, signal, enabled=use_color),
            terminal_signal(judgment, signal, enabled=use_color),
        ))
    lines.extend((
        render_section("主要結果", number=1),
        render_table(
            ("指標", "No filter", labels["active_title"], "差異", "判讀"),
            metric_rows,
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    lines.append(render_section("年度報酬", number=2))
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        yearly_rows = []
        for row in yearly.to_dict("records"):
            signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            yearly_rows.append((
                int(row["year"]),
                _format_metric(row.get("no_filter_return_pct"), digits=2, unit="%"),
                _format_metric(row.get(active_yearly_column), digits=2, unit="%"),
                terminal_signal(
                    _format_metric(row.get("delta_pct"), digits=2, unit="%", signed=True),
                    signal, enabled=use_color,
                ),
                terminal_signal(signal_label(signal), signal, enabled=use_color),
                "是" if row.get("is_full_year") else "否",
            ))
        lines.append(render_table(
            ("年度", "No filter", labels["active_title"], "差異", "判讀", "完整年度"),
            yearly_rows,
            alignments=("right", "right", "right", "right", "left", "center"),
        ))

    if strategy_diagnostics:
        left = strategy_diagnostics.get("no_filter") or {}
        right = strategy_diagnostics.get("score_ranking") or {}
        diagnostic_rows = []
        for label, key, preference in (
            ("Orderable Score coverage", "orderable_score_coverage_rate", "higher"),
            ("選中候選 Target percentile", "selected_target_percentile_mean", "higher"),
            ("Target top-k retention", "target_top_k_retention_mean", "higher"),
            ("Target opportunity gap (R)", "target_opportunity_gap_r_mean", "lower"),
            ("選中候選 Target mean (R)", "selected_target_mean_r", "higher"),
        ):
            left_value, right_value = left.get(key), right.get(key)
            delta_value = (
                None if left_value is None or right_value is None
                else float(right_value) - float(left_value)
            )
            signal = signal_for_delta(delta_value, preference=preference)
            diagnostic_rows.append((
                label,
                _format_metric(left_value, digits=4),
                _format_metric(right_value, digits=4),
                terminal_signal(
                    _format_metric(delta_value, digits=4, signed=True),
                    signal, enabled=use_color,
                ),
                terminal_signal(signal_label(signal), signal, enabled=use_color),
            ))
        lines.extend((
            render_section("Selection 選股診斷（Future Target 僅於回放後 join）", number=3),
            render_table(
                ("指標", "Baseline", "Score Sort", "差異", "判讀"),
                diagnostic_rows,
                alignments=("left", "right", "right", "right", "left"),
            ),
            "Future Target 未進入候選排序、資金配置或成交決策。",
        ))

    lines.append(render_section("判讀限制", number=4 if strategy_diagnostics else 3))
    if not metadata.get("lookahead_safe_active_param_schedule"):
        lines.append("⚠️ 本次使用單一／static 參數，只能視為敏感度診斷，不是無前視部署證據。")
    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        lines.append(
            "本報表使用 Selection point-in-time Scores 與當期歷史 active params；"
            "可用於 Selection 內決定是否進入參數適應，正式效果仍須由凍結後 OOS 驗證。"
            if metadata.get("score_source") == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "本報表是既有 forward period 的排序機制比較，不得依結果回頭調整模型或排序規則。"
        )
    else:
        lines.append("本報表只驗證固定 active 操作點，不得依結果回頭調整 threshold、模型或訓練條件。")
    return "\n".join(lines)

def _strategy_pair_report_components(
    payload: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any] | None,
]:
    """Resolve canonical readable-report inputs from one completed pair payload."""

    metadata = dict(payload.get("metadata") or {})
    comparison_mode = str(metadata.get("comparison_mode") or "")
    labels = _comparison_labels(comparison_mode)
    active_name = labels["active_name"]
    baseline = dict(payload.get("no_filter") or {})
    quality = dict(payload.get(active_name) or {})
    delta = dict(payload.get(f"{active_name}_minus_no_filter") or {})
    if not metadata or not baseline or not quality or not delta:
        raise ValueError("strategy pair payload缺少可讀報表必要欄位")
    yearly = pd.DataFrame(list(payload.get("yearly") or []))
    diagnostics_raw = payload.get("selection_diagnostics")
    diagnostics = dict(diagnostics_raw) if isinstance(diagnostics_raw, dict) else None
    return metadata, baseline, quality, delta, yearly, diagnostics

def render_strategy_pair_simple_report(
    payload: dict[str, Any],
    *,
    color: bool | None = None,
) -> str:
    """Render the canonical human-readable strategy pair report from JSON payload."""

    metadata, baseline, quality, delta, yearly, diagnostics = (
        _strategy_pair_report_components(payload)
    )
    return _render_strategy_console_report(
        metadata,
        baseline,
        quality,
        delta,
        yearly,
        diagnostics,
        color=color,
    )

def render_strategy_pair_markdown(payload: dict[str, Any]) -> str:
    """Render the canonical persistent Markdown strategy pair report from JSON payload."""

    metadata, baseline, quality, delta, yearly, diagnostics = (
        _strategy_pair_report_components(payload)
    )
    return _markdown_report(
        metadata,
        baseline,
        quality,
        delta,
        yearly,
        diagnostics,
    )

def materialize_strategy_pair_readable_report(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
) -> Path:
    """Persist the required human-readable report for a completed strategy pair."""

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "strategy_comparison.md"
    report_path.write_text(render_strategy_pair_markdown(payload), encoding="utf-8")
    return report_path

def _remove_legacy_html_outputs(output_dir: Path) -> None:
    for filename in ("strategy_comparison.html",):
        path = output_dir / filename
        if path.is_file():
            path.unlink()

# Stable public aliases for read-only consumers.
scenario_summary = _scenario_summary
