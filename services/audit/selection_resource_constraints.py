"""Read-only attribution of C58-derived K/R0 resource constraints on DL selection."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.path_utils import project_relative_display_path
from core.report_metrics import (
    MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS,
    MFE_SAFETY_STAGE_ATTRIBUTION_METRICS,
    RESOURCE_CONSTRAINT_ATTRIBUTION_METRICS,
)
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.strategy_compare_diagnostics import (
    strategy_replay_score_event_frames,
)
from services.audit.mfe_safety_truth import (
    AuditBlockedError,
    QUADRANT_KEYS,
    attach_quadrants,
    build_truth_geometry,
    distribution_for_keys,
    filter_period,
    finite_float,
    normalize_date,
    normalize_ticker,
    resolve_truth_provider_contract,
    score_event_keys,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_pipeline_sidecars,
    load_strategy_arm_replay_sidecars,
    load_strategy_compare_source,
)

SUPPORTED_AUDIT_TYPE = "selection_resource_constraint_attribution"

_STAGE_ORDER = ("orderable", "raw_top_k", "planned", "filled")
_STAGE_DISPLAY = {
    "orderable": "Orderable pool",
    "raw_top_k": "Raw Top-K",
    "planned": "Planned orders",
    "filled": "Filled buys",
}


def _as_bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    normalized = raw.fillna("").astype(str).str.strip().str.lower()
    truthy = normalized.isin({"1", "true", "t", "yes", "y"})
    falsey = normalized.isin({"0", "false", "f", "no", "n", ""})
    invalid = ~(truthy | falsey)
    if bool(invalid.any()):
        sample = raw.loc[invalid].iloc[0]
        raise AuditBlockedError(f"{column}含無法解析的bool值: {sample!r}")
    return truthy



def _normalize_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "Date",
        "Pre_Market_Free_Slots",
        "Orderable_Candidates",
        "Resource_Aware_Pre_Market_Order_Limit",
        "Resource_Aware_Max_DL_Eligible",
        "Resource_Aware_Direct_Score_Order_Feasible",
        "Resource_Aware_Baseline_Reserved_Milli",
        "Resource_Aware_Reserved_Milli",
        "Resource_Aware_Selected",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"score-ranking daily capacity缺少欄位: {missing}")
    table["trade_date"] = pd.to_datetime(table["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if bool(table["trade_date"].isna().any()):
        raise AuditBlockedError("score-ranking daily capacity含無效Date")
    table["eligible"] = _as_bool_series(table, "Resource_Aware_Max_DL_Eligible")
    table["direct_feasible"] = _as_bool_series(
        table, "Resource_Aware_Direct_Score_Order_Feasible"
    )

    # Resource-aware diagnostics are intentionally blank on many non-eligible
    # days.  Validate the numeric contract on Max-DL eligible days only; a blank
    # non-eligible K must not make an otherwise complete persisted sidecar BLOCKED.
    numeric_columns = {
        "free_slots": "Pre_Market_Free_Slots",
        "orderable_candidates": "Orderable_Candidates",
        "k": "Resource_Aware_Pre_Market_Order_Limit",
        "r0_milli": "Resource_Aware_Baseline_Reserved_Milli",
        "reserved_milli": "Resource_Aware_Reserved_Milli",
        "selected_count": "Resource_Aware_Selected",
    }
    eligible_mask = table["eligible"]
    for output_column, source_column in numeric_columns.items():
        parsed = pd.to_numeric(table[source_column], errors="coerce")
        invalid_eligible = eligible_mask & parsed.isna()
        if bool(invalid_eligible.any()):
            bad = table.loc[invalid_eligible, source_column].iloc[0]
            raise AuditBlockedError(
                f"{source_column}在Max-DL eligible day含無法解析數值: {bad!r}"
            )
        table[output_column] = parsed.fillna(0)
    for column in ("free_slots", "orderable_candidates", "k", "selected_count"):
        table[column] = table[column].astype(int)
    for column in ("r0_milli", "reserved_milli"):
        table[column] = table[column].astype(np.int64)
    return table


def summarize_resource_contract(capacity: pd.DataFrame) -> dict[str, Any]:
    table = _normalize_capacity(capacity)
    eligible = table.loc[table["eligible"]].copy()
    if eligible.empty:
        raise AuditBlockedError("score-ranking daily capacity沒有Max-DL eligible days")
    headroom = (
        (eligible["k"] < eligible["free_slots"])
        & (eligible["orderable_candidates"] > eligible["k"])
    )
    direct_infeasible = ~eligible["direct_feasible"]
    positive_r0 = eligible["r0_milli"] > 0
    exact_r0_binding = positive_r0 & (
        eligible["reserved_milli"] == eligible["r0_milli"]
    )
    r0_slack_pct = pd.Series(dtype=float)
    if bool(positive_r0.any()):
        r0_slack_pct = (
            (eligible.loc[positive_r0, "reserved_milli"] - eligible.loc[positive_r0, "r0_milli"])
            / eligible.loc[positive_r0, "r0_milli"]
            * 100.0
        )
    return {
        "eligible_days": int(len(eligible)),
        "direct_feasible_days": int(eligible["direct_feasible"].sum()),
        "direct_infeasible_days": int(direct_infeasible.sum()),
        "direct_infeasible_pct": float(direct_infeasible.mean() * 100.0),
        "k_headroom_days": int(headroom.sum()),
        "k_headroom_pct": float(headroom.mean() * 100.0),
        "k_headroom_slot_days": int(
            (eligible.loc[headroom, "free_slots"] - eligible.loc[headroom, "k"]).sum()
        ),
        "median_k": float(eligible["k"].median()),
        "median_free_slots": float(eligible["free_slots"].median()),
        "selected_count_equals_k_pct": float(
            (eligible["selected_count"] == eligible["k"]).mean() * 100.0
        ),
        "r0_positive_days": int(positive_r0.sum()),
        "r0_exact_binding_days": int(exact_r0_binding.sum()),
        "r0_exact_binding_pct": (
            float(exact_r0_binding.loc[positive_r0].mean() * 100.0)
            if bool(positive_r0.any())
            else None
        ),
        "median_final_r0_slack_pct": (
            float(r0_slack_pct.median()) if not r0_slack_pct.empty else None
        ),
        "eligible_trade_dates": sorted(eligible["trade_date"].astype(str).unique().tolist()),
        "direct_feasible_trade_dates": sorted(
            eligible.loc[eligible["direct_feasible"], "trade_date"].astype(str).unique().tolist()
        ),
        "direct_infeasible_trade_dates": sorted(
            eligible.loc[direct_infeasible, "trade_date"].astype(str).unique().tolist()
        ),
        "k_headroom_trade_dates": sorted(
            eligible.loc[headroom, "trade_date"].astype(str).unique().tolist()
        ),
    }


def _period_from_source(source: Any) -> tuple[str, str]:
    payload = source.result.get("comparison_period")
    if not isinstance(payload, Mapping):
        raise AuditBlockedError(f"{source.profile_id} Strategy Compare缺少canonical comparison_period")
    start = normalize_date(payload.get("start"))
    end = normalize_date(payload.get("end"))
    if not start or not end:
        raise AuditBlockedError(f"{source.profile_id} comparison_period格式無效")
    if source.settings.start_date and start != str(source.settings.start_date):
        raise AuditBlockedError(
            f"{source.profile_id} start_date與目前config不一致: artifact={start}, config={source.settings.start_date}"
        )
    if source.settings.end_date and end != str(source.settings.end_date):
        raise AuditBlockedError(
            f"{source.profile_id} end_date與目前config不一致: artifact={end}, config={source.settings.end_date}"
        )
    return start, end


def _occurrence_event_lookup(orderable: pd.DataFrame) -> pd.DataFrame:
    orderable_events, _ = strategy_replay_score_event_frames(orderable, None)
    keys = ["ticker", "trade_date", "signal_date"]
    lookup = orderable_events[[*keys, "score_event_date"]].copy()
    lookup["ticker"] = lookup["ticker"].map(normalize_ticker)
    for column in ("trade_date", "signal_date", "score_event_date"):
        lookup[column] = lookup[column].map(normalize_date)
    counts = lookup.groupby(keys, dropna=False)["score_event_date"].nunique(dropna=False)
    if bool((counts > 1).any()):
        raise AuditBlockedError("orderable occurrence無法唯一解析score_event_date")
    return lookup.drop_duplicates(keys, keep="first")


def _execution_with_score_event(
    execution: pd.DataFrame,
    *,
    orderable: pd.DataFrame,
) -> pd.DataFrame:
    work = pd.DataFrame(execution).copy()
    required = {"ticker", "trade_date", "signal_date", "chosen_qty", "entry_filled"}
    missing = sorted(required - set(work.columns))
    if missing:
        raise AuditBlockedError(f"score-ranking execution缺少欄位: {missing}")
    work["ticker"] = work["ticker"].map(normalize_ticker)
    for column in ("trade_date", "signal_date"):
        work[column] = work[column].map(normalize_date)
    lookup = _occurrence_event_lookup(orderable)
    joined = work.merge(
        lookup,
        on=["ticker", "trade_date", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    unresolved = joined["score_event_date"].fillna("").astype(str).str.strip().eq("")
    if bool(unresolved.any()):
        bad = joined.loc[unresolved].iloc[0]
        raise AuditBlockedError(
            "planned execution無法對回orderable score event: "
            f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, signal_date={bad.get('signal_date')}"
        )
    chosen = pd.to_numeric(joined["chosen_qty"], errors="coerce")
    if bool(chosen.isna().any()):
        raise AuditBlockedError("score-ranking execution chosen_qty含無效值")
    joined = joined.loc[chosen > 0].copy()
    joined["entry_filled_bool"] = _as_bool_series(joined, "entry_filled")
    return joined


def _raw_top_k_with_score_event(trace: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(trace).copy()
    required = {
        "stage", "stage_rank", "ticker", "trade_date", "signal_date",
        "breakout_quality_score_date", "pre_market_order_limit",
    }
    missing = sorted(required - set(work.columns))
    if missing:
        raise AuditBlockedError(f"selector trace缺少欄位: {missing}")
    work = work.loc[work["stage"].fillna("").astype(str).eq("raw_top_n")].copy()
    if work.empty:
        raise AuditBlockedError("selector trace沒有raw_top_n stage rows")
    work["ticker"] = work["ticker"].map(normalize_ticker)
    work["trade_date"] = work["trade_date"].map(normalize_date)
    work["signal_date"] = work["signal_date"].map(normalize_date)
    score_date = work["breakout_quality_score_date"].map(normalize_date)
    work["score_event_date"] = score_date.where(score_date.ne(""), work["signal_date"])
    if bool(work["score_event_date"].eq("").any()):
        raise AuditBlockedError("raw_top_n存在無法解析的score event date")
    return work


def _event_keys_for_days(frame: pd.DataFrame, trade_dates: Sequence[str]) -> pd.DataFrame:
    dates = set(str(value) for value in trade_dates)
    work = pd.DataFrame(frame)
    if "trade_date" not in work.columns or "score_event_date" not in work.columns:
        raise AuditBlockedError("stage frame缺少trade_date/score_event_date")
    subset = work.loc[work["trade_date"].astype(str).isin(dates)].copy()
    keys = pd.DataFrame(
        {
            "ticker": subset["ticker"].map(normalize_ticker),
            "date": subset["score_event_date"].map(normalize_date),
        }
    )
    keys = keys.loc[keys["ticker"].ne("") & keys["date"].ne("")]
    return keys.drop_duplicates(["ticker", "date"])


def _membership_overlap_pct(raw: pd.DataFrame, planned: pd.DataFrame, trade_dates: Sequence[str]) -> float | None:
    dates = set(str(value) for value in trade_dates)
    if not dates:
        return None
    raw_work = pd.DataFrame(raw).loc[pd.DataFrame(raw)["trade_date"].astype(str).isin(dates)]
    planned_work = pd.DataFrame(planned).loc[pd.DataFrame(planned)["trade_date"].astype(str).isin(dates)]
    values: list[float] = []
    for day in sorted(dates):
        raw_set = set(raw_work.loc[raw_work["trade_date"].astype(str).eq(day), "ticker"].map(normalize_ticker))
        planned_set = set(planned_work.loc[planned_work["trade_date"].astype(str).eq(day), "ticker"].map(normalize_ticker))
        denominator = len(raw_set)
        if denominator <= 0:
            continue
        values.append(len(raw_set & planned_set) / denominator * 100.0)
    return None if not values else float(np.mean(values))


def build_transition_summary(
    truth: pd.DataFrame,
    *,
    raw: pd.DataFrame,
    planned: pd.DataFrame,
    trade_dates: Sequence[str],
) -> dict[str, Any]:
    raw_dist = distribution_for_keys(
        truth,
        _event_keys_for_days(raw, trade_dates),
        allow_empty=True,
    )
    planned_dist = distribution_for_keys(
        truth,
        _event_keys_for_days(planned, trade_dates),
        allow_empty=True,
    )
    raw_high = finite_float(raw_dist.get("high_mfe_total_pct"))
    planned_high = finite_float(planned_dist.get("high_mfe_total_pct"))
    raw_lmhs = finite_float(raw_dist.get("low_mfe_high_safety_pct"))
    planned_lmhs = finite_float(planned_dist.get("low_mfe_high_safety_pct"))
    return {
        "trade_days": int(len(set(str(value) for value in trade_dates))),
        "raw": raw_dist,
        "planned": planned_dist,
        "delta_high_mfe_pp": (
            None if raw_high is None or planned_high is None else planned_high - raw_high
        ),
        "delta_low_mfe_high_safety_pp": (
            None if raw_lmhs is None or planned_lmhs is None else planned_lmhs - raw_lmhs
        ),
        "mean_raw_membership_retained_pct": _membership_overlap_pct(raw, planned, trade_dates),
    }


def _stage_geometry(
    truth: pd.DataFrame,
    *,
    orderable_events: pd.DataFrame,
    raw: pd.DataFrame,
    planned: pd.DataFrame,
    filled: pd.DataFrame,
    eligible_dates: Sequence[str],
) -> dict[str, Any]:
    frames = {
        "orderable": orderable_events,
        "raw_top_k": raw,
        "planned": planned,
        "filled": filled,
    }
    return {
        stage: distribution_for_keys(
            truth,
            _event_keys_for_days(frame, eligible_dates),
            allow_empty=True,
        )
        for stage, frame in frames.items()
    }


def _validate_raw_stage_capacity(
    raw: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    normalized = _normalize_capacity(capacity)
    eligible = normalized.loc[normalized["eligible"]].copy()
    raw_work = pd.DataFrame(raw).copy()
    raw_work["trade_date"] = raw_work["trade_date"].map(normalize_date)
    raw_work["ticker"] = raw_work["ticker"].map(normalize_ticker)
    raw_work = raw_work.loc[raw_work["trade_date"].ne("") & raw_work["ticker"].ne("")]
    raw_counts = raw_work.groupby("trade_date")["ticker"].nunique()
    for _, row in eligible.iterrows():
        trade_date = str(row["trade_date"])
        expected_k = int(row["k"])
        actual = int(raw_counts.get(trade_date, 0))
        if actual != expected_k:
            raise AuditBlockedError(
                "selector trace raw_top_n與daily-capacity K不一致: "
                f"date={trade_date}, raw={actual}, K={expected_k}"
            )
    extra_dates = sorted(set(raw_counts.index.astype(str)) - set(eligible["trade_date"].astype(str)))
    if extra_dates:
        # Trace may contain non-Max-DL days only if producer semantics changed;
        # fail closed instead of silently mixing unmatched stage populations.
        raise AuditBlockedError(
            "selector trace raw_top_n含非Max-DL eligible日期: " + ", ".join(extra_dates[:5])
        )


def _filled_selected_consistency(
    *,
    orderable: pd.DataFrame,
    selected: pd.DataFrame,
    filled: pd.DataFrame,
) -> dict[str, Any]:
    _, selected_events = strategy_replay_score_event_frames(orderable, selected)
    if selected_events is None:
        return {"selected_rows": 0, "filled_rows": int(len(filled)), "key_match_pct": None}
    occurrence_cols = ["ticker", "trade_date", "signal_date"]
    def occurrence_key(row: pd.Series) -> tuple[str, str, str]:
        return (
            normalize_ticker(row.get("ticker")),
            normalize_date(row.get("trade_date")),
            normalize_date(row.get("signal_date")),
        )

    selected_keys = {
        occurrence_key(row)
        for _, row in selected_events[occurrence_cols].fillna("").iterrows()
    }
    filled_keys = {
        occurrence_key(row)
        for _, row in filled[occurrence_cols].fillna("").iterrows()
    }
    union = selected_keys | filled_keys
    return {
        "selected_rows": int(len(selected_keys)),
        "filled_rows": int(len(filled_keys)),
        "key_match_pct": (100.0 if not union else len(selected_keys & filled_keys) / len(union) * 100.0),
    }


def _fingerprint_payload(definition: AuditDefinition, source_refs: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"definition": definition.as_dict(), "sources": source_refs},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _format(value: Any, *, unit: str = "", digits: int = 2) -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}{unit}"


def _relative(path: Path, root: Path) -> str:
    return project_relative_display_path(path, project_root=root)


def _render_stage_table(mode_result: Mapping[str, Any], arm_order: Sequence[str]) -> str:
    from core.console_report import render_table

    metrics = MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS
    headers = [
        "Arm", "Stage", "N", "Coverage",
        *(metric.label for metric in metrics),
        next(item.label for item in MFE_SAFETY_STAGE_ATTRIBUTION_METRICS if item.key == "high_mfe_total_pct"),
    ]
    rows = []
    for arm_id in arm_order:
        for stage in _STAGE_ORDER:
            item = mode_result["arms"][arm_id]["stage_geometry"][stage]
            rows.append([
                arm_id,
                _STAGE_DISPLAY[stage],
                str(item["truth_covered_rows"]),
                _format(item["truth_coverage_pct"], unit="%"),
                *[_format(item.get(metric.key), unit=metric.unit, digits=metric.digits) for metric in metrics],
                _format(item.get("high_mfe_total_pct"), unit="%"),
            ])
    return render_table(headers, rows)


def _render_transition_table(mode_result: Mapping[str, Any], arm_order: Sequence[str]) -> str:
    from core.console_report import render_table

    headers = [
        "Arm", "日別狀態", "Days", "Raw High-MFE", "Planned High-MFE", "Δ High-MFE",
        "Raw LM/HS", "Planned LM/HS", "Δ LM/HS", "Raw membership retained",
    ]
    rows = []
    for arm_id in arm_order:
        transitions = mode_result["arms"][arm_id]["transitions"]
        for state_key, state_label in (
            ("all_eligible", "All eligible"),
            ("direct_feasible", "Raw direct-feasible"),
            ("direct_infeasible", "Raw direct-infeasible"),
        ):
            item = transitions[state_key]
            rows.append([
                arm_id,
                state_label,
                str(item["trade_days"]),
                _format(item["raw"].get("high_mfe_total_pct"), unit="%"),
                _format(item["planned"].get("high_mfe_total_pct"), unit="%"),
                _format(item.get("delta_high_mfe_pp"), unit=" pp"),
                _format(item["raw"].get("low_mfe_high_safety_pct"), unit="%"),
                _format(item["planned"].get("low_mfe_high_safety_pct"), unit="%"),
                _format(item.get("delta_low_mfe_high_safety_pp"), unit=" pp"),
                _format(item.get("mean_raw_membership_retained_pct"), unit="%"),
            ])
    return render_table(headers, rows)


def _render_resource_table(mode_result: Mapping[str, Any], arm_order: Sequence[str]) -> str:
    from core.console_report import render_table

    headers = [
        "Arm", "Eligible days", "Direct infeasible", "K<headroom days", "Headroom slot-days",
        "Median K", "Median free slots", "Final=R0", "Median final R0 slack", "selected=K",
    ]
    rows = []
    for arm_id in arm_order:
        item = mode_result["arms"][arm_id]["resource_contract"]
        rows.append([
            arm_id,
            str(item["eligible_days"]),
            f"{item['direct_infeasible_days']} ({item['direct_infeasible_pct']:.2f}%)",
            f"{item['k_headroom_days']} ({item['k_headroom_pct']:.2f}%)",
            str(item["k_headroom_slot_days"]),
            _format(item["median_k"], digits=1),
            _format(item["median_free_slots"], digits=1),
            (
                f"{item['r0_exact_binding_days']} ({item['r0_exact_binding_pct']:.2f}%)"
                if item.get("r0_exact_binding_pct") is not None
                else "-"
            ),
            _format(item.get("median_final_r0_slack_pct"), unit="%"),
            _format(item["selected_count_equals_k_pct"], unit="%"),
        ])
    return render_table(headers, rows)


def _render_baseline_context(mode_result: Mapping[str, Any]) -> str:
    from core.console_report import render_table

    rows = []
    for key, label in (("candidate_pool", "C58 orderable pool"), ("c58_selected", "C58 filled buys")):
        item = mode_result["baseline_context"][key]
        rows.append([
            label,
            str(item["truth_covered_rows"]),
            _format(item["truth_coverage_pct"], unit="%"),
            _format(item.get("high_mfe_total_pct"), unit="%"),
            _format(item.get("low_mfe_high_safety_pct"), unit="%"),
        ])
    return render_table(("Reference", "N", "Coverage", "High-MFE", "Low-MFE / High-Safety"), rows)


def render_result(result: Mapping[str, Any]) -> str:
    from core.console_report import render_key_values, render_section, render_title

    lines = [render_title("K / R0 Selection Constraint Attribution Audit")]
    lines.append(render_key_values([
        ("Audit", result["audit_id"]),
        ("Threshold", f"same-day percentile >= {result['percentile_cutoff']:.2f} = High"),
        ("Arms", ", ".join(result["arm_order"])),
        ("決策問題", result["decision_question"]),
    ]))
    section_number = 1
    for mode_id in result["evaluation_order"]:
        mode = result["evaluations"][mode_id]
        lines.append(render_section(f"{mode['display_name']}｜Reference geometry", number=section_number)); section_number += 1
        lines.append(_render_baseline_context(mode))
        lines.append(render_section(f"{mode['display_name']}｜Matched eligible-day stage geometry", number=section_number)); section_number += 1
        lines.append(_render_stage_table(mode, result["arm_order"]))
        lines.append(render_section(f"{mode['display_name']}｜Raw Top-K → Planned attribution", number=section_number)); section_number += 1
        lines.append(_render_transition_table(mode, result["arm_order"]))
        lines.append(render_section(f"{mode['display_name']}｜K / R0 contract diagnostics", number=section_number)); section_number += 1
        lines.append(_render_resource_table(mode, result["arm_order"]))
    lines.append(render_section("判讀邊界"))
    lines.append(
        "K headroom只表示C58-derived K低於當日physical free slots且候選供給仍足，"
        "不等價於第K+1檔已通過canonical cash reservation。Raw direct-infeasible只證明Raw Top-K"
        "未通過現行K/R0/cash聯合契約；既有exact-constrained sidecar沒有逐日保存raw count/reserved deficit時，"
        "本Audit不把該日硬歸因為R0-only或cash-only。Final reserved==R0是exact floor binding的描述證據，"
        "不是唯一因果判準。"
    )
    lines.append(render_section("工件輸出"))
    lines.append(render_key_values([(label, path) for label, path in result["artifacts"].items()]))
    return "\n".join(lines)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ])


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# K / R0 Selection Constraint Attribution Audit",
        "",
        f"- Audit: `{result['audit_id']}`",
        f"- Threshold: same-day percentile >= {result['percentile_cutoff']:.2f} = High",
        f"- Arms: {', '.join(result['arm_order'])}",
        f"- 決策問題: {result['decision_question']}",
        "",
    ]
    for mode_id in result["evaluation_order"]:
        mode = result["evaluations"][mode_id]
        lines += [f"## {mode['display_name']}", "", "### Reference geometry", ""]
        ref_rows = []
        for key, label in (("candidate_pool", "C58 orderable pool"), ("c58_selected", "C58 filled buys")):
            item = mode["baseline_context"][key]
            ref_rows.append([
                label, item["truth_covered_rows"], f"{item['truth_coverage_pct']:.2f}%",
                _format(item.get("high_mfe_total_pct"), unit="%"),
                _format(item.get("low_mfe_high_safety_pct"), unit="%"),
            ])
        lines += [_markdown_table(["Reference", "N", "Coverage", "High-MFE", "LM/HS"], ref_rows), ""]
        lines += ["### Matched eligible-day stage geometry", ""]
        stage_rows = []
        for arm_id in result["arm_order"]:
            for stage in _STAGE_ORDER:
                item = mode["arms"][arm_id]["stage_geometry"][stage]
                stage_rows.append([
                    arm_id, _STAGE_DISPLAY[stage], item["truth_covered_rows"],
                    _format(item["truth_coverage_pct"], unit="%"),
                    _format(item.get("high_mfe_high_safety_pct"), unit="%"),
                    _format(item.get("high_mfe_low_safety_pct"), unit="%"),
                    _format(item.get("low_mfe_high_safety_pct"), unit="%"),
                    _format(item.get("low_mfe_low_safety_pct"), unit="%"),
                    _format(item.get("high_mfe_total_pct"), unit="%"),
                ])
        lines += [_markdown_table(
            ["Arm", "Stage", "N", "Coverage", "HM/HS", "HM/LS", "LM/HS", "LM/LS", "High-MFE"],
            stage_rows,
        ), ""]
        lines += ["### Raw Top-K → Planned attribution", ""]
        transition_rows = []
        for arm_id in result["arm_order"]:
            for state_key, state_label in (
                ("all_eligible", "All eligible"),
                ("direct_feasible", "Raw direct-feasible"),
                ("direct_infeasible", "Raw direct-infeasible"),
            ):
                item = mode["arms"][arm_id]["transitions"][state_key]
                transition_rows.append([
                    arm_id, state_label, item["trade_days"],
                    _format(item["raw"].get("high_mfe_total_pct"), unit="%"),
                    _format(item["planned"].get("high_mfe_total_pct"), unit="%"),
                    _format(item.get("delta_high_mfe_pp"), unit=" pp"),
                    _format(item["raw"].get("low_mfe_high_safety_pct"), unit="%"),
                    _format(item["planned"].get("low_mfe_high_safety_pct"), unit="%"),
                    _format(item.get("delta_low_mfe_high_safety_pp"), unit=" pp"),
                    _format(item.get("mean_raw_membership_retained_pct"), unit="%"),
                ])
        lines += [_markdown_table(
            ["Arm", "State", "Days", "Raw High-MFE", "Planned High-MFE", "Δ High-MFE", "Raw LM/HS", "Planned LM/HS", "Δ LM/HS", "Membership retained"],
            transition_rows,
        ), ""]
        lines += ["### K / R0 contract diagnostics", ""]
        resource_rows = []
        for arm_id in result["arm_order"]:
            item = mode["arms"][arm_id]["resource_contract"]
            resource_rows.append([
                arm_id, item["eligible_days"],
                f"{item['direct_infeasible_days']} ({item['direct_infeasible_pct']:.2f}%)",
                f"{item['k_headroom_days']} ({item['k_headroom_pct']:.2f}%)",
                item["k_headroom_slot_days"],
                _format(item.get("r0_exact_binding_pct"), unit="%"),
                _format(item.get("median_final_r0_slack_pct"), unit="%"),
                _format(item.get("selected_count_equals_k_pct"), unit="%"),
            ])
        lines += [_markdown_table(
            ["Arm", "Eligible days", "Direct infeasible", "K headroom", "Headroom slot-days", "Final=R0", "Median R0 slack", "selected=K"],
            resource_rows,
        ), ""]
    lines += [
        "## 判讀邊界",
        "",
        "- `K headroom` 只證明 `K < free_slots` 且 `orderable_candidates > K`，不宣稱第 K+1 檔 cash-feasible。",
        "- `Raw direct-infeasible` 只證明 Raw Top-K 沒通過目前 `K + R0 + canonical cash reservation` 聯合契約。",
        "- 既有 exact-constrained sidecar若沒有 raw count/reserved deficit，就不把失敗日硬拆成 R0-only / cash-only。",
        "- `Final reserved == R0` 是 exact floor binding 的描述證據，不是唯一因果判準。",
        "",
    ]
    return "\n".join(lines)


def _write_stage_csv(path: Path, result: Mapping[str, Any]) -> None:
    fields = [
        "evaluation", "arm_id", "stage", "raw_rows", "truth_covered_rows", "truth_coverage_pct",
        *QUADRANT_KEYS, "high_mfe_total_pct", "high_safety_total_pct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode_id in result["evaluation_order"]:
            mode = result["evaluations"][mode_id]
            for arm_id in result["arm_order"]:
                for stage in _STAGE_ORDER:
                    item = mode["arms"][arm_id]["stage_geometry"][stage]
                    writer.writerow({
                        "evaluation": mode_id,
                        "arm_id": arm_id,
                        "stage": stage,
                        **{field: item.get(field) for field in fields if field not in {"evaluation", "arm_id", "stage"}},
                    })


def _write_resource_csv(path: Path, result: Mapping[str, Any]) -> None:
    fields = [
        "evaluation", "arm_id", "state", "trade_days",
        "raw_high_mfe_pct", "planned_high_mfe_pct", "delta_high_mfe_pp",
        "raw_lmhs_pct", "planned_lmhs_pct", "delta_lmhs_pp", "membership_retained_pct",
        "eligible_days", "direct_infeasible_days", "direct_infeasible_pct",
        "k_headroom_days", "k_headroom_pct", "k_headroom_slot_days",
        "median_k", "median_free_slots", "r0_exact_binding_days", "r0_exact_binding_pct",
        "median_final_r0_slack_pct", "selected_count_equals_k_pct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode_id in result["evaluation_order"]:
            mode = result["evaluations"][mode_id]
            for arm_id in result["arm_order"]:
                resource = mode["arms"][arm_id]["resource_contract"]
                for state_key in ("all_eligible", "direct_feasible", "direct_infeasible"):
                    item = mode["arms"][arm_id]["transitions"][state_key]
                    writer.writerow({
                        "evaluation": mode_id,
                        "arm_id": arm_id,
                        "state": state_key,
                        "trade_days": item["trade_days"],
                        "raw_high_mfe_pct": item["raw"].get("high_mfe_total_pct"),
                        "planned_high_mfe_pct": item["planned"].get("high_mfe_total_pct"),
                        "delta_high_mfe_pp": item.get("delta_high_mfe_pp"),
                        "raw_lmhs_pct": item["raw"].get("low_mfe_high_safety_pct"),
                        "planned_lmhs_pct": item["planned"].get("low_mfe_high_safety_pct"),
                        "delta_lmhs_pp": item.get("delta_low_mfe_high_safety_pp"),
                        "membership_retained_pct": item.get("mean_raw_membership_retained_pct"),
                        **{key: resource.get(key) for key in fields if key in resource},
                    })


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError(f"unsupported audit_type: {definition.audit_type}")
    root = Path(project_root).resolve()
    source_cfg = definition.source
    dims = definition.dimensions
    evaluation_order = tuple(str(v) for v in source_cfg.get("evaluation_profile_ids", ()))
    arm_order = tuple(str(v) for v in source_cfg.get("strategy_arm_ids", ()))
    baseline_arm_id = str(source_cfg.get("baseline_arm_id") or "").strip()
    cutoff = float(dims.get("same_day_percentile_cutoff"))
    percentile_method = str(dims.get("percentile_method") or "average_zero_based")
    if not evaluation_order or not arm_order or not baseline_arm_id:
        raise ValueError(f"{definition.audit_id} source設定不完整")

    truth, truth_source = build_truth_geometry(
        definition, root, percentile_method=percentile_method
    )
    truth = attach_quadrants(truth, cutoff=cutoff)
    evaluations: dict[str, Any] = {}
    source_refs: dict[str, Any] = {"truth": truth_source, "strategy": {}}

    for profile_id in evaluation_order:
        strategy_source = load_strategy_compare_source(root, profile_id=profile_id)
        period = _period_from_source(strategy_source)
        period_truth = filter_period(truth, period[0], period[1])
        baseline = load_strategy_arm_replay_sidecars(
            root, source=strategy_source, arm_id=baseline_arm_id
        )
        baseline_pool_keys = filter_period(
            score_event_keys(pd.DataFrame(baseline["orderable"])), period[0], period[1]
        )
        baseline_selected_keys = filter_period(
            score_event_keys(pd.DataFrame(baseline["orderable"]), pd.DataFrame(baseline["selected"])),
            period[0], period[1],
        )
        baseline_context = {
            "candidate_pool": distribution_for_keys(period_truth, baseline_pool_keys),
            "c58_selected": distribution_for_keys(period_truth, baseline_selected_keys),
        }
        arm_payloads: dict[str, Any] = {}
        arm_sources: dict[str, Any] = {}

        for arm_id in arm_order:
            arm_setting = next(
                (item for item in strategy_source.settings.enabled_arms if item.arm_id == arm_id),
                None,
            )
            if arm_setting is None or not arm_setting.dl_enabled:
                raise AuditBlockedError(f"{profile_id}/{arm_id}不是目前enabled DL-on arm")
            options = dict(arm_setting.dl_runtime_options or {})
            if options.get("preserve_k_r0") is not True:
                raise AuditBlockedError(f"{profile_id}/{arm_id}目前不是preserve_k_r0 arm")

            evidence = load_strategy_arm_pipeline_sidecars(
                root, source=strategy_source, arm_id=arm_id
            )
            orderable = pd.DataFrame(evidence["orderable"]).copy()
            selector_trace = pd.DataFrame(evidence["selector_trace"]).copy()
            execution = pd.DataFrame(evidence["execution"]).copy()
            selected = pd.DataFrame(evidence["selected"]).copy()
            capacity = pd.DataFrame(evidence["daily_capacity"]).copy()
            resource = summarize_resource_contract(capacity)
            eligible_dates = resource["eligible_trade_dates"]

            orderable_events, _ = strategy_replay_score_event_frames(orderable, None)
            orderable_events["ticker"] = orderable_events["ticker"].map(normalize_ticker)
            orderable_events["trade_date"] = orderable_events["trade_date"].map(normalize_date)
            orderable_events["score_event_date"] = orderable_events["score_event_date"].map(normalize_date)
            raw = _raw_top_k_with_score_event(selector_trace)
            _validate_raw_stage_capacity(raw, capacity)
            planned = _execution_with_score_event(execution, orderable=orderable)
            filled = planned.loc[planned["entry_filled_bool"]].copy()

            stage_geometry = _stage_geometry(
                period_truth,
                orderable_events=orderable_events,
                raw=raw,
                planned=planned,
                filled=filled,
                eligible_dates=eligible_dates,
            )
            transitions = {
                "all_eligible": build_transition_summary(
                    period_truth, raw=raw, planned=planned, trade_dates=eligible_dates
                ),
                "direct_feasible": build_transition_summary(
                    period_truth,
                    raw=raw,
                    planned=planned,
                    trade_dates=resource["direct_feasible_trade_dates"],
                ),
                "direct_infeasible": build_transition_summary(
                    period_truth,
                    raw=raw,
                    planned=planned,
                    trade_dates=resource["direct_infeasible_trade_dates"],
                ),
            }
            consistency = _filled_selected_consistency(
                orderable=orderable, selected=selected, filled=filled
            )
            if consistency.get("key_match_pct") != 100.0:
                raise AuditBlockedError(
                    f"{profile_id}/{arm_id} execution filled與selected_buys occurrence不一致: "
                    f"{consistency}"
                )
            arm_payloads[arm_id] = {
                "stage_geometry": stage_geometry,
                "transitions": transitions,
                "resource_contract": resource,
                "filled_selected_consistency": consistency,
            }
            arm_sources[arm_id] = {
                "pair_dir": _relative(Path(evidence["pair_dir"]), root),
                "orderable": _relative(Path(evidence["orderable_path"]), root),
                "selector_trace": _relative(Path(evidence["selector_trace_path"]), root),
                "execution": _relative(Path(evidence["execution_path"]), root),
                "selected_buys": _relative(Path(evidence["selected_path"]), root),
                "daily_capacity": _relative(Path(evidence["daily_capacity_path"]), root),
            }

        evaluations[profile_id] = {
            "display_name": strategy_source.settings.profile_label,
            "period": {"start": period[0], "end": period[1]},
            "strategy_config_fingerprint": strategy_source.config_fingerprint,
            "baseline_context": baseline_context,
            "arms": arm_payloads,
        }
        source_refs["strategy"][profile_id] = {
            "run_dir": _relative(strategy_source.run_dir, root),
            "config_fingerprint": strategy_source.config_fingerprint,
            "baseline_arm_id": baseline_arm_id,
            "baseline_pair_dir": _relative(Path(baseline["pair_dir"]), root),
            "arms": arm_sources,
        }

    fingerprint = _fingerprint_payload(definition, source_refs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_dir = output_root / "runs" / f"{timestamp}_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "selection_k_r0_attribution.json"
    stage_csv = run_dir / "stage_geometry.csv"
    resource_csv = run_dir / "resource_attribution.csv"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "module_id": definition.module_id,
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "decision_question": str(definition.outcomes.get("decision_question") or ""),
        "critical_uncertainty": str(definition.outcomes.get("critical_uncertainty") or ""),
        "stopping_condition": str(definition.outcomes.get("stopping_condition") or ""),
        "percentile_cutoff": cutoff,
        "percentile_method": percentile_method,
        "arm_order": list(arm_order),
        "baseline_arm_id": baseline_arm_id,
        "evaluation_order": list(evaluation_order),
        "truth_sources": truth_source,
        "evaluations": evaluations,
        "attribution_boundary": {
            "k_headroom": "K < physical free slots and orderable candidates > K; not proof K+1 is cash-feasible",
            "direct_infeasible": "Raw Top-K failed combined K/R0/canonical cash reservation contract",
            "r0_specific": "Do not infer R0-only vs cash-only unless persisted raw deficit evidence exists",
            "r0_exact_binding": "Final reserved == R0 is descriptive exact-floor binding evidence",
        },
        "artifacts": {
            "完整JSON": _relative(json_path, root),
            "Stage CSV": _relative(stage_csv, root),
            "Resource CSV": _relative(resource_csv, root),
            "詳細Markdown": _relative(report_path, root),
            "Manifest": _relative(manifest_path, root),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_stage_csv(stage_csv, result)
    _write_resource_csv(resource_csv, result)
    report_path.write_text(_render_markdown(result), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "audit_definition": definition.as_dict(),
        "config_fingerprint": fingerprint,
        "source_refs": source_refs,
        "artifacts": result["artifacts"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest.json").write_text(json.dumps({
        "run_dir": _relative(run_dir, root),
        "report": _relative(report_path, root),
        "result": _relative(json_path, root),
        "config_fingerprint": fingerprint,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_dir = output_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "audit.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "audit.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "stage_geometry.csv").write_bytes(stage_csv.read_bytes())
    (latest_dir / "resource_attribution.csv").write_bytes(resource_csv.read_bytes())
    (latest_dir / "manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    return result


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    blockers: list[str] = []
    target_paths: list[str] = []
    strategy_paths: list[str] = []
    try:
        filter_id, _arch, provider_profile_id, mfe_target_id, safety_target_id = (
            resolve_truth_provider_contract(definition)
        )
        dataset_dir = resolve_filter_output_dir(root, filter_id=filter_id)
        dataset = resolve_dataset_paths(dataset_dir)
        target_paths.append(_relative(dataset.summary, root))
        if not dataset.summary.is_file():
            blockers.append("缺少canonical Dataset summary: " + _relative(dataset.summary, root))
        target_paths.append(
            f"profile:{provider_profile_id} / target:{mfe_target_id} / safety:{safety_target_id}"
        )
    except (ValueError, OSError) as exc:
        blockers.append(str(exc))

    source_cfg = definition.source
    arm_order = tuple(str(v) for v in source_cfg.get("strategy_arm_ids", ()))
    baseline_arm_id = str(source_cfg.get("baseline_arm_id") or "").strip()
    for profile_id in tuple(str(v) for v in source_cfg.get("evaluation_profile_ids", ())):
        try:
            strategy_source = load_strategy_compare_source(root, profile_id=profile_id)
            _period_from_source(strategy_source)
            strategy_paths.append(_relative(strategy_source.run_dir, root))
            baseline = load_strategy_arm_replay_sidecars(
                root, source=strategy_source, arm_id=baseline_arm_id
            )
            if pd.DataFrame(baseline["orderable"]).empty or pd.DataFrame(baseline["selected"]).empty:
                raise AuditBlockedError(f"{profile_id}/{baseline_arm_id} baseline row sidecar為空")
            for arm_id in arm_order:
                arm_setting = next(
                    (item for item in strategy_source.settings.enabled_arms if item.arm_id == arm_id),
                    None,
                )
                if arm_setting is None or not arm_setting.dl_enabled:
                    raise AuditBlockedError(f"{profile_id}/{arm_id}不是目前enabled DL-on arm")
                if dict(arm_setting.dl_runtime_options or {}).get("preserve_k_r0") is not True:
                    raise AuditBlockedError(f"{profile_id}/{arm_id}不是preserve_k_r0 arm")
                evidence = load_strategy_arm_pipeline_sidecars(
                    root, source=strategy_source, arm_id=arm_id
                )
                if pd.DataFrame(evidence["orderable"]).empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} orderable sidecar為空")
                if pd.DataFrame(evidence["selector_trace"]).empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} selector trace為空")
                if pd.DataFrame(evidence["execution"]).empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} execution sidecar為空")
                raw = _raw_top_k_with_score_event(pd.DataFrame(evidence["selector_trace"]))
                _validate_raw_stage_capacity(raw, pd.DataFrame(evidence["daily_capacity"]))
                summarize_resource_contract(pd.DataFrame(evidence["daily_capacity"]))
                planned = _execution_with_score_event(
                    pd.DataFrame(evidence["execution"]),
                    orderable=pd.DataFrame(evidence["orderable"]),
                )
                filled = planned.loc[planned["entry_filled_bool"]].copy()
                consistency = _filled_selected_consistency(
                    orderable=pd.DataFrame(evidence["orderable"]),
                    selected=pd.DataFrame(evidence["selected"]),
                    filled=filled,
                )
                if consistency.get("key_match_pct") != 100.0:
                    raise AuditBlockedError(
                        f"{profile_id}/{arm_id} execution filled與selected_buys occurrence不一致: "
                        f"{consistency}"
                    )
        except (ValueError, KeyError, OSError, AuditSourceBlockedError, AuditBlockedError) as exc:
            blockers.append(str(exc))
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "target_paths": target_paths,
        "strategy_paths": strategy_paths,
    }


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    state = preflight(definition, project_root=Path(project_root))
    return {
        "status": str(state.get("status") or "BLOCKED"),
        "reason": "；".join(str(v) for v in state.get("blockers", ())),
        "source": {
            "display": "Completed OOS/Rolling selector trace + execution + MFE/Safety truth",
            "target_paths": list(state.get("target_paths", ())),
            "strategy_paths": list(state.get("strategy_paths", ())),
        },
    }


def run_formal_audit(
    definition: AuditDefinition,
    *,
    project_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    result = run_audit(definition, project_root=Path(project_root))
    if not quiet:
        print(render_result(result))
    return result


__all__ = [
    "SUPPORTED_AUDIT_TYPE",
    "build_transition_summary",
    "collect_status",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
    "summarize_resource_contract",
]
