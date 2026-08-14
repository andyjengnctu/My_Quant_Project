"""Canonical builders for versioned event-style continuous-target artifacts.

These builders are part of the model-work service layer.  They rebuild fixed
Target arrays from the canonical Dataset identity and never depend on historical
Audit reports, Strategy Compare outputs, or OOS-fitted coefficients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
)
from core.console_report import (
    compact_console_enabled,
    console_color_enabled,
    paint,
    print_artifact_paths,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import (
    CONTINUOUS_TARGET_SCHEMA_VERSION,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_ADVERSE_RETURN_FILENAME,
    TARGET_AUDIT_JSON_FILENAME,
    TARGET_AUDIT_MARKDOWN_FILENAME,
    TARGET_DAILY_CSV_FILENAME,
    TARGET_FAVORABLE_RETURN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    TARGET_OPPORTUNITY_BAR_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_RISK_BREACH_BAR_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    StrategyAlignedContinuousTargetSpec,
    build_strategy_aligned_group_targets,
    build_strategy_aligned_no_time_contract,
    build_strategy_aligned_no_time_group_targets,
    load_validated_continuous_target_component_arrays,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.dataset_store import load_npy, save_npy_atomic
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from filters.breakout_quality.workflow_io import (
    PROJECT_ROOT,
    dataset_paths,
    load_validated_dataset_bundle,
    write_json,
)
from services.breakout_quality.continuous_target_metrics import (
    collapse_group_frame,
    daily_rankability,
    distribution_metrics,
    format_metric,
    format_percent,
    same_day_binary_concordance,
    source_data_end,
    spearman,
)

BUILD_REPORT_SCHEMA_VERSION = 2
_ALL_SPLITS = ("inner_train", "validation", "selection", "oos")
_SELECTION_SPLITS = ("inner_train", "validation", "selection")


def _artifact_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "target_raw_r": target_dir / TARGET_RAW_FILENAME,
        "favorable_return": target_dir / TARGET_FAVORABLE_RETURN_FILENAME,
        "adverse_return_to_peak": target_dir / TARGET_ADVERSE_RETURN_FILENAME,
        "opportunity_bar": target_dir / TARGET_OPPORTUNITY_BAR_FILENAME,
        "first_risk_breach_bar": target_dir / TARGET_RISK_BREACH_BAR_FILENAME,
        "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
    }


def _split_metrics(
    group_frame: pd.DataFrame,
    *,
    split_names: tuple[str, ...],
    source_target_column: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result: dict[str, Any] = {}
    daily_frames: list[pd.DataFrame] = []
    for split_name in split_names:
        subset = group_frame[
            group_frame[f"is_{split_name}"] & group_frame["valid_mask"]
        ].copy()
        metrics = distribution_metrics(subset)
        rankability, daily = daily_rankability(subset, split_name=split_name)
        metrics["same_day_rankability"] = rankability
        metrics["same_day_binary_concordance"] = same_day_binary_concordance(subset)
        if source_target_column:
            metrics["source_vs_no_time_spearman"] = spearman(
                subset[source_target_column], subset["target_raw_r"]
            )
            metrics["mean_delta_vs_source_target"] = (
                float((subset["target_raw_r"] - subset[source_target_column]).mean())
                if not subset.empty
                else None
            )
        result[split_name] = metrics
        daily_frames.append(daily)
    daily_frame = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return result, daily_frame


def _render_base_report(payload: dict[str, Any]) -> str:
    contract = payload["target_contract"]
    lines = [
        "# Continuous Target Build Report",
        "",
        f"- Target：`{contract['target_id']}`",
        f"- Horizon：`{contract['horizon_bars']}` trading bars",
        f"- Risk budget：`{float(contract['risk_budget_return']) * 100:.2f}%`",
        "- Builder：canonical Dataset → fixed Target formula；不依賴歷史Audit或Strategy Compare工件。",
        "- Training：none。Target公式不讀取split/OOS統計量。",
        "",
        "| Split | Groups | Mean | P01 | P50 | P99 | Positive | Unique | Binary AUC | Rankable dates | Non-tie pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in _ALL_SPLITS:
        metrics = payload["split_metrics"][split_name]
        q = metrics.get("quantiles") or {}
        rankability = metrics.get("same_day_rankability") or {}
        lines.append(
            f"| {split_name} | {int(metrics.get('group_count', 0)):,} "
            f"| {format_metric(metrics.get('mean'))} | {format_metric(q.get('p01'))} "
            f"| {format_metric(q.get('p50'))} | {format_metric(q.get('p99'))} "
            f"| {format_percent(metrics.get('positive_rate'))} "
            f"| {format_percent(metrics.get('unique_value_ratio'))} "
            f"| {format_metric(metrics.get('binary_label_auc'))} "
            f"| {format_percent(rankability.get('rankable_date_rate'))} "
            f"| {format_percent(rankability.get('pairwise_non_tie_rate'))} |"
        )
    lines += [
        "",
        "固定公式與target identity沿用`filters/breakout_quality/continuous_target.py`；此報表只描述已建立工件，不是研究gate。",
        "",
    ]
    return "\n".join(lines)


def _render_no_time_report(payload: dict[str, Any]) -> str:
    contract = payload["target_contract"]
    lines = [
        "# No-time Continuous Target Build Report",
        "",
        f"- Target：`{contract['target_id']}`",
        f"- Source Target：`{contract['source_target_id']}`",
        "- 固定公式：`target_raw_r = favorable_r - adverse_r`。",
        "- Builder：canonical Dataset identity + source Target components。",
        "- Historical research approval report：not required；已採用固定公式由Registry/Log保存。",
        "- OOS：不在Target build階段評估，不參與Target公式。",
        "",
        "| Split | Groups | Mean | P01 | P50 | P99 | Positive | Unique | Binary AUC | Rankable dates | Source rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in _SELECTION_SPLITS:
        metrics = payload["split_metrics"][split_name]
        q = metrics.get("quantiles") or {}
        rankability = metrics.get("same_day_rankability") or {}
        lines.append(
            f"| {split_name} | {int(metrics.get('group_count', 0)):,} "
            f"| {format_metric(metrics.get('mean'))} | {format_metric(q.get('p01'))} "
            f"| {format_metric(q.get('p50'))} | {format_metric(q.get('p99'))} "
            f"| {format_percent(metrics.get('positive_rate'))} "
            f"| {format_percent(metrics.get('unique_value_ratio'))} "
            f"| {format_metric(metrics.get('binary_label_auc'))} "
            f"| {format_percent(rankability.get('rankable_date_rate'))} "
            f"| {format_metric(metrics.get('source_vs_no_time_spearman'))} |"
        )
    lines += [
        "",
        "此報表為artifact build diagnostics，不重新執行已結案的歷史研究決策。",
        "",
    ]
    return "\n".join(lines)


def build_strategy_aligned_target_artifacts(
    *,
    filter_id: str,
    allow_stale_source: bool = False,
) -> int:
    summary, indexed_features, _context, labels, events = load_validated_dataset_bundle(
        filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    paths = dataset_paths(filter_id)
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    group_anchor_prices = load_npy(paths.group_anchor_prices)
    future_high_prices = load_npy(paths.future_high_prices)
    future_low_prices = load_npy(paths.future_low_prices)
    future_available_bars = load_npy(paths.future_available_bars)

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    targets = build_strategy_aligned_group_targets(
        group_anchor_prices,
        future_high_prices,
        future_low_prices,
        future_available_bars,
        spec=spec,
    )
    group_count = int(len(group_anchor_prices))
    valid_count = int(np.asarray(targets["valid_mask"], dtype=bool).sum())
    if valid_count <= 0:
        raise ValueError("continuous target沒有任何有效group")

    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=source_data_end(summary, events),
    )
    (
        _split_assignments,
        inner_train_idx,
        validation_idx,
        selection_idx,
        oos_idx,
        split_report,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=bool(BREAKOUT_QUALITY_USE_INNER_VALIDATION),
        inner_validation_months=int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS),
        early_stopping_enabled=bool(
            BREAKOUT_QUALITY_USE_INNER_VALIDATION
            and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) > 0
        ),
    )
    split_group_indices = {
        "inner_train": np.unique(event_group_index[inner_train_idx]),
        "validation": np.unique(event_group_index[validation_idx]),
        "selection": np.unique(event_group_index[selection_idx]),
        "oos": np.unique(event_group_index[oos_idx]),
    }
    group_frame = collapse_group_frame(events, event_group_index, targets, split_group_indices)
    split_metrics, daily_frame = _split_metrics(group_frame, split_names=_ALL_SPLITS)
    payload = {
        "schema_version": BUILD_REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT",
        "builder": "services.breakout_quality.continuous_target_builder",
        "filter_id": str(filter_id),
        "target_contract": spec.contract_payload(),
        "dataset": {
            "dataset_profile": summary.get("dataset"),
            "event_count": int(summary.get("event_count", len(events))),
            "feature_group_count": int(summary.get("feature_group_count", group_count)),
            "source_data_date_range": summary.get("source_data_date_range"),
            "label_policy": summary.get("label_policy"),
        },
        "split_contract": split_report,
        "split_metrics": split_metrics,
        "build_contract": {
            "training_performed": False,
            "historical_audit_required": False,
            "strategy_compare_dependency": False,
            "oos_statistics_used_to_define_target": False,
        },
    }

    target_dir = resolve_continuous_target_dir(PROJECT_ROOT, filter_id, target_id=STRATEGY_ALIGNED_TARGET_ID)
    target_dir.mkdir(parents=True, exist_ok=True)
    array_paths = _artifact_paths(target_dir)
    for name, path in array_paths.items():
        save_npy_atomic(path, np.asarray(targets[name]))
    report_json_path = target_dir / TARGET_AUDIT_JSON_FILENAME
    report_markdown_path = target_dir / TARGET_AUDIT_MARKDOWN_FILENAME
    daily_csv_path = target_dir / TARGET_DAILY_CSV_FILENAME
    write_json(report_json_path, payload)
    report_markdown_path.write_text(_render_base_report(payload), encoding="utf-8")
    daily_frame.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")

    source_artifacts = summary.get("dataset_artifacts")
    manifest = {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": str(filter_id),
        "target_contract": spec.contract_payload(),
        "group_count": group_count,
        "valid_group_count": valid_count,
        "invalid_group_count": int(group_count - valid_count),
        "dataset_policy": summary.get("policy"),
        "dataset_artifact_source": {
            key: value
            for key, value in dict(source_artifacts or {}).items()
            if key in {
                "group_anchor_prices",
                "future_high_prices",
                "future_low_prices",
                "future_available_bars",
                "event_group_index",
                "events_csv",
            }
        },
        "split_policy": outer_policy,
        "split_report": split_report,
        "artifacts": {name: build_file_manifest(path) for name, path in array_paths.items()},
        "audit_outputs": {
            "json": build_file_manifest(report_json_path),
            "markdown": build_file_manifest(report_markdown_path),
            "daily_csv": build_file_manifest(daily_csv_path),
        },
        "build_contract": payload["build_contract"],
        "training_performed": False,
        "runtime_eligible": False,
    }
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    if compact_console_enabled():
        print_artifact_paths(
            (("Target manifest", manifest_path), ("Markdown 報表", report_markdown_path)),
            project_root=PROJECT_ROOT,
        )
        return 0
    print(f"Continuous Target完成 | target={spec.target_id} | valid={valid_count:,}/{group_count:,}")
    print_artifact_paths(
        (("Target manifest", manifest_path), ("Markdown 報表", report_markdown_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


def build_strategy_aligned_no_time_target_artifacts(
    *,
    filter_id: str,
    allow_stale_source: bool = False,
) -> int:
    summary, indexed_features, _context, labels, events = load_validated_dataset_bundle(
        filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    group_count = int(event_group_index.max()) + 1 if len(event_group_index) else 0
    source_manifest, source_arrays = load_validated_continuous_target_component_arrays(
        PROJECT_ROOT,
        filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
        expected_group_count=group_count,
        expected_dataset_policy=summary.get("policy"),
        expected_dataset_artifacts=summary.get("dataset_artifacts"),
    )
    source_contract = source_manifest.get("target_contract") or {}
    target_contract = build_strategy_aligned_no_time_contract(source_contract)
    targets = build_strategy_aligned_no_time_group_targets(
        favorable_return=source_arrays["favorable_return"],
        adverse_return_to_peak=source_arrays["adverse_return_to_peak"],
        opportunity_bar=source_arrays["opportunity_bar"],
        first_risk_breach_bar=source_arrays["first_risk_breach_bar"],
        valid_mask=source_arrays["valid_mask"],
        risk_budget_return=float(target_contract["risk_budget_return"]),
    )
    valid_count = int(np.asarray(targets["valid_mask"], dtype=bool).sum())
    if group_count <= 0 or valid_count <= 0:
        raise ValueError("No-time continuous target沒有可用group target")

    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=source_data_end(summary, events),
    )
    (
        _split_assignments,
        inner_train_idx,
        validation_idx,
        selection_idx,
        _oos_idx,
        split_report,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=bool(BREAKOUT_QUALITY_USE_INNER_VALIDATION),
        inner_validation_months=int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS),
        early_stopping_enabled=bool(
            BREAKOUT_QUALITY_USE_INNER_VALIDATION
            and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) > 0
        ),
    )
    split_group_indices = {
        "inner_train": np.unique(event_group_index[inner_train_idx]),
        "validation": np.unique(event_group_index[validation_idx]),
        "selection": np.unique(event_group_index[selection_idx]),
    }
    frame_targets = dict(targets)
    frame_targets["source_target_raw_r"] = np.asarray(source_arrays["target_raw_r"], dtype=np.float32)
    group_frame = collapse_group_frame(events, event_group_index, frame_targets, split_group_indices)
    split_metrics, daily_frame = _split_metrics(
        group_frame,
        split_names=_SELECTION_SPLITS,
        source_target_column="source_target_raw_r",
    )
    payload = {
        "schema_version": BUILD_REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT",
        "builder": "services.breakout_quality.continuous_target_builder",
        "filter_id": str(filter_id),
        "target_contract": target_contract,
        "source_target": {
            "target_id": STRATEGY_ALIGNED_TARGET_ID,
            "manifest_generated_at_utc": source_manifest.get("generated_at_utc"),
        },
        "dataset": {
            "dataset_profile": summary.get("dataset"),
            "event_count": int(summary.get("event_count", len(events))),
            "feature_group_count": int(summary.get("feature_group_count", group_count)),
            "source_data_date_range": summary.get("source_data_date_range"),
            "label_policy": summary.get("label_policy"),
        },
        "split_contract": split_report,
        "evaluated_splits": list(_SELECTION_SPLITS),
        "oos_evaluated": False,
        "split_metrics": split_metrics,
        "build_contract": {
            "training_performed": False,
            "historical_research_gate_required": False,
            "historical_audit_required": False,
            "fixed_formula_only": True,
            "oos_fitted_coefficient": False,
        },
    }

    target_dir = resolve_continuous_target_dir(
        PROJECT_ROOT,
        filter_id,
        target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    array_paths = _artifact_paths(target_dir)
    for name, path in array_paths.items():
        save_npy_atomic(path, np.asarray(targets[name]))
    report_json_path = target_dir / TARGET_AUDIT_JSON_FILENAME
    report_markdown_path = target_dir / TARGET_AUDIT_MARKDOWN_FILENAME
    daily_csv_path = target_dir / TARGET_DAILY_CSV_FILENAME
    write_json(report_json_path, payload)
    report_markdown_path.write_text(_render_no_time_report(payload), encoding="utf-8")
    daily_frame.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")

    source_manifest_path = resolve_continuous_target_dir(
        PROJECT_ROOT,
        filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
    ) / TARGET_MANIFEST_FILENAME
    manifest = {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": str(filter_id),
        "target_contract": target_contract,
        "group_count": group_count,
        "valid_group_count": valid_count,
        "invalid_group_count": int(group_count - valid_count),
        "dataset_policy": summary.get("policy"),
        "split_policy": outer_policy,
        "split_report": split_report,
        "evaluated_splits": list(_SELECTION_SPLITS),
        "oos_evaluated": False,
        "source_target": {
            "target_id": STRATEGY_ALIGNED_TARGET_ID,
            "manifest": build_file_manifest(source_manifest_path),
        },
        "artifacts": {name: build_file_manifest(path) for name, path in array_paths.items()},
        "audit_outputs": {
            "json": build_file_manifest(report_json_path),
            "markdown": build_file_manifest(report_markdown_path),
            "daily_csv": build_file_manifest(daily_csv_path),
        },
        "build_contract": payload["build_contract"],
        "training_performed": False,
        "runtime_eligible": False,
        "research_only": True,
    }
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    if compact_console_enabled():
        selection = split_metrics["selection"]
        print(
            paint("Continuous Target 完成", "green", enabled=console_color_enabled(), bold=True)
            + f" | valid={valid_count:,}/{group_count:,}"
            + f" | Selection mean={format_metric(selection.get('mean'))}R"
            + f" | AUC={format_metric(selection.get('binary_label_auc'))}"
            + f" | source rho={format_metric(selection.get('source_vs_no_time_spearman'))}"
        )
        print_artifact_paths(
            (("Target manifest", manifest_path), ("Markdown 報表", report_markdown_path)),
            project_root=PROJECT_ROOT,
        )
        return 0
    print(
        "No-time Continuous Target完成 | "
        f"target={STRATEGY_ALIGNED_NO_TIME_TARGET_ID} | valid={valid_count:,}/{group_count:,}"
    )
    print_artifact_paths(
        (("Target manifest", manifest_path), ("Markdown 報表", report_markdown_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = [
    "BUILD_REPORT_SCHEMA_VERSION",
    "build_strategy_aligned_no_time_target_artifacts",
    "build_strategy_aligned_target_artifacts",
]
