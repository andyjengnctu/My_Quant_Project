"""11F selection-only audit for the versioned no-time continuous target."""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    get_breakout_quality_workflow_settings,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
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
    build_strategy_aligned_no_time_contract,
    build_strategy_aligned_no_time_group_targets,
    load_validated_continuous_target_component_arrays,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import DEFAULT_FILTER_ID, DEFAULT_LABEL_POLICY
from filters.breakout_quality.dataset_store import save_npy_atomic
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from tools.filters.breakout_quality.audit_continuous_target import (
    _collapse_group_frame,
    _daily_rankability,
    _distribution_metrics,
    _fmt,
    _pct,
    _same_day_binary_concordance,
    _source_data_end,
    _spearman,
)
from tools.filters.breakout_quality.audit_target_component_attribution import (
    _ranker_dir,
    _read_json,
    _sha256_file,
)
from tools.filters.breakout_quality.audit_target_time_penalty_ablation import (
    ACTUAL_ABLATION_FILENAME,
    AUDIT_DIRNAME as ABLATION_AUDIT_DIRNAME,
    AUDIT_JSON_FILENAME as ABLATION_AUDIT_JSON_FILENAME,
    QUALIFIED_ABLATION_FILENAME,
)
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    load_validated_dataset_bundle,
    write_json,
)

AUDIT_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "11F No-time Target Arrays and Selection-only Learnability Audit"
SELECTION_SPLITS = ("inner_train", "validation", "selection")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11F建立固定No-time target version arrays，並只稽核Inner Train／Validation／Selection；"
            "不評估OOS、不訓練、不選epoch"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--ranker-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=(STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,),
        help="只用於定位已完成11E研究工件；不載入模型或score",
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="僅供歷史重現；允許Dataset來源inventory與目前資料不同",
    )
    parser.add_argument(
        "--approved-workflow-rebuild",
        action="store_true",
        help=(
            "目前active workflow已正式選用No-time target時，允許只依固定公式與"
            "Dataset identity重建工件；不重跑歷史11E研究gate"
        ),
    )
    return parser.parse_args(argv)


def _artifact_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "target_raw_r": target_dir / TARGET_RAW_FILENAME,
        "favorable_return": target_dir / TARGET_FAVORABLE_RETURN_FILENAME,
        "adverse_return_to_peak": target_dir / TARGET_ADVERSE_RETURN_FILENAME,
        "opportunity_bar": target_dir / TARGET_OPPORTUNITY_BAR_FILENAME,
        "first_risk_breach_bar": target_dir / TARGET_RISK_BREACH_BAR_FILENAME,
        "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
    }


def _validated_11e_report(
    *,
    filter_id: str,
    ranker_profile: str,
) -> tuple[dict[str, Any], Path]:
    audit_dir = _ranker_dir(filter_id, ranker_profile) / ABLATION_AUDIT_DIRNAME
    report_path = audit_dir / ABLATION_AUDIT_JSON_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(
            f"11F需要已完成11E report: {report_path}；請先執行audit-target-time-ablation"
        )
    report = _read_json(report_path)
    if not str(report.get("status") or "").startswith("RESULT_AVAILABLE"):
        raise ValueError("11F只接受已完成的11E結果")
    interpretation = report.get("interpretation_contract") or {}
    if bool(interpretation.get("research_only")) is not True:
        raise ValueError("11F預期11E維持research-only")
    if bool(interpretation.get("training_performed")):
        raise ValueError("11F拒絕任何由11E訓練產生的工件")
    if str(report.get("source_continuous_target_id") or "") != STRATEGY_ALIGNED_TARGET_ID:
        raise ValueError("11F 11E source target id不一致")

    artifact_specs = {
        "qualified_ablation": audit_dir / QUALIFIED_ABLATION_FILENAME,
        "actual_trade_ablation": audit_dir / ACTUAL_ABLATION_FILENAME,
    }
    for key, path in artifact_specs.items():
        record = ((report.get("artifacts") or {}).get(key) or {})
        if not isinstance(record, dict) or not path.is_file():
            raise ValueError(f"11F 11E artifact不存在: {key}")
        expected = str(record.get("sha256") or "").lower()
        if not expected or _sha256_file(path).lower() != expected:
            raise ValueError(f"11F偵測到11E artifact SHA256不一致: {key}")

    actual = report.get("actual_trades") or {}
    correlations = actual.get("correlations") or {}
    deltas = actual.get("deltas") or {}
    pass_metrics = (actual.get("by_label") or {}).get("pass") or {}
    pass_correlations = pass_metrics.get("correlations") or {}
    required_values = {
        "overall_delta": deltas.get("spearman_no_time_minus_original"),
        "decile_spread_delta": deltas.get("decile_spread_no_time_minus_original"),
        "pass_original": pass_correlations.get("original_target_vs_realized_r"),
        "pass_no_time": pass_correlations.get("no_time_target_vs_realized_r"),
        "overall_no_time": correlations.get("no_time_target_vs_realized_r"),
    }
    numeric: dict[str, float] = {}
    for name, value in required_values.items():
        try:
            numeric[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"11F 11E gate缺少有限數值: {name}") from exc
        if not math.isfinite(numeric[name]):
            raise ValueError(f"11F 11E gate含非有限數值: {name}")
    if not (
        numeric["overall_delta"] > 0.0
        and numeric["decile_spread_delta"] > 0.0
        and numeric["pass_no_time"] > numeric["pass_original"]
    ):
        raise ValueError("11F只允許在11E overall、PASS與decile spread均改善時建立新Target")
    return report, report_path


def _approved_workflow_rebuild_gate(*, filter_id: str) -> dict[str, Any]:
    settings = get_breakout_quality_workflow_settings()
    if not settings.is_continuous_ranker:
        raise ValueError("approved workflow rebuild只允許continuous ranker workflow")
    if str(settings.filter_id) != str(filter_id):
        raise ValueError("approved workflow rebuild的filter_id與目前workflow不一致")
    if str(settings.continuous_target_id or "") != STRATEGY_ALIGNED_NO_TIME_TARGET_ID:
        raise ValueError("目前workflow未選用No-time continuous target")
    return {
        "approval_basis": "active_workflow_profile",
        "experiment_profile": str(settings.experiment_profile),
        "continuous_target_id": str(settings.continuous_target_id),
        "historical_research_gate_recomputed": False,
        "historical_research_result_reused": True,
        "fixed_formula_only": True,
        "oos_fitted_coefficient": False,
        "overall_spearman_delta": None,
        "pass_spearman_delta": None,
        "decile_spread_delta": None,
    }


def _selection_metrics(
    group_frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result: dict[str, Any] = {}
    daily_frames: list[pd.DataFrame] = []
    for split_name in SELECTION_SPLITS:
        subset = group_frame[
            group_frame[f"is_{split_name}"] & group_frame["valid_mask"]
        ].copy()
        distribution = _distribution_metrics(subset)
        rankability, daily = _daily_rankability(subset, split_name=split_name)
        distribution["same_day_rankability"] = rankability
        distribution["same_day_binary_concordance"] = _same_day_binary_concordance(subset)
        distribution["source_vs_no_time_spearman"] = _spearman(
            subset["source_target_raw_r"], subset["target_raw_r"]
        )
        distribution["mean_delta_vs_source_target"] = float(
            (subset["target_raw_r"] - subset["source_target_raw_r"]).mean()
        ) if not subset.empty else None
        result[split_name] = distribution
        daily_frames.append(daily)
    daily_frame = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return result, daily_frame


def render_markdown(payload: dict[str, Any]) -> str:
    contract = payload["target_contract"]
    lines = [
        "# 11F No-time Continuous Target Selection-only Audit",
        "",
        f"- Target：`{contract['target_id']}`",
        f"- Source Target：`{contract['source_target_id']}`",
        "- 固定公式：`target_raw_r = favorable_r - adverse_r`。",
        "- 狀態：建立獨立version arrays並只完成Selection內可學性稽核；本輪沒有訓練模型。",
        "- OOS邊界：本輪不建立OOS指標、不讀actual trade R、不選epoch、不調loss或normalization。",
        "- 研究誠實性：移除time penalty的假設來自先前迭代OOS歸因；本輪沒有擬合係數或使用OOS rows計算Target。",
        "",
        "## 1. Selection-only分布與同日排序可學性",
        "",
        "| 區段 | Groups | Mean | P01 | P50 | P99 | 正值率 | Unique率 | Binary AUC | 同日可排序日 | Pair非Tie率 | 與11A Target ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in SELECTION_SPLITS:
        metrics = payload["split_metrics"][split_name]
        q = metrics.get("quantiles") or {}
        rankability = metrics.get("same_day_rankability") or {}
        lines.append(
            f"| {split_name} | {int(metrics.get('group_count', 0)):,} "
            f"| {_fmt(metrics.get('mean'))} | {_fmt(q.get('p01'))} | {_fmt(q.get('p50'))} "
            f"| {_fmt(q.get('p99'))} | {_pct(metrics.get('positive_rate'))} "
            f"| {_pct(metrics.get('unique_value_ratio'))} | {_fmt(metrics.get('binary_label_auc'))} "
            f"| {_pct(rankability.get('rankable_date_rate'))} | {_pct(rankability.get('pairwise_non_tie_rate'))} "
            f"| {_fmt(metrics.get('source_vs_no_time_spearman'))} |"
        )
    lines += [
        "",
        "## 2. 與既有二元Label的關係",
        "",
        "| 區段 | PASS target平均 | REJECT target平均 | 同日PASS/REJECT concordance | Top 1%正target貢獻 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in SELECTION_SPLITS:
        metrics = payload["split_metrics"][split_name]
        concordance = metrics.get("same_day_binary_concordance") or {}
        lines.append(
            f"| {split_name} | {_fmt(metrics.get('pass_target_mean'))} "
            f"| {_fmt(metrics.get('reject_target_mean'))} "
            f"| {_fmt(concordance.get('pair_weighted_concordance'))} "
            f"| {_pct(metrics.get('top_1pct_positive_sum_share'))} |"
        )
    gate = payload.get("source_11e_gate") or {}
    lines += ["", "## 3. Target採用依據", ""]
    if gate.get("approval_basis") == "active_workflow_profile":
        lines += [
            f"- Active profile：`{gate.get('experiment_profile')}`。",
            f"- Continuous Target：`{gate.get('continuous_target_id')}`。",
            "- 本次只重建已採用的固定公式工件，不重跑歷史研究gate、不擬合係數。",
        ]
    else:
        lines += [
            f"- Overall ΔSpearman：`{_fmt(gate.get('overall_spearman_delta'))}`。",
            f"- PASS ΔSpearman：`{_fmt(gate.get('pass_spearman_delta'))}`。",
            f"- Decile spread增量：`{_fmt(gate.get('decile_spread_delta'))}`R。",
            "- 以上只作新版本公式的先前研究依據；未參與本輪Selection分布、rankability或任何參數擬合。",
        ]
    lines += [
        "",
        "## 4. 本輪邊界",
        "",
        "- 不建立experiment profile、不訓練、不選epoch、不寫checkpoint。",
        "- 不產生runtime scores、不設定threshold、不覆蓋11A或9A工件。",
        "- 下一步只有在Inner Train與Validation均維持足夠非tie、同日rankability與穩定分布時，才可規劃Selection-only訓練契約。",
        "- OOS仍保留給模型凍結後的迭代研究比較，不在11F報表中預先評估。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    summary, indexed_features, _context, labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(args.allow_stale_source),
    )
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    group_count = int(event_group_index.max()) + 1 if len(event_group_index) else 0
    source_manifest, source_arrays = load_validated_continuous_target_component_arrays(
        PROJECT_ROOT,
        args.filter_id,
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
        raise ValueError("11F沒有可用group target")

    if bool(args.approved_workflow_rebuild):
        source_11e_gate = _approved_workflow_rebuild_gate(filter_id=args.filter_id)
    else:
        report_11e, report_11e_path = _validated_11e_report(
            filter_id=args.filter_id,
            ranker_profile=args.ranker_profile,
        )
        actual_11e = report_11e.get("actual_trades") or {}
        correlations_11e = actual_11e.get("correlations") or {}
        deltas_11e = actual_11e.get("deltas") or {}
        pass_corr_11e = (
            (((actual_11e.get("by_label") or {}).get("pass") or {}).get("correlations") or {})
        )
        source_11e_gate = {
            "approval_basis": "validated_11e_research_gate",
            "report_path": str(report_11e_path),
            "report_sha256": _sha256_file(report_11e_path),
            "overall_original_spearman": correlations_11e.get("original_target_vs_realized_r"),
            "overall_no_time_spearman": correlations_11e.get("no_time_target_vs_realized_r"),
            "overall_spearman_delta": deltas_11e.get("spearman_no_time_minus_original"),
            "pass_original_spearman": pass_corr_11e.get("original_target_vs_realized_r"),
            "pass_no_time_spearman": pass_corr_11e.get("no_time_target_vs_realized_r"),
            "pass_spearman_delta": (
                float(
                    pass_corr_11e["no_time_target_vs_realized_r"]
                    - pass_corr_11e["original_target_vs_realized_r"]
                )
                if pass_corr_11e.get("no_time_target_vs_realized_r") is not None
                and pass_corr_11e.get("original_target_vs_realized_r") is not None
                else None
            ),
            "decile_spread_delta": deltas_11e.get("decile_spread_no_time_minus_original"),
        }

    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=_source_data_end(summary, events),
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
    group_frame = _collapse_group_frame(
        events,
        event_group_index,
        frame_targets,
        split_group_indices,
    )
    split_metrics, daily_frame = _selection_metrics(group_frame)

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": args.filter_id,
        "target_contract": target_contract,
        "source_11a_target": {
            "target_id": STRATEGY_ALIGNED_TARGET_ID,
            "manifest_generated_at_utc": source_manifest.get("generated_at_utc"),
        },
        "source_11e_gate": source_11e_gate,
        "rebuild_mode": (
            "approved_active_workflow"
            if bool(args.approved_workflow_rebuild)
            else "research_gate_validation"
        ),
        "dataset": {
            "dataset_profile": summary.get("dataset"),
            "event_count": int(summary.get("event_count", len(events))),
            "feature_group_count": int(summary.get("feature_group_count", group_count)),
            "source_data_date_range": summary.get("source_data_date_range"),
            "label_policy": summary.get("label_policy"),
        },
        "split_contract": split_report,
        "evaluated_splits": list(SELECTION_SPLITS),
        "oos_evaluated": False,
        "split_metrics": split_metrics,
        "interpretation_contract": {
            "research_only": True,
            "training_performed": False,
            "selection_only_audit": True,
            "oos_rows_scores_labels_or_statistics_evaluated": False,
            "prior_iterative_oos_hypothesis_informed": True,
            "no_oos_fitted_coefficient": True,
            "no_model_profile_checkpoint_threshold_or_runtime_authorized": True,
            "approved_target_rebuild_without_research_gate": bool(
                args.approved_workflow_rebuild
            ),
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }

    target_dir = resolve_continuous_target_dir(
        PROJECT_ROOT,
        args.filter_id,
        target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    array_paths = _artifact_paths(target_dir)
    for name, path in array_paths.items():
        save_npy_atomic(path, np.asarray(targets[name]))
    audit_json_path = target_dir / TARGET_AUDIT_JSON_FILENAME
    audit_markdown_path = target_dir / TARGET_AUDIT_MARKDOWN_FILENAME
    daily_csv_path = target_dir / TARGET_DAILY_CSV_FILENAME
    write_json(audit_json_path, payload)
    audit_markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    daily_frame.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")

    source_manifest_path = resolve_continuous_target_dir(
        PROJECT_ROOT,
        args.filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
    ) / TARGET_MANIFEST_FILENAME
    manifest = {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": args.filter_id,
        "target_contract": target_contract,
        "group_count": group_count,
        "valid_group_count": valid_count,
        "invalid_group_count": int(group_count - valid_count),
        "dataset_policy": summary.get("policy"),
        "split_policy": outer_policy,
        "split_report": split_report,
        "evaluated_splits": list(SELECTION_SPLITS),
        "oos_evaluated": False,
        "source_target": {
            "target_id": STRATEGY_ALIGNED_TARGET_ID,
            "manifest": build_file_manifest(source_manifest_path),
        },
        "source_11e": source_11e_gate,
        "rebuild_mode": (
            "approved_active_workflow"
            if bool(args.approved_workflow_rebuild)
            else "research_gate_validation"
        ),
        "artifacts": {name: build_file_manifest(path) for name, path in array_paths.items()},
        "audit_outputs": {
            "json": build_file_manifest(audit_json_path),
            "markdown": build_file_manifest(audit_markdown_path),
            "daily_csv": build_file_manifest(daily_csv_path),
        },
        "training_performed": False,
        "runtime_eligible": False,
        "research_only": True,
    }
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    write_json(manifest_path, manifest)

    print(
        "No-time continuous target workflow重建完成"
        if bool(args.approved_workflow_rebuild)
        else "11F no-time target selection-only audit完成"
    )
    print(
        f"target={STRATEGY_ALIGNED_NO_TIME_TARGET_ID} groups={group_count:,} valid={valid_count:,}"
    )
    for split_name in SELECTION_SPLITS:
        metrics = split_metrics[split_name]
        rankability = metrics.get("same_day_rankability") or {}
        print(
            f"- {split_name:<11} groups={int(metrics.get('group_count', 0)):,} "
            f"mean={_fmt(metrics.get('mean'))} binary_auc={_fmt(metrics.get('binary_label_auc'))} "
            f"rankable_dates={_pct(rankability.get('rankable_date_rate'))} "
            f"source_rho={_fmt(metrics.get('source_vs_no_time_spearman'))}"
        )
    print("- OOS: not evaluated")
    print(f"已輸出: {manifest_path}")
    print(f"已輸出: {audit_markdown_path}")
    return 0


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "_approved_workflow_rebuild_gate",
    "main",
    "parse_args",
    "render_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
