"""11H research-only attribution of 11G PASS scores to opportunity realization gap."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.breakout_quality import (
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from filters.breakout_quality.continuous_target import (
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_MANIFEST_FILENAME,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    load_validated_continuous_target_component_arrays,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import LABEL_PASS
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from core.console_report import print_artifact_paths
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from services.breakout_quality.ranker_training import (
    RANKER_REPORT_JSON_FILENAME,
    RANKER_SCORE_FILENAME,
    calculate_spearman as _spearman,
)

from tools.audit.primitives import sha256_file as _sha256_file

from tools.audit.breakout_quality.artifact_primitives import (
    read_json as _read_json,
    resolve_ranker_dir as _ranker_dir,
)

AUDIT_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "11H PASS-only Realization-gap Attribution Audit"
AUDIT_DIRNAME = "pass_realization_gap_audit"
AUDIT_JSON_FILENAME = "pass_realization_gap_audit.json"
AUDIT_MARKDOWN_FILENAME = "pass_realization_gap_audit.md"
OOS_PASS_ATTRIBUTION_FILENAME = "oos_pass_realization_gap_attribution.csv"
ACTUAL_PASS_ATTRIBUTION_FILENAME = "actual_pass_realization_gap_attribution.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11H research-only PASS realization-gap attribution；"
            "拆解11G Score對No-time Target成分與實際策略R實現落差的關係"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--ranker-profile",
        default=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        choices=(STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,),
    )
    return parser.parse_args(argv)








def _artifact_hash(record: Any, *, name: str) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"11H來源report缺少artifact: {name}")
    value = str(record.get("sha256") or "").strip().lower()
    if not value:
        raise ValueError(f"11H來源report的artifact缺少SHA256: {name}")
    return value


def _validated_ranker_inputs(
    *,
    filter_id: str,
    ranker_profile: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    ranker_dir = _ranker_dir(filter_id, ranker_profile)
    report_path = ranker_dir / RANKER_REPORT_JSON_FILENAME
    score_path = ranker_dir / RANKER_SCORE_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(
            f"11H需要11G report: {report_path}；請先執行PASS-conditional continuous ranker"
        )
    if not score_path.is_file():
        raise FileNotFoundError(f"11H需要11G scores: {score_path}")
    report = _read_json(report_path)
    if not str(report.get("status") or "").startswith("RESULT_AVAILABLE"):
        raise ValueError("11H只接受已完成的11G結果")
    if str(report.get("experiment_profile") or "") != ranker_profile:
        raise ValueError("11H 11G report experiment_profile不一致")
    training = report.get("training") or {}
    if str(training.get("training_label_scope") or "") != TRAINING_LABEL_SCOPE_PASS_ONLY:
        raise ValueError("11H只接受PASS-only 11G report")
    target_manifest = report.get("target_manifest") or {}
    if str((target_manifest.get("target_contract") or {}).get("target_id") or "") != STRATEGY_ALIGNED_NO_TIME_TARGET_ID:
        raise ValueError("11H只接受11F No-time Target")
    runtime = report.get("runtime_eligibility") or {}
    if bool(runtime.get("eligible")) or str(runtime.get("scope") or "") != "research_only":
        raise ValueError("11H預期11G維持research-only")
    expected_score_hash = _artifact_hash(
        (report.get("artifacts") or {}).get("scores"),
        name="scores",
    )
    if _sha256_file(score_path).lower() != expected_score_hash:
        raise ValueError("11H偵測到11G scores SHA256不一致")

    scores = pd.read_csv(score_path, encoding="utf-8-sig")
    required = {
        "ticker", "date", "group_index", "label", "split",
        "target_raw_r", "model_score", "in_training_label_scope",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"11H 11G scores缺少欄位: {missing}")
    scores["ticker"] = scores["ticker"].astype(str)
    scores["date"] = pd.to_datetime(scores["date"], errors="raise").dt.strftime("%Y-%m-%d")
    for column in ("group_index", "label"):
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.int64)
    for column in ("target_raw_r", "model_score"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scope_values = scores["in_training_label_scope"]
    if scope_values.dtype == bool:
        scores["in_training_label_scope"] = scope_values
    else:
        normalized_scope = scope_values.astype(str).str.strip().str.lower()
        if not bool(normalized_scope.isin({"true", "false"}).all()):
            raise ValueError("11H in_training_label_scope必須是布林值")
        scores["in_training_label_scope"] = normalized_scope.map({"true": True, "false": False})
    if bool(scores["group_index"].duplicated().any()):
        raise ValueError("11H 11G scores每個group_index必須唯一")
    return scores, report, report_path, score_path


def _validated_actual_trade_source(filter_id: str) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    target_dir = resolve_continuous_target_dir(
        PROJECT_ROOT,
        filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
    )
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    trade_path = target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"11H需要11A manifest: {manifest_path}")
    if not trade_path.is_file():
        raise FileNotFoundError(f"11H需要11A trade matches: {trade_path}")
    manifest = _read_json(manifest_path)
    record = ((manifest.get("audit_outputs") or {}).get("trade_matches_csv") or {})
    expected_hash = _artifact_hash(record, name="trade_matches_csv")
    if _sha256_file(trade_path).lower() != expected_hash:
        raise ValueError("11H偵測到11A trade matches SHA256不一致")
    actual = pd.read_csv(trade_path, encoding="utf-8-sig")
    required = {"ticker", "target_date", "r_multiple"}
    missing = sorted(required - set(actual.columns))
    if missing:
        raise ValueError(f"11H actual trade matches缺少欄位: {missing}")
    return actual, trade_path, manifest


def attach_no_time_components(
    frame: pd.DataFrame,
    *,
    arrays: dict[str, np.ndarray],
    risk_budget_return: float,
) -> pd.DataFrame:
    required = {"group_index", "label", "target_raw_r", "model_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"11H attribution frame缺少欄位: {missing}")
    risk_budget = float(risk_budget_return)
    if not math.isfinite(risk_budget) or risk_budget <= 0.0:
        raise ValueError("11H risk_budget_return必須是有限正數")
    work = frame.copy()
    work["group_index"] = pd.to_numeric(work["group_index"], errors="raise").astype(np.int64)
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(np.int64)
    work["target_raw_r"] = pd.to_numeric(work["target_raw_r"], errors="coerce")
    work["model_score"] = pd.to_numeric(work["model_score"], errors="coerce")
    indexes = work["group_index"].to_numpy(dtype=np.int64)
    group_count = len(arrays["target_raw_r"])
    if bool(np.any(indexes < 0)) or bool(np.any(indexes >= group_count)):
        raise ValueError("11H group_index超出No-time target arrays範圍")
    valid = np.asarray(arrays["valid_mask"], dtype=bool)[indexes]
    if not bool(valid.all()):
        raise ValueError("11H attribution rows包含invalid No-time target group")
    favorable = np.asarray(arrays["favorable_return"], dtype=np.float64)[indexes] / risk_budget
    adverse = np.asarray(arrays["adverse_return_to_peak"], dtype=np.float64)[indexes] / risk_budget
    target = np.asarray(arrays["target_raw_r"], dtype=np.float64)[indexes]
    if not bool(np.isfinite(favorable).all() and np.isfinite(adverse).all() and np.isfinite(target).all()):
        raise ValueError("11H No-time target component含非有限值")
    reconstructed = favorable - adverse
    if not bool(np.allclose(target, reconstructed, rtol=0.0, atol=1e-5)):
        raise ValueError("11H No-time target無法由favorable-adverse逐筆重建")
    source_target = work["target_raw_r"].to_numpy(dtype=np.float64)
    if not bool(np.allclose(source_target, target, rtol=0.0, atol=1e-5)):
        raise ValueError("11H scores target與No-time arrays不一致")
    work["favorable_r"] = favorable
    work["adverse_r"] = adverse
    return work


def _rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)


def partial_spearman(
    x: Iterable[float],
    y: Iterable[float],
    controls: Iterable[Iterable[float]],
) -> float | None:
    x_values = np.asarray(list(x), dtype=np.float64)
    y_values = np.asarray(list(y), dtype=np.float64)
    control_values = [np.asarray(list(value), dtype=np.float64) for value in controls]
    if any(len(value) != len(x_values) for value in [y_values, *control_values]):
        raise ValueError("11H partial Spearman欄位長度不一致")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    for value in control_values:
        finite &= np.isfinite(value)
    if int(finite.sum()) < max(4, len(control_values) + 3):
        return None
    x_rank = _rank_average(x_values[finite])
    y_rank = _rank_average(y_values[finite])
    ranked_controls = [_rank_average(value[finite]) for value in control_values]
    design = np.column_stack([np.ones(len(x_rank), dtype=np.float64), *ranked_controls])
    beta_x, *_ = np.linalg.lstsq(design, x_rank, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y_rank, rcond=None)
    residual_x = x_rank - design @ beta_x
    residual_y = y_rank - design @ beta_y
    if float(np.std(residual_x)) <= 0.0 or float(np.std(residual_y)) <= 0.0:
        return None
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _decile_summary(frame: pd.DataFrame, *, sort_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"count_per_decile": 0, "top": {}, "bottom": {}}
    ordered = frame.sort_values(sort_column, kind="mergesort")
    count = max(1, int(math.ceil(len(ordered) * 0.10)))
    columns = (
        "model_score", "target_raw_r", "favorable_r", "adverse_r",
        "r_multiple", "realization_gap_r", "favorable_capture_ratio",
    )

    def summarize(part: pd.DataFrame) -> dict[str, Any]:
        result = {"row_count": int(len(part))}
        for column in columns:
            values = pd.to_numeric(part[column], errors="coerce")
            finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
            result[f"mean_{column}"] = float(finite.mean()) if len(finite) else None
            result[f"median_{column}"] = float(finite.median()) if len(finite) else None
        return result

    return {
        "count_per_decile": int(count),
        "top": summarize(ordered.tail(count)),
        "bottom": summarize(ordered.head(count)),
    }


def realization_gap_metrics(frame: pd.DataFrame, *, include_realized_r: bool) -> dict[str, Any]:
    required = {"model_score", "target_raw_r", "favorable_r", "adverse_r"}
    if include_realized_r:
        required.add("r_multiple")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"11H metrics frame缺少欄位: {missing}")
    work = frame.copy()
    numeric_columns = sorted(required)
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    finite = np.ones(len(work), dtype=bool)
    for column in numeric_columns:
        finite &= np.isfinite(work[column].to_numpy(dtype=np.float64))
    work = work[finite].copy()
    result: dict[str, Any] = {
        "row_count": int(len(work)),
        "correlations": {
            "score_vs_target": _spearman(work["model_score"], work["target_raw_r"]),
            "score_vs_favorable_r": _spearman(work["model_score"], work["favorable_r"]),
            "score_vs_adverse_r": _spearman(work["model_score"], work["adverse_r"]),
        },
    }
    if not include_realized_r:
        return result
    work["realization_gap_r"] = work["target_raw_r"] - work["r_multiple"]
    favorable = work["favorable_r"].to_numpy(dtype=np.float64)
    capture = np.full(len(work), np.nan, dtype=np.float64)
    positive = favorable > 1e-12
    capture[positive] = work.loc[positive, "r_multiple"].to_numpy(dtype=np.float64) / favorable[positive]
    work["favorable_capture_ratio"] = capture
    corr = result["correlations"]
    corr.update({
        "target_vs_realized_r": _spearman(work["target_raw_r"], work["r_multiple"]),
        "favorable_r_vs_realized_r": _spearman(work["favorable_r"], work["r_multiple"]),
        "adverse_r_vs_realized_r": _spearman(work["adverse_r"], work["r_multiple"]),
        "score_vs_realized_r": _spearman(work["model_score"], work["r_multiple"]),
        "score_vs_realization_gap_r": _spearman(work["model_score"], work["realization_gap_r"]),
        "score_vs_favorable_capture_ratio": _spearman(work["model_score"], work["favorable_capture_ratio"]),
        "target_vs_realization_gap_r": _spearman(work["target_raw_r"], work["realization_gap_r"]),
        "target_vs_favorable_capture_ratio": _spearman(work["target_raw_r"], work["favorable_capture_ratio"]),
    })
    result["partial_correlations"] = {
        "score_vs_realized_r_controlling_target": partial_spearman(
            work["model_score"], work["r_multiple"], [work["target_raw_r"]]
        ),
        "score_vs_realized_r_controlling_favorable_and_adverse": partial_spearman(
            work["model_score"], work["r_multiple"], [work["favorable_r"], work["adverse_r"]]
        ),
    }
    result["score_deciles"] = _decile_summary(work, sort_column="model_score")
    result["target_deciles"] = _decile_summary(work, sort_column="target_raw_r")
    result["frame"] = work
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _decile_table_row(name: str, record: dict[str, Any]) -> str:
    return (
        f"| {name} | {int(record.get('row_count', 0)):,} "
        f"| {_fmt(record.get('mean_target_raw_r'))} "
        f"| {_fmt(record.get('mean_favorable_r'))} "
        f"| {_fmt(record.get('mean_adverse_r'))} "
        f"| {_fmt(record.get('mean_r_multiple'))} "
        f"| {_fmt(record.get('mean_realization_gap_r'))} "
        f"| {_fmt(record.get('mean_favorable_capture_ratio'))} |"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    oos = payload["oos_pass"]
    actual = payload["actual_pass"]
    oos_corr = oos.get("correlations") or {}
    actual_corr = actual.get("correlations") or {}
    partial = actual.get("partial_correlations") or {}
    score_deciles = actual.get("score_deciles") or {}
    target_deciles = actual.get("target_deciles") or {}
    lines = [
        "# 11H PASS-only Realization-gap Attribution Audit",
        "",
        "- Runtime：research-only；本輪不訓練、不選epoch、不改Target、loss、sampling或runtime。",
        "- 目的：確認11G在actual PASS trades中學到的No-time Target成分，是否對應可被策略實現的R，或只是更大的未實現機會落差。",
        "",
        "## 1. OOS PASS groups",
        "",
        f"- Rows：`{int(oos.get('row_count', 0)):,}`。",
        f"- Score↔Target：`{_fmt(oos_corr.get('score_vs_target'))}`。",
        f"- Score↔Favorable R：`{_fmt(oos_corr.get('score_vs_favorable_r'))}`。",
        f"- Score↔Adverse R：`{_fmt(oos_corr.get('score_vs_adverse_r'))}`。",
        "",
        "## 2. Actual PASS trades",
        "",
        f"- Rows：`{int(actual.get('row_count', 0)):,}`。",
        f"- Score↔Target：`{_fmt(actual_corr.get('score_vs_target'))}`；Target↔R：`{_fmt(actual_corr.get('target_vs_realized_r'))}`；Score↔R：`{_fmt(actual_corr.get('score_vs_realized_r'))}`。",
        f"- Score↔Favorable R：`{_fmt(actual_corr.get('score_vs_favorable_r'))}`；Score↔Adverse R：`{_fmt(actual_corr.get('score_vs_adverse_r'))}`。",
        f"- Favorable R↔realized R：`{_fmt(actual_corr.get('favorable_r_vs_realized_r'))}`；Adverse R↔realized R：`{_fmt(actual_corr.get('adverse_r_vs_realized_r'))}`。",
        f"- Score↔realization gap（Target−R）：`{_fmt(actual_corr.get('score_vs_realization_gap_r'))}`。",
        f"- Score↔favorable capture ratio（R÷Favorable）：`{_fmt(actual_corr.get('score_vs_favorable_capture_ratio'))}`。",
        f"- Partial Score↔R｜控制Target：`{_fmt(partial.get('score_vs_realized_r_controlling_target'))}`。",
        f"- Partial Score↔R｜控制Favorable與Adverse：`{_fmt(partial.get('score_vs_realized_r_controlling_favorable_and_adverse'))}`。",
        "",
        "## 3. Score decile attribution",
        "",
        "| Score decile | Rows | Target R | Favorable R | Adverse R | Realized R | Target−R gap | R÷Favorable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _decile_table_row("Top 10%", score_deciles.get("top") or {}),
        _decile_table_row("Bottom 10%", score_deciles.get("bottom") or {}),
        "",
        "## 4. Target decile reference",
        "",
        "| Target decile | Rows | Target R | Favorable R | Adverse R | Realized R | Target−R gap | R÷Favorable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _decile_table_row("Top 10%", target_deciles.get("top") or {}),
        _decile_table_row("Bottom 10%", target_deciles.get("bottom") or {}),
        "",
        "## 5. 判定邊界",
        "",
        "- 若Score↔realization gap明顯為正且Score↔capture ratio為負，代表模型偏好潛在機會但策略無法實現；下一步只能建立Selection historical replay的strategy-realization target audit，不再微調MFE型Target。",
        "- 若Score主要學到Adverse而非Favorable，下一步只允許固定Favorable-only消融，不重跑11G超參數。",
        "- 若Score已學到Favorable且Favorable↔R為正，但Score↔R仍負且gap／capture不支持實現率解釋，先檢查trade mapping與策略交互，不直接新增模型。",
        "- OOS為迭代研究證據；本輪所有統計只作失敗歸因，不參與任何擬合。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    scores, ranker_report, ranker_report_path, score_path = _validated_ranker_inputs(
        filter_id=args.filter_id,
        ranker_profile=args.ranker_profile,
    )
    target_manifest, arrays = load_validated_continuous_target_component_arrays(
        PROJECT_ROOT,
        args.filter_id,
        target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    )
    target_contract = target_manifest.get("target_contract") or {}
    risk_budget = float(target_contract.get("risk_budget_return", math.nan))

    oos_pass = scores[
        (scores["split"].astype(str) == "oos")
        & (scores["label"] == LABEL_PASS)
        & scores["in_training_label_scope"]
    ].copy()
    if oos_pass.empty:
        raise ValueError("11H找不到OOS PASS score rows")
    oos_pass = attach_no_time_components(
        oos_pass,
        arrays=arrays,
        risk_budget_return=risk_budget,
    )
    oos_metrics = realization_gap_metrics(oos_pass, include_realized_r=False)

    actual_raw, actual_path, actual_manifest = _validated_actual_trade_source(args.filter_id)
    actual_work = actual_raw.copy()
    actual_work["ticker"] = actual_work["ticker"].astype(str)
    actual_work["target_date"] = pd.to_datetime(
        actual_work["target_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    lookup = oos_pass.rename(columns={"date": "target_date"})[
        ["ticker", "target_date", "group_index", "label", "target_raw_r", "model_score"]
    ]
    for column in ("group_index", "label", "target_raw_r", "model_score"):
        if column in actual_work.columns:
            actual_work = actual_work.drop(columns=[column])
    actual_work = actual_work.merge(
        lookup,
        how="left",
        on=["ticker", "target_date"],
        validate="many_to_one",
    )
    actual_work["r_multiple"] = pd.to_numeric(actual_work["r_multiple"], errors="coerce")
    valid = (
        np.isfinite(actual_work["r_multiple"].to_numpy(dtype=np.float64))
        & np.isfinite(pd.to_numeric(actual_work["model_score"], errors="coerce").to_numpy(dtype=np.float64))
        & (pd.to_numeric(actual_work["label"], errors="coerce").to_numpy(dtype=np.float64) == LABEL_PASS)
    )
    actual_pass = actual_work[valid].copy()
    actual_pass = attach_no_time_components(
        actual_pass,
        arrays=arrays,
        risk_budget_return=risk_budget,
    )
    actual_metrics = realization_gap_metrics(actual_pass, include_realized_r=True)
    actual_frame = actual_metrics.pop("frame")

    expected_pass_count = int(
        (((ranker_report.get("trade_alignment") or {}).get("label_conditional") or {}).get("PASS") or {}).get("matched_trade_count", -1)
    )
    if expected_pass_count >= 0 and int(len(actual_frame)) != expected_pass_count:
        raise ValueError(
            f"11H actual PASS配對數與11G report不一致: {len(actual_frame)} != {expected_pass_count}"
        )

    output_dir = _ranker_dir(args.filter_id, args.ranker_profile) / AUDIT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    oos_path = output_dir / OOS_PASS_ATTRIBUTION_FILENAME
    actual_out_path = output_dir / ACTUAL_PASS_ATTRIBUTION_FILENAME
    json_path = output_dir / AUDIT_JSON_FILENAME
    markdown_path = output_dir / AUDIT_MARKDOWN_FILENAME
    oos_pass.to_csv(oos_path, index=False, encoding="utf-8-sig")
    actual_frame.to_csv(actual_out_path, index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": args.filter_id,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "ranker_profile": args.ranker_profile,
        "continuous_target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        "oos_pass": oos_metrics,
        "actual_pass": actual_metrics,
        "source_11g": {
            "report_path": str(ranker_report_path),
            "report_sha256": _sha256_file(ranker_report_path),
            "score_path": str(score_path),
            "score_sha256": _sha256_file(score_path),
            "status": ranker_report.get("status"),
            "selected_epoch": (ranker_report.get("training") or {}).get("selected_epoch"),
            "oos_pass_score_vs_target": ((ranker_report.get("split_metrics") or {}).get("oos") or {}).get("global_spearman_vs_raw_target"),
            "actual_pass_score_vs_target": ((((ranker_report.get("trade_alignment") or {}).get("label_conditional") or {}).get("PASS") or {}).get("spearman_model_score_vs_target")),
            "actual_pass_score_vs_realized_r": ((((ranker_report.get("trade_alignment") or {}).get("label_conditional") or {}).get("PASS") or {}).get("spearman_model_score_vs_r_multiple")),
        },
        "source_actual_trades": {
            "path": str(actual_path),
            "sha256": _sha256_file(actual_path),
            "manifest_target_id": ((actual_manifest.get("target_contract") or {}).get("target_id")),
        },
        "artifacts": {
            "oos_pass_attribution": {"path": str(oos_path), "sha256": _sha256_file(oos_path)},
            "actual_pass_attribution": {"path": str(actual_out_path), "sha256": _sha256_file(actual_out_path)},
        },
        "interpretation_contract": {
            "research_only": True,
            "training_performed": False,
            "oos_iterative_research_evidence_only": True,
            "no_target_loss_sampling_epoch_threshold_or_runtime_change": True,
            "new_model_not_authorized_until_result_review": True,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    write_json(json_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    oos_corr = oos_metrics.get("correlations") or {}
    actual_corr = actual_metrics.get("correlations") or {}
    partial = actual_metrics.get("partial_correlations") or {}
    print("11H pass realization-gap attribution audit完成")
    print(f"oos_pass={oos_metrics.get('row_count', 0):,} actual_pass={actual_metrics.get('row_count', 0):,}")
    print(
        "- OOS PASS Score↔components: "
        f"target={_fmt(oos_corr.get('score_vs_target'))} "
        f"favorable={_fmt(oos_corr.get('score_vs_favorable_r'))} "
        f"adverse={_fmt(oos_corr.get('score_vs_adverse_r'))}"
    )
    print(
        "- Actual PASS realization: "
        f"score↔R={_fmt(actual_corr.get('score_vs_realized_r'))} "
        f"score↔gap={_fmt(actual_corr.get('score_vs_realization_gap_r'))} "
        f"score↔capture={_fmt(actual_corr.get('score_vs_favorable_capture_ratio'))} "
        f"partial(score,R|target)={_fmt(partial.get('score_vs_realized_r_controlling_target'))}"
    )
    print_artifact_paths(
        [("Markdown 報表", markdown_path), ("完整 JSON", json_path)],
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = [
    "ACTUAL_PASS_ATTRIBUTION_FILENAME",
    "AUDIT_DIRNAME",
    "AUDIT_JSON_FILENAME",
    "AUDIT_MARKDOWN_FILENAME",
    "EXPERIMENT_NAME",
    "OOS_PASS_ATTRIBUTION_FILENAME",
    "attach_no_time_components",
    "parse_args",
    "partial_spearman",
    "realization_gap_metrics",
    "render_markdown",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
