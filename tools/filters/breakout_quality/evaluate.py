"""Evaluate breakout-quality research scores on canonical inner-validation or outer-OOS rows."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from config.breakout_quality_policy import BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
from filters.breakout_quality.artifacts import (
    compute_file_sha256,
    load_model_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    INNER_SPLIT_TRAIN,
    INNER_SPLIT_VALIDATION,
    LABEL_PASS,
    LABEL_REJECT,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_SELECTION,
    RUNTIME_SCOPE_RESEARCH,
    SCORE_COLUMN,
)
from filters.breakout_quality.paths import (
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
)
from filters.breakout_quality.splits import KEY_COLUMNS, normalize_split_assignment_keys
from tools.filters.breakout_quality.common import (
    event_group_summary,
    group_size_weights,
    read_breakout_quality_csv,
    read_json,
)

EVALUATION_SPLIT_VALIDATION = "validation"
EVALUATION_SPLIT_TRAIN = "train"
EVALUATION_SPLIT_OOS = "oos"
EVALUATION_SPLIT_ALL = "all"
EVALUATION_SPLIT_ALIASES = {
    "validation": EVALUATION_SPLIT_VALIDATION,
    "inner_validation": EVALUATION_SPLIT_VALIDATION,
    "train": EVALUATION_SPLIT_TRAIN,
    "inner_train": EVALUATION_SPLIT_TRAIN,
    "oos": EVALUATION_SPLIT_OOS,
    "all": EVALUATION_SPLIT_ALL,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="評估 breakout quality event-level 區分能力")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--threshold", type=float, default=BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)
    parser.add_argument(
        "--split",
        choices=tuple(EVALUATION_SPLIT_ALIASES),
        default=EVALUATION_SPLIT_VALIDATION,
        help=(
            "validation/inner_validation 用於 threshold 選擇；"
            "oos 僅在 threshold 鎖定後評估；train/all 只供診斷"
        ),
    )
    parser.add_argument(
        "--score-path",
        default=None,
        help=(
            "研究 score table；省略時讀 outputs/filters/breakout_quality/<filter_id>/research_scores.csv；"
            "自訂路徑需在同目錄提供 research_scores_manifest.json"
        ),
    )
    return parser.parse_args(argv)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return round(float(values.mean()), 6)
    return round(float(np.average(values, weights=weights)), 6)


def _sum_weight(mask: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(mask.sum())
    return float(weights[mask].sum())


def _metrics(df: pd.DataFrame, *, threshold: float, group_weighted: bool) -> dict:
    valid = df[df["label"].isin([LABEL_PASS, LABEL_REJECT])].copy()
    if valid.empty:
        return {"row_count": 0}
    weights = group_size_weights(valid, range(len(valid))) if group_weighted else None
    score = valid[SCORE_COLUMN].to_numpy(dtype=np.float64)
    pred_pass = score >= float(threshold)
    truth_pass = valid["label"].to_numpy(dtype=np.int64) == LABEL_PASS
    pred_reject = ~pred_pass
    truth_reject = ~truth_pass

    tp = _sum_weight(pred_pass & truth_pass, weights)
    fp = _sum_weight(pred_pass & truth_reject, weights)
    tn = _sum_weight(pred_reject & truth_reject, weights)
    fn = _sum_weight(pred_reject & truth_pass, weights)
    total = tp + fp + tn + fn
    actual_pass = tp + fn
    accepted = tp + fp

    def _ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return round(float(numerator / denominator), 6)

    base_pass_rate = _ratio(actual_pass, total)
    precision = _ratio(tp, accepted)
    return {
        "row_count": int(len(valid)),
        "group_count": int(valid[["ticker", "date"]].drop_duplicates().shape[0]),
        "weight_sum": round(total, 6),
        "threshold": float(threshold),
        "acceptance_rate": _ratio(accepted, total),
        "pass_precision": precision,
        "pass_recall": _ratio(tp, actual_pass),
        "reject_specificity": _ratio(tn, tn + fp),
        "false_rejection_rate": _ratio(fn, actual_pass),
        "accuracy": _ratio(tp + tn, total),
        "base_pass_rate": base_pass_rate,
        "all_pass_baseline_accuracy": base_pass_rate,
        "precision_lift_vs_all_pass": (
            round(float(precision / base_pass_rate), 6)
            if precision is not None and base_pass_rate not in {None, 0.0}
            else None
        ),
        "avg_score": (
            _weighted_mean(score.astype(np.float32), weights)
            if weights is not None
            else round(float(score.mean()), 6)
        ),
        "confusion": {
            "true_pass_pred_pass": round(tp, 6),
            "true_reject_pred_pass": round(fp, 6),
            "true_reject_pred_reject": round(tn, 6),
            "true_pass_pred_reject": round(fn, 6),
        },
    }


def _normalize_score_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_split_assignment_keys(frame)
    return normalized


def _research_manifest_path(score_path: Path, filter_id: str) -> Path:
    canonical_score = resolve_filter_research_score_path(PROJECT_ROOT, filter_id).resolve()
    if score_path.resolve() == canonical_score:
        return resolve_filter_research_manifest_path(PROJECT_ROOT, filter_id)
    return score_path.parent / "research_scores_manifest.json"


def _validate_research_contract(
    *,
    score_path: Path,
    score_frame: pd.DataFrame,
    filter_id: str,
    split_record: dict,
) -> dict:
    manifest_path = _research_manifest_path(score_path, filter_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 research score manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if str(manifest.get("filter_id") or "").strip() != str(filter_id).strip():
        raise ValueError("research score manifest.filter_id 與命令不一致")
    if str(manifest.get("scope") or "").strip() != RUNTIME_SCOPE_RESEARCH:
        raise ValueError("evaluate 只能使用 scope=research 的 score manifest")
    source_split = manifest.get("source_split_assignments")
    if not isinstance(source_split, dict) or source_split.get("sha256") != split_record.get("sha256"):
        raise ValueError("research score manifest 與 canonical split assignment 不一致")
    score_record = manifest.get("score_table")
    if not isinstance(score_record, dict):
        raise ValueError("research score manifest 缺少 score_table")
    if str(score_record.get("filename") or "").strip() != score_path.name:
        raise ValueError("research score filename 與 manifest 不一致")
    if int(score_record.get("size_bytes", -1)) != int(score_path.stat().st_size):
        raise ValueError("research score size 與 manifest 不一致")
    expected_hash = str(score_record.get("sha256") or "").strip().lower()
    if not expected_hash or expected_hash != compute_file_sha256(score_path).lower():
        raise ValueError("research score SHA256 與 manifest 不一致")
    if int(score_record.get("row_count", -1)) != len(score_frame):
        raise ValueError("research score row_count 與 manifest 不一致")
    if list(score_record.get("columns", [])) != list(score_frame.columns):
        raise ValueError("research score columns 與 manifest 不一致")
    return manifest


def _select_rows(frame: pd.DataFrame, split_name: str, outer_policy: dict) -> pd.DataFrame:
    if split_name == EVALUATION_SPLIT_VALIDATION:
        selected = frame[
            (frame["outer_split"] == OUTER_SPLIT_SELECTION)
            & (frame["inner_split"] == INNER_SPLIT_VALIDATION)
        ].copy()
    elif split_name == EVALUATION_SPLIT_TRAIN:
        selected = frame[
            (frame["outer_split"] == OUTER_SPLIT_SELECTION)
            & (frame["inner_split"] == INNER_SPLIT_TRAIN)
        ].copy()
    elif split_name == EVALUATION_SPLIT_OOS:
        selected = frame[frame["outer_split"] == OUTER_SPLIT_OOS].copy()
        if "label_eval_end_date" not in selected.columns:
            raise ValueError("OOS 評估需要 research score 包含 label_eval_end_date")
        effective_end = pd.Timestamp(str(outer_policy.get("effective_oos_end_date") or "")).normalize()
        selected = selected[
            pd.to_datetime(selected["label_eval_end_date"], errors="raise").dt.normalize() <= effective_end
        ].copy()
    else:
        selected = frame.copy()
    if selected.empty:
        raise ValueError(f"指定 split 沒有可評估資料: split={split_name}")
    return selected


def main(argv=None) -> int:
    args = parse_args(argv)
    threshold = float(args.threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold 必須介於 0 與 1")
    split_name = EVALUATION_SPLIT_ALIASES[str(args.split)]
    score_path = (
        Path(args.score_path).expanduser().resolve()
        if args.score_path
        else resolve_filter_research_score_path(PROJECT_ROOT, args.filter_id)
    )
    score_frame = read_breakout_quality_csv(score_path)
    required = {"ticker", "date", "high_len", SCORE_COLUMN, "label"}
    missing = sorted(required - set(score_frame.columns))
    if missing:
        raise ValueError(f"evaluate score table 缺少欄位: {missing}; path={score_path}")

    model_contract = load_model_artifact_contract(str(PROJECT_ROOT), str(args.filter_id))
    split_frame = load_split_assignment_frame(str(PROJECT_ROOT), str(args.filter_id))
    normalized_scores = _normalize_score_keys(score_frame)
    merged = normalized_scores.merge(
        split_frame,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    missing_split = int(merged["outer_split"].isna().sum())
    if missing_split:
        raise ValueError(
            f"research score 有 {missing_split} 列找不到 canonical split assignment；"
            "請勿混用不同 dataset/model/filter_id"
        )
    if len(merged) != len(split_frame):
        raise ValueError(
            f"research score 與 canonical split assignment 必須完整一對一覆蓋: "
            f"scores={len(merged)}, splits={len(split_frame)}"
        )
    split_record = model_contract.manifest.get("split_assignments")
    if not isinstance(split_record, dict):
        raise ValueError("model manifest 缺少 split_assignments")
    research_manifest = _validate_research_contract(
        score_path=score_path,
        score_frame=score_frame,
        filter_id=str(args.filter_id),
        split_record=split_record,
    )
    outer_policy = model_contract.manifest.get("outer_oos_policy")
    if not isinstance(outer_policy, dict):
        raise ValueError("model manifest 缺少 outer_oos_policy")
    selected = _select_rows(merged, split_name, outer_policy)
    selected_dates = pd.to_datetime(selected["date"], errors="raise")
    role = {
        EVALUATION_SPLIT_VALIDATION: "inner_validation_for_threshold_selection",
        EVALUATION_SPLIT_TRAIN: "inner_train_diagnostic_only",
        EVALUATION_SPLIT_OOS: "outer_oos_final_evaluation",
        EVALUATION_SPLIT_ALL: "mixed_research_diagnostic_only",
    }[split_name]
    metrics = {
        "filter_id": args.filter_id,
        "score_path": str(score_path),
        "split": split_name,
        "split_role": role,
        "threshold_selection_allowed": bool(split_name == EVALUATION_SPLIT_VALIDATION),
        "is_final_oos_evaluation": bool(split_name == EVALUATION_SPLIT_OOS),
        "outer_oos_policy": outer_policy,
        "research_scope": research_manifest.get("scope"),
        "selected_row_count": int(len(selected)),
        "selected_date_range": {
            "start": str(selected_dates.min().date()),
            "end": str(selected_dates.max().date()),
        },
        "threshold": threshold,
        "row_level": _metrics(selected, threshold=threshold, group_weighted=False),
        "ticker_date_group_weighted": _metrics(selected, threshold=threshold, group_weighted=True),
        "event_group_summary": event_group_summary(
            selected,
            pd.to_numeric(selected["label"], errors="raise").to_numpy(dtype=np.int64),
        ),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
