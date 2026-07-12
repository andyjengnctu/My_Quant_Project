"""Evaluate fixed-threshold breakout-quality scores on canonical Selection/OOS roles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import (
    compute_file_sha256,
    load_model_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    LABEL_PASS,
    LABEL_REJECT,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_SELECTION,
    RUNTIME_SCOPE_RESEARCH,
    SCORE_COLUMN,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
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

EVALUATION_SPLIT_TRAIN = "train"
EVALUATION_SPLIT_VALIDATION = "validation"
EVALUATION_SPLIT_SELECTION = "selection"
EVALUATION_SPLIT_OOS = "oos"
EVALUATION_SPLIT_ALL = "all"
EVALUATION_SPLITS = (
    EVALUATION_SPLIT_TRAIN,
    EVALUATION_SPLIT_VALIDATION,
    EVALUATION_SPLIT_SELECTION,
    EVALUATION_SPLIT_OOS,
    EVALUATION_SPLIT_ALL,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "以事前固定 threshold 評估 breakout quality；"
            "OOS 結果不得用來回頭調整 threshold、epochs 或模型"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "可省略；若提供，必須與 train.py 寫入 manifest 的 fixed threshold 完全相同，"
            "不可用 OOS 掃描 threshold"
        ),
    )
    parser.add_argument(
        "--split",
        choices=EVALUATION_SPLITS,
        required=True,
        help=("train/validation/selection/all 只供診斷；" "oos 是最終泛化評估"),
    )
    parser.add_argument(
        "--score-path",
        default=None,
        help=(
            "研究 score table；省略時讀 "
            "outputs/filters/breakout_quality/<filter_id>/<model_architecture>/research_scores.csv；"
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


def _research_manifest_path(score_path: Path, filter_id: str) -> Path:
    canonical_score = resolve_filter_research_score_path(
        PROJECT_ROOT,
        filter_id,
    ).resolve()
    if score_path.resolve() == canonical_score:
        return resolve_filter_research_manifest_path(PROJECT_ROOT, filter_id)
    return score_path.parent / "research_scores_manifest.json"


def _validate_research_contract(
    *,
    score_path: Path,
    score_frame: pd.DataFrame,
    filter_id: str,
    split_record: dict,
    model_manifest: dict,
) -> dict:
    manifest_path = _research_manifest_path(score_path, filter_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 research score manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if str(manifest.get("filter_id") or "").strip() != str(filter_id).strip():
        raise ValueError("research score manifest.filter_id 與命令不一致")
    if str(manifest.get("scope") or "").strip() != RUNTIME_SCOPE_RESEARCH:
        raise ValueError("evaluate 只能使用 scope=research 的 score manifest")
    if manifest.get("model_architecture") != model_manifest.get("model_architecture"):
        raise ValueError("research score manifest 與 model architecture 不一致")
    if manifest.get("model_spec") != model_manifest.get("model_spec"):
        raise ValueError("research score manifest 與 model_spec 不一致")
    source_split = manifest.get("source_split_assignments")
    if (
        not isinstance(source_split, dict)
        or source_split.get("sha256") != split_record.get("sha256")
    ):
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
    if split_name == EVALUATION_SPLIT_TRAIN:
        selected = frame[
            (frame["outer_split"] == OUTER_SPLIT_SELECTION)
            & (frame["selection_role"] == SELECTION_ROLE_TRAIN)
        ].copy()
    elif split_name == EVALUATION_SPLIT_VALIDATION:
        selected = frame[
            (frame["outer_split"] == OUTER_SPLIT_SELECTION)
            & (frame["selection_role"] == SELECTION_ROLE_VALIDATION)
        ].copy()
    elif split_name == EVALUATION_SPLIT_SELECTION:
        selected = frame[
            (frame["outer_split"] == OUTER_SPLIT_SELECTION)
            & frame["selection_role"].isin(
                [
                    SELECTION_ROLE_TRAIN,
                    SELECTION_ROLE_VALIDATION,
                    SELECTION_ROLE_INNER_EMBARGO,
                ]
            )
        ].copy()
    elif split_name == EVALUATION_SPLIT_OOS:
        selected = frame[frame["outer_split"] == OUTER_SPLIT_OOS].copy()
        if "label_eval_end_date" not in selected.columns:
            raise ValueError("OOS 評估需要 research score 包含 label_eval_end_date")
        effective_end = pd.Timestamp(
            str(outer_policy.get("effective_oos_end_date") or "")
        ).normalize()
        selected = selected[
            selected["label"].isin([LABEL_PASS, LABEL_REJECT])
            & (
                pd.to_datetime(
                    selected["label_eval_end_date"],
                    errors="raise",
                ).dt.normalize()
                <= effective_end
            )
        ].copy()
    else:
        selected = frame.copy()
    if selected.empty:
        raise ValueError(f"指定 split 沒有可評估資料: split={split_name}")
    return selected


def prepare_evaluation_context(
    *,
    filter_id: str,
    threshold: float | None = None,
    score_path: str | Path | None = None,
) -> dict:
    resolved_score_path = (
        Path(score_path).expanduser().resolve()
        if score_path is not None
        else resolve_filter_research_score_path(PROJECT_ROOT, filter_id)
    )
    score_frame = read_breakout_quality_csv(resolved_score_path)
    required = {"ticker", "date", "high_len", SCORE_COLUMN, "label"}
    missing = sorted(required - set(score_frame.columns))
    if missing:
        raise ValueError(
            f"evaluate score table 缺少欄位: {missing}; path={resolved_score_path}"
        )

    model_contract = load_model_artifact_contract(str(PROJECT_ROOT), str(filter_id))
    split_frame = load_split_assignment_frame(str(PROJECT_ROOT), str(filter_id))
    fixed_threshold = float(model_contract.manifest["fixed_evaluation_threshold"])
    if threshold is not None:
        requested_threshold = float(threshold)
        if not np.isfinite(requested_threshold) or not 0.0 <= requested_threshold <= 1.0:
            raise ValueError("threshold 必須介於 0 與 1")
        if not np.isclose(requested_threshold, fixed_threshold, rtol=0.0, atol=1e-12):
            raise ValueError(
                "breakout quality 的 threshold 已在 train 前固定；"
                f"manifest={fixed_threshold}, requested={requested_threshold}"
            )

    normalized_scores = normalize_split_assignment_keys(score_frame)
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
            "research score 與 canonical split assignment 必須完整一對一覆蓋: "
            f"scores={len(merged)}, splits={len(split_frame)}"
        )

    split_record = model_contract.manifest.get("split_assignments")
    if not isinstance(split_record, dict):
        raise ValueError("model manifest 缺少 split_assignments")
    research_manifest = _validate_research_contract(
        score_path=resolved_score_path,
        score_frame=score_frame,
        filter_id=str(filter_id),
        split_record=split_record,
        model_manifest=model_contract.manifest,
    )
    outer_policy = model_contract.manifest.get("outer_oos_policy")
    if not isinstance(outer_policy, dict):
        raise ValueError("model manifest 缺少 outer_oos_policy")
    return {
        "filter_id": str(filter_id),
        "score_path": resolved_score_path,
        "merged": merged,
        "threshold": fixed_threshold,
        "model_manifest": model_contract.manifest,
        "research_manifest": research_manifest,
        "outer_oos_policy": outer_policy,
    }


def evaluate_split_from_context(context: dict, split_name: str) -> dict:
    if split_name not in EVALUATION_SPLITS:
        raise ValueError(f"不支援的 evaluation split: {split_name}")
    selected = _select_rows(
        context["merged"],
        str(split_name),
        context["outer_oos_policy"],
    )
    selected_dates = pd.to_datetime(selected["date"], errors="raise")
    role = {
        EVALUATION_SPLIT_TRAIN: "inner_train_diagnostic_only",
        EVALUATION_SPLIT_VALIDATION: "inner_validation_epoch_selection_diagnostic",
        EVALUATION_SPLIT_SELECTION: "final_refit_selection_diagnostic_only",
        EVALUATION_SPLIT_OOS: "outer_oos_final_generalization_evaluation",
        EVALUATION_SPLIT_ALL: "mixed_research_diagnostic_only",
    }[str(split_name)]
    model_manifest = context["model_manifest"]
    threshold = float(context["threshold"])
    return {
        "filter_id": context["filter_id"],
        "score_path": str(context["score_path"]),
        "split": str(split_name),
        "split_role": role,
        "threshold": threshold,
        "threshold_source": "model_manifest.fixed_evaluation_threshold",
        "threshold_selection_allowed": False,
        "epoch_selection_allowed": bool(
            split_name == EVALUATION_SPLIT_VALIDATION
            and bool(model_manifest.get("inner_validation_used", False))
        ),
        "is_final_oos_evaluation": bool(split_name == EVALUATION_SPLIT_OOS),
        "reusing_oos_for_tuning_would_change_role_to_validation": bool(
            split_name == EVALUATION_SPLIT_OOS
        ),
        "outer_oos_policy": context["outer_oos_policy"],
        "research_scope": context["research_manifest"].get("scope"),
        "selected_row_count": int(len(selected)),
        "selected_date_range": {
            "start": str(selected_dates.min().date()),
            "end": str(selected_dates.max().date()),
        },
        "row_level": _metrics(
            selected,
            threshold=threshold,
            group_weighted=False,
        ),
        "ticker_date_group_weighted": _metrics(
            selected,
            threshold=threshold,
            group_weighted=True,
        ),
        "event_group_summary": event_group_summary(
            selected,
            pd.to_numeric(selected["label"], errors="raise").to_numpy(dtype=np.int64),
        ),
    }


def evaluate_splits(
    *,
    filter_id: str,
    splits: Iterable[str],
    threshold: float | None = None,
    score_path: str | Path | None = None,
) -> tuple[dict[str, dict], dict]:
    context = prepare_evaluation_context(
        filter_id=filter_id,
        threshold=threshold,
        score_path=score_path,
    )
    results = {
        split_name: evaluate_split_from_context(context, split_name)
        for split_name in splits
    }
    return results, context


def evaluate_breakout_quality(
    *,
    filter_id: str,
    split: str,
    threshold: float | None = None,
    score_path: str | Path | None = None,
) -> dict:
    results, _context = evaluate_splits(
        filter_id=filter_id,
        splits=[split],
        threshold=threshold,
        score_path=score_path,
    )
    return results[split]


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics = evaluate_breakout_quality(
        filter_id=str(args.filter_id),
        split=str(args.split),
        threshold=args.threshold,
        score_path=args.score_path,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "EVALUATION_SPLITS",
    "EVALUATION_SPLIT_ALL",
    "EVALUATION_SPLIT_OOS",
    "EVALUATION_SPLIT_SELECTION",
    "EVALUATION_SPLIT_TRAIN",
    "EVALUATION_SPLIT_VALIDATION",
    "evaluate_breakout_quality",
    "evaluate_split_from_context",
    "evaluate_splits",
    "prepare_evaluation_context",
]


if __name__ == "__main__":
    raise SystemExit(main())
