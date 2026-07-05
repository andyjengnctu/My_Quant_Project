"""Evaluate a breakout quality research score table at event level."""

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
from filters.breakout_quality.contract import DEFAULT_FILTER_ID, LABEL_PASS, LABEL_REJECT, SCORE_COLUMN
from filters.breakout_quality.paths import resolve_filter_research_score_path
from tools.filters.breakout_quality.common import event_group_summary, group_size_weights, read_breakout_quality_csv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="評估 breakout quality event-level 區分能力")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--threshold", type=float, default=BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)
    parser.add_argument(
        "--score-path",
        default=None,
        help="研究 score table；省略時讀 outputs/filters/breakout_quality/<filter_id>/research_scores.csv",
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
    payload = {
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
    return payload


def main(argv=None) -> int:
    args = parse_args(argv)
    threshold = float(args.threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold 必須介於 0 與 1")
    score_path = (
        Path(args.score_path).expanduser().resolve()
        if args.score_path
        else resolve_filter_research_score_path(PROJECT_ROOT, args.filter_id)
    )
    df = read_breakout_quality_csv(score_path)
    required = {"ticker", "date", "high_len", SCORE_COLUMN, "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"evaluate score table 缺少欄位: {missing}; path={score_path}")
    metrics = {
        "filter_id": args.filter_id,
        "score_path": str(score_path),
        "threshold": threshold,
        "row_level": _metrics(df, threshold=threshold, group_weighted=False),
        "ticker_date_group_weighted": _metrics(df, threshold=threshold, group_weighted=True),
        "event_group_summary": event_group_summary(df, df["label"].to_numpy(dtype=np.int64)),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
