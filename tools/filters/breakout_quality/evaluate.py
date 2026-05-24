"""Evaluate breakout quality score table at event level."""

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

from filters.breakout_quality.contract import DEFAULT_FILTER_ID, LABEL_PASS, LABEL_REJECT
from tools.filters.breakout_quality.common import event_group_summary, group_size_weights, scores_csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="評估 breakout quality event-level 區分能力")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    return parser.parse_args(argv)


def _safe_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return round(float(values.mean()), 6)
    return round(float(np.average(values, weights=weights)), 6)


def _bucket_metrics(group: pd.DataFrame, weights: np.ndarray | None) -> dict:
    count = int(len(group))
    if count == 0:
        return {
            "row_count": 0,
            "label_pass_rate": None,
            "avg_dl_quality_score": None,
        }
    if weights is None:
        return {
            "row_count": count,
            "label_pass_rate": _safe_rate(float((group["label"] == LABEL_PASS).mean())),
            "avg_dl_quality_score": _safe_rate(float(group["dl_quality_score"].mean())),
        }
    return {
        "row_count": count,
        "group_weight_sum": round(float(weights.sum()), 6),
        "label_pass_rate": _weighted_mean((group["label"].to_numpy() == LABEL_PASS).astype(np.float32), weights),
        "avg_dl_quality_score": _weighted_mean(group["dl_quality_score"].to_numpy(dtype=np.float32), weights),
    }


def _metrics(df: pd.DataFrame, *, group_weighted: bool) -> dict:
    valid = df[df["label"].isin([LABEL_PASS, LABEL_REJECT])].copy()
    if valid.empty:
        return {"row_count": 0}
    if group_weighted:
        weights = group_size_weights(valid, range(len(valid)))
    else:
        weights = None

    pass_mask = valid["dl_pass"].astype(bool).to_numpy()
    pred = pass_mask.astype(np.int64)
    truth = (valid["label"].to_numpy(dtype=np.int64) == LABEL_PASS).astype(np.int64)
    correct = (pred == truth).astype(np.float32)

    payload = {
        "row_count": int(len(valid)),
        "group_count": int(valid[["ticker", "date"]].drop_duplicates().shape[0]),
        "model_pass_rate": _safe_rate(float(pass_mask.mean())) if weights is None else _weighted_mean(pass_mask.astype(np.float32), weights),
        "accuracy": _safe_rate(float(correct.mean())) if weights is None else _weighted_mean(correct, weights),
        "all": _bucket_metrics(valid, weights),
    }
    pass_group = valid[pass_mask]
    reject_group = valid[~pass_mask]
    if weights is None:
        payload["pass_group"] = _bucket_metrics(pass_group, None)
        payload["reject_group"] = _bucket_metrics(reject_group, None)
    else:
        payload["pass_group"] = _bucket_metrics(pass_group, weights[pass_mask])
        payload["reject_group"] = _bucket_metrics(reject_group, weights[~pass_mask])
    return payload


def main(argv=None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(scores_csv_path(args.filter_id))
    metrics = {
        "filter_id": args.filter_id,
        "row_level": _metrics(df, group_weighted=False),
        "ticker_date_group_weighted": _metrics(df, group_weighted=True),
        "event_group_summary": event_group_summary(df, df["label"].to_numpy(dtype=np.int64)),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
