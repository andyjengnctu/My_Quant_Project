"""Evaluate breakout quality score table at event level."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import pandas as pd

from filters.breakout_quality.contract import DEFAULT_FILTER_ID, LABEL_PASS, LABEL_REJECT
from tools.filters.breakout_quality.common import scores_csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="評估 breakout quality event-level 區分能力")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    return parser.parse_args(argv)


def _group_metrics(df: pd.DataFrame) -> dict:
    valid = df[df["label"].isin([LABEL_PASS, LABEL_REJECT])].copy()
    if valid.empty:
        return {"event_count": 0}
    rows = {}
    for name, group in (("all", valid), ("pass_group", valid[valid["dl_pass"]]), ("reject_group", valid[~valid["dl_pass"]])):
        count = int(len(group))
        rows[name] = {
            "event_count": count,
            "label_pass_rate": None if count == 0 else round(float((group["label"] == LABEL_PASS).mean()), 6),
            "avg_dl_quality_score": None if count == 0 else round(float(group["dl_quality_score"].mean()), 6),
        }
    rows["model_pass_rate"] = round(float(valid["dl_pass"].mean()), 6)
    rows["accuracy"] = round(float(((valid["dl_pass"] & (valid["label"] == LABEL_PASS)) | (~valid["dl_pass"] & (valid["label"] == LABEL_REJECT))).mean()), 6)
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(scores_csv_path(args.filter_id))
    metrics = _group_metrics(df)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
