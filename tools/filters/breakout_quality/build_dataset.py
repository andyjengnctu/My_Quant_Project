"""Build event-level dataset for breakout quality filter."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import CONTEXT_COLUMNS, DEFAULT_FILTER_ID, FEATURE_COLUMNS
from filters.breakout_quality.features import build_breakout_quality_dataset_for_frame
from filters.breakout_quality.source_inventory import build_source_data_inventory
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    add_policy_args,
    build_policy_from_args,
    dataset_npz_path,
    dataset_output_dir,
    event_group_summary,
    events_csv_path,
    label_counts,
    load_dataset_frames,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="建立 breakout quality filter 事件資料集")
    parser.add_argument("--dataset", default="reduced", choices=("reduced", "full"))
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--max-tickers", type=int, default=0)
    add_policy_args(parser)
    return parser.parse_args(argv)


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, str | None]:
    if frame.empty or column not in frame.columns:
        return {"start": None, "end": None}
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return {"start": None, "end": None}
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}



def _frames_date_range(frames: dict[str, pd.DataFrame]) -> dict[str, str | None]:
    starts = []
    ends = []
    for frame in frames.values():
        if frame.empty:
            continue
        dates = pd.to_datetime(frame.index, errors="raise")
        starts.append(dates.min())
        ends.append(dates.max())
    if not starts:
        return {"start": None, "end": None}
    return {"start": str(min(starts).date()), "end": str(max(ends).date())}

def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    policy = build_policy_from_args(args)
    requested_max_tickers = max(0, int(args.max_tickers or 0))
    source_inventory_before = build_source_data_inventory(PROJECT_ROOT, args.dataset)
    out_dir = dataset_output_dir(args.filter_id)
    min_rows = max(policy.high_len_max + 10, policy.feature_window_bars + policy.label_horizon_bars + 10)
    frames = load_dataset_frames(PROJECT_ROOT, args.dataset, min_rows=min_rows)
    benchmark = frames.get(policy.benchmark_ticker)
    if benchmark is None:
        raise FileNotFoundError(f"找不到 benchmark ticker: {policy.benchmark_ticker}")

    tickers = [ticker for ticker in sorted(frames) if ticker != policy.benchmark_ticker]
    if requested_max_tickers > 0:
        tickers = tickers[:requested_max_tickers]

    features = []
    contexts = []
    labels = []
    events = []
    for idx, ticker in enumerate(tickers, start=1):
        dataset = build_breakout_quality_dataset_for_frame(frames[ticker], benchmark, ticker=ticker, policy=policy)
        if len(dataset.labels) == 0:
            continue
        features.append(dataset.features)
        contexts.append(dataset.context)
        labels.append(dataset.labels)
        events.append(dataset.events)
        print(f"[{idx}/{len(tickers)}] {ticker} events={len(dataset.labels)} labels={label_counts(dataset.labels)}")

    if features:
        X = np.concatenate(features, axis=0).astype(np.float32)
        C = np.concatenate(contexts, axis=0).astype(np.float32)
        y = np.concatenate(labels, axis=0).astype(np.int64)
        event_df = pd.concat(events, ignore_index=True)
    else:
        X = np.empty((0, policy.feature_window_bars, len(FEATURE_COLUMNS)), dtype=np.float32)
        C = np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        event_df = pd.DataFrame(columns=[
            "ticker",
            "date",
            "label_eval_start_date",
            "label_eval_end_date",
            "high_len",
            "breakout_level",
            "label",
            "label_reason",
            "anchor_price",
            "pass_barrier_price",
            "reject_barrier_price",
            "max_upside_return",
            "max_downside_return",
            "first_hit_bar",
        ])

    source_inventory_after = build_source_data_inventory(PROJECT_ROOT, args.dataset)
    if source_inventory_after != source_inventory_before:
        raise RuntimeError("來源 CSV 在 build_dataset 執行期間發生變更；請完成資料更新後重新執行")

    dataset_path = dataset_npz_path(args.filter_id)
    event_path = events_csv_path(args.filter_id)
    np.savez_compressed(dataset_path, features=X, context=C, labels=y)
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")
    summary = {
        "filter_id": args.filter_id,
        "dataset": args.dataset,
        "policy": policy.as_manifest_payload(),
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "label_counts": label_counts(y),
        "event_count": int(len(y)),
        "event_group_summary": event_group_summary(event_df, y),
        "event_date_range": _date_range(event_df, "date"),
        "label_information_end_date_range": _date_range(event_df, "label_eval_end_date"),
        "source_data_date_range": _frames_date_range(frames),
        "source_data_inventory": source_inventory_after,
        "source_selection": {
            "requested_max_tickers": requested_max_tickers,
            "selected_ticker_count": int(len(tickers)),
        },
        "dataset_artifacts": {
            "dataset_npz": build_file_manifest(dataset_path),
            "events_csv": build_file_manifest(event_path),
        },
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(out_dir / "dataset_summary.json", summary)
    print(f"已輸出: {dataset_path}")
    print(f"已輸出: {event_path}")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
