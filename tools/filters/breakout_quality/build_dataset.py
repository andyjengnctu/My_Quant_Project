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

from filters.breakout_quality.contract import DEFAULT_FILTER_ID, FEATURE_COLUMNS, CONTEXT_COLUMNS
from filters.breakout_quality.features import build_breakout_quality_dataset_for_frame
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    add_policy_args,
    build_policy_from_args,
    dataset_npz_path,
    events_csv_path,
    label_counts,
    load_dataset_frames,
    write_json,
    dataset_output_dir,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="建立 breakout quality filter 事件資料集")
    parser.add_argument("--dataset", default="reduced", choices=("reduced", "full"))
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--max-tickers", type=int, default=0)
    add_policy_args(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    policy = build_policy_from_args(args)
    min_rows = max(policy.high_len_max + 10, policy.feature_window_bars + policy.label_horizon_bars + 10)
    frames = load_dataset_frames(PROJECT_ROOT, args.dataset, min_rows=min_rows)
    benchmark = frames.get(policy.benchmark_ticker)
    if benchmark is None:
        raise FileNotFoundError(f"找不到 benchmark ticker: {policy.benchmark_ticker}")

    tickers = [ticker for ticker in sorted(frames) if ticker != policy.benchmark_ticker]
    if int(args.max_tickers or 0) > 0:
        tickers = tickers[: int(args.max_tickers)]

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

    out_dir = dataset_output_dir(args.filter_id)
    if features:
        X = np.concatenate(features, axis=0).astype(np.float32)
        C = np.concatenate(contexts, axis=0).astype(np.float32)
        y = np.concatenate(labels, axis=0).astype(np.int64)
        event_df = pd.concat(events, ignore_index=True)
    else:
        X = np.empty((0, policy.feature_window_bars, len(FEATURE_COLUMNS)), dtype=np.float32)
        C = np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        event_df = pd.DataFrame()

    np.savez_compressed(dataset_npz_path(args.filter_id), features=X, context=C, labels=y)
    event_df.to_csv(events_csv_path(args.filter_id), index=False, encoding="utf-8-sig")
    summary = {
        "filter_id": args.filter_id,
        "dataset": args.dataset,
        "policy": policy.as_manifest_payload(),
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "label_counts": label_counts(y),
        "event_count": int(len(y)),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(out_dir / "dataset_summary.json", summary)
    print(f"已輸出: {dataset_npz_path(args.filter_id)}")
    print(f"已輸出: {events_csv_path(args.filter_id)}")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
