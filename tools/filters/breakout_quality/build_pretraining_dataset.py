"""Build Selection-only unlabeled rolling windows for TS2Vec pretraining."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_FEATURE_WINDOW_BARS,
    BREAKOUT_QUALITY_PRETRAINING_FAMILY,
    BREAKOUT_QUALITY_PRETRAINING_STRIDE,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.features import build_breakout_quality_sequence_feature
from filters.breakout_quality.pretraining_store import (
    PRETRAINING_DATASET_FORMAT,
    PRETRAINING_DATASET_SCHEMA_VERSION,
    build_file_record,
    compute_pretraining_configuration_fingerprint,
    resolve_pretraining_dataset_paths,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    discover_dataset_csv_inputs,
    load_dataset_frame,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="建立只含 Selection endpoint 的 TS2Vec 未標記 rolling-window dataset"
    )
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--stride", type=int, default=BREAKOUT_QUALITY_PRETRAINING_STRIDE)
    parser.add_argument("--max-tickers", type=int, default=0)
    return parser.parse_args(argv)


def _combine_chunks(destination: Path, chunks: list[Path], *, shape: tuple[int, ...]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    memmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=shape)
    cursor = 0
    for chunk_path in chunks:
        chunk = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
        next_cursor = cursor + len(chunk)
        memmap[cursor:next_cursor] = chunk
        cursor = next_cursor
    memmap.flush()
    del memmap
    if cursor != shape[0]:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"pretraining chunk count 不一致: expected={shape[0]}, actual={cursor}")
    temp_path.replace(destination)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    stride = int(args.stride)
    max_tickers = int(args.max_tickers)
    if stride < 1:
        raise ValueError("--stride 必須 >= 1")
    if max_tickers < 0:
        raise ValueError("--max-tickers 必須 >= 0")

    source_inventory_before = build_source_data_inventory(PROJECT_ROOT, args.dataset)
    csv_inputs, duplicate_lines = discover_dataset_csv_inputs(PROJECT_ROOT, args.dataset)
    for line in duplicate_lines:
        print(line)
    input_map = {ticker: path for ticker, path in csv_inputs}
    benchmark_path = input_map.get(DEFAULT_LABEL_POLICY.benchmark_ticker)
    if benchmark_path is None:
        raise FileNotFoundError(
            f"找不到 benchmark ticker: {DEFAULT_LABEL_POLICY.benchmark_ticker}"
        )
    min_rows = int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS) + 10
    benchmark = load_dataset_frame(
        benchmark_path,
        DEFAULT_LABEL_POLICY.benchmark_ticker,
        min_rows=min_rows,
    )
    source_end = pd.Timestamp(benchmark.index.max()).strftime("%Y-%m-%d")
    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=source_end,
    )
    selection_start = str(outer_policy["selection_start_date"])
    selection_end = str(outer_policy["selection_end_date"])
    selection_start_ts = pd.Timestamp(selection_start)
    selection_end_ts = pd.Timestamp(selection_end)

    tickers = [ticker for ticker in sorted(input_map) if ticker != DEFAULT_LABEL_POLICY.benchmark_ticker]
    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    paths = resolve_pretraining_dataset_paths(
        PROJECT_ROOT,
        args.filter_id,
        family=BREAKOUT_QUALITY_PRETRAINING_FAMILY,
        stride=stride,
    )
    paths.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ts2vec_pretraining_dataset_",
        dir=paths.output_dir.parent,
    ) as temp_dir_text:
        temp_root = Path(temp_dir_text)
        temp_dir = temp_root / "artifact"
        temp_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = temp_root / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths: list[Path] = []
        index_path = temp_dir / paths.index.name
        wrote_header = False
        window_count = 0
        processed_tickers = 0
        skipped_tickers = 0
        observed_start: pd.Timestamp | None = None
        observed_end: pd.Timestamp | None = None

        for position, ticker in enumerate(tickers, start=1):
            try:
                stock = load_dataset_frame(input_map[ticker], ticker, min_rows=min_rows)
            except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
                skipped_tickers += 1
                print(f"[skip] {ticker}: {exc}")
                continue
            dates = pd.DatetimeIndex(stock.index)
            eligible = np.flatnonzero(
                (dates >= selection_start_ts)
                & (dates <= selection_end_ts)
                & (np.arange(len(stock)) >= int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS) - 1)
            )
            eligible = eligible[::stride]
            sequences: list[np.ndarray] = []
            rows: list[dict[str, object]] = []
            for event_pos in eligible.tolist():
                sequence = build_breakout_quality_sequence_feature(
                    stock,
                    benchmark,
                    event_pos=int(event_pos),
                    policy=DEFAULT_LABEL_POLICY,
                )
                if sequence is None:
                    continue
                endpoint = pd.Timestamp(stock.index[int(event_pos)]).normalize()
                sequences.append(sequence)
                rows.append(
                    {
                        "window_index": window_count + len(rows),
                        "ticker": str(ticker),
                        "date": endpoint.strftime("%Y-%m-%d"),
                    }
                )
                observed_start = endpoint if observed_start is None else min(observed_start, endpoint)
                observed_end = endpoint if observed_end is None else max(observed_end, endpoint)
            if sequences:
                chunk = np.stack(sequences).astype(np.float32, copy=False)
                chunk_path = chunk_dir / f"{len(chunk_paths):06d}_windows.npy"
                with chunk_path.open("wb") as handle:
                    np.save(handle, chunk, allow_pickle=False)
                chunk_paths.append(chunk_path)
                pd.DataFrame(rows).to_csv(
                    index_path,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8" if wrote_header else "utf-8-sig",
                )
                wrote_header = True
                window_count += len(rows)
            processed_tickers += 1
            if position == 1 or position % 25 == 0 or position == len(tickers):
                print(
                    f"Selection rolling windows {position}/{len(tickers)} | "
                    f"{ticker} | windows={window_count:,} | elapsed={time.perf_counter()-started:.1f}s"
                )

        if window_count < 2 or not wrote_header:
            raise ValueError("Selection-only pretraining windows 不足")
        windows_path = temp_dir / paths.windows.name
        _combine_chunks(
            windows_path,
            chunk_paths,
            shape=(
                window_count,
                int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS),
                len(FEATURE_COLUMNS),
            ),
        )
        source_inventory_after = build_source_data_inventory(PROJECT_ROOT, args.dataset)
        if source_inventory_after != source_inventory_before:
            raise RuntimeError("來源 CSV 在 pretraining dataset build 期間發生變更")
        configuration = {
            "family": BREAKOUT_QUALITY_PRETRAINING_FAMILY,
            "dataset_profile": str(args.dataset),
            "stride": stride,
            "window_bars": int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS),
            "feature_columns": list(FEATURE_COLUMNS),
            "selection_start_date": selection_start,
            "selection_end_date": selection_end,
            "outer_policy_fingerprint": outer_policy["policy_fingerprint_sha256"],
            "source_inventory_sha256": source_inventory_after["csv_inventory_sha256"],
            "requested_max_tickers": int(max_tickers),
        }
        summary = {
            "schema_version": PRETRAINING_DATASET_SCHEMA_VERSION,
            "format": PRETRAINING_DATASET_FORMAT,
            **configuration,
            "configuration_fingerprint": compute_pretraining_configuration_fingerprint(configuration),
            "window_count": int(window_count),
            "observed_window_end_date_range": {
                "start": observed_start.strftime("%Y-%m-%d") if observed_start is not None else None,
                "end": observed_end.strftime("%Y-%m-%d") if observed_end is not None else None,
            },
            "processed_ticker_count": int(processed_tickers),
            "skipped_ticker_count": int(skipped_tickers),
            "requested_max_tickers": int(max_tickers),
            "source_data_inventory": source_inventory_after,
            "outer_oos_policy": outer_policy,
            "oos_windows_used": False,
            "artifacts": {
                "windows": build_file_record(windows_path),
                "index": build_file_record(index_path),
            },
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        write_json(temp_dir / paths.summary.name, summary)
        if paths.output_dir.exists():
            shutil.rmtree(paths.output_dir)
        temp_dir.replace(paths.output_dir)

    print(f"已建立 TS2Vec Selection-only pretraining dataset: {paths.output_dir}")
    print(f"windows={window_count:,}, date={selection_start}~{selection_end}, stride={stride}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
