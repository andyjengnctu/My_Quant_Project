"""Build or quickly relabel the indexed breakout quality dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import Counter
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from config.breakout_quality import BREAKOUT_QUALITY_MODEL_ARCHITECTURE
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    DEFAULT_FILTER_ID,
    FEATURE_COLUMNS,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    load_npy,
    save_npy_atomic,
)
from filters.breakout_quality.features import (
    EVENT_COLUMNS,
    build_breakout_quality_dataset_for_frame,
    label_from_cached_path,
)
from filters.breakout_quality.market_set import (
    MARKET_SET_FEATURE_COLUMNS,
    build_market_daily_base_features,
    market_set_contract_payload,
)
from filters.breakout_quality.models.spec import get_model_spec
from filters.breakout_quality.source_inventory import build_source_data_inventory
from core.console_report import (
    compact_console_enabled,
    console_color_enabled,
    paint,
    print_artifact_paths,
)
from core.display_common import InlineProgress, render_elapsed
from filters.breakout_quality.workflow_io import (
    PROJECT_ROOT,
    add_policy_args,
    build_policy_from_args,
    dataset_output_dir,
    dataset_paths,
    discover_dataset_csv_inputs,
    event_group_summary,
    label_counts,
    load_dataset_frame,
    load_validated_dataset_bundle,
    read_breakout_quality_csv,
    read_json,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="建立或快速重貼 breakout quality filter 事件資料集")
    parser.add_argument("--dataset", default="reduced", choices=("reduced", "full"))
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument(
        "--relabel-only",
        action="store_true",
        help="沿用 feature bank 與 future path cache，只依目前 label policy 更新 labels/events",
    )
    add_policy_args(parser)
    return parser.parse_args(argv)


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, str | None]:
    if frame.empty or column not in frame.columns:
        return {"start": None, "end": None}
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return {"start": None, "end": None}
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}


def _merge_date_range(
    current_start: pd.Timestamp | None,
    current_end: pd.Timestamp | None,
    frame: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if frame.empty:
        return current_start, current_end
    dates = pd.to_datetime(frame.index, errors="raise")
    frame_start = dates.min()
    frame_end = dates.max()
    return (
        frame_start if current_start is None else min(current_start, frame_start),
        frame_end if current_end is None else max(current_end, frame_end),
    )


def _iso_date_from_ordinal(value: int) -> str | None:
    ordinal = int(value)
    if ordinal < 0:
        return None
    return str(np.datetime64(ordinal, "D"))


def _write_events_atomic(path: Path, frame: pd.DataFrame) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)


def _build_artifact_records(paths) -> dict[str, dict]:
    return {
        artifact_name: build_file_manifest(artifact_path)
        for artifact_name, artifact_path in paths.artifact_paths().items()
    }


def _build_storage_summary(
    *,
    feature_group_count: int,
    event_count: int,
    feature_window_bars: int,
    feature_count: int,
) -> dict:
    group_count = int(feature_group_count)
    expanded_bytes = int(
        event_count
        * int(feature_window_bars)
        * int(feature_count)
        * np.dtype(np.float32).itemsize
    )
    bank_bytes = int(
        group_count
        * int(feature_window_bars)
        * int(feature_count)
        * np.dtype(np.float32).itemsize
    )
    return {
        "feature_group_count": group_count,
        "event_count": int(event_count),
        "events_per_feature_group": round(float(event_count / group_count), 6) if group_count else 0.0,
        "feature_bank_bytes": bank_bytes,
        "expanded_event_feature_bytes": expanded_bytes,
        "feature_storage_reduction_ratio": round(float(expanded_bytes / bank_bytes), 6) if bank_bytes else 0.0,
    }


def _save_npy_chunk(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _combine_npy_chunks_atomic(
    destination: Path,
    chunk_paths: list[Path],
    *,
    shape: tuple[int, ...],
    dtype,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not chunk_paths:
        save_npy_atomic(destination, np.empty(shape, dtype=dtype))
        return
    temp_path = destination.with_name(destination.name + ".tmp")
    temp_path.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(temp_path, mode="w+", dtype=dtype, shape=shape)
    offset = 0
    try:
        for chunk_path in chunk_paths:
            chunk = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
            count = int(len(chunk))
            target[offset:offset + count] = chunk
            offset += count
            del chunk
        if offset != int(shape[0]):
            raise RuntimeError(
                f"chunk row count 不一致: destination={destination.name}, expected={shape[0]}, actual={offset}"
            )
        target.flush()
    finally:
        del target
    temp_path.replace(destination)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temp_path)
    temp_path.replace(destination)


def _merge_event_date_range(
    current_start: pd.Timestamp | None,
    current_end: pd.Timestamp | None,
    frame: pd.DataFrame,
    column: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if frame.empty or column not in frame.columns:
        return current_start, current_end
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return current_start, current_end
    frame_start = dates.min()
    frame_end = dates.max()
    return (
        frame_start if current_start is None else min(current_start, frame_start),
        frame_end if current_end is None else max(current_end, frame_end),
    )


def _format_timestamp_range(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> dict[str, str | None]:
    return {
        "start": str(start.date()) if start is not None else None,
        "end": str(end.date()) if end is not None else None,
    }


def _render_full_build_progress(
    *,
    index: int,
    total: int,
    ticker: str,
    event_count: int,
    group_count: int,
    elapsed_sec: float,
    color: bool = False,
) -> str:
    ratio = (100.0 * int(index) / int(total)) if int(total) > 0 else 100.0
    return (
        f"Dataset 建立 {int(index):>3}/{int(total)} ({ratio:5.1f}%) | "
        f"{ticker} | events={int(event_count):,} | groups={int(group_count):,} | "
        f"{render_elapsed(elapsed_sec, color=color)}"
    )


def _skip_reason_label(exc: Exception) -> str:
    message = str(exc).strip()
    if "有效資料不足" in message:
        return "有效資料不足"
    if "缺少必要欄位" in message:
        return "欄位不完整"
    if isinstance(exc, (UnicodeDecodeError, OSError)):
        return "檔案讀取失敗"
    return type(exc).__name__


def _render_skip_summary(reason_counts: Counter[str]) -> str:
    total = sum(int(count) for count in reason_counts.values())
    if total <= 0:
        return "跳過=0"
    details = "；".join(
        f"{reason}={int(count):,}"
        for reason, count in sorted(
            reason_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    )
    return f"跳過={total:,}（{details}）"


def _full_build(args, policy, *, started: float) -> int:
    requested_max_tickers = max(0, int(args.max_tickers or 0))
    compact_console = compact_console_enabled()
    source_inventory_before = build_source_data_inventory(PROJECT_ROOT, args.dataset)
    out_dir = dataset_output_dir(args.filter_id)
    paths = dataset_paths(args.filter_id)
    min_rows = max(policy.high_len_max + 10, policy.feature_window_bars + 10)
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    market_set_required = bool(model_spec.requires_market_set)
    if market_set_required and int(model_spec.market_set_history_bars or 0) != int(
        policy.feature_window_bars
    ):
        raise ValueError(
            "Market Set architecture 要求 market history bars 與 feature window 一致: "
            f"market={model_spec.market_set_history_bars}, feature={policy.feature_window_bars}"
        )

    csv_inputs, duplicate_lines = discover_dataset_csv_inputs(PROJECT_ROOT, args.dataset)
    if duplicate_lines:
        if compact_console:
            print(f"[注意] 重複來源檔={len(duplicate_lines):,}，已依既有規則去重。")
        else:
            for line in duplicate_lines:
                print(line)
    input_map = {ticker: path for ticker, path in csv_inputs}
    benchmark_path = input_map.get(policy.benchmark_ticker)
    if benchmark_path is None:
        raise FileNotFoundError(f"找不到 benchmark ticker: {policy.benchmark_ticker}")
    benchmark = load_dataset_frame(benchmark_path, policy.benchmark_ticker, min_rows=min_rows)

    tickers = [ticker for ticker in sorted(input_map) if ticker != policy.benchmark_ticker]
    if requested_max_tickers > 0:
        tickers = tickers[:requested_max_tickers]

    source_start, source_end = _merge_date_range(None, None, benchmark)
    event_start = event_end = None
    label_end_start = label_end_end = None
    processed_ticker_count = 0
    event_count = 0
    group_count = 0
    valid_group_count = 0
    rows_per_group_min: int | None = None
    rows_per_group_max = 0
    accumulated_label_counts = {"pass": 0, "reject": 0, "invalid": 0, "total": 0}

    chunk_names = (
        "feature_bank",
        "event_context",
        "event_labels",
        "event_group_index",
        "group_anchor_prices",
        "future_high_prices",
        "future_low_prices",
        "future_available_bars",
        "future_date_ordinals",
    )
    chunk_paths: dict[str, list[Path]] = {name: [] for name in chunk_names}
    skipped_ticker_count = 0
    skipped_reason_counts: Counter[str] = Counter()
    progress = InlineProgress()
    color_enabled = console_color_enabled(progress.stream)
    market_group_date_chunks: list[np.ndarray] = []
    market_valid_ticker_count = 0

    with tempfile.TemporaryDirectory(prefix=".breakout_quality_build_", dir=out_dir) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        temp_events_path = temp_dir / "events.csv"
        wrote_events_header = False
        market_features_memmap = None
        market_valid_memmap = None
        market_dates = pd.DatetimeIndex(benchmark.index)
        market_date_to_index = {pd.Timestamp(value): index for index, value in enumerate(market_dates)}
        if market_set_required:
            market_features_temp = temp_dir / "market_daily_features.npy"
            market_valid_temp = temp_dir / "market_daily_valid_mask.npy"
            market_features_memmap = np.lib.format.open_memmap(
                market_features_temp,
                mode="w+",
                dtype=np.float32,
                shape=(len(market_dates), len(tickers), len(MARKET_SET_FEATURE_COLUMNS)),
            )
            market_valid_memmap = np.lib.format.open_memmap(
                market_valid_temp,
                mode="w+",
                dtype=np.bool_,
                shape=(len(market_dates), len(tickers)),
            )
            market_features_memmap[:] = 0.0
            market_valid_memmap[:] = False

        for idx, ticker in enumerate(tickers, start=1):
            try:
                stock_frame = load_dataset_frame(input_map[ticker], ticker, min_rows=min_rows)
            except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
                skipped_ticker_count += 1
                skipped_reason_counts[_skip_reason_label(exc)] += 1
                progress.update(
                    _render_full_build_progress(
                        index=idx,
                        total=len(tickers),
                        ticker=ticker,
                        event_count=event_count,
                        group_count=group_count,
                        elapsed_sec=time.perf_counter() - started,
                        color=color_enabled,
                    )
                )
                continue
            processed_ticker_count += 1
            source_start, source_end = _merge_date_range(source_start, source_end, stock_frame)
            if market_set_required:
                assert market_features_memmap is not None and market_valid_memmap is not None
                market_values, market_valid = build_market_daily_base_features(
                    stock_frame,
                    market_dates,
                )
                market_features_memmap[:, idx - 1, :] = market_values
                market_valid_memmap[:, idx - 1] = market_valid
                if bool(market_valid.any()):
                    market_valid_ticker_count += 1
            dataset = build_breakout_quality_dataset_for_frame(
                stock_frame,
                benchmark,
                ticker=ticker,
                policy=policy,
            )
            if len(dataset.labels) == 0:
                progress.update(
                    _render_full_build_progress(
                        index=idx,
                        total=len(tickers),
                        ticker=ticker,
                        event_count=event_count,
                        group_count=group_count,
                        elapsed_sec=time.perf_counter() - started,
                        color=color_enabled,
                    )
                )
                del stock_frame, dataset
                continue

            local_group_count = int(len(dataset.feature_bank))
            local_event_count = int(len(dataset.labels))
            adjusted_group_index = dataset.event_group_index.astype(np.int64) + int(group_count)
            event_frame = dataset.events.copy()
            event_frame["group_index"] = adjusted_group_index.astype(np.int32)
            if market_set_required:
                group_dates = (
                    event_frame[["group_index", "date"]]
                    .drop_duplicates("group_index")
                    .sort_values("group_index")
                )
                if len(group_dates) != local_group_count:
                    raise RuntimeError(f"{ticker} market group/date mapping 不完整")
                mapped_dates = np.asarray(
                    [
                        market_date_to_index.get(pd.Timestamp(value), -1)
                        for value in pd.to_datetime(group_dates["date"], errors="raise")
                    ],
                    dtype=np.int32,
                )
                if np.any(mapped_dates < int(model_spec.market_set_history_bars or 0) - 1):
                    raise RuntimeError(f"{ticker} market group date 無法提供完整歷史視窗")
                market_group_date_chunks.append(mapped_dates)

            arrays = {
                "feature_bank": dataset.feature_bank,
                "event_context": dataset.context,
                "event_labels": dataset.labels,
                "event_group_index": adjusted_group_index.astype(np.int32),
                "group_anchor_prices": dataset.group_anchor_prices,
                "future_high_prices": dataset.future_high_prices,
                "future_low_prices": dataset.future_low_prices,
                "future_available_bars": dataset.future_available_bars,
                "future_date_ordinals": dataset.future_date_ordinals,
            }
            chunk_number = len(chunk_paths["feature_bank"])
            for name, array in arrays.items():
                chunk_path = temp_dir / f"{chunk_number:06d}_{name}.npy"
                _save_npy_chunk(chunk_path, array)
                chunk_paths[name].append(chunk_path)

            event_frame.to_csv(
                temp_events_path,
                mode="a" if wrote_events_header else "w",
                header=not wrote_events_header,
                index=False,
                encoding="utf-8" if wrote_events_header else "utf-8-sig",
            )
            wrote_events_header = True

            local_counts = label_counts(dataset.labels)
            for key in accumulated_label_counts:
                accumulated_label_counts[key] += int(local_counts[key])
            local_group_sizes = np.bincount(
                dataset.event_group_index.astype(np.int64),
                minlength=local_group_count,
            )
            if local_group_count and bool(np.any(local_group_sizes <= 0)):
                raise RuntimeError(f"{ticker} 存在沒有事件列的 feature group")
            if local_group_count:
                local_group_labels_min = np.full((local_group_count,), 2, dtype=np.int8)
                local_group_labels_max = np.full((local_group_count,), -2, dtype=np.int8)
                np.minimum.at(local_group_labels_min, dataset.event_group_index, dataset.labels)
                np.maximum.at(local_group_labels_max, dataset.event_group_index, dataset.labels)
                if bool(np.any(local_group_labels_min != local_group_labels_max)):
                    raise RuntimeError(f"{ticker} 同一 ticker/date group 出現混合 label")
                valid_group_count += int(np.isin(local_group_labels_min, [LABEL_REJECT, LABEL_PASS]).sum())
                local_min = int(local_group_sizes.min())
                local_max = int(local_group_sizes.max())
                rows_per_group_min = local_min if rows_per_group_min is None else min(rows_per_group_min, local_min)
                rows_per_group_max = max(rows_per_group_max, local_max)

            event_start, event_end = _merge_event_date_range(
                event_start, event_end, event_frame, "date"
            )
            label_end_start, label_end_end = _merge_event_date_range(
                label_end_start, label_end_end, event_frame, "label_eval_end_date"
            )
            event_count += local_event_count
            group_count += local_group_count
            progress.update(
                _render_full_build_progress(
                    index=idx,
                    total=len(tickers),
                    ticker=ticker,
                    event_count=event_count,
                    group_count=group_count,
                    elapsed_sec=time.perf_counter() - started,
                    color=color_enabled,
                )
            )
            del stock_frame, dataset, event_frame, arrays

        progress.finish(
            f"{paint('Dataset 建立完成', 'green', enabled=color_enabled, bold=True)} | "
            f"股票={processed_ticker_count:,}/{len(tickers):,} | "
            f"{_render_skip_summary(skipped_reason_counts)} | "
            f"events={event_count:,} groups={group_count:,} | "
            f"耗時 {render_elapsed(time.perf_counter() - started, color=color_enabled)}"
        )
        if market_set_required:
            assert market_features_memmap is not None and market_valid_memmap is not None
            market_features_memmap.flush()
            market_valid_memmap.flush()
            del market_features_memmap, market_valid_memmap
        source_inventory_after = build_source_data_inventory(PROJECT_ROOT, args.dataset)
        if source_inventory_after != source_inventory_before:
            raise RuntimeError("來源 CSV 在 build_dataset 執行期間發生變更；請完成資料更新後重新執行")

        _combine_npy_chunks_atomic(
            paths.feature_bank,
            chunk_paths["feature_bank"],
            shape=(group_count, int(policy.feature_window_bars), len(FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        _combine_npy_chunks_atomic(
            paths.event_context,
            chunk_paths["event_context"],
            shape=(event_count, len(CONTEXT_COLUMNS)),
            dtype=np.float32,
        )
        _combine_npy_chunks_atomic(
            paths.event_labels,
            chunk_paths["event_labels"],
            shape=(event_count,),
            dtype=np.int8,
        )
        _combine_npy_chunks_atomic(
            paths.event_group_index,
            chunk_paths["event_group_index"],
            shape=(event_count,),
            dtype=np.int32,
        )
        _combine_npy_chunks_atomic(
            paths.group_anchor_prices,
            chunk_paths["group_anchor_prices"],
            shape=(group_count,),
            dtype=np.float64,
        )
        for destination, name in (
            (paths.future_high_prices, "future_high_prices"),
            (paths.future_low_prices, "future_low_prices"),
        ):
            _combine_npy_chunks_atomic(
                destination,
                chunk_paths[name],
                shape=(group_count, int(policy.label_path_cache_bars)),
                dtype=np.float64,
            )
        _combine_npy_chunks_atomic(
            paths.future_available_bars,
            chunk_paths["future_available_bars"],
            shape=(group_count,),
            dtype=np.int16,
        )
        _combine_npy_chunks_atomic(
            paths.future_date_ordinals,
            chunk_paths["future_date_ordinals"],
            shape=(group_count, int(policy.label_path_cache_bars)),
            dtype=np.int32,
        )
        if wrote_events_header:
            _copy_file_atomic(temp_events_path, paths.events)
        else:
            _write_events_atomic(paths.events, pd.DataFrame(columns=list(EVENT_COLUMNS)))

        if market_set_required:
            if len(market_group_date_chunks) == 0 and group_count > 0:
                raise RuntimeError("market group date mapping 缺失")
            group_market_date_index = (
                np.concatenate(market_group_date_chunks).astype(np.int32, copy=False)
                if market_group_date_chunks
                else np.empty((0,), dtype=np.int32)
            )
            if len(group_market_date_index) != group_count:
                raise RuntimeError(
                    "market group date mapping 長度不一致: "
                    f"actual={len(group_market_date_index)}, expected={group_count}"
                )
            _copy_file_atomic(temp_dir / "market_daily_features.npy", paths.market_daily_features)
            _copy_file_atomic(temp_dir / "market_daily_valid_mask.npy", paths.market_daily_valid_mask)
            save_npy_atomic(
                paths.market_date_ordinals,
                market_dates.to_numpy(dtype="datetime64[D]").astype(np.int32),
            )
            save_npy_atomic(paths.group_market_date_index, group_market_date_index)
            ticker_frame = pd.DataFrame(
                {"market_ticker_index": np.arange(len(tickers), dtype=np.int32), "ticker": tickers}
            )
            _write_events_atomic(paths.market_tickers, ticker_frame)

    paths.legacy_dataset.unlink(missing_ok=True)
    storage_summary = _build_storage_summary(
        feature_group_count=group_count,
        event_count=event_count,
        feature_window_bars=int(policy.feature_window_bars),
        feature_count=len(FEATURE_COLUMNS),
    )
    group_summary = {
        "group_key": "ticker/date",
        "group_count": int(group_count),
        "valid_group_count": int(valid_group_count),
        "rows_per_group_min": int(rows_per_group_min or 0),
        "rows_per_group_max": int(rows_per_group_max),
        "rows_per_group_mean": round(float(event_count / group_count), 6) if group_count else 0.0,
        "mixed_label_group_count": 0,
        "mixed_valid_label_group_count": 0,
    }
    summary = {
        "filter_id": args.filter_id,
        "dataset": args.dataset,
        "dataset_storage_schema_version": DATASET_STORAGE_SCHEMA_VERSION,
        "dataset_storage_format": DATASET_STORAGE_FORMAT,
        "policy": policy.as_manifest_payload(),
        "feature_cache_policy": policy.feature_cache_manifest_payload(),
        "label_policy": policy.label_manifest_payload(),
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "label_counts": accumulated_label_counts,
        "event_count": int(event_count),
        "feature_group_count": int(group_count),
        "storage_summary": storage_summary,
        "event_group_summary": group_summary,
        "event_date_range": _format_timestamp_range(event_start, event_end),
        "label_information_end_date_range": _format_timestamp_range(label_end_start, label_end_end),
        "source_data_date_range": _format_timestamp_range(source_start, source_end),
        "source_data_inventory": source_inventory_after,
        "source_selection": {
            "requested_max_tickers": requested_max_tickers,
            "selected_ticker_count": int(len(tickers)),
            "processed_ticker_count": int(processed_ticker_count),
            "skipped_ticker_count": int(skipped_ticker_count),
            "skipped_reason_counts": {
                str(reason): int(count)
                for reason, count in sorted(skipped_reason_counts.items())
            },
        },
        "dataset_artifacts": _build_artifact_records(paths),
        "market_set_contract": (market_set_contract_payload() if market_set_required else None),
        "market_set_artifacts": (
            {
                artifact_name: build_file_manifest(artifact_path)
                for artifact_name, artifact_path in paths.market_set_artifact_paths().items()
            }
            if market_set_required
            else None
        ),
        "market_set_summary": (
            {
                "market_date_count": int(len(benchmark.index)),
                "market_ticker_count": int(len(tickers)),
                "market_valid_ticker_count": int(market_valid_ticker_count),
                "feature_count": int(len(MARKET_SET_FEATURE_COLUMNS)),
                "feature_columns": list(MARKET_SET_FEATURE_COLUMNS),
                "group_market_date_count": int(group_count),
            }
            if market_set_required
            else None
        ),
        "build_mode": "full_feature_bank_rebuild",
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(paths.summary, summary)
    print_artifact_paths(
        (
            ("Indexed feature bank", paths.feature_bank),
            ("Event arrays", paths.event_context.parent),
            ("Events CSV", paths.events),
            ("Dataset summary", paths.summary),
        ),
        project_root=PROJECT_ROOT,
    )
    if not compact_console:
        print(
            "storage="
            f"events={storage_summary['event_count']} groups={storage_summary['feature_group_count']} "
            f"dedup={storage_summary['feature_storage_reduction_ratio']}x"
        )
    return 0


def _relabel_only(args, policy, *, started: float) -> int:
    paths = dataset_paths(args.filter_id)
    summary = read_json(paths.summary)
    stored_profile = str(summary.get("dataset") or "").strip().lower()
    if stored_profile != str(args.dataset).strip().lower():
        raise ValueError(
            f"relabel-only dataset profile 不一致: existing={stored_profile}, requested={args.dataset}"
        )
    stored_selection = summary.get("source_selection")
    stored_max_tickers = (
        int(stored_selection.get("requested_max_tickers", -1))
        if isinstance(stored_selection, dict)
        else -1
    )
    requested_max_tickers = max(0, int(args.max_tickers or 0))
    if stored_max_tickers != requested_max_tickers:
        raise ValueError("relabel-only 不可改變 ticker coverage；請執行完整 build-dataset")
    if summary.get("feature_cache_policy") != policy.feature_cache_manifest_payload():
        raise ValueError("feature cache policy 已變更，不能 relabel-only；請執行完整 build-dataset")
    if int(policy.label_horizon_bars) > int(policy.label_path_cache_bars):
        raise ValueError("label horizon 超過 future path cache，不能 relabel-only")

    load_validated_dataset_bundle(args.filter_id, require_current_source=True)
    anchor_prices = load_npy(paths.group_anchor_prices)
    future_high = load_npy(paths.future_high_prices)
    future_low = load_npy(paths.future_low_prices)
    available = load_npy(paths.future_available_bars)
    date_ordinals = load_npy(paths.future_date_ordinals)
    event_group_index = np.asarray(load_npy(paths.event_group_index), dtype=np.int64)
    events = read_breakout_quality_csv(paths.events)
    group_count = int(len(future_high))

    group_labels = np.empty((group_count,), dtype=np.int8)
    reasons: list[str] = []
    max_upside = np.empty((group_count,), dtype=np.float64)
    max_downside = np.empty((group_count,), dtype=np.float64)
    decision_mfe = np.empty((group_count,), dtype=np.float64)
    decision_mae = np.empty((group_count,), dtype=np.float64)
    decision_ratio = np.empty((group_count,), dtype=np.float64)
    first_hit = np.empty((group_count,), dtype=np.float64)
    eval_start_dates: list[str | None] = []
    eval_end_dates: list[str | None] = []
    horizon = int(policy.label_horizon_bars)
    for group_index in range(group_count):
        result = label_from_cached_path(
            future_high[group_index],
            future_low[group_index],
            anchor_price=float(anchor_prices[group_index]),
            available_bars=int(available[group_index]),
            policy=policy,
        )
        group_labels[group_index] = int(result.label)
        reasons.append(result.reason)
        max_upside[group_index] = result.max_upside_return
        max_downside[group_index] = result.max_downside_return
        decision_mfe[group_index] = result.decision_mfe_return
        decision_mae[group_index] = result.decision_mae_return
        decision_ratio[group_index] = result.decision_reward_risk_ratio
        first_hit[group_index] = result.first_hit_bar
        eval_start_dates.append(_iso_date_from_ordinal(date_ordinals[group_index, 0]))
        eval_end_dates.append(
            _iso_date_from_ordinal(date_ordinals[group_index, horizon - 1])
            if int(available[group_index]) >= horizon
            else None
        )

    event_labels = group_labels[event_group_index]
    events["label_eval_start_date"] = [eval_start_dates[idx] for idx in event_group_index]
    events["label_eval_end_date"] = [eval_end_dates[idx] for idx in event_group_index]
    events["label"] = event_labels.astype(np.int8)
    event_reasons = np.asarray([reasons[idx] for idx in event_group_index], dtype=object)
    events["label_reason"] = event_reasons
    event_anchors = np.asarray(anchor_prices[event_group_index], dtype=np.float64)
    display_anchors = event_anchors.copy()
    display_anchors[event_reasons == "insufficient_future"] = np.nan
    events["anchor_price"] = display_anchors
    events["pass_barrier_price"] = display_anchors * (1.0 + float(policy.min_mfe_return))
    events["reject_barrier_price"] = display_anchors * (1.0 + float(policy.max_adverse_return))
    events["max_upside_return"] = max_upside[event_group_index]
    events["max_downside_return"] = max_downside[event_group_index]
    events["decision_mfe_return"] = decision_mfe[event_group_index]
    events["decision_mae_return"] = decision_mae[event_group_index]
    events["decision_reward_risk_ratio"] = decision_ratio[event_group_index]
    events["first_hit_bar"] = first_hit[event_group_index]
    events = events.reindex(columns=list(EVENT_COLUMNS))

    save_npy_atomic(paths.event_labels, event_labels.astype(np.int8))
    _write_events_atomic(paths.events, events)

    summary["policy"] = policy.as_manifest_payload()
    summary["label_policy"] = policy.label_manifest_payload()
    summary["label_counts"] = label_counts(event_labels)
    summary["event_group_summary"] = event_group_summary(events, event_labels)
    summary["label_information_end_date_range"] = _date_range(events, "label_eval_end_date")
    summary["dataset_artifacts"] = _build_artifact_records(paths)
    summary["build_mode"] = "label_only_refresh"
    summary["elapsed_sec"] = round(time.perf_counter() - started, 3)
    summary["relabel_count"] = int(summary.get("relabel_count", 0)) + 1
    write_json(paths.summary, summary)
    print(
        f"已完成 label-only refresh: events={len(event_labels)} groups={group_count} "
        f"labels={label_counts(event_labels)}"
    )
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    policy = build_policy_from_args(args)
    if bool(args.relabel_only):
        return _relabel_only(args, policy, started=started)
    return _full_build(args, policy, started=started)


if __name__ == "__main__":
    raise SystemExit(main())
