"""Shared CLI helpers for breakout quality filter tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.data_utils import discover_unique_csv_inputs, sanitize_ohlcv_dataframe
from core.dataset_profiles import get_dataset_dir, normalize_dataset_profile_key
from filters.breakout_quality.contract import DEFAULT_FILTER_ID, DEFAULT_LABEL_POLICY, BreakoutQualityLabelPolicy
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.paths import ensure_filter_output_dir, resolve_filter_model_dir

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_project_root_to_path() -> None:
    root_text = str(PROJECT_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def build_policy_from_args(args) -> BreakoutQualityLabelPolicy:
    return BreakoutQualityLabelPolicy(
        feature_window_bars=int(args.feature_window),
        label_horizon_bars=int(args.label_horizon),
        high_len_min=int(args.high_len_min),
        high_len_max=int(args.high_len_max),
        high_len_step=int(args.high_len_step),
        label_atr_len=int(args.label_atr_len),
        label_atr_buy_tol=float(args.label_atr_buy_tol),
        label_atr_times_init=float(args.label_atr_times_init),
        positive_mfe_r=float(args.positive_mfe_r),
        negative_mae_r=float(args.negative_mae_r),
        reject_confirm_mfe_r=float(args.reject_confirm_mfe_r),
        dead_mfe_r=float(args.dead_mfe_r),
        evaluate_from_bars_after_entry=int(args.evaluate_from_bars_after_entry),
        benchmark_ticker=str(args.benchmark_ticker),
    )


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    p = DEFAULT_LABEL_POLICY
    parser.add_argument("--feature-window", type=int, default=p.feature_window_bars)
    parser.add_argument("--label-horizon", type=int, default=p.label_horizon_bars)
    parser.add_argument("--high-len-min", type=int, default=p.high_len_min)
    parser.add_argument("--high-len-max", type=int, default=p.high_len_max)
    parser.add_argument("--high-len-step", type=int, default=p.high_len_step)
    parser.add_argument("--label-atr-len", type=int, default=p.label_atr_len)
    parser.add_argument("--label-atr-buy-tol", type=float, default=p.label_atr_buy_tol)
    parser.add_argument("--label-atr-times-init", type=float, default=p.label_atr_times_init)
    parser.add_argument("--positive-mfe-r", type=float, default=p.positive_mfe_r)
    parser.add_argument("--negative-mae-r", type=float, default=p.negative_mae_r)
    parser.add_argument("--reject-confirm-mfe-r", type=float, default=p.reject_confirm_mfe_r)
    parser.add_argument("--dead-mfe-r", type=float, default=p.dead_mfe_r)
    parser.add_argument("--evaluate-from-bars-after-entry", type=int, default=p.evaluate_from_bars_after_entry)
    parser.add_argument("--benchmark-ticker", default=p.benchmark_ticker)


def load_dataset_frames(project_root: Path, dataset: str, *, min_rows: int = 50) -> dict[str, pd.DataFrame]:
    profile = normalize_dataset_profile_key(dataset)
    data_dir = get_dataset_dir(str(project_root), profile)
    csv_inputs, duplicate_lines = discover_unique_csv_inputs(data_dir)
    if duplicate_lines:
        for line in duplicate_lines:
            print(line)
    frames: dict[str, pd.DataFrame] = {}
    for ticker, path in csv_inputs:
        try:
            raw_df = pd.read_csv(path, low_memory=False)
            df, _stats = sanitize_ohlcv_dataframe(raw_df, ticker=ticker, min_rows=min_rows)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            print(f"[skip] {ticker}: {exc}")
            continue
        frames[str(ticker)] = df
    return frames


def dataset_output_dir(filter_id: str) -> Path:
    return ensure_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)


def dataset_npz_path(filter_id: str) -> Path:
    return dataset_output_dir(filter_id) / "dataset.npz"


def events_csv_path(filter_id: str) -> Path:
    return dataset_output_dir(filter_id) / "events.csv"


def scores_csv_path(filter_id: str) -> Path:
    return dataset_output_dir(filter_id) / "scores.csv"


def model_dir(filter_id: str) -> Path:
    path = resolve_filter_model_dir(str(PROJECT_ROOT), filter_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def label_counts(labels: Iterable[int]) -> dict[str, int]:
    arr = np.asarray(list(labels), dtype=np.int64)
    return {
        "pass": int((arr == 1).sum()),
        "reject": int((arr == 0).sum()),
        "ignore": int((arr == -1).sum()),
        "total": int(arr.size),
    }


def event_group_keys(events: pd.DataFrame) -> pd.Series:
    missing = [col for col in ("ticker", "date") if col not in events.columns]
    if missing:
        raise KeyError(f"events.csv 缺少 group 欄位: {missing}")
    return events["ticker"].astype(str) + "\x1f" + events["date"].astype(str)


def group_size_weights(events: pd.DataFrame, indices: Iterable[int]) -> np.ndarray:
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return np.empty((0,), dtype=np.float32)
    keys = event_group_keys(events).iloc[idx].reset_index(drop=True)
    counts = keys.value_counts(sort=False)
    weights = keys.map(lambda key: 1.0 / float(counts[key])).to_numpy(dtype=np.float32)
    return weights


def chronological_group_split_indices(
    events: pd.DataFrame,
    labels: np.ndarray,
    *,
    val_ratio: float,
    label_pass: int,
    label_reject: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    valid_idx = np.flatnonzero((labels_arr == int(label_pass)) | (labels_arr == int(label_reject)))
    if valid_idx.size == 0:
        report = {
            "strategy": "ticker_date_group_chronological",
            "valid_row_count": 0,
            "group_count": 0,
            "train_row_count": 0,
            "val_row_count": 0,
            "train_group_count": 0,
            "val_group_count": 0,
            "overlap_group_count": 0,
        }
        return valid_idx, valid_idx, report

    valid_events = events.iloc[valid_idx].copy()
    valid_events["_row_index"] = valid_idx
    valid_events["_group_key"] = event_group_keys(events).iloc[valid_idx].to_numpy()
    groups = valid_events[["ticker", "date", "_group_key"]].drop_duplicates("_group_key").copy()
    groups["_date_sort"] = pd.to_datetime(groups["date"], errors="coerce")
    groups["_date_text"] = groups["date"].astype(str)
    groups = groups.sort_values(["_date_sort", "_date_text", "ticker"], kind="mergesort").reset_index(drop=True)

    group_count = int(len(groups))
    if group_count <= 1:
        train_groups = set(groups["_group_key"].tolist())
        val_groups: set[str] = set()
    else:
        split = int(round(group_count * (1.0 - float(val_ratio))))
        split = min(max(split, 1), group_count - 1)
        train_groups = set(groups.iloc[:split]["_group_key"].tolist())
        val_groups = set(groups.iloc[split:]["_group_key"].tolist())

    train_mask = valid_events["_group_key"].isin(train_groups).to_numpy()
    val_mask = valid_events["_group_key"].isin(val_groups).to_numpy()
    train_idx = valid_events.loc[train_mask, "_row_index"].to_numpy(dtype=np.int64)
    val_idx = valid_events.loc[val_mask, "_row_index"].to_numpy(dtype=np.int64)
    overlap = train_groups.intersection(val_groups)

    def _date_range(idx: np.ndarray) -> dict[str, str | None]:
        if idx.size == 0:
            return {"start": None, "end": None}
        dates = pd.to_datetime(events.iloc[idx]["date"], errors="coerce")
        if dates.notna().any():
            return {
                "start": str(dates.min().date()),
                "end": str(dates.max().date()),
            }
        texts = events.iloc[idx]["date"].astype(str)
        return {"start": str(texts.min()), "end": str(texts.max())}

    report = {
        "strategy": "ticker_date_group_chronological",
        "val_ratio": float(val_ratio),
        "valid_row_count": int(valid_idx.size),
        "group_count": group_count,
        "train_row_count": int(train_idx.size),
        "val_row_count": int(val_idx.size),
        "train_group_count": int(len(train_groups)),
        "val_group_count": int(len(val_groups)),
        "overlap_group_count": int(len(overlap)),
        "train_date_range": _date_range(train_idx),
        "val_date_range": _date_range(val_idx),
    }
    return train_idx, val_idx, report


def event_group_summary(events: pd.DataFrame, labels: Iterable[int]) -> dict:
    if events.empty:
        return {"group_count": 0}
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    if labels_arr.size != len(events):
        raise ValueError(f"labels 長度與 events 不一致: labels={labels_arr.size}, events={len(events)}")
    frame = events[["ticker", "date"]].copy()
    frame["label"] = labels_arr
    frame["_group_key"] = event_group_keys(events).to_numpy()
    group_sizes = frame.groupby("_group_key", sort=False).size()
    label_nunique = frame.groupby("_group_key", sort=False)["label"].nunique()
    valid = frame[frame["label"].isin([0, 1])].copy()
    valid_group_count = int(valid["_group_key"].nunique()) if not valid.empty else 0
    valid_label_nunique = valid.groupby("_group_key", sort=False)["label"].nunique() if not valid.empty else pd.Series(dtype="int64")
    return {
        "group_key": "ticker/date",
        "group_count": int(group_sizes.size),
        "valid_group_count": valid_group_count,
        "rows_per_group_min": int(group_sizes.min()) if group_sizes.size else 0,
        "rows_per_group_max": int(group_sizes.max()) if group_sizes.size else 0,
        "rows_per_group_mean": round(float(group_sizes.mean()), 6) if group_sizes.size else 0.0,
        "mixed_label_group_count": int((label_nunique > 1).sum()) if not label_nunique.empty else 0,
        "mixed_valid_label_group_count": int((valid_label_nunique > 1).sum()) if not valid_label_nunique.empty else 0,
    }


__all__ = [
    "PROJECT_ROOT",
    "add_policy_args",
    "build_policy_from_args",
    "dataset_npz_path",
    "chronological_group_split_indices",
    "dataset_output_dir",
    "event_group_keys",
    "event_group_summary",
    "events_csv_path",
    "group_size_weights",
    "label_counts",
    "load_dataset_frames",
    "model_dir",
    "read_json",
    "scores_csv_path",
    "write_json",
]
