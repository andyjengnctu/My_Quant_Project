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
            raw_df = pd.read_csv(path)
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


__all__ = [
    "PROJECT_ROOT",
    "add_policy_args",
    "build_policy_from_args",
    "dataset_npz_path",
    "dataset_output_dir",
    "events_csv_path",
    "label_counts",
    "load_dataset_frames",
    "model_dir",
    "read_json",
    "scores_csv_path",
    "write_json",
]
