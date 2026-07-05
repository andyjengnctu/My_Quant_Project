"""Shared CLI helpers for breakout quality filter tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.data_utils import discover_unique_csv_inputs, sanitize_ohlcv_dataframe
from core.dataset_profiles import get_dataset_dir, normalize_dataset_profile_key
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    FEATURE_COLUMNS,
    BreakoutQualityLabelPolicy,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.paths import (
    ensure_filter_model_dir,
    ensure_filter_output_dir,
    resolve_filter_artifact_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_project_root_to_path() -> None:
    root_text = str(PROJECT_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _parse_high_len_values(raw_value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--high-lens 不可為空")
    values = tuple(sorted({int(token) for token in tokens}))
    if values[0] < 1:
        raise ValueError("--high-lens 只能包含正整數")
    return values


def build_policy_from_args(args) -> BreakoutQualityLabelPolicy:
    legacy_values = (args.high_len_min, args.high_len_max, args.high_len_step)
    if any(value is not None for value in legacy_values):
        if not all(value is not None for value in legacy_values):
            raise ValueError("使用舊式 high_len range 參數時，--high-len-min/--high-len-max/--high-len-step 必須同時提供")
        low = int(args.high_len_min)
        high = int(args.high_len_max)
        step = int(args.high_len_step)
        if low < 1 or high < low or step < 1:
            raise ValueError("high_len range 必須滿足 min>=1、max>=min、step>=1")
        high_len_values = tuple(range(low, high + 1, step))
    else:
        high_len_values = _parse_high_len_values(args.high_lens)

    policy = BreakoutQualityLabelPolicy(
        feature_window_bars=int(args.feature_window),
        label_horizon_bars=int(args.label_horizon),
        high_len_values=high_len_values,
        label_atr_len=int(args.label_atr_len),
        label_atr_buy_tol=float(args.label_atr_buy_tol),
        label_atr_times_init=float(args.label_atr_times_init),
        positive_mfe_r=float(args.positive_mfe_r),
        negative_mae_r=float(args.negative_mae_r),
        reject_confirm_mfe_r=float(args.reject_confirm_mfe_r),
        dead_mfe_r=float(args.dead_mfe_r),
        evaluate_from_bars_after_entry=int(args.evaluate_from_bars_after_entry),
        benchmark_ticker=str(args.benchmark_ticker).strip(),
    )
    if policy.feature_window_bars < 1 or policy.label_horizon_bars < 1 or policy.label_atr_len < 1:
        raise ValueError("feature window、label horizon 與 ATR length 必須 >= 1")
    if policy.label_atr_buy_tol < 0 or policy.label_atr_times_init <= 0:
        raise ValueError("label ATR buy tolerance 必須 >= 0，initial stop ATR 倍數必須 > 0")
    if policy.positive_mfe_r <= 0 or policy.negative_mae_r >= 0:
        raise ValueError("positive_mfe_r 必須 > 0，negative_mae_r 必須 < 0")
    if policy.reject_confirm_mfe_r < 0 or policy.reject_confirm_mfe_r > policy.positive_mfe_r:
        raise ValueError("reject_confirm_mfe_r 必須介於 0 與 positive_mfe_r")
    if policy.dead_mfe_r < 0 or policy.dead_mfe_r > policy.positive_mfe_r:
        raise ValueError("dead_mfe_r 必須介於 0 與 positive_mfe_r")
    if policy.evaluate_from_bars_after_entry < 0:
        raise ValueError("evaluate_from_bars_after_entry 必須 >= 0")
    if not policy.benchmark_ticker:
        raise ValueError("benchmark_ticker 不可空白")
    policy.high_lens()
    return policy


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    p = DEFAULT_LABEL_POLICY
    parser.add_argument("--feature-window", type=int, default=p.feature_window_bars)
    parser.add_argument("--label-horizon", type=int, default=p.label_horizon_bars)
    parser.add_argument(
        "--high-lens",
        default=",".join(str(value) for value in p.high_lens()),
        help="逗號分隔的正式 high_len coverage；預設同時覆蓋 optimizer 搜尋值與策略預設值",
    )
    parser.add_argument("--high-len-min", type=int, default=None, help="相容舊命令；使用時需與 max/step 同時提供")
    parser.add_argument("--high-len-max", type=int, default=None, help="相容舊命令；使用時需與 min/step 同時提供")
    parser.add_argument("--high-len-step", type=int, default=None, help="相容舊命令；使用時需與 min/max 同時提供")
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
    return resolve_filter_artifact_paths(PROJECT_ROOT, filter_id).score_path


def model_dir(filter_id: str) -> Path:
    return ensure_filter_model_dir(PROJECT_ROOT, filter_id)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根節點必須是 object: {path}")
    return payload



def load_validated_dataset_bundle(
    filter_id: str,
    *,
    expected_policy: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    summary_path = dataset_output_dir(filter_id) / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 dataset_summary.json: {summary_path}")
    summary = read_json(summary_path)
    if str(summary.get("filter_id", "")).strip() != str(filter_id).strip():
        raise ValueError("dataset_summary.filter_id 與命令不一致")
    if list(summary.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("dataset_summary feature_columns 與正式 contract 不一致")
    if list(summary.get("context_columns", [])) != list(CONTEXT_COLUMNS):
        raise ValueError("dataset_summary context_columns 與正式 contract 不一致")
    policy = summary.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("dataset_summary 缺少 policy object")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError(
            "目前 dataset policy 與 model manifest policy 不一致；不可用不同 feature/label/high_len 契約匯出同一模型分數"
        )

    artifact_records = summary.get("dataset_artifacts")
    if not isinstance(artifact_records, dict):
        raise ValueError("dataset_summary 缺少 dataset_artifacts；請用新版 build_dataset 重建")
    expected_paths = {
        "dataset_npz": dataset_npz_path(filter_id),
        "events_csv": events_csv_path(filter_id),
    }
    for artifact_name, artifact_path in expected_paths.items():
        record = artifact_records.get(artifact_name)
        if not isinstance(record, dict):
            raise ValueError(f"dataset_summary 缺少 artifact record: {artifact_name}")
        if str(record.get("filename", "")).strip() != artifact_path.name:
            raise ValueError(f"dataset artifact filename 不一致: {artifact_name}")
        expected_size = int(record.get("size_bytes", -1))
        actual_size = int(artifact_path.stat().st_size)
        if expected_size != actual_size:
            raise ValueError(
                f"dataset artifact size 不一致: {artifact_name}; expected={expected_size}, actual={actual_size}"
            )
        expected_hash = str(record.get("sha256", "")).strip().lower()
        actual_hash = compute_file_sha256(artifact_path).lower()
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(
                f"dataset artifact SHA256 不一致: {artifact_name}; expected={expected_hash}, actual={actual_hash}"
            )

    with np.load(expected_paths["dataset_npz"]) as data:
        required_arrays = {"features", "context", "labels"}
        missing_arrays = sorted(required_arrays - set(data.files))
        if missing_arrays:
            raise ValueError(f"dataset.npz 缺少 arrays: {missing_arrays}")
        features = data["features"].astype(np.float32)
        context = data["context"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
    events = read_breakout_quality_csv(expected_paths["events_csv"])
    if features.ndim != 3 or context.ndim != 2 or labels.ndim != 1:
        raise ValueError(f"dataset shape 不合法: X={features.shape}, C={context.shape}, y={labels.shape}")
    if len(features) != len(context) or len(features) != len(labels) or len(features) != len(events):
        raise ValueError(
            f"dataset bundle 長度不一致: X={len(features)}, C={len(context)}, y={len(labels)}, events={len(events)}"
        )
    if features.shape[2] != len(FEATURE_COLUMNS) or context.shape[1] != len(CONTEXT_COLUMNS):
        raise ValueError("dataset feature/context 維度與正式 contract 不一致")
    feature_window_bars = int(policy.get("feature_window_bars", -1))
    if feature_window_bars < 1 or features.shape[1] != feature_window_bars:
        raise ValueError(
            f"dataset feature window 與 policy 不一致: X={features.shape[1]}, policy={feature_window_bars}"
        )
    if not np.isfinite(features).all() or not np.isfinite(context).all():
        raise ValueError("dataset features/context 含 NaN 或 infinite")
    required_event_columns = {"ticker", "date", "high_len", "label", "label_eval_end_date"}
    missing_event_columns = sorted(required_event_columns - set(events.columns))
    if missing_event_columns:
        raise ValueError(f"events.csv 缺少必要欄位: {missing_event_columns}")
    event_labels = pd.to_numeric(events["label"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(event_labels, labels):
        raise ValueError("events.csv label 與 dataset.npz labels 不一致")
    policy_high_lens = policy.get("high_len_values")
    if not isinstance(policy_high_lens, list) or not policy_high_lens:
        raise ValueError("dataset policy.high_len_values 必須是非空 list")
    allowed_high_lens = {int(value) for value in policy_high_lens}
    observed_high_lens = set(pd.to_numeric(events["high_len"], errors="raise").astype(int).tolist())
    if not observed_high_lens.issubset(allowed_high_lens):
        raise ValueError("events.csv 含 policy 未宣告的 high_len")
    if int(summary.get("event_count", -1)) != len(events):
        raise ValueError("dataset_summary.event_count 與 artifacts 不一致")
    return summary, features, context, labels, events

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
    return keys.map(lambda key: 1.0 / float(counts[key])).to_numpy(dtype=np.float32)


def chronological_group_split_indices(
    events: pd.DataFrame,
    labels: np.ndarray,
    *,
    val_ratio: float,
    label_pass: int,
    label_reject: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ratio = float(val_ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"val_ratio 必須介於 0 與 1，收到 {val_ratio!r}")

    labels_arr = np.asarray(labels, dtype=np.int64)
    valid_idx = np.flatnonzero((labels_arr == int(label_pass)) | (labels_arr == int(label_reject)))
    empty_report = {
        "strategy": "event_date_chronological_with_label_end_embargo",
        "val_ratio": ratio,
        "valid_row_count": int(valid_idx.size),
        "group_count": 0,
        "train_row_count": 0,
        "val_row_count": 0,
        "embargo_dropped_row_count": 0,
        "train_group_count": 0,
        "val_group_count": 0,
        "overlap_group_count": 0,
        "overlap_event_date_count": 0,
    }
    if valid_idx.size == 0:
        return valid_idx, valid_idx, empty_report

    required = {"ticker", "date", "label_eval_end_date"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(
            f"breakout quality chronological split 缺少欄位 {missing}；請用新版 build_dataset 重建資料，禁止無 embargo 訓練"
        )

    valid_events = events.iloc[valid_idx].copy()
    valid_events["_row_index"] = valid_idx
    valid_events["_group_key"] = event_group_keys(events).iloc[valid_idx].to_numpy()
    valid_events["_event_date"] = pd.to_datetime(valid_events["date"], errors="raise").dt.normalize()
    valid_events["_label_end_date"] = pd.to_datetime(valid_events["label_eval_end_date"], errors="raise").dt.normalize()

    unique_dates = np.sort(valid_events["_event_date"].unique())
    if len(unique_dates) <= 1:
        train_mask = np.ones(len(valid_events), dtype=bool)
        val_mask = np.zeros(len(valid_events), dtype=bool)
        embargo_mask = np.zeros(len(valid_events), dtype=bool)
        val_start = None
    else:
        split = int(round(len(unique_dates) * (1.0 - ratio)))
        split = min(max(split, 1), len(unique_dates) - 1)
        val_start = pd.Timestamp(unique_dates[split])
        before_val = valid_events["_event_date"] < val_start
        safe_label_end = valid_events["_label_end_date"] < val_start
        train_mask = (before_val & safe_label_end).to_numpy(dtype=bool)
        embargo_mask = (before_val & ~safe_label_end).to_numpy(dtype=bool)
        val_mask = (valid_events["_event_date"] >= val_start).to_numpy(dtype=bool)

    train_idx = valid_events.loc[train_mask, "_row_index"].to_numpy(dtype=np.int64)
    val_idx = valid_events.loc[val_mask, "_row_index"].to_numpy(dtype=np.int64)
    train_groups = set(valid_events.loc[train_mask, "_group_key"].tolist())
    val_groups = set(valid_events.loc[val_mask, "_group_key"].tolist())
    train_dates = set(valid_events.loc[train_mask, "_event_date"].tolist())
    val_dates = set(valid_events.loc[val_mask, "_event_date"].tolist())

    def _date_range(idx: np.ndarray, column: str = "date") -> dict[str, str | None]:
        if idx.size == 0:
            return {"start": None, "end": None}
        dates = pd.to_datetime(events.iloc[idx][column], errors="raise")
        return {"start": str(dates.min().date()), "end": str(dates.max().date())}

    report = {
        **empty_report,
        "group_count": int(valid_events["_group_key"].nunique()),
        "event_date_count": int(len(unique_dates)),
        "validation_start_date": str(val_start.date()) if val_start is not None else None,
        "train_row_count": int(train_idx.size),
        "val_row_count": int(val_idx.size),
        "embargo_dropped_row_count": int(np.asarray(embargo_mask, dtype=bool).sum()),
        "train_group_count": int(len(train_groups)),
        "val_group_count": int(len(val_groups)),
        "overlap_group_count": int(len(train_groups.intersection(val_groups))),
        "overlap_event_date_count": int(len(train_dates.intersection(val_dates))),
        "train_date_range": _date_range(train_idx),
        "train_label_end_date_range": _date_range(train_idx, "label_eval_end_date"),
        "val_date_range": _date_range(val_idx),
        "val_label_end_date_range": _date_range(val_idx, "label_eval_end_date"),
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
    "chronological_group_split_indices",
    "dataset_npz_path",
    "dataset_output_dir",
    "event_group_keys",
    "event_group_summary",
    "events_csv_path",
    "group_size_weights",
    "label_counts",
    "load_dataset_frames",
    "load_validated_dataset_bundle",
    "model_dir",
    "read_breakout_quality_csv",
    "read_json",
    "scores_csv_path",
    "write_json",
]
