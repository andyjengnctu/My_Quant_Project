"""Formal read-only data loader shared by continuous-ranker workflows and strategy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_experiment_profile
from filters.breakout_quality.continuous_target import load_validated_continuous_target_arrays
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.models.spec import get_model_spec, validate_model_sequence_length
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from filters.breakout_quality.workflow_io import PROJECT_ROOT, load_validated_dataset_bundle


@dataclass(frozen=True)
class ContinuousRankerDataBundle:
    summary: dict[str, Any]
    events: pd.DataFrame
    labels: np.ndarray
    feature_bank: np.ndarray
    event_group_index: np.ndarray
    group_table: pd.DataFrame
    group_context: np.ndarray
    raw_target: np.ndarray
    target_valid: np.ndarray
    target_manifest: dict[str, Any]
    outer_policy: dict[str, Any]
    profile: Any
    model_spec: Any


def build_same_date_percentile_targets(
    values: np.ndarray,
    valid_mask: np.ndarray,
    group_dates: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return [0,1] average-rank percentiles using only values from the same date."""

    target = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    dates = pd.to_datetime(pd.Series(group_dates), errors="raise").dt.normalize()
    if target.ndim != 1 or valid.ndim != 1 or target.shape != valid.shape or len(dates) != len(target):
        raise ValueError("same-date percentile target input shape不一致")
    result = np.full(target.shape, np.nan, dtype=np.float32)
    work = pd.DataFrame(
        {
            "date": dates,
            "target": target,
            "group_index": np.arange(len(target), dtype=np.int64),
        }
    )
    work = work[valid & np.isfinite(target)].copy()
    for _date, day in work.groupby("date", sort=True):
        count = int(len(day))
        if count == 1:
            percentile = np.array([0.5], dtype=np.float64)
        else:
            ranks = day["target"].rank(method="average").to_numpy(dtype=np.float64)
            percentile = (ranks - 1.0) / float(count - 1)
        result[day["group_index"].to_numpy(dtype=np.int64)] = percentile.astype(np.float32)
    if bool(np.any(valid & ~np.isfinite(result))):
        raise ValueError("valid target無法建立same-date percentile")
    if bool(np.any(np.isfinite(result) & ((result < 0.0) | (result > 1.0)))):
        raise ValueError("same-date percentile超出[0,1]")
    return result


def _validate_group_consistency(events: pd.DataFrame) -> None:
    required = {"ticker", "date", "group_index", "label_eval_end_date"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"continuous ranker dataset 缺少欄位: {missing}")
    work = events[["group_index", "ticker", "date", "label_eval_end_date"]].copy()
    grouped = work.groupby("group_index", sort=False)
    mixed = grouped.agg(
        ticker_count=("ticker", "nunique"),
        date_count=("date", "nunique"),
    )
    mixed["label_end_count"] = grouped["label_eval_end_date"].nunique(dropna=False)
    invalid = mixed[
        (mixed["ticker_count"] != 1)
        | (mixed["date_count"] != 1)
        | (mixed["label_end_count"] != 1)
    ]
    if not invalid.empty:
        raise ValueError(
            "continuous ranker 同group的ticker/date/label_eval_end_date必須一致: "
            f"invalid_groups={len(invalid)}"
        )


def _group_table(events: pd.DataFrame, event_group_index: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    frame = events[["ticker", "date", "group_index"]].copy()
    frame["event_row"] = np.arange(len(frame), dtype=np.int64)
    frame["label"] = np.asarray(labels, dtype=np.int64)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    group = frame.drop_duplicates("group_index", keep="first").sort_values("group_index", kind="mergesort")
    expected = np.arange(len(group), dtype=np.int64)
    observed = pd.to_numeric(group["group_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("continuous ranker要求group_index連續完整")
    representative_rows = group["event_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        np.asarray(event_group_index[representative_rows], dtype=np.int64),
        expected,
    ):
        raise ValueError("continuous ranker group representative與event_group_index不一致")
    mixed = frame.groupby("group_index", sort=False)["label"].nunique()
    if bool((mixed != 1).any()):
        raise ValueError("continuous ranker發現同group混合binary label")
    return group.reset_index(drop=True)


def source_data_end(summary: dict[str, Any], events: pd.DataFrame) -> str:
    source_range = summary.get("source_data_date_range")
    if isinstance(source_range, dict):
        value = str(source_range.get("end") or "").strip()
        if value:
            return value
    return str(pd.to_datetime(events["label_eval_end_date"], errors="raise").max().date())


def load_continuous_ranker_data(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    preload_feature_bank: bool,
    allow_stale_source: bool,
    project_root: str | Path = PROJECT_ROOT,
) -> ContinuousRankerDataBundle:
    """Load the validated dataset/target bundle without importing training tools."""

    profile = get_breakout_quality_experiment_profile(experiment_profile)
    model_spec = get_model_spec(model_architecture)
    if (
        bool(model_spec.requires_market_set)
        or bool(model_spec.use_dataset_context)
        or bool(model_spec.derived_context_features)
    ):
        raise ValueError("continuous ranker data loader只支援 sequence-only architecture")

    summary, indexed_features, context, labels, events = load_validated_dataset_bundle(
        filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    _validate_group_consistency(events)
    feature_bank = (
        np.array(indexed_features.feature_bank, dtype=np.float32, copy=True, order="C")
        if bool(preload_feature_bank)
        else indexed_features.feature_bank
    )
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    group_table = _group_table(events, event_group_index, labels_array)
    representative_rows = group_table["event_row"].to_numpy(dtype=np.int64)
    group_table = group_table.copy()
    group_table["label_eval_end_date"] = pd.to_datetime(
        events.iloc[representative_rows]["label_eval_end_date"], errors="raise"
    ).dt.normalize().to_numpy()
    group_context = np.asarray(context[representative_rows], dtype=np.float32)
    if bool(preload_feature_bank):
        group_context = np.array(group_context, dtype=np.float32, copy=True, order="C")
    validate_model_sequence_length(model_spec, int(feature_bank.shape[1]))

    target_manifest, raw_target, target_valid = load_validated_continuous_target_arrays(
        Path(project_root),
        filter_id,
        target_id=str(profile.continuous_target_id),
        expected_group_count=int(len(group_table)),
        expected_dataset_policy=summary.get("policy"),
        expected_dataset_artifacts=summary.get("dataset_artifacts"),
    )
    outer_policy = resolve_breakout_quality_outer_policy(
        Path(project_root),
        source_data_end_date=source_data_end(summary, events),
    )
    return ContinuousRankerDataBundle(
        summary=dict(summary),
        events=events.copy(),
        labels=labels_array,
        feature_bank=feature_bank,
        event_group_index=event_group_index,
        group_table=group_table,
        group_context=group_context,
        raw_target=np.asarray(raw_target, dtype=np.float32),
        target_valid=np.asarray(target_valid, dtype=bool),
        target_manifest=dict(target_manifest),
        outer_policy=dict(outer_policy),
        profile=profile,
        model_spec=model_spec,
    )


__all__ = ["ContinuousRankerDataBundle", "build_same_date_percentile_targets", "load_continuous_ranker_data"]
