"""Shared Selection replay Audit target/identity primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.rolling_oos_params import get_active_param_date_range
from filters.breakout_quality.continuous_target import STRATEGY_ALIGNED_NO_TIME_TARGET_ID, load_validated_continuous_target_arrays
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.workflow_io import PROJECT_ROOT, load_validated_dataset_bundle

SELECTION_STRATEGY_REALIZATION_AUDIT_DIRNAME = "selection_strategy_realization_audit"

def selection_strategy_realization_output_dir(filter_id: str):
    return resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id) / "continuous_targets" / STRATEGY_ALIGNED_NO_TIME_TARGET_ID / SELECTION_STRATEGY_REALIZATION_AUDIT_DIRNAME

def date_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return pd.Timestamp(text).strftime("%Y-%m-%d")

def validate_param_coverage(source: dict[str, Any], *, start_date: str, end_date: str) -> tuple[str, str, str]:
    kind = str(source.get("kind") or "")
    payload = source.get("payload") or {}
    if kind == "rolling_active_param_ensemble":
        first, last = get_active_param_ensemble_date_range(payload)
    elif kind == "rolling_oos_param_schedule":
        first, last = get_active_param_date_range(payload)
    else:
        raise ValueError(f"11I只接受rolling lookahead-safe params，收到: {kind}")
    if date_text(first) > date_text(start_date) or date_text(last) < date_text(end_date):
        raise ValueError(
            "11I nested params未完整覆蓋Selection replay期間: "
            f"params={first}~{last}, requested={start_date}~{end_date}"
        )
    return kind, date_text(first), date_text(last)

def resolve_target_date(frame: pd.DataFrame) -> pd.Series:
    signal = frame.get("signal_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    candidate = frame.get("candidate_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    entry = frame.get("entry_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    trade = frame.get("trade_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    target = signal.where(signal != "", candidate)
    target = target.where(target != "", entry)
    target = target.where(target != "", trade)
    return pd.to_datetime(target, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

def target_lookup(filter_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary, _bank, labels, event_group_index, events = load_validated_dataset_bundle(filter_id)
    group_count = int(summary.get("feature_group_count", 0) or len(np.unique(event_group_index)))
    manifest, target, valid = load_validated_continuous_target_arrays(
        PROJECT_ROOT,
        filter_id,
        target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        expected_group_count=group_count,
        expected_dataset_policy=summary.get("policy"),
    )
    frame = events[["ticker", "date", "group_index", "label"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype(np.int64)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    mixed = frame.groupby("group_index", sort=False)["label"].nunique()
    if bool((mixed != 1).any()):
        raise ValueError("11I dataset同group存在混合label")
    group = frame.drop_duplicates("group_index", keep="first").sort_values("group_index", kind="mergesort")
    if len(group) != len(target):
        raise ValueError("11I group lookup與No-time target長度不一致")
    ids = group["group_index"].to_numpy(dtype=np.int64)
    group["target_raw_r"] = np.asarray(target, dtype=np.float64)[ids]
    group["target_valid"] = np.asarray(valid, dtype=bool)[ids]
    return group.rename(columns={"date": "target_date"}).reset_index(drop=True), manifest

def attach_targets(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ticker"] = work["ticker"].astype(str)
    work["target_date"] = resolve_target_date(work)
    merged = work.merge(
        lookup[["ticker", "target_date", "group_index", "label", "target_raw_r", "target_valid"]],
        how="left",
        on=["ticker", "target_date"],
        validate="many_to_one",
    )
    merged["target_match"] = merged["target_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(merged["target_raw_r"], errors="coerce")
    )
    return merged

def unique_signals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        ["ticker", "target_date", "trade_date", "candidate_type"], kind="mergesort"
    ).drop_duplicates(["ticker", "target_date"], keep="first").reset_index(drop=True)

__all__ = ["SELECTION_STRATEGY_REALIZATION_AUDIT_DIRNAME", "selection_strategy_realization_output_dir", "date_text", "validate_param_coverage", "resolve_target_date", "target_lookup", "attach_targets", "unique_signals"]
