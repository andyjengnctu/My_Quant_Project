"""Canonical score table loading and runtime lookup for breakout quality filter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import load_runtime_artifact_contract, validate_required_high_len
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    SCORE_COLUMN,
    SCORE_TABLE_REQUIRED_COLUMNS,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv


def resolve_score_table_path(project_root: str, filter_id: str = DEFAULT_FILTER_ID) -> Path:
    contract = load_runtime_artifact_contract(str(project_root), str(filter_id))
    path = contract.paths.score_path
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 breakout quality 正式 score table: {path}。"
            "正式 runtime 只接受 models/filters/breakout_quality/<filter_id>/<model_architecture>/<experiment_profile>/scores.csv 單一路徑。"
        )
    return path.resolve()


@lru_cache(maxsize=16)
def load_score_table(project_root: str, filter_id: str = DEFAULT_FILTER_ID) -> pd.DataFrame:
    path = resolve_score_table_path(project_root, filter_id=filter_id)
    table = read_breakout_quality_csv(path)
    missing = sorted(set(SCORE_TABLE_REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"breakout quality score table 缺少欄位: {missing}; path={path}")
    table = table.copy()
    contract = load_runtime_artifact_contract(str(project_root), str(filter_id))
    score_metadata = contract.manifest["score_table"]
    expected_columns = list(score_metadata["columns"])
    if list(table.columns) != expected_columns:
        raise ValueError(
            f"breakout quality score table columns 與 manifest 不一致: "
            f"manifest={expected_columns}, actual={list(table.columns)}, path={path}"
        )
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["high_len"] = table["high_len"].astype(int)
    table[SCORE_COLUMN] = pd.to_numeric(table[SCORE_COLUMN], errors="raise").astype(float)
    scores = table[SCORE_COLUMN].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError(f"breakout quality score table 含非有限分數; path={path}")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError(f"breakout quality score 必須介於 0 與 1; path={path}")
    if table.duplicated(["ticker", "date", "high_len"]).any():
        dup_count = int(table.duplicated(["ticker", "date", "high_len"]).sum())
        raise ValueError(f"breakout quality score table 有重複 key: {dup_count}; path={path}")
    observed_start = str(table["date"].min())
    observed_end = str(table["date"].max())
    expected_date_range = score_metadata["event_date_range"]
    if observed_start != str(expected_date_range["start"]) or observed_end != str(expected_date_range["end"]):
        raise ValueError(
            f"breakout quality score table 日期範圍與 manifest 不一致: "
            f"manifest={expected_date_range}, actual={{'start': {observed_start!r}, 'end': {observed_end!r}}}"
        )
    if pd.Timestamp(observed_start).date() < contract.available_from or pd.Timestamp(observed_end).date() > contract.available_through:
        raise ValueError("breakout quality score table 實際日期超出 runtime_eligibility 有效期間")

    indexed = table.set_index(["ticker", "date", "high_len"]).sort_index()
    expected_row_count = int(score_metadata["row_count"])
    if expected_row_count != len(indexed):
        raise ValueError(
            f"breakout quality score table row_count 與 manifest 不一致: "
            f"manifest={expected_row_count}, actual={len(indexed)}"
        )

    unavailable_record = contract.manifest.get("conservative_unscorable_events")
    if unavailable_record is not None:
        unavailable_path = path.with_name(DEFAULT_UNAVAILABLE_SCORE_FILENAME)
        unavailable = read_breakout_quality_csv(unavailable_path)
        expected_unavailable_columns = ["ticker", "date", "high_len", "reason"]
        if list(unavailable.columns) != expected_unavailable_columns:
            raise ValueError(
                "breakout quality unavailable score columns 與 manifest 不一致: "
                f"expected={expected_unavailable_columns}, actual={list(unavailable.columns)}"
            )
        unavailable = unavailable.copy()
        unavailable["ticker"] = unavailable["ticker"].astype(str)
        unavailable["date"] = pd.to_datetime(unavailable["date"], errors="raise").dt.strftime("%Y-%m-%d")
        unavailable["high_len"] = pd.to_numeric(unavailable["high_len"], errors="raise").astype(int)
        unavailable["reason"] = unavailable["reason"].astype(str).str.strip()
        if bool((unavailable["reason"] == "").any()):
            raise ValueError("breakout quality unavailable score reason 不可為空")
        if len(unavailable) != int(unavailable_record["row_count"]):
            raise ValueError("breakout quality unavailable score row_count 與 manifest 不一致")
        if unavailable.duplicated(["ticker", "date", "high_len"]).any():
            raise ValueError("breakout quality unavailable score table 有重複 key")
        actual_reason_counts = {
            str(key): int(value)
            for key, value in unavailable["reason"].value_counts().sort_index().items()
        }
        expected_reason_counts = {
            str(key): int(value)
            for key, value in dict(unavailable_record.get("reason_counts") or {}).items()
        }
        if actual_reason_counts != expected_reason_counts:
            raise ValueError("breakout quality unavailable score reason_counts 與 manifest 不一致")
        if not unavailable.empty:
            unavailable_index = pd.MultiIndex.from_frame(
                unavailable[["ticker", "date", "high_len"]]
            )
            fallback_scores = indexed.reindex(unavailable_index)[SCORE_COLUMN].to_numpy(
                dtype=np.float64, copy=False
            )
            if not np.isfinite(fallback_scores).all() or not bool(np.all(fallback_scores == 0.0)):
                raise ValueError(
                    "breakout quality unavailable event 必須存在於正式 score table 且保守分數為 0.0"
                )

    observed_high_lens = set(int(value) for value in indexed.index.get_level_values("high_len").unique())
    if not observed_high_lens.issubset(set(contract.high_len_values)):
        raise ValueError("breakout quality score table 含 manifest 未宣告的 high_len")
    return indexed


@lru_cache(maxsize=16)
def load_shared_group_score_table(
    project_root: str,
    filter_id: str = DEFAULT_FILTER_ID,
) -> pd.DataFrame:
    """Return the canonical ticker/date score table for sequence-only runtime models."""

    contract = load_runtime_artifact_contract(str(project_root), str(filter_id))
    if not contract.shared_group_score_broadcast:
        raise ValueError(
            "breakout quality artifact 未宣告 shared_group_score_broadcast，"
            "不可改用 ticker/date lookup"
        )

    event_table = load_score_table(str(project_root), str(filter_id)).reset_index()
    score_counts = event_table.groupby(["ticker", "date"], sort=False)[SCORE_COLUMN].nunique(
        dropna=False
    )
    inconsistent = score_counts[score_counts != 1]
    if not inconsistent.empty:
        sample_keys = [
            f"{ticker}/{event_date}"
            for ticker, event_date in list(inconsistent.index[:5])
        ]
        raise ValueError(
            "breakout quality shared group score 同一 ticker/date 出現不一致分數: "
            f"group_count={len(inconsistent)}, sample={sample_keys}"
        )

    shared = (
        event_table.drop_duplicates(["ticker", "date"], keep="first")
        .set_index(["ticker", "date"])[[SCORE_COLUMN]]
        .sort_index()
    )
    return shared


def build_pass_condition_from_score_table(
    df: pd.DataFrame,
    *,
    ticker: str,
    high_len: int,
    score_threshold: float,
    candidate_condition: np.ndarray,
    project_root: str,
    filter_id: str = DEFAULT_FILTER_ID,
) -> np.ndarray:
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("啟用 breakout quality filter 時必須提供 ticker")
    threshold = float(score_threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"breakout_quality_score_threshold 必須介於 0 與 1，收到 {score_threshold!r}")

    out = np.ones(len(df), dtype=bool)
    candidate_mask = np.asarray(candidate_condition, dtype=bool)
    if candidate_mask.shape != (len(df),):
        raise ValueError(
            f"breakout quality candidate_condition shape 不一致: expected={(len(df),)}, actual={candidate_mask.shape}"
        )
    if len(df) == 0:
        return out

    contract = load_runtime_artifact_contract(str(project_root), str(filter_id))
    validate_required_high_len(contract, int(high_len))
    score_table = (
        load_shared_group_score_table(str(project_root), filter_id=str(filter_id))
        if contract.shared_group_score_broadcast
        else load_score_table(str(project_root), filter_id=str(filter_id))
    )
    timestamps = pd.to_datetime(df.index, errors="raise")
    after_coverage_mask = np.asarray(
        [timestamp.date() > contract.available_through for timestamp in timestamps],
        dtype=bool,
    )
    uncovered_candidate_mask = after_coverage_mask & candidate_mask
    if uncovered_candidate_mask.any():
        uncovered_dates = list(timestamps[uncovered_candidate_mask].strftime("%Y-%m-%d")[:5])
        raise ValueError(
            f"breakout quality score table 已過期且出現未覆蓋候選事件: "
            f"available_through={contract.available_through}, filter_id={filter_id}, "
            f"missing_count={int(uncovered_candidate_mask.sum())}, sample_dates={uncovered_dates}"
        )

    active_mask = np.asarray(
        [contract.available_from <= timestamp.date() <= contract.available_through for timestamp in timestamps],
        dtype=bool,
    )
    active_candidate_mask = active_mask & candidate_mask
    if not active_candidate_mask.any():
        return out

    active_dates = timestamps[active_candidate_mask].strftime("%Y-%m-%d")
    if contract.shared_group_score_broadcast:
        keys = pd.MultiIndex.from_arrays(
            [[str(ticker)] * len(active_dates), active_dates],
            names=["ticker", "date"],
        )
    else:
        keys = pd.MultiIndex.from_arrays(
            [[str(ticker)] * len(active_dates), active_dates, [int(high_len)] * len(active_dates)],
            names=["ticker", "date", "high_len"],
        )
    matched = score_table.reindex(keys)
    values = matched[SCORE_COLUMN].to_numpy(dtype=np.float64, copy=False)
    missing_mask = ~np.isfinite(values)
    if missing_mask.any():
        missing_dates = list(active_dates[missing_mask][:5])
        lookup_key = (
            "ticker/date shared group"
            if contract.shared_group_score_broadcast
            else "ticker/date/high_len event"
        )
        raise ValueError(
            f"breakout quality score table 缺少正式候選事件: ticker={ticker}, high_len={high_len}, "
            f"lookup_key={lookup_key}, missing_count={int(missing_mask.sum())}, "
            f"sample_dates={missing_dates}"
        )
    out[active_candidate_mask] = values >= threshold
    return out


__all__ = [
    "build_pass_condition_from_score_table",
    "load_score_table",
    "load_shared_group_score_table",
    "resolve_score_table_path",
]
