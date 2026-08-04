"""Validated point-in-time binary score loading for optimizer research."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import SCORE_COLUMN

BINARY_PIT_SCORE_SOURCE = "binary_point_in_time"
BINARY_PIT_REQUIRED_COLUMNS = (
    "ticker",
    "date",
    "group_index",
    SCORE_COLUMN,
    "fold_id",
    "model_information_cutoff",
)


def _read_manifest(path: str) -> dict:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到Binary PIT manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Binary PIT manifest無法讀取: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Binary PIT manifest必須是JSON object")
    if str(payload.get("schema_type") or "") != "binary_point_in_time_scores":
        raise ValueError("Binary PIT manifest schema_type不合法")
    return payload


@lru_cache(maxsize=8)
def load_binary_point_in_time_score_table(
    manifest_path: str,
    scores_path: str,
) -> tuple[pd.DataFrame, dict]:
    manifest = _read_manifest(manifest_path)
    path = Path(scores_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到Binary PIT scores: {path}")
    table = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(BINARY_PIT_REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Binary PIT scores缺少欄位: {missing}")
    table = table[list(BINARY_PIT_REQUIRED_COLUMNS)].copy()
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["group_index"] = pd.to_numeric(table["group_index"], errors="raise").astype(np.int64)
    table[SCORE_COLUMN] = pd.to_numeric(table[SCORE_COLUMN], errors="raise").astype(float)
    table["fold_id"] = table["fold_id"].astype(str)
    table["model_information_cutoff"] = pd.to_datetime(
        table["model_information_cutoff"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    scores = table[SCORE_COLUMN].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all() or bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("Binary PIT scores必須全部為0到1的有限值")
    if table.duplicated(["ticker", "date"]).any():
        raise ValueError("Binary PIT scores同一ticker/date不可重複")
    if table.duplicated(["group_index"]).any():
        raise ValueError("Binary PIT scores同一group_index不可重複")
    score_dates = pd.to_datetime(table["date"], errors="raise")
    cutoffs = pd.to_datetime(table["model_information_cutoff"], errors="raise")
    if bool((cutoffs >= score_dates).any()):
        raise ValueError("Binary PIT model_information_cutoff必須早於score date")
    score_record = dict(manifest.get("score_table") or {})
    expected_rows = int(score_record.get("row_count", -1))
    if expected_rows != len(table):
        raise ValueError(
            "Binary PIT score row_count與manifest不一致: "
            f"expected={expected_rows}, actual={len(table)}"
        )
    period = dict(manifest.get("score_period") or {})
    actual_start = str(score_dates.min().date()) if len(table) else None
    actual_end = str(score_dates.max().date()) if len(table) else None
    if actual_start != str(period.get("start")) or actual_end != str(period.get("end")):
        raise ValueError("Binary PIT score period與manifest不一致")
    indexed = table.set_index(["ticker", "date"])[[SCORE_COLUMN]].sort_index()
    return indexed, manifest


def build_pass_condition_from_binary_point_in_time_scores(
    df: pd.DataFrame,
    *,
    ticker: str,
    score_threshold: float,
    candidate_condition: np.ndarray,
    manifest_path: str,
    scores_path: str,
) -> np.ndarray:
    threshold = float(score_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Binary PIT threshold必須介於0與1")
    candidate_mask = np.asarray(candidate_condition, dtype=bool)
    if candidate_mask.shape != (len(df),):
        raise ValueError("Binary PIT candidate_condition shape不一致")
    out = np.ones(len(df), dtype=bool)
    if len(df) == 0 or not candidate_mask.any():
        return out
    score_table, manifest = load_binary_point_in_time_score_table(
        str(manifest_path), str(scores_path)
    )
    period = dict(manifest.get("score_period") or {})
    available_from = pd.Timestamp(str(period.get("start"))).normalize()
    available_through = pd.Timestamp(str(period.get("end"))).normalize()
    timestamps = pd.to_datetime(df.index, errors="raise").normalize()
    after_coverage_mask = np.asarray(timestamps > available_through, dtype=bool)
    uncovered_candidate_mask = after_coverage_mask & candidate_mask
    if bool(uncovered_candidate_mask.any()):
        samples = timestamps[uncovered_candidate_mask].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(
            "Binary PIT score table已過期且出現未覆蓋候選事件: "
            f"available_through={available_through.date()}, sample={samples}"
        )
    active_mask = np.asarray(
        (timestamps >= available_from) & (timestamps <= available_through),
        dtype=bool,
    )
    active_candidate_mask = active_mask & candidate_mask
    if not bool(active_candidate_mask.any()):
        return out
    active_dates = timestamps[active_candidate_mask]
    keys = pd.MultiIndex.from_arrays(
        [[str(ticker)] * len(active_dates), active_dates.strftime("%Y-%m-%d")],
        names=["ticker", "date"],
    )
    matched = score_table.reindex(keys)[SCORE_COLUMN].to_numpy(dtype=np.float64, copy=False)
    # Historical optimizer candidates absent from the validated feature bank are rejected
    # conservatively instead of being silently admitted.
    matched = np.where(np.isfinite(matched), matched, 0.0)
    out[active_candidate_mask] = matched >= threshold
    return out


__all__ = [
    "BINARY_PIT_SCORE_SOURCE",
    "BINARY_PIT_REQUIRED_COLUMNS",
    "build_pass_condition_from_binary_point_in_time_scores",
    "load_binary_point_in_time_score_table",
]
