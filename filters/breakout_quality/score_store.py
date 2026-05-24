"""Score table loading and lookup for breakout quality filter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import DEFAULT_FILTER_ID, DEFAULT_SCORE_FILENAME
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.paths import resolve_filter_model_dir, resolve_filter_output_dir


def _candidate_score_paths(project_root: str, filter_id: str) -> tuple[Path, ...]:
    model_dir = resolve_filter_model_dir(project_root, filter_id)
    output_dir = resolve_filter_output_dir(project_root, filter_id)
    return (
        model_dir / DEFAULT_SCORE_FILENAME,
        output_dir / DEFAULT_SCORE_FILENAME,
        output_dir / f"{filter_id}_scores.csv",
        resolve_filter_output_dir(project_root) / f"{filter_id}_scores.csv",
    )


def resolve_score_table_path(project_root: str, filter_id: str = DEFAULT_FILTER_ID, explicit_path: str | None = None) -> Path:
    if explicit_path is not None and str(explicit_path).strip() != "":
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = Path(project_root) / path
        return path.resolve()
    for path in _candidate_score_paths(project_root, filter_id):
        if path.is_file():
            return path.resolve()
    searched = "\n".join(str(path) for path in _candidate_score_paths(project_root, filter_id))
    raise FileNotFoundError(f"找不到 breakout quality score table。filter_id={filter_id}\n已搜尋:\n{searched}")


@lru_cache(maxsize=16)
def load_score_table(project_root: str, filter_id: str = DEFAULT_FILTER_ID, explicit_path: str | None = None) -> pd.DataFrame:
    path = resolve_score_table_path(project_root, filter_id=filter_id, explicit_path=explicit_path)
    table = read_breakout_quality_csv(path)
    required = {"ticker", "date", "high_len", "dl_pass"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"breakout quality score table 缺少欄位: {missing}; path={path}")
    table = table.copy()
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["high_len"] = table["high_len"].astype(int)
    table["dl_pass"] = table["dl_pass"].astype(bool)
    if table.duplicated(["ticker", "date", "high_len"]).any():
        dup_count = int(table.duplicated(["ticker", "date", "high_len"]).sum())
        raise ValueError(f"breakout quality score table 有重複 key: {dup_count}; path={path}")
    return table.set_index(["ticker", "date", "high_len"]).sort_index()


def build_pass_condition_from_score_table(
    df: pd.DataFrame,
    *,
    ticker: str,
    high_len: int,
    project_root: str,
    filter_id: str = DEFAULT_FILTER_ID,
    score_path: str | None = None,
) -> np.ndarray:
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("啟用 breakout quality filter 時必須提供 ticker")
    score_table = load_score_table(str(project_root), filter_id=str(filter_id), explicit_path=score_path)
    out = np.zeros(len(df), dtype=bool)
    if len(df) == 0:
        return out
    dates = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    keys = pd.MultiIndex.from_arrays(
        [[str(ticker)] * len(df), dates, [int(high_len)] * len(df)],
        names=["ticker", "date", "high_len"],
    )
    matched = score_table.reindex(keys)
    values = matched["dl_pass"].to_numpy(dtype=object, copy=False)
    out[:] = np.asarray([False if pd.isna(value) else bool(value) for value in values], dtype=bool)
    return out


__all__ = ["build_pass_condition_from_score_table", "load_score_table", "resolve_score_table_path"]
