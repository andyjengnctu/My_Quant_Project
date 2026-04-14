from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.portfolio_fast_data import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades

LOCAL_REGRESSION_SHARED_PREP_CACHE_ENV = "LOCAL_REGRESSION_SHARED_PREP_CACHE_DIR"
_CACHE_INDEX_NAME = "shared_prep_cache_index.pkl"


def resolve_shared_prep_cache_dir_from_env(environ: Optional[Dict[str, str]] = None) -> Optional[Path]:
    env = os.environ if environ is None else environ
    raw_value = str(env.get(LOCAL_REGRESSION_SHARED_PREP_CACHE_ENV, "") or "").strip()
    if not raw_value:
        return None
    return Path(raw_value)


def _entry_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{str(ticker).strip()}.pkl"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _dump_pickle(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_shared_prep_cache_entry(ticker: str, *, cache_dir: Optional[Path] = None, environ: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    resolved_cache_dir = resolve_shared_prep_cache_dir_from_env(environ) if cache_dir is None else cache_dir
    if resolved_cache_dir is None:
        return None
    entry_path = _entry_path(resolved_cache_dir, ticker)
    if not entry_path.exists():
        return None
    payload = _load_pickle(entry_path)
    if not isinstance(payload, dict):
        return None
    return payload


def build_shared_prep_cache(project_root: Path, data_dir: Path, params, cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_inputs, duplicate_issues = discover_unique_csv_inputs(str(data_dir))
    min_rows = get_required_min_rows(params)
    prepared_count = 0
    skipped_count = 0
    cache_members = []

    for ticker, file_path in csv_inputs:
        raw_df = pd.read_csv(file_path)
        if len(raw_df) < min_rows:
            skipped_count += 1
            entry = {
                "ticker": ticker,
                "file_path": os.path.abspath(str(file_path)),
                "status": "skipped_insufficient",
                "raw_row_count": int(len(raw_df)),
                "min_rows": int(min_rows),
            }
        else:
            clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows)
            prep_df, logs, single_stats = prep_stock_data_and_trades(clean_df, params, return_stats=True)
            fast_data = pack_prepared_stock_data(prep_df)
            sorted_dates = sorted(get_fast_dates(fast_data))
            if not sorted_dates:
                sorted_dates = sorted(prep_df.index)
            start_year = int(pd.Timestamp(sorted_dates[0]).year) if sorted_dates else int(pd.Timestamp(prep_df.index.min()).year)
            entry = {
                "ticker": ticker,
                "file_path": os.path.abspath(str(file_path)),
                "status": "ready",
                "clean_df": clean_df,
                "sanitize_stats": dict(sanitize_stats),
                "prepared_df": prep_df,
                "standalone_logs": list(logs),
                "single_stats": dict(single_stats),
                "fast_data": fast_data,
                "sorted_dates": list(sorted_dates),
                "start_year": start_year,
            }
            prepared_count += 1

        entry_path = _entry_path(cache_dir, ticker)
        _dump_pickle(entry_path, entry)
        cache_members.append(str(entry_path.name))

    index_payload = {
        "status": "PASS",
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "csv_count": len(csv_inputs),
        "duplicate_issue_count": len(duplicate_issues),
        "duplicate_issues": list(duplicate_issues),
        "prepared_count": prepared_count,
        "skipped_count": skipped_count,
        "cache_members": sorted(cache_members),
    }
    _dump_pickle(cache_dir / _CACHE_INDEX_NAME, index_payload)
    return index_payload
