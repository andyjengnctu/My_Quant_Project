from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.portfolio_fast_data import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades
from core.strategy_params import strategy_params_to_dict

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


def _build_consistency_params_snapshot(params) -> Dict[str, Any]:
    from tools.validate.checks import make_consistency_params

    consistency_params = make_consistency_params(params)
    return strategy_params_to_dict(consistency_params, include_runtime=True)


def _strip_portfolio_validation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    slim_payload = dict(payload)
    slim_payload.pop("df_trades", None)
    return slim_payload


def _strip_debug_trade_log_df(debug_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    required_cols = ["日期", "動作", "單筆實質損益"]
    if debug_df is None:
        return pd.DataFrame(columns=required_cols)
    if not isinstance(debug_df, pd.DataFrame):
        raise TypeError(f"debug_trade_log payload 需要 DataFrame，收到: {type(debug_df).__name__}")
    available_cols = [col for col in required_cols if col in debug_df.columns]
    if len(available_cols) != len(required_cols):
        missing_cols = [col for col in required_cols if col not in debug_df.columns]
        raise KeyError(f"debug_trade_log 缺少必要欄位: {missing_cols}")
    return debug_df.loc[:, required_cols].copy()


def _build_validation_consistency_cache_entry(*, ticker: str, clean_df: pd.DataFrame, prep_df: pd.DataFrame, logs, fast_data, sorted_dates, start_year: int, params) -> Dict[str, Any]:
    from tools.validate.checks import build_consistency_parity_params, make_consistency_params
    from tools.validate.tool_adapters import run_debug_trade_log_check, run_portfolio_sim_tool_check

    consistency_params = make_consistency_params(params)
    parity_params = build_consistency_parity_params(consistency_params)
    portfolio_stats = run_portfolio_sim_tool_check(
        ticker,
        "",
        parity_params,
        prepared_df=prep_df,
        standalone_logs=logs,
        packed_fast_data=fast_data,
        sorted_dates=sorted_dates,
        start_year=start_year,
    )
    debug_df, debug_module_path = run_debug_trade_log_check(
        ticker,
        clean_df,
        consistency_params,
        prepared_df=prep_df,
    )
    return {
        "status": "ready",
        "consistency_params_snapshot": strategy_params_to_dict(consistency_params, include_runtime=True),
        "portfolio_sim_stats": _strip_portfolio_validation_payload(portfolio_stats),
        "debug_trade_log_df": _strip_debug_trade_log_df(debug_df),
        "debug_module_path": debug_module_path,
    }


def build_shared_prep_cache(project_root: Path, data_dir: Path, params, cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_inputs, duplicate_issues = discover_unique_csv_inputs(str(data_dir))
    min_rows = get_required_min_rows(params)
    prepared_count = 0
    skipped_count = 0
    validation_cache_ready_count = 0
    validation_cache_error_count = 0
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
            validation_consistency_cache = {
                "status": "not_built",
                "consistency_params_snapshot": _build_consistency_params_snapshot(params),
            }
            try:
                validation_consistency_cache = _build_validation_consistency_cache_entry(
                    ticker=ticker,
                    clean_df=clean_df,
                    prep_df=prep_df,
                    logs=logs,
                    fast_data=fast_data,
                    sorted_dates=sorted_dates,
                    start_year=start_year,
                    params=params,
                )
                validation_cache_ready_count += 1
            except (FileNotFoundError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                validation_consistency_cache = {
                    "status": "error",
                    "consistency_params_snapshot": _build_consistency_params_snapshot(params),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                validation_cache_error_count += 1

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
                "validation_consistency_cache": validation_consistency_cache,
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
        "validation_cache_ready_count": validation_cache_ready_count,
        "validation_cache_error_count": validation_cache_error_count,
        "cache_members": sorted(cache_members),
    }
    _dump_pickle(cache_dir / _CACHE_INDEX_NAME, index_payload)
    return index_payload
