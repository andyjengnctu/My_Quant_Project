"""Workbench stock-name read service backed by canonical FinMind security master.

Names are display-only metadata. Ticker remains the identity used by Trading,
Research and account state. This module performs no provider call and creates no
second name database; it reads the verified Market Data V2 TaiwanStockInfo view.
"""
from __future__ import annotations

from pathlib import Path
import threading

import pandas as pd

from services.trading.market_data_v2_view import TradingMarketDataV2View

_STOCK_NAME_CACHE_LOCK = threading.RLock()
_STOCK_NAME_CACHE: dict[str, dict[str, str]] = {}


def normalize_workbench_ticker(value) -> str:
    return str(value or "").strip().upper()


def normalize_workbench_stock_name(value) -> str:
    return " ".join(str(value or "").strip().split())


def build_workbench_stock_name_map(frame: pd.DataFrame | None) -> dict[str, str]:
    """Project latest official ``stock_name`` per ticker from TaiwanStockInfo."""

    if frame is None or frame.empty or "stock_id" not in frame.columns or "stock_name" not in frame.columns:
        return {}
    work = frame.copy()
    work["stock_id"] = work["stock_id"].map(normalize_workbench_ticker)
    work["stock_name"] = work["stock_name"].map(normalize_workbench_stock_name)
    work = work.loc[(work["stock_id"] != "") & (work["stock_name"] != "")].copy()
    if work.empty:
        return {}
    if "date" in work.columns:
        work["__date"] = pd.to_datetime(work["date"], errors="coerce")
        work["__order"] = range(len(work))
        work = work.sort_values(["stock_id", "__date", "__order"], kind="stable", na_position="first")
    return {
        str(row.stock_id): str(row.stock_name)
        for row in work.drop_duplicates(subset=["stock_id"], keep="last").itertuples(index=False)
    }


def invalidate_workbench_stock_name_cache(project_root=None) -> None:
    with _STOCK_NAME_CACHE_LOCK:
        if project_root is None:
            _STOCK_NAME_CACHE.clear()
            return
        _STOCK_NAME_CACHE.pop(str(Path(project_root).resolve()), None)


def load_workbench_stock_name_map(project_root, *, force: bool = False) -> dict[str, str]:
    root = Path(project_root).resolve()
    cache_key = str(root)
    with _STOCK_NAME_CACHE_LOCK:
        if not force and cache_key in _STOCK_NAME_CACHE:
            return dict(_STOCK_NAME_CACHE[cache_key])
    try:
        view = TradingMarketDataV2View.open(root)
        frame = view.read_dataset_frame(
            "TaiwanStockInfo",
            columns=("date", "stock_id", "stock_name"),
        )
        name_map = build_workbench_stock_name_map(frame)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        # Synthetic/minimal fixtures and a not-yet-ready Market Data archive may
        # legitimately lack stock_name. Names must never block Workbench startup.
        name_map = {}
    with _STOCK_NAME_CACHE_LOCK:
        _STOCK_NAME_CACHE[cache_key] = dict(name_map)
    return dict(name_map)


def workbench_stock_name(project_root, ticker, *, default: str = "-") -> str:
    ticker_key = normalize_workbench_ticker(ticker)
    if not ticker_key:
        return default
    return load_workbench_stock_name_map(project_root).get(ticker_key) or default


def format_workbench_stock_label(project_root, ticker, *, source_label: str | None = None) -> str:
    ticker_key = normalize_workbench_ticker(ticker)
    if not ticker_key:
        return ""
    name = workbench_stock_name(project_root, ticker_key)
    parts = [ticker_key, name]
    source = str(source_label or "").strip()
    if source:
        parts.append(source)
    return " | ".join(parts)


def enrich_workbench_stock_rows(project_root, rows, *, ticker_key: str = "ticker") -> list[dict]:
    name_map = load_workbench_stock_name_map(project_root)
    output = []
    for raw in list(rows or []):
        row = dict(raw or {})
        ticker = normalize_workbench_ticker(row.get(ticker_key))
        row["stock_name"] = name_map.get(ticker) or "-"
        output.append(row)
    return output
