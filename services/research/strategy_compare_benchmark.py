"""Restore Strategy SOP benchmark report evidence without replaying strategies.

This adapter consumes the same adjusted 0050 CSV and the verified baseline
replay date cohort, and delegates price-return calculation to the canonical
portfolio benchmark producer. It never derives 0050 from strategy PnL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data_utils import resolve_unique_csv_path, sanitize_ohlcv_dataframe
from core.dataset_profiles import get_dataset_dir
from core.portfolio_benchmark import _build_benchmark_period_stats
from services.portfolio_replay import PORTFOLIO_DEFAULT_BENCHMARK_TICKER


def resolve_strategy_compare_benchmark_yearly(
    *,
    pair_payload: dict[str, Any],
    pair_dir: Path,
    project_root: Path,
    dataset: str,
    comparison_start: str,
    comparison_end: str,
) -> list[dict[str, Any]]:
    """Return existing 0050 evidence or refresh only a legacy missing sidecar.

    Consumers must have validated the pair's scientific identity and completed
    file-integrity contract before calling this for a historical REUSE. Missing
    price/cohort evidence is an error, not permission to run a portfolio replay.
    """
    rows = list(pair_payload.get("benchmark_yearly") or [])
    if rows:
        return rows

    metadata = dict(pair_payload.get("metadata") or {})
    expected_period = {"start": str(comparison_start), "end": str(comparison_end)}
    if dict(metadata.get("comparison_period") or {}) != expected_period:
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050 backfill comparison period mismatch")
    if str(metadata.get("dataset") or "") != str(dataset):
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050 backfill dataset mismatch")
    ticker = str(PORTFOLIO_DEFAULT_BENCHMARK_TICKER)
    if str(metadata.get("benchmark_ticker") or ticker) != ticker:
        raise ValueError("Strategy SOP v2 INCOMPLETE: benchmark ticker is not canonical 0050")

    data_dir = Path(get_dataset_dir(str(project_root), dataset)).resolve()
    recorded_data_dir = str(metadata.get("data_dir") or "").strip()
    if recorded_data_dir and Path(recorded_data_dir).resolve() != data_dir:
        raise ValueError("Strategy SOP v2 INCOMPLETE: historical benchmark dataset source changed")
    if not data_dir.is_dir():
        raise ValueError("Strategy SOP v2 INCOMPLETE: canonical adjusted dataset unavailable")

    equity_path = Path(pair_dir) / "no_filter_equity.csv"
    if not equity_path.is_file():
        raise ValueError("Strategy SOP v2 INCOMPLETE: verified baseline equity-date evidence absent")
    benchmark_column = f"Benchmark_{ticker}_Pct"
    try:
        equity = pd.read_csv(
            equity_path, usecols=["Date", benchmark_column], encoding="utf-8-sig",
        )
        dates = pd.DatetimeIndex(pd.to_datetime(equity["Date"], errors="raise"))
        saved_benchmark_pct = pd.to_numeric(equity[benchmark_column], errors="raise")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("Strategy SOP v2 INCOMPLETE: baseline equity-date evidence invalid") from exc
    if (
        dates.empty or dates.hasnans or not dates.is_unique
        or not dates.is_monotonic_increasing
        or dates.min() < pd.Timestamp(comparison_start)
        or dates.max() > pd.Timestamp(comparison_end)
    ):
        raise ValueError("Strategy SOP v2 INCOMPLETE: baseline equity-date cohort invalid")

    benchmark_file, _duplicates = resolve_unique_csv_path(data_dir, ticker)
    benchmark_df, _sanitation = sanitize_ohlcv_dataframe(
        pd.read_csv(benchmark_file), ticker,
    )
    # The canonical portfolio benchmark engine accepts date->Close fast data.
    benchmark_data = {
        date: {"Close": float(close)}
        for date, close in benchmark_df["Close"].items()
    }
    result = _build_benchmark_period_stats(
        benchmark_data=benchmark_data,
        sorted_dates=tuple(dates),
        start_idx=0,
    )
    # Historical results may predate this report field. Their saved benchmark
    # curve is an independent, verified portfolio-producer witness for the
    # adjusted-price vintage; current CSV values must reproduce that curve.
    start_price = result.get("benchmark_start_price")
    if not start_price or float(start_price) <= 0:
        raise ValueError("Strategy SOP v2 INCOMPLETE: canonical 0050 starting price absent")
    closes = benchmark_df["Close"].reindex(dates, method="ffill")
    expected_pct = (closes.to_numpy(dtype=float) / float(start_price) - 1.0) * 100.0
    actual_pct = saved_benchmark_pct.to_numpy(dtype=float)
    if (
        not np.isfinite(expected_pct).all()
        or not np.isfinite(actual_pct).all()
        or not np.allclose(expected_pct, actual_pct, rtol=0.0, atol=1e-7)
    ):
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050 price vintage disagrees with saved replay")
    restored = [dict(row) for row in result.get("bm_yearly_return_rows") or []]
    yearly = list(pair_payload.get("yearly") or [])
    expected_years = {int(row["year"]) for row in yearly}
    actual_years = {int(row["year"]) for row in restored}
    if not expected_years or expected_years != actual_years:
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050 backfill yearly cohort mismatch")
    # The annual boundaries are produced by the same portfolio replay calendar.
    by_year = {int(row["year"]): row for row in restored}
    for row in yearly:
        year = int(row["year"])
        reference = by_year[year]
        if (
            str(row.get("start_date") or "") != reference["start_date"]
            or str(row.get("end_date") or "") != reference["end_date"]
        ):
            raise ValueError(f"Strategy SOP v2 INCOMPLETE: 0050/{year} replay interval mismatch")
    return restored
