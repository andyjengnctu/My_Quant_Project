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
from filters.breakout_quality.source_inventory import build_source_data_inventory
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


def resolve_strategy_compare_robustness_benchmark_yearly(
    *, project_root: Path, contract: dict[str, Any], seed_yearly_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Restore S4 from the canonical 0050 producer for an all-per-seed suite.

    End-to-end robustness has NO fixed arms: its DL-off references are themselves
    seed-sensitive (e.g. C96/C97/C98). Their durable seed/year observations
    survive retention cleanup, but replay-equity CSVs do not. The shared 0050
    reference is independent of strategy fitting seeds; read the contract-pinned
    adjusted-price dataset and delegate returns to the portfolio benchmark owner.
    Never infer 0050 from strategy returns or trigger replay.
    """
    period = dict(contract.get("comparison_period") or {})
    dataset = str(contract.get("dataset") or "")
    if not dataset or not period.get("start") or not period.get("end"):
        raise ValueError("Strategy SOP v2 INCOMPLETE: robustness 0050 evaluation identity absent")
    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Strategy SOP v2 INCOMPLETE: robustness 0050 period invalid")

    evidence = pd.DataFrame(seed_yearly_frame)
    required = {"arm_id", "seed", "year", "is_complete_year"}
    if evidence.empty or not required.issubset(evidence.columns):
        raise ValueError("Strategy SOP v2 INCOMPLETE: robustness seed yearly cohort absent")
    expected_years = set(range(int(start.year), int(end.year) + 1))
    if set(evidence["year"].astype(int)) != expected_years:
        raise ValueError("Strategy SOP v2 INCOMPLETE: robustness 0050 yearly cohort mismatch")
    # All arm/seed results must refer to precisely the same year set and
    # full-year status. Row counts alone could hide duplicated/missing units.
    if evidence["is_complete_year"].isna().any() or not pd.api.types.is_bool_dtype(evidence["is_complete_year"]):
        raise ValueError("Strategy SOP v2 INCOMPLETE: robustness yearly completeness invalid")
    observed_keys = list(zip(
        evidence["arm_id"].astype(str), evidence["seed"].astype(int),
        evidence["year"].astype(int),
    ))
    units = {(arm, seed) for arm, seed, _year in observed_keys}
    declared_arms = {str(item["arm_id"]) for item in contract.get("stochastic_arms") or ()}
    declared_seeds = {int(seed) for seed in contract.get("resolved_seeds") or ()}
    if declared_arms or declared_seeds:
        if not declared_arms or not declared_seeds or units != {
            (arm, seed) for arm in declared_arms for seed in declared_seeds
        }:
            raise ValueError("Strategy SOP v2 INCOMPLETE: robustness benchmark units not complete")
    expected_keys = {(arm, seed, year) for arm, seed in units for year in expected_years}
    if (
        len(observed_keys) != len(expected_keys)
        or set(observed_keys) != expected_keys
        or evidence.groupby("year")["is_complete_year"].nunique(dropna=False).ne(1).any()
    ):
        raise ValueError("Strategy SOP v2 INCOMPLETE: robustness yearly seed coverage differs")
    full_years = {
        int(year) for year, group in evidence.groupby("year")
        if bool(group["is_complete_year"].iloc[0])
    }

    # The contract's scientific data inventory pins the adjusted-price
    # generation. A changed source is a blocker, not permission to silently
    # backfill a new 0050 vintage into an old completed robustness run.
    pinned = dict(contract.get("dataset_identity") or {})
    if not pinned or pinned.get("status") == "SOURCE_DATA_UNAVAILABLE":
        raise ValueError("Strategy SOP v2 INCOMPLETE: trusted dataset identity absent")
    current = build_source_data_inventory(project_root, dataset)
    if current != pinned:
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050 adjusted-price dataset identity drift")

    ticker = str(PORTFOLIO_DEFAULT_BENCHMARK_TICKER)
    data_dir = Path(get_dataset_dir(str(project_root), dataset)).resolve()
    path, _duplicates = resolve_unique_csv_path(data_dir, ticker)
    frame, _sanitation = sanitize_ohlcv_dataframe(pd.read_csv(path), ticker)
    days = pd.DatetimeIndex(frame.index)
    days = days[(days >= start) & (days <= end)]
    if days.empty or {int(y) for y in days.year} != expected_years:
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050 adjusted-price cohort missing years")
    # A complete strategy year cannot be paired with only a partial price year.
    for year in full_years:
        year_days = days[days.year == year]
        if not len(year_days) or year_days[0].month != 1 or year_days[-1].month != 12:
            raise ValueError(f"Strategy SOP v2 INCOMPLETE: 0050/{year} complete-year coverage absent")
    price_data = {dt: {"Close": float(close)} for dt, close in frame["Close"].items()}
    produced = _build_benchmark_period_stats(
        benchmark_data=price_data, sorted_dates=tuple(days), start_idx=0,
    )
    yearly = [dict(row) for row in produced.get("bm_yearly_return_rows") or ()]
    if (
        {int(row["year"]) for row in yearly} != expected_years
        or len(yearly) != len(expected_years)
        or not all(np.isfinite(float(row["year_return_pct"])) for row in yearly)
    ):
        raise ValueError("Strategy SOP v2 INCOMPLETE: canonical 0050 yearly result incomplete")
    return yearly


def collect_consistent_benchmark_yearly(
    evidence_by_arm: dict[str, dict[str, Any]], *, source_kind: str = "fixed",
) -> list[dict[str, Any]]:
    """Only 0050 evidence owner validates cross-arm yearly-source agreement.

    The source rows are already canonical; this checks identity/equality without
    recomputing price returns, changing annual boundaries, or filling missing data.
    """
    sources = [list(item.get("benchmark_yearly") or []) for item in evidence_by_arm.values()]
    if source_kind not in {"single", "fixed"}:
        raise ValueError("Strategy SOP v2 INCOMPLETE: unregistered benchmark evidence source kind")
    if not sources or not all(sources):
        detail = ("source missing canonical 0050 yearly evidence" if source_kind == "single"
                  else "fixed baseline missing canonical 0050 yearly evidence")
        raise ValueError("Strategy SOP v2 INCOMPLETE: " + detail)
    if any(rows != sources[0] for rows in sources[1:]):
        detail = ("baseline 0050 source disagreement" if source_kind == "single"
                  else "fixed 0050 benchmark sources inconsistent")
        raise ValueError("Strategy SOP v2 INCOMPLETE: " + detail)
    return sources[0]


def resolve_strategy_robustness_report_benchmark_yearly(
    *, project_root: Path, contract: dict[str, Any],
    fixed_results: dict[str, dict[str, Any]], seed_yearly_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Canonical Fixed/All-seed 0050 reporting decision, not a Strategy-side fork."""
    if fixed_results:
        return collect_consistent_benchmark_yearly(fixed_results)
    return resolve_strategy_compare_robustness_benchmark_yearly(
        project_root=project_root, contract=contract,
        seed_yearly_frame=seed_yearly_frame,
    )


def collect_strategy_pair_benchmark_yearly(
    pair_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Single-seed Strategy source consensus, owned by 0050 evidence producer."""
    return collect_consistent_benchmark_yearly({
        str(name): dict(item["payload"]) for name, item in pair_payloads.items()
    }, source_kind="single")
