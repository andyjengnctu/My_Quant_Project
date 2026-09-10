from __future__ import annotations

import pandas as pd

from core.portfolio_stats import calc_annual_return_pct, calc_curve_stats, calc_plain_romd, calc_portfolio_score
from services.optimizer.outer_rolling_fold_context import (
    normalize_optimizer_seed_ensemble_fold_row,
    optimizer_seed_ensemble_row_sort_key,
)
from services.optimizer.outer_rolling_formatting import _safe_float
from services.optimizer.outer_rolling_policy import REPORT_POLICY_NAMES


def _normalize_equity_curve_rows(curve_rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for raw in list(curve_rows or []):
        try:
            date = pd.Timestamp(raw.get("date") or raw.get("Date")).strftime("%Y-%m-%d")
            equity = float(raw.get("equity", raw.get("Equity", 0.0)) or 0.0)
            strategy_return_pct = float(raw.get("strategy_return_pct", raw.get("Strategy_Return_Pct", 0.0)) or 0.0)
            benchmark_return_pct = float(raw.get("benchmark_return_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not date or equity <= 0.0:
            continue
        normalized.append({
            "date": date,
            "equity": equity,
            "strategy_return_pct": strategy_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
        })
    return sorted(normalized, key=lambda row: row["date"])


def _derive_curve_initial_capital(curve_rows: list[dict], explicit_initial_capital: float | int | None = None) -> float:
    explicit = _safe_float(explicit_initial_capital, 0.0)
    if explicit > 0.0:
        return explicit
    curve = _normalize_equity_curve_rows(curve_rows)
    if not curve:
        return 0.0
    first = curve[0]
    denominator = 1.0 + float(first.get("strategy_return_pct", 0.0)) / 100.0
    if denominator <= 0.0:
        return float(first.get("equity", 0.0))
    return float(first.get("equity", 0.0)) / denominator


def _stitch_strategy_equity_curves(rows: list[dict], *, policy_name: str | None = None, best_finalist: bool = False) -> dict:
    ordered_rows = sorted((normalize_optimizer_seed_ensemble_fold_row(row) for row in list(rows or [])), key=optimizer_seed_ensemble_row_sort_key)
    stitched: list[dict] = []
    chain_initial = 0.0
    current_start = 0.0
    score_total_r = 0.0
    score_median_samples = []
    for row in ordered_rows:
        if best_finalist:
            raw_curve = _normalize_equity_curve_rows(list(row.get("best_finalist_equity_curve") or []))
            raw_initial = _derive_curve_initial_capital(raw_curve, row.get("best_finalist_initial_capital"))
            row_score_total_r = _safe_float(row.get("best_finalist_score_total_r", row.get("best_finalist_total_r", 0.0)), 0.0)
            row_score_median_r = _safe_float(row.get("best_finalist_score_median_r", row.get("best_finalist_median_r", 0.0)), 0.0)
        else:
            policy = dict(row.get(str(policy_name)) or {})
            raw_curve = _normalize_equity_curve_rows(list(policy.get("rank_1_equity_curve") or []))
            raw_initial = _derive_curve_initial_capital(raw_curve, policy.get("rank_1_initial_capital"))
            row_score_total_r = _safe_float(policy.get("rank_1_score_total_r", policy.get("rank_1_total_r", 0.0)), 0.0)
            row_score_median_r = _safe_float(policy.get("rank_1_score_median_r", policy.get("rank_1_median_r", 0.0)), 0.0)
        if not raw_curve or raw_initial <= 0.0:
            continue
        score_total_r += float(row_score_total_r)
        score_median_samples.append(float(row_score_median_r))
        if current_start <= 0.0:
            current_start = raw_initial
            chain_initial = raw_initial
        scale = current_start / raw_initial
        for point in raw_curve:
            stitched.append({
                "date": point["date"],
                "equity": float(point["equity"]) * scale,
            })
        current_start = float(stitched[-1]["equity"])
    score_median_r = float(pd.Series(score_median_samples).median()) if score_median_samples else 0.0
    return {
        "initial_equity": float(chain_initial),
        "curve": stitched,
        "score_total_r": float(score_total_r),
        "score_median_r": float(score_median_r),
        "score_r_source": "single_stock",
    }


def _stitch_benchmark_equity_curve(rows: list[dict]) -> dict:
    ordered_rows = sorted((normalize_optimizer_seed_ensemble_fold_row(row) for row in list(rows or [])), key=optimizer_seed_ensemble_row_sort_key)
    stitched: list[dict] = []
    chain_initial = 0.0
    current_start = 0.0
    for row in ordered_rows:
        curve_source = None
        best_curve = _normalize_equity_curve_rows(list(row.get("best_finalist_equity_curve") or []))
        if best_curve:
            curve_source = best_curve
            raw_initial = _derive_curve_initial_capital(best_curve, row.get("best_finalist_initial_capital"))
        else:
            raw_initial = 0.0
            for policy_name in REPORT_POLICY_NAMES:
                policy = dict(row.get(policy_name) or {})
                policy_curve = _normalize_equity_curve_rows(list(policy.get("rank_1_equity_curve") or []))
                if policy_curve:
                    curve_source = policy_curve
                    raw_initial = _derive_curve_initial_capital(policy_curve, policy.get("rank_1_initial_capital"))
                    break
        if not curve_source:
            continue
        if current_start <= 0.0:
            current_start = raw_initial if raw_initial > 0.0 else 1.0
            chain_initial = current_start
        for point in curve_source:
            benchmark_factor = 1.0 + float(point.get("benchmark_return_pct", 0.0)) / 100.0
            stitched.append({
                "date": point["date"],
                "equity": current_start * benchmark_factor,
            })
        current_start = float(stitched[-1]["equity"])
    return {"initial_equity": float(chain_initial), "curve": stitched}


def _month_end_equities_from_curve(curve: list[dict], *, initial_equity: float) -> list[float]:
    values = []
    if initial_equity > 0.0:
        values.append(float(initial_equity))
    current_month = None
    previous_equity = None
    for point in list(curve or []):
        try:
            ts = pd.Timestamp(point.get("date"))
            equity = float(point.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        month_key = (int(ts.year), int(ts.month))
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            if previous_equity is not None:
                values.append(float(previous_equity))
            current_month = month_key
        previous_equity = equity
    if previous_equity is not None:
        values.append(float(previous_equity))
    return values


def _calc_full_year_return_metrics_from_curve(curve: list[dict]) -> dict:
    by_year: dict[int, dict] = {}
    for point in list(curve or []):
        try:
            ts = pd.Timestamp(point.get("date"))
            equity = float(point.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not pd.notna(ts) or equity <= 0.0:
            continue
        year = int(ts.year)
        bucket = by_year.setdefault(
            year,
            {"first_date": ts, "first_equity": equity, "last_date": ts, "last_equity": equity},
        )
        if ts < bucket["first_date"]:
            bucket["first_date"] = ts
            bucket["first_equity"] = equity
        if ts > bucket["last_date"]:
            bucket["last_date"] = ts
            bucket["last_equity"] = equity

    rows = []
    for year in sorted(by_year):
        bucket = by_year[year]
        first_date = bucket["first_date"]
        last_date = bucket["last_date"]
        if not bool(first_date.month == 1 and last_date.month == 12):
            continue
        start_equity = float(bucket["first_equity"])
        end_equity = float(bucket["last_equity"])
        year_return_pct = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0.0 else 0.0
        rows.append({
            "year": int(year),
            "year_return_pct": float(year_return_pct),
            "start_equity": float(start_equity),
            "end_equity": float(end_equity),
            "is_full_year": True,
        })
    return {
        "full_year_count": int(len(rows)),
        "min_full_year_return_pct": float(min((row["year_return_pct"] for row in rows), default=0.0)),
        "yearly_return_rows": rows,
    }



def _calc_full_month_return_metrics_from_curve(curve: list[dict]) -> dict:
    by_month: dict[tuple[int, int], dict] = {}
    for point in list(curve or []):
        try:
            ts = pd.Timestamp(point.get("date"))
            equity = float(point.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not pd.notna(ts) or equity <= 0.0:
            continue
        key = (int(ts.year), int(ts.month))
        bucket = by_month.setdefault(
            key,
            {"first_date": ts, "first_equity": equity, "last_date": ts, "last_equity": equity},
        )
        if ts < bucket["first_date"]:
            bucket["first_date"] = ts
            bucket["first_equity"] = equity
        if ts > bucket["last_date"]:
            bucket["last_date"] = ts
            bucket["last_equity"] = equity

    rows = []
    for year, month in sorted(by_month):
        bucket = by_month[(year, month)]
        start_equity = float(bucket["first_equity"])
        end_equity = float(bucket["last_equity"])
        month_return_pct = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0.0 else 0.0
        rows.append({
            "year": int(year),
            "month": int(month),
            "period": f"{int(year):04d}-{int(month):02d}",
            "month_return_pct": float(month_return_pct),
            "start_equity": float(start_equity),
            "end_equity": float(end_equity),
            "is_full_month": True,
        })
    return {
        "full_month_count": int(len(rows)),
        "min_month_return_pct": float(min((row["month_return_pct"] for row in rows), default=0.0)),
        "monthly_return_rows": rows,
    }

def _calc_full_quarter_return_metrics_from_curve(curve: list[dict]) -> dict:
    by_quarter: dict[tuple[int, int], dict] = {}
    for point in list(curve or []):
        try:
            ts = pd.Timestamp(point.get("date"))
            equity = float(point.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not pd.notna(ts) or equity <= 0.0:
            continue
        quarter = int((ts.month - 1) // 3 + 1)
        key = (int(ts.year), int(quarter))
        bucket = by_quarter.setdefault(
            key,
            {"first_date": ts, "first_equity": equity, "last_date": ts, "last_equity": equity},
        )
        if ts < bucket["first_date"]:
            bucket["first_date"] = ts
            bucket["first_equity"] = equity
        if ts > bucket["last_date"]:
            bucket["last_date"] = ts
            bucket["last_equity"] = equity

    rows = []
    for year, quarter in sorted(by_quarter):
        bucket = by_quarter[(year, quarter)]
        first_date = bucket["first_date"]
        last_date = bucket["last_date"]
        first_month = (int(quarter) - 1) * 3 + 1
        last_month = first_month + 2
        if not bool(first_date.month == first_month and last_date.month == last_month):
            continue
        start_equity = float(bucket["first_equity"])
        end_equity = float(bucket["last_equity"])
        quarter_return_pct = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0.0 else 0.0
        rows.append({
            "year": int(year),
            "quarter": int(quarter),
            "period": f"{int(year)}Q{int(quarter)}",
            "quarter_return_pct": float(quarter_return_pct),
            "start_equity": float(start_equity),
            "end_equity": float(end_equity),
            "is_full_quarter": True,
        })
    return {
        "full_quarter_count": int(len(rows)),
        "min_quarter_return_pct": float(min((row["quarter_return_pct"] for row in rows), default=0.0)),
        "quarterly_return_rows": rows,
    }

def _calc_stitched_curve_metrics(stitched: dict, *, benchmark_plain_romd: bool = False) -> dict:
    initial_equity = _safe_float(stitched.get("initial_equity"), 0.0)
    curve = list(stitched.get("curve") or [])
    if initial_equity <= 0.0 or not curve:
        return {
            "score": 0.0,
            "plain_romd_score": 0.0,
            "return_pct": 0.0,
            "mdd_pct": 0.0,
            "annual_return_pct": 0.0,
            "full_year_count": 0,
            "min_full_year_return_pct": 0.0,
            "yearly_return_rows": [],
            "full_month_count": 0,
            "min_month_return_pct": 0.0,
            "monthly_return_rows": [],
            "full_quarter_count": 0,
            "min_quarter_return_pct": 0.0,
            "quarterly_return_rows": [],
            "r_squared": 0.0,
            "monthly_win_rate": 0.0,
            "curve_points": 0,
        }
    final_equity = float(curve[-1].get("equity", initial_equity) or initial_equity)
    return_pct = (final_equity / initial_equity - 1.0) * 100.0
    peak = initial_equity
    max_drawdown = 0.0
    for point in curve:
        equity = float(point.get("equity", 0.0) or 0.0)
        if equity > peak:
            peak = equity
        if peak > 0.0:
            drawdown = (peak - equity) / peak * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    first_date = pd.Timestamp(curve[0]["date"])
    last_date = pd.Timestamp(curve[-1]["date"])
    years = max(0.0, ((last_date - first_date).days + 1) / 365.25)
    annual_return_pct = calc_annual_return_pct(initial_equity, final_equity, years)
    monthly_equities = _month_end_equities_from_curve(curve, initial_equity=initial_equity)
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    full_year_metrics = _calc_full_year_return_metrics_from_curve(curve)
    full_month_metrics = _calc_full_month_return_metrics_from_curve(curve)
    full_quarter_metrics = _calc_full_quarter_return_metrics_from_curve(curve)
    min_full_year_return_pct = float(full_year_metrics.get("min_full_year_return_pct", 0.0))
    min_month_return_pct = float(full_month_metrics.get("min_month_return_pct", 0.0))
    min_quarter_return_pct = float(full_quarter_metrics.get("min_quarter_return_pct", 0.0))
    score_total_r = _safe_float(stitched.get("score_total_r", 0.0), 0.0)
    score_median_r = _safe_float(stitched.get("score_median_r", 0.0), 0.0)
    plain_romd_score = calc_plain_romd(return_pct, max_drawdown)
    if benchmark_plain_romd:
        score = plain_romd_score
    else:
        score = calc_portfolio_score(
            return_pct,
            max_drawdown,
            monthly_win_rate,
            r_squared,
            annual_return_pct=annual_return_pct,
            min_full_year_return_pct=min_full_year_return_pct,
            min_month_return_pct=min_month_return_pct,
            min_quarter_return_pct=min_quarter_return_pct,
            total_r=score_total_r,
            median_r=score_median_r,
        )
    return {
        "score": float(score),
        "plain_romd_score": float(plain_romd_score),
        "return_pct": float(return_pct),
        "mdd_pct": float(max_drawdown),
        "annual_return_pct": float(annual_return_pct),
        "full_year_count": int(full_year_metrics.get("full_year_count", 0)),
        "min_full_year_return_pct": min_full_year_return_pct,
        "yearly_return_rows": list(full_year_metrics.get("yearly_return_rows", [])),
        "full_month_count": int(full_month_metrics.get("full_month_count", 0)),
        "min_month_return_pct": min_month_return_pct,
        "monthly_return_rows": list(full_month_metrics.get("monthly_return_rows", [])),
        "full_quarter_count": int(full_quarter_metrics.get("full_quarter_count", 0)),
        "min_quarter_return_pct": min_quarter_return_pct,
        "quarterly_return_rows": list(full_quarter_metrics.get("quarterly_return_rows", [])),
        "r_squared": float(r_squared),
        "monthly_win_rate": float(monthly_win_rate),
        "score_total_r": float(score_total_r),
        "score_median_r": float(score_median_r),
        "score_r_source": str(stitched.get("score_r_source", "single_stock")),
        "curve_points": int(len(curve)),
        "start_date": str(curve[0].get("date", "")),
        "end_date": str(curve[-1].get("date", "")),
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
    }
