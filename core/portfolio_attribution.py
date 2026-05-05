from __future__ import annotations

from collections import defaultdict
from typing import Any

from config.training_policy import (
    DOMINANT_YEAR_HIGH_POSITIVE_PNL_SHARE,
    DOMINANT_YEAR_NARROW_POSITIVE_SYMBOL_COUNT,
    DOMINANT_YEAR_NARROW_POSITIVE_TRADE_COUNT,
    DOMINANT_YEAR_TOP_TRADE_OUTLIER_PNL_SHARE,
)

DOMINANT_YEAR_DEPENDENCY_DIAGNOSTICS_VERSION = "dominant_year_dependency_v2_intuitive"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resolve_year(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    raw_year = getattr(raw_value, "year", None)
    if raw_year is not None:
        try:
            year = int(raw_year)
            return year if year > 0 else None
        except (TypeError, ValueError):
            return None
    raw_text = str(raw_value).strip()
    if len(raw_text) >= 4:
        try:
            year = int(raw_text[:4])
            return year if year > 0 else None
        except ValueError:
            return None
    return None


def _resolve_trade_entry_year(trade: dict) -> int | None:
    direct_year = _resolve_year(trade.get("entry_year"))
    if direct_year is not None:
        return direct_year
    return _resolve_year(trade.get("entry_trade_date"))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0


def _empty_diagnostics(*, status: str, positive_total_pnl: float = 0.0) -> dict:
    return {
        "diagnostics_version": DOMINANT_YEAR_DEPENDENCY_DIAGNOSTICS_VERSION,
        "dependency_warning": False,
        "dependency_status": str(status),
        "dependency_reason": [],
        "positive_total_pnl": float(positive_total_pnl),
        "unassigned_positive_pnl": 0.0,
        "dominant_entry_year": None,
        "dominant_year_positive_pnl": 0.0,
        "dominant_year_positive_pnl_share": 0.0,
        "dominant_year_pnl_share": 0.0,
        "dominant_year_positive_trade_count": 0,
        "dominant_year_trade_count": 0,
        "dominant_year_positive_symbol_count": 0,
        "dominant_year_symbol_count": 0,
        "top_trade_pnl_share_in_dominant_year": 0.0,
        "thresholds": _threshold_snapshot(),
    }


def _threshold_snapshot() -> dict:
    return {
        "high_positive_pnl_share": float(DOMINANT_YEAR_HIGH_POSITIVE_PNL_SHARE),
        "narrow_positive_trade_count": int(DOMINANT_YEAR_NARROW_POSITIVE_TRADE_COUNT),
        "narrow_positive_symbol_count": int(DOMINANT_YEAR_NARROW_POSITIVE_SYMBOL_COUNT),
        "top_trade_outlier_pnl_share": float(DOMINANT_YEAR_TOP_TRADE_OUTLIER_PNL_SHARE),
    }


def build_dominant_year_dependency_diagnostics(closed_trades_stats) -> dict:
    """Diagnose whether positive train PnL is concentrated in an intuitive way.

    WARN only when both conditions hold:
    1. one entry year contributes a high share of total positive train PnL;
    2. that dominant year is narrow by trade count, symbol count, or one top-trade outlier.

    This avoids penalizing a parameter set solely because it broadly captured a genuine bull year.
    """

    trades = [dict(trade) for trade in list(closed_trades_stats or []) if isinstance(trade, dict)]
    positive_total_pnl = sum(max(_coerce_float(trade.get("pnl")), 0.0) for trade in trades)
    if positive_total_pnl <= 0.0:
        return _empty_diagnostics(status="NO_POSITIVE_PNL", positive_total_pnl=0.0)

    year_positive_pnl = defaultdict(float)
    year_positive_trade_count = defaultdict(int)
    year_symbol_positive_pnl: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    year_positive_trade_pnls: dict[int, list[float]] = defaultdict(list)
    unassigned_positive_pnl = 0.0

    for trade in trades:
        pnl = max(_coerce_float(trade.get("pnl")), 0.0)
        if pnl <= 0.0:
            continue
        entry_year = _resolve_trade_entry_year(trade)
        if entry_year is None:
            unassigned_positive_pnl += pnl
            continue
        entry_year = int(entry_year)
        year_positive_pnl[entry_year] += pnl
        year_positive_trade_count[entry_year] += 1
        year_positive_trade_pnls[entry_year].append(pnl)
        ticker = str(trade.get("ticker") or "").strip()
        if ticker:
            year_symbol_positive_pnl[entry_year][ticker] += pnl

    if not year_positive_pnl:
        diagnostics = _empty_diagnostics(status="MISSING_ENTRY_YEAR", positive_total_pnl=positive_total_pnl)
        diagnostics["unassigned_positive_pnl"] = float(unassigned_positive_pnl)
        return diagnostics

    dominant_entry_year, dominant_year_positive_pnl = max(
        year_positive_pnl.items(),
        key=lambda item: (float(item[1]), -int(item[0])),
    )
    dominant_entry_year = int(dominant_entry_year)
    dominant_share = _safe_ratio(dominant_year_positive_pnl, positive_total_pnl)
    positive_trade_count = int(year_positive_trade_count.get(dominant_entry_year, 0))
    positive_symbol_count = int(len(year_symbol_positive_pnl.get(dominant_entry_year, {})))
    top_trade_pnl_share = _safe_ratio(
        max(year_positive_trade_pnls.get(dominant_entry_year, []), default=0.0),
        dominant_year_positive_pnl,
    )

    temporal_concentration = dominant_share >= float(DOMINANT_YEAR_HIGH_POSITIVE_PNL_SHARE)
    narrow_trade_source = positive_trade_count <= int(DOMINANT_YEAR_NARROW_POSITIVE_TRADE_COUNT)
    narrow_symbol_source = positive_symbol_count <= int(DOMINANT_YEAR_NARROW_POSITIVE_SYMBOL_COUNT)
    top_trade_outlier = top_trade_pnl_share >= float(DOMINANT_YEAR_TOP_TRADE_OUTLIER_PNL_SHARE)
    dependency_warning = bool(
        temporal_concentration and (narrow_trade_source or narrow_symbol_source or top_trade_outlier)
    )

    dependency_reason = []
    if dependency_warning:
        dependency_reason.append("dominant_year_positive_pnl_share_gte_high")
        if narrow_trade_source:
            dependency_reason.append("dominant_year_positive_trade_count_lte_narrow")
        if narrow_symbol_source:
            dependency_reason.append("dominant_year_positive_symbol_count_lte_narrow")
        if top_trade_outlier:
            dependency_reason.append("top_trade_pnl_share_in_dominant_year_gte_outlier")

    return {
        "diagnostics_version": DOMINANT_YEAR_DEPENDENCY_DIAGNOSTICS_VERSION,
        "dependency_warning": dependency_warning,
        "dependency_status": "WARN" if dependency_warning else "OK",
        "dependency_reason": dependency_reason,
        "positive_total_pnl": float(positive_total_pnl),
        "unassigned_positive_pnl": float(unassigned_positive_pnl),
        "dominant_entry_year": dominant_entry_year,
        "dominant_year_positive_pnl": float(dominant_year_positive_pnl),
        "dominant_year_positive_pnl_share": float(dominant_share),
        "dominant_year_pnl_share": float(dominant_share),
        "dominant_year_positive_trade_count": positive_trade_count,
        "dominant_year_trade_count": positive_trade_count,
        "dominant_year_positive_symbol_count": positive_symbol_count,
        "dominant_year_symbol_count": positive_symbol_count,
        "top_trade_pnl_share_in_dominant_year": float(top_trade_pnl_share),
        "thresholds": _threshold_snapshot(),
    }
