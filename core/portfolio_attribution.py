from __future__ import annotations

from collections import defaultdict
from typing import Any

from config.training_policy import (
    DOMINANT_YEAR_MAX_TOP_TRADE_PNL_SHARE,
    DOMINANT_YEAR_MIN_EFFECTIVE_POSITIVE_YEAR_COUNT,
    DOMINANT_YEAR_MIN_EFFECTIVE_SYMBOL_COUNT,
    DOMINANT_YEAR_MIN_EFFECTIVE_TRADE_COUNT,
)


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


def _effective_count_from_positive_values(values) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    total = sum(positives)
    if total <= 0.0:
        return 0.0
    share_square_sum = sum((value / total) ** 2 for value in positives)
    if share_square_sum <= 0.0:
        return 0.0
    return 1.0 / share_square_sum


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0


def build_dominant_year_dependency_diagnostics(closed_trades_stats) -> dict:
    """Diagnose whether positive train PnL is structurally concentrated.

    The warning intentionally requires both temporal concentration and narrow internal sources,
    so it does not penalize a parameter set only because it captured a genuine broad bull year.
    """

    trades = [dict(trade) for trade in list(closed_trades_stats or []) if isinstance(trade, dict)]
    positive_total_pnl = sum(max(_coerce_float(trade.get("pnl")), 0.0) for trade in trades)
    if positive_total_pnl <= 0.0:
        return {
            "dependency_warning": False,
            "dependency_status": "NO_POSITIVE_PNL",
            "dependency_reason": [],
            "positive_total_pnl": 0.0,
            "dominant_entry_year": None,
            "dominant_year_positive_pnl": 0.0,
            "dominant_year_pnl_share": 0.0,
            "effective_positive_year_count": 0.0,
            "dominant_year_trade_count": 0,
            "dominant_year_symbol_count": 0,
            "effective_trade_count_in_dominant_year": 0.0,
            "effective_symbol_count_in_dominant_year": 0.0,
            "top_trade_pnl_share_in_dominant_year": 0.0,
        }

    year_positive_pnl = defaultdict(float)
    for trade in trades:
        pnl = max(_coerce_float(trade.get("pnl")), 0.0)
        if pnl <= 0.0:
            continue
        entry_year = _resolve_trade_entry_year(trade)
        if entry_year is None:
            continue
        year_positive_pnl[int(entry_year)] += pnl

    if not year_positive_pnl:
        return {
            "dependency_warning": False,
            "dependency_status": "MISSING_ENTRY_YEAR",
            "dependency_reason": [],
            "positive_total_pnl": float(positive_total_pnl),
            "dominant_entry_year": None,
            "dominant_year_positive_pnl": 0.0,
            "dominant_year_pnl_share": 0.0,
            "effective_positive_year_count": 0.0,
            "dominant_year_trade_count": 0,
            "dominant_year_symbol_count": 0,
            "effective_trade_count_in_dominant_year": 0.0,
            "effective_symbol_count_in_dominant_year": 0.0,
            "top_trade_pnl_share_in_dominant_year": 0.0,
        }

    dominant_entry_year, dominant_year_positive_pnl = max(
        year_positive_pnl.items(),
        key=lambda item: (float(item[1]), -int(item[0])),
    )
    effective_positive_year_count = _effective_count_from_positive_values(year_positive_pnl.values())
    dominant_year_trades = [
        trade for trade in trades if _resolve_trade_entry_year(trade) == int(dominant_entry_year)
    ]
    dominant_positive_trade_pnls = [
        max(_coerce_float(trade.get("pnl")), 0.0)
        for trade in dominant_year_trades
        if max(_coerce_float(trade.get("pnl")), 0.0) > 0.0
    ]
    effective_trade_count = _effective_count_from_positive_values(dominant_positive_trade_pnls)

    symbol_positive_pnl = defaultdict(float)
    for trade in dominant_year_trades:
        pnl = max(_coerce_float(trade.get("pnl")), 0.0)
        if pnl <= 0.0:
            continue
        ticker = str(trade.get("ticker") or "").strip()
        if ticker:
            symbol_positive_pnl[ticker] += pnl
    effective_symbol_count = _effective_count_from_positive_values(symbol_positive_pnl.values())
    top_trade_pnl_share = _safe_ratio(max(dominant_positive_trade_pnls, default=0.0), dominant_year_positive_pnl)

    temporal_concentration = effective_positive_year_count < float(DOMINANT_YEAR_MIN_EFFECTIVE_POSITIVE_YEAR_COUNT)
    trade_concentration = effective_trade_count < float(DOMINANT_YEAR_MIN_EFFECTIVE_TRADE_COUNT)
    symbol_concentration = effective_symbol_count < float(DOMINANT_YEAR_MIN_EFFECTIVE_SYMBOL_COUNT)
    top_trade_outlier = top_trade_pnl_share >= float(DOMINANT_YEAR_MAX_TOP_TRADE_PNL_SHARE)
    dependency_warning = bool(
        temporal_concentration and (trade_concentration or symbol_concentration or top_trade_outlier)
    )

    dependency_reason = []
    if temporal_concentration:
        dependency_reason.append("effective_positive_year_count_lt_min")
    if trade_concentration:
        dependency_reason.append("effective_trade_count_in_dominant_year_lt_min")
    if symbol_concentration:
        dependency_reason.append("effective_symbol_count_in_dominant_year_lt_min")
    if top_trade_outlier:
        dependency_reason.append("top_trade_pnl_share_in_dominant_year_gte_max")

    return {
        "dependency_warning": dependency_warning,
        "dependency_status": "WARN" if dependency_warning else "OK",
        "dependency_reason": dependency_reason if dependency_warning else [],
        "positive_total_pnl": float(positive_total_pnl),
        "dominant_entry_year": int(dominant_entry_year),
        "dominant_year_positive_pnl": float(dominant_year_positive_pnl),
        "dominant_year_pnl_share": float(_safe_ratio(dominant_year_positive_pnl, positive_total_pnl)),
        "effective_positive_year_count": float(effective_positive_year_count),
        "dominant_year_trade_count": int(len(dominant_year_trades)),
        "dominant_year_symbol_count": int(len(symbol_positive_pnl)),
        "effective_trade_count_in_dominant_year": float(effective_trade_count),
        "effective_symbol_count_in_dominant_year": float(effective_symbol_count),
        "top_trade_pnl_share_in_dominant_year": float(top_trade_pnl_share),
        "thresholds": {
            "min_effective_positive_year_count": float(DOMINANT_YEAR_MIN_EFFECTIVE_POSITIVE_YEAR_COUNT),
            "min_effective_trade_count": float(DOMINANT_YEAR_MIN_EFFECTIVE_TRADE_COUNT),
            "min_effective_symbol_count": float(DOMINANT_YEAR_MIN_EFFECTIVE_SYMBOL_COUNT),
            "max_top_trade_pnl_share": float(DOMINANT_YEAR_MAX_TOP_TRADE_PNL_SHARE),
        },
    }
