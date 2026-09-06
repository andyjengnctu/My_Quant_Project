"""Canonical identity and date normalization for live Trading domain entities."""
from __future__ import annotations

from datetime import date, datetime


def normalize_trading_ticker(value: object) -> str:
    """Return the canonical Trading ticker identity or fail fast."""

    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("Trading ticker 不可為空")
    if any(ch.isspace() for ch in ticker):
        raise ValueError(f"Trading ticker 不可包含空白: {ticker!r}")
    return ticker


def normalize_trading_date(
    value: object | None,
    *,
    field_name: str = "date",
    allow_none: bool = True,
) -> str | None:
    """Return canonical ``YYYY-MM-DD`` for Trading dates.

    Trading scanner rows, account positions, broker orders and fill
    reconciliation all compare calendar dates.  They must therefore share one
    parser instead of mixing ``datetime.date``, pandas and free-form strings.
    """

    if value is None or str(value).strip() == "":
        if allow_none:
            return None
        raise ValueError(f"{field_name} 必填")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} 必須是 YYYY-MM-DD: {value!r}") from exc


def require_trading_date_after(
    value: object,
    *,
    after: object,
    field_name: str,
    after_field_name: str,
) -> str:
    """Normalize and require ``value`` to be strictly later than ``after``."""

    current = normalize_trading_date(value, field_name=field_name, allow_none=False)
    reference = normalize_trading_date(after, field_name=after_field_name, allow_none=False)
    if current <= reference:
        raise ValueError(f"{field_name} 必須晚於 {after_field_name}: {current} <= {reference}")
    return current


def require_trading_date_not_before(
    value: object,
    *,
    earliest: object,
    field_name: str,
    earliest_field_name: str,
) -> str:
    """Normalize and require ``value`` to be on/after ``earliest``."""

    current = normalize_trading_date(value, field_name=field_name, allow_none=False)
    reference = normalize_trading_date(earliest, field_name=earliest_field_name, allow_none=False)
    if current < reference:
        raise ValueError(f"{field_name} 不得早於 {earliest_field_name}: {current} < {reference}")
    return current


__all__ = [
    "normalize_trading_ticker",
    "normalize_trading_date",
    "require_trading_date_after",
    "require_trading_date_not_before",
]
