"""Canonical market-evidence validation for manually recorded broker fills.

The account journal remains broker truth, but an explicitly entered fill must be
compatible with verified raw market evidence before it is accepted.  This module
uses ``TaiwanStockPrice`` only as raw execution evidence; strategy/model price
semantics continue to use the canonical adjusted-price source.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import pandas as pd

from core.exact_accounting import MILLI_SCALE, price_to_milli, round_price_to_tick_milli
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from services.trading.market_data_v2_view import TradingMarketDataV2View


TRADING_FILL_EVIDENCE_DATASET = "TaiwanStockPrice"
_TAIPEI = timezone(timedelta(hours=8))


def _current_taipei_date() -> date:
    return datetime.now(_TAIPEI).date()


def _positive_price_milli(value, *, field_name: str) -> int:
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name}必須是數字") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise ValueError(f"{field_name}必須是大於 0 的有限數值")
    # AI: Validate the entered broker fact before fixed-point conversion. A
    # sub-milli off-tick value must not silently round into an admissible price.
    milli_value = int(price_to_milli(decimal_value))
    if decimal_value != Decimal(milli_value) / Decimal(MILLI_SCALE):
        raise ValueError(f"{field_name} {value} 不符合台股合法跳動單位")
    return milli_value



def validate_trading_fill_quantity(value) -> int:
    """AI: Reject fractional/nonfinite quantities before any int coercion."""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("成交股數必須是正整數") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError("成交股數必須是正整數")
    return int(number)


def validate_pending_fill_terms(entry, *, qty, price, trade_date) -> None:
    """AI: Shared original-intent boundaries for a fill and later corrections."""
    count = validate_trading_fill_quantity(qty)
    if count > int(entry.get("planned_qty") or 0):
        raise ValueError("成交股數不得超過原掛單規劃股數")
    fill_date = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    order_date = normalize_trading_date(entry.get("planned_trade_date"), field_name="planned_trade_date", allow_none=False)
    if fill_date <= order_date:
        raise ValueError(f"成交日 {fill_date} 必須嚴格晚於掛單日 {order_date}")
    price_milli = _positive_price_milli(price, field_name="成交價")
    limit = entry.get("limit_price")
    if limit is not None and price_milli > price_to_milli(limit):
        raise ValueError("成交價不得高於原掛單買入限價")


def _market_number(row: pd.Series, field: str, *, ticker: str, trade_date: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
    if pd.isna(value) or not math.isfinite(float(value)):
        raise ValueError(f"{ticker} {trade_date} raw 市場資料的 {field} 不合法")
    return float(value)


def load_trading_actual_fill_market_evidence(
    project_root,
    *,
    ticker: object,
    trade_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> dict[str, object]:
    """Load exact-date raw OHLCV evidence used by all manual fill validation.

    Keeping the row lookup here prevents the Workbench option builders from
    re-implementing a second definition of a valid market-evidence day.
    """

    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    date_text = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    view = market_view or TradingMarketDataV2View.open(root)
    frame = view.read_dataset_frame(
        TRADING_FILL_EVIDENCE_DATASET,
        columns=("date", "stock_id", "max", "min", "Trading_Volume"),
        data_id=ticker_key,
        start_date=date_text,
        end_date=date_text,
    )
    if frame.empty:
        raise ValueError(
            f"{ticker_key} {date_text} 沒有正式 TaiwanStockPrice 交易資料；"
            "請先更新 Trading 資料或確認成交日"
        )
    if len(frame.index) != 1:
        raise RuntimeError(f"{ticker_key} {date_text} raw 市場證據不是唯一 row")

    row = frame.iloc[0]
    row_date = normalize_trading_date(row.get("date"), field_name="market.date", allow_none=False)
    row_ticker = normalize_trading_ticker(row.get("stock_id"))
    if row_date != date_text or row_ticker != ticker_key:
        raise RuntimeError(f"{ticker_key} {date_text} raw 市場證據 identity 不一致")

    low = _market_number(row, "min", ticker=ticker_key, trade_date=date_text)
    high = _market_number(row, "max", ticker=ticker_key, trade_date=date_text)
    volume = _market_number(row, "Trading_Volume", ticker=ticker_key, trade_date=date_text)
    if low <= 0 or high <= 0 or high < low:
        raise ValueError(f"{ticker_key} {date_text} raw 高低價資料不合法")
    if volume <= 0:
        raise ValueError(f"{ticker_key} {date_text} 沒有有效成交量，不接受成交登錄")
    return {
        "ticker": ticker_key,
        "trade_date": date_text,
        "market_low": low,
        "market_high": high,
        "trading_volume": volume,
        "evidence_dataset": TRADING_FILL_EVIDENCE_DATASET,
        "evidence_semantics": "daily_ohlcv_possible_fill",
    }


def list_trading_actual_fill_dates(
    project_root,
    *,
    ticker: object,
    market_view: TradingMarketDataV2View | None = None,
    today: date | str | None = None,
    latest_date: date | str | None = None,
    after_date: date | str | None = None,
) -> tuple[str, ...]:
    """Return locally evidenced positive-volume dates eligible for a fill.

    ``latest_date`` clamps the list to the currently finalized Trading horizon.
    ``after_date`` is an exclusive lower bound used by pending-order fills so
    the UI and service share the same ``fill_date > order_date`` contract.
    """

    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    if today is None:
        today_date = _current_taipei_date()
    elif isinstance(today, date):
        today_date = today
    else:
        today_date = date.fromisoformat(str(today))
    latest_bound = today_date
    if latest_date is not None:
        latest_bound = min(latest_bound, latest_date if isinstance(latest_date, date) else date.fromisoformat(str(latest_date)))
    exclusive_lower = None if after_date is None else (after_date if isinstance(after_date, date) else date.fromisoformat(str(after_date)))
    view = market_view or TradingMarketDataV2View.open(root)
    frame = view.read_dataset_frame(
        TRADING_FILL_EVIDENCE_DATASET,
        columns=("date", "stock_id", "Trading_Volume"),
        data_id=ticker_key,
    )
    if frame.empty:
        return ()
    dates = pd.to_datetime(frame.get("date"), errors="coerce")
    volumes = pd.to_numeric(frame.get("Trading_Volume"), errors="coerce")
    stock_ids = frame.get("stock_id")
    if stock_ids is None:
        return ()
    normalized_tickers = stock_ids.astype(str).str.strip().str.upper()
    mask = dates.notna() & volumes.notna() & (volumes > 0) & (normalized_tickers == ticker_key)
    if not mask.any():
        return ()
    eligible = dates.loc[mask].dt.date
    values = sorted({
        value.isoformat()
        for value in eligible
        if value <= latest_bound and (exclusive_lower is None or value > exclusive_lower)
    })
    return tuple(values)


def validate_trading_actual_fill(
    project_root,
    *,
    ticker: object,
    price,
    trade_date: object,
    market_view: TradingMarketDataV2View | None = None,
    today: date | str | None = None,
    latest_date: date | str | None = None,
) -> dict[str, object]:
    """Validate one actual broker fill against exact-date raw market evidence.

    Daily OHLCV proves that the entered price was *possible* on that date; it
    does not claim tick-by-tick proof that an exact trade print existed.
    """

    ticker_key = normalize_trading_ticker(ticker)
    date_text = normalize_trading_date(trade_date, field_name="trade_date", allow_none=False)
    if today is None:
        today_date = _current_taipei_date()
    elif isinstance(today, date):
        today_date = today
    else:
        today_date = date.fromisoformat(str(today))
    trade_day = date.fromisoformat(date_text)
    if trade_day > today_date:
        raise ValueError(f"成交日 {date_text} 尚未到來；不可登錄未來成交")
    if latest_date is not None:
        latest_bound = latest_date if isinstance(latest_date, date) else date.fromisoformat(str(latest_date))
        if trade_day > latest_bound:
            raise ValueError(f"成交日 {date_text} 超過目前最新 finalized 日期 {latest_bound.isoformat()}")

    fill_price_milli = _positive_price_milli(price, field_name="成交價")
    rounded_milli = int(round_price_to_tick_milli(price, direction="nearest", ticker=ticker_key))
    if fill_price_milli != rounded_milli:
        raise ValueError(f"{ticker_key} 成交價 {price} 不符合台股合法跳動單位")

    evidence = load_trading_actual_fill_market_evidence(
        project_root,
        ticker=ticker_key,
        trade_date=date_text,
        market_view=market_view,
    )
    low = float(evidence["market_low"])
    high = float(evidence["market_high"])
    low_milli = int(price_to_milli(low))
    high_milli = int(price_to_milli(high))
    if fill_price_milli < low_milli or fill_price_milli > high_milli:
        raise ValueError(
            f"{ticker_key} {date_text} 成交價 {price} 超出當日價格範圍 "
            f"[{low:g}, {high:g}]"
        )

    result = dict(evidence)
    result["price_milli"] = fill_price_milli
    return result


__all__ = [
    "TRADING_FILL_EVIDENCE_DATASET",
    "list_trading_actual_fill_dates",
    "load_trading_actual_fill_market_evidence",
    "validate_trading_actual_fill",
    "validate_trading_fill_quantity",
    "validate_pending_fill_terms",
]
