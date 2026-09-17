"""Canonical Trading order-form date/price constraints.

The Workbench may make invalid inputs hard to select, but the same contracts are
also callable from service code so UI affordances never become the authority.
Pending-entry constraints use only decision-time schedule/reference evidence;
actual-fill constraints may use exact-date raw OHLCV because the fill is already
broker truth being recorded after the fact.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from core.exact_accounting import (
    get_tick_milli,
    infer_security_profile,
    milli_to_price,
    price_to_milli,
    round_price_to_tick_milli,
)
from core.price_utils import calc_limit_down_price, calc_limit_up_price
from core.runtime_utils import get_taipei_now
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from services.trading.actual_fill_validation import (
    list_trading_actual_fill_dates,
    load_trading_actual_fill_market_evidence,
)
from services.trading.market_data_v2_view import TradingMarketDataV2View


ORDER_FORM_DATE_KIND_PENDING = "pending_order"
ORDER_FORM_DATE_KIND_FILL = "actual_fill"
TRADING_ORDER_CALENDAR_DATASET = "TaiwanStockTradingDate"
TRADING_ORDER_REFERENCE_DATASET = "TaiwanStockPrice"


def _price_text_from_milli(value: int) -> str:
    return f"{float(milli_to_price(int(value))):.3f}".rstrip("0").rstrip(".")


def build_legal_tick_price_options(
    low_price,
    high_price,
    *,
    ticker: object,
    security_profile=None,
) -> tuple[str, ...]:
    """Enumerate every legal Taiwan tick within an inclusive price band."""

    ticker_key = normalize_trading_ticker(ticker)
    profile = infer_security_profile(ticker_key) if security_profile is None else security_profile
    low_milli = int(round_price_to_tick_milli(low_price, direction="up", security_profile=profile))
    high_milli = int(round_price_to_tick_milli(high_price, direction="down", security_profile=profile))
    if low_milli <= 0 or high_milli < low_milli:
        return ()

    values: list[str] = []
    current = low_milli
    # A valid Taiwan order band is small, but keep a structural cap as corruption
    # protection rather than allowing a malformed market row to loop forever.
    for _ in range(100_000):
        if current > high_milli:
            break
        values.append(_price_text_from_milli(current))
        step = int(get_tick_milli(current, security_profile=profile))
        if step <= 0:
            raise RuntimeError("Trading tick ladder 回傳非正數跳動單位")
        current += step
    else:
        raise RuntimeError("Trading 合法價格集合異常過大；拒絕建立輸入選項")
    return tuple(values)


def _resolve_pending_order_reference_close(
    project_root,
    *,
    ticker: str,
    information_date: str,
    market_view: TradingMarketDataV2View,
) -> float:
    frame = market_view.read_dataset_frame(
        TRADING_ORDER_REFERENCE_DATASET,
        columns=("date", "stock_id", "close"),
        data_id=ticker,
        start_date=information_date,
        end_date=information_date,
    )
    if frame.empty or len(frame.index) != 1:
        raise ValueError(f"{ticker} {information_date} 缺盤前法定價格帶參考收盤價")
    row = frame.iloc[0]
    row_date = normalize_trading_date(row.get("date"), field_name="market.date", allow_none=False)
    row_ticker = normalize_trading_ticker(row.get("stock_id"))
    close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
    if row_date != information_date or row_ticker != ticker or pd.isna(close) or float(close) <= 0:
        raise ValueError(f"{ticker} {information_date} 盤前法定價格帶參考資料不合法")
    return float(close)


def resolve_next_pending_order_date(
    project_root,
    *,
    information_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> str:
    """Return the first currently-known scheduled session after information date."""

    root = Path(project_root).resolve()
    info_date = normalize_trading_date(information_date, field_name="information_date", allow_none=False)
    view = market_view or TradingMarketDataV2View.open(root)
    frame = view.read_dataset_frame(
        TRADING_ORDER_CALENDAR_DATASET,
        columns=("date",),
        start_date=info_date,
    )
    if frame.empty:
        raise ValueError(f"{info_date} 之後沒有可用 TaiwanStockTradingDate；無法建立盤前掛單日期")
    values = pd.to_datetime(frame.get("date"), errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    future = sorted({value for value in values.tolist() if value > info_date})
    if not future:
        raise ValueError(f"{info_date} 之後沒有已知下一交易日；請先更新 Trading 市場資料")
    return future[0]


def validate_pending_order_trade_date(
    project_root,
    *,
    information_date: object,
    planned_trade_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> str:
    planned = normalize_trading_date(planned_trade_date, field_name="planned_trade_date", allow_none=False)
    expected = resolve_next_pending_order_date(
        project_root,
        information_date=information_date,
        market_view=market_view,
    )
    if planned != expected:
        raise ValueError(f"掛單日 {planned} 不符合目前盤前可執行交易日 {expected}")
    return planned




def resolve_pending_order_price_band(
    project_root,
    *,
    ticker: object,
    information_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> tuple[float, float, float]:
    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    info_date = normalize_trading_date(information_date, field_name="information_date", allow_none=False)
    view = market_view or TradingMarketDataV2View.open(root)
    reference_price = _resolve_pending_order_reference_close(
        root, ticker=ticker_key, information_date=info_date, market_view=view
    )
    profile = infer_security_profile(ticker_key)
    low = float(calc_limit_down_price(reference_price, ticker=ticker_key, security_profile=profile))
    high = float(calc_limit_up_price(reference_price, ticker=ticker_key, security_profile=profile))
    if low <= 0 or high < low:
        raise ValueError(f"{ticker_key} {info_date} 無法建立合法盤前價格帶")
    return low, high, reference_price


def validate_pending_order_limit_price(
    project_root,
    *,
    ticker: object,
    information_date: object,
    limit_price,
    market_view: TradingMarketDataV2View | None = None,
) -> float:
    ticker_key = normalize_trading_ticker(ticker)
    try:
        price = float(limit_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("掛單限價必須是有效數字") from exc
    if not pd.notna(price) or price <= 0:
        raise ValueError("掛單限價必須 > 0")
    rounded_milli = int(round_price_to_tick_milli(price, direction="nearest", ticker=ticker_key))
    price_milli = int(price_to_milli(price))
    if rounded_milli != price_milli:
        raise ValueError(f"{ticker_key} 掛單限價 {limit_price} 不符合台股合法跳動單位")
    low, high, _reference = resolve_pending_order_price_band(
        project_root, ticker=ticker_key, information_date=information_date, market_view=market_view
    )
    if price_milli < int(price_to_milli(low)) or price_milli > int(price_to_milli(high)):
        raise ValueError(f"{ticker_key} 掛單限價 {price:g} 超出盤前合法價格帶 [{low:g}, {high:g}]")
    return price


def build_trading_pending_order_form_constraints(
    project_root,
    *,
    ticker: object,
    information_date: object,
    selected_date: object | None = None,
    include_fill_dates: bool = False,
    pending_limit_price=None,
    market_view: TradingMarketDataV2View | None = None,
    today: date | str | None = None,
) -> dict[str, Any]:
    """Build one state-aware date/price option set for the shared pending form."""

    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    info_date = normalize_trading_date(information_date, field_name="information_date", allow_none=False)
    view = market_view or TradingMarketDataV2View.open(root)
    preferred_order_date = resolve_next_pending_order_date(
        root,
        information_date=info_date,
        market_view=view,
    )
    order_dates = (preferred_order_date,)
    fill_dates = (
        list_trading_actual_fill_dates(root, ticker=ticker_key, market_view=view, today=today)
        if include_fill_dates
        else ()
    )
    allowed_dates = tuple(sorted(set(order_dates).union(fill_dates)))
    chosen = normalize_trading_date(
        selected_date or preferred_order_date,
        field_name="selected_date",
        allow_none=False,
    )

    selected_kind = None
    price_low = None
    price_high = None
    reference_price = None
    if include_fill_dates and chosen in set(fill_dates):
        selected_kind = ORDER_FORM_DATE_KIND_FILL
        evidence = load_trading_actual_fill_market_evidence(
            root,
            ticker=ticker_key,
            trade_date=chosen,
            market_view=view,
        )
        price_low = float(evidence["market_low"])
        price_high = float(evidence["market_high"])
        if pending_limit_price is not None:
            price_high = min(price_high, float(pending_limit_price))
    elif chosen in set(order_dates):
        selected_kind = ORDER_FORM_DATE_KIND_PENDING
        price_low, price_high, reference_price = resolve_pending_order_price_band(
            root,
            ticker=ticker_key,
            information_date=info_date,
            market_view=view,
        )

    options: tuple[str, ...] = ()
    if price_low is not None and price_high is not None and price_high >= price_low:
        options = build_legal_tick_price_options(price_low, price_high, ticker=ticker_key)

    return {
        "ticker": ticker_key,
        "information_date": info_date,
        "preferred_order_date": preferred_order_date,
        "order_dates": order_dates,
        "fill_dates": tuple(fill_dates),
        "allowed_dates": allowed_dates,
        "selected_date": chosen,
        "selected_date_kind": selected_kind,
        "price_options": options,
        "price_min": price_low,
        "price_max": price_high,
        "reference_price": reference_price,
    }


__all__ = [
    "ORDER_FORM_DATE_KIND_FILL",
    "ORDER_FORM_DATE_KIND_PENDING",
    "TRADING_ORDER_CALENDAR_DATASET",
    "TRADING_ORDER_REFERENCE_DATASET",
    "build_legal_tick_price_options",
    "build_trading_pending_order_form_constraints",
    "resolve_next_pending_order_date",
    "resolve_pending_order_price_band",
    "validate_pending_order_limit_price",
    "validate_pending_order_trade_date",
]
