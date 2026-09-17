"""Canonical Trading order-form date/price constraints.

Workbench widgets consume these contracts to disable invalid dates/prices before
submission.  Service submit paths re-run the same validators, so UI state is
never authority.

Historical pending-entry backfill is intentionally bounded by the latest
finalized Trading information date.  Because every selectable historical date
is already finalized, the shared order form can constrain its price list to the
exact-date raw daily price range plus the Taiwan tick ladder.  Actual fills add
the stronger chronology rule that fill_date must be strictly after order_date.
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


def _calendar_dates_through(
    project_root,
    *,
    latest_finalized_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> tuple[str, ...]:
    root = Path(project_root).resolve()
    latest = normalize_trading_date(
        latest_finalized_date,
        field_name="latest_finalized_date",
        allow_none=False,
    )
    view = market_view or TradingMarketDataV2View.open(root)
    frame = view.read_dataset_frame(
        TRADING_ORDER_CALENDAR_DATASET,
        columns=("date",),
        end_date=latest,
    )
    if frame.empty:
        return ()
    values = pd.to_datetime(frame.get("date"), errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    return tuple(sorted({value for value in values.tolist() if value <= latest}))


def list_trading_pending_order_dates(
    project_root,
    *,
    ticker: object,
    latest_finalized_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> tuple[str, ...]:
    """Return historical dates eligible for a backfilled pending order.

    A selectable order date must be a canonical Trading session, must not be
    later than the latest finalized date, and must have positive-volume raw
    evidence for this ticker.  The last condition guarantees that the shared
    price combobox can be built from an exact-date range instead of accepting a
    free-form price with unknown day context.
    """

    root = Path(project_root).resolve()
    latest = normalize_trading_date(
        latest_finalized_date,
        field_name="latest_finalized_date",
        allow_none=False,
    )
    view = market_view or TradingMarketDataV2View.open(root)
    calendar_dates = set(
        _calendar_dates_through(
            root,
            latest_finalized_date=latest,
            market_view=view,
        )
    )
    evidence_dates = set(
        list_trading_actual_fill_dates(
            root,
            ticker=ticker,
            market_view=view,
            latest_date=latest,
        )
    )
    return tuple(sorted(calendar_dates.intersection(evidence_dates)))


def resolve_preferred_pending_order_date(
    project_root,
    *,
    ticker: object,
    latest_finalized_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> str:
    dates = list_trading_pending_order_dates(
        project_root,
        ticker=ticker,
        latest_finalized_date=latest_finalized_date,
        market_view=market_view,
    )
    if not dates:
        ticker_key = normalize_trading_ticker(ticker)
        latest = normalize_trading_date(latest_finalized_date, allow_none=False)
        raise ValueError(f"{ticker_key} 截至 {latest} 沒有可補登的合法掛單日期")
    return dates[-1]


def resolve_next_pending_order_date(
    project_root,
    *,
    information_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> str:
    """Compatibility helper for legacy callers that need the next scheduled day.

    New Workbench pending-entry flows do not use this helper; they use
    ``list_trading_pending_order_dates`` bounded by finalized history.
    """

    root = Path(project_root).resolve()
    info_date = normalize_trading_date(information_date, field_name="information_date", allow_none=False)
    view = market_view or TradingMarketDataV2View.open(root)
    frame = view.read_dataset_frame(
        TRADING_ORDER_CALENDAR_DATASET,
        columns=("date",),
        start_date=info_date,
    )
    if frame.empty:
        raise ValueError(f"{info_date} 之後沒有可用 TaiwanStockTradingDate")
    values = pd.to_datetime(frame.get("date"), errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    future = sorted({value for value in values.tolist() if value > info_date})
    if not future:
        raise ValueError(f"{info_date} 之後沒有已知下一交易日")
    return future[0]


def validate_pending_order_trade_date(
    project_root,
    *,
    ticker: object,
    latest_finalized_date: object,
    planned_trade_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> str:
    planned = normalize_trading_date(planned_trade_date, field_name="planned_trade_date", allow_none=False)
    allowed = set(
        list_trading_pending_order_dates(
            project_root,
            ticker=ticker,
            latest_finalized_date=latest_finalized_date,
            market_view=market_view,
        )
    )
    if planned not in allowed:
        latest = normalize_trading_date(latest_finalized_date, allow_none=False)
        ticker_key = normalize_trading_ticker(ticker)
        raise ValueError(
            f"{ticker_key} 掛單日 {planned} 不是截至最新 finalized 日期 {latest} 的合法可補登交易日"
        )
    return planned


def resolve_pending_order_price_band(
    project_root,
    *,
    ticker: object,
    planned_trade_date: object,
    market_view: TradingMarketDataV2View | None = None,
) -> tuple[float, float, float | None]:
    """Resolve exact-date possible order-price band for historical backfill."""

    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    planned = normalize_trading_date(planned_trade_date, field_name="planned_trade_date", allow_none=False)
    view = market_view or TradingMarketDataV2View.open(root)
    evidence = load_trading_actual_fill_market_evidence(
        root,
        ticker=ticker_key,
        trade_date=planned,
        market_view=view,
    )
    low = float(evidence["market_low"])
    high = float(evidence["market_high"])
    if low <= 0 or high < low:
        raise ValueError(f"{ticker_key} {planned} 無法建立合法價格區間")
    return low, high, None


def validate_pending_order_limit_price(
    project_root,
    *,
    ticker: object,
    planned_trade_date: object,
    latest_finalized_date: object,
    limit_price,
    market_view: TradingMarketDataV2View | None = None,
) -> float:
    ticker_key = normalize_trading_ticker(ticker)
    planned = validate_pending_order_trade_date(
        project_root,
        ticker=ticker_key,
        latest_finalized_date=latest_finalized_date,
        planned_trade_date=planned_trade_date,
        market_view=market_view,
    )
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
        project_root,
        ticker=ticker_key,
        planned_trade_date=planned,
        market_view=market_view,
    )
    if price_milli < int(price_to_milli(low)) or price_milli > int(price_to_milli(high)):
        raise ValueError(
            f"{ticker_key} {planned} 掛單價格 {price:g} 不在該日可證實價格區間 [{low:g}, {high:g}]"
        )
    return price


def build_trading_actual_fill_form_constraints(
    project_root,
    *,
    ticker: object,
    selected_date: object | None = None,
    latest_finalized_date: object | None = None,
    earliest_exclusive_date: object | None = None,
    max_price=None,
    market_view: TradingMarketDataV2View | None = None,
    auto_select_preferred: bool = True,
) -> dict[str, Any]:
    """Build canonical actual-fill calendar and price options."""

    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    view = market_view or TradingMarketDataV2View.open(root)
    latest = None if latest_finalized_date is None else normalize_trading_date(
        latest_finalized_date,
        field_name="latest_finalized_date",
        allow_none=False,
    )
    earliest = None if earliest_exclusive_date is None else normalize_trading_date(
        earliest_exclusive_date,
        field_name="earliest_exclusive_date",
        allow_none=False,
    )
    fill_dates = list_trading_actual_fill_dates(
        root,
        ticker=ticker_key,
        market_view=view,
        latest_date=latest,
        after_date=earliest,
    )
    preferred = fill_dates[-1] if fill_dates else None
    chosen = None
    if selected_date is not None and str(selected_date).strip():
        chosen = normalize_trading_date(selected_date, field_name="selected_date", allow_none=False)
    elif preferred is not None and auto_select_preferred:
        chosen = preferred

    price_low = None
    price_high = None
    options: tuple[str, ...] = ()
    if chosen in set(fill_dates):
        evidence = load_trading_actual_fill_market_evidence(
            root,
            ticker=ticker_key,
            trade_date=chosen,
            market_view=view,
        )
        price_low = float(evidence["market_low"])
        price_high = float(evidence["market_high"])
        if max_price is not None:
            price_high = min(price_high, float(max_price))
        if price_high >= price_low:
            options = build_legal_tick_price_options(price_low, price_high, ticker=ticker_key)

    return {
        "ticker": ticker_key,
        "latest_finalized_date": latest,
        "earliest_exclusive_date": earliest,
        "preferred_fill_date": preferred,
        "fill_dates": tuple(fill_dates),
        "allowed_dates": tuple(fill_dates),
        "selected_date": chosen,
        "selected_date_kind": ORDER_FORM_DATE_KIND_FILL if chosen in set(fill_dates) else None,
        "price_options": options,
        "price_min": price_low,
        "price_max": price_high,
        "price_evidence": "daily_ohlcv_possible_fill",
    }


def build_trading_pending_order_form_constraints(
    project_root,
    *,
    ticker: object,
    information_date: object,
    selected_date: object | None = None,
    include_fill_dates: bool = False,
    pending_limit_price=None,
    pending_order_date: object | None = None,
    latest_finalized_date: object | None = None,
    market_view: TradingMarketDataV2View | None = None,
    today: date | str | None = None,
) -> dict[str, Any]:
    """Build one state-aware date/price option set for the shared pending form."""

    del today  # retained for call-site compatibility; finalized date is authoritative.
    root = Path(project_root).resolve()
    ticker_key = normalize_trading_ticker(ticker)
    info_date = normalize_trading_date(information_date, field_name="information_date", allow_none=False)
    latest = normalize_trading_date(
        latest_finalized_date or info_date,
        field_name="latest_finalized_date",
        allow_none=False,
    )
    view = market_view or TradingMarketDataV2View.open(root)
    order_dates = list_trading_pending_order_dates(
        root,
        ticker=ticker_key,
        latest_finalized_date=latest,
        market_view=view,
    )
    preferred_order_date = order_dates[-1] if order_dates else None
    order_date_for_fill = None if pending_order_date is None else normalize_trading_date(
        pending_order_date,
        field_name="pending_order_date",
        allow_none=False,
    )
    fill_dates = ()
    if include_fill_dates:
        fill_dates = build_trading_actual_fill_form_constraints(
            root,
            ticker=ticker_key,
            latest_finalized_date=latest,
            earliest_exclusive_date=order_date_for_fill,
            max_price=pending_limit_price,
            market_view=view,
        )["fill_dates"]

    allowed_dates = tuple(sorted(set(order_dates).union(fill_dates)))
    chosen = None
    if selected_date is not None and str(selected_date).strip():
        chosen = normalize_trading_date(selected_date, field_name="selected_date", allow_none=False)
    elif preferred_order_date is not None:
        chosen = preferred_order_date

    selected_kind = None
    price_low = None
    price_high = None
    reference_price = None
    # Existing pending entries interpret dates strictly later than the frozen
    # order date as fill input; the order date itself remains editable order context.
    if (
        include_fill_dates
        and order_date_for_fill is not None
        and chosen is not None
        and chosen > order_date_for_fill
        and chosen in set(fill_dates)
    ):
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
            planned_trade_date=chosen,
            market_view=view,
        )

    options: tuple[str, ...] = ()
    if price_low is not None and price_high is not None and price_high >= price_low:
        options = build_legal_tick_price_options(price_low, price_high, ticker=ticker_key)

    return {
        "ticker": ticker_key,
        "information_date": info_date,
        "latest_finalized_date": latest,
        "preferred_order_date": preferred_order_date,
        "pending_order_date": order_date_for_fill,
        "order_dates": tuple(order_dates),
        "fill_dates": tuple(fill_dates),
        "allowed_dates": allowed_dates,
        "selected_date": chosen,
        "selected_date_kind": selected_kind,
        "price_options": options,
        "price_min": price_low,
        "price_max": price_high,
        "reference_price": reference_price,
        "price_evidence": (
            "daily_ohlcv_possible_fill" if selected_kind == ORDER_FORM_DATE_KIND_FILL
            else "daily_ohlcv_historical_order_range" if selected_kind == ORDER_FORM_DATE_KIND_PENDING
            else None
        ),
    }


__all__ = [
    "ORDER_FORM_DATE_KIND_FILL",
    "ORDER_FORM_DATE_KIND_PENDING",
    "TRADING_ORDER_CALENDAR_DATASET",
    "TRADING_ORDER_REFERENCE_DATASET",
    "build_legal_tick_price_options",
    "build_trading_actual_fill_form_constraints",
    "build_trading_pending_order_form_constraints",
    "list_trading_pending_order_dates",
    "resolve_next_pending_order_date",
    "resolve_pending_order_price_band",
    "resolve_preferred_pending_order_date",
    "validate_pending_order_limit_price",
    "validate_pending_order_trade_date",
]
