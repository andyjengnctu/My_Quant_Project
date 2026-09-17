"""Shared completed-daily Market Data V2 context for live Trading positions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_identity import normalize_trading_date
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_consumer import (
    build_trading_v2_ohlcv_frames,
    load_trading_v2_sanitized_ohlcv_frame,
)
from services.trading.market_data_v2_view import TradingMarketDataV2View
from services.trading.order_state import load_trading_order_state


def load_trading_position_market_frame(
    *,
    view: TradingMarketDataV2View,
    ticker: str,
    params,
    allowed_date: str,
) -> pd.DataFrame:
    return load_trading_v2_sanitized_ohlcv_frame(
        view,
        ticker=ticker,
        through_date=allowed_date,
        min_rows=get_required_min_rows(params),
    )


def load_trading_position_market_frames(
    *,
    view: TradingMarketDataV2View,
    params_by_ticker: dict[str, object],
    allowed_date: str,
) -> dict[str, pd.DataFrame]:
    """Batch-load completed-daily position frames without changing per-ticker semantics."""

    tickers = tuple(dict.fromkeys(str(ticker).strip() for ticker in params_by_ticker if str(ticker).strip()))
    if not tickers:
        return {}
    raw_by_ticker = build_trading_v2_ohlcv_frames(
        view,
        tickers=tickers,
        through_date=allowed_date,
    )
    cutoff = pd.Timestamp(allowed_date).normalize()
    clean_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        params = params_by_ticker[ticker]
        raw = raw_by_ticker[ticker]
        clean, _stats = sanitize_ohlcv_dataframe(
            raw,
            ticker,
            min_rows=int(get_required_min_rows(params)),
        )
        if pd.Timestamp(clean.index.max()).normalize() > cutoff:
            raise RuntimeError(
                f"Trading V2 consumer data 含超過 cutoff 的日K: {ticker} "
                f"latest={pd.Timestamp(clean.index.max()).strftime('%Y-%m-%d')} > allowed={allowed_date}"
            )
        clean_by_ticker[ticker] = clean
    return clean_by_ticker


def resolve_trading_strategy_position_sources(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], TradingMarketDataV2View | None]:
    account = load_trading_account_state(project_root, required=False)
    if account is None:
        return {}, {}, None
    orders = load_trading_order_state(project_root, required=False)
    if orders is None:
        orders = {"orders": {}}
    has_strategy_positions = any(
        isinstance(record, dict) and record.get("source") == POSITION_SOURCE_STRATEGY_FILL
        for record in (account.get("positions") or {}).values()
    )
    if not has_strategy_positions:
        return account, orders, None
    return account, orders, TradingMarketDataV2View.open(project_root)


__all__ = [
    "normalize_trading_date",
    "load_trading_position_market_frame",
    "load_trading_position_market_frames",
    "resolve_trading_strategy_position_sources",
]
