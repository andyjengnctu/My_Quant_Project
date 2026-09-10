"""Shared completed-daily Market Data V2 context for live Trading positions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.data_utils import get_required_min_rows
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_identity import normalize_trading_date
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_consumer import load_trading_v2_sanitized_ohlcv_frame
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
    "resolve_trading_strategy_position_sources",
]
