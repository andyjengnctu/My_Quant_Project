"""Resolved execution policy for the non-blocking Trading Market Data V2 archive sidecar."""
from __future__ import annotations

from dataclasses import dataclass

from config.market_data import MARKET_DATA_V2_TRADING_SYNC_POLICY


@dataclass(frozen=True)
class MarketDataTradingSyncPolicy:
    enabled: bool
    execution_fail_closed: bool
    recent_repair_calendar_days: int
    event_repair_calendar_days: int


def get_market_data_trading_sync_policy() -> MarketDataTradingSyncPolicy:
    raw = MARKET_DATA_V2_TRADING_SYNC_POLICY
    if not isinstance(raw, dict):
        raise ValueError("MARKET_DATA_V2_TRADING_SYNC_POLICY 必須是 mapping")
    recent = int(raw.get("recent_repair_calendar_days") or 0)
    event = int(raw.get("event_repair_calendar_days") or 0)
    if recent < 1 or event < 1:
        raise ValueError("Market Data V2 Trading repair window 必須 >= 1 calendar day")
    return MarketDataTradingSyncPolicy(
        enabled=bool(raw.get("enabled")),
        execution_fail_closed=bool(raw.get("execution_fail_closed")),
        recent_repair_calendar_days=recent,
        event_repair_calendar_days=event,
    )


__all__ = ["MarketDataTradingSyncPolicy", "get_market_data_trading_sync_policy"]
