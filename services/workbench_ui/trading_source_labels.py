from __future__ import annotations

from core.trading_source import (
    TRADING_SOURCE_FAMILY_CUSTOM,
    TRADING_SOURCE_FAMILY_STRATEGY,
    normalize_trading_source_family,
)

TRADING_SOURCE_STRATEGY_LABEL = "策略"
TRADING_SOURCE_CUSTOM_LABEL = "自選"


def trading_source_display_label(
    *,
    origin: object = None,
    source: object = None,
    default: str = "-",
) -> str:
    """Map canonical Trading source family to the two user-facing labels."""

    family = normalize_trading_source_family(origin, source)
    if family == TRADING_SOURCE_FAMILY_STRATEGY:
        return TRADING_SOURCE_STRATEGY_LABEL
    if family == TRADING_SOURCE_FAMILY_CUSTOM:
        return TRADING_SOURCE_CUSTOM_LABEL
    return str(default)


__all__ = [
    "TRADING_SOURCE_CUSTOM_LABEL",
    "TRADING_SOURCE_STRATEGY_LABEL",
    "trading_source_display_label",
]
