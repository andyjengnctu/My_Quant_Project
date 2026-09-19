from __future__ import annotations

TRADING_SOURCE_FAMILY_STRATEGY = "strategy"
TRADING_SOURCE_FAMILY_CUSTOM = "custom"

_STRATEGY_SOURCE_KEYS = frozenset({
    "scanner",
    "scanner_strategy",
    "strategy",
    "strategy_fill",
    "策略",
    "策略成交",
})
_CUSTOM_SOURCE_KEYS = frozenset({
    "manual",
    "manual_selected",
    "manual_adopted",
    "manual_managed",
    "custom",
    "自選",
    "手選",
    "手動成交",
    "手動管理",
    "自行買入",
})


def normalize_trading_source_family(*values: object) -> str | None:
    """Return the canonical strategy/custom family without changing raw lineage truth."""

    normalized = {
        str(value or "").strip().lower()
        for value in values
        if str(value or "").strip()
    }
    if normalized.intersection(_STRATEGY_SOURCE_KEYS):
        return TRADING_SOURCE_FAMILY_STRATEGY
    if normalized.intersection(_CUSTOM_SOURCE_KEYS):
        return TRADING_SOURCE_FAMILY_CUSTOM
    return None


__all__ = [
    "TRADING_SOURCE_FAMILY_CUSTOM",
    "TRADING_SOURCE_FAMILY_STRATEGY",
    "normalize_trading_source_family",
]
