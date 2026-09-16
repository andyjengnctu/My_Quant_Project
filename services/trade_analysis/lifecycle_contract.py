"""Shared trade-lifecycle display contract for Research and Trading charts.

The strategy/execution domains own how a lifecycle state is produced.  This
module owns only the stable visual semantics of that state so Research replay
and Trading broker-truth projection cannot map SHADOW/POSITION geometry to
separate line families.
"""
from __future__ import annotations

from typing import Any


TRADE_LIFECYCLE_SIGNAL = "SIGNAL"
TRADE_LIFECYCLE_SHADOW = "SHADOW"
TRADE_LIFECYCLE_POSITION = "POSITION"
TRADE_LIFECYCLE_STATES = frozenset(
    {TRADE_LIFECYCLE_SIGNAL, TRADE_LIFECYCLE_SHADOW, TRADE_LIFECYCLE_POSITION}
)

TRADE_TRANSACTION_LINE_KEYS = (
    "stop_line",
    "tp_line",
    "limit_line",
    "entry_line",
    "shadow_stop_line",
    "shadow_tp_line",
    "shadow_limit_line",
    "shadow_entry_line",
)


def normalize_trade_lifecycle_state(value: object) -> str:
    state = str(value or "").strip().upper()
    if state not in TRADE_LIFECYCLE_STATES:
        raise ValueError(f"不支援的 trade lifecycle state: {value!r}")
    return state


def iter_trade_lifecycle_line_values(
    lifecycle_state: object,
    *,
    stop_price: Any = None,
    tp_price: Any = None,
    limit_price: Any = None,
    entry_price: Any = None,
):
    """Yield the canonical chart-line mapping for one lifecycle state.

    SIGNAL intentionally owns no transaction line on its information bar.
    SHADOW uses only the shadow line family.  POSITION uses only the actual line
    family.  Both Research and Trading consume this exact mapping.
    """
    state = normalize_trade_lifecycle_state(lifecycle_state)
    if state == TRADE_LIFECYCLE_SIGNAL:
        return ()
    prefix = "shadow_" if state == TRADE_LIFECYCLE_SHADOW else ""
    return (
        (f"{prefix}stop_line", stop_price),
        (f"{prefix}tp_line", tp_price),
        (f"{prefix}limit_line", limit_price),
        (f"{prefix}entry_line", entry_price),
    )


__all__ = [
    "TRADE_LIFECYCLE_POSITION",
    "TRADE_LIFECYCLE_SHADOW",
    "TRADE_LIFECYCLE_SIGNAL",
    "TRADE_LIFECYCLE_STATES",
    "TRADE_TRANSACTION_LINE_KEYS",
    "iter_trade_lifecycle_line_values",
    "normalize_trade_lifecycle_state",
]
