"""Compatibility facade for the canonical lifecycle core.

Runtime lifecycle ownership lives in :mod:`core.trade_lifecycle`.  This module is
kept as a stable import seam for older callers; it must not infer lifecycle state
from rendered chart geometry.
"""
from core.trade_lifecycle import (  # noqa: F401
    TRADE_LIFECYCLE_POSITION,
    TRADE_LIFECYCLE_SHADOW,
    TRADE_LIFECYCLE_SIGNAL,
    TRADE_LIFECYCLE_STATES,
    TRADE_TRANSACTION_LINE_KEYS,
    build_prefill_lifecycle_timeline,
    build_trade_lifecycle_row,
    iter_trade_lifecycle_line_values,
    lifecycle_rows_to_index,
    merge_lifecycle_row,
    normalize_trade_lifecycle_state,
    record_lifecycle_row,
)

__all__ = [
    "TRADE_LIFECYCLE_POSITION",
    "TRADE_LIFECYCLE_SHADOW",
    "TRADE_LIFECYCLE_SIGNAL",
    "TRADE_LIFECYCLE_STATES",
    "TRADE_TRANSACTION_LINE_KEYS",
    "build_prefill_lifecycle_timeline",
    "build_trade_lifecycle_row",
    "iter_trade_lifecycle_line_values",
    "lifecycle_rows_to_index",
    "merge_lifecycle_row",
    "normalize_trade_lifecycle_state",
    "record_lifecycle_row",
]
