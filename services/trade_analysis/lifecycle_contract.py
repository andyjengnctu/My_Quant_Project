"""Shared trade lifecycle contract for Research and Trading inspection.

Research owns strategy semantics and may simulate fills. Trading owns broker/account
execution truth. Both domains share the same user-facing lifecycle vocabulary and
line geometry:

    SIGNAL -> SHADOW -> POSITION

The helpers in this module are deliberately domain-neutral. They convert a chart's
strategy evidence into a per-bar lifecycle context and map lifecycle states to the
same line families. Trading can therefore reuse Research's counterfactual strategy
path before a real fill while replacing POSITION with canonical account truth after
the fill, rather than maintaining a second strategy state machine.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence


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
    """Yield canonical chart-line mapping for one lifecycle state.

    SIGNAL owns no transaction line on its information bar. SHADOW uses only the
    shadow line family. POSITION uses only the actual line family. Research and
    Trading both consume this exact mapping.
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


def _finite_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _series_value(payload: Mapping[str, Any], key: str, idx: int) -> float | None:
    values = payload.get(key)
    if values is None:
        return None
    try:
        if idx < 0 or idx >= len(values):
            return None
        return _finite_float(values[idx])
    except TypeError:
        return None


def _marker_index(marker: Mapping[str, Any]) -> int | None:
    try:
        return int(marker.get("x"))
    except (TypeError, ValueError):
        return None


def _signal_index(item: Mapping[str, Any]) -> int | None:
    try:
        return int(item.get("x"))
    except (TypeError, ValueError):
        return None


def build_strategy_lifecycle_timeline_from_chart_payload(
    chart_payload: Mapping[str, Any] | None,
    *,
    buy_trace_names: Sequence[str],
) -> dict[int, dict[str, Any]]:
    """Derive Research strategy lifecycle context from canonical chart evidence.

    This is *strategy* truth, not broker truth. A Research simulated POSITION is
    intentionally preserved as a counterfactual strategy state. Trading may map
    that state to SHADOW until an effective account fill exists. That is the key
    seam which keeps Research/Trading stop/target/trailing geometry identical
    before the real fill without treating a Research fill as an actual trade.
    """
    payload = dict(chart_payload or {})
    labels = list(payload.get("date_labels") or payload.get("dates") or [])
    total = len(labels)
    if total <= 0:
        return {}

    signal_meta_by_index: dict[int, dict[str, Any]] = {}
    for raw in list(payload.get("signal_annotations") or []):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("signal_type") or "").strip().lower() != "buy":
            continue
        idx = _signal_index(raw)
        if idx is None or idx < 0 or idx >= total:
            continue
        signal_meta_by_index[idx] = deepcopy(dict(raw.get("meta") or {}))

    buy_markers_by_index: dict[int, list[dict[str, Any]]] = {}
    marker_groups = dict(payload.get("marker_groups") or {})
    for trace_name in buy_trace_names:
        for raw in list(marker_groups.get(str(trace_name)) or []):
            if not isinstance(raw, Mapping):
                continue
            idx = _marker_index(raw)
            if idx is None or idx < 0 or idx >= total:
                continue
            buy_markers_by_index.setdefault(idx, []).append(deepcopy(dict(raw)))

    timeline: dict[int, dict[str, Any]] = {}
    latest_signal_meta: dict[str, Any] = {}
    latest_buy_meta: dict[str, Any] = {}
    latest_buy_qty: int | None = None

    for idx in range(total):
        if idx in signal_meta_by_index:
            latest_signal_meta = deepcopy(signal_meta_by_index[idx])
        if idx in buy_markers_by_index:
            marker = buy_markers_by_index[idx][-1]
            latest_buy_meta = deepcopy(dict(marker.get("meta") or {}))
            try:
                latest_buy_qty = int(marker.get("qty") or 0) or None
            except (TypeError, ValueError):
                latest_buy_qty = None

        actual = {
            "stop_price": _series_value(payload, "stop_line", idx),
            "tp_price": _series_value(payload, "tp_line", idx),
            "limit_price": _series_value(payload, "limit_line", idx),
            "entry_price": _series_value(payload, "entry_line", idx),
        }
        shadow = {
            "stop_price": _series_value(payload, "shadow_stop_line", idx),
            "tp_price": _series_value(payload, "shadow_tp_line", idx),
            "limit_price": _series_value(payload, "shadow_limit_line", idx),
            "entry_price": _series_value(payload, "shadow_entry_line", idx),
        }
        has_shadow = any(value is not None for value in shadow.values())
        has_actual = any(value is not None for value in actual.values())
        has_actual_position_geometry = any(
            actual.get(key) is not None for key in ("entry_price", "stop_price", "tp_price")
        )
        has_fill_marker = idx in buy_markers_by_index
        has_signal = idx in signal_meta_by_index

        if has_shadow:
            state = TRADE_LIFECYCLE_SHADOW
            values = shadow
        elif has_actual_position_geometry or has_fill_marker:
            state = TRADE_LIFECYCLE_POSITION
            values = actual
        elif has_actual:
            # A pre-fill normal order can use the actual line family for its limit
            # preview in older Research payloads. Semantically it is still SHADOW.
            state = TRADE_LIFECYCLE_SHADOW
            values = actual
        elif has_signal:
            state = TRADE_LIFECYCLE_SIGNAL
            values = {"stop_price": None, "tp_price": None, "limit_price": None, "entry_price": None}
        else:
            continue

        state_row = {
            "state": state,
            "display_state": {
                TRADE_LIFECYCLE_SIGNAL: "買訊",
                TRADE_LIFECYCLE_SHADOW: "SHADOW",
                TRADE_LIFECYCLE_POSITION: "持股",
            }[state],
            "source": "research_strategy_lifecycle",
            **values,
            "reserved_capital": latest_signal_meta.get("reserved_capital"),
            "planned_qty": latest_signal_meta.get("qty"),
            "buy_capital": latest_buy_meta.get("buy_capital"),
            "buy_qty": latest_buy_qty,
        }
        timeline[idx] = state_row
    return timeline


__all__ = [
    "TRADE_LIFECYCLE_POSITION",
    "TRADE_LIFECYCLE_SHADOW",
    "TRADE_LIFECYCLE_SIGNAL",
    "TRADE_LIFECYCLE_STATES",
    "TRADE_TRANSACTION_LINE_KEYS",
    "build_strategy_lifecycle_timeline_from_chart_payload",
    "iter_trade_lifecycle_line_values",
    "normalize_trade_lifecycle_state",
]
