"""Canonical trade lifecycle semantics shared by Research and Trading.

The lifecycle is strategy/execution state, not a rendering artifact.  Renderers
must consume these rows; they must never infer SIGNAL/SHADOW/POSITION from chart
lines or markers.

Research and Trading share the same pre-fill shadow engine.  The only domain
specific seam is fill truth: Research may replace SHADOW with its simulated fill,
while Trading replaces it only with effective account/broker truth.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math
from typing import Any, Mapping, Sequence

import numpy as np

from core.entry_plans import build_counterfactual_shadow_position_from_plan
from core.extended_signals import clone_shadow_position, shadow_has_reached_exit_barrier
from core.position_step import SETTLEMENT_BASIS_LEDGER_NET, execute_bar_step


TRADE_LIFECYCLE_PENDING = "PENDING"
TRADE_LIFECYCLE_SIGNAL = "SIGNAL"
TRADE_LIFECYCLE_SHADOW = "SHADOW"
TRADE_LIFECYCLE_POSITION = "POSITION"
TRADE_LIFECYCLE_STATES = frozenset(
    {TRADE_LIFECYCLE_SIGNAL, TRADE_LIFECYCLE_PENDING, TRADE_LIFECYCLE_SHADOW, TRADE_LIFECYCLE_POSITION}
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


_DISPLAY_STATE = {
    TRADE_LIFECYCLE_PENDING: "掛單",
    TRADE_LIFECYCLE_SIGNAL: "買訊",
    TRADE_LIFECYCLE_SHADOW: "SHADOW",
    TRADE_LIFECYCLE_POSITION: "持股",
}


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
    pending_limit_price: Any = None,
):
    """Map one lifecycle row to the canonical transaction-line family."""
    state = normalize_trade_lifecycle_state(lifecycle_state)
    if state in {TRADE_LIFECYCLE_SIGNAL, TRADE_LIFECYCLE_PENDING}:
        # AI: A confirmed order can predate management evidence. Only its
        # accepted limit may be drawn; absence of Shadow cannot erase the order.
        return () if pending_limit_price is None else (("shadow_limit_line", pending_limit_price),)
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


def _optional_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _date_text(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            # Fall through to the stable string-prefix parser below.
            text = str(value).strip()
        else:  # pragma: no cover - return above documents the successful branch
            text = ""
    else:
        text = str(value).strip()
    if not text:
        return None
    prefix = text[:10]
    try:
        datetime.strptime(prefix, "%Y-%m-%d")
    except ValueError:
        return None
    return prefix


def build_trade_lifecycle_row(
    lifecycle_state: object,
    *,
    source: str,
    signal_date: object = None,
    information_date: object = None,
    entry_type: object = None,
    limit_price: object = None,
    entry_price: object = None,
    stop_price: object = None,
    tp_price: object = None,
    reserved_capital: object = None,
    buy_capital: object = None,
    planned_qty: object = None,
    buy_qty: object = None,
    position_qty: object = None,
    remaining_order_qty: object = None,
    **extra: Any,
) -> dict[str, Any]:
    state = normalize_trade_lifecycle_state(lifecycle_state)
    row = {
        "state": state,
        "display_state": _DISPLAY_STATE[state],
        "source": str(source or "canonical_lifecycle"),
        "signal_date": _date_text(signal_date),
        "information_date": _date_text(information_date),
        "entry_type": str(entry_type or "normal"),
        "limit_price": _finite_float(limit_price),
        "entry_price": _finite_float(entry_price),
        "stop_price": _finite_float(stop_price),
        "tp_price": _finite_float(tp_price),
        "reserved_capital": _finite_float(reserved_capital),
        "buy_capital": _finite_float(buy_capital),
        "planned_qty": _optional_int(planned_qty),
        "buy_qty": _optional_int(buy_qty),
        "position_qty": _optional_int(position_qty),
        "remaining_order_qty": _optional_int(remaining_order_qty),
    }
    row.update(extra)
    return row


def merge_lifecycle_row(
    current: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge same-day lifecycle evidence with semantic precedence.

    A day can be observed pre-market as SHADOW and later become POSITION after a
    fill.  POSITION therefore wins over SHADOW, and SHADOW wins over SIGNAL.
    Values from the winning row are authoritative; missing metadata may be filled
    from the lower-precedence row.
    """
    if not isinstance(current, Mapping):
        return deepcopy(dict(incoming))
    precedence = {
        TRADE_LIFECYCLE_SIGNAL: 1,
        TRADE_LIFECYCLE_PENDING: 2,
        TRADE_LIFECYCLE_SHADOW: 3,
        TRADE_LIFECYCLE_POSITION: 4,
    }
    current_state = normalize_trade_lifecycle_state(current.get("state"))
    incoming_state = normalize_trade_lifecycle_state(incoming.get("state"))
    if precedence[incoming_state] >= precedence[current_state]:
        winner = deepcopy(dict(incoming))
        fallback = current
    else:
        winner = deepcopy(dict(current))
        fallback = incoming
    for key, value in dict(fallback).items():
        if winner.get(key) is None and value is not None:
            winner[key] = deepcopy(value)
    return winner


def record_lifecycle_row(
    store: dict[str, dict[str, Any]],
    current_date: object,
    row: Mapping[str, Any],
) -> None:
    date_text = _date_text(current_date)
    if date_text is None:
        raise ValueError(f"lifecycle current_date 無效: {current_date!r}")
    store[date_text] = merge_lifecycle_row(store.get(date_text), row)


def lifecycle_rows_to_index(
    date_labels: Sequence[object],
    rows_by_date: Mapping[str, Mapping[str, Any]] | None,
) -> dict[int, dict[str, Any]]:
    by_date = dict(rows_by_date or {})
    timeline: dict[int, dict[str, Any]] = {}
    for idx, raw_date in enumerate(date_labels):
        date_text = _date_text(raw_date)
        if date_text is None:
            continue
        row = by_date.get(date_text)
        if isinstance(row, Mapping):
            timeline[idx] = deepcopy(dict(row))
    return timeline


def _series(values: Sequence[Any] | None, total: int, *, default: float = np.nan) -> list[Any]:
    if values is None:
        return [default] * total
    result = list(values)
    if len(result) < total:
        result.extend([default] * (total - len(result)))
    return result[:total]


def _shadow_tp(position: Mapping[str, Any]) -> float | None:
    if bool(position.get("sold_half", False)) or position.get("pending_exit_action") == "TP_HALF":
        return None
    return _finite_float(position.get("tp_half"))


def _shadow_row_from_position(
    plan: Mapping[str, Any],
    position: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return build_trade_lifecycle_row(
        TRADE_LIFECYCLE_SHADOW,
        source=source,
        signal_date=plan.get("signal_date"),
        information_date=plan.get("information_date"),
        entry_type=plan.get("entry_type"),
        limit_price=plan.get("limit_price"),
        entry_price=position.get("entry_fill_price"),
        stop_price=position.get("sl"),
        tp_price=_shadow_tp(position),
        reserved_capital=plan.get("reserved_capital"),
        planned_qty=plan.get("planned_qty"),
        remaining_order_qty=plan.get("planned_qty"),
        shadow_position_state=clone_shadow_position(position),
        origin_signal_date=plan.get("origin_signal_date"),
        plan_as_of_date=plan.get("plan_as_of_date"),
        entry_atr=plan.get("entry_atr"),
        origin_resolution=plan.get("origin_resolution"),
    )


def _build_prefill_management_timeline(
    *,
    date_labels: Sequence[object],
    open_values: Sequence[Any],
    high_values: Sequence[Any],
    low_values: Sequence[Any],
    close_values: Sequence[Any],
    volume_values: Sequence[Any] | None,
    atr_values: Sequence[Any] | None,
    sell_signals: Sequence[Any] | None,
    plan: Mapping[str, Any],
    params: Any,
) -> dict[int, dict[str, Any]]:
    """Build the canonical pre-fill lifecycle for one frozen strategy plan.

    The information bar is always ``SIGNAL``.  From the next completed trading
    bar onward the plan is ``SHADOW`` until the caller's fill/cancel boundary.
    Shadow geometry is advanced with the same canonical position primitives used
    by Research's extended-shadow path; no rendered chart line or simulated fill
    is consulted.

    The row stored for a SHADOW day is the *completed-bar* shadow state for that
    day.  This matches Research chart semantics: D+1 first creates the
    counterfactual anchor using D+1 OHLC, and later bars advance that same shadow
    through ``execute_bar_step`` before the day's row is exposed.  Consequently
    the shadow stop may tighten but can never move backward after SHADOW starts.
    """
    if not isinstance(plan, Mapping):
        return {}
    labels = [_date_text(value) for value in date_labels]
    total = len(labels)
    if total <= 0:
        return {}

    signal_date = _date_text(plan.get("signal_date")) or _date_text(plan.get("information_date"))
    information_date = _date_text(plan.get("information_date")) or signal_date
    end_date = _date_text(plan.get("end_date"))
    end_before_date = _date_text(plan.get("end_before_date"))
    if signal_date is None:
        return {}
    # AI: Dated geometry is an input contract, not a renderer inference. Never
    # emit even a SIGNAL outside the plan's fill/cancel or availability boundary.
    plan_as_of = _date_text(plan.get("plan_as_of_date"))
    if plan_as_of is not None and signal_date < plan_as_of:
        raise ValueError("PIT: lifecycle origin precedes frozen plan availability")
    if ((end_before_date is not None and signal_date >= end_before_date)
            or (end_date is not None and signal_date > end_date)):
        return {}

    entry_type = str(plan.get("entry_type") or "normal")
    source = str(plan.get("source") or "prefill_plan")
    timeline: dict[int, dict[str, Any]] = {}

    # The signal/information bar owns no transaction line.  It still carries the
    # frozen plan values so the sidebar can show the same planned transaction
    # information as Research/Trading Center without pretending that a fill or
    # shadow position already exists on that bar.
    signal_idx = next((idx for idx, value in enumerate(labels) if value == signal_date), None)
    seeded_shadow = plan.get("shadow_position_state")
    shadow_as_of = _date_text(plan.get("shadow_as_of_date"))
    resume_shadow = isinstance(seeded_shadow, Mapping) and shadow_as_of is not None
    if resume_shadow and shadow_as_of != signal_date:
        raise ValueError("PIT: dated Shadow must resume at its own information bar")
    if resume_shadow and signal_idx is not None:
        timeline[signal_idx] = _shadow_row_from_position(plan, seeded_shadow, source=source)
        if shadow_has_reached_exit_barrier(seeded_shadow):
            timeline[signal_idx]["prefill_terminated"] = True
            timeline[signal_idx]["prefill_terminated_date"] = signal_date
            return timeline
    elif signal_idx is not None:
        timeline[signal_idx] = build_trade_lifecycle_row(
            TRADE_LIFECYCLE_SIGNAL,
            source=source,
            signal_date=signal_date,
            information_date=information_date,
            entry_type=entry_type,
            limit_price=plan.get("limit_price"),
            entry_price=None,
            stop_price=plan.get("stop_price"),
            tp_price=plan.get("tp_price"),
            reserved_capital=plan.get("reserved_capital"),
            planned_qty=plan.get("planned_qty"),
            remaining_order_qty=plan.get("planned_qty"),
            origin_signal_date=plan.get("origin_signal_date"),
            plan_as_of_date=plan.get("plan_as_of_date"),
            origin_resolution=plan.get("origin_resolution"),
        )

    # Params are needed only to advance SHADOW.  Keep the SIGNAL row even when
    # frozen Params are unavailable so missing runtime evidence never erases the
    # information-bar state.
    if params is None:
        return timeline

    opens = _series(open_values, total)
    highs = _series(high_values, total)
    lows = _series(low_values, total)
    closes = _series(close_values, total)
    volumes = _series(volume_values, total)
    atrs = _series(atr_values, total)
    sells = _series(sell_signals, total, default=False)

    start_idx = next(
        (
            idx
            for idx, date_text in enumerate(labels)
            if date_text is not None
            and date_text > signal_date
            and (end_before_date is None or date_text < end_before_date)
            and (end_date is None or date_text <= end_date)
        ),
        None,
    )
    if start_idx is None:
        return timeline

    planned_qty = _optional_int(plan.get("planned_qty"))
    seeded_shadow = plan.get("shadow_position_state")
    seeded_qty = _optional_int((seeded_shadow or {}).get("qty")) if isinstance(seeded_shadow, Mapping) else None
    # Historical lineage created before planned_qty was persisted still has
    # canonical L/S/TP/ATR evidence.  Quantity is irrelevant to stop/trailing
    # geometry except TP-half bookkeeping, so use a private geometry quantity
    # rather than dropping SHADOW entirely.  Never expose this fallback as the
    # user-facing planned quantity.
    geometry_qty = planned_qty or seeded_qty or 1000

    fixed_plan = {
        "limit_price": plan.get("limit_price"),
        "init_sl": plan.get("stop_price"),
        "init_trail": plan.get("init_trail", plan.get("stop_price")),
        "target_price": plan.get("tp_price"),
        "entry_atr": plan.get("entry_atr"),
        "qty": int(geometry_qty),
        "ticker": plan.get("ticker"),
        "security_profile": deepcopy(plan.get("security_profile")),
    }
    if _finite_float(fixed_plan.get("limit_price")) is None:
        return timeline

    # AI: A preserved mature Shadow is already complete through its as-of bar.
    # Resume it, rather than recreating an anchor or applying that bar twice.
    if resume_shadow and signal_idx is None:
        raise ValueError("Dated Shadow replay requires its original information bar")
    shadow_position = clone_shadow_position(seeded_shadow) if resume_shadow else None
    previous_idx = signal_idx if resume_shadow else None
    for idx in range(start_idx, total):
        date_text = labels[idx]
        if date_text is None:
            continue
        if end_before_date is not None and date_text >= end_before_date:
            break
        if end_date is not None and date_text > end_date:
            break

        if shadow_position is None:
            # Same counterfactual anchor primitive used by Research when a
            # normal signal becomes an extended/shadow trade after no real fill.
            shadow_position = build_counterfactual_shadow_position_from_plan(
                fixed_plan,
                t_open=opens[idx],
                t_high=highs[idx],
                t_low=lows[idx],
                params=params,
                ticker=plan.get("ticker"),
                security_profile=plan.get("security_profile"),
                trade_date=date_text,
            )
            if shadow_position is None or int(shadow_position.get("qty", 0) or 0) <= 0:
                break
        else:
            # Research advances the existing shadow using yesterday's ATR/sell
            # evidence plus today's completed OHLC.  Do that *before* exposing
            # today's row so Trading and Research show the same completed-bar
            # management state.
            prev_idx = int(previous_idx) if previous_idx is not None else idx - 1
            pre_update = clone_shadow_position(shadow_position)
            try:
                y_ind_sell = bool(sells[prev_idx])
            except (TypeError, ValueError):
                y_ind_sell = False
            updated, _freed_cash, _pnl, _events = execute_bar_step(
                clone_shadow_position(shadow_position),
                atrs[prev_idx],
                y_ind_sell,
                closes[prev_idx],
                opens[idx],
                highs[idx],
                lows[idx],
                closes[idx],
                volumes[idx],
                params,
                current_date=date_text,
                y_high=highs[prev_idx],
                settlement_basis=SETTLEMENT_BASIS_LEDGER_NET,
            )
            # Research writes the day's pre-entry preview before clearing a
            # shadow that hits an exit barrier.  Preserve that visible geometry
            # on the terminal day, then stop the lifecycle.
            if updated is None or int(updated.get("qty", 0) or 0) <= 0:
                timeline[idx] = _shadow_row_from_position(plan, pre_update, source=source)
                # AI: Preserve chart geometry while exposing the terminal event
                # to live lifecycle consumers. The pre-update position by itself
                # cannot tell a synchronizer that this shadow has already exited.
                timeline[idx]["prefill_terminated"] = True
                timeline[idx]["prefill_terminated_date"] = date_text
                break
            shadow_position = updated

        timeline[idx] = _shadow_row_from_position(plan, shadow_position, source=source)
        previous_idx = idx
        if shadow_has_reached_exit_barrier(shadow_position, day_low=lows[idx], day_high=highs[idx]):
            timeline[idx]["prefill_terminated"] = True
            timeline[idx]["prefill_terminated_date"] = date_text
            break

    return timeline


def overlay_confirmed_order_intent(
    timeline: Mapping[int, Mapping[str, Any]],
    *,
    date_labels: Sequence[object],
    plan: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Join order facts to management rows without inventing earlier geometry.

    AI: Order existence and strategy phase have independent time boundaries.
    This is the sole owner of their temporal join, shared by full chart replay
    and date-only consumers. The order cannot create a fill, a Shadow anchor or
    Stop/Target evidence. Fill/cancel boundaries close both axes.
    """
    result = {int(i): deepcopy(dict(row)) for i, row in timeline.items()}
    intent = plan.get("order_intent")
    if not isinstance(intent, Mapping):
        return result
    start = _date_text(intent.get("planned_trade_date"))
    if start is None:
        return result
    end = _date_text(plan.get("end_date"))
    end_before = _date_text(plan.get("end_before_date"))
    origin = _date_text(plan.get("signal_date"))
    terminal_date = None
    for idx, raw_date in enumerate(date_labels):
        date = _date_text(raw_date)
        if date is None:
            continue
        row = result.get(idx)
        if isinstance(row, Mapping) and (row.get("prefill_terminated") or
                (row.get("shadow_position_state") or {}).get("pending_exit_action") in {"STOP", "TP_HALF"}):
            terminal_date = date
        if date < start or (end is not None and date > end) or (end_before is not None and date >= end_before):
            continue
        versions = [decision for decision in intent.get("decisions") or ()
                    if _date_text(decision.get("effective_date")) is not None
                    and _date_text(decision.get("effective_date")) <= date]
        decision = versions[-1] if versions else intent
        if row is None:
            row = build_trade_lifecycle_row(
                TRADE_LIFECYCLE_PENDING, source=str(plan.get("source") or "confirmed_order_intent"),
                signal_date=plan.get("origin_signal_date") or plan.get("signal_date"),
                information_date=decision.get("information_date"), entry_type=plan.get("entry_type"),
                limit_price=decision.get("limit_price"), planned_qty=decision.get("planned_qty"),
                remaining_order_qty=decision.get("planned_qty"), reserved_capital=decision.get("reserved_capital"),
                origin_signal_date=plan.get("origin_signal_date"), plan_as_of_date=plan.get("plan_as_of_date"),
                management_evidence_status=("BEFORE_ORIGIN" if origin is not None and date < origin else
                                            "TERMINATED" if terminal_date is not None else "UNAVAILABLE"),
            )
            result[idx] = row
        row["confirmed_pending_order"] = True
        row["order_state"] = "PENDING"
        row["planned_trade_date"] = start
        row["pending_entry_id"] = intent.get("pending_entry_id")
        row["order_limit_price"] = _finite_float(decision.get("limit_price"))
        row["order_qty"] = _optional_int(decision.get("planned_qty"))
        row["order_reserved_capital"] = _finite_float(decision.get("reserved_capital"))
        # AI: The sidebar and limit line expose the same accepted order. Keep
        # management-plan reference values separate; a later quantity/price
        # edit must not overwrite earlier order facts or change Shadow state.
        if row.get("state") != TRADE_LIFECYCLE_PENDING:
            for name in ("limit_price", "planned_qty", "reserved_capital"):
                row["management_" + name] = row.get(name)
        row["limit_price"] = row["order_limit_price"]
        row["planned_qty"] = row["order_qty"]
        row["remaining_order_qty"] = row["order_qty"]
        row["reserved_capital"] = row["order_reserved_capital"]
        # The order day has no active transaction lines. From the next completed
        # bar a factual buy limit survives even when management starts later.
        row["pending_limit_line"] = row["order_limit_price"] if date > start else None
        if row.get("state") == TRADE_LIFECYCLE_SIGNAL:
            row["display_state"] = _DISPLAY_STATE[TRADE_LIFECYCLE_PENDING]
        if terminal_date is not None and row.get("state") == TRADE_LIFECYCLE_PENDING:
            row["prefill_terminated"] = True
            row["prefill_terminated_date"] = terminal_date
    return dict(sorted(result.items()))


def build_prefill_lifecycle_timeline(
    *, date_labels: Sequence[object], open_values: Sequence[Any], high_values: Sequence[Any],
    low_values: Sequence[Any], close_values: Sequence[Any], volume_values: Sequence[Any] | None,
    atr_values: Sequence[Any] | None, sell_signals: Sequence[Any] | None,
    plan: Mapping[str, Any], params: Any,
) -> dict[int, dict[str, Any]]:
    """Canonical pre-fill projection: strategy management plus confirmed intent."""
    if not isinstance(plan, Mapping):
        return {}
    management = _build_prefill_management_timeline(
        date_labels=date_labels, open_values=open_values, high_values=high_values, low_values=low_values,
        close_values=close_values, volume_values=volume_values, atr_values=atr_values,
        sell_signals=sell_signals, plan=plan, params=params,
    )
    return overlay_confirmed_order_intent(management, date_labels=date_labels, plan=plan)


def build_prefill_lifecycle_timeline_from_plans(
    *,
    date_labels: Sequence[object],
    open_values: Sequence[Any],
    high_values: Sequence[Any],
    low_values: Sequence[Any],
    close_values: Sequence[Any],
    volume_values: Sequence[Any] | None,
    atr_values: Sequence[Any] | None,
    sell_signals: Sequence[Any] | None,
    plans: Sequence[Mapping[str, Any]],
    params: Any,
) -> dict[int, dict[str, Any]]:
    """Build one non-overlapping pre-fill timeline from historical signal plans.

    Each signal owns its information bar and all later SHADOW bars until its
    canonical invalidation.  If a newer eligible signal appears while an older
    shadow would otherwise still be alive, the newer signal supersedes the older
    display lifecycle from its information bar onward.  This mirrors the strategy
    candidate lifecycle while remaining independent from any Research or Trading
    fill source.
    """
    merged: dict[int, dict[str, Any]] = {}
    owner_key: dict[int, tuple[str, int]] = {}
    for sequence, raw_plan in enumerate(list(plans or [])):
        if not isinstance(raw_plan, Mapping):
            continue
        plan = dict(raw_plan)
        timeline = build_prefill_lifecycle_timeline(
            date_labels=date_labels,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            volume_values=volume_values,
            atr_values=atr_values,
            sell_signals=sell_signals,
            plan=plan,
            params=params,
        )
        signal_key = str(_date_text(plan.get("signal_date")) or _date_text(plan.get("information_date")) or "")
        key = (signal_key, int(sequence))
        for idx, row in timeline.items():
            if idx not in owner_key or key >= owner_key[idx]:
                merged[int(idx)] = deepcopy(dict(row))
                owner_key[int(idx)] = key
    return merged


__all__ = [
    "TRADE_LIFECYCLE_PENDING",
    "TRADE_LIFECYCLE_POSITION",
    "TRADE_LIFECYCLE_SHADOW",
    "TRADE_LIFECYCLE_SIGNAL",
    "TRADE_LIFECYCLE_STATES",
    "TRADE_TRANSACTION_LINE_KEYS",
    "build_prefill_lifecycle_timeline",
    "build_prefill_lifecycle_timeline_from_plans",
    "build_trade_lifecycle_row",
    "iter_trade_lifecycle_line_values",
    "lifecycle_rows_to_index",
    "merge_lifecycle_row",
    "normalize_trade_lifecycle_state",
    "record_lifecycle_row",
    "overlay_confirmed_order_intent",
]


def build_prefill_lifecycle_from_frame(*, frame, plan, params, indicator_cache=None):
    """Bind every plan to indicators computed with that plan's frozen Params.

    AI: A renderer cannot attach current-parameter ATR/sell arrays to an older
    plan. The optional operation-local cache is keyed by Params and frame content,
    never only ticker/date; it is not another persistent artifact identity.
    """
    import pandas as pd
    from core.file_integrity import canonical_json_sha256
    from core.params_io import params_to_json_dict
    from core.signal_utils import generate_signals, unpack_precomputed_signals
    import hashlib

    if frame.empty:
        return {}
    from core.prefill_origin import materialize_prefill_origin
    plan = materialize_prefill_origin(frame=frame, plan=plan, params=params)
    checkpoint = plan.pop("management_checkpoint", None)
    if checkpoint is not None:
        # AI: A dated checkpoint is a cache, not permission to splice another
        # origin into a reconstructed prefix. Validate the management handoff
        # against the same canonical replay before resuming exact saved state.
        from core.position_management import POSITION_MANAGEMENT_FIELDS
        from core.serialization_utils import json_native_value
        rebuilt = build_prefill_lifecycle_from_frame(
            frame=frame, plan=plan, params=params, indicator_cache=indicator_cache,
        )
        checkpoint_date = str(checkpoint["signal_date"])
        labels = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in frame.index]
        in_scope = (checkpoint_date in labels
                    and checkpoint_date <= str(plan.get("end_date") or "9999-12-31")
                    and checkpoint_date < str(plan.get("end_before_date") or "9999-12-31"))
        if not in_scope:
            return rebuilt
        checkpoint_idx = labels.index(checkpoint_date)
        preceding = [row for idx, row in rebuilt.items() if idx < checkpoint_idx]
        if any(row.get("prefill_terminated") for row in preceding) and not shadow_has_reached_exit_barrier(checkpoint.get("shadow_position_state")):
            raise ValueError("Frozen Shadow checkpoint conflicts with an earlier canonical exit")
        resolved = dict((rebuilt.get(checkpoint_idx) or {}).get("shadow_position_state") or {})
        saved = dict(checkpoint.get("shadow_position_state") or {})
        def management_identity(position):
            return json_native_value({key: position[key] for key in POSITION_MANAGEMENT_FIELDS if key in position})
        compatible = bool(resolved) and management_identity(resolved) == management_identity(saved)
        if not compatible:
            for idx, row in rebuilt.items():
                if idx >= checkpoint_idx:
                    row["management_checkpoint_resolution"] = "recomputed_incompatible_derived_checkpoint"
            return rebuilt
        earlier = {idx: row for idx, row in rebuilt.items() if idx < checkpoint_idx}
        later = build_prefill_lifecycle_from_frame(frame=frame, plan=checkpoint, params=params, indicator_cache=indicator_cache)
        earlier.update(later)
        return overlay_confirmed_order_intent(earlier, date_labels=list(frame.index), plan=plan)
    if params is None:
        return build_prefill_lifecycle_timeline(
            date_labels=list(frame.index), open_values=frame["Open"], high_values=frame["High"],
            low_values=frame["Low"], close_values=frame["Close"], volume_values=frame["Volume"],
            atr_values=None, sell_signals=None, plan=plan, params=None,
        )
    frame_digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    key = (str(plan.get("ticker") or ""), canonical_json_sha256(params_to_json_dict(params)), frame_digest)
    cached = None if indicator_cache is None else indicator_cache.get(key)
    if cached is None:
        cached = unpack_precomputed_signals(generate_signals(frame, params, ticker=plan.get("ticker")))
        if indicator_cache is not None:
            indicator_cache[key] = cached
    atr, _buy, sell, _limits = cached
    return build_prefill_lifecycle_timeline(
        date_labels=list(frame.index), open_values=frame["Open"], high_values=frame["High"],
        low_values=frame["Low"], close_values=frame["Close"], volume_values=frame["Volume"],
        atr_values=atr, sell_signals=sell, plan=plan, params=params,
    )
