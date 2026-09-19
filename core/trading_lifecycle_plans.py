"""Canonical conversion of frozen intent evidence into lifecycle input.

AI: A registered order date is not proof that its geometry existed on that day.
Replay and acquisition use the dated immutable seed; a refreshed snapshot cannot
move its Stop/Target, indicators or management state into the past.
"""
from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

from core.trade_lifecycle import build_prefill_lifecycle_from_frame
from core.trading_identity import normalize_trading_date


PREFILL_ORIGIN_CONTRACT_VERSION = 5


def frozen_lineage_origin_seed(lineage, fallback=None):
    """Use first-signal evidence for management, never a daily derived snapshot."""
    origin = dict((lineage.get("signal_origin") or {}).get("origin_candidate") or {})
    return deepcopy(dict(origin.get("execution_plan_seed") or lineage.get("prefill_origin_seed") or lineage.get("execution_plan_seed") or fallback or {}))


def frozen_lineage_origin_date(lineage, fallback=None, *, information_date=None):
    """Resolve when the selected immutable geometry was known, not when registered.

    AI: Older persisted seeds already carry trade_date. Prefer that evidence to
    a later registration date, but never substitute the requested historical
    order/signal date for a missing geometry timestamp.
    """
    seed = frozen_lineage_origin_seed(lineage, fallback)
    origin_record = dict(lineage.get("signal_origin") or {})
    origin = dict(origin_record.get("origin_candidate") or {})
    recorded = seed.get("trade_date")
    if not recorded:
        if origin.get("execution_plan_seed"):
            recorded = origin.get("trade_date") or origin_record.get("observed_date")
        else:
            recorded = lineage.get("candidate_trade_date") or information_date
    as_of = normalize_trading_date(recorded, field_name="frozen_plan.trade_date", allow_none=False)
    if seed.get("management_information_date"):
        as_of = max(as_of, normalize_trading_date(
            seed["management_information_date"], field_name="frozen_plan.management_information_date", allow_none=False,
        ))
    return as_of


def build_pending_order_intent(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project accepted order decisions, never infer them from management values.

    AI: A backdated first registration is an explicit user fact. Subsequent
    changed decisions become available on their own information date; a daily
    management refresh with the same accepted values is not a new decision.
    """
    if not entry.get("planned_trade_date"):
        return None
    snapshots = list(entry.get("order_intent_history") or ()) + [entry]
    decisions = []
    previous_values = None
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping) or not snapshot.get("planned_trade_date"):
            continue
        order_date = normalize_trading_date(snapshot["planned_trade_date"], allow_none=False)
        values = (order_date, snapshot.get("limit_price"), snapshot.get("planned_qty"), snapshot.get("reserved_cost"))
        if values == previous_values:
            continue
        info = normalize_trading_date(snapshot.get("information_date"), allow_none=False)
        decision_date = normalize_trading_date(snapshot.get("order_decision_date") or info, allow_none=False)
        effective = order_date if not decisions else max(order_date, decision_date)
        decisions.append({
            "effective_date": effective, "planned_trade_date": order_date,
            "information_date": info, "limit_price": snapshot.get("limit_price"),
            "planned_qty": snapshot.get("planned_qty"), "reserved_capital": snapshot.get("reserved_cost"),
        })
        previous_values = values
    if not decisions:
        return None
    return {**decisions[0], "pending_entry_id": entry.get("pending_entry_id"), "decisions": decisions}


def build_pending_prefill_plan(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One dated origin for synchronization, charts and confirmed acquisition."""
    lineage = dict(entry.get("management_lineage") or {})
    seed = frozen_lineage_origin_seed(lineage, entry.get("execution_plan_seed"))
    as_of = frozen_lineage_origin_date(
        lineage, entry.get("execution_plan_seed"), information_date=entry.get("information_date"),
    )
    manual = str(entry.get("origin") or "") == "manual_selected"
    order_intent = build_pending_order_intent(entry)
    requested_origin = (
        (order_intent or {}).get("planned_trade_date") or entry.get("planned_trade_date") or entry.get("information_date")
        if manual else entry.get("signal_date") or lineage.get("signal_date") or entry.get("information_date")
    )
    requested_origin = normalize_trading_date(requested_origin, field_name="prefill.origin_date", allow_none=False)
    # AI: Keep the user's historical order intact; only the management origin is
    # bounded by available evidence. No recomputation with today's Params on a
    # made-up historical signal and no blanket suppression of genuine exits.
    signal_date = max(requested_origin, as_of)
    shadow = deepcopy(seed.get("shadow_position_state"))
    plan = {
        # AI: User-confirmed order facts are not dated by the management seed.
        # This separate axis restores the historical reservation/limit without
        # projecting a later Stop/Target or pretending that a Shadow existed.
        "order_intent": order_intent,
        "source": "trading_pending_frozen_lineage",
        "signal_date": signal_date,
        "origin_signal_date": requested_origin,
        "plan_as_of_date": as_of,
        "information_date": entry.get("information_date"),
        "entry_type": seed.get("entry_type") or "normal",
        "limit_price": seed.get("limit_price", entry.get("limit_price")),
        "stop_price": seed.get("init_sl", entry.get("init_sl")),
        "init_trail": seed.get("init_trail", entry.get("init_trail")),
        "tp_price": seed.get("target_price", entry.get("target_price")),
        "entry_atr": seed.get("entry_atr", entry.get("entry_atr")),
        "ticker": entry.get("ticker"),
        "security_profile": deepcopy(seed.get("security_profile")),
        "planned_qty": seed.get("qty", entry.get("planned_qty")),
        "reserved_capital": seed.get("reserved_cost", entry.get("reserved_cost")),
        "shadow_position_state": shadow,
        "shadow_as_of_date": as_of if isinstance(shadow, Mapping) else None,
    }
    # AI: Persisted policy is the input authority, not the timestamp on a later
    # derived price snapshot. Historical materialization is performed only by
    # the common frame-bound producer; adapters may not move prices backwards.
    frozen = lineage.get("frozen_params")
    if requested_origin < as_of and isinstance(frozen, Mapping):
        from core.file_integrity import canonical_json_sha256
        from core.params_io import build_params_from_mapping, params_to_json_dict
        actual_sha = canonical_json_sha256(dict(frozen))
        expected_sha = lineage.get("frozen_params_sha256")
        if expected_sha and str(expected_sha) != actual_sha:
            raise ValueError("Frozen pre-fill Params hash mismatch")
        intent = plan.get("order_intent") or {}
        if not manual or intent:
            plan["origin_reconstruction"] = {
                "origin_date": requested_origin,
                "kind": "manual_order" if manual else "strategy_signal",
                "accepted_limit_price": intent.get("limit_price") if manual else None,
                "frozen_params_sha256": canonical_json_sha256(params_to_json_dict(build_params_from_mapping(dict(frozen)))),
            }
    return plan


def resolve_confirmed_entry_plan_from_frame(*, entry, frame, params, fill_date):
    """Resolve Shadow immediately BEFORE a confirmed fill, without future data.

    The accepted order quantity/limit remains an input, not a resizable proposal.
    This only supplies the management state inherited at acquisition.
    """
    fill = pd.Timestamp(fill_date).normalize()
    from core.prefill_origin import materialize_prefill_origin
    before = frame.loc[frame.index < fill]
    plan = materialize_prefill_origin(frame=before, plan=build_pending_prefill_plan(entry), params=params)
    plan["end_before_date"] = fill.strftime("%Y-%m-%d")
    frozen = dict(entry.get("management_lineage") or {})
    seed = deepcopy(dict(frozen.get("execution_plan_seed") or entry.get("execution_plan_seed") or {}))
    # AI: A later derived snapshot is never an acquisition origin.
    seed.pop("shadow_position_state", None)
    origin_seed = frozen_lineage_origin_seed(frozen, seed)
    for key in ("init_sl", "init_trail", "target_price", "entry_atr", "target_reference_price"):
        if key in origin_seed:
            seed[key] = deepcopy(origin_seed[key])
    if plan.get("origin_resolution"):
        for key, field in (("init_sl", "stop_price"), ("init_trail", "init_trail"),
                           ("target_price", "tp_price"), ("entry_atr", "entry_atr")):
            seed[key] = deepcopy(plan[field])
        seed.pop("target_reference_price", None)
    signal_date = pd.Timestamp(plan["signal_date"])
    seed["management_information_date"] = signal_date.strftime("%Y-%m-%d")
    if fill <= signal_date:
        raise ValueError(
            f"PIT: confirmed fill {fill:%Y-%m-%d} must be after frozen management origin "
            f"{signal_date:%Y-%m-%d}; historical order date {plan['origin_signal_date']} "
            "does not prove earlier Stop/Target availability"
        )
    origin_shadow = dict(plan.get("shadow_position_state") or {})
    from core.extended_signals import shadow_has_reached_exit_barrier
    if shadow_has_reached_exit_barrier(origin_shadow):
        raise ValueError("Frozen pre-fill lifecycle terminated before the reported strategy fill")
    if before.empty or before.index.max() <= signal_date:
        if isinstance(plan.get("shadow_position_state"), Mapping):
            seed["shadow_position_state"] = deepcopy(plan["shadow_position_state"])
        return seed
    if not any(before.index == signal_date):
        raise ValueError("Cannot reconstruct pre-fill lineage without its original information bar")
    timeline = build_prefill_lifecycle_from_frame(frame=before, plan=plan, params=params)
    if not timeline:
        raise ValueError("Pre-fill replay returned no state despite completed pre-fill sessions")
    row = timeline[max(timeline)]
    shadow = dict(row.get("shadow_position_state") or {})
    if row.get("prefill_terminated") or shadow_has_reached_exit_barrier(shadow):
        raise ValueError("Frozen pre-fill lifecycle terminated before the reported strategy fill")
    if shadow:
        seed["shadow_position_state"] = deepcopy(shadow)
        seed["init_sl"] = shadow["sl"]
        seed["init_trail"] = shadow["trailing_stop"]
        seed["target_price"] = shadow["tp_half"]
    seed["management_information_date"] = before.index.max().strftime("%Y-%m-%d")
    return seed
