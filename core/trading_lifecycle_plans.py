"""Canonical conversion of frozen intent evidence into lifecycle input.

AI: Inputs preserve the original signal and the user's confirmed choices. The
frame gateway binds indicators to frozen Params, and acquisition excludes its
own and all future bars when inheriting pre-fill management.
"""
from copy import deepcopy
from typing import Any, Mapping
import pandas as pd
from core.trade_lifecycle import build_prefill_lifecycle_from_frame

def frozen_lineage_origin_seed(lineage, fallback=None):
    """Use first-signal evidence for management, never the accepted quantity."""
    origin = dict((lineage.get("signal_origin") or {}).get("origin_candidate") or {})
    return deepcopy(dict(origin.get("execution_plan_seed") or lineage.get("execution_plan_seed") or fallback or {}))


def build_pending_prefill_plan(entry: Mapping[str, Any]) -> dict[str, Any]:
    # AI: Replay always starts from the immutable confirmed plan, not yesterday's
    # derived stop/target. Otherwise repeated daily reads rewrite the past and
    # differ from a single replay to the same finalized date.
    lineage = dict(entry.get("management_lineage") or {})
    seed = frozen_lineage_origin_seed(lineage, entry.get("execution_plan_seed"))
    manual = str(entry.get("origin") or "") == "manual_selected"
    signal_date = (
        entry.get("planned_trade_date") or entry.get("information_date")
        if manual else entry.get("signal_date") or entry.get("information_date")
    )
    return {
        "source": "trading_pending_frozen_lineage",
        "signal_date": signal_date,
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
    }


def resolve_confirmed_entry_plan_from_frame(*, entry, frame, params, fill_date):
    """Resolve Shadow immediately BEFORE a confirmed fill, without future data.

    The accepted order quantity/limit remains an input, not a resizable proposal.
    This only supplies the management state inherited at acquisition.
    """
    fill = pd.Timestamp(fill_date).normalize()
    plan = build_pending_prefill_plan(entry)
    plan["end_before_date"] = fill.strftime("%Y-%m-%d")
    frozen = dict(entry.get("management_lineage") or {})
    seed = deepcopy(dict(frozen.get("execution_plan_seed") or entry.get("execution_plan_seed") or {}))
    # AI: A later derived snapshot is never an acquisition origin.
    seed.pop("shadow_position_state", None)
    origin_seed = frozen_lineage_origin_seed(frozen, seed)
    for key in ("init_sl", "init_trail", "target_price", "entry_atr", "target_reference_price"):
        if key in origin_seed:
            seed[key] = deepcopy(origin_seed[key])
    before = frame.loc[frame.index < fill]
    signal_date = pd.Timestamp(plan["signal_date"])
    seed["management_information_date"] = signal_date.strftime("%Y-%m-%d")
    if fill <= signal_date:
        raise ValueError("Confirmed fill must be after its frozen signal/order date")
    if before.empty or before.index.max() <= signal_date:
        return seed
    if not any(before.index == signal_date):
        raise ValueError("Cannot reconstruct pre-fill lineage without its original information bar")
    timeline = build_prefill_lifecycle_from_frame(frame=before, plan=plan, params=params)
    if not timeline:
        raise ValueError("Pre-fill replay returned no state despite completed pre-fill sessions")
    row = timeline[max(timeline)]
    shadow = dict(row.get("shadow_position_state") or {})
    if row.get("prefill_terminated") or shadow.get("pending_exit_action") in {"STOP", "TP_HALF"}:
        raise ValueError("Frozen pre-fill lifecycle terminated before the reported strategy fill")
    if shadow:
        seed["shadow_position_state"] = deepcopy(shadow)
        seed["init_sl"] = shadow["sl"]
        seed["init_trail"] = shadow["trailing_stop"]
        seed["target_price"] = shadow["tp_half"]
    seed["management_information_date"] = before.index.max().strftime("%Y-%m-%d")
    return seed
