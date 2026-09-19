"""Automatic Trading lifecycle synchronization to the latest finalized state.

Pending orders and managed positions are two phases of the same frozen strategy
lineage.  This service advances both phases before Workbench read models are
rendered.  Historical broker/user facts (ticker, order date, fill date, fill
price/qty) are never rewritten; only derived planning/management state moves
forward.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.data_utils import get_required_min_rows
from core.entry_plans import validate_accepted_entry_reservation
from core.file_integrity import canonical_json_sha256
from core.exact_accounting import price_to_milli
from core.params_io import build_params_from_mapping
from core.serialization_utils import json_native_value
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trade_lifecycle import build_prefill_lifecycle_from_frame
from core.trading_lifecycle_plans import PREFILL_ORIGIN_CONTRACT_VERSION, build_pending_prefill_plan
from services.trading.lifecycle_context import resolve_trading_lifecycle_context
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from services.trading.account_state import load_trading_account_state
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.pending_entry_service import recover_trading_pending_entry_transaction
from services.trading.market_data_consumer import (
    load_trading_v2_consumer_state,
    load_trading_v2_sanitized_ohlcv_frame,
    open_trading_v2_consumer_view,
)
from services.trading.pending_entry_state import (
    PENDING_ENTRY_STATUS_ACTIVE,
    load_trading_pending_entry_state,
    project_trading_pending_intent_entries,
    update_trading_pending_entry,
)
from services.trading.position_rollforward import (
    build_trading_position_rollforward_snapshot,
    run_trading_position_rollforward,
)

from services.trading.lifecycle_sync_status import (
    SYNC_STATUS_LATEST, SYNC_STATUS_PENDING, SYNC_STATUS_FAILED,
)

TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION = 1
# AI: Invalidate old derived refreshes without changing any strategy identity.
PENDING_MANAGEMENT_SYNC_CONTRACT_VERSION = 1


def _available_cash_milli_for_pending(
    account: Mapping[str, Any] | None,
    active_entries: list[Mapping[str, Any]],
    current: Mapping[str, Any],
) -> int:
    if not isinstance(account, Mapping) or account.get("cash_milli") is None:
        raise RuntimeError("Trading cash 尚未設定，不能同步掛單規劃")
    current_id = str(current.get("pending_entry_id") or "")
    other_reserved = sum(
        int(row.get("reserved_cost_milli") or 0)
        for row in active_entries
        if str(row.get("pending_entry_id") or "") != current_id
    )
    available_milli = max(0, int(account.get("cash_milli") or 0) - other_reserved)
    return available_milli



def _latest_shadow_state(
    project_root: Path,
    *,
    entry: Mapping[str, Any],
    latest_finalized_date: str,
    params,
    lifecycle_context=None,
) -> tuple[dict[str, Any] | None, str | None]:
    ticker = normalize_trading_ticker(entry.get("ticker"))
    view = lifecycle_context.view if lifecycle_context is not None else open_trading_v2_consumer_view(project_root)
    frame = load_trading_v2_sanitized_ohlcv_frame(
        view,
        ticker=ticker,
        through_date=latest_finalized_date,
        min_rows=get_required_min_rows(params),
    )
    if frame.empty:
        raise RuntimeError(f"{ticker} 沒有可用 Trading adjusted 日K")
    timeline = build_prefill_lifecycle_from_frame(
        frame=frame, plan=build_pending_prefill_plan(entry), params=params,
    )
    if not timeline:
        return None, None
    latest_index = max(timeline)
    row = dict(timeline[latest_index])
    row_date = pd.Timestamp(frame.index[int(latest_index)]).strftime("%Y-%m-%d")
    # AI: An outstanding order remains visible after strategy termination; the
    # error must still identify the actual terminal bar, not the latest display.
    return row, row.get("prefill_terminated_date") or row_date


def _sync_one_pending(
    project_root: Path,
    *,
    entry: Mapping[str, Any],
    active_entries: list[Mapping[str, Any]],
    account: Mapping[str, Any] | None,
    latest_finalized_date: str,
    lifecycle_context=None,
) -> dict[str, Any]:
    entry_id = str(entry.get("pending_entry_id") or "")
    ticker = normalize_trading_ticker(entry.get("ticker"))
    evaluated = normalize_trading_date(
        entry.get("evaluated_through_date") or entry.get("information_date"),
        field_name="pending_entry.evaluated_through_date",
        allow_none=False,
    )
    prefill_origin = build_pending_prefill_plan(entry)
    if prefill_origin["plan_as_of_date"] > latest_finalized_date:
        raise RuntimeError(f"{ticker} frozen plan availability is ahead of the pinned finalized boundary")
    lineage = dict(entry.get("management_lineage") or {})
    frozen_params = lineage.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise RuntimeError(f"{ticker} pending entry lacks frozen params")
    params = build_params_from_mapping(dict(frozen_params))
    accepted_plan = dict(entry.get("execution_plan_seed") or {})
    # AI: Accepted economics have duplicated display fields. Reject disagreement
    # rather than quietly choosing one version and rewriting broker/user intent.
    for field, seed_field in (("planned_qty", "qty"), ("reserved_cost_milli", "reserved_cost_milli")):
        if accepted_plan.get(seed_field) is not None and accepted_plan[seed_field] != entry.get(field):
            raise ValueError(f"{ticker} accepted pending {field} disagrees with its execution plan")
        accepted_plan[seed_field] = entry.get(field)
    if accepted_plan.get("limit_price") is not None and price_to_milli(accepted_plan["limit_price"]) != price_to_milli(entry.get("limit_price")):
        raise ValueError(f"{ticker} accepted pending limit disagrees with its execution plan")
    accepted_plan["limit_price"] = entry.get("limit_price")
    accepted_plan["reserved_cost"] = entry.get("reserved_cost")
    validate_accepted_entry_reservation(
        accepted_plan,
        available_cash_milli=_available_cash_milli_for_pending(account, active_entries, entry),
        params=params,
    )
    context_fingerprint = canonical_json_sha256({
        "pending_management_sync_contract_version": PENDING_MANAGEMENT_SYNC_CONTRACT_VERSION,
        "prefill_origin_contract_version": PREFILL_ORIGIN_CONTRACT_VERSION,
        "prefill_origin": prefill_origin,
        "market_context": lifecycle_context.fingerprint if lifecycle_context is not None else latest_finalized_date,
        "lineage": entry.get("management_lineage"),
        "planned_qty": entry.get("planned_qty"),
        "planned_trade_date": entry.get("planned_trade_date"),
    })
    if entry.get("evaluation_context_fingerprint") == context_fingerprint:
        return {"pending_entry_id": entry_id, "ticker": ticker, "status": SYNC_STATUS_LATEST, "changed": False}
    if evaluated > latest_finalized_date:
        raise RuntimeError(f"{ticker} pending evaluation is ahead of the pinned finalized boundary")

    shadow_row, shadow_date = _latest_shadow_state(
        project_root,
        entry=entry,
        latest_finalized_date=latest_finalized_date,
        params=params,
        lifecycle_context=lifecycle_context,
    )

    replacement = deepcopy(dict(entry))
    seed = deepcopy(dict(replacement.get("execution_plan_seed") or {}))
    if shadow_row is not None:
        shadow_state = dict(shadow_row.get("shadow_position_state") or {})
        if shadow_row.get("prefill_terminated") or shadow_state.get("pending_exit_action") in {"STOP", "TP_HALF"}:
            raise RuntimeError(
                f"{ticker} frozen lineage 已於 {shadow_date or '-'} 觸發 shadow exit；ACTIVE 掛單需確認後結案"
            )
        if shadow_row.get("stop_price") is not None:
            seed["init_sl"] = float(shadow_row["stop_price"])
        if shadow_row.get("tp_price") is not None:
            seed["target_price"] = float(shadow_row["tp_price"])
        if shadow_row.get("entry_atr") is not None:
            seed["entry_atr"] = float(shadow_row["entry_atr"])
        if shadow_state:
            seed["shadow_position_state"] = json_native_value(shadow_state)
            if shadow_state.get("trailing_stop") is not None:
                seed["init_trail"] = float(shadow_state["trailing_stop"])

    seed["trade_date"] = latest_finalized_date
    seed["planned_trade_date"] = entry.get("planned_trade_date")
    # AI: Only management state advances. The saved ceiling, chosen quantity,
    # order limit, reservation, capital basis and user overrides remain intact.
    # Explicit create/amend commands alone can accept a new quantity decision.
    replacement["execution_plan_seed"] = seed
    replacement["init_sl"] = float(seed["init_sl"])
    replacement["init_trail"] = float(seed["init_trail"])
    replacement["target_price"] = float(seed["target_price"])
    replacement["entry_atr"] = seed.get("entry_atr")
    replacement["evaluated_through_date"] = latest_finalized_date
    replacement["evaluation_context_fingerprint"] = context_fingerprint
    replacement["sync_error"] = None
    updated = update_trading_pending_entry(
        project_root,
        pending_entry_id=entry_id,
        replacement=replacement,
    )
    return {
        "pending_entry_id": entry_id,
        "ticker": ticker,
        "status": SYNC_STATUS_LATEST,
        "changed": True,
        "evaluated_through_date": updated.get("evaluated_through_date"),
    }


@serialized_trading_state_mutation
def run_trading_lifecycle_sync(project_root: str | Path) -> dict[str, Any]:
    """Idempotently advance pending + position lifecycle state before reads."""

    root = Path(project_root).resolve()
    # AI: Finish a durable account-commit/pending-close transaction before validating
    # any active reservation. Otherwise the same filled order can reserve cash twice.
    recover_trading_pending_entry_transaction(root)
    consumer_state = load_trading_v2_consumer_state(root, required=False)
    if consumer_state is None:
        return {
            "schema_version": TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION,
            "latest_finalized_date": None,
            "status": SYNC_STATUS_PENDING,
            "pending_results": [],
            "pending_errors": {},
            "position_result": {"status": "NO_DATA"},
            "position_errors": {},
            "position_status_by_ticker": {},
            "position_due_tickers": [],
        }
    lifecycle_context = resolve_trading_lifecycle_context(root, consumer_state=consumer_state)
    latest_finalized_date = lifecycle_context.finalized_date

    pending_state = load_trading_pending_entry_state(root, required=False)
    active_entries = [] if pending_state is None else [
        deepcopy(row)
        for row in project_trading_pending_intent_entries(pending_state).values()
        if str(row.get("status") or "") == PENDING_ENTRY_STATUS_ACTIVE
    ]
    active_entries.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("pending_entry_id") or "")))
    account = load_trading_account_state(root, required=False)
    pending_results: list[dict[str, Any]] = []
    pending_errors: dict[str, str] = {}
    for entry in active_entries:
        entry_id = str(entry.get("pending_entry_id") or "")
        ticker = str(entry.get("ticker") or "")
        try:
            result = _sync_one_pending(
                root,
                entry=entry,
                active_entries=active_entries,
                account=account,
                latest_finalized_date=latest_finalized_date,
                lifecycle_context=lifecycle_context,
            )
            pending_results.append(result)
        except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            pending_errors[entry_id] = f"{type(exc).__name__}: {exc}"
            pending_results.append({
                "pending_entry_id": entry_id,
                "ticker": ticker,
                "status": SYNC_STATUS_FAILED,
                "changed": False,
                "error": pending_errors[entry_id],
            })

    position_error = None
    try:
        position_result = run_trading_position_rollforward(root, lifecycle_context=lifecycle_context)
    except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        position_error = f"{type(exc).__name__}: {exc}"
        position_result = {"status": SYNC_STATUS_FAILED, "error": position_error}

    try:
        post_rollforward = build_trading_position_rollforward_snapshot(root, lifecycle_context=lifecycle_context)
    except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        post_rollforward = {"due_tickers": []}
        if position_error is None:
            position_error = f"{type(exc).__name__}: {exc}"

    due_tickers = {str(value) for value in list(post_rollforward.get("due_tickers") or [])}
    account_after = load_trading_account_state(root, required=False) or {}
    position_status_by_ticker: dict[str, str] = {}
    # AI: A returned partial failure is as real as a raised exception. Never
    # advertise a failed manual activation or protection refresh as up-to-date.
    reconcile = dict(position_result.get("manual_management_reconcile") or {})
    position_errors = {
        str(ticker): str(error)
        for ticker, error in dict(reconcile.get("errors") or {}).items()
    }
    position_errors.update({str(k): str(v) for k, v in dict(position_result.get("errors") or {}).items()})
    protection_error = position_result.get("protection_plan_refresh_error")
    if protection_error and position_error is None:
        position_error = str(protection_error)
    for ticker in sorted((account_after.get("positions") or {}).keys()):
        if position_error is not None or ticker in position_errors:
            position_status_by_ticker[ticker] = SYNC_STATUS_FAILED
            position_errors[ticker] = position_error or position_errors[ticker]
        elif ticker in due_tickers:
            position_status_by_ticker[ticker] = SYNC_STATUS_PENDING
        else:
            position_status_by_ticker[ticker] = SYNC_STATUS_LATEST

    overall = SYNC_STATUS_LATEST
    if pending_errors or position_error or position_errors:
        overall = SYNC_STATUS_FAILED
    elif due_tickers:
        overall = SYNC_STATUS_PENDING
    return {
        "schema_version": TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION,
        "latest_finalized_date": latest_finalized_date,
        "status": overall,
        "pending_results": pending_results,
        "pending_errors": pending_errors,
        "position_result": position_result,
        "position_errors": position_errors,
        "position_status_by_ticker": position_status_by_ticker,
        "position_due_tickers": sorted(due_tickers),
    }


__all__ = [
    "SYNC_STATUS_FAILED",
    "SYNC_STATUS_LATEST",
    "SYNC_STATUS_PENDING",
    "TRADING_LIFECYCLE_SYNC_SCHEMA_VERSION",
    "run_trading_lifecycle_sync",
]
