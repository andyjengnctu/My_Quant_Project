"""Live adapter for the canonical session replay; never infer broker fills."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import pandas as pd

from core.entry_plans import build_position_from_frozen_entry_plan
from core.exact_accounting import milli_to_price, price_to_milli
from core.file_integrity import canonical_json_sha256
from core.params_io import build_params_from_mapping
from core.position_management import POSITION_MANAGEMENT_FIELDS
from core.trading_position_projection import CONFIRMED_POSITION_ORIGIN_CONTRACT_VERSION, build_confirmed_position_origin, collect_confirmed_position_events, project_full_exit_obligation
from core.position_replay import POSITION_REPLAY_CONTRACT_VERSION, replay_confirmed_position_management
from core.trading_lifecycle_plans import PREFILL_ORIGIN_CONTRACT_VERSION
from core.runtime_domains import RUNTIME_DOMAIN_TRADING
from core.serialization_utils import json_native_value
from core.trading_account_state import (
    MANAGEMENT_STATUS_ACTIVE, MANAGEMENT_SELL_SIGNAL_INDICATOR, MANAGEMENT_SELL_SIGNAL_STOP,
    MANAGED_POSITION_SOURCES, POSITION_SOURCE_MANUAL_ADOPTED, POSITION_SOURCE_MANUAL_MANAGED,
    effective_trading_account_events,
)
from core.trading_stop_exit_progress import build_trading_stop_exit_progress
from services.trading.account_state import (
    activate_existing_manual_trading_position_management, load_trading_account_state,
    rollforward_trading_strategy_management,
)
from services.trading.account_trade_entry import build_manual_adopted_management_context
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.lifecycle_context import TradingLifecycleContext, resolve_trading_lifecycle_context
from services.trading.position_market_context import (
    load_trading_position_market_frame, normalize_trading_date,
    resolve_trading_strategy_position_sources,
)
from services.trading.protection_planning import build_trading_protection_plan
from services.trading.pending_entry_links import resolve_position_pending_entry
from services.trading.pending_entry_state import load_trading_pending_entry_state, project_trading_pending_intent_entries
from services.trading.strategy_param_runtime import resolve_trading_position_management_binding

TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION = 1


def _build_initial_management_replay_position(record, *, binding, params):
    return build_confirmed_position_origin(record, binding=binding, params=params)


def _replay_position_sell_obligation(record, *, binding, frame, params):
    if str((record.get("strategy_management") or {}).get("sell_signal") or "").strip():
        return None
    result = replay_confirmed_position_management(
        build_confirmed_position_origin(record, binding=binding, params=params, frame=frame), frame=frame, params=params,
        start_date=(record.get("strategy_management") or {}).get("management_start_date"),
    )
    return project_full_exit_obligation(result["full_exit_obligation"])


def _confirmed_events(account, record, *, through_date):
    return collect_confirmed_position_events(effective_trading_account_events(account), record, through_date=through_date)


def _evaluation_identity(record, binding, context, events, prefill_entry=None):
    management = dict(record.get("strategy_management") or {})
    return canonical_json_sha256(json_native_value({
        "replay_contract_version": POSITION_REPLAY_CONTRACT_VERSION,
        "acquisition_origin_contract_version": CONFIRMED_POSITION_ORIGIN_CONTRACT_VERSION,
        "prefill_origin_contract_version": PREFILL_ORIGIN_CONTRACT_VERSION,
        "frozen_binding": binding,
        "accepted_prefill_entry": prefill_entry,
        "market_context": context.fingerprint,
        "lineage_id": binding.get("lineage_id"),
        "frozen_params_sha256": binding.get("frozen_params_sha256"),
        "entry_seed": management.get("entry_execution_plan") or binding.get("execution_plan_seed"),
        "entry_date": (record.get("broker") or {}).get("entry_date"),
        "confirmed_events": events,
        "broker": record.get("broker"),
    }))


def _position_prefill_entry(account, pending_entries, record):
    return resolve_position_pending_entry(
        record, pending_entries, account_events=effective_trading_account_events(account),
        account_audit_events=account.get("events") or (),
    )


def reconcile_trading_manual_position_management(project_root: str | Path, *, lifecycle_context=None) -> dict[str, Any]:
    """Normalize development-era manual holdings to one managed product contract.

    Broker quantity/cost/cash are never changed.  ``manual_adopted`` positions,
    plus prior compatibility activations whose management started later than the
    broker entry date, are rebuilt from the original entry date and advanced to
    the current finalized date with the same canonical management primitives.
    """
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account = load_trading_account_state(root, required=False)
    if not account:
        return {"status": "NO_ACCOUNT", "promoted": [], "errors": {}}

    promoted: list[str] = []
    errors: dict[str, str] = {}
    state = account
    for ticker in sorted(state.get("positions") or {}):
        record = (state.get("positions") or {}).get(ticker)
        if not isinstance(record, dict):
            continue
        source = str(record.get("source") or "")
        broker = dict(record.get("broker") or {})
        entry_date = normalize_trading_date(broker.get("entry_date"))
        management = dict(record.get("strategy_management") or {})
        lineage = dict(record.get("management_lineage") or {})
        management_start = normalize_trading_date(management.get("management_start_date"))
        needs_normalization = source == POSITION_SOURCE_MANUAL_ADOPTED
        if source == POSITION_SOURCE_MANUAL_MANAGED:
            lineage_origin = str(lineage.get("origin") or "")
            legacy_origin = lineage_origin in {"manual_adopted_management", "manual_managed"}
            needs_normalization = bool(
                legacy_origin
                and entry_date is not None
                and management_start != entry_date
            )
        if not needs_normalization:
            continue
        try:
            context = build_manual_adopted_management_context(
                root,
                ticker=str(ticker),
                broker=broker,
                lifecycle_context=lifecycle_context,
            )
            state = activate_existing_manual_trading_position_management(
                root,
                ticker=str(ticker),
                management_lineage=dict(context["management_lineage"]),
                position_state=dict(context["position_state"]),
                management_start_date=context["management_start_date"],
                last_rollforward_date=context.get("last_rollforward_date"),
                initial_position_state=dict(context["initial_position_state"]),
                expected_revision=int(state["revision"]),
            )
            promoted.append(str(ticker))
        except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            errors[str(ticker)] = f"{type(exc).__name__}: {exc}"
            # Refresh after a failed optimistic mutation attempt so a later
            # ticker never reuses a stale revision.
            latest = load_trading_account_state(root, required=False)
            if latest is not None:
                state = latest
    return {
        "status": "OK" if not errors else ("PARTIAL" if promoted else "NO_CHANGE"),
        "promoted": promoted,
        "promoted_count": len(promoted),
        "errors": errors,
    }



def build_trading_position_rollforward_snapshot(project_root: str | Path, *, lifecycle_context=None):
    root = Path(project_root).resolve()
    account = load_trading_account_state(root, required=False)
    if not account or not account.get("positions"):
        return {"schema_version": 1, "runtime_domain": RUNTIME_DOMAIN_TRADING, "due_count": 0,
                "due_tickers": [], "positions": [], "strategy_position_count": 0, "managed_position_count": 0}
    context = lifecycle_context or resolve_trading_lifecycle_context(root)
    account, orders, _view = resolve_trading_strategy_position_sources(root, lifecycle_context=context)
    rows = []
    pending_entries = list(project_trading_pending_intent_entries(
        load_trading_pending_entry_state(root, required=False)).values())
    for ticker, record in sorted(account.get("positions", {}).items()):
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        management = dict(record.get("strategy_management") or {})
        binding = resolve_trading_position_management_binding(record, orders=orders)
        events = _confirmed_events(account, record, through_date=context.finalized_date)
        prefill_entry = _position_prefill_entry(account, pending_entries, record)
        fingerprint = _evaluation_identity(record, binding, context, events, prefill_entry)
        due = str(management.get("evaluation_context_fingerprint") or "") != fingerprint
        rows.append({"ticker": ticker, "entry_date": (record.get("broker") or {}).get("entry_date"),
                     "last_rollforward_date": management.get("last_rollforward_date"),
                     "evaluated_through_date": management.get("evaluated_through_date"),
                     "target_rollforward_date": context.finalized_date if due else None, "due": due,
                     "sell_signal": management.get("sell_signal")})
    due_tickers = [row["ticker"] for row in rows if row["due"]]
    return {"schema_version": 1, "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "allowed_completed_date": context.finalized_date, "strategy_position_count": len(rows),
            "managed_position_count": len(rows), "due_count": len(due_tickers), "due_tickers": due_tickers, "positions": rows}


def run_trading_position_rollforward(project_root: str | Path, *, lifecycle_context=None):
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account = load_trading_account_state(root, required=False)
    if not account:
        return {"status": "NO_ACCOUNT", "runtime_domain": RUNTIME_DOMAIN_TRADING,
                "processed_position_count": 0, "processed_bar_count": 0, "positions": []}
    context = lifecycle_context or resolve_trading_lifecycle_context(root)
    reconciliation = reconcile_trading_manual_position_management(root, lifecycle_context=context)
    account, orders, market_view = resolve_trading_strategy_position_sources(root, lifecycle_context=context)
    pending_entries = list(project_trading_pending_intent_entries(
        load_trading_pending_entry_state(root, required=False)).values())
    updates, rows, errors = {}, [], {}
    signal_tickers = []
    for ticker, record in sorted(account.get("positions", {}).items()):
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            continue
        try:
            management = dict(record.get("strategy_management") or {})
            if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
                raise RuntimeError("Managed position is not active")
            binding = resolve_trading_position_management_binding(record, orders=orders)
            events = _confirmed_events(account, record, through_date=context.finalized_date)
            prefill_entry = _position_prefill_entry(account, pending_entries, record)
            fingerprint = _evaluation_identity(record, binding, context, events, prefill_entry)
            if management.get("evaluation_context_fingerprint") == fingerprint:
                continue
            params = build_params_from_mapping(binding["frozen_params"])
            frame = load_trading_position_market_frame(view=market_view, ticker=ticker, params=params, allowed_date=context.finalized_date)
            if frame.index.max() > pd.Timestamp(context.finalized_date):
                raise RuntimeError("Market frame exceeds the pinned finalized boundary")
            acquisition = build_confirmed_position_origin(
                record, binding=binding, params=params,
                account_events=effective_trading_account_events(account), frame=frame,
                prefill_entry=prefill_entry,
            )
            projection = replay_confirmed_position_management(
                acquisition,
                frame=frame, params=params, start_date=record["broker"].get("entry_date"),
                confirmed_quantity_events=events,
            )
            replayed = projection["position_state"]
            if int(replayed["qty"]) != int(record["broker"]["qty"]):
                raise RuntimeError("Confirmed event replay quantity does not match broker truth")
            # Only copy the core-owned management fields; economics stay actual.
            position = deepcopy(management["position_state"])
            for key in POSITION_MANAGEMENT_FIELDS:
                # Optional obligations from an incompatible old origin must not
                # survive merely because the canonical replay no longer has one.
                position.pop(key, None)
                if key in replayed:
                    position[key] = deepcopy(replayed[key])
            obligation = project_full_exit_obligation(projection["full_exit_obligation"])
            legacy_id = str(binding.get("entry_order_id") or "")
            stop_progress = build_trading_stop_exit_progress(
                orders, ticker=ticker, entry_order_id=str(binding["lineage_key"]),
                compatible_entry_order_ids=[legacy_id] if legacy_id else None,
            )
            if stop_progress.get("triggered"):
                obligation = {"sell_signal": MANAGEMENT_SELL_SIGNAL_STOP,
                    "sell_signal_date": str(stop_progress["trigger_trade_date"]),
                    "sell_signal_trigger_price_milli": int(position["sl_milli"])}
            if obligation is not None:
                signal_tickers.append(ticker)
            updates[ticker] = {
                "position_state": position, "processed_through_date": projection["processed_through_date"],
                "processed_bar_count": len(projection["processed_dates"]), "rebuild": True,
                "evaluation_context_fingerprint": fingerprint, "evaluated_through_date": context.finalized_date,
                "replay_contract_version": POSITION_REPLAY_CONTRACT_VERSION,
                "derived_exit_obligation": obligation,
                "acquisition_replay": {
                    "origin_contract_version": CONFIRMED_POSITION_ORIGIN_CONTRACT_VERSION,
                    "prefill_origin_contract_version": PREFILL_ORIGIN_CONTRACT_VERSION,
                    "frozen_params_sha256": binding.get("frozen_params_sha256"),
                    "pending_entry_id": (prefill_entry or {}).get("pending_entry_id"),
                    "stored_entry_state_sha256": canonical_json_sha256(json_native_value(management.get("entry_position_state"))),
                    "resolved_entry_state_sha256": canonical_json_sha256(json_native_value(acquisition)),
                },
            }
            rows.append({"ticker": ticker, "processed_bar_count": len(projection["processed_dates"]),
                         "processed_from_date": projection["processed_dates"][0],
                         "processed_through_date": projection["processed_through_date"],
                         "previous_stop_milli": int(management["position_state"]["sl_milli"]),
                         "stop_milli": int(position["sl_milli"])})
        except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    if updates:
        account = rollforward_trading_strategy_management(root, updates=updates, expected_revision=int(account["revision"]))
    protection_error = None
    if updates:
        try:
            build_trading_protection_plan(root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            protection_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "PARTIAL" if errors else ("READY" if updates else "UP_TO_DATE"),
        "runtime_domain": RUNTIME_DOMAIN_TRADING, "manual_management_reconcile": reconciliation,
        "account_revision": int(account["revision"]), "allowed_completed_date": context.finalized_date,
        "processed_position_count": len(rows), "processed_bar_count": sum(r["processed_bar_count"] for r in rows),
        "positions": rows, "errors": errors, "sell_signal_count": len(signal_tickers),
        "sell_signal_tickers": signal_tickers, "protection_plan_refresh_error": protection_error,
    }


__all__ = ["TRADING_POSITION_ROLLFORWARD_SCHEMA_VERSION", "reconcile_trading_manual_position_management",
           "build_trading_position_rollforward_snapshot", "run_trading_position_rollforward"]
