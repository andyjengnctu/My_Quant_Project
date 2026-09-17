"""Canonical persistent state for user-confirmed pre-market Trading entry intents.

Scanner rows and generated proposed orders are advisory artifacts.  A pending entry
is the user's frozen pre-market decision and therefore survives Scanner/Params
refreshes until it is explicitly filled or closed as no-fill.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.event_hash_chain import compute_event_hash
from core.file_integrity import atomic_write_json, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING
from core.runtime_utils import get_taipei_now
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from core.trading_state_paths import resolve_trading_pending_entry_state_path
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.strategy_param_runtime import (
    validate_trading_position_management_lineage,
    validate_trading_position_strategy_lineage,
)

TRADING_PENDING_ENTRY_SCHEMA_VERSION = 1
PENDING_ENTRY_STATUS_ACTIVE = "ACTIVE"
PENDING_ENTRY_STATUS_CANCELLED_NO_FILL = "CANCELLED_NO_FILL"
PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED = "CANCELLED_USER_DELETED"
PENDING_ENTRY_STATUS_FILLED = "FILLED"
PENDING_ENTRY_STATUSES = (
    PENDING_ENTRY_STATUS_ACTIVE,
    PENDING_ENTRY_STATUS_CANCELLED_NO_FILL,
    PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED,
    PENDING_ENTRY_STATUS_FILLED,
)


def _timestamp() -> str:
    return get_taipei_now().isoformat(timespec="seconds")


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": TRADING_PENDING_ENTRY_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "revision": -1,
        "updated_at": None,
        "entries": {},
        "events": [],
    }


def _append_event(state: dict[str, Any], *, mutation_type: str, details: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(state)
    revision = int(updated.get("revision") if updated.get("revision") is not None else -1) + 1
    previous_hash = None if not updated["events"] else updated["events"][-1]["event_hash"]
    event = {
        "revision": revision,
        "mutation_id": uuid4().hex,
        "mutation_type": str(mutation_type),
        "timestamp": _timestamp(),
        "prev_event_hash": previous_hash,
        "details": deepcopy(details),
    }
    event["event_hash"] = compute_event_hash(event)
    updated["revision"] = revision
    updated["updated_at"] = event["timestamp"]
    updated["events"].append(event)
    return updated


def validate_trading_pending_entry_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ValueError("Trading pending entry state 必須是 object")
    if int(state.get("schema_version") or -1) != TRADING_PENDING_ENTRY_SCHEMA_VERSION:
        raise ValueError("Trading pending entry state schema 不相容")
    if str(state.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading pending entry runtime domain 不合法")
    revision = int(state.get("revision") if state.get("revision") is not None else -1)
    entries = state.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Trading pending entries 必須是 object")
    for entry_id, raw in entries.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Trading pending entry 不合法: {entry_id}")
        if str(raw.get("pending_entry_id") or "") != str(entry_id):
            raise ValueError(f"Trading pending entry id 不一致: {entry_id}")
        status = str(raw.get("status") or "")
        if status not in PENDING_ENTRY_STATUSES:
            raise ValueError(f"Trading pending entry status 不合法: {entry_id}/{status}")
        normalize_trading_ticker(raw.get("ticker"))
        normalize_trading_date(raw.get("information_date"), field_name="information_date", allow_none=False)
        if raw.get("planned_trade_date") is not None:
            normalize_trading_date(raw.get("planned_trade_date"), field_name="planned_trade_date", allow_none=False)
        plan = raw.get("execution_plan_seed")
        if not isinstance(plan, dict):
            raise ValueError(f"Trading pending entry 缺少 execution_plan_seed: {entry_id}")
        if int(raw.get("planned_qty") or 0) <= 0:
            raise ValueError(f"Trading pending entry planned_qty 必須 > 0: {entry_id}")
        if int(raw.get("reserved_cost_milli") or 0) <= 0:
            raise ValueError(f"Trading pending entry reserved_cost_milli 必須 > 0: {entry_id}")
        lineage = raw.get("management_lineage")
        if not isinstance(lineage, dict) or not str(lineage.get("lineage_id") or "").strip():
            raise ValueError(f"Trading pending entry 缺少 frozen management lineage: {entry_id}")
        origin = str(raw.get("origin") or "")
        try:
            if origin == "scanner_strategy":
                validate_trading_position_strategy_lineage(lineage)
            elif origin == "manual_selected":
                validate_trading_position_management_lineage(lineage)
            else:
                raise ValueError(f"未知掛單來源: {origin or '-'}")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Trading pending entry frozen lineage 不合法: {entry_id}: {exc}") from exc
        if status == PENDING_ENTRY_STATUS_FILLED and not isinstance(raw.get("fill"), dict):
            raise ValueError(f"FILLED pending entry 缺少 fill evidence: {entry_id}")
    events = state.get("events")
    if not isinstance(events, list) or len(events) != revision + 1:
        raise ValueError("Trading pending entry event count/revision 不一致")
    previous_hash = None
    for expected_revision, event in enumerate(events):
        if int(event.get("revision") if event.get("revision") is not None else -1) != expected_revision:
            raise ValueError("Trading pending entry event revision 不連續")
        if event.get("prev_event_hash") != previous_hash:
            raise ValueError("Trading pending entry event hash chain 斷裂")
        if event.get("event_hash") != compute_event_hash(event):
            raise ValueError("Trading pending entry event hash 不一致")
        previous_hash = event["event_hash"]


def load_trading_pending_entry_state(project_root, *, required: bool = False) -> dict[str, Any] | None:
    path = resolve_trading_pending_entry_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Trading pending entry state 尚未建立: {path}")
        return None
    state = load_json_strict(path)
    validate_trading_pending_entry_state(state)
    return state


def _locked_for_information_date(entry: dict[str, Any], current_information_date: str | None) -> bool:
    # Workbench is a pre/post-market planning tool.  D4 is enforced while an entry
    # remains ACTIVE; once the user closes the intent (delete/no-fill/cancel), the
    # planning reservation is no longer reusable within the same active workflow and
    # can be released immediately for the next pre-market allocation pass.
    return str(entry.get("status") or "") == PENDING_ENTRY_STATUS_ACTIVE


def project_trading_pending_entry_state(
    state: dict[str, Any] | None,
    *,
    current_information_date: str | None,
) -> dict[str, Any]:
    source = _empty_state() if state is None else deepcopy(state)
    rows = [deepcopy(row) for row in source["entries"].values()]
    rows.sort(key=lambda row: (str(row.get("information_date") or ""), str(row.get("created_at") or ""), str(row.get("pending_entry_id") or "")))
    active_rows = [row for row in rows if str(row.get("status")) == PENDING_ENTRY_STATUS_ACTIVE]
    locked_rows = [row for row in rows if _locked_for_information_date(row, current_information_date)]
    stale_active = []
    if current_information_date is not None:
        current_date = normalize_trading_date(current_information_date, field_name="current_information_date", allow_none=False)
        stale_active = [
            row for row in active_rows
            if normalize_trading_date(row.get("information_date"), field_name="information_date", allow_none=False) < current_date
        ]
    return {
        "schema_version": TRADING_PENDING_ENTRY_SCHEMA_VERSION,
        "revision": int(source.get("revision") if source.get("revision") is not None else -1),
        "updated_at": source.get("updated_at"),
        "current_information_date": current_information_date,
        "entries": rows,
        "active_entries": active_rows,
        "locked_entries": locked_rows,
        "active_count": len(active_rows),
        "locked_count": len(locked_rows),
        "reserved_total_milli": sum(int(row.get("reserved_cost_milli") or 0) for row in locked_rows),
        "stale_active_entries": stale_active,
        "stale_active_count": len(stale_active),
    }


@serialized_trading_state_mutation
def create_trading_pending_entry(project_root, *, entry: dict[str, Any]) -> dict[str, Any]:
    path = resolve_trading_pending_entry_state_path(project_root)
    state = load_trading_pending_entry_state(project_root, required=False) or _empty_state()
    payload = deepcopy(dict(entry or {}))
    entry_id = str(payload.get("pending_entry_id") or uuid4().hex)
    if entry_id in state["entries"]:
        raise ValueError(f"Trading pending entry 已存在: {entry_id}")
    ticker = normalize_trading_ticker(payload.get("ticker"))
    for row in state["entries"].values():
        if str(row.get("status")) == PENDING_ENTRY_STATUS_ACTIVE and normalize_trading_ticker(row.get("ticker")) == ticker:
            raise ValueError(f"{ticker} 已有 active 掛單")
    payload["pending_entry_id"] = entry_id
    payload["ticker"] = ticker
    payload["status"] = PENDING_ENTRY_STATUS_ACTIVE
    payload["created_at"] = str(payload.get("created_at") or _timestamp())
    payload["closed_at"] = None
    payload["fill"] = None
    updated = deepcopy(state)
    updated["entries"][entry_id] = payload
    updated = _append_event(updated, mutation_type="create_pending_entry", details={"pending_entry_id": entry_id, "entry": payload})
    validate_trading_pending_entry_state(updated)
    atomic_write_json(path, updated)
    return deepcopy(updated["entries"][entry_id])


@serialized_trading_state_mutation
def update_trading_pending_entry(
    project_root,
    *,
    pending_entry_id: str,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Atomically replace one ACTIVE pre-market intent without double-reserving resources."""
    path = resolve_trading_pending_entry_state_path(project_root)
    state = load_trading_pending_entry_state(project_root, required=True)
    entry_id = str(pending_entry_id or "").strip()
    if entry_id not in state["entries"]:
        raise ValueError(f"找不到 Trading pending entry: {entry_id}")
    current = state["entries"][entry_id]
    if str(current.get("status")) != PENDING_ENTRY_STATUS_ACTIVE:
        raise ValueError(f"只有 ACTIVE 掛單可修改: {entry_id}")

    payload = deepcopy(dict(replacement or {}))
    ticker = normalize_trading_ticker(payload.get("ticker"))
    for other_id, row in state["entries"].items():
        if str(other_id) == entry_id:
            continue
        if str(row.get("status")) == PENDING_ENTRY_STATUS_ACTIVE and normalize_trading_ticker(row.get("ticker")) == ticker:
            raise ValueError(f"{ticker} 已有另一筆 active 掛單")

    payload["pending_entry_id"] = entry_id
    payload["ticker"] = ticker
    payload["status"] = PENDING_ENTRY_STATUS_ACTIVE
    payload["created_at"] = str(current.get("created_at") or payload.get("created_at") or _timestamp())
    payload["updated_at"] = _timestamp()
    payload["closed_at"] = None
    payload["fill"] = None

    updated = deepcopy(state)
    updated["entries"][entry_id] = payload
    updated = _append_event(
        updated,
        mutation_type="update_pending_entry",
        details={
            "pending_entry_id": entry_id,
            "ticker": ticker,
            "previous_ticker": current.get("ticker"),
            "entry": payload,
        },
    )
    validate_trading_pending_entry_state(updated)
    atomic_write_json(path, updated)
    return deepcopy(updated["entries"][entry_id])


@serialized_trading_state_mutation
def cancel_trading_pending_entry(project_root, *, pending_entry_id: str, note: str | None = None) -> dict[str, Any]:
    path = resolve_trading_pending_entry_state_path(project_root)
    state = load_trading_pending_entry_state(project_root, required=True)
    entry_id = str(pending_entry_id or "").strip()
    if entry_id not in state["entries"]:
        raise ValueError(f"找不到 Trading pending entry: {entry_id}")
    entry = state["entries"][entry_id]
    if str(entry.get("status")) != PENDING_ENTRY_STATUS_ACTIVE:
        raise ValueError(f"只有 ACTIVE 掛單可標記無成交: {entry_id}")
    updated = deepcopy(state)
    updated_entry = updated["entries"][entry_id]
    updated_entry["status"] = PENDING_ENTRY_STATUS_CANCELLED_NO_FILL
    updated_entry["closed_at"] = _timestamp()
    updated_entry["cancel_note"] = None if note is None else str(note)
    updated_entry["resource_lock_through_information_date"] = False
    updated = _append_event(
        updated,
        mutation_type="cancel_pending_entry_no_fill",
        details={"pending_entry_id": entry_id, "ticker": updated_entry["ticker"], "note": updated_entry.get("cancel_note")},
    )
    validate_trading_pending_entry_state(updated)
    atomic_write_json(path, updated)
    return deepcopy(updated_entry)


@serialized_trading_state_mutation
def delete_trading_pending_entry(project_root, *, pending_entry_id: str, note: str | None = None) -> dict[str, Any]:
    """Close a user-created pre-market pending entry and release its resources now.

    The close reason remains auditable, but resource semantics are intentionally
    identical to no-fill/cancel: only ACTIVE pending entries reserve cash/slots.
    """
    path = resolve_trading_pending_entry_state_path(project_root)
    state = load_trading_pending_entry_state(project_root, required=True)
    entry_id = str(pending_entry_id or "").strip()
    if entry_id not in state["entries"]:
        raise ValueError(f"找不到 Trading pending entry: {entry_id}")
    entry = state["entries"][entry_id]
    if str(entry.get("status")) != PENDING_ENTRY_STATUS_ACTIVE:
        raise ValueError(f"只有 ACTIVE 掛單可刪除: {entry_id}")
    updated = deepcopy(state)
    updated_entry = updated["entries"][entry_id]
    updated_entry["status"] = PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED
    updated_entry["closed_at"] = _timestamp()
    updated_entry["cancel_note"] = None if note is None else str(note)
    updated_entry["resource_lock_through_information_date"] = False
    updated = _append_event(
        updated,
        mutation_type="delete_pending_entry",
        details={"pending_entry_id": entry_id, "ticker": updated_entry["ticker"], "note": updated_entry.get("cancel_note")},
    )
    validate_trading_pending_entry_state(updated)
    atomic_write_json(path, updated)
    return deepcopy(updated_entry)


@serialized_trading_state_mutation
def mark_trading_pending_entry_filled(
    project_root,
    *,
    pending_entry_id: str,
    fill: dict[str, Any],
) -> dict[str, Any]:
    path = resolve_trading_pending_entry_state_path(project_root)
    state = load_trading_pending_entry_state(project_root, required=True)
    entry_id = str(pending_entry_id or "").strip()
    if entry_id not in state["entries"]:
        raise ValueError(f"找不到 Trading pending entry: {entry_id}")
    entry = state["entries"][entry_id]
    if str(entry.get("status")) != PENDING_ENTRY_STATUS_ACTIVE:
        raise ValueError(f"只有 ACTIVE 掛單可轉為成交: {entry_id}")
    updated = deepcopy(state)
    updated_entry = updated["entries"][entry_id]
    updated_entry["status"] = PENDING_ENTRY_STATUS_FILLED
    updated_entry["closed_at"] = _timestamp()
    updated_entry["fill"] = deepcopy(dict(fill or {}))
    updated = _append_event(
        updated,
        mutation_type="fill_pending_entry",
        details={"pending_entry_id": entry_id, "ticker": updated_entry["ticker"], "fill": updated_entry["fill"]},
    )
    validate_trading_pending_entry_state(updated)
    atomic_write_json(path, updated)
    return deepcopy(updated_entry)


__all__ = [
    "TRADING_PENDING_ENTRY_SCHEMA_VERSION",
    "PENDING_ENTRY_STATUS_ACTIVE",
    "PENDING_ENTRY_STATUS_CANCELLED_NO_FILL",
    "PENDING_ENTRY_STATUS_CANCELLED_USER_DELETED",
    "PENDING_ENTRY_STATUS_FILLED",
    "validate_trading_pending_entry_state",
    "load_trading_pending_entry_state",
    "project_trading_pending_entry_state",
    "create_trading_pending_entry",
    "update_trading_pending_entry",
    "cancel_trading_pending_entry",
    "delete_trading_pending_entry",
    "mark_trading_pending_entry_filled",
]
