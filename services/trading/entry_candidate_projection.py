"""Account-aware projection of canonical Scanner candidates for new entry intents.

The Scanner snapshot remains immutable scientific/advisory truth.  This module only
projects which rows are currently actionable in Trading after account/pending state
is applied, so a ticker cannot simultaneously appear as a fresh candidate and an
ACTIVE pending entry or open position.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.trading_identity import normalize_trading_ticker


def _position_tickers(account_snapshot: Mapping[str, Any] | None) -> set[str]:
    snapshot = dict(account_snapshot or {})
    rows = snapshot.get("positions") or []
    if isinstance(rows, Mapping):
        values = rows.keys()
    else:
        values = [row.get("ticker") for row in rows if isinstance(row, Mapping)]
    tickers: set[str] = set()
    for value in values:
        try:
            tickers.add(normalize_trading_ticker(value))
        except (TypeError, ValueError):
            continue
    return tickers


def _active_pending_tickers(pending_snapshot: Mapping[str, Any] | None) -> set[str]:
    snapshot = dict(pending_snapshot or {})
    rows = snapshot.get("active_entries")
    if not isinstance(rows, list):
        rows = [
            row for row in list(snapshot.get("entries") or [])
            if isinstance(row, Mapping) and str(row.get("status") or "") == "ACTIVE"
        ]
    tickers: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            tickers.add(normalize_trading_ticker(row.get("ticker")))
        except (TypeError, ValueError):
            continue
    return tickers


def project_trading_entry_candidate_payload(
    candidate_payload: Mapping[str, Any] | None,
    *,
    account_snapshot: Mapping[str, Any] | None,
    pending_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a view projection without mutating the canonical Scanner snapshot."""
    payload = deepcopy(dict(candidate_payload or {}))
    held = _position_tickers(account_snapshot)
    pending = _active_pending_tickers(pending_snapshot)
    excluded = held | pending
    rows = [dict(row) for row in list(payload.get("candidate_rows") or []) if isinstance(row, Mapping)]
    visible = []
    excluded_rows = []
    for row in rows:
        try:
            ticker = normalize_trading_ticker(row.get("ticker"))
        except (TypeError, ValueError):
            visible.append(row)
            continue
        if ticker in excluded:
            excluded_rows.append(row)
        else:
            visible.append(row)
    payload["candidate_rows"] = visible
    payload["entry_projection"] = {
        "held_tickers": sorted(held),
        "active_pending_tickers": sorted(pending),
        "excluded_tickers": sorted(excluded),
        "excluded_count": len(excluded_rows),
        "visible_count": len(visible),
    }
    return payload


__all__ = ["project_trading_entry_candidate_payload"]
