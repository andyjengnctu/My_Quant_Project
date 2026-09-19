"""Persistent first-observation strategy binding, upstream of pending orders.

AI: Scanner's current-parameter output is a proposal, not permission to rebind
an existing signal. An active signal is re-evaluated by the canonical Scanner
producer under its original representative Params until that lifecycle ends.
No strategy/indicator/candidate rule is implemented in this storage adapter.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from core.breakout_reentry import BREAKOUT_REENTRY_SOURCE
from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.params_io import build_params_from_mapping
from core.portfolio_ensemble import annotate_ensemble_candidate, sort_aggregated_ensemble_candidate_rows
from core.serialization_utils import json_native_value
from core.trading_state_paths import resolve_trading_state_root
from services.scanner.stock_processor import process_raw_stock_frame_actionable_detail
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.strategy_param_runtime import (
    resolve_trading_candidate_frozen_params, serialize_trading_candidate_member_params,
)

SIGNAL_LINEAGE_SCHEMA_VERSION = 1


def resolve_trading_signal_lineage_path(project_root):
    return resolve_trading_state_root(project_root) / "signal_lineages.json"


def _persistable(row):
    result = deepcopy(dict(row))
    result["ensemble_member_params_by_key"] = serialize_trading_candidate_member_params(result)
    result.pop("signal_lineage", None)
    result.pop("params_obj", None)
    result.pop("_ensemble_context", None)
    return json_native_value(result)


def _signal_key(row, strategy_id):
    ticker = str(row.get("ticker") or "").strip()
    signal_date = str(row.get("signal_date") or "")[:10]
    if not ticker or not signal_date:
        raise ValueError("A candidate cannot freeze without ticker/original signal date")
    return canonical_json_sha256({"strategy_id": strategy_id, "ticker": ticker, "signal_date": signal_date})


def _record(row, *, strategy_id, observed_date, source_binding):
    frozen = _persistable(row)
    params, member = resolve_trading_candidate_frozen_params(frozen)
    del params
    identity = {
        "signal_key": _signal_key(frozen, strategy_id),
        "params_signature": member["params_signature"],
        "representative_member_key": member["member_key"],
        "origin_execution_plan_seed": frozen.get("execution_plan_seed"),
        "source_binding": source_binding,
        "frozen_candidate_sha256": canonical_json_sha256(frozen),
    }
    return {
        "lineage_id": canonical_json_sha256(identity), "identity": identity,
        "ticker": frozen["ticker"], "signal_date": frozen["signal_date"],
        "observed_date": observed_date, "status": "ACTIVE", "origin_candidate": frozen,
    }


def _validate_registry(state):
    if state.get("schema_version") != SIGNAL_LINEAGE_SCHEMA_VERSION or not isinstance(state.get("signals"), dict):
        raise ValueError("Trading signal lineage schema is incompatible")
    for key, row in state["signals"].items():
        if not isinstance(row, dict) or key != row.get("identity", {}).get("signal_key"):
            raise ValueError("Trading signal lineage key is corrupt")
        if canonical_json_sha256(row["identity"]) != row.get("lineage_id"):
            raise ValueError("Trading frozen signal origin was modified")
        if canonical_json_sha256(row["origin_candidate"]) != row["identity"]["frozen_candidate_sha256"]:
            raise ValueError("Trading frozen candidate evidence was modified")
        params, member = resolve_trading_candidate_frozen_params(row["origin_candidate"])
        del params
        if member["params_signature"] != row["identity"]["params_signature"]:
            raise ValueError("Trading frozen signal Params differ from original identity")


@serialized_trading_state_mutation
def freeze_trading_candidate_lineages(
    project_root, *, candidate_rows, prepared_frames, information_date,
    strategy_id, source_binding, previous_snapshot=None, pre_persist_guard=None,
):
    path = resolve_trading_signal_lineage_path(project_root)
    existed = path.exists()
    state = load_json_strict(path) if existed else {"schema_version": SIGNAL_LINEAGE_SCHEMA_VERSION, "signals": {}}
    _validate_registry(state)
    old_state = deepcopy(state)
    signals = state["signals"]
    # Bootstrap only from durable evidence, never from guessed historical params.
    if not existed and isinstance(previous_snapshot, Mapping):
        for old in previous_snapshot.get("candidate_rows") or []:
            if (old.get("execution_plan_seed") or {}).get("entry_source") == BREAKOUT_REENTRY_SOURCE:
                continue
            rec = _record(old, strategy_id=strategy_id,
                          observed_date=previous_snapshot.get("latest_data_date"),
                          source_binding=previous_snapshot.get("param_binding_sha256"))
            signals[rec["identity"]["signal_key"]] = rec
    current = {_signal_key(row, strategy_id): row for row in candidate_rows}
    output, occupied = [], set()
    for key, record in sorted(signals.items(), key=lambda item: (str(item[1]["signal_date"]), item[0])):
        if record["status"] != "ACTIVE":
            continue
        ticker = record["ticker"]
        if ticker not in prepared_frames:
            # Out-of-universe is not an implicit lifecycle termination or a buy permission.
            continue
        frozen = record["origin_candidate"]
        params, member = resolve_trading_candidate_frozen_params(frozen)
        proposed = current.get(key)
        if proposed is not None and proposed.get("params_signature") == member["params_signature"]:
            refreshed = deepcopy(proposed)
        else:
            frame = prepared_frames[ticker]
            frame = frame.loc[frame.index <= information_date]
            raw = process_raw_stock_frame_actionable_detail(frame, ticker, params)
            if raw is None or raw.get("kind") not in {"buy", "extended", "extended_tbd"}:
                # AI: Absence/eligibility is not evidence of strategy termination.
                # Keep the original binding so a returning signal cannot rebind.
                continue
            if str(raw.get("signal_date")) != str(record["signal_date"]):
                record["status"] = "CLOSED"
                record["closed_observed_date"] = information_date
                continue
            refreshed = annotate_ensemble_candidate(raw, member=member, params_obj=params, member_key=member["member_key"])
        # Membership and representative at birth are durable strategy identity.
        for field in ("ensemble_member_key", "ensemble_member_keys", "ensemble_member_count", "ensemble_vote_count", "ensemble_min_agree", "ensemble_member_params_by_key", "params_signature"):
            if field in frozen:
                refreshed[field] = deepcopy(frozen[field])
        refreshed["params_obj"] = params
        refreshed["signal_lineage"] = deepcopy(record)
        output.append(refreshed)
        occupied.add(ticker)
    for key, row in current.items():
        if key in signals or row["ticker"] in occupied:
            continue
        rec = _record(row, strategy_id=strategy_id, observed_date=information_date, source_binding=source_binding)
        signals[key] = rec
        new_row = deepcopy(row)
        new_row["signal_lineage"] = deepcopy(rec)
        output.append(new_row)
        occupied.add(row["ticker"])
    _validate_registry(state)
    if pre_persist_guard is not None:
        pre_persist_guard()
    if state != old_state:
        atomic_write_json(path, state)
    return sort_aggregated_ensemble_candidate_rows(output)
