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

import pandas as pd

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


def _normalized_lineage_day(value, *, field_name: str) -> str:
    try:
        day = pd.Timestamp(str(value)).normalize()
    except Exception as exc:
        raise ValueError(f"Trading signal lineage {field_name} is invalid: {value!r}") from exc
    if pd.isna(day):
        raise ValueError(f"Trading signal lineage {field_name} is invalid: {value!r}")
    return day.date().isoformat()


def _can_create_first_observation(row, *, information_date: str) -> bool:
    """Return whether current scan evidence may create a brand-new live lineage.

    A current-Params historical replay may expose an old ``extended`` state, but
    that replay is not durable evidence that Trading actually observed the signal
    on its original date. Historical continuations are therefore eligible only
    through an already-persisted lineage or the explicit previous-snapshot
    bootstrap above. A genuinely new live lineage must originate on today's
    information date.
    """

    signal_day = _normalized_lineage_day(row.get("signal_date"), field_name="signal_date")
    observed_day = _normalized_lineage_day(information_date, field_name="information_date")
    if signal_day > observed_day:
        raise ValueError(
            f"Trading candidate signal_date {signal_day} is later than information_date {observed_day}"
        )
    return signal_day == observed_day


def _close_impossible_current_replay_origins(
    signals: dict, *, information_date: str, source_observation_days: dict
) -> int:
    """Repair precisely evidenced retroactive current-Params first observations.

    The affected first observation may be from today's binding, or from the
    previous durable snapshot (which can be from an earlier market day). We
    never close an arbitrary historical lineage just because it is old: both
    the originating binding and its first-observed day must match evidence.
    """

    current_day = _normalized_lineage_day(information_date, field_name="information_date")
    closed = 0
    for record in signals.values():
        if record.get("status") != "ACTIVE":
            continue
        identity = record.get("identity") or {}
        binding = str(identity.get("source_binding") or "")
        allowed_days = source_observation_days.get(binding, ())
        if not allowed_days:
            continue
        record_observed_day = _normalized_lineage_day(
            record.get("observed_date"), field_name="observed_date"
        )
        if record_observed_day not in allowed_days:
            continue
        signal_day = _normalized_lineage_day(record.get("signal_date"), field_name="signal_date")
        if signal_day >= record_observed_day:
            continue
        record["status"] = "CLOSED"
        record["closed_observed_date"] = current_day
        record["closed_reason"] = "INVALID_RETROACTIVE_CURRENT_PARAMS_REPLAY"
        closed += 1
    return closed


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


def _prepared_frame_through_information_date(frame, *, ticker: str, information_date: str):
    """PIT-cut either canonical raw Scanner frames or already-indexed frames.

    V2 Scanner ``prepared_frames`` are raw six-column frames with a RangeIndex
    and a Date/Time column.  Older synthetic/direct callers may provide an
    already-sanitized DatetimeIndex frame.  Date ownership remains with the
    canonical Scanner sanitizer; this helper only enforces the information-date
    boundary before the frozen-Params producer is called.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{ticker} prepared Scanner frame 必須是 pandas.DataFrame")
    cutoff = pd.Timestamp(str(information_date)).normalize()
    date_col = "Time" if "Time" in frame.columns else ("Date" if "Date" in frame.columns else None)
    if date_col is not None:
        # Cast through string first so YYYYMMDD integer representations are not
        # interpreted by pandas as nanoseconds from epoch.  Unparseable rows are
        # kept for the canonical Scanner sanitizer to reject consistently.
        dates = pd.to_datetime(frame[date_col].astype("string"), errors="coerce").dt.normalize()
        return frame.loc[dates.isna() | (dates <= cutoff)].copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.loc[frame.index.normalize() <= cutoff].copy()
    raise RuntimeError(
        f"{ticker} prepared Scanner frame 缺 Date/Time 欄位且 index 非 DatetimeIndex；"
        "無法安全套用 information-date PIT cutoff"
    )


@serialized_trading_state_mutation
def freeze_trading_candidate_lineages(
    project_root, *, candidate_rows, prepared_frames, information_date,
    strategy_id, source_binding, previous_snapshot=None, pre_persist_guard=None,
):
    # Core ensemble resolves competing signal dates before the frozen-state
    # boundary. Never let a dict keyed by signal identity silently turn two
    # simultaneous candidates for one ticker into two live Trading proposals.
    candidate_rows = list(candidate_rows or [])
    candidate_tickers = [str(row.get("ticker") or "").strip() for row in candidate_rows]
    if len(candidate_tickers) != len(set(candidate_tickers)):
        raise ValueError("Trading Scanner 一檔股票不能同時建立多條候選 signal lineage")
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
    # AI: Permit repair under today's binding, plus the precisely identified
    # previous-snapshot binding if it was trained through its own information
    # day. The snapshot may be older than today's scan (e.g. an overnight restart).
    observed_day = _normalized_lineage_day(information_date, field_name="information_date")
    repair_days: dict[str, set[str]] = {}
    if str(source_binding or "").strip():
        repair_days[str(source_binding)] = {observed_day}
    if isinstance(previous_snapshot, Mapping):
        snapshot_day = str(previous_snapshot.get("latest_data_date") or "")[:10]
        snapshot_param_day = str(previous_snapshot.get("param_latest_data_date") or "")[:10]
        snapshot_binding = str(previous_snapshot.get("param_binding_sha256") or "")
        if snapshot_day and snapshot_day == snapshot_param_day and snapshot_day <= observed_day and snapshot_binding:
            repair_days.setdefault(snapshot_binding, set()).add(snapshot_day)
    _close_impossible_current_replay_origins(
        signals, information_date=information_date, source_observation_days=repair_days
    )
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
            frame = _prepared_frame_through_information_date(
                prepared_frames[ticker], ticker=ticker, information_date=information_date
            )
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
        # AI: A backtest replay under today's trained Params is not historical
        # first-observation evidence. Only today's signal may open a brand-new
        # live lineage here; older signals require an existing lineage or the
        # durable previous-snapshot bootstrap handled above.
        if not _can_create_first_observation(row, information_date=information_date):
            continue
        rec = _record(row, strategy_id=strategy_id, observed_date=information_date, source_binding=source_binding)
        signals[key] = rec
        new_row = deepcopy(row)
        new_row["signal_lineage"] = deepcopy(rec)
        output.append(new_row)
        occupied.add(row["ticker"])
    # Legacy persisted sources must also fail closed if two active frozen
    # lineages for one ticker would be offered simultaneously; do not rewrite
    # either protected historical origin to guess which was intended.
    output_tickers = [str(row.get("ticker") or "").strip() for row in output]
    if len(output_tickers) != len(set(output_tickers)):
        raise RuntimeError("Trading frozen lineage 同股票出現多條候選；禁止重複建議掛單")
    _validate_registry(state)
    if pre_persist_guard is not None:
        pre_persist_guard()
    if state != old_state:
        atomic_write_json(path, state)
    return sort_aggregated_ensemble_candidate_rows(output)
