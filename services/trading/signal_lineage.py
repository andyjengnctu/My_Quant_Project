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
    # An extended/extended_tbd row is never itself the original BUY signal,
    # even if malformed replay metadata sets its signal_date to today. A stale
    # per-ticker scanner row also cannot create today's frozen live signal.
    trade_day = _normalized_lineage_day(row.get("trade_date"), field_name="trade_date")
    return (signal_day == observed_day == trade_day and str(row.get("kind")) == "buy")


FIRST_OBSERVATION_CURRENT_SIGNAL = "current_signal"
FIRST_OBSERVATION_PREVIOUS_SNAPSHOT = "previous_snapshot"


def _first_observation_has_durable_origin(record) -> bool:
    """A continuation can only originate from an observed signal or verified snapshot.

    Old registries did not record how a first-seen historical continuation was
    admitted. Merely retaining its binding hash is not first-observation proof:
    that hash can become neither current nor previous after later retrains.
    """
    observed = _normalized_lineage_day(record.get("observed_date"), field_name="observed_date")
    signal = _normalized_lineage_day(record.get("signal_date"), field_name="signal_date")
    if signal > observed:
        raise ValueError("Trading first-observed signal is dated in the future")
    if (record.get("identity") or {}).get("first_observation_source") == FIRST_OBSERVATION_PREVIOUS_SNAPSHOT:
        return True
    origin = record.get("origin_candidate") or {}
    try:
        origin_trade = _normalized_lineage_day(origin.get("trade_date"), field_name="origin.trade_date")
    except ValueError:
        # AI: Legacy signed origins without a recorded trade day do not become
        # authentic by assumption; quarantine rather than aborting the Scanner.
        return False
    return signal == observed == origin_trade and str(origin.get("kind")) == "buy"


def _close_impossible_current_replay_origins(
    signals: dict, *, information_date: str, source_observation_days: dict
) -> int:
    """Quarantine unverified retroactive first observations across ALL bindings.

    The original hotfix only checked the current/previous Params binding. Once
    those bindings rotated, the same invalid origins became active again. A
    first observation predating its live signal requires durable prior-snapshot
    provenance, not a particular Params binding or a recent snapshot file.

    Closure is an eligibility quarantine, NOT a retrospective trade exit. The
    complete frozen origin and explicit audit reason remain in the registry.
    """
    current_day = _normalized_lineage_day(information_date, field_name="information_date")
    closed = 0
    for record in signals.values():
        if record.get("status") != "ACTIVE" or _first_observation_has_durable_origin(record):
            continue
        identity = record.get("identity") or {}
        binding = str(identity.get("source_binding") or "")
        observed = _normalized_lineage_day(record.get("observed_date"), field_name="observed_date")
        known_bug_origin = observed in source_observation_days.get(binding, ())
        record["status"] = "CLOSED"
        record["closed_observed_date"] = current_day
        record["closed_reason"] = (
            "INVALID_RETROACTIVE_CURRENT_PARAMS_REPLAY" if known_bug_origin
            else "UNVERIFIED_RETROACTIVE_FIRST_OBSERVATION"
        )
        closed += 1
    return closed


def _bootstrap_durable_previous_snapshot(signals, *, previous_snapshot, strategy_id):
    """Recover proven frozen origins, not current-Params historical projections.

    A snapshot from the bad Scanner version may embed an extended_tbd row that
    was originally adopted retroactively. Such rows are not valid bootstrap
    evidence even if the snapshot itself was saved to disk.
    """
    prior_date = _normalized_lineage_day(previous_snapshot.get("latest_data_date"), field_name="snapshot.latest_data_date")
    for old in previous_snapshot.get("candidate_rows") or []:
        if (old.get("execution_plan_seed") or {}).get("entry_source") == BREAKOUT_REENTRY_SOURCE:
            continue
        # The saved candidate itself must have been actionable on that exact
        # information day, never a stale ticker frame or later data revision.
        if _normalized_lineage_day(old.get("trade_date"), field_name="snapshot.trade_date") != prior_date:
            continue
        nested = old.get("signal_lineage")
        if isinstance(nested, Mapping):
            if (nested.get("status") != "ACTIVE"
                    or str(nested.get("ticker")) != str(old.get("ticker"))
                    or str(nested.get("signal_date")) != str(old.get("signal_date"))
                    or _normalized_lineage_day(nested.get("observed_date"), field_name="nested.observed_date") > prior_date
                    or not _first_observation_has_durable_origin(nested)):
                continue
            # The snapshot must carry an intact original frozen Params identity.
            key = _signal_key(old, strategy_id)
            if str((nested.get("identity") or {}).get("signal_key")) != key:
                continue
            candidate = deepcopy(dict(nested))
            _validate_registry({"schema_version": SIGNAL_LINEAGE_SCHEMA_VERSION, "signals": {key: candidate}})
        else:
            # Legacy snapshots without frozen lineage cannot bootstrap an old
            # continuation. A genuine same-day BUY is sufficient independent
            # evidence, even without the newer embedded lineage format.
            if (str(old.get("kind")) != "buy"
                    or _normalized_lineage_day(old.get("signal_date"), field_name="snapshot.signal_date") != prior_date
                    or _normalized_lineage_day(old.get("trade_date"), field_name="snapshot.trade_date") != prior_date):
                continue
            candidate = _record(
                old, strategy_id=strategy_id, observed_date=prior_date,
                source_binding=previous_snapshot.get("param_binding_sha256"),
                first_observation_source=FIRST_OBSERVATION_PREVIOUS_SNAPSHOT,
            )
            key = candidate["identity"]["signal_key"]
        signals.setdefault(key, candidate)


def _record(row, *, strategy_id, observed_date, source_binding, first_observation_source=None):
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
    if first_observation_source is not None:
        identity["first_observation_source"] = str(first_observation_source)
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
        source = row["identity"].get("first_observation_source")
        if source is not None and source not in {
            FIRST_OBSERVATION_CURRENT_SIGNAL, FIRST_OBSERVATION_PREVIOUS_SNAPSHOT
        }:
            raise ValueError("Trading signal first observation source is invalid")
        if source == FIRST_OBSERVATION_CURRENT_SIGNAL and not _first_observation_has_durable_origin(row):
            raise ValueError("Trading live signal cannot originate before its observation date")


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


def _canonical_tracking_terminal_for_lineage(raw, *, record, information_date: str) -> bool:
    """Only canonical end-of-bar tracking identity may terminate a frozen signal.

    Missing Scanner eligibility, stale data, insufficient rows and a missing
    producer result are *not* terminal evidence. A completed PIT replay whose
    latest bar equals the Scanner information day and whose live tracking
    identity no longer contains this origin is terminal evidence.
    """
    if not isinstance(raw, Mapping):
        return False
    evidence = raw.get("canonical_tracking_evidence")
    if not isinstance(evidence, Mapping):
        return False
    through_day = _normalized_lineage_day(
        evidence.get("through_date"), field_name="tracking.through_date"
    )
    if through_day != _normalized_lineage_day(information_date, field_name="information_date"):
        return False
    original = _normalized_lineage_day(record.get("signal_date"), field_name="signal_date")
    tracked = evidence.get("tracked_signal_date")
    future_signal = evidence.get("today_new_signal_date")
    live_dates = {
        _normalized_lineage_day(day, field_name="tracking.signal_date")
        for day in (tracked, future_signal) if day
    }
    return original not in live_dates


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
    # Snapshot recovery must preserve the original signed frozen Params source.
    # Current replay projections are never admissible as historical evidence.
    if not existed and isinstance(previous_snapshot, Mapping):
        _bootstrap_durable_previous_snapshot(
            signals, previous_snapshot=previous_snapshot, strategy_id=strategy_id
        )
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
    quarantined_count = _close_impossible_current_replay_origins(
        signals, information_date=information_date, source_observation_days=repair_days
    )
    if quarantined_count:
        print(
            f"Trading Scanner lineage 修復：隔離 {quarantined_count} 條缺乏首次買訊證據的歷史延續；"
            "保留原始紀錄與關閉原因。",
            flush=True,
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
            raw = process_raw_stock_frame_actionable_detail(
                frame, ticker, params, include_lifecycle_evidence=True
            )
            if _canonical_tracking_terminal_for_lineage(
                raw, record=record, information_date=information_date
            ):
                record["status"] = "CLOSED"
                record["closed_observed_date"] = information_date
                record["closed_reason"] = "CANONICAL_SIGNAL_TRACKING_TERMINATED"
                continue
            if raw is None or raw.get("kind") not in {"buy", "extended", "extended_tbd"}:
                # AI: A non-qualifying day without explicit terminal tracking
                # evidence is not a reason to rebind a durable frozen origin.
                continue
            if str(raw.get("signal_date")) != str(record["signal_date"]):
                # Another observed signal is not itself a terminal event;
                # closure is owned by the canonical tracking evidence above.
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
        rec = _record(
            row, strategy_id=strategy_id, observed_date=information_date,
            source_binding=source_binding, first_observation_source=FIRST_OBSERVATION_CURRENT_SIGNAL,
        )
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
