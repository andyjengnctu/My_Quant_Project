"""Read-only Oracle adapter for MR-13CL's canonical daily-universal Net-R target.

One existing daily ranker target provider supplies BOTH Exact K/R0 and Direct
No-K/No-R0.  Neither selector gets a second label or a special replay path.
This source is hindsight only, never a legal PIT/Trading score publication.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_trade40_target import (
    HORIZON_SESSIONS, LABEL_ID, TARGET_ID, SOURCE_IDS, P16_HANDOFF_DATE,
    collect_canonical_exchange_sessions,
)
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs
from services.audit.min_trade40_target import rank_trade40_outcome_frame


class DailyTrade40OracleBlockedError(ValueError):
    """Missing or inconsistent canonical target/feature/maturity evidence."""


def build_daily_universal_trade40_oracle_scores(
    *, project_root: Path, dataset: str, filter_id: str,
    profile_id: str, cutoff: str,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Return all-feature-eligible, sufficiently mature ex-post daily ranks.

    The official MR-13CL dataset owner computes the *actual* counterfactual
    ledger Net-R; this adapter checks it, rather than implementing a second
    execution/fee/P16 teacher. Invalid but mature labels are ranked *last*,
    solely for the hindsight selector. Their underlying Net-R remains NULL.
    No scores are invented for non-mature stock-days or absent features.
    """
    root = Path(project_root).resolve()
    profile = get_breakout_quality_experiment_profile(profile_id)
    recipe = get_continuous_ranker_execution_recipe(profile_id)
    if (str(profile.continuous_target_id) != TARGET_ID
            or str(recipe.continuous_target_id) != TARGET_ID
            or recipe.target_policy.materialization_mode
            != CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH
            or int(recipe.target_policy.horizon_bars or 0) != HORIZON_SESSIONS):
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle profile/target/horizon identity mismatch")
    try:
        bundle = load_profile_continuous_ranker_data(
            filter_id=filter_id, model_architecture=str(profile.model_architecture),
            experiment_profile=profile_id, preload_feature_bank=False,
            allow_stale_source=False, project_root=root,
        )
        summary = dict(bundle.summary or {})
        if (str(summary.get("dataset") or "") != str(dataset)
                or str(summary.get("filter_id") or "") != str(filter_id)
                or str(summary.get("sample_scope") or "") != "daily_eligible_stock_days"):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle source dataset/filter/feature universe mismatch")
        manifest = dict(bundle.target_manifest or {})
        contract = dict(manifest.get("target_contract") or {})
        if (str(manifest.get("target_id")) != TARGET_ID
                or str(contract.get("target_id")) != TARGET_ID
                or str(contract.get("label_id")) != LABEL_ID
                or int(contract.get("horizon_sessions") or 0) != HORIZON_SESSIONS
                or str(contract.get("sample_scope")) != "daily_eligible_stock_days"
                or contract.get("portfolio_state_inputs") is not False):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle canonical target manifest mismatch")
        # The target is historical P2 -> legally effective P16, not the P16
        # signal-event sample. Never swap in a per-portfolio-seed event target.
        teacher = dict(contract.get("teacher") or {})
        if contract.get("historical_teacher_handoff") != (
            "P2 before 2021-01-01; canonical P16 Min from 2021-01-01"
        ):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle historical teacher handoff mismatch")
        # Provenance is produced by the sole historical Min schedule owner.
        # Both source hashes and each PIT-effective period must be attributable;
        # otherwise a missing historical teacher could silently rank as a loss.
        sources = teacher.get("sources")
        periods = teacher.get("periods")
        if not isinstance(sources, list) or not isinstance(periods, list):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle teacher provenance incomplete")
        source_by_id = {
            str(row.get("source_id")): row for row in sources
            if isinstance(row, Mapping) and row.get("source_id")
        }
        if len(source_by_id) != len(sources) or set(source_by_id) != set(SOURCE_IDS):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle P2/P16 source identity incomplete")
        if any(not isinstance(source_by_id[source].get("sha256"), str)
               or len(source_by_id[source]["sha256"]) != 64
               or any(char not in "0123456789abcdef" for char in source_by_id[source]["sha256"].lower())
               for source in SOURCE_IDS):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle P2/P16 source SHA invalid")
        covered_periods = set()
        for period in periods:
            if not isinstance(period, Mapping) or period.get("source_id") not in source_by_id:
                raise DailyTrade40OracleBlockedError("Daily-universal Oracle teacher period/source mismatch")
            try:
                start = pd.Timestamp(period["start"]).normalize()
                end = (pd.Timestamp(period["end"]).normalize()
                       if period.get("end") is not None else None)
            except (KeyError, TypeError, ValueError) as exc:
                raise DailyTrade40OracleBlockedError("Daily-universal Oracle invalid teacher period dates") from exc
            if (str(period["source_id"]) == SOURCE_IDS[0]
                    and (start >= P16_HANDOFF_DATE or end is None or end >= P16_HANDOFF_DATE)):
                raise DailyTrade40OracleBlockedError("Daily-universal Oracle P2 historical teacher overlaps P16 period")
            if str(period["source_id"]) == SOURCE_IDS[1] and start < P16_HANDOFF_DATE:
                raise DailyTrade40OracleBlockedError("Daily-universal Oracle P16 teacher is backfilled before 2021")
            if end is not None and end < start:
                raise DailyTrade40OracleBlockedError("Daily-universal Oracle reversed teacher period")
            covered_periods.add(str(period["source_id"]))
        if covered_periods != set(SOURCE_IDS):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle historical P2/P16 teacher has missing effective periods")
        csv_inputs, duplicates = discover_dataset_csv_inputs(root, dataset)
        if duplicates:
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle duplicate source ticker")
        exchange = collect_canonical_exchange_sessions(
            csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
        )
        digest = sha256("\n".join(exchange.strftime("%Y-%m-%d")).encode("ascii")).hexdigest()
        if digest != str(contract.get("exchange_calendar_sha256") or ""):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle exchange calendar changed since target materialization")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DailyTrade40OracleBlockedError):
            raise
        raise DailyTrade40OracleBlockedError(
            "Daily-universal Oracle canonical target source unavailable/incompatible: "
            f"{type(exc).__name__}: {str(exc).replace(str(root), '.').replace(chr(92), '/')}"
        ) from exc

    group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
    count = len(group)
    if (int(summary.get("score_eligible_sample_count", -1)) != count
            or int(manifest.get("score_eligible_sample_count", -1)) != count):
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle feature-eligible stock-day count mismatch")
    raw = np.asarray(bundle.raw_target, dtype=np.float64)
    valid = np.asarray(bundle.target_valid, dtype=bool)
    if (not count or len(raw) != count or len(valid) != count
            or not {"ticker", "date", "label_eval_end_date"}.issubset(group.columns)
            or (valid & ~np.isfinite(raw)).any()
            or ((~valid) & np.isfinite(raw)).any()):
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle raw target/validity/stock-day shape or NULL contract invalid")
    dates = pd.DatetimeIndex(pd.to_datetime(group["date"], errors="raise")).normalize()
    positions = exchange.get_indexer(dates)
    if (positions < 0).any():
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle stock-day absent from verified exchange calendar")
    usable_calendar = positions + HORIZON_SESSIONS < len(exchange)
    expiry = pd.DatetimeIndex([exchange[pos + HORIZON_SESSIONS]
                               if mature else pd.NaT
                               for pos, mature in zip(positions, usable_calendar)])
    cutoff_day = pd.Timestamp(cutoff).normalize()
    matured = np.asarray(usable_calendar & (expiry <= cutoff_day), dtype=bool)
    if bool((valid & matured).any()):
        stored_deadline = pd.DatetimeIndex(pd.to_datetime(group["label_eval_end_date"], errors="raise")).normalize()
        if not stored_deadline[valid & matured].equals(expiry[valid & matured]):
            raise DailyTrade40OracleBlockedError("Daily-universal Oracle valid-label 40-session maturity mismatch")
    # Includes *all* eligible stock-days, including censored ones, in each
    # date's denominator. An invalid label's NULL is never a fake 0R.
    usable = matured & (dates <= cutoff_day)
    if not bool(usable.any()):
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle has no 40-session mature stock-days")
    names = group.loc[usable, "ticker"].map(normalize_ticker).tolist()
    day_strings = dates[usable].strftime("%Y-%m-%d").tolist()
    keys = list(zip(names, day_strings))
    if len(set(keys)) != len(keys):
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle duplicate feature-eligible ticker/date")
    selected_valid = valid[usable]
    selected_raw = raw[usable]
    if not selected_valid.any():
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle has no settled Net-R outcomes")
    daily_counts = pd.DataFrame({"date": day_strings, "valid": selected_valid}).groupby("date")["valid"].sum()
    if (daily_counts == 0).any():
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle at least one mature day has no settled Net-R outcomes")
    # Reuse the event Oracle's identical percentile + excluded-last ranking
    # capability; it is ordering-only and cannot mutate original NULL labels.
    scores = rank_trade40_outcome_frame(pd.DataFrame({
        "ticker": names, "date": day_strings,
        "realized_net_r": np.where(selected_valid, selected_raw, np.nan),
    }))
    if set(scores) != set(keys) or not all(np.isfinite(value) for value in scores.values()):
        raise DailyTrade40OracleBlockedError("Daily-universal Oracle daily ranks incomplete")
    raw_digest = sha256(raw.tobytes() + valid.tobytes()).hexdigest()
    rank_sha = sha256()
    for ticker, day in sorted(scores):
        rank_sha.update(f"{ticker}|{day}|{scores[(ticker, day)]:.17g}\n".encode("ascii"))
    evidence = {
        "label_id": LABEL_ID, "target_id": TARGET_ID,
        "profile_id": profile_id, "key_mode": "information_date",
        "cohort": "daily_eligible_stock_days_40_exchange_session_mature",
        "mature_eligible_rows": len(keys),
        "settled_net_r_rows": int(selected_valid.sum()),
        "censored_rank_last_rows": int((~selected_valid).sum()),
        "unmatured_excluded_rows": int((~matured).sum()),
        "first_date": min(day_strings), "last_date": max(day_strings),
        "research_cutoff": str(cutoff_day.date()),
        "rank_sha256": rank_sha.hexdigest(),
        "raw_label_sha256": raw_digest,
        "calendar_sha256": digest,
        "teacher": teacher,
        "rank_policy": "average_zero_based_completed_net_r_censored_rank_last_no_r_imputation",
        "oracle_hindsight_only": True,
    }
    return scores, evidence


__all__ = ["DailyTrade40OracleBlockedError", "build_daily_universal_trade40_oracle_scores"]
