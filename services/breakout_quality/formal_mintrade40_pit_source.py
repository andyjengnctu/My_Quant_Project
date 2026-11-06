"""Strict PIT source preflight for a future formal MinTrade40 DL label provider.

This is an eligibility contract, not a substitute for the canonical Oracle's
trade-path calculation or for Optimizer-produced per-seed parameter truth.  It
must be called on verified, original Optimizer lineage before labels are FIT.

No active model/strategy is registered until a producer can supply certified
historical pre-OOS Min parameter schedules and per-fold maturity evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.mintrade40_teacher_asof import mintrade40_teacher_asof_sha256
from core.portfolio_param_runtime import build_portfolio_params_signature
from filters.breakout_quality.daily_trade40_target import (
    ExecutableMinPeriod, resolve_historical_min_period,
    HORIZON_SESSIONS, compute_counterfactual_trade40_batch,
)
from services.breakout_quality.formal_mintrade_outcome_cache import (
    load_or_compute_formal_mintrade_outcomes,
)


class FormalMinTrade40PITBlocked(ValueError):
    """A per-seed historical parameter/label source is not FIT-legal."""


@dataclass(frozen=True)
class FormalMinTrade40PITSource:
    """Verified candidate lineage, NOT a self-certifying historical artifact.

    The owner of ``param_sha256`` and ``publication_dates`` must be canonical
    Optimizer. This adapter refuses missing evidence instead of imputing it.
    """
    seed: int
    param_sha256: str
    schedule: tuple[ExecutableMinPeriod, ...]
    # The canonical Optimizer must attest the publication date for every
    # schedule period in the same order. Merely knowing its effective date does
    # not prove the parameter was available when the stock-day occurred.
    publication_dates: tuple[str, ...]
    source_id: str = "formal_min_roos_seed"
    # Boundary between historical FIT teacher and the canonical OOS P16.
    # Pending signals reset on this first evaluation session as in the Oracle.
    oos_start_date: str | None = None



def slice_formal_mintrade40_pit_source_for_fold(
    source: FormalMinTrade40PITSource,
    *,
    fit_cutoff: Any,
) -> tuple[FormalMinTrade40PITSource, str]:
    """Return only historically available teacher periods for ONE FIT cutoff.

    The caller must first obtain the parent through the canonical Optimizer
    provenance loader.  The original parameter file SHA is retained for source
    verification, but the as-of schedule digest intentionally excludes future
    periods: extending the 2020 source must not change the 2017 teacher's
    *semantic* identity.  This digest alone is not a full FIT authorization.

    Truncating a period's end at the cutoff does not refit or replace its
    parameter values.  In particular this cannot backfill an absent earlier
    period or reinterpret later optimized parameters as historical ones.
    """
    try:
        cutoff = pd.Timestamp(fit_cutoff).normalize()
    except (TypeError, ValueError) as exc:
        raise FormalMinTrade40PITBlocked("Invalid MinTrade40 fold FIT cutoff") from exc
    if pd.isna(cutoff):
        raise FormalMinTrade40PITBlocked("Invalid MinTrade40 fold FIT cutoff")
    if (type(source.seed) is not int or source.seed <= 0 or
            len(source.param_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in source.param_sha256.lower()) or
            source.source_id != "formal_min_roos_seed" or
            not source.schedule or
            len(source.schedule) != len(source.publication_dates)):
        raise FormalMinTrade40PITBlocked("Unverified formal MinTrade40 parent source identity")

    periods = []
    publication_dates = []
    previous_end = None
    for period, raw_published in zip(source.schedule, source.publication_dates):
        try:
            published = pd.Timestamp(raw_published).normalize()
        except (TypeError, ValueError) as exc:
            raise FormalMinTrade40PITBlocked("Invalid MinTrade40 source publication date") from exc
        if (pd.isna(published) or pd.isna(period.start) or
                (period.end is not None and (pd.isna(period.end) or period.end < period.start)) or
                (previous_end is not None and period.start <= previous_end) or
                (previous_end is None and periods)):
            raise FormalMinTrade40PITBlocked("Overlapping or invalid MinTrade40 teacher schedule")
        if (period.source_id != source.source_id or
                build_portfolio_params_signature(period.params) != period.params_signature or
                period.params.use_breakout_quality_filter or period.params.use_breakout_quality_ranking):
            raise FormalMinTrade40PITBlocked("Unverified MinTrade40 teacher period identity")
        if published > period.start:
            raise FormalMinTrade40PITBlocked("Future-optimized MinTrade40 teacher publication")
        previous_end = period.end
        if period.start > cutoff:
            continue
        end = cutoff if period.end is None or period.end > cutoff else period.end
        periods.append(replace(period, end=end))
        publication_dates.append(str(published.date()))

    if not periods:
        raise FormalMinTrade40PITBlocked(
            "No historical canonical MinTrade40 teacher available at FIT cutoff"
        )
    as_of_source = replace(
        source, schedule=tuple(periods), publication_dates=tuple(publication_dates)
    )
    # Model FIT will separately bind the source price universe, samples, labels,
    # training settings and seed.  This digest covers teacher semantics only.
    teacher_digest = mintrade40_teacher_asof_sha256(
        seed=source.seed, fit_cutoff=cutoff, oos_start_date=source.oos_start_date,
        records=[
            {"start": str(p.start.date()),
             "end": None if p.end is None else str(p.end.date()),
             "params_signature": p.params_signature, "published": d}
            for p, d in zip(source.schedule, source.publication_dates)
        ],
    )
    return as_of_source, teacher_digest


def verify_formal_mintrade40_pit_source(
    source: FormalMinTrade40PITSource,
    *,
    expected_seed: int,
    information_dates: Iterable[Any],
    executable_dates: Iterable[Any],
    fit_cutoff: Any,
) -> dict[str, Any]:
    """Reject unsupported historical teacher dates before FIT/score materialization.

    Every training sample uses the *next executable ticker session*, not the
    information date. A parameter must have been published no later than that
    session. The full label's maturity remains a separate mandatory fold gate.
    This checker is independent of hindsight Oracle cache and never reads OOS
    target rows or manufactures missing historical parameter periods.
    """
    if type(source.seed) is not int or source.seed != expected_seed:
        raise FormalMinTrade40PITBlocked("Seed-specific formal Min teacher identity mismatch")
    if (len(source.param_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in source.param_sha256.lower())):
        raise FormalMinTrade40PITBlocked("Missing canonical Optimizer parameter SHA256")
    if source.source_id != "formal_min_roos_seed":
        raise FormalMinTrade40PITBlocked("Retired or nonformal Min teacher source rejected")
    schedule = source.schedule
    if not schedule or len(schedule) != len(source.publication_dates):
        raise FormalMinTrade40PITBlocked("Canonical Optimizer historical publication evidence missing")
    if any(period.source_id != source.source_id for period in schedule):
        raise FormalMinTrade40PITBlocked("Formal Min teacher period source mismatch")
    prev_end = None
    published = []
    for index, (period, raw_day) in enumerate(zip(schedule, source.publication_dates)):
        publication = pd.Timestamp(raw_day).normalize()
        if (period.end is not None and period.end < period.start) or (
                index > 0 and (prev_end is None or period.start <= prev_end)):
            raise FormalMinTrade40PITBlocked("Formal Min teacher periods overlap or are unordered")
        if publication > period.start:
            raise FormalMinTrade40PITBlocked("Future-fitted Min parameters backfilled to earlier dates")
        if build_portfolio_params_signature(period.params) != period.params_signature:
            raise FormalMinTrade40PITBlocked("Formal Min teacher parameter signature mismatch")
        if period.params.use_breakout_quality_filter or period.params.use_breakout_quality_ranking:
            raise FormalMinTrade40PITBlocked("Formal Min teacher must be DL-off")
        prev_end = period.end
        published.append(publication)
    info = pd.DatetimeIndex(list(information_dates)).normalize()
    trades = pd.DatetimeIndex(list(executable_dates)).normalize()
    cutoff = pd.Timestamp(fit_cutoff).normalize()
    if any(period.start > cutoff for period in schedule):
        raise FormalMinTrade40PITBlocked("Future-dated Min schedule cannot enter earlier FIT")
    if not len(info) or len(info) != len(trades):
        raise FormalMinTrade40PITBlocked("Missing stock-day information/execution pairs")
    if info.hasnans or trades.hasnans or (info >= trades).any():
        raise FormalMinTrade40PITBlocked("Invalid next-session information/execution chronology")
    if (trades > cutoff).any():
        raise FormalMinTrade40PITBlocked("Post-FIT-cutoff training sample forbidden")
    if info.has_duplicates or trades.has_duplicates or not info.is_monotonic_increasing:
        raise FormalMinTrade40PITBlocked("Duplicate or unordered stock-day training sample")
    period_index_by_id = {id(item): index for index, item in enumerate(schedule)}
    for info_day, trade_day in zip(info, trades):
        period = resolve_historical_min_period(schedule, trade_day)
        if period is None:
            raise FormalMinTrade40PITBlocked(
                f"No historical seed-specific formal Min P16 teacher for "
                f"information date {info_day.date()} (trade {trade_day.date()})"
            )
        period_index = period_index_by_id[id(period)]
        if published[period_index] > trade_day:
            raise FormalMinTrade40PITBlocked("Formal Min P16 teacher is from the future")
    # The digest only binds the candidate evidence. It is NOT an attestation of
    # original Optimizer bytes; the Optimizer owner must verify source.sha first.
    payload = "|".join(
        (str(source.seed), source.param_sha256,
         *(f"{p.start.date()}:{p.end.date() if p.end is not None else ''}:"
           f"{p.params_signature}:{d.date()}" for p, d in zip(schedule, published)),
         str(cutoff.date()), str(info[0].date()), str(trades[-1].date()))
    )
    return {
        "seed": source.seed,
        "param_sha256": source.param_sha256,
        "sample_count": len(info),
        "fit_cutoff": str(cutoff.date()),
        "source_candidate_sha256": sha256(payload.encode("utf-8")).hexdigest(),
        "validation": "PIT_SCHEDULE_PREFLIGHT_ONLY_NOT_FIT_AUTHORIZATION",
    }


@dataclass(frozen=True)
class FormalMinTrade40PITLabels:
    """PIT-matured monetary outcomes; censored stock-days are still NULL."""

    raw_net_r: np.ndarray
    completed: np.ndarray
    censored: np.ndarray
    matured: np.ndarray
    maturity_dates: np.ndarray
    preflight: dict[str, Any]
    raw_net_pnl: np.ndarray | None = None


def materialize_formal_mintrade40_pit_labels(
    source: FormalMinTrade40PITSource,
    *,
    expected_seed: int,
    frame: pd.DataFrame,
    positions: np.ndarray,
    ticker: str,
    exchange_calendar: pd.DatetimeIndex,
    fit_cutoff: Any,
    research_end_is_terminal: bool = False,
    horizon_sessions: int | None = HORIZON_SESSIONS,
    outcome_metric: str = "net_r",
    cache_project_root: str | None = None,
) -> FormalMinTrade40PITLabels:
    """Compute historical FIT-safe canonical Min NetR or NetPnL, fixed/full-cycle.

    No model optimization, score ranking, Oracle percentile mapping or portfolio
    REPLAY is performed here. Only the canonical per-stock execution owner can
    produce Net-R. Full-horizon maturity is sufficient. A natural early exit
    can also mature before FIT, while a closeout forced solely by the FIT
    cutoff is NOT treated as a completed 40D label. Unknowns remain NaN.
    """
    if outcome_metric not in {"net_r", "net_pnl"}:
        raise FormalMinTrade40PITBlocked("Unknown canonical Min realized outcome metric")
    if horizon_sessions is not None and (
        type(horizon_sessions) is not int or horizon_sessions <= 0
    ):
        raise FormalMinTrade40PITBlocked("Invalid canonical Min horizon")
    cutoff = pd.Timestamp(fit_cutoff).normalize()
    source, as_of_digest = slice_formal_mintrade40_pit_source_for_fold(
        source, fit_cutoff=cutoff,
    )
    frame = frame.loc[pd.DatetimeIndex(frame.index) <= cutoff].copy()
    if frame.empty:
        raise FormalMinTrade40PITBlocked("No pre-FIT-cutoff price observations")
    if not pd.DatetimeIndex(frame.index).is_monotonic_increasing:
        raise FormalMinTrade40PITBlocked("Non-canonical ticker price chronology")
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or not len(positions):
        raise FormalMinTrade40PITBlocked("No historical training candidate positions")
    if np.any(positions < 0) or np.any(positions + 1 >= len(frame)):
        raise FormalMinTrade40PITBlocked("Candidate lacks a pre-FIT-cutoff executable session")
    dates = pd.DatetimeIndex(frame.index).normalize()
    if np.any(np.diff(positions) <= 0):
        raise FormalMinTrade40PITBlocked("Training sample positions must be unique and ordered")
    preflight = verify_formal_mintrade40_pit_source(
        source, expected_seed=expected_seed,
        information_dates=dates.take(positions),
        executable_dates=dates.take(positions + 1),
        fit_cutoff=cutoff,
    )
    preflight["fold_teacher_as_of_sha256"] = as_of_digest
    calendar = pd.DatetimeIndex(exchange_calendar).normalize()
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise FormalMinTrade40PITBlocked("Non-canonical market session calendar")
    calendar = calendar[calendar <= cutoff]
    market_pos = calendar.get_indexer(dates.take(positions))
    if np.any(market_pos < 0):
        raise FormalMinTrade40PITBlocked("Training stock-day is absent from exchange calendar")
    def compute_batch(*, include_net_pnl: bool):
        return compute_counterfactual_trade40_batch(
            frame, positions, ticker=ticker, schedule=source.schedule,
            exchange_calendar=calendar, horizon_sessions=horizon_sessions,
            research_cutoff=cutoff, include_net_pnl=include_net_pnl,
            signal_state_reset_date=source.oos_start_date,
        )

    if cache_project_root is not None:
        # Fixed-horizon and Full-cycle outcomes each memoize their own exact
        # broker execution. NetR and NetPnL share the corresponding batch.
        # Retain a disposable, verified per-ticker execution batch; the PIT
        # source preflight and maturity/NULL gates above and below are ALWAYS
        # evaluated afresh, regardless of cache hits.
        batch = load_or_compute_formal_mintrade_outcomes(
            project_root=Path(cache_project_root), frame=frame,
            positions=positions, ticker=ticker, exchange_calendar=calendar,
            source_sha256=source.param_sha256, seed=source.seed,
            schedule_signatures=tuple(
                f"{period.start}:{period.end}:{period.params_signature}:{published}"
                for period, published in zip(source.schedule, source.publication_dates)
            ),
            horizon_sessions=horizon_sessions, research_cutoff=cutoff,
            signal_state_reset_date=source.oos_start_date,
            compute=lambda: compute_batch(include_net_pnl=True),
        )
    else:
        batch = compute_batch(include_net_pnl=outcome_metric == "net_pnl")
    if len(batch.target_raw_r) != len(positions):
        raise FormalMinTrade40PITBlocked("Canonical MinTrade40 label count mismatch")
    full_horizon_matured = (
        market_pos + horizon_sessions < len(calendar)
        if horizon_sessions is not None else np.zeros(len(positions), dtype=bool)
    )
    # Do not infer natural maturity from a calendar comparison. On a weekend
    # cutoff the last Friday is *before* the requested cutoff but can still be
    # an artificial forced closeout. The canonical simulator owns that reason.
    if batch.research_cutoff_forced_mask is None:
        raise FormalMinTrade40PITBlocked("Canonical MinTrade40 cutoff-resolution evidence missing")
    forced = np.asarray(batch.research_cutoff_forced_mask, dtype=bool)
    if len(forced) != len(positions) or np.any(forced & ~batch.valid_mask):
        raise FormalMinTrade40PITBlocked("Invalid MinTrade40 cutoff-resolution evidence")
    resolved_before_cutoff = (
        np.asarray(batch.maturity_dates, dtype="datetime64[D]")
        <= np.datetime64(cutoff.date(), "D")
    )
    natural_early_matured = resolved_before_cutoff & ~forced & np.asarray(batch.valid_mask, dtype=bool)
    # At the *declared research end*, canonical Oracle closeouts are actual
    # evaluation truth. Such forced rows are not mature historical FIT labels:
    # earlier folds still gate them by their own published label-end dates.
    matured = full_horizon_matured | natural_early_matured
    if research_end_is_terminal:
        matured |= forced & resolved_before_cutoff
    matured &= resolved_before_cutoff
    completed = matured & np.asarray(batch.valid_mask, dtype=bool)
    censored = matured & ~np.asarray(batch.valid_mask, dtype=bool)
    raw_values = (batch.target_raw_r if outcome_metric == "net_r"
                  else batch.target_raw_pnl)
    if raw_values is None:
        raise FormalMinTrade40PITBlocked("Missing canonical broker-ledger NetPnL")
    # Preserve broker-ledger currency precision at the source boundary. The
    # trainer may later convert the money label to float32 for tensor ranking;
    # do not truncate the original economic evidence here.
    raw = np.where(completed, raw_values, np.nan).astype(
        np.float64 if outcome_metric == "net_pnl" else np.float32
    )
    if not np.isfinite(raw[completed]).all() or np.isfinite(raw[~completed]).any():
        raise FormalMinTrade40PITBlocked("NULL and PIT label masks disagree")
    maturity_dates = np.asarray(batch.maturity_dates, dtype="datetime64[D]").copy()
    maturity_dates[~matured] = np.datetime64("NaT")
    # Always preserve the documented NetR field. NetPnL is a separate,
    # canonical broker-ledger outcome, never an alias disguised as R units.
    canonical_net_r = (raw if outcome_metric == "net_r" else np.where(
        completed, batch.target_raw_r, np.nan).astype(np.float32))
    return FormalMinTrade40PITLabels(
        raw_net_r=canonical_net_r, completed=completed, censored=censored,
        matured=matured, maturity_dates=maturity_dates, preflight=preflight,
        raw_net_pnl=raw if outcome_metric == "net_pnl" else None,
    )


__all__ = [
    "FormalMinTrade40PITBlocked", "FormalMinTrade40PITSource",
    "slice_formal_mintrade40_pit_source_for_fold",
    "verify_formal_mintrade40_pit_source",
    "FormalMinTrade40PITLabels", "materialize_formal_mintrade40_pit_labels",
]
