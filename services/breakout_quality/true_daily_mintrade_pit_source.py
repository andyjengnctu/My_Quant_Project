"""PIT-gated TrueDaily NetPnL supervision using the existing Min execution owner.

The Oracle's signed 2021+ hindsight labels are never a training dependency.
Historical stock-days are simulated independently with the same canonical
next-market-session previous-close limit and historical seed Min parameters.
"""
from __future__ import annotations

from hashlib import sha256
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.file_integrity import compute_file_sha256
from filters.breakout_quality.true_daily_mintrade_target import (
    compute_true_daily_mintrade_batch,
)
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked,
    FormalMinTrade40PITLabels,
    FormalMinTrade40PITSource,
    slice_formal_mintrade40_pit_source_for_fold,
    verify_formal_mintrade40_pit_source,
)
from services.breakout_quality.formal_mintrade_outcome_cache import (
    load_or_compute_formal_mintrade_outcomes,
)

def materialize_true_daily_mintrade_pit_labels(
    source: FormalMinTrade40PITSource, *, expected_seed: int,
    frame: pd.DataFrame, positions: np.ndarray, ticker: str,
    exchange_calendar: pd.DatetimeIndex, fit_cutoff: Any,
    research_end_is_terminal: bool = False,
    cache_project_root: str | None = None,
) -> FormalMinTrade40PITLabels:
    """Return only completed and legally matured TrueDaily broker outcomes.

    The full frame and Min teacher are truncated to the materialization cutoff.
    The per-fold model FIT still separately requires target maturity strictly
    before its information cutoff; training never reads a hindsight Oracle cache.
    """
    cutoff = pd.Timestamp(fit_cutoff).normalize()
    if pd.isna(cutoff):
        raise FormalMinTrade40PITBlocked("TrueDaily FIT cutoff invalid")
    source, asof = slice_formal_mintrade40_pit_source_for_fold(
        source, fit_cutoff=cutoff,
    )
    frame = frame.loc[pd.DatetimeIndex(frame.index) <= cutoff].copy()
    dates = pd.DatetimeIndex(frame.index).normalize()
    positions = np.asarray(positions, dtype=np.int64)
    if (positions.ndim != 1 or len(positions) == 0 or
            np.any(positions < 0) or np.any(positions + 1 >= len(frame)) or
            np.any(np.diff(positions) <= 0) or not dates.is_monotonic_increasing):
        raise FormalMinTrade40PITBlocked("TrueDaily FIT ticker/session index invalid")
    calendar = pd.DatetimeIndex(exchange_calendar).normalize()
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise FormalMinTrade40PITBlocked("TrueDaily market session calendar invalid")
    calendar = calendar[calendar <= cutoff]
    if calendar.empty or np.any(calendar.get_indexer(dates.take(positions)) < 0):
        raise FormalMinTrade40PITBlocked("TrueDaily PIT information day not on market calendar")
    preflight = verify_formal_mintrade40_pit_source(
        source, expected_seed=expected_seed,
        information_dates=dates.take(positions),
        executable_dates=dates.take(positions + 1), fit_cutoff=cutoff,
    )
    preflight["fold_teacher_as_of_sha256"] = asof
    preflight["entry_policy"] = "next_exchange_session_information_close_limit_v1"

    def calculate():
        batch = compute_true_daily_mintrade_batch(
            frame, positions, ticker=ticker, schedule=source.schedule,
            exchange_calendar=calendar, research_cutoff=cutoff,
        )
        # The shared cache requires its bookkeeping closeout count to be
        # bounded by the number of cutoff-truncated paths. The canonical
        # TrueDaily primitive does not report a horizon-truncation counter.
        return replace(
            batch, cutoff_truncated_rows=max(
                int(batch.cutoff_truncated_rows),
                int(batch.research_cutoff_closeout_rows),
            ),
        )

    if cache_project_root is None:
        batch = calculate()
    else:
        # The cache-key-only ticker prefix separates Daily entries from
        # event entries without modifying the existing shared cache owner.
        # Source identity also includes the distinct Daily entry primitive.
        primitive = (Path(__file__).resolve().parents[2] /
                     "filters/breakout_quality/true_daily_mintrade_target.py")
        digest = sha256((source.param_sha256 + ":" +
                         compute_file_sha256(primitive)).encode("ascii")).hexdigest()
        batch = load_or_compute_formal_mintrade_outcomes(
            project_root=Path(cache_project_root), frame=frame,
            positions=positions, ticker=f"true_daily::{ticker}", exchange_calendar=calendar,
            source_sha256=digest, seed=source.seed,
            schedule_signatures=tuple(
                f"{period.start}:{period.end}:{period.params_signature}:{published}"
                for period, published in zip(source.schedule, source.publication_dates)
            ),
            horizon_sessions=None, research_cutoff=cutoff,
            signal_state_reset_date=source.oos_start_date,
            compute=calculate,
        )
    valid = np.asarray(batch.valid_mask, dtype=bool)
    forced = batch.research_cutoff_forced_mask
    pnl = batch.target_raw_pnl
    if (pnl is None or forced is None or len(valid) != len(positions) or
            len(pnl) != len(valid) or len(forced) != len(valid)):
        raise FormalMinTrade40PITBlocked("TrueDaily broker outcomes missing complete path evidence")
    forced = np.asarray(forced, dtype=bool)
    maturity = np.asarray(batch.maturity_dates, dtype="datetime64[D]").copy()
    if ((forced & ~valid).any() or (~np.isfinite(pnl[valid])).any() or
            np.isfinite(pnl[~valid]).any() or np.isnat(maturity[valid]).any() or
            (maturity[valid] > np.datetime64(cutoff.date(), "D")).any()):
        raise FormalMinTrade40PITBlocked("TrueDaily maturity/NULL/positive evidence inconsistent")
    # Forced end-of-research settlement is an observable evaluation outcome,
    # but never a naturally matured training label for an earlier FIT cutoff.
    completed = valid & (~forced | bool(research_end_is_terminal))
    maturity[~completed] = np.datetime64("NaT")
    net_pnl = np.where(completed, pnl, np.nan).astype(np.float64)
    net_r = np.where(completed, batch.target_raw_r, np.nan).astype(np.float32)
    return FormalMinTrade40PITLabels(
        raw_net_r=net_r, raw_net_pnl=net_pnl,
        completed=completed, censored=np.zeros(len(completed), dtype=bool),
        matured=completed.copy(), maturity_dates=maturity, preflight=preflight,
    )


__all__ = ["materialize_true_daily_mintrade_pit_labels"]
