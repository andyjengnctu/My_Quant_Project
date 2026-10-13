"""Immutable, PIT-gated trade-count source for experimental daily-universal inputs.

A historical value in today's FinMind archive is NOT evidence it was available
on the historical model decision date.  Only an actual *contemporaneously finalized*
READY Provider Snapshot may authorize an observation.  The earliest verified
snapshot value for a ticker/source date is kept, rather than retroactively using
revisions that appeared after that stock-day decision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.file_integrity import canonical_json_sha256
from filters.breakout_quality.trade_activity_features import previous_session_trade_count
from filters.breakout_quality.trade_count_capture_preflight import preflight_trade_count_capture_evidence
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import ProviderSnapshotView

TRADE_COUNT_SOURCE_DATASET = "TaiwanStockPriceAdj"
TRADE_COUNT_SOURCE_FIELD = "Trading_turnover"
TRADE_COUNT_PIT_CONTRACT = "archived_first_observation_t_minus_1_v1"


def load_verified_trade_count_observations(project_root: Path) -> dict[str, Any]:
    """Return earliest actually observed trade counts and their verified source identity.

    Provider snapshots are read-only; their ledger and each parquet SHA are checked
    through the canonical repository/view.  One recent archive cannot backdate
    observations into 2021 simply because it contains rows labelled 2021.
    """
    root = Path(project_root).resolve()
    snapshots = preflight_trade_count_capture_evidence(root)["eligible_snapshots"]
    rows: dict[str, dict[pd.Timestamp, tuple[float, pd.Timestamp]]] = {}
    participating: list[tuple[str, str]] = []
    for captured, fingerprint in snapshots:
        archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=fingerprint)
        view = ProviderSnapshotView(project_root=root, archive=archive)
        if not view.dataset_artifacts(TRADE_COUNT_SOURCE_DATASET):
            continue
        # Capture after the source row's trade date is required, but the *snapshot
        # commit day* is the earliest justified availability day for that row.
        participating.append((captured.strftime("%Y-%m-%d"), fingerprint))
        for frame in view.iter_verified_frames(
            TRADE_COUNT_SOURCE_DATASET,
            columns=("date", "stock_id", TRADE_COUNT_SOURCE_FIELD),
        ):
            if frame.empty:
                continue
            dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            counts = pd.to_numeric(frame[TRADE_COUNT_SOURCE_FIELD], errors="coerce")
            tickers = frame["stock_id"].astype(str).str.strip()
            # Only near-capture source rows can possibly qualify for a T-1
            # signal.  Discard years of old backfilled rows before Python iteration.
            plausible = dates >= captured - pd.Timedelta(days=45)
            for ticker, date, count in zip(tickers[plausible], dates[plausible], counts[plausible]):
                if not ticker or pd.isna(count) or not np.isfinite(float(count)) or float(count) < 0:
                    continue
                if date > captured:
                    # Impossible for a snapshot to certify an observation in its future.
                    raise ValueError("trade-count Provider Snapshot包含capture之後的交易日")
                keyed = rows.setdefault(ticker, {})
                prior = keyed.get(date)
                if prior is None:
                    keyed[date] = (float(count), captured)
                elif prior[1] == captured and prior[0] != float(count):
                    raise ValueError("A1 PIT Data Gate：同一snapshot同日同股成交筆數互相矛盾")
                # First contemporaneously observed value is deliberately frozen.
    if not participating:
        raise FileNotFoundError("A1 PIT Data Gate：READY archives缺少 TaiwanStockPriceAdj 成交筆數")
    identity = {
        "source_contract": TRADE_COUNT_PIT_CONTRACT,
        "dataset": TRADE_COUNT_SOURCE_DATASET,
        "field": TRADE_COUNT_SOURCE_FIELD,
        "capture_policy": "first_verified_provider_snapshot_finalized_taipei_day",
        "input_lag": "previous_official_exchange_session",
        "participating_snapshot_fingerprints": [fingerprint for _date, fingerprint in participating],
        "participating_capture_days": [
            {"capture_day": day, "snapshot_fingerprint": fingerprint}
            for day, fingerprint in participating
        ],
    }
    identity["fingerprint"] = canonical_json_sha256(identity)
    return {"identity": identity, "rows": rows}


def load_verified_exchange_calendar(project_root: Path, snapshot_fingerprint: str) -> pd.DatetimeIndex:
    root = Path(project_root).resolve()
    archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
    view = ProviderSnapshotView(project_root=root, archive=archive)
    rows = []
    for frame in view.iter_verified_frames("TaiwanStockTradingDate", columns=("date",)):
        rows.extend(pd.to_datetime(frame["date"], errors="raise").dt.normalize().tolist())
    if not rows:
        raise ValueError("A1 PIT Data Gate：Provider archive缺少 TaiwanStockTradingDate")
    return pd.DatetimeIndex(sorted(set(rows)))


__all__ = [
    "TRADE_COUNT_PIT_CONTRACT",
    "preflight_trade_count_capture_evidence",
    "load_verified_trade_count_observations",
    "load_verified_exchange_calendar",
    "previous_session_trade_count",
]


def resolve_trade_count_feature_source(
    *, model_architecture: str, project_root,
) -> dict[str, Any] | None:
    """Resolve from architecture capability, never MR/profile name."""
    from filters.breakout_quality.models.architectures import get_architecture_descriptor

    desc = get_architecture_descriptor(model_architecture)
    if not desc.has_capability("lagged_trade_count_input"):
        return None
    root = Path(project_root).resolve()
    verified = load_verified_trade_count_observations(root)
    fingerprints = verified["identity"]["participating_snapshot_fingerprints"]
    calendar = load_verified_exchange_calendar(root, fingerprints[-1])
    return {
        **verified,
        "calendar": calendar,
    }

__all__.append("resolve_trade_count_feature_source")
