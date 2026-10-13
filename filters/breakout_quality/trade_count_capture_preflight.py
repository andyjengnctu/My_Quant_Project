"""Metadata-only PIT preflight for the non-rebuildable trade-count input.

This lower-layer check never grants row-level data validity: the provider service
verifies ledger, parquet SHA and observation timing during the canonical load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.file_integrity import load_json_strict
from core.market_data_provider_snapshot import validate_provider_snapshot_payload
from core.market_data_storage_contract import (
    MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT,
    MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME,
)


def _snapshot_capture_day(payload: dict[str, Any]) -> pd.Timestamp:
    """Conservatively require capture to finish strictly before the input bar date."""
    raw = payload.get("finalized_at")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("trade-count PIT source必須有snapshot finalized_at證據")
    captured = pd.to_datetime(raw, utc=True, errors="raise")
    return pd.Timestamp(captured).tz_convert("Asia/Taipei").normalize().tz_localize(None)


def _snapshot_refs(root: Path):
    base = root / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT
    if not base.is_dir():
        raise FileNotFoundError("A1 PIT Data Gate：缺少 Market Data V2 Provider archive")
    refs = []
    for path in base.glob(f"*/{MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME}"):
        try:
            payload = validate_provider_snapshot_payload(load_json_strict(path))
        except (OSError, TypeError, ValueError):
            continue
        capture_day = _snapshot_capture_day(payload)
        refs.append((capture_day, str(payload["snapshot_fingerprint"])))
    if not refs:
        raise FileNotFoundError("A1 PIT Data Gate：沒有可驗證的 READY Provider Snapshot")
    # A snapshot fingerprint resolves to one archive. Do not silently attach
    # multiple different finalized dates to one identical snapshot identity.
    unique: dict[str, pd.Timestamp] = {}
    for captured, fingerprint in refs:
        previous = unique.setdefault(fingerprint, captured)
        if previous != captured:
            raise ValueError("A1 PIT Data Gate：相同Provider Snapshot fingerprint有不同capture日")
    return sorted(set(refs))


def preflight_trade_count_capture_evidence(project_root: Path) -> dict[str, Any]:
    """Check the non-rebuildable historical capture-time prerequisite early.

    This is intentionally a *metadata-only preflight*: an eligible finalized
    date is necessary, never sufficient, for PIT acceptance. The canonical
    loader still verifies ledger, parquet SHA, source rows and observation dates.
    """
    root = Path(project_root).resolve()
    snapshots = _snapshot_refs(root)
    from config.market_data import RESEARCH_REQUIRED_CUTOFF
    from config.breakout_quality import BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE

    research_end = pd.Timestamp(RESEARCH_REQUIRED_CUTOFF)
    eligible = [(capture, fp) for capture, fp in snapshots if capture <= research_end]
    if not eligible:
        raise ValueError(
            "A1 PIT Data Gate BLOCKED: no contemporaneous Provider Snapshot "
            f"finalized by frozen Research cutoff {research_end.date()}"
        )
    first_oos = pd.Timestamp(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE)
    if eligible[0][0] >= first_oos:
        raise ValueError(
            "A1 PIT Data Gate BLOCKED: oldest verified Provider Snapshot capture "
            f"{eligible[0][0].date()} is not earlier than OOS start {first_oos.date()}; "
            "historical Trading_turnover rows in a later archive are not PIT proof"
        )
    return {
        "eligible_snapshots": eligible,
        "earliest_capture_day": eligible[0][0].date().isoformat(),
        "verified_manifest_count": len(eligible),
    }


__all__ = ["preflight_trade_count_capture_evidence"]
