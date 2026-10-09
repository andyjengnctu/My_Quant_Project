"""Fast source-data freshness fingerprint for breakout quality datasets."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from core.data_utils import OHLCV_SANITIZATION_CONTRACT_VERSION, discover_unique_csv_inputs
from core.dataset_profiles import DATASET_PROFILE_FULL, get_dataset_dir, normalize_dataset_profile_key

SOURCE_DATA_INVENTORY_SCHEMA_VERSION = 1
SOURCE_DATA_INVENTORY_ALGORITHM = "sha256(ticker,relative_path,size_bytes,mtime_ns)"
# AI: This one-way compatibility proof is specific to the 2026-09-12 change.
PRE_FINITE_STABLE_OHLCV_CONTRACT_VERSION = 2


def _build_source_data_inventory(
    project_root: str | Path, dataset: str, *, legacy_csv_only: bool = False
) -> dict[str, Any]:
    """One metadata algorithm owner for current and archived CSV-only identity."""

    profile = normalize_dataset_profile_key(dataset)
    data_dir = Path(get_dataset_dir(str(project_root), profile)).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到 breakout quality source data directory: {data_dir}")

    csv_inputs, duplicate_lines = discover_unique_csv_inputs(str(data_dir))
    if not csv_inputs:
        raise FileNotFoundError(f"breakout quality source data directory 沒有可用 CSV: {data_dir}")

    digest = hashlib.sha256()
    total_size_bytes = 0
    latest_mtime_ns = 0
    for ticker, raw_path in csv_inputs:
        path = Path(raw_path).resolve()
        stat = path.stat()
        size_bytes = int(stat.st_size)
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
        relative_path = path.relative_to(data_dir).as_posix()

        total_size_bytes += size_bytes
        latest_mtime_ns = max(latest_mtime_ns, mtime_ns)
        for token in (str(ticker), relative_path, str(size_bytes), str(mtime_ns)):
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")

    generation_lineage = None
    fingerprint_algorithm = SOURCE_DATA_INVENTORY_ALGORITHM
    if not legacy_csv_only:
        # AI: Invalidate derived artifacts when canonical OHLCV input semantics change.
        digest.update(f"ohlcv_sanitization_contract={OHLCV_SANITIZATION_CONTRACT_VERSION}\0".encode("utf-8"))
        fingerprint_algorithm += "+ohlcv_sanitization_contract"

    if profile == DATASET_PROFILE_FULL and not legacy_csv_only:
        from config.market_data import RESEARCH_DATA_GENERATION_V2
        from core.market_data_research_promotion import (
            get_effective_research_data_generation,
            load_active_research_v2_promotion,
        )

        active = get_effective_research_data_generation(project_root)
        if active.generation_id == RESEARCH_DATA_GENERATION_V2:
            promoted = load_active_research_v2_promotion(project_root, required=True)
            generation_lineage = {
                "generation_id": active.generation_id,
                "frozen_cutoff": str(active.cutoff),
                "promotion_fingerprint": promoted.promotion_fingerprint,
                "materialization_fingerprint": promoted.materialization_fingerprint,
            }
            for token in (
                generation_lineage["generation_id"],
                generation_lineage["frozen_cutoff"],
                generation_lineage["promotion_fingerprint"],
                generation_lineage["materialization_fingerprint"],
            ):
                digest.update(str(token).encode("utf-8"))
                digest.update(b"\0")
            fingerprint_algorithm += "+research_v2_generation_lineage"

    payload = {
        "schema_version": SOURCE_DATA_INVENTORY_SCHEMA_VERSION,
        "dataset_profile": profile,
        "csv_file_count": int(len(csv_inputs)),
        "csv_total_bytes": int(total_size_bytes),
        "csv_latest_mtime_ns": int(latest_mtime_ns),
        "csv_inventory_sha256": digest.hexdigest(),
        "fingerprint_algorithm": fingerprint_algorithm,
        "ignored_duplicate_count": int(len(duplicate_lines or [])),
    }
    if not legacy_csv_only:
        payload["ohlcv_sanitization_contract_version"] = OHLCV_SANITIZATION_CONTRACT_VERSION
    if generation_lineage is not None:
        payload["research_market_data_lineage"] = generation_lineage
    return payload


def build_source_data_inventory(project_root: str | Path, dataset: str) -> dict[str, Any]:
    """Build canonical metadata fingerprint including source cleaning semantics."""
    return _build_source_data_inventory(project_root, dataset)


@lru_cache(maxsize=8)
def _legacy_sanitizer_semantics_unchanged(
    root: str, dataset: str, current_inventory_sha256: str
) -> bool:
    """Fail-closed content scan for the archived sanitizer-v1 -> v2 delta.

    Prior sanitization could differ from v2 for non-finite OHLCV or unstable
    sorting of duplicate trading dates. Reject both patterns across ALL sources,
    not just the tickers that were selected for a particular fitted model.
    """
    import numpy as np
    import pandas as pd

    data_dir = Path(get_dataset_dir(root, dataset)).resolve()
    csv_inputs, _ = discover_unique_csv_inputs(str(data_dir))
    if not csv_inputs:
        return False
    required = ("Open", "High", "Low", "Close", "Volume")
    try:
        for _, raw_path in csv_inputs:
            frame = pd.read_csv(
                raw_path,
                usecols=lambda name: str(name).capitalize() in (*required, "Time", "Date"),
            )
            frame.columns = [str(col).capitalize() for col in frame.columns]
            date_key = "Time" if "Time" in frame.columns else "Date"
            if date_key not in frame.columns or any(col not in frame.columns for col in required):
                return False
            numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
            if not np.isfinite(numeric).all():
                return False
            dates = pd.to_datetime(frame[date_key], errors="coerce")
            if dates.isna().any() or dates.duplicated().any():
                return False
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        return False
    # A source changed while the scan was running: never cache a passing proof.
    return (
        build_source_data_inventory(root, dataset).get("csv_inventory_sha256")
        == current_inventory_sha256
    )


def legacy_source_inventory_semantically_equivalent(
    project_root: str | Path, *, published: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Check narrowly allowed source-equivalence WITHOUT rewriting old evidence.

    Require exact historical CSV-only digest, exact current digest, and a full
    scan showing the two modified sanitizer behaviors cannot change any row.
    A version mismatch alone is never sufficient to approve reuse.
    """
    if dict(published) == dict(current):
        return True
    if (
        current.get("ohlcv_sanitization_contract_version") != PRE_FINITE_STABLE_OHLCV_CONTRACT_VERSION
        or current.get("fingerprint_algorithm") != SOURCE_DATA_INVENTORY_ALGORITHM + "+ohlcv_sanitization_contract"
        or published.get("fingerprint_algorithm") != SOURCE_DATA_INVENTORY_ALGORITHM
        or "ohlcv_sanitization_contract_version" in published
        or "research_market_data_lineage" in current
        or "research_market_data_lineage" in published
    ):
        return False
    dataset = current.get("dataset_profile")
    if dataset not in ("full", "reduced") or published.get("dataset_profile") != dataset:
        return False
    root = str(Path(project_root).resolve())
    try:
        if dict(current) != build_source_data_inventory(root, dataset):
            return False
        if dict(published) != _build_source_data_inventory(root, dataset, legacy_csv_only=True):
            return False
        return _legacy_sanitizer_semantics_unchanged(
            root, dataset, str(current.get("csv_inventory_sha256"))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


__all__ = [
    "SOURCE_DATA_INVENTORY_ALGORITHM",
    "SOURCE_DATA_INVENTORY_SCHEMA_VERSION",
    "build_source_data_inventory",
    "legacy_source_inventory_semantically_equivalent",
]
