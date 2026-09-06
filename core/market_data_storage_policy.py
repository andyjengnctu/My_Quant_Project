"""Market Data V2 storage execution policy resolver.

Storage knobs affect publication strategy and operational headroom only.  They
must not enter Research scientific identity.
"""
from __future__ import annotations

from dataclasses import dataclass

from config.market_data import MARKET_DATA_V2_STORAGE_POLICY


@dataclass(frozen=True)
class MarketDataStoragePolicy:
    format: str
    compression: str
    minimum_free_bytes: int
    staging_headroom_multiplier: float
    minimum_staging_headroom_bytes: int


def get_market_data_storage_policy() -> MarketDataStoragePolicy:
    raw = dict(MARKET_DATA_V2_STORAGE_POLICY)
    resolved = MarketDataStoragePolicy(
        format=str(raw.get("format") or "").strip().lower(),
        compression=str(raw.get("compression") or "").strip().lower(),
        minimum_free_bytes=int(raw.get("minimum_free_bytes") or 0),
        staging_headroom_multiplier=float(raw.get("staging_headroom_multiplier") or 0.0),
        minimum_staging_headroom_bytes=int(raw.get("minimum_staging_headroom_bytes") or 0),
    )
    if resolved.format != "parquet":
        raise ValueError(f"Market Data V2 storage format 目前只支援 parquet: {resolved.format!r}")
    if resolved.compression != "zstd":
        raise ValueError(f"Market Data V2 parquet compression 目前只支援 zstd: {resolved.compression!r}")
    if resolved.minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes 不得小於 0")
    if resolved.staging_headroom_multiplier < 1.0:
        raise ValueError("staging_headroom_multiplier 必須 >= 1")
    if resolved.minimum_staging_headroom_bytes < 0:
        raise ValueError("minimum_staging_headroom_bytes 不得小於 0")
    return resolved


__all__ = ["MarketDataStoragePolicy", "get_market_data_storage_policy"]
