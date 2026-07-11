"""Fast source-data freshness fingerprint for breakout quality datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.data_utils import discover_unique_csv_inputs
from core.dataset_profiles import get_dataset_dir, normalize_dataset_profile_key

SOURCE_DATA_INVENTORY_SCHEMA_VERSION = 1
SOURCE_DATA_INVENTORY_ALGORITHM = "sha256(ticker,relative_path,size_bytes,mtime_ns)"


def build_source_data_inventory(project_root: str | Path, dataset: str) -> dict[str, Any]:
    """Build a fast metadata fingerprint for the CSV inputs selected by a dataset profile."""

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

    return {
        "schema_version": SOURCE_DATA_INVENTORY_SCHEMA_VERSION,
        "dataset_profile": profile,
        "csv_file_count": int(len(csv_inputs)),
        "csv_total_bytes": int(total_size_bytes),
        "csv_latest_mtime_ns": int(latest_mtime_ns),
        "csv_inventory_sha256": digest.hexdigest(),
        "fingerprint_algorithm": SOURCE_DATA_INVENTORY_ALGORITHM,
        "ignored_duplicate_count": int(len(duplicate_lines or [])),
    }


__all__ = [
    "SOURCE_DATA_INVENTORY_ALGORITHM",
    "SOURCE_DATA_INVENTORY_SCHEMA_VERSION",
    "build_source_data_inventory",
]
