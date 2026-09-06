"""Canonical content identity for the live Trading CSV dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.dataset_dates import resolve_latest_dataset_date
from core.file_integrity import compute_file_sha256


def build_trading_dataset_fingerprint(data_dir: str | Path) -> dict[str, Any]:
    """Fingerprint every Trading CSV member by path, size and SHA256 content."""

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Trading dataset directory 不存在: {root}")
    members = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if not members:
        raise FileNotFoundError(f"Trading dataset 沒有 CSV: {root}")

    members_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    total_bytes = 0
    for path in members:
        rel = path.relative_to(root).as_posix()
        size = int(path.stat().st_size)
        file_sha = compute_file_sha256(path)
        total_bytes += size
        members_digest.update(rel.encode("utf-8")); members_digest.update(b"\0")
        members_digest.update(str(size).encode("ascii")); members_digest.update(b"\0")
        content_digest.update(rel.encode("utf-8")); content_digest.update(b"\0")
        content_digest.update(file_sha.encode("ascii")); content_digest.update(b"\0")

    return {
        "csv_count": len(members),
        "csv_total_bytes": total_bytes,
        "csv_members_sha256": members_digest.hexdigest(),
        "csv_content_sha256": content_digest.hexdigest(),
        "fingerprint_algorithm": "sha256",
        "latest_data_date": str(resolve_latest_dataset_date(root)),
    }


def assert_trading_dataset_fingerprint_matches(
    data_dir: str | Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when live Trading CSV content diverges from its bound snapshot."""

    actual = build_trading_dataset_fingerprint(data_dir)
    fields = (
        "csv_count",
        "csv_total_bytes",
        "csv_members_sha256",
        "csv_content_sha256",
        "fingerprint_algorithm",
        "latest_data_date",
    )
    mismatches = [field for field in fields if actual.get(field) != expected.get(field)]
    if mismatches:
        raise RuntimeError(
            "Trading dataset content 已與 canonical market-data snapshot 不一致: "
            + ",".join(mismatches)
            + "；請重新執行「1 更新 Trading 資料」"
        )
    return actual


__all__ = [
    "build_trading_dataset_fingerprint",
    "assert_trading_dataset_fingerprint_matches",
]
