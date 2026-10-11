"""Completed Strategy Compare pair-file integrity, independent of scientific identity.

New completed runs pin all pair-local CSV/Markdown and auxiliary JSON *bytes*
in both their run result and manifest. The primary pair JSON is checked against
the run-level canonical result semantically by the existing reuse owner, so
format-only edits to that JSON remain compatible. Historical runs without an
integrity record are accepted through the explicitly legacy-unpinned path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.file_integrity import (
    compute_file_sha256,
    is_sha256_hex,
    load_json_object_or_none,
)

PAIR_FILE_INTEGRITY_FIELD = "pair_file_integrity"
PAIR_FILE_INTEGRITY_SCHEMA_VERSION = 1
_INTEGRITY_SUFFIXES = frozenset({".csv", ".md", ".json"})
_PRIMARY_PAIR_JSON = "strategy_comparison.json"


def _tracked_pair_files(pair_dir: Path) -> dict[str, Path]:
    """Only direct, independently published pair files may enter the digest set."""
    files: dict[str, Path] = {}
    for path in pair_dir.iterdir():
        if path.suffix.lower() not in _INTEGRITY_SUFFIXES or path.name == _PRIMARY_PAIR_JSON:
            continue
        if path.is_symlink():
            raise ValueError(f"pair integrity 不允許符號連結: {path.name}")
        if path.is_file():
            files[path.name] = path
    return dict(sorted(files.items()))


def build_completed_pair_file_integrity(
    pair_directories: Mapping[str, Path],
) -> dict[str, Any]:
    """Capture reproducible byte-digests after ALL pair post-processing finishes."""
    pairs: dict[str, dict[str, str]] = {}
    for pair_id, raw_dir in sorted(pair_directories.items()):
        pair_dir = Path(raw_dir)
        if not pair_dir.is_dir() or not (pair_dir / _PRIMARY_PAIR_JSON).is_file():
            raise ValueError(f"無法封存未完成的 Strategy Compare pair: {pair_id}")
        files = _tracked_pair_files(pair_dir)
        if not any(name.endswith(".csv") for name in files) or not (pair_dir / "strategy_comparison.md").is_file():
            raise ValueError(f"Strategy Compare pair 缺少基本 CSV／Markdown: {pair_id}")
        pairs[str(pair_id)] = {
            name: compute_file_sha256(path) for name, path in files.items()
        }
    if not pairs:
        raise ValueError("Strategy Compare 完成結果缺少可封存 pair")
    return {
        "schema_version": PAIR_FILE_INTEGRITY_SCHEMA_VERSION,
        "pairs": pairs,
    }


def completed_pair_file_integrity_matches(
    *,
    run_dir: Path,
    run_payload: Mapping[str, Any],
    pair_id: str,
    pair_dir: Path,
    required_paths: tuple[Path, ...],
) -> bool:
    """Fail closed on malformed/partial *new* evidence; keep old runs reusable.

    The record lives in BOTH completed-run JSON and manifest. For runs that
    predate it, the existing pair-JSON and scientific identity checks still
    apply. Neither this check nor its record changes pair-cache fingerprints.
    """
    try:
        manifest = load_json_object_or_none(Path(run_dir) / "manifest.json")
        result_record = run_payload.get(PAIR_FILE_INTEGRITY_FIELD)
        manifest_record = (
            manifest.get(PAIR_FILE_INTEGRITY_FIELD)
            if isinstance(manifest, dict) else None
        )
        if result_record is None and manifest_record is None:
            # Historical, explicitly unpinned run; do not invalidate legacy REUSE.
            return True
        if not isinstance(result_record, dict) or result_record != manifest_record:
            return False
        if type(result_record.get("schema_version")) is not int or result_record["schema_version"] != PAIR_FILE_INTEGRITY_SCHEMA_VERSION:
            return False
        pairs = result_record.get("pairs")
        if not isinstance(pairs, dict):
            return False
        digests = pairs.get(str(pair_id))
        if not isinstance(digests, dict) or not digests:
            return False
        required = {path.name for path in required_paths if path.name != _PRIMARY_PAIR_JSON}
        if not required.issubset(digests):
            return False
        recorded_names = set(digests)
        # Detect altered, removed, or unrecorded auxiliary outputs as well as
        # all required CSV/MD. Never follow path components from an artifact.
        if any(
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or Path(name).name != name
            or Path(name).suffix.lower() not in _INTEGRITY_SUFFIXES
            or name == _PRIMARY_PAIR_JSON
            for name in recorded_names
        ):
            return False
        actual_files = _tracked_pair_files(Path(pair_dir))
        if set(actual_files) != recorded_names:
            return False
        for name, path in actual_files.items():
            digest = digests.get(name)
            if not is_sha256_hex(digest):
                return False
            if compute_file_sha256(path) != str(digest).strip().lower():
                return False
        return True
    except (OSError, ValueError, TypeError):
        # A corrupted or temporarily inaccessible completed artifact is NOT reusable.
        return False


__all__ = [
    "PAIR_FILE_INTEGRITY_FIELD",
    "PAIR_FILE_INTEGRITY_SCHEMA_VERSION",
    "build_completed_pair_file_integrity",
    "completed_pair_file_integrity_matches",
]
