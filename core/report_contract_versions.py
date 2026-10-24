"""Report publication approval and compact legacy compatibility fingerprints.

Only this fixed ledger is required. Old schema files, engineering Golden clones
and an external archive are intentionally NOT runtime or validation dependencies.
Historical scientific artifacts preserve their own report ID/fingerprint and
are checked against this ledger without requiring obsolete presentation bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

REGISTRY_RELATIVE_PATH = "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
_SHA16 = re.compile(r"^[0-9a-f]{16}$")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate contract JSON key: {key}")
        result[key] = value
    return result


def load_report_contract_versions(project_root: Path | None = None) -> dict[str, tuple[dict[str, Any], ...]]:
    """Validate one ledger; no per-version schema file or archive read."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    data = json.loads((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"),
                      object_pairs_hook=_strict_object)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("report contract ledger schema_version must be 1")
    reports = data.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("report contract ledger has no records")

    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for item in reports:
        if not isinstance(item, dict):
            raise ValueError("invalid report ledger row")
        report_id = item.get("report_id")
        if not isinstance(report_id, str) or not re.fullmatch(r"[a-z][a-z0-9_.]+", report_id):
            raise ValueError(f"invalid report_id: {report_id!r}")
        if report_id in result:
            raise ValueError(f"duplicate report_id: {report_id}")
        versions = item.get("versions")
        if not isinstance(versions, list) or not versions:
            raise ValueError(f"{report_id}: no report versions")
        previous: int | None = None
        rows: list[dict[str, Any]] = []
        for row in versions:
            if not isinstance(row, dict) or set(row) != {"contract_id", "version", "fingerprint"}:
                raise ValueError(f"{report_id}: expected compact version/fingerprint record")
            version = row["version"]
            if type(version) is not int or version < 1:
                raise ValueError(f"{report_id}: invalid version {version}")
            if previous is not None and version != previous + 1:
                raise ValueError(f"{report_id}: appended versions must be consecutive")
            previous = version
            if row["contract_id"] != f"{report_id}@v{version}":
                raise ValueError(f"{report_id}: contract_id/version mismatch")
            if not isinstance(row["fingerprint"], str) or not _SHA16.fullmatch(row["fingerprint"]):
                raise ValueError(f"{report_id}@v{version}: invalid fingerprint")
            rows.append(dict(row))
        result[report_id] = tuple(rows)
    if tuple(result) != tuple(sorted(result)):
        raise ValueError("report contract ledger must be sorted by report_id")
    return result


def active_report_contract_versions(project_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {name: rows[-1] for name, rows in load_report_contract_versions(project_root).items()}


def validate_active_report_contract_versions(current: Mapping[str, tuple[int, str]],
                                             project_root: Path | None = None) -> tuple[str, ...]:
    try:
        approved = active_report_contract_versions(project_root)
    except (ValueError, OSError, UnicodeError) as exc:
        return (f"REPORT_CONTRACT_REGISTRY_INVALID: {exc}",)
    issues: list[str] = []
    if set(current) != set(approved):
        issues.append(f"REPORT_CONTRACT_SET_CHANGED: current={sorted(current)} approved={sorted(approved)}")
    for report_id, (version, fingerprint) in current.items():
        expected = approved.get(report_id)
        if expected is None:
            continue
        if version != expected["version"]:
            issues.append(f"UNAUTHORIZED_REPORT_VERSION: {report_id} current=v{version} approved=v{expected['version']}")
        if fingerprint != expected["fingerprint"]:
            issues.append(f"UNAUTHORIZED_REPORT_SCHEMA: {report_id} current={fingerprint} approved={expected['fingerprint']}")
    return tuple(issues)
