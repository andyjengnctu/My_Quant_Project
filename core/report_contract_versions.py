"""Append-only scientific contract IDs/versions for persistent research reports.

The version ledger, not a Python-side duplicated approval map, owns approved
versions. This provides traceable change control, NOT cryptographic protection
against an attacker who can modify the ledger, validator, and all independent
roots together. User review of each new version remains mandatory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

REGISTRY_RELATIVE_PATH = 'doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json'
_SHA16 = re.compile(r'^[0-9a-f]{16}$')
_SHA64 = re.compile(r'^[0-9a-f]{64}$')


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate contract JSON key: {key}')
        result[key] = value
    return result


def load_report_contract_versions(project_root: Path | None = None) -> dict[str, tuple[dict[str, Any], ...]]:
    """Validate every historical version and any immutable definition file."""
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    path = root / REGISTRY_RELATIVE_PATH
    data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_strict_object)
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('report scientific contract registry schema_version must be 1')
    reports = data.get('reports')
    if not isinstance(reports, list) or not reports:
        raise ValueError('report scientific contract registry has no records')
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for item in reports:
        if not isinstance(item, dict):
            raise ValueError('invalid report registry row')
        report_id = item.get('report_id')
        if not isinstance(report_id, str) or not re.fullmatch(r'[a-z][a-z0-9_.]+', report_id):
            raise ValueError(f'invalid report_id: {report_id!r}')
        if report_id in result:
            raise ValueError(f'duplicate report_id: {report_id}')
        versions = item.get('versions')
        if not isinstance(versions, list) or not versions:
            raise ValueError(f'{report_id}: missing contract versions')
        previous: int | None = None
        records: list[dict[str, Any]] = []
        for row in versions:
            if not isinstance(row, dict):
                raise ValueError(f'{report_id}: invalid version record')
            version = row.get('version')
            if type(version) is not int or version < 1:
                raise ValueError(f'{report_id}: invalid version {version}')
            if previous is not None and version != previous + 1:
                raise ValueError(f'{report_id}: new versions must append exactly +1')
            previous = version
            if row.get('contract_id') != f'{report_id}@v{version}':
                raise ValueError(f'{report_id}: contract_id/version mismatch')
            fingerprint = row.get('fingerprint')
            if not isinstance(fingerprint, str) or not _SHA16.fullmatch(fingerprint):
                raise ValueError(f'{report_id}@v{version}: invalid fingerprint')
            doc_path = row.get('definition_path')
            doc_hash = row.get('definition_sha256')
            if (doc_path is None) != (doc_hash is None):
                raise ValueError(f'{report_id}@v{version}: incomplete immutable definition identity')
            if records and doc_path is None:
                raise ValueError(f'{report_id}@v{version}: revised contract requires immutable definition')
            if doc_path is not None:
                if not isinstance(doc_path, str) or not isinstance(doc_hash, str) or not _SHA64.fullmatch(doc_hash):
                    raise ValueError(f'{report_id}@v{version}: invalid definition path/hash')
                full = (root / doc_path).resolve()
                if not full.is_relative_to(root) or not doc_path.startswith('doc/scientific_contracts/') or not full.is_file():
                    raise ValueError(f'{report_id}@v{version}: scientific definition missing/invalid path')
                if hashlib.sha256(full.read_bytes()).hexdigest() != doc_hash:
                    raise ValueError(f'{report_id}@v{version}: scientific definition changed without new version')
                if not full.name.endswith((f'_v{version}.md',f'_v{version}.json')):
                    raise ValueError(f'{report_id}@v{version}: expected versioned definition filename')
                if full.suffix == '.json':
                    snapshot = json.loads(full.read_text(encoding='utf-8'), object_pairs_hook=_strict_object)
                    digest = hashlib.sha256(json.dumps(
                        snapshot, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                    ).encode('utf-8')).hexdigest()[:16]
                    if digest != fingerprint:
                        raise ValueError(f'{report_id}@v{version}: archived schema fingerprint drift')
            records.append(dict(row))
        result[report_id] = tuple(records)
    if tuple(result) != tuple(sorted(result)):
        raise ValueError('report scientific contract registry must be sorted by report_id')
    return result


def active_report_contract_versions(project_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {name: rows[-1] for name,rows in load_report_contract_versions(project_root).items()}


def validate_active_report_contract_versions(current: Mapping[str, tuple[int, str]],
                                             project_root: Path | None = None) -> tuple[str,...]:
    try:
        approved=active_report_contract_versions(project_root)
    except (ValueError,OSError,UnicodeError) as exc:
        return (f'REPORT_CONTRACT_REGISTRY_INVALID: {exc}',)
    issues: list[str]=[]
    if set(current)!=set(approved):
        issues.append(f'REPORT_CONTRACT_SET_CHANGED: current={sorted(current)} approved={sorted(approved)}')
    for report_id,(version,fingerprint) in current.items():
        expected=approved.get(report_id)
        if expected is None:
            continue
        if version != expected['version']:
            issues.append(f'UNAUTHORIZED_REPORT_VERSION: {report_id} current=v{version} approved=v{expected["version"]}')
        if fingerprint != expected['fingerprint']:
            issues.append(f'UNAUTHORIZED_REPORT_SCHEMA: {report_id} current={fingerprint} approved={expected["fingerprint"]}')
    return tuple(issues)
