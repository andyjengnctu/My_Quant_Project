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
from zipfile import BadZipFile, ZipFile

REGISTRY_RELATIVE_PATH = 'doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json'
ARCHIVE_RELATIVE_PATH = 'doc/scientific_contracts/CONTRACT_EVIDENCE_HISTORY.zip'
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
    # A single archive holds the byte-identical historical definitions. The
    # ledger remains the only index/identity owner; no on-disk *_vN.json files
    # are needed. Future authorized versions append a definition to this archive
    # and a new ledger row, NOT another standalone JSON file.
    try:
        with ZipFile(root / ARCHIVE_RELATIVE_PATH) as archive:
            paths = archive.namelist()
            if len(paths) != len(set(paths)):
                raise ValueError('duplicate historical contract archive member')
            definitions = {
                name: archive.read(name) for name in paths
                if name.startswith('doc/scientific_contracts/')
            }
            if any(name.startswith('/') or '..' in Path(name).parts for name in paths):
                raise ValueError('unsafe historical contract archive path')
    except (BadZipFile, OSError, KeyError) as exc:
        raise ValueError(f'historical scientific contract archive missing/invalid: {exc}') from exc
    referenced_definitions: set[str] = set()
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
                if (not doc_path.startswith('doc/scientific_contracts/')
                        or '..' in Path(doc_path).parts
                        or not doc_path.endswith((f'_v{version}.md', f'_v{version}.json'))):
                    raise ValueError(f'{report_id}@v{version}: invalid archived definition name')
                raw = definitions.get(doc_path)
                if raw is None:
                    raise ValueError(f'{report_id}@v{version}: archived scientific definition missing')
                if doc_path in referenced_definitions:
                    raise ValueError(f'{report_id}@v{version}: archived definition referenced twice')
                referenced_definitions.add(doc_path)
                if hashlib.sha256(raw).hexdigest() != doc_hash:
                    raise ValueError(f'{report_id}@v{version}: archived scientific definition changed without new version')
                # During one-time migration, old loose files may coexist with
                # the archive. Reject conflicting copies rather than silently
                # trusting either; they can be removed after validation.
                old_file = root / doc_path
                if old_file.is_file() and old_file.read_bytes() != raw:
                    raise ValueError(f'{report_id}@v{version}: loose definition conflicts with archive')
                if doc_path.endswith('.json'):
                    snapshot = json.loads(raw.decode('utf-8'), object_pairs_hook=_strict_object)
                    digest = hashlib.sha256(json.dumps(
                        snapshot, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                    ).encode('utf-8')).hexdigest()[:16]
                    if digest != fingerprint:
                        raise ValueError(f'{report_id}@v{version}: archived schema fingerprint drift')
            records.append(dict(row))
        result[report_id] = tuple(records)
    if referenced_definitions != set(definitions):
        raise ValueError('historical archive/ledger definition inventory differs')
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
