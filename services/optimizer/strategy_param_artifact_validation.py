"""Optimizer-owned artifact binding checks shared across parameter producers.

These checks validate *publication evidence* only.  Callers own scientific
identity, authorization, coverage, repair eligibility and action policy.
Requirements are evaluated in caller order so diagnostic precedence is stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from core.strategy_param_artifacts import compute_strategy_param_file_sha256


# (manifest field, comparison mode, expected value, diagnostic status)
ManifestRequirement = tuple[str, str, Any, str]


def first_manifest_identity_mismatch(
    manifest: Mapping[str, Any], requirements: Sequence[ManifestRequirement],
) -> str | None:
    """Return the first failed *caller-ordered* publication identity check.

    ``mapping_subset`` permits legacy/publication-only extra manifest fields,
    but every key of the current build contract must still be identical.
    ``boolean`` intentionally compares truthiness for legacy manifests.
    """
    for field, mode, expected, status in requirements:
        actual = manifest.get(field)
        if mode == "string":
            matches = str(actual or "") == str(expected)
        elif mode == "integer":
            matches = int(actual or -1) == int(expected)
        elif mode == "boolean":
            matches = bool(actual) == bool(expected)
        elif mode == "mapping":
            matches = dict(actual or {}) == dict(expected)
        elif mode == "mapping_subset":
            current = dict(actual or {})
            expected_fields = dict(expected)
            matches = {key: current.get(key) for key in expected_fields} == expected_fields
        else:
            raise ValueError(f"unsupported manifest identity comparison: {mode}")
        if not matches:
            return status
    return None


def manifest_artifact_sha_matches(
    manifest: Mapping[str, Any], target: Path, *, sha_field: str = "artifact_sha256",
) -> bool:
    """Compare manifest or nested-record physical SHA (not scientific hash)."""
    return str(manifest.get(sha_field) or "") == compute_strategy_param_file_sha256(target)


def resolve_artifact_reuse_action(
    status: str, *, existed: bool, blocked_statuses: frozenset[str] | set[str] = frozenset(),
    repair_available: bool = False,
) -> str:
    """Map a producer's ordered status to its explicitly authorized action."""
    if status == "READY":
        return "REUSE"
    if status in blocked_statuses:
        return "BLOCKED"
    if repair_available:
        return "REPAIR"
    return "REBUILD" if existed else "BUILD"
