"""Shared, read-only FIT contract version probe across Research and Strategy consumers.

The scientific version and current settings belong to the canonical experiment
registry; this module only compares publication evidence against that owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from core.file_integrity import load_json_object_or_none


def current_fit_contract_artifact_identity(experiment_profile: str) -> dict | None:
    """Immutable FIT source lineage for consumers of versioned model scores.

    v1 has no additional pair/report lineage field so unchanged historical
    scientific results retain their existing fingerprint. Explicit v2+ upgrades
    invalidate the entire dependent result, even if a score projection happens
    to contain identical numeric values.
    """
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile

    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    version = int(profile.fit_contract_version)
    return (
        {"version": version, "experiment_settings": profile.as_manifest_payload()}
        if version > 1 else None
    )


def published_fit_contract_version_state(
    *, experiment_profile: str, publication_dir: Path, rolling: bool,
) -> tuple[str, str]:
    """Version-only FIT staleness probe, shared by menus and consumers.

    A *declared* version upgrade authorizes whole-model rebuild. Unproven
    artifacts or a same-version contract mismatch do not authorize refitting.
    No experiment-name branch or per-fold repair policy is encoded here.
    """
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile

    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    expected = int(profile.fit_contract_version)
    publication_dir = Path(publication_dir)
    # A process killed after the same-volume rename but before building must
    # not hide the previous v1 manifest and silently take the normal RESUME path.
    # The sole bounded backup is inspected read-only; the producer's rebuild
    # guard restores it before the next complete version rebuild.
    backup = publication_dir.with_name(
        "." + publication_dir.name + ".superseded_fit_backup"
    )
    if backup.is_symlink() or publication_dir.is_symlink():
        return "BLOCKED", "FIT version publication/backup is a symlink"
    if backup.exists():
        if not backup.is_dir():
            return "BLOCKED", "FIT version rebuild backup is not a directory"
        if publication_dir.exists():
            return "BLOCKED", "unfinished FIT version rebuild has both publication and backup"
        publication_dir = backup
    if rolling:
        top_path = publication_dir / "selection_point_in_time_manifest.json"
        top = load_json_object_or_none(top_path)
        if top is None:
            return ("BLOCKED", "Rolling manifest unreadable") if top_path.exists() else ("MISSING", "Rolling manifest missing")
        if str(top.get("experiment_profile") or "") != str(experiment_profile):
            return "BLOCKED", "Rolling experiment_profile conflicts with registered FIT owner"
        folds = top.get("folds")
        if (not isinstance(folds, list) or not folds
                or any(not isinstance(row, Mapping) for row in folds)):
            return "BLOCKED", "Rolling fold inventory is missing/invalid"
        fold_ids = [str(row.get("fold_id") or "") for row in folds]
        if len(set(fold_ids)) != len(fold_ids) or any(
            not fid or fid in {".", ".."} or "/" in fid or "\\" in fid
            for fid in fold_ids
        ):
            return "BLOCKED", "Rolling fold inventory is missing/invalid"
        records = [load_json_object_or_none(publication_dir / "folds" / fid / "manifest.json") for fid in fold_ids]
    else:
        model_manifest_path = publication_dir / "manifest.json"
        record = load_json_object_or_none(model_manifest_path)
        if record is None:
            return ("BLOCKED", "Forward model manifest unreadable") if model_manifest_path.exists() else ("MISSING", "Forward model manifest missing")
        records = [record]
    oldest_version = expected
    for record in records:
        if not isinstance(record, dict):
            return "BLOCKED", "FIT manifest missing/corrupt; version cannot be proven"
        if str(record.get("experiment_profile") or "") != str(experiment_profile):
            return "BLOCKED", "FIT experiment_profile mismatch"
        published = record.get("experiment_settings")
        if not isinstance(published, Mapping):
            return "BLOCKED", "FIT experiment_settings absent; cannot infer version"
        raw_version = published.get("fit_contract_version", 1)
        if type(raw_version) is not int:
            return "BLOCKED", "FIT contract version invalid"
        actual = int(raw_version)
        if actual < 1:
            return "BLOCKED", "FIT contract version invalid"
        if actual > expected:
            return "BLOCKED", f"FIT contract downgrade refused: published=v{actual}, current=v{expected}"
        if actual < expected:
            oldest_version = min(oldest_version, actual)
            continue
        if dict(published) != profile.as_manifest_payload():
            return "BLOCKED", "same FIT version but experiment settings changed without version bump"
    # AI: A stale fold cannot authorize replacing a later unverified/conflicting FIT.
    # Validate the whole publication before granting a whole-model rebuild.
    if oldest_version < expected:
        return "UPGRADE_REQUIRED", f"FIT contract version changed: published=v{oldest_version}, current=v{expected}"
    return "CURRENT", f"FIT contract v{expected} verified"

