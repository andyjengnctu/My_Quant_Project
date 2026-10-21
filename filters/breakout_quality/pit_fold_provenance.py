"""Canonical physical Fold provenance for Selection PIT Score and Audit consumers.

The producer defines the records in its aggregate manifest.  Neither the
Audit publisher nor the Strategy Ranking reader may infer Fold readiness from
aggregate SCORE hashes alone.  Keep this read-only and independent of services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.file_integrity import file_change_signature
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
from filters.breakout_quality.paths import resolve_filter_point_in_time_fold_dir

FOLD_MANIFEST_FILENAME = "manifest.json"
FOLD_SCORE_FILENAME = "scores.csv"
FOLD_VALIDATION_SCORE_FILENAME = "validation_scores.csv"

FOLD_EVIDENCE_FILES = (
    ("fold_manifest", FOLD_MANIFEST_FILENAME),
    ("checkpoint", DEFAULT_MODEL_FILENAME),
    ("scores", FOLD_SCORE_FILENAME),
    ("validation_scores", FOLD_VALIDATION_SCORE_FILENAME),
)


def pit_fold_evidence_paths(
    project_root: str | Path, *, filter_id: str, model_architecture: str,
    experiment_profile: str, fold_id: str,
    point_in_time_dir_override: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve one safe Fold ID in either the canonical or isolated namespace."""
    name = str(fold_id or "").strip()
    if (
        not name or name in {".", ".."} or Path(name).name != name
        or "/" in name or "\\" in name
    ):
        raise ValueError(f"Rolling PIT fold_id不是安全的單一資料夾名稱: {fold_id!r}")
    root = Path(project_root)
    if point_in_time_dir_override not in (None, ""):
        override = Path(point_in_time_dir_override)
        base = override if override.is_absolute() else root / override
        fold_dir = base / "folds" / name
    else:
        fold_dir = resolve_filter_point_in_time_fold_dir(
            root, filter_id, name, model_architecture, experiment_profile,
        )
    return {key: fold_dir / filename for key, filename in FOLD_EVIDENCE_FILES}


def pit_fold_evidence_records(manifest: Mapping[str, Any]) -> tuple[tuple[str, str, Any], ...]:
    """One manifest-to-artifact binding, shared by initial and final validation."""
    rows = manifest.get("folds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Rolling PIT manifest缺少fold records")
    identities = set()
    records = []
    for fold in rows:
        if not isinstance(fold, Mapping):
            raise ValueError("Rolling PIT fold record格式錯誤")
        fold_id = str(fold.get("fold_id") or "").strip()
        if not fold_id or fold_id in identities:
            raise ValueError("Rolling PIT manifest fold_id重複或空白")
        identities.add(fold_id)
        artifacts = fold.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"Rolling PIT {fold_id} artifacts格式錯誤")
        for key, _filename in FOLD_EVIDENCE_FILES:
            record = fold.get("fold_manifest") if key == "fold_manifest" else artifacts.get(key)
            if not isinstance(record, Mapping) or not record.get("sha256"):
                raise ValueError(f"Rolling PIT {fold_id} {key}缺少來源雜湊")
            records.append((fold_id, key, record))
    return tuple(records)


def validate_pit_fold_evidence(
    project_root: str | Path, *, filter_id: str, model_architecture: str,
    experiment_profile: str, manifest: Mapping[str, Any],
    point_in_time_dir_override: str | Path | None = None,
    error_type: type[Exception] = ValueError,
    error_context: str = "來源",
) -> None:
    """Fail closed for *all* Fold files referenced by the aggregate revision."""
    last_id = None
    paths: dict[str, Path] = {}
    for fold_id, key, expected in pit_fold_evidence_records(manifest):
        if fold_id != last_id:
            paths = pit_fold_evidence_paths(
                project_root, filter_id=filter_id,
                model_architecture=model_architecture,
                experiment_profile=experiment_profile, fold_id=fold_id,
                point_in_time_dir_override=point_in_time_dir_override,
            )
            last_id = fold_id
        try:
            actual = build_file_manifest(paths[key])
        except OSError as exc:
            raise error_type(f"Rolling PIT {fold_id} {key}{error_context}遺失或不可讀") from exc
        if actual != expected:
            raise error_type(f"Rolling PIT {fold_id} {key}{error_context}雜湊不一致")


def pit_fold_file_signatures(
    project_root: str | Path, *, filter_id: str, model_architecture: str,
    experiment_profile: str, manifest: Mapping[str, Any],
    point_in_time_dir_override: str | Path | None = None,
) -> tuple:
    """Cheap cache hint including Fold bytes, not a replacement for SHA checks."""
    observed = []
    last_id = None
    paths: dict[str, Path] = {}
    for fold_id, key, _record in pit_fold_evidence_records(manifest):
        if fold_id != last_id:
            paths = pit_fold_evidence_paths(
                project_root, filter_id=filter_id,
                model_architecture=model_architecture,
                experiment_profile=experiment_profile, fold_id=fold_id,
                point_in_time_dir_override=point_in_time_dir_override,
            )
            last_id = fold_id
        observed.append((fold_id, key, file_change_signature(paths[key])))
    return tuple(observed)
