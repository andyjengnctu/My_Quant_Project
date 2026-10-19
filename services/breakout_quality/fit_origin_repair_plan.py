"""Canonical, read-only FIT-origin repair preflight for Research model workflows.

This module *selects work*, never certifies legacy weights as REUSE or creates
scientific identities. The existing FIT owner validates all checkpoint bytes,
upstream dependencies and post-build lineage before a controlled rebuild commits.
A score-universe migration does not imply Forward-trained weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from core.breakout_quality_policy import (
    get_breakout_quality_rolling_test_mode,
    get_breakout_quality_workflow_settings,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    resolve_filter_artifact_paths,
)
from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec
from services.breakout_quality.fitted_model_artifacts import resolve_rolling_fitted_folds_root
from services.breakout_quality.fitting_data_dependencies import context_source_for_profile


@dataclass(frozen=True)
class FitOriginRepairCandidate:
    model_id: str
    profile: str
    previous_checkpoint_sha256: str


@dataclass(frozen=True)
class FitOriginRepairBlocker:
    model_id: str
    profile: str
    reason: str


def _load_json(path: Path) -> Mapping | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _same_file(path: Path, expected: Mapping) -> bool:
    return path.is_file() and build_file_manifest(path) == dict(expected)


def _missing_context_origin(
    project_root: Path, model_id: str, profile: str, score_start: str,
) -> tuple[FitOriginRepairCandidate | None, FitOriginRepairBlocker | None]:
    """Locate only a proven *need to rebuild*, not permission to reuse a FIT."""
    try:
        context_source = context_source_for_profile(profile)
    except ValueError:
        # A status probe is not the scientific authorization gate. The
        # canonical producer will reject an unregistered requested profile.
        return None, None
    if maybe_predicted_context_artifact_spec(context_source) is None:
        return None, None
    settings = get_breakout_quality_workflow_settings(experiment_profile=profile)
    folds = resolve_rolling_fitted_folds_root(
        project_root=project_root, filter_id=str(settings.filter_id),
        model_architecture=str(settings.model_architecture), experiment_profile=profile,
    )
    matching = []
    if not folds.is_dir():
        return None, None  # No legacy first fold: canonical producer decides BUILD.
    for path in sorted(folds.glob("*/manifest.json")):
        manifest = _load_json(path)
        if manifest is None:
            continue  # Existing producer/readiness validator owns corrupt Fold diagnostics.
        planned = manifest.get("planned_periods")
        if (isinstance(planned, Mapping)
            and str(planned.get("score_start") or "") == score_start):
            matching.append((path, manifest))
    if not matching:
        return None, None
    if len(matching) != 1:
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling first-fold identity ambiguous")

    first_path, first = matching[0]
    migration = first.get("migration")
    # Scoring refresh metadata is not a FIT origin. Restrict to the known
    # historical rescore shape; do not guess the source of arbitrary migrations.
    if not (isinstance(migration, Mapping)
            and migration.get("kind") == "expanded_daily_score_universe_checkpoint_reuse"
            and migration.get("training_contract_unchanged") is True
            and migration.get("source_checkpoint_sha_preserved") is True
            and not first.get("fitting_origin")
            and not first.get("fitting_context_provenance")):
        return None, None
    if str(first.get("experiment_profile") or "") != profile:
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling first-fold profile mismatch")
    declared = dict(first.get("artifacts") or {}).get("checkpoint")
    model_path = first_path.parent / "model.pt"
    if not isinstance(declared, Mapping) or not _same_file(model_path, declared):
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling checkpoint SHA mismatch")
    sha = declared.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling checkpoint SHA invalid")

    aggregate = _load_json(folds.parent / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME)
    if aggregate is None or not isinstance(aggregate.get("folds"), list):
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling aggregate proof unavailable")
    bound = [entry for entry in aggregate["folds"] if isinstance(entry, Mapping)
             and entry.get("fold_id") == first.get("fold_id")]
    if (len(bound) != 1 or not isinstance(bound[0].get("fold_manifest"), Mapping)
        or not _same_file(first_path, bound[0]["fold_manifest"])):
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling aggregate/Fold SHA mismatch")

    paths = resolve_filter_artifact_paths(
        project_root, str(settings.filter_id), str(settings.model_architecture), profile,
    )
    original = {"sha256": sha, "size_bytes": declared.get("size_bytes")}
    for witness in (
        paths.model_path,
        paths.model_dir / "historical_forward_fits" / sha / "model" / "model.pt",
    ):
        if witness.is_file():
            witness_record = build_file_manifest(witness)
            if (witness_record.get("sha256") == original["sha256"]
                and witness_record.get("size_bytes") == original["size_bytes"]):
                return None, None  # Existing FIT verifier must assess that witness.

    if not bool(settings.rolling_authorized):
        return None, FitOriginRepairBlocker(model_id, profile, "Rolling FIT rebuilding not authorized")
    journal = (project_root / "outputs" / "filters" / "breakout_quality"
               / str(settings.filter_id) / "audit" / "fit_repair"
               / (profile + ".in_progress.json"))
    if journal.exists():
        return None, FitOriginRepairBlocker(
            model_id, profile, "Interrupted FIT repair transaction requires evidence-preserving recovery",
        )
    return FitOriginRepairCandidate(model_id, profile, sha), None


def plan_fit_origin_repairs(
    *, project_root: Path, model_profiles: Iterable[tuple[str, str]],
) -> tuple[tuple[FitOriginRepairCandidate, ...], tuple[FitOriginRepairBlocker, ...]]:
    """Read-only preflight across the requested configured model set."""
    score_start = str(get_breakout_quality_rolling_test_mode("rolling").score_start_date)
    candidates: list[FitOriginRepairCandidate] = []
    blockers: list[FitOriginRepairBlocker] = []
    for model_id, profile in model_profiles:
        candidate, blocker = _missing_context_origin(
            Path(project_root), str(model_id), str(profile), score_start,
        )
        if candidate is not None:
            candidates.append(candidate)
        if blocker is not None:
            blockers.append(blocker)
    return tuple(candidates), tuple(blockers)
