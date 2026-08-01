"""Canonical paths for breakout quality dataset, model, experiment, and report artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.breakout_quality_experiments import (
    BASELINE_EXPERIMENT_PROFILE,
    normalize_breakout_quality_experiment_profile,
)
from core.model_paths import resolve_models_dir
from core.output_paths import build_output_dir
from filters.breakout_quality.contract import (
    DEFAULT_EXPERIMENT_PROFILE,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MODEL_ARCHITECTURE,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FILTER_FAMILY,
)
from filters.breakout_quality.models.spec import normalize_model_architecture


@dataclass(frozen=True)
class BreakoutQualityArtifactPaths:
    model_architecture: str
    experiment_profile: str
    model_dir: Path
    model_path: Path
    manifest_path: Path
    score_path: Path
    split_path: Path


def normalize_filter_id(filter_id: str) -> str:
    resolved = str(filter_id).strip()
    if not resolved:
        raise ValueError("filter_id 不可為空白")
    if resolved in {".", ".."} or "\x00" in resolved:
        raise ValueError("filter_id 只能是安全的單一資料夾名稱")
    if Path(resolved).name != resolved or "/" in resolved or "\\" in resolved:
        raise ValueError("filter_id 只能是安全的單一資料夾名稱")
    return resolved


def resolve_model_architecture(model_architecture: str | None = None) -> str:
    return normalize_model_architecture(model_architecture or DEFAULT_MODEL_ARCHITECTURE)


def resolve_experiment_profile(experiment_profile: str | None = None) -> str:
    return normalize_breakout_quality_experiment_profile(
        experiment_profile or DEFAULT_EXPERIMENT_PROFILE
    )


def resolve_filter_model_family_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
) -> Path:
    resolved_filter_id = normalize_filter_id(filter_id)
    return Path(resolve_models_dir(str(project_root))) / "filters" / FILTER_FAMILY / resolved_filter_id


def resolve_filter_model_architecture_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
) -> Path:
    architecture = resolve_model_architecture(model_architecture)
    return resolve_filter_model_family_dir(project_root, filter_id) / architecture


def resolve_filter_model_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    profile = resolve_experiment_profile(experiment_profile)
    return (
        resolve_filter_model_architecture_dir(project_root, filter_id, model_architecture)
        / profile
    )


def _artifact_paths_from_dir(
    *,
    architecture: str,
    experiment_profile: str,
    model_dir: Path,
) -> BreakoutQualityArtifactPaths:
    return BreakoutQualityArtifactPaths(
        model_architecture=architecture,
        experiment_profile=experiment_profile,
        model_dir=model_dir,
        model_path=model_dir / DEFAULT_MODEL_FILENAME,
        manifest_path=model_dir / DEFAULT_MANIFEST_FILENAME,
        score_path=model_dir / DEFAULT_SCORE_FILENAME,
        split_path=model_dir / DEFAULT_SPLIT_FILENAME,
    )


def resolve_filter_artifact_paths(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> BreakoutQualityArtifactPaths:
    architecture = resolve_model_architecture(model_architecture)
    profile = resolve_experiment_profile(experiment_profile)
    return _artifact_paths_from_dir(
        architecture=architecture,
        experiment_profile=profile,
        model_dir=resolve_filter_model_dir(
            project_root,
            filter_id,
            architecture,
            profile,
        ),
    )


def resolve_existing_filter_artifact_paths(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> BreakoutQualityArtifactPaths:
    """Resolve current profile paths, with read-only fallback for old baseline layout."""

    canonical = resolve_filter_artifact_paths(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
    if canonical.manifest_path.is_file():
        return canonical
    if canonical.experiment_profile != BASELINE_EXPERIMENT_PROFILE:
        return canonical
    legacy_dir = resolve_filter_model_architecture_dir(
        project_root,
        filter_id,
        canonical.model_architecture,
    )
    legacy_manifest = legacy_dir / DEFAULT_MANIFEST_FILENAME
    if not legacy_manifest.is_file():
        return canonical
    return _artifact_paths_from_dir(
        architecture=canonical.model_architecture,
        experiment_profile=BASELINE_EXPERIMENT_PROFILE,
        model_dir=legacy_dir,
    )


def ensure_filter_model_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    model_dir = resolve_filter_model_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def resolve_filter_output_dir(
    project_root: str | os.PathLike[str], filter_id: str | None = None
) -> Path:
    """Return the shared dataset output directory; it is model/experiment independent."""

    base_dir = Path(build_output_dir(project_root, "filters")) / FILTER_FAMILY
    if filter_id is None or str(filter_id).strip() == "":
        return base_dir
    return base_dir / normalize_filter_id(filter_id)


def resolve_filter_model_architecture_output_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
) -> Path:
    architecture = resolve_model_architecture(model_architecture)
    return resolve_filter_output_dir(project_root, filter_id=filter_id) / architecture


def resolve_filter_model_output_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    profile = resolve_experiment_profile(experiment_profile)
    return (
        resolve_filter_model_architecture_output_dir(
            project_root,
            filter_id,
            model_architecture,
        )
        / profile
    )


def resolve_filter_research_score_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_model_output_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / "research_scores.csv"


def resolve_filter_research_manifest_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_model_output_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / "research_scores_manifest.json"


def resolve_existing_filter_research_score_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    canonical = resolve_filter_research_score_path(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
    if canonical.is_file() or resolve_experiment_profile(experiment_profile) != BASELINE_EXPERIMENT_PROFILE:
        return canonical
    legacy = (
        resolve_filter_model_architecture_output_dir(
            project_root,
            filter_id,
            model_architecture,
        )
        / "research_scores.csv"
    )
    return legacy if legacy.is_file() else canonical


def resolve_existing_filter_research_manifest_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    canonical = resolve_filter_research_manifest_path(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
    if canonical.is_file() or resolve_experiment_profile(experiment_profile) != BASELINE_EXPERIMENT_PROFILE:
        return canonical
    legacy = (
        resolve_filter_model_architecture_output_dir(
            project_root,
            filter_id,
            model_architecture,
        )
        / "research_scores_manifest.json"
    )
    return legacy if legacy.is_file() else canonical



POINT_IN_TIME_DIRNAME = "point_in_time"
SELECTION_POINT_IN_TIME_SCORE_FILENAME = "selection_point_in_time_scores.csv"
SELECTION_POINT_IN_TIME_MANIFEST_FILENAME = "selection_point_in_time_manifest.json"
SELECTION_POINT_IN_TIME_COVERAGE_FILENAME = "selection_point_in_time_coverage.csv"
SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME = "selection_point_in_time_audit.json"
SELECTION_POINT_IN_TIME_AUDIT_MARKDOWN_FILENAME = "selection_point_in_time_audit.md"


def resolve_filter_point_in_time_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_model_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / POINT_IN_TIME_DIRNAME


def resolve_filter_point_in_time_fold_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    fold_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    normalized_fold_id = str(fold_id).strip()
    if not normalized_fold_id or Path(normalized_fold_id).name != normalized_fold_id:
        raise ValueError("point-in-time fold_id 必須是安全的單一資料夾名稱")
    return resolve_filter_point_in_time_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / "folds" / normalized_fold_id


def resolve_selection_point_in_time_score_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_point_in_time_dir(
        project_root, filter_id, model_architecture, experiment_profile
    ) / SELECTION_POINT_IN_TIME_SCORE_FILENAME


def resolve_selection_point_in_time_manifest_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_point_in_time_dir(
        project_root, filter_id, model_architecture, experiment_profile
    ) / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME


def resolve_selection_point_in_time_coverage_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_point_in_time_dir(
        project_root, filter_id, model_architecture, experiment_profile
    ) / SELECTION_POINT_IN_TIME_COVERAGE_FILENAME


def resolve_filter_point_in_time_output_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_model_output_dir(
        project_root, filter_id, model_architecture, experiment_profile
    ) / POINT_IN_TIME_DIRNAME


def resolve_selection_point_in_time_audit_json_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_point_in_time_output_dir(
        project_root, filter_id, model_architecture, experiment_profile
    ) / SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME


def resolve_selection_point_in_time_audit_markdown_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_point_in_time_output_dir(
        project_root, filter_id, model_architecture, experiment_profile
    ) / SELECTION_POINT_IN_TIME_AUDIT_MARKDOWN_FILENAME

def resolve_filter_report_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_model_output_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / "reports"


def ensure_filter_report_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    report_dir = resolve_filter_report_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def resolve_filter_report_markdown_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_report_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / "evaluation_report.md"


def resolve_filter_report_json_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_report_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    ) / "evaluation_metrics.json"


def ensure_filter_output_dir(
    project_root: str | os.PathLike[str], filter_id: str | None = None
) -> Path:
    out_dir = resolve_filter_output_dir(project_root, filter_id=filter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_filter_model_output_dir(
    project_root: str | os.PathLike[str],
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> Path:
    out_dir = resolve_filter_model_output_dir(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


__all__ = [
    "BreakoutQualityArtifactPaths",
    "ensure_filter_model_dir",
    "ensure_filter_model_output_dir",
    "ensure_filter_output_dir",
    "ensure_filter_report_dir",
    "normalize_filter_id",
    "resolve_existing_filter_artifact_paths",
    "resolve_existing_filter_research_manifest_path",
    "resolve_existing_filter_research_score_path",
    "resolve_experiment_profile",
    "resolve_filter_artifact_paths",
    "resolve_filter_model_architecture_dir",
    "resolve_filter_model_architecture_output_dir",
    "resolve_filter_model_dir",
    "resolve_filter_model_family_dir",
    "resolve_filter_model_output_dir",
    "resolve_filter_output_dir",
    "resolve_filter_research_manifest_path",
    "resolve_filter_research_score_path",
    "resolve_filter_report_dir",
    "resolve_filter_report_json_path",
    "resolve_filter_report_markdown_path",
    "resolve_filter_point_in_time_dir",
    "resolve_filter_point_in_time_fold_dir",
    "resolve_filter_point_in_time_output_dir",
    "resolve_selection_point_in_time_score_path",
    "resolve_selection_point_in_time_manifest_path",
    "resolve_selection_point_in_time_coverage_path",
    "resolve_selection_point_in_time_audit_json_path",
    "resolve_selection_point_in_time_audit_markdown_path",
    "resolve_model_architecture",
]
