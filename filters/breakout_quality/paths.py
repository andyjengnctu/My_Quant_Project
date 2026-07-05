"""Canonical paths for breakout quality filter artifacts and tool outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from core.model_paths import resolve_models_dir
from core.output_paths import build_output_dir
from filters.breakout_quality.contract import (
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FILTER_FAMILY,
)


@dataclass(frozen=True)
class BreakoutQualityArtifactPaths:
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


def resolve_filter_model_dir(project_root: str | os.PathLike[str], filter_id: str) -> Path:
    resolved_filter_id = normalize_filter_id(filter_id)
    return Path(resolve_models_dir(str(project_root))) / "filters" / FILTER_FAMILY / resolved_filter_id


def resolve_filter_artifact_paths(
    project_root: str | os.PathLike[str],
    filter_id: str,
) -> BreakoutQualityArtifactPaths:
    model_dir = resolve_filter_model_dir(project_root, filter_id)
    return BreakoutQualityArtifactPaths(
        model_dir=model_dir,
        model_path=model_dir / DEFAULT_MODEL_FILENAME,
        manifest_path=model_dir / DEFAULT_MANIFEST_FILENAME,
        score_path=model_dir / DEFAULT_SCORE_FILENAME,
        split_path=model_dir / DEFAULT_SPLIT_FILENAME,
    )


def ensure_filter_model_dir(project_root: str | os.PathLike[str], filter_id: str) -> Path:
    model_dir = resolve_filter_model_dir(project_root, filter_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def resolve_filter_output_dir(project_root: str | os.PathLike[str], filter_id: str | None = None) -> Path:
    base_dir = Path(build_output_dir(project_root, "filters")) / FILTER_FAMILY
    if filter_id is None or str(filter_id).strip() == "":
        return base_dir
    return base_dir / normalize_filter_id(filter_id)



def resolve_filter_research_score_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
) -> Path:
    return resolve_filter_output_dir(project_root, filter_id=filter_id) / "research_scores.csv"


def resolve_filter_research_manifest_path(
    project_root: str | os.PathLike[str],
    filter_id: str,
) -> Path:
    return resolve_filter_output_dir(project_root, filter_id=filter_id) / "research_scores_manifest.json"

def ensure_filter_output_dir(project_root: str | os.PathLike[str], filter_id: str | None = None) -> Path:
    out_dir = resolve_filter_output_dir(project_root, filter_id=filter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


__all__ = [
    "BreakoutQualityArtifactPaths",
    "ensure_filter_model_dir",
    "ensure_filter_output_dir",
    "normalize_filter_id",
    "resolve_filter_artifact_paths",
    "resolve_filter_model_dir",
    "resolve_filter_output_dir",
    "resolve_filter_research_manifest_path",
    "resolve_filter_research_score_path",
]
