"""Shared output namespace resolution; no trainer or publication dependencies."""
from __future__ import annotations
from pathlib import Path
from filters.breakout_quality.paths import (
    build_filter_artifact_paths_from_dir, resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
)


def resolve_ranker_training_output_paths(args, *, project_root: Path):
    model_override = str(getattr(args, "model_output_dir", "") or "").strip()
    research_override = str(getattr(args, "research_output_dir", "") or "").strip()
    if bool(model_override) != bool(research_override):
        raise ValueError("model-output-dir與research-output-dir必須同時指定")
    if model_override:
        artifact_paths = build_filter_artifact_paths_from_dir(
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(args.experiment_profile),
            model_dir=Path(model_override),
        )
        output_dir = Path(research_override).resolve()
        return artifact_paths, output_dir
    return (
        resolve_filter_artifact_paths(
            project_root, args.filter_id, str(args.model_architecture), args.experiment_profile
        ),
        resolve_filter_model_output_dir(
            project_root, args.filter_id, str(args.model_architecture), args.experiment_profile
        ),
    )

