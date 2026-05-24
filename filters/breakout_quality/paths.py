"""Paths for breakout quality filter artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from core.model_paths import resolve_models_dir
from core.output_paths import ensure_output_dir
from filters.breakout_quality.contract import FILTER_FAMILY


def resolve_filter_model_dir(project_root: str | os.PathLike[str], filter_id: str) -> Path:
    resolved_filter_id = str(filter_id).strip()
    if not resolved_filter_id:
        raise ValueError("filter_id 不可為空白")
    return Path(resolve_models_dir(str(project_root))) / "filters" / FILTER_FAMILY / resolved_filter_id


def resolve_filter_output_dir(project_root: str | os.PathLike[str], filter_id: str | None = None) -> Path:
    base_dir = Path(ensure_output_dir(project_root, "filters")) / FILTER_FAMILY
    if filter_id is None or str(filter_id).strip() == "":
        return base_dir
    return base_dir / str(filter_id).strip()


def ensure_filter_output_dir(project_root: str | os.PathLike[str], filter_id: str | None = None) -> Path:
    out_dir = resolve_filter_output_dir(project_root, filter_id=filter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


__all__ = ["ensure_filter_output_dir", "resolve_filter_model_dir", "resolve_filter_output_dir"]
