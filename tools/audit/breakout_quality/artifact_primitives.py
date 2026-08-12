"""Artifact-resolution primitives shared by Breakout Quality audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.breakout_quality import BREAKOUT_QUALITY_MODEL_ARCHITECTURE
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.workflow_io import PROJECT_ROOT
from tools.audit.primitives import sha256_file


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取JSON失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {path}")
    return payload


def resolve_ranker_dir(filter_id: str, ranker_profile: str) -> Path:
    return resolve_filter_model_output_dir(PROJECT_ROOT, filter_id, BREAKOUT_QUALITY_MODEL_ARCHITECTURE, ranker_profile)


__all__ = ["read_json", "resolve_ranker_dir", "sha256_file"]
