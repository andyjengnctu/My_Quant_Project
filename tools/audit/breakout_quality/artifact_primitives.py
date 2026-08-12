"""Artifact-resolution primitives shared by Breakout Quality audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.breakout_quality import BREAKOUT_QUALITY_MODEL_ARCHITECTURE
from filters.breakout_quality.continuous_target import (
    TARGET_ADVERSE_RETURN_FILENAME,
    TARGET_FAVORABLE_RETURN_FILENAME,
    TARGET_OPPORTUNITY_BAR_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_RISK_BREACH_BAR_FILENAME,
    TARGET_VALID_MASK_FILENAME,
)
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.workflow_io import PROJECT_ROOT
from tools.audit.primitives import sha256_file


def continuous_target_component_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "target_raw_r": target_dir / TARGET_RAW_FILENAME,
        "favorable_return": target_dir / TARGET_FAVORABLE_RETURN_FILENAME,
        "adverse_return_to_peak": target_dir / TARGET_ADVERSE_RETURN_FILENAME,
        "opportunity_bar": target_dir / TARGET_OPPORTUNITY_BAR_FILENAME,
        "first_risk_breach_bar": target_dir / TARGET_RISK_BREACH_BAR_FILENAME,
        "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
    }


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


__all__ = ["continuous_target_component_paths", "read_json", "resolve_ranker_dir", "sha256_file"]
