"""Canonical read-only contract for frozen-ranker Expected-R calibration artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from filters.breakout_quality.calibration_lookup_common import (
    CalibrationLookupContract,
    load_calibration_lookup,
    lookup_calibration_value,
    resolve_calibration_paths,
    validate_calibration_artifact,
)

EXPECTED_R_CALIBRATION_SCHEMA_VERSION = 1
EXPECTED_R_CALIBRATION_METHOD = "daily_score_percentile_nonnegative_affine_v1"
EXPECTED_R_LOOKUP_FILENAME = "expected_r_lookup.csv.gz"
EXPECTED_R_MANIFEST_FILENAME = "manifest.json"
EXPECTED_R_REPORT_FILENAME = "report.md"
EXPECTED_R_REQUIRED_COLUMNS = (
    "ticker",
    "date",
    "group_index",
    "breakout_quality_score",
    "daily_score_percentile",
    "expected_r",
    "calibration_cutoff_exclusive",
)

_EXPECTED_R_CONTRACT = CalibrationLookupContract(
    label="Expected-R",
    artifact_dir_name="expected_r_calibration",
    schema_version=EXPECTED_R_CALIBRATION_SCHEMA_VERSION,
    method=EXPECTED_R_CALIBRATION_METHOD,
    lookup_filename=EXPECTED_R_LOOKUP_FILENAME,
    required_columns=EXPECTED_R_REQUIRED_COLUMNS,
    value_column="expected_r",
    available_key="expected_r_available",
    unavailable_reason_key="expected_r_unavailable_reason",
    cutoff_result_key="expected_r_calibration_cutoff_exclusive",
    attach_date_range_attrs=True,
)


def _validate_phase(phase_id: str) -> str:
    phase = str(phase_id).strip()
    if phase not in {"selection_pit", "forward_oos"}:
        raise ValueError(f"Expected-R calibration phase不支援: {phase!r}")
    return phase


def resolve_expected_r_calibration_dir(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
) -> Path:
    return resolve_expected_r_calibration_paths(
        project_root, filter_id=filter_id, model_architecture=model_architecture,
        experiment_profile=experiment_profile, phase_id=phase_id,
    )["dir"]


def resolve_expected_r_calibration_paths(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
) -> dict[str, Path]:
    return resolve_calibration_paths(
        _EXPECTED_R_CONTRACT,
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id=_validate_phase(phase_id),
    )


def validate_expected_r_calibration_artifact(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
    expected_source_score_sha256: str | None = None,
    expected_fit_score_sha256: str | None = None,
    expected_selection_runtime_start_date: str | None = None,
    expected_forward_frozen_cutoff_exclusive: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    phase = _validate_phase(phase_id)
    paths = resolve_expected_r_calibration_paths(
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id=phase,
    )
    return validate_calibration_artifact(
        _EXPECTED_R_CONTRACT,
        paths=paths,
        expected_identity={
            "filter_id": str(filter_id),
            "model_architecture": str(model_architecture),
            "experiment_profile": str(experiment_profile),
            "phase_id": phase,
        },
        expected_source_score_sha256=expected_source_score_sha256,
        expected_fit_score_sha256=expected_fit_score_sha256,
        expected_selection_runtime_start_date=expected_selection_runtime_start_date,
        expected_forward_frozen_cutoff_exclusive=expected_forward_frozen_cutoff_exclusive,
    )


def load_expected_r_lookup(path: str | Path) -> pd.DataFrame:
    return load_calibration_lookup(_EXPECTED_R_CONTRACT, path)


def lookup_expected_r(
    *,
    lookup_path: str | Path,
    ticker: str,
    score_date,
    expected_score: float | None,
) -> dict[str, Any]:
    return lookup_calibration_value(
        _EXPECTED_R_CONTRACT,
        lookup_path=lookup_path,
        ticker=ticker,
        score_date=score_date,
        expected_score=expected_score,
    )


__all__ = [
    "EXPECTED_R_CALIBRATION_SCHEMA_VERSION",
    "EXPECTED_R_CALIBRATION_METHOD",
    "EXPECTED_R_LOOKUP_FILENAME",
    "resolve_expected_r_calibration_dir",
    "resolve_expected_r_calibration_paths",
    "validate_expected_r_calibration_artifact",
    "load_expected_r_lookup",
    "lookup_expected_r",
]
