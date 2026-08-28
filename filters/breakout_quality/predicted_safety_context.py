"""PIT-safe predicted-safety context and conditional MFE target contract for MR-13AD."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
    PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION,
    PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
    PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
    PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
    PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
    get_predicted_safety_context_contract,
)
from filters.breakout_quality.conditional_mfe_safety import (
    build_same_date_residual_percentile,
)
from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)
from filters.breakout_quality.paths import resolve_filter_model_dir

PREDICTED_SAFETY_CONTEXT_FILENAME = "predicted_safety_context.csv.gz"
PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME = "manifest.json"
STAGE1_PROFILE = PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE
STAGE1_ARCHITECTURE = PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE
STAGE1_RESEARCH_ID = PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID
STAGE1_SEED = PREDICTED_SAFETY_CONTEXT_STAGE1_SEED
CONTEXT_COLUMN = "predicted_safety_percentile"


@dataclass(frozen=True)
class PredictedSafetyConditionalMfeTargets:
    pure_mfe_percentile: np.ndarray
    residual: np.ndarray
    residual_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.asarray(self.residual_percentile, dtype=np.float32)


def resolve_predicted_safety_context_dir(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> Path:
    return (
        resolve_filter_model_dir(
            project_root,
            filter_id,
            model_architecture,
            experiment_profile,
        )
        / "upstream"
        / "predicted_safety_context"
    )


def predicted_safety_context_contract() -> dict[str, Any]:
    return get_predicted_safety_context_contract()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_predicted_safety_context(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = resolve_predicted_safety_context_dir(
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    score_path = directory / PREDICTED_SAFETY_CONTEXT_FILENAME
    manifest_path = directory / PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("MR-13AD缺少PIT-safe predicted-safety context artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION:
        raise ValueError("predicted-safety context schema不一致")
    if str(manifest.get("filter_id") or "") != str(filter_id):
        raise ValueError("predicted-safety context filter_id不一致")
    if str(manifest.get("model_architecture") or "") != str(model_architecture):
        raise ValueError("predicted-safety context model_architecture不一致")
    if str(manifest.get("experiment_profile") or "") != str(experiment_profile):
        raise ValueError("predicted-safety context experiment_profile不一致")
    expected = predicted_safety_context_contract()
    if dict(manifest.get("contract") or {}) != expected:
        raise ValueError("predicted-safety context scientific contract不一致")
    if expected_dataset_policy is not None and manifest.get("dataset_policy") != expected_dataset_policy:
        raise ValueError("predicted-safety context dataset_policy與目前Dataset不一致")
    if expected_dataset_artifacts is not None and manifest.get("dataset_artifact_source") != expected_dataset_artifacts:
        raise ValueError("predicted-safety context Dataset artifact identity已改變")
    artifact = dict(manifest.get("artifact") or {})
    if str(artifact.get("filename") or "") != score_path.name:
        raise ValueError("predicted-safety context filename不一致")
    if int(artifact.get("size_bytes", -1)) != int(score_path.stat().st_size):
        raise ValueError("predicted-safety context size不一致")
    if str(artifact.get("sha256") or "").lower() != _sha256(score_path).lower():
        raise ValueError("predicted-safety context SHA256不一致")

    source_stage1 = dict(manifest.get("source_stage1") or {})
    source_paths = {
        "selection_score": directory / "stage1_selection_crossfit" / "selection_point_in_time_scores.csv",
        "selection_manifest": directory / "stage1_selection_crossfit" / "selection_point_in_time_manifest.json",
        "forward_score": directory / "stage1_forward_fixed" / "selection_point_in_time_scores.csv",
        "forward_manifest": directory / "stage1_forward_fixed" / "selection_point_in_time_manifest.json",
    }
    for key, path in source_paths.items():
        record = dict(source_stage1.get(key) or {})
        if not path.is_file():
            raise ValueError(f"predicted-safety context source_stage1缺少{key}")
        if str(record.get("filename") or "") != path.name:
            raise ValueError(f"predicted-safety context source_stage1 {key} filename不一致")
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"predicted-safety context source_stage1 {key} size不一致")
        if str(record.get("sha256") or "").lower() != _sha256(path).lower():
            raise ValueError(f"predicted-safety context source_stage1 {key} SHA256不一致")
    frame = pd.read_csv(
        score_path, encoding="utf-8-sig", dtype={"ticker": "string"}, low_memory=False
    )
    required = {
        "ticker", "date", "predicted_safety_score", CONTEXT_COLUMN,
        "context_phase", "fold_id", "model_information_cutoff",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"predicted-safety context缺少欄位: {missing}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["model_information_cutoff"] = pd.to_datetime(
        frame["model_information_cutoff"], errors="raise"
    ).dt.normalize()
    for column in ("predicted_safety_score", CONTEXT_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(f"predicted-safety context {column}必須finite且位於[0,1]")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("predicted-safety context ticker/date不得重複")
    selection = frame["context_phase"].astype(str).eq("selection_crossfit")
    if bool((frame.loc[selection, "model_information_cutoff"] >= frame.loc[selection, "date"]).any()):
        raise ValueError("selection predicted-safety context information cutoff必須早於score date")
    forward = frame["context_phase"].astype(str).eq("forward_fixed_pre_oos")
    if bool(forward.any()):
        cutoffs = frame.loc[forward, "model_information_cutoff"].drop_duplicates()
        folds = frame.loc[forward, "fold_id"].astype(str).drop_duplicates()
        if len(cutoffs) != 1 or len(folds) != 1:
            raise ValueError("Forward predicted-safety context必須使用單一pre-OOS fit")
        if not bool((frame.loc[forward, "model_information_cutoff"] < frame.loc[forward, "date"]).all()):
            raise ValueError("Forward predicted-safety context cutoff必須早於所有OOS score date")
    return frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True), manifest


def build_predicted_safety_conditional_mfe_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    predicted_safety_percentile: np.ndarray,
) -> PredictedSafetyConditionalMfeTargets:
    required = {"date", "target_favorable_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"MR-13AD target缺少canonical欄位: {missing}")
    valid = np.asarray(valid_mask, dtype=bool)
    context = np.asarray(predicted_safety_percentile, dtype=np.float32)
    if valid.shape != context.shape or len(valid) != len(group_table):
        raise ValueError("MR-13AD target/context shape不一致")
    favorable = pd.to_numeric(
        group_table["target_favorable_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    component_valid = valid & np.isfinite(favorable) & np.isfinite(context)
    if not np.array_equal(component_valid, valid):
        raise ValueError("MR-13AD valid rows必須同時有finite Pure-MFE與PIT-safe safety context")
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    pure_mfe = build_same_date_percentile_targets(favorable, valid, dates)
    residual, residual_percentile = build_same_date_residual_percentile(
        context,
        pure_mfe,
        valid,
        dates,
    )
    return PredictedSafetyConditionalMfeTargets(
        pure_mfe_percentile=np.asarray(pure_mfe, dtype=np.float32),
        residual=residual,
        residual_percentile=residual_percentile,
    )


__all__ = [
    "CONTEXT_COLUMN",
    "PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID",
    "PREDICTED_SAFETY_CONTEXT_FILENAME",
    "PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME",
    "STAGE1_ARCHITECTURE",
    "STAGE1_PROFILE",
    "STAGE1_RESEARCH_ID",
    "STAGE1_SEED",
    "PredictedSafetyConditionalMfeTargets",
    "build_predicted_safety_conditional_mfe_targets",
    "load_validated_predicted_safety_context",
    "predicted_safety_context_contract",
    "resolve_predicted_safety_context_dir",
]
