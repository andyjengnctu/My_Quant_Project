"""PIT-safe predicted-upside context and conditional low-adverse target contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filters.breakout_quality.conditional_mfe_safety import (
    build_same_date_residual_percentile,
)
from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)
from filters.breakout_quality.paths import resolve_filter_model_dir

PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID = (
    "daily_predicted_upside_conditional_low_adverse_v1"
)
PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION = 1
PREDICTED_UPSIDE_CONTEXT_FILENAME = "predicted_upside_context.csv.gz"
PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME = "manifest.json"
STAGE1_PROFILE = "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise"
STAGE1_ARCHITECTURE = "inception_time_v1"
STAGE1_RESEARCH_ID = "MR-13K"
STAGE1_SEED = 42
CONTEXT_COLUMN = "predicted_upside_percentile"


@dataclass(frozen=True)
class PredictedUpsideConditionalSafetyTargets:
    low_adverse_percentile: np.ndarray
    residual: np.ndarray
    residual_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.asarray(self.residual_percentile, dtype=np.float32)


def resolve_predicted_upside_context_dir(
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
        / "predicted_upside_context"
    )


def predicted_upside_context_contract() -> dict[str, Any]:
    return {
        "schema_version": PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION,
        "stage1_profile": STAGE1_PROFILE,
        "stage1_architecture": STAGE1_ARCHITECTURE,
        "stage1_research_id": STAGE1_RESEARCH_ID,
        "stage1_seed": STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_pure_mfe",
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "same_date_low_adverse_percentile",
        "stage2_target": "same_date_percentile_of_low_adverse_residual_given_predicted_upside_percentile",
        "stage2_context_used_as_input": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_predicted_upside_context(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = resolve_predicted_upside_context_dir(
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    score_path = directory / PREDICTED_UPSIDE_CONTEXT_FILENAME
    manifest_path = directory / PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("MR-13AC缺少PIT-safe predicted-upside context artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION:
        raise ValueError("predicted-upside context schema不一致")
    if str(manifest.get("filter_id") or "") != str(filter_id):
        raise ValueError("predicted-upside context filter_id不一致")
    if str(manifest.get("model_architecture") or "") != str(model_architecture):
        raise ValueError("predicted-upside context model_architecture不一致")
    if str(manifest.get("experiment_profile") or "") != str(experiment_profile):
        raise ValueError("predicted-upside context experiment_profile不一致")
    expected = predicted_upside_context_contract()
    if dict(manifest.get("contract") or {}) != expected:
        raise ValueError("predicted-upside context scientific contract不一致")
    if expected_dataset_policy is not None and manifest.get("dataset_policy") != expected_dataset_policy:
        raise ValueError("predicted-upside context dataset_policy與目前Dataset不一致")
    if expected_dataset_artifacts is not None and manifest.get("dataset_artifact_source") != expected_dataset_artifacts:
        raise ValueError("predicted-upside context Dataset artifact identity已改變")
    artifact = dict(manifest.get("artifact") or {})
    if str(artifact.get("filename") or "") != score_path.name:
        raise ValueError("predicted-upside context filename不一致")
    if int(artifact.get("size_bytes", -1)) != int(score_path.stat().st_size):
        raise ValueError("predicted-upside context size不一致")
    if str(artifact.get("sha256") or "").lower() != _sha256(score_path).lower():
        raise ValueError("predicted-upside context SHA256不一致")

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
            raise ValueError(f"predicted-upside context source_stage1缺少{key}")
        if str(record.get("filename") or "") != path.name:
            raise ValueError(f"predicted-upside context source_stage1 {key} filename不一致")
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"predicted-upside context source_stage1 {key} size不一致")
        if str(record.get("sha256") or "").lower() != _sha256(path).lower():
            raise ValueError(f"predicted-upside context source_stage1 {key} SHA256不一致")
    frame = pd.read_csv(
        score_path, encoding="utf-8-sig", dtype={"ticker": "string"}, low_memory=False
    )
    required = {
        "ticker", "date", "predicted_upside_score", CONTEXT_COLUMN,
        "context_phase", "fold_id", "model_information_cutoff",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"predicted-upside context缺少欄位: {missing}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["model_information_cutoff"] = pd.to_datetime(
        frame["model_information_cutoff"], errors="raise"
    ).dt.normalize()
    for column in ("predicted_upside_score", CONTEXT_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(f"predicted-upside context {column}必須finite且位於[0,1]")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("predicted-upside context ticker/date不得重複")
    selection = frame["context_phase"].astype(str).eq("selection_crossfit")
    if bool((frame.loc[selection, "model_information_cutoff"] >= frame.loc[selection, "date"]).any()):
        raise ValueError("selection predicted-upside context information cutoff必須早於score date")
    forward = frame["context_phase"].astype(str).eq("forward_fixed_pre_oos")
    if bool(forward.any()):
        cutoffs = frame.loc[forward, "model_information_cutoff"].drop_duplicates()
        folds = frame.loc[forward, "fold_id"].astype(str).drop_duplicates()
        if len(cutoffs) != 1 or len(folds) != 1:
            raise ValueError("Forward predicted-upside context必須使用單一pre-OOS fit")
        if not bool((frame.loc[forward, "model_information_cutoff"] < frame.loc[forward, "date"]).all()):
            raise ValueError("Forward predicted-upside context cutoff必須早於所有OOS score date")
    return frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True), manifest


def build_predicted_upside_conditional_low_adverse_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    predicted_upside_percentile: np.ndarray,
) -> PredictedUpsideConditionalSafetyTargets:
    required = {"date", "target_adverse_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"MR-13AC target缺少canonical欄位: {missing}")
    valid = np.asarray(valid_mask, dtype=bool)
    context = np.asarray(predicted_upside_percentile, dtype=np.float32)
    if valid.shape != context.shape or len(valid) != len(group_table):
        raise ValueError("MR-13AC target/context shape不一致")
    adverse = pd.to_numeric(group_table["target_adverse_r"], errors="coerce").to_numpy(dtype=np.float64)
    component_valid = valid & np.isfinite(adverse) & np.isfinite(context)
    if not np.array_equal(component_valid, valid):
        raise ValueError("MR-13AC valid rows必須同時有finite adverse與PIT-safe upside context")
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    low_adverse = build_same_date_percentile_targets(-adverse, valid, dates)
    residual, residual_percentile = build_same_date_residual_percentile(
        context,
        low_adverse,
        valid,
        dates,
    )
    return PredictedUpsideConditionalSafetyTargets(
        low_adverse_percentile=np.asarray(low_adverse, dtype=np.float32),
        residual=residual,
        residual_percentile=residual_percentile,
    )


__all__ = [
    "CONTEXT_COLUMN",
    "PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID",
    "PREDICTED_UPSIDE_CONTEXT_FILENAME",
    "PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME",
    "STAGE1_ARCHITECTURE",
    "STAGE1_PROFILE",
    "STAGE1_RESEARCH_ID",
    "STAGE1_SEED",
    "PredictedUpsideConditionalSafetyTargets",
    "build_predicted_upside_conditional_low_adverse_targets",
    "load_validated_predicted_upside_context",
    "predicted_upside_context_contract",
    "resolve_predicted_upside_context_dir",
]
