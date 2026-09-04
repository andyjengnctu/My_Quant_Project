"""Canonical artifact capability registry for PIT-safe predicted contexts.

Scientific target semantics remain owned by ``config.breakout_quality``.  This module
owns only the reusable persistent-artifact contract shared by upside/safety context
producers and consumers: storage owner, filenames, columns, Stage-1 provenance and
validation.  Adding another predicted-context source should register one capability
here instead of branching every consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
)
from core.breakout_quality_registry import (
    PREDICTED_SAFETY_CONTEXT_OWNER_ARCHITECTURE,
    PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
    PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION,
    PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
    PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
    PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
    PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
    PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION,
    PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE,
    PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE,
    PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID,
    PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED,
    get_predicted_safety_context_contract,
    get_predicted_upside_context_contract,
)
from filters.breakout_quality.paths import resolve_filter_model_dir


@dataclass(frozen=True)
class PredictedContextArtifactSpec:
    source: str
    artifact_type: str
    builder_type: str
    artifact_subdir: str
    filename: str
    manifest_filename: str
    schema_version: int
    predicted_score_column: str
    context_column: str
    context_name: str
    manifest_key: str
    target_contract_kind: str
    stage1_profile: str
    stage1_architecture: str
    stage1_research_id: str
    stage1_seed: int
    contract_provider: Callable[[], dict[str, Any]]
    owner_architecture: str | None = None
    owner_profile: str | None = None
    missing_artifact_message: str = ""
    invalid_status: str = "PREDICTED_CONTEXT_MISSING_OR_INVALID"
    ready_description: str = "重用canonical PIT-safe predicted context"
    dataset_not_ready_description: str = "canonical Dataset未就緒，predicted context不可建立"
    invalid_description_prefix: str = "缺少或無效的canonical PIT-safe predicted context："
    producer_label: str = "predicted context"

    def resolve_owner_architecture(self, requested: str) -> str:
        return str(self.owner_architecture or requested)

    def resolve_owner_profile(self, requested: str) -> str:
        return str(self.owner_profile or requested)

    def contract(self) -> dict[str, Any]:
        return dict(self.contract_provider())


_PREDICTED_CONTEXT_ARTIFACT_SPECS: dict[str, PredictedContextArtifactSpec] = {
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE: PredictedContextArtifactSpec(
        source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
        artifact_type="predicted_upside_context",
        builder_type="breakout_quality_predicted_upside_context",
        artifact_subdir="predicted_upside_context",
        filename="predicted_upside_context.csv.gz",
        manifest_filename="manifest.json",
        schema_version=PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION,
        predicted_score_column="predicted_upside_score",
        context_column="predicted_upside_percentile",
        context_name="upside",
        manifest_key="predicted_upside_context_manifest",
        target_contract_kind="predicted_upside_context",
        stage1_profile=PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE,
        stage1_architecture=PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE,
        stage1_research_id=PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID,
        stage1_seed=PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED,
        contract_provider=get_predicted_upside_context_contract,
        missing_artifact_message="MR-13AC缺少PIT-safe predicted-upside context artifact",
        invalid_status="PREDICTED_UPSIDE_CONTEXT_MISSING_OR_INVALID",
        ready_description="重用MR-13AC PIT-safe predicted-upside context",
        dataset_not_ready_description="canonical Dataset未就緒，predicted-upside context不可建立",
        invalid_description_prefix="缺少或無效的MR-13AC PIT-safe predicted-upside context：",
        producer_label="MR-13AC predicted-upside",
    ),
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY: PredictedContextArtifactSpec(
        source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        artifact_type="predicted_safety_context",
        builder_type="breakout_quality_predicted_safety_context",
        artifact_subdir="predicted_safety_context",
        filename="predicted_safety_context.csv.gz",
        manifest_filename="manifest.json",
        schema_version=PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION,
        predicted_score_column="predicted_safety_score",
        context_column="predicted_safety_percentile",
        context_name="safety",
        manifest_key="predicted_safety_context_manifest",
        target_contract_kind="predicted_safety_context",
        stage1_profile=PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        stage1_architecture=PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        stage1_research_id=PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        stage1_seed=PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        contract_provider=get_predicted_safety_context_contract,
        owner_architecture=PREDICTED_SAFETY_CONTEXT_OWNER_ARCHITECTURE,
        owner_profile=PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
        missing_artifact_message="MR-13AD缺少PIT-safe predicted-safety context artifact",
        invalid_status="PREDICTED_SAFETY_CONTEXT_MISSING_OR_INVALID",
        ready_description="重用canonical PIT-safe predicted-safety context",
        dataset_not_ready_description="canonical Dataset未就緒，predicted-safety context不可建立",
        invalid_description_prefix="缺少或無效的canonical PIT-safe predicted-safety context：",
        producer_label="MR-13AD/MR-13AE predicted-safety",
    ),
}


def predicted_context_artifact_specs() -> tuple[PredictedContextArtifactSpec, ...]:
    return tuple(_PREDICTED_CONTEXT_ARTIFACT_SPECS.values())


def get_predicted_context_artifact_spec(source: str) -> PredictedContextArtifactSpec:
    try:
        return _PREDICTED_CONTEXT_ARTIFACT_SPECS[str(source)]
    except KeyError as exc:
        raise ValueError(f"不支援的predicted context capability: {source!r}") from exc


def maybe_predicted_context_artifact_spec(
    source: str,
) -> PredictedContextArtifactSpec | None:
    return _PREDICTED_CONTEXT_ARTIFACT_SPECS.get(str(source))


def resolve_predicted_context_dir(
    source: str,
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> Path:
    spec = get_predicted_context_artifact_spec(source)
    return (
        resolve_filter_model_dir(
            project_root,
            filter_id,
            spec.resolve_owner_architecture(model_architecture),
            spec.resolve_owner_profile(experiment_profile),
        )
        / "upstream"
        / spec.artifact_subdir
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_predicted_context(
    source: str,
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate one registered predicted-context artifact."""

    spec = get_predicted_context_artifact_spec(source)
    directory = resolve_predicted_context_dir(
        source,
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    score_path = directory / spec.filename
    manifest_path = directory / spec.manifest_filename
    label = f"predicted-{spec.context_name} context"
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(spec.missing_artifact_message)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != int(spec.schema_version):
        raise ValueError(f"{label} schema不一致")
    if str(manifest.get("filter_id") or "") != str(filter_id):
        raise ValueError(f"{label} filter_id不一致")
    expected_architecture = spec.resolve_owner_architecture(model_architecture)
    expected_profile = spec.resolve_owner_profile(experiment_profile)
    if str(manifest.get("model_architecture") or "") != expected_architecture:
        suffix = " canonical owner architecture" if spec.owner_architecture else " model_architecture"
        raise ValueError(f"{label}{suffix}不一致")
    if str(manifest.get("experiment_profile") or "") != expected_profile:
        suffix = " canonical owner profile" if spec.owner_profile else " experiment_profile"
        raise ValueError(f"{label}{suffix}不一致")
    expected = spec.contract()
    if dict(manifest.get("contract") or {}) != expected:
        raise ValueError(f"{label} scientific contract不一致")
    if expected_dataset_policy is not None and manifest.get("dataset_policy") != expected_dataset_policy:
        raise ValueError(f"{label} dataset_policy與目前Dataset不一致")
    if expected_dataset_artifacts is not None and manifest.get("dataset_artifact_source") != expected_dataset_artifacts:
        raise ValueError(f"{label} Dataset artifact identity已改變")

    artifact = dict(manifest.get("artifact") or {})
    if str(artifact.get("filename") or "") != score_path.name:
        raise ValueError(f"{label} filename不一致")
    if int(artifact.get("size_bytes", -1)) != int(score_path.stat().st_size):
        raise ValueError(f"{label} size不一致")
    if str(artifact.get("sha256") or "").lower() != _sha256(score_path).lower():
        raise ValueError(f"{label} SHA256不一致")

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
            raise ValueError(f"{label} source_stage1缺少{key}")
        if str(record.get("filename") or "") != path.name:
            raise ValueError(f"{label} source_stage1 {key} filename不一致")
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"{label} source_stage1 {key} size不一致")
        if str(record.get("sha256") or "").lower() != _sha256(path).lower():
            raise ValueError(f"{label} source_stage1 {key} SHA256不一致")

    frame = pd.read_csv(
        score_path, encoding="utf-8-sig", dtype={"ticker": "string"}, low_memory=False
    )
    required = {
        "ticker",
        "date",
        spec.predicted_score_column,
        spec.context_column,
        "context_phase",
        "fold_id",
        "model_information_cutoff",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少欄位: {missing}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["model_information_cutoff"] = pd.to_datetime(
        frame["model_information_cutoff"], errors="raise"
    ).dt.normalize()
    for column in (spec.predicted_score_column, spec.context_column):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(f"{label} {column}必須finite且位於[0,1]")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError(f"{label} ticker/date不得重複")

    selection = frame["context_phase"].astype(str).eq("selection_crossfit")
    if bool((frame.loc[selection, "model_information_cutoff"] >= frame.loc[selection, "date"]).any()):
        raise ValueError(f"selection {label} information cutoff必須早於score date")
    forward = frame["context_phase"].astype(str).eq("forward_fixed_pre_oos")
    if bool(forward.any()):
        cutoffs = frame.loc[forward, "model_information_cutoff"].drop_duplicates()
        folds = frame.loc[forward, "fold_id"].astype(str).drop_duplicates()
        if len(cutoffs) != 1 or len(folds) != 1:
            raise ValueError(f"Forward {label}必須使用單一pre-OOS fit")
        if not bool((frame.loc[forward, "model_information_cutoff"] < frame.loc[forward, "date"]).all()):
            raise ValueError(f"Forward {label} cutoff必須早於所有OOS score date")
    return frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True), manifest


__all__ = [
    "PredictedContextArtifactSpec",
    "get_predicted_context_artifact_spec",
    "load_validated_predicted_context",
    "maybe_predicted_context_artifact_spec",
    "predicted_context_artifact_specs",
    "resolve_predicted_context_dir",
]
