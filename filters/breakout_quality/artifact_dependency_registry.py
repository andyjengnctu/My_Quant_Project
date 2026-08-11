"""Canonical breakout-quality artifact dependency and readiness registry.

This module owns the reusable model-upstream truth chain used by model research,
Strategy Compare and Multiple-seed robustness.  It intentionally does not train models
or create scientific artifacts; callers decide which work type is allowed to produce a
missing node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    get_breakout_quality_experiment_profile,
)
from core.console_report import project_relative_display_path
from filters.breakout_quality.continuous_target import (
    TARGET_MANIFEST_FILENAME,
    load_validated_continuous_target_manifest,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.dataset_readiness import collect_dataset_readiness

ARTIFACT_DATASET_CORE = "dataset_core"
ARTIFACT_CONTINUOUS_TARGET = "continuous_target"
ARTIFACT_MODEL_CHECKPOINT = "model_checkpoint"
ARTIFACT_FORWARD_SCORE = "forward_score"
ARTIFACT_SELECTION_PIT_SCORE = "selection_pit_score"
ARTIFACT_SELECTION_PIT_AUDIT = "selection_pit_audit"

PRODUCER_MODEL_TRAINING = "model_training"
PRODUCER_EXISTING_ARTIFACT = "existing_artifact"
PRODUCER_STRATEGY_COMPARE_DETERMINISTIC = "strategy_compare_deterministic_rebuild"
PRODUCER_STRATEGY_COMPARE_CHECKPOINT = "strategy_compare_checkpoint_rebuild"


@dataclass(frozen=True)
class ArtifactDependencySpec:
    artifact_type: str
    dependencies: tuple[str, ...]
    producer_work_type: str


ARTIFACT_DEPENDENCY_REGISTRY: dict[str, ArtifactDependencySpec] = {
    ARTIFACT_DATASET_CORE: ArtifactDependencySpec(
        artifact_type=ARTIFACT_DATASET_CORE,
        dependencies=(),
        producer_work_type=PRODUCER_MODEL_TRAINING,
    ),
    ARTIFACT_CONTINUOUS_TARGET: ArtifactDependencySpec(
        artifact_type=ARTIFACT_CONTINUOUS_TARGET,
        dependencies=(ARTIFACT_DATASET_CORE,),
        producer_work_type=PRODUCER_MODEL_TRAINING,
    ),
    ARTIFACT_MODEL_CHECKPOINT: ArtifactDependencySpec(
        artifact_type=ARTIFACT_MODEL_CHECKPOINT,
        dependencies=(ARTIFACT_DATASET_CORE, ARTIFACT_CONTINUOUS_TARGET),
        producer_work_type=PRODUCER_MODEL_TRAINING,
    ),
    ARTIFACT_FORWARD_SCORE: ArtifactDependencySpec(
        artifact_type=ARTIFACT_FORWARD_SCORE,
        dependencies=(ARTIFACT_MODEL_CHECKPOINT,),
        producer_work_type=PRODUCER_STRATEGY_COMPARE_DETERMINISTIC,
    ),
    ARTIFACT_SELECTION_PIT_SCORE: ArtifactDependencySpec(
        artifact_type=ARTIFACT_SELECTION_PIT_SCORE,
        dependencies=(ARTIFACT_DATASET_CORE, ARTIFACT_CONTINUOUS_TARGET),
        producer_work_type=PRODUCER_STRATEGY_COMPARE_CHECKPOINT,
    ),
    ARTIFACT_SELECTION_PIT_AUDIT: ArtifactDependencySpec(
        artifact_type=ARTIFACT_SELECTION_PIT_AUDIT,
        dependencies=(ARTIFACT_SELECTION_PIT_SCORE,),
        producer_work_type=PRODUCER_STRATEGY_COMPARE_CHECKPOINT,
    ),
}


@dataclass(frozen=True)
class ArtifactReadiness:
    artifact_type: str
    ready: bool
    status: str
    path: Path
    dependencies: tuple[str, ...]
    producer_work_type: str
    description: str

    def as_dict(self, *, project_root: str | Path) -> dict[str, Any]:
        root = Path(project_root).resolve()
        return {
            "artifact_type": self.artifact_type,
            "ready": bool(self.ready),
            "status": self.status,
            "path": project_relative_display_path(self.path, project_root=root),
            "dependencies": list(self.dependencies),
            "producer_work_type": self.producer_work_type,
            "description": self.description,
        }



def required_upstream_artifact_types(experiment_profile: str) -> tuple[str, ...]:
    """Return persistent upstream truth nodes required by the profile.

    Daily-universal rankers derive their fixed target lazily from canonical OHLCV and
    therefore do not require a persistent event-group Continuous Target artifact.
    """

    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    if str(profile.training_sample_scope) == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
        return (ARTIFACT_DATASET_CORE, ARTIFACT_CONTINUOUS_TARGET)
    return (ARTIFACT_DATASET_CORE,)


def dependency_types_for(
    artifact_type: str,
    *,
    experiment_profile: str,
) -> tuple[str, ...]:
    spec = ARTIFACT_DEPENDENCY_REGISTRY.get(str(artifact_type))
    if spec is None:
        raise KeyError(f"未註冊的breakout-quality artifact type: {artifact_type}")
    required_upstream = set(required_upstream_artifact_types(experiment_profile))
    return tuple(
        dependency
        for dependency in spec.dependencies
        if dependency != ARTIFACT_CONTINUOUS_TARGET
        or ARTIFACT_CONTINUOUS_TARGET in required_upstream
    )


def collect_model_upstream_readiness(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int = 0,
) -> tuple[ArtifactReadiness, ...]:
    """Validate canonical Dataset/Target truth with one shared implementation."""

    root = Path(project_root).resolve()
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    dataset_readiness = collect_dataset_readiness(
        root,
        filter_id=str(filter_id),
        dataset=str(dataset),
        max_tickers=int(max_tickers),
        model_architecture=str(model_architecture),
    )
    dataset_ready = bool(dataset_readiness.ready)
    dataset_status = (
        "READY"
        if dataset_ready
        else f"DATASET_{str(dataset_readiness.refresh_mode).upper()}_REQUIRED"
    )
    dataset_description = (
        "重用canonical Dataset truth"
        if dataset_ready
        else "canonical Dataset需更新：" + "；".join(dataset_readiness.reasons)
    )
    rows: list[ArtifactReadiness] = [
        ArtifactReadiness(
            artifact_type=ARTIFACT_DATASET_CORE,
            ready=dataset_ready,
            status=dataset_status,
            path=dataset_readiness.summary_path,
            dependencies=(),
            producer_work_type=(
                PRODUCER_EXISTING_ARTIFACT if dataset_ready else PRODUCER_MODEL_TRAINING
            ),
            description=dataset_description,
        )
    ]

    if (
        str(profile.training_sample_scope)
        == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    ):
        target_id = str(profile.continuous_target_id or "")
        target_manifest = (
            resolve_continuous_target_dir(root, str(filter_id), target_id=target_id)
            / TARGET_MANIFEST_FILENAME
        )
        target_error: Exception | None = None
        target_ready = False
        if dataset_ready and dataset_readiness.summary is not None:
            try:
                load_validated_continuous_target_manifest(
                    root,
                    str(filter_id),
                    target_id=target_id,
                    expected_dataset_policy=dataset_readiness.summary.get("policy"),
                    expected_dataset_artifacts=dataset_readiness.summary.get("dataset_artifacts"),
                )
                target_ready = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                target_error = exc
        target_status = "READY" if target_ready else "CONTINUOUS_TARGET_MISSING_OR_INVALID"
        if target_ready:
            target_description = "重用canonical Continuous Target truth"
        elif not dataset_ready:
            target_description = "canonical Dataset未就緒，Continuous Target不可驗證／建立"
        elif target_error is not None:
            target_description = (
                "缺少或無效的canonical Continuous Target："
                f"{type(target_error).__name__}: {target_error}"
            )
        else:
            target_description = "缺少或無效的canonical Continuous Target"
        rows.append(
            ArtifactReadiness(
                artifact_type=ARTIFACT_CONTINUOUS_TARGET,
                ready=target_ready,
                status=target_status,
                path=target_manifest,
                dependencies=(ARTIFACT_DATASET_CORE,),
                producer_work_type=(
                    PRODUCER_EXISTING_ARTIFACT
                    if target_ready
                    else PRODUCER_MODEL_TRAINING
                ),
                description=target_description,
            )
        )
    return tuple(rows)


def model_upstream_prerequisite_blockers(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int = 0,
) -> tuple[str, ...]:
    """Return user-facing blockers from the shared upstream readiness snapshot."""

    root = Path(project_root).resolve()
    blockers: list[str] = []
    for item in collect_model_upstream_readiness(
        root,
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        dataset=str(dataset),
        max_tickers=int(max_tickers),
    ):
        if item.ready:
            continue
        display_path = project_relative_display_path(item.path, project_root=root)
        blockers.append(f"{item.description}: {display_path}")
    return tuple(blockers)


__all__ = [
    "ARTIFACT_CONTINUOUS_TARGET",
    "ARTIFACT_DATASET_CORE",
    "ARTIFACT_DEPENDENCY_REGISTRY",
    "ARTIFACT_FORWARD_SCORE",
    "ARTIFACT_MODEL_CHECKPOINT",
    "ARTIFACT_SELECTION_PIT_AUDIT",
    "ARTIFACT_SELECTION_PIT_SCORE",
    "ArtifactDependencySpec",
    "ArtifactReadiness",
    "PRODUCER_EXISTING_ARTIFACT",
    "PRODUCER_MODEL_TRAINING",
    "PRODUCER_STRATEGY_COMPARE_CHECKPOINT",
    "PRODUCER_STRATEGY_COMPARE_DETERMINISTIC",
    "collect_model_upstream_readiness",
    "dependency_types_for",
    "model_upstream_prerequisite_blockers",
    "required_upstream_artifact_types",
]
