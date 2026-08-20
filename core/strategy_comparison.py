"""通用策略績效比較設定、前置工件計畫與驗證契約。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


STRATEGY_DL_RUNTIME_MODE_HARD_FILTER = 'hard-filter'
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY = 'resource-aware-binary'
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET = 'resource-aware-binary-basket'
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS = 'resource-aware-continuous'
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING = (
    'resource-aware-continuous-capital-preserving'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL = (
    'resource-aware-continuous-max-dl'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT = (
    'resource-aware-continuous-max-dl-feasible-ascent'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD = (
    'resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT = (
    'resource-aware-continuous-expected-pnl-feasible-ascent'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT = (
    'resource-aware-continuous-excess-alpha-feasible-ascent'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT = (
    'resource-aware-continuous-excess-alpha-no-r0-feasible-ascent'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-excess-alpha-constrained-optimal'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-constrained-optimal'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-safety-constrained-optimal'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-residual-safety-constrained-optimal'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-no-r0-constrained-optimal'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-capital-no-r0-constrained-optimal'
)
STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-capital-pareto-no-r0-constrained-optimal'
)
SUPPORTED_STRATEGY_DL_RUNTIME_MODES = (
    STRATEGY_DL_RUNTIME_MODE_HARD_FILTER,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
)


@dataclass(frozen=True)
class StrategyArtifactBuilder:
    enabled: bool
    builder_type: str
    options: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "builder_type": self.builder_type,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class StrategyPreparationPolicy:
    auto_prepare: bool
    reuse_ready_artifacts: bool
    rebuild_stale_artifacts: bool
    resume_parameter_training: bool
    require_confirmation: bool
    reuse_completed_results: bool = True
    reuse_shared_baseline: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "auto_prepare": bool(self.auto_prepare),
            "reuse_ready_artifacts": bool(self.reuse_ready_artifacts),
            "rebuild_stale_artifacts": bool(self.rebuild_stale_artifacts),
            "resume_parameter_training": bool(self.resume_parameter_training),
            "require_confirmation": bool(self.require_confirmation),
            "reuse_completed_results": bool(self.reuse_completed_results),
            "reuse_shared_baseline": bool(self.reuse_shared_baseline),
        }


@dataclass(frozen=True)
class StrategyParameterSource:
    source_id: str
    path_template: str | None
    description: str
    identity_manifest_path: str | None = None
    trained_with_dl_id: str | None = None
    artifact_contract: Mapping[str, Any] | None = None
    builder: StrategyArtifactBuilder | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path_template": self.path_template,
            "description": self.description,
            "identity_manifest_path": self.identity_manifest_path,
            "trained_with_dl_id": self.trained_with_dl_id,
            "artifact_contract": (
                None if self.artifact_contract is None else dict(self.artifact_contract)
            ),
            "builder": None if self.builder is None else self.builder.as_dict(),
        }


@dataclass(frozen=True)
class StrategyDLSource:
    dl_id: str
    filter_id: str
    model_architecture: str
    experiment_profile: str
    threshold: float | None
    description: str
    score_source: str = "canonical_runtime"
    forward_scores_builder: StrategyArtifactBuilder | None = None
    point_in_time_score_start_date: str | None = None
    point_in_time_score_end_date: str | None = None
    point_in_time_fold_months: int | None = None
    point_in_time_fold_anchor_date: str | None = None
    point_in_time_single_score_block: bool = False
    point_in_time_dirname: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dl_id": self.dl_id,
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "threshold": None if self.threshold is None else float(self.threshold),
            "description": self.description,
            "score_source": self.score_source,
            "forward_scores_builder": (
                None
                if self.forward_scores_builder is None
                else self.forward_scores_builder.as_dict()
            ),
            "point_in_time_score_start_date": self.point_in_time_score_start_date,
            "point_in_time_score_end_date": self.point_in_time_score_end_date,
            "point_in_time_fold_months": (
                None
                if self.point_in_time_fold_months is None
                else int(self.point_in_time_fold_months)
            ),
            "point_in_time_fold_anchor_date": self.point_in_time_fold_anchor_date,
            "point_in_time_single_score_block": bool(self.point_in_time_single_score_block),
            "point_in_time_dirname": self.point_in_time_dirname,
        }


@dataclass(frozen=True)
class StrategyComparisonArm:
    arm_id: str
    enabled: bool
    name: str
    description: str
    param_source: str
    rule_policy: str
    dl_enabled: bool
    dl_id: str | None
    dl_runtime_mode: str | None
    dl_runtime_options: Mapping[str, Any] | None = None
    robustness_role: str = "off"

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "enabled": bool(self.enabled),
            "name": self.name,
            "description": self.description,
            "param_source": self.param_source,
            "rule_policy": self.rule_policy,
            "dl_enabled": bool(self.dl_enabled),
            "dl_id": self.dl_id,
            "dl_runtime_mode": self.dl_runtime_mode,
            "dl_runtime_options": (
                None if self.dl_runtime_options is None else dict(self.dl_runtime_options)
            ),
            "robustness_role": self.robustness_role,
        }


MULTI_SEED_GPU_TRAIN_WORKERS_MIN = 1
MULTI_SEED_GPU_TRAIN_WORKERS_MAX = 2


@dataclass(frozen=True)
class StrategyMultiSeedRobustnessSettings:
    robustness_id: str
    label: str
    enabled: bool
    profile_id: str
    seed_count: int
    seed_generator_seed: int
    gpu_train_workers: int
    cpu_replay_workers: int
    reuse_completed: bool
    console_mode: str
    progress_interval_seconds: float
    yearly_report: bool
    keep_checkpoints: bool
    keep_scores: bool
    keep_replay_details: bool
    keep_attribution_source: bool
    romd_reference_baselines: Mapping[str, Mapping[str, str]]
    fixed_arm_ids: tuple[str, ...]
    stochastic_arm_ids: tuple[str, ...]
    paired_contrasts: tuple[Mapping[str, str], ...]
    output_root: str
    model_work_root: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "robustness_id": self.robustness_id,
            "label": self.label,
            "enabled": bool(self.enabled),
            "profile_id": self.profile_id,
            "seed_count": int(self.seed_count),
            "seed_generator_seed": int(self.seed_generator_seed),
            "gpu_train_workers": int(self.gpu_train_workers),
            "cpu_replay_workers": int(self.cpu_replay_workers),
            "reuse_completed": bool(self.reuse_completed),
            "console_mode": self.console_mode,
            "progress_interval_seconds": float(self.progress_interval_seconds),
            "yearly_report": bool(self.yearly_report),
            "keep_checkpoints": bool(self.keep_checkpoints),
            "keep_scores": bool(self.keep_scores),
            "keep_replay_details": bool(self.keep_replay_details),
            "keep_attribution_source": bool(self.keep_attribution_source),
            "romd_reference_baselines": {
                str(key): dict(value)
                for key, value in self.romd_reference_baselines.items()
            },
            "fixed_arm_ids": list(self.fixed_arm_ids),
            "stochastic_arm_ids": list(self.stochastic_arm_ids),
            "paired_contrasts": [dict(item) for item in self.paired_contrasts],
            "output_root": self.output_root,
            "model_work_root": self.model_work_root,
        }


def validate_strategy_multi_seed_robustness_settings(
    settings: StrategyMultiSeedRobustnessSettings,
) -> None:
    if not str(settings.robustness_id).strip():
        raise ValueError("multi-seed robustness robustness_id不可空白")
    if not str(settings.label).strip():
        raise ValueError("multi-seed robustness label不可空白")
    if not str(settings.profile_id).strip():
        raise ValueError("multi-seed robustness profile_id不可空白")
    if int(settings.seed_count) < 2:
        raise ValueError("multi-seed robustness seed_count必須>=2")
    if int(settings.seed_generator_seed) < 0:
        raise ValueError("multi-seed robustness seed_generator_seed必須>=0")
    gpu_train_workers = int(settings.gpu_train_workers)
    if not (
        MULTI_SEED_GPU_TRAIN_WORKERS_MIN
        <= gpu_train_workers
        <= MULTI_SEED_GPU_TRAIN_WORKERS_MAX
    ):
        raise ValueError(
            "multi-seed robustness gpu_train_workers必須介於"
            f"{MULTI_SEED_GPU_TRAIN_WORKERS_MIN}～{MULTI_SEED_GPU_TRAIN_WORKERS_MAX}"
        )
    if int(settings.cpu_replay_workers) < 1:
        raise ValueError("multi-seed robustness cpu_replay_workers必須>=1")
    if str(settings.console_mode) not in {"compact", "verbose"}:
        raise ValueError("multi-seed robustness console_mode只支援compact/verbose")
    if float(settings.progress_interval_seconds) <= 0:
        raise ValueError("multi-seed robustness progress_interval_seconds必須>0")
    configured_reference_keys = {
        str(key).strip() for key in settings.romd_reference_baselines
    }
    if "min" not in configured_reference_keys or not configured_reference_keys.issubset({"min", "full"}):
        raise ValueError(
            "multi-seed robustness romd_reference_baselines必須至少定義min；full為可選語意基準"
        )
    for key, raw in settings.romd_reference_baselines.items():
        spec = dict(raw or {})
        if not str(spec.get("param_source") or "").strip():
            raise ValueError(f"multi-seed robustness {key} reference缺少param_source")
        if not str(spec.get("rule_policy") or "").strip():
            raise ValueError(f"multi-seed robustness {key} reference缺少rule_policy")
    fixed_ids = tuple(str(value).strip() for value in settings.fixed_arm_ids if str(value).strip())
    stochastic_ids = tuple(str(value).strip() for value in settings.stochastic_arm_ids if str(value).strip())
    if not fixed_ids:
        raise ValueError("multi-seed robustness至少需要一個fixed arm ID")
    if not stochastic_ids:
        raise ValueError("multi-seed robustness至少需要一個stochastic arm ID")
    if len(set(fixed_ids)) != len(fixed_ids):
        raise ValueError("multi-seed robustness fixed_arm_ids不得重複")
    if len(set(stochastic_ids)) != len(stochastic_ids):
        raise ValueError("multi-seed robustness stochastic_arm_ids不得重複")
    overlap = sorted(set(fixed_ids) & set(stochastic_ids))
    if overlap:
        raise ValueError(f"multi-seed robustness fixed/stochastic arms不得重疊: {overlap}")
    contrast_ids: set[str] = set()
    for raw in settings.paired_contrasts:
        spec = dict(raw or {})
        contrast_id = str(spec.get("contrast_id") or "").strip()
        left = str(spec.get("left") or "").strip()
        right = str(spec.get("right") or "").strip()
        if not contrast_id or not left or not right:
            raise ValueError("multi-seed robustness paired_contrasts必須提供contrast_id/left/right")
        if contrast_id in contrast_ids:
            raise ValueError(f"multi-seed robustness paired contrast ID重複: {contrast_id}")
        if left == right:
            raise ValueError(f"multi-seed robustness paired contrast不得自比: {contrast_id}")
        contrast_ids.add(contrast_id)
    _validate_relative_path(settings.output_root, field_name="multi_seed.output_root")
    _validate_relative_path(settings.model_work_root, field_name="multi_seed.model_work_root")


@dataclass(frozen=True)
class StrategyRuntimeIntegrationSettings:
    label: str
    enabled: bool
    selection_profile_id: str
    forward_profile_id: str
    selection_candidate_arm_id: str
    forward_candidate_arm_id: str
    selection_robustness_id: str
    forward_robustness_id: str
    output_root: str
    comparison_anchor_experiment_profile: str
    require_strict_romd_majority: bool = True
    max_selector_latency_ms: float = 10000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "enabled": bool(self.enabled),
            "selection_profile_id": self.selection_profile_id,
            "forward_profile_id": self.forward_profile_id,
            "selection_candidate_arm_id": self.selection_candidate_arm_id,
            "forward_candidate_arm_id": self.forward_candidate_arm_id,
            "selection_robustness_id": self.selection_robustness_id,
            "forward_robustness_id": self.forward_robustness_id,
            "output_root": self.output_root,
            "comparison_anchor_experiment_profile": self.comparison_anchor_experiment_profile,
            "require_strict_romd_majority": bool(self.require_strict_romd_majority),
            "max_selector_latency_ms": float(self.max_selector_latency_ms),
        }


def validate_strategy_runtime_integration_settings(
    settings: StrategyRuntimeIntegrationSettings,
) -> None:
    for field_name in (
        "label",
        "selection_profile_id",
        "forward_profile_id",
        "selection_candidate_arm_id",
        "forward_candidate_arm_id",
        "selection_robustness_id",
        "forward_robustness_id",
    ):
        if not str(getattr(settings, field_name)).strip():
            raise ValueError(f"runtime integration {field_name}不可空白")
    if settings.selection_profile_id == settings.forward_profile_id:
        raise ValueError("runtime integration Selection/Forward profile不得相同")
    if settings.selection_robustness_id == settings.forward_robustness_id:
        raise ValueError("runtime integration Selection/Forward robustness不得相同")
    _validate_relative_path(settings.output_root, field_name="runtime_integration.output_root")
    if float(settings.max_selector_latency_ms) <= 0.0:
        raise ValueError("runtime integration max_selector_latency_ms必須>0")


@dataclass(frozen=True)
class StrategyComparisonContrast:
    contrast_id: str
    enabled: bool
    left: str
    right: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "enabled": bool(self.enabled),
            "left": self.left,
            "right": self.right,
            "description": self.description,
        }


@dataclass(frozen=True)
class StrategyComparisonSettings:
    schema_version: int
    profile_id: str
    profile_label: str
    dataset: str
    start_date: str | None
    end_date: str | None
    param_policy: str
    max_positions: int
    rotation: str
    output_root: str
    reuse_output_roots: tuple[str, ...]
    preparation: StrategyPreparationPolicy
    parameter_sources: Mapping[str, StrategyParameterSource]
    dl_sources: Mapping[str, StrategyDLSource]
    arms: Mapping[str, StrategyComparisonArm]
    contrasts: Mapping[str, StrategyComparisonContrast]

    @property
    def enabled_arms(self) -> tuple[StrategyComparisonArm, ...]:
        return tuple(arm for arm in self.arms.values() if arm.enabled)

    @property
    def enabled_contrasts(self) -> tuple[StrategyComparisonContrast, ...]:
        return tuple(item for item in self.contrasts.values() if item.enabled)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "dataset": self.dataset,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "param_policy": self.param_policy,
            "max_positions": int(self.max_positions),
            "rotation": self.rotation,
            "output_root": self.output_root,
            "reuse_output_roots": list(self.reuse_output_roots),
            "preparation": self.preparation.as_dict(),
            "parameter_sources": {
                key: value.as_dict() for key, value in self.parameter_sources.items()
            },
            "dl_sources": {
                key: value.as_dict() for key, value in self.dl_sources.items()
            },
            "arms": {key: value.as_dict() for key, value in self.arms.items()},
            "contrasts": {
                key: value.as_dict() for key, value in self.contrasts.items()
            },
        }


@dataclass(frozen=True)
class StrategyPreparationAction:
    action_id: str
    artifact_key: str
    action: str
    builder_type: str | None
    description: str
    path: str
    dependencies: tuple[str, ...] = ()
    producer_work_type: str | None = None
    execution_priority: int = 100

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "artifact_key": self.artifact_key,
            "action": self.action,
            "builder_type": self.builder_type,
            "description": self.description,
            "path": self.path,
            "dependencies": list(self.dependencies),
            "producer_work_type": self.producer_work_type,
            "execution_priority": int(self.execution_priority),
        }


@dataclass(frozen=True)
class StrategyPreparationPlan:
    overall_status: str
    actions: tuple[StrategyPreparationAction, ...]

    @classmethod
    def from_actions(
        cls, actions: tuple[StrategyPreparationAction, ...] | list[StrategyPreparationAction]
    ) -> "StrategyPreparationPlan":
        normalized = tuple(actions)
        cls._validate_dependencies(normalized)
        action_names = {item.action for item in normalized}
        overall_status = (
            "BLOCKED"
            if "BLOCKED" in action_names
            else "PREPARABLE"
            if action_names & {"BUILD", "REBUILD"}
            else "READY"
        )
        return cls(overall_status=overall_status, actions=normalized)

    @staticmethod
    def _validate_dependencies(actions: tuple[StrategyPreparationAction, ...]) -> None:
        action_by_key = {item.artifact_key: item for item in actions}
        if len(action_by_key) != len(actions):
            raise ValueError("前置工件計畫artifact_key不得重複")
        for item in actions:
            unknown = sorted(set(item.dependencies) - set(action_by_key))
            if unknown:
                raise ValueError(
                    f"前置工件{item.artifact_key}依賴未知工件: {', '.join(unknown)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visited:
                return
            if key in visiting:
                raise ValueError(f"前置工件依賴形成循環: {key}")
            visiting.add(key)
            for dependency in action_by_key[key].dependencies:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in action_by_key:
            visit(key)

    @property
    def blocked(self) -> bool:
        return self.overall_status == "BLOCKED"

    @property
    def preparable(self) -> bool:
        return self.overall_status == "PREPARABLE"

    @property
    def action_by_key(self) -> dict[str, StrategyPreparationAction]:
        return {item.artifact_key: item for item in self.actions}

    def select(
        self, required_artifact_keys: tuple[str, ...] | list[str] | set[str]
    ) -> "StrategyPreparationPlan":
        required = {str(value) for value in required_artifact_keys}
        action_by_key = self.action_by_key
        unknown = sorted(required - set(action_by_key))
        if unknown:
            raise ValueError("前置工件計畫缺少要求工件: " + ", ".join(unknown))

        selected: set[str] = set()

        def include(key: str) -> None:
            if key in selected:
                return
            selected.add(key)
            for dependency in action_by_key[key].dependencies:
                include(dependency)

        for key in required:
            include(key)
        return StrategyPreparationPlan.from_actions(
            [item for item in self.actions if item.artifact_key in selected]
        )

    def next_runnable_action(
        self, *, executed_signatures: set[tuple[str, str, str | None, str]] | None = None
    ) -> StrategyPreparationAction | None:
        action_by_key = self.action_by_key
        executed = executed_signatures or set()
        candidates: list[StrategyPreparationAction] = []
        for item in self.actions:
            if item.action not in {"BUILD", "REBUILD"}:
                continue
            signature = (item.artifact_key, item.action, item.builder_type, item.path)
            if signature in executed:
                continue
            dependency_actions = [action_by_key[key].action for key in item.dependencies]
            if all(value == "REUSE" for value in dependency_actions):
                candidates.append(item)
        if not candidates:
            return None
        return sorted(
            candidates, key=lambda item: (int(item.execution_priority), item.artifact_key)
        )[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "actions": [item.as_dict() for item in self.actions],
        }


def _validate_relative_path(value: str | None, *, field_name: str) -> None:
    if value in (None, ""):
        return
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name}必須是專案root相對路徑: {value}")


def _validate_builder(
    builder: StrategyArtifactBuilder | None,
    *,
    field_name: str,
    allowed_types: set[str],
) -> None:
    if builder is None:
        return
    if builder.builder_type not in allowed_types:
        raise ValueError(
            f"{field_name}.builder_type不支援: {builder.builder_type}; "
            f"allowed={sorted(allowed_types)}"
        )
    if not isinstance(builder.options, Mapping):
        raise ValueError(f"{field_name}.options必須是mapping")
    if builder.builder_type == "forward_oos_scores":
        if str(builder.options.get("scope") or "") != "forward_oos":
            raise ValueError(f"{field_name}.scope必須是forward_oos")
        if int(builder.options.get("inference_batch_size") or 0) < 1:
            raise ValueError(f"{field_name}.inference_batch_size必須>=1")
        if int(builder.options.get("inference_workers") or 0) < 1:
            raise ValueError(f"{field_name}.inference_workers必須>=1")
    if builder.builder_type == "oos_param_freeze":
        source_param_source_id = str(builder.options.get("source_param_source_id") or "").strip()
        if not source_param_source_id:
            raise ValueError(f"{field_name}.source_param_source_id不可空白")
        freeze_effective_date = str(builder.options.get("freeze_effective_date") or "").strip()
        freeze_cutoff_date = str(builder.options.get("freeze_cutoff_date") or "").strip()
        try:
            effective = date.fromisoformat(freeze_effective_date)
            cutoff = date.fromisoformat(freeze_cutoff_date)
        except ValueError as exc:
            raise ValueError(f"{field_name}.freeze date必須是YYYY-MM-DD") from exc
        if effective != cutoff + timedelta(days=1):
            raise ValueError(f"{field_name}.freeze effective必須是cutoff隔日")
        output_relative_dir = str(builder.options.get("output_relative_dir") or "").strip()
        _validate_relative_path(output_relative_dir, field_name=f"{field_name}.output_relative_dir")
    if builder.builder_type == "binary_dl_min_roos_rolling":
        parameter_set = str(builder.options.get("parameter_set") or "").lower()
        if parameter_set not in {"p2", "p3"}:
            raise ValueError(f"{field_name}.parameter_set必須是p2或p3")
        if int(builder.options.get("trials_per_fold") or 0) < 1:
            raise ValueError(f"{field_name}.trials_per_fold必須>=1")
        if float(builder.options.get("fixed_risk") or 0.0) <= 0.0:
            raise ValueError(f"{field_name}.fixed_risk必須>0")
        cap = float(builder.options.get("max_position_cap_pct") or 0.0)
        if not 0.0 < cap <= 1.0:
            raise ValueError(f"{field_name}.max_position_cap_pct必須介於0與1")
        p3_variant = builder.options.get("p3_variant")
        if p3_variant not in (None, ""):
            _validate_relative_path(str(p3_variant), field_name=f"{field_name}.p3_variant")
            if "/" in str(p3_variant) or "\\" in str(p3_variant):
                raise ValueError(f"{field_name}.p3_variant只允許單一資料夾名稱")
        for option_name in ("resume", "build_binary_pit", "binary_pit_resume", "quiet"):
            if option_name in builder.options and not isinstance(builder.options[option_name], bool):
                raise ValueError(f"{field_name}.{option_name}必須是bool")
    if builder.builder_type in {"extending_min_roos_stitch", "extending_full_roos_stitch"}:
        for option_name in ("historical_params_path", "current_params_path", "output_relative_dir"):
            value = str(builder.options.get(option_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name}.{option_name}不可空白")
            _validate_relative_path(value, field_name=f"{field_name}.{option_name}")
        if "quiet" in builder.options and not isinstance(builder.options["quiet"], bool):
            raise ValueError(f"{field_name}.quiet必須是bool")
    if builder.builder_type == "selection_historical_p2":
        if str(builder.options.get("parameter_set") or "").lower() != "p2_history":
            raise ValueError(f"{field_name}.parameter_set必須是p2_history")
        if int(builder.options.get("trials_per_fold") or 0) < 1:
            raise ValueError(f"{field_name}.trials_per_fold必須>=1")
        if int(builder.options.get("train_window_months") or 0) < 1:
            raise ValueError(f"{field_name}.train_window_months必須>=1")
        if int(builder.options.get("oos_months") or 0) < 1:
            raise ValueError(f"{field_name}.oos_months必須>=1")
        if float(builder.options.get("fixed_risk") or 0.0) <= 0.0:
            raise ValueError(f"{field_name}.fixed_risk必須>0")
        cap = float(builder.options.get("max_position_cap_pct") or 0.0)
        if not 0.0 < cap <= 1.0:
            raise ValueError(f"{field_name}.max_position_cap_pct必須介於0與1")
        if "optimizer_seed" not in builder.options:
            raise ValueError(f"{field_name}.optimizer_seed不可省略")
        if int(builder.options["optimizer_seed"]) < 0:
            raise ValueError(f"{field_name}.optimizer_seed必須>=0")
        for option_name in ("resume", "quiet"):
            if option_name in builder.options and not isinstance(builder.options[option_name], bool):
                raise ValueError(f"{field_name}.{option_name}必須是bool")
    if builder.builder_type == "selection_historical_full_roos":
        if str(builder.options.get("parameter_set") or "").lower() != "p4_history":
            raise ValueError(f"{field_name}.parameter_set必須是p4_history")
        if int(builder.options.get("trials_per_fold") or 0) < 1:
            raise ValueError(f"{field_name}.trials_per_fold必須>=1")
        if int(builder.options.get("train_window_months") or 0) < 1:
            raise ValueError(f"{field_name}.train_window_months必須>=1")
        if int(builder.options.get("oos_months") or 0) < 1:
            raise ValueError(f"{field_name}.oos_months必須>=1")
        if float(builder.options.get("fixed_risk") or 0.0) <= 0.0:
            raise ValueError(f"{field_name}.fixed_risk必須>0")
        cap = float(builder.options.get("max_position_cap_pct") or 0.0)
        if not 0.0 < cap <= 1.0:
            raise ValueError(f"{field_name}.max_position_cap_pct必須介於0與1")
        if "optimizer_seed" not in builder.options or int(builder.options["optimizer_seed"]) < 0:
            raise ValueError(f"{field_name}.optimizer_seed必須>=0")
        for option_name in ("resume", "quiet"):
            if option_name in builder.options and not isinstance(builder.options[option_name], bool):
                raise ValueError(f"{field_name}.{option_name}必須是bool")
    if builder.builder_type == "selection_pit_from_existing_folds":
        for option_name in ("resume", "allow_stale_source"):
            if option_name in builder.options and not isinstance(builder.options[option_name], bool):
                raise ValueError(f"{field_name}.{option_name}必須是bool")


def validate_strategy_comparison_settings(settings: StrategyComparisonSettings) -> None:
    if settings.schema_version < 2:
        raise ValueError("strategy comparison schema_version必須>=2")
    if not str(settings.profile_id).strip() or not str(settings.profile_label).strip():
        raise ValueError("strategy comparison profile identity不可空白")
    if settings.dataset not in {"reduced", "full"}:
        raise ValueError("strategy comparison dataset必須是reduced或full")
    if settings.param_policy not in {"base-finalist-best", "base-finalists-agree"}:
        raise ValueError("strategy comparison param_policy必須是正式rolling selector")
    if settings.max_positions < 1:
        raise ValueError("strategy comparison max_positions必須>=1")
    if settings.rotation not in {"off", "on"}:
        raise ValueError("strategy comparison rotation必須是off或on")
    if (settings.start_date is None) != (settings.end_date is None):
        raise ValueError("strategy comparison start_date與end_date必須同時設定或同時留空")
    _validate_relative_path(settings.output_root, field_name="output_root")
    for index, path in enumerate(settings.reuse_output_roots):
        _validate_relative_path(path, field_name=f"reuse_output_roots[{index}]")
        if str(path) == str(settings.output_root):
            raise ValueError("reuse_output_roots不得重複目前output_root")

    if not settings.parameter_sources:
        raise ValueError("至少需要一個parameter source")
    for key, source in settings.parameter_sources.items():
        if key != source.source_id or not key.strip():
            raise ValueError(f"parameter source key／source_id不一致: {key!r}")
        _validate_relative_path(
            source.path_template,
            field_name=f"parameter_sources[{key}].path_template",
        )
        _validate_relative_path(
            source.identity_manifest_path,
            field_name=f"parameter_sources[{key}].identity_manifest_path",
        )
        if source.trained_with_dl_id and source.trained_with_dl_id not in settings.dl_sources:
            raise ValueError(
                f"parameter source {key}引用不存在的trained_with_dl_id: "
                f"{source.trained_with_dl_id}"
            )
        _validate_builder(
            source.builder,
            field_name=f"parameter_sources[{key}].builder",
            allowed_types={
                "binary_dl_min_roos_rolling",
                "extending_min_roos_stitch",
                "extending_full_roos_stitch",
                "oos_param_freeze",
                "selection_historical_p2",
                "selection_historical_full_roos",
            },
        )
        if source.builder is not None and source.builder.enabled:
            options = dict(source.builder.options)
            parameter_set = str(options.get("parameter_set") or "").lower()
            configured_model_source = str(options.get("model_source_id") or "").strip() or None
            if parameter_set == "p3":
                if not source.trained_with_dl_id:
                    raise ValueError(f"parameter source {key}的P3 builder必須設定trained_with_dl_id")
                if (
                    configured_model_source is not None
                    and configured_model_source != source.trained_with_dl_id
                ):
                    raise ValueError(
                        f"parameter source {key}的model_source_id必須與trained_with_dl_id一致"
                    )
            elif parameter_set == "p2" and source.trained_with_dl_id is not None:
                raise ValueError(f"parameter source {key}的P2 builder不得設定trained_with_dl_id")
            if source.builder.builder_type == "selection_historical_p2" and source.trained_with_dl_id is not None:
                raise ValueError(f"parameter source {key}的Selection historical P2 builder不得設定trained_with_dl_id")
            if source.builder.builder_type == "oos_param_freeze":
                source_param_source_id = str(options.get("source_param_source_id") or "").strip()
                if source_param_source_id == key or source_param_source_id not in settings.parameter_sources:
                    raise ValueError(
                        f"parameter source {key}的OOS freeze source無效: {source_param_source_id!r}"
                    )
                if source.trained_with_dl_id is not None:
                    raise ValueError(f"parameter source {key}的OOS freeze builder不得設定trained_with_dl_id")

    for key, source in settings.dl_sources.items():
        if key != source.dl_id or not key.strip():
            raise ValueError(f"DL source key／dl_id不一致: {key!r}")
        if not source.filter_id or not source.model_architecture or not source.experiment_profile:
            raise ValueError(f"DL source identity不可空白: {key}")
        if not str(source.score_source).strip():
            raise ValueError(f"DL source score_source不可空白: {key}")
        if source.score_source == "canonical_runtime":
            if source.threshold is None or not 0.0 <= float(source.threshold) <= 1.0:
                raise ValueError(f"canonical DL source threshold必須介於0與1: {key}")
            if (
                source.forward_scores_builder is not None
                and source.forward_scores_builder.builder_type != "forward_oos_scores"
            ):
                raise ValueError(f"canonical DL source只允許forward_oos_scores builder: {key}")
        elif source.score_source == "continuous_ranker_oos":
            if source.threshold is not None:
                raise ValueError(f"continuous DL source不得設定binary threshold: {key}")
            if source.forward_scores_builder is not None:
                raise ValueError(f"continuous DL source不得由策略比較自動訓練／重建score: {key}")
        elif source.score_source == "selection_point_in_time":
            if source.threshold is not None:
                raise ValueError(f"Selection PIT continuous DL source不得設定binary threshold: {key}")
            if source.point_in_time_fold_months is not None and int(source.point_in_time_fold_months) < 1:
                raise ValueError(f"Selection PIT fold months必須>=1: {key}")
            if source.point_in_time_score_start_date not in (None, ""):
                try:
                    date.fromisoformat(str(source.point_in_time_score_start_date))
                except ValueError as exc:
                    raise ValueError(
                        f"Selection PIT score start必須是YYYY-MM-DD: {key}/{source.point_in_time_score_start_date!r}"
                    ) from exc
            if source.point_in_time_score_end_date not in (None, "") and str(source.point_in_time_score_end_date).lower() != "auto":
                try:
                    date.fromisoformat(str(source.point_in_time_score_end_date))
                except ValueError as exc:
                    raise ValueError(
                        f"Selection PIT score end必須是YYYY-MM-DD或auto: {key}/{source.point_in_time_score_end_date!r}"
                    ) from exc
            anchor = (
                None
                if source.point_in_time_fold_anchor_date in (None, "")
                else str(source.point_in_time_fold_anchor_date).strip()
            )
            if anchor is not None:
                try:
                    date.fromisoformat(anchor)
                except ValueError as exc:
                    raise ValueError(
                        f"Selection PIT fold anchor必須是YYYY-MM-DD: {key}/{anchor!r}"
                    ) from exc
            dirname = None if source.point_in_time_dirname in (None, "") else str(source.point_in_time_dirname).strip()
            if dirname is not None and (Path(dirname).name != dirname or dirname in {".", ".."}):
                raise ValueError(f"Selection PIT dirname必須是安全單一資料夾名稱: {key}/{dirname!r}")
            if (
                source.forward_scores_builder is not None
                and source.forward_scores_builder.builder_type
                != "selection_pit_from_existing_folds"
            ):
                raise ValueError(
                    f"Selection PIT只允許以既有fold/checkpoint重建推論工件，不得訓練模型: {key}"
                )
        else:
            raise ValueError(f"DL source score_source不支援: {key}/{source.score_source}")
        if source.score_source != "selection_point_in_time" and (
            source.point_in_time_score_start_date not in (None, "")
            or source.point_in_time_score_end_date not in (None, "")
            or source.point_in_time_fold_months is not None
            or source.point_in_time_fold_anchor_date not in (None, "")
            or bool(source.point_in_time_single_score_block)
            or source.point_in_time_dirname not in (None, "")
        ):
            raise ValueError(f"非Selection PIT source不得設定Rolling mode欄位: {key}")
        _validate_builder(
            source.forward_scores_builder,
            field_name=f"dl_sources[{key}].forward_scores_builder",
            allowed_types={"forward_oos_scores", "selection_pit_from_existing_folds"},
        )

    if len(settings.enabled_arms) < 2:
        raise ValueError("至少必須啟用兩個策略比較對象")
    for key, arm in settings.arms.items():
        if key != arm.arm_id or not key.strip():
            raise ValueError(f"arm key／arm_id不一致: {key!r}")
        if arm.robustness_role not in {"off", "fixed_baseline", "stochastic"}:
            raise ValueError(
                f"arm {key} robustness_role不支援: {arm.robustness_role!r}"
            )
        if arm.robustness_role == "fixed_baseline" and arm.dl_enabled:
            raise ValueError(f"arm {key} fixed_baseline不得啟用DL")
        if arm.robustness_role == "stochastic" and not arm.dl_enabled:
            raise ValueError(f"arm {key} stochastic robustness必須是DL-on")
        if arm.param_source not in settings.parameter_sources:
            raise ValueError(f"arm {key}引用不存在的param_source: {arm.param_source}")
        if arm.rule_policy not in {"formal", "all_off"}:
            raise ValueError(f"arm {key} rule_policy不支援: {arm.rule_policy}")
        parameter_source = settings.parameter_sources[arm.param_source]
        if arm.dl_enabled:
            if not arm.dl_id or arm.dl_id not in settings.dl_sources:
                raise ValueError(f"arm {key}啟用DL但dl_id無效: {arm.dl_id}")
            if arm.dl_runtime_mode not in SUPPORTED_STRATEGY_DL_RUNTIME_MODES:
                raise ValueError(
                    f"arm {key}的dl_runtime_mode不支援: {arm.dl_runtime_mode}; "
                    f"allowed={SUPPORTED_STRATEGY_DL_RUNTIME_MODES}"
                )
            options = dict(arm.dl_runtime_options or {})
            if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD:
                raw_max_age = options.get('stale_score_membership_guard_max_age_days')
                if isinstance(raw_max_age, bool) or not isinstance(raw_max_age, int) or raw_max_age < 0:
                    raise ValueError(
                        f"arm {key} stale-score guard必須指定非負整數 stale_score_membership_guard_max_age_days"
                    )
            if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT:
                fit_dl_id = str(options.get("expected_r_fit_dl_id") or "").strip()
                if not fit_dl_id or fit_dl_id not in settings.dl_sources:
                    raise ValueError(f"arm {key} Expected-PnL必須指定合法expected_r_fit_dl_id")
                if not str(options.get("expected_r_calibration_method") or "").strip():
                    raise ValueError(f"arm {key} Expected-PnL calibration method不可空白")
                if options.get("preserve_k_r0") is not True:
                    raise ValueError(f"arm {key} 第一階段Expected-PnL必須preserve_k_r0=True")
                if options.get("negative_expected_r_allowed") is not True:
                    raise ValueError(
                        f"arm {key} 第一階段Expected-PnL不得以Expected R負值改變K/R0"
                    )
            if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT:
                fit_dl_id = str(options.get("expected_excess_r_fit_dl_id") or "").strip()
                if not fit_dl_id or fit_dl_id not in settings.dl_sources:
                    raise ValueError(f"arm {key} Excess-Alpha必須指定合法expected_excess_r_fit_dl_id")
                if not str(options.get("expected_excess_r_calibration_method") or "").strip():
                    raise ValueError(f"arm {key} Excess-Alpha calibration method不可空白")
                if options.get("preserve_k_r0") is not True:
                    raise ValueError(f"arm {key} 第一階段Excess-Alpha必須preserve_k_r0=True")
                if options.get("negative_expected_excess_r_allowed") is not True:
                    raise ValueError(
                        f"arm {key} 第一階段Excess-Alpha不得以負Expected Excess-R改變K/R0"
                    )
                if options.get("selection_only") is not True:
                    raise ValueError(f"arm {key} 第一階段Excess-Alpha必須selection_only=True")
            if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT:
                fit_dl_id = str(options.get("expected_excess_r_fit_dl_id") or "").strip()
                if not fit_dl_id or fit_dl_id not in settings.dl_sources:
                    raise ValueError(f"arm {key} Excess-Alpha no-R0必須指定合法expected_excess_r_fit_dl_id")
                if not str(options.get("expected_excess_r_calibration_method") or "").strip():
                    raise ValueError(f"arm {key} Excess-Alpha no-R0 calibration method不可空白")
                if options.get("preserve_k") is not True:
                    raise ValueError(f"arm {key} Excess-Alpha no-R0必須preserve_k=True")
                if options.get("preserve_r0") is not False:
                    raise ValueError(f"arm {key} Excess-Alpha no-R0必須preserve_r0=False")
                if options.get("r0_minimum_repair") is not False:
                    raise ValueError(f"arm {key} Excess-Alpha no-R0必須r0_minimum_repair=False")
                if options.get("negative_expected_excess_r_allowed") is not True:
                    raise ValueError(
                        f"arm {key} Excess-Alpha no-R0不得以負Expected Excess-R改變K"
                    )
                if options.get("selection_only") is not True:
                    raise ValueError(f"arm {key} Excess-Alpha no-R0必須selection_only=True")
            if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL:
                fit_dl_id = str(options.get("expected_excess_r_fit_dl_id") or "").strip()
                if not fit_dl_id or fit_dl_id not in settings.dl_sources:
                    raise ValueError(f"arm {key} Excess-Alpha constrained必須指定合法expected_excess_r_fit_dl_id")
                if not str(options.get("expected_excess_r_calibration_method") or "").strip():
                    raise ValueError(f"arm {key} Excess-Alpha constrained calibration method不可空白")
                if options.get("preserve_k_r0") is not True:
                    raise ValueError(f"arm {key} Excess-Alpha constrained必須preserve_k_r0=True")
                if options.get("constrained_solver") != "exact_branch_and_bound_v1":
                    raise ValueError(f"arm {key} constrained_solver必須為exact_branch_and_bound_v1")
                if options.get("negative_expected_excess_r_allowed") is not True:
                    raise ValueError(
                        f"arm {key} Excess-Alpha constrained不得以負Expected Excess-R改變K/R0"
                    )
                if options.get("selection_only") is not True:
                    raise ValueError(f"arm {key} Excess-Alpha constrained必須selection_only=True")
            if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL:
                if options.get("preserve_k_r0") is not True:
                    raise ValueError(f"arm {key} Score constrained必須preserve_k_r0=True")
                if options.get("constrained_solver") != "exact_branch_and_bound_v1":
                    raise ValueError(f"arm {key} constrained_solver必須為exact_branch_and_bound_v1")
            if arm.dl_runtime_mode in {
                STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
                STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
            }:
                if options.get("preserve_k_r0") is not True:
                    raise ValueError(f"arm {key} Score+Safety constrained必須preserve_k_r0=True")
                if options.get("constrained_solver") != "exact_branch_and_bound_v1":
                    raise ValueError(f"arm {key} constrained_solver必須為exact_branch_and_bound_v1")
                expected_constraint = (
                    "baseline_residual_coverage_and_score_sum_floor_v1"
                    if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL
                    else "baseline_coverage_and_score_sum_floor_v1"
                )
                if options.get("safety_constraint") != expected_constraint:
                    raise ValueError(f"arm {key} safety_constraint contract不支援")
                if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL:
                    if options.get("safety_residualization") != "same_day_rank_ols_v1":
                        raise ValueError(f"arm {key} residual safety必須使用same_day_rank_ols_v1")
                safety_dl_id = str(options.get("safety_dl_id") or "").strip()
                if not safety_dl_id or safety_dl_id not in settings.dl_sources:
                    raise ValueError(f"arm {key} 必須指定合法safety_dl_id")
                if safety_dl_id == arm.dl_id:
                    raise ValueError(f"arm {key} safety_dl_id不得與primary dl_id相同")
                primary_source = settings.dl_sources[arm.dl_id]
                safety_source = settings.dl_sources[safety_dl_id]
                allowed_safety_sources = {"selection_point_in_time", "continuous_ranker_oos"}
                if (
                    primary_source.score_source not in allowed_safety_sources
                    or safety_source.score_source != primary_source.score_source
                ):
                    raise ValueError(
                        f"arm {key} dual-model safety primary/secondary必須使用同階段PIT或Forward score source"
                    )
                expected_selection_only = primary_source.score_source == "selection_point_in_time"
                if options.get("selection_only") is not expected_selection_only:
                    raise ValueError(
                        f"arm {key} dual-model safety selection_only與score source階段不一致"
                    )
                if any(
                    str(options.get(name) or '').strip()
                    for name in (
                        'expected_excess_r_fit_dl_id',
                        'expected_excess_r_calibration_method',
                        'expected_r_fit_dl_id',
                        'expected_r_calibration_method',
                    )
                ):
                    raise ValueError(f"arm {key} Score+Safety constrained不得依賴Expected-R/Excess-R calibration")
            if arm.dl_runtime_mode in {
                STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
                STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
                STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
            }:
                if options.get("preserve_k") is not True:
                    raise ValueError(f"arm {key} Score no-R0 exact必須preserve_k=True")
                if options.get("preserve_r0") is not False:
                    raise ValueError(f"arm {key} Score no-R0 exact必須preserve_r0=False")
                if options.get("r0_minimum_repair") is not False:
                    raise ValueError(f"arm {key} Score no-R0 exact必須r0_minimum_repair=False")
                if options.get("constrained_solver") != "exact_branch_and_bound_v1":
                    raise ValueError(f"arm {key} constrained_solver必須為exact_branch_and_bound_v1")
                if arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL:
                    if options.get("pareto_selection") != "normalized_product_v1":
                        raise ValueError(f"arm {key} Pareto selection method不支援")
                    if options.get("pareto_quality") != "score_sum_max_coverage_first":
                        raise ValueError(f"arm {key} Pareto quality contract不支援")
                    if options.get("pareto_capital") != "canonical_reserved_cost_milli":
                        raise ValueError(f"arm {key} Pareto capital contract不支援")
                if options.get("selection_only") is not True:
                    raise ValueError(f"arm {key} Score no-R0 exact目前只允許Selection PIT")
                selection_only = options.get("selection_only")
                if not isinstance(selection_only, bool):
                    raise ValueError(f"arm {key} Score constrained selection_only必須為bool")
                source_is_selection_pit = (
                    settings.dl_sources[arm.dl_id].score_source == "selection_point_in_time"
                )
                if selection_only is not source_is_selection_pit:
                    raise ValueError(
                        f"arm {key} Score constrained selection_only必須與DL source stage一致: "
                        f"score_source={settings.dl_sources[arm.dl_id].score_source!r}"
                    )
                if any(
                    str(options.get(name) or '').strip()
                    for name in (
                        'expected_excess_r_fit_dl_id',
                        'expected_excess_r_calibration_method',
                        'expected_r_fit_dl_id',
                        'expected_r_calibration_method',
                    )
                ):
                    raise ValueError(f"arm {key} Score constrained不得依賴Expected-R/Excess-R calibration")
            trained_with = parameter_source.trained_with_dl_id
            if trained_with is not None and arm.dl_id != trained_with:
                raise ValueError(
                    "DL-aware參數只能搭配訓練時相同的DL runtime: "
                    f"arm={key}, trained_with={trained_with}, runtime={arm.dl_id}"
                )
        else:
            if arm.dl_id is not None:
                raise ValueError(f"arm {key}關閉DL時dl_id必須為None")
            if arm.dl_runtime_mode is not None:
                raise ValueError(f"arm {key}關閉DL時dl_runtime_mode必須為None")
            if arm.dl_runtime_options not in (None, {}):
                raise ValueError(f"arm {key}關閉DL時dl_runtime_options必須為空")

    enabled_ids = {arm.arm_id for arm in settings.enabled_arms}
    for key, contrast in settings.contrasts.items():
        if key != contrast.contrast_id or not key.strip():
            raise ValueError(f"contrast key／contrast_id不一致: {key!r}")
        if contrast.left not in settings.arms or contrast.right not in settings.arms:
            raise ValueError(f"contrast {key}引用不存在的arm")
        if contrast.enabled and (
            contrast.left not in enabled_ids or contrast.right not in enabled_ids
        ):
            raise ValueError(
                f"啟用的contrast {key}兩端都必須是啟用arm: "
                f"{contrast.left}, {contrast.right}"
            )

    groups: dict[
        tuple[str, str],
        dict[str, StrategyComparisonArm | dict[tuple[str, str], StrategyComparisonArm] | None],
    ] = {}
    for arm in settings.arms.values():
        group = groups.setdefault(
            (arm.param_source, arm.rule_policy),
            {"off": None, "on": {}},
        )
        if not arm.dl_enabled:
            if group["off"] is not None:
                raise ValueError(
                    "同一param_source／rule_policy只能定義一個DL-off基準: "
                    f"{arm.param_source}/{arm.rule_policy}"
                )
            group["off"] = arm
            continue
        on_arms = group["on"]
        if not isinstance(on_arms, dict):
            raise TypeError("strategy comparison group on-arm contract錯誤")
        dl_id = str(arm.dl_id or "")
        runtime_mode = str(arm.dl_runtime_mode or "")
        runtime_key = (dl_id, runtime_mode)
        if runtime_key in on_arms:
            raise ValueError(
                "同一param_source／rule_policy不得重複定義相同DL source/runtime mode: "
                f"{arm.param_source}/{arm.rule_policy}/{dl_id}/{runtime_mode}"
            )
        on_arms[runtime_key] = arm

    enabled_groups = {
        (arm.param_source, arm.rule_policy) for arm in settings.enabled_arms
    }
    for group_key in enabled_groups:
        group = groups[group_key]
        off_arm = group["off"]
        on_arms = group["on"]
        if not isinstance(on_arms, dict):
            raise TypeError("strategy comparison group on-arm contract錯誤")
        enabled_on = [arm for arm in on_arms.values() if arm.enabled]
        if enabled_on and (
            not isinstance(off_arm, StrategyComparisonArm) or not off_arm.enabled
        ):
            raise ValueError(
                "啟用的DL模型比較必須共用一個已啟用DL-off基準: "
                f"{group_key}"
            )
        if not enabled_on and (
            not isinstance(off_arm, StrategyComparisonArm) or not off_arm.enabled
        ):
            raise ValueError(
                "啟用比較群組必須至少包含一個DL-off baseline: "
                f"{group_key}"
            )


def strategy_comparison_fingerprint(
    settings: StrategyComparisonSettings,
    *,
    artifact_identities: Mapping[str, Any] | None = None,
) -> str:
    enabled_arms = settings.enabled_arms
    enabled_contrasts = settings.enabled_contrasts
    parameter_source_ids = {arm.param_source for arm in enabled_arms}
    dl_source_ids = {
        arm.dl_id for arm in enabled_arms if arm.dl_enabled and arm.dl_id
    }
    for source_id in parameter_source_ids:
        trained_with = settings.parameter_sources[source_id].trained_with_dl_id
        if trained_with:
            dl_source_ids.add(trained_with)
    effective_settings = {
        "schema_version": int(settings.schema_version),
        "dataset": settings.dataset,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "param_policy": settings.param_policy,
        "max_positions": int(settings.max_positions),
        "rotation": settings.rotation,
        "output_root": settings.output_root,
        "preparation": settings.preparation.as_dict(),
        "parameter_sources": {
            source_id: settings.parameter_sources[source_id].as_dict()
            for source_id in sorted(parameter_source_ids)
        },
        "dl_sources": {
            dl_id: settings.dl_sources[dl_id].as_dict()
            for dl_id in sorted(dl_source_ids)
        },
        "arms": {arm.arm_id: arm.as_dict() for arm in enabled_arms},
        "contrasts": {
            item.contrast_id: item.as_dict() for item in enabled_contrasts
        },
    }
    payload = {
        "settings": effective_settings,
        "artifact_identities": dict(artifact_identities or {}),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


__all__ = [
    "MULTI_SEED_GPU_TRAIN_WORKERS_MIN",
    "MULTI_SEED_GPU_TRAIN_WORKERS_MAX",
    "StrategyArtifactBuilder",
    "StrategyComparisonArm",
    "StrategyComparisonContrast",
    "StrategyComparisonSettings",
    "StrategyDLSource",
    "StrategyParameterSource",
    "StrategyPreparationAction",
    "StrategyPreparationPlan",
    "StrategyPreparationPolicy",
    "StrategyRuntimeIntegrationSettings",
    "strategy_comparison_fingerprint",
    "validate_strategy_comparison_settings",
    "validate_strategy_runtime_integration_settings",
]
