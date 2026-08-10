"""通用策略績效比較設定、前置工件計畫與驗證契約。"""

from __future__ import annotations

from dataclasses import dataclass
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
SUPPORTED_STRATEGY_DL_RUNTIME_MODES = (
    STRATEGY_DL_RUNTIME_MODE_HARD_FILTER,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
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
        }


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "artifact_key": self.artifact_key,
            "action": self.action,
            "builder_type": self.builder_type,
            "description": self.description,
            "path": self.path,
        }


@dataclass(frozen=True)
class StrategyPreparationPlan:
    overall_status: str
    actions: tuple[StrategyPreparationAction, ...]

    @property
    def blocked(self) -> bool:
        return self.overall_status == "BLOCKED"

    @property
    def preparable(self) -> bool:
        return self.overall_status == "PREPARABLE"

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
        if not isinstance(off_arm, StrategyComparisonArm) or not off_arm.enabled:
            raise ValueError(
                "啟用的DL模型比較必須共用一個已啟用DL-off基準: "
                f"{group_key}"
            )
        if not enabled_on:
            raise ValueError(
                "啟用的DL-off基準至少需要一個已啟用DL-on比較對象: "
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
    "StrategyArtifactBuilder",
    "StrategyComparisonArm",
    "StrategyComparisonContrast",
    "StrategyComparisonSettings",
    "StrategyDLSource",
    "StrategyParameterSource",
    "StrategyPreparationAction",
    "StrategyPreparationPlan",
    "StrategyPreparationPolicy",
    "strategy_comparison_fingerprint",
    "validate_strategy_comparison_settings",
]
