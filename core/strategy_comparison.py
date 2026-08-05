"""通用策略績效比較設定契約與驗證。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StrategyParameterSource:
    source_id: str
    path_template: str | None
    description: str
    identity_manifest_path: str | None = None
    trained_with_dl_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path_template": self.path_template,
            "description": self.description,
            "identity_manifest_path": self.identity_manifest_path,
            "trained_with_dl_id": self.trained_with_dl_id,
        }


@dataclass(frozen=True)
class StrategyDLSource:
    dl_id: str
    filter_id: str
    model_architecture: str
    experiment_profile: str
    threshold: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dl_id": self.dl_id,
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "threshold": float(self.threshold),
            "description": self.description,
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
    dataset: str
    start_date: str | None
    end_date: str | None
    param_policy: str
    max_positions: int
    rotation: str
    output_root: str
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
            "dataset": self.dataset,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "param_policy": self.param_policy,
            "max_positions": int(self.max_positions),
            "rotation": self.rotation,
            "output_root": self.output_root,
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


def _validate_relative_path(value: str | None, *, field_name: str) -> None:
    if value in (None, ""):
        return
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name}必須是專案root相對路徑: {value}")


def validate_strategy_comparison_settings(settings: StrategyComparisonSettings) -> None:
    if settings.schema_version < 1:
        raise ValueError("strategy comparison schema_version必須>=1")
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

    if not settings.parameter_sources:
        raise ValueError("至少需要一個parameter source")
    for key, source in settings.parameter_sources.items():
        if key != source.source_id or not key.strip():
            raise ValueError(f"parameter source key／source_id不一致: {key!r}")
        _validate_relative_path(source.path_template, field_name=f"parameter_sources[{key}].path_template")
        _validate_relative_path(
            source.identity_manifest_path,
            field_name=f"parameter_sources[{key}].identity_manifest_path",
        )
        if source.trained_with_dl_id and source.trained_with_dl_id not in settings.dl_sources:
            raise ValueError(
                f"parameter source {key}引用不存在的trained_with_dl_id: "
                f"{source.trained_with_dl_id}"
            )

    for key, source in settings.dl_sources.items():
        if key != source.dl_id or not key.strip():
            raise ValueError(f"DL source key／dl_id不一致: {key!r}")
        if not source.filter_id or not source.model_architecture or not source.experiment_profile:
            raise ValueError(f"DL source identity不可空白: {key}")
        if not 0.0 <= float(source.threshold) <= 1.0:
            raise ValueError(f"DL source threshold必須介於0與1: {key}")

    if len(settings.enabled_arms) < 2:
        raise ValueError("至少必須啟用兩個策略比較對象")
    for key, arm in settings.arms.items():
        if key != arm.arm_id or not key.strip():
            raise ValueError(f"arm key／arm_id不一致: {key!r}")
        if arm.param_source not in settings.parameter_sources:
            raise ValueError(f"arm {key}引用不存在的param_source: {arm.param_source}")
        if arm.rule_policy not in {"formal", "all_off"}:
            raise ValueError(f"arm {key} rule_policy不支援: {arm.rule_policy}")
        if arm.dl_enabled:
            if not arm.dl_id or arm.dl_id not in settings.dl_sources:
                raise ValueError(f"arm {key}啟用DL但dl_id無效: {arm.dl_id}")
        elif arm.dl_id is not None:
            raise ValueError(f"arm {key}關閉DL時dl_id必須為None")

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

    groups: dict[tuple[str, str], dict[bool, StrategyComparisonArm]] = {}
    for arm in settings.arms.values():
        group = groups.setdefault((arm.param_source, arm.rule_policy), {})
        if bool(arm.dl_enabled) in group:
            raise ValueError(
                "同一param_source／rule_policy不得重複定義相同DL狀態: "
                f"{arm.param_source}/{arm.rule_policy}"
            )
        group[bool(arm.dl_enabled)] = arm
    enabled_groups = {
        (arm.param_source, arm.rule_policy) for arm in settings.enabled_arms
    }
    for group_key in enabled_groups:
        states = groups[group_key]
        if False not in states or True not in states:
            raise ValueError(
                "目前底層canonical replay以同參數DL-off／DL-on pair對帳；"
                f"啟用群組必須同時在config定義兩種狀態: {group_key}"
            )
        if bool(states[False].enabled) != bool(states[True].enabled):
            raise ValueError(
                "同一param_source／rule_policy的DL-off與DL-on必須一起開啟或關閉，"
                "避免停用arm仍被隱性執行: "
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
        "parameter_sources": {
            source_id: settings.parameter_sources[source_id].as_dict()
            for source_id in sorted(parameter_source_ids)
        },
        "dl_sources": {
            dl_id: settings.dl_sources[dl_id].as_dict()
            for dl_id in sorted(dl_source_ids)
        },
        "arms": {
            arm.arm_id: arm.as_dict() for arm in enabled_arms
        },
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
    "StrategyComparisonArm",
    "StrategyComparisonContrast",
    "StrategyComparisonSettings",
    "StrategyDLSource",
    "StrategyParameterSource",
    "strategy_comparison_fingerprint",
    "validate_strategy_comparison_settings",
]
