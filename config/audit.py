"""Project-wide Audit policy.

Only currently decision-relevant formal Audits belong in this config. Completed
or rejected research Audits must be removed from the runtime catalog/config and
kept as results in the experiment registry/log instead of remaining disabled
forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

AUDIT_SCHEMA_VERSION = 8
AUDIT_OUTPUT_ROOT = "outputs/audit"
AUDIT_ACTIVE_MODULE_ID = "breakout_quality"

AUDIT_MODULES: dict[str, dict[str, Any]] = {
    "breakout_quality": {
        "enabled": True,
        "audits": {
            "mfe-safety-target-geometry": {
                "enabled": True,
                "audit_type": "continuous_truth_strategy_quadrants",
                "description": (
                    "只讀比較實際MFE×Safety母體、orderable candidate pool與目前策略arms，"
                    "驗證conditional-MFE target geometry是否仍把選股推向High-MFE / Low-Safety。"
                ),
                "source": {
                    "evaluation_profile_ids": (
                        "extending_window_oos",
                        "extending_window_rolling",
                    ),
                    "strategy_arm_ids": ("C58", "C59", "C65", "C66"),
                    "candidate_pool_arm_id": "C58",
                    "focus_arm_id": "C66",
                    "mfe_target_id": "daily_full_horizon_pure_mfe_r_v1",
                    "safety_target_id": "daily_full_horizon_low_adverse_r_v1",
                },
                "dimensions": {
                    "truth_join_keys": ("ticker", "date"),
                    "same_day_percentile_cutoff": 0.50,
                    "percentile_method": "average_zero_based",
                    "cohort_order": (
                        "population",
                        "candidate_pool",
                        "C58",
                        "C59",
                        "C65",
                        "C66",
                    ),
                },
                "outcomes": {
                    "decision_question": (
                        "C66的實際選股是否仍大量落在High-MFE / Low-Safety，"
                        "使J=U-E(U|S)雖可學但未對齊absolute high MFE under absolute high Safety？"
                    ),
                    "critical_uncertainty": (
                        "C66改善Conditional-MFE排序後，selection probability實際被推向哪個future-truth quadrant。"
                    ),
                    "stopping_condition": (
                        "取得OOS與Rolling的母體、orderable pool、C58/C59/C65/C66四象限分布與相對母體enrichment後，"
                        "證據即足以決定維持target或進下一個受控target experiment；不再追加同問題Audit。"
                    ),
                },
                "output_subdir": "breakout_quality/mfe_safety_target_geometry",
            },
        },
    },
}


@dataclass(frozen=True)
class AuditDefinition:
    module_id: str
    audit_id: str
    enabled: bool
    audit_type: str
    description: str
    source: Mapping[str, Any]
    dimensions: Mapping[str, Any]
    outcomes: Mapping[str, Any]
    output_subdir: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "audit_id": self.audit_id,
            "enabled": bool(self.enabled),
            "audit_type": self.audit_type,
            "description": self.description,
            "source": dict(self.source),
            "dimensions": dict(self.dimensions),
            "outcomes": dict(self.outcomes),
            "output_subdir": self.output_subdir,
        }


def _validate_relative_path(value: str, *, field_name: str) -> None:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name}必須是專案root相對路徑: {value}")


def _validate_definition(definition: AuditDefinition) -> None:
    if not definition.module_id or not definition.audit_id:
        raise ValueError("audit module_id／audit_id不可空白")
    if not definition.audit_type:
        raise ValueError(f"{definition.audit_id}.audit_type不可空白")
    if not isinstance(definition.source, Mapping):
        raise ValueError(f"{definition.audit_id}.source必須是mapping")
    if not isinstance(definition.dimensions, Mapping):
        raise ValueError(f"{definition.audit_id}.dimensions必須是mapping")
    if not isinstance(definition.outcomes, Mapping):
        raise ValueError(f"{definition.audit_id}.outcomes必須是mapping")
    if not str(definition.output_subdir).strip():
        raise ValueError(f"{definition.audit_id}.output_subdir不可空白")
    _validate_relative_path(
        definition.output_subdir,
        field_name=f"{definition.audit_id}.output_subdir",
    )


def get_active_audit_module_id() -> str:
    module_id = str(AUDIT_ACTIVE_MODULE_ID).strip()
    if not module_id:
        raise ValueError("AUDIT_ACTIVE_MODULE_ID不可空白")
    raw = AUDIT_MODULES.get(module_id)
    if not isinstance(raw, dict):
        raise ValueError(f"AUDIT_ACTIVE_MODULE_ID不存在: {module_id}")
    if not bool(raw.get("enabled", False)):
        raise ValueError(f"AUDIT_ACTIVE_MODULE_ID目前未啟用: {module_id}")
    return module_id


def get_audit_module_ids(*, enabled_only: bool = True) -> tuple[str, ...]:
    module_ids: list[str] = []
    for module_id, raw in AUDIT_MODULES.items():
        if not isinstance(raw, dict):
            raise ValueError(f"AUDIT_MODULES[{module_id!r}]必須是mapping")
        if enabled_only and not bool(raw.get("enabled", False)):
            continue
        module_ids.append(str(module_id))
    return tuple(module_ids)


def get_audit_definitions(module_id: str) -> tuple[AuditDefinition, ...]:
    module_key = str(module_id).strip()
    raw_module = AUDIT_MODULES.get(module_key)
    if not isinstance(raw_module, dict):
        return ()
    if not bool(raw_module.get("enabled", False)):
        return ()
    raw_audits = raw_module.get("audits")
    if not isinstance(raw_audits, dict):
        raise ValueError(f"AUDIT_MODULES[{module_key!r}].audits必須是mapping")
    definitions: list[AuditDefinition] = []
    for audit_id, raw in raw_audits.items():
        if not isinstance(raw, dict):
            raise ValueError(f"audit設定必須是mapping: {module_key}/{audit_id}")
        definition = AuditDefinition(
            module_id=module_key,
            audit_id=str(audit_id),
            enabled=bool(raw.get("enabled", False)),
            audit_type=str(raw.get("audit_type") or "").strip(),
            description=str(raw.get("description") or "").strip(),
            source=dict(raw.get("source") or {}),
            dimensions=dict(raw.get("dimensions") or {}),
            outcomes=dict(raw.get("outcomes") or {}),
            output_subdir=str(raw.get("output_subdir") or "").strip(),
        )
        _validate_definition(definition)
        definitions.append(definition)
    return tuple(definitions)


def get_enabled_audit_definitions(module_id: str) -> tuple[AuditDefinition, ...]:
    return tuple(item for item in get_audit_definitions(module_id) if item.enabled)


def validate_audit_config() -> None:
    if int(AUDIT_SCHEMA_VERSION) < 1:
        raise ValueError("AUDIT_SCHEMA_VERSION必須>=1")
    if not str(AUDIT_OUTPUT_ROOT).strip():
        raise ValueError("AUDIT_OUTPUT_ROOT不可空白")
    _validate_relative_path(AUDIT_OUTPUT_ROOT, field_name="AUDIT_OUTPUT_ROOT")
    if not isinstance(AUDIT_MODULES, dict) or not AUDIT_MODULES:
        raise ValueError("AUDIT_MODULES必須是非空mapping")
    for module_id in AUDIT_MODULES:
        get_audit_definitions(str(module_id))


validate_audit_config()

__all__ = [
    "AUDIT_ACTIVE_MODULE_ID",
    "AUDIT_MODULES",
    "AUDIT_OUTPUT_ROOT",
    "AUDIT_SCHEMA_VERSION",
    "AuditDefinition",
    "get_active_audit_module_id",
    "get_audit_definitions",
    "get_audit_module_ids",
    "get_enabled_audit_definitions",
    "validate_audit_config",
]
