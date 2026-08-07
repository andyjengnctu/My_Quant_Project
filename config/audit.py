"""正式 Audit／診斷設定。

所有 Audit 對象、來源、分層與輸出政策集中於此；Audit 只讀既有正式工件，
不得自行修改策略、Label、模型或 runtime。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

AUDIT_SCHEMA_VERSION = 1
AUDIT_OUTPUT_ROOT = "outputs/audit"

# 每個模組可獨立管理自己的 Audit；正式 App 只執行其模組下 enabled=True 的項目。
AUDIT_MODULES: dict[str, dict[str, Any]] = {
    "breakout_quality": {
        "enabled": True,
        "audits": {
            "a9_pass_quality": {
                "enabled": True,
                "audit_type": "pass_quality",
                "description": "A9 PASS 內部品質：Score、candidate age、candidate type 與 Label／Realized R",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "arm_id": "C12",
                },
                "dimensions": {
                    "score_quantile_groups": 5,
                    "candidate_age_quantile_groups": 5,
                    "candidate_type": True,
                },
                "outcomes": {
                    "label_quality": True,
                    "realized_r": True,
                },
                "output_subdir": "breakout_quality/a9_pass_quality",
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
    if definition.audit_type not in {"pass_quality"}:
        raise ValueError(f"不支援的audit_type: {definition.audit_type}")
    source = dict(definition.source)
    if str(source.get("kind") or "") != "strategy_compare":
        raise ValueError(f"{definition.audit_id}.source.kind目前只支援strategy_compare")
    run = str(source.get("run") or "").strip()
    if not run:
        raise ValueError(f"{definition.audit_id}.source.run不可空白")
    if run != "latest":
        _validate_relative_path(run, field_name=f"{definition.audit_id}.source.run")
    if not str(source.get("arm_id") or "").strip():
        raise ValueError(f"{definition.audit_id}.source.arm_id不可空白")

    dimensions = dict(definition.dimensions)
    for key in ("score_quantile_groups", "candidate_age_quantile_groups"):
        try:
            groups = int(dimensions.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{definition.audit_id}.{key}必須是整數") from exc
        if groups < 2:
            raise ValueError(f"{definition.audit_id}.{key}必須>=2")
    if not isinstance(dimensions.get("candidate_type"), bool):
        raise ValueError(f"{definition.audit_id}.candidate_type必須是bool")

    outcomes = dict(definition.outcomes)
    for key in ("label_quality", "realized_r"):
        if not isinstance(outcomes.get(key), bool):
            raise ValueError(f"{definition.audit_id}.{key}必須是bool")
    if not str(definition.output_subdir).strip():
        raise ValueError(f"{definition.audit_id}.output_subdir不可空白")
    _validate_relative_path(
        definition.output_subdir,
        field_name=f"{definition.audit_id}.output_subdir",
    )


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
    "AUDIT_MODULES",
    "AUDIT_OUTPUT_ROOT",
    "AUDIT_SCHEMA_VERSION",
    "AuditDefinition",
    "get_audit_definitions",
    "get_enabled_audit_definitions",
    "validate_audit_config",
]
