"""Project-wide Audit policy.

Audit implementation inventory lives in ``tools/audit/catalog.py``.  This file only
selects which read-only audits are active and provides user-adjustable source,
dimension, outcome, and output policy.  Audit code must never mutate strategy,
labels, models, parameters, or runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

AUDIT_SCHEMA_VERSION = 3
AUDIT_OUTPUT_ROOT = "outputs/audit"
AUDIT_ACTIVE_MODULE_ID = "breakout_quality"

AUDIT_MODULES: dict[str, dict[str, Any]] = {
    "breakout_quality": {
        "enabled": True,
        "audits": {
            "a9-pass-quality": {
                "enabled": False,
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
            "a9-pass-persistence": {
                "enabled": False,
                "audit_type": "pass_persistence",
                "description": "A9 PASS persistence：unique event、candidate-day與selected false-positive放大",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "arm_id": "C12",
                },
                "dimensions": {
                    "selected_amplification": True,
                },
                "outcomes": {
                    "label_quality": True,
                    "realized_r": True,
                },
                "output_subdir": "breakout_quality/a9_pass_persistence",
            },
            "a9-selection-confidence": {
                "enabled": False,
                "audit_type": "selection_confidence",
                "description": "A9 confidence在DL Selection Mode多PASS競爭時對Event Label／Realized R的排序力",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "arm_id": "C12",
                },
                "dimensions": {
                    "minimum_competing_pass_candidates": 2,
                    "score_quantile_groups": 5,
                },
                "outcomes": {
                    "label_quality": True,
                    "realized_r": True,
                },
                "output_subdir": "breakout_quality/a9_selection_confidence",
            },
            "c15-strategy-attribution": {
                "enabled": False,
                "audit_type": "strategy_attribution",
                "description": "C15相對C3／C12的wealth-path、selection、capital geometry、slot occupancy與trade contribution歸因",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "candidate_arm_id": "C15",
                    "comparator_arm_ids": ["C3", "C12"],
                },
                "dimensions": {
                    "focus_year": 2024,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "log_wealth_path": True,
                    "selection_changes": True,
                    "capital_geometry": True,
                    "slot_occupancy": True,
                    "trade_contribution": True,
                },
                "output_subdir": "breakout_quality/c15_strategy_attribution",
            },
            "c15-source-attribution": {
                "enabled": False,
                "audit_type": "strategy_attribution",
                "description": "C15相對C14的跨run同runtime source attribution；隔離MR-12A all-label與MR-11G pass-only score source差異",
                "source": {
                    "kind": "strategy_compare",
                    "candidate_arm_id": "C15",
                    "comparator_arm_ids": ["C14"],
                    "arm_runs": {
                        "C15": {"config_fingerprint": "4da3217c83bd"},
                        "C14": {"config_fingerprint": "902c90b40dc2"},
                    },
                },
                "dimensions": {
                    "focus_year": 2024,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "log_wealth_path": True,
                    "selection_changes": True,
                    "capital_geometry": True,
                    "slot_occupancy": True,
                    "trade_contribution": True,
                },
                "output_subdir": "breakout_quality/c15_source_attribution",
            },
            "c23-c25-pit-realization": {
                "enabled": False,
                "audit_type": "strategy_realization_capture",
                "description": "Selection PIT直接部署失敗歸因：比較baseline與設定中的PIT ranking arms之exclusive trades、fill、sizing、holding、slot occupancy與Target→Realized capture",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "baseline_arm_id": "C23",
                    "candidate_arm_ids": ["C24", "C25"],
                },
                "dimensions": {
                    "focus_year": 2020,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "selection_changes": True,
                    "exclusive_trade_realization": True,
                    "capital_geometry": True,
                    "slot_occupancy": True,
                    "target_capture": True,
                },
                "output_subdir": "breakout_quality/c23_c25_pit_realization_capture",
            },
            "c23-c25-pit-fold-runtime": {
                "enabled": False,
                "audit_type": "pit_fold_runtime_attribution",
                "description": "Selection PIT fold drift／mixed-fold runtime歸因：檢查orderable pool跨fold score混合與winner capture損失是否集中於fold transition",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "baseline_arm_id": "C23",
                    "candidate_arm_ids": ["C24", "C25"],
                },
                "dimensions": {
                    "fold_boundary_window_days": 30,
                    "focus_year": 2020,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "runtime_fold_mixing": True,
                    "exclusive_winner_capture": True,
                    "fold_boundary_attribution": True,
                },
                "output_subdir": "breakout_quality/c23_c25_pit_fold_runtime",
            },
            "c23-c25-pit-target-realization": {
                "enabled": False,
                "audit_type": "pit_target_realization_attribution",
                "description": "Selection PIT Target→realized R／score-age歸因：檢查exclusive winner capture損失是否集中於較舊signal→entry age，區分event Target老化與Target公式本身失配",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "baseline_arm_id": "C23",
                    "candidate_arm_ids": ["C24", "C25"],
                },
                "dimensions": {
                    "score_age_quantile_groups": 4,
                    "focus_year": 2020,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "actual_exclusive_trade_alignment": True,
                    "target_realization_gap": True,
                    "score_age_attribution": True,
                },
                "output_subdir": "breakout_quality/c23_c25_pit_target_realization",
            },
            "c23-c26-pit-portfolio-translation": {
                "enabled": False,
                "audit_type": "strategy_attribution",
                "description": "SR-C26在Selection內已修復same-param selection R但仍落後baseline的portfolio translation歸因：拆exclusive/common trade PnL、capital geometry、slot occupancy與wealth path",
                "source": {
                    "kind": "strategy_compare",
                    "run": "latest",
                    "candidate_arm_id": "C26",
                    "comparator_arm_ids": ["C23", "C25"],
                },
                "dimensions": {
                    "focus_year": 2020,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "log_wealth_path": True,
                    "selection_changes": True,
                    "capital_geometry": True,
                    "slot_occupancy": True,
                    "trade_contribution": True,
                    "risk_dollar_translation": True,
                },
                "output_subdir": "breakout_quality/c23_c26_pit_portfolio_translation",
            },
            "forward-robustness-portfolio-translation": {
                "enabled": True,
                "audit_type": "robustness_portfolio_translation",
                "description": "Forward-OOS multi-seed ranking edge到trade-set、risk-dollar sizing、slot occupancy與wealth path的全seed轉化歸因",
                "source": {
                    "kind": "multi_seed_robustness",
                    "robustness_id": "forward_oos",
                    "run": "latest",
                    "candidate_arm_id": "C29",
                    "comparator_arm_id": "C20",
                },
                "dimensions": {
                    "focus_year": 2023,
                    "top_month_count": 5,
                    "top_trade_count": 20,
                },
                "outcomes": {
                    "trade_set_translation": True,
                    "risk_dollar_translation": True,
                    "slot_occupancy": True,
                    "wealth_path": True,
                    "all_seed_required": True,
                },
                "output_subdir": "breakout_quality/forward_robustness_portfolio_translation",
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
