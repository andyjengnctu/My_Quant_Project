"""Project-wide Audit policy.

Only currently decision-relevant formal Audits belong in this config. A prior
result may remain active when its evidence is still required by an unresolved
decision question (for example K/R0 attribution); only genuinely closed or
superseded research Audits should be retired from runtime config/catalog.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.breakout_quality import (
    BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
    BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
)
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
)
from pathlib import Path
from typing import Any, Mapping

AUDIT_SCHEMA_VERSION = 15
AUDIT_OUTPUT_ROOT = "outputs/audit"
AUDIT_ACTIVE_MODULE_ID = "breakout_quality"


@dataclass(frozen=True)
class ReusableAuditDefinition:
    module_id: str
    report_id: str
    enabled: bool
    report_type: str
    description: str
    source: Mapping[str, Any]
    dimensions: Mapping[str, Any]
    output_subdir: str

    @property
    def audit_id(self) -> str:
        """Compatibility alias used by the shared read-only runner."""

        return self.report_id

    @property
    def audit_type(self) -> str:
        return self.report_type

    @property
    def outcomes(self) -> Mapping[str, Any]:
        return {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "report_type": self.report_type,
            "description": self.description,
            "source": dict(self.source),
            "dimensions": dict(self.dimensions),
            "output_subdir": self.output_subdir,
        }


AUDIT_REUSABLE_REPORTS: dict[str, dict[str, Any]] = {
    "breakout_quality": {
        "opportunity_selection": {
            "enabled": True,
            "report_type": "opportunity_selection_attribution",
            "description": (
                "Daily→Breakout→Orderable→Planned opportunity/selection attribution；"
                "固定比較actual MFE×Safety geometry，不含K/R0/cash/slot binding decomposition。"
            ),
            "source": {
                "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                "strategy_result_fingerprints": {
                    "extending_window_oos": "b12582a9de23",
                    "extending_window_rolling": "3bca1e932f2e",
                },
                "control_arm_id": "C71",
                "treatment_arm_id": "C74",
            },
            "dimensions": {
                "truth_high_cutoff": 0.50,
                "percentile_method": "average_zero_based",
                "truth_geometry_bins": 5,
                "breakdown": "overall",
            },
            "output_subdir": "breakout_quality/reusable/opportunity_selection",
        },
        "trade_outcome_path": {
            "enabled": True,
            "report_type": "trade_outcome_path_attribution",
            "description": (
                "Planned→Filled→Realized trade outcome/path attribution；"
                "比較first-passage、MFE/adverse與common/control-only/treatment-only cohorts。"
            ),
            "source": {
                "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                "strategy_result_fingerprints": {
                    "extending_window_oos": "b12582a9de23",
                    "extending_window_rolling": "3bca1e932f2e",
                },
                "control_arm_id": "C71",
                "treatment_arm_id": "C74",
            },
            "dimensions": {
                "truth_high_cutoff": 0.50,
                "percentile_method": "average_zero_based",
                "first_passage_thresholds_r": (1.0, 2.0, 3.0),
                "breakdown": "overall",
            },
            "output_subdir": "breakout_quality/reusable/trade_outcome_path",
        },
        "portfolio_drawdown": {
            "enabled": True,
            "report_type": "portfolio_drawdown_attribution",
            "description": (
                "Realized trades→portfolio capital/drawdown attribution；"
                "固定使用exact peak→trough MTM reconcile與truth quadrant contribution。"
            ),
            "source": {
                "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                "strategy_result_fingerprints": {
                    "extending_window_oos": "b12582a9de23",
                    "extending_window_rolling": "3bca1e932f2e",
                },
                "control_arm_id": "C71",
                "treatment_arm_id": "C74",
            },
            "dimensions": {
                "truth_high_cutoff": 0.50,
                "percentile_method": "average_zero_based",
                "drawdown_top_n": 1,
                "breakdown": "overall",
            },
            "output_subdir": "breakout_quality/reusable/portfolio_drawdown",
        },
    }
}

AUDIT_MODULES: dict[str, dict[str, Any]] = {
    "breakout_quality": {
        "enabled": True,
        "audits": {
            "AUD-mr13ab-survival-increment": {
                "enabled": True,
                "audit_type": "mr13ab_survival_increment",
                "description": (
                    "只讀MR-13AB frozen Forward OOS score/report與其內嵌MR-13K full-horizon reference target；"
                    "只在first-breach target真正改寫的rows與兩Target要求相反排序的same-date pairs上，"
                    "檢驗MR-13AB frozen score是否呈現survival ordering。"
                ),
                "source": {
                    "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                    "model_architecture": "inception_time_v1",
                    "candidate_profile_id": DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                    "candidate_research_id": "MR-13AB",
                    "candidate_target_id": DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
                    "reference_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                    "reference_research_id": "MR-13K",
                    "reference_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                    "seed": 42,
                },
                "dimensions": {
                    "changed_tolerance_r": 1e-6,
                    "top_fraction": 0.10,
                    "percentile_method": "average_zero_based",
                },
                "outcomes": {
                    "decision_question": (
                        "MR-13AB frozen score是否在first-breach改寫rows／conflict pairs上真正遵循survival-adjusted ordering，"
                        "同時保留既有Pure-MFE ranking能力？"
                    ),
                    "critical_uncertainty": (
                        "MR-13AB aggregate learnability可能只由92%+未改Target rows支撐；必須隔離7.63% changed rows與"
                        "target-order conflict pairs，直接檢查同一frozen score在AB truth與K reference truth衝突處偏向哪一方。"
                    ),
                    "stopping_condition": (
                        "一次 reference-target-controlled frozen-score Audit 足以決定 "
                        "SURVIVAL_INCREMENT_CONFIRMED → Model Gate PASS/進PIT-safe Rolling，"
                        "或 NO_INCREMENT → STOP；不得延伸barrier/lambda/threshold/backbone sweep。"
                    ),
                },
                "output_subdir": "breakout_quality/mr13ab_survival_increment",
            },
            # Retained while the K/R0 mechanism remains an active research question.
            "AUD-selection-k-r0-attribution": {
                "enabled": True,
                "audit_type": "selection_resource_constraint_attribution",
                "description": (
                    "保留C58-derived K/R0 resource conversion evidence；拆解C59/C65/C66 "
                    "Orderable→Raw Top-K→Planned→Filled與K headroom/R0 binding。"
                ),
                "source": {
                    "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                    "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                    "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                    "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                    "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                    "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                    "strategy_result_fingerprints": {
                        "extending_window_oos": "fc215f2140d6",
                        "extending_window_rolling": "94ec6685b7b0",
                    },
                    "strategy_arm_ids": ("C59", "C65", "C66"),
                    "baseline_arm_id": "C58",
                },
                "dimensions": {
                    "same_day_percentile_cutoff": 0.50,
                    "percentile_method": "average_zero_based",
                },
                "outcomes": {
                    "decision_question": "C58-derived K/R0/cash contract在何處改寫Raw DL selection？",
                    "critical_uncertainty": "K、R0與canonical cash feasibility對Raw→Planned MFE/Safety geometry的共同影響。",
                    "stopping_condition": "K/R0 treatment已取得足以GO/REJECT/NEXT EXPERIMENT的跨OOS/Rolling evidence後才退役此Audit。",
                },
                "output_subdir": "breakout_quality/selection_k_r0_attribution",
            },
            "AUD-mr13r-joint-capital-drawdown": {
                "enabled": False,
                "audit_type": "mr13r_joint_capital_drawdown",
                "description": (
                    "只讀C71-C74 completed OOS/Rolling sidecars與canonical MFE×Safety truth；"
                    "判斷MR-13R兩head是否已含HM/HS joint signal、Safety如何轉成capital exposure，"
                    "以及single-stock adverse改善未轉成portfolio RoMD的drawdown機制。"
                ),
                "source": {
                    "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                    "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                    "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                    "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                    "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                    "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                    "strategy_result_fingerprints": {
                        "extending_window_oos": "b12582a9de23",
                        "extending_window_rolling": "3bca1e932f2e",
                    },
                    "strategy_arm_ids": ("C71", "C72", "C73", "C74"),
                    "joint_signal_anchor_arm_id": "C71",
                },
                "dimensions": {
                    "truth_high_cutoff": 0.50,
                    "percentile_method": "average_zero_based",
                    "joint_signal_bins": 5,
                    "drawdown_top_n": 5,
                },
                "outcomes": {
                    "decision_question": (
                        "MR-13R現有Raw Safety＋Conditional-MFE是否已含可利用HM/HS joint signal；"
                        "Safety→Exposure與single-stock adverse→portfolio MDD之間各由何種mechanism主導？"
                    ),
                    "critical_uncertainty": (
                        "joint signal是selector conversion缺口或model objective缺口；"
                        "capital exposure是stop-distance/notional/holding conversion；"
                        "RoMD缺口是否主要來自drawdown期的同期交易共振。"
                    ),
                    "stopping_condition": (
                        "只補足能決定NEXT=selector mechanism、new joint model target或portfolio construction的最小證據；"
                        "不做threshold/weight scan、不重跑strategy、不訓練模型。"
                    ),
                },
                "output_subdir": "breakout_quality/mr13r_joint_capital_drawdown",
            },
            "AUD-c69-marginal-position-attribution": {
                "enabled": False,
                "audit_type": "c69_marginal_position_attribution",
                "description": (
                    "C68/C69 matched marginal-position attribution：直接檢查K-Flex多出的K+1/K+2/K+3+ "
                    "positions是否帶回High-MFE、同時惡化Safety/path conversion。"
                ),
                "source": {
                    "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                    "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                    "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                    "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                    "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                    "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                    "strategy_result_fingerprints": {
                        "extending_window_oos": "fc215f2140d6",
                        "extending_window_rolling": "94ec6685b7b0",
                    },
                    "control_arm_id": "C68",
                    "treatment_arm_id": "C69",
                },
                "dimensions": {
                    "same_day_percentile_cutoff": 0.50,
                    "percentile_method": "average_zero_based",
                    "final_selector_stage": "feasible_ascent_final",
                    "first_passage_thresholds_r": (1.0, 2.0, 3.0),
                },
                "outcomes": {
                    "decision_question": "C69 extra positions是高MFE但低Safety/path conversion，還是單純較弱score tail？",
                    "critical_uncertainty": "C69相對matched C68新增positions的MFE/Safety、Adverse、Realized R與first-passage品質。",
                    "stopping_condition": "能決定unrestricted K-Flex REJECT，或有足夠證據轉向absolute MFE×Safety formulation後停止。",
                },
                "output_subdir": "breakout_quality/c69_marginal_position_attribution",
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


def get_reusable_audit_definitions(
    module_id: str,
    *,
    enabled_only: bool = False,
) -> tuple[ReusableAuditDefinition, ...]:
    raw_module = AUDIT_REUSABLE_REPORTS.get(str(module_id), {})
    definitions: list[ReusableAuditDefinition] = []
    for report_id, raw in raw_module.items():
        definition = ReusableAuditDefinition(
            module_id=str(module_id),
            report_id=str(report_id),
            enabled=bool(raw.get("enabled", False)),
            report_type=str(raw.get("report_type") or "").strip(),
            description=str(raw.get("description") or "").strip(),
            source=dict(raw.get("source") or {}),
            dimensions=dict(raw.get("dimensions") or {}),
            output_subdir=str(raw.get("output_subdir") or "").strip(),
        )
        if not definition.report_type or not definition.output_subdir:
            raise ValueError(f"Reusable Audit report contract不完整: {module_id}/{report_id}")
        _validate_relative_path(
            definition.output_subdir,
            field_name=f"{module_id}/{report_id}.output_subdir",
        )
        if not enabled_only or definition.enabled:
            definitions.append(definition)
    return tuple(definitions)


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
        get_reusable_audit_definitions(str(module_id))


validate_audit_config()

__all__ = [
    "AUDIT_ACTIVE_MODULE_ID",
    "AUDIT_MODULES",
    "AUDIT_OUTPUT_ROOT",
    "AUDIT_REUSABLE_REPORTS",
    "AUDIT_SCHEMA_VERSION",
    "AuditDefinition",
    "ReusableAuditDefinition",
    "get_active_audit_module_id",
    "get_audit_definitions",
    "get_audit_module_ids",
    "get_enabled_audit_definitions",
    "get_reusable_audit_definitions",
    "validate_audit_config",
]
