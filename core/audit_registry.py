"""Canonical formal/reusable Audit definition registry.

This module owns Audit schema/version and declarative Audit definitions.  User-selected
active module/output location remain in ``config.audit``; validation and typed resolution
live in ``core.audit_policy``.
"""

from __future__ import annotations

from typing import Any

from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
)
from core.breakout_quality_policy import (
    BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
    BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
)
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
)

AUDIT_SCHEMA_VERSION = 16

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
            "AUD-c80-c81-allocator-path-attribution": {
                "enabled": True,
                "audit_type": "c80_c81_allocator_path_attribution",
                "description": (
                    "只讀同一MR-13AH score的C80 exact K/R0 constrained與C81 No-K/No-R0 direct；"
                    "拆解membership substitution、MFE→realized path conversion與exact MTM drawdown，"
                    "回答為何C81 selection/MFE不差但RoMD/MDD差，以及joint K/R0 allocator為何改善。"
                ),
                "source": {
                    "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                    "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                    "truth_provider_profile_id": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                    "mfe_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
                    "safety_target_id": DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
                    "evaluation_profile_ids": ("extending_window_oos", "extending_window_rolling"),
                    "strategy_result_fingerprints": {
                        "extending_window_oos": "4952368ccecb",
                        "extending_window_rolling": "927bbb373d0f",
                    },
                    "constrained_arm_id": "C80",
                    "direct_arm_id": "C81",
                },
                "dimensions": {
                    "truth_high_cutoff": 0.50,
                    "percentile_method": "average_zero_based",
                    "first_passage_thresholds_r": (1.0, 2.0, 3.0),
                    "adverse_bucket_edges_r": (0.5, 1.0),
                    "drawdown_top_n": 1,
                },
                "outcomes": {
                    "decision_question": (
                        "同一MR-13AH score下，為何C81 No-K/No-R0具有較好的selection/MFE與first-passage，"
                        "但realized EV/RoMD/MDD較差；C80的joint K/R0/cash exact basket contract如何改善？"
                    ),
                    "critical_uncertainty": (
                        "改善是否來自K/R0在raw score tail不可直接滿足resource contract時進行membership substitution，"
                        "進而降低HM/LS/adverse/giveback與同步MTM drawdown，而非單純曝險或投入量差異。"
                    ),
                    "stopping_condition": (
                        "OOS與Rolling均能定位Raw→C80 substitution、C80/C81 path conversion與最大MDD contributor後停止；"
                        "不得在此Audit調K/R0、重訓模型或strategy replay。"
                    ),
                },
                "output_subdir": "breakout_quality/c80_c81_allocator_path_attribution",
            },
        },
    },
}



__all__ = [
    "AUDIT_MODULES",
    "AUDIT_REUSABLE_REPORTS",
    "AUDIT_SCHEMA_VERSION",
]
