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



__all__ = [
    "AUDIT_MODULES",
    "AUDIT_REUSABLE_REPORTS",
    "AUDIT_SCHEMA_VERSION",
]
