"""Canonical formal/reusable Audit definition registry.

This module owns Audit schema/version and declarative Audit definitions.  User-selected
active module/output location remain in ``config.audit``; validation and typed resolution
live in ``core.audit_policy``.
"""

from __future__ import annotations

from typing import Any

from config.market_data import RESEARCH_REQUIRED_CUTOFF

from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE,
)
from core.breakout_quality_policy import (
    BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
    BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
)
from core.breakout_quality_runtime import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
)

AUDIT_SCHEMA_VERSION = 17

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

        "target_oracle_romd": {
            "enabled": True,
            "report_type": "target_ground_truth_strategy_romd",
            "description": "Reusable hindsight target truth -> Exact / Direct Broker-Cash RoMD (NON-PIT)",
            "source": {
                "strategy_profile_id": "broker_cash_dl_conversion_rolling",
                "baseline_arm_id": "C96",
                "exact_arm_id": "C100",
                "direct_arm_id": "C101",
                "filter_id": BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
                "model_architecture": BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
                "pure_mfe40_provider": DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                "mfe20_d20_provider": DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE,
                "first_breach_provider": DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                # AI: The Oracle comparison list is the dependency SSOT.
                # Score recipes declare their consumed truth provider; Direct
                # rows share an existing score and change only the selector.
                # Removing a comparison automatically retires its unused
                # provider workload; historical provider profiles stay in the
                # catalog for read-only provenance.
                # AI: Retired Min signal-event ranking is not selectable.
                # The DailyUniversal Net-R truth uses one formal per-seed
                # Min path capability with a declared session horizon.
                "daily_trade40_profile": "daily_universal_full_horizon_mae_gated_mfe_t40_full_list_ndcg_pairwise",
                # Historical comparison slots: immutable names retained by the
                # canonical Registry, never executable and never reassignable.
                # They are NOT part of the active comparison inventory.
                "retired_oracle_targets": {
                    "MinTrade40_NetR": "LABEL-mintrade40_realized_net_r_v1",
                    "MinTrade40_NetR_Direct": "LABEL-mintrade40_realized_net_r_v1",
                },
                "oracle_comparisons": (
                    {"target": "CG40_EqualRank", "description": "40D MFE + full D40 Safety, 50/50 daily rank", "score": "equal_rank_mfe_safety", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "MFE40_Control", "description": "40D MFE only (control, not target recommendation)", "score": "mfe", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "R40", "description": "40D close-to-close terminal return divided by canonical risk budget R", "score": "terminal_r", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "T40", "description": "(1 - min(MAE40, 1R)) * R40 - min(MAE40, 1R); MAE40 is nonnegative in R", "score": "terminal_mae_gate", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "T40-MFE", "description": "(1 - min(MAE40_R,1)) * MFE40_R - min(MAE40_R,1); full 40D path", "score": "mfe_mae_gate", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "MFE40_Direct", "description": "Same MFE40_Control score; canonical No-K/No-R0 direct allocation", "score_source": "MFE40_Control", "selector": "direct"},
                    {"target": "R40_Direct", "description": "Same R40 score; canonical No-K/No-R0 direct allocation", "score_source": "R40", "selector": "direct"},
                    {"target": "T40_Direct", "description": "Same T40 score; canonical No-K/No-R0 direct allocation", "score_source": "T40", "selector": "direct"},
                    {"target": "T40-MFE_Direct", "description": "Same T40-MFE score; canonical No-K/No-R0 direct allocation", "score_source": "T40-MFE", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade40_NetR", "description": "Formal Min ROOS Net-R, natural exit / 40 sessions / Research cutoff (earliest), hindsight ranking", "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile", "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade40_NetR_Direct", "description": "Same formal per-seed Min ROOS daily truth and ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade40_NetR", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade60_NetR", "description": "Formal Min ROOS Net-R, natural exit / 60 sessions / Research cutoff (earliest), hindsight ranking", "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile", "horizon_sessions": 60, "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade60_NetR_Direct", "description": "Same formal 60-session Min ROOS daily truth and ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade60_NetR", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade_NetR", "description": "Formal per-seed Min ROOS stock-day Net-R through natural exit or Research-data-end closeout; no session cap", "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile", "horizon_mode": "full_trade_cycle", "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade_NetR_Direct", "description": "Same full-cycle Min Net-R target and ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade_NetR", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade_NetPnL", "description": "Formal Min full-cycle realized broker-ledger NetPnL at canonical reference sizing; natural exit or Research-cutoff closeout, hindsight ranking", "score": "daily_trade_path_net_pnl", "provider": "daily_trade40_profile", "horizon_mode": "full_trade_cycle", "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade_NetPnL_Direct", "description": "Same full-cycle Min NetPnL perfect daily ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade_NetPnL", "selector": "direct"},
                ),
            },
            "dimensions": {
                "percentile_method": "average_zero_based",
                # Oracle portfolios run through the same Research cutoff as
                # canonical Research DL/Strategy Compare. Do not reserve label
                # maturity days: all Oracle truth is settled at that cutoff.
                "oracle_evaluation_end_date": RESEARCH_REQUIRED_CUTOFF,
                "oracle_cohort": "canonical_feature_eligible_oracle_stock_days",
                "oracle_score_method": "canonical_daily_truth_percentile_v2",
                "evaluation_mode": "rolling",
                "iterative_oos_evidence": True,
            },
            "output_subdir": "breakout_quality/reusable/target_oracle_romd",
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
    # Stable module registration survives one-off retirement; reusable Audit remains separate.
    "breakout_quality": {
        "enabled": True,
        "audits": {},
    },
}



__all__ = [
    "AUDIT_MODULES",
    "AUDIT_REUSABLE_REPORTS",
    "AUDIT_SCHEMA_VERSION",
]
