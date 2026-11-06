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
                # Removed only from the CURRENT Oracle comparison inventory.
                # These 60-session scientific identities are retained solely
                # to read completed historical evidence. Unlike retired aliases,
                # their independent synthetic historical calculation remains
                # testable, but formal selection requires current registration.
                "historical_only_oracle_targets": {
                    "CG40_EqualRank": "40D MFE + full D40 Safety, 50/50 daily rank",
                    "DailyUniversal_MinTrade60_NetR": "LABEL-daily_universal_formal_min_roos_trade60_netr_v1",
                    "DailyUniversal_MinTrade60_NetR_Direct": "LABEL-daily_universal_formal_min_roos_trade60_netr_v1",
                    "DailyUniversal_MinTrade60_NetPnL": "LABEL-daily_universal_formal_min_roos_trade60_netpnl_v1",
                    "DailyUniversal_MinTrade60_NetPnL_Direct": "LABEL-daily_universal_formal_min_roos_trade60_netpnl_v1",
                },
                "oracle_comparisons": (
                    {"target": "MFE40_Control", "description": "40D MFE only (control, not target recommendation)", "score": "mfe", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "R40", "description": "40D close-to-close terminal return divided by canonical risk budget R", "score": "terminal_r", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "T40", "description": "(1 - min(MAE40, 1R)) * R40 - min(MAE40, 1R); MAE40 is nonnegative in R", "score": "terminal_mae_gate", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "T40-MFE", "description": "(1 - min(MAE40_R,1)) * MFE40_R - min(MAE40_R,1); full 40D path", "score": "mfe_mae_gate", "provider": "pure_mfe40_provider", "selector": "exact"},
                    {"target": "MFE40_Direct", "description": "Same MFE40_Control score; canonical No-K/No-R0 direct allocation", "score_source": "MFE40_Control", "selector": "direct"},
                    {"target": "R40_Direct", "description": "Same R40 score; canonical No-K/No-R0 direct allocation", "score_source": "R40", "selector": "direct"},
                    {"target": "T40_Direct", "description": "Same T40 score; canonical No-K/No-R0 direct allocation", "score_source": "T40", "selector": "direct"},
                    {"target": "T40_Positive", "description": "Formal perfect T40 percentile rank with independent perfect binary P(raw T40>0) admission; Exact K/R0", "score_source": "T40", "selector": "exact", "positive_output": "t40_positive_probability_v1", "strategy_arm_id": "C142"},
                    {"target": "T40_Positive_Direct", "description": "Same formal perfect T40 rank and P(raw T40>0) admission; Direct No-K/No-R0", "score_source": "T40", "selector": "direct", "positive_output": "t40_positive_probability_v1", "strategy_arm_id": "C143"},
                    {"target": "T40-MFE_Direct", "description": "Same T40-MFE score; canonical No-K/No-R0 direct allocation", "score_source": "T40-MFE", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade40_NetR", "description": "Formal Min ROOS Net-R, natural exit / 40 sessions / Research cutoff (earliest), hindsight ranking", "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile", "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade40_NetR_Direct", "description": "Same formal per-seed Min ROOS daily truth and ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade40_NetR", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade_NetR", "description": "Formal per-seed Min ROOS stock-day Net-R through natural exit or Research-data-end closeout; no session cap", "score": "daily_trade_path_net_r", "provider": "daily_trade40_profile", "horizon_mode": "full_trade_cycle", "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade_NetR_Direct", "description": "Same full-cycle Min Net-R target and ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade_NetR", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade_NetPnL", "description": "Formal Min full-cycle realized broker-ledger NetPnL at canonical reference sizing; natural exit or Research-cutoff closeout, hindsight ranking", "score": "daily_trade_path_net_pnl", "provider": "daily_trade40_profile", "horizon_mode": "full_trade_cycle", "selector": "exact"},
                    {"target": "DailyUniversal_MinTrade_NetPnL_Direct", "description": "Same full-cycle Min NetPnL perfect daily ranks; canonical No-K/No-R0 selector", "score_source": "DailyUniversal_MinTrade_NetPnL", "selector": "direct"},
                    {"target": "DailyUniversal_MinTrade_NetPnL_Positive", "description": "Formal perfect dual-output NetPnL rank plus Positive membership under Exact K/R0; common model-score admission", "score_source": "DailyUniversal_MinTrade_NetPnL", "selector": "exact", "positive_output": "netpnl_positive_probability_v1", "strategy_arm_id": "C125"},
                    {"target": "DailyUniversal_MinTrade_NetPnL_Positive_Direct", "description": "Formal perfect dual-output NetPnL rank plus Positive membership under Direct No-K/No-R0; historical raw-gated 204.88 is NON-PIT evidence, not reusable", "score_source": "DailyUniversal_MinTrade_NetPnL", "selector": "direct", "positive_output": "netpnl_positive_probability_v1", "strategy_arm_id": "C126"},
                    {"target": "TrueDailyUniversal_MinTrade_NetPnL_Positive", "description": "Every feature-eligible stock-day has an independent next-session prior-close limit buy attempt; canonical full-cycle Min broker NetPnL, perfect rank plus Positive output, Exact K/R0", "score": "daily_trade_path_true_net_pnl", "provider": "daily_trade40_profile", "horizon_mode": "full_trade_cycle", "selector": "exact", "positive_output": "netpnl_positive_probability_v1", "strategy_arm_id": "C135"},
                    {"target": "TrueDailyUniversal_MinTrade_NetPnL_Positive", "description": "Every feature-eligible stock-day has an independent next-session prior-close limit buy attempt; canonical full-cycle Min broker NetPnL, perfect rank plus Positive output, Exact K/R0", "score": "daily_trade_path_true_net_pnl", "provider": "daily_trade40_profile", "horizon_mode": "full_trade_cycle", "selector": "exact", "positive_output": "netpnl_positive_probability_v1", "strategy_arm_id": "C133", "strategy": "Daily"},
                    {"target": "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct", "description": "Same hypothetical daily entry full-cycle Min truth / Positive output; Direct No-K/No-R0", "score_source": "TrueDailyUniversal_MinTrade_NetPnL_Positive", "selector": "direct", "positive_output": "netpnl_positive_probability_v1", "strategy_arm_id": "C136"},
                    {"target": "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct", "description": "Same hypothetical daily entry full-cycle Min truth / Positive output; Direct No-K/No-R0", "score_source": "TrueDailyUniversal_MinTrade_NetPnL_Positive", "selector": "direct", "positive_output": "netpnl_positive_probability_v1", "strategy_arm_id": "C134", "strategy": "Daily"},
                    # AI註: Raw TrueDaily truth is shared; Positive admission is comparison-local.
                    {"target": "TrueDailyUniversal_MinTrade_NetPnL", "description": "Same TrueDaily full-cycle Min broker NetPnL truth; continuous perfect rank without Positive admission, Daily Exact K/R0", "score_source": "TrueDailyUniversal_MinTrade_NetPnL_Positive", "selector": "exact", "strategy_arm_id": "C137", "strategy": "Daily"},
                    {"target": "TrueDailyUniversal_MinTrade_NetPnL_Direct", "description": "Same TrueDaily full-cycle Min broker NetPnL truth; continuous perfect rank without Positive admission, Daily Direct No-K/No-R0", "score_source": "TrueDailyUniversal_MinTrade_NetPnL_Positive", "selector": "direct", "strategy_arm_id": "C138", "strategy": "Daily"},
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
        # Generic FIT-cohort audit: model and seed inventory are resolved at
        # runtime from canonical Model Compare/Test and benchmark policies.
        "dl_fit_sample_coverage": {
            "enabled": True,
            "report_type": "dl_fit_sample_coverage",
            "description": (
                "所有DL的Rolling FIT年份／樣本／成熟／Event-vs-Daily覆蓋稽核；"
                "逐Fold對照正式簽章，只讀、不重訓。"
            ),
            "source": {
                "model_inventory": "canonical_model_compare_test_settings",
                "seed_inventory": "canonical_model_robustness_benchmark",
                "fit_source": "canonical_pit_fold_manifest_and_training_group_selection",
            },
            "dimensions": {"mode": "rolling_oos", "reference_fold": "latest_verified"},
            "output_subdir": "breakout_quality/reusable/dl_fit_sample_coverage",
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
        # All completed one-time studies are retired; the generic menu and
        # reusable Audit definitions remain available without a study handler.
        "audits": {},
    },
}


# Only the canonical raw-score recipe decides a perfect Positive output's
# supervised meaning. Consumers must not infer it from an MR/C arm identity.
_ORACLE_POSITIVE_OUTPUT_BY_RECIPE = {
    "terminal_mae_gate": "t40_positive_probability_v1",
    "daily_trade_path_net_pnl": "netpnl_positive_probability_v1",
    "daily_trade_path_true_net_pnl": "netpnl_positive_probability_v1",
}


def oracle_positive_output_for_recipe(recipe: str) -> str | None:
    return _ORACLE_POSITIVE_OUTPUT_BY_RECIPE.get(str(recipe))


def oracle_scientific_target_name(legacy_receipt_key: str) -> str:
    """One-way normalization of historical Replay IDs into scientific Target names.

    Only for semantic reporting/validation; never for constructing receipt IDs.
    """
    name = str(legacy_receipt_key)
    if name.endswith("_Direct"):
        name = name[:-len("_Direct")]
    if name == "MFE40_Control":
        name = "MFE40"
    # Historical names are only inputs to this one-way display mapping.
    # The old DailyUniversal_ meant Breakout; the independent TrueDaily truth
    # is now scientifically called DailyUniversal_ (not TrueDailyUniversal_).
    if name.startswith("TrueDailyUniversal_"):
        name = "DailyUniversal_" + name[len("TrueDailyUniversal_"):]
    elif name.startswith("DailyUniversal_"):
        name = name[len("DailyUniversal_"):]
    return name


def oracle_target_semantics(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Scientific Target / Selector / Strategy, separate from signed Replay keys.

    Existing ``row['target']`` is an immutable execution receipt key, not the
    scientific Target displayed to users.  This is the sole legacy-name
    boundary; it must not feed score/truth/portfolio/reuse fingerprints.

    Positive is a scientific admission property of Target.  Direct and
    Control are *not* target properties.  Historical DailyUniversal_ means
    Breakout-signal truth (prefix removed); historical TrueDailyUniversal_
    means independent daily-entry truth (displayed as DailyUniversal_).
    """
    raw = str(row.get("target") or "")
    selector = str(row.get("selector") or "").lower()
    strategy = str(row.get("strategy", "Breakout"))
    if not raw or selector not in {"exact", "direct"} or strategy not in {"Breakout", "Daily"}:
        raise ValueError(f"Invalid Oracle naming dimensions: {raw}/{selector}/{strategy}")
    if raw.endswith("_Direct") != (selector == "direct"):
        raise ValueError(f"Legacy receipt selector mismatch: {raw} / {selector}")
    name = oracle_scientific_target_name(raw)
    if raw.startswith("DailyUniversal_") and strategy != "Breakout":
        raise ValueError(f"DailyUniversal is historically Breakout: {raw}")
    if "_Control" in name or name.endswith("_Direct"):
        raise ValueError(f"Non-target suffix remains in scientific Target: {name}")
    if name.endswith("_Positive") != bool(row.get("positive_output")):
        raise ValueError(f"Positive Target/admission contract mismatch: {raw}")
    return name, selector.capitalize(), strategy


def oracle_comparison_key(row: Mapping[str, Any]) -> str:
    """Stable Target x Strategy comparison key, not a new Target identity.

    Legacy Breakout receipts retain their *exact* old Target key; Daily uses a
    separate execution key, so neither strategy can consume the other receipt.
    The original scientific Target remains row['target'] in both cases.
    """
    target = str(row.get("target") or "")
    strategy = str(row.get("strategy", "Breakout"))
    if not target or "::" in target or strategy not in {"Breakout", "Daily"}:
        raise ValueError(f"Invalid Oracle Target x Strategy: {target} / {strategy}")
    return target if strategy == "Breakout" else f"{target}::Daily"


def resolve_current_oracle_comparisons(source: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """One canonical executable Target x Strategy inventory for both views.

    The same scientific Target may appear with two entry strategies. Duplicate
    pairs are invalid; retired *Target* identities never become executable.
    """
    from collections.abc import Mapping

    registered = source.get("oracle_comparisons")
    if not isinstance(registered, (tuple, list)):
        raise ValueError("Canonical Oracle comparison inventory must be ordered")
    inactive = (
        set(source.get("retired_oracle_targets") or ())
        | set(source.get("historical_only_oracle_targets") or ())
    )
    keys: set[str] = set()
    scientific_keys: set[tuple[str, str, str]] = set()
    active: list[dict[str, Any]] = []
    for row in registered:
        if not isinstance(row, Mapping):
            raise ValueError("Canonical Oracle comparison must be a mapping")
        name = str(row.get("target") or "")
        key = oracle_comparison_key(row)
        if key in keys:
            raise ValueError(f"Canonical Oracle comparison duplicate Target x Strategy: {key}")
        if name in inactive:
            raise ValueError(f"Historical-only Oracle target cannot be executable: {name}")
        keys.add(key)
        if row.get("enabled", True):
            scientific_key = oracle_target_semantics(row)
            if scientific_key in scientific_keys:
                raise ValueError(f"Duplicate Target / Selector / Strategy: {scientific_key}")
            scientific_keys.add(scientific_key)
            active.append(dict(row))
    return tuple(active)

__all__ = [
    "AUDIT_MODULES",
    "AUDIT_REUSABLE_REPORTS",
    "AUDIT_SCHEMA_VERSION",
    "resolve_current_oracle_comparisons",
    "oracle_comparison_key",
    "oracle_target_semantics",
    "oracle_scientific_target_name",
    "oracle_positive_output_for_recipe",
]


def oracle_supervised_target_id(comparison):
    """Resolve the canonical supervised label for a raw Oracle score recipe.

    Oracle truth artifacts and supervised FIT artifacts can have different
    artifact IDs; their economic target correspondence is owned here, not
    inferred from model names by each consumer. Callers resolve score_source
    to its raw recipe before using this function.
    """
    from core.breakout_quality_runtime import (
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        DAILY_TERMINAL_R40_TARGET_ID, DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
        DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
        DAILY_FORMAL_MINTRADE40_NETR_TARGET_ID,
        DAILY_FORMAL_MINTRADE_FULLCYCLE_NETR_TARGET_ID,
        DAILY_FORMAL_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
        TRUE_DAILY_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
    )
    recipe = comparison.get("score")
    if comparison.get("horizon_mode") == "full_trade_cycle":
        return {
            "daily_trade_path_net_r": DAILY_FORMAL_MINTRADE_FULLCYCLE_NETR_TARGET_ID,
            "daily_trade_path_net_pnl": DAILY_FORMAL_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
            "daily_trade_path_true_net_pnl": TRUE_DAILY_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
        }.get(recipe)
    if recipe == "daily_trade_path_net_r":
        return (DAILY_FORMAL_MINTRADE40_NETR_TARGET_ID
                if comparison.get("horizon_sessions") in (None, 40) else None)
    return {
        "mfe": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        "terminal_r": DAILY_TERMINAL_R40_TARGET_ID,
        "terminal_mae_gate": DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
        "mfe_mae_gate": DAILY_FULL_HORIZON_MAE_GATED_MFE_T40_TARGET_ID,
    }.get(recipe)
