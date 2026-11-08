"""Canonical formal/reusable Audit definition registry.

This module owns Audit schema/version and declarative Audit definitions.  User-selected
active module/output location remain in ``config.audit``; validation and typed resolution
live in ``core.audit_policy``.
"""

from __future__ import annotations

from typing import Any, Mapping

from config.market_data import RESEARCH_REQUIRED_CUTOFF
from core.research_summary import (
    get_research_summary_oracle_comparisons,
    get_research_summary_baseline_arm_ids,
)

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
                # AI: Summary membership projects the executable Oracle inventory.
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
                    "T40_Positive_Direct": "Retired 2026-11-06: old CJ+CW dual-Model Direct C143; receipt kept historically",
                    "T40_Positive": "Retired 2026-11-07: Default-only C144; historical Exact C142 and Default receipts retained",
                    "DailyUniversal_MinTrade60_NetR": "LABEL-daily_universal_formal_min_roos_trade60_netr_v1",
                    "DailyUniversal_MinTrade60_NetR_Direct": "LABEL-daily_universal_formal_min_roos_trade60_netr_v1",
                    "DailyUniversal_MinTrade60_NetPnL": "LABEL-daily_universal_formal_min_roos_trade60_netpnl_v1",
                    "DailyUniversal_MinTrade60_NetPnL_Direct": "LABEL-daily_universal_formal_min_roos_trade60_netpnl_v1",
                },
                # AI: Summary owns the baselines and comparison membership;
                # Target validation and Summary consume the same projections.
                "additional_parameter_baselines": {
                    family: arm_id for family, arm_id in get_research_summary_baseline_arm_ids().items()
                    if family != "min"
                },
                "oracle_comparisons": get_research_summary_oracle_comparisons(),
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
    "daily_trade_path_full_net_pnl": "netpnl_positive_probability_v1",
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
    if not raw or selector not in {"exact", "direct", "default"} or strategy not in {"Breakout", "Daily"}:
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


def oracle_parameter_family(row: Mapping[str, Any]) -> str:
    """Economic parameter family of an Oracle treatment (default: legacy Min)."""
    family = str(row.get("parameter_family") or "min").strip().lower()
    if not family or "::" in family or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in family):
        raise ValueError(f"Invalid Oracle parameter family: {family!r}")
    return family


def oracle_display_strategy(*, strategy: str, parameter_family: str) -> str:
    """Presentation-only entry label: family belongs to Strategy, never Target.

    This must not be used to construct canonical Oracle keys, FIT/Replay
    identities, Score sources or saved receipt payloads.
    """
    name = str(strategy)
    if name not in {"Breakout", "Daily"}:
        raise ValueError(f"Invalid Oracle display entry strategy: {name}")
    family = oracle_parameter_family({"parameter_family": parameter_family})
    return name if family == "min" else f"{name} ({family.capitalize()})"


def oracle_comparison_key(row: Mapping[str, Any]) -> str:
    """Stable Target x Entry x parameter-family Replay key, not a new Target.

    Historic Breakout/Min receipts remain unchanged.  Daily uses ::Daily;
    an independent parameter family uses ::Family.  No old keys migrate.
    """
    target = str(row.get("target") or "")
    strategy = str(row.get("strategy", "Breakout"))
    family = oracle_parameter_family(row)
    if not target or "::" in target or strategy not in {"Breakout", "Daily"}:
        raise ValueError(f"Invalid Oracle Target x Strategy: {target} / {strategy}")
    qualifiers = ([strategy] if strategy != "Breakout" else [])
    if family != "min":
        qualifiers.append(family.capitalize())
    return "::".join((target, *qualifiers))


def oracle_baseline_arm_ids(source: Mapping[str, Any]) -> dict[str, str]:
    """One baseline per declared economic parameter family.

    The control arm is NOT a duplicate Oracle target.  Both [1] consumers
    obtain its identity here, so adding a family cannot be display-only.
    """
    base = str(source["baseline_arm_id"])
    additional = source.get("additional_parameter_baselines") or {}
    if not isinstance(additional, Mapping):
        raise ValueError("Oracle parameter baselines must be a mapping")
    result = {"min": base}
    for family, arm_id in additional.items():
        normalized = oracle_parameter_family({"parameter_family": family})
        if normalized in result or not isinstance(arm_id, str) or not arm_id:
            raise ValueError(f"Duplicate or invalid Oracle baseline family: {family}")
        result[normalized] = arm_id
    return result


def oracle_baseline_display_name(source: Mapping[str, Any], arm_id: str) -> str:
    """One unscored-control label for the Oracle plan and overview.

    Baselines are registered Strategy arm IDs, not hindsight Targets.  The
    identity is displayed without a family-specific presentation alias.
    """
    arm = str(arm_id)
    if arm not in oracle_baseline_arm_ids(source).values():
        raise ValueError(f"Unknown Oracle no-DL control: {arm}")
    return arm


def validate_oracle_compare_family_coverage(
    source: Mapping[str, Any], *, active_arm_ids, arm_registry: Mapping[str, Any],
) -> None:
    """Fail fast if an active additional-family DL lacks [1] Oracle registration.

    [4] remains an independent comparison suite.  Only DL arms in economic
    families intentionally declared for Oracle are required to have a matching
    [1] treatment.  Multiple DL models may share a single Min Target Oracle,
    so the existing Min family retains its original shared-score semantics.
    """
    baselines = oracle_baseline_arm_ids(source)
    registered = resolve_current_oracle_comparisons(dict(source))
    claimed_arms = {}
    for row in registered:
        family = oracle_parameter_family(row)
        arm_id = str(row.get("strategy_arm_id") or "")
        if family == "min" or not arm_id:
            continue
        if arm_id in claimed_arms:
            raise ValueError(f"Strategy has duplicate Oracle comparisons: {arm_id}")
        claimed_arms[arm_id] = row
    for family, control_id in baselines.items():
        if family == "min":
            continue
        if control_id not in arm_registry:
            raise ValueError(f"Oracle family baseline not registered: {control_id}")
        baseline = arm_registry[control_id]
        signature = tuple(baseline[key] for key in ("param_source", "param_policy", "rule_policy"))
        for arm_id in active_arm_ids:
            arm = arm_registry[arm_id]
            if not arm.get("dl_enabled") or tuple(arm[key] for key in (
                    "param_source", "param_policy", "rule_policy")) != signature:
                continue
            row = claimed_arms.get(arm_id)
            if row is None or oracle_parameter_family(row) != family:
                raise ValueError(
                    f"[4] arm {arm_id} belongs to Oracle {family} family but [1] "
                    "has no executable matching comparison"
                )
            if str(row.get("strategy", "Breakout")) != str(arm.get("entry_strategy", "Breakout")):
                raise ValueError(f"[1]/[4] entry policy differs for {arm_id}")
            selector = str(row["selector"])
            expected_mode = ("resource-aware-continuous-score-no-k-no-r0"
                             if selector == "direct" else
                             "resource-aware-continuous-score-constrained-optimal")
            if arm.get("dl_runtime_mode") != expected_mode:
                raise ValueError(f"[1]/[4] selector differs for {arm_id}")


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
    scientific_keys: set[tuple[str, str, str, str]] = set()
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
            family = oracle_parameter_family(row)
            if family not in oracle_baseline_arm_ids(source):
                raise ValueError(f"Oracle parameter family has no baseline: {family}")
            scientific_key = (*oracle_target_semantics(row), family)
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
    "oracle_parameter_family",
    "oracle_baseline_arm_ids",
    "oracle_baseline_display_name",
    "oracle_display_strategy",
    "validate_oracle_compare_family_coverage",
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
        FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID,
    )
    recipe = comparison.get("score")
    if comparison.get("horizon_mode") == "full_trade_cycle":
        return {
            "daily_trade_path_net_r": DAILY_FORMAL_MINTRADE_FULLCYCLE_NETR_TARGET_ID,
            "daily_trade_path_net_pnl": DAILY_FORMAL_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
            "daily_trade_path_true_net_pnl": TRUE_DAILY_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
            "daily_trade_path_full_net_pnl": FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID,
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
