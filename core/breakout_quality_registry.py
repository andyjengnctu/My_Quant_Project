"""Breakout-quality scientific/profile registry and immutable contract schemas.

This module owns scientific profile identifiers, experiment declarations,
research-spec metadata, and reusable context scientific contracts.  It must not depend
on user-selected current settings from ``config.breakout_quality``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from config.research import RESEARCH_SINGLE_SEED
from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN
from core.breakout_policy import build_breakout_optimizer_high_len_values
from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
from config.execution_policy import (
    DEFAULT_FIXED_RISK,
    DEFAULT_MAX_POSITION_CAP_PCT,
    DEFAULT_PORTFOLIO_MAX_POSITIONS,
    DEFAULT_PORTFOLIO_ROTATION,
)
from core.breakout_quality_runtime import (
    FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID,
    DAILY_MIN_P18_FULLCYCLE_NETPNL_TARGET_ID,
    DAILY_MIN_P19_FULLCYCLE_NETPNL_TARGET_ID,
    DAILY_FULL_P20_FULLCYCLE_NETPNL_TARGET_ID,
    PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
    PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
    PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_QUANTILE_AUX_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_ADAPTIVE_HORIZON_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_JOINT_MFE_D40_QUADRANT,
    TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY,
    TRAINING_OBJECTIVE_DAILY_NETPNL_PAYOFF_BINARY,
    TRAINING_OBJECTIVE_DAILY_SCALAR_SIGN_POSITIVE_BINARY,
    CONTINUOUS_RANKER_TRAINER_EVENT,
    CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
    CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
    CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM,
    CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK,
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
    SUPPORTED_CONTINUOUS_RANKER_SAFETY_SUPERVISION_SOURCES,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_FINITE_SECONDARY_TARGET,
    SUPPORTED_CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPES,
    SUPPORTED_CONTINUOUS_RANKER_PAIR_WEIGHT_POLICIES,
    get_continuous_ranker_pair_weight_policy,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT,
    SUPPORTED_CONTINUOUS_RANKER_PAIRWISE_REDUCTIONS,
    ContinuousRankerContextPolicy,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_EXTERNAL,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_RISK_NORMALIZED,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_NONE,
    CONTINUOUS_RANKER_TARGET_POSTPROCESS_EQUAL_RANK_MFE_LOW_ADVERSE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_NONE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_UPSIDE_LOW_ADVERSE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PREDICTED_SAFETY_MFE,
    CONTINUOUS_RANKER_TARGET_CONTEXT_TRANSFORM_PRESERVE_PURE_MFE,
    CONTINUOUS_RANKER_BATCH_MODE_SHUFFLED,
    CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
    CONTINUOUS_RANKER_TARGET_BUILDER_PERCENTILE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE,
    CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R,
    CONTINUOUS_RANKER_TARGET_BUILDER_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_TARGET_BUILDER_PARETO_COMPONENTS,
    CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_TARGET_BUILDER_CONDITIONAL_MFE_SINGLE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_PRIMARY,
    CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_HMHS,
    CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE_JOINT_MIN,
    CONTINUOUS_RANKER_TARGET_BUILDER_DIRECT_HMHS,
    CONTINUOUS_RANKER_LOSS_HANDLER_PERCENTILE_MSE,
    CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R,
    CONTINUOUS_RANKER_LOSS_HANDLER_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_CONDITIONAL_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SAFETY_MFE_JOINT_TRI_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_LISTWISE,
    CONTINUOUS_RANKER_AUX_TARGET_NONE,
    CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_AUX_TARGET_CONDITIONAL_MFE_OPPORTUNITY,
    CONTINUOUS_RANKER_SEMANTICS_DEFAULT,
    CONTINUOUS_RANKER_SEMANTICS_PAIRWISE,
    CONTINUOUS_RANKER_SEMANTICS_LISTWISE,
    CONTINUOUS_RANKER_SEMANTICS_RAW_R,
    CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN,
    CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS,
    CONTINUOUS_RANKER_SCORE_TRANSFORM_PROBABILITY,
    CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R,
    CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_MEAN_BATCH,
    CONTINUOUS_RANKER_EPOCH_LOSS_AGGREGATION_WEIGHTED,
    ContinuousRankerTargetPolicy,
    ContinuousRankerTrainingPolicy,
    ContinuousRankerObjectivePolicy,
    ContinuousRankerDependencySpec,
    BreakoutQualityOutputSchema,
    BREAKOUT_QUALITY_OUTPUT_SCHEMA,
    ContinuousRankerExecutionRecipe,
    get_continuous_ranker_training_policy,
    get_profile_enabled_continuous_ranker_training_objectives,
    build_continuous_ranker_execution_recipe,
)


# =============================================================================
# INTERNAL PROFILE DEFINITIONS AND SUPPORTED VALUES — normally do not edit
# =============================================================================

BASELINE_EXPERIMENT_PROFILE = "baseline"
ADAMW_ONLY_EXPERIMENT_PROFILE = "adamw_only"
ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE = "adam_warmup_cosine"
HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE = "history_masking_only"
UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE = "unique_group_sampling"
UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE = "unique_group_date_balanced"
STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE = "strategy_aligned_daily_percentile_mse"
STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE = "strategy_aligned_no_time_pass_magnitude_mse"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE = "strategy_aligned_no_time_all_event_mse"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE = "strategy_aligned_no_time_all_event_pairwise"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE = "strategy_aligned_no_time_all_event_listwise"
DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE = "daily_universal_no_time_pairwise"
DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE = "daily_universal_no_time_pairwise_gap_weighted"
DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE = "daily_universal_no_time_percentile_mse"
DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE = "daily_universal_no_time_upper_tail_pairwise"
DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE = "daily_universal_no_time_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE = "daily_universal_no_time_r_huber"
DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE = "daily_universal_no_time_r_mse"
DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_first_risk_breach_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_upside_conditional_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_context_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_winner_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_predicted_safety_conflict_discounted_pure_mfe_full_list_ndcg_pairwise"
)

PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION = 1
PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE = (
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)
PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE = "inception_time_v1"
PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID = "MR-13K"
PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED = 42


def get_predicted_upside_context_contract() -> dict[str, Any]:
    """Lightweight MR-13AC stacking contract shared by config/training/artifact consumers."""

    return {
        "schema_version": PREDICTED_UPSIDE_CONTEXT_SCHEMA_VERSION,
        "stage1_profile": PREDICTED_UPSIDE_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_UPSIDE_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_UPSIDE_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_UPSIDE_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_pure_mfe",
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "same_date_low_adverse_percentile",
        "stage2_target": "same_date_percentile_of_low_adverse_residual_given_predicted_upside_percentile",
        "stage2_context_used_as_input": True,
    }

PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION = 1
PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE = (
    "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise"
)
PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE = "inception_time_v1"
PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID = "MR-13M"
PREDICTED_SAFETY_CONTEXT_STAGE1_SEED = 42


def get_predicted_safety_context_contract() -> dict[str, Any]:
    """MR-13AD reverse-control stacking contract shared by all consumers."""

    return {
        "schema_version": PREDICTED_SAFETY_CONTEXT_SCHEMA_VERSION,
        "stage1_profile": PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety",
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "same_date_pure_mfe_percentile",
        "stage2_target": "same_date_percentile_of_pure_mfe_residual_given_predicted_safety_percentile",
        "stage2_context_used_as_input": True,
    }


# Backward-compatible canonical owner of the reusable MR-13M predicted-safety PIT context.
# MR-13AE consumes the exact same Stage-1 artifact; the owner path remains MR-13AD so the
# already-built 10+1 fold context can be reused without retraining or copying.
PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE = (
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
)
PREDICTED_SAFETY_CONTEXT_OWNER_ARCHITECTURE = "inception_time_predicted_safety_context_v1"


def get_predicted_safety_pure_mfe_contract() -> dict[str, Any]:
    """MR-13AE consumer contract: canonical Pure-MFE target plus PIT-safe safety context."""

    return {
        "schema_version": 1,
        "stage1_profile": PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety",
        "context_artifact_owner_profile": PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "canonical_full_horizon_pure_mfe_r",
        "stage2_target": "exact_mr13k_pure_mfe_order_no_residualization",
        "stage2_context_used_as_input": True,
    }


def get_predicted_safety_pair_weight_contract(pair_weight_policy: str) -> dict[str, Any]:
    """Canonical Pure-MFE pair-weight contract for PIT-safe predicted-Safety policies."""

    policy = get_continuous_ranker_pair_weight_policy(pair_weight_policy)
    if policy.context_source != CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY:
        raise ValueError(
            f"pair weight policy不是predicted-Safety context: {pair_weight_policy!r}"
        )
    if not policy.contract_pair_safety_weight or not policy.contract_pair_weight_combination:
        raise ValueError(f"pair weight policy缺少scientific contract metadata: {pair_weight_policy!r}")

    return {
        "schema_version": 1,
        "stage1_profile": PREDICTED_SAFETY_CONTEXT_STAGE1_PROFILE,
        "stage1_architecture": PREDICTED_SAFETY_CONTEXT_STAGE1_ARCHITECTURE,
        "stage1_research_id": PREDICTED_SAFETY_CONTEXT_STAGE1_RESEARCH_ID,
        "stage1_seed": PREDICTED_SAFETY_CONTEXT_STAGE1_SEED,
        "context_semantic": "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety",
        "context_artifact_owner_profile": PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
        "selection_context": "expanding_cross_fitted_point_in_time",
        "forward_context": "single_fixed_pre_oos_fit",
        "full_fit_selection_score_forbidden": True,
        "oos_statistics_for_training_forbidden": True,
        "selection_training_universe": "pit_context_covered_rows_only",
        "stage2_response": "canonical_full_horizon_pure_mfe_r",
        "stage2_target": "exact_mr13k_pure_mfe_order_no_residualization",
        "stage2_context_used_as_input": False,
        "pair_base_relevance": "mr13k_full_list_delta_ndcg",
        "pair_safety_weight": policy.contract_pair_safety_weight,
        "pair_weight_combination": policy.contract_pair_weight_combination,
        "pair_weight_reduction": "normalized_weighted_mean_over_comparable_same_date_pairs",
        "bucket_or_threshold": None,
        "lambda_or_temperature": None,
    }


def get_high_safety_weighted_pure_mfe_contract() -> dict[str, Any]:
    """Backward-compatible MR-13AF min-Safety pair-weight contract."""

    return get_predicted_safety_pair_weight_contract(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY
    )
DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE = (
    "daily_universal_full_horizon_mfe_adverse_dual_mse"
)
DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_BIGRU_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_bigru_full_horizon_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_equal_rank_mfe_low_adverse_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_pareto_mfe_low_adverse_pairwise"
)
DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_conditional_mfe_single_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_conditional_mfe_duo_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_duo_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_context_weighted_pure_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_weighted_full_horizon_opportunity_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_context_weighted_full_horizon_opportunity_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_HORIZON_MAE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_full_horizon_mae_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_FULL_D40_SAFETY_HS_CONDITIONAL_MFE_PROFILE = (
    "daily_universal_shared_full_d40_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE = (
    "daily_universal_shared_full_d20_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_TERMINAL_R40_PAIRWISE_PROFILE = "daily_universal_terminal_r40_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_TERMINAL_R40_POSITIVE_PROFILE = "daily_universal_terminal_r40_positive_binary"
DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PAIRWISE_PROFILE = "daily_universal_terminal_mae_gated_t40_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_POSITIVE_PROFILE = "daily_universal_terminal_mae_gated_t40_positive_binary"
DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE = "daily_universal_terminal_mae_gated_t40_probabilistic_pairwise"
DAILY_UNIVERSAL_FULL_HORIZON_MAE_GATED_MFE_T40_PAIRWISE_PROFILE = "daily_universal_full_horizon_mae_gated_mfe_t40_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_COUNTERFACTUAL_MINTRADE40_NETR_PAIRWISE_PROFILE = "daily_universal_counterfactual_mintrade40_netr_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_FORMAL_MINTRADE40_NETR_PAIRWISE_PROFILE = "daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETR_PAIRWISE_PROFILE = "daily_universal_formal_mintrade_fullcycle_netr_per_seed_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE = "daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise"
FORMAL_MINTRADE_NETPNL_DECISION_CONTEXT_PROFILE = "daily_formal_mintrade_fullcycle_netpnl_decision_context_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_JOINT_MFE_D40_QUADRANT_PROFILE = "daily_universal_joint_mfe_d40_quadrant_cross_entropy"
DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE = "daily_universal_formal_mintrade_fullcycle_netpnl_positive_binary"
FULLTRADE_NETPNL_POSITIVE_PROFILE = "daily_universal_formal_fulltrade_fullcycle_netpnl_positive_binary"
FULLTRADE_NETPNL_RANK_PROFILE = "daily_universal_formal_fulltrade_fullcycle_netpnl_full_list_ndcg_pairwise"
DAILY_MIN_P18_NETPNL_RANK_PROFILE = "daily_universal_min_p18_mintrade_netpnl_full_list_ndcg_pairwise"
DAILY_MIN_P18_NETPNL_POSITIVE_PROFILE = "daily_universal_min_p18_mintrade_netpnl_positive_binary"
from core.daily_trade_path_capabilities import DAILY_MIN_P21_FULLCYCLE_NETPNL_TARGET_ID

DAILY_MIN_P21_NETPNL_RANK_PROFILE = "daily_universal_min_p21_mintrade_netpnl_full_list_ndcg_pairwise"
DAILY_MIN_P21_NETPNL_PAYOFF_PROFILE = "daily_universal_min_p21_netpnl_utility_binary"
DAILY_MIN_P21_NETPNL_POSITIVE_PROFILE = "daily_universal_min_p21_mintrade_netpnl_positive_binary"
DAILY_MIN_P19_NETPNL_RANK_PROFILE = "daily_universal_min_p19_mintrade_netpnl_full_list_ndcg_pairwise"
DAILY_MIN_P19_NETPNL_POSITIVE_PROFILE = "daily_universal_min_p19_mintrade_netpnl_positive_binary"
DAILY_FULL_P20_NETPNL_RANK_PROFILE = "daily_universal_full_p20_fulltrade_netpnl_full_list_ndcg_pairwise"
DAILY_FULL_P20_NETPNL_POSITIVE_PROFILE = "daily_universal_full_p20_fulltrade_netpnl_positive_binary"
TRUE_DAILY_MINTRADE_NETPNL_POSITIVE_PROFILE = "true_daily_universal_mintrade_fullcycle_netpnl_positive_binary"
TRUE_DAILY_MINTRADE_NETPNL_PAIRWISE_PROFILE = "true_daily_universal_mintrade_fullcycle_netpnl_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_DIRECT_U40_HUBER_PROFILE = "daily_universal_full_horizon_mfe_minus_d40_huber"
DAILY_UNIVERSAL_DIRECT_U40_PAIRWISE_PROFILE = "daily_universal_full_horizon_mfe_minus_d40_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_PIT_TRADE_COUNT_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_pit_trade_count_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_PIT_TRADE_COUNT_PLACEBO_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_pit_trade_count_placebo_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_QUANTILE_AUX_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_adverse_quantile_aux_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_RAW_INPUT_ADJUSTED_TARGET_SHARED_SAFETY_HS_CONDITIONAL_MFE_PROFILE = (
    "daily_universal_raw_input_adjusted_target_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_RAW_MASK_ADJUSTED_TARGET_SHARED_SAFETY_HS_CONDITIONAL_MFE_PROFILE = (
    "daily_universal_raw_mask_adjusted_target_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_ADAPTIVE_HORIZON_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_adaptive_horizon_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_ADAPTIVE_INPUT_CONTEXT_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_adaptive_input_context_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_dynamic_hypergraph_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_RELATION_CHANGE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_dynamic_hypergraph_relation_change_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SCC_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_scc_pretrained_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FUTURE_PATH_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_future_path_pretrained_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_price_volume_structure_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_LOCAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_price_volume_structure_local_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PRICE_VOLUME_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_price_volume_multiscale_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_price_volume_position_aware_multiscale_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_priority_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_priority_stratified_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_hs_qualification_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_hs_boundary_weighted_qualification_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_task_specific_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_attn_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_task_specific_safety_attn_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_SELF_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_self_attn_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_PAIRWISE_RELATION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_pairwise_relation_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_PATCH_SAFETY_INCEPTION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_patch_safety_inception_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_WINDOW_RF_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_full_window_rf_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_wide_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_600bar_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_DEEP_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_deep_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_shared_safety_hs_conditional_mfe_600bar_wide_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_day_token_transformer_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_gru_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_BIGRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_bigru_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_GRU_BF16_GUARDED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_gru_bf16_guarded_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_GRU_BF16_BACKWARD_SCALED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_gru_bf16_backward_scaled_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_hmhs_tri_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_hmhs_single_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_hmhs_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_attn_pool_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_modern_tcn_attn_pool_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_safety_raw_mfe_joint_min_patch_transformer_attn_pool_mlp_head_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE = (
    "daily_universal_full_horizon_no_breach_patch_transformer_full_list_ndcg_pairwise"
)
DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE = "daily_universal_risk_normalized_net_full_list_ndcg_pairwise"
DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE = "daily_universal_risk_context_net_full_list_ndcg_pairwise"


TRAINING_SAMPLING_ALL_EVENT_ROWS = "all_event_rows_group_weighted"
TRAINING_SAMPLING_UNIQUE_TICKER_DATE = "unique_ticker_date"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES = (
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
)

TIME_WEIGHT_MODE_NONE = "none"
TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT = "year_balanced_sqrt"
TIME_WEIGHT_MODE_DATE_BALANCED = "date_balanced"
SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES = (
    TIME_WEIGHT_MODE_NONE,
    TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT,
    TIME_WEIGHT_MODE_DATE_BALANCED,
)

TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM = "batch_weight_sum"
TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE = "fixed_batch_size"

TRAINING_LABEL_SCOPE_ALL = "all_labels"
TRAINING_LABEL_SCOPE_PASS_ONLY = "pass_only"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES = (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
)

TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS = "breakout_event_groups"
TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS = "daily_eligible_stock_days"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES = (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)

TRAINING_OBJECTIVE_BINARY_CLASSIFICATION = "binary_classification"
CONTINUOUS_RANKER_TRAINING_OBJECTIVES = (
    get_profile_enabled_continuous_ranker_training_objectives()
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES = (
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    *CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS = (
    TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
)

LR_SCHEDULE_NONE = "none"
LR_SCHEDULE_LINEAR_WARMUP_COSINE = "linear_warmup_cosine"
AUGMENTATION_NONE = "none"
AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK = "old_history_contiguous_mask"

SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS = ("adam", "adamw")
SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES = (
    LR_SCHEDULE_NONE,
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
)
SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS = (
    AUGMENTATION_NONE,
    AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
)

NUMERICAL_EXECUTION_POLICY_ARCHITECTURE_DEFAULT = "architecture_default"
NUMERICAL_EXECUTION_POLICY_BF16_DYNAMIC_BACKWARD_SCALING = "bf16_dynamic_backward_scaling"
SUPPORTED_BREAKOUT_QUALITY_NUMERICAL_EXECUTION_POLICIES = (
    NUMERICAL_EXECUTION_POLICY_ARCHITECTURE_DEFAULT,
    NUMERICAL_EXECUTION_POLICY_BF16_DYNAMIC_BACKWARD_SCALING,
)


@dataclass(frozen=True)
class BreakoutQualityExperimentProfile:
    name: str
    optimizer_name: str
    lr_schedule_name: str = LR_SCHEDULE_NONE
    augmentation_name: str = "none"
    augmentation_probability: float = 0.0
    augmentation_protected_recent_bars: int = 0
    augmentation_min_mask_bars: int = 0
    augmentation_max_mask_bars: int = 0
    lr_warmup_fraction: float = 0.0
    lr_minimum_ratio: float = 1.0
    training_sampling_mode: str = TRAINING_SAMPLING_ALL_EVENT_ROWS
    time_weight_mode: str | None = None
    training_weight_reduction: str = TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM
    training_objective: str = TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
    continuous_target_id: str | None = None
    # A new science contract may project raw PIT scores to a common per-date
    # selector scale. Historical profiles keep the default raw score unchanged.
    portfolio_score_scale: str = "raw"
    loss_name: str = "cross_entropy"
    epoch_selection_metric: str = "validation_loss"
    training_label_scope: str = TRAINING_LABEL_SCOPE_ALL
    training_sample_scope: str = TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    raw_r_huber_delta_r: float | None = None
    model_architecture: str | None = None
    encoder_pretraining_profile: str | None = None
    fit_contract_version: int = 1
    score_calibration: str | None = None
    numerical_execution_policy: str = NUMERICAL_EXECUTION_POLICY_ARCHITECTURE_DEFAULT
    bf16_backward_retry_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("experiment profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("experiment profile name 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 optimizer: {self.optimizer_name!r}")
        if self.lr_schedule_name not in SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES:
            raise ValueError(f"不支援的 LR schedule: {self.lr_schedule_name!r}")
        if self.augmentation_name not in SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS:
            raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")
        if self.training_sampling_mode not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES:
            raise ValueError(
                f"不支援的 training sampling mode: {self.training_sampling_mode!r}"
            )
        if (
            self.time_weight_mode is not None
            and self.time_weight_mode not in SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES
        ):
            raise ValueError(f"不支援的 time weight mode: {self.time_weight_mode!r}")
        if self.training_weight_reduction not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS:
            raise ValueError(
                f"不支援的 training weight reduction: {self.training_weight_reduction!r}"
            )
        if self.training_objective not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES:
            raise ValueError(f"不支援的 training objective: {self.training_objective!r}")
        if self.training_label_scope not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES:
            raise ValueError(f"不支援的 training label scope: {self.training_label_scope!r}")
        if self.training_sample_scope not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLE_SCOPES:
            raise ValueError(f"不支援的 training sample scope: {self.training_sample_scope!r}")
        numerical_policy = str(self.numerical_execution_policy).strip().lower()
        if numerical_policy not in SUPPORTED_BREAKOUT_QUALITY_NUMERICAL_EXECUTION_POLICIES:
            raise ValueError(f"不支援的 numerical execution policy: {self.numerical_execution_policy!r}")
        retry_scales = tuple(float(value) for value in self.bf16_backward_retry_scales)
        if numerical_policy == NUMERICAL_EXECUTION_POLICY_ARCHITECTURE_DEFAULT:
            if retry_scales:
                raise ValueError("architecture-default numerical policy 不得指定 BF16 backward retry scales")
        elif numerical_policy == NUMERICAL_EXECUTION_POLICY_BF16_DYNAMIC_BACKWARD_SCALING:
            if not retry_scales:
                raise ValueError("BF16 dynamic backward scaling 必須指定至少一個 retry scale")
            if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in retry_scales):
                raise ValueError("BF16 backward retry scale 必須是介於0與1之間的有限值")
            if any(right >= left for left, right in zip(retry_scales, retry_scales[1:])):
                raise ValueError("BF16 backward retry scales 必須嚴格遞減")
        if self.model_architecture is not None:
            architecture = str(self.model_architecture).strip().lower()
            if not architecture or architecture != self.model_architecture:
                raise ValueError("profile model_architecture 必須是非空白小寫名稱")
            if any(token in architecture for token in ("/", "\\", "\x00")):
                raise ValueError("profile model_architecture 必須是安全名稱")
        if self.score_calibration is not None:
            raise ValueError("Single-ranker Positive calibration is retired; use a registered binary classifier")
        if type(self.fit_contract_version) is not int or self.fit_contract_version < 1:
            raise ValueError("FIT contract version must be a positive integer")
        if self.encoder_pretraining_profile is not None:
            raise ValueError("Encoder pretraining profiles are retired")
        if self.portfolio_score_scale not in {"raw", "daily_percentile"}:
            raise ValueError("unsupported portfolio score scale")
        if self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            if self.continuous_target_id is not None:
                raise ValueError("binary classification profile 不得指定 continuous_target_id")
            if self.loss_name != "cross_entropy" or self.epoch_selection_metric != "validation_loss":
                raise ValueError("binary classification profile 必須使用 cross_entropy / validation_loss")
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                raise ValueError("binary classification profile 必須使用all_labels scope")
            if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
                raise ValueError("binary classification profile 必須使用breakout_event_groups sample scope")
        elif self.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            if not str(self.continuous_target_id or "").strip():
                raise ValueError("continuous ranker profile 必須指定 continuous_target_id")
            training_policy = get_continuous_ranker_training_policy(
                self.training_objective
            )
            training_policy.validate_profile_loss_metric(
                loss_name=self.loss_name,
                epoch_selection_metric=self.epoch_selection_metric,
            )
            if self.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
                if self.loss_name == "huber_raw_r":
                    if self.raw_r_huber_delta_r is None or not math.isfinite(float(self.raw_r_huber_delta_r)) or float(self.raw_r_huber_delta_r) <= 0.0:
                        raise ValueError("Huber direct R regression必須指定正有限 raw_r_huber_delta_r")
                elif self.raw_r_huber_delta_r is not None:
                    raise ValueError("MSE direct R regression不得指定 raw_r_huber_delta_r")
            elif self.raw_r_huber_delta_r is not None:
                raise ValueError("非direct R regression profile不得指定 raw_r_huber_delta_r")
            if self.training_sampling_mode != TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                raise ValueError("continuous ranker只允許 unique ticker/date sampling")
            if self.time_weight_mode not in {None, TIME_WEIGHT_MODE_NONE}:
                raise ValueError("continuous ranker不允許 time weighting")
            if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
                raise ValueError("continuous ranker只允許 batch_weight_sum")
        if (
            self.training_weight_reduction == TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE
            and self.time_weight_mode != TIME_WEIGHT_MODE_DATE_BALANCED
        ):
            raise ValueError(
                "fixed_batch_size training weight reduction 目前只允許 date_balanced profile"
            )
        warmup_fraction = float(self.lr_warmup_fraction)
        minimum_ratio = float(self.lr_minimum_ratio)
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            if warmup_fraction != 0.0 or minimum_ratio != 1.0:
                raise ValueError("無 LR schedule 時 warmup 必須為 0、minimum ratio 必須為 1")
        elif self.lr_schedule_name == LR_SCHEDULE_LINEAR_WARMUP_COSINE:
            if not 0.0 < warmup_fraction < 1.0:
                raise ValueError("linear warmup fraction 必須介於 0 與 1 之間")
            if not 0.0 < minimum_ratio <= 1.0:
                raise ValueError("minimum LR ratio 必須介於 0 與 1 之間")
        augmentation_parameters = self.augmentation_parameters()
        if self.augmentation_name == AUGMENTATION_NONE:
            if (
                float(self.augmentation_probability) != 0.0
                or int(self.augmentation_protected_recent_bars) != 0
                or int(self.augmentation_min_mask_bars) != 0
                or int(self.augmentation_max_mask_bars) != 0
                or augmentation_parameters
            ):
                raise ValueError("augmentation=none 時不可帶 augmentation 參數")
        elif self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            probability = float(self.augmentation_probability)
            protected_recent_bars = int(self.augmentation_protected_recent_bars)
            min_mask_bars = int(self.augmentation_min_mask_bars)
            max_mask_bars = int(self.augmentation_max_mask_bars)
            if not 0.0 < probability <= 1.0:
                raise ValueError("masking augmentation probability 必須介於 0 與 1 之間")
            if protected_recent_bars < 1:
                raise ValueError("masking protected_recent_bars 必須 >=1")
            if min_mask_bars < 1 or max_mask_bars < min_mask_bars:
                raise ValueError("masking bars 必須滿足 1 <= min <= max")

    def lr_schedule_parameters(self) -> dict[str, float]:
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            return {}
        return {
            "warmup_fraction": float(self.lr_warmup_fraction),
            "minimum_lr_ratio": float(self.lr_minimum_ratio),
        }

    def augmentation_parameters(self) -> dict[str, int | float]:
        if self.augmentation_name == AUGMENTATION_NONE:
            return {}
        if self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            return {
                "probability": float(self.augmentation_probability),
                "protected_recent_bars": int(self.augmentation_protected_recent_bars),
                "min_mask_bars": int(self.augmentation_min_mask_bars),
                "max_mask_bars": int(self.augmentation_max_mask_bars),
            }
        raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")

    def as_manifest_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "optimizer_name": self.optimizer_name,
            "lr_schedule_name": self.lr_schedule_name,
            "augmentation_name": self.augmentation_name,
        }
        schedule_parameters = self.lr_schedule_parameters()
        if schedule_parameters:
            payload["lr_schedule_parameters"] = schedule_parameters
        augmentation_parameters = self.augmentation_parameters()
        if augmentation_parameters:
            payload["augmentation_parameters"] = augmentation_parameters
        if self.training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS:
            payload["training_sampling_mode"] = self.training_sampling_mode
        if self.time_weight_mode is not None:
            payload["time_weight_mode"] = self.time_weight_mode
        if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
            payload["training_weight_reduction"] = self.training_weight_reduction
        if self.training_objective != TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            payload.update({
                "training_objective": self.training_objective,
                "continuous_target_id": self.continuous_target_id,
                "loss_name": self.loss_name,
                "epoch_selection_metric": self.epoch_selection_metric,
            })
            if self.raw_r_huber_delta_r is not None:
                payload["raw_r_huber_delta_r"] = float(self.raw_r_huber_delta_r)
            if self.portfolio_score_scale != "raw":
                payload["portfolio_score_scale"] = self.portfolio_score_scale
            # The event-group scope is the historical continuous-ranker default.
            # Omit it so existing MR-12 artifact identities remain byte-for-byte
            # compatible; only new non-default sample scopes are explicit.
            if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
                payload["training_sample_scope"] = self.training_sample_scope
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                payload["training_label_scope"] = self.training_label_scope
        if self.model_architecture is not None:
            payload["model_architecture"] = str(self.model_architecture)
        if self.score_calibration is not None:
            payload["score_calibration"] = self.score_calibration
        if self.fit_contract_version != 1:
            payload["fit_contract_version"] = int(self.fit_contract_version)
        if self.numerical_execution_policy != NUMERICAL_EXECUTION_POLICY_ARCHITECTURE_DEFAULT:
            payload["numerical_execution_policy"] = str(self.numerical_execution_policy)
            payload["bf16_backward_retry_scales"] = [
                float(value) for value in self.bf16_backward_retry_scales
            ]
        return payload


_EXPERIMENT_PROFILES = {
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="listnet_top_one_cross_entropy",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_FULL_D40_SAFETY_HS_CONDITIONAL_MFE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_FULL_D40_SAFETY_HS_CONDITIONAL_MFE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_TERMINAL_R40_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_TERMINAL_R40_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_terminal_r40_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_terminal_mae_gated_t40_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY,
        continuous_target_id="daily_terminal_mae_gated_t40_v1",
        loss_name="pairwise_bernoulli_probability",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_TERMINAL_R40_POSITIVE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_TERMINAL_R40_POSITIVE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SCALAR_SIGN_POSITIVE_BINARY,
        continuous_target_id="daily_terminal_r40_v1",
        loss_name="scalar_sign_positive_cross_entropy",
        epoch_selection_metric="positive_validation_cross_entropy",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_MAE_GATED_MFE_T40_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_MAE_GATED_MFE_T40_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_mae_gated_mfe_t40_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_formal_mintrade_fullcycle_netpnl_per_seed_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY,
        continuous_target_id="daily_formal_mintrade_fullcycle_netpnl_per_seed_v1",
        loss_name="netpnl_positive_cross_entropy",
        epoch_selection_metric="positive_validation_cross_entropy",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
        fit_contract_version=3,
    ),
    DAILY_MIN_P21_NETPNL_RANK_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_MIN_P21_NETPNL_RANK_PROFILE,
        optimizer_name="adam", training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=DAILY_MIN_P21_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="pairwise_logistic", epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_MIN_P21_NETPNL_PAYOFF_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_MIN_P21_NETPNL_PAYOFF_PROFILE,
        optimizer_name="adam", training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_NETPNL_PAYOFF_BINARY,
        continuous_target_id=DAILY_MIN_P21_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="netpnl_payoff_cross_entropy",
        epoch_selection_metric="payoff_validation_cross_entropy",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1", fit_contract_version=4,
    ),
    DAILY_MIN_P21_NETPNL_POSITIVE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_MIN_P21_NETPNL_POSITIVE_PROFILE,
        optimizer_name="adam", training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY,
        continuous_target_id=DAILY_MIN_P21_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="netpnl_positive_cross_entropy",
        epoch_selection_metric="positive_validation_cross_entropy",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1", fit_contract_version=1,
    ),
    DAILY_FULL_P20_NETPNL_RANK_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_FULL_P20_NETPNL_RANK_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=DAILY_FULL_P20_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="pairwise_logistic", epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
        fit_contract_version=3,
    ),
    DAILY_FULL_P20_NETPNL_POSITIVE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_FULL_P20_NETPNL_POSITIVE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY,
        continuous_target_id=DAILY_FULL_P20_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="netpnl_positive_cross_entropy",
        epoch_selection_metric="positive_validation_cross_entropy",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1", fit_contract_version=3,
    ),
    FULLTRADE_NETPNL_RANK_PROFILE: BreakoutQualityExperimentProfile(
        name=FULLTRADE_NETPNL_RANK_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="pairwise_logistic", epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
        fit_contract_version=2,
    ),
    FULLTRADE_NETPNL_POSITIVE_PROFILE: BreakoutQualityExperimentProfile(
        name=FULLTRADE_NETPNL_POSITIVE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY,
        continuous_target_id=FULLTRADE_FULLCYCLE_NETPNL_TARGET_ID,
        loss_name="netpnl_positive_cross_entropy",
        epoch_selection_metric="positive_validation_cross_entropy",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1", fit_contract_version=2,
    ),

    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_first_risk_breach_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_20bar_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
}

# AI: History extension preserves the original target and training settings.
from dataclasses import replace as _replace_study
from config.breakout_quality_history import CO_PROFILE, HISTORY_PROFILE
STOCK_EMA_PROFILE = CO_PROFILE + "_stock_ema"
CALENDAR_BARS_PROFILE = CO_PROFILE + "_calendar_bars"
for _name, _architecture in (
    (HISTORY_PROFILE, "inception_time_v1"),
    (STOCK_EMA_PROFILE, "inception_time_stock_ema_v1"),
    (CALENDAR_BARS_PROFILE, "inception_time_calendar_bars_v1"),
):
    _EXPERIMENT_PROFILES[_name] = _replace_study(
        _EXPERIMENT_PROFILES[CO_PROFILE], name=_name, model_architecture=_architecture,
    )

SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES = tuple(_EXPERIMENT_PROFILES)
SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES = tuple(
    name
    for name, profile in _EXPERIMENT_PROFILES.items()
    if profile.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
)


def normalize_breakout_quality_experiment_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _EXPERIMENT_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality experiment profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_experiment_profile(
    value: str,
) -> BreakoutQualityExperimentProfile:
    return _EXPERIMENT_PROFILES[
        normalize_breakout_quality_experiment_profile(value)
    ]


@dataclass(frozen=True)
class ContinuousRankerResearchSpec:
    profile_name: str
    model_research_id: str
    experiment_name: str
    phase: str
    trainer_family: str
    target_description: str
    objective_description: str
    metric_scope: str
    score_semantic_id: str
    pairwise_reduction: str | None = None
    pair_weight_policy: str | None = None
    secondary_pair_scope: str = CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
    secondary_pair_scope_threshold: float | None = None
    safety_supervision_source: str = CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK
    reference_profile_name: str | None = None
    evaluation_reference_profile_name: str | None = None
    model_gate_reference_profile_name: str | None = None
    # Historical/research PIT authorization.  This may remain true for archived
    # evidence that must still be readable/reconstructable.  Current OOS/Rolling
    # execution is governed separately by current_time_validation_authorized.
    selection_pit_authorized: bool = True
    current_time_validation_authorized: bool = False

    def __post_init__(self) -> None:
        if self.safety_supervision_source not in SUPPORTED_CONTINUOUS_RANKER_SAFETY_SUPERVISION_SOURCES:
            raise ValueError(f"unsupported safety supervision source: {self.safety_supervision_source!r}")
        if self.profile_name not in _EXPERIMENT_PROFILES:
            raise ValueError(f"continuous ranker research spec引用未知profile: {self.profile_name}")
        profile = _EXPERIMENT_PROFILES[self.profile_name]
        if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            raise ValueError(f"continuous ranker research spec只接受continuous profile: {self.profile_name}")
        if self.trainer_family not in {
            CONTINUOUS_RANKER_TRAINER_EVENT,
            CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        }:
            raise ValueError(f"不支援的continuous ranker trainer family: {self.trainer_family}")
        expected_family = (
            CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL
            if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
            else CONTINUOUS_RANKER_TRAINER_EVENT
        )
        if self.trainer_family != expected_family:
            raise ValueError(
                "continuous ranker trainer family與training sample scope不一致: "
                f"profile={self.profile_name}, expected={expected_family}, actual={self.trainer_family}"
            )
        training_policy = get_continuous_ranker_training_policy(
            profile.training_objective
        )
        training_policy.validate_research_spec(
            pairwise_reduction=self.pairwise_reduction,
            pair_weight_policy=self.pair_weight_policy,
            secondary_pair_scope=self.secondary_pair_scope,
            secondary_pair_scope_threshold=self.secondary_pair_scope_threshold,
        )
        for reference_field in (
            "reference_profile_name",
            "evaluation_reference_profile_name",
            "model_gate_reference_profile_name",
        ):
            reference_value = getattr(self, reference_field)
            if reference_value is None:
                continue
            reference = str(reference_value).strip()
            if reference not in _EXPERIMENT_PROFILES:
                raise ValueError(
                    f"continuous ranker {reference_field}不存在: {reference_value}"
                )
            if reference == self.profile_name:
                raise ValueError(f"continuous ranker {reference_field}不得等於自身")
        if self.current_time_validation_authorized and not self.selection_pit_authorized:
            raise ValueError(
                "current time validation authorization必須建立在PIT research authorization上: "
                f"{self.profile_name}"
            )
        for field_name in (
            "model_research_id",
            "experiment_name",
            "phase",
            "target_description",
            "objective_description",
            "metric_scope",
            "score_semantic_id",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(
                    f"continuous ranker research spec缺少{field_name}: {self.profile_name}"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "profile_name": self.profile_name,
            "model_research_id": self.model_research_id,
            "experiment_name": self.experiment_name,
            "phase": self.phase,
            "trainer_family": self.trainer_family,
            "target_description": self.target_description,
            "objective_description": self.objective_description,
            "metric_scope": self.metric_scope,
            "score_semantic_id": self.score_semantic_id,
            "pairwise_reduction": self.pairwise_reduction,
            "reference_profile_name": self.reference_profile_name,
            "evaluation_reference_profile_name": self.evaluation_reference_profile_name,
            "selection_pit_authorized": bool(self.selection_pit_authorized),
            "current_time_validation_authorized": bool(
                self.current_time_validation_authorized
            ),
        }
        if self.secondary_pair_scope != CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL:
            payload["secondary_pair_scope"] = self.secondary_pair_scope
            payload["secondary_pair_scope_threshold"] = float(self.secondary_pair_scope_threshold)
        if self.model_gate_reference_profile_name is not None:
            payload["model_gate_reference_profile_name"] = self.model_gate_reference_profile_name
        if self.safety_supervision_source != CONTINUOUS_RANKER_SAFETY_SUPERVISION_ADVERSE_TO_PEAK:
            payload["safety_supervision_source"] = self.safety_supervision_source
        return payload


_CONTINUOUS_RANKER_RESEARCH_SPECS = {
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
        model_research_id="MR-12A",
        experiment_name="MR-12A All-event No-time Continuous Ranker",
        phase="12A",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_all_event_rank_percentile_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日all-event No-time target percentile的MSE",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        model_research_id="MR-12B",
        experiment_name="MR-12B All-event No-time Pairwise Ranker",
        phase="12B",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_all_event_order_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日all-event No-time target ordering的RankNet pairwise logistic loss",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
        model_research_id="MR-12C",
        experiment_name="MR-12C All-event No-time ListNet Top-one Ranker",
        phase="12C",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_all_event_listnet_distribution_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日all-event No-time完整候選榜單的ListNet top-one cross-entropy",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
    ),
    DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
        model_research_id="MR-13C",
        experiment_name="MR-13C Daily Universal Percentile Regression",
        phase="13C",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_percentile_of_daily_opportunity_no_time_r_v1",
        objective_description="同日全部合法stock-day No-time opportunity target percentile的MSE",
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
    ),
    DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE,
        model_research_id="MR-13D",
        experiment_name="MR-13D Daily Universal Upper-tail Relevance-weighted Pairwise Ranker",
        phase="13D",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description=(
            "同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss；"
            "pair依兩端daily target percentile算術平均作upper-tail relevance權重"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    ),
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13E",
        experiment_name="MR-13E Daily Universal Full-list Delta-NDCG-weighted Pairwise Ranker",
        phase="13E",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description=(
            "同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss；"
            "pair依目前預測完整榜單交換造成的raw-percentile Delta-NDCG作權重"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13H",
        experiment_name="MR-13H Daily Universal Full-horizon No-breach Ranker",
        phase="13H",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_full_horizon_opportunity_r_v1",
        objective_description=(
            "MR-13E同一daily-universal full-list Delta-NDCG pairwise objective；"
            "唯一變更為future low觸及risk barrier不再截斷固定40D target path"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13K",
        experiment_name="MR-13K Daily Universal Full-horizon Pure-MFE Ranker",
        phase="13K",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_full_horizon_pure_mfe_r_v1",
        objective_description=(
            "MR-13H同一daily-universal full-list Delta-NDCG pairwise objective；"
            "唯一變更為adverse-to-peak仍保留診斷但不再從固定40D target扣除"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=True,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AK",
        experiment_name="MR-13AK A1 Shared-AH",
        phase="13AK",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile"
        ),
        objective_description=(
            "A0 MR-13AH的唯一control change：把external MR-13M Safety DL改成同一InceptionTime shared encoder的Raw Safety head。"
            "Safety head使用普通full-list Delta-NDCG RankNet；Pure-MFE final head只讀shared latent，不接Safety prediction。"
            "Raw Safety probability先按每個完整交易日轉成與AH相同的average-rank percentile S；"
            "MFE full-list Delta-NDCG pair weight乘detach(S_i)×detach(S_j)，pair direction仍100%由Pure-MFE決定；"
            "兩head loss固定等權平均，final runtime/model-gate score只使用MFE head。"
        ),
        metric_scope="all_stock_days_shared_safety_weighted_raw_mfe_dual_head",
        score_semantic_id="daily_shared_safety_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AO",
        experiment_name="MR-13AO True-HS Conditional-MFE Shared Ranker",
        phase="13AO",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Full-universe samples仍全部進shared InceptionTime encoder且全部監督Raw Safety head；"
            "true HS固定為same-date Safety percentile>=0.50。Conditional-MFE head只讀shared latent，"
            "且只有true-HS items先形成獨立sublist後才計算full-list Delta-NDCG RankNet；LS rows不參與MFE pair、"
            "predicted rank position、IDCG或Delta-NDCG geometry。MFE head不接predicted Safety、不使用Safety pair weight；"
            "兩head loss固定1:1。Seed42 Forward Model Gate only，不授權PIT/strategy/robustness。"
        ),
        metric_scope="full_universe_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_FULL_D40_SAFETY_HS_CONDITIONAL_MFE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_FULL_D40_SAFETY_HS_CONDITIONAL_MFE_PROFILE,
        model_research_id="MR-13CG",
        experiment_name="MR-13CG AO Full D40 Safety and True-HS Cohort Control",
        phase="13CG",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_percentile(-full_40bar_max_adverse_from_anchor/R); "
            "head2=pure_mfe_percentile_within_same_date_D40_low_adverse_true_hs_only"
        ),
        objective_description=(
            "Strict AO control with Safety truth consistently defined by the full 40-bar "
            "anchor-relative maximum adverse decline D40: the Safety head uses same-date "
            "percentile(-D40/R), and true-HS is THAT same percentile>=0.50. "
            "Conditional-MFE is ranked strictly within this D40 true-HS sublist. "
            "Keep AO adjusted 300x10 input, shared two-head InceptionTime, full-date cohort, "
            "Delta-NDCG RankNet losses 1:1, optimizer, Seed42, folds, validation, "
            "and predicted Safety P50 then Conditional-MFE inference unchanged. "
            "Scientific contrast vs CC: CC froze original adverse-to-peak true-HS; "
            "CG changes its membership. No score mixing, thresholds, strategy or Trading promotion."
        ),
        metric_scope="full_universe_full_d40_safety_plus_d40_true_hs_conditional_mfe",
        score_semantic_id="daily_d40_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        safety_supervision_source=CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_TERMINAL_R40_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_TERMINAL_R40_PAIRWISE_PROFILE,
        model_research_id="MR-13CI",
        experiment_name="MR-13CI Terminal R40 Full-list Ranker",
        phase="13CI",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same-date rank of the fixed day-40 terminal Close return in canonical R",
        objective_description=(
            "MR-13K fixed single-head InceptionTime 300x10, Adam, daily-universal "
            "Delta-NDCG pairwise, Train/Validation maturity and 40D folds; "
            "only future outcome changes from full 40D MFE to signed 40th Close return. "
            "No MAE gating, no fitting on OOS, no strategy state, no production promotion."
        ),
        metric_scope="all_stock_days_terminal_r40",
        score_semantic_id="daily_terminal_r40_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PAIRWISE_PROFILE,
        model_research_id="MR-13CJ",
        experiment_name="MR-13CJ Terminal T40 MAE-gated Full-list Ranker",
        phase="13CJ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same-date rank of T40=(1-min(MAE40_R,1))*R40-min(MAE40_R,1)",
        objective_description=(
            "Matched MR-13CI single-head 300x10 InceptionTime, full-list Delta-NDCG, "
            "Adam, split, PIT-valid fitting and common 40D cohort; only ground-truth "
            "changes to nonnegative full-path MAE-gated signed day-40 return. "
            "MAE>=1R gives -1R; neither MAE nor future R40 enter model features/inference. "
            "No OOS-fitted thresholds and no Trading promotion."
        ),
        metric_scope="all_stock_days_terminal_mae_gated_t40",
        score_semantic_id="daily_terminal_mae_gated_t40_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE,
        model_research_id="MR-13CX",
        experiment_name="MR-13CX T40 Pairwise Rank Confidence",
        phase="13CX",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "Same mature daily-universal terminal MAE-gated T40 as MR-13CJ; "
            "supervision is strict within-date T40 pair preference, never the sign of T40"
        ),
        objective_description=(
            "Single independently fitted 300x10 InceptionTime two-coordinate output. "
            "Proper Bernoulli loss of antisymmetric context-interaction comparison "
            "P(T40_i>T40_j); self-comparisons and equal/missing T40 pairs excluded. "
            "Daily score is the expected fraction of opponents beaten across the "
            "PIT score-eligible date universe. No T40 absolute-R regression, Positive "
            "head, probability gate, hand-tuned rank blend, Top-N or OOS fitting. "
            "One FIT serves Breakout/Daily Exact/Direct. CJ identity is unchanged."
        ),
        metric_scope="all_stock_days_terminal_mae_gated_t40_probabilistic_rank",
        score_semantic_id="daily_terminal_mae_gated_t40_expected_pairwise_win_fraction",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_TERMINAL_R40_POSITIVE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_TERMINAL_R40_POSITIVE_PROFILE,
        model_research_id="MR-13DL",
        experiment_name="MR-13DL Daily R40 Positive Binary Classifier",
        phase="13DL",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "Same mature daily-universal terminal R40 as MR-13CI; "
            "Positive iff raw R40>0, nonpositive iff <=0; incomplete 40D path stays NULL"
        ),
        objective_description=(
            "Independent two-class cross-entropy classifier with 300x10 InceptionTime, "
            "Adam and the existing PIT-mature stock-day splits. Outputs P(R40>0); "
            "the unchanged CI model ranks R40. No CI weights, strategy membership, "
            "NetPnL labels, payoff weights, calibration or OOS-fitted threshold. "
            "One fold/seed FIT and PIT source serves both P16 Exact and Direct."
        ),
        metric_scope="all_stock_days_terminal_r40_positive_binary",
        score_semantic_id="daily_terminal_r40_positive_probability",
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_MAE_GATED_MFE_T40_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_MAE_GATED_MFE_T40_PAIRWISE_PROFILE,
        model_research_id="MR-13CK",
        experiment_name="MR-13CK T40-MFE Full-horizon MAE-gated Ranker",
        phase="13CK",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same-date rank of (1-min(MAE40_R,1))*MFE40_R-min(MAE40_R,1)",
        objective_description=(
            "Matched MR-13CJ single-head 300x10 InceptionTime, Adam, full-list "
            "Delta-NDCG, 40D maturity and PIT fitting; only the supervised reward "
            "term changes from terminal R40 to full-horizon MFE40_R. "
            "The complete 40D MAE gate remains fixed; future MFE/MAE are labels only, "
            "never inputs at inference. Research-only; no Trading promotion."
        ),
        metric_scope="all_stock_days_full_horizon_mae_gated_mfe_t40",
        score_semantic_id="daily_full_horizon_mae_gated_mfe_t40_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE,
        model_research_id="MR-13CO",
        experiment_name="MR-13CO Formal MinTrade Full-cycle NetPnL Seed-scoped Full-list Ranker",
        phase="13CO",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "Same-seed formal Min ROOS full-cycle realized broker NetPnL; "
            "natural exit or Research-end closeout, no 40/60-day cap; "
            "unknown/unfilled are NULL, never 0"
        ),
        objective_description=(
            "Original per-seed canonical historical Optimizer + 2021+ Min P16, "
            "information-time source attestation and mature-label masking; "
            "single-head 300x10 InceptionTime, full-list Delta-NDCG pairwise. "
            "No retired MR-13CL teacher or hindsight Oracle scores enter FIT."
        ),
        metric_scope="all_stock_days_formal_mintrade_fullcycle_netpnl_per_seed",
        score_semantic_id="daily_formal_mintrade_fullcycle_netpnl_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
        model_research_id="MR-13CP",
        experiment_name="MR-13CP Full-cycle NetPnL Binary Classifier",
        phase="13CP",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "Same per-seed full-cycle matured broker NetPnL as MR-13CO; "
            "binary outcome NetPnL>0; censored NULL never creates a negative label"
        ),
        objective_description=(
            "Independent two-class NetPnL>0 classifier with categorical cross-entropy. "
            "Uses the same PIT-matured target and folds as the separate NetPnL ranker; "
            "outputs P(Positive) only, never used for NetPnL ranking."
        ),
        metric_scope="all_stock_days_formal_mintrade_fullcycle_netpnl_positive_binary",
        score_semantic_id="daily_formal_mintrade_fullcycle_netpnl_positive_probability",
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_MIN_P21_NETPNL_RANK_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_MIN_P21_NETPNL_RANK_PROFILE,
        model_research_id="MR-13DH", experiment_name="MR-13DH Daily Min/P21 broker NetPnL ranker",
        phase="13DH", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="P21 6D daily EMA+volume filtered prior-close-limit full-cycle broker NetPnL from independently optimized same-seed historical and OOS P21 teachers; no fill/unmatured NULL.",
        objective_description="Independent 300x10 full-list Delta-NDCG pairwise FIT; not P16 CU/CT, P19 DD/DE or P20 DB/DC supervised labels.",
        metric_scope="all_stock_days_daily_min_p21_fullcycle_netpnl",
        score_semantic_id="daily_min_p21_fullcycle_netpnl_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),
    DAILY_MIN_P21_NETPNL_PAYOFF_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_MIN_P21_NETPNL_PAYOFF_PROFILE,
        model_research_id="MR-13DJ", experiment_name="MR-13DJ Daily Min/P21 Positive learned decision",
        phase="13DJ", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="Same PIT-matured same-seed P21 broker NetPnL as DH/DI; no fill and unmatured outcomes stay NULL.",
        objective_description="Independent 300x10 InceptionTime binary CE weighted by absolute NetPnL; FIT-only mean weight normalization. Two logits learn trade/cash decision; no upstream DL inputs or threshold search.",
        metric_scope="all_stock_days_daily_min_p21_fullcycle_netpnl_payoff_binary",
        score_semantic_id="daily_min_p21_positive_payoff_mass_share_and_trade_decision",
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),
    DAILY_MIN_P21_NETPNL_POSITIVE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_MIN_P21_NETPNL_POSITIVE_PROFILE,
        model_research_id="MR-13DI", experiment_name="MR-13DI Daily Min/P21 NetPnL Positive",
        phase="13DI", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="Same PIT-matured P21 raw broker NetPnL as DH; class 1 iff NetPnL>0, class 0 otherwise; no-fill/unmatured NULL.",
        objective_description="Independent two-class cross-entropy P(NetPnL>0) classifier; admission only, not Rank.",
        metric_scope="all_stock_days_daily_min_p21_fullcycle_netpnl_positive_binary",
        score_semantic_id="daily_min_p21_fullcycle_netpnl_positive_probability",
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),
    DAILY_FULL_P20_NETPNL_RANK_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_FULL_P20_NETPNL_RANK_PROFILE,
        model_research_id="MR-13DB",
        experiment_name="MR-13DB Daily Full/P20 broker NetPnL ranker",
        phase="13DB", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="Every eligible Daily stock-day under the PIT P20 19D Full execution parameter schedule: realized broker NetPnL at natural exit, NULL for unfilled, excluded or unmatured outcomes.",
        objective_description="Independent 300x10 full-list Delta-NDCG pairwise Rank FIT against P20 Daily Full outcomes; P17 CZ and P18 CU labels are not interchangeable.",
        metric_scope="all_stock_days_daily_full_p20_fullcycle_netpnl",
        score_semantic_id="daily_full_p20_fullcycle_netpnl_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),
    DAILY_FULL_P20_NETPNL_POSITIVE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_FULL_P20_NETPNL_POSITIVE_PROFILE,
        model_research_id="MR-13DC",
        experiment_name="MR-13DC Daily Full/P20 NetPnL>0 classifier",
        phase="13DC", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="Identical PIT P20 full-cycle raw broker NetPnL as MR-13DB; positive iff settled NetPnL>0; no-fill/unmatured=NULL.",
        objective_description="Independent two-class cross-entropy P(NetPnL>0) FIT; score used only for Positive admission at 0.5, never as Rank.",
        metric_scope="all_stock_days_daily_full_p20_fullcycle_netpnl_positive_binary",
        score_semantic_id="daily_full_p20_fullcycle_netpnl_positive_probability",
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),
    FULLTRADE_NETPNL_RANK_PROFILE: ContinuousRankerResearchSpec(
        profile_name=FULLTRADE_NETPNL_RANK_PROFILE,
        model_research_id="MR-13CZ",
        experiment_name="MR-13CZ Full P17 realized broker NetPnL ranker",
        phase="13CZ", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="Same-seed Full/P17 historical + 2021+ Optimizer; broker-cash NetPnL at natural exit or legal terminal closeout, no fixed-day cap. Unfilled and unmatured are NULL.",
        objective_description="300x10 InceptionTime, full-list Delta-NDCG pairwise; original Full/P17 DL-off Optimizer teacher and PIT-matured outcomes, no Oracle truth in FIT.",
        metric_scope="all_stock_days_formal_fulltrade_netpnl_per_seed",
        score_semantic_id="daily_formal_fulltrade_fullcycle_netpnl_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),
    FULLTRADE_NETPNL_POSITIVE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=FULLTRADE_NETPNL_POSITIVE_PROFILE,
        model_research_id="MR-13DA",
        experiment_name="MR-13DA Full P17 NetPnL>0 classifier",
        phase="13DA", trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="Same PIT-matured per-seed Full/P17 NetPnL as MR-13CZ; NetPnL>0 positive, censored/unmatured NULL.",
        objective_description="Separate two-class cross-entropy probability predictor; never replaces NetPnL ranking. Same historical Full/P17 teacher and matured cohort as CZ.",
        metric_scope="all_stock_days_formal_fulltrade_netpnl_positive_binary",
        score_semantic_id="daily_formal_fulltrade_fullcycle_netpnl_positive_probability",
        selection_pit_authorized=True, current_time_validation_authorized=True,
    ),

    DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AB",
        experiment_name="MR-13AB Daily Universal Pure-MFE Before First Risk Breach Ranker",
        phase="13AB",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_first_risk_breach_pure_mfe_r_v1",
        objective_description=(
            "MR-13K exact controlled target contrast：daily-universal universe、InceptionTime、"
            "full-list Delta-NDCG pairwise objective、Seed42、split、optimizer、epoch selection、"
            "training/sample scope與40D fixed R scale全部不變；唯一scientific dimension為"
            "Pure-MFE future path沿用MR-13E same-bar adverse-first first-risk-breach truncation。"
            "adverse-to-peak只保留diagnostic，不從target扣除；首根即breach為0R。"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_first_risk_breach_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_FULL_D20_SAFETY_HS_CONDITIONAL_MFE_PROFILE,
        model_research_id="MR-13CH",
        experiment_name="MR-13CH CG Fixed 20-bar MFE20 / D20 / True-HS Contrast",
        phase="13CH",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_percentile(-full_20bar_max_adverse_from_anchor/R); "
            "head2=pure_MFE20_percentile_within_same_date_D20_low_adverse_true_hs_only"
        ),
        objective_description=(
            "Only label horizon changes from CG's 40 trading bars to 20: "
            "MFE20, full-path D20, D20 P50 True-HS and HS-scoped conditional MFE20. "
            "Preserve CG's adjusted 300x10 stock+0050 inputs, shared InceptionTime two-head "
            "topology, identical losses/optimizer/seed/cutoff/fold cadence and MFE-head inference. "
            "For strict paired AO/CG U40 comparison, retain the same conservative "
            "40-bar complete stock-day training/evaluation universe and conservative "
            "40-bar information-availability maturity for PIT fitting, even though "
            "MFE20/D20 extrema use strictly the first 20 future bars. This prevents "
            "unavailable post-20-bar path eligibility from entering the 20-bar fit. "
            "U20/U40 metrics are read-only evaluation. "
            "No candidate conditioning, OOS fitting, or Trading promotion."
        ),
        metric_scope="daily_full_d20_safety_plus_d20_true_hs_conditional_mfe",
        score_semantic_id="daily_d20_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        safety_supervision_source=CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_FULL_D40_SAFETY_HS_CONDITIONAL_MFE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
}

for _name, _id, _phase, _description in (
    (HISTORY_PROFILE, "MR-13DN", "13DN", "CO controlled early real-event history extension"),
    (STOCK_EMA_PROFILE, "MR-13DO", "13DO", "CO plus full-history stock close EMA inputs"),
    (CALENDAR_BARS_PROFILE, "MR-13DP", "13DP", "CO plus closed weekly and monthly OHLCV branches"),
):
    _CONTINUOUS_RANKER_RESEARCH_SPECS[_name] = _replace_study(
        _CONTINUOUS_RANKER_RESEARCH_SPECS[CO_PROFILE],
        profile_name=_name, model_research_id=_id, phase=_phase,
        experiment_name=_description, objective_description=_description,
    )

_CONTINUOUS_RANKER_RESEARCH_SPECS[STOCK_EMA_PROFILE] = _replace_study(
    _CONTINUOUS_RANKER_RESEARCH_SPECS[STOCK_EMA_PROFILE],
    model_gate_reference_profile_name=CO_PROFILE,
)

_CONTINUOUS_RANKER_RESEARCH_SPECS[CALENDAR_BARS_PROFILE] = _replace_study(
    _CONTINUOUS_RANKER_RESEARCH_SPECS[CALENDAR_BARS_PROFILE],
    model_gate_reference_profile_name=CO_PROFILE,
)


def _continuous_ranker_profile_names_from_experiment_registry() -> tuple[str, ...]:
    """Return continuous-profile membership/order from the canonical profile owner."""

    return tuple(
        name
        for name, profile in _EXPERIMENT_PROFILES.items()
        if profile.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES
    )


def _validate_continuous_ranker_research_spec_attachments() -> tuple[str, ...]:
    """Fail fast when continuous research metadata drifts from its profile owner.

    ``_EXPERIMENT_PROFILES`` owns supported profile membership and order.  The
    research-spec registry is a keyed metadata attachment only; its insertion
    order is intentionally non-semantic and must never become a second supported
    profile list.
    """

    owner_profiles = _continuous_ranker_profile_names_from_experiment_registry()
    owner_set = set(owner_profiles)
    attachment_keys = tuple(_CONTINUOUS_RANKER_RESEARCH_SPECS)
    attachment_set = set(attachment_keys)

    missing = tuple(name for name in owner_profiles if name not in attachment_set)
    extra = tuple(name for name in attachment_keys if name not in owner_set)
    key_mismatches = tuple(
        key
        for key, spec in _CONTINUOUS_RANKER_RESEARCH_SPECS.items()
        if spec.profile_name != key
    )

    seen_research_ids: set[str] = set()
    duplicate_research_ids: list[str] = []
    for spec in _CONTINUOUS_RANKER_RESEARCH_SPECS.values():
        if spec.model_research_id in seen_research_ids:
            duplicate_research_ids.append(spec.model_research_id)
        else:
            seen_research_ids.add(spec.model_research_id)

    if missing or extra or key_mismatches or duplicate_research_ids:
        details = []
        if missing:
            details.append(f"missing_specs={missing}")
        if extra:
            details.append(f"extra_specs={extra}")
        if key_mismatches:
            details.append(f"key_profile_mismatches={key_mismatches}")
        if duplicate_research_ids:
            details.append(
                "duplicate_model_research_ids="
                f"{tuple(sorted(set(duplicate_research_ids)))}"
            )
        raise RuntimeError(
            "continuous ranker profile/research-spec registry contract drift: "
            + "; ".join(details)
        )

    return owner_profiles


SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES = (
    _validate_continuous_ranker_research_spec_attachments()
)


def get_continuous_ranker_research_spec(
    experiment_profile: str,
) -> ContinuousRankerResearchSpec:
    profile_name = normalize_breakout_quality_experiment_profile(experiment_profile)
    spec = _CONTINUOUS_RANKER_RESEARCH_SPECS.get(profile_name)
    if spec is None:
        raise ValueError(
            f"continuous ranker profile缺少research spec登記: {profile_name}"
        )
    return spec
