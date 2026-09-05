"""Breakout-quality scientific/profile registry and immutable contract schemas.

This module owns scientific profile identifiers, experiment/pretraining declarations,
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
    PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
    PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
    PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_ADAPTIVE_HORIZON_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
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
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
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


def merge_breakout_quality_model_profiles(
    *groups: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Merge ordered model/profile groups while rejecting identity collisions.

    Exact duplicates are intentionally de-duplicated.  Reusing one MR id for a different
    profile, or one profile under a different MR id, is a configuration error and fails
    before any training/comparison workflow can run.
    """

    rows: list[tuple[str, str]] = []
    by_model_id: dict[str, str] = {}
    by_profile: dict[str, str] = {}
    for group in groups:
        for raw_model_id, raw_profile in group:
            model_id = str(raw_model_id).strip()
            profile = str(raw_profile).strip()
            if not model_id or not profile:
                raise ValueError("模型研究model id/profile不得為空")
            existing_profile = by_model_id.get(model_id)
            existing_model_id = by_profile.get(profile)
            if existing_profile is not None and existing_profile != profile:
                raise ValueError(
                    f"模型研究model id重複綁定不同profile: {model_id} -> "
                    f"{existing_profile!r} / {profile!r}"
                )
            if existing_model_id is not None and existing_model_id != model_id:
                raise ValueError(
                    f"模型研究profile重複綁定不同model id: {profile} -> "
                    f"{existing_model_id!r} / {model_id!r}"
                )
            if existing_profile is not None:
                continue
            by_model_id[model_id] = profile
            by_profile[profile] = model_id
            rows.append((model_id, profile))
    return tuple(rows)


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

TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE = "ts2vec_selection_only"
STOCK_CODE_CLASSIFICATION_ENCODER_PRETRAINING_PROFILE = "stock_code_classification_v1"
FUTURE_PATH_STATISTICS_ENCODER_PRETRAINING_PROFILE = "future_path_statistics_v1"
FUTURE_PATH_STATISTICS_TARGET_NAMES = (
    "terminal_return_5d", "terminal_return_10d", "terminal_return_20d", "terminal_return_40d",
    "max_downside_10d", "max_downside_20d", "max_downside_40d",
    "max_upside_10d", "max_upside_20d", "max_upside_40d",
    "realized_volatility_10d", "realized_volatility_20d",
)

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
    loss_name: str = "cross_entropy"
    epoch_selection_metric: str = "validation_loss"
    training_label_scope: str = TRAINING_LABEL_SCOPE_ALL
    training_sample_scope: str = TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    raw_r_huber_delta_r: float | None = None
    model_architecture: str | None = None
    encoder_pretraining_profile: str | None = None
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
        if self.encoder_pretraining_profile is not None:
            get_breakout_quality_encoder_pretraining_profile(
                self.encoder_pretraining_profile
            )
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
            # The event-group scope is the historical continuous-ranker default.
            # Omit it so existing MR-12 artifact identities remain byte-for-byte
            # compatible; only new non-default sample scopes are explicit.
            if self.training_sample_scope != TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
                payload["training_sample_scope"] = self.training_sample_scope
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                payload["training_label_scope"] = self.training_label_scope
        if self.model_architecture is not None:
            payload["model_architecture"] = str(self.model_architecture)
        if self.encoder_pretraining_profile is not None:
            payload["encoder_pretraining"] = (
                get_breakout_quality_encoder_pretraining_profile(
                    self.encoder_pretraining_profile
                ).as_manifest_payload()
            )
        if self.numerical_execution_policy != NUMERICAL_EXECUTION_POLICY_ARCHITECTURE_DEFAULT:
            payload["numerical_execution_policy"] = str(self.numerical_execution_policy)
            payload["bf16_backward_retry_scales"] = [
                float(value) for value in self.bf16_backward_retry_scales
            ]
        return payload



@dataclass(frozen=True)
class BreakoutQualityPretrainingProfile:
    name: str
    family: str
    optimizer_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    min_crop_bars: int
    mask_probability: float
    contrastive_alpha: float
    temporal_unit: int

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("pretraining profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining profile name 必須是安全的單一資料夾名稱")
        if str(self.family).strip().lower() != self.family or not self.family:
            raise ValueError("pretraining family 必須是非空白小寫名稱")
        if any(token in self.family for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining family 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 pretraining optimizer: {self.optimizer_name!r}")
        if int(self.epochs) < 1 or int(self.batch_size) < 2:
            raise ValueError("pretraining epochs 必須 >=1 且 batch_size 必須 >=2")
        if (
            float(self.learning_rate) <= 0.0
            or float(self.weight_decay) < 0.0
            or float(self.gradient_clip_norm) < 0.0
        ):
            raise ValueError(
                "pretraining learning_rate 必須 >0，weight_decay與gradient_clip_norm必須 >=0"
            )
        if int(self.min_crop_bars) < 2 or int(self.temporal_unit) < 0:
            raise ValueError("pretraining min_crop_bars 必須 >=2 且 temporal_unit 必須 >=0")
        if not 0.0 <= float(self.mask_probability) < 1.0:
            raise ValueError("pretraining mask_probability 必須介於0（含）與1（不含）")
        if not 0.0 <= float(self.contrastive_alpha) <= 1.0:
            raise ValueError("pretraining contrastive_alpha 必須介於0與1")

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "optimizer_name": self.optimizer_name,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "min_crop_bars": int(self.min_crop_bars),
            "mask_probability": float(self.mask_probability),
            "contrastive_alpha": float(self.contrastive_alpha),
            "temporal_unit": int(self.temporal_unit),
        }


_PRETRAINING_PROFILES = {
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE: BreakoutQualityPretrainingProfile(
        name=TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
        family="ts2vec_v1",
        optimizer_name="adamw",
        epochs=10,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        min_crop_bars=60,
        mask_probability=0.5,
        contrastive_alpha=0.5,
        temporal_unit=0,
    ),
}
SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES = tuple(_PRETRAINING_PROFILES)


def normalize_breakout_quality_pretraining_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _PRETRAINING_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality pretraining profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_pretraining_profile(
    value: str,
) -> BreakoutQualityPretrainingProfile:
    return _PRETRAINING_PROFILES[normalize_breakout_quality_pretraining_profile(value)]


def build_breakout_quality_pretraining_profile_payload(
    value: str,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    gradient_clip_norm: float | None = None,
    min_crop_bars: int | None = None,
    mask_probability: float | None = None,
    contrastive_alpha: float | None = None,
    temporal_unit: int | None = None,
) -> dict[str, Any]:
    """Return the named profile payload with explicit CLI overrides applied.

    Formal workflow runs use the profile defaults. The override path remains available for
    isolated development experiments, while downstream canonical training can reject an
    encoder whose stored payload differs from the active named profile.
    """

    profile = get_breakout_quality_pretraining_profile(value)
    resolved = BreakoutQualityPretrainingProfile(
        name=profile.name,
        family=profile.family,
        optimizer_name=profile.optimizer_name,
        epochs=profile.epochs if epochs is None else int(epochs),
        batch_size=profile.batch_size if batch_size is None else int(batch_size),
        learning_rate=(
            profile.learning_rate if learning_rate is None else float(learning_rate)
        ),
        weight_decay=profile.weight_decay if weight_decay is None else float(weight_decay),
        gradient_clip_norm=(
            profile.gradient_clip_norm
            if gradient_clip_norm is None
            else float(gradient_clip_norm)
        ),
        min_crop_bars=(
            profile.min_crop_bars if min_crop_bars is None else int(min_crop_bars)
        ),
        mask_probability=(
            profile.mask_probability
            if mask_probability is None
            else float(mask_probability)
        ),
        contrastive_alpha=(
            profile.contrastive_alpha
            if contrastive_alpha is None
            else float(contrastive_alpha)
        ),
        temporal_unit=profile.temporal_unit if temporal_unit is None else int(temporal_unit),
    )
    return resolved.as_manifest_payload()


@dataclass(frozen=True)
class BreakoutQualityEncoderPretrainingProfile:
    """Formal encoder-initialization recipe for downstream ranker experiments.

    This is deliberately separate from the legacy TS2Vec artifact profile above.
    Encoder pretraining here is part of the downstream experiment identity and is
    rebuilt inside each fitting scope; it is not a reusable cross-fold checkpoint.
    """

    name: str
    task: str
    optimizer_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    sampling_mode: str
    source_scope: str
    downstream_finetune: str
    loss_name: str = "cross_entropy"
    target_names: tuple[str, ...] = ()
    target_standardization: str | None = None
    smooth_l1_beta: float | None = None
    future_path_max_horizon_bars: int | None = None

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("encoder pretraining profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("encoder pretraining profile name 必須是安全名稱")
        supported_tasks = {"stock_code_classification", "future_path_statistics_regression"}
        if self.task not in supported_tasks:
            raise ValueError(f"不支援的 encoder pretraining task: {self.task!r}")
        if self.optimizer_name != "adam":
            raise ValueError("encoder pretraining 固定使用 Adam")
        if int(self.epochs) < 1 or int(self.batch_size) < 2:
            raise ValueError("encoder pretraining epochs 必須>=1且batch_size必須>=2")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("encoder pretraining learning_rate 必須>0")
        if float(self.weight_decay) < 0.0 or float(self.gradient_clip_norm) < 0.0:
            raise ValueError("encoder pretraining weight_decay/gradient_clip_norm 必須>=0")
        if self.sampling_mode != "ticker_balanced_one_window_per_ticker_per_epoch":
            raise ValueError("encoder pretraining sampling mode 不符合scientific contract")
        if self.downstream_finetune != "full_unfrozen":
            raise ValueError("encoder pretraining downstream 必須full-unfrozen fine-tune")
        if self.task == "stock_code_classification":
            if self.source_scope != "downstream_fitting_rows_only":
                raise ValueError("SCC encoder pretraining source scope 必須限制於downstream fitting rows")
            if self.loss_name != "cross_entropy":
                raise ValueError("SCC encoder pretraining 必須使用cross_entropy")
            if (
                self.target_names
                or self.target_standardization is not None
                or self.smooth_l1_beta is not None
                or self.future_path_max_horizon_bars is not None
            ):
                raise ValueError("SCC encoder pretraining 不得宣告future-path regression設定")
        else:
            expected_targets = FUTURE_PATH_STATISTICS_TARGET_NAMES
            if self.source_scope != "matured_downstream_fitting_rows_only":
                raise ValueError("future-path pretraining source scope 必須限制於fitting cutoff前已成熟rows")
            if self.loss_name != "smooth_l1":
                raise ValueError("future-path encoder pretraining 必須使用smooth_l1")
            if tuple(self.target_names) != expected_targets:
                raise ValueError("future-path encoder pretraining target set不符合scientific contract")
            if self.target_standardization != "fitting_scope_zscore":
                raise ValueError("future-path encoder pretraining 必須使用fitting-scope z-score")
            if self.smooth_l1_beta is None or not math.isfinite(float(self.smooth_l1_beta)) or float(self.smooth_l1_beta) <= 0.0:
                raise ValueError("future-path encoder pretraining smooth_l1_beta必須為有限正數")
            if int(self.future_path_max_horizon_bars or 0) != 40:
                raise ValueError("future-path encoder pretraining max horizon必須為40 bars")

    def as_manifest_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "task": self.task,
            "optimizer_name": self.optimizer_name,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "sampling_mode": self.sampling_mode,
            "source_scope": self.source_scope,
            "downstream_finetune": self.downstream_finetune,
        }
        if self.task == "future_path_statistics_regression":
            payload.update({
                "loss_name": self.loss_name,
                "target_names": list(self.target_names),
                "target_standardization": self.target_standardization,
                "smooth_l1_beta": float(self.smooth_l1_beta),
                "future_path_max_horizon_bars": int(self.future_path_max_horizon_bars),
            })
        return payload


_ENCODER_PRETRAINING_PROFILES = {
    STOCK_CODE_CLASSIFICATION_ENCODER_PRETRAINING_PROFILE: BreakoutQualityEncoderPretrainingProfile(
        name=STOCK_CODE_CLASSIFICATION_ENCODER_PRETRAINING_PROFILE,
        task="stock_code_classification",
        optimizer_name="adam",
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
        sampling_mode="ticker_balanced_one_window_per_ticker_per_epoch",
        source_scope="downstream_fitting_rows_only",
        downstream_finetune="full_unfrozen",
    ),
    FUTURE_PATH_STATISTICS_ENCODER_PRETRAINING_PROFILE: BreakoutQualityEncoderPretrainingProfile(
        name=FUTURE_PATH_STATISTICS_ENCODER_PRETRAINING_PROFILE,
        task="future_path_statistics_regression",
        optimizer_name="adam",
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
        sampling_mode="ticker_balanced_one_window_per_ticker_per_epoch",
        source_scope="matured_downstream_fitting_rows_only",
        downstream_finetune="full_unfrozen",
        loss_name="smooth_l1",
        target_names=FUTURE_PATH_STATISTICS_TARGET_NAMES,
        target_standardization="fitting_scope_zscore",
        smooth_l1_beta=1.0,
        future_path_max_horizon_bars=40,
    ),
}
SUPPORTED_BREAKOUT_QUALITY_ENCODER_PRETRAINING_PROFILES = tuple(
    _ENCODER_PRETRAINING_PROFILES
)


def get_breakout_quality_encoder_pretraining_profile(
    value: str,
) -> BreakoutQualityEncoderPretrainingProfile:
    normalized = str(value).strip().lower()
    try:
        return _ENCODER_PRETRAINING_PROFILES[normalized]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_ENCODER_PRETRAINING_PROFILES)
        raise ValueError(
            f"不支援的 encoder pretraining profile: {value!r}；可用值: {allowed}"
        ) from exc

_EXPERIMENT_PROFILES = {
    BASELINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=BASELINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
    ),
    ADAMW_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAMW_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adamw",
    ),
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        lr_schedule_name=LR_SCHEDULE_LINEAR_WARMUP_COSINE,
        lr_warmup_fraction=0.05,
        lr_minimum_ratio=0.10,
    ),
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        augmentation_name=AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
        augmentation_probability=0.50,
        augmentation_protected_recent_bars=60,
        augmentation_min_mask_bars=10,
        augmentation_max_mask_bars=30,
    ),
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    ),
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        time_weight_mode=TIME_WEIGHT_MODE_DATE_BALANCED,
        training_weight_reduction=TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    ),
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
    ),
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
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
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
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
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="patch_token_transformer_ranker_v1",
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
    DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_predicted_upside_context_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_predicted_safety_context_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id=PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_predicted_safety_context_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="dual_mse_raw_r",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_low_adverse_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_BIGRU_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_BIGRU_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_low_adverse_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="bigru_ranker_v1",
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_equal_rank_mfe_low_adverse_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_pareto_pair_concordance",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="conditional_safety_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_conditional_mfe_safety_v1",
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
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
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_opportunity_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_conditional_mfe_v1",
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
    DAILY_UNIVERSAL_ADAPTIVE_HORIZON_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_ADAPTIVE_HORIZON_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_ADAPTIVE_HORIZON_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="adaptive_horizon_safety_aux_bce_plus_dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_adaptive_horizon_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_ADAPTIVE_INPUT_CONTEXT_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_ADAPTIVE_INPUT_CONTEXT_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_adaptive_input_context_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_dynamic_hypergraph_mfe_v1",
    ),
    DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_RELATION_CHANGE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_RELATION_CHANGE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_dynamic_hypergraph_relation_change_mfe_v1",
    ),
    DAILY_UNIVERSAL_SCC_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SCC_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
        encoder_pretraining_profile=STOCK_CODE_CLASSIFICATION_ENCODER_PRETRAINING_PROFILE,
    ),
    DAILY_UNIVERSAL_FUTURE_PATH_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_FUTURE_PATH_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
        encoder_pretraining_profile=FUTURE_PATH_STATISTICS_ENCODER_PRETRAINING_PROFILE,
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_price_volume_structure_v1",
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_LOCAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_LOCAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_price_volume_structure_local_v1",
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PRICE_VOLUME_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_price_volume_multiscale_v1",
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_price_volume_position_aware_multiscale_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_priority_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_priority_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_task_specific_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_attn_mfe_v1",
    ),
    DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_task_specific_safety_attn_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_SELF_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_SELF_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_self_attn_mfe_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_PAIRWISE_RELATION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_PAIRWISE_RELATION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_pairwise_relation_mfe_v1",
    ),

DAILY_UNIVERSAL_PATCH_SAFETY_INCEPTION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
    name=DAILY_UNIVERSAL_PATCH_SAFETY_INCEPTION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    optimizer_name="adam",
    training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
    loss_name="dual_head_pairwise_logistic",
    epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
    training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    model_architecture="patch_transformer_safety_inception_mfe_v1",
),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_WINDOW_RF_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_WINDOW_RF_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_full_window_rf_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_wide_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_600bar_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_DEEP_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_DEEP_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_deep_v1",
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_shared_safety_mfe_600bar_wide_v1",
    ),
    DAILY_UNIVERSAL_DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="day_token_transformer_shared_safety_mfe_v1",
    ),
    DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="gru_shared_safety_mfe_v2",
    ),
    DAILY_UNIVERSAL_BIGRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_BIGRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="gru_shared_safety_mfe_v4",
    ),
    DAILY_UNIVERSAL_GRU_BF16_GUARDED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_GRU_BF16_GUARDED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="gru_shared_safety_mfe_v3",
    ),
    DAILY_UNIVERSAL_GRU_BF16_BACKWARD_SCALED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_GRU_BF16_BACKWARD_SCALED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="dual_head_pairwise_logistic",
        epoch_selection_metric="hs_conditional_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="gru_shared_safety_mfe_v3",
        numerical_execution_policy=NUMERICAL_EXECUTION_POLICY_BF16_DYNAMIC_BACKWARD_SCALING,
        bf16_backward_retry_scales=(
            0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ),
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_hmhs_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_hmhs_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_hmhs_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_safety_raw_mfe_joint_attn_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="modern_tcn_safety_raw_mfe_joint_attn_mlp_v1",
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="tri_head_pairwise_logistic",
        epoch_selection_metric="raw_mfe_mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1",
    ),
    DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
        continuous_target_id="daily_full_horizon_pure_mfe_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="hmhs_pairwise_concordance",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_v1",
    ),
    DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="huber_raw_r",
        epoch_selection_metric="validation_huber_raw_r",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        raw_r_huber_delta_r=1.0,
    ),
    DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        continuous_target_id="daily_opportunity_no_time_r_v1",
        loss_name="mse_raw_r",
        epoch_selection_metric="validation_mse_raw_r",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_risk_normalized_net_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    ),
    DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="daily_risk_normalized_net_opportunity_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
        training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        model_architecture="inception_time_risk_context_v1",
    ),
}

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
    reference_profile_name: str | None = None
    evaluation_reference_profile_name: str | None = None
    model_gate_reference_profile_name: str | None = None
    # Historical/research PIT authorization.  This may remain true for archived
    # evidence that must still be readable/reconstructable.  Current OOS/Rolling
    # execution is governed separately by current_time_validation_authorized.
    selection_pit_authorized: bool = True
    current_time_validation_authorized: bool = False

    def __post_init__(self) -> None:
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
        return payload


_CONTINUOUS_RANKER_RESEARCH_SPECS = {
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        model_research_id="MR-11B",
        experiment_name="11B Strategy-aligned Daily Percentile Ranker",
        phase="11B",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_rank_percentile_of_strategy_aligned_opportunity_r_v1",
        objective_description="同日11A target percentile的MSE",
        metric_scope="all_labels",
        score_semantic_id="opportunity_rank",
    ),
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        model_research_id="MR-11G",
        experiment_name="11G PASS-conditional No-time Magnitude Ranker",
        phase="11G",
        trainer_family=CONTINUOUS_RANKER_TRAINER_EVENT,
        target_description="same_date_pass_only_rank_percentile_of_strategy_aligned_opportunity_no_time_r_v1",
        objective_description="同日PASS-only No-time target percentile的MSE",
        metric_scope="pass_only",
        score_semantic_id="opportunity_rank",
    ),
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
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        model_research_id="MR-13A",
        experiment_name="MR-13A Daily Universal No-time Pairwise Ranker",
        phase="13A",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description="同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss",
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    ),
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
        model_research_id="MR-13B",
        experiment_name="MR-13B Daily Universal Target-gap-weighted Pairwise Ranker",
        phase="13B",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_opportunity_no_time_r_v1",
        objective_description=(
            "同日全部合法stock-day No-time target ordering的RankNet pairwise logistic loss；"
            "pair依daily target percentile距離加權並於date內正規化"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
        selection_pit_authorized=False,
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
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AA",
        experiment_name="MR-13AA MR-13H Target + Frozen Patch Transformer Architecture Control",
        phase="13AA",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_full_horizon_opportunity_r_v1",
        objective_description=(
            "MR-13H exact controlled architecture contrast：target、daily-universal universe、"
            "full-list Delta-NDCG pairwise objective、Seed42、split、optimizer、epoch selection、"
            "training/sample scope全部不變；唯一scientific dimension為temporal architecture由"
            "InceptionTime換成MR-13Z已freeze的historical-recipe Patch Transformer。"
            "Patch recipe固定non-overlap patch=10、embedding=128、depth=3、heads=4、FFN=256、"
            "sinusoidal position、LayerNorm、dropout=0.10與mean pooling；不做architecture tuning。"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
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
    DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AC",
        experiment_name="MR-13AC PIT-safe Predicted-Upside Conditional Low-Adverse Ranker",
        phase="13AC",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "same_date_percentile_of_low_adverse_residual_given_PIT_safe_predicted_pure_mfe_percentile"
        ),
        objective_description=(
            "Plan C-M controlled Model Gate：Stage-1固定MR-13K Pure-MFE ranker。Selection context只允許expanding cross-fitted/PIT-safe same-day predicted-upside percentile；"
            "因此Stage-2 Selection training universe只使用已有合法PIT context覆蓋的rows，不以前視或full-fit score回填較早rows。"
            "Forward OOS context固定單一pre-2021 Stage-1 fit，不在OOS期間refit。Stage-2固定MR-13M InceptionTime/full-list Delta-NDCG/Seed42/split/optimizer，"
            "輸入只新增1個PIT-safe upside percentile scalar；label為同日Low-Adverse percentile對該predicted-upside percentile含intercept OLS residual後再同日percentile化。"
            "不讀full-fit Selection score、不使用OOS統計、portfolio state、threshold、lambda、score fusion或backbone tuning。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_predicted_upside_conditional_low_adverse_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AD",
        experiment_name="MR-13AD PIT-safe Predicted-Safety Conditional MFE Reverse-Control",
        phase="13AD",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "same_date_percentile_of_pure_mfe_residual_given_PIT_safe_predicted_low_adverse_safety_percentile"
        ),
        objective_description=(
            "MR-13AC methodology reverse-control：Stage-1固定MR-13M Low-Adverse ranker。Selection context只允許expanding cross-fitted/PIT-safe same-day predicted-safety percentile；"
            "Stage-2 Selection training universe只使用已有合法PIT context覆蓋的rows，不以前視或full-fit score回填。Forward OOS context固定單一pre-2021 Stage-1 fit。"
            "Stage-2固定MR-13K InceptionTime/full-list Delta-NDCG/Seed42/split/optimizer/epoch-selection，輸入只新增1個PIT-safe safety percentile scalar；"
            "label為同日Pure-MFE percentile對predicted-safety percentile含intercept OLS residual後再同日percentile化。"
            "Model Gate同時評估own-target learnability、Pure-MFE/upside與Low-Adverse/downside evidence；不使用portfolio state、threshold、lambda、score fusion或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_predicted_safety_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AE",
        experiment_name="MR-13AE PIT-safe Predicted-Safety Context Pure-MFE Control",
        phase="13AE",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_PIT_safe_predicted_safety_context",
        objective_description=(
            "MR-13AD follow-up controlled cell：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2回復MR-13K canonical full-horizon Pure-MFE target/order，不做conditional residualization。"
            "MR-13K InceptionTime/full-list Delta-NDCG/Seed42/split/optimizer/epoch-selection/training scope全部固定，"
            "唯一scientific change是GAP latent後direct concat 1個PIT-safe predicted-safety percentile scalar。"
            "Selection只使用cross-fitted context-covered rows，Forward只使用single fixed pre-OOS context；"
            "Model Gate先驗證Pure-MFE learnability是否接近MR-13K，再檢查高分股Adverse/High-Safety是否改善；"
            "不使用portfolio state、threshold、lambda、fusion、product或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_predicted_safety_context_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AF",
        experiment_name="MR-13AF PIT-safe High-Safety-Weighted Pure-MFE Ranker",
        phase="13AF",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_PIT_safe_predicted_safety_pair_weighting",
        objective_description=(
            "MR-13AE anti-shortcut controlled cell：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2回復MR-13K inception_time_v1，不把Safety scalar輸入network，Pure-MFE target/order完全不變。"
            "MR-13K full-list Delta-NDCG pair relevance乘上min(S_i,S_j)，S為同日PIT-safe predicted-Safety percentile；"
            "最終loss以所有同日comparable pairs做normalized weighted mean，所以只改relative supervision importance、不改loss scale。"
            "高Safety×高Safety的MFE比較保留高權重；任何含低Safety股票的pair自然降權。"
            "沒有bucket、Safety cutoff、lambda、temperature、joint head、residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_high_safety_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_WINNER_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AG",
        experiment_name="MR-13AG PIT-safe MFE-Winner-Safety-Weighted Pure-MFE Ranker",
        phase="13AG",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_MFE_winner_predicted_safety_pair_weighting",
        objective_description=(
            "MR-13AF directional anti-shortcut follow-up：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2固定MR-13K inception_time_v1，Safety不進network，Pure-MFE target/order與full-list Delta-NDCG relevance完全不變。"
            "每個同日non-tied MFE pair只將Delta-NDCG乘上較高Pure-MFE item本身的PIT-safe predicted-Safety percentile S_winner；"
            "因此safe MFE winner > unsafe loser保留高權重，而unsafe MFE winner > safe loser自然降權，pair方向永不因Safety反轉。"
            "最終loss仍為normalized weighted mean；沒有額外mean normalization、bucket、cutoff、lambda、temperature、joint head、"
            "residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_mfe_winner_safety_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_PRODUCT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AH",
        experiment_name="MR-13AH PIT-safe Safety-Product-Weighted Pure-MFE Ranker",
        phase="13AH",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_symmetric_predicted_safety_product_pair_weighting",
        objective_description=(
            "MR-13AF symmetric-weight concentration follow-up：Stage-1完全重用MR-13M Low-Adverse PIT-safe predicted-safety context；"
            "Stage-2固定MR-13K inception_time_v1，Safety不進network，Pure-MFE target/order與full-list Delta-NDCG relevance完全不變。"
            "每個同日non-tied MFE pair只將Delta-NDCG乘上兩端PIT-safe predicted-Safety percentile乘積 S_i*S_j；"
            "pair weighting保持完全對稱，不依MFE winner方向改變，因此不重複MR-13AG directional supervision。"
            "相較MR-13AF min(S_i,S_j)，product會更集中於high-Safety manifold並更強壓低low/low pair；"
            "最終loss仍為normalized weighted mean；沒有bucket、cutoff、lambda、exponent、temperature、joint head、residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_safety_product_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        # Historical model-level flags remain frozen after the failed Forward Model Gate.
        # B313 makes current work-item membership the execution SSOT: when MR-13AH is
        # selected in the shared Model Compare/Test List it may run current Rolling /
        # Robustness workflows; outside that list these frozen flags remain fail-closed.
        # Fixed-Window remains retired and this does not imply model/production promotion.
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PREDICTED_SAFETY_CONFLICT_DISCOUNTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AJ",
        experiment_name="MR-13AJ PIT-safe Conflict-Only Unsafe-Winner-Discounted Pure-MFE Ranker",
        phase="13AJ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="exact_MR-13K_full_horizon_Pure-MFE_order_with_conflict_only_unsafe_winner_predicted_safety_discount",
        objective_description=(
            "AF/AG/AH follow-up：Stage-1仍完全重用MR-13M/Seed42 PIT-safe predicted-Safety context；Stage-2固定MR-13K inception_time_v1、"
            "Pure-MFE target/order、full-list Delta-NDCG、split、optimizer與epoch selection。Safety不進network，pair truth永不反轉。"
            "若MFE ordering與predicted-Safety ordering一致或Safety tie，保留完整MR-13K pair weight=Delta-NDCG；"
            "只有MFE winner較不安全的conflict pair才乘該winner的Safety percentile。"
            "此設計保留AG遺失的aligned MFE supervision，同時比AF更精準地只削弱unsafe-high-MFE conflict pressure；"
            "沒有bucket、cutoff、lambda、exponent、temperature、joint head、residual target、portfolio state或OOS fitting。"
        ),
        metric_scope="all_context_covered_stock_days",
        score_semantic_id="daily_conflict_discounted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE,
        model_research_id="MR-13L",
        experiment_name="MR-13L Daily Universal Decomposed MFE-Adverse Regression",
        phase="13L",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="daily_full_horizon_opportunity_r_v1_decomposed_into_favorable_r_and_adverse_to_peak_r",
        objective_description=(
            "MR-13H相同full-horizon target/universe/architecture；兩個既有輸出神經元分別直接預測"
            "favorable MFE R與adverse-to-peak R，primary loss為兩分量等權MSE平均，"
            "正式model score固定為Predicted MFE R - Predicted adverse R；不使用auxiliary loss、不調lambda"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_decomposed_opportunity_r",
        reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13M",
        experiment_name="MR-13M Daily Universal Full-horizon Low-Adverse Ranker",
        phase="13M",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_negative_daily_full_horizon_adverse_to_peak_r_v1",
        objective_description=(
            "MR-13H/13K相同40D full-horizon earliest-max-MFE peak與fixed R scale；"
            "只以到達該peak前的adverse-to-peak R取負值作排序Target，越高代表path risk越小；"
            "沿用full-list Delta-NDCG weighted RankNet，不加入MFE或strategy state"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_low_adverse_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_BIGRU_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_BIGRU_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BK",
        experiment_name="MR-13BK MR-13M Low-Adverse Target on BJ BiGRU Backbone",
        phase="13BK",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_negative_daily_full_horizon_adverse_to_peak_r_v1",
        objective_description=(
            "Controlled backbone-isolation experiment after MR-13BJ showed improved overall/MFE ranking but mixed Safety behavior. "
            "MR-13M target/universe/40D earliest-max-MFE adverse-to-peak definition, full-list Delta-NDCG pairwise loss, "
            "single rank head, Seed42/split/Adam/gradient clip/date-coherent batch semantics and mean-Daily-Spearman epoch selection remain fixed. "
            "The only scientific treatment is the temporal backbone/readout: MR-13M InceptionTime GAP is replaced by the MR-13BJ "
            "1-layer bidirectional GRU with hidden274 per direction, concatenated final forward/backward states and the same FP32 recurrent/head execution strategy. "
            "Both GRU directions consume only the same 300 decision-time bars, so no post-decision information is introduced. "
            "Primary contrast is MR-13BK vs MR-13M on the identical Low-Adverse target. Seed42 Forward only first; "
            "material all-stock Daily/Global rho and Pair improvement with same-direction Breakout evidence is required before any Rolling."
        ),
        metric_scope="mr13m_low_adverse_target_bigru_backbone_control",
        score_semantic_id="daily_bigru_full_horizon_low_adverse_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13N",
        experiment_name="MR-13N Daily Universal Equal-rank MFE + Low-Adverse Ranker",
        phase="13N",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_equal_weight_mean_of_mfe_percentile_and_low_adverse_percentile",
        objective_description=(
            "MR-13K Pure-MFE與MR-13M low-adverse兩個已證實可學component先各自轉為同日[0,1] percentile；"
            "正式Target固定為兩者等權平均，不掃lambda；沿用full-list Delta-NDCG weighted RankNet。"
            "MR-13H economic Target只在checkpoint後作reference evaluation，不參與training或epoch selection"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_equal_rank_mfe_low_adverse",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        evaluation_reference_profile_name=DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE,
        model_research_id="MR-13O",
        experiment_name="MR-13O Daily Universal Pareto MFE + Low-Adverse Pairwise Ranker",
        phase="13O",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "same_date_strict_pareto_dominance_pairs_over_mfe_percentile_and_low_adverse_percentile; "
            "economic_target_remains_daily_full_horizon_opportunity_r_v1"
        ),
        objective_description=(
            "同日只有當一個stock-day在MFE percentile與low-adverse percentile兩者都嚴格高於另一個stock-day時才提供RankNet supervision；"
            "兩component互有優劣的trade-off pair完全排除。pair等權，不使用MFE/adverse混合係數；"
            "epoch selection只依Validation mean daily Pareto pair concordance，MR-13H economic R不參與選模"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_full_horizon_pareto_dominance_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13P",
        experiment_name="MR-13P Daily Universal Single-model Conditional MFE-Safety Ranker",
        phase="13P",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "primary=same_date_pure_mfe_percentile; "
            "secondary=same_date_percentile_of_low_adverse_residual_given_true_mfe_percentile"
        ),
        objective_description=(
            "單一shared InceptionTime encoder；Pure-MFE primary head與MFE-conditioned residual-safety head皆使用"
            "full-list Delta-NDCG RankNet。secondary target由同日low-adverse percentile對true MFE percentile"
            "含intercept OLS residual後再同日percentile化；conditional head只接收stop-gradient primary prediction，"
            "不做score fusion／threshold／loss-weight sweep"
        ),
        metric_scope="all_stock_days_dual_head",
        score_semantic_id="daily_conditional_mfe_safety_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13Q",
        experiment_name="MR-13Q Daily Universal Reverse-Conditional MFE Single-head Ranker",
        phase="13Q",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_percentile_of_pure_mfe_residual_given_true_low_adverse_safety_percentile",
        objective_description=(
            "Single-head InceptionTime直接學J=U-E(U|S)：U為同日Pure-MFE percentile、S為同日low-adverse Safety percentile；"
            "每日日內含intercept OLS後取MFE residual並再percentile化。只使用full-list Delta-NDCG RankNet，"
            "不建立Safety head、不做scalar MFE/adverse composite、threshold或loss-weight sweep。"
        ),
        metric_scope="all_stock_days_reverse_conditional_mfe",
        score_semantic_id="daily_reverse_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13R",
        experiment_name="MR-13R Daily Universal Safety-conditioned MFE Duo-head Ranker",
        phase="13R",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "condition=same_date_low_adverse_safety_percentile; "
            "final=same_date_percentile_of_pure_mfe_residual_given_true_safety_percentile"
        ),
        objective_description=(
            "單一shared InceptionTime encoder；Raw Safety auxiliary head先學S，Conditional-MFE final head接收shared latent與"
            "stop-gradient Safety prediction並學同一J=U-E(U|S)。兩head均使用full-list Delta-NDCG RankNet、固定等尺度loss平均；"
            "epoch selection與strategy score只看Conditional-MFE head，Safety head不直接進selector。"
        ),
        metric_scope="all_stock_days_safety_conditioned_mfe_dual_head",
        score_semantic_id="daily_safety_conditioned_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13S",
        experiment_name="MR-13S Daily Universal Safety-conditioned Raw-MFE Duo-head Ranker",
        phase="13S",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "condition=same_date_low_adverse_safety_percentile; "
            "final=same_date_pure_mfe_percentile"
        ),
        objective_description=(
            "MR-13R controlled contrast：shared InceptionTime、Raw Safety auxiliary、stop-gradient Safety context與"
            "兩head full-list Delta-NDCG等權loss全部不變；唯一scientific change是final head由residual J改學absolute Pure-MFE percentile U。"
            "epoch selection只依Raw-MFE head Validation mean daily Spearman；model-only Gate，不授權PIT或strategy conversion。"
        ),
        metric_scope="all_stock_days_safety_conditioned_raw_mfe_dual_head",
        score_semantic_id="daily_safety_conditioned_raw_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
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
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AL",
        experiment_name="MR-13AL A2 Shared-AH + Safety Context",
        phase="13AL",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile"
        ),
        objective_description=(
            "A1 MR-13AK的唯一control change：Safety/MFE shared encoder、兩head truth、Safety full-list Delta-NDCG、"
            "AH detach(S_i)×detach(S_j) Pure-MFE pair weighting、pair direction、equal head-loss mean、Seed/split/optimizer全部固定；"
            "final MFE head由shared latent only改為concat(shared latent, detach(Raw Safety probability))。"
            "Safety context不接受MFE loss gradient；final model-gate score仍只使用MFE head。"
        ),
        metric_scope="all_stock_days_shared_safety_context_weighted_raw_mfe_dual_head",
        score_semantic_id="daily_shared_safety_context_weighted_pure_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AM",
        experiment_name="MR-13AM A3 Shared-AH Economic Target Control",
        phase="13AM",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_percentile_of_daily_full_horizon_opportunity_r_v1"
        ),
        objective_description=(
            "MR-13AK A1的單一target control：shared InceptionTime、獨立Raw Safety/final head topology、"
            "Safety full-list Delta-NDCG、same-date predicted-Safety percentile detach(S_i)×detach(S_j) pair weighting、"
            "pair direction、equal head-loss mean、Seed/split/optimizer全部固定；唯一scientific change為final head truth由"
            "Pure-MFE percentile改成MR-13H daily_full_horizon_opportunity_r_v1 percentile。"
            "不使用MR-13AL Safety-context input；final Model-Gate score只使用economic-target head。"
        ),
        metric_scope="all_stock_days_shared_safety_weighted_full_horizon_opportunity_dual_head",
        score_semantic_id="daily_shared_safety_weighted_full_horizon_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AN",
        experiment_name="MR-13AN AM Economic Target + A2 Safety Context",
        phase="13AN",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_percentile_of_daily_full_horizon_opportunity_r_v1"
        ),
        objective_description=(
            "MR-13AM的唯一control change：economic target、shared encoder、Safety truth/loss、"
            "same-date predicted-Safety percentile detach(S_i)×detach(S_j) pair weighting、pair direction、"
            "equal head-loss mean、Seed/split/optimizer全部固定；final economic-target head由shared latent only改為"
            "concat(shared latent, detach(Raw Safety probability))。Safety context不接受primary loss gradient；"
            "final Model-Gate score仍只使用economic-target head。"
        ),
        metric_scope="all_stock_days_shared_safety_context_weighted_full_horizon_opportunity_dual_head",
        score_semantic_id="daily_shared_safety_context_weighted_full_horizon_opportunity_rank",
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
    DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BS",
        experiment_name="MR-13BS AO + Same-Date Dynamic Hypergraph Safety Residual",
        phase="13BS",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO supervised scientific contract exact control：300x10、shared InceptionTime encoder、full-universe "
            "continuous Safety full-list Delta-NDCG、true-HS=P50 Conditional-MFE、1:1 head weighting、Seed42/Adam/"
            "split/epoch-selection與Pred-Safety→Conditional-MFE inference固定。唯一treatment為Safety-only same-date "
            "low-rank dynamic hypergraph residual。每個完整交易日先由AO shared latent形成Nxd nodes；relational branch "
            "只讀stop-gradient latent，以learned soft incidence A=softmax(XW)映射到固定16個latent hyperedges，hyperedge "
            "state用membership-weighted mean，stock relational state再由A映回。concat(detached stock latent, relational "
            "state)經single hidden projection與learned gate產生2-logit Safety residual；final residual projection固定zero-init，"
            "故same-seed step-0 outputs與AO exact一致。Graph branch不得回傳gradient至AO encoder；AO Raw Safety base path仍"
            "正常更新shared encoder，Conditional-MFE完全不讀graph state。Training/inference均固定one complete date per "
            "relational batch，不跨date建graph；不加入Hawkes、pairwise graph、sector/static prior、relation trend、multi-head "
            "HGAT或hyperedge-count sweep。Primary reference=MR-13AO；Seed42 Forward first，只有material Safety共同改善才"
            "考慮下一階段relation-trend/tail-event。"
        ),
        metric_scope="ao_with_same_date_dynamic_hypergraph_safety_residual",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BL",
        experiment_name="MR-13BL AO + Price-Volume Structural Safety Representation",
        phase="13BL",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO target/universe/300x10 input/split/Seed42/Adam/gradient-clip/date-coherent batch/"
            "dual-head 1:1 full-list Delta-NDCG objective/epoch-selection全部固定。唯一scientific treatment是"
            "從既有stock OHLCV前5個normalized sequence channels在batch-time deterministic rasterize Price-Time-Volume structure："
            "128 time bins x 64 ATR-normalized price bins，固定+/-16 ATR，channels=body/wick/relative-volume*body/"
            "relative-volume*wick，並由同一raster建立64-bin Volume-at-Price。small 2D CNN + 1D VAP CNN只產生"
            "Safety residual representation；Raw-MFE head仍只讀MR-13AO shared InceptionTime latent，structure branch不收MFE loss gradient。"
            "不使用人工支撐/壓力label、pivot threshold、RSI/MACD、portfolio/strategy state或post-decision資料；不建立expanded map artifact。"
            "Seed42 Forward only first，Primary reference=MR-13AO。"
        ),
        metric_scope="ao_price_volume_structure_safety_representation_control",
        score_semantic_id="daily_true_hs_conditional_mfe_rank_price_volume_structure_safety",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_LOCAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_LOCAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BM",
        experiment_name="MR-13BM BL + Local Price-Volume Zone Tokens",
        phase="13BM",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13BL global Price-Time-Volume field/VAP、MR-13AO target/universe/300x10 input/split/Seed42/Adam/"
            "gradient-clip/date-coherent batch/dual-head 1:1 full-list Delta-NDCG objective/epoch-selection全部固定。"
            "唯一scientific treatment是在BL同一PIT raster上增加local price-zone token representation：固定只看當前價格+/-4 ATR，"
            "每個price-bin token包含signed distance、full-window與latest-quarter body/wick/relative-volume-body/"
            "relative-volume-wick mass、VAP與volume-structure recency；不使用pivot/support/resistance label或top-K zone heuristic。"
            "single-head query由AO latent+BL global structure residual形成，對local zone tokens做32-d scaled dot-product attention，"
            "所得local residual再只補Raw Safety latent；Raw-MFE path與BL same-seed exact，local branch不收MFE loss gradient。"
            "此設計直接測global field與local zone evidence是否具有互補/interaction information，不改BL raster解析度、ATR span或CNN width。"
            "Primary reference=MR-13BL，AO保留scientific base reference；Seed42 Forward only first。"
        ),
        metric_scope="bl_global_plus_local_price_volume_structure_safety_control",
        score_semantic_id="daily_true_hs_conditional_mfe_rank_price_volume_global_local_safety",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PRICE_VOLUME_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BN",
        experiment_name="MR-13BN Global + High-Resolution Local Price-Volume 2D",
        phase="13BN",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13BL global 128x64x4 Price-Time-Volume field/VAP、MR-13AO target/universe/300x10 input/split/Seed42/"
            "Adam/gradient-clip/date-coherent batch/dual-head 1:1 full-list Delta-NDCG objective/epoch-selection全部固定。"
            "唯一scientific treatment是新增recent 80-bar、+/-4 ATR、96x64x4的high-resolution local Price-Time-Volume 2D map；"
            "channels仍為body/wick/relative-volume*body/relative-volume*wick。local 2D CNN產生32-d latent，與BL global Safety residual"
            "投影後做explicit element-wise cross-scale interaction，再以bias-free projection形成Safety-only residual。"
            "不使用BM price-zone aggregation、pivot/support/resistance label、top-K heuristic或新資料源；Raw-MFE path與BL same-seed exact。"
            "Primary reference=MR-13BL；MR-13BM保留local-zone historical control；Seed42 Forward only first。"
        ),
        metric_scope="bl_global_plus_high_resolution_local_2d_price_volume_safety_control",
        score_semantic_id="daily_true_hs_conditional_mfe_rank_price_volume_multiscale_safety",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_PRICE_VOLUME_STRUCTURE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BO",
        experiment_name="MR-13BO Position-Aware Multi-scale Price-Volume Geometry",
        phase="13BO",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13BN global/local Price-Time-Volume spans/resolutions、MR-13AO target/universe/300x10 input/split/Seed42/"
            "Adam/gradient-clip/date-coherent batch/dual-head 1:1 full-list Delta-NDCG objective/epoch-selection全部固定。"
            "唯一scientific treatment是讓global/local raster encoder顯式position-aware：原body/wick/relative-volume*body/"
            "relative-volume*wick 4個evidence channels各追加signed-price coordinate與time-age coordinate，形成6-channel map；"
            "64-bin VAP由mass-only改為[mass,signed-price-coordinate] 2 channels。signed price以decision-day close為0並以各branch ATR span"
            "正規化到[-1,+1]；time-age由oldest=-1到newest=+1。Global=300 bars,+/-16 ATR,128x64；local=recent80 bars,"
            "+/-4 ATR,96x64；local main effect與local*global interaction沿用BN topology。沒有pivot/support label、technical indicator、"
            "新資料源或target/loss變更；position-aware residual只補Raw Safety，Raw-MFE path與AO/BN same-seed common path exact。"
            "Primary reference=MR-13BN，MR-13BL保留global-field reference；Seed42 Forward only first。"
        ),
        metric_scope="position_aware_multiscale_price_volume_safety_control",
        score_semantic_id="daily_true_hs_conditional_mfe_rank_price_volume_position_aware_multiscale_safety",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_PRICE_VOLUME_MULTISCALE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AP",
        experiment_name="MR-13AP HS-Priority MFE Shared Ranker",
        phase="13AP",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=LS_relevance_0_else_0.5_plus_0.5_times_same_date_MFE_percentile_within_true_HS"
        ),
        objective_description=(
            "MR-13AO strict follow-up after true-HS upside learnability was confirmed but LS extrapolation amplified contamination. "
            "Architecture、Raw Safety head、full-universe encoder exposure、Seed42、split、optimizer與full-list Delta-NDCG固定。"
            "唯一核心改動是final ranking head不再mask LS：true LS全部明確監督為最差relevance=0；true HS依HS cohort內MFE percentile映射到[0.5,1.0]，"
            "因此任一HS truth嚴格高於任一LS，HS內仍依MFE排序，LS內全部tie。Final head只讀shared latent、不吃predicted Safety、不做Safety pair weighting；"
            "inference直接在all-daily universe使用final score，不使用Pred-Safety gate。Seed42 Forward Model Gate only。"
        ),
        metric_scope="all_daily_non_compensatory_hs_priority_then_mfe",
        score_semantic_id="daily_hs_priority_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AQ",
        experiment_name="MR-13AQ HS-Priority Stratified-MFE Shared Ranker",
        phase="13AQ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=MR13AP_truth_LS_0_else_0.5_plus_0.5_times_same_date_MFE_percentile_within_true_HS"
        ),
        objective_description=(
            "MR-13AP strict loss-aggregation control：architecture、Safety auxiliary、AP final truth、all-daily sample exposure、"
            "Seed42/split/optimizer/full-list Delta-NDCG與direct all-daily inference全部固定。唯一change是final-head comparable pairs依"
            "true-HS boundary分為HS↔LS與HS↔HS兩個stratum；兩者保留同一full-list predicted-rank/IDCG/Delta-NDCG geometry，"
            "但各自先以自身Delta-NDCG weight sum正規化，再固定1:1平均。LS↔LS仍tie無direction。此設計只移除AP的boundary supervision-mass dominance，"
            "不改任何pair direction、不加入lambda sweep或Pred-Safety gate。Seed42 Forward Model Gate only。"
        ),
        metric_scope="all_daily_non_compensatory_hs_priority_pair_stratified_then_mfe",
        score_semantic_id="daily_hs_priority_stratified_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AR",
        experiment_name="MR-13AR Direct HS-Qualification + Conditional-MFE Shared Ranker",
        phase="13AR",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=true_HS_indicator_from_same_date_low_adverse_safety_percentile_gte_0.50; "
            "head2=same_date_pure_mfe_percentile_within_true_HS_only"
        ),
        objective_description=(
            "MR-13AO strict primary-head supervision control：shared InceptionTime、full-universe encoder exposure、"
            "true-HS definition、Conditional-MFE target/sublist、independent heads、1:1 head weighting、Seed42/split/optimizer、"
            "epoch selection與Pred-HS P50→Conditional-MFE inference全部固定。唯一scientific change是第一head不再學continuous "
            "Low-Adverse percentile ordering，而將same-date Safety percentile>=0.50轉成binary HS qualification truth；"
            "因此只有HS↔LS pairs有qualification direction，HS↔HS與LS↔LS為tie。Conditional-MFE仍只有true-HS sublist產生gradient，"
            "不吃qualification prediction、不使用Safety pair weight。Seed42 Forward Model Gate only。"
        ),
        metric_scope="direct_hs_qualification_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_hs_qualification_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AS",
        experiment_name="MR-13AS Boundary-Weighted HS-Qualification + Conditional-MFE Shared Ranker",
        phase="13AS",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=true_HS_indicator_from_same_date_low_adverse_safety_percentile_gte_0.50; "
            "head2=same_date_pure_mfe_percentile_within_true_HS_only"
        ),
        objective_description=(
            "MR-13AR strict boundary-supervision control：architecture、true-HS definition=P50、binary HS/LS pair direction、"
            "true-HS Conditional-MFE target/sublist、independent heads、1:1 head weighting、Seed42/split/optimizer、epoch selection與"
            "Pred-HS P50→Conditional-MFE inference全部固定。唯一scientific change是qualification head的HS↔LS pair supervision在"
            "canonical full-list Delta-NDCG weight之上再乘`1-|SafetyPct_i-SafetyPct_j|`；跨界pair越接近P50權重越高，"
            "P90↔P10等容易遠距pair降權。SafetyPct只作truth-side supervision weight，不進model input、不改pair direction、"
            "不新增cutoff/lambda/temperature。Conditional-MFE head與MR-13AR完全相同。Seed42 Forward Model Gate only。"
        ),
        metric_scope="boundary_weighted_hs_qualification_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_boundary_weighted_hs_qualification_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AV",
        experiment_name="MR-13AV AO-Objective Task-Specific Safety/MFE Representation Ranker",
        phase="13AV",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO strict representation-level control：AO continuous Safety full-list Delta-NDCG、"
            "true-HS=P50 Conditional-MFE target/sublist、1:1 head weighting、raw 300x10 input、Seed42/split/optimizer、"
            "epoch selection與Pred-Safety same-date P50→Conditional-MFE inference全部固定。唯一scientific change是"
            "architecture切換為task-specific high-level representation：較低層完整InceptionTime residual groups共享，"
            "最後一個完整residual group由canonical depth-residual_every自動分成Safety與MFE兩支；不新增branch depth/width/lambda。"
            "Safety head只讀Safety-specific latent，Conditional-MFE head只讀MFE-specific latent且不吃predicted Safety。"
            "Seed42 Forward Model Gate first；不改Safety supervision、HS threshold、attention/input feature或strategy。"
        ),
        metric_scope="task_specific_safety_representation_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_task_specific_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AW",
        experiment_name="MR-13AW AO-Objective Safety Temporal Attention Pooling Ranker",
        phase="13AW",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO strict temporal-representation control：AO continuous Safety full-list Delta-NDCG、"
            "true-HS=P50 Conditional-MFE target/sublist、1:1 head weighting、raw 300x10 input、Seed42/split/optimizer、"
            "epoch selection與Pred-Safety same-date P50→Conditional-MFE inference全部固定。唯一scientific change是"
            "Safety head從shared InceptionTime feature map的global-average pooling改為單一1x1 scalar temporal scorer + "
            "softmax-over-time weighted pooling；MFE head仍使用原shared feature map global-average pooling。"
            "attention不新增head count/hidden width/query count/window/temperature等可調參數；不使用AV task-specific branch、"
            "不改Safety truth/loss/HS threshold/input feature或strategy。Primary Gate先看完整continuous Safety ranking learnability；"
            "P50 qualification與HM/HS只作downstream diagnostic。Seed42 Forward first。"
        ),
        metric_scope="safety_temporal_attention_pool_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_safety_attention_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_TASK_SPECIFIC_SAFETY_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AX",
        experiment_name="MR-13AX AO-Objective Task-Specific Safety Attention Pooling Ranker",
        phase="13AX",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AV strict pooling-interaction control：MR-13AO continuous Safety full-list Delta-NDCG、"
            "true-HS=P50 Conditional-MFE target/sublist、1:1 head weighting、raw 300x10 input、Seed42/split/optimizer、"
            "epoch selection與Pred-Safety→Conditional-MFE inference全部固定；AV較低層shared residual groups與最後一個"
            "task-specific Safety/MFE residual group亦固定。唯一scientific change是Safety-specific feature map由GAP改為"
            "單一1x1 scalar temporal scorer + softmax-over-time weighted pooling；MFE-specific feature map仍使用GAP。"
            "attention不新增head count/hidden width/query count/window/temperature等可調參數；不改Safety truth/loss、"
            "HS threshold/input feature或strategy。Primary Gate先看完整continuous Safety ranking learnability，P50/HMHS只作"
            "downstream diagnostic。Seed42 Forward first。"
        ),
        metric_scope="task_specific_safety_temporal_attention_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_task_specific_safety_attention_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_SELF_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_SELF_ATTN_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13AY",
        experiment_name="MR-13AY AO-Objective Safety Temporal Self-Attention Interaction Ranker",
        phase="13AY",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO strict temporal-interaction control：AO shared InceptionTime feature map、continuous Safety full-list "
            "Delta-NDCG、true-HS=P50 Conditional-MFE target/sublist、1:1 head weighting、raw 300x10 input、Seed42/split/"
            "optimizer、epoch selection與Pred-Safety→Conditional-MFE inference全部固定。唯一scientific change是Safety "
            "path在原shared feature map上加入單一single-head Q/K/V temporal self-attention interaction，Q/K/V維度固定等於"
            "canonical channel width，使用canonical scaled-dot-product 1/sqrt(C)，context residual加回feature map後仍用原GAP。"
            "MFE path完全維持AO shared feature map→GAP。沒有FFN、positional encoding、dropout、attention window、head-count、"
            "hidden-width或temperature新knob；不改Safety truth/loss/HS cutoff/input feature/strategy。Primary Gate看完整Safety "
            "Daily/Global rho與Pair是否實質離開AO/AW/AX平台且Breakout同方向。Seed42 Forward first。"
        ),
        metric_scope="safety_temporal_self_attention_interaction_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_safety_temporal_self_attention_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_PAIRWISE_RELATION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_PAIRWISE_RELATION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BP",
        experiment_name="MR-13BP AO-Objective Explicit Pairwise Temporal-Relation Safety Ranker",
        phase="13BP",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO scientific contract exact control：AO shared InceptionTime、continuous Safety full-list Delta-NDCG、"
            "true-HS=P50 Conditional-MFE target/sublist、1:1 head weighting、raw 300x10 input、Seed42/Adam/split/"
            "epoch selection與Pred-Safety→Conditional-MFE inference全部固定。唯一scientific treatment是Safety path在"
            "single-head temporal self-attention score加入由既有stock OHLCV batch-time deterministic建立的explicit "
            "pairwise relation bias：delta-log Close/High/Low、delta canonical normalized-log-volume、normalized signed "
            "time distance；5D relation經固定8D MLP輸出scalar additive attention bias。Raw price relation由canonical "
            "P/anchor_close-1 channels使用log1p差取得，不新增normalization、persistent artifact、pivot/extrema/support label或"
            "future-confirmed structure。Conditional-MFE仍直接讀AO shared feature map→GAP，relation branch只由Safety loss更新。"
            "Primary scientific reference=MR-13AO；MR-13AY為mechanistic self-attention-without-explicit-relation control。"
            "Seed42 Forward first；若只有小增量，不做relation feature/window/hidden/head/topology sweep。"
        ),
        metric_scope="explicit_pairwise_temporal_relation_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_explicit_pairwise_temporal_relation_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_RELATION_CHANGE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_DYNAMIC_HYPERGRAPH_RELATION_CHANGE_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BT",
        experiment_name="MR-13BT AO + Dynamic Hypergraph Relation-Change Safety Residual",
        phase="13BT",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Final controlled OHLCV cross-stock relational gate after MR-13BS same-date state alone failed to move Safety. "
            "MR-13AO 300x10/shared InceptionTime/full-universe Safety/true-HS Conditional-MFE/1:1 objective/Seed42/split/"
            "epoch-selection/inference remain exact. MR-13BS k=16 soft dynamic hypergraph state residual is retained unchanged. "
            "The only new treatment is one previous-trading-date relation change: current and immediately prior full score-eligible "
            "stock universes are encoded by the same shared encoder in no-grad/no-BatchNorm-state-update relation views; the same "
            "incidence projection anchors 16 latent hyperedges across dates; edge_change=current_edge_state-previous_edge_state is "
            "mapped back through current membership and added as a second gated zero-init two-logit Safety residual. Previous-date "
            "rows are input-only and strictly earlier than the decision date. Conditional-MFE never reads relational state. No Hawkes, "
            "pairwise graph, sector/static prior, multi-head/layer HGNN, hyperedge-count, lag-length, EMA, or graph sweep. Primary "
            "reference=MR-13AO; MR-13BS is the immediate mechanistic control. Seed42 Forward first; if Safety remains marginal, close "
            "the OHLCV cross-stock relational family and move to genuinely new PIT-safe information."
        ),
        metric_scope="ao_with_dynamic_hypergraph_previous_date_relation_change_safety_residual",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_ADAPTIVE_HORIZON_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_ADAPTIVE_HORIZON_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BU",
        experiment_name="MR-13BU Adaptive-Horizon Safety Shared Ranker",
        phase="13BU",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "final_head1_exact_MR13AO_40bar_same_date_low_adverse_safety_percentile_over_full_universe; "
            "auxiliary_head1_1_to_40bar_same_date_adverse_path_safety_percentiles; "
            "head2_exact_MR13AO_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "User-authorized new supervision family after OHLCV cross-stock relational family closure. MR-13AO raw 300x10 input, "
            "shared InceptionTime base, canonical 40-bar Safety economics, true-HS=P50 Conditional-MFE target/sublist, Seed42, "
            "Adam/split/PIT/epoch-selection and Pred-Safety->Conditional-MFE inference remain exact. The only scientific treatment "
            "is adaptive future-horizon Safety supervision: each fitting batch materializes PIT-matured adverse-to-best-peak prefixes "
            "for horizons 1..40, converts each horizon to same-date low-adverse Safety percentiles, and hard-asserts horizon 40 equals "
            "the canonical AO Safety target. A 40-horizon two-logit trajectory head plus sample-specific softmax horizon gate learns "
            "effective horizon; its symmetric Safety residual is multiplied by a zero-initialized scalar gain so same-seed step-0 "
            "final Safety remains AO-exact. Safety composite loss is equal mean of canonical 40-bar full-list Delta-NDCG pairwise "
            "loss and 1..40 soft-BCE trajectory loss; the top-level Safety-composite versus true-HS Conditional-MFE composition remains "
            "equal mean with no lambda sweep. Conditional-MFE topology/input/supervision is unchanged and does not consume horizon "
            "weights. Maximum future maturity remains 40 bars; the model learns effective horizon inside that fixed legal envelope. "
            "Primary reference=MR-13AO and AO must remain in every Compare/Test list. Seed42 Forward first; if Raw Safety Daily/Global/"
            "Pair and Pred-HS purity do not materially improve, close this adaptive-horizon family without horizon-count, auxiliary-loss, "
            "input-length, alternate-path-target, or gate-capacity sweep, then return to truly orthogonal PIT-safe information."
        ),
        metric_scope="ao_with_adaptive_future_horizon_safety_trajectory",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_ADAPTIVE_INPUT_CONTEXT_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_ADAPTIVE_INPUT_CONTEXT_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BV",
        experiment_name="MR-13BV Adaptive Input-Context Safety Shared Ranker",
        phase="13BV",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_40bar_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "User-authorized adaptive past-context experiment after MR-13BU adaptive future-horizon Forward failed. All target/loss "
            "semantics revert exactly to MR-13AO: canonical 40-bar Safety full-list Delta-NDCG, true-HS=P50 Conditional-MFE sublist, "
            "equal head weighting, Seed42, Adam, split/PIT/epoch-selection and Pred-Safety->Conditional-MFE inference. Maximum input "
            "remains the canonical 300x10 window. The only scientific treatment is a Safety-only adaptive look-back residual over fixed "
            "30/60/120/300-bar suffix views. The same AO InceptionTime encoder weights encode every view; 30/60/120-bar auxiliary "
            "views are stop-gradient and cannot update BatchNorm running state, while the canonical 300-bar latent remains the sole "
            "Conditional-MFE input. A sample-specific softmax gate chooses the four look-back weights and a zero-initialized two-logit "
            "projection of adaptive-minus-300bar latent preserves same-seed AO-exact Safety at step 0. No future-horizon adaptation, "
            "input-window sweep, extra target, loss weight, temperature, hidden gate, or 600-bar extension is included. Primary "
            "reference=MR-13AO and AO remains in Compare/Test. Seed42 Forward first; if Raw Safety Daily/Global/Pair, Pred-HS purity "
            "and Breakout Safety do not materially improve without Conditional-MFE trade-off, close adaptive-input family without tuning."
        ),
        metric_scope="ao_with_adaptive_30_60_120_300bar_safety_input_context",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SCC_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SCC_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BQ",
        experiment_name="MR-13BQ SCC-Pretrained AO Shared Ranker",
        phase="13BQ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO supervised scientific contract exact control：同一300x10 input、shared InceptionTime architecture、"
            "continuous Safety full-list Delta-NDCG、true-HS=P50 Conditional-MFE sublist、1:1 head weighting、Seed42/Adam/"
            "split/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。唯一scientific treatment是在每次"
            "downstream fitting前，僅以該fitting scope內daily-universal rows做PIT-safe Stock Code Classification encoder "
            "pretraining：每epoch每ticker均勻抽一個合法300-bar window、100 fixed epochs、Adam lr=1e-3、batch=128、"
            "weight_decay=1e-4、clip=1.0、無augmentation。Inner epoch-selection只允許Inner-Train pretraining，Validation完全"
            "不參與；final refit重新random initialize並只用完整Selection重新pretrain。temporary SCC head隨後丟棄，AO Safety/"
            "MFE heads fresh initialize，shared encoder從第一個AO step起full-unfrozen fine-tune；ticker identity不成為正式"
            "inference input。Primary reference=MR-13AO。Seed42 Forward first；若只得到既有約1-3%級marginal shift，不做"
            "pretrain LR/epoch/freeze/batch/task sweep。"
        ),
        metric_scope="ao_with_pit_safe_stock_code_encoder_pretraining",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_FUTURE_PATH_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_FUTURE_PATH_PRETRAINED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BR",
        experiment_name="MR-13BR Future-Path-Pretrained AO Shared Ranker",
        phase="13BR",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "exact_MR13AO_head1_same_date_low_adverse_safety_percentile_over_full_universe; "
            "exact_MR13AO_head2_same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "User-authorized second pretraining hypothesis after MR-13BQ SCC failed. MR-13AO supervised 300x10 input, "
            "shared InceptionTime architecture, continuous Safety full-list Delta-NDCG, true-HS=P50 Conditional-MFE, "
            "1:1 head weighting, Seed42/Adam/split/epoch selection and Pred-Safety→Conditional-MFE inference remain exact. "
            "The only scientific treatment is fitting-scope PIT-safe future-path encoder pretraining. For each epoch, one "
            "matured 300-bar window per ticker is sampled uniformly. A temporary 12-output regression head predicts raw "
            "future path statistics: 5/10/20/40d terminal return, 10/20/40d max downside, 10/20/40d max upside and "
            "10/20d realized volatility. Targets are generated only when the full 40-bar path ends on/before the fitting "
            "cutoff, standardized with fitting-scope-only z-scores, and optimized by SmoothL1(beta=1) for 100 fixed epochs "
            "with Adam lr=1e-3, batch=128, weight_decay=1e-4 and clip=1.0. No breakout/candidate/portfolio state enters "
            "pretraining. Inner validation is untouched; final Selection pretraining is rebuilt from scratch. Temporary "
            "future-path head is discarded, same-seed AO Safety/MFE heads are fresh initialized, and the shared encoder is "
            "full-unfrozen from the first AO step. Primary reference=MR-13AO. Seed42 Forward first; if Safety does not "
            "materially break the existing ceiling, do not sweep horizon/input length/MAP/masking/foundation-model variants."
        ),
        metric_scope="ao_with_pit_safe_future_path_encoder_pretraining",
        score_semantic_id="daily_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),

DAILY_UNIVERSAL_PATCH_SAFETY_INCEPTION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
    profile_name=DAILY_UNIVERSAL_PATCH_SAFETY_INCEPTION_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    model_research_id="MR-13AZ",
    experiment_name="MR-13AZ AO-Objective Safety Patch Transformer + Conditional-MFE InceptionTime Ranker",
    phase="13AZ",
    trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
    target_description=(
        "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
        "head2=same_date_pure_mfe_percentile_within_true_hs_only"
    ),
    objective_description=(
        "MR-13AO objective exact control with a dual-encoder temporal representation contrast：continuous Safety full-list "
        "Delta-NDCG、true-HS=P50 Conditional-MFE target/sublist、1:1 head weighting、raw 300x10、Seed42/split/optimizer、"
        "epoch selection與Pred-Safety→Conditional-MFE inference固定。Safety encoder改為MR-13Z已freeze的historical 9F "
        "Patch Transformer recipe：non-overlap patch=10、embedding=128、3 layers、4 heads、FFN=256、sinusoidal position、"
        "dropout=0.10、patch-token global mean；Conditional-MFE encoder保留AO-form InceptionTime + GAP。兩encoder獨立，"
        "Safety loss只更新Patch encoder，Conditional-MFE loss只更新InceptionTime encoder；不做fusion、auxiliary Safety、"
        "patch/depth/head/embedding/FFN/dropout/pooling sweep，也不改Safety truth/loss、HS cutoff/input feature/strategy。"
        "Primary Gate看完整Safety Daily/Global rho與Pair是否實質離開AO平台且Breakout同方向，同時Conditional-MFE不得崩。"
    ),
    metric_scope="independent_patch_safety_plus_inception_true_hs_conditional_mfe",
    score_semantic_id="daily_patch_safety_then_inception_true_hs_conditional_mfe_rank",
    pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
    secondary_pair_scope_threshold=0.50,
    model_gate_reference_profile_name=(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    ),
    selection_pit_authorized=False,
    current_time_validation_authorized=False,
),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_WINDOW_RF_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_WINDOW_RF_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BA",
        experiment_name="MR-13BA AO-Objective Full-Window Receptive-Field InceptionTime Ranker",
        phase="13BA",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO strict receptive-field capacity control：continuous Safety full-list Delta-NDCG、true-HS=P50 "
            "Conditional-MFE target/sublist、shared InceptionTime topology、1:1 head weighting、raw 300x10 input、"
            "Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。唯一scientific "
            "change是InceptionTime temporal receptive field由AO的229 bars擴至完整覆蓋300-bar input；depth=6、filters=32、"
            "bottleneck=32、residual_every=3、kernels=(39,19,9)、GAP與head topology不變。只在最後一個residual group"
            "把temporal dilation改為2，module dilations=(1,1,1,1,2,2)，actual receptive field=305；trainable params exact不變，"
            "不增加input bars、不改target/loss/head/pooling/backbone family、不做RF/dilation/kernel sweep。"
            "Primary Gate先看完整Safety Daily/Global rho與Pair能否實質離開AO平台且Breakout同方向，並要求HS-only "
            "Conditional-MFE不崩。若無明顯增量，full-window RF不足假說停止，再進純capacity scaling。"
        ),
        metric_scope="full_window_receptive_field_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_full_window_rf_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BB",
        experiment_name="MR-13BB AO-Objective Wide InceptionTime Capacity Control",
        phase="13BB",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO strict pure-capacity control after MR-13BA rejected receptive-field insufficiency：continuous Safety "
            "full-list Delta-NDCG、true-HS=P50 Conditional-MFE target/sublist、shared InceptionTime topology、1:1 head "
            "weighting、raw 300x10 input、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。"
            "唯一scientific change是channel capacity：Inception filters=32→64、bottleneck=32→64；depth=6、"
            "residual_every=3、kernels=(39,19,9)、module dilations全1、actual receptive field=229、GAP與head topology不變。"
            "feature_count=10下trainable params由AO 473,734增加至1,885,446（約3.98x）。不改input bars、RF、target/loss/head/"
            "pooling/backbone family，不做width sweep。Primary Gate先看完整Safety Daily/Global rho與Pair是否實質離開AO平台且"
            "Breakout同方向，同時Pred-HS true-LS需下降且HS-only Conditional-MFE不得崩。若仍無明顯增量，停止單純"
            "InceptionTime parameter-capacity不足假說，下一步才進backbone-family controlled contrast。"
        ),
        metric_scope="wide_capacity_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_wide_capacity_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BC",
        experiment_name="MR-13BC AO-Objective 600-Bar Long-Horizon Input Control",
        phase="13BC",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO long-horizon information control after MR-13BA rejected 300-bar RF insufficiency and MR-13BB "
            "rejected pure capacity scaling at 300 bars：continuous Safety full-list Delta-NDCG、true-HS=P50 Conditional-MFE "
            "target/sublist、shared InceptionTime、1:1 head weighting、filters/bottleneck=32/32、depth=6、kernels=(39,19,9)、"
            "GAP/heads、Seed42/split/optimizer/epoch selection與Pred-Safety→Conditional-MFE inference全部固定。"
            "Scientific treatment是可利用歷史由raw 300x10擴為raw 600x10；architecture declaratively要求input_window_bars=600。"
            "為使600-bar input真正可由高階feature共同利用，僅以parameter-neutral dilation把RF同步擴至609："
            "module dilations=(1,1,1,1,6,6)，trainable parameter count維持AO約473,734。"
            "這個RF變化是600-bar input的必要可利用性條件，不解讀為獨立RF treatment；BA已單獨證明300-bar下RF 229→305沒有Safety增量。"
            "Primary Gate需優先在與AO/BA的common eligible stock-days上判讀Safety Daily/Global rho、Pair、Pred-HS true-LS與"
            "P45-P55 boundary；Breakout需同方向且HS-only Conditional-MFE不得崩。若600/32仍無明顯增量，停止input-horizon不足"
            "假說並轉backbone-family controlled contrast；若600/32明顯改善，下一輪才允許600/64 capacity interaction。"
        ),
        metric_scope="long_horizon_600bar_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_600bar_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_DEEP_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_DEEP_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BD",
        experiment_name="MR-13BD AO-Objective Deep Hierarchy Capacity/RF-Matched Control",
        phase="13BD",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "MR-13AO deep-hierarchy control after BA/BB/BC found no Safety breakthrough from RF, 4x width capacity, or 600-bar history. "
            "Continuous Safety full-list Delta-NDCG, true-HS=P50 Conditional-MFE, 1:1 loss, raw 300x10 input, shared InceptionTime family, "
            "GAP/heads, Seed42/split/optimizer/epoch selection and Pred-Safety->Conditional-MFE inference are fixed. Scientific treatment is "
            "hierarchical composition depth=6->12 while keeping final latent width filters=32 and holding total parameter/RF scale near AO by "
            "compensating bottleneck=24 and kernels=(21,11,5): feature_count=10 trainable params=475,702 vs AO 473,734 (+0.42%), RF=241 vs AO 229 (+5.2%), "
            "residual_every=3 and dilations all1. The smaller bottleneck/kernels are matching controls rather than optimization knobs; no depth/kernel/width sweep. "
            "Primary Gate requires full Safety Daily/Global rho and Pair to materially leave the AO platform with lower Pred-HS true-LS and same-direction Breakout; "
            "HS-only Conditional-MFE must not collapse. If FAIL, close InceptionTime RF/width/input/depth physical-limit family and move to a genuinely different backbone family."
        ),
        metric_scope="deep_hierarchy_capacity_rf_matched_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_deep_hierarchy_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_WIDE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BE",
        experiment_name="MR-13BE AO-Objective 600-Bar × Wide-Capacity Interaction Control",
        phase="13BE",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Factorial interaction control after BB showed 64/64 capacity alone fails at 300 bars and BC showed 600-bar history alone fails at 32/32. "
            "MR-13AO continuous Safety full-list Delta-NDCG, true-HS=P50 Conditional-MFE, 1:1 loss, shared InceptionTime depth=6, kernels=(39,19,9), "
            "GAP/heads, Seed42/split/optimizer/epoch selection and Pred-Safety->Conditional-MFE inference remain fixed. This cell intentionally combines the two already-isolated factors: "
            "raw input_window_bars=600 with parameter-neutral RF coverage via module dilations=(1,1,1,1,6,6), plus filters/bottleneck=64/64. "
            "Thus BE has the same 600-bar eligibility/RF as BC and the same width/capacity as BB; no new target, head, loss, depth, kernel or backbone-family change is introduced. "
            "Primary causal contrast is BE vs BC on the identical 600-bar eligible universe, asking whether added capacity becomes useful only when longer history is available. "
            "BC is therefore the frozen model-gate reference. AO/BB remain completed factorial context, not fitting inputs. If BE does not materially improve full Safety Daily/Global rho, Pair and Pred-HS purity over BC "
            "with same-direction Breakout and non-collapsing HS-only Conditional-MFE, close the input×capacity interaction and stop InceptionTime physical-limit scaling before moving to a new backbone family."
        ),
        metric_scope="long_horizon_600bar_wide_capacity_interaction_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_600bar_wide_capacity_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_600BAR_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BF",
        experiment_name="MR-13BF AO-Objective Full-Resolution Day-Token Transformer Control",
        phase="13BF",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Backbone-family control after BA/BB/BC/BD/BE found no Safety breakthrough from receptive field, capacity@300, 600-bar history, deeper hierarchy, or 600-bar×wide interaction. "
            "MR-13AO target/loss/true-HS scope, raw 300x10 input, independent linear Safety/MFE heads, mean pooling, Seed42/split/optimizer/epoch selection and Pred-Safety->Conditional-MFE inference remain fixed. "
            "The only primary treatment is the temporal backbone: InceptionTime is replaced by a full-resolution day-token Transformer with one trading day per token (patch_size=stride=1), sinusoidal positions, "
            "global self-attention from layer 1, d_model=96, depth=6, heads=4 and FFN=212. No 10-bar Patch compression is used. Dropout=0 matches AO. "
            "Feature_count=10 trainable params are 473,692 versus AO 473,734 (-0.009%), so capacity is effectively matched. "
            "Primary Gate requires a material full-Safety Daily/Global rho, Pair and Pred-HS purity improvement over AO with same-direction Breakout and non-collapsing true-HS Conditional-MFE. "
            "If FAIL, do not tune token size/heads/d_model/depth/FFN on iterative OOS; close this raw global-attention backbone hypothesis and reassess whether the remaining ceiling is information/target uncertainty rather than model size/topology."
        ),
        metric_scope="parameter_matched_full_resolution_day_token_transformer_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_day_token_transformer_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BG",
        experiment_name="MR-13BG AO-Objective Parameter-Matched GRU Backbone Control",
        phase="13BG",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "User-authorized recurrent-backbone contrast after MR-13BF full-resolution global attention failed to break the Safety ceiling. "
            "MR-13AO raw 300x10 input, target/loss/true-HS scope, independent linear Safety/MFE heads, Seed42/split/optimizer/epoch selection and Pred-Safety->Conditional-MFE inference remain fixed. "
            "The only primary treatment is shared temporal backbone: InceptionTime is replaced by a 1-layer unidirectional GRU with hidden_size=391, dropout=0 and final recurrent state pooling. "
            "This is the execution-feasible pre-result revision of the original 3-layer/176 implementation; feature_count=10 trainable params are 474,287 versus AO 473,734 (+0.117%), so capacity remains effectively matched. "
            "Primary Gate requires material full-Safety Daily/Global rho, Pair and Pred-HS purity improvement over AO, with P45-P55 boundary improvement and non-collapsing true-HS Conditional-MFE. "
            "If FAIL, do not sweep GRU hidden size/layers/bidirectionality on iterative OOS; close recurrent-backbone shopping and return to information/target-uncertainty research."
        ),
        metric_scope="parameter_matched_gated_recurrent_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_gru_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_BIGRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_BIGRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BJ",
        experiment_name="MR-13BJ Parameter-Matched Bidirectional GRU Recurrent-State Control",
        phase="13BJ",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "User-authorized controlled GRU-family extension after MR-13BG Rolling remained mixed. "
            "This explicitly overrides the prior BG tuning-closed stop rule for one bidirectionality test only. "
            "MR-13BG raw 300x10 input, 1-layer recurrent depth, FP32 recurrent/head numerical policy, target/loss/true-HS scope, independent linear Safety/MFE heads, Seed42/split/Adam/gradient clip/date-coherent batch semantics/epoch selection/inference remain fixed. "
            "The only primary treatment is recurrent directionality/readout: unidirectional hidden391 final state is replaced by a 1-layer bidirectional GRU with hidden274 per direction and concatenated final forward/backward states. "
            "Feature_count=10 trainable params are 472,380 versus AO 473,734 (-0.286%) and BG 474,287 (-0.402%), preserving matched capacity without adding a projection layer. "
            "Because both directions consume only the same 300 decision-time bars, no post-decision data or look-ahead is introduced. "
            "Primary Forward gate requires material Safety Daily/Global rho, Pair and Pred-HS purity improvement over BG/AO while preserving or improving HS-only Conditional-MFE and top-tail HM/HS/MFE; otherwise close pure GRU topology extensions without hidden/layer/bidirectional sweep."
        ),
        metric_scope="parameter_matched_bidirectional_gated_recurrent_safety_plus_true_hs_conditional_mfe",
        score_semantic_id="daily_bigru_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_GRU_BF16_GUARDED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_GRU_BF16_GUARDED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BH",
        experiment_name="MR-13BH GRU v2 BF16 Guarded-Retry Numerical Control",
        phase="13BH",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Numerical-control follow-up to MR-13BG after the same 1-layer/hidden391 GRU topology showed a distinct BF16 convergence trajectory but the BF16 final refit became non-finite. "
            "MR-13BG FP32-island artifact remains unchanged and is the frozen primary reference. MR-13BH keeps the same raw 300x10 input, GRU topology/parameter count, target/loss/true-HS scope, Seed42, split, optimizer, gradient clip, date-coherent batch membership/order, epoch selection and inference. "
            "The only treatment is recurrent numerical execution: normal forward/backward stays in canonical CUDA BF16 outer autocast. If the canonical finite-margin/loss guard or pre-step gradient finite guard detects a non-finite batch, no optimizer step is taken; the exact same batch is recomputed once with autocast disabled (FP32), then the single canonical optimizer step is applied. "
            "No batch is skipped, no extra optimizer update is added, and no LR/epoch/hidden/layer tuning is authorized. Primary contrast is BH BF16-guarded vs BG FP32-island. This control asks whether the earlier BF16 late-epoch validation regime can produce a legal final OOS model when only the numerical blow-up step is rescued."
        ),
        metric_scope="gru_bf16_guarded_retry_numerical_control",
        score_semantic_id="daily_gru_bf16_guarded_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_GRU_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_GRU_BF16_BACKWARD_SCALED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_GRU_BF16_BACKWARD_SCALED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13BI",
        experiment_name="MR-13BI GRU Pure-BF16 Dynamic Backward-Scaling Numerical Control",
        phase="13BI",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile_over_full_universe; "
            "head2=same_date_pure_mfe_percentile_within_true_hs_only"
        ),
        objective_description=(
            "Numerical-control follow-up after MR-13BH showed that inserting a full-FP32 same-batch recompute at the first non-finite BF16 gradient materially changed the final-refit trajectory. "
            "MR-13BI reuses the exact MR-13BH 1-layer/hidden391 outer-BF16 GRU topology, 474,287 params, raw 300x10 input, target/loss/true-HS scope, Seed42, split, Adam, date-coherent batch membership/order, gradient clip, epoch selection and inference. "
            "Normal batches remain pure BF16. If and only if backward produces a non-finite gradient, the failed gradients are discarded and the exact same batch is recomputed under the same BF16 autocast with progressively smaller loss scales 1/2 through 1/256. Once finite, FP32 parameter-gradient buffers are unscaled back to the canonical gradient magnitude, canonical clip_norm is applied, and exactly one optimizer step is taken. "
            "No FP32 forward/backward fallback is permitted; non-finite forward/loss fails fast. Primary contrast is BI vs BH for numerical rescue behavior, with BG retained as the complete FP32-island scientific reference. No LR/epoch/hidden/layer/bidirectional tuning is authorized."
        ),
        metric_scope="gru_pure_bf16_dynamic_backward_scaling_numerical_control",
        score_semantic_id="daily_gru_pure_bf16_scaled_backward_safety_then_true_hs_conditional_mfe_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.50,
        model_gate_reference_profile_name=(
            DAILY_UNIVERSAL_GRU_BF16_GUARDED_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13T",
        experiment_name="MR-13T Daily Universal Safety + Raw-MFE + Direct HM/HS Tri-head Ranker",
        phase="13T",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=direct_hmhs_indicator_of_both_percentiles_ge_0.5"
        ),
        objective_description=(
            "MR-13S strict controlled contrast：300×10 raw normalized stock/0050 OHLCV、shared InceptionTime、"
            "Raw Safety head、stop-gradient Safety-conditioned Raw-MFE head、Seed42/split/optimizer與Raw-MFE epoch selection全部不變；"
            "唯一新增shared latent上的Direct HM/HS head，target=1[S>=0.5 and U>=0.5]。三head各自使用full-list Delta-NDCG RankNet，"
            "loss固定等權平均；HM/HS head不接Safety/MFE score arithmetic，不使用人工feature、breakout state、strategy state、threshold sweep或OOS fitting。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_direct_hmhs_tri_head",
        score_semantic_id="daily_safety_raw_mfe_direct_hmhs_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13V",
        experiment_name="MR-13V MR-13T Control + Nonlinear Direct HM/HS MLP Head",
        phase="13V",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=direct_hmhs_indicator_of_both_percentiles_ge_0.5"
        ),
        objective_description=(
            "MR-13T strict architecture contrast：canonical raw 300×10、shared InceptionTime trunk、Raw Safety head、"
            "stop-gradient Safety-conditioned Raw-MFE head、S/U/H targets、Seed42、daily-universal split、optimizer、"
            "full-list Delta-NDCG、三head等權loss與Raw-MFE Validation epoch selection全部不變；唯一scientific change是"
            "Direct HM/HS readout由Linear(latent,2)改為固定Linear(latent,latent)->ReLU->Linear(latent,2)。"
            "hidden width等於shared latent width，不設dropout、不做width/depth/activation/lambda sweep；Joint head仍不讀Safety/MFE predicted scores。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_direct_hmhs_nonlinear_head",
        score_semantic_id="daily_safety_raw_mfe_direct_hmhs_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13W",
        experiment_name="MR-13W MR-13V Architecture + Continuous Joint-Min Target",
        phase="13W",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13V strict target contrast：raw 300×10、shared InceptionTime trunk、Raw Safety head、"
            "stop-gradient Safety-conditioned Raw-MFE head、nonlinear Joint MLP architecture、Seed42、split、optimizer、"
            "full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部不變；唯一fitting change是"
            "第三head supervision由binary 1[S>=0.5 and U>=0.5]改為continuous min(S,U)。"
            "不重新rank min target、不設threshold/temperature/weight，不使用OOS統計或strategy state。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_nonlinear_head",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13X",
        experiment_name="MR-13X MR-13W Target + Joint Temporal Attention Pooling",
        phase="13X",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13W strict representation contrast：raw 300×10、InceptionTime trunk、Safety/Raw-MFE marginal global-average paths、"
            "S/U/Jmin targets、128->128->2 ReLU joint MLP、Seed42、split、optimizer、full-list Delta-NDCG、三head等權loss與"
            "Raw-MFE Validation epoch selection全部不變；唯一scientific change是Joint-Min branch在shared temporal feature map上"
            "使用parameter-free-shape scalar 1x1 attention scorer + softmax(time) + weighted temporal sum，取代Joint branch global mean。"
            "不加positional encoding/multi-head attention/dropout/temperature/attention-width sweep；marginal heads仍只讀原global mean。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_attention_pool",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13Y",
        experiment_name="MR-13Y Joint-Min ModernTCN Raw-data Architecture Comparison",
        phase="13Y",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13X strict raw-data architecture-family contrast：canonical raw 300×10、S/U/Jmin targets、"
            "Safety/Raw-MFE marginal global-average semantics、Joint scalar temporal attention pooling、latent-width→latent-width→2 ReLU joint MLP、"
            "Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部不變；"
            "唯一research dimension是temporal trunk由InceptionTime換成ModernTCN。ModernTCN固定沿用historical 9B未調參recipe："
            "6 blocks、96 channels、kernel51 depthwise temporal conv、4x pointwise expansion、BatchNorm、dropout0.10；"
            "head/attention widths僅隨trunk latent width自然為96，不另加adapter或capacity sweep。historical modern_tcn_v1仍維持legacy read-only。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_modern_tcn_attention_pool",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13Z",
        experiment_name="MR-13Z Joint-Min Patch Transformer Raw-data Architecture Comparison",
        phase="13Z",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "head1=same_date_low_adverse_safety_percentile; "
            "head2=same_date_pure_mfe_percentile; "
            "head3=min(same_date_low_adverse_safety_percentile,same_date_pure_mfe_percentile)"
        ),
        objective_description=(
            "MR-13X strict raw-data architecture-family contrast：canonical raw 300×10、S/U/Jmin targets、"
            "Safety/Raw-MFE marginal global-average semantics、Joint scalar temporal attention pooling、latent-width→latent-width→2 ReLU joint MLP、"
            "Seed42、split、optimizer、full-list Delta-NDCG、三head固定等權loss與Raw-MFE Validation epoch selection全部不變；"
            "唯一research dimension是temporal trunk由InceptionTime換成historical-recipe Patch Transformer。Patch Transformer固定9F未調參recipe："
            "non-overlap patch=10 bars、embedding=128、3 encoder layers、4 heads、FFN=256、sinusoidal position、dropout=0.10；"
            "marginal heads對30 patch tokens作global mean，Joint head對同一token map作scalar softmax attention；"
            "不做patch/depth/head/embedding/FFN/position/dropout/pooling sweep。historical patch_transformer_v1維持legacy read-only。"
        ),
        metric_scope="all_stock_days_safety_raw_mfe_joint_min_patch_transformer_attention_pool",
        score_semantic_id="daily_safety_raw_mfe_joint_min_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=True,
        current_time_validation_authorized=True,
    ),
    DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13U",
        experiment_name="MR-13U Daily Universal Direct HM/HS H-only Single-head Ranker",
        phase="13U",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description=(
            "direct_hmhs_indicator=1[same_date_low_adverse_safety_percentile>=0.5 "
            "and same_date_pure_mfe_percentile>=0.5]"
        ),
        objective_description=(
            "MR-13T learnability ablation：canonical raw 300×10 normalized stock/0050 OHLCV、InceptionTime trunk、"
            "Seed42、daily-universal split、optimizer與full-list Delta-NDCG pairwise logistic不變；移除Safety/MFE heads與loss，"
            "只保留單一Direct HM/HS classifier，讓encoder僅受H supervision。Epoch selection事前固定為Validation HM/HS Pair concordance，"
            "同值才以Validation global PR-AUC tie-break；不加人工feature、不用breakout/strategy state、不做OOS fitting。"
        ),
        metric_scope="all_stock_days_direct_hmhs_h_only",
        score_semantic_id="daily_direct_hmhs_h_only_research",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
        current_time_validation_authorized=False,
    ),
    DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE,
        model_research_id="MR-13F",
        experiment_name="MR-13F Daily Universal Direct-R Huber Regression",
        phase="13F",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="raw_daily_opportunity_no_time_r_v1_in_R_units",
        objective_description=(
            "同日全部合法stock-day直接預測daily_opportunity_no_time_r_v1 raw R；"
            "使用Huber loss且delta固定為1R，two-logit margin直接解讀為Predicted R"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_predicted_r",
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE,
        model_research_id="MR-13G",
        experiment_name="MR-13G Daily Universal Direct-R Mean Regression",
        phase="13G",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="raw_daily_opportunity_no_time_r_v1_in_R_units",
        objective_description=(
            "同日全部合法stock-day直接預測daily_opportunity_no_time_r_v1 raw R；"
            "使用MSE以population optimum對齊conditional mean E[R|X]，two-logit margin直接解讀為Predicted R"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_predicted_r",
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_RISK_NORMALIZED_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13I",
        experiment_name="MR-13I Daily Universal Cost-adjusted Risk-normalized 40D Ranker",
        phase="13I",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_date_all_stock_order_of_daily_risk_normalized_net_opportunity_r_v1",
        objective_description=(
            "MR-13E universe/backbone/40D horizon不變；只把target改為以historical-effective Min ROOS "
            "atr_len+atr_times_init定義generic initial risk distance，並使用canonical 1% sizing/fee/tax後的40D opportunity R；"
            "full-list Delta-NDCG pairwise objective不變"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_risk_normalized_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        selection_pit_authorized=False,
    ),
    DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE: ContinuousRankerResearchSpec(
        profile_name=DAILY_UNIVERSAL_RISK_CONTEXT_NET_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        model_research_id="MR-13J",
        experiment_name="MR-13J Daily Universal Risk-context 40D Ranker",
        phase="13J",
        trainer_family=CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        target_description="same_target_as_MR-13I_daily_risk_normalized_net_opportunity_r_v1",
        objective_description=(
            "Target/universe/horizon/loss與MR-13I完全相同；唯一新增decision-time universal risk/economic geometry context "
            "(risk distance, capital per risk, cost per risk, risk capacity)"
        ),
        metric_scope="all_stock_days",
        score_semantic_id="daily_risk_normalized_opportunity_rank",
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    ),
}

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

