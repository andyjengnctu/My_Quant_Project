"""Shared continuous-ranker training semantics contract.

Trainer artifact producers and runtime artifact validators must resolve the same
profile-driven semantics from this module.  In particular, pairwise weighting is
owned by the experiment-agnostic execution recipe rather than by research
identity or a consumer-specific hard-coded default.
"""

from __future__ import annotations

from typing import Any

from config.breakout_quality_runtime import (
    CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT,
    CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
    CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
    CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE,
    CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY,
    get_continuous_ranker_primary_pair_weight_policy,
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY,
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE,
    CONTINUOUS_RANKER_SEMANTICS_DEFAULT,
    CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS,
    CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R,
    CONTINUOUS_RANKER_SEMANTICS_LISTWISE,
    CONTINUOUS_RANKER_SEMANTICS_PAIRWISE,
    CONTINUOUS_RANKER_SEMANTICS_RAW_R,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE,
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS,
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN,)
from config.breakout_quality_runtime_resolver import (
    get_continuous_ranker_execution_recipe,
)
from config.breakout_quality import (
    get_predicted_safety_pair_weight_contract,
    get_predicted_safety_context_contract,
    get_predicted_safety_pure_mfe_contract,
    get_predicted_upside_context_contract,
)


PAIRWISE_TRAINING_CONTRACT = {
    "pair_scope": "same_date_non_tied_target_pairs",
    "pair_weighting": CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    "model_margin": "pass_logit_minus_reject_logit",
    "batching": "whole_date_pack_no_date_split",
    "runtime_score": "softmax_pass_probability",
}


DUAL_COMPONENT_R_REGRESSION_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "target": "daily_full_horizon_opportunity_r_v1_decomposed_components",
    "prediction": {
        "reject_output": "predicted_adverse_to_peak_r",
        "pass_output": "predicted_favorable_mfe_r",
    },
    "loss": "mean_mse_over_two_primary_r_components",
    "component_weighting": "equal_by_mean_reduction_no_lambda",
    "runtime_score": "predicted_favorable_mfe_r_minus_predicted_adverse_to_peak_r",
    "batching": "shuffled_unique_group_batches",
}


CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "primary_target": "same_date_pure_mfe_percentile",
    "conditional_target": (
        "same_date_percentile_of_low_adverse_residual_after_same_date_OLS_on_true_pure_mfe_percentile"
    ),
    "target_conditioning": "supervision_only_true_mfe_percentile_no_future_feature",
    "architecture": "shared_encoder_primary_mfe_head_plus_mfe_conditioned_safety_head",
    "conditional_context": "stop_gradient_primary_mfe_probability",
    "primary_head_gradient_from_conditional_loss": False,
    "shared_encoder_gradient_from_both_heads": True,
    "head_losses": "full_list_delta_ndcg_pairwise_logistic_each_head",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "conditional_safety_mean_daily_spearman",
    "runtime_status": "model_gate_only_no_strategy_score_fusion",
    "batching": "whole_date_pack_no_date_split",
}


CONDITIONAL_MFE_SINGLE_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "target": "same_date_percentile_of_pure_mfe_residual_after_same_date_OLS_on_true_low_adverse_safety_percentile",
    "target_conditioning": "supervision_only_true_safety_percentile_no_future_feature",
    "architecture": "single_head_inception_time",
    "head_losses": "full_list_delta_ndcg_pairwise_logistic",
    "epoch_selection": "conditional_mfe_mean_daily_spearman",
    "runtime_score": "conditional_mfe_pass_probability",
    "batching": "whole_date_pack_no_date_split",
}

SAFETY_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "safety_target": "same_date_low_adverse_safety_percentile",
    "conditional_mfe_target": "same_date_percentile_of_pure_mfe_residual_after_same_date_OLS_on_true_low_adverse_safety_percentile",
    "target_conditioning": "supervision_only_true_safety_percentile_no_future_feature",
    "architecture": "shared_encoder_raw_safety_head_plus_safety_conditioned_mfe_head",
    "conditional_context": "stop_gradient_raw_safety_probability",
    "safety_head_gradient_from_conditional_loss": False,
    "shared_encoder_gradient_from_both_heads": True,
    "head_losses": "full_list_delta_ndcg_pairwise_logistic_each_head",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "conditional_mfe_mean_daily_spearman",
    "runtime_score": "conditional_mfe_pass_probability_only",
    "batching": "whole_date_pack_no_date_split",
}

SAFETY_RAW_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "safety_target": "same_date_low_adverse_safety_percentile",
    "raw_mfe_target": "same_date_pure_mfe_percentile",
    "architecture": "shared_encoder_raw_safety_head_plus_safety_conditioned_mfe_head",
    "conditional_context": "stop_gradient_raw_safety_probability",
    "safety_head_gradient_from_mfe_loss": False,
    "shared_encoder_gradient_from_both_heads": True,
    "head_losses": "full_list_delta_ndcg_pairwise_logistic_each_head",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "raw_mfe_mean_daily_spearman",
    "runtime_status": "model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}

SHARED_SAFETY_WEIGHTED_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "safety_target": "same_date_low_adverse_safety_percentile",
    "raw_mfe_target": "same_date_pure_mfe_percentile",
    "architecture": "shared_encoder_raw_safety_plus_final_mfe_topology_defined_by_model_spec",
    "mfe_head_inputs": "defined_by_model_architecture_contract",
    "mfe_pair_context": "stop_gradient_same_date_average_rank_percentile_of_raw_safety_probability",
    "mfe_pair_safety_weight": "same_date_predicted_safety_percentile_i_times_j",
    "mfe_pair_weight_combination": "delta_ndcg_times_product_detached_same_date_model_safety_percentile",
    "mfe_pair_direction": "pure_mfe_only_never_reversed_by_safety",
    "safety_head_gradient_from_mfe_loss": False,
    "shared_encoder_gradient_from_both_heads": True,
    "head_losses": "safety_full_list_delta_ndcg_plus_safety_product_weighted_mfe_full_list_delta_ndcg",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "external_predicted_safety_dependency": False,
    "epoch_selection": "raw_mfe_mean_daily_spearman",
    "runtime_score": "raw_mfe_pass_probability_only",
    "runtime_status": "model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}

SHARED_SAFETY_HS_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days_full_universe_encoder_exposure",
    "safety_target": "same_date_low_adverse_safety_percentile_over_full_universe",
    "conditional_mfe_target": "same_date_pure_mfe_percentile_within_true_hs_cohort",
    "true_hs_definition": "same_date_low_adverse_safety_percentile_gte_0.50",
    "architecture": "shared_encoder_independent_raw_safety_and_conditional_mfe_heads",
    "conditional_mfe_head_inputs": "shared_latent_only_no_predicted_safety_context",
    "safety_supervision_scope": "all_rows_same_date_full_list_delta_ndcg",
    "conditional_mfe_supervision_scope": "true_hs_items_only_sublist_before_rank_positions_idcg_and_delta_ndcg",
    "ls_conditional_mfe_membership": "zero_pairs_zero_rank_position_zero_idcg_zero_delta_ndcg",
    "conditional_mfe_pair_safety_weight": "none",
    "shared_encoder_gradient": "safety_all_rows_plus_conditional_mfe_true_hs_only",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "hs_conditional_mfe_mean_daily_spearman",
    "runtime_score": "conditional_mfe_pass_probability_after_predicted_safety_qualification",
    "runtime_status": "seed42_forward_model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days_full_universe_encoder_exposure",
    "qualification_target": "indicator_of_same_date_low_adverse_safety_percentile_gte_0.50",
    "qualification_pair_scope": "same_date_hs_vs_ls_only_same_cohort_ties_excluded",
    "conditional_mfe_target": "same_date_pure_mfe_percentile_within_true_hs_cohort",
    "true_hs_definition": "same_date_low_adverse_safety_percentile_gte_0.50",
    "architecture": "shared_encoder_independent_hs_qualification_and_conditional_mfe_heads",
    "conditional_mfe_head_inputs": "shared_latent_only_no_predicted_safety_context",
    "qualification_supervision_scope": "all_rows_but_only_hs_vs_ls_pairs_have_direction",
    "conditional_mfe_supervision_scope": "true_hs_items_only_sublist_before_rank_positions_idcg_and_delta_ndcg",
    "ls_conditional_mfe_membership": "zero_pairs_zero_rank_position_zero_idcg_zero_delta_ndcg",
    "conditional_mfe_pair_safety_weight": "none",
    "shared_encoder_gradient": "hs_qualification_all_rows_plus_conditional_mfe_true_hs_only",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "hs_conditional_mfe_mean_daily_spearman_same_as_mr13ao",
    "runtime_score": "conditional_mfe_pass_probability_after_predicted_hs_qualification",
    "runtime_status": "seed42_forward_model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}

SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days_full_universe_encoder_exposure",
    "continuous_safety_target": "same_date_low_adverse_safety_percentile_over_full_universe",
    "qualification_target": "indicator_of_same_date_low_adverse_safety_percentile_gte_0.50",
    "safety_head_supervision": "same_logits_dual_supervision_continuous_safety_plus_binary_hs",
    "safety_branch_loss": "fixed_equal_mean_of_continuous_safety_and_binary_hs_pairwise_losses",
    "conditional_mfe_target": "same_date_pure_mfe_percentile_within_true_hs_cohort",
    "true_hs_definition": "same_date_low_adverse_safety_percentile_gte_0.50",
    "architecture": "same_mr13ar_shared_encoder_independent_safety_and_conditional_mfe_heads",
    "conditional_mfe_head_inputs": "shared_latent_only_no_predicted_safety_context",
    "continuous_safety_supervision_scope": "all_rows_same_date_full_list_delta_ndcg",
    "qualification_supervision_scope": "all_rows_but_only_hs_vs_ls_pairs_have_direction",
    "conditional_mfe_supervision_scope": "true_hs_items_only_sublist_before_rank_positions_idcg_and_delta_ndcg",
    "ls_conditional_mfe_membership": "zero_pairs_zero_rank_position_zero_idcg_zero_delta_ndcg",
    "conditional_mfe_pair_safety_weight": "none",
    "shared_encoder_gradient": "dual_supervised_safety_all_rows_plus_conditional_mfe_true_hs_only",
    "head_weighting": "safety_branch_0.5_conditional_mfe_0.5_with_safety_branch_split_0.5_0.5",
    "effective_component_weights": "continuous_safety_0.25_binary_hs_0.25_conditional_mfe_0.50_no_sweep",
    "epoch_selection": "hs_conditional_mfe_mean_daily_spearman_same_as_mr13ar",
    "runtime_score": "conditional_mfe_pass_probability_after_predicted_hs_qualification_same_safety_head",
    "runtime_status": "seed42_forward_model_gate_only_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days_full_universe_encoder_exposure",
    "safety_target": "ls_relevance_zero_hs_relevance_same_date_low_adverse_safety_percentile",
    "true_hs_definition": "same_date_low_adverse_safety_percentile_gte_0.50",
    "safety_top_k": "per_date_true_hs_item_count_no_k_sweep",
    "safety_ranking_geometry": "delta_ndcg_at_true_hs_count_predicted_discounts_zero_below_k",
    "safety_within_hs_priority": "higher_low_adverse_safety_percentile_has_higher_relevance",
    "conditional_mfe_target": "same_date_pure_mfe_percentile_within_true_hs_cohort",
    "architecture": "same_shared_encoder_independent_safety_and_conditional_mfe_heads",
    "conditional_mfe_head_inputs": "shared_latent_only_no_predicted_safety_context",
    "conditional_mfe_supervision_scope": "true_hs_items_only_sublist_before_rank_positions_idcg_and_delta_ndcg",
    "ls_conditional_mfe_membership": "zero_pairs_zero_rank_position_zero_idcg_zero_delta_ndcg",
    "conditional_mfe_pair_safety_weight": "none",
    "shared_encoder_gradient": "top_hs_safety_all_rows_plus_conditional_mfe_true_hs_only",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "hs_conditional_mfe_mean_daily_spearman_same_as_mr13at",
    "runtime_score": "conditional_mfe_pass_probability_after_predicted_hs_top_half_qualification",
    "runtime_status": "seed42_forward_model_gate_only_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


SHARED_SAFETY_HS_PRIORITY_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days_full_universe_encoder_and_final_head_supervision",
    "safety_target": "same_date_low_adverse_safety_percentile_over_full_universe",
    "priority_mfe_target": "ls_equals_0_else_0.5_plus_0.5_times_same_date_mfe_percentile_within_true_hs",
    "true_hs_definition": "same_date_low_adverse_safety_percentile_gte_0.50",
    "ranking_constraint": "every_true_hs_strictly_above_every_true_ls_then_mfe_order_within_hs",
    "ls_ordering": "all_true_ls_tied_at_worst_relevance_no_ls_internal_direction",
    "architecture": "shared_encoder_independent_raw_safety_and_hs_priority_mfe_heads",
    "priority_head_inputs": "shared_latent_only_no_predicted_safety_context",
    "safety_supervision_scope": "all_rows_same_date_full_list_delta_ndcg",
    "priority_supervision_scope": "all_rows_same_date_full_list_delta_ndcg",
    "priority_pair_safety_weight": "none",
    "shared_encoder_gradient": "safety_all_rows_plus_priority_all_rows",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "hs_priority_mfe_mean_daily_spearman",
    "runtime_score": "hs_priority_mfe_pass_probability_direct_all_daily_ranking",
    "runtime_status": "seed42_forward_model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days_full_universe_encoder_and_final_head_supervision",
    "safety_target": "same_date_low_adverse_safety_percentile_over_full_universe",
    "priority_mfe_target": "mr13ap_truth_ls_equals_0_else_0.5_plus_0.5_times_same_date_mfe_percentile_within_true_hs",
    "true_hs_definition": "same_date_low_adverse_safety_percentile_gte_0.50",
    "ranking_constraint": "every_true_hs_strictly_above_every_true_ls_then_mfe_order_within_hs",
    "ls_ordering": "all_true_ls_tied_at_worst_relevance_no_ls_internal_direction",
    "architecture": "shared_encoder_independent_raw_safety_and_hs_priority_mfe_heads",
    "priority_head_inputs": "shared_latent_only_no_predicted_safety_context",
    "safety_supervision_scope": "all_rows_same_date_full_list_delta_ndcg",
    "priority_supervision_scope": "all_rows_same_date_full_list_delta_ndcg_pair_stratified",
    "priority_pair_strata": "hs_vs_ls_boundary_and_hs_vs_hs_upside",
    "priority_stratum_geometry": "same_full_list_predicted_rank_positions_idcg_and_delta_ndcg",
    "priority_stratum_normalization": "each_stratum_normalized_by_own_delta_ndcg_weight_sum_then_fixed_equal_mean",
    "priority_pair_safety_weight": "none",
    "shared_encoder_gradient": "safety_all_rows_plus_pair_stratified_priority_all_rows",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "epoch_selection": "hs_priority_mfe_mean_daily_spearman",
    "runtime_score": "hs_priority_mfe_pass_probability_direct_all_daily_ranking",
    "runtime_status": "seed42_forward_model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


SHARED_SAFETY_WEIGHTED_PRIMARY_DUO_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "safety_target": "same_date_low_adverse_safety_percentile",
    "primary_target": "same_date_percentile_of_profile_continuous_target",
    "architecture": "shared_encoder_raw_safety_plus_final_primary_topology_defined_by_model_spec",
    "primary_head_inputs": "defined_by_model_architecture_contract",
    "primary_pair_context": "stop_gradient_same_date_average_rank_percentile_of_raw_safety_probability",
    "primary_pair_safety_weight": "same_date_predicted_safety_percentile_i_times_j",
    "primary_pair_weight_combination": "delta_ndcg_times_product_detached_same_date_model_safety_percentile",
    "primary_pair_direction": "profile_primary_target_only_never_reversed_by_safety",
    "safety_head_gradient_from_primary_loss": False,
    "shared_encoder_gradient_from_both_heads": True,
    "head_losses": "safety_full_list_delta_ndcg_plus_safety_product_weighted_primary_full_list_delta_ndcg",
    "head_weighting": "fixed_equal_mean_no_lambda_sweep",
    "external_predicted_safety_dependency": False,
    "epoch_selection": "primary_target_mean_daily_spearman",
    "runtime_score": "primary_target_pass_probability_only",
    "runtime_status": "model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


SAFETY_RAW_MFE_HMHS_TRI_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "safety_target": "same_date_low_adverse_safety_percentile",
    "raw_mfe_target": "same_date_pure_mfe_percentile",
    "joint_hmhs_target": "indicator_of_safety_percentile_ge_0.5_and_pure_mfe_percentile_ge_0.5",
    "architecture": "shared_encoder_raw_safety_plus_safety_conditioned_raw_mfe_plus_direct_hmhs_head",
    "conditional_context": "stop_gradient_raw_safety_probability_for_raw_mfe_head_only",
    "joint_head_inputs": "shared_raw_latent_only_no_safety_or_mfe_score_arithmetic",
    "classifier_isolation": "each_head_loss_updates_own_classifier_only",
    "shared_encoder_gradient_from_all_heads": True,
    "head_losses": "full_list_delta_ndcg_pairwise_logistic_each_head",
    "head_weighting": "fixed_equal_mean_three_heads_no_lambda_sweep",
    "epoch_selection": "raw_mfe_mean_daily_spearman_same_as_mr13s_control",
    "runtime_status": "model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}

SAFETY_RAW_MFE_JOINT_MIN_TRI_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "safety_target": "same_date_low_adverse_safety_percentile",
    "raw_mfe_target": "same_date_pure_mfe_percentile",
    "joint_min_target": "min_same_date_safety_and_pure_mfe_percentiles",
    "architecture": "shared_encoder_raw_safety_plus_safety_conditioned_raw_mfe_plus_nonlinear_joint_head",
    "conditional_context": "stop_gradient_raw_safety_probability_for_raw_mfe_head_only",
    "joint_head_inputs": "shared_raw_latent_only_no_safety_or_mfe_score_arithmetic",
    "classifier_isolation": "each_head_loss_updates_own_classifier_only",
    "shared_encoder_gradient_from_all_heads": True,
    "head_losses": "full_list_delta_ndcg_pairwise_logistic_each_head",
    "head_weighting": "fixed_equal_mean_three_heads_no_lambda_sweep",
    "epoch_selection": "raw_mfe_mean_daily_spearman_same_as_mr13v_control",
    "runtime_status": "model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}

DIRECT_HMHS_SINGLE_HEAD_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "joint_hmhs_target": "indicator_of_safety_percentile_ge_0.5_and_pure_mfe_percentile_ge_0.5",
    "architecture": "single_head_inception_time_raw_300x10",
    "joint_head_inputs": "shared_raw_latent_only",
    "removed_auxiliary_heads": "raw_safety_and_raw_mfe",
    "head_losses": "full_list_delta_ndcg_pairwise_logistic",
    "epoch_selection": "validation_hmhs_pairwise_concordance_then_global_pr_auc_tiebreak",
    "runtime_status": "model_gate_only_no_pit_no_strategy_conversion",
    "batching": "whole_date_pack_no_date_split",
}


RAW_R_REGRESSION_TRAINING_CONTRACT = {
    "sample_scope": "daily_eligible_stock_days",
    "target": "daily_opportunity_no_time_r_v1_raw_r",
    "prediction": "pass_logit_minus_reject_logit_margin_in_r_units",
    "batching": "shuffled_unique_group_batches",
    "runtime_score": "predicted_r",
}

LISTWISE_TRAINING_CONTRACT = {
    "list_scope": "same_date_full_candidate_list",
    "target_distribution": "softmax_daily_percentile",
    "prediction_distribution": "softmax_pass_minus_reject_margin",
    "tie_handling": "equal_target_equal_distribution_weight",
    "date_weighting": "equal_rankable_date_weight",
    "model_margin": "pass_logit_minus_reject_logit",
    "batching": "whole_date_pack_no_date_split",
    "runtime_score": "softmax_pass_probability",
}


_EXTENDED_SEMANTIC_KEYS = (
    "pairwise_contract",
    "listwise_contract",
    "raw_r_regression_contract",
    "dual_component_r_regression_contract",
    "conditional_mfe_safety_contract",
    "conditional_mfe_single_head_contract",
    "safety_conditional_mfe_duo_head_contract",
    "safety_raw_mfe_duo_head_contract",
    "shared_safety_weighted_mfe_duo_head_contract",
    "shared_safety_weighted_primary_duo_head_contract",
    "safety_raw_mfe_hmhs_tri_head_contract",
    "safety_raw_mfe_joint_min_tri_head_contract",
    "direct_hmhs_single_head_contract",
)


def _extended_semantics(*, batching: str, **contracts: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"batching": str(batching)}
    result.update({key: None for key in _EXTENDED_SEMANTIC_KEYS})
    result.update(contracts)
    return result


def _pairwise_contract_for_recipe(recipe) -> dict[str, Any]:
    contract = dict(PAIRWISE_TRAINING_CONTRACT)
    contract["pair_weighting"] = str(recipe.pairwise_reduction)
    context_policy = recipe.context_policy
    if context_policy.source == CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE:
        contract["predicted_upside_context_contract"] = get_predicted_upside_context_contract()
    if context_policy.source == CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY:
        if context_policy.has_role(CONTINUOUS_RANKER_CONTEXT_ROLE_TARGET_TRANSFORM):
            contract["predicted_safety_context_contract"] = get_predicted_safety_context_contract()
        elif context_policy.has_role(CONTINUOUS_RANKER_CONTEXT_ROLE_MODEL_INPUT):
            contract["predicted_safety_context_contract"] = get_predicted_safety_pure_mfe_contract()
        if context_policy.has_role(CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT):
            contract["predicted_safety_pair_weight_contract"] = get_predicted_safety_pair_weight_contract(recipe.objective_policy.pair_weight_policy)
    primary_truth_weight_policy = get_continuous_ranker_primary_pair_weight_policy(recipe.training_objective)
    if primary_truth_weight_policy != CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE:
        contract["primary_truth_pair_weight_policy"] = primary_truth_weight_policy
        if primary_truth_weight_policy == CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY:
            contract["primary_truth_pair_weight_formula"] = "1_minus_abs_same_date_safety_percentile_pair_gap"
            contract["primary_truth_pair_weight_role"] = "supervision_weight_only_never_pair_direction_or_model_input"
    if recipe.objective_policy.pair_target_schema == CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_PARETO_COMPONENTS:
        contract.update({
            "pair_scope": "same_date_strict_pareto_dominance_pairs",
            "target_components": [
                "same_date_mfe_percentile",
                "same_date_low_adverse_percentile",
            ],
            "tradeoff_pair_handling": "excluded_no_gradient",
            "tie_handling": "excluded_no_gradient",
            "epoch_selection": "mean_daily_pareto_pair_concordance",
        })
    return contract


_PAIRWISE_SPECIALIZED_CONTRACTS = {
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SAFETY: (
        "conditional_mfe_safety_contract",
        CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_CONDITIONAL_MFE_SINGLE: (
        "conditional_mfe_single_head_contract",
        CONDITIONAL_MFE_SINGLE_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_CONDITIONAL_MFE: (
        "safety_conditional_mfe_duo_head_contract",
        SAFETY_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE: (
        "safety_raw_mfe_duo_head_contract",
        SAFETY_RAW_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE: (
        "shared_safety_weighted_mfe_duo_head_contract",
        SHARED_SAFETY_WEIGHTED_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY: (
        "shared_safety_weighted_primary_duo_head_contract",
        SHARED_SAFETY_WEIGHTED_PRIMARY_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_CONDITIONAL_MFE: (
        "shared_safety_hs_conditional_mfe_duo_head_contract",
        SHARED_SAFETY_HS_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE: (
        "shared_hs_qualification_conditional_mfe_duo_head_contract",
        SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE: (
        "shared_dual_supervised_hs_conditional_mfe_duo_head_contract",
        SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE: (
        "shared_top_hs_safety_conditional_mfe_duo_head_contract",
        SHARED_TOP_HS_SAFETY_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_MFE: (
        "shared_safety_hs_priority_mfe_duo_head_contract",
        SHARED_SAFETY_HS_PRIORITY_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE: (
        "shared_safety_hs_priority_stratified_mfe_duo_head_contract",
        SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_DUO_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_HMHS: (
        "safety_raw_mfe_hmhs_tri_head_contract",
        SAFETY_RAW_MFE_HMHS_TRI_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_SAFETY_RAW_MFE_JOINT_MIN: (
        "safety_raw_mfe_joint_min_tri_head_contract",
        SAFETY_RAW_MFE_JOINT_MIN_TRI_HEAD_TRAINING_CONTRACT,
    ),
    CONTINUOUS_RANKER_SEMANTICS_DIRECT_HMHS: (
        "direct_hmhs_single_head_contract",
        DIRECT_HMHS_SINGLE_HEAD_TRAINING_CONTRACT,
    ),
}


def training_semantics(profile) -> dict[str, Any]:
    """Return canonical artifact semantics for one continuous-ranker profile."""

    recipe = get_continuous_ranker_execution_recipe(profile.name)
    semantics_key = str(recipe.training_policy.semantics_contract_key)

    if semantics_key in _PAIRWISE_SPECIALIZED_CONTRACTS:
        slot, base_contract = _PAIRWISE_SPECIALIZED_CONTRACTS[semantics_key]
        pairwise_contract = _pairwise_contract_for_recipe(recipe)
        contract = dict(base_contract)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        primary_truth_weight_policy = get_continuous_ranker_primary_pair_weight_policy(recipe.training_objective)
        if primary_truth_weight_policy != CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE:
            contract["qualification_pair_weighting"] = primary_truth_weight_policy
            if primary_truth_weight_policy == CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY:
                contract["qualification_pair_weight_formula"] = "1_minus_abs_same_date_safety_percentile_pair_gap"
                contract["qualification_pair_weight_role"] = "truth_side_supervision_only_no_model_input_no_direction_change"
        if semantics_key == CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_MFE:
            # Artifact semantics describe the selected topology from the canonical model
            # spec, while the loss/target contract remains experiment-agnostic. This
            # preserves exact A1 stored semantics and gives A2 its own truthful topology
            # description without branching on MR/profile/architecture identity.
            from filters.breakout_quality.models.spec import get_model_spec

            model_spec = get_model_spec(str(profile.model_architecture))
            topology_contract = model_spec.final_mfe_topology_contract()
            if topology_contract is None:
                raise ValueError(
                    "shared Safety-weighted MFE semantics需要canonical final-MFE topology contract"
                )
            contract.update(topology_contract)
        elif semantics_key == CONTINUOUS_RANKER_SEMANTICS_SHARED_SAFETY_WEIGHTED_PRIMARY:
            from filters.breakout_quality.models.spec import get_model_spec

            model_spec = get_model_spec(str(profile.model_architecture))
            topology_contract = model_spec.final_mfe_topology_contract()
            if topology_contract is None:
                raise ValueError(
                    "shared Safety-weighted primary semantics需要canonical final-head topology contract"
                )
            contract["architecture"] = topology_contract["architecture"]
            contract["primary_head_inputs"] = topology_contract["mfe_head_inputs"]
            contract["primary_target_id"] = str(profile.continuous_target_id)
        return _extended_semantics(
            batching=contract["batching"],
            pairwise_contract=pairwise_contract,
            **{slot: contract},
        )

    if semantics_key == CONTINUOUS_RANKER_SEMANTICS_PAIRWISE:
        pairwise_contract = _pairwise_contract_for_recipe(recipe)
        return _extended_semantics(
            batching=pairwise_contract["batching"],
            pairwise_contract=pairwise_contract,
        )

    if semantics_key == CONTINUOUS_RANKER_SEMANTICS_LISTWISE:
        return _extended_semantics(
            batching=LISTWISE_TRAINING_CONTRACT["batching"],
            listwise_contract=dict(LISTWISE_TRAINING_CONTRACT),
        )

    if semantics_key == CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R:
        contract = dict(DUAL_COMPONENT_R_REGRESSION_TRAINING_CONTRACT)
        return _extended_semantics(
            batching=contract["batching"],
            dual_component_r_regression_contract=contract,
        )

    if semantics_key == CONTINUOUS_RANKER_SEMANTICS_RAW_R:
        contract = dict(RAW_R_REGRESSION_TRAINING_CONTRACT)
        contract["loss"] = str(profile.loss_name)
        contract["huber_delta_r"] = (
            None
            if profile.raw_r_huber_delta_r is None
            else float(profile.raw_r_huber_delta_r)
        )
        return _extended_semantics(
            batching=contract["batching"],
            raw_r_regression_contract=contract,
        )

    if semantics_key != CONTINUOUS_RANKER_SEMANTICS_DEFAULT:
        raise ValueError(f"unsupported continuous-ranker semantics capability: {semantics_key}")
    # Preserve the historical compact payload for percentile regression profiles.
    return {
        "batching": "shuffled_unique_group_batches",
        "pairwise_contract": None,
        "listwise_contract": None,
        "raw_r_regression_contract": None,
        "dual_component_r_regression_contract": None,
        "conditional_mfe_safety_contract": None,
    }


def training_semantics_mismatches(profile, payload) -> list[str]:
    """Return objective-relevant semantic mismatches for one stored artifact.

    Historical artifacts may omit keys whose canonical value is ``None`` because
    those keys were added later for other objectives.  Non-null semantics remain
    strict, and unknown keys are rejected so schema evolution cannot silently
    change the trained objective.
    """

    expected = training_semantics(profile)
    actual = dict(payload or {})
    mismatches: list[str] = []

    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        mismatches.append(f"unexpected_keys={unexpected}")

    for key, expected_value in expected.items():
        if expected_value is None:
            if key in actual and actual[key] not in (None, {}):
                mismatches.append(
                    f"{key}: expected non-applicable, actual={actual[key]!r}"
                )
            continue
        if key not in actual:
            mismatches.append(f"{key}: missing")
            continue
        if actual[key] != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r}, actual={actual[key]!r}"
            )
    return mismatches


__all__ = [
    "CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT",
    "DUAL_COMPONENT_R_REGRESSION_TRAINING_CONTRACT",
    "LISTWISE_TRAINING_CONTRACT",
    "PAIRWISE_TRAINING_CONTRACT",
    "RAW_R_REGRESSION_TRAINING_CONTRACT",
    "SAFETY_RAW_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SHARED_SAFETY_WEIGHTED_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SHARED_SAFETY_HS_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SHARED_DUAL_SUPERVISED_HS_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SHARED_SAFETY_HS_PRIORITY_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_DUO_HEAD_TRAINING_CONTRACT",
    "SAFETY_RAW_MFE_HMHS_TRI_HEAD_TRAINING_CONTRACT",
    "SAFETY_RAW_MFE_JOINT_MIN_TRI_HEAD_TRAINING_CONTRACT",
    "training_semantics",
    "training_semantics_mismatches",
]
