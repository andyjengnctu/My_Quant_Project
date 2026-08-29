"""Shared continuous-ranker training semantics contract.

Trainer artifact producers and runtime artifact validators must resolve the same
profile-driven semantics from this module.  In particular, pairwise weighting is
owned by the experiment-agnostic execution recipe rather than by research
identity or a consumer-specific hard-coded default.
"""

from __future__ import annotations

from typing import Any

from config.breakout_quality import (
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    get_continuous_ranker_execution_recipe,
    PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
    get_predicted_upside_context_contract,
    PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
    PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
    get_predicted_safety_context_contract,
    get_predicted_safety_pure_mfe_contract,
    get_high_safety_weighted_pure_mfe_contract,
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


def training_semantics(profile) -> dict[str, Any]:
    """Return canonical artifact semantics for one continuous-ranker profile."""

    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        contract = dict(CONDITIONAL_MFE_SINGLE_HEAD_TRAINING_CONTRACT)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": contract,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        contract = dict(SAFETY_CONDITIONAL_MFE_DUO_HEAD_TRAINING_CONTRACT)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": contract,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        contract = dict(SAFETY_RAW_MFE_DUO_HEAD_TRAINING_CONTRACT)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": contract,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        contract = dict(DIRECT_HMHS_SINGLE_HEAD_TRAINING_CONTRACT)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": contract,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        contract = dict(SAFETY_RAW_MFE_HMHS_TRI_HEAD_TRAINING_CONTRACT)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": contract,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        contract = dict(SAFETY_RAW_MFE_JOINT_MIN_TRI_HEAD_TRAINING_CONTRACT)
        contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": contract,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        conditional_contract = dict(CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT)
        conditional_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        return {
            "batching": conditional_contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": conditional_contract,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective in {
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    }:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
        target_id = str(profile.continuous_target_id or "")
        if target_id == PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID:
            pairwise_contract["predicted_upside_context_contract"] = get_predicted_upside_context_contract()
        if target_id == PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID:
            pairwise_contract["predicted_safety_context_contract"] = get_predicted_safety_context_contract()
        if target_id == PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID:
            pairwise_contract["predicted_safety_context_contract"] = get_predicted_safety_pure_mfe_contract()
        if str(recipe.pairwise_reduction) == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG:
            pairwise_contract["predicted_safety_pair_weight_contract"] = get_high_safety_weighted_pure_mfe_contract()
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING:
            pairwise_contract.update({
                "pair_scope": "same_date_strict_pareto_dominance_pairs",
                "target_components": [
                    "same_date_mfe_percentile",
                    "same_date_low_adverse_percentile",
                ],
                "tradeoff_pair_handling": "excluded_no_gradient",
                "tie_handling": "excluded_no_gradient",
                "epoch_selection": "mean_daily_pareto_pair_concordance",
            })
        return {
            "batching": pairwise_contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING:
        return {
            "batching": LISTWISE_TRAINING_CONTRACT["batching"],
            "pairwise_contract": None,
            "listwise_contract": dict(LISTWISE_TRAINING_CONTRACT),
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
        contract = dict(DUAL_COMPONENT_R_REGRESSION_TRAINING_CONTRACT)
        return {
            "batching": contract["batching"],
            "pairwise_contract": None,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": contract,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
        contract = dict(RAW_R_REGRESSION_TRAINING_CONTRACT)
        contract["loss"] = str(profile.loss_name)
        contract["huber_delta_r"] = (
            None
            if profile.raw_r_huber_delta_r is None
            else float(profile.raw_r_huber_delta_r)
        )
        return {
            "batching": contract["batching"],
            "pairwise_contract": None,
            "listwise_contract": None,
            "raw_r_regression_contract": contract,
            "dual_component_r_regression_contract": None,
            "conditional_mfe_safety_contract": None,
            "conditional_mfe_single_head_contract": None,
            "safety_conditional_mfe_duo_head_contract": None,
            "safety_raw_mfe_duo_head_contract": None,
            "safety_raw_mfe_hmhs_tri_head_contract": None,
            "safety_raw_mfe_joint_min_tri_head_contract": None,
            "direct_hmhs_single_head_contract": None,
        }
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
    "SAFETY_RAW_MFE_HMHS_TRI_HEAD_TRAINING_CONTRACT",
    "SAFETY_RAW_MFE_JOINT_MIN_TRI_HEAD_TRAINING_CONTRACT",
    "training_semantics",
    "training_semantics_mismatches",
]
