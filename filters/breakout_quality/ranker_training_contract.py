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
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    get_continuous_ranker_execution_recipe,
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

    if profile.training_objective in {
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    }:
        recipe = get_continuous_ranker_execution_recipe(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(recipe.pairwise_reduction)
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
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING:
        return {
            "batching": LISTWISE_TRAINING_CONTRACT["batching"],
            "pairwise_contract": None,
            "listwise_contract": dict(LISTWISE_TRAINING_CONTRACT),
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
        contract = dict(DUAL_COMPONENT_R_REGRESSION_TRAINING_CONTRACT)
        return {
            "batching": contract["batching"],
            "pairwise_contract": None,
            "listwise_contract": None,
            "raw_r_regression_contract": None,
            "dual_component_r_regression_contract": contract,
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
        }
    return {
        "batching": "shuffled_unique_group_batches",
        "pairwise_contract": None,
        "listwise_contract": None,
        "raw_r_regression_contract": None,
        "dual_component_r_regression_contract": None,
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
    "DUAL_COMPONENT_R_REGRESSION_TRAINING_CONTRACT",
    "LISTWISE_TRAINING_CONTRACT",
    "PAIRWISE_TRAINING_CONTRACT",
    "RAW_R_REGRESSION_TRAINING_CONTRACT",
    "training_semantics",
    "training_semantics_mismatches",
]
