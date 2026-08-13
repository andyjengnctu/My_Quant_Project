"""Shared continuous-ranker training semantics contract.

Trainer artifact producers and runtime artifact validators must resolve the same
profile-driven semantics from this module.  In particular, pairwise weighting is
owned by ``ContinuousRankerResearchSpec.pairwise_reduction`` rather than by a
consumer-specific hard-coded default.
"""

from __future__ import annotations

from typing import Any

from config.breakout_quality import (
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    get_continuous_ranker_research_spec,
)


PAIRWISE_TRAINING_CONTRACT = {
    "pair_scope": "same_date_non_tied_target_pairs",
    "pair_weighting": CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    "model_margin": "pass_logit_minus_reject_logit",
    "batching": "whole_date_pack_no_date_split",
    "runtime_score": "softmax_pass_probability",
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

    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING:
        spec = get_continuous_ranker_research_spec(profile.name)
        pairwise_contract = dict(PAIRWISE_TRAINING_CONTRACT)
        pairwise_contract["pair_weighting"] = str(spec.pairwise_reduction)
        return {
            "batching": pairwise_contract["batching"],
            "pairwise_contract": pairwise_contract,
            "listwise_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING:
        return {
            "batching": LISTWISE_TRAINING_CONTRACT["batching"],
            "pairwise_contract": None,
            "listwise_contract": dict(LISTWISE_TRAINING_CONTRACT),
        }
    return {
        "batching": "shuffled_unique_group_batches",
        "pairwise_contract": None,
        "listwise_contract": None,
    }


__all__ = [
    "LISTWISE_TRAINING_CONTRACT",
    "PAIRWISE_TRAINING_CONTRACT",
    "training_semantics",
]
