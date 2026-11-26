"""Declarative FIT-only sample weighting; never an input/Target/selector policy."""
from __future__ import annotations

ORIGINAL_SIGNAL_INVERSE_FREQUENCY = "original_signal_inverse_frequency_v1"
EPISODE_COLUMN = "fit_original_signal_episode"


def fit_sample_weight_contract(policy: str | None) -> dict | None:
    if policy is None:
        return None
    if policy != ORIGINAL_SIGNAL_INVERSE_FREQUENCY:
        raise ValueError(f"Unknown FIT sample weight policy: {policy!r}")
    return {
        "policy": policy,
        "episode": "same_seed_ticker_canonical_original_signal_date",
        "count": "unique_legal_ticker_information_date_in_this_fit_stage_only",
        "item_weight": "1 / episode_fit_row_count",
        "pair_multiplier": "item_weight_i * item_weight_j",
        "loss": "sum(delta_ndcg * pair_multiplier * logistic) / sum(delta_ndcg * pair_multiplier)",
        "validation": "original_unweighted_spearman_and_percentile_mse",
        "refit": "recount_from_canonical_final_ids",
        "role": "detached_supervision_metadata_only",
    }
