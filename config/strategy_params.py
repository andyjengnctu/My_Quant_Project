"""Research strategy-parameter producer configuration.

The Research menu selects only a stable work type.  The active scientific
strategy-parameter target is selected here, analogous to ``ACTIVE_MODEL_ID`` in
``config.research``.  Optimizer execution code consumes the resolved profile;
it must not infer the current target from menu choices or Strategy Compare arm
IDs.
"""

from __future__ import annotations

from config.research import RESEARCH_MARKET_DATA_CUTOFF


# Current Research strategy-parameter target.  Changing this value changes what
# Research [2] trains; the interactive menu must not offer a second target selector.
# Round 6 activates the single-seed adaptive plateau validation producer.
# PARAM-P5 fixed-checkpoint validation and PARAM-P6~P8 fitting-budget controls remain
# immutable registered references; only this current Research target changes.
ACTIVE_STRATEGY_PARAM_PROFILE = "validation_plateau_current"


# Validation is a reusable Optimizer capability.  Profiles opt in/out below;
# protocol settings live here rather than inside an experiment-specific engine.
STRATEGY_PARAM_VALIDATION_PROTOCOLS: dict[str, dict] = {
    "validation_budget_refit_v1": {
        "holdout_months": 12,
        "checkpoints": (50, 100, 150, 200, 250, 300),
        "selection_policy": "base_finalists_agree",
        "selection_metric": "canonical_optimizer_portfolio_score",
        "tie_break": "smaller_budget",
        "oos_feedback_used": False,
        "full_refit_required": True,
    },
    "validation_plateau_refit_v2": {
        "holdout_months": 12,
        "warmup_trials": 50,
        "patience_trials": 50,
        "observation_interval": 1,
        "min_delta": 0.0,
        "selection_policy": "base_finalists_agree",
        "selection_metric": "canonical_optimizer_portfolio_score",
        "tie_break": "smaller_budget",
        "oos_feedback_used": False,
        "full_refit_required": True,
        "reuse_same_policy_signature": True,
    },
}


# Strategy fitting stochasticity is a scientific dimension independent from
# Robustness replication.  Trading may intentionally draw a fresh random bundle
# on each production retrain; controlled Research arms pin the bundle so arm-to-
# arm comparisons are reproducible.  The bundle itself is derived by the policy
# resolver rather than duplicated here as literal seed values.
STRATEGY_PARAM_FITTING_SEED_PROTOCOLS: dict[str, dict] = {
    "single_seed_42": {
        "enabled": False,
        "seed_count": 1,
        "seed_mode": "fixed_single_seed",
        "generator_seed": 42,
        "min_agree": 1,
        "trials_per_seed": 300,
    },
    "research_fixed_multiseed_v1": {
        "enabled": True,
        "seed_count": 4,
        "seed_mode": "fixed_generated_bundle",
        "generator_seed": 42,
        "min_agree": "auto",
        "trials_per_seed": 300,
    },
    "research_fixed_multiseed_8x300_v1": {
        "enabled": True,
        "seed_count": 8,
        "seed_mode": "fixed_generated_bundle",
        "generator_seed": 42,
        "min_agree": "auto",
        "trials_per_seed": 300,
    },
    "research_fixed_multiseed_4x1000_v1": {
        "enabled": True,
        "seed_count": 4,
        "seed_mode": "fixed_generated_bundle",
        "generator_seed": 42,
        "min_agree": "auto",
        "trials_per_seed": 1000,
    },
}


# Scientific producer profiles.  Min/Full are filter-profile dimensions, not menu
# work types.  Validation is opt-in per profile and may be disabled for another
# strategy-parameter target without changing the Research menu or Optimizer engine.
STRATEGY_PARAM_CANONICAL_PROFILE_BY_FAMILY_POLICY: dict[tuple[str, str], str] = {
    ("min", "base-finalist-best"): "canonical_min_base_finalist_best",
    ("full", "base-finalist-best"): "canonical_full_base_finalist_best",
    ("full", "base-finalists-agree"): "canonical_full_base_finalists_agree",
}


STRATEGY_PARAM_PROFILES: dict[str, dict] = {
    "canonical_min_base_finalist_best": {
        "label": "Canonical Min Base-Finalist-Best",
        "strategy_id": "breakout",
        "filter_profile": "min",
        "parameter_policy": "base-finalist-best",
        "parameter_set_identity": "min_rolling",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {"enabled": False},
        "fitting_seed_protocol": "single_seed_42",
        "registered_scientific_fingerprint_sha256": (
            "783eb920f8dc95830d4cfb50f15f1e20e953816b6a79de0d180411e4c35b428d"
        ),
    },
    "canonical_full_base_finalist_best": {
        "label": "Canonical Full Base-Finalist-Best",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalist-best",
        "parameter_set_identity": "full_rolling",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {"enabled": False},
        "fitting_seed_protocol": "single_seed_42",
        "registered_scientific_fingerprint_sha256": (
            "77fa547f9b9ad35ff2efbcc6beece618449833ef324f0d6d3306230edc970ad9"
        ),
    },
    "canonical_full_base_finalists_agree": {
        "label": "Canonical Full Base-Finalists-Agree",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalists-agree",
        "parameter_set_identity": "full_rolling",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {"enabled": False},
        "fitting_seed_protocol": "single_seed_42",
        "registered_scientific_fingerprint_sha256": (
            "7f8d7c77c8624c1f071c98bf2df6be94141cf02cbcf3e5b7b0e3a8f78357d440"
        ),
    },
    "validation_budget_current": {
        "label": "Validation-Budget Refit",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalists-agree",
        "parameter_set_identity": "PARAM-P5",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {
            "enabled": True,
            "protocol": "validation_budget_refit_v1",
        },
        "fitting_seed_protocol": "single_seed_42",
        "artifact_slug": "param_p5_validation_budget_refit",
    },
    "validation_plateau_current": {
        "label": "Adaptive Plateau Validation Refit",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalists-agree",
        "parameter_set_identity": "PARAM-P9",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {
            "enabled": True,
            "protocol": "validation_plateau_refit_v2",
        },
        "fitting_seed_protocol": "single_seed_42",
        "artifact_slug": "param_p9_validation_plateau_refit",
    },
    "production_multiseed_current": {
        "label": "Production-Style Fixed Multi-Seed",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalists-agree",
        "parameter_set_identity": "PARAM-P6",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {
            "enabled": False,
        },
        "fitting_seed_protocol": "research_fixed_multiseed_v1",
        "artifact_slug": "param_p6_production_multiseed",
        # PARAM-P6 / SR-C85 is the first Strategy Parameter profile whose
        # scientific semantics are permanently registered.  This digest is
        # the canonical resolved scientific contract (including Research data
        # generation identity), not a display/runtime fingerprint.  Any future
        # result-affecting change must allocate a new scientific identity
        # rather than mutate PARAM-P6/SR-C85 in place.
        "registered_scientific_fingerprint_sha256": (
            "b6162e8debd663a54fc3a56d2f738e7a217e11128a1fa2148dac01006bae976b"
        ),
    },
    "production_multiseed_8x300_compare": {
        "label": "Fixed Multi-Seed 8x300",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalists-agree",
        "parameter_set_identity": "PARAM-P7",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {"enabled": False},
        "fitting_seed_protocol": "research_fixed_multiseed_8x300_v1",
        "artifact_slug": "param_p7_multiseed_8x300",
        "registered_scientific_fingerprint_sha256": (
            "2fbe14e7d72d5e71936cfa152934cdb027f414910a2b737309c0488ec9b1da3c"
        ),
    },
    "production_multiseed_4x1000_compare": {
        "label": "Fixed Multi-Seed 4x1000",
        "strategy_id": "breakout",
        "filter_profile": "full",
        "parameter_policy": "base-finalists-agree",
        "parameter_set_identity": "PARAM-P8",
        "dataset": "full",
        "schedule_start_date": "2021-01-01",
        "coverage_end_date": RESEARCH_MARKET_DATA_CUTOFF,
        "validation": {"enabled": False},
        "fitting_seed_protocol": "research_fixed_multiseed_4x1000_v1",
        "artifact_slug": "param_p8_multiseed_4x1000",
        "registered_scientific_fingerprint_sha256": (
            "dbcbaf0653895bbfdf93462cb57845a565497d6fa4cef2baf5e7782ce0c26697"
        ),
    },
}


__all__ = [
    "ACTIVE_STRATEGY_PARAM_PROFILE",
    "STRATEGY_PARAM_CANONICAL_PROFILE_BY_FAMILY_POLICY",
    "STRATEGY_PARAM_FITTING_SEED_PROTOCOLS",
    "STRATEGY_PARAM_PROFILES",
    "STRATEGY_PARAM_VALIDATION_PROTOCOLS",
]
