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
# Round 3 activates the production-style fixed multi-seed Research producer after
# fitting-seed bundles and robustness replication bundles were made orthogonal.
ACTIVE_STRATEGY_PARAM_PROFILE = "production_multiseed_current"


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
}


# Scientific producer profiles.  Min/Full are filter-profile dimensions, not menu
# work types.  Validation is opt-in per profile and may be disabled for another
# strategy-parameter target without changing the Research menu or Optimizer engine.
STRATEGY_PARAM_PROFILES: dict[str, dict] = {
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
}


__all__ = [
    "ACTIVE_STRATEGY_PARAM_PROFILE",
    "STRATEGY_PARAM_FITTING_SEED_PROTOCOLS",
    "STRATEGY_PARAM_PROFILES",
    "STRATEGY_PARAM_VALIDATION_PROTOCOLS",
]
