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
ACTIVE_STRATEGY_PARAM_PROFILE = "validation_budget_current"


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
    },
}


__all__ = [
    "ACTIVE_STRATEGY_PARAM_PROFILE",
    "STRATEGY_PARAM_PROFILES",
    "STRATEGY_PARAM_VALIDATION_PROTOCOLS",
]
