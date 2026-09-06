"""Project Research declarative configuration.

The interactive menu selects only a work type. Active model/provider selection
and artifact-preparation settings are declared here; typed resolution belongs to
``core.research_policy``.
"""

from config.market_data import ACTIVE_RESEARCH_DATA_GENERATION, RESEARCH_DATA_GENERATIONS

ACTIVE_MODEL_ID = "breakout_quality"

# Canonical single-seed identity for ordinary Research workflows.
RESEARCH_SINGLE_SEED = 42

# Backward-compatible active Research cutoff alias. Market-data generation
# identity/lifecycle is owned by config.market_data; V2 must not become active
# until its bootstrap common-complete snapshot is frozen.
RESEARCH_MARKET_DATA_CUTOFF = RESEARCH_DATA_GENERATIONS[ACTIVE_RESEARCH_DATA_GENERATION]["cutoff"]

# Model-specific application providers. Add future models here without adding a
# new executable under apps/.
MODEL_RESEARCH_PROVIDERS: dict[str, dict[str, str]] = {
    "breakout_quality": {
        "module": "services.research.breakout_quality_application",
        "menu_handler": "run_model_training_menu",
        "status_handler": "show_model_status",
        "cli_handler": "main",
        "strategy_artifact_handler": "prepare_strategy_compare_artifacts",
    },
}

# Research-wide deterministic artifact orchestration policy. Domain configs may
# add domain-specific reuse knobs, but missing/stale/corrupt/partial handling
# must not fork.
RESEARCH_ARTIFACT_PREPARATION = {
    "auto_prepare": True,
    "reuse_ready_artifacts": True,
    "rebuild_stale_artifacts": True,
    "resume_partial_artifacts": True,
    "require_single_confirmation": True,
}

__all__ = [
    "ACTIVE_MODEL_ID",
    "RESEARCH_SINGLE_SEED",
    "RESEARCH_MARKET_DATA_CUTOFF",
    "MODEL_RESEARCH_PROVIDERS",
    "RESEARCH_ARTIFACT_PREPARATION",
]
