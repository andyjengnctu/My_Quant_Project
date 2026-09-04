"""Project Research declarative configuration.

The interactive menu selects only a work type. Active model/provider selection
and artifact-preparation settings are declared here; typed resolution belongs to
``core.research_policy``.
"""

ACTIVE_MODEL_ID = "breakout_quality"

# Canonical single-seed identity for ordinary Research workflows.
RESEARCH_SINGLE_SEED = 42

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
    "MODEL_RESEARCH_PROVIDERS",
    "RESEARCH_ARTIFACT_PREPARATION",
]
