"""Project research entry configuration.

The interactive menu selects only a work type.  The active model and its provider
are selected here so model identities never become menu options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ACTIVE_MODEL_ID = "breakout_quality"

# Model-specific application providers.  Add future models here without adding a
# new executable under apps/.
MODEL_RESEARCH_PROVIDERS: dict[str, dict[str, str]] = {
    "breakout_quality": {
        "module": "tools.filters.breakout_quality.application",
        "menu_handler": "run_model_training_menu",
        "status_handler": "show_model_status",
        "cli_handler": "main",
        "strategy_prerequisite_handler": "prepare_strategy_compare_model_artifacts",
        "strategy_upstream_handler": "prepare_strategy_compare_model_upstream_artifacts",
    },
}


@dataclass(frozen=True)
class ModelResearchProvider:
    model_id: str
    module: str
    menu_handler: str
    status_handler: str
    cli_handler: str
    strategy_prerequisite_handler: str
    strategy_upstream_handler: str


def get_active_model_research_provider() -> ModelResearchProvider:
    model_id = str(ACTIVE_MODEL_ID).strip()
    if not model_id:
        raise ValueError("ACTIVE_MODEL_ID不可空白")
    raw = MODEL_RESEARCH_PROVIDERS.get(model_id)
    if not isinstance(raw, Mapping):
        raise ValueError(f"ACTIVE_MODEL_ID尚未登記research provider: {model_id}")
    values = {
        key: str(raw.get(key) or "").strip()
        for key in (
            "module",
            "menu_handler",
            "status_handler",
            "cli_handler",
            "strategy_prerequisite_handler",
            "strategy_upstream_handler",
        )
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"MODEL_RESEARCH_PROVIDERS[{model_id!r}]缺少欄位: {', '.join(missing)}"
        )
    return ModelResearchProvider(model_id=model_id, **values)


__all__ = [
    "ACTIVE_MODEL_ID",
    "MODEL_RESEARCH_PROVIDERS",
    "ModelResearchProvider",
    "get_active_model_research_provider",
]
