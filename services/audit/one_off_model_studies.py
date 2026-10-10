"""One-off exploratory model execution inside the existing Audit navigation.

This does not register a formal/read-only Audit handler or change standard Model
Research menus. Scientific identity and producer ownership remain in the model
research registry and its original service module.
"""
from __future__ import annotations

from importlib import import_module

from core.breakout_quality_registry import (
    ExploratoryModelResearchStudy,
    get_exploratory_model_research_study,
    list_exploratory_model_research_studies,
)
from core.model_research_menu_contract import MODEL_RESEARCH_WORK_TYPES

# One-off CLI action names are retained for compatibility with the earlier
# isolated CLI. The source of the user's labels/numbering is the existing
# immutable Model Research work-type contract, not another model-specific UI.
ONE_OFF_ACTIONS: dict[str, str] = {
    "forward": "run_forward_menu",
    "rolling": "run_rolling_menu",
    "compare-forward": "run_forward_comparison",
    "compare-rolling": "run_rolling_comparison",
    "robust-forward": "run_forward_robustness",
    "robust-rolling": "run_rolling_robustness",
}

_WORK_TO_ACTION: dict[str, str] = {
    "forward": "forward",
    "rolling": "rolling",
    "compare": "compare-forward",
    "rolling_compare": "compare-rolling",
    "robustness": "robust-forward",
    "rolling_robustness": "robust-rolling",
}


def get_one_off_model_studies(module_id: str) -> tuple[ExploratoryModelResearchStudy, ...]:
    """Discover isolated studies without granting them formal Audit authorization."""
    if str(module_id) != "breakout_quality":
        return ()
    return list_exploratory_model_research_studies()


def one_off_model_study_menu_items() -> tuple[tuple[str, str, str], ...]:
    """Use the existing six model work labels without changing the standard menu."""
    return tuple(
        (work.choice, _WORK_TO_ACTION[work.action], work.label)
        for work in MODEL_RESEARCH_WORK_TYPES
        if work.action in _WORK_TO_ACTION
    )


def execute_one_off_model_study(study_id: str, action: str) -> int:
    """Delegate only to the registered isolated producer, never to formal AO."""
    if action not in ONE_OFF_ACTIONS:
        raise ValueError(f"不支援的一次性研究工作: {action}")
    study = get_exploratory_model_research_study(study_id)
    producer = import_module(study.service_module)
    handler = getattr(producer, ONE_OFF_ACTIONS[action], None)
    if not callable(handler):
        raise RuntimeError(f"已註冊探索研究缺少 {action} Producer: {study.service_module}")
    return int(handler() or 0)


__all__ = [
    "ONE_OFF_ACTIONS",
    "execute_one_off_model_study",
    "get_one_off_model_studies",
    "one_off_model_study_menu_items",
]
