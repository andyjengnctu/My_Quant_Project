"""Immutable Continuous-DL Research menu work types.

These items describe the historical seven standard user operations only.
Model/scientific identities, experimental services and capabilities must not
control menu inventory, wording, numbering or interaction flow.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResearchWorkType:
    choice: str
    action: str
    label: str


MODEL_RESEARCH_WORK_TYPES: tuple[ModelResearchWorkType, ...] = (
    ModelResearchWorkType("1", "forward", "Forward OOS 模型訓練"),
    ModelResearchWorkType("2", "rolling", "Rolling OOS 模型訓練"),
    ModelResearchWorkType("3", "compare", "Forward OOS 模型比較"),
    ModelResearchWorkType("4", "rolling_compare", "Rolling OOS 模型比較"),
    ModelResearchWorkType("5", "robustness", "Forward OOS Robustness 模型測試"),
    ModelResearchWorkType("6", "rolling_robustness", "Rolling OOS Robustness 模型測試"),
    ModelResearchWorkType("7", "timing", "Timing Mode｜Rolling 訓練前後比較  [工程]"),
)


def model_research_menu_items() -> tuple[tuple[str, str, str], ...]:
    return tuple((work.choice, work.action, work.label) for work in MODEL_RESEARCH_WORK_TYPES)
