"""Stable Research model work-type interface, independent of scientific identity.

Every configured model scope uses the exact same seven selectable workflows.
The registry selects a scientific producer, never the UI's inventory/order.
Missing implementations are integration defects; they cannot hide menu items.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class ModelResearchWorkType:
    choice: str
    action: str
    label: str
    producer_method: str


MODEL_RESEARCH_WORK_TYPES: tuple[ModelResearchWorkType, ...] = (
    ModelResearchWorkType("1", "forward", "Forward OOS 模型訓練", "run_forward_menu"),
    ModelResearchWorkType("2", "rolling", "Rolling OOS 模型訓練", "run_rolling_menu"),
    ModelResearchWorkType("3", "compare", "Forward OOS 模型比較", "run_forward_comparison"),
    ModelResearchWorkType("4", "rolling_compare", "Rolling OOS 模型比較", "run_rolling_comparison"),
    ModelResearchWorkType("5", "robustness", "Forward OOS Robustness 模型測試", "run_forward_robustness"),
    ModelResearchWorkType("6", "rolling_robustness", "Rolling OOS Robustness 模型測試", "run_rolling_robustness"),
    ModelResearchWorkType("7", "timing", "Timing Mode｜Rolling 訓練前後比較  [工程]", "run_timing_menu"),
)


def model_research_menu_items() -> tuple[tuple[str, str, str], ...]:
    return tuple((work.choice, work.action, work.label) for work in MODEL_RESEARCH_WORK_TYPES)


def validate_research_producer(module: ModuleType) -> None:
    """Fail closed on broken implementation, without altering any menu entries."""
    missing = tuple(
        item.producer_method for item in MODEL_RESEARCH_WORK_TYPES
        if not callable(getattr(module, item.producer_method, None))
    )
    if missing:
        raise RuntimeError(
            f"Research workflow producer {module.__name__} 缺少固定共用工作介面: "
            + ", ".join(missing)
        )


def model_research_producer_method(action: str) -> str:
    for item in MODEL_RESEARCH_WORK_TYPES:
        if item.action == action:
            return item.producer_method
    raise ValueError(f"未知Research模型工作類型: {action!r}")
