"""Run an isolated, one-off model research study without changing Research menus.

This command is an engineering entry into an existing registered experimental
producer. It does not authorize the study for formal PIT, Standard SOP or Trading.
"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.breakout_quality_registry import get_exploratory_model_research_study

# Stable work types; study identity, feature semantics and producer remain in
# the existing scientific registry and service, never in a standard Research menu.
ONE_OFF_ACTIONS = {
    "forward": "run_forward_menu",
    "rolling": "run_rolling_menu",
    "compare-forward": "run_forward_comparison",
    "compare-rolling": "run_rolling_comparison",
    "robust-forward": "run_forward_robustness",
    "robust-rolling": "run_rolling_robustness",
}


def execute(study_id: str, action: str) -> int:
    if action not in ONE_OFF_ACTIONS:
        raise ValueError(f"不支援的一次性研究工作: {action}")
    study = get_exploratory_model_research_study(study_id)
    producer = import_module(study.service_module)
    handler = getattr(producer, ONE_OFF_ACTIONS[action], None)
    if not callable(handler):
        raise RuntimeError(f"已註冊探索研究缺少 {action} Producer: {study.service_module}")
    return int(handler() or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一次性隔離模型研究；不變更標準 Research 選單、比較清單或報表"
    )
    parser.add_argument("study_id", help="已註冊探索研究 ID，例如 MR-13BY")
    parser.add_argument("action", choices=tuple(ONE_OFF_ACTIONS))
    args = parser.parse_args(argv)
    return execute(args.study_id, args.action)


if __name__ == "__main__":
    raise SystemExit(main())
