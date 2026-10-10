"""Run an isolated, one-off model research study without changing Research menus.

This command is an engineering entry into an existing registered experimental
producer. It does not authorize the study for formal PIT, Standard SOP or Trading.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.audit.one_off_model_studies import (
    ONE_OFF_ACTIONS,
    execute_one_off_model_study,
)


def execute(study_id: str, action: str) -> int:
    """Backward-compatible CLI wrapper; the Audit menu uses the same service."""
    return execute_one_off_model_study(study_id, action)


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
