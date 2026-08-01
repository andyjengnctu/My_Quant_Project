"""Backward-compatible adapter to the unified breakout-quality entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint


def main(argv=None) -> int:
    from apps.breakout_quality import main as breakout_quality_main

    forwarded = list(sys.argv[1:] if argv is None else argv)
    return int(
        breakout_quality_main(
            ["apps/breakout_quality_strategy_compare.py", "strategy-compare", *forwarded]
        )
        or 0
    )


if __name__ == "__main__":
    run_cli_entrypoint(main)
