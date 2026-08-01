"""Backward-compatible CLI adapter for breakout-quality strategy comparison."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.breakout_quality import main as breakout_quality_main  # noqa: E402


def main(argv=None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return int(breakout_quality_main([str(Path(__file__)), "strategy-compare", *forwarded]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
