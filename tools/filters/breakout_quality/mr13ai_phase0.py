"""Thin CLI wrapper for the disposable MR-13AI Phase-0 diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from services.breakout_quality.mr13ai_phase0 import main


if __name__ == "__main__":
    raise SystemExit(main())
