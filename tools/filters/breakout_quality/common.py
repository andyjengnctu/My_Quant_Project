"""Legacy compatibility alias for canonical breakout-quality workflow I/O helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.breakout_quality import workflow_io as _impl

if __name__ == "__main__":
    raise SystemExit("workflow_io is a library module")

sys.modules[__name__] = _impl
