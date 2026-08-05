"""Legacy compatibility alias for the canonical strategy comparison engine."""

from __future__ import annotations

import sys

from filters.breakout_quality import strategy_compare_engine as _engine

if __name__ == "__main__":
    raise SystemExit(_engine.main())

sys.modules[__name__] = _engine
