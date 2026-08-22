"""Compatibility wrapper for the formal Breakout Quality research application."""

from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] if "filters/breakout_quality" in str(Path(__file__)) else Path(__file__).resolve().parents[2]))
    from services.research.breakout_quality_application import main
    raise SystemExit(main())

import sys
from services.research import breakout_quality_application as _impl

sys.modules[__name__] = _impl
