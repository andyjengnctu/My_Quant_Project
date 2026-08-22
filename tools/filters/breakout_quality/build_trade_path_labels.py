"""Compatibility wrapper for the canonical Breakout Quality service."""

from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] if "filters/breakout_quality" in str(Path(__file__)) else Path(__file__).resolve().parents[2]))
    from services.breakout_quality.trade_path_label_builder import main
    raise SystemExit(main())

import sys
from services.breakout_quality import trade_path_label_builder as _impl

sys.modules[__name__] = _impl
