"""Compatibility wrapper for the canonical Breakout Quality service."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from services.breakout_quality.binary_point_in_time_scores import main
    raise SystemExit(main())

import sys
from services.breakout_quality import binary_point_in_time_scores as _impl

sys.modules[__name__] = _impl
