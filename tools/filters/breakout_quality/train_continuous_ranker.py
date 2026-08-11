"""Compatibility wrapper for the canonical Breakout Quality service."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from services.breakout_quality.train_continuous_ranker import main
    raise SystemExit(main())

import sys
from services.breakout_quality import train_continuous_ranker as _impl

sys.modules[__name__] = _impl
