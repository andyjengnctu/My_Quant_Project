"""Historical compatibility alias for optimizer-owned strategy parameter training.

Current parameter production lives in :mod:`services.optimizer.strategy_param_training`.
Importers of this historical module receive the optimizer module object itself, so legacy
monkeypatch/test seams keep working without creating a second implementation or state.
"""
from __future__ import annotations

import sys
from services.optimizer import strategy_param_training as _optimizer_impl

if __name__ == "__main__":
    raise SystemExit(_optimizer_impl.main())

sys.modules[__name__] = _optimizer_impl
