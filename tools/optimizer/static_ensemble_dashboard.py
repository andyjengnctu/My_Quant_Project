"""Compatibility alias for canonical Optimizer dashboard helpers."""

from __future__ import annotations

import sys
from services.optimizer import static_ensemble_dashboard as _impl

sys.modules[__name__] = _impl
