"""Compatibility alias for canonical Optimizer benchmark helpers."""

from __future__ import annotations

import sys
from services.optimizer import benchmark as _impl

sys.modules[__name__] = _impl
