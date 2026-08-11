"""Compatibility alias for canonical optimizer walk-forward primitives."""

import sys
from services.optimizer import walk_forward as _impl

sys.modules[__name__] = _impl
