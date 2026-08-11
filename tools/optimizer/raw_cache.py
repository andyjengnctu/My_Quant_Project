"""Compatibility alias for canonical optimizer raw-data cache primitives."""

import sys
from services.optimizer import raw_cache as _impl

sys.modules[__name__] = _impl
