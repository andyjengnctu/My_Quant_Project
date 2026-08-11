"""Compatibility alias for the canonical optimizer service module."""

import sys
from services.optimizer import outer_rolling_oos as _impl

sys.modules[__name__] = _impl
