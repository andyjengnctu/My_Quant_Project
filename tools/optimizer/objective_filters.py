"""Compatibility alias for the canonical optimizer service module."""

import sys
from services.optimizer import objective_filters as _impl

sys.modules[__name__] = _impl
