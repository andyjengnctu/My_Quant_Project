"""Compatibility alias for the canonical optimizer service module."""

import sys
from services.optimizer import profile as _impl

sys.modules[__name__] = _impl
