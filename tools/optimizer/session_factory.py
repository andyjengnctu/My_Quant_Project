"""Compatibility alias for the canonical optimizer service module."""

import sys
from services.optimizer import session_factory as _impl

sys.modules[__name__] = _impl
