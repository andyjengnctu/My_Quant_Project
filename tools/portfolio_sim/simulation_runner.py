"""Compatibility alias for the canonical portfolio replay service."""

import sys
from services import portfolio_replay as _impl

sys.modules[__name__] = _impl
