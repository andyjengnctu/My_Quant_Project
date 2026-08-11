"""Compatibility alias for canonical portfolio replay runtime helpers."""

import sys
from services import portfolio_replay_runtime as _impl

sys.modules[__name__] = _impl
