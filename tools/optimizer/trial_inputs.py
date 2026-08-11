"""Compatibility alias for canonical optimizer trial-input preparation."""

import sys
from services.optimizer import trial_inputs as _impl

sys.modules[__name__] = _impl
