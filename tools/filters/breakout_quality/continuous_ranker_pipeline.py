"""Compatibility wrapper for the canonical Breakout Quality service."""

import sys
from services.breakout_quality import continuous_ranker_pipeline as _impl

sys.modules[__name__] = _impl
