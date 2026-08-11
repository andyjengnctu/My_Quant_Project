"""Compatibility wrapper for the canonical Breakout Quality service."""

import sys
from services.breakout_quality import train_daily_ranker as _impl

sys.modules[__name__] = _impl
