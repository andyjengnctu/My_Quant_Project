"""Canonical economic teacher identity for Daily Full/P20 NetPnL.

Shared by the independent P20 FIT-source and PIT-materialization services.
Neither service imports the other to obtain this identity.
"""

from core.breakout_quality_runtime import DAILY_FULL_P20_SOURCE_ID as SOURCE_ID

__all__ = ["SOURCE_ID"]
