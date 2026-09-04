"""Acyclic gateway from generic consumers to continuous-ranker runtime recipes.

The scientific/profile registry owns experiment identity, ``config.breakout_quality`` owns
user selections, and ``core.breakout_quality_policy`` resolves them into runtime recipes.
This forwarding gateway preserves the historical import seam without duplicating resolver
logic or introducing a reverse runtime dependency.
"""

from core.breakout_quality_policy import (
    get_continuous_ranker_execution_recipe,
)

__all__ = ("get_continuous_ranker_execution_recipe",)
