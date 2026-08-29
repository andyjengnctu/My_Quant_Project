"""Acyclic gateway from generic consumers to continuous-ranker runtime recipes.

The scientific/profile configuration owner resolves experiment identity.  The pure runtime
module owns capability contracts and builders.  Keeping this forwarding gateway separate
prevents ``config.breakout_quality`` and ``config.breakout_quality_runtime`` from importing
each other while preserving one canonical recipe resolver.
"""

from config.breakout_quality import get_continuous_ranker_execution_recipe

__all__ = ("get_continuous_ranker_execution_recipe",)
