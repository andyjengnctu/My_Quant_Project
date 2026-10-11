"""Trading services."""

# AI: Bind the optional breakout-quality Core port at the Trading service boundary.
from filters.breakout_quality import runtime as _breakout_quality_runtime_adapter
