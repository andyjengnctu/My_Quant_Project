"""Formal Fixed Min PIT teacher construction and verification.

The core policy owns immutable parameters. This service binds those parameters
to the canonical MinTrade PIT label source without reversing package layers.
"""
from __future__ import annotations

from core.fixed_min_policy import (
    FIXED_MIN_ID, FIXED_MIN_FIRST_EFFECTIVE_DATE,
    fixed_min_params, fixed_min_signature, fixed_min_scientific_sha256,
)
from core.portfolio_param_runtime import build_portfolio_params_signature

def fixed_min_source(seed: int):
    """The same immutable constant params as the strategy (no Optimizer/P16)."""
    if type(seed) is not int or seed <= 0:
        raise ValueError("Fixed Min requires a positive FIT seed")
    import pandas as pd
    from filters.breakout_quality.daily_trade40_target import ExecutableMinPeriod
    from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITSource
    period = ExecutableMinPeriod(
        start=pd.Timestamp(FIXED_MIN_FIRST_EFFECTIVE_DATE), end=None,
        params=fixed_min_params(), source_id=FIXED_MIN_ID,
        params_signature=fixed_min_signature(),
    )
    return FormalMinTrade40PITSource(
        seed=seed, param_sha256=fixed_min_scientific_sha256(),
        schedule=(period,), publication_dates=(FIXED_MIN_FIRST_EFFECTIVE_DATE,),
        source_id=FIXED_MIN_ID, oos_start_date=None,
    )


def verify_fixed_min_source(source) -> bool:
    """Strictly prove one identical constant, without accepting fake Optimizer lineage."""
    if source.source_id != FIXED_MIN_ID:
        return False
    if (source.param_sha256 != fixed_min_scientific_sha256()
            or source.publication_dates != (FIXED_MIN_FIRST_EFFECTIVE_DATE,)
            or source.oos_start_date is not None
            or len(source.schedule) != 1):
        return False
    period = source.schedule[0]
    return bool(
        period.source_id == FIXED_MIN_ID
        and str(period.start.date()) == FIXED_MIN_FIRST_EFFECTIVE_DATE
        and period.params_signature == fixed_min_signature()
        and build_portfolio_params_signature(period.params) == fixed_min_signature()
    )
