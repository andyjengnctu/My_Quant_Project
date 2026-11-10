"""P18 source validation before canonical Daily Min PIT/maturity execution."""
from dataclasses import replace
from filters.breakout_quality.daily_min_p18_target import SOURCE_ID, verify_daily_min_p18_params
from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITBlocked


def materialize_daily_min_p18_pit_labels(source, *, expected_seed, **kwargs):
    """Validate P18 provenance before delegating the unchanged PIT numerical owner."""
    if (source.source_id != SOURCE_ID or not source.schedule
            or any(period.source_id != SOURCE_ID for period in source.schedule)):
        raise FormalMinTrade40PITBlocked("Daily Min/P18 teacher identity mismatch")
    try:
        for period in source.schedule:
            verify_daily_min_p18_params(period.params)
    except ValueError as exc:
        raise FormalMinTrade40PITBlocked(str(exc)) from exc
    # The generic maturity verifier accepts this formal schedule capability.
    # Its parameter SHA retains P18's distinct historical/current lineage.
    formal = replace(source, source_id="formal_full_roos_seed", schedule=tuple(
        replace(period, source_id="formal_full_roos_seed") for period in source.schedule))
    from services.breakout_quality.true_daily_mintrade_pit_source import materialize_true_daily_mintrade_pit_labels
    labels = materialize_true_daily_mintrade_pit_labels(formal, expected_seed=expected_seed, **kwargs)
    return replace(labels, preflight={**labels.preflight, "economic_teacher_source_id": SOURCE_ID})
