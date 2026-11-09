"""P19 Daily Min 4D hypothetical NetPnL, no Breakout or P20 Full filters.

Uses the existing single-next-session ATR limit and Min broker/exit engine.
A P19 source must use its own historical/current PARAM-P19 schedule. The
P20 Full source is not accepted and neither P18 nor P20 labels can be reused.
"""
from __future__ import annotations

from filters.breakout_quality.daily_full_p20_target import compute_daily_full_p20_batch
from core.file_integrity import canonical_json_sha256, compute_file_sha256
from pathlib import Path


_P19_DISABLED_FILTERS = (
    "use_bb", "use_kc", "use_vol", "use_breakout_ema_filter",
    "use_breakout_return_filter", "use_breakout_false_filter",
    "use_breakout_quality_filter", "use_breakout_quality_ranking",
    "use_breakout_buy",
)


def verify_daily_min_p19_params(params) -> None:
    """One owner for P19's all-off entry filter invariant."""
    if any(bool(getattr(params, flag, False)) for flag in _P19_DISABLED_FILTERS):
        raise ValueError("Daily Min/P19 may not activate Breakout or Full filters")


def daily_min_p19_execution_sha256() -> str:
    """Full P19 runtime closure for FIT labels and hindsight Oracle cache."""
    root = Path(__file__).resolve().parents[2]
    return canonical_json_sha256({
        "p19_execution": compute_file_sha256(Path(__file__)),
        "next_session_min_engine": compute_file_sha256(
            root / "filters/breakout_quality/daily_full_p20_target.py"),
        "daily_entry_gates": compute_file_sha256(
            root / "core/daily_common_entry_filters.py"),
        "broker_min": compute_file_sha256(
            root / "filters/breakout_quality/trade_path_label.py"),
    })


def compute_daily_min_p19_batch(frame, positions, *, ticker, schedule,
                                exchange_calendar, research_cutoff):
    """Real P19 ATR buy tolerance, one exchange day expiry, full Min exits.

    P19 uses the P20 engine's common no-signal broker primitive with *all*
    P20-only gates disabled; the 4D source controls the signed ATR limit.
    """
    if not schedule:
        raise ValueError("Daily Min/P19 requires a signed 4D Min schedule")
    for period in schedule:
        verify_daily_min_p19_params(period.params)
    return compute_daily_full_p20_batch(
        frame, positions, ticker=ticker, schedule=schedule,
        exchange_calendar=exchange_calendar, research_cutoff=research_cutoff,
    )


__all__ = ["compute_daily_min_p19_batch", "verify_daily_min_p19_params",
           "daily_min_p19_execution_sha256"]
