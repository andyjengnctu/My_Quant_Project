"""P18 supervision: genuine Daily close-limit entries and P18 Min exits.

The numerical path and PIT/maturity rules are the existing canonical Daily
Min capabilities. Only the independently verified P18 teacher is admitted.
"""
from __future__ import annotations

from pathlib import Path

from core.file_integrity import canonical_json_sha256, compute_file_sha256
from core.breakout_quality_runtime import DAILY_MIN_P18_SOURCE_ID

SOURCE_ID = DAILY_MIN_P18_SOURCE_ID
_DISABLED_FILTERS = (
    "use_bb", "use_kc", "use_vol", "use_breakout_ema_filter",
    "use_breakout_return_filter", "use_breakout_false_filter",
    "use_breakout_quality_filter", "use_breakout_quality_ranking", "use_breakout_buy",
)


def verify_daily_min_p18_params(params):
    if any(bool(getattr(params, flag, False)) for flag in _DISABLED_FILTERS):
        raise ValueError("Daily Min/P18 requires scoreless Min parameters with all entry filters off")


def daily_min_p18_execution_sha256():
    root = Path(__file__).resolve().parents[2]
    return canonical_json_sha256({
        relative: compute_file_sha256(root / relative) for relative in (
            "filters/breakout_quality/daily_min_p18_target.py",
            "filters/breakout_quality/true_daily_mintrade_target.py",
            "services/breakout_quality/true_daily_mintrade_pit_source.py",
            "services/breakout_quality/daily_min_p18_pit_source.py",
            "filters/breakout_quality/trade_path_label.py",
        )
    })


