"""Pure projection of canonical Portfolio results into persisted Oracle evidence.

Accounting and period statistics remain owned by Portfolio/reporting. This
module only validates and selects the existing strategy observation fields.
"""
from __future__ import annotations

import math
from typing import Any

from filters.breakout_quality.strategy_compare_reporting import (
    _scenario_summary, _yearly_frame, _to_json_native,
)


class OracleObservationError(ValueError):
    """A Portfolio result cannot supply complete finite Oracle evidence."""


def oracle_observation(replay: dict[str, Any], *, target: str) -> dict[str, Any]:
    summary = _scenario_summary(replay)
    required = ("total_return_pct", "max_drawdown_pct", "return_over_max_drawdown", "expected_value_r", "trade_count", "avg_exposure_pct", "win_rate_pct")
    if any(not math.isfinite(float(summary[name])) for name in (*required, "annual_return_pct")):
        raise OracleObservationError(f"Oracle {target}缺少有效portfolio performance metrics")
    if not 0.0 <= float(summary["win_rate_pct"]) <= 100.0:
        raise OracleObservationError(f"Oracle {target} has invalid completed-trade win rate")
    yearly = _yearly_frame(replay["profile"], "strategy")
    return {
        "target": target,
        "total_return_pct": float(summary["total_return_pct"]),
        "annualized_return_pct": float(summary["annual_return_pct"]),
        "mdd_pct": float(summary["max_drawdown_pct"]),
        "romd": float(summary["return_over_max_drawdown"]),
        "ev_a_r": float(summary["expected_value_r"]),
        "win_rate_pct": float(summary["win_rate_pct"]),
        "trades": int(summary["trade_count"]),
        "exposure_pct": float(summary["avg_exposure_pct"]),
        "yearly": _to_json_native(yearly.to_dict("records")),
    }
