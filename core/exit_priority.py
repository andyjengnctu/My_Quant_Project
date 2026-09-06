"""Canonical exit-priority semantics shared by backtest and live Trading."""
from __future__ import annotations


EXIT_SAME_BAR_PRIORITY_STOP_OVER_TP = "STOP_OVER_TP"


def resolve_stop_tp_hits(*, stop_hit: bool, tp_hit: bool) -> tuple[bool, bool]:
    """Apply the canonical same-bar priority: STOP dominates TP."""

    stop = bool(stop_hit)
    tp = bool(tp_hit)
    if stop:
        tp = False
    return stop, tp


__all__ = ["EXIT_SAME_BAR_PRIORITY_STOP_OVER_TP", "resolve_stop_tp_hits"]
