"""Shared local-min dependency-stat payload primitives."""

from __future__ import annotations


def empty_local_min_dependency_stats() -> dict:
    return {
        "dependency_signal_total": 0,
        "dependency_signal_evaluated": 0,
        "dependency_portfolio_total": 0,
        "dependency_portfolio_evaluated": 0,
        "dependency_mixed_total": 0,
        "dependency_mixed_evaluated": 0,
        "dependency_unknown_total": 0,
        "dependency_unknown_evaluated": 0,
        "signal_reuse_candidate_total": 0,
        "signal_reuse_candidate_evaluated": 0,
        "signal_recompute_required_total": 0,
        "signal_recompute_required_evaluated": 0,
        "dependency_field_total_counts": {},
        "dependency_field_evaluated_counts": {},
    }


__all__ = ["empty_local_min_dependency_stats"]
