"""Hindsight-only, explicitly authorized positive-NetPnL entry gate.

The ordinary Research/Trading selector is unchanged outside this context.
The future result is never a PIT score or training feature.  Eligibility uses
canonical raw broker-ledger NetPnL, NOT its [0,1] daily percentile.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Mapping

from core.portfolio_candidate_admission import scoped_candidate_admission

POSITIVE_FULLCYCLE_NETPNL_GATE = 'fullcycle_reference_net_pnl_strictly_positive_v1'

_ACTIVE: ContextVar[Mapping[tuple[str, str], float] | None] = ContextVar(
    'oracle_positive_fullcycle_netpnl_gate', default=None,
)


def active_positive_oracle_truth() -> Mapping[tuple[str, str], float] | None:
    return _ACTIVE.get()


@contextmanager
def positive_oracle_entry_gate(
    raw_net_pnl: Mapping[tuple[str, str], float],
) -> Iterator[None]:
    if _ACTIVE.get() is not None:
        raise RuntimeError('Nested hindsight positive-NetPnL gate is forbidden')
    if not raw_net_pnl:
        raise ValueError('Positive-NetPnL gate requires a certified raw-label map')
    token = _ACTIVE.set(raw_net_pnl)
    try:
        with scoped_candidate_admission(lambda row: positive_gate_candidate_status(row)[0]):
            yield
    finally:
        _ACTIVE.reset(token)


def positive_gate_candidate_status(row: Mapping) -> tuple[bool, str]:
    truth = _ACTIVE.get()
    if truth is None:
        return True, 'not_active'
    rank = row.get('breakout_quality_rank') or {}
    if not isinstance(rank, Mapping):
        return False, 'missing_score'
    day = str(rank.get('score_date') or row.get('breakout_quality_score_date') or '')[:10]
    ticker = str(row.get('ticker') or '')
    # The candidate must have a valid canonical score at this same information date.
    if not bool(rank.get('available', False)):
        return False, 'missing_score'
    value = truth.get((ticker, day))
    if value is None:
        return False, 'null_target'
    return (True, 'positive') if value > 0.0 else (False, 'nonpositive_target')


def filter_positive_oracle_candidates(rows):
    if _ACTIVE.get() is None:
        return list(rows or [])
    return [row for row in list(rows or []) if positive_gate_candidate_status(row)[0]]
