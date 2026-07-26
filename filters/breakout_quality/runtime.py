"""Runtime bridge used by signal generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import DEFAULT_FILTER_ID
from filters.breakout_quality.score_store import (
    build_pass_condition_from_score_table,
    lookup_breakout_quality_candidate_score,
)


def resolve_project_root_from_runtime() -> str:
    return str(Path(__file__).resolve().parents[2])


def build_breakout_quality_filter_pass_condition(
    df: pd.DataFrame,
    *,
    ticker: str,
    high_len: int,
    score_threshold: float,
    candidate_condition: np.ndarray,
    filter_id: str = DEFAULT_FILTER_ID,
    project_root: str | None = None,
) -> np.ndarray:
    root = resolve_project_root_from_runtime() if project_root is None else str(project_root)
    return build_pass_condition_from_score_table(
        df,
        ticker=ticker,
        high_len=int(high_len),
        score_threshold=float(score_threshold),
        candidate_condition=candidate_condition,
        project_root=root,
        filter_id=str(filter_id),
    )


def resolve_breakout_quality_candidate_rank(
    *,
    ticker: str,
    signal_date,
    high_len: int,
    filter_id: str = DEFAULT_FILTER_ID,
    project_root: str | None = None,
) -> dict:
    root = resolve_project_root_from_runtime() if project_root is None else str(project_root)
    return lookup_breakout_quality_candidate_score(
        project_root=root,
        ticker=str(ticker),
        signal_date=signal_date,
        high_len=int(high_len),
        filter_id=str(filter_id),
    )


__all__ = [
    "build_breakout_quality_filter_pass_condition",
    "resolve_breakout_quality_candidate_rank",
    "resolve_project_root_from_runtime",
]
