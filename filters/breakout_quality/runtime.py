"""Runtime bridge used by signal generation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import DEFAULT_FILTER_ID
from filters.breakout_quality.score_store import build_pass_condition_from_score_table


def resolve_project_root_from_runtime() -> str:
    return str(Path(__file__).resolve().parents[2])


def build_breakout_quality_filter_pass_condition(
    df: pd.DataFrame,
    *,
    ticker: str,
    high_len: int,
    filter_id: str = DEFAULT_FILTER_ID,
    project_root: str | None = None,
    score_path: str | None = None,
) -> np.ndarray:
    root = resolve_project_root_from_runtime() if project_root is None else str(project_root)
    env_score_path = os.environ.get("V16_BREAKOUT_QUALITY_SCORE_PATH")
    resolved_score_path = score_path if score_path not in {None, ""} else env_score_path
    return build_pass_condition_from_score_table(
        df,
        ticker=ticker,
        high_len=int(high_len),
        project_root=root,
        filter_id=str(filter_id),
        score_path=resolved_score_path,
    )


__all__ = ["build_breakout_quality_filter_pass_condition", "resolve_project_root_from_runtime"]
