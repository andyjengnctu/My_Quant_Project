"""Shared rolling-optimizer policy helpers for breakout-quality strategy workflows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from core.raw_universe_contract import RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD
from core.walk_forward_policy import load_walk_forward_policy
from strategies.breakout.search_space import get_breakout_optimizer_required_min_rows


def build_rolling_base_policy(
    *,
    root: Path,
    baseline_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical rolling selection policy from a baseline schedule."""

    policy = load_walk_forward_policy(str(root))
    meta = dict(baseline_contract["meta"])
    first_oos = pd.Timestamp(meta["first_oos_date"])
    last_oos = pd.Timestamp(meta["last_oos_date"])
    train_months = int(meta["train_window_months"])
    training_start = first_oos - pd.DateOffset(months=train_months)
    policy.update(
        {
            "model_mode": "oos",
            "study_scope": "split",
            "adaptation_scope": "selection_score_ranking_rolling_validation",
            "evaluation_scope": "rolling_selection_diagnostic",
            "objective_mode": "split_train_romd",
            "selection_start_year": int(training_start.year),
            "train_start_year": int(training_start.year),
            "search_train_end_year": int((first_oos - pd.Timedelta(days=1)).year),
            "selection_start_date": training_start.strftime("%Y-%m-%d"),
            "train_start_date": training_start.strftime("%Y-%m-%d"),
            "search_train_end_date": (first_oos - pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            "oos_start_year": int(first_oos.year),
            "oos_end_year": int(last_oos.year),
            "oos_start_date": first_oos.strftime("%Y-%m-%d"),
            "oos_end_date": (last_oos + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"),
            "latest_data_date": (last_oos + pd.offsets.MonthEnd(1)).strftime(
                "%Y-%m-%d"
            ),
            "min_train_years": max(1, int(math.ceil(train_months / 12.0))),
            "train_window_months": train_months,
            RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: int(
                get_breakout_optimizer_required_min_rows()
            ),
        }
    )
    return policy


def build_outer_rolling_argv(
    *,
    args,
    baseline_contract: dict[str, Any],
) -> list[str]:
    """Build the canonical outer-rolling schedule arguments."""

    meta = dict(baseline_contract["meta"])
    return [
        "--outer-first-oos-date",
        str(meta["first_oos_date"]),
        "--outer-last-oos-date",
        str(meta["last_oos_date"]),
        "--outer-train-window-months",
        str(int(meta["train_window_months"])),
        "--outer-oos-months",
        str(int(meta["oos_horizon_months"])),
        "--trials",
        str(int(args.trials_per_fold)),
    ]


__all__ = ["build_outer_rolling_argv", "build_rolling_base_policy"]
