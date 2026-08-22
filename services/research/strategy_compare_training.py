"""Shared Strategy Compare model-training command contract.

Single-seed and multi-seed Strategy Compare differ only in the supplied seed/output
namespace.  Canonical trainer CLI construction lives here so scientific/execution
options cannot drift between the two orchestration modes.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

from core.console_report import COMPACT_CONSOLE_ENV
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
)

STRATEGY_COMPARE_TRAINER_LOG_TAIL_CHARS = 5000
STRATEGY_COMPARE_TRAINER_ENV: Mapping[str, str] = {
    COMPACT_CONSOLE_ENV: "0",
    "PYTHONUNBUFFERED": "1",
    "BREAKOUT_QUALITY_EPOCH_PROGRESS_MARKERS": "1",
}


def build_strategy_compare_trainer_command(
    *,
    source: Any,
    workflow: Any,
    seed: int,
    score_start_date: str | None = None,
    score_end_date: str | None = None,
    point_in_time_dir_override: str | Path | None = None,
    checkpoint_cache_root: str | Path | None = None,
    model_output_dir: str | Path | None = None,
    research_output_dir: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[str], list[str], dict[str, object]]:
    """Build one canonical trainer command for any Strategy Compare seed."""

    score_source = str(source.score_source)
    if score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        args = [
            "--filter-id", str(source.filter_id),
            "--model-architecture", str(source.model_architecture),
            "--experiment-profile", str(source.experiment_profile),
            "--seed", str(int(seed)),
        ]
        if model_output_dir is not None:
            args.extend(["--model-output-dir", str(Path(model_output_dir).resolve())])
        if research_output_dir is not None:
            args.extend(["--research-output-dir", str(Path(research_output_dir).resolve())])
        return (
            [
                sys.executable,
                "-m",
                "tools.filters.breakout_quality.train_continuous_ranker",
                *args,
            ],
            args,
            {"score_source": score_source},
        )

    if score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        raise ValueError(f"不支援的Strategy Compare score source: {score_source!r}")

    fold_months = int(
        source.point_in_time_fold_months
        if source.point_in_time_fold_months is not None
        else workflow.point_in_time_fold_months
    )
    resolved_start = str(
        score_start_date
        if score_start_date not in (None, "")
        else source.point_in_time_score_start_date
        if source.point_in_time_score_start_date not in (None, "")
        else workflow.point_in_time_score_start_date
    )
    raw_end = (
        score_end_date
        if score_end_date not in (None, "")
        else source.point_in_time_score_end_date
        if source.point_in_time_score_end_date not in (None, "")
        else workflow.point_in_time_score_end_date
    )
    resolved_end = None if raw_end in (None, "") else str(raw_end)

    args = [
        "--filter-id", str(source.filter_id),
        "--model-architecture", str(source.model_architecture),
        "--experiment-profile", str(source.experiment_profile),
        "--score-start-date", resolved_start,
        "--fold-months", str(fold_months),
        "--inner-validation-months", str(int(workflow.point_in_time_inner_validation_months)),
        "--seed", str(int(seed)),
        "--resume" if resume else "--no-resume",
    ]
    fold_anchor = (
        None
        if source.point_in_time_fold_anchor_date in (None, "")
        else str(source.point_in_time_fold_anchor_date)
    )
    if fold_anchor is not None:
        args.extend(["--fold-anchor-date", fold_anchor])
    if bool(source.point_in_time_single_score_block):
        args.append("--single-score-block")
    if resolved_end is not None:
        args.extend(["--score-end-date", resolved_end])
    if point_in_time_dir_override is not None:
        args.extend([
            "--point-in-time-dir-override",
            str(Path(point_in_time_dir_override).resolve()),
        ])
    if checkpoint_cache_root is not None:
        args.extend([
            "--checkpoint-cache-root",
            str(Path(checkpoint_cache_root).resolve()),
        ])
    return (
        [
            sys.executable,
            "-m",
            "tools.filters.breakout_quality.build_point_in_time_scores",
            *args,
        ],
        args,
        {
            "score_source": score_source,
            "score_start_date": resolved_start,
            "score_end_date": resolved_end,
            "fold_months": fold_months,
        },
    )


__all__ = [
    "STRATEGY_COMPARE_TRAINER_ENV",
    "STRATEGY_COMPARE_TRAINER_LOG_TAIL_CHARS",
    "build_strategy_compare_trainer_command",
]
