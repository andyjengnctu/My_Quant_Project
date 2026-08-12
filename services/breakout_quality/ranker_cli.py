"""Formal continuous-ranker CLI dispatcher.

The event trainer and daily-universal trainer intentionally do not import each
other.  This module owns profile-based dispatch so the service dependency graph
remains acyclic while both trainers share ``ranker_training``.
"""

from __future__ import annotations

from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from services.breakout_quality import train_continuous_ranker as event_ranker


def main(argv=None) -> int:
    args = event_ranker.parse_args(argv)
    event_ranker.validate_args(args)
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        from services.breakout_quality.train_daily_ranker import run as run_daily_ranker

        return int(run_daily_ranker(args))
    return int(event_ranker.run(args))


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
