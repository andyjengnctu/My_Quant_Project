"""Shared Rolling model SCORE/Audit source resolution, independent of consumers.

The ranking-score store remains the scientific owner of both validators.  This
adapter only selects its complete-Audit contract or its independently verified
SCORE contract when Audit evidence is absent/stale.  In particular, an Audit
failure must not make a valid scored fold look like an untrained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_ranking_contract,
    load_selection_point_in_time_score_source_contract,
)


@dataclass(frozen=True)
class RollingModelSource:
    """Verified source plus optional complete evaluation; no REUSE policy here."""

    source_contract: Any | None
    complete_contract: Any | None
    audit_error: str | None
    score_error: str | None


def load_rolling_model_source(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    point_in_time_dir_override: Path | None,
    expected_seed: int | None = None,
) -> RollingModelSource:
    """Prefer fully verified Ranking/Audit; fail over only to verified SCORE.

    The score-only fallback is for *classification* of missing downstream Audit
    evidence, not permission to REUSE the Audit or a fitted checkpoint.
    """
    args = (
        str(project_root), str(filter_id), str(model_architecture),
        str(experiment_profile),
    )
    try:
        complete = load_selection_point_in_time_ranking_contract(
            *args, require_model_validation_pass=False,
            point_in_time_dir_override=point_in_time_dir_override,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        audit_error = f"{type(exc).__name__}: {exc}"
    else:
        return RollingModelSource(complete, complete, None, None)

    try:
        score = load_selection_point_in_time_score_source_contract(
            *args, point_in_time_dir_override=point_in_time_dir_override,
            expected_seed=expected_seed,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return RollingModelSource(None, None, audit_error, f"{type(exc).__name__}: {exc}")
    return RollingModelSource(score, None, audit_error, None)
