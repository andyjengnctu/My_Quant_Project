"""Reusable admission capability for a lone binary PIT classifier.

Legacy dual-model Positive remains owned by positive_netpnl_contract.py: that
reviewed code and its Oracle certificate must not change when new admission
compositions are registered. The only new behavior is classifier-only admission
with unchanged Default candidate ordering; no probability-to-rank conversion.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from core.positive_netpnl_contract import (
    POSITIVE_P_CLASS_COLUMN,
    POSITIVE_PROBABILITY_THRESHOLD,
    POSITIVE_SCORE_CONTRACT,
    requires_positive_score as _requires_dual_positive_score,
    positive_score_admitted as _dual_positive_score_admitted,
)

PRIMARY_CLASSIFIER_SOURCE_MODE = "primary_classifier"


def uses_primary_positive_classifier(options: Mapping | None) -> bool:
    return str((options or {}).get("positive_source_mode") or "") == PRIMARY_CLASSIFIER_SOURCE_MODE


def requires_positive_score(options: Mapping | None) -> bool:
    """Declare and validate the unique Positive admission source topology."""
    options = dict(options or {})
    if not uses_primary_positive_classifier(options):
        return _requires_dual_positive_score(options)
    allowed = {
        "positive_source_mode", "positive_score_column", "positive_probability_threshold",
    }
    present = {str(k) for k in options if str(k).startswith("positive_")}
    if present != allowed or options.get("primary_score_column") != POSITIVE_P_CLASS_COLUMN:
        raise ValueError("Positive-only PIT admission requires one primary binary score column, no rank/secondary source")
    threshold = options.get("positive_probability_threshold")
    if (options.get("positive_score_column") != POSITIVE_P_CLASS_COLUMN
            or type(threshold) not in (int, float)
            or not math.isfinite(float(threshold)) or threshold != POSITIVE_PROBABILITY_THRESHOLD):
        raise ValueError("Positive-only classifier requires the formal P(positive)>0.5 output")
    if options.get("selection_only") is not True:
        raise ValueError("Positive-only classifier must never alter capital or rank allocation")
    return True


def positive_score_admitted(row: Mapping) -> bool:
    """For primary-only, gate on valid PIT probability; never use it for rank."""
    options = row.get("breakout_quality_ranking_options") or {}
    if not isinstance(options, Mapping):
        raise ValueError("Ranking options must be a mapping")
    if not uses_primary_positive_classifier(options):
        return _dual_positive_score_admitted(row)
    requires_positive_score(options)
    payload = row.get("breakout_quality_rank")
    if not isinstance(payload, Mapping) or not payload.get("available"):
        return False
    information_day = str(payload.get("score_date") or "")[:10]
    if not information_day:
        return False
    try:
        probability = float(payload.get("score"))
    except (ValueError, TypeError):
        return False
    return math.isfinite(probability) and 0.0 <= probability <= 1.0 and probability > POSITIVE_PROBABILITY_THRESHOLD


def admit_positive_score_rows(rows):
    return [row for row in list(rows or ()) if positive_score_admitted(row)]
