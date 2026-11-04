"""Full-cycle NetPnL Positive supervision and score-admission semantics.

Prediction-time eligibility uses the calibrated column of the same scalar
ranker (or perfect Oracle score output), never an out-of-band future outcome.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

POSITIVE_PROBABILITY_THRESHOLD = 0.5
POSITIVE_P_CLASS_COLUMN = "positive_p_positive"
POSITIVE_RANK_CALIBRATION = "positive_rank_isotonic_validation_v1"
POSITIVE_SCORE_CONTRACT = "single_ranker_calibrated_positive_v2"


def requires_positive_score(options: Mapping | None) -> bool:
    """Discover the inline output capability; reject retired dual-DL wiring."""
    options = options or {}
    if options.get("positive_dl_id"):
        raise ValueError("Retired independent Positive DL source; rebuild the strategy contract")
    allowed = {"positive_score_column", "positive_prediction_contract", "positive_probability_threshold"}
    supplied = {key for key in options if str(key).startswith("positive_")}
    if supplied - allowed:
        raise ValueError("Unknown or retired Positive score options")
    column = options.get("positive_score_column")
    if not supplied:
        return False
    if (column != POSITIVE_P_CLASS_COLUMN
            or options.get("positive_prediction_contract") != POSITIVE_SCORE_CONTRACT):
        raise ValueError("Positive ranker score-consumption contract missing or stale")
    return True


def positive_labels(raw_net_pnl) -> np.ndarray:
    """1 for strictly positive matured NetPnL, 0 for zero/loss, NaN for censored."""
    raw = np.asarray(raw_net_pnl, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("Positive NetPnL target requires a one-dimensional raw label")
    labels = np.full(raw.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(raw)
    labels[valid] = (raw[valid] > 0.0).astype(np.float32)
    return labels


def positive_score_admitted(row: Mapping) -> bool:
    """Fail closed on missing/misaligned columns; retain the same model rank."""
    options = row.get("breakout_quality_ranking_options") or {}
    if not isinstance(options, Mapping):
        raise ValueError("Ranking options must be a mapping")
    if not requires_positive_score(options):
        return True
    rank = row.get("breakout_quality_rank")
    if not isinstance(rank, Mapping):
        return False
    if not rank.get("available") or not rank.get("positive_available"):
        return False
    primary_day = str(rank.get("score_date") or "")[:10]
    positive_day = str(rank.get("positive_score_date") or "")[:10]
    if not primary_day or not positive_day or primary_day != positive_day:
        return False
    threshold = float(options.get("positive_probability_threshold", POSITIVE_PROBABILITY_THRESHOLD))
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Positive probability threshold must be in (0,1)")
    try:
        probability = float(rank.get("positive_probability"))
    except (ValueError, TypeError):
        return False
    return math.isfinite(probability) and 0.0 <= probability <= 1.0 and probability > threshold


def admit_positive_score_rows(rows):
    """Apply the same binary score decision before Exact/Direct cash reservation."""
    return [row for row in list(rows or []) if positive_score_admitted(row)]
