"""Full-cycle NetPnL Positive supervision and score-admission semantics.

Prediction-time eligibility uses only the auxiliary PIT/Oracle *score output*,
never a future trade outcome.  The CO ranking score is not modified.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

POSITIVE_PROBABILITY_THRESHOLD = 0.5
POSITIVE_P_CLASS_COLUMN = "positive_p_positive"


def positive_labels(raw_net_pnl) -> np.ndarray:
    """1 for strictly positive matured NetPnL, 0 for zero/loss, NaN for censored."""
    raw = np.asarray(raw_net_pnl, dtype=np.float32)
    if raw.ndim != 1:
        raise ValueError("Positive NetPnL target requires a one-dimensional raw label")
    labels = np.full(raw.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(raw)
    labels[valid] = (raw[valid] > 0.0).astype(np.float32)
    return labels


def positive_score_admitted(row: Mapping) -> bool:
    """Fail closed on missing/misaligned scores; rank remains CO's score."""
    options = row.get("breakout_quality_ranking_options") or {}
    if not isinstance(options, Mapping) or not options.get("positive_dl_id"):
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
