"""Full-cycle NetPnL Positive supervision and score-admission semantics.

Prediction-time eligibility uses an independent PIT-matured binary-classifier
probability (or perfect Oracle output); the ranker is a separate DL.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

POSITIVE_PROBABILITY_THRESHOLD = 0.5
POSITIVE_P_CLASS_COLUMN = "positive_p_positive"
POSITIVE_RANK_CALIBRATION = "positive_rank_isotonic_validation_v1"
POSITIVE_SCORE_CONTRACT = "positive_netpnl_dual_output_v1"


def requires_positive_score(options: Mapping | None) -> bool:
    """The Positive admission stage must name its own PIT classifier DL.

    Oracle may provide a separate perfect score map; strategy execution may not
    silently fall back to a calibrated primary ranker or to future NetPnL.
    """
    options = options or {}
    supplied = {str(key) for key in options if str(key).startswith("positive_")}
    if not supplied:
        return False
    allowed = {
        "positive_dl_id", "positive_score_column", "positive_probability_threshold",
        "positive_filter_id", "positive_model_architecture", "positive_experiment_profile",
        "positive_score_source", "positive_score_path_override",
        "positive_score_manifest_path_override",
    }
    if supplied - allowed:
        raise ValueError("Unsupported Positive score options: " + repr(sorted(supplied - allowed)))
    if not str(options.get("positive_dl_id") or "").strip():
        raise ValueError("Positive admission requires a separately registered DL source")
    if options.get("positive_score_column") != POSITIVE_P_CLASS_COLUMN:
        raise ValueError("Positive binary classifier must supply the P(positive) score column")
    threshold = options.get("positive_probability_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Positive probability threshold must be finite in (0,1)")
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
    """Fail closed on missing/misaligned independent classifier scores."""
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
