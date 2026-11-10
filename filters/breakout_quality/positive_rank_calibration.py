"""Read-only codec for retired single-ranker Positive calibration evidence.

Current Positive predictions require independently fitted binary classifiers.
Historical signed payload validation remains available for attribution; no
calibration fitter or runtime score conversion may revive the rejected design.
The small no-op adapters preserve current uncalibrated checkpoint envelopes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.positive_netpnl_contract import (
    POSITIVE_RANK_CALIBRATION,
)
from filters.breakout_quality.rank_calibration import (
    MonotonicCurve,
)

PAYLOAD_KEY = "score_calibration"


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def calibration_enabled(profile) -> bool:
    value = getattr(profile, "score_calibration", None)
    if value is None:
        return False
    raise ValueError("Retired single-ranker Positive calibration is read-only historical evidence")


def validate_score_calibration(payload: Mapping[str, Any]) -> MonotonicCurve:
    if not isinstance(payload, Mapping):
        raise ValueError("Missing Positive score calibration")
    data = dict(payload)
    signature = data.pop("sha256", None)
    if signature != _digest(data) or data.get("contract") != POSITIVE_RANK_CALIBRATION:
        raise ValueError("Stale or corrupt Positive score calibration")
    x = np.asarray(data.get("x"), dtype=np.float64)
    y = np.asarray(data.get("y"), dtype=np.float64)
    if (x.ndim != 1 or not len(x) or y.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or (np.diff(x) <= 0).any() or (np.diff(y) < 0).any()
            or ((x < 0) | (x > 1)).any() or ((y < 0) | (y > 1)).any()
            or data.get("rank_preserved") is not True
            or int(data.get("validation_count", 0)) < 2
            or int(data.get("selected_epoch", 0)) < 1):
        raise ValueError("Invalid monotone Positive score calibration")
    dates = {key: pd.Timestamp(data[key]) for key in (
        "information_cutoff", "validation_start", "validation_end",
        "validation_maturity_end", "inner_train_maturity_end")}
    if (any(pd.isna(value) for value in dates.values())
            or dates["validation_start"] > dates["validation_end"]
            or dates["validation_maturity_end"] < dates["validation_end"]
            or dates["validation_maturity_end"] > dates["information_cutoff"]
            or dates["inner_train_maturity_end"] >= dates["validation_start"]
            or data.get("refit_policy") != "freeze_selected_epoch_validation_curve_for_final_refit"):
        raise ValueError("Positive calibration evidence has an illegal cutoff or refit policy")
    return MonotonicCurve(x, y)


def checkpoint_calibration_payload(model, profile) -> dict[str, Any]:
    calibration_enabled(profile)
    return {}


def restore_score_calibration(model, profile, checkpoint) -> None:
    calibration_enabled(profile)
    if checkpoint.get(PAYLOAD_KEY) is not None:
        raise ValueError("Retired single-ranker Positive calibration cannot supply current FIT scores")


def calibrated_score_columns(model, profile, scores) -> dict[str, np.ndarray]:
    calibration_enabled(profile)
    return {}
