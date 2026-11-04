"""Frozen PIT-legal Positive calibration of a single continuous NetPnL ranker.

The scalar ranking loss/output is unchanged.  A monotone lookup is fitted only
on the selected epoch's inner-validation predictions, then carried unchanged
with the final-refit checkpoint. It is not another neural model or head. The
validation curve is transferred to the refit model; it is not a claim of
out-of-sample calibration quality, which must be measured separately.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.positive_netpnl_contract import (
    POSITIVE_P_CLASS_COLUMN, POSITIVE_RANK_CALIBRATION, positive_labels,
)
from filters.breakout_quality.rank_calibration import (
    MonotonicCurve, fit_weighted_increasing_isotonic,
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
    if value != POSITIVE_RANK_CALIBRATION:
        raise ValueError(f"Unsupported score calibration: {value!r}")
    return True


def fit_score_calibration(
    scores, raw_target, group_table, *, train_ids, validation_ids,
    information_cutoff: str, selected_epoch: int, profile,
) -> dict[str, Any]:
    """Validate chronology and fit the existing canonical isotonic primitive."""
    if not calibration_enabled(profile):
        raise ValueError("The fitting profile does not declare Positive calibration")
    train = np.asarray(train_ids, dtype=np.int64)
    valid = np.asarray(validation_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    raw = np.asarray(raw_target)
    if (train.ndim != 1 or valid.ndim != 1 or values.shape != valid.shape
            or len(train) == 0 or len(valid) < 2
            or len(np.unique(train)) != len(train) or len(np.unique(valid)) != len(valid)
            or np.intersect1d(train, valid).size
            or min(train.min(), valid.min()) < 0
            or max(train.max(), valid.max()) >= len(group_table)
            or raw.shape != (len(group_table),)):
        raise ValueError("Invalid or overlapping Positive calibration FIT/validation groups")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Positive calibration requires finite scalar rank probabilities")
    cutoff = pd.Timestamp(information_cutoff).normalize()
    if pd.isna(cutoff) or int(selected_epoch) < 1:
        raise ValueError("Positive calibration requires a legal cutoff and selected epoch")
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    maturity = pd.to_datetime(group_table["label_eval_end_date"], errors="coerce").dt.normalize()
    fit_ids = np.concatenate((train, valid))
    if (dates.iloc[fit_ids].isna().any() or maturity.iloc[fit_ids].isna().any()
            or (maturity.iloc[fit_ids] > cutoff).any()
            or (dates.iloc[fit_ids] > cutoff).any()
            or (maturity.iloc[fit_ids].to_numpy() < dates.iloc[fit_ids].to_numpy()).any()
            or dates.iloc[train].max() >= dates.iloc[valid].min()
            or maturity.iloc[train].max() >= dates.iloc[valid].min()):
        raise ValueError("Positive calibration violates PIT maturity/purged validation chronology")
    if not np.isfinite(raw[fit_ids]).all():
        raise ValueError("Censored NetPnL cannot enter Positive FIT or calibration")
    labels = positive_labels(raw[valid]).astype(np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("Censored NetPnL cannot enter Positive calibration")
    # A degenerate ranker has one score: retain a constant empirical rate,
    # rather than inventing a threshold or crashing an otherwise legal FIT.
    if np.unique(values).size == 1:
        curve = MonotonicCurve(np.array([values[0]]), np.array([labels.mean()]))
    else:
        curve, _ = fit_weighted_increasing_isotonic(values, labels)
    keys = group_table.iloc[valid][["ticker", "date", "label_eval_end_date"]].copy()
    keys["rank_score"] = values
    keys["raw_net_pnl"] = raw[valid].astype(np.float64)
    evidence_hash = hashlib.sha256(keys.to_csv(index=False).encode("utf-8")).hexdigest()
    payload = {
        "contract": POSITIVE_RANK_CALIBRATION,
        "experiment_settings": profile.as_manifest_payload(),
        "x": curve.x.tolist(), "y": curve.y.tolist(),
        "selected_epoch": int(selected_epoch),
        "information_cutoff": str(cutoff.date()),
        "validation_start": str(dates.iloc[valid].min().date()),
        "validation_end": str(dates.iloc[valid].max().date()),
        "validation_maturity_end": str(maturity.iloc[valid].max().date()),
        "inner_train_maturity_end": str(maturity.iloc[train].max().date()),
        "validation_count": int(len(valid)),
        "validation_evidence_sha256": evidence_hash,
        "validation_group_ids_sha256": hashlib.sha256(valid.astype("<i8").tobytes()).hexdigest(),
        "rank_preserved": True,
        "refit_policy": "freeze_selected_epoch_validation_curve_for_final_refit",
    }
    payload["sha256"] = _digest(payload)
    validate_score_calibration(payload)
    return payload


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


def attach_score_calibration(model, profile, epoch_selection, *, selected_epoch: int) -> None:
    if not calibration_enabled(profile):
        return
    payload = dict((epoch_selection or {}).get(PAYLOAD_KEY) or {})
    validate_score_calibration(payload)
    if payload.get("experiment_settings") != profile.as_manifest_payload():
        raise ValueError("Calibration belongs to a different model/Target/FIT contract")
    if int(payload["selected_epoch"]) != int(selected_epoch):
        raise ValueError("Calibration belongs to a different selected epoch")
    model._score_calibration = payload


def checkpoint_calibration_payload(model, profile) -> dict[str, Any]:
    if not calibration_enabled(profile):
        return {}
    payload = dict(getattr(model, "_score_calibration", {}) or {})
    validate_score_calibration(payload)
    return {PAYLOAD_KEY: payload}


def restore_score_calibration(model, profile, checkpoint) -> None:
    if not calibration_enabled(profile):
        return
    attach_score_calibration(
        model, profile, checkpoint,
        selected_epoch=int(checkpoint.get("selected_epoch", 0)),
    )


def calibrated_score_columns(model, profile, scores) -> dict[str, np.ndarray]:
    if not calibration_enabled(profile):
        return {}
    curve = validate_score_calibration(getattr(model, "_score_calibration", None))
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid primary rank scores for Positive calibration")
    probability = curve.predict(scores).astype(np.float32)
    return {POSITIVE_P_CLASS_COLUMN: probability}
