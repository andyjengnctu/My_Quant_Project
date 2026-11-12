"""Payoff-sensitive binary learning; no probability threshold fitting.

The classifier estimates positive versus negative *payoff mass*. Its Bayes
decision is trade iff expected net payoff is positive. Only the supplied FIT
rows determine loss weights; inference consumes prices and two model logits.
"""
from __future__ import annotations

import numpy as np


def payoff_training_targets(raw_target, fit_ids):
    raw = np.asarray(raw_target, dtype=np.float64)
    ids = np.asarray(fit_ids, dtype=np.int64)
    if raw.ndim != 1 or ids.ndim != 1 or not ids.size:
        raise ValueError("Payoff training requires a nonempty explicit FIT cohort")
    if np.any(ids < 0) or np.any(ids >= len(raw)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid payoff FIT indices")
    values = raw[ids]
    if not np.isfinite(values).all():
        raise ValueError("Payoff FIT cannot contain unknown or immature outcomes")
    scale = np.mean(np.abs(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Payoff FIT requires nonzero realized outcomes")
    result = np.full((len(raw), 2), np.nan, dtype=np.float32)
    result[ids, 0] = values > 0
    result[ids, 1] = np.abs(values) / scale
    return result


def payoff_cross_entropy(torch, logits, target):
    if logits.ndim != 2 or logits.shape[1] != 2 or target.shape != logits.shape:
        raise ValueError("Payoff CE requires two logits and [label, FIT-scaled weight]")
    if not torch.isfinite(target).all() or not torch.isfinite(logits).all():
        raise ValueError("Payoff CE requires finite FIT targets and logits")
    labels, weights = target[:, 0], target[:, 1]
    if torch.any((labels != 0) & (labels != 1)) or torch.any(weights < 0):
        raise ValueError("Invalid payoff class or sample weight")
    if not torch.any(weights > 0):
        return None
    losses = torch.nn.functional.cross_entropy(logits.float(), labels.long(), reduction="none")
    return (losses * weights.float()).mean()


def payoff_validation_metrics(logits, raw_target):
    """Selection on the caller's legal validation cohort, never OOS statistics."""
    values = np.asarray(raw_target, dtype=np.float64)
    scores = np.asarray(logits, dtype=np.float64)
    if (scores.shape != (len(values), 2) or not len(values)
            or not np.isfinite(scores).all() or not np.isfinite(values).all()):
        raise ValueError("Invalid payoff validation cohort")
    weight_sum = np.abs(values).sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Payoff validation requires nonzero realized outcomes")
    labels = (values > 0).astype(np.int64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    weights = np.abs(values) / weight_sum
    return {
        "cross_entropy": float(-np.sum(weights * log_probs[np.arange(len(values)), labels])),
        "joint_event_brier": float(np.sum(weights * (np.exp(log_probs[:, 1]) - labels) ** 2)),
        "weighting": "absolute_net_payoff",
    }


def payoff_trade_decisions(logits):
    scores = np.asarray(logits)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("Payoff decision requires finite binary logits")
    # AI: Decode directly before probability rounding. Ties retain cash.
    return (scores[:, 1] > scores[:, 0]).astype(np.float32)
