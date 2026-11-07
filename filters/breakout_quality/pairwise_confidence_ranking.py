"""Reusable same-date probabilistic pairwise ranking capability.

Each stock emits two *latent comparison coordinates*, not a gain estimate, a
Positive probability or a calibrated per-stock probability.  For stock i,j:

    logit P(i outranks j) = (u_i - u_j) + u_i*v_j - v_i*u_j

The skew-symmetric interaction allows context-dependent pair reliability and
non-transitive comparisons.  A sigmoid of one scalar score difference would
preserve every original ranking and is therefore deliberately not used.

Training uses strictly same-date, non-tied T40 pair outcomes with unweighted
Bernoulli log loss. Inference computes each name's expected fraction of wins
against the entire *decision-time score-eligible* same-date stock universe.
There is no fixed Top-N, confidence threshold or manually tuned blending weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validated_dates(dates, count: int) -> np.ndarray:
    values = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize()
    if len(values) != count or values.isna().any():
        raise ValueError("pairwise probability dates must align with model rows")
    return values.to_numpy(dtype="datetime64[ns]")


def _validate_embedding(embedding, *, count: int | None = None) -> np.ndarray:
    z = np.asarray(embedding)
    if z.ndim != 2 or z.shape[1] != 2 or (count is not None and len(z) != count):
        raise ValueError("pairwise probability embedding must have shape [N,2]")
    if not np.isfinite(z).all():
        raise ValueError("pairwise probability embedding must be finite")
    return z.astype(np.float64, copy=False)


def expected_pairwise_win_fraction(embedding, dates) -> np.ndarray:
    """Expected fraction of same-date comparisons won; exact 0.5 for singleton.

    This score is within [0,1] but is NOT P(T40>0), nor a calibrated absolute
    chance of a profitable trade.  Its benchmark universe is a date-specific
    roster decided before future T40 is observed.
    """
    z = _validate_embedding(embedding)
    days = _validated_dates(dates, len(z))
    scores = np.empty(len(z), dtype=np.float32)
    for day in pd.unique(days):
        pos = np.flatnonzero(days == day)
        n = len(pos)
        if n == 1:
            scores[pos] = 0.5
            continue
        u = z[pos, 0]
        v = z[pos, 1]
        pair_logit = (u[:, None] - u[None, :]) + (
            u[:, None] * v[None, :] - v[:, None] * u[None, :]
        )
        # Numerically stable logistic, without SciPy dependency.
        probability = np.empty_like(pair_logit)
        positive = pair_logit >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-pair_logit[positive]))
        exponent = np.exp(pair_logit[~positive])
        probability[~positive] = exponent / (1.0 + exponent)
        # Every diagonal equals 0.5 by antisymmetry; exclude self comparison.
        scores[pos] = ((probability.sum(axis=1) - 0.5) / float(n - 1)).astype(np.float32)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise FloatingPointError("expected pairwise win fraction is invalid")
    return scores


def probabilistic_pairwise_loss(torch, embedding, target, dates):
    """Mean proper Bernoulli loss over all same-date strict T40 pairs.

    No pair subsampling or target-gap weighting: changing event prevalence by
    arbitrary pair weights would change the meaning of the predicted pair
    probabilities. Tied / missing maturity targets are excluded, never class 0.
    """
    import torch.nn.functional as F

    if embedding.ndim != 2 or int(embedding.shape[1]) != 2:
        raise ValueError("pairwise probability trainer requires [N,2] outputs")
    t = target.float()
    if t.ndim != 1 or len(t) != int(embedding.shape[0]):
        raise ValueError("pairwise probability training target must be [N]")
    days = _validated_dates(dates, len(t))
    loss_sum = None
    comparable_count = 0
    for day in pd.unique(days):
        pos_np = np.flatnonzero(days == day)
        if len(pos_np) < 2:
            continue
        pos = torch.as_tensor(pos_np, device=embedding.device, dtype=torch.long)
        z = embedding.index_select(0, pos).float()
        y = t.index_select(0, pos)
        i, j = torch.triu_indices(len(pos_np), len(pos_np), offset=1, device=embedding.device)
        comparable = torch.isfinite(y[i]) & torch.isfinite(y[j]) & (y[i] != y[j])
        if not bool(comparable.any().item()):
            continue
        i = i[comparable]
        j = j[comparable]
        u, v = z[:, 0], z[:, 1]
        pair_logit = (u[i] - u[j]) + (u[i] * v[j] - v[i] * u[j])
        label = (y[i] > y[j]).to(dtype=pair_logit.dtype)
        one_loss = F.binary_cross_entropy_with_logits(pair_logit, label, reduction="sum")
        loss_sum = one_loss if loss_sum is None else loss_sum + one_loss
        comparable_count += int(i.numel())
    if not comparable_count:
        return None, 0
    return loss_sum / comparable_count, comparable_count


__all__ = ["expected_pairwise_win_fraction", "probabilistic_pairwise_loss"]
