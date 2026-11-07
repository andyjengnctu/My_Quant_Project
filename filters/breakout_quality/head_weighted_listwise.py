"""T40-agnostic head-priority probabilistic listwise ranking primitives.

Use full *mature* same-date training lists, and a single learned scalar utility
(pass minus reject logit). At each distinct target level, predict a choice from
the remaining list. Equal-truth items share uniform winner mass and are removed
together: no accidental ticker-order supervision inside equal T40 values.

The successive choice cross-entropies are weighted by the *change* in the
canonical logarithmic rank discount d(r)=1/log2(r+1). Marginal discounts
telescope; their total mass stays bounded even for very large stock universes.
No fixed K, reward-R regression, sign classifier or manual mixing coefficient.
The resulting bounded scalar inference score is ranking utility, NOT a
calibrated P(top 1) or a per-stock P(positive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_dates(dates, n: int) -> np.ndarray:
    days = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize()
    if len(days) != n or days.isna().any():
        raise ValueError("listwise same-date membership must align with rows")
    return days.to_numpy(dtype="datetime64[ns]")


def _discount(rank: int) -> float:
    if rank < 1:
        raise ValueError("rank discount expects positive 1-based rank")
    return float(1.0 / np.log2(rank + 1.0))


def head_weighted_sequential_choice_loss(torch, margin, target, dates):
    """Return date-equal loss and contributing date count.

    Each tie group uses uniform target mass for its members and never learns an
    arbitrary within-group permutation. The final group has no lower group to
    distinguish, so it contributes no training signal.
    """
    if margin.ndim != 1 or target.ndim != 1 or len(margin) != len(target):
        raise ValueError("head-weighted listwise expects aligned scalar utilities/targets")
    days = _aligned_dates(dates, len(target))
    losses = []
    for day in pd.unique(days):
        positions = np.flatnonzero(days == day)
        if len(positions) < 2:
            continue
        idx = torch.as_tensor(positions, dtype=torch.long, device=margin.device)
        scores = margin.index_select(0, idx).float()
        values = target.index_select(0, idx).float()
        valid = torch.isfinite(values) & torch.isfinite(scores)
        if not bool(valid.any().item()):
            continue
        scores, values = scores[valid], values[valid]
        if len(values) < 2:
            continue
        order = torch.argsort(values.detach(), descending=True, stable=True)
        scores = scores.index_select(0, order)
        values = values.index_select(0, order).detach()
        # logcumsumexp gives exact suffix softmax denominators in O(N) after sort.
        suffix = torch.flip(torch.logcumsumexp(torch.flip(scores, dims=(0,)), dim=0), dims=(0,))
        distinct, counts = torch.unique_consecutive(values, return_counts=True)
        if len(distinct) < 2:
            continue
        starts = torch.cumsum(counts, dim=0) - counts
        group_losses = []
        group_weights = []
        for start, count in zip(starts[:-1].tolist(), counts[:-1].tolist()):
            end = start + count
            # Discount mass for positions [start+1, end] with a single tie group.
            weight = _discount(start + 1) - _discount(end + 1)
            if weight <= 0.0:
                raise ValueError("head-weighted listwise weights must decrease")
            # Expected CE when a uniformly selected tied winner is chosen next.
            group_losses.append(suffix[start] - scores[start:end].mean())
            group_weights.append(weight)
        weights = torch.as_tensor(group_weights, device=margin.device, dtype=scores.dtype)
        day_loss = torch.sum(torch.stack(group_losses) * weights) / weights.sum()
        losses.append(day_loss)
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)


def mean_daily_head_discounted_gain(group_dates, target_percentiles, predicted_scores):
    """K-free validation gain: earlier predicted positions have larger influence.

    This is an epoch-selection diagnostic, not a modification of the published
    common SOP. True target is the mature same-date T40 percentile in [0,1].
    """
    y = np.asarray(target_percentiles, dtype=np.float64)
    s = np.asarray(predicted_scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("head-discounted gain expects aligned 1D scores and targets")
    days = _aligned_dates(group_dates, len(y))
    qualities = []
    for day in pd.unique(days):
        pos = np.flatnonzero(days == day)
        valid = np.isfinite(y[pos]) & np.isfinite(s[pos])
        if valid.sum() < 2:
            continue
        yt, st = y[pos][valid], s[pos][valid]
        if np.unique(yt).size < 2:
            continue
        n = len(yt)
        ranks = np.arange(1, n + 2, dtype=np.float64)
        discount = 1.0 / np.log2(ranks + 1.0)
        marginal_weights = discount[:-1] - discount[1:]
        chosen = yt[np.argsort(-st, kind="mergesort")]
        ideal = np.sort(yt)[::-1]
        best = np.dot(marginal_weights, ideal)
        worst = np.dot(marginal_weights, ideal[::-1])
        if best > worst:
            qualities.append(float((np.dot(marginal_weights, chosen) - worst) / (best - worst)))
    if not qualities:
        return None
    return float(np.mean(qualities))


__all__ = ["head_weighted_sequential_choice_loss", "mean_daily_head_discounted_gain"]
