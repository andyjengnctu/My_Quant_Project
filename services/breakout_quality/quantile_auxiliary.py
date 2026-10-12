"""Date-balanced conditional quantile supervision for continuous-ranker auxiliary heads.

This primitive operates only on existing matured training targets. It neither
selects epochs nor transforms prediction/strategy outputs.
"""

from __future__ import annotations

import numpy as np


def date_balanced_pinball_loss(torch, predicted, observed, dates, *, quantile_levels):
    """Equal-mean over valid dates of within-date mean and equal quantile mean."""
    if predicted.ndim != 2 or observed.ndim != 1:
        raise ValueError("quantile predictions必須[N,Q]、adverse target必須[N]")
    levels = tuple(float(q) for q in quantile_levels)
    if (
        predicted.shape[0] != observed.shape[0]
        or predicted.shape[1] != len(levels)
        or len(dates) != int(observed.shape[0])
    ):
        raise ValueError("quantile prediction/target/date membership維度不一致")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(observed).all()):
        raise FloatingPointError("quantile inputs包含non-finite數值")
    if bool((observed < 0).any()):
        raise ValueError("Adverse R target不可為負")
    if bool((predicted[:, 1:] < predicted[:, :-1]).any()):
        raise ValueError("模型量化頭發生quantile crossing")
    differences = observed.float().unsqueeze(1) - predicted.float()
    quantiles = torch.tensor(levels, dtype=differences.dtype, device=differences.device).view(1, -1)
    row_losses = torch.maximum(quantiles * differences, (quantiles - 1.0) * differences).mean(dim=1)
    # Date grouping follows the same full-date batch semantics as the pairwise
    # supervision; disjoint intra-day chunks must never be treated as dates.
    dates_array = np.asarray(dates)
    daily_losses = []
    for date in np.unique(dates_array):
        row_positions = np.flatnonzero(dates_array == date)
        positions = torch.as_tensor(row_positions, dtype=torch.long, device=predicted.device)
        daily_losses.append(row_losses.index_select(0, positions).mean())
    if not daily_losses:
        raise ValueError("quantile auxiliary training batch不可為空")
    return torch.stack(daily_losses).mean()



def summarize_date_balanced_quantiles(predicted, observed, dates, *, quantile_levels):
    """Read-only OOS-ready quantile diagnostics without altering Standard SOP.

    Report equal-date empirical coverage and Pinball, not a globally stock-day
    weighted estimate.  Call only on fixed, previously trained predictions;
    never feed the returned OOS statistics into fitting or epoch selection.
    """
    quantiles = np.asarray(tuple(quantile_levels), dtype=np.float64)
    prediction = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    groups = np.asarray(dates)
    if (
        prediction.ndim != 2
        or target.ndim != 1
        or prediction.shape[0] != target.shape[0]
        or prediction.shape[1] != len(quantiles)
        or len(groups) != len(target)
        or not len(target)
    ):
        raise ValueError("quantile OOS prediction/target/date shape不一致或為空")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("quantile OOS assessment必須使用已成熟finite targets")
    if (target < 0).any() or (prediction < 0).any() or (np.diff(prediction, axis=1) < 0).any():
        raise ValueError("quantile OOS predictions/Adverse R必須非負且不交叉")
    if (
        quantiles.ndim != 1
        or not len(quantiles)
        or not np.isfinite(quantiles).all()
        or (quantiles <= 0).any() or (quantiles >= 1).any()
        or (np.diff(quantiles) <= 0).any()
    ):
        raise ValueError("quantile OOS quantile levels必須在(0,1)且嚴格遞增")
    error = target[:, None] - prediction
    pinball = np.maximum(quantiles[None, :] * error, (quantiles[None, :] - 1) * error)
    coverage = (target[:, None] <= prediction).astype(np.float64)
    daily_pinball = []
    daily_coverage = []
    for day in np.unique(groups):
        mask = groups == day
        daily_pinball.append(pinball[mask].mean(axis=0))
        daily_coverage.append(coverage[mask].mean(axis=0))
    mean_pinball_by_quantile = np.mean(daily_pinball, axis=0)
    mean_coverage_by_quantile = np.mean(daily_coverage, axis=0)
    return {
        "date_count": int(len(daily_pinball)),
        "row_count": int(len(target)),
        "mean_date_pinball": float(mean_pinball_by_quantile.mean()),
        "mean_date_pinball_by_quantile": tuple(float(x) for x in mean_pinball_by_quantile),
        "mean_date_empirical_coverage_by_quantile": tuple(float(x) for x in mean_coverage_by_quantile),
    }


__all__ = ["date_balanced_pinball_loss", "summarize_date_balanced_quantiles"]
