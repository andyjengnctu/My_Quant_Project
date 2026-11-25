"""Causal full-history indicators appended to canonical sequence inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stock_ema_contract(spans: tuple[int, ...]) -> dict:
    if not spans or any(type(span) is not int or span < 1 for span in spans):
        raise ValueError("EMA spans must be nonempty positive integers")
    if tuple(sorted(set(spans))) != spans:
        raise ValueError("EMA spans must be unique and increasing")
    return {
        "contract_id": "full_history_stock_close_ema_v1",
        "spans": list(spans),
        "feature_columns": [f"Stock_Close_EMA_{span}" for span in spans],
        "history": "complete_canonical_stock_history_before_window_slicing",
        "initialization": "first_finite_positive_close",
        "alpha": "2/(span+1)",
        "adjust": False,
        "invalid_close": "skip_without_updating",
        "normalization": "ema_divided_by_sample_information_close_minus_one",
        "dtype": "float64_recurrence_float32_normalized_input",
    }


def full_history_stock_emas(closes: np.ndarray, spans: tuple[int, ...]) -> np.ndarray:
    """Compute each date's EMA from the full causal prefix, never a sample window."""
    stock_ema_contract(spans)
    values = np.asarray(closes, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("EMA closes must be one-dimensional")
    observed = pd.Series(values).where(np.isfinite(values) & (values > 0.0))
    return np.column_stack([
        observed.ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()
        for span in spans
    ])
