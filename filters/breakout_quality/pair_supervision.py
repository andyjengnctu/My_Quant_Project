"""PIT-only supervision membership over the unchanged Daily Universal input bank.

The canonical bank owns immutable sanitized OHLCV source arrays and their date
grids. This adapter reads those existing source views, calls the canonical signal
builder, and joins to its original stock-day rows. It does not consume the
post-model diagnostic event cache, future labels or portfolio state.
"""
from __future__ import annotations

from hashlib import sha256
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import TrainingPairPartition
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.features import build_candidate_event_positions

_MEMBERSHIP_CACHE = WeakKeyDictionary()


def resolve_training_pair_membership(partition, feature_bank, group_table):
    """Return an aligned, read-only boolean mask; never drop or reorder input rows."""
    if not isinstance(partition, TrainingPairPartition):
        raise ValueError("Training pair membership needs a canonical partition contract")
    if group_table is None or len(group_table) != len(feature_bank):
        raise ValueError("Training pair membership needs the complete canonical stock-day table")
    required = ("_frame_arrays", "_frame_dates", "_ticker_ids", "_source_positions")
    if any(not hasattr(feature_bank, name) for name in required):
        raise ValueError("Training pair membership requires canonical daily OHLCV source views")
    keys = group_table[["ticker", "date"]]
    if keys.duplicated().any():
        raise ValueError("Training pair membership has duplicate stock-day keys")
    high_lens = DEFAULT_LABEL_POLICY.high_lens()
    digest = sha256(pd.util.hash_pandas_object(keys, index=False).values.tobytes()).hexdigest()
    cache_key = (partition.source, high_lens, digest)
    cached = _MEMBERSHIP_CACHE.get(feature_bank, {}).get(cache_key)
    if cached is not None:
        return cached
    membership = np.zeros(len(group_table), dtype=bool)
    ticker_ids = np.asarray(feature_bank._ticker_ids)
    positions = np.asarray(feature_bank._source_positions)
    if len(ticker_ids) != len(keys) or len(positions) != len(keys):
        raise ValueError("Daily source views and membership rows disagree")
    order = np.argsort(ticker_ids, kind="stable")
    counts = np.bincount(ticker_ids, minlength=len(feature_bank._frame_arrays))
    offsets = np.r_[0, np.cumsum(counts)]
    for ticker_id, (array, dates) in enumerate(zip(
            feature_bank._frame_arrays, feature_bank._frame_dates, strict=True)):
        ids = order[offsets[ticker_id]:offsets[ticker_id + 1]]
        if not len(ids):
            continue
        grid = pd.DatetimeIndex(dates)
        aligned = pd.DatetimeIndex(pd.to_datetime(group_table.iloc[ids]["date"]))
        if not aligned.equals(grid.take(positions[ids])):
            raise ValueError("Training membership dates differ from canonical source positions")
        frame = pd.DataFrame(array, columns=["Open", "High", "Low", "Close", "Volume"], index=grid)
        candidate_dates = {event["date"] for event in build_candidate_event_positions(frame, high_lens)}
        membership[ids] = aligned.isin(candidate_dates)
    membership.setflags(write=False)
    _MEMBERSHIP_CACHE.setdefault(feature_bank, {})[cache_key] = membership
    return membership
