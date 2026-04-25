"""Reusable per-process feature cache for optimizer and future strategy pipelines."""

from collections import OrderedDict
from typing import Callable, Hashable

import numpy as np


DEFAULT_FEATURE_BANK_MAX_ITEMS = 512


class FeatureBank:
    """Small FIFO cache for deterministic array features.

    The cache is intentionally per process and bounded.  It is suitable for
    optimizer workers that repeatedly evaluate different parameter sets over
    the same immutable market data.  Trading rules never live here; callers
    cache only pure feature arrays such as ATR or rolling means.
    """

    def __init__(self, max_items: int = DEFAULT_FEATURE_BANK_MAX_ITEMS):
        try:
            resolved_max_items = int(max_items)
        except (TypeError, ValueError):
            resolved_max_items = DEFAULT_FEATURE_BANK_MAX_ITEMS
        self.max_items = max(0, resolved_max_items)
        self._cache: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Hashable, builder: Callable[[], np.ndarray]) -> np.ndarray:
        if self.max_items <= 0:
            self.misses += 1
            return builder()

        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.misses += 1
        value = np.asarray(builder())
        self._cache[key] = value
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return value

    def snapshot_stats(self) -> dict:
        return {
            "max_items": int(self.max_items),
            "size": int(len(self._cache)),
            "hits": int(self.hits),
            "misses": int(self.misses),
        }


def coerce_feature_bank(feature_bank):
    if isinstance(feature_bank, FeatureBank):
        return feature_bank
    return None
