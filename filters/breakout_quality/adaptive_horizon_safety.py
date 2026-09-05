"""Adaptive-horizon Safety supervision derived from canonical daily-universal paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)


class AdaptiveHorizonSafetyTargetProvider:
    """Lazy same-date 1..H Safety trajectory targets for training batches."""

    def __init__(
        self,
        *,
        feature_bank,
        group_dates,
        final_safety_target: np.ndarray,
        horizon_bars: int,
    ) -> None:
        if not hasattr(feature_bank, "future_adverse_to_best_peak_path"):
            raise ValueError(
                "adaptive-horizon Safety需要canonical daily feature provider的future path capability"
            )
        self.feature_bank = feature_bank
        self.group_dates = pd.Series(pd.to_datetime(group_dates, errors="raise")).dt.normalize()
        self.final_safety_target = np.asarray(final_safety_target, dtype=np.float32)
        self.horizon_bars = int(horizon_bars)
        if self.horizon_bars < 2:
            raise ValueError("adaptive-horizon Safety至少需要2個future horizons")
        if len(self.group_dates) != len(self.final_safety_target):
            raise ValueError("adaptive-horizon Safety dates/target長度不一致")
        # Training revisits the same complete-date batches every epoch.  Cache exact
        # id-order keys so raw 1..H future paths and same-date percentiles are built
        # once per stage rather than once per epoch.  This is fitting-scope local and
        # carries no information across Inner/Selection stages.
        self._target_cache: dict[bytes, np.ndarray] = {}

    def targets_for_ids(self, group_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            return np.empty((0, self.horizon_bars), dtype=np.float32)
        cache_key = np.ascontiguousarray(ids, dtype=np.int64).tobytes()
        cached = self._target_cache.get(cache_key)
        if cached is not None:
            return cached
        adverse_path = self.feature_bank.future_adverse_to_best_peak_path(
            ids, horizon_bars=self.horizon_bars
        ).astype(np.float64, copy=False)
        dates = self.group_dates.iloc[ids].to_numpy()
        valid = np.ones(len(ids), dtype=bool)
        target = np.empty_like(adverse_path, dtype=np.float32)
        for column in range(self.horizon_bars):
            target[:, column] = build_same_date_percentile_targets(
                -adverse_path[:, column], valid, dates
            ).astype(np.float32)
        expected_final = self.final_safety_target[ids]
        if not np.allclose(
            target[:, -1], expected_final, rtol=0.0, atol=1e-6, equal_nan=False
        ):
            max_delta = float(np.max(np.abs(target[:, -1] - expected_final)))
            raise ValueError(
                "adaptive-horizon 40-bar Safety target未精確回到canonical AO target: "
                f"max_delta={max_delta:.8f}"
            )
        target.setflags(write=False)
        self._target_cache[cache_key] = target
        return target


__all__ = ["AdaptiveHorizonSafetyTargetProvider"]
