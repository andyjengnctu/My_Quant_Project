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
        # Production fitting prepares the complete fitting-scope trajectory once.
        # This avoids rebuilding identical 1..H paths/ranks in thousands of
        # complete-date mini-batches during the first epoch.  The prepared matrix is
        # fitting-scope local and never crosses Inner/Selection boundaries.
        self._prepared_targets: np.ndarray | None = None
        self._prepared_lookup: np.ndarray | None = None
        # Keep the lazy exact-batch cache as a compatibility fallback for isolated
        # callers/tests that do not invoke prepare_ids().
        self._target_cache: dict[bytes, np.ndarray] = {}

    @staticmethod
    def _same_date_percentile_matrix(
        values: np.ndarray,
        dates: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Vectorized average-rank percentiles for many horizons at once.

        This is exactly the column-wise equivalent of
        ``build_same_date_percentile_targets`` for finite values, but pandas performs
        one grouped rank over the whole H-column matrix instead of H separate
        groupby/rank passes per training batch.
        """

        # Keep the canonical path's float32 storage.  Promoting finite float32 values
        # to float64 cannot change ordering/ties, while retaining float32 here avoids
        # an unnecessary fitting-scope-sized copy before pandas emits float64 ranks.
        matrix = np.asarray(values)
        if matrix.ndim != 2:
            raise ValueError("adaptive-horizon percentile matrix必須是二維")
        date_series = pd.Series(pd.to_datetime(dates, errors="raise")).dt.normalize().reset_index(drop=True)
        if len(date_series) != len(matrix):
            raise ValueError("adaptive-horizon percentile matrix dates長度不一致")
        if not bool(np.isfinite(matrix).all()):
            raise ValueError("adaptive-horizon percentile matrix含non-finite value")
        if len(matrix) == 0:
            return np.empty(matrix.shape, dtype=np.float32)

        frame = pd.DataFrame(matrix, copy=False)
        date_codes, _ = pd.factorize(date_series, sort=False)
        if bool(np.any(date_codes < 0)):
            raise ValueError("adaptive-horizon percentile matrix date factorization失敗")
        ranks = frame.groupby(date_codes, sort=False).rank(method="average")
        # Pandas Copy-on-Write may expose ``to_numpy(copy=False)`` as a read-only
        # view (and newer Pandas versions increasingly use CoW semantics).  Do not
        # mutate that view.  Normalize into the final float32 result buffer instead;
        # this preserves canonical rank arithmetic while avoiding a second full-size
        # float64 copy of the fitting-scope rank matrix.
        rank_values = ranks.to_numpy(dtype=np.float64, copy=False)
        counts_by_code = np.bincount(date_codes)
        counts = counts_by_code[date_codes].astype(np.int64, copy=False)
        singleton = counts == 1
        denominator = np.maximum(counts - 1, 1).astype(np.float64, copy=False)
        percentile = np.empty(rank_values.shape, dtype=np.float32)
        np.subtract(rank_values, 1.0, out=percentile, casting="unsafe")
        np.divide(percentile, denominator[:, None], out=percentile)
        if bool(np.any(singleton)):
            percentile[singleton] = 0.5
        if not bool(np.isfinite(percentile).all()):
            raise ValueError("adaptive-horizon percentile matrix產生non-finite value")
        return percentile

    def prepare_ids(self, group_ids: np.ndarray) -> None:
        """Eagerly materialize one fitting scope without changing target semantics."""

        ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            self._prepared_targets = np.empty((0, self.horizon_bars), dtype=np.float32)
            self._prepared_lookup = np.full(len(self.group_dates), -1, dtype=np.int32)
            return
        if bool(np.any(ids < 0)) or bool(np.any(ids >= len(self.group_dates))):
            raise IndexError("adaptive-horizon prepare group id超出範圍")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("adaptive-horizon prepare group_ids不得重複")

        adverse_path = self.feature_bank.future_adverse_to_best_peak_path(
            ids, horizon_bars=self.horizon_bars
        ).astype(np.float32, copy=False)
        target = np.empty_like(adverse_path, dtype=np.float32)
        if self.horizon_bars > 1:
            target[:, :-1] = self._same_date_percentile_matrix(
                -adverse_path[:, :-1], self.group_dates.iloc[ids]
            )
        expected_final = self.final_safety_target[ids]
        if bool(np.any(~np.isfinite(expected_final))):
            raise ValueError("adaptive-horizon canonical final Safety target含non-finite value")
        target[:, -1] = expected_final
        if not np.array_equal(target[:, -1], expected_final):
            raise RuntimeError("adaptive-horizon terminal Safety未直接保持canonical AO target")

        lookup = np.full(len(self.group_dates), -1, dtype=np.int32)
        lookup[ids] = np.arange(len(ids), dtype=np.int32)
        target.setflags(write=False)
        lookup.setflags(write=False)
        self._prepared_targets = target
        self._prepared_lookup = lookup
        self._target_cache.clear()

    def targets_for_ids(self, group_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            return np.empty((0, self.horizon_bars), dtype=np.float32)
        if self._prepared_targets is not None and self._prepared_lookup is not None:
            if bool(np.any(ids < 0)) or bool(np.any(ids >= len(self._prepared_lookup))):
                raise IndexError("adaptive-horizon prepared target group id超出範圍")
            local = self._prepared_lookup[ids]
            if bool(np.any(local < 0)):
                raise ValueError("adaptive-horizon batch含fitting scope以外group id")
            return self._prepared_targets[local]
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
        # Horizons 1..H-1 are genuinely new trajectory supervision.  The terminal
        # H-bar Safety target already has a canonical SSOT in the AO target builder,
        # so do not re-rank an independently materialized float path here.  Even a
        # sub-ULP raw-path difference can change tie/order geometry by one same-date
        # rank step on a large cross-section.  Consume the canonical endpoint
        # directly and keep the scientific contract exact instead of weakening the
        # equality tolerance.
        for column in range(self.horizon_bars - 1):
            target[:, column] = build_same_date_percentile_targets(
                -adverse_path[:, column], valid, dates
            ).astype(np.float32)
        expected_final = self.final_safety_target[ids]
        if bool(np.any(~np.isfinite(expected_final))):
            raise ValueError("adaptive-horizon canonical final Safety target含non-finite value")
        target[:, -1] = expected_final
        if not np.array_equal(target[:, -1], expected_final):
            raise RuntimeError("adaptive-horizon terminal Safety未直接保持canonical AO target")
        target.setflags(write=False)
        self._target_cache[cache_key] = target
        return target


__all__ = ["AdaptiveHorizonSafetyTargetProvider"]
