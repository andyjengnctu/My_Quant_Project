"""Shared rank-calibration math for frozen daily-universal score mappings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonotonicCurve:
    x: np.ndarray
    y: np.ndarray

    def predict(self, values) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 1:
            raise ValueError("rank calibration predict values必須是一維")
        if not np.isfinite(raw).all():
            raise ValueError("rank calibration predict values必須全部有限")
        if self.x.size == 0 or self.y.size != self.x.size:
            raise ValueError("rank calibration curve為空或維度不一致")
        return np.interp(raw, self.x, self.y, left=self.y[0], right=self.y[-1])


def add_daily_score_percentile(score_rows: pd.DataFrame) -> pd.DataFrame:
    frame = score_rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["breakout_quality_score"] = pd.to_numeric(
        frame["breakout_quality_score"], errors="raise"
    ).astype(float)
    frame["daily_score_percentile"] = frame.groupby(
        "date", sort=False
    )["breakout_quality_score"].rank(method="average", pct=True, ascending=True)
    values = frame["daily_score_percentile"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("daily score percentile建立失敗")
    return frame


def build_canonical_target_frame(bundle) -> pd.DataFrame:
    groups = bundle.group_table[
        ["ticker", "date", "group_index", "label_eval_end_date"]
    ].copy()
    groups["ticker"] = groups["ticker"].fillna("").astype(str).str.strip()
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d")
    groups["label_eval_end_date"] = pd.to_datetime(
        groups["label_eval_end_date"], errors="coerce"
    )
    groups["target_raw_r"] = np.asarray(bundle.raw_target, dtype=np.float64)
    groups["target_valid"] = np.asarray(bundle.target_valid, dtype=bool) & np.isfinite(
        groups["target_raw_r"].to_numpy(dtype=np.float64)
    )
    return groups


def mature_target_rows(frame: pd.DataFrame, *, cutoff_exclusive: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_exclusive).normalize()
    dates = pd.to_datetime(frame["date"], errors="raise")
    eval_end = pd.to_datetime(frame["label_eval_end_date"], errors="coerce")
    eligible = frame[
        frame["target_valid"].astype(bool)
        & (dates < cutoff)
        & (eval_end < cutoff)
    ].copy()
    return eligible


def add_daily_excess_r(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["daily_target_mean_r"] = np.nan
    out["target_excess_r"] = np.nan
    valid = out["target_valid"].astype(bool) & np.isfinite(
        pd.to_numeric(out["target_raw_r"], errors="coerce").to_numpy(dtype=np.float64)
    )
    if not bool(valid.any()):
        return out
    valid_rows = out.loc[valid, ["date", "target_raw_r"]].copy()
    day_means = valid_rows.groupby("date", sort=False)["target_raw_r"].mean()
    out.loc[valid, "daily_target_mean_r"] = out.loc[valid, "date"].map(day_means).astype(float)
    out.loc[valid, "target_excess_r"] = (
        out.loc[valid, "target_raw_r"].astype(float)
        - out.loc[valid, "daily_target_mean_r"].astype(float)
    )
    values = out.loc[valid, "target_excess_r"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("daily excess-R建立後包含非有限值")
    return out


def fit_weighted_increasing_isotonic(x, y) -> tuple[MonotonicCurve, dict[str, Any]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("isotonic fit x/y必須為同長度一維")
    if x_values.size < 2 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("isotonic fit至少需要2筆有限樣本")

    grouped = pd.DataFrame({"x": x_values, "y": y_values}).groupby(
        "x", sort=True, as_index=False
    ).agg(y_mean=("y", "mean"), weight=("y", "size"))
    unique_x = grouped["x"].to_numpy(dtype=np.float64)
    means = grouped["y_mean"].to_numpy(dtype=np.float64)
    weights = grouped["weight"].to_numpy(dtype=np.float64)
    if unique_x.size < 2:
        raise ValueError("isotonic fit percentile沒有足夠變異")

    blocks: list[dict[str, float | int]] = []
    for index, (mean, weight) in enumerate(zip(means, weights)):
        blocks.append({
            "start": int(index),
            "end": int(index),
            "weight": float(weight),
            "weighted_sum": float(mean * weight),
        })
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["weighted_sum"]) / float(left["weight"])
            right_mean = float(right["weighted_sum"]) / float(right["weight"])
            if left_mean <= right_mean:
                break
            merged = {
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "weighted_sum": float(left["weighted_sum"]) + float(right["weighted_sum"]),
            }
            blocks[-2:] = [merged]

    fitted_unique = np.empty_like(unique_x)
    for block in blocks:
        block_mean = float(block["weighted_sum"]) / float(block["weight"])
        fitted_unique[int(block["start"]): int(block["end"]) + 1] = block_mean
    if bool((np.diff(fitted_unique) < -1e-12).any()):
        raise RuntimeError("isotonic PAVA輸出非單調")

    curve = MonotonicCurve(x=unique_x, y=fitted_unique)
    fitted_all = curve.predict(x_values)
    mse = float(np.mean((fitted_all - y_values) ** 2))
    mae = float(np.mean(np.abs(fitted_all - y_values)))
    weighted_mean_prediction = float(np.mean(fitted_all))
    return curve, {
        "sample_count": int(x_values.size),
        "unique_percentile_count": int(unique_x.size),
        "isotonic_block_count": int(len(blocks)),
        "target_mean_r": float(np.mean(y_values)),
        "prediction_mean_r": weighted_mean_prediction,
        "prediction_min_r": float(fitted_unique.min()),
        "prediction_max_r": float(fitted_unique.max()),
        "mae_r": mae,
        "rmse_r": float(math.sqrt(mse)),
    }


__all__ = [
    "MonotonicCurve",
    "add_daily_score_percentile",
    "build_canonical_target_frame",
    "mature_target_rows",
    "add_daily_excess_r",
    "fit_weighted_increasing_isotonic",
]
