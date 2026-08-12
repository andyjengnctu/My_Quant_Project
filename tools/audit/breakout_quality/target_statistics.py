"""Shared continuous-target Audit statistics; not model-training metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT

PERCENTILES = (0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)

def source_data_end(summary: dict[str, Any], events: pd.DataFrame) -> str:
    source_range = summary.get("source_data_date_range")
    if isinstance(source_range, dict):
        value = str(source_range.get("end") or "").strip()
        if value:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(pd.to_datetime(events["label_eval_end_date"], errors="raise").max().date())

def collapse_group_frame(
    events: pd.DataFrame,
    event_group_index: np.ndarray,
    targets: dict[str, np.ndarray],
    split_group_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = events.copy()
    frame["group_index"] = np.asarray(event_group_index, dtype=np.int64)
    checked_columns = (
        "ticker",
        "date",
        "label",
        "max_upside_return",
        "max_downside_return",
        "decision_mfe_return",
        "decision_mae_return",
        "decision_reward_risk_ratio",
        "first_hit_bar",
    )
    for column in checked_columns:
        if column not in frame.columns:
            raise ValueError(f"continuous target audit events.csv 缺少欄位: {column}")
        nonunique = frame.groupby("group_index", sort=False)[column].nunique(dropna=False)
        if bool((nonunique > 1).any()):
            sample = nonunique[nonunique > 1].index[:5].tolist()
            raise ValueError(f"同一group_index的{column}不一致: sample={sample}")

    group_frame = (
        frame.sort_index()
        .drop_duplicates("group_index", keep="first")
        .sort_values("group_index")
        .reset_index(drop=True)
    )
    expected = np.arange(len(group_frame), dtype=np.int64)
    observed = group_frame["group_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("continuous target group frame未完整覆蓋連續group_index")

    group_frame["date"] = pd.to_datetime(group_frame["date"], errors="raise").dt.normalize()
    for name, array in targets.items():
        if len(array) != len(group_frame):
            raise ValueError(f"continuous target array長度不一致: {name}")
        group_frame[name] = np.asarray(array)

    for split_name, indices in split_group_indices.items():
        mask = np.zeros(len(group_frame), dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        group_frame[f"is_{split_name}"] = mask
    return group_frame

def finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]

def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 3:
        return None
    x_rank = pd.Series(x_values[valid]).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_values[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])

def binary_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [LABEL_REJECT, LABEL_PASS])
    y = y[valid]
    s = s[valid]
    positive_count = int((y == LABEL_PASS).sum())
    negative_count = int((y == LABEL_REJECT).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_positive = float(ranks[y == LABEL_PASS].sum())
    auc = (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)
    return float(auc)

def positive_sum_concentration(values: np.ndarray, top_fraction: float) -> float | None:
    positive = np.sort(values[values > 0.0])
    if positive.size == 0:
        return None
    top_count = max(1, int(math.ceil(float(positive.size) * float(top_fraction))))
    denominator = float(positive.sum())
    if denominator <= 0.0:
        return None
    return float(positive[-top_count:].sum() / denominator)

def distribution_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = finite_values(frame["target_raw_r"])
    if values.size == 0:
        return {"group_count": 0}
    quantiles = np.quantile(values, PERCENTILES)
    quantile_payload = {
        f"p{int(round(percentile * 100)):02d}": float(value)
        for percentile, value in zip(PERCENTILES, quantiles)
    }
    labels = frame.loc[np.isfinite(frame["target_raw_r"]), "label"].to_numpy(dtype=np.int64)
    pass_values = values[labels == LABEL_PASS]
    reject_values = values[labels == LABEL_REJECT]
    return {
        "group_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "quantiles": quantile_payload,
        "positive_rate": float((values > 0.0).mean()),
        "nonpositive_rate": float((values <= 0.0).mean()),
        "exact_zero_rate": float((values == 0.0).mean()),
        "unique_value_count": int(np.unique(values).size),
        "unique_value_ratio": float(np.unique(values).size / values.size),
        "top_1pct_positive_sum_share": positive_sum_concentration(values, 0.01),
        "top_5pct_positive_sum_share": positive_sum_concentration(values, 0.05),
        "pass_target_mean": float(pass_values.mean()) if pass_values.size else None,
        "pass_target_median": float(np.median(pass_values)) if pass_values.size else None,
        "reject_target_mean": float(reject_values.mean()) if reject_values.size else None,
        "reject_target_median": float(np.median(reject_values)) if reject_values.size else None,
        "binary_label_auc": binary_auc(labels, values),
        "spearman_with_max_upside": spearman(frame["target_raw_r"], frame["max_upside_return"]),
        "spearman_with_decision_mfe": spearman(frame["target_raw_r"], frame["decision_mfe_return"]),
        "spearman_with_negative_decision_mae": spearman(
            frame["target_raw_r"],
            -pd.to_numeric(frame["decision_mae_return"], errors="coerce"),
        ),
    }

def tied_pair_count(counts: np.ndarray) -> int:
    values = np.asarray(counts, dtype=np.int64)
    return int(np.sum(values * (values - 1) // 2))

def daily_rankability(frame: pd.DataFrame, *, split_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for date_value, day in frame.groupby("date", sort=True):
        values = finite_values(day["target_raw_r"])
        if values.size == 0:
            continue
        unique, counts = np.unique(values, return_counts=True)
        pair_count = int(values.size * (values.size - 1) // 2)
        tied_pairs = tied_pair_count(counts)
        rows.append(
            {
                "split": str(split_name),
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "group_count": int(values.size),
                "unique_target_count": int(unique.size),
                "unique_target_ratio": float(unique.size / values.size),
                "pair_count": pair_count,
                "tied_pair_count": int(tied_pairs),
                "comparable_pair_count": int(pair_count - tied_pairs),
                "target_min": float(values.min()),
                "target_max": float(values.max()),
                "target_spread": float(values.max() - values.min()),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"date_count": 0}, daily
    multi = daily[daily["group_count"] >= 2].copy()
    if multi.empty:
        return {
            "date_count": int(len(daily)),
            "multi_candidate_date_count": 0,
        }, daily
    total_pairs = int(multi["pair_count"].sum())
    comparable_pairs = int(multi["comparable_pair_count"].sum())
    rankable = multi["unique_target_count"] >= 2
    return {
        "date_count": int(len(daily)),
        "multi_candidate_date_count": int(len(multi)),
        "multi_candidate_date_rate": float(len(multi) / len(daily)),
        "rankable_date_count": int(rankable.sum()),
        "rankable_date_rate": float(rankable.mean()),
        "median_candidates_per_multi_candidate_date": float(multi["group_count"].median()),
        "p90_candidates_per_multi_candidate_date": float(multi["group_count"].quantile(0.90)),
        "mean_unique_target_ratio": float(multi["unique_target_ratio"].mean()),
        "pairwise_non_tie_rate": (
            float(comparable_pairs / total_pairs) if total_pairs > 0 else None
        ),
        "median_target_spread": float(multi["target_spread"].median()),
        "p10_target_spread": float(multi["target_spread"].quantile(0.10)),
    }, daily

def same_day_binary_concordance(frame: pd.DataFrame) -> dict[str, Any]:
    date_scores: list[float] = []
    weighted_numerator = 0.0
    weighted_denominator = 0
    for _date, day in frame.groupby("date", sort=False):
        pass_values = finite_values(day.loc[day["label"] == LABEL_PASS, "target_raw_r"])
        reject_values = finite_values(day.loc[day["label"] == LABEL_REJECT, "target_raw_r"])
        if pass_values.size == 0 or reject_values.size == 0:
            continue
        comparisons = pass_values[:, None] - reject_values[None, :]
        score = float((comparisons > 0.0).mean() + 0.5 * (comparisons == 0.0).mean())
        pair_count = int(comparisons.size)
        date_scores.append(score)
        weighted_numerator += score * pair_count
        weighted_denominator += pair_count
    return {
        "mixed_label_date_count": int(len(date_scores)),
        "mean_date_concordance": float(np.mean(date_scores)) if date_scores else None,
        "median_date_concordance": float(np.median(date_scores)) if date_scores else None,
        "pair_weighted_concordance": (
            float(weighted_numerator / weighted_denominator)
            if weighted_denominator > 0
            else None
        ),
        "comparable_pass_reject_pair_count": int(weighted_denominator),
    }

def format_metric(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.{digits}f}"

def format_percent(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value) * 100:.2f}%"

__all__ = ["source_data_end", "collapse_group_frame", "finite_values", "spearman", "distribution_metrics", "daily_rankability", "same_day_binary_concordance", "format_metric", "format_percent"]
