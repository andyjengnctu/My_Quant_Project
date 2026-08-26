"""Canonical MFE × Safety truth geometry shared by Strategy Compare and Audits.

This module owns only post-replay diagnostic truth geometry.  It never participates
in runtime ranking, fitting, parameter selection, or strategy execution.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_experiment_profile
from core.path_utils import project_relative_display_path
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
)
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data

QUADRANT_KEYS = (
    "high_mfe_high_safety_pct",
    "high_mfe_low_safety_pct",
    "low_mfe_high_safety_pct",
    "low_mfe_low_safety_pct",
)

TRUTH_GEOMETRY_QUINTILES = 5


def normalize_ticker(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d")


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ensure_daily_percentile(frame: pd.DataFrame, *, method: str) -> tuple[pd.DataFrame, str]:
    table = pd.DataFrame(frame).copy()
    if "percentile" in table:
        percentile = pd.to_numeric(table["percentile"], errors="coerce")
        valid = percentile.notna() & percentile.between(0.0, 1.0, inclusive="both")
        if valid.all():
            table["percentile"] = percentile.astype(float)
            return table, "canonical percentile column"
    rank_method = str(method).strip().lower()
    if rank_method != "average_zero_based":
        raise ValueError(
            "same-day percentile method目前只接受與既有project rank contract一致的average_zero_based"
        )
    grouped = table.groupby("date", sort=False)["value"]
    average_rank = grouped.rank(method="average") - 1.0
    group_size = grouped.transform("size").astype(float)
    denominator = group_size - 1.0
    denominator_array = denominator.to_numpy(dtype=float)
    percentile = np.full(len(table), 0.5, dtype=float)
    np.divide(
        average_rank.to_numpy(dtype=float),
        denominator_array,
        out=percentile,
        where=denominator_array > 0.0,
    )
    table["percentile"] = percentile
    return table, "derived from canonical raw truth via same-day average zero-based rank / (N-1)"


def validate_truth_provider_contract(
    *,
    provider_profile_id: str,
    mfe_target_id: str,
    safety_target_id: str,
) -> None:
    profile = get_breakout_quality_experiment_profile(str(provider_profile_id))
    if str(profile.continuous_target_id or "") != str(mfe_target_id):
        raise ValueError(
            "MFE/Safety truth provider profile target不一致: "
            f"profile={provider_profile_id}, target={profile.continuous_target_id!r}, "
            f"expected={mfe_target_id!r}"
        )
    if str(safety_target_id) != DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID:
        raise ValueError(
            "MFE/Safety safety target目前必須使用canonical "
            f"{DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID!r}"
        )


def build_truth_geometry(
    *,
    project_root: Path,
    filter_id: str,
    model_architecture: str,
    provider_profile_id: str,
    mfe_target_id: str,
    safety_target_id: str,
    percentile_method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_truth_provider_contract(
        provider_profile_id=provider_profile_id,
        mfe_target_id=mfe_target_id,
        safety_target_id=safety_target_id,
    )
    root = Path(project_root).resolve()
    bundle = load_profile_continuous_ranker_data(
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(provider_profile_id),
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table.copy()
    required = {"ticker", "date", "target_adverse_r"}
    missing = sorted(required - set(groups.columns))
    if missing:
        raise ValueError(f"canonical daily truth provider缺少欄位: {missing}")
    if len(groups) != len(bundle.raw_target) or len(groups) != len(bundle.target_valid):
        raise ValueError("canonical daily truth provider group/target長度不一致")
    target_contract = dict(dict(bundle.target_manifest or {}).get("target_contract") or {})
    if str(target_contract.get("target_id") or "") != str(mfe_target_id):
        raise ValueError("canonical daily truth provider target identity不一致")

    valid = np.asarray(bundle.target_valid, dtype=bool)
    mfe_value = pd.to_numeric(pd.Series(bundle.raw_target, index=groups.index), errors="coerce")
    adverse_value = pd.to_numeric(groups["target_adverse_r"], errors="coerce")
    valid &= np.isfinite(mfe_value.to_numpy(dtype=float))
    valid &= np.isfinite(adverse_value.to_numpy(dtype=float))
    if not bool(valid.any()):
        raise ValueError("canonical daily truth provider沒有有效MFE/Safety rows")

    base = pd.DataFrame({
        "ticker": groups.loc[valid, "ticker"].map(normalize_ticker).to_numpy(),
        "date": pd.to_datetime(groups.loc[valid, "date"], errors="raise")
        .dt.strftime("%Y-%m-%d").to_numpy(),
        "mfe_value": mfe_value.loc[valid].to_numpy(dtype=float),
        "safety_value": (-adverse_value.loc[valid]).to_numpy(dtype=float),
    })
    if bool(base.duplicated(["ticker", "date"]).any()):
        raise ValueError("canonical daily truth provider存在重複ticker/date")

    mfe = base[["ticker", "date", "mfe_value"]].rename(columns={"mfe_value": "value"})
    safety = base[["ticker", "date", "safety_value"]].rename(columns={"safety_value": "value"})
    mfe, mfe_percentile_source = ensure_daily_percentile(mfe, method=percentile_method)
    safety, safety_percentile_source = ensure_daily_percentile(safety, method=percentile_method)
    joined = mfe[["ticker", "date", "percentile"]].rename(
        columns={"percentile": "mfe_percentile"}
    ).merge(
        safety[["ticker", "date", "percentile"]].rename(
            columns={"percentile": "safety_percentile"}
        ),
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("Pure-MFE與Low-Adverse Safety canonical truth沒有共同ticker/date")

    dataset_root = resolve_filter_output_dir(root, filter_id=str(filter_id))
    source = {
        "provider": "canonical profile-aware daily sample provider",
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "provider_profile_id": str(provider_profile_id),
        "dataset_root": project_relative_display_path(dataset_root, project_root=root),
        "mfe_target_id": str(mfe_target_id),
        "mfe_truth_source": "bundle.raw_target",
        "mfe_percentile_source": mfe_percentile_source,
        "safety_target_id": str(safety_target_id),
        "safety_truth_source": "-bundle.group_table.target_adverse_r",
        "safety_percentile_source": safety_percentile_source,
        "joined_truth_rows": int(len(joined)),
    }
    return joined, source


def attach_quadrants(frame: pd.DataFrame, *, cutoff: float) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    mfe_high = table["mfe_percentile"] >= float(cutoff)
    safety_high = table["safety_percentile"] >= float(cutoff)
    table["quadrant"] = np.select(
        (
            mfe_high & safety_high,
            mfe_high & ~safety_high,
            ~mfe_high & safety_high,
            ~mfe_high & ~safety_high,
        ),
        QUADRANT_KEYS,
        default="",
    )
    return table


def distribution_for_keys(
    truth: pd.DataFrame,
    keys: pd.DataFrame | None,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if keys is None:
        covered = pd.DataFrame(truth).copy()
        raw_count = int(len(covered))
    else:
        key_frame = pd.DataFrame(keys)[["ticker", "date"]].drop_duplicates()
        raw_count = int(len(key_frame))
        covered = key_frame.merge(truth, on=["ticker", "date"], how="inner", validate="one_to_one")
    covered_count = int(len(covered))
    if covered_count <= 0:
        if allow_empty:
            return {
                "raw_rows": raw_count,
                "truth_covered_rows": 0,
                "truth_coverage_pct": 0.0,
                **{key: None for key in QUADRANT_KEYS},
                "high_mfe_total_pct": None,
                "high_safety_total_pct": None,
            }
        raise ValueError("cohort與MFE/Safety truth沒有任何共同ticker/date")
    counts = covered["quadrant"].value_counts().to_dict()
    result: dict[str, Any] = {
        "raw_rows": raw_count,
        "truth_covered_rows": covered_count,
        "truth_coverage_pct": (covered_count / raw_count * 100.0) if raw_count else 100.0,
    }
    for key in QUADRANT_KEYS:
        result[key] = float(counts.get(key, 0)) / covered_count * 100.0
    result["high_mfe_total_pct"] = float(
        result["high_mfe_high_safety_pct"] + result["high_mfe_low_safety_pct"]
    )
    result["high_safety_total_pct"] = float(
        result["high_mfe_high_safety_pct"] + result["low_mfe_high_safety_pct"]
    )
    return result


def fixed_quintiles_from_percentiles(
    values: pd.Series | np.ndarray,
    *,
    bins: int = TRUTH_GEOMETRY_QUINTILES,
) -> np.ndarray:
    """Map canonical [0,1] percentiles into fixed 1..bins buckets.

    This helper never re-ranks a subset.  Callers must pass the canonical
    daily-universal percentile values already owned by this domain.
    """

    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1 or not bool(np.isfinite(numeric).all()):
        raise ValueError("MFE/Safety percentile必須為finite 1D")
    if bool(np.any((numeric < 0.0) | (numeric > 1.0))):
        raise ValueError("MFE/Safety percentile必須位於[0,1]")
    count = int(bins)
    if count <= 1:
        raise ValueError("MFE/Safety quintile bins必須>1")
    return np.minimum(count - 1, np.floor(numeric * count).astype(np.int64)) + 1


def truth_geometry_5x5(
    truth: pd.DataFrame,
    keys: pd.DataFrame | None = None,
    *,
    bins: int = TRUTH_GEOMETRY_QUINTILES,
) -> dict[str, Any]:
    """Summarize canonical actual Safety×MFE joint support.

    ``keys`` only filters membership; percentile values are always inherited from
    ``truth``.  Therefore breakout/orderable/planned cohorts can be compared to the
    daily-universal geometry without silently re-ranking inside the subset.
    """

    table = pd.DataFrame(truth).copy()
    required = {"ticker", "date", "mfe_percentile", "safety_percentile"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"MFE/Safety truth geometry缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    table["date"] = table["date"].map(normalize_date)
    if keys is not None:
        key_frame = pd.DataFrame(keys).copy()
        if not {"ticker", "date"}.issubset(key_frame.columns):
            raise ValueError("MFE/Safety geometry keys必須包含ticker/date")
        key_frame["ticker"] = key_frame["ticker"].map(normalize_ticker)
        key_frame["date"] = key_frame["date"].map(normalize_date)
        key_frame = key_frame.loc[
            key_frame["ticker"].ne("") & key_frame["date"].ne("")
        ].drop_duplicates(["ticker", "date"])
        raw_rows = int(len(key_frame))
        table = key_frame.merge(
            table,
            on=["ticker", "date"],
            how="inner",
            validate="one_to_one",
        )
    else:
        raw_rows = int(len(table))
    if table.empty:
        return {
            "raw_rows": raw_rows,
            "truth_covered_rows": 0,
            "truth_coverage_pct": 0.0,
            "safety_to_mfe_spearman": None,
            "cells": [[{"n": 0, "population_pct": None, "expected_n_independent": None, "independence_enrichment": None} for _ in range(int(bins))] for _ in range(int(bins))],
            "s5_m5": {"n": 0, "population_pct": None, "expected_n_independent": None, "independence_enrichment": None},
            "s4plus_m4plus": {"n": 0, "population_pct": None, "expected_n_independent": None, "independence_enrichment": None},
            "percentile_policy": "canonical_daily_universal_percentiles_no_subset_rerank",
        }

    safety = pd.to_numeric(table["safety_percentile"], errors="coerce").to_numpy(dtype=float)
    mfe = pd.to_numeric(table["mfe_percentile"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(safety) & np.isfinite(mfe)
    table = table.loc[valid].copy()
    safety = safety[valid]
    mfe = mfe[valid]
    if table.empty:
        raise ValueError("MFE/Safety geometry cohort沒有finite percentile")
    safety_q = fixed_quintiles_from_percentiles(safety, bins=bins)
    mfe_q = fixed_quintiles_from_percentiles(mfe, bins=bins)
    population = int(len(table))
    row_counts = np.asarray([(safety_q == level).sum() for level in range(1, int(bins) + 1)], dtype=np.int64)
    col_counts = np.asarray([(mfe_q == level).sum() for level in range(1, int(bins) + 1)], dtype=np.int64)

    def summarize(mask: np.ndarray, expected_n: float) -> dict[str, Any]:
        n = int(mask.sum())
        return {
            "n": n,
            "population_pct": float(n / population * 100.0),
            "expected_n_independent": float(expected_n),
            "independence_enrichment": None if expected_n <= 0.0 else float(n / expected_n),
        }

    cells: list[list[dict[str, Any]]] = []
    for safety_level in range(1, int(bins) + 1):
        row: list[dict[str, Any]] = []
        for mfe_level in range(1, int(bins) + 1):
            expected = float(
                row_counts[safety_level - 1]
                * col_counts[mfe_level - 1]
                / population
            )
            row.append(summarize((safety_q == safety_level) & (mfe_q == mfe_level), expected))
        cells.append(row)

    highest = int(bins)
    upper_start = max(1, highest - 1)
    s5_expected = float(row_counts[-1] * col_counts[-1] / population)
    upper_rows = int(row_counts[upper_start - 1 :].sum())
    upper_cols = int(col_counts[upper_start - 1 :].sum())
    upper_expected = float(upper_rows * upper_cols / population)
    daily_rhos: list[float] = []
    relation_frame = pd.DataFrame({
        "date": table["date"].to_numpy(),
        "safety": safety,
        "mfe": mfe,
    })
    for _date, day in relation_frame.groupby("date", sort=False):
        if len(day) < 2 or day["safety"].nunique() < 2 or day["mfe"].nunique() < 2:
            continue
        rho = day["safety"].corr(day["mfe"], method="spearman")
        finite = finite_float(rho)
        if finite is not None:
            daily_rhos.append(finite)
    return {
        "raw_rows": raw_rows,
        "truth_covered_rows": population,
        "truth_coverage_pct": float(population / raw_rows * 100.0) if raw_rows else 100.0,
        "safety_to_mfe_spearman": finite_float(
            pd.Series(safety).corr(pd.Series(mfe), method="spearman")
        ),
        "safety_to_mfe_mean_daily_spearman": (
            None if not daily_rhos else float(np.mean(daily_rhos))
        ),
        "valid_days": int(len(daily_rhos)),
        "cells": cells,
        "s5_m5": summarize((safety_q == highest) & (mfe_q == highest), s5_expected),
        "s4plus_m4plus": summarize(
            (safety_q >= upper_start) & (mfe_q >= upper_start),
            upper_expected,
        ),
        "percentile_policy": "canonical_daily_universal_percentiles_no_subset_rerank",
        "independence_expected_method": "scope_marginals_n_times_p_s_times_p_m",
    }


def filter_period(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    result = pd.DataFrame(frame)
    if start_date:
        result = result.loc[result["date"] >= str(start_date)]
    if end_date:
        result = result.loc[result["date"] <= str(end_date)]
    return result.copy()


__all__ = [
    "QUADRANT_KEYS",
    "TRUTH_GEOMETRY_QUINTILES",
    "attach_quadrants",
    "build_truth_geometry",
    "distribution_for_keys",
    "ensure_daily_percentile",
    "filter_period",
    "finite_float",
    "fixed_quintiles_from_percentiles",
    "normalize_date",
    "normalize_ticker",
    "truth_geometry_5x5",
    "validate_truth_provider_contract",
]
