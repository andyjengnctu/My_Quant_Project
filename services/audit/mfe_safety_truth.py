"""Shared read-only MFE × Safety truth geometry primitives for formal Audits."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AuditDefinition
from config.breakout_quality import get_breakout_quality_experiment_profile
from core.path_utils import project_relative_display_path
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
)
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.strategy_compare_diagnostics import (
    strategy_replay_score_event_frames,
)
from services.audit.strategy_compare_source import AuditSourceBlockedError

AuditBlockedError = AuditSourceBlockedError

QUADRANT_KEYS = (
    "high_mfe_high_safety_pct",
    "high_mfe_low_safety_pct",
    "low_mfe_high_safety_pct",
    "low_mfe_low_safety_pct",
)


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


def resolve_truth_provider_contract(
    definition: AuditDefinition,
) -> tuple[str, str, str, str, str]:
    source = definition.source
    filter_id = str(source.get("filter_id") or "").strip()
    model_architecture = str(source.get("model_architecture") or "").strip()
    provider_profile_id = str(source.get("truth_provider_profile_id") or "").strip()
    mfe_target_id = str(source.get("mfe_target_id") or "").strip()
    safety_target_id = str(source.get("safety_target_id") or "").strip()
    missing = [
        name
        for name, value in (
            ("filter_id", filter_id),
            ("model_architecture", model_architecture),
            ("truth_provider_profile_id", provider_profile_id),
            ("mfe_target_id", mfe_target_id),
            ("safety_target_id", safety_target_id),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"{definition.audit_id} truth provider設定不完整: {', '.join(missing)}"
        )
    profile = get_breakout_quality_experiment_profile(provider_profile_id)
    if str(profile.continuous_target_id or "") != mfe_target_id:
        raise ValueError(
            f"{definition.audit_id} truth provider profile target不一致: "
            f"profile={provider_profile_id}, target={profile.continuous_target_id!r}, "
            f"expected={mfe_target_id!r}"
        )
    if safety_target_id != DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID:
        raise ValueError(
            f"{definition.audit_id} safety target目前必須使用canonical "
            f"{DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID!r}"
        )
    return filter_id, model_architecture, provider_profile_id, mfe_target_id, safety_target_id


def build_truth_geometry(
    definition: AuditDefinition,
    project_root: Path,
    *,
    percentile_method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    (
        filter_id,
        model_architecture,
        provider_profile_id,
        mfe_target_id,
        safety_target_id,
    ) = resolve_truth_provider_contract(definition)
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=provider_profile_id,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=project_root,
    )
    groups = bundle.group_table.copy()
    required = {"ticker", "date", "target_adverse_r"}
    missing = sorted(required - set(groups.columns))
    if missing:
        raise AuditBlockedError(f"canonical daily truth provider缺少欄位: {missing}")
    if len(groups) != len(bundle.raw_target) or len(groups) != len(bundle.target_valid):
        raise AuditBlockedError("canonical daily truth provider group/target長度不一致")
    target_contract = dict(dict(bundle.target_manifest or {}).get("target_contract") or {})
    if str(target_contract.get("target_id") or "") != mfe_target_id:
        raise AuditBlockedError("canonical daily truth provider target identity不一致")

    valid = np.asarray(bundle.target_valid, dtype=bool)
    mfe_value = pd.to_numeric(pd.Series(bundle.raw_target, index=groups.index), errors="coerce")
    adverse_value = pd.to_numeric(groups["target_adverse_r"], errors="coerce")
    valid &= np.isfinite(mfe_value.to_numpy(dtype=float))
    valid &= np.isfinite(adverse_value.to_numpy(dtype=float))
    if not bool(valid.any()):
        raise AuditBlockedError("canonical daily truth provider沒有有效MFE/Safety rows")

    base = pd.DataFrame(
        {
            "ticker": groups.loc[valid, "ticker"].astype(str).str.strip().to_numpy(),
            "date": pd.to_datetime(groups.loc[valid, "date"], errors="raise")
            .dt.strftime("%Y-%m-%d")
            .to_numpy(),
            "mfe_value": mfe_value.loc[valid].to_numpy(dtype=float),
            "safety_value": (-adverse_value.loc[valid]).to_numpy(dtype=float),
        }
    )
    if bool(base.duplicated(["ticker", "date"]).any()):
        raise AuditBlockedError("canonical daily truth provider存在重複ticker/date")

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
        raise AuditBlockedError("Pure-MFE與Low-Adverse Safety canonical truth沒有共同ticker/date")

    dataset_root = resolve_filter_output_dir(project_root, filter_id=filter_id)
    source = {
        "provider": "canonical profile-aware daily sample provider",
        "filter_id": filter_id,
        "model_architecture": model_architecture,
        "provider_profile_id": provider_profile_id,
        "dataset_root": project_relative_display_path(dataset_root, project_root=project_root),
        "mfe_target_id": mfe_target_id,
        "mfe_truth_source": "bundle.raw_target",
        "mfe_percentile_source": mfe_percentile_source,
        "safety_target_id": safety_target_id,
        "safety_truth_source": "-bundle.group_table.target_adverse_r",
        "safety_percentile_source": safety_percentile_source,
        "joined_truth_rows": int(len(joined)),
    }
    return joined, source


def attach_quadrants(frame: pd.DataFrame, *, cutoff: float) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    mfe_high = table["mfe_percentile"] >= float(cutoff)
    safety_high = table["safety_percentile"] >= float(cutoff)
    labels = (
        "high_mfe_high_safety_pct",
        "high_mfe_low_safety_pct",
        "low_mfe_high_safety_pct",
        "low_mfe_low_safety_pct",
    )
    table["quadrant"] = np.select(
        (
            mfe_high & safety_high,
            mfe_high & ~safety_high,
            ~mfe_high & safety_high,
            ~mfe_high & ~safety_high,
        ),
        labels,
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
        covered = key_frame.merge(
            truth,
            on=["ticker", "date"],
            how="inner",
            validate="one_to_one",
        )
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
        raise AuditBlockedError("cohort與MFE/Safety truth沒有任何共同ticker/date")
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


def score_event_keys(
    orderable: pd.DataFrame,
    selected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    orderable_events, selected_events = strategy_replay_score_event_frames(orderable, selected)
    frame = orderable_events if selected_events is None else selected_events
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    keys = pd.DataFrame(
        {
            "ticker": frame.get("ticker", pd.Series("", index=frame.index)).map(normalize_ticker),
            "date": frame.get("score_event_date", pd.Series("", index=frame.index)).map(normalize_date),
        }
    )
    keys = keys.loc[keys["ticker"].ne("") & keys["date"].ne("")]
    return keys.drop_duplicates(["ticker", "date"]).sort_values(
        ["date", "ticker"], kind="stable"
    )


def filter_period(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    result = pd.DataFrame(frame)
    if start_date:
        result = result.loc[result["date"] >= start_date]
    if end_date:
        result = result.loc[result["date"] <= end_date]
    return result.copy()


__all__ = [
    "AuditBlockedError",
    "QUADRANT_KEYS",
    "attach_quadrants",
    "build_truth_geometry",
    "distribution_for_keys",
    "ensure_daily_percentile",
    "filter_period",
    "finite_float",
    "normalize_date",
    "normalize_ticker",
    "resolve_truth_provider_contract",
    "score_event_keys",
]
