"""Audit adapter for the canonical MFE × Safety truth geometry owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.audit_policy import AuditDefinition
from filters.breakout_quality.mfe_safety_geometry import (
    QUADRANT_KEYS,
    attach_quadrants,
    build_truth_geometry as _build_truth_geometry,
    distribution_for_keys,
    ensure_daily_percentile,
    filter_period,
    finite_float,
    normalize_date,
    normalize_ticker,
    truth_geometry_5x5,
    validate_truth_provider_contract,
)
from filters.breakout_quality.strategy_compare_diagnostics import strategy_replay_score_event_frames
from services.audit.strategy_compare_source import AuditSourceBlockedError

AuditBlockedError = AuditSourceBlockedError


def resolve_truth_provider_contract(
    definition: AuditDefinition,
) -> tuple[str, str, str, str, str]:
    source = definition.source
    values = (
        str(source.get("filter_id") or "").strip(),
        str(source.get("model_architecture") or "").strip(),
        str(source.get("truth_provider_profile_id") or "").strip(),
        str(source.get("mfe_target_id") or "").strip(),
        str(source.get("safety_target_id") or "").strip(),
    )
    labels = (
        "filter_id", "model_architecture", "truth_provider_profile_id",
        "mfe_target_id", "safety_target_id",
    )
    missing = [label for label, value in zip(labels, values) if not value]
    if missing:
        raise ValueError(
            f"{definition.audit_id} truth provider設定不完整: {', '.join(missing)}"
        )
    validate_truth_provider_contract(
        provider_profile_id=values[2],
        mfe_target_id=values[3],
        safety_target_id=values[4],
    )
    return values


def build_truth_geometry(
    definition: AuditDefinition,
    project_root: Path,
    *,
    percentile_method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    filter_id, architecture, provider_profile, mfe_target, safety_target = (
        resolve_truth_provider_contract(definition)
    )
    try:
        return _build_truth_geometry(
            project_root=Path(project_root),
            filter_id=filter_id,
            model_architecture=architecture,
            provider_profile_id=provider_profile,
            mfe_target_id=mfe_target,
            safety_target_id=safety_target,
            percentile_method=percentile_method,
        )
    except ValueError as exc:
        raise AuditBlockedError(str(exc)) from exc


def score_event_keys(
    orderable: pd.DataFrame,
    selected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    orderable_events, selected_events = strategy_replay_score_event_frames(orderable, selected)
    frame = orderable_events if selected_events is None else selected_events
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    keys = pd.DataFrame({
        "ticker": frame.get("ticker", pd.Series("", index=frame.index)).map(normalize_ticker),
        "date": frame.get("score_event_date", pd.Series("", index=frame.index)).map(normalize_date),
    })
    keys = keys.loc[keys["ticker"].ne("") & keys["date"].ne("")]
    return keys.drop_duplicates(["ticker", "date"]).sort_values(["date", "ticker"], kind="stable")


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
    "truth_geometry_5x5",
    "resolve_truth_provider_contract",
    "score_event_keys",
]
