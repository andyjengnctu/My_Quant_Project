"""Profile-driven sample and score-eligibility contracts for continuous rankers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT


def build_score_eligibility_contract(experiment_profile: str | Any) -> dict[str, Any]:
    """Describe which rows may receive runtime/PIT scores without future information."""

    profile = (
        get_breakout_quality_experiment_profile(experiment_profile)
        if isinstance(experiment_profile, str)
        else experiment_profile
    )
    sample_scope = str(profile.training_sample_scope)
    if sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        return {
            "schema_version": 1,
            "training_sample_scope": sample_scope,
            "eligibility_basis": "feature_history_only",
            "future_target_required_for_score": False,
            "target_valid_required_for_training": True,
        }
    if sample_scope == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
        return {
            "schema_version": 1,
            "training_sample_scope": sample_scope,
            "eligibility_basis": "canonical_breakout_event_membership",
            "future_target_required_for_score": False,
            "target_valid_required_for_training": True,
        }
    raise ValueError(f"不支援的continuous ranker sample scope: {sample_scope!r}")


def resolve_forward_oos_score_group_ids(bundle) -> np.ndarray:
    """Return every row eligible for frozen Forward-OOS inference.

    Score availability is a prediction-time property.  It follows the configured
    sample universe and Forward-OOS date window, and must not depend on whether the
    row's future target is complete enough for later model-quality evaluation.
    """

    group_table = bundle.group_table
    if "date" not in group_table.columns:
        raise ValueError("continuous ranker group table缺少date，無法解析Forward-OOS score universe")
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    policy = dict(bundle.outer_policy or {})
    start_text = str(policy.get("oos_start_date") or "").strip()
    end_text = str(policy.get("effective_oos_end_date") or "").strip()
    if not start_text or not end_text:
        raise ValueError("continuous ranker outer policy缺少Forward-OOS start/end")
    start = pd.Timestamp(start_text).normalize()
    end = pd.Timestamp(end_text).normalize()
    if end < start:
        raise ValueError("continuous ranker Forward-OOS end不可早於start")
    mask = np.asarray((dates >= start) & (dates <= end), dtype=bool)
    ids = np.flatnonzero(mask).astype(np.int64)
    if ids.size == 0:
        raise ValueError(
            "continuous ranker Forward-OOS score universe為空: "
            f"period={start.date()}~{end.date()}"
        )
    return ids


def resolve_forward_oos_target_evaluable_group_ids(bundle) -> np.ndarray:
    """Return the Forward-OOS subset allowed to use future targets for metrics only."""

    inference_ids = resolve_forward_oos_score_group_ids(bundle)
    target_valid = np.asarray(bundle.target_valid, dtype=bool)
    if len(target_valid) != len(bundle.group_table):
        raise ValueError("continuous ranker target_valid與group table長度不一致")
    mask = target_valid.copy()
    profile = bundle.profile
    if str(profile.training_sample_scope) == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
        if "label" not in bundle.group_table.columns:
            raise ValueError("event continuous ranker group table缺少label")
        labels = pd.to_numeric(bundle.group_table["label"], errors="raise").to_numpy(dtype=np.int64)
        mask &= np.isin(labels, [LABEL_REJECT, LABEL_PASS])
    return inference_ids[mask[inference_ids]]


__all__ = [
    "build_score_eligibility_contract",
    "resolve_forward_oos_score_group_ids",
    "resolve_forward_oos_target_evaluable_group_ids",
]
