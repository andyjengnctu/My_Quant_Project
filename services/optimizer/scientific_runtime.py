"""Optimizer runtime accessors for an injected Strategy Parameter scientific contract.

The Strategy Parameter producer resolves scientific semantics once and stores the
resolved payload on :class:`OptimizerSession`.  Inner Optimizer consumers use
these helpers instead of re-reading mutable scientific config.  Generic Optimizer
sessions without an injected contract retain the legacy config-backed fallback.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from core.scientific_runtime_context import get_scientific_section, normalize_scientific_contract
from core.training_policy import (
    is_optimizer_local_min_review_enabled,
    resolve_optimizer_base_finalists_agree_min_agree,
    resolve_optimizer_local_finalists_agree_min_agree,
    resolve_optimizer_local_min_score_finalist_top_k,
    resolve_optimizer_retention_finalists_agree_min_agree,
)


def get_session_scientific_contract(session) -> dict[str, Any] | None:
    return normalize_scientific_contract(
        getattr(session, "strategy_param_scientific_contract", None)
    )


def get_session_scientific_section(session, section_name: str) -> dict[str, Any] | None:
    return get_scientific_section(
        getattr(session, "strategy_param_scientific_contract", None), section_name
    )


def resolve_session_local_min_review_enabled(session, *, model_mode=None) -> bool:
    selection = get_session_scientific_section(session, "selection_and_anti_overfit")
    if isinstance(selection, dict):
        local = selection.get("local_min_review")
        if isinstance(local, Mapping):
            normalized = str(model_mode or "").strip().lower()
            if normalized == "full":
                return bool(local.get("full_mode_enabled", local.get("default_enabled", False)))
            return bool(local.get("default_enabled", False))
    return bool(is_optimizer_local_min_review_enabled(model_mode=model_mode))


def resolve_session_local_min_finalist_top_k(session, n_trials: int) -> int:
    selection = get_session_scientific_section(session, "selection_and_anti_overfit")
    if isinstance(selection, dict):
        local = selection.get("local_min_review")
        if isinstance(local, Mapping):
            requested_trials = max(0, int(n_trials))
            rate = float(local.get("finalist_top_k_rate", 0.0) or 0.0)
            minimum = int(local.get("finalist_top_k_min", 1) or 1)
            return max(minimum, int(math.ceil(requested_trials * rate)))
    return int(resolve_optimizer_local_min_score_finalist_top_k(n_trials))


def resolve_session_inner_validation_policy(session) -> dict[str, Any]:
    selection = get_session_scientific_section(session, "selection_and_anti_overfit")
    if isinstance(selection, dict) and isinstance(selection.get("inner_validation"), Mapping):
        return dict(selection["inner_validation"])
    import config.training_policy as settings

    return {
        "enabled": bool(settings.OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED),
        "min_score": float(settings.OPTIMIZER_INNER_VALIDATE_MIN_SCORE),
        "max_rank_percentile": float(settings.OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE),
        "holdout_years": int(settings.OPTIMIZER_INNER_VALIDATE_HOLDOUT_YEARS),
    }


def resolve_session_dominant_year_policy(session) -> dict[str, Any]:
    selection = get_session_scientific_section(session, "selection_and_anti_overfit")
    if isinstance(selection, dict) and isinstance(selection.get("dominant_year_dependency"), Mapping):
        return dict(selection["dominant_year_dependency"])
    import config.training_policy as settings

    return {
        "enabled": bool(settings.OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED),
        "high_positive_pnl_share": float(settings.DOMINANT_YEAR_HIGH_POSITIVE_PNL_SHARE),
        "narrow_positive_trade_count": int(settings.DOMINANT_YEAR_NARROW_POSITIVE_TRADE_COUNT),
        "narrow_positive_symbol_count": int(settings.DOMINANT_YEAR_NARROW_POSITIVE_SYMBOL_COUNT),
        "top_trade_outlier_pnl_share": float(settings.DOMINANT_YEAR_TOP_TRADE_OUTLIER_PNL_SHARE),
    }


def resolve_session_fixed_tp_percent(session, fallback: float) -> float:
    selection = get_session_scientific_section(session, "selection_and_anti_overfit")
    if isinstance(selection, dict) and selection.get("fixed_tp_percent") is not None:
        return float(selection["fixed_tp_percent"])
    return float(fallback)


def resolve_session_finalists_agree_requested(session, policy_name: str, fallback):
    selection = get_session_scientific_section(session, "selection_and_anti_overfit")
    if isinstance(selection, dict):
        finalists = selection.get("finalists_agree")
        if isinstance(finalists, Mapping):
            key = {
                "base_finalists_agree": "base_min_agree",
                "local_finalists_agree": "local_min_agree",
                "retention_finalists_agree": "retention_min_agree",
            }.get(str(policy_name))
            if key and key in finalists:
                return finalists[key]
    return fallback


def resolve_finalists_agree_min_agree(policy_name: str, member_count: int, requested) -> int:
    name = str(policy_name)
    if name == "local_finalists_agree":
        return int(resolve_optimizer_local_finalists_agree_min_agree(member_count, requested))
    if name == "retention_finalists_agree":
        return int(resolve_optimizer_retention_finalists_agree_min_agree(member_count, requested))
    return int(resolve_optimizer_base_finalists_agree_min_agree(member_count, requested))


__all__ = [
    "get_session_scientific_contract",
    "get_session_scientific_section",
    "resolve_finalists_agree_min_agree",
    "resolve_session_dominant_year_policy",
    "resolve_session_finalists_agree_requested",
    "resolve_session_fixed_tp_percent",
    "resolve_session_inner_validation_policy",
    "resolve_session_local_min_finalist_top_k",
    "resolve_session_local_min_review_enabled",
]
