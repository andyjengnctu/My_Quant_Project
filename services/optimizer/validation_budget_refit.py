"""Validation-selected Optimizer search budget with full-selection refit.

This module owns the reusable search-budget-selection primitive.  It never reads
OOS results when selecting the budget: a fold's final 12 months of the selection
window are the only validation rows.  The chosen budget is then consumed by a
fresh Optimizer study over the complete selection window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.strategy_param_policy import get_strategy_param_validation_protocol
from services.optimizer.outer_rolling_evaluation import (
    _evaluate_finalist_ensemble_oos_metrics,
)
from services.optimizer.outer_rolling_policy import _build_policy_items
from services.optimizer.robustness import list_local_min_score_finalists


@dataclass(frozen=True)
class ValidationBudgetSpec:
    protocol: str
    holdout_months: int
    checkpoints: tuple[int, ...]
    policy_name: str
    selection_metric: str
    tie_break: str = "smaller_budget"
    oos_feedback_used: bool = False
    full_refit_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "holdout_months": int(self.holdout_months),
            "checkpoints": [int(value) for value in self.checkpoints],
            "policy_name": self.policy_name,
            "selection_metric": self.selection_metric,
            "tie_break": self.tie_break,
            "oos_feedback_used": bool(self.oos_feedback_used),
            "full_refit_required": bool(self.full_refit_required),
        }


class _StudyTrialPrefix:
    """Minimal Optuna-study view exposing only the first N completed trials."""

    def __init__(self, source_study, count: int):
        trials = list(getattr(source_study, "trials", []) or [])
        self.trials = trials[: int(count)]


def resolve_validation_budget_spec(
    max_trials: int, *, protocol_id: str = "validation_budget_refit_v1"
) -> ValidationBudgetSpec:
    max_trials = int(max_trials)
    configured = get_strategy_param_validation_protocol(protocol_id)
    raw = tuple(int(value) for value in configured.checkpoints)
    checkpoints = tuple(value for value in raw if 0 < value <= max_trials)
    if not checkpoints:
        raise ValueError("validation-budget checkpoints在max_trials內沒有合法值")
    if checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError("validation-budget checkpoints必須嚴格遞增且不可重複")
    if checkpoints[-1] != max_trials:
        raise ValueError(
            "validation-budget最後checkpoint必須等於fold最大trial budget: "
            f"last={checkpoints[-1]}, max={max_trials}"
        )
    if configured.selection_policy != "base_finalists_agree":
        raise ValueError(
            "validation-budget refit capability目前要求base_finalists_agree selector，"
            f"actual={configured.selection_policy!r}"
        )
    return ValidationBudgetSpec(
        protocol=configured.protocol_id,
        holdout_months=int(configured.holdout_months),
        checkpoints=checkpoints,
        policy_name=configured.selection_policy,
        selection_metric=configured.selection_metric,
        tie_break=configured.tie_break,
        oos_feedback_used=configured.oos_feedback_used,
        full_refit_required=configured.full_refit_required,
    )


def build_inner_validation_policy(
    full_fold_policy: dict[str, Any],
    *,
    selection_start_date: str,
    selection_end_date: str,
    spec: ValidationBudgetSpec,
) -> tuple[dict[str, Any], str, str]:
    """Split a selection window into inner-train + trailing validation months."""

    selection_start = pd.Timestamp(selection_start_date).normalize()
    selection_end = pd.Timestamp(selection_end_date).normalize()
    validation_start = (
        selection_end - pd.DateOffset(months=int(spec.holdout_months)) + pd.Timedelta(days=1)
    ).normalize()
    inner_train_end = (validation_start - pd.Timedelta(days=1)).normalize()
    if validation_start <= selection_start or inner_train_end < selection_start:
        raise ValueError(
            "validation-budget holdout切分後沒有合法inner-train期間: "
            f"selection={selection_start.date()}~{selection_end.date()}, "
            f"holdout={spec.holdout_months}m"
        )

    policy = dict(full_fold_policy)
    policy["selection_start_year"] = int(selection_start.year)
    policy["train_start_year"] = int(selection_start.year)
    policy["search_train_end_year"] = int(inner_train_end.year)
    policy["oos_start_year"] = int(validation_start.year)
    policy["selection_start_date"] = selection_start.strftime("%Y-%m-%d")
    policy["train_start_date"] = selection_start.strftime("%Y-%m-%d")
    policy["search_train_end_date"] = inner_train_end.strftime("%Y-%m-%d")
    policy["oos_start_date"] = validation_start.strftime("%Y-%m-%d")
    policy["oos_end_date"] = selection_end.strftime("%Y-%m-%d")
    policy["optimizer_search_protocol"] = f"{spec.protocol}:inner_train"
    policy["validation_budget_refit"] = spec.as_dict()
    return (
        policy,
        validation_start.strftime("%Y-%m-%d"),
        selection_end.strftime("%Y-%m-%d"),
    )


def select_trial_budget_from_validation(
    *,
    study,
    session,
    objective_mode: str,
    spec: ValidationBudgetSpec,
    validation_start_date: str,
    validation_end_date: str,
) -> dict[str, Any]:
    """Choose N* from prefixed trials using validation-only portfolio score.

    Each checkpoint sees only the first N trials.  The formal policy remains
    base-finalists-agree.  Ties are deterministically resolved toward the smaller
    budget to avoid paying extra search complexity without validation evidence.
    """

    trials = list(getattr(study, "trials", []) or [])
    if len(trials) < int(spec.checkpoints[-1]):
        raise RuntimeError(
            "validation-budget inner search未完成最大checkpoint: "
            f"completed={len(trials)}, required={spec.checkpoints[-1]}"
        )

    records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    original_trials = int(getattr(session, "n_trials", spec.checkpoints[-1]) or spec.checkpoints[-1])
    try:
        for checkpoint in spec.checkpoints:
            prefix = _StudyTrialPrefix(study, int(checkpoint))
            session.n_trials = int(checkpoint)
            finalists = list_local_min_score_finalists(
                prefix,
                session=session,
                objective_mode=str(objective_mode),
                include_trial=None,
                show_progress=False,
                include_oos_diagnostics=False,
                single_finalist_fast_path=True,
                selection_pruning=True,
            )
            policy_items = _build_policy_items(finalists, objective_mode=str(objective_mode))
            item = policy_items.get(spec.policy_name)
            if item is None:
                record = {
                    "trials": int(checkpoint),
                    "available": False,
                    "validation_score": None,
                    "plain_romd_score": None,
                    "return_pct": None,
                    "mdd_pct": None,
                    "member_count": 0,
                }
            else:
                with session.optimizer_runtime_context():
                    metrics = _evaluate_finalist_ensemble_oos_metrics(
                        session=session,
                        item=item,
                        policy_name=spec.policy_name,
                        oos_year=int(pd.Timestamp(validation_start_date).year),
                        oos_start_date=str(validation_start_date),
                        oos_end_date=str(validation_end_date),
                    )
                score = float(metrics.get("score", float("nan")))
                available = bool(math.isfinite(score))
                record = {
                    "trials": int(checkpoint),
                    "available": available,
                    "validation_score": score if available else None,
                    "plain_romd_score": (
                        float(metrics.get("plain_romd_score"))
                        if available and metrics.get("plain_romd_score") is not None
                        else None
                    ),
                    "return_pct": (
                        float(metrics.get("return_pct"))
                        if available and metrics.get("return_pct") is not None
                        else None
                    ),
                    "mdd_pct": (
                        float(metrics.get("mdd_pct"))
                        if available and metrics.get("mdd_pct") is not None
                        else None
                    ),
                    "member_count": int(len(list(item.get("params_ensemble") or []))),
                }
            records.append(record)
            if not record["available"]:
                continue
            if selected is None:
                selected = record
                continue
            current_score = float(record["validation_score"])
            selected_score = float(selected["validation_score"])
            if current_score > selected_score + 1e-12:
                selected = record
            elif abs(current_score - selected_score) <= 1e-12 and int(record["trials"]) < int(selected["trials"]):
                selected = record
    finally:
        session.n_trials = int(original_trials)

    if selected is None:
        raise RuntimeError("validation-budget所有checkpoint都沒有可用validation score")

    return {
        **spec.as_dict(),
        "validation_start_date": str(validation_start_date),
        "validation_end_date": str(validation_end_date),
        "selected_trials": int(selected["trials"]),
        "selected_validation_score": float(selected["validation_score"]),
        "checkpoint_results": records,
    }


def validation_budget_protocol_enabled(policy: dict[str, Any] | None) -> bool:
    protocol = get_strategy_param_validation_protocol("validation_budget_refit_v1")
    return str(dict(policy or {}).get("optimizer_search_protocol") or "").strip() == str(
        protocol.protocol_id
    )


__all__ = [
    "ValidationBudgetSpec",
    "build_inner_validation_policy",
    "resolve_validation_budget_spec",
    "select_trial_budget_from_validation",
    "validation_budget_protocol_enabled",
]
