"""Validation-selected Optimizer search budget with full-selection refit.

The capability has two versioned protocols:
- ``validation_budget_refit_v1`` preserves the registered PARAM-P5 fixed-checkpoint
  control exactly.
- ``validation_plateau_refit_v2`` observes the canonical base-finalists-agree policy
  after a warmup and stops the inner search after a configured validation plateau.

Neither protocol may use OOS feedback.  Validation only chooses the scalar search
budget N*.  The chosen budget is consumed by a fresh Optimizer study over the full
selection window.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.seed_ensemble_policy import normalize_seed_ensemble_members
from core.strategy_param_policy import get_strategy_param_validation_protocol
from services.optimizer.outer_rolling_evaluation import (
    _evaluate_finalist_ensemble_oos_metrics,
)
from services.optimizer.outer_rolling_params import (
    _materialize_fixed_strategy_param_overrides_in_members,
)
from services.optimizer.outer_rolling_policy import _build_policy_items
from services.optimizer.robustness import list_local_min_score_finalists


@dataclass(frozen=True)
class ValidationBudgetSpec:
    protocol: str
    mode: str
    holdout_months: int
    checkpoints: tuple[int, ...]
    warmup_trials: int | None
    patience_trials: int | None
    observation_interval: int | None
    min_delta: float | None
    policy_name: str
    selection_metric: str
    tie_break: str = "smaller_budget"
    oos_feedback_used: bool = False
    full_refit_required: bool = True
    reuse_same_policy_signature: bool = False
    max_trials: int = 0

    @property
    def adaptive(self) -> bool:
        return self.mode == "adaptive_plateau"

    def as_dict(self) -> dict[str, Any]:
        # Keep PARAM-P5/C84 v1 payload byte-semantically unchanged.  New v2 fields
        # appear only for the new scientific identity.
        payload: dict[str, Any] = {
            "protocol": self.protocol,
            "holdout_months": int(self.holdout_months),
        }
        if self.mode == "fixed_checkpoints":
            payload["checkpoints"] = [int(value) for value in self.checkpoints]
        elif self.mode == "adaptive_plateau":
            payload.update(
                {
                    "warmup_trials": int(self.warmup_trials or 0),
                    "patience_trials": int(self.patience_trials or 0),
                    "observation_interval": int(self.observation_interval or 0),
                    "min_delta": float(self.min_delta or 0.0),
                    "reuse_same_policy_signature": bool(
                        self.reuse_same_policy_signature
                    ),
                }
            )
        else:
            raise ValueError(f"不支援的validation budget mode: {self.mode!r}")
        payload.update(
            {
                "policy_name": self.policy_name,
                "selection_metric": self.selection_metric,
                "tie_break": self.tie_break,
                "oos_feedback_used": bool(self.oos_feedback_used),
                "full_refit_required": bool(self.full_refit_required),
            }
        )
        return payload


class _StudyTrialPrefix:
    """Minimal Optuna-study view exposing only the first N trials."""

    def __init__(self, source_study, count: int):
        trials = list(getattr(source_study, "trials", []) or [])
        self.trials = trials[: int(count)]


def resolve_validation_budget_spec(
    max_trials: int, *, protocol_id: str = "validation_budget_refit_v1"
) -> ValidationBudgetSpec:
    max_trials = int(max_trials)
    if max_trials < 1:
        raise ValueError("validation-budget max_trials必須>0")
    configured = get_strategy_param_validation_protocol(protocol_id)
    if configured.selection_policy != "base_finalists_agree":
        raise ValueError(
            "validation-budget refit capability目前要求base_finalists_agree selector，"
            f"actual={configured.selection_policy!r}"
        )

    if configured.mode == "fixed_checkpoints":
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
        warmup_trials = None
        patience_trials = None
        observation_interval = None
        min_delta = None
        reuse_same_policy_signature = False
    elif configured.mode == "adaptive_plateau":
        checkpoints = ()
        warmup_trials = int(configured.warmup_trials or 0)
        patience_trials = int(configured.patience_trials or 0)
        observation_interval = int(configured.observation_interval or 0)
        min_delta = float(configured.min_delta or 0.0)
        reuse_same_policy_signature = bool(configured.reuse_same_policy_signature)
        if warmup_trials > max_trials:
            raise ValueError(
                "adaptive validation warmup不得超過fold最大trial budget: "
                f"warmup={warmup_trials}, max={max_trials}"
            )
        if patience_trials < 1 or observation_interval < 1:
            raise ValueError("adaptive validation patience/observation_interval必須>0")
    else:
        raise ValueError(f"不支援的validation protocol mode: {configured.mode!r}")

    return ValidationBudgetSpec(
        protocol=configured.protocol_id,
        mode=configured.mode,
        holdout_months=int(configured.holdout_months),
        checkpoints=checkpoints,
        warmup_trials=warmup_trials,
        patience_trials=patience_trials,
        observation_interval=observation_interval,
        min_delta=min_delta,
        policy_name=configured.selection_policy,
        selection_metric=configured.selection_metric,
        tie_break=configured.tie_break,
        oos_feedback_used=configured.oos_feedback_used,
        full_refit_required=configured.full_refit_required,
        reuse_same_policy_signature=reuse_same_policy_signature,
        max_trials=max_trials,
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
        selection_end
        - pd.DateOffset(months=int(spec.holdout_months))
        + pd.Timedelta(days=1)
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


def _build_policy_item_for_prefix(
    *,
    study,
    session,
    objective_mode: str,
    spec: ValidationBudgetSpec,
    trial_count: int,
) -> dict[str, Any] | None:
    prefix = _StudyTrialPrefix(study, int(trial_count))
    original_trials = int(getattr(session, "n_trials", spec.max_trials) or spec.max_trials)
    try:
        session.n_trials = int(trial_count)
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
        policy_items = _build_policy_items(
            finalists,
            objective_mode=str(objective_mode),
            session=session,
        )
        item = policy_items.get(spec.policy_name)
        return None if item is None else dict(item)
    finally:
        session.n_trials = int(original_trials)


def _policy_replay_signature(*, session, item: dict[str, Any], spec: ValidationBudgetSpec) -> str:
    members = _materialize_fixed_strategy_param_overrides_in_members(
        item.get("params_ensemble"),
        getattr(session, "fixed_strategy_param_overrides", None),
    )
    normalized = normalize_seed_ensemble_members(members)
    params_payload = [dict(member.get("params") or {}) for member in normalized]
    if not params_payload:
        raise ValueError("validation policy signature缺少可用params_ensemble")
    material = {
        "policy_name": str(spec.policy_name),
        "params_ensemble": params_payload,
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_metrics_record(metrics: dict[str, Any]) -> dict[str, Any]:
    score = float(metrics.get("score", float("nan")))
    available = bool(math.isfinite(score))
    return {
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
    }


def _evaluate_policy_item_on_validation(
    *,
    session,
    item: dict[str, Any],
    spec: ValidationBudgetSpec,
    validation_start_date: str,
    validation_end_date: str,
) -> dict[str, Any]:
    with session.optimizer_runtime_context():
        metrics = _evaluate_finalist_ensemble_oos_metrics(
            session=session,
            item=item,
            policy_name=spec.policy_name,
            oos_year=int(pd.Timestamp(validation_start_date).year),
            oos_start_date=str(validation_start_date),
            oos_end_date=str(validation_end_date),
        )
    record = _validation_metrics_record(metrics)
    record["member_count"] = int(len(list(item.get("params_ensemble") or [])))
    return record


def select_trial_budget_from_validation(
    *,
    study,
    session,
    objective_mode: str,
    spec: ValidationBudgetSpec,
    validation_start_date: str,
    validation_end_date: str,
) -> dict[str, Any]:
    """Choose N* from the registered fixed-checkpoint v1 control."""

    if spec.mode != "fixed_checkpoints":
        raise ValueError(
            "select_trial_budget_from_validation只支援fixed-checkpoint protocol；"
            f"actual={spec.protocol}"
        )
    trials = list(getattr(study, "trials", []) or [])
    if len(trials) < int(spec.checkpoints[-1]):
        raise RuntimeError(
            "validation-budget inner search未完成最大checkpoint: "
            f"completed={len(trials)}, required={spec.checkpoints[-1]}"
        )

    records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for checkpoint in spec.checkpoints:
        item = _build_policy_item_for_prefix(
            study=study,
            session=session,
            objective_mode=objective_mode,
            spec=spec,
            trial_count=int(checkpoint),
        )
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
            record = {
                "trials": int(checkpoint),
                **_evaluate_policy_item_on_validation(
                    session=session,
                    item=item,
                    spec=spec,
                    validation_start_date=validation_start_date,
                    validation_end_date=validation_end_date,
                ),
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
        elif (
            abs(current_score - selected_score) <= 1e-12
            and int(record["trials"]) < int(selected["trials"])
        ):
            selected = record

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


class AdaptiveValidationBudgetController:
    """Optuna callback implementing v2 validation plateau early stop.

    Validation is observational only: it never modifies trial values, sampler state,
    search space, finalists, or TPE suggestions.  It can only stop future inner-train
    trials after the configured plateau.  State is persisted in study user attrs so a
    SQLite resume continues the same patience counter without OOS input.
    """

    _STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        session,
        objective_mode: str,
        spec: ValidationBudgetSpec,
        validation_start_date: str,
        validation_end_date: str,
    ) -> None:
        if not spec.adaptive:
            raise ValueError("AdaptiveValidationBudgetController需要adaptive protocol")
        self.session = session
        self.objective_mode = str(objective_mode)
        self.spec = spec
        self.validation_start_date = str(validation_start_date)
        self.validation_end_date = str(validation_end_date)
        self.state_key = f"strategy_param_validation_state::{spec.protocol}"

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "protocol": self.spec.protocol,
            "last_observed_trials": 0,
            "best_trials": None,
            "best_validation_score": None,
            "last_improvement_trial": None,
            "stop_trial": None,
            "stop_reason": None,
            "actual_validation_replays": 0,
            "reused_validation_observations": 0,
            "validation_wall_sec": 0.0,
            "signature_cache": {},
            "observation_results": [],
        }

    def _load_state(self, study) -> dict[str, Any]:
        raw = dict(getattr(study, "user_attrs", {}) or {}).get(self.state_key)
        if not isinstance(raw, dict):
            return self._empty_state()
        state = dict(raw)
        if (
            int(state.get("schema_version", 0) or 0) != self._STATE_SCHEMA_VERSION
            or str(state.get("protocol") or "") != self.spec.protocol
        ):
            raise RuntimeError("adaptive validation resume state與current protocol不相容")
        state.setdefault("signature_cache", {})
        state.setdefault("observation_results", [])
        return state

    def _save_state(self, study, state: dict[str, Any]) -> None:
        study.set_user_attr(self.state_key, state)

    def _should_observe(self, trial_count: int) -> bool:
        warmup = int(self.spec.warmup_trials or 0)
        interval = int(self.spec.observation_interval or 1)
        count = int(trial_count)
        return count >= warmup and ((count - warmup) % interval == 0)

    def _observe_prefix(self, study, state: dict[str, Any], trial_count: int) -> None:
        count = int(trial_count)
        if count <= int(state.get("last_observed_trials", 0) or 0):
            return
        started = time.perf_counter()
        item = _build_policy_item_for_prefix(
            study=study,
            session=self.session,
            objective_mode=self.objective_mode,
            spec=self.spec,
            trial_count=count,
        )
        record: dict[str, Any] = {
            "trials": count,
            "available": False,
            "validation_score": None,
            "plain_romd_score": None,
            "return_pct": None,
            "mdd_pct": None,
            "member_count": 0,
            "policy_signature": None,
            "validation_source": "UNAVAILABLE",
            "improved": False,
        }
        if item is not None:
            signature = _policy_replay_signature(
                session=self.session,
                item=item,
                spec=self.spec,
            )
            record["policy_signature"] = signature
            cache = dict(state.get("signature_cache") or {})
            cached = cache.get(signature) if self.spec.reuse_same_policy_signature else None
            if isinstance(cached, dict):
                metrics_record = dict(cached)
                record.update(metrics_record)
                record["validation_source"] = "REUSED_SAME_SIGNATURE"
                state["reused_validation_observations"] = int(
                    state.get("reused_validation_observations", 0) or 0
                ) + 1
            else:
                replay_started = time.perf_counter()
                metrics_record = _evaluate_policy_item_on_validation(
                    session=self.session,
                    item=item,
                    spec=self.spec,
                    validation_start_date=self.validation_start_date,
                    validation_end_date=self.validation_end_date,
                )
                replay_sec = max(0.0, time.perf_counter() - replay_started)
                record.update(metrics_record)
                record["validation_source"] = "EVALUATED"
                record["validation_replay_sec"] = float(replay_sec)
                state["actual_validation_replays"] = int(
                    state.get("actual_validation_replays", 0) or 0
                ) + 1
                cache[signature] = dict(metrics_record)
                state["signature_cache"] = cache

        if bool(record.get("available")):
            score = float(record["validation_score"])
            best_raw = state.get("best_validation_score")
            best_score = None if best_raw is None else float(best_raw)
            min_delta = float(self.spec.min_delta or 0.0)
            improved = best_score is None or score > best_score + min_delta + 1e-12
            if improved:
                state["best_validation_score"] = float(score)
                state["best_trials"] = count
                state["last_improvement_trial"] = count
                record["improved"] = True

        observation_sec = max(0.0, time.perf_counter() - started)
        record["observation_wall_sec"] = float(observation_sec)
        state["validation_wall_sec"] = float(
            state.get("validation_wall_sec", 0.0) or 0.0
        ) + float(observation_sec)
        state["last_observed_trials"] = count
        observations = list(state.get("observation_results") or [])
        observations.append(record)
        state["observation_results"] = observations

        last_improvement = state.get("last_improvement_trial")
        patience = int(self.spec.patience_trials or 0)
        if last_improvement is not None and count - int(last_improvement) >= patience:
            state["stop_trial"] = count
            state["stop_reason"] = "validation_plateau"

    def catch_up(self, study) -> dict[str, Any]:
        """Reconcile persisted validation state with already-completed study trials."""

        state = self._load_state(study)
        completed = min(len(list(getattr(study, "trials", []) or [])), int(self.spec.max_trials))
        if state.get("stop_reason"):
            return state
        start = max(1, int(state.get("last_observed_trials", 0) or 0) + 1)
        for count in range(start, completed + 1):
            if self._should_observe(count):
                self._observe_prefix(study, state, count)
                self._save_state(study, state)
                if state.get("stop_reason"):
                    break
        return state

    def callback(self, study, _trial) -> None:
        state = self._load_state(study)
        if state.get("stop_reason"):
            study.stop()
            return
        completed = len(list(getattr(study, "trials", []) or []))
        if self._should_observe(completed):
            self._observe_prefix(study, state, completed)
            self._save_state(study, state)
        if state.get("stop_reason"):
            study.stop()

    def result(self, study) -> dict[str, Any]:
        state = self.catch_up(study)
        completed = min(len(list(getattr(study, "trials", []) or [])), int(self.spec.max_trials))
        if state.get("best_trials") is None or state.get("best_validation_score") is None:
            raise RuntimeError("adaptive validation沒有可用validation score，無法選擇refit budget")
        if not state.get("stop_reason"):
            state["stop_trial"] = completed
            state["stop_reason"] = "max_trials"
            self._save_state(study, state)
        selected_trials = int(state["best_trials"])
        stop_trial = int(state.get("stop_trial") or completed)
        if selected_trials < int(self.spec.warmup_trials or 0) or selected_trials > stop_trial:
            raise RuntimeError(
                "adaptive validation selected budget不合法: "
                f"selected={selected_trials}, stop={stop_trial}"
            )
        return {
            **self.spec.as_dict(),
            "validation_start_date": self.validation_start_date,
            "validation_end_date": self.validation_end_date,
            "selected_trials": selected_trials,
            "selected_validation_score": float(state["best_validation_score"]),
            "stop_trial": stop_trial,
            "stop_reason": str(state.get("stop_reason") or "max_trials"),
            "last_improvement_trial": int(
                state.get("last_improvement_trial") or selected_trials
            ),
            "observation_count": len(list(state.get("observation_results") or [])),
            "actual_validation_replays": int(
                state.get("actual_validation_replays", 0) or 0
            ),
            "reused_validation_observations": int(
                state.get("reused_validation_observations", 0) or 0
            ),
            "validation_wall_sec": float(state.get("validation_wall_sec", 0.0) or 0.0),
            "observation_results": list(state.get("observation_results") or []),
        }


def validation_budget_protocol_enabled(policy: dict[str, Any] | None) -> bool:
    protocol_id = str(dict(policy or {}).get("optimizer_search_protocol") or "").strip()
    if not protocol_id or protocol_id.endswith(":inner_train"):
        return False
    try:
        protocol = get_strategy_param_validation_protocol(protocol_id)
    except ValueError:
        return False
    return bool(protocol.full_refit_required) and protocol.selection_policy == "base_finalists_agree"


__all__ = [
    "AdaptiveValidationBudgetController",
    "ValidationBudgetSpec",
    "build_inner_validation_policy",
    "resolve_validation_budget_spec",
    "select_trial_budget_from_validation",
    "validation_budget_protocol_enabled",
]
