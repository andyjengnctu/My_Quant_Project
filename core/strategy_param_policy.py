"""Typed resolution for Research strategy-parameter producer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from config.strategy_params import (
    ACTIVE_STRATEGY_PARAM_PROFILE,
    STRATEGY_PARAM_PROFILES,
    STRATEGY_PARAM_VALIDATION_PROTOCOLS,
)


@dataclass(frozen=True)
class StrategyParamValidationProtocol:
    protocol_id: str
    holdout_months: int
    checkpoints: tuple[int, ...]
    selection_policy: str
    selection_metric: str
    tie_break: str
    oos_feedback_used: bool
    full_refit_required: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol_id,
            "holdout_months": int(self.holdout_months),
            "checkpoints": [int(value) for value in self.checkpoints],
            "policy_name": self.selection_policy,
            "selection_metric": self.selection_metric,
            "tie_break": self.tie_break,
            "oos_feedback_used": bool(self.oos_feedback_used),
            "full_refit_required": bool(self.full_refit_required),
        }


@dataclass(frozen=True)
class StrategyParamTrainingProfile:
    profile_id: str
    label: str
    strategy_id: str
    filter_profile: str
    parameter_policy: str
    parameter_set_identity: str
    dataset: str
    schedule_start_date: str
    coverage_end_date: str
    validation_enabled: bool
    validation_protocol: str | None


def get_strategy_param_validation_protocol(protocol_id: str) -> StrategyParamValidationProtocol:
    key = str(protocol_id or "").strip()
    raw = STRATEGY_PARAM_VALIDATION_PROTOCOLS.get(key)
    if not key or not isinstance(raw, Mapping):
        raise ValueError(f"未登記strategy parameter validation protocol: {key or '(blank)'}")
    checkpoints = tuple(int(value) for value in tuple(raw.get("checkpoints") or ()))
    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError(f"validation protocol checkpoints必須嚴格遞增且不可重複: {key}")
    holdout = int(raw.get("holdout_months") or 0)
    if holdout <= 0:
        raise ValueError(f"validation protocol holdout_months必須>0: {key}")
    selection_policy = str(raw.get("selection_policy") or "").strip()
    selection_metric = str(raw.get("selection_metric") or "").strip()
    tie_break = str(raw.get("tie_break") or "").strip()
    if not selection_policy or not selection_metric or not tie_break:
        raise ValueError(f"validation protocol缺少selection contract: {key}")
    return StrategyParamValidationProtocol(
        protocol_id=key,
        holdout_months=holdout,
        checkpoints=checkpoints,
        selection_policy=selection_policy,
        selection_metric=selection_metric,
        tie_break=tie_break,
        oos_feedback_used=bool(raw.get("oos_feedback_used", False)),
        full_refit_required=bool(raw.get("full_refit_required", True)),
    )


def get_strategy_param_training_profile(profile_id: str) -> StrategyParamTrainingProfile:
    key = str(profile_id or "").strip()
    raw = STRATEGY_PARAM_PROFILES.get(key)
    if not key or not isinstance(raw, Mapping):
        raise ValueError(f"未登記strategy parameter training profile: {key or '(blank)'}")
    values = {
        name: str(raw.get(name) or "").strip()
        for name in (
            "label",
            "strategy_id",
            "filter_profile",
            "parameter_policy",
            "parameter_set_identity",
            "dataset",
            "schedule_start_date",
            "coverage_end_date",
        )
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"STRATEGY_PARAM_PROFILES[{key!r}]缺少欄位: {', '.join(missing)}")
    start = date.fromisoformat(values["schedule_start_date"][:10])
    end = date.fromisoformat(values["coverage_end_date"][:10])
    if end < start:
        raise ValueError(
            f"strategy parameter coverage期間不合法: {start.isoformat()}~{end.isoformat()}"
        )
    validation = dict(raw.get("validation") or {})
    enabled = bool(validation.get("enabled", False))
    protocol = str(validation.get("protocol") or "").strip() or None
    if enabled:
        if protocol is None:
            raise ValueError(f"validation已啟用但未指定protocol: {key}")
        get_strategy_param_validation_protocol(protocol)
    elif protocol is not None:
        raise ValueError(f"validation關閉時不得指定protocol: {key}")
    return StrategyParamTrainingProfile(
        profile_id=key,
        label=values["label"],
        strategy_id=values["strategy_id"],
        filter_profile=values["filter_profile"],
        parameter_policy=values["parameter_policy"],
        parameter_set_identity=values["parameter_set_identity"],
        dataset=values["dataset"],
        schedule_start_date=start.isoformat(),
        coverage_end_date=end.isoformat(),
        validation_enabled=enabled,
        validation_protocol=protocol,
    )


def get_active_strategy_param_training_profile() -> StrategyParamTrainingProfile:
    return get_strategy_param_training_profile(ACTIVE_STRATEGY_PARAM_PROFILE)


__all__ = [
    "StrategyParamTrainingProfile",
    "StrategyParamValidationProtocol",
    "get_active_strategy_param_training_profile",
    "get_strategy_param_training_profile",
    "get_strategy_param_validation_protocol",
]
