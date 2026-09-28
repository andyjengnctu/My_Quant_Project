"""Typed resolution for Research strategy-parameter producer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from config.strategy_params import (
    ACTIVE_STRATEGY_PARAM_PROFILE,
    STRATEGY_PARAM_FITTING_SEED_PROTOCOLS,
    STRATEGY_PARAM_PROFILES,
    STRATEGY_PARAM_VALIDATION_PROTOCOLS,
)
from core.seed_ensemble_policy import (
    generate_deterministic_seed_ensemble,
    resolve_seed_ensemble_min_agree,
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
    fitting_seed_protocol: str


@dataclass(frozen=True)
class StrategyParamFittingSeedProtocol:
    protocol_id: str
    enabled: bool
    seed_count: int
    seed_mode: str
    generator_seed: int
    min_agree: int
    min_agree_requested: str | int
    trials_per_seed: int
    resolved_seeds: tuple[int, ...]

    def as_seed_ensemble_policy(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "seed_count": int(self.seed_count),
            "min_agree": int(self.min_agree),
            "min_agree_requested": self.min_agree_requested,
            "min_agree_max": int(self.seed_count),
            "min_agree_rule": "clamp_1_to_seed_count",
            "seed_mode": str(self.seed_mode),
            "params_storage": "same_json",
            "generator_seed": int(self.generator_seed),
            "resolved_seeds": [int(value) for value in self.resolved_seeds],
        }


def get_strategy_param_fitting_seed_protocol(
    protocol_id: str,
) -> StrategyParamFittingSeedProtocol:
    key = str(protocol_id or "").strip()
    raw = STRATEGY_PARAM_FITTING_SEED_PROTOCOLS.get(key)
    if not key or not isinstance(raw, Mapping):
        raise ValueError(f"未登記strategy parameter fitting seed protocol: {key or '(blank)'}")
    enabled = bool(raw.get("enabled", False))
    seed_count = int(raw.get("seed_count") or 0)
    generator_seed = int(raw.get("generator_seed") or 0)
    trials_per_seed = int(raw.get("trials_per_seed") or 0)
    seed_mode = str(raw.get("seed_mode") or "").strip()
    min_agree_requested = raw.get("min_agree", "auto")
    if seed_count < 1 or trials_per_seed < 1 or not seed_mode:
        raise ValueError(f"fitting seed protocol contract不完整: {key}")
    if seed_mode == "fixed_single_seed":
        if enabled or seed_count != 1:
            raise ValueError(f"fixed_single_seed protocol必須是disabled且seed_count=1: {key}")
        resolved = (int(generator_seed),)
    elif seed_mode == "fixed_generated_bundle":
        if not enabled or seed_count < 2:
            raise ValueError(f"fixed_generated_bundle protocol必須enabled且seed_count>=2: {key}")
        resolved = tuple(
            int(value)
            for value in generate_deterministic_seed_ensemble(
                seed_count, generator_seed=generator_seed
            )
        )
    else:
        raise ValueError(f"不支援的fitting seed_mode: {seed_mode!r} ({key})")
    return StrategyParamFittingSeedProtocol(
        protocol_id=key,
        enabled=enabled,
        seed_count=seed_count,
        seed_mode=seed_mode,
        generator_seed=generator_seed,
        min_agree=resolve_seed_ensemble_min_agree(seed_count, min_agree_requested),
        min_agree_requested=min_agree_requested,
        trials_per_seed=trials_per_seed,
        resolved_seeds=resolved,
    )



def build_replicated_fitting_seed_ensemble_policy(
    protocol_id: str, *, replication_seed: int
) -> dict[str, Any]:
    """Resolve one deterministic fitting-seed bundle for an outer robustness replicate.

    ``replication_seed`` belongs to the robustness benchmark layer.  It is used only
    as the deterministic bundle generator seed; the optimizer members inside the
    fitting procedure remain the resolved multi-seed bundle.  This keeps fitting
    stochasticity orthogonal to robustness replication stochasticity.
    """

    protocol = get_strategy_param_fitting_seed_protocol(protocol_id)
    if not protocol.enabled or protocol.seed_count < 2:
        raise ValueError(
            "replicated fitting-seed bundle只支援enabled multi-seed protocol: "
            f"{protocol_id}"
        )
    seed0 = int(replication_seed)
    if seed0 <= 0:
        raise ValueError("robustness replication seed必須>0")
    resolved = tuple(
        int(value)
        for value in generate_deterministic_seed_ensemble(
            protocol.seed_count, generator_seed=seed0
        )
    )
    return {
        "enabled": True,
        "seed_count": int(protocol.seed_count),
        "min_agree": int(protocol.min_agree),
        "min_agree_requested": protocol.min_agree_requested,
        "min_agree_max": int(protocol.seed_count),
        "min_agree_rule": "clamp_1_to_seed_count",
        "seed_mode": "fixed_generated_bundle",
        "params_storage": "same_json",
        "generator_seed": int(seed0),
        "resolved_seeds": [int(value) for value in resolved],
    }

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
    fitting_seed_protocol = str(raw.get("fitting_seed_protocol") or "").strip()
    if not fitting_seed_protocol:
        raise ValueError(f"strategy parameter profile缺少fitting_seed_protocol: {key}")
    get_strategy_param_fitting_seed_protocol(fitting_seed_protocol)
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
        fitting_seed_protocol=fitting_seed_protocol,
    )


def get_active_strategy_param_training_profile() -> StrategyParamTrainingProfile:
    return get_strategy_param_training_profile(ACTIVE_STRATEGY_PARAM_PROFILE)


__all__ = [
    "StrategyParamTrainingProfile",
    "StrategyParamFittingSeedProtocol",
    "StrategyParamValidationProtocol",
    "get_strategy_param_fitting_seed_protocol",
    "get_active_strategy_param_training_profile",
    "get_strategy_param_training_profile",
    "get_strategy_param_validation_protocol",
]
