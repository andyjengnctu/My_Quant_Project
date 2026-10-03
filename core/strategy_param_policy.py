"""Typed resolution for Research strategy-parameter producer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any, Mapping

from config.strategy_params import (
    ACTIVE_STRATEGY_PARAM_PROFILE,
    STRATEGY_PARAM_CANONICAL_PROFILE_BY_FAMILY_POLICY,
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
    mode: str
    holdout_months: int
    checkpoints: tuple[int, ...]
    warmup_trials: int | None
    observation_start_trial: int | None
    min_trials_before_stop: int | None
    patience_trials: int | None
    observation_interval: int | None
    min_delta: float | None
    selection_policy: str
    selection_metric: str
    tie_break: str
    oos_feedback_used: bool
    full_refit_required: bool
    reuse_same_policy_signature: bool

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": self.protocol_id,
            "holdout_months": int(self.holdout_months),
        }
        if self.mode == "fixed_checkpoints":
            # Preserve the registered PARAM-P5/C84 v1 scientific payload exactly.
            payload["checkpoints"] = [int(value) for value in self.checkpoints]
        elif self.mode == "adaptive_plateau":
            adaptive_payload = {
                "patience_trials": int(self.patience_trials or 0),
                "observation_interval": int(self.observation_interval or 0),
                "min_delta": float(self.min_delta or 0.0),
                "reuse_same_policy_signature": bool(self.reuse_same_policy_signature),
            }
            if self.observation_start_trial is not None and self.min_trials_before_stop is not None:
                adaptive_payload.update(
                    {
                        "observation_start_trial": int(self.observation_start_trial),
                        "min_trials_before_stop": int(self.min_trials_before_stop),
                    }
                )
            else:
                # Preserve legacy v2 payload exactly so old partial artifacts remain
                # identifiable and cannot be mistaken for the corrected v3 contract.
                adaptive_payload["warmup_trials"] = int(self.warmup_trials or 0)
            payload.update(adaptive_payload)
        else:
            raise ValueError(f"不支援的validation protocol mode: {self.mode!r}")
        payload.update(
            {
                "policy_name": self.selection_policy,
                "selection_metric": self.selection_metric,
                "tie_break": self.tie_break,
                "oos_feedback_used": bool(self.oos_feedback_used),
                "full_refit_required": bool(self.full_refit_required),
            }
        )
        return payload


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
    artifact_slug: str | None
    registered_scientific_fingerprint_sha256: str | None


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
    holdout = int(raw.get("holdout_months") or 0)
    if holdout <= 0:
        raise ValueError(f"validation protocol holdout_months必須>0: {key}")

    raw_checkpoints = tuple(raw.get("checkpoints") or ())
    if raw_checkpoints:
        mode = "fixed_checkpoints"
        checkpoints = tuple(int(value) for value in raw_checkpoints)
        if checkpoints != tuple(sorted(set(checkpoints))) or any(value <= 0 for value in checkpoints):
            raise ValueError(f"validation protocol checkpoints必須嚴格遞增且>0: {key}")
        warmup_trials = None
        observation_start_trial = None
        min_trials_before_stop = None
        patience_trials = None
        observation_interval = None
        min_delta = None
        reuse_same_policy_signature = False
    else:
        mode = "adaptive_plateau"
        checkpoints = ()
        patience_trials = int(raw.get("patience_trials") or 0)
        observation_interval = int(raw.get("observation_interval") or 0)
        min_delta = float(raw.get("min_delta") or 0.0)
        reuse_same_policy_signature = bool(raw.get("reuse_same_policy_signature", True))
        explicit_start = "observation_start_trial" in raw or "min_trials_before_stop" in raw
        if explicit_start:
            warmup_trials = None
            observation_start_trial = int(raw.get("observation_start_trial") or 0)
            min_trials_before_stop = int(raw.get("min_trials_before_stop") or 0)
            if observation_start_trial <= 0 or min_trials_before_stop <= 0:
                raise ValueError(
                    f"adaptive validation observation_start/min_trials_before_stop必須>0: {key}"
                )
            if min_trials_before_stop < observation_start_trial:
                raise ValueError(
                    f"adaptive validation min_trials_before_stop不得早於observation_start_trial: {key}"
                )
        else:
            # Legacy v2 semantics: warmup controlled both observation start and
            # stop eligibility.  Current P9 no longer uses this branch.
            warmup_trials = int(raw.get("warmup_trials") or 0)
            observation_start_trial = None
            min_trials_before_stop = None
            if warmup_trials <= 0:
                raise ValueError(f"adaptive validation protocol warmup_trials必須>0: {key}")
        if patience_trials <= 0 or observation_interval <= 0:
            raise ValueError(
                f"adaptive validation protocol patience/observation_interval必須>0: {key}"
            )
        if min_delta < 0.0:
            raise ValueError(f"adaptive validation protocol min_delta不得<0: {key}")

    selection_policy = str(raw.get("selection_policy") or "").strip()
    selection_metric = str(raw.get("selection_metric") or "").strip()
    tie_break = str(raw.get("tie_break") or "").strip()
    if not selection_policy or not selection_metric or not tie_break:
        raise ValueError(f"validation protocol缺少selection contract: {key}")
    return StrategyParamValidationProtocol(
        protocol_id=key,
        mode=mode,
        holdout_months=holdout,
        checkpoints=checkpoints,
        warmup_trials=warmup_trials,
        observation_start_trial=observation_start_trial,
        min_trials_before_stop=min_trials_before_stop,
        patience_trials=patience_trials,
        observation_interval=observation_interval,
        min_delta=min_delta,
        selection_policy=selection_policy,
        selection_metric=selection_metric,
        tie_break=tie_break,
        oos_feedback_used=bool(raw.get("oos_feedback_used", False)),
        full_refit_required=bool(raw.get("full_refit_required", True)),
        reuse_same_policy_signature=reuse_same_policy_signature,
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
    artifact_slug = str(raw.get("artifact_slug") or "").strip() or None
    if artifact_slug is not None:
        if Path(artifact_slug).name != artifact_slug or artifact_slug in {".", ".."}:
            raise ValueError(f"strategy parameter artifact_slug必須是安全單一資料夾名稱: {key}")
    registered_scientific_fingerprint_sha256 = str(
        raw.get("registered_scientific_fingerprint_sha256") or ""
    ).strip().lower() or None
    if (
        registered_scientific_fingerprint_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", registered_scientific_fingerprint_sha256) is None
    ):
        raise ValueError(
            "registered_scientific_fingerprint_sha256必須是64字元小寫SHA256: "
            f"{key}"
        )
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
        artifact_slug=artifact_slug,
        registered_scientific_fingerprint_sha256=registered_scientific_fingerprint_sha256,
    )


def resolve_validation_strategy_param_training_profile(
    parameter_set_identity: str, validation_protocol_id: str
) -> StrategyParamTrainingProfile:
    """Resolve the unique declarative profile owning a validation publication identity."""

    parameter_set = str(parameter_set_identity or "").strip()
    protocol_id = str(validation_protocol_id or "").strip()
    matches: list[StrategyParamTrainingProfile] = []
    for profile_id in STRATEGY_PARAM_PROFILES:
        profile = get_strategy_param_training_profile(str(profile_id))
        if (
            profile.validation_enabled
            and profile.parameter_set_identity == parameter_set
            and profile.validation_protocol == protocol_id
        ):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError(
            "validation strategy parameter publication identity必須唯一對應profile: "
            f"parameter_set={parameter_set or '(blank)'}, "
            f"protocol={protocol_id or '(blank)'}, matches={[item.profile_id for item in matches]}"
        )
    profile = matches[0]
    if not profile.artifact_slug:
        raise ValueError(f"validation strategy parameter profile缺少artifact_slug: {profile.profile_id}")
    return profile


def resolve_canonical_strategy_param_training_profile(
    family: str, parameter_policy: str
) -> StrategyParamTrainingProfile | None:
    """Resolve the registered canonical single-seed profile for one policy output.

    Canonical Min/Full schedules may emit multiple policy outputs from the same
    physical Optimizer study.  The profile is therefore keyed by
    ``(family, parameter_policy)`` rather than by artifact path.  The mapping is
    declarative config; callers must not infer C-arm identity or duplicate it.
    """

    normalized_family = str(family or "").strip().lower()
    raw_policy = str(parameter_policy or "").strip()
    compare_policy = {
        "base_finalist_best": "base-finalist-best",
        "base_finalists_agree": "base-finalists-agree",
    }.get(raw_policy, raw_policy)
    profile_id = STRATEGY_PARAM_CANONICAL_PROFILE_BY_FAMILY_POLICY.get(
        (normalized_family, compare_policy)
    )
    if profile_id is None:
        return None
    profile = get_strategy_param_training_profile(str(profile_id))
    if profile.filter_profile != normalized_family or profile.parameter_policy != compare_policy:
        raise ValueError(
            "canonical strategy parameter profile mapping與profile內容不一致: "
            f"family={normalized_family}, policy={compare_policy}, profile={profile.profile_id}"
        )
    if profile.validation_enabled:
        raise ValueError(
            "canonical non-validation strategy parameter profile不得啟用validation: "
            f"{profile.profile_id}"
        )
    return profile


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
    "resolve_validation_strategy_param_training_profile",
    "resolve_canonical_strategy_param_training_profile",
]
