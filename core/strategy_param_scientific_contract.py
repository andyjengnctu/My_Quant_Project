"""Canonical scientific contract for Strategy Parameter fitting.

The contract has two layers:

* :func:`build_strategy_param_runtime_scientific_contract` resolves the actual
  semantics of one Optimizer fitting invocation from explicit producer inputs.
* :func:`build_strategy_param_scientific_contract` resolves a registered
  Strategy Parameter profile into the same runtime contract and adds stable
  human/artifact references.

Artifact identity migration is intentionally separate.  Runtime consumers may
use the resolved scientific payload without changing legacy artifact keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import config.execution_policy as _execution_settings
import strategies.breakout.schema as _breakout_schema
import strategies.breakout.search_space as _breakout_search
from core.dataset_profiles import build_dataset_generation_identity
from core.execution_policy import build_execution_policy_snapshot
from core.file_integrity import canonical_json_sha256
from core.selection_policy import SELECTION_POLICY_PARAM_SPECS
from core.strategy_param_policy import (
    get_strategy_param_fitting_seed_protocol,
    get_strategy_param_training_profile,
    get_strategy_param_validation_protocol,
)
from core.training_policy import (
    build_optimizer_scientific_score_policy_snapshot,
    build_optimizer_scientific_selection_policy_snapshot,
    build_training_threshold_snapshot,
    get_strategy_parameter_training_policy_snapshot,
)

STRATEGY_PARAM_SCIENTIFIC_CONTRACT_SCHEMA_VERSION = 1
STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _normalize_spec_map(specs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(field_name): {str(key): _jsonable(value) for key, value in dict(spec).items()}
        for field_name, spec in specs.items()
    }


def _build_breakout_parameter_generation_contract() -> dict[str, Any]:
    return {
        "strategy_id": "breakout",
        "search_space": _jsonable(_breakout_search.BREAKOUT_OPTIMIZER_SEARCH_SPACE),
        "inactive_param_values": _jsonable(
            _breakout_search.BREAKOUT_OPTIMIZER_INACTIVE_PARAM_VALUES
        ),
        "breakout_param_schema": _normalize_spec_map(_breakout_schema.BREAKOUT_PARAM_SPECS),
        "selection_param_schema": _normalize_spec_map(SELECTION_POLICY_PARAM_SPECS),
    }


def _build_parameter_generation_contract(strategy_id: str) -> dict[str, Any]:
    key = str(strategy_id or "").strip().lower()
    if key == "breakout":
        return _build_breakout_parameter_generation_contract()
    raise ValueError(f"尚未登記strategy parameter scientific contract builder: {strategy_id!r}")


def _build_execution_scientific_contract(
    execution_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(build_execution_policy_snapshot())
    payload["max_positions"] = int(_execution_settings.DEFAULT_PORTFOLIO_MAX_POSITIONS)
    payload["rotation"] = str(_execution_settings.DEFAULT_PORTFOLIO_ROTATION)
    for key, value in dict(execution_overrides or {}).items():
        if key not in payload:
            raise ValueError(f"未知scientific execution override: {key}")
        payload[key] = value
    return _jsonable(payload)


def _build_fitting_contract(profile) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    scientific = dict(protocol.as_seed_ensemble_policy())
    scientific["trials_per_seed"] = int(protocol.trials_per_seed)
    references = {"fitting_seed_protocol": str(protocol.protocol_id)}
    return _jsonable(scientific), references


def _build_validation_contract(profile) -> tuple[dict[str, Any], dict[str, Any]]:
    if not profile.validation_enabled:
        return {"enabled": False}, {"validation_protocol": None}
    protocol = get_strategy_param_validation_protocol(str(profile.validation_protocol))
    scientific = {"enabled": True, **dict(protocol.as_dict())}
    scientific.pop("protocol", None)
    return _jsonable(scientific), {"validation_protocol": str(protocol.protocol_id)}


def _wrap_scientific_contract(
    scientific: Mapping[str, Any],
    *,
    references: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _jsonable(dict(scientific))
    return {
        "references": _jsonable(dict(references or {})),
        "scientific_contract": normalized,
        "scientific_fingerprint_sha256": canonical_json_sha256(normalized),
    }


def build_strategy_param_runtime_scientific_contract(
    project_root: str | Path,
    *,
    strategy_id: str,
    filter_profile: str,
    parameter_policy: str,
    dataset_profile: str,
    schedule_start_date: str,
    coverage_end_date: str,
    train_window_months: int,
    oos_horizon_months: int,
    fitting: Mapping[str, Any],
    validation: Mapping[str, Any],
    execution_overrides: Mapping[str, Any] | None = None,
    dataset_generation_identity_override: Mapping[str, Any] | None = None,
    references: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one immutable runtime scientific payload from producer inputs.

    This builder is deliberately profile-agnostic.  It is the single boundary
    used by both registered Research profiles and legacy/canonical producer
    calls, preventing inner Optimizer code from re-reading mutable scientific
    globals after a fitting invocation has started.
    """

    root = Path(project_root).resolve()
    dataset_identity = (
        dict(dataset_generation_identity_override)
        if isinstance(dataset_generation_identity_override, Mapping)
        else build_dataset_generation_identity(root, str(dataset_profile))
    )
    scientific = {
        "schema_version": STRATEGY_PARAM_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
        "strategy": {
            "strategy_id": str(strategy_id),
            "filter_profile": str(filter_profile),
            "parameter_policy": str(parameter_policy),
        },
        "dataset": {
            "profile": str(dataset_profile),
            "generation_identity": dataset_identity,
        },
        "schedule": {
            "schedule_start_date": str(schedule_start_date),
            "coverage_end_date": str(coverage_end_date),
            "train_window_months": int(train_window_months),
            "oos_horizon_months": int(oos_horizon_months),
        },
        "parameter_generation": _build_parameter_generation_contract(strategy_id),
        "score": build_optimizer_scientific_score_policy_snapshot(),
        "hard_gates": build_training_threshold_snapshot(),
        "selection_and_anti_overfit": {
            "parameter_policy": str(parameter_policy),
            **build_optimizer_scientific_selection_policy_snapshot(),
        },
        "execution": _build_execution_scientific_contract(execution_overrides),
        "fitting": _jsonable(dict(fitting)),
        "validation": _jsonable(dict(validation)),
    }
    return _wrap_scientific_contract(scientific, references=references)


def build_strategy_param_scientific_contract(
    project_root: str | Path,
    profile_id: str,
    *,
    execution_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one complete Strategy Parameter scientific contract.

    Human/current identifiers are returned as references but deliberately
    excluded from the scientific fingerprint.  The fingerprint is derived from
    resolved semantics, so renaming a profile does not create a scientific
    change while changing any result-affecting value does.

    Round 1 callers must treat this as read-only evidence.  Existing runtime and
    artifact identity stay on their legacy paths until the explicit migration
    rounds.
    """

    root = Path(project_root).resolve()
    profile = get_strategy_param_training_profile(profile_id)
    training = get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling")
    fitting, fitting_refs = _build_fitting_contract(profile)
    validation, validation_refs = _build_validation_contract(profile)

    references = {
        "profile_id": str(profile.profile_id),
        "parameter_set_identity": str(profile.parameter_set_identity),
        **fitting_refs,
        **validation_refs,
    }
    return build_strategy_param_runtime_scientific_contract(
        root,
        strategy_id=str(profile.strategy_id),
        filter_profile=str(profile.filter_profile),
        parameter_policy=str(profile.parameter_policy),
        dataset_profile=str(profile.dataset),
        schedule_start_date=str(profile.schedule_start_date),
        coverage_end_date=str(profile.coverage_end_date),
        train_window_months=int(training["train_window_months"]),
        oos_horizon_months=int(training["oos_horizon_months"]),
        fitting=fitting,
        validation=validation,
        execution_overrides=execution_overrides,
        references=references,
    )


def build_strategy_param_scientific_fingerprint(
    project_root: str | Path,
    profile_id: str,
    *,
    execution_overrides: Mapping[str, Any] | None = None,
) -> str:
    return str(
        build_strategy_param_scientific_contract(
            project_root,
            profile_id,
            execution_overrides=execution_overrides,
        )["scientific_fingerprint_sha256"]
    )


def derive_strategy_param_scientific_identity_from_registered_profile(
    project_root: str | Path,
    profile_id: str,
    *,
    fitting_override: Mapping[str, Any],
    execution_overrides: Mapping[str, Any] | None = None,
    reference_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive one immutable runtime identity from a registered base profile.

    Robustness fitting reuses the registered strategy/search/score/gate/execution
    semantics but intentionally substitutes the benchmark-controlled fitting seed
    or fitting-seed bundle.  The registered profile remains the authorization and
    drift anchor; the derived fingerprint binds the concrete robustness fit.
    """

    base = build_registered_strategy_param_scientific_identity(
        project_root,
        profile_id,
        execution_overrides=execution_overrides,
    )
    scientific = _jsonable(dict(base["scientific_contract"]))
    scientific["fitting"] = _jsonable(dict(fitting_override))
    references = {
        **dict(base.get("references") or {}),
        **_jsonable(dict(reference_overrides or {})),
    }
    derived = _wrap_scientific_contract(scientific, references=references)
    return {
        "schema_version": STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
        "source_profile_id": str(base["profile_id"]),
        "source_registered": bool(base.get("registered", False)),
        "source_registered_scientific_fingerprint_sha256": base.get(
            "registered_scientific_fingerprint_sha256"
        ),
        "source_scientific_fingerprint_sha256": str(
            base["scientific_fingerprint_sha256"]
        ),
        "source_registration_matches": bool(
            base.get("registration_matches", False)
        ),
        "scientific_fingerprint_sha256": str(
            derived["scientific_fingerprint_sha256"]
        ),
        "scientific_contract": dict(derived["scientific_contract"]),
        "references": dict(derived.get("references") or {}),
    }


def build_registered_strategy_param_scientific_identity(
    project_root: str | Path,
    profile_id: str,
    *,
    execution_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and validate one profile's immutable scientific identity.

    Profiles may remain unregistered while they are still mutable research
    configuration.  Once ``registered_scientific_fingerprint_sha256`` is
    present, the resolved semantics must match it exactly.  This is the source
    registration gate; artifact manifests separately bind the same full
    contract/fingerprint to concrete PARAM outputs.
    """

    profile = get_strategy_param_training_profile(profile_id)
    resolved = build_strategy_param_scientific_contract(
        project_root,
        profile_id,
        execution_overrides=execution_overrides,
    )
    actual = str(resolved["scientific_fingerprint_sha256"])
    registered = profile.registered_scientific_fingerprint_sha256
    return {
        "schema_version": STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
        "profile_id": str(profile.profile_id),
        "parameter_set_identity": str(profile.parameter_set_identity),
        "registered": registered is not None,
        "registered_scientific_fingerprint_sha256": registered,
        "scientific_fingerprint_sha256": actual,
        "registration_matches": bool(registered is not None and registered == actual),
        "scientific_contract": dict(resolved["scientific_contract"]),
        "references": dict(resolved.get("references") or {}),
    }


__all__ = [
    "STRATEGY_PARAM_SCIENTIFIC_CONTRACT_SCHEMA_VERSION",
    "STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA_VERSION",
    "build_registered_strategy_param_scientific_identity",
    "derive_strategy_param_scientific_identity_from_registered_profile",
    "build_strategy_param_runtime_scientific_contract",
    "build_strategy_param_scientific_contract",
    "build_strategy_param_scientific_fingerprint",
]
