"""Canonical scientific contract for Strategy Parameter fitting.

Round 1 introduces this resolver as an observation-only boundary.  Existing
Optimizer runtime, artifacts and legacy fingerprints do not consume it yet.
Later migration rounds will inject the resolved payload once at the producer
boundary and bind named parameter artifacts/experiments to its fingerprint.
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

    scientific = {
        "schema_version": STRATEGY_PARAM_SCIENTIFIC_CONTRACT_SCHEMA_VERSION,
        "strategy": {
            "strategy_id": str(profile.strategy_id),
            "filter_profile": str(profile.filter_profile),
            "parameter_policy": str(profile.parameter_policy),
        },
        "dataset": {
            "profile": str(profile.dataset),
            "generation_identity": build_dataset_generation_identity(root, profile.dataset),
        },
        "schedule": {
            "schedule_start_date": str(profile.schedule_start_date),
            "coverage_end_date": str(profile.coverage_end_date),
            "train_window_months": int(training["train_window_months"]),
            "oos_horizon_months": int(training["oos_horizon_months"]),
        },
        "parameter_generation": _build_parameter_generation_contract(profile.strategy_id),
        "score": build_optimizer_scientific_score_policy_snapshot(),
        "hard_gates": build_training_threshold_snapshot(),
        "selection_and_anti_overfit": {
            "parameter_policy": str(profile.parameter_policy),
            **build_optimizer_scientific_selection_policy_snapshot(),
        },
        "execution": _build_execution_scientific_contract(execution_overrides),
        "fitting": fitting,
        "validation": validation,
    }
    scientific = _jsonable(scientific)
    references = {
        "profile_id": str(profile.profile_id),
        "parameter_set_identity": str(profile.parameter_set_identity),
        **fitting_refs,
        **validation_refs,
    }
    return {
        "references": references,
        "scientific_contract": scientific,
        "scientific_fingerprint_sha256": canonical_json_sha256(scientific),
    }


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


__all__ = [
    "STRATEGY_PARAM_SCIENTIFIC_CONTRACT_SCHEMA_VERSION",
    "build_strategy_param_scientific_contract",
    "build_strategy_param_scientific_fingerprint",
]
