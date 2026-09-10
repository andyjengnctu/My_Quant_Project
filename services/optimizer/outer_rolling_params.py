from __future__ import annotations

from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT
from core.config import V16StrategyParams
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.seed_ensemble_policy import renumber_seed_ensemble_members
from services.optimizer.study_utils import build_best_params_payload_from_trial


def materialize_fixed_strategy_param_overrides_in_payload(
    params_payload: dict | None,
    fixed_strategy_param_overrides: dict | None,
) -> dict:
    """Return a full validated strategy payload with fixed runtime fields applied."""
    payload = params_to_json_dict(V16StrategyParams())
    payload.update(dict(params_payload or {}))
    payload.update(dict(fixed_strategy_param_overrides or {}))
    return params_to_json_dict(build_params_from_mapping(payload))


def build_effective_trial_params_payload(*, session, trial) -> dict:
    """Rebuild the exact params used by the optimizer objective and runtime."""
    return build_best_params_payload_from_trial(
        trial,
        fixed_tp_percent=getattr(
            session,
            "optimizer_fixed_tp_percent",
            OPTIMIZER_FIXED_TP_PERCENT,
        ),
        fixed_strategy_param_overrides=getattr(
            session,
            "fixed_strategy_param_overrides",
            None,
        ),
    )


def materialize_fixed_strategy_param_overrides_in_active_param_payload(
    payload: dict,
    fixed_strategy_param_overrides: dict | None,
) -> dict:
    """Apply one fixed-runtime contract to every params node in an active-param artifact."""
    resolved = dict(payload or {})
    overrides = dict(fixed_strategy_param_overrides or {})
    if not overrides:
        return resolved

    for field_name in ("params_by_oos_year", "params_by_effective_date"):
        mapping = dict(resolved.get(field_name) or {})
        if mapping:
            resolved[field_name] = {
                str(key): materialize_fixed_strategy_param_overrides_in_payload(
                    dict(value or {}),
                    overrides,
                )
                for key, value in mapping.items()
            }

    for field_name in ("params_ensemble_by_effective_date",):
        mapping = dict(resolved.get(field_name) or {})
        if not mapping:
            continue
        normalized_mapping = {}
        for key, raw_members in mapping.items():
            members = []
            for raw_member in list(raw_members or []):
                member = dict(raw_member or {})
                member["params"] = materialize_fixed_strategy_param_overrides_in_payload(
                    dict(member.get("params") or {}),
                    overrides,
                )
                members.append(member)
            normalized_mapping[str(key)] = members
        resolved[field_name] = normalized_mapping

    if isinstance(resolved.get("params_ensemble"), list):
        members = []
        for raw_member in list(resolved.get("params_ensemble") or []):
            member = dict(raw_member or {})
            member["params"] = materialize_fixed_strategy_param_overrides_in_payload(
                dict(member.get("params") or {}),
                overrides,
            )
            members.append(member)
        resolved["params_ensemble"] = members

    return resolved


def _materialize_fixed_strategy_param_overrides_in_members(
    raw_members,
    fixed_strategy_param_overrides: dict | None,
) -> list[dict]:
    members = renumber_seed_ensemble_members(raw_members)
    overrides = dict(fixed_strategy_param_overrides or {})
    if not overrides:
        return members
    normalized = []
    for raw_member in members:
        member = dict(raw_member)
        member["params"] = materialize_fixed_strategy_param_overrides_in_payload(
            dict(member.get("params") or {}),
            overrides,
        )
        normalized.append(member)
    return renumber_seed_ensemble_members(normalized)
