from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from core.active_param_ensemble import (
    build_active_param_ensemble_schedule,
    is_active_param_ensemble_payload,
)
from core.file_integrity import load_json_strict
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.rolling_oos_params import build_active_param_schedule, is_rolling_oos_param_set_payload


def build_portfolio_params_signature(params: Any) -> str:
    payload = params_to_json_dict(params)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_active_param_objects_from_payload(payload: Mapping[str, Any], *, fixed_risk: float | None = None) -> list[dict]:
    schedule = build_active_param_schedule(payload)
    resolved: list[dict] = []
    for record in schedule:
        params = build_params_from_mapping(record["params"])
        if fixed_risk is not None:
            params.fixed_risk = float(fixed_risk)
        item = dict(record)
        item["params_obj"] = params
        item["params_signature"] = build_portfolio_params_signature(params)
        resolved.append(item)
    return resolved


def build_active_param_ensemble_objects_from_payload(payload: Mapping[str, Any], *, fixed_risk: float | None = None) -> list[dict]:
    schedule = build_active_param_ensemble_schedule(payload)
    resolved: list[dict] = []
    for record in schedule:
        item = dict(record)
        members: list[dict] = []
        for member in record.get("members") or []:
            params = build_params_from_mapping(member["params"])
            if fixed_risk is not None:
                params.fixed_risk = float(fixed_risk)
            member_item = dict(member)
            member_item["params_obj"] = params
            member_item["params_signature"] = build_portfolio_params_signature(params)
            member_item["member_key"] = str(member_item.get("member_index") or member_item.get("seed") or len(members) + 1)
            members.append(member_item)
        if not members:
            raise ValueError(f"active-param ensemble 生效日 {record.get('effective_date_text') or '-'} 沒有可用 members")
        item["members"] = members
        item["params_obj"] = members[0]["params_obj"]
        item["params_signature"] = members[0]["params_signature"]
        resolved.append(item)
    return resolved




def build_params_schedule_rows_from_payload(
    payload: Mapping[str, Any],
    *,
    fixed_risk: float | None = None,
    include_single_param: bool = False,
) -> list[dict]:
    """Return dashboard-ready parameter schedule rows from any supported param payload.

    This is the single display adapter for Workbench / portfolio_sim / optimizer final
    reports. It preserves the source semantics:
    - active-param ensemble payloads keep every effective period and every seed/member;
    - rolling active-param payloads keep every effective period;
    - legacy single-param payloads are returned only when explicitly requested.
    """
    if not isinstance(payload, Mapping):
        return []

    if is_active_param_ensemble_payload(payload):
        return build_active_param_ensemble_objects_from_payload(payload, fixed_risk=fixed_risk)

    if is_rolling_oos_param_set_payload(payload):
        if isinstance(payload.get("params_ensemble_by_effective_date"), Mapping) and payload.get("params_ensemble_by_effective_date"):
            return build_active_param_ensemble_objects_from_payload(payload, fixed_risk=fixed_risk)
        return build_active_param_objects_from_payload(payload, fixed_risk=fixed_risk)

    if not include_single_param:
        return []

    params = build_params_from_mapping(payload)
    if fixed_risk is not None:
        params.fixed_risk = float(fixed_risk)
    return [{
        "mode": "single",
        "effective_date_text": "",
        "effective_end_date_text": "",
        "year": "",
        "params": dict(payload),
        "params_obj": params,
        "params_signature": build_portfolio_params_signature(params),
    }]


def load_portfolio_param_source_from_json(json_file: str | os.PathLike[str], *, fixed_risk: float | None = None) -> dict:
    """Load a runtime parameter source for tools that must accept either single params or active-param schedules.

    This is the single runtime loader for formal checks and portfolio replay entry points that
    need to understand both legacy single-param JSON and the active-param ensemble contract.
    `core.params_io.load_params_from_json()` intentionally remains strict and only accepts
    a single param.json.
    """
    path = os.fspath(json_file)
    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"參數檔根層必須是 object/dict，收到 {type(payload).__name__}")

    if is_active_param_ensemble_payload(payload):
        schedule = build_active_param_ensemble_objects_from_payload(payload, fixed_risk=fixed_risk)
        first_member = schedule[0]["members"][0]
        return {
            "source_type": "active_param_ensemble",
            "payload": payload,
            "schedule": schedule,
            "primary_params": first_member["params_obj"],
            "primary_params_signature": first_member["params_signature"],
            "member_count": int(sum(len(record.get("members") or []) for record in schedule)),
        }

    if is_rolling_oos_param_set_payload(payload):
        schedule = build_active_param_objects_from_payload(payload, fixed_risk=fixed_risk)
        first_record = schedule[0]
        return {
            "source_type": "active_param_schedule",
            "payload": payload,
            "schedule": schedule,
            "primary_params": first_record["params_obj"],
            "primary_params_signature": first_record["params_signature"],
            "member_count": int(len(schedule)),
        }

    params = build_params_from_mapping(payload)
    if fixed_risk is not None:
        params.fixed_risk = float(fixed_risk)
    return {
        "source_type": "single_param",
        "payload": payload,
        "schedule": [],
        "primary_params": params,
        "primary_params_signature": build_portfolio_params_signature(params),
        "member_count": 1,
    }


def load_portfolio_primary_params_from_json(json_file: str | os.PathLike[str], *, fixed_risk: float | None = None):
    """Return the primary StrategyParams object from a single-param or active-param runtime source.

    Formal single-stock checks need one deterministic StrategyParams instance. When the shipped
    run_best file is an ensemble, they validate the first declared member while portfolio replay
    entry points continue to use the full ensemble source.
    """
    return load_portfolio_param_source_from_json(json_file, fixed_risk=fixed_risk)["primary_params"]
