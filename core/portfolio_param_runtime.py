from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from core.active_param_ensemble import build_active_param_ensemble_schedule
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.rolling_oos_params import build_active_param_schedule


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
