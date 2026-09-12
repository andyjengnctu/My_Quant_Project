"""Trading adapter for canonical single/ensemble Strategy Optimizer artifacts.

This module does not define ensemble semantics.  It exposes the same static active-param
ensemble members and min-agree policy already consumed by the canonical portfolio replay
engine, while adding Trading-specific validation and deterministic member lookup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.active_param_ensemble import get_active_param_ensemble_policy
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.portfolio_param_runtime import build_portfolio_params_signature, load_portfolio_param_source_from_json


def load_trading_strategy_param_runtime(selected_path: str | Path) -> dict[str, Any]:
    path = Path(selected_path)
    source = load_portfolio_param_source_from_json(path)
    source_type = str(source.get("source_type") or "")

    if source_type == "single_param":
        member = {
            "member_key": "1",
            "member_index": 1,
            "seed": None,
            "params_obj": source["primary_params"],
            "params_signature": source["primary_params_signature"],
        }
        return {
            "source": source,
            "source_type": source_type,
            "members": [member],
            "member_count": 1,
            "min_agree": 1,
            "primary_params": source["primary_params"],
        }

    if source_type != "active_param_ensemble":
        raise RuntimeError(
            "Trading 目前只接受單一參數或 static active-param ensemble；"
            f"收到 source_type={source_type or '-'}"
        )
    schedule = list(source.get("schedule") or [])
    if len(schedule) != 1 or str(schedule[0].get("mode") or "") != "static":
        raise RuntimeError("Trading 目前只接受 static active-param ensemble，禁止 rolling param schedule")
    members = [dict(member) for member in list(schedule[0].get("members") or [])]
    if not members:
        raise RuntimeError("Trading active-param ensemble 沒有可用 members")
    for idx, member in enumerate(members, 1):
        member_key = str(member.get("member_key") or member.get("member_index") or member.get("seed") or idx)
        member["member_key"] = member_key
        if member.get("params_obj") is None or not str(member.get("params_signature") or ""):
            raise RuntimeError(f"Trading ensemble member 缺少 params runtime identity: member={member_key}")

    policy = get_active_param_ensemble_policy(source["payload"])
    min_agree = int(policy.get("min_agree") or 0)
    if min_agree < 1 or min_agree > len(members):
        raise RuntimeError(
            "Trading ensemble min_agree 不合法；"
            f"min_agree={min_agree}, members={len(members)}"
        )
    return {
        "source": source,
        "source_type": source_type,
        "members": members,
        "member_count": len(members),
        "min_agree": min_agree,
        "ensemble_policy": policy,
        "primary_params": source["primary_params"],
    }


def serialize_trading_candidate_member_params(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return JSON-safe immutable voter Params keyed by ensemble member identity."""

    raw = candidate.get("ensemble_member_params_by_key") or {}
    if not isinstance(raw, Mapping):
        raise RuntimeError("Trading candidate ensemble_member_params_by_key 必須是 object")
    serialized: dict[str, dict[str, Any]] = {}
    for raw_key, value in raw.items():
        member_key = str(raw_key or "").strip()
        if not member_key:
            raise RuntimeError("Trading candidate voter Params 缺少 member key")
        if isinstance(value, Mapping):
            params_obj = build_params_from_mapping(dict(value))
            payload = params_to_json_dict(params_obj)
        else:
            payload = params_to_json_dict(value)
            params_obj = build_params_from_mapping(payload)
        if build_portfolio_params_signature(params_obj) != build_portfolio_params_signature(build_params_from_mapping(payload)):
            raise RuntimeError(f"Trading candidate voter Params 無法穩定序列化: member={member_key}")
        serialized[member_key] = payload
    return serialized


def resolve_trading_candidate_frozen_params(candidate: Mapping[str, Any]):
    """Resolve representative Params from the candidate's immutable agreeing-voter lineage."""

    member_key = str(candidate.get("ensemble_member_key") or "").strip()
    signature = str(candidate.get("params_signature") or "").strip()
    raw_map = candidate.get("ensemble_member_params_by_key") or {}
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise RuntimeError("Trading candidate 缺少 immutable ensemble voter Params lineage")
    if not member_key:
        if len(raw_map) == 1:
            member_key = str(next(iter(raw_map)))
        else:
            raise RuntimeError("Trading ensemble candidate 缺少 representative member key")
    payload = raw_map.get(member_key)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Trading candidate representative member Params 不存在: member={member_key}")
    params_obj = build_params_from_mapping(dict(payload))
    resolved_signature = build_portfolio_params_signature(params_obj)
    if signature and resolved_signature != signature:
        raise RuntimeError(
            "Trading candidate frozen Params signature 不一致；"
            f"member={member_key}, expected={signature[:12]}, actual={resolved_signature[:12]}"
        )
    member = {
        "member_key": member_key,
        "params_obj": params_obj,
        "params_signature": resolved_signature,
    }
    return params_obj, member


def resolve_trading_candidate_params(param_runtime: Mapping[str, Any], candidate: Mapping[str, Any]):
    """Resolve the exact representative member frozen by canonical ensemble aggregation.

    ``load_trading_strategy_param_runtime`` exposes ``members`` directly, while the
    Scanner runtime intentionally wraps the same canonical members as ``param_members``.
    Both are adapters over the same artifact identity; accepting either wrapper keeps
    the candidate -> proposed-order seam semantic-preserving without duplicating vote
    or member-selection logic.
    """

    members = list(param_runtime.get("members") or param_runtime.get("param_members") or [])
    if not members:
        raise RuntimeError("Trading Params runtime 沒有可用 member")
    if len(members) == 1:
        expected_signature = str(members[0].get("params_signature") or "")
        candidate_signature = str(candidate.get("params_signature") or "")
        if candidate_signature and candidate_signature != expected_signature:
            raise RuntimeError("Trading candidate params_signature 與單一 Params artifact 不一致")
        return members[0]["params_obj"], members[0]

    signature = str(candidate.get("params_signature") or "").strip()
    member_key = str(candidate.get("ensemble_member_key") or "").strip()
    if not signature:
        raise RuntimeError("Trading ensemble candidate 缺少 representative params_signature")
    matches = [member for member in members if str(member.get("params_signature") or "") == signature]
    if member_key:
        matches = [member for member in matches if str(member.get("member_key") or "") == member_key]
    if len(matches) != 1:
        raise RuntimeError(
            "Trading ensemble candidate 無法唯一解析 representative Params；"
            f"member_key={member_key or '-'}, signature={signature[:12]}"
        )
    return matches[0]["params_obj"], matches[0]


__all__ = [
    "load_trading_strategy_param_runtime",
    "serialize_trading_candidate_member_params",
    "resolve_trading_candidate_frozen_params",
    "resolve_trading_candidate_params",
]
