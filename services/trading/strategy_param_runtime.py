"""Trading adapter for canonical single/ensemble Strategy Optimizer artifacts.

This module does not define ensemble semantics.  It exposes the same static active-param
ensemble members and min-agree policy already consumed by the canonical portfolio replay
engine, while adding Trading-specific validation and deterministic member lookup.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from core.active_param_ensemble import get_active_param_ensemble_policy
from core.file_integrity import canonical_json_sha256
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



TRADING_POSITION_STRATEGY_LINEAGE_SCHEMA_VERSION = 1


def build_trading_candidate_strategy_lineage(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze Scanner strategy identity directly onto the resulting position.

    This is the canonical strategy-management lineage for non-OMS Trading.  A
    broker order may still exist for legacy compatibility, but position management
    must not require it in order to recover the params used at entry time.
    """
    params_obj, member = resolve_trading_candidate_frozen_params(candidate)
    frozen_params = params_to_json_dict(params_obj)
    frozen_params_sha256 = canonical_json_sha256(frozen_params)
    member_key = str(member.get("member_key") or candidate.get("ensemble_member_key") or "").strip()
    params_signature = str(member.get("params_signature") or candidate.get("params_signature") or "").strip()
    seed = deepcopy(dict(candidate.get("execution_plan_seed") or {}))
    identity_payload = {
        "params_signature": params_signature,
        "ensemble_member_key": member_key,
        "frozen_params_sha256": frozen_params_sha256,
        "execution_plan_seed": seed,
    }
    return {
        "schema_version": TRADING_POSITION_STRATEGY_LINEAGE_SCHEMA_VERSION,
        "lineage_id": canonical_json_sha256(identity_payload),
        "params_signature": params_signature,
        "ensemble_member_key": member_key,
        "frozen_params": frozen_params,
        "frozen_params_sha256": frozen_params_sha256,
        "execution_plan_seed": seed,
        "origin": "scanner_candidate",
    }



def build_trading_order_strategy_lineage(order: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a legacy broker ENTRY order into the same position lineage contract."""
    frozen_params = order.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise RuntimeError("Trading entry order 缺少 frozen_params")
    frozen_params = dict(frozen_params)
    frozen_sha = canonical_json_sha256(frozen_params)
    expected_sha = str(order.get("frozen_params_sha256") or "")
    if expected_sha and expected_sha != frozen_sha:
        raise RuntimeError("Trading entry order frozen_params hash 不一致")
    params_obj = build_params_from_mapping(frozen_params)
    params_signature = str(order.get("params_signature") or build_portfolio_params_signature(params_obj))
    member_key = str(order.get("ensemble_member_key") or "").strip()
    execution_seed = {
        "entry_type": str(order.get("entry_type") or "normal"),
        "security_profile": deepcopy(order.get("security_profile")),
    }
    identity_payload = {
        "params_signature": params_signature,
        "ensemble_member_key": member_key,
        "frozen_params_sha256": frozen_sha,
        "entry_order_id": str(order.get("order_id") or ""),
    }
    return {
        "schema_version": TRADING_POSITION_STRATEGY_LINEAGE_SCHEMA_VERSION,
        "lineage_id": canonical_json_sha256(identity_payload),
        "params_signature": params_signature,
        "ensemble_member_key": member_key,
        "frozen_params": frozen_params,
        "frozen_params_sha256": frozen_sha,
        "execution_plan_seed": execution_seed,
        "origin": "legacy_entry_order",
        "legacy_entry_order_id": str(order.get("order_id") or "") or None,
    }

def validate_trading_position_strategy_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(lineage or {})
    if int(payload.get("schema_version") or -1) != TRADING_POSITION_STRATEGY_LINEAGE_SCHEMA_VERSION:
        raise RuntimeError("Trading position strategy_lineage schema 不相容")
    frozen_params = payload.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise RuntimeError("Trading position strategy_lineage 缺少 frozen_params")
    frozen_params = dict(frozen_params)
    actual_sha = canonical_json_sha256(frozen_params)
    expected_sha = str(payload.get("frozen_params_sha256") or "")
    if actual_sha != expected_sha:
        raise RuntimeError("Trading position strategy_lineage frozen_params hash 不一致")
    params_obj = build_params_from_mapping(frozen_params)
    actual_signature = build_portfolio_params_signature(params_obj)
    expected_signature = str(payload.get("params_signature") or "")
    if expected_signature and actual_signature != expected_signature:
        raise RuntimeError("Trading position strategy_lineage params_signature 不一致")
    lineage_id = str(payload.get("lineage_id") or "").strip()
    if not lineage_id:
        raise RuntimeError("Trading position strategy_lineage 缺少 lineage_id")
    return {
        **payload,
        "frozen_params": frozen_params,
        "frozen_params_sha256": actual_sha,
        "params_signature": actual_signature,
        "lineage_id": lineage_id,
    }


def resolve_trading_position_strategy_binding(
    record: Mapping[str, Any],
    *,
    orders: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve immutable strategy params from position SSOT with legacy OMS fallback."""
    lineage = record.get("strategy_lineage") if isinstance(record, Mapping) else None
    broker = dict(record.get("broker") or {}) if isinstance(record, Mapping) else {}
    legacy_entry_order_id = str(broker.get("entry_order_id") or "").strip()
    if isinstance(lineage, Mapping):
        checked = validate_trading_position_strategy_lineage(lineage)
        return {
            "lineage_id": checked["lineage_id"],
            "lineage_key": f"POSITION:{checked['lineage_id']}",
            "entry_order_id": legacy_entry_order_id or None,
            "frozen_params": deepcopy(checked["frozen_params"]),
            "frozen_params_sha256": checked["frozen_params_sha256"],
            "params_signature": checked["params_signature"],
            "ensemble_member_key": str(checked.get("ensemble_member_key") or ""),
            "execution_plan_seed": deepcopy(dict(checked.get("execution_plan_seed") or {})),
            "source": "position_strategy_lineage",
        }

    if legacy_entry_order_id and isinstance(orders, Mapping):
        order = dict((orders.get("orders") or {}).get(legacy_entry_order_id) or {})
        frozen_params = order.get("frozen_params")
        if isinstance(frozen_params, Mapping):
            frozen_params = dict(frozen_params)
            actual_sha = canonical_json_sha256(frozen_params)
            expected_sha = str(order.get("frozen_params_sha256") or "")
            if expected_sha and expected_sha != actual_sha:
                raise RuntimeError("Trading legacy entry order frozen_params hash 不一致")
            params_obj = build_params_from_mapping(frozen_params)
            return {
                "lineage_id": legacy_entry_order_id,
                "lineage_key": legacy_entry_order_id,
                "entry_order_id": legacy_entry_order_id,
                "frozen_params": frozen_params,
                "frozen_params_sha256": actual_sha,
                "params_signature": build_portfolio_params_signature(params_obj),
                "ensemble_member_key": str(order.get("ensemble_member_key") or ""),
                "execution_plan_seed": deepcopy(dict(order.get("execution_plan_seed") or {})),
                "source": "legacy_entry_order",
            }
    raise RuntimeError("Trading strategy position 缺少 immutable strategy_lineage / legacy frozen params")

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
    "TRADING_POSITION_STRATEGY_LINEAGE_SCHEMA_VERSION",
    "build_trading_candidate_strategy_lineage",
    "build_trading_order_strategy_lineage",
    "validate_trading_position_strategy_lineage",
    "resolve_trading_position_strategy_binding",
    "resolve_trading_candidate_params",
]
