"""Trading adapter for canonical single/ensemble Strategy Optimizer artifacts.

This module does not define ensemble semantics.  It exposes the same static active-param
ensemble members and min-agree policy already consumed by the canonical portfolio replay
engine, while adding Trading-specific validation and deterministic member lookup.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from core.active_param_ensemble import (
    get_active_param_ensemble_policy,
    get_active_param_ensemble_record_for_date,
    resolve_active_param_ensemble_mode,
)
from core.seed_ensemble_policy import resolve_seed_ensemble_min_agree
from core.file_integrity import canonical_json_sha256
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.portfolio_param_runtime import build_portfolio_params_signature, load_portfolio_param_source_from_json


def _scheduled_effective_end_date(payload: Mapping[str, Any], *, effective_date: str) -> str:
    if not effective_date:
        return ""
    meta = dict(payload.get("meta") or {})
    months = int(meta.get("oos_horizon_months") or 12)
    if months < 1:
        raise ValueError("Trading rolling Params oos_horizon_months必須>=1")
    start = datetime.strptime(str(effective_date)[:10], "%Y-%m-%d").date()
    month_index = (start.year * 12 + (start.month - 1)) + months
    next_start = date(month_index // 12, month_index % 12 + 1, 1)
    return (next_start - timedelta(days=1)).isoformat()


def _rolling_fold_metadata(payload: Mapping[str, Any], *, effective_date: str) -> dict[str, Any]:
    for raw in list(payload.get("folds") or []):
        if not isinstance(raw, Mapping):
            continue
        start = str(raw.get("effective_start") or raw.get("oos_start_date") or "").strip()
        if start == str(effective_date):
            return dict(raw)
    return {}


def load_trading_strategy_param_runtime(
    selected_path: str | Path,
    *,
    information_date: str | None = None,
) -> dict[str, Any]:
    """Resolve the legal Trading Params member set for one information date.

    Rolling schedules are PIT-resolved by effective date.  When ``information_date``
    is omitted, the last covered schedule record is selected for compatibility with
    current-state display callers; Scanner/binding callers pass the market date
    explicitly.
    """
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
            "effective_date": "",
            "effective_end_date": "",
            "training_data_date": "",
        }

    if source_type != "active_param_ensemble":
        raise RuntimeError(
            "Trading只接受單一參數或 active-param ensemble；"
            f"收到 source_type={source_type or '-'}"
        )
    schedule = list(source.get("schedule") or [])
    if not schedule:
        raise RuntimeError("Trading active-param ensemble 沒有可用 schedule")
    mode = resolve_active_param_ensemble_mode(source["payload"])
    if mode == "rolling":
        if information_date in (None, ""):
            selected_record = dict(schedule[-1])
        else:
            raw_record = get_active_param_ensemble_record_for_date(
                source["payload"], information_date
            )
            effective_date = str(raw_record.get("effective_date_text") or "")
            matches = [
                dict(record)
                for record in schedule
                if str(record.get("effective_date_text") or "") == effective_date
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Trading rolling Params 無法唯一解析 information-date record: "
                    f"information_date={information_date}, effective_date={effective_date}"
                )
            selected_record = matches[0]
    elif mode == "static":
        if len(schedule) != 1:
            raise RuntimeError("Trading static active-param ensemble schedule 不唯一")
        selected_record = dict(schedule[0])
    else:
        raise RuntimeError(f"Trading active-param ensemble mode不合法: {mode or '-'}")

    members = [dict(member) for member in list(selected_record.get("members") or [])]
    if not members:
        raise RuntimeError("Trading active-param ensemble 沒有可用 members")
    for idx, member in enumerate(members, 1):
        member_key = str(member.get("member_key") or member.get("member_index") or member.get("seed") or idx)
        member["member_key"] = member_key
        if member.get("params_obj") is None or not str(member.get("params_signature") or ""):
            raise RuntimeError(f"Trading ensemble member 缺少 params runtime identity: member={member_key}")

    policy = get_active_param_ensemble_policy(source["payload"])
    effective_date = str(selected_record.get("effective_date_text") or "")
    fold = _rolling_fold_metadata(source["payload"], effective_date=effective_date) if mode == "rolling" else {}
    scheduled_effective_end_date = (
        _scheduled_effective_end_date(source["payload"], effective_date=effective_date)
        if mode == "rolling"
        else str(selected_record.get("effective_end_date_text") or "")
    )
    if mode == "rolling" and information_date not in (None, ""):
        info_date = datetime.strptime(str(information_date)[:10], "%Y-%m-%d").date()
        if scheduled_effective_end_date and info_date > datetime.strptime(scheduled_effective_end_date, "%Y-%m-%d").date():
            raise RuntimeError(
                "Trading rolling Params 已超過目前PIT refit period；請重新訓練 Params。"
                f" information_date={info_date.isoformat()}, valid_through={scheduled_effective_end_date}"
            )
    requested_min_agree = fold.get("min_agree")
    if requested_min_agree in (None, ""):
        requested_min_agree = fold.get("min_agree_requested")
    if requested_min_agree in (None, ""):
        requested_min_agree = policy.get("min_agree_requested", policy.get("min_agree", "auto"))
    min_agree = resolve_seed_ensemble_min_agree(len(members), requested_min_agree)
    if min_agree < 1 or min_agree > len(members):
        raise RuntimeError(
            "Trading ensemble min_agree 不合法；"
            f"min_agree={min_agree}, members={len(members)}"
        )
    training_data_date = str(fold.get("selection_end_date") or "").strip()
    return {
        "source": source,
        "source_type": source_type,
        "schedule_mode": mode,
        "members": members,
        "member_count": len(members),
        "min_agree": int(min_agree),
        "ensemble_policy": policy,
        "primary_params": members[0]["params_obj"],
        "effective_date": effective_date,
        "effective_end_date": str(selected_record.get("effective_end_date_text") or ""),
        "scheduled_effective_end_date": scheduled_effective_end_date,
        "training_data_date": training_data_date,
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
        # Preserve the pre-fill lifecycle evidence on the position itself.  The
        # account journal is the durable source after the Scanner snapshot has
        # moved on, so historical Trading inspection must not depend on today's
        # candidate pool to reconstruct SIGNAL -> SHADOW -> POSITION.
        "candidate_trade_date": str(candidate.get("trade_date") or seed.get("trade_date") or "") or None,
        "signal_date": str(candidate.get("signal_date") or candidate.get("trade_date") or seed.get("trade_date") or "") or None,
        "planned_qty": candidate.get("proj_qty"),
        "planned_cost": candidate.get("proj_cost"),
        "candidate_kind": str(candidate.get("kind") or seed.get("entry_source") or "normal"),
        "origin": "scanner_candidate",
        "signal_lineage_id": (candidate.get("signal_lineage") or {}).get("lineage_id"),
        "signal_origin": deepcopy(candidate.get("signal_lineage")),
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
    execution_seed = deepcopy(dict(order.get("execution_plan_seed") or {}))
    execution_seed.setdefault("entry_type", str(order.get("entry_type") or "normal"))
    execution_seed.setdefault("security_profile", deepcopy(order.get("security_profile")))
    from core.exact_accounting import milli_to_price
    for field, source in (("init_sl", "init_sl_milli"), ("init_trail", "init_trail_milli"),
                          ("target_price", "target_price_milli"), ("limit_price", "limit_price_milli"),
                          ("entry_atr", "entry_atr_milli")):
        if source in order and order[source] is not None:
            execution_seed.setdefault(field, milli_to_price(int(order[source])))
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
        "signal_date": order.get("signal_date") or order.get("information_date"),
        "candidate_trade_date": order.get("information_date"),
        "signal_lineage_id": (order.get("signal_lineage") or {}).get("lineage_id"),
        "signal_origin": deepcopy(order.get("signal_lineage")),
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
            "signal_date": checked.get("signal_date"),
            "signal_lineage_id": checked.get("signal_lineage_id"),
            "signal_origin": deepcopy(checked.get("signal_origin")),
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
                "execution_plan_seed": build_trading_order_strategy_lineage(order)["execution_plan_seed"],
                "signal_date": order.get("signal_date") or order.get("information_date"),
                "source": "legacy_entry_order",
            }
    raise RuntimeError("Trading strategy position 缺少 immutable strategy_lineage / legacy frozen params")


TRADING_POSITION_MANAGEMENT_LINEAGE_SCHEMA_VERSION = 1

def build_trading_manual_management_lineage(
    *,
    params,
    execution_plan_seed: Mapping[str, Any],
    information_date: str,
    origin: str,
    planned_qty: int | None = None,
    planned_cost: float | None = None,
    target_reference_close: float | None = None,
) -> dict[str, Any]:
    """Freeze current canonical primary Params for a user-selected managed position.

    Manual selection is not a Scanner vote.  It therefore freezes the current
    artifact's primary Params while keeping an explicit manual origin.
    """
    frozen_params = params_to_json_dict(params)
    frozen_params_sha256 = canonical_json_sha256(frozen_params)
    params_signature = build_portfolio_params_signature(params)
    seed = deepcopy(dict(execution_plan_seed or {}))
    identity_payload = {
        "params_signature": params_signature,
        "frozen_params_sha256": frozen_params_sha256,
        "execution_plan_seed": seed,
        "information_date": str(information_date),
        "origin": str(origin),
        "target_reference_close": target_reference_close,
    }
    return {
        "schema_version": TRADING_POSITION_MANAGEMENT_LINEAGE_SCHEMA_VERSION,
        "lineage_id": canonical_json_sha256(identity_payload),
        "params_signature": params_signature,
        "ensemble_member_key": "PRIMARY",
        "frozen_params": frozen_params,
        "frozen_params_sha256": frozen_params_sha256,
        "execution_plan_seed": seed,
        "candidate_trade_date": str(information_date),
        "signal_date": None,
        "planned_qty": planned_qty,
        "planned_cost": planned_cost,
        "candidate_kind": "manual",
        "origin": str(origin),
        "target_reference_close": target_reference_close,
    }


def validate_trading_position_management_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(lineage or {})
    if int(payload.get("schema_version") or -1) != TRADING_POSITION_MANAGEMENT_LINEAGE_SCHEMA_VERSION:
        raise RuntimeError("Trading position management_lineage schema 不相容")
    frozen_params = payload.get("frozen_params")
    if not isinstance(frozen_params, Mapping):
        raise RuntimeError("Trading position management_lineage 缺少 frozen_params")
    frozen_params = dict(frozen_params)
    actual_sha = canonical_json_sha256(frozen_params)
    expected_sha = str(payload.get("frozen_params_sha256") or "")
    if actual_sha != expected_sha:
        raise RuntimeError("Trading position management_lineage frozen_params hash 不一致")
    params_obj = build_params_from_mapping(frozen_params)
    actual_signature = build_portfolio_params_signature(params_obj)
    expected_signature = str(payload.get("params_signature") or "")
    if expected_signature and actual_signature != expected_signature:
        raise RuntimeError("Trading position management_lineage params_signature 不一致")
    lineage_id = str(payload.get("lineage_id") or "").strip()
    if not lineage_id:
        raise RuntimeError("Trading position management_lineage 缺少 lineage_id")
    return {
        **payload,
        "frozen_params": frozen_params,
        "frozen_params_sha256": actual_sha,
        "params_signature": actual_signature,
        "lineage_id": lineage_id,
    }


def resolve_trading_position_management_binding(
    record: Mapping[str, Any],
    *,
    orders: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve frozen management Params for strategy or manual-managed positions."""
    lineage = record.get("management_lineage") if isinstance(record, Mapping) else None
    if isinstance(lineage, Mapping):
        checked = validate_trading_position_management_lineage(lineage)
        return {
            "lineage_id": checked["lineage_id"],
            "lineage_key": f"POSITION:{checked['lineage_id']}",
            "entry_order_id": None,
            "frozen_params": deepcopy(checked["frozen_params"]),
            "frozen_params_sha256": checked["frozen_params_sha256"],
            "params_signature": checked["params_signature"],
            "ensemble_member_key": str(checked.get("ensemble_member_key") or "PRIMARY"),
            "execution_plan_seed": deepcopy(dict(checked.get("execution_plan_seed") or {})),
            "signal_date": checked.get("signal_date"),
            "signal_lineage_id": checked.get("signal_lineage_id"),
            "signal_origin": deepcopy(checked.get("signal_origin")),
            "origin": str(checked.get("origin") or "manual_managed"),
            "source": "position_management_lineage",
        }
    return resolve_trading_position_strategy_binding(record, orders=orders)


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
    "TRADING_POSITION_MANAGEMENT_LINEAGE_SCHEMA_VERSION",
    "build_trading_manual_management_lineage",
    "validate_trading_position_management_lineage",
    "resolve_trading_position_management_binding",
    "resolve_trading_candidate_params",
]
