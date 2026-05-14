from __future__ import annotations

import copy
import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Mapping

from core.seed_ensemble_policy import normalize_seed_ensemble_members

ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE = "optimizer_active_param_ensemble"
ACTIVE_PARAM_ENSEMBLE_SCHEMA_VERSION = 1
ACTIVE_PARAM_ENSEMBLE_MODE_STATIC = "static"
ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING = "rolling"
STATIC_ENSEMBLE_EFFECTIVE_DATE = date(1900, 1, 1)
STATIC_ENSEMBLE_EFFECTIVE_END_DATE = date(9999, 12, 31)


def load_json_file(path: str | os.PathLike[str]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根層必須是 object/dict，收到 {type(payload).__name__}")
    return payload


def parse_effective_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        text = f"{text}-01-01"
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"無法解析參數生效日: {value!r}") from exc


def coerce_trade_date(value: Any) -> date:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    return parse_effective_date(value)


def _payload_effective_end_map(payload: Mapping[str, Any]) -> dict[str, date]:
    end_by_start: dict[str, date] = {}
    for item in list(payload.get("folds") or []):
        if not isinstance(item, Mapping):
            continue
        raw_start = item.get("effective_start") or item.get("oos_start_date")
        raw_end = item.get("effective_end") or item.get("oos_end_date")
        if not raw_start or not raw_end:
            continue
        try:
            start_date = parse_effective_date(raw_start).isoformat()
            end_by_start[start_date] = parse_effective_date(raw_end)
        except ValueError:
            continue
    return end_by_start


def _payload_summary_oos_end(payload: Mapping[str, Any]) -> date | None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    candidates = [summary.get("oos_end_date"), payload.get("oos_end_date")]
    oos_period = str(summary.get("oos_period") or payload.get("oos_period") or "").strip()
    if "~" in oos_period:
        candidates.append(oos_period.split("~")[-1].strip())
    for raw in candidates:
        if not raw:
            continue
        try:
            return parse_effective_date(raw)
        except ValueError:
            continue
    return None


def resolve_active_param_ensemble_mode(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    raw_mode = str(payload.get("mode") or payload.get("param_ensemble_mode") or "").strip().lower()
    if raw_mode in {ACTIVE_PARAM_ENSEMBLE_MODE_STATIC, ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING}:
        return raw_mode
    if isinstance(payload.get("params_ensemble_by_effective_date"), Mapping):
        return ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING
    if isinstance(payload.get("params_ensemble"), list):
        return ACTIVE_PARAM_ENSEMBLE_MODE_STATIC
    return ""


def is_active_param_ensemble_payload(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    schema_type = str(payload.get("schema_type") or payload.get("type") or "").strip()
    if schema_type == ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE:
        return True
    return resolve_active_param_ensemble_mode(payload) in {
        ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
        ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    }


def is_rolling_active_param_ensemble_payload(payload: Mapping[str, Any] | None) -> bool:
    return is_active_param_ensemble_payload(payload) and resolve_active_param_ensemble_mode(payload) == ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING


def is_static_active_param_ensemble_payload(payload: Mapping[str, Any] | None) -> bool:
    return is_active_param_ensemble_payload(payload) and resolve_active_param_ensemble_mode(payload) == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC


def _build_static_active_param_ensemble_schedule(payload: Mapping[str, Any]) -> list[dict]:
    members = normalize_seed_ensemble_members(payload.get("params_ensemble"))
    if not members:
        raise ValueError("static active-param ensemble 缺少 params_ensemble 或沒有可用 params")
    raw_start = payload.get("effective_start") or payload.get("effective_date") or payload.get("start_date")
    raw_end = payload.get("effective_end") or payload.get("end_date")
    effective_date = parse_effective_date(raw_start) if raw_start else STATIC_ENSEMBLE_EFFECTIVE_DATE
    effective_end = parse_effective_date(raw_end) if raw_end else STATIC_ENSEMBLE_EFFECTIVE_END_DATE
    return [{
        "mode": ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
        "effective_date": effective_date,
        "effective_date_text": effective_date.isoformat() if raw_start else "",
        "effective_end_date": effective_end,
        "effective_end_date_text": effective_end.isoformat() if raw_end else "",
        "year": int(effective_date.year),
        "members": members,
    }]


def _build_rolling_active_param_ensemble_schedule(payload: Mapping[str, Any]) -> list[dict]:
    params_ensemble_by_effective_date = payload.get("params_ensemble_by_effective_date")
    if not isinstance(params_ensemble_by_effective_date, Mapping) or not params_ensemble_by_effective_date:
        raise ValueError("rolling active-param ensemble 缺少 params_ensemble_by_effective_date")
    effective_end_by_start = _payload_effective_end_map(payload)
    records: list[dict] = []
    for raw_date, raw_members in params_ensemble_by_effective_date.items():
        effective_date = parse_effective_date(raw_date)
        members = normalize_seed_ensemble_members(raw_members)
        if not members:
            raise ValueError(f"生效日 {raw_date} 的 ensemble 參數組必須至少包含一組 params")
        effective_end = effective_end_by_start.get(effective_date.isoformat())
        records.append({
            "mode": ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
            "effective_date": effective_date,
            "effective_date_text": effective_date.isoformat(),
            "effective_end_date": effective_end,
            "effective_end_date_text": effective_end.isoformat() if effective_end is not None else "",
            "year": int(effective_date.year),
            "members": members,
        })
    records.sort(key=lambda item: item["effective_date"])
    summary_oos_end = _payload_summary_oos_end(payload)
    for idx, record in enumerate(records):
        if record.get("effective_end_date") is not None:
            continue
        if idx + 1 < len(records):
            effective_end = records[idx + 1]["effective_date"] - timedelta(days=1)
        elif summary_oos_end is not None:
            effective_end = summary_oos_end
        else:
            effective_end = date(int(record["effective_date"].year), 12, 31)
        record["effective_end_date"] = effective_end
        record["effective_end_date_text"] = effective_end.isoformat()
    return records


def build_active_param_ensemble_schedule(payload: Mapping[str, Any]) -> list[dict]:
    mode = resolve_active_param_ensemble_mode(payload)
    if mode == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        return _build_static_active_param_ensemble_schedule(payload)
    if mode == ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING:
        return _build_rolling_active_param_ensemble_schedule(payload)
    raise ValueError("active-param ensemble JSON 缺少 mode / params_ensemble / params_ensemble_by_effective_date")


def get_active_param_ensemble_record_for_date(payload: Mapping[str, Any], trade_date: Any) -> dict:
    schedule = build_active_param_ensemble_schedule(payload)
    mode = resolve_active_param_ensemble_mode(payload)
    if mode == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        return copy.deepcopy(schedule[0])
    current_date = coerce_trade_date(trade_date)
    selected = None
    for record in schedule:
        if record["effective_date"] <= current_date:
            selected = record
        else:
            break
    if selected is None:
        first_date = schedule[0]["effective_date_text"]
        raise KeyError(f"{current_date.isoformat()} 早於 rolling ensemble 第一個參數生效日 {first_date}")
    return copy.deepcopy(selected)


def get_active_param_ensemble_members_for_date(payload: Mapping[str, Any], trade_date: Any) -> list[dict]:
    return copy.deepcopy(get_active_param_ensemble_record_for_date(payload, trade_date)["members"])


def get_active_param_ensemble_date_range(payload: Mapping[str, Any]) -> tuple[str, str]:
    schedule = build_active_param_ensemble_schedule(payload)
    if resolve_active_param_ensemble_mode(payload) == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        return "", ""
    first_date = schedule[0]["effective_date"]
    last_end = schedule[-1].get("effective_end_date") or schedule[-1]["effective_date"]
    return first_date.isoformat(), last_end.isoformat()


def get_active_param_ensemble_year_range(payload: Mapping[str, Any]) -> tuple[int, int]:
    first_date, last_date = get_active_param_ensemble_date_range(payload)
    if not first_date or not last_date:
        return 0, 0
    return int(first_date[:4]), int(last_date[:4])


def load_active_param_ensemble_set(path: str | os.PathLike[str]) -> dict:
    payload = load_json_file(path)
    if not is_active_param_ensemble_payload(payload):
        raise ValueError(f"不是 active-param ensemble JSON: {path}")
    build_active_param_ensemble_schedule(payload)
    return payload


def is_active_param_ensemble_file(path: str | os.PathLike[str]) -> bool:
    try:
        return is_active_param_ensemble_payload(load_json_file(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return False


def build_static_active_param_ensemble_payload(
    *,
    members: Any,
    random_seed_ensemble: Mapping[str, Any] | None = None,
    selector: str = "static",
    meta: Mapping[str, Any] | None = None,
    created_at: str = "",
) -> dict:
    normalized_members = normalize_seed_ensemble_members(members)
    if not normalized_members:
        raise ValueError("static active-param ensemble payload 必須至少包含一組 params")
    payload = {
        "schema_type": ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
        "schema_version": ACTIVE_PARAM_ENSEMBLE_SCHEMA_VERSION,
        "mode": ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
        "type": "active_param_ensemble",
        "selector": str(selector),
        "params_ensemble": normalized_members,
    }
    if created_at:
        payload["created_at"] = str(created_at)
    if random_seed_ensemble is not None:
        payload["random_seed_ensemble"] = dict(random_seed_ensemble)
    if meta is not None:
        payload["meta"] = dict(meta)
    return payload

def get_active_param_ensemble_policy(payload: Mapping[str, Any]) -> dict:
    policy = payload.get("random_seed_ensemble") if isinstance(payload.get("random_seed_ensemble"), Mapping) else {}
    schedule = build_active_param_ensemble_schedule(payload)
    member_counts = [len(record.get("members") or []) for record in schedule if record.get("members")]
    actual_min_members = max(1, min(member_counts) if member_counts else 1)
    actual_max_members = max(member_counts) if member_counts else actual_min_members
    min_agree = policy.get("min_agree_requested", policy.get("min_agree", "auto"))
    from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot
    resolved = build_seed_ensemble_policy_snapshot(
        enabled=actual_min_members > 1,
        seed_count=actual_min_members,
        min_agree=min_agree,
    )
    requested_seed_count = policy.get("seed_count")
    if requested_seed_count is not None:
        try:
            requested_seed_count = int(requested_seed_count)
        except (TypeError, ValueError):
            requested_seed_count = None
    if requested_seed_count is not None:
        resolved["seed_count_requested"] = int(requested_seed_count)
    resolved["member_count_min"] = int(actual_min_members)
    resolved["member_count_max"] = int(actual_max_members)
    if requested_seed_count is not None and int(requested_seed_count) != int(actual_min_members):
        resolved["member_count_mismatch"] = True
        resolved["note"] = "seed_count/min_agree 已依 JSON 實際 members 數量修正，避免要求超過可用 members 的共識數。"
    else:
        resolved["member_count_mismatch"] = actual_min_members != actual_max_members
    return resolved


def format_active_param_ensemble_summary_lines(payload: Mapping[str, Any]) -> list[str]:
    mode = resolve_active_param_ensemble_mode(payload)
    schedule = build_active_param_ensemble_schedule(payload)
    policy = get_active_param_ensemble_policy(payload)
    if mode == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        return [
            f"Active-param ensemble：static｜members={policy['seed_count']}｜min_agree={policy['min_agree']}/{policy['seed_count']}",
            "參數模式：全期間使用同一組 N-seed 共識參數。",
        ]
    first_date, last_date = get_active_param_ensemble_date_range(payload)
    return [
        f"Active-param ensemble：rolling｜members={policy['seed_count']}｜min_agree={policy['min_agree']}/{policy['seed_count']}",
        f"Active param schedule：{first_date} ~ {last_date}｜effective_dates={len(schedule)}",
    ]
