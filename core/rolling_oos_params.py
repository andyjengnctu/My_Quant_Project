from __future__ import annotations

import copy
import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Mapping

from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
    ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING,
    build_active_param_ensemble_schedule as _build_generic_active_param_ensemble_schedule,
    is_rolling_active_param_ensemble_payload,
)

ROLLING_OOS_PARAM_SET_SCHEMA_TYPE = "rolling_oos_param_set"
ROLLING_OOS_USAGE = "validation_only"
ROLLING_OOS_POLICY_NAMES = ("base", "base_retention_gt_min", "local", "retention")


def is_rolling_oos_param_set_payload(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if is_rolling_active_param_ensemble_payload(payload):
        return True
    schema_type = str(payload.get("schema_type") or payload.get("type") or "").strip()
    mode = str(payload.get("mode") or "").strip().lower()
    if schema_type == ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE:
        return mode == ACTIVE_PARAM_ENSEMBLE_MODE_ROLLING
    if schema_type == ROLLING_OOS_PARAM_SET_SCHEMA_TYPE:
        return True
    if schema_type == "outer_rolling_oos_param_set":
        return True
    if str(payload.get("usage") or "").strip() != ROLLING_OOS_USAGE:
        return False
    return (
        isinstance(payload.get("params_by_oos_year"), Mapping)
        or isinstance(payload.get("params_by_effective_date"), Mapping)
        or isinstance(payload.get("params_ensemble_by_effective_date"), Mapping)
    )


def load_json_file(path: str | os.PathLike[str]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根層必須是 object/dict，收到 {type(payload).__name__}")
    return payload


def load_rolling_oos_param_set(path: str | os.PathLike[str]) -> dict:
    payload = load_json_file(path)
    if not is_rolling_oos_param_set_payload(payload):
        raise ValueError(f"不是 rolling OOS active-param replay 參數組 JSON: {path}")
    params_by_year = payload.get("params_by_oos_year")
    params_by_effective_date = payload.get("params_by_effective_date")
    params_ensemble_by_effective_date = payload.get("params_ensemble_by_effective_date")
    if (
        (not isinstance(params_by_year, Mapping) or not params_by_year)
        and (not isinstance(params_by_effective_date, Mapping) or not params_by_effective_date)
        and (not isinstance(params_ensemble_by_effective_date, Mapping) or not params_ensemble_by_effective_date)
    ):
        raise ValueError(
            f"rolling OOS 參數組缺少 params_by_oos_year / params_by_effective_date / params_ensemble_by_effective_date: {path}"
        )
    return payload


def _parse_effective_date(value: Any) -> date:
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


def _coerce_trade_date(value: Any) -> date:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    return _parse_effective_date(value)


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
            start_date = _parse_effective_date(raw_start).isoformat()
            end_by_start[start_date] = _parse_effective_date(raw_end)
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
            return _parse_effective_date(raw)
        except ValueError:
            continue
    return None


def build_active_param_ensemble_schedule(payload: Mapping[str, Any]) -> list[dict]:
    return _build_generic_active_param_ensemble_schedule(payload)


def build_active_param_schedule(payload: Mapping[str, Any]) -> list[dict]:
    params_by_effective_date = payload.get("params_by_effective_date")
    records: list[dict] = []
    effective_end_by_start = _payload_effective_end_map(payload)
    if isinstance(params_by_effective_date, Mapping) and params_by_effective_date:
        for raw_date, params in params_by_effective_date.items():
            if not isinstance(params, Mapping):
                raise ValueError(f"生效日 {raw_date} 的參數必須是 object/dict")
            effective_date = _parse_effective_date(raw_date)
            effective_end = effective_end_by_start.get(effective_date.isoformat())
            records.append({
                "effective_date": effective_date,
                "effective_date_text": effective_date.isoformat(),
                "effective_end_date": effective_end,
                "effective_end_date_text": effective_end.isoformat() if effective_end is not None else "",
                "year": int(effective_date.year),
                "params": dict(params),
            })
    else:
        params_by_year = payload.get("params_by_oos_year") or {}
        if not isinstance(params_by_year, Mapping) or not params_by_year:
            if isinstance(payload.get("params_ensemble_by_effective_date"), Mapping) and payload.get("params_ensemble_by_effective_date"):
                raise ValueError("此 rolling OOS JSON 僅包含 ensemble 參數組；單一 active-param replay 請改用 params_by_oos_year / params_by_effective_date")
            raise ValueError("rolling OOS 參數組缺少 params_by_oos_year / params_by_effective_date")
        for raw_year, params in params_by_year.items():
            if not isinstance(params, Mapping):
                raise ValueError(f"OOS 期間 {raw_year} 的參數必須是 object/dict")
            raw_text = str(raw_year).strip()
            try:
                if len(raw_text) == 6 and raw_text.isdigit():
                    year = int(raw_text[:4])
                    month = int(raw_text[4:])
                    effective_date = date(year, month, 1)
                    effective_end = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
                else:
                    year = int(raw_text)
                    effective_date = date(year, 1, 1)
                    effective_end = date(year, 12, 31)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"OOS 期間必須是 YYYY 或 YYYYMM，收到: {raw_year!r}") from exc
            records.append({
                "effective_date": effective_date,
                "effective_date_text": effective_date.isoformat(),
                "effective_end_date": effective_end,
                "effective_end_date_text": effective_end.isoformat(),
                "year": int(year),
                "params": dict(params),
            })
    records.sort(key=lambda item: item["effective_date"])
    if not records:
        raise ValueError("rolling OOS 參數組沒有任何可用 active param")
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


def get_active_param_record_for_date(payload: Mapping[str, Any], trade_date: Any) -> dict:
    schedule = build_active_param_schedule(payload)
    current_date = _coerce_trade_date(trade_date)
    selected = None
    for record in schedule:
        if record["effective_date"] <= current_date:
            selected = record
        else:
            break
    if selected is None:
        first_date = schedule[0]["effective_date_text"]
        raise KeyError(f"{current_date.isoformat()} 早於 rolling OOS 第一個參數生效日 {first_date}")
    return dict(selected)


def get_active_params_for_date(payload: Mapping[str, Any], trade_date: Any) -> dict:
    return copy.deepcopy(get_active_param_record_for_date(payload, trade_date)["params"])


def get_active_param_date_range(payload: Mapping[str, Any]) -> tuple[str, str]:
    schedule = build_active_param_schedule(payload)
    first_date = schedule[0]["effective_date"]
    last_end = schedule[-1].get("effective_end_date") or schedule[-1]["effective_date"]
    return first_date.isoformat(), last_end.isoformat()


def get_active_param_year_range(payload: Mapping[str, Any]) -> tuple[int, int]:
    first_date, last_date = get_active_param_date_range(payload)
    return int(first_date[:4]), int(last_date[:4])


def is_rolling_oos_param_set_file(path: str | os.PathLike[str]) -> bool:
    try:
        return is_rolling_oos_param_set_payload(load_json_file(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return False


def get_rolling_oos_params_for_year(payload: Mapping[str, Any], year: int) -> dict:
    return get_active_params_for_date(payload, date(int(year), 1, 1))


def build_rolling_oos_display_label(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or "rolling").strip()
    summary = payload.get("summary") or {}
    oos_period = str(summary.get("oos_period") or payload.get("oos_period") or "").strip()
    basename = os.path.basename(str(path))
    if oos_period:
        return f"rolling_oos_{selector} | active-param replay | OOS {oos_period} | {basename}"
    return f"rolling_oos_{selector} | active-param replay | {basename}"


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:.2f}%"


def _rolling_chained_oos_available(chained: Mapping[str, Any]) -> bool:
    if not isinstance(chained, Mapping):
        return False
    return chained.get("available") is not False


def format_rolling_oos_summary_lines(payload: Mapping[str, Any]) -> list[str]:
    selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or "-")
    summary = payload.get("summary") or {}
    chained = payload.get("chained_oos") or summary.get("chained_oos") or {}
    lines = [
        "參數模式：Rolling OOS active-param replay 參數組",
        "用途：歷史 active-param replay；模擬口徑與實際操作一致，每日使用該日期已生效的 active param",
        f"selector：{selector}",
        f"OOS：{summary.get('oos_period') or chained.get('oos_period') or '-'}",
    ]
    if chained:
        lines.append(f"串連方式：{chained.get('method', '-')}" )
        if not _rolling_chained_oos_available(chained):
            reason = str(chained.get("unavailable_reason") or "unavailable").strip()
            lines.append(f"策略串連報酬：N/A（{reason}）")
        else:
            if "rank_1_return_pct" in chained:
                lines.append(f"策略串連報酬：{_fmt_pct(chained.get('rank_1_return_pct'))}")
            if "benchmark_return_pct" in chained:
                lines.append(f"0050 串連報酬：{_fmt_pct(chained.get('benchmark_return_pct'))}")
            if "alpha_return_pct" in chained:
                lines.append(f"相對 0050：{_fmt_pct(chained.get('alpha_return_pct'))}")
    try:
        schedule = build_active_param_schedule(payload)
    except ValueError:
        schedule = []
    if schedule:
        lines.append(
            f"active param 生效日：{schedule[0]['effective_date_text']}~{schedule[-1].get('effective_end_date_text') or schedule[-1]['effective_date_text']}（共 {len(schedule)} 版）"
        )
    return lines
