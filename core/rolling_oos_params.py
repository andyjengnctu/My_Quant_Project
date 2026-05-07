from __future__ import annotations

import json
import os
from typing import Any, Mapping

ROLLING_OOS_PARAM_SET_SCHEMA_TYPE = "rolling_oos_param_set"
ROLLING_OOS_USAGE = "validation_only"
ROLLING_OOS_POLICY_NAMES = ("base", "local", "retention")


def is_rolling_oos_param_set_payload(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    schema_type = str(payload.get("schema_type") or payload.get("type") or "").strip()
    if schema_type == ROLLING_OOS_PARAM_SET_SCHEMA_TYPE:
        return True
    if schema_type == "outer_rolling_oos_param_set":
        return True
    return isinstance(payload.get("params_by_oos_year"), Mapping) and str(payload.get("usage") or "").strip() == ROLLING_OOS_USAGE


def load_json_file(path: str | os.PathLike[str]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根層必須是 object/dict，收到 {type(payload).__name__}")
    return payload


def load_rolling_oos_param_set(path: str | os.PathLike[str]) -> dict:
    payload = load_json_file(path)
    if not is_rolling_oos_param_set_payload(payload):
        raise ValueError(f"不是 rolling OOS 年度參數組 JSON: {path}")
    params_by_year = payload.get("params_by_oos_year")
    if not isinstance(params_by_year, Mapping) or not params_by_year:
        raise ValueError(f"rolling OOS 參數組缺少 params_by_oos_year: {path}")
    return payload


def is_rolling_oos_param_set_file(path: str | os.PathLike[str]) -> bool:
    try:
        return is_rolling_oos_param_set_payload(load_json_file(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return False


def get_rolling_oos_params_for_year(payload: Mapping[str, Any], year: int) -> dict:
    params_by_year = payload.get("params_by_oos_year") or {}
    if not isinstance(params_by_year, Mapping):
        raise ValueError("rolling OOS 參數組 params_by_oos_year 必須是 object/dict")
    key = str(int(year))
    params = params_by_year.get(key)
    if params is None:
        available = ", ".join(sorted(str(k) for k in params_by_year.keys()))
        raise KeyError(f"rolling OOS 參數組沒有 {key} 年參數；可用年份: {available}")
    if not isinstance(params, Mapping):
        raise ValueError(f"rolling OOS 參數 {key} 必須是 object/dict")
    return dict(params)


def build_rolling_oos_display_label(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or "rolling").strip()
    summary = payload.get("summary") or {}
    oos_period = str(summary.get("oos_period") or payload.get("oos_period") or "").strip()
    basename = os.path.basename(str(path))
    if oos_period:
        return f"rolling_oos_{selector} | validation only | OOS {oos_period} | {basename}"
    return f"rolling_oos_{selector} | validation only | {basename}"


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:.2f}%"


def format_rolling_oos_summary_lines(payload: Mapping[str, Any]) -> list[str]:
    selector = str(payload.get("selector") or (payload.get("meta") or {}).get("selector") or "-")
    summary = payload.get("summary") or {}
    chained = payload.get("chained_oos") or summary.get("chained_oos") or {}
    lines = [
        "參數模式：Rolling OOS 驗證參數組",
        "用途：validation only，不是實盤單一 param.json",
        f"selector：{selector}",
        f"OOS：{summary.get('oos_period') or chained.get('oos_period') or '-'}",
    ]
    if chained:
        lines.append(f"串連方式：{chained.get('method', '-')}" )
        if "rank_1_return_pct" in chained:
            lines.append(f"策略串連報酬：{_fmt_pct(chained.get('rank_1_return_pct'))}")
        if "benchmark_return_pct" in chained:
            lines.append(f"0050 串連報酬：{_fmt_pct(chained.get('benchmark_return_pct'))}")
        if "alpha_return_pct" in chained:
            lines.append(f"相對 0050：{_fmt_pct(chained.get('alpha_return_pct'))}")
    params_by_year = payload.get("params_by_oos_year") or {}
    if isinstance(params_by_year, Mapping):
        years = sorted(str(year) for year in params_by_year.keys())
        if years:
            lines.append(f"年度參數：{years[0]}~{years[-1]}（共 {len(years)} 年）")
    return lines
