import json
import os
from dataclasses import fields

from core.strategy_params import (
    V16StrategyParams,
    normalize_runtime_param_value,
    strategy_params_to_dict,
    validate_strategy_param_ranges,
)
from config.execution_policy import RUNTIME_PARAM_DEFAULTS, RUNTIME_PARAM_TYPES
from core.log_utils import format_exception_summary


PARAM_FIELDS = tuple(fields(V16StrategyParams))
PARAM_FIELD_TYPES = {field.name: field.type for field in PARAM_FIELDS}
PARAM_FIELD_NAMES = tuple(field.name for field in PARAM_FIELDS)
RUNTIME_PARAM_NAMES = tuple(RUNTIME_PARAM_DEFAULTS)


# # (AI註: 參數載入時先做型別收斂，避免錯型別延後到回測/優化流程才爆炸)
def _coerce_param_value(field_name, raw_value, expected_type):
    if expected_type is bool:
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in ("true", "1", "yes", "y", "on"):
                return True
            if normalized in ("false", "0", "no", "n", "off"):
                return False
        raise ValueError(f"參數 {field_name} 需要 bool，收到 {raw_value!r}")

    if expected_type is int:
        if isinstance(raw_value, bool):
            raise ValueError(f"參數 {field_name} 需要 int，收到 {raw_value!r}")
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            if raw_value.is_integer():
                return int(raw_value)
            raise ValueError(f"參數 {field_name} 需要整數，收到 {raw_value!r}")
        if isinstance(raw_value, str):
            text = raw_value.strip()
            try:
                return int(text)
            except ValueError:
                try:
                    numeric_value = float(text)
                except ValueError as e:
                    raise ValueError(f"參數 {field_name} 需要 int，收到 {raw_value!r}") from e
                if numeric_value.is_integer():
                    return int(numeric_value)
                raise ValueError(f"參數 {field_name} 需要整數，收到 {raw_value!r}")
        raise ValueError(f"參數 {field_name} 需要 int，收到 {raw_value!r}")

    if expected_type is float:
        if isinstance(raw_value, bool):
            raise ValueError(f"參數 {field_name} 需要 float，收到 {raw_value!r}")
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            text = raw_value.strip()
            try:
                return float(text)
            except ValueError as e:
                raise ValueError(f"參數 {field_name} 需要 float，收到 {raw_value!r}") from e
        raise ValueError(f"參數 {field_name} 需要 float，收到 {raw_value!r}")

    if not isinstance(raw_value, expected_type):
        raise ValueError(f"參數 {field_name} 需要 {expected_type.__name__}，收到 {raw_value!r}")

    return raw_value


def _validate_param_payload(data):
    if not isinstance(data, dict):
        raise ValueError(f"參數檔根層必須是 object/dict，收到 {type(data).__name__}")

    allowed_keys = set(PARAM_FIELD_NAMES) | set(RUNTIME_PARAM_NAMES)
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"參數檔含未知欄位: {unknown_keys}")

    missing_keys = [field_name for field_name in PARAM_FIELD_NAMES if field_name not in data]
    if missing_keys:
        raise ValueError(f"參數檔缺少必要欄位: {missing_keys}")


def params_to_json_dict(params):
    return strategy_params_to_dict(params, include_runtime=True)


def build_params_from_mapping(data):
    _validate_param_payload(data)
    coerced_values = {}
    for field_name in PARAM_FIELD_NAMES:
        coerced_values[field_name] = _coerce_param_value(
            field_name,
            data[field_name],
            PARAM_FIELD_TYPES[field_name],
        )
    validate_strategy_param_ranges(coerced_values)
    params = V16StrategyParams(**coerced_values)

    for field_name in RUNTIME_PARAM_NAMES:
        if field_name not in data:
            continue
        raw_value = data[field_name]
        if raw_value is None:
            setattr(params, field_name, None)
            continue
        coerced_value = _coerce_param_value(
            field_name,
            raw_value,
            RUNTIME_PARAM_TYPES[field_name],
        )
        setattr(params, field_name, normalize_runtime_param_value(field_name, coerced_value))

    return params


def load_params_from_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return build_params_from_mapping(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"讀取參數檔 {json_file} 失敗: {format_exception_summary(e, include_traceback=False)}") from e
