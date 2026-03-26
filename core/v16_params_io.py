import json
import os
from dataclasses import fields

from core.v16_config import V16StrategyParams
from core.v16_log_utils import format_exception_summary


PARAM_FIELD_TYPES = {field.name: field.type for field in fields(V16StrategyParams)}


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


def load_params_from_json(json_file):
    params = V16StrategyParams()
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, value in data.items():
            expected_type = PARAM_FIELD_TYPES.get(key)
            if expected_type is not None:
                setattr(params, key, _coerce_param_value(key, value, expected_type))

        return params
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"讀取參數檔 {json_file} 失敗: {format_exception_summary(e)}") from e