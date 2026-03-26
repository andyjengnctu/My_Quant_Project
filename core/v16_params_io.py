import json
import os
from dataclasses import fields

from core.v16_config import V16StrategyParams
from core.v16_log_utils import format_exception_summary


PARAM_FIELDS = tuple(fields(V16StrategyParams))
PARAM_FIELD_TYPES = {field.name: field.type for field in PARAM_FIELDS}
PARAM_FIELD_NAMES = tuple(field.name for field in PARAM_FIELDS)


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

    unknown_keys = sorted(set(data) - set(PARAM_FIELD_NAMES))
    if unknown_keys:
        raise ValueError(f"參數檔含未知欄位: {unknown_keys}")

    missing_keys = [field_name for field_name in PARAM_FIELD_NAMES if field_name not in data]
    if missing_keys:
        raise ValueError(f"參數檔缺少必要欄位: {missing_keys}")


def _validate_param_ranges(coerced_values):
    def ensure(field_name, condition, rule_text):
        if not condition:
            raise ValueError(f"參數 {field_name} 驗證失敗: {rule_text}，收到 {coerced_values[field_name]!r}")

    ensure('high_len', coerced_values['high_len'] >= 1, '需 >= 1')
    ensure('atr_len', coerced_values['atr_len'] >= 1, '需 >= 1')
    ensure('atr_buy_tol', coerced_values['atr_buy_tol'] >= 0.0, '需 >= 0')
    ensure('atr_times_init', coerced_values['atr_times_init'] > 0.0, '需 > 0')
    ensure('atr_times_trail', coerced_values['atr_times_trail'] > 0.0, '需 > 0')
    ensure('tp_percent', 0.0 <= coerced_values['tp_percent'] < 1.0, '需滿足 0 <= tp_percent < 1')
    ensure('bb_len', coerced_values['bb_len'] >= 1, '需 >= 1')
    ensure('bb_mult', coerced_values['bb_mult'] > 0.0, '需 > 0')
    ensure('kc_len', coerced_values['kc_len'] >= 1, '需 >= 1')
    ensure('kc_mult', coerced_values['kc_mult'] > 0.0, '需 > 0')
    ensure('vol_short_len', coerced_values['vol_short_len'] >= 1, '需 >= 1')
    ensure('vol_long_len', coerced_values['vol_long_len'] >= 1, '需 >= 1')
    ensure('vol_long_len', coerced_values['vol_long_len'] >= coerced_values['vol_short_len'], '需 >= vol_short_len')
    ensure('initial_capital', coerced_values['initial_capital'] > 0.0, '需 > 0')
    ensure('fixed_risk', 0.0 < coerced_values['fixed_risk'] <= 1.0, '需滿足 0 < fixed_risk <= 1')
    ensure('buy_fee', coerced_values['buy_fee'] >= 0.0, '需 >= 0')
    ensure('sell_fee', coerced_values['sell_fee'] >= 0.0, '需 >= 0')
    ensure('tax_rate', coerced_values['tax_rate'] >= 0.0, '需 >= 0')
    ensure('min_fee', coerced_values['min_fee'] >= 0.0, '需 >= 0')
    ensure('min_history_trades', coerced_values['min_history_trades'] >= 0, '需 >= 0')
    ensure('min_history_win_rate', 0.0 <= coerced_values['min_history_win_rate'] <= 1.0, '需滿足 0 <= min_history_win_rate <= 1')


def params_to_json_dict(params):
    return {field_name: getattr(params, field_name) for field_name in PARAM_FIELD_NAMES}


def build_params_from_mapping(data):
    _validate_param_payload(data)
    coerced_values = {}
    for field_name in PARAM_FIELD_NAMES:
        coerced_values[field_name] = _coerce_param_value(
            field_name,
            data[field_name],
            PARAM_FIELD_TYPES[field_name],
        )
    _validate_param_ranges(coerced_values)
    return V16StrategyParams(**coerced_values)


def load_params_from_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return build_params_from_mapping(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"讀取參數檔 {json_file} 失敗: {format_exception_summary(e)}") from e
