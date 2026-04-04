from dataclasses import MISSING, dataclass, fields
from typing import Any

from config.runtime_defaults import RUNTIME_PARAM_DEFAULTS, RUNTIME_PARAM_SPECS


BREAKOUT_PARAM_SPECS = {
    "high_len": {"type": int, "default": 201, "min_value": 1},
    "atr_len": {"type": int, "default": 14, "min_value": 1},
    "atr_buy_tol": {"type": float, "default": 1.5, "min_value": 0.0},
    "atr_times_init": {"type": float, "default": 2.0, "min_value": 0.0, "strict_gt": True},
    "atr_times_trail": {"type": float, "default": 3.5, "min_value": 0.0, "strict_gt": True},
    "tp_percent": {"type": float, "default": 0.5, "min_value": 0.0, "max_value": 1.0, "max_exclusive": True},
    "use_bb": {"type": bool, "default": True},
    "bb_len": {"type": int, "default": 20, "min_value": 1},
    "bb_mult": {"type": float, "default": 2.0, "min_value": 0.0, "strict_gt": True},
    "use_kc": {"type": bool, "default": False},
    "kc_len": {"type": int, "default": 20, "min_value": 1},
    "kc_mult": {"type": float, "default": 2.0, "min_value": 0.0, "strict_gt": True},
    "use_vol": {"type": bool, "default": True},
    "vol_short_len": {"type": int, "default": 5, "min_value": 1},
    "vol_long_len": {"type": int, "default": 19, "min_value": 1},
    "initial_capital": {"type": float, "default": 1000000.0, "min_value": 0.0, "strict_gt": True},
    "fixed_risk": {"type": float, "default": 0.01, "min_value": 0.0, "strict_gt": True, "max_value": 1.0},
    "buy_fee": {"type": float, "default": 0.001425 * 0.28, "min_value": 0.0},
    "sell_fee": {"type": float, "default": 0.001425 * 0.28, "min_value": 0.0},
    "tax_rate": {"type": float, "default": 0.003, "min_value": 0.0},
    "min_fee": {"type": float, "default": 20.0, "min_value": 0.0},
    "use_compounding": {"type": bool, "default": True},
    "min_history_trades": {"type": int, "default": 0, "min_value": 0},
    "min_history_ev": {"type": float, "default": 0.0},
    "min_history_win_rate": {"type": float, "default": 0.30, "min_value": 0.0, "max_value": 1.0},
}


def _build_rule_text(spec):
    min_value = spec.get("min_value")
    max_value = spec.get("max_value")
    strict_gt = spec.get("strict_gt", False)
    max_exclusive = spec.get("max_exclusive", False)
    rules = []
    if min_value is not None:
        rules.append(f"需 {'>' if strict_gt else '>='} {min_value:g}")
    if max_value is not None:
        rules.append(f"需 {'<' if max_exclusive else '<='} {max_value:g}")
    return " 且 ".join(rules) if rules else "需符合策略參數規格"


# 單一真理來源：breakout 策略 defaults / guardrail 由同一份 spec 驅動

def validate_strategy_param_ranges(param_values):
    for field_name, spec in BREAKOUT_PARAM_SPECS.items():
        value = param_values[field_name]
        min_value = spec.get("min_value")
        max_value = spec.get("max_value")
        strict_gt = spec.get("strict_gt", False)
        max_exclusive = spec.get("max_exclusive", False)

        if min_value is not None:
            if strict_gt:
                condition = value > min_value
            else:
                condition = value >= min_value
            if not condition:
                raise ValueError(f"參數 {field_name} 驗證失敗: {_build_rule_text(spec)}，收到 {value!r}")

        if max_value is not None:
            if max_exclusive:
                condition = value < max_value
            else:
                condition = value <= max_value
            if not condition:
                raise ValueError(f"參數 {field_name} 驗證失敗: {_build_rule_text(spec)}，收到 {value!r}")

    if param_values["vol_long_len"] < param_values["vol_short_len"]:
        raise ValueError(
            f"參數 vol_long_len 驗證失敗: 需 >= vol_short_len，收到 {param_values['vol_long_len']!r} < {param_values['vol_short_len']!r}"
        )
    return param_values


# runtime 工具參數獨立驗證，但仍集中在參數契約這個單一真理來源

def normalize_runtime_param_value(field_name: str, raw_value: Any):
    spec = RUNTIME_PARAM_SPECS[field_name]
    expected_type = spec["type"]
    allow_none = spec["allow_none"]
    min_value = spec["min_value"]

    if raw_value is None:
        if allow_none:
            return None
        raise ValueError(f"參數 {field_name} 不可為 None")

    if expected_type is int:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"參數 {field_name} 需要 int 或 None，收到 {raw_value!r}")
        if raw_value < min_value:
            raise ValueError(f"參數 {field_name} 驗證失敗: 需 >= {min_value:g}，收到 {raw_value!r}")
        return raw_value

    if expected_type is float:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"參數 {field_name} 需要 float，收到 {raw_value!r}")
        normalized = float(raw_value)
        if normalized <= min_value:
            raise ValueError(f"參數 {field_name} 驗證失敗: 需 > {min_value:g}，收到 {raw_value!r}")
        return normalized

    raise TypeError(f"未知 runtime 參數型別: {field_name} -> {expected_type!r}")


# dataclass 直接建構 / setattr 也必須走同一套型別 guardrail

def normalize_strategy_param_value(field_name: str, raw_value: Any, expected_type: Any):
    if expected_type is bool:
        if isinstance(raw_value, bool):
            return raw_value
        raise ValueError(f"參數 {field_name} 需要 bool，收到 {raw_value!r}")

    if expected_type is int:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"參數 {field_name} 需要 int，收到 {raw_value!r}")
        return raw_value

    if expected_type is float:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"參數 {field_name} 需要 float，收到 {raw_value!r}")
        return float(raw_value)

    if not isinstance(raw_value, expected_type):
        raise ValueError(f"參數 {field_name} 需要 {expected_type.__name__}，收到 {raw_value!r}")

    return raw_value


@dataclass
class V16StrategyParams:
    high_len: int = BREAKOUT_PARAM_SPECS["high_len"]["default"]
    atr_len: int = BREAKOUT_PARAM_SPECS["atr_len"]["default"]
    atr_buy_tol: float = BREAKOUT_PARAM_SPECS["atr_buy_tol"]["default"]
    atr_times_init: float = BREAKOUT_PARAM_SPECS["atr_times_init"]["default"]
    atr_times_trail: float = BREAKOUT_PARAM_SPECS["atr_times_trail"]["default"]
    tp_percent: float = BREAKOUT_PARAM_SPECS["tp_percent"]["default"]
    use_bb: bool = BREAKOUT_PARAM_SPECS["use_bb"]["default"]
    bb_len: int = BREAKOUT_PARAM_SPECS["bb_len"]["default"]
    bb_mult: float = BREAKOUT_PARAM_SPECS["bb_mult"]["default"]
    use_kc: bool = BREAKOUT_PARAM_SPECS["use_kc"]["default"]
    kc_len: int = BREAKOUT_PARAM_SPECS["kc_len"]["default"]
    kc_mult: float = BREAKOUT_PARAM_SPECS["kc_mult"]["default"]
    use_vol: bool = BREAKOUT_PARAM_SPECS["use_vol"]["default"]
    vol_short_len: int = BREAKOUT_PARAM_SPECS["vol_short_len"]["default"]
    vol_long_len: int = BREAKOUT_PARAM_SPECS["vol_long_len"]["default"]
    initial_capital: float = BREAKOUT_PARAM_SPECS["initial_capital"]["default"]
    fixed_risk: float = BREAKOUT_PARAM_SPECS["fixed_risk"]["default"]
    buy_fee: float = BREAKOUT_PARAM_SPECS["buy_fee"]["default"]
    sell_fee: float = BREAKOUT_PARAM_SPECS["sell_fee"]["default"]
    tax_rate: float = BREAKOUT_PARAM_SPECS["tax_rate"]["default"]
    min_fee: float = BREAKOUT_PARAM_SPECS["min_fee"]["default"]
    use_compounding: bool = BREAKOUT_PARAM_SPECS["use_compounding"]["default"]
    min_history_trades: int = BREAKOUT_PARAM_SPECS["min_history_trades"]["default"]
    min_history_ev: float = BREAKOUT_PARAM_SPECS["min_history_ev"]["default"]
    min_history_win_rate: float = BREAKOUT_PARAM_SPECS["min_history_win_rate"]["default"]

    def __post_init__(self):
        snapshot = _build_strategy_param_snapshot(self)
        field_types = {field.name: field.type for field in fields(type(self))}

        for field_name, expected_type in field_types.items():
            normalized_value = normalize_strategy_param_value(field_name, snapshot[field_name], expected_type)
            object.__setattr__(self, field_name, normalized_value)
            snapshot[field_name] = normalized_value

        validate_strategy_param_ranges(snapshot)

    def __setattr__(self, name, value):
        field_map = type(self).__dataclass_fields__

        if name in RUNTIME_PARAM_DEFAULTS:
            normalized_value = normalize_runtime_param_value(name, value)
            object.__setattr__(self, name, normalized_value)
            return

        if name not in field_map:
            raise AttributeError(f"未知參數欄位: {name}")

        expected_type = field_map[name].type
        normalized_value = normalize_strategy_param_value(name, value, expected_type)

        had_old_value = hasattr(self, name)
        old_value = getattr(self, name) if had_old_value else MISSING
        object.__setattr__(self, name, normalized_value)

        try:
            validate_strategy_param_ranges(_build_strategy_param_snapshot(self))
        except ValueError:
            if had_old_value:
                object.__setattr__(self, name, old_value)
            elif hasattr(self, name):
                object.__delattr__(self, name)
            raise


def strategy_params_to_dict(params, include_runtime=False):
    payload = {field.name: getattr(params, field.name) for field in fields(V16StrategyParams)}
    if include_runtime:
        for field_name in RUNTIME_PARAM_DEFAULTS:
            if hasattr(params, field_name):
                payload[field_name] = getattr(params, field_name)
    return payload


def build_runtime_param_raw_value(params, field_name):
    if isinstance(params, dict):
        return params.get(field_name, RUNTIME_PARAM_DEFAULTS[field_name])
    return getattr(params, field_name, RUNTIME_PARAM_DEFAULTS[field_name])


def _build_strategy_param_snapshot(instance):
    snapshot = {}
    for field in fields(type(instance)):
        if hasattr(instance, field.name):
            snapshot[field.name] = getattr(instance, field.name)
        elif field.default is not MISSING:
            snapshot[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            snapshot[field.name] = field.default_factory()
    return snapshot
