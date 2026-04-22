"""共用策略參數契約：聚合 breakout 專屬參數、training policy selection gate、execution policy 與 runtime。"""

from dataclasses import MISSING, dataclass, fields
from typing import Any

from config.execution_policy import EXECUTION_POLICY_PARAM_SPECS, RUNTIME_PARAM_DEFAULTS, RUNTIME_PARAM_SPECS
from config.training_policy import SELECTION_POLICY_PARAM_SPECS
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS, validate_breakout_param_ranges


STRATEGY_PARAM_SPECS = {
    **BREAKOUT_PARAM_SPECS,
    **SELECTION_POLICY_PARAM_SPECS,
    **EXECUTION_POLICY_PARAM_SPECS,
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


def _validate_spec_ranges(param_values, spec_map):
    for field_name, spec in spec_map.items():
        value = param_values[field_name]
        min_value = spec.get("min_value")
        max_value = spec.get("max_value")
        strict_gt = spec.get("strict_gt", False)
        max_exclusive = spec.get("max_exclusive", False)

        if min_value is not None:
            condition = value > min_value if strict_gt else value >= min_value
            if not condition:
                raise ValueError(f"參數 {field_name} 驗證失敗: {_build_rule_text(spec)}，收到 {value!r}")

        if max_value is not None:
            condition = value < max_value if max_exclusive else value <= max_value
            if not condition:
                raise ValueError(f"參數 {field_name} 驗證失敗: {_build_rule_text(spec)}，收到 {value!r}")


def validate_strategy_param_ranges(param_values):
    validate_breakout_param_ranges(param_values, build_rule_text=_build_rule_text)
    _validate_spec_ranges(param_values, SELECTION_POLICY_PARAM_SPECS)
    _validate_spec_ranges(param_values, EXECUTION_POLICY_PARAM_SPECS)

    if param_values.get("use_compounding") is not True:
        raise ValueError("參數 use_compounding 目前僅支援 True；正式口徑固定使用複利資金。")

    return param_values


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
    min_history_trades: int = SELECTION_POLICY_PARAM_SPECS["min_history_trades"]["default"]
    min_history_ev: float = SELECTION_POLICY_PARAM_SPECS["min_history_ev"]["default"]
    min_history_win_rate: float = SELECTION_POLICY_PARAM_SPECS["min_history_win_rate"]["default"]
    initial_capital: float = EXECUTION_POLICY_PARAM_SPECS["initial_capital"]["default"]
    fixed_risk: float = EXECUTION_POLICY_PARAM_SPECS["fixed_risk"]["default"]
    max_position_cap_pct: float = EXECUTION_POLICY_PARAM_SPECS["max_position_cap_pct"]["default"]
    buy_fee: float = EXECUTION_POLICY_PARAM_SPECS["buy_fee"]["default"]
    sell_fee: float = EXECUTION_POLICY_PARAM_SPECS["sell_fee"]["default"]
    tax_rate: float = EXECUTION_POLICY_PARAM_SPECS["tax_rate"]["default"]
    min_fee: float = EXECUTION_POLICY_PARAM_SPECS["min_fee"]["default"]
    use_compounding: bool = EXECUTION_POLICY_PARAM_SPECS["use_compounding"]["default"]

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


__all__ = [
    "BREAKOUT_PARAM_SPECS",
    "EXECUTION_POLICY_PARAM_SPECS",
    "SELECTION_POLICY_PARAM_SPECS",
    "STRATEGY_PARAM_SPECS",
    "V16StrategyParams",
    "build_runtime_param_raw_value",
    "normalize_runtime_param_value",
    "normalize_strategy_param_value",
    "strategy_params_to_dict",
    "validate_strategy_param_ranges",
]
