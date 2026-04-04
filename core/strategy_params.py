from dataclasses import MISSING, dataclass, fields
from typing import Any

from config.runtime_defaults import (
    RUNTIME_PARAM_DEFAULTS,
    RUNTIME_PARAM_SPECS,
    RUNTIME_PARAM_TYPES,
)


# # (AI註: 單一真理來源 - 直接建立 dataclass、JSON 載入、optimizer 固定覆寫都共用同一套數值 guardrail)
def validate_strategy_param_ranges(param_values):
    def ensure(field_name, condition, rule_text):
        if not condition:
            raise ValueError(f"參數 {field_name} 驗證失敗: {rule_text}，收到 {param_values[field_name]!r}")

    ensure('high_len', param_values['high_len'] >= 1, '需 >= 1')
    ensure('atr_len', param_values['atr_len'] >= 1, '需 >= 1')
    ensure('atr_buy_tol', param_values['atr_buy_tol'] >= 0.0, '需 >= 0')
    ensure('atr_times_init', param_values['atr_times_init'] > 0.0, '需 > 0')
    ensure('atr_times_trail', param_values['atr_times_trail'] > 0.0, '需 > 0')
    ensure('tp_percent', 0.0 <= param_values['tp_percent'] < 1.0, '需滿足 0 <= tp_percent < 1')
    ensure('bb_len', param_values['bb_len'] >= 1, '需 >= 1')
    ensure('bb_mult', param_values['bb_mult'] > 0.0, '需 > 0')
    ensure('kc_len', param_values['kc_len'] >= 1, '需 >= 1')
    ensure('kc_mult', param_values['kc_mult'] > 0.0, '需 > 0')
    ensure('vol_short_len', param_values['vol_short_len'] >= 1, '需 >= 1')
    ensure('vol_long_len', param_values['vol_long_len'] >= 1, '需 >= 1')
    ensure('vol_long_len', param_values['vol_long_len'] >= param_values['vol_short_len'], '需 >= vol_short_len')
    ensure('initial_capital', param_values['initial_capital'] > 0.0, '需 > 0')
    ensure('fixed_risk', 0.0 < param_values['fixed_risk'] <= 1.0, '需滿足 0 < fixed_risk <= 1')
    ensure('buy_fee', param_values['buy_fee'] >= 0.0, '需 >= 0')
    ensure('sell_fee', param_values['sell_fee'] >= 0.0, '需 >= 0')
    ensure('tax_rate', param_values['tax_rate'] >= 0.0, '需 >= 0')
    ensure('min_fee', param_values['min_fee'] >= 0.0, '需 >= 0')
    ensure('min_history_trades', param_values['min_history_trades'] >= 0, '需 >= 0')
    ensure('min_history_win_rate', 0.0 <= param_values['min_history_win_rate'] <= 1.0, '需滿足 0 <= min_history_win_rate <= 1')
    return param_values


# # (AI註: runtime 工具參數獨立驗證，但仍集中在參數契約這個單一真理來源)
def normalize_runtime_param_value(field_name: str, raw_value: Any):
    spec = RUNTIME_PARAM_SPECS[field_name]
    expected_type = spec['type']
    allow_none = spec['allow_none']
    min_value = spec['min_value']

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


# # (AI註: dataclass 直接建構 / 直接 setattr 也必須走同一套型別 guardrail，避免錯型別延後到策略流程才爆炸)
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
    # 1. 核心指標參數
    high_len: int = 201
    atr_len: int = 14

    # 2. 停損停利與進場風控
    atr_buy_tol: float = 1.5
    atr_times_init: float = 2.0
    atr_times_trail: float = 3.5
    tp_percent: float = 0.5

    # 3. 三大濾網開關與參數
    use_bb: bool = True
    bb_len: int = 20
    bb_mult: float = 2.0

    use_kc: bool = False
    kc_len: int = 20
    kc_mult: float = 2.0

    use_vol: bool = True
    vol_short_len: int = 5
    vol_long_len: int = 19

    # 4. 資金管理參數
    initial_capital: float = 1000000
    fixed_risk: float = 0.01
    buy_fee: float = 0.001425 * 0.28
    sell_fee: float = 0.001425 * 0.28
    tax_rate: float = 0.003
    min_fee: float = 20
    use_compounding: bool = True

    # 5. 歷史績效濾網 (AI 將接管這些設定)
    min_history_trades: int = 0
    min_history_ev: float = 0.0
    min_history_win_rate: float = 0.30

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


# # (AI註: 統一由 dataclass 欄位快照成 dict，避免 params_io / optimizer 各自手抄欄位)
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
