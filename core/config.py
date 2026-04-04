# core/config.py
from dataclasses import MISSING, dataclass, fields
from typing import Any

# ==========================================
# 🌟 全域戰略切換開關 (System-Wide Strategy Switches)
# ==========================================
# 1. 期望值 (EV) 算法切換
# 'A' = 嚴格 R_Multiple 期望值 (Mean R)
# 'B' = 傳統實際盈虧期望值 (Win% * Payoff - Loss%)
EV_CALC_METHOD = 'A'

# 2. 買入優先序切換開關
# 'PROJ_COST' = 優先買入能消耗最多資金的標的 (資金效率極大化)
# 'EV'        = 優先買入期望值最高的標的 (單筆質量極大化)
# 'HIST_WIN_X_TRADES' = 優先買入歷史勝率 × 交易次數最高的標的 (穩定度 × 樣本數)
BUY_SORT_METHOD = 'HIST_WIN_X_TRADES'

# 3. 系統評分 (Score) 算法切換
# 'LOG_R2' = 結合對數 R 平方與月度勝率的不對稱模型 (容許暴漲，尋找平穩向上的聖杯)
# 'RoMD'   = 傳統報酬回撤比 (只看總報酬與最大回撤)
SCORE_CALC_METHOD = 'RoMD'

# 4-1. Optimizer 投組績效門檻 
MIN_FULL_YEAR_RETURN_PCT = -30.0     # 最低完整年度報酬率下限（%），避免某些完整年度虧損過深，預設: -20
MIN_ANNUAL_TRADES = 5.0              # 最低年化交易次數門檻，避免交易過少導致樣本不足與統計不穩定
MIN_BUY_FILL_RATE = 70.0             # 最低保留後買進成交率門檻（%），避免盤前保留很多但實際難成交，預設: 80

MIN_TRADE_WIN_RATE = 35.0            # 最低完整交易勝率門檻（%），限制 round-trip 交易勝率不可過低，預設: 40
MAX_PORTFOLIO_MDD_PCT = 45.0         # 投組最大回撤上限（%），限制整體權益曲線最大跌幅不可過大
MIN_MONTHLY_WIN_RATE = 35.0          # 最低月勝率門檻（%），避免只靠少數大賺月份撐起總報酬，預設: 45
MIN_EQUITY_CURVE_R_SQUARED = 0.40    # 權益曲線最小 R² 門檻，要求整體走勢具備基本平滑與趨勢一致性
# ==========================================

# # (AI註: 僅影響顯示，不改實際排序與優化邏輯)
SYSTEM_SCORE_DISPLAY_MULTIPLIER = 1000.0


RUNTIME_PARAM_SPECS = {
    'optimizer_max_workers': {'type': int, 'default': None, 'allow_none': True, 'min_value': 1},
    'scanner_max_workers': {'type': int, 'default': None, 'allow_none': True, 'min_value': 1},
    'scanner_live_capital': {'type': float, 'default': 2_000_000.0, 'allow_none': False, 'min_value': 0.0},
}

RUNTIME_PARAM_DEFAULTS = {field_name: spec['default'] for field_name, spec in RUNTIME_PARAM_SPECS.items()}
RUNTIME_PARAM_TYPES = {field_name: spec['type'] for field_name, spec in RUNTIME_PARAM_SPECS.items()}


def _get_runtime_param_raw_value(params, field_name):
    if isinstance(params, dict):
        return params.get(field_name, RUNTIME_PARAM_DEFAULTS[field_name])
    return getattr(params, field_name, RUNTIME_PARAM_DEFAULTS[field_name])


def _resolve_compounding_capital(value, fallback_value):
    if value is None:
        return max(0.0, float(fallback_value))
    return max(0.0, float(value))


def resolve_single_backtest_sizing_capital(params, current_capital=None):
    return _resolve_compounding_capital(current_capital, params.initial_capital)


def resolve_portfolio_sizing_equity(current_equity, initial_capital, params):
    return _resolve_compounding_capital(current_equity, initial_capital)


def resolve_portfolio_entry_budget(available_cash, initial_capital, params):
    return _resolve_compounding_capital(available_cash, initial_capital)


def resolve_scanner_live_capital(params):
    raw_value = _get_runtime_param_raw_value(params, 'scanner_live_capital')
    return normalize_runtime_param_value('scanner_live_capital', raw_value)


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


# # (AI註: runtime 工具參數獨立驗證，但仍集中在 config 這個單一真理來源)
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
