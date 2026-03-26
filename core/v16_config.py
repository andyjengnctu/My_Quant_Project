# core/v16_config.py
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
MIN_FULL_YEAR_RETURN_PCT = -20.0     # 最低完整年度報酬率下限（%），避免某些完整年度虧損過深
MIN_ANNUAL_TRADES = 5.0              # 最低年化交易次數門檻，避免交易過少導致樣本不足與統計不穩定
MIN_BUY_FILL_RATE = 80.0             # 最低保留後買進成交率門檻（%），避免盤前保留很多但實際難成交

MIN_TRADE_WIN_RATE = 40.0            # 最低完整交易勝率門檻（%），限制 round-trip 交易勝率不可過低
MAX_PORTFOLIO_MDD_PCT = 45.0         # 投組最大回撤上限（%），限制整體權益曲線最大跌幅不可過大
MIN_MONTHLY_WIN_RATE = 45.0          # 最低月勝率門檻（%），避免只靠少數大賺月份撐起總報酬
MIN_EQUITY_CURVE_R_SQUARED = 0.40    # 權益曲線最小 R² 門檻，要求整體走勢具備基本平滑與趨勢一致性
# ==========================================

# # (AI註: 僅影響顯示，不改實際排序與優化邏輯)
SYSTEM_SCORE_DISPLAY_MULTIPLIER = 1000.0


RUNTIME_PARAM_DEFAULTS = {
    'optimizer_max_workers': None,
    'scanner_max_workers': None,
}


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
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"參數 {field_name} 需要 int 或 None，收到 {raw_value!r}")
    if raw_value < 1:
        raise ValueError(f"參數 {field_name} 驗證失敗: 需 >= 1，收到 {raw_value!r}")
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
        validate_strategy_param_ranges(strategy_params_to_dict(self))

    def __setattr__(self, name, value):
        field_names = type(self).__dataclass_fields__

        if name in RUNTIME_PARAM_DEFAULTS:
            normalized_value = normalize_runtime_param_value(name, value)
            object.__setattr__(self, name, normalized_value)
            return

        if name not in field_names:
            raise AttributeError(f"未知參數欄位: {name}")

        had_old_value = hasattr(self, name)
        old_value = getattr(self, name) if had_old_value else MISSING
        object.__setattr__(self, name, value)

        try:
            validate_strategy_param_ranges(_build_strategy_param_snapshot(self))
        except Exception:
            if had_old_value:
                object.__setattr__(self, name, old_value)
            else:
                try:
                    object.__delattr__(self, name)
                except AttributeError:
                    pass
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
