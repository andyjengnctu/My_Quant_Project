"""共用執行政策：資金、費用、複利與 runtime 執行預設。"""

from decimal import Decimal

from config.broker_accounting import (
    BROKER_ELECTRONIC_FEE_EFFECTIVE_RATIO,
    BROKER_FEE_RATE,
    BROKER_MIN_FEE_TWD,
    BROKER_STOCK_TAX_RATE,
)

# Registered Research scientific contracts are frozen under the historical
# 72%-rebate assumption. Their default parameters cannot be changed in place
# without changing the scientific fitting identity and invalidating REUSE.
# Yuanta's actual six-tenths policy has a separate explicit opt-in definition;
# a new registered Research identity is required before making it current.
RESEARCH_FEE_EFFECTIVE_RATIO = Decimal("0.28")
DEFAULT_FEE_RATE = float(BROKER_FEE_RATE * RESEARCH_FEE_EFFECTIVE_RATIO)
YUANTA_RESEARCH_FEE_EFFECTIVE_RATIO = BROKER_ELECTRONIC_FEE_EFFECTIVE_RATIO
YUANTA_RESEARCH_FEE_RATE = float(BROKER_FEE_RATE * YUANTA_RESEARCH_FEE_EFFECTIVE_RATIO)


def build_yuanta_research_execution_overrides() -> dict[str, float]:
    """Explicit result-affecting inputs for a *new* Research scientific identity.

    Registered historical profiles must reject this changed contract until a
    separate Research identity is authorized, rather than reinterpret P6/P14.
    """
    return {
        "buy_fee": YUANTA_RESEARCH_FEE_RATE,
        "sell_fee": YUANTA_RESEARCH_FEE_RATE,
        "tax_rate": float(BROKER_STOCK_TAX_RATE),
        "min_fee": float(BROKER_MIN_FEE_TWD),
    }

# Canonical portfolio execution defaults.  Other configs may expose aliases, but
# these literals have one owner here.
DEFAULT_PORTFOLIO_MAX_POSITIONS = 10
DEFAULT_PORTFOLIO_ROTATION = "off"
DEFAULT_FIXED_RISK = 0.01
DEFAULT_MAX_POSITION_CAP_PCT = 0.30

EXECUTION_POLICY_PARAM_SPECS = {
    "initial_capital": {"type": float, "default": 1_000_000.0, "min_value": 0.0, "strict_gt": True},  # 回測/投組初始本金
    "fixed_risk": {"type": float, "default": DEFAULT_FIXED_RISK, "min_value": 0.0, "strict_gt": True, "max_value": 1.0},  # 單筆固定風險比例
    "max_position_cap_pct": {"type": float, "default": DEFAULT_MAX_POSITION_CAP_PCT, "min_value": 0.0, "strict_gt": True, "max_value": 1.0},  # 單一標的最大資金占總資產比例
    "buy_fee": {"type": float, "default": DEFAULT_FEE_RATE, "min_value": 0.0},  # 買進手續費率，預設券商手續費 × 折扣
    "sell_fee": {"type": float, "default": DEFAULT_FEE_RATE, "min_value": 0.0},  # 賣出手續費率，預設券商手續費 × 折扣
    "tax_rate": {"type": float, "default": float(BROKER_STOCK_TAX_RATE), "min_value": 0.0},  # 賣出交易稅率
    "min_fee": {"type": float, "default": float(BROKER_MIN_FEE_TWD), "min_value": 0.0},  # 最低手續費金額
    "min_entry_notional": {"type": float, "default": 50_126.0, "min_value": 0.0},  # 單筆買進最低成交名目金額，避免最低手續費扭曲零碎單
    "use_compounding": {"type": bool, "default": True},  # 是否使用複利資金口徑
}

RUNTIME_PARAM_SPECS = {
    "optimizer_max_workers": {"type": int, "default": None, "allow_none": True, "min_value": 1},  # optimizer worker 數上限
    "scanner_max_workers": {"type": int, "default": None, "allow_none": True, "min_value": 1},  # scanner worker 數上限
    "scanner_live_capital": {"type": float, "default": 2_000_000.0, "allow_none": False, "min_value": 0.0},  # scanner 參考投入本金
}

RUNTIME_PARAM_DEFAULTS = {field_name: spec["default"] for field_name, spec in RUNTIME_PARAM_SPECS.items()}
RUNTIME_PARAM_TYPES = {field_name: spec["type"] for field_name, spec in RUNTIME_PARAM_SPECS.items()}

