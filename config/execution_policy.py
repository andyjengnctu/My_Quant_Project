"""共用執行政策：資金、費用、複利與 runtime 執行預設。"""

EXECUTION_POLICY_PARAM_SPECS = {
    "initial_capital": {"type": float, "default": 1_000_000.0, "min_value": 0.0, "strict_gt": True},  # (AI註: 回測/投組初始本金，預設 1,000,000.0)
    "fixed_risk": {"type": float, "default": 0.01, "min_value": 0.0, "strict_gt": True, "max_value": 1.0},  # (AI註: 單筆固定風險比例，預設 0.01)
    "buy_fee": {"type": float, "default": 0.001425 * 0.28, "min_value": 0.0},  # (AI註: 買進手續費率，預設 0.001425 * 0.28)
    "sell_fee": {"type": float, "default": 0.001425 * 0.28, "min_value": 0.0},  # (AI註: 賣出手續費率，預設 0.001425 * 0.28)
    "tax_rate": {"type": float, "default": 0.003, "min_value": 0.0},  # (AI註: 賣出交易稅率，預設 0.003)
    "min_fee": {"type": float, "default": 20.0, "min_value": 0.0},  # (AI註: 最低手續費金額，預設 20.0)
    "use_compounding": {"type": bool, "default": True},  # (AI註: 是否使用複利資金口徑，預設 True)
}

RUNTIME_PARAM_SPECS = {
    "optimizer_max_workers": {"type": int, "default": None, "allow_none": True, "min_value": 1},  # (AI註: optimizer worker 數上限，預設 None)
    "scanner_max_workers": {"type": int, "default": None, "allow_none": True, "min_value": 1},  # (AI註: scanner worker 數上限，預設 None)
    "scanner_live_capital": {"type": float, "default": 2_000_000.0, "allow_none": False, "min_value": 0.0},  # (AI註: scanner 參考投入本金，預設 2,000,000.0)
}

RUNTIME_PARAM_DEFAULTS = {field_name: spec["default"] for field_name, spec in RUNTIME_PARAM_SPECS.items()}
RUNTIME_PARAM_TYPES = {field_name: spec["type"] for field_name, spec in RUNTIME_PARAM_SPECS.items()}


def build_execution_policy_snapshot():
    return {field_name: spec["default"] for field_name, spec in EXECUTION_POLICY_PARAM_SPECS.items()}


def build_runtime_param_snapshot():
    return dict(RUNTIME_PARAM_DEFAULTS)
