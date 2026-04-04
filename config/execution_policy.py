"""共用執行政策：資金、費用與複利設定。"""

EXECUTION_POLICY_PARAM_SPECS = {
    "initial_capital": {"type": float, "default": 1_000_000.0, "min_value": 0.0, "strict_gt": True},  # 回測/投組初始本金，預設 1,000,000.0
    "fixed_risk": {"type": float, "default": 0.01, "min_value": 0.0, "strict_gt": True, "max_value": 1.0},  # 單筆固定風險比例，預設 0.01
    "buy_fee": {"type": float, "default": 0.001425 * 0.28, "min_value": 0.0},  # 買進手續費率，預設 0.001425 * 0.28
    "sell_fee": {"type": float, "default": 0.001425 * 0.28, "min_value": 0.0},  # 賣出手續費率，預設 0.001425 * 0.28
    "tax_rate": {"type": float, "default": 0.003, "min_value": 0.0},  # 賣出交易稅率，預設 0.003
    "min_fee": {"type": float, "default": 20.0, "min_value": 0.0},  # 最低手續費金額，預設 20.0
    "use_compounding": {"type": bool, "default": True},  # 是否使用複利資金口徑，預設 True
}


def build_execution_policy_snapshot():
    return {field_name: spec["default"] for field_name, spec in EXECUTION_POLICY_PARAM_SPECS.items()}
