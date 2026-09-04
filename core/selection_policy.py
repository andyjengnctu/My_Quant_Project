"""Canonical history-selection strategy-parameter schema."""

SELECTION_POLICY_PARAM_SPECS = {
    "use_history_threshold": {"type": bool, "default": True},  # 是否啟用歷史績效門檻
    "min_history_trades": {"type": int, "default": 0, "min_value": 0},  # 歷史績效最少交易次數門檻
    "min_history_ev": {"type": float, "default": -1.0},  # 歷史績效最小期望值門檻
    "min_history_win_rate": {"type": float, "default": 0.30, "min_value": 0.0, "max_value": 1.0},  # 歷史績效最小勝率門檻
}


def build_selection_policy_snapshot() -> dict:
    return {field_name: spec["default"] for field_name, spec in SELECTION_POLICY_PARAM_SPECS.items()}
