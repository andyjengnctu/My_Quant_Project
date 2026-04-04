"""共用選股政策：投組層與 scanner 的歷史績效門檻。"""

SELECTION_POLICY_PARAM_SPECS = {
    "min_history_trades": {"type": int, "default": 0, "min_value": 0},  # 歷史績效最少交易次數門檻，預設 0
    "min_history_ev": {"type": float, "default": 0.0},  #  歷史績效最小期望值門檻，預設 0.0
    "min_history_win_rate": {"type": float, "default": 0.30, "min_value": 0.0, "max_value": 1.0},  # 歷史績效最小勝率門檻，預設 0.30
}


def build_selection_policy_snapshot():
    return {field_name: spec["default"] for field_name, spec in SELECTION_POLICY_PARAM_SPECS.items()}
