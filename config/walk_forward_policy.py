"""Walk-forward / optimizer 主設定。"""

# 一般模式建議只調整 train_start_year、min_train_years、test_window_months。
# search_train_end_year 保留為進階覆寫；若設為 None，會自動等於
# train_start_year + min_train_years - 1，避免日常使用同時維護兩個年限旋鈕。

WALK_FORWARD_POLICY = {
    "train_start_year": 2012,  # 主搜尋訓練資料的起始年；早於此年的資料不納入 wf 主訓練區間。
    "min_train_years": 8,  # walk-forward 每個 OOS 視窗開始前，至少要先累積的訓練年數。
    "test_window_months": 6,  # 每個 walk-forward OOS / test 視窗的月數；目前以半年度為一窗。
    "search_train_end_year": None,  # 主搜尋訓練資料的結束年；None 代表自動用 train_start_year + min_train_years - 1。
    "objective_mode": "wf_gate_median",  # optimizer 評分模式；legacy_base_score=舊版單次分數，wf_gate_median=walk-forward gate 中位數。
    "regime_up_threshold_pct": 8.0,  # 市場 regime 分類中，判定上漲窗的報酬門檻（%）。
    "regime_down_threshold_pct": -8.0,  # 市場 regime 分類中，判定下跌窗的報酬門檻（%）；介於上下門檻間視為盤整。
    "min_window_bars": 20,  # 單一 walk-forward 視窗至少要有的 K 棒數；不足時該窗不納入正式比較。
    "gate_min_median_score": 0.0,  # upgrade gate：walk-forward 視窗分數中位數下限。
    "gate_min_worst_ret_pct": -8.0,  # upgrade gate：所有視窗中最差報酬率下限（%）。
    "gate_min_flat_median_score": 0.0,  # upgrade gate：盤整 regime 視窗分數中位數下限。
    "compare_worst_ret_tolerance_pct": 1.0,  # compare gate：候選版相對 champion 可容忍的最差視窗報酬劣化幅度（%）。
    "compare_max_mdd_tolerance_pct": 2.0,  # compare gate：候選版相對 champion 可容忍的最大回撤劣化幅度（%）。
}
