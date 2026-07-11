"""User-adjustable breakout quality filter policy values."""

from __future__ import annotations

from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN, build_breakout_optimizer_high_len_values

BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1"  # (AI註: 未由 CLI 指定時使用的模型、資料集與輸出識別碼。)
BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD = 0.50  # (AI註: 在查看 OOS 前鎖定的 PASS 分數門檻；不由 OOS 自動調整。)

BREAKOUT_QUALITY_USE_INNER_VALIDATION = True  # (AI註: 是否以 Selection 尾端資料選 best epoch，再用完整 Selection 重訓。)
BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS = 24  # (AI註: Inner validation 從 Selection 結尾往前保留的月份數。)
BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE = 5  # (AI註: Validation loss 連續幾個 epoch 未改善後停止 epoch 搜尋；0 表示跑滿上限。)
BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA = 0.0  # (AI註: Validation loss 至少下降多少才視為新最佳 epoch。)
BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES = 20  # (AI註: 開啟 inner validation 時要求的最少有效 validation rows。)

BREAKOUT_QUALITY_FEATURE_WINDOW_BARS = 60  # (AI註: 每個事件輸入模型的歷史特徵交易日數。)
BREAKOUT_QUALITY_LABEL_HORIZON_BARS = 40  # (AI註: 進場後用來判定 PASS、REJECT 或 IGNORE 的未來交易日數。)
BREAKOUT_QUALITY_LABEL_ATR_LEN = 14  # (AI註: 建立標籤與 R 倍數基準時使用的 ATR 計算期數。)
BREAKOUT_QUALITY_LABEL_ATR_BUY_TOL = 1.5  # (AI註: 判定突破後是否可成交時允許的買價偏離 ATR 倍數。)
BREAKOUT_QUALITY_LABEL_ATR_TIMES_INIT = 2.0  # (AI註: 標籤模擬的初始風險距離，以 ATR 倍數表示。)
BREAKOUT_QUALITY_POSITIVE_MFE_R = 1.5  # (AI註: 將事件標記為 PASS 所需達到的最大有利變動 R 倍數。)
BREAKOUT_QUALITY_NEGATIVE_MAE_R = -1.0  # (AI註: 將事件視為不利失敗時使用的最大不利變動 R 倍數界線。)
BREAKOUT_QUALITY_REJECT_CONFIRM_MFE_R = 1.0  # (AI註: REJECT 標籤判定中，用來確認後續反彈仍不足的 MFE R 上限。)
BREAKOUT_QUALITY_DEAD_MFE_R = 0.5  # (AI註: 區分無效／低動能事件時使用的最低 MFE R 界線。)
BREAKOUT_QUALITY_EVALUATE_FROM_BARS_AFTER_ENTRY = 1  # (AI註: 自進場後第幾根交易 bar 起開始計算標籤路徑。)
BREAKOUT_QUALITY_BENCHMARK_TICKER = "0050"  # (AI註: 建立相對市場特徵時使用的基準 ETF 代號。)

BREAKOUT_QUALITY_EXTRA_HIGH_LENS = (BREAKOUT_DEFAULT_HIGH_LEN,)  # (AI註: 除 optimizer grid 外，仍須納入 dataset 的額外有效 high_len。)


def build_breakout_quality_default_high_len_values() -> tuple[int, ...]:
    values = set(build_breakout_optimizer_high_len_values())
    for value in BREAKOUT_QUALITY_EXTRA_HIGH_LENS:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("BREAKOUT_QUALITY_EXTRA_HIGH_LENS 只能包含正整數")
        values.add(normalized)
    return tuple(sorted(values))


__all__ = [
    "BREAKOUT_QUALITY_BENCHMARK_TICKER",
    "BREAKOUT_QUALITY_DEAD_MFE_R",
    "BREAKOUT_QUALITY_DEFAULT_FILTER_ID",
    "BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD",
    "BREAKOUT_QUALITY_EVALUATE_FROM_BARS_AFTER_ENTRY",
    "BREAKOUT_QUALITY_EXTRA_HIGH_LENS",
    "BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA",
    "BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE",
    "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS",
    "BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS",
    "BREAKOUT_QUALITY_LABEL_ATR_BUY_TOL",
    "BREAKOUT_QUALITY_LABEL_ATR_LEN",
    "BREAKOUT_QUALITY_LABEL_ATR_TIMES_INIT",
    "BREAKOUT_QUALITY_LABEL_HORIZON_BARS",
    "BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES",
    "BREAKOUT_QUALITY_NEGATIVE_MAE_R",
    "BREAKOUT_QUALITY_POSITIVE_MFE_R",
    "BREAKOUT_QUALITY_REJECT_CONFIRM_MFE_R",
    "BREAKOUT_QUALITY_USE_INNER_VALIDATION",
    "build_breakout_quality_default_high_len_values",
]
