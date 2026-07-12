"""User-adjustable breakout quality filter policy values."""

from __future__ import annotations

from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN, build_breakout_optimizer_high_len_values

BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1"  # 未由 CLI 指定時使用的模型、資料集與輸出識別碼。
BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD = 0.50  # 在查看 OOS 前鎖定的 PASS 分數門檻；不由 OOS 自動調整。

BREAKOUT_QUALITY_DEFAULT_EPOCHS = 20  # 關閉 inner validation 時為固定訓練輪數；開啟時為 epoch 搜尋上限。
BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE = 128  # 每次梯度更新使用的訓練 rows 數。
BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE = 0.001  # Adam optimizer 的預設 learning rate。
BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED = 42  # 模型初始化、Dropout 與每個 epoch 資料洗牌的預設亂數種子。
BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE = 4096  # Train／Validation／Selection 完整評估的分批大小；只降低推論記憶體，不抽樣、不改訓練資料或模型更新。
BREAKOUT_QUALITY_EVALUATION_WORKERS = 4  # 完整評估時並行處理互相獨立的 inference batches；每個 worker 維持單執行緒，輸入 batch 與最終 reduction 順序不變。
BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES = 0  # 訓練時預先準備後續 batches 的數量；RAM preload 開啟時預設 0，慢速磁碟可自行調高；不改 batch 順序或 optimizer 更新。
BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK = True  # 訓練前將去重 feature bank 與小型事件陣列載入 RAM，降低 mmap 隨機讀取；資料值與列順序不變。
BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES = 20  # 開始訓練前要求的最少有效 train rows。

BREAKOUT_QUALITY_USE_INNER_VALIDATION = True  # 是否以 Selection 尾端資料選 best epoch，再用完整 Selection 重訓。
BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS = 24  # Inner validation 從 Selection 結尾往前保留的月份數。
BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE = 3  # Validation loss 連續幾個 epoch 未改善後停止 epoch 搜尋；0 表示跑滿上限。
BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA = 0.0  # Validation loss 至少下降多少才視為新最佳 epoch。
BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES = 20  # 開啟 inner validation 時要求的最少有效 validation rows。

BREAKOUT_QUALITY_FEATURE_WINDOW_BARS = 300  # 每個事件輸入模型的歷史特徵交易日數。
BREAKOUT_QUALITY_LABEL_HORIZON_BARS = 40  # 自突破訊號隔日起，用來判定 PASS 或 REJECT 的未來交易日數；資料不足或無效者不產生有效 Label。
BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS = 120  # 快取每個 ticker/date 的未來 K 線路徑長度；調整門檻或不超過此值的 horizon 時只需快速 relabel。
BREAKOUT_QUALITY_LABEL_PASS_RETURN = 0.10  # 相對突破訊號日收盤價，先上漲至此報酬率即標記為 PASS。
BREAKOUT_QUALITY_LABEL_REJECT_RETURN = -0.05  # 相對突破訊號日收盤價；若先碰此跌幅，或完整觀察期內未先達 PASS 門檻，均標記為 REJECT。
BREAKOUT_QUALITY_BENCHMARK_TICKER = "0050"  # 建立相對市場特徵時使用的基準 ETF 代號。

BREAKOUT_QUALITY_EXTRA_HIGH_LENS = (BREAKOUT_DEFAULT_HIGH_LEN,)  # 除 optimizer grid 外，仍須納入 dataset 的額外有效 high_len。


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
    "BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE",
    "BREAKOUT_QUALITY_DEFAULT_EPOCHS",
    "BREAKOUT_QUALITY_DEFAULT_FILTER_ID",
    "BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE",
    "BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED",
    "BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD",
    "BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE",
    "BREAKOUT_QUALITY_EVALUATION_WORKERS",
    "BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK",
    "BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES",
    "BREAKOUT_QUALITY_EXTRA_HIGH_LENS",
    "BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA",
    "BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE",
    "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS",
    "BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS",
    "BREAKOUT_QUALITY_LABEL_HORIZON_BARS",
    "BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS",
    "BREAKOUT_QUALITY_LABEL_PASS_RETURN",
    "BREAKOUT_QUALITY_LABEL_REJECT_RETURN",
    "BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES",
    "BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES",
    "BREAKOUT_QUALITY_USE_INNER_VALIDATION",
    "build_breakout_quality_default_high_len_values",
]
