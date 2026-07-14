"""User-adjustable breakout quality filter policy values."""

from __future__ import annotations

from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN, build_breakout_optimizer_high_len_values

BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "multiscale_cnn_v5"  # 模型架構；可設為 tiny_cnn_v1、multiscale_cnn_v1、multiscale_cnn_v2、multiscale_cnn_v3、multiscale_cnn_v4、multiscale_cnn_v5 或 residual_tcn_v1。v5 與 v1 相同使用 Level 輸入，只將 Long Branch channels 由 16 降為 12。
BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1"  # 未由 CLI 指定時使用的模型、資料集與輸出識別碼。
BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD = 0.50  # 在查看 OOS 前鎖定的 PASS 分數門檻；不由 OOS 自動調整。

BREAKOUT_QUALITY_DEFAULT_EPOCHS = 100  # 關閉 inner validation 時為固定訓練輪數；開啟時為 epoch 搜尋上限。
BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE = 128  # 每次梯度更新使用的訓練 rows 數。
BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE = 0.0003  # Adam optimizer 的預設 learning rate；中型多尺度 CNN 使用較低 learning rate 抑制快速過度擬合。
BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY = 0.0001  # Adam 的 L2 weight decay；0 表示關閉。
BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM = 1.0  # 每次更新前的全域 gradient norm 上限；0 表示關閉。
BREAKOUT_QUALITY_FINAL_REFIT_MODE = "selected_epochs"  # Selection 重訓方式；matched_optimizer_steps 會匹配 best epoch 的 optimizer updates，selected_epochs 為舊式固定相同 epoch 數。
BREAKOUT_QUALITY_CLASS_WEIGHT_MODE = "none"  # Cross-entropy 類別權重；none 不平衡補償，inverse_frequency 依訓練資料加權。PASS／REJECT 接近均衡時建議 none。
BREAKOUT_QUALITY_TIME_WEIGHT_MODE = "none"  # 時間權重；none 僅保留 ticker/date group weighting，year_balanced_sqrt 以年份 group 數平方根反比做溫和平衡。先使用 none 建立乾淨對照。
BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED = 42  # 模型初始化、Dropout 與每個 epoch 資料洗牌的預設亂數種子。
BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE = 4096  # Train／Validation／Selection 完整評估與分數匯出的分批大小；不抽樣、不改模型更新或輸出列序。
BREAKOUT_QUALITY_EVALUATION_WORKERS = 4  # 每個完整資料區段評估／分數匯出的 inference workers；每個 worker 維持單執行緒，batch 與最終列序不變。
BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION = False  # 同時評估 Inner Train 與 Validation；峰值最多使用 2 × EVALUATION_WORKERS，只做 read-only inference。
BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES = 0  # 訓練時預先準備後續 batches 的數量；RAM preload 開啟時預設 0，慢速磁碟可自行調高；不改 batch 順序或 optimizer 更新。
BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK = True  # 訓練與分數匯出前將去重 feature bank 與小型事件陣列載入 RAM；資料值與列順序不變。
BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES = 20  # 開始訓練前要求的最少有效 train rows。

BREAKOUT_QUALITY_USE_INNER_VALIDATION = True  # 是否以 Selection 尾端資料選 best epoch，再用完整 Selection 重訓。
BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS = 24  # Inner validation 從 Selection 結尾往前保留的月份數。
BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE = 1  # Validation loss 連續幾個 epoch 未改善後停止 epoch 搜尋；0 表示跑滿上限。
BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA = 0.0  # Validation loss 至少下降多少才視為新最佳 epoch。
BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES = 20  # 開啟 inner validation 時要求的最少有效 validation rows。

BREAKOUT_QUALITY_FEATURE_WINDOW_BARS = 300  # 每個事件輸入模型的歷史特徵交易日數。
BREAKOUT_QUALITY_LABEL_HORIZON_BARS = 40  # 自突破訊號隔日起，用來判定 PASS 或 REJECT 的未來交易日數；資料不足或無效者不產生有效 Label。
BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS = 120  # 快取每個 ticker/date 的未來 K 線路徑長度；調整門檻或不超過此值的 horizon 時只需快速 relabel。
BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN = 0.05  # PASS 至少要求的最大有利漲幅（MFE）；必須嚴格大於此值。
BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO = 2.0  # PASS 的最低 MFE／MAE；必須嚴格大於 1，且實際判定也採嚴格大於。
BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN = -0.10  # 最大容許不利跌幅；Low 觸及或跌破此值即保守標記為 REJECT。
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
    "BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM",
    "BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED",
    "BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD",
    "BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY",
    "BREAKOUT_QUALITY_FINAL_REFIT_MODE",
    "BREAKOUT_QUALITY_CLASS_WEIGHT_MODE",
    "BREAKOUT_QUALITY_TIME_WEIGHT_MODE",
    "BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE",
    "BREAKOUT_QUALITY_EVALUATION_WORKERS",
    "BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION",
    "BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK",
    "BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES",
    "BREAKOUT_QUALITY_EXTRA_HIGH_LENS",
    "BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA",
    "BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE",
    "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS",
    "BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS",
    "BREAKOUT_QUALITY_LABEL_HORIZON_BARS",
    "BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS",
    "BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN",
    "BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN",
    "BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO",
    "BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES",
    "BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES",
    "BREAKOUT_QUALITY_MODEL_ARCHITECTURE",
    "BREAKOUT_QUALITY_USE_INNER_VALIDATION",
    "build_breakout_quality_default_high_len_values",
]
