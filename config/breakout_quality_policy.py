"""User-adjustable breakout quality filter policy values.

Editable settings are grouped by operational purpose. Automatically derived values and helper
calculations are centralized at the bottom. Public names remain stable because runtime, research
tools, validators, and historical artifact reconstruction import them directly.
"""

from __future__ import annotations

import math

from config.breakout_policy import BREAKOUT_DEFAULT_HIGH_LEN, build_breakout_optimizer_high_len_values
from config.breakout_quality_experiments import (
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
    get_breakout_quality_pretraining_profile,
)


# =============================================================================
# 1. Active model identity and runtime decision defaults
# =============================================================================

BREAKOUT_QUALITY_MODEL_ARCHITECTURE = "inception_time_v1"  # 9A accepted：目前排序／高品質研究基準；10A Candidate-conditioned Market Set 已由完整 OOS 淘汰。
BREAKOUT_QUALITY_EXPERIMENT_PROFILE = "unique_group_sampling"  # 8F accepted 基準：每個 unique ticker/date group 每個 epoch 只參與一次 optimizer sampling。
BREAKOUT_QUALITY_DEFAULT_FILTER_ID = "breakout_quality_v1"  # 9A正式Dataset與模型工件根路徑；Market Set實驗工件保留於獨立legacy filter id。
BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD = 0.50  # 在查看 OOS 前鎖定的 PASS 分數門檻；不由 OOS 自動調整。


# =============================================================================
# 2. Dataset universe, input window, and label policy
# =============================================================================

BREAKOUT_QUALITY_FEATURE_WINDOW_BARS = 300  # 每個事件輸入模型的歷史特徵交易日數。
BREAKOUT_QUALITY_BENCHMARK_TICKER = "0050"  # 建立相對市場特徵時使用的基準 ETF 代號。

BREAKOUT_QUALITY_LABEL_HORIZON_BARS = 40  # 自突破訊號隔日起，用來判定 PASS 或 REJECT 的未來交易日數；資料不足或無效者不產生有效 Label。
BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS = 120  # 快取每個 ticker/date 的未來 K 線路徑長度；調整門檻或不超過此值的 horizon 時只需快速 relabel。
BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN = 0.05  # PASS 至少要求的最大有利漲幅（MFE）；必須嚴格大於此值。
BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO = 2.0  # PASS 的最低 MFE／MAE；必須嚴格大於 1，且實際判定也採嚴格大於。
BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN = -0.10  # 最大容許不利跌幅；Low 觸及或跌破此值即保守標記為 REJECT。

# Dataset 必須涵蓋 optimizer 搜尋網格，並可額外納入正式預設值或其他指定 high_len。
BREAKOUT_QUALITY_EXTRA_HIGH_LENS = (BREAKOUT_DEFAULT_HIGH_LEN,)


# =============================================================================
# 3. Active architecture settings: InceptionTime
# =============================================================================

# TARGET 是希望模型至少覆蓋的時間範圍；程式會依 depth 自動產生 3 個近似
# 1x／1/2x／1/4x 的正奇數 kernels，並將實際 receptive field 寫入 model manifest。
# 現行 target=228、depth=6 會得到 kernels=(39, 19, 9)，實際 receptive field=229。
# 若要測完整約 600 bars，可同時設定 FEATURE_WINDOW_BARS=600、
# TARGET_RECEPTIVE_FIELD_BARS=600；depth=6 時會得到 kernels=(101, 51, 25)，
# 實際 receptive field=601。
BREAKOUT_QUALITY_INCEPTION_DEPTH = 6
BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS = 228
BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY = 3

# Stage 0／1 learned market-set branch settings。Global-query v1與10A Candidate-conditioned v1
# 均已由完整 OOS 淘汰，只保留legacy checkpoint／manifest嚴格重建與歷史研究重現。
BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS = 300
BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES = (
    "close_return",
    "overnight_return",
    "intraday_return",
    "high_low_range",
    "log_volume_change",
)
BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM = 32
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS = 32
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE = 15
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE = 10
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS = (1, 2, 4)
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION = "group_norm"
BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS = 8
BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT = 4  # Legacy Global Market Set v1 learned queries。
BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT = 1  # 10A每個候選事件產生一個動態query。
BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS = 4
BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM = 128
BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM = 128
BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO = 0.80
BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS = 0  # 0 = 使用當時資料集中全部股票；正整數可作資源受控實驗。
BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH = 4  # 每個 market microbatch 最多物化幾個日期；optimizer logical batch 仍固定使用 DEFAULT_BATCH_SIZE 個事件。


# =============================================================================
# 4. Supervised training and optimization
# =============================================================================

BREAKOUT_QUALITY_DEFAULT_EPOCHS = 200  # 關閉 inner validation 時為固定訓練輪數；開啟時為 epoch 搜尋上限。
BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE = 128  # 8F accepted 基準；unique_group_sampling 時代表 128 個 unique ticker/date groups。8H batch 64 已淘汰。
BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE = 0.0003  # optimizer 的預設 learning rate；9A首輪沿用8F以隔離架構差異。
BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY = 0.0001  # optimizer weight decay；0 表示關閉。Adam 為 coupled L2，AdamW 為 decoupled weight decay。
BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM = 1.0  # 每次更新前的全域 gradient norm 上限；0 表示關閉。
BREAKOUT_QUALITY_CLASS_WEIGHT_MODE = "none"  # Cross-entropy 類別權重；none 不平衡補償，inverse_frequency 依訓練資料加權。PASS／REJECT 接近均衡時建議 none。
BREAKOUT_QUALITY_TIME_WEIGHT_MODE = "none"  # 8F accepted 基準；8K date-balanced training 已由完整 OOS 淘汰。
BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED = 42  # 模型初始化、Dropout 與每個 epoch 資料洗牌的預設亂數種子。
BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES = 20  # 開始訓練前要求的最少有效 train rows。


# =============================================================================
# 5. Inner validation, early stopping, and final refit
# =============================================================================

BREAKOUT_QUALITY_USE_INNER_VALIDATION = True  # 是否以 Selection 尾端資料選 best epoch；正式模型後續採 best checkpoint 或完整 Selection refit，由 FINAL_REFIT_MODE 決定。
BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS = 24  # Inner validation 從 Selection 結尾往前保留的月份數。
BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE = 1  # 8F accepted 基準；Validation loss 連續 1 個 epoch 未改善即停止，避免 unique-group training 迅速重新過擬合。
BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA = 0.0  # Validation loss 至少下降多少才視為新最佳 epoch。
BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES = 20  # 開啟 inner validation 時要求的最少有效 validation rows。
BREAKOUT_QUALITY_FINAL_REFIT_MODE = "selected_epochs"  # 8F accepted 基準：Inner Validation 選出 epoch 後，重新初始化並以完整 eligible Selection 重訓相同 epoch 數。


# =============================================================================
# 6. Device, numerical reproducibility, and data-loading performance
# =============================================================================

BREAKOUT_QUALITY_TORCH_DEVICE = "auto"  # auto 優先使用 CUDA；CUDA 不可用時退回 CPU。
BREAKOUT_QUALITY_USE_MIXED_PRECISION = True  # 僅在 CUDA 啟用；auto dtype 優先 bfloat16，否則 float16。
BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE = "auto"  # auto／bfloat16／float16。
BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS = True  # 固定 PyTorch deterministic algorithms 與 cuDNN deterministic。
BREAKOUT_QUALITY_ALLOW_TF32 = False  # 保持跨裝置數值契約；不使用 TF32。
BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES = 0  # 訓練時預先準備後續 batches 的數量；RAM preload 開啟時預設 0，慢速磁碟可自行調高；不改 batch 順序或 optimizer 更新。
BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK = True  # 訓練與分數匯出前將去重 feature bank 與小型事件陣列載入 RAM；資料值與列順序不變。


# =============================================================================
# 7. Evaluation and score-export performance
# =============================================================================

BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE = 4096  # Train／Validation／Selection 完整評估與分數匯出的分批大小；不抽樣、不改模型更新或輸出列序。
BREAKOUT_QUALITY_EVALUATION_WORKERS = 4  # CPU評估可平行；CUDA固定使用單一GPU serial batches，batch與最終列序不變。
BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION = False  # 同時評估 Inner Train 與 Validation；峰值最多使用 2 × EVALUATION_WORKERS，只做 read-only inference。


# =============================================================================
# 8. Legacy compatibility: 9C Selection-only TS2Vec pretraining
# =============================================================================

# 只供舊工件重建；不是 active architecture 的新實驗入口。
BREAKOUT_QUALITY_PRETRAINING_PROFILE = TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE
BREAKOUT_QUALITY_PRETRAINING_STRIDE = 5  # Dataset sampling設定；每個ticker在Selection endpoint每5個交易日建立一窗。


# =============================================================================
# 9. Derived values and helper calculations (not user-adjustable)
# =============================================================================

# 下列 pretraining 欄位由具名 profile 自動展開，請調整 profile 定義而非直接修改衍生值。
_BREAKOUT_QUALITY_PRETRAINING_SETTINGS = get_breakout_quality_pretraining_profile(
    BREAKOUT_QUALITY_PRETRAINING_PROFILE
)
BREAKOUT_QUALITY_PRETRAINING_FAMILY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.family
BREAKOUT_QUALITY_PRETRAINING_EPOCHS = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.epochs
BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.batch_size
BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.learning_rate
BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.weight_decay
BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM = (
    _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.gradient_clip_norm
)
BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.min_crop_bars
BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.mask_probability
BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.contrastive_alpha
BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT = _BREAKOUT_QUALITY_PRETRAINING_SETTINGS.temporal_unit


def build_breakout_quality_default_high_len_values() -> tuple[int, ...]:
    values = set(build_breakout_optimizer_high_len_values())
    for value in BREAKOUT_QUALITY_EXTRA_HIGH_LENS:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("BREAKOUT_QUALITY_EXTRA_HIGH_LENS 只能包含正整數")
        values.add(normalized)
    return tuple(sorted(values))


def _nearest_positive_odd(value: float, *, minimum: int = 3) -> int:
    lower = max(int(minimum), int(math.floor(float(value))))
    if lower % 2 == 0:
        lower -= 1
    lower = max(int(minimum), lower)
    if lower % 2 == 0:
        lower += 1
    upper = lower + 2
    if abs(float(value) - lower) <= abs(upper - float(value)):
        return int(lower)
    return int(upper)


def build_breakout_quality_inception_kernel_sizes() -> tuple[int, int, int]:
    depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
    target = int(BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)
    residual_every = int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY)
    feature_window = int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS)
    if depth < 1:
        raise ValueError("BREAKOUT_QUALITY_INCEPTION_DEPTH 必須 >= 1")
    if residual_every < 1 or depth % residual_every != 0:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY 必須 >= 1，且必須整除 INCEPTION_DEPTH"
        )
    minimum_target = 1 + depth * 8
    if target < minimum_target:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS 過小；"
            f"depth={depth} 時至少需要 {minimum_target} bars，才能保留三尺度 kernels"
        )
    if target > feature_window:
        raise ValueError(
            "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS 不得大於 "
            "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS；模型不能從不存在的更早歷史學習"
        )

    minimum_max_kernel = math.ceil(1.0 + (target - 1) / depth)
    max_kernel = int(minimum_max_kernel)
    if max_kernel % 2 == 0:
        max_kernel += 1
    middle_kernel = _nearest_positive_odd(max_kernel / 2.0)
    short_kernel = _nearest_positive_odd(max_kernel / 4.0)
    kernels = (int(max_kernel), int(middle_kernel), int(short_kernel))
    if len(set(kernels)) != 3 or not (kernels[0] > kernels[1] > kernels[2] >= 3):
        raise ValueError(f"InceptionTime 自動產生的 kernels 不合法: {kernels}")
    return kernels


def resolve_breakout_quality_inception_receptive_field_bars() -> int:
    kernels = build_breakout_quality_inception_kernel_sizes()
    return 1 + int(BREAKOUT_QUALITY_INCEPTION_DEPTH) * (max(kernels) - 1)


__all__ = [
    "BREAKOUT_QUALITY_BENCHMARK_TICKER",
    "BREAKOUT_QUALITY_TORCH_DEVICE",
    "BREAKOUT_QUALITY_USE_MIXED_PRECISION",
    "BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE",
    "BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS",
    "BREAKOUT_QUALITY_ALLOW_TF32",
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
    "BREAKOUT_QUALITY_PRETRAINING_PROFILE",
    "BREAKOUT_QUALITY_PRETRAINING_FAMILY",
    "BREAKOUT_QUALITY_PRETRAINING_STRIDE",
    "BREAKOUT_QUALITY_PRETRAINING_EPOCHS",
    "BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE",
    "BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE",
    "BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY",
    "BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM",
    "BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS",
    "BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY",
    "BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA",
    "BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT",
    "BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES",
    "BREAKOUT_QUALITY_EXTRA_HIGH_LENS",
    "BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA",
    "BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE",
    "BREAKOUT_QUALITY_FEATURE_WINDOW_BARS",
    "BREAKOUT_QUALITY_INCEPTION_DEPTH",
    "BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS",
    "BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY",
    "BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS",
    "BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES",
    "BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM",
    "BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS",
    "BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE",
    "BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE",
    "BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS",
    "BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION",
    "BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS",
    "BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT",
    "BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT",
    "BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS",
    "BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM",
    "BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM",
    "BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO",
    "BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS",
    "BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH",
    "BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS",
    "BREAKOUT_QUALITY_LABEL_HORIZON_BARS",
    "BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS",
    "BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN",
    "BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN",
    "BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO",
    "BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES",
    "BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES",
    "BREAKOUT_QUALITY_EXPERIMENT_PROFILE",
    "BREAKOUT_QUALITY_MODEL_ARCHITECTURE",
    "BREAKOUT_QUALITY_USE_INNER_VALIDATION",
    "build_breakout_quality_default_high_len_values",
    "build_breakout_quality_inception_kernel_sizes",
    "resolve_breakout_quality_inception_receptive_field_bars",
]
