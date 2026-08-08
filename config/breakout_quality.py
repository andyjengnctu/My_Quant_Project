"""Single source of truth for breakout-quality configuration.

Edit only the user-settings section at the top of this file. Named profile definitions,
validation, derived values, and helper functions are centralized below. This is the only
breakout-quality configuration module; the former policy, experiments, and workflow modules
were removed to keep one user-facing source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from config.breakout_policy import (
    BREAKOUT_DEFAULT_HIGH_LEN,
    build_breakout_optimizer_high_len_values,
)
from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT


# =============================================================================
# USER SETTINGS — edit this section only
# =============================================================================

# =============================================================================
# 0. Main menu workflow selector
# =============================================================================

# This is the only setting normally changed to switch the main menu model workflow.
# - 9A binary filter: "unique_group_sampling"
# - continuous PIT ranker (PASS-only): "strategy_aligned_no_time_pass_magnitude_mse"
# - MR-12A all-event continuous MSE: "strategy_aligned_no_time_all_event_mse"
# - MR-12B all-event pairwise ranker: "strategy_aligned_no_time_all_event_pairwise"
BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE = "strategy_aligned_no_time_all_event_pairwise"

# (AI註: Breakout-quality全部正式模型流程共用此Seed；CLI --seed只作單次覆寫。)
BREAKOUT_QUALITY_RANDOM_SEED = 42


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
BREAKOUT_QUALITY_PRETRAINING_PROFILE = "ts2vec_selection_only"
BREAKOUT_QUALITY_PRETRAINING_STRIDE = 5  # Dataset sampling設定；每個ticker在Selection endpoint每5個交易日建立一窗。

# =============================================================================
# 9. Selection point-in-time score contract
# =============================================================================

# These settings are used only when the selected experiment profile is a continuous ranker.
# Use "auto" to resolve the earliest legal monthly score start from the current Dataset,
# Continuous Target, inner-validation window, and minimum group-count contract.
# None for the end date means use the canonical Selection end from the outer split policy.
BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE = "auto"
# Strategy adaptation compares the extended PIT history against this prior official start.
# It is only a coverage reference, not a lower bound for model training or scoring.
BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE = "2014-01-01"
BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE: str | None = None
BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS = 12
BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS = 24
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS = 20
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS = 20
BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS = 1
BREAKOUT_QUALITY_POINT_IN_TIME_RESUME = True


# =============================================================================
# 10. Strategy workflow defaults
# =============================================================================

BREAKOUT_QUALITY_STRATEGY_DATASET = "full"
BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY = "base-finalist-best"
BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS = 10
BREAKOUT_QUALITY_STRATEGY_ROTATION = "off"
BREAKOUT_QUALITY_STRATEGY_ADAPT_FIXED_RISK = 0.01
BREAKOUT_QUALITY_STRATEGY_ADAPT_MAX_POSITION_CAP_PCT = 0.30

# "auto" resolves from the selected experiment profile:
# - binary classification -> hard-filter / canonical_runtime / original
# - continuous ranker     -> score-ranking / selection_point_in_time /
#                            breakout_quality_score_desc
BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE = "auto"
BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE = "auto"
BREAKOUT_QUALITY_STRATEGY_BUY_SORT = "auto"

# =============================================================================
# INTERNAL PROFILE DEFINITIONS AND SUPPORTED VALUES — normally do not edit
# =============================================================================

BASELINE_EXPERIMENT_PROFILE = "baseline"
ADAMW_ONLY_EXPERIMENT_PROFILE = "adamw_only"
ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE = "adam_warmup_cosine"
HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE = "history_masking_only"
UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE = "unique_group_sampling"
UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE = "unique_group_date_balanced"
STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE = "strategy_aligned_daily_percentile_mse"
STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE = "strategy_aligned_no_time_pass_magnitude_mse"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE = "strategy_aligned_no_time_all_event_mse"
STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE = "strategy_aligned_no_time_all_event_pairwise"
TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE = "ts2vec_selection_only"

TRAINING_SAMPLING_ALL_EVENT_ROWS = "all_event_rows_group_weighted"
TRAINING_SAMPLING_UNIQUE_TICKER_DATE = "unique_ticker_date"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES = (
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
)

TIME_WEIGHT_MODE_NONE = "none"
TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT = "year_balanced_sqrt"
TIME_WEIGHT_MODE_DATE_BALANCED = "date_balanced"
SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES = (
    TIME_WEIGHT_MODE_NONE,
    TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT,
    TIME_WEIGHT_MODE_DATE_BALANCED,
)

TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM = "batch_weight_sum"
TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE = "fixed_batch_size"

TRAINING_LABEL_SCOPE_ALL = "all_labels"
TRAINING_LABEL_SCOPE_PASS_ONLY = "pass_only"
SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES = (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
)

TRAINING_OBJECTIVE_BINARY_CLASSIFICATION = "binary_classification"
TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION = "daily_percentile_regression"
TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING = "daily_pairwise_ranking"
CONTINUOUS_RANKER_TRAINING_OBJECTIVES = (
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES = (
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    *CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
)
SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS = (
    TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
)

LR_SCHEDULE_NONE = "none"
LR_SCHEDULE_LINEAR_WARMUP_COSINE = "linear_warmup_cosine"
AUGMENTATION_NONE = "none"
AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK = "old_history_contiguous_mask"

SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS = ("adam", "adamw")
SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES = (
    LR_SCHEDULE_NONE,
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
)
SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS = (
    AUGMENTATION_NONE,
    AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
)


@dataclass(frozen=True)
class BreakoutQualityExperimentProfile:
    name: str
    optimizer_name: str
    lr_schedule_name: str = LR_SCHEDULE_NONE
    augmentation_name: str = "none"
    augmentation_probability: float = 0.0
    augmentation_protected_recent_bars: int = 0
    augmentation_min_mask_bars: int = 0
    augmentation_max_mask_bars: int = 0
    lr_warmup_fraction: float = 0.0
    lr_minimum_ratio: float = 1.0
    training_sampling_mode: str = TRAINING_SAMPLING_ALL_EVENT_ROWS
    time_weight_mode: str | None = None
    training_weight_reduction: str = TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM
    training_objective: str = TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
    continuous_target_id: str | None = None
    loss_name: str = "cross_entropy"
    epoch_selection_metric: str = "validation_loss"
    training_label_scope: str = TRAINING_LABEL_SCOPE_ALL

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("experiment profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("experiment profile name 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 optimizer: {self.optimizer_name!r}")
        if self.lr_schedule_name not in SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES:
            raise ValueError(f"不支援的 LR schedule: {self.lr_schedule_name!r}")
        if self.augmentation_name not in SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS:
            raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")
        if self.training_sampling_mode not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES:
            raise ValueError(
                f"不支援的 training sampling mode: {self.training_sampling_mode!r}"
            )
        if (
            self.time_weight_mode is not None
            and self.time_weight_mode not in SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES
        ):
            raise ValueError(f"不支援的 time weight mode: {self.time_weight_mode!r}")
        if self.training_weight_reduction not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS:
            raise ValueError(
                f"不支援的 training weight reduction: {self.training_weight_reduction!r}"
            )
        if self.training_objective not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES:
            raise ValueError(f"不支援的 training objective: {self.training_objective!r}")
        if self.training_label_scope not in SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES:
            raise ValueError(f"不支援的 training label scope: {self.training_label_scope!r}")
        if self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            if self.continuous_target_id is not None:
                raise ValueError("binary classification profile 不得指定 continuous_target_id")
            if self.loss_name != "cross_entropy" or self.epoch_selection_metric != "validation_loss":
                raise ValueError("binary classification profile 必須使用 cross_entropy / validation_loss")
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                raise ValueError("binary classification profile 必須使用all_labels scope")
        elif self.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
            if not str(self.continuous_target_id or "").strip():
                raise ValueError("continuous ranker profile 必須指定 continuous_target_id")
            expected_loss = (
                "mse"
                if self.training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION
                else "pairwise_logistic"
            )
            if self.loss_name != expected_loss:
                raise ValueError(
                    "continuous ranker loss與training objective不一致: "
                    f"objective={self.training_objective}, expected={expected_loss}, actual={self.loss_name}"
                )
            if self.epoch_selection_metric != "mean_daily_spearman":
                raise ValueError("continuous ranker profile 必須以 mean_daily_spearman 選 epoch")
            if self.training_sampling_mode != TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                raise ValueError("continuous ranker只允許 unique ticker/date sampling")
            if self.time_weight_mode not in {None, TIME_WEIGHT_MODE_NONE}:
                raise ValueError("continuous ranker不允許 time weighting")
            if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
                raise ValueError("continuous ranker只允許 batch_weight_sum")
        if (
            self.training_weight_reduction == TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE
            and self.time_weight_mode != TIME_WEIGHT_MODE_DATE_BALANCED
        ):
            raise ValueError(
                "fixed_batch_size training weight reduction 目前只允許 date_balanced profile"
            )
        warmup_fraction = float(self.lr_warmup_fraction)
        minimum_ratio = float(self.lr_minimum_ratio)
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            if warmup_fraction != 0.0 or minimum_ratio != 1.0:
                raise ValueError("無 LR schedule 時 warmup 必須為 0、minimum ratio 必須為 1")
        elif self.lr_schedule_name == LR_SCHEDULE_LINEAR_WARMUP_COSINE:
            if not 0.0 < warmup_fraction < 1.0:
                raise ValueError("linear warmup fraction 必須介於 0 與 1 之間")
            if not 0.0 < minimum_ratio <= 1.0:
                raise ValueError("minimum LR ratio 必須介於 0 與 1 之間")
        augmentation_parameters = self.augmentation_parameters()
        if self.augmentation_name == AUGMENTATION_NONE:
            if (
                float(self.augmentation_probability) != 0.0
                or int(self.augmentation_protected_recent_bars) != 0
                or int(self.augmentation_min_mask_bars) != 0
                or int(self.augmentation_max_mask_bars) != 0
                or augmentation_parameters
            ):
                raise ValueError("augmentation=none 時不可帶 augmentation 參數")
        elif self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            probability = float(self.augmentation_probability)
            protected_recent_bars = int(self.augmentation_protected_recent_bars)
            min_mask_bars = int(self.augmentation_min_mask_bars)
            max_mask_bars = int(self.augmentation_max_mask_bars)
            if not 0.0 < probability <= 1.0:
                raise ValueError("masking augmentation probability 必須介於 0 與 1 之間")
            if protected_recent_bars < 1:
                raise ValueError("masking protected_recent_bars 必須 >=1")
            if min_mask_bars < 1 or max_mask_bars < min_mask_bars:
                raise ValueError("masking bars 必須滿足 1 <= min <= max")

    def lr_schedule_parameters(self) -> dict[str, float]:
        if self.lr_schedule_name == LR_SCHEDULE_NONE:
            return {}
        return {
            "warmup_fraction": float(self.lr_warmup_fraction),
            "minimum_lr_ratio": float(self.lr_minimum_ratio),
        }

    def augmentation_parameters(self) -> dict[str, int | float]:
        if self.augmentation_name == AUGMENTATION_NONE:
            return {}
        if self.augmentation_name == AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK:
            return {
                "probability": float(self.augmentation_probability),
                "protected_recent_bars": int(self.augmentation_protected_recent_bars),
                "min_mask_bars": int(self.augmentation_min_mask_bars),
                "max_mask_bars": int(self.augmentation_max_mask_bars),
            }
        raise ValueError(f"不支援的 augmentation: {self.augmentation_name!r}")

    def as_manifest_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "optimizer_name": self.optimizer_name,
            "lr_schedule_name": self.lr_schedule_name,
            "augmentation_name": self.augmentation_name,
        }
        schedule_parameters = self.lr_schedule_parameters()
        if schedule_parameters:
            payload["lr_schedule_parameters"] = schedule_parameters
        augmentation_parameters = self.augmentation_parameters()
        if augmentation_parameters:
            payload["augmentation_parameters"] = augmentation_parameters
        if self.training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS:
            payload["training_sampling_mode"] = self.training_sampling_mode
        if self.time_weight_mode is not None:
            payload["time_weight_mode"] = self.time_weight_mode
        if self.training_weight_reduction != TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
            payload["training_weight_reduction"] = self.training_weight_reduction
        if self.training_objective != TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
            payload.update({
                "training_objective": self.training_objective,
                "continuous_target_id": self.continuous_target_id,
                "loss_name": self.loss_name,
                "epoch_selection_metric": self.epoch_selection_metric,
            })
            if self.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
                payload["training_label_scope"] = self.training_label_scope
        return payload



@dataclass(frozen=True)
class BreakoutQualityPretrainingProfile:
    name: str
    family: str
    optimizer_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    min_crop_bars: int
    mask_probability: float
    contrastive_alpha: float
    temporal_unit: int

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if not normalized_name or normalized_name != self.name:
            raise ValueError("pretraining profile name 必須是非空白小寫名稱")
        if any(token in normalized_name for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining profile name 必須是安全的單一資料夾名稱")
        if str(self.family).strip().lower() != self.family or not self.family:
            raise ValueError("pretraining family 必須是非空白小寫名稱")
        if any(token in self.family for token in ("/", "\\", "\x00")):
            raise ValueError("pretraining family 必須是安全的單一資料夾名稱")
        if self.optimizer_name not in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS:
            raise ValueError(f"不支援的 pretraining optimizer: {self.optimizer_name!r}")
        if int(self.epochs) < 1 or int(self.batch_size) < 2:
            raise ValueError("pretraining epochs 必須 >=1 且 batch_size 必須 >=2")
        if (
            float(self.learning_rate) <= 0.0
            or float(self.weight_decay) < 0.0
            or float(self.gradient_clip_norm) < 0.0
        ):
            raise ValueError(
                "pretraining learning_rate 必須 >0，weight_decay與gradient_clip_norm必須 >=0"
            )
        if int(self.min_crop_bars) < 2 or int(self.temporal_unit) < 0:
            raise ValueError("pretraining min_crop_bars 必須 >=2 且 temporal_unit 必須 >=0")
        if not 0.0 <= float(self.mask_probability) < 1.0:
            raise ValueError("pretraining mask_probability 必須介於0（含）與1（不含）")
        if not 0.0 <= float(self.contrastive_alpha) <= 1.0:
            raise ValueError("pretraining contrastive_alpha 必須介於0與1")

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "optimizer_name": self.optimizer_name,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "min_crop_bars": int(self.min_crop_bars),
            "mask_probability": float(self.mask_probability),
            "contrastive_alpha": float(self.contrastive_alpha),
            "temporal_unit": int(self.temporal_unit),
        }


_PRETRAINING_PROFILES = {
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE: BreakoutQualityPretrainingProfile(
        name=TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
        family="ts2vec_v1",
        optimizer_name="adamw",
        epochs=10,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        min_crop_bars=60,
        mask_probability=0.5,
        contrastive_alpha=0.5,
        temporal_unit=0,
    ),
}
SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES = tuple(_PRETRAINING_PROFILES)


def normalize_breakout_quality_pretraining_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _PRETRAINING_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality pretraining profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_pretraining_profile(
    value: str,
) -> BreakoutQualityPretrainingProfile:
    return _PRETRAINING_PROFILES[normalize_breakout_quality_pretraining_profile(value)]


def build_breakout_quality_pretraining_profile_payload(
    value: str,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    gradient_clip_norm: float | None = None,
    min_crop_bars: int | None = None,
    mask_probability: float | None = None,
    contrastive_alpha: float | None = None,
    temporal_unit: int | None = None,
) -> dict[str, Any]:
    """Return the named profile payload with explicit CLI overrides applied.

    Formal workflow runs use the profile defaults. The override path remains available for
    isolated development experiments, while downstream canonical training can reject an
    encoder whose stored payload differs from the active named profile.
    """

    profile = get_breakout_quality_pretraining_profile(value)
    resolved = BreakoutQualityPretrainingProfile(
        name=profile.name,
        family=profile.family,
        optimizer_name=profile.optimizer_name,
        epochs=profile.epochs if epochs is None else int(epochs),
        batch_size=profile.batch_size if batch_size is None else int(batch_size),
        learning_rate=(
            profile.learning_rate if learning_rate is None else float(learning_rate)
        ),
        weight_decay=profile.weight_decay if weight_decay is None else float(weight_decay),
        gradient_clip_norm=(
            profile.gradient_clip_norm
            if gradient_clip_norm is None
            else float(gradient_clip_norm)
        ),
        min_crop_bars=(
            profile.min_crop_bars if min_crop_bars is None else int(min_crop_bars)
        ),
        mask_probability=(
            profile.mask_probability
            if mask_probability is None
            else float(mask_probability)
        ),
        contrastive_alpha=(
            profile.contrastive_alpha
            if contrastive_alpha is None
            else float(contrastive_alpha)
        ),
        temporal_unit=profile.temporal_unit if temporal_unit is None else int(temporal_unit),
    )
    return resolved.as_manifest_payload()

_EXPERIMENT_PROFILES = {
    BASELINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=BASELINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
    ),
    ADAMW_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAMW_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adamw",
    ),
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        lr_schedule_name=LR_SCHEDULE_LINEAR_WARMUP_COSINE,
        lr_warmup_fraction=0.05,
        lr_minimum_ratio=0.10,
    ),
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        augmentation_name=AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK,
        augmentation_probability=0.50,
        augmentation_protected_recent_bars=60,
        augmentation_min_mask_bars=10,
        augmentation_max_mask_bars=30,
    ),
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    ),
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE: BreakoutQualityExperimentProfile(
        name=UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        time_weight_mode=TIME_WEIGHT_MODE_DATE_BALANCED,
        training_weight_reduction=TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    ),
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
    ),
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="mse",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE: BreakoutQualityExperimentProfile(
        name=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        optimizer_name="adam",
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        continuous_target_id="strategy_aligned_opportunity_no_time_r_v1",
        loss_name="pairwise_logistic",
        epoch_selection_metric="mean_daily_spearman",
        training_label_scope=TRAINING_LABEL_SCOPE_ALL,
    ),
}

SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES = tuple(_EXPERIMENT_PROFILES)
SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES = tuple(
    name
    for name, profile in _EXPERIMENT_PROFILES.items()
    if profile.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION
)


def normalize_breakout_quality_experiment_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _EXPERIMENT_PROFILES:
        allowed = ", ".join(SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES)
        raise ValueError(
            f"不支援的 breakout quality experiment profile: {value!r}；可用值: {allowed}"
        )
    return normalized


def get_breakout_quality_experiment_profile(
    value: str,
) -> BreakoutQualityExperimentProfile:
    return _EXPERIMENT_PROFILES[
        normalize_breakout_quality_experiment_profile(value)
    ]


def resolve_breakout_quality_random_seed() -> int:
    """Return the single configured breakout-quality random seed."""

    resolved = int(BREAKOUT_QUALITY_RANDOM_SEED)
    if resolved < 0:
        raise ValueError("breakout quality random seed 必須是>=0的整數")
    return resolved

# =============================================================================
# DERIVED VALUES AND HELPER FUNCTIONS — do not edit unless changing implementation
# =============================================================================

# Workflow filter and architecture intentionally follow the active canonical identity.
# Users switch the main-menu model type with BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE.
BREAKOUT_QUALITY_WORKFLOW_FILTER_ID = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE = BREAKOUT_QUALITY_MODEL_ARCHITECTURE


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

# =============================================================================
# WORKFLOW RESOLUTION AND VALIDATION — do not edit unless changing implementation
# =============================================================================

WORKFLOW_STRATEGY_MODE_AUTO = "auto"
WORKFLOW_STRATEGY_MODE_HARD_FILTER = "hard-filter"
WORKFLOW_STRATEGY_MODE_SCORE_RANKING = "score-ranking"
SUPPORTED_WORKFLOW_STRATEGY_MODES = (
    WORKFLOW_STRATEGY_MODE_AUTO,
    WORKFLOW_STRATEGY_MODE_HARD_FILTER,
    WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
)

WORKFLOW_SCORE_SOURCE_AUTO = "auto"
WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME = "selection_point_in_time"
WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS = "final_selection_model_oos"
WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME = "canonical_runtime"
SUPPORTED_WORKFLOW_SCORE_SOURCES = (
    WORKFLOW_SCORE_SOURCE_AUTO,
    WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS,
    WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
)

WORKFLOW_BUY_SORT_AUTO = "auto"
WORKFLOW_BUY_SORT_ORIGINAL = "original"
WORKFLOW_BUY_SORT_SCORE_DESC = "breakout_quality_score_desc"

@dataclass(frozen=True)
class BreakoutQualityWorkflowSettings:
    filter_id: str
    model_architecture: str
    experiment_profile: str
    training_objective: str
    continuous_target_id: str | None
    training_label_scope: str
    seed: int
    point_in_time_score_start_date: str
    point_in_time_coverage_reference_start_date: str
    point_in_time_score_end_date: str | None
    point_in_time_fold_months: int
    point_in_time_inner_validation_months: int
    point_in_time_min_train_groups: int
    point_in_time_min_validation_groups: int
    point_in_time_min_score_groups: int
    point_in_time_resume: bool
    strategy_dataset: str
    strategy_param_policy: str
    strategy_max_positions: int
    strategy_rotation: str
    strategy_adapt_trials_per_fold: int
    strategy_adapt_fixed_risk: float
    strategy_adapt_max_position_cap_pct: float
    strategy_comparison_mode: str
    strategy_score_source: str
    strategy_buy_sort: str

    @property
    def is_binary_classification(self) -> bool:
        return self.training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION

    @property
    def is_continuous_ranker(self) -> bool:
        return self.training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "filter_id": self.filter_id,
            "model_architecture": self.model_architecture,
            "experiment_profile": self.experiment_profile,
            "training_objective": self.training_objective,
            "continuous_target_id": self.continuous_target_id,
            "training_label_scope": self.training_label_scope,
            "seed": int(self.seed),
            "point_in_time": {
                "enabled": bool(self.is_continuous_ranker),
                "score_start_date": self.point_in_time_score_start_date,
                "coverage_reference_start_date": (
                    self.point_in_time_coverage_reference_start_date
                ),
                "score_end_date": self.point_in_time_score_end_date,
                "fold_months": int(self.point_in_time_fold_months),
                "inner_validation_months": int(
                    self.point_in_time_inner_validation_months
                ),
                "min_train_groups": int(self.point_in_time_min_train_groups),
                "min_validation_groups": int(
                    self.point_in_time_min_validation_groups
                ),
                "min_score_groups": int(self.point_in_time_min_score_groups),
                "resume": bool(self.point_in_time_resume),
            },
            "strategy": {
                "dataset": self.strategy_dataset,
                "param_policy": self.strategy_param_policy,
                "max_positions": int(self.strategy_max_positions),
                "rotation": self.strategy_rotation,
                "adaptation": {
                    "trials_per_fold": int(self.strategy_adapt_trials_per_fold),
                    "trial_source": "OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT",
                    "fixed_risk": float(self.strategy_adapt_fixed_risk),
                    "max_position_cap_pct": float(
                        self.strategy_adapt_max_position_cap_pct
                    ),
                },
                "comparison_mode": self.strategy_comparison_mode,
                "score_source": self.strategy_score_source,
                "buy_sort": self.strategy_buy_sort,
            },
        }


def _resolve_strategy_defaults(training_objective: str) -> tuple[str, str, str]:
    if training_objective == TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
        return (
            WORKFLOW_STRATEGY_MODE_HARD_FILTER,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
            WORKFLOW_BUY_SORT_ORIGINAL,
        )
    if training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        return (
            WORKFLOW_STRATEGY_MODE_SCORE_RANKING,
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_BUY_SORT_SCORE_DESC,
        )
    raise ValueError(f"不支援的 workflow training objective: {training_objective!r}")


def _resolve_auto(value: str, *, auto_value: str, resolved_default: str) -> str:
    normalized = str(value).strip()
    return resolved_default if normalized == auto_value else normalized


def get_breakout_quality_workflow_settings() -> BreakoutQualityWorkflowSettings:
    random_seed = resolve_breakout_quality_random_seed()
    profile = get_breakout_quality_experiment_profile(
        BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
    )
    if profile.training_objective not in {
        TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
        *CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    }:
        raise ValueError(
            "workflow experiment profile必須是binary classification或continuous ranker"
        )
    try:
        coverage_reference_start = date.fromisoformat(
            str(BREAKOUT_QUALITY_POINT_IN_TIME_COVERAGE_REFERENCE_START_DATE).strip()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "point-in-time coverage reference start date必須是YYYY-MM-DD合法日期"
        ) from exc
    if int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS) < 1:
        raise ValueError("point-in-time fold months 必須 >= 1")
    if int(BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS) < 1:
        raise ValueError("point-in-time inner validation months 必須 >= 1")
    if min(
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS),
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS),
        int(BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS),
    ) < 1:
        raise ValueError("point-in-time minimum group counts 必須 >= 1")
    if BREAKOUT_QUALITY_STRATEGY_DATASET not in {"reduced", "full"}:
        raise ValueError("strategy dataset 必須是 reduced 或 full")
    if BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY not in {
        "auto",
        "base-finalist-best",
        "base-finalists-agree",
    }:
        raise ValueError(
            "strategy param policy 必須是auto、base-finalist-best或base-finalists-agree"
        )
    if int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS) < 1:
        raise ValueError("strategy max positions 必須 >= 1")
    if BREAKOUT_QUALITY_STRATEGY_ROTATION not in {"off", "on"}:
        raise ValueError("strategy rotation 必須是 off 或 on")
    if int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT) < 1:
        raise ValueError("outer rolling optimizer trials per fold 必須 >= 1")
    if not 0.0 < float(BREAKOUT_QUALITY_STRATEGY_ADAPT_FIXED_RISK) <= 1.0:
        raise ValueError("strategy adaptation fixed risk 必須介於0與1")
    if not 0.0 < float(BREAKOUT_QUALITY_STRATEGY_ADAPT_MAX_POSITION_CAP_PCT) <= 1.0:
        raise ValueError("strategy adaptation max position cap pct 必須介於0與1")

    raw_comparison_mode = str(BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE).strip()
    if raw_comparison_mode not in SUPPORTED_WORKFLOW_STRATEGY_MODES:
        raise ValueError(
            "strategy comparison mode 必須是auto、hard-filter或score-ranking"
        )
    raw_score_source = str(BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE).strip()
    if raw_score_source not in SUPPORTED_WORKFLOW_SCORE_SOURCES:
        raise ValueError(
            "strategy score source 必須是auto、selection_point_in_time、"
            "final_selection_model_oos或canonical_runtime"
        )
    raw_buy_sort = str(BREAKOUT_QUALITY_STRATEGY_BUY_SORT).strip()
    if not raw_buy_sort:
        raise ValueError("strategy buy sort不可為空白")

    default_mode, default_score_source, default_buy_sort = _resolve_strategy_defaults(
        profile.training_objective
    )
    strategy_comparison_mode = _resolve_auto(
        raw_comparison_mode,
        auto_value=WORKFLOW_STRATEGY_MODE_AUTO,
        resolved_default=default_mode,
    )
    strategy_score_source = _resolve_auto(
        raw_score_source,
        auto_value=WORKFLOW_SCORE_SOURCE_AUTO,
        resolved_default=default_score_source,
    )
    strategy_buy_sort = _resolve_auto(
        raw_buy_sort,
        auto_value=WORKFLOW_BUY_SORT_AUTO,
        resolved_default=default_buy_sort,
    )

    if strategy_comparison_mode == WORKFLOW_STRATEGY_MODE_HARD_FILTER:
        if strategy_score_source != WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME:
            raise ValueError("hard-filter策略比較只接受canonical_runtime score source")
        if strategy_buy_sort != WORKFLOW_BUY_SORT_ORIGINAL:
            raise ValueError("hard-filter策略比較必須沿用original buy-sort")
    elif strategy_comparison_mode == WORKFLOW_STRATEGY_MODE_SCORE_RANKING:
        if strategy_score_source not in {
            WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS,
            WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
        }:
            raise ValueError("score-ranking策略比較缺少合法score source")
        if strategy_buy_sort != WORKFLOW_BUY_SORT_SCORE_DESC:
            raise ValueError(
                "score-ranking策略比較目前只支援breakout_quality_score_desc"
            )
    else:
        raise ValueError(
            f"不支援的 resolved strategy comparison mode: {strategy_comparison_mode!r}"
        )

    return BreakoutQualityWorkflowSettings(
        filter_id=str(BREAKOUT_QUALITY_WORKFLOW_FILTER_ID),
        model_architecture=str(BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE),
        experiment_profile=str(BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE),
        training_objective=str(profile.training_objective),
        continuous_target_id=(
            None
            if profile.continuous_target_id is None
            else str(profile.continuous_target_id)
        ),
        training_label_scope=str(profile.training_label_scope),
        seed=random_seed,
        point_in_time_score_start_date=str(
            BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_START_DATE
        ),
        point_in_time_coverage_reference_start_date=(
            coverage_reference_start.isoformat()
        ),
        point_in_time_score_end_date=(
            None
            if BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE is None
            else str(BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_END_DATE)
        ),
        point_in_time_fold_months=int(BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_MONTHS),
        point_in_time_inner_validation_months=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_INNER_VALIDATION_MONTHS
        ),
        point_in_time_min_train_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_TRAIN_GROUPS
        ),
        point_in_time_min_validation_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_VALIDATION_GROUPS
        ),
        point_in_time_min_score_groups=int(
            BREAKOUT_QUALITY_POINT_IN_TIME_MIN_SCORE_GROUPS
        ),
        point_in_time_resume=bool(BREAKOUT_QUALITY_POINT_IN_TIME_RESUME),
        strategy_dataset=str(BREAKOUT_QUALITY_STRATEGY_DATASET),
        strategy_param_policy=str(BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY),
        strategy_max_positions=int(BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS),
        strategy_rotation=str(BREAKOUT_QUALITY_STRATEGY_ROTATION),
        strategy_adapt_trials_per_fold=int(
            OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
        ),
        strategy_adapt_fixed_risk=float(BREAKOUT_QUALITY_STRATEGY_ADAPT_FIXED_RISK),
        strategy_adapt_max_position_cap_pct=float(
            BREAKOUT_QUALITY_STRATEGY_ADAPT_MAX_POSITION_CAP_PCT
        ),
        strategy_comparison_mode=strategy_comparison_mode,
        strategy_score_source=strategy_score_source,
        strategy_buy_sort=strategy_buy_sort,
    )

__all__ = [
    'BREAKOUT_QUALITY_BENCHMARK_TICKER',
    'BREAKOUT_QUALITY_TORCH_DEVICE',
    'BREAKOUT_QUALITY_USE_MIXED_PRECISION',
    'BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE',
    'BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS',
    'BREAKOUT_QUALITY_ALLOW_TF32',
    'BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE',
    'BREAKOUT_QUALITY_DEFAULT_EPOCHS',
    'BREAKOUT_QUALITY_DEFAULT_FILTER_ID',
    'BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE',
    'BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM',
    'BREAKOUT_QUALITY_RANDOM_SEED',
    'BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD',
    'BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY',
    'BREAKOUT_QUALITY_FINAL_REFIT_MODE',
    'BREAKOUT_QUALITY_CLASS_WEIGHT_MODE',
    'BREAKOUT_QUALITY_TIME_WEIGHT_MODE',
    'BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE',
    'BREAKOUT_QUALITY_EVALUATION_WORKERS',
    'BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION',
    'BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK',
    'BREAKOUT_QUALITY_PRETRAINING_PROFILE',
    'BREAKOUT_QUALITY_PRETRAINING_FAMILY',
    'BREAKOUT_QUALITY_PRETRAINING_STRIDE',
    'BREAKOUT_QUALITY_PRETRAINING_EPOCHS',
    'BREAKOUT_QUALITY_PRETRAINING_BATCH_SIZE',
    'BREAKOUT_QUALITY_PRETRAINING_LEARNING_RATE',
    'BREAKOUT_QUALITY_PRETRAINING_WEIGHT_DECAY',
    'BREAKOUT_QUALITY_PRETRAINING_GRADIENT_CLIP_NORM',
    'BREAKOUT_QUALITY_PRETRAINING_MIN_CROP_BARS',
    'BREAKOUT_QUALITY_PRETRAINING_MASK_PROBABILITY',
    'BREAKOUT_QUALITY_PRETRAINING_CONTRASTIVE_ALPHA',
    'BREAKOUT_QUALITY_PRETRAINING_TEMPORAL_UNIT',
    'BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES',
    'BREAKOUT_QUALITY_EXTRA_HIGH_LENS',
    'BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA',
    'BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE',
    'BREAKOUT_QUALITY_FEATURE_WINDOW_BARS',
    'BREAKOUT_QUALITY_INCEPTION_DEPTH',
    'BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS',
    'BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY',
    'BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS',
    'BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES',
    'BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION',
    'BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS',
    'BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT',
    'BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT',
    'BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS',
    'BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM',
    'BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM',
    'BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO',
    'BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS',
    'BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH',
    'BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS',
    'BREAKOUT_QUALITY_LABEL_HORIZON_BARS',
    'BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS',
    'BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN',
    'BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN',
    'BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO',
    'BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES',
    'BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES',
    'BREAKOUT_QUALITY_EXPERIMENT_PROFILE',
    'BREAKOUT_QUALITY_MODEL_ARCHITECTURE',
    'BREAKOUT_QUALITY_USE_INNER_VALIDATION',
    'build_breakout_quality_default_high_len_values',
    'build_breakout_quality_inception_kernel_sizes',
    'resolve_breakout_quality_inception_receptive_field_bars',
    'ADAMW_ONLY_EXPERIMENT_PROFILE',
    'ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE',
    'AUGMENTATION_NONE',
    'AUGMENTATION_OLD_HISTORY_CONTIGUOUS_MASK',
    'BASELINE_EXPERIMENT_PROFILE',
    'HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE',
    'UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE',
    'UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE',
    'STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE',
    'STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE',
    'TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE',
    'BreakoutQualityExperimentProfile',
    'BreakoutQualityPretrainingProfile',
    'LR_SCHEDULE_LINEAR_WARMUP_COSINE',
    'LR_SCHEDULE_NONE',
    'SUPPORTED_BREAKOUT_QUALITY_AUGMENTATIONS',
    'SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES',
    'SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES',
    'SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES',
    'SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS',
    'SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_SAMPLING_MODES',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_OBJECTIVES',
    'CONTINUOUS_RANKER_TRAINING_OBJECTIVES',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_LABEL_SCOPES',
    'TRAINING_LABEL_SCOPE_ALL',
    'TRAINING_LABEL_SCOPE_PASS_ONLY',
    'TRAINING_OBJECTIVE_BINARY_CLASSIFICATION',
    'TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION',
    'TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING',
    'TRAINING_SAMPLING_ALL_EVENT_ROWS',
    'TRAINING_SAMPLING_UNIQUE_TICKER_DATE',
    'TIME_WEIGHT_MODE_DATE_BALANCED',
    'TIME_WEIGHT_MODE_NONE',
    'TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT',
    'SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES',
    'TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM',
    'TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE',
    'SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS',
    'get_breakout_quality_experiment_profile',
    'resolve_breakout_quality_random_seed',
    'get_breakout_quality_pretraining_profile',
    'build_breakout_quality_pretraining_profile_payload',
    'normalize_breakout_quality_experiment_profile',
    'normalize_breakout_quality_pretraining_profile',
    'BREAKOUT_QUALITY_STRATEGY_BUY_SORT',
    'BREAKOUT_QUALITY_STRATEGY_ADAPT_FIXED_RISK',
    'BREAKOUT_QUALITY_STRATEGY_ADAPT_MAX_POSITION_CAP_PCT',
    'BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE',
    'BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE',
    'BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE',
    'BreakoutQualityWorkflowSettings',
    'SUPPORTED_WORKFLOW_SCORE_SOURCES',
    'SUPPORTED_WORKFLOW_STRATEGY_MODES',
    'WORKFLOW_BUY_SORT_AUTO',
    'WORKFLOW_BUY_SORT_ORIGINAL',
    'WORKFLOW_BUY_SORT_SCORE_DESC',
    'WORKFLOW_SCORE_SOURCE_AUTO',
    'WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME',
    'WORKFLOW_SCORE_SOURCE_FINAL_SELECTION_MODEL_OOS',
    'WORKFLOW_SCORE_SOURCE_SELECTION_POINT_IN_TIME',
    'WORKFLOW_STRATEGY_MODE_AUTO',
    'WORKFLOW_STRATEGY_MODE_HARD_FILTER',
    'WORKFLOW_STRATEGY_MODE_SCORE_RANKING',
    'get_breakout_quality_workflow_settings',
]
