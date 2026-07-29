"""Train the configured breakout-quality temporal classifier with optional inner validation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import time
import warnings

import numpy as np
import pandas as pd

from config.breakout_quality_experiments import (
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
    LR_SCHEDULE_NONE,
    SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES,
    SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS,
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES,
    SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS,
    TIME_WEIGHT_MODE_DATE_BALANCED,
    TIME_WEIGHT_MODE_NONE,
    TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT,
    TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    build_breakout_quality_pretraining_profile_payload,
    get_breakout_quality_experiment_profile,
)

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_PRETRAINING_FAMILY,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_PRETRAINING_STRIDE,
    BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES,
    BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_ALLOW_TF32,
)
from core.display_common import render_elapsed
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.augmentation import (
    TrainingAugmentationPlan,
    apply_training_augmentation,
    build_training_augmentation_plan,
    validate_training_augmentation_sequence_length,
)
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_FILTER_ID,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    LABEL_PASS,
    LABEL_REJECT,
    RUNTIME_SCOPE_NOT_EXPORTED,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_THRESHOLD_SOURCE,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
    TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
)
from filters.breakout_quality.dataset_store import IndexedFeatureBank
from filters.breakout_quality.market_set import IndexedMarketSetBank
from filters.breakout_quality.inference import (
    forward_breakout_quality_model,
    market_batch_to_torch,
    materialize_indexed_feature_inputs,
    strict_parallel_batched_logits,
)
from filters.breakout_quality.lr_schedule import (
    build_learning_rate_schedule_plan as _build_learning_rate_schedule_plan,
    learning_rate_for_optimizer_step as _learning_rate_for_optimizer_step,
)
from filters.breakout_quality.models.spec import (
    MANTIS_V2_FROZEN_LINEAR_V1,
    MOMENT_1_BASE_FROZEN_LINEAR_V1,
    TS2VEC_FROZEN_LINEAR_V1,
)
from filters.breakout_quality.mantis_pretrained import (
    load_mantis_v2_pretrained_encoder_state,
)
from filters.breakout_quality.moment_pretrained import (
    load_moment_pretrained_encoder_state,
)
from filters.breakout_quality.pretraining_store import (
    load_validated_pretrained_encoder_manifest,
    load_validated_pretraining_dataset,
    resolve_pretrained_encoder_paths,
)
from filters.breakout_quality.model import (
    build_model,
    count_trainable_parameters,
    get_model_spec,
    validate_model_sequence_length,
    require_torch,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
    TorchExecutionPlan,
    autocast_context,
    build_grad_scaler,
    resolve_torch_execution_plan,
    seed_torch,
)
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from tools.filters.breakout_quality.common import (
    event_group_summary,
    event_group_keys,
    group_size_weights,
    label_counts,
    load_validated_dataset_bundle,
    load_validated_market_set_bank,
    model_dir,
    write_json,
)


FINAL_REFIT_MODE_MATCHED_OPTIMIZER_STEPS = "matched_optimizer_steps"
FINAL_REFIT_MODE_SELECTED_EPOCHS = "selected_epochs"
FINAL_REFIT_MODES = (
    FINAL_REFIT_MODE_MATCHED_OPTIMIZER_STEPS,
    FINAL_REFIT_MODE_SELECTED_EPOCHS,
)
CLASS_WEIGHT_MODE_NONE = "none"
CLASS_WEIGHT_MODE_INVERSE_FREQUENCY = "inverse_frequency"
CLASS_WEIGHT_MODES = (CLASS_WEIGHT_MODE_NONE, CLASS_WEIGHT_MODE_INVERSE_FREQUENCY)
TIME_WEIGHT_MODES = tuple(SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES)
TRAINING_WEIGHT_REDUCTIONS = tuple(SUPPORTED_BREAKOUT_QUALITY_TRAINING_WEIGHT_REDUCTIONS)
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAMW = "adamw"
OPTIMIZER_NAMES = tuple(SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "訓練 config 指定的 breakout quality 時序 CNN；可選擇完整 Selection 固定 epochs，"
            "或使用 inner validation 選 epoch 後以完整 Selection 重訓"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--epochs",
        type=int,
        default=BREAKOUT_QUALITY_DEFAULT_EPOCHS,
        help=(
            "inner validation 關閉時為正式固定 epoch 數；開啟時為 epoch 搜尋上限"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
        help=(
            "完整 Train／Validation／Selection 評估的推論 batch size；"
            "只影響記憶體與執行速度，不抽樣、不改模型訓練"
        ),
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_WORKERS,
        help=(
            "完整 Train／Validation／Selection 評估的並行 inference worker 數；"
            "每個 worker 維持單執行緒，batch 與 reduction 順序不變"
        ),
    )
    parser.add_argument(
        "--parallel-split-evaluation",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
        help=(
            "是否同時執行 Inner Train 與 Validation 的完整評估；"
            "只平行 read-only inference，不改模型更新或指標口徑"
        ),
    )
    parser.add_argument(
        "--train-prefetch-batches",
        type=int,
        default=BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
        help="預先準備後續訓練 batches 的數量；0 表示關閉，不改 batch 順序",
    )
    parser.add_argument(
        "--preload-feature-bank",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
        help="是否在訓練前將去重 feature bank 與事件小型陣列載入 RAM",
    )
    parser.add_argument(
        "--experiment-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
        default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        help=(
            "訓練實驗設定；模型架構固定由 policy 管理。"
            "profile 可獨立指定 optimizer、step-based LR schedule、augmentation "
            "與 training sampling unit"
        ),
    )
    parser.add_argument("--lr", type=float, default=BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
        help="optimizer weight decay；0 表示關閉",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
        help="每次 optimizer update 前的全域 gradient norm 上限；0 表示關閉",
    )
    parser.add_argument(
        "--final-refit-mode",
        choices=FINAL_REFIT_MODES,
        default=BREAKOUT_QUALITY_FINAL_REFIT_MODE,
        help=(
            "inner validation 選出 epoch 後的完整 Selection 重訓方式；"
            "matched_optimizer_steps 匹配 optimizer updates，selected_epochs 使用舊式相同 epoch 數"
        ),
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=CLASS_WEIGHT_MODES,
        default=BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
        help="Cross-entropy 類別權重模式",
    )
    parser.add_argument(
        "--time-weight-mode",
        choices=TIME_WEIGHT_MODES,
        default=BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
        help="訓練 sample 的時間權重模式",
    )
    parser.add_argument("--seed", type=int, default=BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
        help="在查看 OOS 前固定的評估 threshold；inner validation 不會最佳化此值",
    )
    parser.add_argument(
        "--use-inner-validation",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_USE_INNER_VALIDATION,
        help=(
            "是否以 Selection 最後 N 個月作 inner validation 選 epoch，"
            "之後再用完整 eligible Selection 重訓"
        ),
    )
    parser.add_argument(
        "--inner-validation-months",
        type=int,
        default=BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
        help="inner validation 使用 Selection 尾端幾個月；目前單折 OOS 的 24 個月即 2019-2020",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
        help="inner validation 開啟時，validation loss 未改善可容忍的 epoch 數；0 表示跑滿上限",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
        help="validation loss 至少改善多少才算新最佳",
    )
    parser.add_argument(
        "--device",
        choices=SUPPORTED_TORCH_DEVICES,
        default=BREAKOUT_QUALITY_TORCH_DEVICE,
        help="訓練裝置；auto 優先 CUDA，否則 CPU",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_USE_MIXED_PRECISION,
        help="CUDA 上是否使用 autocast mixed precision",
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=SUPPORTED_MIXED_PRECISION_DTYPES,
        default=BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
        help="auto 優先 bfloat16，裝置不支援時使用 float16",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
        help="是否要求 PyTorch deterministic algorithms",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_ALLOW_TF32,
        help="是否允許 CUDA TF32；預設關閉以維持數值契約",
    )
    parser.add_argument("--min-train-samples", type=int, default=BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES)
    parser.add_argument(
        "--min-validation-samples",
        type=int,
        default=BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    )
    args = parser.parse_args(argv)
    experiment = get_breakout_quality_experiment_profile(args.experiment_profile)
    args.optimizer_name = experiment.optimizer_name
    args.lr_schedule_name = experiment.lr_schedule_name
    args.augmentation_name = experiment.augmentation_name
    args.training_sampling_mode = experiment.training_sampling_mode
    args.training_weight_reduction = experiment.training_weight_reduction
    return args


def validate_training_args(args) -> None:
    max_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    evaluation_batch_size = int(args.evaluation_batch_size)
    evaluation_workers = int(args.evaluation_workers)
    train_prefetch_batches = int(args.train_prefetch_batches)
    experiment = get_breakout_quality_experiment_profile(args.experiment_profile)
    optimizer_name = str(args.optimizer_name).strip().lower()
    lr_schedule_name = str(args.lr_schedule_name).strip().lower()
    augmentation_name = str(args.augmentation_name).strip().lower()
    training_sampling_mode = str(args.training_sampling_mode).strip().lower()
    learning_rate = float(args.lr)
    weight_decay = float(args.weight_decay)
    gradient_clip_norm = float(args.gradient_clip_norm)
    fixed_threshold = float(args.fixed_threshold)
    validation_months = int(args.inner_validation_months)
    patience = int(args.early_stopping_patience)
    min_delta = float(args.early_stopping_min_delta)
    min_train_samples = int(args.min_train_samples)
    min_validation_samples = int(args.min_validation_samples)
    final_refit_mode = str(args.final_refit_mode).strip().lower()
    class_weight_mode = str(args.class_weight_mode).strip().lower()
    time_weight_mode = str(args.time_weight_mode).strip().lower()
    training_weight_reduction = str(args.training_weight_reduction).strip().lower()
    device_request = str(args.device).strip().lower()
    mixed_precision_dtype = str(args.mixed_precision_dtype).strip().lower()
    if device_request not in SUPPORTED_TORCH_DEVICES:
        raise ValueError(f"device 不合法: {device_request}")
    if mixed_precision_dtype not in SUPPORTED_MIXED_PRECISION_DTYPES:
        raise ValueError(f"mixed-precision-dtype 不合法: {mixed_precision_dtype}")
    if (
        max_epochs < 1
        or batch_size < 1
        or evaluation_batch_size < 1
        or evaluation_workers < 1
        or train_prefetch_batches < 0
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not np.isfinite(weight_decay)
        or weight_decay < 0
        or not np.isfinite(gradient_clip_norm)
        or gradient_clip_norm < 0
    ):
        raise ValueError(
            "epochs、batch-size、evaluation-batch-size、evaluation-workers 必須 >=1，"
            "train-prefetch-batches、weight-decay、gradient-clip-norm 必須 >=0，lr 必須 >0"
        )
    if optimizer_name not in OPTIMIZER_NAMES:
        raise ValueError(f"experiment profile optimizer 不合法: {optimizer_name}")
    if lr_schedule_name not in SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES:
        raise ValueError(f"experiment profile LR schedule 不合法: {lr_schedule_name}")
    if optimizer_name != experiment.optimizer_name:
        raise ValueError(
            "experiment profile 與 optimizer_name 不一致: "
            f"profile={experiment.name}, optimizer={optimizer_name}, "
            f"expected={experiment.optimizer_name}"
        )
    if lr_schedule_name != experiment.lr_schedule_name:
        raise ValueError(
            "experiment profile 與 lr_schedule_name 不一致: "
            f"profile={experiment.name}, schedule={lr_schedule_name}, "
            f"expected={experiment.lr_schedule_name}"
        )
    if augmentation_name != experiment.augmentation_name:
        raise ValueError(
            "experiment profile 與 augmentation_name 不一致: "
            f"profile={experiment.name}, augmentation={augmentation_name}, "
            f"expected={experiment.augmentation_name}"
        )
    if training_sampling_mode != experiment.training_sampling_mode:
        raise ValueError(
            "experiment profile 與 training_sampling_mode 不一致: "
            f"profile={experiment.name}, sampling={training_sampling_mode}, "
            f"expected={experiment.training_sampling_mode}"
        )
    if validation_months < 1:
        raise ValueError("inner-validation-months 必須 >=1")
    if patience < 0 or min_delta < 0:
        raise ValueError("early-stopping-patience 與 min-delta 必須 >=0")
    if min_train_samples < 1 or min_validation_samples < 1:
        raise ValueError("min-train-samples 與 min-validation-samples 必須 >=1")
    if not np.isfinite(fixed_threshold) or not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("fixed-threshold 必須介於 0 與 1")
    if final_refit_mode not in FINAL_REFIT_MODES:
        raise ValueError(f"final-refit-mode 不合法: {final_refit_mode}")
    if class_weight_mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"class-weight-mode 不合法: {class_weight_mode}")
    if time_weight_mode not in TIME_WEIGHT_MODES:
        raise ValueError(f"time-weight-mode 不合法: {time_weight_mode}")
    if training_weight_reduction not in TRAINING_WEIGHT_REDUCTIONS:
        raise ValueError(
            f"training weight reduction 不合法: {training_weight_reduction}"
        )
    if training_weight_reduction != experiment.training_weight_reduction:
        raise ValueError(
            "experiment profile 與 training_weight_reduction 不一致: "
            f"profile={experiment.name}, reduction={training_weight_reduction}, "
            f"expected={experiment.training_weight_reduction}"
        )
    if (
        experiment.time_weight_mode is not None
        and time_weight_mode != experiment.time_weight_mode
    ):
        raise ValueError(
            "experiment profile 與 time_weight_mode 不一致: "
            f"profile={experiment.name}, time_weight={time_weight_mode}, "
            f"expected={experiment.time_weight_mode}"
        )
    if (
        time_weight_mode == TIME_WEIGHT_MODE_DATE_BALANCED
        and training_weight_reduction != TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE
    ):
        raise ValueError(
            "date_balanced time weight 必須由 unique_group_date_balanced profile "
            "搭配 fixed_batch_size reduction 使用"
        )
    if (
        training_weight_reduction == TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE
        and time_weight_mode != TIME_WEIGHT_MODE_DATE_BALANCED
    ):
        raise ValueError("fixed_batch_size reduction 只允許 date_balanced time weight")


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _format_date_range(value: object) -> str:
    if not isinstance(value, dict):
        return "-"
    start = str(value.get("start") or "-")
    end = str(value.get("end") or "-")
    return f"{start} ~ {end}"


def _render_epoch_selection_progress(
    *,
    epoch: int,
    max_epochs: int,
    train_loss: float,
    validation_loss: float,
    elapsed_sec: float,
    improved: bool,
    color: bool = False,
) -> str:
    best_marker = " | ★ 新最佳" if improved else ""
    return (
        f"  Epoch {int(epoch):>2}/{int(max_epochs)} | "
        f"Train Loss {float(train_loss):.6f} | "
        f"Val Loss {float(validation_loss):.6f} | "
        f"耗時 {render_elapsed(elapsed_sec, color=color)}{best_marker}"
    )


def _render_epoch_selection_result(
    *,
    completed_epochs: int,
    max_epochs: int,
    best_epoch: int,
    best_validation_loss: float,
) -> str:
    if int(completed_epochs) < int(max_epochs):
        completion = f"Early stopping 於 Epoch {int(completed_epochs)}"
    else:
        completion = f"完成 {int(completed_epochs)} Epoch"
    return (
        f"  結果：{completion} | Best Epoch {int(best_epoch)} | "
        f"最低 Val Loss {float(best_validation_loss):.6f}"
    )


def _render_full_selection_progress(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    elapsed_sec: float,
    color: bool = False,
) -> str:
    return (
        f"  Epoch {int(epoch):>2}/{int(epochs)} | "
        f"Train Loss {float(train_loss):.6f} | "
        f"耗時 {render_elapsed(elapsed_sec, color=color)}"
    )


def _render_training_summary(
    *,
    split_report: dict,
    use_inner_validation: bool,
    final_train_loss: float,
    training_sampling_mode: str,
    time_weight_mode: str,
    training_weight_reduction: str,
    inner_train_sampling_summary: dict[str, object],
    final_refit_sampling_summary: dict[str, object],
) -> str:
    lines = [
        "訓練摘要",
        (
            "  - Training Sampling："
            f"{training_sampling_mode}；"
            f"Inner {int(inner_train_sampling_summary['source_row_count']):,}→"
            f"{int(inner_train_sampling_summary['sampled_row_count']):,}；"
            f"Final {int(final_refit_sampling_summary['source_row_count']):,}→"
            f"{int(final_refit_sampling_summary['sampled_row_count']):,}"
        ),
        (
            "  - Training Weight："
            f"mode={time_weight_mode}；reduction={training_weight_reduction}"
        ),
    ]
    if use_inner_validation:
        lines.extend(
            [
                (
                    "  - Inner Train："
                    f"{_format_count(split_report['selection_train_row_count'])} rows / "
                    f"{_format_count(split_report['selection_train_group_count'])} groups；"
                    f"{_format_date_range(split_report.get('selection_train_date_range'))}"
                ),
                (
                    "  - Validation："
                    f"{_format_count(split_report['inner_validation_row_count'])} rows / "
                    f"{_format_count(split_report['inner_validation_group_count'])} groups；"
                    f"{_format_date_range(split_report.get('inner_validation_date_range'))}"
                ),
            ]
        )
    lines.extend(
        [
            (
                f"  - {'Final Refit' if use_inner_validation else 'Selection'}："
                f"{_format_count(split_report['final_refit_row_count'])} rows / "
                f"{_format_count(split_report['final_refit_group_count'])} groups；"
                f"{_format_date_range(split_report.get('final_refit_date_range'))}"
            ),
            (
                "  - OOS（未參與訓練）："
                f"{_format_count(split_report['oos_evaluable_row_count'])} rows / "
                f"{_format_count(split_report['oos_group_count'])} groups；"
                f"{_format_date_range(split_report.get('oos_date_range'))}"
            ),
            f"  - Final Loss：{float(final_train_loss):.6f}",
        ]
    )
    return "\n".join(lines)


def _class_weights(
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    *,
    mode: str,
):
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == CLASS_WEIGHT_MODE_NONE:
        return np.ones((2,), dtype=np.float32)
    if normalized_mode != CLASS_WEIGHT_MODE_INVERSE_FREQUENCY:
        raise ValueError(f"不支援的 class weight mode: {mode!r}")

    y_arr = y_train.astype(np.int64)
    w_arr = sample_weights.astype(np.float64)
    counts = np.array(
        [
            float(w_arr[y_arr == LABEL_REJECT].sum()),
            float(w_arr[y_arr == LABEL_PASS].sum()),
        ],
        dtype=np.float32,
    )
    counts[counts <= 0] = 1.0
    total = float(counts.sum())
    return total / (2.0 * counts)


def _time_weighted_group_weights(
    events: pd.DataFrame,
    indices: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, dict]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0,), dtype=np.float32), {
            "mode": str(mode),
            "group_count": 0,
            "weight_sum": 0.0,
            "year_group_counts": {},
            "year_weight_multipliers": {},
            "date_count": 0,
        }

    base_weights = group_size_weights(events, idx).astype(np.float64)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == TIME_WEIGHT_MODE_NONE:
        weighted = base_weights
        multipliers: dict[int, float] = {}
        year_group_counts: dict[int, int] = {}
    elif normalized_mode == TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT:
        subset = events.iloc[idx]
        years = pd.to_datetime(subset["date"], errors="raise").dt.year.astype(int)
        keys = event_group_keys(events).iloc[idx].reset_index(drop=True)
        group_frame = pd.DataFrame(
            {
                "group_key": keys.to_numpy(),
                "year": years.to_numpy(),
            }
        ).drop_duplicates("group_key", keep="first")
        counts_series = group_frame.groupby("year", sort=True)["group_key"].size()
        year_group_counts = {int(year): int(count) for year, count in counts_series.items()}
        raw_multipliers = {
            int(year): 1.0 / math.sqrt(float(count))
            for year, count in year_group_counts.items()
        }
        row_multipliers = years.map(raw_multipliers).to_numpy(dtype=np.float64)
        weighted = base_weights * row_multipliers
        base_sum = float(base_weights.sum())
        weighted_sum = float(weighted.sum())
        if weighted_sum <= 0.0:
            raise ValueError("year-balanced sample weights 總和必須 > 0")
        normalization = base_sum / weighted_sum
        weighted *= normalization
        multipliers = {
            year: float(value * normalization)
            for year, value in raw_multipliers.items()
        }
        date_summary = {}
    elif normalized_mode == TIME_WEIGHT_MODE_DATE_BALANCED:
        subset = events.iloc[idx]
        dates = pd.to_datetime(subset["date"], errors="raise").dt.strftime("%Y-%m-%d")
        keys = event_group_keys(events).iloc[idx].reset_index(drop=True)
        group_frame = pd.DataFrame(
            {
                "group_key": keys.to_numpy(),
                "date": dates.to_numpy(),
            }
        ).drop_duplicates("group_key", keep="first")
        mixed_date_counts = group_frame.groupby("group_key", sort=False)["date"].nunique()
        if bool((mixed_date_counts != 1).any()):
            raise ValueError("date-balanced sample weights 發現同 group 對應多個日期")
        counts_series = group_frame.groupby("date", sort=True)["group_key"].size()
        if counts_series.empty or bool((counts_series < 1).any()):
            raise ValueError("date-balanced sample weights 缺少有效交易日 group")
        date_group_counts = {str(date): int(count) for date, count in counts_series.items()}
        raw_multipliers = {
            str(date): 1.0 / float(count)
            for date, count in date_group_counts.items()
        }
        row_multipliers = dates.map(raw_multipliers).to_numpy(dtype=np.float64)
        weighted = base_weights * row_multipliers
        base_sum = float(base_weights.sum())
        weighted_sum = float(weighted.sum())
        if weighted_sum <= 0.0:
            raise ValueError("date-balanced sample weights 總和必須 > 0")
        normalization = base_sum / weighted_sum
        weighted *= normalization
        normalized_row_multipliers = row_multipliers * normalization
        weighted_by_date = pd.Series(weighted, index=dates.to_numpy()).groupby(level=0).sum()
        target_date_weight = base_sum / float(len(date_group_counts))
        date_totals = weighted_by_date.to_numpy(dtype=np.float64)
        group_counts = np.asarray(list(date_group_counts.values()), dtype=np.float64)
        date_summary = {
            "date_count": int(len(date_group_counts)),
            "date_group_count_min": int(group_counts.min()),
            "date_group_count_max": int(group_counts.max()),
            "date_group_count_mean": round(float(group_counts.mean()), 6),
            "date_weight_multiplier_min": round(float(normalized_row_multipliers.min()), 8),
            "date_weight_multiplier_max": round(float(normalized_row_multipliers.max()), 8),
            "target_total_weight_per_date": round(float(target_date_weight), 8),
            "actual_total_weight_per_date_min": round(float(date_totals.min()), 8),
            "actual_total_weight_per_date_max": round(float(date_totals.max()), 8),
        }
        year_group_counts = {}
        multipliers = {}
    else:
        raise ValueError(f"不支援的 time weight mode: {mode!r}")

    if normalized_mode == TIME_WEIGHT_MODE_NONE:
        date_summary = {}
    return weighted.astype(np.float32), {
        "mode": normalized_mode,
        "group_count": int(round(float(base_weights.sum()))),
        "weight_sum": round(float(weighted.sum()), 6),
        "year_group_counts": {str(year): count for year, count in year_group_counts.items()},
        "year_weight_multipliers": {
            str(year): round(value, 8) for year, value in multipliers.items()
        },
        **date_summary,
    }


def _build_sample_weights(
    events: pd.DataFrame,
    row_count: int,
    split_indices: dict[str, np.ndarray],
    *,
    time_weight_mode: str,
) -> tuple[np.ndarray, dict[str, dict]]:
    weights = np.zeros((int(row_count),), dtype=np.float32)
    summaries: dict[str, dict] = {}
    occupied = np.zeros((int(row_count),), dtype=bool)
    for split_name, raw_indices in split_indices.items():
        idx = np.asarray(raw_indices, dtype=np.int64)
        if idx.size == 0:
            summaries[str(split_name)] = {
                "mode": str(time_weight_mode),
                "group_count": 0,
                "weight_sum": 0.0,
                "year_group_counts": {},
                "year_weight_multipliers": {},
                "date_count": 0,
            }
            continue
        if bool(np.any(occupied[idx])):
            raise ValueError(f"sample weight split indices 重疊: {split_name}")
        subset_weights, summary = _time_weighted_group_weights(
            events,
            idx,
            mode=time_weight_mode,
        )
        weights[idx] = subset_weights
        occupied[idx] = True
        summaries[str(split_name)] = summary
    return weights, summaries


def _resolve_training_sampling_indices(
    events: pd.DataFrame,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    mode: str,
    model_spec,
) -> tuple[np.ndarray, dict[str, object]]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError(f"training sampling indices 必須是 1D: {idx.shape}")
    if idx.size == 0:
        raise ValueError("training sampling 至少需要一筆 eligible row")
    if np.unique(idx).size != idx.size:
        raise ValueError("training sampling indices 不可包含重複 row index")
    if int(idx.min()) < 0 or int(idx.max()) >= len(events):
        raise ValueError("training sampling indices 超出 events 範圍")

    normalized_mode = str(mode).strip().lower()
    ordered_idx = np.sort(idx, kind="stable")
    group_keys = event_group_keys(events).iloc[ordered_idx].to_numpy(dtype=object)
    unique_group_count = int(pd.unique(group_keys).size)
    if normalized_mode == TRAINING_SAMPLING_ALL_EVENT_ROWS:
        return idx, {
            "mode": normalized_mode,
            "sampling_unit": "event_row",
            "batch_size_unit": "event_rows",
            "source_row_count": int(idx.size),
            "sampled_row_count": int(idx.size),
            "unique_group_count": unique_group_count,
            "duplicate_rows_removed": 0,
            "representative_rule": None,
            "uses_all_eligible_rows": True,
            "uses_all_eligible_groups": True,
        }

    if normalized_mode != TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
        raise ValueError(f"不支援的 training sampling mode: {mode!r}")
    if bool(model_spec.use_dataset_context) or bool(model_spec.derived_context_features):
        raise ValueError(
            "unique ticker/date sampling 只允許不讀取 dataset/derived context 的 sequence-only architecture"
        )
    required_columns = {"ticker", "date", "group_index"}
    missing = sorted(required_columns - set(events.columns))
    if missing:
        raise KeyError(f"unique group sampling 缺少 events 欄位: {missing}")

    frame = pd.DataFrame(
        {
            "row_index": ordered_idx,
            "group_key": group_keys,
            "feature_group_index": pd.to_numeric(
                events.iloc[ordered_idx]["group_index"], errors="raise"
            ).to_numpy(dtype=np.int64),
            "label": np.asarray(labels, dtype=np.int64)[ordered_idx],
        }
    )
    grouped = frame.groupby("group_key", sort=False)
    mixed_label = grouped["label"].nunique()
    if bool((mixed_label != 1).any()):
        bad = mixed_label[mixed_label != 1].index.astype(str).tolist()[:5]
        raise ValueError(f"unique group sampling 發現 ticker/date 混合 label: {bad}")
    mixed_feature_group = grouped["feature_group_index"].nunique()
    if bool((mixed_feature_group != 1).any()):
        bad = mixed_feature_group[mixed_feature_group != 1].index.astype(str).tolist()[:5]
        raise ValueError(
            f"unique group sampling 發現 ticker/date 對應多個 feature group: {bad}"
        )

    representatives = (
        frame.drop_duplicates("group_key", keep="first")["row_index"]
        .to_numpy(dtype=np.int64)
    )
    if representatives.size != unique_group_count:
        raise RuntimeError(
            "unique group sampling representative 數與 group 數不一致: "
            f"representatives={representatives.size}, groups={unique_group_count}"
        )
    representative_group_keys = (
        event_group_keys(events).iloc[representatives].to_numpy()
    )
    if np.unique(representative_group_keys).size != representatives.size:
        raise RuntimeError("unique group sampling representative 仍含重複 ticker/date")

    return representatives, {
        "mode": normalized_mode,
        "sampling_unit": "unique_ticker_date_group",
        "batch_size_unit": "unique_ticker_date_groups",
        "source_row_count": int(idx.size),
        "sampled_row_count": int(representatives.size),
        "unique_group_count": unique_group_count,
        "duplicate_rows_removed": int(idx.size - representatives.size),
        "representative_rule": "minimum_original_event_row_index",
        "uses_all_eligible_rows": bool(representatives.size == idx.size),
        "uses_all_eligible_groups": True,
    }

def _resolve_final_refit_target_steps(
    *,
    mode: str,
    selected_epoch: int,
    selected_optimizer_steps: int,
    final_batches_per_epoch: int,
) -> tuple[int, bool]:
    normalized_mode = str(mode).strip().lower()
    if int(selected_epoch) < 1 or int(final_batches_per_epoch) < 1:
        raise ValueError("selected_epoch 與 final_batches_per_epoch 必須 >=1")
    if normalized_mode == FINAL_REFIT_MODE_MATCHED_OPTIMIZER_STEPS:
        if int(selected_optimizer_steps) < 1:
            raise ValueError("matched_optimizer_steps 需要 selected_optimizer_steps >=1")
        target = max(int(selected_optimizer_steps), int(final_batches_per_epoch))
        return target, bool(target > int(selected_optimizer_steps))
    if normalized_mode == FINAL_REFIT_MODE_SELECTED_EPOCHS:
        return int(selected_epoch) * int(final_batches_per_epoch), False
    raise ValueError(f"不支援的 final refit mode: {mode!r}")


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def _preload_training_arrays(
    X: IndexedFeatureBank,
    C: np.ndarray,
    y: np.ndarray,
    *,
    enabled: bool,
) -> tuple[IndexedFeatureBank, np.ndarray, np.ndarray]:
    materialized_features, materialized_context = materialize_indexed_feature_inputs(
        X,
        C,
        enabled=enabled,
    )
    y_memory = np.array(y, dtype=np.int64, copy=True, order="C")
    return materialized_features, materialized_context, y_memory


def _materialize_training_microbatch(
    X,
    C: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    rows: np.ndarray,
    market_set_bank: IndexedMarketSetBank | None,
):
    market_batch = (
        None
        if market_set_bank is None
        else market_set_bank.materialize_for_event_rows(X.event_group_index, rows)
    )
    return (
        X[rows],
        C[rows],
        y[rows],
        sample_weights[rows],
        market_batch,
    )


def _build_training_microbatch_rows(
    X,
    shuffled: np.ndarray,
    *,
    batch_size: int,
    market_set_bank: IndexedMarketSetBank | None,
) -> list[np.ndarray]:
    """Build memory-bounded physical microbatches without defining optimizer steps."""

    normalized = np.asarray(shuffled, dtype=np.int64)
    if market_set_bank is None:
        return [
            normalized[start:start + int(batch_size)]
            for start in range(0, len(normalized), int(batch_size))
        ]
    if not isinstance(X, IndexedFeatureBank):
        raise TypeError("market-set training 需要 IndexedFeatureBank")
    market_dates = market_set_bank.market_date_indices_for_event_rows(
        X.event_group_index, normalized
    )
    unique_dates = np.unique(market_dates)
    date_to_block = {
        int(date_index): int(position // market_set_bank.max_dates_per_batch)
        for position, date_index in enumerate(unique_dates.tolist())
    }
    rows_by_block: dict[int, list[int]] = {}
    block_order: list[int] = []
    for row, date_index in zip(normalized.tolist(), market_dates.tolist()):
        block = date_to_block[int(date_index)]
        if block not in rows_by_block:
            rows_by_block[block] = []
            block_order.append(block)
        rows_by_block[block].append(int(row))
    microbatches: list[np.ndarray] = []
    for block in block_order:
        rows = np.asarray(rows_by_block[block], dtype=np.int64)
        for start in range(0, len(rows), int(batch_size)):
            microbatches.append(rows[start:start + int(batch_size)])
    return microbatches


def _build_training_optimizer_batches(
    X,
    shuffled: np.ndarray,
    *,
    batch_size: int,
    market_set_bank: IndexedMarketSetBank | None,
) -> list[list[np.ndarray]]:
    """Pack market microbatches into logical event batches.

    `max_dates_per_batch` is strictly a market-memory boundary. It must not
    increase optimizer updates or silently change the configured 128-group
    training batch. Each returned outer item is one optimizer step; its inner
    arrays are market-memory microbatches whose combined row count is at most
    `batch_size`.
    """

    normalized_batch_size = int(batch_size)
    if normalized_batch_size < 1:
        raise ValueError("batch_size 必須 >=1")
    microbatches = _build_training_microbatch_rows(
        X,
        shuffled,
        batch_size=normalized_batch_size,
        market_set_bank=market_set_bank,
    )
    if market_set_bank is None:
        return [[rows] for rows in microbatches]

    optimizer_batches: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    current_count = 0
    for rows in microbatches:
        offset = 0
        while offset < len(rows):
            remaining = normalized_batch_size - current_count
            take = min(remaining, len(rows) - offset)
            current.append(np.asarray(rows[offset:offset + take], dtype=np.int64))
            current_count += int(take)
            offset += int(take)
            if current_count == normalized_batch_size:
                optimizer_batches.append(current)
                current = []
                current_count = 0
    if current:
        optimizer_batches.append(current)

    expected_rows = np.asarray(shuffled, dtype=np.int64)
    actual_rows = np.concatenate(
        [np.concatenate(batch) for batch in optimizer_batches]
    ) if optimizer_batches else np.empty((0,), dtype=np.int64)
    if actual_rows.size != expected_rows.size or not np.array_equal(
        np.sort(actual_rows), np.sort(expected_rows)
    ):
        raise AssertionError("market-set logical batch packing 未完整保留 shuffled rows")
    return optimizer_batches


def _training_batch_count(
    X,
    indices: np.ndarray,
    *,
    batch_size: int,
    market_set_bank: IndexedMarketSetBank | None,
) -> int:
    return len(
        _build_training_optimizer_batches(
            X,
            np.asarray(indices, dtype=np.int64),
            batch_size=batch_size,
            market_set_bank=market_set_bank,
        )
    )


def _materialize_training_optimizer_batch(
    X,
    C: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    optimizer_batch_rows: list[np.ndarray],
    market_set_bank: IndexedMarketSetBank | None,
):
    return [
        _materialize_training_microbatch(
            X,
            C,
            y,
            sample_weights,
            rows,
            market_set_bank,
        )
        for rows in optimizer_batch_rows
    ]


def _iter_training_batches(
    X,
    C: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    shuffled: np.ndarray,
    *,
    batch_size: int,
    prefetch_batches: int,
    market_set_bank: IndexedMarketSetBank | None,
):
    optimizer_batches = _build_training_optimizer_batches(
        X,
        shuffled,
        batch_size=batch_size,
        market_set_bank=market_set_bank,
    )
    if int(prefetch_batches) <= 0 or len(optimizer_batches) <= 1:
        for optimizer_batch_rows in optimizer_batches:
            yield _materialize_training_optimizer_batch(
                X,
                C,
                y,
                sample_weights,
                optimizer_batch_rows,
                market_set_bank,
            )
        return

    queue_depth = min(int(prefetch_batches), len(optimizer_batches))
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="breakout-quality-prefetch",
    ) as executor:
        pending = deque(
            executor.submit(
                _materialize_training_optimizer_batch,
                X,
                C,
                y,
                sample_weights,
                optimizer_batches[index],
                market_set_bank,
            )
            for index in range(queue_depth)
        )
        next_index = queue_depth
        while pending:
            future = pending.popleft()
            batch_arrays = future.result()
            if next_index < len(optimizer_batches):
                pending.append(
                    executor.submit(
                        _materialize_training_optimizer_batch,
                        X,
                        C,
                        y,
                        sample_weights,
                        optimizer_batches[next_index],
                        market_set_bank,
                    )
                )
                next_index += 1
            yield batch_arrays


def _evaluate(
    torch,
    model,
    X,
    C,
    y,
    indices,
    sample_weights,
    class_weights,
    *,
    evaluation_batch_size: int,
    evaluation_workers: int,
    market_set_bank: IndexedMarketSetBank | None = None,
    execution_plan: TorchExecutionPlan | None = None,
):
    if len(indices) == 0:
        return {
            "loss": None,
            "accuracy": None,
            "pass_rate": None,
            "row_accuracy": None,
            "row_pass_rate": None,
            "row_count": 0,
            "group_weight_sum": 0.0,
        }
    import torch.nn.functional as F

    batch_size = int(evaluation_batch_size)
    if batch_size < 1:
        raise ValueError("evaluation_batch_size 必須 >=1")

    model.eval()
    idx = np.asarray(indices, dtype=np.int64)
    targets_np = y[idx].astype(np.int64)
    weights_np = sample_weights[idx].astype(np.float32)
    loss_items_np = np.empty((idx.size,), dtype=np.float32)
    pred = np.empty((idx.size,), dtype=np.int64)
    logits_np = strict_parallel_batched_logits(
        torch,
        model,
        X,
        C,
        indices=idx,
        batch_size=batch_size,
        workers=evaluation_workers,
        execution_plan=execution_plan,
        market_set_bank=market_set_bank,
    )

    with torch.no_grad():
        for start in range(0, int(idx.size), batch_size):
            stop = min(start + batch_size, int(idx.size))
            logits = torch.from_numpy(logits_np[start:stop])
            target = torch.from_numpy(targets_np[start:stop])
            loss_items = F.cross_entropy(
                logits,
                target,
                weight=(None if class_weights is None else class_weights.detach().cpu()),
                reduction="none",
            )
            loss_items_np[start:stop] = loss_items.cpu().numpy()
            pred[start:stop] = torch.argmax(logits, dim=1).cpu().numpy()

        weights_t = torch.from_numpy(weights_np)
        loss_items_t = torch.from_numpy(loss_items_np)
        denom = torch.clamp(weights_t.sum(), min=1e-12)
        loss = ((loss_items_t * weights_t).sum() / denom).item()

    correct = (pred == targets_np).astype(np.float32)
    pred_pass = (pred == LABEL_PASS).astype(np.float32)
    return {
        "loss": round(float(loss), 6),
        "accuracy": round(_weighted_average(correct, weights_np), 6),
        "pass_rate": round(_weighted_average(pred_pass, weights_np), 6),
        "row_accuracy": round(float(correct.mean()), 6),
        "row_pass_rate": round(float(pred_pass.mean()), 6),
        "row_count": int(idx.size),
        "group_weight_sum": round(float(weights_np.sum()), 6),
    }


def _evaluate_inner_splits(
    torch,
    model,
    X,
    C,
    y,
    train_idx,
    validation_idx,
    sample_weights,
    class_weights,
    *,
    evaluation_batch_size: int,
    evaluation_workers: int,
    parallel: bool,
    market_set_bank: IndexedMarketSetBank | None = None,
    execution_plan: TorchExecutionPlan | None = None,
):
    common_kwargs = {
        "market_set_bank": market_set_bank,
        "evaluation_batch_size": int(evaluation_batch_size),
        "evaluation_workers": int(evaluation_workers),
        "execution_plan": execution_plan,
    }
    if not parallel:
        train_metrics = _evaluate(
            torch, model, X, C, y, train_idx, sample_weights, class_weights,
            **common_kwargs,
        )
        validation_metrics = _evaluate(
            torch, model, X, C, y, validation_idx, sample_weights, class_weights,
            **common_kwargs,
        )
        return train_metrics, validation_metrics

    validation_model = copy.deepcopy(model).eval()
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="breakout-quality-split-eval",
    ) as executor:
        train_future = executor.submit(
            _evaluate,
            torch,
            model,
            X,
            C,
            y,
            train_idx,
            sample_weights,
            class_weights,
            **common_kwargs,
        )
        validation_future = executor.submit(
            _evaluate,
            torch,
            validation_model,
            X,
            C,
            y,
            validation_idx,
            sample_weights,
            class_weights,
            **common_kwargs,
        )
        return train_future.result(), validation_future.result()


def _max_iso_date(events: pd.DataFrame, indices: np.ndarray, column: str) -> str | None:
    if len(indices) == 0:
        return None
    values = pd.to_datetime(
        events.iloc[np.asarray(indices, dtype=np.int64)][column],
        errors="raise",
    )
    return str(values.max().date())


def _build_optimizer(
    torch,
    *,
    optimizer_name: str,
    parameters,
    learning_rate: float,
    weight_decay: float,
):
    normalized = str(optimizer_name).strip().lower()
    kwargs = {
        "lr": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    if normalized == OPTIMIZER_ADAM:
        return torch.optim.Adam(parameters, **kwargs)
    if normalized == OPTIMIZER_ADAMW:
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"optimizer-name 不合法: {optimizer_name}")


def _set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    resolved = float(learning_rate)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = resolved


def _load_pretrained_encoder_for_training(
    torch,
    *,
    filter_id: str,
    experiment_profile: str,
    dataset_summary: dict,
    outer_oos_policy: dict,
    model_spec,
):
    architecture = str(model_spec.architecture)
    if architecture == MOMENT_1_BASE_FROZEN_LINEAR_V1:
        state, external_record = load_moment_pretrained_encoder_state(
            model_spec=model_spec
        )
        return state, None, external_record
    if architecture == MANTIS_V2_FROZEN_LINEAR_V1:
        state, external_record = load_mantis_v2_pretrained_encoder_state(
            model_spec=model_spec
        )
        return state, None, external_record
    if architecture != TS2VEC_FROZEN_LINEAR_V1:
        return None, None, None
    dataset_profile = str(dataset_summary.get("dataset") or "").strip()
    if not dataset_profile:
        raise ValueError("TS2Vec training 缺少 supervised dataset profile")
    source_selection = dataset_summary.get("source_selection")
    if not isinstance(source_selection, dict):
        raise ValueError("TS2Vec training 缺少 supervised dataset source_selection")
    expected_max_tickers = max(
        0, int(source_selection.get("requested_max_tickers", -1))
    )
    pretrain_summary, _windows, _index = load_validated_pretraining_dataset(
        PROJECT_ROOT,
        filter_id,
        dataset_profile=dataset_profile,
        family=BREAKOUT_QUALITY_PRETRAINING_FAMILY,
        stride=int(BREAKOUT_QUALITY_PRETRAINING_STRIDE),
        expected_selection_start=str(outer_oos_policy["selection_start_date"]),
        expected_selection_end=str(outer_oos_policy["selection_end_date"]),
        expected_window_bars=int(DEFAULT_LABEL_POLICY.feature_window_bars),
        expected_max_tickers=expected_max_tickers,
        require_current_source=True,
        load_windows=False,
    )
    paths = resolve_pretrained_encoder_paths(
        PROJECT_ROOT,
        filter_id,
        model_architecture=model_spec.architecture,
        experiment_profile=experiment_profile,
    )
    pretraining_manifest = load_validated_pretrained_encoder_manifest(
        paths,
        expected_architecture=model_spec.architecture,
        expected_experiment_profile=experiment_profile,
        expected_dataset_fingerprint=str(pretrain_summary["configuration_fingerprint"]),
        expected_model_spec=model_spec.as_manifest_payload(),
        expected_pretraining_profile=(
            build_breakout_quality_pretraining_profile_payload(
                BREAKOUT_QUALITY_PRETRAINING_PROFILE
            )
        ),
    )
    payload = torch.load(paths.encoder, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("pretrained encoder payload 必須是 object")
    if payload.get("model_spec") != model_spec.as_manifest_payload():
        raise ValueError("pretrained encoder payload model_spec 不一致")
    expected_pretraining_profile = build_breakout_quality_pretraining_profile_payload(
        BREAKOUT_QUALITY_PRETRAINING_PROFILE
    )
    if payload.get("pretraining_profile") != expected_pretraining_profile:
        raise ValueError("pretrained encoder payload pretraining_profile 不一致")
    if str(payload.get("pretraining_dataset_fingerprint") or "") != str(
        pretrain_summary["configuration_fingerprint"]
    ):
        raise ValueError("pretrained encoder payload dataset fingerprint 不一致")
    state = payload.get("encoder_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("pretrained encoder payload 缺少 encoder_state_dict")
    record = {
        "manifest": pretraining_manifest,
        "dataset_summary": {
            "family": pretrain_summary["family"],
            "stride": int(pretrain_summary["stride"]),
            "window_count": int(pretrain_summary["window_count"]),
            "selection_start_date": pretrain_summary["selection_start_date"],
            "selection_end_date": pretrain_summary["selection_end_date"],
            "configuration_fingerprint": pretrain_summary["configuration_fingerprint"],
        },
    }
    return state, record, None


def _new_training_state(
    torch,
    *,
    feature_count: int,
    context_count: int,
    y: np.ndarray,
    train_idx: np.ndarray,
    sample_weights: np.ndarray,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    class_weight_mode: str,
    seed: int,
    execution_plan: TorchExecutionPlan,
    pretrained_encoder_state=None,
):
    seed_torch(torch, seed=int(seed), plan=execution_plan)
    model = build_model(
        feature_count,
        context_count,
        pretrained_encoder_state=pretrained_encoder_state,
    ).to(execution_plan.device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("breakout quality model 沒有可訓練參數")
    optimizer = _build_optimizer(
        torch,
        optimizer_name=optimizer_name,
        parameters=trainable_parameters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    class_weights_np = _class_weights(
        y[train_idx],
        sample_weights[train_idx],
        mode=class_weight_mode,
    )
    class_weights = (
        None
        if str(class_weight_mode).strip().lower() == CLASS_WEIGHT_MODE_NONE
        else torch.tensor(
            class_weights_np,
            dtype=torch.float32,
            device=execution_plan.device,
        )
    )
    return model, optimizer, class_weights_np, class_weights

def _train_one_epoch(
    torch,
    *,
    model,
    optimizer,
    X: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    sample_weights: np.ndarray,
    class_weights,
    market_set_bank: IndexedMarketSetBank | None,
    batch_size: int,
    shuffle_seed: int,
    prefetch_batches: int,
    gradient_clip_norm: float,
    augmentation_plan: TrainingAugmentationPlan,
    lr_schedule_plan: dict[str, object],
    optimizer_step_offset: int,
    training_weight_reduction: str,
    execution_plan: TorchExecutionPlan,
    grad_scaler=None,
    max_batches: int | None = None,
) -> tuple[float, int, float, float, dict[str, int | str]]:
    import torch.nn.functional as F

    model.train()
    rng = np.random.default_rng(int(shuffle_seed))
    augmentation_rng = np.random.default_rng(
        np.random.SeedSequence([int(shuffle_seed), 0x7A01])
    )
    shuffled = rng.permutation(train_idx)
    batch_losses = []
    completed_batches = 0
    first_learning_rate = float("nan")
    last_learning_rate = float("nan")
    augmentation_summary: dict[str, int | str] = {
        "name": augmentation_plan.name,
        "sample_count": 0,
        "augmented_sample_count": 0,
        "masked_bar_count": 0,
    }
    for optimizer_microbatches in _iter_training_batches(
        X,
        C,
        y,
        sample_weights,
        shuffled,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        market_set_bank=market_set_bank,
    ):
        if max_batches is not None and completed_batches >= int(max_batches):
            break
        if not optimizer_microbatches:
            raise AssertionError("training optimizer batch 不得為空")
        xb_np = np.concatenate([item[0] for item in optimizer_microbatches], axis=0)
        cb_np = np.concatenate([item[1] for item in optimizer_microbatches], axis=0)
        yb_np = np.concatenate([item[2] for item in optimizer_microbatches], axis=0)
        wb_np = np.concatenate([item[3] for item in optimizer_microbatches], axis=0)
        if len(xb_np) > int(batch_size):
            raise AssertionError("logical optimizer batch 超過設定 batch_size")
        xb_np, batch_augmentation = apply_training_augmentation(
            xb_np,
            plan=augmentation_plan,
            rng=augmentation_rng,
        )
        for key in ("sample_count", "augmented_sample_count", "masked_bar_count"):
            augmentation_summary[key] = int(augmentation_summary[key]) + int(
                batch_augmentation[key]
            )
        current_learning_rate = _learning_rate_for_optimizer_step(
            lr_schedule_plan,
            int(optimizer_step_offset) + completed_batches,
        )
        _set_optimizer_learning_rate(optimizer, current_learning_rate)
        if completed_batches == 0:
            first_learning_rate = current_learning_rate
        last_learning_rate = current_learning_rate
        xb = torch.from_numpy(xb_np).to(execution_plan.device)
        cb = torch.from_numpy(cb_np).to(execution_plan.device)
        yb = torch.from_numpy(yb_np).to(execution_plan.device)
        wb = torch.from_numpy(wb_np).to(execution_plan.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch, execution_plan):
            if bool(getattr(model, "requires_market_set", False)):
                if market_set_bank is None:
                    raise ValueError("requires_market_set model 缺少 market_set_bank")
                candidate_embedding = model.encode_candidate(xb)
                logits_parts = []
                event_offset = 0
                for item in optimizer_microbatches:
                    micro_size = int(len(item[0]))
                    market_batch = item[4]
                    if market_batch is None:
                        raise ValueError("market-set optimizer microbatch 缺少 market inputs")
                    sequences, history_mask, valid_stock_mask, event_to_market = (
                        market_batch_to_torch(
                            torch,
                            market_batch,
                            execution_plan.device,
                        )
                    )
                    candidate_microbatch = candidate_embedding[
                        event_offset:event_offset + micro_size
                    ]
                    market_embedding = model.encode_market_for_events(
                        candidate_microbatch,
                        sequences,
                        history_mask,
                        valid_stock_mask,
                        event_to_market,
                    )
                    if event_to_market.ndim != 1 or event_to_market.shape[0] != micro_size:
                        raise ValueError("market microbatch event_to_market shape 不一致")
                    logits_parts.append(
                        model.fuse_embeddings(
                            candidate_microbatch,
                            market_embedding,
                        )
                    )
                    event_offset += micro_size
                if event_offset != int(xb.shape[0]):
                    raise AssertionError("market microbatch rows 未完整覆蓋 logical batch")
                logits = torch.cat(logits_parts, dim=0)
            else:
                if len(optimizer_microbatches) != 1 or optimizer_microbatches[0][4] is not None:
                    raise AssertionError("非 market-set model 收到不合法 microbatch")
                logits = forward_breakout_quality_model(model, xb, cb, None)
            loss_items = F.cross_entropy(
                logits,
                yb,
                weight=class_weights,
                reduction="none",
            )
            weighted_loss_sum = (loss_items * wb).sum()
        if training_weight_reduction == TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM:
            loss = weighted_loss_sum / torch.clamp(wb.sum(), min=1e-12)
        elif training_weight_reduction == TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE:
            # Keep globally normalized date weights intact. Divide by the
            # unweighted logical sample count; market microbatches are only a
            # memory boundary and must not change the optimizer denominator.
            loss = weighted_loss_sum / float(loss_items.numel())
        else:
            raise ValueError(
                f"不支援的 training weight reduction: {training_weight_reduction!r}"
            )
        if grad_scaler is None:
            loss.backward()
            if float(gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(gradient_clip_norm),
                )
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()
            if float(gradient_clip_norm) > 0.0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(gradient_clip_norm),
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        batch_losses.append(float(loss.item()))
        completed_batches += 1
    mean_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
    return (
        mean_loss,
        int(completed_batches),
        float(first_learning_rate),
        float(last_learning_rate),
        augmentation_summary,
    )

def _select_epoch_with_inner_validation(
    torch,
    *,
    X: np.ndarray,
    C: np.ndarray,
    market_set_bank: IndexedMarketSetBank | None,
    y: np.ndarray,
    events: pd.DataFrame,
    train_idx: np.ndarray,
    training_sampling_idx: np.ndarray,
    training_sampling_summary: dict[str, object],
    validation_idx: np.ndarray,
    max_epochs: int,
    batch_size: int,
    optimizer_name: str,
    lr_schedule_name: str,
    lr_warmup_fraction: float,
    lr_minimum_ratio: float,
    augmentation_plan: TrainingAugmentationPlan,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    class_weight_mode: str,
    time_weight_mode: str,
    training_weight_reduction: str,
    seed: int,
    patience: int,
    min_delta: float,
    evaluation_batch_size: int,
    evaluation_workers: int,
    parallel_split_evaluation: bool,
    train_prefetch_batches: int,
    execution_plan: TorchExecutionPlan,
    pretrained_encoder_state=None,
):
    training_sample_weights, sample_weight_summaries = _build_sample_weights(
        events,
        len(y),
        {"inner_train": training_sampling_idx},
        time_weight_mode=time_weight_mode,
    )
    evaluation_sample_weights, _ = _build_sample_weights(
        events,
        len(y),
        {
            "inner_train": train_idx,
            "inner_validation": validation_idx,
        },
        time_weight_mode=TIME_WEIGHT_MODE_NONE,
    )
    model, optimizer, class_weights_np, class_weights = _new_training_state(
        torch,
        feature_count=X.shape[2],
        context_count=C.shape[1],
        y=y,
        train_idx=training_sampling_idx,
        sample_weights=training_sample_weights,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weight_mode=class_weight_mode,
        seed=seed,
        execution_plan=execution_plan,
        pretrained_encoder_state=pretrained_encoder_state,
    )
    grad_scaler = build_grad_scaler(torch, execution_plan)
    batches_per_epoch = _training_batch_count(
        X,
        training_sampling_idx,
        batch_size=batch_size,
        market_set_bank=market_set_bank,
    )
    schedule_plan = _build_learning_rate_schedule_plan(
        schedule_name=lr_schedule_name,
        base_learning_rate=learning_rate,
        total_optimizer_steps=int(max_epochs) * batches_per_epoch,
        warmup_fraction=lr_warmup_fraction,
        minimum_lr_ratio=lr_minimum_ratio,
    )
    cumulative_optimizer_steps = 0
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_metrics = None
    epochs_without_improvement = 0
    history = []
    print("\nEpoch 選擇（依 Validation Loss）")
    color_time = bool(sys.stdout.isatty())
    for epoch in range(1, int(max_epochs) + 1):
        epoch_started = time.perf_counter()
        (
            batch_loss,
            optimizer_steps,
            lr_start,
            lr_end,
            augmentation_summary,
        ) = _train_one_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            X=X,
            C=C,
            y=y,
            train_idx=training_sampling_idx,
            sample_weights=training_sample_weights,
            class_weights=class_weights,
            market_set_bank=market_set_bank,
            batch_size=batch_size,
            shuffle_seed=int(seed) + epoch,
            prefetch_batches=train_prefetch_batches,
            gradient_clip_norm=gradient_clip_norm,
            augmentation_plan=augmentation_plan,
            lr_schedule_plan=schedule_plan,
            optimizer_step_offset=cumulative_optimizer_steps,
            training_weight_reduction=training_weight_reduction,
            execution_plan=execution_plan,
            grad_scaler=grad_scaler,
        )
        cumulative_optimizer_steps += optimizer_steps
        train_metrics, validation_metrics = _evaluate_inner_splits(
            torch,
            model,
            X,
            C,
            y,
            train_idx,
            validation_idx,
            evaluation_sample_weights,
            class_weights,
            market_set_bank=market_set_bank,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
            parallel=parallel_split_evaluation,
            execution_plan=execution_plan,
        )
        validation_loss = float(validation_metrics["loss"])
        epoch_elapsed = time.perf_counter() - epoch_started
        improved = validation_loss < (
            best_validation_loss - float(min_delta)
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = int(epoch)
            best_validation_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": int(epoch),
                "batch_loss": round(float(batch_loss), 6),
                "optimizer_steps": int(optimizer_steps),
                "cumulative_optimizer_steps": int(cumulative_optimizer_steps),
                "learning_rate_start": round(float(lr_start), 12),
                "learning_rate_end": round(float(lr_end), 12),
                "training_augmentation": augmentation_summary,
                "inner_train_metrics": train_metrics,
                "inner_validation_metrics": validation_metrics,
                "elapsed_sec": round(float(epoch_elapsed), 3),
                "is_best_epoch": bool(improved),
            }
        )
        print(
            _render_epoch_selection_progress(
                epoch=epoch,
                max_epochs=max_epochs,
                train_loss=train_metrics["loss"],
                validation_loss=validation_loss,
                elapsed_sec=epoch_elapsed,
                improved=improved,
                color=color_time,
            )
        )
        if int(patience) > 0 and epochs_without_improvement >= int(patience):
            break
    if best_epoch < 1 or best_validation_metrics is None:
        raise ValueError("inner validation 無法選出合法 best_epoch")
    print(
        _render_epoch_selection_result(
            completed_epochs=len(history),
            max_epochs=max_epochs,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )
    )
    return {
        "best_epoch": int(best_epoch),
        "best_validation_loss": round(float(best_validation_loss), 6),
        "best_validation_metrics": best_validation_metrics,
        "completed_epochs": int(len(history)),
        "batches_per_epoch": int(batches_per_epoch),
        "best_optimizer_steps": int(
            int(best_epoch) * math.ceil(len(training_sampling_idx) / int(batch_size))
        ),
        "sample_weight_summaries": sample_weight_summaries,
        "training_sampling": dict(training_sampling_summary),
        "learning_rate_schedule": {
            **schedule_plan,
            "actual_optimizer_steps": int(cumulative_optimizer_steps),
            "last_applied_learning_rate": round(
                float(history[-1]["learning_rate_end"]),
                12,
            ),
        },
        "history": history,
        "class_weights_reject_pass": [
            round(float(value), 8) for value in class_weights_np.tolist()
        ],
    }


def _fit_full_selection(
    torch,
    *,
    X: np.ndarray,
    C: np.ndarray,
    market_set_bank: IndexedMarketSetBank | None,
    y: np.ndarray,
    events: pd.DataFrame,
    train_idx: np.ndarray,
    training_sampling_idx: np.ndarray,
    training_sampling_summary: dict[str, object],
    epochs: int,
    target_optimizer_steps: int | None,
    batch_size: int,
    optimizer_name: str,
    lr_schedule_name: str,
    lr_warmup_fraction: float,
    lr_minimum_ratio: float,
    augmentation_plan: TrainingAugmentationPlan,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    class_weight_mode: str,
    time_weight_mode: str,
    training_weight_reduction: str,
    seed: int,
    phase_name: str,
    evaluation_batch_size: int,
    evaluation_workers: int,
    train_prefetch_batches: int,
    execution_plan: TorchExecutionPlan,
    pretrained_encoder_state=None,
):
    training_sample_weights, sample_weight_summaries = _build_sample_weights(
        events,
        len(y),
        {"final_refit": training_sampling_idx},
        time_weight_mode=time_weight_mode,
    )
    evaluation_sample_weights, _ = _build_sample_weights(
        events,
        len(y),
        {"final_refit": train_idx},
        time_weight_mode=TIME_WEIGHT_MODE_NONE,
    )
    model, optimizer, class_weights_np, class_weights = _new_training_state(
        torch,
        feature_count=X.shape[2],
        context_count=C.shape[1],
        y=y,
        train_idx=training_sampling_idx,
        sample_weights=training_sample_weights,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weight_mode=class_weight_mode,
        seed=seed,
        execution_plan=execution_plan,
        pretrained_encoder_state=pretrained_encoder_state,
    )
    grad_scaler = build_grad_scaler(torch, execution_plan)
    batches_per_epoch = _training_batch_count(
        X,
        training_sampling_idx,
        batch_size=batch_size,
        market_set_bank=market_set_bank,
    )
    requested_steps = (
        int(target_optimizer_steps)
        if target_optimizer_steps is not None
        else int(epochs) * batches_per_epoch
    )
    if requested_steps < batches_per_epoch:
        raise ValueError(
            "final refit optimizer steps 不足以讓全部 eligible training sampling units 至少使用一次: "
            f"target={requested_steps}, minimum={batches_per_epoch}"
        )
    planned_cycles = int(math.ceil(requested_steps / batches_per_epoch))
    equivalent_epochs = float(requested_steps) / float(batches_per_epoch)
    schedule_plan = _build_learning_rate_schedule_plan(
        schedule_name=lr_schedule_name,
        base_learning_rate=learning_rate,
        total_optimizer_steps=requested_steps,
        warmup_fraction=lr_warmup_fraction,
        minimum_lr_ratio=lr_minimum_ratio,
    )

    history = []
    phase_title = (
        "完整 Selection 重訓"
        if phase_name == "full_refit"
        else "完整 Selection 訓練"
    )
    if target_optimizer_steps is None:
        print(f"\n{phase_title}（{int(epochs)} Epoch）")
    else:
        print(
            f"\n{phase_title}（matched steps={requested_steps:,}；"
            f"約 {equivalent_epochs:.3f} Epoch）"
        )
    color_time = bool(sys.stdout.isatty())
    completed_optimizer_steps = 0
    for cycle in range(1, planned_cycles + 1):
        epoch_started = time.perf_counter()
        remaining_steps = requested_steps - completed_optimizer_steps
        max_batches = min(batches_per_epoch, remaining_steps)
        (
            batch_loss,
            optimizer_steps,
            lr_start,
            lr_end,
            augmentation_summary,
        ) = _train_one_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            X=X,
            C=C,
            y=y,
            train_idx=training_sampling_idx,
            sample_weights=training_sample_weights,
            class_weights=class_weights,
            market_set_bank=market_set_bank,
            batch_size=batch_size,
            shuffle_seed=int(seed) + cycle,
            prefetch_batches=train_prefetch_batches,
            gradient_clip_norm=gradient_clip_norm,
            augmentation_plan=augmentation_plan,
            lr_schedule_plan=schedule_plan,
            optimizer_step_offset=completed_optimizer_steps,
            training_weight_reduction=training_weight_reduction,
            execution_plan=execution_plan,
            grad_scaler=grad_scaler,
            max_batches=max_batches,
        )
        if optimizer_steps != max_batches:
            raise RuntimeError(
                "final refit 實際 optimizer steps 與計畫不一致: "
                f"actual={optimizer_steps}, planned={max_batches}"
            )
        completed_optimizer_steps += optimizer_steps
        metrics = _evaluate(
            torch,
            model,
            X,
            C,
            y,
            train_idx,
            evaluation_sample_weights,
            class_weights,
            market_set_bank=market_set_bank,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
            execution_plan=execution_plan,
        )
        epoch_elapsed = time.perf_counter() - epoch_started
        cycle_fraction = float(optimizer_steps) / float(batches_per_epoch)
        history.append(
            {
                "epoch": int(cycle),
                "epoch_fraction": round(cycle_fraction, 8),
                "batch_loss": round(float(batch_loss), 6),
                "optimizer_steps": int(optimizer_steps),
                "cumulative_optimizer_steps": int(completed_optimizer_steps),
                "target_optimizer_steps": int(requested_steps),
                "learning_rate_start": round(float(lr_start), 12),
                "learning_rate_end": round(float(lr_end), 12),
                "training_augmentation": augmentation_summary,
                "selection_metrics": metrics,
                "elapsed_sec": round(float(epoch_elapsed), 3),
            }
        )
        progress = _render_full_selection_progress(
            epoch=cycle,
            epochs=planned_cycles,
            train_loss=metrics["loss"],
            elapsed_sec=epoch_elapsed,
            color=color_time,
        )
        if cycle_fraction < 1.0:
            progress += f" | Partial {cycle_fraction:.3f} Epoch"
        print(progress)
    if not history:
        raise ValueError("完整 Selection 訓練至少需要 1 個 optimizer step")
    if completed_optimizer_steps != requested_steps:
        raise RuntimeError(
            "final refit optimizer steps 未完成: "
            f"actual={completed_optimizer_steps}, target={requested_steps}"
        )
    final_metrics = dict(history[-1]["selection_metrics"])
    return {
        "model": model,
        "history": history,
        "final_metrics": final_metrics,
        "class_weights_reject_pass": [
            round(float(value), 8) for value in class_weights_np.tolist()
        ],
        "sample_weight_summaries": sample_weight_summaries,
        "training_sampling": dict(training_sampling_summary),
        "batches_per_epoch": int(batches_per_epoch),
        "target_optimizer_steps": int(requested_steps),
        "actual_optimizer_steps": int(completed_optimizer_steps),
        "equivalent_epochs": round(float(equivalent_epochs), 8),
        "completed_epoch_cycles": int(planned_cycles),
        "last_epoch_fraction": round(
            float(history[-1]["epoch_fraction"]),
            8,
        ),
        "learning_rate_schedule": {
            **schedule_plan,
            "actual_optimizer_steps": int(completed_optimizer_steps),
            "last_applied_learning_rate": round(
                float(history[-1]["learning_rate_end"]),
                12,
            ),
        },
    }

def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    max_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    evaluation_batch_size = int(args.evaluation_batch_size)
    evaluation_workers = int(args.evaluation_workers)
    parallel_split_evaluation = bool(args.parallel_split_evaluation)
    train_prefetch_batches = int(args.train_prefetch_batches)
    preload_feature_bank = bool(args.preload_feature_bank)
    experiment = get_breakout_quality_experiment_profile(args.experiment_profile)
    experiment_profile = experiment.name
    training_sampling_mode = experiment.training_sampling_mode
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    optimizer_name = experiment.optimizer_name
    lr_schedule_name = experiment.lr_schedule_name
    lr_schedule_parameters = experiment.lr_schedule_parameters()
    lr_warmup_fraction = float(lr_schedule_parameters.get("warmup_fraction", 0.0))
    lr_minimum_ratio = float(lr_schedule_parameters.get("minimum_lr_ratio", 1.0))
    augmentation_parameters = experiment.augmentation_parameters()
    augmentation_plan = build_training_augmentation_plan(
        name=experiment.augmentation_name,
        parameters=augmentation_parameters,
    )
    learning_rate = float(args.lr)
    weight_decay = float(args.weight_decay)
    gradient_clip_norm = float(args.gradient_clip_norm)
    fixed_threshold = float(args.fixed_threshold)
    use_inner_validation = bool(args.use_inner_validation)
    validation_months = int(args.inner_validation_months)
    patience = int(args.early_stopping_patience)
    min_delta = float(args.early_stopping_min_delta)
    min_train_samples = int(args.min_train_samples)
    min_validation_samples = int(args.min_validation_samples)
    final_refit_mode = str(args.final_refit_mode).strip().lower()
    class_weight_mode = str(args.class_weight_mode).strip().lower()
    time_weight_mode = str(args.time_weight_mode).strip().lower()
    training_weight_reduction = experiment.training_weight_reduction
    validate_training_args(args)

    dataset_summary, X, C, y, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=True,
    )
    market_set_bank = (
        load_validated_market_set_bank(
            args.filter_id,
            dataset_summary=dataset_summary,
            expected_model_spec=model_spec,
        )
        if bool(model_spec.requires_market_set)
        else None
    )
    X, C, y = _preload_training_arrays(
        X,
        C,
        y,
        enabled=preload_feature_bank,
    )
    validate_training_augmentation_sequence_length(
        augmentation_plan,
        sequence_length=int(X.shape[1]),
    )
    validate_model_sequence_length(model_spec, int(X.shape[1]))
    gc.collect()

    torch, _nn = require_torch()
    execution_plan = resolve_torch_execution_plan(
        torch,
        requested_device=str(args.device),
        mixed_precision=bool(args.mixed_precision),
        mixed_precision_dtype=str(args.mixed_precision_dtype),
        deterministic_algorithms=bool(args.deterministic_algorithms),
        allow_tf32=bool(args.allow_tf32),
    )
    if execution_plan.device_type == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            if "cannot set number of interop threads" not in str(exc):
                raise
            warnings.warn(
                f"torch.set_num_interop_threads(1) skipped: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    if execution_plan.device_type == "cuda" and parallel_split_evaluation:
        raise ValueError("CUDA 訓練不允許 parallel-split-evaluation；請使用單一 GPU serial evaluation")

    print(
        "torch="
        f"device={execution_plan.device_type}, "
        f"mixed_precision={execution_plan.mixed_precision_enabled}, "
        f"dtype={execution_plan.autocast_dtype_name}, "
        f"deterministic={execution_plan.deterministic_algorithms}, "
        f"tf32={execution_plan.allow_tf32}"
    )

    source_data_range = dataset_summary.get("source_data_date_range")
    source_data_end = (
        str(source_data_range.get("end") or "").strip()
        if isinstance(source_data_range, dict)
        else ""
    )
    if not source_data_end:
        source_data_end = str(
            pd.to_datetime(
                events["label_eval_end_date"],
                errors="raise",
            ).max().date()
        )
    outer_oos_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=source_data_end,
    )
    (
        pretrained_encoder_state,
        pretraining_record,
        external_pretrained_encoder_record,
    ) = _load_pretrained_encoder_for_training(
        torch,
        filter_id=args.filter_id,
        experiment_profile=experiment_profile,
        dataset_summary=dataset_summary,
        outer_oos_policy=outer_oos_policy,
        model_spec=model_spec,
    )
    early_stopping_enabled = bool(use_inner_validation and patience > 0)
    (
        split_assignments,
        inner_train_idx,
        inner_validation_idx,
        final_refit_idx,
        _oos_idx,
        split_report,
    ) = build_selection_oos_split_assignments(
        events,
        y,
        outer_policy=outer_oos_policy,
        use_inner_validation=use_inner_validation,
        inner_validation_months=validation_months,
        early_stopping_enabled=early_stopping_enabled,
    )
    if int(split_report.get("overlap_group_count", 0)) != 0:
        raise ValueError(f"Selection/OOS group overlap 不應發生: {split_report}")
    if int(split_report.get("overlap_event_date_count", 0)) != 0:
        raise ValueError(f"Selection/OOS event date overlap 不應發生: {split_report}")
    if int(split_report.get("inner_train_validation_overlap_group_count", 0)) != 0:
        raise ValueError(f"inner train/validation group overlap 不應發生: {split_report}")
    if int(split_report.get("inner_train_validation_overlap_event_date_count", 0)) != 0:
        raise ValueError(f"inner train/validation event date overlap 不應發生: {split_report}")
    if len(inner_train_idx) < min_train_samples:
        raise ValueError(
            "可訓練樣本不足: "
            f"inner_train={len(inner_train_idx)}, min={min_train_samples}, "
            f"labels={label_counts(y)}"
        )
    if len(set(y[inner_train_idx].tolist())) < 2:
        raise ValueError(
            "inner train 必須同時包含 PASS/REJECT，"
            f"labels={label_counts(y[inner_train_idx])}"
        )
    if use_inner_validation:
        if len(inner_validation_idx) < min_validation_samples:
            raise ValueError(
                "inner validation 樣本不足: "
                f"validation={len(inner_validation_idx)}, "
                f"min={min_validation_samples}"
            )
        if len(set(y[inner_validation_idx].tolist())) < 2:
            raise ValueError(
                "inner validation 必須同時包含 PASS/REJECT，"
                f"labels={label_counts(y[inner_validation_idx])}"
            )

    inner_train_idx = np.asarray(inner_train_idx, dtype=np.int64)
    inner_validation_idx = np.asarray(inner_validation_idx, dtype=np.int64)
    final_refit_idx = np.asarray(final_refit_idx, dtype=np.int64)
    inner_train_sampling_idx, inner_train_sampling_summary = (
        _resolve_training_sampling_indices(
            events,
            y,
            inner_train_idx,
            mode=training_sampling_mode,
            model_spec=model_spec,
        )
    )
    final_refit_sampling_idx, final_refit_sampling_summary = (
        _resolve_training_sampling_indices(
            events,
            y,
            final_refit_idx,
            mode=training_sampling_mode,
            model_spec=model_spec,
        )
    )
    if len(inner_train_sampling_idx) < min_train_samples:
        raise ValueError(
            "可訓練 sampling units 不足: "
            f"sampled={len(inner_train_sampling_idx)}, min={min_train_samples}, "
            f"mode={training_sampling_mode}"
        )
    if len(set(y[inner_train_sampling_idx].tolist())) < 2:
        raise ValueError(
            "sampled inner train 必須同時包含 PASS/REJECT，"
            f"mode={training_sampling_mode}, labels={label_counts(y[inner_train_sampling_idx])}"
        )

    epoch_selection = None
    selected_optimizer_steps = None
    minimum_full_pass_applied = False
    if use_inner_validation:
        epoch_selection = _select_epoch_with_inner_validation(
            torch,
            X=X,
            C=C,
            market_set_bank=market_set_bank,
            y=y,
            events=events,
            train_idx=inner_train_idx,
            training_sampling_idx=inner_train_sampling_idx,
            training_sampling_summary=inner_train_sampling_summary,
            validation_idx=inner_validation_idx,
            max_epochs=max_epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            lr_schedule_name=lr_schedule_name,
            lr_warmup_fraction=lr_warmup_fraction,
            lr_minimum_ratio=lr_minimum_ratio,
            augmentation_plan=augmentation_plan,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            class_weight_mode=class_weight_mode,
            time_weight_mode=time_weight_mode,
            training_weight_reduction=training_weight_reduction,
            seed=int(args.seed),
            patience=patience,
            min_delta=min_delta,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
            parallel_split_evaluation=parallel_split_evaluation,
            train_prefetch_batches=train_prefetch_batches,
            execution_plan=execution_plan,
            pretrained_encoder_state=pretrained_encoder_state,
        )
        selected_epoch = int(epoch_selection["best_epoch"])
        selected_optimizer_steps = int(epoch_selection["best_optimizer_steps"])
        training_mode = TRAINING_MODE_INNER_VALIDATION_FULL_REFIT
        epoch_selection_source = "inner_validation_loss"
        phase_name = "full_refit"
        final_batches_per_epoch = _training_batch_count(
            X,
            final_refit_sampling_idx,
            batch_size=batch_size,
            market_set_bank=market_set_bank,
        )
        resolved_target_steps, minimum_full_pass_applied = (
            _resolve_final_refit_target_steps(
                mode=final_refit_mode,
                selected_epoch=selected_epoch,
                selected_optimizer_steps=selected_optimizer_steps,
                final_batches_per_epoch=final_batches_per_epoch,
            )
        )
        final_target_optimizer_steps = (
            resolved_target_steps
            if final_refit_mode == FINAL_REFIT_MODE_MATCHED_OPTIMIZER_STEPS
            else None
        )
    else:
        selected_epoch = max_epochs
        training_mode = TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION
        epoch_selection_source = "fixed_cli_epochs"
        phase_name = "fixed"
        final_target_optimizer_steps = None
        final_refit_mode = FINAL_REFIT_MODE_SELECTED_EPOCHS

    final_fit = _fit_full_selection(
        torch,
        X=X,
        C=C,
        market_set_bank=market_set_bank,
        y=y,
        events=events,
        train_idx=final_refit_idx,
        training_sampling_idx=final_refit_sampling_idx,
        training_sampling_summary=final_refit_sampling_summary,
        epochs=selected_epoch,
        target_optimizer_steps=final_target_optimizer_steps,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        lr_schedule_name=lr_schedule_name,
        lr_warmup_fraction=lr_warmup_fraction,
        lr_minimum_ratio=lr_minimum_ratio,
        augmentation_plan=augmentation_plan,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        class_weight_mode=class_weight_mode,
        time_weight_mode=time_weight_mode,
        training_weight_reduction=training_weight_reduction,
        seed=int(args.seed),
        phase_name=phase_name,
        evaluation_batch_size=evaluation_batch_size,
        evaluation_workers=evaluation_workers,
        train_prefetch_batches=train_prefetch_batches,
        execution_plan=execution_plan,
        pretrained_encoder_state=pretrained_encoder_state,
    )
    final_refit_plan = {
        "mode": final_refit_mode,
        "inner_train_row_count": int(len(inner_train_idx)),
        "inner_train_sampling_row_count": int(len(inner_train_sampling_idx)),
        "final_refit_row_count": int(len(final_refit_idx)),
        "final_refit_sampling_row_count": int(len(final_refit_sampling_idx)),
        "training_sampling_mode": training_sampling_mode,
        "training_weight_reduction": training_weight_reduction,
        "batch_size": int(batch_size),
        "batch_size_unit": final_refit_sampling_summary["batch_size_unit"],
        "inner_train_batches_per_epoch": (
            int(epoch_selection["batches_per_epoch"])
            if epoch_selection is not None
            else None
        ),
        "selected_epoch": int(selected_epoch),
        "selected_optimizer_steps": selected_optimizer_steps,
        "final_refit_batches_per_epoch": int(final_fit["batches_per_epoch"]),
        "target_optimizer_steps": int(final_fit["target_optimizer_steps"]),
        "actual_optimizer_steps": int(final_fit["actual_optimizer_steps"]),
        "equivalent_epochs": float(final_fit["equivalent_epochs"]),
        "completed_epoch_cycles": int(final_fit["completed_epoch_cycles"]),
        "last_epoch_fraction": float(final_fit["last_epoch_fraction"]),
        "minimum_full_pass_applied": bool(minimum_full_pass_applied),
        "all_eligible_selection_rows_seen_at_least_once": bool(
            final_refit_sampling_summary["uses_all_eligible_rows"]
            and int(final_fit["actual_optimizer_steps"])
            >= int(final_fit["batches_per_epoch"])
        ),
        "all_eligible_selection_groups_seen_at_least_once": bool(
            final_refit_sampling_summary["uses_all_eligible_groups"]
            and int(final_fit["actual_optimizer_steps"])
            >= int(final_fit["batches_per_epoch"])
        ),
    }
    model = final_fit["model"]
    final_train_metrics = final_fit["final_metrics"]

    out_dir = model_dir(args.filter_id, experiment_profile=experiment_profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = resolve_filter_artifact_paths(
        PROJECT_ROOT, args.filter_id, experiment_profile=experiment_profile
    )
    for stale_path in (
        artifact_paths.score_path,
        resolve_filter_research_score_path(
            PROJECT_ROOT, args.filter_id, experiment_profile=experiment_profile
        ),
        resolve_filter_research_manifest_path(
            PROJECT_ROOT, args.filter_id, experiment_profile=experiment_profile
        ),
    ):
        stale_path.unlink(missing_ok=True)

    trainable_parameter_count = count_trainable_parameters(model)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    frozen_parameter_count = total_parameter_count - trainable_parameter_count
    torch.save(
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "feature_count": int(X.shape[2]),
            "context_count": int(C.shape[1]),
            "sequence_length": int(X.shape[1]),
            "model_spec": model_spec.as_manifest_payload(),
            "experiment_profile": experiment_profile,
            "experiment_settings": experiment.as_manifest_payload(),
            "torch_execution": execution_plan.as_manifest_payload(),
            "trainable_parameter_count": int(trainable_parameter_count),
            "total_parameter_count": int(total_parameter_count),
            "frozen_parameter_count": int(frozen_parameter_count),
            "self_supervised_pretraining": pretraining_record,
            "external_pretrained_encoder": external_pretrained_encoder_record,
            "market_set_contract": (
                dataset_summary.get("market_set_contract")
                if bool(model_spec.requires_market_set)
                else None
            ),
            "market_set_artifacts": (
                dataset_summary.get("market_set_artifacts")
                if bool(model_spec.requires_market_set)
                else None
            ),
        },
        artifact_paths.model_path,
    )
    split_assignments.to_csv(
        artifact_paths.split_path,
        index=False,
        encoding="utf-8-sig",
    )

    model_information_cutoff = _max_iso_date(
        events,
        final_refit_idx,
        "label_eval_end_date",
    )
    if (
        model_information_cutoff is None
        or model_information_cutoff >= str(outer_oos_policy["oos_start_date"])
    ):
        raise ValueError(
            "breakout quality model_information_cutoff 必須早於既有 OOS 起點: "
            f"cutoff={model_information_cutoff}, "
            f"oos_start={outer_oos_policy['oos_start_date']}"
        )
    split_record = {
        **build_file_manifest(artifact_paths.split_path),
        "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
        "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
        "columns": list(split_assignments.columns),
        "row_count": int(len(split_assignments)),
        "outer_split_counts": split_report["outer_split_counts"],
        "selection_role_counts": split_report["selection_role_counts"],
        "group_key": "ticker/date/high_len",
        "source_dataset_artifacts": dataset_summary.get("dataset_artifacts"),
        "source_market_set_artifacts": (
            dataset_summary.get("market_set_artifacts")
            if bool(model_spec.requires_market_set)
            else None
        ),
    }
    manifest = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "filter_family": FILTER_FAMILY,
        "filter_id": args.filter_id,
        "model_architecture": model_spec.architecture,
        "experiment_profile": experiment_profile,
        "experiment_settings": experiment.as_manifest_payload(),
        "model_spec": model_spec.as_manifest_payload(),
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "frozen_parameter_count": int(frozen_parameter_count),
        "self_supervised_pretraining": pretraining_record,
        "external_pretrained_encoder": external_pretrained_encoder_record,
        "market_set_contract": (
            dataset_summary.get("market_set_contract")
            if bool(model_spec.requires_market_set)
            else None
        ),
        "market_set_artifacts": (
            dataset_summary.get("market_set_artifacts")
            if bool(model_spec.requires_market_set)
            else None
        ),
        "sequence_length": int(X.shape[1]),
        "model_filename": DEFAULT_MODEL_FILENAME,
        "manifest_filename": DEFAULT_MANIFEST_FILENAME,
        "score_filename": DEFAULT_SCORE_FILENAME,
        "split_filename": DEFAULT_SPLIT_FILENAME,
        "model": build_file_manifest(artifact_paths.model_path),
        "split_assignments": split_record,
        "outer_oos_policy": outer_oos_policy,
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "score_decision": {
            "score_column": SCORE_COLUMN,
            "comparison": SCORE_COMPARISON,
            "threshold_source": SCORE_THRESHOLD_SOURCE,
        },
        "fixed_evaluation_threshold": fixed_threshold,
        "threshold_policy": {
            "mode": "fixed_before_oos",
            "evaluation_threshold": fixed_threshold,
            "runtime_source": SCORE_THRESHOLD_SOURCE,
            "optimized_by_train": False,
            "oos_tuning_allowed": False,
        },
        "training_mode": training_mode,
        "max_epochs": max_epochs,
        "selected_epoch": selected_epoch,
        "fixed_epochs": selected_epoch,
        "completed_epochs": int(final_fit["completed_epoch_cycles"]),
        "final_refit_plan": final_refit_plan,
        "epoch_selection_source": epoch_selection_source,
        "early_stopping_enabled": early_stopping_enabled,
        "early_stopping_patience": patience if use_inner_validation else None,
        "early_stopping_min_delta": min_delta if use_inner_validation else None,
        "inner_validation_used": use_inner_validation,
        "inner_validation_months": validation_months if use_inner_validation else None,
        "training_uses_all_eligible_selection_rows": bool(
            final_refit_sampling_summary["uses_all_eligible_rows"]
        ),
        "training_uses_all_eligible_selection_groups": True,
        "training_sampling": {
            "mode": training_sampling_mode,
            "inner_train": inner_train_sampling_summary,
            "final_refit": final_refit_sampling_summary,
            "validation_sampling_enabled": False,
            "oos_sampling_enabled": False,
        },
        "oos_predictions_used_during_training": False,
        "oos_metrics_emitted_by_train": False,
        "model_information_cutoff": model_information_cutoff,
        "policy": dataset_summary.get("policy"),
        "source_dataset": dataset_summary,
        "event_group_summary": event_group_summary(events, y),
        "split_report": split_report,
        "inner_train_label_counts": label_counts(y[inner_train_idx]),
        "inner_validation_label_counts": (
            label_counts(y[inner_validation_idx]) if use_inner_validation else None
        ),
        "final_refit_label_counts": label_counts(y[final_refit_idx]),
        "inner_validation_epoch_selection": epoch_selection,
        "final_refit_metrics": final_train_metrics,
        "class_weight_mode": class_weight_mode,
        "class_weights_reject_pass": final_fit["class_weights_reject_pass"],
        "time_weight_mode": time_weight_mode,
        "training_weight_reduction": training_weight_reduction,
        "sample_weight_summaries": {
            "epoch_selection": (
                epoch_selection.get("sample_weight_summaries")
                if isinstance(epoch_selection, dict)
                else None
            ),
            "final_refit": final_fit["sample_weight_summaries"]["final_refit"],
        },
        "seed": int(args.seed),
        "optimizer_name": optimizer_name,
        "lr_schedule_name": lr_schedule_name,
        "augmentation_name": augmentation_plan.name,
        "training_augmentation": {
            "name": augmentation_plan.name,
            "parameters": augmentation_parameters,
            "epoch_selection": (
                [
                    row.get("training_augmentation")
                    for row in epoch_selection.get("history", [])
                ]
                if isinstance(epoch_selection, dict)
                else None
            ),
            "final_refit": [
                row.get("training_augmentation")
                for row in final_fit.get("history", [])
            ],
            "validation_augmented": False,
            "oos_augmented": False,
        },
        "learning_rate_schedule": {
            "name": lr_schedule_name,
            "parameters": lr_schedule_parameters,
            "epoch_selection": (
                epoch_selection.get("learning_rate_schedule")
                if isinstance(epoch_selection, dict)
                else None
            ),
            "final_refit": final_fit["learning_rate_schedule"],
        },
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "batch_size": batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "evaluation_workers": evaluation_workers,
        "parallel_split_evaluation": parallel_split_evaluation,
        "train_prefetch_batches": train_prefetch_batches,
        "preload_feature_bank": preload_feature_bank,
        "torch_execution": execution_plan.as_manifest_payload(),
        "evaluation_execution": {
            "mode": (
                "cuda_serial_chunked"
                if execution_plan.device_type == "cuda"
                else "parallel_chunked_full_split"
            ),
            "inner_train_validation_concurrent": parallel_split_evaluation,
            "workers": (1 if execution_plan.device_type == "cuda" else evaluation_workers),
            "per_worker_torch_threads": 1,
            "batch_boundaries_changed": False,
            "reduction_order_changed": False,
            "uses_all_requested_rows": True,
            "training_sampling_enabled": bool(
                training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS
            ),
            "training_sampling_mode": training_sampling_mode,
            "training_sampling_unit": final_refit_sampling_summary["sampling_unit"],
            "training_batch_boundaries_changed": bool(
                training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS
            ),
            "training_order_changed": bool(
                training_sampling_mode != TRAINING_SAMPLING_ALL_EVENT_ROWS
            ),
            "final_metrics_reused_from_last_full_epoch_evaluation": True,
            "optimizer_name": optimizer_name,
            "lr_schedule_name": lr_schedule_name,
            "lr_schedule_parameters": lr_schedule_parameters,
            "augmentation_name": augmentation_plan.name,
            "augmentation_parameters": augmentation_parameters,
            "validation_augmented": False,
            "oos_augmented": False,
            "optimizer_zero_grad_set_to_none": True,
            "optimizer_weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "training_batch_prefetch": train_prefetch_batches,
            "feature_bank_preloaded": preload_feature_bank,
            "final_refit_mode": final_refit_mode,
            "class_weight_mode": class_weight_mode,
            "time_weight_mode": time_weight_mode,
            "training_weight_reduction": training_weight_reduction,
            "final_refit_target_optimizer_steps": int(final_fit["target_optimizer_steps"]),
            "final_refit_actual_optimizer_steps": int(final_fit["actual_optimizer_steps"]),
        },
        "training_history": final_fit["history"],
        "runtime_eligibility": {
            "eligible": False,
            "scope": RUNTIME_SCOPE_NOT_EXPORTED,
            "model_information_cutoff": model_information_cutoff,
            "reason": (
                "train.py only writes model/split artifacts; "
                "formal runtime scores require export_scores.py --scope forward_oos"
            ),
        },
        "no_lookahead_guarantee": (
            "features use D0 and earlier only; outer Selection/OOS dates come from "
            "core.walk_forward_policy; inner validation, when enabled, is confined "
            "to Selection and only selects epoch; the final model is refit on all "
            "eligible Selection groups with the configured refit-step policy; fixed threshold "
            "is committed before OOS; "
            "training sampling uses ticker/date and existing feature-group identity only; "
            "TS2Vec pretraining, when configured, uses only rolling-window endpoints inside Selection and never OOS; "
            "OOS predictions and metrics are not used by train.py"
        ),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(artifact_paths.manifest_path, manifest)
    print()
    print(
        _render_training_summary(
            split_report=split_report,
            use_inner_validation=use_inner_validation,
            final_train_loss=final_train_metrics["loss"],
            training_sampling_mode=training_sampling_mode,
            time_weight_mode=time_weight_mode,
            training_weight_reduction=training_weight_reduction,
            inner_train_sampling_summary=inner_train_sampling_summary,
            final_refit_sampling_summary=final_refit_sampling_summary,
        )
    )
    print("\n輸出工件")
    print(f"  - Model：{artifact_paths.model_path}")
    print(f"  - Split：{artifact_paths.split_path}")
    print(f"  - Manifest：{artifact_paths.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
