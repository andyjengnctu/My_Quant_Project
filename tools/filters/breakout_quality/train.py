"""Train the configured breakout-quality temporal CNN with optional inner validation."""

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
    BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES,
    BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_OPTIMIZER_NAME,
    BREAKOUT_QUALITY_SUPPORTED_OPTIMIZERS,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
)
from core.display_common import render_elapsed
from filters.breakout_quality.artifacts import build_file_manifest
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
from filters.breakout_quality.inference import (
    materialize_indexed_feature_inputs,
    strict_parallel_batched_logits,
)
from filters.breakout_quality.model import (
    build_model,
    count_trainable_parameters,
    get_model_spec,
    require_torch,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
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
TIME_WEIGHT_MODE_NONE = "none"
TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT = "year_balanced_sqrt"
TIME_WEIGHT_MODES = (TIME_WEIGHT_MODE_NONE, TIME_WEIGHT_MODE_YEAR_BALANCED_SQRT)
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAMW = "adamw"
OPTIMIZER_NAMES = tuple(BREAKOUT_QUALITY_SUPPORTED_OPTIMIZERS)


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
        "--optimizer-name",
        choices=OPTIMIZER_NAMES,
        default=BREAKOUT_QUALITY_OPTIMIZER_NAME,
        help="訓練 optimizer；adam 使用 coupled L2，adamw 使用 decoupled weight decay",
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
    parser.add_argument("--min-train-samples", type=int, default=BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES)
    parser.add_argument(
        "--min-validation-samples",
        type=int,
        default=BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    )
    return parser.parse_args(argv)


def validate_training_args(args) -> None:
    max_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    evaluation_batch_size = int(args.evaluation_batch_size)
    evaluation_workers = int(args.evaluation_workers)
    train_prefetch_batches = int(args.train_prefetch_batches)
    optimizer_name = str(args.optimizer_name).strip().lower()
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
        raise ValueError(f"optimizer-name 不合法: {optimizer_name}")
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
) -> str:
    lines = ["訓練摘要"]
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
    else:
        raise ValueError(f"不支援的 time weight mode: {mode!r}")

    return weighted.astype(np.float32), {
        "mode": normalized_mode,
        "group_count": int(round(float(base_weights.sum()))),
        "weight_sum": round(float(weighted.sum()), 6),
        "year_group_counts": {str(year): count for year, count in year_group_counts.items()},
        "year_weight_multipliers": {
            str(year): round(value, 8) for year, value in multipliers.items()
        },
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


def _materialize_training_batch(
    X,
    C: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        X[batch],
        C[batch],
        y[batch],
        sample_weights[batch],
    )


def _iter_training_batches(
    X,
    C: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    shuffled: np.ndarray,
    *,
    batch_size: int,
    prefetch_batches: int,
):
    batches = [
        shuffled[start:start + int(batch_size)]
        for start in range(0, len(shuffled), int(batch_size))
    ]
    if int(prefetch_batches) <= 0 or len(batches) <= 1:
        for batch in batches:
            yield _materialize_training_batch(
                X, C, y, sample_weights, batch
            )
        return

    queue_depth = min(int(prefetch_batches), len(batches))
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="breakout-quality-prefetch",
    ) as executor:
        pending = deque(
            executor.submit(
                _materialize_training_batch,
                X,
                C,
                y,
                sample_weights,
                batches[index],
            )
            for index in range(queue_depth)
        )
        next_index = queue_depth
        while pending:
            future = pending.popleft()
            batch_arrays = future.result()
            if next_index < len(batches):
                pending.append(
                    executor.submit(
                        _materialize_training_batch,
                        X,
                        C,
                        y,
                        sample_weights,
                        batches[next_index],
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
    )

    with torch.no_grad():
        for start in range(0, int(idx.size), batch_size):
            stop = min(start + batch_size, int(idx.size))
            logits = torch.from_numpy(logits_np[start:stop])
            target = torch.from_numpy(targets_np[start:stop])
            loss_items = F.cross_entropy(
                logits,
                target,
                weight=class_weights,
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
):
    common_kwargs = {
        "evaluation_batch_size": int(evaluation_batch_size),
        "evaluation_workers": int(evaluation_workers),
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
):
    torch.manual_seed(int(seed))
    model = build_model(feature_count, context_count)
    optimizer = _build_optimizer(
        torch,
        optimizer_name=optimizer_name,
        parameters=model.parameters(),
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
        else torch.tensor(class_weights_np, dtype=torch.float32)
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
    batch_size: int,
    shuffle_seed: int,
    prefetch_batches: int,
    gradient_clip_norm: float,
    max_batches: int | None = None,
) -> tuple[float, int]:
    import torch.nn.functional as F

    model.train()
    rng = np.random.default_rng(int(shuffle_seed))
    shuffled = rng.permutation(train_idx)
    batch_losses = []
    completed_batches = 0
    for xb_np, cb_np, yb_np, wb_np in _iter_training_batches(
        X,
        C,
        y,
        sample_weights,
        shuffled,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
    ):
        if max_batches is not None and completed_batches >= int(max_batches):
            break
        xb = torch.from_numpy(xb_np)
        cb = torch.from_numpy(cb_np)
        yb = torch.from_numpy(yb_np)
        wb = torch.from_numpy(wb_np)
        optimizer.zero_grad(set_to_none=True)
        loss_items = F.cross_entropy(
            model(xb, cb),
            yb,
            weight=class_weights,
            reduction="none",
        )
        loss = (loss_items * wb).sum() / torch.clamp(wb.sum(), min=1e-12)
        loss.backward()
        if float(gradient_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
            )
        optimizer.step()
        batch_losses.append(float(loss.item()))
        completed_batches += 1
    mean_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
    return mean_loss, int(completed_batches)

def _select_epoch_with_inner_validation(
    torch,
    *,
    X: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    events: pd.DataFrame,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    max_epochs: int,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    class_weight_mode: str,
    time_weight_mode: str,
    seed: int,
    patience: int,
    min_delta: float,
    evaluation_batch_size: int,
    evaluation_workers: int,
    parallel_split_evaluation: bool,
    train_prefetch_batches: int,
):
    training_sample_weights, sample_weight_summaries = _build_sample_weights(
        events,
        len(y),
        {"inner_train": train_idx},
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
        train_idx=train_idx,
        sample_weights=training_sample_weights,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weight_mode=class_weight_mode,
        seed=seed,
    )
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_metrics = None
    epochs_without_improvement = 0
    history = []
    print("\nEpoch 選擇（依 Validation Loss）")
    color_time = bool(sys.stdout.isatty())
    for epoch in range(1, int(max_epochs) + 1):
        epoch_started = time.perf_counter()
        batch_loss, optimizer_steps = _train_one_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            X=X,
            C=C,
            y=y,
            train_idx=train_idx,
            sample_weights=training_sample_weights,
            class_weights=class_weights,
            batch_size=batch_size,
            shuffle_seed=int(seed) + epoch,
            prefetch_batches=train_prefetch_batches,
            gradient_clip_norm=gradient_clip_norm,
        )
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
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
            parallel=parallel_split_evaluation,
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
                "cumulative_optimizer_steps": int(epoch * optimizer_steps),
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
        "batches_per_epoch": int(math.ceil(len(train_idx) / int(batch_size))),
        "best_optimizer_steps": int(
            int(best_epoch) * math.ceil(len(train_idx) / int(batch_size))
        ),
        "sample_weight_summaries": sample_weight_summaries,
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
    y: np.ndarray,
    events: pd.DataFrame,
    train_idx: np.ndarray,
    epochs: int,
    target_optimizer_steps: int | None,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    class_weight_mode: str,
    time_weight_mode: str,
    seed: int,
    phase_name: str,
    evaluation_batch_size: int,
    evaluation_workers: int,
    train_prefetch_batches: int,
):
    training_sample_weights, sample_weight_summaries = _build_sample_weights(
        events,
        len(y),
        {"final_refit": train_idx},
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
        train_idx=train_idx,
        sample_weights=training_sample_weights,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weight_mode=class_weight_mode,
        seed=seed,
    )
    batches_per_epoch = int(math.ceil(len(train_idx) / int(batch_size)))
    requested_steps = (
        int(target_optimizer_steps)
        if target_optimizer_steps is not None
        else int(epochs) * batches_per_epoch
    )
    if requested_steps < batches_per_epoch:
        raise ValueError(
            "final refit optimizer steps 不足以讓全部 eligible Selection rows 至少使用一次: "
            f"target={requested_steps}, minimum={batches_per_epoch}"
        )
    planned_cycles = int(math.ceil(requested_steps / batches_per_epoch))
    equivalent_epochs = float(requested_steps) / float(batches_per_epoch)

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
        batch_loss, optimizer_steps = _train_one_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            X=X,
            C=C,
            y=y,
            train_idx=train_idx,
            sample_weights=training_sample_weights,
            class_weights=class_weights,
            batch_size=batch_size,
            shuffle_seed=int(seed) + cycle,
            prefetch_batches=train_prefetch_batches,
            gradient_clip_norm=gradient_clip_norm,
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
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
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
        "batches_per_epoch": int(batches_per_epoch),
        "target_optimizer_steps": int(requested_steps),
        "actual_optimizer_steps": int(completed_optimizer_steps),
        "equivalent_epochs": round(float(equivalent_epochs), 8),
        "completed_epoch_cycles": int(planned_cycles),
        "last_epoch_fraction": round(
            float(history[-1]["epoch_fraction"]),
            8,
        ),
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
    optimizer_name = str(args.optimizer_name).strip().lower()
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
    validate_training_args(args)

    dataset_summary, X, C, y, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=True,
    )
    X, C, y = _preload_training_arrays(
        X,
        C,
        y,
        enabled=preload_feature_bank,
    )
    gc.collect()

    torch, _nn = require_torch()
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

    epoch_selection = None
    selected_optimizer_steps = None
    minimum_full_pass_applied = False
    if use_inner_validation:
        epoch_selection = _select_epoch_with_inner_validation(
            torch,
            X=X,
            C=C,
            y=y,
            events=events,
            train_idx=inner_train_idx,
            validation_idx=inner_validation_idx,
            max_epochs=max_epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            class_weight_mode=class_weight_mode,
            time_weight_mode=time_weight_mode,
            seed=int(args.seed),
            patience=patience,
            min_delta=min_delta,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
            parallel_split_evaluation=parallel_split_evaluation,
            train_prefetch_batches=train_prefetch_batches,
        )
        selected_epoch = int(epoch_selection["best_epoch"])
        selected_optimizer_steps = int(epoch_selection["best_optimizer_steps"])
        training_mode = TRAINING_MODE_INNER_VALIDATION_FULL_REFIT
        epoch_selection_source = "inner_validation_loss"
        phase_name = "full_refit"
        final_batches_per_epoch = int(
            math.ceil(len(final_refit_idx) / int(batch_size))
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
        y=y,
        events=events,
        train_idx=final_refit_idx,
        epochs=selected_epoch,
        target_optimizer_steps=final_target_optimizer_steps,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        class_weight_mode=class_weight_mode,
        time_weight_mode=time_weight_mode,
        seed=int(args.seed),
        phase_name=phase_name,
        evaluation_batch_size=evaluation_batch_size,
        evaluation_workers=evaluation_workers,
        train_prefetch_batches=train_prefetch_batches,
    )
    final_refit_plan = {
        "mode": final_refit_mode,
        "inner_train_row_count": int(len(inner_train_idx)),
        "final_refit_row_count": int(len(final_refit_idx)),
        "batch_size": int(batch_size),
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
            int(final_fit["actual_optimizer_steps"])
            >= int(final_fit["batches_per_epoch"])
        ),
    }
    model = final_fit["model"]
    final_train_metrics = final_fit["final_metrics"]

    out_dir = model_dir(args.filter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = resolve_filter_artifact_paths(PROJECT_ROOT, args.filter_id)
    for stale_path in (
        artifact_paths.score_path,
        resolve_filter_research_score_path(PROJECT_ROOT, args.filter_id),
        resolve_filter_research_manifest_path(PROJECT_ROOT, args.filter_id),
    ):
        stale_path.unlink(missing_ok=True)

    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    trainable_parameter_count = count_trainable_parameters(model)
    torch.save(
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "model_state_dict": model.state_dict(),
            "feature_count": int(X.shape[2]),
            "context_count": int(C.shape[1]),
            "sequence_length": int(X.shape[1]),
            "model_spec": model_spec.as_manifest_payload(),
            "trainable_parameter_count": int(trainable_parameter_count),
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
    }
    manifest = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "filter_family": FILTER_FAMILY,
        "filter_id": args.filter_id,
        "model_architecture": model_spec.architecture,
        "model_spec": model_spec.as_manifest_payload(),
        "trainable_parameter_count": int(trainable_parameter_count),
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
        "training_uses_all_eligible_selection_rows": True,
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
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "batch_size": batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "evaluation_workers": evaluation_workers,
        "parallel_split_evaluation": parallel_split_evaluation,
        "train_prefetch_batches": train_prefetch_batches,
        "preload_feature_bank": preload_feature_bank,
        "evaluation_execution": {
            "mode": "parallel_chunked_full_split",
            "inner_train_validation_concurrent": parallel_split_evaluation,
            "workers": evaluation_workers,
            "per_worker_torch_threads": 1,
            "batch_boundaries_changed": False,
            "reduction_order_changed": False,
            "uses_all_requested_rows": True,
            "training_sampling_enabled": False,
            "training_order_changed": False,
            "final_metrics_reused_from_last_full_epoch_evaluation": True,
            "optimizer_name": optimizer_name,
            "optimizer_zero_grad_set_to_none": True,
            "optimizer_weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "training_batch_prefetch": train_prefetch_batches,
            "feature_bank_preloaded": preload_feature_bank,
            "final_refit_mode": final_refit_mode,
            "class_weight_mode": class_weight_mode,
            "time_weight_mode": time_weight_mode,
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
            "eligible Selection rows with the configured refit-step policy; fixed threshold "
            "is committed before OOS; "
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
        )
    )
    print("\n輸出工件")
    print(f"  - Model：{artifact_paths.model_path}")
    print(f"  - Split：{artifact_paths.split_path}")
    print(f"  - Manifest：{artifact_paths.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
