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
    group_size_weights,
    label_counts,
    load_validated_dataset_bundle,
    model_dir,
    write_json,
)


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
    parser.add_argument("--lr", type=float, default=BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
        help="Adam L2 weight decay；0 表示關閉",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
        help="每次 optimizer update 前的全域 gradient norm 上限；0 表示關閉",
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
    learning_rate = float(args.lr)
    weight_decay = float(args.weight_decay)
    gradient_clip_norm = float(args.gradient_clip_norm)
    fixed_threshold = float(args.fixed_threshold)
    validation_months = int(args.inner_validation_months)
    patience = int(args.early_stopping_patience)
    min_delta = float(args.early_stopping_min_delta)
    min_train_samples = int(args.min_train_samples)
    min_validation_samples = int(args.min_validation_samples)
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
    if validation_months < 1:
        raise ValueError("inner-validation-months 必須 >=1")
    if patience < 0 or min_delta < 0:
        raise ValueError("early-stopping-patience 與 min-delta 必須 >=0")
    if min_train_samples < 1 or min_validation_samples < 1:
        raise ValueError("min-train-samples 與 min-validation-samples 必須 >=1")
    if not np.isfinite(fixed_threshold) or not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("fixed-threshold 必須介於 0 與 1")


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


def _class_weights(y_train: np.ndarray, sample_weights: np.ndarray):
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


def _build_sample_weights(events: pd.DataFrame, row_count: int, indices: np.ndarray):
    weights = np.zeros((row_count,), dtype=np.float32)
    weights[indices] = group_size_weights(events, indices)
    return weights


def _new_training_state(
    torch,
    *,
    feature_count: int,
    context_count: int,
    y: np.ndarray,
    train_idx: np.ndarray,
    sample_weights: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    seed: int,
):
    torch.manual_seed(int(seed))
    model = build_model(feature_count, context_count)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    class_weights_np = _class_weights(y[train_idx], sample_weights[train_idx])
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
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
) -> float:
    import torch.nn.functional as F

    model.train()
    rng = np.random.default_rng(int(shuffle_seed))
    shuffled = rng.permutation(train_idx)
    batch_losses = []
    for xb_np, cb_np, yb_np, wb_np in _iter_training_batches(
        X,
        C,
        y,
        sample_weights,
        shuffled,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
    ):
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
    return float(np.mean(batch_losses)) if batch_losses else float("nan")


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
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
    patience: int,
    min_delta: float,
    evaluation_batch_size: int,
    evaluation_workers: int,
    parallel_split_evaluation: bool,
    train_prefetch_batches: int,
):
    sample_weights = _build_sample_weights(
        events,
        len(y),
        np.concatenate([train_idx, validation_idx]),
    )
    model, optimizer, class_weights_np, class_weights = _new_training_state(
        torch,
        feature_count=X.shape[2],
        context_count=C.shape[1],
        y=y,
        train_idx=train_idx,
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
        batch_loss = _train_one_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            X=X,
            C=C,
            y=y,
            train_idx=train_idx,
            sample_weights=sample_weights,
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
            sample_weights,
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
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
    phase_name: str,
    evaluation_batch_size: int,
    evaluation_workers: int,
    train_prefetch_batches: int,
):
    sample_weights = _build_sample_weights(events, len(y), train_idx)
    model, optimizer, class_weights_np, class_weights = _new_training_state(
        torch,
        feature_count=X.shape[2],
        context_count=C.shape[1],
        y=y,
        train_idx=train_idx,
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    history = []
    phase_title = (
        "完整 Selection 重訓"
        if phase_name == "full_refit"
        else "完整 Selection 訓練"
    )
    print(f"\n{phase_title}（{int(epochs)} Epoch）")
    color_time = bool(sys.stdout.isatty())
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        batch_loss = _train_one_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            X=X,
            C=C,
            y=y,
            train_idx=train_idx,
            sample_weights=sample_weights,
            class_weights=class_weights,
            batch_size=batch_size,
            shuffle_seed=int(seed) + epoch,
            prefetch_batches=train_prefetch_batches,
            gradient_clip_norm=gradient_clip_norm,
        )
        metrics = _evaluate(
            torch,
            model,
            X,
            C,
            y,
            train_idx,
            sample_weights,
            class_weights,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
        )
        epoch_elapsed = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": int(epoch),
                "batch_loss": round(float(batch_loss), 6),
                "selection_metrics": metrics,
                "elapsed_sec": round(float(epoch_elapsed), 3),
            }
        )
        print(
            _render_full_selection_progress(
                epoch=epoch,
                epochs=epochs,
                train_loss=metrics["loss"],
                elapsed_sec=epoch_elapsed,
                color=color_time,
            )
        )
    if not history:
        raise ValueError("完整 Selection 訓練至少需要 1 個 epoch")
    final_metrics = dict(history[-1]["selection_metrics"])
    return {
        "model": model,
        "history": history,
        "final_metrics": final_metrics,
        "class_weights_reject_pass": [
            round(float(value), 8) for value in class_weights_np.tolist()
        ],
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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            seed=int(args.seed),
            patience=patience,
            min_delta=min_delta,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_workers=evaluation_workers,
            parallel_split_evaluation=parallel_split_evaluation,
            train_prefetch_batches=train_prefetch_batches,
        )
        selected_epoch = int(epoch_selection["best_epoch"])
        training_mode = TRAINING_MODE_INNER_VALIDATION_FULL_REFIT
        epoch_selection_source = "inner_validation_loss"
        phase_name = "full_refit"
    else:
        selected_epoch = max_epochs
        training_mode = TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION
        epoch_selection_source = "fixed_cli_epochs"
        phase_name = "fixed"

    final_fit = _fit_full_selection(
        torch,
        X=X,
        C=C,
        y=y,
        events=events,
        train_idx=final_refit_idx,
        epochs=selected_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        seed=int(args.seed),
        phase_name=phase_name,
        evaluation_batch_size=evaluation_batch_size,
        evaluation_workers=evaluation_workers,
        train_prefetch_batches=train_prefetch_batches,
    )
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
        "completed_epochs": selected_epoch,
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
        "class_weights_reject_pass": final_fit["class_weights_reject_pass"],
        "seed": int(args.seed),
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
            "optimizer_zero_grad_set_to_none": True,
            "optimizer_weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "training_batch_prefetch": train_prefetch_batches,
            "feature_bank_preloaded": preload_feature_bank,
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
            "eligible Selection rows; fixed threshold is committed before OOS; "
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
