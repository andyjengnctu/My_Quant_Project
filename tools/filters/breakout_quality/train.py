"""Train Tiny CNN breakout quality filter from dataset.npz."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from config.breakout_quality_policy import BREAKOUT_QUALITY_INNER_VALIDATION_RATIO
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_FILTER_ID,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    LABEL_PASS,
    LABEL_REJECT,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_THRESHOLD_SOURCE,
    RUNTIME_SCOPE_NOT_EXPORTED,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
)
from filters.breakout_quality.model import build_model, require_torch
from filters.breakout_quality.splits import (
    build_outer_inner_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
)
from tools.filters.breakout_quality.common import (
    event_group_summary,
    group_size_weights,
    label_counts,
    model_dir,
    load_validated_dataset_bundle,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="訓練 breakout quality Tiny CNN")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inner-validation-ratio",
        "--val-ratio",
        dest="inner_validation_ratio",
        type=float,
        default=BREAKOUT_QUALITY_INNER_VALIDATION_RATIO,
        help="Selection period 內保留給 inner validation 的 event-date 比例；--val-ratio 為相容別名",
    )
    parser.add_argument("--min-train-samples", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="validation loss 連續未改善幾個 epoch 後停止；<=0 表示不啟用",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="validation loss 至少改善多少才更新 best model",
    )
    return parser.parse_args(argv)


def _class_weights(y_train: np.ndarray, sample_weights: np.ndarray):
    y_arr = y_train.astype(np.int64)
    w_arr = sample_weights.astype(np.float64)
    counts = np.array([
        float(w_arr[y_arr == LABEL_REJECT].sum()),
        float(w_arr[y_arr == LABEL_PASS].sum()),
    ], dtype=np.float32)
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


def _evaluate(torch, model, X, C, y, indices, sample_weights, class_weights):
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

    model.eval()
    idx = np.asarray(indices, dtype=np.int64)
    weights_np = sample_weights[idx].astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X[idx]), torch.from_numpy(C[idx]))
        target = torch.from_numpy(y[idx].astype(np.int64))
        weights_t = torch.from_numpy(weights_np)
        loss_items = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
        denom = torch.clamp(weights_t.sum(), min=1e-12)
        loss = ((loss_items * weights_t).sum() / denom).item()
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    correct = (pred == y[idx]).astype(np.float32)
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


def _max_iso_date(events: pd.DataFrame, indices: np.ndarray, column: str) -> str | None:
    if len(indices) == 0:
        return None
    values = pd.to_datetime(events.iloc[np.asarray(indices, dtype=np.int64)][column], errors="raise")
    return str(values.max().date())


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    if int(args.epochs) < 1 or int(args.batch_size) < 1 or float(args.lr) <= 0:
        raise ValueError("epochs、batch-size 必須 >=1，lr 必須 >0")

    dataset_summary, X, C, y, events = load_validated_dataset_bundle(args.filter_id)

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
    torch.manual_seed(int(args.seed))
    source_data_range = dataset_summary.get("source_data_date_range")
    source_data_end = (
        str(source_data_range.get("end") or "").strip()
        if isinstance(source_data_range, dict)
        else ""
    )
    if not source_data_end:
        source_data_end = str(pd.to_datetime(events["label_eval_end_date"], errors="raise").max().date())
    outer_oos_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=source_data_end,
    )
    split_assignments, train_idx, val_idx, oos_idx, split_report = build_outer_inner_split_assignments(
        events,
        y,
        outer_policy=outer_oos_policy,
        inner_validation_ratio=float(args.inner_validation_ratio),
    )
    if int(split_report.get("overlap_group_count", 0)) != 0:
        raise ValueError(f"inner train/validation/OOS group overlap 不應發生: {split_report}")
    if int(split_report.get("overlap_event_date_count", 0)) != 0:
        raise ValueError(f"inner train/validation/OOS event date overlap 不應發生: {split_report}")
    if len(train_idx) < int(args.min_train_samples):
        raise ValueError(f"可訓練樣本不足: train={len(train_idx)}, min={args.min_train_samples}, labels={label_counts(y)}")
    for split_name, split_idx in (("inner train", train_idx), ("inner validation", val_idx)):
        if len(set(y[split_idx].tolist())) < 2:
            raise ValueError(
                f"{split_name} 必須同時包含 PASS/REJECT，labels={label_counts(y[split_idx])}"
            )

    sample_weights = np.zeros((len(y),), dtype=np.float32)
    sample_weights[train_idx] = group_size_weights(events, train_idx)
    sample_weights[val_idx] = group_size_weights(events, val_idx)

    model = build_model(X.shape[2], C.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    class_weights_np = _class_weights(y[train_idx], sample_weights[train_idx])
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    batch_size = max(1, int(args.batch_size))
    best_state = None
    best_monitor_value = None
    best_epoch = 0
    best_train_metrics = None
    best_val_metrics = None
    epochs_without_improvement = 0
    training_history = []
    early_stopped = False
    min_delta = max(0.0, float(args.early_stopping_min_delta))
    patience = int(args.early_stopping_patience)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        rng = np.random.default_rng(int(args.seed) + epoch)
        shuffled = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            xb = torch.from_numpy(X[batch])
            cb = torch.from_numpy(C[batch])
            yb = torch.from_numpy(y[batch].astype(np.int64))
            wb = torch.from_numpy(sample_weights[batch].astype(np.float32))
            optimizer.zero_grad()
            import torch.nn.functional as F
            loss_items = F.cross_entropy(model(xb, cb), yb, weight=class_weights, reduction="none")
            loss = (loss_items * wb).sum() / torch.clamp(wb.sum(), min=1e-12)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_metrics = _evaluate(torch, model, X, C, y, train_idx, sample_weights, class_weights)
        val_metrics = _evaluate(torch, model, X, C, y, val_idx, sample_weights, class_weights)
        monitor_name = "val_loss" if val_metrics["loss"] is not None else "train_loss"
        monitored = val_metrics["loss"] if val_metrics["loss"] is not None else train_metrics["loss"]
        if monitored is None:
            raise RuntimeError("train/val loss 皆無法計算，無法選擇 best model")
        monitored_float = float(monitored)
        improved = best_monitor_value is None or monitored_float < (float(best_monitor_value) - min_delta)
        if improved:
            best_monitor_value = monitored_float
            best_epoch = int(epoch)
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_record = {
            "epoch": int(epoch),
            "loss": round(float(np.mean(losses)), 6),
            "monitor": monitor_name,
            "monitor_value": round(monitored_float, 6),
            "is_best": bool(improved),
            "train": train_metrics,
            "val": val_metrics,
        }
        training_history.append(epoch_record)
        print(
            f"epoch={epoch}/{args.epochs} loss={np.mean(losses):.6f} "
            f"monitor={monitor_name}:{monitored_float:.6f} best_epoch={best_epoch} "
            f"train={train_metrics} val={val_metrics}",
            flush=True,
        )
        if patience > 0 and epochs_without_improvement >= patience:
            early_stopped = True
            print(
                f"early_stopping triggered: patience={patience}, "
                f"best_epoch={best_epoch}, best_{monitor_name}={best_monitor_value:.6f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    artifact_paths = resolve_filter_artifact_paths(PROJECT_ROOT, args.filter_id)
    out_model_dir = model_dir(args.filter_id)
    stale_score_paths = (
        artifact_paths.score_path,
        resolve_filter_research_score_path(PROJECT_ROOT, args.filter_id),
        resolve_filter_research_manifest_path(PROJECT_ROOT, args.filter_id),
    )
    for stale_path in stale_score_paths:
        if stale_path.exists():
            stale_path.unlink()
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_count": int(X.shape[2]),
        "context_count": int(C.shape[1]),
    }, artifact_paths.model_path)
    split_assignments.to_csv(artifact_paths.split_path, index=False, encoding="utf-8-sig")

    used_idx = np.concatenate([train_idx, val_idx])
    model_information_cutoff = _max_iso_date(events, used_idx, "label_eval_end_date")
    if model_information_cutoff is None or model_information_cutoff >= str(outer_oos_policy["oos_start_date"]):
        raise ValueError(
            "breakout quality model_information_cutoff 必須早於既有 OOS 起點: "
            f"cutoff={model_information_cutoff}, oos_start={outer_oos_policy['oos_start_date']}"
        )
    split_record = {
        **build_file_manifest(artifact_paths.split_path),
        "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
        "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
        "columns": list(split_assignments.columns),
        "row_count": int(len(split_assignments)),
        "outer_split_counts": split_report["outer_split_counts"],
        "inner_split_counts": split_report["inner_split_counts"],
        "group_key": "ticker/date/high_len",
        "source_dataset_artifacts": dataset_summary.get("dataset_artifacts"),
    }
    manifest = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "filter_family": FILTER_FAMILY,
        "filter_id": args.filter_id,
        "model_type": "tiny_cnn",
        "model_filename": DEFAULT_MODEL_FILENAME,
        "score_filename": DEFAULT_SCORE_FILENAME,
        "model": build_file_manifest(artifact_paths.model_path),
        "split_filename": DEFAULT_SPLIT_FILENAME,
        "split_assignments": split_record,
        "outer_oos_policy": outer_oos_policy,
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "score_decision": {
            "score_column": SCORE_COLUMN,
            "comparison": SCORE_COMPARISON,
            "threshold_source": SCORE_THRESHOLD_SOURCE,
        },
        "policy": dataset_summary.get("policy"),
        "dataset_summary": dataset_summary,
        "dataset_artifacts": dataset_summary.get("dataset_artifacts"),
        "event_group_summary": event_group_summary(events, y),
        "label_counts": label_counts(y),
        "split_report": split_report,
        "model_information_cutoff": model_information_cutoff,
        "runtime_eligibility": {
            "eligible": False,
            "scope": RUNTIME_SCOPE_NOT_EXPORTED,
            "reason": "scores.csv 尚未以 forward_oos 或 rolling_oos 範圍匯出",
        },
        "sample_weighting": {
            "enabled": True,
            "method": "1 / labeled row count per ticker/date group",
            "group_key": "ticker/date",
        },
        "class_weights": {
            "reject": round(float(class_weights_np[LABEL_REJECT]), 6),
            "pass": round(float(class_weights_np[LABEL_PASS]), 6),
        },
        "inner_train_label_counts": label_counts(y[train_idx]),
        "inner_validation_label_counts": label_counts(y[val_idx]),
        "oos_evaluable_label_counts": label_counts(y[oos_idx]),
        "inner_train_metrics": _evaluate(torch, model, X, C, y, train_idx, sample_weights, class_weights),
        "inner_validation_metrics": _evaluate(torch, model, X, C, y, val_idx, sample_weights, class_weights),
        "oos_predictions_used_during_training_or_early_stopping": False,
        "oos_metrics_emitted_by_train": False,
        "best_epoch": int(best_epoch),
        "best_monitor": "inner_validation_loss",
        "best_monitor_value": round(float(best_monitor_value), 6) if best_monitor_value is not None else None,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "training_history": training_history,
        "early_stopping": {
            "enabled": patience > 0,
            "patience": int(patience),
            "min_delta": float(min_delta),
            "triggered": bool(early_stopped),
        },
        "seed": int(args.seed),
        "inner_validation_ratio": float(args.inner_validation_ratio),
        "epochs": int(args.epochs),
        "epochs_trained": int(epoch),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "no_lookahead_contract": (
            "features use D0 and earlier only; outer selection/OOS dates come from core.walk_forward_policy; "
            "inner train rows require label_eval_end_date before inner validation start; "
            "inner validation rows require label_eval_end_date before OOS start; "
            "OOS predictions are not used by training or early stopping"
        ),
    }
    write_json(out_model_dir / DEFAULT_MANIFEST_FILENAME, manifest)
    print(f"split_report={split_report}")
    print(f"best_epoch={best_epoch} best_monitor_value={best_monitor_value}")
    print(f"已輸出: {artifact_paths.model_path}")
    print(f"已輸出: {artifact_paths.split_path}")
    print(f"已輸出: {artifact_paths.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
