"""Train Tiny CNN breakout quality filter from dataset.npz."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MODEL_FILENAME,
    FEATURE_COLUMNS,
    CONTEXT_COLUMNS,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.model import build_model, require_torch
from tools.filters.breakout_quality.common import (
    chronological_group_split_indices,
    dataset_npz_path,
    events_csv_path,
    label_counts,
    model_dir,
    read_breakout_quality_csv,
    read_json,
    write_json,
    dataset_output_dir,
    event_group_summary,
    group_size_weights,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="訓練 breakout quality Tiny CNN")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
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


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    torch, _nn = require_torch()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(int(args.seed))
    data = np.load(dataset_npz_path(args.filter_id))
    X = data["features"].astype(np.float32)
    C = data["context"].astype(np.float32)
    y = data["labels"].astype(np.int64)
    events = read_breakout_quality_csv(events_csv_path(args.filter_id))
    if len(events) != len(y):
        raise ValueError(f"events.csv 與 dataset.npz labels 長度不一致: events={len(events)}, labels={len(y)}")

    train_idx, val_idx, split_report = chronological_group_split_indices(
        events,
        y,
        val_ratio=float(args.val_ratio),
        label_pass=LABEL_PASS,
        label_reject=LABEL_REJECT,
    )
    if int(split_report.get("overlap_group_count", 0)) != 0:
        raise ValueError(f"train/val group overlap 不應發生: {split_report}")
    if len(train_idx) < int(args.min_train_samples):
        raise ValueError(f"可訓練樣本不足: train={len(train_idx)}, min={args.min_train_samples}, labels={label_counts(y)}")
    if len(set(y[train_idx].tolist())) < 2:
        raise ValueError(f"train split 必須同時包含 PASS/REJECT，labels={label_counts(y[train_idx])}")

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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

    out_model_dir = model_dir(args.filter_id)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_count": int(X.shape[2]),
        "context_count": int(C.shape[1]),
    }, out_model_dir / DEFAULT_MODEL_FILENAME)
    summary_path = dataset_output_dir(args.filter_id) / "dataset_summary.json"
    dataset_summary = read_json(summary_path) if summary_path.exists() else {}
    manifest = {
        "filter_id": args.filter_id,
        "model_type": "tiny_cnn",
        "model_filename": DEFAULT_MODEL_FILENAME,
        "score_filename": "scores.csv",
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "dataset_summary": dataset_summary,
        "event_group_summary": event_group_summary(events, y),
        "label_counts": label_counts(y),
        "split_report": split_report,
        "sample_weighting": {
            "enabled": True,
            "method": "1 / labeled row count per ticker/date group",
            "group_key": "ticker/date",
        },
        "class_weights": {
            "reject": round(float(class_weights_np[LABEL_REJECT]), 6),
            "pass": round(float(class_weights_np[LABEL_PASS]), 6),
        },
        "train_label_counts": label_counts(y[train_idx]),
        "val_label_counts": label_counts(y[val_idx]) if len(val_idx) else {"pass": 0, "reject": 0, "ignore": 0, "total": 0},
        "train_metrics": _evaluate(torch, model, X, C, y, train_idx, sample_weights, class_weights),
        "val_metrics": _evaluate(torch, model, X, C, y, val_idx, sample_weights, class_weights),
        "best_epoch": int(best_epoch),
        "best_monitor": "val_loss" if len(val_idx) else "train_loss",
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
        "epochs": int(args.epochs),
        "epochs_trained": int(epoch),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "no_lookahead_contract": "features use D0 and earlier only; labels use future path only for supervised training",
    }
    write_json(out_model_dir / DEFAULT_MANIFEST_FILENAME, manifest)
    print(f"split_report={split_report}")
    print(f"best_epoch={best_epoch} best_monitor_value={best_monitor_value}")
    print(f"已輸出: {out_model_dir / DEFAULT_MODEL_FILENAME}")
    print(f"已輸出: {out_model_dir / DEFAULT_MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
