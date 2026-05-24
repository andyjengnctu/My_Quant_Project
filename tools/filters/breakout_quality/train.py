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
    dataset_npz_path,
    events_csv_path,
    label_counts,
    model_dir,
    read_json,
    scores_csv_path,
    write_json,
    dataset_output_dir,
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
    return parser.parse_args(argv)


def _chronological_split_indices(labels: np.ndarray, val_ratio: float):
    valid = np.flatnonzero((labels == LABEL_PASS) | (labels == LABEL_REJECT))
    if valid.size == 0:
        return valid, valid
    split = int(round(valid.size * (1.0 - float(val_ratio))))
    split = min(max(split, 1), max(1, valid.size - 1)) if valid.size > 1 else valid.size
    return valid[:split], valid[split:]


def _class_weights(y_train: np.ndarray):
    counts = np.bincount(y_train.astype(np.int64), minlength=2).astype(np.float32)
    counts[counts <= 0] = 1.0
    total = counts.sum()
    return total / (2.0 * counts)


def _evaluate(torch, model, X, C, y, indices):
    if len(indices) == 0:
        return {"loss": None, "accuracy": None, "pass_rate": None}
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X[indices]), torch.from_numpy(C[indices]))
        target = torch.from_numpy(y[indices].astype(np.int64))
        loss = F.cross_entropy(logits, target).item()
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    return {
        "loss": round(float(loss), 6),
        "accuracy": round(float((pred == y[indices]).mean()), 6),
        "pass_rate": round(float((pred == LABEL_PASS).mean()), 6),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    torch, nn = require_torch()
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
    train_idx, val_idx = _chronological_split_indices(y, float(args.val_ratio))
    if len(train_idx) < int(args.min_train_samples):
        raise ValueError(f"可訓練樣本不足: train={len(train_idx)}, min={args.min_train_samples}, labels={label_counts(y)}")
    if len(set(y[train_idx].tolist())) < 2:
        raise ValueError(f"train split 必須同時包含 PASS/REJECT，labels={label_counts(y[train_idx])}")

    model = build_model(X.shape[2], C.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    weights = torch.tensor(_class_weights(y[train_idx]), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    batch_size = max(1, int(args.batch_size))
    best_state = None
    best_val_loss = None
    train_idx = np.asarray(train_idx, dtype=np.int64)
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
            optimizer.zero_grad()
            loss = criterion(model(xb, cb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_metrics = _evaluate(torch, model, X, C, y, train_idx)
        val_metrics = _evaluate(torch, model, X, C, y, val_idx)
        monitored = val_metrics["loss"] if val_metrics["loss"] is not None else train_metrics["loss"]
        if best_val_loss is None or float(monitored) < float(best_val_loss):
            best_val_loss = float(monitored)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch}/{args.epochs} loss={np.mean(losses):.6f} train={train_metrics} val={val_metrics}", flush=True)

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
        "label_counts": label_counts(y),
        "train_label_counts": label_counts(y[train_idx]),
        "val_label_counts": label_counts(y[val_idx]) if len(val_idx) else {"pass": 0, "reject": 0, "ignore": 0, "total": 0},
        "train_metrics": _evaluate(torch, model, X, C, y, train_idx),
        "val_metrics": _evaluate(torch, model, X, C, y, val_idx),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "no_lookahead_contract": "features use D0 and earlier only; labels use future path only for supervised training",
    }
    write_json(out_model_dir / DEFAULT_MANIFEST_FILENAME, manifest)
    print(f"已輸出: {out_model_dir / DEFAULT_MODEL_FILENAME}")
    print(f"已輸出: {out_model_dir / DEFAULT_MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
