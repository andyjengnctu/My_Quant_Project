"""Train the breakout-quality Tiny CNN with fixed epochs on the full Selection period."""

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

from config.breakout_quality_policy import BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
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
    RUNTIME_SCOPE_NOT_EXPORTED,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_THRESHOLD_SOURCE,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
)
from filters.breakout_quality.model import build_model, require_torch
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

TRAINING_MODE = "fixed_epoch_full_selection"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "使用完整 Selection period 與固定 epochs 訓練 breakout quality Tiny CNN；"
            "不使用 inner validation 或 early stopping"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="事前固定的訓練 epoch 數；不會依 OOS 結果調整",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
        help="在查看 OOS 前固定的評估 threshold",
    )
    parser.add_argument("--min-train-samples", type=int, default=20)
    return parser.parse_args(argv)


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
        loss_items = F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            reduction="none",
        )
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
    values = pd.to_datetime(
        events.iloc[np.asarray(indices, dtype=np.int64)][column],
        errors="raise",
    )
    return str(values.max().date())


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.lr)
    fixed_threshold = float(args.fixed_threshold)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs、batch-size 必須 >=1，lr 必須 >0")
    if not np.isfinite(fixed_threshold) or not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("fixed-threshold 必須介於 0 與 1")

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
    split_assignments, train_idx, _oos_idx, split_report = (
        build_selection_oos_split_assignments(
            events,
            y,
            outer_policy=outer_oos_policy,
        )
    )
    if int(split_report.get("overlap_group_count", 0)) != 0:
        raise ValueError(f"Selection/OOS group overlap 不應發生: {split_report}")
    if int(split_report.get("overlap_event_date_count", 0)) != 0:
        raise ValueError(f"Selection/OOS event date overlap 不應發生: {split_report}")
    if len(train_idx) < int(args.min_train_samples):
        raise ValueError(
            "可訓練樣本不足: "
            f"selection_train={len(train_idx)}, min={args.min_train_samples}, "
            f"labels={label_counts(y)}"
        )
    if len(set(y[train_idx].tolist())) < 2:
        raise ValueError(
            "Selection train 必須同時包含 PASS/REJECT，"
            f"labels={label_counts(y[train_idx])}"
        )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    sample_weights = np.zeros((len(y),), dtype=np.float32)
    sample_weights[train_idx] = group_size_weights(events, train_idx)

    model = build_model(X.shape[2], C.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights_np = _class_weights(y[train_idx], sample_weights[train_idx])
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    training_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(int(args.seed) + epoch)
        shuffled = rng.permutation(train_idx)
        batch_losses = []
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            xb = torch.from_numpy(X[batch])
            cb = torch.from_numpy(C[batch])
            yb = torch.from_numpy(y[batch].astype(np.int64))
            wb = torch.from_numpy(sample_weights[batch].astype(np.float32))
            optimizer.zero_grad()
            import torch.nn.functional as F

            loss_items = F.cross_entropy(
                model(xb, cb),
                yb,
                weight=class_weights,
                reduction="none",
            )
            loss = (loss_items * wb).sum() / torch.clamp(
                wb.sum(),
                min=1e-12,
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_metrics = _evaluate(
            torch,
            model,
            X,
            C,
            y,
            train_idx,
            sample_weights,
            class_weights,
        )
        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        training_history.append(
            {
                "epoch": int(epoch),
                "batch_loss": round(epoch_loss, 6),
                "selection_train_metrics": train_metrics,
            }
        )
        print(
            f"epoch={epoch}/{epochs} "
            f"loss={epoch_loss:.6f} "
            f"train={train_metrics}"
        )

    final_train_metrics = _evaluate(
        torch,
        model,
        X,
        C,
        y,
        train_idx,
        sample_weights,
        class_weights,
    )

    out_dir = model_dir(args.filter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = resolve_filter_artifact_paths(PROJECT_ROOT, args.filter_id)
    for stale_path in (
        artifact_paths.score_path,
        resolve_filter_research_score_path(PROJECT_ROOT, args.filter_id),
        resolve_filter_research_manifest_path(PROJECT_ROOT, args.filter_id),
    ):
        stale_path.unlink(missing_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_count": int(X.shape[2]),
            "context_count": int(C.shape[1]),
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
        train_idx,
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
        "training_mode": TRAINING_MODE,
        "fixed_epochs": epochs,
        "completed_epochs": epochs,
        "early_stopping_enabled": False,
        "inner_validation_used": False,
        "training_uses_all_eligible_selection_rows": True,
        "oos_predictions_used_during_training": False,
        "oos_metrics_emitted_by_train": False,
        "model_information_cutoff": model_information_cutoff,
        "policy": dataset_summary.get("policy"),
        "source_dataset": dataset_summary,
        "event_group_summary": event_group_summary(events, y),
        "split_report": split_report,
        "selection_train_label_counts": label_counts(y[train_idx]),
        "selection_train_metrics": final_train_metrics,
        "class_weights_reject_pass": [
            round(float(value), 8)
            for value in class_weights_np.tolist()
        ],
        "seed": int(args.seed),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "training_history": training_history,
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
            "core.walk_forward_policy; all eligible Selection labels must end before "
            "OOS start; fixed epochs and threshold policy are committed before OOS; "
            "OOS predictions and metrics are not used by train.py"
        ),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(artifact_paths.manifest_path, manifest)
    print(f"split_report={split_report}")
    print(
        f"training_mode={TRAINING_MODE} "
        f"fixed_epochs={epochs} "
        f"fixed_threshold={fixed_threshold:.6f} "
        f"final_train_loss={final_train_metrics['loss']}"
    )
    print(f"已輸出: {artifact_paths.model_path}")
    print(f"已輸出: {artifact_paths.split_path}")
    print(f"已輸出: {artifact_paths.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
