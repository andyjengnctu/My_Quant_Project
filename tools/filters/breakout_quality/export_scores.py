"""Export breakout quality research or formal forward-OOS score tables."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import warnings

import numpy as np
import pandas as pd

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
)

from filters.breakout_quality.artifacts import (
    build_file_manifest,
    load_model_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    DEFAULT_SCORE_FILENAME,
    LABEL_PASS,
    SCORE_COLUMN,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    RUNTIME_SCOPE_FORWARD_OOS,
    RUNTIME_SCOPE_RESEARCH,
)
from filters.breakout_quality.inference import (
    materialize_indexed_feature_inputs,
    strict_parallel_batched_logits,
)
from filters.breakout_quality.model import build_model, require_torch
from filters.breakout_quality.paths import (
    ensure_filter_output_dir,
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
)
from tools.filters.breakout_quality.common import load_validated_dataset_bundle, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="匯出 breakout quality 分數表")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--scope",
        choices=(RUNTIME_SCOPE_RESEARCH, RUNTIME_SCOPE_FORWARD_OOS),
        default=RUNTIME_SCOPE_RESEARCH,
        help=(
            "research 寫入 outputs/，不變更正式工件；"
            "forward_oos 才會把模型資訊截止日之後的事件寫入 canonical scores.csv"
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
        help="分數匯出的固定推論 batch size；不改列序或 score 定義",
    )
    parser.add_argument(
        "--inference-workers",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_WORKERS,
        help="分數匯出的平行 read-only inference workers",
    )
    parser.add_argument(
        "--preload-feature-bank",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
        help="匯出前是否將 feature bank 與 context 載入 RAM",
    )
    return parser.parse_args(argv)


def _date_range(values: pd.Series) -> dict[str, str | None]:
    dates = pd.to_datetime(values, errors="raise")
    if dates.empty:
        return {"start": None, "end": None}
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}


def _build_score_record(
    score_path: Path,
    *,
    out_cols: list[str],
    row_count: int,
    high_len_values: list[int],
    event_range: dict[str, str | None],
    dataset_summary: dict,
) -> dict:
    score_record = build_file_manifest(score_path)
    score_record.update(
        {
            "schema_version": SCORE_TABLE_SCHEMA_VERSION,
            "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
            "columns": out_cols,
            "row_count": int(row_count),
            "high_len_values": [int(value) for value in high_len_values],
            "event_date_range": event_range,
            "source_dataset_artifacts": dataset_summary.get("dataset_artifacts"),
        }
    )
    return score_record


def main(argv=None) -> int:
    args = parse_args(argv)
    inference_batch_size = int(args.inference_batch_size)
    inference_workers = int(args.inference_workers)
    preload_feature_bank = bool(args.preload_feature_bank)
    if inference_batch_size < 1 or inference_workers < 1:
        raise ValueError("inference-batch-size 與 inference-workers 必須 >=1")
    torch, _nn = require_torch()
    if int(torch.get_num_threads()) != 1:
        torch.set_num_threads(1)
    if int(torch.get_num_interop_threads()) != 1:
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
    model_contract = load_model_artifact_contract(str(PROJECT_ROOT), str(args.filter_id))
    artifact_paths = model_contract.paths
    manifest = model_contract.manifest
    split_assignments = load_split_assignment_frame(str(PROJECT_ROOT), str(args.filter_id))
    model_policy = manifest.get("policy")
    if not isinstance(model_policy, dict):
        raise ValueError("model manifest 缺少 policy object")
    dataset_summary, features, context, _labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=model_policy,
    )
    features, context = materialize_indexed_feature_inputs(
        features,
        context,
        enabled=preload_feature_bank,
    )
    if args.scope == RUNTIME_SCOPE_RESEARCH:
        split_record = manifest.get("split_assignments")
        if not isinstance(split_record, dict):
            raise ValueError("model manifest 缺少 split_assignments")
        if split_record.get("source_dataset_artifacts") != dataset_summary.get("dataset_artifacts"):
            raise ValueError(
                "research export 的 dataset 與模型 canonical split assignment 不一致；"
                "請勿重建同一 filter_id 的 dataset 後沿用舊 model"
            )
        if len(split_assignments) != len(events):
            raise ValueError(
                f"research dataset 與 split assignment row_count 不一致: events={len(events)}, split={len(split_assignments)}"
            )

    checkpoint = torch.load(artifact_paths.model_path, map_location="cpu")
    feature_count = int(checkpoint["feature_count"])
    context_count = int(checkpoint["context_count"])
    if feature_count != features.shape[2] or context_count != context.shape[1]:
        raise ValueError(
            f"model checkpoint 與 dataset 維度不一致: model=({feature_count}, {context_count}), "
            f"dataset=({features.shape[2]}, {context.shape[1]})"
        )
    model = build_model(feature_count, context_count)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logits_np = strict_parallel_batched_logits(
        torch,
        model,
        features,
        context,
        indices=None,
        batch_size=inference_batch_size,
        workers=inference_workers,
    )
    pass_probabilities = np.empty((len(events),), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(events), inference_batch_size):
            stop = min(start + inference_batch_size, len(events))
            logits = torch.from_numpy(logits_np[start:stop])
            pass_probabilities[start:stop] = (
                torch.softmax(logits, dim=1)[:, LABEL_PASS].cpu().numpy()
            )

    scored = events.copy()
    scored[SCORE_COLUMN] = pass_probabilities
    information_cutoff = str(manifest.get("model_information_cutoff", "")).strip()
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS:
        if not information_cutoff:
            raise ValueError("forward_oos 需要 manifest.model_information_cutoff")
        outer_policy = manifest.get("outer_oos_policy")
        if not isinstance(outer_policy, dict):
            raise ValueError("forward_oos 需要 manifest.outer_oos_policy")
        oos_start = pd.Timestamp(str(outer_policy.get("oos_start_date") or "")).normalize()
        configured_oos_end_text = str(outer_policy.get("configured_oos_end_date") or "").strip()
        source_range = dataset_summary.get("source_data_date_range")
        source_end_text = (
            str(source_range.get("end") or "").strip()
            if isinstance(source_range, dict)
            else ""
        )
        if not source_end_text:
            source_end_text = str(pd.to_datetime(scored["date"], errors="raise").max().date())
        oos_end = pd.Timestamp(configured_oos_end_text or source_end_text).normalize()
        event_dates = pd.to_datetime(scored["date"], errors="raise").dt.normalize()
        cutoff = pd.Timestamp(information_cutoff).normalize()
        scored = scored[
            (event_dates >= oos_start)
            & (event_dates <= oos_end)
            & (event_dates > cutoff)
        ].copy()
        if scored.empty:
            raise ValueError(
                "forward_oos 沒有落在既有 walk_forward_policy OOS 區間且晚於 model_information_cutoff 的事件；"
                f"oos={oos_start.date()}~{oos_end.date()}, cutoff={cutoff.date()}"
            )

    research_optional = [
        "label_eval_start_date",
        "label_eval_end_date",
        "label",
        "label_reason",
        "anchor_price",
        "pass_barrier_price",
        "reject_barrier_price",
        "max_upside_return",
        "max_downside_return",
        "first_hit_bar",
    ]
    optional_columns = research_optional if args.scope == RUNTIME_SCOPE_RESEARCH else []
    out_cols = list(SCORE_TABLE_REQUIRED_COLUMNS) + [column for column in optional_columns if column in scored.columns]

    policy = dataset_summary["policy"]
    high_len_values = policy.get("high_len_values")
    if not isinstance(high_len_values, list) or not high_len_values:
        high_len_values = sorted({int(value) for value in scored["high_len"].tolist()})
    event_range = _date_range(scored["date"])

    if args.scope == RUNTIME_SCOPE_RESEARCH:
        ensure_filter_output_dir(PROJECT_ROOT, filter_id=args.filter_id)
        score_path = resolve_filter_research_score_path(PROJECT_ROOT, args.filter_id)
        research_manifest_path = resolve_filter_research_manifest_path(PROJECT_ROOT, args.filter_id)
        scored[out_cols].to_csv(score_path, index=False, encoding="utf-8-sig")
        research_manifest = {
            "filter_id": str(args.filter_id),
            "scope": RUNTIME_SCOPE_RESEARCH,
            "model_information_cutoff": information_cutoff,
            "runtime_eligible": False,
            "outer_oos_policy": manifest.get("outer_oos_policy"),
            "source_split_assignments": manifest.get("split_assignments"),
            "score_table": _build_score_record(
                score_path,
                out_cols=out_cols,
                row_count=len(scored),
                high_len_values=[int(value) for value in high_len_values],
                event_range=event_range,
                dataset_summary=dataset_summary,
            ),
            "inference_execution": {
                "mode": "strict_parallel_fixed_batches",
                "batch_size": inference_batch_size,
                "workers": inference_workers,
                "feature_bank_preloaded": preload_feature_bank,
                "batch_boundaries_changed": False,
                "output_row_order_changed": False,
            },
            "reason": (
                "research rows include outer Selection and OOS; epochs and threshold must be fixed before OOS, "
                "and research output must never replace canonical runtime scores.csv"
            ),
        }
        write_json(research_manifest_path, research_manifest)
        print(f"已輸出研究分數: {score_path}")
        print(f"已輸出研究契約: {research_manifest_path}")
        print(f"scope={args.scope} rows={len(scored)} date_range={event_range}")
        return 0

    score_path = artifact_paths.score_path
    scored[out_cols].to_csv(score_path, index=False, encoding="utf-8-sig")
    source_data_range = dataset_summary.get("source_data_date_range")
    if not isinstance(source_data_range, dict):
        source_data_range = event_range
    available_through = str(source_data_range.get("end") or event_range["end"] or "")
    available_from = str(event_range["start"] or "")
    manifest["score_table"] = _build_score_record(
        score_path,
        out_cols=out_cols,
        row_count=len(scored),
        high_len_values=[int(value) for value in high_len_values],
        event_range=event_range,
        dataset_summary=dataset_summary,
    )
    manifest["score_inference_execution"] = {
        "mode": "strict_parallel_fixed_batches",
        "batch_size": inference_batch_size,
        "workers": inference_workers,
        "feature_bank_preloaded": preload_feature_bank,
        "batch_boundaries_changed": False,
        "output_row_order_changed": False,
    }
    manifest["runtime_eligibility"] = {
        "eligible": True,
        "scope": RUNTIME_SCOPE_FORWARD_OOS,
        "available_from": available_from,
        "available_through": available_through,
        "model_information_cutoff": information_cutoff,
        "reason": (
            "score rows are inside the shared walk_forward_policy OOS window, "
            "strictly after model_information_cutoff, and were produced by a fixed-epoch model"
        ),
    }
    manifest["score_filename"] = DEFAULT_SCORE_FILENAME
    write_json(artifact_paths.manifest_path, manifest)
    print(f"已輸出正式單一路徑: {score_path}")
    print(f"已更新: {artifact_paths.manifest_path}")
    print(f"scope={args.scope} rows={len(scored)} date_range={event_range}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
