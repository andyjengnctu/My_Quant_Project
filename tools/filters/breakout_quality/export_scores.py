"""Export breakout quality research or formal forward-OOS score tables."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import pandas as pd

from filters.breakout_quality.artifacts import build_file_manifest, load_model_artifact_contract
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
    torch, _nn = require_torch()
    model_contract = load_model_artifact_contract(str(PROJECT_ROOT), str(args.filter_id))
    artifact_paths = model_contract.paths
    manifest = model_contract.manifest
    model_policy = manifest.get("policy")
    if not isinstance(model_policy, dict):
        raise ValueError("model manifest 缺少 policy object")
    dataset_summary, features, context, _labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=model_policy,
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
    with torch.no_grad():
        logits = model(torch.from_numpy(features), torch.from_numpy(context))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    scored = events.copy()
    scored[SCORE_COLUMN] = probabilities[:, LABEL_PASS]
    information_cutoff = str(manifest.get("model_information_cutoff", "")).strip()
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS:
        if not information_cutoff:
            raise ValueError("forward_oos 需要 manifest.model_information_cutoff")
        cutoff = pd.Timestamp(information_cutoff)
        scored = scored[pd.to_datetime(scored["date"], errors="raise") > cutoff].copy()
        if scored.empty:
            raise ValueError(
                "forward_oos 沒有可匯出的事件；請保留既有 model.pt，使用更新到 cutoff 之後的資料重新 build_dataset，"
                "不要重新 train 後再匯出同一份資料"
            )

    research_optional = [
        "entry_date",
        "label_eval_start_date",
        "label_eval_end_date",
        "label",
        "label_reason",
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
            "score_table": _build_score_record(
                score_path,
                out_cols=out_cols,
                row_count=len(scored),
                high_len_values=[int(value) for value in high_len_values],
                event_range=event_range,
                dataset_summary=dataset_summary,
            ),
            "reason": "research rows may overlap model train/validation and must never replace canonical runtime scores.csv",
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
    manifest["runtime_eligibility"] = {
        "eligible": True,
        "scope": RUNTIME_SCOPE_FORWARD_OOS,
        "available_from": available_from,
        "available_through": available_through,
        "model_information_cutoff": information_cutoff,
        "reason": "score rows are strictly after model_information_cutoff",
    }
    manifest["score_filename"] = DEFAULT_SCORE_FILENAME
    write_json(artifact_paths.manifest_path, manifest)
    print(f"已輸出正式單一路徑: {score_path}")
    print(f"已更新: {artifact_paths.manifest_path}")
    print(f"scope={args.scope} rows={len(scored)} date_range={event_range}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
