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

from config.breakout_quality_experiments import SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_ALLOW_TF32,
)

from filters.breakout_quality.artifacts import (
    build_file_manifest,
    load_model_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    BreakoutQualityLabelPolicy,
    LABEL_PASS,
    FEATURE_COLUMNS,
    CONTEXT_COLUMNS,
    SCORE_COLUMN,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    RUNTIME_SCOPE_FORWARD_OOS,
    RUNTIME_SCOPE_RESEARCH,
)
from filters.breakout_quality.dataset_store import IndexedFeatureBank
from filters.breakout_quality.features import build_breakout_quality_inference_dataset_for_frame
from filters.breakout_quality.inference import (
    materialize_indexed_feature_inputs,
    strict_parallel_batched_logits,
    strict_unique_group_batched_logits,
)
from filters.breakout_quality.model import build_model, require_torch
from filters.breakout_quality.models.spec import (
    model_spec_from_manifest,
    validate_model_sequence_length,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
    resolve_torch_execution_plan,
)
from filters.breakout_quality.paths import (
    BreakoutQualityArtifactPaths,
    ensure_filter_model_output_dir,
    ensure_filter_output_dir,
    resolve_filter_artifact_paths,
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from tools.filters.breakout_quality.common import (
    discover_dataset_csv_inputs,
    load_dataset_frame,
    load_validated_dataset_bundle,
    load_validated_market_set_bank,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="匯出 breakout quality 分數表")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--experiment-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
        default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        help="要讀取／輸出的訓練實驗 profile",
    )
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
        "--device",
        choices=SUPPORTED_TORCH_DEVICES,
        default=BREAKOUT_QUALITY_TORCH_DEVICE,
        help="推論裝置；auto 優先 CUDA",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=SUPPORTED_MIXED_PRECISION_DTYPES,
        default=BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_ALLOW_TF32,
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


def _validate_export_scope_model_support(scope: str, model_spec) -> None:
    if str(scope) == RUNTIME_SCOPE_FORWARD_OOS and bool(model_spec.requires_market_set):
        raise ValueError(
            "Market Set architecture 目前只允許 research score export；"
            "Stage 0/1 尚未建立正式 scanner forward market-bank 契約"
        )


def _resolve_forward_export_write_paths(
    *,
    filter_id: str,
    experiment_profile: str,
    loaded_paths: BreakoutQualityArtifactPaths,
    project_root: str | Path = PROJECT_ROOT,
) -> BreakoutQualityArtifactPaths:
    """Return canonical writable paths and reject read-only legacy fallbacks."""

    canonical = resolve_filter_artifact_paths(
        project_root,
        filter_id,
        model_architecture=loaded_paths.model_architecture,
        experiment_profile=experiment_profile,
    )
    if loaded_paths.model_dir.resolve() != canonical.model_dir.resolve():
        raise ValueError(
            "舊 baseline 無 experiment profile 子目錄的工件僅供唯讀相容；"
            "請先用 baseline profile 重新訓練至 canonical profile 路徑，"
            "再匯出正式 forward-OOS scores"
        )
    return canonical


def _policy_from_manifest(payload: dict) -> BreakoutQualityLabelPolicy:
    return BreakoutQualityLabelPolicy(
        feature_window_bars=int(payload["feature_window_bars"]),
        label_horizon_bars=int(payload["label_horizon_bars"]),
        label_path_cache_bars=int(payload["label_path_cache_bars"]),
        high_len_values=tuple(int(value) for value in payload["high_len_values"]),
        min_mfe_return=float(payload["min_mfe_return"]),
        min_reward_risk_ratio=float(payload["min_reward_risk_ratio"]),
        max_adverse_return=float(payload["max_adverse_return"]),
        benchmark_ticker=str(payload["benchmark_ticker"]),
    )


def _resolve_forward_runtime_dates(manifest: dict) -> dict[str, pd.Timestamp | None]:
    """Resolve signal-score coverage separately from the OOS execution window."""

    outer_policy = manifest.get("outer_oos_policy")
    if not isinstance(outer_policy, dict):
        raise ValueError("forward_oos 需要 manifest.outer_oos_policy")
    information_cutoff_text = str(manifest.get("model_information_cutoff", "")).strip()
    if not information_cutoff_text:
        raise ValueError("forward_oos 需要 manifest.model_information_cutoff")

    cutoff = pd.Timestamp(information_cutoff_text).normalize()
    execution_start = pd.Timestamp(str(outer_policy.get("oos_start_date") or "")).normalize()
    configured_end_text = str(outer_policy.get("configured_oos_end_date") or "").strip()
    execution_end = pd.Timestamp(configured_end_text).normalize() if configured_end_text else None
    # (AI註: cutoff 當日收盤後模型資訊已完整，可用該日 breakout signal
    # 建立下一交易日盤前訂單；正式 score 因此從 cutoff 當日開始。)
    score_signal_start = cutoff
    if score_signal_start > execution_start:
        raise ValueError(
            "forward_oos model_information_cutoff 必須早於策略 OOS 執行起日: "
            f"cutoff={cutoff.date()}, execution_start={execution_start.date()}"
        )
    return {
        "model_information_cutoff": cutoff,
        "score_signal_start": score_signal_start,
        "execution_start": execution_start,
        "execution_end": execution_end,
    }


def _merge_source_range(
    current_start: pd.Timestamp | None,
    current_end: pd.Timestamp | None,
    frame: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if frame.empty:
        return current_start, current_end
    start = pd.Timestamp(frame.index.min()).normalize()
    end = pd.Timestamp(frame.index.max()).normalize()
    return (start if current_start is None else min(current_start, start), end if current_end is None else max(current_end, end))


def _build_forward_runtime_inputs(
    *,
    dataset_summary: dict,
    policy: BreakoutQualityLabelPolicy,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None,
    require_context: bool,
) -> tuple[IndexedFeatureBank, np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    dataset_profile = str(dataset_summary.get("dataset") or "").strip().lower()
    if dataset_profile not in {"full", "reduced"}:
        raise ValueError("forward_oos dataset summary 缺少合法 dataset profile")
    csv_inputs, duplicate_lines = discover_dataset_csv_inputs(PROJECT_ROOT, dataset_profile)
    input_map = {str(ticker): Path(path) for ticker, path in csv_inputs}
    benchmark_path = input_map.get(str(policy.benchmark_ticker))
    if benchmark_path is None:
        raise FileNotFoundError(f"找不到 benchmark ticker: {policy.benchmark_ticker}")
    # # (AI註: forward score export 必須覆蓋 runtime 可能載入的所有股票。
    # #        資料不足 300 bars 的股票仍可能形成較短 high_len 突破，
    # #        因此只要求一列有效 OHLCV，再由 feature builder 明確標為不可評分。)
    source_min_rows = 1
    benchmark = load_dataset_frame(
        benchmark_path,
        str(policy.benchmark_ticker),
        min_rows=source_min_rows,
    )

    # # (AI註: 正式 runtime score 必須覆蓋目前 Portfolio 可載入的完整 universe，
    # #        不得沿用訓練 Dataset 的 --max-tickers 限制；benchmark ticker 本身也可能
    # #        被正式策略視為候選，因此同樣必須有 score 或明確不可評分紀錄。)
    tickers = sorted(input_map)
    source_selection = dataset_summary.get("source_selection")
    training_requested_max_tickers = 0
    if isinstance(source_selection, dict):
        training_requested_max_tickers = max(
            0,
            int(source_selection.get("requested_max_tickers") or 0),
        )

    feature_chunks: list[np.ndarray] = []
    context_chunks: list[np.ndarray] = []
    group_index_chunks: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    unavailable_frames: list[pd.DataFrame] = []
    group_offset = 0
    skipped_tickers: list[dict[str, str]] = []
    source_start, source_end = _merge_source_range(None, None, benchmark)

    for ticker in tickers:
        try:
            stock = (
                benchmark
                if ticker == str(policy.benchmark_ticker)
                else load_dataset_frame(
                    input_map[ticker],
                    ticker,
                    min_rows=source_min_rows,
                )
            )
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            skipped_tickers.append({"ticker": ticker, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        source_start, source_end = _merge_source_range(source_start, source_end, stock)
        inference_dataset = build_breakout_quality_inference_dataset_for_frame(
            stock,
            benchmark,
            ticker=ticker,
            policy=policy,
            start_date=start_date,
            end_date=end_date,
            require_context=require_context,
        )
        local_group_count = int(len(inference_dataset.feature_bank))
        if local_group_count:
            feature_chunks.append(inference_dataset.feature_bank)
            context_chunks.append(inference_dataset.context)
            group_index_chunks.append(
                inference_dataset.event_group_index.astype(np.int64) + int(group_offset)
            )
            event_frames.append(inference_dataset.events)
            group_offset += local_group_count
        if not inference_dataset.unavailable_events.empty:
            unavailable_frames.append(inference_dataset.unavailable_events)

    features = (
        np.concatenate(feature_chunks, axis=0)
        if feature_chunks
        else np.empty((0, int(policy.feature_window_bars), len(FEATURE_COLUMNS)), dtype=np.float32)
    )
    context = (
        np.concatenate(context_chunks, axis=0)
        if context_chunks
        else np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32)
    )
    event_group_index = (
        np.concatenate(group_index_chunks, axis=0).astype(np.int32, copy=False)
        if group_index_chunks
        else np.empty((0,), dtype=np.int32)
    )
    events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=["ticker", "date", "high_len"])
    )
    unavailable = (
        pd.concat(unavailable_frames, ignore_index=True)
        if unavailable_frames
        else pd.DataFrame(columns=["ticker", "date", "high_len", "reason"])
    )
    candidate_keys = pd.concat(
        [events[["ticker", "date", "high_len"]], unavailable[["ticker", "date", "high_len"]]],
        ignore_index=True,
    )
    if candidate_keys.empty:
        raise ValueError("forward_oos current runtime candidate universe 為空")
    if candidate_keys.duplicated(["ticker", "date", "high_len"]).any():
        raise ValueError("forward_oos current runtime candidate universe 出現重複 ticker/date/high_len")
    if len(context) != len(events) or len(event_group_index) != len(events):
        raise ValueError("forward_oos inference context/event mapping 長度不一致")

    coverage_end = source_end
    if coverage_end is not None and end_date is not None:
        coverage_end = min(coverage_end, pd.Timestamp(end_date).normalize())
    source_range = {
        "start": None if source_start is None else str(source_start.date()),
        "end": None if coverage_end is None else str(coverage_end.date()),
    }
    metadata = {
        "dataset_profile": dataset_profile,
        "source_data_inventory": build_source_data_inventory(PROJECT_ROOT, dataset_profile),
        "source_data_date_range": source_range,
        "training_requested_max_tickers": training_requested_max_tickers,
        "runtime_ticker_limit": 0,
        "processed_ticker_count": int(len(tickers) - len(skipped_tickers)),
        "skipped_tickers": skipped_tickers,
        "duplicate_input_messages": list(duplicate_lines),
        "candidate_event_row_count": int(len(candidate_keys)),
        "model_scored_event_row_count": int(len(events)),
        "conservative_reject_event_row_count": int(len(unavailable)),
    }
    return IndexedFeatureBank(features, event_group_index), context, events, unavailable, metadata


def _build_score_record(
    score_path: Path,
    *,
    out_cols: list[str],
    row_count: int,
    high_len_values: list[int],
    event_range: dict[str, str | None],
    dataset_summary: dict,
    runtime_source: dict | None = None,
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
    if runtime_source is not None:
        score_record["runtime_source"] = runtime_source
    return score_record


def main(argv=None) -> int:
    args = parse_args(argv)
    inference_batch_size = int(args.inference_batch_size)
    inference_workers = int(args.inference_workers)
    preload_feature_bank = bool(args.preload_feature_bank)
    if inference_batch_size < 1 or inference_workers < 1:
        raise ValueError("inference-batch-size 與 inference-workers 必須 >=1")
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
    model_contract = load_model_artifact_contract(
        str(PROJECT_ROOT),
        str(args.filter_id),
        experiment_profile=str(args.experiment_profile),
    )
    artifact_paths = model_contract.paths
    writable_paths = None
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS:
        writable_paths = _resolve_forward_export_write_paths(
            filter_id=str(args.filter_id),
            experiment_profile=str(args.experiment_profile),
            loaded_paths=artifact_paths,
        )
    manifest = model_contract.manifest
    split_assignments = load_split_assignment_frame(
        str(PROJECT_ROOT),
        str(args.filter_id),
        experiment_profile=str(args.experiment_profile),
    )
    model_policy = manifest.get("policy")
    if not isinstance(model_policy, dict):
        raise ValueError("model manifest 缺少 policy object")
    dataset_summary, features, context, _labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=model_policy,
    )
    if args.scope == RUNTIME_SCOPE_RESEARCH:
        features, context = materialize_indexed_feature_inputs(
            features,
            context,
            enabled=preload_feature_bank,
        )
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
    if not isinstance(checkpoint, dict):
        raise ValueError("breakout quality model checkpoint 根節點必須是 object")
    feature_count = int(checkpoint["feature_count"])
    context_count = int(checkpoint["context_count"])
    sequence_length = int(checkpoint.get("sequence_length", -1))
    if (
        feature_count != features.shape[2]
        or context_count != context.shape[1]
        or sequence_length != features.shape[1]
    ):
        raise ValueError(
            "model checkpoint 與 dataset 維度不一致: "
            f"model=({sequence_length}, {feature_count}, {context_count}), "
            f"dataset=({features.shape[1]}, {features.shape[2]}, {context.shape[1]})"
        )
    checkpoint_spec_payload = checkpoint.get("model_spec")
    if not isinstance(checkpoint_spec_payload, dict):
        raise ValueError("model checkpoint 缺少 model_spec")
    checkpoint_spec = model_spec_from_manifest(checkpoint_spec_payload)
    validate_model_sequence_length(checkpoint_spec, sequence_length)
    _validate_export_scope_model_support(args.scope, checkpoint_spec)
    market_set_bank = (
        load_validated_market_set_bank(
            args.filter_id,
            dataset_summary=dataset_summary,
            expected_model_spec=checkpoint_spec,
        )
        if bool(checkpoint_spec.requires_market_set)
        else None
    )
    if args.scope == RUNTIME_SCOPE_RESEARCH:
        split_record = manifest.get("split_assignments")
        if not isinstance(split_record, dict):
            raise ValueError("model manifest 缺少 split_assignments")
        expected_market_artifacts = (
            dataset_summary.get("market_set_artifacts")
            if bool(checkpoint_spec.requires_market_set)
            else None
        )
        if split_record.get("source_market_set_artifacts") != expected_market_artifacts:
            raise ValueError(
                "research export 的 market-set dataset 與模型 split assignment 不一致；"
                "請完整重建並重新訓練該 filter_id"
            )
    manifest_spec_payload = manifest.get("model_spec")
    if checkpoint_spec_payload != manifest_spec_payload:
        raise ValueError("model checkpoint.model_spec 與 manifest.model_spec 不一致")
    if checkpoint_spec.architecture != artifact_paths.model_architecture:
        raise ValueError("model checkpoint architecture 與工件路徑不一致")
    checkpoint_profile = str(
        checkpoint.get("experiment_profile") or "baseline"
    ).strip()
    manifest_profile = str(
        manifest.get("experiment_profile") or "baseline"
    ).strip()
    if checkpoint_profile != manifest_profile or checkpoint_profile != artifact_paths.experiment_profile:
        raise ValueError("model checkpoint experiment profile 與 manifest／工件路徑不一致")
    if checkpoint.get("experiment_settings") != manifest.get("experiment_settings"):
        raise ValueError("model checkpoint experiment settings 與 manifest 不一致")
    for field_name in (
        "trainable_parameter_count",
        "total_parameter_count",
        "frozen_parameter_count",
        "self_supervised_pretraining",
        "external_pretrained_encoder",
        "market_set_contract",
        "market_set_artifacts",
    ):
        if checkpoint.get(field_name) != manifest.get(field_name):
            raise ValueError(f"model checkpoint {field_name} 與 manifest 不一致")
    checkpoint_torch_execution = checkpoint.get("torch_execution")
    manifest_torch_execution = manifest.get("torch_execution")
    if checkpoint_torch_execution is not None or manifest_torch_execution is not None:
        if checkpoint_torch_execution != manifest_torch_execution:
            raise ValueError("model checkpoint torch_execution 與 manifest 不一致")
    model = build_model(
        feature_count,
        context_count,
        model_spec=checkpoint_spec_payload,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(execution_plan.device)
    model.eval()
    shared_group_score_broadcast = not bool(checkpoint_spec.use_dataset_context)
    unavailable_events = pd.DataFrame(columns=["ticker", "date", "high_len", "reason"])
    runtime_source_metadata = None
    forward_runtime_dates = None
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS:
        forward_runtime_dates = _resolve_forward_runtime_dates(manifest)
        information_cutoff = str(manifest.get("model_information_cutoff", "")).strip()
        policy = _policy_from_manifest(model_policy)
        features, context, events, unavailable_events, runtime_source_metadata = _build_forward_runtime_inputs(
            dataset_summary=dataset_summary,
            policy=policy,
            start_date=forward_runtime_dates["score_signal_start"],
            end_date=forward_runtime_dates["execution_end"],
            require_context=bool(checkpoint_spec.use_dataset_context),
        )
        runtime_source_metadata.update(
            {
                "score_signal_start_date": str(forward_runtime_dates["score_signal_start"].date()),
                "strategy_execution_start_date": str(forward_runtime_dates["execution_start"].date()),
            }
        )
        features, context = materialize_indexed_feature_inputs(
            features,
            context,
            enabled=preload_feature_bank,
        )
        if (
            feature_count != features.shape[2]
            or context_count != context.shape[1]
            or sequence_length != features.shape[1]
        ):
            raise ValueError(
                "model checkpoint 與 current runtime candidate feature 維度不一致: "
                f"model=({sequence_length}, {feature_count}, {context_count}), "
                f"runtime=({features.shape[1]}, {features.shape[2]}, {context.shape[1]})"
            )

    if shared_group_score_broadcast:
        group_logits_np, event_to_group = strict_unique_group_batched_logits(
            torch,
            model,
            features,
            context,
            batch_size=inference_batch_size,
            workers=inference_workers,
            execution_plan=execution_plan,
            market_set_bank=market_set_bank,
        )
        with torch.no_grad():
            group_pass_probabilities = (
                torch.softmax(torch.from_numpy(group_logits_np), dim=1)[:, LABEL_PASS]
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        pass_probabilities = np.asarray(
            group_pass_probabilities[event_to_group],
            dtype=np.float32,
        )
        inference_input_row_count = int(group_logits_np.shape[0])
    else:
        logits_np = strict_parallel_batched_logits(
            torch,
            model,
            features,
            context,
            indices=None,
            batch_size=inference_batch_size,
            workers=inference_workers,
            execution_plan=execution_plan,
            market_set_bank=market_set_bank,
        )
        pass_probabilities = np.empty((len(events),), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(events), inference_batch_size):
                stop = min(start + inference_batch_size, len(events))
                logits = torch.from_numpy(logits_np[start:stop])
                pass_probabilities[start:stop] = (
                    torch.softmax(logits, dim=1)[:, LABEL_PASS].cpu().numpy()
                )
        inference_input_row_count = int(len(events))

    scored = events.copy()
    scored[SCORE_COLUMN] = pass_probabilities
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS and not unavailable_events.empty:
        conservative_rows = unavailable_events[["ticker", "date", "high_len"]].copy()
        conservative_rows[SCORE_COLUMN] = np.float32(0.0)
        scored = pd.concat([scored, conservative_rows], ignore_index=True)
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS:
        scored = scored.sort_values(["ticker", "date", "high_len"], kind="mergesort").reset_index(drop=True)
    information_cutoff = str(manifest.get("model_information_cutoff", "")).strip()
    if args.scope == RUNTIME_SCOPE_FORWARD_OOS:
        if forward_runtime_dates is None:
            raise RuntimeError("forward_oos runtime dates 尚未初始化")
        source_range = (runtime_source_metadata or {}).get("source_data_date_range")
        if not isinstance(source_range, dict):
            source_range = dataset_summary.get("source_data_date_range")
        source_end_text = (
            str(source_range.get("end") or "").strip()
            if isinstance(source_range, dict)
            else ""
        )
        if not source_end_text:
            source_end_text = str(pd.to_datetime(scored["date"], errors="raise").max().date())
        score_end = pd.Timestamp(
            forward_runtime_dates["execution_end"] or pd.Timestamp(source_end_text)
        ).normalize()
        event_dates = pd.to_datetime(scored["date"], errors="raise").dt.normalize()
        scored = scored[
            (event_dates >= forward_runtime_dates["score_signal_start"])
            & (event_dates <= score_end)
            & (event_dates >= forward_runtime_dates["model_information_cutoff"])
        ].copy()
        if scored.empty:
            raise ValueError(
                "forward_oos 沒有位於 model_information_cutoff 當日或之後、且可供 OOS 盤前決策使用的訊號事件；"
                f"signal_start={forward_runtime_dates['score_signal_start'].date()}~{score_end.date()}, "
                f"execution_start={forward_runtime_dates['execution_start'].date()}, "
                f"cutoff={forward_runtime_dates['model_information_cutoff'].date()}"
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
        "decision_mfe_return",
        "decision_mae_return",
        "decision_reward_risk_ratio",
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
        ensure_filter_model_output_dir(
            PROJECT_ROOT, args.filter_id, experiment_profile=args.experiment_profile
        )
        score_path = resolve_filter_research_score_path(
            PROJECT_ROOT, args.filter_id, experiment_profile=args.experiment_profile
        )
        research_manifest_path = resolve_filter_research_manifest_path(
            PROJECT_ROOT, args.filter_id, experiment_profile=args.experiment_profile
        )
        scored[out_cols].to_csv(score_path, index=False, encoding="utf-8-sig")
        research_manifest = {
            "filter_id": str(args.filter_id),
            "model_architecture": checkpoint_spec.architecture,
            "experiment_profile": manifest.get("experiment_profile", "baseline"),
            "experiment_settings": manifest.get("experiment_settings"),
            "model_spec": checkpoint_spec.as_manifest_payload(),
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
                "mode": (
                    (
                        "cuda_serial_unique_group_fixed_batches"
                        if execution_plan.device_type == "cuda"
                        else "strict_parallel_unique_group_fixed_batches"
                    )
                    if shared_group_score_broadcast
                    else (
                        "cuda_serial_fixed_batches"
                        if execution_plan.device_type == "cuda"
                        else "strict_parallel_fixed_batches"
                    )
                ),
                "batch_size": inference_batch_size,
                "workers": (1 if execution_plan.device_type == "cuda" else inference_workers),
                "torch_execution": execution_plan.as_manifest_payload(),
                "feature_bank_preloaded": preload_feature_bank,
                "market_set_input": (
                    dataset_summary.get("market_set_contract")
                    if bool(checkpoint_spec.requires_market_set)
                    else None
                ),
                "inference_unit": (
                    "unique_ticker_date_feature_group"
                    if shared_group_score_broadcast
                    else "event_row"
                ),
                "inference_input_row_count": inference_input_row_count,
                "output_event_row_count": int(len(scored)),
                "shared_group_score_broadcast": shared_group_score_broadcast,
                "inference_batch_boundaries_defined_over": (
                    "unique_ticker_date_feature_group"
                    if shared_group_score_broadcast
                    else "event_row"
                ),
                "event_row_batch_boundaries_preserved": not shared_group_score_broadcast,
                "output_row_order_changed": bool(args.scope == RUNTIME_SCOPE_FORWARD_OOS),
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

    if writable_paths is None:
        raise RuntimeError("forward_oos writable paths 尚未初始化")
    score_path = writable_paths.score_path
    scored[out_cols].to_csv(score_path, index=False, encoding="utf-8-sig")
    source_data_range = (runtime_source_metadata or {}).get("source_data_date_range")
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
        runtime_source=runtime_source_metadata,
    )
    unavailable_path = score_path.with_name(DEFAULT_UNAVAILABLE_SCORE_FILENAME)
    unavailable_events.to_csv(unavailable_path, index=False, encoding="utf-8-sig")
    unavailable_record = build_file_manifest(unavailable_path)
    unavailable_record.update(
        {
            "columns": ["ticker", "date", "high_len", "reason"],
            "row_count": int(len(unavailable_events)),
            "reason_counts": {
                str(key): int(value)
                for key, value in unavailable_events["reason"].value_counts().sort_index().items()
            },
            "conservative_runtime_score": 0.0,
            "runtime_action": "reject",
        }
    )
    manifest["conservative_unscorable_events"] = unavailable_record
    manifest["score_inference_execution"] = {
        "mode": (
            (
                "cuda_serial_unique_group_fixed_batches"
                if execution_plan.device_type == "cuda"
                else "strict_parallel_unique_group_fixed_batches"
            )
            if shared_group_score_broadcast
            else (
                "cuda_serial_fixed_batches"
                if execution_plan.device_type == "cuda"
                else "strict_parallel_fixed_batches"
            )
        ),
        "batch_size": inference_batch_size,
        "workers": (1 if execution_plan.device_type == "cuda" else inference_workers),
        "torch_execution": execution_plan.as_manifest_payload(),
        "feature_bank_preloaded": preload_feature_bank,
        "inference_unit": (
            "unique_ticker_date_feature_group"
            if shared_group_score_broadcast
            else "event_row"
        ),
        "inference_input_row_count": inference_input_row_count,
        "output_event_row_count": int(len(scored)),
        "model_scored_event_row_count": int(len(events)),
        "conservative_reject_event_row_count": int(len(unavailable_events)),
        "shared_group_score_broadcast": shared_group_score_broadcast,
        "inference_batch_boundaries_defined_over": (
            "unique_ticker_date_feature_group"
            if shared_group_score_broadcast
            else "event_row"
        ),
        "event_row_batch_boundaries_preserved": not shared_group_score_broadcast,
        "output_row_order_changed": bool(args.scope == RUNTIME_SCOPE_FORWARD_OOS),
    }
    if forward_runtime_dates is None:
        raise RuntimeError("forward_oos runtime dates 尚未初始化")
    manifest["runtime_eligibility"] = {
        "eligible": True,
        "scope": RUNTIME_SCOPE_FORWARD_OOS,
        "available_from": available_from,
        "available_through": available_through,
        "required_signal_start": str(forward_runtime_dates["score_signal_start"].date()),
        "execution_start": str(forward_runtime_dates["execution_start"].date()),
        "model_information_cutoff": information_cutoff,
        "reason": (
            "score rows begin at model_information_cutoff close and include the pre-execution signal anchor "
            "required by next-session OOS orders; strategy execution still begins at outer_oos_policy.oos_start_date"
        ),
    }
    manifest["score_filename"] = DEFAULT_SCORE_FILENAME
    write_json(writable_paths.manifest_path, manifest)
    print(f"已輸出正式單一路徑: {score_path}")
    print(f"已輸出不可評分事件稽核: {unavailable_path}")
    print(
        "runtime candidate coverage: "
        f"total={len(scored):,}, model_scored={len(events):,}, "
        f"conservative_reject={len(unavailable_events):,}"
    )
    print(f"已更新: {writable_paths.manifest_path}")
    print(f"scope={args.scope} rows={len(scored)} date_range={event_range}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
