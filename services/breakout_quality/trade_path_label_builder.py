"""Build the A2 realized trade-path label dataset without overwriting 9A."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd

from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
)
from core.file_integrity import canonical_json_sha256 as _canonical_hash
from core.dataset_profiles import get_dataset_dir
from core.runtime_utils import get_taipei_now
from filters.breakout_quality.artifacts import build_file_manifest, compute_file_sha256
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_title,
)
from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    TRADE_PATH_FILTER_ID,
    label_manifest_payload_from_policy_manifest,
    trade_path_label_policy_payload,
)
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    resolve_dataset_paths,
    save_npy_atomic,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_BASE_FILTER_ID,
    TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH,
    TRADE_PATH_HISTORICAL_TEACHER_PARAMS_RELATIVE_PATH,
    TRADE_PATH_LABEL_CONTRACT_VERSION,
    TRADE_PATH_LABEL_ID,
    TRADE_PATH_REASON_INACTIVE_HIGH_LEN,
    TRADE_PATH_REASON_SIGNAL_DATE_MISSING,
    TRADE_PATH_REASON_TEACHER_UNAVAILABLE,
    TRADE_PATH_RESEARCH_FILTER_ID,
    TRADE_PATH_SELECTION_BASELINE_PARAMS_RELATIVE_PATH,
    build_signal_cache,
    build_trade_path_excluded_event_update,
    load_active_param_schedule,
    merge_active_param_schedules,
    simulate_realized_trade_path_label,
)
from filters.breakout_quality.workflow_io import (
    PROJECT_ROOT,
    dataset_output_dir,
    discover_dataset_csv_inputs,
    event_group_summary,
    label_counts,
    load_dataset_frame,
    load_validated_dataset_bundle,
    read_json,
    write_json,
)
from services.research.strategy_compare_engine import (
    PARAM_POLICY_BASE_FINALIST_BEST,
)
from services.optimizer.strategy_param_training import (
    prepare_selection_historical_p2_params,
)

SCHEMA_VERSION = 2
DEFAULT_WORKERS = 4


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "建立A2 realized trade-path Label Dataset；初次miss buy維持pending，"
            "延續成交後依正式淨Realized R標記PASS／REJECT"
        )
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default="full")
    parser.add_argument("--source-filter-id", default=TRADE_PATH_BASE_FILTER_ID)
    parser.add_argument("--filter-id", default=TRADE_PATH_RESEARCH_FILTER_ID)
    parser.add_argument(
        "--historical-teacher-trials-per-fold",
        type=int,
        default=int(settings.strategy_trials_per_fold),
    )
    parser.add_argument("--max-positions", type=int, default=int(settings.strategy_max_positions))
    parser.add_argument("--rotation", choices=("off", "on"), default=str(settings.strategy_rotation))
    parser.add_argument("--fixed-risk", type=float, default=float(settings.strategy_fixed_risk))
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=float(settings.strategy_max_position_cap_pct),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)



def _ticker_shard_path(shard_dir: Path, ticker: str) -> Path:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(ticker))
    digest = hashlib.sha256(str(ticker).encode("utf-8")).hexdigest()[:12]
    return shard_dir / f"{safe}_{digest}.csv"


def _load_valid_ticker_shard(
    path: Path,
    *,
    expected_event_indices: set[int],
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, low_memory=False)
        if "_event_index" not in frame.columns:
            return None
        observed = set(
            pd.to_numeric(frame["_event_index"], errors="raise").astype(np.int64).tolist()
        )
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None
    if len(frame) != len(expected_event_indices) or observed != expected_event_indices:
        return None
    return frame.sort_values("_event_index", kind="mergesort").reset_index(drop=True)


def _write_ticker_shard(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("_event_index", kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return frame


def _ensure_historical_teacher_params(root: Path, args) -> Path:
    result = prepare_selection_historical_p2_params(
        project_root=root,
        dataset=str(args.dataset),
        param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        trials_per_fold=int(args.historical_teacher_trials_per_fold),
        max_positions=int(args.max_positions),
        rotation=str(args.rotation),
        fixed_risk=float(args.fixed_risk),
        max_position_cap_pct=float(args.max_position_cap_pct),
        optimizer_seed=int(get_breakout_quality_workflow_settings().seed),
        resume_parameter_training=bool(args.resume),
        quiet=bool(args.quiet),
    )
    return Path(result["params_path"])


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, str | None]:
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"start": None, "end": None}
    return {
        "start": values.min().strftime("%Y-%m-%d"),
        "end": values.max().strftime("%Y-%m-%d"),
    }


def _ticker_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ticker = str(payload["ticker"])
    path = Path(payload["path"])
    rows = pd.DataFrame(payload["rows"])
    historical_schedule = load_active_param_schedule(payload["historical_params_path"])
    forward_schedule = load_active_param_schedule(payload["forward_params_path"])
    schedule = merge_active_param_schedules(historical_schedule, forward_schedule)
    frame = load_dataset_frame(path, ticker, min_rows=50)
    frame.attrs["ticker"] = ticker
    index_map = {
        pd.Timestamp(value).strftime("%Y-%m-%d"): index
        for index, value in enumerate(frame.index)
    }
    signal_cache: dict[str, Any] = {}
    output: list[dict[str, Any]] = []
    for group_index, group_rows in rows.groupby("group_index", sort=False):
        signal_date = str(group_rows.iloc[0]["date"])
        schedule_item = schedule.resolve(signal_date)
        updates_by_event_index: dict[int, dict[str, Any]] = {}
        if schedule_item is None:
            for _, row in group_rows.iterrows():
                updates_by_event_index[int(row["_event_index"])] = (
                    build_trade_path_excluded_event_update(
                        TRADE_PATH_REASON_TEACHER_UNAVAILABLE,
                        end_date=signal_date,
                        teacher_effective_date=None,
                    )
                )
        else:
            effective_date, params = schedule_item
            matching = group_rows[
                pd.to_numeric(group_rows["high_len"], errors="raise").astype(int)
                == int(params.high_len)
            ]
            for _, row in group_rows.iterrows():
                updates_by_event_index[int(row["_event_index"])] = (
                    build_trade_path_excluded_event_update(
                        TRADE_PATH_REASON_INACTIVE_HIGH_LEN,
                        end_date=signal_date,
                        teacher_effective_date=effective_date,
                    )
                )
            if len(matching) > 1:
                raise ValueError(
                    f"同一ticker/date有多筆teacher high_len: {ticker}/{signal_date}"
                )
            if len(matching) == 1:
                signal_pos = index_map.get(signal_date)
                selected = matching.iloc[0]
                event_index = int(selected["_event_index"])
                if signal_pos is None:
                    updates_by_event_index[event_index] = (
                        build_trade_path_excluded_event_update(
                            TRADE_PATH_REASON_SIGNAL_DATE_MISSING,
                            end_date=signal_date,
                            teacher_effective_date=effective_date,
                        )
                    )
                else:
                    if effective_date not in signal_cache:
                        signal_cache[effective_date] = build_signal_cache(
                            frame, params, ticker=ticker
                        )
                    result = simulate_realized_trade_path_label(
                        frame,
                        ticker=ticker,
                        signal_pos=int(signal_pos),
                        params=params,
                        teacher_effective_date=effective_date,
                        precomputed_signals=signal_cache[effective_date],
                    )
                    updates_by_event_index[event_index] = result.as_event_update()
        for event_index, updates in updates_by_event_index.items():
            output.append({"_event_index": event_index, **updates})
    return output


def _build_artifact_records(paths) -> dict[str, dict[str, Any]]:
    return {
        name: build_file_manifest(path)
        for name, path in paths.artifact_paths().items()
    }


def _build_dataset(root: Path, args, historical_path: Path, forward_path: Path) -> dict[str, Any]:
    source_filter_id = str(args.source_filter_id)
    target_filter_id = str(args.filter_id)
    if source_filter_id == target_filter_id:
        raise ValueError("trade-path derived dataset不得覆蓋base filter dataset")
    base_summary, _features, _context, _labels, base_events = load_validated_dataset_bundle(
        source_filter_id,
        require_current_source=True,
    )
    source_paths = resolve_dataset_paths(dataset_output_dir(source_filter_id))
    target_paths = resolve_dataset_paths(dataset_output_dir(target_filter_id))
    work_dir = target_paths.output_dir / ".trade_path_build"
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "event_updates.csv"
    shard_dir = work_dir / "ticker_shards"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "label_contract_version": TRADE_PATH_LABEL_CONTRACT_VERSION,
        "label_id": TRADE_PATH_LABEL_ID,
        "dataset": str(args.dataset),
        "source_filter_id": source_filter_id,
        "source_dataset_sha256": compute_file_sha256(source_paths.summary),
        "historical_teacher_sha256": compute_file_sha256(historical_path),
        "forward_teacher_sha256": compute_file_sha256(forward_path),
        "source_inventory": build_source_data_inventory(root, args.dataset),
    }
    identity_sha = _canonical_hash(identity)
    identity_path = work_dir / "build_identity.json"
    prior_identity = None
    if identity_path.is_file():
        try:
            prior_identity = read_json(identity_path)
        except (OSError, ValueError, json.JSONDecodeError):
            prior_identity = None
    identity_changed = (
        bool(args.force_rebuild)
        or not isinstance(prior_identity, dict)
        or prior_identity.get("identity_sha256") != identity_sha
    )
    if identity_changed:
        result_path.unlink(missing_ok=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
    write_json(identity_path, {**identity, "identity_sha256": identity_sha})

    events = base_events.copy().reset_index(drop=True)
    events["_event_index"] = np.arange(len(events), dtype=np.int64)
    updates = None
    if result_path.is_file() and bool(args.resume):
        try:
            candidate = pd.read_csv(result_path, low_memory=False)
            if "_event_index" not in candidate.columns:
                raise ValueError("completed event updates缺少_event_index")
            observed = set(
                pd.to_numeric(candidate["_event_index"], errors="raise")
                .astype(np.int64)
                .tolist()
            )
            expected = set(events["_event_index"].astype(np.int64).tolist())
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            candidate = None
            observed = set()
            expected = set(events["_event_index"].astype(np.int64).tolist())
        if candidate is not None and len(candidate) == len(events) and observed == expected:
            updates = candidate
        else:
            result_path.unlink(missing_ok=True)

    if updates is None:
        csv_inputs, duplicate_lines = discover_dataset_csv_inputs(root, args.dataset)
        for line in duplicate_lines:
            print(line)
        path_by_ticker = {str(ticker): str(path) for ticker, path in csv_inputs}
        shard_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        shard_frames: dict[str, pd.DataFrame] = {}
        expected_tickers: list[str] = []
        for ticker, ticker_rows in events.groupby("ticker", sort=True):
            ticker_text = str(ticker)
            expected_tickers.append(ticker_text)
            path = path_by_ticker.get(ticker_text)
            if path is None:
                raise FileNotFoundError(f"找不到Label來源CSV: ticker={ticker_text}")
            expected_indices = set(ticker_rows["_event_index"].astype(np.int64).tolist())
            shard_path = _ticker_shard_path(shard_dir, ticker_text)
            prior_shard = (
                _load_valid_ticker_shard(
                    shard_path, expected_event_indices=expected_indices
                )
                if bool(args.resume)
                else None
            )
            if prior_shard is not None:
                shard_frames[ticker_text] = prior_shard
                continue
            shard_path.unlink(missing_ok=True)
            jobs.append(
                {
                    "ticker": ticker_text,
                    "path": path,
                    "rows": ticker_rows[
                        ["_event_index", "date", "high_len", "group_index"]
                    ].to_dict("records"),
                    "historical_params_path": str(historical_path),
                    "forward_params_path": str(forward_path),
                }
            )

        completed = len(shard_frames)
        workers = max(1, int(args.workers))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {
                executor.submit(_ticker_worker, job): str(job["ticker"]) for job in jobs
            }
            for future in as_completed(future_to_ticker):
                ticker_text = future_to_ticker[future]
                frame = _write_ticker_shard(
                    _ticker_shard_path(shard_dir, ticker_text), future.result()
                )
                shard_frames[ticker_text] = frame
                completed += 1
                if not bool(args.quiet) and (
                    completed == len(expected_tickers) or completed % 25 == 0
                ):
                    print(
                        f"trade-path labels: {completed}/{len(expected_tickers)} tickers"
                    )

        missing_tickers = sorted(set(expected_tickers) - set(shard_frames))
        if missing_tickers:
            raise ValueError(
                "trade-path label ticker shards不完整: "
                f"missing={missing_tickers[:10]}"
            )
        updates = pd.concat(
            [shard_frames[ticker] for ticker in expected_tickers],
            ignore_index=True,
        )
        observed = set(
            pd.to_numeric(updates["_event_index"], errors="raise")
            .astype(np.int64)
            .tolist()
        )
        expected = set(events["_event_index"].astype(np.int64).tolist())
        if len(updates) != len(events) or observed != expected:
            missing = sorted(expected - observed)
            raise ValueError(
                "trade-path label更新未完整覆蓋events: "
                f"actual={len(updates)}, expected={len(events)}, sample_missing={missing[:10]}"
            )
        updates.sort_values("_event_index", kind="mergesort").to_csv(
            result_path, index=False, encoding="utf-8-sig"
        )

    updates = updates.set_index("_event_index").sort_index()
    if len(updates) != len(events):
        raise ValueError("trade-path event updates長度不一致")
    for column in updates.columns:
        events[column] = updates[column].to_numpy()
    events["label"] = pd.to_numeric(events["label"], errors="raise").astype(np.int8)
    events["label_status"] = events["label_status"].astype(str).str.strip()
    events["label_reason"] = events["label_reason"].astype(str).str.strip()
    events.drop(columns=["_event_index"], inplace=True)
    event_labels = events["label"].to_numpy(dtype=np.int8)
    label_status_counts = {
        str(key): int(value)
        for key, value in events["label_status"].value_counts(dropna=False).sort_index().items()
    }
    label_reason_counts = {
        str(key): int(value)
        for key, value in events["label_reason"].value_counts(dropna=False).sort_index().items()
    }

    shared_arrays = (
        "feature_bank",
        "event_context",
        "event_group_index",
        "group_anchor_prices",
        "future_high_prices",
        "future_low_prices",
        "future_available_bars",
        "future_date_ordinals",
    )
    for name in shared_arrays:
        _hardlink_or_copy(getattr(source_paths, name), getattr(target_paths, name))
    save_npy_atomic(target_paths.event_labels, event_labels)
    target_paths.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(target_paths.events, index=False, encoding="utf-8-sig")
    target_paths.legacy_dataset.unlink(missing_ok=True)

    policy = trade_path_label_policy_payload()
    summary = dict(base_summary)
    summary.update(
        {
            "filter_id": target_filter_id,
            "dataset_storage_schema_version": DATASET_STORAGE_SCHEMA_VERSION,
            "dataset_storage_format": DATASET_STORAGE_FORMAT,
            "policy": policy,
            "feature_cache_policy": {
                key: policy[key]
                for key in (
                    "feature_window_bars",
                    "label_path_cache_bars",
                    "high_len_values",
                    "high_len_min",
                    "high_len_max",
                    "benchmark_ticker",
                )
            },
            "label_policy": label_manifest_payload_from_policy_manifest(policy),
            "label_counts": label_counts(event_labels),
            "label_status_counts": label_status_counts,
            "label_reason_counts": label_reason_counts,
            "event_count": int(len(events)),
            "event_group_summary": event_group_summary(events, event_labels),
            "label_information_end_date_range": _date_range(
                events, "label_eval_end_date"
            ),
            "dataset_artifacts": _build_artifact_records(target_paths),
            "market_set_contract": None,
            "market_set_artifacts": None,
            "market_set_summary": None,
            "build_mode": "derived_feature_bank_trade_path_relabel",
            "trade_path_label": {
                "label_id": TRADE_PATH_LABEL_ID,
                "label_contract_version": TRADE_PATH_LABEL_CONTRACT_VERSION,
                "source_filter_id": source_filter_id,
                "source_dataset_summary_sha256": compute_file_sha256(source_paths.summary),
                "historical_teacher_path": project_relative_display_path(
                    historical_path, project_root=root
                ),
                "historical_teacher_sha256": compute_file_sha256(historical_path),
                "forward_teacher_path": project_relative_display_path(
                    forward_path, project_root=root
                ),
                "forward_teacher_sha256": compute_file_sha256(forward_path),
                "event_scope": "original_breakout_lifecycle",
                "initial_miss_buy_status": "pending",
                "filled_data_end_rule": "formal_single_stock_forced_closeout",
                "sizing_capital_rule": "same_explicit_single_stock_sizing_capital",
                "unfilled_terminal_rule": "exclude_from_binary_training",
            },
        }
    )
    write_json(target_paths.summary, summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    counts = dict(summary.get("label_counts") or {})
    status_counts = dict(summary.get("label_status_counts") or {})
    trade_path = dict(summary.get("trade_path_label") or {})
    print("\n" + render_title("A2 Realized Trade-path Label Dataset"))
    print(
        render_key_values(
            (
                ("Filter ID", summary.get("filter_id")),
                ("Label ID", trade_path.get("label_id")),
                ("Dataset", summary.get("dataset")),
                ("Events", summary.get("event_count")),
                ("PASS", status_counts.get("PASS", counts.get("pass"))),
                ("REJECT", status_counts.get("REJECT", counts.get("reject"))),
                ("EXCLUDED", status_counts.get("EXCLUDED", counts.get("invalid"))),
                ("Label End", summary.get("label_information_end_date_range")),
                ("Initial Miss Buy", trade_path.get("initial_miss_buy_status")),
                ("Filled Data End", trade_path.get("filled_data_end_rule")),
                ("Unfilled Terminal", trade_path.get("unfilled_terminal_rule")),
            )
        )
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    root = PROJECT_ROOT.resolve()
    if str(args.filter_id) != TRADE_PATH_FILTER_ID:
        raise ValueError(
            "trade-path研究filter_id固定為"
            f"{TRADE_PATH_FILTER_ID!r}，避免覆蓋正式9A工件"
        )
    if int(args.workers) < 1:
        raise ValueError("--workers必須>=1")
    historical_path = root / TRADE_PATH_HISTORICAL_TEACHER_PARAMS_RELATIVE_PATH
    forward_path = root / TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH
    print("\n" + render_title("A2 Realized Trade-path Label Build"))
    print(
        render_key_values(
            (
                ("Source Dataset", str(args.source_filter_id)),
                ("Derived Dataset", str(args.filter_id)),
                ("Historical Teacher", project_relative_display_path(historical_path, project_root=root)),
                ("Forward Teacher", project_relative_display_path(forward_path, project_root=root)),
                ("Workers", int(args.workers)),
                ("Resume", bool(args.resume)),
            )
        )
    )
    if bool(args.plan_only):
        print(render_section("Plan"))
        print("先建立Selection historical canonical Min ROOS no-DL teacher params，再合併既有forward teacher params，最後沿用feature bank建立新Label Dataset。")
        return 0
    historical_path = _ensure_historical_teacher_params(root, args)
    if not forward_path.is_file():
        raise FileNotFoundError(
            "缺少forward teacher params："
            f"{project_relative_display_path(forward_path, project_root=root)}；"
            "請先由正式策略參數適應流程建立目前config指定的forward teacher工件。"
        )
    summary = _build_dataset(root, args, historical_path, forward_path)
    summary["elapsed_sec"] = round(time.perf_counter() - started, 3)
    target_paths = resolve_dataset_paths(dataset_output_dir(str(args.filter_id)))
    write_json(target_paths.summary, summary)
    _print_summary(summary)
    print_artifact_paths(
        (
            ("Derived feature bank", target_paths.feature_bank),
            ("Trade-path labels", target_paths.event_labels),
            ("Events", target_paths.events),
            ("Summary", target_paths.summary),
        ),
        project_root=root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
