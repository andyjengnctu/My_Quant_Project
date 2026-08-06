"""Shared CLI helpers for breakout quality filter tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.data_utils import discover_unique_csv_inputs, sanitize_ohlcv_dataframe
from core.dataset_profiles import get_dataset_dir, normalize_dataset_profile_key
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    FEATURE_COLUMNS,
    LABEL_INVALID,
    LABEL_PASS,
    LABEL_REJECT,
    TRADE_PATH_LABEL_CONTRACT_VERSION,
    TRADE_PATH_LABEL_OBJECTIVE,
    TRADE_PATH_LABEL_STATUS_EXCLUDED,
    TRADE_PATH_LABEL_STATUS_PASS,
    TRADE_PATH_LABEL_STATUS_REJECT,
    BreakoutQualityLabelPolicy,
    label_manifest_payload_from_policy_manifest,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    IndexedFeatureBank,
    dataset_artifact_metadata_reasons,
    market_set_artifact_metadata_reasons,
    load_npy,
    resolve_dataset_paths,
)
from filters.breakout_quality.market_set import (
    IndexedMarketSetBank,
    MARKET_SET_FEATURE_COLUMNS,
    market_set_contract_payload,
)
from filters.breakout_quality.paths import (
    ensure_filter_model_dir,
    ensure_filter_output_dir,
    resolve_filter_artifact_paths,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_project_root_to_path() -> None:
    root_text = str(PROJECT_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _parse_high_len_values(raw_value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--high-lens 不可為空")
    values = tuple(sorted({int(token) for token in tokens}))
    if values[0] < 1:
        raise ValueError("--high-lens 只能包含正整數")
    return values


def build_policy_from_args(args) -> BreakoutQualityLabelPolicy:
    legacy_values = (args.high_len_min, args.high_len_max, args.high_len_step)
    if any(value is not None for value in legacy_values):
        if not all(value is not None for value in legacy_values):
            raise ValueError("使用舊式 high_len range 參數時，--high-len-min/--high-len-max/--high-len-step 必須同時提供")
        low = int(args.high_len_min)
        high = int(args.high_len_max)
        step = int(args.high_len_step)
        if low < 1 or high < low or step < 1:
            raise ValueError("high_len range 必須滿足 min>=1、max>=min、step>=1")
        high_len_values = tuple(range(low, high + 1, step))
    else:
        high_len_values = _parse_high_len_values(args.high_lens)

    return BreakoutQualityLabelPolicy(
        feature_window_bars=int(args.feature_window),
        label_horizon_bars=int(args.label_horizon),
        label_path_cache_bars=int(args.label_path_cache),
        high_len_values=high_len_values,
        min_mfe_return=float(args.min_mfe_return),
        min_reward_risk_ratio=float(args.min_reward_risk_ratio),
        max_adverse_return=float(args.max_adverse_return),
        benchmark_ticker=str(args.benchmark_ticker).strip(),
    )


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    p = DEFAULT_LABEL_POLICY
    parser.add_argument("--feature-window", type=int, default=p.feature_window_bars)
    parser.add_argument("--label-horizon", type=int, default=p.label_horizon_bars)
    parser.add_argument(
        "--label-path-cache",
        type=int,
        default=p.label_path_cache_bars,
        help="每個 ticker/date 保存的未來 K 線路徑長度；horizon 不超過此值時可快速 relabel",
    )
    parser.add_argument(
        "--high-lens",
        default=",".join(str(value) for value in p.high_lens()),
        help="逗號分隔的正式 high_len coverage；預設同時覆蓋 optimizer 搜尋值與策略預設值",
    )
    parser.add_argument("--high-len-min", type=int, default=None, help="相容舊命令；使用時需與 max/step 同時提供")
    parser.add_argument("--high-len-max", type=int, default=None, help="相容舊命令；使用時需與 min/step 同時提供")
    parser.add_argument("--high-len-step", type=int, default=None, help="相容舊命令；使用時需與 min/max 同時提供")
    parser.add_argument(
        "--min-mfe-return",
        "--pass-return-threshold",
        dest="min_mfe_return",
        type=float,
        default=p.min_mfe_return,
        help="PASS 要求的最低 MFE；最大漲幅必須嚴格大於此值（舊參數名稱仍可相容使用）",
    )
    parser.add_argument(
        "--min-reward-risk-ratio",
        type=float,
        default=p.min_reward_risk_ratio,
        help="PASS 要求的最低 MFE/MAE；必須 > 1，實際 ratio 也必須嚴格大於此值",
    )
    parser.add_argument(
        "--max-adverse-return",
        "--reject-return-threshold",
        dest="max_adverse_return",
        type=float,
        default=p.max_adverse_return,
        help="最大容許不利跌幅；Low 觸及或跌破即 REJECT（舊參數名稱仍可相容使用）",
    )
    parser.add_argument("--benchmark-ticker", default=p.benchmark_ticker)


def discover_dataset_csv_inputs(
    project_root: Path,
    dataset: str,
) -> tuple[list[tuple[str, Path]], list[str]]:
    profile = normalize_dataset_profile_key(dataset)
    data_dir = get_dataset_dir(str(project_root), profile)
    csv_inputs, duplicate_lines = discover_unique_csv_inputs(data_dir)
    normalized = [(str(ticker), Path(path)) for ticker, path in csv_inputs]
    return normalized, list(duplicate_lines or [])


def load_dataset_frame(path: Path, ticker: str, *, min_rows: int = 50) -> pd.DataFrame:
    raw_df = pd.read_csv(path, low_memory=False)
    frame, _stats = sanitize_ohlcv_dataframe(raw_df, ticker=ticker, min_rows=min_rows)
    return frame


def load_dataset_frames(project_root: Path, dataset: str, *, min_rows: int = 50) -> dict[str, pd.DataFrame]:
    csv_inputs, duplicate_lines = discover_dataset_csv_inputs(project_root, dataset)
    for line in duplicate_lines:
        print(line)
    frames: dict[str, pd.DataFrame] = {}
    for ticker, path in csv_inputs:
        try:
            frames[ticker] = load_dataset_frame(path, ticker, min_rows=min_rows)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            print(f"[skip] {ticker}: {exc}")
    return frames


def dataset_output_dir(filter_id: str) -> Path:
    return ensure_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)


def dataset_paths(filter_id: str):
    return resolve_dataset_paths(dataset_output_dir(filter_id))


def dataset_npz_path(filter_id: str) -> Path:
    """Legacy path retained only for cleanup and explicit incompatibility diagnostics."""

    return dataset_paths(filter_id).legacy_dataset


def events_csv_path(filter_id: str) -> Path:
    return dataset_paths(filter_id).events


def scores_csv_path(
    filter_id: str,
    *,
    experiment_profile: str | None = None,
) -> Path:
    return resolve_filter_artifact_paths(
        PROJECT_ROOT, filter_id, experiment_profile=experiment_profile
    ).score_path


def model_dir(
    filter_id: str,
    *,
    experiment_profile: str | None = None,
) -> Path:
    return ensure_filter_model_dir(
        PROJECT_ROOT, filter_id, experiment_profile=experiment_profile
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根節點必須是 object: {path}")
    return payload


def _validate_artifact_hash(artifact_name: str, artifact_path: Path, record: dict) -> None:
    expected_hash = str(record.get("sha256", "")).strip().lower()
    actual_hash = compute_file_sha256(artifact_path).lower()
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"dataset artifact SHA256 不一致: {artifact_name}; expected={expected_hash}, actual={actual_hash}"
        )


def _validate_future_paths(
    group_anchor_prices: np.ndarray,
    future_high_prices: np.ndarray,
    future_low_prices: np.ndarray,
    future_available_bars: np.ndarray,
) -> None:
    if np.isinf(group_anchor_prices).any() or np.isinf(future_high_prices).any() or np.isinf(future_low_prices).any():
        raise ValueError("future price cache 含 infinite")
    if np.any(~np.isfinite(group_anchor_prices)) or np.any(group_anchor_prices <= 0.0):
        raise ValueError("group_anchor_prices 含無效價格")
    path_bars = int(future_high_prices.shape[1])
    if np.any(future_available_bars < 0) or np.any(future_available_bars > path_bars):
        raise ValueError("future_available_bars 超出 future path cache 範圍")
    chunk_size = 4096
    for start in range(0, len(future_available_bars), chunk_size):
        stop = min(start + chunk_size, len(future_available_bars))
        for local_index, available in enumerate(future_available_bars[start:stop]):
            group_index = start + local_index
            count = int(available)
            if count <= 0:
                continue
            highs = future_high_prices[group_index, :count]
            lows = future_low_prices[group_index, :count]
            finite_pair = np.isfinite(highs) & np.isfinite(lows)
            if bool(np.any(finite_pair & (highs < lows))):
                raise ValueError(f"future price cache 含 high < low: group_index={group_index}")


def load_validated_dataset_bundle(
    filter_id: str,
    *,
    expected_policy: dict | None = None,
    require_current_source: bool = False,
) -> tuple[dict, IndexedFeatureBank, np.ndarray, np.ndarray, pd.DataFrame]:
    paths = dataset_paths(filter_id)
    if not paths.summary.is_file():
        raise FileNotFoundError(f"找不到 dataset_summary.json: {paths.summary}")
    summary = read_json(paths.summary)
    if str(summary.get("filter_id", "")).strip() != str(filter_id).strip():
        raise ValueError("dataset_summary.filter_id 與命令不一致")
    if int(summary.get("dataset_storage_schema_version", -1)) != DATASET_STORAGE_SCHEMA_VERSION:
        raise ValueError("dataset storage schema 不相容；請用新版 build-dataset 重建")
    if str(summary.get("dataset_storage_format") or "") != DATASET_STORAGE_FORMAT:
        raise ValueError("dataset storage format 不相容；請用新版 build-dataset 重建")
    if list(summary.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("dataset_summary feature_columns 與正式 contract 不一致")
    if list(summary.get("context_columns", [])) != list(CONTEXT_COLUMNS):
        raise ValueError("dataset_summary context_columns 與正式 contract 不一致")
    policy = summary.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("dataset_summary 缺少 policy object")
    expected_label_policy = label_manifest_payload_from_policy_manifest(policy)
    if summary.get("label_policy") != expected_label_policy:
        raise ValueError("dataset_summary label_policy 與正式 policy 不一致")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError(
            "目前 dataset policy 與 model manifest policy 不一致；不可用不同 feature/label/high_len 契約匯出同一模型分數"
        )

    if require_current_source:
        dataset_profile = str(summary.get("dataset") or "").strip().lower()
        stored_inventory = summary.get("source_data_inventory")
        if dataset_profile not in {"reduced", "full"} or not isinstance(stored_inventory, dict):
            raise ValueError("dataset 缺少有效 source_data_inventory；請先由 workflow 更新 dataset")
        current_inventory = build_source_data_inventory(PROJECT_ROOT, dataset_profile)
        if stored_inventory != current_inventory:
            raise ValueError(
                "來源 CSV 已更新，現有 breakout quality dataset 已過期；請先由 workflow 自動重建"
            )

    artifact_records = summary.get("dataset_artifacts")
    metadata_reasons = dataset_artifact_metadata_reasons(paths, artifact_records)
    if metadata_reasons:
        raise ValueError("dataset artifact metadata 不一致: " + "; ".join(metadata_reasons))
    assert isinstance(artifact_records, dict)
    for artifact_name, artifact_path in paths.artifact_paths().items():
        record = artifact_records[artifact_name]
        _validate_artifact_hash(artifact_name, artifact_path, record)

    feature_bank = load_npy(paths.feature_bank)
    context = load_npy(paths.event_context)
    labels = load_npy(paths.event_labels)
    event_group_index = load_npy(paths.event_group_index)
    group_anchor_prices = load_npy(paths.group_anchor_prices)
    future_high_prices = load_npy(paths.future_high_prices)
    future_low_prices = load_npy(paths.future_low_prices)
    future_available_bars = load_npy(paths.future_available_bars)
    future_date_ordinals = load_npy(paths.future_date_ordinals)
    events = read_breakout_quality_csv(paths.events)

    if feature_bank.ndim != 3 or context.ndim != 2 or labels.ndim != 1 or event_group_index.ndim != 1:
        raise ValueError(
            "dataset shape 不合法: "
            f"bank={feature_bank.shape}, C={context.shape}, y={labels.shape}, group_index={event_group_index.shape}"
        )
    if (
        group_anchor_prices.ndim != 1
        or future_high_prices.ndim != 2
        or future_low_prices.ndim != 2
        or future_available_bars.ndim != 1
        or future_date_ordinals.ndim != 2
    ):
        raise ValueError("future path cache shape 不合法")
    event_count = len(events)
    group_count = len(feature_bank)
    if len(context) != event_count or len(labels) != event_count or len(event_group_index) != event_count:
        raise ValueError(
            f"dataset event 長度不一致: C={len(context)}, y={len(labels)}, index={len(event_group_index)}, events={event_count}"
        )
    if (
        len(group_anchor_prices) != group_count
        or len(future_high_prices) != group_count
        or len(future_low_prices) != group_count
        or len(future_available_bars) != group_count
        or len(future_date_ordinals) != group_count
    ):
        raise ValueError("dataset group 長度不一致")
    if future_high_prices.shape != future_low_prices.shape or future_high_prices.shape != future_date_ordinals.shape:
        raise ValueError("future path cache shape 不一致")
    if group_count and (int(event_group_index.min()) < 0 or int(event_group_index.max()) >= group_count):
        raise ValueError("event_group_index 超出 feature bank 範圍")
    if feature_bank.shape[2] != len(FEATURE_COLUMNS) or context.shape[1] != len(CONTEXT_COLUMNS):
        raise ValueError("dataset feature/context 維度與正式 contract 不一致")
    feature_window_bars = int(policy.get("feature_window_bars", -1))
    path_cache_bars = int(policy.get("label_path_cache_bars", -1))
    if feature_window_bars < 1 or feature_bank.shape[1] != feature_window_bars:
        raise ValueError("dataset feature window 與 policy 不一致")
    if path_cache_bars < 1 or future_high_prices.shape[1] != path_cache_bars:
        raise ValueError("future path cache 長度與 policy 不一致")
    if not np.isfinite(feature_bank).all() or not np.isfinite(context).all():
        raise ValueError("dataset feature bank/context 含 NaN 或 infinite")
    _validate_future_paths(group_anchor_prices, future_high_prices, future_low_prices, future_available_bars)
    if np.any(future_date_ordinals < -1):
        raise ValueError("future_date_ordinals 含非法值")

    required_event_columns = {"ticker", "date", "high_len", "group_index", "label", "label_eval_end_date"}
    missing_event_columns = sorted(required_event_columns - set(events.columns))
    if missing_event_columns:
        raise ValueError(f"events.csv 缺少必要欄位: {missing_event_columns}")
    event_labels = pd.to_numeric(events["label"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(event_labels, np.asarray(labels, dtype=np.int64)):
        raise ValueError("events.csv label 與 event_labels.npy 不一致")
    observed_label_counts = label_counts(event_labels)
    if summary.get("label_counts") != observed_label_counts:
        raise ValueError("dataset_summary label_counts 與 event labels 不一致")
    if str(policy.get("label_objective") or "") == TRADE_PATH_LABEL_OBJECTIVE:
        from filters.breakout_quality.trade_path_label import TRADE_PATH_LABEL_REASON_STATUS

        trade_path_required_columns = {
            "label_status",
            "label_reason",
            "trade_path_fill_type",
            "trade_path_fill_date",
            "trade_path_entry_price",
            "trade_path_exit_date",
            "trade_path_exit_price",
            "trade_path_exit_reason",
            "trade_path_initial_missed_buy",
            "trade_path_forced_closeout",
            "trade_path_sizing_capital",
            "trade_path_realized_net_pnl",
            "trade_path_realized_net_r",
        }
        missing_trade_path_columns = sorted(trade_path_required_columns - set(events.columns))
        if missing_trade_path_columns:
            raise ValueError(
                "trade-path events.csv缺少必要欄位: "
                f"{missing_trade_path_columns}"
            )
        if int(policy.get("label_contract_version", -1)) != TRADE_PATH_LABEL_CONTRACT_VERSION:
            raise ValueError("trade-path label contract版本不相容；請重建Dataset")
        trade_path_meta = summary.get("trade_path_label")
        if not isinstance(trade_path_meta, dict) or int(
            trade_path_meta.get("label_contract_version", -1)
        ) != TRADE_PATH_LABEL_CONTRACT_VERSION:
            raise ValueError("dataset_summary.trade_path_label contract版本不一致")
        observed_status = events["label_status"].astype(str).str.strip()
        expected_status = pd.Series(
            np.where(
                event_labels == LABEL_PASS,
                TRADE_PATH_LABEL_STATUS_PASS,
                np.where(
                    event_labels == LABEL_REJECT,
                    TRADE_PATH_LABEL_STATUS_REJECT,
                    TRADE_PATH_LABEL_STATUS_EXCLUDED,
                ),
            ),
            index=events.index,
        )
        if not observed_status.equals(expected_status):
            raise ValueError("trade-path label_status與label數值不一致")
        observed_reason = events["label_reason"].astype(str).str.strip()
        expected_reason_status = observed_reason.map(TRADE_PATH_LABEL_REASON_STATUS)
        if expected_reason_status.isna().any():
            unknown = sorted(set(observed_reason[expected_reason_status.isna()].tolist()))
            raise ValueError(f"trade-path events含未知label_reason: {unknown}")
        if not expected_reason_status.equals(observed_status):
            raise ValueError("trade-path label_reason與label_status不一致")
        status_counts = {
            str(key): int(value)
            for key, value in observed_status.value_counts(dropna=False).sort_index().items()
        }
        reason_counts = {
            str(key): int(value)
            for key, value in observed_reason.value_counts(dropna=False).sort_index().items()
        }
        if summary.get("label_status_counts") != status_counts:
            raise ValueError("dataset_summary label_status_counts與events不一致")
        if summary.get("label_reason_counts") != reason_counts:
            raise ValueError("dataset_summary label_reason_counts與events不一致")
        realized_r = pd.to_numeric(events["trade_path_realized_net_r"], errors="coerce")
        realized_pnl = pd.to_numeric(events["trade_path_realized_net_pnl"], errors="coerce")
        completed_mask = observed_status.isin(
            {TRADE_PATH_LABEL_STATUS_PASS, TRADE_PATH_LABEL_STATUS_REJECT}
        )
        excluded_mask = observed_status == TRADE_PATH_LABEL_STATUS_EXCLUDED
        if realized_r[completed_mask].isna().any() or realized_pnl[completed_mask].isna().any():
            raise ValueError("trade-path PASS／REJECT必須有完整realized PnL與R")
        if (realized_r[observed_status == TRADE_PATH_LABEL_STATUS_PASS] <= 0.0).any():
            raise ValueError("trade-path PASS必須為正realized net R")
        if (realized_r[observed_status == TRADE_PATH_LABEL_STATUS_REJECT] > 0.0).any():
            raise ValueError("trade-path REJECT不得為正realized net R")
        if (realized_pnl[observed_status == TRADE_PATH_LABEL_STATUS_PASS] <= 0.0).any():
            raise ValueError("trade-path PASS必須同時為正realized net PnL")
        if (realized_pnl[observed_status == TRADE_PATH_LABEL_STATUS_REJECT] > 0.0).any():
            raise ValueError("trade-path REJECT不得為正realized net PnL")
        if realized_r[excluded_mask].notna().any() or realized_pnl[excluded_mask].notna().any():
            raise ValueError("trade-path EXCLUDED不得帶realized PnL或R")
        completed_required = (
            events.loc[completed_mask, [
                "trade_path_fill_type",
                "trade_path_fill_date",
                "trade_path_entry_price",
                "trade_path_exit_date",
                "trade_path_exit_price",
                "trade_path_exit_reason",
                "trade_path_sizing_capital",
            ]]
            .isna()
            .any(axis=1)
        )
        if bool(completed_required.any()):
            raise ValueError("trade-path PASS／REJECT缺少完整進出路徑欄位")
    csv_group_index = pd.to_numeric(events["group_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(csv_group_index, np.asarray(event_group_index, dtype=np.int64)):
        raise ValueError("events.csv group_index 與 event_group_index.npy 不一致")
    observed_group_indices = np.unique(csv_group_index)
    expected_group_indices = np.arange(group_count, dtype=np.int64)
    if not np.array_equal(observed_group_indices, expected_group_indices):
        raise ValueError(
            "events.csv 必須完整覆蓋連續 feature group indices；"
            f"observed={observed_group_indices.size}, expected={group_count}"
        )
    group_key_frame = events[["ticker", "date", "group_index"]].drop_duplicates()
    if group_key_frame.duplicated(["ticker", "date"], keep=False).any():
        raise ValueError("同一 ticker/date 不得對應多個 feature group_index")
    if group_key_frame.duplicated(["group_index"], keep=False).any():
        raise ValueError("同一 feature group_index 不得混用多個 ticker/date")
    if len(group_key_frame) != group_count:
        raise ValueError(
            "ticker/date 與 feature group_index 必須一對一完整覆蓋: "
            f"keys={len(group_key_frame)}, groups={group_count}"
        )
    policy_high_lens = policy.get("high_len_values")
    if not isinstance(policy_high_lens, list) or not policy_high_lens:
        raise ValueError("dataset policy.high_len_values 必須是非空 list")
    allowed_high_lens = {int(value) for value in policy_high_lens}
    observed_high_lens = set(pd.to_numeric(events["high_len"], errors="raise").astype(int).tolist())
    if not observed_high_lens.issubset(allowed_high_lens):
        raise ValueError("events.csv 含 policy 未宣告的 high_len")
    if int(summary.get("event_count", -1)) != event_count:
        raise ValueError("dataset_summary.event_count 與 artifacts 不一致")
    if int(summary.get("feature_group_count", -1)) != group_count:
        raise ValueError("dataset_summary.feature_group_count 與 artifacts 不一致")

    indexed_features = IndexedFeatureBank(feature_bank, event_group_index)
    return summary, indexed_features, context, labels, events


def load_validated_market_set_bank(
    filter_id: str,
    *,
    dataset_summary: dict | None = None,
    expected_model_spec=None,
) -> IndexedMarketSetBank:
    paths = dataset_paths(filter_id)
    summary = dataset_summary if dataset_summary is not None else read_json(paths.summary)
    if not isinstance(summary, dict):
        raise ValueError("market set dataset summary 必須是 object")
    expected_contract = market_set_contract_payload()
    if summary.get("market_set_contract") != expected_contract:
        raise ValueError(
            "dataset market_set_contract 與目前設定不一致；請完整重建 Dataset"
        )
    if expected_model_spec is not None:
        if not bool(getattr(expected_model_spec, "requires_market_set", False)):
            raise ValueError("只有 requires_market_set architecture 可載入 market set bank")
        if int(getattr(expected_model_spec, "market_set_history_bars", 0) or 0) != int(
            expected_contract["history_bars"]
        ):
            raise ValueError("model spec 與 dataset market history bars 不一致")
        if list(getattr(expected_model_spec, "market_set_base_features", ())) != list(
            MARKET_SET_FEATURE_COLUMNS
        ):
            raise ValueError("model spec 與 dataset market feature columns 不一致")
    artifact_records = summary.get("market_set_artifacts")
    metadata_reasons = market_set_artifact_metadata_reasons(paths, artifact_records)
    if metadata_reasons:
        raise ValueError("market set artifact metadata 不一致: " + "; ".join(metadata_reasons))
    assert isinstance(artifact_records, dict)
    for artifact_name, artifact_path in paths.market_set_artifact_paths().items():
        _validate_artifact_hash(artifact_name, artifact_path, artifact_records[artifact_name])

    daily_features = load_npy(paths.market_daily_features)
    daily_valid_mask = load_npy(paths.market_daily_valid_mask)
    market_date_ordinals = load_npy(paths.market_date_ordinals)
    group_market_date_index = load_npy(paths.group_market_date_index)
    tickers = read_breakout_quality_csv(paths.market_tickers)
    if list(tickers.columns) != ["market_ticker_index", "ticker"]:
        raise ValueError("market_tickers.csv 欄位契約不一致")
    expected_ticker_indices = np.arange(len(tickers), dtype=np.int64)
    observed_ticker_indices = pd.to_numeric(
        tickers["market_ticker_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(observed_ticker_indices, expected_ticker_indices):
        raise ValueError("market_tickers.csv index 必須連續且從 0 開始")
    if daily_features.shape[1] != len(tickers):
        raise ValueError("market daily feature ticker count 與 market_tickers.csv 不一致")
    if len(group_market_date_index) != int(summary.get("feature_group_count", -1)):
        raise ValueError("group_market_date_index 與 feature_group_count 不一致")
    return IndexedMarketSetBank(
        daily_features,
        daily_valid_mask,
        market_date_ordinals,
        group_market_date_index,
        history_bars=int(expected_contract["history_bars"]),
        min_valid_history_ratio=float(expected_contract["min_valid_history_ratio"]),
        max_stocks=int(expected_contract["max_stocks"]),
        max_dates_per_batch=int(
            getattr(expected_model_spec, "market_set_max_dates_per_batch", 0) or 4
        ),
    )


def label_counts(labels: Iterable[int]) -> dict[str, int]:
    arr = np.asarray(list(labels), dtype=np.int64)
    allowed = np.asarray([LABEL_INVALID, LABEL_REJECT, LABEL_PASS], dtype=np.int64)
    unexpected = np.unique(arr[~np.isin(arr, allowed)])
    if unexpected.size:
        raise ValueError(f"breakout quality labels 含未知值: {unexpected.tolist()}")
    return {
        "pass": int((arr == LABEL_PASS).sum()),
        "reject": int((arr == LABEL_REJECT).sum()),
        "invalid": int((arr == LABEL_INVALID).sum()),
        "total": int(arr.size),
    }


def event_group_keys(events: pd.DataFrame) -> pd.Series:
    missing = [col for col in ("ticker", "date") if col not in events.columns]
    if missing:
        raise KeyError(f"events.csv 缺少 group 欄位: {missing}")
    return events["ticker"].astype(str) + "\x1f" + events["date"].astype(str)


def group_size_weights(events: pd.DataFrame, indices: Iterable[int]) -> np.ndarray:
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return np.empty((0,), dtype=np.float32)
    keys = event_group_keys(events).iloc[idx].reset_index(drop=True)
    counts = keys.value_counts(sort=False)
    return keys.map(lambda key: 1.0 / float(counts[key])).to_numpy(dtype=np.float32)


def event_group_summary(events: pd.DataFrame, labels: Iterable[int]) -> dict:
    if events.empty:
        return {"group_count": 0}
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    if labels_arr.size != len(events):
        raise ValueError(f"labels 長度與 events 不一致: labels={labels_arr.size}, events={len(events)}")
    frame = events[["ticker", "date"]].copy()
    frame["label"] = labels_arr
    frame["_group_key"] = event_group_keys(events).to_numpy()
    group_sizes = frame.groupby("_group_key", sort=False).size()
    label_nunique = frame.groupby("_group_key", sort=False)["label"].nunique()
    valid = frame[frame["label"].isin([LABEL_REJECT, LABEL_PASS])].copy()
    valid_group_count = int(valid["_group_key"].nunique()) if not valid.empty else 0
    valid_label_nunique = (
        valid.groupby("_group_key", sort=False)["label"].nunique()
        if not valid.empty
        else pd.Series(dtype="int64")
    )
    return {
        "group_key": "ticker/date",
        "group_count": int(group_sizes.size),
        "valid_group_count": valid_group_count,
        "rows_per_group_min": int(group_sizes.min()) if group_sizes.size else 0,
        "rows_per_group_max": int(group_sizes.max()) if group_sizes.size else 0,
        "rows_per_group_mean": round(float(group_sizes.mean()), 6) if group_sizes.size else 0.0,
        "mixed_label_group_count": int((label_nunique > 1).sum()) if not label_nunique.empty else 0,
        "mixed_valid_label_group_count": int((valid_label_nunique > 1).sum()) if not valid_label_nunique.empty else 0,
    }


__all__ = [
    "PROJECT_ROOT",
    "add_policy_args",
    "build_policy_from_args",
    "dataset_npz_path",
    "dataset_output_dir",
    "dataset_paths",
    "discover_dataset_csv_inputs",
    "event_group_keys",
    "event_group_summary",
    "events_csv_path",
    "group_size_weights",
    "label_counts",
    "load_dataset_frame",
    "load_dataset_frames",
    "load_validated_dataset_bundle",
    "load_validated_market_set_bank",
    "model_dir",
    "read_breakout_quality_csv",
    "read_json",
    "scores_csv_path",
    "write_json",
]
