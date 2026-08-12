"""Build and audit the fixed 11A strategy-aligned continuous target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import (
    CONTINUOUS_TARGET_SCHEMA_VERSION,
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_ADVERSE_RETURN_FILENAME,
    TARGET_AUDIT_JSON_FILENAME,
    TARGET_AUDIT_MARKDOWN_FILENAME,
    TARGET_DAILY_CSV_FILENAME,
    TARGET_FAVORABLE_RETURN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    TARGET_OPPORTUNITY_BAR_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_RISK_BREACH_BAR_FILENAME,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    StrategyAlignedContinuousTargetSpec,
    build_strategy_aligned_group_targets,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import (
    DEFAULT_FILTER_ID,
    DEFAULT_LABEL_POLICY,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.dataset_store import load_npy, save_npy_atomic
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from core.console_report import (
    compact_console_enabled,
    print_artifact_paths,
    project_relative_display_path,
)
from filters.breakout_quality.workflow_io import (
    dataset_paths,
    load_validated_dataset_bundle,
)
from filters.breakout_quality.strategy_compare_engine import canonical_strategy_compare_output_dir_names
from filters.breakout_quality.trade_attribution import reconstruct_round_trips

from tools.audit.breakout_quality.artifact_primitives import continuous_target_component_paths as _artifact_paths
from tools.audit.breakout_quality.target_statistics import (
    collapse_group_frame as _collapse_group_frame,
    daily_rankability as _daily_rankability,
    distribution_metrics as _distribution_metrics,
    finite_values as _finite,
    format_metric as _fmt,
    format_percent as _pct,
    same_day_binary_concordance as _same_day_binary_concordance,
    source_data_end as _source_data_end,
    spearman as _spearman,
)

TARGET_AUDIT_SCHEMA_VERSION = 1
_SPLIT_ORDER = ("inner_train", "validation", "selection", "oos")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "建立11A固定連續target arrays，並稽核Selection／Validation／OOS分布、"
            "同日排序可學性與選填的實際Round-trip R方向；本命令不訓練模型"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--round-trips",
        default=None,
        help=(
            "選填 no_filter_round_trips.csv；未指定時會依序搜尋active 9A全部正式strategy_compare輸出，"
            "優先hard-filter標準目錄，再搜尋score-ranking變體"
        ),
    )
    parser.add_argument(
        "--trade-history",
        default=None,
        help=(
            "選填 no_filter_trades.csv；只在未指定--round-trips時使用，並以canonical交易歸因邏輯"
            "於記憶體重建round trips"
        ),
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="允許來源CSV inventory已更新時仍稽核既有Dataset；預設fail-fast",
    )
    return parser.parse_args(argv)


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_native(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )






















def _active_strategy_compare_root(filter_id: str) -> Path:
    return resolve_filter_model_output_dir(
        PROJECT_ROOT,
        filter_id,
        BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    )


def _active_strategy_compare_dir(filter_id: str) -> Path:
    """Return the canonical hard-filter strategy comparison directory."""

    return _active_strategy_compare_root(filter_id) / "strategy_compare"


def _strategy_compare_candidate_dirs(filter_id: str) -> list[Path]:
    """Resolve active 9A strategy outputs in deterministic semantic priority order."""

    root = _active_strategy_compare_root(filter_id)
    preferred_names = canonical_strategy_compare_output_dir_names()
    ordered = [root / name for name in preferred_names]
    if root.is_dir():
        ordered.extend(sorted(path for path in root.glob("strategy_compare*") if path.is_dir()))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in ordered:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _load_strategy_compare_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "strategy_comparison.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取strategy comparison metadata失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"strategy comparison metadata格式錯誤: {path}")
    return dict(metadata)


def _strategy_compare_dir_is_eligible(directory: Path, filter_id: str) -> tuple[bool, dict[str, Any]]:
    metadata = _load_strategy_compare_metadata(directory)
    if not metadata:
        return True, {}
    expected = {
        "filter_id": str(filter_id),
        "model_architecture": str(BREAKOUT_QUALITY_MODEL_ARCHITECTURE),
        "experiment_profile": str(BREAKOUT_QUALITY_EXPERIMENT_PROFILE),
    }
    for key, value in expected.items():
        observed = str(metadata.get(key) or "")
        if observed and observed != value:
            return False, metadata
    comparison_design = str(metadata.get("comparison_design") or "")
    if comparison_design and comparison_design != "historical_active_param_oos":
        return False, metadata
    return True, metadata


def _resolve_round_trip_path(filter_id: str, explicit_path: str | None) -> tuple[Path | None, str]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到round-trip檔案: {path}")
        return path, "explicit"
    for index, directory in enumerate(_strategy_compare_candidate_dirs(filter_id)):
        eligible, _ = _strategy_compare_dir_is_eligible(directory, filter_id)
        if not eligible:
            continue
        candidate = directory / "no_filter_round_trips.csv"
        if candidate.is_file():
            return candidate, (
                "active_9a_standard_path"
                if index == 0
                else "active_9a_strategy_compare_discovery"
            )
    return None, "not_found"


def _resolve_trade_history_path(
    filter_id: str,
    explicit_path: str | None = None,
) -> tuple[Path | None, str]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到trade history檔案: {path}")
        return path, "explicit_trade_history"
    for index, directory in enumerate(_strategy_compare_candidate_dirs(filter_id)):
        eligible, _ = _strategy_compare_dir_is_eligible(directory, filter_id)
        if not eligible:
            continue
        candidate = directory / "no_filter_trades.csv"
        if candidate.is_file():
            return candidate, (
                "active_9a_trade_history_fallback"
                if index == 0
                else "active_9a_strategy_compare_trade_history_discovery"
            )
    return None, "not_found"


def _load_round_trip_source(
    filter_id: str,
    explicit_round_trips: str | None,
    explicit_trade_history: str | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if explicit_round_trips:
        round_trip_path, round_trip_path_source = _resolve_round_trip_path(
            filter_id,
            explicit_round_trips,
        )
        assert round_trip_path is not None
        return pd.read_csv(round_trip_path, encoding="utf-8-sig"), {
            "path": str(round_trip_path),
            "path_source": round_trip_path_source,
            "source_kind": "round_trip_csv",
            "round_trips_reconstructed": False,
        }

    if explicit_trade_history:
        trade_history_path, trade_history_path_source = _resolve_trade_history_path(
            filter_id,
            explicit_trade_history,
        )
        assert trade_history_path is not None
        trade_history = pd.read_csv(trade_history_path, encoding="utf-8-sig")
        round_trips = reconstruct_round_trips(trade_history, scenario="no_filter")
        return round_trips, {
            "path": str(trade_history_path),
            "path_source": trade_history_path_source,
            "source_kind": "reconstructed_from_no_filter_trades",
            "round_trips_reconstructed": True,
        }

    attempted_paths: list[str] = []
    for index, directory in enumerate(_strategy_compare_candidate_dirs(filter_id)):
        eligible, metadata = _strategy_compare_dir_is_eligible(directory, filter_id)
        attempted_paths.extend(
            [
                str(directory / "no_filter_round_trips.csv"),
                str(directory / "no_filter_trades.csv"),
            ]
        )
        if not eligible:
            continue
        round_trip_path = directory / "no_filter_round_trips.csv"
        if round_trip_path.is_file():
            return pd.read_csv(round_trip_path, encoding="utf-8-sig"), {
                "path": str(round_trip_path),
                "path_source": (
                    "active_9a_standard_path"
                    if index == 0
                    else "active_9a_strategy_compare_discovery"
                ),
                "source_kind": "round_trip_csv",
                "round_trips_reconstructed": False,
                "strategy_compare_metadata": metadata,
            }
        trade_history_path = directory / "no_filter_trades.csv"
        if trade_history_path.is_file():
            trade_history = pd.read_csv(trade_history_path, encoding="utf-8-sig")
            round_trips = reconstruct_round_trips(trade_history, scenario="no_filter")
            return round_trips, {
                "path": str(trade_history_path),
                "path_source": (
                    "active_9a_trade_history_fallback"
                    if index == 0
                    else "active_9a_strategy_compare_trade_history_discovery"
                ),
                "source_kind": "reconstructed_from_no_filter_trades",
                "round_trips_reconstructed": True,
                "strategy_compare_metadata": metadata,
            }

    return None, {
        "path_source": "not_found",
        "source_kind": "not_found",
        "round_trips_reconstructed": False,
        "attempted_paths": attempted_paths,
    }


def _first_nonempty_date(row: pd.Series) -> tuple[str, str]:
    for column in ("quality_score_date", "signal_date", "candidate_date"):
        if column not in row.index:
            continue
        raw_value = row.get(column)
        if pd.isna(raw_value):
            continue
        value = str(raw_value).strip()
        if value:
            return pd.Timestamp(value).strftime("%Y-%m-%d"), column
    return "", ""


def _trade_alignment_diagnostic(
    group_frame: pd.DataFrame,
    *,
    filter_id: str,
    explicit_round_trips: str | None,
    explicit_trade_history: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trades, source = _load_round_trip_source(
        filter_id,
        explicit_round_trips,
        explicit_trade_history,
    )
    if trades is None:
        return {
            "available": False,
            "reason": "no_filter_round_trips.csv and no_filter_trades.csv not found",
            **source,
            "formula_tuned_from_trade_r": False,
            "diagnostic_only": True,
        }, pd.DataFrame()

    missing = sorted({"ticker", "r_multiple"} - set(trades.columns))
    if missing:
        raise ValueError(f"round-trip檔案缺少欄位: {missing}")
    lookup = group_frame[["ticker", "date", "target_raw_r"]].copy()
    lookup["ticker"] = lookup["ticker"].astype(str)
    lookup["target_date"] = lookup["date"].dt.strftime("%Y-%m-%d")
    lookup = lookup.drop(columns=["date"])
    if lookup.duplicated(["ticker", "target_date"], keep=False).any():
        raise ValueError("continuous target trade lookup的ticker/date必須唯一")

    matched = trades.copy()
    resolved = matched.apply(_first_nonempty_date, axis=1)
    matched["target_date"] = [item[0] for item in resolved]
    matched["target_date_source"] = [item[1] for item in resolved]
    matched["ticker"] = matched["ticker"].astype(str)
    matched["r_multiple"] = pd.to_numeric(matched["r_multiple"], errors="coerce")
    matched = matched.merge(lookup, how="left", on=["ticker", "target_date"], validate="many_to_one")
    valid = matched[
        np.isfinite(pd.to_numeric(matched["r_multiple"], errors="coerce"))
        & np.isfinite(pd.to_numeric(matched["target_raw_r"], errors="coerce"))
    ].copy()
    if valid.empty:
        return {
            "available": True,
            **source,
            "trade_count": int(len(matched)),
            "matched_trade_count": 0,
            "coverage_rate": 0.0,
            "formula_tuned_from_trade_r": False,
            "diagnostic_only": True,
        }, matched

    valid = valid.sort_values("target_raw_r", kind="mergesort").reset_index(drop=True)
    target_values = valid["target_raw_r"].to_numpy(dtype=np.float64)
    r_values = valid["r_multiple"].to_numpy(dtype=np.float64)
    top_count = max(1, int(math.ceil(len(valid) * 0.10)))
    bottom_count = top_count
    top = valid.tail(top_count)
    bottom = valid.head(bottom_count)
    positive_r_total = float(np.maximum(r_values, 0.0).sum())
    top_positive_r = float(np.maximum(top["r_multiple"].to_numpy(dtype=np.float64), 0.0).sum())
    large_winner_mask = r_values >= 2.0
    median_target = float(np.median(target_values))
    result = {
        "available": True,
        **source,
        "trade_count": int(len(matched)),
        "matched_trade_count": int(len(valid)),
        "coverage_rate": float(len(valid) / len(matched)) if len(matched) else None,
        "spearman_target_vs_r_multiple": _spearman(target_values, r_values),
        "target_positive_vs_trade_positive_accuracy": float(((target_values > 0.0) == (r_values > 0.0)).mean()),
        "all_trade_average_r": float(r_values.mean()),
        "top_target_decile_average_r": float(top["r_multiple"].mean()),
        "bottom_target_decile_average_r": float(bottom["r_multiple"].mean()),
        "top_target_decile_positive_r_share": (
            float(top_positive_r / positive_r_total) if positive_r_total > 0.0 else None
        ),
        "large_winner_count_r_ge_2": int(large_winner_mask.sum()),
        "large_winner_top_half_target_retention": (
            float((target_values[large_winner_mask] >= median_target).mean())
            if bool(large_winner_mask.any())
            else None
        ),
        "formula_tuned_from_trade_r": False,
        "diagnostic_only": True,
    }
    return result, matched


def build_continuous_target_audit(
    group_frame: pd.DataFrame,
    *,
    target_contract: dict[str, Any],
    split_report: dict[str, Any],
    dataset_summary: dict[str, Any],
    trade_alignment: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits: dict[str, Any] = {}
    daily_frames: list[pd.DataFrame] = []
    for split_name in _SPLIT_ORDER:
        subset = group_frame[group_frame[f"is_{split_name}"] & group_frame["valid_mask"]].copy()
        distribution = _distribution_metrics(subset)
        rankability, daily = _daily_rankability(subset, split_name=split_name)
        distribution["same_day_rankability"] = rankability
        distribution["same_day_binary_concordance"] = _same_day_binary_concordance(subset)
        splits[split_name] = distribution
        daily_frames.append(daily)

    all_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    payload = {
        "schema_version": TARGET_AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "11A Strategy-aligned Continuous Outcome Target",
        "status": "IMPLEMENTED_AUDIT_ONLY",
        "training_performed": False,
        "target_contract": target_contract,
        "dataset": {
            "filter_id": str(dataset_summary.get("filter_id") or DEFAULT_FILTER_ID),
            "dataset_profile": dataset_summary.get("dataset"),
            "event_count": int(dataset_summary.get("event_count", len(group_frame))),
            "feature_group_count": int(dataset_summary.get("feature_group_count", len(group_frame))),
            "source_data_date_range": dataset_summary.get("source_data_date_range"),
            "label_policy": dataset_summary.get("label_policy"),
        },
        "split_contract": split_report,
        "split_metrics": splits,
        "trade_r_alignment": trade_alignment,
        "interpretation_contract": {
            "selection_only_for_future_training_and_epoch_selection": True,
            "oos_statistics_are_iterative_research_evidence_only": True,
            "target_formula_uses_no_split_or_oos_statistics": True,
            "this_audit_does_not_authorize_training_or_runtime_deployment": True,
        },
    }
    return payload, all_daily






def render_continuous_target_audit_markdown(payload: dict[str, Any]) -> str:
    contract = payload["target_contract"]
    lines = [
        "# 11A Strategy-aligned Continuous Target Audit",
        "",
        f"- Target：`{contract['target_id']}`",
        f"- Horizon：`{contract['horizon_bars']}` trading bars",
        f"- Risk budget：`{float(contract['risk_budget_return']) * 100:.2f}%`",
        f"- Full-horizon time penalty：`{contract['full_horizon_time_penalty_r']:.4f}R`",
        "- 狀態：只完成target arrays與可學性稽核；本輪沒有訓練模型。",
        f"- 無前視邊界：target由各事件固定未來{int(contract['horizon_bars'])}交易日建立；公式不讀取Selection／Validation／OOS統計量。",
        "",
        "## 1. 固定公式",
        "",
        "```text",
        "target_raw_r",
        f"= 突破後、首次-{float(contract['risk_budget_return']) * 100:.2f}%風險觸發前的最大有利漲幅 ÷ {float(contract['risk_budget_return']) * 100:.2f}%風險預算",
        f"- 到達該高點前必須承受的最大不利跌幅 ÷ {float(contract['risk_budget_return']) * 100:.2f}%風險預算",
        f"- {float(contract['full_horizon_time_penalty_r']):.4f}R × 到達高點所用時間比例",
        "```",
        "",
        "同一日High／Low先後未知時採保守adverse-first；若Low先觸及風險障礙，該日High不計入可用機會。",
        "",
        "## 2. 分布與同日排序可學性",
        "",
        "| 區段 | Groups | Mean | P01 | P50 | P99 | 正值率 | Unique率 | Binary AUC | 同日可排序日 | Pair非Tie率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in _SPLIT_ORDER:
        metrics = payload["split_metrics"][split_name]
        q = metrics.get("quantiles") or {}
        rankability = metrics.get("same_day_rankability") or {}
        lines.append(
            f"| {split_name} | {int(metrics.get('group_count', 0)):,} "
            f"| {_fmt(metrics.get('mean'))} | {_fmt(q.get('p01'))} | {_fmt(q.get('p50'))} "
            f"| {_fmt(q.get('p99'))} | {_pct(metrics.get('positive_rate'))} "
            f"| {_pct(metrics.get('unique_value_ratio'))} | {_fmt(metrics.get('binary_label_auc'))} "
            f"| {_pct(rankability.get('rankable_date_rate'))} | {_pct(rankability.get('pairwise_non_tie_rate'))} |"
        )

    lines += [
        "",
        "## 3. 與既有二元Label的關係",
        "",
        "| 區段 | PASS target平均 | REJECT target平均 | 同日PASS/REJECT pair concordance | Top 1%正target貢獻 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in _SPLIT_ORDER:
        metrics = payload["split_metrics"][split_name]
        concordance = metrics.get("same_day_binary_concordance") or {}
        lines.append(
            f"| {split_name} | {_fmt(metrics.get('pass_target_mean'))} "
            f"| {_fmt(metrics.get('reject_target_mean'))} "
            f"| {_fmt(concordance.get('pair_weighted_concordance'))} "
            f"| {_pct(metrics.get('top_1pct_positive_sum_share'))} |"
        )

    trade = payload.get("trade_r_alignment") or {}
    lines += ["", "## 4. 實際Round-trip R方向診斷", ""]
    if not bool(trade.get("available")):
        lines.append(f"- 未取得可用no-filter round-trip檔案：`{trade.get('reason', 'not available')}`。")
        attempted_paths = list(trade.get("attempted_paths") or [])
        if attempted_paths:
            lines.append("- 已搜尋active 9A正式策略比較輸出：")
            for path in attempted_paths:
                lines.append(f"  - `{path}`")
        lines.append("- 這不影響target分布與同日可排序稽核，但尚不能確認target與實際交易R方向一致。")
    else:
        source_note = (
            "（由no_filter_trades.csv以canonical交易歸因邏輯重建）"
            if bool(trade.get("round_trips_reconstructed"))
            else ""
        )
        lines += [
            f"- 來源：`{trade.get('path')}`{source_note}",
            f"- 配對：`{trade.get('matched_trade_count', 0)}` / `{trade.get('trade_count', 0)}`；coverage `{_pct(trade.get('coverage_rate'))}`",
            f"- Spearman(target, realized R)：`{_fmt(trade.get('spearman_target_vs_r_multiple'))}`",
            f"- Top target decile平均R：`{_fmt(trade.get('top_target_decile_average_r'))}`；Bottom decile：`{_fmt(trade.get('bottom_target_decile_average_r'))}`",
            f"- ≥2R大贏家位於target上半部比例：`{_pct(trade.get('large_winner_top_half_target_retention'))}`",
            "- 本診斷不參與target公式、normalization、clip或任何參數擬合。",
        ]

    lines += [
        "",
        "## 5. 本輪邊界",
        "",
        "- 不建立新模型architecture，不訓練、不選epoch、不調threshold。",
        "- 不使用OOS統計定義target、normalization或loss。",
        "- 下一步是否進入11A regression training，必須依本稽核的target分布、tie／極端值與實際R方向結果再判定。",
        "",
    ]
    return "\n".join(lines)



def main(argv=None) -> int:
    args = parse_args(argv)
    if args.round_trips and args.trade_history:
        raise ValueError("--round-trips與--trade-history只能擇一指定")
    summary, indexed_features, _context, labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(args.allow_stale_source),
    )
    paths = dataset_paths(args.filter_id)
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    group_anchor_prices = load_npy(paths.group_anchor_prices)
    future_high_prices = load_npy(paths.future_high_prices)
    future_low_prices = load_npy(paths.future_low_prices)
    future_available_bars = load_npy(paths.future_available_bars)

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    targets = build_strategy_aligned_group_targets(
        group_anchor_prices,
        future_high_prices,
        future_low_prices,
        future_available_bars,
        spec=spec,
    )
    group_count = int(len(group_anchor_prices))
    valid_count = int(np.asarray(targets["valid_mask"], dtype=bool).sum())
    if valid_count == 0:
        raise ValueError("11A continuous target沒有任何有效group")

    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=_source_data_end(summary, events),
    )
    (
        split_assignments,
        inner_train_idx,
        validation_idx,
        selection_idx,
        oos_idx,
        split_report,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=bool(BREAKOUT_QUALITY_USE_INNER_VALIDATION),
        inner_validation_months=int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS),
        early_stopping_enabled=bool(
            BREAKOUT_QUALITY_USE_INNER_VALIDATION
            and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) > 0
        ),
    )
    del split_assignments
    split_group_indices = {
        "inner_train": np.unique(event_group_index[inner_train_idx]),
        "validation": np.unique(event_group_index[validation_idx]),
        "selection": np.unique(event_group_index[selection_idx]),
        "oos": np.unique(event_group_index[oos_idx]),
    }
    group_frame = _collapse_group_frame(events, event_group_index, targets, split_group_indices)
    trade_alignment, trade_matches = _trade_alignment_diagnostic(
        group_frame,
        filter_id=args.filter_id,
        explicit_round_trips=args.round_trips,
        explicit_trade_history=args.trade_history,
    )
    payload, daily_frame = build_continuous_target_audit(
        group_frame,
        target_contract=spec.contract_payload(),
        split_report=split_report,
        dataset_summary={**summary, "filter_id": args.filter_id},
        trade_alignment=trade_alignment,
    )

    target_dir = resolve_continuous_target_dir(PROJECT_ROOT, args.filter_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    array_paths = _artifact_paths(target_dir)
    for name, path in array_paths.items():
        save_npy_atomic(path, np.asarray(targets[name]))

    audit_json_path = target_dir / TARGET_AUDIT_JSON_FILENAME
    audit_markdown_path = target_dir / TARGET_AUDIT_MARKDOWN_FILENAME
    daily_csv_path = target_dir / TARGET_DAILY_CSV_FILENAME
    trade_matches_path = target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME
    _write_json(audit_json_path, payload)
    audit_markdown_path.write_text(
        render_continuous_target_audit_markdown(payload),
        encoding="utf-8",
    )
    daily_frame.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")
    if not trade_matches.empty:
        trade_matches.to_csv(trade_matches_path, index=False, encoding="utf-8-sig")
    elif trade_matches_path.exists():
        trade_matches_path.unlink()

    source_artifacts = summary.get("dataset_artifacts")
    manifest = {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": str(args.filter_id),
        "target_contract": spec.contract_payload(),
        "group_count": group_count,
        "valid_group_count": valid_count,
        "invalid_group_count": int(group_count - valid_count),
        "dataset_policy": summary.get("policy"),
        "dataset_artifact_source": {
            key: value
            for key, value in dict(source_artifacts or {}).items()
            if key in {
                "group_anchor_prices",
                "future_high_prices",
                "future_low_prices",
                "future_available_bars",
                "event_group_index",
                "events_csv",
            }
        },
        "split_policy": outer_policy,
        "split_report": split_report,
        "artifacts": {name: build_file_manifest(path) for name, path in array_paths.items()},
        "audit_outputs": {
            "json": build_file_manifest(audit_json_path),
            "markdown": build_file_manifest(audit_markdown_path),
            "daily_csv": build_file_manifest(daily_csv_path),
            **(
                {"trade_matches_csv": build_file_manifest(trade_matches_path)}
                if trade_matches_path.is_file()
                else {}
            ),
        },
        "training_performed": False,
        "runtime_eligible": False,
    }
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)

    if compact_console_enabled():
        print_artifact_paths(
            [("Target manifest", manifest_path), ("Markdown 報表", audit_markdown_path)],
            project_root=PROJECT_ROOT,
        )
        return 0

    print("11A continuous target audit完成")
    print(f"target={spec.target_id} groups={group_count:,} valid={valid_count:,}")
    for split_name in _SPLIT_ORDER:
        metrics = payload["split_metrics"][split_name]
        rankability = metrics.get("same_day_rankability") or {}
        print(
            f"- {split_name:<11} groups={int(metrics.get('group_count', 0)):,} "
            f"mean={_fmt(metrics.get('mean'))} binary_auc={_fmt(metrics.get('binary_label_auc'))} "
            f"rankable_dates={_pct(rankability.get('rankable_date_rate'))}"
        )
    if trade_alignment.get("available"):
        source_label = (
            "reconstructed_from_no_filter_trades"
            if bool(trade_alignment.get("round_trips_reconstructed"))
            else "round_trip_csv"
        )
        print(
            "- trade R: "
            f"source={source_label} "
            f"matched={trade_alignment.get('matched_trade_count')}/{trade_alignment.get('trade_count')} "
            f"spearman={_fmt(trade_alignment.get('spearman_target_vs_r_multiple'))}"
        )
        trade_path = trade_alignment.get("path")
        if trade_path and not compact_console_enabled():
            print(
                "  path="
                + project_relative_display_path(trade_path, project_root=PROJECT_ROOT)
            )
    else:
        attempted = list(trade_alignment.get("attempted_paths") or [])
        print("- trade R: active 9A全部strategy_compare輸出均未找到可用no-filter交易工件，略過實際R方向診斷")
        if attempted and not compact_console_enabled():
            print(f"  searched={len(attempted)} paths；可用 --round-trips 或 --trade-history 明確指定")
    print_artifact_paths(
        [("Target manifest", manifest_path), ("Markdown 報表", audit_markdown_path)],
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = [
    "TARGET_AUDIT_SCHEMA_VERSION",
    "build_continuous_target_audit",
    "main",
    "parse_args",
    "render_continuous_target_audit_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
