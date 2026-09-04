"""Strategy-level multiple-seed robustness for configured continuous DL arms.

The user-facing entry lives under Strategy Compare, while model fitting is still
performed by the canonical continuous-ranker trainer in an isolated subprocess.
Only aggregate seed metrics are persistent by default; per-seed model/score/replay
artifacts are temporary work products and are removed after their metrics are stored.
"""

from __future__ import annotations

import argparse
import gzip
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Lock
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_workflow_settings,
)
from core.training_policy import resolve_robustness_benchmark_seeds
from config.strategy_compare import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
)
from config.execution_policy import (
    DEFAULT_FIXED_RISK,
    DEFAULT_MAX_POSITION_CAP_PCT,
)
from core.file_integrity import (
    atomic_replace_with_retry,
    atomic_write_text,
    canonical_json_sha256,
    load_json_object_or_none,
)
from core.training_progress import (
    read_trainer_epoch_progress,
    read_trainer_pit_progress,
    render_training_unit_progress,
)
from core.training_scheduler import pop_next_seed_diverse_unit
from core.console_report import (
    console_color_enabled,
    paint,
    project_relative_display_path,
    render_title,
)
from core.display_common import FixedProgressBlock, InlineProgress, format_elapsed
from core.report_metrics import (
    CORE_STRATEGY_RESULT_METRICS,
    EXECUTION_STRATEGY_RESULT_METRICS,
    R_MODEL_PREDICTION_METRICS,
    R_SELECTION_TRANSLATION_METRICS,
    TRADE_RESULT_METRICS,
)
from core.strategy_param_artifacts import (
    resolve_strategy_param_benchmark_artifact_path,
    resolve_strategy_param_benchmark_manifest_path,
)
from core.strategy_comparison import (
    StrategyComparisonArm,
    resolve_strategy_comparison_arm_param_policy,
    strategy_comparison_param_artifact_key,
    strategy_comparison_param_binding_key,
)
from filters.breakout_quality.artifacts import build_file_manifest, compute_file_sha256
from core.strategy_param_artifacts import (
    STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
    compute_strategy_param_scientific_sha256,
)
from filters.breakout_quality.point_in_time_schedule import (
    build_point_in_time_fold_periods,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
)
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    resolve_strategy_param_evaluation_identity_sha256,
)
from filters.breakout_quality.strategy_result_state import (
    RESULT_ACTION_REBUILD_CONTEXT,
    RESULT_ACTION_REUSE,
    RESULT_ACTION_RUN,
    resolve_strategy_result_state,
)
from filters.breakout_quality.strategy_compare_contracts import COMPARISON_MODE_SCORE_RANKING
from services.research.strategy_compare_replay import (
    _load_reusable_no_filter_baseline,
)
from services.research.strategy_compare_execution import (
    run_strategy_compare_active_arm,
    run_strategy_compare_baseline_arm,
)
from filters.breakout_quality.strategy_compare_dl_artifacts import (
    resolve_arm_runtime_dl_source_ids,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from services.research.strategy_compare_training import (
    run_strategy_compare_training_unit,
    validate_strategy_compare_training_artifacts,
)
from services.research.training_process import (
    terminate_registered_training_processes,
)
from services.optimizer.strategy_param_service import (
    ensure_robustness_benchmark_strategy_parameter_artifact,
    inspect_robustness_benchmark_strategy_parameter_artifact,
)
from core.report_style import (
    signal_for_delta,
    styled_workflow_status,
    terminal_signal,
    finite_number as _finite_or_none,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from services.research.strategy_comparison import (
    _load_direct_selection_r,
    collect_artifact_status,
    render_strategy_execution_plan_surface,
)
from services.research.strategy_multi_seed_reporting import (
    _report_settings_for_contract,
    render_multi_seed_robustness_report,
)
from services.research.strategy_compare_reuse import (
    _find_reusable_baseline_source,
)
from filters.breakout_quality.strategy_compare_runtime import _arm_runtime_spec
from filters.breakout_quality.strategy_compare_reporting import capacity_summary
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from services.research.strategy_compare_preparation_status import (
    collect_preparation_status,
)
from services.research.strategy_compare_preparation import (
    prepare_strategy_parameter_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILENAME = "robustness_report.md"
SUMMARY_FILENAME = "robustness_summary.json"
SEED_RESULTS_FILENAME = "seed_results.csv"
SEED_YEARLY_RESULTS_FILENAME = "seed_yearly_returns.csv"
MANIFEST_FILENAME = "manifest.json"
SCIENTIFIC_OBSERVATIONS_MANIFEST_FILENAME = "scientific_observations_manifest.json"
LATEST_FILENAME = "latest.json"
ATTRIBUTION_SOURCE_DIRNAME = "attribution_source"
ATTRIBUTION_SOURCE_SCHEMA_VERSION = 2
SCIENTIFIC_OBSERVATIONS_SCHEMA_VERSION = 1
DURABLE_RESULT_FILENAMES = {
    "seed_results": SEED_RESULTS_FILENAME,
    "seed_yearly_results": SEED_YEARLY_RESULTS_FILENAME,
    "summary": SUMMARY_FILENAME,
    "report": REPORT_FILENAME,
}
SCIENTIFIC_DURABLE_RESULT_KEYS = ("seed_results", "seed_yearly_results")
ROBUSTNESS_SCHEMA_VERSION = 12
ROBUSTNESS_SCIENTIFIC_CONTRACT_VERSION = 6
ROBUSTNESS_MODEL_ARTIFACT_CONTRACT_VERSION = 2

MEAN_METRICS: tuple[tuple[str, str, str], ...] = (
    ("報酬", "total_return_pct", "%"),
    ("MDD", "max_drawdown_pct", "%"),
    ("RoMD", "return_over_max_drawdown", ""),
    ("年化", "annual_return_pct", "%"),
    ("EV", "expected_value_r", " R"),
    ("Payoff", "payoff_ratio", ""),
    ("曝險", "avg_exposure_pct", "%"),
    ("交易", "trade_count", ""),
    ("勝率", "win_rate_pct", "%"),
    ("月勝率", "monthly_win_rate_pct", "%"),
    ("Log R²", "log_r_squared", ""),
    ("DL選擇R", "direct_selection_r", " R"),
)

# ``MEAN_METRICS`` is the v8 scientific-observation compatibility set.  Keep it
# stable so completed runs remain reusable; the human report below is sourced from
# the canonical Strategy Compare metric registry instead of this legacy list.
COMMON_STRATEGY_REPORT_METRIC_KEYS = tuple(dict.fromkeys(
    metric.key
    for metric in (
        *CORE_STRATEGY_RESULT_METRICS,
        *TRADE_RESULT_METRICS,
        *EXECUTION_STRATEGY_RESULT_METRICS,
    )
))
MODEL_PREDICTION_METRIC_KEYS = tuple(metric.key for metric in R_MODEL_PREDICTION_METRICS)
SELECTION_TRANSLATION_METRIC_KEYS = tuple(metric.key for metric in R_SELECTION_TRANSLATION_METRICS)
ROBUSTNESS_OPTIONAL_SEED_METRIC_KEYS = tuple(dict.fromkeys((
    *COMMON_STRATEGY_REPORT_METRIC_KEYS,
    *MODEL_PREDICTION_METRIC_KEYS,
    *SELECTION_TRANSLATION_METRIC_KEYS,
    "continuous_target_id",
)))


def _read_json(path: Path) -> dict[str, Any]:
    payload = load_json_object_or_none(path, encoding="utf-8-sig")
    if payload is None:
        raise ValueError(f"JSON根節點必須是object或檔案無法讀取: {path}")
    return payload

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _atomic_write_csv(
    path: Path, frame: pd.DataFrame, *, sort_columns: tuple[str, ...]
) -> None:
    """Atomically publish one resumable CSV checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(list(sort_columns), kind="mergesort")
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        ordered.to_csv(temp, index=False, encoding="utf-8-sig")
        atomic_replace_with_retry(temp, path)
    except BaseException as exc:
        if temp.exists():
            try:
                temp.unlink()
            except OSError as cleanup_exc:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "atomic temp cleanup failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
        raise


def _mean_metric(frame: pd.DataFrame, key: str) -> float | None:
    if key not in frame.columns:
        return None
    values = pd.to_numeric(frame[key], errors="coerce").dropna()
    return None if values.empty else float(values.mean())




def _read_attribution_csv(unit_manifest: dict[str, Any], role: str) -> pd.DataFrame:
    item = dict(dict(unit_manifest.get("files") or {}).get(role) or {})
    relative = str(item.get("path") or "").strip()
    if not relative:
        return pd.DataFrame()
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("robustness attribution source path必須位於專案root內") from exc
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", compression="gzip")


def _backfill_seed_common_metrics_from_attribution(
    frame: pd.DataFrame,
    *,
    seed_yearly_frame: pd.DataFrame,
    run_root: Path | None,
    contract: dict[str, Any],
) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    if out.empty:
        return out
    for key in ROBUSTNESS_OPTIONAL_SEED_METRIC_KEYS:
        if key not in out.columns:
            out[key] = np.nan if key != "continuous_target_id" else ""

    for index, row in out.iterrows():
        arm_id = str(row["arm_id"])
        seed = int(row["seed"])
        yearly = seed_yearly_frame[
            (seed_yearly_frame.get("arm_id", pd.Series(dtype=str)).astype(str) == arm_id)
            & (pd.to_numeric(seed_yearly_frame.get("seed", pd.Series(dtype=float)), errors="coerce") == seed)
        ] if not seed_yearly_frame.empty else pd.DataFrame()
        if _finite_or_none(out.at[index, "min_full_year_return_pct"]) is None and not yearly.empty:
            complete = yearly.loc[yearly.get("is_complete_year", True).astype(bool)] if "is_complete_year" in yearly.columns else yearly
            values = pd.to_numeric(complete.get("return_pct"), errors="coerce").dropna()
            if not values.empty:
                out.at[index, "min_full_year_return_pct"] = float(values.min())

        if run_root is None:
            continue
        manifest = _read_attribution_unit_manifest(
            run_root,
            arm_id=arm_id,
            seed=seed,
            expected_fingerprint=str(contract["fingerprint"]),
        )
        if manifest is None:
            continue

        need_trade_r = any(
            _finite_or_none(out.at[index, key]) is None
            for key in ("portfolio_avg_r", "portfolio_median_r")
        )
        if need_trade_r:
            trades = _read_attribution_csv(manifest, "trades")
            if not trades.empty:
                closed = reconstruct_round_trips(trades, scenario=arm_id)
                r_values = pd.to_numeric(closed.get("r_multiple"), errors="coerce").dropna()
                if not r_values.empty:
                    if _finite_or_none(out.at[index, "portfolio_avg_r"]) is None:
                        out.at[index, "portfolio_avg_r"] = float(r_values.mean())
                    if _finite_or_none(out.at[index, "portfolio_median_r"]) is None:
                        out.at[index, "portfolio_median_r"] = float(r_values.median())

        capacity_keys = tuple(metric.key for metric in EXECUTION_STRATEGY_RESULT_METRICS if metric.key != "reserved_buy_fill_rate_pct")
        if any(_finite_or_none(out.at[index, key]) is None for key in capacity_keys):
            capacity = _read_attribution_csv(manifest, "daily_capacity")
            if not capacity.empty:
                summary = capacity_summary({"portfolio_capacity_rows": capacity.to_dict("records")})
                for key in capacity_keys:
                    if _finite_or_none(out.at[index, key]) is None and key in summary:
                        out.at[index, key] = summary[key]

        if _finite_or_none(out.at[index, "reserved_buy_fill_rate_pct"]) is None:
            execution = _read_attribution_csv(manifest, "execution")
            if not execution.empty and "entry_filled" in execution.columns:
                filled = execution["entry_filled"].astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
                valid = filled.dropna()
                if not valid.empty:
                    out.at[index, "reserved_buy_fill_rate_pct"] = float(valid.astype(bool).mean() * 100.0)
    return out


def _model_prediction_metrics_from_training_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    report_path = str(artifacts.get("report") or "").strip()
    if not report_path:
        return {}
    path = Path(report_path).resolve()
    if not path.is_file():
        return {}
    report = _read_json(path)
    split = dict((report.get("split_metrics") or {}).get("oos") or {})
    top = _finite_or_none(split.get("top_score_decile_raw_target_mean"))
    bottom = _finite_or_none(split.get("bottom_score_decile_raw_target_mean"))
    return {
        "continuous_target_id": str(report.get("continuous_target_id") or ""),
        "mean_daily_spearman": _finite_or_none(split.get("mean_daily_spearman")),
        "global_spearman": _finite_or_none(split.get("global_spearman_vs_raw_target")),
        "pairwise_concordance": _finite_or_none(split.get("pairwise_concordance")),
        "top_target_r": top,
        "bottom_target_r": bottom,
        "top_bottom_target_spread_r": None if top is None or bottom is None else float(top - bottom),
    }



def _reusable_fixed_scenario_metrics(settings, arm: StrategyComparisonArm) -> dict[str, Any]:
    status = collect_artifact_status(project_root=PROJECT_ROOT, settings=settings)
    reusable_dir = _find_reusable_baseline_source(
        root=PROJECT_ROOT, settings=settings, status=status, off_arm=arm
    )
    if reusable_dir is None:
        return {}
    payload = _read_json(reusable_dir / "strategy_comparison.json")
    return dict(payload.get("no_filter") or {})


def _build_common_report_payload(
    *,
    summary: dict[str, Any],
    seed_frame: pd.DataFrame,
    seed_yearly_frame: pd.DataFrame,
    run_root: Path | None,
) -> dict[str, Any]:
    contract = dict(summary.get("contract") or {})
    settings = get_strategy_comparison_settings(str(contract["profile_id"]))
    report_settings = _report_settings_for_contract(contract, settings)
    stochastic_ids = {
        str(dict(item or {}).get("arm_id") or "") for item in contract.get("stochastic_arms") or ()
    }
    enriched = _backfill_seed_common_metrics_from_attribution(
        seed_frame,
        seed_yearly_frame=seed_yearly_frame,
        run_root=run_root,
        contract=contract,
    )
    legacy_mean_by_id = {
        str(row.get("arm_id") or ""): dict(row)
        for row in summary.get("mean_strategy_metrics") or ()
    }
    yearly_stats = list(summary.get("yearly_statistics") or ())
    scenarios: dict[str, dict[str, Any]] = {}
    r_rows: list[dict[str, Any]] = []
    yearly_by_id: dict[str, dict[int, float | None]] = {}

    for arm in report_settings.enabled_arms:
        if arm.arm_id in stochastic_ids:
            rows = enriched[enriched["arm_id"].astype(str) == arm.arm_id]
            scenario = {key: _mean_metric(rows, key) for key in COMMON_STRATEGY_REPORT_METRIC_KEYS}
            direct_selection = _mean_metric(rows, "direct_selection_r")
            target_ids = [
                str(value).strip() for value in rows.get("continuous_target_id", pd.Series(dtype=str)).tolist()
                if str(value).strip() and str(value).lower() != "nan"
            ]
            target_id = target_ids[0] if target_ids and len(set(target_ids)) == 1 else ""
            r_row = {
                "arm_id": arm.arm_id,
                "name": arm.name,
                "continuous_target_id": target_id,
                "portfolio_avg_r": scenario.get("portfolio_avg_r"),
                "portfolio_median_r": scenario.get("portfolio_median_r"),
                "direct_selection_delta_r": direct_selection,
            }
            for key in (*MODEL_PREDICTION_METRIC_KEYS, *SELECTION_TRANSLATION_METRIC_KEYS):
                r_row[key] = _mean_metric(rows, key)
        else:
            scenario = dict(legacy_mean_by_id.get(arm.arm_id) or {})
            missing_common = [
                key for key in COMMON_STRATEGY_REPORT_METRIC_KEYS
                if _finite_or_none(scenario.get(key)) is None
            ]
            if missing_common:
                try:
                    reusable_metrics = _reusable_fixed_scenario_metrics(report_settings, arm)
                except (FileNotFoundError, ValueError, RuntimeError, OSError):
                    reusable_metrics = {}
                for key in missing_common:
                    if key in reusable_metrics:
                        scenario[key] = reusable_metrics.get(key)
            fixed_years = [row for row in yearly_stats if str(row.get("arm_id") or "") == arm.arm_id]
            complete_values = [
                _finite_or_none(row.get("mean")) for row in fixed_years if bool(row.get("is_complete_year", True))
            ]
            complete_values = [value for value in complete_values if value is not None]
            if _finite_or_none(scenario.get("min_full_year_return_pct")) is None and complete_values:
                scenario["min_full_year_return_pct"] = min(complete_values)
            r_row = {
                "arm_id": arm.arm_id,
                "name": arm.name,
                "continuous_target_id": "",
                "portfolio_avg_r": scenario.get("portfolio_avg_r"),
                "portfolio_median_r": scenario.get("portfolio_median_r"),
                "direct_selection_delta_r": None,
                **{key: None for key in (*MODEL_PREDICTION_METRIC_KEYS, *SELECTION_TRANSLATION_METRIC_KEYS)},
            }
        scenarios[arm.arm_id] = scenario
        r_rows.append(r_row)

        values: dict[int, float | None] = {}
        if arm.arm_id in stochastic_ids and not seed_yearly_frame.empty:
            arm_yearly = seed_yearly_frame[seed_yearly_frame["arm_id"].astype(str) == arm.arm_id]
            for year, group in arm_yearly.groupby("year", sort=True):
                values[int(year)] = _mean_metric(group, "return_pct")
        else:
            for row in yearly_stats:
                if str(row.get("arm_id") or "") == arm.arm_id:
                    values[int(row["year"])] = _finite_or_none(row.get("mean"))
        yearly_by_id[arm.arm_id] = values

    return {
        "scenarios": scenarios,
        "r_analysis": r_rows,
        "yearly_by_id": yearly_by_id,
        "seed_common_metric_coverage": {
            key: int(pd.to_numeric(enriched.get(key), errors="coerce").notna().sum())
            for key in COMMON_STRATEGY_REPORT_METRIC_KEYS
            if key in enriched.columns
        },
    }


def resolve_multi_seed_values(*, seed_count: int, generator_seed: int) -> tuple[int, ...]:
    """Compatibility facade over the benchmark seed SSOT in training_policy."""

    return tuple(
        resolve_robustness_benchmark_seeds(
            seed_count=int(seed_count),
            generator_seed=int(generator_seed),
        )
    )


def _robustness_arms(
    settings, robustness=None,
) -> tuple[tuple[StrategyComparisonArm, ...], tuple[StrategyComparisonArm, ...]]:
    if robustness is None:
        robustness = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    enabled_by_id = {arm.arm_id: arm for arm in settings.enabled_arms}
    try:
        fixed = tuple(enabled_by_id[arm_id] for arm_id in robustness.fixed_arm_ids)
        stochastic = tuple(enabled_by_id[arm_id] for arm_id in robustness.stochastic_arm_ids)
    except KeyError as exc:
        raise ValueError(f"multi-seed robustness引用未啟用或不存在的arm: {exc.args[0]}") from exc
    if not stochastic:
        raise ValueError("multi-seed robustness至少需要一個per-seed arm")
    if robustness.benchmark_id is None and not fixed:
        raise ValueError("legacy multi-seed robustness需要fixed baseline")
    return fixed, stochastic


def _model_seed_sensitive_arms(
    settings, robustness, benchmark_arms: tuple[StrategyComparisonArm, ...]
) -> tuple[StrategyComparisonArm, ...]:
    enabled = {arm.arm_id: arm for arm in benchmark_arms}
    try:
        return tuple(enabled[arm_id] for arm_id in robustness.model_seed_sensitive_arm_ids)
    except KeyError as exc:
        raise ValueError(
            f"robustness model-seed arm不在benchmark strategy arms: {exc.args[0]}"
        ) from exc


def _benchmark_param_binding(
    *, settings, robustness, arm: StrategyComparisonArm, seed: int
) -> dict[str, Any]:
    if robustness.benchmark_id is None:
        raise ValueError("historical robustness沒有benchmark parameter binding")
    source = settings.parameter_sources[str(arm.param_source)]
    family = str(source.canonical_family or "").strip()
    mode = str(source.canonical_evaluation_mode or "").strip()
    if not family or not mode:
        raise ValueError(
            f"benchmark arm必須使用canonical strategy parameter source: {arm.arm_id}/{arm.param_source}"
        )
    policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
    path = resolve_strategy_param_benchmark_artifact_path(
        PROJECT_ROOT,
        benchmark_id=str(robustness.benchmark_id),
        seed=int(seed),
        family=family,
        evaluation_mode=mode,
        policy=policy,
    )
    manifest_path = resolve_strategy_param_benchmark_manifest_path(
        PROJECT_ROOT,
        benchmark_id=str(robustness.benchmark_id),
        seed=int(seed),
        family=family,
        evaluation_mode=mode,
    )
    return {
        "arm_id": arm.arm_id,
        "seed": int(seed),
        "family": family,
        "evaluation_mode": mode,
        "param_policy": policy,
        "path": path,
        "manifest_path": manifest_path,
    }


def _benchmark_parameter_plan_rows(
    *, settings, robustness, benchmark_arms: tuple[StrategyComparisonArm, ...],
    comparison_end: str | None,
) -> tuple[list[tuple[str, str, str]], list[str], dict[tuple[str, int], dict[str, Any]]]:
    if robustness.benchmark_id is None:
        return [], [], {}
    rows: list[tuple[str, str, str]] = []
    blockers: list[str] = []
    bindings: dict[tuple[str, int], dict[str, Any]] = {}
    seen_units: set[tuple[int, str, str, str]] = set()
    for seed in robustness.resolved_seeds:
        for arm in benchmark_arms:
            binding = _benchmark_param_binding(
                settings=settings, robustness=robustness, arm=arm, seed=int(seed)
            )
            bindings[(arm.arm_id, int(seed))] = binding
            unit = (
                int(seed),
                str(binding["family"]),
                str(binding["evaluation_mode"]),
                str(binding["param_policy"]),
            )
            if unit in seen_units:
                continue
            seen_units.add(unit)
            path = Path(binding["path"])
            manifest_path = Path(binding["manifest_path"])
            label = (
                f"param-benchmark:{robustness.benchmark_id}:seed={seed}:"
                f"{binding['family']}:{binding['evaluation_mode']}:{binding['param_policy']}"
            )
            if comparison_end in (None, ""):
                if path.is_file() or manifest_path.is_file():
                    rows.append((
                        "CHECK", label,
                        "共同策略期間解析後由canonical Optimizer service驗證manifest／coverage／build contract；"
                        "不得因檔案存在直接視為REUSE",
                    ))
                else:
                    rows.append((
                        "BUILD", label,
                        "由canonical Optimizer以相同benchmark seed與統一trials/fold自動建立；"
                        "不得fallback到production Seed42策略參數",
                    ))
                continue
            state = inspect_robustness_benchmark_strategy_parameter_artifact(
                PROJECT_ROOT,
                benchmark_id=str(robustness.benchmark_id),
                seed=int(seed),
                family=str(binding["family"]),
                evaluation_mode=str(binding["evaluation_mode"]),
                policy=str(binding["param_policy"]),
                comparison_end_date=str(comparison_end),
                dataset=str(settings.dataset),
                max_positions=int(settings.max_positions),
                rotation=str(settings.rotation),
                fixed_risk=float(DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            )
            action = str(state["action"])
            if action == "REUSE":
                detail = project_relative_display_path(path, project_root=PROJECT_ROOT)
            elif action == "REPAIR":
                detail = (
                    "策略參數payload與Optimizer fitting contract相同；只快速修復benchmark publication manifest，"
                    "不重新執行fold optimizer"
                )
            elif action == "REBUILD":
                detail = (
                    "既有benchmark策略參數缺少／過期／scientific identity不符；由canonical Optimizer service自動REBUILD／RESUME"
                )
            else:
                detail = (
                    "由canonical Optimizer以相同benchmark seed與統一trials/fold自動建立；"
                    "不得fallback到production Seed42策略參數"
                )
            rows.append((action, label, detail))
    return rows, blockers, bindings



def _prepare_benchmark_strategy_parameter_artifacts(
    *, settings, robustness, benchmark_arms: tuple[StrategyComparisonArm, ...],
    comparison_start: str, comparison_end: str
) -> dict[tuple[str, int], dict[str, Any]]:
    if robustness.benchmark_id is None:
        return {}
    workflow = get_breakout_quality_workflow_settings()
    built_by_unit: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in robustness.resolved_seeds:
        for arm in benchmark_arms:
            binding = _benchmark_param_binding(
                settings=settings, robustness=robustness, arm=arm, seed=int(seed)
            )
            unit = (
                int(seed), str(binding["family"]), str(binding["evaluation_mode"]),
                str(binding["param_policy"]),
            )
            result = built_by_unit.get(unit)
            if result is None:
                result = ensure_robustness_benchmark_strategy_parameter_artifact(
                    PROJECT_ROOT,
                    benchmark_id=str(robustness.benchmark_id),
                    seed=int(seed),
                    family=str(binding["family"]),
                    evaluation_mode=str(binding["evaluation_mode"]),
                    policy=str(binding["param_policy"]),
                    comparison_end_date=str(comparison_end),
                    dataset=str(settings.dataset),
                    filter_id=str(workflow.filter_id),
                    model_architecture=str(workflow.model_architecture),
                    experiment_profile=str(workflow.experiment_profile),
                    max_positions=int(settings.max_positions),
                    rotation=str(settings.rotation),
                    fixed_risk=float(DEFAULT_FIXED_RISK),
                    max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
                    resume_parameter_training=True,
                    quiet=False,
                )
                built_by_unit[unit] = result
                print(
                    "[" + styled_workflow_status(str(result["action"])) + "]"
                    + f" seed={seed} {binding['family']}/{binding['evaluation_mode']} "
                    + f"{binding['param_policy']}"
                )
            result_path = Path(result["path"])
            bindings[(arm.arm_id, int(seed))] = {
                **binding,
                "path": result_path,
                "manifest_path": Path(result["manifest_path"]),
                # Full schedule identity remains the robustness scientific lineage.
                "sha256": str(result["scientific_sha256"]),
                # Replay/context reuse is evaluation-bound: OOS freezes in-memory,
                # Rolling consumes the full effective-date schedule.
                "evaluation_sha256": resolve_strategy_param_evaluation_identity_sha256(
                    result_path,
                    evaluation_mode=str(binding["evaluation_mode"]),
                    start_date=str(comparison_start),
                    end_date=str(comparison_end),
                ),
                "file_sha256": str(result["sha256"]),
                "manifest_sha256": compute_file_sha256(Path(result["manifest_path"])),
            }
    return bindings


def _benchmark_identity_payload(
    bindings: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    """Return parameter scientific identity plus non-scientific publication hashes.

    ``sha256`` is the stable runtime-effective schedule identity.  ``file_sha256``
    and ``manifest_sha256`` remain available only for provenance / exact-byte legacy
    migration; contract construction removes them before computing the scientific
    fingerprint.
    """

    return {
        f"{arm_id}:seed={seed}": {
            "identity_schema": STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
            "family": item["family"],
            "evaluation_mode": item["evaluation_mode"],
            "param_policy": item["param_policy"],
            "sha256": item["sha256"],
            "file_sha256": item.get("file_sha256"),
            "manifest_sha256": item.get("manifest_sha256"),
        }
        for (arm_id, seed), item in sorted(bindings.items())
    }


def _scientific_benchmark_parameter_identities(
    identities: dict[str, Any],
) -> dict[str, Any]:
    """Strip byte-level publication provenance from robustness scientific identity."""

    result: dict[str, Any] = {}
    for key, raw in dict(identities or {}).items():
        item = dict(raw or {})
        result[str(key)] = {
            field: item.get(field)
            for field in (
                "identity_schema", "family", "evaluation_mode", "param_policy", "sha256"
            )
            if item.get(field) not in (None, "")
        }
    return result


def _benchmark_parameter_publication_identities(
    identities: dict[str, Any],
) -> dict[str, Any]:
    """Return exact-byte provenance excluded from scientific fingerprinting."""

    result: dict[str, Any] = {}
    for key, raw in dict(identities or {}).items():
        item = dict(raw or {})
        publication = {
            field: item.get(field)
            for field in ("file_sha256", "manifest_sha256")
            if item.get(field) not in (None, "")
        }
        if publication:
            result[str(key)] = publication
    return result


def _strategy_only_benchmark_arms(
    benchmark_arms: tuple[StrategyComparisonArm, ...], model_arms: tuple[StrategyComparisonArm, ...]
) -> tuple[StrategyComparisonArm, ...]:
    model_ids = {arm.arm_id for arm in model_arms}
    return tuple(arm for arm in benchmark_arms if arm.arm_id not in model_ids)


def _resolve_same_seed_baseline_arm(
    *, settings, strategy_only_arms: tuple[StrategyComparisonArm, ...], model_arm: StrategyComparisonArm
) -> StrategyComparisonArm:
    model_policy = resolve_strategy_comparison_arm_param_policy(settings, model_arm)
    matches = [
        candidate for candidate in strategy_only_arms
        if candidate.param_source == model_arm.param_source
        and resolve_strategy_comparison_arm_param_policy(settings, candidate) == model_policy
        and candidate.rule_policy == model_arm.rule_policy
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "model benchmark arm無法唯一解析same-seed DL-off baseline: "
            f"arm={model_arm.arm_id}, matches={[item.arm_id for item in matches]}"
        )
    return matches[0]


def _strategy_only_baseline_context_available(
    *, output_dir: Path, settings, arm: StrategyComparisonArm, binding: dict[str, Any],
    comparison_start: str, comparison_end: str,
) -> bool:
    """Validate transient DL-off replay context before spending GPU time.

    Missing/corrupt retained context is a normal cache-miss signal.  The shared
    comparison loader wraps JSON/file read failures in ``RuntimeError`` for
    user-facing diagnostics, so unwrap only those known I/O/schema causes here;
    unexpected RuntimeError still propagates instead of being silently hidden.
    """

    all_off = arm.rule_policy == "all_off"
    try:
        _load_reusable_no_filter_baseline(
            output_dir,
            comparison_mode=COMPARISON_MODE_SCORE_RANKING,
            expected_dataset=str(settings.dataset),
            expected_params_sha256=str(binding["evaluation_sha256"]),
            expected_param_policy=resolve_strategy_comparison_arm_param_policy(settings, arm),
            expected_param_evaluation_mode=str(binding["evaluation_mode"]),
            expected_optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            expected_shared_param_overrides=(
                dict(ALL_RULE_FILTERS_OFF_OVERRIDES) if all_off else {}
            ),
            expected_max_positions=int(settings.max_positions),
            expected_enable_rotation=settings.rotation == "on",
            expected_start_date=str(comparison_start),
            expected_end_date=str(comparison_end),
        )
        return True
    except (
        OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError,
        TypeError, ValueError, pd.errors.ParserError,
    ):
        return False
    except RuntimeError as exc:
        if isinstance(exc.__cause__, (OSError, UnicodeDecodeError, json.JSONDecodeError)):
            return False
        raise


def _strategy_only_baseline_action(
    *, scientific_result_exists: bool, baseline_context_required: bool, baseline_context_available: bool
) -> str:
    """Choose whether a strategy-only benchmark unit needs replay work.

    Scientific seed results are durable.  The baseline replay directory is only a
    transient dependency for a model-arm replay, so a completed unit must not be
    rerun merely because normal retention cleanup removed that directory.
    """

    state = resolve_strategy_result_state(
        verified_evidence={} if scientific_result_exists else None,
        context_required=baseline_context_required,
        context_ready=baseline_context_available,
    )
    return {
        RESULT_ACTION_RUN: "RUN_SCIENTIFIC",
        RESULT_ACTION_REBUILD_CONTEXT: "REBUILD_CONTEXT",
        RESULT_ACTION_REUSE: "REUSE",
    }[state.action]


def _run_strategy_only_benchmark_unit(
    *, settings, arm: StrategyComparisonArm, seed: int, seed_order: int, arm_order: int,
    params_path: Path, comparison_start: str, comparison_end: str, output_dir: Path,
    strategy_param_sha256: str, strategy_param_manifest_sha256: str,
    param_evaluation_mode: str,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    started = time.perf_counter()
    payload = run_strategy_compare_baseline_arm(
        settings=settings,
        arm=arm,
        project_root=PROJECT_ROOT,
        params_path=str(params_path),
        output_dir=output_dir,
        comparison_start=str(comparison_start),
        comparison_end=str(comparison_end),
        quiet=True,
        param_evaluation_mode=str(param_evaluation_mode),
    )
    metrics = dict(payload.get("no_filter") or {})
    return {
        "arm_id": arm.arm_id,
        "name": arm.name,
        "seed": int(seed),
        "arm_order": int(arm_order),
        "seed_order": int(seed_order),
        "selected_epoch": 0,
        "fold_count": None,
        "training_elapsed_sec": 0.0,
        "replay_elapsed_sec": round(time.perf_counter() - started, 3),
        "model_sha256": None,
        "score_sha256": None,
        "runtime_source_identity_sha256": None,
        "strategy_param_sha256": str(strategy_param_sha256),
        "strategy_param_manifest_sha256": str(strategy_param_manifest_sha256),
        **{key: metrics.get(key) for _label, key, _unit in MEAN_METRICS},
        **{key: metrics.get(key) for key in COMMON_STRATEGY_REPORT_METRIC_KEYS},
        **{key: None for key in (*MODEL_PREDICTION_METRIC_KEYS, "continuous_target_id")},
        "yearly": _normalize_yearly_rows(
            payload.get("yearly"), arm_id=arm.arm_id, name=arm.name, seed=int(seed),
            seed_order=int(seed_order), arm_order=int(arm_order), result_side="no_filter",
        ),
        "baseline_dir": str(output_dir.resolve()),
    }

def _required_parameter_sources(
    fixed_arms: tuple[StrategyComparisonArm, ...],
    stochastic_arms: tuple[StrategyComparisonArm, ...],
) -> tuple[str, ...]:
    return tuple(sorted({arm.param_source for arm in (*fixed_arms, *stochastic_arms)}))


def _training_source_groups(
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    *,
    settings,
) -> tuple[tuple[str, tuple[tuple[int, StrategyComparisonArm], ...]], ...]:
    groups: dict[str, list[tuple[int, StrategyComparisonArm]]] = {}
    order: list[str] = []
    for arm_order, arm in enumerate(stochastic_arms, start=1):
        if not arm.dl_enabled or not str(arm.dl_id or "").strip():
            continue
        for dl_id in resolve_arm_runtime_dl_source_ids(settings, arm):
            if dl_id not in groups:
                groups[dl_id] = []
                order.append(dl_id)
            groups[dl_id].append((arm_order, arm))
    return tuple((dl_id, tuple(groups[dl_id])) for dl_id in order)


def _contract_stochastic_arms(
    contract: dict[str, Any], settings,
) -> tuple[StrategyComparisonArm, ...]:
    arm_ids = [str(dict(item or {}).get("arm_id") or "").strip() for item in contract.get("stochastic_arms") or ()]
    arms: list[StrategyComparisonArm] = []
    for arm_id in arm_ids:
        arm = settings.arms.get(arm_id)
        if arm is None:
            raise ValueError(f"robustness contract引用不存在的stochastic arm: {arm_id}")
        arms.append(arm)
    return tuple(arms)


def _contract_paired_contrasts(
    contract: dict[str, Any], settings, stochastic_arms: tuple[StrategyComparisonArm, ...],
) -> tuple[dict[str, Any], ...]:
    stochastic_by_id = {arm.arm_id: arm for arm in stochastic_arms}
    raw_specs = tuple(contract.get("paired_contrasts") or ())
    if not raw_specs and len(stochastic_arms) == 2:
        left, right = stochastic_arms
        raw_specs = ({
            "contrast_id": "legacy_default",
            "left": left.arm_id,
            "right": right.arm_id,
            "description": "歷史兩arm robustness預設同seed比較",
        },)
    resolved: list[dict[str, Any]] = []
    for raw in raw_specs:
        spec = dict(raw or {})
        left_id = str(spec.get("left") or "").strip()
        right_id = str(spec.get("right") or "").strip()
        left = stochastic_by_id.get(left_id)
        right = stochastic_by_id.get(right_id)
        if left is None or right is None:
            raise ValueError(
                "robustness contract paired contrast不屬於其stochastic arms: "
                f"left={left_id}, right={right_id}"
            )
        resolved.append({
            "contrast_id": str(spec.get("contrast_id") or f"{left_id}_vs_{right_id}"),
            "description": str(spec.get("description") or ""),
            "left": left,
            "right": right,
        })
    return tuple(resolved)


def _model_upstream_rows(status: dict[str, Any]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Render upstream work directly from the shared Strategy preparation plan."""

    plan = status.get("preparation_plan")
    if plan is None:
        raise RuntimeError("Strategy Compare status缺少preparation_plan")
    rows: list[tuple[str, str, str]] = []
    blockers: list[str] = []
    for item in plan.actions:
        if not str(item.artifact_key).startswith("model-upstream:"):
            continue
        rows.append((item.action, item.artifact_key, item.description))
        if item.action == "BLOCKED":
            blockers.append(f"{item.artifact_key}: {item.description}")
    return rows, blockers


def _comparison_period_from_status(status: dict[str, Any]) -> tuple[str, str]:
    """Read the period resolved by the shared Strategy preparation owner."""

    period = dict(status.get("comparison_period") or {})
    start = str(period.get("start") or "").strip()
    end = str(period.get("end") or "").strip()
    if not start or not end:
        raise RuntimeError(
            "Multiple-seed robustness尚無法由共用Strategy preparation解析共同策略期間"
        )
    return start, end

def _parameter_plan_rows(
    *, settings, status: dict[str, Any], required_sources: tuple[str, ...]
) -> tuple[list[tuple[str, str, str]], list[str]]:
    action_by_key = {
        item.artifact_key: item for item in status["preparation_plan"].actions
    }
    rows: list[tuple[str, str, str]] = []
    blockers: list[str] = []
    for source_id in required_sources:
        key = f"param:{source_id}"
        item = action_by_key.get(key)
        if item is None:
            blockers.append(f"前置計畫缺少{key}")
            rows.append(("BLOCKED", key, "正式前置計畫缺少此參數工件"))
            continue
        rows.append((item.action, item.artifact_key, item.description))
        if item.action == "BLOCKED":
            blockers.append(f"{item.artifact_key}: {item.description}")
    return rows, blockers


def _render_robustness_execution_plan(
    *, cfg, settings, status: dict[str, Any], fixed_arms, stochastic_arms
) -> tuple[str, dict[str, Any]]:
    # Current end-to-end benchmark repeats the complete Compare Suite per seed.
    # Canonical production params are not a robustness prerequisite; every current
    # arm resolves benchmark-only same-seed params, while model arms additionally
    # retrain their DL sources with that exact seed.
    if cfg.benchmark_id is not None:
        model_arms = _model_seed_sensitive_arms(settings, cfg, tuple(stochastic_arms))
        required_sources = ()
    else:
        model_arms = tuple(stochastic_arms)
        required_sources = _required_parameter_sources(tuple(fixed_arms), tuple(stochastic_arms))
    param_rows, param_blockers = _parameter_plan_rows(
        settings=settings, status=status, required_sources=required_sources
    )
    upstream_rows, upstream_blockers = _model_upstream_rows(status)
    blockers = [*param_blockers, *upstream_blockers]
    upstream_pending = any(row[0] in {"BUILD", "REBUILD", "RESUME"} for row in upstream_rows)
    if upstream_pending and not upstream_blockers:
        start = end = None
        period_text = "待自動補建canonical Dataset／Target後解析"
        period_error = None
    else:
        try:
            start, end = _comparison_period_from_status(status)
            period_text = f"{start} ～ {end}"
            period_error = None
        except (FileNotFoundError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            start = end = None
            period_text = f"BLOCKED: {type(exc).__name__}: {exc}"
            period_error = str(exc)
            blockers.append(str(exc))

    benchmark_rows, benchmark_blockers, benchmark_bindings = _benchmark_parameter_plan_rows(
        settings=settings, robustness=cfg, benchmark_arms=tuple(stochastic_arms),
        comparison_end=end,
    )
    blockers.extend(benchmark_blockers)
    pending_work = any(
        row[0] in {"BUILD", "REBUILD", "RESUME", "CHECK"}
        for row in (*param_rows, *benchmark_rows, *upstream_rows)
    )
    overall = "BLOCKED" if blockers else "PREPARABLE" if pending_work else "READY"
    rows = [*upstream_rows, *param_rows, *benchmark_rows]
    robustness_unit_states: dict[tuple[str, int], dict[str, Any]] = {}
    robustness_contract: dict[str, Any] | None = None
    if (
        overall == "READY"
        and cfg.benchmark_id is not None
        and start is not None
        and end is not None
    ):
        try:
            resolved_bindings: dict[tuple[str, int], dict[str, Any]] = {}
            for unit, item in benchmark_bindings.items():
                path = Path(item["path"])
                manifest_path = Path(item["manifest_path"])
                if not path.is_file() or not manifest_path.is_file():
                    raise FileNotFoundError(f"benchmark parameter artifact missing: {unit}")
                resolved_bindings[unit] = {
                    **item,
                    "sha256": compute_strategy_param_scientific_sha256(path),
                    "evaluation_sha256": resolve_strategy_param_evaluation_identity_sha256(
                        path,
                        evaluation_mode=str(item["evaluation_mode"]),
                        start_date=str(start),
                        end_date=str(end),
                    ),
                    "file_sha256": compute_file_sha256(path),
                    "manifest_sha256": compute_file_sha256(manifest_path),
                }
            robustness_contract = build_multi_seed_robustness_contract(
                robustness_id=cfg.robustness_id,
                comparison_period={"start": start, "end": end},
                artifact_identities=dict(status.get("artifact_identities") or {}),
                benchmark_parameter_identities=_benchmark_identity_payload(resolved_bindings),
            )
            _existing, _existing_yearly, robustness_unit_states = (
                _resolve_existing_scientific_unit_states(
                    run_root=_run_root(robustness_contract),
                    contract=robustness_contract,
                    stochastic_arms=tuple(stochastic_arms),
                    reuse_completed=bool(cfg.reuse_completed),
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            blockers.append(f"既有Robustness result state無法驗證: {type(exc).__name__}: {exc}")
            overall = "BLOCKED"
            robustness_unit_states = {}
            robustness_contract = None
    for arm in fixed_arms:
        rows.append((
            "RUN/REUSE",
            arm.name,
            "Legacy fixed reference",
        ))
    model_ids = {arm.arm_id for arm in model_arms}
    for arm in stochastic_arms:
        arm_states = [
            robustness_unit_states.get((arm.arm_id, int(seed)))
            for seed in cfg.resolved_seeds
        ]
        reuse_count = sum(
            1 for state in arm_states
            if isinstance(state, dict) and state.get("action") == RESULT_ACTION_REUSE
        )
        if arm_states and reuse_count == len(arm_states):
            action = "REUSE"
            description = f"{reuse_count}/{len(arm_states)} seed scientific results identity一致"
        elif reuse_count:
            action = "RESUME"
            description = (
                f"{reuse_count}/{len(arm_states)} seed結果REUSE；只執行其餘缺失單元"
            )
        elif arm.arm_id in model_ids:
            description = (
                f"{cfg.seed_count}個固定benchmark seeds；同seed strategy params + canonical DL trainer + replay"
            )
            action = "TRAIN+REPLAY"
        else:
            description = (
                f"{cfg.seed_count}個固定benchmark seeds；同seed strategy optimizer params + DL-off replay"
            )
            action = "PARAM+REPLAY"
        rows.append((action, arm.name, description))
    training_sources = _training_source_groups(tuple(model_arms), settings=settings)
    optimizer_families = tuple(sorted({
        str(settings.parameter_sources[str(arm.param_source)].canonical_family or "").strip()
        for arm in stochastic_arms
        if str(settings.parameter_sources[str(arm.param_source)].canonical_family or "").strip()
    })) if cfg.benchmark_id is not None else ()
    pit_workload_rows: list[tuple[str, str]] = []
    pit_training_sources = tuple(
        (dl_id, arm_entries)
        for dl_id, arm_entries in training_sources
        if settings.dl_sources[str(dl_id)].score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
    )
    if pit_training_sources and start is not None and end is not None:
        fold_counts = []
        for dl_id, _arm_entries in pit_training_sources:
            fold_counts.append(
                1
                if _source_point_in_time_single_score_block(settings=settings, dl_id=str(dl_id))
                else len(
                    build_point_in_time_fold_periods(
                        start,
                        end,
                        fold_months=_source_point_in_time_fold_months(
                            settings=settings, dl_id=str(dl_id)
                        ),
                        fold_anchor=_source_point_in_time_fold_anchor_date(
                            settings=settings, dl_id=str(dl_id)
                        ),
                        single_score_block=False,
                    )
                )
            )
        if fold_counts:
            unique_fold_counts = sorted(set(fold_counts))
            fold_text = str(unique_fold_counts[0]) if len(unique_fold_counts) == 1 else "/".join(map(str, unique_fold_counts))
            pit_workload_rows += [
                ("PIT folds", f"{fold_text} folds / DL source / seed"),
                (
                    "PIT fold工作量",
                    f"{cfg.seed_count * sum(fold_counts)} slots（實際新訓練會扣除REUSE／rescore／跨mode checkpoint重用）",
                ),
            ]
    metadata_rows = (
        ("整體狀態", overall),
        ("設定檔", "config/strategy_compare.py"),
        ("比較階段", f"{settings.profile_label} ({settings.profile_id})"),
        ("共同策略期間", period_text),
        ("Seed數量", cfg.seed_count),
        (
            "Seed generator",
            f"deterministic benchmark / generator_seed={cfg.seed_generator_seed}",
        ),
        ("GPU training", f"workers={cfg.gpu_train_workers}"),
        ("CPU strategy replay", f"workers={cfg.cpu_replay_workers}"),
        ("Benchmark ID", cfg.benchmark_id or "-"),
        ("Benchmark seeds", ",".join(str(v) for v in cfg.resolved_seeds)),
        (
            "Strategy trials/fold",
            cfg.strategy_trials_per_fold if cfg.strategy_trials_per_fold is not None else "-",
        ),
        (
            "策略Benchmark單元",
            f"{cfg.seed_count * len(stochastic_arms)}（{len(stochastic_arms)} per-seed strategy arms × {cfg.seed_count} seeds）",
        ),
        *(
            ((
                "策略Optimizer搜尋",
                f"{cfg.seed_count * len(optimizer_families)}（{len(optimizer_families)} families × {cfg.seed_count} seeds；Best/Agree共用search）",
            ),)
            if optimizer_families else ()
        ),
        (
            "模型訓練單元",
            f"{cfg.seed_count * len(training_sources)}（{len(training_sources)} unique DL sources × {cfg.seed_count} seeds）",
        ),
        *tuple(pit_workload_rows),
    )
    rendered_plan = render_strategy_execution_plan_surface(
        title=f"{cfg.label} 本次執行計畫",
        metadata_rows=metadata_rows,
        action_rows=rows,
    )
    return rendered_plan, {
        "overall_status": overall,
        "blockers": blockers,
        "required_param_sources": required_sources,
        "benchmark_param_bindings": benchmark_bindings,
        "robustness_unit_states": robustness_unit_states,
        "robustness_contract": robustness_contract,
        "comparison_period": None if start is None else {"start": start, "end": end},
        "period_error": period_error,
        "model_upstream_prepare_required": bool(upstream_pending),
    }


def _dataset_identity_snapshot(dataset: str) -> dict[str, Any]:
    try:
        return build_source_data_inventory(PROJECT_ROOT, dataset)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return {
            "status": "SOURCE_DATA_UNAVAILABLE",
            "dataset": str(dataset),
            "detail": str(exc),
        }


def _source_point_in_time_score_start_date(*, settings, dl_id: str) -> str | None:
    value = settings.dl_sources[str(dl_id)].point_in_time_score_start_date
    return None if value in (None, "") else str(value)


def _source_point_in_time_score_end_date(*, settings, dl_id: str) -> str | None:
    value = settings.dl_sources[str(dl_id)].point_in_time_score_end_date
    return None if value in (None, "") else str(value)


def _source_point_in_time_single_score_block(*, settings, dl_id: str) -> bool:
    return bool(settings.dl_sources[str(dl_id)].point_in_time_single_score_block)


def _source_point_in_time_fold_months(*, settings, dl_id: str) -> int:
    source = settings.dl_sources[str(dl_id)]
    if source.point_in_time_fold_months is not None:
        return int(source.point_in_time_fold_months)
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(source.experiment_profile)
    )
    return int(workflow.point_in_time_fold_months)


def _source_point_in_time_fold_anchor_date(*, settings, dl_id: str) -> str | None:
    source = settings.dl_sources[str(dl_id)]
    value = source.point_in_time_fold_anchor_date
    return None if value in (None, "") else str(value)


def _point_in_time_training_policy_snapshot(
    *, experiment_profile: str, fold_months: int | None = None, fold_anchor_date: str | None = None,
    single_score_block: bool = False, score_start_date: str | None = None, score_end_date: str | None = None
) -> dict[str, Any]:
    workflow = get_breakout_quality_workflow_settings(experiment_profile=str(experiment_profile))
    return {
        "fold_months": int(
            workflow.point_in_time_fold_months if fold_months is None else fold_months
        ),
        "fold_anchor_date": (None if fold_anchor_date in (None, "") else str(fold_anchor_date)),
        "single_score_block": bool(single_score_block),
        "score_start_date": None if score_start_date in (None, "") else str(score_start_date),
        "score_end_date": None if score_end_date in (None, "") else str(score_end_date),
        "inner_validation_months": int(workflow.point_in_time_inner_validation_months),
        "min_train_groups": int(workflow.point_in_time_min_train_groups),
        "min_validation_groups": int(workflow.point_in_time_min_validation_groups),
        "min_score_groups": int(workflow.point_in_time_min_score_groups),
        "resume": bool(workflow.point_in_time_resume),
    }


def _continuous_training_defaults_snapshot() -> dict[str, Any]:
    return {
        "epochs": int(BREAKOUT_QUALITY_DEFAULT_EPOCHS),
        "batch_size": int(BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE),
        "evaluation_batch_size": int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE),
        "train_prefetch_batches": int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES),
        "train_prefetch_workers": int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS),
        "learning_rate": float(BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE),
        "weight_decay": float(BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY),
        "gradient_clip_norm": float(BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM),
        "use_inner_validation": bool(BREAKOUT_QUALITY_USE_INNER_VALIDATION),
        "inner_validation_months": int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS),
        "early_stopping_patience": int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE),
        "early_stopping_min_delta": float(BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA),
        "final_refit_mode": str(BREAKOUT_QUALITY_FINAL_REFIT_MODE),
        "device": str(BREAKOUT_QUALITY_TORCH_DEVICE),
        "mixed_precision": bool(BREAKOUT_QUALITY_USE_MIXED_PRECISION),
        "mixed_precision_dtype": str(BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE),
        "deterministic_algorithms": bool(BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS),
        "allow_tf32": bool(BREAKOUT_QUALITY_ALLOW_TF32),
        "preload_feature_bank": bool(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK),
    }


def build_multi_seed_robustness_contract(
    *,
    robustness_id: str | None = None,
    comparison_period: dict[str, Any] | None = None,
    artifact_identities: dict[str, Any] | None = None,
    benchmark_parameter_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robustness = get_strategy_multi_seed_robustness_settings(robustness_id)
    settings = get_strategy_comparison_settings(robustness.profile_id)
    fixed, stochastic = _robustness_arms(settings, robustness)
    seeds = tuple(int(value) for value in robustness.resolved_seeds)
    resolved_period = dict(comparison_period or {})
    if not resolved_period:
        resolved_period = {"start": settings.start_date, "end": settings.end_date}
    identities = dict(artifact_identities or {})
    raw_benchmark_parameter_identities = dict(benchmark_parameter_identities or {})
    scientific_benchmark_parameter_identities = _scientific_benchmark_parameter_identities(
        raw_benchmark_parameter_identities
    )
    benchmark_parameter_publication_identities = _benchmark_parameter_publication_identities(
        raw_benchmark_parameter_identities
    )
    parameter_identities = {}
    for arm in (*fixed, *stochastic):
        param_policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
        binding_key = strategy_comparison_param_binding_key(arm.param_source, param_policy)
        raw_identity = identities.get(
            strategy_comparison_param_artifact_key(arm.param_source, param_policy)
        )
        if isinstance(raw_identity, dict):
            parameter_identities[binding_key] = {
                key: raw_identity.get(key)
                for key in ("sha256", "coverage_start", "coverage_end", "param_policy")
            }
    reference_baselines: dict[str, dict[str, Any]] = {}
    for reference_key, spec in robustness.romd_reference_baselines.items():
        expected_param_policy = str(spec.get("param_policy") or settings.param_policy).strip()
        reference_pool = stochastic if robustness.benchmark_id is not None else fixed
        matches = [
            arm for arm in reference_pool
            if arm.param_source == spec["param_source"]
            and resolve_strategy_comparison_arm_param_policy(settings, arm) == expected_param_policy
            and arm.rule_policy == spec["rule_policy"]
            and not arm.dl_enabled
        ]
        if len(matches) != 1:
            raise ValueError(
                "multi-seed RoMD reference無法唯一解析same-seed benchmark baseline: "
                f"reference={reference_key}, matches={len(matches)}"
            )
        matched = matches[0]
        reference_baselines[str(reference_key)] = {
            "arm_id": matched.arm_id, "name": matched.name,
            "param_source": matched.param_source,
            "param_policy": resolve_strategy_comparison_arm_param_policy(settings, matched),
            "rule_policy": matched.rule_policy,
        }
    scientific_reference_baselines = {
        key: {
            field: value
            for field, value in spec.items()
            if field != "name"
        }
        for key, spec in reference_baselines.items()
    }
    def source_contract(dl_id: str) -> dict[str, Any]:
        source = settings.dl_sources[str(dl_id)]
        return {
            "dl_id": source.dl_id,
            "filter_id": source.filter_id,
            "model_architecture": source.model_architecture,
            "experiment_profile": source.experiment_profile,
            "threshold": source.threshold,
            "score_source": source.score_source,
            "experiment_profile_manifest": get_breakout_quality_experiment_profile(
                source.experiment_profile
            ).as_manifest_payload(),
            "point_in_time_training_policy": (
                _point_in_time_training_policy_snapshot(
                    experiment_profile=source.experiment_profile,
                    fold_months=_source_point_in_time_fold_months(
                        settings=settings, dl_id=str(dl_id)
                    ),
                    fold_anchor_date=_source_point_in_time_fold_anchor_date(
                        settings=settings, dl_id=str(dl_id)
                    ),
                    single_score_block=_source_point_in_time_single_score_block(
                        settings=settings, dl_id=str(dl_id)
                    ),
                    score_start_date=_source_point_in_time_score_start_date(
                        settings=settings, dl_id=str(dl_id)
                    ),
                    score_end_date=_source_point_in_time_score_end_date(
                        settings=settings, dl_id=str(dl_id)
                    ),
                )
                if str(source.score_source) == "selection_point_in_time"
                else None
            ),
        }

    def benchmark_arm_contract(arm: StrategyComparisonArm) -> dict[str, Any]:
        payload = {
            "arm_id": arm.arm_id,
            "param_source": arm.param_source,
            "param_policy": resolve_strategy_comparison_arm_param_policy(settings, arm),
            "rule_policy": arm.rule_policy,
            "strategy_seed_sensitive": arm.arm_id in set(robustness.benchmark_strategy_arm_ids),
            "model_seed_sensitive": arm.arm_id in set(robustness.model_seed_sensitive_arm_ids),
        }
        if arm.dl_enabled and arm.dl_id:
            payload.update({
                "dl_id": arm.dl_id,
                "dl_runtime_mode": arm.dl_runtime_mode,
                "dl_runtime_options": dict(arm.dl_runtime_options or {}),
                "runtime_dl_sources": [source_contract(dl_id) for dl_id in resolve_arm_runtime_dl_source_ids(settings, arm)],
            })
        return payload

    scientific = {
        "scientific_contract_version": ROBUSTNESS_SCIENTIFIC_CONTRACT_VERSION,
        "robustness_id": robustness.robustness_id,
        "profile_id": settings.profile_id,
        "suite_id": settings.suite_id,
        "dataset": settings.dataset,
        "dataset_identity": _dataset_identity_snapshot(settings.dataset),
        "param_policy": "per-arm",
        "arm_param_policies": {
            arm.arm_id: resolve_strategy_comparison_arm_param_policy(settings, arm)
            for arm in (*fixed, *stochastic)
        },
        "max_positions": int(settings.max_positions),
        "rotation": settings.rotation,
        "comparison_period": resolved_period,
        "parameter_artifact_identities": parameter_identities,
        "benchmark_id": robustness.benchmark_id,
        "benchmark_parameter_artifact_identities": scientific_benchmark_parameter_identities,
        "seed_count": int(robustness.seed_count),
        "seed_generator_seed": int(robustness.seed_generator_seed),
        "resolved_seeds": list(seeds),
        "strategy_trials_per_fold": robustness.strategy_trials_per_fold,
        "seed_pairing": (
            "same_seed_strategy_optimizer_and_all_dl_sources"
            if robustness.benchmark_id is not None else "legacy_model_seed_only"
        ),
        "benchmark_strategy_arm_ids": list(robustness.benchmark_strategy_arm_ids),
        "model_seed_sensitive_arm_ids": list(robustness.model_seed_sensitive_arm_ids),
        "consensus_reference_arm_ids": list(robustness.consensus_reference_arm_ids),
        "training_defaults": _continuous_training_defaults_snapshot(),
        "romd_reference_baselines": scientific_reference_baselines,
        "fixed_arms": [
            {
                "arm_id": arm.arm_id,
                "param_source": arm.param_source,
                "param_policy": resolve_strategy_comparison_arm_param_policy(settings, arm),
                "rule_policy": arm.rule_policy,
            }
            for arm in fixed
        ],
        "stochastic_arms": [
            benchmark_arm_contract(arm) for arm in stochastic
        ],
    }
    model_artifact_identity = _model_artifact_identity_payload(scientific)
    contract = {
        **scientific,
        "benchmark_parameter_publication_identities": benchmark_parameter_publication_identities,
        "model_artifact_fingerprint": canonical_json_sha256(
            model_artifact_identity, length=16
        ),
        "romd_reference_baselines": reference_baselines,
        "paired_contrasts": [dict(item) for item in robustness.paired_contrasts],
        "report_schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "label": robustness.label,
        "execution_options": {
            "gpu_train_workers": int(robustness.gpu_train_workers),
            "cpu_replay_workers": int(robustness.cpu_replay_workers),
            "console_mode": robustness.console_mode,
            "progress_interval_seconds": float(robustness.progress_interval_seconds),
            "yearly_report": bool(robustness.yearly_report),
            "checkpoint_cache_root": robustness.checkpoint_cache_root,
        },
        "retention": {
            "keep_checkpoints": bool(robustness.keep_checkpoints),
            "keep_scores": bool(robustness.keep_scores),
            "keep_replay_details": bool(robustness.keep_replay_details),
            "keep_attribution_source": bool(robustness.keep_attribution_source),
        },
    }
    # 只有scientific identity改變才換fingerprint；console/report/parallelism/retention不觸發重訓。
    contract["fingerprint"] = canonical_json_sha256(scientific, length=16)
    return contract


def _model_artifact_identity_payload(scientific: dict[str, Any]) -> dict[str, Any]:
    """Return the DL fitting/score identity without strategy-parameter bindings.

    Strategy params affect C59/C60 replay results, not E/K/M model fitting.  Keep
    model work reusable across parameter-only robustness fingerprints whenever
    retention has preserved the required model/score artifacts.
    """

    model_sources: dict[str, dict[str, Any]] = {}
    for raw_arm in tuple(scientific.get("stochastic_arms") or ()):
        arm = dict(raw_arm or {})
        if not bool(arm.get("model_seed_sensitive")):
            continue
        for raw_source in tuple(arm.get("runtime_dl_sources") or ()):
            source = dict(raw_source or {})
            dl_id = str(source.get("dl_id") or "").strip()
            if dl_id:
                model_sources[dl_id] = source
    return {
        "model_artifact_contract_version": ROBUSTNESS_MODEL_ARTIFACT_CONTRACT_VERSION,
        "robustness_id": scientific.get("robustness_id"),
        "profile_id": scientific.get("profile_id"),
        "dataset": scientific.get("dataset"),
        "dataset_identity": scientific.get("dataset_identity"),
        "comparison_period": scientific.get("comparison_period"),
        "benchmark_id": scientific.get("benchmark_id"),
        # Seed membership belongs to the robustness result contract, not to the
        # model-artifact namespace. Each model child directory is already keyed
        # by the concrete seed, so expanding N=2 -> N=4 must not invalidate the
        # already fitted artifacts for seeds 1-2.
        "training_defaults": scientific.get("training_defaults"),
        "model_sources": [model_sources[key] for key in sorted(model_sources)],
    }


def _run_root(contract: dict[str, Any]) -> Path:
    cfg = get_strategy_multi_seed_robustness_settings(str(contract["robustness_id"]))
    return (PROJECT_ROOT / cfg.output_root / str(contract["fingerprint"])).resolve()


def _model_work_root(contract: dict[str, Any]) -> Path:
    cfg = get_strategy_multi_seed_robustness_settings(str(contract["robustness_id"]))
    model_fingerprint = str(
        contract.get("model_artifact_fingerprint") or contract["fingerprint"]
    )
    return (PROJECT_ROOT / cfg.model_work_root / model_fingerprint).resolve()


def _unit_key(arm_id: str, seed: int) -> str:
    return f"{arm_id}__seed_{int(seed)}"



def _attribution_source_root(run_root: Path) -> Path:
    return Path(run_root).resolve() / ATTRIBUTION_SOURCE_DIRNAME


def _attribution_unit_dir(run_root: Path, arm_id: str, seed: int) -> Path:
    return _attribution_source_root(run_root) / _unit_key(arm_id, seed)


def _gzip_copy_deterministic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            shutil.copyfileobj(src, gz)


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _write_compact_attribution_source(
    *,
    pair_dir: Path,
    destination_dir: Path,
    job: dict[str, Any],
    arm: StrategyComparisonArm,
    dl,
    runtime_spec: dict[str, str],
) -> dict[str, Any]:
    prefix = str(runtime_spec["active_key"])
    source_files = {
        "trades": pair_dir / f"{prefix}_trades.csv",
        "equity": pair_dir / f"{prefix}_equity.csv",
        "daily_capacity": pair_dir / f"{prefix}_daily_capacity.csv",
        "selected_buys": pair_dir / f"{prefix}_selected_buys.csv",
        "execution": pair_dir / f"{prefix}_execution.csv",
    }
    missing = [path.name for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "compact attribution source缺少replay工件: " + ", ".join(missing)
        )
    destination_dir = Path(destination_dir).resolve()
    try:
        destination_dir.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("compact attribution destination必須位於專案root內") from exc
    shutil.rmtree(destination_dir, ignore_errors=True)
    destination_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for role, source in source_files.items():
        destination = destination_dir / f"{role}.csv.gz"
        _gzip_copy_deterministic(source, destination)
        files[role] = {
            "path": project_relative_display_path(destination, project_root=PROJECT_ROOT),
            "source_filename": source.name,
            "source_sha256": compute_file_sha256(source),
            "sha256": compute_file_sha256(destination),
            "row_count": _csv_row_count(source),
        }
    payload = {
        "schema_version": ATTRIBUTION_SOURCE_SCHEMA_VERSION,
        "scientific_fingerprint": str(job["scientific_fingerprint"]),
        "robustness_id": str(job["robustness_id"]),
        "profile_id": str(job["profile_id"]),
        "arm_id": arm.arm_id,
        "arm_name": arm.name,
        "arm_order": int(job["arm_order"]),
        "seed": int(job["seed"]),
        "seed_order": int(job["seed_order"]),
        "comparison_period": {
            "start": str(job["comparison_start"]),
            "end": str(job["comparison_end"]),
        },
        "param_source": arm.param_source,
        "rule_policy": arm.rule_policy,
        "dl_runtime_mode": arm.dl_runtime_mode,
        "dl_runtime_options": dict(arm.dl_runtime_options or {}),
        "dl_source": {
            "dl_id": dl.dl_id,
            "filter_id": dl.filter_id,
            "model_architecture": dl.model_architecture,
            "experiment_profile": dl.experiment_profile,
            "score_source": dl.score_source,
            "threshold": dl.threshold,
        },
        "training_identity": {
            "selected_epoch": int(job.get("selected_epoch", 0) or 0),
            "fold_count": job.get("fold_count"),
            "model_sha256": str(job.get("model_sha256") or "") or None,
            "score_sha256": str(job.get("score_sha256") or "") or None,
            "runtime_source_identity_sha256": str(job.get("runtime_source_identity_sha256") or "") or None,
        },
        "runtime_training_identities": dict(job.get("runtime_source_identities") or {}),
        "scientific_observation_validation": {
            "status": "PENDING",
            "verified_at_utc": None,
            "source": None,
        },
        "files": files,
    }
    _write_json(destination_dir / MANIFEST_FILENAME, payload)
    return payload


def _read_attribution_unit_manifest(
    run_root: Path,
    *,
    arm_id: str,
    seed: int,
    expected_fingerprint: str,
    require_verified: bool = True,
) -> dict[str, Any] | None:
    unit_dir = _attribution_unit_dir(run_root, arm_id, seed)
    path = unit_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
        if int(payload.get("schema_version") or 0) != ATTRIBUTION_SOURCE_SCHEMA_VERSION:
            return None
        if str(payload.get("scientific_fingerprint") or "") != str(expected_fingerprint):
            return None
        if str(payload.get("arm_id") or "") != str(arm_id) or int(payload.get("seed", -1)) != int(seed):
            return None
        validation = dict(payload.get("scientific_observation_validation") or {})
        if require_verified and str(validation.get("status") or "") != "VERIFIED":
            return None
        files = dict(payload.get("files") or {})
        for role in ("trades", "equity", "daily_capacity", "selected_buys", "execution"):
            item = dict(files.get(role) or {})
            relative = str(item.get("path") or "")
            if not relative:
                return None
            file_path = (PROJECT_ROOT / relative).resolve()
            try:
                file_path.relative_to(PROJECT_ROOT)
            except ValueError:
                return None
            if not file_path.is_file():
                return None
            expected_sha = str(item.get("sha256") or "").strip().lower()
            if expected_sha and compute_file_sha256(file_path).lower() != expected_sha:
                return None
        return payload
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _mark_attribution_unit_verified(
    run_root: Path,
    *,
    arm_id: str,
    seed: int,
    expected_fingerprint: str,
    source: str,
) -> None:
    payload = _read_attribution_unit_manifest(
        run_root,
        arm_id=arm_id,
        seed=seed,
        expected_fingerprint=expected_fingerprint,
        require_verified=False,
    )
    if payload is None:
        raise RuntimeError(f"compact attribution source無法在驗證後標記READY: {arm_id}/seed={seed}")
    payload["scientific_observation_validation"] = {
        "status": "VERIFIED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
    }
    _write_json(_attribution_unit_dir(run_root, arm_id, seed) / MANIFEST_FILENAME, payload)


def _attribution_ready_units(
    run_root: Path,
    *,
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    seeds: tuple[int, ...],
    fingerprint: str,
) -> set[tuple[str, int]]:
    ready: set[tuple[str, int]] = set()
    for arm in stochastic_arms:
        for seed in seeds:
            if _read_attribution_unit_manifest(
                run_root,
                arm_id=arm.arm_id,
                seed=int(seed),
                expected_fingerprint=fingerprint,
            ) is not None:
                ready.add((arm.arm_id, int(seed)))
    return ready



def _write_attribution_index_manifest(
    run_root: Path,
    *,
    contract: dict[str, Any],
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    seeds: tuple[int, ...],
) -> Path:
    root = _attribution_source_root(run_root)
    units: list[dict[str, Any]] = []
    for arm in stochastic_arms:
        for seed in seeds:
            payload = _read_attribution_unit_manifest(
                run_root,
                arm_id=arm.arm_id,
                seed=int(seed),
                expected_fingerprint=str(contract["fingerprint"]),
            )
            if payload is None:
                raise RuntimeError(
                    f"compact attribution source不完整: {_unit_key(arm.arm_id, int(seed))}"
                )
            units.append({
                "arm_id": arm.arm_id,
                "arm_name": arm.name,
                "seed": int(seed),
                "seed_order": int(payload["seed_order"]),
                "manifest_path": project_relative_display_path(
                    _attribution_unit_dir(run_root, arm.arm_id, int(seed)) / MANIFEST_FILENAME,
                    project_root=PROJECT_ROOT,
                ),
            })
    index = {
        "schema_version": ATTRIBUTION_SOURCE_SCHEMA_VERSION,
        "scientific_fingerprint": str(contract["fingerprint"]),
        "robustness_id": str(contract["robustness_id"]),
        "profile_id": str(contract["profile_id"]),
        "comparison_period": dict(contract.get("comparison_period") or {}),
        "unit_count": len(units),
        "units": sorted(units, key=lambda item: (int(item["seed_order"]), str(item["arm_id"]))),
        "read_only_audit_source": True,
    }
    path = root / MANIFEST_FILENAME
    _write_json(path, index)
    return path


def _validate_rebuilt_observation(
    *,
    existing_row: dict[str, Any],
    rebuilt: dict[str, Any],
    existing_yearly: pd.DataFrame,
    tolerance: float = 1e-6,
) -> None:
    for _label, key, _unit in MEAN_METRICS:
        left = pd.to_numeric(pd.Series([existing_row.get(key)]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([rebuilt.get(key)]), errors="coerce").iloc[0]
        if pd.isna(left) and pd.isna(right):
            continue
        if pd.isna(left) or pd.isna(right) or not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"attribution rebuild scientific observation漂移: metric={key}, existing={left}, rebuilt={right}"
            )

    def optional_int(value: Any) -> int | None:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(parsed) else int(parsed)

    for key in ("selected_epoch", "fold_count"):
        left = optional_int(existing_row.get(key))
        right = optional_int(rebuilt.get(key))
        if left != right:
            raise RuntimeError(
                f"attribution rebuild training identity漂移: field={key}, existing={left}, rebuilt={right}"
            )
    def optional_hash(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip().lower()

    for key in ("model_sha256", "score_sha256", "runtime_source_identity_sha256"):
        left = optional_hash(existing_row.get(key))
        right = optional_hash(rebuilt.get(key))
        if left and left != right:
            raise RuntimeError(
                f"attribution rebuild artifact identity漂移: field={key}, existing={left}, rebuilt={right}"
            )

    rebuilt_yearly = pd.DataFrame(rebuilt.get("yearly") or [])
    expected = existing_yearly.copy()
    if not expected.empty:
        expected = expected[["year", "return_pct", "is_complete_year"]].sort_values("year").reset_index(drop=True)
    if not rebuilt_yearly.empty:
        rebuilt_yearly = rebuilt_yearly[["year", "return_pct", "is_complete_year"]].sort_values("year").reset_index(drop=True)
    if len(expected) != len(rebuilt_yearly):
        raise RuntimeError("attribution rebuild年度observation列數漂移")
    for left, right in zip(expected.to_dict("records"), rebuilt_yearly.to_dict("records")):
        if int(left["year"]) != int(right["year"]) or bool(left["is_complete_year"]) != bool(right["is_complete_year"]):
            raise RuntimeError("attribution rebuild年度observation identity漂移")
        if not math.isclose(float(left["return_pct"]), float(right["return_pct"]), rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError("attribution rebuild年度報酬漂移")

def _load_seed_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return frame
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


def _validate_seed_results_frame(
    frame: pd.DataFrame,
    *,
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    seeds: tuple[int, ...],
) -> None:
    if frame.empty:
        return
    required_columns = {
        "arm_id", "seed", "arm_order", "seed_order",
        *(key for _label, key, _unit in MEAN_METRICS),
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "multi-seed seed_results缺少欄位: " + ", ".join(missing_columns)
        )
    keys = [
        (str(row.arm_id), int(row.seed)) for row in frame.itertuples(index=False)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("multi-seed seed_results存在重複arm/seed observation")
    expected = {
        (arm.arm_id, int(seed)) for arm in stochastic_arms for seed in seeds
    }
    unexpected = sorted(set(keys) - expected)
    if unexpected:
        preview = ", ".join(_unit_key(arm_id, seed) for arm_id, seed in unexpected[:4])
        raise ValueError(f"multi-seed seed_results含非本contract observation: {preview}")
    arm_order = {arm.arm_id: index for index, arm in enumerate(stochastic_arms, start=1)}
    seed_order = {int(seed): index for index, seed in enumerate(seeds, start=1)}
    for row in frame.itertuples(index=False):
        if int(row.arm_order) != arm_order[str(row.arm_id)]:
            raise ValueError(f"multi-seed seed_results arm_order不一致: {row.arm_id}")
        if int(row.seed_order) != seed_order[int(row.seed)]:
            raise ValueError(f"multi-seed seed_results seed_order不一致: {row.seed}")


def _write_seed_results(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_csv(path, frame, sort_columns=("arm_order", "seed_order"))


def _normalized_hash_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return str(value).strip().lower()


def _validate_seed_result_scientific_identities(
    frame: pd.DataFrame, *, contract: dict[str, Any]
) -> None:
    """Reject stale/cross-run rows even when their arm/seed keys look valid."""

    if frame.empty or contract.get("benchmark_id") in (None, ""):
        return
    required = {"strategy_param_sha256", "strategy_param_manifest_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "multi-seed seed_results缺少strategy parameter identity欄位: "
            + ", ".join(missing)
        )
    identities = dict(contract.get("benchmark_parameter_artifact_identities") or {})
    model_ids = {str(value) for value in tuple(contract.get("model_seed_sensitive_arm_ids") or ())}
    for row in frame.to_dict("records"):
        arm_id = str(row.get("arm_id") or "")
        seed = int(row.get("seed"))
        expected = dict(identities.get(f"{arm_id}:seed={seed}") or {})
        if not expected:
            raise ValueError(
                f"multi-seed seed_results缺少contract parameter identity: {arm_id}/seed={seed}"
            )
        actual_param = _normalized_hash_value(row.get("strategy_param_sha256"))
        actual_manifest = _normalized_hash_value(row.get("strategy_param_manifest_sha256"))
        if actual_param != _normalized_hash_value(expected.get("sha256")):
            raise ValueError(
                f"multi-seed seed_results strategy param SHA stale: {arm_id}/seed={seed}"
            )
        if not actual_manifest:
            raise ValueError(
                f"multi-seed seed_results缺少strategy param manifest provenance: {arm_id}/seed={seed}"
            )
        if arm_id in model_ids:
            arm_contract = next(
                (
                    dict(item or {})
                    for item in tuple(contract.get("stochastic_arms") or ())
                    if str(dict(item or {}).get("arm_id") or "") == arm_id
                ),
                {},
            )
            runtime_sources = tuple(arm_contract.get("runtime_dl_sources") or ())
            score_sources = {
                str(dict(item or {}).get("score_source") or "").strip()
                for item in runtime_sources
                if str(dict(item or {}).get("score_source") or "").strip()
            }
            required_model_keys = ["score_sha256", "runtime_source_identity_sha256"]
            if score_sources != {"selection_point_in_time"}:
                required_model_keys.insert(0, "model_sha256")
            else:
                fold_count = pd.to_numeric(
                    pd.Series([row.get("fold_count")]), errors="coerce"
                ).iloc[0]
                if pd.isna(fold_count) or int(fold_count) < 1:
                    raise ValueError(
                        "multi-seed PIT model observation缺少合法fold_count: "
                        f"{arm_id}/seed={seed}"
                    )
            for key in required_model_keys:
                if not _normalized_hash_value(row.get(key)):
                    raise ValueError(
                        f"multi-seed model observation缺少{key}: {arm_id}/seed={seed}"
                    )


def _comparison_years(comparison_start: str, comparison_end: str) -> tuple[int, ...]:
    start_year = int(pd.Timestamp(comparison_start).year)
    end_year = int(pd.Timestamp(comparison_end).year)
    if end_year < start_year:
        raise ValueError(
            f"multi-seed comparison period年份顛倒: {comparison_start}~{comparison_end}"
        )
    return tuple(range(start_year, end_year + 1))


def _validate_seed_yearly_results_frame(
    frame: pd.DataFrame,
    *,
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    seeds: tuple[int, ...],
    expected_years: tuple[int, ...],
    require_complete_units: bool,
) -> None:
    if frame.empty:
        if require_complete_units:
            raise ValueError("multi-seed seed_yearly_results為空")
        return
    required = {
        "arm_id", "seed", "arm_order", "seed_order", "year",
        "return_pct", "is_complete_year",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "multi-seed seed_yearly_results缺少欄位: " + ", ".join(missing)
        )
    keys = [
        (str(row.arm_id), int(row.seed), int(row.year))
        for row in frame.itertuples(index=False)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("multi-seed seed_yearly_results存在重複arm/seed/year observation")
    expected_units = {
        (arm.arm_id, int(seed)) for arm in stochastic_arms for seed in seeds
    }
    expected_year_set = {int(year) for year in expected_years}
    expected_keys = {
        (arm_id, seed, year)
        for arm_id, seed in expected_units
        for year in expected_year_set
    }
    actual_units = {(arm_id, seed) for arm_id, seed, _year in keys}
    unexpected_units = sorted(actual_units - expected_units)
    if unexpected_units:
        raise ValueError(
            "multi-seed seed_yearly_results含非本contract observation: "
            + ", ".join(_unit_key(arm_id, seed) for arm_id, seed in unexpected_units[:4])
        )
    unexpected_years = sorted({year for _arm_id, _seed, year in keys} - expected_year_set)
    if unexpected_years:
        raise ValueError(
            "multi-seed seed_yearly_results含comparison period外年度: "
            + ", ".join(str(year) for year in unexpected_years[:6])
        )
    if require_complete_units and set(keys) != expected_keys:
        missing_count = len(expected_keys - set(keys))
        raise ValueError(
            "multi-seed seed_yearly_results年度單元不完整: "
            f"expected={len(expected_keys)}, actual={len(keys)}, missing={missing_count}"
        )
    arm_order = {arm.arm_id: index for index, arm in enumerate(stochastic_arms, start=1)}
    seed_order = {int(seed): index for index, seed in enumerate(seeds, start=1)}
    for row in frame.itertuples(index=False):
        if int(row.arm_order) != arm_order[str(row.arm_id)]:
            raise ValueError(f"multi-seed seed_yearly_results arm_order不一致: {row.arm_id}")
        if int(row.seed_order) != seed_order[int(row.seed)]:
            raise ValueError(f"multi-seed seed_yearly_results seed_order不一致: {row.seed}")


def _validate_summary_seed_aggregates(
    summary: dict[str, Any], *, seed_frame: pd.DataFrame, contract: dict[str, Any]
) -> None:
    rows = {
        str(row.get("arm_id") or ""): dict(row)
        for row in tuple(summary.get("mean_strategy_metrics") or ())
        if str(row.get("type") or "") == "Multi-seed"
    }
    settings = get_strategy_comparison_settings(str(contract["profile_id"]))
    stochastic = _contract_stochastic_arms(contract, settings)
    for arm in stochastic:
        group = seed_frame[seed_frame["arm_id"].astype(str) == arm.arm_id]
        rendered = rows.get(arm.arm_id)
        if rendered is None or int(rendered.get("n") or -1) != len(group):
            raise ValueError(f"robustness summary seed aggregate缺少／N不一致: {arm.arm_id}")
        for _label, key, _unit in MEAN_METRICS:
            expected = _mean_or_none(group[key]) if key in group.columns else None
            actual_raw = rendered.get(key)
            actual = _finite_or_none(actual_raw)
            if expected is None and actual is None:
                continue
            if expected is None or actual is None or not math.isclose(
                float(expected), float(actual), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"robustness summary與seed_results不一致: arm={arm.arm_id}, metric={key}"
                )


def _build_durable_result_artifacts(run_root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for key, filename in DURABLE_RESULT_FILENAMES.items():
        path = run_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"robustness durable result缺少: {path}")
        artifacts[key] = {
            "path": project_relative_display_path(path, project_root=PROJECT_ROOT),
            **build_file_manifest(path),
        }
    return artifacts


def _validate_durable_result_artifacts(
    run_root: Path, artifacts: dict[str, Any], *, keys: tuple[str, ...] | None = None
) -> bool:
    """Validate durable artifacts at the requested evidence layer.

    ``seed_results`` and ``seed_yearly_results`` are scientific observations.
    ``summary`` and ``report`` are derived presentation artifacts that may be
    regenerated after renderer/schema maintenance without invalidating the
    expensive stochastic observations behind them.
    """

    if not artifacts:
        return False
    selected_keys = tuple(keys or DURABLE_RESULT_FILENAMES.keys())
    for key in selected_keys:
        filename = DURABLE_RESULT_FILENAMES.get(str(key))
        if filename is None:
            raise KeyError(f"未知robustness durable artifact key: {key}")
        item = dict(artifacts.get(key) or {})
        path = run_root / filename
        if not path.is_file() or Path(str(item.get("filename") or "")).name != filename:
            return False
        if int(item.get("size_bytes") or -1) != int(path.stat().st_size):
            return False
        expected_sha = _normalized_hash_value(item.get("sha256"))
        if not expected_sha or compute_file_sha256(path).lower() != expected_sha:
            return False
    return True


def _scientific_observation_manifest_path(run_root: Path) -> Path:
    return Path(run_root) / SCIENTIFIC_OBSERVATIONS_MANIFEST_FILENAME


def _build_scientific_observation_artifacts(
    run_root: Path, *, attribution_index_path: Path | None
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for key in SCIENTIFIC_DURABLE_RESULT_KEYS:
        filename = DURABLE_RESULT_FILENAMES[key]
        path = Path(run_root) / filename
        if not path.is_file():
            raise FileNotFoundError(f"robustness scientific observation缺少: {path}")
        artifacts[key] = {
            "path": project_relative_display_path(path, project_root=PROJECT_ROOT),
            **build_file_manifest(path),
        }
    if attribution_index_path is not None:
        index_path = Path(attribution_index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"robustness attribution index缺少: {index_path}")
        artifacts["attribution_index"] = {
            "path": project_relative_display_path(index_path, project_root=PROJECT_ROOT),
            **build_file_manifest(index_path),
        }
    return artifacts


def _write_scientific_observation_manifest(
    run_root: Path,
    *,
    contract: dict[str, Any],
    seed_frame: pd.DataFrame,
    seed_yearly_frame: pd.DataFrame,
    attribution_index_path: Path | None,
) -> dict[str, Any]:
    """Commit expensive robustness observations independently from run/report status.

    This READY manifest is the commit marker for stochastic evidence within the
    exact scientific fingerprint. It intentionally excludes summary/report
    presentation so renderer refreshes or lifecycle retries do not alter evidence
    integrity. Cross-fingerprint result migration is deliberately unsupported.
    """

    payload = {
        "schema_version": SCIENTIFIC_OBSERVATIONS_SCHEMA_VERSION,
        "status": "READY",
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_fingerprint": str(contract["fingerprint"]),
        "contract": contract,
        "seed_strategy_observations": int(len(seed_frame)),
        "seed_yearly_observations": int(len(seed_yearly_frame)),
        "artifacts": _build_scientific_observation_artifacts(
            run_root, attribution_index_path=attribution_index_path
        ),
    }
    _write_json(_scientific_observation_manifest_path(run_root), payload)
    return payload


def _load_validated_scientific_observations(
    run_root: Path,
    *,
    contract: dict[str, Any],
    scientific_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    """Load and validate the one canonical stochastic-observation evidence set."""

    fingerprint = str(contract.get("fingerprint") or "")
    settings = get_strategy_comparison_settings(str(contract["profile_id"]))
    stochastic = _contract_stochastic_arms(contract, settings)
    seeds = tuple(int(value) for value in tuple(contract.get("resolved_seeds") or ()))
    if not seeds:
        raise ValueError("robustness scientific observation缺少resolved seeds")

    seed_frame = _load_seed_results(Path(run_root) / SEED_RESULTS_FILENAME)
    _validate_seed_results_frame(seed_frame, stochastic_arms=stochastic, seeds=seeds)
    _validate_seed_result_scientific_identities(seed_frame, contract=contract)
    expected_units = {(arm.arm_id, int(seed)) for arm in stochastic for seed in seeds}
    actual_units = {
        (str(row.arm_id), int(row.seed)) for row in seed_frame.itertuples(index=False)
    }
    if actual_units != expected_units:
        raise ValueError("robustness scientific seed observation單元不完整")

    yearly_frame = _load_seed_yearly_results(Path(run_root) / SEED_YEARLY_RESULTS_FILENAME)
    comparison_period = dict(contract.get("comparison_period") or {})
    _validate_seed_yearly_results_frame(
        yearly_frame,
        stochastic_arms=stochastic,
        seeds=seeds,
        expected_years=_comparison_years(
            str(comparison_period.get("start") or ""),
            str(comparison_period.get("end") or ""),
        ),
        require_complete_units=True,
    )

    if scientific_artifacts is not None and not _validate_durable_result_artifacts(
        run_root,
        scientific_artifacts,
        keys=SCIENTIFIC_DURABLE_RESULT_KEYS,
    ):
        raise ValueError("robustness scientific seed artifact SHA/size不一致")

    model_ids = {
        str(value) for value in tuple(contract.get("model_seed_sensitive_arm_ids") or ())
    }
    if bool(dict(contract.get("retention") or {}).get("keep_attribution_source")):
        if scientific_artifacts is not None:
            attribution_item = dict(scientific_artifacts.get("attribution_index") or {})
            attribution_path = _attribution_source_root(run_root) / MANIFEST_FILENAME
            expected_sha = _normalized_hash_value(attribution_item.get("sha256"))
            if (
                not attribution_path.is_file()
                or Path(str(attribution_item.get("filename") or "")).name != MANIFEST_FILENAME
                or int(attribution_item.get("size_bytes") or -1) != int(attribution_path.stat().st_size)
                or not expected_sha
                or compute_file_sha256(attribution_path).lower() != expected_sha
            ):
                raise ValueError("robustness attribution index SHA/size不一致")
        model_arms = tuple(arm for arm in stochastic if arm.arm_id in model_ids)
        ready = _attribution_ready_units(
            run_root,
            stochastic_arms=model_arms,
            seeds=seeds,
            fingerprint=fingerprint,
        )
        if len(ready) != len(model_arms) * len(seeds):
            raise ValueError("robustness compact attribution source不完整")

    return {
        "stochastic_arms": stochastic,
        "seeds": seeds,
        "seed_frame": seed_frame,
        "seed_yearly_frame": yearly_frame,
    }


def _validate_scientific_observation_manifest(
    run_root: Path,
) -> dict[str, Any] | None:
    """Validate exact-fingerprint scientific evidence.

    No legacy backfill or cross-fingerprint migration is attempted. If the READY
    marker is absent or incompatible, the caller must rebuild under the current
    scientific identity.
    """

    evidence_path = _scientific_observation_manifest_path(run_root)
    if not evidence_path.is_file():
        return None
    try:
        evidence = _read_json(evidence_path)
        if int(evidence.get("schema_version") or 0) != SCIENTIFIC_OBSERVATIONS_SCHEMA_VERSION:
            return None
        if str(evidence.get("status") or "") != "READY":
            return None
        contract = dict(evidence.get("contract") or {})
        fingerprint = str(contract.get("fingerprint") or "")
        if not fingerprint or fingerprint != str(evidence.get("scientific_fingerprint") or ""):
            return None
        if fingerprint != Path(run_root).name:
            return None
        validated = _load_validated_scientific_observations(
            run_root,
            contract=contract,
            scientific_artifacts=dict(evidence.get("artifacts") or {}),
        )
        return {
            "evidence": evidence,
            "contract": contract,
            **validated,
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

def _validate_completed_run(
    run_root: Path,
    *,
    expected_contract: dict[str, Any] | None = None,
    backfill_integrity: bool,
) -> dict[str, Any] | None:
    """Validate one completed result for exact-fingerprint no-op REUSE."""

    manifest_path = run_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
        if str(manifest.get("status") or "") != "COMPLETED":
            return None
        contract = dict(manifest.get("contract") or {})
        fingerprint = str(contract.get("fingerprint") or "")
        if not fingerprint or fingerprint != run_root.name:
            return None
        if expected_contract is not None and fingerprint != str(expected_contract.get("fingerprint") or ""):
            return None
        validated = _load_validated_scientific_observations(
            run_root,
            contract=contract,
            scientific_artifacts=None,
        )
        stochastic = tuple(validated["stochastic_arms"])
        seeds = tuple(int(value) for value in validated["seeds"])
        seed_frame = validated["seed_frame"]
        yearly_frame = validated["seed_yearly_frame"]
        summary = _read_json(run_root / SUMMARY_FILENAME)
        summary_contract = dict(summary.get("contract") or {})
        if str(summary_contract.get("fingerprint") or "") != fingerprint:
            return None
        _validate_summary_seed_aggregates(summary, seed_frame=seed_frame, contract=contract)
        if not (run_root / REPORT_FILENAME).is_file():
            return None
        durable = dict(manifest.get("durable_artifacts") or {})
        if durable:
            if not _validate_durable_result_artifacts(
                run_root, durable, keys=tuple(DURABLE_RESULT_FILENAMES.keys())
            ):
                return None
        elif backfill_integrity:
            manifest["durable_artifacts"] = _build_durable_result_artifacts(run_root)
            manifest["integrity_backfilled_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)
        return {
            "manifest": manifest,
            "contract": contract,
            "stochastic_arms": stochastic,
            "seeds": seeds,
            "seed_frame": seed_frame,
            "seed_yearly_frame": yearly_frame,
            "summary": summary,
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _resolve_existing_scientific_unit_states(
    *,
    run_root: Path,
    contract: dict[str, Any],
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    reuse_completed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    """Resolve durable per-seed scientific result states through one owner.

    Both the robustness execution plan and executor call this function.  A seed CSV row
    alone is insufficient: exact current contract identities and the complete expected
    yearly observation set are required before a unit becomes REUSE.
    """

    seeds = tuple(int(value) for value in contract["resolved_seeds"])
    seed_path = run_root / SEED_RESULTS_FILENAME
    yearly_path = run_root / SEED_YEARLY_RESULTS_FILENAME
    existing = _load_seed_results(seed_path)
    _validate_seed_results_frame(existing, stochastic_arms=stochastic_arms, seeds=seeds)
    _validate_seed_result_scientific_identities(existing, contract=contract)
    existing_yearly = _load_seed_yearly_results(yearly_path)
    expected_years = _comparison_years(
        str(contract["comparison_period"]["start"]),
        str(contract["comparison_period"]["end"]),
    )
    _validate_seed_yearly_results_frame(
        existing_yearly,
        stochastic_arms=stochastic_arms,
        seeds=seeds,
        expected_years=expected_years,
        require_complete_units=False,
    )
    completed_rows = (
        {(str(row.arm_id), int(row.seed)) for row in existing.itertuples(index=False)}
        if reuse_completed and not existing.empty
        else set()
    )
    if completed_rows:
        expected_year_set = set(expected_years)
        yearly_by_unit: dict[tuple[str, int], set[int]] = {}
        for row in existing_yearly.itertuples(index=False):
            unit = (str(row.arm_id), int(row.seed))
            yearly_by_unit.setdefault(unit, set()).add(int(row.year))
        completed_rows &= {
            unit for unit, years in yearly_by_unit.items()
            if years == expected_year_set
        }
    unit_states: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in stochastic_arms:
        for seed in seeds:
            unit = (arm.arm_id, int(seed))
            unit_states[unit] = resolve_strategy_result_state(
                verified_evidence={"arm_id": arm.arm_id, "seed": int(seed)}
                if unit in completed_rows
                else None,
                missing_reason=f"{arm.arm_id}/seed={seed}尚無完整identity一致結果",
            ).as_dict()
    return existing, existing_yearly, unit_states


def _resolved_period(settings, status: dict[str, Any]) -> tuple[str, str]:
    period = dict(status.get("comparison_period") or {})
    start = str(period.get("start") or settings.start_date or "").strip()
    end = str(period.get("end") or settings.end_date or "").strip()
    if not start or not end:
        raise RuntimeError("multi-seed robustness無法解析共同策略比較期間")
    return start, end



def _train_one_unit(job: dict[str, Any]) -> dict[str, Any]:
    """Run one robustness seed through the shared single/multi training lifecycle."""

    settings = job["settings"]
    seed = int(job["seed"])
    model_dir = Path(job["model_dir"]).resolve()
    research_dir = Path(job["research_dir"]).resolve()
    train_log_path = research_dir / "train.log"
    dl_id = str(job["dl_id"])
    dl = settings.dl_sources[dl_id]
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(dl.experiment_profile)
    )
    is_selection_pit = str(dl.score_source) == SCORE_SOURCE_SELECTION_POINT_IN_TIME
    checkpoint_cache_root = (
        None
        if job.get("checkpoint_cache_root") in (None, "")
        else Path(str(job["checkpoint_cache_root"])).resolve()
    )
    started = time.perf_counter()
    artifacts = None
    reusable_artifact_roots_exist = model_dir.is_dir() and (
        is_selection_pit or research_dir.is_dir()
    )
    if bool(job.get("reuse_completed")) and reusable_artifact_roots_exist:
        try:
            artifacts = validate_strategy_compare_training_artifacts(
                project_root=PROJECT_ROOT,
                source=dl,
                workflow=workflow,
                seed=seed,
                model_dir=model_dir,
                research_dir=research_dir,
                comparison_start=str(job["comparison_start"]),
                comparison_end=str(job["comparison_end"]),
                point_in_time_dir_override=(model_dir if is_selection_pit else None),
            )
        except (FileNotFoundError, ValueError):
            artifacts = None
    if artifacts is not None:
        artifacts["model_prediction_metrics"] = (
            _model_prediction_metrics_from_training_artifacts(artifacts)
        )
        return {
            "artifacts": artifacts,
            "reused": True,
            "wall_elapsed_sec": round(time.perf_counter() - started, 3),
        }

    if is_selection_pit:
        # Isolated benchmark namespace may resume completed PIT folds.  Remove only
        # top-level aggregate artifacts so the canonical builder can reconstruct them.
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            SELECTION_POINT_IN_TIME_SCORE_FILENAME,
            SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
            SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
        ):
            try:
                (model_dir / filename).unlink()
            except FileNotFoundError:
                pass
    else:
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(research_dir, ignore_errors=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    trained = run_strategy_compare_training_unit(
        project_root=PROJECT_ROOT,
        source=dl,
        workflow=workflow,
        seed=seed,
        model_dir=model_dir,
        research_dir=research_dir,
        log_path=train_log_path,
        comparison_start=str(job["comparison_start"]),
        comparison_end=str(job["comparison_end"]),
        point_in_time_dir_override=(model_dir if is_selection_pit else None),
        checkpoint_cache_root=checkpoint_cache_root,
        model_output_dir=(model_dir if not is_selection_pit else None),
        research_output_dir=(research_dir if not is_selection_pit else None),
        registry=job["trainer_registry"],
        registry_lock=job["trainer_registry_lock"],
        registry_key=str(job["trainer_key"]),
        failure_prefix=(
            f"multi-seed模型訓練失敗: dl_id={dl_id}, seed_index={job['seed_order']}"
        ),
        resume=True,
    )
    artifacts = dict(trained["artifacts"])
    artifacts["model_prediction_metrics"] = (
        _model_prediction_metrics_from_training_artifacts(artifacts)
    )
    return {
        "artifacts": artifacts,
        "reused": False,
        "wall_elapsed_sec": round(time.perf_counter() - started, 3),
    }


def _training_units(
    *,
    seeds: tuple[int, ...],
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    settings,
    completed: set[tuple[str, int]],
    model_root: Path,
    run_root: Path,
    comparison_start: str,
    comparison_end: str,
    reuse_completed: bool,
    benchmark_id: str | None = None,
    checkpoint_cache_root: str | None = None,
) -> deque[dict[str, Any]]:
    units: deque[dict[str, Any]] = deque()
    source_groups = _training_source_groups(stochastic_arms, settings=settings)
    for seed_order, seed in enumerate(seeds, start=1):
        for source_order, (dl_id, arm_entries) in enumerate(source_groups, start=1):
            replay_entries = tuple(
                (arm_order, arm)
                for arm_order, arm in arm_entries
                if (arm.arm_id, int(seed)) not in completed
            )
            if not replay_entries:
                continue
            representative_arm = arm_entries[0][1]
            source_unit = _unit_key(dl_id, int(seed))
            checkpoint_cache_dir = (
                None
                if checkpoint_cache_root in (None, "") or benchmark_id in (None, "")
                else (
                    PROJECT_ROOT
                    / str(checkpoint_cache_root)
                    / str(benchmark_id)
                    / _unit_key(str(dl_id), int(seed))
                ).resolve()
            )
            units.append({
                "arm": representative_arm,
                "replay_arms": replay_entries,
                "dl_id": dl_id,
                "source_order": int(source_order),
                "source_count": int(len(source_groups)),
                "settings": settings,
                "seed": int(seed),
                "seed_order": int(seed_order),
                "model_dir": str(model_root / source_unit / "model"),
                "research_dir": str(run_root / "work" / "training" / source_unit),
                "comparison_start": str(comparison_start),
                "comparison_end": str(comparison_end),
                "reuse_completed": bool(reuse_completed),
                "benchmark_id": benchmark_id,
                "checkpoint_cache_root": (
                    None
                    if checkpoint_cache_dir is None
                    else str(checkpoint_cache_dir)
                ),
            })
    return units


def _pop_next_training_unit(
    pending_trainings: deque[dict[str, Any]],
    training_futures: dict[Future, dict[str, Any]],
) -> dict[str, Any]:
    return pop_next_seed_diverse_unit(
        pending_trainings,
        training_futures.values(),
    )


def _normalize_yearly_rows(
    yearly: Any, *, arm_id: str, name: str, seed: int | None, seed_order: int | None, arm_order: int,
    result_side: str = "score_ranking",
) -> list[dict[str, Any]]:
    side = str(result_side).strip()
    if side not in {"score_ranking", "no_filter"}:
        raise ValueError(f"不支援的年度報酬side: {side!r}")
    primary_column = "score_ranking_return_pct" if side == "score_ranking" else "no_filter_return_pct"
    rows: list[dict[str, Any]] = []
    for raw in list(yearly or []):
        item = dict(raw or {})
        value = item.get(primary_column)
        if value is None:
            raise ValueError(
                f"年度報酬缺少{side} canonical欄位: year={item.get('year')!r}, column={primary_column}"
            )
        rows.append({
            "arm_id": str(arm_id), "name": str(name),
            "seed": None if seed is None else int(seed),
            "seed_order": None if seed_order is None else int(seed_order),
            "arm_order": int(arm_order),
            "year": int(item.get("year")),
            "return_pct": float(value),
            "is_complete_year": bool(item.get("is_full_year", item.get("is_complete_year", item.get("complete_year", False)))),
        })
    return rows


def _coerce_bool_series(series: pd.Series, *, field_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "是": True, "false": False, "0": False, "no": False, "否": False}
    invalid = sorted(set(normalized[~normalized.isin(mapping)].tolist()))
    if invalid:
        raise ValueError(f"{field_name}包含無法解析的布林值: {invalid[:5]}")
    return normalized.map(mapping).astype(bool)


def _load_seed_yearly_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return frame
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if "is_complete_year" in frame.columns:
        frame["is_complete_year"] = _coerce_bool_series(
            frame["is_complete_year"], field_name="seed_yearly_results.is_complete_year"
        )
    return frame


def _write_seed_yearly_results(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_csv(
        path, frame, sort_columns=("arm_order", "seed_order", "year")
    )


def _baseline_work_dir(run_root: Path, arm: StrategyComparisonArm) -> Path:
    return run_root / "work" / "baselines" / arm.arm_id


def _load_local_fixed_baseline_if_compatible(
    *, candidate: Path, settings, status: dict[str, Any], arm: StrategyComparisonArm,
    comparison_start: str, comparison_end: str,
) -> dict[str, Any] | None:
    """Reuse a fixed baseline left by an interrupted robustness run."""

    payload_path = candidate / "strategy_comparison.json"
    required_names = (
        "strategy_comparison.json", "yearly_returns_comparison.csv",
        "no_filter_equity.csv", "no_filter_trades.csv",
        "no_filter_daily_capacity.csv", "no_filter_orderable_candidates.csv",
    )
    if not all((candidate / name).is_file() for name in required_names):
        return None
    try:
        payload = _read_json(payload_path)
        metadata = dict(payload.get("metadata") or {})
        identity = dict(
            (status.get("artifact_identities") or {}).get(
                strategy_comparison_param_artifact_key(
                    arm.param_source,
                    resolve_strategy_comparison_arm_param_policy(settings, arm),
                )
            )
            or {}
        )
        expected_param_sha = str(identity.get("sha256") or "").strip().lower()
        if not expected_param_sha:
            return None
        all_off = arm.rule_policy == "all_off"
        param_evaluation_mode = str(
            settings.parameter_sources[arm.param_source].canonical_evaluation_mode or "rolling"
        )
        expected = {
            "dataset": str(settings.dataset),
            "params_file_sha256": expected_param_sha,
            "requested_param_policy": resolve_strategy_comparison_arm_param_policy(settings, arm),
            "param_evaluation_mode": param_evaluation_mode,
            "optional_entry_filter_policy": (
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            "shared_param_overrides": dict(ALL_RULE_FILTERS_OFF_OVERRIDES) if all_off else {},
            "max_positions": int(settings.max_positions),
            "enable_rotation": settings.rotation == "on",
            "comparison_period": {
                "start": str(comparison_start), "end": str(comparison_end),
            },
        }
        actual = {
            "dataset": str(metadata.get("dataset") or ""),
            "params_file_sha256": str(metadata.get("params_file_sha256") or "").strip().lower(),
            "requested_param_policy": str(metadata.get("requested_param_policy") or ""),
            "param_evaluation_mode": str(metadata.get("param_evaluation_mode") or ""),
            "optional_entry_filter_policy": str(metadata.get("optional_entry_filter_policy") or ""),
            "shared_param_overrides": dict(metadata.get("shared_param_overrides") or {}),
            "max_positions": int(metadata.get("max_positions") or 0),
            "enable_rotation": bool(metadata.get("enable_rotation")),
            "comparison_period": dict(metadata.get("comparison_period") or {}),
        }
        if actual != expected or not dict(payload.get("no_filter") or {}):
            return None
        return payload
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _run_fixed_baselines(*, settings, status, run_root: Path, fixed_arms) -> dict[str, dict[str, Any]]:
    start, end = _resolved_period(settings, status)
    results: dict[str, dict[str, Any]] = {}
    for arm in fixed_arms:
        path_text = str((status.get("resolved_arm_parameter_paths") or {}).get(arm.arm_id) or "")
        if not path_text:
            raise RuntimeError(f"robustness baseline缺少arm param binding: {arm.arm_id}")
        arm_param_policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
        output_dir = _baseline_work_dir(run_root, arm)
        payload = _load_local_fixed_baseline_if_compatible(
            candidate=output_dir, settings=settings, status=status, arm=arm,
            comparison_start=start, comparison_end=end,
        )
        reusable_dir = output_dir if payload is not None else _find_reusable_baseline_source(
            root=PROJECT_ROOT, settings=settings, status=status, off_arm=arm
        )
        if reusable_dir is not None:
            if payload is None:
                payload = _read_json(reusable_dir / "strategy_comparison.json")
            results[arm.arm_id] = {
                "arm_id": arm.arm_id,
                "name": arm.name,
                "metrics": dict(payload.get("no_filter") or {}),
                "yearly": _normalize_yearly_rows(
                    payload.get("yearly"), arm_id=arm.arm_id, name=arm.name,
                    seed=None, seed_order=None, arm_order=list(fixed_arms).index(arm) + 1,
                    result_side="no_filter",
                ),
                "baseline_dir": str(reusable_dir.resolve()),
                "source": "REUSE",
            }
            color_enabled = console_color_enabled()
            print(
                styled_workflow_status("[BASELINE REUSE]")
                + f" {arm.name} | {project_relative_display_path(reusable_dir, project_root=PROJECT_ROOT)}"
            )
            continue
        if output_dir.exists():
            shutil.rmtree(output_dir)
        param_evaluation_mode = str(
            settings.parameter_sources[arm.param_source].canonical_evaluation_mode or "rolling"
        )
        payload = run_strategy_compare_baseline_arm(
            settings=settings,
            arm=arm,
            project_root=PROJECT_ROOT,
            params_path=path_text,
            output_dir=output_dir,
            comparison_start=start,
            comparison_end=end,
            quiet=True,
            param_evaluation_mode=param_evaluation_mode,
        )
        results[arm.arm_id] = {
            "arm_id": arm.arm_id,
            "name": arm.name,
            "metrics": dict(payload.get("no_filter") or {}),
            "yearly": _normalize_yearly_rows(
                payload.get("yearly"), arm_id=arm.arm_id, name=arm.name,
                seed=None, seed_order=None, arm_order=list(fixed_arms).index(arm) + 1,
                result_side="no_filter",
            ),
            "baseline_dir": str(output_dir.resolve()),
            "source": "RUN",
        }
    return results


def _replay_one_unit(job: dict[str, Any]) -> dict[str, Any]:
    settings = get_strategy_comparison_settings(str(job["profile_id"]))
    arm = settings.arms[str(job["arm_id"])]
    dl = settings.dl_sources[str(arm.dl_id)]
    start = str(job["comparison_start"]); end = str(job["comparison_end"])
    pair_dir = Path(str(job["pair_dir"])).resolve()
    baseline_dir = Path(str(job["baseline_dir"])).resolve()
    if pair_dir.exists(): shutil.rmtree(pair_dir)
    started = time.perf_counter()
    runtime_spec = _arm_runtime_spec(arm)
    score_overrides = {
        str(key): dict(value or {})
        for key, value in dict(job.get("score_overrides") or {}).items()
    }
    payload = run_strategy_compare_active_arm(
        settings=settings,
        arm=arm,
        project_root=PROJECT_ROOT,
        params_path=str(job["params_path"]),
        output_dir=pair_dir,
        comparison_start=start,
        comparison_end=end,
        param_evaluation_mode=str(job.get("param_evaluation_mode") or "rolling"),
        score_overrides=score_overrides,
        expected_seed_override=int(job["seed"]),
        baseline_reuse_dir=baseline_dir,
        quiet=True,
        capture_execution_diagnostics=bool(job.get("keep_attribution_source")),
        # Multi-seed robustness只需要策略績效、trade-set與compact attribution。
        # Selection future-target join是post-replay離線診斷；daily-universal profile
        # 會重建完整daily stock-day target universe，不能在每個seed replay重做。
        capture_selection_target_diagnostics=False,
    )
    metrics = dict(payload.get("score_ranking") or {})
    metrics["direct_selection_r"] = _load_direct_selection_r(
        pair_dir, root=PROJECT_ROOT, active_trades_filename=runtime_spec["active_trades_filename"]
    )
    model_prediction = dict(job.get("model_prediction_metrics") or {})
    result = {
        "arm_id": arm.arm_id, "name": arm.name, "seed": int(job["seed"]),
        "arm_order": int(job["arm_order"]), "seed_order": int(job["seed_order"]),
        "selected_epoch": int(job.get("selected_epoch", 0) or 0),
        "fold_count": job.get("fold_count"),
        "training_elapsed_sec": float(job.get("training_elapsed_sec", 0.0) or 0.0),
        "replay_elapsed_sec": round(time.perf_counter() - started, 3),
        "model_sha256": str(job.get("model_sha256") or "") or None,
        "score_sha256": str(job.get("score_sha256") or "") or None,
        "runtime_source_identity_sha256": str(job.get("runtime_source_identity_sha256") or "") or None,
        "strategy_param_sha256": str(job.get("strategy_param_sha256") or "") or None,
        "strategy_param_manifest_sha256": str(job.get("strategy_param_manifest_sha256") or "") or None,
        **{key: metrics.get(key) for _label, key, _unit in MEAN_METRICS},
        **{key: metrics.get(key) for key in COMMON_STRATEGY_REPORT_METRIC_KEYS},
        **{key: model_prediction.get(key) for key in (*MODEL_PREDICTION_METRIC_KEYS, "continuous_target_id")},
        "yearly": _normalize_yearly_rows(
            payload.get("yearly"), arm_id=arm.arm_id, name=arm.name, seed=int(job["seed"]),
            seed_order=int(job["seed_order"]), arm_order=int(job["arm_order"]),
        ),
    }
    if bool(job.get("keep_attribution_source")):
        _write_compact_attribution_source(
            pair_dir=pair_dir,
            destination_dir=Path(str(job["attribution_dir"])).resolve(),
            job=job,
            arm=arm,
            dl=dl,
            runtime_spec=runtime_spec,
        )
    result_path = Path(str(job["result_path"])).resolve(); _write_json(result_path, result)
    if not bool(job.get("keep_replay_details")): shutil.rmtree(pair_dir, ignore_errors=True)
    return result


def _result_row(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "yearly"}


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _distribution_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "median", "std", "min", "p25", "p75", "max")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }



def _same_seed_metric_comparison(
    *,
    seed_frame: pd.DataFrame,
    left: StrategyComparisonArm,
    right: StrategyComparisonArm,
    metric_key: str,
    require_same_parameter_runtime_universe: bool = False,
) -> dict[str, Any] | None:
    if metric_key not in seed_frame.columns:
        return None
    if require_same_parameter_runtime_universe and (
        left.param_source != right.param_source
        or left.rule_policy != right.rule_policy
    ):
        return None
    paired = seed_frame.loc[
        seed_frame["arm_id"].isin((left.arm_id, right.arm_id)),
        ["arm_id", "seed", metric_key],
    ].copy()
    paired[metric_key] = pd.to_numeric(paired[metric_key], errors="coerce")
    pivot = paired.dropna().pivot(index="seed", columns="arm_id", values=metric_key)
    if left.arm_id not in pivot.columns or right.arm_id not in pivot.columns:
        return None
    pivot = pivot[[left.arm_id, right.arm_id]].dropna()
    if pivot.empty:
        return None
    delta = pivot[right.arm_id].to_numpy(dtype=float) - pivot[left.arm_id].to_numpy(dtype=float)
    stats = _distribution_stats(delta)
    tie = np.isclose(delta, 0.0, rtol=0.0, atol=1e-12)
    return {
        "metric_key": metric_key,
        "left_arm_id": left.arm_id,
        "right_arm_id": right.arm_id,
        "left": left.name,
        "right": right.name,
        "n": int(len(delta)),
        "left_gt_right_count": int(np.sum((delta < 0.0) & ~tie)),
        "right_gt_left_count": int(np.sum((delta > 0.0) & ~tie)),
        "tie_count": int(np.sum(tie)),
        **{f"right_minus_left_{key}": value for key, value in stats.items()},
    }


def _pairwise_distribution_comparison(
    *, seed_frame: pd.DataFrame, left: StrategyComparisonArm, right: StrategyComparisonArm,
) -> dict[str, Any] | None:
    left_values = pd.to_numeric(
        seed_frame.loc[seed_frame["arm_id"] == left.arm_id, "return_over_max_drawdown"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    right_values = pd.to_numeric(
        seed_frame.loc[seed_frame["arm_id"] == right.arm_id, "return_over_max_drawdown"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    if not len(left_values) or not len(right_values):
        return None
    return {
        "left_arm_id": left.arm_id,
        "right_arm_id": right.arm_id,
        "left": left.name,
        "right": right.name,
        "pairwise_left_gt_right_probability": float(
            np.mean(left_values[:, None] > right_values[None, :])
        ),
        "left_n": int(len(left_values)),
        "right_n": int(len(right_values)),
        "pair_count": int(len(left_values) * len(right_values)),
    }


def _paired_seed_translation_diagnostic(
    *,
    settings: StrategyComparisonSettings,
    seed_frame: pd.DataFrame,
    left: StrategyComparisonArm,
    right: StrategyComparisonArm,
    resolved_seeds: tuple[int, ...] = (),
) -> dict[str, Any] | None:
    """Describe whether same-seed selection-R improvements translate to strategy gains."""
    if (
        left.param_source != right.param_source
        or resolve_strategy_comparison_arm_param_policy(settings, left)
        != resolve_strategy_comparison_arm_param_policy(settings, right)
        or left.rule_policy != right.rule_policy
    ):
        return None

    required_metrics = (
        "direct_selection_r",
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "expected_value_r",
    )
    required_columns = {"arm_id", "seed", *required_metrics}
    if not required_columns.issubset(seed_frame.columns):
        return None

    subset = seed_frame.loc[
        seed_frame["arm_id"].isin((left.arm_id, right.arm_id)),
        ["arm_id", "seed", *required_metrics],
    ].copy()
    for key in required_metrics:
        subset[key] = pd.to_numeric(subset[key], errors="coerce")
    left_rows = subset[subset["arm_id"] == left.arm_id].set_index("seed")
    right_rows = subset[subset["arm_id"] == right.arm_id].set_index("seed")
    common_seeds = left_rows.index.intersection(right_rows.index)
    if common_seeds.empty:
        return None

    seed_order = {int(seed): index for index, seed in enumerate(resolved_seeds, start=1)}
    ordered_seeds = sorted(
        (int(seed) for seed in common_seeds),
        key=lambda seed: (seed_order.get(seed, len(seed_order) + 1), seed),
    )
    rows: list[dict[str, Any]] = []
    for fallback_index, seed in enumerate(ordered_seeds, start=1):
        left_row = left_rows.loc[seed]
        right_row = right_rows.loc[seed]
        if isinstance(left_row, pd.DataFrame) or isinstance(right_row, pd.DataFrame):
            raise ValueError(f"multi-seed aggregate同一arm/seed不得重複: seed={seed}")
        values = {
            key: (float(right_row[key]) - float(left_row[key]))
            for key in required_metrics
        }
        if not all(math.isfinite(value) for value in values.values()):
            continue
        selection_delta = values["direct_selection_r"]
        romd_delta = values["return_over_max_drawdown"]
        selection_sign = 1 if selection_delta > 0 else -1 if selection_delta < 0 else 0
        romd_sign = 1 if romd_delta > 0 else -1 if romd_delta < 0 else 0
        rows.append({
            "seed_index": int(seed_order.get(seed, fallback_index)),
            "right_minus_left_direct_selection_r": selection_delta,
            "right_minus_left_total_return_pct": values["total_return_pct"],
            "right_minus_left_max_drawdown_pct": values["max_drawdown_pct"],
            "right_minus_left_return_over_max_drawdown": romd_delta,
            "right_minus_left_expected_value_r": values["expected_value_r"],
            "selection_r_sign": selection_sign,
            "romd_sign": romd_sign,
        })
    if not rows:
        return None
    rows.sort(key=lambda row: int(row["seed_index"]))

    def _rank_corr(x_key: str, y_key: str) -> float | None:
        frame = pd.DataFrame({
            "x": [row[x_key] for row in rows],
            "y": [row[y_key] for row in rows],
        }).dropna()
        if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
            return None
        value = frame["x"].corr(frame["y"], method="spearman")
        return None if pd.isna(value) else float(value)

    non_tie_rows = [
        row for row in rows
        if int(row["selection_r_sign"]) != 0 and int(row["romd_sign"]) != 0
    ]
    selection_up = [row for row in rows if int(row["selection_r_sign"]) > 0]
    selection_up_romd_up = [row for row in selection_up if int(row["romd_sign"]) > 0]
    return {
        "left_arm_id": left.arm_id,
        "right_arm_id": right.arm_id,
        "left": left.name,
        "right": right.name,
        "n": int(len(rows)),
        "selection_r_positive_count": int(len(selection_up)),
        "selection_r_positive_romd_positive_count": int(len(selection_up_romd_up)),
        "selection_r_positive_romd_nonpositive_count": int(len(selection_up) - len(selection_up_romd_up)),
        "sign_concordant_count": int(sum(
            int(row["selection_r_sign"]) == int(row["romd_sign"])
            for row in non_tie_rows
        )),
        "sign_discordant_count": int(sum(
            int(row["selection_r_sign"]) != int(row["romd_sign"])
            for row in non_tie_rows
        )),
        "sign_non_tie_n": int(len(non_tie_rows)),
        "selection_r_delta_vs_romd_spearman": _rank_corr(
            "right_minus_left_direct_selection_r",
            "right_minus_left_return_over_max_drawdown",
        ),
        "selection_r_delta_vs_return_spearman": _rank_corr(
            "right_minus_left_direct_selection_r",
            "right_minus_left_total_return_pct",
        ),
        "seed_rows": rows,
    }


def _yearly_same_seed_comparison(
    *, seed_yearly_frame: pd.DataFrame, left: StrategyComparisonArm, right: StrategyComparisonArm,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if seed_yearly_frame.empty:
        return rows
    paired = seed_yearly_frame[
        seed_yearly_frame["arm_id"].isin((left.arm_id, right.arm_id))
    ].copy()
    for year, group in paired.groupby("year", sort=True):
        pivot = group.pivot(index="seed", columns="arm_id", values="return_pct")
        if left.arm_id not in pivot.columns or right.arm_id not in pivot.columns:
            continue
        pivot = pivot[[left.arm_id, right.arm_id]].dropna()
        if pivot.empty:
            continue
        delta = pivot[right.arm_id].to_numpy(dtype=float) - pivot[left.arm_id].to_numpy(dtype=float)
        stats = _distribution_stats(delta)
        tie = np.isclose(delta, 0.0, rtol=0.0, atol=1e-12)
        rows.append({
            "year": int(year),
            "left_arm_id": left.arm_id,
            "right_arm_id": right.arm_id,
            "left": left.name,
            "right": right.name,
            "n": int(len(delta)),
            "left_gt_right_count": int(np.sum((delta < 0) & ~tie)),
            "right_gt_left_count": int(np.sum((delta > 0) & ~tie)),
            "tie_count": int(np.sum(tie)),
            **{f"right_minus_left_{key}": value for key, value in stats.items()},
        })
    return rows


def _paired_comparison_summaries(
    *, contract: dict[str, Any], settings, seed_frame: pd.DataFrame, seed_yearly_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    stochastic = _contract_stochastic_arms(contract, settings)
    resolved_seeds = tuple(int(value) for value in contract.get("resolved_seeds") or ())
    summaries: list[dict[str, Any]] = []
    for spec in _contract_paired_contrasts(contract, settings, stochastic):
        left = spec["left"]
        right = spec["right"]
        summaries.append({
            "contrast_id": spec["contrast_id"],
            "description": spec["description"],
            "left_arm_id": left.arm_id,
            "right_arm_id": right.arm_id,
            "left": left.name,
            "right": right.name,
            "romd_same_seed": _same_seed_metric_comparison(
                seed_frame=seed_frame, left=left, right=right,
                metric_key="return_over_max_drawdown",
            ),
            "direct_selection_r_same_seed": _same_seed_metric_comparison(
                seed_frame=seed_frame, left=left, right=right,
                metric_key="direct_selection_r",
                require_same_parameter_runtime_universe=True,
            ),
            "selection_r_to_strategy": _paired_seed_translation_diagnostic(
                settings=settings, seed_frame=seed_frame, left=left, right=right,
                resolved_seeds=resolved_seeds,
            ),
            "romd_distribution": _pairwise_distribution_comparison(
                seed_frame=seed_frame, left=left, right=right,
            ),
            "yearly_same_seed": _yearly_same_seed_comparison(
                seed_yearly_frame=seed_yearly_frame, left=left, right=right,
            ),
        })
    return summaries


def _upgrade_derived_report_summary(
    summary: dict[str, Any],
    *,
    seed_frame: pd.DataFrame,
    seed_yearly_frame: pd.DataFrame | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    contract = dict(summary.get("contract") or {})
    profile_id = str(contract.get("profile_id") or "").strip()
    if not profile_id:
        return summary
    settings = get_strategy_comparison_settings(profile_id)
    stochastic = _contract_stochastic_arms(contract, settings)
    resolved_seeds = tuple(int(value) for value in contract.get("resolved_seeds") or ())
    if resolved_seeds:
        _validate_seed_results_frame(seed_frame, stochastic_arms=stochastic, seeds=resolved_seeds)
    upgraded = dict(summary)
    contract["report_schema_version"] = ROBUSTNESS_SCHEMA_VERSION
    upgraded["contract"] = contract
    # Historical summaries may not have persisted the yearly frame here; preserve their existing annual section.
    paired = _paired_comparison_summaries(
        contract=contract,
        settings=settings,
        seed_frame=seed_frame,
        seed_yearly_frame=(pd.DataFrame() if seed_yearly_frame is None else seed_yearly_frame),
    )
    upgraded["paired_comparisons"] = paired
    legacy = paired[0] if len(paired) == 1 else None
    upgraded["romd_distribution_comparison"] = None if legacy is None else legacy.get("romd_distribution")
    upgraded["romd_same_seed_comparison"] = None if legacy is None else legacy.get("romd_same_seed")
    upgraded["direct_selection_r_same_seed_comparison"] = None if legacy is None else legacy.get("direct_selection_r_same_seed")
    upgraded["selection_r_to_strategy_same_seed_translation"] = None if legacy is None else legacy.get("selection_r_to_strategy")
    upgraded["schema_version"] = ROBUSTNESS_SCHEMA_VERSION
    upgraded["common_strategy_report"] = _build_common_report_payload(
        summary=upgraded,
        seed_frame=seed_frame,
        seed_yearly_frame=(pd.DataFrame() if seed_yearly_frame is None else seed_yearly_frame),
        run_root=run_root,
    )
    upgraded["report_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return upgraded


def _robustness_summary(
    *, contract: dict[str, Any], fixed_results: dict[str, dict[str, Any]],
    seed_frame: pd.DataFrame, seed_yearly_frame: pd.DataFrame,
    run_root: Path | None = None,
) -> dict[str, Any]:
    settings = get_strategy_comparison_settings(str(contract["profile_id"]))
    current_robustness = get_strategy_multi_seed_robustness_settings(str(contract.get("robustness_id") or contract["profile_id"]))
    fixed, current_stochastic = _robustness_arms(settings, current_robustness)
    stochastic = _contract_stochastic_arms(contract, settings)
    # Current execution uses current roles; historical report refresh uses contract-pinned arms.
    if {arm.arm_id for arm in current_stochastic} != {arm.arm_id for arm in stochastic}:
        fixed = tuple(
            settings.arms[str(dict(item).get("arm_id"))]
            for item in contract.get("fixed_arms") or ()
        )
    mean_rows: list[dict[str, Any]] = []
    for arm in fixed:
        metrics = dict(fixed_results[arm.arm_id]["metrics"])
        metrics.setdefault("direct_selection_r", 0.0)
        mean_rows.append({
            "arm_id": arm.arm_id, "name": arm.name, "type": "Fixed", "n": 1,
            **{key: metrics.get(key) for _label, key, _unit in MEAN_METRICS},
            **{key: metrics.get(key) for key in COMMON_STRATEGY_REPORT_METRIC_KEYS},
        })
    for arm in stochastic:
        rows = seed_frame[seed_frame["arm_id"] == arm.arm_id].copy()
        enriched_rows = _backfill_seed_common_metrics_from_attribution(
            rows, seed_yearly_frame=seed_yearly_frame, run_root=run_root, contract=contract
        )
        mean_rows.append({
            "arm_id": arm.arm_id, "name": arm.name, "type": "Multi-seed", "n": int(len(rows)),
            **{key: _mean_or_none(rows[key]) for _label, key, _unit in MEAN_METRICS},
            **{key: _mean_metric(enriched_rows, key) for key in COMMON_STRATEGY_REPORT_METRIC_KEYS},
        })

    by_arm_id = {row["arm_id"]: row for row in mean_rows}
    references = dict(contract.get("romd_reference_baselines") or {})
    min_ref_id = str(dict(references.get("min") or {}).get("arm_id") or "")
    full_ref_id = str(dict(references.get("full") or {}).get("arm_id") or "")
    min_baseline = by_arm_id.get(min_ref_id)
    full_baseline = by_arm_id.get(full_ref_id)
    if min_baseline is None:
        raise ValueError("multi-seed summary缺少config指定的Min benchmark baseline")
    if "full" in references and full_baseline is None:
        raise ValueError("multi-seed summary缺少config指定的Full benchmark baseline")

    def same_seed_beats_count(arm_id: str, reference_arm_id: str) -> int | None:
        if not reference_arm_id:
            return None
        left = seed_frame.loc[
            seed_frame["arm_id"].astype(str) == str(arm_id), ["seed", "return_over_max_drawdown"]
        ].rename(columns={"return_over_max_drawdown": "left"})
        right = seed_frame.loc[
            seed_frame["arm_id"].astype(str) == str(reference_arm_id), ["seed", "return_over_max_drawdown"]
        ].rename(columns={"return_over_max_drawdown": "right"})
        if left.empty or right.empty:
            return None
        paired = left.merge(right, on="seed", how="inner")
        if paired.empty:
            return None
        return int(np.sum(pd.to_numeric(paired["left"], errors="coerce") > pd.to_numeric(paired["right"], errors="coerce")))

    romd_rows: list[dict[str, Any]] = []
    for row in mean_rows:
        if row["type"] == "Fixed":
            value = float(row["return_over_max_drawdown"])
            romd_rows.append({
                "arm_id": row["arm_id"], "name": row["name"], "n": 1,
                "mean": value, "median": value, "std": None, "cv": None,
                "min": value, "p25": None, "p75": None, "max": value,
                "beats_min_count": None, "beats_full_count": None,
            })
            continue
        values = pd.to_numeric(
            seed_frame.loc[seed_frame["arm_id"] == row["arm_id"], "return_over_max_drawdown"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        stats = _distribution_stats(values)
        mean = float(stats["mean"])
        romd_rows.append({
            "arm_id": row["arm_id"], "name": row["name"], "n": int(len(values)),
            **stats,
            "cv": (None if math.isclose(mean, 0.0, abs_tol=1e-12) else float(float(stats["std"]) / abs(mean))),
            "beats_min_count": same_seed_beats_count(row["arm_id"], min_ref_id),
            "beats_full_count": (
                None if full_baseline is None else same_seed_beats_count(row["arm_id"], full_ref_id)
            ),
        })

    yearly_statistics: list[dict[str, Any]] = []
    if bool(dict(contract.get("execution_options") or {}).get("yearly_report", True)):
        for arm in fixed:
            for item in fixed_results[arm.arm_id].get("yearly", []):
                yearly_statistics.append({
                    "arm_id": arm.arm_id, "name": arm.name, "type": "Fixed", "n": 1,
                    "year": int(item["year"]), "is_complete_year": bool(item["is_complete_year"]),
                    "mean": float(item["return_pct"]), "median": float(item["return_pct"]),
                    "std": None, "min": float(item["return_pct"]), "p25": None, "p75": None, "max": float(item["return_pct"]),
                })
        for arm in stochastic:
            arm_yearly = seed_yearly_frame[seed_yearly_frame["arm_id"] == arm.arm_id].copy()
            for year, group in arm_yearly.groupby("year", sort=True):
                values = pd.to_numeric(group["return_pct"], errors="coerce").dropna().to_numpy(dtype=float)
                stats = _distribution_stats(values)
                yearly_statistics.append({
                    "arm_id": arm.arm_id, "name": arm.name, "type": "Multi-seed", "n": int(len(values)),
                    "year": int(year), "is_complete_year": bool(group["is_complete_year"].astype(bool).all()),
                    **stats,
                })

    paired = _paired_comparison_summaries(
        contract=contract,
        settings=settings,
        seed_frame=seed_frame,
        seed_yearly_frame=seed_yearly_frame,
    )
    legacy = paired[0] if len(paired) == 1 else None
    flattened_yearly = [
        {"contrast_id": item["contrast_id"], **row}
        for item in paired for row in item.get("yearly_same_seed") or []
    ]
    payload = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "status": "RESULT_AVAILABLE", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract, "mean_strategy_metrics": mean_rows,
        "romd_statistics": romd_rows,
        "paired_comparisons": paired,
        "romd_distribution_comparison": None if legacy is None else legacy.get("romd_distribution"),
        "romd_same_seed_comparison": None if legacy is None else legacy.get("romd_same_seed"),
        "direct_selection_r_same_seed_comparison": None if legacy is None else legacy.get("direct_selection_r_same_seed"),
        "selection_r_to_strategy_same_seed_translation": None if legacy is None else legacy.get("selection_r_to_strategy"),
        "yearly_statistics": yearly_statistics,
        "yearly_same_seed_comparison": flattened_yearly,
    }
    payload["common_strategy_report"] = _build_common_report_payload(
        summary=payload,
        seed_frame=seed_frame,
        seed_yearly_frame=seed_yearly_frame,
        run_root=run_root,
    )
    return payload





















def _print_report_tables(summary: dict[str, Any]) -> None:
    print("\n" + render_multi_seed_robustness_report(summary, target="console").rstrip())



def show_multi_seed_robustness_status(*, robustness_id: str | None = None) -> None:
    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    settings = get_strategy_comparison_settings(cfg.profile_id)
    status = collect_artifact_status(settings=settings)
    fixed, stochastic = _robustness_arms(settings, cfg)
    model_arms = (
        _model_seed_sensitive_arms(settings, cfg, tuple(stochastic))
        if cfg.benchmark_id is not None else tuple(stochastic)
    )
    plan_text, plan = _render_robustness_execution_plan(
        cfg=cfg, settings=settings, status=status, fixed_arms=fixed, stochastic_arms=stochastic
    )
    print("\n" + plan_text)
    if plan["comparison_period"] is None or plan["overall_status"] != "READY":
        print("Fingerprint       ：前置完成後依正式param identity與共同期間解析")
        return
    benchmark_bindings = dict(plan.get("benchmark_param_bindings") or {})
    resolved_benchmark_bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for unit, item in benchmark_bindings.items():
        path = Path(item["path"])
        manifest_path = Path(item["manifest_path"])
        if not path.is_file() or not manifest_path.is_file():
            print("Fingerprint       ：benchmark strategy param前置完成後解析")
            return
        resolved_benchmark_bindings[unit] = {
            **item,
            "sha256": compute_strategy_param_scientific_sha256(path),
            "evaluation_sha256": resolve_strategy_param_evaluation_identity_sha256(
                path,
                evaluation_mode=str(item["evaluation_mode"]),
                start_date=str(plan["comparison_period"]["start"]),
                end_date=str(plan["comparison_period"]["end"]),
            ),
            "file_sha256": compute_file_sha256(path),
            "manifest_sha256": compute_file_sha256(manifest_path),
        }
    contract = build_multi_seed_robustness_contract(
        robustness_id=cfg.robustness_id, comparison_period=dict(plan["comparison_period"]),
        artifact_identities=dict(status.get("artifact_identities") or {}),
        benchmark_parameter_identities=_benchmark_identity_payload(resolved_benchmark_bindings),
    )
    run_root = _run_root(contract)
    existing, _existing_yearly, unit_states = _resolve_existing_scientific_unit_states(
        run_root=run_root,
        contract=contract,
        stochastic_arms=tuple(stochastic),
        reuse_completed=bool(cfg.reuse_completed),
    )
    seeds = tuple(int(value) for value in contract["resolved_seeds"])
    scientific_reuse_count = sum(
        1 for state in unit_states.values()
        if state.get("action") == RESULT_ACTION_REUSE
    )
    print(f"Fingerprint       ：{contract['fingerprint']}")
    ready_attribution = _attribution_ready_units(
        run_root, stochastic_arms=tuple(model_arms), seeds=seeds, fingerprint=str(contract["fingerprint"])
    ) if cfg.keep_attribution_source else set()
    print(f"已完成seed結果    ：{scientific_reuse_count}/{len(stochastic) * cfg.seed_count}")
    if cfg.keep_attribution_source:
        print(f"Attribution工件   ：{len(ready_attribution)}/{len(model_arms) * cfg.seed_count}")
    print("永久輸出：")
    print(f"- {project_relative_display_path(run_root / MANIFEST_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / SEED_RESULTS_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / SEED_YEARLY_RESULTS_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / SUMMARY_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / REPORT_FILENAME, project_root=PROJECT_ROOT)}")
    if cfg.keep_attribution_source:
        print(f"- {project_relative_display_path(_attribution_source_root(run_root), project_root=PROJECT_ROOT)}")


def show_latest_multi_seed_robustness_report(*, robustness_id: str | None = None) -> None:
    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    latest = (PROJECT_ROOT / cfg.output_root / LATEST_FILENAME).resolve()
    if not latest.is_file():
        raise FileNotFoundError("尚無Multiple-seed robustness最新結果")
    pointer = _read_json(latest)
    report = (PROJECT_ROOT / str(pointer["report_path"])).resolve()
    summary_path = (
        PROJECT_ROOT / str(pointer.get("summary_path") or "")
    ).resolve()
    if summary_path.is_file():
        summary = _read_json(summary_path)
        needs_upgrade = (
            int(summary.get("schema_version") or 0) < ROBUSTNESS_SCHEMA_VERSION
            or "paired_comparisons" not in summary
            or "direct_selection_r_same_seed_comparison" not in summary
            or "selection_r_to_strategy_same_seed_translation" not in summary
            or "common_strategy_report" not in summary
        )
        seed_results_path = summary_path.parent / SEED_RESULTS_FILENAME
        seed_yearly_results_path = summary_path.parent / SEED_YEARLY_RESULTS_FILENAME
        seed_frame = _load_seed_results(seed_results_path)
        seed_yearly_frame = _load_seed_yearly_results(seed_yearly_results_path)
        if (
            needs_upgrade
            and not seed_frame.empty
            and "direct_selection_r" in seed_frame.columns
        ):
            upgraded = _upgrade_derived_report_summary(
                summary,
                seed_frame=seed_frame,
                seed_yearly_frame=seed_yearly_frame,
                run_root=summary_path.parent,
            )
            _write_json(summary_path, upgraded)
            atomic_write_text(
                report, render_multi_seed_robustness_report(upgraded, target="markdown")
            )
    if not summary_path.is_file():
        raise FileNotFoundError(f"最新robustness摘要不存在: {summary_path}")
    summary = _read_json(summary_path)
    atomic_write_text(
        report, render_multi_seed_robustness_report(summary, target="markdown")
    )
    _print_report_tables(summary)


def _cleanup_unit_artifacts(
    *, model_dir: Path, research_dir: Path, keep_checkpoints: bool, keep_scores: bool,
    score_source: str,
) -> None:
    if str(score_source) != "selection_point_in_time":
        if not keep_checkpoints:
            shutil.rmtree(model_dir, ignore_errors=True)
        if not keep_scores:
            shutil.rmtree(research_dir, ignore_errors=True)
        return

    # Selection PIT把fold checkpoints與aggregate scores放在同一個isolated root。
    # retention knobs仍維持獨立語意：scores可只留top-level aggregate；
    # checkpoints可只留fold tree，不因目錄共置而互相綁定。
    if not keep_checkpoints:
        shutil.rmtree(model_dir / "folds", ignore_errors=True)
    if not keep_scores:
        for filename in (
            SELECTION_POINT_IN_TIME_SCORE_FILENAME,
            SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
        ):
            try:
                (model_dir / filename).unlink()
            except FileNotFoundError:
                pass
    if not keep_checkpoints and not keep_scores:
        shutil.rmtree(model_dir, ignore_errors=True)
    if not keep_scores:
        shutil.rmtree(research_dir, ignore_errors=True)


def _manifest_progress_payload(
    *,
    manifest: dict[str, Any],
    completed: set[tuple[str, int]],
    total_units: int,
    current_training: dict[str, Any] | None,
    futures: dict[Future, dict[str, Any]] | None = None,
    training_futures: dict[Future, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = dict(manifest)
    payload["status"] = "RUNNING"
    payload["completed_units"] = int(len(completed))
    payload["total_units"] = int(total_units)
    payload["completed_unit_keys"] = sorted(
        _unit_key(arm_id, seed) for arm_id, seed in completed
    )
    payload["current_training"] = current_training
    active_trainings = []
    for meta in (training_futures or {}).values():
        item = {
            "dl_id": str(meta["dl_id"]),
            "name": str(meta["arm"].name),
            "seed": int(meta["seed"]),
            "seed_order": int(meta["seed_order"]),
            "source_order": int(meta["source_order"]),
            "source_count": int(meta["source_count"]),
            "replay_arm_ids": [arm.arm_id for _arm_order, arm in meta["replay_arms"]],
        }
        pit_progress = read_trainer_pit_progress(
            Path(str(meta["research_dir"])).resolve() / "train.log"
        )
        if pit_progress is not None:
            item["pit_saved_folds"] = int(pit_progress[0])
            item["pit_expected_folds"] = int(pit_progress[1])
        active_trainings.append(item)
    payload["active_trainings"] = active_trainings
    payload["active_replays"] = [
        {
            "arm_id": str(meta["arm_id"]),
            "name": str(meta["name"]),
            "seed": int(meta["seed"]),
            "seed_order": int(meta["seed_order"]),
            "arm_order": int(meta["arm_order"]),
        }
        for meta in (futures or {}).values()
    ]
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def run_multi_seed_robustness(
    *,
    robustness_id: str | None = None,
    confirm: bool = True,
    model_upstream_preparer: Callable[[], int] | None = None,
) -> dict[str, Any]:
    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    if not cfg.enabled:
        raise RuntimeError("Multiple-seed robustness目前由config關閉")
    settings = get_strategy_comparison_settings(cfg.profile_id)
    fixed_arms, stochastic_arms = _robustness_arms(settings, cfg)
    model_arms = (
        _model_seed_sensitive_arms(settings, cfg, tuple(stochastic_arms))
        if cfg.benchmark_id is not None else tuple(stochastic_arms)
    )
    strategy_only_arms = _strategy_only_benchmark_arms(tuple(stochastic_arms), tuple(model_arms))
    status = collect_artifact_status(settings=settings)
    plan_text, plan = _render_robustness_execution_plan(
        cfg=cfg, settings=settings, status=status, fixed_arms=fixed_arms, stochastic_arms=stochastic_arms
    )
    print("\n" + plan_text)
    if plan["overall_status"] == "BLOCKED":
        raise RuntimeError(
            "Multiple-seed robustness前置BLOCKED；存在無法由正式builder確定性補建的依賴。"
        )
    if confirm:
        try:
            raw = input("👉 按 Enter 執行全部前置、訓練與回放；輸入 0 返回：").strip().lower()
        except EOFError:
            return {}
        if raw in {"0", "q", "quit", "exit"}:
            return {}
        if raw not in {"", "1"}:
            print("輸入無效，本次不執行。")
            return {}

    if bool(plan.get("model_upstream_prepare_required")):
        if model_upstream_preparer is None:
            raise RuntimeError(
                "Multiple-seed robustness需要canonical model-training provider補建Dataset／Target，"
                "但目前呼叫端未提供upstream preparer；請由apps/research.py正式入口執行。"
            )
        upstream_code = int(model_upstream_preparer() or 0)
        if upstream_code != 0:
            raise RuntimeError(
                "Multiple-seed robustness canonical model upstream自動補建失敗: "
                f"returncode={upstream_code}"
            )

    status = collect_artifact_status(settings=settings)
    post_upstream_rows, post_upstream_blockers = _model_upstream_rows(status)
    post_upstream_pending = [
        row for row in post_upstream_rows if row[0] in {"BUILD", "REBUILD", "RESUME"}
    ]
    if post_upstream_blockers or post_upstream_pending:
        detail = [*post_upstream_blockers, *(row[2] for row in post_upstream_pending)]
        raise RuntimeError(
            "Multiple-seed robustness模型上游工件在自動補建後仍未就緒: "
            + "；".join(detail)
        )

    # Re-read the same shared Strategy preparation owner after upstream production.
    # It alone resolves the current/planned comparison period for both single- and
    # multi-seed flows; robustness must not maintain a second period resolver.
    status = collect_preparation_status(
        project_root=PROJECT_ROOT,
        settings=settings,
    )
    comparison_start, comparison_end = _comparison_period_from_status(status)
    status = prepare_strategy_parameter_artifacts(
        project_root=PROJECT_ROOT,
        settings=settings,
        status=status,
        required_source_ids=tuple(plan["required_param_sources"]),
        status_refresher=lambda: collect_preparation_status(
            project_root=PROJECT_ROOT,
            settings=settings,
        ),
    )
    resolved_params = dict(status.get("resolved_arm_parameter_paths") or {})
    missing_params = sorted(
        arm.arm_id for arm in fixed_arms
        if not resolved_params.get(arm.arm_id) or not Path(resolved_params[arm.arm_id]).is_file()
    )
    if missing_params:
        raise RuntimeError(
            "Multiple-seed robustness策略參數前置完成後仍缺arm binding: "
            + ", ".join(missing_params)
        )
    benchmark_bindings = _prepare_benchmark_strategy_parameter_artifacts(
        settings=settings, robustness=cfg, benchmark_arms=tuple(stochastic_arms),
        comparison_start=str(comparison_start), comparison_end=str(comparison_end),
    )
    status = dict(status)
    status["comparison_period"] = {"start": comparison_start, "end": comparison_end}
    contract = build_multi_seed_robustness_contract(
        robustness_id=cfg.robustness_id,
        comparison_period={"start": comparison_start, "end": comparison_end},
        artifact_identities=dict(status.get("artifact_identities") or {}),
        benchmark_parameter_identities=_benchmark_identity_payload(benchmark_bindings),
    )
    run_root = _run_root(contract)
    model_root = _model_work_root(contract)
    run_root.mkdir(parents=True, exist_ok=True)
    if cfg.reuse_completed:
        completed_run = _validate_completed_run(
            run_root, expected_contract=contract, backfill_integrity=True
        )
        if completed_run is not None:
            summary = dict(completed_run["summary"])
            seed_frame = completed_run["seed_frame"]
            seed_yearly_frame = completed_run["seed_yearly_frame"]
            needs_upgrade = (
                int(summary.get("schema_version") or 0) < ROBUSTNESS_SCHEMA_VERSION
                or "paired_comparisons" not in summary
                or "common_strategy_report" not in summary
            )
            if needs_upgrade:
                summary = _upgrade_derived_report_summary(
                    summary, seed_frame=seed_frame,
                    seed_yearly_frame=seed_yearly_frame, run_root=run_root,
                )
                _write_json(run_root / SUMMARY_FILENAME, summary)
                atomic_write_text(
                    run_root / REPORT_FILENAME,
                    render_multi_seed_robustness_report(summary, target="markdown"),
                )
                manifest_payload = dict(completed_run["manifest"])
                manifest_payload["durable_artifacts"] = _build_durable_result_artifacts(run_root)
                manifest_payload["report_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_root / MANIFEST_FILENAME, manifest_payload)
            print(
                styled_workflow_status("[ROBUSTNESS REUSE]")
                + f" fingerprint={contract['fingerprint']} | completed scientific result"
            )
            _print_report_tables(summary)
            return summary
    model_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / MANIFEST_FILENAME
    seed_results_path = run_root / SEED_RESULTS_FILENAME
    seed_yearly_results_path = run_root / SEED_YEARLY_RESULTS_FILENAME
    manifest = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "comparison_period": {"start": comparison_start, "end": comparison_end},
        "seed_results_path": project_relative_display_path(seed_results_path, project_root=PROJECT_ROOT),
        "seed_yearly_results_path": project_relative_display_path(
            seed_yearly_results_path, project_root=PROJECT_ROOT
        ),
    }
    _write_json(manifest_path, manifest)

    started_total = time.perf_counter()
    seeds = tuple(int(value) for value in contract["resolved_seeds"])
    try:
        existing, existing_yearly, scientific_unit_states = (
            _resolve_existing_scientific_unit_states(
                run_root=run_root,
                contract=contract,
                stochastic_arms=tuple(stochastic_arms),
                reuse_completed=bool(cfg.reuse_completed),
            )
        )
        fixed_results = _run_fixed_baselines(
            settings=settings, status=status, run_root=run_root, fixed_arms=fixed_arms
        )
    except BaseException as exc:
        manifest.update({
            "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_stage": "resume_preflight_or_fixed_baseline",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resumable": True,
        })
        _write_json(manifest_path, manifest)
        print(
            styled_workflow_status("[FAILED]")
            + " Multiple-seed robustness前置回放階段已停止；可由同一入口接續。"
            + f" manifest={project_relative_display_path(manifest_path, project_root=PROJECT_ROOT)}"
        )
        raise
    scientific_completed = {
        unit
        for unit, state in scientific_unit_states.items()
        if str(state.get("action") or "") == RESULT_ACTION_REUSE
    }
    model_arm_ids = {arm.arm_id for arm in model_arms}
    strategy_only_completed = {unit for unit in scientific_completed if unit[0] not in model_arm_ids}
    attribution_ready = (
        _attribution_ready_units(
            run_root,
            stochastic_arms=tuple(model_arms),
            seeds=seeds,
            fingerprint=str(contract["fingerprint"]),
        )
        if cfg.keep_attribution_source
        else {unit for unit in scientific_completed if unit[0] in model_arm_ids}
    )
    completed = set(strategy_only_completed | (scientific_completed & attribution_ready))
    rows = (
        [row for row in existing.to_dict("records") if (str(row.get("arm_id")), int(row.get("seed"))) in scientific_completed]
        if not existing.empty and cfg.reuse_completed else []
    )
    yearly_rows = (
        [row for row in existing_yearly.to_dict("records") if (str(row.get("arm_id")), int(row.get("seed"))) in scientific_completed]
        if cfg.reuse_completed and not existing_yearly.empty else []
    )
    existing_by_unit = {
        (str(row.get("arm_id")), int(row.get("seed"))): dict(row)
        for row in rows
    }

    # C61/C58 are per-seed scientific baselines.  Their durable seed results are
    # reusable; the replay directory itself is transient and is rebuilt only when
    # an unfinished model arm actually needs baseline_reuse_dir.
    benchmark_baseline_dirs: dict[tuple[str, int], str] = {}
    stochastic_order = {arm.arm_id: index for index, arm in enumerate(stochastic_arms, start=1)}
    model_replay_pending = {
        (arm.arm_id, int(seed))
        for seed in seeds
        for arm in model_arms
        if (arm.arm_id, int(seed)) not in completed
    }
    required_baseline_units: set[tuple[str, int]] = set()
    for model_arm in model_arms:
        baseline_arm = _resolve_same_seed_baseline_arm(
            settings=settings, strategy_only_arms=tuple(strategy_only_arms), model_arm=model_arm
        )
        for seed in seeds:
            if (model_arm.arm_id, int(seed)) in model_replay_pending:
                required_baseline_units.add((baseline_arm.arm_id, int(seed)))

    strategy_only_actions: dict[tuple[str, int], str] = {}
    strategy_results_changed = False
    for seed_order, seed in enumerate(seeds, start=1):
        for arm in strategy_only_arms:
            unit_identity = (arm.arm_id, int(seed))
            binding = benchmark_bindings[unit_identity]
            output_dir = run_root / "work" / "benchmark_baselines" / _unit_key(arm.arm_id, int(seed))
            baseline_context_required = unit_identity in required_baseline_units
            baseline_context_available = (
                _strategy_only_baseline_context_available(
                    output_dir=output_dir, settings=settings, arm=arm, binding=binding,
                    comparison_start=comparison_start, comparison_end=comparison_end,
                )
                if baseline_context_required
                else False
            )
            action = _strategy_only_baseline_action(
                scientific_result_exists=unit_identity in scientific_completed,
                baseline_context_required=baseline_context_required,
                baseline_context_available=baseline_context_available,
            )
            strategy_only_actions[unit_identity] = action
            if action == "REUSE":
                if baseline_context_available and unit_identity in required_baseline_units:
                    benchmark_baseline_dirs[unit_identity] = str(output_dir.resolve())
                continue

            result = _run_strategy_only_benchmark_unit(
                settings=settings,
                arm=arm,
                seed=int(seed),
                seed_order=int(seed_order),
                arm_order=int(stochastic_order[arm.arm_id]),
                params_path=Path(binding["path"]),
                comparison_start=str(comparison_start),
                comparison_end=str(comparison_end),
                output_dir=output_dir,
                strategy_param_sha256=str(binding["sha256"]),
                strategy_param_manifest_sha256=str(binding["manifest_sha256"]),
                param_evaluation_mode=str(binding["evaluation_mode"]),
            )
            benchmark_baseline_dirs[unit_identity] = str(output_dir.resolve())
            if action == "REBUILD_CONTEXT":
                continue

            rows = [
                row for row in rows
                if (str(row.get("arm_id")), int(row.get("seed"))) != unit_identity
            ]
            rows.append(_result_row(result))
            yearly_rows = [
                row for row in yearly_rows
                if (str(row.get("arm_id")), int(row.get("seed"))) != unit_identity
            ]
            yearly_rows.extend(result["yearly"])
            scientific_completed.add(unit_identity)
            completed.add(unit_identity)
            strategy_results_changed = True
    if strategy_results_changed:
        seed_frame_checkpoint = pd.DataFrame(rows)
        yearly_frame_checkpoint = pd.DataFrame(yearly_rows)
        if not seed_frame_checkpoint.empty:
            _write_seed_results(seed_results_path, seed_frame_checkpoint)
        if not yearly_frame_checkpoint.empty:
            _write_seed_yearly_results(seed_yearly_results_path, yearly_frame_checkpoint)

    total_units = len(stochastic_arms) * len(seeds)
    training_sources = _training_source_groups(stochastic_arms, settings=settings)
    training_unit_count = len(training_sources) * len(seeds)
    done_units = len(completed)
    print("\n" + render_title("Multiple-seed robustness 執行"))
    print(
        f"Seeds={len(seeds)} | stochastic arms={len(stochastic_arms)} | replay units={total_units} | "
        f"unique DL sources={len(training_sources)} | train units={training_unit_count}"
    )
    print(f"GPU train workers={cfg.gpu_train_workers} | CPU replay workers={cfg.cpu_replay_workers}")
    if cfg.keep_attribution_source:
        expected_model_units = {
            (arm.arm_id, int(seed)) for arm in model_arms for seed in seeds
        }
        completed_model_units = scientific_completed & expected_model_units
        pending_observations = expected_model_units - scientific_completed
        missing_attr = completed_model_units - attribution_ready
        print(
            f"Compact attribution source={len(attribution_ready)}/{len(expected_model_units)} ready"
            + (f" | scientific observation pending={len(pending_observations)}" if pending_observations else "")
            + (f" | attribution rebuild={len(missing_attr)}" if missing_attr else "")
        )
    color_enabled = console_color_enabled()
    progress = InlineProgress()
    training_progress = FixedProgressBlock()
    min_ref_arm = str(dict(dict(contract.get("romd_reference_baselines") or {}).get("min") or {}).get("arm_id") or "")

    def print_event(text: str) -> None:
        training_progress.clear()
        progress.print_line(text)

    def progress_update(text: str) -> None:
        if training_futures:
            print_event(text)
            return
        training_progress.clear()
        if cfg.console_mode == "verbose":
            progress.print_line(text)
            return
        progress.update(text)

    current_training: dict[str, Any] | None = None
    active_trainers: dict[str, subprocess.Popen] = {}
    active_trainers_lock = Lock()
    replay_executor = ThreadPoolExecutor(
        max_workers=int(cfg.cpu_replay_workers),
        thread_name_prefix="strategy-replay",
    )
    training_executor = ThreadPoolExecutor(
        max_workers=int(cfg.gpu_train_workers),
        thread_name_prefix="strategy-gpu-train",
    )
    replay_futures: dict[Future, dict[str, Any]] = {}
    training_futures: dict[Future, dict[str, Any]] = {}
    ready_replays: deque[tuple[dict[str, Any], dict[str, Any]]] = deque()
    pending_trainings = _training_units(
        seeds=seeds,
        stochastic_arms=stochastic_arms,
        settings=settings,
        completed=completed,
        model_root=model_root,
        run_root=run_root,
        comparison_start=comparison_start,
        comparison_end=comparison_end,
        reuse_completed=cfg.reuse_completed,
        benchmark_id=cfg.benchmark_id,
        checkpoint_cache_root=cfg.checkpoint_cache_root,
    )
    training_cleanup_targets = []
    for meta in pending_trainings:
        meta["trainer_registry"] = active_trainers
        meta["trainer_registry_lock"] = active_trainers_lock
        meta["trainer_key"] = _unit_key(str(meta["dl_id"]), int(meta["seed"]))
        training_cleanup_targets.append({
            "model_dir": Path(str(meta["model_dir"])),
            "research_dir": Path(str(meta["research_dir"])),
            "score_source": str(settings.dl_sources[str(meta["dl_id"])].score_source),
        })
    trained_artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    queued_replay_units: set[tuple[str, int]] = set()

    displayed_done_units = 0
    for seed_order, seed in enumerate(seeds, start=1):
        for arm_order, arm in enumerate(stochastic_arms, start=1):
            unit_key = (arm.arm_id, int(seed))
            strategy_action = strategy_only_actions.get(unit_key)
            if unit_key in completed:
                displayed_done_units += 1
            if strategy_action == "RUN_SCIENTIFIC":
                print_event(
                    "[" + styled_workflow_status("DONE") + f" {displayed_done_units}/{total_units}]"
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name}"
                )
            elif strategy_action == "REBUILD_CONTEXT":
                print_event(
                    styled_workflow_status("[BASELINE CONTEXT REBUILD]")
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name} | scientific result reuse"
                )
            elif unit_key in completed:
                print_event(
                    "[" + styled_workflow_status("REUSE") + f" {displayed_done_units}/{total_units}]"
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name}"
                )
            elif unit_key in scientific_completed:
                print_event(
                    styled_workflow_status("[REBUILD ATTRIBUTION]")
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name} | scientific result reuse"
                )

    def current_training_snapshot() -> dict[str, Any] | None:
        if not training_futures:
            return None
        meta = next(iter(training_futures.values()))
        return {
            "dl_id": str(meta["dl_id"]),
            "name": meta["arm"].name,
            "seed": int(meta["seed"]),
            "seed_order": int(meta["seed_order"]),
            "source_order": int(meta["source_order"]),
            "source_count": int(meta["source_count"]),
            "replay_arm_ids": [arm.arm_id for _arm_order, arm in meta["replay_arms"]],
            "state": "TRAINING",
        }

    def write_progress_manifest() -> None:
        nonlocal manifest, current_training
        current_training = current_training_snapshot()
        manifest = _manifest_progress_payload(
            manifest=manifest,
            completed=completed,
            total_units=total_units,
            current_training=current_training,
            futures=replay_futures,
            training_futures=training_futures,
        )
        _write_json(manifest_path, manifest)

    write_progress_manifest()

    def harvest_replays() -> None:
        nonlocal done_units, rows, yearly_rows
        finished = [future for future in replay_futures if future.done()]
        for future in finished:
            meta = replay_futures.pop(future)
            result = future.result()
            unit_identity = (str(result["arm_id"]), int(result["seed"]))
            if unit_identity in scientific_completed:
                existing_row = existing_by_unit.get(unit_identity)
                if existing_row is None:
                    raise RuntimeError("attribution rebuild找不到既有scientific observation")
                unit_yearly = existing_yearly[
                    (existing_yearly["arm_id"].astype(str) == unit_identity[0])
                    & (existing_yearly["seed"].astype(int) == unit_identity[1])
                ] if not existing_yearly.empty else pd.DataFrame()
                _validate_rebuilt_observation(
                    existing_row=existing_row,
                    rebuilt=result,
                    existing_yearly=unit_yearly,
                )
                if cfg.keep_attribution_source:
                    _mark_attribution_unit_verified(
                        run_root,
                        arm_id=unit_identity[0],
                        seed=unit_identity[1],
                        expected_fingerprint=str(contract["fingerprint"]),
                        source="existing_scientific_observation_rebuild",
                    )
            else:
                rows = [
                    row for row in rows
                    if not (
                        str(row.get("arm_id")) == str(result["arm_id"])
                        and int(row.get("seed")) == int(result["seed"])
                    )
                ]
                rows.append(_result_row(result))
                _write_seed_results(seed_results_path, pd.DataFrame(rows))
                yearly_rows = [
                    row for row in yearly_rows
                    if not (
                        str(row.get("arm_id")) == str(result["arm_id"])
                        and int(row.get("seed")) == int(result["seed"])
                    )
                ]
                yearly_rows.extend(list(result.get("yearly") or []))
                _write_seed_yearly_results(seed_yearly_results_path, pd.DataFrame(yearly_rows))
                scientific_completed.add(unit_identity)
                existing_by_unit[unit_identity] = _result_row(result)
                if cfg.keep_attribution_source:
                    _mark_attribution_unit_verified(
                        run_root,
                        arm_id=unit_identity[0],
                        seed=unit_identity[1],
                        expected_fingerprint=str(contract["fingerprint"]),
                        source="new_scientific_observation",
                    )
            completed.add(unit_identity)
            done_units = len(completed)
            romd = float(result.get("return_over_max_drawdown"))
            min_row = next((
                row for row in rows
                if str(row.get("arm_id")) == min_ref_arm and int(row.get("seed")) == int(result["seed"])
            ), None)
            delta_min = None if min_row is None else romd - float(min_row.get("return_over_max_drawdown"))
            done_tag = "[" + styled_workflow_status("DONE") + f" {done_units}/{total_units}]"
            delta_text = (
                "ΔMin=N/A"
                if delta_min is None else terminal_signal(
                    f"ΔMin={delta_min:+.2f}",
                    signal_for_delta(delta_min, preference="higher"),
                    enabled=color_enabled,
                )
            )
            print_event(
                f"{done_tag} seed {meta['seed_order']}/{len(seeds)} | "
                f"對象 {meta['arm_order']}/{len(stochastic_arms)} {meta['name']} | "
                f"RoMD={romd:.2f} | {delta_text} | "
                f"train={format_elapsed(result.get('training_elapsed_sec', 0))} | "
                f"replay={format_elapsed(result.get('replay_elapsed_sec', 0))} | "
                f"total={format_elapsed(time.perf_counter()-started_total)}"
            )
        if finished:
            write_progress_manifest()

    def submit_ready_replays() -> None:
        while ready_replays and len(replay_futures) < int(cfg.cpu_replay_workers):
            replay_job, meta = ready_replays.popleft()
            meta["submitted_at"] = time.perf_counter()
            future = replay_executor.submit(_replay_one_unit, replay_job)
            replay_futures[future] = meta
            progress_update(
                paint("[REPLAY QUEUED]", "cyan", enabled=color_enabled, bold=True)
                + f" seed {meta['seed_order']}/{len(seeds)} | "
                f"對象 {meta['arm_order']}/{len(stochastic_arms)} {meta['name']} "
                + f"| CPU running={len(replay_futures)}/{cfg.cpu_replay_workers}"
            )
        if ready_replays or replay_futures:
            write_progress_manifest()

    def queue_replay_if_ready(
        *,
        arm_order: int,
        arm: StrategyComparisonArm,
        seed: int,
        seed_order: int,
    ) -> None:
        unit_identity = (arm.arm_id, int(seed))
        if unit_identity in completed or unit_identity in queued_replay_units:
            return
        required_dl_ids = resolve_arm_runtime_dl_source_ids(settings, arm)
        source_keys = tuple((dl_id, int(seed)) for dl_id in required_dl_ids)
        if any(key not in trained_artifacts for key in source_keys):
            return
        baseline_arm = _resolve_same_seed_baseline_arm(
            settings=settings, strategy_only_arms=tuple(strategy_only_arms), model_arm=arm
        )
        baseline_dir = benchmark_baseline_dirs.get((baseline_arm.arm_id, int(seed)))
        if not baseline_dir:
            raise RuntimeError(
                f"model benchmark arm缺少same-seed baseline replay: {baseline_arm.arm_id}/seed={seed}"
            )
        source_artifacts = {
            dl_id: dict(trained_artifacts[(dl_id, int(seed))])
            for dl_id in required_dl_ids
        }
        primary = source_artifacts[str(arm.dl_id)]
        runtime_source_identities = {
            dl_id: {
                "selected_epoch": int(artifacts.get("selected_epoch", 0) or 0),
                "fold_count": artifacts.get("fold_count"),
                "model_sha256": str(artifacts.get("model_sha256") or "") or None,
                "score_sha256": str(artifacts.get("score_sha256") or "") or None,
                "score_available_from": str(artifacts.get("score_available_from") or "") or None,
                "score_available_through": str(artifacts.get("score_available_through") or "") or None,
            }
            for dl_id, artifacts in source_artifacts.items()
        }
        score_overrides = {
            dl_id: {
                "score_path": str(artifacts["score"]),
                "manifest_path": (
                    None if not artifacts.get("manifest") else str(artifacts["manifest"])
                ),
                "execution_start": str(artifacts.get("score_execution_start") or ""),
            }
            for dl_id, artifacts in source_artifacts.items()
        }
        unit = _unit_key(arm.arm_id, int(seed))
        pair_dir = run_root / "work" / "replay" / unit
        result_path = run_root / "work" / "results" / f"{unit}.json"
        replay_job = {
            "profile_id": settings.profile_id,
            "arm_id": arm.arm_id,
            "arm_order": int(arm_order),
            "seed": int(seed),
            "seed_order": int(seed_order),
            "params_path": str(benchmark_bindings[(arm.arm_id, int(seed))]["path"]),
            "strategy_param_sha256": str(benchmark_bindings[(arm.arm_id, int(seed))]["sha256"]),
            "strategy_param_manifest_sha256": str(benchmark_bindings[(arm.arm_id, int(seed))]["manifest_sha256"]),
            "param_evaluation_mode": str(benchmark_bindings[(arm.arm_id, int(seed))]["evaluation_mode"]),
            "comparison_start": comparison_start,
            "comparison_end": comparison_end,
            "baseline_dir": baseline_dir,
            "score_path": primary["score"],
            "score_manifest_path": primary.get("manifest"),
            "score_execution_start": primary["score_execution_start"],
            "score_overrides": score_overrides,
            "selected_epoch": primary["selected_epoch"],
            "fold_count": primary.get("fold_count"),
            "training_elapsed_sec": float(sum(
                float(artifacts.get("training_elapsed_sec", 0.0) or 0.0)
                for artifacts in source_artifacts.values()
            )),
            "model_sha256": primary.get("model_sha256"),
            "score_sha256": primary.get("score_sha256"),
            "runtime_source_identities": runtime_source_identities,
            "runtime_source_identity_sha256": canonical_json_sha256(runtime_source_identities),
            "model_prediction_metrics": dict(primary.get("model_prediction_metrics") or {}),
            "pair_dir": str(pair_dir),
            "result_path": str(result_path),
            "scientific_fingerprint": str(contract["fingerprint"]),
            "robustness_id": cfg.robustness_id,
            "keep_replay_details": cfg.keep_replay_details,
            "keep_attribution_source": cfg.keep_attribution_source,
            "attribution_dir": str(_attribution_unit_dir(run_root, arm.arm_id, int(seed))),
        }
        ready_replays.append((
            replay_job,
            {
                "arm_id": arm.arm_id,
                "name": arm.name,
                "seed": int(seed),
                "seed_order": int(seed_order),
                "arm_order": int(arm_order),
            },
        ))
        queued_replay_units.add(unit_identity)

    def harvest_trainings() -> None:
        finished = [future for future in training_futures if future.done()]
        for future in finished:
            meta = training_futures.pop(future)
            trained = future.result()
            artifacts = dict(trained["artifacts"])
            trained_artifacts[(str(meta["dl_id"]), int(meta["seed"]))] = artifacts
            fold_reuse = dict(artifacts.get("fold_reuse_summary") or {})
            built_folds = int(fold_reuse.get("built", 0) or 0)
            if trained["reused"]:
                tag = "[TRAIN ARTIFACT REUSE]"
            elif fold_reuse and built_folds == 0:
                tag = "[PIT RECOVER]"
            else:
                tag = "[TRAIN DONE]"
            fold_note = ""
            if fold_reuse:
                fold_note = (
                    " | folds="
                    f"reuse:{int(fold_reuse.get('exact_reuse', 0) or 0)}"
                    f"/rescore:{int(fold_reuse.get('local_checkpoint_rescore', 0) or 0)}"
                    f"/fitting-cache:{int(fold_reuse.get('fitting_identity_checkpoint_reuse', 0) or 0)}"
                    f"/migrate:{int(fold_reuse.get('legacy_migration', 0) or 0)}"
                    f"/built:{built_folds}"
                )
            print_event(
                styled_workflow_status(tag)
                + f" seed {meta['seed_order']}/{len(seeds)} | "
                f"模型來源 {meta['source_order']}/{meta['source_count']} {meta['dl_id']} "
                + f"| replay targets={len(meta['replay_arms'])} "
                + f"| epoch={artifacts['selected_epoch']} "
                + f"| elapsed={format_elapsed(artifacts.get('training_elapsed_sec', trained['wall_elapsed_sec']))}"
                + fold_note
                            )
            for arm_order, arm in meta["replay_arms"]:
                queue_replay_if_ready(
                    arm_order=int(arm_order),
                    arm=arm,
                    seed=int(meta["seed"]),
                    seed_order=int(meta["seed_order"]),
                )
        if finished:
            write_progress_manifest()

    def render_training_progress(now: float) -> None:
        if not training_futures:
            training_progress.clear()
            return
        active_metas = sorted(
            training_futures.values(),
            key=lambda item: (int(item["seed_order"]), int(item["source_order"])),
        )
        lines: list[str] = []
        for worker_index, meta in enumerate(active_metas, start=1):
            log_path = Path(str(meta["research_dir"])).resolve() / "train.log"
            unit_text = render_training_unit_progress(
                unit_id=str(meta["dl_id"]),
                source_index=int(meta["source_order"]),
                source_count=int(meta["source_count"]),
                elapsed_seconds=now - float(meta["submitted_at"]),
                pit_progress=read_trainer_pit_progress(log_path),
                epoch_progress=read_trainer_epoch_progress(log_path),
            )
            action = str(meta.get("display_action_tag") or "[TRAIN]")
            lines.append(
                paint(action, "cyan", enabled=color_enabled, bold=True)
                + f" worker {worker_index}/{cfg.gpu_train_workers}"
                + f" | seed {int(meta['seed_order'])}/{len(seeds)} ({int(meta['seed'])})"
                + " | "
                + unit_text
            )
        training_progress.update(lines)

    def submit_trainings() -> None:
        while pending_trainings and len(training_futures) < int(cfg.gpu_train_workers):
            meta = _pop_next_training_unit(pending_trainings, training_futures)
            meta["submitted_at"] = time.perf_counter()
            future = training_executor.submit(_train_one_unit, meta)
            training_futures[future] = meta
            # Fold state is owned by the canonical trainer log and parsed by
            # core.training_progress for both single- and multi-seed callers.
            meta["display_action_tag"] = "[TRAIN]"
        if training_futures:
            write_progress_manifest()
            render_training_progress(time.perf_counter())

    next_print = time.perf_counter() + float(cfg.progress_interval_seconds)
    try:
        submit_trainings()
        while pending_trainings or training_futures or ready_replays or replay_futures:
            harvest_replays()
            harvest_trainings()
            submit_ready_replays()
            submit_trainings()
            now = time.perf_counter()
            if now >= next_print and (training_futures or replay_futures or ready_replays):
                if training_futures:
                    render_training_progress(now)
                else:
                    progress_update(
                        paint("[REPLAY PROGRESS]", "cyan", enabled=color_enabled, bold=True)
                        + f" CPU replay={len(replay_futures)}/{cfg.cpu_replay_workers}"
                        + f" | queued={len(ready_replays)}"
                        + f" | completed={done_units}/{total_units}"
                        + f" | total={format_elapsed(now-started_total)}"
                    )
                next_print = now + float(cfg.progress_interval_seconds)
            if pending_trainings or training_futures or ready_replays or replay_futures:
                time.sleep(0.25)
        harvest_replays()
        training_progress.clear()
    except BaseException as exc:
        terminate_registered_training_processes(active_trainers, active_trainers_lock)
        write_progress_manifest()
        manifest.update({
            "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_stage": "isolated_training_or_strategy_replay",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resumable": True,
        })
        _write_json(manifest_path, manifest)
        print_event(
            styled_workflow_status("[FAILED]")
            + " Multiple-seed robustness已停止；可修正後由同一入口接續。"
            + f" manifest={project_relative_display_path(manifest_path, project_root=PROJECT_ROOT)}"
        )
        raise
    finally:
        training_executor.shutdown(wait=True, cancel_futures=True)
        replay_executor.shutdown(wait=True, cancel_futures=False)

    # 同一DL source/seed可fan-out多個selector replay；必須等所有replay完成後才清除
    # 共用的isolated model/score，避免第一個arm完成就刪掉第二個arm仍需讀取的工件。
    for target in training_cleanup_targets:
        _cleanup_unit_artifacts(
            model_dir=target["model_dir"],
            research_dir=target["research_dir"],
            keep_checkpoints=cfg.keep_checkpoints,
            keep_scores=cfg.keep_scores,
            score_source=target["score_source"],
        )

    try:
        seed_frame = _load_seed_results(seed_results_path)
        _validate_seed_results_frame(
            seed_frame, stochastic_arms=stochastic_arms, seeds=seeds
        )
        _validate_seed_result_scientific_identities(seed_frame, contract=contract)
        expected = len(stochastic_arms) * len(seeds)
        if len(seed_frame) != expected:
            raise RuntimeError(
                f"multi-seed robustness結果不完整: expected={expected}, actual={len(seed_frame)}"
            )
        seed_yearly_frame = _load_seed_yearly_results(seed_yearly_results_path)
        _validate_seed_yearly_results_frame(
            seed_yearly_frame, stochastic_arms=stochastic_arms, seeds=seeds,
            expected_years=_comparison_years(comparison_start, comparison_end),
            require_complete_units=True,
        )
        attribution_index_path = None
        if cfg.keep_attribution_source:
            expected_attribution = len(model_arms) * len(seeds)
            ready_attribution = _attribution_ready_units(
                run_root,
                stochastic_arms=tuple(model_arms),
                seeds=seeds,
                fingerprint=str(contract["fingerprint"]),
            )
            if len(ready_attribution) != expected_attribution:
                raise RuntimeError(
                    f"compact attribution source不完整: expected={expected_attribution}, actual={len(ready_attribution)}"
                )
            attribution_index_path = _write_attribution_index_manifest(
                run_root,
                contract=contract,
                stochastic_arms=tuple(model_arms),
                seeds=seeds,
            )
        _write_scientific_observation_manifest(
            run_root,
            contract=contract,
            seed_frame=seed_frame,
            seed_yearly_frame=seed_yearly_frame,
            attribution_index_path=attribution_index_path,
        )
        summary = _robustness_summary(
            contract=contract, fixed_results=fixed_results, seed_frame=seed_frame,
            seed_yearly_frame=seed_yearly_frame, run_root=run_root,
        )
        summary["elapsed_sec"] = round(time.perf_counter() - started_total, 3)
        _write_json(run_root / SUMMARY_FILENAME, summary)
        report_text = render_multi_seed_robustness_report(summary, target="markdown")
        atomic_write_text(run_root / REPORT_FILENAME, report_text)
        manifest.update({
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": summary["elapsed_sec"],
            "durable_artifacts": _build_durable_result_artifacts(run_root),
            "completed_seed_strategy_observations": int(len(seed_frame)),
            "attribution_source_units": (len(model_arms) * len(seeds) if cfg.keep_attribution_source else 0),
            "attribution_source_manifest_path": (
                None if attribution_index_path is None else project_relative_display_path(attribution_index_path, project_root=PROJECT_ROOT)
            ),
            "summary_path": project_relative_display_path(run_root / SUMMARY_FILENAME, project_root=PROJECT_ROOT),
            "report_path": project_relative_display_path(run_root / REPORT_FILENAME, project_root=PROJECT_ROOT),
        })
        _write_json(manifest_path, manifest)
        latest_path = (PROJECT_ROOT / cfg.output_root / LATEST_FILENAME).resolve()
        _write_json(latest_path, {
            "fingerprint": contract["fingerprint"],
            "report_path": project_relative_display_path(run_root / REPORT_FILENAME, project_root=PROJECT_ROOT),
            "summary_path": project_relative_display_path(run_root / SUMMARY_FILENAME, project_root=PROJECT_ROOT),
            "attribution_source_manifest_path": (
                None if attribution_index_path is None else project_relative_display_path(attribution_index_path, project_root=PROJECT_ROOT)
            ),
        })
    except BaseException as exc:
        manifest.update({
            "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_stage": "summary_or_report_finalization",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resumable": True,
        })
        _write_json(manifest_path, manifest)
        print(
            styled_workflow_status("[FAILED]")
            + " Multiple-seed robustness彙總階段失敗；seed結果已保留，可由同一入口接續。"
            + f" manifest={project_relative_display_path(manifest_path, project_root=PROJECT_ROOT)}"
        )
        raise
    selection_score_tree = any(
        str(settings.dl_sources[str(arm.dl_id)].score_source) == "selection_point_in_time"
        for arm in model_arms
    )
    if not cfg.keep_replay_details:
        shutil.rmtree(run_root / "work" / "replay", ignore_errors=True)
    shutil.rmtree(run_root / "work" / "results", ignore_errors=True)
    shutil.rmtree(run_root / "work" / "baselines", ignore_errors=True)
    if selection_score_tree or not cfg.keep_scores:
        shutil.rmtree(run_root / "work" / "training", ignore_errors=True)
    if not cfg.keep_checkpoints and (not selection_score_tree or not cfg.keep_scores):
        shutil.rmtree(model_root, ignore_errors=True)
    if not cfg.keep_replay_details and (selection_score_tree or not cfg.keep_scores):
        shutil.rmtree(run_root / "work", ignore_errors=True)
    if progress.inline:
        progress.finish()
    _print_report_tables(summary)
    print(f"\n總耗時：{format_elapsed(summary['elapsed_sec'])}")
    cleanup_tag = styled_workflow_status("暫存清理")
    print(
        f"{cleanup_tag}：checkpoints={'保留' if cfg.keep_checkpoints else '已清除'}｜"
        f"scores={'保留' if cfg.keep_scores else '已清除'}｜"
        f"replay details={'保留' if cfg.keep_replay_details else '已清除'}"
        + ("｜fitting-identity checkpoint cache=保留" if cfg.checkpoint_cache_root else "")
    )
    print("永久工件：")
    permanent_paths = [
        manifest_path,
        _scientific_observation_manifest_path(run_root),
        seed_results_path,
        seed_yearly_results_path,
    ]
    if cfg.keep_attribution_source:
        permanent_paths.append(_attribution_source_root(run_root) / MANIFEST_FILENAME)
    permanent_paths.extend([run_root / SUMMARY_FILENAME, run_root / REPORT_FILENAME])
    for path in permanent_paths:
        print("- " + project_relative_display_path(path, project_root=PROJECT_ROOT))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--replay-worker", default=None)
    args, _unknown = parser.parse_known_args(argv)
    if args.replay_worker:
        job = _read_json(Path(args.replay_worker).resolve())
        _replay_one_unit(job)
        return 0
    raise RuntimeError("此module只供apps/research.py或內部worker呼叫")


if __name__ == "__main__":
    raise SystemExit(main())
