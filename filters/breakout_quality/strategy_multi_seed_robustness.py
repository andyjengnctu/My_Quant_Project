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
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
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
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_workflow_settings,
)
from config.strategy_compare import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
)
from config.execution_policy import (
    DEFAULT_FIXED_RISK,
    DEFAULT_MAX_POSITION_CAP_PCT,
)
from core.file_integrity import canonical_json_sha256
from core.console_report import (
    console_color_enabled,
    paint,
    project_relative_display_path,
    render_section,
    render_table,
    render_title,
)
from core.display_common import InlineProgress
from core.report_metrics import (
    CORE_STRATEGY_RESULT_METRICS,
    EXECUTION_STRATEGY_RESULT_METRICS,
    R_MODEL_PREDICTION_METRICS,
    R_SELECTION_TRANSLATION_METRICS,
    ROBUSTNESS_ROMD_DISTRIBUTION_METRICS,
    ROBUSTNESS_SEED_DELTA_METRICS,
    ROBUSTNESS_YEARLY_DELTA_METRICS,
    ROBUSTNESS_YEARLY_DISTRIBUTION_METRICS,
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
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    resolve_filter_output_dir,
)
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    DAILY_RANKER_OOS_SCORE_FILENAME,
    CONTINUOUS_RANKER_SCORE_FILENAME,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_score_table_from_path,
)
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
)
from filters.breakout_quality.strategy_compare_engine import run_comparison
from filters.breakout_quality.strategy_compare_contracts import COMPARISON_MODE_SCORE_RANKING
from filters.breakout_quality.strategy_compare_replay import (
    _load_reusable_no_filter_baseline,
    run_standalone_baseline,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from services.optimizer.strategy_param_service import (
    ensure_robustness_benchmark_strategy_parameter_artifact,
    inspect_robustness_benchmark_strategy_parameter_artifact,
)
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    best_worst_signals,
    signal_for_delta,
    styled_signal,
    terminal_signal,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.strategy_comparison import (
    _arm_runtime_spec,
    _resolved_ranking_options,
    _find_reusable_baseline_source,
    _load_direct_selection_r,
    collect_artifact_status,
    render_strategy_aggregate_report,
)
from filters.breakout_quality.strategy_compare_reporting import capacity_summary
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from filters.breakout_quality.artifact_dependency_registry import (
    collect_model_upstream_preparation_plan,
)
from filters.breakout_quality.strategy_compare_preparation_status import (
    collect_artifact_status as collect_strategy_preparation_status,
)
from filters.breakout_quality.strategy_compare_preparation import (
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
ROBUSTNESS_SCHEMA_VERSION = 11
ROBUSTNESS_SCIENTIFIC_CONTRACT_VERSION = 5
ROBUSTNESS_MODEL_ARTIFACT_CONTRACT_VERSION = 2
TRAINER_TERMINATION_GRACE_SECONDS = 5.0

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
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON根節點必須是object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


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
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean_metric(frame: pd.DataFrame, key: str) -> float | None:
    if key not in frame.columns:
        return None
    values = pd.to_numeric(frame[key], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _report_settings_for_contract(contract: dict[str, Any], settings):
    contract_ids = {
        str(dict(item or {}).get("arm_id") or "").strip()
        for item in (*tuple(contract.get("fixed_arms") or ()), *tuple(contract.get("stochastic_arms") or ()))
    }
    contract_ids.discard("")
    if not contract_ids:
        return settings
    missing_ids = contract_ids.difference(settings.arms)
    if missing_ids:
        raise ValueError(
            "robustness report contract引用不存在的arm: " + ", ".join(sorted(missing_ids))
        )
    # Report order belongs to the Compare Suite, not to execution roles.  Fixed/benchmark
    # membership may change without changing the human comparison matrix/order.
    selected = {
        arm.arm_id: replace(arm, enabled=True)
        for arm in settings.enabled_arms
        if arm.arm_id in contract_ids
    }
    selected_ids = set(selected)
    selected_contrasts = {
        contrast_id: contrast
        for contrast_id, contrast in settings.contrasts.items()
        if contrast.enabled and contrast.left in selected_ids and contrast.right in selected_ids
    }
    return replace(settings, arms=selected, contrasts=selected_contrasts)


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
    rng = random.Random(int(generator_seed))
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < int(seed_count):
        value = int(rng.randrange(1, 2**31 - 1))
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


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
    if not fixed or not stochastic:
        raise ValueError("multi-seed robustness需要fixed baseline與stochastic arms")
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
    *, settings, robustness, benchmark_arms: tuple[StrategyComparisonArm, ...], comparison_end: str
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
                    paint(f"[PARAM {result['action']}]", "green", enabled=console_color_enabled(), bold=True)
                    + f" seed={seed} {binding['family']}/{binding['evaluation_mode']} "
                    + f"{binding['param_policy']}"
                )
            bindings[(arm.arm_id, int(seed))] = {
                **binding,
                "path": Path(result["path"]),
                "manifest_path": Path(result["manifest_path"]),
                # ``sha256`` is the replay-scientific strategy-param identity.
                # Byte-level publication integrity stays separately auditable.
                "sha256": str(result["scientific_sha256"]),
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
            expected_params_sha256=str(binding["sha256"]),
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

    if not scientific_result_exists:
        return "RUN_SCIENTIFIC"
    if baseline_context_required and not baseline_context_available:
        return "REBUILD_CONTEXT"
    return "REUSE"


def _run_strategy_only_benchmark_unit(
    *, settings, arm: StrategyComparisonArm, seed: int, seed_order: int, arm_order: int,
    params_path: Path, comparison_start: str, comparison_end: str, output_dir: Path,
    strategy_param_sha256: str, strategy_param_manifest_sha256: str,
    param_evaluation_mode: str,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    all_off = arm.rule_policy == "all_off"
    started = time.perf_counter()
    payload = run_standalone_baseline(
        project_root=PROJECT_ROOT,
        dataset=settings.dataset,
        params_path=str(params_path),
        param_policy=resolve_strategy_comparison_arm_param_policy(settings, arm),
        max_positions=settings.max_positions,
        enable_rotation=settings.rotation == "on",
        optional_entry_filter_policy=(
            OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
        ),
        output_dir_override=output_dir,
        comparison_start_date=str(comparison_start),
        comparison_end_date=str(comparison_end),
        quiet=True,
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
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


def _arm_training_dl_ids(settings, arm: StrategyComparisonArm) -> tuple[str, ...]:
    """Return every stochastic model source that must share the arm seed.

    Dual-model safety arms are a single scientific strategy unit.  Their primary
    opportunity source and secondary safety source must therefore be trained with
    the same generated seed; fixing the secondary model at Seed42 would only test
    partial robustness of the arm.
    """

    primary = str(arm.dl_id or "").strip()
    if not primary:
        raise ValueError(f"stochastic arm缺少dl_id: {arm.arm_id}")
    source_ids = [primary]
    safety_dl_id = str(dict(arm.dl_runtime_options or {}).get("safety_dl_id") or "").strip()
    if safety_dl_id:
        if safety_dl_id not in settings.dl_sources:
            raise ValueError(f"stochastic arm引用不存在的safety_dl_id: {arm.arm_id}/{safety_dl_id}")
        source_ids.append(safety_dl_id)
    unique = tuple(dict.fromkeys(source_ids))
    score_sources = {str(settings.dl_sources[dl_id].score_source) for dl_id in unique}
    expected = (
        "selection_point_in_time"
        if settings.profile_id in {"selection_pit", "extending_window_oos", "extending_window_rolling"}
        else "continuous_ranker_oos"
    )
    if score_sources != {expected}:
        raise ValueError(
            "stochastic arm的primary/secondary score source與robustness階段不一致: "
            f"arm={arm.arm_id}, expected={expected}, actual={sorted(score_sources)}"
        )
    return unique


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
        for dl_id in _arm_training_dl_ids(settings, arm):
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


def _model_upstream_rows(settings, stochastic_arms) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Render model upstream work from the shared Research preparation contract."""

    rows: list[tuple[str, str, str]] = []
    blockers: list[str] = []
    seen: set[tuple[str, str]] = set()
    for dl_id, _arm_entries in _training_source_groups(tuple(stochastic_arms), settings=settings):
        dl = settings.dl_sources[str(dl_id)]
        key = (str(dl.filter_id), str(dl.experiment_profile))
        if key in seen:
            continue
        seen.add(key)
        plan = collect_model_upstream_preparation_plan(
            PROJECT_ROOT,
            filter_id=str(dl.filter_id),
            model_architecture=str(dl.model_architecture),
            experiment_profile=str(dl.experiment_profile),
            dataset=str(settings.dataset),
            max_tickers=0,
        )
        pending = [item for item in plan.actions if item.action != "REUSE"]
        if not pending:
            profile = get_breakout_quality_experiment_profile(str(dl.experiment_profile))
            description = (
                "重用canonical Dataset／source OHLCV truth；daily windows與固定target由"
                "canonical trainer即時計算，不需要legacy market-set工件"
                if str(profile.training_sample_scope)
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                else "重用canonical Dataset／Continuous Target truth；isolated trainer不得建立新的Label／Target定義"
            )
            rows.append(("REUSE", f"model-upstream:{dl.experiment_profile}", description))
            continue
        if plan.blocked:
            reasons = [item.description for item in pending if item.action == "BLOCKED"]
            blockers.extend(reasons)
            rows.append((
                "BLOCKED",
                f"model-upstream:{dl.experiment_profile}",
                "；".join(reasons) or "Research dependency graph判定model upstream不可自動補建",
            ))
            continue
        action = "REBUILD" if any(item.action == "REBUILD" for item in pending) else (
            "RESUME" if any(item.action == "RESUME" for item in pending) else "BUILD"
        )
        rows.append((
            action,
            f"model-upstream:{dl.experiment_profile}",
            "；".join(item.description for item in pending)
            + "；確認後由canonical model-training producer依Research dependency graph補建並re-plan",
        ))
    return rows, blockers


def _comparison_period_from_upstream(settings, stochastic_arms, status: dict[str, Any]) -> tuple[str, str]:
    if settings.start_date is not None and settings.end_date is not None:
        start = pd.Timestamp(str(settings.start_date)).normalize()
        end = pd.Timestamp(str(settings.end_date)).normalize()
        if end < start:
            raise RuntimeError("Multiple-seed robustness設定期間不合法")
        return str(start.date()), str(end.date())

    runtime_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen_sources: set[tuple[str, str]] = set()
    for dl_id, _arm_entries in _training_source_groups(tuple(stochastic_arms), settings=settings):
        dl = settings.dl_sources[str(dl_id)]
        filter_id = str(dl.filter_id)
        source_key = (filter_id, str(dl.experiment_profile))
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        dataset = resolve_dataset_paths(
            resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
        )
        summary = _read_json(dataset.summary)
        if summary is None:
            raise RuntimeError(
                "Multiple-seed robustness無法讀取canonical Dataset summary: "
                + project_relative_display_path(dataset.summary, project_root=PROJECT_ROOT)
            )
        stored_dataset = str(summary.get("dataset") or "").strip().lower()
        expected_dataset = str(settings.dataset).strip().lower()
        if stored_dataset and stored_dataset != expected_dataset:
            raise RuntimeError(
                "Multiple-seed robustness Dataset profile不一致: "
                f"expected={expected_dataset}, actual={stored_dataset}"
            )
        source_range = dict(summary.get("source_data_date_range") or {})
        source_end = str(source_range.get("end") or "").strip()
        if not source_end:
            raise RuntimeError(
                "Multiple-seed robustness無法由Dataset解析Forward OOS期間: "
                + project_relative_display_path(dataset.summary, project_root=PROJECT_ROOT)
            )
        outer = resolve_breakout_quality_outer_policy(
            PROJECT_ROOT, source_data_end_date=source_end
        )
        configured_start = _source_point_in_time_score_start_date(
            settings=settings, dl_id=str(dl_id)
        )
        configured_end = _source_point_in_time_score_end_date(
            settings=settings, dl_id=str(dl_id)
        )
        period_start = pd.Timestamp(
            str(configured_start or outer["oos_start_date"])
        ).normalize()
        period_end = pd.Timestamp(
            str(
                outer["effective_oos_end_date"]
                if configured_end in (None, "") or str(configured_end).lower() == "auto"
                else configured_end
            )
        ).normalize()
        runtime_periods.append((period_start, period_end))
    if not runtime_periods:
        raise RuntimeError("Multiple-seed robustness沒有可解析Forward OOS期間的stochastic source")
    start = max(item[0] for item in runtime_periods)
    end = min(item[1] for item in runtime_periods)
    if end < start:
        raise RuntimeError("Multiple-seed robustness stochastic sources沒有共同Forward OOS期間")
    return str(start.date()), str(end.date())


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


def _pit_fold_count_for_period(start: str, end: str, fold_months: int) -> int:
    current = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    count = 0
    while current <= end_ts:
        count += 1
        current = current + pd.DateOffset(months=int(fold_months))
    return count


def _render_robustness_execution_plan(
    *, cfg, settings, status: dict[str, Any], fixed_arms, stochastic_arms
) -> tuple[str, dict[str, Any]]:
    # Current end-to-end benchmark uses fixed production consensus references only
    # as context. C61/C58/C59/C60 all require per-seed benchmark strategy params;
    # C59/C60 additionally retrain DL sources with the exact same seed.
    if cfg.benchmark_id is not None:
        model_arms = _model_seed_sensitive_arms(settings, cfg, tuple(stochastic_arms))
        required_sources = tuple(sorted({arm.param_source for arm in fixed_arms}))
    else:
        model_arms = tuple(stochastic_arms)
        required_sources = _required_parameter_sources(tuple(fixed_arms), tuple(stochastic_arms))
    param_rows, param_blockers = _parameter_plan_rows(
        settings=settings, status=status, required_sources=required_sources
    )
    upstream_rows, upstream_blockers = _model_upstream_rows(settings, model_arms)
    blockers = [*param_blockers, *upstream_blockers]
    upstream_pending = any(row[0] in {"BUILD", "REBUILD", "RESUME"} for row in upstream_rows)
    if upstream_pending and not upstream_blockers:
        start = end = None
        period_text = "待自動補建canonical Dataset／Target後解析"
        period_error = None
    else:
        try:
            start, end = _comparison_period_from_upstream(settings, model_arms, status)
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
    rows = [*param_rows, *benchmark_rows, *upstream_rows]
    for arm in fixed_arms:
        rows.append((
            "RUN/REUSE",
            arm.name,
            "Production finalists-agree consensus reference；固定顯示，不作same-seed paired baseline",
        ))
    model_ids = {arm.arm_id for arm in model_arms}
    for arm in stochastic_arms:
        if arm.arm_id in model_ids:
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
    pit_workload_lines: list[str] = [
        f"Benchmark ID       ：{cfg.benchmark_id or '-'}",
        f"Benchmark seeds    ：{','.join(str(v) for v in cfg.resolved_seeds)}",
        f"Strategy trials/fold：{cfg.strategy_trials_per_fold if cfg.strategy_trials_per_fold is not None else '-'}",
        f"策略Benchmark單元  ：{cfg.seed_count * len(stochastic_arms)}（{len(stochastic_arms)} per-seed strategy arms × {cfg.seed_count} seeds）",
        f"模型訓練單元       ：{cfg.seed_count * len(training_sources)}（{len(training_sources)} unique DL sources × {cfg.seed_count} seeds）",
    ]
    if settings.profile_id in {"selection_pit", "extending_window_oos", "extending_window_rolling"} and start is not None and end is not None:
        fold_counts = []
        for dl_id, _arm_entries in training_sources:
            fold_counts.append(
                1
                if _source_point_in_time_single_score_block(settings=settings, dl_id=str(dl_id))
                else _pit_fold_count_for_period(
                    start, end, _source_point_in_time_fold_months(settings=settings, dl_id=str(dl_id))
                )
            )
        if fold_counts:
            unique_fold_counts = sorted(set(fold_counts))
            fold_text = str(unique_fold_counts[0]) if len(unique_fold_counts) == 1 else "/".join(map(str, unique_fold_counts))
            pit_workload_lines += [
                f"PIT folds          ：{fold_text} folds / DL source / seed",
                f"PIT fold工作量     ：{cfg.seed_count * sum(fold_counts)} slots（實際新訓練會扣除REUSE／rescore／跨mode checkpoint重用）",
            ]
    color_enabled = console_color_enabled()
    action_colors = {
        "READY": "green", "REUSE": "green", "BUILD": "yellow", "REBUILD": "yellow",
        "CHECK": "yellow", "PREPARABLE": "yellow", "BLOCKED": "red", "RUN/REUSE": "cyan",
        "TRAIN+REPLAY": "cyan", "PARAM+REPLAY": "cyan",
    }
    colored_rows = [
        (paint(action, action_colors.get(str(action), "gray"), enabled=color_enabled, bold=True), item, description)
        for action, item, description in rows
    ]
    overall_color = {"READY": "green", "PREPARABLE": "yellow", "BLOCKED": "red"}.get(overall, "gray")
    lines = [
        render_title(f"{cfg.label} 本次執行計畫"),
        f"整體狀態          ：{paint(overall, overall_color, enabled=color_enabled, bold=True)}",
        f"比較階段          ：{settings.profile_label} ({settings.profile_id})",
        f"共同策略期間      ：{period_text}",
        f"Seed數量          ：{cfg.seed_count}",
        f"Seed generator    ：deterministic benchmark / generator_seed={cfg.seed_generator_seed}",
        f"GPU training      ：workers={cfg.gpu_train_workers}",
        f"CPU strategy replay：workers={cfg.cpu_replay_workers}",
        *pit_workload_lines,
        render_table(("動作", "項目", "說明"), colored_rows),
    ]
    return "\n".join(lines), {
        "overall_status": overall,
        "blockers": blockers,
        "required_param_sources": required_sources,
        "benchmark_param_bindings": benchmark_bindings,
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
                "runtime_dl_sources": [source_contract(dl_id) for dl_id in _arm_training_dl_ids(settings, arm)],
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
            "initial_checkpoint_cache_root": robustness.initial_checkpoint_cache_root,
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


_SEED_EXPANSION_NON_SCIENTIFIC_KEYS = frozenset({
    "benchmark_parameter_publication_identities",
    "model_artifact_fingerprint",
    "paired_contrasts",
    "report_schema_version",
    "label",
    "execution_options",
    "retention",
    "fingerprint",
})


def _benchmark_parameter_identities_for_seeds(
    identities: dict[str, Any], seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Return replay-scientific per-seed parameter identities only."""

    suffixes = tuple(f":seed={int(seed)}" for seed in seeds)
    result: dict[str, Any] = {}
    for key, raw in dict(identities or {}).items():
        if not str(key).endswith(suffixes):
            continue
        item = dict(raw or {})
        item.pop("manifest_sha256", None)
        item.pop("file_sha256", None)
        result[str(key)] = item
    return result


def _benchmark_parameter_identities_compatible(
    candidate_contract: dict[str, Any],
    current_contract: dict[str, Any],
    *,
    seeds: tuple[int, ...],
) -> bool:
    """Compare per-seed strategy-param identity with one safe legacy migration.

    New contracts store the runtime-scientific SHA.  Legacy contracts stored raw
    JSON byte SHA under the same field.  A legacy row can be upgraded only when its
    old SHA exactly equals the *current* parameter file SHA; exact bytes prove that
    the old observation used the same payload.  If a publication refresh already
    replaced those bytes, we deliberately refuse to guess.
    """

    candidate = _benchmark_parameter_identities_for_seeds(
        dict(candidate_contract.get("benchmark_parameter_artifact_identities") or {}), seeds
    )
    current = _benchmark_parameter_identities_for_seeds(
        dict(current_contract.get("benchmark_parameter_artifact_identities") or {}), seeds
    )
    current_publication = dict(
        current_contract.get("benchmark_parameter_publication_identities") or {}
    )
    if set(candidate) != set(current):
        return False
    for key in sorted(current):
        old = dict(candidate[key] or {})
        new = dict(current[key] or {})
        for field in ("family", "evaluation_mode", "param_policy"):
            if str(old.get(field) or "") != str(new.get(field) or ""):
                return False
        old_schema = str(old.get("identity_schema") or "").strip()
        new_schema = str(new.get("identity_schema") or "").strip()
        old_sha = _normalized_hash_value(old.get("sha256"))
        new_sha = _normalized_hash_value(new.get("sha256"))
        if old_schema:
            if old_schema != new_schema or old_sha != new_sha:
                return False
            continue
        # Pre-runtime-identity contract: sha256 was the raw JSON file SHA.
        current_file_sha = _normalized_hash_value(
            dict(current_publication.get(key) or {}).get("file_sha256")
        )
        if not current_file_sha or old_sha != current_file_sha:
            return False
    return True


def _rebind_legacy_strategy_param_identity_rows(
    frame: pd.DataFrame,
    *,
    source_contract: dict[str, Any],
    current_contract: dict[str, Any],
) -> pd.DataFrame:
    """Upgrade safely proven legacy raw-SHA rows to runtime-scientific SHA."""

    if frame.empty:
        return frame
    output = frame.copy()
    source_identities = dict(source_contract.get("benchmark_parameter_artifact_identities") or {})
    current_identities = dict(current_contract.get("benchmark_parameter_artifact_identities") or {})
    current_publication = dict(current_contract.get("benchmark_parameter_publication_identities") or {})
    for idx, row in output.iterrows():
        key = f"{str(row.get('arm_id') or '')}:seed={int(row.get('seed'))}"
        old = dict(source_identities.get(key) or {})
        new = dict(current_identities.get(key) or {})
        if not old or not new:
            continue
        if str(old.get("identity_schema") or "").strip():
            continue
        old_sha = _normalized_hash_value(row.get("strategy_param_sha256"))
        current_file_sha = _normalized_hash_value(
            dict(current_publication.get(key) or {}).get("file_sha256")
        )
        if old_sha and current_file_sha and old_sha == current_file_sha:
            output.at[idx, "strategy_param_sha256"] = str(new.get("sha256") or "")
    return output


def _seed_expansion_compatibility_payload(
    contract: dict[str, Any], *, seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Normalize a robustness contract for compatible observation reuse.

    Seed membership may stay the same (engineering-only fingerprint migration) or
    expand by strict prefix.  Replay-relevant parameter payload identity and every
    other scientific condition must remain identical; volatile benchmark-manifest
    publication metadata is intentionally excluded. OOS and Rolling remain isolated
    because robustness/profile/source contracts stay in this payload.
    """

    normalized = {
        str(key): value
        for key, value in dict(contract or {}).items()
        if str(key) not in _SEED_EXPANSION_NON_SCIENTIFIC_KEYS
        and str(key) not in {
            "seed_count",
            "resolved_seeds",
            "benchmark_parameter_artifact_identities",
        }
    }
    normalized["seed_count"] = len(seeds)
    normalized["resolved_seeds"] = [int(seed) for seed in seeds]
    return normalized


def _seed_expansion_source_run(
    *, contract: dict[str, Any], current_run_root: Path,
) -> tuple[Path, dict[str, Any], tuple[int, ...]] | None:
    """Find the largest completed compatible run reusable by this run.

    A source may have the same seed set (engineering-only fingerprint migration)
    or a strict seed prefix (N expansion).
    """

    current_seeds = tuple(int(value) for value in contract.get("resolved_seeds") or ())
    if not current_seeds:
        return None
    cfg = get_strategy_multi_seed_robustness_settings(str(contract["robustness_id"]))
    output_root = (PROJECT_ROOT / cfg.output_root).resolve()
    if not output_root.is_dir():
        return None
    best: tuple[Path, dict[str, Any], tuple[int, ...]] | None = None
    for candidate_root in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if candidate_root.resolve() == Path(current_run_root).resolve():
            continue
        validated = _validate_scientific_observation_manifest(
            candidate_root,
            backfill_legacy_completed=True,
        )
        if validated is None:
            continue
        candidate_contract = dict(validated["contract"])
        candidate_seeds = tuple(
            int(value) for value in candidate_contract.get("resolved_seeds") or ()
        )
        if not candidate_seeds or len(candidate_seeds) > len(current_seeds):
            continue
        if current_seeds[: len(candidate_seeds)] != candidate_seeds:
            continue
        if _seed_expansion_compatibility_payload(
            candidate_contract, seeds=candidate_seeds
        ) != _seed_expansion_compatibility_payload(
            contract, seeds=candidate_seeds
        ):
            continue
        if not _benchmark_parameter_identities_compatible(
            candidate_contract, contract, seeds=candidate_seeds
        ):
            continue
        if best is None or len(candidate_seeds) > len(best[2]):
            best = (candidate_root.resolve(), candidate_contract, candidate_seeds)
    return best


def _rebind_attribution_unit_for_seed_expansion(
    *, source_run_root: Path, destination_run_root: Path, arm_id: str, seed: int,
    source_fingerprint: str, destination_fingerprint: str, seed_order: int,
) -> None:
    source_payload = _read_attribution_unit_manifest(
        source_run_root,
        arm_id=arm_id,
        seed=seed,
        expected_fingerprint=source_fingerprint,
    )
    if source_payload is None:
        raise RuntimeError(
            f"seed expansion缺少已驗證compact attribution source: {arm_id}/seed={seed}"
        )
    source_dir = _attribution_unit_dir(source_run_root, arm_id, seed)
    destination_dir = _attribution_unit_dir(destination_run_root, arm_id, seed)
    shutil.rmtree(destination_dir, ignore_errors=True)
    shutil.copytree(source_dir, destination_dir)
    payload = _read_json(destination_dir / MANIFEST_FILENAME)
    payload["scientific_fingerprint"] = str(destination_fingerprint)
    payload["seed_order"] = int(seed_order)
    payload["scientific_observation_validation"] = {
        "status": "VERIFIED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"seed_expansion_reuse:{source_fingerprint}",
    }
    for role, raw_item in dict(payload.get("files") or {}).items():
        item = dict(raw_item or {})
        filename = Path(str(item.get("path") or f"{role}.csv.gz")).name
        destination = destination_dir / filename
        if not destination.is_file():
            raise RuntimeError(
                f"seed expansion attribution copy缺少檔案: {arm_id}/seed={seed}/{filename}"
            )
        item["path"] = project_relative_display_path(
            destination, project_root=PROJECT_ROOT
        )
        item["sha256"] = compute_file_sha256(destination)
        payload["files"][role] = item
    _write_json(destination_dir / MANIFEST_FILENAME, payload)


def _import_seed_expansion_results(
    *, contract: dict[str, Any], run_root: Path,
    stochastic_arms: tuple[StrategyComparisonArm, ...],
    model_arms: tuple[StrategyComparisonArm, ...],
    keep_attribution_source: bool,
) -> dict[str, Any] | None:
    """Import compatible completed observations without replaying old seeds."""

    source = _seed_expansion_source_run(contract=contract, current_run_root=run_root)
    if source is None:
        return None
    source_root, source_contract, source_seeds = source
    source_results = _load_seed_results(source_root / SEED_RESULTS_FILENAME)
    source_yearly = _load_seed_yearly_results(source_root / SEED_YEARLY_RESULTS_FILENAME)
    source_results = _rebind_legacy_strategy_param_identity_rows(
        source_results, source_contract=source_contract, current_contract=contract
    )
    _validate_seed_results_frame(
        source_results, stochastic_arms=stochastic_arms, seeds=source_seeds
    )
    expected_units = {
        (arm.arm_id, int(seed)) for arm in stochastic_arms for seed in source_seeds
    }
    actual_result_units = set() if source_results.empty else {
        (str(row.arm_id), int(row.seed)) for row in source_results.itertuples(index=False)
    }
    if actual_result_units != expected_units:
        raise RuntimeError("seed expansion來源缺少完整seed策略結果")
    actual_yearly_units = set() if source_yearly.empty else {
        (str(row.arm_id), int(row.seed)) for row in source_yearly.itertuples(index=False)
    }
    if expected_units - actual_yearly_units:
        raise RuntimeError("seed expansion來源缺少完整年度結果")

    source_fingerprint = str(source_contract.get("fingerprint") or "")
    destination_fingerprint = str(contract["fingerprint"])
    if keep_attribution_source:
        model_ids = {arm.arm_id for arm in model_arms}
        for arm_id, seed in sorted(expected_units):
            if arm_id not in model_ids:
                continue
            _rebind_attribution_unit_for_seed_expansion(
                source_run_root=source_root,
                destination_run_root=run_root,
                arm_id=arm_id,
                seed=int(seed),
                source_fingerprint=source_fingerprint,
                destination_fingerprint=destination_fingerprint,
                seed_order=list(contract["resolved_seeds"]).index(int(seed)) + 1,
            )

    current_results_path = run_root / SEED_RESULTS_FILENAME
    current_yearly_path = run_root / SEED_YEARLY_RESULTS_FILENAME
    current_results = _load_seed_results(current_results_path)
    current_yearly = _load_seed_yearly_results(current_yearly_path)
    merged_results = pd.concat([source_results, current_results], ignore_index=True)
    if not merged_results.empty:
        merged_results = merged_results.drop_duplicates(
            subset=["arm_id", "seed"], keep="last"
        )
    merged_yearly = pd.concat([source_yearly, current_yearly], ignore_index=True)
    if not merged_yearly.empty:
        merged_yearly = merged_yearly.drop_duplicates(
            subset=["arm_id", "seed", "year"], keep="last"
        )
    _write_seed_results(current_results_path, merged_results)
    _write_seed_yearly_results(current_yearly_path, merged_yearly)
    return {
        "source_fingerprint": source_fingerprint,
        "reused_seeds": [int(seed) for seed in source_seeds],
        "reused_observations": len(expected_units),
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
            for key in ("model_sha256", "score_sha256", "runtime_source_identity_sha256"):
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

    This READY manifest is the commit marker for reusable stochastic evidence.  It
    intentionally excludes summary/report presentation so later renderer refreshes,
    lifecycle retries, or a failed report finalization cannot make completed model
    observations disappear from cross-fingerprint reuse discovery.
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


def _load_legacy_scientific_observations_for_backfill(
    run_root: Path,
) -> dict[str, Any] | None:
    """Recover complete legacy observations independently of lifecycle status.

    Older runs used ``manifest.json`` both as RUNNING/FAILED lifecycle state and as
    the only discovery anchor for completed observations.  A later failed resume can
    therefore overwrite ``COMPLETED`` even though the expensive seed/yearly/compact
    attribution evidence is still complete.  Validate the evidence itself and use
    lifecycle metadata only to recover the original scientific contract.
    """

    lifecycle_path = Path(run_root) / MANIFEST_FILENAME
    if not lifecycle_path.is_file():
        return None
    try:
        lifecycle = _read_json(lifecycle_path)
        contract = dict(lifecycle.get("contract") or {})
        fingerprint = str(contract.get("fingerprint") or "")
        if not fingerprint or fingerprint != Path(run_root).name:
            return None
        durable = dict(lifecycle.get("durable_artifacts") or {})
        if durable and not _validate_durable_result_artifacts(
            run_root, durable, keys=SCIENTIFIC_DURABLE_RESULT_KEYS
        ):
            return None
        validated = _load_validated_scientific_observations(
            run_root, contract=contract, scientific_artifacts=None
        )
        return {"contract": contract, **validated}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _validate_scientific_observation_manifest(
    run_root: Path,
    *,
    backfill_legacy_completed: bool,
) -> dict[str, Any] | None:
    """Validate reusable scientific evidence without consulting run lifecycle status.

    New runs publish ``scientific_observations_manifest.json`` before derived report
    generation.  Legacy completed runs can be validated once through the previous
    durable-result contract and upgraded in place.  After the READY marker exists,
    later RUNNING/FAILED presentation attempts cannot invalidate these observations.
    """

    evidence_path = _scientific_observation_manifest_path(run_root)
    if not evidence_path.is_file():
        if not backfill_legacy_completed:
            return None
        legacy = _load_legacy_scientific_observations_for_backfill(run_root)
        if legacy is None:
            return None
        contract = dict(legacy["contract"])
        stochastic = tuple(legacy["stochastic_arms"])
        seeds = tuple(int(value) for value in legacy["seeds"])
        model_ids = {
            str(value) for value in tuple(contract.get("model_seed_sensitive_arm_ids") or ())
        }
        attribution_index_path = None
        if bool(dict(contract.get("retention") or {}).get("keep_attribution_source")):
            model_arms = tuple(arm for arm in stochastic if arm.arm_id in model_ids)
            attribution_index_path = _write_attribution_index_manifest(
                run_root,
                contract=contract,
                stochastic_arms=model_arms,
                seeds=seeds,
            )
        _write_scientific_observation_manifest(
            run_root,
            contract=contract,
            seed_frame=legacy["seed_frame"],
            seed_yearly_frame=legacy["seed_yearly_frame"],
            attribution_index_path=attribution_index_path,
        )

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
    require_derived_artifacts: bool = True,
) -> dict[str, Any] | None:
    """Validate one completed robustness result at the requested evidence layer.

    Exact-run presentation REUSE requires the derived summary/report. Compatible
    cross-fingerprint observation reuse only requires validated scientific seed
    observations plus compact attribution; stale/missing derived presentation must
    not trigger multi-hour model retraining.
    """

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
        summary: dict[str, Any] = {}
        if require_derived_artifacts:
            summary = _read_json(run_root / SUMMARY_FILENAME)
            summary_contract = dict(summary.get("contract") or {})
            if str(summary_contract.get("fingerprint") or "") != fingerprint:
                return None
            _validate_summary_seed_aggregates(summary, seed_frame=seed_frame, contract=contract)
            if not (run_root / REPORT_FILENAME).is_file():
                return None
        durable = dict(manifest.get("durable_artifacts") or {})
        if durable:
            integrity_keys = (
                tuple(DURABLE_RESULT_FILENAMES.keys())
                if require_derived_artifacts
                else SCIENTIFIC_DURABLE_RESULT_KEYS
            )
            if not _validate_durable_result_artifacts(
                run_root, durable, keys=integrity_keys
            ):
                return None
        elif backfill_integrity and require_derived_artifacts:
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


def _resolved_period(settings, status: dict[str, Any]) -> tuple[str, str]:
    period = dict(status.get("comparison_period") or {})
    start = str(period.get("start") or settings.start_date or "").strip()
    end = str(period.get("end") or settings.end_date or "").strip()
    if not start or not end:
        raise RuntimeError("multi-seed robustness無法解析共同策略比較期間")
    return start, end


def _validate_training_artifacts(
    *, arm: StrategyComparisonArm, dl_id: str, seed: int, model_dir: Path, research_dir: Path, settings,
    comparison_start: str, comparison_end: str,
) -> dict[str, Any]:
    dl = settings.dl_sources[str(dl_id)]
    if str(dl.score_source) == "selection_point_in_time":
        score = model_dir / SELECTION_POINT_IN_TIME_SCORE_FILENAME
        manifest_path = model_dir / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
        for label, path in (("PIT score", score), ("PIT manifest", manifest_path)):
            if not path.is_file():
                raise FileNotFoundError(f"multi-seed {label}工件不存在: {path}")
        manifest = _read_json(manifest_path)
        for field, expected in (
            ("filter_id", dl.filter_id),
            ("experiment_profile", dl.experiment_profile),
            ("model_architecture", dl.model_architecture),
        ):
            if str(manifest.get(field) or "") != str(expected):
                raise ValueError(f"multi-seed PIT manifest {field}不一致")
        if int(manifest.get("seed", -1)) != int(seed):
            raise ValueError("multi-seed PIT manifest seed不一致")
        expected_fold_months = _source_point_in_time_fold_months(
            settings=settings, dl_id=str(dl_id)
        )
        if int(manifest.get("fold_months", -1) or -1) != int(expected_fold_months):
            raise ValueError(
                "multi-seed PIT manifest fold_months不一致: "
                f"expected={expected_fold_months}, actual={manifest.get('fold_months')}"
            )
        expected_single_block = _source_point_in_time_single_score_block(
            settings=settings, dl_id=str(dl_id)
        )
        if bool(manifest.get("single_score_block", False)) != bool(expected_single_block):
            raise ValueError(
                "multi-seed PIT manifest single_score_block不一致: "
                f"expected={expected_single_block}, actual={manifest.get('single_score_block')}"
            )
        expected_anchor = _source_point_in_time_fold_anchor_date(
            settings=settings, dl_id=str(dl_id)
        )
        actual_anchor = str(manifest.get("fold_anchor_date") or "").strip() or None
        if expected_anchor is not None and actual_anchor != expected_anchor:
            raise ValueError(
                "multi-seed PIT manifest fold_anchor_date不一致: "
                f"expected={expected_anchor}, actual={actual_anchor}"
            )
        period = dict(manifest.get("score_period") or {})
        actual_start = pd.Timestamp(period.get("start")).strftime("%Y-%m-%d")
        actual_end = pd.Timestamp(period.get("end")).strftime("%Y-%m-%d")
        if actual_start != pd.Timestamp(comparison_start).strftime("%Y-%m-%d") or actual_end != pd.Timestamp(comparison_end).strftime("%Y-%m-%d"):
            raise ValueError(
                "multi-seed PIT只允許策略比較期間所需fold: "
                f"expected={comparison_start}~{comparison_end}, actual={actual_start}~{actual_end}"
            )
        table = load_selection_point_in_time_score_table_from_path(
            str(score), manifest_path=str(manifest_path)
        )
        folds = list(manifest.get("folds") or [])
        if not folds:
            raise ValueError("multi-seed PIT manifest沒有folds")
        epochs = [int(dict(item).get("selected_epoch", 0) or 0) for item in folds]
        if any(epoch < 1 for epoch in epochs):
            raise ValueError("multi-seed PIT fold selected_epoch不合法")
        return {
            "score": str(score.resolve()),
            "manifest": str(manifest_path.resolve()),
            "score_sha256": compute_file_sha256(score),
            "selected_epoch": int(round(float(np.median(epochs)))),
            "training_elapsed_sec": float(manifest.get("elapsed_sec", 0.0) or 0.0),
            "score_execution_start": str(table.attrs.get("available_from") or actual_start),
            "score_available_from": str(table.attrs.get("available_from") or actual_start),
            "score_available_through": str(table.attrs.get("available_through") or actual_end),
            "fold_count": int(manifest.get("fold_count", len(folds)) or len(folds)),
            "fold_reuse_summary": {
                str(key): int(value or 0)
                for key, value in dict(manifest.get("fold_reuse_summary") or {}).items()
            },
        }

    profile = get_breakout_quality_experiment_profile(dl.experiment_profile)
    score_name = (
        DAILY_RANKER_OOS_SCORE_FILENAME
        if str(profile.training_sample_scope) == "daily_eligible_stock_days"
        else CONTINUOUS_RANKER_SCORE_FILENAME
    )
    paths = {
        "model": model_dir / "model.pt", "manifest": model_dir / "manifest.json",
        "report": research_dir / CONTINUOUS_RANKER_REPORT_FILENAME, "score": research_dir / score_name,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"multi-seed {label}工件不存在: {path}")
    manifest = _read_json(paths["manifest"]); report = _read_json(paths["report"])
    for payload, label in ((manifest, "manifest"), (report, "report")):
        if str(payload.get("filter_id") or "") != dl.filter_id:
            raise ValueError(f"multi-seed {label} filter_id不一致")
        if str(payload.get("experiment_profile") or "") != dl.experiment_profile:
            raise ValueError(f"multi-seed {label} experiment profile不一致")
        if str(payload.get("model_architecture") or "") != dl.model_architecture:
            raise ValueError(f"multi-seed {label} architecture不一致")
    training = dict(report.get("training") or {})
    if int(training.get("seed", -1)) != int(seed):
        raise ValueError("multi-seed report seed不一致")
    outer_policy = dict(manifest.get("outer_oos_policy") or {})
    execution_start_raw = str(outer_policy.get("oos_start_date") or "").strip()
    if not execution_start_raw:
        raise ValueError("multi-seed manifest缺少outer_oos_policy.oos_start_date")
    execution_start = pd.Timestamp(execution_start_raw).strftime("%Y-%m-%d")
    score_table = load_continuous_ranker_oos_score_table_from_path(str(paths["score"]), dl.experiment_profile)
    score_available_from = str(score_table.attrs.get("available_from") or "")
    score_available_through = str(score_table.attrs.get("available_through") or "")
    if not score_available_from or not score_available_through:
        raise ValueError("multi-seed score table缺少日期範圍metadata")
    required_start = pd.Timestamp(comparison_start).strftime("%Y-%m-%d")
    required_end = pd.Timestamp(comparison_end).strftime("%Y-%m-%d")
    if execution_start > score_available_from:
        raise ValueError("multi-seed isolated score時間契約不一致")
    if pd.Timestamp(score_available_from) > pd.Timestamp(required_start):
        raise ValueError(
            "multi-seed score起始覆蓋不足: "
            f"required<={required_start}, actual={score_available_from}"
        )
    if pd.Timestamp(score_available_through) < pd.Timestamp(required_end):
        raise ValueError(
            "multi-seed score結束覆蓋不足: "
            f"required>={required_end}, actual={score_available_through}"
        )
    artifact_payload = {
        **{key: str(path.resolve()) for key, path in paths.items()},
        "model_sha256": compute_file_sha256(paths["model"]),
        "score_sha256": compute_file_sha256(paths["score"]),
        "selected_epoch": int(training.get("selected_epoch", 0) or 0),
        "training_elapsed_sec": float(report.get("elapsed_sec", 0.0) or 0.0),
        "score_execution_start": execution_start,
        "score_available_from": score_available_from,
        "score_available_through": score_available_through,
        "fold_count": None,
    }
    artifact_payload["model_prediction_metrics"] = _model_prediction_metrics_from_training_artifacts(artifact_payload)
    return artifact_payload



PIT_FOLD_MANIFEST_FILENAME = "manifest.json"
PIT_FOLD_MODEL_FILENAME = "model.pt"
INITIAL_CHECKPOINT_CACHE_MANIFEST_FILENAME = "cache_manifest.json"


def _initial_checkpoint_training_identity(manifest: dict[str, Any]) -> str:
    """Fingerprint only the model-fitting contract, excluding score-only horizon fields."""

    planned = dict(manifest.get("planned_periods") or {})
    observed = dict(manifest.get("observed_periods") or {})
    group_counts = dict(manifest.get("group_counts") or {})
    event_row_counts = dict(manifest.get("event_row_counts") or {})
    payload = {
        key: manifest.get(key)
        for key in (
            "filter_id",
            "model_architecture",
            "experiment_profile",
            "continuous_target_id",
            "training_label_scope",
            "training_sample_scope",
            "seed",
            "model_information_cutoff",
            "model_spec",
            "experiment_settings",
            "training_settings",
            "source_contract",
            "lookahead_contract",
        )
    }
    payload["planned_periods"] = {
        key: planned.get(key)
        for key in ("validation_start", "validation_end", "score_start", "history_start")
    }
    payload["observed_periods"] = {
        phase: observed.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    payload["group_counts"] = {
        phase: group_counts.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    payload["event_row_counts"] = {
        phase: event_row_counts.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    return canonical_json_sha256(payload, length=32)


def _initial_checkpoint_cache_dir(
    *, cache_root: str | None, benchmark_id: str | None, dl_id: str, seed: int,
) -> Path | None:
    if cache_root in (None, "") or benchmark_id in (None, ""):
        return None
    return (
        PROJECT_ROOT
        / str(cache_root)
        / str(benchmark_id)
        / _unit_key(str(dl_id), int(seed))
    ).resolve()


def _find_initial_pit_fold_dir(model_dir: Path, *, comparison_start: str) -> Path | None:
    folds_root = Path(model_dir).resolve() / "folds"
    if not folds_root.is_dir():
        return None
    target_start = pd.Timestamp(comparison_start).strftime("%Y-%m-%d")
    matches: list[Path] = []
    for candidate in sorted(path for path in folds_root.iterdir() if path.is_dir()):
        manifest_path = candidate / PIT_FOLD_MANIFEST_FILENAME
        model_path = candidate / PIT_FOLD_MODEL_FILENAME
        if not (manifest_path.is_file() and model_path.is_file()):
            continue
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        planned = dict(manifest.get("planned_periods") or {})
        raw_start = str(planned.get("score_start") or "").strip()
        if not raw_start:
            continue
        if pd.Timestamp(raw_start).strftime("%Y-%m-%d") == target_start:
            matches.append(candidate)
    if len(matches) > 1:
        raise RuntimeError(
            "multi-seed PIT初始checkpoint fold不唯一: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def _publish_initial_checkpoint_cache(
    *,
    model_dir: Path,
    cache_dir: Path | None,
    comparison_start: str,
    benchmark_id: str | None,
    dl_id: str,
    seed: int,
) -> dict[str, Any] | None:
    """Persist the common 2021 fitted model outside disposable OOS/Rolling work trees."""

    if cache_dir is None:
        return None
    source_fold_dir = _find_initial_pit_fold_dir(
        model_dir, comparison_start=comparison_start
    )
    if source_fold_dir is None:
        raise RuntimeError(
            f"multi-seed PIT找不到score_start={comparison_start}的初始checkpoint fold"
        )
    source_manifest_path = source_fold_dir / PIT_FOLD_MANIFEST_FILENAME
    source_model_path = source_fold_dir / PIT_FOLD_MODEL_FILENAME
    source_manifest = _read_json(source_manifest_path)
    source_checkpoint_manifest = build_file_manifest(source_model_path)
    if source_checkpoint_manifest != dict(source_manifest.get("artifacts") or {}).get("checkpoint"):
        raise RuntimeError(
            f"multi-seed PIT初始checkpoint hash/size與fold manifest不一致: {source_fold_dir.name}"
        )
    source_model_sha = str(source_checkpoint_manifest["sha256"])
    training_identity = _initial_checkpoint_training_identity(source_manifest)

    existing_manifest_path = cache_dir / PIT_FOLD_MANIFEST_FILENAME
    existing_model_path = cache_dir / PIT_FOLD_MODEL_FILENAME
    if existing_manifest_path.is_file() and existing_model_path.is_file():
        try:
            existing_manifest = _read_json(existing_manifest_path)
            existing_checkpoint_manifest = build_file_manifest(existing_model_path)
            if existing_checkpoint_manifest != dict(existing_manifest.get("artifacts") or {}).get(
                "checkpoint"
            ):
                raise ValueError("cached checkpoint manifest mismatch")
            existing_identity = _initial_checkpoint_training_identity(existing_manifest)
            existing_model_sha = str(existing_checkpoint_manifest["sha256"])
        except (OSError, ValueError, json.JSONDecodeError):
            existing_identity = ""
            existing_model_sha = ""
        if existing_identity == training_identity and existing_model_sha != source_model_sha:
            raise RuntimeError(
                "同一benchmark seed／DL source／fitting contract產生不同初始checkpoint；"
                f"dl_id={dl_id}, seed={seed}, cached={existing_model_sha}, current={source_model_sha}"
            )
        if existing_identity == training_identity and existing_model_sha == source_model_sha:
            return {
                "cache_dir": str(cache_dir),
                "model_sha256": source_model_sha,
                "training_identity": training_identity,
                "reused_existing_cache": True,
            }

    staging = cache_dir.parent / (
        f".{cache_dir.name}.publish_{os.getpid()}_{time.time_ns()}"
    )
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source_model_path, staging / PIT_FOLD_MODEL_FILENAME)
        shutil.copy2(source_manifest_path, staging / PIT_FOLD_MANIFEST_FILENAME)
        _write_json(
            staging / INITIAL_CHECKPOINT_CACHE_MANIFEST_FILENAME,
            {
                "schema_version": 1,
                "benchmark_id": benchmark_id,
                "dl_id": str(dl_id),
                "seed": int(seed),
                "comparison_start": pd.Timestamp(comparison_start).strftime("%Y-%m-%d"),
                "training_identity": training_identity,
                "model_sha256": source_model_sha,
                "source_fold_id": str(source_manifest.get("fold_id") or source_fold_dir.name),
                "source_score_reused": False,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        staging.replace(cache_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "cache_dir": str(cache_dir),
        "model_sha256": source_model_sha,
        "training_identity": training_identity,
        "reused_existing_cache": False,
    }

def _training_command(
    *, arm: StrategyComparisonArm, dl_id: str, seed: int, model_dir: Path, research_dir: Path, settings,
    comparison_start: str, comparison_end: str, checkpoint_reuse_source_fold_dir: Path | None = None,
) -> list[str]:
    dl = settings.dl_sources[str(dl_id)]
    if str(dl.score_source) == "selection_point_in_time":
        command = [
            sys.executable, "-m", "tools.filters.breakout_quality.build_point_in_time_scores",
            "--filter-id", dl.filter_id,
            "--model-architecture", dl.model_architecture,
            "--experiment-profile", dl.experiment_profile,
            "--seed", str(int(seed)),
            "--fold-months", str(
                _source_point_in_time_fold_months(settings=settings, dl_id=str(dl_id))
            ),
            "--score-start-date", str(comparison_start),
            "--score-end-date", str(comparison_end),
            "--point-in-time-dir-override", str(model_dir.resolve()),
        ]
        fold_anchor_date = _source_point_in_time_fold_anchor_date(
            settings=settings, dl_id=str(dl_id)
        )
        if fold_anchor_date is not None:
            command.extend(["--fold-anchor-date", fold_anchor_date])
        if _source_point_in_time_single_score_block(settings=settings, dl_id=str(dl_id)):
            command.append("--single-score-block")
        if checkpoint_reuse_source_fold_dir is not None and checkpoint_reuse_source_fold_dir.is_dir():
            command.extend([
                "--checkpoint-reuse-source-fold-dir",
                str(checkpoint_reuse_source_fold_dir.resolve()),
            ])
        return command
    return [
        sys.executable, "-m", "tools.filters.breakout_quality.train_continuous_ranker",
        "--filter-id", dl.filter_id, "--model-architecture", dl.model_architecture,
        "--experiment-profile", dl.experiment_profile, "--seed", str(int(seed)),
        "--model-output-dir", str(model_dir.resolve()),
        "--research-output-dir", str(research_dir.resolve()),
    ]


def _terminate_trainer_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=TRAINER_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _terminate_active_trainers(
    registry: dict[str, subprocess.Popen],
    registry_lock: Lock,
) -> None:
    with registry_lock:
        processes = list(registry.values())
    for proc in processes:
        _terminate_trainer_process(proc)


def _train_one_unit(job: dict[str, Any]) -> dict[str, Any]:
    arm: StrategyComparisonArm = job["arm"]
    settings = job["settings"]
    seed = int(job["seed"])
    model_dir = Path(job["model_dir"]).resolve()
    research_dir = Path(job["research_dir"]).resolve()
    train_log_path = research_dir / "train.log"
    dl_id = str(job["dl_id"])
    dl = settings.dl_sources[dl_id]
    is_selection_pit = str(dl.score_source) == "selection_point_in_time"
    checkpoint_cache_dir = (
        None
        if job.get("initial_checkpoint_cache_dir") in (None, "")
        else Path(str(job["initial_checkpoint_cache_dir"])).resolve()
    )
    checkpoint_reuse_source_fold_dir = (
        checkpoint_cache_dir
        if is_selection_pit and checkpoint_cache_dir is not None and checkpoint_cache_dir.is_dir()
        else None
    )
    started = time.perf_counter()
    artifacts = None
    reusable_artifact_roots_exist = model_dir.is_dir() and (
        is_selection_pit or research_dir.is_dir()
    )
    if bool(job.get("reuse_completed")) and reusable_artifact_roots_exist:
        try:
            artifacts = _validate_training_artifacts(
                arm=arm,
                dl_id=dl_id,
                seed=seed,
                model_dir=model_dir,
                research_dir=research_dir,
                settings=settings,
                comparison_start=str(job["comparison_start"]),
                comparison_end=str(job["comparison_end"]),
            )
        except (FileNotFoundError, ValueError):
            artifacts = None
    if artifacts is not None:
        cache = (
            _publish_initial_checkpoint_cache(
                model_dir=model_dir,
                cache_dir=checkpoint_cache_dir,
                comparison_start=str(job["comparison_start"]),
                benchmark_id=job.get("benchmark_id"),
                dl_id=dl_id,
                seed=seed,
            )
            if is_selection_pit and checkpoint_cache_dir is not None
            else None
        )
        return {
            "artifacts": artifacts,
            "reused": True,
            "initial_checkpoint_cache": cache,
            "wall_elapsed_sec": round(time.perf_counter() - started, 3),
        }

    if is_selection_pit:
        # PIT builder本身具備fold-level resume；完整seed尚未完成時只清除top-level
        # aggregate工件，保留已完成fold checkpoint/score，避免中斷後把合法partial
        # folds全部刪掉而從fold 1重訓。
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
    env = dict(os.environ)
    env["BREAKOUT_QUALITY_COMPACT_CONSOLE"] = "1"
    trainer_registry = job["trainer_registry"]
    trainer_registry_lock = job["trainer_registry_lock"]
    trainer_key = str(job["trainer_key"])
    with train_log_path.open("w", encoding="utf-8") as train_log:
        proc = subprocess.Popen(
            _training_command(
                arm=arm,
                dl_id=dl_id,
                seed=seed,
                model_dir=model_dir,
                research_dir=research_dir,
                settings=settings,
                comparison_start=str(job["comparison_start"]),
                comparison_end=str(job["comparison_end"]),
                checkpoint_reuse_source_fold_dir=checkpoint_reuse_source_fold_dir,
            ),
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=train_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with trainer_registry_lock:
            trainer_registry[trainer_key] = proc
        try:
            returncode = proc.wait()
        except BaseException:
            _terminate_trainer_process(proc)
            raise
        finally:
            with trainer_registry_lock:
                trainer_registry.pop(trainer_key, None)
    if int(returncode or 0) != 0:
        train_tail = ""
        if train_log_path.is_file():
            train_tail = train_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"multi-seed模型訓練失敗: arm={arm.name}, seed_index={job['seed_order']}; "
            f"returncode={returncode}; log_tail={train_tail}"
        )
    artifacts = _validate_training_artifacts(
        arm=arm,
        dl_id=dl_id,
        seed=seed,
        model_dir=model_dir,
        research_dir=research_dir,
        settings=settings,
        comparison_start=str(job["comparison_start"]),
        comparison_end=str(job["comparison_end"]),
    )
    cache = (
        _publish_initial_checkpoint_cache(
            model_dir=model_dir,
            cache_dir=checkpoint_cache_dir,
            comparison_start=str(job["comparison_start"]),
            benchmark_id=job.get("benchmark_id"),
            dl_id=dl_id,
            seed=seed,
        )
        if is_selection_pit and checkpoint_cache_dir is not None
        else None
    )
    return {
        "artifacts": artifacts,
        "reused": False,
        "initial_checkpoint_cache": cache,
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
    initial_checkpoint_cache_root: str | None = None,
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
            initial_checkpoint_cache_dir = _initial_checkpoint_cache_dir(
                cache_root=initial_checkpoint_cache_root,
                benchmark_id=benchmark_id,
                dl_id=str(dl_id),
                seed=int(seed),
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
                "initial_checkpoint_cache_dir": (
                    None
                    if initial_checkpoint_cache_dir is None
                    else str(initial_checkpoint_cache_dir)
                ),
            })
    return units


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _pit_saved_fold_progress(meta: dict[str, Any]) -> tuple[int, int] | None:
    settings = meta["settings"]
    dl = settings.dl_sources[str(meta["dl_id"])]
    if str(dl.score_source) != "selection_point_in_time":
        return None
    expected = _pit_fold_count_for_period(
        str(meta["comparison_start"]),
        str(meta["comparison_end"]),
        _source_point_in_time_fold_months(settings=settings, dl_id=str(meta["dl_id"])),
    )
    folds_root = Path(str(meta["model_dir"])).resolve() / "folds"
    if not folds_root.is_dir():
        return 0, expected
    saved = sum(
        1
        for path in folds_root.iterdir()
        if path.is_dir()
        and (path / PIT_FOLD_MANIFEST_FILENAME).is_file()
        and (path / PIT_FOLD_MODEL_FILENAME).is_file()
    )
    return min(saved, expected), expected


def _training_progress_label(meta: dict[str, Any], *, now: float, seed_count: int) -> str:
    label = (
        f"seed {meta['seed_order']}/{seed_count} {meta['dl_id']} "
        f"targets={len(meta['replay_arms'])} "
        f"{_format_elapsed(now-float(meta['submitted_at']))}"
    )
    pit_progress = _pit_saved_fold_progress(meta)
    if pit_progress is not None:
        saved, expected = pit_progress
        label += f" [PIT saved folds={saved}/{expected}]"
    return label


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
                paint("[BASELINE REUSE]", "green", enabled=color_enabled, bold=True)
                + f" {arm.name} | {project_relative_display_path(reusable_dir, project_root=PROJECT_ROOT)}"
            )
            continue
        if output_dir.exists():
            shutil.rmtree(output_dir)
        all_off = arm.rule_policy == "all_off"
        param_evaluation_mode = str(
            settings.parameter_sources[arm.param_source].canonical_evaluation_mode or "rolling"
        )
        payload = run_standalone_baseline(
            project_root=PROJECT_ROOT,
            dataset=settings.dataset,
            params_path=path_text,
            param_policy=arm_param_policy,
            max_positions=settings.max_positions,
            enable_rotation=settings.rotation == "on",
            optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off
                else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            output_dir_override=output_dir,
            comparison_start_date=start,
            comparison_end_date=end,
            quiet=True,
            shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
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
    all_off = arm.rule_policy == "all_off"
    started = time.perf_counter(); runtime_spec = _arm_runtime_spec(arm)
    is_selection = str(dl.score_source) == "selection_point_in_time"
    score_overrides = {
        str(key): dict(value or {})
        for key, value in dict(job.get("score_overrides") or {}).items()
    }
    primary_override = dict(score_overrides.get(str(arm.dl_id)) or {})
    payload = run_comparison(
        project_root=PROJECT_ROOT, dataset=settings.dataset, params_path=str(job["params_path"]),
        param_policy=resolve_strategy_comparison_arm_param_policy(settings, arm), max_positions=settings.max_positions,
        enable_rotation=settings.rotation == "on", comparison_mode=runtime_spec["comparison_mode"],
        ranking_policy=runtime_spec["ranking_policy"], ranking_options=_resolved_ranking_options(
            settings,
            arm,
            continuous_score_overrides=score_overrides,
        ),
        optional_entry_filter_policy=(OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT),
        filter_id=dl.filter_id, score_source=dl.score_source, model_architecture=dl.model_architecture,
        experiment_profile=dl.experiment_profile, threshold=dl.threshold, output_dir_override=pair_dir,
        comparison_start_date=start, comparison_end_date=end, quiet=True,
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
        baseline_reuse_dir=baseline_dir,
        param_evaluation_mode=str(job.get("param_evaluation_mode") or "rolling"),
        continuous_score_path_override=(
            None if is_selection else str(primary_override.get("score_path") or job["score_path"])
        ),
        continuous_score_execution_start_override=(
            None if is_selection else str(primary_override.get("execution_start") or job["score_execution_start"])
        ),
        selection_pit_score_path_override=(
            str(primary_override.get("score_path") or job["score_path"]) if is_selection else None
        ),
        selection_pit_manifest_path_override=(
            str(primary_override.get("manifest_path") or job["score_manifest_path"]) if is_selection else None
        ),
        selection_pit_expected_seed_override=(int(job["seed"]) if is_selection else None),
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



def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"



def _style_text(text: str, signal: str | None, *, target: str) -> str:
    if not signal or text == "-":
        return text
    return styled_signal(text, signal, target=target)


def _format_metric_value(value: Any, metric, *, target: str, signal: str | None = None) -> str:
    text = _fmt(value, digits=int(metric.digits), suffix=str(metric.unit))
    return _style_text(text, signal, target=target)


def _distribution_signals(
    rows: list[dict[str, Any]],
    *,
    metrics,
    identity_key: str,
) -> dict[str, dict[str, str]]:
    return {
        metric.key: best_worst_signals(
            {str(row[identity_key]): row.get(metric.key) for row in rows},
            preference=str(metric.preference),
        )
        for metric in metrics
    }


def _right_minus_left_signal(value: Any, *, preference: str) -> str | None:
    if preference not in {"higher", "lower"}:
        return None
    return signal_for_delta(value, preference=preference)


def _win_count_signals(right_count: int, left_count: int) -> tuple[str | None, str | None]:
    if right_count == left_count:
        return None, None
    if right_count > left_count:
        return SIGNAL_POSITIVE, SIGNAL_NEGATIVE
    return SIGNAL_NEGATIVE, SIGNAL_POSITIVE


def _direction_signal(selection_delta: float, romd_delta: float) -> str:
    selection_signal = signal_for_delta(selection_delta, preference="higher")
    romd_signal = signal_for_delta(romd_delta, preference="higher")
    if selection_signal == SIGNAL_POSITIVE and romd_signal == SIGNAL_POSITIVE:
        return SIGNAL_POSITIVE
    if selection_signal == SIGNAL_NEGATIVE and romd_signal == SIGNAL_NEGATIVE:
        return SIGNAL_NEGATIVE
    return SIGNAL_WARNING


def render_multi_seed_robustness_report(
    summary: dict[str, Any],
    *,
    target: str = "markdown",
) -> str:
    """Render robustness using the canonical Strategy Compare report format.

    Sections 1-4 come from the exact same top-level Strategy Compare renderer used
    by the configured comparison profile.  Multi-seed-only evidence is appended
    from section 5 onward.
    """

    contract = dict(summary["contract"])
    settings = _report_settings_for_contract(
        contract, get_strategy_comparison_settings(str(contract["profile_id"]))
    )
    common = dict(summary.get("common_strategy_report") or {})
    scenarios = dict(common.get("scenarios") or {})
    r_analysis = list(common.get("r_analysis") or [])
    yearly_by_id = {
        str(arm_id): {int(year): value for year, value in dict(values or {}).items()}
        for arm_id, values in dict(common.get("yearly_by_id") or {}).items()
    }
    if not scenarios:
        raise ValueError("robustness報表缺少canonical核心策略結果")
    if not yearly_by_id:
        raise ValueError("robustness報表缺少canonical年度結果")

    canonical = render_strategy_aggregate_report(
        settings=settings,
        comparison_period=dict(contract.get("comparison_period") or {}),
        fingerprint=str(contract["fingerprint"]),
        scenarios=scenarios,
        diagnostics={"r_analysis": r_analysis},
        yearly_by_id=yearly_by_id,
        target=target,
        title="策略績效比較",
        fingerprint_label="Scientific fingerprint",
        extra_metadata=(
            ("比較階段", str(contract.get("label") or "Multiple-seed robustness")),
            ("Benchmark ID", str(contract.get("benchmark_id") or "-")),
            ("Strategy trials/fold", str(contract.get("strategy_trials_per_fold") or "-")),
            ("Seed generator", str(contract.get("seed_generator_seed") or "-")),
            (
                "Seeds",
                f"{int(contract['seed_count'])}（deterministic generated；不作best-seed選擇）",
            ),
            (
                "Report schema",
                f"{int(summary['schema_version'])}（不影響scientific fingerprint）",
            ),
        ),
    ).rstrip()

    sections: list[str] = [canonical]

    romd_source_rows: list[dict[str, Any]] = []
    for source in summary["romd_statistics"]:
        row = dict(source)
        n = int(row["n"])
        row["beats_min_rate"] = (
            None if row.get("beats_min_count") is None or n <= 0
            else float(row["beats_min_count"]) / float(n)
        )
        row["beats_full_rate"] = (
            None if row.get("beats_full_count") is None or n <= 0
            else float(row["beats_full_count"]) / float(n)
        )
        romd_source_rows.append(row)
    romd_signals = _distribution_signals(
        romd_source_rows,
        metrics=ROBUSTNESS_ROMD_DISTRIBUTION_METRICS,
        identity_key="arm_id",
    )
    romd_rows = []
    romd_metric_by_key = {metric.key: metric for metric in ROBUSTNESS_ROMD_DISTRIBUTION_METRICS}
    for row in romd_source_rows:
        n = int(row["n"])
        arm_id = str(row["arm_id"])
        metric_values = []
        for key in ("mean", "median", "std", "cv", "min", "p25", "p75", "max"):
            metric = romd_metric_by_key[key]
            metric_values.append(
                _format_metric_value(
                    row.get(key),
                    metric,
                    target=target,
                    signal=romd_signals.get(key, {}).get(arm_id),
                )
            )
        beat_values = []
        for count_key, rate_key in (
            ("beats_min_count", "beats_min_rate"),
            ("beats_full_count", "beats_full_rate"),
        ):
            if row.get(count_key) is None:
                beat_values.append("-")
                continue
            text = f"{int(row[count_key])}/{n}"
            beat_values.append(
                _style_text(
                    text,
                    romd_signals.get(rate_key, {}).get(arm_id),
                    target=target,
                )
            )
        romd_rows.append([row["name"], str(n), *metric_values, *beat_values])
    sections.extend([
        render_section("5. RoMD完整統計"),
        render_table(
            ["比較對象", "N", "Mean", "Median", "Std", "CV", "Min", "P25", "P75", "Max", "勝Min", "勝Full"],
            romd_rows,
        ),
    ])

    paired = list(summary.get("paired_comparisons") or [])
    if not paired:
        legacy_same = summary.get("romd_same_seed_comparison")
        if isinstance(legacy_same, dict):
            paired = [{
                "contrast_id": "legacy_default",
                "description": "歷史兩arm robustness同seed比較",
                "left": legacy_same.get("left"),
                "right": legacy_same.get("right"),
                "romd_same_seed": legacy_same,
                "direct_selection_r_same_seed": summary.get("direct_selection_r_same_seed_comparison"),
                "selection_r_to_strategy": summary.get("selection_r_to_strategy_same_seed_translation"),
                "romd_distribution": summary.get("romd_distribution_comparison"),
                "yearly_same_seed": summary.get("yearly_same_seed_comparison") or [],
            }]

    next_section = 6
    if paired:
        sections.append(render_section(f"{next_section}. 設定中的同seed contrasts"))
        paired_blocks: list[str] = []
        for item in paired:
            left = str(item.get("left") or "Left")
            right = str(item.get("right") or "Right")
            block: list[str] = [f"{right} − {left}"]
            description = str(item.get("description") or "").strip()
            if description:
                block.append(f"用途：{description}")

            same = item.get("romd_same_seed")
            if isinstance(same, dict):
                n = int(same["n"])
                right_signal, left_signal = _win_count_signals(
                    int(same["right_gt_left_count"]),
                    int(same["left_gt_right_count"]),
                )
                right_count_text = _style_text(
                    f"{same['right_gt_left_count']}/{n}", right_signal, target=target
                )
                left_count_text = _style_text(
                    f"{same['left_gt_right_count']}/{n}", left_signal, target=target
                )
                delta_parts = []
                for label, key, preference in (
                    ("Mean", "right_minus_left_mean", "higher"),
                    ("Median", "right_minus_left_median", "higher"),
                    ("Std", "right_minus_left_std", "neutral"),
                    ("Min", "right_minus_left_min", "higher"),
                    ("P25", "right_minus_left_p25", "higher"),
                    ("P75", "right_minus_left_p75", "higher"),
                    ("Max", "right_minus_left_max", "higher"),
                ):
                    value = same.get(key)
                    signal = _right_minus_left_signal(value, preference=preference)
                    delta_parts.append(
                        f"{label} {_style_text(_fmt(value), signal, target=target)}"
                    )
                block.extend([
                    f"RoMD：{same['right']} > {same['left']} {right_count_text}；"
                    f"{same['left']} > {same['right']} {left_count_text}；Tie {same['tie_count']}/{n}。",
                    "ΔRoMD " + "；".join(delta_parts) + "。",
                ])

            direct_same = item.get("direct_selection_r_same_seed")
            if isinstance(direct_same, dict):
                n = int(direct_same["n"])
                right_signal, left_signal = _win_count_signals(
                    int(direct_same["right_gt_left_count"]),
                    int(direct_same["left_gt_right_count"]),
                )
                right_count_text = _style_text(
                    f"{direct_same['right_gt_left_count']}/{n}", right_signal, target=target
                )
                left_count_text = _style_text(
                    f"{direct_same['left_gt_right_count']}/{n}", left_signal, target=target
                )
                mean_text = _style_text(
                    _fmt(direct_same.get("right_minus_left_mean"), suffix=" R"),
                    _right_minus_left_signal(
                        direct_same.get("right_minus_left_mean"), preference="higher"
                    ),
                    target=target,
                )
                median_text = _style_text(
                    _fmt(direct_same.get("right_minus_left_median"), suffix=" R"),
                    _right_minus_left_signal(
                        direct_same.get("right_minus_left_median"), preference="higher"
                    ),
                    target=target,
                )
                block.extend([
                    f"DL選擇R：{direct_same['right']} > {direct_same['left']} {right_count_text}；"
                    f"{direct_same['left']} > {direct_same['right']} {left_count_text}；Tie {direct_same['tie_count']}/{n}。",
                    f"ΔDL選擇R Mean {mean_text}；Median {median_text}；"
                    f"Std {_fmt(direct_same['right_minus_left_std'], suffix=' R')}。",
                ])

            translation = item.get("selection_r_to_strategy")
            if isinstance(translation, dict):
                pair_rows = []
                delta_metric_by_key = {
                    metric.key: metric for metric in ROBUSTNESS_SEED_DELTA_METRICS
                }
                for row in translation.get("seed_rows") or []:
                    selection_delta = float(row["right_minus_left_direct_selection_r"])
                    romd_delta = float(row["right_minus_left_return_over_max_drawdown"])
                    if selection_delta > 0 and romd_delta > 0:
                        verdict = "ranking↑／RoMD↑"
                    elif selection_delta > 0:
                        verdict = "ranking↑／RoMD↓"
                    elif romd_delta > 0:
                        verdict = "ranking↓／RoMD↑"
                    else:
                        verdict = "ranking↓／RoMD↓"
                    values = []
                    for key in (
                        "right_minus_left_direct_selection_r",
                        "right_minus_left_total_return_pct",
                        "right_minus_left_max_drawdown_pct",
                        "right_minus_left_return_over_max_drawdown",
                        "right_minus_left_expected_value_r",
                    ):
                        metric = delta_metric_by_key[key]
                        value = row.get(key)
                        values.append(
                            _format_metric_value(
                                value,
                                metric,
                                target=target,
                                signal=_right_minus_left_signal(
                                    value, preference=metric.preference
                                ),
                            )
                        )
                    pair_rows.append([
                        f"S{int(row['seed_index'])}",
                        *values,
                        _style_text(
                            verdict,
                            _direction_signal(selection_delta, romd_delta),
                            target=target,
                        ),
                    ])
                if pair_rows:
                    block.extend([
                        "",
                        render_table(
                            ["Seed", "ΔDL選擇R", "ΔReturn", "ΔMDD", "ΔRoMD", "ΔEV", "方向"],
                            pair_rows,
                        ),
                    ])
                positive_n = int(translation["selection_r_positive_count"])
                total_n = int(translation["n"])
                translated_n = int(translation["selection_r_positive_romd_positive_count"])
                positive_signal = signal_for_delta(
                    (float(positive_n) / float(total_n)) - 0.5,
                    preference="higher",
                ) if total_n > 0 else None
                translated_signal = signal_for_delta(
                    (float(translated_n) / float(positive_n)) - 0.5,
                    preference="higher",
                ) if positive_n > 0 else None
                concordant_n = int(translation["sign_concordant_count"])
                discordant_n = int(translation["sign_discordant_count"])
                concordant_signal, discordant_signal = _win_count_signals(
                    concordant_n, discordant_n
                )
                spearman_romd = translation.get("selection_r_delta_vs_romd_spearman")
                spearman_return = translation.get("selection_r_delta_vs_return_spearman")
                block.extend([
                    f"ΔDL選擇R>0："
                    f"{_style_text(f'{positive_n}/{total_n}', positive_signal, target=target)}；"
                    f"其中ΔRoMD>0："
                    f"{_style_text(f'{translated_n}/{positive_n if positive_n else 0}', translated_signal, target=target)}。",
                    f"方向一致："
                    f"{_style_text(f'{concordant_n}/{translation['sign_non_tie_n']}', concordant_signal, target=target)}；"
                    f"方向相反："
                    f"{_style_text(f'{discordant_n}/{translation['sign_non_tie_n']}', discordant_signal, target=target)}。",
                    f"Spearman(ΔDL選擇R, ΔRoMD)="
                    f"{_style_text(_fmt(spearman_romd, digits=3), _right_minus_left_signal(spearman_romd, preference='higher'), target=target)}；"
                    f"Spearman(ΔDL選擇R, ΔReturn)="
                    f"{_style_text(_fmt(spearman_return, digits=3), _right_minus_left_signal(spearman_return, preference='higher'), target=target)}。",
                ])

            compare = item.get("romd_distribution")
            if isinstance(compare, dict):
                probability = float(compare["pairwise_left_gt_right_probability"])
                probability_text = _style_text(
                    f"{probability * 100:.2f}%",
                    signal_for_delta(probability - 0.5, preference="higher"),
                    target=target,
                )
                left_n = int(compare.get("left_n") or 0)
                right_n = int(compare.get("right_n") or 0)
                pair_shape = (
                    f"{left_n}×{right_n}={int(compare['pair_count'])} all-pairs"
                    if left_n > 0 and right_n > 0
                    else f"{int(compare['pair_count'])} all-pairs"
                )
                block.append(
                    f"跨seed分布 P({compare['left']} > {compare['right']})="
                    f"{probability_text}"
                    f"（{pair_shape}；非same-seed配對勝率）。"
                )

            annual_pair = list(item.get("yearly_same_seed") or [])
            if annual_pair:
                yearly_delta_metric_by_key = {
                    metric.key: metric for metric in ROBUSTNESS_YEARLY_DELTA_METRICS
                }
                pair_rows = []
                for row in annual_pair:
                    metric_values = []
                    for key in (
                        "right_minus_left_mean",
                        "right_minus_left_median",
                        "right_minus_left_std",
                    ):
                        metric = yearly_delta_metric_by_key[key]
                        value = row.get(key)
                        metric_values.append(
                            _format_metric_value(
                                value,
                                metric,
                                target=target,
                                signal=_right_minus_left_signal(
                                    value, preference=metric.preference
                                ),
                            )
                        )
                    right_signal, left_signal = _win_count_signals(
                        int(row["right_gt_left_count"]),
                        int(row["left_gt_right_count"]),
                    )
                    pair_rows.append([
                        str(row["year"]),
                        str(row["n"]),
                        *metric_values,
                        _style_text(
                            f"{row['right_gt_left_count']}/{row['n']}",
                            right_signal,
                            target=target,
                        ),
                        _style_text(
                            f"{row['left_gt_right_count']}/{row['n']}",
                            left_signal,
                            target=target,
                        ),
                        f"{row['tie_count']}/{row['n']}",
                    ])
                block.extend([
                    "",
                    render_table(
                        ["年度", "N", "Δ右-左 Mean", "Median", "Std", "右勝", "左勝", "Tie"],
                        pair_rows,
                    ),
                ])

            paired_blocks.append("\n".join(block))
        sections.append("\n\n".join(paired_blocks))
        next_section += 1

    yearly = list(summary.get("yearly_statistics") or [])
    if yearly:
        rows = []
        yearly_metric_by_key = {
            metric.key: metric for metric in ROBUSTNESS_YEARLY_DISTRIBUTION_METRICS
        }
        yearly_by_year: dict[int, list[dict[str, Any]]] = {}
        for source in yearly:
            yearly_by_year.setdefault(int(source["year"]), []).append(dict(source))
        for year in sorted(yearly_by_year):
            year_rows = yearly_by_year[year]
            year_signals = _distribution_signals(
                year_rows,
                metrics=ROBUSTNESS_YEARLY_DISTRIBUTION_METRICS,
                identity_key="arm_id",
            )
            for row in sorted(year_rows, key=lambda x: (str(x["type"]), str(x["name"]))):
                arm_id = str(row["arm_id"])
                metric_values = []
                for key in ("mean", "median", "std", "min", "p25", "p75", "max"):
                    metric = yearly_metric_by_key[key]
                    metric_values.append(
                        _format_metric_value(
                            row.get(key),
                            metric,
                            target=target,
                            signal=year_signals.get(key, {}).get(arm_id),
                        )
                    )
                rows.append([
                    str(row["year"]) + ("" if row.get("is_complete_year") else "*"),
                    row["name"],
                    row["type"],
                    str(row["n"]),
                    *metric_values,
                ])
        sections.extend([
            render_section(f"{next_section}. 歷年報酬跨seed完整統計"),
            render_table(
                ["年度", "比較對象", "類型", "N", "Mean", "Median", "Std", "Min", "P25", "P75", "Max"],
                rows,
            ) + "\n* 非完整年度。",
        ])
        next_section += 1

    references = dict(contract.get("romd_reference_baselines") or {})
    min_name = str(dict(references.get("min") or {}).get("name") or "Min baseline")
    full_name = str(dict(references.get("full") or {}).get("name") or "Full baseline")
    sections.extend([
        render_section(f"{next_section}. 限制"),
        "\n".join((
            "- 前四張表直接重用Strategy Compare canonical aggregate renderer；Multi-seed arm顯示per-seed canonical metric的Mean，Fixed arm顯示正式baseline值。",
            "- Multi-seed專屬分布表同樣使用core/report_style.py：有明確方向的metric在同欄可比較arm間只標綠＝最佳、紅＝最差、其餘白；same-seed右減左欄位則依共用metric direction判讀，ΔMDD採lower-is-better，N／Type／Tie等中性欄不硬判。",
            "- 舊robustness工件若未永久保存model prediction／Future Target conversion欄位，report-only refresh會顯示`-`，不為補報表重訓或重跑strategy replay。",
            "- resolved seeds只用於重現；不得挑best seed或依本報表組seed ensemble。",
            (
                f"- {full_name}／{min_name}在current end-to-end benchmark中使用per-seed strategy params；只有C62/C63 finalists-agree屬固定production consensus context。"
                if "full" in references
                else f"- {min_name}在current end-to-end benchmark中使用per-seed strategy params；fixed finalists-agree僅作production consensus context。"
            ),
            "- 同一DL source／seed只訓練一次，允許fan-out到不同runtime selector replay；此reuse不改變模型scientific condition。",
            "- Extending-Window Rolling robustness只評估config既定scientific condition；不得依結果回頭調整training semantics或使用future fold結果擬合當下模型。",
        )),
    ])
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"



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
            "file_sha256": compute_file_sha256(path),
            "manifest_sha256": compute_file_sha256(manifest_path),
        }
    contract = build_multi_seed_robustness_contract(
        robustness_id=cfg.robustness_id, comparison_period=dict(plan["comparison_period"]),
        artifact_identities=dict(status.get("artifact_identities") or {}),
        benchmark_parameter_identities=_benchmark_identity_payload(resolved_benchmark_bindings),
    )
    run_root = _run_root(contract)
    existing = _load_seed_results(run_root / SEED_RESULTS_FILENAME)
    seeds = tuple(int(value) for value in contract["resolved_seeds"])
    _validate_seed_results_frame(existing, stochastic_arms=stochastic, seeds=seeds)
    print(f"Fingerprint       ：{contract['fingerprint']}")
    ready_attribution = _attribution_ready_units(
        run_root, stochastic_arms=tuple(model_arms), seeds=seeds, fingerprint=str(contract["fingerprint"])
    ) if cfg.keep_attribution_source else set()
    print(f"已完成seed結果    ：{len(existing)}/{len(stochastic) * cfg.seed_count}")
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
            _atomic_write_text(
                report, render_multi_seed_robustness_report(upgraded, target="markdown")
            )
    if not summary_path.is_file():
        raise FileNotFoundError(f"最新robustness摘要不存在: {summary_path}")
    summary = _read_json(summary_path)
    _atomic_write_text(
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
        pit_progress = _pit_saved_fold_progress(meta)
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

    post_upstream_rows, post_upstream_blockers = _model_upstream_rows(
        settings, model_arms
    )
    post_upstream_pending = [
        row for row in post_upstream_rows if row[0] in {"BUILD", "REBUILD", "RESUME"}
    ]
    if post_upstream_blockers or post_upstream_pending:
        detail = [*post_upstream_blockers, *(row[2] for row in post_upstream_pending)]
        raise RuntimeError(
            "Multiple-seed robustness模型上游工件在自動補建後仍未就緒: "
            + "；".join(detail)
        )

    # Dataset/Target preparation may change the valid OOS horizon.  Resolve the period
    # from the freshly rebuilt canonical upstream truth *before* dispatching Optimizer
    # parameter producers, then bind that period into the shared Strategy preparation
    # graph.  This prevents stale comparison_end=None snapshots after partial/cold/stale
    # rebuilds and applies equally when only part of outputs/models was removed.
    comparison_start, comparison_end = _comparison_period_from_upstream(
        settings, model_arms, status
    )
    period_override = {"start": str(comparison_start), "end": str(comparison_end)}
    status = collect_strategy_preparation_status(
        project_root=PROJECT_ROOT,
        settings=settings,
        comparison_period_override=period_override,
    )
    status = prepare_strategy_parameter_artifacts(
        project_root=PROJECT_ROOT,
        settings=settings,
        status=status,
        required_source_ids=tuple(plan["required_param_sources"]),
        status_refresher=lambda: collect_strategy_preparation_status(
            project_root=PROJECT_ROOT,
            settings=settings,
            comparison_period_override=period_override,
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
        comparison_end=str(comparison_end),
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
                _atomic_write_text(
                    run_root / REPORT_FILENAME,
                    render_multi_seed_robustness_report(summary, target="markdown"),
                )
                manifest_payload = dict(completed_run["manifest"])
                manifest_payload["durable_artifacts"] = _build_durable_result_artifacts(run_root)
                manifest_payload["report_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_root / MANIFEST_FILENAME, manifest_payload)
            print(
                paint("[ROBUSTNESS REUSE]", "green", enabled=console_color_enabled(), bold=True)
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
    seed_expansion_reuse = None
    if cfg.reuse_completed:
        seed_expansion_reuse = _import_seed_expansion_results(
            contract=contract,
            run_root=run_root,
            stochastic_arms=tuple(stochastic_arms),
            model_arms=tuple(model_arms),
            keep_attribution_source=bool(cfg.keep_attribution_source),
        )
        if seed_expansion_reuse is not None:
            manifest["seed_expansion_reuse"] = dict(seed_expansion_reuse)
            _write_json(manifest_path, manifest)
            reused_seeds_text = ",".join(
                str(value) for value in seed_expansion_reuse["reused_seeds"]
            )
            reuse_label = (
                "[COMPATIBLE RESULT REUSE]"
                if len(seed_expansion_reuse["reused_seeds"]) == len(seeds)
                else "[SEED EXPANSION REUSE]"
            )
            print(
                paint(reuse_label, "green", enabled=console_color_enabled(), bold=True)
                + f" seeds={reused_seeds_text} | observations={seed_expansion_reuse['reused_observations']}"
            )
    try:
        existing = _load_seed_results(seed_results_path)
        _validate_seed_results_frame(existing, stochastic_arms=stochastic_arms, seeds=seeds)
        _validate_seed_result_scientific_identities(existing, contract=contract)
        existing_yearly = _load_seed_yearly_results(seed_yearly_results_path)
        expected_years = _comparison_years(comparison_start, comparison_end)
        _validate_seed_yearly_results_frame(
            existing_yearly, stochastic_arms=stochastic_arms, seeds=seeds,
            expected_years=expected_years, require_complete_units=False,
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
            paint("[FAILED]", "red", enabled=console_color_enabled(), bold=True)
            + " Multiple-seed robustness前置回放階段已停止；可由同一入口接續。"
            + f" manifest={project_relative_display_path(manifest_path, project_root=PROJECT_ROOT)}"
        )
        raise
    scientific_completed = {
        (str(row.arm_id), int(row.seed))
        for row in existing.itertuples(index=False)
    } if not existing.empty and cfg.reuse_completed else set()
    if scientific_completed:
        expected_year_set = set(_comparison_years(comparison_start, comparison_end))
        yearly_by_unit: dict[tuple[str, int], set[int]] = {}
        if not existing_yearly.empty:
            for row in existing_yearly.itertuples(index=False):
                unit = (str(row.arm_id), int(row.seed))
                yearly_by_unit.setdefault(unit, set()).add(int(row.year))
        complete_yearly_units = {
            unit for unit, years in yearly_by_unit.items()
            if years == expected_year_set
        }
        scientific_completed &= complete_yearly_units
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
    min_ref_arm = str(dict(dict(contract.get("romd_reference_baselines") or {}).get("min") or {}).get("arm_id") or "")

    def progress_update(text: str) -> None:
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
        initial_checkpoint_cache_root=cfg.initial_checkpoint_cache_root,
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
                progress.print_line(
                    paint(f"[DONE {displayed_done_units}/{total_units}]", "green", enabled=color_enabled, bold=True)
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name}"
                )
            elif strategy_action == "REBUILD_CONTEXT":
                progress.print_line(
                    paint("[BASELINE CONTEXT REBUILD]", "yellow", enabled=color_enabled, bold=True)
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name} | scientific result reuse"
                )
            elif unit_key in completed:
                progress.print_line(
                    paint(f"[REUSE {displayed_done_units}/{total_units}]", "green", enabled=color_enabled, bold=True)
                    + f" seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name}"
                )
            elif unit_key in scientific_completed:
                progress.print_line(
                    paint("[REBUILD ATTRIBUTION]", "yellow", enabled=color_enabled, bold=True)
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
            done_tag = paint(
                f"[DONE {done_units}/{total_units}]", "green", enabled=color_enabled, bold=True
            )
            delta_text = (
                "ΔMin=N/A"
                if delta_min is None else terminal_signal(
                    f"ΔMin={delta_min:+.2f}",
                    signal_for_delta(delta_min, preference="higher"),
                    enabled=color_enabled,
                )
            )
            progress.print_line(
                f"{done_tag} seed {meta['seed_order']}/{len(seeds)} | "
                f"對象 {meta['arm_order']}/{len(stochastic_arms)} {meta['name']} | "
                f"RoMD={romd:.2f} | {delta_text} | "
                f"train={_format_elapsed(result.get('training_elapsed_sec', 0))} | "
                f"replay={_format_elapsed(result.get('replay_elapsed_sec', 0))} | "
                f"total={_format_elapsed(time.perf_counter()-started_total)}"
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
        required_dl_ids = _arm_training_dl_ids(settings, arm)
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
            cache_meta = dict(trained.get("initial_checkpoint_cache") or {})
            cache_note = (
                " | 2021 checkpoint cache="
                + ("reuse" if bool(cache_meta.get("reused_existing_cache")) else "publish")
                if cache_meta else ""
            )
            fold_note = ""
            if fold_reuse:
                fold_note = (
                    " | folds="
                    f"reuse:{int(fold_reuse.get('exact_reuse', 0) or 0)}"
                    f"/rescore:{int(fold_reuse.get('local_checkpoint_rescore', 0) or 0)}"
                    f"/cross-mode:{int(fold_reuse.get('cross_mode_initial_checkpoint_reuse', 0) or 0)}"
                    f"/migrate:{int(fold_reuse.get('legacy_migration', 0) or 0)}"
                    f"/built:{built_folds}"
                )
            progress.print_line(
                paint(tag, "green", enabled=color_enabled, bold=True)
                + f" seed {meta['seed_order']}/{len(seeds)} | "
                f"模型來源 {meta['source_order']}/{meta['source_count']} {meta['dl_id']} "
                + f"| replay targets={len(meta['replay_arms'])} "
                + f"| epoch={artifacts['selected_epoch']} "
                + f"| elapsed={_format_elapsed(artifacts.get('training_elapsed_sec', trained['wall_elapsed_sec']))}"
                + fold_note
                + cache_note
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

    def submit_trainings() -> None:
        while pending_trainings and len(training_futures) < int(cfg.gpu_train_workers):
            meta = pending_trainings.popleft()
            meta["submitted_at"] = time.perf_counter()
            future = training_executor.submit(_train_one_unit, meta)
            training_futures[future] = meta
            pit_progress = _pit_saved_fold_progress(meta)
            if pit_progress is None:
                action_tag = "[TRAIN]"
                resume_note = ""
            else:
                saved_folds, expected_folds = pit_progress
                if saved_folds >= expected_folds:
                    action_tag = "[PIT VERIFY/RECOVER]"
                elif saved_folds > 0:
                    action_tag = "[TRAIN RESUME]"
                else:
                    action_tag = "[TRAIN]"
                resume_note = f" | saved checkpoints={saved_folds}/{expected_folds}"
            progress_update(
                paint(action_tag, "cyan", enabled=color_enabled, bold=True)
                + f" seed {meta['seed_order']}/{len(seeds)} | "
                f"模型來源 {meta['source_order']}/{meta['source_count']} {meta['dl_id']} "
                + f"| replay targets={len(meta['replay_arms'])} "
                + resume_note
                + f" | GPU processes={len(training_futures)}/{cfg.gpu_train_workers} "
                + f"| completed={done_units}/{total_units}"
            )
        if training_futures:
            write_progress_manifest()

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
                training_labels = ", ".join(
                    _training_progress_label(meta, now=now, seed_count=len(seeds))
                    for meta in training_futures.values()
                )
                replay_labels = ", ".join(
                    f"seed {meta['seed_order']}/{len(seeds)} {meta['name']} "
                    f"{_format_elapsed(now-float(meta.get('submitted_at', now)))}"
                    for meta in replay_futures.values()
                )
                tag = "[TRAIN]" if training_futures else "[REPLAY]"
                details = []
                if training_labels:
                    details.append(training_labels)
                if replay_labels:
                    details.append("replay: " + replay_labels)
                if ready_replays:
                    details.append(f"replay queued={len(ready_replays)}")
                progress_update(
                    paint(tag, "cyan", enabled=color_enabled, bold=True)
                    + f" GPU running={len(training_futures)}/{cfg.gpu_train_workers}"
                    + (f" | {' | '.join(details)}" if details else "")
                    + f" | CPU replay={len(replay_futures)}/{cfg.cpu_replay_workers}"
                    + f" | total={_format_elapsed(now-started_total)}"
                )
                next_print = now + float(cfg.progress_interval_seconds)
            if pending_trainings or training_futures or ready_replays or replay_futures:
                time.sleep(0.25)
        harvest_replays()
    except BaseException as exc:
        _terminate_active_trainers(active_trainers, active_trainers_lock)
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
        progress.print_line(
            paint("[FAILED]", "red", enabled=color_enabled, bold=True)
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
        _atomic_write_text(run_root / REPORT_FILENAME, report_text)
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
            paint("[FAILED]", "red", enabled=console_color_enabled(), bold=True)
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
    print(f"\n總耗時：{_format_elapsed(summary['elapsed_sec'])}")
    cleanup_tag = paint("暫存清理", "green", enabled=color_enabled, bold=True)
    print(
        f"{cleanup_tag}：checkpoints={'保留' if cfg.keep_checkpoints else '已清除'}｜"
        f"scores={'保留' if cfg.keep_scores else '已清除'}｜"
        f"replay details={'保留' if cfg.keep_replay_details else '已清除'}"
        + ("｜2021 shared checkpoint cache=保留" if cfg.initial_checkpoint_cache_root else "")
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
