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
import random
import shutil
import subprocess
import sys
from threading import Lock
import time
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
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
from core.file_integrity import canonical_json_sha256
from core.console_report import (
    console_color_enabled,
    paint,
    project_relative_display_path,
    render_table,
    render_title,
)
from core.display_common import InlineProgress
from core.strategy_comparison import StrategyComparisonArm
from filters.breakout_quality.artifacts import compute_file_sha256
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
from filters.breakout_quality.strategy_compare_replay import run_standalone_baseline
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from core.report_style import (
    signal_for_delta,
    terminal_signal,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.strategy_comparison import (
    _arm_runtime_spec,
    _find_reusable_baseline_source,
    _load_direct_selection_r,
    collect_artifact_status,
)
from filters.breakout_quality.strategy_compare_preparation import (
    model_upstream_prerequisite_blockers,
    prepare_strategy_parameter_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILENAME = "robustness_report.md"
SUMMARY_FILENAME = "robustness_summary.json"
SEED_RESULTS_FILENAME = "seed_results.csv"
SEED_YEARLY_RESULTS_FILENAME = "seed_yearly_returns.csv"
MANIFEST_FILENAME = "manifest.json"
LATEST_FILENAME = "latest.json"
ATTRIBUTION_SOURCE_DIRNAME = "attribution_source"
ATTRIBUTION_SOURCE_SCHEMA_VERSION = 2
ROBUSTNESS_SCHEMA_VERSION = 8
ROBUSTNESS_SCIENTIFIC_CONTRACT_VERSION = 1
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON根節點必須是object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )



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


def _required_parameter_sources(
    fixed_arms: tuple[StrategyComparisonArm, ...],
    stochastic_arms: tuple[StrategyComparisonArm, ...],
) -> tuple[str, ...]:
    return tuple(sorted({arm.param_source for arm in (*fixed_arms, *stochastic_arms)}))


def _training_source_groups(
    stochastic_arms: tuple[StrategyComparisonArm, ...],
) -> tuple[tuple[str, tuple[tuple[int, StrategyComparisonArm], ...]], ...]:
    groups: dict[str, list[tuple[int, StrategyComparisonArm]]] = {}
    order: list[str] = []
    for arm_order, arm in enumerate(stochastic_arms, start=1):
        dl_id = str(arm.dl_id or "").strip()
        if not dl_id:
            raise ValueError(f"stochastic arm缺少dl_id: {arm.arm_id}")
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
    rows: list[tuple[str, str, str]] = []
    blockers: list[str] = []
    seen: set[tuple[str, str]] = set()
    for arm in stochastic_arms:
        dl = settings.dl_sources[str(arm.dl_id)]
        key = (str(dl.filter_id), str(dl.experiment_profile))
        if key in seen:
            continue
        seen.add(key)
        reasons = model_upstream_prerequisite_blockers(
            PROJECT_ROOT,
            filter_id=str(dl.filter_id),
            model_architecture=str(dl.model_architecture),
            experiment_profile=str(dl.experiment_profile),
            dataset=str(settings.dataset),
            max_tickers=0,
        )
        if reasons:
            blockers.extend(reasons)
            rows.append((
                "BLOCKED",
                f"model-upstream:{dl.experiment_profile}",
                "；".join(reasons),
            ))
        else:
            profile = get_breakout_quality_experiment_profile(str(dl.experiment_profile))
            if str(profile.training_sample_scope) == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
                description = (
                    "重用canonical Dataset／source OHLCV truth；daily windows與固定target由"
                    "canonical trainer即時計算，不需要legacy market-set工件"
                )
            else:
                description = (
                    "重用canonical Dataset／Continuous Target truth；isolated trainer不得建立"
                    "新的Label／Target定義"
                )
            rows.append((
                "REUSE",
                f"model-upstream:{dl.experiment_profile}",
                description,
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
    seen_filters: set[str] = set()
    for arm in stochastic_arms:
        dl = settings.dl_sources[str(arm.dl_id)]
        filter_id = str(dl.filter_id)
        if filter_id in seen_filters:
            continue
        seen_filters.add(filter_id)
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
        runtime_periods.append((
            pd.Timestamp(str(outer["oos_start_date"])).normalize(),
            pd.Timestamp(str(outer["effective_oos_end_date"])).normalize(),
        ))
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
    required_sources = _required_parameter_sources(fixed_arms, stochastic_arms)
    param_rows, param_blockers = _parameter_plan_rows(
        settings=settings, status=status, required_sources=required_sources
    )
    upstream_rows, upstream_blockers = _model_upstream_rows(settings, stochastic_arms)
    blockers = [*param_blockers, *upstream_blockers]
    try:
        start, end = _comparison_period_from_upstream(settings, stochastic_arms, status)
        period_text = f"{start} ～ {end}"
        period_error = None
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        start = end = None
        period_text = f"BLOCKED: {type(exc).__name__}: {exc}"
        period_error = str(exc)
        blockers.append(str(exc))

    pending_param = any(row[0] in {"BUILD", "REBUILD"} for row in param_rows)
    overall = "BLOCKED" if blockers else "PREPARABLE" if pending_param else "READY"
    rows = [*param_rows, *upstream_rows]
    for arm in fixed_arms:
        rows.append((
            "RUN/REUSE",
            arm.name,
            "固定DL-off baseline；identity一致時重用，否則正式回放一次",
        ))
    for arm in stochastic_arms:
        rows.append((
            "TRAIN+REPLAY",
            arm.name,
            f"{cfg.seed_count}個deterministic generated seeds；isolated canonical trainer + same strategy replay；若scientific observation已完成但compact attribution缺少，僅重建同一observation的歸因工件",
        ))
    training_sources = _training_source_groups(tuple(stochastic_arms))
    pit_workload_lines: list[str] = [
        f"模型訓練單元       ：{cfg.seed_count * len(training_sources)}（{len(training_sources)} unique DL sources × {cfg.seed_count} seeds）",
        f"策略Replay單元     ：{cfg.seed_count * len(stochastic_arms)}（{len(stochastic_arms)} stochastic arms × {cfg.seed_count} seeds）",
    ]
    if settings.profile_id == "selection_pit" and start is not None and end is not None:
        fold_counts = []
        for dl_id, _arm_entries in training_sources:
            dl = settings.dl_sources[str(dl_id)]
            workflow = get_breakout_quality_workflow_settings(
                experiment_profile=str(dl.experiment_profile)
            )
            fold_counts.append(
                _pit_fold_count_for_period(start, end, int(workflow.point_in_time_fold_months))
            )
        unique_fold_counts = sorted(set(fold_counts))
        fold_text = str(unique_fold_counts[0]) if len(unique_fold_counts) == 1 else "/".join(map(str, unique_fold_counts))
        total_fold_trainings = cfg.seed_count * sum(fold_counts)
        pit_workload_lines += [
            f"PIT folds          ：{fold_text} folds / DL source / seed（只建策略比較期間所需fold）",
            f"預計fold trainings ：{total_fold_trainings}",
        ]
    color_enabled = console_color_enabled()
    action_colors = {
        "READY": "green",
        "REUSE": "green",
        "BUILD": "yellow",
        "REBUILD": "yellow",
        "PREPARABLE": "yellow",
        "BLOCKED": "red",
        "RUN/REUSE": "cyan",
        "TRAIN+REPLAY": "cyan",
    }
    colored_rows = [
        (
            paint(action, action_colors.get(str(action), "gray"), enabled=color_enabled, bold=True),
            item,
            description,
        )
        for action, item, description in rows
    ]
    overall_color = {"READY": "green", "PREPARABLE": "yellow", "BLOCKED": "red"}.get(overall, "gray")
    lines = [
        render_title(f"{cfg.label} 本次執行計畫"),
        f"整體狀態          ：{paint(overall, overall_color, enabled=color_enabled, bold=True)}",
        f"比較階段          ：{settings.profile_label} ({settings.profile_id})",
        f"共同策略期間      ：{period_text}",
        f"Seed數量          ：{cfg.seed_count}",
        f"Seed generator    ：deterministic / generator_seed={cfg.seed_generator_seed}",
        f"GPU training      ：workers={cfg.gpu_train_workers}",
        f"CPU strategy replay：workers={cfg.cpu_replay_workers}",
        f"Console mode        ：{cfg.console_mode}",
        f"年度報表顯示        ：{'on' if cfg.yearly_report else 'off'}（raw yearly永遠保留）",
        f"永久保留模型/Scores/Replay/Attribution：{cfg.keep_checkpoints}/{cfg.keep_scores}/{cfg.keep_replay_details}/{cfg.keep_attribution_source}",
        *pit_workload_lines,
        render_table(("動作", "項目", "說明"), colored_rows),
    ]
    return "\n".join(lines), {
        "overall_status": overall,
        "blockers": blockers,
        "required_param_sources": required_sources,
        "comparison_period": None if start is None else {"start": start, "end": end},
        "period_error": period_error,
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


def _point_in_time_training_policy_snapshot(*, experiment_profile: str) -> dict[str, Any]:
    workflow = get_breakout_quality_workflow_settings(experiment_profile=str(experiment_profile))
    return {
        "fold_months": int(workflow.point_in_time_fold_months),
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
) -> dict[str, Any]:
    robustness = get_strategy_multi_seed_robustness_settings(robustness_id)
    settings = get_strategy_comparison_settings(robustness.profile_id)
    fixed, stochastic = _robustness_arms(settings, robustness)
    seeds = resolve_multi_seed_values(
        seed_count=robustness.seed_count,
        generator_seed=robustness.seed_generator_seed,
    )
    resolved_period = dict(comparison_period or {})
    if not resolved_period:
        resolved_period = {"start": settings.start_date, "end": settings.end_date}
    identities = dict(artifact_identities or {})
    required_param_sources = sorted({arm.param_source for arm in (*fixed, *stochastic)})
    parameter_identities = {}
    for source_id in required_param_sources:
        raw_identity = identities.get(f"param:{source_id}")
        if isinstance(raw_identity, dict):
            parameter_identities[source_id] = {
                key: raw_identity.get(key)
                for key in ("sha256", "coverage_start", "coverage_end")
            }
    reference_baselines: dict[str, dict[str, Any]] = {}
    for reference_key, spec in robustness.romd_reference_baselines.items():
        matches = [
            arm for arm in fixed
            if arm.param_source == spec["param_source"] and arm.rule_policy == spec["rule_policy"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "multi-seed RoMD reference無法唯一解析fixed baseline: "
                f"reference={reference_key}, matches={len(matches)}"
            )
        matched = matches[0]
        reference_baselines[str(reference_key)] = {
            "arm_id": matched.arm_id, "name": matched.name,
            "param_source": matched.param_source, "rule_policy": matched.rule_policy,
        }
    scientific_reference_baselines = {
        key: {
            field: value
            for field, value in spec.items()
            if field != "name"
        }
        for key, spec in reference_baselines.items()
    }
    scientific = {
        "scientific_contract_version": ROBUSTNESS_SCIENTIFIC_CONTRACT_VERSION,
        "robustness_id": robustness.robustness_id,
        "profile_id": settings.profile_id,
        "dataset": settings.dataset,
        "dataset_identity": _dataset_identity_snapshot(settings.dataset),
        "param_policy": settings.param_policy,
        "max_positions": int(settings.max_positions),
        "rotation": settings.rotation,
        "comparison_period": resolved_period,
        "parameter_artifact_identities": parameter_identities,
        "seed_count": int(robustness.seed_count),
        "seed_generator_seed": int(robustness.seed_generator_seed),
        "resolved_seeds": list(seeds),
        "training_defaults": _continuous_training_defaults_snapshot(),
        "romd_reference_baselines": scientific_reference_baselines,
        "fixed_arms": [
            {"arm_id": arm.arm_id, "param_source": arm.param_source, "rule_policy": arm.rule_policy}
            for arm in fixed
        ],
        "stochastic_arms": [
            {
                "arm_id": arm.arm_id, "param_source": arm.param_source, "rule_policy": arm.rule_policy,
                "dl_id": arm.dl_id, "dl_runtime_mode": arm.dl_runtime_mode,
                "dl_runtime_options": dict(arm.dl_runtime_options or {}),
                "dl_source": {
                    "dl_id": settings.dl_sources[str(arm.dl_id)].dl_id,
                    "filter_id": settings.dl_sources[str(arm.dl_id)].filter_id,
                    "model_architecture": settings.dl_sources[str(arm.dl_id)].model_architecture,
                    "experiment_profile": settings.dl_sources[str(arm.dl_id)].experiment_profile,
                    "threshold": settings.dl_sources[str(arm.dl_id)].threshold,
                    "score_source": settings.dl_sources[str(arm.dl_id)].score_source,
                },
                "experiment_profile": get_breakout_quality_experiment_profile(
                    settings.dl_sources[str(arm.dl_id)].experiment_profile
                ).as_manifest_payload(),
                "point_in_time_training_policy": (
                    _point_in_time_training_policy_snapshot(
                        experiment_profile=settings.dl_sources[str(arm.dl_id)].experiment_profile
                    )
                    if str(settings.dl_sources[str(arm.dl_id)].score_source) == "selection_point_in_time"
                    else None
                ),
            }
            for arm in stochastic
        ],
    }
    contract = {
        **scientific,
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


def _run_root(contract: dict[str, Any]) -> Path:
    cfg = get_strategy_multi_seed_robustness_settings(str(contract["robustness_id"]))
    return (PROJECT_ROOT / cfg.output_root / str(contract["fingerprint"])).resolve()


def _model_work_root(contract: dict[str, Any]) -> Path:
    cfg = get_strategy_multi_seed_robustness_settings(str(contract["robustness_id"]))
    return (PROJECT_ROOT / cfg.model_work_root / str(contract["fingerprint"])).resolve()


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
        },
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

    for key in ("model_sha256", "score_sha256"):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["arm_order", "seed_order"], kind="mergesort").to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _resolved_period(settings, status: dict[str, Any]) -> tuple[str, str]:
    period = dict(status.get("comparison_period") or {})
    start = str(period.get("start") or settings.start_date or "").strip()
    end = str(period.get("end") or settings.end_date or "").strip()
    if not start or not end:
        raise RuntimeError("multi-seed robustness無法解析共同策略比較期間")
    return start, end


def _validate_training_artifacts(
    *, arm: StrategyComparisonArm, seed: int, model_dir: Path, research_dir: Path, settings,
    comparison_start: str, comparison_end: str,
) -> dict[str, Any]:
    dl = settings.dl_sources[str(arm.dl_id)]
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
    if execution_start > score_available_from:
        raise ValueError("multi-seed isolated score時間契約不一致")
    return {
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


def _training_command(
    *, arm: StrategyComparisonArm, seed: int, model_dir: Path, research_dir: Path, settings,
    comparison_start: str, comparison_end: str,
) -> list[str]:
    dl = settings.dl_sources[str(arm.dl_id)]
    if str(dl.score_source) == "selection_point_in_time":
        return [
            sys.executable, "-m", "tools.filters.breakout_quality.build_point_in_time_scores",
            "--filter-id", dl.filter_id,
            "--model-architecture", dl.model_architecture,
            "--experiment-profile", dl.experiment_profile,
            "--seed", str(int(seed)),
            "--score-start-date", str(comparison_start),
            "--score-end-date", str(comparison_end),
            "--point-in-time-dir-override", str(model_dir.resolve()),
        ]
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
    dl = settings.dl_sources[str(arm.dl_id)]
    is_selection_pit = str(dl.score_source) == "selection_point_in_time"
    started = time.perf_counter()
    artifacts = None
    if bool(job.get("reuse_completed")) and model_dir.is_dir() and research_dir.is_dir():
        try:
            artifacts = _validate_training_artifacts(
                arm=arm,
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
        return {
            "artifacts": artifacts,
            "reused": True,
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
                seed=seed,
                model_dir=model_dir,
                research_dir=research_dir,
                settings=settings,
                comparison_start=str(job["comparison_start"]),
                comparison_end=str(job["comparison_end"]),
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
        seed=seed,
        model_dir=model_dir,
        research_dir=research_dir,
        settings=settings,
        comparison_start=str(job["comparison_start"]),
        comparison_end=str(job["comparison_end"]),
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
) -> deque[dict[str, Any]]:
    units: deque[dict[str, Any]] = deque()
    source_groups = _training_source_groups(stochastic_arms)
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
            })
    return units


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _pit_saved_fold_progress(meta: dict[str, Any]) -> tuple[int, int] | None:
    arm: StrategyComparisonArm = meta["arm"]
    settings = meta["settings"]
    dl = settings.dl_sources[str(arm.dl_id)]
    if str(dl.score_source) != "selection_point_in_time":
        return None
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(dl.experiment_profile)
    )
    expected = _pit_fold_count_for_period(
        str(meta["comparison_start"]),
        str(meta["comparison_end"]),
        int(workflow.point_in_time_fold_months),
    )
    folds_root = Path(str(meta["model_dir"])).resolve() / "folds"
    if not folds_root.is_dir():
        return 0, expected
    saved = sum(
        1
        for path in folds_root.iterdir()
        if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
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
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["arm_order", "seed_order", "year"], kind="mergesort").to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _baseline_work_dir(run_root: Path, arm: StrategyComparisonArm) -> Path:
    return run_root / "work" / "baselines" / arm.arm_id


def _run_fixed_baselines(*, settings, status, run_root: Path, fixed_arms) -> dict[str, dict[str, Any]]:
    start, end = _resolved_period(settings, status)
    results: dict[str, dict[str, Any]] = {}
    for arm in fixed_arms:
        path_text = str((status.get("resolved_parameter_paths") or {}).get(arm.param_source) or "")
        if not path_text:
            raise RuntimeError(f"robustness baseline缺少param source: {arm.param_source}")
        reusable_dir = _find_reusable_baseline_source(
            root=PROJECT_ROOT, settings=settings, status=status, off_arm=arm
        )
        if reusable_dir is not None:
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
        output_dir = _baseline_work_dir(run_root, arm)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        all_off = arm.rule_policy == "all_off"
        payload = run_standalone_baseline(
            project_root=PROJECT_ROOT,
            dataset=settings.dataset,
            params_path=path_text,
            param_policy=settings.param_policy,
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
    payload = run_comparison(
        project_root=PROJECT_ROOT, dataset=settings.dataset, params_path=str(job["params_path"]),
        param_policy=settings.param_policy, max_positions=settings.max_positions,
        enable_rotation=settings.rotation == "on", comparison_mode=runtime_spec["comparison_mode"],
        ranking_policy=runtime_spec["ranking_policy"], ranking_options=dict(arm.dl_runtime_options or {}),
        optional_entry_filter_policy=(OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT),
        filter_id=dl.filter_id, score_source=dl.score_source, model_architecture=dl.model_architecture,
        experiment_profile=dl.experiment_profile, threshold=dl.threshold, output_dir_override=pair_dir,
        comparison_start_date=start, comparison_end_date=end, quiet=True,
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
        baseline_reuse_dir=baseline_dir,
        continuous_score_path_override=(None if is_selection else str(job["score_path"])),
        continuous_score_execution_start_override=(None if is_selection else str(job["score_execution_start"])),
        selection_pit_score_path_override=(str(job["score_path"]) if is_selection else None),
        selection_pit_manifest_path_override=(str(job["score_manifest_path"]) if is_selection else None),
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
    result = {
        "arm_id": arm.arm_id, "name": arm.name, "seed": int(job["seed"]),
        "arm_order": int(job["arm_order"]), "seed_order": int(job["seed_order"]),
        "selected_epoch": int(job.get("selected_epoch", 0) or 0),
        "fold_count": job.get("fold_count"),
        "training_elapsed_sec": float(job.get("training_elapsed_sec", 0.0) or 0.0),
        "replay_elapsed_sec": round(time.perf_counter() - started, 3),
        "model_sha256": str(job.get("model_sha256") or "") or None,
        "score_sha256": str(job.get("score_sha256") or "") or None,
        **{key: metrics.get(key) for _label, key, _unit in MEAN_METRICS},
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
        "pair_count": int(len(left_values) * len(right_values)),
    }


def _paired_seed_translation_diagnostic(
    *,
    seed_frame: pd.DataFrame,
    left: StrategyComparisonArm,
    right: StrategyComparisonArm,
    resolved_seeds: tuple[int, ...] = (),
) -> dict[str, Any] | None:
    """Describe whether same-seed selection-R improvements translate to strategy gains."""
    if left.param_source != right.param_source or left.rule_policy != right.rule_policy:
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
                seed_frame=seed_frame, left=left, right=right,
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
    upgraded["report_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return upgraded


def _robustness_summary(
    *, contract: dict[str, Any], fixed_results: dict[str, dict[str, Any]],
    seed_frame: pd.DataFrame, seed_yearly_frame: pd.DataFrame,
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
        })
    for arm in stochastic:
        rows = seed_frame[seed_frame["arm_id"] == arm.arm_id].copy()
        mean_rows.append({
            "arm_id": arm.arm_id, "name": arm.name, "type": "Multi-seed", "n": int(len(rows)),
            **{key: _mean_or_none(rows[key]) for _label, key, _unit in MEAN_METRICS},
        })

    fixed_by_arm_id = {row["arm_id"]: row for row in mean_rows if row["type"] == "Fixed"}
    references = dict(contract.get("romd_reference_baselines") or {})
    min_baseline = fixed_by_arm_id.get(str(dict(references.get("min") or {}).get("arm_id") or ""))
    full_baseline = fixed_by_arm_id.get(str(dict(references.get("full") or {}).get("arm_id") or ""))
    if min_baseline is None or full_baseline is None:
        raise ValueError("multi-seed summary缺少config指定的Min/Full fixed baseline")

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
            "beats_min_count": int(np.sum(values > float(min_baseline["return_over_max_drawdown"]))),
            "beats_full_count": int(np.sum(values > float(full_baseline["return_over_max_drawdown"]))),
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
    return {
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


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def esc(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(esc(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(esc(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def render_multi_seed_robustness_report(summary: dict[str, Any]) -> str:
    mean_headers = ["比較對象", "類型", "N"] + [f"{label} Mean" for label, _key, _unit in MEAN_METRICS]
    mean_rows = []
    for row in summary["mean_strategy_metrics"]:
        cells = [row["name"], row["type"], str(row["n"])]
        for _label, key, unit in MEAN_METRICS:
            digits = 1 if key == "trade_count" else 4 if key == "log_r_squared" else 2
            cells.append(_fmt(row.get(key), digits=digits, suffix=unit))
        mean_rows.append(cells)
    romd_headers = ["比較對象", "N", "Mean", "Median", "Std", "CV", "Min", "P25", "P75", "Max", "勝Min", "勝Full"]
    romd_rows = []
    for row in summary["romd_statistics"]:
        n = int(row["n"])
        romd_rows.append([
            row["name"], str(n), _fmt(row["mean"]), _fmt(row["median"]), _fmt(row["std"]),
            _fmt(row["cv"]), _fmt(row["min"]), _fmt(row["p25"]), _fmt(row["p75"]), _fmt(row["max"]),
            "-" if row["beats_min_count"] is None else f"{row['beats_min_count']}/{n}",
            "-" if row["beats_full_count"] is None else f"{row['beats_full_count']}/{n}",
        ])
    contract = dict(summary["contract"])
    lines = [
        f"# {contract.get('label') or 'Multiple-seed Robustness'}", "",
        f"- Profile：`{contract['profile_id']}`",
        f"- Seeds：`{contract['seed_count']}`（deterministic generated；不作best-seed選擇）",
        f"- Scientific fingerprint：`{contract['fingerprint']}`",
        f"- Report schema：`{summary['schema_version']}`（不影響scientific fingerprint）", "",
        "## 1. 平均策略績效", "", _markdown_table(mean_headers, mean_rows), "",
        "## 2. RoMD完整統計", "", _markdown_table(romd_headers, romd_rows),
    ]

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

    next_section = 3
    if paired:
        lines += ["", f"## {next_section}. 設定中的同seed contrasts", ""]
        for item in paired:
            left = str(item.get("left") or "Left")
            right = str(item.get("right") or "Right")
            lines += [f"### {right} − {left}", ""]
            description = str(item.get("description") or "").strip()
            if description:
                lines += [f"- 用途：{description}"]
            same = item.get("romd_same_seed")
            if isinstance(same, dict):
                n = int(same["n"])
                lines += [
                    f"- RoMD：{same['right']} > {same['left']} {same['right_gt_left_count']}/{n}；"
                    f"{same['left']} > {same['right']} {same['left_gt_right_count']}/{n}；Tie {same['tie_count']}/{n}。",
                    f"- ΔRoMD Mean {_fmt(same['right_minus_left_mean'])}；Median {_fmt(same['right_minus_left_median'])}；"
                    f"Std {_fmt(same['right_minus_left_std'])}；Min {_fmt(same['right_minus_left_min'])}；"
                    f"P25 {_fmt(same['right_minus_left_p25'])}；P75 {_fmt(same['right_minus_left_p75'])}；Max {_fmt(same['right_minus_left_max'])}。",
                ]
            direct_same = item.get("direct_selection_r_same_seed")
            if isinstance(direct_same, dict):
                n = int(direct_same["n"])
                lines += [
                    f"- DL選擇R：{direct_same['right']} > {direct_same['left']} {direct_same['right_gt_left_count']}/{n}；"
                    f"{direct_same['left']} > {direct_same['right']} {direct_same['left_gt_right_count']}/{n}；Tie {direct_same['tie_count']}/{n}。",
                    f"- ΔDL選擇R Mean {_fmt(direct_same['right_minus_left_mean'], suffix=' R')}；"
                    f"Median {_fmt(direct_same['right_minus_left_median'], suffix=' R')}；Std {_fmt(direct_same['right_minus_left_std'], suffix=' R')}。",
                ]
            translation = item.get("selection_r_to_strategy")
            if isinstance(translation, dict):
                pair_rows = []
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
                    pair_rows.append([
                        f"S{int(row['seed_index'])}",
                        _fmt(selection_delta, suffix=" R"),
                        _fmt(row["right_minus_left_total_return_pct"], suffix="%"),
                        _fmt(row["right_minus_left_max_drawdown_pct"], suffix="%"),
                        _fmt(romd_delta),
                        _fmt(row["right_minus_left_expected_value_r"], suffix=" R"),
                        verdict,
                    ])
                if pair_rows:
                    lines += ["", _markdown_table(
                        ["Seed", "ΔDL選擇R", "ΔReturn", "ΔMDD", "ΔRoMD", "ΔEV", "方向"], pair_rows
                    )]
                positive_n = int(translation["selection_r_positive_count"])
                translated_n = int(translation["selection_r_positive_romd_positive_count"])
                lines += [
                    f"- ΔDL選擇R>0：{positive_n}/{translation['n']}；其中ΔRoMD>0：{translated_n}/{positive_n if positive_n else 0}。",
                    f"- 方向一致：{translation['sign_concordant_count']}/{translation['sign_non_tie_n']}；"
                    f"方向相反：{translation['sign_discordant_count']}/{translation['sign_non_tie_n']}。",
                    f"- Spearman(ΔDL選擇R, ΔRoMD)={_fmt(translation.get('selection_r_delta_vs_romd_spearman'), digits=3)}；"
                    f"Spearman(ΔDL選擇R, ΔReturn)={_fmt(translation.get('selection_r_delta_vs_return_spearman'), digits=3)}。",
                ]
            compare = item.get("romd_distribution")
            if isinstance(compare, dict):
                lines += [
                    f"- Cross-seed P({compare['left']} > {compare['right']})="
                    f"{float(compare['pairwise_left_gt_right_probability'])*100:.2f}%（{compare['pair_count']} pairs）。"
                ]
            annual_pair = list(item.get("yearly_same_seed") or [])
            if annual_pair:
                pair_rows = [[
                    str(row["year"]), str(row["n"]), _fmt(row["right_minus_left_mean"], suffix="%"),
                    _fmt(row["right_minus_left_median"], suffix="%"), _fmt(row["right_minus_left_std"], suffix="%"),
                    f"{row['right_gt_left_count']}/{row['n']}", f"{row['left_gt_right_count']}/{row['n']}", f"{row['tie_count']}/{row['n']}",
                ] for row in annual_pair]
                lines += ["", _markdown_table(
                    ["年度", "N", "Δ右-左 Mean", "Median", "Std", "右勝", "左勝", "Tie"], pair_rows
                )]
            lines += [""]
        next_section += 1

    yearly = list(summary.get("yearly_statistics") or [])
    if yearly:
        lines += ["", f"## {next_section}. 歷年報酬完整統計", ""]
        rows = []
        for row in sorted(yearly, key=lambda x: (int(x["year"]), str(x["type"]), str(x["name"]))):
            rows.append([
                str(row["year"]) + ("" if row.get("is_complete_year") else "*"), row["name"], row["type"], str(row["n"]),
                _fmt(row["mean"], suffix="%"), _fmt(row["median"], suffix="%"), _fmt(row["std"], suffix="%"),
                _fmt(row["min"], suffix="%"), _fmt(row["p25"], suffix="%"), _fmt(row["p75"], suffix="%"), _fmt(row["max"], suffix="%"),
            ])
        lines += [_markdown_table(["年度", "比較對象", "類型", "N", "Mean", "Median", "Std", "Min", "P25", "P75", "Max"], rows), "", "* 非完整年度。"]
        next_section += 1

    references = dict(contract.get("romd_reference_baselines") or {})
    min_name = str(dict(references.get("min") or {}).get("name") or "Min baseline")
    full_name = str(dict(references.get("full") or {}).get("name") or "Full baseline")
    lines += ["", f"## {next_section}. 限制", "",
        "- resolved seeds只用於重現；不得挑best seed或依本報表組seed ensemble。",
        f"- {full_name}／{min_name}沒有DL訓練seed，因此以固定正式baseline值放入同一表。",
        "- 同一DL source／seed只訓練一次，允許fan-out到不同runtime selector replay；此reuse不改變模型scientific condition。",
        "- Selection PIT與Forward-OOS robustness均只評估既定scientific condition；不得依結果回頭調整training semantics。", ""]
    return "\n".join(lines)


def _color_delta(text: str, value: Any, *, preference: str = "higher") -> str:
    return terminal_signal(text, signal_for_delta(value, preference=preference))


def _print_report_tables(summary: dict[str, Any]) -> None:
    color_enabled = console_color_enabled()
    print("\n" + render_title(str(dict(summary.get("contract") or {}).get("label") or "Multiple-seed robustness")))
    mean_rows = []
    for row in summary["mean_strategy_metrics"]:
        mean_rows.append([
            row["name"], row["type"], row["n"],
            _fmt(row.get("total_return_pct"), suffix="%"),
            _fmt(row.get("max_drawdown_pct"), suffix="%"),
            _fmt(row.get("return_over_max_drawdown")),
            _fmt(row.get("annual_return_pct"), suffix="%"),
            _fmt(row.get("expected_value_r"), suffix=" R"),
            _fmt(row.get("avg_exposure_pct"), suffix="%"),
            _fmt(row.get("direct_selection_r"), suffix=" R"),
        ])
    print("\n1. 平均策略績效")
    print(render_table(
        ["比較對象", "類型", "N", "Return", "MDD", "RoMD", "Annual", "EV", "Exposure", "DL選擇R"],
        mean_rows,
    ))

    romd_rows = []
    for row in summary["romd_statistics"]:
        n = int(row["n"])
        romd_rows.append([
            row["name"], n, _fmt(row["mean"]), _fmt(row["median"]), _fmt(row["std"]),
            _fmt(row["cv"]), _fmt(row["min"]), _fmt(row["p25"]), _fmt(row["p75"]), _fmt(row["max"]),
            "-" if row["beats_min_count"] is None else f"{row['beats_min_count']}/{n}",
            "-" if row["beats_full_count"] is None else f"{row['beats_full_count']}/{n}",
        ])
    print("\n2. RoMD完整統計")
    print(render_table(["比較對象", "N", "Mean", "Median", "Std", "CV", "Min", "P25", "P75", "Max", "勝Min", "勝Full"], romd_rows))

    paired = list(summary.get("paired_comparisons") or [])
    if not paired:
        same = summary.get("romd_same_seed_comparison")
        if isinstance(same, dict):
            paired = [{
                "contrast_id": "legacy_default",
                "left": same.get("left"), "right": same.get("right"),
                "romd_same_seed": same,
                "direct_selection_r_same_seed": summary.get("direct_selection_r_same_seed_comparison"),
                "selection_r_to_strategy": summary.get("selection_r_to_strategy_same_seed_translation"),
            }]
    if paired:
        print("\n3. 設定中的同seed contrasts")
        for item in paired:
            same = item.get("romd_same_seed")
            if not isinstance(same, dict):
                continue
            n = int(same["n"])
            delta = same["right_minus_left_mean"]
            print(f"\n{same['right']} − {same['left']}")
            print(
                f"RoMD右勝左={same['right_gt_left_count']}/{n} | 左勝右={same['left_gt_right_count']}/{n} | "
                f"Tie={same['tie_count']}/{n} | ΔRoMD Mean="
                + _color_delta(_fmt(delta), delta)
                + f" | Median {_fmt(same['right_minus_left_median'])} | Std {_fmt(same['right_minus_left_std'])}"
            )
            direct = item.get("direct_selection_r_same_seed")
            if isinstance(direct, dict):
                d = direct["right_minus_left_mean"]
                print(
                    f"DL選擇R右勝左={direct['right_gt_left_count']}/{direct['n']} | ΔMean="
                    + _color_delta(_fmt(d, suffix=" R"), d)
                )
            translation = item.get("selection_r_to_strategy")
            if isinstance(translation, dict):
                positive_n = int(translation["selection_r_positive_count"])
                translated_n = int(translation["selection_r_positive_romd_positive_count"])
                print(
                    f"ΔDL選擇R>0={positive_n}/{translation['n']} | 其中ΔRoMD>0="
                    f"{translated_n}/{positive_n if positive_n else 0} | 方向一致="
                    f"{translation['sign_concordant_count']}/{translation['sign_non_tie_n']}"
                )

    yearly = list(summary.get("yearly_statistics") or [])
    if yearly:
        print("\n歷年報酬 Mean")
        names = [row["name"] for row in summary["mean_strategy_metrics"]]
        by_key = {(int(row["year"]), str(row["name"])): row for row in yearly}
        table_rows = []
        for year in sorted({int(row["year"]) for row in yearly}):
            sample = next(row for row in yearly if int(row["year"]) == year)
            cells = [str(year) + ("" if sample.get("is_complete_year") else "*")]
            for name in names:
                record = by_key.get((year, name))
                cells.append("-" if record is None else _fmt(record["mean"], suffix="%"))
            table_rows.append(cells)
        print(render_table(["年度", *names], table_rows))
        print("* 非完整年度")


def show_multi_seed_robustness_status(*, robustness_id: str | None = None) -> None:
    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    settings = get_strategy_comparison_settings(cfg.profile_id)
    status = collect_artifact_status(settings=settings)
    fixed, stochastic = _robustness_arms(settings, cfg)
    plan_text, plan = _render_robustness_execution_plan(
        cfg=cfg, settings=settings, status=status, fixed_arms=fixed, stochastic_arms=stochastic
    )
    print("\n" + plan_text)
    if plan["comparison_period"] is None or plan["overall_status"] != "READY":
        print("Fingerprint       ：前置完成後依正式param identity與共同期間解析")
        return
    contract = build_multi_seed_robustness_contract(
        robustness_id=cfg.robustness_id, comparison_period=dict(plan["comparison_period"]),
        artifact_identities=dict(status.get("artifact_identities") or {}),
    )
    run_root = _run_root(contract)
    existing = _load_seed_results(run_root / SEED_RESULTS_FILENAME)
    seeds = tuple(int(value) for value in contract["resolved_seeds"])
    _validate_seed_results_frame(existing, stochastic_arms=stochastic, seeds=seeds)
    print(f"Fingerprint       ：{contract['fingerprint']}")
    ready_attribution = _attribution_ready_units(
        run_root, stochastic_arms=stochastic, seeds=seeds, fingerprint=str(contract["fingerprint"])
    ) if cfg.keep_attribution_source else set()
    print(f"已完成seed結果    ：{len(existing)}/{len(stochastic) * cfg.seed_count}")
    if cfg.keep_attribution_source:
        print(f"Attribution工件   ：{len(ready_attribution)}/{len(stochastic) * cfg.seed_count}")
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
            )
            _write_json(summary_path, upgraded)
            report.write_text(
                render_multi_seed_robustness_report(upgraded),
                encoding="utf-8",
            )
    if not report.is_file():
        raise FileNotFoundError(f"最新robustness報表不存在: {report}")
    print(report.read_text(encoding="utf-8"))


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


def run_multi_seed_robustness(*, robustness_id: str | None = None, confirm: bool = True) -> dict[str, Any]:
    cfg = get_strategy_multi_seed_robustness_settings(robustness_id)
    if not cfg.enabled:
        raise RuntimeError("Multiple-seed robustness目前由config關閉")
    settings = get_strategy_comparison_settings(cfg.profile_id)
    fixed_arms, stochastic_arms = _robustness_arms(settings, cfg)
    status = collect_artifact_status(settings=settings)
    plan_text, plan = _render_robustness_execution_plan(
        cfg=cfg, settings=settings, status=status, fixed_arms=fixed_arms, stochastic_arms=stochastic_arms
    )
    print("\n" + plan_text)
    if plan["overall_status"] == "BLOCKED":
        raise RuntimeError(
            "Multiple-seed robustness前置BLOCKED；請先由正式模型訓練入口準備上游真理工件，"
            "或修復無builder的策略參數工件。"
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

    status = prepare_strategy_parameter_artifacts(
        project_root=PROJECT_ROOT,
        settings=settings,
        status=status,
        required_source_ids=tuple(plan["required_param_sources"]),
        status_refresher=lambda: collect_artifact_status(settings=settings),
    )
    post_upstream_rows, post_upstream_blockers = _model_upstream_rows(
        settings, stochastic_arms
    )
    del post_upstream_rows
    if post_upstream_blockers:
        raise RuntimeError(
            "Multiple-seed robustness模型上游工件在前置後仍BLOCKED: "
            + "；".join(post_upstream_blockers)
        )
    resolved_params = dict(status.get("resolved_parameter_paths") or {})
    required_param_sources = _required_parameter_sources(fixed_arms, stochastic_arms)
    missing_params = sorted(
        source for source in required_param_sources
        if not resolved_params.get(source) or not Path(resolved_params[source]).is_file()
    )
    if missing_params:
        raise RuntimeError(
            "Multiple-seed robustness策略參數前置完成後仍缺工件: "
            + ", ".join(missing_params)
        )
    comparison_start, comparison_end = _comparison_period_from_upstream(
        settings, stochastic_arms, status
    )
    status = dict(status)
    status["comparison_period"] = {"start": comparison_start, "end": comparison_end}
    contract = build_multi_seed_robustness_contract(
        robustness_id=cfg.robustness_id,
        comparison_period={"start": comparison_start, "end": comparison_end},
        artifact_identities=dict(status.get("artifact_identities") or {}),
    )
    run_root = _run_root(contract)
    model_root = _model_work_root(contract)
    run_root.mkdir(parents=True, exist_ok=True)
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
        existing = _load_seed_results(seed_results_path)
        _validate_seed_results_frame(existing, stochastic_arms=stochastic_arms, seeds=seeds)
        existing_yearly = _load_seed_yearly_results(seed_yearly_results_path)
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
    baseline_by_group = {
        (arm.param_source, arm.rule_policy): fixed_results[arm.arm_id]["baseline_dir"]
        for arm in fixed_arms
    }
    scientific_completed = {
        (str(row.arm_id), int(row.seed))
        for row in existing.itertuples(index=False)
    } if not existing.empty and cfg.reuse_completed else set()
    if scientific_completed:
        yearly_units = set() if existing_yearly.empty else {
            (str(row.arm_id), int(row.seed)) for row in existing_yearly.itertuples(index=False)
        }
        scientific_completed &= yearly_units
    attribution_ready = (
        _attribution_ready_units(
            run_root,
            stochastic_arms=stochastic_arms,
            seeds=seeds,
            fingerprint=str(contract["fingerprint"]),
        )
        if cfg.keep_attribution_source
        else set(scientific_completed)
    )
    completed = set(scientific_completed & attribution_ready)
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
    total_units = len(stochastic_arms) * len(seeds)
    training_sources = _training_source_groups(stochastic_arms)
    training_unit_count = len(training_sources) * len(seeds)
    done_units = len(completed)
    print("\n" + render_title("Multiple-seed robustness 執行"))
    print(
        f"Seeds={len(seeds)} | stochastic arms={len(stochastic_arms)} | replay units={total_units} | "
        f"unique DL sources={len(training_sources)} | train units={training_unit_count}"
    )
    print(f"GPU train workers={cfg.gpu_train_workers} | CPU replay workers={cfg.cpu_replay_workers}")
    if cfg.keep_attribution_source:
        missing_attr = scientific_completed - attribution_ready
        print(
            f"Compact attribution source={len(attribution_ready)}/{total_units} ready"
            + (f" | rebuild={len(missing_attr)}" if missing_attr else "")
        )
    color_enabled = console_color_enabled()
    progress = InlineProgress()
    min_ref_arm = str(dict(dict(contract.get("romd_reference_baselines") or {}).get("min") or {}).get("arm_id") or "")
    min_baseline_romd = float(fixed_results[min_ref_arm]["metrics"]["return_over_max_drawdown"])

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

    for seed_order, seed in enumerate(seeds, start=1):
        for arm_order, arm in enumerate(stochastic_arms, start=1):
            unit_key = (arm.arm_id, int(seed))
            if unit_key in completed:
                progress.print_line(
                    paint(f"[REUSE {done_units}/{total_units}]", "green", enabled=color_enabled, bold=True)
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
            delta_min = romd - min_baseline_romd
            done_tag = paint(
                f"[DONE {done_units}/{total_units}]", "green", enabled=color_enabled, bold=True
            )
            delta_text = terminal_signal(
                f"ΔMin={delta_min:+.2f}",
                signal_for_delta(delta_min, preference="higher"),
                enabled=color_enabled,
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

    def harvest_trainings() -> None:
        finished = [future for future in training_futures if future.done()]
        for future in finished:
            meta = training_futures.pop(future)
            trained = future.result()
            artifacts = dict(trained["artifacts"])
            tag = "[TRAIN REUSE]" if trained["reused"] else "[TRAIN DONE]"
            progress.print_line(
                paint(tag, "green", enabled=color_enabled, bold=True)
                + f" seed {meta['seed_order']}/{len(seeds)} | "
                f"模型來源 {meta['source_order']}/{meta['source_count']} {meta['dl_id']} "
                + f"| replay targets={len(meta['replay_arms'])} "
                + f"| epoch={artifacts['selected_epoch']} "
                + f"| elapsed={_format_elapsed(artifacts.get('training_elapsed_sec', trained['wall_elapsed_sec']))}"
            )
            for arm_order, arm in meta["replay_arms"]:
                baseline_dir = baseline_by_group.get((arm.param_source, arm.rule_policy))
                if not baseline_dir:
                    raise RuntimeError(
                        f"stochastic arm找不到同參數fixed baseline: {arm.param_source}/{arm.rule_policy}"
                    )
                unit = _unit_key(arm.arm_id, int(meta["seed"]))
                pair_dir = run_root / "work" / "replay" / unit
                result_path = run_root / "work" / "results" / f"{unit}.json"
                replay_job = {
                    "profile_id": settings.profile_id,
                    "arm_id": arm.arm_id,
                    "arm_order": int(arm_order),
                    "seed": int(meta["seed"]),
                    "seed_order": int(meta["seed_order"]),
                    "params_path": str(resolved_params[arm.param_source]),
                    "comparison_start": comparison_start,
                    "comparison_end": comparison_end,
                    "baseline_dir": baseline_dir,
                    "score_path": artifacts["score"],
                    "score_manifest_path": artifacts.get("manifest"),
                    "score_execution_start": artifacts["score_execution_start"],
                    "selected_epoch": artifacts["selected_epoch"],
                    "fold_count": artifacts.get("fold_count"),
                    "training_elapsed_sec": artifacts["training_elapsed_sec"],
                    "model_sha256": artifacts.get("model_sha256"),
                    "score_sha256": artifacts.get("score_sha256"),
                    "pair_dir": str(pair_dir),
                    "result_path": str(result_path),
                    "scientific_fingerprint": str(contract["fingerprint"]),
                    "robustness_id": cfg.robustness_id,
                    "keep_replay_details": cfg.keep_replay_details,
                    "keep_attribution_source": cfg.keep_attribution_source,
                    "attribution_dir": str(_attribution_unit_dir(run_root, arm.arm_id, int(meta["seed"]))),
                }
                ready_replays.append((
                    replay_job,
                    {
                        "arm_id": arm.arm_id,
                        "name": arm.name,
                        "seed": int(meta["seed"]),
                        "seed_order": int(meta["seed_order"]),
                        "arm_order": int(arm_order),
                    },
                ))
        if finished:
            write_progress_manifest()

    def submit_trainings() -> None:
        while pending_trainings and len(training_futures) < int(cfg.gpu_train_workers):
            meta = pending_trainings.popleft()
            meta["submitted_at"] = time.perf_counter()
            future = training_executor.submit(_train_one_unit, meta)
            training_futures[future] = meta
            progress_update(
                paint("[TRAIN]", "cyan", enabled=color_enabled, bold=True)
                + f" seed {meta['seed_order']}/{len(seeds)} | "
                f"模型來源 {meta['source_order']}/{meta['source_count']} {meta['dl_id']} "
                + f"| replay targets={len(meta['replay_arms'])} "
                + f"| GPU running={len(training_futures)}/{cfg.gpu_train_workers} "
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
        expected = len(stochastic_arms) * len(seeds)
        if len(seed_frame) != expected:
            raise RuntimeError(
                f"multi-seed robustness結果不完整: expected={expected}, actual={len(seed_frame)}"
            )
        seed_yearly_frame = _load_seed_yearly_results(seed_yearly_results_path)
        expected_yearly_units = {(arm.arm_id, int(seed)) for arm in stochastic_arms for seed in seeds}
        actual_yearly_units = set() if seed_yearly_frame.empty else {
            (str(row.arm_id), int(row.seed)) for row in seed_yearly_frame.itertuples(index=False)
        }
        missing_yearly = expected_yearly_units - actual_yearly_units
        if missing_yearly:
            raise RuntimeError(f"multi-seed年度結果不完整: missing_observations={len(missing_yearly)}")
        attribution_index_path = None
        if cfg.keep_attribution_source:
            ready_attribution = _attribution_ready_units(
                run_root,
                stochastic_arms=stochastic_arms,
                seeds=seeds,
                fingerprint=str(contract["fingerprint"]),
            )
            if len(ready_attribution) != expected:
                raise RuntimeError(
                    f"compact attribution source不完整: expected={expected}, actual={len(ready_attribution)}"
                )
            attribution_index_path = _write_attribution_index_manifest(
                run_root,
                contract=contract,
                stochastic_arms=stochastic_arms,
                seeds=seeds,
            )
        summary = _robustness_summary(
            contract=contract, fixed_results=fixed_results, seed_frame=seed_frame,
            seed_yearly_frame=seed_yearly_frame,
        )
        summary["elapsed_sec"] = round(time.perf_counter() - started_total, 3)
        _write_json(run_root / SUMMARY_FILENAME, summary)
        report_text = render_multi_seed_robustness_report(summary)
        (run_root / REPORT_FILENAME).write_text(report_text, encoding="utf-8")
        manifest.update({
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": summary["elapsed_sec"],
            "completed_seed_strategy_observations": int(len(seed_frame)),
            "attribution_source_units": (expected if cfg.keep_attribution_source else 0),
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
        for arm in stochastic_arms
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
    )
    print("永久工件：")
    permanent_paths = [manifest_path, seed_results_path, seed_yearly_results_path]
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
