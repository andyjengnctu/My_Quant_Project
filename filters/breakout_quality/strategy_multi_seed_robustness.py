"""Strategy-level multiple-seed robustness for configured continuous DL arms.

The user-facing entry lives under Strategy Compare, while model fitting is still
performed by the canonical continuous-ranker trainer in an isolated subprocess.
Only aggregate seed metrics are persistent by default; per-seed model/score/replay
artifacts are temporary work products and are removed after their metrics are stored.
"""

from __future__ import annotations

import argparse
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
    get_breakout_quality_experiment_profile,
)
from config.strategy_compare import (
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_settings,
)
from core.console_report import project_relative_display_path, render_table, render_title
from core.strategy_comparison import StrategyComparisonArm
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    DAILY_RANKER_OOS_SCORE_FILENAME,
    CONTINUOUS_RANKER_SCORE_FILENAME,
)
from filters.breakout_quality.strategy_compare_engine import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    run_comparison,
    run_standalone_baseline,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.strategy_comparison import (
    _arm_runtime_spec,
    _find_reusable_baseline_source,
    collect_artifact_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILENAME = "robustness_report.md"
SUMMARY_FILENAME = "robustness_summary.json"
SEED_RESULTS_FILENAME = "seed_results.csv"
MANIFEST_FILENAME = "manifest.json"
LATEST_FILENAME = "latest.json"
ROBUSTNESS_SCHEMA_VERSION = 2

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


def _canonical_hash(payload: Any, *, length: int = 16) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


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


def _robustness_arms(settings) -> tuple[tuple[StrategyComparisonArm, ...], tuple[StrategyComparisonArm, ...]]:
    fixed = tuple(
        arm for arm in settings.enabled_arms if arm.robustness_role == "fixed_baseline"
    )
    stochastic = tuple(
        arm for arm in settings.enabled_arms if arm.robustness_role == "stochastic"
    )
    if not fixed or not stochastic:
        raise ValueError("multi-seed robustness需要fixed baseline與stochastic arms")
    return fixed, stochastic


def _dataset_identity_snapshot(dataset: str) -> dict[str, Any]:
    try:
        return build_source_data_inventory(PROJECT_ROOT, dataset)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return {
            "status": "SOURCE_DATA_UNAVAILABLE",
            "dataset": str(dataset),
            "detail": str(exc),
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
    comparison_period: dict[str, Any] | None = None,
    artifact_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robustness = get_strategy_multi_seed_robustness_settings()
    settings = get_strategy_comparison_settings(robustness.profile_id)
    fixed, stochastic = _robustness_arms(settings)
    seeds = resolve_multi_seed_values(
        seed_count=robustness.seed_count,
        generator_seed=robustness.seed_generator_seed,
    )
    resolved_period = dict(comparison_period or {})
    if not resolved_period:
        resolved_period = {"start": settings.start_date, "end": settings.end_date}
    identities = dict(artifact_identities or {})
    required_param_sources = sorted(
        {arm.param_source for arm in (*fixed, *stochastic)}
    )
    parameter_identities = {}
    for source_id in required_param_sources:
        raw_identity = identities.get(f"param:{source_id}")
        if not isinstance(raw_identity, dict):
            continue
        parameter_identities[source_id] = {
            key: raw_identity.get(key)
            for key in (
                "sha256", "identity_status", "coverage_start",
                "coverage_end", "coverage_status",
            )
        }
    training_defaults = _continuous_training_defaults_snapshot()
    reference_baselines: dict[str, dict[str, Any]] = {}
    for reference_key, spec in robustness.romd_reference_baselines.items():
        matches = [
            arm for arm in fixed
            if arm.param_source == spec["param_source"]
            and arm.rule_policy == spec["rule_policy"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "multi-seed RoMD reference無法唯一解析fixed baseline: "
                f"reference={reference_key}, matches={len(matches)}"
            )
        matched = matches[0]
        reference_baselines[str(reference_key)] = {
            "arm_id": matched.arm_id,
            "name": matched.name,
            "param_source": matched.param_source,
            "rule_policy": matched.rule_policy,
        }
    contract = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "profile_id": settings.profile_id,
        "strategy_schema_version": int(settings.schema_version),
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
        "training_defaults": training_defaults,
        "romd_reference_baselines": reference_baselines,
        "fixed_arms": [
            {
                "arm_id": arm.arm_id,
                "param_source": arm.param_source,
                "rule_policy": arm.rule_policy,
            }
            for arm in fixed
        ],
        "stochastic_arms": [
            {
                "arm_id": arm.arm_id,
                "param_source": arm.param_source,
                "rule_policy": arm.rule_policy,
                "dl_id": arm.dl_id,
                "dl_runtime_mode": arm.dl_runtime_mode,
                "dl_runtime_options": dict(arm.dl_runtime_options or {}),
                "dl_source": settings.dl_sources[str(arm.dl_id)].as_dict(),
                "experiment_profile": get_breakout_quality_experiment_profile(
                    settings.dl_sources[str(arm.dl_id)].experiment_profile
                ).as_manifest_payload(),
            }
            for arm in stochastic
        ],
        "parallelism": {
            "gpu_train_workers": int(robustness.gpu_train_workers),
            "cpu_replay_workers": int(robustness.cpu_replay_workers),
            "design": "single_gpu_training_queue_overlapped_with_cpu_replay_queue",
        },
        "retention": {
            "keep_checkpoints": bool(robustness.keep_checkpoints),
            "keep_scores": bool(robustness.keep_scores),
            "keep_replay_details": bool(robustness.keep_replay_details),
        },
    }
    fingerprint_payload = {
        key: value
        for key, value in contract.items()
        if key not in {"parallelism", "retention"}
    }
    contract["fingerprint"] = _canonical_hash(fingerprint_payload, length=16)
    return contract


def _run_root(contract: dict[str, Any]) -> Path:
    cfg = get_strategy_multi_seed_robustness_settings()
    return (PROJECT_ROOT / cfg.output_root / str(contract["fingerprint"])).resolve()


def _model_work_root(contract: dict[str, Any]) -> Path:
    cfg = get_strategy_multi_seed_robustness_settings()
    return (PROJECT_ROOT / cfg.model_work_root / str(contract["fingerprint"])).resolve()


def _unit_key(arm_id: str, seed: int) -> str:
    return f"{arm_id}__seed_{int(seed)}"


def _load_seed_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return frame
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


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
    *, arm: StrategyComparisonArm, seed: int, model_dir: Path, research_dir: Path, settings
) -> dict[str, Any]:
    dl = settings.dl_sources[str(arm.dl_id)]
    profile = get_breakout_quality_experiment_profile(dl.experiment_profile)
    score_name = (
        DAILY_RANKER_OOS_SCORE_FILENAME
        if str(profile.training_sample_scope) == "daily_eligible_stock_days"
        else CONTINUOUS_RANKER_SCORE_FILENAME
    )
    paths = {
        "model": model_dir / "model.pt",
        "manifest": model_dir / "manifest.json",
        "report": research_dir / CONTINUOUS_RANKER_REPORT_FILENAME,
        "score": research_dir / score_name,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"multi-seed {label}工件不存在: {path}")
    manifest = _read_json(paths["manifest"])
    report = _read_json(paths["report"])
    for payload, label in ((manifest, "manifest"), (report, "report")):
        if str(payload.get("experiment_profile") or "") != dl.experiment_profile:
            raise ValueError(f"multi-seed {label} experiment profile不一致")
        if str(payload.get("model_architecture") or "") != dl.model_architecture:
            raise ValueError(f"multi-seed {label} architecture不一致")
    training = dict(report.get("training") or {})
    if int(training.get("seed", -1)) != int(seed):
        raise ValueError(
            f"multi-seed report seed不一致: expected={seed}, actual={training.get('seed')}"
        )
    return {
        **{key: str(path.resolve()) for key, path in paths.items()},
        "model_sha256": compute_file_sha256(paths["model"]),
        "score_sha256": compute_file_sha256(paths["score"]),
        "selected_epoch": int(training.get("selected_epoch", 0) or 0),
        "training_elapsed_sec": float(report.get("elapsed_sec", 0.0) or 0.0),
    }


def _training_command(
    *, arm: StrategyComparisonArm, seed: int, model_dir: Path, research_dir: Path, settings
) -> list[str]:
    dl = settings.dl_sources[str(arm.dl_id)]
    return [
        sys.executable,
        "-m",
        "tools.filters.breakout_quality.train_continuous_ranker",
        "--filter-id",
        dl.filter_id,
        "--model-architecture",
        dl.model_architecture,
        "--experiment-profile",
        dl.experiment_profile,
        "--seed",
        str(int(seed)),
        "--model-output-dir",
        str(model_dir.resolve()),
        "--research-output-dir",
        str(research_dir.resolve()),
    ]


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


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
                "baseline_dir": str(reusable_dir.resolve()),
                "source": "REUSE",
            }
            print(
                f"[BASELINE REUSE] {arm.name} | "
                f"{project_relative_display_path(reusable_dir, project_root=PROJECT_ROOT)}"
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
            "baseline_dir": str(output_dir.resolve()),
            "source": "RUN",
        }
    return results


def _replay_one_unit(job: dict[str, Any]) -> dict[str, Any]:
    settings = get_strategy_comparison_settings(str(job["profile_id"]))
    arm = settings.arms[str(job["arm_id"])]
    dl = settings.dl_sources[str(arm.dl_id)]
    start = str(job["comparison_start"])
    end = str(job["comparison_end"])
    pair_dir = Path(str(job["pair_dir"])).resolve()
    baseline_dir = Path(str(job["baseline_dir"])).resolve()
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    all_off = arm.rule_policy == "all_off"
    started = time.perf_counter()
    runtime_spec = _arm_runtime_spec(arm)
    payload = run_comparison(
        project_root=PROJECT_ROOT,
        dataset=settings.dataset,
        params_path=str(job["params_path"]),
        param_policy=settings.param_policy,
        max_positions=settings.max_positions,
        enable_rotation=settings.rotation == "on",
        comparison_mode=runtime_spec["comparison_mode"],
        ranking_policy=runtime_spec["ranking_policy"],
        ranking_options=dict(arm.dl_runtime_options or {}),
        optional_entry_filter_policy=(
            OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            if all_off
            else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
        ),
        filter_id=dl.filter_id,
        score_source=dl.score_source,
        model_architecture=dl.model_architecture,
        experiment_profile=dl.experiment_profile,
        threshold=dl.threshold,
        output_dir_override=pair_dir,
        comparison_start_date=start,
        comparison_end_date=end,
        quiet=True,
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
        baseline_reuse_dir=baseline_dir,
        continuous_score_path_override=str(job["score_path"]),
    )
    metrics = dict(payload.get("score_ranking") or {})
    attribution_path = pair_dir / "trade_attribution.json"
    direct_selection_r = None
    if attribution_path.is_file():
        attribution = _read_json(attribution_path)
        direct_selection_r = float(
            dict(attribution.get("r_attribution") or {}).get(
                "exclusive_selection_delta_r", 0.0
            )
        )
    metrics["direct_selection_r"] = direct_selection_r
    result = {
        "arm_id": arm.arm_id,
        "name": arm.name,
        "seed": int(job["seed"]),
        "arm_order": int(job["arm_order"]),
        "seed_order": int(job["seed_order"]),
        "selected_epoch": int(job.get("selected_epoch", 0) or 0),
        "training_elapsed_sec": float(job.get("training_elapsed_sec", 0.0) or 0.0),
        "replay_elapsed_sec": round(time.perf_counter() - started, 3),
        **{key: metrics.get(key) for _label, key, _unit in MEAN_METRICS},
    }
    result_path = Path(str(job["result_path"])).resolve()
    _write_json(result_path, result)
    if not bool(job.get("keep_replay_details")):
        shutil.rmtree(pair_dir, ignore_errors=True)
    return result


def _result_row(result: dict[str, Any]) -> dict[str, Any]:
    return dict(result)


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _robustness_summary(
    *, contract: dict[str, Any], fixed_results: dict[str, dict[str, Any]], seed_frame: pd.DataFrame
) -> dict[str, Any]:
    settings = get_strategy_comparison_settings(str(contract["profile_id"]))
    fixed, stochastic = _robustness_arms(settings)
    mean_rows: list[dict[str, Any]] = []
    for arm in fixed:
        metrics = dict(fixed_results[arm.arm_id]["metrics"])
        metrics.setdefault("direct_selection_r", 0.0)
        mean_rows.append(
            {
                "arm_id": arm.arm_id,
                "name": arm.name,
                "type": "Fixed",
                "n": 1,
                **{key: metrics.get(key) for _label, key, _unit in MEAN_METRICS},
            }
        )
    for arm in stochastic:
        rows = seed_frame[seed_frame["arm_id"] == arm.arm_id].copy()
        mean_rows.append(
            {
                "arm_id": arm.arm_id,
                "name": arm.name,
                "type": "Multi-seed",
                "n": int(len(rows)),
                **{key: _mean_or_none(rows[key]) for _label, key, _unit in MEAN_METRICS},
            }
        )

    fixed_by_arm_id = {
        row["arm_id"]: row for row in mean_rows if row["type"] == "Fixed"
    }
    references = dict(contract.get("romd_reference_baselines") or {})
    min_reference = dict(references.get("min") or {})
    full_reference = dict(references.get("full") or {})
    min_baseline = fixed_by_arm_id.get(str(min_reference.get("arm_id") or ""))
    full_baseline = fixed_by_arm_id.get(str(full_reference.get("arm_id") or ""))
    if min_baseline is None or full_baseline is None:
        raise ValueError("multi-seed summary缺少config指定的Min/Full fixed baseline")
    romd_rows: list[dict[str, Any]] = []
    for row in mean_rows:
        if row["type"] == "Fixed":
            value = float(row["return_over_max_drawdown"])
            romd_rows.append(
                {
                    "arm_id": row["arm_id"], "name": row["name"], "n": 1,
                    "mean": value, "median": value, "std": None, "cv": None,
                    "min": value, "p25": None, "p75": None, "max": value,
                    "beats_min_count": None, "beats_full_count": None,
                }
            )
            continue
        values = pd.to_numeric(
            seed_frame.loc[seed_frame["arm_id"] == row["arm_id"], "return_over_max_drawdown"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        romd_rows.append(
            {
                "arm_id": row["arm_id"], "name": row["name"], "n": int(len(values)),
                "mean": mean,
                "median": float(np.median(values)),
                "std": std,
                "cv": (None if math.isclose(mean, 0.0, abs_tol=1e-12) else float(std / abs(mean))),
                "min": float(np.min(values)),
                "p25": float(np.quantile(values, 0.25)),
                "p75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
                "beats_min_count": (
                    None if min_baseline is None else int(np.sum(values > float(min_baseline["return_over_max_drawdown"])))
                ),
                "beats_full_count": (
                    None if full_baseline is None else int(np.sum(values > float(full_baseline["return_over_max_drawdown"])))
                ),
            }
        )
    stochastic_rows = [row for row in romd_rows if int(row["n"]) > 1]
    distribution_compare = None
    if len(stochastic_rows) == 2:
        a, b = stochastic_rows
        av = pd.to_numeric(
            seed_frame.loc[seed_frame["arm_id"] == a["arm_id"], "return_over_max_drawdown"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        bv = pd.to_numeric(
            seed_frame.loc[seed_frame["arm_id"] == b["arm_id"], "return_over_max_drawdown"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if len(av) and len(bv):
            distribution_compare = {
                "left": a["name"],
                "right": b["name"],
                "pairwise_left_gt_right_probability": float(np.mean(av[:, None] > bv[None, :])),
                "pair_count": int(len(av) * len(bv)),
            }
    return {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "status": "RESULT_AVAILABLE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "mean_strategy_metrics": mean_rows,
        "romd_statistics": romd_rows,
        "romd_distribution_comparison": distribution_compare,
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
        for label, key, unit in MEAN_METRICS:
            digits = 1 if key == "trade_count" else 4 if key == "log_r_squared" else 2
            cells.append(_fmt(row.get(key), digits=digits, suffix=unit))
        mean_rows.append(cells)

    romd_headers = [
        "比較對象", "N", "Mean", "Median", "Std", "CV", "Min", "P25", "P75", "Max", "勝Min", "勝Full"
    ]
    romd_rows = []
    for row in summary["romd_statistics"]:
        n = int(row["n"])
        romd_rows.append([
            row["name"], str(n), _fmt(row["mean"]), _fmt(row["median"]), _fmt(row["std"]),
            _fmt(row["cv"]), _fmt(row["min"]), _fmt(row["p25"]), _fmt(row["p75"]), _fmt(row["max"]),
            "-" if row["beats_min_count"] is None else f"{row['beats_min_count']}/{n}",
            "-" if row["beats_full_count"] is None else f"{row['beats_full_count']}/{n}",
        ])

    lines = [
        "# Multiple-seed Robustness",
        "",
        f"- Profile：`{summary['contract']['profile_id']}`",
        f"- Seeds：`{summary['contract']['seed_count']}`（deterministic generated；resolved values只存manifest，不作best-seed選擇）",
        f"- Fingerprint：`{summary['contract']['fingerprint']}`",
        "- 判讀原則：不挑最佳seed、不做seed ensemble；主表比較各策略指標的seed平均，第二表只看RoMD完整分布。",
        "",
        "## 1. 平均策略績效",
        "",
        _markdown_table(mean_headers, mean_rows),
        "",
        "## 2. RoMD完整統計",
        "",
        _markdown_table(romd_headers, romd_rows),
    ]
    compare = summary.get("romd_distribution_comparison")
    if isinstance(compare, dict):
        lines.extend([
            "",
            "### RoMD分布交叉比較",
            "",
            f"- P({compare['left']} > {compare['right']}) = {float(compare['pairwise_left_gt_right_probability'])*100:.2f}% "
            f"（{compare['pair_count']}組cross-seed pairs）",
        ])
    references = dict(summary["contract"].get("romd_reference_baselines") or {})
    min_name = str(dict(references.get("min") or {}).get("name") or "Min baseline")
    full_name = str(dict(references.get("full") or {}).get("name") or "Full baseline")
    lines.extend([
        "",
        "## 3. 限制",
        "",
        "- resolved seeds只用於重現；報表不指定seed 42或任何best seed。",
        f"- {full_name}／{min_name}沒有DL訓練seed，因此以固定正式baseline值放入同一平均績效表。",
        "- Forward-OOS已屬iterative research OOS evidence；不得用本報表挑training hyperparameter或best seed。",
        "",
    ])
    return "\n".join(lines)


def _print_report_tables(summary: dict[str, Any]) -> None:
    print("\n" + render_title("Multiple-seed Robustness"))
    headers = ["比較對象", "類型", "N"] + [
        f"{label} Mean" for label, _key, _unit in MEAN_METRICS
    ]
    rows = []
    for row in summary["mean_strategy_metrics"]:
        cells = [row["name"], row["type"], row["n"]]
        for _label, key, unit in MEAN_METRICS:
            digits = 1 if key == "trade_count" else 4 if key == "log_r_squared" else 2
            cells.append(_fmt(row.get(key), digits=digits, suffix=unit))
        rows.append(cells)
    print(render_table(headers, rows))
    print("\nRoMD完整統計")
    headers = ["比較對象", "N", "Mean", "Median", "Std", "CV", "Min", "P25", "P75", "Max", "勝Min", "勝Full"]
    rows = []
    for row in summary["romd_statistics"]:
        n = int(row["n"])
        rows.append([
            row["name"], n, _fmt(row["mean"]), _fmt(row["median"]), _fmt(row["std"]), _fmt(row["cv"]),
            _fmt(row["min"]), _fmt(row["p25"]), _fmt(row["p75"]), _fmt(row["max"]),
            "-" if row["beats_min_count"] is None else f"{row['beats_min_count']}/{n}",
            "-" if row["beats_full_count"] is None else f"{row['beats_full_count']}/{n}",
        ])
    print(render_table(headers, rows))


def show_multi_seed_robustness_status() -> None:
    cfg = get_strategy_multi_seed_robustness_settings()
    settings = get_strategy_comparison_settings(cfg.profile_id)
    status = collect_artifact_status(settings=settings)
    period = dict(status.get("comparison_period") or {})
    contract = build_multi_seed_robustness_contract(
        comparison_period=period,
        artifact_identities=dict(status.get("artifact_identities") or {}),
    )
    fixed, stochastic = _robustness_arms(settings)
    run_root = _run_root(contract)
    existing = _load_seed_results(run_root / SEED_RESULTS_FILENAME)
    print("\n" + render_title("Multiple-seed robustness 設定"))
    print(f"比較階段          ：{settings.profile_label} ({settings.profile_id})")
    print(f"Seed數量          ：{cfg.seed_count}")
    print(f"Seed generator    ：deterministic / generator_seed={cfg.seed_generator_seed}")
    print(f"GPU training      ：workers={cfg.gpu_train_workers}")
    print(f"CPU strategy replay：workers={cfg.cpu_replay_workers}")
    print(f"Fingerprint       ：{contract['fingerprint']}")
    print(f"已完成seed結果    ：{len(existing)}/{len(stochastic) * cfg.seed_count}")
    print("比較對象：")
    for arm in fixed:
        print(f"- {arm.name:<16} fixed baseline")
    for arm in stochastic:
        print(f"- {arm.name:<16} {cfg.seed_count} random seeds")
    print("永久輸出：")
    print(f"- {project_relative_display_path(run_root / MANIFEST_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / SEED_RESULTS_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / SUMMARY_FILENAME, project_root=PROJECT_ROOT)}")
    print(f"- {project_relative_display_path(run_root / REPORT_FILENAME, project_root=PROJECT_ROOT)}")


def show_latest_multi_seed_robustness_report() -> None:
    cfg = get_strategy_multi_seed_robustness_settings()
    latest = (PROJECT_ROOT / cfg.output_root / LATEST_FILENAME).resolve()
    if not latest.is_file():
        raise FileNotFoundError("尚無Multiple-seed robustness最新結果")
    pointer = _read_json(latest)
    report = PROJECT_ROOT / str(pointer["report_path"])
    if not report.is_file():
        raise FileNotFoundError(f"最新robustness報表不存在: {report}")
    print(report.read_text(encoding="utf-8"))


def _cleanup_unit_artifacts(*, model_dir: Path, research_dir: Path, keep_checkpoints: bool, keep_scores: bool) -> None:
    if not keep_checkpoints:
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
) -> dict[str, Any]:
    payload = dict(manifest)
    payload["status"] = "RUNNING"
    payload["completed_units"] = int(len(completed))
    payload["total_units"] = int(total_units)
    payload["completed_unit_keys"] = sorted(
        _unit_key(arm_id, seed) for arm_id, seed in completed
    )
    payload["current_training"] = current_training
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


def run_multi_seed_robustness(*, confirm: bool = True) -> dict[str, Any]:
    cfg = get_strategy_multi_seed_robustness_settings()
    if not cfg.enabled:
        raise RuntimeError("Multiple-seed robustness目前由config關閉")
    settings = get_strategy_comparison_settings(cfg.profile_id)
    fixed_arms, stochastic_arms = _robustness_arms(settings)
    status = collect_artifact_status(settings=settings)
    resolved_params = dict(status.get("resolved_parameter_paths") or {})
    required_param_sources = {arm.param_source for arm in (*fixed_arms, *stochastic_arms)}
    missing_params = sorted(source for source in required_param_sources if not resolved_params.get(source))
    if missing_params:
        raise RuntimeError(
            "Multiple-seed robustness缺少策略參數工件，請先由Strategy Compare正式前置建立: "
            + ", ".join(missing_params)
        )
    comparison_start, comparison_end = _resolved_period(settings, status)
    contract = build_multi_seed_robustness_contract(
        comparison_period={"start": comparison_start, "end": comparison_end},
        artifact_identities=dict(status.get("artifact_identities") or {}),
    )
    run_root = _run_root(contract)
    model_root = _model_work_root(contract)
    run_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / MANIFEST_FILENAME
    seed_results_path = run_root / SEED_RESULTS_FILENAME
    manifest = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "status": "PLANNED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "comparison_period": {"start": comparison_start, "end": comparison_end},
        "seed_results_path": project_relative_display_path(seed_results_path, project_root=PROJECT_ROOT),
    }
    _write_json(manifest_path, manifest)

    if confirm:
        show_multi_seed_robustness_status()
        try:
            raw = input("👉 按 Enter 執行；輸入 0 返回：").strip().lower()
        except EOFError:
            return {}
        if raw in {"0", "q", "quit", "exit"}:
            return {}
        if raw not in {"", "1"}:
            print("輸入無效，本次不執行。")
            return {}

    started_total = time.perf_counter()
    fixed_results = _run_fixed_baselines(
        settings=settings, status=status, run_root=run_root, fixed_arms=fixed_arms
    )
    baseline_by_group = {
        (arm.param_source, arm.rule_policy): fixed_results[arm.arm_id]["baseline_dir"]
        for arm in fixed_arms
    }
    existing = _load_seed_results(seed_results_path)
    completed = {
        (str(row.arm_id), int(row.seed))
        for row in existing.itertuples(index=False)
    } if not existing.empty and cfg.reuse_completed else set()
    rows = existing.to_dict("records") if not existing.empty and cfg.reuse_completed else []
    seeds = tuple(int(value) for value in contract["resolved_seeds"])
    total_units = len(stochastic_arms) * len(seeds)
    done_units = len(completed)
    print("\n" + render_title("Multiple-seed robustness 執行"))
    print(f"Seeds={len(seeds)} | stochastic arms={len(stochastic_arms)} | work units={total_units}")
    print(f"GPU train workers={cfg.gpu_train_workers} | CPU replay workers={cfg.cpu_replay_workers}")

    current_training: dict[str, Any] | None = None
    manifest = _manifest_progress_payload(
        manifest=manifest,
        completed=completed,
        total_units=total_units,
        current_training=current_training,
    )
    _write_json(manifest_path, manifest)

    executor = ThreadPoolExecutor(max_workers=int(cfg.cpu_replay_workers))
    futures: dict[Future, dict[str, Any]] = {}

    def harvest(*, wait_all: bool = False) -> None:
        nonlocal done_units, rows, manifest
        while True:
            finished = [future for future in futures if future.done()]
            if not finished and not wait_all:
                return
            if not finished and wait_all and futures:
                time.sleep(0.25)
                continue
            for future in finished:
                meta = futures.pop(future)
                result = future.result()
                rows = [
                    row for row in rows
                    if not (str(row.get("arm_id")) == str(result["arm_id"]) and int(row.get("seed")) == int(result["seed"]))
                ]
                rows.append(_result_row(result))
                frame = pd.DataFrame(rows)
                _write_seed_results(seed_results_path, frame)
                completed.add((str(result["arm_id"]), int(result["seed"])))
                done_units = len(completed)
                manifest = _manifest_progress_payload(
                    manifest=manifest,
                    completed=completed,
                    total_units=total_units,
                    current_training=current_training,
                    futures=futures,
                )
                _write_json(manifest_path, manifest)
                print(
                    f"[DONE {done_units}/{total_units}] seed {meta['seed_order']}/{len(seeds)} | "
                    f"對象 {meta['arm_order']}/{len(stochastic_arms)} {meta['name']} | "
                    f"RoMD={_fmt(result.get('return_over_max_drawdown'))} | "
                    f"train={_format_elapsed(result.get('training_elapsed_sec', 0))} | "
                    f"replay={_format_elapsed(result.get('replay_elapsed_sec', 0))} | "
                    f"total={_format_elapsed(time.perf_counter()-started_total)}"
                )
                _cleanup_unit_artifacts(
                    model_dir=Path(meta["model_dir"]), research_dir=Path(meta["research_dir"]),
                    keep_checkpoints=cfg.keep_checkpoints, keep_scores=cfg.keep_scores,
                )
            if not wait_all or not futures:
                return

    try:
        for seed_order, seed in enumerate(seeds, start=1):
            for arm_order, arm in enumerate(stochastic_arms, start=1):
                if (arm.arm_id, seed) in completed:
                    print(
                        f"[REUSE {done_units}/{total_units}] seed {seed_order}/{len(seeds)} | "
                        f"對象 {arm_order}/{len(stochastic_arms)} {arm.name}"
                    )
                    continue
                harvest()
                unit = _unit_key(arm.arm_id, seed)
                model_dir = model_root / unit / "model"
                research_dir = run_root / "work" / "training" / unit
                pair_dir = run_root / "work" / "replay" / unit
                result_path = run_root / "work" / "results" / f"{unit}.json"
                train_started = time.perf_counter()
                artifacts = None
                if cfg.reuse_completed and model_dir.is_dir() and research_dir.is_dir():
                    try:
                        artifacts = _validate_training_artifacts(
                            arm=arm, seed=seed, model_dir=model_dir,
                            research_dir=research_dir, settings=settings,
                        )
                    except (FileNotFoundError, ValueError):
                        artifacts = None
                if artifacts is not None:
                    print(
                        f"[TRAIN REUSE] seed {seed_order}/{len(seeds)} | "
                        f"對象 {arm_order}/{len(stochastic_arms)} {arm.name} | "
                        f"epoch={artifacts['selected_epoch']}"
                    )
                else:
                    shutil.rmtree(model_dir, ignore_errors=True)
                    shutil.rmtree(research_dir, ignore_errors=True)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    research_dir.mkdir(parents=True, exist_ok=True)
                    current_training = {
                        "arm_id": arm.arm_id,
                        "name": arm.name,
                        "seed": int(seed),
                        "seed_order": int(seed_order),
                        "arm_order": int(arm_order),
                        "state": "TRAINING",
                    }
                    manifest = _manifest_progress_payload(
                        manifest=manifest,
                        completed=completed,
                        total_units=total_units,
                        current_training=current_training,
                        futures=futures,
                    )
                    _write_json(manifest_path, manifest)
                    print(
                        f"[TRAIN] seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} "
                        f"{arm.name} | completed={done_units}/{total_units} | total={_format_elapsed(time.perf_counter()-started_total)}"
                    )
                    env = dict(os.environ)
                    env["BREAKOUT_QUALITY_COMPACT_CONSOLE"] = "1"
                    train_log_path = research_dir / "train.log"
                    with train_log_path.open("w", encoding="utf-8") as train_log:
                        proc = subprocess.Popen(
                            _training_command(
                                arm=arm, seed=seed, model_dir=model_dir, research_dir=research_dir, settings=settings
                            ),
                            cwd=str(PROJECT_ROOT),
                            env=env,
                            stdout=train_log,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        next_print = time.perf_counter() + 30.0
                        while proc.poll() is None:
                            harvest()
                            now = time.perf_counter()
                            if now >= next_print:
                                replaying = len(futures)
                                print(
                                    f"  seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name} "
                                    f"training={_format_elapsed(now-train_started)} | CPU replay running={replaying} | "
                                    f"total={_format_elapsed(now-started_total)}"
                                )
                                next_print = now + 30.0
                            time.sleep(0.5)
                    if int(proc.returncode or 0) != 0:
                        train_tail = ""
                        if train_log_path.is_file():
                            train_tail = train_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                        raise RuntimeError(
                            f"multi-seed模型訓練失敗: arm={arm.name}, seed_index={seed_order}; "
                            f"returncode={proc.returncode}; log_tail={train_tail}"
                        )
                    artifacts = _validate_training_artifacts(
                        arm=arm, seed=seed, model_dir=model_dir, research_dir=research_dir, settings=settings
                    )
                    print(
                        f"[TRAIN DONE] seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} {arm.name} | "
                        f"epoch={artifacts['selected_epoch']} | elapsed={_format_elapsed(time.perf_counter()-train_started)}"
                    )
                current_training = None
                baseline_dir = baseline_by_group.get((arm.param_source, arm.rule_policy))
                if not baseline_dir:
                    raise RuntimeError(
                        f"stochastic arm找不到同參數fixed baseline: {arm.param_source}/{arm.rule_policy}"
                    )
                job = {
                    "profile_id": settings.profile_id,
                    "arm_id": arm.arm_id,
                    "arm_order": arm_order,
                    "seed": seed,
                    "seed_order": seed_order,
                    "params_path": str(resolved_params[arm.param_source]),
                    "comparison_start": comparison_start,
                    "comparison_end": comparison_end,
                    "baseline_dir": baseline_dir,
                    "score_path": artifacts["score"],
                    "selected_epoch": artifacts["selected_epoch"],
                    "training_elapsed_sec": artifacts["training_elapsed_sec"],
                    "pair_dir": str(pair_dir),
                    "result_path": str(result_path),
                    "keep_replay_details": cfg.keep_replay_details,
                }
                # CPU replay queue的容量限制放在「提交下一個replay之前」，而不是
                # 前一個replay剛queue完就等待。如此workers=1時，Replay A可與
                # 下一個seed/model的GPU Train B真正重疊；只有Train B完成、準備
                # 提交Replay B時，才需要等待Replay A釋放CPU replay slot。
                while len(futures) >= int(cfg.cpu_replay_workers):
                    harvest()
                    if len(futures) >= int(cfg.cpu_replay_workers):
                        time.sleep(0.25)
                future = executor.submit(_replay_one_unit, job)
                futures[future] = {
                    "arm_id": arm.arm_id,
                    "name": arm.name,
                    "seed": seed,
                    "seed_order": seed_order,
                    "arm_order": arm_order,
                    "model_dir": str(model_dir),
                    "research_dir": str(research_dir),
                }
                manifest = _manifest_progress_payload(
                    manifest=manifest,
                    completed=completed,
                    total_units=total_units,
                    current_training=current_training,
                    futures=futures,
                )
                _write_json(manifest_path, manifest)
                print(
                    f"[REPLAY QUEUED] seed {seed_order}/{len(seeds)} | 對象 {arm_order}/{len(stochastic_arms)} "
                    f"{arm.name} | CPU running={len(futures)}/{cfg.cpu_replay_workers}"
                )
        harvest(wait_all=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    seed_frame = _load_seed_results(seed_results_path)
    expected = len(stochastic_arms) * len(seeds)
    if len(seed_frame) != expected:
        raise RuntimeError(
            f"multi-seed robustness結果不完整: expected={expected}, actual={len(seed_frame)}"
        )
    summary = _robustness_summary(
        contract=contract, fixed_results=fixed_results, seed_frame=seed_frame
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
        "summary_path": project_relative_display_path(run_root / SUMMARY_FILENAME, project_root=PROJECT_ROOT),
        "report_path": project_relative_display_path(run_root / REPORT_FILENAME, project_root=PROJECT_ROOT),
    })
    _write_json(manifest_path, manifest)
    latest_path = (PROJECT_ROOT / cfg.output_root / LATEST_FILENAME).resolve()
    _write_json(latest_path, {
        "fingerprint": contract["fingerprint"],
        "report_path": project_relative_display_path(run_root / REPORT_FILENAME, project_root=PROJECT_ROOT),
        "summary_path": project_relative_display_path(run_root / SUMMARY_FILENAME, project_root=PROJECT_ROOT),
    })
    if not cfg.keep_replay_details:
        shutil.rmtree(run_root / "work" / "replay", ignore_errors=True)
    shutil.rmtree(run_root / "work" / "results", ignore_errors=True)
    shutil.rmtree(run_root / "work" / "baselines", ignore_errors=True)
    if not cfg.keep_scores:
        shutil.rmtree(run_root / "work" / "training", ignore_errors=True)
    if not cfg.keep_checkpoints:
        shutil.rmtree(model_root, ignore_errors=True)
    if not cfg.keep_replay_details and not cfg.keep_scores:
        shutil.rmtree(run_root / "work", ignore_errors=True)
    _print_report_tables(summary)
    print(f"\n總耗時：{_format_elapsed(summary['elapsed_sec'])}")
    print("永久工件：")
    for path in (manifest_path, seed_results_path, run_root / SUMMARY_FILENAME, run_root / REPORT_FILENAME):
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
