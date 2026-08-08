"""Breakout-quality strategy parameter training service and legacy 4×2 gate.

The matrix is:
P0 original ROOS with formal rules, P1 original ROOS with all rule-based
filters disabled, P2 risk parameters trained with DL disabled, and P3 risk
parameters trained with DL enabled using binary point-in-time scores.  Every
parameter set is replayed once with the binary filter off (A) and once on (B).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    get_breakout_quality_workflow_settings,
)
from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT
from core.dataset_profiles import get_dataset_dir
from core.model_paths import resolve_models_dir
from core.runtime_utils import get_taipei_now
from filters.breakout_quality.artifacts import compute_file_sha256, load_model_artifact_contract
from filters.breakout_quality.binary_pit_score_store import (
    BINARY_PIT_IN_COVERAGE_MISSING_POLICY,
    BINARY_PIT_POST_COVERAGE_POLICY,
    BINARY_PIT_PRE_COVERAGE_POLICY,
    BINARY_PIT_SCORE_SOURCE,
    load_binary_point_in_time_score_table,
)
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.strategy_rule_policies import (
    ALL_OFF_INACTIVE_VALUE_OVERRIDES,
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)
from strategies.breakout.search_space import get_breakout_optimizer_required_min_rows
from tools.filters.breakout_quality.build_binary_point_in_time_scores import (
    build_binary_point_in_time_scores,
)
from filters.breakout_quality.strategy_optimizer_policy import (
    build_outer_rolling_argv,
    build_rolling_base_policy,
)
from filters.breakout_quality.strategy_compare_engine import (
    COMPARISON_MODE_HARD_FILTER,
    OPTIONAL_ENTRY_FILTER_FIELDS,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    PARAM_POLICIES,
    PARAM_POLICY_BASE_FINALIST_BEST,
    _load_param_source,
    _resolve_params_path,
    _validate_requested_param_policy,
    run_comparison,
)
from tools.optimizer.outer_rolling_oos import FOLD_FIXED_STRATEGY_OVERRIDES_KEY, run_outer_rolling_oos
from tools.optimizer.prep import load_all_raw_data
from tools.optimizer.runtime import create_optimizer_study
from tools.optimizer.session_factory import (
    build_optimizer_session,
    configure_optuna_logging,
    ensure_study_effective_policy_compatible,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 3
EXPERIMENT_STATUS = "FOUR_BY_TWO_COMPLETE"
RISK_SEARCH_FIELDS = ("atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail")
EXPERIMENT_RELATIVE_DIR = Path(
    "models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling"
)
BINARY_PIT_RELATIVE_DIR = Path(
    "models/research/breakout_quality/binary_point_in_time_scores"
)


def _parse_args(argv=None):
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description="執行4種參數 × Binary DL關／開的8操作點風險參數適應Gate"
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default=settings.strategy_dataset)
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument("--param-policy", choices=PARAM_POLICIES, default=PARAM_POLICY_BASE_FINALIST_BEST)
    parser.add_argument("--trials-per-fold", "--trials", dest="trials_per_fold", type=int, default=settings.strategy_adapt_trials_per_fold)
    parser.add_argument("--max-positions", type=int, default=settings.strategy_max_positions)
    parser.add_argument("--rotation", choices=("off", "on"), default=settings.strategy_rotation)
    parser.add_argument("--fixed-risk", type=float, default=settings.strategy_adapt_fixed_risk)
    parser.add_argument("--max-position-cap-pct", type=float, default=settings.strategy_adapt_max_position_cap_pct)
    parser.add_argument(
        "--build-binary-pit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="P3缺少Binary PIT時自動建立；可用--no-build-binary-pit只做前置檢查",
    )
    parser.add_argument(
        "--binary-pit-resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume-parameter-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否接續既有rolling optimizer studies",
    )
    parser.add_argument("--parameter-set", choices=("p2", "p3", "both"), default="both")
    parser.add_argument(
        "--p3-variant",
        default=None,
        help="P3工件子目錄名稱；留空維持既有canonical P3路徑",
    )
    parser.add_argument("--train-only", action="store_true", help="只建立指定參數工件，不執行4×2 replay")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_baseline_contract(*, root: Path, args) -> dict[str, Any]:
    params_path = _resolve_params_path(
        root=root, params_path=None, param_policy=args.param_policy, allow_static_diagnostic=False
    )
    if not params_path.is_file():
        raise FileNotFoundError(f"找不到Baseline rolling active params: {params_path}")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    source = _load_param_source(params_path)
    policy = _validate_requested_param_policy(source, args.param_policy)
    if str(source.get("kind") or "") != "rolling_active_param_ensemble":
        raise ValueError("4×2 Gate只接受rolling active-param ensemble Baseline")
    if int(policy.get("member_count_min") or 0) != 1 or int(policy.get("member_count_max") or 0) != 1:
        raise ValueError("risk-only fold freeze目前要求每個effective date恰有1個member")
    meta = dict(payload.get("meta") or {})
    required = ("window_mode", "first_oos_date", "last_oos_date", "train_window_months", "oos_horizon_months")
    missing = [key for key in required if meta.get(key) in (None, "")]
    if missing:
        raise ValueError("Baseline rolling params缺少meta欄位: " + ", ".join(missing))
    return {
        "path": params_path,
        "payload": payload,
        "source": source,
        "policy": policy,
        "meta": meta,
        "summary": dict(payload.get("summary") or {}),
        "sha256": compute_file_sha256(params_path),
    }


def _single_member_params_by_effective_date(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping = dict(contract["payload"].get("params_ensemble_by_effective_date") or {})
    output = {}
    for effective_date, raw_members in mapping.items():
        members = list(raw_members or [])
        if len(members) != 1:
            raise ValueError(f"effective_date={effective_date} member數必須為1")
        params = dict((members[0] or {}).get("params") or {})
        if not params:
            raise ValueError(f"active param member缺少params: {effective_date}")
        output[pd.Timestamp(effective_date).strftime("%Y-%m-%d")] = params
    if not output:
        raise ValueError("Baseline沒有params_ensemble_by_effective_date")
    return output


def build_risk_only_fold_overrides(*, baseline_contract, args, training_dl_enabled: bool):
    output = {}
    for effective_date, params in _single_member_params_by_effective_date(baseline_contract).items():
        fixed = dict(params)
        for field_name in RISK_SEARCH_FIELDS:
            fixed.pop(field_name, None)
        fixed.update({field_name: False for field_name in OPTIONAL_ENTRY_FILTER_FIELDS})
        fixed.update(ALL_RULE_FILTERS_OFF_OVERRIDES)
        fixed.update(ALL_OFF_INACTIVE_VALUE_OVERRIDES)
        fixed.update(
            {
                "use_breakout_quality_filter": bool(training_dl_enabled),
                "use_breakout_quality_ranking": False,
                "breakout_quality_filter_id": str(args.filter_id),
                "breakout_quality_score_threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
                "fixed_risk": float(args.fixed_risk),
                "max_position_cap_pct": float(args.max_position_cap_pct),
            }
        )
        output[effective_date] = fixed
    return output


def _binary_pit_paths(root: Path, args) -> tuple[Path, Path]:
    profile_dir = (
        root
        / BINARY_PIT_RELATIVE_DIR
        / str(args.filter_id)
        / str(args.model_architecture)
        / str(args.experiment_profile)
    )
    return profile_dir / "manifest.json", profile_dir / "scores.csv"


def _binary_pit_preflight(*, root: Path, args) -> dict[str, Any]:
    manifest_path, scores_path = _binary_pit_paths(root, args)
    status = "BINARY_PIT_REQUIRED"
    error = ""
    if manifest_path.is_file() and scores_path.is_file():
        try:
            _table, manifest = load_binary_point_in_time_score_table(str(manifest_path), str(scores_path))
            status = "READY"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            manifest = None
    else:
        manifest = None
    return {
        "status": status,
        "ready": status == "READY",
        "manifest_path": project_relative_display_path(manifest_path, project_root=root),
        "scores_path": project_relative_display_path(scores_path, project_root=root),
        "manifest_absolute": str(manifest_path.resolve()),
        "scores_absolute": str(scores_path.resolve()),
        "manifest_sha256": compute_file_sha256(manifest_path) if manifest_path.is_file() else None,
        "scores_sha256": compute_file_sha256(scores_path) if scores_path.is_file() else None,
        "score_period": None if manifest is None else dict(manifest.get("score_period") or {}),
        "error": error,
    }


def _ensure_binary_pit(*, root: Path, args) -> dict[str, Any]:
    preflight = _binary_pit_preflight(root=root, args=args)
    if preflight["ready"] or args.plan_only or not bool(args.build_binary_pit):
        return preflight
    build_args = [
        "--filter-id", str(args.filter_id),
        "--model-architecture", str(args.model_architecture),
        "--experiment-profile", str(args.experiment_profile),
        "--score-start-date", "auto",
        "--score-end-date", "auto",
        "--resume" if bool(args.binary_pit_resume) else "--no-resume",
    ]
    build_binary_point_in_time_scores(project_root=root, argv=build_args)
    preflight = _binary_pit_preflight(root=root, args=args)
    if not preflight["ready"]:
        raise RuntimeError(f"Binary PIT建立後仍不可用: {preflight['error']}")
    return preflight


def _validate_binary_pit_optimizer_coverage(
    *, binary_pit: dict[str, Any], baseline_contract: dict[str, Any]
) -> dict[str, Any]:
    """Validate Binary PIT against the rolling optimizer without demanding impossible history.

    The runtime SSOT already defines three date regions:
    - before PIT coverage: pass-through, which is exactly DL-off;
    - inside PIT coverage: use the PIT score and conservatively reject a missing candidate;
    - after PIT coverage: fail when a candidate appears.

    Therefore an expanding-window model does not need to score dates that precede its
    earliest legal training date.  It must, however, overlap the optimizer Selection
    history and remain current through the latest Selection end.
    """

    if not bool(binary_pit.get("ready")):
        return binary_pit
    meta = dict(baseline_contract.get("meta") or {})
    first_oos = pd.Timestamp(str(meta.get("first_oos_date"))).normalize()
    last_oos = pd.Timestamp(str(meta.get("last_oos_date"))).normalize()
    train_months = int(meta.get("train_window_months") or 0)
    horizon_months = int(meta.get("oos_horizon_months") or 0)
    if train_months < 1 or horizon_months < 1:
        raise ValueError(
            "Baseline缺少合法train_window_months／oos_horizon_months，"
            "無法驗證Binary PIT coverage"
        )
    if first_oos > last_oos:
        raise ValueError("Baseline first_oos_date不可晚於last_oos_date")

    required_start = (first_oos - pd.DateOffset(months=train_months)).normalize()
    required_end = (last_oos - pd.Timedelta(days=1)).normalize()
    period = dict(binary_pit.get("score_period") or {})
    try:
        actual_start = pd.Timestamp(str(period.get("start"))).normalize()
        actual_end = pd.Timestamp(str(period.get("end"))).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Binary PIT score_period不合法") from exc
    if pd.isna(actual_start) or pd.isna(actual_end) or actual_start > actual_end:
        raise ValueError("Binary PIT score_period不合法")
    if actual_end < required_end:
        raise ValueError(
            "Binary PIT尾端未覆蓋optimizer最新歷史Selection："
            f"required_through={required_end.date()}, actual_through={actual_end.date()}"
        )
    if actual_start > required_end:
        raise ValueError(
            "Binary PIT與optimizer歷史Selection完全沒有重疊："
            f"selection={required_start.date()}~{required_end.date()}, "
            f"pit={actual_start.date()}~{actual_end.date()}"
        )

    folds: list[dict[str, Any]] = []
    cursor = first_oos
    while cursor <= last_oos:
        selection_start = (cursor - pd.DateOffset(months=train_months)).normalize()
        selection_end = (cursor - pd.Timedelta(days=1)).normalize()
        total_days = int((selection_end - selection_start).days + 1)
        overlap_start = max(selection_start, actual_start)
        overlap_end = min(selection_end, actual_end)
        scored_days = (
            int((overlap_end - overlap_start).days + 1)
            if overlap_start <= overlap_end
            else 0
        )
        if scored_days == 0:
            coverage_mode = "bootstrap_fallback_only"
            overlap_period = {"start": None, "end": None}
        elif scored_days == total_days:
            coverage_mode = "full_score_history"
            overlap_period = {
                "start": str(overlap_start.date()),
                "end": str(overlap_end.date()),
            }
        else:
            coverage_mode = "partial_score_history"
            overlap_period = {
                "start": str(overlap_start.date()),
                "end": str(overlap_end.date()),
            }
        folds.append(
            {
                "oos_start": str(cursor.date()),
                "selection_start": str(selection_start.date()),
                "selection_end": str(selection_end.date()),
                "pit_overlap": overlap_period,
                "calendar_days": total_days,
                "pit_calendar_days": scored_days,
                "pre_pit_fallback_days": int(total_days - scored_days),
                "calendar_coverage_ratio": float(scored_days / total_days),
                "coverage_mode": coverage_mode,
            }
        )
        cursor = (cursor + pd.DateOffset(months=horizon_months)).normalize()

    total_days = int(sum(int(row["calendar_days"]) for row in folds))
    scored_days = int(sum(int(row["pit_calendar_days"]) for row in folds))
    mode_counts = {
        mode: int(sum(row["coverage_mode"] == mode for row in folds))
        for mode in (
            "bootstrap_fallback_only",
            "partial_score_history",
            "full_score_history",
        )
    }
    coverage = {
        "contract_version": 2,
        "full_history_required": False,
        "optimizer_selection_period": {
            "start": str(required_start.date()),
            "end": str(required_end.date()),
        },
        "pit_score_period": {
            "start": str(actual_start.date()),
            "end": str(actual_end.date()),
        },
        "pre_coverage_policy": BINARY_PIT_PRE_COVERAGE_POLICY,
        "in_coverage_missing_policy": BINARY_PIT_IN_COVERAGE_MISSING_POLICY,
        "post_coverage_policy": BINARY_PIT_POST_COVERAGE_POLICY,
        "fold_count": int(len(folds)),
        "coverage_mode_counts": mode_counts,
        "weighted_calendar_coverage_ratio": float(scored_days / total_days),
        "folds": folds,
    }
    binary_pit = dict(binary_pit)
    # Keep the historical key for artifact readers, but its meaning is now an
    # audited Selection period rather than a 100% coverage requirement.
    binary_pit["optimizer_required_period"] = dict(
        coverage["optimizer_selection_period"]
    )
    binary_pit["optimizer_coverage"] = coverage
    return binary_pit


def _runtime_contract(*, root, args, baseline_contract, fold_overrides, model_artifact, arm_id, training_dl_enabled, binary_pit):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "binary_dl_filter_risk_only_param_adaptation",
        "arm_id": str(arm_id),
        "training_dl_enabled": bool(training_dl_enabled),
        "dataset": str(args.dataset),
        "dataset_identity": build_source_data_inventory(root, args.dataset),
        "baseline_params_path": project_relative_display_path(baseline_contract["path"], project_root=root),
        "baseline_params_sha256": baseline_contract["sha256"],
        "rolling_policy": {key: baseline_contract["meta"][key] for key in ("window_mode", "first_oos_date", "last_oos_date", "train_window_months", "oos_horizon_months")},
        "trials_per_fold": int(args.trials_per_fold),
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "fixed_overrides_by_effective_date": fold_overrides,
        "binary_runtime": (
            None
            if not training_dl_enabled
            else {
                "filter_id": str(args.filter_id),
                "model_architecture": str(args.model_architecture),
                "experiment_profile": str(args.experiment_profile),
                "manifest_sha256": compute_file_sha256(model_artifact.paths.manifest_path),
            }
        ),
        "binary_pit": (
            None
            if not training_dl_enabled
            else {
                "manifest_sha256": binary_pit["manifest_sha256"],
                "scores_sha256": binary_pit["scores_sha256"],
                "score_period": binary_pit["score_period"],
                "optimizer_coverage": dict(binary_pit.get("optimizer_coverage") or {}),
            }
        ),
        "fixed_risk": float(args.fixed_risk),
        "max_position_cap_pct": float(args.max_position_cap_pct),
        "max_positions": int(args.max_positions),
        "rotation": str(args.rotation),
    }
    payload["runtime_identity_sha256"] = _canonical_hash(payload)
    return payload


def _validate_risk_only_params(*, path, baseline_contract, fold_overrides, args, training_dl_enabled, arm_id):
    payload = _load_json(path)
    if payload is None:
        raise FileNotFoundError(f"{arm_id} optimizer未產生參數工件: {path}")
    meta = dict(payload.get("meta") or {})
    for key in ("first_oos_date", "last_oos_date", "train_window_months", "oos_horizon_months"):
        if str(meta.get(key)) != str(baseline_contract["meta"].get(key)):
            raise ValueError(f"{arm_id}參數fold schedule不一致: {key}")
    if int(meta.get("trials_per_fold", 0) or 0) != int(args.trials_per_fold):
        raise ValueError(f"{arm_id}參數trials／fold與目前要求不一致")
    members_by_date = dict(payload.get("params_ensemble_by_effective_date") or {})
    for effective_date, expected_fixed in fold_overrides.items():
        members = list(members_by_date.get(effective_date) or [])
        if not members:
            raise ValueError(f"{arm_id}參數缺少effective date: {effective_date}")
        for member in members:
            params = dict((member or {}).get("params") or {})
            for field_name, expected in expected_fixed.items():
                actual = params.get(field_name)
                matched = (
                    actual is not None and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                    if isinstance(expected, float)
                    else actual == expected
                )
                if not matched:
                    raise ValueError(
                        f"{arm_id}固定契約未落入active params: date={effective_date}, field={field_name}, actual={actual!r}, expected={expected!r}"
                    )
            for field_name in RISK_SEARCH_FIELDS:
                if field_name not in params:
                    raise ValueError(f"{arm_id}缺少risk field: {effective_date}/{field_name}")
    payload["breakout_quality_param_adaptation"] = {
        "mode": "risk_only_training",
        "parameter_set": str(arm_id),
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "fixed_rule_contract": "all_rule_filters_off",
        "training_dl_enabled": bool(training_dl_enabled),
    }
    _write_json(path, payload)
    return payload


def _copy_legacy_p2_if_available(*, output_dir: Path, target_dir: Path) -> bool:
    legacy = output_dir / "a5_dl_off_trained"
    legacy_params = legacy / "active_params" / "roos_base_best.json"
    target_params = target_dir / "active_params" / "roos_base_best.json"
    if target_params.is_file() or not legacy_params.is_file():
        return False
    target_params.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_params, target_params)
    return True


def _temporary_environment(values: dict[str, str]):
    class _Env:
        def __enter__(self):
            self.before = {key: os.environ.get(key) for key in values}
            os.environ.update(values)
            return self
        def __exit__(self, exc_type, exc, tb):
            for key, prior in self.before.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior
            return False
    return _Env()


def _run_optimizer_arm(*, root, args, settings, baseline_contract, model_artifact, output_dir, arm_id, training_dl_enabled, binary_pit):
    arm_dir = output_dir / ("p3_dl_on_trained" if training_dl_enabled else "p2_dl_off_trained")
    if training_dl_enabled and args.p3_variant not in (None, ""):
        arm_dir = arm_dir / str(args.p3_variant)
    active_param_dir = arm_dir / "active_params"
    optimizer_output_dir = arm_dir / "optimizer_runtime"
    active_param_dir.mkdir(parents=True, exist_ok=True)
    migrated = False
    if not training_dl_enabled:
        migrated = _copy_legacy_p2_if_available(output_dir=output_dir, target_dir=arm_dir)
    fold_overrides = build_risk_only_fold_overrides(
        baseline_contract=baseline_contract, args=args, training_dl_enabled=training_dl_enabled
    )
    contract = _runtime_contract(
        root=root,
        args=args,
        baseline_contract=baseline_contract,
        fold_overrides=fold_overrides,
        model_artifact=model_artifact,
        arm_id=arm_id,
        training_dl_enabled=training_dl_enabled,
        binary_pit=binary_pit,
    )
    preflight_path = arm_dir / "rolling_preflight.json"
    params_path = active_param_dir / "roos_base_best.json"
    prior = _load_json(preflight_path)
    reusable = (
        prior is not None
        and str(prior.get("runtime_identity_sha256") or "") == contract["runtime_identity_sha256"]
        and params_path.is_file()
    )
    if migrated:
        reusable = True
    coverage_path = arm_dir / "binary_pit_optimizer_coverage.csv"
    if training_dl_enabled:
        coverage_rows = []
        for raw_row in list(
            dict(binary_pit.get("optimizer_coverage") or {}).get("folds") or []
        ):
            row = dict(raw_row)
            overlap = dict(row.pop("pit_overlap", {}) or {})
            row["pit_overlap_start"] = overlap.get("start")
            row["pit_overlap_end"] = overlap.get("end")
            coverage_rows.append(row)
        pd.DataFrame(coverage_rows).to_csv(
            coverage_path, index=False, encoding="utf-8-sig"
        )
    _write_json(
        preflight_path,
        {
            **contract,
            "status": f"{arm_id}_PREFLIGHT_PASS",
            "legacy_p2_migrated": migrated,
            "binary_pit_optimizer_coverage_path": (
                project_relative_display_path(coverage_path, project_root=root)
                if training_dl_enabled
                else None
            ),
            "created_at": get_taipei_now().isoformat(),
        },
    )
    if not reusable:
        base_policy = build_rolling_base_policy(root=root, baseline_contract=baseline_contract)
        base_policy.update({"adaptation_scope": f"binary_dl_filter_risk_only_{arm_id.lower()}", "evaluation_scope": "rolling_selection_diagnostic"})
        session_spec = {
            "output_dir": str((optimizer_output_dir / "sessions").resolve()),
            FOLD_FIXED_STRATEGY_OVERRIDES_KEY: fold_overrides,
            "runtime_cache_identity": contract["runtime_identity_sha256"],
            "optimizer_fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
            "train_max_positions": int(args.max_positions),
            "train_enable_rotation": str(args.rotation) == "on",
        }
        env_values = {}
        if training_dl_enabled:
            session_spec["runtime_context_spec"] = {
                "module": "filters.breakout_quality.runtime",
                "callable": "breakout_quality_filter_source_context",
                "kwargs": {
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": binary_pit["manifest_absolute"],
                    "scores_path": binary_pit["scores_absolute"],
                },
            }
            env_values = {
                "BREAKOUT_QUALITY_FILTER_SCORE_SOURCE": BINARY_PIT_SCORE_SOURCE,
                "BREAKOUT_QUALITY_BINARY_PIT_MANIFEST": binary_pit["manifest_absolute"],
                "BREAKOUT_QUALITY_BINARY_PIT_SCORES": binary_pit["scores_absolute"],
            }
        outer_environ = dict(os.environ)
        outer_environ.update(env_values)
        outer_environ["V16_MODELS_DIR"] = resolve_models_dir(str(root))
        outer_environ["OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE"] = "sqlite"
        outer_environ["OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES"] = (
            "1" if bool(args.resume_parameter_training) else "0"
        )
        with _temporary_environment(env_values):
            exit_code = run_outer_rolling_oos(
                argv=build_outer_rolling_argv(args=args, baseline_contract=baseline_contract),
                environ=outer_environ,
                project_root=str(root),
                output_dir=str(optimizer_output_dir),
                base_policy=base_policy,
                selected_data_dir=get_dataset_dir(str(root), args.dataset),
                dataset_label=str(args.dataset),
                load_all_raw_data=load_all_raw_data,
                optimizer_required_min_rows=get_breakout_optimizer_required_min_rows(),
                build_optimizer_session=build_optimizer_session,
                create_optimizer_study=create_optimizer_study,
                ensure_study_effective_policy_compatible=ensure_study_effective_policy_compatible,
                configure_optuna_logging=configure_optuna_logging,
                optimizer_seed=int(settings.seed),
                optimizer_session_spec=session_spec,
                default_trials=int(args.trials_per_fold),
                timing_mode=False,
                paramset_models_dir=str(active_param_dir.resolve()),
            )
        if int(exit_code) != 0:
            raise RuntimeError(f"{arm_id} risk-only rolling optimizer失敗: {exit_code}")
    params_payload = _validate_risk_only_params(
        path=params_path,
        baseline_contract=baseline_contract,
        fold_overrides=fold_overrides,
        args=args,
        training_dl_enabled=training_dl_enabled,
        arm_id=arm_id,
    )
    summary = {
        "parameter_set": arm_id,
        "training_dl_enabled": bool(training_dl_enabled),
        "status": "COMPLETED",
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "folds": int(dict(params_payload.get("summary") or {}).get("folds", 0) or 0),
        "trials_per_fold": int(args.trials_per_fold),
        "optimizer_search_reused": bool(reusable),
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "binary_pit_optimizer_coverage": (
            None
            if not training_dl_enabled
            else dict(binary_pit.get("optimizer_coverage") or {})
        ),
    }
    _write_json(arm_dir / "rolling_optimizer_summary.json", summary)
    return {"params_path": params_path, "summary": summary, "contract": contract}


def _run_pair(
    *, root, args, params_path, policy_name, output_dir,
    binary_pit, comparison_start_date, comparison_end_date,
):
    all_off = policy_name == "all_off"
    return run_comparison(
        project_root=root,
        dataset=args.dataset,
        params_path=str(params_path),
        param_policy=args.param_policy,
        max_positions=int(args.max_positions),
        enable_rotation=str(args.rotation) == "on",
        fixed_risk=float(args.fixed_risk),
        max_position_cap_pct=float(args.max_position_cap_pct),
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
        optional_entry_filter_policy=(OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT),
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile),
        output_dir_override=output_dir,
        comparison_start_date=str(comparison_start_date),
        comparison_end_date=str(comparison_end_date),
        quiet=bool(args.quiet),
        shared_param_overrides=(ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None),
        hard_filter_source={
            "score_source": BINARY_PIT_SCORE_SOURCE,
            "manifest_path": str(binary_pit["manifest_absolute"]),
            "scores_path": str(binary_pit["scores_absolute"]),
        },
    )


def _arm_metric(pair: dict, dl_enabled: bool, key: str):
    return dict(pair.get("quality_filter" if dl_enabled else "no_filter") or {}).get(key)


def _delta_value(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _fmt(value, unit="", digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{unit}"


def _operation_rows(matrix):
    rows = []
    for index, info in enumerate(matrix):
        pair = info["result"]
        for prefix, dl in (("A", False), ("B", True)):
            rows.append(
                (
                    f"{prefix}{index}",
                    info["parameter_label"],
                    info["rule_label"],
                    "開" if dl else "關",
                    _fmt(_arm_metric(pair, dl, "total_return_pct"), "%"),
                    _fmt(_arm_metric(pair, dl, "max_drawdown_pct"), "%"),
                    _fmt(_arm_metric(pair, dl, "return_over_max_drawdown")),
                    _fmt(_arm_metric(pair, dl, "expected_value_r"), " R"),
                    _fmt(_arm_metric(pair, dl, "avg_exposure_pct"), "%"),
                    str(int(_arm_metric(pair, dl, "trade_count") or 0)),
                )
            )
    return rows


def _dl_increment_rows(matrix):
    rows = []
    for index, info in enumerate(matrix):
        pair = info["result"]
        rows.append(
            (
                f"B{index}−A{index}",
                info["parameter_label"],
                _fmt(_delta_value(_arm_metric(pair, True, "total_return_pct"), _arm_metric(pair, False, "total_return_pct")), "pp"),
                _fmt(_delta_value(_arm_metric(pair, True, "max_drawdown_pct"), _arm_metric(pair, False, "max_drawdown_pct")), "pp"),
                _fmt(_delta_value(_arm_metric(pair, True, "return_over_max_drawdown"), _arm_metric(pair, False, "return_over_max_drawdown"))),
                _fmt(_delta_value(_arm_metric(pair, True, "expected_value_r"), _arm_metric(pair, False, "expected_value_r")), " R"),
                _fmt(_delta_value(_arm_metric(pair, True, "avg_exposure_pct"), _arm_metric(pair, False, "avg_exposure_pct")), "pp"),
            )
        )
    return rows


def _comparison_row(matrix, label, left_index, left_dl, right_index, right_dl):
    left = matrix[left_index]["result"]
    right = matrix[right_index]["result"]
    return (
        label,
        _fmt(_delta_value(_arm_metric(left, left_dl, "total_return_pct"), _arm_metric(right, right_dl, "total_return_pct")), "pp"),
        _fmt(_delta_value(_arm_metric(left, left_dl, "max_drawdown_pct"), _arm_metric(right, right_dl, "max_drawdown_pct")), "pp"),
        _fmt(_delta_value(_arm_metric(left, left_dl, "return_over_max_drawdown"), _arm_metric(right, right_dl, "return_over_max_drawdown"))),
        _fmt(_delta_value(_arm_metric(left, left_dl, "expected_value_r"), _arm_metric(right, right_dl, "expected_value_r")), " R"),
        _fmt(_delta_value(_arm_metric(left, left_dl, "avg_exposure_pct"), _arm_metric(right, right_dl, "avg_exposure_pct")), "pp"),
    )


def _adaptation_rows(matrix):
    rows = [
        _comparison_row(matrix, "A1−A0：原ROOS關閉rules", 1, False, 0, False),
        _comparison_row(matrix, "A2−A1：DL-off-trained對No-DL", 2, False, 1, False),
        _comparison_row(matrix, "B2−B1：DL-off-trained對DL", 2, True, 1, True),
        _comparison_row(matrix, "A3−A1：DL-on-trained拿掉DL", 3, False, 1, False),
        _comparison_row(matrix, "B3−B1：DL-on-trained對DL", 3, True, 1, True),
        _comparison_row(matrix, "B3−A2：最終公平比較", 3, True, 2, False),
    ]
    dl2 = _delta_value(_arm_metric(matrix[2]["result"], True, "total_return_pct"), _arm_metric(matrix[2]["result"], False, "total_return_pct"))
    dl3 = _delta_value(_arm_metric(matrix[3]["result"], True, "total_return_pct"), _arm_metric(matrix[3]["result"], False, "total_return_pct"))
    rows.append(("Interaction：(B3−A3)−(B2−A2)", _fmt(_delta_value(dl3, dl2), "pp"), "-", "-", "-", "-"))
    return rows


def _risk_param_rows(*, baseline_contract, p2_path, p3_path):
    baseline = _single_member_params_by_effective_date(baseline_contract)
    def _load(path):
        payload = _load_json(path) or {}
        out = {}
        for date_text, members in dict(payload.get("params_ensemble_by_effective_date") or {}).items():
            members = list(members or [])
            if members:
                out[str(date_text)] = dict((members[0] or {}).get("params") or {})
        return out
    p2 = _load(p2_path)
    p3 = _load(p3_path)
    rows = []
    dates = sorted(baseline)
    for field in RISK_SEARCH_FIELDS:
        rows.append((field, ", ".join(str(baseline[d].get(field)) for d in dates), ", ".join(str(p2.get(d, {}).get(field)) for d in dates), ", ".join(str(p3.get(d, {}).get(field)) for d in dates)))
    return rows



def _binary_pit_coverage_rows(binary_pit: dict[str, Any]) -> list[tuple[Any, ...]]:
    coverage = dict(binary_pit.get("optimizer_coverage") or {})
    rows = []
    for row in list(coverage.get("folds") or []):
        overlap = dict(row.get("pit_overlap") or {})
        overlap_text = (
            "-"
            if overlap.get("start") in (None, "")
            else f"{overlap.get('start')}～{overlap.get('end')}"
        )
        rows.append(
            (
                str(row.get("oos_start") or ""),
                f"{row.get('selection_start')}～{row.get('selection_end')}",
                overlap_text,
                f"{float(row.get('calendar_coverage_ratio', 0.0)):.1%}",
                str(row.get("coverage_mode") or ""),
            )
        )
    return rows


def _render_binary_pit_coverage(
    binary_pit: dict[str, Any], *, section_title: str = "Binary PIT optimizer coverage"
) -> str:
    coverage = dict(binary_pit.get("optimizer_coverage") or {})
    if not coverage:
        return render_section(section_title) + "\n尚無coverage契約。"
    counts = dict(coverage.get("coverage_mode_counts") or {})
    selection_period = dict(coverage.get("optimizer_selection_period") or {})
    pit_period = dict(coverage.get("pit_score_period") or {})
    selection_text = (
        "-"
        if not selection_period
        else f"{selection_period.get('start')}～{selection_period.get('end')}"
    )
    pit_text = (
        "-" if not pit_period else f"{pit_period.get('start')}～{pit_period.get('end')}"
    )
    return "\n".join(
        (
            render_section(section_title),
            render_key_values(
                (
                    ("Optimizer Selection", selection_text),
                    ("PIT Score period", pit_text),
                    ("Weighted coverage", f"{float(coverage.get('weighted_calendar_coverage_ratio', 0.0)):.1%}"),
                    ("Bootstrap folds", int(counts.get("bootstrap_fallback_only", 0))),
                    ("Partial folds", int(counts.get("partial_score_history", 0))),
                    ("Full folds", int(counts.get("full_score_history", 0))),
                    ("PIT開始日前", "DL-off pass-through"),
                    ("PIT期間內缺分", "conservative REJECT"),
                    ("PIT尾端過期", "candidate出現即fail-fast"),
                )
            ),
            render_table(
                ("OOS起點", "Selection", "PIT重疊", "Coverage", "模式"),
                _binary_pit_coverage_rows(binary_pit),
            ),
        )
    )

def _render_report(*, args, matrix, p2_arm, p3_arm, binary_pit, baseline_contract, plan_only):
    lines = [
        render_title("Binary DL Filter 4 Parameters × 2 DL States Gate"),
        render_key_values(
            (
                ("參數基準", args.param_policy),
                ("搜尋參數", " / ".join(RISK_SEARCH_FIELDS)),
                ("P2訓練", "rules全關／DL關"),
                ("P3訓練", "rules全關／DL開／Binary PIT"),
                ("Binary PIT", binary_pit["status"]),
                (
                    "DL replay source",
                    (
                        "未執行（train-only參數建立）"
                        if args.train_only
                        else "binary_point_in_time（八操作點一致；process workers已傳遞）"
                    ),
                ),
                ("固定 Buy sort", "原 position-aware buy-sort"),
            )
        ),
        render_section("1. 八個操作點"),
        render_table(
            ("組別", "參數", "規則", "DL", "報酬", "MDD", "RoMD", "EV", "曝險", "交易"),
            [] if matrix is None else _operation_rows(matrix),
        ),
    ]
    if matrix is not None:
        lines.extend(
            (
                render_section("2. 各參數下DL增量"),
                render_table(("比較", "參數", "Δ報酬", "ΔMDD", "ΔRoMD", "ΔEV", "Δ曝險"), _dl_increment_rows(matrix)),
                render_section("3. 規則與參數適應"),
                render_table(("比較", "Δ報酬", "ΔMDD", "ΔRoMD", "ΔEV", "Δ曝險"), _adaptation_rows(matrix)),
                "主判定：B3−A2；Interaction只判斷DL-aware參數是否改善DL增量，不能取代絕對績效。",
            )
        )
    if p2_arm is not None and p3_arm is not None:
        lines.extend(
            (
                render_section("4. 風險參數"),
                render_table(("參數", "P0/P1 原ROOS", "P2 DL-off-trained", "P3 DL-on-trained"), _risk_param_rows(baseline_contract=baseline_contract, p2_path=p2_arm["params_path"], p3_path=p3_arm["params_path"])),
            )
        )
    lines.extend(
        (
            _render_binary_pit_coverage(
                binary_pit, section_title="5. Binary PIT training coverage"
            ),
            render_section("6. 無前視契約"),
            render_key_values(
                (
                    ("PIT Manifest", binary_pit["manifest_path"]),
                    ("PIT Scores", binary_pit["scores_path"]),
                    ("PIT period", binary_pit.get("score_period") or "-"),
                    ("禁止", "final forward-OOS／research／Selection in-sample scores回灌optimizer"),
                )
            ),
        )
    )
    if plan_only:
        lines.append("目前為PLAN ONLY；不訓練P2/P3，也不執行八操作點replay。")
    return "\n\n".join(lines)


def run_param_adaptation_gate(*, project_root=PROJECT_ROOT, argv=None):
    args = _parse_args(argv)
    root = Path(project_root).resolve()
    settings = get_breakout_quality_workflow_settings()
    if args.param_policy != PARAM_POLICY_BASE_FINALIST_BEST:
        raise ValueError("4×2 Gate固定使用base-finalist-best")
    if int(args.trials_per_fold) < 1 or int(args.max_positions) < 1:
        raise ValueError("trials-per-fold與max-positions必須>=1")
    if args.p3_variant not in (None, ""):
        variant = Path(str(args.p3_variant))
        if variant.is_absolute() or len(variant.parts) != 1 or variant.parts[0] in {".", ".."}:
            raise ValueError("p3-variant只允許單一安全資料夾名稱")
    requires_binary_pit = str(args.parameter_set) in {"p3", "both"}
    model_artifact = (
        load_model_artifact_contract(
            str(root),
            str(args.filter_id),
            str(args.model_architecture),
            str(args.experiment_profile),
        )
        if requires_binary_pit
        else None
    )
    baseline_contract = _load_baseline_contract(root=root, args=args)
    if requires_binary_pit:
        binary_pit = _ensure_binary_pit(root=root, args=args)
        if binary_pit["ready"]:
            binary_pit = _validate_binary_pit_optimizer_coverage(
                binary_pit=binary_pit, baseline_contract=baseline_contract
            )
        if not args.plan_only and not binary_pit["ready"]:
            raise FileNotFoundError(
                "P3需要Binary PIT scores；請保留預設--build-binary-pit，或先執行build-binary-point-in-time-scores"
            )
    else:
        binary_pit = _binary_pit_preflight(root=root, args=args)
        binary_pit = {**binary_pit, "status": "NOT_REQUIRED_FOR_P2"}
    output_dir = root / EXPERIMENT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir
    if str(args.parameter_set) == "p3" and args.p3_variant not in (None, ""):
        metadata_dir = output_dir / "p3_dl_on_trained" / str(args.p3_variant)
        metadata_dir.mkdir(parents=True, exist_ok=True)
    plan_path = metadata_dir / "strategy_dl_filter_param_adapt_plan.json"
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLAN_ONLY" if args.plan_only else "TRAINING" if args.train_only else "RUNNING",
        "parameter_set": str(args.parameter_set),
        "p3_variant": None if args.p3_variant in (None, "") else str(args.p3_variant),
        "train_only": bool(args.train_only),
        "matrix": [
            {"index": 0, "parameter_set": "P0", "params": "original_roos", "rules": "formal"},
            {"index": 1, "parameter_set": "P1", "params": "original_roos", "rules": "all_off"},
            {"index": 2, "parameter_set": "P2", "params": "dl_off_trained", "rules": "all_off"},
            {"index": 3, "parameter_set": "P3", "params": "dl_on_trained", "rules": "all_off"},
        ],
        "binary_pit": binary_pit,
        "created_at": get_taipei_now().isoformat(),
    }
    _write_json(plan_path, plan)
    if binary_pit.get("ready"):
        print("\n" + _render_binary_pit_coverage(binary_pit))
    p2_arm = p3_arm = matrix = None
    if not args.plan_only:
        configure_optuna_logging()
        if args.parameter_set in {"p2", "both"}:
            p2_arm = _run_optimizer_arm(
                root=root, args=args, settings=settings, baseline_contract=baseline_contract,
                model_artifact=model_artifact, output_dir=output_dir, arm_id="P2",
                training_dl_enabled=False, binary_pit=binary_pit,
            )
        if args.parameter_set in {"p3", "both"}:
            p3_arm = _run_optimizer_arm(
                root=root, args=args, settings=settings, baseline_contract=baseline_contract,
                model_artifact=model_artifact, output_dir=output_dir, arm_id="P3",
                training_dl_enabled=True, binary_pit=binary_pit,
            )
        if args.train_only:
            comparison_start_date = None
            comparison_end_date = None
        else:
            if p2_arm is None or p3_arm is None:
                raise ValueError("4×2 replay需要parameter-set=both；單獨建立P2/P3時請使用--train-only")
            comparison_start_date = str(baseline_contract["meta"]["first_oos_date"])
            comparison_end_date = str((binary_pit.get("score_period") or {}).get("end") or "")
            if not comparison_end_date:
                raise ValueError("Binary PIT缺少score period end，無法建立4×2 replay")
            pairs = [
                _run_pair(
                    root=root, args=args, params_path=baseline_contract["path"],
                    policy_name="formal", output_dir=output_dir / "replay" / "p0_original_roos_formal",
                    binary_pit=binary_pit, comparison_start_date=comparison_start_date,
                    comparison_end_date=comparison_end_date,
                ),
                _run_pair(
                    root=root, args=args, params_path=baseline_contract["path"],
                    policy_name="all_off", output_dir=output_dir / "replay" / "p1_original_roos_all_off",
                    binary_pit=binary_pit, comparison_start_date=comparison_start_date,
                    comparison_end_date=comparison_end_date,
                ),
                _run_pair(
                    root=root, args=args, params_path=p2_arm["params_path"],
                    policy_name="all_off", output_dir=output_dir / "replay" / "p2_dl_off_trained_all_off",
                    binary_pit=binary_pit, comparison_start_date=comparison_start_date,
                    comparison_end_date=comparison_end_date,
                ),
                _run_pair(
                    root=root, args=args, params_path=p3_arm["params_path"],
                    policy_name="all_off", output_dir=output_dir / "replay" / "p3_dl_on_trained_all_off",
                    binary_pit=binary_pit, comparison_start_date=comparison_start_date,
                    comparison_end_date=comparison_end_date,
                ),
            ]
            labels = [
                ("P0 原ROOS", "原正式設定"),
                ("P1 原ROOS", "Rule-based filters全關"),
                ("P2 DL-off-trained", "Rule-based filters全關"),
                ("P3 DL-on-trained", "Rule-based filters全關"),
            ]
            matrix = [
                {"parameter_label": labels[i][0], "rule_label": labels[i][1], "result": pairs[i]}
                for i in range(4)
            ]
    report = _render_report(args=args, matrix=matrix, p2_arm=p2_arm, p3_arm=p3_arm, binary_pit=binary_pit, baseline_contract=baseline_contract, plan_only=bool(args.plan_only))
    print("\n" + report)
    markdown_path = metadata_dir / "strategy_dl_filter_param_adapt_gate.md"
    json_path = metadata_dir / "strategy_dl_filter_param_adapt_gate.json"
    markdown_path.write_text(report + "\n", encoding="utf-8")
    result = {
        **plan,
        "status": (
            "PLAN_ONLY" if args.plan_only else "PARAMETER_TRAINING_COMPLETE"
            if args.train_only else EXPERIMENT_STATUS
        ),
        "p2_optimizer": None if p2_arm is None else {"params_path": project_relative_display_path(p2_arm["params_path"], project_root=root), "summary": p2_arm["summary"]},
        "p3_optimizer": None if p3_arm is None else {"params_path": project_relative_display_path(p3_arm["params_path"], project_root=root), "summary": p3_arm["summary"]},
        "matrix": matrix,
    }
    _write_json(json_path, result)
    print_artifact_paths((("4×2計畫", plan_path), ("4×2報表", markdown_path), ("4×2 JSON", json_path)), project_root=root)
    return result


def prepare_strategy_parameter_source(
    *,
    project_root=PROJECT_ROOT,
    dataset: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    param_policy: str,
    parameter_set: str,
    trials_per_fold: int,
    max_positions: int,
    rotation: str,
    fixed_risk: float,
    max_position_cap_pct: float,
    p3_variant: str | None = None,
    build_binary_pit: bool = True,
    binary_pit_resume: bool = True,
    resume_parameter_training: bool = True,
    quiet: bool = False,
):
    """Build one configured risk-only rolling parameter source without replay."""

    argv = [
        "--dataset", str(dataset),
        "--filter-id", str(filter_id),
        "--model-architecture", str(model_architecture),
        "--experiment-profile", str(experiment_profile),
        "--param-policy", str(param_policy),
        "--parameter-set", str(parameter_set),
        "--trials-per-fold", str(int(trials_per_fold)),
        "--max-positions", str(int(max_positions)),
        "--rotation", str(rotation),
        "--fixed-risk", str(float(fixed_risk)),
        "--max-position-cap-pct", str(float(max_position_cap_pct)),
        "--build-binary-pit" if build_binary_pit else "--no-build-binary-pit",
        "--binary-pit-resume" if binary_pit_resume else "--no-binary-pit-resume",
        "--resume-parameter-training" if resume_parameter_training else "--no-resume-parameter-training",
        "--train-only",
    ]
    if p3_variant not in (None, ""):
        argv.extend(("--p3-variant", str(p3_variant)))
    if quiet:
        argv.append("--quiet")
    return run_param_adaptation_gate(project_root=project_root, argv=argv)


def main(argv=None):
    run_param_adaptation_gate(argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
