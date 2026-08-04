"""Risk-only rolling parameter adaptation gate for the binary DL hard filter.

Stage 1 (A5/B5) is executable with the current canonical runtime artifacts:
non-risk strategy values are frozen per rolling effective date, only ATR risk/
execution parameters are searched, and the selected params are replayed with the
binary DL filter off/on.

Stage 2 (A6/B6) requires point-in-time binary scores inside every optimizer
training window.  The current final-model forward-OOS score table is not a legal
substitute, so this module reports the missing prerequisite instead of leaking
Selection/OOS information into optimizer training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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
from filters.breakout_quality.artifacts import (
    compute_file_sha256,
    load_runtime_artifact_contract,
)
from filters.breakout_quality.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS
from strategies.breakout.search_space import get_breakout_optimizer_required_min_rows
from tools.filters.breakout_quality.strategy_adapt import (
    _build_base_policy,
    _outer_rolling_argv,
)
from tools.filters.breakout_quality.strategy_compare import (
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
from tools.optimizer.outer_rolling_oos import (
    FOLD_FIXED_STRATEGY_OVERRIDES_KEY,
    run_outer_rolling_oos,
)
from tools.optimizer.prep import load_all_raw_data
from tools.optimizer.runtime import create_optimizer_study
from tools.optimizer.session_factory import (
    build_optimizer_session,
    configure_optuna_logging,
    ensure_study_effective_policy_compatible,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
EXPERIMENT_STATUS = "A5_B5_COMPLETED_A6_B6_BINARY_PIT_REQUIRED"
RISK_SEARCH_FIELDS = (
    "atr_len",
    "atr_buy_tol",
    "atr_times_init",
    "atr_times_trail",
)
ALL_RULE_FILTERS_OFF_OVERRIDES = {
    "use_history_threshold": False,
    "use_breakout_reclaim_reentry": False,
    "use_kc": False,
}
ALL_OFF_INACTIVE_VALUE_OVERRIDES = {
    "breakout_ema_len": BREAKOUT_PARAM_SPECS["breakout_ema_len"]["default"],
    "bb_len": BREAKOUT_PARAM_SPECS["bb_len"]["default"],
    "bb_mult": BREAKOUT_PARAM_SPECS["bb_mult"]["default"],
    "kc_len": BREAKOUT_PARAM_SPECS["kc_len"]["default"],
    "kc_mult": BREAKOUT_PARAM_SPECS["kc_mult"]["default"],
    "vol_long_len": BREAKOUT_PARAM_SPECS["vol_long_len"]["default"],
    "vol_breakout_mult": BREAKOUT_PARAM_SPECS["vol_breakout_mult"]["default"],
    "breakout_return_min": BREAKOUT_PARAM_SPECS["breakout_return_min"]["default"],
    "breakout_false_filter_atr_pct_min": BREAKOUT_PARAM_SPECS[
        "breakout_false_filter_atr_pct_min"
    ]["default"],
    "breakout_reclaim_window_bars": BREAKOUT_PARAM_SPECS[
        "breakout_reclaim_window_bars"
    ]["default"],
    "breakout_reclaim_confirm_atr": BREAKOUT_PARAM_SPECS[
        "breakout_reclaim_confirm_atr"
    ]["default"],
    "min_history_trades": 0,
    "min_history_ev": -1.0,
    "min_history_win_rate": 0.0,
}
EXPERIMENT_RELATIVE_DIR = Path(
    "models/research/breakout_quality/"
    "binary_dl_filter_param_adaptation/risk_only_rolling"
)
BINARY_PIT_EXPECTED_RELATIVE_DIR = Path(
    "models/research/breakout_quality/binary_point_in_time_scores"
)


def _parse_args(argv=None):
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "A5/B5：Rule-based filters全關，只重訓ATR風險參數，再比較Binary DL關/開。"
            "A6/B6只有在Binary PIT scores完整時才允許執行；目前禁止用final model"
            "或research scores回灌optimizer歷史訓練。"
        )
    )
    parser.add_argument(
        "--dataset", choices=("reduced", "full"), default=settings.strategy_dataset
    )
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument(
        "--param-policy",
        choices=PARAM_POLICIES,
        default=PARAM_POLICY_BASE_FINALIST_BEST,
    )
    parser.add_argument(
        "--trials-per-fold",
        "--trials",
        dest="trials_per_fold",
        type=int,
        default=settings.strategy_adapt_trials_per_fold,
    )
    parser.add_argument(
        "--max-positions", type=int, default=settings.strategy_max_positions
    )
    parser.add_argument(
        "--rotation", choices=("off", "on"), default=settings.strategy_rotation
    )
    parser.add_argument(
        "--fixed-risk", type=float, default=settings.strategy_adapt_fixed_risk
    )
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=settings.strategy_adapt_max_position_cap_pct,
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
        root=root,
        params_path=None,
        param_policy=args.param_policy,
        allow_static_diagnostic=False,
    )
    if not params_path.is_file():
        raise FileNotFoundError(f"找不到Baseline rolling active params: {params_path}")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    source = _load_param_source(params_path)
    policy = _validate_requested_param_policy(source, args.param_policy)
    if str(source.get("kind") or "") != "rolling_active_param_ensemble":
        raise ValueError("A5/B5只接受rolling active-param ensemble Baseline")
    if int(policy.get("member_count_min") or 0) != 1 or int(
        policy.get("member_count_max") or 0
    ) != 1:
        raise ValueError(
            "risk-only fold freeze目前要求每個effective date恰有1個base-finalist-best member"
        )
    meta = dict(payload.get("meta") or {})
    summary = dict(payload.get("summary") or {})
    required_meta = (
        "window_mode",
        "first_oos_date",
        "last_oos_date",
        "train_window_months",
        "oos_horizon_months",
    )
    missing = [key for key in required_meta if meta.get(key) in (None, "")]
    if missing:
        raise ValueError("Baseline rolling params缺少meta欄位: " + ", ".join(missing))
    return {
        "path": params_path,
        "payload": payload,
        "source": source,
        "policy": policy,
        "meta": meta,
        "summary": summary,
        "sha256": compute_file_sha256(params_path),
    }


def _single_member_params_by_effective_date(
    baseline_contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    mapping = dict(
        baseline_contract["payload"].get("params_ensemble_by_effective_date") or {}
    )
    output: dict[str, dict[str, Any]] = {}
    for effective_date, raw_members in mapping.items():
        members = list(raw_members or [])
        if len(members) != 1:
            raise ValueError(
                "risk-only fold freeze要求每個effective date恰有1個member："
                f"effective_date={effective_date}, members={len(members)}"
            )
        params = dict((members[0] or {}).get("params") or {})
        if not params:
            raise ValueError(f"active param member缺少params: {effective_date}")
        output[pd.Timestamp(effective_date).strftime("%Y-%m-%d")] = params
    if not output:
        raise ValueError("Baseline沒有params_ensemble_by_effective_date")
    return output


def build_risk_only_fold_overrides(
    *,
    baseline_contract: dict[str, Any],
    args,
    training_dl_enabled: bool,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for effective_date, params in _single_member_params_by_effective_date(
        baseline_contract
    ).items():
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
                "breakout_quality_score_threshold": float(
                    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
                ),
                "fixed_risk": float(args.fixed_risk),
                "max_position_cap_pct": float(args.max_position_cap_pct),
            }
        )
        output[effective_date] = fixed
    return output


def _materialize_formal_rule_variant(
    *,
    baseline_contract: dict[str, Any],
    adapted_params_path: Path,
    output_path: Path,
    args,
) -> Path:
    """Keep A5 risk values but restore the original rolling rule configuration.

    The user requested that future diagnostics compare only the complete formal
    rule set against the all-off rule set, instead of repeating the A1→A4
    incremental ablation.  The same adapted ATR values are therefore replayed
    under both rule policies.
    """

    adapted = _load_json(adapted_params_path)
    if adapted is None:
        raise FileNotFoundError(f"找不到A5 adapted params: {adapted_params_path}")
    baseline_map = _single_member_params_by_effective_date(baseline_contract)
    adapted_members = dict(adapted.get("params_ensemble_by_effective_date") or {})
    restored_members: dict[str, list[dict[str, Any]]] = {}
    restored_flat: dict[str, dict[str, Any]] = {}
    for effective_date, baseline_params in baseline_map.items():
        members = list(adapted_members.get(effective_date) or [])
        if not members:
            raise ValueError(
                "A5 formal-rule variant缺少effective date member："
                f"effective_date={effective_date}"
            )
        restored_for_date = []
        for raw_member in members:
            source_member = dict(raw_member or {})
            adapted_params = dict(source_member.get("params") or {})
            merged = dict(baseline_params)
            for field_name in RISK_SEARCH_FIELDS:
                if field_name not in adapted_params:
                    raise ValueError(
                        f"A5 adapted params缺少risk field: {effective_date}/{field_name}"
                    )
                merged[field_name] = adapted_params[field_name]
            if "tp_percent" not in adapted_params:
                raise ValueError(
                    f"A5 adapted params缺少固定tp_percent: {effective_date}"
                )
            merged.update(
                {
                    "use_breakout_quality_filter": False,
                    "use_breakout_quality_ranking": False,
                    "breakout_quality_filter_id": str(args.filter_id),
                    "breakout_quality_score_threshold": float(
                        BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
                    ),
                    "fixed_risk": float(args.fixed_risk),
                    "max_position_cap_pct": float(args.max_position_cap_pct),
                    "tp_percent": float(adapted_params["tp_percent"]),
                }
            )
            source_member["params"] = merged
            restored_for_date.append(source_member)
        restored_members[effective_date] = restored_for_date
        restored_flat[effective_date] = dict(restored_for_date[0]["params"])

    payload = dict(adapted)
    payload["params_ensemble_by_effective_date"] = restored_members
    payload["params_by_effective_date"] = restored_flat
    payload["params_by_oos_year"] = {
        pd.Timestamp(date_text).strftime("%Y%m"): params
        for date_text, params in restored_flat.items()
    }
    payload["breakout_quality_param_adaptation"] = {
        "mode": "risk_only_dl_off_training_formal_rule_replay",
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "rule_policy": "formal_active_params",
        "training_rule_policy": "all_rule_filters_off",
        "training_dl_enabled": False,
    }
    _write_json(output_path, payload)
    return output_path


def _binary_pit_preflight(*, root: Path, args) -> dict[str, Any]:
    profile_dir = (
        root
        / BINARY_PIT_EXPECTED_RELATIVE_DIR
        / str(args.filter_id)
        / str(args.model_architecture)
        / str(args.experiment_profile)
    )
    manifest_path = profile_dir / "manifest.json"
    scores_path = profile_dir / "scores.csv"
    files_present = manifest_path.is_file() and scores_path.is_file()
    status = (
        "PRESENT_EXECUTION_WIRING_REQUIRED"
        if files_present
        else "BINARY_PIT_REQUIRED"
    )
    return {
        "status": status,
        "files_present": bool(files_present),
        "manifest_path": project_relative_display_path(manifest_path, project_root=root),
        "scores_path": project_relative_display_path(scores_path, project_root=root),
        "required_for": ["A6", "B6"],
        "forbidden_substitutes": [
            "canonical final-model forward-OOS scores",
            "research_scores.csv",
            "final model Selection scores",
        ],
    }


def _runtime_contract(
    *,
    root: Path,
    args,
    baseline_contract: dict[str, Any],
    fold_overrides: dict[str, dict[str, Any]],
    runtime_artifact,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "binary_dl_filter_risk_only_param_adaptation",
        "stage": "A5_B5",
        "dataset": str(args.dataset),
        "dataset_identity": build_source_data_inventory(root, args.dataset),
        "baseline_params_path": project_relative_display_path(
            baseline_contract["path"], project_root=root
        ),
        "baseline_params_sha256": baseline_contract["sha256"],
        "rolling_policy": {
            key: baseline_contract["meta"][key]
            for key in (
                "window_mode",
                "first_oos_date",
                "last_oos_date",
                "train_window_months",
                "oos_horizon_months",
            )
        },
        "trials_per_fold": int(args.trials_per_fold),
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "fixed_overrides_by_effective_date": fold_overrides,
        "binary_runtime": {
            "filter_id": str(args.filter_id),
            "model_architecture": str(args.model_architecture),
            "experiment_profile": str(args.experiment_profile),
            "manifest_sha256": compute_file_sha256(runtime_artifact.paths.manifest_path),
            "scores_sha256": compute_file_sha256(runtime_artifact.paths.score_path),
            "available_from": runtime_artifact.available_from.isoformat(),
            "available_through": runtime_artifact.available_through.isoformat(),
        },
        "fixed_risk": float(args.fixed_risk),
        "max_position_cap_pct": float(args.max_position_cap_pct),
        "max_positions": int(args.max_positions),
        "rotation": str(args.rotation),
    }
    payload["runtime_identity_sha256"] = _canonical_hash(payload)
    return payload


def _search_reusable(
    *, preflight_path: Path, params_path: Path, runtime_identity: str
) -> bool:
    preflight = _load_json(preflight_path)
    params = _load_json(params_path)
    if preflight is None or params is None:
        return False
    return (
        str(preflight.get("runtime_identity_sha256") or "") == runtime_identity
        and bool(dict(params.get("params_ensemble_by_effective_date") or {}))
    )


def _validate_risk_only_params(
    *,
    path: Path,
    baseline_contract: dict[str, Any],
    fold_overrides: dict[str, dict[str, Any]],
    args,
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        raise FileNotFoundError(f"A5/B5 optimizer未產生參數工件: {path}")
    meta = dict(payload.get("meta") or {})
    for key in (
        "first_oos_date",
        "last_oos_date",
        "train_window_months",
        "oos_horizon_months",
    ):
        if str(meta.get(key)) != str(baseline_contract["meta"].get(key)):
            raise ValueError(f"A5參數fold schedule不一致: {key}")
    if int(meta.get("trials_per_fold", 0) or 0) != int(args.trials_per_fold):
        raise ValueError("A5參數trials／fold與目前要求不一致")
    members_by_date = dict(payload.get("params_ensemble_by_effective_date") or {})
    for effective_date, expected_fixed in fold_overrides.items():
        members = list(members_by_date.get(effective_date) or [])
        if not members:
            raise ValueError(f"A5參數缺少effective date: {effective_date}")
        for member in members:
            params = dict((member or {}).get("params") or {})
            for field_name, expected in expected_fixed.items():
                actual = params.get(field_name)
                if isinstance(expected, float):
                    matched = actual is not None and math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                else:
                    matched = actual == expected
                if not matched:
                    raise ValueError(
                        "A5 risk-only固定契約未落入active params："
                        f"date={effective_date}, field={field_name}, "
                        f"actual={actual!r}, expected={expected!r}"
                    )
            for field_name in RISK_SEARCH_FIELDS:
                if field_name not in params:
                    raise ValueError(
                        f"A5 active params缺少risk field: {effective_date}/{field_name}"
                    )
    payload["breakout_quality_param_adaptation"] = {
        "mode": "risk_only_dl_off_training",
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "fixed_rule_contract": "all_rule_filters_off",
        "training_dl_enabled": False,
        "a6_b6_status": "BINARY_PIT_REQUIRED",
    }
    _write_json(path, payload)
    return payload


def _run_a5_optimizer(
    *,
    root: Path,
    args,
    settings,
    baseline_contract: dict[str, Any],
    runtime_artifact,
    output_dir: Path,
) -> dict[str, Any]:
    arm_dir = output_dir / "a5_dl_off_trained"
    active_param_dir = arm_dir / "active_params"
    optimizer_output_dir = arm_dir / "optimizer_runtime"
    active_param_dir.mkdir(parents=True, exist_ok=True)
    fold_overrides = build_risk_only_fold_overrides(
        baseline_contract=baseline_contract,
        args=args,
        training_dl_enabled=False,
    )
    contract = _runtime_contract(
        root=root,
        args=args,
        baseline_contract=baseline_contract,
        fold_overrides=fold_overrides,
        runtime_artifact=runtime_artifact,
    )
    preflight_path = arm_dir / "rolling_preflight.json"
    params_path = active_param_dir / "roos_base_best.json"
    reused = _search_reusable(
        preflight_path=preflight_path,
        params_path=params_path,
        runtime_identity=contract["runtime_identity_sha256"],
    )
    _write_json(
        preflight_path,
        {
            **contract,
            "status": "A5_RISK_ONLY_PREFLIGHT_PASS",
            "created_at": get_taipei_now().isoformat(),
        },
    )
    if not reused:
        base_policy = _build_base_policy(
            root=root,
            baseline_contract=baseline_contract,
        )
        base_policy.update(
            {
                "adaptation_scope": "binary_dl_filter_risk_only_a5",
                "evaluation_scope": "rolling_selection_diagnostic",
            }
        )
        session_spec = {
            "output_dir": str((optimizer_output_dir / "sessions").resolve()),
            FOLD_FIXED_STRATEGY_OVERRIDES_KEY: fold_overrides,
            "runtime_cache_identity": contract["runtime_identity_sha256"],
            "optimizer_fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
            "train_max_positions": int(args.max_positions),
            "train_enable_rotation": str(args.rotation) == "on",
        }
        outer_environ = dict(os.environ)
        outer_environ["V16_MODELS_DIR"] = resolve_models_dir(str(root))
        outer_environ["OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE"] = "sqlite"
        outer_environ["OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES"] = "1"
        exit_code = run_outer_rolling_oos(
            argv=_outer_rolling_argv(args=args, baseline_contract=baseline_contract),
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
            raise RuntimeError(f"A5 risk-only rolling optimizer失敗: {exit_code}")
    params_payload = _validate_risk_only_params(
        path=params_path,
        baseline_contract=baseline_contract,
        fold_overrides=fold_overrides,
        args=args,
    )
    summary_path = arm_dir / "rolling_optimizer_summary.json"
    summary = {
        "mode": "risk_only_dl_off_training",
        "status": "COMPLETED",
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "folds": int(dict(params_payload.get("summary") or {}).get("folds", 0) or 0),
        "trials_per_fold": int(args.trials_per_fold),
        "optimizer_search_reused": bool(reused),
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
    }
    _write_json(summary_path, summary)
    return {
        "params_path": params_path,
        "preflight_path": preflight_path,
        "summary_path": summary_path,
        "summary": summary,
        "contract": contract,
    }


def _risk_param_rows(
    *, baseline_contract: dict[str, Any], adapted_params_path: Path
) -> list[tuple[str, str, str, str, str]]:
    adapted = _load_json(adapted_params_path) or {}
    baseline_map = _single_member_params_by_effective_date(baseline_contract)
    adapted_map = {}
    for date_text, members in dict(
        adapted.get("params_ensemble_by_effective_date") or {}
    ).items():
        members = list(members or [])
        if members:
            adapted_map[str(date_text)] = dict((members[0] or {}).get("params") or {})
    rows = []
    for field_name in RISK_SEARCH_FIELDS:
        baseline_values = [baseline_map[d].get(field_name) for d in sorted(baseline_map)]
        adapted_values = [adapted_map.get(d, {}).get(field_name) for d in sorted(baseline_map)]
        rows.append(
            (
                field_name,
                ", ".join(str(v) for v in baseline_values),
                ", ".join(str(v) for v in adapted_values),
                str(min(adapted_values)) if adapted_values else "-",
                str(max(adapted_values)) if adapted_values else "-",
            )
        )
    return rows


def _metric(result: dict[str, Any], arm: str, key: str) -> Any:
    return dict(result.get(arm) or {}).get(key)


def _fmt(value: Any, *, unit: str = "", digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}{unit}"
    return str(value)


def _render_rule_policy_metric_table(
    pair_results: dict[str, dict[str, Any]],
) -> str:
    rows = []
    labels = {
        "formal": "原正式規則（全套）",
        "all_off": "Rule-based filters 全關",
    }
    for policy_name in ("formal", "all_off"):
        result = dict(pair_results.get(policy_name) or {})
        baseline = dict(result.get("no_filter") or {})
        quality = dict(result.get("quality_filter") or {})
        delta = dict(result.get("quality_filter_minus_no_filter") or {})
        rows.append(
            (
                labels[policy_name],
                _fmt(baseline.get("total_return_pct"), unit="%"),
                _fmt(quality.get("total_return_pct"), unit="%"),
                _fmt(delta.get("total_return_pct"), unit="pp"),
                _fmt(baseline.get("return_over_max_drawdown")),
                _fmt(quality.get("return_over_max_drawdown")),
                _fmt(delta.get("return_over_max_drawdown")),
                _fmt(delta.get("expected_value_r"), unit=" R"),
                _fmt(delta.get("avg_exposure_pct"), unit="pp"),
            )
        )
    return render_table(
        (
            "規則政策",
            "DL關報酬",
            "DL開報酬",
            "Δ報酬",
            "DL關RoMD",
            "DL開RoMD",
            "ΔRoMD",
            "ΔEV",
            "Δ曝險",
        ),
        rows,
    )


def _render_report(
    *,
    args,
    baseline_contract: dict[str, Any],
    a5_arm: dict[str, Any] | None,
    pair_results: dict[str, dict[str, Any]] | None,
    binary_pit: dict[str, Any],
    plan_only: bool,
) -> str:
    lines = [
        render_title("Binary DL Filter Risk-only Parameter Adaptation A5／B5 Gate"),
        render_key_values(
            (
                ("參數基準", args.param_policy),
                ("訓練規則", "Rule-based filters 全關"),
                ("搜尋參數", " / ".join(RISK_SEARCH_FIELDS)),
                ("固定 Buy sort", "原 position-aware buy-sort"),
                ("A6／B6", "等待 Binary PIT scores"),
            )
        ),
        render_section("1. 實驗設計"),
        render_table(
            ("組別", "參數訓練環境", "回放 DL", "狀態"),
            (
                ("A5", "Rule-based filters全關、DL關，只搜尋風險參數", "關", "PLAN" if plan_only else "完成"),
                ("B5", "沿用A5同一套參數", "開", "PLAN" if plan_only else "完成"),
                ("A6", "Rule-based filters全關、DL開，只搜尋風險參數", "關", "Binary PIT待建"),
                ("B6", "沿用A6同一套參數", "開", "Binary PIT待建"),
            ),
        ),
    ]
    if a5_arm is not None:
        lines.extend(
            (
                render_section("2. 風險參數變化（依2021～2026 effective dates）"),
                render_table(
                    ("參數", "原ROOS", "A5新ROOS", "新最小", "新最大"),
                    _risk_param_rows(
                        baseline_contract=baseline_contract,
                        adapted_params_path=a5_arm["params_path"],
                    ),
                ),
            )
        )
    if pair_results is not None:
        lines.extend(
            (
                render_section("3. A5風險參數下：全套規則／全關比較"),
                _render_rule_policy_metric_table(pair_results),
                "主判定仍看Rule-based filters全關列的B5−A5；原正式規則列只用來確認規則政策交互作用，不再逐項拆History、Re-entry或KC。",
            )
        )
    lines.extend(
        (
            render_section("4. A6／B6 無前視 Gate"),
            render_key_values(
                (
                    ("狀態", binary_pit["status"]),
                    ("必要 Manifest", binary_pit["manifest_path"]),
                    ("必要 Scores", binary_pit["scores_path"]),
                    (
                        "禁止替代",
                        "final-model forward-OOS／research／Selection in-sample scores",
                    ),
                )
            ),
            "判讀：先執行A5／B5。A6／B6必須另有每個optimizer歷史日期可用的Binary PIT score，且尚需把該PIT source接入optimizer runtime；禁止以final 9A scores替代。",
        )
    )
    return "\n\n".join(lines)


def run_param_adaptation_gate(*, project_root=PROJECT_ROOT, argv=None) -> dict[str, Any]:
    args = _parse_args(argv)
    root = Path(project_root).resolve()
    settings = get_breakout_quality_workflow_settings()
    if args.param_policy != PARAM_POLICY_BASE_FINALIST_BEST:
        raise ValueError("第一輪risk-only adaptation固定使用base-finalist-best")
    if int(args.trials_per_fold) < 1:
        raise ValueError("trials-per-fold必須>=1")
    if int(args.max_positions) < 1:
        raise ValueError("max-positions必須>=1")
    if not 0.0 < float(args.fixed_risk) <= 1.0:
        raise ValueError("fixed-risk必須介於0與1")
    if not 0.0 < float(args.max_position_cap_pct) <= 1.0:
        raise ValueError("max-position-cap-pct必須介於0與1")

    runtime_artifact = load_runtime_artifact_contract(
        str(root),
        str(args.filter_id),
        str(args.model_architecture),
        str(args.experiment_profile),
    )
    baseline_contract = _load_baseline_contract(root=root, args=args)
    binary_pit = _binary_pit_preflight(root=root, args=args)
    output_dir = root / EXPERIMENT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_path = output_dir / "strategy_dl_filter_param_adapt_plan.json"
    plan_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLAN_ONLY" if args.plan_only else EXPERIMENT_STATUS,
        "baseline_params": {
            "path": project_relative_display_path(
                baseline_contract["path"], project_root=root
            ),
            "sha256": baseline_contract["sha256"],
        },
        "risk_search_fields": list(RISK_SEARCH_FIELDS),
        "fixed_rule_contract": {
            "optional_entry_filters": "all_off",
            **ALL_RULE_FILTERS_OFF_OVERRIDES,
        },
        "a6_b6_preflight": binary_pit,
        "created_at": get_taipei_now().isoformat(),
    }
    _write_json(plan_path, plan_payload)

    a5_arm = None
    pair_results = None
    if not args.plan_only:
        configure_optuna_logging()
        a5_arm = _run_a5_optimizer(
            root=root,
            args=args,
            settings=settings,
            baseline_contract=baseline_contract,
            runtime_artifact=runtime_artifact,
            output_dir=output_dir,
        )
        formal_params_path = _materialize_formal_rule_variant(
            baseline_contract=baseline_contract,
            adapted_params_path=a5_arm["params_path"],
            output_path=(output_dir / "a5_dl_off_trained" / "formal_rules" / "roos_base_best.json"),
            args=args,
        )
        formal_result = run_comparison(
            project_root=root,
            dataset=args.dataset,
            params_path=str(formal_params_path),
            param_policy=args.param_policy,
            max_positions=int(args.max_positions),
            enable_rotation=str(args.rotation) == "on",
            fixed_risk=float(args.fixed_risk),
            max_position_cap_pct=float(args.max_position_cap_pct),
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
            optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(args.experiment_profile),
            output_dir_override=(output_dir / "a5_b5_replay" / "formal_rules"),
            quiet=bool(args.quiet),
        )
        all_off_result = run_comparison(
            project_root=root,
            dataset=args.dataset,
            params_path=str(a5_arm["params_path"]),
            param_policy=args.param_policy,
            max_positions=int(args.max_positions),
            enable_rotation=str(args.rotation) == "on",
            fixed_risk=float(args.fixed_risk),
            max_position_cap_pct=float(args.max_position_cap_pct),
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
            optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(args.experiment_profile),
            output_dir_override=(output_dir / "a5_b5_replay" / "all_rules_off"),
            quiet=bool(args.quiet),
            shared_param_overrides=ALL_RULE_FILTERS_OFF_OVERRIDES,
        )
        pair_results = {
            "formal": formal_result,
            "all_off": all_off_result,
        }

    report = _render_report(
        args=args,
        baseline_contract=baseline_contract,
        a5_arm=a5_arm,
        pair_results=pair_results,
        binary_pit=binary_pit,
        plan_only=bool(args.plan_only),
    )
    print("\n" + report)
    markdown_path = output_dir / "strategy_dl_filter_param_adapt_gate.md"
    json_path = output_dir / "strategy_dl_filter_param_adapt_gate.json"
    markdown_path.write_text(report + "\n", encoding="utf-8")
    result = {
        **plan_payload,
        "status": "PLAN_ONLY" if args.plan_only else EXPERIMENT_STATUS,
        "a5_optimizer": None if a5_arm is None else {
            "params_path": project_relative_display_path(
                a5_arm["params_path"], project_root=root
            ),
            "summary": a5_arm["summary"],
        },
        "a5_b5": pair_results,
        "a6_b6": {
            "status": "BINARY_PIT_REQUIRED",
            "preflight": binary_pit,
        },
    }
    _write_json(json_path, result)
    print_artifact_paths(
        (
            ("A5/B5計畫", plan_path),
            ("A5/B5報表", markdown_path),
            ("A5/B5 JSON", json_path),
        ),
        project_root=root,
    )
    return result


def main(argv=None):
    run_param_adaptation_gate(argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
