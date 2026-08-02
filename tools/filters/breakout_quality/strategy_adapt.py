"""Selection PIT Score-ranking strategy parameter adaptation and fitted comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config.breakout_quality import get_breakout_quality_workflow_settings
from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT
from core.dataset_profiles import get_dataset_dir
from core.params_io import build_params_from_mapping, params_to_json_dict
from core.raw_universe_contract import RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD
from core.runtime_utils import get_taipei_now
from core.walk_forward_policy import build_optimizer_effective_policy_fingerprint
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.console_report import (
    compact_console_enabled,
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_selection_point_in_time_ranking_contract,
)
from filters.breakout_quality.runtime import breakout_quality_ranking_source_context
from filters.breakout_quality.source_inventory import build_source_data_inventory
from strategies.breakout.search_space import (
    BREAKOUT_OPTIMIZER_SEARCH_SPACE,
    get_breakout_optimizer_required_min_rows,
)
from tools.filters.breakout_quality.audit_score_ranking_capture import (
    build_score_ranking_capture_audit,
    render_capture_audit_console,
    write_score_ranking_capture_audit_outputs,
)
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_SCORE_RANKING,
    PARAM_POLICY_BASE_FINALIST_BEST,
    _assert_shared_benchmark,
    _delta,
    _flatten_candidate_replay_rows,
    _flatten_selected_buy_rows,
    _run_scenario,
    _scenario_summary,
    _selection_target_lookup,
    _strategy_selection_diagnostics,
    _to_json_native,
    _yearly_frame,
    run_comparison,
)
from tools.optimizer.main import _select_finalist_entry_by_selector
from tools.optimizer.prep import load_all_raw_data
from tools.optimizer.robustness import print_local_min_score_finalist_review
from tools.optimizer.runtime import (
    create_optimizer_study,
    resolve_optimizer_single_fold_search_parallel_trials,
)
from tools.optimizer.session_factory import build_optimizer_session, configure_optuna_logging
from tools.optimizer.study_utils import (
    build_best_params_payload_from_trial,
    is_qualified_trial_value,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
ADAPTATION_STATUS = "FITTED_SELECTION_DIAGNOSTIC"
ADAPTATION_RELATIVE_DIR = Path(
    "models/research/breakout_quality/score_ranking_adaptation"
)
CURRENT_PAIR_MANIFEST_FILENAME = "adaptation_pair_manifest.json"

def _parse_args(argv=None):
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "固定Selection PIT Score Sort執行策略參數適應，並輸出"
            "Baseline／Sort Only／Adapted三組Selection診斷。"
        )
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default=settings.strategy_dataset)
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument(
        "--param-policy",
        choices=(PARAM_POLICY_BASE_FINALIST_BEST,),
        default=PARAM_POLICY_BASE_FINALIST_BEST,
    )
    parser.add_argument("--trials", type=int, default=settings.strategy_adapt_trials)
    parser.add_argument("--max-positions", type=int, default=settings.strategy_max_positions)
    parser.add_argument("--rotation", choices=("off", "on"), default=settings.strategy_rotation)
    parser.add_argument("--fixed-risk", type=float, default=settings.strategy_adapt_fixed_risk)
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=settings.strategy_adapt_max_position_cap_pct,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)

def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_json_native(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

def _current_pair_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "strategy_comparison_json": output_dir / "strategy_comparison.json",
        "strategy_comparison_markdown": output_dir / "strategy_comparison.md",
        "yearly_csv": output_dir / "yearly_returns_comparison.csv",
        "capture_json": output_dir / "score_ranking_capture_audit.json",
        "capture_markdown": output_dir / "score_ranking_capture_audit.md",
        "baseline_equity": output_dir / "no_filter_equity.csv",
        "sort_only_equity": output_dir / "score_ranking_equity.csv",
        "baseline_trades": output_dir / "no_filter_trades.csv",
        "sort_only_trades": output_dir / "score_ranking_trades.csv",
        "baseline_capacity": output_dir / "no_filter_daily_capacity.csv",
        "sort_only_capacity": output_dir / "score_ranking_daily_capacity.csv",
        "baseline_orderable": output_dir / "no_filter_orderable_target_diagnostics.csv",
        "sort_only_orderable": output_dir / "score_ranking_orderable_target_diagnostics.csv",
        "baseline_selected": output_dir / "no_filter_selected_target_diagnostics.csv",
        "sort_only_selected": output_dir / "score_ranking_selected_target_diagnostics.csv",
        "baseline_capture_lifecycle": output_dir / "no_filter_capture_lifecycle.csv",
        "sort_only_capture_lifecycle": output_dir / "score_ranking_capture_lifecycle.csv",
        "capture_yearly": output_dir / "score_ranking_capture_yearly.csv",
        "capture_scenarios": output_dir / "score_ranking_capture_scenarios.csv",
    }

def _load_current_pair_if_compatible(
    *, root: Path, output_dir: Path, runtime_identity_sha256: str
) -> dict[str, Any] | None:
    manifest_path = output_dir / CURRENT_PAIR_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if str(manifest.get("runtime_identity_sha256") or "") != str(
        runtime_identity_sha256
    ):
        return None
    recorded = dict(manifest.get("artifacts") or {})
    expected_paths = _current_pair_artifact_paths(output_dir)
    for key, path in expected_paths.items():
        entry = dict(recorded.get(key) or {})
        expected_display_path = project_relative_display_path(path, project_root=root)
        if (
            not path.is_file()
            or str(entry.get("path") or "") != expected_display_path
            or str(entry.get("sha256") or "") != compute_file_sha256(path)
        ):
            return None
    comparison_path = expected_paths["strategy_comparison_json"]
    try:
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        return None
    return payload

def _write_current_pair_manifest(
    *, root: Path, output_dir: Path, runtime_identity_sha256: str
) -> Path:
    artifact_paths = _current_pair_artifact_paths(output_dir)
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Baseline／Sort Only完成後缺少必要工件: " + ", ".join(missing)
        )
    manifest_path = output_dir / CURRENT_PAIR_MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_identity_sha256": str(runtime_identity_sha256),
            "created_at": get_taipei_now().isoformat(),
            "artifacts": {
                key: {
                    "path": project_relative_display_path(path, project_root=root),
                    "sha256": compute_file_sha256(path),
                }
                for key, path in artifact_paths.items()
            },
        },
    )
    return manifest_path

def _load_or_run_current_pair(
    *, root: Path, args, runtime_contract: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], bool]:
    payload = _load_current_pair_if_compatible(
        root=root,
        output_dir=output_dir,
        runtime_identity_sha256=runtime_contract["runtime_identity_sha256"],
    )
    if payload is not None:
        print("\n[Baseline／Sort Only] identity與工件hash一致，沿用既有結果。")
        return payload, True
    payload = run_comparison(
        project_root=root,
        dataset=args.dataset,
        param_policy=args.param_policy,
        max_positions=args.max_positions,
        enable_rotation=str(args.rotation) == "on",
        fixed_risk=args.fixed_risk,
        max_position_cap_pct=args.max_position_cap_pct,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        filter_id=args.filter_id,
        score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        output_dir_override=output_dir,
        quiet=args.quiet,
    )
    _write_current_pair_manifest(
        root=root,
        output_dir=output_dir,
        runtime_identity_sha256=runtime_contract["runtime_identity_sha256"],
    )
    return payload, False

def _validate_fixed_contract(args, settings) -> None:
    if settings.strategy_comparison_mode != "score-ranking":
        raise ValueError("strategy adaptation只支援continuous-ranker score-ranking workflow")
    if settings.strategy_score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        raise ValueError("strategy adaptation只接受selection_point_in_time Score source")
    fixed_identity = {
        "filter_id": (args.filter_id, settings.filter_id),
        "model_architecture": (args.model_architecture, settings.model_architecture),
        "experiment_profile": (args.experiment_profile, settings.experiment_profile),
    }
    mismatches = [
        f"{field}: actual={actual!r}, workflow={expected!r}"
        for field, (actual, expected) in fixed_identity.items()
        if str(actual) != str(expected)
    ]
    if mismatches:
        raise ValueError(
            "strategy adaptation必須使用目前workflow凍結的模型identity："
            + "; ".join(mismatches)
        )
    if args.param_policy != PARAM_POLICY_BASE_FINALIST_BEST:
        raise ValueError("第一輪strategy adaptation固定使用base-finalist-best")
    if args.rotation not in {"off", "on"}:
        raise ValueError("rotation只接受off或on")
    if int(args.max_positions) < 1:
        raise ValueError("max_positions必須>=1")
    if not 0.0 < float(args.fixed_risk) <= 1.0:
        raise ValueError("fixed_risk必須介於0與1之間")
    if not 0.0 < float(args.max_position_cap_pct) <= 1.0:
        raise ValueError("max_position_cap_pct必須介於0與1之間")
    if int(args.trials) < 1:
        raise ValueError("trials必須>=1")

def _validate_pit_contract_against_settings(pit_contract, settings) -> None:
    manifest = dict(getattr(pit_contract, "manifest", {}) or {})
    checks = {
        "continuous_target_id": (
            str(pit_contract.continuous_target_id),
            str(settings.continuous_target_id),
        ),
        "seed": (int(pit_contract.seed), int(settings.seed)),
        "score_start": (
            str(pit_contract.available_from),
            str(settings.point_in_time_score_start_date),
        ),
        "fold_months": (
            int(manifest.get("fold_months", -1)),
            int(settings.point_in_time_fold_months),
        ),
        "inner_validation_months": (
            int(manifest.get("inner_validation_months", -1)),
            int(settings.point_in_time_inner_validation_months),
        ),
    }
    if settings.point_in_time_score_end_date is not None:
        checks["score_end"] = (
            str(pit_contract.available_through),
            str(settings.point_in_time_score_end_date),
        )
    mismatches = [
        f"{field}: artifact={actual!r}, workflow={expected!r}"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "Selection PIT工件與目前workflow凍結契約不一致："
            + "; ".join(mismatches)
        )

def _build_runtime_contract(
    *, root: Path, args, settings, pit_contract, walk_forward_policy=None
) -> dict[str, Any]:
    score_hash = compute_file_sha256(pit_contract.score_path)
    manifest_hash = compute_file_sha256(pit_contract.manifest_path)
    audit_hash = compute_file_sha256(pit_contract.audit_path)
    search_space_snapshot = {
        "strategy_search_space": BREAKOUT_OPTIMIZER_SEARCH_SPACE,
        "fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
    }
    effective_policy = build_optimizer_effective_policy_fingerprint(
        _build_walk_forward_policy(pit_contract)
        if walk_forward_policy is None
        else walk_forward_policy
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "dataset_identity": build_source_data_inventory(root, args.dataset),
        "selection_period": {
            "start": pit_contract.available_from,
            "end": pit_contract.available_through,
        },
        "optimizer_policy": "base-finalist-best",
        "objective": "split_train_romd",
        "optimizer_effective_policy": effective_policy,
        "search_space_identity_sha256": _canonical_json_sha256(search_space_snapshot),
        "search_space_snapshot": search_space_snapshot,
        "seed": int(settings.seed),
        "pit_identity": {
            "score_csv_sha256": score_hash,
            "manifest_sha256": manifest_hash,
            "audit_sha256": audit_hash,
            "score_path": project_relative_display_path(
                pit_contract.score_path, project_root=root
            ),
            "manifest_path": project_relative_display_path(
                pit_contract.manifest_path, project_root=root
            ),
            "audit_path": project_relative_display_path(
                pit_contract.audit_path, project_root=root
            ),
        },
        "model_identity": {
            "filter_id": str(args.filter_id),
            "architecture": str(args.model_architecture),
            "experiment_profile": str(args.experiment_profile),
            "continuous_target_id": str(pit_contract.continuous_target_id),
        },
        "ranking_mode": "breakout_quality_score_desc",
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "fixed_runtime": {
            "use_breakout_quality_ranking": True,
            "use_breakout_quality_filter": False,
            "fixed_risk": float(args.fixed_risk),
            "max_position_cap_pct": float(args.max_position_cap_pct),
            "max_positions": int(args.max_positions),
            "rotation": str(args.rotation) == "on",
            "tp_percent_policy": {
                "fixed_value": OPTIMIZER_FIXED_TP_PERCENT,
                "searched": OPTIMIZER_FIXED_TP_PERCENT is None,
            },
        },
        "runtime_restrictions": {
            "ranking_switch_searched": False,
            "score_weight_searched": False,
            "score_threshold_searched": False,
            "future_target_used_for_runtime": False,
            "future_target_used_for_objective": False,
            "oos_used_for_fitting": False,
        },
    }
    contract["runtime_identity_sha256"] = _canonical_json_sha256(contract)
    return contract

def _canonical_execution_argv(args) -> list[str]:
    argv = [
        "python",
        "apps/breakout_quality.py",
        "strategy-adapt",
        "--dataset",
        str(args.dataset),
        "--filter-id",
        str(args.filter_id),
        "--model-architecture",
        str(args.model_architecture),
        "--experiment-profile",
        str(args.experiment_profile),
        "--param-policy",
        str(args.param_policy),
        "--trials",
        str(int(args.trials)),
        "--max-positions",
        str(int(args.max_positions)),
        "--rotation",
        str(args.rotation),
        "--fixed-risk",
        str(float(args.fixed_risk)),
        "--max-position-cap-pct",
        str(float(args.max_position_cap_pct)),
    ]
    if bool(args.quiet):
        argv.append("--quiet")
    return argv

def _build_walk_forward_policy(pit_contract) -> dict[str, Any]:
    start = pd.Timestamp(pit_contract.available_from)
    end = pd.Timestamp(pit_contract.available_through)
    required_min_rows = int(get_breakout_optimizer_required_min_rows())
    return {
        "model_mode": "study",
        "study_scope": "full",
        "adaptation_scope": "selection_score_ranking_adaptation",
        "evaluation_scope": "fitted_selection_diagnostic",
        "objective_mode": "split_train_romd",
        "selection_start_year": int(start.year),
        "train_start_year": int(start.year),
        "search_train_end_year": int(end.year),
        "selection_start_date": start.strftime("%Y-%m-%d"),
        "train_start_date": start.strftime("%Y-%m-%d"),
        "search_train_end_date": end.strftime("%Y-%m-%d"),
        "oos_start_year": None,
        "oos_end_year": None,
        "oos_start_date": None,
        "oos_end_date": None,
        "latest_data_date": end.strftime("%Y-%m-%d"),
        "min_train_years": max(1, int(end.year) - int(start.year) + 1),
        RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: required_min_rows,
    }

def _assert_study_identity(study, runtime_contract: dict[str, Any]) -> None:
    key = "breakout_quality_strategy_adaptation_runtime_identity_sha256"
    expected = str(runtime_contract["runtime_identity_sha256"])
    actual = str(getattr(study, "user_attrs", {}).get(key) or "")
    trials = list(getattr(study, "trials", []) or [])
    if trials and actual != expected:
        raise RuntimeError(
            "strategy adaptation study缺少identity或與目前PIT／search space不一致；"
            "禁止接續舊study。請移除對應adaptation_study.db後重跑。"
        )
    study.set_user_attr(key, expected)
    study.set_user_attr("breakout_quality_strategy_adaptation_contract", runtime_contract)

def _select_adapted_params(*, study, session) -> tuple[dict[str, Any], dict[str, Any]]:
    finalists, _best_trial = print_local_min_score_finalist_review(
        study,
        session=session,
        objective_mode=session.objective_mode,
        colors=session.colors,
        winner_trial=None,
        emit_table=True,
        show_progress=True,
    )
    selected_entry = _select_finalist_entry_by_selector(
        finalists,
        objective_mode=session.objective_mode,
        selector="base_finalist_best",
    )
    selected_trial = None if selected_entry is None else selected_entry.get("trial")
    if selected_trial is None or not is_qualified_trial_value(selected_trial.value):
        raise RuntimeError("optimizer完成，但base-finalist-best沒有可匯出的合格trial")
    raw_payload = build_best_params_payload_from_trial(
        selected_trial, fixed_tp_percent=session.optimizer_fixed_tp_percent
    )
    params = session.apply_fixed_strategy_param_overrides(
        build_params_from_mapping(raw_payload)
    )
    payload = params_to_json_dict(params)
    summary = {
        "selected_trial_number": int(selected_trial.number) + 1,
        "selected_trial_value": float(selected_trial.value),
        "selector": "base_finalist_best",
        "base_score": float(selected_entry.get("base_score", selected_trial.value)),
        "local_min_score": float(selected_entry.get("local_min_score", float("nan"))),
        "local_retention": float(selected_entry.get("local_retention", float("nan"))),
        "local_gate": bool(selected_entry.get("gate_pass", False)),
        "completed_trial_count": int(
            sum(str(getattr(trial, "state", "")).endswith("COMPLETE") for trial in study.trials)
        ),
        "total_trial_count": int(len(study.trials)),
    }
    return payload, summary

def _build_param_comparison_rows(
    *,
    params_path: str | Path,
    adapted_params_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare one fitted parameter set with the historical active-param schedule."""

    source_path = Path(params_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    mapping = payload.get("params_by_effective_date")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("參數差異報表需要rolling params_by_effective_date")

    searchable_fields = set(BREAKOUT_OPTIMIZER_SEARCH_SPACE)
    searchable_fields.update(("tp_percent", "fixed_risk", "max_position_cap_pct"))
    rows: list[dict[str, Any]] = []
    for field_name in sorted(searchable_fields):
        if field_name not in adapted_params_payload:
            continue
        historical_values = [
            raw_params.get(field_name)
            for raw_params in mapping.values()
            if isinstance(raw_params, dict) and field_name in raw_params
        ]
        if not historical_values:
            continue
        adapted_value = adapted_params_payload[field_name]
        numeric_values = [
            float(value)
            for value in historical_values
            if not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ]
        if len(numeric_values) == len(historical_values):
            rows.append({
                "parameter": field_name,
                "historical_min": min(numeric_values),
                "historical_median": float(pd.Series(numeric_values).median()),
                "historical_max": max(numeric_values),
                "historical_unique_count": len(set(numeric_values)),
                "adapted_value": adapted_value,
                "comparison_type": "numeric_range",
            })
            continue
        unique_values = sorted(
            {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in historical_values}
        )
        rows.append({
            "parameter": field_name,
            "historical_values": [json.loads(value) for value in unique_values],
            "historical_unique_count": len(unique_values),
            "adapted_value": adapted_value,
            "comparison_type": "categorical_set",
        })
    return rows


def _param_comparison_table_rows(result: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for row in list(result.get("parameter_comparison") or []):
        if row.get("comparison_type") == "numeric_range":
            rows.append((
                str(row.get("parameter")),
                _fmt(row.get("historical_min")),
                _fmt(row.get("historical_median")),
                _fmt(row.get("historical_max")),
                _fmt(row.get("adapted_value")),
            ))
        else:
            historical = ", ".join(str(value) for value in row.get("historical_values") or [])
            rows.append((
                str(row.get("parameter")),
                historical,
                historical,
                historical,
                str(row.get("adapted_value")),
            ))
    return rows


def _run_three_way_comparison(
    *,
    root: Path,
    args,
    pit_contract,
    current_payload: dict[str, Any],
    adapted_params_payload: dict[str, Any],
    current_pair_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    metadata = dict(current_payload["metadata"])
    comparison_dir = Path(current_pair_dir).resolve()
    required = {
        "baseline_trades": comparison_dir / "no_filter_trades.csv",
        "baseline_capacity": comparison_dir / "no_filter_daily_capacity.csv",
        "baseline_selected": comparison_dir / "no_filter_selected_target_diagnostics.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("三組比較缺少Baseline工件: " + ", ".join(missing))

    adapted_params = build_params_from_mapping(adapted_params_payload)
    replay_counts: dict[str, Any] = {}
    ranking_source = {
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "model_architecture": args.model_architecture,
        "experiment_profile": args.experiment_profile,
    }
    adapted_payload = _run_scenario(
        name="adapted_score_ranking",
        data_dir=Path(get_dataset_dir(str(root), args.dataset)).resolve(),
        param_source_kind="single_param",
        params=adapted_params,
        start_date=pit_contract.available_from,
        end_date=pit_contract.available_through,
        max_positions=int(args.max_positions),
        enable_rotation=str(args.rotation) == "on",
        quiet=bool(args.quiet),
        replay_counts=replay_counts,
        ranking_source=ranking_source,
    )
    adapted = _scenario_summary(adapted_payload)
    baseline = dict(current_payload["no_filter"])
    sort_only = dict(current_payload["score_ranking"])

    baseline_equity = pd.read_csv(comparison_dir / "no_filter_equity.csv", encoding="utf-8-sig")
    _assert_shared_benchmark(
        {
            "benchmark_return_pct": baseline["benchmark_return_pct"],
            "benchmark_max_drawdown_pct": baseline["benchmark_max_drawdown_pct"],
            "benchmark_annual_return_pct": baseline["benchmark_annual_return_pct"],
            "equity_curve": baseline_equity,
        },
        adapted_payload,
    )

    adapted_orderable = _flatten_candidate_replay_rows(replay_counts, "orderable_rows")
    adapted_selected = _flatten_selected_buy_rows(adapted_payload["trade_history"])
    lookup = _selection_target_lookup(
        root=root,
        filter_id=args.filter_id,
        architecture=args.model_architecture,
        profile=args.experiment_profile,
    )
    adapted_diag, adapted_orderable_joined, adapted_selected_joined = (
        _strategy_selection_diagnostics(
            orderable=adapted_orderable,
            selected=adapted_selected,
            lookup=lookup,
        )
    )
    existing_diagnostics = dict(current_payload.get("selection_diagnostics") or {})
    baseline_diag = dict(existing_diagnostics.get("no_filter") or {})
    sort_diag = dict(existing_diagnostics.get("score_ranking") or {})

    adapted_payload["equity_curve"].to_csv(
        output_dir / "adapted_equity.csv", index=False, encoding="utf-8-sig"
    )
    adapted_payload["trade_history"].to_csv(
        output_dir / "adapted_trades.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(adapted_payload["profile"].get("portfolio_capacity_rows") or []).to_csv(
        output_dir / "adapted_daily_capacity.csv", index=False, encoding="utf-8-sig"
    )
    adapted_orderable_joined.to_csv(
        output_dir / "adapted_orderable_target_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adapted_selected_joined.to_csv(
        output_dir / "adapted_selected_target_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    adapted_yearly = _yearly_frame(adapted_payload["profile"], "adapted")
    adapted_yearly.to_csv(
        output_dir / "adapted_yearly_returns.csv", index=False, encoding="utf-8-sig"
    )
    current_yearly = pd.DataFrame(current_payload.get("yearly") or [])
    yearly_keys = [
        key
        for key in ("year", "is_full_year", "start_date", "end_date")
        if key in current_yearly.columns and key in adapted_yearly.columns
    ]
    if not yearly_keys:
        yearly_keys = ["year"]
    three_way_yearly = current_yearly.merge(
        adapted_yearly, on=yearly_keys, how="outer", validate="one_to_one"
    ).sort_values("year").reset_index(drop=True)
    three_way_yearly["adapted_minus_baseline_pct"] = (
        three_way_yearly["adapted_return_pct"]
        - three_way_yearly["no_filter_return_pct"]
    )
    three_way_yearly["adapted_minus_sort_only_pct"] = (
        three_way_yearly["adapted_return_pct"]
        - three_way_yearly["score_ranking_return_pct"]
    )
    three_way_yearly.to_csv(
        output_dir / "strategy_adaptation_yearly_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    adapted_capture = build_score_ranking_capture_audit(
        metadata={
            **metadata,
            "comparison_design": "fitted_selection_adaptation_diagnostic",
        },
        baseline_summary=baseline,
        score_sort_summary=adapted,
        baseline_trade_history=pd.read_csv(required["baseline_trades"], encoding="utf-8-sig"),
        score_sort_trade_history=adapted_payload["trade_history"],
        baseline_selected_target_diagnostics=pd.read_csv(
            required["baseline_selected"], encoding="utf-8-sig"
        ),
        score_sort_selected_target_diagnostics=adapted_selected_joined,
        selection_diagnostics={
            "score_ranking_minus_no_filter": _delta(adapted_diag, baseline_diag)
        },
        baseline_daily_capacity=pd.read_csv(
            required["baseline_capacity"], encoding="utf-8-sig"
        ),
        score_sort_daily_capacity=pd.DataFrame(
            adapted_payload["profile"].get("portfolio_capacity_rows") or []
        ),
    )
    adapted_capture_payload = write_score_ranking_capture_audit_outputs(
        result=adapted_capture,
        output_dir=output_dir,
    )

    parameter_comparison = _build_param_comparison_rows(
        params_path=metadata["params_path"],
        adapted_params_payload=adapted_params_payload,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": ADAPTATION_STATUS,
        "metadata": {
            **metadata,
            "adapted_param_source_kind": "single_param_fitted_selection",
            "adapted_result_interpretation": ADAPTATION_STATUS,
            "future_target_used_for_runtime": False,
            "oos_generalization_claimed": False,
        },
        "baseline": baseline,
        "sort_only": sort_only,
        "adapted": adapted,
        "sort_only_minus_baseline": _delta(sort_only, baseline),
        "adapted_minus_baseline": _delta(adapted, baseline),
        "adapted_minus_sort_only": _delta(adapted, sort_only),
        "selection_diagnostics": {
            "baseline": baseline_diag,
            "sort_only": sort_diag,
            "adapted": adapted_diag,
            "adapted_minus_baseline": _delta(adapted_diag, baseline_diag),
            "adapted_minus_sort_only": _delta(adapted_diag, sort_diag),
            "future_target_join_stage": "post_replay_offline_diagnostic_only",
            "future_target_used_for_runtime_sort": False,
        },
        "current_capture_audit": current_payload.get("score_ranking_capture_audit"),
        "adapted_capture_audit": adapted_capture_payload,
        "yearly": three_way_yearly.to_dict("records"),
        "adapted_params": adapted_params_payload,
        "parameter_comparison": parameter_comparison,
    }
    _write_json(output_dir / "strategy_adaptation_comparison.json", result)
    (output_dir / "strategy_adaptation_comparison.md").write_text(
        _render_three_way_markdown(result), encoding="utf-8"
    )
    print("\n" + _render_three_way_console(result))
    print("\n" + render_capture_audit_console(adapted_capture))
    return result

def _metric_rows(result: dict[str, Any]):
    rows = (
        ("淨總報酬", "total_return_pct", "%"),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("Return／MDD", "return_over_max_drawdown", ""),
        ("年化報酬", "annual_return_pct", "%"),
        ("Log R²", "log_r_squared", ""),
        ("月勝率", "monthly_win_rate_pct", "%"),
        ("交易數", "trade_count", ""),
        ("勝率", "win_rate_pct", "%"),
        ("Payoff", "payoff_ratio", ""),
        ("EV", "expected_value_r", " R"),
        ("平均曝險", "avg_exposure_pct", "%"),
        ("平均候選供給", "avg_orderable_candidates", ""),
        ("候選不足日", "candidate_supply_gap_days", " 日"),
        ("平均預留金額", "avg_reserved_total", ""),
        ("平均投入金額", "avg_invested_total", ""),
        ("投入／預留比", "avg_invested_vs_reserved_pct", "%"),
        ("初始停損距離", "avg_stop_distance_pct", "%"),
        ("平均 Realized R", "avg_realized_r", " R"),
        ("平均 Target R", "avg_target_r", " R"),
        ("Aggregate capture", "aggregate_target_capture_ratio", ""),
        ("Median capture", "median_target_capture_ratio", ""),
        ("Target ≥ 0.5R capture", "target_ge_0_5_capture_ratio", ""),
    )
    current_capture = dict(result.get("current_capture_audit") or {})
    adapted_capture = dict(result.get("adapted_capture_audit") or {})
    capture_baseline = dict(current_capture.get("baseline") or {})
    capture_sort_only = dict(current_capture.get("score_sort") or {})
    capture_adapted = dict(adapted_capture.get("score_sort") or {})
    enriched = {
        "baseline": {**result["baseline"], **capture_baseline},
        "sort_only": {**result["sort_only"], **capture_sort_only},
        "adapted": {**result["adapted"], **capture_adapted},
    }
    return rows, enriched

def _fmt(value, unit=""):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    digits = 0 if unit == "" and abs(number) >= 1000 else 4 if "R²" in unit else 2
    return f"{number:.{digits}f}{unit}"

def _three_way_rows_for_keys(
    values: dict[str, dict[str, Any]],
    specs: tuple[tuple[str, str, str], ...],
) -> list[tuple[str, str, str, str]]:
    integer_keys = {"trade_count", "candidate_supply_gap_days"}

    def _value(scenario: str, key: str, unit: str) -> str:
        raw = values[scenario].get(key)
        if key not in integer_keys:
            return _fmt(raw, unit)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return "N/A"
        return "N/A" if not math.isfinite(number) else f"{number:.0f}{unit}"

    return [
        (
            label,
            _value("baseline", key, unit),
            _value("sort_only", key, unit),
            _value("adapted", key, unit),
        )
        for label, key, unit in specs
    ]

def _render_compact_three_way_console(result: dict[str, Any]) -> str:
    _rows, values = _metric_rows(result)
    portfolio_specs = (
        ("淨總報酬", "total_return_pct", "%"),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("報酬／最大回撤", "return_over_max_drawdown", ""),
        ("年化報酬", "annual_return_pct", "%"),
        ("Log R²", "log_r_squared", ""),
        ("月勝率", "monthly_win_rate_pct", "%"),
    )
    trade_specs = (
        ("交易數", "trade_count", ""),
        ("勝率", "win_rate_pct", "%"),
        ("Payoff", "payoff_ratio", ""),
        ("EV", "expected_value_r", " R"),
    )
    capital_specs = (
        ("平均曝險", "avg_exposure_pct", "%"),
        ("平均候選供給", "avg_orderable_candidates", ""),
        ("候選不足日", "candidate_supply_gap_days", " 日"),
        ("平均預留金額", "avg_reserved_total", ""),
        ("平均投入金額", "avg_invested_total", ""),
        ("投入／預留比", "avg_invested_vs_reserved_pct", "%"),
        ("初始停損距離", "avg_stop_distance_pct", "%"),
    )
    target_specs = (
        ("平均 Realized R", "avg_realized_r", " R"),
        ("平均 Target R", "avg_target_r", " R"),
        ("Aggregate capture", "aggregate_target_capture_ratio", ""),
        ("Median capture", "median_target_capture_ratio", ""),
        ("Target ≥ 0.5R capture", "target_ge_0_5_capture_ratio", ""),
    )
    yearly_rows = [
        (
            str(int(row["year"])),
            _fmt(row.get("no_filter_return_pct"), "%"),
            _fmt(row.get("score_ranking_return_pct"), "%"),
            _fmt(row.get("adapted_return_pct"), "%"),
        )
        for row in list(result.get("yearly") or [])
    ]
    diagnostics = result["selection_diagnostics"]
    diag_rows = [
        (
            label,
            _fmt(diagnostics["baseline"].get(key)),
            _fmt(diagnostics["sort_only"].get(key)),
            _fmt(diagnostics["adapted"].get(key)),
        )
        for label, key in (
            ("Score coverage", "orderable_score_coverage_rate"),
            ("Target percentile", "selected_target_percentile_mean"),
            ("Top-k retention", "target_top_k_retention_mean"),
            ("Opportunity gap", "target_opportunity_gap_r_mean"),
            ("Selected Target R", "selected_target_mean_r"),
        )
    ]
    return "\n".join((
        render_title("策略參數適應績效摘要"),
        render_key_values((
            ("狀態", ADAPTATION_STATUS),
            ("比較", "Baseline vs Sort Only vs Adapted"),
            ("判讀", "Selection內擬合診斷；不是泛化結果"),
        )),
        render_section("投組報酬與風險", number=1),
        "讀法：先比較總報酬、回撤與報酬／回撤，再看成長穩定性。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            _three_way_rows_for_keys(values, portfolio_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("單筆交易品質", number=2),
        "讀法：EV與Payoff描述單筆品質，不代表資本已充分投入。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            _three_way_rows_for_keys(values, trade_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("資金配置與持倉", number=3),
        "讀法：持倉格數、停損距離與實際投入金額必須一起判讀。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            _three_way_rows_for_keys(values, capital_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("Target 與實際交易轉換", number=4),
        "讀法：Target是事後機會，Realized R是實際結果，capture衡量轉換效率。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            _three_way_rows_for_keys(values, target_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("年度報酬", number=5),
        render_table(
            ("年度", "Baseline", "Sort Only", "Adapted"),
            yearly_rows,
            alignments=("left", "right", "right", "right"),
        ) if yearly_rows else "無年度資料。",
        render_section("模型選股方向", number=6),
        "讀法：Future Target僅在回放後加入，不參與runtime或optimizer。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            diag_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("參數差異", number=7),
        render_table(
            ("參數", "歷史最小", "歷史中位", "歷史最大", "Adapted"),
            _param_comparison_table_rows(result),
            alignments=("left", "right", "right", "right", "right"),
        ),
        render_section("判讀限制", number=8),
        "Adapted只代表Selection內擬合結果；必須凍結後執行正式OOS，才可決定採用或拒絕。",
    ))

def _render_three_way_console(result: dict[str, Any]) -> str:
    if compact_console_enabled():
        return _render_compact_three_way_console(result)
    rows, values = _metric_rows(result)
    metric_rows = []
    for label, key, unit in rows:
        metric_rows.append((
            label,
            _fmt(values["baseline"].get(key), unit),
            _fmt(values["sort_only"].get(key), unit),
            _fmt(values["adapted"].get(key), unit),
        ))
    yearly_rows = []
    for row in list(result.get("yearly") or []):
        yearly_rows.append((
            str(int(row["year"])),
            _fmt(row.get("no_filter_return_pct"), "%"),
            _fmt(row.get("score_ranking_return_pct"), "%"),
            _fmt(row.get("adapted_return_pct"), "%"),
        ))
    diagnostics = result["selection_diagnostics"]
    diag_rows = []
    for label, key in (
        ("Orderable occurrences", "orderable_occurrences"),
        ("Score coverage", "orderable_score_coverage_rate"),
        ("Selected buys", "selected_buy_rows"),
        ("Target percentile", "selected_target_percentile_mean"),
        ("Top-k retention", "target_top_k_retention_mean"),
        ("Opportunity gap", "target_opportunity_gap_r_mean"),
        ("Selected Target R", "selected_target_mean_r"),
    ):
        diag_rows.append((
            label,
            _fmt(diagnostics["baseline"].get(key)),
            _fmt(diagnostics["sort_only"].get(key)),
            _fmt(diagnostics["adapted"].get(key)),
        ))
    return "\n".join((
        render_title("Selection Score-ranking Strategy Parameter Adaptation"),
        render_key_values((
            ("狀態", ADAPTATION_STATUS),
            ("判讀", "Selection內擬合診斷；不是泛化結果"),
            ("Future Target runtime", "未使用"),
        )),
        render_section("三組策略結果", number=1),
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            metric_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("年度績效", number=2),
        render_table(
            ("年度", "Baseline", "Sort Only", "Adapted"),
            yearly_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("Selection 選股診斷", number=3),
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted"),
            diag_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("參數差異", number=4),
        render_table(
            ("參數", "歷史最小", "歷史中位", "歷史最大", "Adapted"),
            _param_comparison_table_rows(result),
            alignments=("left", "right", "right", "right", "right"),
        ),
        "Adapted結果必須待凍結後正式OOS驗證，才可決定採用或拒絕。",
    ))

def _render_three_way_markdown(result: dict[str, Any]) -> str:
    rows, values = _metric_rows(result)
    lines = [
        "# Selection Score-ranking Strategy Parameter Adaptation",
        "",
        f"- 狀態：`{ADAPTATION_STATUS}`",
        "- Adapted為Selection內擬合診斷，不是泛化結果。",
        "- Future Target只於回放後離線join，未進入runtime或optimizer objective。",
        "",
        "## 三組策略結果",
        "",
        "| 指標 | Baseline | Sort Only | Adapted |",
        "|---|---:|---:|---:|",
    ]
    for label, key, unit in rows:
        lines.append(
            f"| {label} | {_fmt(values['baseline'].get(key), unit)} | "
            f"{_fmt(values['sort_only'].get(key), unit)} | "
            f"{_fmt(values['adapted'].get(key), unit)} |"
        )
    lines.extend((
        "",
        "## 年度績效",
        "",
        "| 年度 | Baseline | Sort Only | Adapted |",
        "|---:|---:|---:|---:|",
    ))
    for row in list(result.get("yearly") or []):
        lines.append(
            f"| {int(row['year'])} | {_fmt(row.get('no_filter_return_pct'), '%')} | "
            f"{_fmt(row.get('score_ranking_return_pct'), '%')} | "
            f"{_fmt(row.get('adapted_return_pct'), '%')} |"
        )
    diagnostics = result["selection_diagnostics"]
    lines.extend((
        "",
        "## Selection 選股診斷",
        "",
        "| 指標 | Baseline | Sort Only | Adapted |",
        "|---|---:|---:|---:|",
    ))
    for label, key in (
        ("Orderable occurrences", "orderable_occurrences"),
        ("Score coverage", "orderable_score_coverage_rate"),
        ("Selected buys", "selected_buy_rows"),
        ("Target percentile", "selected_target_percentile_mean"),
        ("Top-k retention", "target_top_k_retention_mean"),
        ("Opportunity gap", "target_opportunity_gap_r_mean"),
        ("Selected Target R", "selected_target_mean_r"),
    ):
        lines.append(
            f"| {label} | {_fmt(diagnostics['baseline'].get(key))} | "
            f"{_fmt(diagnostics['sort_only'].get(key))} | "
            f"{_fmt(diagnostics['adapted'].get(key))} |"
        )
    lines.extend((
        "",
        "## 參數差異",
        "",
        "| 參數 | 歷史最小 | 歷史中位 | 歷史最大 | Adapted |",
        "|---|---:|---:|---:|---:|",
    ))
    for parameter, minimum, median, maximum, adapted_value in _param_comparison_table_rows(result):
        lines.append(
            f"| {parameter} | {minimum} | {median} | {maximum} | {adapted_value} |"
        )
    lines.extend((
        "",
        "## 判讀限制",
        "",
        "- 本結果只回答策略參數能否在Selection內適應Score Sort後的候選分布。",
        "- 下一步必須凍結模型、Score、排序規則、search space與Adapted params，再執行正式OOS。",
        "",
    ))
    return "\n".join(lines)


def run_adaptation(*, project_root=PROJECT_ROOT, argv=None) -> dict[str, Any]:
    args = _parse_args(argv)
    root = Path(project_root).resolve()
    settings = get_breakout_quality_workflow_settings()
    _validate_fixed_contract(args, settings)
    configure_optuna_logging()

    pit_contract = load_selection_point_in_time_ranking_contract(
        str(root), args.filter_id, args.model_architecture, args.experiment_profile
    )
    _validate_pit_contract_against_settings(pit_contract, settings)
    walk_forward_policy = _build_walk_forward_policy(pit_contract)
    runtime_contract = _build_runtime_contract(
        root=root,
        args=args,
        settings=settings,
        pit_contract=pit_contract,
        walk_forward_policy=walk_forward_policy,
    )
    output_dir = (root / ADAPTATION_RELATIVE_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_pair_dir = output_dir / "baseline_sort_only"

    current_payload, current_pair_reused = _load_or_run_current_pair(
        root=root,
        args=args,
        runtime_contract=runtime_contract,
        output_dir=current_pair_dir,
    )
    current_decision = dict(current_payload.get("score_ranking_capture_audit", {})).get(
        "decision", {}
    )
    if str(current_decision.get("status") or "") != "ADAPTATION_DIAGNOSTIC_SUPPORTED":
        raise RuntimeError(
            "目前Baseline／Sort Only capture audit未通過參數適應前置條件："
            f"status={current_decision.get('status')!r}"
        )

    fixed_overrides = {
        "use_breakout_quality_filter": False,
        "use_breakout_quality_ranking": True,
        "breakout_quality_filter_id": str(args.filter_id),
        "fixed_risk": float(args.fixed_risk),
        "max_position_cap_pct": float(args.max_position_cap_pct),
    }
    ranking_context_factory = lambda: breakout_quality_ranking_source_context(
        score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
    )
    session = build_optimizer_session(
        walk_forward_policy=walk_forward_policy,
        output_dir=str(output_dir / "optimizer_runtime"),
        fixed_strategy_param_overrides=fixed_overrides,
        runtime_context_factory=ranking_context_factory,
        runtime_cache_identity=runtime_contract["runtime_identity_sha256"],
        optimizer_fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
        train_max_positions=args.max_positions,
        train_enable_rotation=str(args.rotation) == "on",
    )
    session.n_trials = int(args.trials)
    session.disable_milestone_dashboard = True

    db_path = output_dir / "adaptation_study.db"
    study = create_optimizer_study(
        f"sqlite:///{db_path.as_posix()}",
        seed=int(settings.seed),
        sampler_kind="tpe",
    )
    _assert_study_identity(study, runtime_contract)
    session.load_raw_data(
        get_dataset_dir(str(root), args.dataset),
        load_all_raw_data=load_all_raw_data,
        required_min_rows=get_breakout_optimizer_required_min_rows(),
        verbose=not args.quiet,
    )
    session.profile_recorder.init_output_files()
    session.profile_recorder.mark_run_started()
    existing_trial_count = int(len(study.trials))
    remaining_trials = max(0, int(args.trials) - existing_trial_count)
    try:
        if remaining_trials > 0:
            study.optimize(
                session.objective,
                n_trials=remaining_trials,
                n_jobs=resolve_optimizer_single_fold_search_parallel_trials(
                    None, sampler_kind="tpe"
                ),
                callbacks=[session.monitoring_callback],
            )
        adapted_params_payload, optimizer_summary = _select_adapted_params(
            study=study, session=session
        )
    finally:
        session.close_trial_prep_executor(force_shared=True)

    adapted_params_path = output_dir / "adapted_best_params.json"
    adapted_manifest_path = output_dir / "adapted_manifest.json"
    optimizer_summary_path = output_dir / "optimizer_summary.json"
    _write_json(adapted_params_path, adapted_params_payload)
    _write_json(
        optimizer_summary_path,
        {
            **optimizer_summary,
            "requested_trials": int(args.trials),
            "existing_trials_before_run": existing_trial_count,
            "trials_started_this_run": remaining_trials,
            "study_trial_count_after_run": int(len(study.trials)),
            "baseline_sort_only_reused": bool(current_pair_reused),
            "seed": int(settings.seed),
            "fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
            "runtime_identity_sha256": runtime_contract["runtime_identity_sha256"],
        },
    )
    comparison_result = _run_three_way_comparison(
        root=root,
        args=args,
        pit_contract=pit_contract,
        current_payload=current_payload,
        adapted_params_payload=adapted_params_payload,
        current_pair_dir=current_pair_dir,
        output_dir=output_dir,
    )
    artifact_paths = {
        "adapted_params": adapted_params_path,
        "optimizer_summary": optimizer_summary_path,
        "comparison_json": output_dir / "strategy_adaptation_comparison.json",
        "comparison_markdown": output_dir / "strategy_adaptation_comparison.md",
        "yearly_csv": output_dir / "strategy_adaptation_yearly_returns.csv",
        "adapted_capture_json": output_dir / "score_ranking_capture_audit.json",
        "adapted_capture_markdown": output_dir / "score_ranking_capture_audit.md",
        "adapted_equity": output_dir / "adapted_equity.csv",
        "adapted_trades": output_dir / "adapted_trades.csv",
        "adapted_daily_capacity": output_dir / "adapted_daily_capacity.csv",
        "adapted_orderable_diagnostics": output_dir / "adapted_orderable_target_diagnostics.csv",
        "adapted_selected_diagnostics": output_dir / "adapted_selected_target_diagnostics.csv",
        "adapted_yearly_returns": output_dir / "adapted_yearly_returns.csv",
        "adaptation_yearly_comparison": output_dir / "strategy_adaptation_yearly_returns.csv",
        "adapted_baseline_capture_lifecycle": output_dir / "no_filter_capture_lifecycle.csv",
        "adapted_score_capture_lifecycle": output_dir / "score_ranking_capture_lifecycle.csv",
        "adapted_capture_yearly": output_dir / "score_ranking_capture_yearly.csv",
        "adapted_capture_scenarios": output_dir / "score_ranking_capture_scenarios.csv",
        "baseline_sort_only_comparison_json": current_pair_dir / "strategy_comparison.json",
        "baseline_sort_only_comparison_markdown": current_pair_dir / "strategy_comparison.md",
        "baseline_sort_only_capture_json": current_pair_dir / "score_ranking_capture_audit.json",
        "baseline_sort_only_capture_markdown": current_pair_dir / "score_ranking_capture_audit.md",
        "baseline_sort_only_manifest": current_pair_dir / CURRENT_PAIR_MANIFEST_FILENAME,
    }
    execution_argv = _canonical_execution_argv(args)
    manifest = {
        **runtime_contract,
        "status": "ADAPTATION_FIT_COMPLETED",
        "result_interpretation": ADAPTATION_STATUS,
        "selected_params": adapted_params_payload,
        "optimizer_summary": optimizer_summary,
        "baseline_sort_only_reused": bool(current_pair_reused),
        "execution_command": " ".join(execution_argv),
        "execution_argv": execution_argv,
        "created_at": get_taipei_now().isoformat(),
        "artifacts": {
            key: {
                "path": project_relative_display_path(path, project_root=root),
                "sha256": compute_file_sha256(path),
            }
            for key, path in artifact_paths.items()
        },
    }
    _write_json(adapted_manifest_path, manifest)
    print_artifact_paths(
        (
            ("Adapted params", adapted_params_path),
            ("Adapted manifest", adapted_manifest_path),
            ("Optimizer summary", optimizer_summary_path),
            ("三組比較 Markdown", output_dir / "strategy_adaptation_comparison.md"),
            ("三組比較 JSON", output_dir / "strategy_adaptation_comparison.json"),
        ),
        project_root=root,
    )
    return comparison_result

def main(argv=None):
    run_adaptation(argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
