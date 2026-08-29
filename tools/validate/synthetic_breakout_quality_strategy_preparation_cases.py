"""Generic Strategy Compare preparation contract checks.

This module owns preparation-plan/builder boundary checks that were previously
embedded in the config-driven application validator.
"""

from __future__ import annotations

from dataclasses import replace
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import tempfile

from core.strategy_comparison import StrategyPreparationAction, StrategyPreparationPlan
from .synthetic_breakout_quality_support import add_check


def append_strategy_compare_preparation_contract_checks(
    *,
    results: list[dict[str, Any]],
    case_id: str,
    project_root: Path,
    settings: Any,
    preparation_source: str,
    orchestration_source: str,
) -> None:
    shared_orchestrator_source = (
        project_root / "services" / "research" / "artifact_orchestrator.py"
    ).read_text(encoding="utf-8")
    research_contract_source = (
        project_root / "core" / "research_orchestration.py"
    ).read_text(encoding="utf-8")
    min_roos_source = settings.parameter_sources["min_roos"]

    from core.research_orchestration import resolve_research_artifact_action
    policy_cases = {
        "REUSE": dict(ready=True, artifact_exists=True, has_builder=True, resumable=False),
        "BUILD": dict(ready=False, artifact_exists=False, has_builder=True, resumable=False),
        "REBUILD": dict(ready=False, artifact_exists=True, has_builder=True, resumable=False),
        "RESUME": dict(ready=False, artifact_exists=True, has_builder=True, resumable=True),
        "BLOCKED": dict(ready=False, artifact_exists=False, has_builder=False, resumable=False),
    }
    resolved_policy_cases = {
        expected: resolve_research_artifact_action(
            **facts,
            auto_prepare=True,
            reuse_ready_artifacts=True,
            rebuild_stale_artifacts=True,
            resume_partial_artifacts=True,
        )
        for expected, facts in policy_cases.items()
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "research_artifact_policy_unifies_reuse_build_rebuild_resume_blocked",
        tuple(policy_cases),
        tuple(
            expected
            for expected in policy_cases
            if resolved_policy_cases[expected] == expected
        ),
    )

    from services.research.artifact_orchestrator import run_research_artifact_preparation
    dataset_build = StrategyPreparationAction(
        action_id="model-upstream:dataset_core",
        artifact_key="model-upstream:dataset_core",
        action="BUILD",
        builder_type="dataset",
        description="dataset build",
        path="outputs/dataset_summary.json",
        execution_priority=10,
    )
    param_waiting = StrategyPreparationAction(
        action_id="param:full_oos",
        artifact_key="param:full_oos",
        action="BUILD",
        builder_type="optimizer",
        description="param build",
        path="models/strategy_params/canonical/full.json",
        dependencies=("model-upstream:dataset_core",),
        execution_priority=20,
    )
    dataset_reuse = replace(dataset_build, action="REUSE", builder_type=None)
    param_reuse = replace(param_waiting, action="REUSE", builder_type=None)
    dependency_plans = iter((
        StrategyPreparationPlan.from_actions((dataset_build, param_waiting)),
        StrategyPreparationPlan.from_actions((dataset_reuse, param_waiting)),
        StrategyPreparationPlan.from_actions((dataset_reuse, param_reuse)),
    ))
    dependency_calls: list[str] = []
    with redirect_stdout(io.StringIO()):
        dependency_outcome = run_research_artifact_preparation(
            plan_refresher=dependency_plans.__next__,
            action_executor=lambda action: dependency_calls.append(action.artifact_key),
            failure_prefix="synthetic research graph",
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "research_orchestrator_replans_after_each_producer_and_orders_dependencies",
        True,
        dependency_calls == ["model-upstream:dataset_core", "param:full_oos"]
        and dependency_outcome.plan.overall_status == "READY"
        and dependency_outcome.waves == 2,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "min_roos_uses_forward_p2_artifact_and_has_auto_builder",
        True,
        "binary_dl_filter_param_adaptation/risk_only_rolling/p2_dl_off_trained"
        in str(min_roos_source.path_template)
        and "trade_path_label/a2_teacher_params" not in str(min_roos_source.path_template)
        and min_roos_source.identity_manifest_path is not None
        and min_roos_source.builder is not None
        and str(min_roos_source.builder.options.get("parameter_set")) == "p2",
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_supports_dependency_waves_after_score_period_becomes_known",
        True,
        "plan_refresher" in shared_orchestrator_source
        and "Mandatory re-plan boundary after every producer invocation" in shared_orchestrator_source
        and "resolve_research_artifact_action" in research_contract_source,
    )

    from config.strategy_compare import get_strategy_comparison_settings as _get_pit_bundle_settings
    from filters.breakout_quality.paths import (
        resolve_filter_model_output_dir,
        resolve_selection_point_in_time_audit_json_path,
        resolve_selection_point_in_time_score_path,
    )
    from services.research.strategy_compare_execution import selection_pit_mode_paths
    from filters.breakout_quality.strategy_compare_pit_contract import (
        resolve_strategy_compare_selection_pit_bundle_dir,
        resolve_strategy_compare_selection_pit_contract_override,
    )
    pit_bundle_settings = _get_pit_bundle_settings("extending_window_rolling")
    default_pit_source = next(
        source
        for source in pit_bundle_settings.dl_sources.values()
        if str(source.score_source) == "selection_point_in_time"
        and source.point_in_time_dirname in (None, "")
    )
    expected_default_pit_dir = resolve_selection_point_in_time_score_path(
        project_root,
        default_pit_source.filter_id,
        default_pit_source.model_architecture,
        default_pit_source.experiment_profile,
    ).parent.resolve()
    expected_default_pit_audit = resolve_selection_point_in_time_audit_json_path(
        project_root,
        default_pit_source.filter_id,
        default_pit_source.model_architecture,
        default_pit_source.experiment_profile,
    ).resolve()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_default_rolling_pit_keeps_model_bundle_and_output_audit_split",
        (
            expected_default_pit_dir,
            True,
            None,
        ),
        (
            resolve_strategy_compare_selection_pit_bundle_dir(
                root=project_root,
                source=default_pit_source,
            ),
            "outputs/filters/breakout_quality" in expected_default_pit_audit.as_posix(),
            resolve_strategy_compare_selection_pit_contract_override(
                root=project_root,
                source=default_pit_source,
            ),
        ),
    )
    training_contract_source = (
        project_root / "services" / "research" / "strategy_compare_training.py"
    ).read_text(encoding="utf-8")
    research_application_source = (
        project_root / "services" / "research" / "breakout_quality_application.py"
    ).read_text(encoding="utf-8")
    robustness_source = (
        project_root / "services" / "research" / "strategy_multi_seed_robustness.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_training_lifecycle_preserves_canonical_none_vs_isolated_pit_override",
        True,
        "point_in_time_dir_override=pit_dir_override" in research_application_source
        and "point_in_time_dir_override=point_in_time_dir_override" in training_contract_source
        and "if point_in_time_dir_override in (None, \"\")" in training_contract_source
        and "point_in_time_dir_override=(model_dir if is_selection_pit else None)"
        in robustness_source,
    )

    oos_pit_settings = _get_pit_bundle_settings("extending_window_oos")
    oos_pit_source = next(
        source
        for source in oos_pit_settings.dl_sources.values()
        if str(source.score_source) == "selection_point_in_time"
        and source.point_in_time_dirname not in (None, "")
    )
    expected_oos_override = (
        resolve_filter_model_output_dir(
            project_root,
            oos_pit_source.filter_id,
            oos_pit_source.model_architecture,
            oos_pit_source.experiment_profile,
        )
        / str(oos_pit_source.point_in_time_dirname)
    ).resolve()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_mode_specific_oos_pit_keeps_colocated_override",
        expected_oos_override,
        resolve_strategy_compare_selection_pit_contract_override(
            root=project_root,
            source=oos_pit_source,
        ),
    )

    default_mode_paths = selection_pit_mode_paths(
        default_pit_source,
        project_root=project_root,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_default_pit_execution_pins_models_score_and_manifest_paths",
        (
            expected_default_pit_dir / "selection_point_in_time_scores.csv",
            expected_default_pit_dir / "selection_point_in_time_manifest.json",
        ),
        (default_mode_paths["score"], default_mode_paths["manifest"]),
    )

    from services.research import strategy_compare_execution as execution_module
    captured_replay_kwargs: list[dict[str, Any]] = []
    with patch.object(
        execution_module,
        "run_comparison",
        side_effect=lambda **kwargs: captured_replay_kwargs.append(dict(kwargs)) or {"ok": True},
    ):
        for arm_id in ("C59", "C60", "C64"):
            execution_module.run_strategy_compare_active_arm(
                settings=pit_bundle_settings,
                arm=pit_bundle_settings.arms[arm_id],
                project_root=project_root,
                params_path="models/strategy_params/canonical/synthetic.json",
                output_dir="outputs/strategy_compare/synthetic",
                comparison_start="2021-01-01",
                comparison_end="2026-03-02",
                param_evaluation_mode="rolling",
            )
    c59_kwargs, c60_kwargs, c64_kwargs = captured_replay_kwargs
    c59_primary = Path(str(c59_kwargs.get("selection_pit_score_path_override") or ""))
    c60_primary = Path(str(c60_kwargs.get("selection_pit_score_path_override") or ""))
    c60_safety = Path(str((c60_kwargs.get("ranking_options") or {}).get("safety_score_path_override") or ""))
    c64_primary = Path(str(c64_kwargs.get("selection_pit_score_path_override") or ""))
    c64_options = dict(c64_kwargs.get("ranking_options") or {})
    c64_safety = Path(str(c64_options.get("safety_score_path_override") or ""))
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "current_c59_c60_execution_consumes_models_score_manifest_bundle",
        True,
        all(
            path.name == "selection_point_in_time_scores.csv"
            and "models/filters/breakout_quality" in path.as_posix()
            and "outputs/" not in path.as_posix()
            for path in (c59_primary, c60_primary, c60_safety, c64_primary, c64_safety)
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "c64_execution_reuses_same_dual_head_pit_artifact_and_selects_conditional_safety_column",
        True,
        c64_primary == c64_safety
        and c64_options.get("safety_score_column") == "conditional_safety_score"
        and not c64_options.get("safety_residualization"),
    )

    from services.research import strategy_compare_preparation as preparation_module
    score_action = StrategyPreparationAction(
        action_id="dl:TP1:forward_scores",
        artifact_key="dl:TP1:forward_scores",
        action="BUILD",
        builder_type="forward_oos_scores",
        description="build scores",
        path="models/scores.csv",
    )
    p2_action = StrategyPreparationAction(
        action_id="param:min_roos",
        artifact_key="param:min_roos",
        action="BUILD",
        builder_type="binary_dl_min_roos_rolling",
        description="build p2",
        path="models/p2.json",
    )
    initial_wave_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="PREPARABLE", actions=(score_action,)
        ),
    }
    second_wave_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="PREPARABLE", actions=(p2_action,)
        ),
    }
    final_wave_status = {
        "comparison_ready": True,
        "overall_status": "READY",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="READY", actions=tuple()
        ),
    }
    with patch.object(
        preparation_module,
        "collect_preparation_status",
        side_effect=(initial_wave_status, second_wave_status, final_wave_status),
    ), patch.object(
        preparation_module,
        "_execute_preparation_action",
    ) as mocked_prepare_action, redirect_stdout(io.StringIO()):
        multi_wave_result = preparation_module.prepare_strategy_comparison_artifacts(
            project_root=project_root,
            settings=settings,
            status=initial_wave_status,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_executes_newly_revealed_dependencies_in_later_wave",
        True,
        multi_wave_result["comparison_ready"]
        and mocked_prepare_action.call_count == 2
        and [
            call.kwargs["action"].artifact_key
            for call in mocked_prepare_action.call_args_list
        ] == ["dl:TP1:forward_scores", "param:min_roos"],
    )

    repeated_model_score = StrategyPreparationAction(
        action_id="dl:CONT13E_ROLL:forward_scores",
        artifact_key="dl:CONT13E_ROLL:forward_scores",
        action="RESUME",
        builder_type="canonical_model_artifacts",
        description="synthetic model resume",
        path="models/filters/breakout_quality/selection_point_in_time_scores.csv",
        producer_work_type="model_training",
    )
    repeated_model_audit = StrategyPreparationAction(
        action_id="dl:CONT13E_ROLL:audit",
        artifact_key="dl:CONT13E_ROLL:audit",
        action="RESUME",
        builder_type="canonical_model_artifacts",
        description="synthetic audit resume",
        path="models/filters/breakout_quality/selection_point_in_time_audit.json",
        producer_work_type="model_training",
    )
    repeated_model_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "dl_sources": {
            "CONT13E_ROLL": {
                "status": "SELECTION_PIT_INVALID (ValueError)",
                "validation_error": "synthetic stale manifest",
            }
        },
        "preparation_plan": StrategyPreparationPlan.from_actions((
            repeated_model_score, repeated_model_audit
        )),
    }
    repeated_external_calls: list[str] = []
    repeated_error = ""
    try:
        with redirect_stdout(io.StringIO()):
            preparation_module.prepare_strategy_comparison_artifacts(
                project_root=project_root,
                settings=settings,
                status=repeated_model_status,
                status_refresher=iter((repeated_model_status, repeated_model_status)).__next__,
                producer_handlers={
                    "model_training": lambda action: repeated_external_calls.append(
                        action.artifact_key
                    ) or 0
                },
            )
    except RuntimeError as exc:
        repeated_error = str(exc)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "external_model_producer_never_repeats_same_scope_after_unsuccessful_replan",
        True,
        len(repeated_external_calls) == 1
        and "已停止再次執行" in repeated_error
        and "synthetic stale manifest" in repeated_error,
    )

    # Regression for the actual cross-producer failure: the rendering snapshot may have
    # comparison_period=None before Dataset/Target preparation.  Parameter execution must
    # consume the freshly re-planned period, never that stale snapshot.
    from config.strategy_compare import get_strategy_comparison_settings as _get_compare_settings
    fresh_period_settings = _get_compare_settings("extending_window_oos")
    fresh_param_action = StrategyPreparationAction(
        action_id="param:full_oos",
        artifact_key="param:full_oos",
        action="BUILD",
        builder_type="canonical_optimizer_strategy_params",
        description="synthetic canonical param build",
        path="models/strategy_params/canonical/full_base_best.json",
        producer_work_type="optimizer_strategy_parameter_service",
    )
    stale_param_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "comparison_period": None,
        "preparation_plan": StrategyPreparationPlan.from_actions((fresh_param_action,)),
    }
    fresh_param_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "preparation_plan": StrategyPreparationPlan.from_actions((fresh_param_action,)),
    }
    final_param_status = {
        "comparison_ready": True,
        "overall_status": "READY",
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "preparation_plan": StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="param:full_oos",
                artifact_key="param:full_oos",
                action="REUSE",
                builder_type=None,
                description="synthetic canonical param reuse",
                path="models/strategy_params/canonical/full_base_best.json",
                producer_work_type="existing_artifact",
            ),
        )),
    }
    forwarded_ensure: dict[str, Any] = {}
    with patch.object(
        preparation_module,
        "ensure_strategy_parameter_artifact",
        side_effect=lambda *args, **kwargs: forwarded_ensure.update(kwargs) or {},
    ), redirect_stdout(io.StringIO()):
        refreshed_param_result = preparation_module.prepare_strategy_parameter_artifacts(
            project_root=project_root,
            settings=fresh_period_settings,
            status=stale_param_status,
            required_source_ids=("full_oos",),
            status_refresher=iter((fresh_param_status, final_param_status)).__next__,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_producer_consumes_fresh_post_upstream_comparison_period",
        True,
        refreshed_param_result["comparison_ready"]
        and forwarded_ensure.get("comparison_end_date") == "2026-03-02",
    )

    from config.strategy_compare import get_strategy_comparison_settings
    from core.active_param_ensemble import (
        ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
        get_active_param_ensemble_date_range,
    )
    from services.research.strategy_compare_preparation_status import (
        _validate_param_artifact,
        _validate_param_training_identity,
        resolve_param_source_path,
    )
    from services.optimizer.strategy_param_training import (
        FULL_ROOS_SEARCH_FIELDS,
        MIN_ROOS_SEARCH_FIELDS,
        prepare_extending_full_roos_params,
        prepare_extending_min_roos_params,
        prepare_oos_frozen_roos_params,
    )

    extending_settings = get_strategy_comparison_settings("extending_window_rolling")
    extending_source = extending_settings.parameter_sources["extending_min_roos"]
    extending_options = dict(extending_source.builder.options)
    extending_full_source = extending_settings.parameter_sources["extending_full_roos"]
    extending_full_options = dict(extending_full_source.builder.options)
    param_filename = "roos_base_best.json"

    def _render_param_path(raw: str) -> str:
        return str(raw).format(param_filename=param_filename)

    def _synthetic_min_roos_payload(first_year: int, last_year: int) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        simple_mapping: dict[str, Any] = {}
        folds: list[dict[str, Any]] = []
        params = {
            "high_len": 60,
            "atr_len": 14,
            "atr_buy_tol": 1.5,
            "atr_times_init": 2.0,
            "atr_times_trail": 3.0,
        }
        for year in range(first_year, last_year + 1):
            effective_start = f"{year}-01-01"
            effective_end = f"{year}-12-31"
            year_params = dict(params)
            year_params["high_len"] = year
            mapping[effective_start] = [{"params": year_params}]
            simple_mapping[effective_start] = dict(year_params)
            folds.append({
                "effective_start": effective_start,
                "effective_end": effective_end,
                "oos_start_date": effective_start,
                "oos_end_date": effective_end,
            })
        return {
            "schema_type": ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
            "schema_version": 1,
            "mode": "rolling",
            "selector": "base_finalist_best",
            "random_seed_ensemble": {"seed_count": 1, "min_agree": 1},
            "meta": {
                "window_mode": "fixed",
                "first_oos_date": f"{first_year}-01-01",
                "last_oos_date": f"{last_year}-12-31",
                "train_window_months": 120,
                "oos_horizon_months": 12,
            },
            "summary": {
                "folds": last_year - first_year + 1,
                "oos_period": f"{first_year}-01-01~{last_year}-12-31",
            },
            "params_ensemble_by_effective_date": mapping,
            "params_by_effective_date": simple_mapping,
            "folds": folds,
            "breakout_quality_param_adaptation": {
                "mode": "min_roos_training",
                "parameter_set": "P2",
                "search_fields": list(MIN_ROOS_SEARCH_FIELDS),
                "fixed_rule_contract": "all_rule_filters_off",
                "training_dl_enabled": False,
            },
        }

    with tempfile.TemporaryDirectory() as extending_temp:
        extending_root = Path(extending_temp)
        historical_path = extending_root / _render_param_path(extending_options["historical_params_path"])
        current_path = extending_root / _render_param_path(extending_options["current_params_path"])
        historical_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.parent.mkdir(parents=True, exist_ok=True)
        historical_path.write_text(
            json.dumps(_synthetic_min_roos_payload(2014, 2020)), encoding="utf-8"
        )
        current_path.write_text(
            json.dumps(_synthetic_min_roos_payload(2021, 2026)), encoding="utf-8"
        )
        prepare_extending_min_roos_params(
            project_root=extending_root,
            param_policy=extending_settings.param_policy,
            historical_params_path=str(extending_options["historical_params_path"]),
            current_params_path=str(extending_options["current_params_path"]),
            output_relative_dir=str(extending_options["output_relative_dir"]),
            quiet=True,
        )
        extending_output = resolve_param_source_path(
            extending_root, extending_settings, "extending_min_roos"
        )
        extending_payload = json.loads(extending_output.read_text(encoding="utf-8"))
        extending_range = get_active_param_ensemble_date_range(extending_payload)
        extending_artifact_ready, extending_artifact_status, _policy = _validate_param_artifact(
            extending_output,
            param_policy=extending_settings.param_policy,
            comparison_start=extending_settings.start_date,
            comparison_end=extending_settings.end_date,
            artifact_contract=dict(extending_source.artifact_contract),
        )
        extending_identity_ready, extending_identity_status, _manifest_path = (
            _validate_param_training_identity(
                root=extending_root,
                settings=extending_settings,
                source_id="extending_min_roos",
            )
        )
        current_payload = json.loads(current_path.read_text(encoding="utf-8"))
        current_payload["meta"]["synthetic_source_change"] = True
        current_path.write_text(json.dumps(current_payload), encoding="utf-8")
        stale_identity_ready, stale_identity_status, _stale_manifest = (
            _validate_param_training_identity(
                root=extending_root,
                settings=extending_settings,
                source_id="extending_min_roos",
            )
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "extending_min_roos_stitch_preserves_true_coverage_and_self_validates",
        True,
        extending_range == ("2014-01-01", "2026-12-31")
        and extending_artifact_ready
        and extending_artifact_status == "READY"
        and extending_identity_ready
        and extending_identity_status == "READY"
        and not stale_identity_ready
        and stale_identity_status == "EXTENDING_STITCH_SOURCE_HASH_MISMATCH:current",
    )


    oos_settings = get_strategy_comparison_settings("extending_window_oos")
    oos_min_source = oos_settings.parameter_sources["oos_min_roos"]
    oos_min_options = dict(oos_min_source.builder.options)
    with tempfile.TemporaryDirectory() as oos_temp:
        oos_root = Path(oos_temp)
        historical_path = oos_root / _render_param_path(extending_options["historical_params_path"])
        current_path = oos_root / _render_param_path(extending_options["current_params_path"])
        historical_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.parent.mkdir(parents=True, exist_ok=True)
        historical_path.write_text(
            json.dumps(_synthetic_min_roos_payload(2014, 2020)), encoding="utf-8"
        )
        current_path.write_text(
            json.dumps(_synthetic_min_roos_payload(2021, 2026)), encoding="utf-8"
        )
        prepare_extending_min_roos_params(
            project_root=oos_root,
            param_policy=oos_settings.param_policy,
            historical_params_path=str(extending_options["historical_params_path"]),
            current_params_path=str(extending_options["current_params_path"]),
            output_relative_dir=str(extending_options["output_relative_dir"]),
            quiet=True,
        )
        oos_dependency_path = resolve_param_source_path(
            oos_root, oos_settings, "extending_min_roos"
        )
        prepare_oos_frozen_roos_params(
            project_root=oos_root,
            param_policy=oos_settings.param_policy,
            source_params_path=str(oos_dependency_path.relative_to(oos_root)),
            output_relative_dir=str(oos_min_options["output_relative_dir"]),
            freeze_effective_date=str(oos_min_options["freeze_effective_date"]),
            freeze_cutoff_date=str(oos_min_options["freeze_cutoff_date"]),
            display_name="Min ROOS",
            quiet=True,
        )
        oos_output = resolve_param_source_path(oos_root, oos_settings, "oos_min_roos")
        oos_payload = json.loads(oos_output.read_text(encoding="utf-8"))
        oos_mapping = dict(oos_payload.get("params_ensemble_by_effective_date") or {})
        oos_range = get_active_param_ensemble_date_range(oos_payload)
        oos_artifact_ready, oos_artifact_status, _oos_policy = _validate_param_artifact(
            oos_output,
            param_policy=oos_settings.param_policy,
            comparison_start="2021-01-01",
            comparison_end="2026-12-31",
            artifact_contract=dict(oos_min_source.artifact_contract),
        )
        oos_identity_ready, oos_identity_status, _oos_manifest = (
            _validate_param_training_identity(
                root=oos_root, settings=oos_settings, source_id="oos_min_roos"
            )
        )
        dependency_payload = json.loads(oos_dependency_path.read_text(encoding="utf-8"))
        dependency_payload["summary"]["synthetic_post_freeze_change"] = True
        oos_dependency_path.write_text(json.dumps(dependency_payload), encoding="utf-8")
        stale_oos_ready, stale_oos_status, _stale_oos_manifest = (
            _validate_param_training_identity(
                root=oos_root, settings=oos_settings, source_id="oos_min_roos"
            )
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "oos_param_freeze_uses_only_2021_effective_member_and_invalidates_on_source_change",
        True,
        oos_range == ("2021-01-01", "2026-12-31")
        and tuple(oos_mapping) == ("2021-01-01",)
        and oos_mapping["2021-01-01"][0]["params"]["high_len"] == 2021
        and "params_by_effective_date" not in oos_payload
        and "params_by_oos_year" not in oos_payload
        and oos_payload["breakout_quality_param_adaptation"]["freeze_cutoff_date"] == "2020-12-31"
        and oos_artifact_ready
        and oos_artifact_status == "READY"
        and oos_identity_ready
        and oos_identity_status == "READY"
        and not stale_oos_ready
        and stale_oos_status == "OOS_FREEZE_SOURCE_HASH_MISMATCH",
    )


    def _synthetic_full_roos_payload(first_year: int, last_year: int) -> dict[str, Any]:
        payload = _synthetic_min_roos_payload(first_year, last_year)
        payload["breakout_quality_param_adaptation"] = {
            "mode": "selection_full_roos_training",
            "parameter_set": "P4_HISTORY",
            "search_fields": list(FULL_ROOS_SEARCH_FIELDS),
            "training_dl_enabled": False,
        }
        return payload

    with tempfile.TemporaryDirectory() as full_temp:
        full_root = Path(full_temp)
        historical_full_path = full_root / _render_param_path(
            extending_full_options["historical_params_path"]
        )
        current_full_path = full_root / _render_param_path(
            extending_full_options["current_params_path"]
        )
        historical_full_path.parent.mkdir(parents=True, exist_ok=True)
        current_full_path.parent.mkdir(parents=True, exist_ok=True)
        historical_full_path.write_text(
            json.dumps(_synthetic_full_roos_payload(2014, 2020)), encoding="utf-8"
        )
        current_full_path.write_text(
            json.dumps(_synthetic_full_roos_payload(2021, 2026)), encoding="utf-8"
        )
        prepare_extending_full_roos_params(
            project_root=full_root,
            param_policy=extending_settings.param_policy,
            historical_params_path=str(extending_full_options["historical_params_path"]),
            current_params_path=str(extending_full_options["current_params_path"]),
            output_relative_dir=str(extending_full_options["output_relative_dir"]),
            quiet=True,
        )
        full_output = resolve_param_source_path(
            full_root, extending_settings, "extending_full_roos"
        )
        full_payload = json.loads(full_output.read_text(encoding="utf-8"))
        full_range = get_active_param_ensemble_date_range(full_payload)
        full_artifact_ready, full_artifact_status, _full_policy = _validate_param_artifact(
            full_output,
            param_policy=extending_settings.param_policy,
            comparison_start=extending_settings.start_date,
            comparison_end=extending_settings.end_date,
            artifact_contract=dict(extending_full_source.artifact_contract),
        )
        full_identity_ready, full_identity_status, _full_manifest = (
            _validate_param_training_identity(
                root=full_root,
                settings=extending_settings,
                source_id="extending_full_roos",
            )
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "extending_full_roos_stitch_restores_current_full_baseline_with_pit_safe_coverage",
        True,
        full_range == ("2014-01-01", "2026-12-31")
        and full_artifact_ready
        and full_artifact_status == "READY"
        and full_identity_ready
        and full_identity_status == "READY"
        and dict(full_payload.get("breakout_quality_param_adaptation") or {}).get("parameter_set")
            == "P4_EXTENDING",
    )

    dependency_reuse = StrategyPreparationAction(
        action_id="dataset:truth", artifact_key="dataset:truth",
        action="REUSE", builder_type=None, description="reuse truth", path="outputs/dataset.json",
        producer_work_type="existing_artifact",
    )
    dependency_build = StrategyPreparationAction(
        action_id="score:forward", artifact_key="score:forward",
        action="BUILD", builder_type="score_builder", description="build score", path="outputs/score.csv",
        dependencies=("dataset:truth",), producer_work_type="strategy_compare_deterministic_rebuild",
    )
    dependency_plan = StrategyPreparationPlan.from_actions(
        (dependency_build, dependency_reuse)
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_declares_dependency_and_producer_contract",
        True,
        dependency_plan.overall_status == "PREPARABLE"
        and dependency_plan.next_runnable_action() is dependency_build
        and dependency_build.as_dict()["dependencies"] == ["dataset:truth"]
        and dependency_build.as_dict()["producer_work_type"]
        == "strategy_compare_deterministic_rebuild",
    )

    dependency_cycle_rejected = False
    try:
        StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="a", artifact_key="a", action="BUILD", builder_type="x",
                description="a", path="a", dependencies=("b",),
            ),
            StrategyPreparationAction(
                action_id="b", artifact_key="b", action="BUILD", builder_type="x",
                description="b", path="b", dependencies=("a",),
            ),
        ))
    except ValueError:
        dependency_cycle_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_rejects_dependency_cycles",
        True,
        dependency_cycle_rejected,
    )

    unknown_dependency_rejected = False
    try:
        StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="score", artifact_key="score", action="BUILD", builder_type="x",
                description="score", path="score", dependencies=("missing:truth",),
            ),
        ))
    except ValueError:
        unknown_dependency_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_rejects_unknown_dependencies",
        True,
        unknown_dependency_rejected,
    )


    from config.strategy_compare import get_strategy_comparison_settings
    from services.optimizer import strategy_param_training as param_training_module

    selection_settings = get_strategy_comparison_settings("selection_pit")
    auto_settings = replace(selection_settings, start_date=None, end_date=None)
    selection_source = auto_settings.parameter_sources["selection_min_roos"]
    auto_action = StrategyPreparationAction(
        action_id="param:selection_min_roos",
        artifact_key="param:selection_min_roos",
        action="REBUILD",
        builder_type="selection_historical_p2",
        description="build selection historical params",
        path="models/selection_min.json",
    )
    forwarded_period: dict[str, Any] = {}
    with patch.object(
        preparation_module,
        "prepare_selection_historical_p2_params",
        side_effect=lambda **kwargs: forwarded_period.update(kwargs),
    ):
        preparation_module._execute_preparation_action(
            root=project_root, settings=auto_settings, action=auto_action
        )
    canonical_period = param_training_module._resolve_selection_historical_oos_period(
        forwarded_period.get("first_oos_date"),
        forwarded_period.get("last_oos_date"),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "auto_period_strategy_profile_keeps_selection_param_source_on_canonical_history",
        True,
        auto_settings.start_date is None
        and auto_settings.end_date is None
        and selection_source.builder is not None
        and forwarded_period.get("first_oos_date") is None
        and forwarded_period.get("last_oos_date") is None
        and canonical_period == ("2014-01-01", "2020-12-31"),
    )

    from filters.breakout_quality import strategy_compare_dl_artifacts as dl_artifacts_module

    pit_source = selection_settings.dl_sources["CONT13E_PIT"]
    with tempfile.TemporaryDirectory() as raw_temp:
        missing_actions: list[StrategyPreparationAction] = []
        missing_row, missing_ready = dl_artifacts_module._collect_selection_pit_source_status(
            root=Path(raw_temp),
            settings=auto_settings,
            dl_id="CONT13E_PIT",
            source=pit_source,
            source_upstream_dependencies=tuple(),
            runtime_required_dl_sources={"CONT13E_PIT"},
            artifact_identities={},
            actions=missing_actions,
            runtime_periods={},
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_missing_pit_is_auto_orchestrated_model_work_not_checkpoint_rebuild",
        True,
        not missing_ready
        and str(missing_row.get("status") or "").startswith("SELECTION_PIT_INVALID")
        and len(missing_actions) == 3
        and all(action.action == "BUILD" for action in missing_actions)
        and all(action.builder_type == "canonical_model_artifacts" for action in missing_actions)
        and all(action.producer_work_type == "model_training" for action in missing_actions)
        and all("canonical model-training producer" in action.description for action in missing_actions)
        and all("BUILD／REBUILD／RESUME" in action.description for action in missing_actions),
    )

    failed_gate_contract = SimpleNamespace(
        seed=42,
        model_validation_gate={"status": "FAIL", "checks": {"synthetic": False}},
        available_from="2016-04-01",
        available_through="2020-12-31",
        manifest={
            "fold_months": 12,
            "single_score_block": False,
            "score_period": {"start": "2014-01-01", "end": "2020-12-31"},
            "folds": [{"selected_epoch": 2}],
        },
        manifest_path=Path("models/synthetic/selection_point_in_time_manifest.json"),
        audit_path=Path("outputs/synthetic/selection_point_in_time_audit.json"),
        score_path=Path("models/synthetic/selection_point_in_time_scores.csv"),
    )
    failed_runtime_periods: dict[str, tuple[str, str]] = {}
    failed_actions = []
    with patch.object(
        dl_artifacts_module,
        "load_validated_selection_pit_strategy_compare_contract",
        return_value=failed_gate_contract,
    ) as failed_gate_loader:
        failed_row, failed_ready = dl_artifacts_module._collect_selection_pit_source_status(
            root=project_root,
            settings=auto_settings,
            dl_id="CONT13E_PIT",
            source=pit_source,
            source_upstream_dependencies=tuple(),
            runtime_required_dl_sources={"CONT13E_PIT"},
            artifact_identities={},
            actions=failed_actions,
            runtime_periods=failed_runtime_periods,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_failed_selection_pit_quality_gate_is_visible_warning_but_legal_artifact_can_replay",
        True,
        failed_ready
        and failed_row.get("status") == "READY_MODEL_GATE_FAIL"
        and (failed_row.get("model_validation_gate") or {}).get("status") == "FAIL"
        and failed_runtime_periods.get("CONT13E_PIT") == ("2016-04-01", "2020-12-31")
        and len(failed_actions) == 3
        and all(action.action == "REUSE" for action in failed_actions)
        and all(action.builder_type is None for action in failed_actions)
        and all("WARN: Model Gate=FAIL" in action.description for action in failed_actions)
        and all("仍允許既定strategy-conversion replay" in action.description for action in failed_actions)
        and failed_gate_loader.call_count == 1
        and failed_gate_loader.call_args.kwargs.get("seed") == 42
        and failed_gate_loader.call_args.kwargs.get("comparison_start") is None
        and failed_gate_loader.call_args.kwargs.get("comparison_end") is None,
    )

    partial_period_rejected = False
    try:
        param_training_module._resolve_selection_historical_oos_period(
            None, "2020-12-31"
        )
    except ValueError:
        partial_period_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_historical_period_requires_both_explicit_dates_or_both_auto",
        True,
        partial_period_rejected,
    )




__all__ = ["append_strategy_compare_preparation_contract_checks"]
