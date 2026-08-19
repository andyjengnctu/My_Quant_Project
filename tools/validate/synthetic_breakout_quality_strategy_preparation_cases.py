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
    min_roos_source = settings.parameter_sources["min_roos"]
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
        "parameter_coverage_is_preflighted_before_any_pair_replay",
        True,
        "PARAM_PERIOD_MISMATCH" in preparation_source
        and "comparison_period" in preparation_source
        and "comparison_start_date=comparison_start" in orchestration_source
        and "comparison_end_date=comparison_end" in orchestration_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_supports_dependency_waves_after_score_period_becomes_known",
        True,
        "max_waves" in preparation_source
        and "重新規劃後沒有可執行且依賴已就緒的動作" in preparation_source,
    )

    from filters.breakout_quality import strategy_compare_preparation as preparation_module
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
        "collect_artifact_status",
        side_effect=(second_wave_status, final_wave_status),
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

    from config.strategy_compare import get_strategy_comparison_settings
    from core.active_param_ensemble import (
        ACTIVE_PARAM_ENSEMBLE_SCHEMA_TYPE,
        get_active_param_ensemble_date_range,
    )
    from filters.breakout_quality.strategy_compare_preparation_status import (
        _validate_param_artifact,
        _validate_param_training_identity,
        resolve_param_source_path,
    )
    from filters.breakout_quality.strategy_param_training import (
        FULL_ROOS_SEARCH_FIELDS,
        MIN_ROOS_SEARCH_FIELDS,
        prepare_extending_full_roos_params,
        prepare_extending_min_roos_params,
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
            mapping[effective_start] = [{"params": dict(params)}]
            simple_mapping[effective_start] = dict(params)
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

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_preflight_identity_tracks_current_min_roos_contract_not_removed_full_baseline",
        True,
        all(token in preparation_source for token in (
            "TRAINING_CONFIG_MISMATCH", "BINARY_PIT_IDENTITY_MISSING",
            "MIN_ROOS_SEARCH_FIELDS_MISMATCH", "trials_per_fold",
            "max_position_cap_pct",
        ))
        and "BASELINE_PARAMS_IDENTITY_MISMATCH" not in preparation_source,
    )

    from config.strategy_compare import get_strategy_comparison_settings
    from filters.breakout_quality import strategy_param_training as param_training_module

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
        "strategy_compare_missing_selection_pit_is_model_work_blocker_not_checkpoint_rebuild",
        True,
        not missing_ready
        and str(missing_row.get("status") or "").startswith("SELECTION_PIT_INVALID")
        and len(missing_actions) == 3
        and all(action.action == "BLOCKED" for action in missing_actions)
        and all(action.builder_type is None for action in missing_actions)
        and all(action.producer_work_type == "model_training" for action in missing_actions)
        and all("Research → [1] 模型訓練 → [2]" in action.description for action in missing_actions),
    )

    failed_gate_contract = SimpleNamespace(
        model_validation_gate={"status": "FAIL", "checks": {"synthetic": False}},
        available_from="2016-04-01",
        available_through="2020-12-31",
        manifest_path=Path("models/synthetic/selection_point_in_time_manifest.json"),
        audit_path=Path("outputs/synthetic/selection_point_in_time_audit.json"),
        score_path=Path("models/synthetic/selection_point_in_time_scores.csv"),
    )
    failed_runtime_periods: dict[str, tuple[str, str]] = {}
    failed_actions = []
    with patch.object(
        dl_artifacts_module,
        "load_selection_point_in_time_ranking_contract",
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
        and failed_gate_loader.call_args.kwargs.get("require_model_validation_pass") is False,
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
