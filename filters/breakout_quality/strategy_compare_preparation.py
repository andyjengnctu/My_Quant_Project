"""Strategy Compare deterministic prerequisite builder/execution service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from core.console_report import project_relative_display_path
from core.strategy_comparison import (
    StrategyComparisonSettings,
    StrategyPreparationAction,
    StrategyPreparationPlan,
)
from filters.breakout_quality.export_scores import export_forward_oos_scores
from filters.breakout_quality.strategy_compare_preparation_status import (
    collect_artifact_status,
    model_upstream_prerequisite_blockers,
    resolve_comparison_period,
    resolve_param_source_path,
)
from services.optimizer.strategy_param_service import ensure_strategy_parameter_artifact
from services.optimizer.strategy_param_training import (
    prepare_extending_full_roos_params,
    prepare_extending_min_roos_params,
    prepare_oos_frozen_roos_params,
    prepare_selection_historical_full_roos_params,
    prepare_selection_historical_p2_params,
    prepare_strategy_parameter_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _execute_preparation_action(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    action: StrategyPreparationAction,
) -> None:
    if action.builder_type == "forward_oos_scores":
        _kind, dl_id, _name = action.artifact_key.split(":", 2)
        source = settings.dl_sources[dl_id]
        builder = source.forward_scores_builder
        if builder is None:
            raise RuntimeError(f"缺少forward score builder: {dl_id}")
        options = dict(builder.options)
        export_forward_oos_scores(
            project_root=root,
            filter_id=source.filter_id,
            model_architecture=source.model_architecture,
            experiment_profile=source.experiment_profile,
            inference_batch_size=int(options.get("inference_batch_size", 4096)),
            inference_workers=int(options.get("inference_workers", 4)),
            device=str(options.get("device", "auto")),
            mixed_precision=bool(options.get("mixed_precision", True)),
            mixed_precision_dtype=str(options.get("mixed_precision_dtype", "bfloat16")),
            deterministic_algorithms=bool(options.get("deterministic_algorithms", True)),
            allow_tf32=bool(options.get("allow_tf32", False)),
            preload_feature_bank=bool(options.get("preload_feature_bank", True)),
        )
        return
    if action.builder_type == "canonical_optimizer_strategy_params":
        _kind, source_id = action.artifact_key.split(":", 1)
        source = settings.parameter_sources[source_id]
        if not source.canonical_family or not source.canonical_evaluation_mode:
            raise RuntimeError(f"canonical optimizer參數來源缺少family/mode: {source_id}")
        ensure_strategy_parameter_artifact(
            root,
            family=source.canonical_family,
            evaluation_mode=source.canonical_evaluation_mode,
            policy=settings.param_policy,
        )
        return
    if action.builder_type in {"extending_min_roos_stitch", "extending_full_roos_stitch"}:
        _kind, source_id = action.artifact_key.split(":", 1)
        source = settings.parameter_sources[source_id]
        builder = source.builder
        if builder is None:
            raise RuntimeError(f"參數來源builder設定不完整: {source_id}")
        options = dict(builder.options)
        stitch = (
            prepare_extending_full_roos_params
            if action.builder_type == "extending_full_roos_stitch"
            else prepare_extending_min_roos_params
        )
        stitch(
            project_root=root,
            param_policy=settings.param_policy,
            historical_params_path=str(options["historical_params_path"]),
            current_params_path=str(options["current_params_path"]),
            output_relative_dir=str(options["output_relative_dir"]),
            quiet=bool(options.get("quiet", False)),
        )
        return
    if action.builder_type == "oos_param_freeze":
        _kind, source_id = action.artifact_key.split(":", 1)
        source = settings.parameter_sources[source_id]
        builder = source.builder
        if builder is None:
            raise RuntimeError(f"參數來源builder設定不完整: {source_id}")
        options = dict(builder.options)
        dependency_id = str(options.get("source_param_source_id") or "").strip()
        if not dependency_id or dependency_id not in settings.parameter_sources:
            raise RuntimeError(f"OOS freeze缺少合法source_param_source_id: {source_id}")
        dependency_path = resolve_param_source_path(root, settings, dependency_id)
        prepare_oos_frozen_roos_params(
            project_root=root,
            param_policy=settings.param_policy,
            source_params_path=project_relative_display_path(dependency_path, project_root=root),
            output_relative_dir=str(options["output_relative_dir"]),
            freeze_effective_date=str(options.get("freeze_effective_date") or "2021-01-01"),
            freeze_cutoff_date=str(options.get("freeze_cutoff_date") or "2020-12-31"),
            display_name=str(options.get("display_name") or source_id),
            quiet=bool(options.get("quiet", False)),
        )
        return
    if action.builder_type == "selection_historical_p2":
        _kind, source_id = action.artifact_key.split(":", 1)
        source = settings.parameter_sources[source_id]
        builder = source.builder
        if builder is None:
            raise RuntimeError(f"參數來源builder設定不完整: {source_id}")
        options = dict(builder.options)
        prepare_selection_historical_p2_params(
            project_root=root,
            dataset=settings.dataset,
            param_policy=settings.param_policy,
            comparison_output_root=str(settings.output_root),
            comparison_output_roots=(
                str(settings.output_root),
                *tuple(str(value) for value in settings.reuse_output_roots),
            ),
            trials_per_fold=int(options["trials_per_fold"]),
            first_oos_date=settings.start_date,
            last_oos_date=settings.end_date,
            train_window_months=int(options["train_window_months"]),
            oos_months=int(options["oos_months"]),
            max_positions=int(settings.max_positions),
            rotation=str(settings.rotation),
            fixed_risk=float(options["fixed_risk"]),
            max_position_cap_pct=float(options["max_position_cap_pct"]),
            optimizer_seed=int(options["optimizer_seed"]),
            resume_parameter_training=bool(
                options.get("resume", settings.preparation.resume_parameter_training)
            ),
            quiet=bool(options.get("quiet", False)),
        )
        return
    if action.builder_type == "selection_historical_full_roos":
        _kind, source_id = action.artifact_key.split(":", 1)
        source = settings.parameter_sources[source_id]
        builder = source.builder
        if builder is None:
            raise RuntimeError(f"參數來源builder設定不完整: {source_id}")
        options = dict(builder.options)
        prepare_selection_historical_full_roos_params(
            project_root=root,
            dataset=settings.dataset,
            param_policy=settings.param_policy,
            trials_per_fold=int(options["trials_per_fold"]),
            first_oos_date=settings.start_date,
            last_oos_date=settings.end_date,
            train_window_months=int(options["train_window_months"]),
            oos_months=int(options["oos_months"]),
            max_positions=int(settings.max_positions),
            rotation=str(settings.rotation),
            fixed_risk=float(options["fixed_risk"]),
            max_position_cap_pct=float(options["max_position_cap_pct"]),
            optimizer_seed=int(options["optimizer_seed"]),
            resume_parameter_training=bool(
                options.get("resume", settings.preparation.resume_parameter_training)
            ),
            quiet=bool(options.get("quiet", False)),
        )
        return
    if action.builder_type == "expected_r_calibration":
        _runtime, arm_id, _artifact = action.artifact_key.split(":", 2)
        arm = settings.arms[arm_id]
        if not arm.dl_id:
            raise RuntimeError(f"Expected-R calibration arm缺少dl_id: {arm_id}")
        source = settings.dl_sources[arm.dl_id]
        from services.breakout_quality.expected_r_calibration import (
            build_expected_r_calibration_artifact,
        )
        build_expected_r_calibration_artifact(
            project_root=root,
            filter_id=source.filter_id,
            model_architecture=source.model_architecture,
            experiment_profile=source.experiment_profile,
            phase_id=settings.profile_id,
            selection_runtime_start_date=(
                settings.start_date if settings.profile_id == "selection_pit" else None
            ),
        )
        return
    if action.builder_type == "expected_excess_r_calibration":
        _runtime, arm_id, _artifact = action.artifact_key.split(":", 2)
        arm = settings.arms[arm_id]
        if not arm.dl_id:
            raise RuntimeError(f"Expected Excess-R calibration arm缺少dl_id: {arm_id}")
        source = settings.dl_sources[arm.dl_id]
        from services.breakout_quality.excess_r_calibration import (
            build_expected_excess_r_calibration_artifact,
        )
        build_expected_excess_r_calibration_artifact(
            project_root=root,
            filter_id=source.filter_id,
            model_architecture=source.model_architecture,
            experiment_profile=source.experiment_profile,
            phase_id=settings.profile_id,
            selection_runtime_start_date=settings.start_date,
        )
        return
    if action.builder_type == "binary_dl_min_roos_rolling":
        _kind, source_id = action.artifact_key.split(":", 1)
        source = settings.parameter_sources[source_id]
        builder = source.builder
        if builder is None:
            raise RuntimeError(f"參數來源builder設定不完整: {source_id}")
        options = dict(builder.options)
        model_source_id = str(
            source.trained_with_dl_id or options.get("model_source_id") or ""
        ).strip()
        if not model_source_id or model_source_id not in settings.dl_sources:
            raise RuntimeError(f"參數來源builder缺少合法model_source_id: {source_id}")
        dl = settings.dl_sources[model_source_id]
        prepare_strategy_parameter_source(
            project_root=root,
            dataset=settings.dataset,
            filter_id=dl.filter_id,
            model_architecture=dl.model_architecture,
            experiment_profile=dl.experiment_profile,
            param_policy=settings.param_policy,
            parameter_set=str(options.get("parameter_set", "p3")),
            trials_per_fold=int(options["trials_per_fold"]),
            max_positions=int(settings.max_positions),
            rotation=str(settings.rotation),
            fixed_risk=float(options.get("fixed_risk", DEFAULT_FIXED_RISK)),
            max_position_cap_pct=float(options.get("max_position_cap_pct", DEFAULT_MAX_POSITION_CAP_PCT)),
            p3_variant=(
                None
                if options.get("p3_variant") in (None, "")
                else str(options.get("p3_variant"))
            ),
            build_binary_pit=bool(options.get("build_binary_pit", True)),
            binary_pit_resume=bool(options.get("binary_pit_resume", True)),
            resume_parameter_training=bool(
                options.get(
                    "resume",
                    settings.preparation.resume_parameter_training,
                )
            ),
            quiet=bool(options.get("quiet", False)),
        )
        return
    raise RuntimeError(f"不支援的前置builder: {action.builder_type}")



def _run_preparation_plan(
    *,
    project_root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    status_refresher: Callable[[], dict[str, Any]],
    required_artifact_keys: tuple[str, ...] | None,
    failure_prefix: str,
) -> dict[str, Any]:
    """Execute one canonical dependency-aware preparation loop.

    ``required_artifact_keys`` limits execution to the requested artifacts plus their
    declared dependency closure.  This is how Multiple-seed robustness reuses the same
    planner while intentionally ignoring unrelated canonical DL-score blockers.
    """

    root = Path(project_root).resolve()
    current = status
    requested_plan: StrategyPreparationPlan = status["preparation_plan"]
    requested_keys = None if required_artifact_keys is None else tuple(required_artifact_keys)
    selected_requested = (
        requested_plan
        if requested_keys is None
        else requested_plan.select(requested_keys)
    )
    if selected_requested.blocked:
        blocked = [
            item.artifact_key
            for item in selected_requested.actions
            if item.action == "BLOCKED"
        ]
        raise RuntimeError(
            f"{failure_prefix}包含BLOCKED工件: " + ", ".join(blocked)
        )
    if selected_requested.overall_status == "READY":
        return current
    if not settings.preparation.auto_prepare:
        raise RuntimeError("目前config已關閉auto_prepare")

    executed_signatures: set[tuple[str, str, str | None, str]] = set()
    max_waves = max(2, len(selected_requested.actions) * 2 + 2)
    for _wave in range(max_waves):
        current_plan: StrategyPreparationPlan = current["preparation_plan"]
        selected = (
            current_plan
            if requested_keys is None
            else current_plan.select(requested_keys)
        )
        if selected.overall_status == "READY":
            current["requested_preparation_plan"] = selected_requested
            return current
        if selected.blocked:
            blocked = [
                item.artifact_key for item in selected.actions if item.action == "BLOCKED"
            ]
            raise RuntimeError(
                f"{failure_prefix}依賴於重新規劃後變成BLOCKED: "
                + ", ".join(blocked)
            )

        action = selected.next_runnable_action(
            executed_signatures=executed_signatures
        )
        if action is None:
            remaining = [
                f"{item.artifact_key}:{item.action}"
                for item in selected.actions
                if item.action != "REUSE"
            ]
            raise RuntimeError(
                f"{failure_prefix}重新規劃後沒有可執行且依賴已就緒的動作: "
                + ", ".join(remaining)
            )

        print(f"\n[前置] {action.description}")
        try:
            _execute_preparation_action(root=root, settings=settings, action=action)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"{failure_prefix}失敗: {action.artifact_key} | action={action.action} | "
                f"path={action.path} | {type(exc).__name__}: {exc}"
            ) from exc
        executed_signatures.add(
            (action.artifact_key, action.action, action.builder_type, action.path)
        )
        current = status_refresher()
        current["requested_preparation_plan"] = selected_requested

    current_plan = current["preparation_plan"]
    selected = (
        current_plan if requested_keys is None else current_plan.select(requested_keys)
    )
    remaining = [
        f"{item.artifact_key}:{item.action}"
        for item in selected.actions
        if item.action != "REUSE"
    ]
    raise RuntimeError(
        f"{failure_prefix}超過最大依賴波次仍未READY: " + ", ".join(remaining)
    )


def prepare_strategy_parameter_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    required_source_ids: tuple[str, ...] | list[str] | set[str],
    status_refresher: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare requested strategy parameters through the canonical dependency runner."""

    root = Path(project_root).resolve()
    requested = tuple(sorted({str(value) for value in required_source_ids}))
    if not requested:
        return status
    unknown = sorted(set(requested) - set(settings.parameter_sources))
    if unknown:
        raise ValueError("策略參數前置包含未知source: " + ", ".join(unknown))
    refresh_status = (
        status_refresher
        if status_refresher is not None
        else lambda: collect_artifact_status(project_root=root, settings=settings)
    )
    return _run_preparation_plan(
        project_root=root,
        settings=settings,
        status=status,
        status_refresher=refresh_status,
        required_artifact_keys=tuple(f"param:{source_id}" for source_id in requested),
        failure_prefix="策略參數前置",
    )


def prepare_strategy_comparison_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    status_refresher: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    refresh_status = (
        status_refresher
        if status_refresher is not None
        else lambda: collect_artifact_status(project_root=root, settings=settings)
    )
    return _run_preparation_plan(
        project_root=root,
        settings=settings,
        status=status,
        status_refresher=refresh_status,
        required_artifact_keys=None,
        failure_prefix="策略比較前置",
    )


__all__ = [
    "collect_artifact_status",
    "model_upstream_prerequisite_blockers",
    "prepare_strategy_comparison_artifacts",
    "prepare_strategy_parameter_artifacts",
    "resolve_comparison_period",
    "resolve_param_source_path",
]
