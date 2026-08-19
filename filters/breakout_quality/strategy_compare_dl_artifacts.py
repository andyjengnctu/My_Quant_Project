"""Strategy Compare DL artifact demand, source resolution, and readiness validation.

Current profile membership determines which DL artifacts are required.  Artifact
validation itself is source-semantic driven and does not own parameter, calibration,
or completed-pair reuse policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    StrategyComparisonSettings,
    StrategyPreparationAction,
)
from filters.breakout_quality.artifact_dependency_registry import (
    PRODUCER_EXISTING_ARTIFACT,
    collect_model_upstream_readiness,
    model_upstream_prerequisite_blockers as _shared_model_upstream_prerequisite_blockers,
)
from filters.breakout_quality.artifacts import (
    compute_file_sha256,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME,
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    resolve_selection_point_in_time_audit_json_path,
    resolve_selection_point_in_time_manifest_path,
    resolve_selection_point_in_time_score_path,
)
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_continuous_ranker_oos_contract,
    load_selection_point_in_time_ranking_contract,
    resolve_continuous_ranker_oos_score_path,
)


def model_upstream_prerequisite_blockers(
    root: Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int = 0,
) -> tuple[str, ...]:
    """Compatibility facade over the canonical artifact readiness registry."""

    return _shared_model_upstream_prerequisite_blockers(
        root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        dataset=dataset,
        max_tickers=max_tickers,
    )


def _preparation_action(
    *,
    action_id: str,
    artifact_key: str,
    action: str,
    builder_type: str | None,
    description: str,
    path: str,
    dependencies: tuple[str, ...] = (),
    producer_work_type: str | None = None,
    execution_priority: int = 100,
) -> StrategyPreparationAction:
    return StrategyPreparationAction(
        action_id=action_id,
        artifact_key=artifact_key,
        action=action,
        builder_type=builder_type,
        description=description,
        path=path,
        dependencies=tuple(dependencies),
        producer_work_type=producer_work_type,
        execution_priority=int(execution_priority),
    )


def resolve_required_artifact_sources(
    settings: StrategyComparisonSettings,
) -> tuple[set[str], set[str], set[str]]:
    """Resolve profile demand without performing any artifact I/O."""

    required_param_sources = {arm.param_source for arm in settings.enabled_arms}
    required_dl_sources = {
        arm.dl_id for arm in settings.enabled_arms if arm.dl_enabled and arm.dl_id
    }
    runtime_required_dl_sources = set(required_dl_sources)
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
        }:
            safety_dl_id = str(
                dict(arm.dl_runtime_options or {}).get("safety_dl_id") or ""
            ).strip()
            if not safety_dl_id or safety_dl_id not in settings.dl_sources:
                raise ValueError(
                    f"dual-model safety arm缺少合法safety_dl_id: {arm.arm_id}"
                )
            required_dl_sources.add(safety_dl_id)
            runtime_required_dl_sources.add(safety_dl_id)
        if (
            arm.dl_runtime_mode
            == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
        ):
            fit_dl_id = str(
                dict(arm.dl_runtime_options or {}).get("expected_r_fit_dl_id") or ""
            ).strip()
            if not fit_dl_id or fit_dl_id not in settings.dl_sources:
                raise ValueError(
                    f"Expected-PnL arm缺少合法expected_r_fit_dl_id: {arm.arm_id}"
                )
            required_dl_sources.add(fit_dl_id)
        if arm.dl_runtime_mode in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        }:
            fit_dl_id = str(
                dict(arm.dl_runtime_options or {}).get("expected_excess_r_fit_dl_id") or ""
            ).strip()
            if not fit_dl_id or fit_dl_id not in settings.dl_sources:
                raise ValueError(
                    f"Excess-Alpha arm缺少合法expected_excess_r_fit_dl_id: {arm.arm_id}"
                )
            required_dl_sources.add(fit_dl_id)
    for source_id in required_param_sources:
        trained_with = settings.parameter_sources[source_id].trained_with_dl_id
        if trained_with:
            required_dl_sources.add(trained_with)
    return required_param_sources, required_dl_sources, runtime_required_dl_sources


def _collect_model_upstream_dependencies(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    source: Any,
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
    cache: dict[tuple[str, str, str], tuple[str, ...]],
) -> tuple[str, ...]:
    upstream_identity = (
        str(source.filter_id),
        str(source.model_architecture),
        str(source.experiment_profile),
    )
    if upstream_identity in cache:
        return cache[upstream_identity]

    upstream_rows = collect_model_upstream_readiness(
        root,
        filter_id=source.filter_id,
        model_architecture=source.model_architecture,
        experiment_profile=source.experiment_profile,
        dataset=settings.dataset,
        max_tickers=0,
    )
    upstream_key_by_type = {
        item.artifact_type: (
            f"model-upstream:{source.filter_id}:{source.model_architecture}:"
            f"{source.experiment_profile}:{item.artifact_type}"
        )
        for item in upstream_rows
    }
    upstream_keys: list[str] = []
    for item in upstream_rows:
        artifact_key = upstream_key_by_type[item.artifact_type]
        dependency_keys = tuple(
            upstream_key_by_type[dependency]
            for dependency in item.dependencies
            if dependency in upstream_key_by_type
        )
        display_path = project_relative_display_path(item.path, project_root=root)
        action = "REUSE" if item.ready else "BLOCKED"
        description = (
            item.description
            if item.ready
            else (
                item.description
                + "；Strategy Compare不得建立Dataset／Label／Target，"
                + "請由模型訓練工作類型執行「準備策略比較所需模型工件」"
            )
        )
        actions.append(
            _preparation_action(
                action_id=artifact_key,
                artifact_key=artifact_key,
                action=action,
                builder_type=None,
                description=description,
                path=display_path,
                dependencies=dependency_keys,
                producer_work_type=(
                    PRODUCER_EXISTING_ARTIFACT
                    if item.ready
                    else item.producer_work_type
                ),
                execution_priority=0,
            )
        )
        artifact_identities[artifact_key] = {
            "path": display_path,
            "sha256": compute_file_sha256(item.path) if item.path.is_file() else None,
            "status": item.status,
        }
        upstream_keys.append(artifact_key)
    cache[upstream_identity] = tuple(upstream_keys)
    return cache[upstream_identity]


def _selection_pit_override_dir(root: Path, source: Any) -> Path | None:
    dirname = None if getattr(source, "point_in_time_dirname", None) in (None, "") else str(source.point_in_time_dirname).strip()
    if dirname is None:
        return None
    return (
        resolve_filter_model_output_dir(
            root, source.filter_id, source.model_architecture, source.experiment_profile
        )
        / dirname
    ).resolve()


def _collect_selection_pit_source_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    dl_id: str,
    source: Any,
    source_upstream_dependencies: tuple[str, ...],
    runtime_required_dl_sources: set[str],
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
    runtime_periods: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], bool]:
    pit_contract = None
    pit_ready = False
    pit_status = "MISSING"
    pit_gate_status: str | None = None
    try:
        pit_override_dir = _selection_pit_override_dir(root, source)
        pit_contract = load_selection_point_in_time_ranking_contract(
            str(root),
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
            require_model_validation_pass=False,
            point_in_time_dir_override=pit_override_dir,
        )
        expected_fold_months = getattr(source, "point_in_time_fold_months", None)
        if expected_fold_months is not None and int(pit_contract.manifest.get("fold_months", -1)) != int(expected_fold_months):
            raise ValueError(
                "Selection PIT fold cadence與Strategy Compare mode不一致: "
                f"expected={int(expected_fold_months)}, actual={pit_contract.manifest.get('fold_months')!r}"
            )
        expected_anchor = getattr(source, "point_in_time_fold_anchor_date", None)
        if expected_anchor not in (None, ""):
            actual_anchor = str(pit_contract.manifest.get("fold_anchor_date") or "").strip()
            if actual_anchor != str(expected_anchor):
                raise ValueError(
                    "Selection PIT fold anchor與Strategy Compare mode不一致: "
                    f"expected={expected_anchor}, actual={actual_anchor or None}"
                )
        pit_gate_status = str(
            pit_contract.model_validation_gate.get("status") or ""
        ).strip().upper() or None
        # Strategy Compare is a research consumer of already-completed PIT artifacts.
        # Artifact legality / PIT chronology remain hard gates, while the model-quality
        # gate is advisory evidence: a FAIL must be visible, but does not preclude the
        # explicitly configured strategy-conversion experiment.
        pit_ready = True
        pit_status = (
            "READY"
            if pit_gate_status == "PASS"
            else f"READY_MODEL_GATE_{pit_gate_status or 'UNKNOWN'}"
        )
        if dl_id in runtime_required_dl_sources:
            runtime_periods[dl_id] = (
                str(pit_contract.available_from),
                str(pit_contract.available_through),
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        pit_status = f"SELECTION_PIT_INVALID ({type(exc).__name__})"

    if pit_contract is not None:
        files = {
            "manifest": pit_contract.manifest_path,
            "audit": pit_contract.audit_path,
            "forward_scores": pit_contract.score_path,
        }
    else:
        pit_override_dir = _selection_pit_override_dir(root, source)
        if pit_override_dir is not None:
            files = {
                "manifest": pit_override_dir / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
                "audit": pit_override_dir / SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME,
                "forward_scores": pit_override_dir / SELECTION_POINT_IN_TIME_SCORE_FILENAME,
            }
        else:
            files = {
                "manifest": resolve_selection_point_in_time_manifest_path(
                    root,
                    source.filter_id,
                    source.model_architecture,
                    source.experiment_profile,
                ),
                "audit": resolve_selection_point_in_time_audit_json_path(
                    root,
                    source.filter_id,
                    source.model_architecture,
                    source.experiment_profile,
                ),
                "forward_scores": resolve_selection_point_in_time_score_path(
                    root,
                    source.filter_id,
                    source.model_architecture,
                    source.experiment_profile,
                ),
            }
    # Rolling PIT scores / manifest / audit are model-research artifacts.
    # Strategy Compare is a consumer only: even when compatible fold checkpoints
    # already exist, rebuilding the PIT bundle also runs the PIT model Gate and
    # therefore belongs to Research -> Model Training -> prepare model artifacts
    # for the matching Fast/Overnight Rolling mode.
    # Keep the configured builder identity for the model-work-type orchestrator,
    # but never turn a missing PIT bundle into a Strategy Compare BUILD action.
    checkpoint_rebuild_blockers: tuple[str, ...] = tuple()
    file_rows: dict[str, Any] = {}
    for key, path in files.items():
        sha256 = compute_file_sha256(path) if path.is_file() else None
        artifact_identities[f"dl:{dl_id}:{key}"] = {
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
        }
        if pit_ready and settings.preparation.reuse_ready_artifacts:
            action = "REUSE"
            description = "重用既有Selection PIT score／audit工件"
            if pit_gate_status not in (None, "", "PASS"):
                description += (
                    f"；WARN: Model Gate={pit_gate_status}，僅作研究診斷，"
                    "本次仍允許既定strategy-conversion replay；不代表runtime promotion資格"
                )
        else:
            action = "BLOCKED"
            description = (
                "缺少或無效的Selection PIT模型工件；Strategy Compare只消費既有PIT "
                "score／manifest／audit，不建立、不重建也不執行PIT Model Gate。"
                "請先執行 Research → [1] 模型訓練 → 準備策略比較所需模型工件，並選擇相同 Fast／Overnight Test。"
            )
        file_rows[key] = {
            "ready": pit_ready,
            "status": pit_status,
            "action": action,
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
        }
        actions.append(
            _preparation_action(
                action_id=f"dl:{dl_id}:{key}",
                artifact_key=f"dl:{dl_id}:{key}",
                action=action,
                builder_type=None,
                description=description,
                path=file_rows[key]["path"],
                dependencies=(
                    (f"dl:{dl_id}:forward_scores",)
                    if key == "audit"
                    else source_upstream_dependencies
                ),
                producer_work_type=(
                    "existing_artifact" if action == "REUSE" else "model_training"
                ),
            )
        )
    return (
        {
            "ready": pit_ready,
            "status": pit_status,
            "identity": source.as_dict(),
            "files": file_rows,
            "checkpoint_rebuild_blockers": list(checkpoint_rebuild_blockers),
            "model_validation_gate": (
                None
                if pit_contract is None
                else dict(pit_contract.model_validation_gate)
            ),
        },
        pit_ready,
    )


def _collect_continuous_ranker_source_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    dl_id: str,
    source: Any,
    artifacts: Any,
    source_upstream_dependencies: tuple[str, ...],
    runtime_required_dl_sources: set[str],
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
    runtime_periods: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], bool]:
    model_ready = False
    runtime_ready = False
    model_status = "MISSING"
    runtime_status = "MISSING"
    continuous_contract = None
    try:
        continuous_contract = load_continuous_ranker_oos_contract(
            root,
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
        model_ready = True
        runtime_ready = True
        model_status = "READY"
        runtime_status = "READY"
        if dl_id in runtime_required_dl_sources:
            runtime_periods[dl_id] = (
                str(continuous_contract.execution_start),
                str(continuous_contract.available_through),
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        model_status = f"CONTINUOUS_MODEL_OR_REPORT_INVALID ({type(exc).__name__})"
        runtime_status = f"CONTINUOUS_OOS_SCORES_INVALID ({type(exc).__name__})"

    score_path = (
        continuous_contract.score_path
        if continuous_contract is not None
        else resolve_continuous_ranker_oos_score_path(
            root,
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
    )
    report_path = (
        continuous_contract.report_path
        if continuous_contract is not None
        else resolve_filter_model_output_dir(
            root,
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
        / CONTINUOUS_RANKER_REPORT_FILENAME
    )
    files = {
        "model": artifacts.model_path,
        "manifest": artifacts.manifest_path,
        "report": report_path,
        "forward_scores": score_path,
    }
    file_rows: dict[str, Any] = {}
    for key, path in files.items():
        sha256 = compute_file_sha256(path) if path.is_file() else None
        artifact_identities[f"dl:{dl_id}:{key}"] = {
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
        }
        ready = model_ready if key in {"model", "manifest", "report"} else runtime_ready
        status = (
            "READY"
            if ready
            else model_status
            if key in {"model", "manifest", "report"}
            else runtime_status
        )
        action = "REUSE" if ready and settings.preparation.reuse_ready_artifacts else "BLOCKED"
        research_label = f"{dl_id}/{source.experiment_profile}"
        description = (
            f"重用既有{research_label} continuous research工件"
            if ready and key in {"model", "manifest", "report"}
            else f"重用既有{research_label} frozen OOS continuous scores"
            if ready
            else (
                f"缺少或無效；請由模型訓練工作類型執行「準備策略比較所需模型工件」"
                f"建立{research_label}工件；策略比較不得自動重訓"
            )
        )
        file_rows[key] = {
            "ready": ready,
            "status": status,
            "action": action,
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
        }
        actions.append(
            _preparation_action(
                action_id=f"dl:{dl_id}:{key}",
                artifact_key=f"dl:{dl_id}:{key}",
                action=action,
                builder_type=None,
                description=description,
                path=file_rows[key]["path"],
                dependencies=(
                    (
                        f"dl:{dl_id}:model",
                        f"dl:{dl_id}:manifest",
                        f"dl:{dl_id}:report",
                    )
                    if key == "forward_scores"
                    else source_upstream_dependencies
                    if key in {"model", "manifest", "report"}
                    else ()
                ),
                producer_work_type=(
                    "existing_artifact" if action == "REUSE" else "model_training"
                ),
            )
        )
    return (
        {
            "ready": bool(model_ready and runtime_ready),
            "status": "READY" if model_ready and runtime_ready else "NOT_READY",
            "identity": source.as_dict(),
            "files": file_rows,
        },
        model_ready,
    )


def _collect_standard_model_source_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    dl_id: str,
    source: Any,
    artifacts: Any,
    source_upstream_dependencies: tuple[str, ...],
    runtime_required_dl_sources: set[str],
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
    runtime_periods: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], bool]:
    model_ready = False
    model_status = "MISSING"
    try:
        contract = load_model_artifact_contract(
            str(root),
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
        model_ready = True
        model_status = "READY"
        if contract.paths.model_architecture != source.model_architecture:
            raise ValueError("model architecture與config不一致")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        model_status = f"MODEL_MISSING_OR_INVALID ({type(exc).__name__})"

    runtime_ready = False
    runtime_status = "MISSING"
    try:
        runtime_contract = load_runtime_artifact_contract(
            str(root),
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
        runtime_ready = True
        runtime_status = "READY"
        if dl_id in runtime_required_dl_sources:
            runtime_periods[dl_id] = (
                str(runtime_contract.execution_start),
                str(runtime_contract.available_through),
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        runtime_status = f"MISSING_OR_STALE ({type(exc).__name__})"

    files = {
        "model": artifacts.model_path,
        "manifest": artifacts.manifest_path,
        "forward_scores": artifacts.score_path,
    }
    file_rows: dict[str, Any] = {}
    for key, path in files.items():
        sha256 = compute_file_sha256(path) if path.is_file() else None
        artifact_identities[f"dl:{dl_id}:{key}"] = {
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
        }
        if key in {"model", "manifest"}:
            ready = model_ready and path.is_file()
            status = "READY" if ready else model_status
            action = "REUSE" if ready else "BLOCKED"
            description = "重用既有模型工件" if ready else "需由模型正式入口建立或修復"
            builder_type = None
        else:
            ready = runtime_ready
            status = runtime_status
            builder = source.forward_scores_builder
            if ready and settings.preparation.reuse_ready_artifacts:
                action = "REUSE"
                description = "重用正式forward-OOS scores"
                builder_type = None
            elif ready:
                if (
                    settings.preparation.auto_prepare
                    and settings.preparation.rebuild_stale_artifacts
                    and builder is not None
                    and builder.enabled
                ):
                    action = "REBUILD"
                    description = "config禁止重用，重新匯出正式forward-OOS scores"
                    builder_type = builder.builder_type
                else:
                    action = "BLOCKED"
                    description = "config禁止重用且未允許重新建立scores"
                    builder_type = None
            elif (
                model_ready
                and settings.preparation.auto_prepare
                and builder is not None
                and builder.enabled
            ):
                action = "REBUILD" if path.exists() else "BUILD"
                if action == "REBUILD" and not settings.preparation.rebuild_stale_artifacts:
                    action = "BLOCKED"
                description = (
                    "使用既有模型重新匯出正式forward-OOS scores"
                    if action in {"BUILD", "REBUILD"}
                    else "scores過期且config禁止自動重建"
                )
                builder_type = builder.builder_type if action != "BLOCKED" else None
            else:
                action = "BLOCKED"
                description = "缺少正式scores且無可用自動builder或模型工件"
                builder_type = None
            status = "READY" if ready else action
        file_rows[key] = {
            "ready": ready,
            "status": status,
            "action": action,
            "path": project_relative_display_path(path, project_root=root),
            "sha256": sha256,
        }
        dependencies = (
            (f"dl:{dl_id}:model", f"dl:{dl_id}:manifest")
            if key == "forward_scores"
            else source_upstream_dependencies
            if key in {"model", "manifest"}
            else ()
        )
        producer_work_type = (
            "strategy_compare_deterministic_rebuild"
            if key == "forward_scores" and action in {"BUILD", "REBUILD"}
            else "existing_artifact"
            if action == "REUSE"
            else "model_training"
        )
        actions.append(
            _preparation_action(
                action_id=f"dl:{dl_id}:{key}",
                artifact_key=f"dl:{dl_id}:{key}",
                action=action,
                builder_type=builder_type,
                description=description,
                path=file_rows[key]["path"],
                dependencies=dependencies,
                producer_work_type=producer_work_type,
                execution_priority=(20 if key == "forward_scores" else 100),
            )
        )
    return (
        {
            "ready": bool(model_ready and runtime_ready),
            "status": "READY" if model_ready and runtime_ready else "NOT_READY",
            "identity": source.as_dict(),
            "files": file_rows,
        },
        model_ready,
    )


def collect_dl_artifact_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    required_dl_sources: set[str],
    runtime_required_dl_sources: set[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[StrategyPreparationAction],
    dict[str, bool],
    dict[str, tuple[str, str]],
]:
    """Collect DL readiness through source-specific validators."""

    dl_rows: dict[str, Any] = {}
    artifact_identities: dict[str, Any] = {}
    actions: list[StrategyPreparationAction] = []
    dl_model_ready: dict[str, bool] = {}
    runtime_periods: dict[str, tuple[str, str]] = {}
    upstream_cache: dict[tuple[str, str, str], tuple[str, ...]] = {}

    for dl_id in settings.dl_sources:
        if dl_id not in required_dl_sources:
            continue
        source = settings.dl_sources[dl_id]
        source_upstream_dependencies = _collect_model_upstream_dependencies(
            root=root,
            settings=settings,
            source=source,
            artifact_identities=artifact_identities,
            actions=actions,
            cache=upstream_cache,
        )
        artifacts = resolve_filter_artifact_paths(
            root,
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )
        common = {
            "root": root,
            "settings": settings,
            "dl_id": dl_id,
            "source": source,
            "source_upstream_dependencies": source_upstream_dependencies,
            "runtime_required_dl_sources": runtime_required_dl_sources,
            "artifact_identities": artifact_identities,
            "actions": actions,
            "runtime_periods": runtime_periods,
        }
        if source.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            row, model_ready = _collect_selection_pit_source_status(**common)
        elif source.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            row, model_ready = _collect_continuous_ranker_source_status(
                artifacts=artifacts,
                **common,
            )
        else:
            row, model_ready = _collect_standard_model_source_status(
                artifacts=artifacts,
                **common,
            )
        dl_rows[dl_id] = row
        dl_model_ready[dl_id] = model_ready

    return dl_rows, artifact_identities, actions, dl_model_ready, runtime_periods


__all__ = [
    "collect_dl_artifact_status",
    "model_upstream_prerequisite_blockers",
    "resolve_required_artifact_sources",
]
