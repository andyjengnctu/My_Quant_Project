"""策略比較前置工件依賴計畫與自動建立服務。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.strategy_comparison import (
    StrategyComparisonSettings,
    StrategyPreparationAction,
    StrategyPreparationPlan,
)
from filters.breakout_quality.artifacts import (
    compute_file_sha256,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
)
from core.console_report import project_relative_display_path
from filters.breakout_quality.export_scores import export_forward_oos_scores
from filters.breakout_quality.paths import resolve_filter_artifact_paths, resolve_filter_model_output_dir
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    load_continuous_ranker_oos_contract,
)
from filters.breakout_quality.strategy_compare_engine import (
    PARAM_POLICY_SPECS,
    _load_param_source,
    _resolve_params_path,
    _validate_requested_param_policy,
)
from filters.breakout_quality.strategy_param_training import (
    prepare_strategy_parameter_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_relative_path(root: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"設定路徑必須是專案root相對路徑: {value}")
    return (root / path).resolve()


def resolve_param_source_path(
    root: Path,
    settings: StrategyComparisonSettings,
    source_id: str,
) -> Path:
    source = settings.parameter_sources[source_id]
    if source.path_template in (None, ""):
        return _resolve_params_path(
            root=root,
            params_path=None,
            param_policy=settings.param_policy,
            allow_static_diagnostic=False,
        ).resolve()
    filename = str(PARAM_POLICY_SPECS[settings.param_policy]["filename"])
    try:
        rendered = str(source.path_template).format(param_filename=filename)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"parameter source {source_id}路徑模板只支援{{param_filename}}"
        ) from exc
    return _resolve_relative_path(root, rendered)


def _validate_param_artifact(
    path: Path,
    *,
    param_policy: str,
    comparison_start: str | None = None,
    comparison_end: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not path.is_file():
        return False, "MISSING", None
    try:
        source = _load_param_source(path)
        policy = dict(_validate_requested_param_policy(source, param_policy))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"INVALID: {type(exc).__name__}: {exc}", None
    kind = str(source.get("kind") or "")
    if kind != "rolling_active_param_ensemble":
        return False, f"INVALID_KIND: {source.get('kind')}", None
    try:
        param_start, param_end = get_active_param_ensemble_date_range(source["payload"])
    except (ValueError, KeyError, TypeError) as exc:
        return False, f"INVALID_PERIOD: {type(exc).__name__}: {exc}", policy
    policy["coverage_start"] = param_start
    policy["coverage_end"] = param_end
    if comparison_start is not None and comparison_end is not None:
        required_start = pd.Timestamp(comparison_start).normalize()
        required_end = pd.Timestamp(comparison_end).normalize()
        actual_start = pd.Timestamp(param_start).normalize()
        actual_end = pd.Timestamp(param_end).normalize()
        if actual_start > required_start or actual_end < required_end:
            return (
                False,
                "PARAM_PERIOD_MISMATCH: "
                f"params={param_start}~{param_end}, "
                f"comparison={required_start.date()}~{required_end.date()}",
                policy,
            )
    return True, "READY", policy


def _resolve_comparison_period(
    *,
    settings: StrategyComparisonSettings,
    runtime_periods: dict[str, tuple[str, str]],
) -> tuple[str | None, str | None, str]:
    if settings.start_date is not None and settings.end_date is not None:
        requested_start = pd.Timestamp(settings.start_date).normalize()
        requested_end = pd.Timestamp(settings.end_date).normalize()
        if requested_end < requested_start:
            raise ValueError("策略比較設定期間不合法")
        for dl_id, (available_start, available_end) in runtime_periods.items():
            if (
                requested_start < pd.Timestamp(available_start).normalize()
                or requested_end > pd.Timestamp(available_end).normalize()
            ):
                raise ValueError(
                    "策略比較設定期間超出DL runtime可用範圍: "
                    f"dl={dl_id}, requested={requested_start.date()}~{requested_end.date()}, "
                    f"available={available_start}~{available_end}"
                )
        return str(requested_start.date()), str(requested_end.date()), "config_explicit"
    if not runtime_periods:
        return None, None, "runtime_pending"
    starts = [pd.Timestamp(value[0]).normalize() for value in runtime_periods.values()]
    ends = [pd.Timestamp(value[1]).normalize() for value in runtime_periods.values()]
    common_start = max(starts)
    common_end = min(ends)
    if common_end < common_start:
        raise ValueError(
            "啟用DL工件沒有共同可比較期間: "
            + ", ".join(
                f"{dl_id}={period[0]}~{period[1]}"
                for dl_id, period in sorted(runtime_periods.items())
            )
        )
    return str(common_start.date()), str(common_end.date()), "dl_runtime_common_overlap"


def _validate_param_training_identity(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    source_id: str,
) -> tuple[bool, str, Path | None]:
    source = settings.parameter_sources[source_id]
    if not source.identity_manifest_path:
        if source.trained_with_dl_id:
            return False, "IDENTITY_MANIFEST_NOT_CONFIGURED", None
        return True, "NOT_REQUIRED", None
    manifest_path = _resolve_relative_path(root, source.identity_manifest_path)
    payload = _read_json(manifest_path)
    if payload is None:
        return False, "IDENTITY_MANIFEST_MISSING_OR_INVALID", manifest_path

    training_dl_enabled = payload.get("training_dl_enabled")
    expected_training_dl_enabled = bool(source.trained_with_dl_id)
    if training_dl_enabled is None:
        return False, "TRAINING_DL_IDENTITY_MISSING", manifest_path
    if bool(training_dl_enabled) != expected_training_dl_enabled:
        return False, (
            "TRAINING_DL_ENABLED_MISMATCH" if expected_training_dl_enabled
            else "TRAINING_DL_DISABLED_MISMATCH"
        ), manifest_path

    if source.trained_with_dl_id:
        dl = settings.dl_sources[source.trained_with_dl_id]
        binary_runtime = dict(payload.get("binary_runtime") or {})
        actual_filter = str(binary_runtime.get("filter_id") or payload.get("filter_id") or "")
        actual_architecture = str(
            binary_runtime.get("model_architecture")
            or payload.get("model_architecture")
            or ""
        )
        actual_profile = str(
            binary_runtime.get("experiment_profile")
            or payload.get("experiment_profile")
            or ""
        )
        if (
            actual_filter != dl.filter_id
            or actual_architecture != dl.model_architecture
            or actual_profile != dl.experiment_profile
        ):
            return False, "DL_IDENTITY_MISMATCH", manifest_path

    builder = source.builder
    if builder is not None and builder.enabled:
        options = dict(builder.options)
        expected_parameter_set = str(options.get("parameter_set") or "").upper()
        if expected_parameter_set and str(payload.get("arm_id") or "").upper() != expected_parameter_set:
            return False, "PARAMETER_SET_IDENTITY_MISMATCH", manifest_path
        expected_fields = {
            "dataset": str(settings.dataset),
            "trials_per_fold": int(options.get("trials_per_fold", 0)),
            "fixed_risk": float(options.get("fixed_risk", 0.0)),
            "max_position_cap_pct": float(options.get("max_position_cap_pct", 0.0)),
            "max_positions": int(settings.max_positions),
            "rotation": str(settings.rotation),
        }
        for field_name, expected in expected_fields.items():
            actual = payload.get(field_name)
            if isinstance(expected, float):
                try:
                    matched = abs(float(actual) - expected) <= 1e-12
                except (TypeError, ValueError):
                    matched = False
            elif isinstance(expected, int):
                try:
                    matched = int(actual) == expected
                except (TypeError, ValueError):
                    matched = False
            else:
                matched = str(actual) == expected
            if not matched:
                return False, f"TRAINING_CONFIG_MISMATCH:{field_name}", manifest_path
        if expected_training_dl_enabled and not isinstance(payload.get("binary_pit"), dict):
            return False, "BINARY_PIT_IDENTITY_MISSING", manifest_path

        baseline_path = _resolve_params_path(
            root=root,
            params_path=None,
            param_policy=settings.param_policy,
            allow_static_diagnostic=False,
        ).resolve()
        if not baseline_path.is_file():
            return False, "BASELINE_PARAMS_MISSING", manifest_path
        expected_baseline_sha = compute_file_sha256(baseline_path)
        if str(payload.get("baseline_params_sha256") or "") != expected_baseline_sha:
            return False, "BASELINE_PARAMS_IDENTITY_MISMATCH", manifest_path
    return True, "READY", manifest_path

def _preparation_action(
    *,
    action_id: str,
    artifact_key: str,
    action: str,
    builder_type: str | None,
    description: str,
    path: str,
) -> StrategyPreparationAction:
    return StrategyPreparationAction(
        action_id=action_id,
        artifact_key=artifact_key,
        action=action,
        builder_type=builder_type,
        description=description,
        path=path,
    )


def collect_artifact_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    required_param_sources = {arm.param_source for arm in settings.enabled_arms}
    required_dl_sources = {
        arm.dl_id for arm in settings.enabled_arms if arm.dl_enabled and arm.dl_id
    }
    for source_id in required_param_sources:
        trained_with = settings.parameter_sources[source_id].trained_with_dl_id
        if trained_with:
            required_dl_sources.add(trained_with)

    dl_rows: dict[str, Any] = {}
    artifact_identities: dict[str, Any] = {}
    actions: list[StrategyPreparationAction] = []
    dl_model_ready: dict[str, bool] = {}
    runtime_periods: dict[str, tuple[str, str]] = {}

    for dl_id in settings.dl_sources:
        if dl_id not in required_dl_sources:
            continue
        source = settings.dl_sources[dl_id]
        artifacts = resolve_filter_artifact_paths(
            root,
            source.filter_id,
            source.model_architecture,
            source.experiment_profile,
        )

        if source.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
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
                runtime_periods[dl_id] = (
                    str(continuous_contract.execution_start),
                    str(continuous_contract.available_through),
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                model_status = f"CONTINUOUS_MODEL_OR_REPORT_INVALID ({type(exc).__name__})"
                runtime_status = f"CONTINUOUS_OOS_SCORES_INVALID ({type(exc).__name__})"
            dl_model_ready[dl_id] = model_ready
            score_path = (
                continuous_contract.score_path
                if continuous_contract is not None
                else resolve_filter_model_output_dir(root, source.filter_id, source.model_architecture, source.experiment_profile) / "continuous_ranker_scores.csv"
            )
            report_path = (
                continuous_contract.report_path
                if continuous_contract is not None
                else resolve_filter_model_output_dir(
                    root, source.filter_id, source.model_architecture, source.experiment_profile
                ) / CONTINUOUS_RANKER_REPORT_FILENAME
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
                status = "READY" if ready else (
                    model_status if key in {"model", "manifest", "report"} else runtime_status
                )
                action = "REUSE" if ready and settings.preparation.reuse_ready_artifacts else "BLOCKED"
                research_label = f"{dl_id}/{source.experiment_profile}"
                description = (
                    f"重用既有{research_label} continuous research工件"
                    if ready and key in {"model", "manifest", "report"}
                    else f"重用既有{research_label} frozen OOS continuous scores"
                    if ready
                    else (
                        f"缺少或無效；請由模型研究入口建立{research_label}工件，"
                        "策略比較不得自動重訓"
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
                    )
                )
            dl_rows[dl_id] = {
                "ready": bool(model_ready and runtime_ready),
                "status": "READY" if model_ready and runtime_ready else "NOT_READY",
                "identity": source.as_dict(),
                "files": file_rows,
            }
            continue

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
        dl_model_ready[dl_id] = model_ready

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
                    builder = source.forward_scores_builder
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
            actions.append(
                _preparation_action(
                    action_id=f"dl:{dl_id}:{key}",
                    artifact_key=f"dl:{dl_id}:{key}",
                    action=action,
                    builder_type=builder_type,
                    description=description,
                    path=file_rows[key]["path"],
                )
            )
        dl_rows[dl_id] = {
            "ready": bool(model_ready and runtime_ready),
            "status": "READY" if model_ready and runtime_ready else "NOT_READY",
            "identity": source.as_dict(),
            "files": file_rows,
        }

    comparison_start, comparison_end, comparison_period_source = _resolve_comparison_period(
        settings=settings,
        runtime_periods=runtime_periods,
    )

    parameter_rows: dict[str, Any] = {}
    resolved_parameter_paths: dict[str, Path] = {}
    for source_id in settings.parameter_sources:
        if source_id not in required_param_sources:
            continue
        source = settings.parameter_sources[source_id]
        path = resolve_param_source_path(root, settings, source_id)
        resolved_parameter_paths[source_id] = path
        artifact_ready, artifact_status, policy = _validate_param_artifact(
            path,
            param_policy=settings.param_policy,
            comparison_start=comparison_start,
            comparison_end=comparison_end,
        )
        identity_ready, identity_status, identity_path = _validate_param_training_identity(
            root=root,
            settings=settings,
            source_id=source_id,
        )
        ready = bool(artifact_ready and identity_ready)
        status = "READY" if ready else (
            identity_status if artifact_ready and not identity_ready else artifact_status
        )
        builder = source.builder
        upstream_ready = (
            True
            if not source.trained_with_dl_id
            else bool(dl_model_ready.get(source.trained_with_dl_id))
        )
        if ready and settings.preparation.reuse_ready_artifacts:
            action = "REUSE"
            description = "重用既有策略參數工件"
            builder_type = None
        elif ready:
            if (
                settings.preparation.auto_prepare
                and settings.preparation.rebuild_stale_artifacts
                and builder is not None
                and builder.enabled
            ):
                action = "REBUILD"
                description = "config禁止重用，重新建立策略參數工件"
                builder_type = builder.builder_type
            else:
                action = "BLOCKED"
                description = "config禁止重用且未允許重新建立策略參數"
                builder_type = None
        elif (
            upstream_ready
            and settings.preparation.auto_prepare
            and builder is not None
            and builder.enabled
        ):
            action = "REBUILD" if (path.exists() or (identity_path is not None and identity_path.exists())) else "BUILD"
            if action == "REBUILD" and not settings.preparation.rebuild_stale_artifacts:
                action = "BLOCKED"
            description = (
                "執行或接續config指定的策略參數訓練"
                if action in {"BUILD", "REBUILD"}
                else "參數工件過期且config禁止自動重建"
            )
            builder_type = builder.builder_type if action != "BLOCKED" else None
        else:
            action = "BLOCKED"
            description = "缺少參數工件且無可用builder或上游模型工件"
            builder_type = None
        sha256 = compute_file_sha256(path) if path.is_file() else None
        display_path = project_relative_display_path(path, project_root=root)
        artifact_identities[f"param:{source_id}"] = {
            "path": display_path,
            "sha256": sha256,
            "identity_status": identity_status,
            "coverage_start": None if policy is None else policy.get("coverage_start"),
            "coverage_end": None if policy is None else policy.get("coverage_end"),
            "coverage_status": artifact_status,
        }
        parameter_rows[source_id] = {
            "ready": ready,
            "status": status,
            "action": action,
            "path": display_path,
            "sha256": sha256,
            "selector": None if policy is None else policy.get("selector"),
            "identity_status": identity_status,
            "coverage_start": None if policy is None else policy.get("coverage_start"),
            "coverage_end": None if policy is None else policy.get("coverage_end"),
            "identity_manifest_path": (
                None
                if identity_path is None
                else project_relative_display_path(identity_path, project_root=root)
            ),
        }
        actions.append(
            _preparation_action(
                action_id=f"param:{source_id}",
                artifact_key=f"param:{source_id}",
                action=action,
                builder_type=builder_type,
                description=description,
                path=display_path,
            )
        )

    action_names = {item.action for item in actions}
    if "BLOCKED" in action_names:
        overall_status = "BLOCKED"
    elif action_names & {"BUILD", "REBUILD"}:
        overall_status = "PREPARABLE"
    else:
        overall_status = "READY"
    plan = StrategyPreparationPlan(
        overall_status=overall_status,
        actions=tuple(actions),
    )
    return {
        "comparison_ready": overall_status == "READY",
        "overall_status": overall_status,
        "preparation_plan": plan,
        "parameters": parameter_rows,
        "dl_sources": dl_rows,
        "artifact_identities": artifact_identities,
        "resolved_parameter_paths": resolved_parameter_paths,
        "comparison_period": (
            None
            if comparison_start is None or comparison_end is None
            else {"start": comparison_start, "end": comparison_end}
        ),
        "comparison_period_source": comparison_period_source,
    }


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
    if action.builder_type == "binary_dl_risk_only_rolling":
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
            trials_per_fold=int(options.get("trials_per_fold", 200)),
            max_positions=int(settings.max_positions),
            rotation=str(settings.rotation),
            fixed_risk=float(options.get("fixed_risk", 0.01)),
            max_position_cap_pct=float(options.get("max_position_cap_pct", 0.30)),
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


def prepare_strategy_comparison_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    requested_plan: StrategyPreparationPlan = status["preparation_plan"]
    if requested_plan.blocked:
        raise RuntimeError("策略比較前置計畫包含BLOCKED工件，無法自動完成")
    if requested_plan.overall_status == "READY":
        return status
    if not settings.preparation.auto_prepare:
        raise RuntimeError("目前config已關閉auto_prepare")

    current = status
    executed_signatures: set[tuple[str, str, str | None, str]] = set()
    max_waves = max(2, len(requested_plan.actions) + 2)
    for _wave in range(max_waves):
        plan: StrategyPreparationPlan = current["preparation_plan"]
        if plan.overall_status == "READY":
            current["requested_preparation_plan"] = requested_plan
            return current
        if plan.blocked:
            blocked = [
                item.artifact_key
                for item in plan.actions
                if item.action == "BLOCKED"
            ]
            raise RuntimeError(
                "策略比較前置依賴於重新規劃後變成BLOCKED: "
                + ", ".join(blocked)
            )

        wave_actions = [
            item for item in plan.actions if item.action in {"BUILD", "REBUILD"}
        ]
        if not wave_actions:
            raise RuntimeError("策略比較前置狀態非READY，但沒有可執行BUILD／REBUILD動作")
        executed_this_wave = 0
        for action in wave_actions:
            signature = (
                action.artifact_key,
                action.action,
                action.builder_type,
                action.path,
            )
            if signature in executed_signatures:
                continue
            print(f"\n[前置] {action.description}")
            try:
                _execute_preparation_action(root=root, settings=settings, action=action)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"前置步驟失敗: {action.artifact_key} | action={action.action} | "
                    f"path={action.path} | {type(exc).__name__}: {exc}"
                ) from exc
            executed_signatures.add(signature)
            executed_this_wave += 1

        current = collect_artifact_status(project_root=root, settings=settings)
        current["requested_preparation_plan"] = requested_plan
        if current["comparison_ready"]:
            return current
        if executed_this_wave == 0:
            remaining = [
                f"{item.artifact_key}:{item.action}"
                for item in current["preparation_plan"].actions
                if item.action != "REUSE"
            ]
            raise RuntimeError(
                "前置工件重新規劃後沒有進展: " + ", ".join(remaining)
            )

    remaining = [
        f"{item.artifact_key}:{item.action}"
        for item in current["preparation_plan"].actions
        if item.action != "REUSE"
    ]
    raise RuntimeError(
        "前置工件超過最大依賴波次仍未READY: " + ", ".join(remaining)
    )


__all__ = [
    "collect_artifact_status",
    "prepare_strategy_comparison_artifacts",
    "resolve_param_source_path",
]
