"""Strategy Compare artifact readiness/status planning.

This module owns deterministic artifact discovery and preparation-plan construction.
Builder execution stays in :mod:`strategy_compare_preparation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    StrategyComparisonSettings,
    StrategyPreparationAction,
    StrategyPreparationPlan,
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
from core.console_report import project_relative_display_path
from filters.breakout_quality.expected_r_calibration import (
    EXPECTED_R_CALIBRATION_METHOD,
    resolve_expected_r_calibration_paths,
    validate_expected_r_calibration_artifact,
)
from filters.breakout_quality.excess_r_calibration import (
    EXPECTED_EXCESS_R_CALIBRATION_METHOD,
    resolve_expected_excess_r_calibration_paths,
    validate_expected_excess_r_calibration_artifact,
)
from filters.breakout_quality.paths import (
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
from filters.breakout_quality.strategy_compare_dl_artifacts import (
    collect_dl_artifact_status as _collect_dl_artifact_status,
    model_upstream_prerequisite_blockers,
    resolve_required_artifact_sources as _resolve_required_artifact_sources,
)
from filters.breakout_quality.strategy_compare_sources import (
    read_json_object_or_none as _read_json,
    resolve_project_relative_path as _resolve_relative_path,
    PARAM_POLICY_SPECS,
    _load_param_source,
    _resolve_params_path,
    _validate_requested_param_policy,
)
from filters.breakout_quality.strategy_param_training import (
    FULL_ROOS_SEARCH_FIELDS,
    MIN_ROOS_SEARCH_FIELDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]



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


def _validate_expected_artifact_contract(
    actual: Any,
    expected: Any,
    *,
    field_path: str = "root",
) -> str | None:
    """Return the first subset-contract mismatch, or ``None`` when matched."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{field_path}: expected object, actual={type(actual).__name__}"
        for key, expected_value in expected.items():
            if key not in actual:
                return f"{field_path}.{key}: missing"
            mismatch = _validate_expected_artifact_contract(
                actual[key], expected_value, field_path=f"{field_path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return f"{field_path}: expected={expected!r}, actual={actual!r}"
    return None


def _validate_param_artifact(
    path: Path,
    *,
    param_policy: str,
    comparison_start: str | None = None,
    comparison_end: str | None = None,
    artifact_contract: dict[str, Any] | None = None,
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
    if artifact_contract:
        raw_payload = _read_json(path)
        if raw_payload is None:
            return False, "PARAM_CONTRACT_INVALID_JSON", policy
        mismatch = _validate_expected_artifact_contract(raw_payload, artifact_contract)
        if mismatch is not None:
            return False, f"PARAM_CONTRACT_MISMATCH: {mismatch}", policy
    return True, "READY", policy


def resolve_comparison_period(
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
        if builder.builder_type in {"extending_min_roos_stitch", "extending_full_roos_stitch"}:
            is_full = builder.builder_type == "extending_full_roos_stitch"
            expected_arm_id = "P4_EXTENDING" if is_full else "P2_EXTENDING"
            expected_search_fields = FULL_ROOS_SEARCH_FIELDS if is_full else MIN_ROOS_SEARCH_FIELDS
            if str(payload.get("builder_type") or "") != builder.builder_type:
                return False, "EXTENDING_STITCH_IDENTITY_MISMATCH", manifest_path
            if str(payload.get("arm_id") or "") != expected_arm_id:
                return False, "EXTENDING_STITCH_ARM_MISMATCH", manifest_path
            if list(payload.get("search_fields") or []) != list(expected_search_fields):
                return False, (
                    "FULL_ROOS_SEARCH_FIELDS_MISMATCH" if is_full
                    else "MIN_ROOS_SEARCH_FIELDS_MISMATCH"
                ), manifest_path
            if str(payload.get("param_policy") or "") != str(settings.param_policy):
                return False, "EXTENDING_STITCH_PARAM_POLICY_MISMATCH", manifest_path

            filename = str(PARAM_POLICY_SPECS[str(settings.param_policy)]["filename"])
            source_artifacts = dict(payload.get("source_artifacts") or {})
            for label, option_name in (
                ("historical", "historical_params_path"),
                ("current", "current_params_path"),
            ):
                raw_path = str(options.get(option_name) or "")
                configured_path = _resolve_relative_path(
                    root, raw_path.format(param_filename=filename)
                )
                if not configured_path.is_file():
                    return False, f"EXTENDING_STITCH_SOURCE_MISSING:{label}", manifest_path
                recorded = dict(source_artifacts.get(label) or {})
                if str(recorded.get("sha256") or "") != compute_file_sha256(configured_path):
                    return False, f"EXTENDING_STITCH_SOURCE_HASH_MISMATCH:{label}", manifest_path

            output_path = resolve_param_source_path(root, settings, source_id)
            if not output_path.is_file():
                return False, "EXTENDING_STITCH_OUTPUT_MISSING", manifest_path
            output_record = dict(payload.get("output_params") or {})
            if str(output_record.get("sha256") or "") != compute_file_sha256(output_path):
                return False, "EXTENDING_STITCH_OUTPUT_HASH_MISMATCH", manifest_path
            try:
                output_source = _load_param_source(output_path)
                actual_start, actual_end = get_active_param_ensemble_date_range(
                    output_source["payload"]
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return False, "EXTENDING_STITCH_OUTPUT_INVALID", manifest_path
            if (
                str(payload.get("coverage_start") or "") != actual_start
                or str(payload.get("coverage_end") or "") != actual_end
            ):
                return False, "EXTENDING_STITCH_COVERAGE_MISMATCH", manifest_path
            return True, "READY", manifest_path
        if builder.builder_type == "oos_param_freeze":
            if str(payload.get("builder_type") or "") != "oos_param_freeze":
                return False, "OOS_FREEZE_IDENTITY_MISMATCH", manifest_path
            if bool(payload.get("training_dl_enabled")):
                return False, "OOS_FREEZE_TRAINING_DL_ENABLED", manifest_path
            expected_effective = str(options.get("freeze_effective_date") or "").strip()
            expected_cutoff = str(options.get("freeze_cutoff_date") or "").strip()
            if str(payload.get("freeze_effective_date") or "") != expected_effective:
                return False, "OOS_FREEZE_EFFECTIVE_DATE_MISMATCH", manifest_path
            if str(payload.get("freeze_cutoff_date") or "") != expected_cutoff:
                return False, "OOS_FREEZE_CUTOFF_DATE_MISMATCH", manifest_path
            if str(payload.get("param_policy") or "") != str(settings.param_policy):
                return False, "OOS_FREEZE_PARAM_POLICY_MISMATCH", manifest_path
            dependency_id = str(options.get("source_param_source_id") or "").strip()
            if dependency_id not in settings.parameter_sources:
                return False, "OOS_FREEZE_SOURCE_ID_INVALID", manifest_path
            source_path = resolve_param_source_path(root, settings, dependency_id)
            if not source_path.is_file():
                return False, "OOS_FREEZE_SOURCE_MISSING", manifest_path
            source_record = dict(payload.get("source_artifact") or {})
            if str(source_record.get("sha256") or "") != compute_file_sha256(source_path):
                return False, "OOS_FREEZE_SOURCE_HASH_MISMATCH", manifest_path
            output_path = resolve_param_source_path(root, settings, source_id)
            if not output_path.is_file():
                return False, "OOS_FREEZE_OUTPUT_MISSING", manifest_path
            output_record = dict(payload.get("output_params") or {})
            if str(output_record.get("sha256") or "") != compute_file_sha256(output_path):
                return False, "OOS_FREEZE_OUTPUT_HASH_MISMATCH", manifest_path
            try:
                output_source = _load_param_source(output_path)
                actual_start, actual_end = get_active_param_ensemble_date_range(output_source["payload"])
                mapping = dict(output_source["payload"].get("params_ensemble_by_effective_date") or {})
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return False, "OOS_FREEZE_OUTPUT_INVALID", manifest_path
            if tuple(mapping) != (expected_effective,):
                return False, "OOS_FREEZE_MULTIPLE_EFFECTIVE_MEMBERS", manifest_path
            if str(payload.get("source_effective_date") or "") != expected_effective:
                return False, "OOS_FREEZE_SOURCE_EFFECTIVE_DATE_MISMATCH", manifest_path
            if (
                str(payload.get("coverage_start") or "") != actual_start
                or str(payload.get("coverage_end") or "") != actual_end
            ):
                return False, "OOS_FREEZE_COVERAGE_MISMATCH", manifest_path
            return True, "READY", manifest_path
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

        if builder.builder_type in {"binary_dl_min_roos_rolling", "selection_historical_p2"}:
            if list(payload.get("search_fields") or []) != list(MIN_ROOS_SEARCH_FIELDS):
                return False, "MIN_ROOS_SEARCH_FIELDS_MISMATCH", manifest_path
        elif builder.builder_type == "selection_historical_full_roos":
            if list(payload.get("search_fields") or []) != list(FULL_ROOS_SEARCH_FIELDS):
                return False, "FULL_ROOS_SEARCH_FIELDS_MISMATCH", manifest_path
    return True, "READY", manifest_path

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




def _collect_expected_r_calibration_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    dl_rows: dict[str, Any],
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
):
    expected_r_rows: dict[str, Any] = {}
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode != STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT:
            continue
        if not arm.dl_id:
            raise ValueError(f"Expected-PnL arm缺少runtime dl_id: {arm.arm_id}")
        options = dict(arm.dl_runtime_options or {})
        if str(options.get("expected_r_calibration_method") or "") != EXPECTED_R_CALIBRATION_METHOD:
            raise ValueError(
                f"Expected-PnL arm calibration method不支援: {arm.arm_id}/"
                f"{options.get('expected_r_calibration_method')!r}"
            )
        if options.get("preserve_k_r0") is not True:
            raise ValueError(f"Expected-PnL arm第一階段必須preserve_k_r0=True: {arm.arm_id}")
        if options.get("negative_expected_r_allowed") is not True:
            raise ValueError(
                f"Expected-PnL arm第一階段不得以Expected R負值改變K/R0: {arm.arm_id}"
            )
        fit_dl_id = str(options.get("expected_r_fit_dl_id") or "").strip()
        if not fit_dl_id or fit_dl_id not in settings.dl_sources:
            raise ValueError(f"Expected-PnL arm缺少合法fit source: {arm.arm_id}")
        runtime_dl = settings.dl_sources[arm.dl_id]
        fit_dl = settings.dl_sources[fit_dl_id]
        if fit_dl.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            raise ValueError(f"Expected-PnL fit source必須是Selection PIT: {arm.arm_id}/{fit_dl_id}")
        if (
            fit_dl.filter_id,
            fit_dl.model_architecture,
            fit_dl.experiment_profile,
        ) != (
            runtime_dl.filter_id,
            runtime_dl.model_architecture,
            runtime_dl.experiment_profile,
        ):
            raise ValueError(
                f"Expected-PnL calibration builder要求runtime/fit為同一frozen ranker identity: "
                f"{arm.arm_id}/{arm.dl_id}/{fit_dl_id}"
            )
        if settings.profile_id in {"selection_pit", "extending_window_oos", "extending_window_rolling"} and runtime_dl.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            raise ValueError(f"Selection Expected-PnL runtime必須使用Selection PIT score: {arm.arm_id}")
        if settings.profile_id == "forward_oos" and runtime_dl.score_source != SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            raise ValueError(f"Forward Expected-PnL runtime必須使用frozen OOS score: {arm.arm_id}")
        runtime_score_key = f"dl:{arm.dl_id}:forward_scores"
        fit_score_key = f"dl:{fit_dl_id}:forward_scores"
        runtime_sha = str((artifact_identities.get(runtime_score_key) or {}).get("sha256") or "")
        fit_sha = str((artifact_identities.get(fit_score_key) or {}).get("sha256") or "")
        expected_selection_start = (
            settings.start_date if settings.profile_id in {"selection_pit", "extending_window_oos", "extending_window_rolling"} else None
        )
        expected_forward_cutoff = (
            (runtime_periods.get(arm.dl_id) or (None, None))[0]
            if settings.profile_id == "forward_oos"
            else None
        )
        paths = resolve_expected_r_calibration_paths(
            root,
            filter_id=runtime_dl.filter_id,
            model_architecture=runtime_dl.model_architecture,
            experiment_profile=runtime_dl.experiment_profile,
            phase_id=settings.profile_id,
        )
        ready, calibration_status, manifest = validate_expected_r_calibration_artifact(
            root,
            filter_id=runtime_dl.filter_id,
            model_architecture=runtime_dl.model_architecture,
            experiment_profile=runtime_dl.experiment_profile,
            phase_id=settings.profile_id,
            expected_source_score_sha256=(runtime_sha or None),
            expected_fit_score_sha256=(fit_sha or None),
            expected_selection_runtime_start_date=expected_selection_start,
            expected_forward_frozen_cutoff_exclusive=expected_forward_cutoff,
        )
        upstream_ready = bool((dl_rows.get(arm.dl_id) or {}).get("ready")) and bool(
            (dl_rows.get(fit_dl_id) or {}).get("ready")
        )
        if ready and settings.preparation.reuse_ready_artifacts:
            action = "REUSE"
            description = "重用frozen MR-13E PIT Expected-R calibration工件"
            builder_type = None
        elif upstream_ready and settings.preparation.auto_prepare:
            action = "REBUILD" if paths["manifest"].exists() or paths["lookup"].exists() else "BUILD"
            if action == "REBUILD" and not settings.preparation.rebuild_stale_artifacts:
                action = "BLOCKED"
                builder_type = None
                description = "Expected-R calibration過期且config禁止自動重建"
            else:
                builder_type = "expected_r_calibration"
                description = (
                    "只用Selection PIT成熟target建立daily percentile→Expected R；"
                    "Forward以frozen OOS execution start為cutoff，不讀Forward target"
                )
        else:
            action = "BLOCKED"
            builder_type = None
            description = "Expected-R calibration上游score未就緒"
        artifact_key = f"runtime:{arm.arm_id}:expected_r_calibration"
        display_path = project_relative_display_path(paths["manifest"], project_root=root)
        actions.append(_preparation_action(
            action_id=artifact_key,
            artifact_key=artifact_key,
            action=action,
            builder_type=builder_type,
            description=description,
            path=display_path,
            dependencies=(runtime_score_key, fit_score_key) if runtime_score_key != fit_score_key else (runtime_score_key,),
            producer_work_type=(
                "existing_artifact" if action == "REUSE"
                else "strategy_compare_deterministic_rebuild" if action in {"BUILD", "REBUILD"}
                else None
            ),
            execution_priority=30,
        ))
        expected_r_rows[arm.arm_id] = {
            "ready": ready,
            "status": calibration_status,
            "action": action,
            "path": display_path,
            "lookup_path": project_relative_display_path(paths["lookup"], project_root=root),
            "fit_dl_id": fit_dl_id,
            "manifest": manifest,
        }
        artifact_identities[artifact_key] = {
            "path": display_path,
            "sha256": compute_file_sha256(paths["manifest"]) if ready and paths["manifest"].is_file() else None,
            "status": calibration_status,
        }

    return expected_r_rows

def _collect_expected_excess_r_calibration_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    dl_rows: dict[str, Any],
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
):
    expected_excess_r_rows: dict[str, Any] = {}
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode not in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        }:
            continue
        if settings.profile_id != "selection_pit":
            raise ValueError(f"Excess-Alpha第一階段只允許Selection PIT: {arm.arm_id}")
        if not arm.dl_id:
            raise ValueError(f"Excess-Alpha arm缺少runtime dl_id: {arm.arm_id}")
        options = dict(arm.dl_runtime_options or {})
        if str(options.get("expected_excess_r_calibration_method") or "") != EXPECTED_EXCESS_R_CALIBRATION_METHOD:
            raise ValueError(
                f"Excess-Alpha arm calibration method不支援: {arm.arm_id}/"
                f"{options.get('expected_excess_r_calibration_method')!r}"
            )
        if arm.dl_runtime_mode in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        }:
            if options.get("preserve_k_r0") is not True or options.get("selection_only") is not True:
                raise ValueError(f"Excess-Alpha第一階段必須preserve_k_r0/selection_only: {arm.arm_id}")
            if (
                arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL
                and options.get("constrained_solver") != "exact_branch_and_bound_v1"
            ):
                raise ValueError(f"Excess-Alpha constrained solver contract不符: {arm.arm_id}")
        else:
            if (
                options.get("preserve_k") is not True
                or options.get("preserve_r0") is not False
                or options.get("r0_minimum_repair") is not False
                or options.get("selection_only") is not True
            ):
                raise ValueError(
                    f"Excess-Alpha no-R0必須preserve_k=True/preserve_r0=False/"
                    f"r0_minimum_repair=False/selection_only=True: {arm.arm_id}"
                )
        if options.get("negative_expected_excess_r_allowed") is not True:
            raise ValueError(
                f"Excess-Alpha第一階段不得以負Expected Excess-R改變K/R0: {arm.arm_id}"
            )
        fit_dl_id = str(options.get("expected_excess_r_fit_dl_id") or "").strip()
        if not fit_dl_id or fit_dl_id not in settings.dl_sources:
            raise ValueError(f"Excess-Alpha arm缺少合法fit source: {arm.arm_id}")
        runtime_dl = settings.dl_sources[arm.dl_id]
        fit_dl = settings.dl_sources[fit_dl_id]
        if runtime_dl.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME or fit_dl.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            raise ValueError(f"Selection Excess-Alpha runtime/fit都必須使用Selection PIT score: {arm.arm_id}")
        if (
            fit_dl.filter_id,
            fit_dl.model_architecture,
            fit_dl.experiment_profile,
        ) != (
            runtime_dl.filter_id,
            runtime_dl.model_architecture,
            runtime_dl.experiment_profile,
        ):
            raise ValueError(
                f"Excess-Alpha calibration要求runtime/fit為同一frozen ranker identity: "
                f"{arm.arm_id}/{arm.dl_id}/{fit_dl_id}"
            )
        runtime_score_key = f"dl:{arm.dl_id}:forward_scores"
        fit_score_key = f"dl:{fit_dl_id}:forward_scores"
        runtime_sha = str((artifact_identities.get(runtime_score_key) or {}).get("sha256") or "")
        fit_sha = str((artifact_identities.get(fit_score_key) or {}).get("sha256") or "")
        paths = resolve_expected_excess_r_calibration_paths(
            root,
            filter_id=runtime_dl.filter_id,
            model_architecture=runtime_dl.model_architecture,
            experiment_profile=runtime_dl.experiment_profile,
            phase_id=settings.profile_id,
        )
        ready, calibration_status, manifest = validate_expected_excess_r_calibration_artifact(
            root,
            filter_id=runtime_dl.filter_id,
            model_architecture=runtime_dl.model_architecture,
            experiment_profile=runtime_dl.experiment_profile,
            phase_id=settings.profile_id,
            expected_source_score_sha256=(runtime_sha or None),
            expected_fit_score_sha256=(fit_sha or None),
            expected_selection_runtime_start_date=settings.start_date,
        )
        upstream_ready = bool((dl_rows.get(arm.dl_id) or {}).get("ready")) and bool(
            (dl_rows.get(fit_dl_id) or {}).get("ready")
        )
        if ready and settings.preparation.reuse_ready_artifacts:
            action = "REUSE"
            description = "重用frozen MR-13E PIT Expected Excess-R calibration工件"
            builder_type = None
        elif upstream_ready and settings.preparation.auto_prepare:
            action = "REBUILD" if paths["manifest"].exists() or paths["lookup"].exists() else "BUILD"
            if action == "REBUILD" and not settings.preparation.rebuild_stale_artifacts:
                action = "BLOCKED"
                builder_type = None
                description = "Expected Excess-R calibration過期且config禁止自動重建"
            else:
                builder_type = "expected_excess_r_calibration"
                description = (
                    "只用Selection PIT成熟target建立daily percentile→Expected Excess-R單調isotonic mapping；"
                    "target先扣同日daily-eligible成熟樣本平均R，不建立absolute Expected-R"
                )
        else:
            action = "BLOCKED"
            builder_type = None
            description = "Expected Excess-R calibration上游score未就緒"
        artifact_key = f"runtime:{arm.arm_id}:expected_excess_r_calibration"
        display_path = project_relative_display_path(paths["manifest"], project_root=root)
        actions.append(_preparation_action(
            action_id=artifact_key,
            artifact_key=artifact_key,
            action=action,
            builder_type=builder_type,
            description=description,
            path=display_path,
            dependencies=(runtime_score_key, fit_score_key) if runtime_score_key != fit_score_key else (runtime_score_key,),
            producer_work_type=(
                "existing_artifact" if action == "REUSE"
                else "strategy_compare_deterministic_rebuild" if action in {"BUILD", "REBUILD"}
                else None
            ),
            execution_priority=30,
        ))
        expected_excess_r_rows[arm.arm_id] = {
            "ready": ready,
            "status": calibration_status,
            "action": action,
            "path": display_path,
            "lookup_path": project_relative_display_path(paths["lookup"], project_root=root),
            "fit_dl_id": fit_dl_id,
            "manifest": manifest,
        }
        artifact_identities[artifact_key] = {
            "path": display_path,
            "sha256": compute_file_sha256(paths["manifest"]) if ready and paths["manifest"].is_file() else None,
            "status": calibration_status,
        }

    return expected_excess_r_rows

def _collect_parameter_artifact_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    required_param_sources: set[str],
    comparison_start: str | None,
    comparison_end: str | None,
    dl_model_ready: dict[str, bool],
    artifact_identities: dict[str, Any],
    actions: list[StrategyPreparationAction],
):
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
            artifact_contract=(
                None if source.artifact_contract is None else dict(source.artifact_contract)
            ),
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
        param_dependency_id = (
            ""
            if builder is None
            else str(dict(builder.options).get("source_param_source_id") or "").strip()
        )
        if param_dependency_id:
            dependency_row = parameter_rows.get(param_dependency_id)
            dependency_action = None if dependency_row is None else str(dependency_row.get("action") or "")
            upstream_ready = bool(
                upstream_ready
                and dependency_row is not None
                and dependency_action in {"REUSE", "BUILD", "REBUILD"}
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
                (
                    (
                        (
                            "合併既有historical/current Min ROOS為Extending-Window rolling schedule"
                            if builder is not None and builder.builder_type == "extending_min_roos_stitch"
                            else (
                                "合併既有historical/current Full ROOS為Extending-Window rolling schedule"
                                if builder is not None and builder.builder_type == "extending_full_roos_stitch"
                                else (
                                    "凍結2020 information cutoff當下合法策略參數供OOS Test全期間使用"
                                    if builder is not None and builder.builder_type == "oos_param_freeze"
                                    else "建立／接續Selection historical Min ROOS單階段rolling參數"
                                )
                            )
                        )
                        if builder is not None and builder.builder_type in {
                            "extending_min_roos_stitch", "extending_full_roos_stitch", "oos_param_freeze", "selection_historical_p2"
                        }
                        else "建立／接續Selection historical Full ROOS rolling參數"
                    )
                    if builder is not None and builder.builder_type in {
                        "extending_min_roos_stitch", "extending_full_roos_stitch", "oos_param_freeze",
                        "selection_historical_p2", "selection_historical_full_roos"
                    }
                    else "執行或接續config指定的策略參數訓練"
                )
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
        upstream_dependencies: tuple[str, ...] = (
            (f"param:{param_dependency_id}",) if param_dependency_id else ()
        )
        if source.trained_with_dl_id:
            upstream_candidates = (
                f"dl:{source.trained_with_dl_id}:model",
                f"dl:{source.trained_with_dl_id}:selection_pit_bundle",
                f"dl:{source.trained_with_dl_id}:forward_scores",
            )
            existing_action_keys = {item.artifact_key for item in actions}
            upstream_dependencies = tuple(dict.fromkeys((
                *upstream_dependencies,
                *tuple(key for key in upstream_candidates if key in existing_action_keys)[:1],
            )))
        actions.append(
            _preparation_action(
                action_id=f"param:{source_id}",
                artifact_key=f"param:{source_id}",
                action=action,
                builder_type=builder_type,
                description=description,
                path=display_path,
                dependencies=upstream_dependencies,
                producer_work_type=(
                    "existing_artifact"
                    if action == "REUSE"
                    else (
                        "strategy_compare_deterministic_rebuild"
                        if builder is not None and builder.builder_type == "oos_param_freeze"
                        else "strategy_parameter_optimization"
                    )
                    if action in {"BUILD", "REBUILD"}
                    else "model_training"
                    if source.trained_with_dl_id and not upstream_ready
                    else None
                ),
                execution_priority=10,
            )
        )

    return parameter_rows, resolved_parameter_paths

def collect_artifact_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings,
) -> dict[str, Any]:
    """Resolve the complete deterministic preparation status for one comparison profile."""

    root = Path(project_root).resolve()
    (
        required_param_sources,
        required_dl_sources,
        runtime_required_dl_sources,
    ) = _resolve_required_artifact_sources(settings)

    (
        dl_rows,
        artifact_identities,
        actions,
        dl_model_ready,
        runtime_periods,
    ) = _collect_dl_artifact_status(
        root=root,
        settings=settings,
        required_dl_sources=required_dl_sources,
        runtime_required_dl_sources=runtime_required_dl_sources,
    )
    expected_r_rows = _collect_expected_r_calibration_status(
        root=root,
        settings=settings,
        dl_rows=dl_rows,
        artifact_identities=artifact_identities,
        actions=actions,
    )
    expected_excess_r_rows = _collect_expected_excess_r_calibration_status(
        root=root,
        settings=settings,
        dl_rows=dl_rows,
        artifact_identities=artifact_identities,
        actions=actions,
    )

    comparison_start, comparison_end, comparison_period_source = resolve_comparison_period(
        settings=settings,
        runtime_periods=runtime_periods,
    )
    parameter_rows, resolved_parameter_paths = _collect_parameter_artifact_status(
        root=root,
        settings=settings,
        required_param_sources=required_param_sources,
        comparison_start=comparison_start,
        comparison_end=comparison_end,
        dl_model_ready=dl_model_ready,
        artifact_identities=artifact_identities,
        actions=actions,
    )

    plan = StrategyPreparationPlan.from_actions(actions)
    overall_status = plan.overall_status
    return {
        "comparison_ready": overall_status == "READY",
        "overall_status": overall_status,
        "preparation_plan": plan,
        "parameters": parameter_rows,
        "dl_sources": dl_rows,
        "expected_r_calibrations": expected_r_rows,
        "expected_excess_r_calibrations": expected_excess_r_rows,
        "artifact_identities": artifact_identities,
        "resolved_parameter_paths": resolved_parameter_paths,
        "comparison_period": (
            None
            if comparison_start is None or comparison_end is None
            else {"start": comparison_start, "end": comparison_end}
        ),
        "comparison_period_source": comparison_period_source,
    }


__all__ = [
    "collect_artifact_status",
    "model_upstream_prerequisite_blockers",
    "resolve_comparison_period",
    "resolve_param_source_path",
]
