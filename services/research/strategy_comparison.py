"""Config-driven Breakout Quality strategy performance comparison orchestration."""

from __future__ import annotations

from core.file_integrity import load_json_object_or_none as _read_json

import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import pandas as pd

from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
)
from core.strategy_compare_policy import (
    get_strategy_comparison_settings,
    get_strategy_runtime_integration_settings,
)
from core.runtime_utils import get_taipei_now
from core.file_integrity import atomic_write_json
from core.display_common import format_elapsed
from core.display import _display_width
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
    StrategyComparisonArm,
    StrategyComparisonSettings,
    StrategyDLSource,
    StrategyPreparationAction,
    StrategyPreparationPlan,
    resolve_strategy_comparison_arm_param_policy,
    resolve_strategy_parameter_consumption_mode,
    strategy_comparison_fingerprint,
)
from core.research_report_contract import (
    require_approved_report_contract, section_contract,
    persistent_report_contract_fingerprint,
)
from core.report_metrics import (
    EXECUTION_STRATEGY_RESULT_METRICS,
    CORE_STRATEGY_RESULT_METRICS,
    PORTFOLIO_RESULT_METRICS,
    TRADE_RESULT_METRICS,
)
from core.report_style import best_worst_signals, styled_signal, styled_workflow_status
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_SCORE_RANKING,
    STRATEGY_COMPARE_SCHEMA_VERSION as STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
)
from filters.breakout_quality.strategy_compare_sources import (
    resolve_project_relative_path as _resolve_relative_path,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
)
from filters.breakout_quality.strategy_compare_diagnostics import (
    backfill_pair_r_conversion_diagnostic,
    backfill_pair_selection_diagnostics,
    backfill_pair_upside_realization_diagnostic,
    build_strategy_diagnostics,
    pair_upside_realization_refresh_required,
    render_strategy_diagnostics_markdown,
    render_mfe_safety_geometry_table,
    render_strategy_selection_quality_table,
    render_upside_survival_summary_table,
)
from filters.breakout_quality.strategy_compare_reporting import (
    materialize_strategy_pair_readable_report,
    render_strategy_pair_simple_report,
)
from services.research.strategy_compare_benchmark import (
    resolve_strategy_compare_benchmark_yearly,
)
from services.research.strategy_compare_pair_integrity import (
    PAIR_FILE_INTEGRITY_FIELD,
    build_completed_pair_file_integrity,
    completed_pair_file_integrity_matches,
)
from services.research.strategy_compare_execution import (
    run_strategy_compare_active_arm,
    run_strategy_compare_baseline_arm,
)
from filters.breakout_quality.strategy_compare_plan import ResolvedComparisonPlan
from filters.breakout_quality.strategy_result_state import (
    RESULT_ACTION_REUSE,
    RESULT_ACTION_WAIT,
    result_state_from_mapping,
)
from filters.breakout_quality.strategy_compare_runtime import (
    _arm_runtime_spec,
    _execution_pairs,
    _standalone_baseline_arms,
)
from services.research.strategy_compare_reuse import (
    _apply_completed_pair_dependency_waivers,
    _apply_completed_pair_frozen_score_reuse,
    _artifact_identity_sha,
    _collect_replay_cache_status,
    _completed_pair_pinned_continuous_score,
    _continuous_score_provenance_entries,
    _current_pair_cache_fingerprint,
    _file_sha256,
    _find_reusable_baseline_source,
    _find_reusable_pair,
    _find_reusable_pair_with_archived_source,
    _historical_continuous_score_provenance_entries,
    _pair_cache_fingerprint_from_payload,
    _pair_cache_required_files,
    _pair_group_id,
    _resolve_completed_pair_continuous_score_binding,
    _resolve_pair_pinned_score_path,
)
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    resolve_filter_model_output_dir,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
)
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from filters.breakout_quality.strategy_rule_policies import (
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)
from services.research.strategy_compare_preparation import (
    collect_preparation_status,
    prepare_strategy_comparison_artifacts,
)
from services.research.strategy_compare_preparation_status import (
    resolve_arm_parameter_evaluation_identities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 9


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, _json_native(payload))






def resolve_comparison_plan(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> ResolvedComparisonPlan:
    """Resolve preparation, reuse, provenance and period exactly once.

    UI rendering and replay execution consume this finalized plan.  Compatibility
    callers may still use :func:`collect_artifact_status`, which returns a detached
    status dictionary derived from the same plan.
    """

    root = Path(project_root).resolve()
    current = settings or get_strategy_comparison_settings()
    status = collect_preparation_status(project_root=root, settings=current)
    status["config_fingerprint"] = strategy_comparison_fingerprint(
        current,
        artifact_identities=status["artifact_identities"],
    )
    replay_cache = _collect_replay_cache_status(
        root=root,
        settings=current,
        status=status,
    )
    status = _apply_completed_pair_dependency_waivers(
        settings=current,
        status=status,
        replay_cache=replay_cache,
    )
    status = _apply_completed_pair_frozen_score_reuse(
        root=root,
        settings=current,
        status=status,
        replay_cache=replay_cache,
    )
    # Frozen completed-pair provenance can restore a comparison period that was not
    # available during the first preparation pass.  Resolve evaluation-bound parameter
    # identities and result states once more from the finalized status so every UI /
    # executor consumes the same canonical state.
    period = dict(status.get("comparison_period") or {})
    status["resolved_arm_parameter_identities"] = resolve_arm_parameter_evaluation_identities(
        project_root=root,
        settings=current,
        resolved_arm_parameter_paths=dict(status.get("resolved_arm_parameter_paths") or {}),
        comparison_start=str(period.get("start") or "") or None,
        comparison_end=str(period.get("end") or "") or None,
    )
    replay_cache = _collect_replay_cache_status(
        root=root, settings=current, status=status
    )
    status = _apply_completed_pair_dependency_waivers(
        settings=current, status=status, replay_cache=replay_cache
    )
    return ResolvedComparisonPlan.from_status(
        settings=current,
        project_root=root,
        status=status,
    )


def collect_artifact_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    """Compatibility view of the canonical resolved comparison plan."""

    return resolve_comparison_plan(
        project_root=project_root,
        settings=settings,
    ).status_dict()


def _strategy_arm_display_sort_key(arm: StrategyComparisonArm) -> tuple[int, int, str]:
    arm_id = str(arm.arm_id or "").strip()
    numeric = arm_id[1:] if arm_id.startswith("C") else ""
    if numeric.isdigit():
        return (0, int(numeric), arm_id)
    return (1, 0, arm_id)


def _display_ordered_strategy_arms(settings: StrategyComparisonSettings) -> tuple[StrategyComparisonArm, ...]:
    """Return human-report arm order without changing scientific/execution order."""

    return tuple(sorted(settings.enabled_arms, key=_strategy_arm_display_sort_key))


def _contrast_display_name(settings: StrategyComparisonSettings, item) -> str:
    """Human-readable contrast label; contrast_id remains an internal stable key."""

    left_arm = settings.arms.get(str(item.left))
    right_arm = settings.arms.get(str(item.right))
    left_name = str(item.left) if left_arm is None else f"{item.left} {left_arm.name}"
    right_name = str(item.right) if right_arm is None else f"{item.right} {right_arm.name}"
    return f"{left_name} vs {right_name}"


def render_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
    status: dict[str, Any] | None = None,
) -> str:
    root = Path(project_root).resolve()
    current = settings or get_strategy_comparison_settings()
    current_status = status or collect_artifact_status(
        project_root=root,
        settings=current,
    )
    arm_rows = [
        (
            "ON" if arm.enabled else "OFF",
            arm.arm_id,
            arm.name,
            arm.param_source,
            resolve_strategy_comparison_arm_param_policy(current, arm),
            arm.rule_policy,
            "DL-on" if arm.dl_enabled else "DL-off",
            arm.description,
        )
        for arm in _display_ordered_strategy_arms(current)
    ]
    contrast_rows = [
        (
            "ON" if item.enabled else "OFF",
            _contrast_display_name(current, item),
            item.left,
            item.right,
            f"{item.description}｜internal_id={item.contrast_id}",
        )
        for item in current.enabled_contrasts
    ]
    plan = current_status.get("preparation_plan")
    display_overall_status = current_status["overall_status"]

    artifact_rows = []
    for source_id, row in current_status["parameters"].items():
        artifact_rows.append((f"param:{source_id}", row["status"], row["action"], row["path"]))
        if row.get("identity_manifest_path"):
            artifact_rows.append(
                (
                    f"param:{source_id}:identity",
                    row["identity_status"],
                    row["action"],
                    row["identity_manifest_path"],
                )
            )
    model_action_by_key = {}
    if isinstance(plan, StrategyPreparationPlan):
        model_action_by_key = {item.artifact_key: item for item in plan.actions}
    for dl_id, row in current_status["dl_sources"].items():
        for key, file_row in row["files"].items():
            artifact_key = f"dl:{dl_id}:{key}"
            plan_item = model_action_by_key.get(artifact_key)
            display_action = (
                plan_item.action if plan_item is not None else file_row["action"]
            )
            artifact_rows.append(
                (artifact_key, file_row["status"], display_action, file_row["path"])
            )
    return "\n\n".join(
        (
            render_title("策略績效比較設定與工件狀態"),
            render_key_values(
                (
                    ("設定檔", "config/strategy_compare.py"),
                    ("比較階段", f"{current.profile_label} ({current.profile_id})"),
                    ("Dataset", current.dataset),
                    (
                        "期間",
                        (
                            f"{current_status['comparison_period']['start']} ～ "
                            f"{current_status['comparison_period']['end']}"
                            if current_status.get("comparison_period")
                            else f"{current.start_date or 'artifact start'} ～ {current.end_date or 'artifact end'}"
                        ),
                    ),
                    (
                        "Param policy",
                        " / ".join(dict.fromkeys(
                            resolve_strategy_comparison_arm_param_policy(current, arm)
                            for arm in current.enabled_arms
                        )),
                    ),
                    ("Max positions", current.max_positions),
                    ("Rotation", current.rotation),
                    ("Config fingerprint", current_status["config_fingerprint"]),
                    ("比較狀態", display_overall_status),
                    ("策略比較自動前置", "on" if current.preparation.auto_prepare else "off"),
                    ("模型權重前置", "Strategy Compare執行時自動偵測／補建"),
                )
            ),
            render_section("1. 比較對象"),
            render_table(
                ("開關", "編號", "名稱", "參數來源", "Param policy", "Rules", "DL", "用途"),
                arm_rows,
            ),
            render_section("2. 差異比較"),
            render_table(
                ("開關", "比較", "左側", "右側", "用途"),
                contrast_rows,
            ),
            render_section("3. 前置工件與預計動作"),
            render_table(("工件", "狀態", "預計動作", "路徑"), artifact_rows),
        )
    )



def _single_line_execution_plan_cell(value: object) -> str:
    """Keep all visible cell content on one output line, without abbreviation."""

    return " ".join(str(value).splitlines())


def render_strategy_execution_plan_surface(
    *,
    title: str,
    metadata_rows: tuple[tuple[object, object], ...],
    action_rows: list[tuple[object, object, object]],
) -> str:
    """Render full execution-plan labels/reasons in one physical line per action."""

    styled_metadata = []
    for label, value in metadata_rows:
        rendered = value
        if str(label) == "整體狀態":
            rendered = styled_workflow_status(value)
        styled_metadata.append((label, rendered))
    styled_actions = [
        (
            styled_workflow_status(action),
            _single_line_execution_plan_cell(item),
            _single_line_execution_plan_cell(description),
        )
        for action, item, description in action_rows
    ]
    return "\n\n".join(
        (
            render_title(title),
            render_key_values(tuple(styled_metadata)),
            render_table(
                ("動作", "項目", "說明"),
                styled_actions,
            ),
        )
    )

def render_execution_plan(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
) -> str:
    plan: StrategyPreparationPlan = status["preparation_plan"]
    ordered_actions = sorted(
        plan.actions,
        key=lambda item: (int(item.execution_priority), item.artifact_key),
    )
    blocked = plan.blocked
    rows = []
    for item in ordered_actions:
        action = item.action
        description = item.description
        if blocked and action in {"BUILD", "REBUILD", "RESUME", "MIGRATE", "DERIVE"}:
            action = "NOT_RUN"
            description = f"{description}｜整體計畫已BLOCKED，本次不執行"
        rows.append((action, item.artifact_key, description))

    replay_cache = dict(status.get("replay_cache") or {})
    cached_pairs = dict(replay_cache.get("pairs") or {})
    arm_states = dict(replay_cache.get("arm_states") or {})
    for arm in _display_ordered_strategy_arms(settings):
        if blocked:
            rows.append(("NOT_RUN", arm.arm_id, f"{arm.name}｜上游前置工件BLOCKED，本次不執行"))
            continue
        state = result_state_from_mapping(dict(arm_states[arm.arm_id]))
        action = state.action
        if action == RESULT_ACTION_REUSE:
            description = (
                f"{arm.name}｜重用identity一致的completed scientific result"
            )
        elif action == RESULT_ACTION_WAIT:
            description = f"{arm.name}｜{state.reason}"
        else:
            description = arm.name
        rows.append((action, arm.arm_id, description))
    stale_diagnostic_arms = [
        arm.arm_id
        for arm in settings.enabled_arms
        if (
            not blocked
            and arm.dl_enabled
            and isinstance(cached_pairs.get(arm.arm_id), dict)
            and pair_upside_realization_refresh_required(
                Path(cached_pairs[arm.arm_id]["source_pair_dir"])
            )
        )
    ]
    if stale_diagnostic_arms:
        rows.append((
            "REFRESH",
            "Diagnostics | Upside Realization",
            "既有pair replay保持REUSE；只回填path-conversion診斷與重產報表："
            + ",".join(stale_diagnostic_arms),
        ))
    # The canonical Strategy Compare produces one aggregate human report, not
    # a separate execution job for each declarative scientific contrast.
    # Keep enabled_contrasts in the report/result contract for traceability;
    # they must not flood the normal RUN/REUSE execution plan.
    rows.append((
        "NOT_RUN" if blocked else "REPORT",
        "策略績效比較（綜合報表）",
        (
            "前置工件BLOCKED，本次不產生報表"
            if blocked else "共用正式績效報表；差異比較只保存為研究工件"
        ),
    ))
    return render_strategy_execution_plan_surface(
        title="本次執行計畫",
        metadata_rows=(
            ("整體狀態", plan.overall_status),
            ("設定檔", "config/strategy_compare.py"),
            ("比較階段", f"{settings.profile_label} ({settings.profile_id})"),
            ("Config fingerprint", status["config_fingerprint"]),
        ),
        action_rows=rows,
    )


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any, *, unit: str = "", digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}{unit}"




def _exclusive_selection_r_from_trade_files(
    pair_dir: Path,
    *,
    active_trades_filename: str,
) -> float:
    baseline_path = pair_dir / "no_filter_trades.csv"
    active_path = pair_dir / active_trades_filename
    if not baseline_path.is_file() or not active_path.is_file():
        raise FileNotFoundError(
            "缺少直接選擇R所需交易工件: "
            f"{baseline_path.name}, {active_path.name}"
        )
    baseline = reconstruct_round_trips(
        pd.read_csv(baseline_path, encoding="utf-8-sig"),
        scenario="no_filter",
    )
    active = reconstruct_round_trips(
        pd.read_csv(active_path, encoding="utf-8-sig"),
        scenario="active",
    )
    left = baseline.set_index("match_key", drop=False) if not baseline.empty else baseline
    right = active.set_index("match_key", drop=False) if not active.empty else active
    left_keys = set(left.index.astype(str)) if not left.empty else set()
    right_keys = set(right.index.astype(str)) if not right.empty else set()
    left_only = left.loc[list(sorted(left_keys - right_keys))] if left_keys - right_keys else baseline.iloc[0:0]
    right_only = right.loc[list(sorted(right_keys - left_keys))] if right_keys - left_keys else active.iloc[0:0]
    left_r = pd.to_numeric(left_only.get("r_multiple"), errors="coerce").fillna(0.0).sum() if not left_only.empty else 0.0
    right_r = pd.to_numeric(right_only.get("r_multiple"), errors="coerce").fillna(0.0).sum() if not right_only.empty else 0.0
    return float(right_r - left_r)


def _load_direct_selection_r(
    pair_dir: Path,
    *,
    root: Path,
    active_trades_filename: str,
) -> float:
    path = pair_dir / "trade_attribution.json"
    if path.is_file():
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(
                "交易歸因工件格式無效: "
                + project_relative_display_path(path, project_root=root)
            )
        value = _metric(
            dict(payload.get("r_attribution") or {}),
            "exclusive_selection_delta_r",
        )
        if value is None:
            raise ValueError("交易歸因缺少exclusive_selection_delta_r")
        return float(value)
    return _exclusive_selection_r_from_trade_files(
        pair_dir,
        active_trades_filename=active_trades_filename,
    )






def _standalone_baseline_group_id(
    settings: StrategyComparisonSettings,
    arm: StrategyComparisonArm,
) -> str:
    param_policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
    return f"{arm.param_source}__{param_policy.replace('-', '_')}__{arm.rule_policy}__dl_off_baseline"




def _assert_same_shared_baseline(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    arm_id: str,
) -> None:
    ignored = {
        "arm_id",
        "param_source",
        "rule_policy",
        "dl_enabled",
        "dl_id",
        "dl_runtime_mode",
        "direct_selection_delta_r",
    }
    keys = {
        key
        for key in (set(existing) | set(candidate)) - ignored
        if not str(key).startswith("resource_aware_selector_timing_")
    }
    for key in sorted(keys):
        left = existing.get(key)
        right = candidate.get(key)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(
                    f"多DL比較的共用基準不一致: arm={arm_id}, key={key}, "
                    f"first={left}, repeated={right}"
                )
        elif left != right:
            raise ValueError(
                f"多DL比較的共用基準不一致: arm={arm_id}, key={key}, "
                f"first={left!r}, repeated={right!r}"
            )


def _scenario_payloads(
    pair_payloads: dict[str, dict[str, Any]],
    direct_r: dict[str, float],
    *,
    settings: StrategyComparisonSettings,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    enabled_ids = {arm.arm_id for arm in settings.enabled_arms}
    for group_id, pair in pair_payloads.items():
        param_source, rule_policy, off_arm, on_arm = pair["arm_contract"]
        if off_arm.arm_id in enabled_ids:
            baseline = {
                **dict(pair["payload"].get("no_filter") or {}),
                "arm_id": off_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": False,
                "dl_id": None,
                "dl_runtime_mode": None,
                "direct_selection_delta_r": 0.0,
            }
            existing = output.get(off_arm.arm_id)
            if existing is None:
                output[off_arm.arm_id] = baseline
            else:
                _assert_same_shared_baseline(
                    existing,
                    baseline,
                    arm_id=off_arm.arm_id,
                )
        if on_arm is None:
            continue
        if on_arm.arm_id in enabled_ids:
            runtime_spec = _arm_runtime_spec(on_arm)
            output[on_arm.arm_id] = {
                **dict(pair["payload"].get(runtime_spec["active_key"]) or {}),
                "arm_id": on_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": True,
                "dl_id": on_arm.dl_id,
                "dl_runtime_mode": on_arm.dl_runtime_mode,
                "direct_selection_delta_r": float(direct_r[group_id]),
            }
    return output


def _report_reference_arm_id(settings: StrategyComparisonSettings) -> str | None:
    integration = get_strategy_runtime_integration_settings()
    if settings.profile_id == integration.selection_profile_id:
        arm_id = integration.selection_candidate_arm_id
    elif settings.profile_id == integration.forward_profile_id:
        arm_id = integration.forward_candidate_arm_id
    else:
        return None
    return arm_id if arm_id in {arm.arm_id for arm in settings.enabled_arms} else None


def _value_with_signal(text: str, signal: str, *, target: str) -> str:
    if text == "-":
        return text
    return styled_signal(text, signal, target=target)


def _metric_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
    metrics: tuple[Any, ...],
    target: str = "plain",
) -> str:
    display_arms = _display_ordered_strategy_arms(settings)
    signals_by_metric = {
        metric.key: best_worst_signals(
            {arm.arm_id: _metric(scenarios[arm.arm_id], metric.key) for arm in display_arms},
            preference=metric.preference,
        )
        for metric in metrics
    }
    rows = []
    for arm in display_arms:
        payload = scenarios[arm.arm_id]
        values = []
        for metric in metrics:
            text = _fmt(payload.get(metric.key), unit=metric.unit, digits=metric.digits)
            signal = signals_by_metric.get(metric.key, {}).get(arm.arm_id)
            if signal:
                text = _value_with_signal(text, signal, target=target)
            values.append(text)
        rows.append((arm.arm_id, arm.name, *values))
    return render_table(
        ("編號", "比較對象", *(metric.label for metric in metrics)),
        rows,
    )


def render_strategy_core_result_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    return _metric_table(
        scenarios,
        settings=settings,
        metrics=CORE_STRATEGY_RESULT_METRICS,
        target=target,
    )




def render_strategy_execution_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    return _metric_table(
        scenarios,
        settings=settings,
        metrics=EXECUTION_STRATEGY_RESULT_METRICS,
        target=target,
    )




def render_safety_gate_sensitivity_table(
    scenarios: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    """Render the controlled Raw-Safety percentile sensitivity curve only."""

    gate_arms = []
    for arm in settings.enabled_arms:
        options = dict(arm.dl_runtime_options or {})
        if str(options.get("safety_gate") or "") != "same_day_orderable_percentile_gte_v1":
            continue
        try:
            cutoff = float(options["safety_percentile_cutoff"])
        except (KeyError, TypeError, ValueError):
            continue
        gate_arms.append((cutoff, arm))
    gate_arms.sort(key=lambda item: (item[0], item[1].arm_id))
    if len(gate_arms) < 2:
        return "目前沒有至少兩個Raw Safety gate cutoff，Sensitivity不適用。"

    geometry = dict(diagnostics.get("mfe_safety_geometry") or {})
    geometry_arms = dict(geometry.get("arms") or {}) if str(geometry.get("status")) == "AVAILABLE" else {}
    upside_by_arm = {
        str(row.get("arm_id") or ""): dict(row)
        for row in list(diagnostics.get("upside_realization") or [])
    }
    rows = []
    for cutoff, arm in gate_arms:
        scenario = dict(scenarios.get(arm.arm_id) or {})
        geo = dict(geometry_arms.get(arm.arm_id) or {})
        upside = dict(upside_by_arm.get(arm.arm_id) or {})
        rows.append((
            arm.arm_id,
            f">={cutoff:.2f}",
            _fmt(scenario.get("avg_exposure_pct"), unit="%"),
            _fmt(geo.get("high_mfe_high_safety_pct"), unit="%"),
            _fmt(geo.get("high_mfe_low_safety_pct"), unit="%"),
            _fmt(geo.get("high_mfe_total_pct"), unit="%"),
            _fmt(geo.get("high_safety_total_pct"), unit="%"),
            _fmt(upside.get("full_horizon_mfe_mean_r"), unit="R"),
            _fmt(upside.get("full_horizon_adverse_to_peak_mean_r"), unit="R"),
            _fmt(upside.get("realized_mean_r"), unit="R"),
            _fmt(scenario.get("return_over_max_drawdown")),
        ))
    table = render_table(
        ("編號", "Raw Safety gate", "平均曝險", "HM/HS", "HM/LS", "High-MFE", "High-Safety", "Full-MFE", "Adverse", "Realized EV", "RoMD"),
        rows,
    )
    return table + "\nGate curve只比較C71/C72/C73同一MR-13R No-K/No-R0 formulation；C64/C66保留於主表作reference，不混入threshold curve。"


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = _metric(left, key)
    b = _metric(right, key)
    return None if a is None or b is None else a - b


def render_strategy_yearly_values_table(
    by_id: dict[str, dict[int, float | None]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    enabled_ids = tuple(arm.arm_id for arm in _display_ordered_strategy_arms(settings))
    years = sorted({year for values in by_id.values() for year in values})
    rows = []
    for year in years:
        signals = best_worst_signals(
            {arm_id: dict(by_id.get(arm_id) or {}).get(year) for arm_id in enabled_ids},
            preference="higher",
        )
        values = []
        for arm_id in enabled_ids:
            value = dict(by_id.get(arm_id) or {}).get(year)
            text = _fmt(value, unit="%")
            signal = signals.get(arm_id)
            if signal:
                text = _value_with_signal(text, signal, target=target)
            values.append(text)
        rows.append((year, *values))
    return render_table(("年度", *enabled_ids), rows)


def _yearly_values_by_id(
    pair_payloads: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> dict[str, dict[int, float | None]]:
    enabled_ids = tuple(arm.arm_id for arm in settings.enabled_arms)
    by_id: dict[str, dict[int, float | None]] = {arm_id: {} for arm_id in enabled_ids}
    for pair in pair_payloads.values():
        _param_source, _rule_policy, off_arm, on_arm = pair["arm_contract"]
        for row in pair["payload"].get("yearly") or []:
            year = int(row["year"])
            if off_arm.arm_id in by_id:
                value = row.get("no_filter_return_pct")
                existing = by_id[off_arm.arm_id].get(year)
                if existing is not None and value is not None:
                    if not math.isclose(
                        float(existing), float(value), rel_tol=0.0, abs_tol=1e-10
                    ):
                        raise ValueError(
                            "多DL比較的共用基準年度報酬不一致: "
                            f"arm={off_arm.arm_id}, year={year}, "
                            f"first={existing}, repeated={value}"
                        )
                else:
                    by_id[off_arm.arm_id][year] = value
            if on_arm is not None and on_arm.arm_id in by_id:
                runtime_spec = _arm_runtime_spec(on_arm)
                by_id[on_arm.arm_id][year] = row.get(runtime_spec["yearly_key"])
    return by_id




def _comparison_period(pair_payloads: dict[str, dict[str, Any]]) -> Any:
    periods = {
        json.dumps(
            dict(pair["payload"].get("metadata") or {}).get("comparison_period"),
            sort_keys=True,
        )
        for pair in pair_payloads.values()
    }
    if len(periods) != 1:
        raise ValueError(f"策略比較期間不一致: {periods}")
    return dict(next(iter(pair_payloads.values()))["payload"].get("metadata") or {}).get(
        "comparison_period"
    )


def render_strategy_run_execution_table(
    execution_summary: dict[str, Any],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    """Render neutral wall-time metadata for one Strategy Compare run.

    Performance timing is execution metadata only.  It must never participate in
    strategy best/worst coloring or alter the scientific fingerprint.
    """

    del target  # Table is intentionally neutral for both console and Markdown.
    arm_rows = dict(execution_summary.get("arms") or {})
    rows: list[tuple[object, ...]] = []
    for arm in _display_ordered_strategy_arms(settings):
        row = dict(arm_rows.get(arm.arm_id) or {})
        rows.append(
            (
                arm.arm_id,
                arm.name,
                str(row.get("action") or "-"),
                format_elapsed(float(row.get("elapsed_sec") or 0.0)),
                format_elapsed(float(row.get("cumulative_sec") or 0.0)),
            )
        )
    post_elapsed = float(execution_summary.get("post_processing_elapsed_sec") or 0.0)
    total_elapsed = float(execution_summary.get("total_elapsed_sec") or 0.0)
    rows.append(
        (
            "REPORT",
            "診斷／報表整理",
            "REPORT",
            format_elapsed(post_elapsed),
            format_elapsed(total_elapsed),
        )
    )
    rows.append(
        (
            "TOTAL",
            "總耗時",
            "TOTAL",
            format_elapsed(total_elapsed),
            format_elapsed(total_elapsed),
        )
    )
    return render_table(("編號", "比較對象", "動作", "本次耗時", "累計時間"), rows)


def _required_numeric(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy SOP v2 INCOMPLETE: {context} missing numeric evidence") from exc
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f"Strategy SOP v2 INCOMPLETE: {context} non-finite evidence")
    return result


def _contract_strategy_metric_table(
    section_id: str, scenarios: dict[str, dict[str, Any]], *,
    settings: StrategyComparisonSettings, target: str = "plain",
) -> str:
    from core.research_report_contract import table_contract
    table = table_contract("strategy.standard_sop", section_id, section_id)
    display_arms = _display_ordered_strategy_arms(settings)
    metric_columns = [c for c in table.columns if c.key not in {"arm_id", "arm_name"}]
    strengths = {
        c.key: best_worst_signals(
            {a.arm_id: _required_numeric(dict(scenarios.get(a.arm_id) or {}).get(c.key),
                                         context=f"{a.arm_id}.{c.key}") for a in display_arms},
            preference=c.preference,
        ) for c in metric_columns
    }
    rows = []
    for arm in display_arms:
        scenario = dict(scenarios.get(arm.arm_id) or {})
        cells = []
        for column in metric_columns:
            numeric = _required_numeric(scenario.get(column.key), context=f"{arm.arm_id}.{column.key}")
            value = _fmt(numeric, digits=column.digits or 0, unit=column.unit)
            if column.format_kind.startswith("signed") and numeric > 0:
                value = "+" + value
            signal = strengths[column.key].get(arm.arm_id)
            cells.append(_value_with_signal(value, signal, target=target) if signal else value)
        rows.append((arm.arm_id, arm.name, *cells))
    return render_table(table.headers, rows, alignments=table.alignments)


def _contract_strategy_yearly_table(
    yearly_by_id: dict[str, dict[int, Any]], *,
    benchmark_yearly: list[dict[str, Any]] | None,
    year_is_complete: dict[int, bool] | None,
    settings: StrategyComparisonSettings, target: str = "plain",
) -> str:
    """S4 uses existing canonical portfolio 0050 rows; renderer never derives prices."""
    from core.research_report_contract import table_contract
    if not benchmark_yearly or not year_is_complete:
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050/year completeness evidence absent")
    table = table_contract("strategy.standard_sop", "yearly_results", "yearly_results")
    arms = _display_ordered_strategy_arms(settings)
    years = sorted({int(y) for values in yearly_by_id.values() for y in values})
    if not years or set(years) != {int(y) for y in year_is_complete}:
        raise ValueError("Strategy SOP v2 INCOMPLETE: yearly cohort/complete years mismatch")
    lookup = {}
    for raw in benchmark_yearly:
        row = dict(raw)
        year = int(row["year"])
        if year in lookup:
            raise ValueError(f"Strategy SOP v2 duplicate 0050 year: {year}")
        if not row.get("start_date") or not row.get("end_date"):
            raise ValueError(f"Strategy SOP v2 INCOMPLETE: 0050/{year} missing interval evidence")
        lookup[year] = row
    if set(lookup) != set(years):
        raise ValueError("Strategy SOP v2 INCOMPLETE: 0050/strategy yearly years mismatch")
    # The common benchmark is a reference price return; broker-cash PnL remains
    # in S1-S3. Different fee semantics are disclosed in immutable v2 definition.
    rows = []
    for year in years:
        _required_numeric(lookup[year].get("year_return_pct"), context=f"0050.{year}")
        values = []
        comparisons = {}
        for arm in arms:
            if year not in yearly_by_id.get(arm.arm_id, {}):
                raise ValueError(f"Strategy SOP v2 INCOMPLETE: {arm.arm_id}/{year} missing")
            comparisons[arm.arm_id] = _required_numeric(yearly_by_id[arm.arm_id][year], context=f"{arm.arm_id}.{year}")
        signals = best_worst_signals(comparisons, preference="higher")
        for arm in arms:
            value = comparisons[arm.arm_id]
            rendered = f"{value:+.2f}%"
            signal = signals.get(arm.arm_id)
            values.append(_value_with_signal(rendered, signal, target=target) if signal else rendered)
        rows.append((str(year) + ("" if year_is_complete[year] else "*"),
                     f"{float(lookup[year]['year_return_pct']):+.2f}%", *values))
    return render_table((table.columns[0].label, table.columns[1].label,
                         *(arm.arm_id for arm in arms)), rows)


def _strategy_year_completeness(pair_payloads: dict[str, dict[str, Any]]) -> dict[int, bool]:
    by_year: dict[int, bool] = {}
    for pair in pair_payloads.values():
        for row in pair["payload"].get("yearly") or ():
            year = int(row["year"])
            if "is_full_year" not in row:
                raise ValueError(f"Strategy SOP v2 INCOMPLETE: missing is_full_year for {year}")
            complete = bool(row["is_full_year"])
            if year in by_year and by_year[year] != complete:
                raise ValueError(f"Strategy SOP v2 complete-year flag inconsistent: {year}")
            by_year[year] = complete
    return by_year


def _strategy_benchmark_yearly(pair_payloads: dict[str, dict[str, Any]]) -> list[dict]:
    evidence = None
    for pair in pair_payloads.values():
        payload = dict(pair["payload"])
        rows = list(payload.get("benchmark_yearly") or [])
        if not rows:
            raise ValueError("Strategy SOP v2 INCOMPLETE: source missing canonical 0050 yearly evidence")
        if evidence is not None and rows != evidence:
            raise ValueError("Strategy SOP v2 INCOMPLETE: baseline 0050 source disagreement")
        evidence = rows
    return list(evidence or [])


def render_strategy_aggregate_report(
    *,
    settings: StrategyComparisonSettings,
    comparison_period: Any,
    fingerprint: str,
    scenarios: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    yearly_by_id: dict[str, dict[int, Any]],
    benchmark_yearly: list[dict[str, Any]] | None = None,
    year_is_complete: dict[int, bool] | None = None,
    target: str = "plain",
    title: str = "策略績效比較",
    fingerprint_label: str = "Config fingerprint",
    extra_metadata: tuple[tuple[object, object], ...] = (),
    execution_summary: dict[str, Any] | None = None,
) -> str:
    """Render the canonical cross-arm Strategy Compare human report.

    Single-seed OOS/Rolling and Multi-seed robustness consume this same top-level
    renderer.  The four performance sections are canonical.  Single-seed runs
    may add the neutral execution-summary section; robustness keeps its own
    method-specific sections after the shared performance surface.
    """

    # Reject unapproved report drift before rendering any persistent SOP surface.
    require_approved_report_contract("strategy.standard_sop")

    metadata_rows = (
        ("期間", comparison_period),
        ("Dataset", settings.dataset),
        (
            "Param policy",
            " / ".join(dict.fromkeys(
                resolve_strategy_comparison_arm_param_policy(settings, arm)
                for arm in settings.enabled_arms
            )),
        ),
        ("Max positions", settings.max_positions),
        ("Rotation", settings.rotation),
        (fingerprint_label, fingerprint),
        ("比較設定", "config/strategy_compare.py"),
        *tuple(extra_metadata),
    )
    if not scenarios:
        raise ValueError("Strategy SOP v2 INCOMPLETE: no scenarios")
    if not any(bool(value) for value in (year_is_complete or {}).values()):
        raise ValueError("Strategy SOP v2 INCOMPLETE: Worst Full Year has no complete year")
    sections = [render_title(title), render_key_values(metadata_rows)]
    for section_id in ("core_performance", "trade_quality", "capital_execution"):
        sections.extend((
            render_section(section_contract("strategy.standard_sop", section_id).display_title),
            _contract_strategy_metric_table(section_id, scenarios, settings=settings, target=target),
        ))
    sections.extend((
        render_section(section_contract("strategy.standard_sop", "yearly_results").display_title),
        _contract_strategy_yearly_table(
            yearly_by_id, benchmark_yearly=benchmark_yearly,
            year_is_complete=year_is_complete, settings=settings, target=target,
        ),
        "* = Partial calendar year; 0050 is the canonical adjusted-price reference (price return, not strategy broker-cash equity).",
    ))
    # Runtime duration is displayed by execution progress, not a scientific chapter.
    return "\n\n".join(sections).rstrip() + "\n"


def _render_report(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    pair_payloads: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    execution_summary: dict[str, Any] | None = None,
    target: str = "plain",
) -> str:
    return render_strategy_aggregate_report(
        settings=settings,
        comparison_period=_comparison_period(pair_payloads),
        fingerprint=str(status["config_fingerprint"]),
        scenarios=scenarios,
        diagnostics=diagnostics,
        yearly_by_id=_yearly_values_by_id(pair_payloads, settings=settings),
        benchmark_yearly=_strategy_benchmark_yearly(pair_payloads),
        year_is_complete=_strategy_year_completeness(pair_payloads),
        execution_summary=execution_summary,
        target=target,
    )



def _run_directory(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    fingerprint: str,
) -> tuple[Path, Path]:
    output_root = _resolve_relative_path(root, settings.output_root)
    enabled_ids = "-".join(arm.arm_id for arm in settings.enabled_arms)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"{timestamp}_{enabled_ids}_{fingerprint}"
    latest_dir = output_root / "latest"
    return run_dir, latest_dir


def run_strategy_comparison(
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
    status: dict[str, Any] | None = None,
    resolved_plan: ResolvedComparisonPlan | None = None,
    auto_prepare: bool = True,
    settings: StrategyComparisonSettings | None = None,
    producer_handlers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = settings or get_strategy_comparison_settings()
    # Authorization precedes dependency preparation and strategy replay.
    require_approved_report_contract("strategy.standard_sop")
    if status is not None and resolved_plan is not None:
        raise ValueError("status與resolved_plan不可同時提供")
    if resolved_plan is None:
        if status is None:
            resolved_plan = resolve_comparison_plan(project_root=root, settings=settings)
        else:
            resolved_plan = ResolvedComparisonPlan.from_status(
                settings=settings,
                project_root=root,
                status=status,
            )
    elif resolved_plan.settings.as_dict() != settings.as_dict():
        raise ValueError("resolved_plan與settings不一致")
    status = resolved_plan.status_dict()
    requested_fingerprint = resolved_plan.config_fingerprint
    requested_plan = resolved_plan.preparation_plan
    if status["overall_status"] == "BLOCKED":
        print("\n" + render_status(project_root=root, settings=settings, status=status))
        raise RuntimeError(
            "目前啟用比較存在Research dependency graph判定為不可自動補建的工件。"
        )
    if status["overall_status"] == "PREPARABLE":
        if not auto_prepare:
            raise RuntimeError("目前工件可自動準備，但本次已停用auto_prepare")
        prepare_strategy_comparison_artifacts(
            project_root=root,
            settings=settings,
            status=status,
            status_refresher=lambda: collect_artifact_status(
                project_root=root, settings=settings
            ),
            producer_handlers=producer_handlers,
        )
        resolved_plan = resolve_comparison_plan(
            project_root=root,
            settings=settings,
        )
        status = resolved_plan.status_dict()
        status["requested_preparation_plan"] = requested_plan
        if resolved_plan.overall_status != "READY":
            raise RuntimeError("前置完成後正式比較狀態仍非READY")

    # READY has already passed ResolvedComparisonPlan.validate_contract().
    # Replay must consume this exact finalized plan instead of recollecting state.
    comparison_period = dict(status.get("comparison_period") or {})
    comparison_start = str(comparison_period.get("start") or "")
    comparison_end = str(comparison_period.get("end") or "")
    if not comparison_start or not comparison_end:
        raise RuntimeError("正式比較缺少已解析的共同comparison period")

    replay_cache = dict(status.get("replay_cache") or {})
    if not replay_cache:
        raise RuntimeError("ResolvedComparisonPlan缺少replay cache")

    run_dir, latest_dir = _run_directory(
        root=root,
        settings=settings,
        fingerprint=status["config_fingerprint"],
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    replay_started = time.perf_counter()
    arm_execution_timing: dict[str, dict[str, Any]] = {}
    pair_payloads: dict[str, dict[str, Any]] = {}
    direct_r: dict[str, float] = {}
    pair_execution: dict[str, dict[str, Any]] = {}
    cached_pairs = dict(replay_cache.get("pairs") or {})
    cached_baseline_groups = dict(replay_cache.get("baseline_groups") or {})
    arm_states = dict(replay_cache.get("arm_states") or {})
    baseline_sources: dict[str, Path] = {
        str(group_key): Path(str(row["source_pair_dir"])).resolve()
        for group_key, row in cached_baseline_groups.items()
        if isinstance(row, dict) and row.get("source_pair_dir")
    }

    for off_arm in _standalone_baseline_arms(settings):
        off_param_policy = resolve_strategy_comparison_arm_param_policy(settings, off_arm)
        group_key = f"{off_arm.param_source}::{off_param_policy}::{off_arm.rule_policy}"
        group_id = _standalone_baseline_group_id(settings, off_arm)
        pair_dir = run_dir / "pairs" / group_id
        arm_state = result_state_from_mapping(dict(arm_states[off_arm.arm_id]))
        baseline_reuse_source = (
            arm_state.source_dir
            if settings.preparation.reuse_shared_baseline
            and arm_state.action == RESULT_ACTION_REUSE
            else None
        )
        arm_started = time.perf_counter()
        baseline_action = arm_state.action
        if quiet and baseline_action == "RUN":
            print(
                f"[RUN] {off_arm.arm_id} {off_arm.name} "
                f"| total={format_elapsed(arm_started - replay_started)}"
            )
        baseline_payload = run_strategy_compare_baseline_arm(
            settings=settings,
            arm=off_arm,
            project_root=root,
            params_path=str(status["resolved_arm_parameter_paths"][off_arm.arm_id]),
            output_dir=pair_dir,
            comparison_start=comparison_start,
            comparison_end=comparison_end,
            quiet=quiet,
            baseline_reuse_dir=baseline_reuse_source,
            param_evaluation_mode=resolve_strategy_parameter_consumption_mode(
                settings.parameter_sources[off_arm.param_source]
            ),
        )
        pair_payloads[group_id] = {
            "arm_contract": (
                off_arm.param_source, off_arm.rule_policy, off_arm, None
            ),
            "payload": baseline_payload,
        }
        arm_finished = time.perf_counter()
        arm_elapsed = arm_finished - arm_started
        arm_execution_timing[off_arm.arm_id] = {
            "action": baseline_action,
            "elapsed_sec": arm_elapsed,
            "cumulative_sec": arm_finished - replay_started,
        }
        pair_execution[off_arm.arm_id] = {
            "action": baseline_action,
            "source_pair_dir": (
                None
                if baseline_reuse_source is None
                else project_relative_display_path(
                    baseline_reuse_source, project_root=root
                )
            ),
            "current_pair_dir": project_relative_display_path(
                pair_dir, project_root=root
            ),
            "standalone_dl_off_baseline": True,
        }
        if quiet:
            action = "REUSE" if baseline_reuse_source is not None else "DONE"
            print(
                f"[{action}] {off_arm.arm_id} {off_arm.name} "
                f"| elapsed={format_elapsed(arm_elapsed)} "
                f"| total={format_elapsed(arm_finished - replay_started)}"
            )
        baseline_sources[group_key] = pair_dir

    for param_source, rule_policy, off_arm, on_arm in _execution_pairs(settings):
        if not on_arm.dl_id:
            raise ValueError(f"DL-on arm缺少dl_id: {on_arm.arm_id}")
        dl: StrategyDLSource = settings.dl_sources[on_arm.dl_id]
        runtime_spec = _arm_runtime_spec(on_arm)
        group_id = _pair_group_id(
            param_source=param_source,
            rule_policy=rule_policy,
            dl_id=on_arm.dl_id,
            dl_runtime_mode=str(on_arm.dl_runtime_mode or ""),
            arm_id=on_arm.arm_id,
        )
        pair_dir = run_dir / "pairs" / group_id
        on_param_policy = resolve_strategy_comparison_arm_param_policy(settings, on_arm)
        baseline_group_key = f"{param_source}::{on_param_policy}::{rule_policy}"
        cache_entry = cached_pairs.get(on_arm.arm_id)
        arm_state = result_state_from_mapping(dict(arm_states[on_arm.arm_id]))

        if arm_state.action == RESULT_ACTION_REUSE:
            if not isinstance(cache_entry, dict) or not arm_state.source_dir:
                raise RuntimeError(
                    f"{on_arm.arm_id} canonical state=REUSE但缺少validated pair evidence"
                )
            arm_started = time.perf_counter()
            source_pair_dir = arm_state.source_dir
            if pair_dir.exists():
                shutil.rmtree(pair_dir)
            shutil.copytree(source_pair_dir, pair_dir)
            # Between planning and copy the cache may have been replaced or
            # damaged. Recheck the COPIED bytes before any diagnostic backfill
            # or performance read; a stale REUSE decision must not publish them.
            source_run_payload = _read_json(
                source_pair_dir.parent.parent / "strategy_comparison.json"
            )
            source_group_id = str(cache_entry.get("source_group_id") or "")
            source_pair_payload = (
                dict(source_run_payload.get("pairs") or {}).get(source_group_id)
                if isinstance(source_run_payload, dict) else None
            )
            if (
                not source_group_id
                or source_pair_dir.name != source_group_id
                or not isinstance(source_pair_payload, dict)
                or _read_json(pair_dir / "strategy_comparison.json") != source_pair_payload
                or not completed_pair_file_integrity_matches(
                    run_dir=source_pair_dir.parent.parent,
                    run_payload=source_run_payload,
                    pair_id=source_group_id,
                    pair_dir=pair_dir,
                    required_paths=_pair_cache_required_files(pair_dir, on_arm=on_arm),
                )
            ):
                raise RuntimeError(
                    "Strategy Compare cache copy與已驗證 completed pair 不一致，拒絕 REUSE"
                )
            pair_payload = _read_json(pair_dir / "strategy_comparison.json")
            if not isinstance(pair_payload, dict):
                raise RuntimeError(
                    "已命中Strategy Compare cache但pair JSON無法讀取: "
                    + project_relative_display_path(pair_dir, project_root=root)
                )
            pair_metadata = dict(pair_payload.get("metadata") or {})
            pair_metadata.update({
                "output_scope": "reused_pair_cache",
                "output_dir": project_relative_display_path(
                    pair_dir, project_root=root
                ),
                "cache_reused_from": project_relative_display_path(
                    source_pair_dir, project_root=root
                ),
            })
            pair_payload["metadata"] = pair_metadata
            backfill_pair_selection_diagnostics(
                pair_payload,
                pair_dir=pair_dir,
                project_root=root,
                active_trades_filename=runtime_spec["active_trades_filename"],
            )
            backfill_pair_r_conversion_diagnostic(
                pair_payload,
                pair_dir=pair_dir,
                active_trades_filename=runtime_spec["active_trades_filename"],
            )
            backfill_pair_upside_realization_diagnostic(
                pair_payload,
                pair_dir=pair_dir,
                project_root=root,
                active_trades_filename=runtime_spec["active_trades_filename"],
            )
            _write_json(pair_dir / "strategy_comparison.json", pair_payload)
            materialize_strategy_pair_readable_report(
                pair_payload,
                output_dir=pair_dir,
            )
            pair_payloads[group_id] = {
                "arm_contract": (param_source, rule_policy, off_arm, on_arm),
                "payload": pair_payload,
            }
            direct_r[group_id] = _load_direct_selection_r(
                pair_dir,
                root=root,
                active_trades_filename=runtime_spec["active_trades_filename"],
            )
            baseline_sources[baseline_group_key] = pair_dir
            if off_arm.arm_id not in arm_execution_timing:
                arm_execution_timing[off_arm.arm_id] = {
                    "action": "REUSE",
                    "elapsed_sec": 0.0,
                    "cumulative_sec": time.perf_counter() - replay_started,
                }
                if quiet:
                    print(
                        f"[REUSE] {off_arm.arm_id} {off_arm.name} "
                        f"| elapsed={format_elapsed(0.0)} "
                        f"| total={format_elapsed(time.perf_counter() - replay_started)}"
                    )
            arm_finished = time.perf_counter()
            arm_elapsed = arm_finished - arm_started
            arm_execution_timing[on_arm.arm_id] = {
                "action": "REUSE",
                "elapsed_sec": arm_elapsed,
                "cumulative_sec": arm_finished - replay_started,
            }
            pair_execution[on_arm.arm_id] = {
                "action": "REUSE",
                "pair_fingerprint": cache_entry.get("fingerprint"),
                "source_pair_dir": project_relative_display_path(
                    source_pair_dir, project_root=root
                ),
                "current_pair_dir": project_relative_display_path(
                    pair_dir, project_root=root
                ),
                "shared_baseline_reused": True,
                "source_artifact_mode": cache_entry.get(
                    "source_artifact_mode", "current_artifacts"
                ),
            }
            if quiet:
                print(
                    f"[REUSE] {on_arm.arm_id} {on_arm.name} "
                    f"| elapsed={format_elapsed(arm_elapsed)} "
                    f"| total={format_elapsed(arm_finished - replay_started)}"
                )
            else:
                print(
                    f"[{on_arm.arm_id}] REUSE 已完成正式pair："
                    + project_relative_display_path(
                        source_pair_dir,
                        project_root=root,
                    )
                )
                print("\n" + render_strategy_pair_simple_report(pair_payload))
                print_artifact_paths(
                    (("策略比較簡易報表", pair_dir / "strategy_comparison.md"),),
                    project_root=root,
                )
            continue

        all_off = rule_policy == "all_off"
        baseline_reuse_source = (
            baseline_sources.get(baseline_group_key)
            if settings.preparation.reuse_shared_baseline
            and runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else None
        )
        arm_started = time.perf_counter()
        baseline_already_accounted = off_arm.arm_id in arm_execution_timing
        active_wall_started: float | None = None
        if quiet and baseline_already_accounted:
            print(
                f"[RUN] {on_arm.arm_id} {on_arm.name} "
                f"| total={format_elapsed(arm_started - replay_started)}"
            )

        def _pair_progress(event: str, detail: dict[str, Any]) -> None:
            nonlocal active_wall_started
            now = time.perf_counter()
            if event == "baseline_start" and not baseline_already_accounted:
                if quiet:
                    action = "REUSE" if bool(detail.get("reused")) else "RUN"
                    print(
                        f"[{action}] {off_arm.arm_id} {off_arm.name} "
                        f"| total={format_elapsed(now - replay_started)}"
                    )
                return
            if event == "baseline_done" and not baseline_already_accounted:
                elapsed = float(detail.get("elapsed_sec") or 0.0)
                action = "REUSE" if bool(detail.get("reused")) else "RUN"
                arm_execution_timing[off_arm.arm_id] = {
                    "action": action,
                    "elapsed_sec": elapsed,
                    "cumulative_sec": now - replay_started,
                }
                if quiet:
                    done_label = "REUSE" if action == "REUSE" else "DONE"
                    print(
                        f"[{done_label}] {off_arm.arm_id} {off_arm.name} "
                        f"| elapsed={format_elapsed(elapsed)} "
                        f"| total={format_elapsed(now - replay_started)}"
                    )
                return
            if event == "active_start":
                active_wall_started = now if not baseline_already_accounted else arm_started
                if quiet and not baseline_already_accounted:
                    print(
                        f"[RUN] {on_arm.arm_id} {on_arm.name} "
                        f"| total={format_elapsed(now - replay_started)}"
                    )

        ranking_options_extra: dict[str, Any] = {}
        if (
            on_arm.dl_runtime_mode
            == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
        ):
            ranking_options_extra["expected_r_calibration_path"] = str(
                (status.get("expected_r_calibrations") or {})[on_arm.arm_id]["lookup_path"]
            )
        elif on_arm.dl_runtime_mode in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        }:
            ranking_options_extra["expected_excess_r_calibration_path"] = str(
                (status.get("expected_excess_r_calibrations") or {})[on_arm.arm_id]["lookup_path"]
            )
        pair_payload = run_strategy_compare_active_arm(
            settings=settings,
            arm=on_arm,
            project_root=root,
            params_path=str(status["resolved_arm_parameter_paths"][on_arm.arm_id]),
            output_dir=pair_dir,
            comparison_start=comparison_start,
            comparison_end=comparison_end,
            param_evaluation_mode=resolve_strategy_parameter_consumption_mode(
                settings.parameter_sources[on_arm.param_source]
            ),
            score_overrides=(status.get("continuous_score_overrides") or {}),
            ranking_options_extra=ranking_options_extra,
            baseline_reuse_dir=baseline_reuse_source,
            quiet=quiet,
            progress_callback=_pair_progress,
            capture_execution_diagnostics=(
                runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            ),
        )
        backfill_pair_r_conversion_diagnostic(
            pair_payload,
            pair_dir=pair_dir,
            active_trades_filename=runtime_spec["active_trades_filename"],
        )
        backfill_pair_upside_realization_diagnostic(
            pair_payload,
            pair_dir=pair_dir,
            project_root=root,
            active_trades_filename=runtime_spec["active_trades_filename"],
        )
        _write_json(pair_dir / "strategy_comparison.json", pair_payload)
        materialize_strategy_pair_readable_report(
            pair_payload,
            output_dir=pair_dir,
        )
        pair_payloads[group_id] = {
            "arm_contract": (param_source, rule_policy, off_arm, on_arm),
            "payload": pair_payload,
        }
        direct_r[group_id] = _load_direct_selection_r(
            pair_dir,
            root=root,
            active_trades_filename=runtime_spec["active_trades_filename"],
        )
        baseline_sources[baseline_group_key] = pair_dir
        arm_finished = time.perf_counter()
        effective_active_started = (
            arm_started if baseline_already_accounted else (active_wall_started or arm_started)
        )
        arm_elapsed = arm_finished - effective_active_started
        if off_arm.arm_id not in arm_execution_timing:
            # Defensive fallback if an engine implementation omitted progress callbacks.
            engine_timing = dict(dict(pair_payload.get("metadata") or {}).get("execution_timing") or {})
            baseline_elapsed = float(engine_timing.get("baseline_elapsed_sec") or 0.0)
            arm_execution_timing[off_arm.arm_id] = {
                "action": str(engine_timing.get("baseline_action") or "RUN"),
                "elapsed_sec": baseline_elapsed,
                "cumulative_sec": max(0.0, arm_finished - replay_started - arm_elapsed),
            }
        arm_execution_timing[on_arm.arm_id] = {
            "action": "RUN",
            "elapsed_sec": arm_elapsed,
            "cumulative_sec": arm_finished - replay_started,
        }
        pair_execution[on_arm.arm_id] = {
            "action": "RUN",
            "pair_fingerprint": _current_pair_cache_fingerprint(
                settings=settings,
                status=status,
                off_arm=off_arm,
                on_arm=on_arm,
            ),
            "source_pair_dir": None,
            "current_pair_dir": project_relative_display_path(
                pair_dir, project_root=root
            ),
            "shared_baseline_reused": baseline_reuse_source is not None,
            "shared_baseline_source": (
                None
                if baseline_reuse_source is None
                else project_relative_display_path(
                    baseline_reuse_source,
                    project_root=root,
                )
            ),
        }
        if quiet:
            print(
                f"[DONE] {on_arm.arm_id} {on_arm.name} "
                f"| elapsed={format_elapsed(arm_elapsed)} "
                f"| total={format_elapsed(arm_finished - replay_started)}"
            )

    # Report-only v2 evidence refresh. Historical completed pairs were already
    # integrity-validated and copied into this run; never mutate their source.
    # A missing 0050 yearly sidecar does not invalidate a completed scientific
    # replay or justify re-running it.
    for group_id, pair in pair_payloads.items():
        pair_payload = pair["payload"]
        if pair_payload.get("benchmark_yearly"):
            continue
        pair_dir = run_dir / "pairs" / group_id
        pair_payload["benchmark_yearly"] = resolve_strategy_compare_benchmark_yearly(
            pair_payload=pair_payload,
            pair_dir=pair_dir,
            project_root=root,
            dataset=settings.dataset,
            comparison_start=comparison_start,
            comparison_end=comparison_end,
        )
        _write_json(pair_dir / "strategy_comparison.json", pair_payload)

    post_processing_started = time.perf_counter()
    scenarios = _scenario_payloads(pair_payloads, direct_r, settings=settings)
    # Hypothesis-specific dashboards belong to explicit read-only Audit only.
    # Persist core Strategy SOP v2 without a permanent diagnostic sidecar.
    diagnostics: dict[str, Any] = {}
    execution_summary = {
        "arms": arm_execution_timing,
        "post_processing_elapsed_sec": time.perf_counter() - post_processing_started,
        "total_elapsed_sec": time.perf_counter() - replay_started,
    }
    report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        diagnostics=diagnostics,
        execution_summary=execution_summary,
        target="markdown",
    )
    console_report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        diagnostics=diagnostics,
        execution_summary=execution_summary,
        target="console",
    )
    execution_summary["post_processing_elapsed_sec"] = (
        time.perf_counter() - post_processing_started
    )
    execution_summary["total_elapsed_sec"] = time.perf_counter() - replay_started
    report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        diagnostics=diagnostics,
        execution_summary=execution_summary,
        target="markdown",
    )
    console_report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        diagnostics=diagnostics,
        execution_summary=execution_summary,
        target="console",
    )
    report_path = run_dir / "strategy_comparison.md"
    json_path = run_dir / "strategy_comparison.json"
    manifest_path = run_dir / "manifest.json"
    report_path.write_text(report, encoding="utf-8")
    pair_file_integrity = build_completed_pair_file_integrity({
        group_id: run_dir / "pairs" / group_id for group_id in pair_payloads
    })
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "report_id": "strategy.standard_sop",
        "report_version": int(require_approved_report_contract("strategy.standard_sop").version),
        "report_contract_fingerprint": persistent_report_contract_fingerprint("strategy.standard_sop"),
        "status": "COMPLETED",
        PAIR_FILE_INTEGRITY_FIELD: pair_file_integrity,
        "created_at": get_taipei_now().isoformat(),
        "config_fingerprint": status["config_fingerprint"],
        "requested_config_fingerprint": requested_fingerprint,
        "settings": settings.as_dict(),
        "artifact_identities": status["artifact_identities"],
        **(
            {"research_dataset_generation_identity": dict(status["research_dataset_generation_identity"])}
            if isinstance(status.get("research_dataset_generation_identity"), dict)
            and status.get("research_dataset_generation_identity")
            else {}
        ),
        "comparison_period": comparison_period,
        "comparison_period_source": status.get("comparison_period_source"),
        "requested_preparation_plan": requested_plan.as_dict(),
        "final_preparation_plan": status["preparation_plan"].as_dict(),
        "scenarios": scenarios,
        "contrasts": {
            item.contrast_id: {
                "left": item.left,
                "right": item.right,
                "description": item.description,
            }
            for item in settings.enabled_contrasts
        },
        "pairs": {
            key: value["payload"] for key, value in pair_payloads.items()
        },
        "pair_execution": pair_execution,
        "execution_summary": execution_summary,
        "replay_reuse_policy": {
            "reuse_completed_results": bool(
                settings.preparation.reuse_completed_results
            ),
            "reuse_shared_baseline": bool(
                settings.preparation.reuse_shared_baseline
            ),
        },
    }
    _write_json(json_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "report_id": payload["report_id"],
            "report_version": payload["report_version"],
            "report_contract_fingerprint": payload["report_contract_fingerprint"],
            PAIR_FILE_INTEGRITY_FIELD: pair_file_integrity,
            "created_at": payload["created_at"],
            "config_path": "config/strategy_compare.py",
            "config_fingerprint": status["config_fingerprint"],
            "requested_config_fingerprint": requested_fingerprint,
            "enabled_arms": [arm.as_dict() for arm in settings.enabled_arms],
            "enabled_contrasts": [
                item.as_dict() for item in settings.enabled_contrasts
            ],
            "artifact_identities": status["artifact_identities"],
            "comparison_period": comparison_period,
            "comparison_period_source": status.get("comparison_period_source"),
            "preparation_policy": settings.preparation.as_dict(),
            "requested_preparation_plan": requested_plan.as_dict(),
            "final_preparation_plan": status["preparation_plan"].as_dict(),
            "pair_execution": pair_execution,
            "execution_summary": execution_summary,
            "replay_reuse_policy": payload["replay_reuse_policy"],
            "run_dir": project_relative_display_path(run_dir, project_root=root),
        },
    )

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source, filename in (
        (report_path, "strategy_comparison.md"),
        (json_path, "strategy_comparison.json"),
        (manifest_path, "manifest.json"),
    ):
        shutil.copy2(source, latest_dir / filename)

    print("\n" + console_report)
    print_artifact_paths(
        (
            ("策略比較Markdown", report_path),
            ("策略比較JSON", json_path),
            ("執行Manifest", manifest_path),
            ("最新結果", latest_dir),
        ),
        project_root=root,
    )
    return payload


def show_strategy_comparison_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = settings or get_strategy_comparison_settings()
    status = collect_artifact_status(project_root=root, settings=settings)
    print("\n" + render_status(project_root=root, settings=settings, status=status))
    return status


__all__ = [
    "ResolvedComparisonPlan",
    "resolve_comparison_plan",
    "collect_artifact_status",
    "render_execution_plan",
    "render_strategy_execution_plan_surface",
    "render_status",
    "render_strategy_aggregate_report",
    "render_strategy_run_execution_table",
    "run_strategy_comparison",
    "show_strategy_comparison_status",
]
