"""Config-driven Breakout Quality strategy performance comparison orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from config.strategy_compare import (
    get_strategy_comparison_settings,
    get_strategy_runtime_integration_settings,
)
from core.runtime_utils import get_taipei_now
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
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
    StrategyComparisonArm,
    StrategyComparisonSettings,
    StrategyDLSource,
    StrategyPreparationAction,
    StrategyPreparationPlan,
    strategy_comparison_fingerprint,
)
from core.report_metrics import (
    EXECUTION_STRATEGY_RESULT_METRICS,
    CORE_STRATEGY_RESULT_METRICS,
    PORTFOLIO_RESULT_METRICS,
    TRADE_RESULT_METRICS,
)
from core.report_style import best_worst_signals, styled_signal
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
    read_json_object_or_none as _read_json,
    resolve_project_relative_path as _resolve_relative_path,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
)
from filters.breakout_quality.strategy_compare_diagnostics import (
    backfill_pair_r_conversion_diagnostic,
    backfill_pair_selection_diagnostics,
    build_strategy_diagnostics,
    render_strategy_diagnostics_markdown,
    render_strategy_r_analysis_table,
)
from filters.breakout_quality.strategy_compare_reporting import (
    materialize_strategy_pair_readable_report,
    render_strategy_pair_simple_report,
)
from filters.breakout_quality.strategy_compare_engine import run_comparison
from filters.breakout_quality.strategy_compare_replay import run_standalone_baseline
from filters.breakout_quality.strategy_compare_plan import ResolvedComparisonPlan
from filters.breakout_quality.strategy_compare_runtime import (
    _arm_runtime_spec,
    _execution_pairs,
    _standalone_baseline_arms,
)
from filters.breakout_quality.strategy_compare_reuse import (
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
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
)
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from filters.breakout_quality.strategy_rule_policies import (
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)
from filters.breakout_quality.strategy_compare_preparation import (
    collect_artifact_status as collect_preparation_status,
    prepare_strategy_comparison_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 7


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_native(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )






def _resolved_ranking_options(
    settings: StrategyComparisonSettings,
    arm: StrategyComparisonArm,
    *,
    continuous_score_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    options = dict(arm.dl_runtime_options or {})
    if arm.dl_runtime_mode != STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL:
        return options
    safety_dl_id = str(options.get("safety_dl_id") or "").strip()
    if not safety_dl_id or safety_dl_id not in settings.dl_sources:
        raise ValueError(f"dual-model safety arm缺少合法safety_dl_id: {arm.arm_id}")
    source = settings.dl_sources[safety_dl_id]
    safety_override = dict((continuous_score_overrides or {}).get(safety_dl_id) or {})
    options.update({
        "safety_score_source": str(source.score_source),
        "safety_model_architecture": str(source.model_architecture),
        "safety_experiment_profile": str(source.experiment_profile),
        "safety_score_path_override": (
            None
            if not safety_override.get("score_path")
            else str(safety_override["score_path"])
        ),
    })
    return options











































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
            arm.rule_policy,
            "DL-on" if arm.dl_enabled else "DL-off",
            arm.description,
        )
        for arm in current.enabled_arms
    ]
    contrast_rows = [
        (
            "ON" if item.enabled else "OFF",
            item.contrast_id,
            item.left,
            item.right,
            item.description,
        )
        for item in current.enabled_contrasts
    ]
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
    for dl_id, row in current_status["dl_sources"].items():
        for key, file_row in row["files"].items():
            artifact_rows.append(
                (f"dl:{dl_id}:{key}", file_row["status"], file_row["action"], file_row["path"])
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
                    ("Param policy", current.param_policy),
                    ("Max positions", current.max_positions),
                    ("Rotation", current.rotation),
                    ("Config fingerprint", current_status["config_fingerprint"]),
                    ("比較狀態", current_status["overall_status"]),
                    ("策略比較自動前置", "on" if current.preparation.auto_prepare else "off"),
                    ("模型權重前置", "模型訓練 → 準備策略比較所需模型工件"),
                )
            ),
            render_section("1. 比較對象"),
            render_table(
                ("開關", "編號", "名稱", "參數來源", "Rules", "DL", "用途"),
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


def render_execution_plan(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
) -> str:
    plan: StrategyPreparationPlan = status["preparation_plan"]
    ordered_actions = sorted(
        plan.actions,
        key=lambda item: (
            0 if item.artifact_key.startswith("param:") else 1,
            item.artifact_key,
        ),
    )
    blocked = plan.overall_status == "BLOCKED"
    rows = []
    for item in ordered_actions:
        action = item.action
        description = item.description
        if blocked and action in {"BUILD", "REBUILD"}:
            action = "NOT_RUN"
            description = f"{description}｜整體計畫已BLOCKED，本次不執行"
        rows.append((action, item.artifact_key, description))

    replay_cache = dict(status.get("replay_cache") or {})
    cached_pairs = dict(replay_cache.get("pairs") or {})
    cached_baselines = dict(replay_cache.get("baseline_groups") or {})
    for arm in settings.enabled_arms:
        if blocked:
            rows.append(("NOT_RUN", arm.arm_id, f"{arm.name}｜上游前置工件BLOCKED，本次不執行"))
            continue
        if arm.dl_enabled:
            action = "REUSE" if cached_pairs.get(arm.arm_id) is not None else "RUN"
            description = (
                f"{arm.name}｜重用已完成且identity一致的正式pair結果"
                if action == "REUSE"
                else arm.name
            )
        else:
            group_key = f"{arm.param_source}::{arm.rule_policy}"
            action = "REUSE" if cached_baselines.get(group_key) is not None else "RUN"
            description = (
                f"{arm.name}｜重用既有正式shared baseline"
                if action == "REUSE"
                else arm.name
            )
        rows.append((action, arm.arm_id, description))
    rows.extend(
        (
            "NOT_RUN" if blocked else "REPORT",
            item.contrast_id,
            f"{item.description}｜上游前置工件BLOCKED，本次不產生報表" if blocked else item.description,
        )
        for item in settings.enabled_contrasts
    )
    return "\n\n".join(
        (
            render_title("本次執行計畫"),
            render_key_values(
                (
                    ("整體狀態", plan.overall_status),
                    ("設定檔", "config/strategy_compare.py"),
                    ("比較階段", f"{settings.profile_label} ({settings.profile_id})"),
                    ("Config fingerprint", status["config_fingerprint"]),
                )
            ),
            render_table(("動作", "項目", "說明"), rows),
        )
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






def _standalone_baseline_group_id(arm: StrategyComparisonArm) -> str:
    return f"{arm.param_source}__{arm.rule_policy}__dl_off_baseline"




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
    signals_by_metric = {
        metric.key: best_worst_signals(
            {arm.arm_id: _metric(scenarios[arm.arm_id], metric.key) for arm in settings.enabled_arms},
            preference=metric.preference,
        )
        for metric in metrics
    }
    rows = []
    for arm in settings.enabled_arms:
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


def _core_result_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    return render_strategy_core_result_table(
        scenarios, settings=settings, target=target
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


def _execution_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    return render_strategy_execution_table(
        scenarios, settings=settings, target=target
    )


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = _metric(left, key)
    b = _metric(right, key)
    return None if a is None or b is None else a - b


def _same_param_direct_selection_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float | None:
    """Return DL selection attribution only within one parameter/runtime universe.

    ``direct_selection_delta_r`` is defined against the DL-off baseline that
    shares the same ``param_source`` and ``rule_policy``.  Subtracting values
    across different parameter sources or rule policies mixes two different
    attribution universes and has no controlled physical interpretation.
    """
    if (
        left.get("param_source") != right.get("param_source")
        or left.get("rule_policy") != right.get("rule_policy")
    ):
        return None
    return _delta(left, right, "direct_selection_delta_r")


def _contrast_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for contrast in settings.enabled_contrasts:
        left = scenarios[contrast.left]
        right = scenarios[contrast.right]
        rows.append(
            (
                contrast.contrast_id,
                contrast.description,
                _fmt(_delta(left, right, "total_return_pct"), unit="pp"),
                _fmt(_delta(left, right, "max_drawdown_pct"), unit="pp"),
                _fmt(_delta(left, right, "return_over_max_drawdown")),
                _fmt(_delta(left, right, "annual_return_pct"), unit="pp"),
                _fmt(_delta(left, right, "expected_value_r"), unit=" R"),
                _fmt(_delta(left, right, "avg_exposure_pct"), unit="pp"),
                _fmt(_delta(left, right, "trade_count"), digits=0),
                _fmt(
                    _same_param_direct_selection_delta(left, right),
                    unit=" R",
                ),
            )
        )
    return render_table(
        (
            "比較",
            "用途",
            "Δ報酬",
            "ΔMDD",
            "ΔRoMD",
            "Δ年化",
            "ΔEV",
            "Δ曝險",
            "Δ交易",
            "Δ同參數DL選擇R",
        ),
        rows,
    )


def _fmt_money_milli(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    number = float(value) / 1000.0
    if not math.isfinite(number):
        return "-"
    return f"{number:,.0f}"


def _resource_aware_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode not in {
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
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
        }:
            continue
        payload = scenarios[arm.arm_id]
        rows.append((
            arm.arm_id,
            arm.name,
            _fmt(payload.get("resource_aware_dl_selection_days"), digits=0),
            _fmt(payload.get("resource_aware_capital_utilization_days"), digits=0),
            _fmt(payload.get("resource_aware_changed_days"), digits=0),
            _fmt(payload.get("resource_aware_promoted_pass_orders"), digits=0),
            _fmt_money_milli(payload.get("resource_aware_pass_reserved_gain_milli")),
            _fmt(payload.get("resource_aware_promoted_score_orders"), digits=0),
            _fmt(payload.get("resource_aware_selected_score_sum_gain"), digits=3),
            _fmt(payload.get("resource_aware_direct_score_order_days"), digits=0),
            _fmt(payload.get("resource_aware_selected_count_delta"), digits=0),
            _fmt_money_milli(payload.get("resource_aware_reserved_delta_milli")),
            _fmt(payload.get("resource_aware_preservation_violation_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_eligible_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_repair_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_fallback_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_seed_fallback_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_feasible_ascent_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_feasible_ascent_local_optimum_days"), digits=0),
            _fmt(payload.get("resource_aware_stale_score_guard_max_age_days"), unit="日", digits=0),
            _fmt(payload.get("resource_aware_stale_score_guard_triggered_days"), digits=0),
            _fmt(payload.get("resource_aware_stale_score_candidate_count"), digits=0),
            _fmt(payload.get("resource_aware_stale_score_guard_blocked_swaps"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_order_count_violation_days"), digits=0),
        ))
    if not rows:
        return "本次沒有啟用Resource-aware arm。"
    return render_table(
        (
            "編號",
            "比較對象",
            "DL選股日",
            "資金利用優先日",
            "實際改單日",
            "新增PASS單",
            "PASS預留資金增量",
            "Continuous新選入單",
            "Selected Score總和增量",
            "直接Score排序可行日",
            "預計選入差",
            "總預留資金增量",
            "資源保護違規日",
            "Max-DL可介入日",
            "Max-DL修復日",
            "Max-DL回退日",
            "Seed原為回退日",
            "Feasible-ascent改善日",
            "1-swap local optimum日",
            "Stale門檻",
            "Stale guard日",
            "Stale候選數",
            "Blocked swaps",
            "Max-DL K違規日",
        ),
        rows,
    )


def _selector_timing_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode not in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
        }:
            continue
        payload = scenarios[arm.arm_id]
        rows.append((
            arm.arm_id,
            arm.name,
            _fmt(payload.get("resource_aware_selector_timing_total_ms"), unit=" ms", digits=2),
            _fmt(payload.get("resource_aware_selector_timing_median_ms"), unit=" ms", digits=3),
            _fmt(payload.get("resource_aware_selector_timing_p95_ms"), unit=" ms", digits=3),
            _fmt(payload.get("resource_aware_selector_timing_max_ms"), unit=" ms", digits=3),
            _fmt(payload.get("resource_aware_max_dl_repair_evaluations"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_feasible_ascent_evaluations"), digits=0),
            _fmt(payload.get("resource_aware_constrained_search_states"), digits=0),
            _fmt(payload.get("resource_aware_constrained_pruned_states"), digits=0),
            _fmt(payload.get("resource_aware_constrained_optimality_certified_days"), digits=0),
        ))
    if not rows:
        return "本次沒有啟用Max-DL selector arm。"
    return render_table(
        (
            "編號",
            "比較對象",
            "Selector總時間",
            "Median/日",
            "P95/日",
            "Max/日",
            "C17 repair eval",
            "Feasible-ascent eval",
            "Exact states",
            "Exact pruned",
            "Exact certified日",
        ),
        rows,
    )


def render_strategy_yearly_values_table(
    by_id: dict[str, dict[int, float | None]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    enabled_ids = tuple(arm.arm_id for arm in settings.enabled_arms)
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


def _yearly_table(
    pair_payloads: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
    target: str = "plain",
) -> str:
    return render_strategy_yearly_values_table(
        _yearly_values_by_id(pair_payloads, settings=settings),
        settings=settings,
        target=target,
    )


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


def render_strategy_aggregate_report(
    *,
    settings: StrategyComparisonSettings,
    comparison_period: Any,
    fingerprint: str,
    scenarios: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    yearly_by_id: dict[str, dict[int, Any]],
    target: str = "plain",
    title: str = "策略績效比較",
    fingerprint_label: str = "Config fingerprint",
    extra_metadata: tuple[tuple[object, object], ...] = (),
) -> str:
    """Render the canonical cross-arm Strategy Compare human report.

    Selection PIT, Forward-OOS and Multi-seed robustness all consume this same
    top-level renderer.  Callers may append method-specific sections after the
    four canonical Strategy Compare sections, but must not reimplement them.
    """

    metadata_rows = (
        ("期間", comparison_period),
        ("Dataset", settings.dataset),
        ("Param policy", settings.param_policy),
        ("Max positions", settings.max_positions),
        ("Rotation", settings.rotation),
        (fingerprint_label, fingerprint),
        ("比較設定", "config/strategy_compare.py"),
        *tuple(extra_metadata),
    )
    return "\n\n".join(
        (
            render_title(title),
            render_key_values(metadata_rows),
            render_section("1. 核心策略結果"),
            render_strategy_core_result_table(
                scenarios, settings=settings, target=target
            ),
            render_section("2. R 預測／轉化"),
            render_strategy_r_analysis_table(diagnostics, target=target),
            render_section("3. 資金／執行"),
            render_strategy_execution_table(
                scenarios, settings=settings, target=target
            ),
            render_section("4. 年度結果"),
            render_strategy_yearly_values_table(
                yearly_by_id, settings=settings, target=target
            ),
        )
    ).rstrip() + "\n"


def _render_report(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    pair_payloads: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    target: str = "plain",
) -> str:
    return render_strategy_aggregate_report(
        settings=settings,
        comparison_period=_comparison_period(pair_payloads),
        fingerprint=str(status["config_fingerprint"]),
        scenarios=scenarios,
        diagnostics=diagnostics,
        yearly_by_id=_yearly_values_by_id(pair_payloads, settings=settings),
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
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = settings or get_strategy_comparison_settings()
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
            "目前啟用比較缺少不可自動產生的上游工件；請依狀態頁使用正式模型入口處理。"
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
    pair_payloads: dict[str, dict[str, Any]] = {}
    direct_r: dict[str, float] = {}
    pair_execution: dict[str, dict[str, Any]] = {}
    cached_pairs = dict(replay_cache.get("pairs") or {})
    cached_baseline_groups = dict(replay_cache.get("baseline_groups") or {})
    baseline_sources: dict[str, Path] = {
        str(group_key): Path(str(row["source_pair_dir"])).resolve()
        for group_key, row in cached_baseline_groups.items()
        if isinstance(row, dict) and row.get("source_pair_dir")
    }

    for off_arm in _standalone_baseline_arms(settings):
        group_key = f"{off_arm.param_source}::{off_arm.rule_policy}"
        group_id = _standalone_baseline_group_id(off_arm)
        pair_dir = run_dir / "pairs" / group_id
        baseline_reuse_source = (
            baseline_sources.get(group_key)
            if settings.preparation.reuse_shared_baseline
            else None
        )
        all_off = off_arm.rule_policy == "all_off"
        baseline_payload = run_standalone_baseline(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_parameter_paths"][off_arm.param_source]),
            param_policy=settings.param_policy,
            max_positions=settings.max_positions,
            enable_rotation=settings.rotation == "on",
            optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off
                else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            output_dir_override=pair_dir,
            comparison_start_date=comparison_start,
            comparison_end_date=comparison_end,
            quiet=quiet,
            shared_param_overrides=(
                ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None
            ),
            baseline_reuse_dir=baseline_reuse_source,
        )
        pair_payloads[group_id] = {
            "arm_contract": (
                off_arm.param_source, off_arm.rule_policy, off_arm, None
            ),
            "payload": baseline_payload,
        }
        pair_execution[off_arm.arm_id] = {
            "action": "REUSE" if baseline_reuse_source is not None else "RUN",
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
            print(f"[{action}] {off_arm.arm_id} {off_arm.name}")
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
        )
        pair_dir = run_dir / "pairs" / group_id
        baseline_group_key = f"{param_source}::{rule_policy}"
        cache_entry = cached_pairs.get(on_arm.arm_id)

        if isinstance(cache_entry, dict) and cache_entry.get("source_pair_dir"):
            source_pair_dir = Path(str(cache_entry["source_pair_dir"])).resolve()
            if pair_dir.exists():
                shutil.rmtree(pair_dir)
            shutil.copytree(source_pair_dir, pair_dir)
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
                print(f"[REUSE] {on_arm.arm_id} {on_arm.name}")
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
        if quiet:
            print(f"[RUN] {on_arm.arm_id} {on_arm.name}")
        pair_payload = run_comparison(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_parameter_paths"][param_source]),
            param_policy=settings.param_policy,
            max_positions=settings.max_positions,
            enable_rotation=settings.rotation == "on",
            fixed_risk=None,
            max_position_cap_pct=None,
            comparison_mode=runtime_spec["comparison_mode"],
            ranking_policy=runtime_spec["ranking_policy"],
            ranking_options=(
                {
                    **_resolved_ranking_options(
                        settings,
                        on_arm,
                        continuous_score_overrides=(status.get("continuous_score_overrides") or {}),
                    ),
                    **(
                        {
                            "expected_r_calibration_path": str(
                                (status.get("expected_r_calibrations") or {})[on_arm.arm_id]["lookup_path"]
                            )
                        }
                        if on_arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                        else {
                            "expected_excess_r_calibration_path": str(
                                (status.get("expected_excess_r_calibrations") or {})[on_arm.arm_id]["lookup_path"]
                            )
                        }
                        if on_arm.dl_runtime_mode in {
                            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
                            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
                            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
                        }
                        else {}
                    ),
                }
            ),
            optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off
                else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            filter_id=dl.filter_id,
            score_source=dl.score_source,
            model_architecture=dl.model_architecture,
            experiment_profile=dl.experiment_profile,
            threshold=dl.threshold,
            output_dir_override=pair_dir,
            comparison_start_date=comparison_start,
            comparison_end_date=comparison_end,
            quiet=quiet,
            shared_param_overrides=(
                ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None
            ),
            baseline_reuse_dir=baseline_reuse_source,
            continuous_score_path_override=(
                str(
                    (status.get("continuous_score_overrides") or {})[on_arm.dl_id][
                        "score_path"
                    ]
                )
                if (
                    dl.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
                    and on_arm.dl_id in (status.get("continuous_score_overrides") or {})
                )
                else None
            ),
            continuous_score_execution_start_override=(
                str(
                    (status.get("continuous_score_overrides") or {})[on_arm.dl_id][
                        "execution_start"
                    ]
                )
                if (
                    dl.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
                    and on_arm.dl_id in (status.get("continuous_score_overrides") or {})
                )
                else None
            ),
            capture_execution_diagnostics=(
                runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            ),
        )
        backfill_pair_r_conversion_diagnostic(
            pair_payload,
            pair_dir=pair_dir,
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
            print(f"[DONE] {on_arm.arm_id} {on_arm.name}")

    scenarios = _scenario_payloads(pair_payloads, direct_r, settings=settings)
    reference_arm_id = _report_reference_arm_id(settings)
    diagnostics = build_strategy_diagnostics(
        project_root=root,
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        reference_arm_id=reference_arm_id,
    )
    report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        diagnostics=diagnostics,
        target="markdown",
    )
    console_report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
        diagnostics=diagnostics,
        target="console",
    )
    report_path = run_dir / "strategy_comparison.md"
    diagnostics_path = run_dir / "strategy_diagnostics.md"
    json_path = run_dir / "strategy_comparison.json"
    manifest_path = run_dir / "manifest.json"
    report_path.write_text(report, encoding="utf-8")
    diagnostics_path.write_text(
        render_strategy_diagnostics_markdown(diagnostics), encoding="utf-8"
    )
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "COMPLETED",
        "created_at": get_taipei_now().isoformat(),
        "config_fingerprint": status["config_fingerprint"],
        "requested_config_fingerprint": requested_fingerprint,
        "settings": settings.as_dict(),
        "artifact_identities": status["artifact_identities"],
        "comparison_period": comparison_period,
        "comparison_period_source": status.get("comparison_period_source"),
        "requested_preparation_plan": requested_plan.as_dict(),
        "final_preparation_plan": status["preparation_plan"].as_dict(),
        "scenarios": scenarios,
        "diagnostics": diagnostics,
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
            "replay_reuse_policy": payload["replay_reuse_policy"],
            "run_dir": project_relative_display_path(run_dir, project_root=root),
        },
    )

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source, filename in (
        (report_path, "strategy_comparison.md"),
        (diagnostics_path, "strategy_diagnostics.md"),
        (json_path, "strategy_comparison.json"),
        (manifest_path, "manifest.json"),
    ):
        shutil.copy2(source, latest_dir / filename)

    print("\n" + console_report)
    print_artifact_paths(
        (
            ("策略比較Markdown", report_path),
            ("間接指標Markdown", diagnostics_path),
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
    "render_status",
    "render_strategy_aggregate_report",
    "run_strategy_comparison",
    "show_strategy_comparison_status",
]
