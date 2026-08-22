"""Config-driven Breakout Quality strategy performance comparison orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import pandas as pd

from config.breakout_quality import get_breakout_quality_workflow_settings
from config.strategy_compare import (
    get_strategy_comparison_settings,
    get_strategy_runtime_integration_settings,
)
from core.runtime_utils import get_taipei_now
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
    strategy_comparison_fingerprint,
)
from core.report_metrics import (
    EXECUTION_STRATEGY_RESULT_METRICS,
    CORE_STRATEGY_RESULT_METRICS,
    PORTFOLIO_RESULT_METRICS,
    TRADE_RESULT_METRICS,
)
from core.report_style import best_worst_signals, signal_for_workflow_status, styled_signal
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
from filters.breakout_quality.strategy_compare_preparation import (
    collect_artifact_status as collect_preparation_status,
    prepare_strategy_comparison_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 8


def _selection_pit_mode_paths(
    dl: StrategyDLSource,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path] | None:
    dirname = None if dl.point_in_time_dirname in (None, "") else str(dl.point_in_time_dirname).strip()
    if dirname is None:
        return None
    base = (
        resolve_filter_model_output_dir(
            Path(project_root).resolve(), dl.filter_id, dl.model_architecture, dl.experiment_profile
        )
        / dirname
    ).resolve()
    return {
        "score": base / SELECTION_POINT_IN_TIME_SCORE_FILENAME,
        "manifest": base / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    }


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
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    options = dict(arm.dl_runtime_options or {})
    if arm.dl_runtime_mode not in {
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    }:
        return options
    safety_dl_id = str(options.get("safety_dl_id") or "").strip()
    if not safety_dl_id or safety_dl_id not in settings.dl_sources:
        raise ValueError(f"dual-model safety arm缺少合法safety_dl_id: {arm.arm_id}")
    source = settings.dl_sources[safety_dl_id]
    safety_override = dict((continuous_score_overrides or {}).get(safety_dl_id) or {})
    if source.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        mode_paths = _selection_pit_mode_paths(source, project_root=project_root)
        if mode_paths is not None:
            safety_override = {
                "score_path": mode_paths["score"],
                "manifest_path": mode_paths["manifest"],
            }
    options.update({
        "safety_filter_id": str(source.filter_id),
        "safety_score_source": str(source.score_source),
        "safety_model_architecture": str(source.model_architecture),
        "safety_experiment_profile": str(source.experiment_profile),
        "safety_score_path_override": (
            None
            if not safety_override.get("score_path")
            else str(safety_override["score_path"])
        ),
        "safety_score_manifest_path_override": (
            None
            if not safety_override.get("manifest_path")
            else str(safety_override["manifest_path"])
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
            resolve_strategy_comparison_arm_param_policy(current, arm),
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



_EXECUTION_PLAN_ITEM_WIDTH = 36
_EXECUTION_PLAN_DESCRIPTION_WIDTH = 44


def _compact_execution_plan_text(value: object, *, max_width: int) -> str:
    """Normalize one execution-plan cell and cap its visible width."""

    text = " ".join(str(value).split())
    if not text:
        return "-"
    width = 0
    out: list[str] = []
    ellipsis = "…"
    ellipsis_width = 1
    if _display_width(text) <= int(max_width):
        return text
    limit = max(1, int(max_width) - ellipsis_width)
    for char in text:
        char_width = _display_width(char)
        if width + char_width > limit:
            break
        out.append(char)
        width += char_width
    return "".join(out).rstrip() + ellipsis


def _compact_execution_plan_item(value: object) -> str:
    """Convert verbose artifact identities to stable human-readable labels."""

    text = " ".join(str(value).split())
    if text.startswith("param-benchmark:"):
        parts = text.split(":")
        seed = next((part.split("=", 1)[1] for part in parts if part.startswith("seed=")), "?")
        family = next((part for part in parts if part in {"full", "min"}), "params")
        policy = next(
            (part for part in parts if part in {"base-finalist-best", "base-finalists-agree"}),
            "",
        )
        policy_label = {
            "base-finalist-best": "Best",
            "base-finalists-agree": "Agree",
        }.get(policy, policy)
        text = f"Bench {family.title()} {policy_label} | {seed}".replace("  ", " ")
    elif text.startswith("param:"):
        source = text.split(":", 1)[1]
        source = source.replace("_oos", "").replace("_rolling", "")
        source = source.replace("_", " ")
        text = f"Params | {source.title()}"
    elif text.startswith("model-upstream:"):
        parts = text.split(":")
        profile = next((part for part in parts if "daily_universal" in part), parts[-1])
        profile = profile.replace("daily_universal_", "")
        profile = profile.replace("full_list_ndcg_pairwise", "")
        profile = profile.replace("full_horizon_", "")
        profile = profile.strip("_") or "canonical"
        text = f"Dataset | {profile.replace('_', '-')}"
    elif text.startswith("dl:"):
        parts = text.split(":")
        if len(parts) >= 3:
            text = f"{parts[1]} | {parts[-1]}"
    elif text.endswith(" Rolling"):
        text = text[:-8]
    elif text.endswith(" OOS"):
        text = text[:-4]
    return _compact_execution_plan_text(text, max_width=_EXECUTION_PLAN_ITEM_WIDTH)


def _compact_execution_plan_description(value: object) -> str:
    """Keep decision-relevant plan detail while removing repetitive producer prose."""

    text = " ".join(str(value).split())
    if "由canonical Optimizer parameter service解析／建立／接續策略參數工件" in text:
        policies = text.split("policies=", 1)[1] if "policies=" in text else ""
        policy_count = len([item for item in policies.split(",") if item.strip()])
        text = "canonical params" + (f" | {policy_count} policies" if policy_count else "")
    elif "由canonical Optimizer以相同benchmark seed與統一trials/fold自動建立" in text:
        text = "same-seed selector | shared search"
    elif text.startswith("canonical Dataset需更新："):
        missing_count = None
        marker = "dataset 工件缺少:"
        if marker in text:
            missing = text.split(marker, 1)[1].split("；", 1)[0]
            missing_count = len([item for item in missing.split(",") if item.strip()])
        summary_bad = "dataset_summary.json 缺少、損壞或不是 JSON object" in text
        parts = [f"Dataset缺件{missing_count}項" if missing_count is not None else "Dataset需更新"]
        if summary_bad:
            parts.append("summary無效")
        if "補建" in text:
            parts.append("自動補建")
        text = " | ".join(parts)
    elif text.startswith("重用canonical Dataset／source OHLCV truth"):
        text = "REUSE canonical Dataset/OHLCV"
    elif "Production finalists-agree consensus reference" in text:
        text = "Production consensus reference"
    elif "同seed strategy optimizer params + DL-off replay" in text:
        seed_prefix = text.split("個固定benchmark seeds", 1)[0]
        text = f"{seed_prefix} seeds | same-seed params | DL off"
    elif "同seed strategy params + canonical DL trainer + replay" in text:
        seed_prefix = text.split("個固定benchmark seeds", 1)[0]
        text = f"{seed_prefix} seeds | same-seed params + DL + replay"
    elif "重用已完成且identity一致的正式pair結果" in text:
        text = "REUSE completed pair"
    elif "重用既有正式shared baseline" in text:
        text = "REUSE shared baseline"
    return _compact_execution_plan_text(text, max_width=_EXECUTION_PLAN_DESCRIPTION_WIDTH)


def render_strategy_execution_plan_surface(
    *,
    title: str,
    metadata_rows: tuple[tuple[object, object], ...],
    action_rows: list[tuple[object, object, object]],
) -> str:
    """Render the shared compact Strategy Compare execution-plan surface.

    Full preparation reasons remain in the canonical plan/status payload.  The
    interactive surface intentionally keeps only decision-relevant text so a
    normal 100-column terminal does not wrap every action row.
    """

    styled_metadata = []
    for label, value in metadata_rows:
        rendered = value
        if str(label) == "整體狀態":
            rendered = styled_signal(
                value,
                signal_for_workflow_status(value),
                target="console",
                bold=True,
            )
        styled_metadata.append((label, rendered))
    styled_actions = [
        (
            styled_signal(
                action,
                signal_for_workflow_status(action),
                target="console",
                bold=True,
            ),
            _compact_execution_plan_item(item),
            _compact_execution_plan_description(description),
        )
        for action, item, description in action_rows
    ]
    return "\n\n".join(
        (
            render_title(title),
            render_key_values(tuple(styled_metadata)),
            render_table(("動作", "項目", "說明"), styled_actions),
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
            group_key = (
                f"{arm.param_source}::"
                f"{resolve_strategy_comparison_arm_param_policy(settings, arm)}::"
                f"{arm.rule_policy}"
            )
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
    for arm in settings.enabled_arms:
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
    execution_summary: dict[str, Any] | None = None,
) -> str:
    """Render the canonical cross-arm Strategy Compare human report.

    Single-seed OOS/Rolling and Multi-seed robustness consume this same top-level
    renderer.  The four performance sections are canonical.  Single-seed runs
    may add the neutral execution-summary section; robustness keeps its own
    method-specific sections after the shared performance surface.
    """

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
    sections = [
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
    ]
    if execution_summary is not None:
        sections.extend(
            (
                render_section("5. 執行摘要"),
                render_strategy_run_execution_table(
                    execution_summary, settings=settings, target=target
                ),
            )
        )
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
        baseline_reuse_source = (
            baseline_sources.get(group_key)
            if settings.preparation.reuse_shared_baseline
            else None
        )
        all_off = off_arm.rule_policy == "all_off"
        arm_started = time.perf_counter()
        baseline_action = "REUSE" if baseline_reuse_source is not None else "RUN"
        if quiet and baseline_action == "RUN":
            print(
                f"[RUN] {off_arm.arm_id} {off_arm.name} "
                f"| total={format_elapsed(arm_started - replay_started)}"
            )
        baseline_payload = run_standalone_baseline(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_arm_parameter_paths"][off_arm.arm_id]),
            param_policy=off_param_policy,
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
            param_evaluation_mode=(
                settings.parameter_sources[off_arm.param_source].canonical_evaluation_mode or "rolling"
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
        )
        pair_dir = run_dir / "pairs" / group_id
        on_param_policy = resolve_strategy_comparison_arm_param_policy(settings, on_arm)
        baseline_group_key = f"{param_source}::{on_param_policy}::{rule_policy}"
        cache_entry = cached_pairs.get(on_arm.arm_id)

        if isinstance(cache_entry, dict) and cache_entry.get("source_pair_dir"):
            arm_started = time.perf_counter()
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

        selection_pit_mode_paths = (
            _selection_pit_mode_paths(dl, project_root=root)
            if dl.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else None
        )
        selection_pit_expected_seed = (
            get_breakout_quality_workflow_settings(
                experiment_profile=dl.experiment_profile
            ).seed
            if selection_pit_mode_paths is not None
            else None
        )
        pair_payload = run_comparison(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_arm_parameter_paths"][on_arm.arm_id]),
            param_policy=on_param_policy,
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
                        project_root=root,
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
            progress_callback=_pair_progress,
            shared_param_overrides=(
                ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None
            ),
            baseline_reuse_dir=baseline_reuse_source,
            param_evaluation_mode=(
                settings.parameter_sources[on_arm.param_source].canonical_evaluation_mode or "rolling"
            ),
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
            selection_pit_score_path_override=(
                None
                if selection_pit_mode_paths is None
                else str(selection_pit_mode_paths["score"])
            ),
            selection_pit_manifest_path_override=(
                None
                if selection_pit_mode_paths is None
                else str(selection_pit_mode_paths["manifest"])
            ),
            selection_pit_expected_seed_override=selection_pit_expected_seed,
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

    post_processing_started = time.perf_counter()
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
    "render_strategy_execution_plan_surface",
    "render_status",
    "render_strategy_aggregate_report",
    "render_strategy_run_execution_table",
    "run_strategy_comparison",
    "show_strategy_comparison_status",
]
