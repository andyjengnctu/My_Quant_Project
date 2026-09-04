"""Canonical runtime/source contract resolution for Strategy Compare pairs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
)
from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
)
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from filters.breakout_quality.artifacts import load_runtime_artifact_contract
from filters.breakout_quality.binary_pit_score_store import (
    BINARY_PIT_SCORE_SOURCE,
    load_binary_point_in_time_score_table,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    SUPPORTED_RANKING_SCORE_SOURCES,
    load_continuous_ranker_oos_contract,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_ranking_contract,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    comparison_labels,
)
from filters.breakout_quality.strategy_compare_diagnostics import (
    _load_isolated_selection_pit_contract,
)
from filters.breakout_quality.strategy_compare_sources import (
    SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES,
)
from services.research.strategy_compare_replay import _resolve_comparison_period


_CONTINUOUS_RANKING_POLICIES = frozenset({
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
})


@dataclass(frozen=True)
class StrategyCompareRuntimeContract:
    comparison_mode: str
    ranking_policy: str
    ranking_options: dict[str, Any]
    optional_entry_filter_policy: str
    score_source: str
    filter_id: str
    model_architecture: str
    experiment_profile: str
    configured_threshold: float
    labels: dict[str, Any]
    runtime_contract: Any
    pit_contract: Any
    continuous_contract: Any
    continuous_score_override: dict[str, Any] | None
    ranking_source: dict[str, Any] | None
    filter_source: dict[str, Any] | None
    binary_filter_manifest: dict[str, Any] | None
    binary_filter_manifest_path: Path | None
    binary_filter_scores_path: Path | None
    effective_score_source: str
    manifest_architecture: str
    manifest_profile: str
    start_date: str
    end_date: str


def _resolve_continuous_score_override_period(
    table: pd.DataFrame, *, execution_start_override=None
) -> tuple[str, str, str]:
    available_from = str(table.attrs.get("available_from") or "")
    available_through = str(table.attrs.get("available_through") or "")
    if not available_from or not available_through:
        raise ValueError("Continuous ranker isolated score table缺少日期範圍metadata")
    if execution_start_override in (None, ""):
        execution_start = available_from
    else:
        execution_start = pd.Timestamp(str(execution_start_override)).strftime("%Y-%m-%d")
        if execution_start > available_from:
            raise ValueError(
                "Continuous ranker isolated execution_start不可晚於第一個Score日："
                f"execution_start={execution_start}, available_from={available_from}"
            )
    return execution_start, available_from, available_through


def resolve_strategy_compare_runtime_contract(
    *,
    root: Path,
    comparison_mode: str,
    ranking_policy: str,
    ranking_options: dict[str, Any] | None,
    optional_entry_filter_policy: str,
    score_source: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    threshold: float | None,
    hard_filter_source: dict[str, Any] | None,
    continuous_score_path_override=None,
    continuous_score_execution_start_override=None,
    selection_pit_score_path_override=None,
    selection_pit_manifest_path_override=None,
    selection_pit_expected_seed_override=None,
    comparison_start_date=None,
    comparison_end_date=None,
) -> StrategyCompareRuntimeContract:
    comparison_mode = str(comparison_mode)
    ranking_policy = str(ranking_policy).strip()
    ranking_options = dict(ranking_options or {})
    optional_entry_filter_policy = str(optional_entry_filter_policy).strip()
    score_source = str(score_source)
    filter_id = str(filter_id)
    model_architecture = str(model_architecture)
    experiment_profile = str(experiment_profile)
    configured_threshold = float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD if threshold is None else threshold)
    if not 0.0 <= configured_threshold <= 1.0:
        raise ValueError("threshold必須介於0與1")
    labels = comparison_labels(comparison_mode)
    if score_source not in SUPPORTED_RANKING_SCORE_SOURCES:
        raise ValueError(f"不支援的 score source: {score_source!r}")
    if ranking_policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的 ranking policy: {ranking_policy!r}")
    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD:
        raw_max_age = ranking_options.get("stale_score_membership_guard_max_age_days")
        if isinstance(raw_max_age, bool) or not isinstance(raw_max_age, int) or raw_max_age < 0:
            raise ValueError("stale-score membership guard必須由config提供非負整數 stale_score_membership_guard_max_age_days")
    if optional_entry_filter_policy not in SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES:
        raise ValueError(f"不支援的 optional entry filter policy: {optional_entry_filter_policy!r}")
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        if score_source != SCORE_SOURCE_CANONICAL_RUNTIME:
            raise ValueError("hard-filter策略比較的score_source欄位必須維持canonical_runtime；研究用Binary PIT須透過hard_filter_source傳入")
        if ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE:
            raise ValueError("hard-filter策略比較不可指定capital-aware ranking policy")
    elif hard_filter_source is not None:
        raise ValueError("hard_filter_source只支援hard-filter策略比較")

    has_continuous_override = continuous_score_path_override not in (None, "")
    has_execution_start_override = continuous_score_execution_start_override not in (None, "")
    if has_continuous_override != has_execution_start_override:
        raise ValueError("continuous_score_path_override與continuous_score_execution_start_override必須成對提供")
    has_selection_pit_score_override = selection_pit_score_path_override not in (None, "")
    has_selection_pit_manifest_override = selection_pit_manifest_path_override not in (None, "")
    if has_selection_pit_score_override != has_selection_pit_manifest_override:
        raise ValueError("selection_pit_score_path_override與selection_pit_manifest_path_override必須成對提供")
    if has_selection_pit_score_override and score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        raise ValueError("Selection PIT isolated override只支援selection_point_in_time score source")

    runtime_contract = pit_contract = continuous_contract = None
    continuous_score_override = ranking_source = filter_source = None
    binary_filter_manifest = None
    binary_filter_manifest_path = binary_filter_scores_path = None
    effective_score_source = score_source

    if hard_filter_source is not None:
        source_payload = dict(hard_filter_source or {})
        source_name = str(source_payload.get("score_source") or "").strip()
        if source_name != BINARY_PIT_SCORE_SOURCE:
            raise ValueError(f"hard_filter_source目前只接受binary_point_in_time: actual={source_name!r}")
        binary_filter_manifest_path = Path(str(source_payload.get("manifest_path") or "")).resolve()
        binary_filter_scores_path = Path(str(source_payload.get("scores_path") or "")).resolve()
        _score_table, binary_filter_manifest = load_binary_point_in_time_score_table(
            str(binary_filter_manifest_path), str(binary_filter_scores_path)
        )
        manifest_architecture = str(binary_filter_manifest.get("model_architecture") or "")
        manifest_profile = str(binary_filter_manifest.get("experiment_profile") or "")
        if manifest_architecture != model_architecture or manifest_profile != experiment_profile:
            raise ValueError(
                "Binary PIT artifact與指定模型identity不一致: "
                f"artifact={manifest_architecture}/{manifest_profile}, requested={model_architecture}/{experiment_profile}"
            )
        artifact_threshold = float(binary_filter_manifest.get("threshold", float("nan")))
        if not math.isclose(artifact_threshold, configured_threshold, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"Binary PIT threshold與固定threshold不一致: artifact={artifact_threshold}, policy={configured_threshold}")
        period = dict(binary_filter_manifest.get("score_period") or {})
        start_date, end_date = str(period.get("start") or ""), str(period.get("end") or "")
        if not start_date or not end_date:
            raise ValueError("Binary PIT manifest缺少score_period")
        filter_source = {
            "score_source": BINARY_PIT_SCORE_SOURCE,
            "manifest_path": str(binary_filter_manifest_path),
            "scores_path": str(binary_filter_scores_path),
        }
        effective_score_source = BINARY_PIT_SCORE_SOURCE
    elif score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        if comparison_mode != COMPARISON_MODE_SCORE_RANKING:
            raise ValueError("Continuous ranker OOS source只支援score-ranking比較")
        if ranking_policy not in _CONTINUOUS_RANKING_POLICIES:
            raise ValueError("Continuous ranker OOS source只允許resource-aware-continuous系列policy")
        if has_continuous_override:
            override_path = Path(str(continuous_score_path_override)).resolve()
            override_table = load_continuous_ranker_oos_score_table_from_path(str(override_path), experiment_profile)
            manifest_architecture, manifest_profile = model_architecture, experiment_profile
            execution_start, available_from, available_through = _resolve_continuous_score_override_period(
                override_table,
                execution_start_override=continuous_score_execution_start_override,
            )
            start_date, end_date = execution_start, available_through
            continuous_score_override = {
                "score_path": str(override_path),
                "execution_start": execution_start,
                "available_from": available_from,
                "available_through": available_through,
            }
        else:
            continuous_contract = load_continuous_ranker_oos_contract(str(root), filter_id, model_architecture, experiment_profile)
            manifest_architecture = continuous_contract.model_architecture
            manifest_profile = continuous_contract.experiment_profile
            start_date, end_date = continuous_contract.execution_start, continuous_contract.available_through
        ranking_source = {
            "score_source": SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
            "model_architecture": manifest_architecture,
            "experiment_profile": manifest_profile,
            "ranking_policy": ranking_policy,
            "ranking_options": ranking_options,
            "score_path_override": None if continuous_score_override is None else continuous_score_override["score_path"],
        }
    elif score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        if comparison_mode != COMPARISON_MODE_SCORE_RANKING:
            raise ValueError("Selection PIT score source只支援score-ranking比較")
        if has_selection_pit_score_override:
            pit_contract = _load_isolated_selection_pit_contract(
                score_path=str(selection_pit_score_path_override),
                manifest_path=str(selection_pit_manifest_path_override),
                filter_id=filter_id,
                model_architecture=model_architecture,
                experiment_profile=experiment_profile,
                expected_seed=None if selection_pit_expected_seed_override is None else int(selection_pit_expected_seed_override),
            )
        else:
            pit_contract = load_selection_point_in_time_ranking_contract(
                str(root), filter_id, model_architecture, experiment_profile,
                require_model_validation_pass=False,
            )
            workflow_settings = get_breakout_quality_workflow_settings()
            if (
                workflow_settings.filter_id == filter_id
                and workflow_settings.model_architecture == model_architecture
                and workflow_settings.experiment_profile == experiment_profile
                and int(pit_contract.seed) != int(workflow_settings.seed)
            ):
                raise ValueError(
                    "Selection PIT工件seed與目前workflow不一致，必須重建："
                    f"artifact={pit_contract.seed}, workflow={workflow_settings.seed}"
                )
        manifest_architecture, manifest_profile = pit_contract.model_architecture, pit_contract.experiment_profile
        start_date, end_date = pit_contract.available_from, pit_contract.available_through
        ranking_source = {
            "score_source": score_source,
            "model_architecture": manifest_architecture,
            "experiment_profile": manifest_profile,
            "ranking_policy": ranking_policy,
            "ranking_options": ranking_options,
            "score_path_override": str(pit_contract.score_path),
            "score_manifest_path_override": str(pit_contract.manifest_path),
        }
    else:
        try:
            runtime_contract = load_runtime_artifact_contract(str(root), filter_id)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                "策略對照需要 active breakout-quality 的正式 forward-OOS scores.csv；"
                "請先執行 export-scores --scope forward_oos。"
            ) from exc
        manifest_architecture = str(runtime_contract.manifest.get("model_architecture") or "")
        manifest_profile = str(runtime_contract.manifest.get("experiment_profile") or "")
        if manifest_architecture != model_architecture or manifest_profile != experiment_profile:
            raise ValueError(
                "runtime artifact與指定模型identity不一致: "
                f"artifact={manifest_architecture}/{manifest_profile}, requested={model_architecture}/{experiment_profile}"
            )
        artifact_threshold = float(runtime_contract.manifest.get("fixed_evaluation_threshold", float("nan")))
        if not math.isclose(artifact_threshold, configured_threshold, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "active threshold與模型OOS前固定threshold不一致: "
                f"artifact={artifact_threshold}, policy={configured_threshold}"
            )
        start_date, end_date = _resolve_comparison_period(runtime_contract)
        if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
            ranking_source = {
                "score_source": SCORE_SOURCE_CANONICAL_RUNTIME,
                "model_architecture": None,
                "experiment_profile": None,
                "ranking_policy": ranking_policy,
                "ranking_options": ranking_options,
            }

    if (comparison_start_date is None) != (comparison_end_date is None):
        raise ValueError("comparison_start_date與comparison_end_date必須同時提供")
    if comparison_start_date is not None:
        default_start, default_end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
        requested_start = pd.Timestamp(str(comparison_start_date)).normalize()
        requested_end = pd.Timestamp(str(comparison_end_date)).normalize()
        if pd.isna(requested_start) or pd.isna(requested_end) or requested_end < requested_start:
            raise ValueError("指定策略比較期間不合法")
        if requested_start < default_start or requested_end > default_end:
            raise ValueError(
                "指定策略比較期間超出Score可用範圍："
                f"requested={requested_start.date()}~{requested_end.date()}, available={default_start.date()}~{default_end.date()}"
            )
        start_date, end_date = requested_start.strftime("%Y-%m-%d"), requested_end.strftime("%Y-%m-%d")

    return StrategyCompareRuntimeContract(
        comparison_mode=comparison_mode,
        ranking_policy=ranking_policy,
        ranking_options=ranking_options,
        optional_entry_filter_policy=optional_entry_filter_policy,
        score_source=score_source,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        configured_threshold=configured_threshold,
        labels=labels,
        runtime_contract=runtime_contract,
        pit_contract=pit_contract,
        continuous_contract=continuous_contract,
        continuous_score_override=continuous_score_override,
        ranking_source=ranking_source,
        filter_source=filter_source,
        binary_filter_manifest=binary_filter_manifest,
        binary_filter_manifest_path=binary_filter_manifest_path,
        binary_filter_scores_path=binary_filter_scores_path,
        effective_score_source=effective_score_source,
        manifest_architecture=manifest_architecture,
        manifest_profile=manifest_profile,
        start_date=str(start_date),
        end_date=str(end_date),
    )
