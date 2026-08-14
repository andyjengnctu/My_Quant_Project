"""Controlled no-filter vs active breakout-quality portfolio comparison."""

from __future__ import annotations

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    COMPARISON_MODES,
    STRATEGY_COMPARE_SCHEMA_VERSION,
    comparison_labels as _comparison_labels,
    comparison_switch_spec as _comparison_switch_spec,
)
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_FIELDS,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    PARAM_POLICIES,
    PARAM_POLICY_AUTO,
    PARAM_POLICY_BASE_FINALIST_BEST,
    PARAM_POLICY_BASE_FINALISTS_AGREE,
    PARAM_POLICY_SPECS,
    SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES,
    _apply_scenario_overrides,
    _assert_controlled_ensemble_pair,
    _assert_controlled_param_pair,
    _assert_controlled_payload_pair,
    _build_controlled_param_source_pair,
    _collect_payload_differences,
    _comparison_output_dir_name,
    _first_existing_comparison_dir,
    _load_param_source,
    _resolve_param_selector,
    _resolve_params_path,
    _rewrite_ensemble_mapping,
    _rewrite_param_mapping,
    _rolling_member_counts,
    _sha256_file,
    _validate_requested_param_policy,
    canonical_strategy_compare_output_dir_names,
)
from filters.breakout_quality.strategy_compare_reporting import (
    _assert_shared_benchmark,
    _build_yearly_comparison,
    _capacity_summary,
    _delta,
    _format_metric,
    _load_existing_comparison_payload,
    _markdown_report,
    _normalize_yearly_completeness,
    _refresh_yearly_summary,
    _remove_legacy_html_outputs,
    _render_strategy_console_report,
    _scenario_summary,
    _strategy_pair_report_components,
    _to_json_native,
    _yearly_frame,
)
from filters.breakout_quality.strategy_compare_replay import (
    _load_reusable_no_filter_baseline,
    _resolve_comparison_period,
    _run_scenario,
    _run_scenario_inside_source_context,
    _standalone_baseline_report,
    _unpack_result,
    run_standalone_baseline,
)
from filters.breakout_quality.strategy_compare_diagnostics import (
    _flatten_candidate_replay_rows,
    _flatten_entry_execution_rows,
    _flatten_repair_mechanism_rows,
    _flatten_selector_trace_rows,
    _flatten_selected_buy_rows,
    _load_isolated_selection_pit_contract,
    _selection_target_lookup,
    _strategy_selection_diagnostics,
)

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    get_breakout_quality_workflow_settings,
)
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir
from core.rolling_oos_params import get_active_param_date_range
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
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from core.console_report import print_artifact_paths, project_relative_display_path
from filters.breakout_quality.trade_attribution import (
    ATTRIBUTION_SCHEMA_VERSION,
    write_trade_attribution_outputs,
)
from services.portfolio_replay import PORTFOLIO_DEFAULT_BENCHMARK_TICKER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = STRATEGY_COMPARE_SCHEMA_VERSION





def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="固定參數比較 baseline 與 breakout-quality hard filter／score ranking 的策略經濟效果。"
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default=DEFAULT_DATASET_PROFILE)
    parser.add_argument("--comparison-mode", choices=COMPARISON_MODES, default=COMPARISON_MODE_HARD_FILTER)
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--score-source", choices=SUPPORTED_RANKING_SCORE_SOURCES,
        default=SCORE_SOURCE_CANONICAL_RUNTIME,
        help="score-ranking使用的分數來源；hard-filter只接受canonical_runtime。",
    )
    parser.add_argument(
        "--ranking-policy",
        choices=SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
        default=BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
        help=(
            "score-ranking排序契約：score=原始Score；capital-adjusted-score="
            "Score×正式預估部署率；capital-bucket-then-score="
            "每日部署率三分桶後桶內按Score；resource-aware-binary="
            "只在盤前cash先成瓶頸時，以Binary PASS做first-improvement；"
            "resource-aware-binary-basket=相同資源Gate下每輪評估全部可行PASS promotion並採用最佳改善；"
            "resource-aware-continuous-capital-preserving=continuous score可跨cash/slot瓶頸介入，"
            "但不得降低同參數DL-off baseline盤前選入數或exact reserved capital；"
            "resource-aware-continuous-max-dl=固定同參數DL-off baseline預留單數與reserved-capital floor，"
            "由continuous score主導Top-K並只在不合法時作deterministic minimum-repair；"
            "resource-aware-continuous-max-dl-feasible-ascent=相同K/R0與DL objective，"
            "在C17合法seed上持續做best-feasible single-swap DL改善直到1-swap local optimum；"
            "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard=同一selector再禁止過舊score驅動membership change；"
            "resource-aware-continuous-expected-pnl-feasible-ascent=frozen rank percentile校準absolute Expected-R後乘canonical planned risk；"
            "resource-aware-continuous-excess-alpha-feasible-ascent=Selection-only frozen rank percentile校準relative Expected Excess-R後乘canonical planned risk；"
            "resource-aware-continuous-excess-alpha-no-r0-feasible-ascent=同一Excess-Alpha objective與固定K，但移除baseline R0 floor及其minimum repair，只保留canonical cash feasibility。"
        ),
    )
    parser.add_argument(
        "--stale-score-membership-guard-max-age-days",
        type=int,
        default=None,
        help="僅stale-score-guard policy使用；正式Strategy Compare由config arm runtime options提供。",
    )
    parser.add_argument(
        "--optional-entry-filters",
        choices=SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES,
        default=OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        help=(
            "current=沿用active params；all-off=兩組都關閉EMA、BB、Volume、"
            "breakout return與false-breakout五個optional entry filters。"
        ),
    )
    parser.add_argument("--model-architecture", default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    parser.add_argument("--experiment-profile", default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE)
    parser.add_argument("--params", default=None, help="可明確指定active-param JSON；亦可由score source與--param-policy自動解析。")
    parser.add_argument(
        "--param-policy", choices=PARAM_POLICIES, default=PARAM_POLICY_AUTO,
        help="可指定 base-finalist-best 或 base-finalists-agree；auto 沿用 --params。",
    )
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--rotation", choices=("off", "on"), default="off")
    parser.add_argument("--fixed-risk", type=float, default=None, help="選填；兩組同時覆寫 fixed_risk。")
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=None,
        help="選填；兩組同時覆寫 max_position_cap_pct。",
    )
    parser.add_argument(
        "--allow-static-diagnostic",
        action="store_true",
        help="允許以單一／static run_best 做非 OOS 診斷；不得視為無前視部署證據。",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="選填；將比較期間縮限於Score與active params共同可用範圍內。",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="選填；必須與--start-date同時使用。",
    )
    parser.add_argument(
        "--attribution-only",
        action="store_true",
        help="只讀取既有 strategy_compare CSV/JSON 產生交易歸因，不重新執行 portfolio replay。",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


















































def render_strategy_pair_simple_report(
    payload: dict[str, Any],
    *,
    color: bool | None = None,
) -> str:
    """Compatibility façade over the canonical reporting module."""

    metadata, baseline, quality, delta, yearly, diagnostics = (
        _strategy_pair_report_components(payload)
    )
    return _render_strategy_console_report(
        metadata, baseline, quality, delta, yearly, diagnostics, color=color
    )


def render_strategy_pair_markdown(payload: dict[str, Any]) -> str:
    """Compatibility façade over the canonical reporting module."""

    metadata, baseline, quality, delta, yearly, diagnostics = (
        _strategy_pair_report_components(payload)
    )
    return _markdown_report(metadata, baseline, quality, delta, yearly, diagnostics)


def materialize_strategy_pair_readable_report(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
) -> Path:
    """Persist the canonical pair report while preserving the historical engine API."""

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "strategy_comparison.md"
    report_path.write_text(render_strategy_pair_markdown(payload), encoding="utf-8")
    return report_path












def run_existing_attribution(*, project_root=PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    comparison_mode = COMPARISON_MODE_HARD_FILTER
    labels = _comparison_labels(comparison_mode)
    filter_id = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
    contract = load_runtime_artifact_contract(str(root), filter_id)
    architecture = str(contract.manifest.get("model_architecture") or "")
    experiment_profile = str(contract.manifest.get("experiment_profile") or "")
    output_root = resolve_filter_model_output_dir(
        str(root), filter_id, architecture, experiment_profile
    )
    output_dir = _first_existing_comparison_dir(
        output_root,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
    )
    payload = _load_existing_comparison_payload(output_dir)
    metadata = dict(payload["metadata"] or {})
    metadata.setdefault("comparison_mode", COMPARISON_MODE_HARD_FILTER)
    if str(metadata.get("filter_id") or "") != filter_id:
        raise ValueError("既有策略比較結果的 filter_id 與目前 active filter 不一致")
    if str(metadata.get("model_architecture") or "") != architecture:
        raise ValueError("既有策略比較結果的 architecture 與目前 runtime artifact 不一致")
    if str(metadata.get("experiment_profile") or "") != experiment_profile:
        raise ValueError("既有策略比較結果的 experiment profile 與目前 runtime artifact 不一致")

    no_filter_trades_path = output_dir / "no_filter_trades.csv"
    quality_trades_path = output_dir / "quality_filter_trades.csv"
    if not no_filter_trades_path.is_file() or not quality_trades_path.is_file():
        raise FileNotFoundError(
            "attribution-only 需要既有 no_filter_trades.csv 與 quality_filter_trades.csv；"
            "請先完成一次正式 strategy compare。"
        )
    no_filter_history = pd.read_csv(no_filter_trades_path, encoding="utf-8-sig")
    quality_history = pd.read_csv(quality_trades_path, encoding="utf-8-sig")

    yearly = _normalize_yearly_completeness(pd.DataFrame(payload.get("yearly") or []))
    baseline = dict(payload.get("no_filter") or {})
    quality = dict(payload.get("quality_filter") or {})
    if not yearly.empty:
        _refresh_yearly_summary(baseline, yearly, return_column="no_filter_return_pct")
        _refresh_yearly_summary(quality, yearly, return_column="quality_filter_return_pct")
    deltas = _delta(quality, baseline)
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["trade_attribution_schema_version"] = ATTRIBUTION_SCHEMA_VERSION
    refreshed = _to_json_native({
        "metadata": metadata,
        "no_filter": baseline,
        labels["active_name"]: quality,
        f"{labels['active_name']}_minus_no_filter": deltas,
        "yearly": yearly.to_dict("records"),
    })
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(refreshed, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    materialize_strategy_pair_readable_report(
        refreshed,
        output_dir=output_dir,
    )
    yearly.to_csv(output_dir / "yearly_returns_comparison.csv", index=False, encoding="utf-8-sig")
    attribution = write_trade_attribution_outputs(
        project_root=root,
        output_dir=output_dir,
        filter_id=filter_id,
        threshold=float(metadata.get("threshold", BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)),
        metadata=metadata,
        no_filter_trade_history=no_filter_history,
        quality_filter_trade_history=quality_history,
        no_filter_portfolio_total_r=baseline.get("portfolio_total_r"),
        quality_filter_portfolio_total_r=quality.get("portfolio_total_r"),
    )
    print_artifact_paths(
        (("交易歸因 Markdown", output_dir / "trade_attribution.md"),),
        project_root=root,
    )
    return attribution










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


def run_comparison(
    *, project_root=PROJECT_ROOT, dataset="full", params_path=None,
    param_policy=PARAM_POLICY_AUTO, max_positions=DEFAULT_PORTFOLIO_MAX_POSITIONS, enable_rotation=False,
    fixed_risk=None, max_position_cap_pct=None, allow_static_diagnostic=False,
    comparison_mode=COMPARISON_MODE_HARD_FILTER,
    ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    ranking_options=None,
    optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
    model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    experiment_profile=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    threshold=None,
    output_dir_override=None,
    comparison_start_date=None,
    comparison_end_date=None,
    quiet=False,
    shared_param_overrides=None,
    hard_filter_source=None,
    baseline_reuse_dir=None,
    continuous_score_path_override=None,
    continuous_score_execution_start_override=None,
    selection_pit_score_path_override=None,
    selection_pit_manifest_path_override=None,
    selection_pit_expected_seed_override=None,
    capture_execution_diagnostics=False,
    capture_selection_target_diagnostics=True,
):
    root = Path(project_root).resolve()
    comparison_mode = str(comparison_mode)
    ranking_policy = str(ranking_policy).strip()
    ranking_options = dict(ranking_options or {})
    optional_entry_filter_policy = str(optional_entry_filter_policy).strip()
    score_source = str(score_source)
    filter_id = str(filter_id)
    model_architecture = str(model_architecture)
    experiment_profile = str(experiment_profile)
    configured_threshold = (
        float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)
        if threshold is None
        else float(threshold)
    )
    if not 0.0 <= configured_threshold <= 1.0:
        raise ValueError("threshold必須介於0與1")
    labels = _comparison_labels(comparison_mode)
    if score_source not in SUPPORTED_RANKING_SCORE_SOURCES:
        raise ValueError(f"不支援的 score source: {score_source!r}")
    if ranking_policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的 ranking policy: {ranking_policy!r}")
    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD:
        raw_max_age = ranking_options.get("stale_score_membership_guard_max_age_days")
        if isinstance(raw_max_age, bool) or not isinstance(raw_max_age, int) or raw_max_age < 0:
            raise ValueError(
                "stale-score membership guard必須由config提供非負整數 stale_score_membership_guard_max_age_days"
            )
    if optional_entry_filter_policy not in SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES:
        raise ValueError(
            f"不支援的 optional entry filter policy: {optional_entry_filter_policy!r}"
        )
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
        raise ValueError(
            "continuous_score_path_override與continuous_score_execution_start_override必須成對提供"
        )
    has_selection_pit_score_override = selection_pit_score_path_override not in (None, "")
    has_selection_pit_manifest_override = selection_pit_manifest_path_override not in (None, "")
    if has_selection_pit_score_override != has_selection_pit_manifest_override:
        raise ValueError(
            "selection_pit_score_path_override與selection_pit_manifest_path_override必須成對提供"
        )
    if has_selection_pit_score_override and score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        raise ValueError("Selection PIT isolated override只支援selection_point_in_time score source")

    runtime_contract = None
    pit_contract = None
    continuous_contract = None
    continuous_score_override = None
    ranking_source = None
    filter_source = None
    binary_filter_manifest = None
    binary_filter_manifest_path = None
    binary_filter_scores_path = None
    effective_score_source = score_source
    if hard_filter_source is not None:
        source_payload = dict(hard_filter_source or {})
        source_name = str(source_payload.get("score_source") or "").strip()
        if source_name != BINARY_PIT_SCORE_SOURCE:
            raise ValueError(
                "hard_filter_source目前只接受binary_point_in_time: "
                f"actual={source_name!r}"
            )
        binary_filter_manifest_path = Path(
            str(source_payload.get("manifest_path") or "")
        ).resolve()
        binary_filter_scores_path = Path(
            str(source_payload.get("scores_path") or "")
        ).resolve()
        _score_table, binary_filter_manifest = load_binary_point_in_time_score_table(
            str(binary_filter_manifest_path), str(binary_filter_scores_path)
        )
        manifest_architecture = str(
            binary_filter_manifest.get("model_architecture") or ""
        )
        manifest_profile = str(
            binary_filter_manifest.get("experiment_profile") or ""
        )
        if manifest_architecture != model_architecture or manifest_profile != experiment_profile:
            raise ValueError(
                "Binary PIT artifact與指定模型identity不一致: "
                f"artifact={manifest_architecture}/{manifest_profile}, "
                f"requested={model_architecture}/{experiment_profile}"
            )
        artifact_threshold = float(binary_filter_manifest.get("threshold", float("nan")))
        if not math.isclose(
            artifact_threshold, configured_threshold, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                "Binary PIT threshold與固定threshold不一致: "
                f"artifact={artifact_threshold}, policy={configured_threshold}"
            )
        period = dict(binary_filter_manifest.get("score_period") or {})
        start_date = str(period.get("start") or "")
        end_date = str(period.get("end") or "")
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
        if ranking_policy not in {
            BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
            BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
            BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
            BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
            BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
        }:
            raise ValueError(
                "Continuous ranker OOS source只允許resource-aware-continuous系列policy"
            )
        if continuous_score_path_override not in (None, ""):
            override_path = Path(str(continuous_score_path_override)).resolve()
            override_table = load_continuous_ranker_oos_score_table_from_path(
                str(override_path), experiment_profile
            )
            manifest_architecture = model_architecture
            manifest_profile = experiment_profile
            execution_start, available_from, available_through = (
                _resolve_continuous_score_override_period(
                    override_table,
                    execution_start_override=continuous_score_execution_start_override,
                )
            )
            start_date = execution_start
            end_date = available_through
            continuous_score_override = {
                "score_path": str(override_path),
                "execution_start": execution_start,
                "available_from": available_from,
                "available_through": available_through,
            }
        else:
            continuous_contract = load_continuous_ranker_oos_contract(
                str(root), filter_id, model_architecture, experiment_profile
            )
            manifest_architecture = continuous_contract.model_architecture
            manifest_profile = continuous_contract.experiment_profile
            start_date = continuous_contract.execution_start
            end_date = continuous_contract.available_through
        ranking_source = {
            "score_source": SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
            "model_architecture": manifest_architecture,
            "experiment_profile": manifest_profile,
            "ranking_policy": ranking_policy,
            "ranking_options": ranking_options,
            "score_path_override": (
                None if continuous_score_override is None else continuous_score_override["score_path"]
            ),
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
                expected_seed=(
                    None if selection_pit_expected_seed_override is None
                    else int(selection_pit_expected_seed_override)
                ),
            )
        else:
            pit_contract = load_selection_point_in_time_ranking_contract(
                str(root), filter_id, model_architecture, experiment_profile
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
        manifest_architecture = pit_contract.model_architecture
        manifest_profile = pit_contract.experiment_profile
        start_date = pit_contract.available_from
        end_date = pit_contract.available_through
        ranking_source = {
            "score_source": score_source,
            "model_architecture": manifest_architecture,
            "experiment_profile": manifest_profile,
            "ranking_policy": ranking_policy,
            "ranking_options": ranking_options,
            "score_path_override": (
                str(pit_contract.score_path) if has_selection_pit_score_override else None
            ),
            "score_manifest_path_override": (
                str(pit_contract.manifest_path) if has_selection_pit_score_override else None
            ),
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
                f"artifact={manifest_architecture}/{manifest_profile}, "
                f"requested={model_architecture}/{experiment_profile}"
            )
        artifact_threshold = float(
            runtime_contract.manifest.get("fixed_evaluation_threshold", float("nan"))
        )
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
        default_start = pd.Timestamp(start_date).normalize()
        default_end = pd.Timestamp(end_date).normalize()
        requested_start = pd.Timestamp(str(comparison_start_date)).normalize()
        requested_end = pd.Timestamp(str(comparison_end_date)).normalize()
        if pd.isna(requested_start) or pd.isna(requested_end) or requested_end < requested_start:
            raise ValueError("指定策略比較期間不合法")
        if requested_start < default_start or requested_end > default_end:
            raise ValueError(
                "指定策略比較期間超出Score可用範圍："
                f"requested={requested_start.date()}~{requested_end.date()}, "
                f"available={default_start.date()}~{default_end.date()}"
            )
        start_date = requested_start.strftime("%Y-%m-%d")
        end_date = requested_end.strftime("%Y-%m-%d")

    param_policy = str(param_policy)
    resolved_params_path = _resolve_params_path(
        root=root, params_path=params_path, param_policy=param_policy,
        allow_static_diagnostic=allow_static_diagnostic, score_source=score_source,
    )
    if not resolved_params_path.is_file():
        source_hint = (
            "請先完成Selection nested ROOS active-param工件"
            if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "請先完成正式rolling OOS active-param工件"
        )
        raise FileNotFoundError(
            f"找不到策略比較參數檔: {resolved_params_path}；{source_hint}。"
        )
    if fixed_risk is not None and not (0.0 < float(fixed_risk) <= 1.0):
        raise ValueError("fixed_risk必須介於0與1")
    if max_position_cap_pct is not None and not (0.0 < float(max_position_cap_pct) <= 1.0):
        raise ValueError("max_position_cap_pct必須介於0與1")
    param_source = _load_param_source(resolved_params_path)
    param_policy_contract = _validate_requested_param_policy(param_source, param_policy)
    (
        param_source_kind,
        no_filter_params,
        quality_params,
        no_filter_param_payload,
        quality_param_payload,
        ensemble_policy,
    ) = _build_controlled_param_source_pair(
        param_source,
        filter_id=filter_id,
        threshold=configured_threshold,
        fixed_risk=None if fixed_risk is None else float(fixed_risk),
        max_position_cap_pct=(
            None if max_position_cap_pct is None else float(max_position_cap_pct)
        ),
        comparison_mode=comparison_mode,
        optional_entry_filter_policy=optional_entry_filter_policy,
        shared_param_overrides=shared_param_overrides,
    )

    is_rolling_source = param_source_kind in {
        "rolling_oos_param_schedule", "rolling_active_param_ensemble"
    }
    if not is_rolling_source and not allow_static_diagnostic:
        raise ValueError(
            "單一／static參數跨歷史回放無法證明無前視；"
            "請使用rolling active-param JSON，或加--allow-static-diagnostic僅作敏感度診斷。"
        )
    data_dir = Path(get_dataset_dir(str(root), dataset)).resolve()
    if param_source_kind == "rolling_oos_param_schedule":
        param_start, param_end = get_active_param_date_range(param_source["payload"])
    elif param_source_kind == "rolling_active_param_ensemble":
        param_start, param_end = get_active_param_ensemble_date_range(param_source["payload"])
    else:
        param_start, param_end = "", ""
    if is_rolling_source and (
        pd.Timestamp(param_start) > pd.Timestamp(start_date)
        or pd.Timestamp(param_end) < pd.Timestamp(end_date)
    ):
        raise ValueError(
            "rolling active-param期間未完整覆蓋策略比較期間："
            f"params={param_start}~{param_end}, comparison={start_date}~{end_date}"
        )

    baseline_replay_counts = {} if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
    baseline_summary_override = None
    baseline_orderable_reused = None
    if baseline_reuse_dir is not None:
        baseline_payload, baseline_summary_override, baseline_orderable_reused = (
            _load_reusable_no_filter_baseline(
                Path(baseline_reuse_dir),
                comparison_mode=comparison_mode,
                expected_dataset=dataset,
                expected_params_sha256=_sha256_file(resolved_params_path),
                expected_param_policy=param_policy,
                expected_optional_entry_filter_policy=optional_entry_filter_policy,
                expected_shared_param_overrides=dict(shared_param_overrides or {}),
                expected_max_positions=max_positions,
                expected_enable_rotation=enable_rotation,
                expected_start_date=start_date,
                expected_end_date=end_date,
            )
        )
        baseline_replay_counts = None
        if not quiet:
            print(
                "[no_filter] 重用既有正式baseline replay："
                + project_relative_display_path(
                    Path(baseline_reuse_dir).resolve(),
                    project_root=root,
                )
            )
    else:
        baseline_payload = _run_scenario(
            name="no_filter", data_dir=data_dir, param_source_kind=param_source_kind,
            params=no_filter_params, start_date=start_date, end_date=end_date,
            max_positions=max_positions, enable_rotation=enable_rotation, quiet=quiet,
            replay_counts=baseline_replay_counts,
        )
    quality_replay_counts = {} if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
    quality_execution_rows = (
        []
        if bool(capture_execution_diagnostics)
        and comparison_mode == COMPARISON_MODE_SCORE_RANKING
        else None
    )
    quality_selector_trace_rows = (
        [] if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
    )
    quality_payload = _run_scenario(
        name=labels["active_name"], data_dir=data_dir,
        param_source_kind=param_source_kind, params=quality_params,
        start_date=start_date, end_date=end_date, max_positions=max_positions,
        enable_rotation=enable_rotation, quiet=quiet,
        replay_counts=quality_replay_counts, replay_execution_rows=quality_execution_rows,
        replay_selector_trace_rows=quality_selector_trace_rows,
        ranking_source=ranking_source, filter_source=filter_source,
    )
    _assert_shared_benchmark(baseline_payload, quality_payload)

    baseline = (
        dict(baseline_summary_override)
        if baseline_summary_override is not None
        else _scenario_summary(baseline_payload)
    )
    quality = _scenario_summary(quality_payload)
    deltas = _delta(quality, baseline)
    yearly = _build_yearly_comparison(
        baseline_payload["profile"], quality_payload["profile"]
    )
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        yearly = yearly.rename(
            columns={"quality_filter_return_pct": "score_ranking_return_pct"}
        )

    if output_dir_override is None:
        output_dir_name = _comparison_output_dir_name(
            comparison_mode,
            labels,
            param_policy=param_policy,
            ranking_policy=ranking_policy,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )
        if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            output_dir_name += "_selection_point_in_time"
        output_dir = (
            resolve_filter_model_output_dir(
                str(root), filter_id, manifest_architecture, manifest_profile
            ) / output_dir_name
        )
        output_scope = "canonical"
    else:
        output_dir = Path(output_dir_override)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir = output_dir.resolve()
        output_scope = "caller_override"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_diagnostics = None
    baseline_selected_joined = pd.DataFrame()
    quality_selected_joined = pd.DataFrame()
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        baseline_orderable = (
            pd.DataFrame(baseline_orderable_reused).copy()
            if baseline_orderable_reused is not None
            else _flatten_candidate_replay_rows(
                baseline_replay_counts or {}, "orderable_rows"
            )
        )
        quality_orderable = _flatten_candidate_replay_rows(
            quality_replay_counts or {}, "orderable_rows"
        )
        baseline_selected = _flatten_selected_buy_rows(baseline_payload["trade_history"])
        quality_selected = _flatten_selected_buy_rows(quality_payload["trade_history"])
        baseline_orderable.to_csv(
            output_dir / "no_filter_orderable_candidates.csv", index=False,
            encoding="utf-8-sig"
        )
        quality_orderable.to_csv(
            output_dir / "score_ranking_orderable_candidates.csv", index=False,
            encoding="utf-8-sig"
        )
        baseline_selected.to_csv(
            output_dir / "no_filter_selected_buys.csv", index=False,
            encoding="utf-8-sig"
        )
        quality_selected.to_csv(
            output_dir / "score_ranking_selected_buys.csv", index=False,
            encoding="utf-8-sig"
        )
        if quality_execution_rows is not None:
            _flatten_entry_execution_rows(quality_execution_rows).to_csv(
                output_dir / "score_ranking_execution.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if quality_selector_trace_rows is not None:
            _flatten_selector_trace_rows(quality_selector_trace_rows).to_csv(
                output_dir / "score_ranking_selector_trace.csv",
                index=False,
                encoding="utf-8-sig",
            )
            _flatten_repair_mechanism_rows(quality_selector_trace_rows).to_csv(
                output_dir / "score_ranking_repair_search_certificate.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if (
            score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            and bool(capture_selection_target_diagnostics)
        ):
            lookup = _selection_target_lookup(
                root=root, filter_id=filter_id, architecture=manifest_architecture,
                profile=manifest_profile,
                score_path_override=(
                    str(pit_contract.score_path) if has_selection_pit_score_override else None
                ),
                manifest_path_override=(
                    str(pit_contract.manifest_path) if has_selection_pit_score_override else None
                ),
            )
            baseline_diag, baseline_orderable_joined, baseline_selected_joined = (
                _strategy_selection_diagnostics(
                    orderable=baseline_orderable,
                    selected=baseline_selected,
                    lookup=lookup,
                )
            )
            quality_diag, quality_orderable_joined, quality_selected_joined = (
                _strategy_selection_diagnostics(
                    orderable=quality_orderable,
                    selected=quality_selected,
                    lookup=lookup,
                )
            )
            strategy_diagnostics = {
                "no_filter": baseline_diag,
                "score_ranking": quality_diag,
                "score_ranking_minus_no_filter": _delta(quality_diag, baseline_diag),
                "future_target_join_stage": "post_replay_offline_diagnostic_only",
                "future_target_used_for_runtime_sort": False,
            }
            baseline_orderable_joined.to_csv(
                output_dir / "no_filter_orderable_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )
            quality_orderable_joined.to_csv(
                output_dir / "score_ranking_orderable_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )
            baseline_selected_joined.to_csv(
                output_dir / "no_filter_selected_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )
            quality_selected_joined.to_csv(
                output_dir / "score_ranking_selected_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )

    if binary_filter_manifest is not None:
        binary_period = dict(binary_filter_manifest.get("score_period") or {})
        score_signal_coverage = {
            "required_start": str(binary_period.get("start") or ""),
            "first_scored_event": str(binary_period.get("start") or ""),
            "available_through": str(binary_period.get("end") or ""),
        }
        score_artifact_metadata = {
            "binary_pit_manifest_path": str(binary_filter_manifest_path),
            "binary_pit_score_path": str(binary_filter_scores_path),
            "score_table": dict(binary_filter_manifest.get("score_table") or {}),
            "binary_pit_information_contract": str(
                binary_filter_manifest.get("information_contract") or ""
            ),
        }
    elif pit_contract is not None:
        score_signal_coverage = {
            "required_start": pit_contract.available_from,
            "first_scored_event": pit_contract.available_from,
            "available_through": pit_contract.available_through,
        }
        score_artifact_metadata = {
            "score_manifest_path": str(pit_contract.manifest_path),
            "score_path": str(pit_contract.score_path),
            "score_audit_path": (None if pit_contract.audit_path is None else str(pit_contract.audit_path)),
            "model_validation_gate": pit_contract.model_validation_gate,
            "runtime_eligibility": dict(pit_contract.manifest.get("runtime_eligibility") or {}),
            "score_table": dict(pit_contract.manifest.get("artifacts", {}).get("scores") or {}),
        }
    elif continuous_score_override is not None:
        score_signal_coverage = {
            "required_start": continuous_score_override["execution_start"],
            "first_scored_event": continuous_score_override["available_from"],
            "available_through": continuous_score_override["available_through"],
        }
        score_artifact_metadata = {
            "score_path": str(continuous_score_override["score_path"]),
            "runtime_eligibility": "multi_seed_robustness_isolated_replay",
            "artifact_override": True,
        }
    elif continuous_contract is not None:
        score_signal_coverage = {
            "required_start": continuous_contract.execution_start,
            "first_scored_event": continuous_contract.available_from,
            "available_through": continuous_contract.available_through,
        }
        score_artifact_metadata = {
            "score_manifest_path": str(continuous_contract.manifest_path),
            "score_path": str(continuous_contract.score_path),
            "continuous_ranker_report_path": str(continuous_contract.report_path),
            "continuous_target_id": continuous_contract.continuous_target_id,
            "training_label_scope": str(continuous_contract.manifest.get("training_label_scope") or ""),
            "runtime_eligibility": "research_strategy_replay_only",
        }
    else:
        score_signal_coverage = {
            "required_start": runtime_contract.required_signal_start.isoformat(),
            "first_scored_event": runtime_contract.available_from.isoformat(),
            "available_through": runtime_contract.available_through.isoformat(),
        }
        score_artifact_metadata = {
            "runtime_manifest_path": str(runtime_contract.paths.manifest_path),
            "runtime_score_path": str(runtime_contract.paths.score_path),
            "runtime_eligibility": dict(runtime_contract.manifest.get("runtime_eligibility") or {}),
            "score_table": dict(runtime_contract.manifest.get("score_table") or {}),
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": comparison_mode,
        "score_ranking_policy": ranking_policy if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None,
        "score_ranking_options": ranking_options if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None,
        "optional_entry_filter_policy": optional_entry_filter_policy,
        "optional_entry_filter_fields": list(OPTIONAL_ENTRY_FILTER_FIELDS),
        "optional_entry_filter_forced_values": (
            {field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS}
            if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            else None
        ),
        "shared_param_overrides": dict(shared_param_overrides or {}),
        "score_source": effective_score_source,
        "hard_filter_source_transport": (
            "contextvar_and_process_environment" if hard_filter_source is not None else None
        ),
        "dataset": dataset,
        "data_dir": str(data_dir),
        "params_path": str(resolved_params_path),
        "params_file_sha256": _sha256_file(resolved_params_path),
        "param_source_kind": param_source_kind,
        "requested_param_policy": param_policy,
        "param_selector": param_policy_contract["selector"],
        "runtime_member_count_min": param_policy_contract["member_count_min"],
        "runtime_member_count_max": param_policy_contract["member_count_max"],
        "runtime_min_agree": param_policy_contract["min_agree"],
        "ranking_scope": (
            "all_candidates_after_single_member_qualification"
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            and param_policy_contract["selector"] == "base_finalist_best"
            else "same_runtime_vote_bucket_only"
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
        ),
        "comparison_design": (
            "selection_point_in_time_active_param_replay"
            if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "continuous_ranker_oos_capital_utilization_first_replay"
            if score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
            else "historical_active_param_oos" if is_rolling_source
            else "static_param_diagnostic"
        ),
        "lookahead_safe_active_param_schedule": bool(is_rolling_source),
        "active_param_period": {"start": param_start, "end": param_end}
        if is_rolling_source else None,
        "active_param_ensemble_policy": ensemble_policy,
        "no_filter_params": no_filter_param_payload,
        f"{labels['active_name']}_params": quality_param_payload,
        "filter_id": filter_id,
        "model_architecture": manifest_architecture,
        "experiment_profile": manifest_profile,
        "threshold": (
            None
            if score_source in {
                SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
                SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            }
            else configured_threshold
        ),
        "threshold_used_as_gate": bool(comparison_mode == COMPARISON_MODE_HARD_FILTER),
        "score_ranking_order": (
            (
                []
                if param_policy_contract["selector"] == "base_finalist_best"
                else ["runtime_member_vote_count_desc"]
            )
            + (
                ["breakout_quality_score_desc"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_SCORE
                else ["capital_adjusted_score_desc"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED
                else ["resource_bottleneck_gate", "binary_pass_promotions", "existing_buy_sort"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY
                else ["resource_bottleneck_gate", "best_improvement_pass_basket", "existing_buy_sort"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET
                else [
                    "same_param_exact_resource_baseline",
                    "fixed_baseline_order_count",
                    "frozen_mr13e_daily_percentile_to_expected_r",
                    "expected_r_times_canonical_planned_initial_risk_top_k",
                    "deterministic_minimum_repair_seed",
                    "best_feasible_single_swap_expected_pnl_ascent",
                    "one_swap_local_optimum",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                else [
                    "same_param_exact_resource_baseline",
                    "fixed_baseline_order_count",
                    "frozen_mr13e_daily_percentile_to_pit_expected_excess_r",
                    "expected_excess_r_times_canonical_planned_initial_risk_top_k",
                    "deterministic_minimum_repair_seed",
                    "best_feasible_single_swap_excess_alpha_ascent",
                    "one_swap_local_optimum",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT
                else [
                    "same_param_exact_resource_baseline",
                    "fixed_baseline_order_count",
                    "frozen_mr13e_daily_percentile_to_pit_expected_excess_r",
                    "expected_excess_r_times_canonical_planned_initial_risk_top_k",
                    "no_baseline_reserved_capital_floor",
                    "no_r0_minimum_repair",
                    "k_only_cash_feasible_seed_if_raw_top_k_cannot_place_k_orders",
                    "best_k_only_cash_feasible_single_swap_excess_alpha_ascent",
                    "one_swap_local_optimum",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                else [
                    "same_param_exact_resource_baseline",
                    "fixed_baseline_order_count",
                    "continuous_score_top_k",
                    "deterministic_minimum_repair_seed",
                    "stale_score_membership_guard",
                    "fresh_only_best_feasible_single_swap_ascent",
                    "one_swap_local_optimum",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
                else [
                    "same_param_exact_resource_baseline",
                    "fixed_baseline_order_count",
                    "continuous_score_top_k",
                    "deterministic_minimum_repair_seed",
                    "best_feasible_single_swap_ascent",
                    "one_swap_local_optimum",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT
                else [
                    "same_param_exact_resource_baseline",
                    "fixed_baseline_order_count",
                    "continuous_score_top_k",
                    "deterministic_minimum_repair_if_needed",
                    "reserved_capital_not_below_baseline",
                    "baseline_fallback_only_if_repair_fails",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL
                else [
                    "same_param_exact_resource_baseline",
                    "continuous_score_desc",
                    "selected_count_not_below_baseline",
                    "reserved_capital_not_below_baseline",
                    "existing_buy_sort_fallback",
                ]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                else ["resource_bottleneck_gate", "continuous_score_desc_if_cash_binding", "existing_buy_sort_fallback"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS
                else ["capital_deployment_bucket_desc", "breakout_quality_score_desc"]
            )
            + (["ticker_deterministic"] if ranking_policy in {BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT} else ["existing_buy_sort", "ticker_deterministic"])
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
        ),
        "capital_aware_ranking_contract": (
            {
                "resource_gate": "canonical_same_param_exact_cash_cap_baseline",
                "dl_intervention": (
                    "all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count"
                    if ranking_policy in {BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT}
                    else "cash_or_slot_binding_with_exact_baseline_resource_preservation"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                    else "only_when_baseline_stops_before_free_slots_with_unselected_candidates"
                ),
                "quality_objective": (
                    "maximize_sum_expected_r_times_canonical_planned_initial_risk_subject_to_same_k_r0"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                    else "maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_r0"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT
                    else "maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_and_true_cash_only"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                    else "maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor"
                    if ranking_policy in {BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD}
                    else "maximize_selected_continuous_score_subject_to_same_param_baseline_resource_floor"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                    else "maximize_selected_continuous_score_without_breaking_cash_binding"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS
                    else "increase_reserved_capital_assigned_to_pass_candidates"
                ),
                "resource_feasibility": (
                    "selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost"
                    if ranking_policy in {BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT, BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT}
                    else "selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains binding"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                    else "selected_count>=baseline_selected_count and reserved_cost>=baseline_reserved_cost"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                    else "cash remains the binding pre-market resource after the selected basket"
                ),
                "selection_objective": (
                    "expected_pnl_top_k_repair_seed_then_best_feasible_single_swap_expected_pnl_ascent"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                    else "expected_excess_alpha_top_k_repair_seed_then_best_feasible_single_swap_excess_alpha_ascent"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT
                    else "expected_excess_alpha_top_k_then_k_only_cash_feasible_seed_if_needed_then_best_single_swap_ascent"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                    else "dl_top_k_repair_seed_then_stale_membership_guard_then_fresh_only_best_feasible_single_swap_ascent"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
                    else "dl_top_k_repair_seed_then_best_feasible_single_swap_ascent"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT
                    else "dl_top_k_then_deterministic_minimum_repair"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL
                    else "best_improvement_continuous_score_with_exact_resource_floor"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                    else "best_improvement_pass_reserved_then_pass_count_then_min_roos_rank"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET
                    else "continuous_score_desc_with_cash_binding_constrained_promotions"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS
                    else "first_improving_pass_reserved_promotion"
                ),
                "fallback": (
                    "same_param_baseline_only_if_expected_pnl_minimum_repair_cannot_reach_k_r0_then_expected_pnl_ascent_continues"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                    else "same_param_baseline_only_if_excess_alpha_minimum_repair_cannot_reach_k_r0_then_excess_alpha_ascent_continues"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT
                    else "same_param_baseline_is_only_a_k_cash_feasible_seed_when_raw_top_k_cannot_place_k_orders; no_r0_repair"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                    else "stale_score_membership_change_blocked_to_same_param_baseline_or_fresh_only_ascent"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
                    else "minimum_repair_seed_may_use_same_param_baseline_then_feasible_ascent_continues"
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT
                    else "canonical_same_param_baseline_order"
                ),
                "future_target_used": False,
                "additional_numeric_thresholds": (
                    [{
                        "name": "stale_score_membership_guard_max_age_days",
                        "value": ranking_options.get("stale_score_membership_guard_max_age_days"),
                        "unit": "calendar_days",
                        "source": "config/strategy_compare.py",
                    }]
                    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
                    else []
                ),
            }
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            and ranking_policy in {
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            }
            else {
                "projected_capital_fraction_source": "canonical_pretrade_proj_cost_div_sizing_capital",
                "deployment_rate": "min(1, projected_capital_fraction / max_position_cap_pct)",
                "capital_bucket_count": 3,
                "capital_bucket_scope": "same_day_score_available_orderable_candidates",
                "future_target_used": False,
            }
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            and ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE
            else None
        ),
        "unscorable_candidate_policy": "fallback_original_buy_sort_without_exclusion",
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "fixed_risk_override": None if fixed_risk is None else float(fixed_risk),
        "max_position_cap_pct_override": (
            None if max_position_cap_pct is None else float(max_position_cap_pct)
        ),
        "output_scope": output_scope,
        "execution_diagnostics_captured": bool(quality_execution_rows is not None),
        "selection_target_diagnostics_captured": bool(strategy_diagnostics is not None),
        "baseline_reused": baseline_reuse_dir is not None,
        "baseline_reuse_source": (
            None
            if baseline_reuse_dir is None
            else project_relative_display_path(
                Path(baseline_reuse_dir).resolve(),
                project_root=root,
            )
        ),
        "output_dir": project_relative_display_path(output_dir, project_root=root),
        "comparison_period": {"start": start_date, "end": end_date},
        "score_signal_coverage": score_signal_coverage,
        "benchmark_ticker": PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
        "controlled_param_difference": [_comparison_switch_spec(comparison_mode)[0]],
        "trade_attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        **score_artifact_metadata,
    }
    json_payload = _to_json_native({
        "metadata": metadata,
        "no_filter": baseline,
        labels["active_name"]: quality,
        f"{labels['active_name']}_minus_no_filter": deltas,
        "yearly": yearly.to_dict("records"),
        "selection_diagnostics": strategy_diagnostics,
    })
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    materialize_strategy_pair_readable_report(
        json_payload,
        output_dir=output_dir,
    )
    _remove_legacy_html_outputs(output_dir)
    baseline_payload["equity_curve"].to_csv(
        output_dir / "no_filter_equity.csv", index=False, encoding="utf-8-sig"
    )
    quality_payload["equity_curve"].to_csv(
        output_dir / f"{labels['active_name']}_equity.csv", index=False,
        encoding="utf-8-sig"
    )
    baseline_payload["trade_history"].to_csv(
        output_dir / "no_filter_trades.csv", index=False, encoding="utf-8-sig"
    )
    quality_payload["trade_history"].to_csv(
        output_dir / f"{labels['active_name']}_trades.csv", index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(
        baseline_payload["profile"].get("portfolio_capacity_rows") or []
    ).to_csv(
        output_dir / "no_filter_daily_capacity.csv", index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(
        quality_payload["profile"].get("portfolio_capacity_rows") or []
    ).to_csv(
        output_dir / f"{labels['active_name']}_daily_capacity.csv", index=False,
        encoding="utf-8-sig"
    )
    yearly.to_csv(
        output_dir / "yearly_returns_comparison.csv", index=False,
        encoding="utf-8-sig"
    )
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        write_trade_attribution_outputs(
            project_root=root, output_dir=output_dir, filter_id=filter_id,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            metadata=metadata,
            no_filter_trade_history=baseline_payload["trade_history"],
            quality_filter_trade_history=quality_payload["trade_history"],
            no_filter_closed_trade_rows=(baseline_payload["profile"] or {}).get("closed_trade_rows"),
            quality_filter_closed_trade_rows=(quality_payload["profile"] or {}).get("closed_trade_rows"),
            no_filter_portfolio_total_r=baseline.get("portfolio_total_r"),
            quality_filter_portfolio_total_r=quality.get("portfolio_total_r"),
        )
    if not quiet:
        print("\n" + render_strategy_pair_simple_report(json_payload))
        artifacts = [
            ("策略比較 Markdown", output_dir / "strategy_comparison.md"),
            ("策略比較 JSON", output_dir / "strategy_comparison.json"),
            ("年度比較 CSV", output_dir / "yearly_returns_comparison.csv"),
        ]
        if comparison_mode == COMPARISON_MODE_HARD_FILTER:
            artifacts.append(("交易歸因 Markdown", output_dir / "trade_attribution.md"))
        print_artifact_paths(artifacts, project_root=root)
    return json_payload

def main(argv=None):
    args = _parse_args(argv)
    if args.attribution_only:
        if args.comparison_mode != COMPARISON_MODE_HARD_FILTER:
            raise ValueError("--attribution-only 目前只支援 hard-filter 既有歸因")
        run_existing_attribution()
        return 0
    if args.max_positions < 1:
        raise ValueError("max_positions 必須 >= 1")
    if (args.start_date is None) != (args.end_date is None):
        raise ValueError("--start-date與--end-date必須同時使用")
    run_comparison(
        dataset=args.dataset,
        params_path=args.params,
        param_policy=args.param_policy,
        max_positions=args.max_positions,
        enable_rotation=args.rotation == "on",
        fixed_risk=args.fixed_risk,
        max_position_cap_pct=args.max_position_cap_pct,
        allow_static_diagnostic=args.allow_static_diagnostic,
        comparison_mode=args.comparison_mode,
        ranking_policy=args.ranking_policy,
        ranking_options=(
            {}
            if args.stale_score_membership_guard_max_age_days is None
            else {
                "stale_score_membership_guard_max_age_days": int(
                    args.stale_score_membership_guard_max_age_days
                )
            }
        ),
        optional_entry_filter_policy=args.optional_entry_filters,
        filter_id=args.filter_id,
        score_source=args.score_source,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        comparison_start_date=args.start_date,
        comparison_end_date=args.end_date,
        quiet=args.quiet,
    )
    return 0


__all__ = [
    "render_strategy_pair_simple_report",
    "render_strategy_pair_markdown",
    "materialize_strategy_pair_readable_report",
    "main", "run_comparison", "run_standalone_baseline", "run_existing_attribution",
    "canonical_strategy_compare_output_dir_names", "_first_existing_comparison_dir",
    "_assert_controlled_param_pair", "_assert_controlled_ensemble_pair",
    "_assert_controlled_payload_pair", "_build_controlled_param_source_pair",
    "_load_param_source", "_capacity_summary", "_normalize_yearly_completeness",
    "_to_json_native", "_resolve_comparison_period",
    "COMPARISON_MODE_HARD_FILTER", "COMPARISON_MODE_SCORE_RANKING",
    "BREAKOUT_QUALITY_RANKING_POLICY_SCORE",
    "BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED",
    "BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET",
    "OPTIONAL_ENTRY_FILTER_POLICY_CURRENT", "OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF",
    "SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES", "OPTIONAL_ENTRY_FILTER_FIELDS",
    "PARAM_POLICY_AUTO", "PARAM_POLICY_BASE_FINALIST_BEST", "PARAM_POLICY_BASE_FINALISTS_AGREE",
    "_resolve_params_path", "_resolve_param_selector", "_validate_requested_param_policy",
]
