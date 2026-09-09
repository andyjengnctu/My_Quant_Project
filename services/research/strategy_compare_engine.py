"""Controlled no-filter vs active breakout-quality portfolio comparison."""

from __future__ import annotations

from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

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
    load_strategy_param_evaluation_view,
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
from filters.breakout_quality.strategy_score_projection import (
    primary_score_column_for_source,
)
from filters.breakout_quality.strategy_compare_reporting import (
    _assert_shared_benchmark,
    _build_yearly_comparison,
    _capacity_summary,
    _delta,
    _format_metric,
    _load_existing_comparison_payload,
    _normalize_yearly_completeness,
    _refresh_yearly_summary,
    _remove_legacy_html_outputs,
    _scenario_summary,
    _to_json_native,
    _yearly_frame,
    materialize_strategy_pair_readable_report,
    render_strategy_pair_markdown,
    render_strategy_pair_simple_report,
)
from services.research.strategy_compare_runtime_contract import resolve_strategy_compare_runtime_contract
from services.research.strategy_compare_replay import (
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
    _continuous_forward_target_lookup,
    _load_isolated_selection_pit_contract,
    _selection_target_lookup,
    _strategy_selection_diagnostics,
    paired_trade_r_conversion_diagnostic,
)

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
)
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
    build_breakout_quality_capital_aware_ranking_contract,
    build_breakout_quality_score_ranking_order,
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
            "resource-aware-continuous-max-dl-matched-feasible-ascent=C68專用matched local control；在相同K/R0下先以score-priority提出membership、"
            "再以canonical execution驗證後做相同best-improvement single-swap ascent，不改歷史feasible-ascent owner；"
            "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard=同一selector再禁止過舊score驅動membership change；"
            "resource-aware-continuous-expected-pnl-feasible-ascent=frozen rank percentile校準absolute Expected-R後乘canonical planned risk；"
            "resource-aware-continuous-excess-alpha-feasible-ascent=Selection-only frozen rank percentile校準relative Expected Excess-R後乘canonical planned risk；"
            "resource-aware-continuous-excess-alpha-no-r0-feasible-ascent=同一Excess-Alpha objective與固定K，但移除baseline R0 floor及其minimum repair，只保留canonical cash feasibility；"
            "resource-aware-continuous-excess-alpha-constrained-optimal=同一Excess-Alpha objective與K/R0，直接以exact branch-and-bound在完整候選universe求canonical cash-feasible constrained optimum；"
            "resource-aware-continuous-score-constrained-optimal=保留各DL source原始continuous score objective與K/R0，直接以共用exact solver求canonical cash-feasible constrained optimum；"
            "resource-aware-continuous-score-no-k-no-r0=沿用原始continuous model_score由高到低與canonical tie，只移除baseline K/R0；count僅受physical slots與canonical cash/sizing/orderability；"
            "resource-aware-continuous-score-k-flex-r0-constrained-optimal=保持同一score objective與R0，僅將exact K放寬為baseline K到physical free slots並先最大化可行count。"
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
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    param_evaluation_mode="rolling",
):
    root = Path(project_root).resolve()
    runtime = resolve_strategy_compare_runtime_contract(
        root=root,
        comparison_mode=comparison_mode,
        ranking_policy=ranking_policy,
        ranking_options=ranking_options,
        optional_entry_filter_policy=optional_entry_filter_policy,
        score_source=score_source,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        threshold=threshold,
        hard_filter_source=hard_filter_source,
        continuous_score_path_override=continuous_score_path_override,
        continuous_score_execution_start_override=continuous_score_execution_start_override,
        selection_pit_score_path_override=selection_pit_score_path_override,
        selection_pit_manifest_path_override=selection_pit_manifest_path_override,
        selection_pit_expected_seed_override=selection_pit_expected_seed_override,
        comparison_start_date=comparison_start_date,
        comparison_end_date=comparison_end_date,
    )
    comparison_mode = runtime.comparison_mode
    ranking_policy = runtime.ranking_policy
    ranking_options = runtime.ranking_options
    optional_entry_filter_policy = runtime.optional_entry_filter_policy
    score_source = runtime.score_source
    filter_id = runtime.filter_id
    model_architecture = runtime.model_architecture
    experiment_profile = runtime.experiment_profile
    configured_threshold = runtime.configured_threshold
    labels = runtime.labels
    runtime_contract = runtime.runtime_contract
    pit_contract = runtime.pit_contract
    continuous_contract = runtime.continuous_contract
    continuous_score_override = runtime.continuous_score_override
    ranking_source = runtime.ranking_source
    filter_source = runtime.filter_source
    binary_filter_manifest = runtime.binary_filter_manifest
    binary_filter_manifest_path = runtime.binary_filter_manifest_path
    binary_filter_scores_path = runtime.binary_filter_scores_path
    effective_score_source = runtime.effective_score_source
    manifest_architecture = runtime.manifest_architecture
    manifest_profile = runtime.manifest_profile
    start_date = runtime.start_date
    end_date = runtime.end_date

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
    param_source, param_identity_sha256 = load_strategy_param_evaluation_view(
        resolved_params_path,
        evaluation_mode=param_evaluation_mode,
        start_date=start_date,
        end_date=end_date,
    )
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
    pair_timing_started = time.perf_counter()
    baseline_timing_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback("baseline_start", {"reused": baseline_reuse_dir is not None})
    if baseline_reuse_dir is not None:
        baseline_payload, baseline_summary_override, baseline_orderable_reused = (
            _load_reusable_no_filter_baseline(
                Path(baseline_reuse_dir),
                comparison_mode=comparison_mode,
                expected_dataset=dataset,
                expected_params_sha256=param_identity_sha256,
                expected_param_policy=param_policy,
                expected_param_evaluation_mode=str(param_evaluation_mode),
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
    baseline_elapsed_sec = time.perf_counter() - baseline_timing_started
    if progress_callback is not None:
        progress_callback(
            "baseline_done",
            {
                "reused": baseline_reuse_dir is not None,
                "elapsed_sec": float(baseline_elapsed_sec),
            },
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
    active_timing_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback("active_start", {})
    quality_payload = _run_scenario(
        name=labels["active_name"], data_dir=data_dir,
        param_source_kind=param_source_kind, params=quality_params,
        start_date=start_date, end_date=end_date, max_positions=max_positions,
        enable_rotation=enable_rotation, quiet=quiet,
        replay_counts=quality_replay_counts, replay_execution_rows=quality_execution_rows,
        replay_selector_trace_rows=quality_selector_trace_rows,
        ranking_source=ranking_source, filter_source=filter_source,
    )
    active_elapsed_sec = time.perf_counter() - active_timing_started
    if progress_callback is not None:
        progress_callback("active_done", {"elapsed_sec": float(active_elapsed_sec)})
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
            score_source in {
                SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
            }
            and bool(capture_selection_target_diagnostics)
        ):
            runtime_primary_score_column = str(
                ranking_options.get("primary_score_column")
                or primary_score_column_for_source(score_source)
            ).strip()
            if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
                lookup = _selection_target_lookup(
                    root=root, filter_id=filter_id, architecture=manifest_architecture,
                    profile=manifest_profile,
                    score_path_override=str(pit_contract.score_path),
                    manifest_path_override=str(pit_contract.manifest_path),
                    score_column=runtime_primary_score_column,
                )
            else:
                forward_score_path = (
                    str(continuous_score_override["score_path"])
                    if continuous_score_override is not None
                    else str(continuous_contract.score_path)
                )
                lookup = _continuous_forward_target_lookup(
                    root=root, filter_id=filter_id, architecture=manifest_architecture,
                    profile=manifest_profile,
                    score_path_override=forward_score_path,
                    score_column=runtime_primary_score_column,
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
                "selection_r_conversion": paired_trade_r_conversion_diagnostic(
                    baseline_payload["trade_history"],
                    quality_payload["trade_history"],
                    baseline_selected_joined,
                    quality_selected_joined,
                ),
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
        "execution_timing": {
            "baseline_action": "REUSE" if baseline_reuse_dir is not None else "RUN",
            "baseline_elapsed_sec": round(float(baseline_elapsed_sec), 6),
            "active_elapsed_sec": round(float(active_elapsed_sec), 6),
            "pair_elapsed_to_metadata_sec": round(float(time.perf_counter() - pair_timing_started), 6),
        },
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
        "params_file_sha256": param_identity_sha256,
        "params_source_file_sha256": _sha256_file(resolved_params_path),
        "param_evaluation_mode": str(param_evaluation_mode),
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
            build_breakout_quality_score_ranking_order(
                ranking_policy,
                include_runtime_member_vote_count=(
                    param_policy_contract["selector"] != "base_finalist_best"
                ),
            )
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            else None
        ),
        "capital_aware_ranking_contract": (
            build_breakout_quality_capital_aware_ranking_contract(
                ranking_policy,
                ranking_options=ranking_options,
            )
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
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
