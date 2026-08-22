"""Strategy Compare portfolio replay and reusable baseline orchestration."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
)
from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.dataset_profiles import get_dataset_dir
from core.portfolio_stats import calc_plain_romd
from core.rolling_oos_params import get_active_param_date_range
from filters.breakout_quality.artifacts import load_runtime_artifact_contract
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_CANONICAL_RUNTIME
from filters.breakout_quality.runtime import (
    breakout_quality_filter_source_execution_context,
    breakout_quality_ranking_source_context,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    STRATEGY_COMPARE_SCHEMA_VERSION,
    comparison_labels as _comparison_labels,
)
from filters.breakout_quality.strategy_compare_diagnostics import (
    _flatten_candidate_replay_rows,
    _flatten_selected_buy_rows,
)
from filters.breakout_quality.strategy_compare_reporting import (
    _format_metric,
    _load_existing_comparison_payload,
    _refresh_yearly_summary,
    _remove_legacy_html_outputs,
    _scenario_summary,
    _to_json_native,
    _yearly_frame,
)
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    PARAM_POLICY_AUTO,
    SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES,
    _build_controlled_param_source_pair,
    apply_strategy_param_evaluation_view,
    strategy_param_source_identity_sha256,
    _load_param_source,
    _resolve_params_path,
    _sha256_file,
    _validate_requested_param_policy,
)
from services.portfolio_replay import (
    PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    load_portfolio_market_context,
    run_portfolio_simulation_prepared,
    run_portfolio_simulation_with_param_ensemble,
    run_portfolio_simulation_with_param_schedule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = STRATEGY_COMPARE_SCHEMA_VERSION
_RESULT_FIELDS = (
    "equity_curve", "trade_history", "total_return_pct", "max_drawdown_pct",
    "trade_count", "win_rate_pct", "expected_value_r", "payoff_ratio",
    "final_equity", "avg_exposure_pct", "max_exposure_pct", "benchmark_return_pct",
    "benchmark_max_drawdown_pct", "missed_buy_count", "missed_sell_count",
    "log_r_squared", "monthly_win_rate_pct", "benchmark_log_r_squared",
    "benchmark_monthly_win_rate_pct", "normal_trade_count", "extended_trade_count",
    "annual_trade_count", "reserved_buy_fill_rate_pct", "annual_return_pct",
    "benchmark_annual_return_pct", "profile",
)

def _unpack_result(result) -> dict[str, Any]:
    if len(result) != len(_RESULT_FIELDS):
        raise ValueError(f"portfolio result 欄位數不一致: expected={len(_RESULT_FIELDS)}, actual={len(result)}")
    payload = dict(zip(_RESULT_FIELDS, result))
    payload["return_over_max_drawdown"] = calc_plain_romd(
        payload["total_return_pct"], payload["max_drawdown_pct"]
    )
    return payload

def _run_scenario(
    *, name, data_dir, param_source_kind, params, start_date, end_date,
    max_positions, enable_rotation, quiet, replay_counts=None,
    replay_execution_rows=None, replay_selector_trace_rows=None, ranking_source=None, filter_source=None,
):
    if not quiet:
        print(f"\n[{name}] 建立市場與訊號快取")
    with ExitStack() as stack:
        if ranking_source:
            stack.enter_context(breakout_quality_ranking_source_context(**ranking_source))
        if filter_source:
            stack.enter_context(breakout_quality_filter_source_execution_context(**filter_source))
        return _run_scenario_inside_source_context(
            name=name, data_dir=data_dir, param_source_kind=param_source_kind,
            params=params, start_date=start_date, end_date=end_date,
            max_positions=max_positions, enable_rotation=enable_rotation, quiet=quiet,
            replay_counts=replay_counts, replay_execution_rows=replay_execution_rows,
            replay_selector_trace_rows=replay_selector_trace_rows,
        )

def _run_scenario_inside_source_context(
    *, name, data_dir, param_source_kind, params, start_date, end_date,
    max_positions, enable_rotation, quiet, replay_counts=None,
    replay_execution_rows=None, replay_selector_trace_rows=None,
):
    if param_source_kind == "single_param":
        context = load_portfolio_market_context(str(data_dir), params, verbose=not quiet)
        if not quiet:
            print(f"[{name}] 執行 {start_date} ～ {end_date}")
        result = run_portfolio_simulation_prepared(
            context["all_dfs_fast"], context["all_trade_logs"], context["sorted_dates"], params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=not quiet,
            pit_stats_index=context.get("all_pit_stats_index"),
            replay_counts=replay_counts,
            replay_execution_rows=replay_execution_rows,
            replay_selector_trace_rows=replay_selector_trace_rows,
        )
    elif param_source_kind in {"static_active_param_ensemble", "rolling_active_param_ensemble"}:
        if not quiet:
            print(f"[{name}] 執行 active-param ensemble replay {start_date} ～ {end_date}")
        result = run_portfolio_simulation_with_param_ensemble(
            str(data_dir), params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
            fixed_risk=None, verbose=not quiet, replay_counts=replay_counts,
            replay_execution_rows=replay_execution_rows,
            replay_selector_trace_rows=replay_selector_trace_rows,
        )
    elif param_source_kind == "rolling_oos_param_schedule":
        if not quiet:
            print(f"[{name}] 執行 rolling active-param replay {start_date} ～ {end_date}")
        result = run_portfolio_simulation_with_param_schedule(
            str(data_dir), params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
            fixed_risk=None, verbose=not quiet, replay_counts=replay_counts,
            replay_execution_rows=replay_execution_rows,
            replay_selector_trace_rows=replay_selector_trace_rows,
        )
    else:
        raise ValueError(f"不支援的參數來源類型: {param_source_kind}")
    return _unpack_result(result)

def _resolve_comparison_period(contract) -> tuple[str, str]:
    start = contract.execution_start
    end = contract.available_through
    if end < start:
        raise ValueError(
            "breakout quality 策略執行期間不合法: "
            f"execution_start={start}, available_through={end}"
        )
    return start.isoformat(), end.isoformat()

def _load_reusable_no_filter_baseline(
    source_dir: Path,
    *,
    comparison_mode: str,
    expected_dataset: str,
    expected_params_sha256: str,
    expected_param_policy: str,
    expected_param_evaluation_mode: str,
    expected_optional_entry_filter_policy: str,
    expected_shared_param_overrides: dict[str, Any],
    expected_max_positions: int,
    expected_enable_rotation: bool,
    expected_start_date: str,
    expected_end_date: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Rehydrate one completed no-filter replay for a new controlled pair.

    This is intentionally limited to score-ranking comparisons.  The baseline
    economic summary is read from the completed pair JSON while date/equity,
    trades, capacity and orderable-candidate artifacts are loaded from the
    canonical CSV outputs.  The new DL arm still executes normally.
    """
    if comparison_mode != COMPARISON_MODE_SCORE_RANKING:
        raise ValueError("shared baseline reuse目前只支援score-ranking比較")
    source_dir = Path(source_dir).resolve()
    payload = _load_existing_comparison_payload(source_dir)
    metadata = dict(payload.get("metadata") or {})
    baseline_summary = dict(payload.get("no_filter") or {})
    if not baseline_summary:
        raise ValueError("可重用baseline缺少no_filter summary")

    expected_contract = {
        "dataset": str(expected_dataset),
        "params_file_sha256": str(expected_params_sha256),
        "requested_param_policy": str(expected_param_policy),
        "param_evaluation_mode": str(expected_param_evaluation_mode),
        "optional_entry_filter_policy": str(expected_optional_entry_filter_policy),
        "shared_param_overrides": dict(expected_shared_param_overrides or {}),
        "max_positions": int(expected_max_positions),
        "enable_rotation": bool(expected_enable_rotation),
        "comparison_period": {
            "start": str(expected_start_date),
            "end": str(expected_end_date),
        },
    }
    actual_contract = {
        "dataset": str(metadata.get("dataset") or ""),
        "params_file_sha256": str(metadata.get("params_file_sha256") or ""),
        "requested_param_policy": str(metadata.get("requested_param_policy") or ""),
        "param_evaluation_mode": str(metadata.get("param_evaluation_mode") or ""),
        "optional_entry_filter_policy": str(
            metadata.get("optional_entry_filter_policy") or ""
        ),
        "shared_param_overrides": dict(metadata.get("shared_param_overrides") or {}),
        "max_positions": int(metadata.get("max_positions") or 0),
        "enable_rotation": bool(metadata.get("enable_rotation")),
        "comparison_period": dict(metadata.get("comparison_period") or {}),
    }
    if actual_contract != expected_contract:
        raise ValueError(
            "可重用baseline與目前replay contract不一致: "
            f"expected={expected_contract}, actual={actual_contract}"
        )

    required_paths = {
        "equity": source_dir / "no_filter_equity.csv",
        "trades": source_dir / "no_filter_trades.csv",
        "capacity": source_dir / "no_filter_daily_capacity.csv",
        "orderable": source_dir / "no_filter_orderable_candidates.csv",
        "yearly": source_dir / "yearly_returns_comparison.csv",
    }
    missing = [path.name for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "可重用baseline缺少正式工件: " + ", ".join(sorted(missing))
        )

    equity = pd.read_csv(required_paths["equity"], encoding="utf-8-sig")
    trades = pd.read_csv(required_paths["trades"], encoding="utf-8-sig")
    capacity = pd.read_csv(required_paths["capacity"], encoding="utf-8-sig")
    orderable = pd.read_csv(required_paths["orderable"], encoding="utf-8-sig")
    yearly = pd.read_csv(required_paths["yearly"], encoding="utf-8-sig")

    yearly_rows: list[dict[str, Any]] = []
    if not yearly.empty and "year" in yearly.columns:
        for row in yearly.to_dict("records"):
            yearly_rows.append({
                "year": row.get("year"),
                "year_return_pct": row.get("no_filter_return_pct"),
                "is_full_year": bool(row.get("is_full_year", False)),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
            })

    baseline_payload: dict[str, Any] = {
        **baseline_summary,
        "equity_curve": equity,
        "trade_history": trades,
        "profile": {
            "yearly_return_rows": yearly_rows,
            "portfolio_capacity_rows": capacity.to_dict("records"),
        },
    }
    return baseline_payload, baseline_summary, orderable

def _standalone_baseline_report(payload: dict[str, Any], *, markdown: bool) -> str:
    metadata = dict(payload.get("metadata") or {})
    baseline = dict(payload.get("no_filter") or {})
    yearly = pd.DataFrame(list(payload.get("yearly") or []))
    if not metadata or not baseline:
        raise ValueError("standalone baseline payload缺少metadata／no_filter")
    period = dict(metadata.get("comparison_period") or {})
    params_path = project_relative_display_path(
        metadata.get("params_path", "-"), project_root=PROJECT_ROOT
    )
    metric_rows = [
        ("淨總報酬", _format_metric(baseline.get("total_return_pct"), digits=2, unit="%")),
        ("最大回撤", _format_metric(baseline.get("max_drawdown_pct"), digits=2, unit="%")),
        ("報酬／最大回撤", _format_metric(baseline.get("return_over_max_drawdown"), digits=2)),
        ("年化報酬", _format_metric(baseline.get("annual_return_pct"), digits=2, unit="%")),
        ("Log R²", _format_metric(baseline.get("log_r_squared"), digits=4)),
        ("月勝率", _format_metric(baseline.get("monthly_win_rate_pct"), digits=2, unit="%")),
        ("交易數", _format_metric(baseline.get("trade_count"), digits=0)),
        ("EV", _format_metric(baseline.get("expected_value_r"), digits=2, unit=" R")),
        ("平均曝險", _format_metric(baseline.get("avg_exposure_pct"), digits=2, unit="%")),
    ]
    yearly_rows = []
    if not yearly.empty:
        for row in yearly.to_dict("records"):
            yearly_rows.append((
                int(row["year"]),
                _format_metric(row.get("no_filter_return_pct"), digits=2, unit="%"),
                "是" if row.get("is_full_year") else "否",
            ))
    if markdown:
        lines = [
            "# Strategy Compare DL-off 基準回放",
            "",
            f"- 期間：`{period.get('start', '')}` ～ `{period.get('end', '')}`",
            f"- 參數檔：`{params_path}`",
            f"- 參數型態：`{metadata.get('param_source_kind', '-')}`",
            f"- 參數 selector：`{metadata.get('param_selector', '-')}`",
            f"- Optional entry filters：`{metadata.get('optional_entry_filter_policy', '-')}`",
            f"- 歷史 active-param 無前視：`{metadata.get('lookahead_safe_active_param_schedule', '-')}`",
            "",
            "## 主要結果",
            "",
            "| 指標 | Baseline |",
            "|---|---:|",
            *[f"| {label} | {value} |" for label, value in metric_rows],
            "",
            "## 年度報酬",
            "",
            "| 年度 | Baseline | 完整年度 |",
            "|---:|---:|:---:|",
            *[f"| {year} | {value} | {full} |" for year, value, full in yearly_rows],
            "",
            "此工件只建立同參數／規則體系的DL-off baseline，不使用任何DL score。",
        ]
        return "\n".join(lines).rstrip() + "\n"
    return "\n\n".join((
        render_title("Strategy Compare DL-off 基準回放"),
        render_key_values((
            ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
            ("參數檔", params_path),
            ("參數型態", metadata.get("param_source_kind", "-")),
            ("參數 selector", metadata.get("param_selector", "-")),
            ("Optional entry filters", metadata.get("optional_entry_filter_policy", "-")),
            ("歷史 active-param 無前視", metadata.get("lookahead_safe_active_param_schedule", "-")),
        )),
        render_section("主要結果", number=1),
        render_table(("指標", "Baseline"), metric_rows, alignments=("left", "right")),
        render_section("年度報酬", number=2),
        render_table(("年度", "Baseline", "完整年度"), yearly_rows, alignments=("right", "right", "center"))
        if yearly_rows else "無年度資料。",
    ))

def run_standalone_baseline(
    *,
    project_root=PROJECT_ROOT,
    dataset="full",
    params_path=None,
    param_policy=PARAM_POLICY_AUTO,
    max_positions=DEFAULT_PORTFOLIO_MAX_POSITIONS,
    enable_rotation=False,
    optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    output_dir_override=None,
    comparison_start_date=None,
    comparison_end_date=None,
    quiet=False,
    shared_param_overrides=None,
    baseline_reuse_dir=None,
    param_evaluation_mode="rolling",
):
    """Run or rehydrate one canonical DL-off strategy baseline.

    Standalone baselines let Strategy Compare keep a benchmark arm even when no
    DL-on arm is enabled for the same parameter/rule group.  The replay uses the
    exact same no-filter parameter construction and portfolio engine as a normal
    controlled pair, and emits the same baseline artifacts so future pairs may
    reuse it without rerunning the portfolio simulation.
    """

    root = Path(project_root).resolve()
    optional_entry_filter_policy = str(optional_entry_filter_policy).strip()
    if optional_entry_filter_policy not in SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES:
        raise ValueError(
            f"不支援的 optional entry filter policy: {optional_entry_filter_policy!r}"
        )
    if (comparison_start_date is None) != (comparison_end_date is None):
        raise ValueError("comparison_start_date與comparison_end_date必須同時提供")
    if comparison_start_date is None:
        raise ValueError("standalone baseline必須由Strategy Compare提供共同comparison period")
    start_date = pd.Timestamp(str(comparison_start_date)).normalize().strftime("%Y-%m-%d")
    end_date = pd.Timestamp(str(comparison_end_date)).normalize().strftime("%Y-%m-%d")
    if pd.Timestamp(end_date) < pd.Timestamp(start_date):
        raise ValueError("standalone baseline比較期間不合法")

    param_policy = str(param_policy)
    resolved_params_path = _resolve_params_path(
        root=root,
        params_path=params_path,
        param_policy=param_policy,
        allow_static_diagnostic=False,
        score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
    )
    if not resolved_params_path.is_file():
        raise FileNotFoundError(f"找不到策略比較參數檔: {resolved_params_path}")
    param_source = _load_param_source(resolved_params_path)
    param_source = apply_strategy_param_evaluation_view(
        param_source,
        evaluation_mode=param_evaluation_mode,
        start_date=start_date,
        end_date=end_date,
    )
    param_identity_sha256 = strategy_param_source_identity_sha256(
        param_source, source_path=resolved_params_path
    )
    param_policy_contract = _validate_requested_param_policy(param_source, param_policy)
    (
        param_source_kind,
        no_filter_params,
        _quality_params,
        no_filter_param_payload,
        _quality_param_payload,
        ensemble_policy,
    ) = _build_controlled_param_source_pair(
        param_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        max_position_cap_pct=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=optional_entry_filter_policy,
        shared_param_overrides=shared_param_overrides,
    )
    is_rolling_source = param_source_kind in {
        "rolling_oos_param_schedule", "rolling_active_param_ensemble"
    }
    if not is_rolling_source:
        raise ValueError("standalone baseline正式比較只接受rolling active-param工件")
    if param_source_kind == "rolling_oos_param_schedule":
        param_start, param_end = get_active_param_date_range(param_source["payload"])
    else:
        param_start, param_end = get_active_param_ensemble_date_range(param_source["payload"])
    if (
        pd.Timestamp(param_start) > pd.Timestamp(start_date)
        or pd.Timestamp(param_end) < pd.Timestamp(end_date)
    ):
        raise ValueError(
            "rolling active-param期間未完整覆蓋standalone baseline："
            f"params={param_start}~{param_end}, comparison={start_date}~{end_date}"
        )

    data_dir = Path(get_dataset_dir(str(root), dataset)).resolve()
    replay_counts: dict[str, Any] | None = {}
    baseline_summary_override = None
    baseline_orderable_reused = None
    if baseline_reuse_dir is not None:
        baseline_payload, baseline_summary_override, baseline_orderable_reused = (
            _load_reusable_no_filter_baseline(
                Path(baseline_reuse_dir),
                comparison_mode=COMPARISON_MODE_SCORE_RANKING,
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
        replay_counts = None
        if not quiet:
            print(
                "[baseline] 重用既有正式DL-off replay："
                + project_relative_display_path(
                    Path(baseline_reuse_dir).resolve(), project_root=root
                )
            )
    else:
        baseline_payload = _run_scenario(
            name="baseline",
            data_dir=data_dir,
            param_source_kind=param_source_kind,
            params=no_filter_params,
            start_date=start_date,
            end_date=end_date,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
            quiet=quiet,
            replay_counts=replay_counts,
        )

    baseline = (
        dict(baseline_summary_override)
        if baseline_summary_override is not None
        else _scenario_summary(baseline_payload)
    )
    yearly = _yearly_frame(baseline_payload["profile"], "no_filter")
    _refresh_yearly_summary(baseline, yearly, return_column="no_filter_return_pct")

    if output_dir_override is None:
        raise ValueError("standalone baseline必須指定output_dir_override")
    output_dir = Path(output_dir_override)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_orderable = (
        pd.DataFrame(baseline_orderable_reused).copy()
        if baseline_orderable_reused is not None
        else _flatten_candidate_replay_rows(replay_counts or {}, "orderable_rows")
    )
    baseline_selected = _flatten_selected_buy_rows(baseline_payload["trade_history"])
    baseline_orderable.to_csv(
        output_dir / "no_filter_orderable_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_selected.to_csv(
        output_dir / "no_filter_selected_buys.csv", index=False, encoding="utf-8-sig"
    )
    baseline_payload["equity_curve"].to_csv(
        output_dir / "no_filter_equity.csv", index=False, encoding="utf-8-sig"
    )
    baseline_payload["trade_history"].to_csv(
        output_dir / "no_filter_trades.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        baseline_payload["profile"].get("portfolio_capacity_rows") or []
    ).to_csv(
        output_dir / "no_filter_daily_capacity.csv", index=False, encoding="utf-8-sig"
    )
    yearly.to_csv(
        output_dir / "yearly_returns_comparison.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
        "comparison_design": "standalone_dl_off_active_param_replay",
        "score_source": "dl_off_baseline_only",
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
        "optional_entry_filter_policy": optional_entry_filter_policy,
        "shared_param_overrides": dict(shared_param_overrides or {}),
        "lookahead_safe_active_param_schedule": True,
        "active_param_period": {"start": param_start, "end": param_end},
        "active_param_ensemble_policy": ensemble_policy,
        "no_filter_params": no_filter_param_payload,
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "baseline_reused": baseline_reuse_dir is not None,
        "baseline_reuse_source": (
            None
            if baseline_reuse_dir is None
            else project_relative_display_path(
                Path(baseline_reuse_dir).resolve(), project_root=root
            )
        ),
        "output_scope": "caller_override",
        "output_dir": project_relative_display_path(output_dir, project_root=root),
        "comparison_period": {"start": start_date, "end": end_date},
        "benchmark_ticker": PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    }
    json_payload = _to_json_native({
        "metadata": metadata,
        "no_filter": baseline,
        "yearly": yearly.to_dict("records"),
    })
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "strategy_comparison.md").write_text(
        _standalone_baseline_report(json_payload, markdown=True), encoding="utf-8"
    )
    _remove_legacy_html_outputs(output_dir)
    if not quiet:
        print("\n" + _standalone_baseline_report(json_payload, markdown=False))
        print_artifact_paths(
            (("策略基準簡易報表", output_dir / "strategy_comparison.md"),),
            project_root=root,
        )
    return json_payload

# Stable public aliases for read-only consumers.
run_scenario = _run_scenario
