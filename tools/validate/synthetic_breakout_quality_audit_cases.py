from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from .checks import bind_checks

from .synthetic_breakout_quality_support import add_check


def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        get_reusable_audit_definitions,
        validate_audit_config,
    )
    from services.audit.catalog import (
        get_domain_cli_commands,
        validate_audit_catalog,
    )
    from services.audit.runner import collect_audit_status

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    validate_audit_config()
    definitions = get_audit_definitions("breakout_quality")
    enabled = get_enabled_audit_definitions("breakout_quality")
    validate_audit_catalog(definitions)

    check_true(
        "formal_audit_config_is_config_driven_and_uses_stable_output_root",
        bool(
                    get_audit_module_ids()[0] == "breakout_quality"
                    and len(definitions) >= len(enabled)
                    and AUDIT_OUTPUT_ROOT == "outputs/audit"
                ),
    )

    status = collect_audit_status(module_id="breakout_quality")
    expected_overall = (
        "DISABLED"
        if not enabled
        else "READY"
        if all(
            row.get("status") == "READY"
            for row in (status.get("rows") or [])
            if bool(row.get("enabled"))
        )
        else "BLOCKED"
    )
    check(
        "audit_runner_overall_status_follows_current_enabled_definitions_without_fixed_audit_id",
        expected_overall,
        status.get("overall_status"),
    )

    commands = get_domain_cli_commands("breakout_quality")
    check_true(
        "research_audit_utilities_remain_catalogued_alongside_formal_gate",
        bool("regime-audit" in commands and "audit-point-in-time-scores" in commands),
    )
    from services.audit.catalog import get_audit_entry
    joint_defs = [item for item in definitions if item.audit_id == "AUD-mr13r-joint-capital-drawdown"]
    joint_entry = get_audit_entry("mr13r_joint_capital_drawdown")
    check_true(
        "mr13r_joint_capital_drawdown_audit_is_formal_read_only_and_strategy_pair_scoped",
        bool(
            len(joint_defs) == 1
            and joint_entry.formal
            and joint_entry.read_only
            and joint_entry.method_id == "portfolio_drawdown_attribution"
            and not joint_defs[0].enabled
            and tuple(joint_defs[0].source.get("strategy_arm_ids", ())) == ("C71", "C72", "C73", "C74")
            and joint_defs[0].source.get("strategy_result_fingerprints", {}).get("extending_window_oos") == "b12582a9de23"
            and joint_defs[0].source.get("strategy_result_fingerprints", {}).get("extending_window_rolling") == "3bca1e932f2e"
        ),
    )

    import apps.research as research_app

    import services.audit.runner as audit_runner

    audit_menu_console = io.StringIO()
    with patch.object(audit_runner, "get_audit_definitions", return_value=()), patch(
        "builtins.input", return_value="0"
    ), redirect_stdout(audit_menu_console):
        audit_menu_rc = research_app._audit_menu()
    audit_menu_text = audit_menu_console.getvalue()
    check_true(
        "research_audit_menu_separates_reusable_one_time_and_history_without_settings_submenu",
        audit_menu_rc == 0
        and "可重複使用的原因分析" in audit_menu_text
        and "一次性專題 Audit" in audit_menu_text
        and "最近結果／歷史 Evidence" in audit_menu_text
        and "查看全部 Audit 設定與工件狀態" not in audit_menu_text,
    )

    reusable_console = io.StringIO()
    with patch.object(audit_runner, "get_audit_definitions", return_value=()), patch(
        "builtins.input", side_effect=["1", "0", "0"]
    ), redirect_stdout(reusable_console):
        research_app._audit_menu()
    reusable_text = reusable_console.getvalue()
    check_true(
        "reusable_audit_menu_keeps_three_purpose_driven_reports_visible",
        "Opportunity／Selection Attribution  [已設定]" in reusable_text
        and "Trade Outcome／Path Attribution  [已設定]" in reusable_text
        and "Portfolio／Drawdown Attribution  [已設定]" in reusable_text
        and "Regime／Stability Attribution" not in reusable_text,
    )
    reusable_defs = get_reusable_audit_definitions("breakout_quality")
    check_true(
        "reusable_audit_reports_are_config_driven_and_not_scientific_aud_ids",
        len(reusable_defs) == 3
        and {item.report_id for item in reusable_defs}
        == {"opportunity_selection", "trade_outcome_path", "portfolio_drawdown"}
        and all(not item.report_id.startswith("AUD-") for item in reusable_defs),
    )

    from filters.breakout_quality.mfe_safety_geometry import truth_geometry_5x5
    truth_fixture = pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(10)],
            "date": ["2025-01-02"] * 5 + ["2025-01-03"] * 5,
            "safety_percentile": [0.02, 0.22, 0.42, 0.62, 0.82] * 2,
            "mfe_percentile": [0.82, 0.62, 0.42, 0.22, 0.02, 0.02, 0.22, 0.42, 0.62, 0.82],
        }
    )
    subset_keys = truth_fixture.iloc[[0, 4, 5, 9]][["ticker", "date"]].copy()
    subset_geometry = truth_geometry_5x5(truth_fixture, subset_keys)
    subset_cells = subset_geometry.get("cells") or []
    check_true(
        "canonical_truth_5x5_filters_membership_without_subset_rerank",
        subset_geometry.get("percentile_policy")
        == "canonical_daily_universal_percentiles_no_subset_rerank"
        and int(subset_geometry.get("truth_covered_rows", 0) or 0) == 4
        and int(subset_cells[0][4].get("n", 0) or 0) == 1
        and int(subset_cells[4][0].get("n", 0) or 0) == 1
        and int(subset_cells[0][0].get("n", 0) or 0) == 1
        and int(subset_cells[4][4].get("n", 0) or 0) == 1,
    )

    from services.audit.reusable_report import audit_section, render_evidence_rows
    markdown_section = audit_section("Key Evidence", 7, target="markdown")
    markdown_evidence = render_evidence_rows(
        (("Actual joint support", "PRESENT", "Daily/Breakout truth support exists"),),
        target="markdown",
    )
    check_true(
        "reusable_audit_reports_use_shared_colored_sections_and_semantic_evidence_style",
        "#42A5F5" in markdown_section
        and "7. Key Evidence" in markdown_section
        and "#188038" in markdown_evidence
        and "PRESENT" in markdown_evidence,
    )

    from services.audit.selection_membership import build_planned_membership as build_reusable_planned
    no_k_orderable = pd.DataFrame({
        "ticker": ["A", "B"],
        "trade_date": ["2025-01-02", "2025-01-02"],
        "signal_date": ["2025-01-01", "2025-01-01"],
        "breakout_quality_score_date": ["2025-01-01", "2025-01-01"],
        "breakout_quality_score": [0.8, 0.7],
        "breakout_quality_daily_score_percentile": [0.9, 0.6],
    })
    no_k_execution = pd.DataFrame({
        "execution_order": [10, 11],
        "ticker": ["A", "B"],
        "trade_date": ["2025-01-02", "2025-01-02"],
        "signal_date": ["2025-01-01", "2025-01-01"],
        "chosen_qty": [1, 1],
        "entry_filled": [True, False],
    })
    no_k_trace = pd.DataFrame({
        "stage": ["raw_top_n"],
        "stage_rank": [1],
        "ticker": ["A"],
        "trade_date": ["2025-01-02"],
        "signal_date": ["2025-01-01"],
        "breakout_quality_score_date": ["2025-01-01"],
        "breakout_quality_score": [0.8],
        "breakout_quality_daily_score_percentile": [0.9],
    })
    reusable_planned = build_reusable_planned(
        orderable=no_k_orderable,
        execution=no_k_execution,
        selector_trace=no_k_trace,
        final_stage="feasible_ascent_final",
    )
    check_true(
        "reusable_planned_membership_uses_canonical_execution_and_does_not_require_max_dl_final_stage",
        len(reusable_planned) == 2
        and list(reusable_planned["ticker"]) == ["A", "B"]
        and list(reusable_planned["stage_rank"]) == [1, 2]
        and list(reusable_planned["score_percentile"]) == [0.9, 0.6]
        and list(reusable_planned["entry_filled_bool"]) == [True, False]
        and set(reusable_planned["stage"]) == {"planned_execution"},
    )

    opportunity_source = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "audit"
        / "opportunity_selection.py"
    ).read_text(encoding="utf-8")
    check_true(
        "reusable_opportunity_preflight_uses_selector_agnostic_planned_sidecars",
        "load_strategy_arm_planned_sidecars(project_root, source=source, arm_id=str(arm_id))"
        in opportunity_source
        and "load_strategy_arm_pipeline_sidecars" not in opportunity_source,
    )

    from config.strategy_compare import get_strategy_comparison_settings
    from filters.breakout_quality.strategy_compare_contracts import (
        COMPARISON_MODE_SCORE_RANKING,
    )
    from services.audit.strategy_compare_source import (
        StrategyCompareAuditSource,
        load_strategy_arm_planned_sidecars,
        load_strategy_arm_pipeline_sidecars,
        load_strategy_arm_replay_sidecars,
        resolve_strategy_result_dir_for_fingerprint,
    )
    with TemporaryDirectory() as temp_dir_text:
        project_root = Path(temp_dir_text)
        run_dir = project_root / "outputs" / "strategy_compare" / "runs" / "synthetic"
        run_dir.mkdir(parents=True)
        baseline_rows = pd.DataFrame(
            {
                "ticker": ["2330"],
                "signal_date": ["2025-01-02"],
                "breakout_quality_score_date": ["2025-01-01"],
            }
        )
        pair_execution = {}
        for active_arm_id in ("C59", "C65"):
            pair_dir = run_dir / "pairs" / active_arm_id.lower()
            pair_dir.mkdir(parents=True)
            (pair_dir / "strategy_comparison.json").write_text(
                json.dumps({"metadata": {"comparison_mode": COMPARISON_MODE_SCORE_RANKING}}),
                encoding="utf-8",
            )
            baseline_rows.to_csv(
                pair_dir / "no_filter_orderable_candidates.csv",
                index=False,
                encoding="utf-8-sig",
            )
            baseline_rows.to_csv(
                pair_dir / "no_filter_selected_buys.csv",
                index=False,
                encoding="utf-8-sig",
            )
            baseline_rows.to_csv(
                pair_dir / "score_ranking_orderable_candidates.csv",
                index=False,
                encoding="utf-8-sig",
            )
            baseline_rows.to_csv(
                pair_dir / "score_ranking_selected_buys.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame({
                "Date": ["2025-01-02"],
                "Pre_Market_Free_Slots": [2],
                "Orderable_Candidates": [3],
                "Resource_Aware_Pre_Market_Order_Limit": [1],
                "Resource_Aware_Max_DL_Eligible": [True],
                "Resource_Aware_Direct_Score_Order_Feasible": [False],
                "Resource_Aware_Baseline_Reserved_Milli": [1000],
                "Resource_Aware_Reserved_Milli": [1000],
                "Resource_Aware_Selected": [1],
            }).to_csv(
                pair_dir / "score_ranking_daily_capacity.csv",
                index=False, encoding="utf-8-sig",
            )
            pd.DataFrame({
                "stage": ["raw_top_n"],
                "stage_rank": [1],
                "ticker": ["2330"],
                "trade_date": ["2025-01-02"],
                "signal_date": ["2025-01-02"],
                "breakout_quality_score_date": ["2025-01-01"],
                "pre_market_order_limit": [1],
            }).to_csv(
                pair_dir / "score_ranking_selector_trace.csv",
                index=False, encoding="utf-8-sig",
            )
            pd.DataFrame({
                "ticker": ["2330"],
                "trade_date": ["2025-01-02"],
                "signal_date": ["2025-01-02"],
                "chosen_qty": [1],
                "entry_filled": [True],
            }).to_csv(
                pair_dir / "score_ranking_execution.csv",
                index=False, encoding="utf-8-sig",
            )
            pair_execution[active_arm_id] = {
                "current_pair_dir": str(pair_dir.relative_to(project_root))
            }

        settings = get_strategy_comparison_settings("extending_window_oos")
        audit_source = StrategyCompareAuditSource(
            profile_id="extending_window_oos",
            settings=settings,
            result_dir=run_dir,
            run_dir=run_dir,
            result={"pair_execution": pair_execution},
            manifest=None,
            config_fingerprint="synthetic",
        )
        baseline_evidence = load_strategy_arm_replay_sidecars(
            project_root,
            source=audit_source,
            arm_id="C58",
        )
        active_evidence = load_strategy_arm_replay_sidecars(
            project_root,
            source=audit_source,
            arm_id="C65",
        )
        pipeline_evidence = load_strategy_arm_pipeline_sidecars(
            project_root,
            source=audit_source,
            arm_id="C65",
        )
        planned_evidence = load_strategy_arm_planned_sidecars(
            project_root,
            source=audit_source,
            arm_id="C65",
        )
        check_true(
            "formal_audit_resolves_shared_dl_off_baseline_via_canonical_execution_group_without_replay",
            bool(
                baseline_evidence["prefix"] == "no_filter"
                and baseline_evidence["paired_baseline_resolved"] is True
                and baseline_evidence["pair_execution_source_arm_id"] == "C59"
                and len(baseline_evidence["orderable"]) == 1
                and len(baseline_evidence["selected"]) == 1
                and active_evidence["prefix"] == "score_ranking"
                and active_evidence["paired_baseline_resolved"] is False
                and len(pipeline_evidence["daily_capacity"]) == 1
                and len(pipeline_evidence["selector_trace"]) == 1
                and len(pipeline_evidence["execution"]) == 1
                and "selector_trace" not in planned_evidence
                and "daily_capacity" not in planned_evidence
                and len(planned_evidence["execution"]) == 1
            ),
        )

    with TemporaryDirectory() as retained_temp_text:
        retained_root = Path(retained_temp_text)
        output_root = retained_root / "outputs" / "strategy_compare" / "extending_window" / "oos_2021_forward"
        old_run = output_root / "runs" / "20260825_old"
        new_run = output_root / "runs" / "20260825_new"
        old_run.mkdir(parents=True)
        new_run.mkdir(parents=True)
        (old_run / "strategy_comparison.json").write_text(
            json.dumps({"config_fingerprint": "aaaaaaaaaaaa"}), encoding="utf-8"
        )
        (new_run / "strategy_comparison.json").write_text(
            json.dumps({"config_fingerprint": "bbbbbbbbbbbb"}), encoding="utf-8"
        )
        pinned_run = resolve_strategy_result_dir_for_fingerprint(
            retained_root,
            "outputs/strategy_compare/extending_window/oos_2021_forward",
            config_fingerprint="aaaaaaaaaaaa",
        )
        check_true(
            "retained_audit_resolves_explicit_historical_strategy_fingerprint_instead_of_newer_current_run",
            pinned_run == old_run.resolve(),
        )



    from services.audit.c69_marginal_positions import (
        build_ordinal_cohorts,
        build_planned_membership,
        strict_matched_trade_dates,
    )
    from services.audit.selection_resource_constraints import summarize_resource_contract
    from filters.breakout_quality.strategy_compare_diagnostics import (
        render_mfe_safety_geometry_table,
    )

    retained_capacity = pd.DataFrame({
        "Date": ["2025-01-02", "2025-01-03"],
        "Pre_Market_Positions": [7, 8],
        "Pre_Market_Free_Slots": [3, 2],
        "Orderable_Candidates": [4, 2],
        "Resource_Aware_Max_DL_Eligible": [True, True],
        "Resource_Aware_Pre_Market_Order_Limit": [1, 2],
        "Resource_Aware_Baseline_Reserved_Milli": [100000, 150000],
        "Resource_Aware_Selected": [1, 2],
        "Resource_Aware_Reserved_Milli": [100000, 160000],
        "Resource_Aware_Direct_Score_Order_Feasible": [False, True],
    })
    retained_summary = summarize_resource_contract(retained_capacity)
    check_true(
        "retained_k_r0_audit_resource_contract_survives_shared_truth_refactor",
        bool(
            retained_summary["eligible_days"] == 2
            and retained_summary["direct_infeasible_days"] == 1
            and retained_summary["k_headroom_days"] == 1
            and retained_summary["median_k"] == 1.5
            and retained_summary["r0_exact_binding_days"] == 1
        ),
    )

    # C68/C69 marginal-position attribution primitives: same candidate state/R0,
    # fixed-K control versus K-Flex treatment with K-relative ordinal cohorts.
    trace_columns = {
        "stage": ["feasible_ascent_final"],
        "stage_rank": [1],
        "ticker": ["A"],
        "trade_date": ["2025-01-02"],
        "signal_date": ["2025-01-01"],
        "breakout_quality_score_date": ["2025-01-01"],
        "breakout_quality_score": [0.80],
        "breakout_quality_daily_score_percentile": [0.80],
    }
    c68_trace = pd.DataFrame(trace_columns)
    c69_trace = pd.DataFrame({
        "stage": ["feasible_ascent_final"] * 3,
        "stage_rank": [1, 2, 3],
        "ticker": ["A", "B", "C"],
        "trade_date": ["2025-01-02"] * 3,
        "signal_date": ["2025-01-01"] * 3,
        "breakout_quality_score_date": ["2025-01-01"] * 3,
        "breakout_quality_score": [0.80, 0.95, 0.90],
        "breakout_quality_daily_score_percentile": [0.80, 0.95, 0.90],
    })
    def capacity(selected, extra):
        return pd.DataFrame({
            "Date": ["2025-01-02"],
            "Pre_Market_Positions": [7],
            "Pre_Market_Free_Slots": [3],
            "Orderable_Candidates": [3],
            "Resource_Aware_Max_DL_Eligible": [True],
            "Resource_Aware_Baseline_K": [1],
            "Resource_Aware_Physical_Free_Slots": [3],
            "Resource_Aware_Baseline_Reserved_Milli": [100000],
            "Resource_Aware_Selected": [selected],
            "Resource_Aware_K_Flex_Extra_Positions": [extra],
        })
    def execution(tickers):
        return pd.DataFrame({
            "execution_order": list(range(1, len(tickers) + 1)),
            "ticker": tickers,
            "trade_date": ["2025-01-02"] * len(tickers),
            "signal_date": ["2025-01-01"] * len(tickers),
            "chosen_qty": [1] * len(tickers),
            "entry_filled": [True] * len(tickers),
        })
    c68_planned = build_planned_membership(
        selector_trace=c68_trace, execution=execution(["A"]), daily_capacity=capacity(1, 0)
    )
    c69_planned = build_planned_membership(
        selector_trace=c69_trace, execution=execution(["A", "B", "C"]), daily_capacity=capacity(3, 2)
    )
    orderable = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "trade_date": ["2025-01-02"] * 3,
        "signal_date": ["2025-01-01"] * 3,
        "breakout_quality_score_date": ["2025-01-01"] * 3,
    })
    matched = strict_matched_trade_dates(
        c68_orderable=orderable, c69_orderable=orderable,
        c68_capacity=capacity(1, 0), c69_capacity=capacity(3, 2),
    )
    truth = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "date": ["2025-01-01"] * 3,
        "mfe_percentile": [0.20, 0.90, 0.80],
        "safety_percentile": [0.90, 0.20, 0.80],
        "quadrant": [
            "low_mfe_high_safety_pct",
            "high_mfe_low_safety_pct",
            "high_mfe_high_safety_pct",
        ],
    })
    path = pd.DataFrame({
        "event_key": ["A|2025-01-02|2025-01-01", "B|2025-01-02|2025-01-01", "C|2025-01-02|2025-01-01"],
        "ticker": ["A", "B", "C"],
        "path_target_available_bool": [True, True, True],
        "full_horizon_mfe_r": [0.4, 1.8, 1.5],
        "full_horizon_adverse_to_peak_r": [0.1, 0.5, 0.2],
        "realized_r": [0.3, -0.4, 0.8],
        "actual_initial_stop_out_bool": [False, True, False],
        "exit_date": ["2025-01-06", "2025-01-03", "2025-01-10"],
        "full_horizon_first_upside_1r_date": [None, "2025-01-04", "2025-01-04"],
        "full_horizon_first_upside_1r_bar": [None, 3, 3],
        "full_horizon_first_upside_2r_date": [None, None, None],
        "full_horizon_first_upside_2r_bar": [None, None, None],
        "full_horizon_first_upside_3r_date": [None, None, None],
        "full_horizon_first_upside_3r_bar": [None, None, None],
    })
    ordinal = build_ordinal_cohorts(
        c69_planned, truth=truth, path=path, thresholds=(1.0, 2.0, 3.0)
    )
    check_true(
        "c69_marginal_audit_builds_k_relative_extra_cohorts_on_strict_matched_state",
        bool(
            matched["strict_matched_days"] == 1
            and ordinal["baseline_1_to_k"]["planned_count"] == 1
            and ordinal["k_plus_1"]["planned_count"] == 1
            and ordinal["k_plus_2"]["planned_count"] == 1
            and ordinal["all_extra"]["planned_count"] == 2
            and ordinal["all_extra"]["high_mfe_total_pct"] == 100.0
            and ordinal["all_extra"]["filled_count"] == 2
        ),
    )

    import filters.breakout_quality.strategy_compare_diagnostics as strategy_diag
    with TemporaryDirectory() as geometry_temp_text:
        geometry_root = Path(geometry_temp_text)
        pair_dir = geometry_root / "outputs" / "strategy_compare" / "pair"
        pair_dir.mkdir(parents=True)
        sidecar_orderable = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "signal_date": ["2025-01-01"] * 3,
            "trade_date": ["2025-01-02"] * 3,
            "breakout_quality_score_date": ["2025-01-01"] * 3,
        })
        sidecar_selected_c68 = sidecar_orderable.iloc[[0]].copy()
        sidecar_selected_c69 = sidecar_orderable.copy()
        sidecar_orderable.to_csv(pair_dir / "no_filter_orderable_candidates.csv", index=False, encoding="utf-8-sig")
        sidecar_selected_c68.to_csv(pair_dir / "no_filter_selected_buys.csv", index=False, encoding="utf-8-sig")
        sidecar_orderable.to_csv(pair_dir / "score_ranking_orderable_candidates.csv", index=False, encoding="utf-8-sig")
        sidecar_selected_c69.to_csv(pair_dir / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig")
        c68_arm = SimpleNamespace(arm_id="C68", name="Control")
        c69_arm = SimpleNamespace(arm_id="C69", name="Treatment")
        geometry_settings = SimpleNamespace(
            profile_id="extending_window_oos", enabled_arms=(c68_arm, c69_arm)
        )
        pair_payloads = {
            "pair": {
                "arm_contract": ("min_oos", "all_off", c68_arm, c69_arm),
                "payload": {
                    "metadata": {
                        "output_dir": str(pair_dir.relative_to(geometry_root)),
                        "comparison_period": {"start": "2025-01-01", "end": "2025-01-31"},
                    }
                },
            }
        }
        canonical_truth = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "date": ["2025-01-01"] * 3,
            "mfe_percentile": [0.20, 0.90, 0.80],
            "safety_percentile": [0.90, 0.20, 0.80],
        })
        with patch.object(
            strategy_diag,
            "build_mfe_safety_truth_geometry",
            return_value=(canonical_truth, {"provider": "synthetic"}),
        ), patch.object(
            strategy_diag,
            "load_official_breakout_candidate_keys",
            return_value={("B", pd.Timestamp("2025-01-01")), ("C", pd.Timestamp("2025-01-01"))},
        ):
            geometry_payload = strategy_diag._build_mfe_safety_main_report_geometry(
                project_root=geometry_root,
                settings=geometry_settings,
                pair_payloads=pair_payloads,
            )
        check_true(
            "strategy_compare_main_geometry_uses_breakout_truth_baseline_and_filled_buys",
            bool(
                geometry_payload["status"] == "AVAILABLE"
                and geometry_payload["cohort"] == "filled_buys"
                and geometry_payload["population"]["raw_rows"] == 2
                and geometry_payload["population"]["name"] == "Breakout candidate truth"
                and geometry_payload["arms"]["C68"]["raw_rows"] == 1
                and geometry_payload["arms"]["C69"]["raw_rows"] == 3
                and geometry_payload["arms"]["C69"]["high_mfe_total_pct"] > geometry_payload["arms"]["C68"]["high_mfe_total_pct"]
            ),
        )

    geometry_text = render_mfe_safety_geometry_table({
        "mfe_safety_geometry": {
            "status": "AVAILABLE",
            "percentile_cutoff": 0.50,
            "population": {
                "arm_id": "POP", "name": "All eligible truth", "raw_rows": 3,
                "truth_coverage_pct": 100.0, "high_mfe_high_safety_pct": 33.33,
                "high_mfe_low_safety_pct": 33.33, "low_mfe_high_safety_pct": 33.34,
                "low_mfe_low_safety_pct": 0.0, "high_mfe_total_pct": 66.66,
                "high_safety_total_pct": 66.67,
            },
            "arms": {
                "C68": {
                    "arm_id": "C68", "name": "Control", "raw_rows": 1,
                    "truth_coverage_pct": 100.0, "high_mfe_high_safety_pct": 0.0,
                    "high_mfe_low_safety_pct": 0.0, "low_mfe_high_safety_pct": 100.0,
                    "low_mfe_low_safety_pct": 0.0, "high_mfe_total_pct": 0.0,
                    "high_safety_total_pct": 100.0,
                },
                "C69": {
                    "arm_id": "C69", "name": "Treatment", "raw_rows": 3,
                    "truth_coverage_pct": 100.0, "high_mfe_high_safety_pct": 33.33,
                    "high_mfe_low_safety_pct": 33.33, "low_mfe_high_safety_pct": 33.34,
                    "low_mfe_low_safety_pct": 0.0, "high_mfe_total_pct": 66.66,
                    "high_safety_total_pct": 66.67,
                },
            },
        }
    })
    check_true(
        "strategy_compare_main_report_geometry_renderer_contains_population_and_arm_quadrants",
        "All eligible truth" in geometry_text
        and "C68" in geometry_text
        and "C69" in geometry_text
        and "HM/HS" in geometry_text
        and "High-MFE" in geometry_text
        and "Filled buys" in geometry_text,
    )

    from services.audit.mr13r_joint_capital_drawdown import (
        build_capital_conversion_analysis,
        build_drawdown_analysis,
        build_joint_signal_analysis,
        render_result as render_mr13r_joint_audit_result,
    )

    joint_orderable = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "trade_date": ["2025-01-02"] * 5,
        "signal_date": ["2025-01-01"] * 5,
        "breakout_quality_score_date": ["2025-01-01"] * 5,
        "breakout_quality_score": [0.1, 0.2, 0.3, 0.4, 0.9],
        "breakout_quality_safety_score": [0.1, 0.2, 0.3, 0.4, 0.9],
        "breakout_quality_safety_score_available": [True] * 5,
    })
    joint_truth = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "date": ["2025-01-01"] * 5,
        "mfe_percentile": [0.0, 0.25, 0.50, 0.75, 1.0],
        "safety_percentile": [0.0, 0.25, 0.50, 0.75, 1.0],
    })
    joint_signal = build_joint_signal_analysis(
        joint_orderable, joint_truth, cutoff=0.50, bins=5
    )
    check_true(
        "mr13r_joint_signal_audit_detects_upper_right_hmhs_enrichment_without_runtime_truth_use",
        bool(
            joint_signal["rows"] == 5
            and joint_signal["upper_right_rows"] == 1
            and joint_signal["upper_right_hmhs_pct"] == 100.0
            and joint_signal["upper_right_hmhs_pct"] > joint_signal["population_hmhs_pct"]
            and joint_signal["primary_to_actual_mfe_mean_daily_spearman"] > 0.99
            and joint_signal["safety_to_actual_safety_mean_daily_spearman"] > 0.99
        ),
    )
    joint_report_text = render_mr13r_joint_audit_result({
        "audit_id": "AUD-mr13r-joint-capital-drawdown",
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "evaluations": {
            "forward_oos": {
                "joint_signal": {key: value for key, value in joint_signal.items() if key != "detail"},
                "arms": {},
            }
        },
    })
    check_true(
        "mr13r_joint_signal_report_exposes_upper_right_n_full_5x5_and_safety_cohorts",
        bool(
            "Predicted S5×M5 upper-right N" in joint_report_text
            and "cell = N / actual HM/HS%" in joint_report_text
            and "Pred Safety \\ Cond-MFE" in joint_report_text
            and "Safety cohort conditional-MFE conversion" in joint_report_text
            and "Joint product → actual HM/HS Daily ρ" in joint_report_text
            and not any(line.lstrip().startswith("|") for line in joint_report_text.splitlines())
        ),
    )

    with TemporaryDirectory() as mechanism_temp_text:
        mechanism_root = Path(mechanism_temp_text)
        pair_dir = mechanism_root / "pair"
        pair_dir.mkdir(parents=True)
        pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07"],
            "Equity": [100.0, 95.0, 90.0, 100.0, 110.0, 99.0, 110.0],
            "Exposure_Pct": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0],
        }).to_csv(pair_dir / "score_ranking_equity.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({
            "Date": ["2025-01-02", "2025-01-04", "2025-01-02", "2025-01-06"],
            "Ticker": ["A", "A", "B", "B"],
            "Type": ["買進 (突破)", "全倉結算", "買進 (突破)", "全倉結算"],
            "進場類型": ["normal", "", "normal", ""],
            "買訊日": ["2025-01-01", "", "2025-01-01", ""],
            "候選日": ["2025-01-01", "", "2025-01-01", ""],
            "成交價": [100.0, 105.0, 100.0, 95.0],
            "停損價": [90.0, 90.0, 98.0, 98.0],
            "該筆總損益": [0.0, 5.0, 0.0, -5.0],
            "R_Multiple": [0.0, 0.5, 0.0, -0.5],
        }).to_csv(pair_dir / "score_ranking_trades.csv", index=False, encoding="utf-8-sig")
        mechanism_orderable = pd.DataFrame({
            "ticker": ["A", "B"],
            "trade_date": ["2025-01-02", "2025-01-02"],
            "signal_date": ["2025-01-01", "2025-01-01"],
            "breakout_quality_score_date": ["2025-01-01", "2025-01-01"],
            "breakout_quality_score": [0.7, 0.8],
            "breakout_quality_safety_score": [0.1, 0.9],
            "breakout_quality_safety_score_available": [True, True],
            "projected_capital_fraction": [0.10, 0.30],
            "projected_capital_deployment_rate": [0.20, 0.50],
            "entry_atr": [10.0, 2.0],
            "orig_limit": [100.0, 100.0],
        })
        mechanism_execution = pd.DataFrame({
            "ticker": ["A", "B"],
            "trade_date": ["2025-01-02", "2025-01-02"],
            "signal_date": ["2025-01-01", "2025-01-01"],
            "chosen_qty": [1, 1],
            "limit_px": [100.0, 100.0],
            "init_sl": [90.0, 98.0],
            "sizing_equity": [1000.0, 1000.0],
            "reserved_cost": [100.0, 300.0],
            "chosen_risk_utilization": [0.60, 0.90],
            "binding_signature": ["RISK_CAP", "POSITION_CAP"],
        })
        mechanism_evidence = {
            "pair_dir": pair_dir,
            "orderable": mechanism_orderable,
            "execution": mechanism_execution,
        }
        capital = build_capital_conversion_analysis(mechanism_evidence)
        check_true(
            "mr13r_capital_conversion_audit_attributes_safety_to_stop_distance_and_reserved_notional",
            bool(
                abs(capital["average_exposure_pct"] - 65.0) < 1e-9
                and capital["raw_safety_to_projected_capital_fraction_daily_spearman"] > 0.99
                and capital["selected_raw_safety_to_stop_distance_daily_spearman"] < -0.99
                and capital["selected_raw_safety_to_reserved_fraction_daily_spearman"] > 0.99
                and capital["mean_holding_calendar_days"] == 3.0
            ),
        )

    with TemporaryDirectory() as drawdown_temp_text:
        drawdown_root = Path(drawdown_temp_text)
        drawdown_pair_dir = drawdown_root / "pair"
        market_dir = drawdown_root / "market"
        drawdown_pair_dir.mkdir(parents=True)
        market_dir.mkdir(parents=True)
        pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
            "Equity": [1000.0, 970.0, 950.0, 1010.0, 1020.0],
            "Exposure_Pct": [20.0, 20.0, 20.0, 10.0, 0.0],
        }).to_csv(drawdown_pair_dir / "score_ranking_equity.csv", index=False, encoding="utf-8-sig")
        drawdown_history = pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-04", "2025-01-05"],
            "Ticker": ["A", "C", "C", "B", "A", "B"],
            "Type": ["買進 (突破)", "買進 (突破)", "全倉結算", "買進 (突破)", "全倉結算", "全倉結算"],
            "進場類型": ["normal", "normal", "", "normal", "", ""],
            "買訊日": ["2025-01-01", "2025-01-01", "", "2025-01-01", "", ""],
            "候選日": ["2025-01-01", "2025-01-01", "", "2025-01-01", "", ""],
            "成交價": [100.0, 100.0, 90.0, 100.0, 120.0, 110.0],
            "停損價": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
            "股數": [1, 1, 1, 1, 1, 1],
            "投入總金額": [100.0, 100.0, 0.0, 100.0, 0.0, 0.0],
            "該筆總損益": [0.0, 0.0, -10.0, 0.0, 20.0, 10.0],
            "R_Multiple": [0.0, 0.0, -1.0, 0.0, 2.0, 1.0],
        })
        drawdown_history.to_csv(
            drawdown_pair_dir / "score_ranking_trades.csv", index=False, encoding="utf-8-sig"
        )
        (drawdown_pair_dir / "strategy_comparison.json").write_text(
            json.dumps({
                "metadata": {
                    "score_ranking_params": {
                        "buy_fee": 0.0,
                        "sell_fee": 0.0,
                        "tax_rate": 0.0,
                        "min_fee": 0.0,
                        "fixed_risk": 0.01,
                    }
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        market_closes = {
            "A": [100.0, 90.0, 80.0, 120.0, 120.0],
            "B": [100.0, 90.0, 80.0, 100.0, 110.0],
            "C": [100.0, 90.0, 90.0, 90.0, 90.0],
        }
        market_dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]
        for ticker, closes in market_closes.items():
            pd.DataFrame({
                "Date": market_dates,
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [1000.0] * len(closes),
            }).to_csv(market_dir / f"{ticker}.csv", index=False)

        drawdown_path = pd.DataFrame({
            "match_key": ["A|2025-01-01|normal|1", "B|2025-01-02|normal|1", "C|2025-01-01|normal|1"],
            "ticker": ["A", "B", "C"],
            "score_event_date": ["2025-01-01"] * 3,
            "entry_date": ["2025-01-01", "2025-01-02", "2025-01-01"],
            "exit_date": ["2025-01-04", "2025-01-05", "2025-01-02"],
            # Deliberately positive in aggregate: this proves final realized R cannot
            # stand in for peak→trough drawdown contribution.
            "realized_r": [2.0, 1.0, -1.0],
        })
        drawdown_truth = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "date": ["2025-01-01"] * 3,
            "mfe_percentile": [0.9, 0.9, 0.2],
            "safety_percentile": [0.9, 0.2, 0.9],
        })
        drawdown = build_drawdown_analysis(
            {
                "pair_dir": drawdown_pair_dir,
                "active_trades": drawdown_history,
                "upside_realization": drawdown_path,
            },
            drawdown_truth,
            cutoff=0.50,
            top_n=2,
            market_data_dir=market_dir,
        )
        first_dd = drawdown["top_episodes"][0]
        contributions = pd.DataFrame(drawdown["position_contributions"])
        check_true(
            "mr13r_drawdown_audit_reconciles_true_peak_to_trough_mtm_instead_of_final_realized_r",
            bool(
                abs(drawdown["max_drawdown_pct"] - 5.0) < 1e-9
                and first_dd["peak_date"] == "2025-01-01"
                and first_dd["trough_date"] == "2025-01-03"
                and abs(first_dd["equity_change"] + 50.0) < 1e-9
                and abs(first_dd["position_mtm_contribution_sum"] + 50.0) < 1e-9
                and abs(first_dd["reconciliation_delta"]) < 1e-9
                and first_dd["peak_held_count"] == 2
                and first_dd["entered_during_drawdown_count"] == 1
                and first_dd["exited_during_drawdown_count"] == 1
                and first_dd["trough_held_count"] == 2
                and first_dd["negative_contributor_count"] == 3
                and abs(first_dd["hmhs_mtm_contribution"] + 20.0) < 1e-9
                and abs(first_dd["hmls_mtm_contribution"] + 20.0) < 1e-9
                and abs(first_dd["lmhs_mtm_contribution"] + 10.0) < 1e-9
                and abs(first_dd["lmls_mtm_contribution"]) < 1e-9
                and abs(float(drawdown_path["realized_r"].sum()) - 2.0) < 1e-9
                and len(contributions) == 3
            ),
        )

    from core.display import _display_width

    readable_report_text = render_mr13r_joint_audit_result({
        "audit_id": "AUD-mr13r-joint-capital-drawdown",
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "evaluations": {
            "forward_oos": {
                "joint_signal": {key: value for key, value in joint_signal.items() if key != "detail"},
                "arms": {
                    "C71": {
                        "capital_conversion": capital,
                        "drawdown": drawdown,
                    }
                },
            }
        },
    })
    readable_widths = [_display_width(line) for line in readable_report_text.splitlines()]
    check_true(
        "mr13r_human_report_uses_aligned_bounded_text_tables_instead_of_markdown_pipes",
        bool(
            max(readable_widths, default=0) <= 100
            and not any(line.lstrip().startswith("|") for line in readable_report_text.splitlines())
            and "Capital conversion｜曝險與持倉" in readable_report_text
            and "Capital conversion｜Safety 關聯" in readable_report_text
            and "Max drawdown｜持倉生命週期" in readable_report_text
            and "Max drawdown｜四象限 MTM contribution" in readable_report_text
        ),
    )


    summary["workflow"] = "config_driven_formal_audit_topology"
    return results, summary
