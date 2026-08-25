from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
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

    import apps.research as research_app

    import services.audit.runner as audit_runner

    audit_menu_console = io.StringIO()
    with patch.object(audit_runner, "get_audit_definitions", return_value=()), patch(
        "builtins.input", return_value="0"
    ), redirect_stdout(audit_menu_console):
        audit_menu_rc = research_app._audit_menu()
    audit_menu_text = audit_menu_console.getvalue()
    check_true(
        "research_audit_menu_keeps_stable_methods_but_no_run_action_without_config",
        audit_menu_rc == 0
        and "策略 Pair／Portfolio Attribution  [未設定]" in audit_menu_text
        and "Trade Path／Upside Survival  [未設定]" in audit_menu_text
        and "Selection／Truth Geometry  [未設定]" in audit_menu_text
        and "跨期／跨 Seed Stability Attribution  [未設定]" in audit_menu_text
        and "執行目前參數設定" not in audit_menu_text,
    )

    from config.strategy_compare import get_strategy_comparison_settings
    from filters.breakout_quality.strategy_compare_contracts import (
        COMPARISON_MODE_SCORE_RANKING,
    )
    from services.audit.strategy_compare_source import (
        StrategyCompareAuditSource,
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

    from types import SimpleNamespace
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
        ):
            geometry_payload = strategy_diag._build_mfe_safety_main_report_geometry(
                project_root=geometry_root,
                settings=geometry_settings,
                pair_payloads=pair_payloads,
            )
        check_true(
            "strategy_compare_main_geometry_uses_filled_buys_and_all_eligible_canonical_truth",
            bool(
                geometry_payload["status"] == "AVAILABLE"
                and geometry_payload["cohort"] == "filled_buys"
                and geometry_payload["population"]["raw_rows"] == 3
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

    summary["workflow"] = "config_driven_formal_audit_topology"
    return results, summary
