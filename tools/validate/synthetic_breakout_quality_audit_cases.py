from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from .checks import raises_expected, bind_synthetic_case, bind_checks

from .synthetic_breakout_quality_support import add_check


def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import AUDIT_OUTPUT_ROOT
    from core.audit_policy import (
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
    results, summary, check, check_true = bind_synthetic_case(case_id, "synthetic_breakout_quality")

    validate_audit_config()
    definitions = get_audit_definitions("breakout_quality")
    enabled = get_enabled_audit_definitions("breakout_quality")
    validate_audit_catalog(definitions)

    import config.audit as audit_cfg
    import core.audit_registry as audit_registry

    check_true(
        "formal_audit_registry_and_user_policy_have_single_owners",
        bool(
            get_audit_module_ids()[0] == "breakout_quality"
            and len(definitions) >= len(enabled)
            and AUDIT_OUTPUT_ROOT == "outputs/audit"
            and not hasattr(audit_cfg, "AUDIT_MODULES")
            and hasattr(audit_registry, "AUDIT_MODULES")
        ),
    )

    # Synthetic contract checks must not read the developer's current research artifacts.
    # Using an isolated root keeps runtime status validation deterministic and bounded even
    # when the real project contains large Strategy Compare / Audit sidecars.
    with TemporaryDirectory(prefix="synthetic_audit_status_") as status_root_text:
        status = collect_audit_status(
            module_id="breakout_quality",
            project_root=Path(status_root_text),
        )
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
        "research_point_in_time_score_validation_remains_catalogued",
        "audit-point-in-time-scores" in commands,
    )
    # One-time Audit may have zero definitions; this is not a failure of reusable Audit.
    check_true(
        "retired_one_time_studies_do_not_change_reusable_audit_contract",
        len(definitions) == len(enabled)
        and {entry.report_id for entry in get_reusable_audit_definitions("breakout_quality")}
        == {"opportunity_selection", "trade_outcome_path", "portfolio_drawdown", "target_oracle_romd"},
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

    empty_one_time_console = io.StringIO()
    # The empty-state check must isolate the registry even if a new one-time
    # experiment is active; do not assert that the live project has zero Audits.
    with patch.object(research_app, "get_audit_definitions", return_value=()), patch(
        "builtins.input", return_value="0"
    ), redirect_stdout(empty_one_time_console):
        research_app._audit_one_time_menu("breakout_quality")
    check_true(
        "empty_one_time_audit_menu_keeps_existing_entry_and_exits_cleanly",
        "=== 一次性專題 Audit ===" in empty_one_time_console.getvalue()
        and "目前沒有一次性 Audit 設定。" in empty_one_time_console.getvalue()
        and "[0]" in empty_one_time_console.getvalue(),
    )

    reusable_console = io.StringIO()
    with patch.object(audit_runner, "get_audit_definitions", return_value=()), patch(
        "builtins.input", side_effect=["1", "0", "0"]
    ), redirect_stdout(reusable_console):
        research_app._audit_menu()
    reusable_text = reusable_console.getvalue()
    check_true(
        "reusable_audit_menu_keeps_registered_reports_visible",
        "Opportunity／Selection Attribution  [已設定]" in reusable_text
        and "Trade Outcome／Path Attribution  [已設定]" in reusable_text
        and "Portfolio／Drawdown Attribution  [已設定]" in reusable_text
        and "Target 驗證" not in reusable_text
        and "Regime／Stability Attribution" not in reusable_text,
    )
    # Target validation has exactly one interactive home at Research [1].
    # The reusable Audit menu no longer shows the same execution route.
    from services.audit.catalog import get_audit_methods
    oracle_method_id = "target_ground_truth_strategy_romd"
    method_ids = [method.method_id for method in get_audit_methods()]
    audit_method_ids = [method.method_id for method in get_audit_methods(menu_location="audit")]
    research_method_ids = [method.method_id for method in get_audit_methods(menu_location="research_root")]
    check_true(
        "oracle_formal_target_route_appears_only_at_research_root",
        method_ids.count(oracle_method_id) == 1
        and research_method_ids == [oracle_method_id]
        and oracle_method_id not in audit_method_ids,
    )
    # A validated, final identity plan must appear BEFORE the only approval.
    # This synthetic owns the Research UI contract (not an existence inventory).
    route_console = io.StringIO()
    prompts = []
    preview = (
        "[Target 驗證 Fixed | Portfolio Replay 正式計畫]\n"
        "Target                S1     S2\n"
        "C96                   REUSE  REUSE\n"
        "DailyUniversal40      RUN    REUSE\n"
        "REUSE 3 / RUN（需回放）1 / BLOCKED 0"
    )
    executions = []
    answers = iter(["", "", "0"])

    def input_after_plan(prompt):
        prompts.append((prompt, preview in route_console.getvalue()))
        return next(answers)

    def fake_target_runner(*args, **kwargs):
        # Audit runner owns certification AND scientific execution. The app
        # cannot print a preliminary existence-based REUSE table or redecide.
        print(preview)
        approval = kwargs["preflight_approval"](preview)
        if not approval:
            return {"target_oracle_romd": {"status": "CANCELLED"}}
        executions.append("replayed missing arms")
        print("[RUN] Target seed=11 | no completed local result")
        return {"target_oracle_romd": {"status": "COMPLETED"}}

    with patch("builtins.input", side_effect=input_after_plan), \
         patch.object(research_app, "render_audit_status") as expensive_status, \
         patch.object(research_app, "run_enabled_audits", side_effect=fake_target_runner) as oracle_runner, \
         patch.object(research_app, "render_latest_audit_summary", return_value="LATEST"), \
         redirect_stdout(route_console):
        route_rc = research_app._interactive_menu()
    root_text = route_console.getvalue()
    check_true(
        "target_validation_runs_once_from_research_default_enter",
        route_rc == 0 and oracle_runner.call_count == 1
        and oracle_runner.call_args.kwargs.get("method_id") == oracle_method_id
        and callable(oracle_runner.call_args.kwargs.get("preflight_approval"))
        and "preflight_display" not in oracle_runner.call_args.kwargs
        and "before_replay" not in oracle_runner.call_args.kwargs
        and not expensive_status.called
        and "[1]" in root_text and "Target 驗證" in root_text
        and "[2]" in root_text and "模型訓練／驗證" in root_text
        and "[5]" in root_text and "Audit／診斷" in root_text
        and "[9]" in root_text and "策略升格" in root_text,
    )
    check_true(
        "target_validation_only_one_confirm_even_if_verified_run_count_changes",
        len(prompts) == 3 and not prompts[0][1] and prompts[1][1]
        and "執行正式 REUSE／RUN 計畫" in prompts[1][0]
        and all("輸入 RUN" not in prompt for prompt, _ in prompts)
        and root_text.count(preview) == 1
        and root_text.index(preview) < root_text.index("[RUN] Target")
        and executions == ["replayed missing arms"]
        and "LATEST" in root_text,
    )

    # Cancel must still build the read-only plan, but never produce/commit any
    # scientific result, even if a cached target score source is missing.
    cancellation = io.StringIO()
    executions.clear()
    with patch("builtins.input", return_value="0") as cancel_input, \
         patch.object(research_app, "render_audit_status") as status, \
         patch.object(research_app, "run_enabled_audits", side_effect=fake_target_runner) as cancelled_runner, \
         patch.object(research_app, "render_latest_audit_summary") as latest, \
         redirect_stdout(cancellation):
        research_app._run_reusable_audit_method("breakout_quality", oracle_method_id)
    check_true(
        "target_single_enter_cancel_is_before_any_expensive_work",
        cancel_input.call_count == 1
        and cancelled_runner.call_count == 1 and not executions
        and not latest.called and not status.called
        and cancellation.getvalue().count(preview) == 1
        and "未授權執行" in cancellation.getvalue(),
    )

    # The exact official status, not a preliminary file-existence guess, owns
    # both the displayed matrix and execution after the same approval.
    changed_plan_console = io.StringIO()
    verified_calls = []
    final_matrix = preview.replace("REUSE 3 / RUN（需回放）1", "REUSE 2 / RUN（需回放）2")
    def exact_plan(*args, **kwargs):
        verified_calls.append(kwargs)
        print(final_matrix)
        assert kwargs["preflight_approval"](final_matrix)
        return {"target_oracle_romd": {"status": "COMPLETED"}}
    with patch("builtins.input", return_value="") as single_input, \
         patch.object(research_app, "run_enabled_audits", side_effect=exact_plan), \
         patch.object(research_app, "render_latest_audit_summary", return_value="LATEST"), \
         redirect_stdout(changed_plan_console):
        result = research_app._run_reusable_audit_method("breakout_quality", oracle_method_id)
    check_true(
        "target_exact_verified_plan_runs_without_more_confirmation",
        result == 0 and single_input.call_count == 1
        and len(verified_calls) == 1 and "before_replay" not in verified_calls[0]
        and "RUN（需回放）2" in changed_plan_console.getvalue()
        and changed_plan_console.getvalue().count(final_matrix) == 1
        and "LATEST" in changed_plan_console.getvalue(),
    )

    # Regression for the real one-stage contract: no intermediate truth BUILD,
    # PENDING or second confirmation. A complete REUSE/RUN matrix MUST be
    # printed before the *first* approval, even with no persisted truth sidecar.
    from services.audit import target_oracle_romd as oracle_handler
    missing_truth_view = io.StringIO()
    truth_prompts = []
    certified_matrix = (
        "[Target 驗證 Fixed | Portfolio Replay 正式計畫]\n"
        "Target                        S1     S2     S3     S4\n"
        "C96                           REUSE  REUSE  REUSE  REUSE\n"
        "DailyUniversal_MinTrade40    REUSE  REUSE  REUSE  REUSE\n"
        "DailyUniversal_MinTrade      RUN    RUN    RUN    RUN\n"
        "REUSE 8 / RUN（需回放）4 / BLOCKED 0"
    )

    def approved_matrix_plan(_definition, **kwargs):
        print(certified_matrix)
        if not kwargs["preflight_approval"](certified_matrix):
            return {"status": "CANCELLED", "read_only": True}
        raise AssertionError("cancelled one-stage plan executed a Producer")

    def abort_after_matrix(prompt):
        truth_prompts.append((prompt, certified_matrix in missing_truth_view.getvalue()))
        return "0"

    def simulate_real_research_runner(_module_id, *, project_root, method_id,
                                      preflight_approval):
        return {"target_oracle_romd": oracle_handler.build_report(
            object(), project_root=project_root, quiet=False,
            preflight_approval=preflight_approval,
        )}

    with patch.object(research_app, "collect_audit_menu_state", return_value={"configured": True}), \
         patch.object(research_app, "run_enabled_audits", side_effect=simulate_real_research_runner), \
         patch.object(research_app, "render_latest_audit_summary") as premature_report, \
         patch.object(oracle_handler, "_run_single_stage_oracle", side_effect=approved_matrix_plan) as planner, \
         patch.object(oracle_handler, "_execute_single_stage_missing") as expensive_producer, \
         patch("builtins.input", side_effect=abort_after_matrix), \
         redirect_stdout(missing_truth_view):
        research_app._run_reusable_audit_method("breakout_quality", oracle_method_id)
    check_true(
        "target_missing_truth_shows_certified_reuse_run_matrix_before_first_and_only_approval",
        len(truth_prompts) == 1 and truth_prompts[0][1]
        and "執行正式 REUSE／RUN 計畫" in truth_prompts[0][0]
        and "PENDING" not in missing_truth_view.getvalue()
        and "BUILD" not in missing_truth_view.getvalue()
        and missing_truth_view.getvalue().count(certified_matrix) == 1
        and missing_truth_view.getvalue().index(certified_matrix)
            < missing_truth_view.getvalue().index("未授權執行")
        and planner.call_count == 1
        and planner.call_args.kwargs["allow_historical_execution_input_drift"] is True
        and not expensive_producer.called and not premature_report.called,
    )

    # Accept one matrix and assert exactly one approval, then RUN only the
    # specifically authorized missing arm; the app may not second-guess the
    # scientific Owner or request another staged approval.
    one_stage = io.StringIO()
    approvals = []
    def one_stage_input(prompt):
        approvals.append((prompt, certified_matrix in one_stage.getvalue()))
        return ""
    def one_stage_runner(*args, **kwargs):
        print(certified_matrix)
        assert kwargs["preflight_approval"](certified_matrix)
        print("[RUN] only missing Portfolio")
        return {"target_oracle_romd": {"status": "COMPLETED"}}
    with patch("builtins.input", side_effect=one_stage_input), \
         patch.object(research_app, "run_enabled_audits", side_effect=one_stage_runner), \
         patch.object(research_app, "render_latest_audit_summary", return_value="LATEST"), \
         redirect_stdout(one_stage):
        research_app._run_reusable_audit_method("breakout_quality", oracle_method_id)
    check_true(
        "target_truth_and_portfolio_are_owned_by_one_certified_approval_only",
        len(approvals) == 1 and approvals[0][1]
        and "執行正式 REUSE／RUN 計畫" in approvals[0][0]
        and "PENDING" not in one_stage.getvalue()
        and one_stage.getvalue().count(certified_matrix) == 1
        and one_stage.getvalue().index(certified_matrix)
            < one_stage.getvalue().index("[RUN] only missing Portfolio"),
    )

    # Entry-point integration belongs to the existing app/menu contract, not
    # the domain Target synthetic. Exercise the *real* Audit runner on Enter.
    # The domain-level tests separately verify canonical per-seed Min parameters.
    with TemporaryDirectory(prefix="synthetic_target_entry_no_fit_") as test_root:
        isolated_root = Path(test_root)
        missing_teacher_status = {
            "module_id": "breakout_quality",
            "method_id": oracle_method_id,
            "audit_id": None,
            "overall_status": "BLOCKED",
            "rows": [{
                "enabled": True,
                "audit_id": "target_oracle_romd",
                "status": "BLOCKED",
                "reason": "formal Min seed parameter source missing",
                "source": {
                    "path": "models/missing_seed.json",
                    "preparable": False,
                    "builder_type": "optimizer_fixed_multiseed_strategy_params",
                },
            }],
        }
        blocked_text = ""
        entry_console = io.StringIO()
        from services.audit import target_oracle_romd as oracle_handler
        with patch.object(research_app, "PROJECT_ROOT", isolated_root), \
             patch("builtins.input", return_value=""), \
             patch.object(audit_runner, "collect_audit_status", return_value=missing_teacher_status), \
             patch("services.research.strategy_compare_preparation.prepare_strategy_parameter_artifacts", side_effect=AssertionError("Optimizer invoked from Target validation")) as optimizer, \
             patch.object(oracle_handler, "run_formal_audit", side_effect=AssertionError("Replay invoked with missing Teacher")) as replay, \
             patch.object(research_app, "render_latest_audit_summary") as latest, \
             redirect_stdout(entry_console):
            try:
                research_app._run_reusable_audit_method("breakout_quality", oracle_method_id)
            except RuntimeError as error:
                blocked_text = str(error)
        check_true(
            "research_target_enter_missing_param_source_is_blocked_without_optimizer_or_replay",
            "BLOCKED" in blocked_text
            and "formal Min seed" in blocked_text
            and "Target 驗證" in entry_console.getvalue()
            and optimizer.call_count == 0
            and replay.call_count == 0
            and latest.call_count == 0
            and not (isolated_root / "models" / "missing_seed.json").exists(),
        )

    reusable_defs = get_reusable_audit_definitions("breakout_quality")
    check_true(
        "reusable_audit_reports_are_registry_driven_and_not_scientific_aud_ids",
        len(reusable_defs) == 4
        and {item.report_id for item in reusable_defs}
        == {"opportunity_selection", "trade_outcome_path", "portfolio_drawdown", "target_oracle_romd"}
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

    from services.audit.selection_membership import event_key_series
    canonical_event_keys = event_key_series(pd.DataFrame({
        "ticker": ["A", "B"],
        "trade_date": ["2025-01-02", "2025-01-03"],
        "signal_date": ["2025-01-01", "2025-01-02"],
    }))
    check_true(
        "reusable_audit_event_identity_has_one_canonical_composition_helper",
        list(canonical_event_keys)
        == ["A|2025-01-02|2025-01-01", "B|2025-01-03|2025-01-02"],
    )

    from services.audit.strategy_compare_source import (
        AuditSourceBlockedError as StrategyAuditSourceBlockedError,
        resolve_strategy_compare_period,
    )
    period_source = SimpleNamespace(
        profile_id="synthetic_profile",
        result={
            "comparison_period": {
                "start": "2025-01-01T12:34:56",
                "end": "2025-01-31T23:59:59",
            }
        },
    )
    check(
        "reusable_audit_comparison_period_uses_one_canonical_source_helper",
        ("2025-01-01", "2025-01-31"),
        resolve_strategy_compare_period(period_source),
    )
    missing_period_blocked = raises_expected(StrategyAuditSourceBlockedError, lambda: resolve_strategy_compare_period(
            SimpleNamespace(profile_id="synthetic_profile", result={})
        ))
    check_true(
        "canonical_reusable_audit_comparison_period_missing_evidence_still_blocks",
        missing_period_blocked,
    )

    audit_root = Path(__file__).resolve().parents[2] / "services" / "audit"
    reusable_period_sources = [
        (audit_root / name).read_text(encoding="utf-8")
        for name in (
            "opportunity_selection.py",
            "trade_outcome_path.py",
            "portfolio_drawdown.py",
        )
    ]
    check_true(
        "reusable_audit_period_consumers_do_not_reimplement_identical_private_period_helpers",
        all(
            "resolve_strategy_compare_period(source)" in source_text
            and "def _period(source)" not in source_text
            for source_text in reusable_period_sources
        ),
    )
    trade_path_source = (audit_root / "trade_outcome_path.py").read_text(encoding="utf-8")
    membership_source = (audit_root / "selection_membership.py").read_text(encoding="utf-8")
    check_true(
        "trade_path_consumes_membership_event_identity_without_duplicate_event_key_builder",
        "event_key_series(table)" in trade_path_source
        and "def _event_key(" not in trade_path_source
        and membership_source.count("def event_key_series(") == 1,
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

    from core.strategy_compare_policy import get_strategy_comparison_settings
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



    from filters.breakout_quality.strategy_compare_diagnostics import render_mfe_safety_geometry_table
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

    from services.audit.portfolio_mtm import build_drawdown_analysis

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
            "canonical_portfolio_mtm_reconciles_true_peak_to_trough_instead_of_final_realized_r",
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

    # AUD-optimizer-score-zero-collapse was a one-time decision Audit.  Its
    # accepted finding is now encoded by the formal PARAM-P12/C91 producer contract,
    # so the dedicated implementation/registration synthetic is intentionally retired.

    summary["workflow"] = "config_driven_formal_audit_topology"
    return results, summary
