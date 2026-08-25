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
        load_strategy_arm_replay_sidecars,
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
            ),
        )

    summary["workflow"] = "config_driven_formal_audit_topology"
    return results, summary
