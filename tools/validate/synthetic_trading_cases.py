from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from .checks import add_check
from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    money_to_milli,
)
from core.trading_account_state import validate_trading_account_state
from services.trading.account_state import (
    TradingAccountRevisionConflict,
    adopt_existing_trading_position,
    correct_existing_trading_position,
    remove_existing_trading_position,
    confirm_trading_sell_fill,
    confirm_trading_strategy_buy_fill,
    get_trading_account_read_model,
    initialize_trading_account_state,
    load_trading_account_state,
    resolve_trading_account_state_path,
    set_trading_cash_balance,
)


def validate_trading_account_state_contract_case(base_params):
    case_id = "TRADING_ACCOUNT_STATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    add_check(results, "trading_account", case_id, "trading_market_data_is_gitignored", True, "/data/trading/" in gitignore)
    add_check(results, "trading_account", case_id, "trading_operational_state_is_gitignored", True, "/state/trading/" in gitignore)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        state_path = resolve_trading_account_state_path(root)
        add_check(results, "trading_account", case_id, "account_state_path_is_under_trading_state_root", str((root / "state" / "trading" / "account.json").resolve()), str(state_path.resolve()))

        state = initialize_trading_account_state(root, cash=1_000_000)
        initial_cash_milli = money_to_milli(1_000_000)
        add_check(results, "trading_account", case_id, "initialize_revision_zero", 0, state["revision"])
        add_check(results, "trading_account", case_id, "initialize_cash_exact_milli", initial_cash_milli, state["cash_milli"])
        add_check(results, "trading_account", case_id, "initialize_event_chain_starts_at_revision_zero", [0], [event["revision"] for event in state["events"]])

        state = set_trading_cash_balance(root, cash=1_100_000, expected_revision=state["revision"], note="synthetic reconciliation")
        add_check(results, "trading_account", case_id, "cash_reconciliation_advances_one_revision", 1, state["revision"])
        add_check(results, "trading_account", case_id, "cash_reconciliation_updates_exact_cash", money_to_milli(1_100_000), state["cash_milli"])

        try:
            set_trading_cash_balance(root, cash=1_200_000, expected_revision=0)
        except TradingAccountRevisionConflict:
            stale_rejected = True
        else:
            stale_rejected = False
        add_check(results, "trading_account", case_id, "stale_revision_is_rejected", True, stale_rejected)

        cash_before_adoption = state["cash_milli"]
        state = adopt_existing_trading_position(
            root,
            ticker="2330",
            qty=1000,
            cost_basis_total=500_000,
            entry_date="2026-08-01",
            expected_revision=state["revision"],
            note="existing broker position",
        )
        adopted = state["positions"]["2330"]
        add_check(results, "trading_account", case_id, "manual_adoption_does_not_change_cash", cash_before_adoption, state["cash_milli"])
        add_check(results, "trading_account", case_id, "manual_adoption_source_is_explicit", "manual_adopted", adopted["source"])
        add_check(results, "trading_account", case_id, "manual_adoption_does_not_forge_strategy_state", None, adopted["strategy_management"]["position_state"])
        add_check(results, "trading_account", case_id, "manual_adoption_management_is_unmanaged", "unmanaged", adopted["strategy_management"]["status"])

        try:
            adopt_existing_trading_position(
                root,
                ticker="2330",
                qty=100,
                cost_basis_total=50_000,
                expected_revision=state["revision"],
            )
        except ValueError:
            duplicate_rejected = True
        else:
            duplicate_rejected = False
        add_check(results, "trading_account", case_id, "duplicate_open_ticker_is_rejected", True, duplicate_rejected)

        cash_before_correction = state["cash_milli"]
        state = correct_existing_trading_position(
            root,
            ticker="2330",
            qty=1200,
            cost_basis_total=540_000,
            entry_date="2026-08-02",
            expected_revision=state["revision"],
            note="synthetic broker correction",
        )
        corrected = state["positions"]["2330"]
        add_check(results, "trading_account", case_id, "manual_correction_does_not_change_cash", cash_before_correction, state["cash_milli"])
        add_check(results, "trading_account", case_id, "manual_correction_updates_broker_qty", 1200, corrected["broker"]["qty"])
        add_check(results, "trading_account", case_id, "manual_correction_updates_broker_cost_basis", money_to_milli(540_000), corrected["broker"]["remaining_cost_basis_milli"])
        add_check(results, "trading_account", case_id, "manual_correction_does_not_create_strategy_state", None, corrected["strategy_management"]["position_state"])

        state = adopt_existing_trading_position(
            root,
            ticker="2454",
            qty=200,
            cost_basis_total=200_000,
            entry_date="2026-08-03",
            expected_revision=state["revision"],
        )
        cash_before_remove = state["cash_milli"]
        state = remove_existing_trading_position(
            root,
            ticker="2454",
            expected_revision=state["revision"],
            note="synthetic erroneous manual row",
        )
        add_check(results, "trading_account", case_id, "manual_remove_does_not_change_cash", cash_before_remove, state["cash_milli"])
        add_check(results, "trading_account", case_id, "manual_remove_deletes_only_selected_position", False, "2454" in state["positions"])

        cash_before_buy = state["cash_milli"]
        expected_buy = build_buy_ledger_from_price(100, 1000, base_params)
        state = confirm_trading_strategy_buy_fill(
            root,
            ticker="2317",
            qty=1000,
            buy_price=100,
            params=base_params,
            trade_date="2026-09-04",
            expected_revision=state["revision"],
            init_sl=90,
            init_trail=90,
            target_price=120,
            limit_price=105,
        )
        strategy_record = state["positions"]["2317"]
        add_check(results, "trading_account", case_id, "strategy_buy_uses_exact_accounting_cash", cash_before_buy - expected_buy["net_buy_total_milli"], state["cash_milli"])
        add_check(results, "trading_account", case_id, "strategy_buy_broker_cost_matches_canonical_position", strategy_record["broker"]["remaining_cost_basis_milli"], strategy_record["strategy_management"]["position_state"]["remaining_cost_basis_milli"])
        add_check(results, "trading_account", case_id, "strategy_buy_management_is_active", "active", strategy_record["strategy_management"]["status"])

        pre_same_day = deepcopy(state)
        try:
            confirm_trading_sell_fill(
                root,
                ticker="2317",
                qty=500,
                exec_price=110,
                params=base_params,
                trade_date="2026-09-04",
                expected_revision=state["revision"],
            )
        except ValueError:
            same_day_rejected = True
        else:
            same_day_rejected = False
        add_check(results, "trading_account", case_id, "same_day_buy_sell_is_rejected", True, same_day_rejected)
        add_check(results, "trading_account", case_id, "rejected_sell_does_not_mutate_persisted_revision", pre_same_day["revision"], load_trading_account_state(root)["revision"])

        try:
            confirm_trading_sell_fill(
                root,
                ticker="2317",
                qty=2000,
                exec_price=110,
                params=base_params,
                trade_date="2026-09-05",
                expected_revision=state["revision"],
            )
        except ValueError:
            oversell_rejected = True
        else:
            oversell_rejected = False
        add_check(results, "trading_account", case_id, "oversell_is_rejected", True, oversell_rejected)
        add_check(results, "trading_account", case_id, "oversell_rejection_keeps_revision", state["revision"], load_trading_account_state(root)["revision"])

        broker_before = deepcopy(state["positions"]["2317"]["broker"])
        cash_before_sell = state["cash_milli"]
        expected_sell = build_sell_ledger_from_price(110, 500, base_params, ticker="2317", trade_date="2026-09-05")
        allocated = allocate_cost_basis_milli(broker_before["remaining_cost_basis_milli"], broker_before["qty"], 500)
        state = confirm_trading_sell_fill(
            root,
            ticker="2317",
            qty=500,
            exec_price=110,
            params=base_params,
            trade_date="2026-09-05",
            expected_revision=state["revision"],
        )
        broker_after = state["positions"]["2317"]["broker"]
        strategy_after = state["positions"]["2317"]["strategy_management"]["position_state"]
        add_check(results, "trading_account", case_id, "sell_cash_uses_canonical_net_proceeds", cash_before_sell + expected_sell["net_sell_total_milli"], state["cash_milli"])
        add_check(results, "trading_account", case_id, "sell_cost_basis_uses_canonical_allocation", broker_before["remaining_cost_basis_milli"] - allocated, broker_after["remaining_cost_basis_milli"])
        add_check(results, "trading_account", case_id, "strategy_and_broker_qty_remain_equal_after_sell", broker_after["qty"], strategy_after["qty"])
        add_check(results, "trading_account", case_id, "strategy_and_broker_cost_remain_equal_after_sell", broker_after["remaining_cost_basis_milli"], strategy_after["remaining_cost_basis_milli"])

        manual_before = deepcopy(state["positions"]["2330"]["broker"])
        cash_before_manual_sell = state["cash_milli"]
        manual_sell = build_sell_ledger_from_price(550, 200, base_params, ticker="2330", trade_date="2026-09-05")
        manual_allocated = allocate_cost_basis_milli(manual_before["remaining_cost_basis_milli"], manual_before["qty"], 200)
        state = confirm_trading_sell_fill(
            root,
            ticker="2330",
            qty=200,
            exec_price=550,
            params=base_params,
            trade_date="2026-09-05",
            expected_revision=state["revision"],
        )
        manual_after = state["positions"]["2330"]["broker"]
        add_check(results, "trading_account", case_id, "manual_position_sell_uses_canonical_net_proceeds", cash_before_manual_sell + manual_sell["net_sell_total_milli"], state["cash_milli"])
        add_check(results, "trading_account", case_id, "manual_position_partial_cost_basis_is_allocated_canonically", manual_before["remaining_cost_basis_milli"] - manual_allocated, manual_after["remaining_cost_basis_milli"])
        add_check(results, "trading_account", case_id, "manual_position_remains_unmanaged_after_broker_sell", "unmanaged", state["positions"]["2330"]["strategy_management"]["status"])
        try:
            correct_existing_trading_position(
                root,
                ticker="2330",
                qty=800,
                cost_basis_total=400_000,
                entry_date="2026-08-02",
                expected_revision=state["revision"],
            )
        except ValueError:
            sold_manual_correction_rejected = True
        else:
            sold_manual_correction_rejected = False
        add_check(results, "trading_account", case_id, "manual_position_with_sell_history_cannot_be_corrected", True, sold_manual_correction_rejected)
        add_check(results, "trading_account", case_id, "sold_manual_correction_reject_keeps_revision", state["revision"], load_trading_account_state(root)["revision"])

        validate_trading_account_state(state)
        expected_revisions = list(range(state["revision"] + 1))
        add_check(results, "trading_account", case_id, "event_revisions_are_contiguous", expected_revisions, [event["revision"] for event in state["events"]])
        add_check(results, "trading_account", case_id, "event_hash_chain_ends_at_current_revision", state["revision"], state["events"][-1]["revision"])

        tampered = deepcopy(state)
        tampered["events"][1]["details"]["cash_milli"] += 1
        try:
            validate_trading_account_state(tampered)
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        add_check(results, "trading_account", case_id, "event_history_tampering_is_rejected", True, tamper_rejected)

        read_model = get_trading_account_read_model(root)
        add_check(results, "trading_account", case_id, "read_model_revision_matches_state", state["revision"], read_model["revision"])
        add_check(results, "trading_account", case_id, "read_model_position_count_matches_open_positions", len(state["positions"]), read_model["position_count"])
        add_check(results, "trading_account", case_id, "persisted_state_is_single_account_file", True, state_path.is_file())
        add_check(results, "trading_account", case_id, "separate_positions_truth_file_is_not_created", False, (state_path.parent / "positions.json").exists())

    summary["checks"] = len(results)
    return results, summary



def validate_trading_daily_workflow_contract_case(base_params):
    case_id = "TRADING_DAILY_WORKFLOW"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    import services.downloader.application as downloader_application
    import services.trading.daily_workflow as daily_workflow

    profile = get_trading_strategy_profile()
    project_root = Path(__file__).resolve().parents[2]

    with patch.object(downloader_application, "get_market_last_date", return_value="2026-09-04"), \
         patch.object(downloader_application, "get_or_update_universe", return_value=["2330", "2454"]), \
         patch.object(downloader_application, "smart_download_vip_data", return_value={
             "total": 2,
             "count_success": 1,
             "count_skipped_latest": 1,
             "last_date_check_error_count": 0,
             "download_error_count": 0,
             "issue_log_path": None,
         }):
        downloader_result = downloader_application.run_trading_dataset_update()
    add_check(results, "trading_daily", case_id, "downloader_application_returns_trading_domain", "trading", downloader_result.get("runtime_domain"))
    add_check(results, "trading_daily", case_id, "downloader_application_returns_market_date", "2026-09-04", downloader_result.get("market_date"))
    add_check(results, "trading_daily", case_id, "downloader_application_returns_ticker_count", 2, downloader_result.get("ticker_count"))

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
        data_dir = Path(paths.data_dir)
        data_dir.mkdir(parents=True)
        pd.DataFrame({
            "Date": ["2026-09-03", "2026-09-04"],
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000, 1100],
        }).to_csv(data_dir / "2330.csv", index=False)

        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_param_payload(latest_date: str, member_count: int = 1):
            members = [
                {"member_index": idx + 1, "seed": idx + 1, "params": params_to_json_dict(base_params)}
                for idx in range(member_count)
            ]
            payload = build_static_active_param_ensemble_payload(
                members=members,
                selector=profile.param_selector,
                meta={
                    "selected_model_mode": "trade",
                    "walk_forward_policy": {"latest_data_date": latest_date},
                },
            )
            selected_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        _write_param_payload("2026-09-04", 1)
        snapshot = daily_workflow.build_trading_daily_workflow_snapshot(root)
        add_check(results, "trading_daily", case_id, "workflow_snapshot_reads_latest_trading_data", "2026-09-04", snapshot.get("latest_data_date"))
        add_check(results, "trading_daily", case_id, "workflow_snapshot_reads_param_latest_date", "2026-09-04", snapshot.get("param_latest_data_date"))
        add_check(results, "trading_daily", case_id, "workflow_snapshot_requires_single_member", 1, snapshot.get("param_member_count"))
        add_check(results, "trading_daily", case_id, "workflow_snapshot_ready_when_data_and_params_match", True, snapshot.get("params_ready_for_scan"))
        add_check(results, "trading_daily", case_id, "workflow_scanner_output_is_trading_scoped", "outputs/trading/scanner", snapshot.get("scanner_output_dir"))

        fake_scan = {
            "count_scanned": 1,
            "elapsed_time": 0.01,
            "count_history_qualified": 1,
            "count_skipped_insufficient": 0,
            "count_sanitized_candidates": 0,
            "max_workers": 1,
            "pool_start_method": "spawn",
            "candidate_rows": [{
                "ticker": "2330",
                "kind": "buy",
                "sort_value": 1.5,
                "expected_value": 0.3,
                "proj_cost": 100000,
                "text": "synthetic candidate",
                "execution_plan_seed": {"ticker": "2330", "limit_price": 102.0, "init_sl": 98.0, "init_trail": 99.0, "target_price": 106.0, "entry_atr": 2.0, "trade_date": "2026-09-04"},
            }],
            "scanner_issue_log_path": None,
        }
        with patch.object(daily_workflow, "run_daily_scanner", return_value=fake_scan) as scanner_mock:
            scan_result = daily_workflow.run_trading_candidate_scan(project_root=root)
        scanner_args = scanner_mock.call_args
        add_check(results, "trading_daily", case_id, "trading_scanner_uses_trading_data_dir", str(data_dir), str(scanner_args.args[0]))
        add_check(results, "trading_daily", case_id, "trading_scanner_injects_trading_output_dir", str((root / "outputs" / "trading" / "scanner").resolve()), str(Path(scanner_args.kwargs["output_dir"]).resolve()))
        add_check(results, "trading_daily", case_id, "trading_scanner_requests_execution_context_for_allocator", True, scanner_args.kwargs.get("include_execution_context"))
        add_check(results, "trading_daily", case_id, "trading_scanner_returns_candidate_rows", 1, len(scan_result.get("candidate_rows") or []))
        add_check(results, "trading_daily", case_id, "trading_scanner_persists_candidate_snapshot", True, (root / "outputs" / "trading" / "scanner" / "candidate_snapshot.json").is_file())
        add_check(results, "trading_daily", case_id, "trading_scanner_carries_matching_data_date", "2026-09-04", scan_result.get("latest_data_date"))

        _write_param_payload("2026-09-03", 1)
        try:
            daily_workflow.run_trading_candidate_scan(project_root=root)
        except RuntimeError as exc:
            stale_rejected = "不是目前最新Trading資料" in str(exc)
        else:
            stale_rejected = False
        add_check(results, "trading_daily", case_id, "stale_trading_params_are_rejected_before_scan", True, stale_rejected)

        _write_param_payload("2026-09-04", 2)
        try:
            daily_workflow.run_trading_candidate_scan(project_root=root)
        except RuntimeError as exc:
            multi_member_rejected = "禁止靜默只取第一組" in str(exc)
        else:
            multi_member_rejected = False
        add_check(results, "trading_daily", case_id, "multi_member_selector_is_failfast_until_scanner_has_ensemble_semantics", True, multi_member_rejected)

    call_order = []
    with patch.object(daily_workflow, "run_trading_market_data_update", side_effect=lambda **_kwargs: call_order.append("data") or {"status": "READY"}), \
         patch("services.trading.position_rollforward.run_trading_position_rollforward", side_effect=lambda **_kwargs: call_order.append("rollforward") or {"status": "UP_TO_DATE"}), \
         patch("services.trading.indicator_exit_planning.build_trading_indicator_exit_plan", side_effect=lambda **_kwargs: call_order.append("indicator") or {"status": "PROPOSED_INDICATOR_EXIT", "exit_count": 0, "exits": []}), \
         patch.object(daily_workflow, "run_trading_strategy_param_training", side_effect=lambda **_kwargs: call_order.append("params") or {"status": "READY"}), \
         patch.object(daily_workflow, "run_trading_candidate_scan", side_effect=lambda **_kwargs: call_order.append("scanner") or {"status": "READY", "candidate_rows": []}):
        workflow_result = daily_workflow.run_trading_daily_workflow(project_root=project_root, environ={})
    add_check(results, "trading_daily", case_id, "daily_workflow_executes_data_rollforward_indicator_params_scanner_in_order", ["data", "rollforward", "indicator", "params", "scanner"], call_order)
    add_check(results, "trading_daily", case_id, "daily_workflow_returns_ready_only_after_all_steps", "READY", workflow_result.get("status"))

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    add_check(results, "trading_daily", case_id, "workbench_exposes_separate_data_button", True, '"1 更新資料"' in panel_source)
    add_check(results, "trading_daily", case_id, "workbench_exposes_separate_param_button", True, '"2 更新 Params"' in panel_source)
    add_check(results, "trading_daily", case_id, "workbench_exposes_scanner_button", True, '"3 Scanner 候選"' in panel_source)
    add_check(results, "trading_daily", case_id, "workbench_exposes_one_click_daily_sequence", True, '"每日流程 1→2→3"' in panel_source)
    add_check(results, "trading_daily", case_id, "workbench_long_workflow_uses_background_thread", True, "threading.Thread(" in panel_source)
    add_check(results, "trading_daily", case_id, "workbench_keeps_scanner_candidates_separate_from_proposed_orders", True, "今日 Scanner 候選" in panel_source and "建議掛單（尚未送單／尚未成交）" in panel_source)

    summary["checks"] = len(results)
    return results, summary

def validate_trading_proposed_order_plan_contract_case(base_params):
    case_id = "TRADING_PROPOSED_ORDERS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.exact_accounting import build_sell_ledger_from_price
    from core.file_integrity import compute_file_sha256
    from core.params_io import params_to_json_dict
    from core.price_utils import adjust_long_sell_fill_price
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    from services.trading.daily_workflow import resolve_trading_candidate_snapshot_path
    from services.trading.order_planning import build_trading_proposed_order_plan

    profile = get_trading_strategy_profile()
    project_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
        data_dir = Path(paths.data_dir)
        data_dir.mkdir(parents=True)
        for ticker, close in (("2330", 100.0), ("2454", 200.0), ("2603", 50.0)):
            pd.DataFrame({
                "Date": ["2026-09-03", "2026-09-04"],
                "Open": [close, close],
                "High": [close + 1, close + 1],
                "Low": [close - 1, close - 1],
                "Close": [close, close],
                "Volume": [1000, 1000],
            }).to_csv(data_dir / f"{ticker}.csv", index=False)

        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_payload = build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(base_params)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        )
        selected_path.write_text(json.dumps(selected_payload, ensure_ascii=False), encoding="utf-8")

        state = initialize_trading_account_state(root, cash=800_000)
        state = adopt_existing_trading_position(
            root,
            ticker="2330",
            qty=1000,
            cost_basis_total=100_000,
            entry_date="2026-01-01",
            expected_revision=state["revision"],
        )
        starting_revision = int(state["revision"])
        starting_cash_milli = int(state["cash_milli"])

        candidate_snapshot_path = resolve_trading_candidate_snapshot_path(root)
        candidate_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_rows = [
            {
                "ticker": "2330",
                "kind": "extended_tbd",
                "sort_value": 3.0,
                "expected_value": 0.5,
                "execution_plan_seed": {"ticker": "2330", "limit_price": 101.0, "init_sl": 95.0, "init_trail": 96.0, "target_price": 107.0, "entry_atr": 2.0, "trade_date": "2026-09-04"},
            },
            {
                "ticker": "2454",
                "kind": "buy",
                "sort_value": 2.0,
                "expected_value": 0.4,
                "execution_plan_seed": {"ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0, "target_price": 210.0, "entry_atr": 4.0, "trade_date": "2026-09-04"},
            },
            {
                "ticker": "2603",
                "kind": "buy",
                "sort_value": 1.0,
                "expected_value": 0.3,
                "execution_plan_seed": {"ticker": "2603", "limit_price": 50.0, "init_sl": 47.0, "init_trail": 48.0, "target_price": 53.0, "entry_atr": 1.0, "trade_date": "2026-09-04"},
            },
        ]
        candidate_snapshot_path.write_text(json.dumps({
            "schema_version": 1,
            "runtime_domain": "trading",
            "strategy_id": profile.strategy_id,
            "param_selector": profile.param_selector,
            "latest_data_date": "2026-09-04",
            "param_latest_data_date": "2026-09-04",
            "selected_params_sha256": compute_file_sha256(selected_path),
            "candidate_rows": candidate_rows,
        }, ensure_ascii=False), encoding="utf-8")

        plan = build_trading_proposed_order_plan(project_root=root)
        add_check(results, "trading_orders", case_id, "proposed_plan_does_not_mutate_account_revision", starting_revision, load_trading_account_state(root)["revision"])
        add_check(results, "trading_orders", case_id, "proposed_plan_does_not_mutate_account_cash", starting_cash_milli, load_trading_account_state(root)["cash_milli"])
        add_check(results, "trading_orders", case_id, "held_ticker_is_not_proposed_again", ["2330"], plan.get("held_tickers_skipped"))
        add_check(results, "trading_orders", case_id, "extended_tbd_is_resolved_against_actual_holdings", False, any(row.get("ticker") == "2330" for row in plan.get("orders") or []))
        add_check(results, "trading_orders", case_id, "proposed_orders_are_not_confirmed", False, plan.get("confirmed"))
        add_check(results, "trading_orders", case_id, "proposed_orders_do_not_claim_account_mutation", False, plan.get("account_mutated"))
        add_check(results, "trading_orders", case_id, "reserved_total_equals_order_sum", int(plan["reserved_total_milli"]), sum(int(row["reserved_cost_milli"]) for row in plan["orders"]))
        add_check(results, "trading_orders", case_id, "reservation_never_exceeds_cash", True, int(plan["reserved_total_milli"]) <= starting_cash_milli)
        add_check(results, "trading_orders", case_id, "reserved_cash_is_locked_across_proposed_orders", starting_cash_milli - int(plan["reserved_total_milli"]), int(plan["cash_after_reservation_milli"]))
        expected_liquidation = build_sell_ledger_from_price(adjust_long_sell_fill_price(100.0, ticker="2330"), 1000, base_params, ticker="2330", trade_date="2026-09-04")["net_sell_total_milli"]
        add_check(results, "trading_orders", case_id, "sizing_equity_uses_mark_to_market_net_liquidation", starting_cash_milli + expected_liquidation, int(round(float(plan["sizing_equity"]) * 1000)))
        add_check(results, "trading_orders", case_id, "proposed_plan_writes_machine_readable_output", True, (root / "outputs" / "trading" / "proposed_orders" / "proposed_orders.json").is_file())
        add_check(results, "trading_orders", case_id, "proposed_plan_writes_human_readable_output", True, (root / "outputs" / "trading" / "proposed_orders" / "proposed_orders.txt").is_file())
        add_check(results, "trading_orders", case_id, "proposed_plan_carries_account_revision", starting_revision, plan.get("account_revision"))
        add_check(results, "trading_orders", case_id, "proposed_plan_carries_candidate_snapshot_identity", compute_file_sha256(candidate_snapshot_path), plan.get("candidate_snapshot_sha256"))

        selected_path.write_text(selected_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        try:
            build_trading_proposed_order_plan(project_root=root)
        except RuntimeError as exc:
            stale_snapshot_rejected = "params 已改變" in str(exc)
        else:
            stale_snapshot_rejected = False
        add_check(results, "trading_orders", case_id, "candidate_snapshot_is_rejected_after_param_artifact_changes", True, stale_snapshot_rejected)

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    add_check(results, "trading_orders", case_id, "workbench_exposes_proposed_order_button", True, '"4 建議掛單"' in panel_source)
    add_check(results, "trading_orders", case_id, "workbench_proposed_orders_use_background_thread", True, 'elif action == "orders"' in panel_source and "threading.Thread(" in panel_source)
    add_check(results, "trading_orders", case_id, "workbench_does_not_confirm_fill_from_proposal_action", False, "confirm_trading_strategy_buy_fill" in panel_source)

    summary["checks"] = len(results)
    return results, summary



def validate_trading_pending_order_state_contract_case(base_params):
    case_id = "TRADING_PENDING_ORDERS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.file_integrity import compute_file_sha256
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_order_state import validate_trading_order_state
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    from services.trading.daily_workflow import resolve_trading_candidate_snapshot_path
    from services.trading.order_planning import build_trading_proposed_order_plan
    from services.trading.order_state import (
        TradingOrderRevisionConflict,
        confirm_trading_order_cancellation,
        confirm_trading_order_submission,
        get_trading_order_read_model,
        load_trading_order_state,
        resolve_trading_order_state_path,
    )

    profile = get_trading_strategy_profile()
    project_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
        data_dir = Path(paths.data_dir)
        data_dir.mkdir(parents=True)
        for ticker, close in (("2454", 200.0), ("2603", 50.0)):
            pd.DataFrame({
                "Date": ["2026-09-03", "2026-09-04"],
                "Open": [close, close],
                "High": [close + 1, close + 1],
                "Low": [close - 1, close - 1],
                "Close": [close, close],
                "Volume": [1000, 1000],
            }).to_csv(data_dir / f"{ticker}.csv", index=False)

        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(base_params)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")

        account = initialize_trading_account_state(root, cash=800_000)
        account_revision = int(account["revision"])
        account_cash_milli = int(account["cash_milli"])

        snapshot_path = resolve_trading_candidate_snapshot_path(root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_rows = [
            {
                "ticker": "2454",
                "kind": "buy",
                "sort_value": 2.0,
                "expected_value": 0.4,
                "execution_plan_seed": {
                    "ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0,
                    "target_price": 210.0, "entry_atr": 4.0, "trade_date": "2026-09-04",
                    "security_profile": {"family": "stock"},
                },
            },
            {
                "ticker": "2603",
                "kind": "buy",
                "sort_value": 1.0,
                "expected_value": 0.3,
                "execution_plan_seed": {
                    "ticker": "2603", "limit_price": 50.0, "init_sl": 47.0, "init_trail": 48.0,
                    "target_price": 53.0, "entry_atr": 1.0, "trade_date": "2026-09-04",
                    "security_profile": {"family": "stock"},
                },
            },
        ]
        snapshot_path.write_text(json.dumps({
            "schema_version": 1,
            "runtime_domain": "trading",
            "strategy_id": profile.strategy_id,
            "param_selector": profile.param_selector,
            "latest_data_date": "2026-09-04",
            "param_latest_data_date": "2026-09-04",
            "selected_params_sha256": compute_file_sha256(selected_path),
            "candidate_rows": candidate_rows,
        }, ensure_ascii=False), encoding="utf-8")

        plan = build_trading_proposed_order_plan(project_root=root)
        order_path = resolve_trading_order_state_path(root)
        add_check(results, "trading_pending", case_id, "proposal_only_does_not_create_order_state", False, order_path.exists())
        add_check(results, "trading_pending", case_id, "order_state_path_is_operational_trading_state", str((root / "state" / "trading" / "orders.json").resolve()), str(order_path.resolve()))
        add_check(results, "trading_pending", case_id, "proposed_schema_preserves_entry_type_for_future_fill", True, all(bool(row.get("entry_type")) for row in plan["orders"]))
        add_check(results, "trading_pending", case_id, "proposed_schema_preserves_security_profile_for_future_fill", True, all(isinstance(row.get("security_profile"), dict) for row in plan["orders"]))

        first = plan["orders"][0]
        ordered = confirm_trading_order_submission(
            root,
            rank=int(first["rank"]),
            ticker=first["ticker"],
            expected_revision=None,
            broker_order_id="SYN-001",
            note="synthetic broker submission",
        )
        validate_trading_order_state(ordered)
        ordered_record = next(iter(ordered["orders"].values()))
        add_check(results, "trading_pending", case_id, "first_confirmed_submission_creates_revision_one", 1, ordered["revision"])
        add_check(results, "trading_pending", case_id, "confirmed_submission_status_is_ordered", "ORDERED", ordered_record["status"])
        add_check(results, "trading_pending", case_id, "confirmed_submission_records_proposed_to_ordered_transition", ["PROPOSED", "ORDERED"], [ordered["events"][-1]["details"]["from_status"], ordered["events"][-1]["details"]["to_status"]])
        add_check(results, "trading_pending", case_id, "broker_order_id_is_preserved", "SYN-001", ordered_record["broker_order_id"])
        add_check(results, "trading_pending", case_id, "order_freezes_plan_fingerprint", plan["plan_fingerprint"], ordered_record["plan_fingerprint"])
        add_check(results, "trading_pending", case_id, "order_freezes_account_revision", account_revision, ordered_record["account_revision"])
        add_check(results, "trading_pending", case_id, "order_freezes_param_sha", plan["selected_params_sha256"], ordered_record["selected_params_sha256"])
        add_check(results, "trading_pending", case_id, "order_freezes_candidate_sha", plan["candidate_snapshot_sha256"], ordered_record["candidate_snapshot_sha256"])
        add_check(results, "trading_pending", case_id, "order_submission_does_not_mutate_account_revision", account_revision, load_trading_account_state(root)["revision"])
        add_check(results, "trading_pending", case_id, "order_submission_does_not_mutate_account_cash", account_cash_milli, load_trading_account_state(root)["cash_milli"])

        try:
            confirm_trading_order_submission(
                root,
                rank=int(first["rank"]),
                ticker=first["ticker"],
                expected_revision=ordered["revision"],
            )
        except ValueError:
            duplicate_rejected = True
        else:
            duplicate_rejected = False
        add_check(results, "trading_pending", case_id, "same_proposal_cannot_be_marked_ordered_twice", True, duplicate_rejected)
        add_check(results, "trading_pending", case_id, "duplicate_rejection_does_not_advance_order_revision", ordered["revision"], load_trading_order_state(root)["revision"])

        try:
            set_trading_cash_balance(root, cash=700_000, expected_revision=account_revision)
        except RuntimeError as exc:
            account_locked = "ORDERED pending orders" in str(exc)
        else:
            account_locked = False
        add_check(results, "trading_pending", case_id, "active_order_blocks_account_reconciliation", True, account_locked)

        try:
            build_trading_proposed_order_plan(project_root=root)
        except RuntimeError as exc:
            replanning_blocked = "ORDERED pending orders" in str(exc)
        else:
            replanning_blocked = False
        add_check(results, "trading_pending", case_id, "active_order_blocks_new_premarket_allocation", True, replanning_blocked)

        order_id = ordered_record["order_id"]
        try:
            confirm_trading_order_cancellation(root, order_id=order_id, expected_revision=0)
        except TradingOrderRevisionConflict:
            stale_order_revision_rejected = True
        else:
            stale_order_revision_rejected = False
        add_check(results, "trading_pending", case_id, "stale_order_revision_is_rejected", True, stale_order_revision_rejected)

        cancelled = confirm_trading_order_cancellation(
            root,
            order_id=order_id,
            expected_revision=ordered["revision"],
            note="synthetic broker cancelled",
        )
        cancelled_record = cancelled["orders"][order_id]
        add_check(results, "trading_pending", case_id, "cancel_transition_advances_one_revision", ordered["revision"] + 1, cancelled["revision"])
        add_check(results, "trading_pending", case_id, "cancelled_order_status", "CANCELLED", cancelled_record["status"])
        add_check(results, "trading_pending", case_id, "cancelled_order_has_timestamp", True, bool(cancelled_record["cancelled_at"]))
        add_check(results, "trading_pending", case_id, "cancellation_does_not_mutate_account", [account_revision, account_cash_milli], [load_trading_account_state(root)["revision"], load_trading_account_state(root)["cash_milli"]])

        try:
            build_trading_proposed_order_plan(project_root=root)
        except RuntimeError as exc:
            same_day_reallocation_blocked = "同日重新 allocation" in str(exc)
        else:
            same_day_reallocation_blocked = False
        add_check(results, "trading_pending", case_id, "cancelled_order_still_locks_same_information_date_allocation", True, same_day_reallocation_blocked)

        reconciled = set_trading_cash_balance(root, cash=700_000, expected_revision=account_revision)
        add_check(results, "trading_pending", case_id, "account_mutation_is_reenabled_after_all_orders_cancelled", account_revision + 1, reconciled["revision"])

        try:
            confirm_trading_order_cancellation(root, order_id=order_id, expected_revision=cancelled["revision"])
        except ValueError:
            double_cancel_rejected = True
        else:
            double_cancel_rejected = False
        add_check(results, "trading_pending", case_id, "cancelled_order_cannot_be_cancelled_twice", True, double_cancel_rejected)

        tampered = load_trading_order_state(root)
        tampered["events"][-1]["details"]["ticker"] = "TAMPER"
        order_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        try:
            load_trading_order_state(root)
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        add_check(results, "trading_pending", case_id, "order_event_hash_tamper_is_rejected", True, tamper_rejected)

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    order_service_source = (project_root / "services" / "trading" / "order_state.py").read_text(encoding="utf-8")
    add_check(results, "trading_pending", case_id, "workbench_requires_explicit_order_submission_confirmation", True, 'text="確認選取已送單"' in panel_source and "confirm_trading_order_submission(" in panel_source)
    add_check(results, "trading_pending", case_id, "workbench_exposes_explicit_broker_cancellation_confirmation", True, 'text="確認選取剩餘委託已取消"' in panel_source and "confirm_trading_order_cancellation(" in panel_source)
    add_check(results, "trading_pending", case_id, "submission_cancel_service_does_not_own_fill_reconciliation", False, "confirm_trading_buy_order_fill(" in order_service_source or "apply_confirmed_strategy_buy_fill(" in order_service_source)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_confirmed_fill_reconciliation_contract_case(base_params):
    case_id = "TRADING_CONFIRMED_FILLS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_order_state import TRADING_ACTIVE_ORDER_STATUSES
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    from services.trading.daily_workflow import resolve_trading_candidate_snapshot_path
    from services.trading.fill_reconciliation import (
        TradingFillRevisionConflict,
        confirm_trading_buy_order_fill,
        recover_trading_fill_transaction,
        resolve_trading_fill_transaction_path,
    )
    from services.trading.order_planning import build_trading_proposed_order_plan
    from services.trading.order_state import (
        confirm_trading_order_cancellation,
        confirm_trading_order_submission,
        load_trading_order_state,
    )

    profile = get_trading_strategy_profile()
    project_root = Path(__file__).resolve().parents[2]

    def prepare(root: Path):
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
        data_dir = Path(paths.data_dir)
        data_dir.mkdir(parents=True)
        pd.DataFrame({
            "Date": ["2026-09-03", "2026-09-04"],
            "Open": [200.0, 200.0], "High": [201.0, 201.0], "Low": [199.0, 199.0],
            "Close": [200.0, 200.0], "Volume": [1000, 1000],
        }).to_csv(data_dir / "2454.csv", index=False)
        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(base_params)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")
        account = initialize_trading_account_state(root, cash=800_000)
        snapshot_path = resolve_trading_candidate_snapshot_path(root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps({
            "schema_version": 1,
            "runtime_domain": "trading",
            "strategy_id": profile.strategy_id,
            "param_selector": profile.param_selector,
            "latest_data_date": "2026-09-04",
            "param_latest_data_date": "2026-09-04",
            "selected_params_sha256": compute_file_sha256(selected_path),
            "candidate_rows": [{
                "ticker": "2454", "kind": "buy", "sort_value": 2.0, "expected_value": 0.4,
                "execution_plan_seed": {
                    "ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0,
                    "target_price": 210.0, "entry_atr": 4.0, "trade_date": "2026-09-04",
                    "security_profile": {"family": "stock"},
                },
            }],
        }, ensure_ascii=False), encoding="utf-8")
        plan = build_trading_proposed_order_plan(project_root=root)
        proposal = plan["orders"][0]
        ordered = confirm_trading_order_submission(
            root,
            rank=int(proposal["rank"]),
            ticker=proposal["ticker"],
            expected_revision=None,
            broker_order_id="FILL-SYN-001",
        )
        order_id = next(iter(ordered["orders"]))
        return selected_path, account, plan, proposal, ordered, order_id

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        selected_path, account, plan, proposal, ordered, order_id = prepare(root)
        ordered_record = ordered["orders"][order_id]
        add_check(results, "trading_fill", case_id, "ordered_freezes_param_payload", params_to_json_dict(base_params), ordered_record.get("frozen_params"))
        add_check(results, "trading_fill", case_id, "ordered_freezes_param_payload_hash", canonical_json_sha256(params_to_json_dict(base_params)), ordered_record.get("frozen_params_sha256"))

        changed = deepcopy(base_params)
        changed.high_len = int(getattr(changed, "high_len")) + 5
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(changed)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")

        order_qty = int(ordered_record["qty"])
        first_qty = max(1, order_qty // 2)
        initial_cash = int(account["cash_milli"])
        first_ledger = build_buy_ledger_from_price(199.0, first_qty, base_params)
        tx_path = resolve_trading_fill_transaction_path(root)
        from core.file_integrity import atomic_write_json as real_atomic_write_json
        write_counter = {"count": 0}

        def crash_after_account_write(path, payload):
            write_counter["count"] += 1
            if write_counter["count"] == 3:
                raise RuntimeError("synthetic crash between account/order commit")
            return real_atomic_write_json(path, payload)

        with patch("services.trading.fill_reconciliation.atomic_write_json", side_effect=crash_after_account_write):
            try:
                confirm_trading_buy_order_fill(
                    root,
                    order_id=order_id,
                    fill_qty=first_qty,
                    fill_price=199.0,
                    trade_date="2026-09-05",
                    expected_order_revision=int(ordered["revision"]),
                    expected_account_revision=int(account["revision"]),
                )
            except RuntimeError as exc:
                interrupted = "synthetic crash" in str(exc)
            else:
                interrupted = False
        add_check(results, "trading_fill", case_id, "two_state_fill_commit_interruption_is_observable", True, interrupted)
        add_check(results, "trading_fill", case_id, "fill_transaction_journal_persists_after_interruption", True, tx_path.is_file())
        add_check(results, "trading_fill", case_id, "fill_transaction_recovery_completes_prepared_targets", True, recover_trading_fill_transaction(root))
        add_check(results, "trading_fill", case_id, "fill_transaction_journal_removed_after_recovery", False, tx_path.exists())

        partial_account = load_trading_account_state(root)
        partial_orders = load_trading_order_state(root)
        partial_record = partial_orders["orders"][order_id]
        add_check(results, "trading_fill", case_id, "first_fill_transitions_to_partial", "PARTIAL", partial_record["status"])
        add_check(results, "trading_fill", case_id, "partial_fill_updates_filled_qty", first_qty, partial_record["filled_qty"])
        add_check(results, "trading_fill", case_id, "partial_fill_keeps_remaining_qty_active", order_qty - first_qty, partial_record["remaining_qty"])
        add_check(results, "trading_fill", case_id, "partial_status_remains_active_order", True, partial_record["status"] in TRADING_ACTIVE_ORDER_STATUSES)
        add_check(results, "trading_fill", case_id, "partial_fill_debits_exact_cash", initial_cash - int(first_ledger["net_buy_total_milli"]), partial_account["cash_milli"])
        add_check(results, "trading_fill", case_id, "partial_fill_creates_actual_position", first_qty, partial_account["positions"]["2454"]["broker"]["qty"])
        add_check(results, "trading_fill", case_id, "partial_fill_binds_position_to_order", order_id, partial_account["positions"]["2454"]["broker"].get("entry_order_id"))
        add_check(results, "trading_fill", case_id, "fill_uses_frozen_params_after_current_artifact_changes", int(getattr(base_params, "high_len")), int(partial_record["frozen_params"]["high_len"]))

        before_order_rev = int(partial_orders["revision"])
        before_account_rev = int(partial_account["revision"])
        try:
            confirm_trading_buy_order_fill(
                root,
                order_id=order_id,
                fill_qty=1,
                fill_price=201.0,
                trade_date="2026-09-05",
                expected_order_revision=before_order_rev,
                expected_account_revision=before_account_rev,
            )
        except ValueError:
            above_limit_rejected = True
        else:
            above_limit_rejected = False
        add_check(results, "trading_fill", case_id, "broker_fill_above_buy_limit_is_rejected", True, above_limit_rejected)
        add_check(results, "trading_fill", case_id, "rejected_fill_does_not_advance_revisions", [before_account_rev, before_order_rev], [load_trading_account_state(root)["revision"], load_trading_order_state(root)["revision"]])

        try:
            confirm_trading_buy_order_fill(
                root,
                order_id=order_id,
                fill_qty=1,
                fill_price=198.5,
                trade_date="2026-09-06",
                expected_order_revision=before_order_rev,
                expected_account_revision=before_account_rev,
            )
        except ValueError:
            cross_day_partial_rejected = True
        else:
            cross_day_partial_rejected = False
        add_check(results, "trading_fill", case_id, "partial_fills_for_one_order_cannot_cross_trade_dates", True, cross_day_partial_rejected)

        remaining = int(partial_record["remaining_qty"])
        second_ledger = build_buy_ledger_from_price(198.0, remaining, base_params)
        final = confirm_trading_buy_order_fill(
            root,
            order_id=order_id,
            fill_qty=remaining,
            fill_price=198.0,
            trade_date="2026-09-05",
            expected_order_revision=before_order_rev,
            expected_account_revision=before_account_rev,
        )
        add_check(results, "trading_fill", case_id, "final_fill_transitions_to_filled", "FILLED", final["status"])
        add_check(results, "trading_fill", case_id, "filled_order_has_zero_remaining_qty", 0, final["remaining_qty"])
        add_check(results, "trading_fill", case_id, "all_partial_fills_accumulate_position_qty", order_qty, final["account"]["positions"]["2454"]["broker"]["qty"])
        add_check(results, "trading_fill", case_id, "all_partial_fills_debit_sum_of_exact_ledgers", initial_cash - int(first_ledger["net_buy_total_milli"]) - int(second_ledger["net_buy_total_milli"]), final["account"]["cash_milli"])
        add_check(results, "trading_fill", case_id, "filled_order_is_no_longer_active", False, final["orders"]["orders"][order_id]["status"] in TRADING_ACTIVE_ORDER_STATUSES)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _selected_path, account, _plan, _proposal, ordered, order_id = prepare(root)
        record = ordered["orders"][order_id]
        order_qty = int(record["qty"])
        first_qty = max(1, order_qty // 3)
        partial = confirm_trading_buy_order_fill(
            root,
            order_id=order_id,
            fill_qty=first_qty,
            fill_price=199.0,
            trade_date="2026-09-05",
            expected_order_revision=int(ordered["revision"]),
            expected_account_revision=int(account["revision"]),
        )
        partial_cash = int(partial["account"]["cash_milli"])
        cancelled = confirm_trading_order_cancellation(
            root,
            order_id=order_id,
            expected_revision=int(partial["order_revision"]),
            note="cancel remaining synthetic qty",
        )
        cancelled_record = cancelled["orders"][order_id]
        add_check(results, "trading_fill", case_id, "partial_order_can_cancel_unfilled_remainder", "CANCELLED", cancelled_record["status"])
        add_check(results, "trading_fill", case_id, "partial_cancel_preserves_filled_qty", first_qty, cancelled_record["filled_qty"])
        add_check(results, "trading_fill", case_id, "partial_cancel_preserves_position_and_cash", [first_qty, partial_cash], [load_trading_account_state(root)["positions"]["2454"]["broker"]["qty"], load_trading_account_state(root)["cash_milli"]])

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    add_check(results, "trading_fill", case_id, "workbench_requires_explicit_broker_fill_confirmation", True, 'text="確認選取成交"' in panel_source and "confirm_trading_buy_order_fill" in panel_source and "confirm_trading_protection_sell_order_fill" in panel_source)
    add_check(results, "trading_fill", case_id, "workbench_fill_inputs_are_actual_qty_price_and_date", True, all(token in panel_source for token in ("本次成交股數", "本次成交價", "成交日 YYYY-MM-DD")))
    add_check(results, "trading_fill", case_id, "workbench_does_not_auto_infer_fill_from_market_bar", False, "t_low" in panel_source or "t_high" in panel_source or "execute_pre_market_entry_plan" in panel_source)

    summary["checks"] = len(results)
    return results, summary

def validate_trading_protection_plan_contract_case(base_params):
    case_id = "TRADING_PROTECTION_PLAN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.file_integrity import atomic_write_json, canonical_json_sha256
    from core.params_io import params_to_json_dict
    from core.price_utils import calc_half_take_profit_sell_qty
    from core.trading_order_state import (
        append_ordered_trading_proposal,
        build_empty_trading_order_state,
    )
    from services.trading.fill_reconciliation import confirm_trading_buy_order_fill
    from services.trading.order_state import (
        load_trading_order_state,
        resolve_trading_order_state_path,
    )
    from services.trading.protection_planning import (
        PROTECTION_BROKER_STATUS,
        PROTECTION_PLAN_STATUS,
        PROTECTION_SAME_BAR_PRIORITY,
        build_trading_protection_plan,
        get_trading_protection_plan_read_model,
        resolve_trading_protection_plan_json_path,
        resolve_trading_protection_plan_text_path,
        validate_trading_protection_plan,
    )

    project_root = Path(__file__).resolve().parents[2]
    frozen_params = params_to_json_dict(base_params)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        account = adopt_existing_trading_position(
            root,
            ticker="2330",
            qty=100,
            cost_basis_total=50_000,
            entry_date="2026-08-01",
            expected_revision=account["revision"],
        )
        account_revision_before_fill = int(account["revision"])

        qty = 100
        limit_price = 100.0
        reserved = build_buy_ledger_from_price(limit_price, qty, base_params)["net_buy_total_milli"]
        order_state = build_empty_trading_order_state(
            timestamp="2026-09-04T09:00:00+08:00",
            mutation_id="synthetic-init-order-state",
        )
        plan_seed = {
            "plan_fingerprint": "synthetic-protection-plan",
            "information_date": "2026-09-04",
            "account_revision": account_revision_before_fill,
            "selected_params_sha256": canonical_json_sha256(frozen_params),
            "candidate_snapshot_sha256": "synthetic-candidate-sha",
            "strategy_id": "full_rule_based_no_dl",
            "param_selector": "base_finalist_best",
        }
        proposal = {
            "rank": 1,
            "ticker": "2454",
            "kind": "buy",
            "entry_type": "normal",
            "qty": qty,
            "limit_price": limit_price,
            "reserved_cost_milli": int(reserved),
            "init_sl": 90.0,
            "init_trail": 90.0,
            "target_price": 110.0,
            "entry_atr": 5.0,
            "security_profile": {},
        }
        order_state = append_ordered_trading_proposal(
            order_state,
            order_id="synthetic-entry-order",
            proposal=proposal,
            plan=plan_seed,
            timestamp="2026-09-04T09:01:00+08:00",
            mutation_id="synthetic-submit-order",
            frozen_params=frozen_params,
        )
        atomic_write_json(resolve_trading_order_state_path(root), order_state)

        first_fill_qty = 40
        partial = confirm_trading_buy_order_fill(
            root,
            order_id="synthetic-entry-order",
            fill_qty=first_fill_qty,
            fill_price=99.0,
            trade_date="2026-09-04",
            expected_order_revision=int(order_state["revision"]),
            expected_account_revision=account_revision_before_fill,
        )
        account_after_partial = partial["account"]
        orders_after_partial = partial["orders"]
        canonical_position = account_after_partial["positions"]["2454"]["strategy_management"]["position_state"]
        account_revision = int(account_after_partial["revision"])
        order_revision = int(orders_after_partial["revision"])

        protection = build_trading_protection_plan(root)
        validate_trading_protection_plan(protection)
        row = protection["positions"][0]
        stop_leg = next(leg for leg in row["legs"] if leg["action"] == "STOP_FULL")
        tp_leg = next(leg for leg in row["legs"] if leg["action"] == "TP_HALF")
        expected_tp_qty = calc_half_take_profit_sell_qty(first_fill_qty, base_params.tp_percent)

        add_check(results, "trading_protection", case_id, "protection_plan_status_is_proposed_only", PROTECTION_PLAN_STATUS, protection["status"])
        add_check(results, "trading_protection", case_id, "protection_plan_never_claims_broker_submission", PROTECTION_BROKER_STATUS, protection["broker_status"])
        add_check(results, "trading_protection", case_id, "protection_plan_does_not_mutate_account_revision", account_revision, load_trading_account_state(root)["revision"])
        add_check(results, "trading_protection", case_id, "protection_plan_does_not_mutate_order_revision", order_revision, load_trading_order_state(root)["revision"])
        add_check(results, "trading_protection", case_id, "manual_adopted_position_is_not_auto_managed", ["2330"], protection["manual_positions_skipped"])
        add_check(results, "trading_protection", case_id, "partial_fill_protects_only_confirmed_held_qty", first_fill_qty, row["position_qty"])
        add_check(results, "trading_protection", case_id, "stop_leg_covers_full_current_position", first_fill_qty, stop_leg["qty"])
        add_check(results, "trading_protection", case_id, "stop_leg_uses_canonical_effective_position_stop", int(canonical_position["sl_milli"]), int(stop_leg["trigger_price_milli"]))
        add_check(results, "trading_protection", case_id, "stop_leg_uses_stop_market_gap_semantics", "STOP_MARKET", stop_leg["order_type"])
        add_check(results, "trading_protection", case_id, "tp_leg_qty_uses_canonical_half_take_profit_rule", expected_tp_qty, tp_leg["qty"])
        add_check(results, "trading_protection", case_id, "tp_leg_uses_canonical_position_target", int(canonical_position["tp_half_milli"]), int(tp_leg["limit_price_milli"]))
        add_check(results, "trading_protection", case_id, "same_bar_stop_has_priority_over_tp", PROTECTION_SAME_BAR_PRIORITY, row["same_bar_priority"])
        add_check(results, "trading_protection", case_id, "protection_plan_uses_no_post_fill_market_data", False, protection["market_data_used"])
        add_check(results, "trading_protection", case_id, "protection_plan_uses_no_discretionary_input", False, protection["discretionary_input_used"])
        add_check(results, "trading_protection", case_id, "protection_json_is_output_not_operational_state", str((root / "outputs" / "trading" / "protection_orders" / "protection_plan.json").resolve()), str(resolve_trading_protection_plan_json_path(root).resolve()))
        add_check(results, "trading_protection", case_id, "protection_human_summary_is_persisted", True, resolve_trading_protection_plan_text_path(root).is_file())
        first_read = get_trading_protection_plan_read_model(root)
        add_check(results, "trading_protection", case_id, "newly_built_protection_plan_is_fresh", True, first_read["fresh"])

        remaining = int(partial["remaining_qty"])
        final = confirm_trading_buy_order_fill(
            root,
            order_id="synthetic-entry-order",
            fill_qty=remaining,
            fill_price=98.0,
            trade_date="2026-09-04",
            expected_order_revision=int(partial["order_revision"]),
            expected_account_revision=int(partial["account_revision"]),
        )
        stale_read = get_trading_protection_plan_read_model(root)
        add_check(results, "trading_protection", case_id, "additional_partial_fill_invalidates_old_protection_source_binding", False, stale_read["fresh"])
        rebuilt = build_trading_protection_plan(root)
        rebuilt_row = rebuilt["positions"][0]
        rebuilt_position = final["account"]["positions"]["2454"]["strategy_management"]["position_state"]
        add_check(results, "trading_protection", case_id, "final_fill_rebuilds_stop_for_full_confirmed_position", qty, rebuilt_row["position_qty"])
        add_check(results, "trading_protection", case_id, "rebuilt_stop_follows_weighted_average_fill_position_state", int(rebuilt_position["sl_milli"]), int(rebuilt_row["effective_stop_milli"]))
        add_check(results, "trading_protection", case_id, "rebuilt_target_follows_weighted_average_fill_position_state", int(rebuilt_position["tp_half_milli"]), int(rebuilt_row["target_price_milli"]))
        add_check(results, "trading_protection", case_id, "source_change_produces_new_protection_fingerprint", True, rebuilt["plan_fingerprint"] != protection["plan_fingerprint"])
        add_check(results, "trading_protection", case_id, "rebuilt_plan_returns_to_fresh", True, get_trading_protection_plan_read_model(root)["fresh"])

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    service_source = (project_root / "services" / "trading" / "protection_planning.py").read_text(encoding="utf-8")
    add_check(results, "trading_protection", case_id, "workbench_exposes_post_fill_protection_plan", True, "成交後 Stop / TP 保護單計畫" in panel_source and "build_trading_protection_plan(" in panel_source)
    add_check(results, "trading_protection", case_id, "workbench_explicitly_marks_protection_as_not_submitted", True, "尚未送券商" in panel_source)
    add_check(results, "trading_protection", case_id, "protection_planner_does_not_own_broker_submission", False, "confirm_trading_order_submission(" in service_source or "append_ordered_trading_proposal(" in service_source)
    add_check(results, "trading_protection", case_id, "protection_planner_does_not_use_market_bar_fill_inference", False, any(token in service_source for token in ("t_high", "t_low", "execute_pre_market_entry_plan")))

    summary["checks"] = len(results)
    return results, summary



def validate_trading_protection_order_submission_contract_case(base_params):
    case_id = "TRADING_PROTECTION_ORDER_SUBMISSION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.exact_accounting import build_buy_ledger_from_price
    from core.file_integrity import atomic_write_json, canonical_json_sha256
    from core.params_io import params_to_json_dict
    from core.trading_order_state import (
        TRADING_ORDER_PURPOSE_PROTECTION_STOP,
        TRADING_ORDER_PURPOSE_PROTECTION_TP,
        TRADING_ORDER_SIDE_SELL,
        active_trading_entry_orders,
        active_trading_protection_orders,
        append_ordered_trading_proposal,
        build_empty_trading_order_state,
        record_trading_buy_order_fill,
        validate_trading_order_state,
    )
    from services.trading.fill_reconciliation import confirm_trading_buy_order_fill
    from services.trading.order_planning import _assert_order_state_allows_new_allocation
    from services.trading.order_state import (
        confirm_trading_order_cancellation,
        get_trading_order_read_model,
        load_trading_order_state,
        resolve_trading_order_state_path,
    )
    from services.trading.protection_order_submission import (
        confirm_trading_protection_leg_submission,
        confirm_trading_protection_oco_submission,
    )
    from services.trading.protection_planning import build_trading_protection_plan, get_trading_protection_plan_read_model

    project_root = Path(__file__).resolve().parents[2]
    frozen_params = params_to_json_dict(base_params)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        qty = 100
        reserved = build_buy_ledger_from_price(100.0, qty, base_params)["net_buy_total_milli"]
        state = build_empty_trading_order_state(timestamp="2026-09-04T09:00:00+08:00", mutation_id="r11-init")
        plan_seed = {
            "plan_fingerprint": "r11-entry-plan",
            "information_date": "2026-09-04",
            "account_revision": int(account["revision"]),
            "selected_params_sha256": canonical_json_sha256(frozen_params),
            "candidate_snapshot_sha256": "r11-candidate",
            "strategy_id": "full_rule_based_no_dl",
            "param_selector": "base_finalist_best",
        }
        proposal = {
            "rank": 1, "ticker": "2454", "kind": "buy", "entry_type": "normal", "qty": qty,
            "limit_price": 100.0, "reserved_cost_milli": int(reserved), "init_sl": 90.0,
            "init_trail": 90.0, "target_price": 110.0, "entry_atr": 5.0, "security_profile": {},
        }
        state = append_ordered_trading_proposal(
            state, order_id="r11-entry", proposal=proposal, plan=plan_seed,
            timestamp="2026-09-04T09:01:00+08:00", mutation_id="r11-submit", frozen_params=frozen_params,
        )
        atomic_write_json(resolve_trading_order_state_path(root), state)
        filled = confirm_trading_buy_order_fill(
            root, order_id="r11-entry", fill_qty=qty, fill_price=99.0, trade_date="2026-09-04",
            expected_order_revision=int(state["revision"]), expected_account_revision=int(account["revision"]),
        )
        account_revision = int(filled["account_revision"])
        protection = build_trading_protection_plan(root)
        add_check(results, "trading_protection_orders", case_id, "protection_plan_starts_fresh", True, get_trading_protection_plan_read_model(root)["fresh"])

        stop_only = confirm_trading_protection_leg_submission(
            root, ticker="2454", action="STOP_FULL", expected_order_revision=int(filled["order_revision"]),
            broker_order_id="R11-STOP-1",
        )
        stop_record = next(row for row in stop_only["orders"].values() if row.get("side") == TRADING_ORDER_SIDE_SELL)
        add_check(results, "trading_protection_orders", case_id, "single_stop_submission_is_sell_ordered", ["SELL", "PROTECTION_STOP", "ORDERED"], [stop_record["side"], stop_record["purpose"], stop_record["status"]])
        add_check(results, "trading_protection_orders", case_id, "single_stop_uses_canonical_stop_market", "STOP_MARKET", stop_record["order_type"])
        add_check(results, "trading_protection_orders", case_id, "single_stop_covers_full_held_qty", qty, stop_record["qty"])
        add_check(results, "trading_protection_orders", case_id, "protection_submission_does_not_mutate_account", account_revision, load_trading_account_state(root)["revision"])
        add_check(results, "trading_protection_orders", case_id, "broker_order_id_is_preserved_for_protection_sell", "R11-STOP-1", stop_record["broker_order_id"])
        add_check(results, "trading_protection_orders", case_id, "protection_plan_remains_fresh_after_sell_submission", True, get_trading_protection_plan_read_model(root)["fresh"])

        before_reject_revision = int(stop_only["revision"])
        try:
            confirm_trading_protection_leg_submission(
                root, ticker="2454", action="TP_HALF", expected_order_revision=before_reject_revision,
                broker_order_id="R11-TP-OVERCOMMIT",
            )
        except RuntimeError as exc:
            tp_overcommit_rejected = "超過實際持股" in str(exc)
        else:
            tp_overcommit_rejected = False
        add_check(results, "trading_protection_orders", case_id, "non_oco_stop_plus_tp_cannot_overcommit_position_qty", True, tp_overcommit_rejected)
        add_check(results, "trading_protection_orders", case_id, "rejected_overcommit_does_not_advance_order_revision", before_reject_revision, load_trading_order_state(root)["revision"])

        stop_order_id = str(stop_record["order_id"])
        cancelled = confirm_trading_order_cancellation(root, order_id=stop_order_id, expected_revision=before_reject_revision)
        try:
            confirm_trading_protection_oco_submission(
                root, ticker="2454", expected_order_revision=int(cancelled["revision"]), broker_oco_group_id="",
            )
        except ValueError:
            oco_requires_explicit_group = True
        else:
            oco_requires_explicit_group = False
        add_check(results, "trading_protection_orders", case_id, "oco_is_never_assumed_without_explicit_broker_group", True, oco_requires_explicit_group)

        oco = confirm_trading_protection_oco_submission(
            root, ticker="2454", expected_order_revision=int(cancelled["revision"]), broker_oco_group_id="R11-OCO-1",
            stop_broker_order_id="R11-STOP-2", tp_broker_order_id="R11-TP-2",
        )
        active_sell = active_trading_protection_orders(oco)
        purposes = {str(row.get("purpose")) for row in active_sell}
        add_check(results, "trading_protection_orders", case_id, "explicit_oco_creates_stop_and_tp_sell_orders", {TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_TP}, purposes)
        add_check(results, "trading_protection_orders", case_id, "oco_group_submission_is_single_revision_mutation", int(cancelled["revision"]) + 1, int(oco["revision"]))
        add_check(results, "trading_protection_orders", case_id, "all_oco_legs_preserve_user_confirmed_group", {"R11-OCO-1"}, {row.get("broker_oco_group_id") for row in active_sell})
        add_check(results, "trading_protection_orders", case_id, "all_oco_legs_require_explicit_native_oco_confirmation", {True}, {bool(row.get("broker_native_oco_confirmed")) for row in active_sell})
        add_check(results, "trading_protection_orders", case_id, "oco_effective_exposure_is_max_leg_not_sum", qty, max(int(row["qty"]) for row in active_sell))
        add_check(results, "trading_protection_orders", case_id, "protection_sell_does_not_become_entry_buy_pending", 0, len(active_trading_entry_orders(oco)))
        add_check(results, "trading_protection_orders", case_id, "two_oco_protection_legs_are_active", 2, len(active_sell))
        add_check(results, "trading_protection_orders", case_id, "oco_submission_still_does_not_mutate_account", account_revision, load_trading_account_state(root)["revision"])

        try:
            _assert_order_state_allows_new_allocation(root, information_date="2026-09-05")
        except RuntimeError:
            next_day_allocation_allowed = False
        else:
            next_day_allocation_allowed = True
        add_check(results, "trading_protection_orders", case_id, "long_lived_protection_sell_does_not_block_next_day_premarket_allocation", True, next_day_allocation_allowed)
        try:
            _assert_order_state_allows_new_allocation(root, information_date="2026-09-04")
        except RuntimeError as exc:
            same_day_buy_lock_preserved = "同日重新 allocation" in str(exc)
        else:
            same_day_buy_lock_preserved = False
        add_check(results, "trading_protection_orders", case_id, "historical_entry_buy_still_locks_same_information_date_allocation", True, same_day_buy_lock_preserved)

        read_model = get_trading_order_read_model(root)
        sell_rows = [row for row in read_model["orders"] if row.get("side") == "SELL"]
        add_check(results, "trading_protection_orders", case_id, "order_read_model_exposes_protection_sell_side_and_purpose", True, all(row.get("purpose") in {TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_TP} for row in sell_rows))
        add_check(results, "trading_protection_orders", case_id, "read_model_counts_active_protection_separately", 2, read_model["active_protection_order_count"])
        add_check(results, "trading_protection_orders", case_id, "read_model_has_no_active_entry_buy_after_entry_filled", 0, read_model["active_entry_order_count"])

        one_sell_id = next(row["order_id"] for row in active_sell)
        try:
            record_trading_buy_order_fill(
                oco, order_id=one_sell_id, fill_id="illegal-sell-as-buy", fill_qty=1, fill_price=90.0,
                trade_date="2026-09-04", net_buy_total_milli=1, timestamp="2026-09-04T10:00:00+08:00", mutation_id="illegal",
            )
        except ValueError:
            sell_fill_rejected_by_buy_path = True
        else:
            sell_fill_rejected_by_buy_path = False
        add_check(results, "trading_protection_orders", case_id, "protection_sell_cannot_use_buy_fill_reconciliation", True, sell_fill_rejected_by_buy_path)

        tampered = json.loads(json.dumps(oco))
        tamper_row = next(row for row in tampered["orders"].values() if row.get("side") == "SELL" and row.get("broker_native_oco_confirmed"))
        tamper_row["broker_oco_group_id"] = None
        try:
            validate_trading_order_state(tampered)
        except ValueError:
            oco_binding_tamper_rejected = True
        else:
            oco_binding_tamper_rejected = False
        add_check(results, "trading_protection_orders", case_id, "oco_confirmed_group_binding_tamper_is_rejected", True, oco_binding_tamper_rejected)

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    service_source = (project_root / "services" / "trading" / "protection_order_submission.py").read_text(encoding="utf-8")
    add_check(results, "trading_protection_orders", case_id, "workbench_requires_explicit_stop_or_tp_submission_confirmation", True, "確認 Stop 已送單" in panel_source and "確認 TP 已送單" in panel_source)
    add_check(results, "trading_protection_orders", case_id, "workbench_requires_explicit_broker_oco_group_confirmation", True, "確認 Stop+TP 已以券商 OCO 送單" in panel_source and "券商 OCO/互斥群組 ID" in panel_source)
    add_check(results, "trading_protection_orders", case_id, "protection_submission_service_does_not_infer_market_fill_or_sell_execution", False, any(token in service_source for token in ("t_high", "t_low", "apply_confirmed_sell_fill", "confirm_trading_sell_fill")))

    summary["checks"] = len(results)
    return results, summary


def validate_trading_protection_sell_fill_reconciliation_contract_case(base_params):
    case_id = "TRADING_PROTECTION_SELL_FILL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.file_integrity import atomic_write_json, canonical_json_sha256
    from core.params_io import params_to_json_dict
    from core.trading_order_state import (
        TRADING_ORDER_PURPOSE_PROTECTION_STOP,
        TRADING_ORDER_PURPOSE_PROTECTION_TP,
        build_empty_trading_order_state,
        append_ordered_trading_proposal,
        validate_trading_order_state,
    )
    from services.trading.fill_reconciliation import (
        confirm_trading_buy_order_fill,
        confirm_trading_protection_sell_order_fill,
    )
    from services.trading.order_state import load_trading_order_state, resolve_trading_order_state_path
    from services.trading.protection_planning import build_trading_protection_plan
    from services.trading.protection_order_submission import (
        confirm_trading_protection_leg_submission,
        confirm_trading_protection_oco_submission,
    )

    frozen_params = params_to_json_dict(base_params)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        qty = 100
        reserved = build_buy_ledger_from_price(100.0, qty, base_params)["net_buy_total_milli"]
        state = build_empty_trading_order_state(timestamp="2026-09-04T09:00:00+08:00", mutation_id="r12-init")
        plan = {
            "plan_fingerprint": "r12-entry-plan", "information_date": "2026-09-04",
            "account_revision": int(account["revision"]), "selected_params_sha256": canonical_json_sha256(frozen_params),
            "candidate_snapshot_sha256": "r12-candidate", "strategy_id": "full_rule_based_no_dl", "param_selector": "base_finalist_best",
        }
        proposal = {
            "rank": 1, "ticker": "2454", "kind": "buy", "entry_type": "normal", "qty": qty,
            "limit_price": 100.0, "reserved_cost_milli": int(reserved), "init_sl": 90.0, "init_trail": 90.0,
            "target_price": 110.0, "entry_atr": 5.0, "security_profile": {},
        }
        state = append_ordered_trading_proposal(
            state, order_id="r12-entry", proposal=proposal, plan=plan,
            timestamp="2026-09-04T09:01:00+08:00", mutation_id="r12-submit", frozen_params=frozen_params,
        )
        atomic_write_json(resolve_trading_order_state_path(root), state)
        bought = confirm_trading_buy_order_fill(
            root, order_id="r12-entry", fill_qty=qty, fill_price=99.0, trade_date="2026-09-04",
            expected_order_revision=int(state["revision"]), expected_account_revision=int(account["revision"]),
        )
        build_trading_protection_plan(root)
        oco = confirm_trading_protection_oco_submission(
            root, ticker="2454", expected_order_revision=int(bought["order_revision"]), broker_oco_group_id="R12-OCO",
            stop_broker_order_id="R12-STOP", tp_broker_order_id="R12-TP",
        )
        tp = next(row for row in oco["orders"].values() if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_TP)
        stop = next(row for row in oco["orders"].values() if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP)
        cash_before = int(load_trading_account_state(root)["cash_milli"])
        first_qty = 20
        first = confirm_trading_protection_sell_order_fill(
            root, order_id=tp["order_id"], fill_qty=first_qty, fill_price=111.0, trade_date="2026-09-05",
            expected_order_revision=int(oco["revision"]), expected_account_revision=int(bought["account_revision"]),
        )
        first_record = first["orders"]["orders"][tp["order_id"]]
        peer = first["orders"]["orders"][stop["order_id"]]
        expected_first_ledger = build_sell_ledger_from_price(111.0, first_qty, base_params, ticker="2454", security_profile={}, trade_date="2026-09-05")
        add_check(results, "trading_protection_sell_fill", case_id, "partial_tp_fill_sets_partial", "PARTIAL", first_record["status"])
        add_check(results, "trading_protection_sell_fill", case_id, "partial_tp_fill_reduces_remaining_qty", int(tp["qty"]) - first_qty, first_record["remaining_qty"])
        add_check(results, "trading_protection_sell_fill", case_id, "partial_tp_fill_reduces_real_held_qty", qty - first_qty, first["account"]["positions"]["2454"]["broker"]["qty"])
        add_check(results, "trading_protection_sell_fill", case_id, "partial_tp_does_not_mark_sold_half_complete", False, first["account"]["positions"]["2454"]["strategy_management"]["position_state"]["sold_half"])
        add_check(results, "trading_protection_sell_fill", case_id, "sell_fill_cash_credit_uses_canonical_exact_ledger", cash_before + int(expected_first_ledger["net_sell_total_milli"]), first["account"]["cash_milli"])
        add_check(results, "trading_protection_sell_fill", case_id, "native_oco_peer_is_cancelled_on_first_actual_fill", "CANCELLED", peer["status"])
        add_check(results, "trading_protection_sell_fill", case_id, "native_oco_cancel_is_same_order_revision_mutation", [stop["order_id"]], first["oco_cancelled_order_ids"])
        add_check(results, "trading_protection_sell_fill", case_id, "sell_fill_advances_account_once", int(bought["account_revision"]) + 1, first["account_revision"])
        add_check(results, "trading_protection_sell_fill", case_id, "sell_fill_advances_orders_once", int(oco["revision"]) + 1, first["order_revision"])

        final_qty = int(first_record["remaining_qty"])
        second = confirm_trading_protection_sell_order_fill(
            root, order_id=tp["order_id"], fill_qty=final_qty, fill_price=112.0, trade_date="2026-09-06",
            expected_order_revision=int(first["order_revision"]), expected_account_revision=int(first["account_revision"]),
        )
        final_record = second["orders"]["orders"][tp["order_id"]]
        add_check(results, "trading_protection_sell_fill", case_id, "protection_sell_partial_fills_may_span_days", {"2026-09-05", "2026-09-06"}, {row["trade_date"] for row in final_record["fills"]})
        add_check(results, "trading_protection_sell_fill", case_id, "final_tp_fill_sets_filled", "FILLED", final_record["status"])
        add_check(results, "trading_protection_sell_fill", case_id, "final_tp_fill_zero_remaining", 0, final_record["remaining_qty"])
        add_check(results, "trading_protection_sell_fill", case_id, "completed_tp_marks_canonical_sold_half", True, second["account"]["positions"]["2454"]["strategy_management"]["position_state"]["sold_half"])
        add_check(results, "trading_protection_sell_fill", case_id, "completed_tp_leaves_half_position", qty - int(tp["qty"]), second["account"]["positions"]["2454"]["broker"]["qty"])
        validate_trading_order_state(second["orders"])

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        qty = 100
        reserved = build_buy_ledger_from_price(100.0, qty, base_params)["net_buy_total_milli"]
        state = build_empty_trading_order_state(timestamp="2026-09-04T09:00:00+08:00", mutation_id="r12s-init")
        plan = {"plan_fingerprint":"r12s-plan","information_date":"2026-09-04","account_revision":0,"selected_params_sha256":canonical_json_sha256(frozen_params),"candidate_snapshot_sha256":"c","strategy_id":"s","param_selector":"p"}
        proposal = {"rank":1,"ticker":"2330","kind":"buy","entry_type":"normal","qty":qty,"limit_price":100.0,"reserved_cost_milli":int(reserved),"init_sl":90.0,"init_trail":90.0,"target_price":110.0,"entry_atr":5.0,"security_profile":{}}
        state = append_ordered_trading_proposal(state,order_id="r12s-entry",proposal=proposal,plan=plan,timestamp="2026-09-04T09:01:00+08:00",mutation_id="s",frozen_params=frozen_params)
        atomic_write_json(resolve_trading_order_state_path(root),state)
        bought = confirm_trading_buy_order_fill(root,order_id="r12s-entry",fill_qty=qty,fill_price=99.0,trade_date="2026-09-04",expected_order_revision=state["revision"],expected_account_revision=account["revision"])
        build_trading_protection_plan(root)
        stop_state = confirm_trading_protection_leg_submission(root,ticker="2330",action="STOP_FULL",expected_order_revision=bought["order_revision"],broker_order_id="R12S-STOP")
        stop = next(row for row in stop_state["orders"].values() if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP)
        stopped = confirm_trading_protection_sell_order_fill(root,order_id=stop["order_id"],fill_qty=qty,fill_price=85.0,trade_date="2026-09-05",expected_order_revision=stop_state["revision"],expected_account_revision=bought["account_revision"])
        add_check(results, "trading_protection_sell_fill", case_id, "stop_market_actual_fill_may_gap_below_trigger", "FILLED", stopped["status"])
        add_check(results, "trading_protection_sell_fill", case_id, "full_stop_fill_closes_position", False, "2330" in stopped["account"]["positions"])

    service_source = (Path(__file__).resolve().parents[2] / "services" / "trading" / "fill_reconciliation.py").read_text(encoding="utf-8")
    panel_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    add_check(results, "trading_protection_sell_fill", case_id, "sell_fill_service_does_not_infer_market_high_low", False, any(token in service_source for token in ("t_high", "t_low", "shadow_fill")))
    add_check(results, "trading_protection_sell_fill", case_id, "workbench_routes_sell_fill_to_sell_reconciliation", True, "confirm_trading_protection_sell_order_fill" in panel_source)
    summary["checks"] = len(results)
    return results, summary



def validate_trading_position_rollforward_contract_case(base_params):
    case_id = "TRADING_POSITION_ROLLFORWARD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
    from core.file_integrity import compute_file_sha256
    from core.params_io import params_to_json_dict
    from core.position_step import rollforward_position_management_from_completed_bar
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.signal_utils import generate_signals, unpack_precomputed_signals
    from core.trading_capabilities import build_trading_capability_snapshot
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    from services.trading.daily_workflow import resolve_trading_candidate_snapshot_path
    from services.trading.fill_reconciliation import confirm_trading_buy_order_fill
    from services.trading.order_planning import build_trading_proposed_order_plan
    from services.trading.order_state import confirm_trading_order_submission, load_trading_order_state
    from services.trading.position_rollforward import (
        build_trading_position_rollforward_snapshot,
        run_trading_position_rollforward,
    )
    from services.trading.protection_order_submission import confirm_trading_protection_leg_submission
    from services.trading.protection_planning import (
        PROTECTION_STOP_ACTION,
        build_trading_protection_plan,
        get_trading_protection_plan_read_model,
    )

    profile = get_trading_strategy_profile()
    project_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
        data_dir = Path(paths.data_dir)
        data_dir.mkdir(parents=True)

        needed = max(320, get_required_min_rows(base_params) + 20)
        dates = pd.bdate_range(end="2026-09-04", periods=needed)
        close = [200.0 + (idx % 7) * 0.1 for idx in range(needed)]
        frame = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close,
            "High": [value + 2.0 for value in close],
            "Low": [value - 2.0 for value in close],
            "Close": close,
            "Volume": [1_000_000] * needed,
        })
        frame.loc[needed - 1, ["Open", "High", "Low", "Close"]] = [199.0, 260.0, 198.0, 250.0]
        csv_path = data_dir / "2454.csv"
        frame.to_csv(csv_path, index=False)

        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(base_params)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")

        account = initialize_trading_account_state(root, cash=800_000)
        snapshot_path = resolve_trading_candidate_snapshot_path(root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps({
            "schema_version": 1,
            "runtime_domain": "trading",
            "strategy_id": profile.strategy_id,
            "param_selector": profile.param_selector,
            "latest_data_date": "2026-09-04",
            "param_latest_data_date": "2026-09-04",
            "selected_params_sha256": compute_file_sha256(selected_path),
            "candidate_rows": [{
                "ticker": "2454", "kind": "buy", "sort_value": 2.0, "expected_value": 0.4,
                "execution_plan_seed": {
                    "ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0,
                    "target_price": 230.0, "entry_atr": 4.0, "trade_date": "2026-09-04",
                    "security_profile": {"family": "stock"},
                },
            }],
        }, ensure_ascii=False), encoding="utf-8")

        plan = build_trading_proposed_order_plan(project_root=root)
        proposal = plan["orders"][0]
        ordered = confirm_trading_order_submission(
            root,
            rank=int(proposal["rank"]),
            ticker=proposal["ticker"],
            expected_revision=None,
            broker_order_id="ROLL-SYN-BUY",
        )
        entry_order_id = next(iter(ordered["orders"]))
        entry_order = ordered["orders"][entry_order_id]
        fill_result = confirm_trading_buy_order_fill(
            root,
            order_id=entry_order_id,
            fill_qty=int(entry_order["qty"]),
            fill_price=199.0,
            trade_date="2026-09-04",
            expected_order_revision=int(ordered["revision"]),
            expected_account_revision=int(account["revision"]),
        )
        account_before = load_trading_account_state(root)
        cash_before = int(account_before["cash_milli"])
        qty_before = int(account_before["positions"]["2454"]["broker"]["qty"])
        position_before = deepcopy(account_before["positions"]["2454"]["strategy_management"]["position_state"])

        build_trading_protection_plan(root)
        order_state = load_trading_order_state(root)
        order_state = confirm_trading_protection_leg_submission(
            root,
            ticker="2454",
            action=PROTECTION_STOP_ACTION,
            expected_order_revision=int(order_state["revision"]),
            broker_order_id="ROLL-SYN-STOP",
        )
        protection_before = get_trading_protection_plan_read_model(root)
        add_check(results, "trading_rollforward", case_id, "pre_rollforward_active_stop_matches_current_position_plan", [], protection_before.get("stale_active_protection_order_ids"))

        changed = deepcopy(base_params)
        changed.atr_times_trail = float(base_params.atr_times_trail) + 5.0
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(changed)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")

        clean_df, _stats = sanitize_ohlcv_dataframe(pd.read_csv(csv_path), "2454", min_rows=get_required_min_rows(base_params))
        atr_values, _buy, _sell, _limits = unpack_precomputed_signals(generate_signals(clean_df, base_params, ticker="2454"))
        expected_position = deepcopy(position_before)
        final_idx = len(clean_df) - 1
        rollforward_position_management_from_completed_bar(
            expected_position,
            completed_high=float(clean_df["High"].iloc[final_idx]),
            completed_atr=float(atr_values[final_idx]),
            params=base_params,
        )

        with patch("services.trading.position_rollforward.latest_allowed_completed_daily_date", return_value="2026-09-04"):
            due = build_trading_position_rollforward_snapshot(root)
            add_check(results, "trading_rollforward", case_id, "confirmed_position_is_due_through_latest_completed_bar", ["2454"], due["due_tickers"])
            rolled = run_trading_position_rollforward(root)
            revision_after_first = int(rolled["account_revision"])
            second = run_trading_position_rollforward(root)

        account_after = load_trading_account_state(root)
        management_after = account_after["positions"]["2454"]["strategy_management"]
        position_after = management_after["position_state"]
        add_check(results, "trading_rollforward", case_id, "rollforward_uses_source_entry_order_frozen_params_not_current_selected_params", int(expected_position["sl_milli"]), int(position_after["sl_milli"]))
        add_check(results, "trading_rollforward", case_id, "rollforward_advances_highest_high_from_completed_bar", int(expected_position["highest_high_since_entry_milli"]), int(position_after["highest_high_since_entry_milli"]))
        add_check(results, "trading_rollforward", case_id, "rollforward_records_processed_completed_date", "2026-09-04", management_after.get("last_rollforward_date"))
        add_check(results, "trading_rollforward", case_id, "rollforward_does_not_change_broker_qty", qty_before, int(account_after["positions"]["2454"]["broker"]["qty"]))
        add_check(results, "trading_rollforward", case_id, "rollforward_does_not_change_cash", cash_before, int(account_after["cash_milli"]))
        add_check(results, "trading_rollforward", case_id, "rollforward_is_idempotent_after_completed_date_consumed", "UP_TO_DATE", second["status"])
        add_check(results, "trading_rollforward", case_id, "idempotent_second_run_does_not_advance_revision", revision_after_first, int(second["account_revision"]))

        protection_after = get_trading_protection_plan_read_model(root)
        add_check(results, "trading_rollforward", case_id, "old_active_stop_is_flagged_stale_after_trailing_state_changes", ["2454"], protection_after.get("stale_active_protection_tickers"))
        add_check(results, "trading_rollforward", case_id, "active_protection_order_is_not_silently_cancelled_or_rewritten", 1, len([row for row in load_trading_order_state(root)["orders"].values() if row.get("status") == "ORDERED" and row.get("side") == "SELL"]))

        future = pd.concat([frame, pd.DataFrame([{
            "Date": "2026-09-05", "Open": 251.0, "High": 270.0, "Low": 250.0, "Close": 265.0, "Volume": 1_000_000,
        }])], ignore_index=True)
        future.to_csv(csv_path, index=False)
        with patch("services.trading.position_rollforward.latest_allowed_completed_daily_date", return_value="2026-09-04"):
            try:
                build_trading_position_rollforward_snapshot(root)
            except RuntimeError as exc:
                provisional_rejected = "尚未完成日K" in str(exc)
            else:
                provisional_rejected = False
        add_check(results, "trading_rollforward", case_id, "rollforward_rejects_dataset_containing_uncompleted_future_daily_bar", True, provisional_rejected)

    capability = build_trading_capability_snapshot()
    add_check(results, "trading_rollforward", case_id, "daily_position_rollforward_capability_is_implemented", True, bool(capability["capabilities"]["daily_position_rollforward"]["implemented"]))
    add_check(results, "trading_rollforward", case_id, "indicator_sell_execution_capability_is_implemented", True, bool(capability["capabilities"]["indicator_sell_execution"]["implemented"]))

    service_source = (project_root / "services" / "trading" / "position_rollforward.py").read_text(encoding="utf-8")
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    daily_source = (project_root / "services" / "trading" / "daily_workflow.py").read_text(encoding="utf-8")
    snapshot_body = service_source.split("def build_trading_position_rollforward_snapshot", 1)[1].split("def run_trading_position_rollforward", 1)[0]
    add_check(results, "trading_rollforward", case_id, "read_only_rollforward_snapshot_does_not_run_fill_recovery", False, "recover_trading_fill_transaction(" in snapshot_body)
    add_check(results, "trading_rollforward", case_id, "rollforward_service_does_not_execute_or_infer_broker_sell", False, any(token in service_source for token in ("confirm_trading_sell_fill(", "confirm_trading_protection_sell_order_fill(", "t_low", "t_open")))
    add_check(results, "trading_rollforward", case_id, "workbench_exposes_explicit_position_rollforward_action", True, '"持股日終推進", "rollforward"' in panel_source and 'elif action == "rollforward"' in panel_source)
    add_check(results, "trading_rollforward", case_id, "daily_workflow_orders_rollforward_and_indicator_after_data_before_param_training", True, daily_source.index("data_result = run_trading_market_data_update") < daily_source.index("rollforward_result = run_trading_position_rollforward") < daily_source.index("indicator_result = build_trading_indicator_exit_plan") < daily_source.index("param_result = run_trading_strategy_param_training"))

    summary["checks"] = len(results)
    return results, summary

def validate_trading_indicator_sell_execution_contract_case(base_params):
    case_id = "TRADING_INDICATOR_SELL_EXECUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.trading_capabilities import build_trading_capability_snapshot
    from core.trading_order_state import (
        TRADING_INDICATOR_ORDER_TYPE_MARKET, TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
        TRADING_ORDER_STATUS_CANCELLED, active_trading_indicator_exit_orders,
        append_ordered_trading_indicator_exit, build_empty_trading_order_state, cancel_ordered_trading_order,
    )

    state = build_empty_trading_order_state(timestamp="2026-09-05T08:00:00+08:00", mutation_id="init")
    exit_row = {
        "ticker":"2454", "qty":1000, "position_qty":1000, "entry_order_id":"ENTRY-1", "entry_trade_date":"2026-09-04",
        "priority":1, "signal_key":"sig-1", "signal_information_date":"2026-09-04", "position_plan_fingerprint":"pos-fp",
        "position_state_sha256":"pos-sha", "frozen_params_sha256":"params-sha", "market_data_sha256":"market-sha",
    }
    plan={"plan_fingerprint":"plan-fp", "account_revision":1}
    ordered = append_ordered_trading_indicator_exit(state, order_id="IND-1", exit_plan=exit_row, plan=plan, timestamp="2026-09-05T08:01:00+08:00", mutation_id="submit", broker_order_id="BROKER-1")
    row=ordered["orders"]["IND-1"]
    add_check(results,"trading_indicator_sell",case_id,"indicator_order_purpose_is_distinct",TRADING_ORDER_PURPOSE_INDICATOR_EXIT,row["purpose"] )
    add_check(results,"trading_indicator_sell",case_id,"indicator_order_is_market",TRADING_INDICATOR_ORDER_TYPE_MARKET,row["order_type"] )
    add_check(results,"trading_indicator_sell",case_id,"indicator_order_is_full_position",row["position_qty_at_submission"],row["qty"] )
    add_check(results,"trading_indicator_sell",case_id,"indicator_order_has_no_trigger",None,row["trigger_price_milli"] )
    add_check(results,"trading_indicator_sell",case_id,"indicator_order_has_no_limit",None,row["limit_price_milli"] )
    add_check(results,"trading_indicator_sell",case_id,"active_indicator_order_is_explicit",1,len(active_trading_indicator_exit_orders(ordered)) )
    cancelled=cancel_ordered_trading_order(ordered,order_id="IND-1",timestamp="2026-09-05T08:02:00+08:00",mutation_id="cancel")
    add_check(results,"trading_indicator_sell",case_id,"cancelled_indicator_can_leave_active_set",0,len(active_trading_indicator_exit_orders(cancelled)) )
    retry=append_ordered_trading_indicator_exit(cancelled,order_id="IND-2",exit_plan=exit_row,plan=plan,timestamp="2026-09-05T08:03:00+08:00",mutation_id="retry")
    add_check(results,"trading_indicator_sell",case_id,"cancelled_signal_retry_advances_attempt",2,retry["orders"]["IND-2"]["signal_attempt"] )
    capability=build_trading_capability_snapshot()
    add_check(results,"trading_indicator_sell",case_id,"all_required_live_capabilities_are_implemented",True,bool(capability.get("all_required_live_capabilities_ready")) )
    add_check(results,"trading_indicator_sell",case_id,"no_implementation_live_blocker_remains",[],list(capability.get("live_blocking_capabilities") or []) )
    project_root=Path(__file__).resolve().parents[2]
    planning=(project_root/"services/trading/indicator_exit_planning.py").read_text(encoding="utf-8")
    submit=(project_root/"services/trading/indicator_exit_order_submission.py").read_text(encoding="utf-8")
    fill=(project_root/"services/trading/fill_reconciliation.py").read_text(encoding="utf-8")
    protection=(project_root/"services/trading/protection_order_submission.py").read_text(encoding="utf-8")
    panel=(project_root/"services/workbench_ui/trading_account_panel.py").read_text(encoding="utf-8")
    add_check(results,"trading_indicator_sell",case_id,"planning_does_not_infer_broker_fill",False,"confirm_trading_indicator_sell_order_fill(" in planning )
    add_check(results,"trading_indicator_sell",case_id,"submission_requires_protection_cancellation",True,"active Stop/TP protection SELL" in submit )
    add_check(results,"trading_indicator_sell",case_id,"fill_reconciliation_uses_canonical_ind_sell_event",True,'event = "IND_SELL"' in fill )
    add_check(results,"trading_indicator_sell",case_id,"protection_submission_blocks_active_indicator_sell",True,"active Indicator MARKET SELL" in protection )
    add_check(results,"trading_indicator_sell",case_id,"workbench_exposes_indicator_market_sell_confirmation",True,"Indicator SELL 計畫" in panel and "確認選取 MARKET SELL 已送單" in panel and "confirm_trading_indicator_sell_order_fill" in panel )

    summary["checks"] = len(results)
    return results, summary

def validate_trading_operations_status_contract_case(base_params):
    case_id = "TRADING_OPERATIONS_STATUS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from services.trading.operations_status import (
        NEXT_BUILD_PROPOSED,
        NEXT_DAY_LOCKED,
        NEXT_INITIALIZE_ACCOUNT,
        NEXT_NO_ENTRY,
        NEXT_RECOVER_FILL,
        NEXT_RECONCILE_ENTRY,
        NEXT_REFRESH_PROTECTION,
        NEXT_REPLACE_PROTECTION,
        NEXT_ROLLFORWARD_POSITIONS,
        NEXT_RUN_SCANNER,
        NEXT_SET_CASH,
        NEXT_SUBMIT_PROPOSED,
        NEXT_SUBMIT_PROTECTION_STOP,
        NEXT_CANCEL_PROTECTION_FOR_INDICATOR,
        NEXT_SUBMIT_INDICATOR_EXIT,
        NEXT_RECONCILE_INDICATOR_EXIT,
        NEXT_UPDATE_PARAMS,
        OPERATIONS_STATUS_BLOCKED,
        OPERATIONS_STATUS_LOCKED_TODAY,
        derive_trading_operations_status,
    )

    workflow = {
        "latest_data_date": "2026-09-04",
        "params_ready_for_scan": True,
        "param_selector": "base_finalist_best",
    }
    account = {"initialized": True, "revision": 7, "cash": 500_000.0, "positions": []}
    orders = {"revision": 4, "orders": []}
    candidate = {"exists": True, "valid": True, "fresh": True, "candidate_count": 3, "information_date": "2026-09-04"}
    proposed = {"exists": False, "valid": False, "fresh": False, "order_count": 0}
    protection = {"exists": False, "fresh": False, "positions": []}
    indicator_clear = {"exists": True, "fresh": True, "exit_count": 0, "exits": [], "active_indicator_exit_order_count": 0, "active_indicator_exit_tickers": []}

    def derive(**overrides):
        inputs = {
            "workflow": deepcopy(workflow),
            "account": deepcopy(account),
            "orders": deepcopy(orders),
            "candidate": deepcopy(candidate),
            "proposed": deepcopy(proposed),
            "protection": deepcopy(protection),
            "indicator_exit": deepcopy(indicator_clear),
            "fill_transaction_pending": False,
            "component_errors": {},
        }
        inputs.update(overrides)
        return derive_trading_operations_status(**inputs)

    recovery = derive(fill_transaction_pending=True)
    add_check(results, "trading_operations", case_id, "pending_fill_transaction_is_top_priority_blocker", NEXT_RECOVER_FILL, recovery["next_action_code"])
    add_check(results, "trading_operations", case_id, "pending_fill_transaction_status_is_blocked", OPERATIONS_STATUS_BLOCKED, recovery["overall_status"])
    add_check(results, "trading_operations", case_id, "pending_fill_transaction_disables_all_workflow_actions", {"data": False, "rollforward": False, "params": False, "scanner": False, "orders": False, "all": False}, recovery["workflow_action_availability"])

    uninitialized = derive(account={"initialized": False, "revision": None, "cash": None, "positions": []})
    add_check(results, "trading_operations", case_id, "uninitialized_account_next_action", NEXT_INITIALIZE_ACCOUNT, uninitialized["next_action_code"])

    no_cash = derive(account={"initialized": True, "revision": 0, "cash": None, "positions": []})
    add_check(results, "trading_operations", case_id, "initialized_account_without_cash_next_action", NEXT_SET_CASH, no_cash["next_action_code"])
    add_check(results, "trading_operations", case_id, "allocation_disabled_without_cash", False, no_cash["workflow_action_availability"]["orders"])

    strategy_account = {
        "initialized": True,
        "revision": 8,
        "cash": 400_000.0,
        "positions": [{"ticker": "2317", "source": "strategy_fill", "qty": 100, "management_status": "active"}],
    }
    rollforward_due = derive(account=strategy_account, position_rollforward={"due_tickers": ["2317"]})
    add_check(results, "trading_operations", case_id, "due_strategy_position_requires_rollforward_before_protection_or_allocation", NEXT_ROLLFORWARD_POSITIONS, rollforward_due["next_action_code"])
    add_check(results, "trading_operations", case_id, "due_strategy_position_enables_explicit_rollforward_action", True, rollforward_due["workflow_action_availability"]["rollforward"])
    add_check(results, "trading_operations", case_id, "due_strategy_position_blocks_new_allocation_until_advanced", False, rollforward_due["workflow_action_availability"]["orders"])

    missing_stop = derive(account=strategy_account)
    add_check(results, "trading_operations", case_id, "strategy_position_without_stop_is_reported", ["2317"], missing_stop["missing_stop_tickers"])
    add_check(results, "trading_operations", case_id, "missing_stop_without_fresh_plan_requires_plan_refresh", NEXT_REFRESH_PROTECTION, missing_stop["next_action_code"])

    fresh_protection = {"exists": True, "fresh": True, "positions": [{"ticker": "2317"}]}
    missing_stop_fresh_plan = derive(account=strategy_account, protection=fresh_protection)
    add_check(results, "trading_operations", case_id, "fresh_plan_without_active_stop_requires_submission", NEXT_SUBMIT_PROTECTION_STOP, missing_stop_fresh_plan["next_action_code"])

    active_stop_orders = {
        "revision": 5,
        "orders": [{
            "order_id": "stop1", "ticker": "2317", "side": "SELL", "purpose": "PROTECTION_STOP",
            "status": "ORDERED", "information_date": "2026-09-04",
        }],
    }
    protected = derive(account=strategy_account, orders=active_stop_orders, protection=fresh_protection)
    add_check(results, "trading_operations", case_id, "active_stop_clears_missing_stop_gap", [], protected["missing_stop_tickers"])
    add_check(results, "trading_operations", case_id, "long_lived_protection_sell_does_not_block_new_allocation", True, protected["workflow_action_availability"]["orders"])

    stale_protection = derive(
        account=strategy_account,
        orders=active_stop_orders,
        protection={**fresh_protection, "stale_active_protection_order_ids": ["stop1"], "stale_active_protection_tickers": ["2317"]},
    )
    add_check(results, "trading_operations", case_id, "stale_active_protection_requires_explicit_cancel_resubmit_sequence", NEXT_REPLACE_PROTECTION, stale_protection["next_action_code"])
    add_check(results, "trading_operations", case_id, "stale_active_protection_blocks_new_allocation", False, stale_protection["workflow_action_availability"]["orders"])

    indicator_due = {"exists": True, "fresh": True, "exit_count": 1, "exits": [{"ticker": "2317", "signal_key": "sig-1"}], "active_indicator_exit_order_count": 0, "active_indicator_exit_tickers": []}
    due_with_stop = derive(account=strategy_account, orders=active_stop_orders, protection=fresh_protection, indicator_exit=indicator_due)
    add_check(results, "trading_operations", case_id, "indicator_due_with_active_protection_requires_cancel_first", NEXT_CANCEL_PROTECTION_FOR_INDICATOR, due_with_stop["next_action_code"])
    due_without_stop = derive(account=strategy_account, protection=fresh_protection, indicator_exit=indicator_due)
    add_check(results, "trading_operations", case_id, "indicator_due_without_protection_requires_market_submission", NEXT_SUBMIT_INDICATOR_EXIT, due_without_stop["next_action_code"])
    active_indicator_orders = {"revision": 6, "orders": [{"order_id":"ind1","ticker":"2317","side":"SELL","purpose":"INDICATOR_EXIT","status":"ORDERED","information_date":"2026-09-04"}]}
    active_indicator = derive(account=strategy_account, orders=active_indicator_orders, protection=fresh_protection, indicator_exit=indicator_due)
    add_check(results, "trading_operations", case_id, "active_indicator_order_requires_reconciliation", NEXT_RECONCILE_INDICATOR_EXIT, active_indicator["next_action_code"])

    active_entry_orders = {
        "revision": 6,
        "orders": [{
            "order_id": "buy1", "ticker": "2330", "side": "BUY", "purpose": "ENTRY_BUY",
            "status": "PARTIAL", "information_date": "2026-09-04",
        }],
    }
    active_entry = derive(orders=active_entry_orders)
    add_check(results, "trading_operations", case_id, "active_entry_buy_requires_reconciliation", NEXT_RECONCILE_ENTRY, active_entry["next_action_code"])
    add_check(results, "trading_operations", case_id, "active_entry_buy_disables_new_allocation", False, active_entry["workflow_action_availability"]["orders"])
    active_entry_with_due = derive(orders=active_entry_orders, account=strategy_account, position_rollforward={"due_tickers": ["2317"]})
    add_check(results, "trading_operations", case_id, "active_entry_reconciliation_precedes_rollforward_that_cannot_mutate_while_entry_is_active", NEXT_RECONCILE_ENTRY, active_entry_with_due["next_action_code"])
    add_check(results, "trading_operations", case_id, "active_entry_disables_rollforward_action", False, active_entry_with_due["workflow_action_availability"]["rollforward"])

    stale_params = derive(workflow={"latest_data_date": "2026-09-04", "params_ready_for_scan": False})
    add_check(results, "trading_operations", case_id, "stale_params_require_update_params", NEXT_UPDATE_PARAMS, stale_params["next_action_code"])
    add_check(results, "trading_operations", case_id, "scanner_disabled_until_params_ready", False, stale_params["workflow_action_availability"]["scanner"])

    workflow_error = derive(component_errors={"workflow": "RuntimeError: broken workflow read model"})
    add_check(results, "trading_operations", case_id, "workflow_read_error_blocks_operations_status", OPERATIONS_STATUS_BLOCKED, workflow_error["overall_status"])
    add_check(results, "trading_operations", case_id, "workflow_read_error_disables_all_workflow_actions", False, any(workflow_error["workflow_action_availability"].values()))
    rollforward_error = derive(component_errors={"position_rollforward": "RuntimeError: broken rollforward snapshot"})
    add_check(results, "trading_operations", case_id, "rollforward_read_error_blocks_operations_status", OPERATIONS_STATUS_BLOCKED, rollforward_error["overall_status"])
    add_check(results, "trading_operations", case_id, "rollforward_read_error_blocks_allocation_and_daily_all", [False, False], [rollforward_error["workflow_action_availability"]["orders"], rollforward_error["workflow_action_availability"]["all"]])

    stale_candidate = derive(candidate={"exists": True, "valid": True, "fresh": False, "candidate_count": 3})
    add_check(results, "trading_operations", case_id, "stale_candidate_requires_scanner", NEXT_RUN_SCANNER, stale_candidate["next_action_code"])

    same_day_history = {
        "revision": 7,
        "orders": [{
            "order_id": "buydone", "ticker": "2330", "side": "BUY", "purpose": "ENTRY_BUY",
            "status": "FILLED", "information_date": "2026-09-04",
        }],
    }
    locked = derive(orders=same_day_history)
    add_check(results, "trading_operations", case_id, "same_information_day_entry_history_locks_reallocation", NEXT_DAY_LOCKED, locked["next_action_code"])
    add_check(results, "trading_operations", case_id, "same_day_lock_has_explicit_status", OPERATIONS_STATUS_LOCKED_TODAY, locked["overall_status"])
    add_check(results, "trading_operations", case_id, "same_day_lock_disables_step4_only", False, locked["workflow_action_availability"]["orders"])
    add_check(results, "trading_operations", case_id, "same_day_lock_keeps_scanner_available", True, locked["workflow_action_availability"]["scanner"])

    sell_fill_history = {"revision": 8, "orders": [{"order_id":"selldone","ticker":"2317","side":"SELL","purpose":"INDICATOR_EXIT","status":"FILLED","information_date":"2026-09-04","latest_fill_trade_date":"2026-09-05"}]}
    sell_locked = derive(orders=sell_fill_history)
    add_check(results, "trading_operations", case_id, "sell_fill_after_latest_completed_date_locks_same_session_reallocation", NEXT_DAY_LOCKED, sell_locked["next_action_code"])
    add_check(results, "trading_operations", case_id, "sell_session_lock_disables_step4", False, sell_locked["workflow_action_availability"]["orders"])
    add_check(results, "trading_operations", case_id, "sell_session_lock_is_explicit", True, sell_locked["same_session_sell_locked"])

    build_proposed = derive()
    add_check(results, "trading_operations", case_id, "fresh_candidate_without_fresh_proposal_requires_step4", NEXT_BUILD_PROPOSED, build_proposed["next_action_code"])

    fresh_proposed = {"exists": True, "valid": True, "fresh": True, "order_count": 2, "information_date": "2026-09-04"}
    submit = derive(proposed=fresh_proposed)
    add_check(results, "trading_operations", case_id, "fresh_proposal_with_orders_requires_explicit_submission", NEXT_SUBMIT_PROPOSED, submit["next_action_code"])
    add_check(results, "trading_operations", case_id, "fresh_proposal_does_not_create_active_broker_order", 0, submit["active_entry_order_count"])

    no_orders = derive(proposed={"exists": True, "valid": True, "fresh": True, "order_count": 0, "information_date": "2026-09-04"})
    add_check(results, "trading_operations", case_id, "fresh_zero_order_plan_marks_no_entry_today", NEXT_NO_ENTRY, no_orders["next_action_code"])

    manual_account = {
        "initialized": True,
        "revision": 9,
        "cash": 500_000.0,
        "positions": [{"ticker": "0050", "source": "manual_adopted", "qty": 1000, "management_status": "unmanaged"}],
    }
    manual_only = derive(account=manual_account)
    add_check(results, "trading_operations", case_id, "manual_adopted_is_not_treated_as_strategy_stop_gap", [], manual_only["missing_stop_tickers"])
    add_check(results, "trading_operations", case_id, "manual_adopted_is_reported_separately", ["0050"], manual_only["manual_tickers"])

    source = (Path(__file__).resolve().parents[2] / "services" / "trading" / "operations_status.py").read_text(encoding="utf-8")
    panel_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    add_check(results, "trading_operations", case_id, "operations_status_is_read_only_composition", False, any(token in source for token in ("atomic_write_json(", "confirm_trading_", "run_trading_market_data_update(")))
    add_check(results, "trading_operations", case_id, "operations_status_disables_hidden_protection_recovery", True, "get_trading_protection_plan_read_model(root, recover_pending_fill=False)" in source)
    add_check(results, "trading_operations", case_id, "operations_status_disables_hidden_indicator_recovery", True, "get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False)" in source)
    rollforward_source = (Path(__file__).resolve().parents[2] / "services" / "trading" / "position_rollforward.py").read_text(encoding="utf-8")
    rollforward_snapshot_body = rollforward_source.split("def build_trading_position_rollforward_snapshot", 1)[1].split("def run_trading_position_rollforward", 1)[0]
    add_check(results, "trading_operations", case_id, "operations_rollforward_snapshot_has_no_hidden_fill_recovery", False, "recover_trading_fill_transaction(" in rollforward_snapshot_body)
    add_check(results, "trading_operations", case_id, "workbench_has_operations_overview", True, "Trading 操作總覽" in panel_source)
    add_check(results, "trading_operations", case_id, "workbench_consumes_operations_status_owner", True, "build_trading_operations_status" in panel_source)
    add_check(results, "trading_operations", case_id, "workbench_exposes_full_state_refresh", True, "全狀態刷新" in panel_source)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_workbench_account_panel_contract_case(base_params):
    case_id = "TRADING_WORKBENCH_ACCOUNT_PANEL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from services.workbench_ui.trading_account_panel import (
        build_trading_account_panel_snapshot,
        parse_trading_money_text,
        parse_trading_qty_text,
    )
    from services.workbench_ui.workbench import PANEL_SPECS, build_workbench_spec

    workbench_spec = build_workbench_spec()
    panel_specs = {row["panel_id"]: row for row in workbench_spec.get("panels", [])}
    add_check(results, "trading_workbench", case_id, "actual_trading_panel_is_registered", True, "trading_account" in panel_specs)
    trading_panel = panel_specs.get("trading_account", {})
    add_check(results, "trading_workbench", case_id, "actual_trading_panel_label", "實際交易", trading_panel.get("tab_label"))
    add_check(
        results,
        "trading_workbench",
        case_id,
        "actual_trading_panel_uses_account_state_backend",
        "services.trading.account_state.get_trading_account_read_model",
        trading_panel.get("backend_runner"),
    )
    add_check(
        results,
        "trading_workbench",
        case_id,
        "actual_trading_panel_uses_dedicated_factory",
        "services.workbench_ui.trading_account_panel:TradingAccountPanel",
        next((row.get("panel_factory_path") for row in PANEL_SPECS if row.get("panel_id") == "trading_account"), None),
    )

    add_check(results, "trading_workbench", case_id, "money_parser_accepts_grouping_and_decimal", "1234567.89", str(parse_trading_money_text("1,234,567.89", "cash")))
    add_check(results, "trading_workbench", case_id, "qty_parser_accepts_grouping", 1000, parse_trading_qty_text("1,000"))
    try:
        parse_trading_qty_text("1.5")
    except ValueError:
        fractional_qty_rejected = True
    else:
        fractional_qty_rejected = False
    add_check(results, "trading_workbench", case_id, "fractional_share_qty_is_rejected", True, fractional_qty_rejected)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        empty_snapshot = build_trading_account_panel_snapshot(root)
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_handles_uninitialized_account", False, empty_snapshot["initialized"])
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_uses_project_relative_state_path", "state/trading/account.json", empty_snapshot["state_path"])

        state = initialize_trading_account_state(root, cash=900_000)
        state = adopt_existing_trading_position(
            root,
            ticker="2330",
            qty=1000,
            cost_basis_total=500_000,
            entry_date="2026-08-01",
            expected_revision=state["revision"],
        )
        snapshot = build_trading_account_panel_snapshot(root)
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_reads_current_revision", state["revision"], snapshot["revision"])
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_reads_cash", 900_000.0, snapshot["cash"])
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_reads_manual_position", "2330", snapshot["positions"][0]["ticker"])
        from core.trading_policy import get_trading_policy_snapshot
        expected_policy = get_trading_policy_snapshot()
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_uses_current_trading_strategy_config", expected_policy["strategy_id"], snapshot["policy"]["strategy_id"])
        add_check(results, "trading_workbench", case_id, "workbench_snapshot_uses_current_param_selector_config", expected_policy["param_selector"], snapshot["policy"]["param_selector"])

    summary["checks"] = len(results)
    return results, summary


__all__ = [
    "validate_trading_account_state_contract_case",
    "validate_trading_daily_workflow_contract_case",
    "validate_trading_proposed_order_plan_contract_case",
    "validate_trading_pending_order_state_contract_case",
    "validate_trading_confirmed_fill_reconciliation_contract_case",
    "validate_trading_protection_plan_contract_case",
    "validate_trading_protection_order_submission_contract_case",
    "validate_trading_protection_sell_fill_reconciliation_contract_case",
    "validate_trading_position_rollforward_contract_case",
    "validate_trading_indicator_sell_execution_contract_case",
    "validate_trading_operations_status_contract_case",
    "validate_trading_workbench_account_panel_contract_case",
]


def validate_trading_prelive_operational_audit_contract_case(_base_params):
    case_id = "TRADING_PRELIVE_OPERATIONAL_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from datetime import datetime
    from zoneinfo import ZoneInfo

    from core.trading_capabilities import build_trading_capability_snapshot
    from core.trading_market_clock import (
        assert_completed_daily_information_date,
        latest_allowed_completed_daily_date,
        select_latest_completed_daily_date,
    )
    from services.downloader import application as downloader_application
    from services.downloader import runtime as downloader_runtime
    from services.downloader import sync as downloader_sync
    from services.downloader import universe as downloader_universe
    from services.trading.operational_audit import (
        TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED,
        build_trading_operational_audit,
        run_trading_operational_audit,
    )

    tz = ZoneInfo("Asia/Taipei")
    morning = datetime(2026, 9, 5, 10, 0, tzinfo=tz)
    after_close = datetime(2026, 9, 5, 14, 30, tzinfo=tz)
    add_check(results, "trading_prelive", case_id, "morning_completed_daily_cutoff_excludes_today", "2026-09-04", latest_allowed_completed_daily_date(now=morning))
    add_check(results, "trading_prelive", case_id, "after_close_completed_daily_cutoff_allows_today", "2026-09-05", latest_allowed_completed_daily_date(now=after_close))
    add_check(results, "trading_prelive", case_id, "provider_provisional_today_row_is_ignored_before_cutoff", "2026-09-04", select_latest_completed_daily_date(["2026-09-04", "2026-09-05"], now=morning))
    add_check(results, "trading_prelive", case_id, "provider_today_row_is_eligible_after_cutoff", "2026-09-05", select_latest_completed_daily_date(["2026-09-04", "2026-09-05"], now=after_close))
    try:
        assert_completed_daily_information_date("2026-09-05", now=morning)
    except RuntimeError:
        provisional_rejected = True
    else:
        provisional_rejected = False
    add_check(results, "trading_prelive", case_id, "intraday_today_information_date_is_rejected", True, provisional_rejected)

    class _FakeLoader:
        def get_data(self, **_kwargs):
            return pd.DataFrame(
                {
                    "date": ["2026-09-04", "2026-09-05"],
                    "open": [100.0, 101.0],
                    "max": [102.0, 103.0],
                    "min": [99.0, 100.0],
                    "close": [101.0, 102.0],
                    "Trading_volume": [1000, 2000],
                }
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        data_dir = temp_root / "data" / "trading" / "tw_stock_data_vip"
        output_dir = temp_root / "outputs" / "trading" / "smart_downloader"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (
            patch.object(downloader_runtime, "SAVE_DIR", str(data_dir)),
            patch.object(downloader_runtime, "OUTPUT_DIR", str(output_dir)),
            patch.object(downloader_runtime, "get_finmind_loader", return_value=_FakeLoader()),
            patch.object(downloader_runtime, "FINMIND_DOWNLOAD_SLEEP_SEC", 0),
        ):
            sync_summary = downloader_sync.smart_download_vip_data(["2330"], "2026-09-04", verbose=False)
        saved = pd.read_csv(data_dir / "2330.csv")
        saved_dates = pd.to_datetime(saved["Date"]).dt.strftime("%Y-%m-%d").tolist()
        add_check(results, "trading_prelive", case_id, "downloader_physically_seals_rows_to_confirmed_market_date", ["2026-09-04"], saved_dates)
        add_check(results, "trading_prelive", case_id, "downloader_reports_trimmed_future_rows", 1, int(sync_summary.get("trimmed_future_row_count") or 0))

    with (
        patch.object(downloader_application, "get_market_last_date", return_value="2026-09-04"),
        patch.object(downloader_application, "get_or_update_universe", return_value=["2330", "2317"]),
        patch.object(downloader_application, "smart_download_vip_data", return_value={
            "count_success": 1,
            "count_skipped_latest": 0,
            "last_date_check_error_count": 0,
            "download_error_count": 1,
            "trimmed_future_row_count": 0,
            "issue_log_path": None,
        }),
        patch.object(downloader_runtime, "get_taipei_now", return_value=after_close),
    ):
        try:
            downloader_application.run_trading_dataset_update()
        except RuntimeError:
            incomplete_download_rejected = True
        else:
            incomplete_download_rejected = False
    add_check(results, "trading_prelive", case_id, "trading_update_rejects_known_ticker_download_failure", True, incomplete_download_rejected)

    with patch.object(downloader_runtime, "get_taipei_now", return_value=morning), patch.object(downloader_runtime, "get_finmind_loader", return_value=_FakeLoader()):
        safe_market_date = downloader_universe.get_market_last_date()
    add_check(results, "trading_prelive", case_id, "market_date_resolver_ignores_provisional_today_row", "2026-09-04", safe_market_date)

    capability = build_trading_capability_snapshot()
    blockers = set(capability.get("live_blocking_capabilities") or [])
    add_check(results, "trading_prelive", case_id, "daily_position_rollforward_is_implemented_and_no_longer_live_blocker", False, "daily_position_rollforward" in blockers)
    add_check(results, "trading_prelive", case_id, "indicator_sell_execution_is_implemented_and_no_longer_live_blocker", False, "indicator_sell_execution" in blockers)
    add_check(results, "trading_prelive", case_id, "completed_daily_bar_seal_is_implemented", True, bool((capability["capabilities"]["completed_daily_bar_seal"]["implemented"])))

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        reduced = root / "data" / "tw_stock_data_vip_reduced"
        reduced.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "Date": ["2026-03-02", "2026-03-03"],
                "Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [1, 1], "Volume": [1, 1],
            }
        ).to_csv(reduced / "2330.csv", index=False)
        audit = build_trading_operational_audit(root)
        add_check(results, "trading_prelive", case_id, "research_cutoff_breach_blocks_live_readiness", TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED, audit["status"])
        add_check(results, "trading_prelive", case_id, "research_cutoff_breach_is_reported", True, any("Research reduced dataset 超過 cutoff" in item for item in audit["blockers"]))
        report = run_trading_operational_audit(root)
        add_check(results, "trading_prelive", case_id, "operational_audit_writes_human_readable_report", True, (root / report["markdown_path"]).is_file())
        add_check(results, "trading_prelive", case_id, "operational_audit_writes_machine_readable_report", True, (root / report["json_path"]).is_file())

    project_root = Path(__file__).resolve().parents[2]
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    audit_source = (project_root / "services" / "trading" / "operational_audit.py").read_text(encoding="utf-8")
    add_check(results, "trading_prelive", case_id, "workbench_exposes_explicit_prelive_audit_action", True, "實盤就緒檢查" in panel_source)
    add_check(results, "trading_prelive", case_id, "operational_audit_does_not_mutate_trading_state", False, any(token in audit_source for token in ("confirm_trading_", "mutate_trading_", "set_trading_cash_balance(")))

    summary["checks"] = len(results)
    return results, summary
