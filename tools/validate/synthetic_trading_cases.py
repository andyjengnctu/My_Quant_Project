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
         patch.object(daily_workflow, "run_trading_strategy_param_training", side_effect=lambda **_kwargs: call_order.append("params") or {"status": "READY"}), \
         patch.object(daily_workflow, "run_trading_candidate_scan", side_effect=lambda **_kwargs: call_order.append("scanner") or {"status": "READY", "candidate_rows": []}):
        workflow_result = daily_workflow.run_trading_daily_workflow(project_root=project_root, environ={})
    add_check(results, "trading_daily", case_id, "daily_workflow_executes_data_params_scanner_in_order", ["data", "params", "scanner"], call_order)
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
    add_check(results, "trading_fill", case_id, "workbench_requires_explicit_broker_fill_confirmation", True, 'text="確認選取成交"' in panel_source and "confirm_trading_buy_order_fill(" in panel_source)
    add_check(results, "trading_fill", case_id, "workbench_fill_inputs_are_actual_qty_price_and_date", True, all(token in panel_source for token in ("本次成交股數", "本次成交價", "成交日 YYYY-MM-DD")))
    add_check(results, "trading_fill", case_id, "workbench_does_not_auto_infer_fill_from_market_bar", False, "t_low" in panel_source or "t_high" in panel_source or "execute_pre_market_entry_plan" in panel_source)

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
    "validate_trading_workbench_account_panel_contract_case",
]
