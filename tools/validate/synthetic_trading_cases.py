from copy import deepcopy
from pathlib import Path
import tempfile

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
    "validate_trading_workbench_account_panel_contract_case",
]
