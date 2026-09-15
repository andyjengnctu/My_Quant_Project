from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from .checks import bind_synthetic_case, bind_checks, add_check, run_bound_checks
from core.file_integrity import atomic_write_json
from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    money_to_milli,
)
from core.trading_account_state import (
    apply_confirmed_sell_fill,
    apply_confirmed_strategy_buy_fill,
    rebuild_trading_account_economics,
    validate_trading_account_state,
)
from core.trading_identity import normalize_trading_date, normalize_trading_ticker
from core.trading_state_paths import (
    resolve_trading_account_state_path as resolve_core_trading_account_state_path,
    resolve_trading_fill_transaction_path as resolve_core_trading_fill_transaction_path,
    resolve_trading_order_state_path as resolve_core_trading_order_state_path,
)
from services.trading.accounting_policy import build_standalone_trading_accounting_params, overlay_trading_accounting_params
from services.trading.account_state import (
    TradingAccountRevisionConflict,
    adopt_existing_trading_position,
    correct_existing_trading_position,
    correct_trading_transaction,
    delete_trading_transaction,
    remove_existing_trading_position,
    record_manual_trading_buy,
    record_manual_trading_sell,
    record_strategy_trading_buy,
    get_trading_account_read_model,
    initialize_trading_account_state,
    load_trading_account_state,
    resolve_trading_account_state_path,
    set_trading_cash_balance,
)

def _publish_synthetic_trading_input_lineage(
    root: Path,
    *,
    market_date: str,
    required_position_tickers=(),
    current_universe_tickers=None,
):
    """Publish canonical V2 Trading data/param lineage for isolated synthetic fixtures."""
    from datetime import datetime, timedelta

    from core.data_utils import discover_unique_csv_inputs
    from core.file_integrity import canonical_json_sha256
    from core.market_data_bootstrap_requests import (
        BootstrapHttpRequest,
        BootstrapRequestManifest,
        build_registry_fingerprint,
    )
    from core.market_data_dataset_registry import (
        BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
        BOOTSTRAP_SINGLE_FULL_RANGE,
        BOOTSTRAP_SINGLE_NO_DATES,
        get_market_dataset_spec,
        get_market_dataset_specs,
    )
    from core.market_data_dataset_readiness import market_data_contract_requires_target_freshness
    from core.market_data_freshness_contract import get_market_data_freshness_contract
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from services.trading.market_data_consumer import publish_trading_v2_consumer_state
    from services.trading.market_data_dataset_state import (
        VALIDATION_STATUS_READY,
        build_initial_market_data_dataset_state,
        publish_market_data_dataset_state,
    )
    from services.trading.market_data_v2_view import TradingMarketDataV2View
    from services.trading.strategy_param_state import publish_trading_strategy_param_binding
    from services.trading.scanner_state import load_trading_scanner_runtime
    from .synthetic_market_data_cases import _publish_ready_provider_snapshot_fixture

    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    csv_inputs, duplicate_issue_lines = discover_unique_csv_inputs(Path(paths.data_dir))
    if duplicate_issue_lines:
        raise RuntimeError(
            "Synthetic Trading fixture 存在同 ticker 重複 CSV："
            + " | ".join(duplicate_issue_lines)
        )
    if not csv_inputs:
        raise RuntimeError("Synthetic Trading fixture 必須先建立至少一個 canonical Trading CSV")

    csv_by_ticker = {str(ticker): Path(path) for ticker, path in csv_inputs}
    all_tickers = sorted(csv_by_ticker)
    if current_universe_tickers is None:
        current_universe_tickers = all_tickers
    current_universe = sorted({normalize_trading_ticker(item) for item in current_universe_tickers})
    unknown_current = sorted(set(current_universe) - set(all_tickers))
    if unknown_current:
        raise RuntimeError(f"Synthetic Trading current universe 缺 CSV fixture: {unknown_current}")

    price_adj_parts = []
    raw_price_parts = []
    trading_dates = {str(market_date)}
    for ticker, path in sorted(csv_by_ticker.items()):
        source = pd.read_csv(path)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = sorted(required.difference(source.columns))
        if missing:
            raise RuntimeError(f"Synthetic Trading CSV 缺欄位: {ticker} {missing}")
        dates = pd.to_datetime(source["Date"], errors="raise").dt.strftime("%Y-%m-%d")
        trading_dates.update(dates.tolist())
        execution_volume = [
            10_000_000_000 if ticker in current_universe and date == str(market_date) else 1
            for date in dates.tolist()
        ]
        price_adj_parts.append(pd.DataFrame({
            "date": dates,
            "stock_id": ticker,
            "open": source["Open"].astype(float),
            "max": source["High"].astype(float),
            "min": source["Low"].astype(float),
            "close": source["Close"].astype(float),
            "Trading_Volume": execution_volume,
        }))
        raw_price_parts.append(pd.DataFrame({
            "date": dates,
            "stock_id": ticker,
            "Trading_Volume": source["Volume"],
        }))

    provider_frames = {
        "TaiwanStockPriceAdj": pd.concat(price_adj_parts, ignore_index=True),
        "TaiwanStockPrice": pd.concat(raw_price_parts, ignore_index=True),
        "TaiwanStockMarketValue": pd.DataFrame({
            "date": [str(market_date)] * len(all_tickers),
            "stock_id": all_tickers,
            "market_value": [10_000_000_000_000 if ticker in current_universe else 1 for ticker in all_tickers],
        }),
        "TaiwanStockTradingDate": pd.DataFrame({"date": sorted(trading_dates)}),
        "TaiwanStockInfo": pd.DataFrame({
            "date": [str(market_date)] * len(all_tickers),
            "stock_id": all_tickers,
            "type": ["twse"] * len(all_tickers),
            "industry_category": ["ETF" if ticker.startswith("00") else "半導體業" for ticker in all_tickers],
        }),
        "TaiwanStockDelisting": pd.DataFrame({
            "date": pd.Series(dtype="string"),
            "stock_id": pd.Series(dtype="string"),
        }),
    }
    required_datasets = tuple(provider_frames)
    canonical_specs = get_market_dataset_specs(included_only=True)
    registry_fingerprint = build_registry_fingerprint(canonical_specs)
    provider_as_of = str(market_date)
    full_range_start = min(trading_dates)
    requests = []
    frames_by_request = {}
    for dataset in required_datasets:
        spec = get_market_dataset_spec(dataset)
        mode = spec.bootstrap_mode
        if mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE:
            for ticker in all_tickers:
                request = BootstrapHttpRequest(
                    dataset=dataset,
                    bootstrap_mode=mode,
                    data_id=ticker,
                    start_date=full_range_start,
                    end_date=provider_as_of,
                )
                requests.append(request)
                frame = provider_frames[dataset]
                frames_by_request[request.request_id] = frame.loc[
                    frame["stock_id"].astype("string").str.strip() == ticker
                ].reset_index(drop=True)
        elif mode == BOOTSTRAP_SINGLE_NO_DATES:
            request = BootstrapHttpRequest(dataset, mode, None, None, None)
            requests.append(request)
            frames_by_request[request.request_id] = provider_frames[dataset].reset_index(drop=True)
        elif mode == BOOTSTRAP_SINGLE_FULL_RANGE:
            request = BootstrapHttpRequest(
                dataset, mode, None, full_range_start, provider_as_of
            )
            requests.append(request)
            frames_by_request[request.request_id] = provider_frames[dataset].reset_index(drop=True)
        else:
            raise RuntimeError(
                f"Synthetic Trading V2 fixture 尚未支援 canonical bootstrap mode: {dataset} -> {mode}"
            )
    requests = tuple(requests)
    manifest_core = {
        "as_of_date": provider_as_of,
        "registry_fingerprint": registry_fingerprint,
        "requests": [request.request_id for request in requests],
        "fixture": "trading_v2_direct_consumer",
    }
    manifest_fingerprint = canonical_json_sha256(manifest_core)
    manifest = BootstrapRequestManifest(
        as_of_date=provider_as_of,
        full_range_start=full_range_start,
        registry_fingerprint=registry_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        historical_instrument_count=len(all_tickers),
        requests=requests,
    )
    started_at = datetime.now().astimezone()
    provider_payload, frame_reader = _publish_ready_provider_snapshot_fixture(
        root=root,
        manifest=manifest,
        manifest_fingerprint=manifest_fingerprint,
        frame_by_request=frames_by_request,
        owner_id="synthetic-trading-v2",
        started_at=started_at,
        finalized_at=(started_at + timedelta(seconds=len(requests) + 1)).isoformat(),
        missing_columns_message="synthetic Trading V2 fixture 缺欄位",
    )
    from services.market_data.daily_pit_universe import build_market_data_v2_daily_pit_universe

    build_market_data_v2_daily_pit_universe(
        root,
        snapshot_fingerprint=str(provider_payload["snapshot_fingerprint"]),
        frame_reader=frame_reader,
    )

    dataset_state = build_initial_market_data_dataset_state(updated_at=started_at)
    for dataset in required_datasets:
        row = dataset_state["datasets"][dataset]
        row.update({
            "status": "READY",
            "last_attempt_target_date": str(market_date),
            "last_success_target_date": str(market_date),
            "last_ready_target_date": str(market_date),
            "latest_data_date": str(market_date),
            "latest_expected_date": str(market_date),
            "schema_status": VALIDATION_STATUS_READY,
            "coverage_status": VALIDATION_STATUS_READY,
        })
        contract = get_market_data_freshness_contract(dataset)
        if market_data_contract_requires_target_freshness(contract, row):
            row["last_exact_ready_target_date"] = str(market_date)
    publish_market_data_dataset_state(root, dataset_state)

    view = TradingMarketDataV2View.open(root)
    publish_trading_v2_consumer_state(
        root,
        market_date=str(market_date),
        required_position_tickers=list(required_position_tickers),
        retained_training_tickers=all_tickers,
        view=view,
    )
    publish_trading_strategy_param_binding(root)
    return load_trading_scanner_runtime(root, verify_dataset_content=True)




def _mutate_synthetic_trading_v2_consumer_membership(root: Path, *, ticker: str = "9999"):
    """Change only V2 consumer-state membership while preserving the verified source view."""
    from core.file_integrity import canonical_json_sha256
    from services.trading.market_data_consumer import (
        TRADING_V2_CONSUMER_STATE_RELATIVE_PATH,
        load_trading_v2_consumer_state,
    )

    payload = dict(load_trading_v2_consumer_state(root, required=True, verify_current_view=False))
    required = sorted(set(payload.get("required_position_tickers") or []) | {normalize_trading_ticker(ticker)})
    retained = sorted(set(payload.get("retained_training_tickers") or []))
    execution = sorted(set(payload.get("current_execution_pool_tickers") or []))
    training = sorted(set(execution) | set(required) | set(retained))
    payload["required_position_tickers"] = required
    payload["training_tickers"] = training
    payload["training_ticker_count"] = len(training)
    payload["state_fingerprint"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "state_fingerprint"}
    )
    path = Path(root).resolve() / TRADING_V2_CONSUMER_STATE_RELATIVE_PATH
    atomic_write_json(path, payload)
    return payload


def _build_synthetic_candidate_snapshot_payload(root: Path, *, candidate_rows):
    """Build a schema-current Trading candidate fixture from canonical Params runtime identity.

    AI note: production Scanner annotates every candidate with immutable agreeing-voter
    Params lineage before persistence.  Synthetic fixtures that construct candidate rows
    directly must pass through the same identity seam instead of weakening snapshot
    validation or hard-coding current signatures.
    """
    from core.portfolio_ensemble import annotate_ensemble_candidate
    from services.trading.scanner_state import TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION, load_trading_scanner_runtime
    from services.trading.strategy_param_runtime import serialize_trading_candidate_member_params

    runtime = load_trading_scanner_runtime(root, verify_dataset_content=True)
    members = list(runtime.get("param_members") or [])
    normalized_rows = []
    for raw_row in list(candidate_rows or []):
        row = dict(raw_row)
        has_lineage = bool(
            str(row.get("ensemble_member_key") or "").strip()
            and str(row.get("params_signature") or "").strip()
            and isinstance(row.get("ensemble_member_params_by_key"), dict)
            and row.get("ensemble_member_params_by_key")
        )
        if not has_lineage:
            if len(members) != 1:
                raise RuntimeError(
                    "Synthetic candidate fixture 若使用 multi-member Params，必須明確提供 agreeing-voter lineage"
                )
            member = dict(members[0])
            member_key = str(member.get("member_key") or member.get("member_index") or member.get("seed") or "1")
            row = annotate_ensemble_candidate(
                row,
                member=member,
                params_obj=member["params_obj"],
                member_key=member_key,
            )
            row["ensemble_vote_count"] = 1
            row["ensemble_min_agree"] = 1
            row["ensemble_member_count"] = 1
            row["ensemble_member_keys"] = [member_key]
            row["ensemble_member_params_by_key"] = {member_key: member["params_obj"]}
            row["param_lineage_source"] = "current_selected_artifact"
        row["ensemble_member_params_by_key"] = serialize_trading_candidate_member_params(row)
        row.pop("params_obj", None)
        row.pop("_ensemble_context", None)
        normalized_rows.append(row)

    from core.portfolio_ensemble import build_ensemble_candidate_display_metrics

    return {
        "schema_version": TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        "runtime_domain": "trading",
        "strategy_id": runtime["profile"].strategy_id,
        "param_selector": runtime["profile"].param_selector,
        "latest_data_date": runtime["latest_data_date"],
        "param_latest_data_date": runtime["param_latest_data_date"],
        "param_member_count": runtime["member_count"],
        "param_min_agree": runtime["param_min_agree"],
        "candidate_display_metrics": build_ensemble_candidate_display_metrics(
            total_member_count=runtime["member_count"]
        ),
        "selected_params_sha256": runtime["selected_params_sha256"],
        "market_data_consumer_state_sha256": runtime["market_data_consumer_state_sha256"],
        "market_data_source_view_fingerprint": runtime["market_data_source_view_fingerprint"],
        "param_binding_sha256": runtime["param_binding_sha256"],
        "scanned_tickers": list(runtime.get("current_universe_tickers") or []),
        "candidate_rows": normalized_rows,
        "stale_candidate_rows_skipped": [],
    }



def validate_trading_account_state_contract_case(base_params):
    case_id = "TRADING_ACCOUNT_STATE"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_account')

    project_root = Path(__file__).resolve().parents[2]
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    check("trading_market_data_is_gitignored", True, "/data/trading/" in gitignore)
    check("trading_operational_state_is_gitignored", True, "/state/trading/" in gitignore)
    check("trading_ticker_identity_uses_single_canonical_normalizer", "2330", normalize_trading_ticker(" 2330 "))
    try:
        normalize_trading_ticker("23 30")
    except ValueError:
        embedded_space_rejected = True
    else:
        embedded_space_rejected = False
    check("trading_ticker_identity_rejects_embedded_whitespace_everywhere", True, embedded_space_rejected)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        state_path = resolve_trading_account_state_path(root)
        check("account_state_path_is_under_trading_state_root", str((root / "state" / "trading" / "account.json").resolve()), str(state_path.resolve()))
        check("account_service_path_reuses_core_state_path_owner", str(resolve_core_trading_account_state_path(root)), str(state_path))
        check("order_state_path_has_single_core_owner", str((root / "state" / "trading" / "orders.json").resolve()), str(resolve_core_trading_order_state_path(root).resolve()))
        check("fill_transaction_path_has_single_core_owner", str((root / "state" / "trading" / "fill_transaction.json").resolve()), str(resolve_core_trading_fill_transaction_path(root).resolve()))

        state = initialize_trading_account_state(root, cash=1_000_000)
        initial_cash_milli = money_to_milli(1_000_000)
        check("initialize_revision_zero", 0, state["revision"])
        check("initialize_cash_exact_milli", initial_cash_milli, state["cash_milli"])
        check("initialize_event_chain_starts_at_revision_zero", [0], [event["revision"] for event in state["events"]])

        state = set_trading_cash_balance(root, cash=1_100_000, expected_revision=state["revision"], note="synthetic reconciliation")
        check("cash_reconciliation_advances_one_revision", 1, state["revision"])
        check("cash_reconciliation_updates_exact_cash", money_to_milli(1_100_000), state["cash_milli"])

        try:
            set_trading_cash_balance(root, cash=1_200_000, expected_revision=0)
        except TradingAccountRevisionConflict:
            stale_rejected = True
        else:
            stale_rejected = False
        check("stale_revision_is_rejected", True, stale_rejected)

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
        check("manual_adoption_does_not_change_cash", cash_before_adoption, state["cash_milli"])
        check("manual_adoption_source_is_explicit", "manual_adopted", adopted["source"])
        check("manual_adoption_does_not_forge_strategy_state", None, adopted["strategy_management"]["position_state"])
        check("manual_adoption_management_is_unmanaged", "unmanaged", adopted["strategy_management"]["status"])

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
        check("duplicate_open_ticker_is_rejected", True, duplicate_rejected)

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
        check("manual_correction_does_not_change_cash", cash_before_correction, state["cash_milli"])
        check("manual_correction_updates_broker_qty", 1200, corrected["broker"]["qty"])
        check("manual_correction_updates_broker_cost_basis", money_to_milli(540_000), corrected["broker"]["remaining_cost_basis_milli"])
        check("manual_correction_does_not_create_strategy_state", None, corrected["strategy_management"]["position_state"])

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
        check("manual_remove_does_not_change_cash", cash_before_remove, state["cash_milli"])
        check("manual_remove_deletes_only_selected_position", False, "2454" in state["positions"])

        cash_before_buy = state["cash_milli"]
        expected_buy = build_buy_ledger_from_price(100, 1000, base_params)
        state = apply_confirmed_strategy_buy_fill(
            state,
            ticker="2317", qty=1000, buy_price=100, params=base_params,
            trade_date="2026-09-04", timestamp="2026-09-04T09:01:00+08:00",
            mutation_id="synthetic-account-buy", init_sl=90, init_trail=90,
            target_price=120, limit_price=105, entry_order_id="SYNTHETIC-ENTRY-2317",
        )
        atomic_write_json(resolve_trading_account_state_path(root), state)
        strategy_record = state["positions"]["2317"]
        check("strategy_buy_uses_exact_accounting_cash", cash_before_buy - expected_buy["net_buy_total_milli"], state["cash_milli"])
        check("strategy_buy_broker_cost_matches_canonical_position", strategy_record["broker"]["remaining_cost_basis_milli"], strategy_record["strategy_management"]["position_state"]["remaining_cost_basis_milli"])
        check("strategy_buy_management_is_active", "active", strategy_record["strategy_management"]["status"])

        pre_same_day = deepcopy(state)
        try:
            apply_confirmed_sell_fill(
                state, ticker="2317", qty=500, exec_price=110, params=base_params,
                trade_date="2026-09-04", timestamp="2026-09-04T10:00:00+08:00",
                mutation_id="synthetic-account-same-day-sell",
            )
        except ValueError:
            same_day_rejected = True
        else:
            same_day_rejected = False
        check("same_day_buy_sell_is_rejected", True, same_day_rejected)
        check("rejected_sell_does_not_mutate_persisted_revision", pre_same_day["revision"], load_trading_account_state(root)["revision"])

        try:
            apply_confirmed_sell_fill(
                state, ticker="2317", qty=2000, exec_price=110, params=base_params,
                trade_date="2026-09-05", timestamp="2026-09-05T10:00:00+08:00",
                mutation_id="synthetic-account-oversell",
            )
        except ValueError:
            oversell_rejected = True
        else:
            oversell_rejected = False
        check("oversell_is_rejected", True, oversell_rejected)
        check("oversell_rejection_keeps_revision", state["revision"], load_trading_account_state(root)["revision"])

        broker_before = deepcopy(state["positions"]["2317"]["broker"])
        cash_before_sell = state["cash_milli"]
        expected_sell = build_sell_ledger_from_price(110, 500, base_params, ticker="2317", trade_date="2026-09-05")
        allocated = allocate_cost_basis_milli(broker_before["remaining_cost_basis_milli"], broker_before["qty"], 500)
        state = apply_confirmed_sell_fill(
            state, ticker="2317", qty=500, exec_price=110, params=base_params,
            trade_date="2026-09-05", timestamp="2026-09-05T10:01:00+08:00",
            mutation_id="synthetic-account-strategy-sell",
        )
        atomic_write_json(resolve_trading_account_state_path(root), state)
        broker_after = state["positions"]["2317"]["broker"]
        strategy_after = state["positions"]["2317"]["strategy_management"]["position_state"]
        check("sell_cash_uses_canonical_net_proceeds", cash_before_sell + expected_sell["net_sell_total_milli"], state["cash_milli"])
        check("sell_cost_basis_uses_canonical_allocation", broker_before["remaining_cost_basis_milli"] - allocated, broker_after["remaining_cost_basis_milli"])
        check("strategy_and_broker_qty_remain_equal_after_sell", broker_after["qty"], strategy_after["qty"])
        check("strategy_and_broker_cost_remain_equal_after_sell", broker_after["remaining_cost_basis_milli"], strategy_after["remaining_cost_basis_milli"])

        manual_before = deepcopy(state["positions"]["2330"]["broker"])
        cash_before_manual_sell = state["cash_milli"]
        manual_sell = build_sell_ledger_from_price(550, 200, base_params, ticker="2330", trade_date="2026-09-05")
        manual_allocated = allocate_cost_basis_milli(manual_before["remaining_cost_basis_milli"], manual_before["qty"], 200)
        state = apply_confirmed_sell_fill(
            state, ticker="2330", qty=200, exec_price=550, params=base_params,
            trade_date="2026-09-05", timestamp="2026-09-05T10:02:00+08:00",
            mutation_id="synthetic-account-manual-sell",
        )
        atomic_write_json(resolve_trading_account_state_path(root), state)
        manual_after = state["positions"]["2330"]["broker"]
        check("manual_position_sell_uses_canonical_net_proceeds", cash_before_manual_sell + manual_sell["net_sell_total_milli"], state["cash_milli"])
        check("manual_position_partial_cost_basis_is_allocated_canonically", manual_before["remaining_cost_basis_milli"] - manual_allocated, manual_after["remaining_cost_basis_milli"])
        check("manual_position_remains_unmanaged_after_broker_sell", "unmanaged", state["positions"]["2330"]["strategy_management"]["status"])
        revision_before_inventory_correction = int(state["revision"])
        canonical_before_inventory_correction = rebuild_trading_account_economics(
            state, accounting_params=build_standalone_trading_accounting_params()
        )
        realized_before_inventory_correction = int(
            canonical_before_inventory_correction["positions"]["2330"]["broker"].get("realized_pnl_milli") or 0
        )
        state = correct_existing_trading_position(
            root,
            ticker="2330",
            qty=800,
            cost_basis_total=400_000,
            entry_date="2026-08-02",
            expected_revision=state["revision"],
        )
        check("manual_position_with_sell_history_can_reconcile_current_inventory", 800, state["positions"]["2330"]["broker"]["qty"])
        check("inventory_reconciliation_advances_one_revision", revision_before_inventory_correction + 1, state["revision"])
        check("inventory_reconciliation_preserves_realized_pnl_history", realized_before_inventory_correction, int(state["positions"]["2330"]["broker"].get("realized_pnl_milli") or 0))

        validate_trading_account_state(state)
        expected_revisions = list(range(state["revision"] + 1))
        check("event_revisions_are_contiguous", expected_revisions, [event["revision"] for event in state["events"]])
        check("event_hash_chain_ends_at_current_revision", state["revision"], state["events"][-1]["revision"])

        tampered = deepcopy(state)
        tampered["events"][1]["details"]["cash_milli"] += 1
        try:
            validate_trading_account_state(tampered)
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        check("event_history_tampering_is_rejected", True, tamper_rejected)

        read_model = get_trading_account_read_model(root)
        check("read_model_revision_matches_state", state["revision"], read_model["revision"])
        check("read_model_position_count_matches_open_positions", len(state["positions"]), read_model["position_count"])
        check("persisted_state_is_single_account_file", True, state_path.is_file())
        check("separate_positions_truth_file_is_not_created", False, (state_path.parent / "positions.json").exists())

    # AI: User-facing broker ledger uses actual 0.001425 (no strategy discount).
    with tempfile.TemporaryDirectory() as temp_dir:
        from services.trading.account_dashboard import build_trading_account_dashboard_read_model

        root = Path(temp_dir)
        state = initialize_trading_account_state(root, cash=1_000_000)
        account_params = build_standalone_trading_accounting_params()
        expected_buy = build_buy_ledger_from_price(100, 1000, account_params)
        state = record_manual_trading_buy(
            root, ticker="2330", qty=1000, price=100, trade_date="2026-09-01", expected_revision=state["revision"]
        )
        broker = state["positions"]["2330"]["broker"]
        check("manual_trade_buy_uses_un-discounted_broker_fee", money_to_milli(142), expected_buy["buy_fee_milli"])
        check("trading_buy_cash_ledger_equals_existing_net_ledger", expected_buy["net_buy_total_milli"], expected_buy["cash_buy_total_milli"])
        check("trading_buy_has_no_research_rebate_receivable", 0, expected_buy["buy_fee_rebate_receivable_milli"])
        check("manual_trade_buy_deducts_holding_cost_from_cash", money_to_milli(1_000_000) - expected_buy["net_buy_total_milli"], state["cash_milli"])
        check("manual_trade_buy_tracks_gross_consideration", expected_buy["gross_buy_milli"], broker["remaining_gross_buy_milli"])
        check("manual_trade_buy_tracks_fee_separately", expected_buy["buy_fee_milli"], broker["remaining_buy_fee_milli"])

        state = record_manual_trading_buy(
            root, ticker="2330", qty=500, price=110, trade_date="2026-09-02", expected_revision=state["revision"]
        )
        try:
            record_manual_trading_sell(
                root, ticker="2330", qty=100, price=120, trade_date="2026-09-02", expected_revision=state["revision"]
            )
        except ValueError:
            same_day_manual_rejected = True
        else:
            same_day_manual_rejected = False
        check("manual_trade_same_day_buy_sell_is_rejected", True, same_day_manual_rejected)

        expected_sell = build_sell_ledger_from_price(120, 500, account_params, ticker="2330", trade_date="2026-09-03")
        state = record_manual_trading_sell(
            root, ticker="2330", qty=500, price=120, trade_date="2026-09-03", expected_revision=state["revision"]
        )
        sell_details = state["events"][-1]["details"]
        check("manual_trade_sell_records_gross_consideration", expected_sell["gross_sell_milli"], sell_details["gross_sell_milli"])
        check("manual_trade_sell_records_fee", expected_sell["sell_fee_milli"], sell_details["sell_fee_milli"])
        check("manual_trade_sell_records_tax", expected_sell["tax_milli"], sell_details["tax_milli"])
        check("trading_sell_cash_ledger_equals_existing_net_ledger", expected_sell["net_sell_total_milli"], expected_sell["cash_sell_total_milli"])
        check("trading_sell_has_no_research_rebate_receivable", 0, expected_sell["sell_fee_rebate_receivable_milli"])
        check("manual_trade_sell_pnl_formula", int(sell_details["gross_sell_milli"]) - int(sell_details["sell_fee_milli"]) - int(sell_details["tax_milli"]) - int(sell_details["allocated_cost_milli"]), int(sell_details["realized_pnl_milli"]))
        try:
            record_manual_trading_buy(
                root, ticker="2330", qty=100, price=119, trade_date="2026-09-03", expected_revision=state["revision"]
            )
        except ValueError:
            same_day_rebuy_rejected = True
        else:
            same_day_rebuy_rejected = False
        check("manual_trade_same_day_sell_then_buy_is_rejected", True, same_day_rebuy_rejected)

        account_model = get_trading_account_read_model(root)
        holding = account_model["positions"][0]
        check("broker_inventory_average_price_excludes_buy_fee", True, abs(float(holding["average_cost"]) - 103.3333333333) < 1e-6)
        check("broker_inventory_holding_cost_includes_buy_fee", holding["remaining_cost_basis"], holding["holding_cost"])
        dashboard = build_trading_account_dashboard_read_model(root)
        check("account_dashboard_exposes_two_buy_detail_rows", 2, len(dashboard["buy_details"]))
        check("account_dashboard_exposes_sell_detail_row", 1, len(dashboard["sell_details"]))
        check("account_dashboard_sell_detail_exposes_offset_cost", True, dashboard["sell_details"][0]["offset_holding_cost"] > 0)
        check("account_dashboard_sell_detail_exposes_offset_gross", True, dashboard["sell_details"][0]["offset_gross_amount"] > 0)
        check("account_dashboard_sell_detail_exposes_offset_buy_fee", True, dashboard["sell_details"][0]["offset_buy_fee"] > 0)

        # Account MTM freshness is intentionally independent from the latest
        # fully-safe Scanner date.  If adjusted prices have a verified 9/15 row
        # while the execution consumer is still finalized at 9/14, the account
        # dashboard marks holdings at 9/15 without advancing Scanner lineage.
        mtm_dataset_state = {
            "datasets": {
                "TaiwanStockPriceAdj": {
                    "validation_contract_version": 3,
                    "schema_status": "READY",
                    "coverage_status": "READY",
                    "latest_data_date": "2026-09-15",
                    "last_ready_target_date": "2026-09-15",
                    "last_exact_ready_target_date": "2026-09-15",
                }
            }
        }
        with (
            patch(
                "services.trading.account_dashboard.load_trading_v2_consumer_state",
                return_value={"market_date": "2026-09-14"},
            ),
            patch(
                "services.trading.account_dashboard.load_market_data_dataset_state",
                return_value=mtm_dataset_state,
            ),
            patch(
                "services.trading.account_dashboard._current_close_by_ticker",
                return_value=({"2330": 121.0}, {}),
            ),
        ):
            mtm_dashboard = build_trading_account_dashboard_read_model(root)
        check("account_dashboard_mtm_date_can_advance_ahead_of_valid_scan_date", "2026-09-15", mtm_dashboard["market_date"])
        check("account_dashboard_preserves_separate_valid_scan_date", "2026-09-14", mtm_dashboard["scan_market_date"])

    # Historical edit/delete must rebuild account truth rather than reverse-mutating
    # only the latest position.  The Workbench may omit expected_revision so the
    # service resolves current truth after acquiring the shared mutation lock.
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        state = initialize_trading_account_state(root, cash=1_000_000)
        state = record_manual_trading_buy(
            root, ticker="2820", qty=22000, price=18.15, trade_date="2026-09-11", expected_revision=state["revision"]
        )
        buy_revision = int(state["revision"])
        state = record_manual_trading_sell(
            root, ticker="2820", qty=22000, price=17.55, trade_date="2026-09-14", expected_revision=state["revision"]
        )
        sell_revision = int(state["revision"])
        state = delete_trading_transaction(root, transaction_revision=sell_revision, expected_revision=None)
        check("historical_full_sell_delete_reopens_inventory", 22000, state["positions"]["2820"]["broker"]["qty"])

        root2 = Path(temp_dir) / "second"
        state2 = initialize_trading_account_state(root2, cash=1_000_000)
        state2 = record_manual_trading_buy(
            root2, ticker="2820", qty=22000, price=18.15, trade_date="2026-09-11", expected_revision=state2["revision"]
        )
        buy_revision2 = int(state2["revision"])
        state2 = record_manual_trading_sell(
            root2, ticker="2820", qty=22000, price=17.55, trade_date="2026-09-14", expected_revision=state2["revision"]
        )
        try:
            delete_trading_transaction(root2, transaction_revision=buy_revision2, expected_revision=None)
        except ValueError as exc:
            dependent_buy_delete_rejected = "賣出股數超過當時庫存" in str(exc)
        else:
            dependent_buy_delete_rejected = False
        check("historical_buy_delete_rejects_oversell_instead_of_inventing_inventory", True, dependent_buy_delete_rejected)
        persisted2 = load_trading_account_state(root2)
        check("historical_buy_delete_rejection_preserves_original_inventory", 0, int((persisted2["positions"].get("2820") or {}).get("broker", {}).get("qty") or 0))
        check("historical_buy_delete_does_not_append_phantom_inventory_event", 0, len([
            event for event in persisted2.get("events", [])
            if str(event.get("mutation_type") or "") == "historical_inventory_reconciliation"
        ]))

        root_partial = Path(temp_dir) / "partial-history"
        partial = initialize_trading_account_state(root_partial, cash=1_000_000)
        partial = record_manual_trading_buy(root_partial, ticker="2330", qty=100, price=10, trade_date="2026-09-01", expected_revision=None)
        partial = record_manual_trading_buy(root_partial, ticker="2330", qty=200, price=10, trade_date="2026-09-02", expected_revision=None)
        deleted_buy_revision = int(partial["revision"])
        partial = record_manual_trading_sell(root_partial, ticker="2330", qty=200, price=12, trade_date="2026-09-03", expected_revision=None)
        try:
            delete_trading_transaction(root_partial, transaction_revision=deleted_buy_revision, expected_revision=None)
        except ValueError as exc:
            partial_delete_rejected = "賣出股數超過當時庫存" in str(exc)
        else:
            partial_delete_rejected = False
        check("partial_historical_buy_delete_rejects_missing_inventory", True, partial_delete_rejected)

        root_cash = Path(temp_dir) / "cash-delete"
        cash_state = initialize_trading_account_state(root_cash, cash=1_000_000)
        cash_before_buy = int(cash_state["cash_milli"])
        cash_state = record_manual_trading_buy(
            root_cash, ticker="7795", qty=1000, price=489, trade_date="2026-09-11", expected_revision=None
        )
        cash_buy_revision = int(cash_state["revision"])
        check("buy_delete_regression_buy_reduces_cash", True, int(cash_state["cash_milli"]) < cash_before_buy)
        cash_state = delete_trading_transaction(
            root_cash, transaction_revision=cash_buy_revision, expected_revision=None
        )
        check("buy_delete_regression_restores_cash_exactly", cash_before_buy, int(cash_state["cash_milli"]))
        check("buy_delete_regression_removes_open_position", False, "7795" in cash_state["positions"])

        root3 = Path(temp_dir) / "third"
        state3 = initialize_trading_account_state(root3, cash=1_000_000)
        state3 = set_trading_cash_balance(root3, cash=900_000, expected_revision=state3["revision"])
        state3 = record_manual_trading_buy(
            root3, ticker="2330", qty=10, price=100, trade_date="2026-09-03", expected_revision=None
        )
        check("latest_revision_inside_lock_allows_user_intent_buy", 2, state3["revision"])

        # Scanner-origin BUY must persist immutable strategy params directly on the
        # position so next-day management never depends on a broker ENTRY order.
        from core.params_io import params_to_json_dict
        from core.portfolio_param_runtime import build_portfolio_params_signature
        from services.trading.strategy_param_runtime import (
            build_trading_candidate_strategy_lineage,
            resolve_trading_position_strategy_binding,
        )
        lineage_root = Path(temp_dir) / "strategy-lineage"
        lineage_state = initialize_trading_account_state(lineage_root, cash=1_000_000)
        params_payload = params_to_json_dict(base_params)
        candidate = {
            "ticker": "2317",
            "ensemble_member_key": "1",
            "params_signature": build_portfolio_params_signature(base_params),
            "ensemble_member_params_by_key": {"1": params_payload},
            "execution_plan_seed": {
                "init_sl": 90, "init_trail": 92, "target_price": 120,
                "limit_price": 100, "entry_atr": 5, "entry_type": "normal",
            },
        }
        lineage = build_trading_candidate_strategy_lineage(candidate)
        lineage_state = record_strategy_trading_buy(
            lineage_root, ticker="2317", qty=100, price=100, trade_date="2026-09-04",
            expected_revision=None, params=base_params,
            execution_plan_seed=candidate["execution_plan_seed"], strategy_lineage=lineage,
        )
        lineage_record = lineage_state["positions"]["2317"]
        check("scanner_strategy_buy_does_not_require_entry_broker_order", None, lineage_record["broker"].get("entry_order_id"))
        check("scanner_strategy_buy_persists_position_strategy_lineage", lineage["lineage_id"], (lineage_record.get("strategy_lineage") or {}).get("lineage_id"))
        binding = resolve_trading_position_strategy_binding(lineage_record, orders={"orders": {}})
        check("position_strategy_lineage_resolves_without_orders_json", "position_strategy_lineage", binding["source"])
        lineage_read = get_trading_account_read_model(lineage_root)
        check("account_read_model_prefers_position_lineage_key", f"POSITION:{lineage['lineage_id']}", lineage_read["positions"][0]["strategy_lineage_key"])
        rebuilt_lineage = rebuild_trading_account_economics(
            lineage_state, accounting_params=build_standalone_trading_accounting_params()
        )
        check("account_replay_preserves_position_strategy_lineage", lineage["lineage_id"], (rebuilt_lineage["positions"]["2317"].get("strategy_lineage") or {}).get("lineage_id"))
        lineage_state = record_manual_trading_sell(
            lineage_root, ticker="2317", qty=100, price=120, trade_date="2026-09-05", expected_revision=None
        )
        from services.trading.account_dashboard import _actual_outcome_metrics, build_trading_account_dashboard_read_model
        lineage_dashboard = build_trading_account_dashboard_read_model(lineage_root)
        lineage_closed = lineage_dashboard["closed_trades"][0]
        lineage_perf = lineage_dashboard["performance"][1]
        check_true("closed_trade_keeps_strategy_r_multiple_as_lineage_diagnostic", lineage_closed["r_mult"] is not None)
        check("performance_actual_closed_win_rate_uses_net_pnl_outcome", 100.0, lineage_perf["win_rate_pct"])
        check("performance_all_wins_has_no_observed_loss_r_unit", None, lineage_perf["expected_value_r"])
        check("performance_all_wins_has_no_fake_risk_reward_sentinel", None, lineage_perf["risk_reward_ratio"])
        empirical = _actual_outcome_metrics([{"pnl": 300.0}, {"pnl": -100.0}])
        check("performance_empirical_risk_unit_is_average_actual_loss", 100.0, empirical["actual_risk_unit"])
        check("performance_empirical_payoff_is_average_win_over_average_loss", 3.0, empirical["risk_reward_ratio"])
        check("performance_empirical_ev_r_is_mean_pnl_over_average_actual_loss", 1.0, empirical["expected_value_r"])
        check("performance_empirical_win_rate_counts_actual_positive_outcomes", 50.0, empirical["win_rate_pct"])
        loss_only = _actual_outcome_metrics([{"pnl": -80.0}, {"pnl": -120.0}])
        check("performance_all_losses_empirical_ev_is_minus_one_r", -1.0, loss_only["expected_value_r"])
        check("performance_all_losses_has_no_win_loss_payoff_ratio", None, loss_only["risk_reward_ratio"])

    # Account performance uses closed lifecycle economics and preserves open
    # stock count/cost even when a current market price is temporarily missing.
    with tempfile.TemporaryDirectory() as temp_dir:
        from services.trading.account_dashboard import build_trading_account_dashboard_read_model

        root = Path(temp_dir)
        perf_state = initialize_trading_account_state(root, cash=1_000_000)
        perf_state = adopt_existing_trading_position(
            root, ticker="2330", qty=100, cost_basis_total=10_000,
            entry_date="2026-09-01", expected_revision=None,
        )
        perf_state = correct_existing_trading_position(
            root, ticker="2330", qty=100, cost_basis_total=20_000,
            entry_date="2026-09-01", expected_revision=None,
        )
        dashboard_open = build_trading_account_dashboard_read_model(root)
        open_perf = dashboard_open["performance"][0]
        check("performance_missing_market_price_keeps_open_stock_count", 1, open_perf["stock_count"])
        check("performance_missing_market_price_keeps_open_cost", 20_000.0, open_perf["cost"])
        check("performance_open_inventory_has_no_fake_trade_win_rate", None, open_perf["win_rate_pct"])
        perf_state = record_manual_trading_sell(
            root, ticker="2330", qty=100, price=300, trade_date="2026-09-02", expected_revision=None
        )
        dashboard_closed = build_trading_account_dashboard_read_model(root)
        closed_perf = dashboard_closed["performance"][1]
        check("performance_closed_cost_consumes_broker_truth_correction", 20_000.0, closed_perf["cost"])
        check("performance_closed_row_uses_realized_pnl", dashboard_closed["closed_trades"][0]["pnl"], closed_perf["pnl"])

    # AI: Pause a real commit after its revision checks; another caller must
    # not accept the same revision and overwrite that pending commit.
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from services.trading.state_lock import TradingStateBusyError

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        initialize_trading_account_state(root, cash=100)
        writing, release = Event(), Event()

        # Coverage instrumentation can make the first worker substantially slower
        # than a normal suite run.  Do not start the competing mutation unless the
        # first worker has actually entered the patched commit window; otherwise
        # the test itself can create an unrelated stale-revision race.
        concurrency_timeout_seconds = 30

        def paused_write(path, payload):
            writing.set()
            if not release.wait(timeout=concurrency_timeout_seconds):
                raise RuntimeError("synthetic commit was not released")
            atomic_write_json(path, payload)

        with patch("services.trading.account_state.atomic_write_json", side_effect=paused_write):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(set_trading_cash_balance, root, cash=90, expected_revision=0)
                entered_commit_window = writing.wait(timeout=concurrency_timeout_seconds)
                try:
                    check("concurrent_cash_test_reaches_commit_window", True, entered_commit_window)
                    contention_rejected = False
                    rejection_type = None
                    if entered_commit_window:
                        try:
                            set_trading_cash_balance(root, cash=80, expected_revision=0)
                        except (TradingStateBusyError, TradingAccountRevisionConflict) as exc:
                            # The correctness invariant is no lost update.  In-process
                            # contention should normally fail at the lock boundary; on
                            # platforms where the first commit wins the race, an atomic
                            # stale-revision rejection is equally safe and must not crash
                            # the whole synthetic coverage suite.
                            contention_rejected = True
                            rejection_type = type(exc).__name__
                    check("concurrent_cash_change_rejects_before_lost_update", True, contention_rejected)
                    check(
                        "concurrent_cash_change_uses_safe_rejection_type",
                        True,
                        rejection_type in {"TradingStateBusyError", "TradingAccountRevisionConflict"},
                    )
                finally:
                    release.set()
                committed = pending.result(timeout=concurrency_timeout_seconds)
        actual = load_trading_account_state(root)
        check("concurrent_cash_change_keeps_one_committed_revision", 1, actual["revision"])
        check("concurrent_cash_change_preserves_first_commit", money_to_milli(90), actual["cash_milli"])
        check("concurrent_cash_change_preserves_event_chain", [0, 1], [row["revision"] for row in actual["events"]])
        next_state = set_trading_cash_balance(root, cash=80, expected_revision=committed["revision"])
        check("completed_commit_releases_lock_for_next_revision", 2, next_state["revision"])

    # AI: Trading account/orders/fill recovery are one in-process mutation domain.
    # The mutex must not depend on project-root path identity: Windows path aliases
    # or an accidentally different root object must never allow a second writer to
    # enter while another Trading commit is paused.
    with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
        root_a, root_b = Path(temp_dir_a), Path(temp_dir_b)
        initialize_trading_account_state(root_a, cash=100)
        initialize_trading_account_state(root_b, cash=200)
        root_a_state_path = resolve_trading_account_state_path(root_a).resolve()
        writing, release = Event(), Event()
        original_atomic_write_json = atomic_write_json

        def pause_root_a_only(path, payload):
            if Path(path).resolve() == root_a_state_path:
                writing.set()
                if not release.wait(timeout=concurrency_timeout_seconds):
                    raise RuntimeError("synthetic cross-root commit was not released")
            original_atomic_write_json(path, payload)

        with patch("services.trading.account_state.atomic_write_json", side_effect=pause_root_a_only):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(set_trading_cash_balance, root_a, cash=90, expected_revision=0)
                entered_commit_window = writing.wait(timeout=concurrency_timeout_seconds)
                try:
                    check("process_wide_cash_test_reaches_commit_window", True, entered_commit_window)
                    cross_root_rejected = False
                    if entered_commit_window:
                        try:
                            set_trading_cash_balance(root_b, cash=180, expected_revision=0)
                        except TradingStateBusyError:
                            cross_root_rejected = True
                    check("process_wide_mutex_rejects_second_project_root", True, cross_root_rejected)
                finally:
                    release.set()
                pending.result(timeout=concurrency_timeout_seconds)
        check("process_wide_mutex_preserves_first_project_commit", money_to_milli(90), load_trading_account_state(root_a)["cash_milli"])
        check("process_wide_mutex_leaves_second_project_unchanged", money_to_milli(200), load_trading_account_state(root_b)["cash_milli"])

    summary["checks"] = len(results)
    return results, summary



def validate_trading_daily_workflow_contract_case(base_params):
    case_id = "TRADING_DAILY_WORKFLOW"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_daily')

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    import services.trading.daily_workflow as daily_workflow

    profile = get_trading_strategy_profile()
    project_root = Path(__file__).resolve().parents[2]

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
        _publish_synthetic_trading_input_lineage(root, market_date="2026-09-04")
        snapshot = daily_workflow.build_trading_daily_workflow_snapshot(root)
        check("workflow_snapshot_reads_latest_trading_data", "2026-09-04", snapshot.get("latest_data_date"))
        check("workflow_snapshot_reads_param_latest_date", "2026-09-04", snapshot.get("param_latest_data_date"))
        check("workflow_snapshot_supports_single_member_identity", 1, snapshot.get("param_member_count"))
        check("workflow_snapshot_single_member_min_agree_is_one", 1, snapshot.get("param_min_agree"))
        check("workflow_snapshot_ready_when_data_and_params_match", True, snapshot.get("params_ready_for_scan"))
        check("workflow_snapshot_requires_canonical_v2_consumer_state", True, snapshot.get("market_data_ready"))
        check("workflow_snapshot_exposes_total_market_count_for_overview", 1, (snapshot.get("current_execution_pool_stats") or {}).get("listed_count"))
        check("workflow_snapshot_exposes_quick_filter_count_for_overview", 1, snapshot.get("current_execution_pool_ticker_count"))
        check("workflow_snapshot_exposes_market_data_source_view_identity", 64, len(str(snapshot.get("market_data_source_view_fingerprint") or "")))
        check("workflow_scanner_output_is_trading_scoped", "outputs/trading/scanner", snapshot.get("scanner_output_dir"))

        fake_scan = {
            "count_scanned": 1,
            "elapsed_time": 0.01,
            "count_history_qualified": 1,
            "count_skipped_insufficient": 0,
            "count_sanitized_candidates": 0,
            "max_workers": 1,
            "pool_start_method": "spawn",
            "scanned_tickers": ["2330"],
            "candidate_rows": [{
                "ticker": "2330",
                "kind": "buy",
                "sort_value": 1.5,
                "expected_value": 0.3,
                "proj_cost": 100000,
                "text": "synthetic candidate",
                "trade_date": "2026-09-04",
                "execution_plan_seed": {"ticker": "2330", "limit_price": 102.0, "init_sl": 98.0, "init_trail": 99.0, "target_price": 106.0, "entry_atr": 2.0, "trade_date": "2026-09-04"},
            }],
            "scanner_issue_log_path": None,
        }
        with patch.object(daily_workflow, "run_daily_scanner", return_value=fake_scan) as scanner_mock:
            scan_result = daily_workflow.run_trading_candidate_scan(project_root=root)
        scanner_args = scanner_mock.call_args
        check("trading_scanner_uses_v2_runtime_data_dir", str((root / "data" / "trading" / "market_data_v2").resolve()), str(Path(scanner_args.args[0]).resolve()))
        check("trading_scanner_injects_v2_prepared_frames", ["2330"], sorted((scanner_args.kwargs.get("prepared_frames") or {}).keys()))
        check("trading_scanner_injects_trading_output_dir", str((root / "outputs" / "trading" / "scanner").resolve()), str(Path(scanner_args.kwargs["output_dir"]).resolve()))
        check("trading_scanner_requests_execution_context_for_allocator", True, scanner_args.kwargs.get("include_execution_context"))
        check("trading_scanner_receives_canonical_current_universe_membership", ["2330"], scanner_args.kwargs.get("ticker_membership"))
        check("trading_scanner_reports_exact_scanned_membership", ["2330"], scan_result.get("scanned_tickers"))
        check("trading_scanner_returns_candidate_rows", 1, len(scan_result.get("candidate_rows") or []))
        check("trading_scanner_persists_candidate_snapshot", True, (root / "outputs" / "trading" / "scanner" / "candidate_snapshot.json").is_file())
        check("trading_scanner_carries_matching_data_date", "2026-09-04", scan_result.get("latest_data_date"))
        check("trading_scanner_carries_market_data_consumer_state_identity", 64, len(str(scan_result.get("market_data_consumer_state_sha256") or "")))
        check("trading_scanner_carries_param_binding_identity", 64, len(str(scan_result.get("param_binding_sha256") or "")))

        _write_param_payload("2026-09-03", 1)
        stale_snapshot = daily_workflow.build_trading_daily_workflow_snapshot(root)
        check("older_trading_params_are_reusable_but_not_implicitly_current", True, stale_snapshot.get("params_reusable"))
        check("older_trading_params_require_explicit_current_binding", False, stale_snapshot.get("params_ready_for_scan"))
        try:
            daily_workflow.run_trading_candidate_scan(project_root=root)
        except RuntimeError:
            stale_rejected = True
        else:
            stale_rejected = False
        check("older_trading_params_are_rejected_before_explicit_reuse_binding", True, stale_rejected)

        reuse_result = daily_workflow.reuse_trading_strategy_params(project_root=root)
        check("explicit_reuse_preserves_original_param_training_date", "2026-09-03", reuse_result.get("param_training_data_date"))
        check("explicit_reuse_binds_current_trading_data_date", "2026-09-04", reuse_result.get("latest_data_date"))
        check("explicit_reuse_records_usage_mode", "reuse_existing", reuse_result.get("param_usage_mode"))
        rebound_snapshot = daily_workflow.build_trading_daily_workflow_snapshot(root)
        check("explicit_reuse_makes_params_ready_for_scan", True, rebound_snapshot.get("params_ready_for_scan"))
        with patch.object(daily_workflow, "run_daily_scanner", return_value=fake_scan):
            reused_scan = daily_workflow.run_trading_candidate_scan(project_root=root)
        check("explicit_reuse_allows_scanner_without_retraining", "2026-09-04", reused_scan.get("latest_data_date"))

        _write_param_payload("2026-09-04", 2)
        from services.trading.strategy_param_state import publish_trading_strategy_param_binding
        publish_trading_strategy_param_binding(root)
        ensemble_snapshot = daily_workflow.build_trading_daily_workflow_snapshot(root)
        check("multi_member_selector_is_supported_by_trading_runtime", 2, ensemble_snapshot.get("param_member_count"))
        check("multi_member_selector_uses_canonical_auto_min_agree", 2, ensemble_snapshot.get("param_min_agree"))
        check("multi_member_selector_is_ready_for_scan", True, ensemble_snapshot.get("params_ready_for_scan"))
        ensemble_scan_a = dict(fake_scan)
        ensemble_scan_b = dict(fake_scan)
        with patch.object(daily_workflow, "run_daily_scanner", side_effect=[ensemble_scan_a, ensemble_scan_b]) as ensemble_scanner_mock:
            ensemble_scan = daily_workflow.run_trading_candidate_scan(project_root=root)
        check("multi_member_scanner_executes_each_finalist_member", 2, ensemble_scanner_mock.call_count)
        check("multi_member_scanner_applies_canonical_agreement", 1, len(ensemble_scan.get("candidate_rows") or []))
        ensemble_row = dict((ensemble_scan.get("candidate_rows") or [{}])[0])
        check("multi_member_candidate_records_vote_count", 2, ensemble_row.get("ensemble_vote_count"))
        check("multi_member_candidate_records_min_agree", 2, ensemble_row.get("ensemble_min_agree"))
        check("multi_member_candidate_keeps_representative_params_identity", True, bool(ensemble_row.get("params_signature")) and bool(ensemble_row.get("ensemble_member_key")))

    call_order = []
    with patch.object(daily_workflow, "run_trading_market_data_update", side_effect=lambda **_kwargs: call_order.append("data") or {"status": "READY"}), \
         patch("services.trading.position_rollforward.run_trading_position_rollforward", side_effect=lambda **_kwargs: call_order.append("rollforward") or {"status": "UP_TO_DATE"}), \
         patch("services.trading.indicator_exit_planning.build_trading_indicator_exit_plan", side_effect=lambda **_kwargs: call_order.append("indicator") or {"status": "PROPOSED_INDICATOR_EXIT", "exit_count": 0, "exits": []}), \
         patch.object(daily_workflow, "run_trading_strategy_param_training", side_effect=lambda **_kwargs: call_order.append("params") or {"status": "READY"}), \
         patch.object(daily_workflow, "run_trading_candidate_scan", side_effect=lambda **_kwargs: call_order.append("scanner") or {"status": "READY", "candidate_rows": []}):
        workflow_result = daily_workflow.run_trading_daily_workflow(project_root=project_root, environ={})
    check("daily_workflow_executes_data_rollforward_indicator_params_scanner_in_order", ["data", "rollforward", "indicator", "params", "scanner"], call_order)
    check("daily_workflow_returns_ready_only_after_all_steps", "READY", workflow_result.get("status"))

    reuse_call_order = []
    with patch.object(daily_workflow, "run_trading_market_data_update", side_effect=lambda **_kwargs: reuse_call_order.append("data") or {"status": "READY"}), \
         patch("services.trading.position_rollforward.run_trading_position_rollforward", side_effect=lambda **_kwargs: reuse_call_order.append("rollforward") or {"status": "UP_TO_DATE"}), \
         patch("services.trading.indicator_exit_planning.build_trading_indicator_exit_plan", side_effect=lambda **_kwargs: reuse_call_order.append("indicator") or {"status": "PROPOSED_INDICATOR_EXIT", "exit_count": 0, "exits": []}), \
         patch.object(daily_workflow, "reuse_trading_strategy_params", side_effect=lambda **_kwargs: reuse_call_order.append("reuse") or {"status": "READY"}), \
         patch.object(daily_workflow, "run_trading_candidate_scan", side_effect=lambda **_kwargs: reuse_call_order.append("scanner") or {"status": "READY", "candidate_rows": []}):
        reuse_workflow_result = daily_workflow.run_trading_daily_workflow(
            project_root=project_root, environ={}, param_mode=daily_workflow.TRADING_PARAM_MODE_REUSE
        )
    check("daily_workflow_reuse_mode_skips_param_retraining", ["data", "rollforward", "indicator", "reuse", "scanner"], reuse_call_order)
    check("daily_workflow_reuse_mode_returns_ready", "READY", reuse_workflow_result.get("status"))

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("workbench_exposes_separate_data_button", True, '"1 更新資料"' in panel_source)
    check("workbench_exposes_separate_param_button", True, '"2 套用 Params"' in panel_source)
    check("workbench_exposes_param_reuse_choice", True, '"沿用既有 Params"' in panel_source and '"重新訓練 Params"' in panel_source)
    check("workbench_uses_shared_downloader_console_progress", True, "MarketDataDailyConsoleProgress" in panel_source)
    check("workbench_exposes_scanner_button", True, '"3 Scanner 候選"' in panel_source)
    check("workbench_exposes_one_click_daily_sequence", True, '"每日流程 1→2→3"' in panel_source)
    check("workbench_long_workflow_uses_background_thread", True, "threading.Thread(" in panel_source)
    check("workbench_exposes_scanner_pool_without_broker_oms_in_primary_layout", True, "今日 Scanner Pool" in panel_source and "advanced_notebook.grid(" not in panel_source)
    check("workbench_overview_uses_data_center_style_kpi_cards", True, "_overview_vars" in panel_source and all(label in panel_source for label in ("同步狀態", "總股數", "符合快篩數", "Scanner 候選數", "剩餘可買數", "策略 / Params")))
    check("workbench_fixed_notes_move_to_fixed_bottom_status_bar", True, all(token in panel_source for token in ("self._footer_bar.grid(row=1", "操作提示｜", "_bind_footer_hint(workflow_box, WORKFLOW_HINT)", "_bind_footer_hint(candidate_box, SCANNER_HINT)", "_bind_footer_hint(trade_box, BUY_ENTRY_HINT)")))
    # AI: This contract is about preserving and rendering the original parameter
    # training date in reuse mode, not about freezing one historical UI phrase.
    # The runtime checks above already prove reuse keeps the original training date;
    # here we only verify that the Workbench overview consumes that canonical field.
    check(
        "workbench_reuse_mode_shows_original_param_training_date",
        True,
        'param_training_date = snapshot.get("param_training_data_date")' in panel_source
        and 'f"訓練至 {param_training_date} | DL {dl_state}"' in panel_source
        and 'self._set_overview_card("strategy_params", strategy_id, param_detail' in panel_source,
    )
    check("workbench_page_does_not_render_artifact_paths", False, any(token in panel_source for token in ("_path_var", "snapshot.get('text_path')", "snapshot.get('json_path')", "scanner_output_dir")))
    audit_body = panel_source.split("def _run_operational_audit", 1)[1].split("def _refresh_all_trading_state", 1)[0]
    check("workbench_operational_audit_dialog_does_not_render_report_path", False, "markdown_path" in audit_body or "report_path" in audit_body)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_actionable_universe_scanner_membership_contract_case(base_params):
    case_id = "TRADING_ACTIONABLE_UNIVERSE_SCANNER_MEMBERSHIP"
    results, summary, check, check_true = bind_synthetic_case(case_id, "trading_scanner_membership")

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    from core.file_integrity import load_json_strict
    from services.scanner import scan_runner
    import services.trading.daily_workflow as daily_workflow
    from services.trading.scanner_state import load_trading_candidate_snapshot

    profile = get_trading_strategy_profile()

    with tempfile.TemporaryDirectory(prefix="trading_scanner_membership_") as temp_dir:
        root = Path(temp_dir)
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
        data_dir = Path(paths.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        current_rows = {
            "Date": ["2026-09-03", "2026-09-04"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        }
        stale_rows = {
            "Date": ["2026-09-02", "2026-09-03"],
            "Open": [50.0, 51.0],
            "High": [52.0, 53.0],
            "Low": [49.0, 50.0],
            "Close": [51.0, 52.0],
            "Volume": [800, 850],
        }
        pd.DataFrame(current_rows).to_csv(data_dir / "2330.csv", index=False)
        pd.DataFrame(current_rows).to_csv(data_dir / "2317.csv", index=False)
        pd.DataFrame(stale_rows).to_csv(data_dir / "2454.csv", index=False)

        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(base_params)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        )
        selected_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        _publish_synthetic_trading_input_lineage(
            root,
            market_date="2026-09-04",
            current_universe_tickers=["2330", "2317"],
            required_position_tickers=["2454"],
        )

        all_inputs, _issues, all_count, all_tickers = scan_runner._prepare_scan_inputs(
            str(data_dir), base_params, output_dir=root / "outputs" / "all"
        )
        check("generic_scanner_default_keeps_full_dataset_membership", ["2317", "2330", "2454"], all_tickers)
        check("generic_scanner_default_scans_all_csv_files", 3, all_count)
        check("generic_scanner_default_paths_include_retained_historical_ticker", True, any(t == "2454" for t, _p in all_inputs))

        filtered_inputs, _issues, filtered_count, filtered_tickers = scan_runner._prepare_scan_inputs(
            str(data_dir),
            base_params,
            output_dir=root / "outputs" / "filtered",
            ticker_membership=["2330", "2317"],
        )
        check("membership_filter_scans_only_requested_tickers", ["2317", "2330"], filtered_tickers)
        check("membership_filter_excludes_retained_historical_ticker", False, any(t == "2454" for t, _p in filtered_inputs))
        check("membership_filter_count_matches_actionable_universe", 2, filtered_count)
        try:
            scan_runner._prepare_scan_inputs(
                str(data_dir),
                base_params,
                output_dir=root / "outputs" / "missing",
                ticker_membership=["2330", "9999"],
            )
        except FileNotFoundError as exc:
            missing_fail_closed = "9999" in str(exc)
        else:
            missing_fail_closed = False
        check("membership_filter_missing_csv_fails_closed", True, missing_fail_closed)

        generic_state = {
            "count_scanned": 2, "elapsed_time": 0.01, "count_history_qualified": 0,
            "count_skipped_insufficient": 0, "count_sanitized_candidates": 0,
            "max_workers": 1, "pool_start_method": "spawn", "rows": [],
            "scanned_tickers": ["2317", "2330"], "scanner_issue_log_path": None,
        }
        with patch.object(scan_runner, "_run_parallel_scan", return_value=generic_state) as parallel_mock, \
             patch.object(scan_runner, "print_scanner_summary", return_value=None):
            generic_result = scan_runner.run_daily_scanner(
                str(data_dir), base_params, ticker_membership=["2330", "2317"]
            )
        check("run_daily_scanner_forwards_optional_membership_to_scan_engine", ["2330", "2317"], parallel_mock.call_args.kwargs.get("ticker_membership"))
        check("run_daily_scanner_returns_actual_scanned_membership", ["2317", "2330"], generic_result.get("scanned_tickers"))

        fake_scan = {
            "count_scanned": 2,
            "elapsed_time": 0.01,
            "count_history_qualified": 1,
            "count_skipped_insufficient": 0,
            "count_sanitized_candidates": 0,
            "max_workers": 1,
            "pool_start_method": "spawn",
            "scanned_tickers": ["2317", "2330"],
            "candidate_rows": [{
                "ticker": "2330",
                "kind": "buy",
                "sort_value": 1.0,
                "expected_value": 0.2,
                "proj_cost": 100000,
                "text": "membership candidate",
                "trade_date": "2026-09-04",
                "execution_plan_seed": {
                    "ticker": "2330", "limit_price": 102.0, "init_sl": 98.0, "init_trail": 99.0,
                    "target_price": 106.0, "entry_atr": 2.0, "trade_date": "2026-09-04",
                },
            }],
            "scanner_issue_log_path": None,
        }
        with patch.object(daily_workflow, "run_daily_scanner", return_value=fake_scan) as scanner_mock:
            scan_result = daily_workflow.run_trading_candidate_scan(project_root=root)
        scanner_kwargs = scanner_mock.call_args.kwargs
        check("trading_candidate_scan_passes_snapshot_current_universe_only", ["2317", "2330"], scanner_kwargs.get("ticker_membership"))
        check("trading_candidate_scan_does_not_pass_required_holding_to_new_buy_scanner", False, "2454" in list(scanner_kwargs.get("ticker_membership") or []))
        check("trading_candidate_scan_persists_scanned_membership", ["2317", "2330"], scan_result.get("scanned_tickers"))
        check("retained_holding_csv_is_not_deleted_by_scanner_membership", True, (data_dir / "2454.csv").is_file())

        candidate_snapshot = load_trading_candidate_snapshot(root, require_current=True)
        check("candidate_snapshot_v4_binds_exact_scanned_membership", ["2317", "2330"], candidate_snapshot.get("scanned_tickers"))
        check("candidate_snapshot_candidate_is_inside_scanned_membership", "2330", candidate_snapshot["candidate_rows"][0]["ticker"])

        legacy_snapshot_path = root / "state" / "trading" / "market_data_snapshot.json"
        legacy_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_snapshot_path.write_text('{"obsolete": true}', encoding="utf-8")
        legacy_drift_snapshot = load_trading_candidate_snapshot(root, require_current=True)
        check("candidate_snapshot_ignores_retired_legacy_snapshot_file", ["2317", "2330"], legacy_drift_snapshot.get("scanned_tickers"))

        from services.trading.market_data_consumer import TRADING_V2_CONSUMER_STATE_RELATIVE_PATH
        consumer_state_path = root / TRADING_V2_CONSUMER_STATE_RELATIVE_PATH
        original_consumer_state = consumer_state_path.read_bytes()
        _mutate_synthetic_trading_v2_consumer_membership(root, ticker="9999")
        try:
            load_trading_candidate_snapshot(root, require_current=True)
        except RuntimeError as exc:
            market_membership_drift_rejected = "consumer state" in str(exc) or "market-data membership" in str(exc) or "dataset content" in str(exc)
        else:
            market_membership_drift_rejected = False
        check("candidate_snapshot_rejects_v2_consumer_state_drift", True, market_membership_drift_rejected)
        consumer_state_path.write_bytes(original_consumer_state)

        candidate_path = root / "outputs" / "trading" / "scanner" / "candidate_snapshot.json"
        tampered = load_json_strict(candidate_path)
        tampered["scanned_tickers"] = ["2330", "2454"]
        atomic_write_json(candidate_path, tampered)
        try:
            load_trading_candidate_snapshot(root, require_current=True)
        except (ValueError, RuntimeError) as exc:
            stale_membership_rejected = "membership" in str(exc) or "scanned" in str(exc)
        else:
            stale_membership_rejected = False
        check("candidate_snapshot_rejects_noncanonical_scanned_membership", True, stale_membership_rejected)

        bad_payload = _build_synthetic_candidate_snapshot_payload(root, candidate_rows=[{
            "ticker": "2454",
            "kind": "buy",
            "trade_date": "2026-09-04",
            "execution_plan_seed": {
                "ticker": "2454", "limit_price": 52.0, "init_sl": 48.0, "init_trail": 49.0,
                "target_price": 56.0, "entry_atr": 1.0, "trade_date": "2026-09-04",
            },
        }])
        try:
            from services.trading.scanner_state import _validate_trading_candidate_snapshot_payload
            _validate_trading_candidate_snapshot_payload(bad_payload)
        except ValueError as exc:
            outside_candidate_rejected = "membership" in str(exc)
        else:
            outside_candidate_rejected = False
        check("candidate_snapshot_rejects_candidate_outside_actionable_membership", True, outside_candidate_rejected)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_proposed_order_plan_contract_case(base_params):
    case_id = "TRADING_PROPOSED_ORDERS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_orders')

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
        _publish_synthetic_trading_input_lineage(root, market_date="2026-09-04", required_position_tickers=["2330"])
        starting_revision = int(state["revision"])
        starting_cash_milli = int(state["cash_milli"])

        candidate_snapshot_path = resolve_trading_candidate_snapshot_path(root)
        candidate_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_rows = [
            {
                "ticker": "2330",
                "trade_date": "2026-09-04",
                "kind": "extended_tbd",
                "sort_value": 3.0,
                "expected_value": 0.5,
                "execution_plan_seed": {"ticker": "2330", "limit_price": 101.0, "init_sl": 95.0, "init_trail": 96.0, "target_price": 107.0, "entry_atr": 2.0, "trade_date": "2026-09-04"},
            },
            {
                "ticker": "2454",
                "trade_date": "2026-09-04",
                "kind": "buy",
                "sort_value": 2.0,
                "expected_value": 0.4,
                "execution_plan_seed": {"ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0, "target_price": 210.0, "entry_atr": 4.0, "trade_date": "2026-09-04"},
            },
            {
                "ticker": "2603",
                "trade_date": "2026-09-04",
                "kind": "buy",
                "sort_value": 1.0,
                "expected_value": 0.3,
                "execution_plan_seed": {"ticker": "2603", "limit_price": 50.0, "init_sl": 47.0, "init_trail": 48.0, "target_price": 53.0, "entry_atr": 1.0, "trade_date": "2026-09-04"},
            },
        ]
        candidate_snapshot_path.write_text(
            json.dumps(_build_synthetic_candidate_snapshot_payload(root, candidate_rows=candidate_rows), ensure_ascii=False),
            encoding="utf-8",
        )

        plan = build_trading_proposed_order_plan(project_root=root)
        check("proposed_plan_does_not_mutate_account_revision", starting_revision, load_trading_account_state(root)["revision"])
        check("proposed_plan_does_not_mutate_account_cash", starting_cash_milli, load_trading_account_state(root)["cash_milli"])
        check("held_ticker_is_not_proposed_again", ["2330"], plan.get("held_tickers_skipped"))
        check("extended_tbd_is_resolved_against_actual_holdings", False, any(row.get("ticker") == "2330" for row in plan.get("orders") or []))
        check("proposed_orders_are_not_confirmed", False, plan.get("confirmed"))
        check("proposed_orders_do_not_claim_account_mutation", False, plan.get("account_mutated"))
        check("reserved_total_equals_order_sum", int(plan["reserved_total_milli"]), sum(int(row["reserved_cost_milli"]) for row in plan["orders"]))
        check("reservation_never_exceeds_cash", True, int(plan["reserved_total_milli"]) <= starting_cash_milli)
        check("reserved_cash_is_locked_across_proposed_orders", starting_cash_milli - int(plan["reserved_total_milli"]), int(plan["cash_after_reservation_milli"]))
        expected_liquidation = build_sell_ledger_from_price(adjust_long_sell_fill_price(100.0, ticker="2330"), 1000, base_params, ticker="2330", trade_date="2026-09-04")["net_sell_total_milli"]
        check("sizing_equity_uses_mark_to_market_net_liquidation", starting_cash_milli + expected_liquidation, int(round(float(plan["sizing_equity"]) * 1000)))
        check("proposed_plan_writes_machine_readable_output", True, (root / "outputs" / "trading" / "proposed_orders" / "proposed_orders.json").is_file())
        check("proposed_plan_writes_human_readable_output", True, (root / "outputs" / "trading" / "proposed_orders" / "proposed_orders.txt").is_file())
        check("proposed_plan_carries_account_revision", starting_revision, plan.get("account_revision"))
        check("proposed_plan_carries_candidate_snapshot_identity", compute_file_sha256(candidate_snapshot_path), plan.get("candidate_snapshot_sha256"))

        selected_path.write_text(selected_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        try:
            build_trading_proposed_order_plan(project_root=root)
        except RuntimeError:
            stale_snapshot_rejected = True
        else:
            stale_snapshot_rejected = False
        check("candidate_snapshot_is_rejected_after_param_artifact_changes", True, stale_snapshot_rejected)

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("workbench_exposes_proposed_order_button", True, '"4 建議掛單"' in panel_source)
    check("workbench_proposed_orders_use_background_thread", True, 'elif action == "orders"' in panel_source and "threading.Thread(" in panel_source)
    check("workbench_does_not_confirm_fill_from_proposal_action", False, "confirm_trading_strategy_buy_fill" in panel_source)

    summary["checks"] = len(results)
    return results, summary



def validate_trading_pending_order_state_contract_case(base_params):
    case_id = "TRADING_PENDING_ORDERS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_pending')

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.file_integrity import compute_file_sha256
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_order_state import validate_trading_order_state
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    from services.trading.daily_workflow import resolve_trading_candidate_snapshot_path
    from services.trading.order_planning import build_trading_proposed_order_plan
    from services.trading.entry_order_submission import confirm_trading_order_submission
    from services.trading.order_state import (
        TradingOrderRevisionConflict,
        confirm_trading_order_cancellation,
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
        _publish_synthetic_trading_input_lineage(root, market_date="2026-09-04")
        account_revision = int(account["revision"])
        account_cash_milli = int(account["cash_milli"])

        snapshot_path = resolve_trading_candidate_snapshot_path(root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_rows = [
            {
                "ticker": "2454",
                "trade_date": "2026-09-04",
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
                "trade_date": "2026-09-04",
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
        snapshot_path.write_text(
            json.dumps(_build_synthetic_candidate_snapshot_payload(root, candidate_rows=candidate_rows), ensure_ascii=False),
            encoding="utf-8",
        )

        plan = build_trading_proposed_order_plan(project_root=root)
        order_path = resolve_trading_order_state_path(root)
        check("proposal_only_does_not_create_order_state", False, order_path.exists())
        check("order_state_path_is_operational_trading_state", str((root / "state" / "trading" / "orders.json").resolve()), str(order_path.resolve()))
        check("proposed_schema_preserves_entry_type_for_future_fill", True, all(bool(row.get("entry_type")) for row in plan["orders"]))
        check("proposed_schema_preserves_security_profile_for_future_fill", True, all(isinstance(row.get("security_profile"), dict) for row in plan["orders"]))

        first = plan["orders"][0]
        blocked_submission_status = {
            "entry_submission_allowed": False,
            "entry_submission_blockers": ["synthetic canonical Operations block"],
        }
        with patch(
            "services.trading.entry_order_submission.build_trading_operations_status",
            return_value=blocked_submission_status,
        ):
            try:
                confirm_trading_order_submission(
                    root,
                    rank=int(first["rank"]),
                    ticker=first["ticker"],
                    expected_revision=None,
                )
            except RuntimeError as exc:
                submission_guarded = "synthetic canonical Operations block" in str(exc)
            else:
                submission_guarded = False
        check("entry_submission_rechecks_canonical_operations_safety_before_ordered_mutation", True, submission_guarded)
        check("blocked_entry_submission_does_not_create_order_state", False, order_path.exists())

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
        check("first_confirmed_submission_creates_revision_one", 1, ordered["revision"])
        check("confirmed_submission_status_is_ordered", "ORDERED", ordered_record["status"])
        check("confirmed_submission_records_proposed_to_ordered_transition", ["PROPOSED", "ORDERED"], [ordered["events"][-1]["details"]["from_status"], ordered["events"][-1]["details"]["to_status"]])
        check("broker_order_id_is_preserved", "SYN-001", ordered_record["broker_order_id"])
        check("order_freezes_plan_fingerprint", plan["plan_fingerprint"], ordered_record["plan_fingerprint"])
        check("order_freezes_account_revision", account_revision, ordered_record["account_revision"])
        check("order_freezes_param_sha", plan["selected_params_sha256"], ordered_record["selected_params_sha256"])
        check("order_freezes_candidate_sha", plan["candidate_snapshot_sha256"], ordered_record["candidate_snapshot_sha256"])
        check("order_submission_does_not_mutate_account_revision", account_revision, load_trading_account_state(root)["revision"])
        check("order_submission_does_not_mutate_account_cash", account_cash_milli, load_trading_account_state(root)["cash_milli"])

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
        check("same_proposal_cannot_be_marked_ordered_twice", True, duplicate_rejected)
        check("duplicate_rejection_does_not_advance_order_revision", ordered["revision"], load_trading_order_state(root)["revision"])

        # Primary Trading is not a broker OMS.  A legacy/compatibility ORDERED
        # record is evidence only: it must not freeze cash reconciliation or a
        # fresh allocation.  Actual account BUY/SELL fills, not hidden order
        # lifecycle state, own the session lock.
        reconciled_while_active = set_trading_cash_balance(
            root, cash=700_000, expected_revision=account_revision
        )
        check(
            "legacy_active_order_does_not_block_account_reconciliation",
            account_revision + 1,
            reconciled_while_active["revision"],
        )
        check(
            "legacy_active_order_reconciliation_updates_cash",
            money_to_milli(700_000),
            reconciled_while_active["cash_milli"],
        )

        replanned_while_active = build_trading_proposed_order_plan(project_root=root)
        check(
            "legacy_active_order_does_not_block_fresh_allocation",
            reconciled_while_active["revision"],
            replanned_while_active["account_revision"],
        )

        order_id = ordered_record["order_id"]
        try:
            confirm_trading_order_cancellation(root, order_id=order_id, expected_revision=0)
        except TradingOrderRevisionConflict:
            stale_order_revision_rejected = True
        else:
            stale_order_revision_rejected = False
        check("stale_order_revision_is_rejected", True, stale_order_revision_rejected)

        cancelled = confirm_trading_order_cancellation(
            root,
            order_id=order_id,
            expected_revision=ordered["revision"],
            note="synthetic broker cancelled",
        )
        cancelled_record = cancelled["orders"][order_id]
        check("cancel_transition_advances_one_revision", ordered["revision"] + 1, cancelled["revision"])
        check("cancelled_order_status", "CANCELLED", cancelled_record["status"])
        check("cancelled_order_has_timestamp", True, bool(cancelled_record["cancelled_at"]))
        check(
            "cancellation_does_not_mutate_account",
            [reconciled_while_active["revision"], reconciled_while_active["cash_milli"]],
            [load_trading_account_state(root)["revision"], load_trading_account_state(root)["cash_milli"]],
        )

        replanned_after_cancel = build_trading_proposed_order_plan(project_root=root)
        check(
            "legacy_cancelled_order_does_not_create_same_day_allocation_lock",
            reconciled_while_active["revision"],
            replanned_after_cancel["account_revision"],
        )

        reconciled_after_cancel = set_trading_cash_balance(
            root, cash=650_000, expected_revision=reconciled_while_active["revision"]
        )
        check(
            "account_mutation_remains_available_after_legacy_order_cancel",
            reconciled_while_active["revision"] + 1,
            reconciled_after_cancel["revision"],
        )

        try:
            confirm_trading_order_cancellation(root, order_id=order_id, expected_revision=cancelled["revision"])
        except ValueError:
            double_cancel_rejected = True
        else:
            double_cancel_rejected = False
        check("cancelled_order_cannot_be_cancelled_twice", True, double_cancel_rejected)

        tampered = load_trading_order_state(root)
        tampered["events"][-1]["details"]["ticker"] = "TAMPER"
        order_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        try:
            load_trading_order_state(root)
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        check("order_event_hash_tamper_is_rejected", True, tamper_rejected)

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    order_service_source = (project_root / "services" / "trading" / "order_state.py").read_text(encoding="utf-8")
    entry_submission_source = (project_root / "services" / "trading" / "entry_order_submission.py").read_text(encoding="utf-8")
    check("workbench_requires_explicit_order_submission_confirmation", True, 'text="確認選取已送單"' in panel_source and "confirm_trading_order_submission(" in panel_source)
    check("workbench_exposes_explicit_broker_cancellation_confirmation", True, 'text="確認選取剩餘委託已取消"' in panel_source and "confirm_trading_order_cancellation(" in panel_source)
    check("submission_cancel_service_does_not_own_fill_reconciliation", False, "confirm_trading_buy_order_fill(" in order_service_source or "apply_confirmed_strategy_buy_fill(" in order_service_source)
    check("generic_order_state_no_longer_owns_entry_submission_orchestration", False, "def confirm_trading_order_submission(" in order_service_source)
    check("entry_submission_consumes_operations_canonical_guard", True, "assert_trading_proposed_submission_allowed" in entry_submission_source and "build_trading_operations_status" in entry_submission_source)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_confirmed_fill_reconciliation_contract_case(base_params):
    case_id = "TRADING_CONFIRMED_FILLS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_fill')

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
    from services.trading.entry_order_submission import confirm_trading_order_submission
    from services.trading.order_state import (
        confirm_trading_order_cancellation,
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
        _publish_synthetic_trading_input_lineage(root, market_date="2026-09-04")
        snapshot_path = resolve_trading_candidate_snapshot_path(root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(_build_synthetic_candidate_snapshot_payload(root, candidate_rows=[{
                "ticker": "2454", "trade_date": "2026-09-04", "kind": "buy", "sort_value": 2.0, "expected_value": 0.4,
                "execution_plan_seed": {
                    "ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0,
                    "target_price": 210.0, "entry_atr": 4.0, "trade_date": "2026-09-04",
                    "security_profile": {"family": "stock"},
                },
            }]), ensure_ascii=False),
            encoding="utf-8",
        )
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
        check("ordered_freezes_param_payload", params_to_json_dict(base_params), ordered_record.get("frozen_params"))
        check("ordered_freezes_param_payload_hash", canonical_json_sha256(params_to_json_dict(base_params)), ordered_record.get("frozen_params_sha256"))

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
        account_params = overlay_trading_accounting_params(base_params)
        first_ledger = build_buy_ledger_from_price(199.0, first_qty, account_params)
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
                    fill_price=Decimal("199.0"),
                    trade_date="2026-09-05",
                    expected_order_revision=int(ordered["revision"]),
                    expected_account_revision=int(account["revision"]),
                )
            except RuntimeError as exc:
                interrupted = "synthetic crash" in str(exc)
            else:
                interrupted = False
        check("two_state_fill_commit_interruption_is_observable", True, interrupted)
        check("fill_transaction_journal_persists_after_interruption", True, tx_path.is_file())
        check("fill_transaction_recovery_completes_prepared_targets", True, recover_trading_fill_transaction(root))
        check("fill_transaction_journal_removed_after_recovery", False, tx_path.exists())

        partial_account = load_trading_account_state(root)
        partial_orders = load_trading_order_state(root)
        partial_record = partial_orders["orders"][order_id]
        check("first_fill_transitions_to_partial", "PARTIAL", partial_record["status"])
        check("partial_fill_updates_filled_qty", first_qty, partial_record["filled_qty"])
        check("partial_fill_keeps_remaining_qty_active", order_qty - first_qty, partial_record["remaining_qty"])
        check("partial_status_remains_active_order", True, partial_record["status"] in TRADING_ACTIVE_ORDER_STATUSES)
        check("partial_fill_debits_exact_cash", initial_cash - int(first_ledger["net_buy_total_milli"]), partial_account["cash_milli"])
        check("partial_fill_creates_actual_position", first_qty, partial_account["positions"]["2454"]["broker"]["qty"])
        check("partial_fill_binds_position_to_order", order_id, partial_account["positions"]["2454"]["broker"].get("entry_order_id"))
        check("fill_uses_frozen_params_after_current_artifact_changes", int(getattr(base_params, "high_len")), int(partial_record["frozen_params"]["high_len"]))

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
        check("broker_fill_above_buy_limit_is_rejected", True, above_limit_rejected)
        check("rejected_fill_does_not_advance_revisions", [before_account_rev, before_order_rev], [load_trading_account_state(root)["revision"], load_trading_order_state(root)["revision"]])

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
        check("partial_fills_for_one_order_cannot_cross_trade_dates", True, cross_day_partial_rejected)

        remaining = int(partial_record["remaining_qty"])
        second_ledger = build_buy_ledger_from_price(198.0, remaining, account_params)
        final = confirm_trading_buy_order_fill(
            root,
            order_id=order_id,
            fill_qty=remaining,
            fill_price=198.0,
            trade_date="2026-09-05",
            expected_order_revision=before_order_rev,
            expected_account_revision=before_account_rev,
        )
        check("final_fill_transitions_to_filled", "FILLED", final["status"])
        check("filled_order_has_zero_remaining_qty", 0, final["remaining_qty"])
        check("all_partial_fills_accumulate_position_qty", order_qty, final["account"]["positions"]["2454"]["broker"]["qty"])
        check("all_partial_fills_debit_sum_of_exact_ledgers", initial_cash - int(first_ledger["net_buy_total_milli"]) - int(second_ledger["net_buy_total_milli"]), final["account"]["cash_milli"])
        check("filled_order_is_no_longer_active", False, final["orders"]["orders"][order_id]["status"] in TRADING_ACTIVE_ORDER_STATUSES)

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
        check("partial_order_can_cancel_unfilled_remainder", "CANCELLED", cancelled_record["status"])
        check("partial_cancel_preserves_filled_qty", first_qty, cancelled_record["filled_qty"])
        check("partial_cancel_preserves_position_and_cash", [first_qty, partial_cash], [load_trading_account_state(root)["positions"]["2454"]["broker"]["qty"], load_trading_account_state(root)["cash_milli"]])

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("workbench_requires_explicit_broker_fill_confirmation", True, 'text="確認選取成交"' in panel_source and "confirm_trading_buy_order_fill" in panel_source and "confirm_trading_protection_sell_order_fill" in panel_source)
    check(
        "workbench_fill_inputs_are_actual_qty_price_and_date",
        True,
        all(token in panel_source for token in ("本次成交股數", "本次成交價", 'text="成交日"', "DatePickerField(")),
    )
    check("workbench_does_not_auto_infer_fill_from_market_bar", False, "t_low" in panel_source or "t_high" in panel_source or "execute_pre_market_entry_plan" in panel_source)

    summary["checks"] = len(results)
    return results, summary

def validate_trading_protection_plan_contract_case(base_params):
    case_id = "TRADING_PROTECTION_PLAN"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_protection')

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
            "information_date": "2026-09-03",
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

        check("protection_plan_status_is_proposed_only", PROTECTION_PLAN_STATUS, protection["status"])
        check("protection_plan_never_claims_broker_submission", PROTECTION_BROKER_STATUS, protection["broker_status"])
        check("protection_plan_does_not_mutate_account_revision", account_revision, load_trading_account_state(root)["revision"])
        check("protection_plan_does_not_mutate_order_revision", order_revision, load_trading_order_state(root)["revision"])
        check("manual_adopted_position_is_not_auto_managed", ["2330"], protection["manual_positions_skipped"])
        check("partial_fill_protects_only_confirmed_held_qty", first_fill_qty, row["position_qty"])
        check("stop_leg_covers_full_current_position", first_fill_qty, stop_leg["qty"])
        check("stop_leg_uses_canonical_effective_position_stop", int(canonical_position["sl_milli"]), int(stop_leg["trigger_price_milli"]))
        check("stop_leg_uses_stop_market_gap_semantics", "STOP_MARKET", stop_leg["order_type"])
        check("tp_leg_qty_uses_canonical_half_take_profit_rule", expected_tp_qty, tp_leg["qty"])
        check("tp_leg_uses_canonical_position_target", int(canonical_position["tp_half_milli"]), int(tp_leg["limit_price_milli"]))
        check("same_bar_stop_has_priority_over_tp", PROTECTION_SAME_BAR_PRIORITY, row["same_bar_priority"])
        check("protection_plan_uses_no_post_fill_market_data", False, protection["market_data_used"])
        check("protection_plan_uses_no_discretionary_input", False, protection["discretionary_input_used"])
        check("protection_json_is_output_not_operational_state", str((root / "outputs" / "trading" / "protection_orders" / "protection_plan.json").resolve()), str(resolve_trading_protection_plan_json_path(root).resolve()))
        check("protection_human_summary_is_persisted", True, resolve_trading_protection_plan_text_path(root).is_file())
        first_read = get_trading_protection_plan_read_model(root)
        check("newly_built_protection_plan_is_fresh", True, first_read["fresh"])

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
        check("additional_partial_fill_invalidates_old_protection_source_binding", False, stale_read["fresh"])
        rebuilt = build_trading_protection_plan(root)
        rebuilt_row = rebuilt["positions"][0]
        rebuilt_position = final["account"]["positions"]["2454"]["strategy_management"]["position_state"]
        check("final_fill_rebuilds_stop_for_full_confirmed_position", qty, rebuilt_row["position_qty"])
        check("rebuilt_stop_follows_weighted_average_fill_position_state", int(rebuilt_position["sl_milli"]), int(rebuilt_row["effective_stop_milli"]))
        check("rebuilt_target_follows_weighted_average_fill_position_state", int(rebuilt_position["tp_half_milli"]), int(rebuilt_row["target_price_milli"]))
        check("source_change_produces_new_protection_fingerprint", True, rebuilt["plan_fingerprint"] != protection["plan_fingerprint"])
        check("rebuilt_plan_returns_to_fresh", True, get_trading_protection_plan_read_model(root)["fresh"])

    # Primary non-OMS path: Scanner position lineage alone must be sufficient for
    # Stop/TP decision planning; orders.json is optional compatibility state.
    with tempfile.TemporaryDirectory() as temp_dir:
        from core.params_io import params_to_json_dict
        from core.portfolio_param_runtime import build_portfolio_params_signature
        from services.trading.strategy_param_runtime import build_trading_candidate_strategy_lineage

        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        params_payload = params_to_json_dict(base_params)
        candidate = {
            "ticker": "2317", "ensemble_member_key": "1",
            "params_signature": build_portfolio_params_signature(base_params),
            "ensemble_member_params_by_key": {"1": params_payload},
            "execution_plan_seed": {
                "init_sl": 90, "init_trail": 92, "target_price": 120,
                "limit_price": 100, "entry_atr": 5, "entry_type": "normal",
            },
        }
        lineage = build_trading_candidate_strategy_lineage(candidate)
        account = record_strategy_trading_buy(
            root, ticker="2317", qty=100, price=100, trade_date="2026-09-04",
            expected_revision=None, params=base_params,
            execution_plan_seed=candidate["execution_plan_seed"], strategy_lineage=lineage,
        )
        check("direct_position_lineage_does_not_create_orders_state", False, resolve_trading_order_state_path(root).exists())
        direct_plan = build_trading_protection_plan(root)
        direct_row = direct_plan["positions"][0]
        check("direct_position_lineage_builds_protection_without_entry_order", f"POSITION:{lineage['lineage_id']}", direct_row["entry_order_id"])
        check("direct_position_lineage_has_no_legacy_entry_order_binding", None, direct_row.get("legacy_entry_order_id"))
        check("direct_position_lineage_protection_plan_is_fresh", True, get_trading_protection_plan_read_model(root)["fresh"])

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    service_source = (project_root / "services" / "trading" / "protection_planning.py").read_text(encoding="utf-8")
    check("workbench_exposes_post_fill_protection_plan", True, "成交後 Stop / TP 保護單計畫" in panel_source and "build_trading_protection_plan(" in panel_source)
    check("workbench_explicitly_marks_protection_as_not_submitted", True, "尚未送券商" in panel_source)
    check("protection_planner_does_not_own_broker_submission", False, "confirm_trading_order_submission(" in service_source or "append_ordered_trading_proposal(" in service_source)
    check("protection_planner_does_not_use_market_bar_fill_inference", False, any(token in service_source for token in ("t_high", "t_low", "execute_pre_market_entry_plan")))

    summary["checks"] = len(results)
    return results, summary



def validate_trading_protection_order_submission_contract_case(base_params):
    case_id = "TRADING_PROTECTION_ORDER_SUBMISSION"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_protection_orders')

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
    from services.trading.operations_status import derive_trading_operations_status
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
            "information_date": "2026-09-03",
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
        check("protection_plan_starts_fresh", True, get_trading_protection_plan_read_model(root)["fresh"])

        stop_only = confirm_trading_protection_leg_submission(
            root, ticker="2454", action="STOP_FULL", expected_order_revision=int(filled["order_revision"]),
            broker_order_id="R11-STOP-1",
        )
        stop_record = next(row for row in stop_only["orders"].values() if row.get("side") == TRADING_ORDER_SIDE_SELL)
        check("single_stop_submission_is_sell_ordered", ["SELL", "PROTECTION_STOP", "ORDERED"], [stop_record["side"], stop_record["purpose"], stop_record["status"]])
        check("single_stop_uses_canonical_stop_market", "STOP_MARKET", stop_record["order_type"])
        check("single_stop_covers_full_held_qty", qty, stop_record["qty"])
        check("protection_submission_does_not_mutate_account", account_revision, load_trading_account_state(root)["revision"])
        check("broker_order_id_is_preserved_for_protection_sell", "R11-STOP-1", stop_record["broker_order_id"])
        check("protection_plan_remains_fresh_after_sell_submission", True, get_trading_protection_plan_read_model(root)["fresh"])

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
        check("non_oco_stop_plus_tp_cannot_overcommit_position_qty", True, tp_overcommit_rejected)
        check("rejected_overcommit_does_not_advance_order_revision", before_reject_revision, load_trading_order_state(root)["revision"])

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
        check("oco_is_never_assumed_without_explicit_broker_group", True, oco_requires_explicit_group)

        oco = confirm_trading_protection_oco_submission(
            root, ticker="2454", expected_order_revision=int(cancelled["revision"]), broker_oco_group_id="R11-OCO-1",
            stop_broker_order_id="R11-STOP-2", tp_broker_order_id="R11-TP-2",
        )
        active_sell = active_trading_protection_orders(oco)
        purposes = {str(row.get("purpose")) for row in active_sell}
        check("explicit_oco_creates_stop_and_tp_sell_orders", {TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_TP}, purposes)
        check("oco_group_submission_is_single_revision_mutation", int(cancelled["revision"]) + 1, int(oco["revision"]))
        check("all_oco_legs_preserve_user_confirmed_group", {"R11-OCO-1"}, {row.get("broker_oco_group_id") for row in active_sell})
        check("all_oco_legs_require_explicit_native_oco_confirmation", {True}, {bool(row.get("broker_native_oco_confirmed")) for row in active_sell})
        check("oco_effective_exposure_is_max_leg_not_sum", qty, max(int(row["qty"]) for row in active_sell))
        check("protection_sell_does_not_become_entry_buy_pending", 0, len(active_trading_entry_orders(oco)))
        check("two_oco_protection_legs_are_active", 2, len(active_sell))
        check("oco_submission_still_does_not_mutate_account", account_revision, load_trading_account_state(root)["revision"])

        current_account_model = {"initialized": True, **get_trading_account_read_model(root)}
        current_order_model = get_trading_order_read_model(root)
        current_protection_model = get_trading_protection_plan_read_model(root, recover_pending_fill=False)
        next_day_status = derive_trading_operations_status(
            workflow={"latest_data_date": "2026-09-05", "params_ready_for_scan": True},
            account=current_account_model,
            orders=current_order_model,
            candidate={"exists": True, "valid": True, "fresh": True, "candidate_count": 0, "information_date": "2026-09-04"},
            proposed={"exists": False, "valid": False, "fresh": False, "order_count": 0},
            protection=current_protection_model,
            indicator_exit={"exists": True, "fresh": True, "exit_count": 0, "exits": []},
            position_rollforward={"due_tickers": []},
        )
        check("long_lived_protection_sell_does_not_block_next_day_premarket_allocation", True, next_day_status["workflow_action_availability"]["orders"])

        entry_only_rows = [row for row in current_order_model["orders"] if row.get("side") == "BUY"]
        same_day_status = derive_trading_operations_status(
            workflow={"latest_data_date": "2026-09-03", "params_ready_for_scan": True},
            account={"initialized": True, "revision": 0, "cash": 1_000_000.0, "positions": [], "latest_buy_trade_date": "2026-09-03"},
            orders={"revision": current_order_model.get("revision"), "orders": entry_only_rows},
            candidate={"exists": True, "valid": True, "fresh": True, "candidate_count": 0, "information_date": "2026-09-03"},
            proposed={"exists": False, "valid": False, "fresh": False, "order_count": 0},
            protection={"exists": False, "fresh": False, "positions": [], "stale_active_protection_order_ids": [], "stale_active_protection_tickers": []},
            indicator_exit={"exists": False, "fresh": False, "exit_count": 0, "exits": []},
            position_rollforward={"due_tickers": []},
        )
        check("historical_entry_buy_still_locks_same_information_date_allocation", False, same_day_status["workflow_action_availability"]["orders"])

        read_model = get_trading_order_read_model(root)
        sell_rows = [row for row in read_model["orders"] if row.get("side") == "SELL"]
        check("order_read_model_exposes_protection_sell_side_and_purpose", True, all(row.get("purpose") in {TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_TP} for row in sell_rows))
        check("read_model_counts_active_protection_separately", 2, read_model["active_protection_order_count"])
        check("read_model_has_no_active_entry_buy_after_entry_filled", 0, read_model["active_entry_order_count"])

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
        check("protection_sell_cannot_use_buy_fill_reconciliation", True, sell_fill_rejected_by_buy_path)

        tampered = json.loads(json.dumps(oco))
        tamper_row = next(row for row in tampered["orders"].values() if row.get("side") == "SELL" and row.get("broker_native_oco_confirmed"))
        tamper_row["broker_oco_group_id"] = None
        try:
            validate_trading_order_state(tampered)
        except ValueError:
            oco_binding_tamper_rejected = True
        else:
            oco_binding_tamper_rejected = False
        check("oco_confirmed_group_binding_tamper_is_rejected", True, oco_binding_tamper_rejected)

    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    service_source = (project_root / "services" / "trading" / "protection_order_submission.py").read_text(encoding="utf-8")
    check("workbench_requires_explicit_stop_or_tp_submission_confirmation", True, "確認 Stop 已送單" in panel_source and "確認 TP 已送單" in panel_source)
    check("workbench_requires_explicit_broker_oco_group_confirmation", True, "確認 Stop+TP 已以券商 OCO 送單" in panel_source and "券商 OCO/互斥群組 ID" in panel_source)
    check("protection_submission_service_does_not_infer_market_fill_or_sell_execution", False, any(token in service_source for token in ("t_high", "t_low", "apply_confirmed_sell_fill", "confirm_trading_sell_fill")))

    summary["checks"] = len(results)
    return results, summary


def validate_trading_protection_sell_fill_reconciliation_contract_case(base_params):
    case_id = "TRADING_PROTECTION_SELL_FILL"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_protection_sell_fill')

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
    from services.trading.order_state import confirm_trading_order_cancellation, load_trading_order_state, resolve_trading_order_state_path
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
            "plan_fingerprint": "r12-entry-plan", "information_date": "2026-09-03",
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
        account_params = overlay_trading_accounting_params(base_params)
        expected_first_ledger = build_sell_ledger_from_price(111.0, first_qty, account_params, ticker="2454", security_profile={}, trade_date="2026-09-05")
        check("partial_tp_fill_sets_partial", "PARTIAL", first_record["status"])
        check("partial_tp_fill_reduces_remaining_qty", int(tp["qty"]) - first_qty, first_record["remaining_qty"])
        check("partial_tp_fill_reduces_real_held_qty", qty - first_qty, first["account"]["positions"]["2454"]["broker"]["qty"])
        check("partial_tp_does_not_mark_sold_half_complete", False, first["account"]["positions"]["2454"]["strategy_management"]["position_state"]["sold_half"])
        check("sell_fill_cash_credit_uses_canonical_exact_ledger", cash_before + int(expected_first_ledger["net_sell_total_milli"]), first["account"]["cash_milli"])
        check("native_oco_peer_remains_broker_truth_until_explicit_reconciliation", "ORDERED", peer["status"])
        check("native_oco_fill_reports_peer_that_requires_broker_reconciliation", [stop["order_id"]], first["oco_peer_reconciliation_order_ids"])
        check("sell_fill_advances_account_once", int(bought["account_revision"]) + 1, first["account_revision"])
        check("sell_fill_advances_orders_once", int(oco["revision"]) + 1, first["order_revision"])

        final_qty = int(first_record["remaining_qty"])
        second = confirm_trading_protection_sell_order_fill(
            root, order_id=tp["order_id"], fill_qty=final_qty, fill_price=112.0, trade_date="2026-09-06",
            expected_order_revision=int(first["order_revision"]), expected_account_revision=int(first["account_revision"]),
        )
        final_record = second["orders"]["orders"][tp["order_id"]]
        check("protection_sell_partial_fills_may_span_days", {"2026-09-05", "2026-09-06"}, {row["trade_date"] for row in final_record["fills"]})
        check("final_tp_fill_sets_filled", "FILLED", final_record["status"])
        check("final_tp_fill_zero_remaining", 0, final_record["remaining_qty"])
        check("completed_tp_marks_canonical_sold_half", True, second["account"]["positions"]["2454"]["strategy_management"]["position_state"]["sold_half"])
        check("completed_tp_leaves_half_position", qty - int(tp["qty"]), second["account"]["positions"]["2454"]["broker"]["qty"])
        check("native_oco_peer_still_not_inferred_cancelled_after_final_fill", "ORDERED", second["orders"]["orders"][stop["order_id"]]["status"])
        reconciled_peer = confirm_trading_order_cancellation(
            root, order_id=stop["order_id"], expected_revision=int(second["order_revision"]),
            note="synthetic explicit broker OCO peer cancellation reconciliation",
        )
        check("explicit_broker_peer_cancellation_is_separate_order_mutation", "CANCELLED", reconciled_peer["orders"][stop["order_id"]]["status"])
        validate_trading_order_state(reconciled_peer)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        qty = 100
        reserved = build_buy_ledger_from_price(100.0, qty, base_params)["net_buy_total_milli"]
        state = build_empty_trading_order_state(timestamp="2026-09-04T09:00:00+08:00", mutation_id="r12s-init")
        plan = {"plan_fingerprint":"r12s-plan","information_date":"2026-09-03","account_revision":0,"selected_params_sha256":canonical_json_sha256(frozen_params),"candidate_snapshot_sha256":"c","strategy_id":"s","param_selector":"p"}
        proposal = {"rank":1,"ticker":"2330","kind":"buy","entry_type":"normal","qty":qty,"limit_price":100.0,"reserved_cost_milli":int(reserved),"init_sl":90.0,"init_trail":90.0,"target_price":110.0,"entry_atr":5.0,"security_profile":{}}
        state = append_ordered_trading_proposal(state,order_id="r12s-entry",proposal=proposal,plan=plan,timestamp="2026-09-04T09:01:00+08:00",mutation_id="s",frozen_params=frozen_params)
        atomic_write_json(resolve_trading_order_state_path(root),state)
        bought = confirm_trading_buy_order_fill(root,order_id="r12s-entry",fill_qty=qty,fill_price=99.0,trade_date="2026-09-04",expected_order_revision=state["revision"],expected_account_revision=account["revision"])
        build_trading_protection_plan(root)
        stop_state = confirm_trading_protection_leg_submission(root,ticker="2330",action="STOP_FULL",expected_order_revision=bought["order_revision"],broker_order_id="R12S-STOP")
        stop = next(row for row in stop_state["orders"].values() if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP)
        stopped = confirm_trading_protection_sell_order_fill(root,order_id=stop["order_id"],fill_qty=qty,fill_price=85.0,trade_date="2026-09-05",expected_order_revision=stop_state["revision"],expected_account_revision=bought["account_revision"])
        check("stop_market_actual_fill_may_gap_below_trigger", "FILLED", stopped["status"])
        check("full_stop_fill_closes_position", False, "2330" in stopped["account"]["positions"])

    service_source = (Path(__file__).resolve().parents[2] / "services" / "trading" / "fill_reconciliation.py").read_text(encoding="utf-8")
    panel_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("sell_fill_service_does_not_infer_market_high_low", False, any(token in service_source for token in ("t_high", "t_low", "shadow_fill")))
    check("workbench_routes_sell_fill_to_sell_reconciliation", True, "confirm_trading_protection_sell_order_fill" in panel_source)
    summary["checks"] = len(results)
    return results, summary



def validate_trading_position_rollforward_contract_case(base_params):
    case_id = "TRADING_POSITION_ROLLFORWARD"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_rollforward')

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
    from services.trading.entry_order_submission import confirm_trading_order_submission
    from services.trading.order_state import load_trading_order_state
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
        dates = pd.bdate_range(end="2026-09-03", periods=needed)
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
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-03"}},
        ), ensure_ascii=False), encoding="utf-8")

        account = initialize_trading_account_state(root, cash=800_000)
        _publish_synthetic_trading_input_lineage(root, market_date="2026-09-03")
        snapshot_path = resolve_trading_candidate_snapshot_path(root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(_build_synthetic_candidate_snapshot_payload(root, candidate_rows=[{
                "ticker": "2454", "trade_date": "2026-09-03", "kind": "buy", "sort_value": 2.0, "expected_value": 0.4,
                "execution_plan_seed": {
                    "ticker": "2454", "limit_price": 200.0, "init_sl": 190.0, "init_trail": 192.0,
                    "target_price": 230.0, "entry_atr": 4.0, "trade_date": "2026-09-03",
                    "security_profile": {"family": "stock"},
                },
            }]), ensure_ascii=False),
            encoding="utf-8",
        )

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

        completed_next_day = pd.concat([frame, pd.DataFrame([{
            "Date": "2026-09-04", "Open": 199.0, "High": 260.0, "Low": 198.0, "Close": 250.0, "Volume": 1_000_000,
        }])], ignore_index=True)
        completed_next_day.to_csv(csv_path, index=False)

        changed = deepcopy(base_params)
        changed.atr_times_trail = float(base_params.atr_times_trail) + 5.0
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(changed)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")
        _publish_synthetic_trading_input_lineage(
            root,
            market_date="2026-09-04",
            current_universe_tickers=["2454"],
            required_position_tickers=["2454"],
        )

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
        check("pre_rollforward_active_stop_matches_current_position_plan", [], protection_before.get("stale_active_protection_order_ids"))

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
            check("confirmed_position_is_due_through_latest_completed_bar", ["2454"], due["due_tickers"])
            rolled = run_trading_position_rollforward(root)
            revision_after_first = int(rolled["account_revision"])
            second = run_trading_position_rollforward(root)

        account_after = load_trading_account_state(root)
        management_after = account_after["positions"]["2454"]["strategy_management"]
        position_after = management_after["position_state"]
        check("rollforward_uses_source_entry_order_frozen_params_not_current_selected_params", int(expected_position["sl_milli"]), int(position_after["sl_milli"]))
        check("rollforward_advances_highest_high_from_completed_bar", int(expected_position["highest_high_since_entry_milli"]), int(position_after["highest_high_since_entry_milli"]))
        check("rollforward_records_processed_completed_date", "2026-09-04", management_after.get("last_rollforward_date"))
        check("rollforward_does_not_change_broker_qty", qty_before, int(account_after["positions"]["2454"]["broker"]["qty"]))
        check("rollforward_does_not_change_cash", cash_before, int(account_after["cash_milli"]))
        check("rollforward_is_idempotent_after_completed_date_consumed", "UP_TO_DATE", second["status"])
        check("idempotent_second_run_does_not_advance_revision", revision_after_first, int(second["account_revision"]))

        protection_after = get_trading_protection_plan_read_model(root)
        check("old_active_stop_is_flagged_stale_after_trailing_state_changes", ["2454"], protection_after.get("stale_active_protection_tickers"))
        check("active_protection_order_is_not_silently_cancelled_or_rewritten", 1, len([row for row in load_trading_order_state(root)["orders"].values() if row.get("status") == "ORDERED" and row.get("side") == "SELL"]))

        future = pd.concat([completed_next_day, pd.DataFrame([{
            "Date": "2026-09-05", "Open": 251.0, "High": 270.0, "Low": 250.0, "Close": 265.0, "Volume": 1_000_000,
        }])], ignore_index=True)
        future.to_csv(csv_path, index=False)
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(changed)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-05"}},
        ), ensure_ascii=False), encoding="utf-8")
        _publish_synthetic_trading_input_lineage(
            root,
            market_date="2026-09-05",
            current_universe_tickers=["2454"],
            required_position_tickers=["2454"],
        )
        with patch("services.trading.position_rollforward.latest_allowed_completed_daily_date", return_value="2026-09-04"):
            future_cutoff = build_trading_position_rollforward_snapshot(root)
        check("rollforward_v2_reader_excludes_uncompleted_future_daily_bar_at_cutoff", [], future_cutoff["due_tickers"])

    capability = build_trading_capability_snapshot()
    check("daily_position_rollforward_capability_is_implemented", True, bool(capability["capabilities"]["daily_position_rollforward"]["implemented"]))
    check("indicator_sell_execution_capability_is_implemented", True, bool(capability["capabilities"]["indicator_sell_execution"]["implemented"]))

    service_source = (project_root / "services" / "trading" / "position_rollforward.py").read_text(encoding="utf-8")
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    daily_source = (project_root / "services" / "trading" / "daily_workflow.py").read_text(encoding="utf-8")
    snapshot_body = service_source.split("def build_trading_position_rollforward_snapshot", 1)[1].split("def run_trading_position_rollforward", 1)[0]
    check("read_only_rollforward_snapshot_does_not_run_fill_recovery", False, "recover_trading_fill_transaction(" in snapshot_body)
    check("rollforward_service_does_not_execute_or_infer_broker_sell", False, any(token in service_source for token in ("confirm_trading_sell_fill(", "confirm_trading_protection_sell_order_fill(", "t_low", "t_open")))
    check("workbench_exposes_explicit_position_rollforward_action", True, '"持股日終推進", "rollforward"' in panel_source and 'elif action == "rollforward"' in panel_source)
    workflow_step_tokens = (
        "data_result = run_trading_market_data_update",
        "rollforward_result = run_trading_position_rollforward",
        "indicator_result = build_trading_indicator_exit_plan",
        "param_result = run_trading_param_step",
    )
    workflow_step_positions = [daily_source.find(token) for token in workflow_step_tokens]
    check(
        "daily_workflow_orders_rollforward_and_indicator_after_data_before_param_step",
        True,
        all(position >= 0 for position in workflow_step_positions) and workflow_step_positions == sorted(workflow_step_positions),
    )

    summary["checks"] = len(results)
    return results, summary

def validate_trading_indicator_sell_execution_contract_case(base_params):
    case_id = "TRADING_INDICATOR_SELL_EXECUTION"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_indicator_sell')

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
    check("indicator_order_purpose_is_distinct",TRADING_ORDER_PURPOSE_INDICATOR_EXIT,row["purpose"] )
    check("indicator_order_is_market",TRADING_INDICATOR_ORDER_TYPE_MARKET,row["order_type"] )
    check("indicator_order_is_full_position",row["position_qty_at_submission"],row["qty"] )
    check("indicator_order_has_no_trigger",None,row["trigger_price_milli"] )
    check("indicator_order_has_no_limit",None,row["limit_price_milli"] )
    check("active_indicator_order_is_explicit",1,len(active_trading_indicator_exit_orders(ordered)) )
    cancelled=cancel_ordered_trading_order(ordered,order_id="IND-1",timestamp="2026-09-05T08:02:00+08:00",mutation_id="cancel")
    check("cancelled_indicator_can_leave_active_set",0,len(active_trading_indicator_exit_orders(cancelled)) )
    retry=append_ordered_trading_indicator_exit(cancelled,order_id="IND-2",exit_plan=exit_row,plan=plan,timestamp="2026-09-05T08:03:00+08:00",mutation_id="retry")
    check("cancelled_signal_retry_advances_attempt",2,retry["orders"]["IND-2"]["signal_attempt"] )
    capability=build_trading_capability_snapshot()
    check("all_required_live_capabilities_are_implemented",True,bool(capability.get("all_required_live_capabilities_ready")) )
    check("no_implementation_live_blocker_remains",[],list(capability.get("live_blocking_capabilities") or []) )
    project_root=Path(__file__).resolve().parents[2]
    planning=(project_root/"services/trading/indicator_exit_planning.py").read_text(encoding="utf-8")
    submit=(project_root/"services/trading/indicator_exit_order_submission.py").read_text(encoding="utf-8")
    fill=(project_root/"services/trading/fill_reconciliation.py").read_text(encoding="utf-8")
    protection=(project_root/"services/trading/protection_order_submission.py").read_text(encoding="utf-8")
    panel=(project_root/"services/workbench_ui/trading_account_panel.py").read_text(encoding="utf-8")
    check("planning_does_not_infer_broker_fill",False,"confirm_trading_indicator_sell_order_fill(" in planning )
    check("submission_requires_protection_cancellation",True,"active Stop/TP protection SELL" in submit )
    check("fill_reconciliation_uses_canonical_ind_sell_event",True,'event = "IND_SELL"' in fill )
    check("protection_submission_blocks_active_indicator_sell",True,"active Indicator MARKET SELL" in protection )
    check("workbench_exposes_indicator_market_sell_confirmation",True,"Indicator SELL 計畫" in panel and "確認選取 MARKET SELL 已送單" in panel and "confirm_trading_indicator_sell_order_fill" in panel )

    summary["checks"] = len(results)
    return results, summary

def validate_trading_operations_status_contract_case(base_params):
    case_id = "TRADING_OPERATIONS_STATUS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_operations')

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
        "current_execution_pool_ticker_count": 120,
        "current_execution_pool_stats": {"listed_count": 300, "qualified_count": 120},
    }
    account = {"initialized": True, "revision": 7, "cash": 500_000.0, "positions": []}
    orders = {"revision": 4, "orders": []}
    candidate = {
        "exists": True, "valid": True, "fresh": True, "candidate_count": 3,
        "candidate_tickers": ["1101", "2317", "2330"],
        "scanned_ticker_count": 120, "information_date": "2026-09-04",
    }
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

    overview_counts = derive()
    check("operations_status_exposes_total_market_count", 300, overview_counts["listed_ticker_count"])
    check("operations_status_exposes_quick_filter_count", 120, overview_counts["quick_filter_qualified_count"])
    check("operations_status_exposes_scanner_candidate_count", 3, overview_counts["scanner_candidate_ticker_count"])
    check("operations_status_exposes_scanner_buyable_count", 3, overview_counts["scanner_buyable_ticker_count"])
    check("operations_status_exposes_portfolio_free_slots", 10, overview_counts["portfolio_free_slot_count"])
    held_candidate_account = {
        "initialized": True, "revision": 8, "cash": 400_000.0,
        "positions": [{"ticker": "2317", "source": "strategy_fill", "qty": 100, "entry_order_id": "ENTRY-2317", "management_status": "active"}],
    }
    held_candidate_counts = derive(account=held_candidate_account)
    check("operations_status_buyable_excludes_held_candidate", 2, held_candidate_counts["scanner_unheld_candidate_ticker_count"])
    check("operations_status_buyable_uses_unheld_candidates", 2, held_candidate_counts["scanner_buyable_ticker_count"])
    near_full_account = {
        "initialized": True, "revision": 9, "cash": 400_000.0,
        "positions": [
            {"ticker": f"9{i:03d}", "source": "manual_adopted", "qty": 100}
            for i in range(9)
        ],
    }
    near_full_counts = derive(account=near_full_account)
    check("operations_status_buyable_respects_max_position_free_slots", 1, near_full_counts["portfolio_free_slot_count"])
    check("operations_status_buyable_is_bounded_by_free_slots", 1, near_full_counts["scanner_buyable_ticker_count"])
    stale_candidate_counts = derive(candidate={**deepcopy(candidate), "fresh": False})
    check("operations_status_buyable_requires_fresh_candidate_snapshot", None, stale_candidate_counts["scanner_buyable_ticker_count"])

    recovery = derive(fill_transaction_pending=True)
    check("pending_fill_transaction_is_top_priority_blocker", NEXT_RECOVER_FILL, recovery["next_action_code"])
    check("pending_fill_transaction_status_is_blocked", OPERATIONS_STATUS_BLOCKED, recovery["overall_status"])
    check("pending_fill_transaction_disables_all_workflow_actions", {"data": False, "rollforward": False, "params": False, "scanner": False, "orders": False, "all": False}, recovery["workflow_action_availability"])

    uninitialized = derive(account={"initialized": False, "revision": None, "cash": None, "positions": []})
    check("uninitialized_account_next_action", NEXT_INITIALIZE_ACCOUNT, uninitialized["next_action_code"])

    no_cash = derive(account={"initialized": True, "revision": 0, "cash": None, "positions": []})
    check("initialized_account_without_cash_next_action", NEXT_SET_CASH, no_cash["next_action_code"])
    check("allocation_disabled_without_cash", False, no_cash["workflow_action_availability"]["orders"])

    strategy_account = {
        "initialized": True,
        "revision": 8,
        "cash": 400_000.0,
        "positions": [{"ticker": "2317", "source": "strategy_fill", "qty": 100, "entry_order_id": "ENTRY-2317", "management_status": "active"}],
    }
    rollforward_due = derive(account=strategy_account, position_rollforward={"due_tickers": ["2317"]})
    check("due_strategy_position_requires_rollforward_before_protection_or_allocation", NEXT_ROLLFORWARD_POSITIONS, rollforward_due["next_action_code"])
    check("due_strategy_position_enables_explicit_rollforward_action", True, rollforward_due["workflow_action_availability"]["rollforward"])
    check("due_strategy_position_blocks_new_allocation_until_advanced", False, rollforward_due["workflow_action_availability"]["orders"])

    missing_stop = derive(account=strategy_account)
    check("strategy_position_without_stop_is_reported", ["2317"], missing_stop["missing_stop_tickers"])
    check("legacy_missing_stop_does_not_override_primary_trading_next_action", NEXT_BUILD_PROPOSED, missing_stop["next_action_code"])

    fresh_protection = {"exists": True, "fresh": True, "positions": [{"ticker": "2317"}]}
    missing_stop_fresh_plan = derive(account=strategy_account, protection=fresh_protection)
    check("legacy_fresh_protection_without_broker_stop_does_not_require_submission", NEXT_BUILD_PROPOSED, missing_stop_fresh_plan["next_action_code"])
    check("legacy_missing_stop_does_not_block_account_aware_allocation", True, missing_stop_fresh_plan["workflow_action_availability"]["orders"])

    active_stop_orders = {
        "revision": 5,
        "orders": [{
            "order_id": "stop1", "ticker": "2317", "side": "SELL", "purpose": "PROTECTION_STOP",
            "entry_order_id": "ENTRY-2317", "status": "ORDERED", "information_date": "2026-09-04",
        }],
    }
    protected = derive(account=strategy_account, orders=active_stop_orders, protection=fresh_protection)
    check("active_stop_clears_missing_stop_gap", [], protected["missing_stop_tickers"])
    check("long_lived_protection_sell_does_not_block_new_allocation", True, protected["workflow_action_availability"]["orders"])

    old_lineage_stop_history = {
        "revision": 6,
        "orders": [{
            "order_id": "old-stop", "ticker": "2317", "side": "SELL", "purpose": "PROTECTION_STOP",
            "entry_order_id": "ENTRY-OLD", "status": "CANCELLED", "filled_qty": 20, "remaining_qty": 80,
            "information_date": "2026-08-20",
        }],
    }
    old_lineage = derive(account=strategy_account, orders=old_lineage_stop_history)
    check("historical_stop_fill_from_old_entry_lineage_never_forces_new_position_exit", [], old_lineage["forced_stop_exit_tickers"])
    check("new_entry_lineage_without_stop_still_reports_missing_protection", ["2317"], old_lineage["missing_stop_tickers"])

    stale_protection = derive(
        account=strategy_account,
        orders=active_stop_orders,
        protection={**fresh_protection, "stale_active_protection_order_ids": ["stop1"], "stale_active_protection_tickers": ["2317"]},
    )
    check("stale_legacy_protection_does_not_override_primary_trading_next_action", NEXT_BUILD_PROPOSED, stale_protection["next_action_code"])
    check("stale_legacy_protection_does_not_block_new_allocation", True, stale_protection["workflow_action_availability"]["orders"])

    indicator_due = {"exists": True, "fresh": True, "exit_count": 1, "exits": [{"ticker": "2317", "entry_order_id": "ENTRY-2317", "signal_key": "sig-1"}], "active_indicator_exit_order_count": 0, "active_indicator_exit_tickers": []}
    due_with_stop = derive(account=strategy_account, orders=active_stop_orders, protection=fresh_protection, indicator_exit=indicator_due)
    check("indicator_due_is_strategy_decision_even_with_legacy_protection", NEXT_SUBMIT_INDICATOR_EXIT, due_with_stop["next_action_code"])
    due_without_stop = derive(account=strategy_account, protection=fresh_protection, indicator_exit=indicator_due)
    check("indicator_due_without_protection_requires_market_submission", NEXT_SUBMIT_INDICATOR_EXIT, due_without_stop["next_action_code"])
    check("indicator_due_blocks_step4_allocation", False, due_without_stop["workflow_action_availability"]["orders"])
    active_indicator_orders = {"revision": 6, "orders": [{"order_id":"ind1","ticker":"2317","side":"SELL","purpose":"INDICATOR_EXIT","entry_order_id":"ENTRY-2317","status":"ORDERED","information_date":"2026-09-04"}]}
    active_indicator = derive(account=strategy_account, orders=active_indicator_orders, protection=fresh_protection, indicator_exit=indicator_due)
    check("active_legacy_indicator_order_does_not_replace_strategy_sell_decision", NEXT_SUBMIT_INDICATOR_EXIT, active_indicator["next_action_code"])

    active_entry_orders = {
        "revision": 6,
        "orders": [{
            "order_id": "buy1", "ticker": "2330", "side": "BUY", "purpose": "ENTRY_BUY",
            "status": "PARTIAL", "information_date": "2026-09-04",
        }],
    }
    active_entry = derive(orders=active_entry_orders)
    check("active_legacy_entry_buy_does_not_override_primary_trading_next_action", NEXT_BUILD_PROPOSED, active_entry["next_action_code"])
    check("active_legacy_entry_buy_does_not_block_new_allocation", True, active_entry["workflow_action_availability"]["orders"])
    active_entry_with_due = derive(orders=active_entry_orders, account=strategy_account, position_rollforward={"due_tickers": ["2317"]})
    check("position_rollforward_precedes_hidden_legacy_entry_reconciliation", NEXT_ROLLFORWARD_POSITIONS, active_entry_with_due["next_action_code"])
    check("active_legacy_entry_does_not_disable_rollforward", True, active_entry_with_due["workflow_action_availability"]["rollforward"])

    stale_params = derive(workflow={"latest_data_date": "2026-09-04", "params_ready_for_scan": False})
    check("stale_params_require_update_params", NEXT_UPDATE_PARAMS, stale_params["next_action_code"])
    check("scanner_disabled_until_params_ready", False, stale_params["workflow_action_availability"]["scanner"])

    workflow_error = derive(component_errors={"workflow": "RuntimeError: broken workflow read model"})
    check("workflow_read_error_blocks_operations_status", OPERATIONS_STATUS_BLOCKED, workflow_error["overall_status"])
    check("workflow_read_error_disables_all_workflow_actions", False, any(workflow_error["workflow_action_availability"].values()))
    rollforward_error = derive(component_errors={"position_rollforward": "RuntimeError: broken rollforward snapshot"})
    check("rollforward_read_error_blocks_operations_status", OPERATIONS_STATUS_BLOCKED, rollforward_error["overall_status"])
    check("rollforward_read_error_blocks_allocation_and_daily_all", [False, False], [rollforward_error["workflow_action_availability"]["orders"], rollforward_error["workflow_action_availability"]["all"]])

    stale_candidate = derive(candidate={"exists": True, "valid": True, "fresh": False, "candidate_count": 3})
    check("stale_candidate_requires_scanner", NEXT_RUN_SCANNER, stale_candidate["next_action_code"])

    same_day_account = {**account, "latest_buy_trade_date": "2026-09-04"}
    locked = derive(account=same_day_account)
    check("same_information_day_entry_history_locks_reallocation", NEXT_DAY_LOCKED, locked["next_action_code"])
    check("same_day_lock_has_explicit_status", OPERATIONS_STATUS_LOCKED_TODAY, locked["overall_status"])
    check("same_day_lock_disables_step4_only", False, locked["workflow_action_availability"]["orders"])
    check("same_day_lock_keeps_scanner_available", True, locked["workflow_action_availability"]["scanner"])

    sell_fill_account = {**account, "latest_sell_trade_date": "2026-09-05"}
    sell_locked = derive(account=sell_fill_account)
    check("sell_fill_after_latest_completed_date_locks_same_session_reallocation", NEXT_DAY_LOCKED, sell_locked["next_action_code"])
    check("sell_session_lock_disables_step4", False, sell_locked["workflow_action_availability"]["orders"])
    check("sell_session_lock_is_explicit", True, sell_locked["same_session_sell_locked"])

    account_error = derive(component_errors={"account": "synthetic unreadable account"})
    check("account_read_error_blocks_step4_allocation", False, account_error["workflow_action_availability"]["orders"])

    orphan_sell_orders = {
        "revision": 9,
        "orders": [{
            "order_id": "orphan-sell", "ticker": "2317", "side": "SELL", "purpose": "INDICATOR_EXIT",
            "entry_order_id": "ENTRY-OLD", "status": "ORDERED", "information_date": "2026-09-04",
        }],
    }
    orphan_sell = derive(account=strategy_account, orders=orphan_sell_orders, protection=fresh_protection)
    check("old_lineage_active_sell_is_reported_as_orphan_not_current_indicator", 0, orphan_sell["active_indicator_exit_order_count"])
    check("orphan_legacy_sell_does_not_block_step4_allocation", True, orphan_sell["workflow_action_availability"]["orders"])
    check("orphan_legacy_sell_does_not_override_primary_trading_next_action", NEXT_BUILD_PROPOSED, orphan_sell["next_action_code"])

    build_proposed = derive()
    check("fresh_candidate_without_fresh_proposal_requires_step4", NEXT_BUILD_PROPOSED, build_proposed["next_action_code"])

    fresh_proposed = {"exists": True, "valid": True, "fresh": True, "order_count": 2, "information_date": "2026-09-04"}
    submit = derive(proposed=fresh_proposed)
    check("fresh_proposal_with_orders_requires_explicit_submission", NEXT_SUBMIT_PROPOSED, submit["next_action_code"])
    check("fresh_proposal_does_not_create_active_broker_order", 0, submit["active_entry_order_count"])

    no_orders = derive(proposed={"exists": True, "valid": True, "fresh": True, "order_count": 0, "information_date": "2026-09-04"})
    check("fresh_zero_order_plan_marks_no_entry_today", NEXT_NO_ENTRY, no_orders["next_action_code"])

    manual_account = {
        "initialized": True,
        "revision": 9,
        "cash": 500_000.0,
        "positions": [{"ticker": "0050", "source": "manual_adopted", "qty": 1000, "management_status": "unmanaged"}],
    }
    manual_only = derive(account=manual_account)
    check("manual_adopted_is_not_treated_as_strategy_stop_gap", [], manual_only["missing_stop_tickers"])
    check("manual_adopted_is_reported_separately", ["0050"], manual_only["manual_tickers"])

    source = (Path(__file__).resolve().parents[2] / "services" / "trading" / "operations_status.py").read_text(encoding="utf-8")
    panel_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("operations_status_is_read_only_composition", False, any(token in source for token in ("atomic_write_json(", "confirm_trading_", "run_trading_market_data_update(")))
    check("operations_status_disables_hidden_protection_recovery", True, "get_trading_protection_plan_read_model(root, recover_pending_fill=False)" in source)
    check("operations_status_disables_hidden_indicator_recovery", True, "get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False)" in source)
    rollforward_source = (Path(__file__).resolve().parents[2] / "services" / "trading" / "position_rollforward.py").read_text(encoding="utf-8")
    rollforward_snapshot_body = rollforward_source.split("def build_trading_position_rollforward_snapshot", 1)[1].split("def run_trading_position_rollforward", 1)[0]
    check("operations_rollforward_snapshot_has_no_hidden_fill_recovery", False, "recover_trading_fill_transaction(" in rollforward_snapshot_body)
    check("workbench_has_operations_overview", True, "Trading 操作總覽" in panel_source)
    check("workbench_consumes_operations_status_owner", True, "build_trading_operations_status" in panel_source)
    check("workbench_exposes_full_state_refresh", True, "全狀態刷新" in panel_source)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_workbench_account_panel_contract_case(base_params):
    case_id = "TRADING_WORKBENCH_ACCOUNT_PANEL"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_workbench')

    from services.workbench_ui.trading_account_panel import (
        TradingAccountPanel,
        build_trading_account_panel_snapshot,
        build_trading_status_segments,
        parse_trading_money_text,
        parse_trading_qty_text,
    )
    from services.workbench_ui.workbench import (
        PANEL_SPECS,
        StockToolsWorkbench,
        build_workbench_spec,
        resolve_workbench_combobox_popup_rows,
        resolve_workbench_combobox_popdown_geometry,
    )
    from types import SimpleNamespace

    # Historical regression: first-use GUI callbacks can compile successfully yet
    # fail at runtime when a newly referenced module global was never imported.
    # Scan every Workbench UI module for unresolved global references so startup /
    # lazy-navigation callbacks cannot hide this class of NameError.
    import builtins as _builtins
    import symtable as _symtable

    workbench_ui_root = Path(__file__).resolve().parents[2] / "services" / "workbench_ui"
    unresolved_workbench_globals = {}
    allowed_runtime_globals = set(dir(_builtins)) | {
        "__file__", "__name__", "__package__", "__spec__", "__loader__",
        "__cached__", "__builtins__",
    }
    for module_path in sorted(workbench_ui_root.glob("*.py")):
        module_source = module_path.read_text(encoding="utf-8")
        table = _symtable.symtable(module_source, str(module_path), "exec")
        module_defs = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
        }
        missing = set()

        def _collect_unresolved_globals(symbol_table):
            for symbol in symbol_table.get_symbols():
                name = symbol.get_name()
                if (
                    symbol.is_global()
                    and symbol.is_referenced()
                    and name not in module_defs
                    and name not in allowed_runtime_globals
                ):
                    missing.add(name)
            for child in symbol_table.get_children():
                _collect_unresolved_globals(child)

        _collect_unresolved_globals(table)
        if missing:
            unresolved_workbench_globals[module_path.name] = sorted(missing)
    check(
        "workbench_ui_modules_have_no_unresolved_global_names",
        {},
        unresolved_workbench_globals,
    )

    workbench_spec = build_workbench_spec()
    panel_specs = {row["panel_id"]: row for row in workbench_spec.get("panels", [])}
    check("actual_trading_panel_is_registered", True, "trading_account" in panel_specs)
    trading_panel = panel_specs.get("trading_account", {})
    accounting_panel = panel_specs.get("accounting_center", {})
    check("accounting_center_panel_is_registered", True, "accounting_center" in panel_specs)
    check("accounting_center_panel_label", "帳務中心", accounting_panel.get("tab_label"))
    check("accounting_center_uses_dedicated_factory", "services.workbench_ui.accounting_center_panel:AccountingCenterPanel", next((row.get("panel_factory_path") for row in PANEL_SPECS if row.get("panel_id") == "accounting_center"), None))
    run_bound_checks(
        check,
        (
            ('trading_center_panel_label', '交易中心', trading_panel.get('tab_label'),),
            ('actual_trading_panel_uses_account_state_backend', 'services.trading.account_state.get_trading_account_read_model', trading_panel.get('backend_runner'),),
            ('actual_trading_panel_uses_dedicated_factory', 'services.workbench_ui.trading_account_panel:TradingAccountPanel', next((row.get('panel_factory_path') for row in PANEL_SPECS if row.get('panel_id') == 'trading_account'), None),),
            ('money_parser_accepts_grouping_and_decimal', '1234567.89', str(parse_trading_money_text('1,234,567.89', 'cash')),),
            ('qty_parser_accepts_grouping', 1000, parse_trading_qty_text('1,000'),),
        ),
    )
    try:
        parse_trading_qty_text("1.5")
    except ValueError:
        fractional_qty_rejected = True
    else:
        fractional_qty_rejected = False
    check("fractional_share_qty_is_rejected", True, fractional_qty_rejected)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        empty_snapshot = build_trading_account_panel_snapshot(root)
        check("workbench_snapshot_handles_uninitialized_account", False, empty_snapshot["initialized"])
        check("workbench_snapshot_uses_project_relative_state_path", "state/trading/account.json", empty_snapshot["state_path"])

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
        check("workbench_snapshot_reads_current_revision", state["revision"], snapshot["revision"])
        check("workbench_snapshot_reads_cash", 900_000.0, snapshot["cash"])
        check("workbench_snapshot_reads_manual_position", "2330", snapshot["positions"][0]["ticker"])
        from core.trading_policy import get_trading_policy_snapshot
        expected_policy = get_trading_policy_snapshot()
        check("workbench_snapshot_uses_current_trading_strategy_config", expected_policy["strategy_id"], snapshot["policy"]["strategy_id"])
        check("workbench_snapshot_uses_current_param_selector_config", expected_policy["param_selector"], snapshot["policy"]["param_selector"])

    from services.workbench_ui.trading_account_panel import build_trading_account_panel_initial_bundle
    with tempfile.TemporaryDirectory() as temp_dir:
        initial_bundle = build_trading_account_panel_initial_bundle(Path(temp_dir))
    expected_initial_keys = {"reconcile", "account", "dashboard", "candidate_read", "candidate_payload", "protection", "indicator", "workflow", "operations"}
    check("workbench_trading_initial_bundle_has_all_canonical_read_models", expected_initial_keys, set(initial_bundle))
    check_true("workbench_trading_initial_bundle_reads_empty_state_without_exception", all(bool(value[0]) for value in initial_bundle.values()))

    sample_segments = build_trading_status_segments("READY | Data 2026-09-11 | 現金 1,500,000 | 持股 0 | 一般說明")
    sample_tones = {text: tone for text, tone in sample_segments if text.strip()}
    check("workbench_status_highlights_ready_token_only", "success", sample_tones.get("READY"))
    check("workbench_status_highlights_date_value", "info", sample_tones.get("2026-09-11"))
    check("workbench_status_highlights_money_value", "info", sample_tones.get("1,500,000"))
    check("workbench_status_keeps_explanatory_text_neutral", True, any(tone == "text" and "一般說明" in text for text, tone in sample_segments))
    warning_segments = build_trading_status_segments("NOT READY | STALE/EMPTY | FAIL：synthetic")
    warning_tones = {text: tone for text, tone in warning_segments if text.strip()}
    check("workbench_status_distinguishes_warning_and_error_tokens", ["warning", "warning", "error"], [warning_tones.get("NOT READY"), warning_tones.get("STALE/EMPTY"), warning_tones.get("FAIL")])

    from core.portfolio_ensemble import build_ensemble_candidate_display_metrics
    ensemble_metrics = build_ensemble_candidate_display_metrics(total_member_count=8)
    check("scanner_pool_ensemble_dynamic_metric_labels", ["共識"], [row.get("label") for row in ensemble_metrics])
    check("scanner_pool_consensus_uses_total_member_denominator", 8, ensemble_metrics[0].get("denominator"))
    check("scanner_pool_single_member_has_no_ensemble_dynamic_columns", [], build_ensemble_candidate_display_metrics(total_member_count=1))

    from services.trading.scanner_state import (
        TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        _validate_trading_candidate_snapshot_payload,
    )
    legacy_candidate_payload = {
        "schema_version": TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION - 1,
        "runtime_domain": "trading",
        "strategy_id": "synthetic",
        "param_selector": "synthetic",
        "latest_data_date": "2026-09-11",
        "selected_params_sha256": "synthetic",
        "param_member_count": 8,
        "param_min_agree": 5,
    }
    try:
        _validate_trading_candidate_snapshot_payload(legacy_candidate_payload)
    except ValueError as exc:
        legacy_schema_rejected = "schema_version" in str(exc)
    else:
        legacy_schema_rejected = False
    check("scanner_pool_dynamic_columns_reject_pre_descriptor_snapshot_schema", True, legacy_schema_rejected)

    descriptor_missing_payload = dict(legacy_candidate_payload)
    descriptor_missing_payload["schema_version"] = TRADING_CANDIDATE_SNAPSHOT_SCHEMA_VERSION
    try:
        _validate_trading_candidate_snapshot_payload(descriptor_missing_payload)
    except ValueError as exc:
        missing_descriptor_rejected = "candidate_display_metrics" in str(exc)
    else:
        missing_descriptor_rejected = False
    check("scanner_pool_current_snapshot_requires_dynamic_column_descriptor", True, missing_descriptor_rejected)

    panel_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("workbench_trading_page_is_vertically_scrollable", True, "self._page_canvas = tk.Canvas" in panel_source and "self._page_scrollbar = ttk.Scrollbar" in panel_source)
    check("workbench_proposed_table_exposes_agreement", True, '"agree": "同意/成員"' in panel_source)
    check("workbench_proposed_target_heading_distinguishes_completion_barrier", True, '"target": "Target / 完成線"' in panel_source)
    check("workbench_proposed_hint_explains_shadow_completion_vs_broker_tp", True, "inherited shadow completion barrier" in panel_source and "tp_percent=0 時不會建立 TP 券商單" in panel_source)
    check("workbench_refresh_reloads_persisted_proposed_plan", True, "def refresh_proposed_order_plan" in panel_source and "load_current_trading_proposed_order_plan" in panel_source)
    check("workbench_refresh_reloads_persisted_scanner_snapshot", True, "def refresh_candidate_snapshot_rows" in panel_source and "load_trading_candidate_snapshot" in panel_source)
    check("workbench_trading_initial_state_reads_off_tk_thread", True, 'name="workbench-trading-initial-state"' in panel_source and "build_trading_account_panel_initial_bundle" in panel_source and "_initial_state_results.put" in panel_source)
    check("workbench_trading_consumer_reconcile_runs_inside_background_bundle", True, 'bundle["reconcile"] = _capture_initial_panel_value' in panel_source and 'reconcile_trading_v2_consumer_state_from_local_evidence(root)' in panel_source)
    operations_refresh_body = panel_source.split("def refresh_operations_status", 1)[1].split("def ", 1)[0]
    check("workbench_trading_status_render_never_reconciles_on_tk_thread", False, "reconcile_trading_v2_consumer_state_from_local_evidence" in operations_refresh_body)
    check("workbench_trading_background_worker_never_calls_tk_after", True, "self.after(0, self._finish_initial_state_load" not in panel_source and "def _drain_initial_state_results" in panel_source)
    check("workbench_trading_constructor_defers_state_reads_until_after_paint", True, "self.after(80, self._start_initial_state_load)" in panel_source and "self.refresh_account()\n        self.refresh_candidate_snapshot_rows()" not in panel_source.split("def __init__", 1)[1].split("def _set_initial_loading_state", 1)[0])
    check("workbench_trading_initial_bundle_reuses_single_operations_snapshot", True, 'bundle["operations"]' in panel_source and "self._suspend_operations_refresh = True" in panel_source)
    check("workbench_trading_center_exposes_scanner_and_buy_entry", True, all(text in panel_source for text in ("今日 Scanner Pool", "買入成交登錄", "登錄買入成交", "BUY_ENTRY_HINT")))
    check("workbench_trading_primary_tables_use_left_stock_inspector_links_without_redundant_footer_buttons", True, all(token in panel_source for token in ('"open": "↗"', '"▣", ticker', "def _on_position_tree_click", "def _open_ticker_in_inspector", "on_open_stock=self._open_candidate_ticker_in_inspector")) and 'text="檢視選取股票"' not in panel_source and 'text="在單股回測檢視"' not in panel_source)
    check("workbench_trading_fixed_annotations_are_contextual_footer_hints", True, "雙擊股票可直接切到單股回測檢視" not in panel_source and "_trade_note_var" not in panel_source and "_bind_footer_hint(candidate_box, SCANNER_HINT)" in panel_source and "_bind_footer_hint(trade_box, BUY_ENTRY_HINT)" in panel_source)
    check("workbench_trading_center_has_no_primary_sell_entry", False, 'text="登錄賣出成交"' in panel_source.split('trade_box = ttk.LabelFrame(content, text="買入成交登錄', 1)[1].split('performance_box = ttk.LabelFrame', 1)[0])
    check("workbench_trading_center_keeps_position_decisions_without_duplicate_account_dashboard", True, "text=\"持股決策\"" in panel_source and "for accounting_section in (header, cash_box, form, performance_box)" in panel_source and "accounting_section.grid_remove()" in panel_source and 'dashboard_box = ttk.LabelFrame' not in panel_source)
    accounting_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "accounting_center_panel.py").read_text(encoding="utf-8")
    check("workbench_accounting_center_exposes_clean_inventory_trade_detail_titles_and_performance", True, all(text in accounting_source for text in ('text="庫存股"', 'text="買入明細"', 'text="賣出明細"', 'text="沖抵明細"', 'text="績效統計"', "持有成本", "買入手續費", "交易稅", "沖抵買入價金", "沖抵買入手續費")))
    check("workbench_accounting_center_owns_direct_inventory_sell_entry_without_broker_order_mapping", True, all(text in accounting_source for text in ('text="賣出成交登錄"', "record_trading_account_inventory_sell", "先在券商完成賣出")) and "對應券商 SELL 單" not in accounting_source)
    check("workbench_accounting_center_supports_transaction_edit_delete_without_primary_existing_inventory_entry", True, all(text in accounting_source for text in ("修改選取買入", "刪除選取買入", "修改選取賣出", "刪除選取賣出", "correct_trading_transaction", "delete_trading_transaction")) and "inv_actions.grid_remove()" in accounting_source)
    date_picker_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "date_picker.py").read_text(encoding="utf-8")
    check("workbench_date_inputs_open_calendar_from_date_field", True, "DatePickerField" in panel_source and "DatePickerField" in accounting_source and 'self.entry.bind("<Button-1>", self._on_entry_click' in date_picker_source and 'text="日曆"' not in date_picker_source)
    paged_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "paged_table.py").read_text(encoding="utf-8")
    check("workbench_tables_share_sort_toggle_page_and_stock_open_contract", True, all(text in paged_source for text in ("page_size", "上一頁", "下一頁", "_header_click", "_row_click", "_open_stock")) and "ttk.Scrollbar" not in paged_source)
    check("workbench_paged_table_supports_runtime_dynamic_columns", True, "def set_columns(" in paged_source)
    check("workbench_scanner_dynamic_columns_are_descriptor_driven_not_selector_named", True, "candidate_display_metrics" in panel_source and "_candidate_metric_formatter" in panel_source and "if selector ==" not in panel_source)
    scanner_columns_block = panel_source.split("def _candidate_static_columns_after_dynamic", 1)[1].split("@staticmethod", 1)[0]
    check("workbench_scanner_pool_shows_market_price_before_buy_limit", True, 'TableColumn("market_price", "市價"' in scanner_columns_block and scanner_columns_block.index('TableColumn("market_price", "市價"') < scanner_columns_block.index('TableColumn("limit_price", "買入限價"'))
    check("workbench_scanner_pool_market_price_uses_candidate_prev_close", True, '"market_price": row.get("prev_close")' in panel_source)
    check("workbench_scanner_pool_hides_median_sort_evidence_column", False, any(row.get("key") == "ensemble_median_sort_value" for row in ensemble_metrics))
    check("workbench_accounting_tables_page_at_twelve_without_inner_scrollbars", True, accounting_source.count("page_size=12") >= 5 and accounting_source.count("ttk.Scrollbar(") == 1 and 'self._page_scrollbar = ttk.Scrollbar' in accounting_source)
    check("workbench_trading_scanner_pages_at_twelve_without_inner_scrollbar", True, "page_size=12" in panel_source and panel_source.count("ttk.Scrollbar(") == 1 and "self._page_scrollbar = ttk.Scrollbar" in panel_source)
    check("workbench_buy_success_navigates_to_refreshed_accounting_center", True, '_open_accounting_center' in panel_source and 'callback(refresh=True)' in panel_source and "已切換至帳務中心並重新整理" in panel_source)
    check("workbench_primary_ui_does_not_mount_broker_oms_notebook", False, "advanced_notebook.grid(" in panel_source or "advanced_notebook.pack(" in panel_source)
    check("workbench_performance_colour_is_cell_scoped", True, "performance=True" in accounting_source and "column.performance" in paged_source and 'tag_configure("gain"' not in accounting_source)
    check("workbench_account_dashboard_uses_revision_market_date_cards_and_fixed_bottom_refresh", True, 'text="帳戶儀表板"' in accounting_source and 'cards = (("revision", "revision"), ("市價日", "market_date")' in accounting_source and 'footer_line = ttk.Frame' in accounting_source and 'text="全狀態刷新"' in accounting_source and 'pack(side="right"' in accounting_source and 'WORKBENCH_INFO if key in {"market_date", "cash", "equity"}' in accounting_source)
    check("workbench_account_performance_labels_empirical_r_semantics", True, '"實績 EV(R)"' in accounting_source and '"實績賺賠比"' in accounting_source)
    check("workbench_performance_is_fixed_three_row_unsortable_schema", True, all(text in accounting_source for text in ("股票檔數", 'TableColumn("value", "價值"', 'TableColumn("cost", "成本"', 'TableColumn("pnl", "損益"', "sortable=False")))
    check("workbench_page_mousewheel_binds_static_dynamic_and_full_accounting_panel", True, "def _bind_page_mousewheel" in accounting_source and "self._bind_page_mousewheel(self)" in accounting_source and "on_mousewheel=self._on_mousewheel" in accounting_source and "self._bind_mousewheel(cell)" in paged_source and "self._bind_pointer_callbacks(cell)" in paged_source)
    from services.workbench_ui.accounting_center_panel import AccountingCenterPanel
    accounting_scroll_calls = []
    accounting_scroll_panel = SimpleNamespace(_canvas=SimpleNamespace(yview_scroll=lambda units, mode: accounting_scroll_calls.append((units, mode))))
    accounting_scroll_result = AccountingCenterPanel._on_mousewheel(
        accounting_scroll_panel,
        SimpleNamespace(num="??", delta=-120),
    )
    check("workbench_accounting_mousewheel_accepts_native_windows_placeholder_num", [(1, "units")], accounting_scroll_calls)
    check("workbench_accounting_mousewheel_consumes_handled_event", "break", accounting_scroll_result)
    class _AliveRefreshThread:
        @staticmethod
        def is_alive():
            return True
    accounting_refresh_panel = SimpleNamespace(
        _refresh_thread=_AliveRefreshThread(),
        _refresh_pending=False,
        _refresh_status_var=SimpleNamespace(set=lambda _value: None),
    )
    AccountingCenterPanel.refresh(accounting_refresh_panel)
    check("workbench_accounting_refresh_marks_followup_pending_when_worker_is_alive", True, accounting_refresh_panel._refresh_pending)
    check("workbench_accounting_refresh_coalesces_mutation_followup_instead_of_dropping_it", True, "self._refresh_pending = True" in accounting_source and "if self._refresh_pending:" in accounting_source and "self.after_idle(self.refresh)" in accounting_source)
    check("workbench_account_mutations_resolve_latest_revision_inside_lock", True, "expected_account_revision=None" in panel_source and "expected_revision=None" in accounting_source)
    check("workbench_centers_use_fixed_contextual_footer_status_bars", True, "self._footer_bar.grid(row=1" in accounting_source and "操作提示｜" in accounting_source and "_bind_footer_hint(sell_entry" in accounting_source and "先在券商完成賣出，再登錄實際股數" in accounting_source and "self._footer_bar.grid(row=1" in panel_source and "操作提示｜" in panel_source)
    check("workbench_primary_ui_has_single_fixed_bottom_right_refresh_per_center", True, panel_source.count('text="全狀態刷新"') == 1 and 'footer_line = ttk.Frame' in panel_source and 'text="全狀態刷新", command=self._refresh_all_trading_state' in panel_source and 'pack(side="right"' in panel_source and accounting_source.count('text="全狀態刷新"') == 1 and 'footer_line = ttk.Frame' in accounting_source and 'text="全狀態刷新", command=self.refresh' in accounting_source)
    overview_schema = panel_source.split('operations_box = ttk.LabelFrame(content, text="Trading 操作總覽"', 1)[1].split('workflow_box = ttk.LabelFrame(content, text="每日 Trading 流程"', 1)[0]
    workflow_schema = panel_source.split('workflow_box = ttk.LabelFrame(content, text="每日 Trading 流程"', 1)[1].split('header = ttk.LabelFrame(content, text="Trading 帳戶"', 1)[0]
    check("workbench_overview_removes_account_card_and_consolidates_strategy_params", True, '帳戶"' not in overview_schema and '策略 / Params' in overview_schema and 'member' not in overview_schema.lower() and 'agree' not in overview_schema.lower())
    check("workbench_overview_exposes_requested_funnel_cards", True, all(label in overview_schema for label in ("同步狀態", "總股數", "符合快篩數", "Scanner 候選數", "剩餘可買數")))
    check("workbench_overview_limits_dynamic_status_to_two_single_line_rows", True, '_operations_next_label.grid(' in overview_schema and '_operations_detail_label.grid(' in overview_schema and overview_schema.count('max_lines=1') >= 2 and '_live_audit_label' not in overview_schema)
    check("workbench_daily_workflow_does_not_repeat_status_rows", True, '_workflow_status_label' not in workflow_schema and '_param_mode_detail_label' not in workflow_schema)
    check("workbench_dynamic_status_stays_out_of_footer", True, '_show_footer_hint(self._operations_detail_var.get())' not in panel_source and '_dashboard_detail_label.pack(' not in panel_source)
    scanner_schema = panel_source.split('candidate_box = ttk.LabelFrame(content, text="今日 Scanner Pool"', 1)[1].split('trade_box = ttk.LabelFrame(content, text="買入成交登錄"', 1)[0]
    candidate_static_schema = panel_source.split('def _candidate_static_columns_after_dynamic', 1)[1].split('def _candidate_metric_formatter', 1)[0]
    check("workbench_scanner_pool_uses_take_profit_reference_qty_cost_without_summary", True, 'TableColumn("target_price", "停利線"' in candidate_static_schema and 'TableColumn("proj_qty", "參考股數"' in candidate_static_schema and 'TableColumn("proj_cost", "參考投入"' in candidate_static_schema and 'Scanner 摘要' not in scanner_schema and candidate_static_schema.index('"停利線"') < candidate_static_schema.index('"參考股數"') < candidate_static_schema.index('"參考投入"'))
    check("workbench_buy_details_follow_inventory_selection", True, "def _apply_inventory_filter" in accounting_source and "if row:" in accounting_source.split("def _apply_inventory_filter", 1)[1].split("def _parse_cash", 1)[0] and "list(self._all_buy_rows)" in accounting_source)
    inspector_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "single_stock_inspector.py").read_text(encoding="utf-8")
    workbench_source = (Path(__file__).resolve().parents[2] / "services" / "workbench_ui" / "workbench.py").read_text(encoding="utf-8")
    check("workbench_single_stock_supports_research_trading_switch", True, all(text in inspector_source for text in ("檢視模式", 'values=("Research", "Trading")', "run_trading_candidate_scan", "load_trading_v2_sanitized_ohlcv_frame")))
    check("workbench_single_stock_exposes_trading_holdings_and_scanner_pool", True, all(text in inspector_source for text in ("持有股", "Scanner Pool", "get_trading_account_read_model")))
    open_ticker_body = inspector_source.split("def open_ticker", 1)[1].split("def ", 1)[0]
    check("workbench_single_stock_cross_panel_navigation_does_not_sync_refresh_auxiliary_lists", False, "_refresh_holdings_options()" in open_ticker_body or "_load_current_trading_candidate_pool()" in open_ticker_body)
    candidate_post_body = inspector_source.split("def _refresh_candidate_options_on_open", 1)[1].split("def ", 1)[0]
    check("workbench_single_stock_candidate_dropdown_lazy_loads_trading_pool", True, "postcommand=self._refresh_candidate_options_on_open" in inspector_source and "sync_ticker=False" in inspector_source)
    check("workbench_single_stock_candidate_dropdown_never_sync_loads_current_snapshot_on_post", False, "_load_current_trading_candidate_pool(" in candidate_post_body)
    check("workbench_single_stock_candidate_dropdown_prefetches_current_pool_in_background", True, "_request_trading_candidate_pool_refresh()" in candidate_post_body and "threading.Thread(" in inspector_source and 'name="workbench-trading-candidate-pool"' in inspector_source)
    check("workbench_single_stock_candidate_pool_reads_shared_persisted_snapshot_without_recalculation", True, "load_trading_candidate_snapshot(WORKBENCH_PROJECT_ROOT, require_current=False)" in inspector_source and "never triggers candidate recalculation" in inspector_source)
    check("workbench_single_stock_combobox_popup_geometry_is_screen_limited", True, "_configure_combobox_popup_geometry" in inspector_source and "resolve_workbench_combobox_popup_rows" in workbench_source and "_fit_posted_combobox_popdown" in workbench_source and 'ttk::combobox::PopdownWindow' in workbench_source)
    check("workbench_single_stock_controls_reflow_with_available_width", True, "_apply_single_stock_controls_layout" in inspector_source and all(mode in inspector_source for mode in ('mode == "wide"', 'mode == "compact"', 'else:')) )
    check(
        "workbench_combobox_popup_rows_shrink_to_available_screen_space",
        5,
        resolve_workbench_combobox_popup_rows(
            value_count=100, screen_height=300, widget_root_y=140, widget_height=20, row_height_px=24
        ),
    )
    check(
        "workbench_combobox_popup_rows_do_not_expand_short_lists",
        3,
        resolve_workbench_combobox_popup_rows(
            value_count=3, screen_height=1080, widget_root_y=80, widget_height=24, row_height_px=24
        ),
    )
    right_edge_geometry = resolve_workbench_combobox_popdown_geometry(
        widget_x=1700, widget_y=100, widget_width=180, widget_height=28,
        popup_width=520, popup_height=400, screen_x=0, screen_y=0,
        screen_width=1920, screen_height=1080,
    )
    check("workbench_combobox_popdown_repositions_inside_right_screen_edge", True, right_edge_geometry["x"] + right_edge_geometry["width"] <= 1920 - 18)
    bottom_edge_geometry = resolve_workbench_combobox_popdown_geometry(
        widget_x=200, widget_y=980, widget_width=180, widget_height=28,
        popup_width=320, popup_height=420, screen_x=0, screen_y=0,
        screen_width=1920, screen_height=1080,
    )
    check("workbench_combobox_popdown_repositions_above_when_bottom_space_is_insufficient", True, bottom_edge_geometry["y"] < 980 and bottom_edge_geometry["y"] >= 18)
    check("workbench_cross_panel_navigation_preserves_candidate_row_to_inspector", True, all(token in workbench_source for token in ('"candidate_row": dict(candidate_row or {})', 'candidate_row=dict(request.get("candidate_row") or {})')) and "on_open_stock=self._open_candidate_ticker_in_inspector" in panel_source and "candidate_row=dict(candidate_row or {})" in panel_source)
    check("workbench_cross_panel_navigation_shares_entire_trading_candidate_pool", True, all(token in workbench_source for token in ('"candidate_rows": [dict(row) for row in list(candidate_rows or [])]', 'candidate_rows=[dict(row) for row in list(request.get("candidate_rows") or [])]')) and "candidate_rows=[dict(row) for row in list(self._candidate_rows or [])]" in panel_source and 'candidate_latest_data_date=self._candidate_payload.get("latest_data_date")' in panel_source)
    check("workbench_primes_single_stock_factory_before_first_cross_panel_navigation", True, "def _prime_initial_panel_factories" in workbench_source and 'self._request_panel_load("single_stock_backtest_inspector")' in workbench_source)
    check("workbench_pending_single_stock_navigation_wins_lazy_import_selection_race", True, 'panel_id == "single_stock_backtest_inspector" and self._pending_single_stock_request' in workbench_source and 'self._notebook.select(host)' in workbench_source)
    check("workbench_single_stock_prefetches_trading_pool_before_trading_mode_first_use", True, 'allow_inactive=True' in inspector_source and 'self.after_idle(lambda: self._request_trading_candidate_pool_refresh(allow_inactive=True))' in inspector_source)
    check("workbench_single_stock_prefetch_validates_same_candidate_freshness_as_trading_center", True, "get_trading_candidate_snapshot_read_model(WORKBENCH_PROJECT_ROOT)" in inspector_source and 'if not bool(read_model.get("fresh"))' in inspector_source)
    check("workbench_single_stock_trading_analysis_uses_finalized_consumer_view", True, "open_trading_v2_consumer_view(" in inspector_source and "TradingMarketDataV2View.open(WORKBENCH_PROJECT_ROOT)" not in inspector_source)
    check("workbench_single_stock_combobox_reflow_rechecks_after_autosize", True, "def _autosize_combobox" in inspector_source and "self._schedule_single_stock_controls_layout()" in inspector_source)
    check("workbench_combobox_popdown_fit_retries_until_tcl_window_is_mapped", True, "WORKBENCH_COMBOBOX_POPUP_FIT_RETRIES" in workbench_source and "int(attempt) + 1" in workbench_source)

    from services.workbench_ui import single_stock_inspector as single_stock_inspector_module
    from services.workbench_ui.single_stock_inspector import (
        SingleStockBacktestInspectorPanel,
        resolve_single_stock_controls_layout_mode,
    )
    with tempfile.TemporaryDirectory() as identity_temp_dir:
        identity_root = Path(identity_temp_dir)
        candidate_path = identity_root / "candidate.json"
        consumer_path = identity_root / "consumer.json"
        binding_path = identity_root / "binding.json"
        for fixture_path in (candidate_path, consumer_path, binding_path):
            fixture_path.write_text("{}", encoding="utf-8")
        with (
            patch.object(single_stock_inspector_module, "WORKBENCH_PROJECT_ROOT", str(identity_root)),
            patch.object(
                single_stock_inspector_module,
                "resolve_trading_candidate_snapshot_path",
                return_value=candidate_path,
            ),
            patch.object(
                single_stock_inspector_module,
                "TRADING_V2_CONSUMER_STATE_RELATIVE_PATH",
                Path("consumer.json"),
            ),
            patch.object(
                single_stock_inspector_module,
                "resolve_trading_strategy_param_binding_path",
                return_value=binding_path,
            ),
        ):
            first_use_identity = SingleStockBacktestInspectorPanel._candidate_snapshot_identity()
    check(
        "workbench_single_stock_first_use_candidate_identity_executes_without_nameerror",
        True,
        len(first_use_identity) == 3 and all(value is not None for value in first_use_identity),
    )
    class _Var:
        def __init__(self, value=""):
            self.value = value
        def set(self, value):
            self.value = value
        def get(self):
            return self.value
    layout_req = {"identity": 420, "candidate": 300, "history": 400, "params": 500, "runtime": 300, "status": 200}
    check("workbench_single_stock_controls_choose_compact_before_1680px_right_edge_clips", "compact", resolve_single_stock_controls_layout_mode(available_width=1680, required_widths=layout_req))
    check("workbench_single_stock_controls_keep_wide_layout_when_1920px_has_safe_room", "wide", resolve_single_stock_controls_layout_mode(available_width=1920, required_widths=layout_req))

    queued = []
    candidate_prefetch_calls = []
    shared_pool_calls = []
    navigation_panel = SimpleNamespace(
        _runtime_domain_var=_Var("Research"),
        _ticker_var=_Var("00938"),
        _trading_candidate_rows_by_ticker={},
        _prefetched_trading_candidate_rows=[],
        _prefetched_trading_candidate_latest_data_date=None,
        _candidate_pool_refresh_token=0,
        _candidate_pool_last_error="old",
        _candidate_pool_checked_identity=None,
        _apply_runtime_domain_controls=lambda: None,
        _apply_trading_candidate_rows=lambda rows, **kwargs: (
            navigation_panel._trading_candidate_rows_by_ticker.update({str(row.get("ticker")): dict(row) for row in rows}),
            shared_pool_calls.append((list(rows), dict(kwargs))),
        ),
        _candidate_snapshot_identity=lambda: ("snapshot",),
        _select_candidate_dropdown_ticker=lambda ticker: None,
        _request_trading_candidate_pool_refresh=lambda: candidate_prefetch_calls.append(True),
        after_idle=lambda callback: queued.append(callback),
        _run_analysis=lambda: None,
    )
    SingleStockBacktestInspectorPanel.open_ticker(
        navigation_panel,
        "2455",
        runtime_domain="trading",
        auto_run=True,
        candidate_row={"ticker": "2455", "kind": "extended_tbd"},
        candidate_rows=[
            {"ticker": "00938", "kind": "extended_tbd"},
            {"ticker": "2455", "kind": "extended_tbd"},
        ],
        candidate_latest_data_date="2026-09-15",
    )
    check("workbench_single_stock_link_keeps_requested_row_ticker", "2455", navigation_panel._ticker_var.get())
    check("workbench_single_stock_link_keeps_requested_candidate_frozen_context", "2455", navigation_panel._trading_candidate_rows_by_ticker.get("2455", {}).get("ticker"))
    check("workbench_single_stock_link_schedules_analysis_without_auxiliary_refresh", 1, len(queued))
    check("workbench_single_stock_link_receives_entire_trading_candidate_pool_immediately", 2, len(shared_pool_calls[0][0]) if shared_pool_calls else 0)
    check("workbench_single_stock_link_does_not_reload_pool_when_trading_center_already_supplied_it", 0, len(candidate_prefetch_calls))
    check("workbench_single_stock_link_invalidates_older_background_pool_prefetch", 1, navigation_panel._candidate_pool_refresh_token)
    check("workbench_single_stock_link_clears_stale_pool_prefetch_error", None, navigation_panel._candidate_pool_last_error)

    # First cross-panel navigation must survive a lazy factory import without any
    # prior manual interaction with the single-stock panel.
    selected = {"panel_id": "trading_account"}
    selected_calls = []
    constructed = []
    class _FakeNotebook:
        def select(self, host):
            selected_calls.append(host)
            selected["panel_id"] = "single_stock_backtest_inspector"
    lazy_nav = SimpleNamespace(
        _panel_loading={"single_stock_backtest_inspector"},
        _panel_hosts={"single_stock_backtest_inspector": "single-host"},
        _panel_status_labels={"single_stock_backtest_inspector": None},
        _panel_factories={},
        _pending_single_stock_request={"ticker": "2455"},
        _notebook=_FakeNotebook(),
        _selected_panel_id=lambda: selected["panel_id"],
        _construct_ready_panel_if_selected=lambda panel_id: constructed.append(panel_id),
    )
    StockToolsWorkbench._finish_panel_load(
        lazy_nav, "single_stock_backtest_inspector", object(), None
    )
    check("workbench_first_trading_link_selects_lazy_single_stock_panel", ["single-host"], selected_calls)
    check("workbench_first_trading_link_constructs_lazy_single_stock_panel", ["single_stock_backtest_inspector"], constructed)

    research_prefetch_apply_calls = []
    research_prefetch_panel = SimpleNamespace(
        _candidate_pool_refresh_token=7,
        _candidate_pool_refresh_thread=object(),
        _candidate_pool_checked_identity=None,
        _candidate_pool_last_error="old",
        _prefetched_trading_candidate_rows=[],
        _prefetched_trading_candidate_latest_data_date=None,
        _runtime_domain_key=lambda: "research",
        _apply_trading_candidate_rows=lambda *args, **kwargs: research_prefetch_apply_calls.append((args, kwargs)),
        _configure_combobox_popup_geometry=lambda *_args, **_kwargs: None,
        _candidate_combo=object(),
    )
    SingleStockBacktestInspectorPanel._finish_trading_candidate_pool_refresh(
        research_prefetch_panel,
        7,
        ("snapshot",),
        {
            "candidate_rows": [{"ticker": "00938"}, {"ticker": "2455"}],
            "latest_data_date": "2026-09-15",
        },
    )
    check("workbench_single_stock_startup_prefetch_caches_trading_pool_while_research_visible", 2, len(research_prefetch_panel._prefetched_trading_candidate_rows))
    check("workbench_single_stock_startup_prefetch_does_not_pollute_research_candidate_dropdown", 0, len(research_prefetch_apply_calls))

    dropdown_ticker_var = _Var("2455")
    dropdown_value_var = _Var("")
    dropdown_mapping = {}
    dropdown_combo = SimpleNamespace(configure=lambda **_kwargs: None)
    dropdown_panel = SimpleNamespace(
        _ticker_var=dropdown_ticker_var,
        _autosize_combobox=lambda *_args, **_kwargs: None,
    )
    SingleStockBacktestInspectorPanel._apply_scan_dropdown(
        dropdown_panel,
        combo=dropdown_combo,
        value_var=dropdown_value_var,
        mapping=dropdown_mapping,
        display_values=["00938|延續|超限幅 0.00%|勝率 0.0%|次 1", "2455|延續|超限幅 0.00%|勝率 31.0%|次 29"],
        rule_key="candidate",
        sync_ticker=False,
    )
    check("workbench_lazy_candidate_dropdown_does_not_overwrite_opened_ticker", "2455", dropdown_ticker_var.get())

    # AI: Exercise persisted read-model reload without creating a Tk window.
    persisted_rows, persisted_status = [], []
    persisted_panel = SimpleNamespace(
        _reload_proposed_order_rows=lambda rows: persisted_rows.extend(list(rows or [])),
        _proposed_status_var=SimpleNamespace(set=persisted_status.append),
    )
    with (
        patch("services.workbench_ui.trading_account_panel.get_trading_proposed_order_plan_read_model", return_value={"exists": True, "valid": True, "fresh": True}),
        patch("services.workbench_ui.trading_account_panel.load_current_trading_proposed_order_plan", return_value={
            "orders": [{"rank": 1, "ticker": "2820"}],
            "account_revision": 0, "sizing_equity": 1_500_000, "reserved_total": 500_000, "cash_after_reservation": 1_000_000,
        }),
    ):
        TradingAccountPanel.refresh_proposed_order_plan(persisted_panel)
    check("workbench_persisted_proposed_plan_reloads_rows", "2820", persisted_rows[0]["ticker"] if persisted_rows else None)
    check_true("workbench_persisted_proposed_plan_refreshes_status", bool(persisted_status and "PROPOSED" in persisted_status[-1]))

    # AI: Completion callbacks now only render command results; canonical state refresh is
    # applied once by the shared background command executor.
    messages = []
    panel = SimpleNamespace(_operations_detail_var=SimpleNamespace(set=messages.append))
    TradingAccountPanel._finish_workflow_success(panel, "data", {
        "market_date": "2026-09-09", "current_execution_pool_ticker_count": 17,
        "training_ticker_count": 23,
        "market_data_v2_archive": {"status": "UPDATED", "data_requests": 31, "usage_requests": 2},
    })
    check("workbench_data_completion_uses_v2_result_fields", True, all(
        text in messages[-1] for text in ("2026-09-09", "新進場池 17", "訓練池 23", "V2 UPDATED", "31/2")
    ))
    check("workbench_data_completion_drops_retired_download_counts", False, "成功 0" in messages[-1])

    command_worker_body = panel_source.split("def _trading_command_worker", 1)[1].split("def _schedule_command_poll", 1)[0]
    command_finish_body = panel_source.split("def _finish_trading_command", 1)[1].split("def _request_state_refresh", 1)[0]
    workflow_success_body = panel_source.split("def _finish_workflow_success", 1)[1].split("def _current_revision", 1)[0]
    check("workbench_trading_commands_use_single_background_executor", True, "def _submit_trading_command" in panel_source and "_command_results.put" in panel_source)
    check("workbench_trading_command_worker_never_calls_tk", False, "self.after(" in command_worker_body or "messagebox." in command_worker_body)
    check("workbench_trading_command_builds_one_consolidated_state_bundle", True, "build_trading_account_panel_initial_bundle" in command_worker_body)
    check("workbench_trading_command_finish_applies_bundle_once", 1, command_finish_body.count("self._apply_state_bundle(bundle)"))
    check("workbench_workflow_completion_no_longer_reloads_disk_models", False, "self.refresh_" in workflow_success_body)
    check("workbench_user_refresh_uses_background_executor", True, "def _request_state_refresh" in panel_source and '"全狀態刷新"' in panel_source and "self._submit_trading_command(" in panel_source.split("def _refresh_all_trading_state", 1)[1].split("def _apply_workflow_action_availability", 1)[0])
    fill_confirmation_body = panel_source.split("def _confirm_selected_fill", 1)[1].split("def _cancel_selected_order", 1)[0]
    check("workbench_fill_confirmation_runs_through_background_executor", True, "self._submit_trading_command(" in fill_confirmation_body)
    check("workbench_fill_post_plan_errors_are_traceable", True, "indicator_error = str(exc)" in fill_confirmation_body and "Indicator 計畫刷新失敗" in fill_confirmation_body)
    check("workbench_account_mutations_run_through_background_executor", True, "def _submit_account_mutation" in panel_source and "self._submit_trading_command(" in panel_source.split("def _submit_account_mutation", 1)[1].split("def refresh_account", 1)[0])

    import queue as _queue
    background_panel = SimpleNamespace(_command_results=_queue.Queue())
    with patch("services.workbench_ui.trading_account_panel.build_trading_account_panel_initial_bundle", return_value={"operations": (True, {"overall_status": "READY"})}):
        TradingAccountPanel._trading_command_worker(background_panel, 9, lambda: {"ok": True}, True)
    token, result, command_error, bundle, bundle_error = background_panel._command_results.get_nowait()
    check("workbench_background_command_preserves_result", [9, True, None, None], [token, bool(result.get("ok")), command_error, bundle_error])
    check("workbench_background_command_returns_consolidated_bundle", True, isinstance(bundle, dict) and "operations" in bundle)

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
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_prelive')

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from core.trading_capabilities import build_trading_capability_snapshot
    from core.trading_market_clock import (
        assert_completed_daily_information_date,
        latest_allowed_completed_daily_date,
        select_latest_completed_daily_date,
        trading_daily_bar_complete_time,
    )
    from services.trading.operational_audit import (
        TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED,
        build_trading_operational_audit,
        run_trading_operational_audit,
    )

    tz = ZoneInfo("Asia/Taipei")
    publish_hour, publish_minute = trading_daily_bar_complete_time()
    after_close = datetime(2026, 9, 5, publish_hour, publish_minute, tzinfo=tz)
    before_provider_publish = after_close - timedelta(minutes=1)
    morning = datetime(2026, 9, 5, 10, 0, tzinfo=tz)
    check("morning_completed_daily_cutoff_excludes_today", "2026-09-04", latest_allowed_completed_daily_date(now=morning))
    check("before_provider_publication_cutoff_excludes_today", "2026-09-04", latest_allowed_completed_daily_date(now=before_provider_publish))
    check("after_provider_publication_grace_allows_today", "2026-09-05", latest_allowed_completed_daily_date(now=after_close))
    check("provider_provisional_today_row_is_ignored_before_cutoff", "2026-09-04", select_latest_completed_daily_date(["2026-09-04", "2026-09-05"], now=morning))
    check("provider_today_row_is_ignored_before_documented_publish", "2026-09-04", select_latest_completed_daily_date(["2026-09-04", "2026-09-05"], now=before_provider_publish))
    check("provider_today_row_is_eligible_after_documented_publish_grace", "2026-09-05", select_latest_completed_daily_date(["2026-09-04", "2026-09-05"], now=after_close))
    try:
        assert_completed_daily_information_date("2026-09-05", now=morning)
    except RuntimeError:
        provisional_rejected = True
    else:
        provisional_rejected = False
    check("intraday_today_information_date_is_rejected", True, provisional_rejected)

    capability = build_trading_capability_snapshot()
    blockers = set(capability.get("live_blocking_capabilities") or [])
    check("daily_position_rollforward_is_implemented_and_no_longer_live_blocker", False, "daily_position_rollforward" in blockers)
    check("indicator_sell_execution_is_implemented_and_no_longer_live_blocker", False, "indicator_sell_execution" in blockers)
    check("completed_daily_bar_seal_is_implemented", True, bool((capability["capabilities"]["completed_daily_bar_seal"]["implemented"])))

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
        check("research_cutoff_breach_blocks_live_readiness", TRADING_OPERATIONAL_AUDIT_STATUS_LIVE_BLOCKED, audit["status"])
        check("research_cutoff_breach_is_reported", True, any("Research reduced dataset 超過 cutoff" in item for item in audit["blockers"]))
        report = run_trading_operational_audit(root)
        check("operational_audit_writes_human_readable_report", True, (root / report["markdown_path"]).is_file())
        check("operational_audit_writes_machine_readable_report", True, (root / report["json_path"]).is_file())

    project_root = Path(__file__).resolve().parents[2]
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    audit_source = (project_root / "services" / "trading" / "operational_audit.py").read_text(encoding="utf-8")
    check("workbench_exposes_explicit_prelive_audit_action", True, "實盤就緒檢查" in panel_source)
    check("operational_audit_does_not_mutate_trading_state", False, any(token in audit_source for token in ("confirm_trading_", "mutate_trading_", "set_trading_cash_balance(")))

    summary["checks"] = len(results)
    return results, summary


def validate_trading_live_readiness_hardening_contract_case(base_params):
    """Audit-derived guards for actionable signal date and TP retry integrity."""
    case_id = "TRADING_LIVE_READINESS_HARDENING"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_live_readiness')

    from datetime import datetime
    from zoneinfo import ZoneInfo

    from core.file_integrity import atomic_write_json, canonical_json_sha256
    from core.params_io import params_to_json_dict
    from core.price_utils import calc_half_take_profit_sell_qty
    from core.trading_order_state import (
        TRADING_ORDER_PURPOSE_PROTECTION_STOP,
        TRADING_ORDER_PURPOSE_PROTECTION_TP,
        append_ordered_trading_proposal,
        build_empty_trading_order_state,
    )
    from services.trading.fill_reconciliation import (
        confirm_trading_buy_order_fill,
        confirm_trading_protection_sell_order_fill,
    )
    from services.trading.order_state import (
        confirm_trading_order_cancellation,
        resolve_trading_order_state_path,
    )
    from services.trading.protection_order_submission import (
        confirm_trading_protection_oco_submission,
        confirm_trading_protection_leg_submission,
    )
    from services.trading.protection_planning import build_trading_protection_plan
    from services.trading.scanner_state import partition_trading_candidate_rows_for_information_date

    current_row = {
        "ticker": "2330",
        "kind": "buy",
        "trade_date": "2026-09-04",
        "execution_plan_seed": {"ticker": "2330", "trade_date": "2026-09-04"},
    }
    stale_row = {
        "ticker": "2454",
        "kind": "buy",
        "trade_date": "2026-09-03",
        "execution_plan_seed": {"ticker": "2454", "trade_date": "2026-09-03"},
    }
    current, stale = partition_trading_candidate_rows_for_information_date(
        [current_row, stale_row], information_date="2026-09-04"
    )
    check("old_ticker_signal_is_not_actionable_on_dataset_latest_date", ["2330"], [row["ticker"] for row in current])
    check("stale_signal_is_explicitly_reported", [{"ticker": "2454", "trade_date": "2026-09-03"}], stale)
    try:
        partition_trading_candidate_rows_for_information_date(
            [{**current_row, "execution_plan_seed": {"ticker": "2330", "trade_date": "2026-09-03"}}],
            information_date="2026-09-04",
        )
    except ValueError:
        seed_mismatch_rejected = True
    else:
        seed_mismatch_rejected = False
    check("candidate_and_execution_seed_date_mismatch_is_fail_fast", True, seed_mismatch_rejected)

    frozen_params = params_to_json_dict(base_params)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        qty = 100
        tp_target = calc_half_take_profit_sell_qty(qty, base_params.tp_percent)
        if tp_target <= 1:
            raise RuntimeError("synthetic TP target 太小，無法驗 partial-cancel retry")
        reserved = build_buy_ledger_from_price(100.0, qty, base_params)["net_buy_total_milli"]
        order_state = build_empty_trading_order_state(
            timestamp="2026-09-03T09:00:00+08:00", mutation_id="live-hardening-init"
        )
        plan = {
            "plan_fingerprint": "live-hardening-entry-plan",
            "information_date": "2026-09-02",
            "account_revision": int(account["revision"]),
            "selected_params_sha256": canonical_json_sha256(frozen_params),
            "candidate_snapshot_sha256": "live-hardening-candidate",
            "strategy_id": "full_rule_based_no_dl",
            "param_selector": "base_finalist_best",
        }
        proposal = {
            "rank": 1,
            "ticker": "2454",
            "kind": "buy",
            "entry_type": "normal",
            "qty": qty,
            "limit_price": 100.0,
            "reserved_cost_milli": int(reserved),
            "init_sl": 90.0,
            "init_trail": 90.0,
            "target_price": 110.0,
            "entry_atr": 5.0,
            "security_profile": {},
        }
        order_state = append_ordered_trading_proposal(
            order_state,
            order_id="live-hardening-entry",
            proposal=proposal,
            plan=plan,
            timestamp="2026-09-03T09:01:00+08:00",
            mutation_id="live-hardening-submit",
            frozen_params=frozen_params,
        )
        atomic_write_json(resolve_trading_order_state_path(root), order_state)
        bought = confirm_trading_buy_order_fill(
            root,
            order_id="live-hardening-entry",
            fill_qty=qty,
            fill_price=99.0,
            trade_date="2026-09-03",
            expected_order_revision=int(order_state["revision"]),
            expected_account_revision=int(account["revision"]),
        )
        first_plan = build_trading_protection_plan(root)
        first_row = first_plan["positions"][0]
        check("tp_target_is_bound_to_original_entry_qty", tp_target, int(first_row["tp_target_qty"]))
        oco = confirm_trading_protection_oco_submission(
            root,
            ticker="2454",
            expected_order_revision=int(bought["order_revision"]),
            broker_oco_group_id="LIVE-HARDENING-OCO",
            stop_broker_order_id="LIVE-HARDENING-STOP",
            tp_broker_order_id="LIVE-HARDENING-TP",
        )
        tp_order = next(
            row for row in oco["orders"].values()
            if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_TP
        )
        partial_qty = max(1, tp_target // 2)
        partial = confirm_trading_protection_sell_order_fill(
            root,
            order_id=tp_order["order_id"],
            fill_qty=partial_qty,
            fill_price=111.0,
            trade_date="2026-09-04",
            expected_order_revision=int(oco["revision"]),
            expected_account_revision=int(bought["account_revision"]),
        )
        cancelled_tp = confirm_trading_order_cancellation(
            root,
            order_id=tp_order["order_id"],
            expected_revision=int(partial["order_revision"]),
            note="synthetic partial TP broker cancel",
        )
        stop_order = next(
            row for row in cancelled_tp["orders"].values()
            if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP
        )
        cancelled = confirm_trading_order_cancellation(
            root,
            order_id=stop_order["order_id"],
            expected_revision=int(cancelled_tp["revision"]),
            note="synthetic explicit broker OCO stop-peer cancellation",
        )
        retry_plan = build_trading_protection_plan(root)
        retry_row = retry_plan["positions"][0]
        expected_remaining = tp_target - partial_qty
        check("tp_retry_keeps_original_target_after_partial_cancel", expected_remaining, int(retry_row["tp_sell_qty"]))
        check("tp_retry_reports_cumulative_confirmed_qty", partial_qty, int(retry_row["tp_confirmed_qty"]))
        wrong_rederived = calc_half_take_profit_sell_qty(qty - partial_qty, base_params.tp_percent)
        check("tp_retry_is_not_rederived_from_reduced_position", False, int(retry_row["tp_sell_qty"]) == int(wrong_rederived) and int(wrong_rederived) != expected_remaining)

        retry_order_state = confirm_trading_protection_leg_submission(
            root,
            ticker="2454",
            action="TP_HALF",
            expected_order_revision=int(cancelled["revision"]),
            broker_order_id="LIVE-HARDENING-TP-RETRY",
        )
        retry_order = next(
            row for row in retry_order_state["orders"].values()
            if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_TP
            and row.get("broker_order_id") == "LIVE-HARDENING-TP-RETRY"
        )
        completed = confirm_trading_protection_sell_order_fill(
            root,
            order_id=retry_order["order_id"],
            fill_qty=expected_remaining,
            fill_price=112.0,
            trade_date="2026-09-04",
            expected_order_revision=int(retry_order_state["revision"]),
            expected_account_revision=int(partial["account_revision"]),
        )
        final_position = completed["account"]["positions"]["2454"]
        check("fragmented_tp_attempts_complete_canonical_sold_half_once", True, bool(final_position["strategy_management"]["position_state"]["sold_half"]))
        check("fragmented_tp_attempts_sell_exact_original_target", qty - tp_target, int(final_position["broker"]["qty"]))

    summary["checks"] = len(results)
    return results, summary


def validate_trading_stop_remainder_forced_exit_contract_case(base_params):
    """A triggered STOP remains a persistent full-exit obligation until flat."""
    case_id = "TRADING_STOP_REMAINDER_FORCED_EXIT"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_stop_remainder')

    from core.file_integrity import atomic_write_json, canonical_json_sha256
    from core.params_io import params_to_json_dict
    from core.trading_account_state import build_trading_account_read_model
    from core.trading_order_state import (
        TRADING_ORDER_PURPOSE_PROTECTION_STOP,
        TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
        TRADING_ORDER_STATUS_PARTIAL,
        append_ordered_trading_proposal,
        build_empty_trading_order_state,
        build_trading_order_read_model,
    )
    from core.trading_stop_exit_progress import build_trading_stop_exit_progress
    from services.trading.account_state import initialize_trading_account_state, load_trading_account_state
    from services.trading.fill_reconciliation import (
        confirm_trading_buy_order_fill,
        confirm_trading_protection_sell_order_fill,
    )
    from services.trading.operations_status import (
        NEXT_RECONCILE_STOP_REMAINDER,
        NEXT_SUBMIT_STOP_REMAINDER,
        derive_trading_operations_status,
    )
    from services.trading.order_state import (
        confirm_trading_order_cancellation,
        load_trading_order_state,
        resolve_trading_order_state_path,
    )
    from services.trading.protection_order_submission import confirm_trading_protection_leg_submission
    from services.trading.protection_planning import (
        PROTECTION_STOP_REMAINDER_ACTION,
        build_trading_protection_plan,
        get_trading_protection_plan_read_model,
    )

    frozen_params = params_to_json_dict(base_params)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        account = initialize_trading_account_state(root, cash=1_000_000)
        qty = 100
        reserved = build_buy_ledger_from_price(100.0, qty, base_params)["net_buy_total_milli"]
        state = build_empty_trading_order_state(
            timestamp="2026-09-03T09:00:00+08:00", mutation_id="stop-rem-init"
        )
        entry_plan = {
            "plan_fingerprint": "stop-rem-entry-plan",
            "information_date": "2026-09-02",
            "account_revision": int(account["revision"]),
            "selected_params_sha256": canonical_json_sha256(frozen_params),
            "candidate_snapshot_sha256": "stop-rem-candidate",
            "strategy_id": "full_rule_based_no_dl",
            "param_selector": "base_finalist_best",
        }
        proposal = {
            "rank": 1, "ticker": "2454", "kind": "buy", "entry_type": "normal", "qty": qty,
            "limit_price": 100.0, "reserved_cost_milli": int(reserved), "init_sl": 90.0, "init_trail": 90.0,
            "target_price": 110.0, "entry_atr": 5.0, "security_profile": {},
        }
        state = append_ordered_trading_proposal(
            state, order_id="stop-rem-entry", proposal=proposal, plan=entry_plan,
            timestamp="2026-09-03T09:01:00+08:00", mutation_id="stop-rem-submit", frozen_params=frozen_params,
        )
        atomic_write_json(resolve_trading_order_state_path(root), state)
        bought = confirm_trading_buy_order_fill(
            root, order_id="stop-rem-entry", fill_qty=qty, fill_price=99.0, trade_date="2026-09-03",
            expected_order_revision=int(state["revision"]), expected_account_revision=int(account["revision"]),
        )
        build_trading_protection_plan(root)
        stop_state = confirm_trading_protection_leg_submission(
            root, ticker="2454", action="STOP_FULL", expected_order_revision=int(bought["order_revision"]),
            broker_order_id="STOP-ORIGINAL",
        )
        stop_order = next(
            row for row in stop_state["orders"].values()
            if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP
        )

        first_stop_qty = 20
        partial = confirm_trading_protection_sell_order_fill(
            root, order_id=stop_order["order_id"], fill_qty=first_stop_qty, fill_price=89.0,
            trade_date="2026-09-04", expected_order_revision=int(stop_state["revision"]),
            expected_account_revision=int(bought["account_revision"]),
        )
        partial_stop = partial["orders"]["orders"][stop_order["order_id"]]
        check("original_stop_partial_fill_stays_active_for_broker_reconciliation", TRADING_ORDER_STATUS_PARTIAL, partial_stop["status"])
        check("first_stop_fill_reduces_real_position", qty - first_stop_qty, int(partial["account"]["positions"]["2454"]["broker"]["qty"]))

        progress = build_trading_stop_exit_progress(
            partial["orders"], ticker="2454", entry_order_id="stop-rem-entry"
        )
        forced_key = str(progress.get("forced_exit_key") or "")
        check("any_confirmed_original_stop_fill_latches_forced_exit", True, bool(progress.get("triggered")))
        check("forced_exit_identity_is_persistent_and_nonempty", True, bool(forced_key))
        check("active_original_stop_remaining_matches_current_position", qty - first_stop_qty, int(progress.get("active_exit_remaining_qty") or 0))

        forced_plan_while_active = build_trading_protection_plan(root)
        forced_row = forced_plan_while_active["positions"][0]
        forced_leg = forced_row["legs"][0]
        check("triggered_stop_plan_never_returns_to_waiting_stop", PROTECTION_STOP_REMAINDER_ACTION, forced_leg["action"])
        check("triggered_stop_plan_is_market_full_remainder", ["MARKET", qty - first_stop_qty], [forced_leg["order_type"], int(forced_leg["qty"])])
        check("triggered_stop_plan_removes_take_profit", 1, len(forced_row["legs"]))
        protection_rm = get_trading_protection_plan_read_model(root, recover_pending_fill=False)
        check("active_partial_original_stop_is_not_false_stale_after_position_qty_reduces", [], protection_rm["stale_active_protection_order_ids"])

        def operation_status(order_state, protection_model):
            return derive_trading_operations_status(
                workflow={"latest_data_date": "2026-09-04", "params_ready_for_scan": True, "param_selector": "base_finalist_best"},
                account={"initialized": True, **build_trading_account_read_model(load_trading_account_state(root, required=True))},
                orders=build_trading_order_read_model(order_state),
                candidate={"exists": True, "valid": True, "fresh": True, "candidate_count": 0, "information_date": "2026-09-04"},
                proposed={"exists": True, "valid": True, "fresh": True, "order_count": 0, "information_date": "2026-09-04"},
                protection=protection_model,
                indicator_exit={"exists": True, "fresh": True, "exit_count": 0, "exits": [], "active_indicator_exit_order_count": 0, "active_indicator_exit_tickers": []},
                position_rollforward={"due_tickers": []},
            )

        active_ops = operation_status(partial["orders"], protection_rm)
        check("active_triggered_stop_surfaces_manual_exit_action_without_oms_reconciliation_gate", NEXT_SUBMIT_STOP_REMAINDER, active_ops["next_action_code"])

        try:
            confirm_trading_protection_leg_submission(
                root, ticker="2454", action=PROTECTION_STOP_REMAINDER_ACTION,
                expected_order_revision=int(partial["order_revision"]), broker_order_id="FORCED-TOO-EARLY",
            )
        except RuntimeError:
            active_original_blocks_duplicate = True
        else:
            active_original_blocks_duplicate = False
        check("forced_market_cannot_duplicate_active_original_stop", True, active_original_blocks_duplicate)

        cancelled = confirm_trading_order_cancellation(
            root, order_id=stop_order["order_id"], expected_revision=int(partial["order_revision"]),
            note="synthetic broker cancelled original stop remainder",
        )
        retry_plan = build_trading_protection_plan(root)
        retry_row = retry_plan["positions"][0]
        retry_leg = retry_row["legs"][0]
        check("cancelled_original_stop_rebuilds_market_remainder_not_stop_market", [PROTECTION_STOP_REMAINDER_ACTION, "MARKET"], [retry_leg["action"], retry_leg["order_type"]])
        check("first_forced_market_qty_is_all_current_remaining_position", qty - first_stop_qty, int(retry_leg["qty"]))
        after_cancel_rm = get_trading_protection_plan_read_model(root, recover_pending_fill=False)
        cancelled_ops = operation_status(cancelled, after_cancel_rm)
        check("cancelled_triggered_stop_requires_forced_market_submission", NEXT_SUBMIT_STOP_REMAINDER, cancelled_ops["next_action_code"])

        forced_state = confirm_trading_protection_leg_submission(
            root, ticker="2454", action=PROTECTION_STOP_REMAINDER_ACTION,
            expected_order_revision=int(cancelled["revision"]), broker_order_id="FORCED-1",
        )
        forced_order = next(
            row for row in forced_state["orders"].values()
            if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            and row.get("broker_order_id") == "FORCED-1"
        )
        check("forced_remainder_order_is_market", "MARKET", forced_order["order_type"])
        check("forced_remainder_order_binds_original_trigger_identity", forced_key, forced_order["stop_forced_exit_key"])
        check("first_forced_remainder_attempt_is_one", 1, int(forced_order["stop_forced_exit_attempt"]))

        reloaded = load_trading_order_state(root, required=True)
        restarted_progress = build_trading_stop_exit_progress(
            reloaded, ticker="2454", entry_order_id="stop-rem-entry"
        )
        check("restart_reconstructs_same_forced_exit_obligation_from_order_history", forced_key, restarted_progress["forced_exit_key"])

        forced_partial_qty = 30
        forced_partial = confirm_trading_protection_sell_order_fill(
            root, order_id=forced_order["order_id"], fill_qty=forced_partial_qty, fill_price=88.0,
            trade_date="2026-09-04", expected_order_revision=int(forced_state["revision"]),
            expected_account_revision=int(partial["account_revision"]),
        )
        remaining_after_two_fills = qty - first_stop_qty - forced_partial_qty
        check("forced_market_partial_fill_keeps_same_stop_accounting_lineage", remaining_after_two_fills, int(forced_partial["account"]["positions"]["2454"]["broker"]["qty"]))
        cancelled_forced = confirm_trading_order_cancellation(
            root, order_id=forced_order["order_id"], expected_revision=int(forced_partial["order_revision"]),
            note="synthetic broker cancelled forced remainder",
        )
        second_plan = build_trading_protection_plan(root)
        second_row = second_plan["positions"][0]
        second_leg = second_row["legs"][0]
        check("forced_retry_qty_rebinds_to_entire_current_remaining_position", remaining_after_two_fills, int(second_leg["qty"]))
        second_state = confirm_trading_protection_leg_submission(
            root, ticker="2454", action=PROTECTION_STOP_REMAINDER_ACTION,
            expected_order_revision=int(cancelled_forced["revision"]), broker_order_id="FORCED-2",
        )
        second_order = next(
            row for row in second_state["orders"].values()
            if row.get("purpose") == TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER
            and row.get("broker_order_id") == "FORCED-2"
        )
        check("forced_retry_increments_attempt_without_changing_obligation_key", [forced_key, 2], [second_order["stop_forced_exit_key"], int(second_order["stop_forced_exit_attempt"])])
        completed = confirm_trading_protection_sell_order_fill(
            root, order_id=second_order["order_id"], fill_qty=remaining_after_two_fills, fill_price=87.0,
            trade_date="2026-09-04", expected_order_revision=int(second_state["revision"]),
            expected_account_revision=int(forced_partial["account_revision"]),
        )
        check("forced_exit_retries_continue_until_position_is_flat", False, "2454" in completed["account"]["positions"])

    project_root = Path(__file__).resolve().parents[2]
    planner_source = (project_root / "services" / "trading" / "indicator_exit_planning.py").read_text(encoding="utf-8")
    submit_source = (project_root / "services" / "trading" / "indicator_exit_order_submission.py").read_text(encoding="utf-8")
    audit_source = (project_root / "services" / "trading" / "operational_audit.py").read_text(encoding="utf-8")
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    check("indicator_plan_explicitly_skips_stop_forced_exit_lineage", True, "stop_forced_exit_skipped" in planner_source and "build_trading_stop_exit_progress" in planner_source)
    check("indicator_submission_defensively_rejects_already_triggered_stop_lineage", True, "剩餘持股必須沿 STOP forced-exit obligation" in submit_source)
    check("prelive_audit_has_explicit_forced_stop_continuity_gate", True, "trading_stop_forced_exit_continuity" in audit_source)
    check("workbench_exposes_explicit_stop_remainder_market_submission", True, "確認 Stop 剩餘 MARKET 已送單" in panel_source)

    summary["checks"] = len(results)
    return results, summary

def validate_trading_ssot_identity_path_scale_contract_case(base_params):
    """Trading identity, current-state paths, and milli scale have one canonical owner."""
    case_id = "TRADING_SSOT_IDENTITY_PATH_SCALE"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_ssot')

    import ast
    import tempfile
    from pathlib import Path

    from core.event_hash_chain import compute_event_hash
    from core.exit_priority import EXIT_SAME_BAR_PRIORITY_STOP_OVER_TP, resolve_stop_tp_hits
    from core.trading_identity import (
        normalize_trading_date,
        normalize_trading_ticker,
        require_trading_date_after,
    )
    from core.trading_state_paths import (
        resolve_trading_account_state_path as core_account_path,
        resolve_trading_fill_transaction_path as core_fill_path,
        resolve_trading_order_state_path as core_order_path,
    )
    from services.trading.account_state import resolve_trading_account_state_path as service_account_path
    from services.trading.fill_reconciliation import resolve_trading_fill_transaction_path as service_fill_path
    from services.trading.order_state import resolve_trading_order_state_path as service_order_path
    from services.trading.scanner_state import partition_trading_candidate_rows_for_information_date

    sample_event = {"revision": 1, "details": {"x": 2}, "event_hash": "must-not-self-hash"}
    check("account_and_order_event_chain_share_canonical_hash_primitive", compute_event_hash({"revision": 1, "details": {"x": 2}}), compute_event_hash(sample_event))
    check("same_bar_stop_tp_priority_has_single_identity", "STOP_OVER_TP", EXIT_SAME_BAR_PRIORITY_STOP_OVER_TP)
    check("same_bar_stop_dominates_tp_in_canonical_helper", (True, False), resolve_stop_tp_hits(stop_hit=True, tp_hit=True))
    check("ticker_identity_strips_outer_space_and_uppercases", "2330", normalize_trading_ticker(" 2330 "))
    try:
        normalize_trading_ticker("23 30")
    except ValueError:
        embedded_space_rejected = True
    else:
        embedded_space_rejected = False
    check("ticker_identity_rejects_embedded_whitespace", True, embedded_space_rejected)

    try:
        partition_trading_candidate_rows_for_information_date(
            [{"ticker": "23 30", "trade_date": "2026-09-05", "execution_plan_seed": {"ticker": "23 30", "trade_date": "2026-09-05"}}],
            information_date="2026-09-05",
        )
    except ValueError:
        scanner_uses_identity_contract = True
    else:
        scanner_uses_identity_contract = False
    check("scanner_consumes_same_ticker_identity_contract", True, scanner_uses_identity_contract)
    check("canonical_trading_date_normalizer_accepts_iso_date", "2026-09-05", normalize_trading_date("2026-09-05", allow_none=False))
    try:
        require_trading_date_after("2026-09-05", after="2026-09-05", field_name="fill", after_field_name="information")
    except ValueError:
        same_day_chronology_rejected = True
    else:
        same_day_chronology_rejected = False
    check("canonical_trading_chronology_rejects_same_day_signal_and_fill", True, same_day_chronology_rejected)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        check("account_service_path_delegates_to_core_owner", core_account_path(root), service_account_path(root))
        check("order_service_path_delegates_to_core_owner", core_order_path(root), service_order_path(root))
        check("fill_journal_service_path_delegates_to_core_owner", core_fill_path(root), service_fill_path(root))
        check("account_current_truth_path_is_canonical", root / "state" / "trading" / "account.json", core_account_path(root))
        check("order_current_truth_path_is_canonical", root / "state" / "trading" / "orders.json", core_order_path(root))
        check("fill_journal_path_is_canonical", root / "state" / "trading" / "fill_transaction.json", core_fill_path(root))

    project_root = Path(__file__).resolve().parents[2]
    scale_modules = [
        project_root / "core" / "position_step.py",
        project_root / "core" / "backtest_core.py",
        project_root / "services" / "trading" / "fill_reconciliation.py",
    ]
    raw_scale_sites = []
    for path in scale_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.BinOp, ast.AugAssign)):
                continue
            op = node.op
            if not isinstance(op, (ast.Div, ast.Mult)):
                continue
            values = []
            if isinstance(node, ast.BinOp):
                values = [node.left, node.right]
            elif isinstance(node, ast.AugAssign):
                values = [node.value]
            if any(isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and float(v.value) == 1000.0 for v in values):
                raw_scale_sites.append(f"{path.relative_to(project_root)}:{getattr(node, 'lineno', '?')}")
    check("execution_consumers_do_not_rederive_milli_scale_with_raw_1000_arithmetic", [], raw_scale_sites)

    order_planning_source = (project_root / "services" / "trading" / "order_planning.py").read_text(encoding="utf-8")
    position_step_source = (project_root / "core" / "position_step.py").read_text(encoding="utf-8")
    account_state_source = (project_root / "core" / "trading_account_state.py").read_text(encoding="utf-8")
    order_state_source = (project_root / "core" / "trading_order_state.py").read_text(encoding="utf-8")
    scanner_state_source = (project_root / "services" / "trading" / "scanner_state.py").read_text(encoding="utf-8")
    position_market_source = (project_root / "services" / "trading" / "position_market_context.py").read_text(encoding="utf-8")
    account_service_source = (project_root / "services" / "trading" / "account_state.py").read_text(encoding="utf-8")
    operations_source = (project_root / "services" / "trading" / "operations_status.py").read_text(encoding="utf-8")
    audit_source = (project_root / "services" / "trading" / "operational_audit.py").read_text(encoding="utf-8")
    entry_submission_source = (project_root / "services" / "trading" / "entry_order_submission.py").read_text(encoding="utf-8")
    proposed_state_source = (project_root / "services" / "trading" / "proposed_order_state.py").read_text(encoding="utf-8")
    check("order_planning_has_no_private_duplicate_order_state_path_resolver", False, "def _resolve_trading_order_state_path" in order_planning_source)
    check("account_and_order_state_do_not_duplicate_event_hash_payload_logic", False, "def _event_hash_payload" in account_state_source or "def _event_hash_payload" in order_state_source)
    check("price_state_display_uses_price_semantic_helper_not_money_alias", True, all(fragment in position_step_source for fragment in (
        "milli_to_price(position['highest_high_since_entry_milli'])",
        "milli_to_price(position['trailing_stop_milli'])",
        "milli_to_price(position['sl_milli'])",
    )))
    check("scanner_and_position_market_context_do_not_define_private_trading_date_normalizers", False, "def _normalize_candidate_date" in scanner_state_source or "def normalize_trading_date" in position_market_source)
    check("runtime_account_service_has_no_direct_fill_mutation_bypass", False, "def confirm_trading_strategy_buy_fill(" in account_service_source or "def confirm_trading_sell_fill(" in account_service_source)
    check("operations_reads_proposed_artifact_from_state_contract_not_allocation_producer", True, "from services.trading.proposed_order_state import" in operations_source and "from services.trading.order_planning import" not in operations_source)
    check("prelive_audit_consumes_operations_sell_coverage_ssot", True, "open_position_sell_coverage_safe" in audit_source and "open_position_sell_coverage_blockers" in audit_source)
    check("entry_submission_rechecks_operations_guard_at_broker_submission_boundary", True, "assert_trading_proposed_submission_allowed" in entry_submission_source)
    check("proposed_artifact_contract_is_separate_from_allocation_producer", True, "def load_current_trading_proposed_order_plan" in proposed_state_source and "def build_trading_proposed_order_plan" not in proposed_state_source)

    summary["checks"] = len(results)
    return results, summary



def validate_trading_operational_safety_ssot_contract_case(base_params):
    """Trading allocation/submission/chronology safety must have canonical owners."""
    case_id = "TRADING_OPERATIONAL_SAFETY_SSOT"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_safety_ssot')

    from core.trading_identity import normalize_trading_date, require_trading_date_after

    check("canonical_trading_date_normalizes_iso_date", "2026-09-04", normalize_trading_date("2026-09-04", allow_none=False))
    try:
        require_trading_date_after(
            "2026-09-04",
            after="2026-09-04",
            field_name="fill_date",
            after_field_name="information_date",
        )
    except ValueError:
        same_day_rejected = True
    else:
        same_day_rejected = False
    check("canonical_chronology_rejects_same_day_completed_bar_fill", True, same_day_rejected)

    project_root = Path(__file__).resolve().parents[2]
    order_core = (project_root / "core" / "trading_order_state.py").read_text(encoding="utf-8")
    operations = (project_root / "services" / "trading" / "operations_status.py").read_text(encoding="utf-8")
    planning = (project_root / "services" / "trading" / "order_planning.py").read_text(encoding="utf-8")
    proposed_state = (project_root / "services" / "trading" / "proposed_order_state.py").read_text(encoding="utf-8")
    entry_submission = (project_root / "services" / "trading" / "entry_order_submission.py").read_text(encoding="utf-8")
    generic_order_state = (project_root / "services" / "trading" / "order_state.py").read_text(encoding="utf-8")
    account_service = (project_root / "services" / "trading" / "account_state.py").read_text(encoding="utf-8")
    audit = (project_root / "services" / "trading" / "operational_audit.py").read_text(encoding="utf-8")
    panel = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    coverage = (project_root / "tools" / "local_regression" / "meta_quality_targets.py").read_text(encoding="utf-8")

    check("core_order_state_consumes_canonical_date_chronology_helpers", True, "require_trading_date_after" in order_core and "require_trading_date_not_before" in order_core)
    check("operations_reads_proposed_state_contract_not_allocation_producer", True, "from services.trading.proposed_order_state import" in operations and "from services.trading.order_planning import" not in operations)
    check("proposed_artifact_contract_has_independent_state_owner", True, "PROPOSED_ORDER_SCHEMA_VERSION" in proposed_state and "def load_current_trading_proposed_order_plan" in proposed_state)
    check("allocation_producer_consumes_operations_canonical_guard_before_and_after_compute", True, planning.count("assert_trading_new_allocation_allowed(") >= 2 and "build_trading_operations_status" in planning)
    check("entry_submission_rechecks_operations_safety_at_ordered_boundary", True, entry_submission.count("assert_trading_proposed_submission_allowed(") >= 2 and "pre_persist_guard=source_guard" in entry_submission)
    check("entry_submission_revalidates_account_proposed_and_param_sources", True, "account_sha_before" in entry_submission and "proposed_sha_before" in entry_submission and "selected_params_sha256" in entry_submission)
    check("generic_order_state_no_longer_owns_entry_submission_orchestration", False, "def confirm_trading_order_submission(" in generic_order_state)
    check("runtime_account_service_has_no_direct_fill_mutation_bypass", False, "def confirm_trading_strategy_buy_fill(" in account_service or "def confirm_trading_sell_fill(" in account_service)
    check("operations_consumes_canonical_stop_progress_owner", True, "is_trading_stop_exit_triggered_from_order_rows" in operations)
    check("prelive_audit_consumes_operations_sell_coverage_instead_of_recomputing", True, "open_position_sell_coverage_safe" in audit and "open_position_sell_coverage_blockers" in audit)
    check("workbench_routes_entry_submission_through_dedicated_safety_service", True, "from services.trading.entry_order_submission import confirm_trading_order_submission" in panel)
    check("new_operational_state_and_submission_modules_are_coverage_targets", True, '"services/trading/entry_order_submission.py"' in coverage and '"services/trading/proposed_order_state.py"' in coverage)

    summary["checks"] = len(results)
    return results, summary

def validate_trading_live_reentry_broker_truth_contract_case(base_params):
    """Live re-entry must be rebuilt from broker-confirmed STOP truth and frozen entry voters."""
    case_id = "TRADING_LIVE_REENTRY_BROKER_TRUTH"
    results, summary, check, check_true = bind_synthetic_case(case_id, "trading_live_reentry")

    from core.params_io import params_to_json_dict
    from core.portfolio_param_runtime import build_portfolio_params_signature
    from services.trading.live_reentry import (
        build_trading_live_reentry_candidate_rows,
        build_trading_live_reentry_watch_records,
    )
    from services.trading.strategy_param_runtime import serialize_trading_candidate_member_params
    import services.trading.daily_workflow as daily_workflow
    import services.trading.live_reentry as live_reentry

    params_a = deepcopy(base_params)
    params_b = deepcopy(base_params)
    params_a.use_breakout_reclaim_reentry = True
    params_b.use_breakout_reclaim_reentry = True
    params_b.high_len = int(params_a.high_len) + 5
    payload_a = params_to_json_dict(params_a)
    payload_b = params_to_json_dict(params_b)
    sig_a = build_portfolio_params_signature(params_a)
    sig_b = build_portfolio_params_signature(params_b)

    entry_order = {
        "order_id": "ENTRY-1",
        "side": "BUY",
        "purpose": "ENTRY_BUY",
        "ticker": "2330",
        "ensemble_member_count": 2,
        "ensemble_min_agree": 2,
        "ensemble_member_key": "m1",
        "ensemble_member_keys": ["m1", "m2"],
        "ensemble_member_params_by_key": {"m1": payload_a, "m2": payload_b},
        "ensemble_member_quality_rank_by_key": {},
        "frozen_params": payload_a,
        "fills": [{
            "fill_id": "BUY-1",
            "trade_date": "2026-09-01",
            "confirmed_at": "2026-09-01T10:00:00+08:00",
        }],
    }
    first_position = {
        "ticker": "2330",
        "entry_type": "normal",
        "entry_trade_date": "2026-09-01",
        "entry_fill_price": 100.0,
        "pure_buy_price": 100.0,
        "initial_stop": 90.0,
        "initial_stop_milli": 90_000,
        "sl": 95.0,
        "sl_milli": 95_000,
        "trailing_stop": 95.0,
        "trailing_stop_milli": 95_000,
        "qty": 100,
    }
    stop_order = {
        "order_id": "STOP-1",
        "side": "SELL",
        "purpose": "PROTECTION_STOP",
        "ticker": "2330",
        "entry_order_id": "ENTRY-1",
        "qty": 100,
        "fills": [
            {
                "fill_id": "STOP-F1",
                "trade_date": "2026-09-10",
                "confirmed_at": "2026-09-10T10:00:00+08:00",
                "qty": 40,
                "position_qty_before_fill": 100,
                "position_qty_after_fill": 60,
                "strategy_position_before_fill": dict(first_position),
            },
            {
                "fill_id": "STOP-F2",
                "trade_date": "2026-09-10",
                "confirmed_at": "2026-09-10T10:01:00+08:00",
                "qty": 60,
                "position_qty_before_fill": 60,
                "position_qty_after_fill": 0,
                "strategy_position_before_fill": {**first_position, "qty": 60},
            },
        ],
    }
    order_state = {"orders": {"ENTRY-1": entry_order, "STOP-1": stop_order}}
    empty_account = {"positions": {}}
    with patch.object(live_reentry, "load_trading_order_state", return_value=order_state), patch.object(
        live_reentry, "load_trading_account_state", return_value=empty_account
    ):
        watches = build_trading_live_reentry_watch_records(Path("/tmp/trading-live-reentry"))
    check("full_broker_stop_creates_one_live_reentry_watch", 1, len(watches))
    watch = watches[0]
    check("live_reentry_watch_uses_original_stop_order_qty", 100, watch.get("exit_qty"))
    check("partial_stop_reconciliation_preserves_pre_stop_position_snapshot", 100, (watch.get("position_snapshot") or {}).get("qty"))
    check("live_reentry_watch_preserves_original_voter_keys", ["m1", "m2"], watch.get("member_keys"))
    check("live_reentry_watch_preserves_original_min_agree", 2, watch.get("min_agree"))
    check_true("live_reentry_watch_preserves_both_voter_params", set((watch.get("member_params_by_key") or {})) == {"m1", "m2"})

    partial_only = deepcopy(order_state)
    partial_only["orders"]["STOP-1"]["fills"] = [deepcopy(stop_order["fills"][0])]
    with patch.object(live_reentry, "load_trading_order_state", return_value=partial_only), patch.object(
        live_reentry, "load_trading_account_state", return_value=empty_account
    ):
        check("partial_stop_does_not_start_reentry_watch", [], build_trading_live_reentry_watch_records(Path("/tmp/trading-live-reentry")))

    tp_only = deepcopy(order_state)
    tp_only["orders"]["STOP-1"]["purpose"] = "PROTECTION_TP"
    with patch.object(live_reentry, "load_trading_order_state", return_value=tp_only), patch.object(
        live_reentry, "load_trading_account_state", return_value=empty_account
    ):
        check("tp_fill_does_not_start_reentry_watch", [], build_trading_live_reentry_watch_records(Path("/tmp/trading-live-reentry")))

    later_buy = deepcopy(order_state)
    later_buy["orders"]["ENTRY-2"] = {
        **deepcopy(entry_order),
        "order_id": "ENTRY-2",
        "fills": [{"fill_id": "BUY-2", "trade_date": "2026-09-11", "confirmed_at": "2026-09-11T10:00:00+08:00"}],
    }
    with patch.object(live_reentry, "load_trading_order_state", return_value=later_buy), patch.object(
        live_reentry, "load_trading_account_state", return_value=empty_account
    ):
        check("later_broker_buy_supersedes_old_stop_reentry_watch", [], build_trading_live_reentry_watch_records(Path("/tmp/trading-live-reentry")))

    held_account = {"positions": {"2330": {"broker": {"qty": 100}}}}
    with patch.object(live_reentry, "load_trading_order_state", return_value=order_state), patch.object(
        live_reentry, "load_trading_account_state", return_value=held_account
    ):
        check("currently_held_ticker_suppresses_reentry_watch", [], build_trading_live_reentry_watch_records(Path("/tmp/trading-live-reentry")))

    legacy_ensemble = deepcopy(order_state)
    legacy_ensemble["orders"]["ENTRY-1"].pop("ensemble_member_params_by_key", None)
    with patch.object(live_reentry, "load_trading_order_state", return_value=legacy_ensemble), patch.object(
        live_reentry, "load_trading_account_state", return_value=empty_account
    ):
        try:
            build_trading_live_reentry_watch_records(Path("/tmp/trading-live-reentry"))
        except RuntimeError as exc:
            legacy_fail_closed = "禁止推測 live re-entry" in str(exc)
        else:
            legacy_fail_closed = False
    check("legacy_ensemble_without_voter_lineage_fails_closed", True, legacy_fail_closed)

    candidate_watch = {
        **watch,
        "member_params_by_key": {"m1": payload_a, "m2": payload_b},
        "member_quality_rank_by_key": {},
    }
    fake_frame = pd.DataFrame({
        "Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0],
        "Close": [101.0, 102.0], "Volume": [1000.0, 1000.0],
    }, index=pd.to_datetime(["2026-09-10", "2026-09-11"]))
    plan = {
        "limit_price": 101.0, "init_sl": 95.0, "init_trail": 95.0, "target_price": 110.0,
        "entry_atr": 2.0, "entry_source": "reentry", "source_entry_order_id": "ENTRY-1",
    }
    row_template = {
        "ticker": "2330", "kind": "reentry", "sort_value": 1.0,
        "execution_plan_seed": dict(plan), "entry_source": "reentry", "source_entry_order_id": "ENTRY-1",
    }
    with patch.object(live_reentry, "build_trading_live_reentry_watch_records", return_value=[candidate_watch]), patch.object(
        live_reentry, "open_trading_v2_consumer_view", return_value=object()
    ), patch.object(live_reentry, "load_trading_v2_sanitized_ohlcv_frame", return_value=fake_frame), patch.object(
        live_reentry, "_replay_member_reentry_signal", return_value=({"source": "reentry"}, plan, True)
    ), patch.object(live_reentry, "run_v16_backtest", return_value={}), patch.object(
        live_reentry, "build_extended_scanner_row_from_plan", return_value=row_template
    ):
        live_rows = build_trading_live_reentry_candidate_rows(
            Path("/tmp/trading-live-reentry"), information_date="2026-09-11"
        )
    check("agreeing_frozen_voters_produce_one_live_reentry_candidate", 1, len(live_rows))
    live_row = live_rows[0]
    check("live_reentry_candidate_identity_is_explicit", "reentry", live_row.get("kind"))
    check("live_reentry_candidate_source_entry_is_preserved", "ENTRY-1", live_row.get("source_entry_order_id"))
    check("live_reentry_candidate_lineage_is_frozen_entry_ensemble", "entry_order_frozen_ensemble", live_row.get("param_lineage_source"))
    check("live_reentry_candidate_reapplies_original_min_agree", 2, live_row.get("ensemble_min_agree"))
    check("live_reentry_candidate_records_two_votes", 2, live_row.get("ensemble_vote_count"))
    persisted_lineage = serialize_trading_candidate_member_params(live_row)
    check_true("live_reentry_candidate_persists_both_voter_params", set(persisted_lineage) == {"m1", "m2"})
    check_true("live_reentry_candidate_preserves_representative_signature", str(live_row.get("params_signature") or "") in {sig_a, sig_b})

    from core.file_integrity import canonical_json_sha256
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING
    from core.trading_order_state import build_empty_trading_order_state, append_ordered_trading_proposal, validate_trading_order_state
    from services.trading.proposed_order_state import (
        PROPOSED_ORDER_SCHEMA_VERSION,
        PROPOSED_ORDER_STATUS,
        load_current_trading_proposed_order_plan,
        resolve_trading_proposed_orders_json_path,
    )
    from services.trading.strategy_param_runtime import resolve_trading_candidate_frozen_params

    persisted_live_row = {**live_row, "ensemble_member_params_by_key": persisted_lineage}
    representative_params, representative_lineage = resolve_trading_candidate_frozen_params(persisted_live_row)
    representative_payload = params_to_json_dict(representative_params)
    reentry_proposal = {
        **persisted_live_row,
        "rank": 1,
        "qty": 100,
        "limit_price": 101.0,
        "reserved_cost_milli": 10_100_000,
        "init_sl": 95.0,
        "init_trail": 95.0,
        "target_price": 110.0,
        "entry_atr": 2.0,
        "entry_type": "reentry",
        "kind": "reentry",
        "ensemble_member_params_by_key": persisted_lineage,
    }
    order_plan = {
        "plan_fingerprint": "plan-live-reentry",
        "information_date": "2026-09-11",
        "account_revision": 0,
        "selected_params_sha256": "a" * 64,
        "candidate_snapshot_sha256": "b" * 64,
        "strategy_id": "synthetic",
        "param_selector": "synthetic",
    }
    order_state_seed = build_empty_trading_order_state(timestamp="2026-09-11T15:00:00+08:00", mutation_id="init")
    ordered_state = append_ordered_trading_proposal(
        order_state_seed, order_id="REENTRY-ORDER-1", proposal=reentry_proposal, plan=order_plan,
        timestamp="2026-09-11T15:01:00+08:00", mutation_id="ordered", frozen_params=representative_payload,
    )
    validate_trading_order_state(ordered_state)
    ordered = ordered_state["orders"]["REENTRY-ORDER-1"]
    check("reentry_order_identity_survives_submission", "reentry", ordered.get("entry_type"))
    check("reentry_order_source_entry_survives_submission", "ENTRY-1", ordered.get("source_entry_order_id"))
    check("reentry_order_frozen_lineage_source_survives_submission", "entry_order_frozen_ensemble", ordered.get("param_lineage_source"))
    check("reentry_order_preserves_original_min_agree", 2, ordered.get("ensemble_min_agree"))
    check_true("reentry_order_persists_all_agreeing_voter_params", set(ordered.get("ensemble_member_params_by_key") or {}) == {"m1", "m2"})
    check("reentry_order_frozen_params_match_representative_member", representative_payload, ordered.get("frozen_params"))
    check("reentry_order_representative_key_matches_candidate", live_row.get("ensemble_member_key"), representative_lineage.get("member_key"))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        proposed_path = resolve_trading_proposed_orders_json_path(temp_root)
        proposed_path.parent.mkdir(parents=True, exist_ok=True)
        proposed_payload = {
            "schema_version": PROPOSED_ORDER_SCHEMA_VERSION,
            "status": PROPOSED_ORDER_STATUS,
            "runtime_domain": RUNTIME_DOMAIN_TRADING,
            "information_date": "2026-09-11",
            "strategy_id": "synthetic",
            "param_selector": "synthetic",
            "selected_params_sha256": "a" * 64,
            "candidate_snapshot_sha256": "b" * 64,
            "account_revision": 0,
            "param_member_count": 8,
            "param_min_agree": 5,
            "orders": [reentry_proposal],
        }
        proposed_payload["plan_fingerprint"] = canonical_json_sha256(proposed_payload)
        atomic_write_json(proposed_path, proposed_payload)
        loaded_proposed = load_current_trading_proposed_order_plan(temp_root, require_current=False)
    check("reentry_proposed_order_allows_frozen_row_level_ensemble_identity", "reentry", loaded_proposed["orders"][0].get("entry_type"))
    check("reentry_proposed_order_row_min_agree_is_independent_of_current_artifact", 2, loaded_proposed["orders"][0].get("ensemble_min_agree"))

    historical_reentry = {
        "ticker": "2330", "kind": "extended", "sort_value": 9.0,
        "execution_plan_seed": {"entry_source": "reentry"},
    }
    normal_row = {
        "ticker": "2317", "kind": "buy", "sort_value": 1.0,
        "execution_plan_seed": {"entry_source": "normal"},
    }
    member = {"member_key": "m1", "member_index": 1, "params_obj": params_a, "params_signature": sig_a}
    fake_scan = {"scanned_tickers": ["2330", "2317"], "candidate_rows": [historical_reentry, normal_row], "elapsed_time": 0.1}
    with patch.object(daily_workflow, "run_daily_scanner", return_value=fake_scan):
        filtered = daily_workflow._run_trading_scanner_param_runtime(
            runtime={"data_dir": Path("/tmp/trading-live-reentry"), "param_members": [member], "param_min_agree": 1},
            output_dir=Path("/tmp/trading-live-reentry/out"),
            expected_scanned_tickers=["2330", "2317"],
            prepared_frames={"2330": fake_frame, "2317": fake_frame},
        )
    check("historical_scanner_reentry_is_not_live_candidate_truth", ["2317"], [row.get("ticker") for row in filtered.get("candidate_rows") or []])

    project_root = Path(__file__).resolve().parents[2]
    fill_source = (project_root / "services" / "trading" / "fill_reconciliation.py").read_text(encoding="utf-8")
    consumer_source = (project_root / "services" / "trading" / "market_data_consumer.py").read_text(encoding="utf-8")
    consumer_update_source = (project_root / "services" / "trading" / "market_data_update.py").read_text(encoding="utf-8")
    check_true("stop_fill_persists_pre_fill_strategy_snapshot", "strategy_position_before_fill=stop_snapshot" in fill_source)
    check_true("stop_fill_persists_before_after_broker_qty", "position_qty_before_fill=position_qty_before_fill" in fill_source and "position_qty_after_fill=position_qty_after_fill" in fill_source)
    check_true("reentry_tickers_are_kept_in_v2_training_membership", "set(required_reentry)" in consumer_source and "required_reentry_tickers" in consumer_source)
    check_true("market_update_derives_reentry_data_obligations_from_broker_truth", "resolve_trading_live_reentry_required_tickers" in consumer_update_source)
    check("market_data_consumer_has_no_hardcoded_active_strategy_identity", False, 'get_trading_data_dependency_spec("full_rule_based_no_dl")' in consumer_source)
    check_true("market_data_consumer_uses_canonical_trading_strategy_identity", "get_trading_strategy_profile().strategy_id" in consumer_source)

    summary["checks"] = len(results)
    return results, summary


def validate_trading_market_data_lineage_contract_case(base_params):
    """Trading V2 Data→Params→Scanner lineage is identity-bound and fail-closed."""
    case_id = "TRADING_MARKET_DATA_LINEAGE"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'trading_data_lineage')

    from core.active_param_ensemble import build_static_active_param_ensemble_payload
    from core.file_integrity import canonical_json_sha256, compute_file_sha256
    from core.params_io import params_to_json_dict
    from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
    from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
    import services.trading.daily_workflow as daily_workflow
    import services.trading.strategy_param_training as param_training
    from services.trading.market_data_consumer import (
        TRADING_V2_CONSUMER_STATE_RELATIVE_PATH,
        get_trading_v2_consumer_state_sha256,
        load_trading_v2_consumer_state,
    )
    from services.trading.scanner_state import (
        load_trading_candidate_snapshot,
        load_trading_scanner_runtime,
        resolve_trading_candidate_snapshot_path,
    )
    from services.trading.strategy_param_state import (
        load_trading_strategy_param_binding,
        publish_trading_strategy_param_binding,
    )

    profile = get_trading_strategy_profile()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = resolve_runtime_domain_paths(
            root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
        )
        legacy_data_dir = Path(paths.data_dir)
        legacy_data_dir.mkdir(parents=True)
        csv_path = legacy_data_dir / "2330.csv"
        base_frame = pd.DataFrame({
            "Date": ["2026-09-03", "2026-09-04"],
            "Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0],
            "Close": [101.0, 102.0], "Volume": [1000, 1100],
        })
        base_frame.to_csv(csv_path, index=False)
        original_legacy_bytes = csv_path.read_bytes()

        selected_path = Path(resolve_trading_selected_strategy_param_path(root))
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(json.dumps(build_static_active_param_ensemble_payload(
            members=[{"member_index": 1, "seed": 1, "params": params_to_json_dict(base_params)}],
            selector=profile.param_selector,
            meta={"selected_model_mode": "trade", "walk_forward_policy": {"latest_data_date": "2026-09-04"}},
        ), ensure_ascii=False), encoding="utf-8")

        _publish_synthetic_trading_input_lineage(
            root,
            market_date="2026-09-04",
            current_universe_tickers=["2330"],
        )
        consumer = load_trading_v2_consumer_state(root, required=True, verify_current_view=True)
        binding = load_trading_strategy_param_binding(
            root, required=True, verify_current=True, verify_dataset_content=True
        )
        runtime = load_trading_scanner_runtime(root, verify_dataset_content=True)
        consumer_state_sha = get_trading_v2_consumer_state_sha256(root)
        check("params_bind_exact_v2_view_identity", consumer["source_view_fingerprint"], binding["market_data_source_view_fingerprint"])
        check("scanner_runtime_consumes_same_v2_view_identity", binding["market_data_source_view_fingerprint"], runtime["market_data_source_view_fingerprint"])
        check("params_bind_exact_v2_consumer_state_sha", consumer_state_sha, binding["market_data_consumer_state_sha256"])
        check("scanner_runtime_consumes_same_v2_consumer_state_sha", consumer_state_sha, runtime["market_data_consumer_state_sha256"])
        check("scanner_runtime_reports_v2_market_data_source", "trading_market_data_v2_historical_latest_view", runtime["market_data_source"])

        changed_legacy = base_frame.copy()
        changed_legacy.loc[1, "Close"] = 999.0
        changed_legacy.to_csv(csv_path, index=False)
        legacy_ignored = load_trading_scanner_runtime(root, verify_dataset_content=True)
        check("legacy_csv_content_drift_no_longer_changes_v2_lineage", runtime["market_data_source_view_fingerprint"], legacy_ignored["market_data_source_view_fingerprint"])
        check("legacy_csv_content_drift_no_longer_changes_v2_consumer_state_sha", runtime["market_data_consumer_state_sha256"], legacy_ignored["market_data_consumer_state_sha256"])
        csv_path.write_bytes(original_legacy_bytes)

        legacy_snapshot_path = root / "state" / "trading" / "market_data_snapshot.json"
        legacy_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_snapshot_path.write_text('{"obsolete": true}', encoding="utf-8")
        legacy_snapshot_ignored = load_trading_scanner_runtime(root, verify_dataset_content=True)
        check("retired_legacy_snapshot_file_no_longer_changes_v2_lineage", runtime["market_data_source_view_fingerprint"], legacy_snapshot_ignored["market_data_source_view_fingerprint"])

        consumer_state_path = root / TRADING_V2_CONSUMER_STATE_RELATIVE_PATH
        original_consumer_state = consumer_state_path.read_bytes()
        _mutate_synthetic_trading_v2_consumer_membership(root, ticker="9999")
        try:
            load_trading_strategy_param_binding(
                root, required=True, verify_current=True, verify_dataset_content=True
            )
        except RuntimeError:
            v2_state_invalidates_params = True
        else:
            v2_state_invalidates_params = False
        check("v2_consumer_state_change_invalidates_existing_params_binding", True, v2_state_invalidates_params)
        try:
            load_trading_scanner_runtime(root, verify_dataset_content=True)
        except RuntimeError:
            v2_state_invalidates_scanner = True
        else:
            v2_state_invalidates_scanner = False
        check("v2_consumer_state_change_invalidates_scanner_runtime", True, v2_state_invalidates_scanner)
        consumer_state_path.write_bytes(original_consumer_state)

        fake_scan = {
            "count_scanned": 1, "elapsed_time": 0.01, "count_history_qualified": 1,
            "count_skipped_insufficient": 0, "count_sanitized_candidates": 0, "max_workers": 1,
            "pool_start_method": "spawn", "scanner_issue_log_path": None,
            "scanned_tickers": ["2330"],
            "candidate_rows": [{
                "ticker": "2330", "trade_date": "2026-09-04", "kind": "buy", "sort_value": 1.0,
                "expected_value": 0.2,
                "execution_plan_seed": {
                    "ticker": "2330", "trade_date": "2026-09-04", "limit_price": 102.0,
                    "init_sl": 98.0, "init_trail": 99.0, "target_price": 106.0, "entry_atr": 2.0,
                },
            }],
        }

        def _scan_and_mutate(*_args, **_kwargs):
            _mutate_synthetic_trading_v2_consumer_membership(root, ticker="9999")
            return fake_scan

        with patch.object(daily_workflow, "run_daily_scanner", side_effect=_scan_and_mutate):
            try:
                daily_workflow.run_trading_candidate_scan(project_root=root)
            except RuntimeError:
                scan_toctou_rejected = not resolve_trading_candidate_snapshot_path(root).is_file()
            else:
                scan_toctou_rejected = False
        check("scanner_refuses_publish_when_v2_consumer_state_changes_during_scan", True, scan_toctou_rejected)
        consumer_state_path.write_bytes(original_consumer_state)

        def _optimizer_and_mutate(**_kwargs):
            _mutate_synthetic_trading_v2_consumer_membership(root, ticker="9999")
            return {
                "selected_params_path": str(selected_path),
                "manifest_path": str(selected_path.parent / "manifest.json"),
            }

        with patch.object(param_training, "run_static_strategy_parameter_training", side_effect=_optimizer_and_mutate):
            try:
                param_training.run_trading_strategy_param_training(project_root=root, environ={})
            except RuntimeError as exc:
                params_toctou_rejected = "consumer state" in str(exc) or "view identity" in str(exc)
            else:
                params_toctou_rejected = False
        check("params_producer_refuses_publish_when_v2_consumer_state_changes_during_training", True, params_toctou_rejected)
        consumer_state_path.write_bytes(original_consumer_state)
        publish_trading_strategy_param_binding(root)

        with patch.object(daily_workflow, "run_daily_scanner", return_value=fake_scan):
            daily_workflow.run_trading_candidate_scan(project_root=root)

        from services.trading.proposed_order_state import (
            PROPOSED_ORDER_SCHEMA_VERSION,
            PROPOSED_ORDER_STATUS,
            load_current_trading_proposed_order_plan,
            resolve_trading_proposed_orders_json_path,
        )
        account = initialize_trading_account_state(root, cash=500_000)
        current_runtime = load_trading_scanner_runtime(root, verify_dataset_content=True)
        proposed_payload = {
            "schema_version": PROPOSED_ORDER_SCHEMA_VERSION,
            "status": PROPOSED_ORDER_STATUS,
            "runtime_domain": "trading",
            "information_date": current_runtime["latest_data_date"],
            "strategy_id": current_runtime["profile"].strategy_id,
            "param_selector": current_runtime["profile"].param_selector,
            "selected_params_sha256": current_runtime["selected_params_sha256"],
            "param_member_count": int(current_runtime["member_count"]),
            "param_min_agree": int(current_runtime["param_min_agree"]),
            "candidate_snapshot_sha256": compute_file_sha256(resolve_trading_candidate_snapshot_path(root)),
            "account_revision": int(account["revision"]),
            "reserved_total": 0.0,
            "orders": [],
        }
        proposed_payload["plan_fingerprint"] = canonical_json_sha256(proposed_payload)
        atomic_write_json(resolve_trading_proposed_orders_json_path(root), proposed_payload)

        _mutate_synthetic_trading_v2_consumer_membership(root, ticker="9999")
        try:
            load_trading_candidate_snapshot(root, require_current=True)
        except RuntimeError:
            candidate_v2_drift_rejected = True
        else:
            candidate_v2_drift_rejected = False
        check("candidate_currentness_rechecks_v2_consumer_state_identity", True, candidate_v2_drift_rejected)
        try:
            load_current_trading_proposed_order_plan(root, require_current=True)
        except RuntimeError:
            proposed_v2_drift_rejected = True
        else:
            proposed_v2_drift_rejected = False
        check("proposed_order_currentness_rechecks_v2_consumer_state_identity", True, proposed_v2_drift_rejected)
        consumer_state_path.write_bytes(original_consumer_state)

    summary["checks"] = len(results)
    return results, summary
