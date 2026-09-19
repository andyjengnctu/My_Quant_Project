"""Canonical Trading account-state schema and pure mutation semantics."""
from __future__ import annotations

from core.entry_plans import build_position_from_frozen_entry_plan
from core.position_management import acknowledge_position_exit, POSITION_MANAGEMENT_FIELDS

from copy import deepcopy
from datetime import date, datetime
import math
from typing import Any

from core.entry_plans import build_position_from_entry_fill
from core.event_hash_chain import compute_event_hash
from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_average_price_from_total_milli,
    calc_initial_risk_total_milli,
    milli_to_money,
    milli_to_price,
    money_to_milli,
    rate_to_ppm,
    sync_position_display_fields,
)
from core.file_integrity import canonical_json_sha256
from core.trading_identity import (
    normalize_trading_date,
    normalize_trading_ticker,
    require_trading_date_after,
)
from core.position_step import execute_confirmed_position_sell_fill
from core.runtime_domains import RUNTIME_DOMAIN_TRADING

TRADING_ACCOUNT_SCHEMA_VERSION = 1
TRADING_ACCOUNT_STATE_FILENAME = "account.json"
TRADING_RUNTIME_DOMAIN = RUNTIME_DOMAIN_TRADING
POSITION_SOURCE_MANUAL_ADOPTED = "manual_adopted"
POSITION_SOURCE_MANUAL_MANAGED = "manual_managed"
POSITION_SOURCE_STRATEGY_FILL = "strategy_fill"
MANAGED_POSITION_SOURCES = (POSITION_SOURCE_STRATEGY_FILL, POSITION_SOURCE_MANUAL_MANAGED)
POSITION_SOURCES = (POSITION_SOURCE_MANUAL_ADOPTED, *MANAGED_POSITION_SOURCES)
MANAGEMENT_STATUS_UNMANAGED = "unmanaged"
MANAGEMENT_STATUS_ACTIVE = "active"
MANAGEMENT_STATUSES = (MANAGEMENT_STATUS_UNMANAGED, MANAGEMENT_STATUS_ACTIVE)
MANAGEMENT_SELL_SIGNAL_STOP = "STOP EXIT"
MANAGEMENT_SELL_SIGNAL_INDICATOR = "INDICATOR SELL"
MANAGEMENT_SELL_SIGNALS = (MANAGEMENT_SELL_SIGNAL_STOP, MANAGEMENT_SELL_SIGNAL_INDICATOR)
ACCOUNT_MUTATION_RECORD_MANAGEMENT_SELL_SIGNAL = "record_strategy_management_sell_signal"


TRADE_MUTATION_BUY = "manual_buy_fill"
TRADE_MUTATION_MANUAL_MANAGED_BUY = "manual_managed_buy_fill"
TRADE_MUTATION_STRATEGY_BUY = "confirm_strategy_buy_fill"
TRADE_MUTATION_STRATEGY_BUY_INCREMENT = "confirm_strategy_buy_fill_increment"
TRADE_MUTATION_SELL = "confirm_sell_fill"
TRADE_MUTATION_VOID = "void_manual_trade"
ACCOUNT_MUTATION_HISTORICAL_INVENTORY = "historical_inventory_reconciliation"
ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT = "activate_manual_management"
TRADE_MUTATIONS = (
    TRADE_MUTATION_BUY,
    TRADE_MUTATION_MANUAL_MANAGED_BUY,
    TRADE_MUTATION_STRATEGY_BUY,
    TRADE_MUTATION_STRATEGY_BUY_INCREMENT,
    TRADE_MUTATION_SELL,
)


def _voided_trade_revisions(state: dict[str, Any]) -> set[int]:
    revisions: set[int] = set()
    for event in state.get("events", []):
        if str(event.get("mutation_type") or "") != TRADE_MUTATION_VOID:
            continue
        details = event.get("details") or {}
        target = details.get("target_revision")
        if target is not None:
            revisions.add(int(target))
    return revisions


def effective_trading_account_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical account events with voided manual trades excluded.

    Void/correction events stay in the immutable audit trail but are not economic
    transactions.  Consumers that reconstruct transaction history must use this
    projection instead of reading raw trade events directly.
    """
    validate_trading_account_state(state)
    voided = _voided_trade_revisions(state)
    rows: list[dict[str, Any]] = []
    for event in state.get("events", []):
        mutation = str(event.get("mutation_type") or "")
        revision = int(event.get("revision") or 0)
        if mutation == TRADE_MUTATION_VOID:
            continue
        if mutation in TRADE_MUTATIONS and revision in voided:
            continue
        rows.append(event)
    return rows


def _effective_trade_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event for event in effective_trading_account_events(state)
        if str(event.get("mutation_type") or "") in TRADE_MUTATIONS
    ]


def _trade_event_by_revision(state: dict[str, Any], revision: int) -> dict[str, Any]:
    target = int(revision)
    for event in _effective_trade_events(state):
        if int(event.get("revision") or -1) == target:
            return event
    raise ValueError(f"找不到可修改的有效交易明細 revision={target}")


def _latest_effective_trade_revision_for_ticker(state: dict[str, Any], ticker: str) -> int | None:
    ticker_key = _normalize_ticker(ticker)
    revisions = []
    for event in _effective_trade_events(state):
        details = event.get("details") or {}
        if _normalize_ticker(details.get("ticker")) == ticker_key:
            revisions.append(int(event.get("revision") or 0))
    return max(revisions) if revisions else None


def _strategy_template_by_ticker(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Best available strategy-state snapshot for economic replay.

    Historical broker/account corrections may need to reopen a strategy position.
    Current open state is preferred; otherwise use the newest immutable position
    snapshot carried by a sell/reconciliation event.  Legacy records that predate
    snapshots intentionally fall back to unmanaged broker truth rather than
    fabricating stop/target state.
    """
    templates: dict[str, tuple[int, dict[str, Any]]] = {}
    for ticker, record in dict(state.get("positions") or {}).items():
        if str(record.get("source") or "") in MANAGED_POSITION_SOURCES:
            templates[_normalize_ticker(ticker)] = (10**12, deepcopy(record))
    for event in state.get("events", []):
        revision = int(event.get("revision") or 0)
        details = dict(event.get("details") or {})
        for key in ("position_before", "position_after", "position", "previous_position", "removed_position"):
            candidate = details.get(key)
            if not isinstance(candidate, dict) or candidate.get("source") not in MANAGED_POSITION_SOURCES:
                continue
            ticker = candidate.get("ticker") or details.get("ticker")
            if not ticker:
                continue
            ticker_key = _normalize_ticker(ticker)
            previous = templates.get(ticker_key)
            if previous is None or revision >= previous[0]:
                templates[ticker_key] = (revision, deepcopy(candidate))
    return {ticker: record for ticker, (_revision, record) in templates.items()}


def _manual_record_from_broker(ticker: str, broker: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "source": POSITION_SOURCE_MANUAL_ADOPTED,
        "broker": deepcopy(broker),
        "strategy_management": {
            "status": MANAGEMENT_STATUS_UNMANAGED,
            "management_start_date": None,
            "position_state": None,
        },
    }


def _record_from_replayed_broker(
    ticker: str,
    source: str,
    broker: dict[str, Any],
    *,
    strategy_templates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach safe management state to a replayed broker inventory."""
    if source not in MANAGED_POSITION_SOURCES:
        return _manual_record_from_broker(ticker, broker)
    template = strategy_templates.get(ticker)
    if not isinstance(template, dict):
        # Old account histories may not carry the pre-sell strategy snapshot.
        # Reopening such inventory must preserve broker/account truth without
        # inventing historical strategy stops, so it becomes unmanaged truth.
        return _manual_record_from_broker(ticker, broker)
    record = deepcopy(template)
    record["ticker"] = ticker
    record["source"] = source
    replay_broker = deepcopy(broker)
    template_broker = dict(template.get("broker") or {})
    if not replay_broker.get("entry_order_id") and template_broker.get("entry_order_id"):
        replay_broker["entry_order_id"] = template_broker.get("entry_order_id")
    record["broker"] = replay_broker
    management = deepcopy(record.get("strategy_management") or {})
    position_state = deepcopy(management.get("position_state") or {})
    if not position_state:
        return _manual_record_from_broker(ticker, broker)
    qty = int(broker.get("qty") or 0)
    cost = int(broker.get("remaining_cost_basis_milli") or 0)
    gross = broker.get("remaining_gross_buy_milli")
    fee = broker.get("remaining_buy_fee_milli")
    price_basis = cost if gross is None else int(gross)
    avg_price_milli = max(1, (price_basis + max(qty, 1) // 2) // max(qty, 1))
    previous_initial_qty = max(1, int(position_state.get("initial_qty") or qty or 1))
    previous_risk = int(position_state.get("initial_risk_total_milli") or 0)
    position_state["qty"] = qty
    position_state["initial_qty"] = int(broker.get("initial_qty") or qty)
    position_state["net_buy_total_milli"] = int(broker.get("initial_cost_basis_milli") or cost)
    position_state["remaining_cost_basis_milli"] = cost
    if gross is not None:
        position_state["gross_buy_milli"] = int(broker.get("initial_gross_buy_milli") or gross)
    if fee is not None:
        position_state["buy_fee_milli"] = int(broker.get("initial_buy_fee_milli") or fee)
    position_state["entry_fill_price_milli"] = avg_price_milli
    position_state["entry_fill_price"] = milli_to_price(avg_price_milli)
    position_state["pure_buy_price_milli"] = avg_price_milli
    position_state["pure_buy_price"] = milli_to_price(avg_price_milli)
    if previous_risk:
        position_state["initial_risk_total_milli"] = int(round(previous_risk * qty / previous_initial_qty))
    management["status"] = MANAGEMENT_STATUS_ACTIVE
    if broker.get("entry_date"):
        position_state["entry_trade_date"] = str(broker["entry_date"])
        if management.get("management_start_date") != str(broker["entry_date"]):
            management["management_start_date"] = str(broker["entry_date"])
            management.pop("evaluation_context_fingerprint", None)
            management.pop("evaluated_through_date", None)
    management["position_state"] = _json_safe(sync_position_display_fields(position_state))
    record["strategy_management"] = management
    return record


def _project_trading_account_economics(
    state: dict[str, Any],
    *,
    accounting_params=None,
    extra_void_revisions: set[int] | None = None,
    synthetic_trade_events: list[dict[str, Any]] | None = None,
    allow_historical_reconciliation: bool = False,
) -> dict[str, Any]:
    """Replay effective account economics from immutable events.

    The replay is the canonical correction engine.  It lets a historical trade be
    voided/replaced without reverse-mutating whatever happens to be the current
    position.  Later trades are replayed against the corrected history, and later
    explicit cash/broker-truth reconciliations naturally override earlier data.
    """
    validate_trading_account_state(state)
    voided = _voided_trade_revisions(state) | {int(v) for v in (extra_void_revisions or set())}
    strategy_templates = _strategy_template_by_ticker(state)

    # Historical inventory reconciliation is tied to an effective SELL at a
    # logical revision.  If that SELL is later voided, its reconciliation must
    # disappear from the economic replay as well instead of creating phantom
    # inventory.  Replacement SELL events retain the same logical revision.
    active_sell_logical_revisions: set[int] = set()
    for existing_event in state.get("events", []):
        existing_revision = int(existing_event.get("revision") or 0)
        if existing_revision in voided:
            continue
        if str(existing_event.get("mutation_type") or "") != TRADE_MUTATION_SELL:
            continue
        existing_details = dict(existing_event.get("details") or {})
        active_sell_logical_revisions.add(
            int(existing_details.get("replacement_for_revision") or existing_revision)
        )
    for raw_synthetic in synthetic_trade_events or []:
        if str(raw_synthetic.get("mutation_type") or "") != TRADE_MUTATION_SELL:
            continue
        synthetic_details = dict(raw_synthetic.get("details") or {})
        active_sell_logical_revisions.add(
            int(synthetic_details.get("replacement_for_revision") or raw_synthetic.get("logical_revision") or raw_synthetic.get("revision") or 0)
        )

    timeline: list[tuple[int, int, int, dict[str, Any]]] = []
    for event in state.get("events", []):
        mutation = str(event.get("mutation_type") or "")
        revision = int(event.get("revision") or 0)
        if mutation == TRADE_MUTATION_VOID:
            continue
        if mutation in TRADE_MUTATIONS and revision in voided:
            continue
        details = dict(event.get("details") or {})
        if mutation == ACCOUNT_MUTATION_HISTORICAL_INVENTORY:
            logical_revision = int(details.get("effective_before_revision") or revision)
            priority = -1
        else:
            logical_revision = int(details.get("replacement_for_revision") or revision)
            priority = 0
        timeline.append((logical_revision, priority, revision, event))
    for idx, raw in enumerate(synthetic_trade_events or []):
        event = deepcopy(raw)
        details = dict(event.get("details") or {})
        logical_revision = int(details.get("replacement_for_revision") or event.get("logical_revision") or 0)
        timeline.append((logical_revision, 0, 10**12 + idx, event))
    timeline.sort(key=lambda item: (item[0], item[1], item[2]))

    cash_milli = None
    positions: dict[str, dict[str, Any]] = {}
    projected_trades: list[dict[str, Any]] = []
    buy_dates: dict[str, set[str]] = {}
    sell_dates: dict[str, set[str]] = {}
    reconciliation_events: list[dict[str, Any]] = []

    def ensure_nonnegative_cash(context: str):
        # Historical account files may begin after the real broker cash history.
        # Intermediate negative cash is therefore not a legal reason to block an
        # otherwise valid correction.  Current/final cash is validated below.
        return None

    for _logical, _priority, actual_revision, event in timeline:
        mutation = str(event.get("mutation_type") or "")
        details = deepcopy(event.get("details") or {})
        if mutation == "initialize":
            value = details.get("cash_milli")
            cash_milli = None if value is None else int(value)
            continue
        if mutation == "set_cash_balance":
            value = details.get("cash_milli")
            previous = details.get("previous_cash_milli")
            if value is None:
                cash_milli = None
            elif cash_milli is None or previous is None:
                cash_milli = int(value)
            else:
                # Reconciliation is an account-cash adjustment, not a second
                # opening-balance anchor.  Reapply the same delta after historical
                # trade edits so deleting a BUY restores its cash even when a later
                # cash reconciliation exists.
                cash_milli = int(cash_milli) + (int(value) - int(previous))
            continue

        ticker_raw = details.get("ticker")
        ticker = None if ticker_raw is None else _normalize_ticker(ticker_raw)
        if mutation == ACCOUNT_MUTATION_HISTORICAL_INVENTORY and ticker:
            required_sell_logical_revision = int(
                details.get("required_sell_logical_revision")
                or details.get("inferred_from_sell_revision")
                or details.get("effective_before_revision")
                or 0
            )
            if required_sell_logical_revision and required_sell_logical_revision not in active_sell_logical_revisions:
                continue
            add_qty = int(details.get("qty") or 0)
            add_cost = int(details.get("cost_basis_milli") or 0)
            if add_qty <= 0 or add_cost <= 0:
                raise ValueError(f"{ticker} historical inventory reconciliation 不合法")
            source = str(details.get("source") or POSITION_SOURCE_MANUAL_ADOPTED)
            add_gross = details.get("gross_buy_milli")
            add_fee = details.get("buy_fee_milli")
            existing = positions.get(ticker)
            if existing is None:
                broker_seed = {
                    "qty": add_qty, "initial_qty": add_qty,
                    "initial_cost_basis_milli": add_cost, "remaining_cost_basis_milli": add_cost,
                    "realized_pnl_milli": 0, "entry_date": details.get("entry_date"),
                }
                if add_gross is not None and add_fee is not None:
                    broker_seed.update({
                        "initial_gross_buy_milli": int(add_gross), "remaining_gross_buy_milli": int(add_gross),
                        "initial_buy_fee_milli": int(add_fee), "remaining_buy_fee_milli": int(add_fee),
                    })
                positions[ticker] = _record_from_replayed_broker(
                    ticker, source, broker_seed, strategy_templates=strategy_templates
                )
            else:
                broker = existing["broker"]
                broker["qty"] = int(broker.get("qty") or 0) + add_qty
                broker["initial_qty"] = int(broker.get("initial_qty") or 0) + add_qty
                broker["remaining_cost_basis_milli"] = int(broker.get("remaining_cost_basis_milli") or 0) + add_cost
                broker["initial_cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or 0) + add_cost
                if add_gross is not None and add_fee is not None and all(k in broker for k in (
                    "initial_gross_buy_milli", "remaining_gross_buy_milli", "initial_buy_fee_milli", "remaining_buy_fee_milli"
                )):
                    broker["initial_gross_buy_milli"] += int(add_gross)
                    broker["remaining_gross_buy_milli"] += int(add_gross)
                    broker["initial_buy_fee_milli"] += int(add_fee)
                    broker["remaining_buy_fee_milli"] += int(add_fee)
                positions[ticker] = _record_from_replayed_broker(
                    ticker, str(existing.get("source") or source), broker, strategy_templates=strategy_templates
                )
            continue
        if mutation == "adopt_manual_position" and ticker:
            qty = int(details.get("qty") or 0)
            cost = int(details.get("cost_basis_total_milli") or 0)
            if qty <= 0 or cost <= 0:
                raise ValueError(f"{ticker} 歷史庫存匯入資料不合法")
            positions[ticker] = _manual_record_from_broker(ticker, {
                "qty": qty,
                "initial_qty": qty,
                "initial_cost_basis_milli": cost,
                "remaining_cost_basis_milli": cost,
                "realized_pnl_milli": 0,
                "entry_date": details.get("entry_date"),
            })
            continue
        if mutation == ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT and ticker:
            existing = positions.get(ticker)
            if not isinstance(existing, dict):
                raise ValueError(f"{ticker} 啟用手選策略管理時不存在可接管庫存")
            positions[ticker] = _record_from_replayed_broker(
                ticker,
                POSITION_SOURCE_MANUAL_MANAGED,
                deepcopy(existing.get("broker") or {}),
                strategy_templates=strategy_templates,
            )
            if positions[ticker].get("source") != POSITION_SOURCE_MANUAL_MANAGED:
                raise ValueError(f"{ticker} 缺少可重播的手選策略管理 template")
            continue
        if mutation == "correct_manual_position" and ticker:
            broker = details.get("broker")
            if isinstance(broker, dict):
                positions[ticker] = _manual_record_from_broker(ticker, broker)
            continue
        if mutation == "remove_manual_position" and ticker:
            positions.pop(ticker, None)
            continue
        if mutation == "correct_position_broker_truth" and ticker:
            position = details.get("position")
            if isinstance(position, dict):
                positions[ticker] = deepcopy(position)
            continue
        if mutation == "remove_position_broker_truth" and ticker:
            positions.pop(ticker, None)
            continue
        if mutation not in TRADE_MUTATIONS or not ticker:
            # rollforward and other management-only events do not change broker
            # inventory/cash.  Current management state is reattached after replay.
            continue

        trade_date = _normalize_iso_date(details.get("trade_date"), field_name="trade_date")
        if trade_date is None:
            raise ValueError(f"{ticker} 交易缺少 trade_date")
        economic_revision = int(event.get("revision") or actual_revision)

        if mutation in {TRADE_MUTATION_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT}:
            if trade_date in sell_dates.get(ticker, set()):
                raise ValueError(f"{ticker} 修正後同一交易日同時存在買入與賣出成交")
            buy_dates.setdefault(ticker, set()).add(trade_date)
            qty = int(details.get("fill_qty") or details.get("qty") or 0)
            price_milli = details.get("fill_price_milli")
            if price_milli is None:
                price_milli = details.get("entry_fill_price_milli")
            if accounting_params is not None and price_milli is not None and qty > 0:
                current_ledger = build_buy_ledger_from_price(milli_to_price(int(price_milli)), qty, accounting_params)
                gross = int(current_ledger["gross_buy_milli"])
                fee = int(current_ledger["buy_fee_milli"])
                net = int(current_ledger["net_buy_total_milli"])
                price_milli = int(current_ledger["fill_price_milli"])
            else:
                gross = int(details.get("gross_buy_milli") or 0)
                fee = int(details.get("buy_fee_milli") or 0)
                net = int(details.get("net_buy_total_milli") or (gross + fee))
            if qty <= 0 or net <= 0:
                raise ValueError(f"{ticker} 修正後買入資料不合法")
            if cash_milli is None:
                raise ValueError("Trading cash 尚未設定，無法重建交易帳務")
            cash_milli = int(cash_milli) - net
            ensure_nonnegative_cash(f"{ticker} {trade_date} 買入")
            source = (
                POSITION_SOURCE_MANUAL_ADOPTED if mutation == TRADE_MUTATION_BUY
                else POSITION_SOURCE_MANUAL_MANAGED if mutation == TRADE_MUTATION_MANUAL_MANAGED_BUY
                else POSITION_SOURCE_STRATEGY_FILL
            )
            existing = positions.get(ticker)
            if existing is None:
                broker = {
                    "qty": qty,
                    "initial_qty": qty,
                    "initial_cost_basis_milli": net,
                    "remaining_cost_basis_milli": net,
                    "initial_gross_buy_milli": gross,
                    "remaining_gross_buy_milli": gross,
                    "initial_buy_fee_milli": fee,
                    "remaining_buy_fee_milli": fee,
                    "realized_pnl_milli": 0,
                    "entry_date": trade_date,
                }
                positions[ticker] = _record_from_replayed_broker(
                    ticker, source, broker, strategy_templates=strategy_templates
                )
            else:
                existing_source = str(existing.get("source") or "")
                if existing_source != source:
                    raise ValueError(f"{ticker} 修正後買入來源與既有庫存來源衝突")
                broker = existing["broker"]
                old_qty = int(broker.get("qty") or 0)
                old_cost = int(broker.get("remaining_cost_basis_milli") or 0)
                broker["qty"] = old_qty + qty
                broker["initial_qty"] = int(broker.get("initial_qty") or old_qty) + qty
                broker["remaining_cost_basis_milli"] = old_cost + net
                broker["initial_cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or old_cost) + net
                if all(key in broker for key in (
                    "initial_gross_buy_milli", "remaining_gross_buy_milli",
                    "initial_buy_fee_milli", "remaining_buy_fee_milli",
                )):
                    broker["initial_gross_buy_milli"] = int(broker["initial_gross_buy_milli"]) + gross
                    broker["remaining_gross_buy_milli"] = int(broker["remaining_gross_buy_milli"]) + gross
                    broker["initial_buy_fee_milli"] = int(broker["initial_buy_fee_milli"]) + fee
                    broker["remaining_buy_fee_milli"] = int(broker["remaining_buy_fee_milli"]) + fee
                entry_date = broker.get("entry_date")
                if entry_date is None or trade_date < str(entry_date):
                    broker["entry_date"] = trade_date
                positions[ticker] = _record_from_replayed_broker(
                    ticker, source, broker, strategy_templates=strategy_templates
                )
            broker_after = positions[ticker]["broker"]
            projected = deepcopy(details)
            projected.update({
                "ticker": ticker,
                "qty": qty if mutation != TRADE_MUTATION_STRATEGY_BUY_INCREMENT else details.get("qty"),
                "fill_qty": qty if mutation == TRADE_MUTATION_STRATEGY_BUY_INCREMENT else details.get("fill_qty"),
                "trade_date": trade_date,
                "fill_price_milli": None if price_milli is None else int(price_milli),
                "gross_buy_milli": gross,
                "buy_fee_milli": fee,
                "net_buy_total_milli": net,
                "remaining_qty": int(broker_after.get("qty") or 0),
                "remaining_cost_basis_milli": int(broker_after.get("remaining_cost_basis_milli") or 0),
                "cash_milli_after": int(cash_milli),
            })
            projected_trades.append({"revision": economic_revision, "mutation_type": mutation, "details": projected})
            continue

        # SELL
        if trade_date in buy_dates.get(ticker, set()):
            raise ValueError(f"{ticker} 修正後同一交易日同時存在買入與賣出成交")
        sell_dates.setdefault(ticker, set()).add(trade_date)
        existing = positions.get(ticker)
        qty = int(details.get("qty") or 0)
        held_qty = 0 if existing is None else int((existing.get("broker") or {}).get("qty") or 0)
        if qty <= 0:
            raise ValueError(f"{ticker} 修正後賣出股數不合法")
        if held_qty < qty:
            if not allow_historical_reconciliation:
                raise ValueError(f"{ticker} 修正後賣出股數超過當時庫存：sell={qty}, held={held_qty}")
            position_before = details.get("position_before")
            target_broker = dict(position_before.get("broker") or {}) if isinstance(position_before, dict) else {}
            target_qty = int(target_broker.get("qty") or 0)
            target_cost = int(target_broker.get("remaining_cost_basis_milli") or 0)
            target_gross = target_broker.get("remaining_gross_buy_milli")
            target_fee = target_broker.get("remaining_buy_fee_milli")
            if target_qty < qty or target_cost <= 0:
                remaining_qty_hint = max(0, int(details.get("remaining_qty") or 0))
                target_qty = max(qty, qty + remaining_qty_hint)
                allocated_cost_hint = int(details.get("allocated_cost_milli") or 0)
                if allocated_cost_hint <= 0:
                    raise ValueError(f"{ticker} 修正後賣出缺少可重建的庫存成本")
                target_cost = int(round((allocated_cost_hint / qty) * target_qty))
                allocated_gross_hint = details.get("allocated_gross_buy_milli")
                if allocated_gross_hint is not None:
                    target_gross = int(round((int(allocated_gross_hint) / qty) * target_qty))
                    target_fee = max(0, target_cost - int(target_gross))
            current_cost = 0 if existing is None else int((existing.get("broker") or {}).get("remaining_cost_basis_milli") or 0)
            current_gross = None if existing is None else (existing.get("broker") or {}).get("remaining_gross_buy_milli")
            current_fee = None if existing is None else (existing.get("broker") or {}).get("remaining_buy_fee_milli")
            add_qty = target_qty - held_qty
            add_cost = max(1, target_cost - current_cost)
            add_gross = None if target_gross is None else max(0, int(target_gross) - int(current_gross or 0))
            add_fee = None if target_fee is None else max(0, int(target_fee) - int(current_fee or 0))
            source_hint = str(
                (position_before or {}).get("source") if isinstance(position_before, dict) else ""
            ) or str(details.get("position_source") or POSITION_SOURCE_MANUAL_ADOPTED)
            recon = {
                "ticker": ticker, "qty": add_qty, "cost_basis_milli": add_cost,
                "gross_buy_milli": add_gross, "buy_fee_milli": add_fee,
                "entry_date": target_broker.get("entry_date"), "source": source_hint,
                "effective_before_revision": economic_revision,
                "inferred_from_sell_revision": economic_revision,
                "required_sell_logical_revision": economic_revision,
                "reason": "historical_trade_edit_requires_external_inventory_provenance",
            }
            reconciliation_events.append(recon)
            broker_seed = {
                "qty": add_qty, "initial_qty": add_qty,
                "initial_cost_basis_milli": add_cost, "remaining_cost_basis_milli": add_cost,
                "realized_pnl_milli": 0, "entry_date": recon.get("entry_date"),
            }
            if add_gross is not None and add_fee is not None:
                broker_seed.update({
                    "initial_gross_buy_milli": add_gross, "remaining_gross_buy_milli": add_gross,
                    "initial_buy_fee_milli": add_fee, "remaining_buy_fee_milli": add_fee,
                })
            if existing is None:
                existing = _record_from_replayed_broker(
                    ticker, source_hint, broker_seed, strategy_templates=strategy_templates
                )
                positions[ticker] = existing
            else:
                b = existing["broker"]
                b["qty"] = int(b.get("qty") or 0) + add_qty
                b["initial_qty"] = int(b.get("initial_qty") or 0) + add_qty
                b["remaining_cost_basis_milli"] = int(b.get("remaining_cost_basis_milli") or 0) + add_cost
                b["initial_cost_basis_milli"] = int(b.get("initial_cost_basis_milli") or 0) + add_cost
                if add_gross is not None and add_fee is not None and all(k in b for k in (
                    "initial_gross_buy_milli", "remaining_gross_buy_milli", "initial_buy_fee_milli", "remaining_buy_fee_milli"
                )):
                    b["initial_gross_buy_milli"] += add_gross; b["remaining_gross_buy_milli"] += add_gross
                    b["initial_buy_fee_milli"] += add_fee; b["remaining_buy_fee_milli"] += add_fee
                positions[ticker] = _record_from_replayed_broker(
                    ticker, str(existing.get("source") or source_hint), b, strategy_templates=strategy_templates
                )
        broker = existing["broker"]
        held_qty = int(broker.get("qty") or 0)
        if qty <= 0 or qty > held_qty:
            raise ValueError(f"{ticker} 修正後賣出股數超過當時庫存：sell={qty}, held={held_qty}")
        entry_date = broker.get("entry_date")
        if entry_date is not None and trade_date <= str(entry_date):
            raise ValueError(f"{ticker} 修正後賣出日必須晚於買入日 {entry_date}")
        exec_price_milli = details.get("exec_price_milli")
        if accounting_params is not None and exec_price_milli is not None and qty > 0:
            current_ledger = build_sell_ledger_from_price(
                milli_to_price(int(exec_price_milli)), qty, accounting_params,
                ticker=ticker, security_profile=details.get("security_profile"), trade_date=trade_date,
            )
            gross_sell = int(current_ledger["gross_sell_milli"])
            sell_fee = int(current_ledger["sell_fee_milli"])
            tax = int(current_ledger["tax_milli"])
            net_sell = int(current_ledger["net_sell_total_milli"])
            exec_price_milli = int(current_ledger["exec_price_milli"])
        else:
            gross_sell = int(details.get("gross_sell_milli") or 0)
            sell_fee = int(details.get("sell_fee_milli") or 0)
            tax = int(details.get("tax_milli") or 0)
            net_sell = int(details.get("net_sell_total_milli") or (gross_sell - sell_fee - tax))
        remaining_cost_before = int(broker.get("remaining_cost_basis_milli") or 0)
        allocated_cost = allocate_cost_basis_milli(remaining_cost_before, held_qty, qty)
        remaining_gross_before = broker.get("remaining_gross_buy_milli")
        remaining_fee_before = broker.get("remaining_buy_fee_milli")
        allocated_gross = None if remaining_gross_before is None else allocate_cost_basis_milli(int(remaining_gross_before), held_qty, qty)
        allocated_buy_fee = (
            None
            if remaining_fee_before is None or allocated_gross is None
            else int(allocated_cost) - int(allocated_gross)
        )
        pnl = net_sell - allocated_cost
        source_before = str(existing.get("source") or POSITION_SOURCE_MANUAL_ADOPTED)
        broker["qty"] = held_qty - qty
        broker["remaining_cost_basis_milli"] = remaining_cost_before - allocated_cost
        broker["realized_pnl_milli"] = int(broker.get("realized_pnl_milli") or 0) + pnl
        if allocated_gross is not None:
            broker["remaining_gross_buy_milli"] = int(remaining_gross_before) - allocated_gross
        if allocated_buy_fee is not None:
            broker["remaining_buy_fee_milli"] = int(remaining_fee_before) - allocated_buy_fee
        if cash_milli is None:
            raise ValueError("Trading cash 尚未設定，無法重建交易帳務")
        cash_milli = int(cash_milli) + net_sell
        remaining_qty = int(broker["qty"])
        projected = deepcopy(details)
        projected.update({
            "ticker": ticker,
            "qty": qty,
            "trade_date": trade_date,
            "exec_price_milli": None if exec_price_milli is None else int(exec_price_milli),
            "gross_sell_milli": gross_sell,
            "sell_fee_milli": sell_fee,
            "tax_milli": tax,
            "net_sell_total_milli": net_sell,
            "allocated_cost_milli": allocated_cost,
            "allocated_gross_buy_milli": allocated_gross,
            "allocated_buy_fee_milli": allocated_buy_fee,
            "realized_pnl_milli": pnl,
            "remaining_qty": max(remaining_qty, 0),
            "cash_milli_after": int(cash_milli),
            "position_source": source_before,
            "strategy_managed": source_before in MANAGED_POSITION_SOURCES,
        })
        projected_trades.append({"revision": economic_revision, "mutation_type": mutation, "details": projected})
        if remaining_qty <= 0:
            positions.pop(ticker, None)
        else:
            positions[ticker] = _record_from_replayed_broker(
                ticker, source_before, broker, strategy_templates=strategy_templates
            )

    # The ledger may have been imported without the broker's complete opening
    # cash history.  Preserve the reconstructed balance rather than blocking a
    # trade correction solely because the historical cash prefix is incomplete.
    return {
        "cash_milli": cash_milli,
        "positions": positions,
        "projected_trades": projected_trades,
        "historical_reconciliations": reconciliation_events,
    }


def project_trading_account_transactions(state: dict[str, Any], *, accounting_params=None) -> list[dict[str, Any]]:
    """Effective trade ledger with cost/charges recomputed after corrections."""
    return deepcopy(_project_trading_account_economics(state, accounting_params=accounting_params)["projected_trades"])


def rebuild_trading_account_economics(state: dict[str, Any], *, accounting_params=None) -> dict[str, Any]:
    """Return the same immutable event ledger with current cash/inventory rebuilt."""
    projection = _project_trading_account_economics(state, accounting_params=accounting_params)
    rebuilt = deepcopy(state)
    rebuilt["cash_milli"] = projection["cash_milli"]
    rebuilt["positions"] = projection["positions"]
    validate_trading_account_state(rebuilt)
    return rebuilt



def _effective_position_source_before_revision(
    state: dict[str, Any],
    ticker: str,
    target_revision: int,
) -> str | None:
    """Infer the open-position source immediately before one historical trade.

    Older account events predate persisted ``position_source`` on sell fills.  The
    immutable event stream is sufficient to infer whether the position being sold
    was a manual-adopted holding or strategy fill without mutating legacy history.
    """
    ticker_key = _normalize_ticker(ticker)
    source: str | None = None
    for event in effective_trading_account_events(state):
        revision = int(event.get("revision") or 0)
        if revision >= int(target_revision):
            break
        mutation = str(event.get("mutation_type") or "")
        details = event.get("details") or {}
        event_ticker = details.get("ticker")
        if event_ticker is None or _normalize_ticker(event_ticker) != ticker_key:
            continue
        if mutation in {"adopt_manual_position", TRADE_MUTATION_BUY}:
            source = POSITION_SOURCE_MANUAL_ADOPTED
        elif mutation == ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT:
            source = POSITION_SOURCE_MANUAL_MANAGED
        elif mutation == TRADE_MUTATION_MANUAL_MANAGED_BUY:
            source = POSITION_SOURCE_MANUAL_MANAGED
        elif mutation in {TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT}:
            source = POSITION_SOURCE_STRATEGY_FILL
        elif mutation == "remove_manual_position":
            source = None
        elif mutation == TRADE_MUTATION_SELL and int(details.get("remaining_qty") or 0) <= 0:
            source = None
    return source


_normalize_ticker = normalize_trading_ticker


_normalize_iso_date = normalize_trading_date


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _build_event(
    *,
    revision: int,
    mutation_id: str,
    mutation_type: str,
    timestamp: str,
    details: dict[str, Any],
    prev_event_hash: str | None,
) -> dict[str, Any]:
    event = {
        "revision": int(revision),
        "mutation_id": str(mutation_id),
        "mutation_type": str(mutation_type),
        "timestamp": str(timestamp),
        "prev_event_hash": prev_event_hash,
        "details": _json_safe(details),
    }
    event["event_hash"] = compute_event_hash(event)
    return event


def build_empty_trading_account_state(
    *,
    timestamp: str,
    mutation_id: str,
    cash: object | None = None,
) -> dict[str, Any]:
    cash_milli = None if cash is None else int(money_to_milli(cash))
    if cash_milli is not None and cash_milli < 0:
        raise ValueError("Trading cash 不可為負數")
    event = _build_event(
        revision=0,
        mutation_id=mutation_id,
        mutation_type="initialize",
        timestamp=timestamp,
        details={"cash_milli": cash_milli},
        prev_event_hash=None,
    )
    state = {
        "schema_version": TRADING_ACCOUNT_SCHEMA_VERSION,
        "runtime_domain": TRADING_RUNTIME_DOMAIN,
        "revision": 0,
        "created_at": str(timestamp),
        "updated_at": str(timestamp),
        "cash_milli": cash_milli,
        "positions": {},
        "events": [event],
    }
    validate_trading_account_state(state)
    return state


def _append_mutation(
    state: dict[str, Any],
    *,
    mutation_id: str,
    mutation_type: str,
    timestamp: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(state)
    previous_revision = int(updated["revision"])
    events = list(updated.get("events", []))
    previous_hash = events[-1]["event_hash"] if events else None
    revision = previous_revision + 1
    events.append(
        _build_event(
            revision=revision,
            mutation_id=mutation_id,
            mutation_type=mutation_type,
            timestamp=timestamp,
            details=details,
            prev_event_hash=previous_hash,
        )
    )
    updated["revision"] = revision
    updated["updated_at"] = str(timestamp)
    updated["events"] = events
    validate_trading_account_state(updated)
    return updated


def set_trading_account_cash(
    state: dict[str, Any],
    *,
    cash: object,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    cash_milli = int(money_to_milli(cash))
    if cash_milli < 0:
        raise ValueError("Trading cash 不可為負數")
    previous_cash_milli = state.get("cash_milli")
    updated = deepcopy(state)
    updated["cash_milli"] = cash_milli
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="set_cash_balance",
        timestamp=timestamp,
        details={
            "previous_cash_milli": previous_cash_milli,
            "cash_milli": cash_milli,
            "note": None if note is None else str(note),
        },
    )


def adopt_manual_trading_position(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    cost_basis_total: object,
    timestamp: str,
    mutation_id: str,
    entry_date: object | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    qty = int(qty)
    if qty <= 0:
        raise ValueError("manual adopted position qty 必須 > 0")
    cost_basis_milli = int(money_to_milli(cost_basis_total))
    if cost_basis_milli <= 0:
        raise ValueError("manual adopted position cost_basis_total 必須 > 0")
    if ticker_key in state["positions"]:
        raise ValueError(f"Trading 已存在 open position: {ticker_key}")

    adopted_entry_date = _normalize_iso_date(entry_date, field_name="entry_date")
    updated = deepcopy(state)
    updated["positions"][ticker_key] = {
        "ticker": ticker_key,
        "source": POSITION_SOURCE_MANUAL_ADOPTED,
        "broker": {
            "qty": qty,
            "initial_qty": qty,
            "initial_cost_basis_milli": cost_basis_milli,
            "remaining_cost_basis_milli": cost_basis_milli,
            "realized_pnl_milli": 0,
            "entry_date": adopted_entry_date,
        },
        "strategy_management": {
            "status": MANAGEMENT_STATUS_UNMANAGED,
            "management_start_date": None,
            "position_state": None,
        },
    }
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="adopt_manual_position",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": qty,
            "cost_basis_total_milli": cost_basis_milli,
            "entry_date": adopted_entry_date,
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def activate_manual_trading_position_management(
    state: dict[str, Any],
    *,
    ticker: object,
    management_lineage: dict[str, Any],
    position_state: dict[str, Any],
    management_start_date: object,
    last_rollforward_date: object | None = None,
    initial_position_state: dict[str, Any] | None = None,
    timestamp: str,
    mutation_id: str,
) -> dict[str, Any]:
    """Normalize one manual holding into canonical frozen-Params management.

    Cash and broker inventory are immutable.  The same mutation is also used to
    normalize an earlier development-era ``manual_managed`` record whose
    management start was detached from its broker entry date; this keeps the
    product contract single-valued without rewriting broker truth.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    current = (state.get("positions") or {}).get(ticker_key)
    if not isinstance(current, dict):
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    current_source = str(current.get("source") or "")
    if current_source not in {POSITION_SOURCE_MANUAL_ADOPTED, POSITION_SOURCE_MANUAL_MANAGED}:
        raise ValueError(f"只有 manual position 可正規化策略管理: {ticker_key}")
    management = dict(current.get("strategy_management") or {})
    if current_source == POSITION_SOURCE_MANUAL_ADOPTED:
        if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
            raise ValueError(f"Trading position 已有 strategy management: {ticker_key}")
    elif management.get("status") != MANAGEMENT_STATUS_ACTIVE:
        raise ValueError(f"Trading manual_managed position management 非 active: {ticker_key}")

    checked_lineage = deepcopy(dict(management_lineage or {}))
    frozen_params = checked_lineage.get("frozen_params")
    frozen_sha = str(checked_lineage.get("frozen_params_sha256") or "")
    if not isinstance(frozen_params, dict) or not frozen_sha or canonical_json_sha256(frozen_params) != frozen_sha:
        raise ValueError(f"Trading management_lineage frozen_params/hash 不合法: {ticker_key}")
    if not str(checked_lineage.get("lineage_id") or "").strip():
        raise ValueError(f"Trading management_lineage 缺少 lineage_id: {ticker_key}")

    broker = deepcopy(dict(current.get("broker") or {}))
    broker_qty = int(broker.get("qty") or 0)
    broker_cost = int(broker.get("remaining_cost_basis_milli") or 0)
    managed_state = _json_safe(deepcopy(dict(position_state or {})))
    if int(managed_state.get("qty") or 0) != broker_qty:
        raise ValueError(f"Trading 手選管理 position/broker qty 不一致: {ticker_key}")
    if int(managed_state.get("remaining_cost_basis_milli") or 0) != broker_cost:
        raise ValueError(f"Trading 手選管理 position/broker cost basis 不一致: {ticker_key}")
    start_date = _normalize_iso_date(management_start_date, field_name="management_start_date")
    if start_date is None:
        raise ValueError("management_start_date 必填")
    rollforward_date = _normalize_iso_date(last_rollforward_date, field_name="last_rollforward_date")
    if rollforward_date is not None and rollforward_date < start_date:
        raise ValueError("last_rollforward_date 不得早於 management_start_date")
    initial_state = None
    if initial_position_state is not None:
        initial_state = _json_safe(deepcopy(dict(initial_position_state)))
        if int(initial_state.get("qty") or 0) != broker_qty:
            raise ValueError(f"Trading 初始管理 position/broker qty 不一致: {ticker_key}")
        if int(initial_state.get("remaining_cost_basis_milli") or 0) != broker_cost:
            raise ValueError(f"Trading 初始管理 position/broker cost basis 不一致: {ticker_key}")

    updated = deepcopy(state)
    updated_record = updated["positions"][ticker_key]
    position_before = deepcopy(updated_record)
    updated_record["source"] = POSITION_SOURCE_MANUAL_MANAGED
    updated_record["management_lineage"] = checked_lineage
    updated_record["strategy_management"] = {
        "status": MANAGEMENT_STATUS_ACTIVE,
        "management_start_date": start_date,
        "last_rollforward_date": rollforward_date,
        "position_state": managed_state,
        "entry_position_state": deepcopy(initial_state),
        "entry_execution_plan": deepcopy(checked_lineage.get("execution_plan_seed") or {}),
    }
    position_after = deepcopy(updated_record)
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type=ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT,
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "management_start_date": start_date,
            "last_rollforward_date": rollforward_date,
            "management_lineage": deepcopy(checked_lineage),
            "initial_position_state": deepcopy(initial_state),
            "position_before": position_before,
            "position_after": position_after,
            "cash_changed": False,
            "broker_truth_changed": False,
        },
    )


def _has_buy_fill_on_date(state: dict[str, Any], ticker: str, trade_date: str) -> bool:
    ticker_key = _normalize_ticker(ticker)
    buy_mutations = {TRADE_MUTATION_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT}
    for event in _effective_trade_events(state):
        if str(event.get("mutation_type") or "") not in buy_mutations:
            continue
        details = event.get("details") or {}
        if _normalize_ticker(details.get("ticker")) == ticker_key and str(details.get("trade_date") or "") == str(trade_date):
            return True
    return False


def _has_sell_fill_on_date(state: dict[str, Any], ticker: str, trade_date: str) -> bool:
    ticker_key = _normalize_ticker(ticker)
    for event in _effective_trade_events(state):
        if str(event.get("mutation_type") or "") != TRADE_MUTATION_SELL:
            continue
        details = event.get("details") or {}
        if _normalize_ticker(details.get("ticker")) == ticker_key and str(details.get("trade_date") or "") == str(trade_date):
            return True
    return False


def _has_confirmed_sell_history(state: dict[str, Any], ticker: str) -> bool:
    ticker_key = _normalize_ticker(ticker)
    for event in _effective_trade_events(state):
        if str(event.get("mutation_type") or "") != TRADE_MUTATION_SELL:
            continue
        details = event.get("details") or {}
        if _normalize_ticker(details.get("ticker")) == ticker_key:
            return True
    return False


def correct_manual_trading_position(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    cost_basis_total: object,
    timestamp: str,
    mutation_id: str,
    entry_date: object | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Correct broker truth for an unmanaged manually adopted holding.

    This is a state-reconciliation operation, not a trade.  It never changes
    cash and is intentionally restricted to holdings with no confirmed sell
    history so that historical accounting is not rewritten.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    record = state["positions"][ticker_key]
    if record.get("source") != POSITION_SOURCE_MANUAL_ADOPTED:
        raise ValueError(f"只有 manual_adopted position 可直接修正 broker truth: {ticker_key}")
    management = record.get("strategy_management") or {}
    if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
        raise ValueError(f"已由策略管理的 position 不可用 manual correction 修改: {ticker_key}")
    if _has_confirmed_sell_history(state, ticker_key):
        raise ValueError(f"已有 confirmed sell history 的 manual position 不可改寫 broker truth: {ticker_key}")

    qty = int(qty)
    if qty <= 0:
        raise ValueError("manual position qty 必須 > 0")
    cost_basis_milli = int(money_to_milli(cost_basis_total))
    if cost_basis_milli <= 0:
        raise ValueError("manual position cost_basis_total 必須 > 0")
    corrected_entry_date = _normalize_iso_date(entry_date, field_name="entry_date")

    updated = deepcopy(state)
    updated_record = updated["positions"][ticker_key]
    previous_broker = deepcopy(updated_record["broker"])
    updated_record["broker"] = {
        "qty": qty,
        "initial_qty": qty,
        "initial_cost_basis_milli": cost_basis_milli,
        "remaining_cost_basis_milli": cost_basis_milli,
        "realized_pnl_milli": 0,
        "entry_date": corrected_entry_date,
    }
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="correct_manual_position",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "previous_broker": previous_broker,
            "broker": deepcopy(updated_record["broker"]),
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def remove_manual_trading_position(
    state: dict[str, Any],
    *,
    ticker: object,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Remove an erroneous unmanaged manual holding without creating a trade."""
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    record = state["positions"][ticker_key]
    if record.get("source") != POSITION_SOURCE_MANUAL_ADOPTED:
        raise ValueError(f"只有 manual_adopted position 可直接移除: {ticker_key}")
    management = record.get("strategy_management") or {}
    if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
        raise ValueError(f"已由策略管理的 position 不可用 manual remove 移除: {ticker_key}")
    if _has_confirmed_sell_history(state, ticker_key):
        raise ValueError(f"已有 confirmed sell history 的 manual position 不可直接移除: {ticker_key}")

    updated = deepcopy(state)
    removed = deepcopy(updated["positions"].pop(ticker_key))
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="remove_manual_position",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "removed_position": removed,
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )



def correct_trading_position_broker_truth(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    cost_basis_total: object,
    timestamp: str,
    mutation_id: str,
    entry_date: object | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Reconcile current broker inventory without fabricating a buy/sell trade.

    This operation is deliberately source-agnostic: both manually adopted and
    strategy-managed positions may be corrected because the broker inventory is
    the account truth.  Strategy stop/target state is preserved while quantity
    and accounting fields are synchronized to the corrected broker truth.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    new_qty = int(qty)
    if new_qty <= 0:
        raise ValueError("position qty 必須 > 0")
    new_cost = int(money_to_milli(cost_basis_total))
    if new_cost <= 0:
        raise ValueError("position cost_basis_total 必須 > 0")
    corrected_entry_date = _normalize_iso_date(entry_date, field_name="entry_date")

    updated = deepcopy(state)
    record = updated["positions"][ticker_key]
    previous = deepcopy(record)
    broker = record["broker"]
    old_qty = max(1, int(broker.get("qty") or 1))
    old_cost = max(1, int(broker.get("remaining_cost_basis_milli") or 1))

    if all(key in broker for key in (
        "initial_gross_buy_milli", "remaining_gross_buy_milli",
        "initial_buy_fee_milli", "remaining_buy_fee_milli",
    )):
        old_gross = int(broker.get("remaining_gross_buy_milli") or 0)
        ratio = old_gross / old_cost if old_cost > 0 else 1.0
        new_gross = max(0, min(new_cost, int(round(new_cost * ratio))))
        new_fee = new_cost - new_gross
        broker["initial_gross_buy_milli"] = new_gross
        broker["remaining_gross_buy_milli"] = new_gross
        broker["initial_buy_fee_milli"] = new_fee
        broker["remaining_buy_fee_milli"] = new_fee
    realized_pnl_milli = int(broker.get("realized_pnl_milli") or 0)
    broker["qty"] = new_qty
    broker["initial_qty"] = new_qty
    broker["initial_cost_basis_milli"] = new_cost
    broker["remaining_cost_basis_milli"] = new_cost
    # Current-inventory reconciliation must not erase PnL already realized by
    # historical sells.  Those fills remain immutable audit evidence.
    broker["realized_pnl_milli"] = realized_pnl_milli
    if corrected_entry_date is not None:
        broker["entry_date"] = corrected_entry_date

    management = record.get("strategy_management") or {}
    position_state = management.get("position_state")
    if record.get("source") in MANAGED_POSITION_SOURCES and isinstance(position_state, dict):
        position_state = deepcopy(position_state)
        gross = int(broker.get("remaining_gross_buy_milli", new_cost) or new_cost)
        fee = int(broker.get("remaining_buy_fee_milli", new_cost - gross) or 0)
        avg_price_milli = max(1, (gross + new_qty // 2) // new_qty)
        old_initial_qty = max(1, int(position_state.get("initial_qty") or old_qty))
        old_risk = int(position_state.get("initial_risk_total_milli") or 0)
        position_state["qty"] = new_qty
        position_state["initial_qty"] = new_qty
        position_state["gross_buy_milli"] = gross
        position_state["buy_fee_milli"] = fee
        position_state["net_buy_total_milli"] = new_cost
        position_state["remaining_cost_basis_milli"] = new_cost
        position_state["entry_fill_price_milli"] = avg_price_milli
        position_state["entry_fill_price"] = milli_to_price(avg_price_milli)
        position_state["pure_buy_price_milli"] = avg_price_milli
        position_state["pure_buy_price"] = milli_to_price(avg_price_milli)
        position_state["initial_risk_total_milli"] = int(round(old_risk * new_qty / old_initial_qty)) if old_risk else 0
        if corrected_entry_date is not None:
            position_state["entry_trade_date"] = corrected_entry_date
            if record.get("source") == POSITION_SOURCE_STRATEGY_FILL:
                management["management_start_date"] = corrected_entry_date
            else:
                prior_management_start = _normalize_iso_date(
                    management.get("management_start_date"), field_name="management_start_date"
                )
                management["management_start_date"] = max(
                    [value for value in (prior_management_start, corrected_entry_date) if value is not None]
                )
        management["position_state"] = _json_safe(sync_position_display_fields(position_state))
        record["strategy_management"] = management

    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="correct_position_broker_truth",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "source": record.get("source"),
            "previous_position": previous,
            "position": deepcopy(record),
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def remove_trading_position_broker_truth(
    state: dict[str, Any],
    *,
    ticker: object,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Remove a current broker inventory row without pretending a SELL occurred."""
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    updated = deepcopy(state)
    removed = deepcopy(updated["positions"].pop(ticker_key))
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="remove_position_broker_truth",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "source": removed.get("source"),
            "removed_position": removed,
            "note": None if note is None else str(note),
            "cash_changed": False,
        },
    )


def apply_strategy_account_buy_correction_fill(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    buy_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    position_template: dict[str, Any],
    increment: bool = False,
) -> dict[str, Any]:
    """Record a corrected strategy BUY while preserving strategy decision state.

    Accounting values are rebuilt from the actual broker fill.  Stop/target and
    roll-forward state come from the pre-correction strategy template, because an
    accounting correction must not silently re-run strategy decisions.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    if _has_sell_fill_on_date(state, ticker_key, trade_date_text):
        raise ValueError(f"{ticker_key} 同一交易日已有賣出成交；Trading 禁止同日買賣")
    add_qty = int(qty)
    if add_qty <= 0:
        raise ValueError("buy qty 必須 > 0")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能修正買入成交")
    ledger = build_buy_ledger_from_price(buy_price, add_qty, params)
    add_cost = int(ledger["net_buy_total_milli"])
    if add_cost > int(cash_milli):
        raise ValueError("Trading cash 不足，不能套用修正後買入成交")

    updated = deepcopy(state)
    existing = updated["positions"].get(ticker_key)
    if increment:
        if existing is None or existing.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            raise ValueError(f"{ticker_key} 缺少可承接策略加碼修正的 position")
        record = existing
    else:
        if existing is not None:
            raise ValueError(f"{ticker_key} 已有 open position，不能重建策略買入")
        record = deepcopy(position_template)
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            raise ValueError("策略買入修正缺少 strategy_fill template")
        updated["positions"][ticker_key] = record

    broker = record["broker"]
    management = record.get("strategy_management") or {}
    position_state = deepcopy(management.get("position_state") or {})
    if not position_state:
        raise ValueError("策略買入修正缺少 position_state")

    if increment:
        old_qty = int(broker.get("qty") or 0)
        old_gross = int(broker.get("remaining_gross_buy_milli") or position_state.get("gross_buy_milli") or 0)
        old_fee = int(broker.get("remaining_buy_fee_milli") or position_state.get("buy_fee_milli") or 0)
        old_cost = int(broker.get("remaining_cost_basis_milli") or position_state.get("remaining_cost_basis_milli") or 0)
        total_qty = old_qty + add_qty
        total_gross = old_gross + int(ledger["gross_buy_milli"])
        total_fee = old_fee + int(ledger["buy_fee_milli"])
        total_cost = old_cost + add_cost
        broker["qty"] = total_qty
        broker["initial_qty"] = int(broker.get("initial_qty") or old_qty) + add_qty
        broker["initial_cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or old_cost) + add_cost
        broker["remaining_cost_basis_milli"] = total_cost
        broker["initial_gross_buy_milli"] = int(broker.get("initial_gross_buy_milli") or old_gross) + int(ledger["gross_buy_milli"])
        broker["remaining_gross_buy_milli"] = total_gross
        broker["initial_buy_fee_milli"] = int(broker.get("initial_buy_fee_milli") or old_fee) + int(ledger["buy_fee_milli"])
        broker["remaining_buy_fee_milli"] = total_fee
    else:
        total_qty = add_qty
        total_gross = int(ledger["gross_buy_milli"])
        total_fee = int(ledger["buy_fee_milli"])
        total_cost = add_cost
        broker["qty"] = total_qty
        broker["initial_qty"] = total_qty
        broker["initial_cost_basis_milli"] = total_cost
        broker["remaining_cost_basis_milli"] = total_cost
        broker["initial_gross_buy_milli"] = total_gross
        broker["remaining_gross_buy_milli"] = total_gross
        broker["initial_buy_fee_milli"] = total_fee
        broker["remaining_buy_fee_milli"] = total_fee
        broker["realized_pnl_milli"] = 0
        broker["entry_date"] = trade_date_text

    avg_price_milli = max(1, (total_gross + total_qty // 2) // total_qty)
    old_initial_qty = max(1, int(position_state.get("initial_qty") or total_qty))
    old_risk = int(position_state.get("initial_risk_total_milli") or 0)
    position_state["qty"] = total_qty
    position_state["initial_qty"] = total_qty
    position_state["gross_buy_milli"] = total_gross
    position_state["buy_fee_milli"] = total_fee
    position_state["net_buy_total_milli"] = total_cost
    position_state["remaining_cost_basis_milli"] = total_cost
    position_state["entry_fill_price_milli"] = avg_price_milli
    position_state["entry_fill_price"] = milli_to_price(avg_price_milli)
    position_state["pure_buy_price_milli"] = avg_price_milli
    position_state["pure_buy_price"] = milli_to_price(avg_price_milli)
    position_state["initial_risk_total_milli"] = int(round(old_risk * total_qty / old_initial_qty)) if old_risk else 0
    if not increment:
        position_state["entry_trade_date"] = trade_date_text
        management["management_start_date"] = trade_date_text
    management["position_state"] = _json_safe(sync_position_display_fields(position_state))
    record["strategy_management"] = management
    updated["cash_milli"] = int(cash_milli) - add_cost

    mutation_type = TRADE_MUTATION_STRATEGY_BUY_INCREMENT if increment else TRADE_MUTATION_STRATEGY_BUY
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type=mutation_type,
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": add_qty if not increment else None,
            "fill_qty": add_qty if increment else None,
            "trade_date": trade_date_text,
            "entry_fill_price_milli": None if increment else avg_price_milli,
            "fill_price_milli": int(ledger["fill_price_milli"]),
            "gross_buy_milli": int(ledger["gross_buy_milli"]) if increment else total_gross,
            "buy_fee_milli": int(ledger["buy_fee_milli"]) if increment else total_fee,
            "net_buy_total_milli": add_cost if increment else total_cost,
            "cumulative_qty": total_qty if increment else None,
            "cumulative_cost_basis_milli": total_cost if increment else None,
            "cash_milli_after": int(updated["cash_milli"]),
            "account_origin": "ACCOUNT_CORRECTION",
        },
    )

def apply_manual_trading_buy_fill(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    buy_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
) -> dict[str, Any]:
    """Record one actual broker BUY entered manually by the user.

    This is account truth, not a scanner/strategy decision.  Existing unmanaged
    manual holdings may be accumulated; strategy-managed holdings must continue
    through their order/fill lineage instead of being silently mixed.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    qty = int(qty)
    if qty <= 0:
        raise ValueError("buy qty 必須 > 0")
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    if _has_sell_fill_on_date(state, ticker_key, trade_date_text):
        raise ValueError(f"{ticker_key} 同一交易日已有賣出成交；Trading 禁止同日買賣")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能登錄買入成交")

    ledger = build_buy_ledger_from_price(buy_price, qty, params)
    net_buy_total_milli = int(ledger["net_buy_total_milli"])
    if net_buy_total_milli > int(cash_milli):
        raise ValueError(
            f"Trading cash 不足：需要 {milli_to_money(net_buy_total_milli):.2f}，"
            f"可用 {milli_to_money(int(cash_milli)):.2f}"
        )

    updated = deepcopy(state)
    existing = updated["positions"].get(ticker_key)
    position_before = None if existing is None else deepcopy(existing)
    if existing is not None:
        if existing.get("source") != POSITION_SOURCE_MANUAL_ADOPTED:
            raise ValueError(f"{ticker_key} 為策略管理持股；買入加碼必須沿用策略 order/fill lineage")
        management = existing.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_UNMANAGED:
            raise ValueError(f"{ticker_key} 已由策略管理，不可用手動帳務買入加碼")
        broker = existing["broker"]
        previous_qty = int(broker["qty"])
        previous_cost = int(broker["remaining_cost_basis_milli"])
        broker["qty"] = previous_qty + qty
        broker["initial_qty"] = int(broker.get("initial_qty") or previous_qty) + qty
        broker["initial_cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or previous_cost) + net_buy_total_milli
        broker["remaining_cost_basis_milli"] = previous_cost + net_buy_total_milli
        if broker.get("entry_date") is None or trade_date_text < str(broker.get("entry_date")):
            broker["entry_date"] = trade_date_text
        # Legacy adopted positions do not have enough evidence to split their
        # historical cost into gross consideration and fee.  Preserve that
        # uncertainty instead of fabricating a broker average price.
        if all(key in broker for key in (
            "initial_gross_buy_milli", "remaining_gross_buy_milli",
            "initial_buy_fee_milli", "remaining_buy_fee_milli",
        )):
            broker["initial_gross_buy_milli"] = int(broker["initial_gross_buy_milli"]) + int(ledger["gross_buy_milli"])
            broker["remaining_gross_buy_milli"] = int(broker["remaining_gross_buy_milli"]) + int(ledger["gross_buy_milli"])
            broker["initial_buy_fee_milli"] = int(broker["initial_buy_fee_milli"]) + int(ledger["buy_fee_milli"])
            broker["remaining_buy_fee_milli"] = int(broker["remaining_buy_fee_milli"]) + int(ledger["buy_fee_milli"])
    else:
        updated["positions"][ticker_key] = {
            "ticker": ticker_key,
            "source": POSITION_SOURCE_MANUAL_ADOPTED,
            "broker": {
                "qty": qty,
                "initial_qty": qty,
                "initial_cost_basis_milli": net_buy_total_milli,
                "remaining_cost_basis_milli": net_buy_total_milli,
                "initial_gross_buy_milli": int(ledger["gross_buy_milli"]),
                "remaining_gross_buy_milli": int(ledger["gross_buy_milli"]),
                "initial_buy_fee_milli": int(ledger["buy_fee_milli"]),
                "remaining_buy_fee_milli": int(ledger["buy_fee_milli"]),
                "realized_pnl_milli": 0,
                "entry_date": trade_date_text,
            },
            "strategy_management": {
                "status": MANAGEMENT_STATUS_UNMANAGED,
                "management_start_date": None,
                "position_state": None,
            },
        }
    updated["cash_milli"] = int(cash_milli) - net_buy_total_milli
    broker_after = updated["positions"][ticker_key]["broker"]
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="manual_buy_fill",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": qty,
            "trade_date": trade_date_text,
            "entry_fill_price_milli": int(ledger["fill_price_milli"]),
            "gross_buy_milli": int(ledger["gross_buy_milli"]),
            "buy_fee_milli": int(ledger["buy_fee_milli"]),
            "net_buy_total_milli": net_buy_total_milli,
            "remaining_qty": int(broker_after["qty"]),
            "remaining_cost_basis_milli": int(broker_after["remaining_cost_basis_milli"]),
            "cash_milli_after": int(updated["cash_milli"]),
            "account_origin": "MANUAL_ACCOUNT_BUY",
            "position_before": position_before,
        },
    )


def _apply_confirmed_managed_buy_fill(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    buy_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    init_sl=None,
    init_trail=None,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    target_reference_price=None,
    security_profile=None,
    entry_type: str = "normal",
    entry_order_id: str | None = None,
    position_source: str,
    lineage_field: str,
    lineage_payload: dict[str, Any] | None,
    mutation_type: str,
    management_start_date: object | None = None,
    execution_plan_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if position_source not in MANAGED_POSITION_SOURCES:
        raise ValueError(f"Trading managed position source 不合法: {position_source}")
    if ticker_key in state["positions"]:
        raise ValueError(f"Trading 已存在 open position，不支援同ticker加碼: {ticker_key}")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能確認買入成交")
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    resolved_management_start = _normalize_iso_date(management_start_date, field_name="management_start_date")
    if resolved_management_start is None:
        resolved_management_start = trade_date_text
    if resolved_management_start < trade_date_text:
        raise ValueError("management_start_date 不得早於實際買入日")
    if _has_sell_fill_on_date(state, ticker_key, trade_date_text):
        raise ValueError(f"{ticker_key} 同一交易日已有賣出成交；Trading 禁止同日買賣")

    entry_seed = deepcopy(dict(execution_plan_seed or (lineage_payload or {}).get("execution_plan_seed") or {}))
    explicit_levels = {
        "init_sl": init_sl, "init_trail": init_trail, "target_price": target_price,
        "limit_price": limit_price, "entry_atr": entry_atr,
        "target_reference_price": target_reference_price, "security_profile": security_profile,
    }
    for key, value in explicit_levels.items():
        if value is not None:
            entry_seed[key] = value
    position_state = build_position_from_frozen_entry_plan(
        entry_seed, buy_price=buy_price, qty=int(qty), params=params,
        entry_type=entry_type, ticker=ticker_key, trade_date=trade_date_text,
    )
    position_state = _json_safe(position_state)
    net_buy_total_milli = int(position_state["net_buy_total_milli"])
    if net_buy_total_milli > int(cash_milli):
        raise ValueError(
            f"Trading cash 不足：需要 {milli_to_money(net_buy_total_milli):.2f}，"
            f"可用 {milli_to_money(int(cash_milli)):.2f}"
        )

    checked_lineage = None if lineage_payload is None else deepcopy(lineage_payload)
    if checked_lineage is None:
        raise ValueError(f"Trading {lineage_field} 必填")
    frozen_params = checked_lineage.get("frozen_params")
    expected_sha = str(checked_lineage.get("frozen_params_sha256") or "")
    if not isinstance(frozen_params, dict) or not expected_sha or canonical_json_sha256(frozen_params) != expected_sha:
        raise ValueError(f"Trading {lineage_field} frozen_params/hash 不合法")
    if not str(checked_lineage.get("lineage_id") or "").strip():
        raise ValueError(f"Trading {lineage_field} 缺少 lineage_id")

    updated = deepcopy(state)
    updated["cash_milli"] = int(cash_milli) - net_buy_total_milli
    record = {
        "ticker": ticker_key,
        "source": position_source,
        lineage_field: checked_lineage,
        "broker": {
            "qty": int(position_state["qty"]),
            "initial_qty": int(position_state["initial_qty"]),
            "initial_cost_basis_milli": net_buy_total_milli,
            "remaining_cost_basis_milli": int(position_state["remaining_cost_basis_milli"]),
            "initial_gross_buy_milli": int(position_state.get("gross_buy_milli", 0) or 0),
            "remaining_gross_buy_milli": int(position_state.get("gross_buy_milli", 0) or 0),
            "initial_buy_fee_milli": int(position_state.get("buy_fee_milli", 0) or 0),
            "remaining_buy_fee_milli": int(position_state.get("buy_fee_milli", 0) or 0),
            "realized_pnl_milli": int(position_state.get("realized_pnl_milli", 0) or 0),
            "entry_date": trade_date_text,
            "entry_order_id": None if entry_order_id is None else str(entry_order_id),
        },
        "strategy_management": {
            "status": MANAGEMENT_STATUS_ACTIVE,
            "management_start_date": resolved_management_start,
            "position_state": position_state,
            "entry_execution_plan": _json_safe(entry_seed),
            "entry_position_state": deepcopy(position_state),
        },
    }
    updated["positions"][ticker_key] = record
    details = {
        "ticker": ticker_key,
        "qty": int(position_state["qty"]),
        "trade_date": trade_date_text,
        "entry_fill_price_milli": int(position_state["entry_fill_price_milli"]),
        "gross_buy_milli": int(position_state.get("gross_buy_milli", 0) or 0),
        "buy_fee_milli": int(position_state.get("buy_fee_milli", 0) or 0),
        "net_buy_total_milli": net_buy_total_milli,
        "cash_milli_after": int(updated["cash_milli"]),
        "entry_order_id": None if entry_order_id is None else str(entry_order_id),
        lineage_field: deepcopy(checked_lineage),
        "position_after": deepcopy(record),
    }
    return _append_mutation(
        updated, mutation_id=mutation_id, mutation_type=mutation_type,
        timestamp=timestamp, details=details,
    )


def apply_confirmed_strategy_buy_fill(
    state: dict[str, Any], *, ticker: object, qty: int, buy_price: object, params,
    timestamp: str, mutation_id: str, trade_date: object, init_sl=None, init_trail=None,
    target_price=None, limit_price=None, entry_atr=None, target_reference_price=None, security_profile=None,
    entry_type: str = "normal", entry_order_id: str | None = None,
    strategy_lineage: dict[str, Any] | None = None,
    execution_plan_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _apply_confirmed_managed_buy_fill(
        state, ticker=ticker, qty=qty, buy_price=buy_price, params=params,
        timestamp=timestamp, mutation_id=mutation_id, trade_date=trade_date,
        init_sl=init_sl, init_trail=init_trail, target_price=target_price,
        limit_price=limit_price, entry_atr=entry_atr, target_reference_price=target_reference_price, security_profile=security_profile,
        entry_type=entry_type, entry_order_id=entry_order_id,
        position_source=POSITION_SOURCE_STRATEGY_FILL, lineage_field="strategy_lineage",
        lineage_payload=strategy_lineage, mutation_type=TRADE_MUTATION_STRATEGY_BUY,
        execution_plan_seed=execution_plan_seed,
    )


def apply_confirmed_manual_managed_buy_fill(
    state: dict[str, Any], *, ticker: object, qty: int, buy_price: object, params,
    timestamp: str, mutation_id: str, trade_date: object, init_sl=None, init_trail=None,
    target_price=None, limit_price=None, entry_atr=None, target_reference_price=None, security_profile=None,
    entry_type: str = "manual", management_lineage: dict[str, Any] | None = None,
    management_start_date: object | None = None,
    execution_plan_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = _apply_confirmed_managed_buy_fill(
        state, ticker=ticker, qty=qty, buy_price=buy_price, params=params,
        timestamp=timestamp, mutation_id=mutation_id, trade_date=trade_date,
        init_sl=init_sl, init_trail=init_trail, target_price=target_price,
        limit_price=limit_price, entry_atr=entry_atr, target_reference_price=target_reference_price, security_profile=security_profile,
        entry_type=entry_type, entry_order_id=None,
        position_source=POSITION_SOURCE_MANUAL_MANAGED, lineage_field="management_lineage",
        lineage_payload=management_lineage, mutation_type=TRADE_MUTATION_MANUAL_MANAGED_BUY,
        management_start_date=management_start_date,
        execution_plan_seed=execution_plan_seed,
    )
    return updated

def apply_confirmed_strategy_buy_fill_increment(
    state: dict[str, Any],
    *,
    entry_order_id: str,
    ticker: object,
    qty: int,
    buy_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    init_sl=None,
    init_trail=None,
    target_price=None,
    limit_price=None,
    entry_atr=None,
    security_profile=None,
    entry_type: str = "normal",
) -> dict[str, Any]:
    """Apply an additional confirmed fill for the same still-active BUY order.

    Partial fills are conservatively limited to one trade date. The broker/account
    cost basis accumulates the exact per-fill ledgers, while the strategy position
    is rebuilt from the gross weighted-average execution price and the original
    frozen entry ATR so stop/trail/target remain on the canonical entry-plan seam.
    """
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 不存在可累積 partial fill 的 open position: {ticker_key}")
    record = state["positions"][ticker_key]
    if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
        raise ValueError(f"Trading partial fill 只能累積到 strategy_fill position: {ticker_key}")
    broker = record.get("broker") or {}
    if str(broker.get("entry_order_id") or "") != str(entry_order_id or ""):
        raise ValueError(f"Trading partial fill order_id 與既有 position 不一致: {ticker_key}")
    if _has_confirmed_sell_history(state, ticker_key):
        raise ValueError(f"已有 confirmed sell history 的 position 不得再累積買入 partial fill: {ticker_key}")

    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    if _has_sell_fill_on_date(state, ticker_key, trade_date_text):
        raise ValueError(f"{ticker_key} 同一交易日已有賣出成交；Trading 禁止同日買賣")
    if str(broker.get("entry_date") or "") != trade_date_text:
        raise ValueError("同一 Trading order 的多次 partial fill 必須發生於同一交易日")
    fill_qty = int(qty)
    if fill_qty <= 0:
        raise ValueError("Trading partial fill qty 必須 > 0")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能確認買入成交")

    management = record.get("strategy_management") or {}
    position_state = management.get("position_state")
    if management.get("status") != MANAGEMENT_STATUS_ACTIVE or not isinstance(position_state, dict):
        raise ValueError(f"Trading strategy position state 不合法: {ticker_key}")

    new_ledger = build_buy_ledger_from_price(buy_price, fill_qty, params)
    new_cost_milli = int(new_ledger["net_buy_total_milli"])
    if new_cost_milli > int(cash_milli):
        raise ValueError(
            f"Trading cash 不足：需要 {milli_to_money(new_cost_milli):.2f}，"
            f"可用 {milli_to_money(int(cash_milli)):.2f}"
        )

    existing_qty = int(position_state["qty"])
    total_qty = existing_qty + fill_qty
    total_gross_milli = int(position_state.get("gross_buy_milli", 0) or 0) + int(new_ledger["gross_buy_milli"])
    total_fee_milli = int(position_state.get("buy_fee_milli", 0) or 0) + int(new_ledger["buy_fee_milli"])
    total_net_milli = int(position_state.get("net_buy_total_milli", 0) or 0) + new_cost_milli
    average_fill_price_milli = (total_gross_milli + total_qty // 2) // total_qty

    acquisition_seed = deepcopy(dict(management.get("entry_execution_plan") or
        (record.get("strategy_lineage") or {}).get("execution_plan_seed") or {}))
    for field, value in (("init_sl", init_sl), ("init_trail", init_trail), ("target_price", target_price),
                         ("limit_price", limit_price), ("entry_atr", entry_atr)):
        if value is not None:
            acquisition_seed.setdefault(field, value)
    rebuilt = build_position_from_frozen_entry_plan(
        acquisition_seed, buy_price=milli_to_price(average_fill_price_milli), qty=total_qty,
        params=params, entry_type=entry_type, ticker=ticker_key,
        security_profile=security_profile, trade_date=trade_date_text,
    )
    rebuilt["gross_buy_milli"] = total_gross_milli
    rebuilt["buy_fee_milli"] = total_fee_milli
    rebuilt["net_buy_total_milli"] = total_net_milli
    rebuilt["remaining_cost_basis_milli"] = total_net_milli
    rebuilt["entry_fill_price_milli"] = int(average_fill_price_milli)
    rebuilt["entry_fill_price"] = milli_to_price(int(average_fill_price_milli))
    rebuilt["pure_buy_price_milli"] = int(average_fill_price_milli)
    rebuilt["pure_buy_price"] = milli_to_price(int(average_fill_price_milli))
    rebuilt["highest_high_since_entry_milli"] = max(
        int(position_state.get("highest_high_since_entry_milli", average_fill_price_milli) or average_fill_price_milli),
        int(average_fill_price_milli),
    )
    rebuilt["highest_high_since_entry"] = milli_to_price(int(rebuilt["highest_high_since_entry_milli"]))
    stop_ledger = build_sell_ledger_from_price(
        rebuilt["initial_stop"],
        total_qty,
        params,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=trade_date_text,
    )
    rebuilt["initial_risk_total_milli"] = calc_initial_risk_total_milli(
        total_net_milli,
        int(stop_ledger["net_sell_total_milli"]),
        rate_to_ppm(params.fixed_risk),
    )
    rebuilt = sync_position_display_fields(rebuilt)

    updated = deepcopy(state)
    updated["cash_milli"] = int(cash_milli) - new_cost_milli
    target = updated["positions"][ticker_key]
    target_broker = target["broker"]
    target_broker["qty"] = total_qty
    target_broker["initial_qty"] = total_qty
    target_broker["initial_cost_basis_milli"] = total_net_milli
    target_broker["remaining_cost_basis_milli"] = total_net_milli
    target_broker["initial_gross_buy_milli"] = total_gross_milli
    target_broker["remaining_gross_buy_milli"] = total_gross_milli
    target_broker["initial_buy_fee_milli"] = total_fee_milli
    target_broker["remaining_buy_fee_milli"] = total_fee_milli
    target["strategy_management"]["position_state"] = _json_safe(rebuilt)
    target["strategy_management"]["entry_execution_plan"] = _json_safe(acquisition_seed)
    target["strategy_management"]["entry_position_state"] = _json_safe(deepcopy(rebuilt))
    target["strategy_management"].pop("evaluation_context_fingerprint", None)
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_strategy_buy_fill_increment",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "entry_order_id": str(entry_order_id),
            "fill_qty": fill_qty,
            "trade_date": trade_date_text,
            "fill_price_milli": int(new_ledger["fill_price_milli"]),
            "gross_buy_milli": int(new_ledger["gross_buy_milli"]),
            "buy_fee_milli": int(new_ledger["buy_fee_milli"]),
            "net_buy_total_milli": new_cost_milli,
            "cumulative_qty": total_qty,
            "cumulative_cost_basis_milli": total_net_milli,
            "cash_milli_after": int(updated["cash_milli"]),
        },
    )


def apply_trading_strategy_management_rollforward(
    state: dict[str, Any],
    *,
    updates: dict[str, dict[str, Any]],
    timestamp: str,
    mutation_id: str,
) -> dict[str, Any]:
    """Persist deterministic completed-bar strategy-management advances.

    ``updates`` is keyed by ticker and must contain a canonical ``position_state``
    plus the last actually processed completed market-data date.  Cash and broker
    accounting are intentionally immutable in this mutation.
    """
    validate_trading_account_state(state)
    if not isinstance(updates, dict) or not updates:
        raise ValueError("Trading rollforward updates 不可為空")

    updated = deepcopy(state)
    details_rows: list[dict[str, Any]] = []
    for ticker_raw in sorted(updates):
        ticker = _normalize_ticker(ticker_raw)
        payload = updates[ticker_raw]
        if not isinstance(payload, dict):
            raise ValueError(f"Trading rollforward payload 不合法: {ticker}")
        processed_bar_count = int(payload.get("processed_bar_count") or 0)
        if processed_bar_count <= 0:
            raise ValueError(f"Trading rollforward processed_bar_count 必須 > 0: {ticker}")
        if ticker not in updated["positions"]:
            raise ValueError(f"Trading rollforward position 不存在: {ticker}")
        record = updated["positions"][ticker]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            raise ValueError(f"Trading rollforward 只允許 managed position: {ticker}")
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise ValueError(f"Trading rollforward position 尚未由策略 active 管理: {ticker}")
        new_position = deepcopy(payload.get("position_state"))
        if not isinstance(new_position, dict):
            raise ValueError(f"Trading rollforward position_state 不合法: {ticker}")
        broker = record.get("broker") or {}
        if int(new_position.get("qty", -1)) != int(broker.get("qty", -2)):
            raise ValueError(f"Trading rollforward 不得改變 broker qty: {ticker}")
        if int(new_position.get("remaining_cost_basis_milli", -1)) != int(broker.get("remaining_cost_basis_milli", -2)):
            raise ValueError(f"Trading rollforward 不得改變 broker cost basis: {ticker}")

        processed_through = _normalize_iso_date(
            payload.get("processed_through_date"), field_name="processed_through_date"
        )
        if processed_through is None:
            raise ValueError(f"Trading rollforward 缺少 processed_through_date: {ticker}")
        previous_date = _normalize_iso_date(
            management.get("last_rollforward_date"), field_name="last_rollforward_date"
        )
        context_fingerprint = str(payload.get("evaluation_context_fingerprint") or "")
        is_context_rebuild = bool(
            payload.get("rebuild") and context_fingerprint
            and context_fingerprint != str(management.get("evaluation_context_fingerprint") or "")
        )
        if previous_date is not None and (
            processed_through < previous_date or (processed_through == previous_date and not is_context_rebuild)
        ):
            raise ValueError(
                f"Trading rollforward date 必須前進: {ticker} {previous_date} -> {processed_through}"
            )

        before_position = management.get("position_state") or {}
        # AI: The management capability cannot acquire accounting write access.
        allowed = frozenset(POSITION_MANAGEMENT_FIELDS)
        old_economics = {key: value for key, value in before_position.items() if key not in allowed}
        new_economics = {key: value for key, value in new_position.items() if key not in allowed}
        if _json_safe(old_economics) != _json_safe(new_economics):
            raise ValueError("Management-only rollforward cannot rewrite acquisition or accounting fields")
        management["position_state"] = _json_safe(new_position)
        management["last_rollforward_date"] = processed_through
        if context_fingerprint:
            management["evaluation_context_fingerprint"] = context_fingerprint
            management["evaluated_through_date"] = str(payload.get("evaluated_through_date") or processed_through)
            management["replay_contract_version"] = payload.get("replay_contract_version")
        previous_obligation = {key: management.get(key) for key in (
            "sell_signal", "sell_signal_date", "sell_signal_trigger_price_milli"
        )}
        if "derived_exit_obligation" in payload:
            if not is_context_rebuild:
                raise ValueError("Derived obligation replacement requires a verified replay context")
            obligation = dict(payload.get("derived_exit_obligation") or {})
            signal = obligation.get("sell_signal")
            if signal is not None and signal not in MANAGEMENT_SELL_SIGNALS:
                raise ValueError("Unknown canonical management exit obligation")
            for key in ("sell_signal", "sell_signal_date", "sell_signal_trigger_price_milli", "execution_after_date"):
                management[key] = obligation.get(key)
        record["strategy_management"] = management
        details_rows.append(
            {
                "ticker": ticker,
                "previous_exit_obligation": previous_obligation,
                "derived_exit_obligation": payload.get("derived_exit_obligation"),
                "evaluation_context_fingerprint": context_fingerprint,
                "previous_rollforward_date": previous_date,
                "processed_through_date": processed_through,
                "processed_bar_count": processed_bar_count,
                "previous_stop_milli": int(before_position.get("sl_milli") or 0),
                "stop_milli": int(new_position.get("sl_milli") or 0),
                "previous_highest_high_milli": int(before_position.get("highest_high_since_entry_milli") or 0),
                "highest_high_since_entry_milli": int(new_position.get("highest_high_since_entry_milli") or 0),
                "cash_changed": False,
                "broker_accounting_changed": False,
            }
        )

    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="rollforward_strategy_management",
        timestamp=timestamp,
        details={"positions": details_rows},
    )


def apply_trading_strategy_management_sell_signals(
    state: dict[str, Any],
    *,
    signals: dict[str, dict[str, Any]],
    timestamp: str,
    mutation_id: str,
) -> dict[str, Any]:
    """Persist one immutable strategy SELL obligation per open managed cycle.

    This records decision truth only.  Broker quantity, cash, fills and position
    economics are never inferred or changed by this mutation.
    """
    validate_trading_account_state(state)
    if not isinstance(signals, dict) or not signals:
        raise ValueError("Trading sell signals 不可為空")

    updated = deepcopy(state)
    details_rows: list[dict[str, Any]] = []
    for ticker_raw in sorted(signals):
        ticker = _normalize_ticker(ticker_raw)
        payload = signals[ticker_raw]
        if not isinstance(payload, dict):
            raise ValueError(f"Trading sell signal payload 不合法: {ticker}")
        if ticker not in updated["positions"]:
            raise ValueError(f"Trading sell signal position 不存在: {ticker}")
        record = updated["positions"][ticker]
        if record.get("source") not in MANAGED_POSITION_SOURCES:
            raise ValueError(f"Trading sell signal 只允許 managed position: {ticker}")
        management = record.get("strategy_management") or {}
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE:
            raise ValueError(f"Trading sell signal position 尚未 active 管理: {ticker}")

        signal = str(payload.get("sell_signal") or "").strip()
        if signal not in MANAGEMENT_SELL_SIGNALS:
            raise ValueError(f"Trading sell_signal 不合法: {ticker} {signal!r}")
        signal_date = _normalize_iso_date(
            payload.get("sell_signal_date"), field_name="sell_signal_date"
        )
        if signal_date is None:
            raise ValueError(f"Trading sell signal 缺少日期: {ticker}")
        entry_date = _normalize_iso_date(
            (record.get("broker") or {}).get("entry_date"), field_name="entry_date"
        )
        if entry_date is not None and signal_date < entry_date:
            raise ValueError(f"Trading sell signal 不得早於成交日: {ticker}")

        existing = str(management.get("sell_signal") or "").strip()
        existing_date = _normalize_iso_date(
            management.get("sell_signal_date"), field_name="sell_signal_date"
        )
        if existing:
            if existing != signal or existing_date != signal_date:
                raise ValueError(
                    f"Trading 已存在不同 sell obligation: {ticker} "
                    f"{existing}@{existing_date} != {signal}@{signal_date}"
                )
            continue

        trigger_milli = payload.get("sell_signal_trigger_price_milli")
        if trigger_milli is not None:
            trigger_milli = int(trigger_milli)
            if trigger_milli <= 0:
                raise ValueError(f"Trading sell trigger price 不合法: {ticker}")
        if signal == MANAGEMENT_SELL_SIGNAL_STOP and trigger_milli is None:
            raise ValueError(f"Trading STOP EXIT 缺少 trigger price: {ticker}")

        management["sell_signal"] = signal
        management["sell_signal_date"] = signal_date
        management["sell_signal_trigger_price_milli"] = trigger_milli
        record["strategy_management"] = management
        lineage = record.get("management_lineage") or record.get("strategy_lineage") or {}
        details_rows.append({
            "ticker": ticker,
            "lineage_id": str(lineage.get("lineage_id") or "") or None,
            "sell_signal": signal,
            "sell_signal_date": signal_date,
            "sell_signal_trigger_price_milli": trigger_milli,
            "broker_accounting_changed": False,
        })

    if not details_rows:
        raise ValueError("Trading sell signals 沒有新的 obligation 可寫入")
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type=ACCOUNT_MUTATION_RECORD_MANAGEMENT_SELL_SIGNAL,
        timestamp=timestamp,
        details={"positions": details_rows},
    )


def apply_confirmed_sell_fill(
    state: dict[str, Any],
    *,
    ticker: object,
    qty: int,
    exec_price: object,
    params,
    timestamp: str,
    mutation_id: str,
    trade_date: object,
    event: str = "MANUAL_CONFIRMED_SELL",
    mark_tp_half_complete: bool = False,
) -> dict[str, Any]:
    validate_trading_account_state(state)
    ticker_key = _normalize_ticker(ticker)
    if ticker_key not in state["positions"]:
        raise ValueError(f"Trading 沒有 open position: {ticker_key}")
    cash_milli = state.get("cash_milli")
    if cash_milli is None:
        raise ValueError("Trading cash 尚未設定，不能確認賣出成交")
    qty = int(qty)
    if qty <= 0:
        raise ValueError("sell qty 必須 > 0")
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    if _has_buy_fill_on_date(state, ticker_key, trade_date_text):
        raise ValueError(f"{ticker_key} 同一交易日已有買入成交；Trading 禁止同日買賣")

    updated = deepcopy(state)
    record = updated["positions"][ticker_key]
    position_before = deepcopy(record)
    broker = record["broker"]
    held_qty = int(broker["qty"])
    if qty > held_qty:
        raise ValueError(f"sell qty 超過持股：sell={qty}, held={held_qty}")
    entry_date = broker.get("entry_date")
    if entry_date is not None:
        trade_date_text = require_trading_date_after(
            trade_date_text,
            after=entry_date,
            field_name="sell trade_date",
            after_field_name="entry_date",
        )

    security_profile = None
    strategy_management = record.get("strategy_management") or {}
    strategy_position = strategy_management.get("position_state")
    if isinstance(strategy_position, dict):
        security_profile = strategy_position.get("security_profile")

    sell_ledger = build_sell_ledger_from_price(
        exec_price,
        qty,
        params,
        ticker=ticker_key,
        security_profile=security_profile,
        trade_date=trade_date_text,
    )
    allocated_cost_milli = allocate_cost_basis_milli(
        int(broker["remaining_cost_basis_milli"]), held_qty, qty
    )
    net_sell_total_milli = int(sell_ledger["net_sell_total_milli"])
    pnl_milli = net_sell_total_milli - int(allocated_cost_milli)

    remaining_gross_before = broker.get("remaining_gross_buy_milli")
    remaining_fee_before = broker.get("remaining_buy_fee_milli")
    allocated_gross_milli = None if remaining_gross_before is None else allocate_cost_basis_milli(int(remaining_gross_before), held_qty, qty)
    allocated_buy_fee_milli = (
        None
        if remaining_fee_before is None or allocated_gross_milli is None
        else int(allocated_cost_milli) - int(allocated_gross_milli)
    )

    broker["qty"] = held_qty - qty
    broker["remaining_cost_basis_milli"] = int(broker["remaining_cost_basis_milli"]) - int(allocated_cost_milli)
    if allocated_gross_milli is not None:
        broker["remaining_gross_buy_milli"] = int(remaining_gross_before) - int(allocated_gross_milli)
    if allocated_buy_fee_milli is not None:
        broker["remaining_buy_fee_milli"] = int(remaining_fee_before) - int(allocated_buy_fee_milli)
    broker["realized_pnl_milli"] = int(broker.get("realized_pnl_milli", 0) or 0) + pnl_milli
    updated["cash_milli"] = int(cash_milli) + net_sell_total_milli

    if isinstance(strategy_position, dict):
        strategy_position = deepcopy(strategy_position)
        freed_cash_milli, strategy_pnl_milli = execute_confirmed_position_sell_fill(
            strategy_position,
            exec_price=exec_price,
            sell_qty=qty,
            params=params,
            trade_date=trade_date_text,
            event=event,
        )
        if int(freed_cash_milli) != net_sell_total_milli or int(strategy_pnl_milli) != pnl_milli:
            raise RuntimeError("Trading broker/strategy sell accounting diverged from canonical position accounting")
        if mark_tp_half_complete and int(strategy_position.get("qty", 0) or 0) > 0:
            acknowledge_position_exit(strategy_position, event="TP_HALF")
        strategy_management["position_state"] = _json_safe(strategy_position)

    remaining_qty = int(broker["qty"])
    if remaining_qty <= 0:
        broker["qty"] = 0
        broker["remaining_cost_basis_milli"] = 0
        if "remaining_gross_buy_milli" in broker:
            broker["remaining_gross_buy_milli"] = 0
        if "remaining_buy_fee_milli" in broker:
            broker["remaining_buy_fee_milli"] = 0
        del updated["positions"][ticker_key]

    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type="confirm_sell_fill",
        timestamp=timestamp,
        details={
            "ticker": ticker_key,
            "qty": qty,
            "trade_date": trade_date_text,
            "exec_price_milli": int(sell_ledger["exec_price_milli"]),
            "gross_sell_milli": int(sell_ledger["gross_sell_milli"]),
            "sell_fee_milli": int(sell_ledger["sell_fee_milli"]),
            "tax_milli": int(sell_ledger["tax_milli"]),
            "net_sell_total_milli": net_sell_total_milli,
            "allocated_cost_milli": int(allocated_cost_milli),
            "allocated_gross_buy_milli": allocated_gross_milli,
            "allocated_buy_fee_milli": allocated_buy_fee_milli,
            "realized_pnl_milli": pnl_milli,
            "remaining_qty": max(remaining_qty, 0),
            "cash_milli_after": int(updated["cash_milli"]),
            "event": str(event),
            "tp_half_complete": bool(mark_tp_half_complete),
            "position_source": position_before.get("source"),
            "strategy_managed": isinstance(strategy_position, dict),
            "position_before": position_before,
        },
    )


def void_manual_trading_transaction(
    state: dict[str, Any],
    *,
    target_revision: int,
    timestamp: str,
    mutation_id: str,
    note: str | None = None,
    params=None,
) -> dict[str, Any]:
    """Void any effective BUY/SELL and rebuild account economics safely.

    The immutable target event stays in the hash chain.  Current cash/inventory
    are projected from all remaining effective events, so later trades are not
    reverse-mutated in-place and legacy full-sell rows do not require a stored
    pre-sell snapshot.
    """
    validate_trading_account_state(state)
    event = _trade_event_by_revision(state, int(target_revision))
    mutation = str(event.get("mutation_type") or "")
    if mutation not in TRADE_MUTATIONS:
        raise ValueError("這筆事件不是可修改/刪除的買賣成交")
    details = dict(event.get("details") or {})
    ticker = _normalize_ticker(details.get("ticker"))
    projection = _project_trading_account_economics(
        state, accounting_params=params, extra_void_revisions={int(target_revision)},
        allow_historical_reconciliation=False,
    )
    updated = deepcopy(state)
    reconciliations: list[dict[str, Any]] = []
    updated["cash_milli"] = projection["cash_milli"]
    updated["positions"] = projection["positions"]
    side = "SELL" if mutation == TRADE_MUTATION_SELL else "BUY"
    return _append_mutation(
        updated,
        mutation_id=mutation_id,
        mutation_type=TRADE_MUTATION_VOID,
        timestamp=timestamp,
        details={
            "target_revision": int(target_revision),
            "target_event_hash": event.get("event_hash"),
            "ticker": ticker,
            "side": side,
            "note": None if note is None else str(note),
            "cash_milli_after": None if updated.get("cash_milli") is None else int(updated["cash_milli"]),
            "projection_rebuilt": True,
            "historical_inventory_reconciliation_count": len(reconciliations),
        },
    )


def replace_trading_transaction(
    state: dict[str, Any],
    *,
    target_revision: int,
    qty: int,
    price: object,
    trade_date: object,
    params,
    timestamp: str,
    mutation_id: str,
) -> dict[str, Any]:
    """Append void + replacement while replaying the replacement at its old slot."""
    validate_trading_account_state(state)
    target = _trade_event_by_revision(state, int(target_revision))
    mutation = str(target.get("mutation_type") or "")
    if mutation not in TRADE_MUTATIONS:
        raise ValueError("這筆事件不是可修改的買賣成交")
    old_details = dict(target.get("details") or {})
    ticker = _normalize_ticker(old_details.get("ticker"))
    qty = int(qty)
    if qty <= 0:
        raise ValueError("成交數量必須 > 0")
    trade_date_text = _normalize_iso_date(trade_date, field_name="trade_date")
    if trade_date_text is None:
        raise ValueError("trade_date 必填")
    logical_revision = int(old_details.get("replacement_for_revision") or target_revision)

    details = {
        "ticker": ticker,
        "trade_date": trade_date_text,
        "replacement_for_revision": logical_revision,
        "replaces_event_revision": int(target_revision),
        "account_origin": "ACCOUNT_CORRECTION",
    }
    if mutation in {TRADE_MUTATION_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT}:
        ledger = build_buy_ledger_from_price(price, qty, params)
        details.update({
            "qty": qty if mutation != TRADE_MUTATION_STRATEGY_BUY_INCREMENT else None,
            "fill_qty": qty if mutation == TRADE_MUTATION_STRATEGY_BUY_INCREMENT else None,
            "entry_fill_price_milli": int(ledger["fill_price_milli"]) if mutation != TRADE_MUTATION_STRATEGY_BUY_INCREMENT else None,
            "fill_price_milli": int(ledger["fill_price_milli"]),
            "gross_buy_milli": int(ledger["gross_buy_milli"]),
            "buy_fee_milli": int(ledger["buy_fee_milli"]),
            "net_buy_total_milli": int(ledger["net_buy_total_milli"]),
        })
    else:
        security_profile = old_details.get("security_profile")
        ledger = build_sell_ledger_from_price(
            price, qty, params, ticker=ticker, security_profile=security_profile, trade_date=trade_date_text
        )
        details.update({
            "qty": qty,
            "exec_price_milli": int(ledger["exec_price_milli"]),
            "gross_sell_milli": int(ledger["gross_sell_milli"]),
            "sell_fee_milli": int(ledger["sell_fee_milli"]),
            "tax_milli": int(ledger["tax_milli"]),
            "net_sell_total_milli": int(ledger["net_sell_total_milli"]),
            "event": str(old_details.get("event") or "ACCOUNT_CORRECTION_SELL"),
        })

    synthetic = {
        "revision": int(target_revision),
        "mutation_type": mutation,
        "details": details,
        "logical_revision": logical_revision,
    }
    projection = _project_trading_account_economics(
        state,
        accounting_params=params,
        extra_void_revisions={int(target_revision)},
        synthetic_trade_events=[synthetic],
        allow_historical_reconciliation=False,
    )
    # Pick the synthetic projected row to persist recalculated cost/PnL metadata.
    projected_details = None
    for row in reversed(projection["projected_trades"]):
        row_details = dict(row.get("details") or {})
        if int(row_details.get("replacement_for_revision") or -1) == logical_revision:
            projected_details = row_details
            break
    if projected_details is None:
        projected_details = details

    updated = deepcopy(state)
    reconciliations: list[dict[str, Any]] = []
    updated["cash_milli"] = projection["cash_milli"]
    updated["positions"] = projection["positions"]
    updated = _append_mutation(
        updated,
        mutation_id=f"{mutation_id}:void",
        mutation_type=TRADE_MUTATION_VOID,
        timestamp=timestamp,
        details={
            "target_revision": int(target_revision),
            "target_event_hash": target.get("event_hash"),
            "ticker": ticker,
            "side": "SELL" if mutation == TRADE_MUTATION_SELL else "BUY",
            "note": "Workbench accounting center edit transaction",
            "cash_milli_after": None if updated.get("cash_milli") is None else int(updated["cash_milli"]),
            "projection_rebuilt": True,
            "historical_inventory_reconciliation_count": len(reconciliations),
        },
    )
    # The replacement is immutable audit evidence.  Economic projection was
    # already computed at its original logical revision, so appending it here
    # must not mutate current broker state a second time.
    return _append_mutation(
        updated,
        mutation_id=f"{mutation_id}:replace",
        mutation_type=mutation,
        timestamp=timestamp,
        details=projected_details,
    )


def validate_trading_account_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("Trading account state 必須是 dict")
    if int(state.get("schema_version", -1)) != TRADING_ACCOUNT_SCHEMA_VERSION:
        raise ValueError("Trading account schema_version 不相容")
    if str(state.get("runtime_domain", "")) != TRADING_RUNTIME_DOMAIN:
        raise ValueError("Trading account runtime_domain 必須是 trading")
    revision = int(state.get("revision", -1))
    if revision < 0:
        raise ValueError("Trading account revision 不可為負數")
    cash_milli = state.get("cash_milli")
    if cash_milli is not None and int(cash_milli) < 0:
        raise ValueError("Trading account cash_milli 不可為負數")

    positions = state.get("positions")
    if not isinstance(positions, dict):
        raise ValueError("Trading account positions 必須是 object")
    for key, record in positions.items():
        ticker = _normalize_ticker(key)
        if ticker != _normalize_ticker(record.get("ticker")):
            raise ValueError(f"Trading position key/ticker 不一致: {key!r}")
        if record.get("source") not in POSITION_SOURCES:
            raise ValueError(f"Trading position source 不合法: {record.get('source')!r}")
        broker = record.get("broker")
        if not isinstance(broker, dict):
            raise ValueError(f"Trading position broker state 缺失: {ticker}")
        qty = int(broker.get("qty", 0) or 0)
        initial_qty = int(broker.get("initial_qty", 0) or 0)
        initial_cost = int(broker.get("initial_cost_basis_milli", 0) or 0)
        remaining_cost = int(broker.get("remaining_cost_basis_milli", 0) or 0)
        if qty <= 0 or initial_qty <= 0 or qty > initial_qty:
            raise ValueError(f"Trading position qty 不合法: {ticker}")
        if initial_cost <= 0 or remaining_cost < 0 or remaining_cost > initial_cost:
            raise ValueError(f"Trading position cost basis 不合法: {ticker}")
        optional_cost_fields = (
            "initial_gross_buy_milli", "remaining_gross_buy_milli",
            "initial_buy_fee_milli", "remaining_buy_fee_milli",
        )
        present_optional_cost_fields = [field for field in optional_cost_fields if field in broker]
        if present_optional_cost_fields and len(present_optional_cost_fields) != len(optional_cost_fields):
            raise ValueError(f"Trading position broker gross/fee accounting 欄位必須成組存在: {ticker}")
        if len(present_optional_cost_fields) == len(optional_cost_fields):
            initial_gross = int(broker["initial_gross_buy_milli"])
            remaining_gross = int(broker["remaining_gross_buy_milli"])
            initial_fee = int(broker["initial_buy_fee_milli"])
            remaining_fee = int(broker["remaining_buy_fee_milli"])
            if min(initial_gross, remaining_gross, initial_fee, remaining_fee) < 0:
                raise ValueError(f"Trading position broker gross/fee accounting 不可為負數: {ticker}")
            if initial_gross + initial_fee != initial_cost:
                raise ValueError(f"Trading position initial gross+fee 必須等於 initial cost: {ticker}")
            if remaining_gross + remaining_fee != remaining_cost:
                raise ValueError(f"Trading position remaining gross+fee 必須等於 remaining cost: {ticker}")
        _normalize_iso_date(broker.get("entry_date"), field_name="entry_date")
        entry_order_id = broker.get("entry_order_id")
        if entry_order_id is not None and not str(entry_order_id).strip():
            raise ValueError(f"Trading position entry_order_id 不合法: {ticker}")
        management = record.get("strategy_management")
        if not isinstance(management, dict) or management.get("status") not in MANAGEMENT_STATUSES:
            raise ValueError(f"Trading strategy management status 不合法: {ticker}")
        if record.get("source") == POSITION_SOURCE_MANUAL_ADOPTED:
            if management.get("status") != MANAGEMENT_STATUS_UNMANAGED or management.get("position_state") is not None:
                raise ValueError(f"manual adopted position 不得偽造 strategy state: {ticker}")
        if record.get("source") in MANAGED_POSITION_SOURCES:
            lineage_field = "strategy_lineage" if record.get("source") == POSITION_SOURCE_STRATEGY_FILL else "management_lineage"
            lineage = record.get(lineage_field)
            if not isinstance(lineage, dict):
                raise ValueError(f"Trading {lineage_field} 必須是 object: {ticker}")
            frozen_params = lineage.get("frozen_params")
            frozen_sha = str(lineage.get("frozen_params_sha256") or "")
            if not isinstance(frozen_params, dict) or not frozen_sha or canonical_json_sha256(frozen_params) != frozen_sha:
                raise ValueError(f"Trading {lineage_field} frozen params/hash 不合法: {ticker}")
            if not str(lineage.get("lineage_id") or "").strip():
                raise ValueError(f"Trading {lineage_field} 缺少 lineage_id: {ticker}")
            position_state = management.get("position_state")
            if management.get("status") != MANAGEMENT_STATUS_ACTIVE or not isinstance(position_state, dict):
                raise ValueError(f"managed fill 必須持有 active position state: {ticker}")
            if int(position_state.get("qty", -1)) != qty:
                raise ValueError(f"strategy/broker qty 不一致: {ticker}")
            if int(position_state.get("remaining_cost_basis_milli", -1)) != remaining_cost:
                raise ValueError(f"strategy/broker cost basis 不一致: {ticker}")
            last_rollforward_date = _normalize_iso_date(
                management.get("last_rollforward_date"), field_name="last_rollforward_date"
            )
            management_start_date = _normalize_iso_date(
                management.get("management_start_date"), field_name="management_start_date"
            )
            if (
                last_rollforward_date is not None
                and management_start_date is not None
                and last_rollforward_date < management_start_date
            ):
                raise ValueError(f"Trading last_rollforward_date 不得早於 management_start_date: {ticker}")
            sell_signal = str(management.get("sell_signal") or "").strip()
            sell_signal_date = _normalize_iso_date(
                management.get("sell_signal_date"), field_name="sell_signal_date"
            )
            trigger_milli = management.get("sell_signal_trigger_price_milli")
            if sell_signal:
                if sell_signal not in MANAGEMENT_SELL_SIGNALS:
                    raise ValueError(f"Trading sell_signal 不合法: {ticker} {sell_signal!r}")
                if sell_signal_date is None:
                    raise ValueError(f"Trading sell_signal 缺少 sell_signal_date: {ticker}")
                if sell_signal_date < str(broker.get("entry_date") or ""):
                    raise ValueError(f"Trading sell_signal_date 不得早於 entry_date: {ticker}")
                if sell_signal == MANAGEMENT_SELL_SIGNAL_STOP:
                    if trigger_milli is None or int(trigger_milli) <= 0:
                        raise ValueError(f"Trading STOP EXIT 缺少合法 trigger price: {ticker}")
                elif trigger_milli is not None and int(trigger_milli) <= 0:
                    raise ValueError(f"Trading sell trigger price 不合法: {ticker}")
            elif sell_signal_date is not None or trigger_milli is not None:
                raise ValueError(f"Trading sell obligation metadata 不完整: {ticker}")

    events = state.get("events")
    if not isinstance(events, list) or len(events) != revision + 1:
        raise ValueError("Trading account event count 必須與 revision 連續一致")
    previous_hash = None
    for expected_revision, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("Trading account event 必須是 object")
        if int(event.get("revision", -1)) != expected_revision:
            raise ValueError("Trading account event revision 不連續")
        if event.get("prev_event_hash") != previous_hash:
            raise ValueError("Trading account event hash chain 斷裂")
        actual_hash = compute_event_hash(event)
        if event.get("event_hash") != actual_hash:
            raise ValueError("Trading account event hash 不一致")
        previous_hash = actual_hash


def build_trading_account_read_model(state: dict[str, Any]) -> dict[str, Any]:
    validate_trading_account_state(state)
    effective_trade_events = [
        event for event in effective_trading_account_events(state)
        if str(event.get("mutation_type") or "") in TRADE_MUTATIONS
    ]
    buy_trade_dates = sorted({
        str((event.get("details") or {}).get("trade_date") or "")
        for event in effective_trade_events
        if str(event.get("mutation_type") or "") in {
            TRADE_MUTATION_BUY, TRADE_MUTATION_MANUAL_MANAGED_BUY, TRADE_MUTATION_STRATEGY_BUY, TRADE_MUTATION_STRATEGY_BUY_INCREMENT
        }
        and str((event.get("details") or {}).get("trade_date") or "")
    })
    sell_trade_dates = sorted({
        str((event.get("details") or {}).get("trade_date") or "")
        for event in effective_trade_events
        if str(event.get("mutation_type") or "") == TRADE_MUTATION_SELL
        and str((event.get("details") or {}).get("trade_date") or "")
    })
    positions = []
    for ticker in sorted(state["positions"]):
        record = state["positions"][ticker]
        broker = record["broker"]
        qty = int(broker["qty"])
        remaining_cost_milli = int(broker["remaining_cost_basis_milli"])
        remaining_gross_milli = broker.get("remaining_gross_buy_milli")
        average_price_basis_milli = remaining_cost_milli if remaining_gross_milli is None else int(remaining_gross_milli)
        positions.append(
            {
                "ticker": ticker,
                "source": record["source"],
                "qty": qty,
                "average_cost": calc_average_price_from_total_milli(average_price_basis_milli, qty),
                "holding_cost": milli_to_money(remaining_cost_milli),
                "remaining_cost_basis": milli_to_money(remaining_cost_milli),
                "realized_pnl": milli_to_money(int(broker.get("realized_pnl_milli", 0) or 0)),
                "entry_date": broker.get("entry_date"),
                "entry_order_id": broker.get("entry_order_id"),
                "strategy_lineage_id": (record.get("strategy_lineage") or {}).get("lineage_id"),
                "management_lineage_id": (record.get("management_lineage") or record.get("strategy_lineage") or {}).get("lineage_id"),
                "strategy_lineage_key": (
                    (
                        "POSITION:" + str((record.get("management_lineage") or record.get("strategy_lineage") or {}).get("lineage_id") or "")
                        if str((record.get("management_lineage") or record.get("strategy_lineage") or {}).get("lineage_id") or "") else None
                    )
                    or str(broker.get("entry_order_id") or "")
                    or None
                ),
                "management_status": record["strategy_management"]["status"],
                "last_rollforward_date": record["strategy_management"].get("last_rollforward_date"),
                "sell_signal": record["strategy_management"].get("sell_signal"),
                "sell_signal_date": record["strategy_management"].get("sell_signal_date"),
                "sell_signal_trigger_price": (
                    None
                    if record["strategy_management"].get("sell_signal_trigger_price_milli") is None
                    else milli_to_price(int(record["strategy_management"]["sell_signal_trigger_price_milli"]))
                ),
                "effective_stop": (
                    None
                    if not isinstance(record["strategy_management"].get("position_state"), dict)
                    else milli_to_price(int(record["strategy_management"]["position_state"].get("sl_milli") or 0))
                ),
                "has_sell_history": _has_confirmed_sell_history(state, ticker),
            }
        )
    return {
        "schema_version": int(state["schema_version"]),
        "revision": int(state["revision"]),
        "cash": None if state.get("cash_milli") is None else milli_to_money(int(state["cash_milli"])),
        "position_count": len(positions),
        "positions": positions,
        "latest_buy_trade_date": buy_trade_dates[-1] if buy_trade_dates else None,
        "latest_sell_trade_date": sell_trade_dates[-1] if sell_trade_dates else None,
        "buy_trade_dates": buy_trade_dates,
        "sell_trade_dates": sell_trade_dates,
        "updated_at": state.get("updated_at"),
    }


__all__ = [
    "TRADING_ACCOUNT_SCHEMA_VERSION",
    "TRADING_ACCOUNT_STATE_FILENAME",
    "POSITION_SOURCE_MANUAL_ADOPTED",
    "POSITION_SOURCE_MANUAL_MANAGED",
    "POSITION_SOURCE_STRATEGY_FILL",
    "MANAGED_POSITION_SOURCES",
    "MANAGEMENT_STATUS_UNMANAGED",
    "MANAGEMENT_STATUS_ACTIVE",
    "MANAGEMENT_SELL_SIGNAL_STOP",
    "MANAGEMENT_SELL_SIGNAL_INDICATOR",
    "MANAGEMENT_SELL_SIGNALS",
    "ACCOUNT_MUTATION_RECORD_MANAGEMENT_SELL_SIGNAL",
    "ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT",
    "build_empty_trading_account_state",
    "set_trading_account_cash",
    "adopt_manual_trading_position",
    "activate_manual_trading_position_management",
    "correct_manual_trading_position",
    "remove_manual_trading_position",
    "apply_manual_trading_buy_fill",
    "apply_confirmed_manual_managed_buy_fill",
    "apply_confirmed_strategy_buy_fill",
    "apply_confirmed_strategy_buy_fill_increment",
    "apply_trading_strategy_management_rollforward",
    "apply_trading_strategy_management_sell_signals",
    "apply_confirmed_sell_fill",
    "void_manual_trading_transaction",
    "effective_trading_account_events",
    "validate_trading_account_state",
    "build_trading_account_read_model",
]
