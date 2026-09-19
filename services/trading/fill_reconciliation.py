"""Recoverable reconciliation of externally confirmed Trading BUY/SELL fills."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.accounting_policy import overlay_trading_accounting_params
from services.trading.entry_lifecycle_context import resolve_trading_confirmed_entry_seed
from services.trading.strategy_param_runtime import build_trading_order_strategy_lineage, resolve_trading_position_management_binding
from core.exact_accounting import milli_to_price, price_to_milli
from core.params_io import build_params_from_mapping
from core.runtime_utils import get_taipei_now
from core.trading_account_state import (
    MANAGED_POSITION_SOURCES,
    apply_confirmed_strategy_buy_fill,
    apply_confirmed_strategy_buy_fill_increment,
    apply_confirmed_sell_fill,
    validate_trading_account_state,
)
from core.trading_fill_transaction import (
    build_trading_fill_transaction_journal,
    validate_trading_fill_transaction_journal,
)
from core.trading_identity import normalize_trading_date, require_trading_date_after
from core.trading_order_state import (
    TRADING_ACTIVE_ORDER_STATUSES,
    record_trading_buy_order_fill,
    record_trading_sell_order_fill,
    TRADING_ORDER_SIDE_SELL,
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP,
    TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER,
    TRADING_ORDER_PURPOSE_PROTECTION_TP,
    validate_trading_order_state,
)
from core.trading_tp_progress import build_trading_tp_half_progress
from core.trading_state_paths import (
    resolve_trading_account_state_path,
    resolve_trading_fill_transaction_path as _resolve_trading_fill_transaction_path,
    resolve_trading_order_state_path,
)


class TradingFillRevisionConflict(RuntimeError):
    pass


def _timestamp() -> str:
    return get_taipei_now().isoformat(timespec="seconds")


def _mutation_id() -> str:
    return uuid4().hex


def _normalize_external_fill_price(fill_price) -> float:
    """Normalize broker-entered prices onto the canonical milli-price seam.

    Workbench parsers deliberately return ``Decimal`` for exact user input, while
    strategy geometry uses canonical float display prices backed by integer milli
    values.  Converting once here prevents Decimal/float arithmetic from leaking
    into position construction without changing the exact persisted price.
    """
    fill_price_milli = price_to_milli(fill_price)
    if fill_price_milli <= 0:
        raise ValueError("Trading fill_price 必須 > 0")
    return milli_to_price(fill_price_milli)


def resolve_trading_fill_transaction_path(project_root) -> Path:
    return _resolve_trading_fill_transaction_path(project_root)


def _read_bytes_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_account(path: Path) -> dict[str, Any]:
    state = load_json_strict(path)
    validate_trading_account_state(state)
    return state


def _read_orders(path: Path) -> dict[str, Any]:
    state = load_json_strict(path)
    validate_trading_order_state(state)
    return state


@serialized_trading_state_mutation
def recover_trading_fill_transaction(project_root) -> bool:
    """Finish a previously prepared two-state fill commit after interruption.

    The journal only permits original->target completion. Any unrelated state drift
    is treated as a conflict rather than being overwritten.
    """
    root = Path(project_root).resolve()
    tx_path = resolve_trading_fill_transaction_path(root)
    if not tx_path.is_file():
        return False

    journal = load_json_strict(tx_path)
    validate_trading_fill_transaction_journal(journal)
    account_path = resolve_trading_account_state_path(root)
    order_path = resolve_trading_order_state_path(root)
    if not account_path.is_file() or not order_path.is_file():
        raise TradingFillRevisionConflict("Trading fill recovery 缺少 account/order state")

    account = _read_account(account_path)
    orders = _read_orders(order_path)
    account_sha = canonical_json_sha256(account)
    order_sha = canonical_json_sha256(orders)
    account_original = str(journal["account_original_sha256"])
    account_target_sha = str(journal["account_target_sha256"])
    order_original = str(journal["order_original_sha256"])
    order_target_sha = str(journal["order_target_sha256"])

    if account_sha not in {account_original, account_target_sha}:
        raise TradingFillRevisionConflict("Trading account 與 pending fill transaction 無法安全 recovery")
    if order_sha not in {order_original, order_target_sha}:
        raise TradingFillRevisionConflict("Trading order state 與 pending fill transaction 無法安全 recovery")

    if account_sha == account_original:
        atomic_write_json(account_path, journal["account_target"])
    if order_sha == order_original:
        atomic_write_json(order_path, journal["order_target"])

    final_account = _read_account(account_path)
    final_orders = _read_orders(order_path)
    if canonical_json_sha256(final_account) != account_target_sha:
        raise TradingFillRevisionConflict("Trading account fill recovery target hash 不一致")
    if canonical_json_sha256(final_orders) != order_target_sha:
        raise TradingFillRevisionConflict("Trading order fill recovery target hash 不一致")
    tx_path.unlink()
    return True


@serialized_trading_state_mutation
def confirm_trading_buy_order_fill(
    project_root,
    *,
    order_id: str,
    fill_qty: int,
    fill_price,
    trade_date,
    expected_order_revision: int,
    expected_account_revision: int,
) -> dict[str, Any]:
    """Apply one broker-confirmed BUY fill to account and order state as one recoverable transaction."""
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account_path = resolve_trading_account_state_path(root)
    order_path = resolve_trading_order_state_path(root)
    if not account_path.is_file():
        raise FileNotFoundError(f"Trading account state 尚未初始化: {account_path}")
    if not order_path.is_file():
        raise FileNotFoundError(f"Trading order state 尚未建立: {order_path}")

    account_bytes_sha = _read_bytes_sha(account_path)
    order_bytes_sha = _read_bytes_sha(order_path)
    account = _read_account(account_path)
    orders = _read_orders(order_path)
    if int(account["revision"]) != int(expected_account_revision):
        raise TradingFillRevisionConflict(
            f"Trading account revision 已變更：expected={expected_account_revision}, current={account['revision']}"
        )
    if int(orders["revision"]) != int(expected_order_revision):
        raise TradingFillRevisionConflict(
            f"Trading order revision 已變更：expected={expected_order_revision}, current={orders['revision']}"
        )

    order_id_text = str(order_id or "").strip()
    record = (orders.get("orders") or {}).get(order_id_text)
    if not isinstance(record, dict):
        raise ValueError(f"Trading order 不存在: {order_id_text}")
    if str(record.get("status") or "") not in TRADING_ACTIVE_ORDER_STATUSES:
        raise ValueError(f"Trading order 目前不可確認成交: {record.get('status')}")
    buy_trade_date = require_trading_date_after(
        trade_date,
        after=record.get("information_date"),
        field_name="BUY fill trade_date",
        after_field_name="information_date",
    )
    frozen_params = record.get("frozen_params")
    if not isinstance(frozen_params, dict):
        raise RuntimeError("Trading ORDERED 缺少 frozen_params；禁止用目前新 params 回填舊掛單")
    params = overlay_trading_accounting_params(build_params_from_mapping(frozen_params))
    normalized_fill_price = _normalize_external_fill_price(fill_price)

    ticker = str(record["ticker"])
    fill_qty_int = int(fill_qty)
    timestamp = _timestamp()
    account_mutation_id = _mutation_id()
    fill_id = _mutation_id()
    order_mutation_id = _mutation_id()
    init_sl = milli_to_price(int(record["init_sl_milli"]))
    init_trail = milli_to_price(int(record["init_trail_milli"]))
    target_price = milli_to_price(int(record["target_price_milli"]))
    limit_price = milli_to_price(int(record["limit_price_milli"]))
    entry_atr = record.get("entry_atr_milli")
    entry_atr = None if entry_atr is None else milli_to_price(int(entry_atr))

    existing = (account.get("positions") or {}).get(ticker)
    if existing is None:
        strategy_lineage = build_trading_order_strategy_lineage(record)
        acquisition_seed = resolve_trading_confirmed_entry_seed(
            root, ticker=ticker, lineage=strategy_lineage, params=params,
            fill_date=buy_trade_date, information_date=record.get("information_date"),
        )
        account_target = apply_confirmed_strategy_buy_fill(
            account,
            ticker=ticker,
            qty=fill_qty_int,
            buy_price=normalized_fill_price,
            params=params,
            timestamp=timestamp,
            mutation_id=account_mutation_id,
            trade_date=buy_trade_date,
            init_sl=init_sl,
            init_trail=init_trail,
            target_price=target_price,
            limit_price=limit_price,
            entry_atr=entry_atr,
            security_profile=record.get("security_profile"),
            entry_type=str(record.get("entry_type") or "normal"),
            entry_order_id=order_id_text,
            strategy_lineage=strategy_lineage,
            execution_plan_seed=acquisition_seed,
        )
    else:
        account_target = apply_confirmed_strategy_buy_fill_increment(
            account,
            entry_order_id=order_id_text,
            ticker=ticker,
            qty=fill_qty_int,
            buy_price=normalized_fill_price,
            params=params,
            timestamp=timestamp,
            mutation_id=account_mutation_id,
            trade_date=buy_trade_date,
            init_sl=init_sl,
            init_trail=init_trail,
            target_price=target_price,
            limit_price=limit_price,
            entry_atr=entry_atr,
            security_profile=record.get("security_profile"),
            entry_type=str(record.get("entry_type") or "normal"),
        )
    account_event = account_target["events"][-1]
    net_buy_total_milli = int((account_event.get("details") or {}).get("net_buy_total_milli") or 0)
    if net_buy_total_milli <= 0:
        raise RuntimeError("Trading account fill mutation 缺少 canonical net_buy_total_milli")

    order_target = record_trading_buy_order_fill(
        orders,
        order_id=order_id_text,
        fill_id=fill_id,
        fill_qty=fill_qty_int,
        fill_price=normalized_fill_price,
        trade_date=buy_trade_date,
        net_buy_total_milli=net_buy_total_milli,
        timestamp=timestamp,
        mutation_id=order_mutation_id,
    )
    if int(account_target["revision"]) != int(account["revision"]) + 1:
        raise RuntimeError("Trading fill account mutation 必須恰好增加一個 revision")
    if int(order_target["revision"]) != int(orders["revision"]) + 1:
        raise RuntimeError("Trading fill order mutation 必須恰好增加一個 revision")

    if _read_bytes_sha(account_path) != account_bytes_sha:
        raise TradingFillRevisionConflict("Trading account 在 fill transaction prepare 前已被修改")
    if _read_bytes_sha(order_path) != order_bytes_sha:
        raise TradingFillRevisionConflict("Trading order state 在 fill transaction prepare 前已被修改")

    tx_path = resolve_trading_fill_transaction_path(root)
    if tx_path.exists():
        raise TradingFillRevisionConflict("Trading 已存在 pending fill transaction")
    journal = build_trading_fill_transaction_journal(
        transaction_id=_mutation_id(),
        created_at=timestamp,
        account_original=account,
        account_target=account_target,
        order_original=orders,
        order_target=order_target,
    )
    atomic_write_json(tx_path, journal)

    # Once PREPARED exists, recovery owns completion. Do not roll back one side.
    if _read_bytes_sha(account_path) != account_bytes_sha or _read_bytes_sha(order_path) != order_bytes_sha:
        raise TradingFillRevisionConflict("Trading state 在 fill transaction commit 前已被其他流程修改")
    atomic_write_json(account_path, account_target)
    atomic_write_json(order_path, order_target)
    recover_trading_fill_transaction(root)

    final_account = _read_account(account_path)
    final_orders = _read_orders(order_path)
    final_record = final_orders["orders"][order_id_text]
    return {
        "status": str(final_record["status"]),
        "order_id": order_id_text,
        "ticker": ticker,
        "fill_qty": fill_qty_int,
        "filled_qty": int(final_record.get("filled_qty") or 0),
        "remaining_qty": int(final_record.get("remaining_qty") or 0),
        "account_revision": int(final_account["revision"]),
        "order_revision": int(final_orders["revision"]),
        "account": final_account,
        "orders": final_orders,
    }


@serialized_trading_state_mutation
def _confirm_trading_sell_order_fill(
    project_root,
    *,
    order_id: str,
    fill_qty: int,
    fill_price,
    trade_date,
    expected_order_revision: int,
    expected_account_revision: int,
    allowed_purposes: frozenset[str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account_path = resolve_trading_account_state_path(root)
    order_path = resolve_trading_order_state_path(root)
    if not account_path.is_file() or not order_path.is_file():
        raise FileNotFoundError("Trading SELL fill reconciliation 缺少 account/order state")
    account_bytes_sha = _read_bytes_sha(account_path)
    order_bytes_sha = _read_bytes_sha(order_path)
    account = _read_account(account_path)
    orders = _read_orders(order_path)
    if int(account["revision"]) != int(expected_account_revision):
        raise TradingFillRevisionConflict(f"Trading account revision 已變更：expected={expected_account_revision}, current={account['revision']}")
    if int(orders["revision"]) != int(expected_order_revision):
        raise TradingFillRevisionConflict(f"Trading order revision 已變更：expected={expected_order_revision}, current={orders['revision']}")
    oid = str(order_id or "").strip()
    record = (orders.get("orders") or {}).get(oid)
    if not isinstance(record, dict):
        raise ValueError(f"Trading order 不存在: {oid}")
    purpose = str(record.get("purpose") or "")
    if str(record.get("side") or "") != TRADING_ORDER_SIDE_SELL or purpose not in allowed_purposes:
        raise ValueError("此入口不允許該 SELL order purpose")
    if str(record.get("status") or "") not in TRADING_ACTIVE_ORDER_STATUSES:
        raise ValueError(f"Trading SELL 目前不可確認成交: {record.get('status')}")
    ticker = str(record.get("ticker") or "")
    position = (account.get("positions") or {}).get(ticker)
    if not isinstance(position, dict) or str(position.get("source") or "") not in MANAGED_POSITION_SOURCES:
        raise RuntimeError(f"Trading SELL 找不到 managed position: {ticker}")
    entry_order_id = str(record.get("entry_order_id") or "")
    binding = resolve_trading_position_management_binding(position, orders=orders)
    allowed_lineage_keys = {str(binding.get("lineage_key") or "")}
    legacy_entry_order_id = str(binding.get("entry_order_id") or "").strip()
    if legacy_entry_order_id:
        allowed_lineage_keys.add(legacy_entry_order_id)
    if entry_order_id not in allowed_lineage_keys:
        raise RuntimeError("Trading SELL strategy lineage 與 account position 不一致")
    sell_trade_date = require_trading_date_after(
        trade_date,
        after=(position.get("broker") or {}).get("entry_date"),
        field_name="SELL fill trade_date",
        after_field_name="entry_date",
    )
    params = overlay_trading_accounting_params(build_params_from_mapping(binding["frozen_params"]))
    normalized_fill_price = _normalize_external_fill_price(fill_price)
    fill_qty_int = int(fill_qty)
    tp_half_complete = False
    if purpose == TRADING_ORDER_PURPOSE_PROTECTION_TP:
        broker = position.get("broker") or {}
        tp_progress = build_trading_tp_half_progress(
            orders,
            ticker=ticker,
            entry_order_id=str(binding.get("lineage_key") or entry_order_id),
            compatible_entry_order_ids=([legacy_entry_order_id] if legacy_entry_order_id else None),
            initial_qty=int(broker.get("initial_qty") or 0),
            tp_percent=params.tp_percent,
        )
        tp_half_complete = (
            int(tp_progress["target_qty"]) > 0
            and int(tp_progress["confirmed_qty"]) + fill_qty_int >= int(tp_progress["target_qty"])
        )
    if purpose == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
        event = "IND_SELL"
    elif purpose in {TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER}:
        event = "STOP"
    else:
        event = "TP_HALF"
    broker_before = position.get("broker") or {}
    strategy_position_before_fill = deepcopy((position.get("strategy_management") or {}).get("position_state"))
    position_qty_before_fill = int(broker_before.get("qty") or 0)
    timestamp = _timestamp()
    account_target = apply_confirmed_sell_fill(
        account, ticker=ticker, qty=fill_qty_int, exec_price=normalized_fill_price, params=params,
        timestamp=timestamp, mutation_id=_mutation_id(), trade_date=sell_trade_date, event=event,
        mark_tp_half_complete=tp_half_complete,
    )
    details = account_target["events"][-1].get("details") or {}
    position_after = ((account_target.get("positions") or {}).get(ticker) or {}).get("broker") or {}
    position_qty_after_fill = int(position_after.get("qty") or 0)
    stop_snapshot = strategy_position_before_fill if event == "STOP" else None
    order_target = record_trading_sell_order_fill(
        orders, order_id=oid, fill_id=_mutation_id(), fill_qty=fill_qty_int, fill_price=normalized_fill_price,
        trade_date=sell_trade_date, net_sell_total_milli=int(details.get("net_sell_total_milli") or 0),
        allocated_cost_milli=int(details.get("allocated_cost_milli") or 0), realized_pnl_milli=int(details.get("realized_pnl_milli") or 0),
        timestamp=timestamp, mutation_id=_mutation_id(),
        strategy_position_before_fill=stop_snapshot,
        position_qty_before_fill=position_qty_before_fill if stop_snapshot is not None else None,
        position_qty_after_fill=position_qty_after_fill if stop_snapshot is not None else None,
    )
    if int(account_target["revision"]) != int(account["revision"]) + 1 or int(order_target["revision"]) != int(orders["revision"]) + 1:
        raise RuntimeError("Trading SELL fill transaction 每個 state 必須恰好增加一個 revision")
    if _read_bytes_sha(account_path) != account_bytes_sha or _read_bytes_sha(order_path) != order_bytes_sha:
        raise TradingFillRevisionConflict("Trading state 在 SELL fill transaction prepare 前已被修改")
    tx_path = resolve_trading_fill_transaction_path(root)
    if tx_path.exists(): raise TradingFillRevisionConflict("Trading 已存在 pending fill transaction")
    journal = build_trading_fill_transaction_journal(
        transaction_id=_mutation_id(), created_at=timestamp, account_original=account, account_target=account_target,
        order_original=orders, order_target=order_target,
    )
    atomic_write_json(tx_path, journal)
    if _read_bytes_sha(account_path) != account_bytes_sha or _read_bytes_sha(order_path) != order_bytes_sha:
        raise TradingFillRevisionConflict("Trading state 在 SELL fill transaction commit 前已被其他流程修改")
    atomic_write_json(account_path, account_target); atomic_write_json(order_path, order_target); recover_trading_fill_transaction(root)
    final_account = _read_account(account_path); final_orders = _read_orders(order_path); final_record = final_orders["orders"][oid]
    return {
        "status": str(final_record["status"]), "order_id": oid, "ticker": ticker, "fill_qty": fill_qty_int,
        "filled_qty": int(final_record.get("filled_qty") or 0), "remaining_qty": int(final_record.get("remaining_qty") or 0),
        "account_revision": int(final_account["revision"]), "order_revision": int(final_orders["revision"]),
        "oco_peer_reconciliation_order_ids": list((final_orders["events"][-1].get("details") or {}).get("oco_peer_reconciliation_order_ids") or []),
        "account": final_account, "orders": final_orders,
    }


def confirm_trading_protection_sell_order_fill(project_root, **kwargs) -> dict[str, Any]:
    return _confirm_trading_sell_order_fill(
        project_root, allowed_purposes=frozenset({TRADING_ORDER_PURPOSE_PROTECTION_STOP, TRADING_ORDER_PURPOSE_PROTECTION_STOP_REMAINDER, TRADING_ORDER_PURPOSE_PROTECTION_TP}), **kwargs
    )


def confirm_trading_indicator_sell_order_fill(project_root, **kwargs) -> dict[str, Any]:
    return _confirm_trading_sell_order_fill(
        project_root, allowed_purposes=frozenset({TRADING_ORDER_PURPOSE_INDICATOR_EXIT}), **kwargs
    )


__all__ = [
    "TradingFillRevisionConflict",
    "resolve_trading_fill_transaction_path",
    "recover_trading_fill_transaction",
    "confirm_trading_buy_order_fill",
    "confirm_trading_protection_sell_order_fill",
    "confirm_trading_indicator_sell_order_fill",
]
