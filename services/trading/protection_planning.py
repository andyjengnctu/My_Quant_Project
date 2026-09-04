"""Derived Trading protection-order plans for confirmed strategy positions.

This module never submits broker orders and never mutates account/order state.  It
mechanically derives the currently executable logical STOP / TP legs from the
canonical strategy position state created by confirmed fills.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.exact_accounting import milli_to_price
from core.file_integrity import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    load_json_strict,
)
from core.params_io import build_params_from_mapping
from core.price_utils import calc_half_take_profit_sell_qty
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from core.trading_account_state import (
    MANAGEMENT_STATUS_ACTIVE,
    POSITION_SOURCE_STRATEGY_FILL,
    validate_trading_account_state,
)
from core.trading_order_state import validate_trading_order_state
from core.runtime_utils import get_taipei_now
from services.trading.account_state import resolve_trading_account_state_path
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.order_state import resolve_trading_order_state_path

PROTECTION_PLAN_SCHEMA_VERSION = 1
PROTECTION_PLAN_STATUS = "PROPOSED_PROTECTION"
PROTECTION_BROKER_STATUS = "NOT_SUBMITTED"
PROTECTION_STOP_ACTION = "STOP_FULL"
PROTECTION_TP_ACTION = "TP_HALF"
PROTECTION_STOP_ORDER_TYPE = "STOP_MARKET"
PROTECTION_TP_ORDER_TYPE = "LIMIT"
PROTECTION_SAME_BAR_PRIORITY = "STOP_OVER_TP"


def resolve_trading_protection_plan_dir(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="protection_orders"))


def resolve_trading_protection_plan_json_path(project_root: str | Path) -> Path:
    return resolve_trading_protection_plan_dir(project_root) / "protection_plan.json"


def resolve_trading_protection_plan_text_path(project_root: str | Path) -> Path:
    return resolve_trading_protection_plan_dir(project_root) / "protection_plan.txt"


def _read_state_with_file_sha(path: Path, validator) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    state = load_json_strict(path)
    validator(state)
    return state, file_sha


def _collect_strategy_sources(
    account: dict[str, Any],
    orders: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    strategy_positions: list[dict[str, Any]] = []
    source_orders: dict[str, dict[str, Any]] = {}
    manual_skipped: list[str] = []

    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            manual_skipped.append(str(ticker))
            continue
        management = record.get("strategy_management") or {}
        position_state = management.get("position_state")
        if management.get("status") != MANAGEMENT_STATUS_ACTIVE or not isinstance(position_state, dict):
            raise RuntimeError(f"Trading strategy position 缺少 active canonical position state: {ticker}")
        broker = record.get("broker") or {}
        order_id = str(broker.get("entry_order_id") or "").strip()
        if not order_id:
            raise RuntimeError(f"Trading strategy position 缺少 entry_order_id，無法取得 frozen params: {ticker}")
        order = (orders.get("orders") or {}).get(order_id)
        if not isinstance(order, dict):
            raise RuntimeError(f"Trading strategy position 對應 entry order 不存在: {ticker} / {order_id}")
        frozen_params = order.get("frozen_params")
        if not isinstance(frozen_params, dict):
            raise RuntimeError(f"Trading entry order 缺少 frozen_params，禁止用 current params 回填保護單: {ticker}")
        expected_params_sha = str(order.get("frozen_params_sha256") or "")
        if expected_params_sha != canonical_json_sha256(frozen_params):
            raise RuntimeError(f"Trading entry order frozen_params hash 不一致: {ticker}")
        filled_qty = int(order.get("filled_qty") or 0)
        if filled_qty <= 0:
            raise RuntimeError(f"Trading strategy position 對應 order 尚無 confirmed fill: {ticker}")
        broker_qty = int(broker.get("qty") or 0)
        position_qty = int(position_state.get("qty") or 0)
        if broker_qty <= 0 or position_qty != broker_qty:
            raise RuntimeError(f"Trading broker/strategy position qty 不一致: {ticker}")
        if broker_qty > filled_qty:
            raise RuntimeError(f"Trading position qty 不得大於 entry order confirmed filled qty: {ticker}")

        strategy_positions.append(
            {
                "ticker": str(ticker),
                "broker": deepcopy(broker),
                "strategy_management": deepcopy(management),
            }
        )
        source_orders[order_id] = deepcopy(order)

    return strategy_positions, source_orders, manual_skipped


def _build_source_hashes(
    strategy_positions: list[dict[str, Any]],
    source_orders: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    return (
        canonical_json_sha256(strategy_positions),
        canonical_json_sha256({key: source_orders[key] for key in sorted(source_orders)}),
    )


def _build_position_plan(
    source: dict[str, Any],
    source_orders: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ticker = str(source["ticker"])
    broker = source["broker"]
    management = source["strategy_management"]
    position = management["position_state"]
    order_id = str(broker["entry_order_id"])
    order = source_orders[order_id]
    params = build_params_from_mapping(order["frozen_params"])

    qty = int(position.get("qty") or 0)
    stop_milli = int(position.get("sl_milli") or 0)
    target_milli = int(position.get("tp_half_milli") or 0)
    entry_fill_milli = int(position.get("entry_fill_price_milli") or 0)
    if qty <= 0 or stop_milli <= 0 or target_milli <= 0 or entry_fill_milli <= 0:
        raise RuntimeError(f"Trading position 無法形成有效 Stop/TP 保護單: {ticker}")

    sold_half = bool(position.get("sold_half", False))
    tp_qty = 0 if sold_half else calc_half_take_profit_sell_qty(qty, params.tp_percent)
    legs: list[dict[str, Any]] = [
        {
            "action": PROTECTION_STOP_ACTION,
            "side": "SELL",
            "order_type": PROTECTION_STOP_ORDER_TYPE,
            "qty": qty,
            "trigger_price_milli": stop_milli,
            "trigger_price": milli_to_price(stop_milli),
            "limit_price_milli": None,
            "limit_price": None,
            "priority": 1,
            "broker_status": PROTECTION_BROKER_STATUS,
        }
    ]
    if tp_qty > 0:
        legs.append(
            {
                "action": PROTECTION_TP_ACTION,
                "side": "SELL",
                "order_type": PROTECTION_TP_ORDER_TYPE,
                "qty": int(tp_qty),
                "trigger_price_milli": None,
                "trigger_price": None,
                "limit_price_milli": target_milli,
                "limit_price": milli_to_price(target_milli),
                "priority": 2,
                "broker_status": PROTECTION_BROKER_STATUS,
            }
        )

    row = {
        "ticker": ticker,
        "position_qty": qty,
        "entry_order_id": order_id,
        "entry_order_status": str(order.get("status") or ""),
        "entry_order_filled_qty": int(order.get("filled_qty") or 0),
        "entry_order_remaining_qty": int(order.get("remaining_qty") or 0),
        "entry_trade_date": broker.get("entry_date"),
        "entry_fill_price_milli": entry_fill_milli,
        "entry_fill_price": milli_to_price(entry_fill_milli),
        "effective_stop_milli": stop_milli,
        "effective_stop": milli_to_price(stop_milli),
        "target_price_milli": target_milli,
        "target_price": milli_to_price(target_milli),
        "sold_half": sold_half,
        "tp_sell_qty": int(tp_qty),
        "same_bar_priority": PROTECTION_SAME_BAR_PRIORITY,
        "frozen_params_sha256": str(order.get("frozen_params_sha256") or ""),
        "legs": legs,
    }
    row["position_plan_fingerprint"] = canonical_json_sha256(row)
    return row


def _build_plan_payload(
    *,
    account: dict[str, Any],
    orders: dict[str, Any],
    strategy_positions: list[dict[str, Any]],
    source_orders: dict[str, dict[str, Any]],
    manual_skipped: list[str],
) -> dict[str, Any]:
    strategy_positions_sha256, source_orders_sha256 = _build_source_hashes(strategy_positions, source_orders)
    position_plans = [_build_position_plan(row, source_orders) for row in strategy_positions]
    identity = {
        "schema_version": PROTECTION_PLAN_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "status": PROTECTION_PLAN_STATUS,
        "broker_status": PROTECTION_BROKER_STATUS,
        "account_revision": int(account["revision"]),
        "order_revision": int(orders["revision"]),
        "strategy_positions_sha256": strategy_positions_sha256,
        "source_orders_sha256": source_orders_sha256,
        "same_bar_priority": PROTECTION_SAME_BAR_PRIORITY,
        "market_data_used": False,
        "discretionary_input_used": False,
        "account_mutated": False,
        "broker_submitted": False,
        "manual_positions_skipped": list(manual_skipped),
        "positions": position_plans,
    }
    payload = dict(identity)
    payload["generated_at"] = get_taipei_now().isoformat(timespec="seconds")
    payload["plan_fingerprint"] = canonical_json_sha256(identity)
    return payload


def validate_trading_protection_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise TypeError("Trading protection plan 必須是 dict")
    if int(plan.get("schema_version", -1)) != PROTECTION_PLAN_SCHEMA_VERSION:
        raise ValueError("Trading protection plan schema_version 不相容")
    if str(plan.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading protection plan runtime_domain 必須是 trading")
    if str(plan.get("status") or "") != PROTECTION_PLAN_STATUS:
        raise ValueError("Trading protection plan status 不合法")
    if bool(plan.get("account_mutated")) or bool(plan.get("broker_submitted")):
        raise ValueError("Trading protection plan 不得修改 account 或宣稱已送券商")
    if bool(plan.get("market_data_used")) or bool(plan.get("discretionary_input_used")):
        raise ValueError("Trading protection plan 不得使用成交後市場資料或 discretionary input")
    if str(plan.get("same_bar_priority") or "") != PROTECTION_SAME_BAR_PRIORITY:
        raise ValueError("Trading protection plan same-bar priority 不合法")
    rows = plan.get("positions")
    if not isinstance(rows, list):
        raise ValueError("Trading protection plan positions 必須是 list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Trading protection position row 必須是 object")
        qty = int(row.get("position_qty") or 0)
        if qty <= 0:
            raise ValueError("Trading protection position qty 必須 > 0")
        legs = row.get("legs")
        if not isinstance(legs, list) or not legs:
            raise ValueError("Trading protection position 至少必須有 STOP leg")
        stop_legs = [leg for leg in legs if leg.get("action") == PROTECTION_STOP_ACTION]
        if len(stop_legs) != 1 or int(stop_legs[0].get("qty") or 0) != qty:
            raise ValueError("Trading protection STOP leg 必須完整保護目前持股 qty")
        if str(stop_legs[0].get("order_type") or "") != PROTECTION_STOP_ORDER_TYPE:
            raise ValueError("Trading protection STOP 必須使用 canonical stop-market semantics")
        tp_legs = [leg for leg in legs if leg.get("action") == PROTECTION_TP_ACTION]
        if len(tp_legs) > 1:
            raise ValueError("Trading protection TP leg 不得重複")
        if tp_legs and int(tp_legs[0].get("qty") or 0) != int(row.get("tp_sell_qty") or 0):
            raise ValueError("Trading protection TP qty 與 canonical half-take-profit qty 不一致")
        if any(str(leg.get("broker_status") or "") != PROTECTION_BROKER_STATUS for leg in legs):
            raise ValueError("Trading protection logical legs 不得宣稱已送券商")
        expected_fp = canonical_json_sha256({key: value for key, value in row.items() if key != "position_plan_fingerprint"})
        if str(row.get("position_plan_fingerprint") or "") != expected_fp:
            raise ValueError("Trading protection position fingerprint 不一致")

    identity = {
        key: value
        for key, value in plan.items()
        if key not in {"generated_at", "plan_fingerprint", "json_path", "text_path"}
    }
    if str(plan.get("plan_fingerprint") or "") != canonical_json_sha256(identity):
        raise ValueError("Trading protection plan fingerprint 不一致")


def _render_protection_plan_text(plan: dict[str, Any]) -> str:
    lines = [
        "Trading 成交後 Stop / TP 保護單計畫",
        "=" * 72,
        f"狀態：{plan['status']} / {plan['broker_status']}",
        f"Account revision：{plan['account_revision']} | Order revision：{plan['order_revision']}",
        "注意：本檔只描述依既定策略機械派生的邏輯保護腿，尚未送至券商。",
        f"同一 bar 同時碰 Stop / TP 時優先序：{plan['same_bar_priority']}",
        "",
    ]
    if not plan["positions"]:
        lines.append("目前沒有可由策略機械派生保護單的 strategy_fill 持股。")
    for row in plan["positions"]:
        lines.extend(
            [
                f"[{row['ticker']}] 持股 {int(row['position_qty']):,} | 成交均價 {row['entry_fill_price']:.3f} | entry order {row['entry_order_status']}",
                f"  STOP_FULL : {int(row['position_qty']):,} 股，Stop trigger {row['effective_stop']:.3f}，STOP_MARKET（尚未送券商）",
            ]
        )
        if int(row["tp_sell_qty"]) > 0:
            lines.append(
                f"  TP_HALF   : {int(row['tp_sell_qty']):,} 股，Limit {row['target_price']:.3f}（尚未送券商）"
            )
        else:
            lines.append("  TP_HALF   : 無可執行半倉數量或已完成半倉停利")
        lines.append("")
    if plan.get("manual_positions_skipped"):
        lines.append("未自動接管的 manual adopted 持股：" + ", ".join(plan["manual_positions_skipped"]))
    lines.append(f"Plan fingerprint：{plan['plan_fingerprint']}")
    return "\n".join(lines).rstrip() + "\n"


def build_trading_protection_plan(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    account_path = resolve_trading_account_state_path(root)
    order_path = resolve_trading_order_state_path(root)
    if not account_path.is_file():
        raise FileNotFoundError(f"Trading account state 尚未初始化: {account_path}")
    if not order_path.is_file():
        raise FileNotFoundError(f"Trading order state 尚未建立: {order_path}")

    account, account_file_sha = _read_state_with_file_sha(account_path, validate_trading_account_state)
    orders, order_file_sha = _read_state_with_file_sha(order_path, validate_trading_order_state)
    strategy_positions, source_orders, manual_skipped = _collect_strategy_sources(account, orders)
    plan = _build_plan_payload(
        account=account,
        orders=orders,
        strategy_positions=strategy_positions,
        source_orders=source_orders,
        manual_skipped=manual_skipped,
    )
    validate_trading_protection_plan(plan)

    if hashlib.sha256(account_path.read_bytes()).hexdigest() != account_file_sha:
        raise RuntimeError("Trading account 在保護單計畫建立期間已變更")
    if hashlib.sha256(order_path.read_bytes()).hexdigest() != order_file_sha:
        raise RuntimeError("Trading order state 在保護單計畫建立期間已變更")

    json_path = resolve_trading_protection_plan_json_path(root)
    text_path = resolve_trading_protection_plan_text_path(root)
    atomic_write_json(json_path, plan)
    atomic_write_text(text_path, _render_protection_plan_text(plan))
    return {
        **plan,
        "json_path": project_relative_display_path(json_path, project_root=root),
        "text_path": project_relative_display_path(text_path, project_root=root),
    }


def load_trading_protection_plan(project_root: str | Path, *, required: bool = False) -> dict[str, Any] | None:
    path = resolve_trading_protection_plan_json_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Trading protection plan 尚未建立: {path}")
        return None
    plan = load_json_strict(path)
    validate_trading_protection_plan(plan)
    return plan


def get_trading_protection_plan_read_model(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    recover_trading_fill_transaction(root)
    plan = load_trading_protection_plan(root, required=False)
    json_path = resolve_trading_protection_plan_json_path(root)
    text_path = resolve_trading_protection_plan_text_path(root)
    if plan is None:
        return {
            "exists": False,
            "fresh": False,
            "status": None,
            "positions": [],
            "position_count": 0,
            "json_path": project_relative_display_path(json_path, project_root=root),
            "text_path": project_relative_display_path(text_path, project_root=root),
        }

    account_path = resolve_trading_account_state_path(root)
    order_path = resolve_trading_order_state_path(root)
    fresh = False
    if account_path.is_file() and order_path.is_file():
        account = load_json_strict(account_path)
        orders = load_json_strict(order_path)
        validate_trading_account_state(account)
        validate_trading_order_state(orders)
        strategy_positions, source_orders, _manual = _collect_strategy_sources(account, orders)
        strategy_sha, source_order_sha = _build_source_hashes(strategy_positions, source_orders)
        fresh = (
            strategy_sha == str(plan.get("strategy_positions_sha256") or "")
            and source_order_sha == str(plan.get("source_orders_sha256") or "")
        )

    rows = []
    for row in plan.get("positions") or []:
        stop_leg = next((leg for leg in row.get("legs") or [] if leg.get("action") == PROTECTION_STOP_ACTION), {})
        tp_leg = next((leg for leg in row.get("legs") or [] if leg.get("action") == PROTECTION_TP_ACTION), {})
        rows.append(
            {
                "ticker": row.get("ticker"),
                "position_qty": int(row.get("position_qty") or 0),
                "entry_fill_price": row.get("entry_fill_price"),
                "stop_qty": int(stop_leg.get("qty") or 0),
                "stop_price": stop_leg.get("trigger_price"),
                "tp_qty": int(tp_leg.get("qty") or 0),
                "target_price": tp_leg.get("limit_price"),
                "entry_order_status": row.get("entry_order_status"),
                "same_bar_priority": row.get("same_bar_priority"),
            }
        )
    return {
        "exists": True,
        "fresh": bool(fresh),
        "status": plan.get("status"),
        "broker_status": plan.get("broker_status"),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "account_revision": plan.get("account_revision"),
        "order_revision": plan.get("order_revision"),
        "position_count": len(rows),
        "positions": rows,
        "manual_positions_skipped": list(plan.get("manual_positions_skipped") or []),
        "json_path": project_relative_display_path(json_path, project_root=root),
        "text_path": project_relative_display_path(text_path, project_root=root),
    }


def load_current_trading_protection_plan(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    snapshot = get_trading_protection_plan_read_model(root)
    if not snapshot.get("exists"):
        raise FileNotFoundError("Trading protection plan 尚未建立；請先建立／刷新保護單計畫")
    if not snapshot.get("fresh"):
        raise RuntimeError("Trading protection plan 已 STALE；請先依最新實際持股／entry order 重新建立")
    return load_trading_protection_plan(root, required=True)


__all__ = [
    "PROTECTION_PLAN_SCHEMA_VERSION",
    "PROTECTION_PLAN_STATUS",
    "PROTECTION_BROKER_STATUS",
    "PROTECTION_STOP_ACTION",
    "PROTECTION_TP_ACTION",
    "PROTECTION_SAME_BAR_PRIORITY",
    "resolve_trading_protection_plan_dir",
    "resolve_trading_protection_plan_json_path",
    "resolve_trading_protection_plan_text_path",
    "validate_trading_protection_plan",
    "build_trading_protection_plan",
    "load_trading_protection_plan",
    "load_current_trading_protection_plan",
    "get_trading_protection_plan_read_model",
]
