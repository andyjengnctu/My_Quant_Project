"""Persistent rule-based indicator-exit obligations for live Trading positions.

A completed daily bar may create an ``ind_sell_signal`` obligation. The
obligation is immutable by ``signal_key`` and survives later market-data
refreshes until its broker-order lifecycle is explicitly handled. This module
never infers a broker submission or fill.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256, load_json_strict
from core.params_io import build_params_from_mapping
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from core.runtime_utils import get_taipei_now
from core.signal_utils import generate_signals, unpack_precomputed_signals
from core.trading_account_state import MANAGEMENT_STATUS_ACTIVE, POSITION_SOURCE_STRATEGY_FILL
from core.trading_market_clock import latest_allowed_completed_daily_date
from core.trading_order_state import (
    TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
    TRADING_ORDER_STATUS_FILLED,
    active_trading_indicator_exit_orders,
)
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.position_market_context import (
    load_trading_position_market_frame,
    normalize_trading_date,
    resolve_trading_strategy_position_sources,
)

INDICATOR_EXIT_PLAN_SCHEMA_VERSION = 1
INDICATOR_EXIT_PLAN_STATUS = "PROPOSED_INDICATOR_EXIT"
INDICATOR_EXIT_BROKER_STATUS = "NOT_SUBMITTED"
INDICATOR_EXIT_ORDER_TYPE = "MARKET"
INDICATOR_EXIT_EXECUTION_SEMANTICS = "COMPLETED_BAR_SIGNAL_NEXT_SESSION_MARKET_SELL"


def resolve_trading_indicator_exit_plan_dir(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="indicator_exit_orders"))


def resolve_trading_indicator_exit_plan_json_path(project_root: str | Path) -> Path:
    return resolve_trading_indicator_exit_plan_dir(project_root) / "indicator_exit_plan.json"


def resolve_trading_indicator_exit_plan_text_path(project_root: str | Path) -> Path:
    return resolve_trading_indicator_exit_plan_dir(project_root) / "indicator_exit_plan.txt"


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signal_key(*, ticker: str, entry_order_id: str, signal_information_date: str, frozen_params_sha256: str) -> str:
    return canonical_json_sha256({
        "ticker": str(ticker),
        "entry_order_id": str(entry_order_id),
        "signal_information_date": str(signal_information_date),
        "frozen_params_sha256": str(frozen_params_sha256),
        "execution_semantics": INDICATOR_EXIT_EXECUTION_SEMANTICS,
    })


def _source_binding(*, ticker: str, record: dict[str, Any], orders: dict[str, Any], file_path: str) -> dict[str, Any]:
    management = record.get("strategy_management") or {}
    position = management.get("position_state")
    if management.get("status") != MANAGEMENT_STATUS_ACTIVE or not isinstance(position, dict):
        raise RuntimeError(f"Trading strategy position 缺少 active canonical position state: {ticker}")
    broker = record.get("broker") or {}
    entry_order_id = str(broker.get("entry_order_id") or "").strip()
    order = (orders.get("orders") or {}).get(entry_order_id)
    if not isinstance(order, dict) or not isinstance(order.get("frozen_params"), dict):
        raise RuntimeError(f"Trading position 缺少來源 entry order frozen params: {ticker}")
    params_sha = canonical_json_sha256(order["frozen_params"])
    if params_sha != str(order.get("frozen_params_sha256") or ""):
        raise RuntimeError(f"Trading entry order frozen_params hash 不一致: {ticker}")
    broker_qty = int(broker.get("qty") or 0)
    position_qty = int(position.get("qty") or 0)
    if broker_qty <= 0 or broker_qty != position_qty:
        raise RuntimeError(f"Trading broker/strategy position qty 不一致: {ticker}")
    return {
        "ticker": str(ticker),
        "entry_order_id": entry_order_id,
        "entry_trade_date": normalize_trading_date(broker.get("entry_date")),
        "position_qty": position_qty,
        "position_state_sha256": canonical_json_sha256(position),
        "last_rollforward_date": normalize_trading_date(management.get("last_rollforward_date")),
        "frozen_params_sha256": params_sha,
        "market_data_sha256": _file_sha256(file_path),
    }


def _collect_current_bindings(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[dict[str, Any]], list[str]]:
    account, orders, csv_map = resolve_trading_strategy_position_sources(root)
    if not account:
        raise FileNotFoundError("Trading account state 尚未初始化")
    bindings: list[dict[str, Any]] = []
    manual: list[str] = []
    for ticker in sorted(account.get("positions") or {}):
        record = account["positions"][ticker]
        if record.get("source") != POSITION_SOURCE_STRATEGY_FILL:
            manual.append(str(ticker)); continue
        file_path = csv_map.get(ticker)
        if not file_path:
            raise FileNotFoundError(f"Trading dataset 缺少持股 {ticker} CSV")
        bindings.append(_source_binding(ticker=str(ticker), record=record, orders=orders, file_path=file_path))
    return account, orders, csv_map, bindings, manual


def _prior_unresolved_row(prior: dict[str, Any] | None, *, binding: dict[str, Any], orders: dict[str, Any]) -> dict[str, Any] | None:
    if not prior:
        return None
    for row in prior.get("exits") or []:
        if str(row.get("ticker")) != binding["ticker"] or str(row.get("entry_order_id")) != binding["entry_order_id"]:
            continue
        related = [o for o in (orders.get("orders") or {}).values() if str(o.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT and str(o.get("signal_key") or "") == str(row.get("signal_key") or "")]
        if any(str(o.get("status") or "") == TRADING_ORDER_STATUS_FILLED for o in related):
            raise RuntimeError(f"Trading Indicator SELL 已 FILLED 但持股仍存在: {binding['ticker']}")
        return deepcopy(row)
    return None


def _build_exit_row(*, binding: dict[str, Any], signal_information_date: str, carried_forward: bool, signal_origin_market_data_sha256: str) -> dict[str, Any]:
    row = {
        "ticker": binding["ticker"],
        "side": "SELL",
        "purpose": TRADING_ORDER_PURPOSE_INDICATOR_EXIT,
        "order_type": INDICATOR_EXIT_ORDER_TYPE,
        "execution_semantics": INDICATOR_EXIT_EXECUTION_SEMANTICS,
        "signal_information_date": signal_information_date,
        "signal_key": _signal_key(
            ticker=binding["ticker"], entry_order_id=binding["entry_order_id"],
            signal_information_date=signal_information_date, frozen_params_sha256=binding["frozen_params_sha256"],
        ),
        "qty": int(binding["position_qty"]),
        "position_qty": int(binding["position_qty"]),
        "entry_order_id": binding["entry_order_id"],
        "entry_trade_date": binding["entry_trade_date"],
        "priority": 1,
        "position_state_sha256": binding["position_state_sha256"],
        "frozen_params_sha256": binding["frozen_params_sha256"],
        "market_data_sha256": binding["market_data_sha256"],
        "signal_origin_market_data_sha256": signal_origin_market_data_sha256,
        "carried_forward": bool(carried_forward),
        "broker_status": INDICATOR_EXIT_BROKER_STATUS,
        "broker_fill_inferred": False,
    }
    row["position_plan_fingerprint"] = canonical_json_sha256(row)
    return row


def validate_trading_indicator_exit_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict): raise TypeError("Trading indicator exit plan 必須是 dict")
    if int(plan.get("schema_version", -1)) != INDICATOR_EXIT_PLAN_SCHEMA_VERSION: raise ValueError("Trading indicator exit plan schema_version 不相容")
    if str(plan.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING: raise ValueError("Trading indicator exit runtime_domain 必須是 trading")
    if str(plan.get("status") or "") != INDICATOR_EXIT_PLAN_STATUS: raise ValueError("Trading indicator exit status 不合法")
    if str(plan.get("execution_semantics") or "") != INDICATOR_EXIT_EXECUTION_SEMANTICS: raise ValueError("Trading indicator exit semantics 不合法")
    if bool(plan.get("broker_submitted")) or bool(plan.get("broker_fill_inferred")): raise ValueError("Indicator plan 不得宣稱已送單／推測成交")
    bindings = plan.get("source_bindings"); exits = plan.get("exits")
    if not isinstance(bindings, list) or not isinstance(exits, list): raise ValueError("Indicator plan bindings/exits 必須是 list")
    by_key = {(str(x.get("ticker")), str(x.get("entry_order_id"))): x for x in bindings}
    seen: set[str] = set()
    for row in exits:
        key=(str(row.get("ticker")), str(row.get("entry_order_id"))); binding=by_key.get(key)
        if binding is None: raise ValueError("Indicator exit 缺少 current source binding")
        if int(row.get("qty") or 0) != int(binding.get("position_qty") or 0) or int(row.get("position_qty") or 0) != int(binding.get("position_qty") or 0): raise ValueError("Indicator exit 必須完整覆蓋持股")
        if str(row.get("order_type")) != INDICATOR_EXIT_ORDER_TYPE or str(row.get("execution_semantics")) != INDICATOR_EXIT_EXECUTION_SEMANTICS: raise ValueError("Indicator exit MARKET semantics 不合法")
        expected_key=_signal_key(ticker=key[0], entry_order_id=key[1], signal_information_date=str(row.get("signal_information_date") or ""), frozen_params_sha256=str(row.get("frozen_params_sha256") or ""))
        if str(row.get("signal_key") or "") != expected_key or expected_key in seen: raise ValueError("Indicator signal_key 不一致或重複")
        seen.add(expected_key)
        expected_fp=canonical_json_sha256({k:v for k,v in row.items() if k!="position_plan_fingerprint"})
        if str(row.get("position_plan_fingerprint") or "") != expected_fp: raise ValueError("Indicator position fingerprint 不一致")
        for field in ("position_state_sha256","frozen_params_sha256","market_data_sha256","signal_origin_market_data_sha256"):
            if not str(row.get(field) or ""): raise ValueError(f"Indicator exit 缺少 {field}")
    if int(plan.get("exit_count") or 0) != len(exits): raise ValueError("Indicator exit_count 不一致")
    identity={k:v for k,v in plan.items() if k not in {"generated_at","plan_fingerprint","json_path","text_path"}}
    if str(plan.get("plan_fingerprint") or "") != canonical_json_sha256(identity): raise ValueError("Indicator plan fingerprint 不一致")


def _render_indicator_exit_plan_text(plan: dict[str, Any]) -> str:
    lines=[
        "Trading Rule-based Indicator Sell 計畫", "="*72,
        f"狀態：{plan['status']} / {INDICATOR_EXIT_BROKER_STATUS}",
        f"Completed-bar cutoff：{plan['allowed_completed_date']}",
        "語意：completed bar ind_sell_signal -> 下一可交易 session 的 MARKET SELL。",
        "本檔只保存策略義務；不宣稱已送券商，也不以日K推測成交。",
        "舊 signal 若尚未完成 broker-order lifecycle，後續資料更新不得將它洗掉。", "",
    ]
    if not plan["exits"]: lines.append("目前沒有待執行的 Indicator SELL 義務。")
    for row in plan["exits"]:
        carry="carried-forward" if row.get("carried_forward") else "new"
        lines.append(f"[{row['ticker']}] signal={row['signal_information_date']} | qty={int(row['qty']):,} | MARKET SELL / {carry}")
        lines.append(f"  signal_key={row['signal_key']}")
    if plan.get("manual_positions_skipped"): lines.append("未自動接管的 manual adopted 持股："+", ".join(plan["manual_positions_skipped"]))
    lines.append(f"Plan fingerprint：{plan['plan_fingerprint']}")
    return "\n".join(lines).rstrip()+"\n"


def build_trading_indicator_exit_plan(project_root: str | Path) -> dict[str, Any]:
    root=Path(project_root).resolve(); recover_trading_fill_transaction(root)
    account, orders, csv_map, bindings, manual=_collect_current_bindings(root)
    allowed_date=latest_allowed_completed_daily_date()
    prior=load_trading_indicator_exit_plan(root, required=False)
    exits=[]
    for binding in bindings:
        carried=_prior_unresolved_row(prior, binding=binding, orders=orders)
        if carried is not None:
            exits.append(_build_exit_row(binding=binding, signal_information_date=str(carried["signal_information_date"]), carried_forward=True, signal_origin_market_data_sha256=str(carried.get("signal_origin_market_data_sha256") or carried.get("market_data_sha256") or binding["market_data_sha256"])))
            continue
        order=(orders.get("orders") or {})[binding["entry_order_id"]]
        params=build_params_from_mapping(order["frozen_params"])
        df=load_trading_position_market_frame(file_path=csv_map[binding["ticker"]], ticker=binding["ticker"], params=params, allowed_date=allowed_date)
        entry_date=binding["entry_trade_date"]
        eligible=df if entry_date is None else df.loc[df.index >= pd.Timestamp(entry_date)]
        if eligible.empty: continue
        latest_date=pd.Timestamp(eligible.index[-1]).strftime("%Y-%m-%d")
        if binding["last_rollforward_date"] is None or binding["last_rollforward_date"] < latest_date:
            raise RuntimeError(f"Trading position 尚未 roll-forward 至最新 completed bar: {binding['ticker']}")
        precomputed=generate_signals(df, params, ticker=binding["ticker"])
        _atr,_buy,sell,_limits=unpack_precomputed_signals(precomputed)
        if bool(sell[len(df)-1]):
            exits.append(_build_exit_row(binding=binding, signal_information_date=latest_date, carried_forward=False, signal_origin_market_data_sha256=binding["market_data_sha256"]))
    identity={
        "schema_version": INDICATOR_EXIT_PLAN_SCHEMA_VERSION, "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "status": INDICATOR_EXIT_PLAN_STATUS, "broker_status": INDICATOR_EXIT_BROKER_STATUS,
        "execution_semantics": INDICATOR_EXIT_EXECUTION_SEMANTICS, "allowed_completed_date": allowed_date,
        "account_revision": int(account["revision"]), "source_bindings_sha256": canonical_json_sha256(bindings),
        "source_bindings": bindings, "manual_positions_skipped": manual, "exit_count": len(exits), "exits": exits,
        "broker_submitted": False, "broker_fill_inferred": False,
    }
    plan=dict(identity); plan["generated_at"]=get_taipei_now().isoformat(timespec="seconds"); plan["plan_fingerprint"]=canonical_json_sha256(identity)
    validate_trading_indicator_exit_plan(plan)
    json_path=resolve_trading_indicator_exit_plan_json_path(root); text_path=resolve_trading_indicator_exit_plan_text_path(root)
    atomic_write_json(json_path, plan); atomic_write_text(text_path, _render_indicator_exit_plan_text(plan))
    return {**plan, "json_path":project_relative_display_path(json_path, project_root=root), "text_path":project_relative_display_path(text_path, project_root=root)}


def load_trading_indicator_exit_plan(project_root: str | Path, *, required: bool=False) -> dict[str, Any] | None:
    path=resolve_trading_indicator_exit_plan_json_path(project_root)
    if not path.is_file():
        if required: raise FileNotFoundError(f"Trading indicator exit plan 尚未建立: {path}")
        return None
    plan=load_json_strict(path); validate_trading_indicator_exit_plan(plan); return plan


def get_trading_indicator_exit_plan_read_model(project_root: str | Path, *, recover_pending_fill: bool=True) -> dict[str, Any]:
    root=Path(project_root).resolve()
    if recover_pending_fill: recover_trading_fill_transaction(root)
    plan=load_trading_indicator_exit_plan(root, required=False)
    json_path=resolve_trading_indicator_exit_plan_json_path(root); text_path=resolve_trading_indicator_exit_plan_text_path(root)
    if plan is None:
        return {"exists":False,"fresh":False,"status":None,"exit_count":0,"exits":[],"active_indicator_exit_order_count":0,"active_indicator_exit_tickers":[],"json_path":project_relative_display_path(json_path,project_root=root),"text_path":project_relative_display_path(text_path,project_root=root)}
    fresh=False; active=[]
    try:
        account, orders, _csv, bindings, _manual=_collect_current_bindings(root)
        fresh=int(account["revision"])==int(plan.get("account_revision")) and canonical_json_sha256(bindings)==str(plan.get("source_bindings_sha256") or "")
        active=active_trading_indicator_exit_orders(orders)
    except (FileNotFoundError, RuntimeError, ValueError):
        fresh=False; active=[]
    rows=[{k:row.get(k) for k in ("ticker","signal_information_date","qty","order_type","entry_trade_date","entry_order_id","signal_key","carried_forward","position_plan_fingerprint")} for row in plan.get("exits") or []]
    return {"exists":True,"fresh":fresh,"status":plan.get("status"),"exit_count":len(rows),"exits":rows,"active_indicator_exit_order_count":len(active),"active_indicator_exit_tickers":sorted({str(x.get("ticker")) for x in active}),"json_path":project_relative_display_path(json_path,project_root=root),"text_path":project_relative_display_path(text_path,project_root=root),"plan_fingerprint":plan.get("plan_fingerprint")}


def load_current_trading_indicator_exit_plan(project_root: str | Path) -> dict[str, Any]:
    model=get_trading_indicator_exit_plan_read_model(project_root)
    if not model.get("exists") or not model.get("fresh"):
        raise RuntimeError("Trading Indicator SELL 計畫不存在或已 STALE，請先刷新")
    return load_trading_indicator_exit_plan(project_root, required=True)


__all__=[
    "INDICATOR_EXIT_PLAN_SCHEMA_VERSION","INDICATOR_EXIT_PLAN_STATUS","INDICATOR_EXIT_BROKER_STATUS","INDICATOR_EXIT_ORDER_TYPE","INDICATOR_EXIT_EXECUTION_SEMANTICS",
    "resolve_trading_indicator_exit_plan_dir","resolve_trading_indicator_exit_plan_json_path","resolve_trading_indicator_exit_plan_text_path",
    "validate_trading_indicator_exit_plan","build_trading_indicator_exit_plan","load_trading_indicator_exit_plan","load_current_trading_indicator_exit_plan","get_trading_indicator_exit_plan_read_model",
]
