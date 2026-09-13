"""Derived Trading account dashboard and performance read models.

`state/trading/account.json` remains the only mutable account truth.  This module
only derives valuation/performance views and may publish read-only snapshots under
`outputs/trading/account/` for inspection and diagnostics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.exact_accounting import (
    build_sell_ledger_from_price,
    calc_risk_budget_milli,
    milli_to_money,
)
from core.file_integrity import atomic_write_json
from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from core.runtime_utils import get_taipei_now
from core.trading_policy import resolve_trading_selected_strategy_param_path
from services.trading.account_state import load_trading_account_state, resolve_trading_account_state_path
from core.trading_account_state import effective_trading_account_events
from services.trading.accounting_policy import (
    build_standalone_trading_accounting_params,
    overlay_trading_accounting_params,
)
from services.trading.market_data_consumer import load_trading_v2_consumer_state
from services.trading.market_data_v2_view import TradingMarketDataV2View
from services.trading.strategy_param_runtime import load_trading_strategy_param_runtime

ACCOUNT_DASHBOARD_SCHEMA_VERSION = 1
ACCOUNT_DASHBOARD_ROLE = "derived_read_only_trading_account_dashboard"
ACCOUNT_DASHBOARD_FILENAME = "account_snapshot.json"
ACCOUNT_PERFORMANCE_FILENAME = "performance_summary.json"


def _safe_pct(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator or 0) <= 0:
        return None
    return float(numerator) / float(denominator) * 100.0


def _current_close_by_ticker(project_root: Path, tickers: list[str], market_date: str | None) -> tuple[dict[str, float], dict[str, str]]:
    if not tickers or not market_date:
        return {}, {}
    view = TradingMarketDataV2View.open(project_root)
    prices: dict[str, float] = {}
    errors: dict[str, str] = {}
    for ticker in tickers:
        try:
            frame = view.read_dataset_frame(
                FINMIND_ADJUSTED_PRICE_DATASET,
                columns=("date", "stock_id", "close"),
                data_id=ticker,
                end_date=market_date,
            )
            if frame.empty:
                errors[ticker] = "Trading V2 無可用收盤價"
                continue
            dated = frame.copy()
            dated["date"] = dated["date"].astype(str)
            dated = dated.loc[dated["date"] <= str(market_date)].sort_values("date")
            if dated.empty:
                errors[ticker] = "Trading V2 無 cutoff 內收盤價"
                continue
            close = float(dated.iloc[-1]["close"])
            if close <= 0:
                errors[ticker] = "Trading V2 收盤價不合法"
                continue
            prices[ticker] = close
        except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    return prices, errors


def _load_primary_params(project_root: Path):
    selected_path = Path(resolve_trading_selected_strategy_param_path(project_root))
    if not selected_path.is_file():
        return None, None, "Trading Params 尚未建立"
    try:
        runtime = load_trading_strategy_param_runtime(selected_path)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        return None, project_relative_display_path(selected_path, project_root=project_root), f"{type(exc).__name__}: {exc}"
    return runtime["primary_params"], project_relative_display_path(selected_path, project_root=project_root), None


def _build_closed_round_trips(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct closed position lifecycles only from immutable account events."""
    active: dict[str, dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    for event in effective_trading_account_events(state):
        mutation = str(event.get("mutation_type") or "")
        details = dict(event.get("details") or {})
        ticker = str(details.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if mutation == "adopt_manual_position":
            active[ticker] = {
                "ticker": ticker,
                "source": "manual_adopted",
                "entry_date": details.get("entry_date"),
                "cost_basis_milli": int(details.get("cost_basis_total_milli") or 0),
                "realized_pnl_milli": 0,
                "net_sell_total_milli": 0,
                "sell_count": 0,
            }
        elif mutation == "correct_manual_position" and ticker in active:
            broker = dict(details.get("broker") or {})
            active[ticker]["entry_date"] = broker.get("entry_date")
            active[ticker]["cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or 0)
        elif mutation == "remove_manual_position":
            active.pop(ticker, None)
        elif mutation == "manual_buy_fill":
            if ticker not in active:
                active[ticker] = {
                    "ticker": ticker,
                    "source": "manual_adopted",
                    "entry_date": details.get("trade_date"),
                    "cost_basis_milli": 0,
                    "realized_pnl_milli": 0,
                    "net_sell_total_milli": 0,
                    "sell_count": 0,
                }
            active[ticker]["cost_basis_milli"] += int(details.get("net_buy_total_milli") or 0)
            if active[ticker].get("entry_date") is None:
                active[ticker]["entry_date"] = details.get("trade_date")
        elif mutation == "confirm_strategy_buy_fill":
            active[ticker] = {
                "ticker": ticker,
                "source": "strategy_fill",
                "entry_date": details.get("trade_date"),
                "cost_basis_milli": int(details.get("net_buy_total_milli") or 0),
                "realized_pnl_milli": 0,
                "net_sell_total_milli": 0,
                "sell_count": 0,
            }
        elif mutation == "confirm_strategy_buy_fill_increment" and ticker in active:
            active[ticker]["cost_basis_milli"] += int(details.get("net_buy_total_milli") or 0)
        elif mutation == "confirm_sell_fill":
            lifecycle = active.get(ticker)
            if lifecycle is None:
                # Defensive compatibility for a historical state that predates a
                # complete event lineage.  Keep the row but do not invent entry cost.
                lifecycle = {
                    "ticker": ticker,
                    "source": "unknown",
                    "entry_date": None,
                    "cost_basis_milli": 0,
                    "realized_pnl_milli": 0,
                    "net_sell_total_milli": 0,
                    "sell_count": 0,
                }
                active[ticker] = lifecycle
            lifecycle["realized_pnl_milli"] += int(details.get("realized_pnl_milli") or 0)
            lifecycle["net_sell_total_milli"] += int(details.get("net_sell_total_milli") or 0)
            lifecycle["sell_count"] += 1
            if int(details.get("remaining_qty") or 0) <= 0:
                cost_milli = int(lifecycle.get("cost_basis_milli") or 0)
                pnl_milli = int(lifecycle.get("realized_pnl_milli") or 0)
                closed.append(
                    {
                        "ticker": ticker,
                        "source": lifecycle.get("source"),
                        "entry_date": lifecycle.get("entry_date"),
                        "exit_date": details.get("trade_date"),
                        "cost_basis": milli_to_money(cost_milli),
                        "net_sell_total": milli_to_money(int(lifecycle.get("net_sell_total_milli") or 0)),
                        "pnl": milli_to_money(pnl_milli),
                        "return_pct": _safe_pct(pnl_milli, cost_milli),
                        "sell_count": int(lifecycle.get("sell_count") or 0),
                    }
                )
                active.pop(ticker, None)
    return closed


def _build_transaction_details(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buys: list[dict[str, Any]] = []
    sells: list[dict[str, Any]] = []
    events = effective_trading_account_events(state)
    trade_mutations = {"manual_buy_fill", "confirm_strategy_buy_fill", "confirm_strategy_buy_fill_increment", "confirm_sell_fill"}
    latest_trade_revision_by_ticker: dict[str, int] = {}
    for event in events:
        if str(event.get("mutation_type") or "") not in trade_mutations:
            continue
        details = dict(event.get("details") or {})
        ticker = str(details.get("ticker") or "").strip().upper()
        if ticker:
            latest_trade_revision_by_ticker[ticker] = max(
                int(event.get("revision") or 0), latest_trade_revision_by_ticker.get(ticker, -1)
            )

    active_source: dict[str, str] = {}
    # Track the BUY events that still belong to the currently open lifecycle for
    # each ticker.  A completed sell closes that lifecycle; a later re-entry starts
    # a new one.  This lets Accounting Center show only the buys corresponding to
    # the selected current inventory instead of mixing in older closed cycles.
    active_buy_revisions: dict[str, set[int]] = {}
    for event in events:
        mutation = str(event.get("mutation_type") or "")
        details = dict(event.get("details") or {})
        ticker = str(details.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        revision = int(event.get("revision") or 0)
        if mutation == "adopt_manual_position":
            active_source[ticker] = "manual_adopted"
            continue
        if mutation == "remove_manual_position":
            active_source.pop(ticker, None)
            continue
        if mutation in {"manual_buy_fill", "confirm_strategy_buy_fill", "confirm_strategy_buy_fill_increment"}:
            qty = int(details.get("qty") or details.get("fill_qty") or 0)
            price_milli = details.get("entry_fill_price_milli")
            if price_milli is None:
                price_milli = details.get("fill_price_milli")
            gross_milli = details.get("gross_buy_milli")
            if gross_milli is None and price_milli is not None and qty > 0:
                gross_milli = int(price_milli) * qty
            net_milli = int(details.get("net_buy_total_milli") or 0)
            fee_milli = details.get("buy_fee_milli")
            if fee_milli is None and gross_milli is not None and net_milli > 0:
                fee_milli = net_milli - int(gross_milli)
            manual = mutation == "manual_buy_fill"
            source = "手動成交" if manual else "策略成交"
            active_source[ticker] = "manual_adopted" if manual else "strategy_fill"
            active_buy_revisions.setdefault(ticker, set()).add(revision)
            buys.append({
                "ticker": ticker,
                "trade_date": details.get("trade_date"),
                "price": None if price_milli is None else milli_to_money(int(price_milli)),
                "qty": qty,
                "gross_amount": None if gross_milli is None else milli_to_money(int(gross_milli)),
                "buy_fee": None if fee_milli is None else milli_to_money(int(fee_milli)),
                "holding_cost": None if net_milli <= 0 else milli_to_money(net_milli),
                "source": source,
                "revision": revision,
                "editable": True,
                "is_latest_ticker_trade": revision == latest_trade_revision_by_ticker.get(ticker),
            })
            continue
        if mutation == "confirm_sell_fill":
            qty = int(details.get("qty") or 0)
            price_milli = details.get("exec_price_milli")
            gross_milli = details.get("gross_sell_milli")
            if gross_milli is None and price_milli is not None and qty > 0:
                gross_milli = int(price_milli) * qty
            cost_milli = int(details.get("allocated_cost_milli") or 0)
            pnl_milli = int(details.get("realized_pnl_milli") or 0)
            source_before = str(details.get("position_source") or active_source.get(ticker) or "")
            strategy_managed = bool(details.get("strategy_managed")) or source_before == "strategy_fill"
            manual_sell = (
                str(details.get("event") or "") == "MANUAL_ACCOUNT_SELL"
                or (source_before == "manual_adopted" and not strategy_managed)
            )
            sells.append({
                "ticker": ticker,
                "trade_date": details.get("trade_date"),
                "qty": qty,
                "price": None if price_milli is None else milli_to_money(int(price_milli)),
                "gross_amount": None if gross_milli is None else milli_to_money(int(gross_milli)),
                "sell_fee": None if details.get("sell_fee_milli") is None else milli_to_money(int(details.get("sell_fee_milli") or 0)),
                "tax": None if details.get("tax_milli") is None else milli_to_money(int(details.get("tax_milli") or 0)),
                "offset_holding_cost": milli_to_money(cost_milli),
                "offset_gross_amount": None if details.get("allocated_gross_buy_milli") is None else milli_to_money(int(details.get("allocated_gross_buy_milli") or 0)),
                "offset_buy_fee": None if details.get("allocated_buy_fee_milli") is None else milli_to_money(int(details.get("allocated_buy_fee_milli") or 0)),
                "pnl": milli_to_money(pnl_milli),
                "return_pct": _safe_pct(pnl_milli, cost_milli),
                "remaining_qty": int(details.get("remaining_qty") or 0),
                "revision": revision,
                "source": "手動成交" if manual_sell else "策略成交",
                "editable": True,
                "is_latest_ticker_trade": revision == latest_trade_revision_by_ticker.get(ticker),
            })
            if int(details.get("remaining_qty") or 0) <= 0:
                active_source.pop(ticker, None)
                active_buy_revisions.pop(ticker, None)

    for row in buys:
        ticker = str(row.get("ticker") or "").strip().upper()
        row["open_position_related"] = int(row.get("revision") or 0) in active_buy_revisions.get(ticker, set())

    buys.sort(key=lambda row: (str(row.get("trade_date") or ""), int(row.get("revision") or 0)), reverse=True)
    sells.sort(key=lambda row: (str(row.get("trade_date") or ""), int(row.get("revision") or 0)), reverse=True)
    return buys, sells


def _performance_row(label: str, rows: list[dict[str, Any]], *, closed_only: bool) -> dict[str, Any]:
    cost = sum(float(row.get("cost_basis") or 0) for row in rows)
    pnl = sum(float(row.get("pnl") or 0) for row in rows)
    profitable = sum(1 for row in rows if float(row.get("pnl") or 0) > 0)
    return {
        "scope": label,
        "position_or_trade_count": len(rows),
        "cost_basis": cost,
        "pnl": pnl,
        "return_pct": None if cost <= 0 else pnl / cost * 100.0,
        "profitable_count": profitable,
        "profitable_rate_pct": None if not rows else profitable / len(rows) * 100.0,
        "rate_semantics": "closed_trade_win_rate" if closed_only else "profitable_position_or_trade_rate",
    }


def build_trading_account_dashboard_read_model(project_root) -> dict[str, Any]:
    root = Path(project_root).resolve()
    state = load_trading_account_state(root, required=False)
    state_path = resolve_trading_account_state_path(root)
    generated_at = get_taipei_now().isoformat(timespec="seconds")
    if state is None:
        return {
            "schema_version": ACCOUNT_DASHBOARD_SCHEMA_VERSION,
            "role": ACCOUNT_DASHBOARD_ROLE,
            "generated_at": generated_at,
            "initialized": False,
            "source_state_path": project_relative_display_path(state_path, project_root=root),
            "source_account_revision": None,
            "market_date": None,
            "positions": [],
            "buy_details": [],
            "sell_details": [],
            "closed_trades": [],
            "performance": [],
            "summary": {
                "cash": None,
                "holdings_market_value": 0.0,
                "holdings_net_liquidation": 0.0,
                "equity": None,
                "realized_pnl_open_positions": 0.0,
                "unrealized_pnl": None,
                "managed_open_risk": None,
                "single_position_risk_budget": None,
                "position_count": 0,
            },
            "warnings": ["Trading account 尚未初始化"],
        }

    warnings: list[str] = []
    consumer_state = None
    try:
        consumer_state = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        warnings.append(f"Trading V2 state: {type(exc).__name__}: {exc}")
    market_date = None if consumer_state is None else str(consumer_state.get("market_date") or "") or None

    open_records = dict(state.get("positions") or {})
    tickers = sorted(open_records)
    prices, price_errors = _current_close_by_ticker(root, tickers, market_date)
    for ticker, error in sorted(price_errors.items()):
        warnings.append(f"{ticker} 市價: {error}")

    params, params_path, params_error = _load_primary_params(root)
    if params_error:
        warnings.append(params_error)
    accounting_params = (
        overlay_trading_accounting_params(params)
        if params is not None
        else build_standalone_trading_accounting_params()
    )

    enriched_positions: list[dict[str, Any]] = []
    holdings_market_value = 0.0
    market_value_complete = True
    holdings_net_liquidation = 0.0
    net_liquidation_complete = True
    open_realized_pnl = 0.0
    open_unrealized_pnl = 0.0
    open_unrealized_complete = True
    managed_open_risk = 0.0
    managed_open_risk_complete = True

    for ticker in tickers:
        record = open_records[ticker]
        broker = dict(record.get("broker") or {})
        management = dict(record.get("strategy_management") or {})
        qty = int(broker.get("qty") or 0)
        remaining_cost_milli = int(broker.get("remaining_cost_basis_milli") or 0)
        realized_milli = int(broker.get("realized_pnl_milli") or 0)
        current_price = prices.get(ticker)
        market_value = None if current_price is None else float(current_price) * qty
        if market_value is not None:
            holdings_market_value += market_value
        else:
            market_value_complete = False
        open_realized_pnl += milli_to_money(realized_milli)

        position_state = management.get("position_state") if isinstance(management.get("position_state"), dict) else None
        effective_stop = None
        if position_state is not None:
            sl_milli = int(position_state.get("sl_milli") or 0)
            effective_stop = None if sl_milli <= 0 else milli_to_money(sl_milli)

        net_liquidation = None
        unrealized = None
        risk_to_stop = None
        if current_price is not None and qty > 0 and market_date:
            try:
                current_ledger = build_sell_ledger_from_price(current_price, qty, accounting_params, ticker=ticker, trade_date=market_date)
                current_net_milli = int(current_ledger["net_sell_total_milli"])
                net_liquidation = milli_to_money(current_net_milli)
                holdings_net_liquidation += net_liquidation
                unrealized = milli_to_money(current_net_milli - remaining_cost_milli)
                open_unrealized_pnl += unrealized
                if effective_stop is not None:
                    stop_ledger = build_sell_ledger_from_price(effective_stop, qty, accounting_params, ticker=ticker, trade_date=market_date)
                    risk_to_stop = milli_to_money(max(current_net_milli - int(stop_ledger["net_sell_total_milli"]), 0))
                    managed_open_risk += risk_to_stop
                elif str(management.get("status") or "") == "active":
                    managed_open_risk_complete = False
            except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                warnings.append(f"{ticker} 淨值/風控: {type(exc).__name__}: {exc}")
                open_unrealized_complete = False
                if str(management.get("status") or "") == "active":
                    managed_open_risk_complete = False
        elif qty > 0:
            net_liquidation_complete = False
            open_unrealized_complete = False
            if str(management.get("status") or "") == "active":
                managed_open_risk_complete = False

        if qty > 0 and net_liquidation is None:
            net_liquidation_complete = False

        initial_cost_milli = int(broker.get("initial_cost_basis_milli") or remaining_cost_milli)
        remaining_gross_milli = broker.get("remaining_gross_buy_milli")
        average_price_basis_milli = remaining_cost_milli if remaining_gross_milli is None else int(remaining_gross_milli)
        total_pnl = None if unrealized is None else milli_to_money(realized_milli) + unrealized
        enriched_positions.append(
            {
                "ticker": ticker,
                "source": record.get("source"),
                "qty": qty,
                "entry_date": broker.get("entry_date"),
                "average_cost": None if qty <= 0 else milli_to_money(average_price_basis_milli) / qty,
                "holding_cost": milli_to_money(remaining_cost_milli),
                "remaining_cost_basis": milli_to_money(remaining_cost_milli),
                "initial_cost_basis": milli_to_money(initial_cost_milli),
                "realized_pnl": milli_to_money(realized_milli),
                "current_price": current_price,
                "market_value": market_value,
                "net_liquidation_value": net_liquidation,
                "unrealized_pnl": unrealized,
                "holding_return_pct": None if unrealized is None else _safe_pct(unrealized, milli_to_money(remaining_cost_milli)),
                "total_pnl": total_pnl,
                "return_pct": None if total_pnl is None else _safe_pct(total_pnl, milli_to_money(initial_cost_milli)),
                "management_status": management.get("status"),
                "effective_stop": effective_stop,
                "risk_to_stop": risk_to_stop,
                "risk_status": "managed" if effective_stop is not None else "unmanaged_or_no_stop",
            }
        )

    cash = None if state.get("cash_milli") is None else milli_to_money(int(state["cash_milli"]))
    displayed_holdings_market_value = holdings_market_value if market_value_complete else None
    displayed_net_liquidation = holdings_net_liquidation if net_liquidation_complete else None
    equity = None if cash is None or displayed_net_liquidation is None else cash + displayed_net_liquidation
    single_position_risk_budget = None
    fixed_risk = None
    if params is not None and equity is not None:
        fixed_risk = float(params.fixed_risk)
        single_position_risk_budget = milli_to_money(calc_risk_budget_milli(equity, fixed_risk))

    buy_details, sell_details = _build_transaction_details(state)
    closed_trades = _build_closed_round_trips(state)
    open_perf_rows = [
        {
            "ticker": row["ticker"],
            "cost_basis": row["holding_cost"],
            "pnl": row["unrealized_pnl"],
        }
        for row in enriched_positions
        if row.get("unrealized_pnl") is not None
    ]
    sold_perf_rows = [
        {"ticker": row["ticker"], "cost_basis": row["offset_holding_cost"], "pnl": row["pnl"]}
        for row in sell_details
    ]
    performance = [
        _performance_row("持有中", open_perf_rows, closed_only=False),
        _performance_row("已賣出", sold_perf_rows, closed_only=True),
        _performance_row("持有+賣出", [*open_perf_rows, *sold_perf_rows], closed_only=False),
    ]

    return {
        "schema_version": ACCOUNT_DASHBOARD_SCHEMA_VERSION,
        "role": ACCOUNT_DASHBOARD_ROLE,
        "generated_at": generated_at,
        "initialized": True,
        "source_state_path": project_relative_display_path(state_path, project_root=root),
        "source_account_revision": int(state["revision"]),
        "source_params_path": params_path,
        "market_date": market_date,
        "fixed_risk": fixed_risk,
        "positions": enriched_positions,
        "buy_details": buy_details,
        "sell_details": sell_details,
        "closed_trades": closed_trades,
        "performance": performance,
        "summary": {
            "cash": cash,
            "holdings_market_value": displayed_holdings_market_value,
            "holdings_net_liquidation": displayed_net_liquidation,
            "equity": equity,
            "realized_pnl_open_positions": open_realized_pnl,
            "unrealized_pnl": open_unrealized_pnl if open_unrealized_complete else None,
            "managed_open_risk": managed_open_risk if managed_open_risk_complete else None,
            "single_position_risk_budget": single_position_risk_budget,
            "position_count": len(enriched_positions),
            "closed_trade_count": len(closed_trades),
        },
        "warnings": warnings,
    }


def publish_trading_account_dashboard_snapshot(project_root, snapshot: dict[str, Any] | None = None) -> dict[str, str]:
    root = Path(project_root).resolve()
    payload = dict(snapshot or build_trading_account_dashboard_read_model(root))
    output_dir = Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="account"))
    output_dir.mkdir(parents=True, exist_ok=True)
    account_path = output_dir / ACCOUNT_DASHBOARD_FILENAME
    performance_path = output_dir / ACCOUNT_PERFORMANCE_FILENAME
    atomic_write_json(account_path, payload)
    atomic_write_json(
        performance_path,
        {
            "schema_version": ACCOUNT_DASHBOARD_SCHEMA_VERSION,
            "role": "derived_read_only_trading_account_performance",
            "generated_at": payload.get("generated_at"),
            "source_state_path": payload.get("source_state_path"),
            "source_account_revision": payload.get("source_account_revision"),
            "market_date": payload.get("market_date"),
            "performance": list(payload.get("performance") or []),
            "buy_details": list(payload.get("buy_details") or []),
            "sell_details": list(payload.get("sell_details") or []),
            "closed_trades": list(payload.get("closed_trades") or []),
        },
    )
    return {
        "account_snapshot_path": project_relative_display_path(account_path, project_root=root),
        "performance_summary_path": project_relative_display_path(performance_path, project_root=root),
    }


__all__ = [
    "ACCOUNT_DASHBOARD_SCHEMA_VERSION",
    "ACCOUNT_DASHBOARD_ROLE",
    "build_trading_account_dashboard_read_model",
    "publish_trading_account_dashboard_snapshot",
]
