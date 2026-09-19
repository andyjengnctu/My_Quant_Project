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
from core.market_data_dataset_readiness import has_current_market_data_dataset_validation
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from core.runtime_utils import get_taipei_now
from core.trading_policy import resolve_trading_selected_strategy_param_path
from services.trading.account_state import load_trading_account_state, resolve_trading_account_state_path
from core.trading_account_state import (
    ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT,
    effective_trading_account_events,
    project_trading_account_transactions,
    rebuild_trading_account_economics,
)
from services.trading.accounting_policy import (
    build_standalone_trading_accounting_params,
    overlay_trading_accounting_params,
)
from services.trading.market_data_consumer import load_trading_v2_consumer_state
from services.trading.market_data_dataset_state import load_market_data_dataset_state
from services.trading.market_data_v2_view import TradingMarketDataV2View
from services.trading.strategy_param_runtime import load_trading_strategy_param_runtime

ACCOUNT_DASHBOARD_SCHEMA_VERSION = 4
ACCOUNT_DASHBOARD_ROLE = "derived_read_only_trading_account_dashboard"
ACCOUNT_DASHBOARD_FILENAME = "account_snapshot.json"
ACCOUNT_PERFORMANCE_FILENAME = "performance_summary.json"


def _safe_pct(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator or 0) <= 0:
        return None
    return float(numerator) / float(denominator) * 100.0


def _resolve_account_valuation_market_date(
    project_root: Path,
    *,
    consumer_state: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Resolve the latest verified adjusted-price date for account MTM.

    Account valuation is a different consumer from Scanner eligibility.  Scanner
    stays on the latest fully-safe execution ``market_date`` while the account
    dashboard may mark current holdings as soon as the canonical adjusted-price
    dataset itself has a newer verified date.
    """

    scan_market_date = (
        None if consumer_state is None else str(consumer_state.get("market_date") or "").strip() or None
    )
    try:
        dataset_state = load_market_data_dataset_state(project_root, required=False)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, TypeError) as exc:
        return scan_market_date, f"Trading V2 dataset state: {type(exc).__name__}: {exc}"
    row = dict(((dataset_state or {}).get("datasets") or {}).get(FINMIND_ADJUSTED_PRICE_DATASET) or {})
    if not has_current_market_data_dataset_validation(row):
        return scan_market_date, None
    latest_price_date = str(row.get("latest_data_date") or "").strip() or None
    if latest_price_date is None:
        return scan_market_date, None
    if scan_market_date is None:
        return latest_price_date, None
    return max(scan_market_date, latest_price_date), None


def _current_close_by_ticker(project_root: Path, tickers: list[str], market_date: str | None) -> tuple[dict[str, float], dict[str, str]]:
    if not tickers or not market_date:
        return {}, {}
    view = TradingMarketDataV2View.open(project_root)
    prices: dict[str, float] = {}
    errors: dict[str, str] = {}
    requested = tuple(dict.fromkeys(str(ticker).strip() for ticker in tickers if str(ticker).strip()))

    # Healthy-path batch read: full-market daily overlay fragments are verified
    # and read once instead of once per holding.  If any batch-level read fails,
    # fall back to the historical per-ticker path so error granularity and
    # fail-soft dashboard behavior stay unchanged.
    try:
        frame = view.read_dataset_frame_many_data_ids(
            FINMIND_ADJUSTED_PRICE_DATASET,
            data_ids=requested,
            columns=("date", "stock_id", "close"),
            end_date=market_date,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError, TypeError):
        frame = None

    if frame is not None:
        if frame.empty:
            return {}, {ticker: "Trading V2 無可用收盤價" for ticker in requested}
        work = frame.copy()
        work["stock_id"] = work["stock_id"].astype(str).str.strip()
        work["date"] = work["date"].astype(str)
        work = work.sort_values(["stock_id", "date"])
        for ticker in requested:
            ticker_rows = work.loc[work["stock_id"] == ticker]
            if ticker_rows.empty:
                errors[ticker] = "Trading V2 無可用收盤價"
                continue
            dated = ticker_rows.loc[ticker_rows["date"] <= str(market_date)]
            if dated.empty:
                errors[ticker] = "Trading V2 無 cutoff 內收盤價"
                continue
            try:
                close = float(dated.iloc[-1]["close"])
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                errors[ticker] = f"{type(exc).__name__}: {exc}"
                continue
            if close <= 0:
                errors[ticker] = "Trading V2 收盤價不合法"
                continue
            prices[ticker] = close
        return prices, errors

    for ticker in requested:
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


def _build_closed_round_trips(state: dict[str, Any], *, accounting_params=None) -> list[dict[str, Any]]:
    """Reconstruct closed position lifecycles from effective, replayed economics."""
    active: dict[str, dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    projected_trades = project_trading_account_transactions(state, accounting_params=accounting_params)
    trade_revisions = {int(row.get("revision") or -1) for row in projected_trades}
    timeline = []
    for event in effective_trading_account_events(state):
        mutation = str(event.get("mutation_type") or "")
        revision = int(event.get("revision") or 0)
        if mutation in {"manual_buy_fill", "confirm_strategy_buy_fill", "confirm_strategy_buy_fill_increment", "confirm_sell_fill"}:
            continue
        timeline.append((revision, revision, event))
    for row in projected_trades:
        details = dict(row.get("details") or {})
        revision = int(row.get("revision") or 0)
        logical_revision = int(details.get("replacement_for_revision") or revision)
        timeline.append((logical_revision, revision, row))
    timeline.sort(key=lambda item: (item[0], item[1]))
    for _logical_revision, revision, event in timeline:
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
        elif mutation == "correct_position_broker_truth":
            position = dict(details.get("position") or {})
            broker = dict(position.get("broker") or {})
            if ticker in active and broker:
                active[ticker]["entry_date"] = broker.get("entry_date")
                active[ticker]["cost_basis_milli"] = int(broker.get("initial_cost_basis_milli") or 0)
                active[ticker]["source"] = str(position.get("source") or active[ticker].get("source") or "unknown")
        elif mutation == ACCOUNT_MUTATION_ACTIVATE_MANUAL_MANAGEMENT and ticker in active:
            position_after = dict(details.get("position_after") or {})
            position_state = dict((position_after.get("strategy_management") or {}).get("position_state") or {})
            active[ticker]["source"] = "manual_managed"
            active[ticker]["initial_risk_total_milli"] = int(
                position_state.get("initial_risk_total_milli") or 0
            )
        elif mutation in {"remove_manual_position", "remove_position_broker_truth"}:
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
        elif mutation in {"confirm_strategy_buy_fill", "manual_managed_buy_fill"}:
            position_after = dict(details.get("position_after") or {})
            position_state = dict((position_after.get("strategy_management") or {}).get("position_state") or {})
            active[ticker] = {
                "ticker": ticker,
                "source": "manual_managed" if mutation == "manual_managed_buy_fill" else "strategy_fill",
                "entry_date": details.get("trade_date"),
                "cost_basis_milli": int(details.get("net_buy_total_milli") or 0),
                "initial_risk_total_milli": int(position_state.get("initial_risk_total_milli") or 0),
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
                        "initial_risk": milli_to_money(int(lifecycle.get("initial_risk_total_milli") or 0)),
                        "r_mult": (
                            None
                            if int(lifecycle.get("initial_risk_total_milli") or 0) <= 0
                            else float(pnl_milli) / float(int(lifecycle.get("initial_risk_total_milli") or 0))
                        ),
                        "sell_count": int(lifecycle.get("sell_count") or 0),
                    }
                )
                active.pop(ticker, None)
    return closed


def _build_transaction_details(state: dict[str, Any], *, accounting_params=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buys: list[dict[str, Any]] = []
    sells: list[dict[str, Any]] = []
    events = project_trading_account_transactions(state, accounting_params=accounting_params)
    trade_mutations = {"manual_buy_fill", "manual_managed_buy_fill", "confirm_strategy_buy_fill", "confirm_strategy_buy_fill_increment", "confirm_sell_fill"}
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
        if mutation in {"manual_buy_fill", "manual_managed_buy_fill", "confirm_strategy_buy_fill", "confirm_strategy_buy_fill_increment"}:
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
            manual_unmanaged = mutation == "manual_buy_fill"
            manual_managed = mutation == "manual_managed_buy_fill"
            source = "手動成交" if manual_unmanaged else ("手動管理" if manual_managed else "策略成交")
            active_source[ticker] = "manual_adopted" if manual_unmanaged else ("manual_managed" if manual_managed else "strategy_fill")
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
                # AI: Acquisition source and execution reason are independent.
                # A manual SELL cannot relabel a strategy-origin holding.
                "source": source_before or None,
                "execution_reason": details.get("event"),
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


def _actual_outcome_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Derive empirical Trading performance from actual net PnL only.

    ``1 R`` is the mean absolute loss among the observed losing samples in the
    requested scope.  Therefore ``expected_value_r`` answers: average net PnL
    per sample / one unit of actually observed loss.  This is intentionally
    independent from Scanner/strategy initial-risk lineage.
    """
    if not rows:
        return {
            "win_rate_pct": None,
            "expected_value_r": None,
            "risk_reward_ratio": None,
            "actual_risk_unit": None,
        }

    pnl_values: list[float] = []
    for row in rows:
        value = row.get("pnl")
        if value is None:
            return {
                "win_rate_pct": None,
                "expected_value_r": None,
                "risk_reward_ratio": None,
                "actual_risk_unit": None,
            }
        pnl_values.append(float(value))

    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    win_rate = len(wins) / len(pnl_values) * 100.0
    avg_win_amount = sum(wins) / len(wins) if wins else None
    avg_loss_amount = abs(sum(losses) / len(losses)) if losses else None

    payoff = None
    if avg_win_amount is not None and avg_loss_amount is not None and avg_loss_amount > 0:
        payoff = avg_win_amount / avg_loss_amount

    # No observed loss means there is no empirical risk unit yet, so EV(R) is
    # mathematically undefined even when every observed sample is profitable.
    expected_value_r = None
    if avg_loss_amount is not None and avg_loss_amount > 0:
        expected_value_r = (sum(pnl_values) / len(pnl_values)) / avg_loss_amount

    return {
        "win_rate_pct": win_rate,
        "expected_value_r": expected_value_r,
        "risk_reward_ratio": payoff,
        "actual_risk_unit": avg_loss_amount,
    }



def _performance_summary_row(
    label: str,
    rows: list[dict[str, Any]],
    *,
    value: float | None,
    pnl: float | None,
    stock_tickers: set[str],
    metric_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cost = sum(float(row.get("cost_basis") or 0) for row in rows)
    metrics = _actual_outcome_metrics(list(metric_rows or []))
    return {
        "scope": label,
        "stock_count": len(stock_tickers),
        "value": value,
        "cost": cost,
        "pnl": pnl,
        "return_pct": None if cost <= 0 or pnl is None else float(pnl) / cost * 100.0,
        **metrics,
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
            "scan_market_date": None,
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
    scan_market_date = None if consumer_state is None else str(consumer_state.get("market_date") or "").strip() or None
    market_date, valuation_date_warning = _resolve_account_valuation_market_date(
        root, consumer_state=consumer_state
    )
    if valuation_date_warning:
        warnings.append(valuation_date_warning)

    params, params_path, params_error = _load_primary_params(root)
    if params_error:
        warnings.append(params_error)
    accounting_params = (
        overlay_trading_accounting_params(params)
        if params is not None
        else build_standalone_trading_accounting_params()
    )
    # Read views always consume current broker-accounting semantics, including
    # migration of legacy fee/tax rounding, without rewriting immutable events.
    state = rebuild_trading_account_economics(state, accounting_params=accounting_params)

    open_records = dict(state.get("positions") or {})
    tickers = sorted(open_records)
    prices, price_errors = _current_close_by_ticker(root, tickers, market_date)
    for ticker, error in sorted(price_errors.items()):
        warnings.append(f"{ticker} 市價: {error}")

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
                "sell_signal": management.get("sell_signal"),
                "sell_signal_date": management.get("sell_signal_date"),
                "effective_stop": effective_stop,
                "trailing_stop": (
                    None if position_state is None or int(position_state.get("trailing_stop_milli") or 0) <= 0
                    else milli_to_money(int(position_state.get("trailing_stop_milli") or 0))
                ),
                "target_price": (
                    None if position_state is None or int(position_state.get("tp_half_milli") or 0) <= 0
                    else milli_to_money(int(position_state.get("tp_half_milli") or 0))
                ),
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

    buy_details, sell_details = _build_transaction_details(state, accounting_params=accounting_params)
    closed_trades = _build_closed_round_trips(state, accounting_params=accounting_params)
    open_perf_rows = [
        {
            "ticker": row["ticker"],
            "value": row.get("net_liquidation_value"),
            "cost_basis": row["holding_cost"],
            "pnl": row.get("unrealized_pnl"),
        }
        for row in enriched_positions
    ]
    closed_perf_rows = [
        {
            "ticker": row["ticker"],
            "value": row.get("net_sell_total"),
            "cost_basis": row["cost_basis"],
            "pnl": row["pnl"],
            "r_mult": row.get("r_mult"),
        }
        for row in closed_trades
        if float(row.get("cost_basis") or 0) > 0
    ]

    open_value = None if not net_liquidation_complete else sum(float(row.get("value") or 0) for row in open_perf_rows)
    open_pnl = None if not open_unrealized_complete else sum(float(row.get("pnl") or 0) for row in open_perf_rows)
    closed_value = sum(float(row.get("value") or 0) for row in closed_perf_rows)
    closed_pnl = sum(float(row.get("pnl") or 0) for row in closed_perf_rows)
    open_tickers = {str(row.get("ticker") or "") for row in open_perf_rows if row.get("ticker")}
    closed_tickers = {str(row.get("ticker") or "") for row in closed_perf_rows if row.get("ticker")}
    combined_pnl = None if open_pnl is None else float(open_pnl) + closed_pnl
    combined_value = None if open_value is None else float(open_value) + closed_value
    performance = [
        _performance_summary_row(
            "庫存股", open_perf_rows, value=open_value, pnl=open_pnl,
            stock_tickers=open_tickers, metric_rows=open_perf_rows,
        ),
        _performance_summary_row(
            "平倉股", closed_perf_rows, value=closed_value, pnl=closed_pnl,
            stock_tickers=closed_tickers, metric_rows=closed_perf_rows,
        ),
        _performance_summary_row(
            "加總", [*open_perf_rows, *closed_perf_rows],
            value=combined_value, pnl=combined_pnl, stock_tickers=open_tickers | closed_tickers,
            metric_rows=[*open_perf_rows, *closed_perf_rows],
        ),
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
        "scan_market_date": scan_market_date,
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
            "scan_market_date": payload.get("scan_market_date"),
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
