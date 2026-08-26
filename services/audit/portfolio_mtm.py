"""Canonical exact peak→trough MTM portfolio attribution shared by reusable and historical Audits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.data_utils import discover_unique_csv_map, sanitize_ohlcv_dataframe
from core.exact_accounting import (
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    coerce_money_like_to_milli,
    milli_to_money,
)
from core.price_utils import adjust_long_sell_fill_price
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from services.audit.mfe_safety_truth import AuditBlockedError, attach_quadrants, normalize_date, normalize_ticker


def _finite(value: Any) -> float | None:
    try:
        number=float(value)
    except (TypeError,ValueError):
        return None
    return number if math.isfinite(number) else None

def _load_equity(pair_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = pair_dir / "score_ranking_equity.csv"
    if not path.is_file():
        raise AuditBlockedError(f"缺少既有equity sidecar: {path.name}")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"Date", "Equity", "Exposure_Pct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditBlockedError(f"equity sidecar缺少欄位: {missing}")
    frame["Date"] = frame["Date"].map(normalize_date)
    frame["Equity"] = pd.to_numeric(frame["Equity"], errors="coerce")
    frame["Exposure_Pct"] = pd.to_numeric(frame["Exposure_Pct"], errors="coerce")
    if frame[["Date", "Equity"]].dropna().empty:
        raise AuditBlockedError("equity sidecar沒有有效Date/Equity")
    return frame, path


def _drawdown_episodes(equity: pd.DataFrame) -> list[dict[str, Any]]:
    table = pd.DataFrame(equity).copy().reset_index(drop=True)
    table["Equity"] = pd.to_numeric(table["Equity"], errors="coerce")
    table = table.loc[table["Equity"].notna() & (table["Equity"] > 0)].copy().reset_index(drop=True)
    if table.empty:
        return []
    peak = table["Equity"].cummax()
    dd = (peak - table["Equity"]) / peak * 100.0
    episodes: list[dict[str, Any]] = []
    start_idx: int | None = None
    peak_idx = 0
    for idx, value in enumerate(dd):
        if value <= 1e-12:
            peak_idx = idx
            if start_idx is not None:
                segment = dd.iloc[start_idx:idx]
                trough_idx = int(segment.idxmax())
                episodes.append({
                    "peak_idx": int(start_idx - 1 if start_idx > 0 else peak_idx),
                    "trough_idx": trough_idx,
                    "recovery_idx": idx,
                    "max_drawdown_pct": float(dd.iloc[trough_idx]),
                })
                start_idx = None
        elif start_idx is None:
            start_idx = idx
    if start_idx is not None:
        segment = dd.iloc[start_idx:]
        trough_idx = int(segment.idxmax())
        episodes.append({
            "peak_idx": int(max(0, start_idx - 1)),
            "trough_idx": trough_idx,
            "recovery_idx": None,
            "max_drawdown_pct": float(dd.iloc[trough_idx]),
        })
    for episode in episodes:
        episode["peak_date"] = str(table.iloc[episode["peak_idx"]]["Date"])
        episode["trough_date"] = str(table.iloc[episode["trough_idx"]]["Date"])
        episode["recovery_date"] = (
            None if episode["recovery_idx"] is None else str(table.iloc[episode["recovery_idx"]]["Date"])
        )
        episode["peak_equity"] = float(table.iloc[episode["peak_idx"]]["Equity"])
        episode["trough_equity"] = float(table.iloc[episode["trough_idx"]]["Equity"])
    return sorted(episodes, key=lambda row: row["max_drawdown_pct"], reverse=True)


_ACCOUNTING_FIELDS = ("buy_fee", "sell_fee", "tax_rate", "min_fee")
_QUADRANT_FIELDS = {
    "high_mfe_high_safety_pct": "hmhs",
    "high_mfe_low_safety_pct": "hmls",
    "low_mfe_high_safety_pct": "lmhs",
    "low_mfe_low_safety_pct": "lmls",
}


def read_pair_accounting_params(pair_dir: Path) -> SimpleNamespace:
    path = Path(pair_dir) / "strategy_comparison.json"
    if not path.is_file():
        raise AuditBlockedError(f"缺少既有Strategy Compare pair metadata: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditBlockedError(f"無法讀取pair accounting metadata: {exc}") from exc
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload, Mapping) else {}
    active = metadata.get("score_ranking_params")
    if not isinstance(active, Mapping):
        raise AuditBlockedError("Strategy Compare pair metadata缺少score_ranking_params")

    candidates: list[Mapping[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if all(field in node for field in _ACCOUNTING_FIELDS):
                candidates.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(active)
    if not candidates:
        raise AuditBlockedError("score_ranking_params找不到封存fee/tax accounting contract")
    policies: dict[tuple[float, float, float, float], Mapping[str, Any]] = {}
    for candidate in candidates:
        try:
            key = tuple(float(candidate[field]) for field in _ACCOUNTING_FIELDS)
        except (TypeError, ValueError) as exc:
            raise AuditBlockedError("score_ranking_params fee/tax contract格式無效") from exc
        policies.setdefault(key, candidate)
    if len(policies) != 1:
        raise AuditBlockedError(
            "Strategy Compare pair存在多組fee/tax accounting contract，無法做唯一MTM reconciliation"
        )
    (buy_fee, sell_fee, tax_rate, min_fee), sample = next(iter(policies.items()))
    return SimpleNamespace(
        buy_fee=buy_fee,
        sell_fee=sell_fee,
        tax_rate=tax_rate,
        min_fee=min_fee,
        fixed_risk=float(sample.get("fixed_risk", 0.01)),
    )


def _positive_int(value: Any, *, label: str) -> int:
    number = _finite(value)
    if number is None or number <= 0 or abs(number - round(number)) > 1e-9:
        raise AuditBlockedError(f"trade sidecar {label}不是正整數: {value}")
    return int(round(number))


def _build_trade_accounts(
    history: pd.DataFrame,
    *,
    pair_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    frame = pd.DataFrame(history).copy().reset_index(drop=True)
    round_trips = reconstruct_round_trips(frame, scenario="score_ranking")
    if round_trips.empty:
        return [], round_trips
    required = {"Date", "Ticker", "Type", "成交價", "股數"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditBlockedError(f"trade sidecar缺少MTM accounting欄位: {missing}")
    params = read_pair_accounting_params(pair_dir)
    round_trip_by_key = {
        str(row["match_key"]): row
        for row in round_trips.to_dict("records")
    }
    occurrence: dict[tuple[str, str, str], int] = {}
    open_accounts: dict[str, dict[str, Any]] = {}
    accounts: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        ticker = normalize_ticker(row.get("Ticker"))
        type_text = str(row.get("Type") or "").strip()
        trade_date = normalize_date(row.get("Date"))
        if not ticker or not type_text or not trade_date:
            continue
        if type_text.startswith("買進 ("):
            if ticker in open_accounts:
                raise AuditBlockedError(
                    f"trade sidecar同ticker未結算即再買進: ticker={ticker}, row={row_index}"
                )
            entry_type = str(row.get("進場類型") or "normal").strip() or "normal"
            key_base = (ticker, trade_date, entry_type)
            match_occurrence = occurrence.get(key_base, 0) + 1
            occurrence[key_base] = match_occurrence
            match_key = f"{ticker}|{trade_date}|{entry_type}|{match_occurrence}"
            round_trip = round_trip_by_key.get(match_key)
            if round_trip is None:
                raise AuditBlockedError(f"trade sidecar無法對上canonical match_key: {match_key}")
            qty = _positive_int(row.get("股數"), label="買進股數")
            entry_price = _finite(row.get("成交價"))
            if entry_price is None or entry_price <= 0:
                raise AuditBlockedError(f"trade sidecar買進成交價無效: {match_key}")
            buy_ledger = build_buy_ledger_from_price(entry_price, qty, params)
            computed_cost_milli = int(buy_ledger["net_buy_total_milli"])
            sidecar_cost = _finite(row.get("投入總金額")) if "投入總金額" in frame.columns else None
            if sidecar_cost is not None and sidecar_cost > 0:
                sidecar_cost_milli = coerce_money_like_to_milli(sidecar_cost)
                if abs(sidecar_cost_milli - computed_cost_milli) > 1:
                    raise AuditBlockedError(
                        "trade sidecar投入總金額與封存accounting contract不一致: "
                        f"{match_key} sidecar={milli_to_money(sidecar_cost_milli):.3f} "
                        f"recomputed={milli_to_money(computed_cost_milli):.3f}"
                    )
                entry_cost_milli = sidecar_cost_milli
            else:
                entry_cost_milli = computed_cost_milli
            account = {
                "match_key": match_key,
                "ticker": ticker,
                "entry_date": trade_date,
                "exit_date": normalize_date(round_trip.get("exit_date")),
                "initial_qty": qty,
                "entry_cost_milli": int(entry_cost_milli),
                "sell_events": [],
                "expected_pnl": _finite(round_trip.get("pnl")),
            }
            accounts.append(account)
            open_accounts[ticker] = account
            continue

        account = open_accounts.get(ticker)
        if account is None:
            continue
        exec_price = _finite(row.get("成交價"))
        qty_value = _finite(row.get("股數"))
        if exec_price is None or exec_price <= 0 or qty_value is None or qty_value <= 0:
            continue
        sell_qty = _positive_int(qty_value, label="賣出股數")
        sell_ledger = build_sell_ledger_from_price(
            exec_price,
            sell_qty,
            params,
            ticker=ticker,
            trade_date=trade_date,
        )
        account["sell_events"].append({
            "date": trade_date,
            "qty": sell_qty,
            "net_sell_total_milli": int(sell_ledger["net_sell_total_milli"]),
        })
        sold_qty = sum(int(event["qty"]) for event in account["sell_events"])
        if sold_qty > int(account["initial_qty"]):
            raise AuditBlockedError(f"trade sidecar賣出股數超過持倉: {account['match_key']}")
        if sold_qty == int(account["initial_qty"]):
            if trade_date != str(account["exit_date"]):
                raise AuditBlockedError(
                    f"trade sidecar cashflow結算日與canonical round-trip不一致: {account['match_key']}"
                )
            final_pnl_milli = -int(account["entry_cost_milli"]) + sum(
                int(event["net_sell_total_milli"]) for event in account["sell_events"]
            )
            expected_pnl = account.get("expected_pnl")
            if expected_pnl is not None:
                expected_milli = coerce_money_like_to_milli(expected_pnl)
                if abs(final_pnl_milli - expected_milli) > 11:
                    raise AuditBlockedError(
                        "trade sidecar final PnL與exact cashflow無法reconcile: "
                        f"{account['match_key']} sidecar={expected_pnl:.2f} "
                        f"cashflow={milli_to_money(final_pnl_milli):.3f}"
                    )
            del open_accounts[ticker]

    if open_accounts:
        raise AuditBlockedError(
            "trade sidecar結束後仍有未reconcile持倉: " + ", ".join(sorted(open_accounts)[:5])
        )
    return accounts, round_trips


def _load_close_series(
    ticker: str,
    *,
    market_csv_map: Mapping[str, str],
    market_close_cache: dict[str, pd.Series],
) -> pd.Series:
    if ticker in market_close_cache:
        return market_close_cache[ticker]
    csv_path = market_csv_map.get(ticker)
    if not csv_path:
        raise AuditBlockedError(f"canonical market data找不到ticker={ticker}")
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
        cleaned, _stats = sanitize_ohlcv_dataframe(raw, ticker, min_rows=1)
    except (OSError, KeyError, ValueError) as exc:
        raise AuditBlockedError(f"canonical market data無法載入ticker={ticker}: {exc}") from exc
    close = pd.to_numeric(cleaned["Close"], errors="coerce").dropna().sort_index()
    if close.empty:
        raise AuditBlockedError(f"canonical market data沒有有效Close: ticker={ticker}")
    market_close_cache[ticker] = close
    return close


def _close_on_or_before(series: pd.Series, date_text: str, *, ticker: str) -> float:
    ts = pd.Timestamp(date_text)
    position = int(series.index.searchsorted(ts, side="right") - 1)
    if position < 0:
        raise AuditBlockedError(f"canonical market data在{date_text}以前沒有Close: ticker={ticker}")
    value = _finite(series.iloc[position])
    if value is None or value <= 0:
        raise AuditBlockedError(f"canonical market data Close無效: ticker={ticker}, date={date_text}")
    return value


def _trade_account_value_at(
    account: Mapping[str, Any],
    date_text: str,
    *,
    params: SimpleNamespace,
    market_csv_map: Mapping[str, str],
    market_close_cache: dict[str, pd.Series],
) -> dict[str, Any]:
    entry_date = str(account["entry_date"])
    if date_text < entry_date:
        return {"value_milli": 0, "remaining_qty": 0, "mark_price": None}
    value_milli = -int(account["entry_cost_milli"])
    remaining_qty = int(account["initial_qty"])
    for event in account["sell_events"]:
        if str(event["date"]) <= date_text:
            value_milli += int(event["net_sell_total_milli"])
            remaining_qty -= int(event["qty"])
    if remaining_qty < 0:
        raise AuditBlockedError(f"trade account remaining qty < 0: {account['match_key']}")
    mark_price = None
    if remaining_qty > 0:
        ticker = str(account["ticker"])
        close_series = _load_close_series(
            ticker,
            market_csv_map=market_csv_map,
            market_close_cache=market_close_cache,
        )
        close_price = _close_on_or_before(close_series, date_text, ticker=ticker)
        mark_price = adjust_long_sell_fill_price(close_price, ticker=ticker)
        liquidation = build_sell_ledger_from_price(
            mark_price,
            remaining_qty,
            params,
            ticker=ticker,
            trade_date=date_text,
        )
        value_milli += int(liquidation["net_sell_total_milli"])
    return {
        "value_milli": int(value_milli),
        "remaining_qty": int(remaining_qty),
        "mark_price": mark_price,
    }


def build_drawdown_analysis(
    evidence: Mapping[str, Any],
    truth: pd.DataFrame,
    *,
    cutoff: float,
    top_n: int,
    market_data_dir: Path,
    market_csv_map: Mapping[str, str] | None = None,
    market_close_cache: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    pair_dir = Path(evidence["pair_dir"])
    equity, _ = _load_equity(pair_dir)
    history = pd.DataFrame(evidence["active_trades"]).copy()
    accounts, round_trips = _build_trade_accounts(history, pair_dir=pair_dir)
    path = pd.DataFrame(evidence["upside_realization"]).copy()
    if path.empty:
        raise AuditBlockedError("upside_realization sidecar為空")
    required_path = {"match_key", "score_event_date", "ticker"}
    missing_path = sorted(required_path - set(path.columns))
    if missing_path:
        raise AuditBlockedError(f"upside_realization缺少MTM attribution key: {missing_path}")
    path["ticker"] = path["ticker"].map(normalize_ticker)
    path["score_event_date"] = path["score_event_date"].map(normalize_date)
    truth_q = attach_quadrants(pd.DataFrame(truth), cutoff=cutoff)
    path = path.merge(
        truth_q,
        left_on=["ticker", "score_event_date"],
        right_on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    if bool(path["match_key"].duplicated(keep=False).any()):
        raise AuditBlockedError("upside_realization match_key不是唯一值")
    quadrant_by_key = {
        str(row["match_key"]): str(row.get("quadrant") or "unclassified")
        for row in path.to_dict("records")
    }

    market_data_dir = Path(market_data_dir)
    if market_csv_map is None:
        try:
            discovered, _issues = discover_unique_csv_map(str(market_data_dir))
        except (OSError, ValueError) as exc:
            raise AuditBlockedError(f"canonical market data inventory失敗: {exc}") from exc
        market_csv_map = {normalize_ticker(key): value for key, value in discovered.items()}
    if market_close_cache is None:
        market_close_cache = {}
    params = read_pair_accounting_params(pair_dir)

    all_episodes = _drawdown_episodes(equity)
    episodes = all_episodes[: int(top_n)]
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for rank, episode in enumerate(episodes, start=1):
        peak = str(episode["peak_date"])
        trough = str(episode["trough_date"])
        peak_equity_milli = coerce_money_like_to_milli(episode["peak_equity"])
        trough_equity_milli = coerce_money_like_to_milli(episode["trough_equity"])
        equity_change_milli = trough_equity_milli - peak_equity_milli
        entry_dates: list[str] = []
        episode_details: list[dict[str, Any]] = []
        total_contribution_milli = 0

        for account in accounts:
            peak_state = _trade_account_value_at(
                account,
                peak,
                params=params,
                market_csv_map=market_csv_map,
                market_close_cache=market_close_cache,
            )
            trough_state = _trade_account_value_at(
                account,
                trough,
                params=params,
                market_csv_map=market_csv_map,
                market_close_cache=market_close_cache,
            )
            contribution_milli = int(trough_state["value_milli"]) - int(peak_state["value_milli"])
            total_contribution_milli += contribution_milli
            entry_date = str(account["entry_date"])
            exit_date = str(account["exit_date"])
            held_at_peak = entry_date <= peak and exit_date > peak
            entered_during = peak < entry_date <= trough
            exited_during = peak < exit_date <= trough
            held_at_trough = entry_date <= trough and exit_date > trough
            relevant = held_at_peak or entered_during or exited_during or held_at_trough
            if not relevant:
                continue
            if entered_during:
                entry_dates.append(entry_date)
            quadrant = quadrant_by_key.get(str(account["match_key"]), "unclassified")
            row = {
                "episode_rank": rank,
                "peak_date": peak,
                "trough_date": trough,
                "match_key": str(account["match_key"]),
                "ticker": str(account["ticker"]),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "quadrant": quadrant,
                "held_at_peak": bool(held_at_peak),
                "entered_during_drawdown": bool(entered_during),
                "exited_during_drawdown": bool(exited_during),
                "held_at_trough": bool(held_at_trough),
                "peak_remaining_qty": int(peak_state["remaining_qty"]),
                "trough_remaining_qty": int(trough_state["remaining_qty"]),
                "peak_mark_price": peak_state["mark_price"],
                "trough_mark_price": trough_state["mark_price"],
                "peak_account_value": milli_to_money(int(peak_state["value_milli"])),
                "trough_account_value": milli_to_money(int(trough_state["value_milli"])),
                "mtm_contribution": milli_to_money(contribution_milli),
                "mtm_contribution_pct_peak_equity": (
                    float(contribution_milli / peak_equity_milli * 100.0)
                    if peak_equity_milli else None
                ),
                "negative_contributor": bool(contribution_milli < 0),
            }
            episode_details.append(row)
            detail_rows.append(row)

        reconciliation_delta_milli = total_contribution_milli - equity_change_milli
        tolerance_milli = max(10, len(episode_details) * 2)
        if abs(reconciliation_delta_milli) > tolerance_milli:
            raise AuditBlockedError(
                "peak→trough position MTM無法與canonical equity reconcile: "
                f"peak={peak}, trough={trough}, "
                f"positions={milli_to_money(total_contribution_milli):.3f}, "
                f"equity={milli_to_money(equity_change_milli):.3f}, "
                f"delta={milli_to_money(reconciliation_delta_milli):.3f}"
            )

        detail_frame = pd.DataFrame(episode_details)
        entry_counts = pd.Series(entry_dates, dtype=str).value_counts()
        duration_days = int((pd.Timestamp(trough) - pd.Timestamp(peak)).days)
        summary: dict[str, Any] = {
            "episode_rank": rank,
            **{key: value for key, value in episode.items() if not key.endswith("_idx")},
            "drawdown_to_trough_calendar_days": duration_days,
            "relevant_trade_count": int(len(detail_frame)),
            "peak_held_count": int(detail_frame.get("held_at_peak", pd.Series(dtype=bool)).sum()),
            "entered_during_drawdown_count": int(detail_frame.get("entered_during_drawdown", pd.Series(dtype=bool)).sum()),
            "exited_during_drawdown_count": int(detail_frame.get("exited_during_drawdown", pd.Series(dtype=bool)).sum()),
            "trough_held_count": int(detail_frame.get("held_at_trough", pd.Series(dtype=bool)).sum()),
            "max_same_day_entries": int(entry_counts.max()) if len(entry_counts) else 0,
            "negative_contributor_count": int(detail_frame.get("negative_contributor", pd.Series(dtype=bool)).sum()),
            "equity_change": milli_to_money(equity_change_milli),
            "position_mtm_contribution_sum": milli_to_money(total_contribution_milli),
            "position_mtm_contribution_pct_peak_equity": (
                float(total_contribution_milli / peak_equity_milli * 100.0)
                if peak_equity_milli else None
            ),
            "reconciliation_delta": milli_to_money(reconciliation_delta_milli),
        }
        for quadrant, prefix in _QUADRANT_FIELDS.items():
            if detail_frame.empty:
                group = detail_frame
            else:
                group = detail_frame.loc[detail_frame["quadrant"] == quadrant]
            contribution = (
                float(pd.to_numeric(group["mtm_contribution"], errors="coerce").sum())
                if "mtm_contribution" in group.columns else 0.0
            )
            summary[f"{prefix}_count"] = int(len(group))
            summary[f"{prefix}_mtm_contribution"] = float(contribution)
            summary[f"{prefix}_mtm_contribution_pct_peak_equity"] = (
                float(contribution / float(episode["peak_equity"]) * 100.0)
                if float(episode["peak_equity"]) else None
            )
        unclassified = (
            detail_frame.loc[~detail_frame["quadrant"].isin(_QUADRANT_FIELDS)]
            if not detail_frame.empty else detail_frame
        )
        unclassified_contribution = (
            float(pd.to_numeric(unclassified["mtm_contribution"], errors="coerce").sum())
            if "mtm_contribution" in unclassified.columns else 0.0
        )
        summary["unclassified_count"] = int(len(unclassified))
        summary["unclassified_mtm_contribution"] = float(unclassified_contribution)
        rows.append(summary)

    return {
        "max_drawdown_pct": (None if not episodes else float(episodes[0]["max_drawdown_pct"])),
        "episode_count": int(len(all_episodes)),
        "top_episodes": rows,
        "position_contributions": detail_rows,
        "round_trip_count": int(len(round_trips)),
        "market_data_dir": str(market_data_dir),
        "accounting_basis": "exact trade cashflows + EOD hypothetical net liquidation at canonical Close",
        "reconciliation_required": True,
    }



__all__=["build_drawdown_analysis", "read_pair_accounting_params"]
