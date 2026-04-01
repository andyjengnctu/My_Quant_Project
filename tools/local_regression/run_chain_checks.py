from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest_core import run_v16_backtest
from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.params_io import load_params_from_json
from core.portfolio_candidates import build_daily_candidates
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import build_normal_setup_index, build_trade_stats_index, calc_mark_to_market_equity, get_pit_stats_from_index, pack_prepared_stock_data, prep_stock_data_and_trades
from core.portfolio_ops import cleanup_extended_signals_for_day, execute_reserved_entries_for_day, settle_portfolio_positions, try_rotate_weakest_position
from core.portfolio_stats import find_sim_start_idx
from tools.debug.trade_log import run_debug_backtest
from core.runtime_utils import parse_no_arg_cli, run_cli_entrypoint
from tools.local_regression.common import PROJECT_ROOT, ensure_reduced_dataset, load_manifest, resolve_run_dir, write_csv, write_json, write_text


def load_market_context(params) -> Dict[str, Any]:
    data_dir = PROJECT_ROOT / "data" / "tw_stock_data_vip_reduced"
    csv_inputs, duplicate_issues = discover_unique_csv_inputs(str(data_dir))
    min_rows = get_required_min_rows(params)
    all_dfs_fast: Dict[str, Any] = {}
    all_trade_logs: Dict[str, Any] = {}
    prepared_frames: Dict[str, pd.DataFrame] = {}
    sanitize_stats_map: Dict[str, Dict[str, int]] = {}
    master_dates = set()
    skipped = []

    for ticker, file_path in csv_inputs:
        raw_df = pd.read_csv(file_path)
        if len(raw_df) < min_rows:
            skipped.append({"ticker": ticker, "reason": f"原始資料列數不足 {len(raw_df)}"})
            continue
        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows)
        prep_df, logs = prep_stock_data_and_trades(df, params)
        all_dfs_fast[ticker] = pack_prepared_stock_data(prep_df)
        all_trade_logs[ticker] = logs
        prepared_frames[ticker] = prep_df
        sanitize_stats_map[ticker] = sanitize_stats
        master_dates.update(prep_df.index)

    discovered_tickers = sorted(prepared_frames.keys())
    return {
        "data_dir": str(data_dir),
        "csv_count": len(csv_inputs),
        "duplicate_issues": duplicate_issues,
        "skipped": skipped,
        "discovered_tickers": discovered_tickers,
        "all_dfs_fast": all_dfs_fast,
        "all_trade_logs": all_trade_logs,
        "prepared_frames": prepared_frames,
        "sanitize_stats_map": sanitize_stats_map,
        "sorted_dates": sorted(master_dates),
    }


def build_target_replay_counts(*, tickers: List[str], all_dfs_fast, all_trade_logs, sorted_dates, params, start_year: int, max_positions: int, enable_rotation: bool):
    pit_stats_index = {ticker: build_trade_stats_index(logs) for ticker, logs in all_trade_logs.items()}
    normal_setup_index = build_normal_setup_index(all_dfs_fast)
    start_idx = find_sim_start_idx(sorted_dates, start_year)

    initial_capital = params.initial_capital
    cash = initial_capital
    current_equity = initial_capital
    portfolio = {}
    active_extended_signals = {}
    trade_history = []
    closed_trades_stats = []
    normal_trade_count = 0
    extended_trade_count = 0
    total_missed_buys = 0
    total_missed_sells = 0

    counts = {ticker: {"candidate_dates": [], "orderable_dates": [], "trade_rows": []} for ticker in tickers}
    for today in sorted_dates[start_idx:]:
        available_cash = cash
        sizing_equity = current_equity if getattr(params, "use_compounding", True) else initial_capital
        sold_today = set()

        candidates_today, orderable_candidates_today, _ = build_daily_candidates(
            normal_setup_index=normal_setup_index,
            active_extended_signals=active_extended_signals,
            portfolio=portfolio,
            sold_today=sold_today,
            all_dfs_fast=all_dfs_fast,
            pit_stats_index=pit_stats_index,
            today=today,
            sizing_equity=sizing_equity,
            params=params,
        )
        for ticker in tickers:
            if any(row["ticker"] == ticker for row in candidates_today):
                counts[ticker]["candidate_dates"].append(today)
            if any(row["ticker"] == ticker for row in orderable_candidates_today):
                counts[ticker]["orderable_dates"].append(today)

        before_trade_rows = len(trade_history)
        cash, normal_trade_count, extended_trade_count = try_rotate_weakest_position(
            portfolio=portfolio,
            orderable_candidates_today=orderable_candidates_today,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
            sold_today=sold_today,
            all_dfs_fast=all_dfs_fast,
            today=today,
            pit_stats_index=pit_stats_index,
            params=params,
            cash=cash,
            closed_trades_stats=closed_trades_stats,
            trade_history=trade_history,
            is_training=False,
            normal_trade_count=normal_trade_count,
            extended_trade_count=extended_trade_count,
        )
        cash, total_missed_sells, normal_trade_count, extended_trade_count = settle_portfolio_positions(
            portfolio=portfolio,
            sold_today=sold_today,
            all_dfs_fast=all_dfs_fast,
            today=today,
            params=params,
            cash=cash,
            closed_trades_stats=closed_trades_stats,
            trade_history=trade_history,
            is_training=False,
            total_missed_sells=total_missed_sells,
            normal_trade_count=normal_trade_count,
            extended_trade_count=extended_trade_count,
        )
        cash, total_missed_buys = execute_reserved_entries_for_day(
            portfolio=portfolio,
            active_extended_signals=active_extended_signals,
            orderable_candidates_today=orderable_candidates_today,
            sold_today=sold_today,
            all_dfs_fast=all_dfs_fast,
            today=today,
            params=params,
            cash=cash,
            available_cash=available_cash,
            max_positions=max_positions,
            trade_history=trade_history,
            is_training=False,
            total_missed_buys=total_missed_buys,
        )
        cleanup_extended_signals_for_day(active_extended_signals=active_extended_signals, portfolio=portfolio, all_dfs_fast=all_dfs_fast, today=today)
        current_equity = calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)

        new_rows = trade_history[before_trade_rows:]
        for row in new_rows:
            ticker = str(row.get("Ticker", "")).strip()
            if ticker in counts:
                counts[ticker]["trade_rows"].append(row)
    return counts


def summarize_ticker(*, ticker: str, params, start_year: int, df: pd.DataFrame, standalone_logs, sanitize_stats, replay_counts) -> Dict[str, Any]:
    single_stats, _ = run_v16_backtest(df.copy(), params, return_logs=True)
    debug_df = run_debug_backtest(df.copy(), ticker, params, export_excel=False, verbose=False)
    pit_index = build_trade_stats_index(standalone_logs)

    setup_rows = []
    for pos in range(1, len(df.index)):
        today = pd.Timestamp(df.index[pos])
        if today.year < start_year:
            continue
        if bool(df["is_setup"].iloc[pos - 1]):
            is_candidate, ev, win_rate, trade_count = get_pit_stats_from_index(pit_index, today, params)
            setup_rows.append({
                "date": today.strftime("%Y-%m-%d"),
                "pit_pass": bool(is_candidate),
                "ev": round(float(ev), 4),
                "win_rate": round(float(win_rate), 4),
                "trade_count": int(trade_count),
            })
    pass_rows = [row for row in setup_rows if row["pit_pass"]]
    trade_rows = replay_counts["trade_rows"]
    filled_count = sum(1 for row in trade_rows if str(row.get("Type", "")).startswith("買進"))
    missed_buy_count = sum(1 for row in trade_rows if "錯失買進" in str(row.get("Type", "")))

    if filled_count > 0 or missed_buy_count > 0:
        blocked_by = ""
    elif len(setup_rows) == 0:
        blocked_by = "no_setup"
    elif len(pass_rows) == 0:
        blocked_by = "history_filter"
    elif len(replay_counts["candidate_dates"]) == 0:
        blocked_by = "held_position_or_same_day_sell"
    elif len(replay_counts["orderable_dates"]) == 0:
        blocked_by = "risk_zero"
    else:
        blocked_by = "cash_slot_sort"

    return {
        "ticker": ticker,
        "single_trade_count": int(single_stats.get("trade_count", 0)),
        "single_missed_buys": int(single_stats.get("missed_buys", 0)),
        "setup_days": len(setup_rows),
        "pit_pass_days": len(pass_rows),
        "candidate_days": len(replay_counts["candidate_dates"]),
        "orderable_days": len(replay_counts["orderable_dates"]),
        "portfolio_trade_rows": len(trade_rows),
        "filled_count": filled_count,
        "portfolio_missed_buy_count": missed_buy_count,
        "debug_row_count": 0 if debug_df is None else len(debug_df),
        "blocked_by": blocked_by,
        "sanitize_dropped": int(sanitize_stats.get("dropped_row_count", 0)),
        "latest_setup_examples": setup_rows[-3:],
        "latest_trade_rows": [
            {"date": str(row.get("Date", "")), "type": str(row.get("Type", "")), "note": str(row.get("備註", ""))}
            for row in trade_rows[-3:]
        ],
    }


def _build_highlights(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    traded_rows = [row for row in summary_rows if row["filled_count"] > 0]
    missed_rows = [row for row in summary_rows if row["portfolio_missed_buy_count"] > 0]
    sanitize_rows = [row for row in summary_rows if row["sanitize_dropped"] > 0]
    blocked_counter = Counter(row["blocked_by"] or "filled_or_missed_buy" for row in summary_rows)
    return {
        "ticker_count": len(summary_rows),
        "traded_ticker_count": len(traded_rows),
        "missed_buy_ticker_count": len(missed_rows),
        "sanitize_issue_ticker_count": len(sanitize_rows),
        "traded_tickers": [row["ticker"] for row in traded_rows[:10]],
        "missed_buy_tickers": [row["ticker"] for row in missed_rows[:10]],
        "sanitize_issue_tickers": [row["ticker"] for row in sanitize_rows[:10]],
        "blocked_by_counts": dict(sorted(blocked_counter.items())),
    }


CHAIN_RERUN_COUNT = 2


def _build_summary_payload(*, manifest, dataset_info, context, summary_rows, portfolio_profile, df_equity, df_trades, failures, runtime_error="") -> Dict[str, Any]:
    duplicate_issues = list(context.get("duplicate_issues", []))
    skipped_rows = list(context.get("skipped", []))
    return {
        "status": "PASS" if not failures else "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": dataset_info,
        "runtime_error": runtime_error,
        "portfolio_snapshot": {
            "equity_rows": int(len(df_equity)),
            "trade_rows": int(len(df_trades)),
            "annual_return_pct": round(float(portfolio_profile.get("annual_return_pct", 0.0)), 4),
            "reserved_buy_fill_rate": round(float(portfolio_profile.get("reserved_buy_fill_rate", 0.0)), 4),
        },
        "ticker_count": len(context["discovered_tickers"]),
        "csv_count": int(context.get("csv_count", 0)),
        "duplicate_issue_count": len(duplicate_issues),
        "skipped_ticker_count": len(skipped_rows),
        "duplicate_issues": duplicate_issues,
        "skipped": skipped_rows,
        "detail_count": len(summary_rows),
        "highlights": _build_highlights(summary_rows),
        "failures": list(failures),
        "rows": summary_rows,
    }


def _compute_chain_summary(*, manifest, dataset_info, params, start_year, max_positions, enable_rotation, benchmark_ticker, write_outputs=False, run_dir=None):
    context = load_market_context(params)
    all_tickers = context["discovered_tickers"]
    portfolio_profile: Dict[str, Any] = {}
    portfolio_result = run_portfolio_timeline(
        context["all_dfs_fast"],
        context["all_trade_logs"],
        context["sorted_dates"],
        start_year,
        params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=context["all_dfs_fast"].get(benchmark_ticker),
        is_training=False,
        profile_stats=portfolio_profile,
        verbose=False,
    )
    df_equity, df_trades = portfolio_result[0], portfolio_result[1]
    replay_counts = build_target_replay_counts(
        tickers=all_tickers,
        all_dfs_fast=context["all_dfs_fast"],
        all_trade_logs=context["all_trade_logs"],
        sorted_dates=context["sorted_dates"],
        params=params,
        start_year=start_year,
        max_positions=max_positions,
        enable_rotation=enable_rotation,
    )

    detail_rows = []
    for ticker in all_tickers:
        row = summarize_ticker(
            ticker=ticker,
            params=params,
            start_year=start_year,
            df=context["prepared_frames"][ticker],
            standalone_logs=context["all_trade_logs"][ticker],
            sanitize_stats=context["sanitize_stats_map"][ticker],
            replay_counts=replay_counts[ticker],
        )
        detail_rows.append(row)

    summary_rows = sorted(detail_rows, key=lambda item: item["ticker"])
    if write_outputs:
        if run_dir is None:
            raise ValueError("write_outputs=True 時必須提供 run_dir")
        detail_dir = run_dir / "chain_details"
        detail_dir.mkdir(parents=True, exist_ok=True)
        for row in summary_rows:
            write_json(detail_dir / f"{row['ticker']}.json", row)
        fieldnames = ["ticker", "single_trade_count", "single_missed_buys", "setup_days", "pit_pass_days", "candidate_days", "orderable_days", "portfolio_trade_rows", "filled_count", "portfolio_missed_buy_count", "debug_row_count", "blocked_by", "sanitize_dropped"]
        write_csv(run_dir / "chain_summary.csv", summary_rows, fieldnames=fieldnames)

    failures = []
    duplicate_issues = list(context.get("duplicate_issues", []))
    skipped_rows = list(context.get("skipped", []))
    if duplicate_issues:
        failures.append(f"duplicate_csv_inputs={len(duplicate_issues)}")
    if skipped_rows:
        failures.append(f"skipped_tickers={len(skipped_rows)}")

    for row in summary_rows:
        if row["orderable_days"] > row["candidate_days"]:
            failures.append(f"{row['ticker']}: orderable_days > candidate_days")
        if row["filled_count"] + row["portfolio_missed_buy_count"] > row["portfolio_trade_rows"]:
            failures.append(f"{row['ticker']}: filled+missed > portfolio_trade_rows")

    summary = _build_summary_payload(
        manifest=manifest,
        dataset_info=dataset_info,
        context=context,
        summary_rows=summary_rows,
        portfolio_profile=portfolio_profile,
        df_equity=df_equity,
        df_trades=df_trades,
        failures=failures,
    )
    return summary


def _canonical_chain_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "portfolio_snapshot": summary.get("portfolio_snapshot", {}),
        "ticker_count": summary.get("ticker_count", 0),
        "csv_count": summary.get("csv_count", 0),
        "duplicate_issue_count": summary.get("duplicate_issue_count", 0),
        "skipped_ticker_count": summary.get("skipped_ticker_count", 0),
        "duplicate_issues": summary.get("duplicate_issues", []),
        "skipped": summary.get("skipped", []),
        "detail_count": summary.get("detail_count", 0),
        "highlights": summary.get("highlights", {}),
        "failures": summary.get("failures", []),
        "rows": summary.get("rows", []),
    }


def _payload_digest(payload: Dict[str, Any]) -> str:
    canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parsed = parse_no_arg_cli(argv, "tools/local_regression/run_chain_checks.py", description="執行 reduced chain checks；不接受額外參數。")
    if parsed["help"]:
        return 0

    manifest = load_manifest()
    run_dir = resolve_run_dir("chain_checks")
    dataset_info = ensure_reduced_dataset()

    try:
        params = load_params_from_json(PROJECT_ROOT / "models" / "best_params.json")
        start_year = int(manifest["portfolio_start_year"])
        max_positions = int(manifest["portfolio_max_positions"])
        enable_rotation = bool(manifest["portfolio_enable_rotation"])
        benchmark_ticker = str(manifest["benchmark_ticker"])

        primary_summary = _compute_chain_summary(
            manifest=manifest,
            dataset_info=dataset_info,
            params=params,
            start_year=start_year,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
            benchmark_ticker=benchmark_ticker,
            write_outputs=True,
            run_dir=run_dir,
        )

        rerun_summaries = []
        rerun_digests = []
        for rerun_index in range(2, CHAIN_RERUN_COUNT + 1):
            rerun_summary = _compute_chain_summary(
                manifest=manifest,
                dataset_info=dataset_info,
                params=params,
                start_year=start_year,
                max_positions=max_positions,
                enable_rotation=enable_rotation,
                benchmark_ticker=benchmark_ticker,
                write_outputs=False,
            )
            rerun_payload = _canonical_chain_payload(rerun_summary)
            rerun_summaries.append({
                "run_index": rerun_index,
                "status": rerun_summary["status"],
                "digest": _payload_digest(rerun_payload),
                "failure_count": len(rerun_summary.get("failures", [])),
            })
            rerun_digests.append(rerun_summaries[-1]["digest"])
            if rerun_summary["status"] != "PASS":
                primary_summary["failures"].append(f"rerun_{rerun_index}_status={rerun_summary['status']}")

        primary_payload = _canonical_chain_payload(primary_summary)
        primary_digest = _payload_digest(primary_payload)
        rerun_match = all(digest == primary_digest for digest in rerun_digests)
        if not rerun_match:
            primary_summary["failures"].append("chain_rerun_inconsistent")

        primary_summary["rerun_consistency"] = {
            "enabled": True,
            "run_count": CHAIN_RERUN_COUNT,
            "primary_digest": primary_digest,
            "all_match": rerun_match,
            "runs": rerun_summaries,
        }
        primary_summary["status"] = "PASS" if not primary_summary["failures"] else "FAIL"
        summary = primary_summary
    except Exception as exc:
        summary = {
            "status": "FAIL",
            "dataset": manifest["dataset"],
            "dataset_info": dataset_info,
            "runtime_error": f"{type(exc).__name__}: {exc}",
            "portfolio_snapshot": {},
            "ticker_count": 0,
            "csv_count": 0,
            "duplicate_issue_count": 0,
            "skipped_ticker_count": 0,
            "duplicate_issues": [],
            "skipped": [],
            "detail_count": 0,
            "highlights": {},
            "failures": [f"runtime_error={type(exc).__name__}"],
            "rows": [],
            "rerun_consistency": {
                "enabled": True,
                "run_count": CHAIN_RERUN_COUNT,
                "primary_digest": "",
                "all_match": False,
                "runs": [],
            },
        }

    write_json(run_dir / "chain_summary.json", summary)
    write_json(run_dir / "chain_checks_summary.json", summary)
    write_text(
        run_dir / "chain_console.log",
        json.dumps(
            {
                "status": summary["status"],
                "ticker_count": summary.get("ticker_count", 0),
                "failures": summary.get("failures", []),
                "runtime_error": summary.get("runtime_error", ""),
                "rerun_consistency": summary.get("rerun_consistency", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(json.dumps({"status": summary["status"], "ticker_count": summary.get("ticker_count", 0), "failures": summary.get("failures", []), "runtime_error": summary.get("runtime_error", "")}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
