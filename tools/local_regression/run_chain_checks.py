from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.params_io import load_params_from_json
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import build_trade_stats_index, get_pit_stats_from_index, pack_prepared_stock_data, prep_stock_data_and_trades
from tools.scanner.stock_processor import process_prepared_stock
from core.runtime_utils import PeakTracedMemoryTracker, parse_no_arg_cli, run_cli_entrypoint
from tools.local_regression.common import PROJECT_ROOT, ensure_reduced_dataset, load_manifest, resolve_run_dir, write_csv, write_json, write_text
from tools.local_regression.shared_prep_cache import load_shared_prep_cache_entry


def _determine_chain_worker_count(total_items: int) -> int:
    if total_items <= 1:
        return 1

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    if cpu_count <= 4:
        return min(total_items, 2)
    if cpu_count <= 8:
        return min(total_items, 4)
    return min(total_items, 6)


def _prepare_market_context_entry(task):
    ticker, file_path, min_rows, params = task
    raw_df = pd.read_csv(file_path)
    if len(raw_df) < min_rows:
        return {
            "ticker": ticker,
            "skipped": {"ticker": ticker, "reason": f"原始資料列數不足 {len(raw_df)}"},
        }

    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows)
    prep_df, logs, single_stats = prep_stock_data_and_trades(df, params, return_stats=True)
    return {
        "ticker": ticker,
        "fast_data": pack_prepared_stock_data(prep_df),
        "logs": logs,
        "prepared_frame": prep_df,
        "sanitize_stats": sanitize_stats,
        "single_stats": single_stats,
        "dates": tuple(prep_df.index),
    }


def _build_scanner_snapshot_entry(task):
    idx, ticker, prep_df, params, sanitize_stats = task
    result = process_prepared_stock(prep_df, ticker, params, sanitize_stats=sanitize_stats)
    return idx, ticker, result


def _summarize_ticker_entry(task):
    idx, kwargs = task
    return idx, summarize_ticker(**kwargs)


def _derive_debug_row_count(*, standalone_logs, replay_counts, single_stats) -> int:
    trade_rows = list(replay_counts.get("trade_rows", []))
    half_take_profit_rows = sum(1 for row in trade_rows if str(row.get("Type", "")).strip() == "半倉停利")
    missed_buy_rows = sum(1 for row in trade_rows if str(row.get("Type", "")).startswith("錯失買進"))
    missed_sell_rows = sum(1 for row in trade_rows if str(row.get("Type", "")).strip() == "錯失賣出")

    expected_missed_buys = int(single_stats.get("missed_buys", 0) or 0)
    expected_missed_sells = int(single_stats.get("missed_sells", 0) or 0)
    if expected_missed_buys > missed_buy_rows:
        missed_buy_rows = expected_missed_buys
    if expected_missed_sells > missed_sell_rows:
        missed_sell_rows = expected_missed_sells

    return int((2 * len(standalone_logs)) + half_take_profit_rows + missed_buy_rows + missed_sell_rows)


def _normalize_scanner_candidate_row(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": str(item.get("kind", "")),
        "ticker": str(item.get("ticker", "")),
        "proj_cost": None if item.get("proj_cost") is None else round(float(item.get("proj_cost")), 6),
        "expected_value": None if item.get("expected_value") is None else round(float(item.get("expected_value")), 6),
        "sort_value": None if item.get("sort_value") is None else round(float(item.get("sort_value")), 6),
        "text": str(item.get("text", "")),
    }


def _build_scanner_snapshot_from_context(context: Dict[str, Any], params) -> Dict[str, Any]:
    candidate_rows = []
    duplicate_issues = list(context.get("duplicate_issues", []))
    scanner_issue_lines = list(duplicate_issues)
    status_rows = []
    count_history_qualified = 0
    count_skipped_insufficient = 0
    count_sanitized_candidates = 0

    discovered_tickers = context["discovered_tickers"]
    worker_count = _determine_chain_worker_count(len(discovered_tickers))
    ordered_results = [None] * len(discovered_tickers)

    if worker_count <= 1:
        for idx, ticker in enumerate(discovered_tickers):
            prep_df = context["prepared_frames"][ticker]
            sanitize_stats = context["sanitize_stats_map"][ticker]
            ordered_results[idx] = (idx, ticker, process_prepared_stock(prep_df, ticker, params, sanitize_stats=sanitize_stats))
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="chain_scanner") as executor:
            future_to_idx = {}
            for idx, ticker in enumerate(discovered_tickers):
                future = executor.submit(
                    _build_scanner_snapshot_entry,
                    (idx, ticker, context["prepared_frames"][ticker], params, context["sanitize_stats_map"][ticker]),
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered_results[idx] = future.result()

    for idx in range(len(discovered_tickers)):
        _idx, ticker, result = ordered_results[idx]
        if result is None:
            status_rows.append({"ticker": str(ticker), "status": "not_candidate", "has_sanitize_issue": False})
            continue

        status, proj_cost, ev, sort_value, msg, result_ticker, sanitize_issue = result
        normalized_ticker = str(result_ticker or ticker)
        status_rows.append({
            "ticker": normalized_ticker,
            "status": str(status),
            "has_sanitize_issue": bool(sanitize_issue),
        })

        if status in ["buy", "extended", "candidate"]:
            count_history_qualified += 1
            if sanitize_issue is not None:
                count_sanitized_candidates += 1
                scanner_issue_lines.append(f"[清洗] {sanitize_issue}")
        elif status == "skip_insufficient":
            count_skipped_insufficient += 1

        if status in ["buy", "extended"]:
            candidate_rows.append(_normalize_scanner_candidate_row({
                "kind": status,
                "ticker": normalized_ticker,
                "proj_cost": proj_cost,
                "expected_value": ev,
                "sort_value": sort_value,
                "text": msg,
            }))

    candidate_rows.sort(key=lambda item: (item["sort_value"], item["ticker"]), reverse=True)
    return {
        "entry_script": "apps/vip_scanner.py",
        "count_scanned": len(discovered_tickers),
        "history_qualified_count": count_history_qualified,
        "skipped_insufficient_count": count_skipped_insufficient,
        "sanitized_candidate_count": count_sanitized_candidates,
        "candidate_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
        "status_rows": status_rows,
        "duplicate_issue_count": len(duplicate_issues),
        "scanner_issue_count": len(scanner_issue_lines),
        "scanner_issue_lines": scanner_issue_lines,
    }


def load_market_context(params) -> Dict[str, Any]:
    data_dir = PROJECT_ROOT / "data" / "tw_stock_data_vip_reduced"
    csv_inputs, duplicate_issues = discover_unique_csv_inputs(str(data_dir))
    min_rows = get_required_min_rows(params)
    cache_hits = 0
    cache_misses = 0
    all_dfs_fast: Dict[str, Any] = {}
    all_trade_logs: Dict[str, Any] = {}
    prepared_frames: Dict[str, pd.DataFrame] = {}
    sanitize_stats_map: Dict[str, Dict[str, int]] = {}
    standalone_stats_map: Dict[str, Dict[str, Any]] = {}
    master_dates = set()
    skipped = []

    worker_count = _determine_chain_worker_count(len(csv_inputs))
    ordered_entries = [None] * len(csv_inputs)

    cache_pending = []
    for idx, (ticker, file_path) in enumerate(csv_inputs):
        cache_entry = load_shared_prep_cache_entry(ticker)
        if isinstance(cache_entry, dict) and cache_entry.get("status") == "ready" and os.path.abspath(str(file_path)) == str(cache_entry.get("file_path", "")):
            cache_hits += 1
            ordered_entries[idx] = {
                "ticker": ticker,
                "fast_data": cache_entry["fast_data"],
                "logs": list(cache_entry["standalone_logs"]),
                "prepared_frame": cache_entry["prepared_df"].copy(),
                "sanitize_stats": dict(cache_entry["sanitize_stats"]),
                "single_stats": dict(cache_entry["single_stats"]),
                "dates": tuple(cache_entry["sorted_dates"]),
            }
            continue
        cache_misses += 1
        cache_pending.append((idx, ticker, file_path))

    if worker_count <= 1:
        for idx, ticker, file_path in cache_pending:
            ordered_entries[idx] = _prepare_market_context_entry((ticker, file_path, min_rows, params))
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="chain_prep") as executor:
            future_to_idx = {}
            for idx, ticker, file_path in cache_pending:
                future = executor.submit(_prepare_market_context_entry, (ticker, file_path, min_rows, params))
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered_entries[idx] = future.result()

    for entry in ordered_entries:
        if entry is None:
            continue
        ticker = entry["ticker"]
        skipped_row = entry.get("skipped")
        if skipped_row is not None:
            skipped.append(skipped_row)
            continue

        all_dfs_fast[ticker] = entry["fast_data"]
        all_trade_logs[ticker] = entry["logs"]
        prepared_frames[ticker] = entry["prepared_frame"]
        sanitize_stats_map[ticker] = entry["sanitize_stats"]
        standalone_stats_map[ticker] = entry["single_stats"]
        master_dates.update(entry["dates"])

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
        "standalone_stats_map": standalone_stats_map,
        "sorted_dates": sorted(master_dates),
        "shared_prep_cache_hits": cache_hits,
        "shared_prep_cache_misses": cache_misses,
    }




def summarize_ticker(*, ticker: str, params, start_year: int, df: pd.DataFrame, single_stats, standalone_logs, sanitize_stats, replay_counts, debug_row_count: int | None = None) -> Dict[str, Any]:
    if debug_row_count is None:
        resolved_debug_row_count = _derive_debug_row_count(
            standalone_logs=standalone_logs,
            replay_counts=replay_counts,
            single_stats=single_stats,
        )
    else:
        resolved_debug_row_count = int(debug_row_count)
    pit_index = build_trade_stats_index(standalone_logs)

    setup_rows = []
    for pos in range(1, len(df.index)):
        today = pd.Timestamp(df.index[pos])
        if today.year < start_year:
            continue
        if bool(df["is_setup"].iloc[pos - 1]):
            is_candidate, ev, win_rate, trade_count, _asset_growth_pct = get_pit_stats_from_index(pit_index, today, params)
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
        "debug_row_count": resolved_debug_row_count,
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


def _build_summary_payload(*, manifest, dataset_info, context, summary_rows, portfolio_profile, df_equity, df_trades, scanner_snapshot, failures, runtime_error="") -> Dict[str, Any]:
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
        "scanner_snapshot": scanner_snapshot,
        "ticker_count": len(context["discovered_tickers"]),
        "csv_count": int(context.get("csv_count", 0)),
        "duplicate_issue_count": len(duplicate_issues),
        "skipped_ticker_count": len(skipped_rows),
        "duplicate_issues": duplicate_issues,
        "skipped": skipped_rows,
        "detail_count": len(summary_rows),
        "shared_prep_cache_hits": int(context.get("shared_prep_cache_hits", 0)),
        "shared_prep_cache_misses": int(context.get("shared_prep_cache_misses", 0)),
        "highlights": _build_highlights(summary_rows),
        "failures": list(failures),
        "rows": summary_rows,
    }


def _compute_chain_summary(*, manifest, dataset_info, context, params, start_year, max_positions, enable_rotation, benchmark_ticker, write_outputs=False, run_dir=None, debug_row_count_cache=None):
    all_tickers = context["discovered_tickers"]
    portfolio_profile: Dict[str, Any] = {}
    replay_counts = {ticker: {"candidate_dates": [], "orderable_dates": [], "trade_rows": []} for ticker in all_tickers}
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
        replay_counts=replay_counts,
    )
    df_equity, df_trades = portfolio_result[0], portfolio_result[1]
    scanner_snapshot = _build_scanner_snapshot_from_context(context, params)

    summary_tasks = []
    for idx, ticker in enumerate(all_tickers):
        summary_tasks.append((idx, {
            "ticker": ticker,
            "params": params,
            "start_year": start_year,
            "df": context["prepared_frames"][ticker],
            "single_stats": context["standalone_stats_map"][ticker],
            "standalone_logs": context["all_trade_logs"][ticker],
            "sanitize_stats": context["sanitize_stats_map"][ticker],
            "replay_counts": replay_counts[ticker],
            "debug_row_count": None if debug_row_count_cache is None else debug_row_count_cache.get(ticker),
        }))

    detail_rows = [None] * len(summary_tasks)
    summary_worker_count = _determine_chain_worker_count(len(summary_tasks))
    if summary_worker_count <= 1:
        for idx, kwargs in summary_tasks:
            detail_rows[idx] = summarize_ticker(**kwargs)
    else:
        with ThreadPoolExecutor(max_workers=summary_worker_count, thread_name_prefix="chain_detail") as executor:
            future_to_idx = {executor.submit(_summarize_ticker_entry, task): task[0] for task in summary_tasks}
            for future in as_completed(future_to_idx):
                idx, row = future.result()
                detail_rows[idx] = row

    summary_rows = sorted((row for row in detail_rows if row is not None), key=lambda item: item["ticker"])
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
        scanner_snapshot=scanner_snapshot,
        failures=failures,
    )
    return summary


def _canonical_chain_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "portfolio_snapshot": summary.get("portfolio_snapshot", {}),
        "scanner_snapshot": summary.get("scanner_snapshot", {}),
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

    with PeakTracedMemoryTracker() as tracker:
        started = time.perf_counter()
        manifest = load_manifest()
        run_dir = resolve_run_dir("chain_checks")
        dataset_info = ensure_reduced_dataset()

        try:
            params = load_params_from_json(PROJECT_ROOT / "models" / "champion_params.json")
            start_year = int(manifest["portfolio_start_year"])
            max_positions = int(manifest["portfolio_max_positions"])
            enable_rotation = bool(manifest["portfolio_enable_rotation"])
            benchmark_ticker = str(manifest["benchmark_ticker"])
            context = load_market_context(params)

            primary_summary = _compute_chain_summary(
                manifest=manifest,
                dataset_info=dataset_info,
                context=context,
                params=params,
                start_year=start_year,
                max_positions=max_positions,
                enable_rotation=enable_rotation,
                benchmark_ticker=benchmark_ticker,
                write_outputs=True,
                run_dir=run_dir,
            )
            debug_row_count_cache = {row["ticker"]: int(row.get("debug_row_count", 0)) for row in primary_summary.get("rows", [])}

            rerun_summaries = []
            rerun_digests = []
            for rerun_index in range(2, CHAIN_RERUN_COUNT + 1):
                rerun_summary = _compute_chain_summary(
                    manifest=manifest,
                    dataset_info=dataset_info,
                    context=context,
                    params=params,
                    start_year=start_year,
                    max_positions=max_positions,
                    enable_rotation=enable_rotation,
                    benchmark_ticker=benchmark_ticker,
                    write_outputs=False,
                    debug_row_count_cache=debug_row_count_cache,
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
                "scanner_snapshot": {},
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

        summary["duration_sec"] = round(time.perf_counter() - started, 3)
        summary["peak_traced_memory_mb"] = tracker.snapshot_peak_mb()
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
                    "scanner_snapshot": {
                        "candidate_count": summary.get("scanner_snapshot", {}).get("candidate_count", 0),
                        "history_qualified_count": summary.get("scanner_snapshot", {}).get("history_qualified_count", 0),
                        "duplicate_issue_count": summary.get("scanner_snapshot", {}).get("duplicate_issue_count", 0),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        print(json.dumps({"status": summary["status"], "ticker_count": summary.get("ticker_count", 0), "failures": summary.get("failures", []), "runtime_error": summary.get("runtime_error", "")}, ensure_ascii=False))
        return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
