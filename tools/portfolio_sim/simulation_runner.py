import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.dataset_profiles import (
    build_missing_dataset_dir_message,
    infer_dataset_profile_key_from_data_dir,
)
from core.display import C_CYAN, C_GREEN, C_GRAY, C_YELLOW, C_RESET
from core.log_utils import format_exception_summary, write_issue_log
from core.params_io import params_to_json_dict
from core.walk_forward_policy import load_walk_forward_policy
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import merge_static_market_with_dynamic, pack_static_market_data, pack_prepared_stock_data, prep_optimizer_stock_data_bundle, prep_stock_data_and_trades
from tools.optimizer.raw_cache import load_all_raw_data
from tools.optimizer.trial_inputs import prepare_trial_inputs
from tools.optimizer.walk_forward import resolve_first_walk_forward_test_boundary
from .runtime_common import LOAD_PROGRESS_EVERY, OUTPUT_DIR, PROJECT_ROOT, ensure_runtime_dirs, is_insufficient_data_error

PORTFOLIO_DEFAULT_BENCHMARK_TICKER = "0050"
PORTFOLIO_PREP_CACHE_SCHEMA_VERSION = 1


def _collect_dataset_market_dates(data_dir: str):
    csv_inputs, _ = discover_unique_csv_inputs(data_dir)
    market_dates = set()
    for _, file_path in csv_inputs:
        try:
            date_series = pd.read_csv(file_path, usecols=["Date"])["Date"]
        except (ValueError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            continue
        parsed_dates = pd.to_datetime(date_series, errors="coerce")
        market_dates.update(ts.normalize().to_pydatetime() for ts in parsed_dates.dropna())
    return sorted(market_dates)


def resolve_default_portfolio_start_year(data_dir: str | None = None) -> int:
    policy = load_walk_forward_policy(PROJECT_ROOT)
    fallback_year = int(policy["train_start_year"])
    if not data_dir or not os.path.isdir(data_dir):
        return fallback_year

    sorted_dates = _collect_dataset_market_dates(data_dir)
    if not sorted_dates:
        return fallback_year

    first_test_boundary = resolve_first_walk_forward_test_boundary(
        sorted_dates,
        min_train_years=int(policy["min_train_years"]),
        train_start_year=int(policy["train_start_year"]),
    )
    if first_test_boundary is not None:
        return int(first_test_boundary.year)
    return fallback_year


def _filter_market_dates_by_end_year(sorted_dates, *, start_year=None, end_year=None):
    resolved_dates = [] if sorted_dates is None else list(sorted_dates)
    if end_year is None:
        return resolved_dates
    resolved_end_year = int(end_year)
    if resolved_end_year < 1900:
        raise ValueError("結束回測年份不可小於 1900")
    if start_year is not None and resolved_end_year < int(start_year):
        raise ValueError("結束回測年份不可早於開始回測年份")
    filtered_dates = [dt for dt in resolved_dates if pd.Timestamp(dt).year <= resolved_end_year]
    if not filtered_dates:
        raise ValueError("結束回測年份早於資料起始年，沒有可回測日期")
    return filtered_dates




def _build_portfolio_data_signature(csv_inputs):
    payload = []
    for ticker, file_path in csv_inputs:
        stat = os.stat(file_path)
        payload.append({
            "ticker": str(ticker),
            "path": os.path.basename(file_path),
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        })
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_portfolio_params_signature(params):
    payload = params_to_json_dict(params)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_portfolio_prepared_cache_paths(data_dir, csv_inputs, params):
    profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
    data_sig = _build_portfolio_data_signature(csv_inputs)
    params_sig = _build_portfolio_params_signature(params)
    combined_payload = {
        "schema_version": PORTFOLIO_PREP_CACHE_SCHEMA_VERSION,
        "profile_key": str(profile_key),
        "data_signature": data_sig,
        "params_signature": params_sig,
    }
    combined = json.dumps(combined_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]
    cache_dir = Path(OUTPUT_DIR) / "prepared_cache"
    stem = f"portfolio_prepared_{str(profile_key).strip().lower()}_{cache_key}"
    return {
        "payload_path": cache_dir / f"{stem}.pkl",
        "meta_path": cache_dir / f"{stem}.json",
        "meta": combined_payload,
    }


def _load_portfolio_prepared_cache(cache_paths):
    payload_path = cache_paths["payload_path"]
    meta_path = cache_paths["meta_path"]
    if not payload_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_meta = cache_paths["meta"]
    if int(meta.get("schema_version", 0) or 0) != PORTFOLIO_PREP_CACHE_SCHEMA_VERSION:
        return None
    for key in ("profile_key", "data_signature", "params_signature"):
        if str(meta.get(key, "")) != str(expected_meta.get(key, "")):
            return None
    try:
        with open(payload_path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not payload.get("all_dfs_fast") or not payload.get("sorted_dates"):
        return None
    return payload


def _save_portfolio_prepared_cache(cache_paths, context):
    payload_path = cache_paths["payload_path"]
    meta_path = cache_paths["meta_path"]
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PORTFOLIO_PREP_CACHE_SCHEMA_VERSION,
        "all_dfs_fast": context.get("all_dfs_fast") or {},
        "all_trade_logs": context.get("all_trade_logs") or {},
        "all_pit_stats_index": context.get("all_pit_stats_index") or {},
        "sorted_dates": context.get("sorted_dates") or [],
    }
    meta = dict(cache_paths["meta"])
    meta["ticker_count"] = int(len(payload["all_dfs_fast"]))
    tmp_payload = payload_path.with_suffix(payload_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_payload, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_payload, payload_path)
    os.replace(tmp_meta, meta_path)

def _resolve_portfolio_prep_workers(raw_data_count: int) -> int:
    raw_override = str(os.environ.get("V16_PORTFOLIO_MAX_WORKERS", "")).strip()
    if raw_override:
        try:
            return max(1, min(int(raw_override), max(1, int(raw_data_count))))
        except ValueError as exc:
            raise ValueError(f"V16_PORTFOLIO_MAX_WORKERS 必須是整數，收到: {raw_override}") from exc

    # # (AI註: 小型/reduced 資料集用單執行緒，避免 process pool 啟動成本吃掉收益；完整台股才啟用並行)
    if int(raw_data_count) < 30:
        return 1
    return max(1, min(os.cpu_count() or 1, 8, int(raw_data_count)))


def _load_portfolio_market_context_sequential(data_dir, params, *, verbose=True):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"{C_CYAN}📦 正在預載入歷史軌跡，構建真實時間軸...{C_RESET}")
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    load_issue_lines = []
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0
    total_skipped_insufficient = 0
    total_sanitize_issue_tickers = 0

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    load_issue_lines.extend(duplicate_file_issue_lines)
    total_files = len(csv_inputs)

    for count, (ticker, file_path) in enumerate(csv_inputs, start=1):
        try:
            raw_df = pd.read_csv(file_path)
            min_rows_needed = get_required_min_rows(params)

            if len(raw_df) < min_rows_needed:
                total_skipped_insufficient += 1
                load_issue_lines.append(
                    f"[資料不足] {ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 筆"
                )
                continue

            df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
            invalid_row_count = sanitize_stats['invalid_row_count']
            duplicate_date_count = sanitize_stats['duplicate_date_count']
            dropped_row_count = sanitize_stats['dropped_row_count']

            total_invalid_rows += invalid_row_count
            total_duplicate_dates += duplicate_date_count
            total_dropped_rows += dropped_row_count

            if dropped_row_count > 0:
                total_sanitize_issue_tickers += 1
                load_issue_lines.append(
                    f"[清洗] {ticker}: 清洗移除 {dropped_row_count} 列 "
                    f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
                )

            prep_df, logs = prep_stock_data_and_trades(df, params)
            master_dates.update(prep_df.index)
            all_dfs_fast[ticker] = pack_prepared_stock_data(prep_df)
            all_trade_logs[ticker] = logs

        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
            if is_insufficient_data_error(e):
                total_skipped_insufficient += 1
                load_issue_lines.append(f"[資料不足] {ticker}: {type(e).__name__}: {e}")
                continue
            raise RuntimeError(
                f"預載入失敗: ticker={ticker} | {format_exception_summary(e)}"
            ) from e

        if count % LOAD_PROGRESS_EVERY == 0 or count == total_files:
            vprint(
                f"{C_GRAY}   預載入進度: [{count}/{total_files}] "
                f"成功:{len(all_dfs_fast)} | 資料不足:{total_skipped_insufficient}{C_RESET}",
                end="\r",
                flush=True,
            )

    load_log_path = write_issue_log("portfolio_sim_load_issues", load_issue_lines, log_dir=OUTPUT_DIR) if load_issue_lines else None

    vprint(" " * 160, end="\r")
    if load_log_path:
        vprint(f"{C_YELLOW}⚠️ 預載入摘要已寫入: {load_log_path}{C_RESET}")
    if not all_dfs_fast:
        raise RuntimeError("未能成功載入任何股票資料！")

    sorted_dates = sorted(master_dates)
    vprint(
        f"\n{C_GREEN}✅ 預處理完成！共載入 {len(all_dfs_fast)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})，"
        f"候選清洗 {total_sanitize_issue_tickers} 檔，"
        f"資料不足跳過 {total_skipped_insufficient} 檔。{C_RESET}\n"
    )
    return {
        "all_dfs_fast": all_dfs_fast,
        "all_trade_logs": all_trade_logs,
        "all_pit_stats_index": {},
        "sorted_dates": sorted_dates,
        "prep_wall_sec": 0.0,
        "prep_mode": "sequential",
    }



def _prepare_portfolio_context_from_raw_sequential(raw_data_cache, params, *, verbose=True):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    all_dfs_fast, all_trade_logs, all_pit_stats_index, master_dates = {}, {}, {}, set()
    prep_issue_lines = []
    total_files = len(raw_data_cache)
    vprint(f"{C_CYAN}📦 正在建立投組快取：標的={total_files}｜workers=1｜PIT history 直接建索引...{C_RESET}")

    for count, (ticker, df) in enumerate(sorted(raw_data_cache.items()), start=1):
        try:
            dynamic_data, logs, pit_stats_index = prep_optimizer_stock_data_bundle(
                df,
                params,
                ticker=ticker,
                include_trade_logs=True,
                include_pit_stats_index=True,
            )
            all_dfs_fast[ticker] = merge_static_market_with_dynamic(pack_static_market_data(df), dynamic_data)
            all_trade_logs[ticker] = logs
            all_pit_stats_index[ticker] = pit_stats_index
            master_dates.update(df.index)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                prep_issue_lines.append(f"[資料不足] {ticker}: {type(exc).__name__}: {exc}")
                continue
            raise RuntimeError(
                f"投組預處理失敗: ticker={ticker} | {format_exception_summary(exc)}"
            ) from exc

        if count % LOAD_PROGRESS_EVERY == 0 or count == total_files:
            vprint(
                f"{C_GRAY}   投組快取進度: [{count}/{total_files}] 成功:{len(all_dfs_fast)}{C_RESET}",
                end="\r",
                flush=True,
            )

    prep_log_path = write_issue_log("portfolio_sim_prep_issues", prep_issue_lines, log_dir=OUTPUT_DIR) if prep_issue_lines else None
    vprint(" " * 160, end="\r")
    if prep_log_path:
        vprint(f"{C_YELLOW}⚠️ 投組預處理摘要已寫入: {prep_log_path}{C_RESET}")
    if not all_dfs_fast:
        raise RuntimeError("未能成功建立任何 portfolio sim 快取標的！")
    vprint(f"\n{C_GREEN}✅ 預處理完成！共載入 {len(all_dfs_fast)} 檔標的。{C_RESET}\n")
    return {
        "all_dfs_fast": all_dfs_fast,
        "all_trade_logs": all_trade_logs,
        "all_pit_stats_index": all_pit_stats_index,
        "sorted_dates": sorted(master_dates),
        "prep_wall_sec": 0.0,
        "prep_mode": "sequential",
    }

def load_portfolio_market_context(data_dir, params, *, verbose=True):
    ensure_runtime_dirs()
    if not data_dir:
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))
    if not os.path.exists(data_dir):
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))

    csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    if len(csv_inputs) < 30:
        return _load_portfolio_market_context_sequential(data_dir, params, verbose=verbose)

    cache_paths = _build_portfolio_prepared_cache_paths(data_dir, csv_inputs, params)
    cached_context = _load_portfolio_prepared_cache(cache_paths)
    if cached_context is not None:
        if verbose:
            print(f"{C_GREEN}📦 投組預處理快取命中｜標的={len(cached_context.get('all_dfs_fast', {}))}{C_RESET}")
        cached_context["prep_wall_sec"] = 0.0
        cached_context["prep_mode"] = "prepared_cache"
        return cached_context

    required_min_rows = get_required_min_rows(params)
    raw_data_cache = load_all_raw_data(data_dir, required_min_rows, OUTPUT_DIR)
    if not raw_data_cache:
        raise RuntimeError("未能成功載入任何股票資料！")

    max_workers = _resolve_portfolio_prep_workers(len(raw_data_cache))
    if verbose:
        print(
            f"{C_CYAN}📦 正在建立投組快取：標的={len(raw_data_cache)}｜workers={max_workers}｜"
            f"PIT history 直接建索引...{C_RESET}"
        )

    if max_workers <= 1:
        # # (AI註: reduced/small dataset 走單執行緒，但直接吃 raw cache，避免 process pool 冷啟動與重讀 CSV)
        context = _prepare_portfolio_context_from_raw_sequential(raw_data_cache, params, verbose=verbose)
        _save_portfolio_prepared_cache(cache_paths, context)
        return context

    prep_result = prepare_trial_inputs(
        raw_data_cache,
        params,
        default_max_workers=max_workers,
        include_trade_logs=True,
        include_pit_stats_index=True,
    )
    prep_failures = prep_result.get("prep_failures") or []
    load_log_path = write_issue_log(
        "portfolio_sim_prep_issues",
        [f"{ticker}: {reason}" for ticker, reason in prep_failures],
        log_dir=OUTPUT_DIR,
    ) if prep_failures else None

    all_dfs_fast = prep_result.get("all_dfs_fast") or {}
    if not all_dfs_fast:
        raise RuntimeError("未能成功建立任何 portfolio sim 快取標的！")

    sorted_dates = sorted(prep_result.get("master_dates") or [])
    if verbose:
        if load_log_path:
            print(f"{C_YELLOW}⚠️ 投組預處理摘要已寫入: {load_log_path}{C_RESET}")
        prep_profile = prep_result.get("prep_profile") or {}
        print(
            f"{C_GREEN}✅ 預處理完成！共載入 {len(all_dfs_fast)} 檔標的，"
            f"模式={prep_result.get('prep_mode')}，workers={max_workers}，"
            f"wall={float(prep_result.get('prep_wall_sec', 0.0)):.2f}s，"
            f"訊號={float(prep_profile.get('generate_signals_sum_sec', 0.0)):.2f}s，"
            f"單股回測={float(prep_profile.get('run_backtest_sum_sec', 0.0)):.2f}s。{C_RESET}\n"
        )

    context = {
        "all_dfs_fast": all_dfs_fast,
        "all_trade_logs": prep_result.get("all_trade_logs") or {},
        "all_pit_stats_index": prep_result.get("all_pit_stats_index") or {},
        "sorted_dates": sorted_dates,
        "prep_wall_sec": float(prep_result.get("prep_wall_sec", 0.0)),
        "prep_mode": prep_result.get("prep_mode"),
    }
    _save_portfolio_prepared_cache(cache_paths, context)
    return context


def run_portfolio_simulation_prepared(all_dfs_fast, all_trade_logs, sorted_dates, params, max_positions=5, enable_rotation=False, start_year=None, end_year=None, benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=True, pit_stats_index=None):
    resolved_start_year = resolve_default_portfolio_start_year() if start_year is None else int(start_year)
    resolved_sorted_dates = _filter_market_dates_by_end_year(sorted_dates, start_year=resolved_start_year, end_year=end_year)
    benchmark_data = all_dfs_fast.get(benchmark_ticker, None)
    if verbose:
        print(" " * 120, end="\r")

    pf_profile = {}
    result = run_portfolio_timeline(
        all_dfs_fast,
        all_trade_logs,
        resolved_sorted_dates,
        resolved_start_year,
        params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=verbose,
        pit_stats_index=pit_stats_index,
    )
    return (*result, pf_profile)


def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False, start_year=None, end_year=None, benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=True):
    context = load_portfolio_market_context(data_dir, params, verbose=verbose)
    result = run_portfolio_simulation_prepared(
        context["all_dfs_fast"],
        context["all_trade_logs"],
        context["sorted_dates"],
        params,
        max_positions=max_positions,
        enable_rotation=enable_rotation,
        start_year=start_year,
        end_year=end_year,
        benchmark_ticker=benchmark_ticker,
        verbose=verbose,
        pit_stats_index=context.get("all_pit_stats_index"),
    )
    if not result or not isinstance(result[-1], dict):
        raise RuntimeError("portfolio simulation result missing mutable profile payload")
    result[-1]["prep_wall_sec"] = float(context.get("prep_wall_sec", 0.0))
    result[-1]["prep_mode"] = context.get("prep_mode")
    return result
