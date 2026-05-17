import hashlib
import json
import os
import pickle
import time
import uuid
from contextlib import suppress
from pathlib import Path

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.dataset_profiles import (
    build_missing_dataset_dir_message,
    infer_dataset_profile_key_from_data_dir,
)
from core.display import C_CYAN, C_GREEN, C_GRAY, C_YELLOW, C_RESET
from core.log_utils import format_exception_summary, write_issue_log
from core.params_io import build_params_from_mapping
from core.portfolio_param_runtime import (
    build_active_param_objects_from_payload,
    build_active_param_ensemble_objects_from_payload,
    build_portfolio_params_signature,
)
from core.walk_forward_policy import load_walk_forward_policy
from core.portfolio_stats import find_sim_start_idx
from core.active_param_ensemble import (
    get_active_param_ensemble_date_range,
    get_active_param_ensemble_policy,
    resolve_active_param_ensemble_mode,
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
)
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import build_normal_setup_index, build_trade_stats_index, merge_static_market_with_dynamic, pack_static_market_data, pack_prepared_stock_data, prep_optimizer_stock_data_bundle, prep_stock_data_and_trades
from tools.optimizer.raw_cache import load_all_raw_data
from tools.optimizer.trial_inputs import prepare_trial_inputs
from tools.optimizer.walk_forward import resolve_first_walk_forward_test_boundary
from .runtime_common import LOAD_PROGRESS_EVERY, OUTPUT_DIR, PROJECT_ROOT, ensure_runtime_dirs, is_insufficient_data_error

PORTFOLIO_DEFAULT_BENCHMARK_TICKER = "0050"
PORTFOLIO_PREP_CACHE_SCHEMA_VERSION = 2


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _portfolio_prepared_cache_include_trade_logs() -> bool:
    return _env_flag("PORTFOLIO_SIM_PREPARED_CACHE_INCLUDE_TRADE_LOGS", False)



def _coerce_schedule_trade_date(value):
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "date"):
        return value.date()
    return pd.Timestamp(value).date()


def _resolve_active_schedule_record(schedule_records, trade_date):
    current_date = _coerce_schedule_trade_date(trade_date)
    selected = None
    for record in schedule_records:
        if record["effective_date"] <= current_date:
            selected = record
        else:
            break
    if selected is None:
        first_date = schedule_records[0]["effective_date_text"] if schedule_records else "N/A"
        raise ValueError(f"{current_date.isoformat()} 早於第一個 active param 生效日 {first_date}")
    return selected


def _load_contexts_for_active_schedule(data_dir, schedule_records, *, verbose=True):
    contexts_by_signature = {}
    contexts_by_effective_date = {}
    for idx, record in enumerate(schedule_records, start=1):
        signature = str(record["params_signature"])
        if signature not in contexts_by_signature:
            if verbose:
                print(
                    f"{C_CYAN}📦 建立 active param 快取 [{idx}/{len(schedule_records)}] "
                    f"生效日={record['effective_date_text']}...{C_RESET}"
                )
            context = load_portfolio_market_context(data_dir, record["params_obj"], verbose=verbose)
            context = dict(context)
            if not context.get("all_pit_stats_index"):
                context["all_pit_stats_index"] = {
                    ticker: build_trade_stats_index(logs)
                    for ticker, logs in (context.get("all_trade_logs") or {}).items()
                }
            context["normal_setup_index"] = build_normal_setup_index(context.get("all_dfs_fast") or {})
            contexts_by_signature[signature] = context
        contexts_by_effective_date[record["effective_date_text"]] = contexts_by_signature[signature]
    return contexts_by_effective_date


def _prepare_context_for_ensemble_replay(context, *, keep_trade_logs=False):
    replay_context = dict(context)
    if not replay_context.get("all_pit_stats_index"):
        replay_context["all_pit_stats_index"] = {
            ticker: build_trade_stats_index(logs)
            for ticker, logs in (replay_context.get("all_trade_logs") or {}).items()
        }
    replay_context["normal_setup_index"] = build_normal_setup_index(replay_context.get("all_dfs_fast") or {})
    if not keep_trade_logs and replay_context.get("all_pit_stats_index"):
        replay_context["all_trade_logs"] = {}
    return replay_context


def _load_contexts_for_active_ensemble_schedule(data_dir, schedule_records, *, verbose=True, keep_trade_logs=False):
    contexts_by_signature = {}
    contexts_by_effective_date = {}
    total_members = sum(len(record.get("members") or []) for record in schedule_records)
    loaded_count = 0
    for record in schedule_records:
        member_contexts = []
        for member in record.get("members") or []:
            loaded_count += 1
            signature = str(member["params_signature"])
            if signature not in contexts_by_signature:
                if verbose:
                    print(
                        f"{C_CYAN}📦 建立 ensemble active param 快取 [{loaded_count}/{total_members}] "
                        f"生效日={record['effective_date_text'] or 'static'} member={member.get('member_index')}...{C_RESET}"
                    )
                context = load_portfolio_market_context(data_dir, member["params_obj"], verbose=verbose)
                contexts_by_signature[signature] = _prepare_context_for_ensemble_replay(
                    context,
                    keep_trade_logs=keep_trade_logs,
                )
            member_contexts.append(contexts_by_signature[signature])
        contexts_by_effective_date[record["effective_date_text"]] = member_contexts
    return contexts_by_effective_date


def _merge_context_market_dates_from_ensemble(contexts_by_effective_date):
    market_dates = set()
    for contexts in contexts_by_effective_date.values():
        for context in contexts:
            market_dates.update(context.get("sorted_dates") or [])
    return sorted(market_dates)


def _merge_context_market_dates(contexts_by_effective_date):
    market_dates = set()
    for context in contexts_by_effective_date.values():
        market_dates.update(context.get("sorted_dates") or [])
    return sorted(market_dates)

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


def _filter_market_dates_by_date_range(sorted_dates, *, start_date=None, end_date=None):
    resolved_dates = [] if sorted_dates is None else list(sorted_dates)
    start_ts = pd.Timestamp(start_date).normalize() if start_date is not None else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date is not None else None
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError("結束回測日期不可早於開始回測日期")
    filtered_dates = []
    for raw_date in resolved_dates:
        ts = pd.Timestamp(raw_date).normalize()
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        filtered_dates.append(raw_date)
    if resolved_dates and (start_ts is not None or end_ts is not None) and not filtered_dates:
        raise ValueError("指定回測日期區間沒有可回測日期")
    return filtered_dates


def _filter_market_dates_by_end_year(sorted_dates, *, start_year=None, end_year=None):
    resolved_dates = [] if sorted_dates is None else list(sorted_dates)
    start_date = f"{int(start_year)}-01-01" if start_year is not None else None
    end_date = None
    if end_year is not None:
        resolved_end_year = int(end_year)
        if resolved_end_year < 1900:
            raise ValueError("結束回測年份不可小於 1900")
        if start_year is not None and resolved_end_year < int(start_year):
            raise ValueError("結束回測年份不可早於開始回測年份")
        end_date = f"{resolved_end_year}-12-31"
    if start_date is None and end_date is None:
        return resolved_dates
    return _filter_market_dates_by_date_range(resolved_dates, start_date=start_date, end_date=end_date)




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


def _build_portfolio_prepared_cache_paths(data_dir, csv_inputs, params, *, include_trade_logs: bool = False):
    profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
    data_sig = _build_portfolio_data_signature(csv_inputs)
    params_sig = build_portfolio_params_signature(params)
    combined_payload = {
        "schema_version": PORTFOLIO_PREP_CACHE_SCHEMA_VERSION,
        "profile_key": str(profile_key),
        "data_signature": data_sig,
        "params_signature": params_sig,
        "include_trade_logs": bool(include_trade_logs),
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


def _build_portfolio_prepared_tmp_path(final_path: Path) -> Path:
    tmp_token = f"{os.getpid()}_{time.monotonic_ns()}_{uuid.uuid4().hex}"
    return final_path.with_name(f"{final_path.name}.{tmp_token}.tmp")


def _cleanup_portfolio_prepared_tmp_path(tmp_path: Path) -> None:
    with suppress(OSError):
        tmp_path.unlink(missing_ok=True)


def _replace_portfolio_prepared_cache_file(tmp_path: Path, final_path: Path, cache_paths) -> bool:
    retry_delays = (0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00)
    last_error = None
    for delay_sec in retry_delays:
        try:
            os.replace(tmp_path, final_path)
            return True
        except PermissionError as exc:
            last_error = exc
            if _load_portfolio_prepared_cache(cache_paths) is not None:
                _cleanup_portfolio_prepared_tmp_path(tmp_path)
                return False
            time.sleep(delay_sec)
    try:
        os.replace(tmp_path, final_path)
        return True
    except PermissionError as exc:
        last_error = exc
        if _load_portfolio_prepared_cache(cache_paths) is not None:
            _cleanup_portfolio_prepared_tmp_path(tmp_path)
            return False
    raise last_error


def _save_portfolio_prepared_cache(cache_paths, context):
    if _load_portfolio_prepared_cache(cache_paths) is not None:
        return

    payload_path = cache_paths["payload_path"]
    meta_path = cache_paths["meta_path"]
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PORTFOLIO_PREP_CACHE_SCHEMA_VERSION,
        "all_dfs_fast": context.get("all_dfs_fast") or {},
        "all_trade_logs": (context.get("all_trade_logs") or {}) if bool((cache_paths.get("meta") or {}).get("include_trade_logs")) else {},
        "all_pit_stats_index": context.get("all_pit_stats_index") or {},
        "sorted_dates": context.get("sorted_dates") or [],
    }
    meta = dict(cache_paths["meta"])
    meta["ticker_count"] = int(len(payload["all_dfs_fast"]))
    tmp_payload = _build_portfolio_prepared_tmp_path(payload_path)
    tmp_meta = _build_portfolio_prepared_tmp_path(meta_path)
    try:
        with open(tmp_payload, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_portfolio_prepared_cache_file(tmp_payload, payload_path, cache_paths)
        _replace_portfolio_prepared_cache_file(tmp_meta, meta_path, cache_paths)
    finally:
        _cleanup_portfolio_prepared_tmp_path(tmp_payload)
        _cleanup_portfolio_prepared_tmp_path(tmp_meta)

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

    include_trade_logs = _portfolio_prepared_cache_include_trade_logs()
    cache_paths = _build_portfolio_prepared_cache_paths(data_dir, csv_inputs, params, include_trade_logs=include_trade_logs)
    cached_context = _load_portfolio_prepared_cache(cache_paths)
    if cached_context is not None:
        if verbose:
            print(f"{C_GREEN}📦 投組預處理快取命中｜標的={len(cached_context.get('all_dfs_fast', {}))}{C_RESET}")
        cached_context["prep_wall_sec"] = 0.0
        cached_context["prep_mode"] = "prepared_cache"
        return cached_context

    required_min_rows = get_required_min_rows(params)
    raw_data_cache = load_all_raw_data(data_dir, required_min_rows, OUTPUT_DIR, verbose=verbose)
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
        include_trade_logs=bool(include_trade_logs),
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
        "all_trade_logs": (prep_result.get("all_trade_logs") or {}) if include_trade_logs else {},
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



def run_portfolio_simulation_with_param_schedule(
    data_dir,
    rolling_payload,
    max_positions=5,
    enable_rotation=False,
    start_year=None,
    end_year=None,
    benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    fixed_risk=None,
    verbose=True,
    return_context=False,
    start_date=None,
    end_date=None,
):
    schedule_records = build_active_param_objects_from_payload(rolling_payload, fixed_risk=fixed_risk)
    if not schedule_records:
        raise ValueError("rolling OOS active-param schedule 為空")

    schedule_start_date = pd.Timestamp(schedule_records[0]["effective_date_text"]).normalize()
    schedule_end_text = str(schedule_records[-1].get("effective_end_date_text") or schedule_records[-1]["effective_date_text"])
    schedule_end_date = pd.Timestamp(schedule_end_text).normalize()
    resolved_start_year = int(schedule_records[0]["year"] if start_year is None else start_year)
    requested_start_date = pd.Timestamp(start_date).normalize() if start_date is not None else pd.Timestamp(year=resolved_start_year, month=1, day=1)
    resolved_start_date = max(requested_start_date, schedule_start_date)
    if end_date is not None:
        requested_end_date = pd.Timestamp(end_date).normalize()
    elif end_year is not None:
        requested_end_date = pd.Timestamp(year=int(end_year), month=12, day=31)
    else:
        requested_end_date = schedule_end_date
    resolved_end_date = min(requested_end_date, schedule_end_date)
    if resolved_end_date < resolved_start_date:
        raise ValueError("active-param replay 日期區間無效：結束日早於開始日")

    contexts_by_effective_date = _load_contexts_for_active_schedule(data_dir, schedule_records, verbose=verbose)
    merged_dates = _merge_context_market_dates(contexts_by_effective_date)
    resolved_sorted_dates = _filter_market_dates_by_date_range(
        merged_dates,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
    )
    if not resolved_sorted_dates:
        raise ValueError("active-param replay 沒有可回測日期")

    first_sim_idx = find_sim_start_idx(resolved_sorted_dates, resolved_start_year)
    if first_sim_idx >= len(resolved_sorted_dates):
        raise ValueError("active-param replay 起始日期沒有可回測日期")
    first_record = _resolve_active_schedule_record(schedule_records, resolved_sorted_dates[first_sim_idx])
    base_context = contexts_by_effective_date[first_record["effective_date_text"]]
    base_params = first_record["params_obj"]
    benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)

    def active_params_resolver(trade_date):
        return _resolve_active_schedule_record(schedule_records, trade_date)["params_obj"]

    def active_context_resolver(trade_date):
        record = _resolve_active_schedule_record(schedule_records, trade_date)
        return contexts_by_effective_date[record["effective_date_text"]]

    pf_profile = {
        "param_policy": "active_param_replay",
        "active_param_schedule": [
            {
                "effective_date": str(record["effective_date_text"]),
                "effective_end_date": str(record.get("effective_end_date_text") or ""),
                "year": int(record["year"]),
                "params_signature": str(record["params_signature"]),
            }
            for record in schedule_records
        ],
    }
    if verbose:
        print(
            f"{C_GREEN}✅ active-param replay 準備完成："
            f"{schedule_records[0]['effective_date_text']}~{schedule_records[-1]['effective_date_text']}，"
            f"共 {len(schedule_records)} 版參數。{C_RESET}"
        )

    result = run_portfolio_timeline(
        base_context.get("all_dfs_fast") or {},
        base_context.get("all_trade_logs") or {},
        resolved_sorted_dates,
        resolved_start_year,
        base_params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=verbose,
        pit_stats_index=base_context.get("all_pit_stats_index"),
        active_params_resolver=active_params_resolver,
        active_context_resolver=active_context_resolver,
    )
    prep_wall_sec = sum(float(ctx.get("prep_wall_sec", 0.0)) for ctx in contexts_by_effective_date.values())
    pf_profile.update({
        "param_policy": "active_param_replay",
        "prep_wall_sec": prep_wall_sec,
        "prep_mode": "active_param_replay",
        "active_replay_start_date": resolved_start_date.strftime("%Y-%m-%d"),
        "active_replay_end_date": resolved_end_date.strftime("%Y-%m-%d"),
    })
    if return_context:
        # # (AI註: Workbench K 線頁只需要可繪圖的市場快取；正式 active-param replay 仍由每日 resolver 決定參數。)
        pf_profile["_workbench_context"] = {
            "all_dfs_fast": base_context.get("all_dfs_fast") or {},
            "all_trade_logs": base_context.get("all_trade_logs") or {},
            "all_pit_stats_index": base_context.get("all_pit_stats_index") or {},
            "sorted_dates": list(resolved_sorted_dates),
            "prep_wall_sec": prep_wall_sec,
            "prep_mode": "active_param_replay",
        }
    return (*result, pf_profile)


def run_portfolio_simulation_with_param_ensemble(
    data_dir,
    ensemble_payload,
    max_positions=5,
    enable_rotation=False,
    start_year=None,
    end_year=None,
    benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    fixed_risk=None,
    verbose=True,
    return_context=False,
    start_date=None,
    end_date=None,
):
    schedule_records = build_active_param_ensemble_objects_from_payload(ensemble_payload, fixed_risk=fixed_risk)
    if not schedule_records:
        raise ValueError("active-param ensemble schedule 為空")

    policy = get_active_param_ensemble_policy(ensemble_payload)
    mode = resolve_active_param_ensemble_mode(ensemble_payload)
    if mode == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        schedule_start_date = None
        schedule_end_date = None
        resolved_start_year = resolve_default_portfolio_start_year(data_dir) if start_year is None else int(start_year)
        requested_start_date = pd.Timestamp(start_date).normalize() if start_date is not None else pd.Timestamp(year=resolved_start_year, month=1, day=1)
        resolved_start_date = requested_start_date
        if end_date is not None:
            resolved_end_date = pd.Timestamp(end_date).normalize()
        elif end_year is not None:
            resolved_end_date = pd.Timestamp(year=int(end_year), month=12, day=31)
        else:
            resolved_end_date = None
    else:
        first_date, last_date = get_active_param_ensemble_date_range(ensemble_payload)
        schedule_start_date = pd.Timestamp(first_date).normalize()
        schedule_end_date = pd.Timestamp(last_date).normalize()
        resolved_start_year = int(schedule_records[0]["year"] if start_year is None else start_year)
        requested_start_date = pd.Timestamp(start_date).normalize() if start_date is not None else pd.Timestamp(year=resolved_start_year, month=1, day=1)
        resolved_start_date = max(requested_start_date, schedule_start_date)
        if end_date is not None:
            requested_end_date = pd.Timestamp(end_date).normalize()
        elif end_year is not None:
            requested_end_date = pd.Timestamp(year=int(end_year), month=12, day=31)
        else:
            requested_end_date = schedule_end_date
        resolved_end_date = min(requested_end_date, schedule_end_date)

    if resolved_end_date is not None and resolved_end_date < resolved_start_date:
        raise ValueError("active-param ensemble replay 日期區間無效：結束日早於開始日")

    contexts_by_effective_date = _load_contexts_for_active_ensemble_schedule(
        data_dir,
        schedule_records,
        verbose=verbose,
        keep_trade_logs=return_context,
    )
    merged_dates = _merge_context_market_dates_from_ensemble(contexts_by_effective_date)
    resolved_sorted_dates = _filter_market_dates_by_date_range(
        merged_dates,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
    )
    if not resolved_sorted_dates:
        raise ValueError("active-param ensemble replay 沒有可回測日期")

    first_sim_idx = find_sim_start_idx(resolved_sorted_dates, resolved_start_year)
    if first_sim_idx >= len(resolved_sorted_dates):
        raise ValueError("active-param ensemble replay 起始日期沒有可回測日期")
    first_record = _resolve_active_schedule_record(schedule_records, resolved_sorted_dates[first_sim_idx])
    base_contexts = contexts_by_effective_date[first_record["effective_date_text"]]
    base_context = base_contexts[0]
    base_params = first_record["members"][0]["params_obj"]
    benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)

    def active_param_ensemble_resolver(trade_date):
        return _resolve_active_schedule_record(schedule_records, trade_date)["members"]

    def active_context_ensemble_resolver(trade_date):
        record = _resolve_active_schedule_record(schedule_records, trade_date)
        return contexts_by_effective_date[record["effective_date_text"]]

    replay_end = resolved_end_date if resolved_end_date is not None else pd.Timestamp(resolved_sorted_dates[-1]).normalize()
    pf_profile = {
        "param_policy": "active_param_ensemble_replay",
        "active_param_ensemble": {
            "mode": mode,
            "seed_count": int(policy["seed_count"]),
            "min_agree": int(policy["min_agree"]),
        },
        "active_param_ensemble_schedule": [
            {
                "effective_date": str(record.get("effective_date_text") or ""),
                "effective_end_date": str(record.get("effective_end_date_text") or ""),
                "year": int(record.get("year") or 0),
                "member_count": int(len(record.get("members") or [])),
                "params_signature": str(record.get("params_signature") or ""),
            }
            for record in schedule_records
        ],
    }
    if verbose:
        date_label = (
            f"{schedule_records[0]['effective_date_text']}~{schedule_records[-1]['effective_date_text']}"
            if mode != ACTIVE_PARAM_ENSEMBLE_MODE_STATIC else "static"
        )
        print(
            f"{C_GREEN}✅ active-param ensemble replay 準備完成："
            f"{date_label}，members={policy['seed_count']}，min_agree={policy['min_agree']}。{C_RESET}"
        )

    result = run_portfolio_timeline(
        base_context.get("all_dfs_fast") or {},
        base_context.get("all_trade_logs") or {},
        resolved_sorted_dates,
        resolved_start_year,
        base_params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=verbose,
        pit_stats_index=base_context.get("all_pit_stats_index"),
        active_param_ensemble_resolver=active_param_ensemble_resolver,
        active_context_ensemble_resolver=active_context_ensemble_resolver,
        ensemble_min_agree=int(policy["min_agree"]),
    )
    prep_wall_sec = sum(float(ctx.get("prep_wall_sec", 0.0)) for contexts in contexts_by_effective_date.values() for ctx in contexts)
    pf_profile.update({
        "param_policy": "active_param_ensemble_replay",
        "prep_wall_sec": prep_wall_sec,
        "prep_mode": "active_param_ensemble_replay",
        "active_replay_start_date": resolved_start_date.strftime("%Y-%m-%d"),
        "active_replay_end_date": replay_end.strftime("%Y-%m-%d"),
    })
    if return_context:
        pf_profile["_workbench_context"] = {
            "all_dfs_fast": base_context.get("all_dfs_fast") or {},
            "all_trade_logs": base_context.get("all_trade_logs") or {},
            "all_pit_stats_index": base_context.get("all_pit_stats_index") or {},
            "sorted_dates": list(resolved_sorted_dates),
            "prep_wall_sec": prep_wall_sec,
            "prep_mode": "active_param_ensemble_replay",
        }
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
