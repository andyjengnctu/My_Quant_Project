import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, sanitize_ohlcv_dataframe
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RESET, C_YELLOW
from core.log_utils import format_exception_summary, write_issue_log
from core.dataset_profiles import (
    build_empty_dataset_dir_message,
    build_missing_dataset_dir_message,
    infer_dataset_profile_key_from_data_dir,
)

RAW_CACHE_SCHEMA_VERSION = 1


def _build_raw_cache_paths(output_dir, profile_key, required_min_rows):
    safe_profile = str(profile_key or "full").strip().lower() or "full"
    cache_dir = Path(output_dir) / "raw_cache"
    stem = f"optimizer_raw_cache_{safe_profile}_minrows{int(required_min_rows)}"
    return {
        "cache_dir": cache_dir,
        "payload_path": cache_dir / f"{stem}.pkl",
        "meta_path": cache_dir / f"{stem}.json",
    }


def _build_raw_cache_signature(csv_inputs, required_min_rows):
    signature_payload = {
        "schema_version": RAW_CACHE_SCHEMA_VERSION,
        "required_min_rows": int(required_min_rows),
        "files": [],
    }
    for ticker, file_path in csv_inputs:
        stat = os.stat(file_path)
        signature_payload["files"].append(
            {
                "ticker": str(ticker),
                "path": os.path.basename(file_path),
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            }
        )
    payload_bytes = json.dumps(signature_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest(), signature_payload


def _load_persisted_raw_cache(paths, expected_signature):
    meta_path = paths["meta_path"]
    payload_path = paths["payload_path"]
    if not meta_path.exists() or not payload_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(meta.get("signature", "")) != str(expected_signature):
        return None
    if int(meta.get("schema_version", 0) or 0) != RAW_CACHE_SCHEMA_VERSION:
        return None
    try:
        with open(payload_path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("signature", "")) != str(expected_signature):
        return None
    raw_data_cache = payload.get("raw_data_cache")
    if not isinstance(raw_data_cache, dict) or not raw_data_cache:
        return None
    return payload


def _save_persisted_raw_cache(paths, *, signature, signature_payload, raw_data_cache, load_issues, totals, profile_key, data_dir, required_min_rows):
    cache_dir = paths["cache_dir"]
    payload_path = paths["payload_path"]
    meta_path = paths["meta_path"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": RAW_CACHE_SCHEMA_VERSION,
        "signature": str(signature),
        "profile_key": str(profile_key),
        "data_dir": str(data_dir),
        "required_min_rows": int(required_min_rows),
        "raw_data_cache": raw_data_cache,
        "load_issues": list(load_issues),
        "totals": dict(totals),
    }
    meta = {
        "schema_version": RAW_CACHE_SCHEMA_VERSION,
        "signature": str(signature),
        "profile_key": str(profile_key),
        "data_dir": str(data_dir),
        "required_min_rows": int(required_min_rows),
        "ticker_count": int(len(raw_data_cache)),
        "signature_payload": signature_payload,
    }

    tmp_payload = payload_path.with_suffix(payload_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_payload, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_payload, payload_path)
    os.replace(tmp_meta, meta_path)


def _write_load_issues_if_needed(load_issues, *, output_dir):
    if not load_issues:
        return None
    return write_issue_log("optimizer_load_issues", load_issues, log_dir=output_dir)


def _print_load_summary(*, fresh_raw_data_cache, totals, load_issues, issue_path, cache_hit):
    source_text = "磁碟快取" if cache_hit else "記憶體快取"
    print(
        f"\n{C_GREEN}✅ {source_text}完成！共載入 {len(fresh_raw_data_cache)} 檔標的，"
        f"移除 {int(totals.get('total_dropped_rows', 0))} 列資料 "
        f"(異常OHLCV={int(totals.get('total_invalid_rows', 0))}, 重複日期={int(totals.get('total_duplicate_dates', 0))})。{C_RESET}\n"
    )
    if load_issues and issue_path:
        print(f"{C_YELLOW}⚠️ 資料載入/清洗摘要共 {len(load_issues)} 筆，已寫入: {issue_path}{C_RESET}")

def is_insufficient_data_message(message):
    return isinstance(message, str) and ("有效資料不足" in message)


def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))


def resolve_optimizer_max_workers(params, default_max_workers):
    configured = getattr(params, "optimizer_max_workers", default_max_workers)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = default_max_workers
    return max(1, configured)


def load_all_raw_data(data_dir, required_min_rows, output_dir):
    if not os.path.exists(data_dir):
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))

    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    if not csv_inputs:
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_empty_dataset_dir_message(profile_key, data_dir))

    profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
    cache_paths = _build_raw_cache_paths(output_dir, profile_key, required_min_rows)
    signature, signature_payload = _build_raw_cache_signature(csv_inputs, required_min_rows)
    persisted_payload = _load_persisted_raw_cache(cache_paths, signature)
    if persisted_payload is not None:
        fresh_raw_data_cache = persisted_payload["raw_data_cache"]
        load_issues = list(duplicate_file_issue_lines)
        load_issues.extend(str(x) for x in persisted_payload.get("load_issues", []))
        totals = dict(persisted_payload.get("totals", {}))
        issue_path = _write_load_issues_if_needed(load_issues, output_dir=output_dir)
        print(f"{C_GRAY}   命中磁碟快取: {cache_paths['payload_path']}{C_RESET}")
        _print_load_summary(
            fresh_raw_data_cache=fresh_raw_data_cache,
            totals=totals,
            load_issues=load_issues,
            issue_path=issue_path,
            cache_hit=True,
        )
        return fresh_raw_data_cache

    load_issues = list(duplicate_file_issue_lines)
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0
    total_files = len(csv_inputs)
    fresh_raw_data_cache = {}

    for count, (ticker, file_path) in enumerate(csv_inputs, start=1):
        try:
            raw_df = pd.read_csv(file_path)
            if len(raw_df) < required_min_rows:
                load_issues.append(f"{ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {required_min_rows} 列")
                continue

            clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=required_min_rows)
            fresh_raw_data_cache[ticker] = clean_df

            invalid_row_count = sanitize_stats["invalid_row_count"]
            duplicate_date_count = sanitize_stats["duplicate_date_count"]
            dropped_row_count = sanitize_stats["dropped_row_count"]

            total_invalid_rows += invalid_row_count
            total_duplicate_dates += duplicate_date_count
            total_dropped_rows += dropped_row_count

            if dropped_row_count > 0:
                load_issues.append(
                    f"{ticker}: 清洗移除 {dropped_row_count} 列 "
                    f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
                )
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                load_issues.append(f"{ticker}: {type(exc).__name__}: {exc}")
                continue
            raise RuntimeError(
                f"optimizer 原始資料快取失敗: ticker={ticker} | {format_exception_summary(exc)}"
            ) from exc

        if count % 50 == 0 or count == total_files:
            print(f"{C_GRAY}   進度: [{count}/{total_files}] 已掃描股票快取...{C_RESET}", end="\r")

    if not fresh_raw_data_cache:
        raise RuntimeError("記憶體快取完成後仍無任何可用標的，無法進行 optimizer。")

    totals = {
        "total_invalid_rows": total_invalid_rows,
        "total_duplicate_dates": total_duplicate_dates,
        "total_dropped_rows": total_dropped_rows,
    }
    _save_persisted_raw_cache(
        cache_paths,
        signature=signature,
        signature_payload=signature_payload,
        raw_data_cache=fresh_raw_data_cache,
        load_issues=load_issues,
        totals=totals,
        profile_key=profile_key,
        data_dir=data_dir,
        required_min_rows=required_min_rows,
    )
    issue_path = _write_load_issues_if_needed(load_issues, output_dir=output_dir)
    _print_load_summary(
        fresh_raw_data_cache=fresh_raw_data_cache,
        totals=totals,
        load_issues=load_issues,
        issue_path=issue_path,
        cache_hit=False,
    )

    return fresh_raw_data_cache
