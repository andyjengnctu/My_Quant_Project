from core.runtime_utils import is_insufficient_data_error
import hashlib
import json
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, sanitize_ohlcv_dataframe
from core.display import C_GRAY, C_GREEN, C_RESET, C_YELLOW
from core.log_utils import format_exception_summary, write_issue_log
from core.dataset_profiles import (
    build_empty_dataset_dir_message,
    build_missing_dataset_dir_message,
    infer_dataset_profile_key_from_data_dir,
)
from core.runtime_utils import resolve_strict_environment_flag as _env_flag

RAW_CACHE_SCHEMA_VERSION = 1
RAW_CACHE_LOCK_POLL_SEC = 0.25
RAW_CACHE_LOCK_STALE_SEC = 6 * 60 * 60
RAW_CACHE_REPLACE_RETRY_COUNT = 20
RAW_CACHE_REPLACE_RETRY_SEC = 0.10


def _raw_cache_use_enabled() -> bool:
    return _env_flag("OPTIMIZER_RAW_CACHE_ENABLED", False)


def _raw_cache_write_enabled() -> bool:
    return _env_flag("OPTIMIZER_RAW_CACHE_WRITE_ENABLED", False)


def _build_raw_cache_paths(output_dir, profile_key, required_min_rows):
    safe_profile = str(profile_key or "full").strip().lower() or "full"
    cache_dir = Path(output_dir) / "raw_cache"
    stem = f"optimizer_raw_cache_{safe_profile}_minrows{int(required_min_rows)}"
    return {
        "cache_dir": cache_dir,
        "payload_path": cache_dir / f"{stem}.pkl",
        "meta_path": cache_dir / f"{stem}.json",
        "lock_dir": cache_dir / f"{stem}.lock",
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


def _safe_unlink(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _atomic_replace_with_retries(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    last_exc = None
    for attempt in range(1, RAW_CACHE_REPLACE_RETRY_COUNT + 1):
        try:
            os.replace(src_path, dst_path)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt >= RAW_CACHE_REPLACE_RETRY_COUNT:
                break
            time.sleep(RAW_CACHE_REPLACE_RETRY_SEC)
        except OSError as exc:
            last_exc = exc
            if attempt >= RAW_CACHE_REPLACE_RETRY_COUNT:
                break
            time.sleep(RAW_CACHE_REPLACE_RETRY_SEC)
    raise RuntimeError(f"optimizer raw cache 寫入失敗：無法取代 {dst_path} | {format_exception_summary(last_exc)}") from last_exc


def _build_unique_tmp_path(target_path):
    target_path = Path(target_path)
    return target_path.with_name(f"{target_path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _raw_cache_lock_stale_seconds():
    value = str(os.environ.get("OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC", "")).strip()
    if not value:
        return float(RAW_CACHE_LOCK_STALE_SEC)
    try:
        return max(60.0, float(value))
    except ValueError as exc:
        raise ValueError(f"OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC 必須是數字秒數，收到: {value}") from exc


class _RawCacheBuildLock:
    def __init__(self, paths):
        self.cache_dir = Path(paths["cache_dir"])
        self.lock_dir = Path(paths["lock_dir"])
        self.owner_path = self.lock_dir / "owner.json"
        self.acquired = False

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stale_sec = _raw_cache_lock_stale_seconds()
        while True:
            try:
                os.mkdir(self.lock_dir)
                owner = {
                    "pid": os.getpid(),
                    "created_at": time.time(),
                    "lock_dir": str(self.lock_dir),
                }
                try:
                    self.owner_path.write_text(json.dumps(owner, ensure_ascii=False, indent=2), encoding="utf-8")
                except OSError as exc:
                    print(f"⚠️ raw cache lock owner 寫入失敗，將繼續持有 lock: {format_exception_summary(exc, include_traceback=False)}")
                self.acquired = True
                return self
            except FileExistsError:
                self._remove_stale_lock_if_needed(stale_sec)
                time.sleep(RAW_CACHE_LOCK_POLL_SEC)
            except OSError as exc:
                raise RuntimeError(f"optimizer raw cache lock 建立失敗: {self.lock_dir} | {format_exception_summary(exc)}") from exc

    def _remove_stale_lock_if_needed(self, stale_sec):
        try:
            mtime = self.lock_dir.stat().st_mtime
        except FileNotFoundError:
            return
        except OSError:
            return
        if time.time() - float(mtime) < float(stale_sec):
            return
        _safe_unlink(self.owner_path)
        try:
            os.rmdir(self.lock_dir)
        except OSError:
            return

    def __exit__(self, exc_type, exc, tb):
        if not self.acquired:
            return False
        _safe_unlink(self.owner_path)
        try:
            os.rmdir(self.lock_dir)
        except OSError as exc:
            print(f"⚠️ raw cache lock 釋放失敗: {format_exception_summary(exc, include_traceback=False)}")
        self.acquired = False
        return False


def _persisted_payload_to_raw_data_cache(persisted_payload, duplicate_file_issue_lines, *, output_dir, verbose=True):
    fresh_raw_data_cache = persisted_payload["raw_data_cache"]
    load_issues = list(duplicate_file_issue_lines)
    load_issues.extend(str(x) for x in persisted_payload.get("load_issues", []))
    totals = dict(persisted_payload.get("totals", {}))
    issue_path = _write_load_issues_if_needed(load_issues, output_dir=output_dir)
    _print_load_summary(
        fresh_raw_data_cache=fresh_raw_data_cache,
        totals=totals,
        load_issues=load_issues,
        issue_path=issue_path,
        cache_hit=True,
        verbose=verbose,
    )
    return fresh_raw_data_cache


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

    tmp_payload = _build_unique_tmp_path(payload_path)
    tmp_meta = _build_unique_tmp_path(meta_path)
    try:
        with open(tmp_payload, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tmp_meta, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        _atomic_replace_with_retries(tmp_payload, payload_path)
        _atomic_replace_with_retries(tmp_meta, meta_path)
    finally:
        _safe_unlink(tmp_payload)
        _safe_unlink(tmp_meta)


def _write_load_issues_if_needed(load_issues, *, output_dir):
    if not load_issues:
        return None
    return write_issue_log("optimizer_load_issues", load_issues, log_dir=output_dir)


def _print_load_summary(*, fresh_raw_data_cache, totals, load_issues, issue_path, cache_hit, verbose=True):
    if not bool(verbose):
        return
    source_text = "磁碟快取命中" if cache_hit else "記憶體快取建立完成"
    dropped_rows = int(totals.get('total_dropped_rows', 0))
    invalid_rows = int(totals.get('total_invalid_rows', 0))
    duplicate_dates = int(totals.get('total_duplicate_dates', 0))
    issue_text = ""
    if load_issues and issue_path:
        issue_text = f"｜資料載入/清洗摘要={len(load_issues)}筆（已寫入 issue log）"
    print(
        f"{C_GREEN}📦 歷史資料：{source_text}｜標的={len(fresh_raw_data_cache)}｜"
        f"清洗移除={dropped_rows}列（異常OHLCV={invalid_rows}, 重複日期={duplicate_dates}）"
        f"{issue_text}{C_RESET}"
    )

def is_insufficient_data_message(message):
    return isinstance(message, str) and ("有效資料不足" in message)


def resolve_optimizer_max_workers(params, default_max_workers):
    configured = getattr(params, "optimizer_max_workers", default_max_workers)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = default_max_workers
    return max(1, configured)


def load_all_raw_data(data_dir, required_min_rows, output_dir, *, verbose=True):
    if not os.path.exists(data_dir):
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    if not csv_inputs:
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_empty_dataset_dir_message(profile_key, data_dir))

    profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
    use_raw_cache = _raw_cache_use_enabled()
    write_raw_cache = _raw_cache_write_enabled()
    cache_paths = None
    signature = None
    signature_payload = None
    if use_raw_cache or write_raw_cache:
        cache_paths = _build_raw_cache_paths(output_dir, profile_key, required_min_rows)
        signature, signature_payload = _build_raw_cache_signature(csv_inputs, required_min_rows)

    if use_raw_cache and cache_paths is not None:
        persisted_payload = _load_persisted_raw_cache(cache_paths, signature)
        if persisted_payload is not None:
            return _persisted_payload_to_raw_data_cache(
                persisted_payload,
                duplicate_file_issue_lines,
                output_dir=output_dir,
                verbose=verbose,
            )

    lock_context = _RawCacheBuildLock(cache_paths) if write_raw_cache and cache_paths is not None else nullcontext()
    with lock_context:
        if use_raw_cache and cache_paths is not None:
            persisted_payload = _load_persisted_raw_cache(cache_paths, signature)
            if persisted_payload is not None:
                return _persisted_payload_to_raw_data_cache(
                    persisted_payload,
                    duplicate_file_issue_lines,
                    output_dir=output_dir,
                    verbose=verbose,
                )

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

            if bool(verbose) and (count % 50 == 0 or count == total_files):
                print(f"{C_GRAY}   進度: [{count}/{total_files}] 已掃描股票快取...{C_RESET}", end="\r")

        if not fresh_raw_data_cache:
            raise RuntimeError("記憶體快取完成後仍無任何可用標的，無法進行 optimizer。")

        totals = {
            "total_invalid_rows": total_invalid_rows,
            "total_duplicate_dates": total_duplicate_dates,
            "total_dropped_rows": total_dropped_rows,
        }
        if write_raw_cache and cache_paths is not None:
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
            verbose=verbose,
        )

        return fresh_raw_data_cache
