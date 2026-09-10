"""Outer-rolling execution-performance and resource-diagnostics helpers.

This service isolates execution-only policy application and resource sampling from
the scientific outer-rolling orchestration.  Config defaults remain owned by
``config.training_performance_policy`` and runtime coercion by
``core.training_performance``.
"""

from __future__ import annotations

import os
import time
from threading import Event, Thread

from core.runtime_utils import resolve_environment_flag as _env_flag
from core.training_performance import (
    build_optimizer_outer_rolling_env_defaults,
    is_optimizer_active_replay_include_pit_stats_index_enabled,
    is_optimizer_active_replay_include_trade_logs_enabled,
    is_optimizer_active_replay_use_prepared_cache_enabled,
    is_optimizer_active_replay_write_prepared_cache_enabled,
    is_optimizer_profile_write_files_enabled,
    is_optimizer_local_min_dependency_stats_enabled,
    resolve_optimizer_local_min_portfolio_dependency_order,
    resolve_optimizer_local_min_signal_dependency_field_order,
    resolve_optimizer_outer_rolling_study_storage_default,
    is_optimizer_resource_write_csv_enabled,
    is_optimizer_single_fold_tpe_parallel_search_allowed_default,
    resolve_optimizer_feature_bank_max_items_default,
    resolve_optimizer_resource_disk_mbps_cap,
    resolve_optimizer_resource_sample_interval_sec,
    resolve_optimizer_rolling_fold_workers_default,
    resolve_optimizer_rolling_parallel_prep_cache_max_items_default,
    resolve_optimizer_rolling_shared_prep_cache_max_items,
    resolve_optimizer_single_fold_local_min_parallel_workers_default,
    resolve_optimizer_single_fold_local_min_process_workers_default,
)


def _mark_resource_probe_fallback(exc: BaseException) -> None:
    # Resource sampling is best-effort only; keep the fallback silent while still
    # binding the exception so broad probe failures remain traceable by contract.
    _ = f"{type(exc).__name__}: {exc}"


def _resolve_rolling_shared_prep_cache_max_items(environ) -> int:
    return resolve_optimizer_rolling_shared_prep_cache_max_items(environ)


def _resolve_rolling_fold_workers(environ, *, timing_mode: bool, fold_count: int | None = None) -> int:
    _ = timing_mode  # 一般模式與 timing mode 共用 training_performance_policy.py 的同一套效能預設。
    default_workers = resolve_optimizer_rolling_fold_workers_default(fold_count) if fold_count is not None else 1
    raw_value = (environ or {}).get("OPTIMIZER_ROLLING_FOLD_WORKERS")
    if raw_value is None:
        raw_value = os.environ.get("OPTIMIZER_ROLLING_FOLD_WORKERS", str(default_workers))
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        resolved = default_workers
    if fold_count is not None:
        try:
            max_workers = max(1, int(fold_count))
        except (TypeError, ValueError):
            max_workers = 8
    else:
        max_workers = 8
    return max(1, min(max_workers, resolved))


def _is_rolling_fold_parallel_enabled(environ, *, timing_mode: bool, fold_count: int) -> bool:
    _ = (environ, timing_mode)
    # 多 fold 一律走 parallel live board 路徑；OPTIMIZER_ROLLING_FOLD_WORKERS 只控制同時執行幾個 fold，
    # 不再讓 workers=1 / >1 分叉成兩套 console 顯示口徑。
    return int(fold_count) > 1


def _env_value_for_display(environ, name: str, default: str) -> str:
    value = None
    if isinstance(environ, dict):
        value = environ.get(name)
    if value is None:
        value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return str(default)
    return str(value).strip()


def _set_env_default(environ, name: str, value: str) -> None:
    if isinstance(environ, dict) and str(environ.get(name, "")).strip():
        return
    if str(os.environ.get(name, "")).strip():
        return
    os.environ[name] = str(value)
    if isinstance(environ, dict):
        environ[name] = str(value)


def _resolve_single_fold_search_parallel_trials(environ, *, sampler_kind: str) -> int:
    from services.optimizer.runtime import resolve_optimizer_single_fold_search_parallel_trials

    return resolve_optimizer_single_fold_search_parallel_trials(
        environ,
        sampler_kind=sampler_kind,
    )


def _apply_outer_rolling_resource_env_defaults(environ, *, timing_mode: bool, fold_count: int) -> None:
    _ = timing_mode  # 本函式只處理效能/資源預設；正式模式與 timing mode 應盡可能一致。
    for name, value in build_optimizer_outer_rolling_env_defaults(fold_count).items():
        _set_env_default(environ, name, value)


def _is_outer_rolling_sqlite_storage_enabled(environ) -> bool:
    value = _env_value_for_display(environ, "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE", resolve_optimizer_outer_rolling_study_storage_default()).strip().lower()
    return value in {"1", "true", "yes", "on", "sqlite", "sqlite_db", "db"}


def _format_parallel_settings_line(environ, *, fold_workers: int, sampler_kind: str = "") -> str:
    rolling_workers = _env_value_for_display(environ, "OPTIMIZER_ROLLING_FOLD_WORKERS", str(int(fold_workers)))
    search_parallel_trials = _resolve_single_fold_search_parallel_trials(environ, sampler_kind=sampler_kind)
    local_min_workers = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS", str(resolve_optimizer_single_fold_local_min_parallel_workers_default()))
    process_workers = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS", str(resolve_optimizer_single_fold_local_min_process_workers_default()))
    parallel_prep_cache_max_items = _env_value_for_display(
        environ,
        "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS",
        str(resolve_optimizer_rolling_parallel_prep_cache_max_items_default()),
    )
    feature_bank_max_items = _env_value_for_display(
        environ,
        "OPTIMIZER_FEATURE_BANK_MAX_ITEMS",
        str(resolve_optimizer_feature_bank_max_items_default()),
    )
    return (
        "平行化設定："
        f"OPTIMIZER_ROLLING_FOLD_WORKERS={rolling_workers} | "
        f"OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS={search_parallel_trials} | "
        f"OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS={local_min_workers} | "
        f"OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS={process_workers} | "
        f"OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS={parallel_prep_cache_max_items} | "
        f"OPTIMIZER_FEATURE_BANK_MAX_ITEMS={feature_bank_max_items}"
    )


def _build_training_performance_alignment_rows(environ, *, fold_count: int, fold_workers: int, sampler_kind: str) -> list[dict]:
    rolling_workers_text = _env_value_for_display(environ, "OPTIMIZER_ROLLING_FOLD_WORKERS", str(int(fold_workers)))
    search_parallel_trials_text = str(_resolve_single_fold_search_parallel_trials(environ, sampler_kind=sampler_kind))
    allow_tpe_parallel = "on" if _env_flag(environ, "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH", is_optimizer_single_fold_tpe_parallel_search_allowed_default()) else "off"
    local_min_workers_text = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS", str(resolve_optimizer_single_fold_local_min_parallel_workers_default()))
    local_min_process_workers_text = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS", str(resolve_optimizer_single_fold_local_min_process_workers_default()))
    parallel_cache_text = _env_value_for_display(
        environ,
        "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS",
        str(resolve_optimizer_rolling_parallel_prep_cache_max_items_default()),
    )
    feature_bank_text = _env_value_for_display(
        environ,
        "OPTIMIZER_FEATURE_BANK_MAX_ITEMS",
        str(resolve_optimizer_feature_bank_max_items_default()),
    )
    profile_write_files = "on" if is_optimizer_profile_write_files_enabled(environ, outer_rolling=True) else "off"
    study_storage = _env_value_for_display(environ, "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE", resolve_optimizer_outer_rolling_study_storage_default())
    resource_write_csv = "on" if is_optimizer_resource_write_csv_enabled(environ) else "off"
    replay_trade_logs = "on" if is_optimizer_active_replay_include_trade_logs_enabled(environ) else "off"
    replay_pit_stats = "on" if is_optimizer_active_replay_include_pit_stats_index_enabled(environ) else "off"
    replay_use_cache = "on" if is_optimizer_active_replay_use_prepared_cache_enabled(environ) else "off"
    replay_write_cache = "on" if is_optimizer_active_replay_write_prepared_cache_enabled(environ) else "off"
    field_order = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER", ",".join(resolve_optimizer_local_min_signal_dependency_field_order()) or "original")
    portfolio_order = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER", resolve_optimizer_local_min_portfolio_dependency_order())
    dep_stats = "on" if is_optimizer_local_min_dependency_stats_enabled() else "off"

    rows = [
        {
            "item": "rolling_fold_workers",
            "normal_mode": rolling_workers_text,
            "timing_mode": rolling_workers_text,
            "consistent": True,
            "result_scope": "不改單一 fold 計算；只改不同 OOS period 的執行順序",
            "display": "console plan / timing JSON summary / timing CSV fold rows",
        },
        {
            "item": "rolling_fold_parallel",
            "normal_mode": "on" if int(fold_count) > 1 else "off",
            "timing_mode": "on" if int(fold_count) > 1 else "off",
            "consistent": True,
            "result_scope": "OOS period fold 彼此獨立；正式 OOS_CHAIN 仍在所有 fold 完成後重算",
            "display": "console plan / timing JSON summary: rolling_fold_parallel",
        },
        {
            "item": "single_fold_search_parallel_trials",
            "normal_mode": f"{search_parallel_trials_text}, tpe_parallel={allow_tpe_parallel}",
            "timing_mode": f"{search_parallel_trials_text}, tpe_parallel={allow_tpe_parallel}",
            "consistent": True,
            "result_scope": "控制同一 fold 內 Optuna trial 併發；TPE 預設保護為 1，避免 trial 序列漂移",
            "display": "console plan / timing JSON summary: single_fold_search_parallel_trials",
        },
        {
            "item": "study_storage",
            "normal_mode": study_storage,
            "timing_mode": study_storage,
            "consistent": True,
            "result_scope": "不改分數；只避免 sqlite I/O",
            "display": "timing JSON summary: study_storage",
        },
        {
            "item": "profile_write_files",
            "normal_mode": profile_write_files,
            "timing_mode": profile_write_files,
            "consistent": True,
            "result_scope": "不改分數；只減少 profile 檔案輸出",
            "display": "timing JSON summary: profile_write_files",
        },
        {
            "item": "parallel_worker_prep_cache",
            "normal_mode": parallel_cache_text,
            "timing_mode": parallel_cache_text,
            "consistent": True,
            "result_scope": "不改分數；0 表示 parallel worker 不保留大型 prepared input cache",
            "display": "console plan / timing JSON summary: parallel_worker_prep_cache_max_items",
        },
        {
            "item": "feature_bank_max_items",
            "normal_mode": feature_bank_text,
            "timing_mode": feature_bank_text,
            "consistent": True,
            "result_scope": "不改特徵計算值；只限制 worker feature cache 大小",
            "display": "console plan / timing JSON summary: prep_feature_bank_max_items",
        },
        {
            "item": "local_min_workers",
            "normal_mode": f"thread={local_min_workers_text}, process={local_min_process_workers_text}",
            "timing_mode": f"thread={local_min_workers_text}, process={local_min_process_workers_text}",
            "consistent": True,
            "result_scope": "不改 local_min 定義；只固定同 fold 內鄰點評估併發口徑",
            "display": "console plan / timing JSON summary: local_min_parallel_workers_max",
        },
        {
            "item": "local_min_dependency_order",
            "normal_mode": f"portfolio={portfolio_order}, signal={field_order}",
            "timing_mode": f"portfolio={portfolio_order}, signal={field_order}",
            "consistent": True,
            "result_scope": "不改鄰點集合；只改 early-stop 前的鄰點排序",
            "display": "timing JSON summary: local_min_portfolio_dependency_deprioritized / local_min_signal_dependency_field_prioritized",
        },
        {
            "item": "local_min_dependency_stats",
            "normal_mode": dep_stats,
            "timing_mode": dep_stats,
            "consistent": True,
            "result_scope": "只新增觀測欄位",
            "display": "console timing summary / timing JSON summary / timing CSV fold rows",
        },
        {
            "item": "resource_csv",
            "normal_mode": resource_write_csv,
            "timing_mode": resource_write_csv,
            "consistent": True,
            "result_scope": "不改分數；只控制 resource 明細 CSV",
            "display": "timing JSON summary: resource_write_csv / resource_csv_rows",
        },
        {
            "item": "active_replay_lightweight",
            "normal_mode": f"trade_logs={replay_trade_logs}, pit_stats={replay_pit_stats}, use_cache={replay_use_cache}, write_cache={replay_write_cache}",
            "timing_mode": f"trade_logs={replay_trade_logs}, pit_stats={replay_pit_stats}, use_cache={replay_use_cache}, write_cache={replay_write_cache}",
            "consistent": True,
            "result_scope": "不改 OOS_CHAIN 權益曲線；只控制附帶 trade log/cache 輸出",
            "display": "timing JSON summary: active_replay_*",
        },
        {
            "item": "sampler_kind",
            "normal_mode": str(sampler_kind),
            "timing_mode": "random",
            "consistent": str(sampler_kind).strip().lower() == "random",
            "result_scope": "正式模式保留 TPE 選參；timing mode 保留 random 量測口徑，這是刻意差異",
            "display": "timing JSON root: sampler_kind / console final timing summary",
        },
        {
            "item": "formal_outputs",
            "normal_mode": "results + paramsets + timing json/csv",
            "timing_mode": "timing json/csv only",
            "consistent": False,
            "result_scope": "輸出範圍刻意不同；不影響計算口徑",
            "display": "console output paths",
        },
    ]
    return rows


def _shutdown_rolling_shared_prep_executor_holder(holder: dict) -> None:
    bundle = holder.pop("bundle", None) if isinstance(holder, dict) else None
    if bundle is None:
        return
    executor = bundle.get("executor")
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=True, cancel_futures=False)


def _resolve_resource_sample_interval_sec(environ) -> float:
    return resolve_optimizer_resource_sample_interval_sec(environ)


def _resolve_parallel_worker_prep_cache_max_items(environ) -> int:
    policy_default = resolve_optimizer_rolling_parallel_prep_cache_max_items_default()
    raw_value = _env_value_for_display(environ, "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS", str(policy_default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = int(policy_default)
    return max(0, min(256, value))


class _ResourceUsageSampler:
    """Low-overhead system resource sampler for rolling timing diagnostics.

    psutil is optional.  On Windows/Linux this class falls back to native OS
    counters so timing mode can still report CPU / memory / disk pressure on
    machines where psutil is not installed.
    """

    def __init__(self, *, interval_sec: float = 2.0):
        self.interval_sec = max(0.5, float(interval_sec or 2.0))
        self.samples: list[dict] = []
        self.available = False
        self.error = ""
        self.mode = "unavailable"
        self._psutil = None
        self._process = None
        self._stop_event = Event()
        self._thread = None
        self._started_at = 0.0
        self._last_disk = None
        self._last_sample_time = None
        self._last_cpu_times = None
        self._last_tree_io = None
        self._disk_mbps_cap = self._resolve_disk_mbps_cap()
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self.available = True
            self.mode = "psutil"
            # Prime CPU counters so first non-zero interval sample is meaningful.
            psutil.cpu_percent(interval=None)
        except Exception as exc:  # optional diagnostics only; keep optimizer runnable without psutil.
            self._psutil = None
            self._process = None
            self.error = f"{type(exc).__name__}: {exc}"
            if os.name == "nt":
                self.available = True
                self.mode = "native_windows"
            elif os.path.exists("/proc/stat") and os.path.exists("/proc/meminfo"):
                self.available = True
                self.mode = "native_linux"
            else:
                self.available = False
                self.mode = "unavailable"

    @staticmethod
    def _resolve_disk_mbps_cap() -> float:
        return resolve_optimizer_resource_disk_mbps_cap()

    def _disk_load_percent(self, *, busy_percent: float, total_mb_s: float) -> float:
        busy = max(0.0, float(busy_percent or 0.0))
        if busy > 0.0:
            return max(0.0, min(100.0, busy))
        return max(0.0, min(100.0, (max(0.0, float(total_mb_s or 0.0)) / self._disk_mbps_cap) * 100.0))

    def start(self) -> None:
        self._started_at = time.perf_counter()
        if not self.available:
            return
        self._sample_once()
        self._thread = Thread(target=self._run, name="optimizer-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.available:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_sec + 0.5))
        self._sample_once()

    def _iter_process_tree(self):
        psutil = self._psutil
        process = self._process
        if psutil is None or process is None:
            return []
        processes = []
        try:
            processes.append(process)
            processes.extend(process.children(recursive=True))
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
        return processes

    def _process_tree_stats_psutil(self) -> dict:
        rss_bytes = 0
        process_count = 0
        read_bytes = 0
        write_bytes = 0
        for proc in self._iter_process_tree():
            try:
                rss_bytes += int(proc.memory_info().rss)
                process_count += 1
            except Exception as exc:
                _mark_resource_probe_fallback(exc)
                continue
            try:
                io = proc.io_counters()
                read_bytes += int(getattr(io, "read_bytes", 0) or 0)
                write_bytes += int(getattr(io, "write_bytes", 0) or 0)
            except Exception as exc:
                _mark_resource_probe_fallback(exc)
        return {
            "process_tree_rss_gb": rss_bytes / (1024 ** 3),
            "process_tree_count": int(process_count),
            "process_tree_read_total_mb": read_bytes / (1024 ** 2),
            "process_tree_write_total_mb": write_bytes / (1024 ** 2),
        }

    @staticmethod
    def _read_linux_cpu_times() -> tuple[int, int] | None:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                parts = handle.readline().strip().split()
            if not parts or parts[0] != "cpu":
                return None
            values = [int(float(x)) for x in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            return int(idle), int(total)
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            return None

    @staticmethod
    def _read_linux_memory() -> dict:
        values = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, raw = line.split(":", 1)
                    amount = raw.strip().split()[0]
                    values[key] = float(amount) * 1024.0
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
        total = float(values.get("MemTotal", 0.0) or 0.0)
        available = float(values.get("MemAvailable", 0.0) or 0.0)
        used = max(0.0, total - available)
        swap_total = float(values.get("SwapTotal", 0.0) or 0.0)
        swap_free = float(values.get("SwapFree", 0.0) or 0.0)
        swap_used = max(0.0, swap_total - swap_free)
        return {
            "memory_percent": (used / total * 100.0) if total > 0 else 0.0,
            "memory_available_gb": available / (1024 ** 3),
            "memory_used_gb": used / (1024 ** 3),
            "swap_percent": (swap_used / swap_total * 100.0) if swap_total > 0 else 0.0,
            "swap_used_gb": swap_used / (1024 ** 3),
        }

    @staticmethod
    def _linux_descendant_pids(root_pid: int) -> list[int]:
        ppid_by_pid: dict[int, int] = {}
        for name in os.listdir("/proc") if os.path.exists("/proc") else []:
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as handle:
                    stat = handle.read()
                # comm may contain spaces and is wrapped in parentheses; ppid is field 4.
                after = stat.rsplit(")", 1)[1].strip().split()
                if len(after) >= 2:
                    ppid_by_pid[pid] = int(after[1])
            except Exception as exc:
                _mark_resource_probe_fallback(exc)
                continue
        children: dict[int, list[int]] = {}
        for pid, ppid in ppid_by_pid.items():
            children.setdefault(ppid, []).append(pid)
        out = [int(root_pid)]
        stack = list(children.get(int(root_pid), []))
        while stack:
            pid = stack.pop()
            out.append(pid)
            stack.extend(children.get(pid, []))
        return out

    @staticmethod
    def _read_linux_process_tree_stats() -> dict:
        rss_bytes = 0
        read_bytes = 0
        write_bytes = 0
        count = 0
        for pid in _ResourceUsageSampler._linux_descendant_pids(os.getpid()):
            try:
                with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if line.startswith("VmRSS:"):
                            rss_bytes += int(line.split()[1]) * 1024
                            break
                count += 1
            except Exception as exc:
                _mark_resource_probe_fallback(exc)
                continue
            try:
                with open(f"/proc/{pid}/io", "r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if line.startswith("read_bytes:"):
                            read_bytes += int(line.split()[1])
                        elif line.startswith("write_bytes:"):
                            write_bytes += int(line.split()[1])
            except Exception as exc:
                _mark_resource_probe_fallback(exc)
        return {
            "process_tree_rss_gb": rss_bytes / (1024 ** 3),
            "process_tree_count": int(count),
            "process_tree_read_total_mb": read_bytes / (1024 ** 2),
            "process_tree_write_total_mb": write_bytes / (1024 ** 2),
        }

    @staticmethod
    def _read_windows_cpu_times() -> tuple[int, int] | None:
        try:
            import ctypes
            from ctypes import wintypes

            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

            idle = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None

            def as_int(ft) -> int:
                return (int(ft.dwHighDateTime) << 32) + int(ft.dwLowDateTime)

            idle_i = as_int(idle)
            kernel_i = as_int(kernel)
            user_i = as_int(user)
            return int(idle_i), int(kernel_i + user_i)
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            return None

    @staticmethod
    def _read_windows_memory() -> dict:
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                raise OSError("GlobalMemoryStatusEx failed")
            total = float(mem.ullTotalPhys or 0)
            avail = float(mem.ullAvailPhys or 0)
            used = max(0.0, total - avail)
            page_total = float(mem.ullTotalPageFile or 0)
            page_avail = float(mem.ullAvailPageFile or 0)
            page_used = max(0.0, page_total - page_avail)
            return {
                "memory_percent": float(mem.dwMemoryLoad or ((used / total * 100.0) if total else 0.0)),
                "memory_available_gb": avail / (1024 ** 3),
                "memory_used_gb": used / (1024 ** 3),
                "swap_percent": (page_used / page_total * 100.0) if page_total > 0 else 0.0,
                "swap_used_gb": page_used / (1024 ** 3),
            }
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            return {
                "memory_percent": 0.0,
                "memory_available_gb": 0.0,
                "memory_used_gb": 0.0,
                "swap_percent": 0.0,
                "swap_used_gb": 0.0,
            }

    @staticmethod
    def _windows_descendant_pids(root_pid: int) -> list[int]:
        try:
            import ctypes
            from ctypes import wintypes

            TH32CS_SNAPPROCESS = 0x00000002
            INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260),
                ]

            kernel32 = ctypes.windll.kernel32
            snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            if snapshot == INVALID_HANDLE_VALUE:
                return [int(root_pid)]
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            ppid_by_pid: dict[int, int] = {}
            try:
                ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
                while ok:
                    ppid_by_pid[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                    ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
            finally:
                kernel32.CloseHandle(snapshot)
            children: dict[int, list[int]] = {}
            for pid, ppid in ppid_by_pid.items():
                children.setdefault(ppid, []).append(pid)
            out = [int(root_pid)]
            stack = list(children.get(int(root_pid), []))
            while stack:
                pid = stack.pop()
                out.append(pid)
                stack.extend(children.get(pid, []))
            return out
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            return [int(root_pid)]

    @staticmethod
    def _read_windows_process_tree_stats() -> dict:
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_VM_READ = 0x0010

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            rss_bytes = 0
            read_bytes = 0
            write_bytes = 0
            count = 0
            for pid in _ResourceUsageSampler._windows_descendant_pids(os.getpid()):
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid))
                if not handle:
                    continue
                try:
                    pmc = PROCESS_MEMORY_COUNTERS()
                    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                        rss_bytes += int(pmc.WorkingSetSize or 0)
                    ioc = IO_COUNTERS()
                    if kernel32.GetProcessIoCounters(handle, ctypes.byref(ioc)):
                        read_bytes += int(ioc.ReadTransferCount or 0)
                        write_bytes += int(ioc.WriteTransferCount or 0)
                    count += 1
                finally:
                    kernel32.CloseHandle(handle)
            return {
                "process_tree_rss_gb": rss_bytes / (1024 ** 3),
                "process_tree_count": int(count),
                "process_tree_read_total_mb": read_bytes / (1024 ** 2),
                "process_tree_write_total_mb": write_bytes / (1024 ** 2),
            }
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            return {
                "process_tree_rss_gb": 0.0,
                "process_tree_count": 0,
                "process_tree_read_total_mb": 0.0,
                "process_tree_write_total_mb": 0.0,
            }

    def _sample_once(self) -> None:
        if self.mode == "psutil":
            self._sample_once_psutil()
        elif self.mode == "native_windows":
            self._sample_once_native(cpu_reader=self._read_windows_cpu_times, memory_reader=self._read_windows_memory, tree_reader=self._read_windows_process_tree_stats)
        elif self.mode == "native_linux":
            self._sample_once_native(cpu_reader=self._read_linux_cpu_times, memory_reader=self._read_linux_memory, tree_reader=self._read_linux_process_tree_stats)

    def _sample_once_psutil(self) -> None:
        psutil = self._psutil
        if psutil is None:
            return
        now = time.perf_counter()
        elapsed = max(0.0, now - float(self._started_at or now))
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            cpu_percent = 0.0
        try:
            mem = psutil.virtual_memory()
            memory_percent = float(getattr(mem, "percent", 0.0) or 0.0)
            memory_available_gb = float(getattr(mem, "available", 0) or 0) / (1024 ** 3)
            memory_used_gb = float(getattr(mem, "used", 0) or 0) / (1024 ** 3)
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            memory_percent = 0.0
            memory_available_gb = 0.0
            memory_used_gb = 0.0
        try:
            swap = psutil.swap_memory()
            swap_percent = float(getattr(swap, "percent", 0.0) or 0.0)
            swap_used_gb = float(getattr(swap, "used", 0) or 0) / (1024 ** 3)
        except Exception as exc:
            _mark_resource_probe_fallback(exc)
            swap_percent = 0.0
            swap_used_gb = 0.0

        read_mb_s = 0.0
        write_mb_s = 0.0
        busy_percent = 0.0
        read_total_mb = 0.0
        write_total_mb = 0.0
        try:
            disk = psutil.disk_io_counters()
            if disk is not None:
                read_total_mb = float(getattr(disk, "read_bytes", 0) or 0) / (1024 ** 2)
                write_total_mb = float(getattr(disk, "write_bytes", 0) or 0) / (1024 ** 2)
                last_disk = self._last_disk
                last_time = self._last_sample_time
                if last_disk is not None and last_time is not None:
                    dt = max(0.001, now - float(last_time))
                    read_delta = max(0, int(getattr(disk, "read_bytes", 0) or 0) - int(getattr(last_disk, "read_bytes", 0) or 0))
                    write_delta = max(0, int(getattr(disk, "write_bytes", 0) or 0) - int(getattr(last_disk, "write_bytes", 0) or 0))
                    read_mb_s = float(read_delta) / (1024 ** 2) / dt
                    write_mb_s = float(write_delta) / (1024 ** 2) / dt
                    busy_now = getattr(disk, "busy_time", None)
                    busy_last = getattr(last_disk, "busy_time", None)
                    if busy_now is not None and busy_last is not None:
                        busy_delta_ms = max(0.0, float(busy_now) - float(busy_last))
                        busy_percent = max(0.0, min(100.0, busy_delta_ms / (dt * 10.0)))
                self._last_disk = disk
                self._last_sample_time = now
        except Exception as exc:
            _mark_resource_probe_fallback(exc)

        tree = self._process_tree_stats_psutil()
        self.samples.append({
            "elapsed_sec": float(elapsed),
            "resource_mode": str(self.mode),
            "cpu_percent": float(cpu_percent),
            "memory_percent": float(memory_percent),
            "memory_available_gb": float(memory_available_gb),
            "memory_used_gb": float(memory_used_gb),
            "swap_percent": float(swap_percent),
            "swap_used_gb": float(swap_used_gb),
            "disk_read_mb_per_sec": float(read_mb_s),
            "disk_write_mb_per_sec": float(write_mb_s),
            "disk_total_mb_per_sec": float(read_mb_s + write_mb_s),
            "disk_load_percent": float(self._disk_load_percent(busy_percent=busy_percent, total_mb_s=read_mb_s + write_mb_s)),
            "disk_busy_percent": float(busy_percent),
            "disk_read_total_mb": float(read_total_mb),
            "disk_write_total_mb": float(write_total_mb),
            "process_tree_rss_gb": float(tree.get("process_tree_rss_gb", 0.0) or 0.0),
            "process_tree_count": int(tree.get("process_tree_count", 0) or 0),
            "process_tree_read_total_mb": float(tree.get("process_tree_read_total_mb", 0.0) or 0.0),
            "process_tree_write_total_mb": float(tree.get("process_tree_write_total_mb", 0.0) or 0.0),
        })

    def _sample_once_native(self, *, cpu_reader, memory_reader, tree_reader) -> None:
        now = time.perf_counter()
        elapsed = max(0.0, now - float(self._started_at or now))
        cpu_percent = 0.0
        cpu_times = cpu_reader()
        if cpu_times is not None:
            last = self._last_cpu_times
            if last is not None:
                idle_delta = max(0, int(cpu_times[0]) - int(last[0]))
                total_delta = max(0, int(cpu_times[1]) - int(last[1]))
                if total_delta > 0:
                    cpu_percent = max(0.0, min(100.0, (1.0 - float(idle_delta) / float(total_delta)) * 100.0))
            self._last_cpu_times = cpu_times
        mem = memory_reader()
        tree = tree_reader()
        read_total_mb = float(tree.get("process_tree_read_total_mb", 0.0) or 0.0)
        write_total_mb = float(tree.get("process_tree_write_total_mb", 0.0) or 0.0)
        read_mb_s = 0.0
        write_mb_s = 0.0
        if self._last_tree_io is not None and self._last_sample_time is not None:
            dt = max(0.001, now - float(self._last_sample_time))
            read_mb_s = max(0.0, read_total_mb - float(self._last_tree_io[0])) / dt
            write_mb_s = max(0.0, write_total_mb - float(self._last_tree_io[1])) / dt
        self._last_tree_io = (read_total_mb, write_total_mb)
        self._last_sample_time = now
        self.samples.append({
            "elapsed_sec": float(elapsed),
            "resource_mode": str(self.mode),
            "cpu_percent": float(cpu_percent),
            "memory_percent": float(mem.get("memory_percent", 0.0) or 0.0),
            "memory_available_gb": float(mem.get("memory_available_gb", 0.0) or 0.0),
            "memory_used_gb": float(mem.get("memory_used_gb", 0.0) or 0.0),
            "swap_percent": float(mem.get("swap_percent", 0.0) or 0.0),
            "swap_used_gb": float(mem.get("swap_used_gb", 0.0) or 0.0),
            "disk_read_mb_per_sec": float(read_mb_s),
            "disk_write_mb_per_sec": float(write_mb_s),
            "disk_total_mb_per_sec": float(read_mb_s + write_mb_s),
            "disk_load_percent": float(self._disk_load_percent(busy_percent=0.0, total_mb_s=read_mb_s + write_mb_s)),
            "disk_busy_percent": 0.0,
            "disk_read_total_mb": float(read_total_mb),
            "disk_write_total_mb": float(write_total_mb),
            "process_tree_rss_gb": float(tree.get("process_tree_rss_gb", 0.0) or 0.0),
            "process_tree_count": int(tree.get("process_tree_count", 0) or 0),
            "process_tree_read_total_mb": float(read_total_mb),
            "process_tree_write_total_mb": float(write_total_mb),
        })

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._sample_once()

    def summary(self) -> dict:
        samples = list(self.samples or [])
        if not self.available:
            return {"resource_sampling_available": False, "resource_sampling_error": self.error}
        if not samples:
            return {"resource_sampling_available": True, "resource_sampling_mode": self.mode, "resource_sample_count": 0}

        def values(key: str) -> list[float]:
            out = []
            for sample in samples:
                try:
                    out.append(float(sample.get(key, 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
            return out

        def avg(key: str) -> float:
            vals = values(key)
            return sum(vals) / float(len(vals)) if vals else 0.0

        def mx(key: str) -> float:
            vals = values(key)
            return max(vals) if vals else 0.0

        def mn(key: str) -> float:
            vals = values(key)
            return min(vals) if vals else 0.0

        first = samples[0]
        last = samples[-1]
        disk_read_delta = max(0.0, float(last.get("disk_read_total_mb", 0.0) or 0.0) - float(first.get("disk_read_total_mb", 0.0) or 0.0))
        disk_write_delta = max(0.0, float(last.get("disk_write_total_mb", 0.0) or 0.0) - float(first.get("disk_write_total_mb", 0.0) or 0.0))
        proc_read_delta = max(0.0, float(last.get("process_tree_read_total_mb", 0.0) or 0.0) - float(first.get("process_tree_read_total_mb", 0.0) or 0.0))
        proc_write_delta = max(0.0, float(last.get("process_tree_write_total_mb", 0.0) or 0.0) - float(first.get("process_tree_write_total_mb", 0.0) or 0.0))
        return {
            "resource_sampling_available": True,
            "resource_sampling_mode": str(self.mode),
            "resource_sampling_error": "" if self.mode == "psutil" else str(self.error or ""),
            "resource_sample_count": int(len(samples)),
            "resource_sample_interval_sec": float(self.interval_sec),
            "cpu_avg_percent": avg("cpu_percent"),
            "cpu_max_percent": mx("cpu_percent"),
            "memory_avg_percent": avg("memory_percent"),
            "memory_max_percent": mx("memory_percent"),
            "memory_min_available_gb": mn("memory_available_gb"),
            "memory_max_used_gb": mx("memory_used_gb"),
            "swap_avg_percent": avg("swap_percent"),
            "swap_max_percent": mx("swap_percent"),
            "swap_max_used_gb": mx("swap_used_gb"),
            "disk_read_mb": disk_read_delta,
            "disk_write_mb": disk_write_delta,
            "disk_total_mb": disk_read_delta + disk_write_delta,
            "disk_read_mb_per_sec_avg": avg("disk_read_mb_per_sec"),
            "disk_read_mb_per_sec_max": mx("disk_read_mb_per_sec"),
            "disk_write_mb_per_sec_avg": avg("disk_write_mb_per_sec"),
            "disk_write_mb_per_sec_max": mx("disk_write_mb_per_sec"),
            "disk_total_mb_per_sec_avg": avg("disk_total_mb_per_sec"),
            "disk_total_mb_per_sec_max": mx("disk_total_mb_per_sec"),
            "disk_busy_avg_percent": avg("disk_busy_percent"),
            "disk_busy_max_percent": mx("disk_busy_percent"),
            "disk_load_avg_percent": avg("disk_load_percent"),
            "disk_load_max_percent": mx("disk_load_percent"),
            "resource_disk_mbps_cap": float(self._disk_mbps_cap),
            "process_tree_rss_avg_gb": avg("process_tree_rss_gb"),
            "process_tree_rss_max_gb": mx("process_tree_rss_gb"),
            "process_tree_count_max": int(mx("process_tree_count")),
            "process_tree_read_mb": proc_read_delta,
            "process_tree_write_mb": proc_write_delta,
            "process_tree_total_io_mb": proc_read_delta + proc_write_delta,
        }
