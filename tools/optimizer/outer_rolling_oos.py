from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from threading import Event, Thread
from contextlib import redirect_stderr, redirect_stdout
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

PARALLEL_FOLD_HEARTBEAT_INTERVAL_SEC = 2.0
PARALLEL_FOLD_LOG_STATUS_MAX_CHARS = 140
PARALLEL_FOLD_PROGRESS_PREFIX = "FOLD_PROGRESS\t"

import pandas as pd

from config.training_policy import (
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_FIXED_TP_PERCENT,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW
from core.params_io import build_params_from_mapping
from core.portfolio_stats import calc_annual_return_pct, calc_curve_stats, calc_portfolio_score
from core.runtime_utils import choose_inline_progress_message, get_taipei_now, is_interactive_console, safe_prompt_choice, stdout_supports_inline_progress, write_inline_progress
from core.rolling_oos_params import ROLLING_OOS_PARAM_SET_SCHEMA_TYPE, ROLLING_OOS_USAGE
from core.strategy_params import build_runtime_param_raw_value
from core.walk_forward_policy import build_optimizer_runtime_policy
from tools.optimizer.param_cache import build_prep_cache_key
from tools.optimizer.prep import prepare_trial_inputs
from tools.optimizer.robustness import (
    _has_dependency_warning,
    _has_inner_validate_pass,
    is_dominant_year_dependency_anti_overfit_enabled,
    is_inner_validate_anti_overfit_enabled,
    list_local_min_score_finalists,
)
from tools.optimizer.study_utils import (
    INVALID_TRIAL_VALUE,
    build_best_params_payload_from_trial,
    is_qualified_trial_value,
)
from tools.optimizer.walk_forward import evaluate_walk_forward


@dataclass
class OuterRollingConfig:
    training_start_year: int
    first_oos_year: int
    last_oos_year: int
    trials_per_fold: int
    window_mode: str = "fixed"
    train_window_years: int = 5
    confirm: bool = True


OOS_SCORE_DECIMALS = 2


def _resolve_rolling_shared_prep_cache_max_items(environ) -> int:
    raw_value = (environ or {}).get("OPTIMIZER_ROLLING_SHARED_PREP_CACHE_MAX_ITEMS")
    if raw_value is None:
        raw_value = os.environ.get("OPTIMIZER_ROLLING_SHARED_PREP_CACHE_MAX_ITEMS", "256")
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        resolved = 256
    return max(0, min(4096, resolved))


def _resolve_rolling_fold_workers(environ, *, timing_mode: bool, fold_count: int | None = None) -> int:
    default_workers = 1
    if bool(timing_mode) and fold_count is not None:
        try:
            default_workers = max(1, int(fold_count))
        except (TypeError, ValueError):
            default_workers = 1
    raw_value = (environ or {}).get("OPTIMIZER_ROLLING_FOLD_WORKERS")
    if raw_value is None:
        raw_value = os.environ.get("OPTIMIZER_ROLLING_FOLD_WORKERS", str(default_workers))
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        resolved = default_workers
    if bool(timing_mode) and fold_count is not None:
        try:
            max_workers = max(1, int(fold_count))
        except (TypeError, ValueError):
            max_workers = 8
    else:
        max_workers = 8
    resolved = max(1, min(max_workers, resolved))
    if not bool(timing_mode):
        return 1
    return resolved


def _is_rolling_fold_parallel_enabled(environ, *, timing_mode: bool, fold_count: int) -> bool:
    return bool(timing_mode) and int(fold_count) > 1 and _resolve_rolling_fold_workers(environ, timing_mode=timing_mode, fold_count=fold_count) > 1


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


def _env_flag(environ, name: str, default: bool) -> bool:
    value = None
    if isinstance(environ, dict):
        value = environ.get(name)
    if value is None:
        value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", "n"}


def _apply_outer_rolling_resource_env_defaults(environ, *, timing_mode: bool, fold_count: int) -> None:
    _set_env_default(environ, "OPTIMIZER_PROFILE_WRITE_FILES", "0")
    _set_env_default(environ, "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE", "memory")
    _set_env_default(environ, "OPTIMIZER_RESOURCE_WRITE_CSV", "0")
    _set_env_default(environ, "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS", "0")
    _set_env_default(environ, "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX", "1")
    _set_env_default(environ, "OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE", "0")
    _set_env_default(environ, "OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE", "0")
    if bool(timing_mode):
        _set_env_default(environ, "OPTIMIZER_ROLLING_FOLD_WORKERS", str(max(1, int(fold_count))))
        _set_env_default(environ, "OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS", "1")
        _set_env_default(environ, "OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS", "0")
        _set_env_default(environ, "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS", "32")


def _is_outer_rolling_sqlite_storage_enabled(environ) -> bool:
    value = _env_value_for_display(environ, "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE", "memory").strip().lower()
    return value in {"1", "true", "yes", "on", "sqlite", "sqlite_db", "db"}


def _format_parallel_settings_line(environ, *, fold_workers: int) -> str:
    rolling_workers = _env_value_for_display(environ, "OPTIMIZER_ROLLING_FOLD_WORKERS", str(int(fold_workers)))
    local_min_workers = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS", "1")
    process_workers = _env_value_for_display(environ, "OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS", "0")
    return (
        "平行化設定："
        f"OPTIMIZER_ROLLING_FOLD_WORKERS={rolling_workers} | "
        f"OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS={local_min_workers} | "
        f"OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS={process_workers}"
    )


def _shutdown_rolling_shared_prep_executor_holder(holder: dict) -> None:
    bundle = holder.pop("bundle", None) if isinstance(holder, dict) else None
    if bundle is None:
        return
    executor = bundle.get("executor")
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=True, cancel_futures=False)


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "N/A"
    total = max(0, int(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_duration_compact(seconds: float | int | None) -> str:
    text = _fmt_duration(seconds)
    if text.startswith("00:"):
        return text[3:]
    return text


def _resolve_resource_sample_interval_sec(environ) -> float:
    raw_value = _env_value_for_display(environ, "OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC", "2.0")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 2.0
    return max(0.5, min(60.0, value))


def _resolve_parallel_worker_prep_cache_max_items(environ) -> int:
    raw_value = _env_value_for_display(environ, "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS", "32")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 32
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
        raw_value = os.environ.get("OPTIMIZER_RESOURCE_DISK_MBPS_CAP", "500")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 500.0
        return max(1.0, min(10000.0, value))

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
        except Exception:
            pass
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
            except Exception:
                continue
            try:
                io = proc.io_counters()
                read_bytes += int(getattr(io, "read_bytes", 0) or 0)
                write_bytes += int(getattr(io, "write_bytes", 0) or 0)
            except Exception:
                pass
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
        except Exception:
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
        except Exception:
            pass
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
            except Exception:
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
            except Exception:
                continue
            try:
                with open(f"/proc/{pid}/io", "r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if line.startswith("read_bytes:"):
                            read_bytes += int(line.split()[1])
                        elif line.startswith("write_bytes:"):
                            write_bytes += int(line.split()[1])
            except Exception:
                pass
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
            cpu_percent = 0.0
        try:
            mem = psutil.virtual_memory()
            memory_percent = float(getattr(mem, "percent", 0.0) or 0.0)
            memory_available_gb = float(getattr(mem, "available", 0) or 0) / (1024 ** 3)
            memory_used_gb = float(getattr(mem, "used", 0) or 0) / (1024 ** 3)
        except Exception:
            memory_percent = 0.0
            memory_available_gb = 0.0
            memory_used_gb = 0.0
        try:
            swap = psutil.swap_memory()
            swap_percent = float(getattr(swap, "percent", 0.0) or 0.0)
            swap_used_gb = float(getattr(swap, "used", 0) or 0) / (1024 ** 3)
        except Exception:
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
        except Exception:
            pass

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

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", str(text))


def _visible_len(text: str) -> int:
    return len(_strip_ansi(str(text)))


def _pad_ansi(text: str, width: int, *, align: str = "<") -> str:
    raw = str(text)
    pad = max(0, int(width) - _visible_len(raw))
    if align == ">":
        return " " * pad + raw
    return raw + " " * pad


def _color_numeric_text(text: str, value: float | int | None) -> str:
    if value is None:
        return str(text)
    v = _safe_float(value, 0.0)
    if v > 0.0:
        return f"{C_GREEN}{text}{C_RESET}"
    if v < 0.0:
        return f"{C_RED}{text}{C_RESET}"
    return str(text)


def _format_score(value) -> str:
    v = _safe_float(value, 0.0)
    return _color_numeric_text(f"{v:.{OOS_SCORE_DECIMALS}f}", v)


def _format_compare(reference_score, rank_1_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    rank_1 = _safe_float(rank_1_score, 0.0)
    gap = rank_1 - ref
    return f"{_color_numeric_text(f'{ref:.{OOS_SCORE_DECIMALS}f}', ref)} ({_color_numeric_text(f'{gap:+.{OOS_SCORE_DECIMALS}f}', gap)})"


def _format_compare_plain(reference_score, rank_1_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    rank_1 = _safe_float(rank_1_score, 0.0)
    return f"{ref:.{OOS_SCORE_DECIMALS}f} ({rank_1 - ref:+.{OOS_SCORE_DECIMALS}f})"


def _prompt_int(label: str, default: int, *, minimum: int | None = None) -> int:
    if not is_interactive_console():
        return int(default)
    raw = input(f"{label:<28} [{int(default)}] : ").strip()
    if raw == "":
        value = int(default)
    else:
        value = int(raw)
    if minimum is not None and value < int(minimum):
        raise ValueError(f"{label} 必須 >= {minimum}，收到: {value}")
    return int(value)


def _prompt_str(label: str, default: str, *, allowed: tuple[str, ...] | None = None) -> str:
    if not is_interactive_console():
        return str(default)
    raw = input(f"{label:<28} [{default}] : ").strip()
    value = str(default if raw == "" else raw).strip().lower()
    if allowed is not None and value not in allowed:
        raise ValueError(f"{label} 必須是 {allowed}，收到: {value}")
    return value


def _extract_cli_value(argv, option_name: str) -> str:
    args = list(argv or [])
    for idx in range(1, len(args)):
        raw = str(args[idx]).strip()
        if raw == option_name and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if raw.startswith(option_name + "="):
            return raw.split("=", 1)[1].strip()
    return ""


def _has_cli_flag(argv, option_name: str) -> bool:
    return any(str(arg).strip() == option_name for arg in list(argv or [])[1:])


def _resolve_latest_year_from_dates(dates) -> int | None:
    years = []
    for raw_date in list(dates or []):
        year = int(getattr(raw_date, "year", 0) or 0)
        if year:
            years.append(year)
    return max(years) if years else None


def _resolve_latest_year_from_csv_data_dir(data_dir: str) -> int | None:
    # AI註: outer rolling OOS 的互動設定只需要 last OOS 預設值；
    # 不應為此先觸發 optimizer 完整資料清洗、快取摘要與 issue log。
    if not os.path.isdir(str(data_dir)):
        return None
    try:
        from core.data_utils import discover_unique_csv_inputs
        csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(str(data_dir))
    except (OSError, ValueError, TypeError):
        return None

    latest_year = None
    date_column_names = {"date", "datetime", "time", "timestamp", "日期"}
    for _ticker, file_path in list(csv_inputs or []):
        try:
            columns = list(pd.read_csv(file_path, nrows=0).columns)
            date_col = next((col for col in columns if str(col).strip().lower() in date_column_names), None)
            if date_col is None:
                continue
            date_values = pd.read_csv(file_path, usecols=[date_col])[date_col]
            if date_values.empty:
                continue
            parsed_dates = pd.to_datetime(date_values, errors="coerce")
            if parsed_dates.isna().all():
                continue
            file_year = int(parsed_dates.dt.year.max())
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError):
            continue
        if latest_year is None or file_year > latest_year:
            latest_year = file_year
    return latest_year


def _resolve_config(argv, environ, *, base_policy: dict, latest_year: int | None, default_trials: int, timing_mode: bool = False) -> OuterRollingConfig:
    env = os.environ if environ is None else environ
    train_start_default = int(base_policy.get("train_start_year", 2016) or 2016)
    first_oos_default = int(base_policy.get("oos_start_year") or base_policy.get("search_train_end_year", train_start_default + 4) + 1)
    last_oos_default = int(latest_year or first_oos_default)
    trials_default = int(default_trials if int(default_trials or 0) > 0 else int(env.get("V16_OUTER_ROLLING_OOS_TRIALS", "500") or 500))
    window_mode_default = str(env.get("V16_OUTER_ROLLING_WINDOW_MODE", "fixed") or "fixed").strip().lower()
    if window_mode_default not in ("fixed", "expanding"):
        window_mode_default = "fixed"
    train_window_default = max(1, int(env.get("V16_OUTER_ROLLING_TRAIN_WINDOW_YEARS", "5") or 5))

    cli_train_start = _extract_cli_value(argv, "--outer-train-start")
    cli_first = _extract_cli_value(argv, "--outer-first-oos")
    cli_last = _extract_cli_value(argv, "--outer-last-oos")
    cli_trials = _extract_cli_value(argv, "--trials")
    cli_window_mode = _extract_cli_value(argv, "--outer-window-mode")
    cli_train_window_years = _extract_cli_value(argv, "--outer-train-window-years")

    if cli_window_mode:
        window_mode = str(cli_window_mode).strip().lower()
        if window_mode not in ("fixed", "expanding"):
            raise ValueError("--outer-window-mode 必須是 fixed 或 expanding")
    elif bool(timing_mode):
        window_mode = window_mode_default
    else:
        window_mode = _prompt_str("window mode", window_mode_default, allowed=("fixed", "expanding"))

    if cli_train_window_years:
        train_window_years = int(cli_train_window_years)
    elif bool(timing_mode):
        train_window_years = train_window_default
    else:
        train_window_years = _prompt_int("train window years", train_window_default, minimum=1)

    if cli_train_start:
        train_start = int(cli_train_start)
    elif str(env.get("V16_OUTER_ROLLING_TRAIN_START", "")).strip():
        train_start = int(str(env["V16_OUTER_ROLLING_TRAIN_START"]).strip())
    elif bool(timing_mode):
        train_start = train_start_default
    else:
        train_start = _prompt_int("training start year", train_start_default, minimum=1900)

    if cli_first:
        first_oos = int(cli_first)
    elif str(env.get("V16_OUTER_ROLLING_FIRST_OOS", "")).strip():
        first_oos = int(str(env["V16_OUTER_ROLLING_FIRST_OOS"]).strip())
    elif bool(timing_mode):
        first_oos = first_oos_default
    else:
        first_oos = _prompt_int("first OOS year", first_oos_default, minimum=train_start + 1)

    if cli_last:
        last_oos = int(cli_last)
    elif str(env.get("V16_OUTER_ROLLING_LAST_OOS", "")).strip():
        last_oos = int(str(env["V16_OUTER_ROLLING_LAST_OOS"]).strip())
    elif bool(timing_mode):
        last_oos = last_oos_default
    else:
        last_oos = _prompt_int("last OOS year", last_oos_default, minimum=first_oos)

    if cli_trials:
        trials = int(cli_trials)
    elif bool(timing_mode):
        trials = trials_default
    else:
        trials = _prompt_int("optimizer trials per fold", trials_default, minimum=1)

    if first_oos > last_oos:
        raise ValueError("first OOS year 不可大於 last OOS year")
    if train_start >= first_oos:
        raise ValueError("training start year 必須小於 first OOS year")
    if train_window_years <= 0:
        raise ValueError("train window years 必須大於 0")
    if window_mode == "fixed" and first_oos - train_window_years < train_start:
        raise ValueError("fixed window 下 first OOS year - train_window_years 不可早於 training start year")
    if trials <= 0:
        raise ValueError("optimizer trials per fold 必須大於 0")
    return OuterRollingConfig(
        train_start,
        first_oos,
        last_oos,
        trials,
        window_mode=window_mode,
        train_window_years=train_window_years,
        confirm=not _has_cli_flag(argv, "--yes"),
    )


def _selection_start_for_oos(config: OuterRollingConfig, oos_year: int) -> int:
    if str(config.window_mode).lower() == "fixed":
        return int(oos_year) - int(config.train_window_years)
    return int(config.training_start_year)


def _print_plan(config: OuterRollingConfig, *, parallel_settings_line: str | None = None):
    years = list(range(config.first_oos_year, config.last_oos_year + 1))
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print("OUTER ROLLING OOS TEST | VERY NEXT 1 YEAR")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")
    print(f"window mode      : {config.window_mode}")
    if str(config.window_mode).lower() == "fixed":
        print(f"train window     : {config.train_window_years} years")
    else:
        print(f"training start   : {config.training_start_year}")
    print("oos feedback     : False")
    print("promotion        : disabled")
    print("oos horizon      : next 1 year only")
    print(f"optimizer trials : {config.trials_per_fold} per fold")
    if parallel_settings_line:
        print(str(parallel_settings_line))
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    print(f"{'fold':<6} | {'selection period':<18} | {'OOS test period':<15}")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    for idx, oos_year in enumerate(years, start=1):
        selection_start = _selection_start_for_oos(config, oos_year)
        print(f"{idx}/{len(years):<4} | {selection_start}~{oos_year - 1:<13} | {oos_year}")
    print(f"{C_GRAY}{'-' * 100}{C_RESET}")
    print(f"LOCAL_MIN_SCORE              : True")
    print(f"INNER_VALIDATE_RANK          : {bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED)}")
    print(f"DOMINANT_YEAR_DEPENDENCY     : {bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED)}")
    print(f"{C_CYAN}{'=' * 100}{C_RESET}")


def _confirm_plan(config: OuterRollingConfig) -> bool:
    if not config.confirm or not is_interactive_console():
        return True
    choice = safe_prompt_choice("開始執行 outer rolling OOS？[Y/n] : ", "Y", ("Y", "N"), "outer rolling OOS 確認")
    return choice.upper() == "Y"


def _profile_avg_float(profile_summary: dict, key: str) -> float:
    avg = dict((profile_summary or {}).get("avg") or {})
    try:
        return float(avg.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_outer_timing_row(
    *,
    fold_idx: int,
    fold_count: int,
    oos_year: int,
    selection_start: int,
    selection_end: int,
    status: str,
    session,
    db_file: str,
    install_shared_cache_sec: float,
    study_create_sec: float,
    optimize_sec: float,
    local_min_review_sec: float,
    oos_diagnostics_sec: float,
    fold_total_sec: float,
    finalists_count: int = 0,
) -> dict:
    profile_summary = session.profile_recorder.build_summary_payload()
    completed_trials = int(getattr(session, "current_session_trial", 0) or 0)
    get_prep_cache_stats = getattr(session, "get_prep_cache_stats", None)
    prep_cache_stats = get_prep_cache_stats() if callable(get_prep_cache_stats) else {}
    get_local_min_stats = getattr(session, "get_local_min_review_stats", None)
    local_min_stats = get_local_min_stats() if callable(get_local_min_stats) else {}
    return {
        "fold": f"{int(fold_idx)}/{int(fold_count)}",
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "oos_year": int(oos_year),
        "selection_period": f"{int(selection_start)}~{int(selection_end)}",
        "status": str(status),
        "requested_trials": int(getattr(session, "n_trials", 0) or 0),
        "completed_trials": completed_trials,
        "finalists_count": int(finalists_count),
        "db_file": str(db_file or "memory"),
        "profile_csv_path": str(session.profile_recorder.csv_path if getattr(session.profile_recorder, "write_files", True) else ""),
        "profile_summary_path": str(session.profile_recorder.summary_path if getattr(session.profile_recorder, "write_files", True) else ""),
        "profile_write_files": bool(getattr(session.profile_recorder, "write_files", True)),
        "study_storage": "sqlite" if db_file else "memory",
        "parallel_fold_log_mode": "progress_only",
        "install_shared_cache_sec": float(install_shared_cache_sec),
        "study_create_sec": float(study_create_sec),
        "optimize_sec": float(optimize_sec),
        "local_min_review_sec": float(local_min_review_sec),
        "oos_diagnostics_sec": float(oos_diagnostics_sec),
        "fold_total_sec": float(fold_total_sec),
        "avg_trial_total_wall_sec": _profile_avg_float(profile_summary, "trial_total_wall_sec"),
        "avg_objective_wall_sec": _profile_avg_float(profile_summary, "objective_wall_sec"),
        "avg_prep_wall_sec": _profile_avg_float(profile_summary, "prep_wall_sec"),
        "avg_portfolio_wall_sec": _profile_avg_float(profile_summary, "portfolio_wall_sec"),
        "avg_score_calc_sec": _profile_avg_float(profile_summary, "score_calc_sec"),
        "avg_filter_rules_sec": _profile_avg_float(profile_summary, "filter_rules_sec"),
        "first_trial_completed_wall_sec": profile_summary.get("first_trial_completed_wall_sec"),
        "prep_cache_hits": int(prep_cache_stats.get("hits", 0) or 0),
        "prep_cache_misses": int(prep_cache_stats.get("misses", 0) or 0),
        "prep_cache_stores": int(prep_cache_stats.get("stores", 0) or 0),
        "prep_cache_evictions": int(prep_cache_stats.get("evictions", 0) or 0),
        "prep_cache_items": int(prep_cache_stats.get("items", 0) or 0),
        "prep_cache_max_items": int(prep_cache_stats.get("max_items", 0) or 0),
        "prep_cache_shared": bool(prep_cache_stats.get("shared", False)),
        "prep_executor_created": int(prep_cache_stats.get("executor_created", 0) or 0),
        "prep_executor_reused": int(prep_cache_stats.get("executor_reused", 0) or 0),
        "prep_executor_shared": bool(prep_cache_stats.get("executor_shared", False)),
        "prep_call_count": int(prep_cache_stats.get("prep_call_count", 0) or 0),
        "prep_wall_sum_sec": float(prep_cache_stats.get("prep_wall_sum_sec", 0.0) or 0.0),
        "prep_worker_total_sum_sec": float(prep_cache_stats.get("prep_worker_total_sum_sec", 0.0) or 0.0),
        "prep_total_sum_sec": float(prep_cache_stats.get("prep_total_sum_sec", 0.0) or 0.0),
        "prep_copy_sum_sec": float(prep_cache_stats.get("prep_copy_sum_sec", 0.0) or 0.0),
        "prep_generate_signals_sum_sec": float(prep_cache_stats.get("prep_generate_signals_sum_sec", 0.0) or 0.0),
        "prep_assign_sum_sec": float(prep_cache_stats.get("prep_assign_sum_sec", 0.0) or 0.0),
        "prep_run_backtest_sum_sec": float(prep_cache_stats.get("prep_run_backtest_sum_sec", 0.0) or 0.0),
        "prep_to_dict_sum_sec": float(prep_cache_stats.get("prep_to_dict_sum_sec", 0.0) or 0.0),
        "prep_static_pack_sum_sec": float(prep_cache_stats.get("prep_static_pack_sum_sec", 0.0) or 0.0),
        "prep_executor_setup_sum_sec": float(prep_cache_stats.get("prep_executor_setup_sum_sec", 0.0) or 0.0),
        "prep_executor_submit_sum_sec": float(prep_cache_stats.get("prep_executor_submit_sum_sec", 0.0) or 0.0),
        "prep_executor_collect_sum_sec": float(prep_cache_stats.get("prep_executor_collect_sum_sec", 0.0) or 0.0),
        "prep_executor_shutdown_sum_sec": float(prep_cache_stats.get("prep_executor_shutdown_sum_sec", 0.0) or 0.0),
        "prep_merge_sum_sec": float(prep_cache_stats.get("prep_merge_sum_sec", 0.0) or 0.0),
        "prep_master_union_sum_sec": float(prep_cache_stats.get("prep_master_union_sum_sec", 0.0) or 0.0),
        "prep_ticker_count_sum": int(prep_cache_stats.get("prep_ticker_count_sum", 0) or 0),
        "prep_batch_count_sum": int(prep_cache_stats.get("prep_batch_count_sum", 0) or 0),
        "prep_ok_count_sum": int(prep_cache_stats.get("prep_ok_count_sum", 0) or 0),
        "prep_fail_count_sum": int(prep_cache_stats.get("prep_fail_count_sum", 0) or 0),
        "prep_feature_bank_hits": int(prep_cache_stats.get("prep_feature_bank_hits", 0) or 0),
        "prep_feature_bank_misses": int(prep_cache_stats.get("prep_feature_bank_misses", 0) or 0),
        "prep_feature_bank_size_max": int(prep_cache_stats.get("prep_feature_bank_size_max", 0) or 0),
        "prep_feature_bank_max_items": int(prep_cache_stats.get("prep_feature_bank_max_items", 0) or 0),
        "local_min_neighbors_total": int(local_min_stats.get("total_neighbors", 0) or 0),
        "local_min_neighbors_evaluated": int(local_min_stats.get("evaluated_neighbors", 0) or 0),
        "local_min_neighbors_skipped": int(local_min_stats.get("skipped_neighbors", 0) or 0),
        "local_min_payload_score_cache_hits": int(local_min_stats.get("payload_score_cache_hits", 0) or 0),
        "local_min_prep_cache_prioritized": int(local_min_stats.get("prep_cache_prioritized", 0) or 0),
        "local_min_order_score_prioritized": int(local_min_stats.get("order_score_prioritized", 0) or 0),
        "local_min_field_order_score_prioritized": int(local_min_stats.get("field_order_score_prioritized", 0) or 0),
        "local_min_early_stops": int(local_min_stats.get("early_stops", 0) or 0),
        "local_min_selection_prunes": int(local_min_stats.get("selection_prunes", 0) or 0),
        "local_min_parallel_workers_max": int(local_min_stats.get("parallel_workers_max", 0) or 0),
        "local_min_parallel_submitted": int(local_min_stats.get("parallel_submitted", 0) or 0),
        "local_min_parallel_completed": int(local_min_stats.get("parallel_completed", 0) or 0),
        "local_min_parallel_cancelled": int(local_min_stats.get("parallel_cancelled", 0) or 0),
        "local_min_hard_fail_stops": int(local_min_stats.get("hard_fail_stops", 0) or 0),
        "local_min_hard_fail_neighbors_skipped": int(local_min_stats.get("hard_fail_neighbors_skipped", 0) or 0),
        "local_min_hard_fail_cancelled": int(local_min_stats.get("hard_fail_cancelled", 0) or 0),
        "rolling_fold_workers_max": int(getattr(session, "rolling_fold_workers_max", 1) or 1),
        "rolling_fold_parallel": bool(getattr(session, "rolling_fold_parallel", False)),
    }


def _sum_timing_rows(rows: list[dict], key: str) -> float:
    total = 0.0
    for row in list(rows or []):
        try:
            total += float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _write_outer_timing_summary(
    *,
    output_dir: str,
    session_ts: str,
    dataset_label: str,
    config: OuterRollingConfig,
    timing_mode: bool,
    optimizer_seed,
    raw_data_load_sec: float,
    active_replay_chain_sec: float,
    report_write_sec: float,
    overall_sec: float,
    fold_timing_rows: list[dict],
    resource_summary: dict | None = None,
    resource_samples: list[dict] | None = None,
) -> dict:
    report_dir = os.path.join(output_dir, "outer_rolling_oos")
    os.makedirs(report_dir, exist_ok=True)
    base = os.path.join(report_dir, f"outer_rolling_oos_timing_{session_ts}")
    csv_path = base + ".csv"
    json_path = base + ".json"
    resource_write_csv = _env_flag(os.environ, "OPTIMIZER_RESOURCE_WRITE_CSV", False)
    resource_csv_path = os.path.join(report_dir, f"outer_rolling_oos_resource_{session_ts}.csv") if resource_write_csv else ""

    if fold_timing_rows:
        fieldnames = list(fold_timing_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fold_timing_rows)

    samples_for_csv = list(resource_samples or [])
    if resource_write_csv and samples_for_csv:
        sample_fieldnames = list(samples_for_csv[0].keys())
        with open(resource_csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
            writer.writeheader()
            writer.writerows(samples_for_csv)
    else:
        resource_csv_path = ""

    optimize_sec = _sum_timing_rows(fold_timing_rows, "optimize_sec")
    local_min_review_sec = _sum_timing_rows(fold_timing_rows, "local_min_review_sec")
    oos_diagnostics_sec = _sum_timing_rows(fold_timing_rows, "oos_diagnostics_sec")
    install_shared_cache_sec = _sum_timing_rows(fold_timing_rows, "install_shared_cache_sec")
    study_create_sec = _sum_timing_rows(fold_timing_rows, "study_create_sec")
    fold_total_sec = _sum_timing_rows(fold_timing_rows, "fold_total_sec")
    completed_trials = sum(int(row.get("completed_trials", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_hits = sum(int(row.get("prep_cache_hits", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_misses = sum(int(row.get("prep_cache_misses", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_stores = sum(int(row.get("prep_cache_stores", 0) or 0) for row in list(fold_timing_rows or []))
    prep_cache_evictions = sum(int(row.get("prep_cache_evictions", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_neighbors_total = sum(int(row.get("local_min_neighbors_total", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_neighbors_evaluated = sum(int(row.get("local_min_neighbors_evaluated", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_payload_score_cache_hits = sum(int(row.get("local_min_payload_score_cache_hits", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_prep_cache_prioritized = sum(int(row.get("local_min_prep_cache_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_order_score_prioritized = sum(int(row.get("local_min_order_score_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_field_order_score_prioritized = sum(int(row.get("local_min_field_order_score_prioritized", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_early_stops = sum(int(row.get("local_min_early_stops", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_selection_prunes = sum(int(row.get("local_min_selection_prunes", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_parallel_workers_max = max((int(row.get("local_min_parallel_workers_max", 0) or 0) for row in list(fold_timing_rows or [])), default=0)
    local_min_parallel_submitted = sum(int(row.get("local_min_parallel_submitted", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_parallel_completed = sum(int(row.get("local_min_parallel_completed", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_parallel_cancelled = sum(int(row.get("local_min_parallel_cancelled", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_hard_fail_stops = sum(int(row.get("local_min_hard_fail_stops", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_hard_fail_neighbors_skipped = sum(int(row.get("local_min_hard_fail_neighbors_skipped", 0) or 0) for row in list(fold_timing_rows or []))
    local_min_hard_fail_cancelled = sum(int(row.get("local_min_hard_fail_cancelled", 0) or 0) for row in list(fold_timing_rows or []))
    rolling_fold_workers_max = max((int(row.get("rolling_fold_workers_max", 1) or 1) for row in list(fold_timing_rows or [])), default=1)
    rolling_fold_parallel = any(bool(row.get("rolling_fold_parallel", False)) for row in list(fold_timing_rows or []))
    prep_executor_created = sum(int(row.get("prep_executor_created", 0) or 0) for row in list(fold_timing_rows or []))
    prep_executor_reused = sum(int(row.get("prep_executor_reused", 0) or 0) for row in list(fold_timing_rows or []))
    prep_call_count = sum(int(row.get("prep_call_count", 0) or 0) for row in list(fold_timing_rows or []))
    prep_ticker_count_sum = sum(int(row.get("prep_ticker_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_batch_count_sum = sum(int(row.get("prep_batch_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_ok_count_sum = sum(int(row.get("prep_ok_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_fail_count_sum = sum(int(row.get("prep_fail_count_sum", 0) or 0) for row in list(fold_timing_rows or []))
    prep_feature_bank_hits = sum(int(row.get("prep_feature_bank_hits", 0) or 0) for row in list(fold_timing_rows or []))
    prep_feature_bank_misses = sum(int(row.get("prep_feature_bank_misses", 0) or 0) for row in list(fold_timing_rows or []))
    prep_feature_bank_size_max = max((int(row.get("prep_feature_bank_size_max", 0) or 0) for row in list(fold_timing_rows or [])), default=0)
    prep_feature_bank_max_items = max((int(row.get("prep_feature_bank_max_items", 0) or 0) for row in list(fold_timing_rows or [])), default=0)
    payload = {
        "type": "outer_rolling_oos_timing",
        "version": 1,
        "created_at": get_taipei_now().isoformat(),
        "timing_mode": bool(timing_mode),
        "dataset_label": str(dataset_label),
        "optimizer_seed": optimizer_seed,
        "sampler_kind": "random" if bool(timing_mode) else "tpe",
        "meta": {
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "trials_per_fold": int(config.trials_per_fold),
            "fold_count": len(fold_timing_rows),
            "completed_trials": int(completed_trials),
        },
        "summary": {
            "overall_sec": float(overall_sec),
            "raw_data_load_once_sec": float(raw_data_load_sec),
            "install_shared_cache_sum_sec": float(install_shared_cache_sec),
            "study_create_sum_sec": float(study_create_sec),
            "optimize_sum_sec": float(optimize_sec),
            "local_min_review_sum_sec": float(local_min_review_sec),
            "oos_diagnostics_sum_sec": float(oos_diagnostics_sec),
            "active_replay_chain_sec": float(active_replay_chain_sec),
            "report_write_sec": float(report_write_sec),
            "fold_total_sum_sec": float(fold_total_sec),
            "avg_optimize_sec_per_completed_trial": (float(optimize_sec) / float(completed_trials)) if completed_trials > 0 else 0.0,
            "avg_fold_total_sec": (float(fold_total_sec) / float(len(fold_timing_rows))) if fold_timing_rows else 0.0,
            "fold_wall_sec": (max((float(row.get("fold_total_sec", 0.0) or 0.0) for row in list(fold_timing_rows or [])), default=0.0) if bool(rolling_fold_parallel) else float(fold_total_sec)),
            "other_overhead_sec": max(0.0, float(overall_sec) - (max((float(row.get("fold_total_sec", 0.0) or 0.0) for row in list(fold_timing_rows or [])), default=0.0) if bool(rolling_fold_parallel) else float(fold_total_sec)) - float(active_replay_chain_sec) - float(raw_data_load_sec) - float(report_write_sec)),
            "prep_cache_hits": int(prep_cache_hits),
            "prep_cache_misses": int(prep_cache_misses),
            "prep_cache_stores": int(prep_cache_stores),
            "prep_cache_evictions": int(prep_cache_evictions),
            "prep_cache_hit_rate": (float(prep_cache_hits) / float(prep_cache_hits + prep_cache_misses)) if (prep_cache_hits + prep_cache_misses) > 0 else 0.0,
            "prep_call_count": int(prep_call_count),
            "prep_wall_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_wall_sum_sec")),
            "prep_worker_total_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_worker_total_sum_sec")),
            "prep_total_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_total_sum_sec")),
            "prep_copy_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_copy_sum_sec")),
            "prep_generate_signals_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_generate_signals_sum_sec")),
            "prep_assign_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_assign_sum_sec")),
            "prep_run_backtest_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_run_backtest_sum_sec")),
            "prep_to_dict_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_to_dict_sum_sec")),
            "prep_static_pack_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_static_pack_sum_sec")),
            "prep_executor_setup_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_setup_sum_sec")),
            "prep_executor_submit_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_submit_sum_sec")),
            "prep_executor_collect_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_collect_sum_sec")),
            "prep_executor_shutdown_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_executor_shutdown_sum_sec")),
            "prep_merge_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_merge_sum_sec")),
            "prep_master_union_sum_sec": float(_sum_timing_rows(fold_timing_rows, "prep_master_union_sum_sec")),
            "prep_ticker_count_sum": int(prep_ticker_count_sum),
            "prep_batch_count_sum": int(prep_batch_count_sum),
            "prep_ok_count_sum": int(prep_ok_count_sum),
            "prep_fail_count_sum": int(prep_fail_count_sum),
            "prep_feature_bank_hits": int(prep_feature_bank_hits),
            "prep_feature_bank_misses": int(prep_feature_bank_misses),
            "prep_feature_bank_hit_rate": (float(prep_feature_bank_hits) / float(prep_feature_bank_hits + prep_feature_bank_misses)) if (prep_feature_bank_hits + prep_feature_bank_misses) > 0 else 0.0,
            "prep_feature_bank_size_max": int(prep_feature_bank_size_max),
            "prep_feature_bank_max_items": int(prep_feature_bank_max_items),
            "local_min_neighbors_total": int(local_min_neighbors_total),
            "local_min_neighbors_evaluated": int(local_min_neighbors_evaluated),
            "local_min_neighbors_skipped": max(0, int(local_min_neighbors_total) - int(local_min_neighbors_evaluated)),
            "local_min_payload_score_cache_hits": int(local_min_payload_score_cache_hits),
            "local_min_prep_cache_prioritized": int(local_min_prep_cache_prioritized),
            "local_min_order_score_prioritized": int(local_min_order_score_prioritized),
            "local_min_field_order_score_prioritized": int(local_min_field_order_score_prioritized),
            "local_min_early_stops": int(local_min_early_stops),
            "local_min_selection_prunes": int(local_min_selection_prunes),
            "local_min_parallel_workers_max": int(local_min_parallel_workers_max),
            "local_min_parallel_submitted": int(local_min_parallel_submitted),
            "local_min_parallel_completed": int(local_min_parallel_completed),
            "local_min_parallel_cancelled": int(local_min_parallel_cancelled),
            "local_min_hard_fail_stops": int(local_min_hard_fail_stops),
            "local_min_hard_fail_neighbors_skipped": int(local_min_hard_fail_neighbors_skipped),
            "local_min_hard_fail_cancelled": int(local_min_hard_fail_cancelled),
            "rolling_fold_workers_max": int(rolling_fold_workers_max),
            "rolling_fold_parallel": bool(rolling_fold_parallel),
            "profile_write_files": str(os.environ.get("OPTIMIZER_PROFILE_WRITE_FILES", "1")).strip().lower() not in {"0", "false", "no", "off"},
            "study_storage": "sqlite" if _is_outer_rolling_sqlite_storage_enabled(os.environ) else "memory",
            "parallel_fold_log_mode": "progress_only" if bool(rolling_fold_parallel) else "normal",
            "parallel_worker_prep_cache_max_items": _resolve_parallel_worker_prep_cache_max_items(os.environ),
            "resource_write_csv": bool(resource_write_csv),
            "resource_csv_rows": int(len(samples_for_csv)) if resource_write_csv and samples_for_csv else 0,
            "active_replay_include_trade_logs": _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS", False),
            "active_replay_include_pit_stats_index": _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX", True),
            "active_replay_use_prepared_cache": _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE", False),
            "active_replay_write_prepared_cache": _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE", False),
            **dict(resource_summary or {}),
            "prep_executor_created": int(prep_executor_created),
            "prep_executor_reused": int(prep_executor_reused),
        },
        "folds": fold_timing_rows,
        "csv_path": csv_path if fold_timing_rows else "",
        "resource_csv_path": resource_csv_path,
        "json_path": json_path,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {"json": json_path, "csv": csv_path if fold_timing_rows else "", "resource_csv": resource_csv_path, "payload": payload}


def _format_resource_usage_line(summary: dict, *, prefix: str = "📏 效能摘要:") -> str:
    summary = dict(summary or {})
    if not bool(summary.get("resource_sampling_available", False)):
        err = str(summary.get("resource_sampling_error", "") or "unavailable")
        return f"{prefix} {C_CYAN}CPU avg = N/A｜MEM avg = N/A｜HD avg = N/A｜resource={err}{C_RESET}"
    cpu_avg = float(summary.get("cpu_avg_percent", 0.0) or 0.0)
    mem_avg = float(summary.get("memory_avg_percent", 0.0) or 0.0)
    hd_avg = float(summary.get("disk_load_avg_percent", summary.get("disk_busy_avg_percent", 0.0)) or 0.0)
    return f"{prefix} {C_CYAN}CPU avg = {cpu_avg:.1f}%｜MEM avg = {mem_avg:.1f}%｜HD avg = {hd_avg:.1f}%{C_RESET}"


def _timing_phase_summary_from_payload(payload: dict) -> dict:
    payload = dict(payload or {})
    summary = dict(payload.get("summary") or {})
    folds = list(payload.get("folds") or [])
    rolling_parallel = bool(summary.get("rolling_fold_parallel", False))
    if rolling_parallel:
        fold_wall = max((float(row.get("fold_total_sec", 0.0) or 0.0) for row in folds), default=0.0)
    else:
        fold_wall = float(summary.get("fold_total_sum_sec", 0.0) or 0.0)
    total = float(summary.get("overall_sec", 0.0) or 0.0)
    chain = float(summary.get("active_replay_chain_sec", 0.0) or 0.0)
    raw = float(summary.get("raw_data_load_once_sec", 0.0) or 0.0)
    report = float(summary.get("report_write_sec", 0.0) or 0.0)
    other = max(0.0, total - fold_wall - chain - raw - report)
    return {
        "total_sec": total,
        "fold_wall_sec": fold_wall,
        "active_replay_chain_sec": chain,
        "raw_data_load_once_sec": raw,
        "report_write_sec": report,
        "other_overhead_sec": other,
    }


def _format_timing_phase_line(payload: dict) -> str:
    phases = _timing_phase_summary_from_payload(payload)
    return (
        f"{C_CYAN}⏱️ 耗時對帳｜"
        f"total={_fmt_duration(phases['total_sec'])}｜"
        f"folds={_fmt_duration(phases['fold_wall_sec'])}｜"
        f"OOS_CHAIN={_fmt_duration(phases['active_replay_chain_sec'])}｜"
        f"setup/收尾={_fmt_duration(phases['other_overhead_sec'])}{C_RESET}"
    )


def _print_outer_timing_summary(payload: dict):
    summary = dict((payload or {}).get("summary") or {})
    print(
        "📏 Outer rolling 測時摘要: "
        f"{C_CYAN}總時間={float(summary.get('overall_sec', 0.0)):.3f}s{C_RESET}｜"
        f"optimizer={float(summary.get('optimize_sum_sec', 0.0)):.3f}s｜"
        f"local_review={float(summary.get('local_min_review_sum_sec', 0.0)):.3f}s｜"
        f"平均={float(summary.get('avg_optimize_sec_per_completed_trial', 0.0)):.3f}s/completed trial"
    )
    print(
        "📏 Cache / local-min 摘要｜"
        f"prep_hit/miss/evict={int(summary.get('prep_cache_hits', 0) or 0)}/"
        f"{int(summary.get('prep_cache_misses', 0) or 0)}/"
        f"{int(summary.get('prep_cache_evictions', 0) or 0)}｜"
        f"hit_rate={float(summary.get('prep_cache_hit_rate', 0.0)):.1%}｜"
        f"parallel_cache_max={int(summary.get('parallel_worker_prep_cache_max_items', 0) or 0)}｜"
        f"local_neighbors={int(summary.get('local_min_neighbors_evaluated', 0) or 0)}/"
        f"{int(summary.get('local_min_neighbors_total', 0) or 0)}｜"
        f"skip={int(summary.get('local_min_neighbors_skipped', 0) or 0)}｜"
        f"order_hint={int(summary.get('local_min_order_score_prioritized', 0) or 0)}｜"
        f"field_hint={int(summary.get('local_min_field_order_score_prioritized', 0) or 0)}｜"
        f"early/prune={int(summary.get('local_min_early_stops', 0) or 0)}/"
        f"{int(summary.get('local_min_selection_prunes', 0) or 0)}"
    )
    print(
        "📏 Prep 細分摘要｜"
        f"calls={int(summary.get('prep_call_count', 0) or 0)}｜"
        f"wall={float(summary.get('prep_wall_sum_sec', 0.0) or 0.0):.3f}s｜"
        f"signals={float(summary.get('prep_generate_signals_sum_sec', 0.0) or 0.0):.3f}s｜"
        f"backtest={float(summary.get('prep_run_backtest_sum_sec', 0.0) or 0.0):.3f}s｜"
        f"pack={float(summary.get('prep_to_dict_sum_sec', 0.0) or 0.0):.3f}s｜"
        f"collect={float(summary.get('prep_executor_collect_sum_sec', 0.0) or 0.0):.3f}s｜"
        f"merge={float(summary.get('prep_merge_sum_sec', 0.0) or 0.0):.3f}s｜"
        f"feature_hit={float(summary.get('prep_feature_bank_hit_rate', 0.0) or 0.0):.1%}"
    )




class _SearchProgress:
    def __init__(self, *, fold_idx: int, fold_count: int, oos_year: int, selection_start: int, selection_end: int, total_trials: int, completed_results: list[dict], overall_start: float):
        self.fold_idx = int(fold_idx)
        self.fold_count = int(fold_count)
        self.oos_year = int(oos_year)
        self.selection_start = int(selection_start)
        self.selection_end = int(selection_end)
        self.total_trials = int(total_trials)
        self.completed_results = completed_results
        self.overall_start = float(overall_start)
        self.stage_start = time.perf_counter()
        self.best_score = float("-inf")
        self.last_render = 0.0
        self.inline_progress_enabled = stdout_supports_inline_progress()
        self.inline_progress_width = 0

    def _eta_stage(self, completed: int) -> float | None:
        if completed <= 0:
            return None
        elapsed = max(0.0, time.perf_counter() - self.stage_start)
        avg = elapsed / float(completed)
        return avg * max(0, self.total_trials - completed)

    def render(self, completed: int, *, force: bool = False):
        if not self.inline_progress_enabled:
            return
        now = time.perf_counter()
        if not force and now - self.last_render < 0.5 and completed < self.total_trials:
            return
        self.last_render = now
        pct = 100.0 * float(completed) / max(1, self.total_trials)
        eta_stage = self._eta_stage(completed)
        elapsed_total = now - self.overall_start
        eta_total = None
        done_folds = len(self.completed_results)
        if done_folds > 0:
            avg_done = sum(float(row.get("elapsed_sec", 0.0)) for row in self.completed_results) / float(done_folds)
            current_remaining = eta_stage or 0.0
            eta_total = current_remaining + avg_done * max(0, self.fold_count - self.fold_idx)
        best_score = self.best_score if self.best_score != float("-inf") else 0.0
        elapsed_text = _fmt_duration_compact(now - self.stage_start)
        eta_stage_text = _fmt_duration_compact(eta_stage)
        eta_total_text = _fmt_duration_compact(eta_total)
        line = choose_inline_progress_message((
            (
                f"[{self.fold_idx}/{self.fold_count}] selection={self.selection_start}~{self.selection_end} | OOS={self.oos_year} | "
                f"OPTIMIZER_SEARCH | 進度={completed}/{self.total_trials} ({pct:5.1f}%) | "
                f"best_score={best_score:.3f} | elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{self.fold_idx}/{self.fold_count}] selection={self.selection_start % 100:02d}~{self.selection_end % 100:02d} | OOS={self.oos_year % 100:02d} | "
                f"search | 進度={completed}/{self.total_trials} ({pct:5.1f}%) | "
                f"best={best_score:.3f} | elapsed={elapsed_text} | eta={eta_stage_text}/{eta_total_text}"
            ),
            (
                f"[{self.fold_idx}/{self.fold_count}] {self.selection_start % 100:02d}~{self.selection_end % 100:02d}>OOS{self.oos_year % 100:02d} | "
                f"search {completed}/{self.total_trials} | best={best_score:.3f} | eta={eta_stage_text}/{eta_total_text}"
            ),
        ))
        self.inline_progress_width = write_inline_progress(line, previous_width=self.inline_progress_width)

    def callback(self, session):
        def _callback(study, trial):
            session.current_session_trial += 1
            if trial.value is not None and is_qualified_trial_value(trial.value):
                self.best_score = max(self.best_score, float(trial.value))
            self.render(int(session.current_session_trial))
        return _callback

    def done(self, completed: int):
        if self.inline_progress_enabled:
            self.render(completed, force=True)
            print()


def _select_winner(finalists: list[dict], *, objective_mode: str):
    eligible = [item for item in finalists if bool(item.get("gate_pass", False))]
    if is_inner_validate_anti_overfit_enabled(objective_mode):
        eligible = [item for item in eligible if _has_inner_validate_pass(item)]
    if is_dominant_year_dependency_anti_overfit_enabled():
        safe = [item for item in eligible if not _has_dependency_warning(item)]
        if safe:
            eligible = safe
    if not eligible:
        return None
    eligible = sorted(
        eligible,
        key=lambda item: (
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("local_retention", float("-inf"))),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return eligible[0]


def _build_local_rank_map(finalists: list[dict]) -> dict[int, int]:
    return {int(item["trial"].number): rank for rank, item in enumerate(finalists or [], start=1) if item.get("trial") is not None}


def _build_retention_rank_map(finalists: list[dict]) -> dict[int, int]:
    ranked = sorted(
        [item for item in list(finalists or []) if item.get("trial") is not None],
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
        reverse=True,
    )
    return {int(item["trial"].number): rank for rank, item in enumerate(ranked, start=1)}


def _select_base_rank1_item(finalists: list[dict]):
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    if not items:
        return None
    return min(items, key=lambda item: (int(item.get("base_rank", 10**9) or 10**9), -float(item.get("base_score", INVALID_TRIAL_VALUE)), int(item["trial"].number)))


def _select_local_rank1_item(finalists: list[dict], *, objective_mode: str):
    winner = _select_winner(finalists, objective_mode=objective_mode)
    if winner is not None:
        return winner
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    return items[0] if items else None


def _select_retention_rank1_item(finalists: list[dict]):
    items = [item for item in list(finalists or []) if item.get("trial") is not None]
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            float(item.get("local_retention", float("-inf"))),
            float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
            float(item.get("base_score", INVALID_TRIAL_VALUE)),
            -int(item["trial"].number),
        ),
    )


def _build_policy_items(finalists: list[dict], *, objective_mode: str) -> dict[str, dict | None]:
    return {
        "base": _select_base_rank1_item(finalists),
        "local": _select_local_rank1_item(finalists, objective_mode=objective_mode),
        "retention": _select_retention_rank1_item(finalists),
    }


def _policy_description(policy_name: str) -> str:
    if policy_name == "base":
        return "Use base_rank #1 params for each OOS year."
    if policy_name == "local":
        return "Use local_rank #1 params for each OOS year."
    if policy_name == "retention":
        return "Use retention_rank #1 params for each OOS year."
    return f"Use {policy_name} params for each OOS year."


def _build_policy_schedule_entry(*, item: dict, policy_name: str, oos_year: int, selection_period: str, local_rank_map: dict[int, int], retention_rank_map: dict[int, int]) -> dict:
    trial = item["trial"]
    trial_number = int(trial.number)
    return {
        "effective_start": f"{int(oos_year)}-01-01",
        "effective_end": f"{int(oos_year)}-12-31",
        "selection": str(selection_period),
        "oos_year": int(oos_year),
        "policy": str(policy_name),
        "selected_trial": trial_number + 1,
        "base_score": float(item.get("base_score", INVALID_TRIAL_VALUE)),
        "base_rank": int(item.get("base_rank", 0) or 0),
        "local_min": float(item.get("local_min_score", INVALID_TRIAL_VALUE)),
        "local_min_exact": bool(item.get("local_min_exact", True)),
        "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
        "local_rank": int(local_rank_map.get(trial_number, 0)),
        "retention": float(item.get("local_retention", 0.0)),
        "retention_rank": int(retention_rank_map.get(trial_number, 0)),
        "params": build_best_params_payload_from_trial(trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT),
    }


def _prep_result_has_pit_index(prep_result) -> bool:
    if not isinstance(prep_result, dict):
        return False
    pit_stats_index = prep_result.get("all_pit_stats_index")
    return isinstance(pit_stats_index, dict) and bool(pit_stats_index)


def _get_or_prepare_oos_inputs(*, session, params):
    # AI註: OOS diagnostics only needs dynamic data + PIT stats index for
    # portfolio replay.  Standalone trade logs are unnecessary when the PIT
    # index is already available, so reuse the normal optimizer prep cache.
    prep_cache_key = build_prep_cache_key(params)
    get_cached_prep = getattr(session, "get_prepared_trial_inputs_from_cache", None)
    prep_result = get_cached_prep(prep_cache_key) if callable(get_cached_prep) else None
    if _prep_result_has_pit_index(prep_result):
        return prep_result

    prep_executor_bundle = session.get_trial_prep_executor_bundle(build_runtime_param_raw_value(params, "optimizer_max_workers"))
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
        include_trade_logs=False,
        include_pit_stats_index=True,
        profile_enabled=False,
    )
    cache_prep = getattr(session, "cache_prepared_trial_inputs", None)
    if callable(cache_prep):
        cache_prep(prep_cache_key, prep_result)
    return prep_result


def _evaluate_next_1y_oos(*, session, trial, oos_year: int, include_equity_curve: bool = False):
    payload = build_best_params_payload_from_trial(trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
    params = build_params_from_mapping(payload)
    prep_result = _get_or_prepare_oos_inputs(session=session, params=params)
    all_dates = sorted(prep_result["master_dates"])
    test_dates = [dt for dt in all_dates if int(getattr(dt, "year", 0) or 0) == int(oos_year)]
    if not test_dates:
        raise RuntimeError(f"OOS {oos_year} 無有效交易日期")
    holdout_period = {
        "label": f"OOS-{int(oos_year)}",
        "train_start": f"{int(session.train_start_year)}-01-01",
        "train_end": f"{int(oos_year) - 1}-12-31",
        "oos_start": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d"),
        "oos_end": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d"),
        "test_dates": test_dates,
    }
    return evaluate_walk_forward(
        all_dfs_fast=prep_result["all_dfs_fast"],
        all_trade_logs=prep_result["all_trade_logs"],
        sorted_dates=all_dates,
        params=params,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        benchmark_ticker="0050",
        train_start_year=int(session.train_start_year),
        min_train_years=int(getattr(session, "walk_forward_policy", {}).get("min_train_years", 1) or 1),
        oos_start_year=int(oos_year),
        pit_stats_index=prep_result.get("all_pit_stats_index"),
        holdout_period=holdout_period,
        include_equity_curve=bool(include_equity_curve),
    )


def _extract_period_metrics(report: dict) -> dict:
    period = dict(report.get("period") or {})
    return {
        "oos_score": float(period.get("test_score_romd", 0.0)),
        "ret_pct": float(period.get("ret_pct", 0.0)),
        "mdd_pct": float(period.get("mdd", 0.0)),
        "trades": int(period.get("trade_count", 0) or 0),
        "benchmark_oos_score": float(period.get("benchmark_score_romd", 0.0)),
        "benchmark_return_pct": float(period.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": float(period.get("benchmark_mdd", 0.0)),
        "initial_capital": float(period.get("initial_capital", 0.0)),
        "equity_curve": list(period.get("equity_curve") or []),
    }


def _format_oos_delta(reference_score, selected_score) -> str:
    ref = _safe_float(reference_score, 0.0)
    selected = _safe_float(selected_score, 0.0)
    return f"{ref:.{OOS_SCORE_DECIMALS}f} ({selected - ref:+.{OOS_SCORE_DECIMALS}f})"


def _evaluate_finalist_oos_diagnostics(*, session, finalists: list[dict], policy_items: dict[str, dict | None], oos_year: int) -> dict:
    best_score = float("-inf")
    best_trial_number = None
    best_trial = None
    best_metrics: dict = {}
    report_cache: dict[int, dict] = {}
    metrics_by_trial: dict[int, dict] = {}
    benchmark_score = 0.0
    benchmark_return_pct = 0.0
    benchmark_mdd_pct = 0.0
    curve_trial_numbers = {
        int(item["trial"].number)
        for item in dict(policy_items or {}).values()
        if item is not None and item.get("trial") is not None
    }

    def _load_trial_metrics(trial, *, include_equity_curve: bool) -> dict:
        trial_number = int(trial.number)
        cached_entry = report_cache.get(trial_number)
        report = None
        if cached_entry is not None:
            cached_has_curve = bool(cached_entry.get("include_equity_curve", False))
            if cached_has_curve or not bool(include_equity_curve):
                report = cached_entry.get("report")
        if report is None:
            report = _evaluate_next_1y_oos(
                session=session,
                trial=trial,
                oos_year=int(oos_year),
                include_equity_curve=bool(include_equity_curve),
            )
            report_cache[trial_number] = {
                "include_equity_curve": bool(include_equity_curve),
                "report": report,
            }
        metrics = _extract_period_metrics(report)
        if bool(include_equity_curve) or trial_number not in metrics_by_trial:
            metrics_by_trial[trial_number] = dict(metrics)
        return dict(metrics)

    for item in list(finalists or []):
        trial = item.get("trial")
        if trial is None:
            continue
        trial_number = int(trial.number)
        metrics = _load_trial_metrics(trial, include_equity_curve=trial_number in curve_trial_numbers)
        benchmark_score = float(metrics.get("benchmark_oos_score", benchmark_score))
        benchmark_return_pct = float(metrics.get("benchmark_return_pct", benchmark_return_pct))
        benchmark_mdd_pct = float(metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
        score = float(metrics["oos_score"])
        if score > best_score:
            best_score = score
            best_trial_number = trial_number
            best_trial = trial
            best_metrics = dict(metrics)
    if best_score == float("-inf"):
        best_score = 0.0
        best_metrics = {}
    elif best_trial is not None:
        best_metrics = _load_trial_metrics(best_trial, include_equity_curve=True)
    policies: dict[str, dict] = {}
    for policy_name, item in dict(policy_items or {}).items():
        if item is None or item.get("trial") is None:
            policies[policy_name] = {
                "rank_1_trial": None,
                "rank_1_oos": 0.0,
                "rank_1_return_pct": 0.0,
                "rank_1_mdd_pct": 0.0,
                "rank_1_trades": 0,
                "best_gap": 0.0 - float(best_score),
                "benchmark_0050_gap": 0.0 - float(benchmark_score),
                "rank_1_initial_capital": 0.0,
                "rank_1_equity_curve": [],
            }
            continue
        trial_number = int(item["trial"].number)
        metrics = _load_trial_metrics(item["trial"], include_equity_curve=True)
        rank_1_oos = float(metrics.get("oos_score", 0.0))
        benchmark_score = float(metrics.get("benchmark_oos_score", benchmark_score))
        benchmark_return_pct = float(metrics.get("benchmark_return_pct", benchmark_return_pct))
        benchmark_mdd_pct = float(metrics.get("benchmark_mdd_pct", benchmark_mdd_pct))
        policies[policy_name] = {
            "rank_1_trial": trial_number + 1,
            "rank_1_oos": rank_1_oos,
            "rank_1_return_pct": float(metrics.get("ret_pct", 0.0)),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_trades": int(metrics.get("trades", 0) or 0),
            "best_gap": rank_1_oos - float(best_score),
            "benchmark_0050_gap": rank_1_oos - float(benchmark_score),
            "rank_1_initial_capital": float(metrics.get("initial_capital", 0.0)),
            "rank_1_equity_curve": list(metrics.get("equity_curve") or []),
        }
    return {
        "best_finalist_oos_score": float(best_score),
        "best_finalist_trial": int(best_trial_number) + 1 if best_trial_number is not None else None,
        "best_finalist_return_pct": float(best_metrics.get("ret_pct", 0.0)),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "best_finalist_trades": int(best_metrics.get("trades", 0) or 0),
        "best_finalist_initial_capital": float(best_metrics.get("initial_capital", 0.0)),
        "best_finalist_equity_curve": list(best_metrics.get("equity_curve") or []),
        "best_finalist_params": build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT) if best_trial is not None else {},
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_mdd_pct": float(benchmark_mdd_pct),
        "policies": policies,
    }


def _compound_return_pct(return_pcts: list[float]) -> float:
    equity = 1.0
    for value in list(return_pcts or []):
        equity *= 1.0 + float(value) / 100.0
    return (equity - 1.0) * 100.0


def _average_float(values: list[float]) -> float:
    nums = [float(value) for value in list(values or [])]
    if not nums:
        return 0.0
    return sum(nums) / float(len(nums))


def _normalize_equity_curve_rows(curve_rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for raw in list(curve_rows or []):
        try:
            date = pd.Timestamp(raw.get("date") or raw.get("Date")).strftime("%Y-%m-%d")
            equity = float(raw.get("equity", raw.get("Equity", 0.0)) or 0.0)
            strategy_return_pct = float(raw.get("strategy_return_pct", raw.get("Strategy_Return_Pct", 0.0)) or 0.0)
            benchmark_return_pct = float(raw.get("benchmark_return_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not date or equity <= 0.0:
            continue
        normalized.append({
            "date": date,
            "equity": equity,
            "strategy_return_pct": strategy_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
        })
    return sorted(normalized, key=lambda row: row["date"])


def _derive_curve_initial_capital(curve_rows: list[dict], explicit_initial_capital: float | int | None = None) -> float:
    explicit = _safe_float(explicit_initial_capital, 0.0)
    if explicit > 0.0:
        return explicit
    curve = _normalize_equity_curve_rows(curve_rows)
    if not curve:
        return 0.0
    first = curve[0]
    denominator = 1.0 + float(first.get("strategy_return_pct", 0.0)) / 100.0
    if denominator <= 0.0:
        return float(first.get("equity", 0.0))
    return float(first.get("equity", 0.0)) / denominator


def _stitch_strategy_equity_curves(rows: list[dict], *, policy_name: str | None = None, best_finalist: bool = False) -> dict:
    ordered_rows = sorted(list(rows or []), key=lambda row: int(row.get("oos_year", 0) or 0))
    stitched: list[dict] = []
    chain_initial = 0.0
    current_start = 0.0
    for row in ordered_rows:
        if best_finalist:
            raw_curve = _normalize_equity_curve_rows(list(row.get("best_finalist_equity_curve") or []))
            raw_initial = _derive_curve_initial_capital(raw_curve, row.get("best_finalist_initial_capital"))
        else:
            policy = dict(row.get(str(policy_name)) or {})
            raw_curve = _normalize_equity_curve_rows(list(policy.get("rank_1_equity_curve") or []))
            raw_initial = _derive_curve_initial_capital(raw_curve, policy.get("rank_1_initial_capital"))
        if not raw_curve or raw_initial <= 0.0:
            continue
        if current_start <= 0.0:
            current_start = raw_initial
            chain_initial = raw_initial
        scale = current_start / raw_initial
        for point in raw_curve:
            stitched.append({
                "date": point["date"],
                "equity": float(point["equity"]) * scale,
            })
        current_start = float(stitched[-1]["equity"])
    return {"initial_equity": float(chain_initial), "curve": stitched}


def _stitch_benchmark_equity_curve(rows: list[dict]) -> dict:
    ordered_rows = sorted(list(rows or []), key=lambda row: int(row.get("oos_year", 0) or 0))
    stitched: list[dict] = []
    chain_initial = 0.0
    current_start = 0.0
    for row in ordered_rows:
        curve_source = None
        best_curve = _normalize_equity_curve_rows(list(row.get("best_finalist_equity_curve") or []))
        if best_curve:
            curve_source = best_curve
            raw_initial = _derive_curve_initial_capital(best_curve, row.get("best_finalist_initial_capital"))
        else:
            raw_initial = 0.0
            for policy_name in ("base", "local", "retention"):
                policy = dict(row.get(policy_name) or {})
                policy_curve = _normalize_equity_curve_rows(list(policy.get("rank_1_equity_curve") or []))
                if policy_curve:
                    curve_source = policy_curve
                    raw_initial = _derive_curve_initial_capital(policy_curve, policy.get("rank_1_initial_capital"))
                    break
        if not curve_source:
            continue
        if current_start <= 0.0:
            current_start = raw_initial if raw_initial > 0.0 else 1.0
            chain_initial = current_start
        for point in curve_source:
            benchmark_factor = 1.0 + float(point.get("benchmark_return_pct", 0.0)) / 100.0
            stitched.append({
                "date": point["date"],
                "equity": current_start * benchmark_factor,
            })
        current_start = float(stitched[-1]["equity"])
    return {"initial_equity": float(chain_initial), "curve": stitched}


def _month_end_equities_from_curve(curve: list[dict], *, initial_equity: float) -> list[float]:
    values = []
    if initial_equity > 0.0:
        values.append(float(initial_equity))
    current_month = None
    previous_equity = None
    for point in list(curve or []):
        try:
            ts = pd.Timestamp(point.get("date"))
            equity = float(point.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        month_key = (int(ts.year), int(ts.month))
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            if previous_equity is not None:
                values.append(float(previous_equity))
            current_month = month_key
        previous_equity = equity
    if previous_equity is not None:
        values.append(float(previous_equity))
    return values


def _calc_stitched_curve_metrics(stitched: dict) -> dict:
    initial_equity = _safe_float(stitched.get("initial_equity"), 0.0)
    curve = list(stitched.get("curve") or [])
    if initial_equity <= 0.0 or not curve:
        return {
            "score": 0.0,
            "return_pct": 0.0,
            "mdd_pct": 0.0,
            "annual_return_pct": 0.0,
            "r_squared": 0.0,
            "monthly_win_rate": 0.0,
            "curve_points": 0,
        }
    final_equity = float(curve[-1].get("equity", initial_equity) or initial_equity)
    return_pct = (final_equity / initial_equity - 1.0) * 100.0
    peak = initial_equity
    max_drawdown = 0.0
    for point in curve:
        equity = float(point.get("equity", 0.0) or 0.0)
        if equity > peak:
            peak = equity
        if peak > 0.0:
            drawdown = (peak - equity) / peak * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    first_date = pd.Timestamp(curve[0]["date"])
    last_date = pd.Timestamp(curve[-1]["date"])
    years = max(0.0, ((last_date - first_date).days + 1) / 365.25)
    annual_return_pct = calc_annual_return_pct(initial_equity, final_equity, years)
    monthly_equities = _month_end_equities_from_curve(curve, initial_equity=initial_equity)
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    score = calc_portfolio_score(
        return_pct,
        max_drawdown,
        monthly_win_rate,
        r_squared,
        annual_return_pct=annual_return_pct,
    )
    return {
        "score": float(score),
        "return_pct": float(return_pct),
        "mdd_pct": float(max_drawdown),
        "annual_return_pct": float(annual_return_pct),
        "r_squared": float(r_squared),
        "monthly_win_rate": float(monthly_win_rate),
        "curve_points": int(len(curve)),
        "start_date": str(curve[0].get("date", "")),
        "end_date": str(curve[-1].get("date", "")),
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
    }


def _build_active_param_replay_payload_from_rows(rows: list[dict], *, policy_name: str | None = None, best_finalist: bool = False) -> dict:
    params_by_oos_year: dict[str, dict] = {}
    params_by_effective_date: dict[str, dict] = {}
    for row in sorted(list(rows or []), key=lambda item: int(item.get("oos_year", 0) or 0)):
        try:
            oos_year = int(row.get("oos_year"))
        except (TypeError, ValueError):
            continue
        if best_finalist:
            params_payload = dict(row.get("best_finalist_params") or {})
            effective_start = f"{oos_year}-01-01"
        else:
            schedule = dict((row.get("policy_schedules") or {}).get(str(policy_name)) or {})
            params_payload = dict(schedule.get("params") or {})
            effective_start = str(schedule.get("effective_start") or f"{oos_year}-01-01")
        if not params_payload:
            continue
        params_by_oos_year[str(oos_year)] = params_payload
        params_by_effective_date[effective_start] = params_payload
    return {
        "schema_type": ROLLING_OOS_PARAM_SET_SCHEMA_TYPE,
        "schema_version": 1,
        "usage": ROLLING_OOS_USAGE,
        "type": "outer_rolling_oos_param_set",
        "active_param_policy": "daily_active_param",
        "params_by_oos_year": params_by_oos_year,
        "params_by_effective_date": params_by_effective_date,
    }


def _active_replay_payload_has_params(payload: dict) -> bool:
    return bool(dict(payload.get("params_by_effective_date") or {}) or dict(payload.get("params_by_oos_year") or {}))


def _extract_active_replay_metrics(result) -> dict:
    if not isinstance(result, tuple) or len(result) < 25:
        raise RuntimeError("active-param replay 回傳格式不完整，無法建立 OOS_CHAIN")
    ret_pct = float(result[2])
    mdd_pct = float(result[3])
    trade_count = int(result[4] or 0)
    bm_ret_pct = float(result[11])
    bm_mdd_pct = float(result[12])
    r_squared = float(result[15])
    monthly_win_rate = float(result[16])
    bm_r_squared = float(result[17])
    bm_monthly_win_rate = float(result[18])
    annual_return_pct = float(result[23])
    bm_annual_return_pct = float(result[24])
    profile = dict(result[-1]) if isinstance(result[-1], dict) else {}
    equity_curve_points = len(profile.get("equity_curve") or [])
    score = calc_portfolio_score(
        ret_pct,
        mdd_pct,
        monthly_win_rate,
        r_squared,
        annual_return_pct=annual_return_pct,
    )
    benchmark_score = calc_portfolio_score(
        bm_ret_pct,
        bm_mdd_pct,
        bm_monthly_win_rate,
        bm_r_squared,
        annual_return_pct=bm_annual_return_pct,
    )
    return {
        "score": float(score),
        "return_pct": float(ret_pct),
        "mdd_pct": float(mdd_pct),
        "annual_return_pct": float(annual_return_pct),
        "r_squared": float(r_squared),
        "monthly_win_rate": float(monthly_win_rate),
        "trade_count": int(trade_count),
        "curve_points": int(equity_curve_points),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(bm_ret_pct),
        "benchmark_mdd_pct": float(bm_mdd_pct),
        "benchmark_annual_return_pct": float(bm_annual_return_pct),
        "benchmark_r_squared": float(bm_r_squared),
        "benchmark_monthly_win_rate": float(bm_monthly_win_rate),
    }


def _build_active_replay_schedule_records(payload: dict) -> list[dict]:
    if not _active_replay_payload_has_params(payload):
        return []
    from tools.portfolio_sim.simulation_runner import _build_active_param_objects_from_payload

    return list(_build_active_param_objects_from_payload(payload, fixed_risk=None))


def _load_active_replay_contexts_by_signature(
    *,
    data_dir: str,
    schedule_groups: dict[str, list[dict]],
    output_dir: str,
    first_year: int | None = None,
    last_year: int | None = None,
    overall_start: float | None = None,
) -> dict[str, dict]:
    from core.data_utils import get_required_min_rows
    from core.portfolio_fast_data import build_normal_setup_index, build_trade_stats_index, pack_static_market_data
    from tools.optimizer.raw_cache import load_all_raw_data
    from tools.portfolio_sim.simulation_runner import load_portfolio_market_context

    contexts_by_signature: dict[str, dict] = {}
    records: list[dict] = []
    by_signature: dict[str, dict] = {}
    for policy_name, group_records in dict(schedule_groups or {}).items():
        for record in list(group_records or []):
            signature = str(record.get("params_signature") or "")
            if not signature:
                continue
            existing = by_signature.get(signature)
            if existing is None:
                existing = dict(record)
                existing["_policies"] = []
                by_signature[signature] = existing
                records.append(existing)
            policies = existing.setdefault("_policies", [])
            if str(policy_name) not in policies:
                policies.append(str(policy_name))

    def _record_year(item: dict) -> int:
        try:
            return int(item.get("year", 0) or 0)
        except (TypeError, ValueError):
            text = str(item.get("effective_date_text") or "")
            try:
                return int(text[:4])
            except (TypeError, ValueError):
                return 0

    records.sort(key=lambda item: (_record_year(item), str(item.get("effective_date_text") or ""), str(item.get("params_signature") or "")))
    total = len(records)
    previous_width = 0
    supports_inline = stdout_supports_inline_progress()
    policy_names = "/".join(sorted({policy for record in records for policy in record.get("_policies", [])})) or "N/A"
    replay_context_start = time.perf_counter()
    years_label = ""
    if first_year and last_year:
        years_label = f" | years={int(first_year)}~{int(last_year)}"

    include_trade_logs = _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS", False)
    include_pit_stats_index = _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX", True)
    use_prepared_cache = _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE", False)
    write_prepared_cache = _env_flag(os.environ, "OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE", False)

    raw_data_cache = None
    static_fast_cache = None
    static_master_dates = None
    active_prep_workers = None
    if total and not (use_prepared_cache or write_prepared_cache):
        required_min_rows = max(get_required_min_rows(record["params_obj"]) for record in records)
        raw_data_cache = load_all_raw_data(data_dir, required_min_rows, output_dir, verbose=False)
        static_fast_cache = {ticker: pack_static_market_data(df) for ticker, df in raw_data_cache.items()}
        static_master_dates = set()
        for df in raw_data_cache.values():
            static_master_dates.update(df.index)
        raw_workers = str(os.environ.get("OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS", "")).strip()
        if raw_workers:
            try:
                active_prep_workers = max(1, min(int(raw_workers), max(1, len(raw_data_cache))))
            except ValueError as exc:
                raise ValueError(f"OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS 必須是整數，收到: {raw_workers}") from exc
        else:
            active_prep_workers = max(1, min(os.cpu_count() or 1, 8, len(raw_data_cache)))

    def _covers_text(index: int, item: dict) -> str:
        year = _record_year(item)
        if year <= 0:
            return "N/A"
        next_year = None
        for follow in records[index:]:
            candidate = _record_year(follow)
            if candidate > year:
                next_year = candidate
                break
        end_year = int(last_year) if last_year else year
        if next_year is not None:
            end_year = min(end_year, int(next_year) - 1)
        if end_year <= year:
            return str(year)
        return f"{year}~{end_year}"

    for idx, record in enumerate(records, start=1):
        signature = str(record["params_signature"])
        policies_text = ",".join(record.get("_policies", [])) or "N/A"
        chain_elapsed = max(0.0, time.perf_counter() - replay_context_start)
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        message = (
            f"{C_CYAN}⏱️ OOS_CHAIN active replay context [{idx}/{total}] | "
            f"covers={_covers_text(idx, record)} | policies={policies_text} | effective={record.get('effective_date_text')} | "
            f"signature={signature[:8]}{years_label} | chain={_fmt_duration(chain_elapsed)}{total_elapsed_text}{C_RESET}"
        )
        if supports_inline:
            previous_width = write_inline_progress(message, previous_width=previous_width)
        else:
            print(message)

        if raw_data_cache is not None:
            prep_result = prepare_trial_inputs(
                raw_data_cache=raw_data_cache,
                params=record["params_obj"],
                default_max_workers=int(active_prep_workers or 1),
                static_fast_cache=static_fast_cache,
                static_master_dates=static_master_dates,
                include_trade_logs=bool(include_trade_logs),
                include_pit_stats_index=bool(include_pit_stats_index),
                profile_enabled=False,
            )
            context = {
                "all_dfs_fast": prep_result.get("all_dfs_fast") or {},
                "all_trade_logs": prep_result.get("all_trade_logs") or {},
                "all_pit_stats_index": prep_result.get("all_pit_stats_index") or {},
                "sorted_dates": sorted(prep_result.get("master_dates") or []),
                "prep_wall_sec": float(prep_result.get("prep_wall_sec", 0.0) or 0.0),
                "prep_mode": f"active_replay_shared_raw_{prep_result.get('prep_mode')}",
            }
        else:
            context = load_portfolio_market_context(data_dir, record["params_obj"], verbose=False)
            context = dict(context)

        if not context.get("all_pit_stats_index"):
            if not context.get("all_trade_logs"):
                raise RuntimeError("active replay context 缺少 PIT stats index；請保留 OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX=1")
            context["all_pit_stats_index"] = {
                ticker: build_trade_stats_index(logs)
                for ticker, logs in (context.get("all_trade_logs") or {}).items()
            }
        context["normal_setup_index"] = build_normal_setup_index(context.get("all_dfs_fast") or {})
        contexts_by_signature[signature] = context

    if total:
        chain_elapsed = max(0.0, time.perf_counter() - replay_context_start)
        total_elapsed_text = ""
        if overall_start is not None:
            total_elapsed_text = f" | total={_fmt_duration(time.perf_counter() - float(overall_start))}"
        summary = (
            f"{C_CYAN}⏱️ OOS_CHAIN active replay context 完成 | "
            f"contexts={total}/{total}{years_label} | policies={policy_names} | "
            f"chain={_fmt_duration(chain_elapsed)}{total_elapsed_text}{C_RESET}"
        )
        if supports_inline:
            write_inline_progress(summary, previous_width=previous_width)
            print()
        else:
            print(summary)
    return contexts_by_signature

def _merge_active_replay_market_dates(contexts_by_signature: dict[str, dict]) -> list:
    market_dates = set()
    for context in dict(contexts_by_signature or {}).values():
        market_dates.update(context.get("sorted_dates") or [])
    return sorted(market_dates)


def _run_active_replay_metrics_from_schedule_records(*, schedule_records: list[dict], contexts_by_signature: dict[str, dict], start_year: int, end_year: int, max_positions: int, enable_rotation: bool, benchmark_ticker: str = "0050") -> dict:
    if not schedule_records:
        return {}
    from core.portfolio_engine import run_portfolio_timeline
    from core.portfolio_stats import find_sim_start_idx
    from tools.portfolio_sim.simulation_runner import _filter_market_dates_by_end_year, _resolve_active_schedule_record

    resolved_sorted_dates = _filter_market_dates_by_end_year(
        _merge_active_replay_market_dates(contexts_by_signature),
        start_year=int(start_year),
        end_year=int(end_year),
    )
    if not resolved_sorted_dates:
        raise ValueError("active-param replay 沒有可回測日期")
    first_sim_idx = find_sim_start_idx(resolved_sorted_dates, int(start_year))
    if first_sim_idx >= len(resolved_sorted_dates):
        raise ValueError("active-param replay 起始年份沒有可回測日期")
    first_record = _resolve_active_schedule_record(schedule_records, resolved_sorted_dates[first_sim_idx])
    base_context = contexts_by_signature[str(first_record["params_signature"])]
    benchmark_data = (base_context.get("all_dfs_fast") or {}).get(benchmark_ticker)

    def active_params_resolver(trade_date):
        return _resolve_active_schedule_record(schedule_records, trade_date)["params_obj"]

    def active_context_resolver(trade_date):
        record = _resolve_active_schedule_record(schedule_records, trade_date)
        return contexts_by_signature[str(record["params_signature"])]

    pf_profile = {
        "param_policy": "active_param_replay",
        "capture_equity_curve": True,
        "active_param_schedule": [
            {
                "effective_date": str(record.get("effective_date_text")),
                "year": int(record.get("year", 0) or 0),
                "params_signature": str(record.get("params_signature")),
            }
            for record in schedule_records
        ],
    }
    result = run_portfolio_timeline(
        base_context.get("all_dfs_fast") or {},
        base_context.get("all_trade_logs") or {},
        resolved_sorted_dates,
        int(start_year),
        first_record["params_obj"],
        int(max_positions),
        bool(enable_rotation),
        benchmark_ticker=str(benchmark_ticker),
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=False,
        pit_stats_index=base_context.get("all_pit_stats_index"),
        active_params_resolver=active_params_resolver,
        active_context_resolver=active_context_resolver,
    )
    return _extract_active_replay_metrics((*result, pf_profile))


def _build_active_replay_chained_oos_summary(
    *,
    rows: list[dict],
    config: OuterRollingConfig,
    selected_data_dir: str,
    output_dir: str,
    max_positions: int,
    enable_rotation: bool,
    overall_start: float | None = None,
) -> dict:
    if not rows:
        return {}
    first_year = min(int(row["oos_year"]) for row in rows)
    last_year = max(int(row["oos_year"]) for row in rows)
    selection_start = min(int(row.get("selection_start_year", str(row.get("selection_period", "0~0")).split("~", 1)[0])) for row in rows)
    selection_end = max(int(row.get("selection_end_year", str(row.get("selection_period", "0~0")).split("~", 1)[-1])) for row in rows)

    payloads = {
        "best": _build_active_param_replay_payload_from_rows(rows, best_finalist=True),
        "base": _build_active_param_replay_payload_from_rows(rows, policy_name="base"),
        "local": _build_active_param_replay_payload_from_rows(rows, policy_name="local"),
        "retention": _build_active_param_replay_payload_from_rows(rows, policy_name="retention"),
    }
    schedule_groups = {name: _build_active_replay_schedule_records(payload) for name, payload in payloads.items()}
    contexts_by_signature = _load_active_replay_contexts_by_signature(
        data_dir=selected_data_dir,
        schedule_groups=schedule_groups,
        output_dir=output_dir,
        first_year=first_year,
        last_year=last_year,
        overall_start=overall_start,
    )

    replay_metrics: dict[str, dict] = {}
    for name, records in schedule_groups.items():
        replay_metrics[name] = _run_active_replay_metrics_from_schedule_records(
            schedule_records=records,
            contexts_by_signature=contexts_by_signature,
            start_year=first_year,
            end_year=last_year,
            max_positions=max_positions,
            enable_rotation=enable_rotation,
        )

    best_metrics = dict(replay_metrics.get("best") or {})
    benchmark_source = dict(best_metrics)
    for policy_name in ("base", "local", "retention"):
        if not benchmark_source and replay_metrics.get(policy_name):
            benchmark_source = dict(replay_metrics[policy_name])
            break

    best_score = float(best_metrics.get("score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    benchmark_score = float(benchmark_source.get("benchmark_oos_score", 0.0))
    benchmark_return = float(benchmark_source.get("benchmark_return_pct", 0.0))
    summary = {
        "method": "continuous_active_param_replay",
        "score_aggregation_method": "continuous_active_param_replay_recomputed_score",
        "return_aggregation_method": "continuous_active_param_replay_total_return",
        "note": "OOS_CHAIN 由 portfolio_sim active-param replay 連續重跑後重算 score；持股、現金與 benchmark 跨年延續，不使用年度 closeout stitch 或年度 score mean。",
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_year}~{last_year}",
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "best_finalist_oos_score": float(best_score),
        "benchmark_oos_score": float(benchmark_score),
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "benchmark_mdd_pct": float(benchmark_source.get("benchmark_mdd_pct", 0.0)),
        "best_finalist_annual_return_pct": float(best_metrics.get("annual_return_pct", 0.0)),
        "benchmark_annual_return_pct": float(benchmark_source.get("benchmark_annual_return_pct", 0.0)),
        "best_finalist_curve_points": int(best_metrics.get("curve_points", 0)),
        "benchmark_curve_points": int(benchmark_source.get("curve_points", 0)),
        "yearly_best_finalist_oos_score": [float(row.get("best_finalist_oos_score", 0.0)) for row in rows],
        "yearly_benchmark_oos_score": [float(row.get("benchmark_oos_score", 0.0)) for row in rows],
    }
    for policy_name in ("base", "local", "retention"):
        metrics = dict(replay_metrics.get(policy_name) or {})
        chain_score = float(metrics.get("score", 0.0))
        chain_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "rank_1_oos": float(chain_score),
            "best_gap": float(chain_score - best_score),
            "benchmark_0050_gap": float(chain_score - benchmark_score),
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "rank_1_curve_points": int(metrics.get("curve_points", 0)),
            "rank_1_trades": int(metrics.get("trade_count", 0)),
            "yearly_oos_score": [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows],
            "yearly_return_pct": [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows],
        }
    return summary

def _build_chained_oos_summary(rows: list[dict], *, chained_override: dict | None = None) -> dict:
    if chained_override is not None:
        return dict(chained_override)
    if not rows:
        return {}
    first_year = min(int(row["oos_year"]) for row in rows)
    last_year = max(int(row["oos_year"]) for row in rows)
    selection_start = min(int(row.get("selection_start_year", str(row.get("selection_period", "0~0")).split("~", 1)[0])) for row in rows)
    selection_end = max(int(row.get("selection_end_year", str(row.get("selection_period", "0~0")).split("~", 1)[-1])) for row in rows)

    best_stitched = _stitch_strategy_equity_curves(rows, best_finalist=True)
    benchmark_stitched = _stitch_benchmark_equity_curve(rows)
    best_metrics = _calc_stitched_curve_metrics(best_stitched)
    benchmark_metrics = _calc_stitched_curve_metrics(benchmark_stitched)
    best_score = float(best_metrics.get("score", 0.0))
    benchmark_score = float(benchmark_metrics.get("score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    benchmark_return = float(benchmark_metrics.get("return_pct", 0.0))

    summary = {
        "method": "fallback_yearly_closeout_stitched_daily_equity",
        "score_aggregation_method": "fallback_yearly_closeout_stitched_daily_equity",
        "return_aggregation_method": "fallback_yearly_closeout_stitched_daily_equity_total_return",
        "note": "fallback：由各年度 next-1Y OOS daily equity 串接後重新計算 score；正式輸出應優先使用 continuous_active_param_replay。",
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_year}~{last_year}",
        "best_finalist_oos_score": float(best_score),
        "benchmark_oos_score": float(benchmark_score),
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "benchmark_mdd_pct": float(benchmark_metrics.get("mdd_pct", 0.0)),
        "best_finalist_annual_return_pct": float(best_metrics.get("annual_return_pct", 0.0)),
        "benchmark_annual_return_pct": float(benchmark_metrics.get("annual_return_pct", 0.0)),
        "best_finalist_curve_points": int(best_metrics.get("curve_points", 0)),
        "benchmark_curve_points": int(benchmark_metrics.get("curve_points", 0)),
        "yearly_best_finalist_oos_score": [float(row.get("best_finalist_oos_score", 0.0)) for row in rows],
        "yearly_benchmark_oos_score": [float(row.get("benchmark_oos_score", 0.0)) for row in rows],
    }
    for policy_name in ("base", "local", "retention"):
        stitched = _stitch_strategy_equity_curves(rows, policy_name=policy_name)
        metrics = _calc_stitched_curve_metrics(stitched)
        chain_score = float(metrics.get("score", 0.0))
        chain_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "rank_1_oos": float(chain_score),
            "best_gap": float(chain_score - best_score),
            "benchmark_0050_gap": float(chain_score - benchmark_score),
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "rank_1_curve_points": int(metrics.get("curve_points", 0)),
            "yearly_oos_score": [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows],
            "yearly_return_pct": [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows],
        }
    return summary


def _resolve_chain_elapsed_sec(rows: list[dict], chained: dict) -> float:
    """Return the OOS_CHAIN elapsed value for display/reporting.

    In serial mode the chain row historically used the sum of fold elapsed time. In
    parallel-fold timing mode, however, the sum of fold elapsed time is total work
    time, not user-visible wall-clock time. When the caller provides an
    ``elapsed_sec`` override in the chained summary, prefer it; otherwise fall back
    to the serial-compatible fold sum.
    """
    try:
        override = chained.get("elapsed_sec")
    except AttributeError:
        override = None
    if override is not None:
        try:
            return max(0.0, float(override))
        except (TypeError, ValueError):
            pass
    total = 0.0
    for item in list(rows or []):
        try:
            total += float(item.get("elapsed_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return max(0.0, total)


def _with_chain_elapsed_override(chained_override: dict | None, *, elapsed_sec: float | None) -> dict:
    payload = dict(chained_override or {})
    if elapsed_sec is not None:
        try:
            payload["elapsed_sec"] = max(0.0, float(elapsed_sec))
        except (TypeError, ValueError):
            pass
    return payload


def _build_chained_oos_row(rows: list[dict], *, chained_override: dict | None = None) -> dict | None:
    if not rows:
        return None
    chained = _build_chained_oos_summary(rows, chained_override=chained_override)
    row = {
        "fold": "OOS_CHAIN",
        "selection_period": chained.get("selection_period", ""),
        "oos_year": chained.get("oos_period", ""),
        "best_finalist_oos_score": float(chained.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "best_finalist_return_pct": float(chained.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "elapsed_sec": _resolve_chain_elapsed_sec(rows, chained),
        "aggregation_method": chained.get("score_aggregation_method") or chained.get("method"),
    }
    for policy_name in ("base", "local", "retention"):
        item = dict(chained.get(policy_name) or {})
        rank_score = float(item.get("rank_1_oos", 0.0))
        row[policy_name] = {
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_return_pct": float(item.get("rank_1_return_pct", 0.0)),
            "rank_1_mdd_pct": float(item.get("rank_1_mdd_pct", 0.0)),
            "best_gap": rank_score - float(chained.get("best_finalist_oos_score", 0.0)),
            "benchmark_0050_gap": rank_score - float(chained.get("benchmark_oos_score", 0.0)),
        }
    return row



def _avg_float_from_rows(rows: list[dict], getter, default: float = 0.0) -> float:
    values: list[float] = []
    for row in list(rows or []):
        try:
            value = float(getter(row))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
        values.append(value)
    return (sum(values) / float(len(values))) if values else float(default)


def _build_oos_avg_row(rows: list[dict]) -> dict | None:
    source_rows = [dict(row) for row in list(rows or []) if str(row.get("fold", "")).upper() not in {"OOS_CHAIN", "OOS_AVG"}]
    if not source_rows:
        return None
    selection_start = min(int(row.get("selection_start_year", str(row.get("selection_period", "0~0")).split("~", 1)[0])) for row in source_rows)
    selection_end = max(int(row.get("selection_end_year", str(row.get("selection_period", "0~0")).split("~", 1)[-1])) for row in source_rows)
    first_oos = min(int(row.get("oos_year", 0) or 0) for row in source_rows)
    last_oos = max(int(row.get("oos_year", 0) or 0) for row in source_rows)
    row = {
        "fold": "OOS_AVG",
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_year": f"{first_oos}~{last_oos}" if first_oos != last_oos else str(first_oos),
        "best_finalist_oos_score": _avg_float_from_rows(source_rows, lambda item: item.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": _avg_float_from_rows(source_rows, lambda item: item.get("benchmark_oos_score", 0.0)),
        "best_finalist_return_pct": _avg_float_from_rows(source_rows, lambda item: item.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": _avg_float_from_rows(source_rows, lambda item: item.get("benchmark_return_pct", 0.0)),
        "elapsed_sec": None,
    }
    for policy_name in ("base", "local", "retention"):
        rank_score = _avg_float_from_rows(source_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_oos", 0.0))
        row[policy_name] = {
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_return_pct": _avg_float_from_rows(source_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_return_pct", 0.0)),
            "rank_1_mdd_pct": _avg_float_from_rows(source_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_mdd_pct", 0.0)),
            "best_gap": rank_score - float(row["best_finalist_oos_score"]),
            "benchmark_0050_gap": rank_score - float(row["benchmark_oos_score"]),
        }
    return row


def _policy_cell_text(policy_row: dict, *, best_score: float, benchmark_score: float, color: bool = True) -> tuple[str, str, str]:
    rank_1 = float(policy_row.get("rank_1_oos", 0.0))
    if color:
        return (
            _format_score(rank_1),
            _format_compare(best_score, rank_1),
            _format_compare(benchmark_score, rank_1),
        )
    return (
        f"{rank_1:.{OOS_SCORE_DECIMALS}f}",
        _format_compare_plain(best_score, rank_1),
        _format_compare_plain(benchmark_score, rank_1),
    )


def _table_separator(width: int = 218) -> str:
    return "-" * int(width)


def _render_results_table(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None) -> str:
    display_rows = list(rows or [])
    if include_oos_avg and display_rows:
        avg_row = _build_oos_avg_row(display_rows)
        if avg_row is not None:
            display_rows = display_rows + [avg_row]
    if include_chain:
        chain_row = _build_chained_oos_row(display_rows, chained_override=chained_override)
        if chain_row is not None:
            display_rows = display_rows + [chain_row]
    if not display_rows:
        return ""
    widths = {
        "fold": 9,
        "selection": 11,
        "oos_year": 9,
        "rank": 8,
        "best": 17,
        "bench": 17,
        "elapsed": 8,
    }
    base_group_width = widths["rank"] + widths["bench"] + 3
    local_group_width = widths["rank"] + widths["best"] + widths["bench"] + 6
    retention_group_width = widths["rank"] + widths["bench"] + 3
    lines: list[str] = []
    lines.append("ROLLING NEXT-1Y OOS RESULTS")
    header1 = (
        f"{_pad_ansi('fold', widths['fold'])} | {_pad_ansi('selection', widths['selection'])} | {_pad_ansi('oos_year', widths['oos_year'])} | "
        f"{_pad_ansi('base', base_group_width)} | "
        f"{_pad_ansi('local*', local_group_width)} | "
        f"{_pad_ansi('retention', retention_group_width)} | {_pad_ansi('elapsed', widths['elapsed'], align='>')}"
    )
    header2 = (
        f"{_pad_ansi('', widths['fold'])} | {_pad_ansi('', widths['selection'])} | {_pad_ansi('', widths['oos_year'])} | "
        f"{_pad_ansi('rank_1', widths['rank'], align='>')} | {_pad_ansi('0050', widths['bench'], align='>')} | "
        f"{_pad_ansi('rank_1', widths['rank'], align='>')} | {_pad_ansi('best', widths['best'], align='>')} | {_pad_ansi('0050', widths['bench'], align='>')} | "
        f"{_pad_ansi('rank_1', widths['rank'], align='>')} | {_pad_ansi('0050', widths['bench'], align='>')} | {_pad_ansi('', widths['elapsed'])}"
    )
    separator = _table_separator(max(_visible_len(header1), _visible_len(header2), 120))
    lines.append(separator)
    lines.append(header1)
    lines.append(header2)
    lines.append(separator)
    total = len(rows or [])
    for idx, row in enumerate(display_rows, start=1):
        fold_kind = str(row.get("fold", "")).upper()
        is_chain = fold_kind == "OOS_CHAIN"
        is_avg = fold_kind == "OOS_AVG"
        fold_text = "OOS_CHAIN" if is_chain else ("OOS_AVG" if is_avg else str(row.get("fold") or f"{idx}/{total}"))
        best_score = float(row.get("best_finalist_oos_score", 0.0))
        benchmark_score = float(row.get("benchmark_oos_score", 0.0))
        base_rank, _base_best, base_bench = _policy_cell_text(row.get("base") or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
        local_rank, local_best, local_bench = _policy_cell_text(row.get("local") or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
        retention_rank, _retention_best, retention_bench = _policy_cell_text(row.get("retention") or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
        line = (
            f"{_pad_ansi(fold_text, widths['fold'])} | {_pad_ansi(str(row.get('selection_period', '')), widths['selection'])} | {_pad_ansi(str(row.get('oos_year', '')), widths['oos_year'])} | "
            f"{_pad_ansi(base_rank, widths['rank'], align='>')} | {_pad_ansi(base_bench, widths['bench'], align='>')} | "
            f"{_pad_ansi(local_rank, widths['rank'], align='>')} | {_pad_ansi(local_best, widths['best'], align='>')} | {_pad_ansi(local_bench, widths['bench'], align='>')} | "
            f"{_pad_ansi(retention_rank, widths['rank'], align='>')} | {_pad_ansi(retention_bench, widths['bench'], align='>')} | "
            f"{_pad_ansi('' if row.get('elapsed_sec') is None else _fmt_duration(row.get('elapsed_sec', 0.0)), widths['elapsed'], align='>')}"
        )
        lines.append(line)
    lines.append(separator)
    return "\n".join(lines)

def _print_completed_results(rows: list[dict]):
    if not rows:
        return
    print("\n" + _render_results_table(rows, color=True, include_chain=True))


def _flatten_policy_for_csv(row: dict, policy_name: str) -> dict:
    policy = dict(row.get(policy_name) or {})
    best = float(row.get("best_finalist_oos_score", 0.0))
    bench = float(row.get("benchmark_oos_score", 0.0))
    rank_1 = float(policy.get("rank_1_oos", 0.0))
    return {
        f"{policy_name}_rank_1_trial": policy.get("rank_1_trial"),
        f"{policy_name}_rank_1_oos": rank_1,
        f"{policy_name}_rank_1_return_pct": float(policy.get("rank_1_return_pct", 0.0)),
        f"{policy_name}_rank_1_mdd_pct": float(policy.get("rank_1_mdd_pct", 0.0)),
        f"{policy_name}_rank_1_trades": int(policy.get("rank_1_trades", 0) or 0),
        f"{policy_name}_best_oos": best,
        f"{policy_name}_best_gap": rank_1 - best,
        f"{policy_name}_0050_oos": bench,
        f"{policy_name}_0050_gap": rank_1 - bench,
    }


def _flatten_row_for_csv(row: dict) -> dict:
    flat = {
        "fold": row.get("fold"),
        "selection": row.get("selection_period"),
        "oos_year": row.get("oos_year"),
        "best_finalist_return_pct": float(row.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(row.get("benchmark_return_pct", 0.0)),
    }
    for policy_name in ("base", "local", "retention"):
        flat.update(_flatten_policy_for_csv(row, policy_name))
    flat["elapsed"] = _fmt_duration(row.get("elapsed_sec", 0.0))
    return flat


def _build_policies_schedule(rows: list[dict]) -> dict:
    policies: dict[str, dict] = {}
    for policy_name in ("base", "local", "retention"):
        policies[policy_name] = {
            "description": _policy_description(policy_name),
            "schedule": [dict(row.get("policy_schedules", {}).get(policy_name) or {}) for row in rows if row.get("policy_schedules", {}).get(policy_name)],
        }
    return policies


def _build_policy_paramset_payload(*, policy_name: str, rows: list[dict], config: OuterRollingConfig, summary: dict) -> dict:
    params_by_oos_year = {}
    params_by_effective_date = {}
    fold_entries = []
    for row in rows:
        schedule = dict((row.get("policy_schedules") or {}).get(policy_name) or {})
        if not schedule:
            continue
        oos_year = str(int(schedule.get("oos_year") or row.get("oos_year")))
        params_payload = dict(schedule.get("params") or {})
        params_by_oos_year[oos_year] = params_payload
        params_by_effective_date[str(schedule.get("effective_start") or f"{oos_year}-01-01")] = params_payload
        policy_metrics = dict(row.get(policy_name) or {})
        fold_entries.append({
            "fold": row.get("fold"),
            "selection_period": row.get("selection_period"),
            "oos_year": int(row.get("oos_year")),
            "effective_start": schedule.get("effective_start"),
            "effective_end": schedule.get("effective_end"),
            "selected_trial": schedule.get("selected_trial"),
            "base_score": schedule.get("base_score"),
            "base_rank": schedule.get("base_rank"),
            "local_min": schedule.get("local_min"),
            "local_rank": schedule.get("local_rank"),
            "retention": schedule.get("retention"),
            "retention_rank": schedule.get("retention_rank"),
            "oos_score": policy_metrics.get("rank_1_oos"),
            "return_pct": policy_metrics.get("rank_1_return_pct"),
            "mdd_pct": policy_metrics.get("rank_1_mdd_pct"),
            "trades": policy_metrics.get("rank_1_trades"),
            "benchmark_return_pct": row.get("benchmark_return_pct"),
            "benchmark_oos_score": row.get("benchmark_oos_score"),
            "best_finalist_return_pct": row.get("best_finalist_return_pct"),
            "best_finalist_oos_score": row.get("best_finalist_oos_score"),
        })
    chain_all = dict(summary.get("chained_oos") or {})
    chain_policy = dict(chain_all.get(policy_name) or {})
    chained_oos = {
        "method": chain_all.get("method"),
        "score_aggregation_method": chain_all.get("score_aggregation_method"),
        "return_aggregation_method": chain_all.get("return_aggregation_method"),
        "note": chain_all.get("note"),
        "selection_period": chain_all.get("selection_period"),
        "oos_period": chain_all.get("oos_period"),
        "rank_1_oos_score": float(chain_policy.get("rank_1_oos", 0.0)),
        "best_finalist_oos_score": float(chain_all.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": float(chain_all.get("benchmark_oos_score", 0.0)),
        "alpha_oos_score": float(chain_policy.get("benchmark_0050_gap", 0.0)),
        "best_gap_score": float(chain_policy.get("best_gap", 0.0)),
        "rank_1_return_pct": float(chain_policy.get("rank_1_return_pct", 0.0)),
        "best_finalist_return_pct": float(chain_all.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(chain_all.get("benchmark_return_pct", 0.0)),
        "alpha_return_pct": float(chain_policy.get("benchmark_0050_gap_pct", 0.0)),
        "best_gap_pct": float(chain_policy.get("best_gap_pct", 0.0)),
    }
    policy_summary = dict(summary.get(policy_name) or {})
    return {
        "schema_type": ROLLING_OOS_PARAM_SET_SCHEMA_TYPE,
        "schema_version": 1,
        "usage": ROLLING_OOS_USAGE,
        "type": "outer_rolling_oos_param_set",
        "created_at": get_taipei_now().isoformat(),
        "selector": str(policy_name),
        "meta": {
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "oos_horizon": "next_1y",
            "oos_feedback_used": False,
            "promotion_enabled": False,
            "trials_per_fold": int(config.trials_per_fold),
            "live_trading_param": False,
            "active_param_policy": "daily_active_param",
            "active_param_policy_note": "驗證 replay 時，每個交易日所有決策都使用該日期已生效的 active param；實盤同理使用當下正式 promote 的最新 param.json。",
        },
        "summary": {
            "folds": int(summary.get("folds", 0)),
            "selection_period": summary.get("selection_period"),
            "oos_period": summary.get("oos_period"),
            "aggregation_method": summary.get("aggregation_method"),
            "return_aggregation_method": summary.get("return_aggregation_method"),
            "selector": str(policy_name),
            "chained_oos_score": float(policy_summary.get("chained_oos_score", 0.0)),
            "chained_return_pct": float(policy_summary.get("chained_return_pct", 0.0)),
            "chained_gap_vs_0050_pct": float(policy_summary.get("chained_gap_vs_0050_pct", 0.0)),
            "yearly_avg_oos_score": float(policy_summary.get("yearly_avg_oos_score", policy_summary.get("avg_oos_score", 0.0))),
            "avg_oos_score": float(policy_summary.get("avg_oos_score", 0.0)),
            "median_oos_score": float(policy_summary.get("median_oos_score", 0.0)),
            "worst_oos_score": float(policy_summary.get("worst_oos_score", 0.0)),
            "positive_years": int(policy_summary.get("positive_years", 0)),
            "total_years": int(policy_summary.get("total_years", 0)),
        },
        "active_param_policy": "daily_active_param",
        "active_param_policy_note": "每日決策使用該日 active param；rolling 只是用歷史 effective date replay，不代表實盤使用年度參數組。",
        "chained_oos": chained_oos,
        "params_by_effective_date": params_by_effective_date,
        "params_by_oos_year": params_by_oos_year,
        "folds": fold_entries,
    }


def _write_policy_paramset_files(*, models_dir: str, rows: list[dict], config: OuterRollingConfig, summary: dict) -> dict:
    os.makedirs(models_dir, exist_ok=True)
    paths = {}
    for policy_name in ("base", "local", "retention"):
        payload = _build_policy_paramset_payload(policy_name=policy_name, rows=rows, config=config, summary=summary)
        path = os.path.join(models_dir, f"rolling_oos_{policy_name}_paramset.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        paths[policy_name] = path
    return paths


def _write_reports(*, project_root: str, output_dir: str, session_ts: str, rows: list[dict], config: OuterRollingConfig, chained_override: dict | None = None) -> dict:
    report_dir = os.path.join(output_dir, "outer_rolling_oos")
    os.makedirs(report_dir, exist_ok=True)
    models_dir = os.path.join(project_root, "models")
    base = os.path.join(report_dir, f"outer_rolling_oos_next1y_{session_ts}")
    json_path = base + ".json"
    csv_path = base + ".csv"
    txt_path = base + ".txt"
    summary = _build_summary(rows, config=config, chained_override=chained_override)
    paramset_paths = _write_policy_paramset_files(models_dir=models_dir, rows=rows, config=config, summary=summary)
    yearly_results = []
    for row in rows:
        light_row = dict(row)
        light_row.pop("policy_schedules", None)
        light_row.pop("best_finalist_equity_curve", None)
        light_row.pop("best_finalist_params", None)
        for policy_name in ("base", "local", "retention"):
            if isinstance(light_row.get(policy_name), dict):
                policy_copy = dict(light_row[policy_name])
                policy_copy.pop("rank_1_equity_curve", None)
                light_row[policy_name] = policy_copy
        yearly_results.append(light_row)
    payload = {
        "type": "outer_rolling_oos_next1y",
        "version": 1,
        "created_at": get_taipei_now().isoformat(),
        "meta": {
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "oos_horizon": "next_1y",
            "oos_feedback_used": False,
            "promotion_enabled": False,
            "trials_per_fold": int(config.trials_per_fold),
        },
        "gate_config": {
            "LOCAL_MIN_SCORE": True,
            "INNER_VALIDATE_RANK": bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED),
            "DOMINANT_YEAR_DEPENDENCY": bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED),
        },
        "summary": summary,
        "yearly_results": yearly_results,
        "policies": _build_policies_schedule(rows),
        "rolling_paramsets": paramset_paths,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    if rows:
        flat_rows = [_flatten_row_for_csv(row) for row in rows]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_final_report(rows, summary, color=False))
    return {"json": json_path, "csv": csv_path, "txt": txt_path, "paramsets": paramset_paths}


def _build_summary(rows: list[dict], *, config: OuterRollingConfig | None = None, chained_override: dict | None = None) -> dict:
    if not rows:
        return {"folds": 0}
    selection_start = min(int(row.get("selection_start_year", 0) or 0) for row in rows)
    selection_end = max(int(row.get("selection_end_year", 0) or 0) for row in rows)
    first_oos = min(int(row["oos_year"]) for row in rows)
    last_oos = max(int(row["oos_year"]) for row in rows)
    chained = _build_chained_oos_summary(rows, chained_override=chained_override)
    summary = {
        "folds": len(rows),
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{first_oos}~{last_oos}",
        "aggregation_method": chained.get("score_aggregation_method") or chained.get("method"),
        "return_aggregation_method": chained.get("return_aggregation_method"),
        "note": chained.get("note"),
        "chained_oos": chained,
    }
    for policy_name in ("base", "local", "retention"):
        scores = [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows]
        returns = [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows]
        benchmark_gaps = [float((row.get(policy_name) or {}).get("benchmark_0050_gap", 0.0)) for row in rows]
        chain_policy = dict(chained.get(policy_name) or {})
        yearly_avg_score = sum(scores) / float(len(scores))
        summary[policy_name] = {
            "chained_oos_score": float(chain_policy.get("rank_1_oos", 0.0)),
            "chained_return_pct": float(chain_policy.get("rank_1_return_pct", 0.0)),
            "chained_gap_vs_best_pct": float(chain_policy.get("best_gap_pct", 0.0)),
            "chained_gap_vs_0050_pct": float(chain_policy.get("benchmark_0050_gap_pct", 0.0)),
            "yearly_avg_oos_score": float(yearly_avg_score),
            "avg_oos_score": float(yearly_avg_score),
            "median_oos_score": float(statistics.median(scores)),
            "worst_oos_score": min(scores),
            "positive_years": sum(1 for value in returns if value > 0.0),
            "win_vs_0050_score": sum(1 for gap in benchmark_gaps if gap > 0.0),
            "total_years": len(scores),
            "yearly_return_pct": returns,
            "yearly_oos_score": scores,
        }
    benchmark_returns = [float(row.get("benchmark_return_pct", 0.0)) for row in rows]
    benchmark_scores = [float(row.get("benchmark_oos_score", 0.0)) for row in rows]
    benchmark_yearly_avg_score = sum(benchmark_scores) / float(len(benchmark_scores))
    summary["benchmark_0050"] = {
        "chained_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "chained_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "yearly_avg_oos_score": float(benchmark_yearly_avg_score),
        "avg_oos_score": float(benchmark_yearly_avg_score),
        "positive_years": sum(1 for value in benchmark_returns if value > 0.0),
        "total_years": len(benchmark_returns),
        "yearly_return_pct": benchmark_returns,
        "yearly_oos_score": benchmark_scores,
    }
    if config is not None:
        summary["window_mode"] = str(config.window_mode)
        summary["train_window_years"] = int(config.train_window_years)
    return summary

def _format_final_report(rows: list[dict], summary: dict, *, color: bool = False) -> str:
    rendered = _render_results_table(rows, color=color, include_chain=True, chained_override=summary.get("chained_oos"))
    lines = []
    if rendered:
        lines.append(rendered)
    return "\n".join(lines) + "\n"



class _ParallelFoldProgressLogFilter:
    def __init__(self, handle):
        self.handle = handle
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        raw = str(text)
        self._buffer += raw
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
                self.handle.write(line + "\n")
                self.handle.flush()
        return len(raw)

    def flush(self):
        if self._buffer.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            self.handle.write(self._buffer)
            self._buffer = ""
        self.handle.flush()


def _tail_text_file(path: str, *, max_lines: int = 8) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-int(max_lines):]).strip()
    except OSError:
        return ""


def _latest_parallel_fold_log_status(path: str, *, max_chars: int = PARALLEL_FOLD_LOG_STATUS_MAX_CHARS) -> str:
    tail = _tail_text_file(path, max_lines=12)
    if not tail:
        return ""
    for line in reversed(tail.splitlines()):
        text = str(line).strip()
        if not text:
            continue
        text = " ".join(text.split())
        if len(text) > int(max_chars):
            text = text[: max(0, int(max_chars) - 3)] + "..."
        return text
    return ""


def _safe_progress_json_loads(raw_text: str) -> dict:
    try:
        payload = json.loads(str(raw_text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_latest_parallel_fold_progress(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return {}
    for line in reversed(lines[-240:]):
        raw = str(line).strip()
        if not raw.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            continue
        payload = _safe_progress_json_loads(raw.split("\t", 1)[1])
        if payload:
            return payload
    return {}


def _write_parallel_fold_progress_event(*, stage: str, fold_idx: int, fold_count: int, oos_year: int, selection_start: int, selection_end: int, **payload) -> None:
    event = {
        "stage": str(stage),
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "oos_year": int(oos_year),
        "selection_start": int(selection_start),
        "selection_end": int(selection_end),
        "ts": time.time(),
    }
    event.update(payload)
    print(PARALLEL_FOLD_PROGRESS_PREFIX + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _parallel_progress_pct(done, total) -> float:
    try:
        total_float = float(total)
        if total_float <= 0.0:
            return 0.0
        return 100.0 * float(done) / total_float
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _format_parallel_fold_progress_line(task: dict, progress: dict, *, log_status: str = "") -> str:
    fold_idx = int(task.get("fold_idx", progress.get("fold_idx", 0)) or 0)
    fold_count = int(task.get("fold_count", progress.get("fold_count", 0)) or 0)
    oos_year = int(task.get("oos_year", progress.get("oos_year", 0)) or 0)
    selection_start = int(progress.get("selection_start", 0) or 0)
    selection_end = int(progress.get("selection_end", 0) or 0)
    if selection_start <= 0 or selection_end <= 0:
        # AI註: task 只含 OOS year；平行狀態列沒有 config 物件時保留 OOS 即可。
        selection_text = "selection=?"
    else:
        selection_text = f"selection={selection_start}~{selection_end}"
    stage = str(progress.get("stage") or "QUEUED").upper()
    elapsed_text = ""
    if progress.get("elapsed_sec") is not None:
        elapsed_text = f" | elapsed={_fmt_duration_compact(progress.get('elapsed_sec'))}"
    if stage == "OPTIMIZER_SEARCH":
        completed = int(progress.get("completed", 0) or 0)
        total = int(progress.get("total", 0) or 0)
        best = progress.get("best_score")
        best_text = "N/A" if best is None else f"{float(best):.3f}"
        return (
            f"[{fold_idx}/{fold_count}] {selection_text} | OOS {oos_year} | "
            f"trial {completed}/{total} | best={best_text}{elapsed_text}"
        )
    if stage == "LOCAL_MIN_REVIEW":
        finalist_idx = int(progress.get("finalist_idx", 0) or 0)
        finalist_total = int(progress.get("finalist_total", 0) or 0)
        neighbor_done = int(progress.get("neighbor_done", 0) or 0)
        neighbor_total = int(progress.get("neighbor_total", 0) or 0)
        current = progress.get("current")
        current_text = "N/A" if current is None else f"{float(current):.3f}"
        best = progress.get("best")
        best_text = "N/A" if best is None else f"{float(best):.3f}"
        status = str(progress.get("status") or "RUN")
        return (
            f"[{fold_idx}/{fold_count}] {selection_text} | OOS {oos_year} | "
            f"local_min finalist {finalist_idx}/{finalist_total} | neighbor {neighbor_done}/{neighbor_total} | "
            f"current={current_text} | best={best_text} | {status}{elapsed_text}"
        )
    if stage == "OOS_DIAGNOSTICS":
        status = str(progress.get("status") or "RUN")
        return f"[{fold_idx}/{fold_count}] {selection_text} | OOS {oos_year} | diagnostics {status}{elapsed_text}"
    if stage == "DONE":
        status = str(progress.get("status") or "done")
        return f"[{fold_idx}/{fold_count}] {selection_text} | OOS {oos_year} | DONE {status}{elapsed_text}"
    if stage in {"START", "RAW_DATA", "STUDY_CREATE"}:
        status = str(progress.get("status") or stage).replace("_", " ")
        return f"[{fold_idx}/{fold_count}] {selection_text} | OOS {oos_year} | {status}{elapsed_text}"
    fallback = str(log_status or "queued").strip()
    return f"[{fold_idx}/{fold_count}] OOS {oos_year} | {fallback}"


class _ParallelFoldProgressBoard:
    def __init__(self, tasks: list[dict]):
        self.tasks = sorted(list(tasks or []), key=lambda item: int(item.get("fold_idx", 0) or 0))
        self.lines: dict[int, str] = {}
        self.rendered_lines = 0
        self.inline = stdout_supports_inline_progress()
        self.started_at = time.perf_counter()

    def update(self, *, pending: set, future_map: dict, completed_rows: list[dict], force: bool = False) -> None:
        pending_tasks = {id(future_map[future]): future_map[future] for future in pending if future in future_map}
        completed_oos = {int(row.get("oos_year", 0) or 0) for row in list(completed_rows or [])}
        new_lines: dict[int, str] = {}
        for task in self.tasks:
            fold_idx = int(task.get("fold_idx", 0) or 0)
            log_path = str(task.get("log_path") or "")
            progress = _read_latest_parallel_fold_progress(log_path)
            log_status = _latest_parallel_fold_log_status(log_path)
            if int(task.get("oos_year", 0) or 0) in completed_oos and not progress:
                progress = {"stage": "DONE", "status": "done"}
            new_lines[fold_idx] = _format_parallel_fold_progress_line(task, progress, log_status=log_status)
        if not force and new_lines == self.lines:
            return
        self.lines = new_lines
        header = (
            f"⏱️ Rolling fold parallel | completed={len(completed_rows)}/{len(self.tasks)} | "
            f"pending={len(pending)} | elapsed={_fmt_duration(time.perf_counter() - self.started_at)}"
        )
        output_lines = [f"{C_CYAN}{header}{C_RESET}"] + [f"{C_GRAY}  {line}{C_RESET}" for _, line in sorted(new_lines.items())]
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in output_lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(output_lines)
        else:
            print("\n".join(output_lines), flush=True)

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.rendered_lines = 0


class _ParallelCompletedResultsBoard:
    def __init__(self):
        self.rendered_lines = 0
        self.inline = stdout_supports_inline_progress()

    def render(self, rows: list[dict]) -> None:
        completed = sorted(list(rows or []), key=lambda item: int(item.get("oos_year", 0) or 0))
        if not completed:
            return
        table = _render_results_table(completed, color=True, include_chain=False, include_oos_avg=True)
        if not table:
            return
        table_lines = table.splitlines()
        if table_lines and table_lines[0].strip().upper() == "ROLLING NEXT-1Y OOS RESULTS":
            table_lines = table_lines[1:]
        lines = table_lines
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
        else:
            print("\n".join(lines), flush=True)

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.rendered_lines = 0


class _ParallelFoldLiveBoard:
    """Render parallel-fold progress and completed results as one refresh block."""

    def __init__(self, tasks: list[dict], *, overall_start: float | None = None, raw_data_load_sec: float = 0.0):
        self.tasks = sorted(list(tasks or []), key=lambda item: int(item.get("fold_idx", 0) or 0))
        self.inline = stdout_supports_inline_progress()
        self.started_at = time.perf_counter()
        self.overall_start = float(overall_start) if overall_start is not None else self.started_at
        self.raw_data_load_sec = max(0.0, float(raw_data_load_sec or 0.0))
        self.rendered_lines = 0
        self.last_lines: list[str] = []
        self.last_render_key: list[str] = []

    def _build_lines(self, *, pending: set, future_map: dict, completed_rows: list[dict], fold_timing_rows: list[dict] | None = None) -> list[str]:
        completed_rows_sorted = sorted(list(completed_rows or []), key=lambda item: int(item.get("oos_year", 0) or 0))
        completed_oos = {int(row.get("oos_year", 0) or 0) for row in completed_rows_sorted}
        fold_wall_elapsed = max(0.0, time.perf_counter() - self.started_at)
        total_elapsed = max(0.0, time.perf_counter() - self.overall_start)
        setup_elapsed = max(0.0, self.started_at - self.overall_start)
        # parallel fold 模式下，raw data 由各 fold worker 自行載入，
        # worker raw 時間已包含在 folds wall 裡；不要再以 raw/other 顯示，避免與總時間對帳時重複或缺項。
        header = (
            f"⏱️ Rolling fold parallel | completed={len(completed_rows_sorted)}/{len(self.tasks)} | "
            f"pending={len(pending)} | total={_fmt_duration(total_elapsed)} | "
            f"folds={_fmt_duration(fold_wall_elapsed)} | setup={_fmt_duration(setup_elapsed)}"
        )
        lines: list[str] = [f"{C_CYAN}{header}{C_RESET}"]
        for task in self.tasks:
            log_path = str(task.get("log_path") or "")
            progress = _read_latest_parallel_fold_progress(log_path)
            if int(task.get("oos_year", 0) or 0) in completed_oos and not progress:
                progress = {
                    "stage": "DONE",
                    "status": "done",
                    "fold_idx": int(task.get("fold_idx", 0) or 0),
                    "fold_count": int(task.get("fold_count", 0) or 0),
                    "oos_year": int(task.get("oos_year", 0) or 0),
                }
            log_status = _latest_parallel_fold_log_status(log_path)
            lines.append(f"{C_GRAY}  {_format_parallel_fold_progress_line(task, progress, log_status=log_status)}{C_RESET}")
        if completed_rows_sorted:
            table = _render_results_table(completed_rows_sorted, color=True, include_chain=False, include_oos_avg=True)
            if table:
                lines.append("")
                lines.extend(table.splitlines())
        return lines

    @staticmethod
    def _stable_render_key(lines: list[str]) -> list[str]:
        # Ignore elapsed-only heartbeat changes; refresh when fold progress or completed results change.
        key: list[str] = []
        for line in list(lines or []):
            text = str(line)
            if "⏱️ 耗時摘要" in text:
                text = "⏱️ 耗時摘要"
            elif "⏱️ Rolling fold parallel" in text:
                for marker in (" | total=", " | elapsed="):
                    if marker in text:
                        text = text.split(marker, 1)[0]
                        break
            elif " | elapsed=" in text:
                text = text.split(" | elapsed=", 1)[0]
            key.append(text)
        return key

    def render(self, *, pending: set, future_map: dict, completed_rows: list[dict], fold_timing_rows: list[dict] | None = None, force: bool = False) -> None:
        lines = self._build_lines(pending=pending, future_map=future_map, completed_rows=completed_rows, fold_timing_rows=fold_timing_rows)
        render_key = self._stable_render_key(lines)
        if not force and render_key == self.last_render_key:
            return
        self.last_lines = list(lines)
        self.last_render_key = list(render_key)
        if self.inline:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                sys.stdout.write("\r" + line + "\x1b[K\n")
            if self.rendered_lines > len(lines):
                for _ in range(self.rendered_lines - len(lines)):
                    sys.stdout.write("\r\x1b[K\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
        else:
            # Non-interactive outputs cannot refresh safely; print only on meaningful changes.
            print("\n".join(lines), flush=True)
            self.rendered_lines = 0

    def close(self) -> None:
        if self.inline and self.rendered_lines > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.rendered_lines = 0



def _print_parallel_fold_result(row: dict) -> None:
    if not row:
        return
    print(f"{C_CYAN}📌 Parallel fold result | fold={row.get('fold')} | selection={row.get('selection_period')} | OOS={row.get('oos_year')}{C_RESET}", flush=True)
    rendered = _render_results_table([row], color=True, include_chain=False)
    if rendered:
        print(rendered, flush=True)


def _consume_parallel_fold_future(*, future, task: dict, rows: list[dict], fold_timing_rows: list[dict], chain_state: dict) -> None:
    fold_idx = int(task["fold_idx"])
    fold_count = int(task.get("fold_count", 0) or 0)
    oos_year = int(task["oos_year"])
    try:
        result = future.result()
    except Exception as exc:
        log_path = str(task.get("log_path") or "")
        tail = _tail_text_file(log_path, max_lines=12)
        detail = f"\n最後 fold log：\n{tail}" if tail else ""
        raise RuntimeError(f"parallel fold failed: fold={fold_idx}/{fold_count} OOS={oos_year} log={log_path}{detail}") from exc
    row = result.get("row")
    if row is not None:
        rows.append(row)
    if result.get("timing_row") is not None:
        fold_timing_rows.append(result["timing_row"])
    if result.get("chain_max_positions") is not None:
        chain_state["chain_max_positions"] = int(result.get("chain_max_positions"))
    if result.get("chain_enable_rotation") is not None:
        chain_state["chain_enable_rotation"] = bool(result.get("chain_enable_rotation"))


def _run_parallel_fold_futures(*, executor, tasks: list[dict], rows: list[dict], fold_timing_rows: list[dict], overall_start: float | None = None, raw_data_load_sec: float = 0.0) -> dict:
    future_map = {executor.submit(_run_outer_rolling_oos_fold_task, task): task for task in tasks}
    pending = set(future_map)
    chain_state = {"chain_max_positions": None, "chain_enable_rotation": None}
    live_board = _ParallelFoldLiveBoard(tasks, overall_start=overall_start, raw_data_load_sec=raw_data_load_sec)
    live_board.render(pending=pending, future_map=future_map, completed_rows=rows, fold_timing_rows=fold_timing_rows, force=True)
    try:
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: int(future_map[item].get("fold_idx", 0) or 0)):
                _consume_parallel_fold_future(
                    future=future,
                    task=future_map[future],
                    rows=rows,
                    fold_timing_rows=fold_timing_rows,
                    chain_state=chain_state,
                )
            live_board.render(pending=pending, future_map=future_map, completed_rows=rows, fold_timing_rows=fold_timing_rows, force=bool(done))
    finally:
        live_board.close()
    return chain_state

class _FoldLogSearchProgress:
    def __init__(self, *, fold_idx: int, fold_count: int, oos_year: int, selection_start: int, selection_end: int, total_trials: int):
        self.fold_idx = int(fold_idx)
        self.fold_count = int(fold_count)
        self.oos_year = int(oos_year)
        self.selection_start = int(selection_start)
        self.selection_end = int(selection_end)
        self.total_trials = int(total_trials)
        self.stage_start = time.perf_counter()
        self.best_score = float("-inf")
        self.last_completed = -1

    def emit(self, completed: int, *, force: bool = False) -> None:
        completed = int(completed)
        if not force and completed == self.last_completed:
            return
        self.last_completed = completed
        best_score = None if self.best_score == float("-inf") else float(self.best_score)
        _write_parallel_fold_progress_event(
            stage="OPTIMIZER_SEARCH",
            fold_idx=self.fold_idx,
            fold_count=self.fold_count,
            oos_year=self.oos_year,
            selection_start=self.selection_start,
            selection_end=self.selection_end,
            completed=completed,
            total=self.total_trials,
            best_score=best_score,
            elapsed_sec=max(0.0, time.perf_counter() - self.stage_start),
        )

    def callback(self, session):
        def _callback(study, trial):
            session.current_session_trial += 1
            if trial.value is not None and is_qualified_trial_value(trial.value):
                self.best_score = max(self.best_score, float(trial.value))
            self.emit(int(session.current_session_trial))
        return _callback

    def done(self, completed: int) -> None:
        self.emit(int(completed), force=True)


def _run_outer_rolling_oos_fold_task(task: dict) -> dict:
    """Run one rolling fold in an isolated process for timing-mode fold parallelism."""
    log_path = str((task or {}).get("log_path") or "")

    def _execute() -> dict:
        from tools.optimizer.main import build_optimizer_session, configure_optuna_logging, _ensure_study_effective_policy_compatible
        from tools.optimizer.prep import load_all_raw_data
        from tools.optimizer.runtime import create_optimizer_study
        from tools.optimizer.session import close_study_storage

        configure_optuna_logging()
        base_policy = dict(task["base_policy"])
        config_payload = dict(task["config"])
        config = OuterRollingConfig(
            training_start_year=int(config_payload["training_start_year"]),
            first_oos_year=int(config_payload["first_oos_year"]),
            last_oos_year=int(config_payload["last_oos_year"]),
            trials_per_fold=int(config_payload["trials_per_fold"]),
            window_mode=str(config_payload.get("window_mode", "fixed")),
            train_window_years=int(config_payload.get("train_window_years", 5)),
            confirm=False,
        )
        output_dir = str(task["output_dir"])
        selected_data_dir = str(task["selected_data_dir"])
        session_ts = str(task["session_ts"])
        optimizer_seed = task.get("optimizer_seed")
        sampler_kind = str(task.get("sampler_kind", "random") or "random")
        fold_idx = int(task["fold_idx"])
        fold_count = int(task["fold_count"])
        oos_year = int(task["oos_year"])
        optimizer_required_min_rows = int(task["optimizer_required_min_rows"])
        fold_workers = int(task.get("fold_workers", 1) or 1)
        timing_mode = bool(task.get("timing_mode", False))
        if not str(os.environ.get("OPTIMIZER_PREP_CACHE_MAX_ITEMS", "")).strip():
            os.environ["OPTIMIZER_PREP_CACHE_MAX_ITEMS"] = str(_resolve_parallel_worker_prep_cache_max_items(os.environ))

        fold_start = time.perf_counter()
        selection_end = int(oos_year) - 1
        selection_start = _selection_start_for_oos(config, int(oos_year))
        print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | parallel fold START | selection={selection_start}~{selection_end}", flush=True)
        _write_parallel_fold_progress_event(
            stage="START",
            fold_idx=fold_idx,
            fold_count=fold_count,
            oos_year=oos_year,
            selection_start=selection_start,
            selection_end=selection_end,
            status="START",
            elapsed_sec=0.0,
        )
        fold_policy = dict(base_policy)
        fold_policy["selection_start_year"] = int(selection_start)
        fold_policy["train_start_year"] = int(selection_start)
        fold_policy["search_train_end_year"] = int(selection_end)
        fold_policy["oos_start_year"] = int(oos_year)
        fold_policy = build_optimizer_runtime_policy(fold_policy, "split")
        objective_mode = str(fold_policy.get("objective_mode", "split_train_romd"))
        session = build_optimizer_session(walk_forward_policy=fold_policy)
        session.rolling_fold_workers_max = int(fold_workers)
        session.rolling_fold_parallel = True
        reset_prep_cache_stats = getattr(session, "reset_prep_cache_stats", None)
        if callable(reset_prep_cache_stats):
            reset_prep_cache_stats()
        chain_max_positions = int(session.train_max_positions)
        chain_enable_rotation = bool(session.train_enable_rotation)
        session.n_trials = int(config.trials_per_fold)
        session.disable_milestone_dashboard = True
        session.timing_mode = bool(timing_mode)
        sqlite_storage_enabled = _is_outer_rolling_sqlite_storage_enabled(os.environ)
        db_file = ""
        db_name = None
        if sqlite_storage_enabled:
            db_dir = os.path.join(output_dir, "outer_rolling_oos", "db")
            os.makedirs(db_dir, exist_ok=True)
            db_file = os.path.join(db_dir, f"outer_oos_{session_ts}_{int(oos_year)}.db")
            db_name = f"sqlite:///{db_file}"
        study = None
        try:
            install_started = time.perf_counter()
            session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            install_shared_cache_sec = max(0.0, time.perf_counter() - install_started)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | raw data loaded | elapsed={_fmt_duration(install_shared_cache_sec)}", flush=True)
            _write_parallel_fold_progress_event(
                stage="RAW_DATA",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                status="raw data loaded",
                elapsed_sec=max(0.0, time.perf_counter() - fold_start),
            )
            session.profile_recorder.init_output_files()
            session.profile_recorder.mark_run_started()
            study_started = time.perf_counter()
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=sampler_kind)
            _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=fold_policy)
            study_create_sec = max(0.0, time.perf_counter() - study_started)
            progress = _FoldLogSearchProgress(
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                total_trials=int(config.trials_per_fold),
            )
            progress.emit(0, force=True)
            optimize_started = time.perf_counter()
            study.optimize(session.objective, n_trials=int(config.trials_per_fold), n_jobs=1, callbacks=[progress.callback(session)])
            optimize_sec = max(0.0, time.perf_counter() - optimize_started)
            trial_count = len(list(getattr(study, "trials", []) or []))
            session.current_session_trial = int(trial_count or config.trials_per_fold)
            progress.done(int(session.current_session_trial))
            print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | optimizer search DONE | elapsed={_fmt_duration(optimize_sec)}", flush=True)

            local_started = time.perf_counter()

            def _parallel_local_min_progress_sink(event: dict) -> None:
                data = dict(event or {})
                _write_parallel_fold_progress_event(
                    stage="LOCAL_MIN_REVIEW",
                    fold_idx=fold_idx,
                    fold_count=fold_count,
                    oos_year=oos_year,
                    selection_start=selection_start,
                    selection_end=selection_end,
                    finalist_idx=int(data.get("finalist_idx", 0) or 0),
                    finalist_total=int(data.get("finalist_total", 0) or 0),
                    trial_number=data.get("trial_number"),
                    neighbor_done=int(data.get("neighbor_done", 0) or 0),
                    neighbor_total=int(data.get("neighbor_total", 0) or 0),
                    current=data.get("current"),
                    best=data.get("best"),
                    status=str(data.get("status") or "RUN"),
                    elapsed_sec=max(0.0, time.perf_counter() - local_started),
                )

            session.outer_rolling_parallel_progress_sink = _parallel_local_min_progress_sink
            session.outer_rolling_local_progress_context = {
                "fold_idx": int(fold_idx),
                "fold_count": int(fold_count),
                "oos_year": int(oos_year),
                "selection_start": int(selection_start),
                "selection_end": int(selection_end),
                "completed_results": [],
                "overall_start": fold_start,
            }
            finalists = list_local_min_score_finalists(
                study,
                session=session,
                objective_mode=objective_mode,
                include_trial=None,
                show_progress=True,
                include_oos_diagnostics=False,
                single_finalist_fast_path=True,
                selection_pruning=True,
            )
            if hasattr(session, "outer_rolling_parallel_progress_sink"):
                delattr(session, "outer_rolling_parallel_progress_sink")
            if hasattr(session, "outer_rolling_local_progress_context"):
                delattr(session, "outer_rolling_local_progress_context")
            local_elapsed = time.perf_counter() - local_started
            print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | local-min review DONE | finalists={len(finalists)} | elapsed={_fmt_duration(local_elapsed)}", flush=True)
            _write_parallel_fold_progress_event(
                stage="LOCAL_MIN_REVIEW",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                finalist_idx=len(finalists),
                finalist_total=len(finalists),
                neighbor_done=1,
                neighbor_total=1,
                current=float(finalists[0].get("local_min_score", 0.0)) if finalists else None,
                best=float(finalists[0].get("local_min_score", 0.0)) if finalists else None,
                status="DONE",
                elapsed_sec=local_elapsed,
            )
            policy_items = _build_policy_items(finalists, objective_mode=objective_mode)
            if not any(item is not None for item in policy_items.values()):
                fold_elapsed = time.perf_counter() - fold_start
                timing_row = _build_outer_timing_row(
                    fold_idx=fold_idx,
                    fold_count=fold_count,
                    oos_year=int(oos_year),
                    selection_start=int(selection_start),
                    selection_end=int(selection_end),
                    status="skipped_no_finalist",
                    session=session,
                    db_file=db_file,
                    install_shared_cache_sec=install_shared_cache_sec,
                    study_create_sec=study_create_sec,
                    optimize_sec=optimize_sec,
                    local_min_review_sec=local_elapsed,
                    oos_diagnostics_sec=0.0,
                    fold_total_sec=fold_elapsed,
                    finalists_count=len(finalists),
                )
                return {
                    "status": "skipped_no_finalist",
                    "fold_idx": fold_idx,
                    "oos_year": int(oos_year),
                    "row": None,
                    "timing_row": timing_row,
                    "chain_max_positions": chain_max_positions,
                    "chain_enable_rotation": chain_enable_rotation,
                    "log_path": log_path,
                }
            local_rank_map = _build_local_rank_map(finalists)
            retention_rank_map = _build_retention_rank_map(finalists)
            diagnostics_started = time.perf_counter()
            _write_parallel_fold_progress_event(
                stage="OOS_DIAGNOSTICS",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                status="RUN",
                elapsed_sec=0.0,
            )
            diagnostics = _evaluate_finalist_oos_diagnostics(
                session=session,
                finalists=finalists,
                policy_items=policy_items,
                oos_year=int(oos_year),
            )
            oos_diagnostics_sec = max(0.0, time.perf_counter() - diagnostics_started)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | OOS diagnostics DONE | elapsed={_fmt_duration(oos_diagnostics_sec)}", flush=True)
            _write_parallel_fold_progress_event(
                stage="OOS_DIAGNOSTICS",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                status="DONE",
                elapsed_sec=oos_diagnostics_sec,
            )
            fold_elapsed = time.perf_counter() - fold_start
            selection_period = f"{selection_start}~{selection_end}"
            policy_schedules = {
                name: _build_policy_schedule_entry(
                    item=item,
                    policy_name=name,
                    oos_year=int(oos_year),
                    selection_period=selection_period,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                )
                for name, item in policy_items.items()
                if item is not None
            }
            row = {
                "fold": f"{fold_idx}/{fold_count}",
                "oos_year": int(oos_year),
                "selection_period": selection_period,
                "selection_start_year": int(selection_start),
                "selection_end_year": int(selection_end),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", 0.0)),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "best_finalist_return_pct": float(diagnostics.get("best_finalist_return_pct", 0.0)),
                "best_finalist_mdd_pct": float(diagnostics.get("best_finalist_mdd_pct", 0.0)),
                "best_finalist_trades": int(diagnostics.get("best_finalist_trades", 0) or 0),
                "best_finalist_initial_capital": float(diagnostics.get("best_finalist_initial_capital", 0.0)),
                "best_finalist_equity_curve": list(diagnostics.get("best_finalist_equity_curve") or []),
                "best_finalist_params": dict(diagnostics.get("best_finalist_params") or {}),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", 0.0)),
                "benchmark_return_pct": float(diagnostics.get("benchmark_return_pct", 0.0)),
                "benchmark_mdd_pct": float(diagnostics.get("benchmark_mdd_pct", 0.0)),
                "base": diagnostics.get("policies", {}).get("base", {}),
                "local": diagnostics.get("policies", {}).get("local", {}),
                "retention": diagnostics.get("policies", {}).get("retention", {}),
                "policy_schedules": policy_schedules,
                "elapsed_sec": float(fold_elapsed),
                "optimizer_search_sec": float(optimize_sec),
                "local_min_review_sec": float(local_elapsed),
                "oos_diagnostics_sec": float(oos_diagnostics_sec),
                "install_shared_cache_sec": float(install_shared_cache_sec),
                "study_create_sec": float(study_create_sec),
            }
            timing_row = _build_outer_timing_row(
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=int(oos_year),
                selection_start=int(selection_start),
                selection_end=int(selection_end),
                status="done",
                session=session,
                db_file=db_file,
                install_shared_cache_sec=install_shared_cache_sec,
                study_create_sec=study_create_sec,
                optimize_sec=optimize_sec,
                local_min_review_sec=local_elapsed,
                oos_diagnostics_sec=oos_diagnostics_sec,
                fold_total_sec=fold_elapsed,
                finalists_count=len(finalists),
            )
            _write_parallel_fold_progress_event(
                stage="DONE",
                fold_idx=fold_idx,
                fold_count=fold_count,
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                status="done",
                elapsed_sec=fold_elapsed,
            )
            return {
                "status": "done",
                "fold_idx": fold_idx,
                "oos_year": int(oos_year),
                "row": row,
                "timing_row": timing_row,
                "chain_max_positions": chain_max_positions,
                "chain_enable_rotation": chain_enable_rotation,
                "log_path": log_path,
            }
        finally:
            session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_handle:
            progress_log = _ParallelFoldProgressLogFilter(log_handle)
            with redirect_stdout(progress_log), redirect_stderr(log_handle):
                return _execute()
    return _execute()



def run_outer_rolling_oos(
    *,
    argv,
    environ,
    project_root: str,
    output_dir: str,
    base_policy: dict,
    selected_data_dir: str,
    dataset_label: str,
    load_all_raw_data,
    optimizer_required_min_rows: int,
    build_optimizer_session,
    create_optimizer_study,
    ensure_study_effective_policy_compatible,
    configure_optuna_logging,
    optimizer_seed=None,
    default_trials: int = 500,
    timing_mode: bool = False,
) -> int:
    from tools.optimizer.session import close_study_storage

    session_ts = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(output_dir, "outer_rolling_oos"), exist_ok=True)

    latest_year = _resolve_latest_year_from_csv_data_dir(selected_data_dir)

    config = _resolve_config(argv, environ, base_policy=base_policy, latest_year=latest_year, default_trials=default_trials, timing_mode=bool(timing_mode))
    years = list(range(config.first_oos_year, config.last_oos_year + 1))
    _apply_outer_rolling_resource_env_defaults(environ, timing_mode=bool(timing_mode), fold_count=len(years))
    fold_workers = _resolve_rolling_fold_workers(environ, timing_mode=bool(timing_mode), fold_count=len(years))
    fold_parallel_enabled = _is_rolling_fold_parallel_enabled(environ, timing_mode=bool(timing_mode), fold_count=len(years))
    _print_plan(config, parallel_settings_line=_format_parallel_settings_line(environ, fold_workers=int(fold_workers)) if fold_parallel_enabled else None)
    if not _confirm_plan(config):
        print(f"{C_YELLOW}已取消 outer rolling OOS。{C_RESET}")
        return 0

    configure_optuna_logging()
    rows: list[dict] = []
    fold_timing_rows: list[dict] = []
    chain_max_positions: int | None = None
    chain_enable_rotation: bool | None = None
    rolling_shared_prep_cache = OrderedDict()
    rolling_shared_prep_cache_max_items = _resolve_rolling_shared_prep_cache_max_items(environ)
    rolling_shared_local_min_order_score_cache = {}
    rolling_shared_local_min_field_order_score_cache = {}
    rolling_shared_prep_executor_holder = {}
    overall_start = time.perf_counter()
    resource_sampler = _ResourceUsageSampler(interval_sec=_resolve_resource_sample_interval_sec(environ))
    resource_sampler.start()
    sampler_kind = "random" if bool(timing_mode) else "tpe"
    shared_raw_context = None
    if fold_parallel_enabled:
        raw_data_load_sec = 0.0
        print(f"{C_CYAN}⏱️ Rolling 資料載入模式：parallel folds 自行載入 raw cache | folds={len(years)}{C_RESET}")
    else:
        shared_load_start = time.perf_counter()
        shared_data_policy = build_optimizer_runtime_policy(dict(base_policy), "split")
        shared_data_session = build_optimizer_session(walk_forward_policy=shared_data_policy)
        try:
            shared_data_session.load_raw_data(selected_data_dir, load_all_raw_data=load_all_raw_data, required_min_rows=optimizer_required_min_rows)
            shared_raw_context = {
                "raw_data_cache": dict(shared_data_session.raw_data_cache),
                "static_fast_cache": dict(shared_data_session.static_fast_cache),
                "master_dates": set(shared_data_session.master_dates),
                "sorted_master_dates": list(shared_data_session.sorted_master_dates),
            }
        finally:
            shared_data_session.close_trial_prep_executor()
        raw_data_load_sec = max(0.0, time.perf_counter() - shared_load_start)
        print(f"{C_CYAN}⏱️ Rolling 共用資料快取完成：raw_data_load_once={raw_data_load_sec:.3f}s | folds={len(years)}{C_RESET}")

    if fold_parallel_enabled:
        log_dir = os.path.join(output_dir, "outer_rolling_oos", "fold_logs", session_ts)
        os.makedirs(log_dir, exist_ok=True)
        config_payload = {
            "training_start_year": int(config.training_start_year),
            "first_oos_year": int(config.first_oos_year),
            "last_oos_year": int(config.last_oos_year),
            "trials_per_fold": int(config.trials_per_fold),
            "window_mode": str(config.window_mode),
            "train_window_years": int(config.train_window_years),
        }
        tasks = []
        for fold_idx, oos_year in enumerate(years, start=1):
            selection_end = int(oos_year) - 1
            selection_start = _selection_start_for_oos(config, int(oos_year))
            tasks.append({
                "project_root": str(project_root),
                "output_dir": str(output_dir),
                "base_policy": dict(base_policy),
                "selected_data_dir": str(selected_data_dir),
                "optimizer_required_min_rows": int(optimizer_required_min_rows),
                "session_ts": str(session_ts),
                "config": dict(config_payload),
                "fold_idx": int(fold_idx),
                "fold_count": int(len(years)),
                "oos_year": int(oos_year),
                "optimizer_seed": optimizer_seed,
                "sampler_kind": str(sampler_kind),
                "timing_mode": bool(timing_mode),
                "fold_workers": int(fold_workers),
                "log_path": os.path.join(log_dir, f"fold_{int(fold_idx):02d}_oos_{int(oos_year)}.log"),
            })
            # Fold log path is kept internally for diagnostics, but not printed during normal progress.
        with ProcessPoolExecutor(max_workers=int(fold_workers)) as executor:
            chain_state = _run_parallel_fold_futures(
                executor=executor,
                tasks=tasks,
                rows=rows,
                fold_timing_rows=fold_timing_rows,
                overall_start=overall_start,
                raw_data_load_sec=raw_data_load_sec,
            )
            if chain_state.get("chain_max_positions") is not None:
                chain_max_positions = int(chain_state.get("chain_max_positions"))
            if chain_state.get("chain_enable_rotation") is not None:
                chain_enable_rotation = bool(chain_state.get("chain_enable_rotation"))
        rows.sort(key=lambda item: int(item.get("oos_year", 0) or 0))
        fold_timing_rows.sort(key=lambda item: int(item.get("fold_idx", 0) or 0))
        for idx, row in enumerate(rows, start=1):
            row["fold"] = f"{idx}/{len(years)}"

    years_to_run = [] if fold_parallel_enabled else years

    for fold_idx, oos_year in enumerate(years_to_run, start=1):
        fold_start = time.perf_counter()
        selection_end = int(oos_year) - 1
        selection_start = _selection_start_for_oos(config, int(oos_year))
        print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | parallel fold START | selection={selection_start}~{selection_end}", flush=True)
        fold_policy = dict(base_policy)
        fold_policy["selection_start_year"] = int(selection_start)
        fold_policy["train_start_year"] = int(selection_start)
        fold_policy["search_train_end_year"] = int(selection_end)
        fold_policy["oos_start_year"] = int(oos_year)
        fold_policy = build_optimizer_runtime_policy(fold_policy, "split")
        objective_mode = str(fold_policy.get("objective_mode", "split_train_romd"))
        session = build_optimizer_session(walk_forward_policy=fold_policy)
        attach_shared_executor = getattr(session, "attach_shared_trial_prep_executor_holder", None)
        if callable(attach_shared_executor):
            attach_shared_executor(rolling_shared_prep_executor_holder)
        if rolling_shared_prep_cache_max_items > 0:
            attach_shared_cache = getattr(session, "attach_shared_prepared_trial_input_cache", None)
            if callable(attach_shared_cache):
                attach_shared_cache(rolling_shared_prep_cache, max_items=rolling_shared_prep_cache_max_items)
        attach_order_score_cache = getattr(session, "attach_shared_local_min_order_score_cache", None)
        if callable(attach_order_score_cache):
            attach_order_score_cache(rolling_shared_local_min_order_score_cache)
        attach_field_order_score_cache = getattr(session, "attach_shared_local_min_field_order_score_cache", None)
        if callable(attach_field_order_score_cache):
            attach_field_order_score_cache(rolling_shared_local_min_field_order_score_cache)
        reset_prep_cache_stats = getattr(session, "reset_prep_cache_stats", None)
        if callable(reset_prep_cache_stats):
            reset_prep_cache_stats()
        chain_max_positions = int(session.train_max_positions)
        chain_enable_rotation = bool(session.train_enable_rotation)
        session.n_trials = int(config.trials_per_fold)
        session.disable_milestone_dashboard = True
        session.timing_mode = bool(timing_mode)
        sqlite_storage_enabled = _is_outer_rolling_sqlite_storage_enabled(os.environ)
        db_file = ""
        db_name = None
        if sqlite_storage_enabled:
            db_dir = os.path.join(output_dir, "outer_rolling_oos", "db")
            os.makedirs(db_dir, exist_ok=True)
            db_file = os.path.join(db_dir, f"outer_oos_{session_ts}_{int(oos_year)}.db")
            db_name = f"sqlite:///{db_file}"
        study = None
        try:
            print(f"\n{C_CYAN}[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year}{C_RESET}")
            install_started = time.perf_counter()
            session.install_raw_data_cache(
                selected_data_dir,
                shared_raw_context["raw_data_cache"],
                static_fast_cache=shared_raw_context["static_fast_cache"],
                master_dates=shared_raw_context["master_dates"],
                sorted_master_dates=shared_raw_context["sorted_master_dates"],
            )
            install_shared_cache_sec = max(0.0, time.perf_counter() - install_started)
            session.profile_recorder.init_output_files()
            session.profile_recorder.mark_run_started()
            study_started = time.perf_counter()
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=sampler_kind)
            ensure_study_effective_policy_compatible(study=study, walk_forward_policy=fold_policy)
            study_create_sec = max(0.0, time.perf_counter() - study_started)
            progress = _SearchProgress(
                fold_idx=fold_idx,
                fold_count=len(years),
                oos_year=oos_year,
                selection_start=selection_start,
                selection_end=selection_end,
                total_trials=config.trials_per_fold,
                completed_results=rows,
                overall_start=overall_start,
            )
            optimize_started = time.perf_counter()
            study.optimize(session.objective, n_trials=int(config.trials_per_fold), n_jobs=1, callbacks=[progress.callback(session)])
            optimize_sec = max(0.0, time.perf_counter() - optimize_started)
            progress.done(int(session.current_session_trial))

            local_started = time.perf_counter()
            session.outer_rolling_local_progress_context = {
                "fold_idx": int(fold_idx),
                "fold_count": int(len(years)),
                "oos_year": int(oos_year),
                "selection_start": int(selection_start),
                "selection_end": int(selection_end),
                "completed_results": rows,
                "overall_start": overall_start,
            }
            finalists = list_local_min_score_finalists(
                study,
                session=session,
                objective_mode=objective_mode,
                include_trial=None,
                show_progress=True,
                include_oos_diagnostics=False,
                single_finalist_fast_path=True,
                selection_pruning=True,
            )
            if hasattr(session, "outer_rolling_local_progress_context"):
                delattr(session, "outer_rolling_local_progress_context")
            local_elapsed = time.perf_counter() - local_started
            prep_cache_stats = session.get_prep_cache_stats() if hasattr(session, "get_prep_cache_stats") else {}
            print(
                f"[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year} | LOCAL_MIN_REVIEW DONE | "
                f"finalists={len(finalists)} | best_local={float(finalists[0].get('local_min_score', 0.0)) if finalists else 0.0:.3f} "
                f"#{int(finalists[0]['trial'].number) + 1 if finalists else 0} | "
                f"prep_cache_hit/miss/evict={int(prep_cache_stats.get('hits', 0))}/{int(prep_cache_stats.get('misses', 0))}/{int(prep_cache_stats.get('evictions', 0))} | "
                f"elapsed={_fmt_duration(local_elapsed)}"
            )
            policy_items = _build_policy_items(finalists, objective_mode=objective_mode)
            if not any(item is not None for item in policy_items.values()):
                fold_elapsed = time.perf_counter() - fold_start
                fold_timing_rows.append(_build_outer_timing_row(
                    fold_idx=fold_idx,
                    fold_count=len(years),
                    oos_year=int(oos_year),
                    selection_start=int(selection_start),
                    selection_end=int(selection_end),
                    status="skipped_no_finalist",
                    session=session,
                    db_file=db_file,
                    install_shared_cache_sec=install_shared_cache_sec,
                    study_create_sec=study_create_sec,
                    optimize_sec=optimize_sec,
                    local_min_review_sec=local_elapsed,
                    oos_diagnostics_sec=0.0,
                    fold_total_sec=fold_elapsed,
                    finalists_count=len(finalists),
                ))
                print(f"{C_YELLOW}[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year} | 無可用 finalist，略過。{C_RESET}")
                continue
            local_rank_map = _build_local_rank_map(finalists)
            retention_rank_map = _build_retention_rank_map(finalists)
            diagnostics_started = time.perf_counter()
            diagnostics = _evaluate_finalist_oos_diagnostics(
                session=session,
                finalists=finalists,
                policy_items=policy_items,
                oos_year=int(oos_year),
            )
            oos_diagnostics_sec = max(0.0, time.perf_counter() - diagnostics_started)
            print(f"[{fold_idx}/{fold_count}] OOS {oos_year} | OOS diagnostics DONE | elapsed={_fmt_duration(oos_diagnostics_sec)}", flush=True)
            fold_elapsed = time.perf_counter() - fold_start
            selection_period = f"{selection_start}~{selection_end}"
            policy_schedules = {
                name: _build_policy_schedule_entry(
                    item=item,
                    policy_name=name,
                    oos_year=int(oos_year),
                    selection_period=selection_period,
                    local_rank_map=local_rank_map,
                    retention_rank_map=retention_rank_map,
                )
                for name, item in policy_items.items()
                if item is not None
            }
            row = {
                "fold": f"{fold_idx}/{len(years)}",
                "oos_year": int(oos_year),
                "selection_period": selection_period,
                "selection_start_year": int(selection_start),
                "selection_end_year": int(selection_end),
                "best_finalist_oos_score": float(diagnostics.get("best_finalist_oos_score", 0.0)),
                "best_finalist_trial": diagnostics.get("best_finalist_trial"),
                "best_finalist_return_pct": float(diagnostics.get("best_finalist_return_pct", 0.0)),
                "best_finalist_mdd_pct": float(diagnostics.get("best_finalist_mdd_pct", 0.0)),
                "best_finalist_trades": int(diagnostics.get("best_finalist_trades", 0) or 0),
                "best_finalist_initial_capital": float(diagnostics.get("best_finalist_initial_capital", 0.0)),
                "best_finalist_equity_curve": list(diagnostics.get("best_finalist_equity_curve") or []),
                "best_finalist_params": dict(diagnostics.get("best_finalist_params") or {}),
                "benchmark_oos_score": float(diagnostics.get("benchmark_oos_score", 0.0)),
                "benchmark_return_pct": float(diagnostics.get("benchmark_return_pct", 0.0)),
                "benchmark_mdd_pct": float(diagnostics.get("benchmark_mdd_pct", 0.0)),
                "base": diagnostics.get("policies", {}).get("base", {}),
                "local": diagnostics.get("policies", {}).get("local", {}),
                "retention": diagnostics.get("policies", {}).get("retention", {}),
                "policy_schedules": policy_schedules,
                "elapsed_sec": float(fold_elapsed),
                "optimizer_search_sec": float(optimize_sec),
                "local_min_review_sec": float(local_elapsed),
                "oos_diagnostics_sec": float(oos_diagnostics_sec),
                "install_shared_cache_sec": float(install_shared_cache_sec),
                "study_create_sec": float(study_create_sec),
            }
            rows.append(row)
            fold_timing_rows.append(_build_outer_timing_row(
                fold_idx=fold_idx,
                fold_count=len(years),
                oos_year=int(oos_year),
                selection_start=int(selection_start),
                selection_end=int(selection_end),
                status="done",
                session=session,
                db_file=db_file,
                install_shared_cache_sec=install_shared_cache_sec,
                study_create_sec=study_create_sec,
                optimize_sec=optimize_sec,
                local_min_review_sec=local_elapsed,
                oos_diagnostics_sec=oos_diagnostics_sec,
                fold_total_sec=fold_elapsed,
                finalists_count=len(finalists),
            ))
            local_oos = float((row.get("local") or {}).get("rank_1_oos", 0.0))
            print(
                f"{C_GREEN}[{fold_idx}/{len(years)}] selection={selection_start}~{selection_end} | OOS {oos_year} | DONE | "
                f"local_rank_1_oos={local_oos:.{OOS_SCORE_DECIMALS}f} | best={_format_compare_plain(row['best_finalist_oos_score'], local_oos)} | "
                f"0050={_format_compare_plain(row['benchmark_oos_score'], local_oos)} | elapsed={_fmt_duration(fold_elapsed)}{C_RESET}"
            )
            _print_completed_results(rows)
        finally:
            flush_field_order_cache = getattr(session, "flush_shared_local_min_field_order_score_cache", None)
            if callable(flush_field_order_cache):
                flush_field_order_cache()
            session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    _shutdown_rolling_shared_prep_executor_holder(rolling_shared_prep_executor_holder)

    resolved_chain_max_positions = int(chain_max_positions if chain_max_positions is not None else 10)
    resolved_chain_enable_rotation = bool(chain_enable_rotation if chain_enable_rotation is not None else False)
    active_replay_started = time.perf_counter()
    active_replay_chained = _build_active_replay_chained_oos_summary(
        rows=rows,
        config=config,
        selected_data_dir=selected_data_dir,
        output_dir=output_dir,
        max_positions=resolved_chain_max_positions,
        enable_rotation=resolved_chain_enable_rotation,
        overall_start=overall_start,
    ) if rows else {}
    active_replay_chain_sec = max(0.0, time.perf_counter() - active_replay_started)
    final_report_chain_elapsed_sec = max(0.0, time.perf_counter() - overall_start)
    active_replay_chained_for_report = _with_chain_elapsed_override(
        active_replay_chained,
        elapsed_sec=final_report_chain_elapsed_sec,
    )
    report_write_started = time.perf_counter()
    paths = {}
    if not bool(timing_mode):
        paths = _write_reports(project_root=project_root, output_dir=output_dir, session_ts=session_ts, rows=rows, config=config, chained_override=active_replay_chained_for_report)
    report_write_sec = max(0.0, time.perf_counter() - report_write_started)
    resource_sampler.stop()
    timing_paths = _write_outer_timing_summary(
        output_dir=output_dir,
        session_ts=session_ts,
        dataset_label=dataset_label,
        config=config,
        timing_mode=bool(timing_mode),
        optimizer_seed=optimizer_seed,
        raw_data_load_sec=raw_data_load_sec,
        active_replay_chain_sec=active_replay_chain_sec,
        report_write_sec=report_write_sec,
        overall_sec=max(0.0, time.perf_counter() - overall_start),
        fold_timing_rows=fold_timing_rows,
        resource_summary=resource_sampler.summary(),
        resource_samples=resource_sampler.samples,
    )
    print(f"\n{C_CYAN}FINAL REPORT{C_RESET}")
    print(_format_final_report(rows, _build_summary(rows, config=config, chained_override=active_replay_chained_for_report), color=True))
    if bool(timing_mode):
        print(_format_timing_phase_line(timing_paths.get("payload") or {}))
        print(f"{C_GREEN}已輸出：{timing_paths['json']}{C_RESET}")
        if timing_paths.get("csv"):
            print(f"{C_GREEN}已輸出：{timing_paths['csv']}{C_RESET}")
        if timing_paths.get("resource_csv"):
            print(f"{C_GREEN}已輸出：{timing_paths['resource_csv']}{C_RESET}")
    else:
        print(f"{C_GREEN}已輸出：{paths['txt']}{C_RESET}")
        print(f"{C_GREEN}已輸出：{paths['json']}{C_RESET}")
        print(f"{C_GREEN}已輸出：{paths['csv']}{C_RESET}")
        print(f"{C_GREEN}已輸出：{timing_paths['json']}{C_RESET}")
        if timing_paths.get("csv"):
            print(f"{C_GREEN}已輸出：{timing_paths['csv']}{C_RESET}")
        if timing_paths.get("resource_csv"):
            print(f"{C_GREEN}已輸出：{timing_paths['resource_csv']}{C_RESET}")
        for policy_name, paramset_path in dict(paths.get("paramsets") or {}).items():
            print(f"{C_GREEN}已輸出 rolling {policy_name} 年度參數組：{paramset_path}{C_RESET}")
    _print_outer_timing_summary(timing_paths.get("payload", {}))
    print(_format_resource_usage_line(dict((timing_paths.get("payload") or {}).get("summary") or {})))
    return 0
