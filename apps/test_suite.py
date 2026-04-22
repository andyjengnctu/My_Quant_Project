from __future__ import annotations

import ctypes
import os
import sys
import time
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args
from core.test_suite_reporting import TEST_SUITE_STEP_LABELS, print_test_suite_human_summary
from tools.local_regression.formal_pipeline import DATASET_REQUIRED_STEPS, FORMAL_STEP_ORDER

REGRESSION_STEP_ORDER = FORMAL_STEP_ORDER
STEP_LABELS = TEST_SUITE_STEP_LABELS
EXTRA_STEP_LABELS = {"done": "完成"}
DISPLAY_STEP_ORDER = [
    "preflight",
    "dataset_prepare",
    "ml_smoke",
    "quick_gate",
    "chain_checks",
    "consistency",
    "meta_quality",
    "done",
]
BAR_WIDTH = 28
STEP_EXPECTED_SECONDS = {
    "preflight": 1.0,
    "dataset_prepare": 1.0,
    "quick_gate": 7.0,
    "consistency": 36.0,
    "chain_checks": 13.0,
    "ml_smoke": 5.0,
    "meta_quality": 2.0,
    "done": 1.0,
}
PENDING_STATUS = "等待中"
RUNNING_STATUS = "執行中"
FINALIZING_STATUS = "整理中"

# consistency step 的正式 coverage 以 synthetic registry、`--help`、文件與 `doc/TEST_SUITE_CHECKLIST.md` 為準；註解僅供閱讀，不作可機械比對的 formal contract。


class _COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class _SMALL_RECT(ctypes.Structure):
    _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short), ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]


class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", _COORD),
        ("dwCursorPosition", _COORD),
        ("wAttributes", ctypes.c_ushort),
        ("srWindow", _SMALL_RECT),
        ("dwMaximumWindowSize", _COORD),
    ]


def _char_display_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        if ch not in {"\n", "\r"}:
            width += _char_display_width(ch)
    return width


def _truncate_display_width(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    result = []
    width = 0
    for ch in str(text):
        if ch in {"\n", "\r"}:
            continue
        ch_width = _char_display_width(ch)
        if width + ch_width > max_width:
            break
        result.append(ch)
        width += ch_width
    return "".join(result)


class ConsoleProgress:
    def __init__(self) -> None:
        self.suite_started = time.time()
        self.rendered_once = False
        self.finalized = False
        self.last_render_ts = 0.0
        self.step_order = list(DISPLAY_STEP_ORDER)
        self.display_index_map = {name: index for index, name in enumerate(self.step_order, start=1)}
        self.major_total = len(self.step_order)
        self.win32_handle = self._get_windows_console_handle()
        self.use_win32_redraw = self.win32_handle is not None
        self.use_ansi_redraw = (not self.use_win32_redraw) and self._supports_ansi_redraw()
        self.anchor_row: int | None = None
        self.reserved_line_count = 0
        self.console_width = self._get_console_width()
        self._render_lock = threading.RLock()
        self.step_states: Dict[str, Dict[str, Any]] = {
            name: {
                "name": name,
                "major_index": index,
                "major_total": self.major_total,
                "status": "PENDING",
                "display_status": PENDING_STATUS,
                "elapsed_sec": 0.0,
                "duration_sec": None,
                "execution_mode": "serial",
            }
            for index, name in enumerate(self.step_order, start=1)
        }

    def _supports_ansi_redraw(self) -> bool:
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False
        if os.name != "nt":
            return True
        if self._enable_windows_virtual_terminal():
            return True
        return bool(
            os.environ.get("WT_SESSION")
            or os.environ.get("ANSICON")
            or os.environ.get("TERM")
            or os.environ.get("ConEmuANSI") == "ON"
        )

    def _get_windows_console_handle(self):
        if os.name != "nt":
            return None
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return None
        try:
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            if handle in (0, -1):
                return None
            info = _CONSOLE_SCREEN_BUFFER_INFO()
            if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)) == 0:
                return None
            return handle
        except Exception as exc:
            _ = exc
            return None

    def _get_console_width(self) -> int:
        try:
            return max(int(os.get_terminal_size(sys.stdout.fileno()).columns), 40)
        except Exception as exc:
            _ = exc
            return 120

    def _refresh_console_metrics(self) -> None:
        self.console_width = self._get_console_width()

    def _get_cursor_row(self) -> int | None:
        if self.win32_handle is None:
            return None
        try:
            info = _CONSOLE_SCREEN_BUFFER_INFO()
            if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(self.win32_handle, ctypes.byref(info)) == 0:
                return None
            return int(info.dwCursorPosition.Y)
        except Exception as exc:
            _ = exc
            return None

    def _move_cursor(self, x: int, y: int) -> bool:
        if self.win32_handle is None:
            return False
        try:
            coord = _COORD(int(x), int(y))
            return bool(ctypes.windll.kernel32.SetConsoleCursorPosition(self.win32_handle, coord))
        except Exception as exc:
            _ = exc
            return False

    def _fit_console_line(self, text: str) -> str:
        max_width = max(int(self.console_width) - 1, 20)
        trimmed = _truncate_display_width(text, max_width)
        padding = max_width - _display_width(trimmed)
        return trimmed + (" " * max(padding, 0))

    def _enable_windows_virtual_terminal(self) -> bool:
        if os.name != "nt":
            return False
        try:
            kernel32 = ctypes.windll.kernel32
            std_output_handle = -11
            enable_virtual_terminal_processing = 0x0004
            handle = kernel32.GetStdHandle(std_output_handle)
            if handle in (0, -1):
                return False
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
                return False
            desired_mode = mode.value | enable_virtual_terminal_processing
            if kernel32.SetConsoleMode(handle, desired_mode) == 0:
                return False
            return True
        except Exception as exc:
            _ = exc
            return False

    def _format_label(self, state: Dict[str, Any]) -> str:
        label = STEP_LABELS.get(state["name"], EXTRA_STEP_LABELS.get(state["name"], state["name"]))
        if str(state.get("execution_mode", "")).strip() == "parallel" and state["name"] not in {"preflight", "dataset_prepare", "meta_quality", "done"}:
            return f"{label}（並行）"
        return label

    def _build_bar(self, state: Dict[str, Any]) -> str:
        status = state["status"]
        if status == "PASS":
            return "█" * BAR_WIDTH
        if status == "FAIL":
            return "█" * BAR_WIDTH
        if status in {"RUNNING", "FINALIZING"}:
            expected = max(float(STEP_EXPECTED_SECONDS.get(state["name"], 10.0)), 0.5)
            elapsed = max(float(state.get("elapsed_sec") or 0.0), 0.0)
            ratio = min(elapsed / expected, 0.97)
            filled = max(1, min(BAR_WIDTH - 1, int(round(BAR_WIDTH * ratio))))
            return "█" * filled + "░" * (BAR_WIDTH - filled)
        return "░" * BAR_WIDTH

    def _format_line(self, state: Dict[str, Any]) -> str:
        index = int(state["major_index"])
        total = int(state["major_total"])
        label = self._format_label(state)
        bar = self._build_bar(state)
        status = state["display_status"]
        if state["status"] in {"RUNNING", "FINALIZING"}:
            detail = f"{status} | {float(state.get('elapsed_sec') or 0.0):.1f}s"
        elif state["status"] in {"PASS", "FAIL"}:
            detail = f"{status} | {float(state.get('duration_sec') or 0.0):.2f}s"
        else:
            detail = status

        line = f"[{index}/{total}] {bar} {label} | {detail}"
        if state["name"] == "done":
            elapsed_total = time.time() - self.suite_started
            line += f" | 總經過 {elapsed_total:.1f}s"
        return line

    def _render_block(self, *, force: bool = False) -> None:
        if self.finalized:
            return
        now = time.time()
        if not force and (now - self.last_render_ts) < 0.1:
            return
        self.last_render_ts = now
        lines = [self._format_line(self.step_states[name]) for name in self.step_order]
        if self.use_win32_redraw:
            self._refresh_console_metrics()
            if self.anchor_row is None:
                self.anchor_row = self._get_cursor_row()
            if self.anchor_row is not None:
                if self.reserved_line_count < len(lines):
                    current_row = self._get_cursor_row()
                    if current_row is not None and self._move_cursor(0, current_row):
                        sys.stdout.write("\n" * len(lines))
                        sys.stdout.flush()
                        self.reserved_line_count = len(lines)
                    self._move_cursor(0, self.anchor_row)
                prepared = [self._fit_console_line(line) for line in lines]
                rendered = False
                for offset, line in enumerate(prepared):
                    if not self._move_cursor(0, self.anchor_row + offset):
                        rendered = False
                        break
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    rendered = True
                if rendered:
                    self._move_cursor(0, self.anchor_row + len(lines))
                    self.rendered_once = True
                    return
        if self.use_ansi_redraw and self.rendered_once:
            sys.stdout.write(f"\x1b[{len(lines)}F")
            for index, line in enumerate(lines):
                sys.stdout.write("\x1b[2K")
                sys.stdout.write(line)
                if index < len(lines) - 1:
                    sys.stdout.write("\n")
            sys.stdout.write("\n")
        else:
            block = "\n".join(lines)
            sys.stdout.write(block + "\n")
        sys.stdout.flush()
        self.rendered_once = True

    def _ensure_step_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload["name"])
        state = self.step_states.setdefault(
            name,
            {
                "name": name,
                "major_index": int(payload.get("major_index", 0) or 0),
                "major_total": int(payload.get("major_total", self.major_total) or self.major_total),
                "status": "PENDING",
                "display_status": PENDING_STATUS,
                "elapsed_sec": 0.0,
                "duration_sec": None,
                "execution_mode": str(payload.get("execution_mode", "serial") or "serial"),
            },
        )
        state["major_index"] = int(self.display_index_map.get(name, state.get("major_index", 0)) or 0)
        state["major_total"] = self.major_total
        if "execution_mode" in payload:
            state["execution_mode"] = str(payload.get("execution_mode") or "serial")
        return state

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        with self._render_lock:
            if event == "step_start":
                state = self._ensure_step_state(payload)
                state["status"] = "RUNNING"
                state["display_status"] = RUNNING_STATUS
                state["elapsed_sec"] = 0.0
                state["duration_sec"] = None
                self._render_block(force=True)
                return

            if event == "step_progress":
                state = self._ensure_step_state(payload)
                state["status"] = "RUNNING"
                state["display_status"] = RUNNING_STATUS
                state["elapsed_sec"] = float(payload.get("elapsed_sec", state.get("elapsed_sec", 0.0)) or 0.0)
                self._render_block(force=False)
                return

            if event == "step_finish":
                state = self._ensure_step_state(payload)
                state["status"] = str(payload.get("status", "FAIL") or "FAIL")
                state["display_status"] = state["status"]
                state["duration_sec"] = float(payload.get("duration_sec", 0.0) or 0.0)
                state["elapsed_sec"] = state["duration_sec"]
                self._render_block(force=True)
                return

            if event == "finalizing":
                state = self._ensure_step_state({"name": "done", **payload})
                state["status"] = "RUNNING"
                state["display_status"] = FINALIZING_STATUS
                state["elapsed_sec"] = float(payload.get("elapsed_sec", state.get("elapsed_sec", 0.0)) or 0.0)
                self._render_block(force=True)
                return

            if event == "done":
                state = self._ensure_step_state({"name": "done", **payload})
                state["status"] = str(payload.get("overall_status", "FAIL") or "FAIL")
                state["display_status"] = state["status"]
                state["duration_sec"] = float(payload.get("elapsed_sec", payload.get("duration_sec", 0.0)) or 0.0)
                state["elapsed_sec"] = state["duration_sec"]
                self._render_block(force=True)
                self.finalized = True


def _print_human_summary(result: Dict[str, Any]) -> None:
    print_test_suite_human_summary(
        result,
        regression_step_order=REGRESSION_STEP_ORDER,
        dataset_required_steps=DATASET_REQUIRED_STEPS,
        step_labels=STEP_LABELS,
    )


def main(argv=None) -> int:
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/test_suite.py")
        print(f"用法: python {program_name}")
        print("說明: reduced 一鍵測試正式入口；會串接 formal pipeline、meta quality、artifact integrity，以及交易口徑一致性、保守出場解讀、debug-backtest 現金路徑等穩定主題；若失敗再依主控台建議用 run_all.py --only 重跑失敗步驟。")
        return 0

    from tools.local_regression.run_all import execute_all

    print("=== Test Suite (reduced) ===")
    print("[前置檢查] 先檢查目前 Python 環境是否已具備 requirements；不自動安裝。")
    progress = ConsoleProgress()
    result = execute_all(progress_callback=progress)
    _print_human_summary(result)
    return 0 if result["overall_status"] == "PASS" else 1


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)
