from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args
from core.test_suite_reporting import TEST_SUITE_STEP_LABELS, print_test_suite_human_summary
from tools.local_regression.formal_pipeline import DATASET_REQUIRED_STEPS, FORMAL_STEP_ORDER

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
REGRESSION_STEP_ORDER = FORMAL_STEP_ORDER

STEP_LABELS = TEST_SUITE_STEP_LABELS

# consistency step 的正式 coverage 以 synthetic registry、`--help`、文件與 `doc/TEST_SUITE_CHECKLIST.md` 為準；註解僅供閱讀，不作可機械比對的 formal contract。

class ConsoleProgress:
    _STEP_INDEX_TO_NAME = {
        1: "preflight",
        2: "dataset_prepare",
        3: "quick_gate",
        4: "consistency",
        5: "chain_checks",
        6: "ml_smoke",
        7: "meta_quality",
        8: "done",
    }
    _BAR_WIDTH = 28
    _RUN_SEGMENT_WIDTH = 6

    def __init__(self) -> None:
        self.suite_started = time.time()
        self._lock = threading.RLock()
        self._render_event = threading.Event()
        self._stop_event = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._last_screen_line_count = 0
        self._supports_live_redraw = self._detect_live_redraw_support()
        self._step_states: Dict[str, Dict[str, Any]] = {}
        for index in range(1, 9):
            name = self._STEP_INDEX_TO_NAME[index]
            self._step_states[name] = {
                "major_index": index,
                "name": name,
                "label": "完成" if name == "done" else STEP_LABELS.get(name, name),
                "execution_mode": "serial",
                "state": "pending",
                "started_at": None,
                "duration_sec": None,
                "status": None,
                "timeout_sec": None,
            }

    def _detect_live_redraw_support(self) -> bool:
        if sys.stdout.isatty():
            self._try_enable_windows_vt_mode()
            return True
        env = os.environ
        indicators = (
            env.get("WT_SESSION"),
            env.get("ANSICON"),
            env.get("TERM"),
            env.get("ConEmuANSI") == "ON",
            env.get("PYCHARM_HOSTED"),
            env.get("VSCODE_PID"),
        )
        if any(indicators):
            self._try_enable_windows_vt_mode()
            return True
        return False

    @staticmethod
    def _try_enable_windows_vt_mode() -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            if handle in (0, -1):
                return
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
                return
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            return

    def _ensure_render_thread(self) -> None:
        if self._render_thread is not None:
            return
        self._render_thread = threading.Thread(target=self._render_loop, name="test-suite-progress", daemon=True)
        self._render_thread.start()

    def _format_label(self, payload: Dict[str, Any]) -> str:
        raw_name = str(payload.get("name", "") or "").strip()
        label = STEP_LABELS.get(raw_name, raw_name)
        if raw_name == "done":
            label = str(payload.get("label", "") or "").strip() or "完成"
        if str(payload.get("execution_mode", "") or "").strip() == "parallel":
            return f"{label}（並行）"
        return label

    @staticmethod
    def _status_symbol(state: str, status: str | None) -> str:
        if state == "done":
            return "✓" if status == "PASS" else "✗"
        if state == "running":
            return "→"
        return "·"

    def _build_bar(self, step: Dict[str, Any], now: float) -> str:
        state = str(step.get("state", "pending"))
        if state == "done":
            return "█" * self._BAR_WIDTH
        if state != "running":
            return "░" * self._BAR_WIDTH
        started_at = float(step.get("started_at") or now)
        elapsed = max(0.0, now - started_at)
        head = int(elapsed * 8.0) % self._BAR_WIDTH
        chars = ["░"] * self._BAR_WIDTH
        for offset in range(self._RUN_SEGMENT_WIDTH):
            idx = (head + offset) % self._BAR_WIDTH
            chars[idx] = "█" if offset >= self._RUN_SEGMENT_WIDTH - 2 else "▓"
        return "".join(chars)

    def _step_detail_text(self, step: Dict[str, Any], now: float) -> str:
        state = str(step.get("state", "pending"))
        status = str(step.get("status") or "") or None
        duration_sec = step.get("duration_sec")
        started_at = step.get("started_at")
        if state == "done":
            if duration_sec is None and started_at is not None:
                duration_sec = max(0.0, now - float(started_at))
            precision = ".1f" if step.get("name") == "done" else ".2f"
            return f"{status or 'PASS'} | {format(float(duration_sec or 0.0), precision)}s"
        if state == "running":
            elapsed_sec = max(0.0, now - float(started_at or now))
            return f"執行中 | {elapsed_sec:.1f}s"
        return "等待"

    def _build_lines(self) -> list[str]:
        now = time.time()
        lines: list[str] = []
        for index in range(1, 9):
            step = self._step_states[self._STEP_INDEX_TO_NAME[index]]
            symbol = self._status_symbol(str(step.get("state", "pending")), step.get("status"))
            bar = self._build_bar(step, now)
            label = str(step.get("label", step.get("name", "")))
            detail = self._step_detail_text(step, now)
            lines.append(f"[{index}/8] {bar} {symbol} {label:<20} {detail}")
        return lines

    def _render_lines(self, lines: list[str]) -> None:
        if self._last_screen_line_count > 0:
            sys.stdout.write(f"\x1b[{self._last_screen_line_count}F")
        total_lines = max(self._last_screen_line_count, len(lines))
        for idx in range(total_lines):
            sys.stdout.write("\x1b[2K")
            if idx < len(lines):
                sys.stdout.write(lines[idx])
            if idx < total_lines - 1:
                sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_screen_line_count = len(lines)

    def _render_once(self) -> None:
        with self._lock:
            if not self._supports_live_redraw:
                return
            self._render_lines(self._build_lines())

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            self._render_event.wait(timeout=0.12)
            self._render_event.clear()
            self._render_once()

    def _signal_render(self) -> None:
        if not self._supports_live_redraw:
            return
        self._ensure_render_thread()
        self._render_event.set()

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            now = time.time()
            if event == "step_start":
                name = str(payload.get("name", "") or "").strip()
                step = self._step_states.get(name)
                if step is not None:
                    step["label"] = self._format_label(payload)
                    step["execution_mode"] = str(payload.get("execution_mode", "serial") or "serial")
                    step["state"] = "running"
                    step["started_at"] = now
                    step["duration_sec"] = None
                    step["status"] = None
                    step["timeout_sec"] = payload.get("timeout_sec")
                self._signal_render()
                return

            if event == "step_progress":
                name = str(payload.get("name", "") or "").strip()
                step = self._step_states.get(name)
                if step is not None and step.get("state") == "running":
                    elapsed_hint = payload.get("elapsed_sec")
                    if isinstance(elapsed_hint, (int, float)) and step.get("started_at") is not None:
                        step["duration_sec"] = float(elapsed_hint)
                self._signal_render()
                return

            if event == "step_finish":
                name = str(payload.get("name", "") or "").strip()
                step = self._step_states.get(name)
                if step is not None:
                    step["label"] = self._format_label(payload)
                    step["state"] = "done"
                    step["status"] = str(payload.get("status", "FAIL") or "FAIL")
                    step["duration_sec"] = float(payload.get("duration_sec", 0.0) or 0.0)
                    if step.get("started_at") is None:
                        step["started_at"] = now - float(step["duration_sec"])
                self._signal_render()
                return

            if event == "finalizing":
                step = self._step_states["done"]
                step["label"] = str(payload.get("label", "") or "整理輸出與打包 bundle")
                step["state"] = "running"
                step["started_at"] = now
                step["duration_sec"] = None
                step["status"] = None
                self._signal_render()
                return

            if event == "done":
                step = self._step_states["done"]
                step["label"] = "完成"
                step["state"] = "done"
                step["status"] = str(payload.get("overall_status", "FAIL") or "FAIL")
                step["duration_sec"] = max(0.0, now - self.suite_started)
                if step.get("started_at") is None:
                    step["started_at"] = self.suite_started
                self._signal_render()
                return

    def close(self) -> None:
        if self._supports_live_redraw:
            self._render_once()
        self._stop_event.set()
        self._render_event.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=1.0)
        with self._lock:
            if self._supports_live_redraw and self._last_screen_line_count > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._last_screen_line_count = 0


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
    try:
        result = execute_all(progress_callback=progress)
    finally:
        progress.close()
    _print_human_summary(result)
    return 0 if result["overall_status"] == "PASS" else 1


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)
