from __future__ import annotations

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

    def __init__(self) -> None:
        self.suite_started = time.time()
        self._lock = threading.RLock()
        self._render_event = threading.Event()
        self._stop_event = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._last_screen_line_count = 0
        self._snapshot_counter = 0
        self._supports_live_redraw = bool(sys.stdout.isatty())
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

    def _build_overall_bar(self, completed_count: int, total_count: int, running_count: int) -> str:
        width = 28
        fraction = 0.0 if total_count <= 0 else min((completed_count + (0.35 * running_count)) / total_count, 0.99 if completed_count < total_count else 1.0)
        filled = int(round(width * fraction))
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)

    def _status_symbol(self, state: str, status: str | None) -> str:
        if state == "done":
            return "✓" if status == "PASS" else "✗"
        if state == "running":
            return "→"
        return "·"

    def _step_detail_text(self, step: Dict[str, Any], now: float) -> str:
        state = str(step.get("state", "pending"))
        status = str(step.get("status") or "") or None
        duration_sec = step.get("duration_sec")
        started_at = step.get("started_at")
        if state == "done":
            if duration_sec is None and started_at is not None:
                duration_sec = max(0.0, now - float(started_at))
            if step.get("name") == "done":
                return f"{status or 'PASS'} | {float(duration_sec or 0.0):.1f}s"
            return f"{status or 'PASS'} | {float(duration_sec or 0.0):.2f}s"
        if state == "running":
            elapsed_sec = max(0.0, now - float(started_at or now))
            timeout_sec = step.get("timeout_sec")
            if isinstance(timeout_sec, (int, float)) and float(timeout_sec) > 0:
                return f"執行中 {elapsed_sec:.1f}s / timeout {float(timeout_sec):.0f}s"
            return f"執行中 {elapsed_sec:.1f}s"
        return "等待"

    def _build_lines(self) -> list[str]:
        now = time.time()
        completed_steps = 0
        running_steps = 0
        pending_steps = 0
        step_lines: list[str] = []
        for index in range(1, 9):
            step = self._step_states[self._STEP_INDEX_TO_NAME[index]]
            state = str(step.get("state", "pending"))
            if state == "done":
                completed_steps += 1
            elif state == "running":
                running_steps += 1
            else:
                pending_steps += 1
            symbol = self._status_symbol(state, step.get("status"))
            label = str(step.get("label", step.get("name", "")))
            detail = self._step_detail_text(step, now)
            step_lines.append(f"[{index}/8] {symbol} {label:<20} {detail}")

        elapsed_total = now - self.suite_started
        header = (
            f"[總進度] {self._build_overall_bar(completed_steps, 8, running_steps)} "
            f"完成 {completed_steps}/8 | 執行中 {running_steps} | 等待 {pending_steps} | 經過 {elapsed_total:.1f}s"
        )
        running_labels = [
            f"{step['label']} {max(0.0, now - float(step['started_at'] or now)):.1f}s"
            for index in range(1, 9)
            for step in [self._step_states[self._STEP_INDEX_TO_NAME[index]]]
            if str(step.get('state', 'pending')) == 'running'
        ]
        footer = "[目前] " + (" | ".join(running_labels) if running_labels else "無")
        return [header, *step_lines, footer]

    def _render_lines(self, lines: list[str], *, force_snapshot: bool = False) -> None:
        if self._supports_live_redraw:
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
            return

        if not force_snapshot:
            return
        self._snapshot_counter += 1
        print(f"[進度快照 #{self._snapshot_counter}]")
        for line in lines:
            print(line)

    def _render_once(self, *, force_snapshot: bool = False) -> None:
        with self._lock:
            lines = self._build_lines()
            self._render_lines(lines, force_snapshot=force_snapshot)

    def _render_loop(self) -> None:
        fallback_interval = 1.0
        next_snapshot = time.time()
        while not self._stop_event.is_set():
            self._render_event.wait(timeout=0.2)
            self._render_event.clear()
            if self._supports_live_redraw:
                self._render_once()
                continue
            now = time.time()
            if now >= next_snapshot:
                self._render_once(force_snapshot=True)
                next_snapshot = now + fallback_interval

    def _signal_render(self) -> None:
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
