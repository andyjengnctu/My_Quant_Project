from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import enable_line_buffered_stdout
from tools.local_regression.common import read_json
from tools.local_regression.run_all import execute_all

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
STEP_LABELS = {
    "quick_gate": "quick gate",
    "chain_checks": "chain checks",
    "ml_smoke": "ml smoke",
}


class ConsoleProgress:
    def __init__(self) -> None:
        self.spinner_index = 0
        self.last_render_ts = 0.0
        self.last_line_width = 0
        self.major_total = 1

    def _resolve_progress(self, payload: Dict[str, Any], *, default_index: int = 1) -> tuple[int, int]:
        major_total = int(payload.get("major_total", self.major_total or 1))
        major_total = max(major_total, 1)
        self.major_total = major_total
        major_index = int(payload.get("major_index", default_index))
        major_index = min(max(major_index, 1), major_total)
        return major_index, major_total

    def _build_bar(self, major_index: int, major_total: int, *, done: bool) -> str:
        width = 24
        completed_units = major_index if done else max(major_index - 1, 0)
        filled = int(width * completed_units / major_total)
        return "█" * filled + "░" * (width - filled)

    def _render_inline(self, text: str) -> None:
        padded = text
        if self.last_line_width > len(text):
            padded = text + (" " * (self.last_line_width - len(text)))
        print(f"\r{padded}", end="", flush=True)
        self.last_line_width = max(self.last_line_width, len(text))

    def _finish_line(self, text: str) -> None:
        self._render_inline(text)
        print()
        self.last_line_width = 0

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        if event == "dataset_ready":
            major_index, major_total = self._resolve_progress(payload)
            line = (
                f"[{major_index}/{major_total}] "
                f"{self._build_bar(major_index, major_total, done=True)} "
                f"資料就緒 | {payload['dataset_info'].get('csv_count', 0)} 檔 reduced"
            )
            self._finish_line(line)
            return

        if event == "step_start":
            major_index, major_total = self._resolve_progress(payload)
            label = STEP_LABELS.get(payload['name'], payload['name'])
            line = (
                f"[{major_index}/{major_total}] "
                f"{self._build_bar(major_index, major_total, done=False)} "
                f"準備執行 {label}"
            )
            self._render_inline(line)
            self.last_render_ts = 0.0
            return

        if event == "step_progress":
            if now - self.last_render_ts < 0.15:
                return
            self.last_render_ts = now
            major_index, major_total = self._resolve_progress(payload)
            spinner = SPINNER_FRAMES[self.spinner_index % len(SPINNER_FRAMES)]
            self.spinner_index += 1
            label = STEP_LABELS.get(payload['name'], payload['name'])
            line = (
                f"[{major_index}/{major_total}] "
                f"{self._build_bar(major_index, major_total, done=False)} "
                f"{spinner} 執行中 {label} | {payload['elapsed_sec']:.1f}s"
            )
            self._render_inline(line)
            return

        if event == "step_finish":
            major_index, major_total = self._resolve_progress(payload)
            symbol = "✓" if payload['status'] == "PASS" else "✗"
            label = STEP_LABELS.get(payload['name'], payload['name'])
            line = (
                f"[{major_index}/{major_total}] "
                f"{self._build_bar(major_index, major_total, done=True)} "
                f"{symbol} {label} | {payload['status']} | {payload['duration_sec']:.2f}s"
            )
            self._finish_line(line)
            return

        if event == "finalizing":
            major_index, major_total = self._resolve_progress(payload)
            line = (
                f"[{major_index}/{major_total}] "
                f"{self._build_bar(major_index, major_total, done=False)} "
                "整理輸出與打包 bundle"
            )
            self._render_inline(line)
            return

        if event == "done":
            major_index, major_total = self._resolve_progress(payload, default_index=self.major_total)
            symbol = "✓" if payload['overall_status'] == "PASS" else "✗"
            line = (
                f"[{major_index}/{major_total}] "
                f"{self._build_bar(major_index, major_total, done=True)} "
                f"{symbol} 完成 | {payload['overall_status']}"
            )
            self._finish_line(line)


def _read_summary_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _print_human_summary(result: Dict[str, Any]) -> None:
    latest_dir = Path(result["latest_dir"])
    master = _read_summary_if_exists(latest_dir / "master_summary.json")
    quick = _read_summary_if_exists(latest_dir / "quick_gate_summary.json")
    chain = _read_summary_if_exists(latest_dir / "chain_summary.json")
    ml = _read_summary_if_exists(latest_dir / "ml_smoke_summary.json")

    print("\n=== Local Regression 結果整理 ===")
    print(f"整體狀態: {result['overall_status']} | 失敗數: {result['failures']}")
    print(f"latest 目錄: {result['latest_dir']}")
    print(f"根目錄 bundle: {result['root_bundle_copy']}")

    script_map = {item["name"]: item for item in master.get("scripts", [])}
    if script_map:
        print("\n[步驟摘要]")
        for key in ("quick_gate", "chain_checks", "ml_smoke"):
            item = script_map.get(key, {})
            print(f"- {key}: {item.get('status', 'N/A')} | {item.get('duration_sec', 0.0)}s")

    print("\n[重點結果]")
    print(f"- quick gate: step_count={quick.get('step_count', 0)} | failed_count={quick.get('failed_count', 0)}")

    portfolio_snapshot = chain.get("portfolio_snapshot", {})
    selected_tickers = chain.get("selected_tickers", [])
    print(
        "- chain checks: "
        f"tickers={','.join(selected_tickers)} | "
        f"trade_rows={portfolio_snapshot.get('trade_rows', 0)} | "
        f"reserved_buy_fill_rate={portfolio_snapshot.get('reserved_buy_fill_rate', 0.0)}"
    )
    print(
        "- ml smoke: "
        f"db_trial_count={ml.get('db_trial_count', 0)} | "
        f"status={ml.get('status', 'N/A')}"
    )

    if result["overall_status"] != "PASS":
        failed_names = [item["name"] for item in master.get("scripts", []) if item.get("status") != "PASS"]
        if failed_names:
            print(f"\n失敗步驟: {', '.join(failed_names)}")


def main() -> int:
    enable_line_buffered_stdout()
    print("=== Local Regression (reduced) ===")
    progress = ConsoleProgress()
    result = execute_all(progress_callback=progress)
    _print_human_summary(result)
    return 0 if result["overall_status"] == "PASS" else 1


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
