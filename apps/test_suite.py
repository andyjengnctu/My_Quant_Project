from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
STEP_LABELS = {
    "quick_gate": "quick gate",
    "consistency": "consistency",
    "chain_checks": "chain checks",
    "ml_smoke": "ml smoke",
    "meta_quality": "meta quality",
    "preflight": "preflight",
    "dataset_prepare": "dataset prepare",
    "manifest": "manifest",
}


class ConsoleProgress:
    def __init__(self) -> None:
        self.spinner_index = 0
        self.last_render_ts = 0.0
        self.last_line_width = 0
        self.suite_started = time.time()

    def _build_bar(self, major_index: int, major_total: int, *, done: bool) -> str:
        width = 28
        completed_units = major_index if done else max(major_index - 1, 0)
        filled = int(width * completed_units / max(major_total, 1))
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

    def _step_text(self, payload: Dict[str, Any], body: str, *, done: bool, progress_index: int | None = None) -> str:
        major_index = int(payload.get("major_index", 0) or 0)
        major_total = int(payload.get("major_total", 1) or 1)
        display_index = major_index if progress_index is None else int(progress_index)
        elapsed_total = time.time() - self.suite_started
        return (
            f"[{display_index}/{major_total}] "
            f"{self._build_bar(display_index, major_total, done=done)} "
            f"{body} | 累計 {elapsed_total:.1f}s"
        )

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        if event == "step_start":
            label = STEP_LABELS.get(payload["name"], payload["name"])
            self._render_inline(self._step_text(payload, f"準備執行 {label}", done=False))
            self.last_render_ts = 0.0
            return

        if event == "step_progress":
            if now - self.last_render_ts < 0.15:
                return
            self.last_render_ts = now
            spinner = SPINNER_FRAMES[self.spinner_index % len(SPINNER_FRAMES)]
            self.spinner_index += 1
            label = STEP_LABELS.get(payload["name"], payload["name"])
            self._render_inline(
                self._step_text(
                    payload,
                    f"{spinner} 執行中 {label} | {payload['elapsed_sec']:.1f}s",
                    done=False,
                )
            )
            return

        if event == "step_finish":
            symbol = "✓" if payload["status"] == "PASS" else "✗"
            label = STEP_LABELS.get(payload["name"], payload["name"])
            self._finish_line(
                self._step_text(
                    payload,
                    f"{symbol} {label} | {payload['status']} | {payload['duration_sec']:.2f}s",
                    done=True,
                )
            )
            return

        if event == "finalizing":
            self._render_inline(self._step_text(payload, "整理輸出與打包 bundle", done=False))
            return

        if event == "done":
            if payload["overall_status"] == "PASS":
                self._finish_line(self._step_text(payload, f"✓ 完成 | {payload['overall_status']}", done=True))
            else:
                failed_at = int(payload.get("failed_at_major_index", payload.get("major_index", 0)) or 0)
                self._finish_line(
                    self._step_text(
                        payload,
                        f"✗ 結束 | {payload['overall_status']}",
                        done=False,
                        progress_index=failed_at,
                    )
                )

def _print_human_summary(result: Dict[str, Any]) -> None:
    master = {
        "scripts": result.get("scripts", []),
        "failures": result.get("failures", 0),
        "overall_status": result.get("overall_status", "FAIL"),
    }
    selected_steps = set(result.get("selected_steps", []))
    failed_step_names = set(result.get("failed_step_names", []))
    not_run_step_names = set(result.get("not_run_step_names", []))
    step_payloads = result.get("step_payloads", {})
    quick = step_payloads.get("quick_gate", {})
    consistency = step_payloads.get("consistency", {})
    chain = step_payloads.get("chain_checks", {})
    ml = step_payloads.get("ml_smoke", {})
    meta = step_payloads.get("meta_quality", {})

    def _payload_issue_text(payload: Dict[str, Any]) -> str:
        if not payload:
            return "missing_summary_file"
        reasons = []
        if payload.get("error_type"):
            reasons.append("summary_unreadable")
        reported_status = str(payload.get("status", "") or "").strip()
        if reported_status and reported_status != "PASS":
            reasons.append(f"reported_status={reported_status}")
        return ", ".join(reasons)

    def _payload_detail_text(payload: Dict[str, Any]) -> str:
        if not payload:
            return ""
        details = []
        error_type = str(payload.get("error_type", "") or "").strip()
        if error_type:
            details.append(f"error_type={error_type}")
        error_message = str(payload.get("error_message", "") or payload.get("runtime_error", "") or "").strip()
        if error_message:
            details.append(f"error_message={error_message}")
        failed_packages = [str(item).strip() for item in payload.get("failed_packages", []) if str(item).strip()]
        if failed_packages:
            details.append("failed_packages=" + ",".join(failed_packages))
        failed_steps = [str(item).strip() for item in payload.get("failed_steps", []) if str(item).strip()]
        if failed_steps:
            details.append("failed_steps=" + ",".join(failed_steps))
        failures = [str(item).strip() for item in payload.get("failures", []) if str(item).strip()]
        if failures:
            details.append("failures=" + ",".join(failures))
        summary_write_error = str(payload.get("summary_write_error", "") or "").strip()
        if summary_write_error:
            details.append(f"summary_write_error={summary_write_error}")
        if "fail_count" in payload:
            try:
                details.append(f"fail_count={int(payload.get('fail_count', 0))}")
            except (TypeError, ValueError):
                pass
        if "failed_count" in payload:
            try:
                details.append(f"failed_count={int(payload.get('failed_count', 0))}")
            except (TypeError, ValueError):
                pass
        return ", ".join(details)

    def _blocked_reason(step_name: str) -> str:
        if step_name not in not_run_step_names:
            return ""
        if "manifest" in failed_step_names:
            return "blocked_by_manifest"
        if step_name == "dataset_prepare" and "preflight" in failed_step_names:
            return "blocked_by_preflight"
        if step_name in {"quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"}:
            if "preflight" in failed_step_names:
                return "blocked_by_preflight"
            if "dataset_prepare" in failed_step_names:
                return "blocked_by_dataset_prepare"
        return "blocked_before_step"

    def _step_overview(name: str, payload: Dict[str, Any], script_item: Dict[str, Any]) -> tuple[str, str]:
        if name in {"quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"} and name not in selected_steps:
            return "SKIP", "not_selected"
        if name in not_run_step_names:
            return "NOT_RUN", _blocked_reason(name)
        issue_text = _payload_issue_text(payload)
        if issue_text:
            payload_detail = _payload_detail_text(payload)
            detail_text = ", ".join(part for part in [issue_text, payload_detail] if part)
            return script_item.get("status", payload.get("status", "FAIL") or "FAIL"), detail_text
        if script_item:
            return script_item.get("status", "PASS"), ""
        reported_status = str(payload.get("status", "") or "").strip()
        if reported_status:
            return reported_status, ""
        return "FAIL", "missing_summary_file"

    print("\n" + "=" * 78)
    print(" Test Suite 結果整理")
    print("=" * 78)
    print(f"整體狀態 : {result['overall_status']} | 失敗步驟 : {result['failures']}")
    manifest_payload = step_payloads.get("manifest", {})
    if manifest_payload:
        print(
            f"manifest    : FAIL | {manifest_payload.get('error_type', '')}: "
            f"{manifest_payload.get('error_message', '')}"
        )

    preflight = result.get("preflight", {})
    preflight_status, preflight_detail = _step_overview("preflight", preflight, {})
    if preflight_status == "PASS":
        failed_packages = preflight.get("failed_packages", [])
        failed_text = ", ".join(failed_packages) if failed_packages else "(none)"
        print(
            f"preflight   : {preflight_status} | "
            f"{preflight.get('duration_sec', 0.0):.2f}s | failed_packages={failed_text}"
        )
    else:
        print(f"preflight   : {preflight_status} | {preflight_detail}")

    dataset_prepare = step_payloads.get("dataset_prepare", {})
    dataset_status, dataset_detail = _step_overview("dataset_prepare", dataset_prepare, {})
    if dataset_status == "PASS":
        print(
            f"dataset prep: PASS | {dataset_prepare.get('duration_sec', 0.0):.2f}s | "
            f"csv_count={dataset_prepare.get('csv_count', 0)} | "
            f"source={dataset_prepare.get('source', '')}"
        )
    else:
        print(f"dataset prep: {dataset_status} | {dataset_detail}")
    retention = result.get("retention", {})
    print(f"bundle 模式 : {result.get('bundle_mode', 'unknown')}")
    print(f"歷史 bundle : {result.get('archived_bundle', '')}")
    print(f"根目錄 bundle : {result['root_bundle_copy']}")
    print(f"bundle 檔數 : {len(result.get('bundle_entries', []))}")
    print(f"retention  : removed={retention.get('removed_count', 0)} | bytes={retention.get('removed_bytes', 0)}")

    script_map = {item["name"]: item for item in master.get("scripts", [])}
    print("\n[步驟摘要]")
    for key in ("quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"):
        item = script_map.get(key, {})
        status, detail = _step_overview(key, step_payloads.get(key, {}), item)
        if item:
            detail_suffix = f" | {detail}" if detail and status != "PASS" else ""
            print(f"- {STEP_LABELS.get(key, key):<13} {status:<8} {item.get('duration_sec', 0.0):>6.2f}s{detail_suffix}")
        else:
            print(f"- {STEP_LABELS.get(key, key):<13} {status:<8} {detail}")

    print("\n[重點結果]")
    quick_script = script_map.get("quick_gate", {})
    quick_status, quick_detail = _step_overview("quick_gate", quick, quick_script)
    if quick_status == "PASS":
        print(f"- quick gate : step_count={quick.get('step_count', 0)} | failed_count={quick.get('failed_count', 0)}")
    else:
        fallback = ", ".join(quick_script.get("failure_reasons", []))
        print(f"- quick gate : {quick_status} | {quick_detail or fallback}")

    consistency_script = script_map.get("consistency", {})
    consistency_status, consistency_detail = _step_overview("consistency", consistency, consistency_script)
    if consistency_status == "PASS":
        print(
            "- consistency: "
            f"total_checks={consistency.get('total_checks', 0)} | "
            f"fail_count={consistency.get('fail_count', 0)} | "
            f"skip_count={consistency.get('skip_count', 0)} | "
            f"real_tickers={consistency.get('real_ticker_count', 0)}"
        )
    else:
        fallback = ", ".join(consistency_script.get("failure_reasons", []))
        print(f"- consistency: {consistency_status} | {consistency_detail or fallback}")

    chain_script = script_map.get("chain_checks", {})
    chain_status, chain_detail = _step_overview("chain_checks", chain, chain_script)
    if chain_status == "PASS":
        portfolio_snapshot = chain.get("portfolio_snapshot", {})
        highlights = chain.get("highlights", {})
        print(
            "- chain checks: "
            f"ticker_count={chain.get('ticker_count', 0)} | "
            f"traded_ticker_count={highlights.get('traded_ticker_count', 0)} | "
            f"missed_buy_ticker_count={highlights.get('missed_buy_ticker_count', 0)} | "
            f"trade_rows={portfolio_snapshot.get('trade_rows', 0)} | "
            f"reserved_buy_fill_rate={portfolio_snapshot.get('reserved_buy_fill_rate', 0.0)}"
        )
        blocked_by_counts = highlights.get("blocked_by_counts", {})
        if blocked_by_counts:
            blocked_preview = ", ".join(f"{key}:{value}" for key, value in list(blocked_by_counts.items())[:6])
            print(f"  blocked_by : {blocked_preview}")
    else:
        fallback = ", ".join(chain_script.get("failure_reasons", []))
        print(f"- chain checks: {chain_status} | {chain_detail or fallback}")

    ml_script = script_map.get("ml_smoke", {})
    ml_status, ml_detail = _step_overview("ml_smoke", ml, ml_script)
    if ml_status == "PASS":
        print(
            "- ml smoke   : "
            f"db_trial_count={ml.get('db_trial_count', 0)} | "
            f"status={ml.get('status', 'N/A')}"
        )
    else:
        fallback = ", ".join(ml_script.get("failure_reasons", []))
        print(f"- ml smoke   : {ml_status} | {ml_detail or fallback}")

    meta_script = script_map.get("meta_quality", {})
    meta_status, meta_detail = _step_overview("meta_quality", meta, meta_script)
    if meta_status == "PASS":
        coverage = meta.get("coverage", {})
        checklist = meta.get("checklist", {})
        totals = coverage.get("totals", {})
        print(
            "- meta quality: "
            f"fail_count={meta.get('fail_count', 0)} | "
            f"coverage_percent={totals.get('percent_covered', 0.0)} | "
            f"todo_ids={len(checklist.get('todo_ids', []))}"
        )
    else:
        fallback = ", ".join(meta_script.get("failure_reasons", []))
        print(f"- meta quality: {meta_status} | {meta_detail or fallback}")

    if result["overall_status"] != "PASS":
        failed_names = [item["name"] for item in master.get("scripts", []) if item.get("status") != "PASS"]
        if not failed_names:
            failed_names = [str(name) for name in result.get("failed_step_names", []) if str(name).strip()]
        if failed_names:
            print(f"\n失敗步驟 : {', '.join(failed_names)}")
        rerun_command = result.get("suggested_rerun_command", "")
        if rerun_command:
            print(f"建議重跑 : {rerun_command}")
    print("=" * 78)


def main(argv=None) -> int:
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/test_suite.py")
        print(f"用法: python {program_name}")
        print("說明: reduced 一鍵測試正式入口；會串接所有已實作測試（含 meta quality），若失敗再依主控台建議用 run_all.py --only 重跑失敗步驟。")
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
