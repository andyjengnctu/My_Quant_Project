from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.local_regression.common import (
    OUTPUT_ROOT,
    build_artifacts_manifest,
    build_bundle_zip,
    build_python_env,
    ensure_dir,
    ensure_reduced_dataset,
    finalize_latest,
    gather_recent_console_tail,
    load_manifest,
    publish_root_bundle_copy,
    resolve_git_commit,
    taipei_now,
    timestamp_text,
    write_json,
    write_text,
)

SCRIPT_ORDER = [
    ("quick_gate", "tools/local_regression/run_quick_gate.py", "quick_gate_summary.json", 900),
    ("consistency", "tools/validate/cli.py", "validate_consistency_summary.json", 900),
    ("chain_checks", "tools/local_regression/run_chain_checks.py", "chain_summary.json", 900),
    ("ml_smoke", "tools/local_regression/run_ml_smoke.py", "ml_smoke_summary.json", 900),
]

MAJOR_STEP_COUNT = 2 + len(SCRIPT_ORDER)
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _emit_progress(progress_callback: Optional[ProgressCallback], event: str, payload: Dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event, payload)


def _run_script(
    *,
    name: str,
    relative_script: str,
    timeout_sec: int,
    env: Dict[str, str],
    log_path: Path,
    progress_callback: Optional[ProgressCallback],
    major_index: int,
) -> Dict[str, Any]:
    started = time.time()
    process = None
    returncode = 1
    timed_out = False

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {sys.executable} {relative_script}\n")
        log_file.flush()
        try:
            process = subprocess.Popen(
                [sys.executable, relative_script],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            while True:
                returncode = process.poll()
                elapsed_sec = round(time.time() - started, 1)
                if returncode is not None:
                    break
                if elapsed_sec >= timeout_sec:
                    timed_out = True
                    process.kill()
                    process.wait(timeout=5)
                    returncode = 124
                    log_file.write(f"\n[local_regression] TIMEOUT after {timeout_sec} seconds\n")
                    log_file.flush()
                    break
                _emit_progress(progress_callback, "step_progress", {
                    "name": name,
                    "major_index": major_index,
                    "major_total": MAJOR_STEP_COUNT,
                    "elapsed_sec": elapsed_sec,
                    "timeout_sec": timeout_sec,
                })
                time.sleep(0.2)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    duration_sec = round(time.time() - started, 3)
    return {
        "returncode": returncode,
        "duration_sec": duration_sec,
        "timed_out": timed_out,
        "log_path": str(log_path),
    }


def execute_all(progress_callback: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    manifest = load_manifest()
    dataset_info = ensure_reduced_dataset()
    run_dir = ensure_dir(OUTPUT_ROOT / "runs" / timestamp_text())
    shared_env = build_python_env({"V16_LOCAL_REGRESSION_RUN_DIR": str(run_dir)})

    _emit_progress(progress_callback, "dataset_ready", {
        "major_index": 1,
        "major_total": MAJOR_STEP_COUNT,
        "dataset_info": dataset_info,
        "label": "準備 reduced 測試資料",
        "run_dir": str(run_dir),
    })

    script_summaries: List[Dict[str, Any]] = []
    overall_ok = True
    for script_offset, (name, relative_script, summary_name, timeout_sec) in enumerate(SCRIPT_ORDER, start=2):
        _emit_progress(progress_callback, "step_start", {
            "name": name,
            "major_index": script_offset,
            "major_total": MAJOR_STEP_COUNT,
            "timeout_sec": timeout_sec,
        })
        log_path = run_dir / f"{name}.log"
        run_result = _run_script(
            name=name,
            relative_script=relative_script,
            timeout_sec=timeout_sec,
            env=shared_env,
            log_path=log_path,
            progress_callback=progress_callback,
            major_index=script_offset,
        )

        summary_path = run_dir / summary_name
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"status": "PASS" if run_result["returncode"] == 0 else "FAIL"}
        script_summary = {
            "name": name,
            "status": summary_payload.get("status", "FAIL"),
            "returncode": run_result["returncode"],
            "duration_sec": run_result["duration_sec"],
            "summary_file": summary_path.name,
            "timed_out": run_result["timed_out"],
        }
        script_summaries.append(script_summary)
        if run_result["returncode"] != 0 or summary_payload.get("status") != "PASS":
            overall_ok = False

        _emit_progress(progress_callback, "step_finish", {
            **script_summary,
            "major_index": script_offset,
            "major_total": MAJOR_STEP_COUNT,
        })

    _emit_progress(progress_callback, "finalizing", {
        "major_index": MAJOR_STEP_COUNT,
        "major_total": MAJOR_STEP_COUNT,
        "label": "整理輸出與打包 bundle",
    })

    write_text(run_dir / "console_tail.txt", gather_recent_console_tail(run_dir) + "\n")
    master_summary = {
        "overall_status": "PASS" if overall_ok else "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": dataset_info,
        "timestamp": taipei_now().isoformat(),
        "git_commit": resolve_git_commit(),
        "run_dir": str(run_dir),
        "scripts": script_summaries,
        "failures": sum(1 for item in script_summaries if item["status"] != "PASS"),
    }
    write_json(run_dir / "master_summary.json", master_summary)
    write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))
    bundle_path = build_bundle_zip(run_dir, str(manifest["bundle_name"]))
    root_bundle_copy = publish_root_bundle_copy(bundle_path)
    latest_dir = finalize_latest(run_dir)
    master_summary["bundle"] = str(bundle_path)
    master_summary["root_bundle_copy"] = str(root_bundle_copy)
    master_summary["latest_dir"] = str(latest_dir)
    write_json(run_dir / "master_summary.json", master_summary)
    write_json(latest_dir / "master_summary.json", master_summary)

    result = {
        "overall_status": master_summary["overall_status"],
        "dataset": manifest["dataset"],
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "bundle": str(latest_dir / manifest["bundle_name"]),
        "root_bundle_copy": str(root_bundle_copy),
        "scripts": script_summaries,
        "failures": master_summary["failures"],
        "major_index": MAJOR_STEP_COUNT,
        "major_total": MAJOR_STEP_COUNT,
    }
    _emit_progress(progress_callback, "done", result)
    return result


def main() -> int:
    result = execute_all()
    print(json.dumps({
        "overall_status": result["overall_status"],
        "latest_dir": result["latest_dir"],
        "bundle": result["bundle"],
        "root_bundle_copy": result["root_bundle_copy"],
    }, ensure_ascii=False))
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
