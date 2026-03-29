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
    build_artifacts_manifest,
    build_bundle_zip,
    build_python_env,
    cleanup_staging_dir,
    create_staging_run_dir,
    ensure_reduced_dataset,
    gather_recent_console_tail,
    load_manifest,
    publish_root_bundle_copy,
    read_json_if_exists,
    resolve_git_commit,
    select_bundle_paths,
    taipei_now,
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
                    log_file.write(f"\n[test_suite] TIMEOUT after {timeout_sec} seconds\n")
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
    run_dir = create_staging_run_dir()
    shared_env = build_python_env({"V16_LOCAL_REGRESSION_RUN_DIR": str(run_dir)})

    _emit_progress(progress_callback, "dataset_ready", {
        "major_index": 1,
        "major_total": MAJOR_STEP_COUNT,
        "dataset_info": dataset_info,
        "label": "準備 reduced 測試資料",
    })

    script_summaries: List[Dict[str, Any]] = []
    overall_ok = True
    try:
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
            summary_payload = read_json_if_exists(summary_path)
            if not summary_payload:
                summary_payload = {"status": "PASS" if run_result["returncode"] == 0 else "FAIL"}
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
        step_payloads = {
            "quick_gate": read_json_if_exists(run_dir / "quick_gate_summary.json"),
            "consistency": read_json_if_exists(run_dir / "validate_consistency_summary.json"),
            "chain_checks": read_json_if_exists(run_dir / "chain_summary.json"),
            "ml_smoke": read_json_if_exists(run_dir / "ml_smoke_summary.json"),
        }
        master_summary = {
            "overall_status": "PASS" if overall_ok else "FAIL",
            "dataset": manifest["dataset"],
            "dataset_info": dataset_info,
            "timestamp": taipei_now().isoformat(),
            "git_commit": resolve_git_commit(),
            "scripts": script_summaries,
            "failures": sum(1 for item in script_summaries if item["status"] != "PASS"),
        }
        write_json(run_dir / "master_summary.json", master_summary)
        write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))

        bundle_mode = "minimum_set" if overall_ok else "debug_bundle"
        bundle_paths = select_bundle_paths(run_dir, overall_ok=overall_ok)
        master_summary["bundle_mode"] = bundle_mode
        master_summary["bundle_entries"] = [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths]
        write_json(run_dir / "master_summary.json", master_summary)
        write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))

        bundle_path = build_bundle_zip(run_dir, str(manifest["bundle_name"]), include_paths=bundle_paths)
        root_bundle_copy = publish_root_bundle_copy(bundle_path)

        result = {
            "overall_status": master_summary["overall_status"],
            "dataset": manifest["dataset"],
            "bundle": str(root_bundle_copy),
            "root_bundle_copy": str(root_bundle_copy),
            "bundle_mode": bundle_mode,
            "bundle_entries": [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths],
            "scripts": script_summaries,
            "step_payloads": step_payloads,
            "failures": master_summary["failures"],
            "major_index": MAJOR_STEP_COUNT,
            "major_total": MAJOR_STEP_COUNT,
        }
        _emit_progress(progress_callback, "done", result)
        return result
    finally:
        cleanup_staging_dir(run_dir)


def main() -> int:
    result = execute_all()
    print(json.dumps({
        "overall_status": result["overall_status"],
        "bundle": result["bundle"],
        "bundle_mode": result["bundle_mode"],
    }, ensure_ascii=False))
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
