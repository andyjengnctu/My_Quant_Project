from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
    ("chain_checks", "tools/local_regression/run_chain_checks.py", "chain_summary.json", 900),
    ("ml_smoke", "tools/local_regression/run_ml_smoke.py", "ml_smoke_summary.json", 900),
]


def main() -> int:
    manifest = load_manifest()
    ensure_reduced_dataset()
    run_dir = ensure_dir(OUTPUT_ROOT / "runs" / timestamp_text())
    shared_env = build_python_env({"V16_LOCAL_REGRESSION_RUN_DIR": str(run_dir)})

    script_summaries: List[Dict[str, Any]] = []
    overall_ok = True
    for name, relative_script, summary_name, timeout_sec in SCRIPT_ORDER:
        log_path = run_dir / f"{name}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {sys.executable} {relative_script}\n")
            log_file.flush()
            try:
                proc = subprocess.run(
                    [sys.executable, relative_script],
                    cwd=str(PROJECT_ROOT),
                    env=shared_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_sec,
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                log_file.write(f"\n[local_regression] TIMEOUT after {timeout_sec} seconds\n")
        duration_sec = round(time.time() - started, 3)
        summary_path = run_dir / summary_name
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"status": "PASS" if returncode == 0 else "FAIL"}
        script_summaries.append({
            "name": name,
            "status": summary_payload.get("status", "FAIL"),
            "returncode": returncode,
            "duration_sec": duration_sec,
            "summary_file": summary_path.name,
        })
        if returncode != 0 or summary_payload.get("status") != "PASS":
            overall_ok = False

    write_text(run_dir / "console_tail.txt", gather_recent_console_tail(run_dir) + "\n")
    master_summary = {
        "overall_status": "PASS" if overall_ok else "FAIL",
        "dataset": manifest["dataset"],
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
    print(json.dumps({
        "overall_status": master_summary["overall_status"],
        "latest_dir": str(latest_dir),
        "bundle": str(latest_dir / manifest["bundle_name"]),
        "root_bundle_copy": str(root_bundle_copy),
    }, ensure_ascii=False))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
