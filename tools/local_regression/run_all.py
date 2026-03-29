from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.local_regression.common import OUTPUT_ROOT, build_artifacts_manifest, build_bundle_zip, ensure_dir, ensure_reduced_dataset, finalize_latest, gather_recent_console_tail, load_manifest, resolve_git_commit, taipei_now, timestamp_text, write_json, write_text

SCRIPT_ORDER = [
    ("quick_gate", "tools/local_regression/run_quick_gate.py", "quick_gate_summary.json"),
    ("chain_checks", "tools/local_regression/run_chain_checks.py", "chain_checks_summary.json"),
    ("ml_smoke", "tools/local_regression/run_ml_smoke.py", "ml_smoke_summary.json"),
]


def list_child_run_dirs() -> Dict[str, Path]:
    child_dirs: Dict[str, Path] = {}
    if not OUTPUT_ROOT.exists():
        return child_dirs
    for path in OUTPUT_ROOT.iterdir():
        if not path.is_dir() or path.name in {"latest", "runs"}:
            continue
        child_dirs[path.name] = path
    return child_dirs


def detect_new_child_dir(before: Dict[str, Path], after: Dict[str, Path], script_name: str) -> Optional[Path]:
    new_names = [name for name in after if name not in before]
    if new_names:
        candidates = [after[name] for name in new_names if name.endswith(f"_{script_name}")]
        if not candidates:
            candidates = [after[name] for name in new_names]
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    same_suffix = [path for name, path in after.items() if name.endswith(f"_{script_name}")]
    if same_suffix:
        return sorted(same_suffix, key=lambda p: p.stat().st_mtime)[-1]
    return None


def copy_child_artifacts(child_dir: Path, dest_dir: Path) -> None:
    for item in child_dir.iterdir():
        target = dest_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> int:
    manifest = load_manifest()
    ensure_reduced_dataset()
    run_dir = ensure_dir(OUTPUT_ROOT / "runs" / timestamp_text())

    script_summaries: List[Dict[str, Any]] = []
    overall_ok = True
    for name, relative_script, summary_name in SCRIPT_ORDER:
        log_path = run_dir / f"{name}.log"
        before_dirs = list_child_run_dirs()
        started = time.time()
        log_path.write_text(f"$ {sys.executable} {relative_script}\n", encoding="utf-8")
        cmd = (
            f'cd "{PROJECT_ROOT}" && '
            f'export PYTHONUNBUFFERED=1 V16_DATASET_PROFILE=reduced V16_VALIDATE_DATASET=reduced && '
            f'"{sys.executable}" "{relative_script}" >> "{log_path}" 2>&1'
        )
        returncode = os.system(f'bash -lc {json.dumps(cmd)}')
        if returncode > 255:
            returncode = returncode >> 8
        duration_sec = round(time.time() - started, 3)
        after_dirs = list_child_run_dirs()
        child_dir = detect_new_child_dir(before_dirs, after_dirs, name)
        if child_dir is not None:
            copy_child_artifacts(child_dir, run_dir)
        summary_path = run_dir / summary_name
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"status": "PASS" if returncode == 0 else "FAIL"}
        script_summaries.append({
            "name": name,
            "status": summary_payload.get("status", "FAIL"),
            "returncode": returncode,
            "duration_sec": duration_sec,
            "summary_file": summary_path.name,
            "child_run_dir": None if child_dir is None else str(child_dir),
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
    build_bundle_zip(run_dir, str(manifest["bundle_name"]))
    latest_dir = finalize_latest(run_dir)
    print(json.dumps({"overall_status": master_summary["overall_status"], "latest_dir": str(latest_dir), "bundle": str(latest_dir / manifest["bundle_name"])}, ensure_ascii=False))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
