"""Isolated fold-concurrency sweep for Outer-Rolling timing mode.

Each worker-count case runs in a fresh Python process so process-local caches,
working sets, and executor state cannot leak across A/B cases.  The sweep only
changes execution knobs; scientific inputs, seeds, folds, and objective logic
remain owned by the canonical Optimizer path.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

from core.runtime_utils import get_taipei_now


_TIMING_JSON_RE = re.compile(r"^outer_rolling_oos_timing_\d{8}_\d{6}\.json$")


def parse_outer_rolling_timing_worker_sweep(raw_value) -> list[int]:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("--timing-fold-workers 需要至少一個正整數，例如 1,2,3,4,6。")
    workers: list[int] = []
    seen: set[int] = set()
    for raw_part in text.split(","):
        part = str(raw_part or "").strip()
        if not part:
            raise ValueError(f"--timing-fold-workers 格式不合法: {text!r}")
        try:
            value = int(part)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--timing-fold-workers 只接受逗號分隔正整數，收到: {part!r}") from exc
        if value < 1:
            raise ValueError(f"--timing-fold-workers 必須 >= 1，收到: {value}")
        if value in seen:
            continue
        workers.append(value)
        seen.add(value)
    if not workers:
        raise ValueError("--timing-fold-workers 需要至少一個正整數。")
    return workers


def _strip_cli_value_option(argv: Iterable[str], option: str) -> list[str]:
    args = list(argv or [])
    tail = [str(item) for item in args[1:]]
    out: list[str] = []
    idx = 0
    while idx < len(tail):
        raw = str(tail[idx])
        if raw == option:
            idx += 2
            continue
        if raw.startswith(option + "="):
            idx += 1
            continue
        out.append(raw)
        idx += 1
    return out


def _timing_json_snapshot(report_dir: Path) -> dict[Path, tuple[int, int]]:
    if not report_dir.is_dir():
        return {}
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in report_dir.glob("outer_rolling_oos_timing_*.json"):
        if not path.is_file() or not _TIMING_JSON_RE.match(path.name):
            continue
        resolved = path.resolve()
        stat = resolved.stat()
        snapshot[resolved] = (int(stat.st_mtime_ns), int(stat.st_size))
    return snapshot


def _load_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"timing JSON root 必須是 object: {path.name}")
    return payload


def _project_relative(path: Path | str, project_root: Path) -> str:
    try:
        return os.path.relpath(str(path), str(project_root)).replace(os.sep, "/")
    except ValueError:
        return Path(path).name


def _build_case_row(*, requested_workers: int, returncode: int, timing_path: Path | None, payload: dict | None, project_root: Path) -> dict:
    timing_payload = dict(payload or {})
    meta = dict(timing_payload.get("meta") or {})
    summary = dict(timing_payload.get("summary") or {})
    overall_sec = float(summary.get("overall_sec", 0.0) or 0.0)
    completed_trials = int(meta.get("completed_trials", summary.get("completed_trials", 0)) or 0)
    return {
        "requested_fold_workers": int(requested_workers),
        "effective_fold_workers": int(summary.get("rolling_fold_workers_max", 0) or 0),
        "status": "PASS" if int(returncode) == 0 and timing_path is not None else "FAIL",
        "returncode": int(returncode),
        "completed_trials": int(completed_trials),
        "overall_sec": overall_sec,
        "overall_trials_per_sec": (float(completed_trials) / overall_sec) if completed_trials > 0 and overall_sec > 0 else 0.0,
        "avg_trial_sec": float(summary.get("avg_optimize_sec_per_completed_trial", 0.0) or 0.0),
        "fold_wall_sec": float(summary.get("fold_wall_sec", 0.0) or 0.0),
        "cpu_avg_percent": float(summary.get("cpu_avg_percent", 0.0) or 0.0),
        "cpu_max_percent": float(summary.get("cpu_max_percent", 0.0) or 0.0),
        "memory_avg_percent": float(summary.get("memory_avg_percent", 0.0) or 0.0),
        "memory_max_percent": float(summary.get("memory_max_percent", 0.0) or 0.0),
        "memory_min_available_gb": float(summary.get("memory_min_available_gb", 0.0) or 0.0),
        "process_tree_rss_max_gb": float(summary.get("process_tree_rss_max_gb", 0.0) or 0.0),
        "process_tree_count_max": int(summary.get("process_tree_count_max", 0) or 0),
        "commit_max_percent": float(summary.get("commit_max_percent", 0.0) or 0.0),
        "commit_max_used_gb": float(summary.get("commit_max_used_gb", 0.0) or 0.0),
        "commit_min_available_gb": float(summary.get("commit_min_available_gb", 0.0) or 0.0),
        "disk_read_mb": float(summary.get("disk_read_mb", 0.0) or 0.0),
        "disk_write_mb": float(summary.get("disk_write_mb", 0.0) or 0.0),
        "disk_total_mb_per_sec_avg": float(summary.get("disk_total_mb_per_sec_avg", 0.0) or 0.0),
        "disk_total_mb_per_sec_max": float(summary.get("disk_total_mb_per_sec_max", 0.0) or 0.0),
        "timing_json": "" if timing_path is None else _project_relative(timing_path, project_root),
    }


def _write_sweep_artifacts(*, output_dir: Path, project_root: Path, workers: list[int], rows: list[dict], child_args: list[str]) -> tuple[Path, Path]:
    report_dir = output_dir / "outer_rolling_oos"
    report_dir.mkdir(parents=True, exist_ok=True)
    session_ts = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    base = report_dir / f"outer_rolling_oos_timing_sweep_{session_ts}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")
    payload = {
        "type": "outer_rolling_oos_timing_worker_sweep",
        "version": 1,
        "created_at": get_taipei_now().isoformat(),
        "requested_fold_workers": [int(value) for value in workers],
        "inner_search_parallel_trials": 1,
        "resource_csv_enabled": True,
        "execution_isolation": "fresh_subprocess_per_worker_case",
        "scientific_semantics": "unchanged; execution strategy only",
        "base_args": list(child_args),
        "cases": list(rows),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    fieldnames = list(rows[0].keys()) if rows else ["requested_fold_workers", "status", "returncode"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _print_comparison(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n====================================================================================================")
    print(" Rolling timing｜fold concurrency A/B")
    print("====================================================================================================")
    print(" workers  status  total(s)  avg_trial  CPU avg/max   MEM avg/max   RSS max  avail min  commit max  disk avg/max")
    print(" -------  ------  --------  ---------  ------------  ------------  -------  ---------  ----------  ------------")
    for row in rows:
        requested = int(row.get("requested_fold_workers", 0) or 0)
        effective = int(row.get("effective_fold_workers", 0) or 0)
        worker_text = str(requested) if effective in {0, requested} else f"{requested}->{effective}"
        print(
            f" {worker_text:>7}  {str(row.get('status') or ''):<6}  "
            f"{float(row.get('overall_sec', 0.0) or 0.0):>8.1f}  "
            f"{float(row.get('avg_trial_sec', 0.0) or 0.0):>9.3f}  "
            f"{float(row.get('cpu_avg_percent', 0.0) or 0.0):>5.1f}/{float(row.get('cpu_max_percent', 0.0) or 0.0):<5.1f}  "
            f"{float(row.get('memory_avg_percent', 0.0) or 0.0):>5.1f}/{float(row.get('memory_max_percent', 0.0) or 0.0):<5.1f}  "
            f"{float(row.get('process_tree_rss_max_gb', 0.0) or 0.0):>7.2f}  "
            f"{float(row.get('memory_min_available_gb', 0.0) or 0.0):>9.2f}  "
            f"{float(row.get('commit_max_percent', 0.0) or 0.0):>10.1f}  "
            f"{float(row.get('disk_total_mb_per_sec_avg', 0.0) or 0.0):>5.1f}/{float(row.get('disk_total_mb_per_sec_max', 0.0) or 0.0):<5.1f}"
        )


def run_outer_rolling_timing_worker_sweep(*, argv, environ, project_root: str, output_dir: str, workers: list[int]) -> int:
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    report_dir = output / "outer_rolling_oos"
    child_tail = _strip_cli_value_option(argv, "--timing-fold-workers")
    if "--yes" not in child_tail:
        child_tail.append("--yes")
    command_prefix = [sys.executable, str(root / "apps" / "research.py"), "optimizer", "advanced"]

    base_env = dict(os.environ)
    if environ is not None:
        for key, value in dict(environ).items():
            base_env[str(key)] = str(value)

    rows: list[dict] = []
    failure_code = 0
    for case_idx, worker_count in enumerate(list(workers or []), start=1):
        print("\n====================================================================================================")
        print(
            f" Rolling timing case {case_idx}/{len(workers)} | fold_workers={int(worker_count)} | "
            "inner_search_parallel_trials=1"
        )
        print("====================================================================================================")
        before = _timing_json_snapshot(report_dir)
        child_env = dict(base_env)
        child_env["OPTIMIZER_ROLLING_FOLD_WORKERS"] = str(int(worker_count))
        child_env["OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS"] = "1"
        child_env["OPTIMIZER_RESOURCE_WRITE_CSV"] = "1"
        completed = subprocess.run(
            [*command_prefix, *child_tail],
            cwd=str(root),
            env=child_env,
            check=False,
        )
        after = _timing_json_snapshot(report_dir)
        new_paths = sorted(
            (path for path, signature in after.items() if before.get(path) != signature),
            key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        )
        timing_path = new_paths[-1] if new_paths else None
        payload = None
        if timing_path is not None:
            try:
                payload = _load_json_object(timing_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"⚠️ timing JSON 讀取失敗：{timing_path.name} | {type(exc).__name__}: {exc}")
        row = _build_case_row(
            requested_workers=int(worker_count),
            returncode=int(completed.returncode),
            timing_path=timing_path,
            payload=payload,
            project_root=root,
        )
        rows.append(row)
        if int(completed.returncode) != 0 or timing_path is None:
            failure_code = int(completed.returncode) if int(completed.returncode) != 0 else 1
            print("⚠️ 此 concurrency case 未完成，為避免在資源壓力下繼續放大 worker，本次 sweep 停止於此。")
            break

    json_path, csv_path = _write_sweep_artifacts(
        output_dir=output,
        project_root=root,
        workers=list(workers or []),
        rows=rows,
        child_args=child_tail,
    )
    _print_comparison(rows)
    print("\n💾 Rolling timing sweep 輸出")
    print(f"  JSON: {_project_relative(json_path, root)}")
    print(f"  CSV : {_project_relative(csv_path, root)}")
    return int(failure_code)
