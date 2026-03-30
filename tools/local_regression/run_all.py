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

from core.output_paths import output_dir_path
from core.output_retention import RetentionRule, apply_retention_rules
from core.runtime_utils import has_help_flag, resolve_cli_program_name, run_cli_entrypoint
from tools.validate.preflight_env import format_preflight_summary, run_preflight
from tools.local_regression.common import (
    archive_bundle_history,
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
STEP_NAMES = [name for name, *_ in SCRIPT_ORDER]
DATASET_REQUIRED_STEPS = {"consistency", "chain_checks", "ml_smoke"}
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _normalize_selected_steps(selected_steps: Optional[List[str]]) -> List[str]:
    if not selected_steps:
        return list(STEP_NAMES)

    normalized: List[str] = []
    seen = set()
    invalid: List[str] = []
    for raw_name in selected_steps:
        name = str(raw_name).strip()
        if not name:
            continue
        if name not in STEP_NAMES:
            invalid.append(name)
            continue
        if name in seen:
            continue
        normalized.append(name)
        seen.add(name)

    if invalid:
        valid_text = ", ".join(STEP_NAMES)
        raise ValueError(f"--only 只接受 {valid_text}，收到: {', '.join(invalid)}")
    if not normalized:
        raise ValueError("--only 不可為空；可接受的步驟: " + ", ".join(STEP_NAMES))
    return normalized


def _parse_cli_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = list(sys.argv[1:] if argv is None else argv[1:])
    if not args:
        return {"only_steps": None}

    only_steps: Optional[List[str]] = None
    index = 0
    while index < len(args):
        arg = str(args[index]).strip()
        if arg in {"-h", "--help"}:
            return {"help": True}
        if arg == "--only":
            if index + 1 >= len(args):
                raise ValueError("--only 缺少值；例: --only quick_gate,consistency")
            raw_value = str(args[index + 1]).strip()
            only_steps = [item.strip() for item in raw_value.split(",")]
            index += 2
            continue
        if arg.startswith("--only="):
            raw_value = arg.split("=", 1)[1].strip()
            only_steps = [item.strip() for item in raw_value.split(",")]
            index += 1
            continue
        raise ValueError(f"不支援的參數: {arg}")

    return {"only_steps": _normalize_selected_steps(only_steps)}


def _major_step_total(*, selected_steps: List[str], include_dataset: bool) -> int:
    return 1 + (1 if include_dataset else 0) + len(selected_steps) + 1


def _suggest_rerun_command(failed_steps: List[str]) -> str:
    if not failed_steps:
        return ""
    joined = ",".join(failed_steps)
    return f"python tools/local_regression/run_all.py --only {joined}"


def _read_step_payloads(run_dir: Path) -> Dict[str, Any]:
    return {
        "preflight": read_json_if_exists(run_dir / "preflight_summary.json"),
        "quick_gate": read_json_if_exists(run_dir / "quick_gate_summary.json"),
        "consistency": read_json_if_exists(run_dir / "validate_consistency_summary.json"),
        "chain_checks": read_json_if_exists(run_dir / "chain_summary.json"),
        "ml_smoke": read_json_if_exists(run_dir / "ml_smoke_summary.json"),
    }


def _emit_progress(progress_callback: Optional[ProgressCallback], event: str, payload: Dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event, payload)


def _build_retention_rules(manifest: Dict[str, Any]) -> List[RetentionRule]:
    return [
        RetentionRule(
            name="local_regression_history",
            target_dir=output_dir_path(PROJECT_ROOT, "local_regression"),
            patterns=["to_chatgpt_bundle_*.zip"],
            keep_last_n=int(manifest["local_regression_keep_last_n"]),
            max_age_days=int(manifest["local_regression_max_age_days"]),
        ),
        RetentionRule(
            name="validate_consistency",
            target_dir=output_dir_path(PROJECT_ROOT, "validate_consistency"),
            patterns=["*"],
            keep_last_n=int(manifest["summary_tools_keep_last_n"]),
            max_age_days=int(manifest["summary_tools_max_age_days"]),
        ),
        RetentionRule(
            name="portfolio_sim",
            target_dir=output_dir_path(PROJECT_ROOT, "portfolio_sim"),
            patterns=["*"],
            keep_last_n=int(manifest["summary_tools_keep_last_n"]),
            max_age_days=int(manifest["summary_tools_max_age_days"]),
        ),
        RetentionRule(
            name="ml_optimizer",
            target_dir=output_dir_path(PROJECT_ROOT, "ml_optimizer"),
            patterns=["*"],
            keep_last_n=int(manifest["detail_tools_keep_last_n"]),
            max_age_days=int(manifest["detail_tools_max_age_days"]),
        ),
        RetentionRule(
            name="vip_scanner",
            target_dir=output_dir_path(PROJECT_ROOT, "vip_scanner"),
            patterns=["*"],
            keep_last_n=int(manifest["detail_tools_keep_last_n"]),
            max_age_days=int(manifest["detail_tools_max_age_days"]),
        ),
        RetentionRule(
            name="smart_downloader",
            target_dir=output_dir_path(PROJECT_ROOT, "smart_downloader"),
            patterns=["*"],
            keep_last_n=int(manifest["detail_tools_keep_last_n"]),
            max_age_days=int(manifest["detail_tools_max_age_days"]),
        ),
        RetentionRule(
            name="debug_trade_log",
            target_dir=output_dir_path(PROJECT_ROOT, "debug_trade_log"),
            patterns=["*"],
            keep_last_n=int(manifest["detail_tools_keep_last_n"]),
            max_age_days=int(manifest["detail_tools_max_age_days"]),
        ),
        RetentionRule(
            name="local_regression_staging",
            target_dir=output_dir_path(PROJECT_ROOT, "local_regression") / "_staging",
            patterns=["*"],
            keep_last_n=0,
            max_age_days=1,
            include_dirs=True,
        ),
    ]


def _apply_output_retention(manifest: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(manifest.get("retention_enabled", True)):
        return {"removed_count": 0, "removed_bytes": 0, "removed_entries": []}
    return apply_retention_rules(_build_retention_rules(manifest))


def _run_preflight(run_dir: Path) -> Dict[str, Any]:
    started = time.time()
    payload = run_preflight()
    duration_sec = round(time.time() - started, 3)
    write_json(run_dir / "preflight_summary.json", payload)
    write_text(run_dir / "preflight_summary.txt", format_preflight_summary(payload) + "\n")
    return {
        "status": payload["status"],
        "duration_sec": duration_sec,
        "failed_packages": payload.get("failed_packages", []),
        "python_executable": payload.get("python_executable", sys.executable),
        "summary_file": "preflight_summary.json",
        "summary_text_file": "preflight_summary.txt",
    }


def _run_script(
    *,
    name: str,
    relative_script: str,
    timeout_sec: int,
    env: Dict[str, str],
    log_path: Path,
    progress_callback: Optional[ProgressCallback],
    major_index: int,
    major_total: int,
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
                    "major_total": major_total,
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


def execute_all(
    progress_callback: Optional[ProgressCallback] = None,
    selected_steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    manifest = load_manifest()
    run_dir = create_staging_run_dir()
    shared_env = build_python_env({"V16_LOCAL_REGRESSION_RUN_DIR": str(run_dir)})
    selected_step_names = _normalize_selected_steps(selected_steps)
    include_dataset = any(name in DATASET_REQUIRED_STEPS for name in selected_step_names)
    major_total = _major_step_total(selected_steps=selected_step_names, include_dataset=include_dataset)

    try:
        _emit_progress(progress_callback, "step_start", {
            "name": "preflight",
            "major_index": 1,
            "major_total": major_total,
            "timeout_sec": 0,
        })
        preflight_summary = _run_preflight(run_dir)
        _emit_progress(progress_callback, "step_finish", {
            "name": "preflight",
            "status": preflight_summary["status"],
            "duration_sec": preflight_summary["duration_sec"],
            "major_index": 1,
            "major_total": major_total,
        })

        if preflight_summary["status"] != "PASS":
            preflight_payload = read_json_if_exists(run_dir / "preflight_summary.json")
            master_summary = {
                "overall_status": "FAIL",
                "dataset": manifest["dataset"],
                "dataset_info": {},
                "timestamp": taipei_now().isoformat(),
                "git_commit": resolve_git_commit(),
                "scripts": [],
                "selected_steps": selected_step_names,
                "failures": 1,
                "preflight": preflight_payload,
                "bundle_mode": "preflight_failed",
                "failed_step_names": ["preflight"],
                "suggested_rerun_command": "",
            }
            write_json(run_dir / "master_summary.json", master_summary)
            write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))
            bundle_paths = [
                run_dir / "master_summary.json",
                run_dir / "preflight_summary.json",
                run_dir / "preflight_summary.txt",
                run_dir / "artifacts_manifest.json",
            ]
            bundle_path = build_bundle_zip(run_dir, str(manifest["bundle_name"]), include_paths=bundle_paths)
            archived_bundle = archive_bundle_history(bundle_path)
            root_bundle_copy = publish_root_bundle_copy(archived_bundle)
            retention = _apply_output_retention(manifest)
            result = {
                "overall_status": "FAIL",
                "dataset": manifest["dataset"],
                "bundle": str(root_bundle_copy),
                "archived_bundle": str(archived_bundle),
                "root_bundle_copy": str(root_bundle_copy),
                "bundle_mode": "preflight_failed",
                "bundle_entries": [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths if path.exists()],
                "scripts": [],
                "step_payloads": {"preflight": preflight_payload},
                "failures": 1,
                "failed_step_names": ["preflight"],
                "retention": {
                    "removed_count": retention.get("removed_count", 0),
                    "removed_bytes": retention.get("removed_bytes", 0),
                },
                "major_index": 1,
                "major_total": major_total,
                "preflight": preflight_summary,
                "selected_steps": selected_step_names,
                "suggested_rerun_command": "",
            }
            _emit_progress(progress_callback, "done", result)
            return result

        dataset_info: Dict[str, Any] = {}
        next_major_index = 2
        if include_dataset:
            try:
                dataset_info = ensure_reduced_dataset()
            except Exception as exc:
                dataset_prepare_summary = {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                write_json(run_dir / "dataset_prepare_summary.json", dataset_prepare_summary)
                write_text(
                    run_dir / "dataset_prepare_summary.txt",
                    f"dataset_prepare failed: {type(exc).__name__}: {exc}\n",
                )
                preflight_payload = read_json_if_exists(run_dir / "preflight_summary.json")
                master_summary = {
                    "overall_status": "FAIL",
                    "dataset": manifest["dataset"],
                    "dataset_info": {},
                    "timestamp": taipei_now().isoformat(),
                    "git_commit": resolve_git_commit(),
                    "scripts": [],
                    "selected_steps": selected_step_names,
                    "failures": 1,
                    "preflight": preflight_payload,
                    "dataset_prepare": dataset_prepare_summary,
                    "bundle_mode": "dataset_prepare_failed",
                    "failed_step_names": ["dataset_prepare"],
                    "suggested_rerun_command": "",
                }
                write_json(run_dir / "master_summary.json", master_summary)
                write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))
                bundle_paths = [
                    run_dir / "master_summary.json",
                    run_dir / "preflight_summary.json",
                    run_dir / "preflight_summary.txt",
                    run_dir / "dataset_prepare_summary.json",
                    run_dir / "dataset_prepare_summary.txt",
                    run_dir / "artifacts_manifest.json",
                ]
                bundle_path = build_bundle_zip(run_dir, str(manifest["bundle_name"]), include_paths=bundle_paths)
                archived_bundle = archive_bundle_history(bundle_path)
                root_bundle_copy = publish_root_bundle_copy(archived_bundle)
                retention = _apply_output_retention(manifest)
                result = {
                    "overall_status": "FAIL",
                    "dataset": manifest["dataset"],
                    "bundle": str(root_bundle_copy),
                    "archived_bundle": str(archived_bundle),
                    "root_bundle_copy": str(root_bundle_copy),
                    "bundle_mode": "dataset_prepare_failed",
                    "bundle_entries": [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths if path.exists()],
                    "scripts": [],
                    "step_payloads": {"preflight": preflight_payload, "dataset_prepare": dataset_prepare_summary},
                    "failures": 1,
                    "failed_step_names": ["dataset_prepare"],
                    "retention": {
                        "removed_count": retention.get("removed_count", 0),
                        "removed_bytes": retention.get("removed_bytes", 0),
                    },
                    "major_index": next_major_index,
                    "major_total": major_total,
                    "preflight": preflight_summary,
                    "selected_steps": selected_step_names,
                    "suggested_rerun_command": "",
                }
                _emit_progress(progress_callback, "done", result)
                return result
            _emit_progress(progress_callback, "dataset_ready", {
                "major_index": next_major_index,
                "major_total": major_total,
                "dataset_info": dataset_info,
                "label": "準備 reduced 測試資料",
            })
            next_major_index += 1

        selected_script_order = [item for item in SCRIPT_ORDER if item[0] in selected_step_names]
        script_summaries: List[Dict[str, Any]] = []
        overall_ok = True
        for script_offset, (name, relative_script, summary_name, timeout_sec) in enumerate(selected_script_order, start=next_major_index):
            _emit_progress(progress_callback, "step_start", {
                "name": name,
                "major_index": script_offset,
                "major_total": major_total,
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
                major_total=major_total,
            )

            summary_path = run_dir / summary_name
            summary_payload = read_json_if_exists(summary_path)
            summary_exists = summary_path.exists()
            reported_status = summary_payload.get("status", "") if summary_payload else ""
            effective_status = "PASS"
            failure_reasons: List[str] = []
            if not summary_exists:
                effective_status = "FAIL"
                failure_reasons.append("missing_summary_file")
            if run_result["timed_out"]:
                effective_status = "FAIL"
                failure_reasons.append("timed_out")
            if run_result["returncode"] != 0:
                effective_status = "FAIL"
                failure_reasons.append(f"returncode={run_result['returncode']}")
            if reported_status != "PASS":
                effective_status = "FAIL"
                failure_reasons.append(f"reported_status={reported_status or 'missing'}")
            script_summary = {
                "name": name,
                "status": effective_status,
                "reported_status": reported_status or "missing",
                "returncode": run_result["returncode"],
                "duration_sec": run_result["duration_sec"],
                "summary_file": summary_path.name,
                "summary_exists": summary_exists,
                "timed_out": run_result["timed_out"],
                "failure_reasons": failure_reasons,
            }
            script_summaries.append(script_summary)
            if effective_status != "PASS":
                overall_ok = False

            _emit_progress(progress_callback, "step_finish", {
                **script_summary,
                "major_index": script_offset,
                "major_total": major_total,
            })

        _emit_progress(progress_callback, "finalizing", {
            "major_index": major_total,
            "major_total": major_total,
            "label": "整理輸出與打包 bundle",
        })

        write_text(run_dir / "console_tail.txt", gather_recent_console_tail(run_dir) + "\n")
        step_payloads = _read_step_payloads(run_dir)
        master_summary = {
            "overall_status": "PASS" if overall_ok else "FAIL",
            "dataset": manifest["dataset"],
            "dataset_info": dataset_info,
            "timestamp": taipei_now().isoformat(),
            "git_commit": resolve_git_commit(),
            "scripts": script_summaries,
            "selected_steps": selected_step_names,
            "failures": sum(1 for item in script_summaries if item["status"] != "PASS"),
            "preflight": step_payloads["preflight"],
        }
        failed_step_names = [item["name"] for item in script_summaries if item["status"] != "PASS"]
        suggested_rerun_command = _suggest_rerun_command(failed_step_names)

        write_json(run_dir / "master_summary.json", master_summary)
        write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))

        bundle_mode = "minimum_set" if overall_ok else "debug_bundle"
        bundle_paths = select_bundle_paths(run_dir, overall_ok=overall_ok)
        master_summary["bundle_mode"] = bundle_mode
        master_summary["bundle_entries"] = [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths]
        master_summary["failed_step_names"] = failed_step_names
        master_summary["suggested_rerun_command"] = suggested_rerun_command
        write_json(run_dir / "master_summary.json", master_summary)
        write_json(run_dir / "artifacts_manifest.json", build_artifacts_manifest(run_dir))

        bundle_path = build_bundle_zip(run_dir, str(manifest["bundle_name"]), include_paths=bundle_paths)
        archived_bundle = archive_bundle_history(bundle_path)
        root_bundle_copy = publish_root_bundle_copy(archived_bundle)
        retention = _apply_output_retention(manifest)

        result = {
            "overall_status": master_summary["overall_status"],
            "dataset": manifest["dataset"],
            "bundle": str(root_bundle_copy),
            "archived_bundle": str(archived_bundle),
            "root_bundle_copy": str(root_bundle_copy),
            "bundle_mode": bundle_mode,
            "bundle_entries": [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths],
            "scripts": script_summaries,
            "step_payloads": step_payloads,
            "failures": master_summary["failures"],
            "retention": {
                "removed_count": retention.get("removed_count", 0),
                "removed_bytes": retention.get("removed_bytes", 0),
            },
            "major_index": major_total,
            "major_total": major_total,
            "preflight": preflight_summary,
            "selected_steps": selected_step_names,
            "failed_step_names": failed_step_names,
            "suggested_rerun_command": suggested_rerun_command,
        }
        _emit_progress(progress_callback, "done", result)
        return result
    finally:
        cleanup_staging_dir(run_dir)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv if argv is None else argv)
    if has_help_flag(args):
        program_name = resolve_cli_program_name(args, "tools/local_regression/run_all.py")
        print(f"用法: python {program_name} [--only quick_gate,consistency,chain_checks,ml_smoke]")
        print("說明: 預設先跑完整 reduced regression；若完整入口已找出失敗步驟，可再用 --only 只重跑指定步驟。")
        return 0

    try:
        parsed = _parse_cli_args(args)
    except ValueError as exc:
        print(f"參數錯誤: {exc}", file=sys.stderr)
        program_name = resolve_cli_program_name(args, "tools/local_regression/run_all.py")
        print(f"用法: python {program_name} [--only quick_gate,consistency,chain_checks,ml_smoke]", file=sys.stderr)
        return 2

    result = execute_all(selected_steps=parsed.get("only_steps"))
    print(json.dumps({
        "overall_status": result["overall_status"],
        "bundle": result["bundle"],
        "archived_bundle": result.get("archived_bundle", ""),
        "bundle_mode": result["bundle_mode"],
        "retention": result.get("retention", {}),
        "preflight": result.get("preflight", {}),
        "selected_steps": result.get("selected_steps", []),
        "failed_step_names": result.get("failed_step_names", []),
        "suggested_rerun_command": result.get("suggested_rerun_command", ""),
    }, ensure_ascii=False))
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
