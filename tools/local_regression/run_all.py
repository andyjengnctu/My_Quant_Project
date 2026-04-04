from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.output_paths import output_dir_path
from core.output_retention import RetentionRule, apply_retention_rules
from core.runtime_utils import has_help_flag, resolve_cli_program_name, run_cli_entrypoint
from tools.local_regression.formal_pipeline import DATASET_REQUIRED_STEPS, FORMAL_COMMAND_ORDER, FORMAL_STEP_ORDER
from tools.validate.preflight_env import REQUIREMENTS_PATH, format_preflight_summary, run_preflight
from tools.local_regression.common import (
    archive_bundle_history,
    build_artifacts_manifest,
    DATASET_INFO_KEYS,
    LOCAL_REGRESSION_RUN_DIR_ENV,
    build_bundle_zip,
    build_python_env,
    cleanup_staging_dir,
    create_staging_run_dir,
    ensure_reduced_dataset,
    gather_recent_console_tail,
    MANIFEST_DEFAULTS,
    load_manifest,
    publish_root_bundle_copy,
    read_json_if_exists,
    resolve_git_commit,
    select_bundle_paths,
    taipei_now,
    write_json,
    write_text,
)

SCRIPT_ORDER = list(FORMAL_COMMAND_ORDER)
SCRIPT_TIMEOUT_GRACE_SEC = 30
STEP_NAMES = list(FORMAL_STEP_ORDER)
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _format_optional_int_detail(key: str, raw_value: Any) -> str:
    try:
        return f"{key}={int(raw_value)}"
    except (TypeError, ValueError) as exc:
        raw_preview = repr(raw_value)
        if len(raw_preview) > 80:
            raw_preview = raw_preview[:77] + "..."
        return f"{key}=INVALID({type(exc).__name__}:{raw_preview})"


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


def _resolve_step_timeout(name: str, manifest: Dict[str, Any]) -> int:
    if name == "ml_smoke":
        base_timeout = max(int(manifest["ml_smoke_timeout_sec"]), int(manifest["subprocess_timeout_sec"]))
        return base_timeout + SCRIPT_TIMEOUT_GRACE_SEC
    return int(manifest["subprocess_timeout_sec"])


def _suggest_rerun_command(failed_steps: List[str]) -> str:
    if not failed_steps:
        return ""
    joined = ",".join(failed_steps)
    return f"python tools/local_regression/run_all.py --only {joined}"


def _compute_not_run_step_names(
    *,
    selected_step_names: List[str],
    failed_step_names: List[str],
    completed_script_names: Optional[List[str]] = None,
    include_dataset: bool,
) -> List[str]:
    completed = set(completed_script_names or [])
    failed = set(failed_step_names)
    not_run: List[str] = []

    if "manifest" in failed:
        not_run.append("preflight")
        if include_dataset:
            not_run.append("dataset_prepare")
        for name in selected_step_names:
            if name not in completed:
                not_run.append(name)
        return not_run

    if "preflight" in failed:
        if include_dataset:
            not_run.append("dataset_prepare")
        for name in selected_step_names:
            if name not in completed:
                not_run.append(name)
        return not_run

    if "dataset_prepare" in failed:
        for name in selected_step_names:
            if name not in completed:
                not_run.append(name)
        return not_run

    return [name for name in selected_step_names if name not in completed]


def _safe_read_json_with_error(path: Path) -> Dict[str, Any]:
    try:
        return read_json_if_exists(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "summary_file": path.name,
            "summary_unreadable": True,
        }


def _payload_failure_reasons(payload: Dict[str, Any]) -> List[str]:
    if not payload:
        return ["missing_summary_file"]

    reasons: List[str] = []
    if payload.get("summary_unreadable"):
        reasons.append("summary_unreadable")

    reported_status = str(payload.get("status", "") or "").strip()
    if reported_status != "PASS":
        reasons.append(f"reported_status={reported_status or 'missing'}")

    error_type = str(payload.get("error_type", "") or "").strip()
    if error_type:
        reasons.append(f"error_type={error_type}")
    error_message = str(payload.get("error_message", "") or payload.get("runtime_error", "") or "").strip()
    if error_message:
        reasons.append(f"error_message={error_message}")
    failed_packages = [str(item).strip() for item in payload.get("failed_packages", []) if str(item).strip()]
    if failed_packages:
        reasons.append("failed_packages=" + ",".join(failed_packages))
    summary_write_error = str(payload.get("summary_write_error", "") or "").strip()
    if summary_write_error:
        reasons.append(f"summary_write_error={summary_write_error}")
    return reasons


def _read_step_payloads(run_dir: Path) -> Dict[str, Any]:
    payloads = {
        "preflight": _safe_read_json_with_error(run_dir / "preflight_summary.json"),
        "dataset_prepare": _safe_read_json_with_error(run_dir / "dataset_prepare_summary.json"),
    }
    for name, _, summary_name in SCRIPT_ORDER:
        payloads[name] = _safe_read_json_with_error(run_dir / summary_name)
    return payloads


def _emit_progress(progress_callback: Optional[ProgressCallback], event: str, payload: Dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event, payload)


def _run_inline_step_with_progress(
    *,
    name: str,
    major_index: int,
    major_total: int,
    progress_callback: Optional[ProgressCallback],
    func: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    holder: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            holder["result"] = func()
        except (KeyboardInterrupt, SystemExit) as exc:
            holder["fatal_exc"] = exc
        except Exception as exc:
            holder["exc"] = exc

    started = time.time()
    worker = threading.Thread(target=_runner, name=f"local_regression::{name}", daemon=True)
    worker.start()
    last_emit_ts = 0.0
    while worker.is_alive():
        worker.join(timeout=0.2)
        if not worker.is_alive():
            break
        now = time.time()
        if now - last_emit_ts >= 0.2:
            last_emit_ts = now
            _emit_progress(progress_callback, "step_progress", {
                "name": name,
                "major_index": major_index,
                "major_total": major_total,
                "elapsed_sec": round(now - started, 1),
                "timeout_sec": 0,
            })
    if "fatal_exc" in holder:
        raise holder["fatal_exc"]
    if "exc" in holder:
        raise holder["exc"]
    return holder["result"]


def _build_bundle_entries(run_dir: Path, bundle_paths: List[Path]) -> List[str]:
    return [str(path.relative_to(run_dir)).replace("\\", "/") for path in bundle_paths if path.exists()]


def _build_dataset_prepare_placeholder(*, include_dataset: bool, blocked_by: str) -> Dict[str, Any]:
    if not include_dataset:
        return {"status": "SKIP", "reason": "not_required"}
    return {"status": "NOT_RUN", "blocked_by": blocked_by}


def _write_stable_artifacts_manifest(run_dir: Path, *, max_rounds: int = 6) -> Path:
    manifest_path = run_dir / "artifacts_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"artifact_count": 0, "artifacts": []})

    for _ in range(max_rounds):
        write_json(manifest_path, build_artifacts_manifest(run_dir))
        try:
            payload = read_json_if_exists(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            break

        self_size = None
        for item in payload.get("artifacts", []):
            if item.get("relative_path") == manifest_path.name:
                try:
                    self_size = int(item.get("size_bytes", -1))
                except (TypeError, ValueError):
                    self_size = None
                break
        if self_size == manifest_path.stat().st_size:
            break
    return manifest_path


def _finalize_early_failure(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    selected_step_names: List[str],
    major_index: int,
    major_total: int,
    bundle_mode: str,
    failed_step_names: List[str],
    preflight_payload: Dict[str, Any],
    include_dataset: bool,
    dataset_prepare_payload: Optional[Dict[str, Any]] = None,
    preflight_summary: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    step_payloads = {"preflight": preflight_payload}
    if dataset_prepare_payload is not None:
        step_payloads["dataset_prepare"] = dataset_prepare_payload

    payload_failures = []
    for name, payload in step_payloads.items():
        reasons = _payload_failure_reasons(payload)
        if reasons:
            payload_failures.append({"name": name, "failure_reasons": reasons})

    not_run_step_names = _compute_not_run_step_names(
        selected_step_names=selected_step_names,
        failed_step_names=failed_step_names,
        completed_script_names=[],
        include_dataset=include_dataset,
    )

    dataset_prepare_summary = dataset_prepare_payload
    if dataset_prepare_summary is None:
        dataset_prepare_summary = _build_dataset_prepare_placeholder(
            include_dataset=include_dataset,
            blocked_by="preflight",
        )

    master_summary = {
        "overall_status": "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": {},
        "timestamp": taipei_now().isoformat(),
        "git_commit": resolve_git_commit(),
        "scripts": [],
        "selected_steps": selected_step_names,
        "failures": len(failed_step_names),
        "preflight": preflight_payload,
        "dataset_prepare": dataset_prepare_summary,
        "payload_failures": payload_failures,
        "bundle_mode": bundle_mode,
        "failed_step_names": failed_step_names,
        "not_run_step_names": not_run_step_names,
        "suggested_rerun_command": "",
    }

    write_json(run_dir / "master_summary.json", master_summary)
    artifacts_manifest_path = _write_stable_artifacts_manifest(run_dir)

    bundle_paths = [run_dir / "master_summary.json", run_dir / "preflight_summary.json", run_dir / "preflight_summary.txt"]
    if dataset_prepare_payload is not None:
        bundle_paths.extend([run_dir / "dataset_prepare_summary.json", run_dir / "dataset_prepare_summary.txt"])
    bundle_paths.append(artifacts_manifest_path)

    master_summary["bundle_entries"] = _build_bundle_entries(run_dir, bundle_paths)
    write_json(run_dir / "master_summary.json", master_summary)
    _write_stable_artifacts_manifest(run_dir)

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
        "bundle_mode": bundle_mode,
        "bundle_entries": _build_bundle_entries(run_dir, bundle_paths),
        "scripts": [],
        "step_payloads": step_payloads,
        "failures": len(failed_step_names),
        "failed_step_names": failed_step_names,
        "not_run_step_names": not_run_step_names,
        "retention": {
            "removed_count": retention.get("removed_count", 0),
            "removed_bytes": retention.get("removed_bytes", 0),
        },
        "major_index": major_total,
        "major_total": major_total,
        "failed_at_major_index": major_index,
        "preflight": preflight_summary or preflight_payload,
        "selected_steps": selected_step_names,
        "suggested_rerun_command": "",
    }
    _emit_progress(progress_callback, "done", result)
    return result


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


def _safe_format_preflight_summary(payload: Dict[str, Any]) -> str:
    try:
        return format_preflight_summary(payload)
    except Exception as exc:
        lines = [
            f"status          : {payload.get('status', 'FAIL')}",
            f"python          : {payload.get('python_executable', sys.executable)}",
            f"duration_sec    : {payload.get('duration_sec', 0.0)}",
            f"runtime_error   : {payload.get('runtime_error', '')}",
            f"formatter_error : {type(exc).__name__}: {exc}",
        ]
        failed_packages = payload.get("failed_packages", [])
        if failed_packages:
            lines.append("failed_packages : " + ", ".join(str(item) for item in failed_packages))
        return "\n".join(lines)


def _run_preflight(run_dir: Path, *, selected_step_names: List[str]) -> Dict[str, Any]:
    started = time.time()
    payload = run_preflight(selected_steps=selected_step_names)
    duration_sec = round(time.time() - started, 3)
    payload_with_duration = {**payload, "duration_sec": duration_sec}
    write_json(run_dir / "preflight_summary.json", payload_with_duration)
    write_text(run_dir / "preflight_summary.txt", _safe_format_preflight_summary(payload_with_duration) + "\n")
    return {
        "status": payload_with_duration["status"],
        "duration_sec": duration_sec,
        "failed_packages": payload_with_duration.get("failed_packages", []),
        "python_executable": payload_with_duration.get("python_executable", sys.executable),
        "summary_file": "preflight_summary.json",
        "summary_text_file": "preflight_summary.txt",
    }


def _write_dataset_prepare_summary(run_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(payload)
    duration_sec = round(float(summary.get("duration_sec", 0.0) or 0.0), 3)
    summary["duration_sec"] = duration_sec
    write_json(run_dir / "dataset_prepare_summary.json", summary)
    if summary.get("status") == "PASS":
        lines = [
            f"status      : {summary.get('status', 'PASS')}",
            f"duration_sec: {duration_sec}",
            f"dataset_dir : {summary.get('dataset_dir', '')}",
            f"source      : {summary.get('source', '')}",
            f"csv_count   : {summary.get('csv_count', 0)}",
            f"total_bytes : {summary.get('csv_total_bytes', 0)}",
            f"reused      : {summary.get('reused_existing', False)}",
        ]
        if summary.get('csv_members_sha256'):
            lines.append(f"members_sha : {summary.get('csv_members_sha256', '')}")
        if summary.get('csv_content_sha256'):
            lines.append(f"content_sha : {summary.get('csv_content_sha256', '')}")
        if "extracted_files" in summary:
            lines.append(f"extracted   : {summary.get('extracted_files', 0)}")
        write_text(run_dir / "dataset_prepare_summary.txt", "\n".join(lines) + "\n")
    else:
        write_text(
            run_dir / "dataset_prepare_summary.txt",
            "\n".join([
                f"status      : {summary.get('status', 'FAIL')}",
                f"duration_sec: {duration_sec}",
                f"error_type  : {summary.get('error_type', '')}",
                f"error_msg   : {summary.get('error_message', '')}",
            ]) + "\n",
        )
    return summary


def _write_manifest_failure_bundle(
    *,
    run_dir: Path,
    selected_step_names: List[str],
    include_dataset: bool,
    manifest_error: Exception,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    fallback_manifest = dict(MANIFEST_DEFAULTS)
    major_total = _major_step_total(selected_steps=selected_step_names, include_dataset=include_dataset)
    error_payload = {
        "status": "FAIL",
        "error_type": type(manifest_error).__name__,
        "error_message": str(manifest_error),
        "summary_file": "manifest_summary.json",
        "summary_text_file": "manifest_summary.txt",
    }
    write_json(run_dir / "manifest_summary.json", error_payload)
    write_text(
        run_dir / "manifest_summary.txt",
        "\n".join([
            "status      : FAIL",
            f"error_type  : {error_payload['error_type']}",
            f"error_msg   : {error_payload['error_message']}",
        ]) + "\n",
    )

    not_run_step_names = _compute_not_run_step_names(
        selected_step_names=selected_step_names,
        failed_step_names=["manifest"],
        completed_script_names=[],
        include_dataset=include_dataset,
    )

    master_summary = {
        "overall_status": "FAIL",
        "dataset": fallback_manifest["dataset"],
        "dataset_info": {},
        "timestamp": taipei_now().isoformat(),
        "git_commit": resolve_git_commit(),
        "scripts": [],
        "selected_steps": selected_step_names,
        "failures": 1,
        "preflight": {"status": "NOT_RUN", "blocked_by": "manifest"},
        "dataset_prepare": _build_dataset_prepare_placeholder(
            include_dataset=include_dataset,
            blocked_by="manifest",
        ),
        "payload_failures": [],
        "bundle_mode": "manifest_failed",
        "failed_step_names": ["manifest"],
        "not_run_step_names": not_run_step_names,
        "suggested_rerun_command": "",
        "manifest": error_payload,
    }
    write_json(run_dir / "master_summary.json", master_summary)
    artifacts_manifest_path = _write_stable_artifacts_manifest(run_dir)
    bundle_paths = [
        run_dir / "master_summary.json",
        run_dir / "manifest_summary.json",
        run_dir / "manifest_summary.txt",
        artifacts_manifest_path,
    ]
    master_summary["bundle_entries"] = _build_bundle_entries(run_dir, bundle_paths)
    write_json(run_dir / "master_summary.json", master_summary)
    _write_stable_artifacts_manifest(run_dir)

    bundle_path = build_bundle_zip(run_dir, str(fallback_manifest["bundle_name"]), include_paths=bundle_paths)
    archived_bundle = archive_bundle_history(bundle_path)
    root_bundle_copy = publish_root_bundle_copy(archived_bundle)
    retention = _apply_output_retention(fallback_manifest)

    result = {
        "overall_status": "FAIL",
        "dataset": fallback_manifest["dataset"],
        "bundle": str(root_bundle_copy),
        "archived_bundle": str(archived_bundle),
        "root_bundle_copy": str(root_bundle_copy),
        "bundle_mode": "manifest_failed",
        "bundle_entries": _build_bundle_entries(run_dir, bundle_paths),
        "scripts": [],
        "step_payloads": {"manifest": error_payload},
        "failures": 1,
        "failed_step_names": ["manifest"],
        "not_run_step_names": not_run_step_names,
        "retention": {
            "removed_count": retention.get("removed_count", 0),
            "removed_bytes": retention.get("removed_bytes", 0),
        },
        "major_index": major_total,
        "major_total": major_total,
        "failed_at_major_index": 0,
        "selected_steps": selected_step_names,
        "suggested_rerun_command": "",
    }
    _emit_progress(progress_callback, "done", result)
    return result


def _safe_write_dataset_prepare_summary(run_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _write_dataset_prepare_summary(run_dir, payload)
    except Exception as exc:
        duration_sec = round(float(payload.get("duration_sec", 0.0) or 0.0), 3)
        fallback_summary = {
            "status": "FAIL",
            "duration_sec": duration_sec,
            "error_type": payload.get("error_type", type(exc).__name__),
            "error_message": payload.get("error_message", str(exc)),
            "summary_write_error": f"{type(exc).__name__}: {exc}",
        }
        try:
            write_json(run_dir / "dataset_prepare_summary.json", fallback_summary)
        except OSError:
            pass
        try:
            write_text(
                run_dir / "dataset_prepare_summary.txt",
                "\n".join([
                    f"status      : {fallback_summary['status']}",
                    f"duration_sec: {duration_sec}",
                    f"error_type  : {fallback_summary['error_type']}",
                    f"error_msg   : {fallback_summary['error_message']}",
                    f"write_error : {fallback_summary['summary_write_error']}",
                ]) + "\n",
            )
        except OSError:
            pass
        return fallback_summary


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
    error_type = ""
    error_message = ""

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {sys.executable} {relative_script}\n")
        log_file.flush()
        try:
            try:
                process = subprocess.Popen(
                    [sys.executable, *shlex.split(relative_script, posix=True)],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                returncode = 127
                log_file.write(f"[test_suite] launch_failed: {error_type}: {error_message}\n")
                log_file.flush()
                process = None

            while process is not None:
                returncode = process.poll()
                elapsed_sec = round(time.time() - started, 1)
                if returncode is not None:
                    break
                if elapsed_sec >= timeout_sec:
                    timed_out = True
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired as exc:
                        error_type = type(exc).__name__
                        error_message = str(exc)
                    returncode = 124
                    log_file.write(f"\n[test_suite] TIMEOUT after {timeout_sec} seconds\n")
                    if error_type:
                        log_file.write(f"[test_suite] wait_after_kill_failed: {error_type}: {error_message}\n")
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
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    if not error_type:
                        error_type = type(exc).__name__
                        error_message = str(exc)
                    log_file.write(f"[test_suite] final_wait_failed: {type(exc).__name__}: {exc}\n")
                    log_file.flush()

    duration_sec = round(time.time() - started, 3)
    return {
        "returncode": returncode,
        "duration_sec": duration_sec,
        "timed_out": timed_out,
        "log_path": str(log_path),
        "error_type": error_type,
        "error_message": error_message,
    }


def execute_all(

    progress_callback: Optional[ProgressCallback] = None,
    selected_steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    selected_step_names = _normalize_selected_steps(selected_steps)
    include_dataset = any(name in DATASET_REQUIRED_STEPS for name in selected_step_names)
    run_dir = create_staging_run_dir()
    try:
        manifest = load_manifest()
    except Exception as exc:
        return _write_manifest_failure_bundle(
            run_dir=run_dir,
            selected_step_names=selected_step_names,
            include_dataset=include_dataset,
            manifest_error=exc,
            progress_callback=progress_callback,
        )
    shared_env = build_python_env({LOCAL_REGRESSION_RUN_DIR_ENV: str(run_dir)})
    major_total = _major_step_total(selected_steps=selected_step_names, include_dataset=include_dataset)

    try:
        _emit_progress(progress_callback, "step_start", {
            "name": "preflight",
            "major_index": 1,
            "major_total": major_total,
            "timeout_sec": 0,
        })
        preflight_started = time.time()
        try:
            preflight_summary = _run_inline_step_with_progress(
                name="preflight",
                major_index=1,
                major_total=major_total,
                progress_callback=progress_callback,
                func=lambda: _run_preflight(run_dir, selected_step_names=selected_step_names),
            )
        except Exception as exc:
            preflight_duration_sec = round(time.time() - preflight_started, 3)
            preflight_runtime_error = f"{type(exc).__name__}: {exc}"
            preflight_payload = {
                "status": "FAIL",
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "requirements_path": str(REQUIREMENTS_PATH),
                "checked_packages": [],
                "failed_packages": [],
                "checks": [],
                "runtime_error": preflight_runtime_error,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "duration_sec": preflight_duration_sec,
            }
            write_json(run_dir / "preflight_summary.json", preflight_payload)
            write_text(
                run_dir / "preflight_summary.txt",
                _safe_format_preflight_summary(preflight_payload) + f"\nruntime_error   : {preflight_runtime_error}\n",
            )
            preflight_summary = {
                "status": "FAIL",
                "duration_sec": preflight_duration_sec,
                "failed_packages": [],
                "python_executable": sys.executable,
                "summary_file": "preflight_summary.json",
                "summary_text_file": "preflight_summary.txt",
                "runtime_error": preflight_runtime_error,
            }

        _emit_progress(progress_callback, "step_finish", {
            "name": "preflight",
            "status": preflight_summary["status"],
            "duration_sec": preflight_summary["duration_sec"],
            "major_index": 1,
            "major_total": major_total,
        })

        if preflight_summary["status"] != "PASS":
            preflight_payload = read_json_if_exists(run_dir / "preflight_summary.json")
            return _finalize_early_failure(
                run_dir=run_dir,
                manifest=manifest,
                selected_step_names=selected_step_names,
                major_index=1,
                major_total=major_total,
                bundle_mode="preflight_failed",
                failed_step_names=["preflight"],
                preflight_payload=preflight_payload,
                include_dataset=include_dataset,
                preflight_summary=preflight_summary,
                progress_callback=progress_callback,
            )

        dataset_info: Dict[str, Any] = {}
        next_major_index = 2
        if include_dataset:
            _emit_progress(progress_callback, "step_start", {
                "name": "dataset_prepare",
                "major_index": next_major_index,
                "major_total": major_total,
                "timeout_sec": 0,
            })
            dataset_prepare_started = time.time()

            def _prepare_dataset() -> Dict[str, Any]:
                dataset_info_local = ensure_reduced_dataset()
                return _safe_write_dataset_prepare_summary(
                    run_dir,
                    {
                        "status": "PASS",
                        "duration_sec": round(time.time() - dataset_prepare_started, 3),
                        **dataset_info_local,
                    },
                )

            try:
                dataset_prepare_summary = _run_inline_step_with_progress(
                    name="dataset_prepare",
                    major_index=next_major_index,
                    major_total=major_total,
                    progress_callback=progress_callback,
                    func=_prepare_dataset,
                )
                dataset_info = {
                    key: dataset_prepare_summary[key]
                    for key in DATASET_INFO_KEYS
                    if key in dataset_prepare_summary
                }
            except Exception as exc:
                dataset_prepare_summary = _safe_write_dataset_prepare_summary(
                    run_dir,
                    {
                        "status": "FAIL",
                        "duration_sec": round(time.time() - dataset_prepare_started, 3),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

            if dataset_prepare_summary["status"] != "PASS":
                _emit_progress(progress_callback, "step_finish", {
                    "name": "dataset_prepare",
                    "status": dataset_prepare_summary["status"],
                    "duration_sec": dataset_prepare_summary["duration_sec"],
                    "major_index": next_major_index,
                    "major_total": major_total,
                })
                preflight_payload = read_json_if_exists(run_dir / "preflight_summary.json")
                return _finalize_early_failure(
                    run_dir=run_dir,
                    manifest=manifest,
                    selected_step_names=selected_step_names,
                    major_index=next_major_index,
                    major_total=major_total,
                    bundle_mode="dataset_prepare_failed",
                    failed_step_names=["dataset_prepare"],
                    preflight_payload=preflight_payload,
                    include_dataset=include_dataset,
                    dataset_prepare_payload=dataset_prepare_summary,
                    preflight_summary=preflight_summary,
                    progress_callback=progress_callback,
                )

            _emit_progress(progress_callback, "step_finish", {
                "name": "dataset_prepare",
                "status": dataset_prepare_summary["status"],
                "duration_sec": dataset_prepare_summary["duration_sec"],
                "major_index": next_major_index,
                "major_total": major_total,
            })
            next_major_index += 1

        selected_script_order = [item for item in SCRIPT_ORDER if item[0] in selected_step_names]
        script_summaries: List[Dict[str, Any]] = []
        overall_ok = True
        for script_offset, (name, relative_script, summary_name) in enumerate(selected_script_order, start=next_major_index):
            timeout_sec = _resolve_step_timeout(name, manifest)
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
            summary_exists = summary_path.exists()
            summary_read_error = ""
            try:
                summary_payload = read_json_if_exists(summary_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                summary_payload = {}
                summary_read_error = f"{type(exc).__name__}: {exc}"
            reported_status = summary_payload.get("status", "") if summary_payload else ""
            effective_status = "PASS"
            failure_reasons: List[str] = []
            if not summary_exists:
                effective_status = "FAIL"
                failure_reasons.append("missing_summary_file")
            if summary_read_error:
                effective_status = "FAIL"
                failure_reasons.append("summary_unreadable")
            if run_result["timed_out"]:
                effective_status = "FAIL"
                failure_reasons.append("timed_out")
            if run_result["returncode"] != 0:
                effective_status = "FAIL"
                failure_reasons.append(f"returncode={run_result['returncode']}")
            if reported_status != "PASS":
                effective_status = "FAIL"
                failure_reasons.append(f"reported_status={reported_status or 'missing'}")
                failed_steps = [str(item).strip() for item in summary_payload.get("failed_steps", []) if str(item).strip()]
                if failed_steps:
                    failure_reasons.append("failed_steps=" + ",".join(failed_steps))
                failures = [str(item).strip() for item in summary_payload.get("failures", []) if str(item).strip()]
                if failures:
                    failure_reasons.append("summary_failures=" + ",".join(failures))
                runtime_error = str(summary_payload.get("runtime_error", "") or summary_payload.get("error_message", "") or "").strip()
                if runtime_error:
                    failure_reasons.append(f"summary_error={runtime_error}")
                if "fail_count" in summary_payload:
                    failure_reasons.append(_format_optional_int_detail("fail_count", summary_payload.get("fail_count", 0)))
                if "failed_count" in summary_payload:
                    failure_reasons.append(_format_optional_int_detail("failed_count", summary_payload.get("failed_count", 0)))
            if run_result.get("error_type"):
                effective_status = "FAIL"
                failure_reasons.append(f"launch_error={run_result['error_type']}")
            script_summary = {
                "name": name,
                "status": effective_status,
                "reported_status": reported_status or "missing",
                "returncode": run_result["returncode"],
                "duration_sec": run_result["duration_sec"],
                "summary_file": summary_path.name,
                "summary_exists": summary_exists,
                "summary_read_error": summary_read_error,
                "timed_out": run_result["timed_out"],
                "error_type": run_result.get("error_type", ""),
                "error_message": run_result.get("error_message", ""),
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
        payload_failures = []
        for required_name in ["preflight", *(["dataset_prepare"] if include_dataset else [])]:
            reasons = _payload_failure_reasons(step_payloads.get(required_name, {}))
            if reasons:
                payload_failures.append({"name": required_name, "failure_reasons": reasons})

        completed_script_names = [item["name"] for item in script_summaries]
        failed_script_names = [item["name"] for item in script_summaries if item["status"] != "PASS"]
        failed_step_names = failed_script_names + [item["name"] for item in payload_failures]
        not_run_step_names = _compute_not_run_step_names(
            selected_step_names=selected_step_names,
            failed_step_names=failed_step_names,
            completed_script_names=completed_script_names,
            include_dataset=include_dataset,
        )
        overall_status = "PASS" if (overall_ok and not payload_failures) else "FAIL"
        suggested_rerun_command = _suggest_rerun_command(failed_script_names)
        master_summary = {
            "overall_status": overall_status,
            "dataset": manifest["dataset"],
            "dataset_info": dataset_info,
            "timestamp": taipei_now().isoformat(),
            "git_commit": resolve_git_commit(),
            "scripts": script_summaries,
            "selected_steps": selected_step_names,
            "failures": len(failed_step_names),
            "preflight": step_payloads["preflight"],
            "dataset_prepare": step_payloads["dataset_prepare"],
            "payload_failures": payload_failures,
            "not_run_step_names": not_run_step_names,
        }

        write_json(run_dir / "master_summary.json", master_summary)
        artifacts_manifest_path = _write_stable_artifacts_manifest(run_dir)

        bundle_mode = "minimum_set" if overall_status == "PASS" else "debug_bundle"
        bundle_paths = select_bundle_paths(run_dir, overall_ok=overall_status == "PASS")
        master_summary["bundle_mode"] = bundle_mode
        master_summary["bundle_entries"] = _build_bundle_entries(run_dir, bundle_paths)
        master_summary["failed_step_names"] = failed_step_names
        master_summary["suggested_rerun_command"] = suggested_rerun_command
        write_json(run_dir / "master_summary.json", master_summary)
        artifacts_manifest_path = _write_stable_artifacts_manifest(run_dir)
        if overall_ok and artifacts_manifest_path not in bundle_paths:
            bundle_paths.append(artifacts_manifest_path)
        master_summary["bundle_entries"] = _build_bundle_entries(run_dir, bundle_paths)
        write_json(run_dir / "master_summary.json", master_summary)
        _write_stable_artifacts_manifest(run_dir)

        bundle_path = build_bundle_zip(run_dir, str(manifest["bundle_name"]), include_paths=bundle_paths)
        archived_bundle = archive_bundle_history(bundle_path)
        root_bundle_copy = publish_root_bundle_copy(archived_bundle)
        retention = _apply_output_retention(manifest)

        result = {
            "overall_status": overall_status,
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
            "not_run_step_names": not_run_step_names,
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
        print(f"用法: python {program_name} [--only quick_gate,consistency,chain_checks,ml_smoke,meta_quality]")
        print("說明: 預設先跑完整 reduced regression（含 meta quality）；若完整入口已找出失敗步驟，可再用 --only 只重跑指定步驟。")
        return 0

    try:
        parsed = _parse_cli_args(args)
    except ValueError as exc:
        print(f"參數錯誤: {exc}", file=sys.stderr)
        program_name = resolve_cli_program_name(args, "tools/local_regression/run_all.py")
        print(f"用法: python {program_name} [--only quick_gate,consistency,chain_checks,ml_smoke,meta_quality]", file=sys.stderr)
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
