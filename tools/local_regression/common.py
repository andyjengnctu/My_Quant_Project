from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_REGRESSION_DIR = PROJECT_ROOT / "tools" / "local_regression"
DEFAULT_MANIFEST_PATH = LOCAL_REGRESSION_DIR / "manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "local_regression"
REDUCED_DATASET_DIR = PROJECT_ROOT / "data" / "tw_stock_data_vip_reduced"
MANIFEST_DEFAULTS: Dict[str, Any] = {
    "benchmark_ticker": "0050",
    "bundle_name": "to_chatgpt_bundle.zip",
    "dataset": "reduced",
    "detail_tools_keep_last_n": 5,
    "detail_tools_max_age_days": 14,
    "local_regression_keep_last_n": 20,
    "local_regression_max_age_days": 30,
    "ml_smoke_timeout_sec": 300,
    "ml_smoke_trials": 1,
    "performance_quick_gate_max_sec": 45,
    "performance_consistency_max_sec": 60,
    "performance_chain_checks_max_sec": 60,
    "performance_ml_smoke_max_sec": 45,
    "performance_meta_quality_max_sec": 30,
    "performance_total_max_sec": 180,
    "performance_optimizer_trial_avg_max_sec": 15,
    "portfolio_enable_rotation": False,
    "portfolio_max_positions": 10,
    "portfolio_start_year": 2015,
    "retention_enabled": True,
    "subprocess_timeout_sec": 300,
    "summary_tools_keep_last_n": 10,
    "summary_tools_max_age_days": 30,
}
MANIFEST_INT_FIELDS = {
    "detail_tools_keep_last_n",
    "detail_tools_max_age_days",
    "local_regression_keep_last_n",
    "local_regression_max_age_days",
    "ml_smoke_timeout_sec",
    "ml_smoke_trials",
    "performance_quick_gate_max_sec",
    "performance_consistency_max_sec",
    "performance_chain_checks_max_sec",
    "performance_ml_smoke_max_sec",
    "performance_meta_quality_max_sec",
    "performance_total_max_sec",
    "performance_optimizer_trial_avg_max_sec",
    "portfolio_max_positions",
    "portfolio_start_year",
    "subprocess_timeout_sec",
    "summary_tools_keep_last_n",
    "summary_tools_max_age_days",
}
MANIFEST_BOOL_FIELDS = {
    "portfolio_enable_rotation",
    "retention_enabled",
}
MANIFEST_STR_FIELDS = {
    "benchmark_ticker",
    "bundle_name",
    "dataset",
}



class LocalRegressionError(RuntimeError):
    pass


_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _contains_any_path_separator(value: str) -> bool:
    return ("/" in value) or ("\\" in value)


def _is_windows_absolute_path(raw_value: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(raw_value)) or raw_value.startswith("\\\\")


def _split_cross_platform_parts(raw_value: str) -> tuple[str, tuple[str, ...]]:
    normalized = raw_value.replace("\\", "/")
    return normalized, PurePosixPath(normalized).parts


def _normalize_bundle_name(bundle_name: Any) -> str:
    raw_value = os.fspath(bundle_name).strip()
    if raw_value == "":
        raise LocalRegressionError("manifest 欄位 bundle_name 不可空白")
    if _contains_any_path_separator(raw_value):
        raise LocalRegressionError("manifest 欄位 bundle_name 只能是單一 zip 檔名，不可包含路徑分隔")

    bundle_path = Path(raw_value)
    if bundle_path.is_absolute():
        raise LocalRegressionError("manifest 欄位 bundle_name 不可為絕對路徑")

    parts = bundle_path.parts
    if len(parts) != 1 or parts[0] in {"", ".", ".."}:
        raise LocalRegressionError("manifest 欄位 bundle_name 只能是單一 zip 檔名")

    normalized = parts[0]
    if not normalized.lower().endswith(".zip"):
        raise LocalRegressionError("manifest 欄位 bundle_name 需要 .zip 副檔名")
    return normalized



def _resolve_local_regression_run_dir_from_env(raw_value: str) -> Path:
    normalized = os.fspath(raw_value).strip()
    if normalized == "":
        raise LocalRegressionError("V16_LOCAL_REGRESSION_RUN_DIR 不可空白")

    normalized_value, normalized_parts = _split_cross_platform_parts(normalized)
    path_obj = Path(normalized)
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
    elif _is_windows_absolute_path(normalized):
        raise LocalRegressionError(f"V16_LOCAL_REGRESSION_RUN_DIR 必須落在 {(OUTPUT_ROOT / '_staging').resolve()}")
    else:
        if any(part in {"", ".", ".."} for part in normalized_parts):
            raise LocalRegressionError("V16_LOCAL_REGRESSION_RUN_DIR 不可包含 . 或 ..")
        resolved = (PROJECT_ROOT / normalized_value).resolve()

    allowed_root = (OUTPUT_ROOT / "_staging").resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise LocalRegressionError(f"V16_LOCAL_REGRESSION_RUN_DIR 必須落在 {allowed_root}") from exc

    ensure_dir(resolved)
    return resolved

def taipei_now() -> datetime:
    return datetime.now(TAIPEI_TZ)


def timestamp_text() -> str:
    return taipei_now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    manifest_path = DEFAULT_MANIFEST_PATH if path is None else Path(path)
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise LocalRegressionError(f"manifest 根層必須是 object/dict，收到: {type(payload).__name__}")

    unknown_keys = sorted(set(payload) - set(MANIFEST_DEFAULTS))
    if unknown_keys:
        raise LocalRegressionError(f"manifest 含未知欄位: {unknown_keys}")

    merged: Dict[str, Any] = {**MANIFEST_DEFAULTS, **payload}
    for field_name in sorted(MANIFEST_INT_FIELDS):
        raw_value = merged[field_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise LocalRegressionError(f"manifest 欄位 {field_name} 需要 int，收到: {raw_value!r}")
        if raw_value < 0:
            raise LocalRegressionError(f"manifest 欄位 {field_name} 需要 >= 0，收到: {raw_value!r}")

    for field_name in sorted(MANIFEST_BOOL_FIELDS):
        raw_value = merged[field_name]
        if not isinstance(raw_value, bool):
            raise LocalRegressionError(f"manifest 欄位 {field_name} 需要 bool，收到: {raw_value!r}")

    for field_name in sorted(MANIFEST_STR_FIELDS):
        raw_value = str(merged[field_name]).strip()
        if raw_value == "":
            raise LocalRegressionError(f"manifest 欄位 {field_name} 不可空白")
        merged[field_name] = raw_value

    merged["bundle_name"] = _normalize_bundle_name(merged["bundle_name"])
    dataset = merged["dataset"].lower()
    if dataset != "reduced":
        raise LocalRegressionError(f"local regression 只支援 reduced，收到: {dataset}")
    merged["dataset"] = dataset
    merged["manifest_path"] = str(manifest_path)
    return merged


def build_python_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("V16_DATASET_PROFILE", "reduced")
    env.setdefault("V16_VALIDATE_DATASET", "reduced")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return env


def run_command(
    args: List[str],
    *,
    input_text: Optional[str] = None,
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    started = time.time()
    cmd_text = " ".join(args)
    try:
        proc = subprocess.run(
            args,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(PROJECT_ROOT if cwd is None else cwd),
            env=build_python_env(env),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
        stderr = (stderr + ("\n" if stderr else "") + f"TimeoutExpired: command exceeded {timeout} seconds").strip()
        return {
            "args": args,
            "cmd": cmd_text,
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr,
            "duration_sec": round(elapsed, 3),
            "timed_out": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timeout_sec": timeout,
        }
    except OSError as exc:
        elapsed = time.time() - started
        return {
            "args": args,
            "cmd": cmd_text,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_sec": round(elapsed, 3),
            "timed_out": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timeout_sec": timeout,
        }

    elapsed = time.time() - started
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    return {
        "args": args,
        "cmd": cmd_text,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_sec": round(elapsed, 3),
        "timed_out": False,
        "error_type": "",
        "error_message": "",
        "timeout_sec": timeout,
    }


def summarize_result(name: str, ok: bool, *, detail: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail}
    if extra:
        payload.update(extra)
    return payload


def resolve_run_dir(script_name: str) -> Path:
    env_run_dir = os.environ.get("V16_LOCAL_REGRESSION_RUN_DIR", "").strip()
    if env_run_dir:
        return _resolve_local_regression_run_dir_from_env(env_run_dir)
    run_dir = OUTPUT_ROOT / "_staging" / f"{timestamp_text()}_{script_name}_{uuid.uuid4().hex[:8]}"
    ensure_dir(run_dir)
    return run_dir


def create_staging_run_dir(prefix: str = "test_suite") -> Path:
    run_dir = OUTPUT_ROOT / "_staging" / f"{timestamp_text()}_{prefix}_{uuid.uuid4().hex[:8]}"
    ensure_dir(run_dir)
    return run_dir


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def build_bundle_zip(run_dir: Path, bundle_name: str, *, include_paths: Optional[List[Path]] = None) -> Path:
    normalized_bundle_name = _normalize_bundle_name(bundle_name)
    bundle_path = run_dir / normalized_bundle_name
    if bundle_path.exists():
        bundle_path.unlink()

    if include_paths is None:
        include_iter = [path for path in sorted(run_dir.rglob("*")) if path.is_file() and path != bundle_path]
    else:
        include_iter = []
        seen = set()
        for raw_path in include_paths:
            path = raw_path.resolve()
            if not path.exists() or not path.is_file() or path == bundle_path.resolve():
                continue
            rel = path.relative_to(run_dir.resolve())
            if rel in seen:
                continue
            seen.add(rel)
            include_iter.append(path)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in include_iter:
            zf.write(file_path, file_path.relative_to(run_dir))
    return bundle_path


def select_bundle_paths(run_dir: Path, *, overall_ok: bool) -> List[Path]:
    if overall_ok:
        preferred = [
            run_dir / "master_summary.json",
            run_dir / "preflight_summary.json",
            run_dir / "preflight_summary.txt",
            run_dir / "dataset_prepare_summary.json",
            run_dir / "dataset_prepare_summary.txt",
            run_dir / "quick_gate_summary.json",
            run_dir / "validate_consistency_summary.json",
            run_dir / "chain_summary.json",
            run_dir / "chain_summary.csv",
            run_dir / "ml_smoke_summary.json",
            run_dir / "console_tail.txt",
            run_dir / "artifacts_manifest.json",
        ]
        return [path for path in preferred if path.exists()]
    return [path for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "to_chatgpt_bundle.zip"]


def cleanup_staging_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


def gather_recent_console_tail(run_dir: Path, limit_lines: int = 80) -> str:
    chunks: List[str] = []
    for log_path in sorted(run_dir.glob("*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        chunks.append(f"===== {log_path.name} =====\n" + "\n".join(lines[-limit_lines:]))
    return "\n\n".join(chunks).strip()


def resolve_git_commit() -> str:
    try:
        result = run_command(["git", "rev-parse", "--short", "HEAD"], timeout=30)
        if result["returncode"] == 0:
            return result["stdout"].strip()
        err_text = result["stderr"].strip() or result["stdout"].strip() or f"returncode={result['returncode']}"
        print(
            f"⚠️ git rev-parse 失敗，git_commit 將標記為 unknown: {err_text}",
            file=sys.stderr,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"⚠️ git rev-parse 失敗，git_commit 將標記為 unknown: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    return "unknown"


def build_artifacts_manifest(run_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_dir():
            continue
        artifacts.append({
            "relative_path": str(path.relative_to(run_dir)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
        })
    return {"artifact_count": len(artifacts), "artifacts": artifacts}


def _current_dataset_csv_members() -> List[str]:
    if not REDUCED_DATASET_DIR.is_dir():
        return []
    return sorted(
        str(path.relative_to(REDUCED_DATASET_DIR)).replace("\\", "/")
        for path in REDUCED_DATASET_DIR.rglob("*.csv")
    )


def ensure_reduced_dataset() -> Dict[str, Any]:
    actual_members = _current_dataset_csv_members()
    if not actual_members:
        raise FileNotFoundError(
            f"找不到 reduced dataset；請確認 {REDUCED_DATASET_DIR} 存在，且內含至少一個 CSV 檔案。"
        )

    return {
        "dataset_dir": str(REDUCED_DATASET_DIR),
        "source": "repo_data_dir",
        "csv_count": len(actual_members),
        "reused_existing": True,
    }


def archive_bundle_history(bundle_path: Path, *, prefix: str = "to_chatgpt_bundle") -> Path:
    archive_dir = ensure_dir(OUTPUT_ROOT)
    archive_name = f"{prefix}_{timestamp_text()}_{uuid.uuid4().hex[:8]}.zip"
    archive_path = archive_dir / archive_name
    shutil.move(str(bundle_path), str(archive_path))
    return archive_path


def publish_root_bundle_copy(bundle_path: Path, *, prefix: str = "to_chatgpt_bundle") -> Path:
    root_dir = PROJECT_ROOT
    latest_name = bundle_path.name if bundle_path.name.startswith(prefix) else f"{prefix}_{timestamp_text()}_{uuid.uuid4().hex[:8]}.zip"
    latest_path = root_dir / latest_name
    for old_path in sorted(root_dir.glob(f"{prefix}*.zip")):
        if old_path.resolve() == bundle_path.resolve():
            continue
        try:
            old_path.unlink()
        except FileNotFoundError:
            pass
    shutil.copy2(bundle_path, latest_path)
    return latest_path
