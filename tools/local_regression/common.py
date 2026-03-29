from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_REGRESSION_DIR = PROJECT_ROOT / "tools" / "local_regression"
DEFAULT_MANIFEST_PATH = LOCAL_REGRESSION_DIR / "manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "local_regression"
REDUCED_DATASET_DIR = PROJECT_ROOT / "data" / "tw_stock_data_vip_reduced"


class LocalRegressionError(RuntimeError):
    pass


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
    dataset = str(payload.get("dataset", "reduced")).strip().lower()
    if dataset != "reduced":
        raise LocalRegressionError(f"local regression 只支援 reduced，收到: {dataset}")
    payload.setdefault("portfolio_start_year", 2015)
    payload.setdefault("portfolio_max_positions", 10)
    payload.setdefault("portfolio_enable_rotation", False)
    payload.setdefault("benchmark_ticker", "0050")
    payload.setdefault("ml_smoke_trials", 1)
    payload.setdefault("ml_smoke_timeout_sec", 300)
    payload.setdefault("subprocess_timeout_sec", 300)
    payload.setdefault("bundle_name", "to_chatgpt_bundle.zip")
    payload["dataset"] = dataset
    payload["manifest_path"] = str(manifest_path)
    return payload


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
    elapsed = time.time() - started
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    return {
        "args": args,
        "cmd": " ".join(args),
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_sec": round(elapsed, 3),
    }


def summarize_result(name: str, ok: bool, *, detail: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail}
    if extra:
        payload.update(extra)
    return payload


def resolve_run_dir(script_name: str) -> Path:
    env_run_dir = os.environ.get("V16_LOCAL_REGRESSION_RUN_DIR", "").strip()
    if env_run_dir:
        run_dir = Path(env_run_dir).resolve()
        ensure_dir(run_dir)
        return run_dir
    run_dir = OUTPUT_ROOT / f"{timestamp_text()}_{script_name}"
    ensure_dir(run_dir)
    return run_dir


def finalize_latest(run_dir: Path) -> Path:
    latest_dir = OUTPUT_ROOT / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return latest_dir


def build_bundle_zip(run_dir: Path, bundle_name: str) -> Path:
    bundle_path = run_dir / bundle_name
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_dir() or file_path == bundle_path:
                continue
            zf.write(file_path, file_path.relative_to(run_dir))
    return bundle_path


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
    except Exception:
        pass
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


def ensure_reduced_dataset() -> Dict[str, Any]:
    if REDUCED_DATASET_DIR.is_dir() and any(REDUCED_DATASET_DIR.glob("*.csv")):
        return {
            "dataset_dir": str(REDUCED_DATASET_DIR),
            "source": "existing",
            "csv_count": sum(1 for _ in REDUCED_DATASET_DIR.glob("*.csv")),
        }

    candidate_paths: List[Path] = []
    env_zip = os.environ.get("V16_PROJECT_DATA_ZIP", "").strip()
    if env_zip:
        candidate_paths.append(Path(env_zip))
    candidate_paths.append(PROJECT_ROOT / "data.zip")
    candidate_paths.append(Path("/mnt/data/data.zip"))
    zip_path = next((path for path in candidate_paths if path.is_file()), None)
    if zip_path is None:
        raise FileNotFoundError(
            f"找不到 data.zip；請把本投資專案的 data.zip 放到 {PROJECT_ROOT / 'data.zip'}，或設定 V16_PROJECT_DATA_ZIP。"
        )

    ensure_dir(PROJECT_ROOT / "data")
    extracted_files = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and "tw_stock_data_vip_reduced/" in m.filename.replace("\\", "/")]
        if not members:
            raise FileNotFoundError(f"{zip_path} 內找不到 tw_stock_data_vip_reduced")
        for member in members:
            parts = [part for part in member.filename.replace("\\", "/").split("/") if part]
            dataset_idx = parts.index("tw_stock_data_vip_reduced")
            relative_parts = parts[dataset_idx:]
            target_path = PROJECT_ROOT / "data" / Path(*relative_parts)
            ensure_dir(target_path.parent)
            with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1

    if not REDUCED_DATASET_DIR.is_dir() or not any(REDUCED_DATASET_DIR.glob("*.csv")):
        raise FileNotFoundError(f"自動解壓後仍找不到 {REDUCED_DATASET_DIR}")

    return {
        "dataset_dir": str(REDUCED_DATASET_DIR),
        "source": str(zip_path),
        "extracted_files": extracted_files,
        "csv_count": sum(1 for _ in REDUCED_DATASET_DIR.glob("*.csv")),
    }


def _matches_retained_prefix(relative_path: str, retained_prefixes: Sequence[str]) -> bool:
    return any(relative_path == prefix or relative_path.startswith(prefix + "/") for prefix in retained_prefixes)


def collect_minimum_retention(*, run_dir: Path, overall_ok: bool, manifest: Dict[str, Any]) -> Tuple[Set[str], List[str]]:
    retained_files: Set[str] = {
        "master_summary.json",
        "artifacts_manifest.json",
        "console_tail.txt",
        "quick_gate_summary.json",
        "validate_consistency_summary.json",
        "chain_summary.json",
        "chain_summary.csv",
        "ml_smoke_summary.json",
    }
    retained_prefixes: List[str] = []

    if overall_ok:
        if bool(manifest.get("keep_logs_on_pass", False)):
            retained_files.update({path.name for path in run_dir.glob("*.log")})
        if bool(manifest.get("keep_chain_details_on_pass", False)):
            retained_prefixes.append("chain_details")
        if bool(manifest.get("keep_validate_full_reports_on_pass", False)):
            retained_files.update({path.name for path in run_dir.glob("consistency_full_scan_*.csv")})
            retained_files.update({path.name for path in run_dir.glob("consistency_failures_*.csv")})
            retained_files.update({path.name for path in run_dir.glob("consistency_issues_*.xlsx")})
    else:
        retained_files.update({path.name for path in run_dir.glob("*.log")})
        retained_files.update({path.name for path in run_dir.glob("consistency_full_scan_*.csv")})
        retained_files.update({path.name for path in run_dir.glob("consistency_failures_*.csv")})
        retained_files.update({path.name for path in run_dir.glob("consistency_issues_*.xlsx")})
        retained_prefixes.append("chain_details")

    return retained_files, retained_prefixes


def prune_run_dir(run_dir: Path, *, retained_files: Set[str], retained_prefixes: Sequence[str]) -> None:
    for path in sorted(run_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        relative_path = str(path.relative_to(run_dir)).replace("\\", "/")
        keep = relative_path in retained_files or _matches_retained_prefix(relative_path, retained_prefixes)
        if path.is_file() and not keep:
            path.unlink()
        elif path.is_dir() and not keep:
            try:
                path.rmdir()
            except OSError:
                pass


def publish_root_bundle_copy(bundle_path: Path, *, prefix: str = "to_chatgpt_bundle") -> Path:
    root_dir = PROJECT_ROOT
    latest_name = f"{prefix}_{timestamp_text()}_{uuid.uuid4().hex[:8]}.zip"
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
