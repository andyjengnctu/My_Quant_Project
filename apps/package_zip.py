from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import get_taipei_now, parse_no_arg_cli, run_cli_entrypoint

HELP_DESCRIPTION = "清除 Python 快取、歸檔舊 ZIP，並將目前 working tree 的 tracked/untracked 非忽略檔打成乾淨 ZIP。"
EXCLUDED_DIR_NAMES = {"__pycache__", "arch"}
EXCLUDED_SUFFIXES = {".pyc"}


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("找不到 git，請先安裝並確認 git 已加入 PATH。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        command_text = " ".join(["git", *args])
        message = stderr if stderr else f"git 指令失敗：{command_text}"
        raise RuntimeError(message) from exc


def _sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip(".-") or "snapshot"


def _get_current_branch_name() -> str:
    branch_name = _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch_name == "HEAD":
        return "detached"
    return branch_name


def _get_head_short_sha() -> str:
    return _run_git("rev-parse", "--short", "HEAD").stdout.strip()


def _remove_python_caches() -> tuple[int, int]:
    removed_cache_dirs = 0
    removed_pyc_files = 0

    for cache_dir in sorted(PROJECT_ROOT.rglob("__pycache__")):
        if not cache_dir.is_dir():
            continue
        shutil.rmtree(cache_dir, ignore_errors=False)
        removed_cache_dirs += 1

    for pyc_file in sorted(PROJECT_ROOT.rglob("*.pyc")):
        if not pyc_file.is_file():
            continue
        pyc_file.unlink()
        removed_pyc_files += 1

    return removed_cache_dirs, removed_pyc_files


def _archive_existing_root_zips() -> int:
    archive_dir = PROJECT_ROOT / "arch"
    archive_dir.mkdir(exist_ok=True)
    moved_count = 0

    for zip_path in sorted(PROJECT_ROOT.glob("*.zip")):
        if not zip_path.is_file():
            continue
        shutil.move(str(zip_path), archive_dir / zip_path.name)
        moved_count += 1

    return moved_count


def _should_skip(relative_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in relative_path.parts):
        return True
    return relative_path.suffix.lower() in EXCLUDED_SUFFIXES


def _collect_package_paths() -> list[Path]:
    result = _run_git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    seen: set[str] = set()
    package_paths: list[Path] = []

    for raw_path in result.stdout.split("\0"):
        if raw_path == "":
            continue
        relative_path = Path(raw_path)
        normalized_key = PurePosixPath(*relative_path.parts).as_posix()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)

        absolute_path = PROJECT_ROOT / relative_path
        if not absolute_path.is_file():
            continue
        if _should_skip(relative_path):
            continue
        package_paths.append(relative_path)

    package_paths.sort(key=lambda path: PurePosixPath(*path.parts).as_posix())
    return package_paths


def _build_zip(branch_label: str, head_sha: str, package_paths: list[Path]) -> Path:
    if not package_paths:
        raise RuntimeError("找不到可打包檔案；請確認目前目錄是 git repo，且檔案未被全部忽略。")

    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    zip_path = PROJECT_ROOT / f"{branch_label}_{timestamp}_{head_sha}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        for relative_path in package_paths:
            absolute_path = PROJECT_ROOT / relative_path
            zip_file.write(absolute_path, arcname=PurePosixPath(*relative_path.parts).as_posix())
    return zip_path


def main(argv=None) -> int:
    parsed = parse_no_arg_cli(argv, "apps/package_zip.py", description=HELP_DESCRIPTION)
    if parsed["help"]:
        return 0

    branch_name = _get_current_branch_name()
    branch_label = _sanitize_filename_component(branch_name)
    head_sha = _get_head_short_sha()
    removed_cache_dirs, removed_pyc_files = _remove_python_caches()
    moved_count = _archive_existing_root_zips()
    package_paths = _collect_package_paths()
    zip_path = _build_zip(branch_label, head_sha, package_paths)

    print(f"[package_zip] branch={branch_name} sha={head_sha}")
    print(f"[package_zip] removed __pycache__={removed_cache_dirs} *.pyc={removed_pyc_files}")
    print(f"[package_zip] archived old root zips={moved_count}")
    print(f"[package_zip] packaged files={len(package_paths)}")
    print(f"[package_zip] output={zip_path}")
    return 0


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)
