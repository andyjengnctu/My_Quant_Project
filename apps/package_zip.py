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

from core.runtime_utils import has_help_flag, get_taipei_now, resolve_cli_program_name, run_cli_entrypoint, validate_cli_args

HELP_DESCRIPTION = "清除 Python 快取、歸檔舊 package ZIP，並將目前 working tree 的 tracked/untracked 非忽略檔打成乾淨 ZIP；新 package 會包含 reduced dataset，舊 package 移入 arch/ 時會自動移除 reduced dataset 以節省空間；可選擇先 commit，再於打包後執行 test suite。"
EXCLUDED_DIR_NAMES = {"__pycache__", "arch"}
EXCLUDED_SUFFIXES = {".pyc"}
ARCHIVE_STRIP_DIR_PREFIXES = {
    PurePosixPath("data/tw_stock_data_vip_reduced"),
}
ROOT_BUNDLE_PREFIX = "to_chatgpt_bundle_"
COMMIT_MESSAGE_OPTION = "--commit-message"
RUN_TEST_SUITE_OPTION = "--run-test-suite"


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


def _run_python_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到可執行檔：{command[0]}") from exc


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


def _is_root_bundle_zip(zip_path: Path) -> bool:
    return zip_path.name.startswith(ROOT_BUNDLE_PREFIX)


def _should_strip_from_archived_zip(zip_member_name: str) -> bool:
    member_path = PurePosixPath(zip_member_name)
    for stripped_prefix in ARCHIVE_STRIP_DIR_PREFIXES:
        try:
            member_path.relative_to(stripped_prefix)
            return True
        except ValueError:
            continue
    return False


def _strip_reduced_data_from_archived_zip(zip_path: Path) -> int:
    temp_zip_path = zip_path.with_suffix(f"{zip_path.suffix}.tmp")
    removed_members = 0

    with zipfile.ZipFile(zip_path, mode="r") as source_zip:
        members = source_zip.infolist()
        kept_members = [member for member in members if not _should_strip_from_archived_zip(member.filename)]
        removed_members = len(members) - len(kept_members)
        if removed_members == 0:
            return 0

        with zipfile.ZipFile(temp_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target_zip:
            for member in kept_members:
                with source_zip.open(member, mode="r") as source_file:
                    data = source_file.read()
                target_zip.writestr(member, data)

    temp_zip_path.replace(zip_path)
    return removed_members


def _archive_existing_root_package_zips() -> tuple[int, int]:
    archive_dir = PROJECT_ROOT / "arch"
    archive_dir.mkdir(exist_ok=True)
    moved_count = 0
    stripped_member_count = 0

    for zip_path in sorted(PROJECT_ROOT.glob("*.zip")):
        if not zip_path.is_file() or _is_root_bundle_zip(zip_path):
            continue
        archived_zip_path = archive_dir / zip_path.name
        shutil.move(str(zip_path), archived_zip_path)
        stripped_member_count += _strip_reduced_data_from_archived_zip(archived_zip_path)
        moved_count += 1

    return moved_count, stripped_member_count


def _should_skip(relative_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in relative_path.parts):
        return True

    suffix = relative_path.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return True

    return False


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


def _parse_cli_args(argv=None):
    args = list(sys.argv if argv is None else argv)
    program_name = resolve_cli_program_name(args, "apps/package_zip.py")
    if has_help_flag(args):
        print(f"用法: python {program_name} [{COMMIT_MESSAGE_OPTION} <message>] [{RUN_TEST_SUITE_OPTION}]")
        print(f"說明: {HELP_DESCRIPTION}")
        print(f"選項: {COMMIT_MESSAGE_OPTION} 先 git add -A 並 commit；{RUN_TEST_SUITE_OPTION} 於打包後執行 python apps/test_suite.py")
        return {"help": True, "program_name": program_name, "commit_message": None, "run_test_suite": False}

    validate_cli_args(args, value_options=(COMMIT_MESSAGE_OPTION,), flag_options=(RUN_TEST_SUITE_OPTION,))
    parsed = {"help": False, "program_name": program_name, "commit_message": None, "run_test_suite": False}

    idx = 1
    while idx < len(args):
        raw_arg = str(args[idx]).strip()
        option_name, has_inline_value, inline_value = raw_arg.partition("=")
        if option_name == COMMIT_MESSAGE_OPTION:
            if has_inline_value:
                parsed["commit_message"] = inline_value.strip()
                idx += 1
                continue
            parsed["commit_message"] = str(args[idx + 1]).strip()
            idx += 2
            continue
        if option_name == RUN_TEST_SUITE_OPTION:
            parsed["run_test_suite"] = True
            idx += 1
            continue
        idx += 1

    return parsed


def _has_pending_changes() -> bool:
    status_output = _run_git("status", "--porcelain", "--untracked-files=all").stdout
    return status_output.strip() != ""


def _commit_all_changes(commit_message: str) -> tuple[bool, str | None]:
    if not _has_pending_changes():
        return False, None

    _run_git("add", "-A")
    commit_result = _run_git("commit", "-m", commit_message)
    commit_line = (commit_result.stdout or "").strip().splitlines()
    return True, commit_line[0] if commit_line else None


def _run_test_suite() -> None:
    command = [sys.executable, "apps/test_suite.py"]
    completed = _run_python_command(command)
    if completed.returncode != 0:
        raise RuntimeError(f"test_suite 執行失敗，exit code={completed.returncode}")


def main(argv=None) -> int:
    parsed = _parse_cli_args(argv)
    if parsed["help"]:
        return 0

    commit_summary = "skipped"
    commit_message = parsed.get("commit_message")
    if commit_message:
        committed, commit_headline = _commit_all_changes(commit_message)
        commit_summary = "created" if committed else "no_changes"
        if committed and commit_headline:
            print(f"[package_zip] commit={commit_headline}")
        elif not committed:
            print("[package_zip] commit=skip (working tree clean)")

    branch_name = _get_current_branch_name()
    branch_label = _sanitize_filename_component(branch_name)
    head_sha = _get_head_short_sha()
    removed_cache_dirs, removed_pyc_files = _remove_python_caches()
    moved_count, stripped_member_count = _archive_existing_root_package_zips()
    package_paths = _collect_package_paths()
    zip_path = _build_zip(branch_label, head_sha, package_paths)

    print(f"[package_zip] branch={branch_name} sha={head_sha}")
    print(f"[package_zip] removed __pycache__={removed_cache_dirs} *.pyc={removed_pyc_files}")
    print(f"[package_zip] archived old root zips={moved_count} stripped_reduced_members={stripped_member_count}")
    print(f"[package_zip] packaged files={len(package_paths)}")
    print(f"[package_zip] output={zip_path}")

    test_suite_summary = "not_run"
    if parsed.get("run_test_suite"):
        _run_test_suite()
        test_suite_summary = "pass"
        print("[package_zip] test_suite=pass")

    if not parsed.get("run_test_suite"):
        print(f"[package_zip] commit_status={commit_summary} test_suite={test_suite_summary}")
    return 0


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)
