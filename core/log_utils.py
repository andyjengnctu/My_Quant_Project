# core/log_utils.py
import os
import traceback
from pathlib import Path

from core.runtime_utils import get_taipei_now


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT_PATH = Path(PROJECT_ROOT).resolve()
OUTPUTS_ROOT_PATH = (PROJECT_ROOT_PATH / "outputs").resolve()


def _normalize_log_file_prefix(prefix):
    if prefix is None:
        raise ValueError("prefix 必填，且不可包含路徑分隔或 . / ..")

    raw_prefix = os.fspath(prefix).strip()
    if not raw_prefix:
        raise ValueError("prefix 必填，且不可包含路徑分隔或 . / ..")

    prefix_path = Path(raw_prefix)
    if prefix_path.is_absolute():
        raise ValueError("prefix 不可為絕對路徑")

    parts = prefix_path.parts
    if len(parts) != 1 or parts[0] in {"", ".", ".."}:
        raise ValueError("prefix 不可包含路徑分隔或 . / ..")

    return parts[0]


def _resolve_project_scoped_path(path_value, *, field_name, allow_file_name_only):
    if path_value is None:
        raise ValueError(f"{field_name} 必填，避免把檔案直接輸出到 outputs/ 根目錄")

    raw_value = os.fspath(path_value).strip()
    if not raw_value:
        raise ValueError(f"{field_name} 必填，避免把檔案直接輸出到 outputs/ 根目錄")

    path_obj = Path(raw_value)
    if path_obj.is_absolute():
        resolved_path = path_obj.resolve(strict=False)
    else:
        if any(part in {"", ".", ".."} for part in path_obj.parts):
            raise ValueError(f"{field_name} 不可包含 . 或 ..，避免寫出專案目錄")
        resolved_path = (PROJECT_ROOT_PATH / path_obj).resolve(strict=False)

    try:
        resolved_path.relative_to(PROJECT_ROOT_PATH)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必須落在專案目錄內，避免寫到專案外部") from exc

    if not allow_file_name_only and resolved_path.parent == PROJECT_ROOT_PATH:
        raise ValueError(f"{field_name} 必須包含目錄，避免把檔案直接輸出到專案根目錄")

    return str(resolved_path)


def resolve_log_dir(log_dir=None):
    return _resolve_project_scoped_path(log_dir, field_name="log_dir", allow_file_name_only=False)


def build_timestamped_log_path(prefix, log_dir=None, timestamp=None):
    resolved_log_dir = resolve_log_dir(log_dir)
    normalized_prefix = _normalize_log_file_prefix(prefix)
    os.makedirs(resolved_log_dir, exist_ok=True)
    ts = timestamp or get_taipei_now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(resolved_log_dir, f"{normalized_prefix}_{ts}.log")


def write_issue_log(prefix, lines, log_dir=None, timestamp=None):
    if not lines:
        return None

    log_path = build_timestamped_log_path(prefix, log_dir=log_dir, timestamp=timestamp)
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return log_path


def append_issue_log(log_path, lines):
    if not lines:
        return log_path

    resolved_log_path = _resolve_project_scoped_path(log_path, field_name="log_path", allow_file_name_only=False)
    resolved_log_path_obj = Path(resolved_log_path)
    if resolved_log_path_obj.parent == OUTPUTS_ROOT_PATH:
        raise ValueError("log_path 不可直接輸出到 outputs/ 根目錄")

    resolved_log_dir = os.path.dirname(resolved_log_path)

    os.makedirs(resolved_log_dir, exist_ok=True)
    with open(resolved_log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return resolved_log_path


# # (AI註: 例外摘要格式集中管理；user-facing 錯誤預設可關閉 traceback 尾端，避免 headless CLI 汙染)
def format_exception_summary(exc, tail_lines=3, *, include_traceback=True):
    tb_lines = traceback.format_exc().strip().splitlines()
    tb_tail = " | ".join(tb_lines[-tail_lines:]) if tb_lines else ""
    msg = f"{type(exc).__name__}: {exc}"
    if include_traceback and tb_tail:
        return f"{msg} | Traceback: {tb_tail}"
    return msg
