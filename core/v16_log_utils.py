# core/v16_log_utils.py
import os
import time
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs")


def resolve_log_dir(log_dir=None):
    if not log_dir:
        return DEFAULT_LOG_DIR
    if os.path.isabs(log_dir):
        return log_dir
    return os.path.join(PROJECT_ROOT, log_dir)


def build_timestamped_log_path(prefix, log_dir=None, timestamp=None):
    resolved_log_dir = resolve_log_dir(log_dir)
    os.makedirs(resolved_log_dir, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(resolved_log_dir, f"{prefix}_{ts}.log")


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

    resolved_log_path = log_path
    if not os.path.isabs(resolved_log_path):
        resolved_log_path = os.path.join(PROJECT_ROOT, resolved_log_path)

    os.makedirs(os.path.dirname(resolved_log_path) or DEFAULT_LOG_DIR, exist_ok=True)
    with open(resolved_log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return resolved_log_path


# # (AI註: 防錯透明化 - 統一例外摘要格式，保留型別、訊息與 traceback 尾端，避免各腳本口徑分裂)
def format_exception_summary(exc, tail_lines=3):
    tb_lines = traceback.format_exc().strip().splitlines()
    tb_tail = " | ".join(tb_lines[-tail_lines:]) if tb_lines else ""
    msg = f"{type(exc).__name__}: {exc}"
    if tb_tail:
        return f"{msg} | Traceback: {tb_tail}"
    return msg
