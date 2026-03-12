# core/v16_log_utils.py
import os
import time
import traceback


def build_timestamped_log_path(prefix, log_dir="outputs", timestamp=None):
    os.makedirs(log_dir, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{prefix}_{ts}.log")


def write_issue_log(prefix, lines, log_dir="outputs", timestamp=None):
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

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return log_path

# # (AI註: 防錯透明化 - 統一例外摘要格式，保留型別、訊息與 traceback 尾端，避免各腳本口徑分裂)
def format_exception_summary(exc, tail_lines=3):
    tb_lines = traceback.format_exc().strip().splitlines()
    tb_tail = " | ".join(tb_lines[-tail_lines:]) if tb_lines else ""
    msg = f"{type(exc).__name__}: {exc}"
    if tb_tail:
        return f"{msg} | Traceback: {tb_tail}"
    return msg
