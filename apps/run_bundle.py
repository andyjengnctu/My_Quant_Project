from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_LOG_PATH = "doc/BREAKOUT_QUALITY_EXPERIMENT_LOG.md"


def warn_if_experiment_log_staged() -> None:
    """Advisory only: this is a scientific-result log, not a patch changelog."""
    diff = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--", EXPERIMENT_LOG_PATH],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if diff.returncode != 0:
        print("[Experiment Log] 無法讀取 staged diff；跳過非阻擋式新增提醒。")
        return
    additions = [
        line[1:] for line in diff.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not additions:
        return
    new_headings = [
        line for line in additions
        if line.startswith("## ") or line.startswith("### ")
    ]
    engineering_only = [
        title for title in new_headings
        if re.search(r"工程|Formal|synthetic|Double.Check|ZIP|REUSE|修補|介面|打包|測試閉環", title, re.I)
    ]
    print(
        f"[Experiment Log] 本輪新增 {len(additions)} 行、{len(new_headings)} 個章節。"
        "僅允許已執行的研究數值、結論、負結果或科學有效性更正；"
        "工程修補／Formal Test／ZIP／REUSE 不可當研究日誌。"
        "（僅提醒，不阻擋 commit／Test Suite）"
    )
    for title in engineering_only[:3]:
        print(f"[Experiment Log] 疑似工程流水帳標題（請確認是否應移除）：{title}")



def run_cmd(cmd: list[str], step_name: str, *, check: bool = True, quiet_stdout: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL if quiet_stdout else None
    )
    if check and result.returncode != 0:
        raise SystemExit(f"{step_name} failed with exit code {result.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage changes, commit the current snapshot, package it, then run formal tests."
    )
    parser.add_argument(
        "-m",
        "--message",
        default="",
        help="Git commit message. If omitted, a timestamped default is used.",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Package the current working tree and run formal tests without git add/commit.",
    )
    args = parser.parse_args()

    print("[1/5] Repo root: .")
    run_cmd(["git", "rev-parse", "--show-toplevel"], "git rev-parse", quiet_stdout=True)

    staged_changes = False
    if not args.no_commit:
        print("[2/5] Stage changes")
        run_cmd(["git", "add", "-A"], "git add -A")
        diff_result = run_cmd(
            ["git", "diff", "--cached", "--quiet"],
            "git diff --cached --quiet",
            check=False,
        )
        if diff_result.returncode == 1:
            staged_changes = True
            warn_if_experiment_log_staged()
        elif diff_result.returncode != 0:
            raise SystemExit(
                f"git diff --cached --quiet failed with exit code {diff_result.returncode}"
            )
    else:
        print("[2/5] --no-commit set. Skip git add/commit.")

    if args.no_commit:
        print("[3/5] --no-commit set. Skip commit.")
    elif staged_changes:
        message = args.message.strip() or f"bundle run {datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"[3/5] Commit current snapshot: {message}")
        run_cmd(["git", "commit", "-m", message], "git commit")
    else:
        print("[3/5] No staged changes. Skip commit.")

    print("[4/5] Run package_zip.py before formal tests")
    run_cmd([sys.executable, "apps/package_zip.py"], "python apps/package_zip.py")

    print("[5/5] Run test_suite.py after package")
    run_cmd([sys.executable, "apps/test_suite.py"], "python apps/test_suite.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
