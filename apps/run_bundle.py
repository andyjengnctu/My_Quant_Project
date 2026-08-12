from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], step_name: str, *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=REPO_ROOT)
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

    print(f"[1/5] Repo root: {REPO_ROOT}")
    run_cmd(["git", "rev-parse", "--show-toplevel"], "git rev-parse")

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
