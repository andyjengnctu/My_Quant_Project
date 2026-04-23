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
        description="Stage/commit changes, then run package_zip.py and test_suite.py."
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
        help="Skip git add/commit and only run package_zip.py and test_suite.py.",
    )
    args = parser.parse_args()

    print(f"[1/5] Repo root: {REPO_ROOT}")
    run_cmd(["git", "rev-parse", "--show-toplevel"], "git rev-parse")

    if not args.no_commit:
        print("[2/5] Stage changes")
        run_cmd(["git", "add", "-A"], "git add -A")

        diff_result = run_cmd(["git", "diff", "--cached", "--quiet"], "git diff --cached --quiet", check=False)
        if diff_result.returncode == 1:
            message = args.message.strip() or f"bundle run {datetime.now():%Y-%m-%d %H:%M:%S}"
            print(f"[3/5] Commit: {message}")
            run_cmd(["git", "commit", "-m", message], "git commit")
        elif diff_result.returncode == 0:
            print("[3/5] No staged changes. Skip commit.")
        else:
            raise SystemExit(
                f"git diff --cached --quiet failed with exit code {diff_result.returncode}"
            )
    else:
        print("[2/5] --no-commit set. Skip git add/commit.")

    print("[4/5] Run package_zip.py")
    run_cmd([sys.executable, "apps/package_zip.py"], "python apps/package_zip.py")

    print("[5/5] Run test_suite.py")
    run_cmd([sys.executable, "apps/test_suite.py"], "python apps/test_suite.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
