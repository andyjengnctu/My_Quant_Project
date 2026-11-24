from __future__ import annotations

import argparse
import re
import subprocess
import sys
import json
from hashlib import sha256
import zipfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps import package_zip
from apps.test_suite import REGRESSION_STEP_ORDER
from services.research.oracle_build_attestation import (
    EVIDENCE_DIRECTORY, OracleBuildBlockedError, prepare_oracle_build_candidate,
    finalize_oracle_build_attestation, publish_oracle_build_attestation,
    require_oracle_build_attestation,
)
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


def _git_head() -> str:
    return package_zip._run_git("rev-parse", "HEAD").stdout.strip()


def _formal_bundle_paths() -> set[Path]:
    return set(REPO_ROOT.glob("to_chatgpt_bundle_*.zip"))


def _read_formal_summary(previous: set[Path]):
    created = _formal_bundle_paths() - previous
    if len(created) != 1:
        raise OracleBuildBlockedError("Formal verification did not produce one unambiguous evidence bundle")
    path = next(iter(created))
    with zipfile.ZipFile(path) as archive:
        content = archive.read("master_summary.json")
    return json.loads(content), sha256(content).hexdigest(), path


def main(argv=None) -> int:
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
    args = parser.parse_args(argv)

    print("[1/6] Repo root: .")
    run_cmd(["git", "rev-parse", "--show-toplevel"], "git rev-parse", quiet_stdout=True)

    staged_changes = False
    if not args.no_commit:
        print("[2/6] Stage changes")
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
        print("[2/6] --no-commit set. Skip git add/commit.")

    if args.no_commit:
        print("[3/6] --no-commit set. Skip commit.")
    elif staged_changes:
        message = args.message.strip() or f"bundle run {datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"[3/6] Commit current snapshot: {message}")
        run_cmd(["git", "commit", "-m", message], "git commit")
    else:
        print("[3/6] No staged changes. Skip commit.")

    print("[4/6] Freeze package before formal tests")
    snapshot = package_zip.create_package_snapshot()
    frozen_commit = _git_head()
    candidate = None
    admission_failure = None
    try:
        candidate = prepare_oracle_build_candidate(REPO_ROOT)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, OracleBuildBlockedError) as exc:
        admission_failure = str(exc) if isinstance(exc, OracleBuildBlockedError) else type(exc).__name__
        print(f"[Oracle build] IMPLEMENTATION_UNVERIFIED: {admission_failure}")

    previous_bundles = _formal_bundle_paths()
    print("[5/6] Run test_suite.py after frozen package")
    result = run_cmd([sys.executable, "apps/test_suite.py"], "python apps/test_suite.py", check=False)
    if result.returncode != 0:
        print("[Oracle build] Formal FAIL; original commit and source package retained.")
        return int(result.returncode)
    print("[6/6] Verify frozen sources and finalize Oracle evidence")
    try:
        if candidate is None:
            raise OracleBuildBlockedError(admission_failure or "Oracle equivalence evidence missing")
        if _git_head() != frozen_commit:
            raise OracleBuildBlockedError("Git HEAD changed during formal verification")
        package_zip.verify_frozen_package(snapshot)
        summary, summary_sha, formal_bundle = _read_formal_summary(previous_bundles)
        bundle_sha = sha256(formal_bundle.read_bytes()).hexdigest()
        name = f"{EVIDENCE_DIRECTORY}/formal_bundle_{bundle_sha[:16]}.zip"
        candidate["evidence_files"][name] = {"path": str(formal_bundle), "sha256": bundle_sha}
        candidate["evidence_artifacts"][name] = bundle_sha
        attestation = finalize_oracle_build_attestation(
            REPO_ROOT, candidate=candidate, formal_summary=summary,
            required_steps=REGRESSION_STEP_ORDER, formal_summary_sha256=summary_sha, commit=frozen_commit,
        )
        package_zip.finalize_frozen_package(snapshot, attestation=attestation,
                                            evidence_files=candidate["evidence_files"])
        publish_oracle_build_attestation(REPO_ROOT, attestation, evidence_files=candidate["evidence_files"])
        require_oracle_build_attestation(REPO_ROOT)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as exc:
        detail = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        print(f"[Oracle build] IMPLEMENTATION_UNVERIFIED: {detail}; commit and package retained.")
        return 1
    print("[Oracle build] VERIFIED; frozen source package includes the certificate and its evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
