from __future__ import annotations

import shutil
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
LOCAL_REGRESSION_RETENTION_MODE = "count"


@dataclass(frozen=True)
class RetentionRule:
    name: str
    target_dir: Path
    patterns: Sequence[str]
    keep_last_n: int
    max_age_days: int
    include_dirs: bool = False
    mode: str = "both"
    accepts: Callable[[Path], bool] | None = None


def local_regression_bundle_rule(project_root: Path, settings: dict) -> RetentionRule:
    """One policy shared by the diagnostic producer and the cleaner."""
    return RetentionRule(
        name="local_regression_history",
        target_dir=Path(project_root) / "outputs/local_regression",
        patterns=("to_chatgpt_bundle_*.zip",),
        keep_last_n=settings["local_regression_keep_last_n"],
        max_age_days=settings["local_regression_max_age_days"],
        mode=settings.get("local_regression_retention_mode", LOCAL_REGRESSION_RETENTION_MODE),
        accepts=is_local_regression_bundle,
    )


def is_local_regression_bundle(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            info = archive.getinfo("master_summary.json")
            # The JSON is read, never extracted or executed.
            payload = json.loads(archive.read(info))
            return (isinstance(payload, dict) and payload.get("overall_status") in {"PASS", "FAIL"}
                    and isinstance(payload.get("scripts"), list) and payload.get("dataset") == "reduced")
    except (OSError, BadZipFile, KeyError, ValueError, TypeError, RuntimeError, NotImplementedError):
        return False


def expired_paths(rule: RetentionRule, *, now: datetime | None = None) -> List[Path]:
    """Plan only; retention is never permission to follow a link or delete a tree."""
    if (type(rule.keep_last_n) is not int or rule.keep_last_n < 0
            or type(rule.max_age_days) is not int or rule.max_age_days < 0
            or not isinstance(rule.mode, str) or rule.mode not in {"both", "either", "count", "age"}):
        raise ValueError("保留規則的份數、天數或判斷模式無效")
    current_time = now or datetime.now(TAIPEI_TZ)
    candidates = sorted(_iter_matching_paths(rule), key=lambda p: (_mtime_key(p), p.name), reverse=True)
    keep_set = set(candidates[:rule.keep_last_n])
    age_cutoff = current_time - timedelta(days=rule.max_age_days)
    expired = []
    for path in candidates:
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, TAIPEI_TZ)
        except FileNotFoundError:
            continue
        over_count, over_age = path not in keep_set, modified_at <= age_cutoff
        remove = {"both": over_count and over_age, "either": over_count or over_age,
                  "count": over_count, "age": over_age}[rule.mode]
        if remove:
            expired.append(path)
    return expired


def _iter_matching_paths(rule: RetentionRule) -> List[Path]:
    matches: List[Path] = []
    seen = set()
    if not rule.target_dir.exists():
        return matches
    for pattern in rule.patterns:
        for path in sorted(rule.target_dir.glob(pattern)):
            if path in seen:
                continue
            if not rule.include_dirs and not path.is_file():
                continue
            if path.is_symlink() or path.absolute() != path.resolve():
                continue
            if rule.include_dirs and not path.exists():
                continue
            if rule.accepts is not None and not rule.accepts(path):
                continue
            seen.add(path)
            matches.append(path)
    return matches


def _mtime_key(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _remove_path(path: Path) -> int:
    if path.is_dir():
        total = 0
        for sub in path.rglob("*"):
            if sub.is_file():
                try:
                    total += sub.stat().st_size
                except FileNotFoundError:
                    continue
        shutil.rmtree(path, ignore_errors=True)
        return total
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    try:
        path.unlink()
    except FileNotFoundError:
        return 0
    return int(size)


def apply_retention_rules(rules: Iterable[RetentionRule], *, now: datetime | None = None) -> Dict[str, object]:
    current_time = now or datetime.now(TAIPEI_TZ)
    removed_entries: List[Dict[str, object]] = []

    for rule in rules:
        for path in expired_paths(rule, now=current_time):
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, TAIPEI_TZ)
            except FileNotFoundError:
                continue
            removed_bytes = _remove_path(path)
            removed_entries.append(
                {
                    "rule": rule.name,
                    "path": str(path),
                    "bytes": removed_bytes,
                    "modified_at": modified_at.isoformat(),
                }
            )

    return {
        "removed_count": len(removed_entries),
        "removed_bytes": int(sum(int(item["bytes"]) for item in removed_entries)),
        "removed_entries": removed_entries,
    }
