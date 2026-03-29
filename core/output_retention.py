from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo

TAIPEI_TZ = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class RetentionRule:
    name: str
    target_dir: Path
    patterns: Sequence[str]
    keep_last_n: int
    max_age_days: int
    include_dirs: bool = False


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
            if rule.include_dirs and not path.exists():
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
                    pass
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
        candidates = sorted(_iter_matching_paths(rule), key=_mtime_key, reverse=True)
        if not candidates:
            continue

        keep_set = set(candidates[: max(int(rule.keep_last_n), 0)])
        age_cutoff = current_time - timedelta(days=max(int(rule.max_age_days), 0))

        for path in candidates:
            if path in keep_set:
                continue
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, TAIPEI_TZ)
            except FileNotFoundError:
                continue
            if modified_at > age_cutoff:
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
