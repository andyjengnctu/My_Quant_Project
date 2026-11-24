"""Storage lifecycle contracts; independent of FIT/Replay identities.

Producers declare storage roles after validating their own result. Consumers
pin dependencies with ``requires``. Cleanup never derives a role from a model
name, file extension, age, or an inactive research ID.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from core.storage_paths import clean_path, plain, relative_path
from core.file_integrity import (
    compute_observed_file_sha256, exclusive_file_lock, file_change_signature,
    is_sha256_hex,
)

MANIFEST_NAME = "artifact_lifecycle.json"
SCHEMA = "artifact_lifecycle_v1"
ROLES = frozenset({"result", "evidence", "resume", "work", "cache", "replica"})
STATES = frozenset({"RUNNING", "RESUMABLE", "COMPLETED", "SUPERSEDED", "ABANDONED"})
RETAINED_ROLES = frozenset({"result", "evidence", "resume"})


def lifecycle_lock_path(root: Path) -> Path:
    return Path(root) / "outputs/project_cleanup/artifact_lifecycle.lock"


def process_is_absent(pid: int) -> bool:
    """An inaccessible/reused PID is not proof that a writer has ended."""
    if type(pid) is not int or not 0 < pid <= 0x7fffffff:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        lib = ctypes.WinDLL("kernel32", use_last_error=True)
        lib.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        lib.OpenProcess.restype = wintypes.HANDLE
        lib.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = lib.OpenProcess(0x1000, False, pid)
        if handle:
            lib.CloseHandle(handle)
            return False
        return ctypes.get_last_error() == 87
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


@contextmanager
def lifecycle_access(root: Path):
    """Writers and the cleaner coordinate lifecycle/reference publication."""
    lock = lifecycle_lock_path(root)
    if not clean_path(lock) or (lock.exists() and (not plain(lock.lstat()) or lock.stat().st_nlink != 1)):
        raise ValueError("工件生命週期鎖路徑異常")
    with exclusive_file_lock(lock, conflict_message="工件正在發布或清理"):
        yield


@dataclass(frozen=True)
class StorageArtifact:
    path: Path
    role: str
    sha256: str
    requires: tuple[Path, ...] = ()
    replacement: Path | None = None


@dataclass(frozen=True)
class StorageLifecycle:
    marker: Path
    producer: str
    state: str
    subject: str
    workspace: Path
    artifacts: tuple[StorageArtifact, ...]
    publication: tuple[tuple[Path, str], ...]
    roots: tuple[Path, ...]
    revision: tuple
    active_pid: int | None = None


def read_lifecycle(root: Path, marker: Path) -> StorageLifecycle:
    root = Path(root)
    marker = Path(marker)
    file = root / marker
    if not clean_path(file) or not plain(file.lstat()):
        raise ValueError("生命週期紀錄不是一般檔案")
    revision = file_change_signature(file)
    payload = json.loads(file.read_text(encoding="utf-8-sig"))
    if file_change_signature(file) != revision:
        raise ValueError("生命週期紀錄在讀取時變動")
    return _parse_lifecycle(marker, payload, revision)


def _parse_lifecycle(marker: Path, payload: dict, revision: tuple) -> StorageLifecycle:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("生命週期紀錄格式未受支援")
    producer = payload.get("producer")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("生命週期缺少產生者")
    state = payload.get("state")
    if not isinstance(state, str) or state not in STATES:
        raise ValueError("生命週期工作狀態不完整")
    workspace = relative_path(payload.get("workspace"))
    # Explicit, bounded workspace ownership. Never allow a broad output domain
    # or a scientific model directory to become disposable through a sidecar.
    scope = tuple(part.casefold() for part in workspace.parts)
    if (len(scope) < 4 or scope[0] != "outputs"
            or "trading" in scope[:3]
            or scope[1] == "project_cleanup"
            or marker != workspace / MANIFEST_NAME):
        raise ValueError("生命週期範圍必須是獨立 outputs 工作目錄")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("生命週期缺少逐檔用途")
    artifacts = []
    paths = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("生命週期工件紀錄必須是 object")
        path = relative_path(row.get("path"))
        role = row.get("role")
        if (path in paths or not isinstance(role, str) or role not in ROLES or path == marker
                or not path.is_relative_to(workspace) or path == workspace
                or not isinstance(row.get("sha256"), str)
                or not is_sha256_hex(row.get("sha256"))):
            raise ValueError("生命週期工件角色、範圍或內容證據無效")
        paths.add(path)
        requirements = row.get("requires", [])
        if not isinstance(requirements, list):
            raise ValueError("工件依賴必須是清單")
        replacement = row.get("replacement")
        artifacts.append(StorageArtifact(
            path, role, row["sha256"].lower(),
            tuple(relative_path(value) for value in requirements),
            relative_path(replacement) if replacement is not None else None,
        ))
    publication = payload.get("publication")
    if not isinstance(publication, dict):
        raise ValueError("缺少已發布成果的內容證據")
    published = []
    for name, digest in publication.items():
        if not isinstance(digest, str) or not is_sha256_hex(digest):
            raise ValueError("已發布成果缺少內容雜湊")
        published.append((relative_path(name), digest.lower()))
    roots = payload.get("retain", [])
    if not isinstance(roots, list):
        raise ValueError("保留依賴根必須是清單")
    subject = payload.get("subject", "")
    if not isinstance(subject, str):
        raise ValueError("工作身分必須是字串")
    active_pid = payload.get("active_pid")
    if active_pid is not None and (type(active_pid) is not int or not 0 < active_pid <= 0x7fffffff):
        raise ValueError("工作程序身分無效")
    return StorageLifecycle(marker, producer, state, subject, workspace,
                            tuple(artifacts), tuple(published),
                            tuple(relative_path(value) for value in roots), revision, active_pid)


def retained_closure(
    records: Iterable[StorageLifecycle], *, external_roots: Iterable[Path] = (),
    active_subjects: Iterable[str] = (), existing_paths: Iterable[Path] | None = None,
) -> set[Path]:
    """A shared dependency graph, including roots in other workspaces."""
    records = tuple(records)
    active = set(active_subjects)
    existing = None if existing_paths is None else set(existing_paths)
    graph = {}
    keep = set(external_roots)
    for record in records:
        keep.add(record.marker)
        keep.update(record.roots)
        # Once every disposable has gone, its small audit record must not pin
        # engineering archives forever. Scientific results have their own roots.
        if (existing is None or any(a.path in existing or a.role in RETAINED_ROLES for a in record.artifacts)
                or record.state in {"RUNNING", "RESUMABLE"}):
            keep.update(path for path, _ in record.publication)
        for artifact in record.artifacts:
            if artifact.path in graph:
                raise ValueError("同一工件被多份生命週期紀錄宣告")
            graph[artifact.path] = artifact.requires
            if (artifact.role in RETAINED_ROLES
                    or record.state in {"RUNNING", "RESUMABLE"}
                    or (record.state in {"SUPERSEDED", "ABANDONED"}
                        and record.subject in active)):
                keep.add(artifact.path)
            if artifact.replacement is not None:
                keep.add(artifact.replacement)
    # A retained directory protects all declared descendants, too.
    keep.update(path for path in graph if any(parent in keep for parent in path.parents))
    pending = list(keep)
    while pending:
        for dependency in graph.get(pending.pop(), ()):
            if dependency not in keep:
                keep.add(dependency)
                pending.append(dependency)
                for child in graph:
                    if child.is_relative_to(dependency) and child not in keep:
                        keep.add(child)
                        pending.append(child)
    return keep


def publish_lifecycle(
    root: Path, *, workspace: Path, producer: str, state: str,
    roles: Mapping[Path, str], publication: Iterable[Path],
    requires: Mapping[Path, Iterable[Path]] | None = None,
    replacements: Mapping[Path, Path] | None = None,
    retain: Iterable[Path] = (), subject: str = "", active_pid: int | None = None,
    expected_sha256: Mapping[Path, str] | None = None,
) -> Path:
    """Publish engineering roles after the producer's canonical validation.

    Call with RUNNING/RESUMABLE before using a workspace, and COMPLETED only
    after its canonical producer has committed and validated final outputs.
    Files are project-relative; this helper creates no scientific identity.
    When supplied, expected_sha256 must cover both roles and publication, so
    the producer's verified snapshot cannot silently change during publishing.
    """
    workspace = relative_path(Path(workspace).as_posix())
    marker = workspace / MANIFEST_NAME
    root = Path(root)
    dependencies, replacements = requires or {}, replacements or {}
    with lifecycle_access(root):
        rows = []
        for path, role in roles.items():
            path = relative_path(Path(path).as_posix())
            target = root / path
            if not clean_path(target) or not plain(target.lstat()):
                raise ValueError("不能發布連結或特殊檔案的清理用途")
            row = {"path": path.as_posix(), "role": role,
                   "sha256": compute_observed_file_sha256(target),
                   "requires": [Path(p).as_posix() for p in dependencies.get(path, ())]}
            if expected_sha256 is not None and row["sha256"] != expected_sha256.get(path):
                raise ValueError("工作檔在原產生者驗證後已變動")
            if path in replacements:
                row["replacement"] = Path(replacements[path]).as_posix()
            rows.append(row)
        published = {}
        for raw in publication:
            path = relative_path(Path(raw).as_posix())
            file = root / path
            if not clean_path(file) or not plain(file.lstat()):
                raise ValueError("發布證據不是專案內的一般檔案")
            digest = compute_observed_file_sha256(file)
            if expected_sha256 is not None and digest != expected_sha256.get(path):
                raise ValueError("發布成果在原產生者驗證後已變動")
            published[path.as_posix()] = digest
        payload = {"schema": SCHEMA, "workspace": workspace.as_posix(),
                   "producer": producer, "state": state, "subject": subject, "active_pid": active_pid,
                   "artifacts": rows, "publication": published,
                   "retain": [Path(p).as_posix() for p in retain]}
        _parse_lifecycle(marker, payload, ())
        target = root / marker
        if not clean_path(target):
            raise ValueError("生命週期紀錄位置經過連結")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name("." + MANIFEST_NAME + ".pending")
        if temp.exists() or temp.is_symlink():
            raise ValueError("有未完成的生命週期發布紀錄")
        try:
            with temp.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Publish only after the shared parser has accepted every field.
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        read_lifecycle(root, marker)
        return target
