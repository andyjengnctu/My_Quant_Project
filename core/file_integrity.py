from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
import re
import shutil
import tempfile
import time
from uuid import uuid4
from pathlib import Path
from typing import Iterator
from typing import Any


# SHA256 digests are canonical lowercase ASCII hex (never a numeric/date token).
# Both the Research and Trading Market Data artifact paths consume this owner;
# do not accept arbitrary 64-character values as valid digest identities.
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


def is_sha256_hex(value: object) -> bool:
    """Whether a value is a complete SHA256 hex digest after legacy normalization."""
    return _SHA256_HEX_RE.fullmatch(str(value or "").strip().lower()) is not None


def require_sha256_hex(
    value: object, *, field: str, error_message: str | None = None
) -> str:
    """Return canonical SHA256 digest; let each domain retain its error wording."""
    text = str(value or "").strip().lower()
    if not _SHA256_HEX_RE.fullmatch(text):
        raise ValueError(error_message if error_message is not None else f"{field} 必須是 SHA256 hex")
    return text


def file_change_signature(path: str | Path) -> tuple[int, int, int, int, int] | tuple[str]:
    """Cheap local invalidation hint, NOT a scientific/content fingerprint.

    Include inode and ctime in addition to mtime/size so normal atomic
    replacements and same-size restored-mtime writes trigger revalidation.
    The owning loader must verify canonical SHA256 on every cache miss.
    """
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return ("missing",)
    return (
        int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
        int(stat.st_mtime_ns), int(stat.st_ctime_ns),
    )


def compute_file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace_with_retry(
    source: str | Path,
    target: str | Path,
    *,
    max_attempts: int = 20,
    retry_delay_seconds: float = 0.10,
) -> Path:
    """Publish one staged sibling file, retrying only transient PermissionError locks."""

    src = Path(source)
    dst = Path(target)
    attempts = int(max_attempts)
    if attempts <= 0:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return dst
        except PermissionError:
            if attempt >= attempts:
                raise
            time.sleep(float(retry_delay_seconds))
    raise RuntimeError("atomic replace retry loop ended unexpectedly")



def sibling_publication_lock_path(artifact_dir: str | Path) -> Path:
    """Stable lock file beside an artifact directory, never inside its movable tree."""
    entry = Path(artifact_dir).resolve()
    if entry.name in {"", ".", ".."}:
        raise ValueError("Artifact lock requires a named directory")
    return entry.parent / ("." + entry.name + ".publish.lock")


@contextmanager
def exclusive_file_lock(
    lock_path: str | Path, *, conflict_message: str = "Another artifact publication is active"
) -> Iterator[None]:
    """Nonblocking cross-process publication lock; OS releases it on process exit.

    Keep the lock file instead of unlinking it (unlinking permits split inodes).
    This is a writer coordination primitive, not an artifact identity or a reader
    transaction. Publication readers continue to enforce manifest and file SHA.
    """
    lock_file = Path(lock_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\x00")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(str(conflict_message)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    max_attempts: int = 20,
    retry_delay_seconds: float = 0.10,
) -> Path:
    """Atomically replace one text file with fsync + transient-lock retry semantics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("w", encoding=encoding) as handle:
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace_with_retry(
            temp, target, max_attempts=max_attempts, retry_delay_seconds=retry_delay_seconds
        )
    except BaseException as exc:
        if temp.exists():
            try:
                temp.unlink()
            except OSError as cleanup_exc:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "atomic temp cleanup failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
        raise
    return target


def reserve_immutable_output_dir(parent: str | Path, timestamp: str) -> Path:
    """Allocate a distinct historical run directory, even for same-second runs."""
    root = Path(parent)
    candidate = root / str(timestamp)
    try:
        candidate.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        candidate = root / f"{timestamp}_{uuid4().hex}"
        candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def publish_latest_snapshot(
    latest_dir: str | Path, files: tuple[tuple[str, bytes], ...],
    *, conflict_message: str = "Another latest snapshot publication is active",
) -> None:
    """Publish legacy multi-file latest views via one shared, bounded owner.

    Stage and fsync every file before changing the visible view; publish in
    caller-declared order (commit metadata last). A recoverable failure rolls
    back all already replaced files. The stable sibling lock serializes only
    cooperating readers/writers. This is *not* an all-files atomic transaction:
    uncoordinated readers may still observe an in-flight transition.
    """
    root = Path(latest_dir)
    names = [str(name) for name, _value in files]
    if not files or len(names) != len(set(names)) or any(
        not name or name in {".", ".."} or Path(name).name != name
        for name in names
    ):
        raise ValueError("latest snapshot requires unique basename-only file names")
    with exclusive_file_lock(
        sibling_publication_lock_path(root), conflict_message=conflict_message,
    ):
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{root.name}.stage.", dir=str(root.parent)) as tmp_name:
            stage = Path(tmp_name)
            staged: list[tuple[Path, Path, Path | None]] = []
            for filename, content in files:
                target = root / filename
                staged_path = stage / filename
                with staged_path.open("wb") as stream:
                    stream.write(bytes(content))
                    stream.flush()
                    os.fsync(stream.fileno())
                backup = stage / (filename + ".backup") if target.is_file() else None
                if backup is not None:
                    shutil.copy2(target, backup)
                staged.append((staged_path, target, backup))
            published: list[tuple[Path, Path | None]] = []
            try:
                for staged_path, target, backup in staged:
                    atomic_replace_with_retry(staged_path, target)
                    published.append((target, backup))
            except BaseException as exc:
                restore_errors = []
                for target, backup in reversed(published):
                    try:
                        if backup is None:
                            target.unlink(missing_ok=True)
                        else:
                            atomic_replace_with_retry(backup, target)
                    except OSError as restore_exc:
                        restore_errors.append(f"{target.name}: {restore_exc}")
                if restore_errors:
                    raise RuntimeError(
                        "latest snapshot publication failed and restoration incomplete: "
                        + "; ".join(restore_errors)
                    ) from exc
                raise


def text_publication_bytes(text: str) -> bytes:
    """Match historical Path.write_text/atomic_write_text newline bytes on this OS.

    Windows TextIOWrapper translates LF to CRLF when newline is unspecified;
    binary latest publication must do the same without a second conversion.
    """
    return str(text).replace("\n", os.linesep).encode("utf-8")


def _json_publication_text(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False, default=str,
    ) + "\n"


def json_publication_bytes(payload: Any) -> bytes:
    """Match the *platform-native* bytes of historical atomic_write_json."""
    return text_publication_bytes(_json_publication_text(payload))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace one JSON file through the shared text-publication primitive."""
    # Feed native *text*, not already-newline-converted bytes, to TextIOWrapper.
    return atomic_write_text(path, _json_publication_text(payload))


def load_json_object_or_none(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> dict[str, Any] | None:
    """Read one JSON object for status/report consumers; missing or malformed files are absent."""

    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding=encoding))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None

def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def load_json_strict(path: str | Path) -> Any:
    """Load RFC-compatible JSON and reject NaN/Infinity constants."""
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json_constant,
    )


def canonical_json_sha256(payload: object, *, length: int | None = None) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if length is None:
        return digest
    if int(length) <= 0:
        raise ValueError("length must be positive when provided")
    return digest[: int(length)]


__all__ = [
    "file_change_signature",
    "atomic_replace_with_retry",
    "exclusive_file_lock",
    "sibling_publication_lock_path",
    "atomic_write_json",
    "json_publication_bytes",
    "text_publication_bytes",
    "publish_latest_snapshot",
    "reserve_immutable_output_dir",
    "atomic_write_text",
    "canonical_json_sha256",
    "compute_file_sha256",
    "load_json_object_or_none",
    "load_json_strict",
    "is_sha256_hex",
    "require_sha256_hex",
]
