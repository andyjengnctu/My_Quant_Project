from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


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


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace one JSON file through the shared text-publication primitive."""

    return atomic_write_text(
        path,
        json.dumps(
            payload, ensure_ascii=False, indent=2, allow_nan=False, default=str
        )
        + "\n",
    )


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
    "atomic_replace_with_retry",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_json_sha256",
    "compute_file_sha256",
    "load_json_object_or_none",
    "load_json_strict",
]
