from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace one JSON file with a uniquely staged sibling temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(
            json.dumps(
                payload, ensure_ascii=False, indent=2, allow_nan=False, default=str
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return target


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


__all__ = ["atomic_write_json", "canonical_json_sha256", "compute_file_sha256", "load_json_strict"]
