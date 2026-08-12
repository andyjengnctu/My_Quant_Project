from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object, *, length: int | None = None) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if length is None:
        return digest
    if int(length) <= 0:
        raise ValueError("length must be positive when provided")
    return digest[: int(length)]


__all__ = ["canonical_json_sha256", "compute_file_sha256"]
