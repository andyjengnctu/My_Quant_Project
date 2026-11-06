"""Signed raw-scalar Positive evidence for price-only Oracle score recipients.

This cache is independent of legacy percentile price panels and per-seed Replay.
No Target or PIT label is materialized while planning a RUN/REUSE matrix.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import zipfile
from typing import Callable, Mapping, Any

import numpy as np

from core.file_integrity import (
    atomic_replace_with_retry, canonical_json_sha256, exclusive_file_lock,
)

SCHEMA = 1


def _digest(scores: Mapping[tuple[str, str], float]) -> str:
    digest = sha256()
    for (ticker, day), number in sorted(scores.items()):
        value = float(number)
        if value not in (0.0, 1.0):
            raise ValueError("Price Positive score must be an exact binary probability")
        digest.update(f"{ticker}|{day}|{int(value)}\n".encode("utf-8"))
    return digest.hexdigest()


def _verified(path: Path, identity: Mapping[str, Any],
              keys: set[tuple[str, str]]) -> dict[tuple[str, str], float] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"meta", "tickers", "days", "classes"}:
                raise ValueError("unexpected Positive evidence fields")
            meta = json.loads(str(archive["meta"].item()))
            tickers = archive["tickers"]
            days = archive["days"]
            classes = archive["classes"]
        if (not isinstance(meta, dict) or meta.get("identity") != dict(identity)
                or tickers.ndim != 1 or days.ndim != 1 or classes.ndim != 1
                or not (len(tickers) == len(days) == len(classes) == meta.get("rows"))
                or tickers.dtype.kind != "U" or days.dtype.kind != "U"
                or classes.dtype.kind != "f"):
            raise ValueError("Positive evidence identity/shape mismatch")
        pairs = list(zip(tickers.tolist(), days.tolist()))
        if len(pairs) != len(keys) or set(pairs) != keys:
            raise ValueError("Positive evidence keys differ from scored PIT universe")
        rows = {key: float(value) for key, value in zip(pairs, classes)}
        digest = _digest(rows)
        receipt = {"identity": dict(identity), "rows": len(rows),
                   "positive": sum(value == 1.0 for value in rows.values()),
                   "score_sha256": digest}
        if (meta.get("receipt_sha256") != canonical_json_sha256(receipt)
                or any(meta.get(k) != v for k, v in receipt.items())):
            raise ValueError("Positive evidence signed contents mismatch")
        return rows
    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Price Positive truth evidence corrupt: {path.name}: {exc}") from exc


def _publish(path: Path, identity: Mapping[str, Any],
             scores: Mapping[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(scores)
    receipt = {"identity": dict(identity), "rows": len(keys),
               "positive": sum(scores[key] == 1.0 for key in keys),
               "score_sha256": _digest(scores)}
    meta = json.dumps({**receipt, "receipt_sha256": canonical_json_sha256(receipt)},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fd, temporary = tempfile.mkstemp(prefix=".price-positive.", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, meta=np.asarray(meta),
                                tickers=np.asarray([key[0] for key in keys]),
                                days=np.asarray([key[1] for key in keys]),
                                classes=np.asarray([scores[key] for key in keys], dtype=np.float32))
            stream.flush()
            os.fsync(stream.fileno())
        with exclusive_file_lock(path.parent / ".publish.lock"):
            previous = _verified(path, identity, set(keys))
            if previous is None:
                atomic_replace_with_retry(temporary, path)
            elif previous != dict(scores):
                raise ValueError("same-identity Positive evidence conflicts")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_or_build_price_positive_scores(
    *, cache_root: Path, identity: Mapping[str, Any],
    canonical_scores: Mapping[tuple[str, str], float],
    build_raw: Callable[[], Mapping[tuple[str, str], float]],
    allow_build: bool,
) -> dict[tuple[str, str], float]:
    """Certified raw>0 output; never infer sign from percentile>0.5.

    Non-finite/missing raw values are BLOCKED rather than classified as zero.
    A signed class map is reused only for the same underlying certified raw
    source, scored universe, algorithm and market input identity.
    """
    if not canonical_scores:
        raise ValueError("Price Positive requires a nonempty canonical score map")
    expected = set(canonical_scores)
    path = Path(cache_root) / "price_positive_truth" / (canonical_json_sha256(identity) + ".npz")
    saved = _verified(path, identity, expected)
    if saved is not None:
        return saved
    if not allow_build:
        raise ValueError("Price Positive truth missing signed artifact after parent BUILD")
    raw = build_raw()
    if set(raw) != expected:
        raise ValueError("Price Positive raw truth has missing/extra scored stock-days")
    result = {}
    for key, number in raw.items():
        value = float(number)
        if not np.isfinite(value):
            raise ValueError("Price Positive raw truth has non-finite scored rows")
        result[key] = 1.0 if value > 0.0 else 0.0
    _publish(path, identity, result)
    return result
