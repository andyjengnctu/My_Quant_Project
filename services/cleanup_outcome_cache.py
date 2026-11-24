"""Deletion proofs for the producer's disposable execution memoization only."""
from pathlib import Path
import re
from zipfile import BadZipFile
import zlib

import numpy as np

from core.compare_csv_storage import clean_path, plain
from core.file_integrity import file_change_signature
from services.breakout_quality.formal_mintrade_outcome_cache import (
    _CACHE_SUBDIR, _read_cache, formal_mintrade_cache_access,
)

CACHE_ROOT = _CACHE_SUBDIR
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def candidate(relative: Path) -> bool:
    relative = Path(relative)
    if not relative.is_relative_to(CACHE_ROOT):
        return False
    local = relative.relative_to(CACHE_ROOT)
    return (len(local.parts) == 2 and local.suffix == ".npz"
            and bool(_HEX.fullmatch(local.stem)) and local.parts[0] == local.stem[:2])


def deletion_lease(root: Path):
    return formal_mintrade_cache_access(root)


def proof(root: Path, relative: Path) -> str:
    """Validate the existing producer schema without trusting names alone.

    These shards cache only the execution batch. Every consumer has the full
    canonical compute path and independently checks its PIT teacher and inputs.
    No historical result or model artifact is an eligible cache shard.
    """
    if not candidate(relative):
        return ""
    path = root / relative
    if not clean_path(path) or not plain(path.lstat()):
        return ""
    before = file_change_signature(path)
    try:
        with np.load(path, allow_pickle=False) as store:
            identity = store["identity"]
            if identity.shape != () or not _HEX.fullmatch(str(identity.item())):
                return ""
            values = store["net_r"]
            if values.ndim != 1:
                return ""
            count = len(values)
            key = str(identity.item())
    except (OSError, ValueError, KeyError, TypeError, BadZipFile, zlib.error, EOFError, OverflowError):
        # Unknown or damaged payloads are retained, never treated as disposal proof.
        return ""
    # Reuse the producer's complete schema, shape, finite-value, maturity,
    # forced-closeout and counter checks instead of copying that contract.
    if _read_cache(path, key, count) is None:
        return ""
    if file_change_signature(path) != before:
        raise ValueError("執行快取在驗證期間變動")
    return "產生者明定可重建且已驗證格式的 Min 執行記憶快取；原始資料與 PIT 證據保留"
