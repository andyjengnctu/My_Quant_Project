"""Small deterministic primitives shared by read-only Audit modules."""

from __future__ import annotations

import math
from pathlib import Path

from core.file_integrity import compute_file_sha256 as sha256_file


def finite_or_none(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None



__all__ = ["finite_or_none", "sha256_file"]
