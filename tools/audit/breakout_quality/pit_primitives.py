"""Shared Selection-PIT identity validation for Audit composition."""

from __future__ import annotations
from typing import Any
import math
from pathlib import Path

import numpy as np
import pandas as pd

def pit_json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): pit_json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pit_json_native(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def candidate_pit_identity(result: dict[str, Any], candidate_id: str) -> dict[str, str]:
    settings = dict(result.get("settings") or {})
    arms = dict(settings.get("arms") or {})
    dl_sources = dict(settings.get("dl_sources") or {})
    arm = dict(arms.get(candidate_id) or {})
    dl_id = str(arm.get("dl_id") or "").strip()
    if not dl_id:
        raise ValueError(f"{candidate_id}缺少dl_id")
    dl = dict(dl_sources.get(dl_id) or {})
    if str(dl.get("score_source") or "") != "selection_point_in_time":
        raise ValueError(f"{candidate_id}不是Selection PIT score source")
    required = {
        "dl_id": dl_id,
        "filter_id": str(dl.get("filter_id") or "").strip(),
        "model_architecture": str(dl.get("model_architecture") or "").strip(),
        "experiment_profile": str(dl.get("experiment_profile") or "").strip(),
    }
    if any(not value for value in required.values()):
        raise ValueError(f"{candidate_id}的PIT DL identity不完整")
    return required

__all__ = ["candidate_pit_identity", "pit_json_native"]
