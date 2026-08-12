"""Shared Selection-PIT identity validation for Audit composition."""

from __future__ import annotations
from typing import Any

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

__all__ = ["candidate_pit_identity"]
