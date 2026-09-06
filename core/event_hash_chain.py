"""Canonical hash primitive for revisioned state-event chains."""
from __future__ import annotations

from typing import Any, Mapping

from core.file_integrity import canonical_json_sha256


def compute_event_hash(event: Mapping[str, Any]) -> str:
    """Hash an event while excluding its self-referential ``event_hash`` field."""

    return canonical_json_sha256({key: value for key, value in event.items() if key != "event_hash"})


__all__ = ["compute_event_hash"]
