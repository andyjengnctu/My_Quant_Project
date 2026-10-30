"""User/project-adjustable Audit policy values.

Formal/reusable Audit definitions are owned by ``core.audit_registry`` and resolved by
``core.audit_policy``.  Keep this module declarative.
"""

from __future__ import annotations

AUDIT_OUTPUT_ROOT = "outputs/audit"
AUDIT_ACTIVE_MODULE_ID = "breakout_quality"

# Oracle Portfolio Replay: concurrent independent RUN seeds (processes).
# Resource-only knob; it does not enter Target/Replay scientific identity.
# Each worker maintains one seed's Target truth and Replay caches.
TARGET_ORACLE_REPLAY_SEED_WORKERS = 4

__all__ = [
    "AUDIT_ACTIVE_MODULE_ID",
    "AUDIT_OUTPUT_ROOT",
    "TARGET_ORACLE_REPLAY_SEED_WORKERS",
]
