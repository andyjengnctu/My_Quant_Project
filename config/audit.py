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
# Shared CPU-process budget across active Seed workers and their ticker workers.
# Outer Seed Replay uses one process per running seed; ticker workers are
# bounded by the remaining budget and cannot change any score/replay identity.
TARGET_ORACLE_CPU_WORKER_BUDGET = 8
TARGET_ORACLE_MINTRADE_STOCK_WORKERS = 6

__all__ = [
    "AUDIT_ACTIVE_MODULE_ID",
    "AUDIT_OUTPUT_ROOT",
    "TARGET_ORACLE_REPLAY_SEED_WORKERS",
    "TARGET_ORACLE_CPU_WORKER_BUDGET",
    "TARGET_ORACLE_MINTRADE_STOCK_WORKERS",
]
