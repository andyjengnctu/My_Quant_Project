"""User/project-adjustable Audit policy values.

Formal/reusable Audit definitions are owned by ``core.audit_registry`` and resolved by
``core.audit_policy``.  Keep this module declarative.
"""

from __future__ import annotations

AUDIT_OUTPUT_ROOT = "outputs/audit"
AUDIT_ACTIVE_MODULE_ID = "breakout_quality"

__all__ = [
    "AUDIT_ACTIVE_MODULE_ID",
    "AUDIT_OUTPUT_ROOT",
]
