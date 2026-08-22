"""Compatibility alias for the formal Audit runner service."""

from __future__ import annotations

import sys
from services.audit import runner as _impl

sys.modules[__name__] = _impl
