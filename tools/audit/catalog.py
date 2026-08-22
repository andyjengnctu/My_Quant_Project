"""Compatibility alias for the formal Audit catalog service."""

from __future__ import annotations

import sys
from services.audit import catalog as _impl

sys.modules[__name__] = _impl
