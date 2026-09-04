"""Canonical implementation-capability snapshot for pre-live Trading safety."""
from __future__ import annotations

from copy import deepcopy

TRADING_CAPABILITY_SCHEMA_VERSION = 1

# These are implementation facts, not user-configurable policy.  A capability
# may turn true only when its canonical production path and invariant tests exist.
_TRADING_CAPABILITIES = {
    "completed_daily_bar_seal": {
        "implemented": True,
        "required_for_live": True,
        "description": "Downloader seals Trading OHLCV to the latest completed daily information date.",
    },
    "broker_order_fill_reconciliation": {
        "implemented": True,
        "required_for_live": True,
        "description": "BUY/Protection SELL broker fills reconcile through canonical exact accounting.",
    },
    "daily_position_rollforward": {
        "implemented": False,
        "required_for_live": True,
        "description": "Overnight strategy positions are not yet advanced from completed daily bars to the next-session trailing-stop state.",
    },
    "indicator_sell_execution": {
        "implemented": False,
        "required_for_live": True,
        "description": "Full rule-based ind_sell_signal is not yet wired to the actual Trading broker-order lifecycle.",
    },
}


def build_trading_capability_snapshot() -> dict[str, object]:
    capabilities = deepcopy(_TRADING_CAPABILITIES)
    blockers = [
        name
        for name, spec in capabilities.items()
        if bool(spec.get("required_for_live")) and not bool(spec.get("implemented"))
    ]
    return {
        "schema_version": TRADING_CAPABILITY_SCHEMA_VERSION,
        "capabilities": capabilities,
        "live_blocking_capabilities": blockers,
        "all_required_live_capabilities_ready": not blockers,
    }


__all__ = [
    "TRADING_CAPABILITY_SCHEMA_VERSION",
    "build_trading_capability_snapshot",
]
