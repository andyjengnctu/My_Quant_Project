"""One immutable execution-consumer boundary for a Trading lifecycle operation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.file_integrity import canonical_json_sha256
from core.trading_identity import normalize_trading_date
from services.trading.market_data_consumer import (
    load_trading_v2_consumer_state, open_trading_v2_consumer_view,
)


@dataclass(frozen=True)
class TradingLifecycleContext:
    finalized_date: str
    fingerprint: str
    consumer_state: Mapping[str, Any]
    view: Any


def resolve_trading_lifecycle_context(
    project_root: str | Path, *, consumer_state: Mapping[str, Any] | None = None,
) -> TradingLifecycleContext:
    # AI: Read/pin once; no downstream component may substitute a newer view or
    # independently derive its own latest day from the wall clock.
    state = deepcopy(dict(consumer_state)) if consumer_state is not None else load_trading_v2_consumer_state(project_root, required=True)
    date = normalize_trading_date(state.get("market_date"), field_name="latest_finalized_date", allow_none=False)
    view = open_trading_v2_consumer_view(project_root, consumer_state=state)
    identity = {
        "market_date": date,
        "state_fingerprint": state.get("state_fingerprint"),
        "provider_snapshot_fingerprint": state.get("provider_snapshot_fingerprint"),
        "source_view_fingerprint": state.get("source_view_fingerprint"),
        "source_overlay_batch_fingerprints": state.get("source_overlay_batch_fingerprints"),
    }
    return TradingLifecycleContext(date, canonical_json_sha256(identity), state, view)
