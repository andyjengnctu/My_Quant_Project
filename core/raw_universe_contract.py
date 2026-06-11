from __future__ import annotations

import re
from numbers import Integral, Real
from typing import Any, Mapping


RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD = "raw_universe_required_min_rows"


def coerce_raw_universe_required_min_rows(value: Any, *, allow_none: bool = True) -> int | None:
    """Normalize the persisted optimizer raw-universe row boundary.

    The value is a replay contract, not a tunable runtime approximation.  Reject
    booleans, fractional numbers and malformed text so ticker-universe selection
    cannot change silently because of permissive ``int()`` coercion.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 不可為空")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            if allow_none:
                return None
            raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 不可為空")
        if re.fullmatch(r"\+?\d+", text) is None:
            raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 必須是正整數，收到: {value!r}")
        resolved = int(text)
    elif isinstance(value, bool):
        raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 必須是正整數，收到 bool")
    elif isinstance(value, Integral):
        resolved = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 必須是正整數，收到: {value!r}")
        resolved = int(numeric)
    else:
        raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 必須是正整數，收到: {value!r}")

    if resolved <= 0:
        raise ValueError(f"{RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD} 必須大於 0，收到: {resolved}")
    return int(resolved)


def resolve_raw_universe_required_min_rows(payload: Mapping[str, Any] | None, *, default: int | None = None) -> int | None:
    """Resolve the optimizer ticker-universe boundary persisted in a replay payload.

    Optimizer search intentionally loads a stable ticker universe based on the
    maximum lookback across the full search space.  Replays must preserve that
    boundary; otherwise a selected parameter set with a shorter lookback can add
    tickers that were excluded during objective evaluation.
    """
    if isinstance(payload, Mapping):
        candidates = [payload.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD)]
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            candidates.append(meta.get(RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD))
        for raw_value in candidates:
            resolved = coerce_raw_universe_required_min_rows(raw_value, allow_none=True)
            if resolved is not None:
                return resolved
    return coerce_raw_universe_required_min_rows(default, allow_none=True)


def build_raw_universe_contract_fields(raw_universe_required_min_rows: Any) -> dict[str, int]:
    resolved = coerce_raw_universe_required_min_rows(raw_universe_required_min_rows, allow_none=True)
    if resolved is None:
        return {}
    return {RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: int(resolved)}
