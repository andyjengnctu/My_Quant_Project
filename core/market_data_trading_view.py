"""Pure contract for the Market Data V2 Trading historical/latest read view.

The Trading view is a domain view over the neutral Provider Snapshot plus the
verified Trading overlay.  It does not own provider download semantics, model
scientific scope, execution eligibility, or the neutral daily PIT universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence

import pandas as pd

from core.file_integrity import canonical_json_sha256

TRADING_V2_VIEW_CONTRACT_VERSION = 1
TRADING_V2_VIEW_ROLE = "latest_operational_view"
TRADING_V2_HISTORICAL_MEMBERSHIP_SOURCE = "neutral_daily_pit_market_universe"
TRADING_V2_CURRENT_EXECUTION_POOL_IS_TRAINING_SOURCE = False


@dataclass(frozen=True)
class TradingV2TrainingHorizon:
    provider_as_of_date: str
    training_through_date: str
    required_datasets: tuple[str, ...]
    dataset_ready_through: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_as_of_date": self.provider_as_of_date,
            "training_through_date": self.training_through_date,
            "required_datasets": list(self.required_datasets),
            "dataset_ready_through": dict(self.dataset_ready_through),
        }


def _iso(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} 必須是 YYYY-MM-DD: {text!r}") from exc


def trading_v2_view_contract_payload() -> dict[str, object]:
    return {
        "contract_version": TRADING_V2_VIEW_CONTRACT_VERSION,
        "role": TRADING_V2_VIEW_ROLE,
        "provider_base": "immutable_market_data_v2_provider_snapshot",
        "overlay": "verified_trading_market_data_v2_batches",
        "same_primary_key_precedence": "later_verified_overlay_wins",
        "training_horizon_rule": "minimum_ready_through_across_required_datasets",
        "provider_snapshot_is_minimum_ready_baseline": True,
        "historical_membership_source": TRADING_V2_HISTORICAL_MEMBERSHIP_SOURCE,
        "current_execution_pool_may_define_historical_training_universe": False,
        "execution_authority_changes_in_this_contract": False,
    }


def trading_v2_view_contract_fingerprint() -> str:
    return canonical_json_sha256(trading_v2_view_contract_payload())


def resolve_trading_v2_training_horizon(
    *,
    provider_as_of_date: str,
    required_datasets: Iterable[str],
    dataset_state: Mapping[str, Mapping[str, object]] | None,
) -> TradingV2TrainingHorizon:
    """Resolve the latest common information date for one production training contract.

    The immutable Provider Snapshot is a complete baseline through
    ``provider_as_of_date``.  Dataset-level Trading state may extend individual
    datasets beyond that date.  A model may train only through the minimum
    READY-through date across its exact required dataset set.
    """

    base = _iso(provider_as_of_date, field="provider_as_of_date")
    required: list[str] = []
    seen: set[str] = set()
    for raw in required_datasets:
        name = str(raw or "").strip()
        if not name:
            raise ValueError("Trading V2 required_datasets 不得包含空白 identity")
        if name in seen:
            continue
        seen.add(name)
        required.append(name)
    if not required:
        raise ValueError("Trading V2 training 必須宣告至少一個 required dataset")

    state = dict(dataset_state or {})
    ready: dict[str, str] = {}
    for dataset in required:
        row = dict(state.get(dataset) or {})
        candidate_raw = row.get("last_ready_target_date")
        if candidate_raw:
            candidate = _iso(candidate_raw, field=f"{dataset}.last_ready_target_date")
            # ``last_ready_target_date`` is the persisted horizon evidence.
            # Runtime authorization separately validates current schema/coverage
            # in services.trading.data_readiness; the view resolver must not
            # collapse a valid historical READY horizon merely because the row's
            # current validation contract changed.  The Provider Snapshot still
            # remains the minimum complete baseline.
            ready[dataset] = max(base, candidate)
        else:
            ready[dataset] = base

    through = min(ready.values())
    return TradingV2TrainingHorizon(
        provider_as_of_date=base,
        training_through_date=through,
        required_datasets=tuple(required),
        dataset_ready_through=ready,
    )


def merge_market_data_v2_fragments(
    fragments: Sequence[pd.DataFrame],
    *,
    primary_key: Iterable[str],
) -> pd.DataFrame:
    """Merge Provider Snapshot + ordered overlay fragments with deterministic precedence.

    ``fragments`` must be ordered oldest-to-newest.  Later fragments replace an
    earlier row only when the registry-declared primary key is identical.  This
    is required for current-vintage datasets such as ``TaiwanStockPriceAdj``:
    a newer verified overlay may legally restate an older adjusted-price row.
    """

    keys = tuple(str(value or "").strip() for value in primary_key if str(value or "").strip())
    if not keys:
        raise ValueError("Market Data V2 latest view 需要 registry primary_key_hint 才能安全 merge")
    if not fragments:
        return pd.DataFrame(columns=list(keys))

    prepared: list[pd.DataFrame] = []
    canonical_columns: list[str] = []
    seen_columns: set[str] = set()
    for rank, raw in enumerate(fragments):
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("Market Data V2 fragment 必須是 pandas.DataFrame")
        frame = raw.copy()
        missing = [key for key in keys if key not in frame.columns]
        if missing and not frame.empty:
            raise ValueError(f"Market Data V2 fragment 缺 primary key 欄位: {missing}")
        for column in frame.columns:
            name = str(column)
            if name not in seen_columns:
                seen_columns.add(name)
                canonical_columns.append(name)
        frame["__v2_source_rank"] = int(rank)
        frame["__v2_row_rank"] = range(len(frame))
        prepared.append(frame)

    combined = pd.concat(prepared, ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=canonical_columns)
    missing = [key for key in keys if key not in combined.columns]
    if missing:
        raise ValueError(f"Market Data V2 combined view 缺 primary key 欄位: {missing}")
    if combined.loc[:, list(keys)].isna().any().any():
        raise ValueError("Market Data V2 latest view primary key 不得含 null")

    combined = combined.sort_values(["__v2_source_rank", "__v2_row_rank"], kind="stable")
    combined = combined.drop_duplicates(list(keys), keep="last")
    combined = combined.sort_values(list(keys), kind="stable").reset_index(drop=True)
    return combined.loc[:, [column for column in canonical_columns if column in combined.columns]]


__all__ = [
    "TRADING_V2_VIEW_CONTRACT_VERSION",
    "TRADING_V2_VIEW_ROLE",
    "TRADING_V2_HISTORICAL_MEMBERSHIP_SOURCE",
    "TRADING_V2_CURRENT_EXECUTION_POOL_IS_TRAINING_SOURCE",
    "TradingV2TrainingHorizon",
    "trading_v2_view_contract_payload",
    "trading_v2_view_contract_fingerprint",
    "resolve_trading_v2_training_horizon",
    "merge_market_data_v2_fragments",
]
