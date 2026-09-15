"""Canonical Market Data V2 pool-layer contract.

The neutral daily PIT market universe describes which instruments exist on a
historical date.  Model context is a date-local projection of that universe and
must never be inferred from the latest Trading execution pool.  Trading
execution eligibility is a separate, date-local new-entry mask.

This module is pure: no provider I/O, no domain storage and no model-specific
scientific policy.  Model-specific context filters remain owned by the model's
declarative contract; this owner only enforces the layering invariant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from core.file_integrity import canonical_json_sha256

MARKET_DATA_POOL_LAYER_CONTRACT_VERSION = 2
MARKET_DATA_MODEL_CONTEXT_POOL_ROLE = "date_local_pit_projection"
MARKET_DATA_TRADING_EXECUTION_POOL_ROLE = "date_local_new_entry_eligibility"
MARKET_DATA_POOL_SOURCE_ROLE = "neutral_daily_pit_market_universe"


@dataclass(frozen=True)
class TradingExecutionPoolStats:
    listed_count: int
    price_exact_date_count: int
    market_value_usable_count: int
    listed_without_exact_price_count: int
    listed_with_exact_price_count: int
    below_min_volume_count: int
    high_volume_count: int
    high_volume_etf_count: int
    high_volume_stock_count: int
    market_cap_pass_stock_count: int
    market_cap_below_min_stock_count: int
    qualified_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "listed_count": int(self.listed_count),
            "price_exact_date_count": int(self.price_exact_date_count),
            "market_value_usable_count": int(self.market_value_usable_count),
            "listed_without_exact_price_count": int(self.listed_without_exact_price_count),
            "listed_with_exact_price_count": int(self.listed_with_exact_price_count),
            "below_min_volume_count": int(self.below_min_volume_count),
            "high_volume_count": int(self.high_volume_count),
            "high_volume_etf_count": int(self.high_volume_etf_count),
            "high_volume_stock_count": int(self.high_volume_stock_count),
            "market_cap_pass_stock_count": int(self.market_cap_pass_stock_count),
            "market_cap_below_min_stock_count": int(self.market_cap_below_min_stock_count),
            "qualified_count": int(self.qualified_count),
        }


def market_data_pool_layer_contract_payload() -> dict[str, object]:
    return {
        "contract_version": MARKET_DATA_POOL_LAYER_CONTRACT_VERSION,
        "layer_order": [
            MARKET_DATA_POOL_SOURCE_ROLE,
            MARKET_DATA_MODEL_CONTEXT_POOL_ROLE,
            MARKET_DATA_TRADING_EXECUTION_POOL_ROLE,
        ],
        "model_context_source": MARKET_DATA_POOL_SOURCE_ROLE,
        "model_context_consumer_specific_projection_required": True,
        "model_context_may_depend_on_current_execution_pool": False,
        "current_execution_pool_may_define_historical_training_universe": False,
        "trading_execution_pool_source": MARKET_DATA_POOL_SOURCE_ROLE,
        "trading_execution_pool_scope": "new_entry_only",
        "trading_price_input_freshness": "exact_target",
        "trading_market_value_input_freshness": "caller_resolved_scan_freshness_policy",
        "execution_pool_must_not_reduce_model_context": True,
        "holdings_data_retention_independent_of_current_execution_pool": True,
    }


def market_data_pool_layer_contract_fingerprint() -> str:
    return canonical_json_sha256(market_data_pool_layer_contract_payload())


def _normalize_unique_members(values: Iterable[object], *, field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            raise ValueError(f"{field} 不得包含空白 ticker")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError(f"{field} 不可為空")
    return tuple(normalized)


def project_daily_model_context_pool(
    daily_pit_market_members: Iterable[object],
    *,
    requested_members: Iterable[object],
) -> tuple[str, ...]:
    """Project one date's model context from that date's PIT market universe.

    ``requested_members`` is the explicit model-specific projection already
    decided by the model/scientific contract.  This function does not choose a
    default context; it only guarantees that context cannot contain a ticker
    that was absent from that date's PIT market universe.
    """

    pit_members = _normalize_unique_members(
        daily_pit_market_members,
        field="daily PIT market universe",
    )
    requested = _normalize_unique_members(requested_members, field="model context pool")
    pit_set = set(pit_members)
    outside = sorted(set(requested) - pit_set)
    if outside:
        raise ValueError(
            "Model context pool 必須是同日 daily PIT market universe 的 projection；"
            f"outside_count={len(outside)} sample={outside[:20]}"
        )
    requested_set = set(requested)
    return tuple(stock_id for stock_id in pit_members if stock_id in requested_set)


def _normalized_numeric_map(
    frame: pd.DataFrame,
    *,
    value_column: str,
    dataset_label: str,
) -> dict[str, float]:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{dataset_label} frame 不可為空")
    required = {"stock_id", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset_label} frame 缺少欄位: {sorted(missing)}")
    work = frame.loc[:, ["stock_id", value_column]].copy()
    work["stock_id"] = work["stock_id"].astype("string").str.strip()
    if work["stock_id"].isna().any() or (work["stock_id"] == "").any():
        raise ValueError(f"{dataset_label} frame 存在空白 stock_id")
    duplicated = sorted(
        work.loc[work["stock_id"].duplicated(keep=False), "stock_id"].astype(str).unique().tolist()
    )
    if duplicated:
        raise ValueError(f"{dataset_label} frame 存在重複 stock_id: {duplicated[:10]}")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    if work[value_column].isna().any():
        raise ValueError(f"{dataset_label} frame 存在非數值 {value_column}")
    return dict(zip(work["stock_id"].astype(str), work[value_column].astype(float)))


def screen_daily_trading_execution_pool(
    daily_market_members: Iterable[Mapping[str, object]],
    *,
    price_rows: pd.DataFrame,
    market_value_rows: pd.DataFrame,
    min_volume: float,
    min_market_cap: float,
) -> tuple[list[str], dict[str, int]]:
    """Apply the existing Trading new-entry liquidity/value mask to one date.

    The caller supplies the date-local market membership.  Today that membership
    may still come from the legacy current TWSE/TPEX source during migration;
    future V2 historical/latest views can supply the neutral daily PIT universe.
    Threshold values remain declarative caller inputs rather than constants here.
    """

    volume_floor = float(min_volume)
    market_cap_floor = float(min_market_cap)
    if volume_floor < 0 or market_cap_floor < 0:
        raise ValueError("Trading execution pool threshold 不得為負數")

    universe_by_sid: dict[str, bool] = {}
    for item in daily_market_members:
        sid = str(item.get("stock_id") or "").strip()
        if not sid:
            raise ValueError("Trading daily market universe 存在空白 stock_id")
        is_etf = bool(item.get("is_etf"))
        prior = universe_by_sid.get(sid)
        if prior is not None and prior != is_etf:
            raise ValueError(f"Trading daily market universe 同一 stock_id ETF identity 不一致: {sid}")
        universe_by_sid[sid] = is_etf
    if not universe_by_sid:
        raise ValueError("Trading daily market universe 不可為空")

    volume_by_sid = _normalized_numeric_map(
        price_rows,
        value_column="trading_volume",
        dataset_label="Trading execution price",
    )
    market_value_by_sid = _normalized_numeric_map(
        market_value_rows,
        value_column="market_value",
        dataset_label="Trading execution market value",
    )

    qualified: list[str] = []
    missing_market_value_for_high_volume_stock: list[str] = []
    listed_without_exact_price = 0
    below_min_volume_count = 0
    high_volume_count = 0
    high_volume_etf_count = 0
    high_volume_stock_count = 0
    market_cap_pass_stock_count = 0
    market_cap_below_min_stock_count = 0
    for sid, is_etf in universe_by_sid.items():
        volume = volume_by_sid.get(sid)
        if volume is None:
            # A listed but non-traded/suspended symbol is not actionable today.
            listed_without_exact_price += 1
            continue
        if volume < volume_floor:
            below_min_volume_count += 1
            continue
        high_volume_count += 1
        if is_etf:
            high_volume_etf_count += 1
            qualified.append(sid)
            continue
        high_volume_stock_count += 1
        cap = market_value_by_sid.get(sid)
        if cap is None or cap <= 0:
            missing_market_value_for_high_volume_stock.append(sid)
            continue
        if cap >= market_cap_floor:
            market_cap_pass_stock_count += 1
            qualified.append(sid)
        else:
            market_cap_below_min_stock_count += 1

    if missing_market_value_for_high_volume_stock:
        sample = ",".join(missing_market_value_for_high_volume_stock[:20])
        raise RuntimeError(
            "Trading execution pool 對高成交量股票缺少符合 Scan freshness policy 的 market_value，依保守原則中止；"
            f"count={len(missing_market_value_for_high_volume_stock)} sample={sample}"
        )

    stats = TradingExecutionPoolStats(
        listed_count=len(universe_by_sid),
        price_exact_date_count=len(volume_by_sid),
        market_value_usable_count=len(market_value_by_sid),
        listed_without_exact_price_count=int(listed_without_exact_price),
        listed_with_exact_price_count=len(universe_by_sid) - int(listed_without_exact_price),
        below_min_volume_count=int(below_min_volume_count),
        high_volume_count=int(high_volume_count),
        high_volume_etf_count=int(high_volume_etf_count),
        high_volume_stock_count=int(high_volume_stock_count),
        market_cap_pass_stock_count=int(market_cap_pass_stock_count),
        market_cap_below_min_stock_count=int(market_cap_below_min_stock_count),
        qualified_count=len(qualified),
    )
    return qualified, stats.as_dict()


__all__ = [
    "MARKET_DATA_POOL_LAYER_CONTRACT_VERSION",
    "MARKET_DATA_MODEL_CONTEXT_POOL_ROLE",
    "MARKET_DATA_TRADING_EXECUTION_POOL_ROLE",
    "MARKET_DATA_POOL_SOURCE_ROLE",
    "TradingExecutionPoolStats",
    "market_data_pool_layer_contract_payload",
    "market_data_pool_layer_contract_fingerprint",
    "project_daily_model_context_pool",
    "screen_daily_trading_execution_pool",
]
