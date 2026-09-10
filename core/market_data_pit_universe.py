"""Neutral Market Data V2 point-in-time market-universe contract.

This module owns date-specific market membership shared by Research and Trading.
It is intentionally independent of Research cutoffs, strategy eligibility, model
context, and execution masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from core.file_integrity import canonical_json_sha256
from core.market_data_instrument_universe import is_historical_market_state_eligible

MARKET_DATA_DAILY_PIT_UNIVERSE_CONTRACT_VERSION = 1
MARKET_DATA_DAILY_PIT_UNIVERSE_CONTRACT_ID = "market_data_v2_daily_pit_market_universe_v1"
MARKET_DATA_DAILY_PIT_UNIVERSE_SOURCE_DATASET = "TaiwanStockPrice"
MARKET_DATA_DAILY_PIT_UNIVERSE_TRADING_CALENDAR_DATASET = "TaiwanStockTradingDate"
MARKET_DATA_DAILY_PIT_UNIVERSE_MARKET_STATE_DATASET = "TaiwanStockInfo"


@dataclass(frozen=True)
class DailyPitMarketUniverseGuard:
    historical_instruments: frozenset[str]
    transition_excluded_through: Mapping[str, str]
    trading_dates: frozenset[str] | None = None
    provider_as_of_date: str | None = None

    def allows(self, *, stock_id: object, date_value: object) -> bool:
        sid = str(stock_id or "").strip()
        parsed = pd.to_datetime(date_value, errors="coerce")
        if not sid or pd.isna(parsed) or sid not in self.historical_instruments:
            return False
        date_text = parsed.strftime("%Y-%m-%d")
        if self.provider_as_of_date is not None and date_text > str(self.provider_as_of_date):
            return False
        if self.trading_dates is not None and date_text not in self.trading_dates:
            return False
        return is_historical_market_state_eligible(
            stock_id=sid,
            date_value=date_text,
            transition_excluded_through=self.transition_excluded_through,
        )


def daily_pit_market_universe_contract_payload() -> dict[str, object]:
    return {
        "contract_version": MARKET_DATA_DAILY_PIT_UNIVERSE_CONTRACT_VERSION,
        "contract_id": MARKET_DATA_DAILY_PIT_UNIVERSE_CONTRACT_ID,
        "source_dataset": MARKET_DATA_DAILY_PIT_UNIVERSE_SOURCE_DATASET,
        "trading_calendar_dataset": MARKET_DATA_DAILY_PIT_UNIVERSE_TRADING_CALENDAR_DATASET,
        "market_state_dataset": MARKET_DATA_DAILY_PIT_UNIVERSE_MARKET_STATE_DATASET,
        "membership_rule": (
            "raw_price_date_presence AND historical_archive_instrument AND taiwan_trading_date "
            "AND after_final_emerging_date_when_transition_history_exists"
        ),
        "provider_as_of_cap_required": True,
        "domain_specific_cutoff_forbidden": True,
        "strategy_execution_filters_excluded": True,
        "model_context_filters_excluded": True,
    }


def daily_pit_market_universe_contract_fingerprint() -> str:
    return canonical_json_sha256(daily_pit_market_universe_contract_payload())


def make_daily_pit_market_universe_guard(
    *,
    historical_instruments: Iterable[str],
    transition_excluded_through: Mapping[str, str],
    trading_dates: Iterable[str] | None = None,
    provider_as_of_date: str | None = None,
) -> DailyPitMarketUniverseGuard:
    pool = frozenset(str(value or "").strip() for value in historical_instruments if str(value or "").strip())
    if not pool:
        raise ValueError("Market Data V2 historical instrument pool 不可為空")
    dates = None
    if trading_dates is not None:
        dates = frozenset(str(value) for value in trading_dates if str(value))
        if not dates:
            raise ValueError("Market Data V2 trading calendar 不可為空")
    as_of = None if provider_as_of_date is None else str(provider_as_of_date)
    if as_of is not None and pd.isna(pd.to_datetime(as_of, errors="coerce")):
        raise ValueError("Market Data V2 provider_as_of_date 不合法")
    return DailyPitMarketUniverseGuard(
        historical_instruments=pool,
        transition_excluded_through={str(key): str(value) for key, value in transition_excluded_through.items()},
        trading_dates=dates,
        provider_as_of_date=as_of,
    )


def filter_daily_pit_market_universe(
    raw_price_rows: pd.DataFrame,
    *,
    guard: DailyPitMarketUniverseGuard,
) -> pd.DataFrame:
    required = {"date", "stock_id"}
    missing = required.difference(raw_price_rows.columns)
    if missing:
        raise ValueError(f"TaiwanStockPrice PIT universe evidence 缺少欄位: {sorted(missing)}")
    frame = raw_price_rows.loc[:, ["date", "stock_id"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["stock_id"] = frame["stock_id"].astype(str).str.strip()
    frame = frame.loc[
        [
            guard.allows(stock_id=stock_id, date_value=date_value)
            for date_value, stock_id in frame[["date", "stock_id"]].itertuples(index=False, name=None)
        ]
    ]
    frame = frame.drop_duplicates(["date", "stock_id"]).sort_values(["date", "stock_id"], kind="stable")
    return frame.reset_index(drop=True)


def build_daily_pit_market_universe(
    raw_price_rows: pd.DataFrame,
    *,
    historical_instruments: Iterable[str],
    transition_excluded_through: Mapping[str, str],
    trading_dates: Iterable[str] | None = None,
    provider_as_of_date: str | None = None,
) -> pd.DataFrame:
    guard = make_daily_pit_market_universe_guard(
        historical_instruments=historical_instruments,
        transition_excluded_through=transition_excluded_through,
        trading_dates=trading_dates,
        provider_as_of_date=provider_as_of_date,
    )
    return filter_daily_pit_market_universe(raw_price_rows, guard=guard)


__all__ = [
    "MARKET_DATA_DAILY_PIT_UNIVERSE_CONTRACT_VERSION",
    "MARKET_DATA_DAILY_PIT_UNIVERSE_CONTRACT_ID",
    "MARKET_DATA_DAILY_PIT_UNIVERSE_SOURCE_DATASET",
    "MARKET_DATA_DAILY_PIT_UNIVERSE_TRADING_CALENDAR_DATASET",
    "MARKET_DATA_DAILY_PIT_UNIVERSE_MARKET_STATE_DATASET",
    "DailyPitMarketUniverseGuard",
    "daily_pit_market_universe_contract_payload",
    "daily_pit_market_universe_contract_fingerprint",
    "make_daily_pit_market_universe_guard",
    "filter_daily_pit_market_universe",
    "build_daily_pit_market_universe",
]
