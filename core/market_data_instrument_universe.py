"""Canonical provider-facing historical stock/ETF universe filtering for Market Data V2."""
from __future__ import annotations

import pandas as pd

from core.file_integrity import canonical_json_sha256

MARKET_DATA_HISTORICAL_UNIVERSE_CONTRACT_VERSION = 1
MARKET_DATA_EQUITY_MARKET_TYPES = frozenset({"twse", "tpex"})
MARKET_DATA_EXCLUDED_INDUSTRY_CATEGORIES = frozenset({"大盤", "index", "所有證券", "etn"})
MARKET_DATA_EXCLUDED_STOCK_IDS = frozenset({"TAIEX", "TPEx"})


def _normalize_category(value: object) -> str:
    return str(value or "").strip().casefold()


def historical_stock_etf_universe_contract_fingerprint() -> str:
    """Stable identity for provider StockInfo/delisting universe semantics."""

    return canonical_json_sha256(
        {
            "contract_version": MARKET_DATA_HISTORICAL_UNIVERSE_CONTRACT_VERSION,
            "sources": ["TaiwanStockInfo", "TaiwanStockDelisting"],
            "market_types": sorted(MARKET_DATA_EQUITY_MARKET_TYPES),
            "excluded_industry_categories": sorted(
                _normalize_category(value) for value in MARKET_DATA_EXCLUDED_INDUSTRY_CATEGORIES
            ),
            "excluded_stock_ids": sorted(MARKET_DATA_EXCLUDED_STOCK_IDS),
            "delisting_policy": "retain_non_special_stock_ids_conservatively",
        }
    )


def build_historical_stock_etf_universe(
    stock_info: pd.DataFrame,
    delisting: pd.DataFrame,
) -> tuple[str, ...]:
    """Return historical TWSE/TPEx stock+ETF ids without index/aggregate securities.

    Current/transition rows from ``TaiwanStockInfo`` are accepted whenever the
    instrument has a TWSE/TPEx row.  Explicit non-equity index/aggregate/ETN rows
    are excluded.  ``TaiwanStockDelisting`` is retained conservatively because it
    is the only source for some historical instruments and does not expose the
    same category field.
    """

    ids: set[str] = set()
    if not stock_info.empty:
        required = {"stock_id", "type", "industry_category"}
        missing = required.difference(stock_info.columns)
        if missing:
            raise ValueError(f"TaiwanStockInfo 缺少欄位: {sorted(missing)}")

        market_type = stock_info["type"].astype(str).str.strip().str.lower()
        categories = stock_info["industry_category"].map(_normalize_category)
        stock_ids = stock_info["stock_id"].astype(str).str.strip()
        excluded_categories = {_normalize_category(value) for value in MARKET_DATA_EXCLUDED_INDUSTRY_CATEGORIES}
        eligible = (
            market_type.isin(MARKET_DATA_EQUITY_MARKET_TYPES)
            & ~categories.isin(excluded_categories)
            & ~stock_ids.isin(MARKET_DATA_EXCLUDED_STOCK_IDS)
        )
        ids.update(value for value in stock_ids.loc[eligible].tolist() if value)

    if not delisting.empty:
        if "stock_id" not in delisting.columns:
            raise ValueError("TaiwanStockDelisting 缺少 stock_id")
        for raw in delisting["stock_id"].tolist():
            value = str(raw or "").strip()
            if value and value not in MARKET_DATA_EXCLUDED_STOCK_IDS:
                ids.add(value)

    normalized = tuple(sorted(ids))
    if not normalized:
        raise ValueError("無法由 TaiwanStockInfo + TaiwanStockDelisting 建立 historical stock/ETF universe")
    return normalized


__all__ = [
    "MARKET_DATA_HISTORICAL_UNIVERSE_CONTRACT_VERSION",
    "MARKET_DATA_EQUITY_MARKET_TYPES",
    "MARKET_DATA_EXCLUDED_INDUSTRY_CATEGORIES",
    "MARKET_DATA_EXCLUDED_STOCK_IDS",
    "historical_stock_etf_universe_contract_fingerprint",
    "build_historical_stock_etf_universe",
]
