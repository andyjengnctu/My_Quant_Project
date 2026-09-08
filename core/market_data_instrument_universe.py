"""Canonical provider-facing historical stock/ETF universe filtering for Market Data V2."""
from __future__ import annotations

import pandas as pd

from core.file_integrity import canonical_json_sha256

MARKET_DATA_HISTORICAL_UNIVERSE_CONTRACT_VERSION = 1
MARKET_DATA_HISTORICAL_MARKET_STATE_GUARD_CONTRACT_VERSION = 1
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
    """Return the provider-bootstrap stock/ETF archive superset.

    This tuple is an *archive request pool*, not point-in-time daily eligibility.
    A stock that later moves from emerging to TWSE/TPEx remains in the archive
    pool so its complete provider history can be preserved.  Research daily
    membership must additionally apply ``build_historical_market_state_guard``
    and must never infer historical eligibility from this timeless pool alone.

    Explicit non-equity index/aggregate/ETN rows are excluded.
    ``TaiwanStockDelisting`` is retained conservatively because it is the only
    source for some historical instruments and does not expose the same category
    field.
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


def build_historical_market_state_guard(
    stock_info: pd.DataFrame,
    *,
    historical_instruments: tuple[str, ...] | list[str] | set[str],
) -> dict[str, str]:
    """Return stock_id -> last-emerging-date exclusion boundaries.

    FinMind ``TaiwanStockInfo`` preserves a completed emerging row when an
    instrument later moves to TWSE/TPEx.  The emerging row's ``date`` stops at
    the date the instrument leaves emerging; the current TWSE/TPEx row's date is
    current-vintage and therefore cannot be interpreted as the listing start.

    For any instrument already admitted to the canonical historical archive pool
    (including delisting-only instruments) that has an emerging row, Research
    must reject raw market rows on or before the final emerging date.
    This reconstructs the market state that was observable on the historical
    date without using the later transition as permission to include earlier
    emerging rows.  Listed-only stocks have no lower exclusion boundary.
    """

    if stock_info.empty:
        raise ValueError("TaiwanStockInfo market-state guard evidence 不可為空")
    required = {"stock_id", "type", "industry_category", "date"}
    missing = required.difference(stock_info.columns)
    if missing:
        raise ValueError(f"TaiwanStockInfo market-state guard 缺少欄位: {sorted(missing)}")

    frame = stock_info.loc[:, ["stock_id", "type", "industry_category", "date"]].copy()
    frame["stock_id"] = frame["stock_id"].astype(str).str.strip()
    frame["type"] = frame["type"].astype(str).str.strip().str.lower()
    frame["category"] = frame["industry_category"].map(_normalize_category)
    frame["parsed_date"] = pd.to_datetime(frame["date"], errors="coerce")

    pool = {str(value or "").strip() for value in historical_instruments}
    pool.discard("")
    if not pool:
        raise ValueError("historical market-state guard instrument pool 不可為空")

    boundaries: dict[str, str] = {}
    for stock_id in sorted(pool):
        emerging = frame.loc[(frame["stock_id"] == stock_id) & (frame["type"] == "emerging")]
        if emerging.empty:
            continue
        if emerging["parsed_date"].isna().any():
            raise ValueError(
                "TaiwanStockInfo transition row 缺合法 date，無法重建 PIT market state: "
                f"stock_id={stock_id}"
            )
        final_emerging = emerging["parsed_date"].max().strftime("%Y-%m-%d")
        boundaries[stock_id] = final_emerging
    return boundaries


def historical_market_state_guard_fingerprint(boundaries: dict[str, str]) -> str:
    normalized = {str(key): str(value) for key, value in sorted(boundaries.items())}
    return canonical_json_sha256(
        {
            "contract_version": MARKET_DATA_HISTORICAL_MARKET_STATE_GUARD_CONTRACT_VERSION,
            "source": "TaiwanStockInfo",
            "rule": "historical_pool_rows_allowed_only_after_final_emerging_date_when_transition_history_exists",
            "transition_excluded_through": normalized,
        }
    )


def is_historical_market_state_eligible(
    *,
    stock_id: object,
    date_value: object,
    transition_excluded_through: dict[str, str],
) -> bool:
    sid = str(stock_id or "").strip()
    if not sid:
        return False
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return False
    date_text = parsed.strftime("%Y-%m-%d")
    boundary = transition_excluded_through.get(sid)
    return boundary is None or date_text > str(boundary)


__all__ = [
    "MARKET_DATA_HISTORICAL_UNIVERSE_CONTRACT_VERSION",
    "MARKET_DATA_HISTORICAL_MARKET_STATE_GUARD_CONTRACT_VERSION",
    "MARKET_DATA_EQUITY_MARKET_TYPES",
    "MARKET_DATA_EXCLUDED_INDUSTRY_CATEGORIES",
    "MARKET_DATA_EXCLUDED_STOCK_IDS",
    "historical_stock_etf_universe_contract_fingerprint",
    "build_historical_stock_etf_universe",
    "build_historical_market_state_guard",
    "historical_market_state_guard_fingerprint",
    "is_historical_market_state_eligible",
]
