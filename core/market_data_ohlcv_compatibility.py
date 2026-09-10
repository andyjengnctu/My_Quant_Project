"""Cross-domain six-column OHLCV compatibility contract for Market Data V2.

The compatibility view is not a provider truth.  Adjusted OHLC comes only from
FinMind ``TaiwanStockPriceAdj`` while Volume comes from raw
``TaiwanStockPrice.Trading_Volume``.  Research and Trading may materialize
different time views, but the field mapping itself is shared.
"""
from __future__ import annotations

import math
import pandas as pd

from core.market_data_adjusted_price_invariance import PROVIDER_VOLUME_FIELD
from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET, FINMIND_RAW_PRICE_ARCHIVE_DATASET

MARKET_DATA_V2_COMPAT_OUTPUT_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
MARKET_DATA_V2_COMPAT_PRICE_FIELD_MAPPING = {
    "Open": "open",
    "High": "max",
    "Low": "min",
    "Close": "close",
}
MARKET_DATA_V2_COMPAT_VOLUME_FIELD_MAPPING = {"Volume": PROVIDER_VOLUME_FIELD}
MARKET_DATA_V2_COMPAT_PRICE_DATASET = FINMIND_ADJUSTED_PRICE_DATASET
MARKET_DATA_V2_COMPAT_VOLUME_DATASET = FINMIND_RAW_PRICE_ARCHIVE_DATASET


def build_market_data_v2_ohlcv_compatibility_frame(
    adjusted: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    stock_id: str,
    through_date: str,
) -> pd.DataFrame:
    """Build one ticker's deterministic six-column compatibility frame.

    The function performs no I/O and never synthesizes a missing provider row.
    PriceAdj dates without raw Volume are rejected instead of silently carrying
    forward or borrowing another day's volume.
    """

    sid = str(stock_id or "").strip()
    if not sid:
        raise ValueError("Market Data V2 compatibility stock_id 不可為空")
    cutoff = pd.Timestamp(through_date).normalize()

    price_needed = {"date", "stock_id", *MARKET_DATA_V2_COMPAT_PRICE_FIELD_MAPPING.values()}
    volume_field = MARKET_DATA_V2_COMPAT_VOLUME_FIELD_MAPPING["Volume"]
    volume_needed = {"date", "stock_id", volume_field}
    price_missing = price_needed.difference(adjusted.columns)
    volume_missing = volume_needed.difference(volume.columns)
    if price_missing:
        raise ValueError(f"Market Data V2 PriceAdj compatibility 缺欄位: {sorted(price_missing)}")
    if volume_missing:
        raise ValueError(f"Market Data V2 raw Volume compatibility 缺欄位: {sorted(volume_missing)}")

    price = adjusted.loc[:, ["date", "stock_id", "open", "max", "min", "close"]].copy()
    raw = volume.loc[:, ["date", "stock_id", volume_field]].copy()
    for frame in (price, raw):
        frame["stock_id"] = frame["stock_id"].astype("string").str.strip()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    price = price.loc[(price["stock_id"] == sid) & price["date"].notna() & (price["date"] <= cutoff)].copy()
    raw = raw.loc[(raw["stock_id"] == sid) & raw["date"].notna() & (raw["date"] <= cutoff)].copy()

    for field in ("open", "max", "min", "close"):
        price[field] = pd.to_numeric(price[field], errors="coerce")
    raw[volume_field] = pd.to_numeric(raw[volume_field], errors="coerce")
    if price.empty:
        raise ValueError(f"Market Data V2 compatibility PriceAdj 對 {sid} 無歷史資料")
    if price[["open", "max", "min", "close"]].isna().any().any():
        raise ValueError(f"Market Data V2 compatibility PriceAdj 對 {sid} 含非數值 OHLC")
    if raw[volume_field].isna().any():
        raise ValueError(f"Market Data V2 compatibility raw Volume 對 {sid} 含非數值")
    for frame, label in ((price, "PriceAdj"), (raw, "raw Volume")):
        if frame.duplicated(subset=["date", "stock_id"]).any():
            raise ValueError(f"Market Data V2 compatibility {label} 對 {sid} 存在重複 date key")

    merged = price.merge(raw, on=["date", "stock_id"], how="left", validate="one_to_one")
    if merged[volume_field].isna().any():
        missing_dates = merged.loc[merged[volume_field].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"Market Data V2 compatibility PriceAdj 日期缺 raw Volume；ticker={sid} "
            f"missing={missing_dates[:20]} count={len(missing_dates)}"
        )
    numeric = merged[["open", "max", "min", "close", volume_field]].to_numpy(dtype=float)
    if not all(math.isfinite(float(value)) for value in numeric.ravel()):
        raise ValueError(f"Market Data V2 compatibility 對 {sid} 含非有限數值")

    out = merged.rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        volume_field: "Volume",
    }).loc[:, list(MARKET_DATA_V2_COMPAT_OUTPUT_COLUMNS)]
    out = out.sort_values("Date", kind="stable").reset_index(drop=True)
    rounded = out["Volume"].round()
    if bool((out["Volume"] == rounded).all()):
        out["Volume"] = rounded.astype("int64")
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


__all__ = [
    "MARKET_DATA_V2_COMPAT_OUTPUT_COLUMNS",
    "MARKET_DATA_V2_COMPAT_PRICE_FIELD_MAPPING",
    "MARKET_DATA_V2_COMPAT_VOLUME_FIELD_MAPPING",
    "MARKET_DATA_V2_COMPAT_PRICE_DATASET",
    "MARKET_DATA_V2_COMPAT_VOLUME_DATASET",
    "build_market_data_v2_ohlcv_compatibility_frame",
]
