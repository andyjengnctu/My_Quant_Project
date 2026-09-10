"""Production Trading consumers over the verified Market Data V2 view.

This module is the consumer-side bridge used after the Round-7 execution cutover.
It performs no provider calls and never reads the transitional Legacy Trading CSV
cache.  Price/volume field mapping remains owned by the cross-domain OHLCV
compatibility contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config.downloader import DOWNLOADER_MIN_MARKET_CAP, DOWNLOADER_MIN_VOLUME
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.runtime_utils import is_insufficient_data_error
from core.display import C_GRAY, C_RESET
from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.log_utils import write_issue_log
from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET, FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.market_data_ohlcv_compatibility import build_market_data_v2_ohlcv_compatibility_frame
from core.market_data_pool_contract import screen_daily_trading_execution_pool
from core.trading_data_dependencies import get_trading_data_dependency_spec
from core.trading_identity import normalize_trading_ticker
from services.trading.market_data_v2_view import TradingMarketDataV2View

TRADING_V2_CONSUMER_STATE_SCHEMA_VERSION = 1
TRADING_V2_CONSUMER_STATE_ROLE = "trading_v2_execution_consumer_state"
TRADING_V2_CONSUMER_STATE_RELATIVE_PATH = Path(
    "state/trading/market_data_v2/execution_consumer_state.json"
)


def _required_v2_datasets() -> tuple[str, ...]:
    return tuple(get_trading_data_dependency_spec("full_rule_based_no_dl").required_v2_datasets)


def _current_market_member_records(view: TradingMarketDataV2View, *, market_date: str) -> list[dict[str, object]]:
    members = tuple(view.daily_pit_market_members(market_date))
    member_set = set(members)
    info = view.read_dataset_frame(
        "TaiwanStockInfo",
        columns=("date", "stock_id", "type", "industry_category"),
    )
    if info.empty:
        raise RuntimeError("Trading V2 execution consumer 無 TaiwanStockInfo identity evidence")
    info = info.copy()
    info["stock_id"] = info["stock_id"].astype("string").str.strip()
    info["type"] = info["type"].astype("string").str.strip().str.lower()
    info["industry_category"] = info["industry_category"].astype("string").str.strip().str.casefold()
    listed = info.loc[info["stock_id"].isin(member_set) & info["type"].isin({"twse", "tpex"})]
    known = set(listed["stock_id"].astype(str))
    missing_identity = sorted(member_set - known)
    if missing_identity:
        raise RuntimeError(
            "Trading V2 current PIT member 缺 TWSE/TPEX TaiwanStockInfo identity；"
            f"count={len(missing_identity)} sample={missing_identity[:20]}"
        )
    etf_ids = set(listed.loc[listed["industry_category"] == "etf", "stock_id"].astype(str).tolist())
    return [{"stock_id": sid, "is_etf": sid in etf_ids} for sid in members]


def resolve_trading_v2_current_execution_pool(
    view: TradingMarketDataV2View,
    *,
    market_date: str,
) -> tuple[list[str], dict[str, int]]:
    """Resolve today's new-entry execution pool from V2-only local evidence."""

    members = _current_market_member_records(view, market_date=market_date)
    price = view.read_dataset_frame(
        FINMIND_ADJUSTED_PRICE_DATASET,
        columns=("date", "stock_id", "Trading_Volume"),
        start_date=market_date,
        end_date=market_date,
    ).rename(columns={"Trading_Volume": "trading_volume"})
    market_value = view.read_dataset_frame(
        "TaiwanStockMarketValue",
        columns=("date", "stock_id", "market_value"),
        start_date=market_date,
        end_date=market_date,
    )
    tickers, stats = screen_daily_trading_execution_pool(
        members,
        price_rows=price,
        market_value_rows=market_value,
        min_volume=DOWNLOADER_MIN_VOLUME,
        min_market_cap=DOWNLOADER_MIN_MARKET_CAP,
    )
    return sorted(normalize_trading_ticker(item) for item in tickers), dict(stats)


def build_trading_v2_ohlcv_frame(
    view: TradingMarketDataV2View,
    *,
    ticker: str,
    through_date: str,
) -> pd.DataFrame:
    sid = normalize_trading_ticker(ticker)
    adjusted = view.read_dataset_frame(
        FINMIND_ADJUSTED_PRICE_DATASET,
        columns=("date", "stock_id", "open", "max", "min", "close"),
        data_id=sid,
        end_date=through_date,
    )
    raw = view.read_dataset_frame(
        FINMIND_RAW_PRICE_ARCHIVE_DATASET,
        columns=("date", "stock_id", "Trading_Volume"),
        data_id=sid,
        end_date=through_date,
    )
    return build_market_data_v2_ohlcv_compatibility_frame(
        adjusted,
        raw,
        stock_id=sid,
        through_date=str(through_date),
    )


def compute_trading_v2_ohlcv_frame_sha256(frame: pd.DataFrame) -> str:
    payload = {
        "columns": [str(column) for column in frame.columns],
        "rows": frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
    }
    return canonical_json_sha256(payload)


def load_trading_v2_sanitized_ohlcv_frame(
    view: TradingMarketDataV2View,
    *,
    ticker: str,
    through_date: str,
    min_rows: int,
) -> pd.DataFrame:
    raw = build_trading_v2_ohlcv_frame(view, ticker=ticker, through_date=through_date)
    clean, _stats = sanitize_ohlcv_dataframe(raw, normalize_trading_ticker(ticker), min_rows=int(min_rows))
    cutoff = pd.Timestamp(through_date).normalize()
    if pd.Timestamp(clean.index.max()).normalize() > cutoff:
        raise RuntimeError(
            f"Trading V2 consumer data 含超過 cutoff 的日K: {ticker} "
            f"latest={pd.Timestamp(clean.index.max()).strftime('%Y-%m-%d')} > allowed={through_date}"
        )
    return clean


def _state_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / TRADING_V2_CONSUMER_STATE_RELATIVE_PATH


def _validate_consumer_state(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading V2 consumer state 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_V2_CONSUMER_STATE_SCHEMA_VERSION:
        raise ValueError("Trading V2 consumer state schema_version 不相容")
    if str(payload.get("role") or "") != TRADING_V2_CONSUMER_STATE_ROLE:
        raise ValueError("Trading V2 consumer state role 不合法")
    if str(payload.get("status") or "") != "READY":
        raise ValueError("Trading V2 consumer state 尚未 READY")
    for field in ("market_date", "source_view_fingerprint", "provider_snapshot_fingerprint", "state_fingerprint"):
        if not str(payload.get(field) or ""):
            raise ValueError(f"Trading V2 consumer state 缺少 {field}")
    for field in ("current_execution_pool_tickers", "required_position_tickers", "training_tickers"):
        values = payload.get(field)
        if not isinstance(values, list):
            raise ValueError(f"Trading V2 consumer state {field} 必須是 list")
        normalized = [normalize_trading_ticker(item) for item in values]
        if normalized != sorted(set(normalized)):
            raise ValueError(f"Trading V2 consumer state {field} 必須排序且去重")
    expected_training = sorted(
        set(payload["current_execution_pool_tickers"])
        | set(payload["required_position_tickers"])
        | set(payload.get("retained_training_tickers") or [])
    )
    if expected_training != list(payload["training_tickers"]):
        raise ValueError("Trading V2 consumer state training membership union 不一致")
    core = {key: value for key, value in payload.items() if key != "state_fingerprint"}
    if str(payload.get("state_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading V2 consumer state fingerprint 不一致")


def load_trading_v2_consumer_state(
    project_root: str | Path,
    *,
    required: bool = True,
    verify_current_view: bool = False,
) -> dict[str, Any] | None:
    root = Path(project_root).resolve()
    path = _state_path(root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading V2 execution consumer state 尚未建立；請先更新 Trading 資料")
        return None
    payload = load_json_strict(path)
    _validate_consumer_state(payload)
    if verify_current_view:
        view = TradingMarketDataV2View.open(root)
        identity = view.view_identity(required_datasets=_required_v2_datasets())
        if str(payload.get("source_view_fingerprint") or "") != str(identity.get("view_fingerprint") or ""):
            raise RuntimeError("Trading V2 execution consumer state 與目前 verified V2 view identity 不一致")
    return payload


def get_trading_v2_consumer_state_sha256(project_root: str | Path) -> str:
    path = _state_path(project_root)
    if not path.is_file():
        raise FileNotFoundError("Trading V2 execution consumer state 尚未建立")
    return compute_file_sha256(path)


def publish_trading_v2_consumer_state(
    project_root: str | Path,
    *,
    market_date: str,
    required_position_tickers: Iterable[str] = (),
    retained_training_tickers: Iterable[str] = (),
    view: TradingMarketDataV2View | None = None,
) -> dict[str, Any]:
    """Publish local V2 consumer membership/lineage after a successful V2 update."""

    root = Path(project_root).resolve()
    local_view = view or TradingMarketDataV2View.open(root)
    required_datasets = _required_v2_datasets()
    horizon = local_view.training_horizon(required_datasets=required_datasets)
    date_text = str(market_date)
    if date_text > str(horizon.training_through_date):
        raise RuntimeError(
            "Trading V2 consumer state market_date 超過 required dataset READY horizon；"
            f"requested={date_text}, ready_through={horizon.training_through_date}"
        )
    execution_tickers, execution_stats = resolve_trading_v2_current_execution_pool(local_view, market_date=date_text)
    required = sorted({normalize_trading_ticker(item) for item in required_position_tickers})
    prior = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    retained = set(normalize_trading_ticker(item) for item in retained_training_tickers)
    if prior is not None:
        retained.update(normalize_trading_ticker(item) for item in prior.get("training_tickers") or [])
    retained.difference_update(execution_tickers)
    retained.difference_update(required)
    retained_sorted = sorted(retained)
    training = sorted(set(execution_tickers) | set(required) | set(retained_sorted))
    if not training:
        raise RuntimeError("Trading V2 consumer state 沒有任何 training/consumer ticker")

    identity = local_view.view_identity(required_datasets=required_datasets)
    state: dict[str, Any] = {
        "schema_version": TRADING_V2_CONSUMER_STATE_SCHEMA_VERSION,
        "role": TRADING_V2_CONSUMER_STATE_ROLE,
        "status": "READY",
        "market_date": date_text,
        "source": "trading_market_data_v2_historical_latest_view",
        "source_view_fingerprint": identity["view_fingerprint"],
        "provider_snapshot_fingerprint": local_view.archive.snapshot_fingerprint,
        "provider_as_of_date": local_view.archive.as_of_date,
        "required_v2_datasets": list(required_datasets),
        "provider_calls": 0,
        "current_execution_pool_tickers": execution_tickers,
        "current_execution_pool_ticker_count": len(execution_tickers),
        "current_execution_pool_stats": execution_stats,
        "required_position_tickers": required,
        "retained_training_tickers": retained_sorted,
        "training_tickers": training,
        "training_ticker_count": len(training),
    }
    state["state_fingerprint"] = canonical_json_sha256(state)
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, state)
    return state


def load_trading_v2_optimizer_raw_data(data_dir, required_min_rows, output_dir, *, verbose=True):
    """Optimizer loader with the legacy loader signature, backed only by V2 view."""

    root = Path(data_dir).resolve()
    state = load_trading_v2_consumer_state(root, required=True, verify_current_view=True)
    view = TradingMarketDataV2View.open(root)
    through_date = str(state["market_date"])
    tickers = list(state["training_tickers"])
    cache: dict[str, pd.DataFrame] = {}
    issues: list[str] = []
    for index, ticker in enumerate(tickers, start=1):
        try:
            cache[ticker] = load_trading_v2_sanitized_ohlcv_frame(
                view,
                ticker=ticker,
                through_date=through_date,
                min_rows=int(required_min_rows),
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                issues.append(f"{ticker}: {type(exc).__name__}: {exc}")
                continue
            raise RuntimeError(f"Trading V2 optimizer raw data 讀取失敗: ticker={ticker} | {type(exc).__name__}: {exc}") from exc
        if bool(verbose) and (index % 50 == 0 or index == len(tickers)):
            print(f"{C_GRAY}   進度: [{index}/{len(tickers)}] 已讀取 Trading V2 stocks...{C_RESET}", end="\r")
    if not cache:
        raise RuntimeError("Trading V2 optimizer raw data 無任何可用標的")
    if issues:
        write_issue_log("optimizer_v2_load_issues", issues, log_dir=output_dir)
    return cache


__all__ = [
    "TRADING_V2_CONSUMER_STATE_SCHEMA_VERSION",
    "TRADING_V2_CONSUMER_STATE_ROLE",
    "TRADING_V2_CONSUMER_STATE_RELATIVE_PATH",
    "resolve_trading_v2_current_execution_pool",
    "build_trading_v2_ohlcv_frame",
    "compute_trading_v2_ohlcv_frame_sha256",
    "load_trading_v2_sanitized_ohlcv_frame",
    "load_trading_v2_consumer_state",
    "get_trading_v2_consumer_state_sha256",
    "publish_trading_v2_consumer_state",
    "load_trading_v2_optimizer_raw_data",
]
