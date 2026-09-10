"""V2-derived Trading six-column OHLCV compatibility materialization.

This is the only production producer for the transitional Trading CSV dataset
after the V2 execution cutover.  It performs zero provider calls: all market
data and current execution-pool evidence come from ``TradingMarketDataV2View``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256
from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET, FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.market_data_ohlcv_compatibility import build_market_data_v2_ohlcv_compatibility_frame
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_dataset_identity import build_trading_dataset_fingerprint, resolve_trading_dataset_member_tickers
from core.trading_identity import normalize_trading_ticker
from services.trading.market_data_consumer import resolve_trading_v2_current_execution_pool
from services.trading.market_data_v2_view import TradingMarketDataV2View

TRADING_V2_COMPATIBILITY_SCHEMA_VERSION = 1
TRADING_V2_COMPATIBILITY_ROLE = "trading_v2_legacy_ohlcv_compatibility_materialization"
TRADING_V2_COMPATIBILITY_REQUIRED_DATASETS = (
    FINMIND_ADJUSTED_PRICE_DATASET,
    FINMIND_RAW_PRICE_ARCHIVE_DATASET,
    "TaiwanStockMarketValue",
    "TaiwanStockTradingDate",
    "TaiwanStockInfo",
    "TaiwanStockDelisting",
)
TRADING_V2_COMPATIBILITY_STATE_RELATIVE_PATH = Path(
    "state/trading/market_data_v2/compatibility_materialization.json"
)


def materialize_trading_v2_compatibility_dataset(
    project_root: str | Path,
    *,
    required_tickers: Iterable[str] = (),
    view: TradingMarketDataV2View | None = None,
    market_date: str | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Materialize the transitional Trading CSV dataset from local V2 truth only."""

    root = Path(project_root).resolve()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    data_dir = Path(paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    local_view = view or TradingMarketDataV2View.open(root)
    horizon = local_view.training_horizon(required_datasets=TRADING_V2_COMPATIBILITY_REQUIRED_DATASETS)
    resolved_date = str(market_date or horizon.training_through_date)
    if resolved_date > horizon.training_through_date:
        raise RuntimeError(
            "Trading V2 compatibility market_date 超過 required dataset READY horizon；"
            f"requested={resolved_date}, ready_through={horizon.training_through_date}"
        )

    execution_tickers, execution_stats = resolve_trading_v2_current_execution_pool(
        local_view, market_date=resolved_date
    )
    required = sorted({normalize_trading_ticker(item) for item in required_tickers})
    retained = [
        normalize_trading_ticker(item)
        for item in resolve_trading_dataset_member_tickers(data_dir, required=False)
    ]
    target = list(dict.fromkeys([*execution_tickers, *required, *retained]))
    if not target:
        raise RuntimeError("Trading V2 compatibility 沒有任何 materialization target")

    prepared: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for index, ticker in enumerate(target, start=1):
        adjusted = local_view.read_dataset_frame(
            FINMIND_ADJUSTED_PRICE_DATASET,
            columns=("date", "stock_id", "open", "max", "min", "close"),
            data_id=ticker,
            end_date=resolved_date,
        )
        raw = local_view.read_dataset_frame(
            FINMIND_RAW_PRICE_ARCHIVE_DATASET,
            columns=("date", "stock_id", "Trading_Volume"),
            data_id=ticker,
            end_date=resolved_date,
        )
        frame = build_market_data_v2_ohlcv_compatibility_frame(
            adjusted, raw, stock_id=ticker, through_date=resolved_date
        )
        if ticker in set(execution_tickers) and (frame.empty or str(frame.iloc[-1]["Date"]) != resolved_date):
            raise RuntimeError(
                "Trading V2 compatibility actionable ticker 缺 target-date OHLCV；"
                f"ticker={ticker}, target={resolved_date}"
            )
        prepared[ticker] = frame.to_csv(index=False, lineterminator="\n")
        row_counts[ticker] = int(len(frame))
        if progress_callback is not None and (index % 100 == 0 or index == len(target)):
            progress_callback(f"[V2 compatibility] {index}/{len(target)} tickers")

    written = 0
    unchanged = 0
    for ticker in target:
        path = data_dir / f"{ticker}.csv"
        text = prepared[ticker]
        if path.is_file():
            try:
                if path.read_text(encoding="utf-8") == text:
                    unchanged += 1
                    continue
            except OSError as exc:
                raise RuntimeError(
                    "Trading V2 compatibility 無法讀取既有 CSV 以確認是否 unchanged；"
                    f"ticker={ticker}, file={path.name}"
                ) from exc
        atomic_write_text(path, text)
        written += 1

    dataset_fingerprint = build_trading_dataset_fingerprint(data_dir)
    if str(dataset_fingerprint.get("latest_data_date") or "") != resolved_date:
        raise RuntimeError(
            "Trading V2 compatibility dataset latest date 與 materialization date 不一致；"
            f"dataset={dataset_fingerprint.get('latest_data_date')}, target={resolved_date}"
        )
    source_view_identity = local_view.view_identity(required_datasets=TRADING_V2_COMPATIBILITY_REQUIRED_DATASETS)
    state = {
        "schema_version": TRADING_V2_COMPATIBILITY_SCHEMA_VERSION,
        "role": TRADING_V2_COMPATIBILITY_ROLE,
        "status": "READY",
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "market_date": resolved_date,
        "source": "trading_market_data_v2_historical_latest_view",
        "source_view_fingerprint": source_view_identity["view_fingerprint"],
        "provider_snapshot_fingerprint": local_view.archive.snapshot_fingerprint,
        "provider_as_of_date": local_view.archive.as_of_date,
        "provider_calls": 0,
        "required_datasets": list(TRADING_V2_COMPATIBILITY_REQUIRED_DATASETS),
        "current_execution_pool_tickers": execution_tickers,
        "current_execution_pool_ticker_count": len(execution_tickers),
        "current_execution_pool_stats": execution_stats,
        "required_position_tickers": required,
        "retained_history_tickers": retained,
        "target_ticker_count": len(target),
        "written_ticker_count": written,
        "unchanged_ticker_count": unchanged,
        "row_count": int(sum(row_counts.values())),
        "dataset_fingerprint": dataset_fingerprint,
        "data_dir": project_relative_display_path(data_dir, project_root=root),
    }
    state["state_fingerprint"] = canonical_json_sha256(state)
    atomic_write_json(root / TRADING_V2_COMPATIBILITY_STATE_RELATIVE_PATH, state)
    return state


__all__ = [
    "TRADING_V2_COMPATIBILITY_SCHEMA_VERSION",
    "TRADING_V2_COMPATIBILITY_ROLE",
    "TRADING_V2_COMPATIBILITY_REQUIRED_DATASETS",
    "TRADING_V2_COMPATIBILITY_STATE_RELATIVE_PATH",
    "resolve_trading_v2_current_execution_pool",
    "materialize_trading_v2_compatibility_dataset",
]
