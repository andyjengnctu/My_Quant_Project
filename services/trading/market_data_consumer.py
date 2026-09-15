"""Production Trading consumers over the verified Market Data V2 view.

This module is the terminal V2-only Trading consumer seam after the Round-8
Legacy retirement. It performs no provider calls and never reads retired Legacy
Trading CSV/snapshot artifacts. Price/volume field mapping remains owned by the
cross-domain OHLCV compatibility contract.
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
from core.market_data_dataset_readiness import market_data_contract_requires_target_freshness
from core.market_data_freshness_contract import get_market_data_freshness_contract
from core.market_data_ohlcv_compatibility import build_market_data_v2_ohlcv_compatibility_frame
from core.market_data_instrument_universe import build_current_stock_etf_universe
from core.market_data_pool_contract import screen_daily_trading_execution_pool
from core.trading_data_dependencies import get_trading_data_dependency_spec
from core.trading_identity import normalize_trading_ticker
from core.trading_policy import get_trading_strategy_profile
from services.trading.market_data_v2_view import TradingMarketDataV2View
from services.trading.market_data_dataset_state import load_market_data_dataset_state

TRADING_V2_CONSUMER_STATE_SCHEMA_VERSION = 1
TRADING_V2_CONSUMER_STATE_ROLE = "trading_v2_execution_consumer_state"
TRADING_V2_CONSUMER_STATE_RELATIVE_PATH = Path(
    "state/trading/market_data_v2/execution_consumer_state.json"
)


def _required_v2_datasets() -> tuple[str, ...]:
    strategy_id = get_trading_strategy_profile().strategy_id
    return tuple(get_trading_data_dependency_spec(strategy_id).required_v2_datasets)


def _current_market_member_records(
    view: TradingMarketDataV2View,
    *,
    market_date: str,
) -> tuple[list[dict[str, object]], int]:
    members = tuple(view.daily_pit_market_members(market_date))
    member_set = set(members)
    info = view.read_dataset_frame(
        "TaiwanStockInfo",
        columns=("date", "stock_id", "type", "industry_category"),
    )
    if info.empty:
        raise RuntimeError("Trading V2 execution consumer 無 TaiwanStockInfo identity evidence")
    delisting = view.read_dataset_frame(
        "TaiwanStockDelisting",
        columns=("date", "stock_id"),
    )
    broad_reference = build_current_stock_etf_universe(
        info,
        delisting,
        as_of_date=str(market_date),
    )
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
    records = [{"stock_id": sid, "is_etf": sid in etf_ids} for sid in members]
    return records, len(broad_reference)


def _execution_market_value_rows(
    view: TradingMarketDataV2View,
    *,
    project_root: Path,
    market_date: str,
) -> pd.DataFrame:
    """Resolve MarketValue rows under its canonical Scan freshness policy."""

    dataset = "TaiwanStockMarketValue"
    state = load_market_data_dataset_state(project_root, required=False)
    row = dict(((state or {}).get("datasets") or {}).get(dataset) or {})
    contract = get_market_data_freshness_contract(dataset)
    exact_target = market_data_contract_requires_target_freshness(contract, row)
    if exact_target:
        return view.read_dataset_frame(
            dataset,
            columns=("date", "stock_id", "market_value"),
            start_date=market_date,
            end_date=market_date,
        )

    latest_data_date = str(row.get("latest_data_date") or "").strip() or None
    if latest_data_date is not None:
        # ``latest_synced`` means use the newest provider-confirmed snapshot that
        # was knowable for this Scan target.  When the current dataset state has
        # already advanced beyond a finalized older Scanner target, the target's
        # own daily snapshot is the newest admissible one in the target-as-of view.
        usable_date = min(latest_data_date, str(market_date))
        frame = view.read_dataset_frame(
            dataset,
            columns=("date", "stock_id", "market_value"),
            start_date=usable_date,
            end_date=usable_date,
        )
        if not frame.empty:
            return frame.reset_index(drop=True)

    # Compatibility fallback for an old state that predates latest_data_date.
    # The target-as-of view still prevents future batches from entering the scan.
    frame = view.read_dataset_frame(
        dataset,
        columns=("date", "stock_id", "market_value"),
        end_date=market_date,
    )
    if frame.empty:
        return frame
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["stock_id"] = work["stock_id"].astype("string").str.strip()
    work = work.loc[
        work["date"].notna()
        & (work["date"] <= str(market_date))
        & work["stock_id"].notna()
        & (work["stock_id"] != "")
    ]
    if work.empty:
        return work.reset_index(drop=True)
    return (
        work.sort_values(["stock_id", "date"], kind="stable")
        .drop_duplicates("stock_id", keep="last")
        .reset_index(drop=True)
    )


def resolve_trading_v2_current_execution_pool(
    view: TradingMarketDataV2View,
    *,
    market_date: str,
    project_root: str | Path | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Resolve today's new-entry execution pool from V2-only local evidence."""

    members, broad_reference_count = _current_market_member_records(view, market_date=market_date)
    price = view.read_dataset_frame(
        FINMIND_ADJUSTED_PRICE_DATASET,
        columns=("date", "stock_id", "Trading_Volume"),
        start_date=market_date,
        end_date=market_date,
    ).rename(columns={"Trading_Volume": "trading_volume"})
    root_value = project_root if project_root is not None else getattr(view, "project_root", None)
    if root_value is None:
        raise ValueError("Trading execution pool 缺 project_root，無法解析 MarketValue Scan freshness policy")
    market_value = _execution_market_value_rows(
        view,
        project_root=Path(root_value).resolve(),
        market_date=market_date,
    )
    tickers, stats = screen_daily_trading_execution_pool(
        members,
        price_rows=price,
        market_value_rows=market_value,
        min_volume=DOWNLOADER_MIN_VOLUME,
        min_market_cap=DOWNLOADER_MIN_MARKET_CAP,
    )
    diagnostics = {
        "stockinfo_broad_reference_count": int(broad_reference_count),
        **dict(stats),
    }
    return sorted(normalize_trading_ticker(item) for item in tickers), diagnostics


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
    overlay_batches = payload.get("source_overlay_batch_fingerprints")
    if overlay_batches is not None:
        if not isinstance(overlay_batches, list):
            raise ValueError("Trading V2 consumer state source_overlay_batch_fingerprints 必須是 list")
        normalized_batches = [str(value).strip() for value in overlay_batches]
        if any(not value for value in normalized_batches) or normalized_batches != list(dict.fromkeys(normalized_batches)):
            raise ValueError("Trading V2 consumer state source_overlay_batch_fingerprints 不得含空值，且須去重並保持發布順序")
    for field in ("current_execution_pool_tickers", "required_position_tickers", "training_tickers"):
        values = payload.get(field)
        if not isinstance(values, list):
            raise ValueError(f"Trading V2 consumer state {field} 必須是 list")
        normalized = [normalize_trading_ticker(item) for item in values]
        if normalized != sorted(set(normalized)):
            raise ValueError(f"Trading V2 consumer state {field} 必須排序且去重")
    reentry_tickers = payload.get("required_reentry_tickers") or []
    if not isinstance(reentry_tickers, list):
        raise ValueError("Trading V2 consumer state required_reentry_tickers 必須是 list")
    normalized_reentry = [normalize_trading_ticker(item) for item in reentry_tickers]
    if normalized_reentry != sorted(set(normalized_reentry)):
        raise ValueError("Trading V2 consumer state required_reentry_tickers 必須排序且去重")
    expected_training = sorted(
        set(payload["current_execution_pool_tickers"])
        | set(payload["required_position_tickers"])
        | set(normalized_reentry)
        | set(payload.get("retained_training_tickers") or [])
    )
    if expected_training != list(payload["training_tickers"]):
        raise ValueError("Trading V2 consumer state training membership union 不一致")
    core = {key: value for key, value in payload.items() if key != "state_fingerprint"}
    if str(payload.get("state_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading V2 consumer state fingerprint 不一致")


def open_trading_v2_consumer_view(
    project_root: str | Path,
    *,
    consumer_state: dict[str, Any] | None = None,
) -> TradingMarketDataV2View:
    """Open the exact immutable V2 view pinned by execution consumer state."""

    root = Path(project_root).resolve()
    state = consumer_state or load_trading_v2_consumer_state(
        root,
        required=True,
        verify_current_view=False,
    )
    market_date = str(state.get("market_date") or "")
    pinned_batches = state.get("source_overlay_batch_fingerprints")
    if isinstance(pinned_batches, list):
        return TradingMarketDataV2View.open_pinned(
            root,
            target_date=market_date,
            overlay_batch_fingerprints=tuple(pinned_batches),
            provider_snapshot_fingerprint=str(state.get("provider_snapshot_fingerprint") or "") or None,
        )
    # Compatibility for pre-pinning consumer states. They remain target-capped
    # and will be replaced on the next legal consumer promotion/publication.
    return TradingMarketDataV2View.open_as_of_target(root, target_date=market_date)


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
        market_date = str(payload.get("market_date") or "")
        view = open_trading_v2_consumer_view(root, consumer_state=payload)
        identity = view.view_identity(
            required_datasets=tuple(payload.get("required_v2_datasets") or _required_v2_datasets()),
            maximum_training_date=market_date,
        )
        if str(payload.get("source_view_fingerprint") or "") != str(identity.get("view_fingerprint") or ""):
            raise RuntimeError(
                "Trading V2 execution consumer state 與其 finalized target view identity 不一致"
            )
    return payload


def get_trading_v2_consumer_state_sha256(project_root: str | Path) -> str:
    path = _state_path(project_root)
    if not path.is_file():
        raise FileNotFoundError("Trading V2 execution consumer state 尚未建立")
    return compute_file_sha256(path)


def reconcile_trading_v2_consumer_state_from_local_evidence(
    project_root: str | Path,
    *,
    now=None,
) -> dict[str, Any]:
    """Provider-free reconciliation of candidate target and finalized consumer.

    This is the local-state counterpart of Auto/Due/Full Update.  It never
    calls FinMind.  When persisted dataset evidence already proves the active
    strategy dependency set READY for the current candidate Scan Target, it
    promotes the execution consumer immediately; otherwise it leaves the prior
    finalized consumer untouched.
    """

    from services.trading.market_data_v2_state import resolve_trading_market_data_update_target_date

    root = Path(project_root).resolve()
    target_date = resolve_trading_market_data_update_target_date(root, now=now)
    if not target_date:
        prior = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
        return {
            "promoted": False,
            "reason": "NO_TARGET",
            "target_date": None,
            "market_date": None if prior is None else prior.get("market_date"),
            "provider_calls": 0,
        }
    result = promote_trading_v2_consumer_state_if_ready(
        root,
        target_date=str(target_date),
    )
    return {
        **dict(result),
        "target_date": str(target_date),
        "provider_calls": 0,
    }


def promote_trading_v2_consumer_state_if_ready(
    project_root: str | Path,
    *,
    target_date: str,
) -> dict[str, Any]:
    """Finalize a newer execution target once active dependencies are READY.

    Promotion is monotonic by market date. Once a target is finalized, later
    same-target archive batches must not rewrite its execution lineage.
    """

    from core.trading_policy import get_trading_strategy_profile
    from services.market_data.provider_snapshot_repository import find_latest_ready_provider_snapshot
    from services.trading.account_state import load_trading_account_state
    from services.trading.data_readiness import build_trading_data_readiness_from_evidence
    from services.trading.live_reentry import resolve_trading_live_reentry_required_tickers
    from services.trading.market_data_dataset_state import load_market_data_dataset_state

    root = Path(project_root).resolve()
    candidate = str(target_date or "").strip()
    if not candidate:
        return {"promoted": False, "reason": "NO_TARGET", "market_date": None}
    if find_latest_ready_provider_snapshot(root) is None:
        return {"promoted": False, "reason": "NO_PROVIDER_SNAPSHOT", "market_date": None}

    prior = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    prior_date = None if prior is None else str(prior.get("market_date") or "").strip() or None
    if prior_date is not None and prior_date >= candidate:
        return {"promoted": False, "reason": "ALREADY_FINALIZED", "market_date": prior_date}

    profile = get_trading_strategy_profile()
    dataset_state = load_market_data_dataset_state(root, required=False)
    readiness = build_trading_data_readiness_from_evidence(
        strategy_id=profile.strategy_id,
        target_date=candidate,
        consumer_state_ready=True,
        dataset_state=dataset_state,
        consumer_state_required=False,
    )
    if not bool(readiness.get("ready")):
        return {
            "promoted": False,
            "reason": "DEPENDENCIES_NOT_READY",
            "market_date": prior_date,
            "blocking_dependencies": list(readiness.get("blocking_dependencies") or []),
        }

    account = load_trading_account_state(root, required=False)
    required_position_tickers = sorted(
        normalize_trading_ticker(ticker)
        for ticker, record in ((account or {}).get("positions") or {}).items()
        if int(((record or {}).get("broker") or {}).get("qty") or 0) > 0
    )
    required_reentry_tickers = resolve_trading_live_reentry_required_tickers(root)
    state = publish_trading_v2_consumer_state(
        root,
        market_date=candidate,
        required_position_tickers=required_position_tickers,
        required_reentry_tickers=required_reentry_tickers,
    )
    return {
        "promoted": True,
        "reason": "PROMOTED",
        "market_date": str(state["market_date"]),
        "consumer_state_fingerprint": str(state["state_fingerprint"]),
        "source_view_fingerprint": str(state["source_view_fingerprint"]),
    }


def publish_trading_v2_consumer_state(
    project_root: str | Path,
    *,
    market_date: str,
    required_position_tickers: Iterable[str] = (),
    required_reentry_tickers: Iterable[str] = (),
    retained_training_tickers: Iterable[str] = (),
    view: TradingMarketDataV2View | None = None,
) -> dict[str, Any]:
    """Publish local V2 consumer membership/lineage after a successful V2 update."""

    root = Path(project_root).resolve()
    date_text = str(market_date)
    local_view = view or TradingMarketDataV2View.open_as_of_target(
        root, target_date=date_text
    )
    required_datasets = _required_v2_datasets()
    horizon = local_view.training_horizon(
        required_datasets=required_datasets,
        maximum_training_date=date_text,
    )
    if date_text > str(horizon.training_through_date):
        raise RuntimeError(
            "Trading V2 consumer state market_date 超過 required dataset READY horizon；"
            f"requested={date_text}, ready_through={horizon.training_through_date}"
        )
    execution_tickers, execution_stats = resolve_trading_v2_current_execution_pool(
        local_view,
        market_date=date_text,
        project_root=root,
    )
    required = sorted({normalize_trading_ticker(item) for item in required_position_tickers})
    required_reentry = sorted({normalize_trading_ticker(item) for item in required_reentry_tickers})
    prior = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    retained = set(normalize_trading_ticker(item) for item in retained_training_tickers)
    if prior is not None:
        retained.update(normalize_trading_ticker(item) for item in prior.get("training_tickers") or [])
    retained.difference_update(execution_tickers)
    retained.difference_update(required)
    retained.difference_update(required_reentry)
    retained_sorted = sorted(retained)
    training = sorted(set(execution_tickers) | set(required) | set(required_reentry) | set(retained_sorted))
    if not training:
        raise RuntimeError("Trading V2 consumer state 沒有任何 training/consumer ticker")

    identity = local_view.view_identity(
        required_datasets=required_datasets,
        maximum_training_date=date_text,
    )
    state: dict[str, Any] = {
        "schema_version": TRADING_V2_CONSUMER_STATE_SCHEMA_VERSION,
        "role": TRADING_V2_CONSUMER_STATE_ROLE,
        "status": "READY",
        "market_date": date_text,
        "source": "trading_market_data_v2_historical_latest_view",
        "source_view_fingerprint": identity["view_fingerprint"],
        "source_overlay_batch_fingerprints": list(identity.get("overlay_batch_fingerprints") or []),
        "provider_snapshot_fingerprint": local_view.archive.snapshot_fingerprint,
        "provider_as_of_date": local_view.archive.as_of_date,
        "required_v2_datasets": list(required_datasets),
        "provider_calls": 0,
        "current_execution_pool_tickers": execution_tickers,
        "current_execution_pool_ticker_count": len(execution_tickers),
        "current_execution_pool_stats": execution_stats,
        "required_position_tickers": required,
        "required_reentry_tickers": required_reentry,
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
    through_date = str(state["market_date"])
    view = open_trading_v2_consumer_view(root, consumer_state=state)
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
    "open_trading_v2_consumer_view",
    "promote_trading_v2_consumer_state_if_ready",
    "reconcile_trading_v2_consumer_state_from_local_evidence",
    "get_trading_v2_consumer_state_sha256",
    "publish_trading_v2_consumer_state",
    "load_trading_v2_optimizer_raw_data",
]
