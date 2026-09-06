"""Canonical persisted identity for the current Trading market-data dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.file_integrity import (
    atomic_write_json,
    canonical_json_sha256,
    compute_file_sha256,
    load_json_strict,
)
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths, resolve_runtime_output_dir
from core.trading_dataset_identity import (
    assert_trading_dataset_fingerprint_matches,
    build_trading_dataset_fingerprint,
)
from core.trading_identity import normalize_trading_date, normalize_trading_ticker

TRADING_MARKET_DATA_SNAPSHOT_SCHEMA_VERSION = 1
TRADING_MARKET_DATA_SNAPSHOT_FILENAME = "market_data_snapshot.json"


def resolve_trading_market_data_snapshot_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="market_data")) / TRADING_MARKET_DATA_SNAPSHOT_FILENAME


def _validate_snapshot(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading market-data snapshot 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_MARKET_DATA_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Trading market-data snapshot schema 不相容；請重新更新 Trading 資料")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading market-data snapshot runtime domain 不合法")
    market_date = normalize_trading_date(payload.get("market_date"), field_name="market_date", allow_none=False)
    fingerprint = payload.get("dataset_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("Trading market-data snapshot 缺少 dataset_fingerprint")
    if str(fingerprint.get("fingerprint_algorithm") or "") != "sha256":
        raise ValueError("Trading market-data snapshot fingerprint algorithm 不合法")
    if str(fingerprint.get("latest_data_date") or "") != market_date:
        raise ValueError("Trading market-data snapshot market_date 與 dataset latest date 不一致")
    if len(str(fingerprint.get("csv_content_sha256") or "")) != 64:
        raise ValueError("Trading market-data snapshot content SHA 不合法")
    required = payload.get("required_position_tickers")
    if not isinstance(required, list):
        raise ValueError("Trading market-data snapshot required_position_tickers 必須是 list")
    normalized = [normalize_trading_ticker(item) for item in required]
    if normalized != sorted(set(normalized)):
        raise ValueError("Trading market-data snapshot required position tickers 必須排序且唯一")
    core = {key: value for key, value in payload.items() if key != "snapshot_fingerprint"}
    if str(payload.get("snapshot_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading market-data snapshot fingerprint 不一致")


def publish_trading_market_data_snapshot(
    project_root: str | Path,
    *,
    market_date: object,
    required_position_tickers: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    date_text = normalize_trading_date(market_date, field_name="market_date", allow_none=False)
    dataset_fingerprint = build_trading_dataset_fingerprint(paths.data_dir)
    if str(dataset_fingerprint.get("latest_data_date") or "") != date_text:
        raise RuntimeError(
            "Trading downloader 完成後 dataset latest date 與確認 market date 不一致："
            f"dataset={dataset_fingerprint.get('latest_data_date')}, market={date_text}"
        )
    required = sorted({normalize_trading_ticker(item) for item in required_position_tickers})
    payload: dict[str, Any] = {
        "schema_version": TRADING_MARKET_DATA_SNAPSHOT_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "market_date": date_text,
        "required_position_tickers": required,
        "dataset_fingerprint": dataset_fingerprint,
    }
    payload["snapshot_fingerprint"] = canonical_json_sha256(payload)
    path = resolve_trading_market_data_snapshot_path(root)
    atomic_write_json(path, payload)
    return payload


def load_trading_market_data_snapshot(
    project_root: str | Path,
    *,
    required: bool = True,
    verify_dataset_content: bool = False,
) -> dict[str, Any] | None:
    root = Path(project_root).resolve()
    path = resolve_trading_market_data_snapshot_path(root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading market-data snapshot 尚未建立；請先執行「1 更新 Trading 資料」")
        return None
    payload = load_json_strict(path)
    _validate_snapshot(payload)
    if verify_dataset_content:
        paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
        assert_trading_dataset_fingerprint_matches(paths.data_dir, dict(payload["dataset_fingerprint"]))
    return payload


def get_trading_market_data_snapshot_sha256(project_root: str | Path) -> str:
    path = resolve_trading_market_data_snapshot_path(project_root)
    if not path.is_file():
        raise FileNotFoundError("Trading market-data snapshot 尚未建立")
    return compute_file_sha256(path)


__all__ = [
    "TRADING_MARKET_DATA_SNAPSHOT_SCHEMA_VERSION",
    "TRADING_MARKET_DATA_SNAPSHOT_FILENAME",
    "resolve_trading_market_data_snapshot_path",
    "publish_trading_market_data_snapshot",
    "load_trading_market_data_snapshot",
    "get_trading_market_data_snapshot_sha256",
]
