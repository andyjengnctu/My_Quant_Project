"""Neutral derived daily PIT market-universe service for Market Data V2."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd

from core.file_integrity import (
    atomic_replace_with_retry,
    atomic_write_json,
    canonical_json_sha256,
    compute_file_sha256,
    load_json_strict,
)
from core.market_data_instrument_universe import (
    build_historical_market_state_guard,
    historical_market_state_guard_fingerprint,
)
from core.market_data_pit_universe import (
    MARKET_DATA_DAILY_PIT_UNIVERSE_MARKET_STATE_DATASET,
    MARKET_DATA_DAILY_PIT_UNIVERSE_SOURCE_DATASET,
    MARKET_DATA_DAILY_PIT_UNIVERSE_TRADING_CALENDAR_DATASET,
    filter_daily_pit_market_universe,
    make_daily_pit_market_universe_guard,
    daily_pit_market_universe_contract_fingerprint,
)
from core.market_data_storage_contract import (
    resolve_market_data_daily_pit_universe_manifest_path,
    resolve_market_data_daily_pit_universe_path,
)
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.market_data.provider_snapshot_view import FrameReader, HashFn, ProviderSnapshotView

DAILY_PIT_UNIVERSE_STATUS_READY = "READY"
DAILY_PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION = 1


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE trading_dates (
            date TEXT PRIMARY KEY
        );
        CREATE TABLE daily_universe (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            PRIMARY KEY(date, stock_id)
        ) WITHOUT ROWID;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def _logical_fingerprint(conn: sqlite3.Connection, query: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in conn.execute(query):
        digest.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def _normalize_trading_dates(frames, *, provider_as_of_date: str) -> tuple[str, ...]:
    values: set[str] = set()
    for frame in frames:
        if "date" not in frame.columns:
            raise ValueError("TaiwanStockTradingDate 缺少 date")
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        values.update(str(value) for value in dates.dropna().tolist() if str(value) <= provider_as_of_date)
    if not values:
        raise ValueError("Market Data V2 neutral daily PIT trading calendar evidence 不可為空")
    return tuple(sorted(values))


def _identity_payload(
    *,
    provider_snapshot_fingerprint: str,
    provider_as_of_date: str,
    historical_instrument_count: int,
    historical_market_state_guard_fingerprint: str,
    trading_date_count: int,
    trading_calendar_fingerprint: str,
    daily_universe_row_count: int,
    daily_universe_date_count: int,
    first_universe_date: str | None,
    latest_universe_date: str | None,
    daily_universe_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": DAILY_PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION,
        "contract_fingerprint": daily_pit_market_universe_contract_fingerprint(),
        "provider_snapshot_fingerprint": str(provider_snapshot_fingerprint),
        "provider_as_of_date": str(provider_as_of_date),
        "historical_instrument_count": int(historical_instrument_count),
        "historical_market_state_guard_fingerprint": str(historical_market_state_guard_fingerprint),
        "trading_date_count": int(trading_date_count),
        "trading_calendar_fingerprint": str(trading_calendar_fingerprint),
        "daily_universe_row_count": int(daily_universe_row_count),
        "daily_universe_date_count": int(daily_universe_date_count),
        "first_universe_date": first_universe_date,
        "latest_universe_date": latest_universe_date,
        "daily_universe_fingerprint": str(daily_universe_fingerprint),
    }


def _validate_manifest(project_root, payload: dict[str, object], *, expected_snapshot_fingerprint: str) -> dict[str, object]:
    if str(payload.get("status") or "") != DAILY_PIT_UNIVERSE_STATUS_READY:
        raise ValueError("Market Data V2 daily PIT universe 尚未 READY")
    identity_fields = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "contract_fingerprint",
            "provider_snapshot_fingerprint",
            "provider_as_of_date",
            "historical_instrument_count",
            "historical_market_state_guard_fingerprint",
            "trading_date_count",
            "trading_calendar_fingerprint",
            "daily_universe_row_count",
            "daily_universe_date_count",
            "first_universe_date",
            "latest_universe_date",
            "daily_universe_fingerprint",
        )
    }
    if int(identity_fields["schema_version"] or 0) != DAILY_PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Market Data V2 daily PIT universe schema version 不一致")
    if str(identity_fields["contract_fingerprint"] or "") != daily_pit_market_universe_contract_fingerprint():
        raise ValueError("Market Data V2 daily PIT universe contract fingerprint drift")
    if str(identity_fields["provider_snapshot_fingerprint"] or "") != str(expected_snapshot_fingerprint):
        raise ValueError("Market Data V2 daily PIT universe Provider Snapshot identity drift")
    expected_identity_fingerprint = canonical_json_sha256(identity_fields)
    if str(payload.get("daily_pit_universe_identity_fingerprint") or "") != expected_identity_fingerprint:
        raise ValueError("Market Data V2 daily PIT universe identity fingerprint drift")
    universe_path = resolve_market_data_daily_pit_universe_path(project_root, expected_snapshot_fingerprint)
    if not universe_path.is_file():
        raise FileNotFoundError("Market Data V2 neutral daily PIT universe artifact 不存在")
    actual_file_hash = compute_file_sha256(universe_path)
    if str(payload.get("daily_universe_file_sha256") or "") != actual_file_hash:
        raise ValueError("Market Data V2 neutral daily PIT universe physical hash drift")
    conn = _connect(universe_path)
    try:
        row_count, universe_fp = _logical_fingerprint(
            conn, "SELECT date, stock_id FROM daily_universe ORDER BY date, stock_id"
        )
        date_count = int(conn.execute("SELECT COUNT(DISTINCT date) FROM daily_universe").fetchone()[0])
        bounds = conn.execute("SELECT MIN(date), MAX(date) FROM daily_universe").fetchone()
        trading_count, trading_fp = _logical_fingerprint(conn, "SELECT date FROM trading_dates ORDER BY date")
    finally:
        conn.close()
    if row_count != int(payload.get("daily_universe_row_count") or 0) or universe_fp != str(payload.get("daily_universe_fingerprint") or ""):
        raise ValueError("Market Data V2 neutral daily PIT universe logical content drift")
    if date_count != int(payload.get("daily_universe_date_count") or 0):
        raise ValueError("Market Data V2 neutral daily PIT universe date count drift")
    if (bounds[0] if bounds else None) != payload.get("first_universe_date") or (bounds[1] if bounds else None) != payload.get("latest_universe_date"):
        raise ValueError("Market Data V2 neutral daily PIT universe date bounds drift")
    if trading_count != int(payload.get("trading_date_count") or 0) or trading_fp != str(payload.get("trading_calendar_fingerprint") or ""):
        raise ValueError("Market Data V2 neutral daily PIT trading calendar drift")
    return payload


def load_market_data_v2_daily_pit_universe(
    project_root,
    *,
    provider_snapshot_fingerprint: str,
    required: bool = True,
) -> dict[str, object] | None:
    manifest_path = resolve_market_data_daily_pit_universe_manifest_path(project_root, provider_snapshot_fingerprint)
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError("Market Data V2 neutral daily PIT universe manifest 不存在")
        return None
    return _validate_manifest(
        project_root,
        load_json_strict(manifest_path),
        expected_snapshot_fingerprint=provider_snapshot_fingerprint,
    )


def build_market_data_v2_daily_pit_universe(
    project_root,
    *,
    snapshot_fingerprint: str | None = None,
    frame_reader: FrameReader | None = None,
    hash_fn: HashFn | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
    existing = load_market_data_v2_daily_pit_universe(
        root,
        provider_snapshot_fingerprint=archive.snapshot_fingerprint,
        required=False,
    )
    if existing is not None:
        return {**existing, "action": "REUSE"}

    view = ProviderSnapshotView(
        project_root=root,
        archive=archive,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
    )
    historical_instruments = view.historical_instruments(source_dataset=MARKET_DATA_DAILY_PIT_UNIVERSE_SOURCE_DATASET)
    stock_info_frames = list(
        view.iter_verified_frames(
            MARKET_DATA_DAILY_PIT_UNIVERSE_MARKET_STATE_DATASET,
            columns=("date", "stock_id", "type", "industry_category"),
        )
    )
    if not stock_info_frames:
        raise ValueError("Market Data V2 neutral daily PIT universe 缺 TaiwanStockInfo evidence")
    stock_info = pd.concat(stock_info_frames, ignore_index=True)
    transition_guard = build_historical_market_state_guard(
        stock_info,
        historical_instruments=historical_instruments,
    )
    market_state_fp = historical_market_state_guard_fingerprint(transition_guard)
    trading_dates = _normalize_trading_dates(
        view.iter_verified_frames(
            MARKET_DATA_DAILY_PIT_UNIVERSE_TRADING_CALENDAR_DATASET,
            columns=("date",),
        ),
        provider_as_of_date=archive.as_of_date,
    )

    output_path = resolve_market_data_daily_pit_universe_path(root, archive.snapshot_fingerprint)
    manifest_path = resolve_market_data_daily_pit_universe_manifest_path(root, archive.snapshot_fingerprint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".daily_pit_universe.", suffix=".sqlite3.tmp", dir=str(output_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        conn = _connect(temp_path)
        try:
            _init_db(conn)
            conn.executemany("INSERT OR IGNORE INTO trading_dates(date) VALUES (?)", ((value,) for value in trading_dates))
            membership_guard = make_daily_pit_market_universe_guard(
                historical_instruments=historical_instruments,
                transition_excluded_through=transition_guard,
                trading_dates=trading_dates,
                provider_as_of_date=archive.as_of_date,
            )
            for frame in view.iter_verified_frames(
                MARKET_DATA_DAILY_PIT_UNIVERSE_SOURCE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = filter_daily_pit_market_universe(frame, guard=membership_guard)
                if not rows.empty:
                    conn.executemany(
                        "INSERT OR IGNORE INTO daily_universe(date, stock_id) VALUES (?, ?)",
                        rows[["date", "stock_id"]].itertuples(index=False, name=None),
                    )
            universe_count, universe_fp = _logical_fingerprint(
                conn, "SELECT date, stock_id FROM daily_universe ORDER BY date, stock_id"
            )
            if universe_count <= 0:
                raise ValueError("Market Data V2 neutral daily PIT universe 不可為空")
            universe_date_count = int(conn.execute("SELECT COUNT(DISTINCT date) FROM daily_universe").fetchone()[0])
            first_date, latest_date = conn.execute("SELECT MIN(date), MAX(date) FROM daily_universe").fetchone()
            trading_count, trading_fp = _logical_fingerprint(conn, "SELECT date FROM trading_dates ORDER BY date")
            identity = _identity_payload(
                provider_snapshot_fingerprint=archive.snapshot_fingerprint,
                provider_as_of_date=archive.as_of_date,
                historical_instrument_count=len(historical_instruments),
                historical_market_state_guard_fingerprint=market_state_fp,
                trading_date_count=trading_count,
                trading_calendar_fingerprint=trading_fp,
                daily_universe_row_count=universe_count,
                daily_universe_date_count=universe_date_count,
                first_universe_date=first_date,
                latest_universe_date=latest_date,
                daily_universe_fingerprint=universe_fp,
            )
            metadata = {
                **identity,
                "daily_pit_universe_identity_fingerprint": canonical_json_sha256(identity),
            }
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ((key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in metadata.items()),
            )
            conn.commit()
        finally:
            conn.close()
        atomic_replace_with_retry(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    built_at = (now or datetime.now().astimezone()).astimezone().isoformat()
    payload = {
        **identity,
        "daily_pit_universe_identity_fingerprint": canonical_json_sha256(identity),
        "status": DAILY_PIT_UNIVERSE_STATUS_READY,
        "daily_universe_file_sha256": compute_file_sha256(output_path),
        "built_at": built_at,
        "provider_calls": 0,
    }
    atomic_write_json(manifest_path, payload)
    validated = load_market_data_v2_daily_pit_universe(
        root,
        provider_snapshot_fingerprint=archive.snapshot_fingerprint,
        required=True,
    )
    assert validated is not None
    return {**validated, "action": "BUILD"}


__all__ = [
    "DAILY_PIT_UNIVERSE_STATUS_READY",
    "DAILY_PIT_UNIVERSE_MANIFEST_SCHEMA_VERSION",
    "load_market_data_v2_daily_pit_universe",
    "build_market_data_v2_daily_pit_universe",
]
