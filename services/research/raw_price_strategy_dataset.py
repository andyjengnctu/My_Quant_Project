"""Research-only raw/unadjusted OHLCV materialization for strategy experiments.

This module deliberately does not change the active Research V2 compatibility
materialization, whose price semantics remain FinMind ``TaiwanStockPriceAdj``.
Instead it derives an isolated, content-verified six-column OHLCV view from the
same frozen Research V2 Provider Snapshot and daily PIT universe, using
``TaiwanStockPrice`` OHLC + Trading_Volume directly.

The resulting dataset is intended for explicitly raw-price strategy identities
only.  It performs no provider calls and never synthesizes adjusted prices.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Mapping

import pandas as pd

from core.file_integrity import canonical_json_sha256, compute_file_sha256, load_json_strict
from core.market_data_adjusted_price_invariance import PROVIDER_VOLUME_FIELD
from core.market_data_contract import FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.market_data_research_materialization import deterministic_compatibility_file_mtime_ns
from services.research.market_data_generation import load_active_research_v2_read_view

RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION = 1
RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID = "research_v2_raw_unadjusted_strategy_ohlcv_v1"
RAW_PRICE_STRATEGY_PRICE_BASIS = "raw_unadjusted"
RAW_PRICE_STRATEGY_OUTPUT_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
RAW_PRICE_STRATEGY_PRICE_FIELDS = ("open", "max", "min", "close")
RAW_PRICE_STRATEGY_RELATIVE_ROOT = Path("outputs") / "research" / "raw_price_strategy_dataset"
_SAFE_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _source_identity(project_root: str | Path) -> tuple[dict[str, object], object]:
    root = Path(project_root).resolve()
    view = load_active_research_v2_read_view(root, verify_materialization_file_content=False)
    freeze = dict(view.freeze_candidate)
    promotion = dict(view.promotion)
    identity = {
        "schema_version": RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION,
        "contract_id": RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID,
        "price_basis": RAW_PRICE_STRATEGY_PRICE_BASIS,
        "dataset_profile": "full",
        "generation_id": "research_v2_raw_unadjusted_strategy",
        "frozen_cutoff": str(view.frozen_cutoff),
        "source_dataset": FINMIND_RAW_PRICE_ARCHIVE_DATASET,
        "price_fields": list(RAW_PRICE_STRATEGY_PRICE_FIELDS),
        "volume_field": PROVIDER_VOLUME_FIELD,
        "output_columns": list(RAW_PRICE_STRATEGY_OUTPUT_COLUMNS),
        "provider_snapshot_fingerprint": str(freeze.get("provider_snapshot_fingerprint") or ""),
        "freeze_candidate_fingerprint": str(freeze.get("freeze_candidate_fingerprint") or ""),
        "daily_universe_file_sha256": str(freeze.get("daily_universe_file_sha256") or ""),
        "promotion_fingerprint": str(promotion.get("promotion_fingerprint") or ""),
    }
    required = (
        "provider_snapshot_fingerprint",
        "freeze_candidate_fingerprint",
        "daily_universe_file_sha256",
        "promotion_fingerprint",
    )
    for field in required:
        value = str(identity[field])
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ValueError(f"raw-price strategy source identity缺少合法SHA256: {field}")
    identity["source_identity_fingerprint"] = canonical_json_sha256(identity)
    return identity, view


def raw_price_strategy_dataset_source_identity(project_root: str | Path) -> dict[str, object]:
    identity, _view = _source_identity(project_root)
    return identity


def _materialization_dir(root: Path, source_fingerprint: str) -> Path:
    return root / RAW_PRICE_STRATEGY_RELATIVE_ROOT / str(source_fingerprint)


def _dataset_dir(root: Path, source_fingerprint: str) -> Path:
    return _materialization_dir(root, source_fingerprint) / "tw_stock_data_vip"


def _manifest_path(root: Path, source_fingerprint: str) -> Path:
    return _materialization_dir(root, source_fingerprint) / "manifest.json"


def _normalize_raw_rows(frame: pd.DataFrame, *, cutoff: str) -> list[tuple[str, str, float, float, float, float, float]]:
    needed = {"date", "stock_id", *RAW_PRICE_STRATEGY_PRICE_FIELDS, PROVIDER_VOLUME_FIELD}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"TaiwanStockPrice raw-price strategy materialization缺欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "stock_id", *RAW_PRICE_STRATEGY_PRICE_FIELDS, PROVIDER_VOLUME_FIELD]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["stock_id"] = local["stock_id"].astype(str).str.strip()
    valid = local["date"].notna() & local["stock_id"].ne("") & local["date"].le(str(cutoff))
    for field in (*RAW_PRICE_STRATEGY_PRICE_FIELDS, PROVIDER_VOLUME_FIELD):
        local[field] = pd.to_numeric(local[field], errors="coerce")
        valid &= local[field].notna()
    local = local.loc[valid]
    rows: list[tuple[str, str, float, float, float, float, float]] = []
    for row in local.itertuples(index=False, name=None):
        values = tuple(float(value) for value in row[2:])
        if any(value in (float("inf"), float("-inf")) for value in values):
            continue
        rows.append((str(row[0]), str(row[1]), *values))
    return rows


def _insert_raw_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, float, float, float, float, float]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO raw_price(date, stock_id, open, high, low, close, volume, conflict)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            conflict = CASE WHEN
                raw_price.open = excluded.open AND raw_price.high = excluded.high
                AND raw_price.low = excluded.low AND raw_price.close = excluded.close
                AND raw_price.volume = excluded.volume
            THEN raw_price.conflict ELSE 1 END
        """,
        rows,
    )


def _build_files(*, root: Path, view, stage_dir: Path, source_identity: Mapping[str, object]) -> tuple[list[dict[str, object]], int]:
    cutoff = str(source_identity["frozen_cutoff"])
    dataset_dir = stage_dir / "tw_stock_data_vip"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    merge_path = stage_dir / ".raw_price_merge.sqlite3"
    conn = sqlite3.connect(merge_path)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE raw_price(
                date TEXT NOT NULL,
                stock_id TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                conflict INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(stock_id, date)
            ) WITHOUT ROWID;
            """
        )
        columns = ("date", "stock_id", "open", "max", "min", "close", PROVIDER_VOLUME_FIELD)
        for frame in view.provider_view.iter_raw_price_strategy_frames(columns=columns):
            _insert_raw_rows(conn, _normalize_raw_rows(frame, cutoff=cutoff))
        conn.commit()
        conflicts = int(conn.execute("SELECT COUNT(*) FROM raw_price WHERE conflict <> 0").fetchone()[0])
        if conflicts:
            raise ValueError(f"raw-price strategy source rows存在conflicting duplicate keys: {conflicts}")

        conn.execute("ATTACH DATABASE ? AS candidate", (str(view.daily_universe_path),))
        missing_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM candidate.daily_universe u
                LEFT JOIN raw_price r ON r.stock_id = u.stock_id AND r.date = u.date
                WHERE u.date <= ? AND r.stock_id IS NULL
                """,
                (cutoff,),
            ).fetchone()[0]
        )
        if missing_count:
            raise ValueError(
                "raw-price strategy materialization缺daily PIT universe所需TaiwanStockPrice OHLCV rows: "
                f"{missing_count}"
            )
        tickers = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT stock_id FROM candidate.daily_universe WHERE date <= ? ORDER BY stock_id",
                (cutoff,),
            )
        ]
        if not tickers:
            raise ValueError("raw-price strategy materialization沒有eligible ticker")

        evidence: list[dict[str, object]] = []
        total_rows = 0
        fixed_mtime_ns = deterministic_compatibility_file_mtime_ns(cutoff)
        for ticker in tickers:
            if not _SAFE_TICKER_RE.fullmatch(ticker) or Path(ticker).name != ticker:
                raise ValueError(f"raw-price strategy ticker path不合法: {ticker!r}")
            frame = pd.read_sql_query(
                """
                SELECT u.date AS Date, r.open AS Open, r.high AS High, r.low AS Low,
                       r.close AS Close, r.volume AS Volume
                FROM candidate.daily_universe u
                JOIN raw_price r ON r.stock_id = u.stock_id AND r.date = u.date
                WHERE u.stock_id = ? AND u.date <= ?
                ORDER BY u.date
                """,
                conn,
                params=(ticker, cutoff),
            )
            if frame.empty:
                continue
            volume = pd.to_numeric(frame["Volume"], errors="raise")
            rounded = volume.round()
            if bool((volume == rounded).all()):
                frame["Volume"] = rounded.astype("int64")
            path = dataset_dir / f"{ticker}.csv"
            frame.loc[:, list(RAW_PRICE_STRATEGY_OUTPUT_COLUMNS)].to_csv(path, index=False, lineterminator="\n")
            os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))
            row_count = int(len(frame))
            total_rows += row_count
            evidence.append(
                {
                    "ticker": ticker,
                    "relative_path": path.relative_to(dataset_dir).as_posix(),
                    "row_count": row_count,
                    "size_bytes": int(path.stat().st_size),
                    "content_sha256": compute_file_sha256(path),
                }
            )
        evidence.sort(key=lambda row: (str(row["ticker"]), str(row["relative_path"])))
        return evidence, total_rows
    finally:
        conn.close()
        merge_path.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def inspect_raw_price_strategy_dataset(project_root: str | Path, *, deep: bool = False) -> dict[str, object]:
    root = Path(project_root).resolve()
    identity, _view = _source_identity(root)
    source_fp = str(identity["source_identity_fingerprint"])
    manifest_path = _manifest_path(root, source_fp)
    dataset_dir = _dataset_dir(root, source_fp)
    payload = _load_manifest(manifest_path)
    if payload is None:
        return {
            "ready": False,
            "status": "RAW_PRICE_DATASET_NOT_MATERIALIZED",
            "source_identity": identity,
            "source_identity_fingerprint": source_fp,
            "manifest_path": manifest_path,
            "dataset_dir": dataset_dir,
        }
    if dict(payload.get("source_identity") or {}) != identity:
        return {
            "ready": False,
            "status": "RAW_PRICE_DATASET_SOURCE_IDENTITY_MISMATCH",
            "source_identity": identity,
            "source_identity_fingerprint": source_fp,
            "manifest_path": manifest_path,
            "dataset_dir": dataset_dir,
        }
    files = list(payload.get("files") or [])
    if not dataset_dir.is_dir() or not files:
        return {
            "ready": False,
            "status": "RAW_PRICE_DATASET_FILES_MISSING",
            "source_identity": identity,
            "source_identity_fingerprint": source_fp,
            "manifest_path": manifest_path,
            "dataset_dir": dataset_dir,
        }
    inventory_sha = canonical_json_sha256(files)
    if inventory_sha != str(payload.get("dataset_inventory_sha256") or ""):
        raise ValueError("raw-price strategy dataset inventory fingerprint drift")
    if deep:
        expected = {str(row.get("relative_path") or "") for row in files}
        actual = {path.relative_to(dataset_dir).as_posix() for path in dataset_dir.glob("*.csv") if path.is_file()}
        if expected != actual:
            raise ValueError("raw-price strategy dataset CSV file set drift")
        for row in files:
            rel = str(row.get("relative_path") or "")
            path = dataset_dir / rel
            if not path.is_file():
                raise FileNotFoundError(f"raw-price strategy dataset CSV不存在: {rel}")
            if int(path.stat().st_size) != int(row.get("size_bytes") or -1):
                raise ValueError(f"raw-price strategy dataset CSV size drift: {rel}")
            if compute_file_sha256(path) != str(row.get("content_sha256") or ""):
                raise ValueError(f"raw-price strategy dataset CSV SHA256 drift: {rel}")
    data_identity = dict(payload.get("data_identity") or {})
    expected_data_fp = canonical_json_sha256({
        "source_identity_fingerprint": source_fp,
        "dataset_inventory_sha256": inventory_sha,
    })
    if str(data_identity.get("materialization_fingerprint") or "") != expected_data_fp:
        raise ValueError("raw-price strategy data identity drift")
    return {
        "ready": True,
        "status": "READY",
        "source_identity": identity,
        "source_identity_fingerprint": source_fp,
        "manifest_path": manifest_path,
        "dataset_dir": dataset_dir,
        "data_identity": data_identity,
        "files": files,
        "total_rows": int(payload.get("total_rows") or 0),
    }


def ensure_raw_price_strategy_dataset(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    state = inspect_raw_price_strategy_dataset(root, deep=False)
    if bool(state.get("ready")):
        return inspect_raw_price_strategy_dataset(root, deep=True)

    identity, view = _source_identity(root)
    source_fp = str(identity["source_identity_fingerprint"])
    final_dir = _materialization_dir(root, source_fp)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{source_fp}.", suffix=".tmp", dir=str(final_dir.parent)))
    try:
        files, total_rows = _build_files(
            root=root,
            view=view,
            stage_dir=stage_dir,
            source_identity=identity,
        )
        inventory_sha = canonical_json_sha256(files)
        materialization_fp = canonical_json_sha256({
            "source_identity_fingerprint": source_fp,
            "dataset_inventory_sha256": inventory_sha,
        })
        data_identity = {
            "schema_version": RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION,
            "dataset_profile": "full",
            "generation_id": "research_v2_raw_unadjusted_strategy",
            "frozen_cutoff": str(identity["frozen_cutoff"]),
            "price_basis": RAW_PRICE_STRATEGY_PRICE_BASIS,
            "source_dataset": FINMIND_RAW_PRICE_ARCHIVE_DATASET,
            "provider_snapshot_fingerprint": str(identity["provider_snapshot_fingerprint"]),
            "freeze_candidate_fingerprint": str(identity["freeze_candidate_fingerprint"]),
            "daily_universe_file_sha256": str(identity["daily_universe_file_sha256"]),
            "source_identity_fingerprint": source_fp,
            "dataset_inventory_sha256": inventory_sha,
            "materialization_fingerprint": materialization_fp,
        }
        manifest = {
            "schema_version": RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION,
            "contract_id": RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID,
            "status": "READY",
            "source_identity": identity,
            "source_identity_fingerprint": source_fp,
            "dataset_inventory_sha256": inventory_sha,
            "data_identity": data_identity,
            "total_rows": int(total_rows),
            "ticker_count": len(files),
            "files": files,
        }
        (stage_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage_dir, final_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return inspect_raw_price_strategy_dataset(root, deep=True)


__all__ = [
    "RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID",
    "RAW_PRICE_STRATEGY_PRICE_BASIS",
    "ensure_raw_price_strategy_dataset",
    "inspect_raw_price_strategy_dataset",
    "raw_price_strategy_dataset_source_identity",
]
