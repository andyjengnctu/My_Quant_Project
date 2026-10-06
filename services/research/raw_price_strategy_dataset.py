"""Research-only raw/unadjusted OHLCV materialization for strategy experiments.

The experiment is deliberately isolated from the canonical adjusted-price Research
pipeline.  Its control population is the exact current ``full`` Research dataset used
by the adjusted-price baseline: ticker set, date grid and Volume are preserved byte-
semantically from that dataset.  Only OHLC is replaced with immutable FinMind
``TaiwanStockPrice`` raw prices from one READY Market Data V2 Provider Snapshot.

This keeps C93-vs-C61 a price-basis experiment instead of accidentally changing the
Research universe at the same time.  No Research V2 promotion pointer is required and
no provider calls or locally synthesized adjusted prices are allowed.
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

from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.data_utils import discover_unique_csv_inputs
from core.dataset_profiles import build_dataset_generation_identity, get_dataset_dir
from core.file_integrity import canonical_json_sha256, compute_file_sha256, load_json_strict
from core.market_data_contract import FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.market_data_research_materialization import deterministic_compatibility_file_mtime_ns
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.research.market_data_v2 import ResearchV2ProviderView

RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION = 2
RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID = "research_raw_unadjusted_controlled_ohlcv_v2"
RAW_PRICE_STRATEGY_PRICE_BASIS = "raw_unadjusted"
RAW_PRICE_STRATEGY_OUTPUT_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
RAW_PRICE_STRATEGY_PRICE_FIELDS = ("open", "max", "min", "close")
RAW_PRICE_STRATEGY_RELATIVE_ROOT = Path("outputs") / "research" / "raw_price_strategy_dataset"
_SAFE_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _control_dataset_inventory(root: Path) -> tuple[Path, list[dict[str, object]], list[str]]:
    """Fingerprint the exact adjusted-price control dataset without hashing all CSVs.

    The canonical dataset-generation identity supplies lifecycle lineage.  File name,
    size and mtime bind this raw experiment to the concrete control bytes used locally;
    the generated raw materialization itself remains content-SHA verified.
    """

    data_dir = Path(get_dataset_dir(str(root), "full")).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Full Research control dataset不存在: {data_dir}")
    discovered, duplicate_issues = discover_unique_csv_inputs(str(data_dir))
    if not discovered:
        raise FileNotFoundError(f"Full Research control dataset沒有CSV: {data_dir}")
    rows: list[dict[str, object]] = []
    for ticker, raw_path in discovered:
        path = Path(raw_path).resolve()
        stat = path.stat()
        rows.append({
            "ticker": str(ticker),
            "file_name": path.name,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return data_dir, rows, list(duplicate_issues)


def _source_identity(project_root: str | Path) -> tuple[dict[str, object], ResearchV2ProviderView, Path]:
    root = Path(project_root).resolve()
    cutoff = str(RESEARCH_MARKET_DATA_CUTOFF)
    control_dir, control_inventory, duplicate_issues = _control_dataset_inventory(root)
    control_generation = build_dataset_generation_identity(root, "full")

    # Neutral Provider Snapshot truth is independent from Research V2 promotion.  This
    # is the critical separation that lets raw-price Research run while V2 remains
    # authorized_not_ready / unpromoted.
    archive = load_ready_provider_snapshot_archive(root)
    if pd.Timestamp(str(archive.as_of_date)).normalize() < pd.Timestamp(cutoff).normalize():
        raise RuntimeError(
            "raw-price strategy所需READY Provider Snapshot coverage不足: "
            f"snapshot={archive.as_of_date}, required_cutoff={cutoff}"
        )
    view = ResearchV2ProviderView(project_root=root, archive=archive)
    identity = {
        "schema_version": RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION,
        "contract_id": RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID,
        "price_basis": RAW_PRICE_STRATEGY_PRICE_BASIS,
        "dataset_profile": "full",
        "generation_id": "research_raw_unadjusted_controlled_v2",
        "frozen_cutoff": cutoff,
        "source_dataset": FINMIND_RAW_PRICE_ARCHIVE_DATASET,
        "price_fields": list(RAW_PRICE_STRATEGY_PRICE_FIELDS),
        "volume_basis": "canonical_full_control_dataset_preserved",
        "universe_basis": "canonical_full_control_dataset_ticker_date_grid",
        "output_columns": list(RAW_PRICE_STRATEGY_OUTPUT_COLUMNS),
        "provider_snapshot_fingerprint": str(archive.snapshot_fingerprint),
        "provider_snapshot_as_of_date": str(archive.as_of_date),
        "control_dataset_generation_identity": dict(control_generation),
        "control_dataset_dir": control_dir.relative_to(root).as_posix() if root in control_dir.parents else str(control_dir),
        "control_dataset_inventory_fingerprint": canonical_json_sha256(control_inventory),
        "control_dataset_file_count": len(control_inventory),
        "control_duplicate_resolution": duplicate_issues,
    }
    provider_fp = str(identity["provider_snapshot_fingerprint"])
    if len(provider_fp) != 64 or any(ch not in "0123456789abcdef" for ch in provider_fp.lower()):
        raise ValueError("raw-price strategy provider snapshot fingerprint不是合法SHA256")
    identity["source_identity_fingerprint"] = canonical_json_sha256(identity)
    return identity, view, control_dir


def raw_price_strategy_dataset_source_identity(project_root: str | Path) -> dict[str, object]:
    identity, _view, _control_dir = _source_identity(project_root)
    return identity


def _materialization_dir(root: Path, source_fingerprint: str) -> Path:
    return root / RAW_PRICE_STRATEGY_RELATIVE_ROOT / str(source_fingerprint)


def _dataset_dir(root: Path, source_fingerprint: str) -> Path:
    return _materialization_dir(root, source_fingerprint) / "tw_stock_data_vip"


def _manifest_path(root: Path, source_fingerprint: str) -> Path:
    return _materialization_dir(root, source_fingerprint) / "manifest.json"


def _normalize_raw_rows(frame: pd.DataFrame, *, cutoff: str) -> list[tuple[str, str, float, float, float, float]]:
    needed = {"date", "stock_id", *RAW_PRICE_STRATEGY_PRICE_FIELDS}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"TaiwanStockPrice raw-price strategy materialization缺欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "stock_id", *RAW_PRICE_STRATEGY_PRICE_FIELDS]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["stock_id"] = local["stock_id"].astype(str).str.strip()
    valid = local["date"].notna() & local["stock_id"].ne("") & local["date"].le(str(cutoff))
    for field in RAW_PRICE_STRATEGY_PRICE_FIELDS:
        local[field] = pd.to_numeric(local[field], errors="coerce")
        valid &= local[field].notna()
    local = local.loc[valid]
    rows: list[tuple[str, str, float, float, float, float]] = []
    for row in local.itertuples(index=False, name=None):
        values = tuple(float(value) for value in row[2:])
        if any(not pd.notna(value) or value in (float("inf"), float("-inf")) for value in values):
            continue
        rows.append((str(row[0]), str(row[1]), *values))
    return rows


def _insert_raw_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, float, float, float, float]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO raw_price(date, stock_id, open, high, low, close, conflict)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            conflict = CASE WHEN
                raw_price.open = excluded.open AND raw_price.high = excluded.high
                AND raw_price.low = excluded.low AND raw_price.close = excluded.close
            THEN raw_price.conflict ELSE 1 END
        """,
        rows,
    )


def _normalize_control_rows(path: Path, *, ticker: str, cutoff: str) -> list[tuple[str, str, float]]:
    try:
        frame = pd.read_csv(path, usecols=["Date", "Volume"])
    except ValueError as exc:
        raise ValueError(f"Full control CSV缺Date/Volume欄位: {path.name}") from exc
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    valid = frame["Date"].notna() & frame["Volume"].notna() & frame["Date"].le(str(cutoff))
    local = frame.loc[valid, ["Date", "Volume"]]
    rows: list[tuple[str, str, float]] = []
    for date, volume in local.itertuples(index=False, name=None):
        value = float(volume)
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            continue
        rows.append((str(date), str(ticker), value))
    return rows


def _build_files(
    *,
    root: Path,
    view: ResearchV2ProviderView,
    control_dir: Path,
    stage_dir: Path,
    source_identity: Mapping[str, object],
) -> tuple[list[dict[str, object]], int]:
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
                conflict INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(stock_id, date)
            ) WITHOUT ROWID;
            CREATE TABLE control_grid(
                date TEXT NOT NULL,
                stock_id TEXT NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY(stock_id, date)
            ) WITHOUT ROWID;
            """
        )
        columns = ("date", "stock_id", "open", "max", "min", "close")
        for frame in view.iter_raw_price_strategy_frames(columns=columns):
            _insert_raw_rows(conn, _normalize_raw_rows(frame, cutoff=cutoff))
        conn.commit()
        conflicts = int(conn.execute("SELECT COUNT(*) FROM raw_price WHERE conflict <> 0").fetchone()[0])
        if conflicts:
            raise ValueError(f"raw-price strategy source rows存在conflicting duplicate keys: {conflicts}")

        discovered, _duplicate_issues = discover_unique_csv_inputs(str(control_dir))
        for ticker, raw_path in discovered:
            conn.executemany(
                "INSERT INTO control_grid(date, stock_id, volume) VALUES (?, ?, ?)",
                _normalize_control_rows(Path(raw_path), ticker=str(ticker), cutoff=cutoff),
            )
        conn.commit()

        missing_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM control_grid c
                LEFT JOIN raw_price r ON r.stock_id = c.stock_id AND r.date = c.date
                WHERE r.stock_id IS NULL
                """
            ).fetchone()[0]
        )
        if missing_count:
            sample = conn.execute(
                """
                SELECT c.stock_id, c.date
                FROM control_grid c
                LEFT JOIN raw_price r ON r.stock_id = c.stock_id AND r.date = c.date
                WHERE r.stock_id IS NULL
                ORDER BY c.stock_id, c.date
                LIMIT 10
                """
            ).fetchall()
            raise ValueError(
                "raw-price strategy materialization缺Full control dataset所需TaiwanStockPrice rows: "
                f"{missing_count}; sample={sample}"
            )
        tickers = [
            str(row[0])
            for row in conn.execute("SELECT DISTINCT stock_id FROM control_grid ORDER BY stock_id")
        ]
        if not tickers:
            raise ValueError("raw-price strategy materialization沒有control ticker")

        evidence: list[dict[str, object]] = []
        total_rows = 0
        fixed_mtime_ns = deterministic_compatibility_file_mtime_ns(cutoff)
        for ticker in tickers:
            if not _SAFE_TICKER_RE.fullmatch(ticker) or Path(ticker).name != ticker:
                raise ValueError(f"raw-price strategy ticker path不合法: {ticker!r}")
            frame = pd.read_sql_query(
                """
                SELECT c.date AS Date, r.open AS Open, r.high AS High, r.low AS Low,
                       r.close AS Close, c.volume AS Volume
                FROM control_grid c
                JOIN raw_price r ON r.stock_id = c.stock_id AND r.date = c.date
                WHERE c.stock_id = ?
                ORDER BY c.date
                """,
                conn,
                params=(ticker,),
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
            evidence.append({
                "ticker": ticker,
                "relative_path": path.relative_to(dataset_dir).as_posix(),
                "row_count": row_count,
                "size_bytes": int(path.stat().st_size),
                "content_sha256": compute_file_sha256(path),
            })
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
    try:
        identity, _view, _control_dir = _source_identity(root)
    except FileNotFoundError as exc:
        return {
            "ready": False,
            "status": f"RAW_PRICE_SOURCE_NOT_READY: {exc}",
            "source_identity": {},
            "source_identity_fingerprint": None,
            "manifest_path": None,
            "dataset_dir": None,
        }
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

    identity, view, control_dir = _source_identity(root)
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
            control_dir=control_dir,
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
            "generation_id": "research_raw_unadjusted_controlled_v2",
            "frozen_cutoff": str(identity["frozen_cutoff"]),
            "price_basis": RAW_PRICE_STRATEGY_PRICE_BASIS,
            "source_dataset": FINMIND_RAW_PRICE_ARCHIVE_DATASET,
            "provider_snapshot_fingerprint": str(identity["provider_snapshot_fingerprint"]),
            "control_dataset_generation_identity": dict(identity["control_dataset_generation_identity"]),
            "control_dataset_inventory_fingerprint": str(identity["control_dataset_inventory_fingerprint"]),
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
