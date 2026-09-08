"""Research Market Data generation materialization, promotion and active read seam.

This service performs no provider calls.  Promotion may only consume a validated
immutable Research V2 freeze candidate, materialize the authorized compatibility
view, persist an immutable promotion artifact, then atomically switch the small
active-generation pointer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Callable, Iterator

import pandas as pd

from config.market_data import RESEARCH_DATA_GENERATION_V1, RESEARCH_DATA_GENERATION_V2
from core.console_report import project_relative_display_path
from core.file_integrity import (
    atomic_write_json,
    canonical_json_sha256,
    compute_file_sha256,
    load_json_strict,
)
from core.market_data_contract import get_active_research_data_generation
from core.market_data_research_materialization import (
    RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY,
    RESEARCH_V2_COMPAT_OUTPUT_COLUMNS,
    RESEARCH_V2_COMPAT_PRICE_FIELD_MAPPING,
    RESEARCH_V2_COMPAT_VOLUME_FIELD_MAPPING,
    build_research_v2_compatibility_materialization_identity_payload,
    deterministic_compatibility_file_mtime_ns,
    materialization_identity_from_payload,
)
from core.market_data_research_promotion import (
    RESEARCH_ACTIVE_POINTER_SCHEMA_VERSION,
    build_research_v2_promotion_identity_payload,
    load_active_research_v2_promotion,
    promotion_identity_from_payload,
)
from core.market_data_research_scope import RESEARCH_V2_ADJUSTED_PRICE_DATASET, RESEARCH_V2_RAW_VOLUME_DATASET
from core.market_data_research_storage_contract import (
    RESEARCH_MARKET_DATA_V2_FREEZE_CANDIDATES_DIRNAME,
    RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT,
    resolve_active_research_generation_path,
    resolve_research_v2_compatibility_dataset_dir,
    resolve_research_v2_daily_universe_path,
    resolve_research_v2_freeze_candidate_manifest_path,
    resolve_research_v2_materialization_dir,
    resolve_research_v2_materialization_manifest_path,
    resolve_research_v2_promotion_dir,
    resolve_research_v2_promotion_manifest_path,
)
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.research.market_data_v2 import (
    FrameReader,
    HashFn,
    ResearchV2ProviderView,
    build_research_v2_freeze_candidate,
    load_research_v2_freeze_candidate,
)

ProgressCallback = Callable[[str], None]
_SAFE_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ActiveResearchV2ReadView:
    project_root: Path
    promotion: dict[str, object]
    freeze_candidate: dict[str, object]
    provider_view: ResearchV2ProviderView
    daily_universe_path: Path
    compatibility_dataset_dir: Path

    @property
    def frozen_cutoff(self) -> str:
        return str(self.promotion["frozen_cutoff"])

    def eligible_stock_ids(self, date_value: str) -> tuple[str, ...]:
        target = str(date_value or "").strip()
        if not target:
            raise ValueError("Research V2 eligible_stock_ids date 不可空白")
        if target > self.frozen_cutoff:
            raise ValueError("Research V2 active read 不得超過 frozen cutoff")
        conn = sqlite3.connect(self.daily_universe_path)
        try:
            rows = conn.execute(
                "SELECT stock_id FROM daily_universe WHERE date = ? ORDER BY stock_id",
                (target,),
            ).fetchall()
        finally:
            conn.close()
        return tuple(str(row[0]) for row in rows)

    def iter_dataset_scope_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...],
    ) -> Iterator[pd.DataFrame]:
        yield from self.provider_view.iter_dataset_scope_frames(dataset, columns=columns)


def _emit(progress_callback: ProgressCallback | None, text: str) -> None:
    if progress_callback is not None:
        progress_callback(str(text))


def _normalize_adjusted_rows(frame: pd.DataFrame, *, cutoff: str) -> list[tuple[str, str, float, float, float, float]]:
    needed = {"date", "stock_id", *RESEARCH_V2_COMPAT_PRICE_FIELD_MAPPING.values()}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"TaiwanStockPriceAdj compatibility materialization 缺欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "stock_id", "open", "max", "min", "close"]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["stock_id"] = local["stock_id"].astype(str).str.strip()
    valid = local["date"].notna() & local["stock_id"].ne("") & local["date"].le(cutoff)
    for field in ("open", "max", "min", "close"):
        local[field] = pd.to_numeric(local[field], errors="coerce")
        valid &= local[field].notna() & local[field].map(lambda value: bool(pd.notna(value)) and float(value) not in (float("inf"), float("-inf")))
    local = local.loc[valid]
    return [
        (str(row.date), str(row.stock_id), float(row.open), float(row.max), float(row.min), float(row.close))
        for row in local.itertuples(index=False)
    ]


def _normalize_volume_rows(frame: pd.DataFrame, *, cutoff: str) -> list[tuple[str, str, float]]:
    source_field = RESEARCH_V2_COMPAT_VOLUME_FIELD_MAPPING["Volume"]
    needed = {"date", "stock_id", source_field}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"TaiwanStockPrice compatibility materialization 缺欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "stock_id", source_field]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["stock_id"] = local["stock_id"].astype(str).str.strip()
    local[source_field] = pd.to_numeric(local[source_field], errors="coerce")
    valid = local["date"].notna() & local["stock_id"].ne("") & local["date"].le(cutoff)
    valid &= local[source_field].notna()
    local = local.loc[valid]
    return [
        (str(row[0]), str(row[1]), float(row[2]))
        for row in local.itertuples(index=False, name=None)
        if float(row[2]) not in (float("inf"), float("-inf"))
    ]


def _insert_adjusted_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, float, float, float, float]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO adjusted(date, stock_id, open, high, low, close, conflict)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            conflict = CASE WHEN
                adjusted.open = excluded.open AND adjusted.high = excluded.high
                AND adjusted.low = excluded.low AND adjusted.close = excluded.close
            THEN adjusted.conflict ELSE 1 END
        """,
        rows,
    )


def _insert_volume_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, float]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO volume(date, stock_id, volume, conflict)
        VALUES (?, ?, ?, 0)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            conflict = CASE WHEN volume.volume = excluded.volume THEN volume.conflict ELSE 1 END
        """,
        rows,
    )


def _build_materialization_files(
    *,
    project_root: Path,
    freeze_candidate: dict[str, object],
    stage_dir: Path,
    frame_reader: FrameReader | None,
    hash_fn: HashFn | None,
    progress_callback: ProgressCallback | None,
) -> tuple[list[dict[str, object]], int]:
    cutoff = str(freeze_candidate["frozen_cutoff"])
    archive = load_ready_provider_snapshot_archive(
        project_root,
        snapshot_fingerprint=str(freeze_candidate["provider_snapshot_fingerprint"]),
    )
    view = ResearchV2ProviderView(
        project_root=project_root,
        archive=archive,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
    )
    universe_path = resolve_research_v2_daily_universe_path(
        project_root,
        str(freeze_candidate["provider_snapshot_fingerprint"]),
    )
    if not universe_path.is_file():
        raise FileNotFoundError("Research V2 promotion daily_universe.sqlite3 不存在")
    actual_universe_hash = compute_file_sha256(universe_path)
    if actual_universe_hash != str(freeze_candidate["daily_universe_file_sha256"]):
        raise ValueError("Research V2 promotion daily universe physical hash drift")

    dataset_dir = stage_dir / "tw_stock_data_vip"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    merge_path = stage_dir / ".compatibility_merge.sqlite3"
    conn = sqlite3.connect(merge_path)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE adjusted(
                date TEXT NOT NULL,
                stock_id TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                conflict INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(stock_id, date)
            ) WITHOUT ROWID;
            CREATE TABLE volume(
                date TEXT NOT NULL,
                stock_id TEXT NOT NULL,
                volume REAL NOT NULL,
                conflict INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(stock_id, date)
            ) WITHOUT ROWID;
            """
        )
        for frame in view.iter_dataset_scope_frames(
            RESEARCH_V2_ADJUSTED_PRICE_DATASET,
            columns=("date", "stock_id", "open", "max", "min", "close"),
        ):
            _insert_adjusted_rows(conn, _normalize_adjusted_rows(frame, cutoff=cutoff))
        for frame in view.iter_dataset_scope_frames(
            RESEARCH_V2_RAW_VOLUME_DATASET,
            columns=("date", "stock_id", RESEARCH_V2_COMPAT_VOLUME_FIELD_MAPPING["Volume"]),
        ):
            _insert_volume_rows(conn, _normalize_volume_rows(frame, cutoff=cutoff))
        conn.commit()
        adjusted_conflicts = int(conn.execute("SELECT COUNT(*) FROM adjusted WHERE conflict <> 0").fetchone()[0])
        volume_conflicts = int(conn.execute("SELECT COUNT(*) FROM volume WHERE conflict <> 0").fetchone()[0])
        if adjusted_conflicts or volume_conflicts:
            raise ValueError(
                "Research V2 compatibility source rows have conflicting duplicate keys: "
                f"adjusted={adjusted_conflicts}, volume={volume_conflicts}"
            )
        conn.execute("ATTACH DATABASE ? AS candidate", (str(universe_path),))
        missing_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM candidate.daily_universe u
                LEFT JOIN adjusted a ON a.stock_id = u.stock_id AND a.date = u.date
                LEFT JOIN volume v ON v.stock_id = u.stock_id AND v.date = u.date
                WHERE u.date <= ? AND (a.stock_id IS NULL OR v.stock_id IS NULL)
                """,
                (cutoff,),
            ).fetchone()[0]
        )
        if missing_count:
            raise ValueError(f"Research V2 compatibility materialization 缺 required OHLCV rows: {missing_count}")
        tickers = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT stock_id FROM candidate.daily_universe WHERE date <= ? ORDER BY stock_id",
                (cutoff,),
            )
        ]
        if not tickers:
            raise ValueError("Research V2 compatibility materialization 沒有任何 eligible ticker")
        evidence: list[dict[str, object]] = []
        total_rows = 0
        fixed_mtime_ns = deterministic_compatibility_file_mtime_ns(cutoff)
        for index, ticker in enumerate(tickers, start=1):
            if not _SAFE_TICKER_RE.fullmatch(ticker) or Path(ticker).name != ticker:
                raise ValueError(f"Research V2 compatibility ticker path 不合法: {ticker!r}")
            frame = pd.read_sql_query(
                """
                SELECT u.date AS Date, a.open AS Open, a.high AS High, a.low AS Low,
                       a.close AS Close, v.volume AS Volume
                FROM candidate.daily_universe u
                JOIN adjusted a ON a.stock_id = u.stock_id AND a.date = u.date
                JOIN volume v ON v.stock_id = u.stock_id AND v.date = u.date
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
            frame.loc[:, list(RESEARCH_V2_COMPAT_OUTPUT_COLUMNS)].to_csv(path, index=False, lineterminator="\n")
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
            if index % 100 == 0 or index == len(tickers):
                _emit(progress_callback, f"[Research V2 materialize] {index}/{len(tickers)} tickers")
        evidence.sort(key=lambda row: (str(row["ticker"]), str(row["relative_path"])))
        return evidence, total_rows
    finally:
        conn.close()
        merge_path.unlink(missing_ok=True)


def validate_research_v2_compatibility_materialization(
    project_root,
    *,
    materialization_fingerprint: str,
    deep: bool = True,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    manifest_path = resolve_research_v2_materialization_manifest_path(root, materialization_fingerprint)
    if not manifest_path.is_file():
        raise FileNotFoundError("Research V2 compatibility materialization manifest 尚未建立")
    payload = load_json_strict(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Research V2 compatibility materialization manifest 必須是 object")
    identity = materialization_identity_from_payload(payload)
    actual_fp = canonical_json_sha256(identity)
    wanted = str(materialization_fingerprint).strip().lower()
    if actual_fp != wanted or str(payload.get("materialization_fingerprint") or "") != wanted:
        raise ValueError("Research V2 compatibility materialization identity drift")
    if str(payload.get("status") or "") != RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY:
        raise ValueError("Research V2 compatibility materialization 尚未 READY")
    freeze = load_research_v2_freeze_candidate(
        root,
        freeze_candidate_fingerprint=str(payload.get("freeze_candidate_fingerprint") or ""),
        required=True,
    )
    expected_identity = build_research_v2_compatibility_materialization_identity_payload(freeze)
    if identity != expected_identity:
        raise ValueError("Research V2 compatibility materialization source contract drift")
    dataset_dir = resolve_research_v2_compatibility_dataset_dir(root, wanted)
    if not dataset_dir.is_dir():
        raise FileNotFoundError("Research V2 compatibility materialized dataset directory 不存在")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Research V2 compatibility materialization files evidence 不合法")
    if canonical_json_sha256(files) != str(payload.get("dataset_inventory_sha256") or ""):
        raise ValueError("Research V2 compatibility materialization inventory fingerprint drift")
    if deep:
        expected_paths = {str(row.get("relative_path") or "") for row in files if isinstance(row, dict)}
        actual_paths = {path.relative_to(dataset_dir).as_posix() for path in dataset_dir.glob("*.csv") if path.is_file()}
        if expected_paths != actual_paths:
            raise ValueError("Research V2 compatibility materialization CSV file set drift")
        for row in files:
            if not isinstance(row, dict):
                raise ValueError("Research V2 compatibility materialization file evidence row 不合法")
            rel = str(row.get("relative_path") or "")
            path = dataset_dir / rel
            if not path.is_file():
                raise FileNotFoundError(f"Research V2 compatibility CSV 不存在: {rel}")
            if int(path.stat().st_size) != int(row.get("size_bytes") or -1):
                raise ValueError(f"Research V2 compatibility CSV size drift: {rel}")
            if compute_file_sha256(path) != str(row.get("content_sha256") or ""):
                raise ValueError(f"Research V2 compatibility CSV SHA256 drift: {rel}")
    return dict(payload)


def build_research_v2_compatibility_materialization(
    project_root,
    *,
    freeze_candidate_fingerprint: str,
    frame_reader: FrameReader | None = None,
    hash_fn: HashFn | None = None,
    now: datetime | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    freeze = load_research_v2_freeze_candidate(
        root,
        freeze_candidate_fingerprint=freeze_candidate_fingerprint,
        required=True,
    )
    identity = build_research_v2_compatibility_materialization_identity_payload(freeze)
    materialization_fp = canonical_json_sha256(identity)
    final_dir = resolve_research_v2_materialization_dir(root, materialization_fp)
    manifest_path = resolve_research_v2_materialization_manifest_path(root, materialization_fp)
    if manifest_path.is_file():
        loaded = validate_research_v2_compatibility_materialization(
            root,
            materialization_fingerprint=materialization_fp,
            deep=True,
        )
        return {**loaded, "reused": True, "provider_calls": 0}
    if final_dir.exists():
        raise ValueError("Research V2 materialization immutable path 已存在但 manifest 缺失")

    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{materialization_fp}.", suffix=".tmp", dir=str(parent)))
    try:
        files, total_rows = _build_materialization_files(
            project_root=root,
            freeze_candidate=freeze,
            stage_dir=stage_dir,
            frame_reader=frame_reader,
            hash_fn=hash_fn,
            progress_callback=progress_callback,
        )
        dataset_inventory_sha256 = canonical_json_sha256(files)
        created_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        payload = {
            **identity,
            "materialization_fingerprint": materialization_fp,
            "status": RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY,
            "created_at": created_at,
            "dataset_inventory_sha256": dataset_inventory_sha256,
            "csv_file_count": len(files),
            "row_count": int(total_rows),
            "full_dataset_dir": project_relative_display_path(
                resolve_research_v2_compatibility_dataset_dir(root, materialization_fp), project_root=root
            ),
            "files": files,
            "provider_calls": 0,
        }
        atomic_write_json(stage_dir / manifest_path.name, payload)
        try:
            os.replace(stage_dir, final_dir)
        except FileExistsError:
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            loaded = validate_research_v2_compatibility_materialization(
                root,
                materialization_fingerprint=materialization_fp,
                deep=True,
            )
            return {**loaded, "reused": True, "provider_calls": 0}
        loaded = validate_research_v2_compatibility_materialization(
            root,
            materialization_fingerprint=materialization_fp,
            deep=True,
        )
        return {**loaded, "reused": False, "provider_calls": 0}
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def discover_research_v2_freeze_candidates(project_root) -> tuple[dict[str, object], ...]:
    root = Path(project_root).resolve()
    base = root / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT / RESEARCH_MARKET_DATA_V2_FREEZE_CANDIDATES_DIRNAME
    rows: list[dict[str, object]] = []
    if not base.is_dir():
        return ()
    for child in sorted(base.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or len(child.name) != 64:
            continue
        try:
            payload = load_research_v2_freeze_candidate(
                root,
                freeze_candidate_fingerprint=child.name,
                required=True,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            continue
        rows.append(payload)
    rows.sort(
        key=lambda row: (
            str(row.get("provider_as_of_date") or ""),
            str(row.get("freeze_candidate_fingerprint") or ""),
        )
    )
    return tuple(rows)


def promote_research_v2(
    project_root,
    *,
    freeze_candidate_fingerprint: str,
    explicit_authorization: bool,
    frame_reader: FrameReader | None = None,
    hash_fn: HashFn | None = None,
    now: datetime | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    if not explicit_authorization:
        raise PermissionError("Research V2 promotion 需要明確使用者 promotion action")
    root = Path(project_root).resolve()
    existing = load_active_research_v2_promotion(root, required=False)
    if existing is not None:
        if existing.freeze_candidate_fingerprint != str(freeze_candidate_fingerprint).strip().lower():
            raise RuntimeError(
                "Research V2 已由不同 freeze candidate ACTIVE；不得以同一 Round-16 seam 隱性 repromotion"
            )
        materialization = validate_research_v2_compatibility_materialization(
            root,
            materialization_fingerprint=existing.materialization_fingerprint,
            deep=True,
        )
        return {
            **existing.payload,
            "promotion_fingerprint": existing.promotion_fingerprint,
            "materialization": materialization,
            "reused": True,
            "provider_calls": 0,
        }

    prior = get_active_research_data_generation(root)
    if prior.generation_id != RESEARCH_DATA_GENERATION_V1:
        raise RuntimeError(f"Research V2 promotion prior generation 不合法: {prior.generation_id}")
    freeze = load_research_v2_freeze_candidate(
        root,
        freeze_candidate_fingerprint=freeze_candidate_fingerprint,
        required=True,
    )
    materialization = build_research_v2_compatibility_materialization(
        root,
        freeze_candidate_fingerprint=freeze_candidate_fingerprint,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
        now=now,
        progress_callback=progress_callback,
    )
    freeze_manifest_path = resolve_research_v2_freeze_candidate_manifest_path(root, freeze_candidate_fingerprint)
    materialization_manifest_path = resolve_research_v2_materialization_manifest_path(
        root,
        str(materialization["materialization_fingerprint"]),
    )
    identity = build_research_v2_promotion_identity_payload(
        freeze_candidate=freeze,
        freeze_candidate_manifest_sha256=compute_file_sha256(freeze_manifest_path),
        materialization=materialization,
        materialization_manifest_sha256=compute_file_sha256(materialization_manifest_path),
        prior_generation_id=prior.generation_id,
    )
    promotion_fp = canonical_json_sha256(identity)
    promotion_dir = resolve_research_v2_promotion_dir(root, promotion_fp)
    promotion_manifest_path = resolve_research_v2_promotion_manifest_path(root, promotion_fp)
    promoted_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    payload = {
        **identity,
        "promotion_fingerprint": promotion_fp,
        "promoted_at": promoted_at,
        "freeze_candidate_manifest_path": project_relative_display_path(freeze_manifest_path, project_root=root),
        "materialization_manifest_path": project_relative_display_path(materialization_manifest_path, project_root=root),
        "full_dataset_dir": project_relative_display_path(
            resolve_research_v2_compatibility_dataset_dir(root, str(materialization["materialization_fingerprint"])),
            project_root=root,
        ),
        "provider_calls": 0,
    }
    if promotion_manifest_path.is_file():
        existing_payload = load_json_strict(promotion_manifest_path)
        if not isinstance(existing_payload, dict):
            raise ValueError("Research V2 promotion manifest 必須是 object")
        if promotion_identity_from_payload(existing_payload) != identity:
            raise ValueError("Research V2 immutable promotion path identity collision/drift")
    else:
        promotion_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(promotion_manifest_path, payload)
    promotion_manifest_sha = compute_file_sha256(promotion_manifest_path)
    pointer_payload = {
        "schema_version": RESEARCH_ACTIVE_POINTER_SCHEMA_VERSION,
        "generation_id": RESEARCH_DATA_GENERATION_V2,
        "promotion_fingerprint": promotion_fp,
        "promotion_manifest_sha256": promotion_manifest_sha,
        "activated_at": promoted_at,
    }
    atomic_write_json(resolve_active_research_generation_path(root), pointer_payload)
    active = load_active_research_v2_promotion(root, required=True)
    if active.promotion_fingerprint != promotion_fp:
        raise RuntimeError("Research V2 active pointer publication verification failed")
    return {
        **payload,
        "materialization": materialization,
        "reused": False,
        "provider_calls": 0,
    }


def load_active_research_v2_read_view(
    project_root,
    *,
    frame_reader: FrameReader | None = None,
    hash_fn: HashFn | None = None,
) -> ActiveResearchV2ReadView:
    root = Path(project_root).resolve()
    active = load_active_research_v2_promotion(root, required=True)
    freeze = load_research_v2_freeze_candidate(
        root,
        freeze_candidate_fingerprint=active.freeze_candidate_fingerprint,
        required=True,
    )
    universe_path = resolve_research_v2_daily_universe_path(
        root,
        str(freeze["provider_snapshot_fingerprint"]),
    )
    if compute_file_sha256(universe_path) != str(freeze["daily_universe_file_sha256"]):
        raise ValueError("active Research V2 daily universe SHA256 drift")
    archive = load_ready_provider_snapshot_archive(
        root,
        snapshot_fingerprint=str(freeze["provider_snapshot_fingerprint"]),
    )
    provider_view = ResearchV2ProviderView(
        project_root=root,
        archive=archive,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
    )
    return ActiveResearchV2ReadView(
        project_root=root,
        promotion=dict(active.payload),
        freeze_candidate=freeze,
        provider_view=provider_view,
        daily_universe_path=universe_path,
        compatibility_dataset_dir=active.full_dataset_dir,
    )


def collect_research_market_data_status(project_root, *, deep: bool = False) -> dict[str, object]:
    root = Path(project_root).resolve()
    active_contract = get_active_research_data_generation(root)
    active = load_active_research_v2_promotion(root, required=False)
    freezes = discover_research_v2_freeze_candidates(root)
    payload: dict[str, object] = {
        "effective_generation_id": active_contract.generation_id,
        "status": active_contract.status,
        "frozen_cutoff": active_contract.cutoff,
        "freeze_candidate_count": len(freezes),
        "freeze_candidates": [
            {
                "freeze_candidate_fingerprint": str(row.get("freeze_candidate_fingerprint") or ""),
                "provider_as_of_date": str(row.get("provider_as_of_date") or ""),
                "provider_snapshot_fingerprint": str(row.get("provider_snapshot_fingerprint") or ""),
            }
            for row in freezes
        ],
        "promotion_fingerprint": None,
        "materialization_fingerprint": None,
        "full_dataset_dir": None,
    }
    if active is not None:
        if deep:
            validate_research_v2_compatibility_materialization(
                root,
                materialization_fingerprint=active.materialization_fingerprint,
                deep=True,
            )
        payload.update(
            {
                "promotion_fingerprint": active.promotion_fingerprint,
                "materialization_fingerprint": active.materialization_fingerprint,
                "full_dataset_dir": project_relative_display_path(active.full_dataset_dir, project_root=root),
            }
        )
    return payload


__all__ = [
    "ActiveResearchV2ReadView",
    "build_research_v2_compatibility_materialization",
    "validate_research_v2_compatibility_materialization",
    "discover_research_v2_freeze_candidates",
    "promote_research_v2",
    "load_active_research_v2_read_view",
    "collect_research_market_data_status",
    "build_research_v2_freeze_candidate",
]
