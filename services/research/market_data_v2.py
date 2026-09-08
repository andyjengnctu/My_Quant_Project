"""Research Market Data V2 candidate builder and immutable provider read view.

The builder is local-only: it consumes one READY neutral Provider Snapshot and
never reads the Trading V2 overlay or calls FinMind.  It materializes a derived
PIT-universe/coverage index plus a NOT_READY candidate manifest.  Promotion to
active/frozen Research truth is intentionally outside this module.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Callable, Iterator

import pandas as pd

from config.market_data import ACTIVE_RESEARCH_DATA_GENERATION, RESEARCH_DATA_GENERATION_V1, RESEARCH_DATA_GENERATION_V2
from core.console_report import project_relative_display_path
from core.file_integrity import atomic_replace_with_retry, atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.market_data_contract import RESEARCH_STATUS_AUTHORIZED_NOT_READY, get_research_data_generation
from core.market_data_research_storage_contract import (
    resolve_research_v2_candidate_dir,
    resolve_research_v2_candidate_manifest_path,
    resolve_research_v2_daily_universe_path,
)
from core.market_data_research_v2 import (
    RESEARCH_V2_ADJUSTED_PRICE_DATASET,
    RESEARCH_V2_CANDIDATE_STATUS_NOT_READY,
    RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS,
    RESEARCH_V2_DAILY_COVERAGE_DATASET,
    RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
    RESEARCH_V2_EVENT_EVIDENCE_DATASET,
    RESEARCH_V2_TRADING_CALENDAR_DATASET,
    build_research_v2_candidate_identity_payload,
    build_research_v2_dataset_assessments,
    research_v2_dataset_assessment_fingerprint,
    ResearchV2ExactCoverageSummary,
    summarize_exact_candidate_coverage_table,
    validate_research_v2_candidate_contract,
)
from core.market_data_storage_contract import resolve_market_data_request_parquet_path
from services.market_data.provider_snapshot_repository import (
    ReadyProviderSnapshotArchive,
    load_ready_provider_snapshot_archive,
)


FrameReader = Callable[[Path, tuple[str, ...] | None], pd.DataFrame]
HashFn = Callable[[Path], str]


class ResearchV2ProviderView:
    """Pinned immutable read view over one neutral Provider Snapshot."""

    def __init__(
        self,
        *,
        project_root,
        archive: ReadyProviderSnapshotArchive,
        frame_reader: FrameReader | None = None,
        hash_fn: HashFn | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.archive = archive
        self._frame_reader = frame_reader or self._read_parquet
        self._hash_fn = hash_fn or compute_file_sha256
        self._artifacts_by_dataset: dict[str, tuple] = {}
        grouped: dict[str, list] = defaultdict(list)
        for item in archive.artifacts:
            grouped[item.dataset].append(item)
        self._artifacts_by_dataset = {key: tuple(value) for key, value in grouped.items()}
        self._assessment_by_dataset = {row.dataset: row for row in build_research_v2_dataset_assessments()}

    @staticmethod
    def _read_parquet(path: Path, columns: tuple[str, ...] | None) -> pd.DataFrame:
        if not columns:
            return pd.read_parquet(path)
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Research V2 Parquet read 需要 pyarrow") from exc
        parquet = pq.ParquetFile(path)
        names = tuple(str(name) for name in parquet.schema_arrow.names)
        missing = [column for column in columns if column not in names]
        if missing:
            if int(parquet.metadata.num_rows) == 0 and not names:
                return pd.DataFrame(columns=list(columns))
            raise ValueError(f"Provider Snapshot parquet 缺少欄位: {missing}")
        return pd.read_parquet(path, columns=list(columns))

    def dataset_artifacts(self, dataset: str):
        name = str(dataset or "").strip()
        if name not in self._assessment_by_dataset:
            raise ValueError(f"Research V2 未登記 dataset: {name}")
        return self._artifacts_by_dataset.get(name, ())

    def historical_instruments(self) -> tuple[str, ...]:
        ids = sorted(
            {
                str(item.data_id or "").strip()
                for item in self.dataset_artifacts(RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET)
                if str(item.data_id or "").strip()
            }
        )
        expected = int(self.archive.payload.get("historical_instrument_count") or 0)
        if len(ids) != expected:
            raise ValueError(
                "Research V2 historical instrument pool 與 Provider Snapshot 不一致: "
                f"actual={len(ids)}, expected={expected}"
            )
        return tuple(ids)

    def iter_dataset_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None = None,
        exact_pit_audit_only: bool = True,
    ) -> Iterator[pd.DataFrame]:
        name = str(dataset or "").strip()
        assessment = self._assessment_by_dataset.get(name)
        if assessment is None:
            raise ValueError(f"Research V2 未登記 dataset: {name}")
        if exact_pit_audit_only and not assessment.automatically_usable_for_pit_audit:
            raise RuntimeError(
                f"Research V2 dataset 尚未取得 PIT audit 自動讀取資格: {name} | {assessment.candidate_status}"
            )
        for artifact in self.dataset_artifacts(name):
            request = artifact.to_request()
            path = resolve_market_data_request_parquet_path(
                self.project_root,
                self.archive.manifest_fingerprint,
                request,
            )
            if not path.is_file():
                raise FileNotFoundError(f"Provider Snapshot artifact 不存在: {project_relative_display_path(path, project_root=self.project_root)}")
            actual_hash = str(self._hash_fn(path) or "").strip().lower()
            if actual_hash != artifact.content_sha256:
                raise ValueError(f"Provider Snapshot artifact SHA256 drift: {request.request_id}")
            frame = self._frame_reader(path, columns)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Research V2 provider frame reader 必須回傳 pandas.DataFrame")
            yield frame



def _connect_universe_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _init_universe_db(conn: sqlite3.Connection) -> None:
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
        CREATE TABLE price_limit_evidence (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            PRIMARY KEY(date, stock_id)
        ) WITHOUT ROWID;
        CREATE TABLE exact_coverage (
            date TEXT PRIMARY KEY,
            universe_count INTEGER NOT NULL,
            price_limit_count INTEGER NOT NULL,
            missing_price_limit_count INTEGER NOT NULL,
            extra_price_limit_count INTEGER NOT NULL,
            exact_complete INTEGER NOT NULL CHECK(exact_complete IN (0, 1))
        );
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def _normalize_date_stock(frame: pd.DataFrame, *, dataset: str) -> list[tuple[str, str]]:
    required = {"date", "stock_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset} 缺少 Research V2 universe/coverage 欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "stock_id"]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["stock_id"] = local["stock_id"].astype(str).str.strip()
    local = local.loc[local["date"].notna() & local["stock_id"].ne("")]
    local = local.drop_duplicates(["date", "stock_id"])
    return [(str(row.date), str(row.stock_id)) for row in local.itertuples(index=False)]


def _stream_logical_fingerprint(conn: sqlite3.Connection, query: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in conn.execute(query):
        encoded = json.dumps(list(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def _build_exact_candidate_sqlite(
    view: ResearchV2ProviderView,
    *,
    output_path: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".daily_universe.", suffix=".sqlite3.tmp", dir=str(output_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        conn = _connect_universe_db(temp_path)
        try:
            _init_universe_db(conn)
            as_of = view.archive.as_of_date
            trading_rows: set[str] = set()
            for frame in view.iter_dataset_frames(RESEARCH_V2_TRADING_CALENDAR_DATASET, columns=("date",)):
                if "date" not in frame.columns:
                    raise ValueError("TaiwanStockTradingDate 缺少 date")
                dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                trading_rows.update(str(value) for value in dates.dropna().tolist() if str(value) <= as_of)
            if not trading_rows:
                raise ValueError("Research V2 trading calendar evidence 不可為空")
            conn.executemany("INSERT OR IGNORE INTO trading_dates(date) VALUES (?)", ((value,) for value in sorted(trading_rows)))

            historical_pool = set(view.historical_instruments())
            for frame in view.iter_dataset_frames(
                RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = [row for row in _normalize_date_stock(frame, dataset=RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET) if row[0] <= as_of and row[1] in historical_pool and row[0] in trading_rows]
                if rows:
                    conn.executemany("INSERT OR IGNORE INTO daily_universe(date, stock_id) VALUES (?, ?)", rows)

            for frame in view.iter_dataset_frames(
                RESEARCH_V2_DAILY_COVERAGE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = [row for row in _normalize_date_stock(frame, dataset=RESEARCH_V2_DAILY_COVERAGE_DATASET) if row[0] <= as_of and row[1] in historical_pool and row[0] in trading_rows]
                if rows:
                    conn.executemany("INSERT OR IGNORE INTO price_limit_evidence(date, stock_id) VALUES (?, ?)", rows)

            # Delisting is an exact event dataset.  Reading it here verifies the
            # immutable artifact chain, but event no-row is legal and therefore
            # it is not a daily coverage denominator.
            delisting_rows = 0
            for frame in view.iter_dataset_frames(
                RESEARCH_V2_EVENT_EVIDENCE_DATASET,
                columns=("date", "stock_id"),
            ):
                delisting_rows += len(_normalize_date_stock(frame, dataset=RESEARCH_V2_EVENT_EVIDENCE_DATASET)) if len(frame) else 0

            conn.execute(
                """
                INSERT INTO exact_coverage(
                    date, universe_count, price_limit_count,
                    missing_price_limit_count, extra_price_limit_count, exact_complete
                )
                SELECT
                    t.date,
                    (SELECT COUNT(*) FROM daily_universe u WHERE u.date = t.date) AS universe_count,
                    (SELECT COUNT(*) FROM price_limit_evidence p WHERE p.date = t.date) AS price_limit_count,
                    (SELECT COUNT(*) FROM daily_universe u
                        LEFT JOIN price_limit_evidence p
                          ON p.date = u.date AND p.stock_id = u.stock_id
                      WHERE u.date = t.date AND p.stock_id IS NULL) AS missing_price_limit_count,
                    (SELECT COUNT(*) FROM price_limit_evidence p
                        LEFT JOIN daily_universe u
                          ON u.date = p.date AND u.stock_id = p.stock_id
                      WHERE p.date = t.date AND u.stock_id IS NULL) AS extra_price_limit_count,
                    CASE WHEN
                        (SELECT COUNT(*) FROM daily_universe u WHERE u.date = t.date) > 0
                        AND (SELECT COUNT(*) FROM daily_universe u
                              LEFT JOIN price_limit_evidence p
                                ON p.date = u.date AND p.stock_id = u.stock_id
                            WHERE u.date = t.date AND p.stock_id IS NULL) = 0
                    THEN 1 ELSE 0 END AS exact_complete
                FROM trading_dates t
                ORDER BY t.date
                """
            )
            universe_count, universe_fingerprint = _stream_logical_fingerprint(
                conn,
                "SELECT date, stock_id FROM daily_universe ORDER BY date, stock_id",
            )
            coverage = pd.read_sql_query(
                "SELECT date, universe_count, price_limit_count, missing_price_limit_count, extra_price_limit_count, exact_complete FROM exact_coverage ORDER BY date",
                conn,
            )
            coverage["exact_complete"] = coverage["exact_complete"].astype(bool)
            summary = summarize_exact_candidate_coverage_table(
                coverage,
                provider_as_of_date=as_of,
                daily_universe_row_count=universe_count,
            )
            metadata = {
                "provider_snapshot_fingerprint": view.archive.snapshot_fingerprint,
                "provider_manifest_fingerprint": view.archive.manifest_fingerprint,
                "provider_as_of_date": as_of,
                "historical_instrument_count": len(historical_pool),
                "daily_universe_row_count": universe_count,
                "daily_universe_fingerprint": universe_fingerprint,
                "exact_coverage_fingerprint": summary.coverage_fingerprint,
                "delisting_event_row_count": int(delisting_rows),
            }
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ((key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in metadata.items()),
            )
            conn.execute("DROP TABLE price_limit_evidence")
            conn.commit()
        finally:
            conn.close()
        atomic_replace_with_retry(temp_path, output_path)
        return {
            **metadata,
            "coverage_summary": asdict(summary),
        }, coverage
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _build_candidate_blockers(contract_stats: dict[str, object], coverage_summary: dict[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = [
        {
            "code": "RESEARCH_DATASET_SCOPE_NOT_AUTHORIZED",
            "reason": "Research V2 尚未宣告未來模型真正 required dataset set；不得把 51 個 archive dataset 自動當成 Research inputs。",
        },
        {
            "code": "ADJUSTED_PRICE_CURRENT_VINTAGE_PIT_REVIEW_REQUIRED",
            "dataset": RESEARCH_V2_ADJUSTED_PRICE_DATASET,
            "reason": "canonical adjusted price 是 current-vintage；retrospective feature/target 必須先證明 PIT legality / representation invariance。",
        },
    ]
    review_count = int(contract_stats.get("review_required_count") or 0)
    if review_count:
        blockers.append(
            {
                "code": "DATASET_SPECIFIC_PIT_REVIEW_REQUIRED",
                "dataset_count": review_count,
                "reason": "review_required datasets 必須逐 dataset 定義 publication/information-time legality 後才能加入 Research scope。",
            }
        )
    if not coverage_summary.get("latest_exact_complete_date"):
        blockers.append(
            {
                "code": "EXACT_CANDIDATE_COMMON_COMPLETE_NOT_ESTABLISHED",
                "reason": "exact-candidate daily universe / coverage 尚未得到任何完整交易日。",
            }
        )
    return blockers


def build_research_v2_candidate(
    project_root,
    *,
    snapshot_fingerprint: str | None = None,
    frame_reader: FrameReader | None = None,
    hash_fn: HashFn | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    generation = get_research_data_generation(RESEARCH_DATA_GENERATION_V2)
    if generation.status != RESEARCH_STATUS_AUTHORIZED_NOT_READY or generation.cutoff is not None:
        raise RuntimeError("Research V2 candidate builder 只允許 authorized_not_ready / cutoff=None 狀態")
    if ACTIVE_RESEARCH_DATA_GENERATION != RESEARCH_DATA_GENERATION_V1:
        raise RuntimeError("Research V2 candidate build 不得在本輪自行切換 ACTIVE Research generation")

    contract_stats = validate_research_v2_candidate_contract()
    archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
    view = ResearchV2ProviderView(
        project_root=root,
        archive=archive,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
    )
    candidate_dir = resolve_research_v2_candidate_dir(root, archive.snapshot_fingerprint)
    universe_path = resolve_research_v2_daily_universe_path(root, archive.snapshot_fingerprint)
    derived, _coverage_table = _build_exact_candidate_sqlite(view, output_path=universe_path)
    del _coverage_table
    coverage_summary = dict(derived["coverage_summary"])
    assessment_rows = build_research_v2_dataset_assessments()
    assessment_fingerprint = research_v2_dataset_assessment_fingerprint(assessment_rows)
    identity = build_research_v2_candidate_identity_payload(
        provider_snapshot_fingerprint=archive.snapshot_fingerprint,
        provider_manifest_fingerprint=archive.manifest_fingerprint,
        provider_as_of_date=archive.as_of_date,
        historical_instrument_count=int(derived["historical_instrument_count"]),
        daily_universe_fingerprint=str(derived["daily_universe_fingerprint"]),
        coverage_summary=ResearchV2ExactCoverageSummary(**coverage_summary),
        assessment_fingerprint=assessment_fingerprint,
    )
    candidate_fingerprint = canonical_json_sha256(identity)
    built_at = (now or datetime.now().astimezone()).astimezone().isoformat()
    payload = {
        **identity,
        "candidate_fingerprint": candidate_fingerprint,
        "built_at": built_at,
        "provider_snapshot_path": project_relative_display_path(archive.path, project_root=root),
        "daily_universe_path": project_relative_display_path(universe_path, project_root=root),
        "daily_universe_file_sha256": compute_file_sha256(universe_path),
        "daily_universe_row_count": int(derived["daily_universe_row_count"]),
        "daily_universe_date_count": int(coverage_summary.get("daily_universe_date_count") or 0),
        "exact_coverage": coverage_summary,
        "dataset_assessments": [asdict(row) for row in assessment_rows],
        "dataset_assessment_counts": contract_stats,
        "blockers": _build_candidate_blockers(contract_stats, coverage_summary),
        "promotion_authorized": False,
        "active_research_generation": ACTIVE_RESEARCH_DATA_GENERATION,
    }
    manifest_path = resolve_research_v2_candidate_manifest_path(root, archive.snapshot_fingerprint)
    atomic_write_json(manifest_path, payload)
    return {
        **payload,
        "manifest_path": project_relative_display_path(manifest_path, project_root=root),
        "provider_calls": 0,
    }


def load_research_v2_candidate(
    project_root,
    *,
    snapshot_fingerprint: str | None = None,
    required: bool = False,
) -> dict[str, object] | None:
    root = Path(project_root).resolve()
    if snapshot_fingerprint is None:
        try:
            archive = load_ready_provider_snapshot_archive(root)
        except FileNotFoundError:
            if required:
                raise
            return None
        snapshot_fingerprint = archive.snapshot_fingerprint
    path = resolve_research_v2_candidate_manifest_path(root, snapshot_fingerprint)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Research V2 candidate manifest 尚未建立")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("Research V2 candidate manifest 必須是 object")
    identity_keys = RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS
    # Reconstruct identity directly from persisted identity fields.  Detailed
    # coverage is supplemental; candidate_fingerprint freezes the semantic core.
    identity = {key: payload.get(key) for key in identity_keys}
    if str(payload.get("candidate_fingerprint") or "") != canonical_json_sha256(identity):
        raise ValueError("Research V2 candidate fingerprint 不一致")
    if str(payload.get("status") or "") != RESEARCH_V2_CANDIDATE_STATUS_NOT_READY:
        raise ValueError("Research V2 candidate 不得由此 layer 宣稱 READY/frozen")
    archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=str(snapshot_fingerprint))
    if str(payload.get("provider_manifest_fingerprint") or "") != archive.manifest_fingerprint:
        raise ValueError("Research V2 candidate provider manifest fingerprint drift")
    if str(payload.get("provider_as_of_date") or "") != archive.as_of_date:
        raise ValueError("Research V2 candidate provider as_of_date drift")
    if str(payload.get("dataset_assessment_fingerprint") or "") != research_v2_dataset_assessment_fingerprint():
        raise ValueError("Research V2 candidate dataset assessment contract drift")
    universe_path = resolve_research_v2_daily_universe_path(root, str(snapshot_fingerprint))
    if not universe_path.is_file():
        raise FileNotFoundError("Research V2 daily universe artifact 不存在")
    expected_universe_hash = str(payload.get("daily_universe_file_sha256") or "").strip().lower()
    if len(expected_universe_hash) != 64 or compute_file_sha256(universe_path) != expected_universe_hash:
        raise ValueError("Research V2 daily universe artifact SHA256 drift")
    if payload.get("frozen_cutoff") is not None or bool(payload.get("promotion_authorized")):
        raise ValueError("Research V2 candidate layer 不得自行 promotion/freeze")
    return payload


__all__ = [
    "ResearchV2ProviderView",
    "build_research_v2_candidate",
    "load_research_v2_candidate",
]
