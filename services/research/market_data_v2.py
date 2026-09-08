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
from core.market_data_research_pit_contract import (
    AUDIT_MODE_EXACT_CANDIDATE,
    DATE_AUDIT_STATUS_NO_ARTIFACTS,
    DATE_AUDIT_STATUS_NO_DATED_ROWS,
    DATE_AUDIT_STATUS_NOT_APPLICABLE,
    DATE_AUDIT_STATUS_READY,
    PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED,
    ResearchV2DatasetDateAudit,
    ResearchV2MechanicalCommonCompleteSummary,
    build_research_v2_pit_review_contracts,
    research_v2_pit_review_contract_fingerprint,
    research_v2_pit_review_contract_payloads,
    summarize_mechanical_common_complete_tail,
    validate_research_v2_pit_review_contracts,
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
    validate_research_v2_required_cutoff,
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

    def _iter_verified_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[pd.DataFrame]:
        name = str(dataset or "").strip()
        if name not in self._assessment_by_dataset:
            raise ValueError(f"Research V2 未登記 dataset: {name}")
        for artifact in self.dataset_artifacts(name):
            request = artifact.to_request()
            path = resolve_market_data_request_parquet_path(
                self.project_root,
                self.archive.manifest_fingerprint,
                request,
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Provider Snapshot artifact 不存在: {project_relative_display_path(path, project_root=self.project_root)}"
                )
            actual_hash = str(self._hash_fn(path) or "").strip().lower()
            if actual_hash != artifact.content_sha256:
                raise ValueError(f"Provider Snapshot artifact SHA256 drift: {request.request_id}")
            frame = self._frame_reader(path, columns)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Research V2 provider frame reader 必須回傳 pandas.DataFrame")
            yield frame

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
        yield from self._iter_verified_frames(name, columns=columns)

    def iter_dataset_contract_audit_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Read immutable rows for Research contract audit, never model consumption.

        Review-required datasets may be inspected here only to establish mechanical
        archive evidence such as date presence.  Current-vintage datasets remain
        blocked so this seam cannot become a backdoor around adjusted-price PIT.
        """
        name = str(dataset or "").strip()
        review = {row.dataset: row for row in build_research_v2_pit_review_contracts()}.get(name)
        if review is None:
            raise ValueError(f"Research V2 未登記 dataset: {name}")
        if review.review_status == PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED:
            raise RuntimeError(f"Research V2 current-vintage dataset 不允許 contract audit 讀取: {name}")
        yield from self._iter_verified_frames(name, columns=columns)



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
        CREATE TABLE dataset_date_presence (
            dataset TEXT NOT NULL,
            date TEXT NOT NULL,
            PRIMARY KEY(dataset, date)
        ) WITHOUT ROWID;
        CREATE TABLE dataset_date_audit (
            dataset TEXT PRIMARY KEY,
            audit_mode TEXT NOT NULL,
            audit_status TEXT NOT NULL,
            observed_date_count INTEGER NOT NULL,
            first_observed_date TEXT,
            latest_observed_date TEXT,
            reason TEXT NOT NULL
        );
        CREATE TABLE mechanical_common_complete (
            date TEXT PRIMARY KEY
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


def _normalize_date_values(frame: pd.DataFrame, *, dataset: str, as_of_date: str) -> tuple[str, ...]:
    if "date" not in frame.columns:
        raise ValueError(f"{dataset} 缺少 Research V2 date-presence audit 欄位: date")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return tuple(sorted({str(value) for value in dates.dropna().tolist() if str(value) <= str(as_of_date)}))


def _stream_logical_fingerprint(conn: sqlite3.Connection, query: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in conn.execute(query):
        encoded = json.dumps(list(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def _build_review_date_audits(
    conn: sqlite3.Connection,
    view: ResearchV2ProviderView,
    *,
    trading_dates: set[str],
    research_cutoff: str,
) -> tuple[tuple[ResearchV2DatasetDateAudit, ...], ResearchV2MechanicalCommonCompleteSummary]:
    contracts = build_research_v2_pit_review_contracts()
    as_of = str(research_cutoff)
    audits: list[ResearchV2DatasetDateAudit] = []
    observed_by_dataset: dict[str, set[str]] = {}

    for contract in contracts:
        if not contract.automatic_date_presence_audit:
            audit = ResearchV2DatasetDateAudit(
                dataset=contract.dataset,
                audit_mode=contract.audit_mode,
                audit_status=DATE_AUDIT_STATUS_NOT_APPLICABLE,
                observed_date_count=0,
                first_observed_date=None,
                latest_observed_date=None,
                reason=contract.reason,
            )
            audits.append(audit)
            continue

        artifacts = view.dataset_artifacts(contract.dataset)
        if not artifacts:
            audit = ResearchV2DatasetDateAudit(
                dataset=contract.dataset,
                audit_mode=contract.audit_mode,
                audit_status=DATE_AUDIT_STATUS_NO_ARTIFACTS,
                observed_date_count=0,
                first_observed_date=None,
                latest_observed_date=None,
                reason="Provider Snapshot 沒有此 dataset artifact evidence",
            )
            audits.append(audit)
            observed_by_dataset[contract.dataset] = set()
            continue

        observed: set[str] = set()
        for frame in view.iter_dataset_contract_audit_frames(contract.dataset, columns=("date",)):
            observed.update(_normalize_date_values(frame, dataset=contract.dataset, as_of_date=as_of))
        observed_by_dataset[contract.dataset] = observed
        if observed:
            conn.executemany(
                "INSERT OR IGNORE INTO dataset_date_presence(dataset, date) VALUES (?, ?)",
                ((contract.dataset, date_value) for date_value in sorted(observed)),
            )
            audit_status = DATE_AUDIT_STATUS_READY
            reason = "immutable Provider Snapshot date presence 已機械驗證；不代表 publication/revision PIT 已授權"
        else:
            audit_status = DATE_AUDIT_STATUS_NO_DATED_ROWS
            reason = "Provider Snapshot artifacts 存在但沒有可解析 dated rows"
        audit = ResearchV2DatasetDateAudit(
            dataset=contract.dataset,
            audit_mode=contract.audit_mode,
            audit_status=audit_status,
            observed_date_count=len(observed),
            first_observed_date=min(observed) if observed else None,
            latest_observed_date=max(observed) if observed else None,
            reason=reason,
        )
        audits.append(audit)

    conn.executemany(
        """
        INSERT INTO dataset_date_audit(
            dataset, audit_mode, audit_status, observed_date_count,
            first_observed_date, latest_observed_date, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                row.dataset,
                row.audit_mode,
                row.audit_status,
                int(row.observed_date_count),
                row.first_observed_date,
                row.latest_observed_date,
                row.reason,
            )
            for row in audits
        ),
    )

    exact_complete_dates = {
        str(row[0])
        for row in conn.execute("SELECT date FROM exact_coverage WHERE exact_complete = 1 ORDER BY date")
    }
    participants = tuple(
        row.dataset
        for row in contracts
        if row.automatic_date_presence_audit and row.contributes_mechanical_trading_daily_ceiling
    )
    summary = summarize_mechanical_common_complete_tail(
        trading_dates=trading_dates,
        exact_complete_dates=exact_complete_dates,
        dataset_audits=audits,
        observed_dates_by_dataset=observed_by_dataset,
        participating_datasets=participants,
    )
    if summary.audit_complete and summary.common_complete_date_count:
        common_dates = set(exact_complete_dates)
        for dataset in participants:
            common_dates.intersection_update(observed_by_dataset.get(dataset, set()))
        conn.executemany(
            "INSERT OR IGNORE INTO mechanical_common_complete(date) VALUES (?)",
            ((date_value,) for date_value in sorted(common_dates)),
        )
    return tuple(audits), summary


def _build_exact_candidate_sqlite(
    view: ResearchV2ProviderView,
    *,
    output_path: Path,
    research_cutoff: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".daily_universe.", suffix=".sqlite3.tmp", dir=str(output_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        conn = _connect_universe_db(temp_path)
        try:
            _init_universe_db(conn)
            cutoff = str(research_cutoff)
            provider_as_of = view.archive.as_of_date
            trading_rows: set[str] = set()
            for frame in view.iter_dataset_frames(RESEARCH_V2_TRADING_CALENDAR_DATASET, columns=("date",)):
                if "date" not in frame.columns:
                    raise ValueError("TaiwanStockTradingDate 缺少 date")
                dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                trading_rows.update(str(value) for value in dates.dropna().tolist() if str(value) <= cutoff)
            if not trading_rows:
                raise ValueError("Research V2 trading calendar evidence 不可為空")
            conn.executemany("INSERT OR IGNORE INTO trading_dates(date) VALUES (?)", ((value,) for value in sorted(trading_rows)))

            historical_pool = set(view.historical_instruments())
            for frame in view.iter_dataset_frames(
                RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = [row for row in _normalize_date_stock(frame, dataset=RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET) if row[0] <= cutoff and row[1] in historical_pool and row[0] in trading_rows]
                if rows:
                    conn.executemany("INSERT OR IGNORE INTO daily_universe(date, stock_id) VALUES (?, ?)", rows)

            for frame in view.iter_dataset_frames(
                RESEARCH_V2_DAILY_COVERAGE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = [row for row in _normalize_date_stock(frame, dataset=RESEARCH_V2_DAILY_COVERAGE_DATASET) if row[0] <= cutoff and row[1] in historical_pool and row[0] in trading_rows]
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
                if len(frame):
                    delisting_rows += sum(
                        1
                        for date_value, _stock_id in _normalize_date_stock(frame, dataset=RESEARCH_V2_EVENT_EVIDENCE_DATASET)
                        if date_value <= cutoff
                    )

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
                provider_as_of_date=provider_as_of,
                daily_universe_row_count=universe_count,
            )
            date_audits, mechanical_summary = _build_review_date_audits(
                conn,
                view,
                trading_dates=trading_rows,
                research_cutoff=cutoff,
            )
            metadata = {
                "provider_snapshot_fingerprint": view.archive.snapshot_fingerprint,
                "provider_manifest_fingerprint": view.archive.manifest_fingerprint,
                "provider_as_of_date": provider_as_of,
                "research_required_cutoff": cutoff,
                "historical_instrument_count": len(historical_pool),
                "daily_universe_row_count": universe_count,
                "daily_universe_fingerprint": universe_fingerprint,
                "exact_coverage_fingerprint": summary.coverage_fingerprint,
                "delisting_event_row_count": int(delisting_rows),
                "pit_review_contract_fingerprint": research_v2_pit_review_contract_fingerprint(),
                "mechanical_common_complete_fingerprint": mechanical_summary.coverage_fingerprint,
                "mechanical_common_complete_ceiling_date": mechanical_summary.common_complete_ceiling_date,
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
            "date_audits": [asdict(row) for row in date_audits],
            "mechanical_common_complete": asdict(mechanical_summary),
        }, coverage
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _build_candidate_blockers(
    contract_stats: dict[str, object],
    coverage_summary: dict[str, object],
    *,
    required_cutoff: str,
    pit_review_stats: dict[str, object],
    mechanical_summary: dict[str, object],
) -> list[dict[str, object]]:
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
        {
            "code": "RESEARCH_COMMON_COMPLETE_NOT_AUTHORIZED",
            "reason": "mechanical completeness ceiling 只代表 archive date-presence evidence；required dataset scope 與 dataset-specific PIT legality 完成前 research_common_complete_cutoff 必須維持 null。",
        },
    ]
    review_count = int(contract_stats.get("review_required_count") or 0)
    if review_count:
        blockers.append(
            {
                "code": "DATASET_SPECIFIC_PIT_REVIEW_REQUIRED",
                "dataset_count": review_count,
                "audit_mode_counts": dict(pit_review_stats.get("mode_counts") or {}),
                "pit_legality_status_counts": dict(pit_review_stats.get("pit_legality_status_counts") or {}),
                "event_information_time_anchor_ready_count": int(
                    pit_review_stats.get("event_information_time_anchor_ready_count") or 0
                ),
                "historical_publication_vintage_blocked_count": int(
                    pit_review_stats.get("historical_publication_vintage_blocked_count") or 0
                ),
                "reason": (
                    "review_required datasets 已有 canonical audit/legality matrix；event rows 最多只取得 conservative "
                    "information-time anchor，periodic/static 缺 historical publication/as-of vintage 時維持 BLOCKED，"
                    "且任何 dataset 都尚未因此取得 model-input authorization。"
                ),
            }
        )
    if not coverage_summary.get("latest_exact_complete_date"):
        blockers.append(
            {
                "code": "EXACT_CANDIDATE_COMMON_COMPLETE_NOT_ESTABLISHED",
                "reason": "exact-candidate daily universe / coverage 尚未得到任何完整交易日。",
            }
        )
    if not bool(mechanical_summary.get("audit_complete")):
        blockers.append(
            {
                "code": "MECHANICAL_DATE_AUDIT_INCOMPLETE",
                "participating_dataset_count": int(mechanical_summary.get("participating_dataset_count") or 0),
                "successful_dataset_count": int(mechanical_summary.get("successful_dataset_count") or 0),
                "reason": "至少一個 trading-daily review dataset 尚未取得可解析 date-presence evidence。",
            }
        )
    elif not mechanical_summary.get("common_complete_ceiling_date"):
        blockers.append(
            {
                "code": "MECHANICAL_COMMON_COMPLETE_TAIL_NOT_ESTABLISHED",
                "reason": "自動 date-presence audit 沒有找到共同連續 trading-date tail。",
            }
        )
    if str(coverage_summary.get("latest_exact_complete_date") or "") != str(required_cutoff):
        blockers.append(
            {
                "code": "RESEARCH_REQUIRED_CUTOFF_EXACT_COVERAGE_NOT_READY",
                "required_cutoff": str(required_cutoff),
                "latest_exact_complete_date": coverage_summary.get("latest_exact_complete_date"),
                "reason": "Research V2 exact-candidate evidence 尚未完整覆蓋固定 Research cutoff。",
            }
        )
    if str(mechanical_summary.get("common_complete_ceiling_date") or "") != str(required_cutoff):
        blockers.append(
            {
                "code": "RESEARCH_REQUIRED_CUTOFF_MECHANICAL_COVERAGE_NOT_READY",
                "required_cutoff": str(required_cutoff),
                "mechanical_common_complete_ceiling_date": mechanical_summary.get("common_complete_ceiling_date"),
                "reason": "Research V2 mechanical date-presence evidence 尚未完整覆蓋固定 Research cutoff。",
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
    required_cutoff = generation.required_cutoff
    if ACTIVE_RESEARCH_DATA_GENERATION != RESEARCH_DATA_GENERATION_V1:
        raise RuntimeError("Research V2 candidate build 不得在本輪自行切換 ACTIVE Research generation")

    contract_stats = validate_research_v2_candidate_contract()
    pit_review_contracts = build_research_v2_pit_review_contracts()
    pit_review_stats = validate_research_v2_pit_review_contracts(pit_review_contracts)
    archive = load_ready_provider_snapshot_archive(root, snapshot_fingerprint=snapshot_fingerprint)
    required_cutoff = validate_research_v2_required_cutoff(
        provider_as_of_date=archive.as_of_date,
        required_cutoff=required_cutoff,
    )
    view = ResearchV2ProviderView(
        project_root=root,
        archive=archive,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
    )
    candidate_dir = resolve_research_v2_candidate_dir(root, archive.snapshot_fingerprint)
    universe_path = resolve_research_v2_daily_universe_path(root, archive.snapshot_fingerprint)
    derived, _coverage_table = _build_exact_candidate_sqlite(
        view,
        output_path=universe_path,
        research_cutoff=required_cutoff,
    )
    del _coverage_table
    coverage_summary = dict(derived["coverage_summary"])
    mechanical_summary = dict(derived["mechanical_common_complete"])
    assessment_rows = build_research_v2_dataset_assessments()
    assessment_fingerprint = research_v2_dataset_assessment_fingerprint(assessment_rows)
    identity = build_research_v2_candidate_identity_payload(
        provider_snapshot_fingerprint=archive.snapshot_fingerprint,
        provider_manifest_fingerprint=archive.manifest_fingerprint,
        provider_as_of_date=archive.as_of_date,
        required_cutoff=required_cutoff,
        historical_instrument_count=int(derived["historical_instrument_count"]),
        daily_universe_fingerprint=str(derived["daily_universe_fingerprint"]),
        coverage_summary=ResearchV2ExactCoverageSummary(**coverage_summary),
        assessment_fingerprint=assessment_fingerprint,
        pit_review_contract_fingerprint=str(pit_review_stats["contract_fingerprint"]),
        dataset_date_audit_fingerprint=canonical_json_sha256(derived["date_audits"]),
        mechanical_common_complete_start_date=mechanical_summary.get("common_complete_tail_start"),
        mechanical_common_complete_ceiling_date=mechanical_summary.get("common_complete_ceiling_date"),
        mechanical_common_complete_tail_date_count=int(mechanical_summary.get("common_complete_tail_date_count") or 0),
        mechanical_common_complete_fingerprint=str(mechanical_summary.get("coverage_fingerprint") or ""),
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
        "pit_review_contracts": research_v2_pit_review_contract_payloads(pit_review_contracts),
        "pit_review_contract_counts": pit_review_stats,
        "dataset_date_audits": list(derived["date_audits"]),
        "mechanical_common_complete": mechanical_summary,
        "blockers": _build_candidate_blockers(
            contract_stats,
            coverage_summary,
            required_cutoff=required_cutoff,
            pit_review_stats=pit_review_stats,
            mechanical_summary=mechanical_summary,
        ),
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
    generation = get_research_data_generation(RESEARCH_DATA_GENERATION_V2)
    expected_required_cutoff = validate_research_v2_required_cutoff(
        provider_as_of_date=archive.as_of_date,
        required_cutoff=generation.required_cutoff,
    )
    if str(payload.get("required_cutoff") or "") != expected_required_cutoff:
        raise ValueError("Research V2 candidate required cutoff drift")
    expected_assessments = [asdict(row) for row in build_research_v2_dataset_assessments()]
    if str(payload.get("dataset_assessment_fingerprint") or "") != research_v2_dataset_assessment_fingerprint():
        raise ValueError("Research V2 candidate dataset assessment contract drift")
    if payload.get("dataset_assessments") != expected_assessments:
        raise ValueError("Research V2 candidate persisted dataset assessments drift")
    expected_pit_reviews = research_v2_pit_review_contract_payloads()
    if str(payload.get("pit_review_contract_fingerprint") or "") != research_v2_pit_review_contract_fingerprint():
        raise ValueError("Research V2 candidate PIT review contract drift")
    if payload.get("pit_review_contracts") != expected_pit_reviews:
        raise ValueError("Research V2 candidate persisted PIT review contracts drift")
    date_audits = payload.get("dataset_date_audits")
    if not isinstance(date_audits, list):
        raise ValueError("Research V2 candidate dataset_date_audits 必須是 list")
    if str(payload.get("dataset_date_audit_fingerprint") or "") != canonical_json_sha256(date_audits):
        raise ValueError("Research V2 candidate dataset date-audit fingerprint drift")
    mechanical = payload.get("mechanical_common_complete")
    if not isinstance(mechanical, dict):
        raise ValueError("Research V2 candidate mechanical_common_complete 必須是 object")
    if str(mechanical.get("coverage_fingerprint") or "") != str(payload.get("mechanical_common_complete_fingerprint") or ""):
        raise ValueError("Research V2 candidate mechanical common-complete fingerprint drift")
    if mechanical.get("common_complete_tail_start") != payload.get("mechanical_common_complete_start_date"):
        raise ValueError("Research V2 candidate mechanical common-complete start drift")
    if mechanical.get("common_complete_ceiling_date") != payload.get("mechanical_common_complete_ceiling_date"):
        raise ValueError("Research V2 candidate mechanical common-complete ceiling drift")
    if int(mechanical.get("common_complete_tail_date_count") or 0) != int(payload.get("mechanical_common_complete_tail_date_count") or 0):
        raise ValueError("Research V2 candidate mechanical common-complete tail count drift")
    universe_path = resolve_research_v2_daily_universe_path(root, str(snapshot_fingerprint))
    if not universe_path.is_file():
        raise FileNotFoundError("Research V2 daily universe artifact 不存在")
    expected_universe_hash = str(payload.get("daily_universe_file_sha256") or "").strip().lower()
    if len(expected_universe_hash) != 64 or compute_file_sha256(universe_path) != expected_universe_hash:
        raise ValueError("Research V2 daily universe artifact SHA256 drift")
    if payload.get("research_common_complete_cutoff") is not None:
        raise ValueError("Research V2 candidate layer 不得自行宣告 research_common_complete_cutoff")
    if payload.get("frozen_cutoff") is not None or bool(payload.get("promotion_authorized")):
        raise ValueError("Research V2 candidate layer 不得自行 promotion/freeze")
    return payload


__all__ = [
    "ResearchV2ProviderView",
    "build_research_v2_candidate",
    "load_research_v2_candidate",
]
