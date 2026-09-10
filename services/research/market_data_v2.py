"""Research Market Data V2 candidate builder and immutable provider read view.

The builder is local-only: it consumes one READY neutral Provider Snapshot and
never reads the Trading V2 overlay or calls FinMind.  It materializes a derived
PIT-universe/coverage index plus a NOT_READY candidate manifest.  Promotion to
active/frozen Research truth is intentionally outside this module.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Callable, Iterator

import pandas as pd

from config.market_data import RESEARCH_DATA_GENERATION_V1, RESEARCH_DATA_GENERATION_V2
from core.console_report import project_relative_display_path
from core.file_integrity import atomic_replace_with_retry, atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.market_data_contract import (
    RESEARCH_STATUS_AUTHORIZED_NOT_READY,
    get_research_data_generation,
)
from core.market_data_research_promotion import get_effective_research_data_generation
from core.market_data_instrument_universe import (
    build_historical_market_state_guard,
    historical_market_state_guard_fingerprint,
)
from core.market_data_pit_universe import make_daily_pit_market_universe_guard
from core.market_data_adjusted_price_revision_proof import (
    ADJUSTED_PRICE_REVISION_STATUS_READY,
)
from core.market_data_adjusted_price_invariance import (
    PROVIDER_PRICE_FIELDS,
    PROVIDER_VOLUME_FIELD,
    adjusted_price_representation_contract_fingerprint,
    adjusted_price_representation_contract_payload,
    get_adjusted_price_representation_contract,
    validate_adjusted_price_representation_contract,
)
from core.market_data_research_scope import (
    RESEARCH_V2_ADJUSTED_PRICE_DATASET,
    RESEARCH_V2_COMMON_COMPLETE_DATASETS,
    RESEARCH_V2_RAW_VOLUME_DATASET,
    build_research_v2_dataset_scope_contracts,
    research_v2_dataset_scope_contract_payloads,
    validate_research_v2_dataset_scope_contracts,
)
from core.market_data_research_freeze import (
    RESEARCH_V2_FREEZE_CANDIDATE_IDENTITY_FIELDS,
    RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY,
    ResearchV2RequiredCommonCompleteSummary,
    build_research_v2_freeze_candidate_identity_payload,
    build_research_v2_required_common_complete,
    required_common_complete_summary_payload,
)
from core.market_data_research_storage_contract import (
    resolve_research_v2_candidate_dir,
    resolve_research_v2_candidate_manifest_path,
    resolve_research_v2_daily_universe_path,
    resolve_research_v2_freeze_candidate_dir,
    resolve_research_v2_freeze_candidate_manifest_path,
    resolve_research_v2_frozen_daily_universe_path,
    resolve_research_v2_frozen_source_candidate_manifest_path,
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
    RESEARCH_V2_CANDIDATE_SCHEMA_VERSION,
    RESEARCH_V2_CANDIDATE_STATUS_NOT_READY,
    RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS,
    RESEARCH_V2_DAILY_COVERAGE_DATASET,
    RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
    RESEARCH_V2_EVENT_EVIDENCE_DATASET,
    RESEARCH_V2_MARKET_STATE_GUARD_DATASET,
    RESEARCH_V2_TRADING_CALENDAR_DATASET,
    build_research_v2_candidate_identity_payload,
    build_research_v2_dataset_assessments,
    research_v2_dataset_assessment_fingerprint,
    ResearchV2ExactCoverageSummary,
    summarize_exact_candidate_coverage_table,
    validate_research_v2_candidate_contract,
    validate_research_v2_required_cutoff,
)
from core.market_data_storage_contract import resolve_market_data_daily_pit_universe_path
from services.market_data.provider_snapshot_repository import (
    ReadyProviderSnapshotArchive,
    load_ready_provider_snapshot_archive,
)
from services.market_data.provider_snapshot_view import ProviderSnapshotView
from services.market_data.daily_pit_universe import (
    build_market_data_v2_daily_pit_universe,
    load_market_data_v2_daily_pit_universe,
)
from services.research.adjusted_price_revision_proof import (
    build_adjusted_price_revision_proof,
    load_adjusted_price_revision_proof,
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
        self._snapshot_view = ProviderSnapshotView(
            project_root=self.project_root,
            archive=archive,
            frame_reader=frame_reader,
            hash_fn=hash_fn,
        )
        self._assessment_by_dataset = {row.dataset: row for row in build_research_v2_dataset_assessments()}
        self._scope_by_dataset = {row.dataset: row for row in build_research_v2_dataset_scope_contracts()}

    def dataset_artifacts(self, dataset: str):
        name = str(dataset or "").strip()
        if name not in self._assessment_by_dataset:
            raise ValueError(f"Research V2 未登記 dataset: {name}")
        return self._snapshot_view.dataset_artifacts(name)

    def historical_instruments(self) -> tuple[str, ...]:
        return self._snapshot_view.historical_instruments(
            source_dataset=RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET
        )

    def historical_market_state_guard(
        self, *, historical_instruments: tuple[str, ...]
    ) -> tuple[dict[str, str], str]:
        """Return the conservative date-specific market-state exclusion guard.

        ``TaiwanStockInfo`` is read here only for historical universe eligibility;
        it is not exposed as a model-input scope seam.  The resulting transition
        projection is pinned into the Research candidate scientific identity.
        """

        frames = list(
            self._iter_verified_frames(
                RESEARCH_V2_MARKET_STATE_GUARD_DATASET,
                columns=("date", "stock_id", "type", "industry_category"),
            )
        )
        if not frames:
            raise ValueError("Research V2 缺 TaiwanStockInfo market-state guard evidence")
        stock_info = pd.concat(frames, ignore_index=True)
        guard = build_historical_market_state_guard(
            stock_info, historical_instruments=historical_instruments
        )
        return guard, historical_market_state_guard_fingerprint(guard)

    def _iter_verified_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...] | None = None,
        data_id: str | None = None,
    ) -> Iterator[pd.DataFrame]:
        name = str(dataset or "").strip()
        if name not in self._assessment_by_dataset:
            raise ValueError(f"Research V2 未登記 dataset: {name}")
        yield from self._snapshot_view.iter_verified_frames(
            name,
            columns=columns,
            data_id=data_id,
        )

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


    def iter_dataset_scope_frames(
        self,
        dataset: str,
        *,
        columns: tuple[str, ...],
        data_id: str | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Read only fields explicitly authorized by the Round-14 foundation scope.

        This is the sole seam through which Round-15 field-aware completeness may
        inspect required datasets, including adjusted-price OHLC operands.  It does
        not grant direct model consumption and rejects optional/unlisted fields.
        """
        name = str(dataset or "").strip()
        scope = self._scope_by_dataset.get(name)
        if scope is None or not scope.required_for_generation:
            raise RuntimeError(f"Research V2 dataset 未被 foundation required scope 授權: {name}")
        allowed = set(scope.evidence_fields)
        for rule in scope.field_authorizations:
            allowed.update(rule.fields)
        requested = tuple(str(column) for column in columns)
        disallowed = sorted(set(requested).difference(allowed))
        if disallowed:
            raise RuntimeError(
                f"Research V2 dataset scope reader 欄位未授權: {name} -> {disallowed}"
            )
        yield from self._iter_verified_frames(name, columns=requested, data_id=data_id)



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
        CREATE TABLE raw_volume_field_evidence (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            volume_value TEXT NOT NULL,
            PRIMARY KEY(date, stock_id)
        ) WITHOUT ROWID;
        CREATE TABLE adjusted_price_operand_evidence (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            open_value TEXT NOT NULL,
            high_value TEXT NOT NULL,
            low_value TEXT NOT NULL,
            close_value TEXT NOT NULL,
            PRIMARY KEY(date, stock_id)
        ) WITHOUT ROWID;
        CREATE TABLE delisting_event_evidence (
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
        CREATE TABLE required_scope_coverage (
            date TEXT PRIMARY KEY,
            universe_count INTEGER NOT NULL,
            raw_volume_valid_count INTEGER NOT NULL,
            price_limit_count INTEGER NOT NULL,
            adjusted_price_operand_count INTEGER NOT NULL,
            missing_raw_volume_count INTEGER NOT NULL,
            missing_price_limit_count INTEGER NOT NULL,
            missing_adjusted_price_operand_count INTEGER NOT NULL,
            required_complete INTEGER NOT NULL CHECK(required_complete IN (0, 1))
        );
        CREATE TABLE required_common_complete (
            date TEXT PRIMARY KEY
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


def _normalize_required_field_values(
    frame: pd.DataFrame,
    *,
    dataset: str,
    value_fields: tuple[str, ...],
) -> list[tuple[str, ...]]:
    required = {"date", "stock_id", *value_fields}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset} 缺少 Research V2 required field-completeness 欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "stock_id", *value_fields]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["stock_id"] = local["stock_id"].astype(str).str.strip()
    valid = local["date"].notna() & local["stock_id"].ne("")
    for field in value_fields:
        numeric = pd.to_numeric(local[field], errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
        local[field] = numeric
        valid &= numeric.notna()
    local = local.loc[valid, ["date", "stock_id", *value_fields]]
    rows: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in local.itertuples(index=False, name=None):
        key = (str(row[0]), str(row[1]))
        values = tuple(format(float(value), ".17g") for value in row[2:])
        normalized = (*key, *values)
        prior = rows.get(key)
        if prior is not None and prior != normalized:
            raise ValueError(f"{dataset} required field evidence duplicate conflict: {key}")
        rows[key] = normalized
    return list(rows.values())



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
    research_scope_contract_fingerprint: str,
    neutral_daily_pit_universe_path: Path,
    neutral_daily_pit_universe_manifest: dict[str, object],
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
            if str(neutral_daily_pit_universe_manifest.get("provider_snapshot_fingerprint") or "") != view.archive.snapshot_fingerprint:
                raise ValueError("Research V2 neutral daily PIT universe Provider Snapshot identity drift")
            if str(neutral_daily_pit_universe_manifest.get("provider_as_of_date") or "") != provider_as_of:
                raise ValueError("Research V2 neutral daily PIT universe provider as-of drift")

            conn.execute("ATTACH DATABASE ? AS neutral_pit", (str(neutral_daily_pit_universe_path),))
            conn.execute(
                "INSERT OR IGNORE INTO trading_dates(date) "
                "SELECT date FROM neutral_pit.trading_dates WHERE date <= ? ORDER BY date",
                (cutoff,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO daily_universe(date, stock_id) "
                "SELECT date, stock_id FROM neutral_pit.daily_universe WHERE date <= ? ORDER BY date, stock_id",
                (cutoff,),
            )
            trading_rows = {str(row[0]) for row in conn.execute("SELECT date FROM trading_dates ORDER BY date")}
            if not trading_rows:
                raise ValueError("Research V2 neutral daily PIT trading calendar projection 不可為空")
            if int(conn.execute("SELECT COUNT(*) FROM daily_universe").fetchone()[0]) <= 0:
                raise ValueError("Research V2 neutral daily PIT universe cutoff projection 不可為空")

            historical_instruments = view.historical_instruments()
            transition_excluded_through, market_state_guard_fingerprint = view.historical_market_state_guard(
                historical_instruments=historical_instruments
            )
            if market_state_guard_fingerprint != str(
                neutral_daily_pit_universe_manifest.get("historical_market_state_guard_fingerprint") or ""
            ):
                raise ValueError("Research V2 neutral daily PIT universe market-state guard drift")
            eligibility_guard = make_daily_pit_market_universe_guard(
                historical_instruments=historical_instruments,
                transition_excluded_through=transition_excluded_through,
                trading_dates=trading_rows,
                provider_as_of_date=cutoff,
            )

            def _pit_member(date_value: str, stock_id: str) -> bool:
                return eligibility_guard.allows(stock_id=stock_id, date_value=date_value)

            for frame in view.iter_dataset_scope_frames(
                RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
                columns=("date", "stock_id", PROVIDER_VOLUME_FIELD),
            ):
                volume_rows = [
                    row
                    for row in _normalize_required_field_values(
                        frame,
                        dataset=RESEARCH_V2_RAW_VOLUME_DATASET,
                        value_fields=(PROVIDER_VOLUME_FIELD,),
                    )
                    if row[0] <= cutoff and _pit_member(row[0], row[1])
                ]
                if volume_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO raw_volume_field_evidence(date, stock_id, volume_value) VALUES (?, ?, ?)",
                        volume_rows,
                    )

            for frame in view.iter_dataset_scope_frames(
                RESEARCH_V2_DAILY_COVERAGE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = [
                    row
                    for row in _normalize_date_stock(frame, dataset=RESEARCH_V2_DAILY_COVERAGE_DATASET)
                    if row[0] <= cutoff and _pit_member(row[0], row[1])
                ]
                if rows:
                    conn.executemany("INSERT OR IGNORE INTO price_limit_evidence(date, stock_id) VALUES (?, ?)", rows)

            for frame in view.iter_dataset_scope_frames(
                RESEARCH_V2_ADJUSTED_PRICE_DATASET,
                columns=("date", "stock_id", *PROVIDER_PRICE_FIELDS),
            ):
                adjusted_rows = [
                    row
                    for row in _normalize_required_field_values(
                        frame,
                        dataset=RESEARCH_V2_ADJUSTED_PRICE_DATASET,
                        value_fields=tuple(PROVIDER_PRICE_FIELDS),
                    )
                    if row[0] <= cutoff and _pit_member(row[0], row[1])
                ]
                if adjusted_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO adjusted_price_operand_evidence(date, stock_id, open_value, high_value, low_value, close_value) VALUES (?, ?, ?, ?, ?, ?)",
                        adjusted_rows,
                    )

            # Delisting remains required event evidence.  Reading it through the
            # scope-authorized seam verifies the immutable artifact chain, while
            # legal no-row means it is never a daily completeness denominator.
            delisting_rows = 0
            for frame in view.iter_dataset_scope_frames(
                RESEARCH_V2_EVENT_EVIDENCE_DATASET,
                columns=("date", "stock_id"),
            ):
                rows = [
                    row
                    for row in _normalize_date_stock(frame, dataset=RESEARCH_V2_EVENT_EVIDENCE_DATASET)
                    if row[0] <= cutoff
                ]
                delisting_rows += len(rows)
                if rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO delisting_event_evidence(date, stock_id) VALUES (?, ?)",
                        rows,
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
            conn.execute(
                """
                INSERT INTO required_scope_coverage(
                    date, universe_count, raw_volume_valid_count, price_limit_count,
                    adjusted_price_operand_count, missing_raw_volume_count,
                    missing_price_limit_count, missing_adjusted_price_operand_count, required_complete
                )
                SELECT
                    t.date,
                    (SELECT COUNT(*) FROM daily_universe u WHERE u.date = t.date),
                    (SELECT COUNT(*) FROM raw_volume_field_evidence v WHERE v.date = t.date),
                    (SELECT COUNT(*) FROM price_limit_evidence p WHERE p.date = t.date),
                    (SELECT COUNT(*) FROM adjusted_price_operand_evidence a WHERE a.date = t.date),
                    (SELECT COUNT(*) FROM daily_universe u
                        LEFT JOIN raw_volume_field_evidence v
                          ON v.date = u.date AND v.stock_id = u.stock_id
                      WHERE u.date = t.date AND v.stock_id IS NULL),
                    (SELECT COUNT(*) FROM daily_universe u
                        LEFT JOIN price_limit_evidence p
                          ON p.date = u.date AND p.stock_id = u.stock_id
                      WHERE u.date = t.date AND p.stock_id IS NULL),
                    (SELECT COUNT(*) FROM daily_universe u
                        LEFT JOIN adjusted_price_operand_evidence a
                          ON a.date = u.date AND a.stock_id = u.stock_id
                      WHERE u.date = t.date AND a.stock_id IS NULL),
                    CASE WHEN
                        (SELECT COUNT(*) FROM daily_universe u WHERE u.date = t.date) > 0
                        AND (SELECT COUNT(*) FROM daily_universe u
                              LEFT JOIN raw_volume_field_evidence v
                                ON v.date = u.date AND v.stock_id = u.stock_id
                            WHERE u.date = t.date AND v.stock_id IS NULL) = 0
                        AND (SELECT COUNT(*) FROM daily_universe u
                              LEFT JOIN price_limit_evidence p
                                ON p.date = u.date AND p.stock_id = u.stock_id
                            WHERE u.date = t.date AND p.stock_id IS NULL) = 0
                        AND (SELECT COUNT(*) FROM daily_universe u
                              LEFT JOIN adjusted_price_operand_evidence a
                                ON a.date = u.date AND a.stock_id = u.stock_id
                            WHERE u.date = t.date AND a.stock_id IS NULL) = 0
                    THEN 1 ELSE 0 END
                FROM trading_dates t
                ORDER BY t.date
                """
            )

            universe_count, universe_fingerprint = _stream_logical_fingerprint(
                conn,
                "SELECT date, stock_id FROM daily_universe ORDER BY date, stock_id",
            )
            _trading_count, trading_source_fingerprint = _stream_logical_fingerprint(
                conn, "SELECT date FROM trading_dates ORDER BY date"
            )
            _volume_count, raw_volume_source_fingerprint = _stream_logical_fingerprint(
                conn, "SELECT date, stock_id, volume_value FROM raw_volume_field_evidence ORDER BY date, stock_id"
            )
            _limit_count, price_limit_source_fingerprint = _stream_logical_fingerprint(
                conn, "SELECT date, stock_id FROM price_limit_evidence ORDER BY date, stock_id"
            )
            _adjusted_count, adjusted_price_source_fingerprint = _stream_logical_fingerprint(
                conn, "SELECT date, stock_id, open_value, high_value, low_value, close_value FROM adjusted_price_operand_evidence ORDER BY date, stock_id"
            )
            _delisting_count, delisting_source_fingerprint = _stream_logical_fingerprint(
                conn, "SELECT date, stock_id FROM delisting_event_evidence ORDER BY date, stock_id"
            )
            required_source_projection = {
                "schema_version": 1,
                "contract_id": "research_v2_required_source_projection_v1",
                "required_cutoff": cutoff,
                "trading_calendar_fingerprint": trading_source_fingerprint,
                "daily_universe_fingerprint": universe_fingerprint,
                "historical_market_state_guard_fingerprint": market_state_guard_fingerprint,
                "raw_volume_source_fingerprint": raw_volume_source_fingerprint,
                "price_limit_presence_fingerprint": price_limit_source_fingerprint,
                "adjusted_price_source_fingerprint": adjusted_price_source_fingerprint,
                "delisting_event_fingerprint": delisting_source_fingerprint,
            }
            required_source_projection_fingerprint = canonical_json_sha256(required_source_projection)
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
            complete_dates_by_dataset = {
                RESEARCH_V2_TRADING_CALENDAR_DATASET: set(trading_rows),
                RESEARCH_V2_RAW_VOLUME_DATASET: {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT date FROM required_scope_coverage WHERE universe_count > 0 AND missing_raw_volume_count = 0 ORDER BY date"
                    )
                },
                RESEARCH_V2_DAILY_COVERAGE_DATASET: {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT date FROM required_scope_coverage WHERE universe_count > 0 AND missing_price_limit_count = 0 ORDER BY date"
                    )
                },
                RESEARCH_V2_ADJUSTED_PRICE_DATASET: {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT date FROM required_scope_coverage WHERE universe_count > 0 AND missing_adjusted_price_operand_count = 0 ORDER BY date"
                    )
                },
            }
            required_common_dates, required_common_summary = build_research_v2_required_common_complete(
                trading_dates=trading_rows,
                complete_dates_by_dataset=complete_dates_by_dataset,
                required_cutoff=cutoff,
                research_scope_contract_fingerprint=str(research_scope_contract_fingerprint),
                participating_datasets=RESEARCH_V2_COMMON_COMPLETE_DATASETS,
            )
            if required_common_dates:
                conn.executemany(
                    "INSERT OR IGNORE INTO required_common_complete(date) VALUES (?)",
                    ((date_value,) for date_value in required_common_dates),
                )
            required_scope_coverage_row_count, required_scope_coverage_fingerprint = _stream_logical_fingerprint(
                conn,
                "SELECT date, universe_count, raw_volume_valid_count, price_limit_count, adjusted_price_operand_count, "
                "missing_raw_volume_count, missing_price_limit_count, missing_adjusted_price_operand_count, required_complete "
                "FROM required_scope_coverage ORDER BY date",
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
                "historical_market_state_guard_dataset": RESEARCH_V2_MARKET_STATE_GUARD_DATASET,
                "historical_market_state_guard_fingerprint": market_state_guard_fingerprint,
                "historical_market_state_transition_count": len(transition_excluded_through),
                "required_source_projection": required_source_projection,
                "required_source_projection_fingerprint": required_source_projection_fingerprint,
                "daily_universe_row_count": universe_count,
                "daily_universe_fingerprint": universe_fingerprint,
                "neutral_daily_pit_universe_identity_fingerprint": str(
                    neutral_daily_pit_universe_manifest.get("daily_pit_universe_identity_fingerprint") or ""
                ),
                "neutral_daily_pit_universe_file_sha256": str(
                    neutral_daily_pit_universe_manifest.get("daily_universe_file_sha256") or ""
                ),
                "exact_coverage_fingerprint": summary.coverage_fingerprint,
                "delisting_event_row_count": int(delisting_rows),
                "pit_review_contract_fingerprint": research_v2_pit_review_contract_fingerprint(),
                "research_scope_contract_fingerprint": str(research_scope_contract_fingerprint),
                "required_scope_coverage_row_count": int(required_scope_coverage_row_count),
                "required_scope_coverage_fingerprint": str(required_scope_coverage_fingerprint),
                "required_common_complete_fingerprint": required_common_summary.coverage_fingerprint,
                "research_common_complete_cutoff": required_common_summary.common_complete_cutoff,
                "required_common_complete_start_date": required_common_summary.common_complete_tail_start,
                "required_common_complete_tail_date_count": int(required_common_summary.common_complete_tail_date_count),
                "mechanical_common_complete_fingerprint": mechanical_summary.coverage_fingerprint,
                "mechanical_common_complete_ceiling_date": mechanical_summary.common_complete_ceiling_date,
            }
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ((key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in metadata.items()),
            )
            conn.execute("DROP TABLE price_limit_evidence")
            conn.execute("DROP TABLE raw_volume_field_evidence")
            conn.execute("DROP TABLE adjusted_price_operand_evidence")
            conn.execute("DROP TABLE delisting_event_evidence")
            conn.commit()
        finally:
            conn.close()
        atomic_replace_with_retry(temp_path, output_path)
        return {
            **metadata,
            "coverage_summary": asdict(summary),
            "required_common_complete": required_common_complete_summary_payload(required_common_summary),
            "date_audits": [asdict(row) for row in date_audits],
            "mechanical_common_complete": asdict(mechanical_summary),
        }, coverage
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _build_candidate_blockers(
    coverage_summary: dict[str, object],
    *,
    required_cutoff: str,
    scope_stats: dict[str, object],
    required_common_complete_summary: dict[str, object] | None = None,
    adjusted_price_revision_proof: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    # Archive-wide diagnostic review is intentionally excluded from foundation
    # readiness.  Round 15 gates only the Round-14 authorized required scope.
    blockers: list[dict[str, object]] = []
    proof_status = str((adjusted_price_revision_proof or {}).get("status") or "")
    if proof_status != ADJUSTED_PRICE_REVISION_STATUS_READY:
        blockers.append(
            {
                "code": "ADJUSTED_PRICE_POST_CUTOFF_REVISION_PROOF_NOT_READY",
                "proof_status": proof_status or None,
                "proof_fingerprint": (adjusted_price_revision_proof or {}).get("proof_fingerprint"),
                "reason": (
                    "FinMind 2026-09-01 TaiwanStockPriceAdj full-history rebuild is outside the Round-13 "
                    "corporate-action scalar proof; current-vintage OHLC cannot freeze until frozen-V1/current "
                    "required stock-days prove one positive scalar per instrument."
                ),
            }
        )
    if required_common_complete_summary is None:
        blockers.append(
            {
                "code": "RESEARCH_COMMON_COMPLETE_NOT_AUTHORIZED",
                "required_dataset_scope": list(scope_stats.get("required_dataset_scope") or []),
                "common_complete_dataset_scope": list(scope_stats.get("common_complete_dataset_scope") or []),
                "reason": "required-scope true common-complete evidence 尚未由 Round-15 contract 提供。",
            }
        )
    elif str(required_common_complete_summary.get("common_complete_cutoff") or "") != str(required_cutoff):
        blockers.append(
            {
                "code": "RESEARCH_REQUIRED_SCOPE_COMMON_COMPLETE_NOT_READY",
                "required_cutoff": str(required_cutoff),
                "common_complete_cutoff": required_common_complete_summary.get("common_complete_cutoff"),
                "common_complete_tail_start": required_common_complete_summary.get("common_complete_tail_start"),
                "common_complete_tail_date_count": int(
                    required_common_complete_summary.get("common_complete_tail_date_count") or 0
                ),
                "required_common_complete_fingerprint": required_common_complete_summary.get("coverage_fingerprint"),
                "reason": (
                    "Round-14 required scope 的 field-aware common-complete tail 尚未覆蓋固定 Research cutoff；"
                    "不得建立 immutable freeze candidate。"
                ),
            }
        )
    if not bool(scope_stats.get("required_scope_authorization_complete")):
        blockers.append(
            {
                "code": "REQUIRED_SCOPE_AUTHORIZATION_INCOMPLETE",
                "reason": "至少一個 Research V2 required dataset/field/representation 尚未取得明確 scientific authorization。",
            }
        )
    if not coverage_summary.get("latest_exact_complete_date"):
        blockers.append(
            {
                "code": "EXACT_CANDIDATE_COMMON_COMPLETE_NOT_ESTABLISHED",
                "reason": "required exact-candidate daily universe / coverage 尚未得到任何完整交易日。",
            }
        )
    if str(coverage_summary.get("latest_exact_complete_date") or "") != str(required_cutoff):
        blockers.append(
            {
                "code": "RESEARCH_REQUIRED_CUTOFF_EXACT_COVERAGE_NOT_READY",
                "required_cutoff": str(required_cutoff),
                "latest_exact_complete_date": coverage_summary.get("latest_exact_complete_date"),
                "reason": "Research V2 required exact-candidate evidence 尚未完整覆蓋固定 Research cutoff。",
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
    active_generation = get_effective_research_data_generation(root)
    if active_generation.generation_id != RESEARCH_DATA_GENERATION_V1:
        raise RuntimeError("Research V2 已 promotion 為 active generation；不得在 active V2 上重建 candidate")

    contract_stats = validate_research_v2_candidate_contract()
    pit_review_contracts = build_research_v2_pit_review_contracts()
    pit_review_stats = validate_research_v2_pit_review_contracts(pit_review_contracts)
    adjusted_price_representation = get_adjusted_price_representation_contract()
    adjusted_price_representation_stats = validate_adjusted_price_representation_contract(adjusted_price_representation)
    research_scope_contracts = build_research_v2_dataset_scope_contracts()
    research_scope_stats = validate_research_v2_dataset_scope_contracts(research_scope_contracts)
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
    neutral_daily_pit = build_market_data_v2_daily_pit_universe(
        root,
        snapshot_fingerprint=archive.snapshot_fingerprint,
        frame_reader=frame_reader,
        hash_fn=hash_fn,
        now=now,
    )
    neutral_daily_pit_path = resolve_market_data_daily_pit_universe_path(
        root, archive.snapshot_fingerprint
    )
    candidate_dir = resolve_research_v2_candidate_dir(root, archive.snapshot_fingerprint)
    universe_path = resolve_research_v2_daily_universe_path(root, archive.snapshot_fingerprint)
    derived, _coverage_table = _build_exact_candidate_sqlite(
        view,
        output_path=universe_path,
        research_cutoff=required_cutoff,
        research_scope_contract_fingerprint=str(research_scope_stats["contract_fingerprint"]),
        neutral_daily_pit_universe_path=neutral_daily_pit_path,
        neutral_daily_pit_universe_manifest=neutral_daily_pit,
    )
    del _coverage_table
    coverage_summary = dict(derived["coverage_summary"])
    adjusted_price_revision_proof_build = build_adjusted_price_revision_proof(
        root,
        provider_view=view,
        daily_universe_path=universe_path,
        required_cutoff=required_cutoff,
        now=now,
    )
    adjusted_price_revision_proof = load_adjusted_price_revision_proof(
        root,
        proof_fingerprint=str(adjusted_price_revision_proof_build["proof_fingerprint"]),
        required=True,
    )
    assert adjusted_price_revision_proof is not None
    required_common_complete_payload = dict(derived["required_common_complete"])
    required_common_complete_summary = ResearchV2RequiredCommonCompleteSummary(
        **{
            **required_common_complete_payload,
            "participating_datasets": tuple(required_common_complete_payload.get("participating_datasets") or ()),
        }
    )
    mechanical_summary = dict(derived["mechanical_common_complete"])
    assessment_rows = build_research_v2_dataset_assessments()
    assessment_fingerprint = research_v2_dataset_assessment_fingerprint(assessment_rows)
    identity = build_research_v2_candidate_identity_payload(
        required_cutoff=required_cutoff,
        required_source_projection_fingerprint=str(derived["required_source_projection_fingerprint"]),
        historical_market_state_guard_fingerprint=str(derived["historical_market_state_guard_fingerprint"]),
        daily_universe_fingerprint=str(derived["daily_universe_fingerprint"]),
        coverage_summary=ResearchV2ExactCoverageSummary(**coverage_summary),
        adjusted_price_representation_contract_fingerprint=str(
            adjusted_price_representation_stats["contract_fingerprint"]
        ),
        adjusted_price_revision_proof_fingerprint=str(adjusted_price_revision_proof["proof_fingerprint"]),
        adjusted_price_revision_proof_status=str(adjusted_price_revision_proof["status"]),
        research_scope_contract_fingerprint=str(research_scope_stats["contract_fingerprint"]),
        required_dataset_scope=research_scope_stats["required_dataset_scope"],
        required_common_complete_summary=required_common_complete_summary,
    )
    candidate_fingerprint = canonical_json_sha256(identity)
    built_at = (now or datetime.now().astimezone()).astimezone().isoformat()
    payload = {
        **identity,
        "candidate_fingerprint": candidate_fingerprint,
        "provider_snapshot_fingerprint": archive.snapshot_fingerprint,
        "provider_manifest_fingerprint": archive.manifest_fingerprint,
        "provider_as_of_date": archive.as_of_date,
        "historical_instrument_count": int(derived["historical_instrument_count"]),
        "required_source_projection": dict(derived["required_source_projection"]),
        "adjusted_price_revision_proof": adjusted_price_revision_proof,
        "built_at": built_at,
        "provider_snapshot_path": project_relative_display_path(archive.path, project_root=root),
        "daily_universe_path": project_relative_display_path(universe_path, project_root=root),
        "daily_universe_file_sha256": compute_file_sha256(universe_path),
        "neutral_daily_pit_universe_identity_fingerprint": str(
            neutral_daily_pit.get("daily_pit_universe_identity_fingerprint") or ""
        ),
        "neutral_daily_pit_universe_file_sha256": str(neutral_daily_pit.get("daily_universe_file_sha256") or ""),
        "neutral_daily_pit_universe_path": project_relative_display_path(
            neutral_daily_pit_path, project_root=root
        ),
        "daily_universe_row_count": int(derived["daily_universe_row_count"]),
        "historical_market_state_transition_count": int(derived["historical_market_state_transition_count"]),
        "daily_universe_date_count": int(coverage_summary.get("daily_universe_date_count") or 0),
        "exact_coverage": coverage_summary,
        "dataset_assessment_fingerprint": assessment_fingerprint,
        "dataset_assessments": [asdict(row) for row in assessment_rows],
        "dataset_assessment_counts": contract_stats,
        "pit_review_contract_fingerprint": str(pit_review_stats["contract_fingerprint"]),
        "pit_review_contracts": research_v2_pit_review_contract_payloads(pit_review_contracts),
        "pit_review_contract_counts": pit_review_stats,
        "adjusted_price_representation_contract": adjusted_price_representation_contract_payload(
            adjusted_price_representation
        ),
        "research_scope_contracts": research_v2_dataset_scope_contract_payloads(research_scope_contracts),
        "research_scope_contract_counts": research_scope_stats,
        "required_scope_coverage_row_count": int(derived["required_scope_coverage_row_count"]),
        "required_scope_coverage_fingerprint": str(derived["required_scope_coverage_fingerprint"]),
        "required_common_complete": required_common_complete_payload,
        "dataset_date_audit_fingerprint": canonical_json_sha256(derived["date_audits"]),
        "dataset_date_audits": list(derived["date_audits"]),
        "mechanical_common_complete_start_date": mechanical_summary.get("common_complete_tail_start"),
        "mechanical_common_complete_ceiling_date": mechanical_summary.get("common_complete_ceiling_date"),
        "mechanical_common_complete_tail_date_count": int(mechanical_summary.get("common_complete_tail_date_count") or 0),
        "mechanical_common_complete_fingerprint": str(mechanical_summary.get("coverage_fingerprint") or ""),
        "mechanical_common_complete": mechanical_summary,
        "blockers": _build_candidate_blockers(
            coverage_summary,
            required_cutoff=required_cutoff,
            scope_stats=research_scope_stats,
            required_common_complete_summary=required_common_complete_payload,
            adjusted_price_revision_proof=adjusted_price_revision_proof,
        ),
        "promotion_authorized": False,
        "active_research_generation": active_generation.generation_id,
    }
    archive_diagnostics = {
        "dataset_assessments": payload["dataset_assessments"],
        "dataset_assessment_counts": payload["dataset_assessment_counts"],
        "pit_review_contracts": payload["pit_review_contracts"],
        "pit_review_contract_counts": payload["pit_review_contract_counts"],
        "research_scope_contracts": payload["research_scope_contracts"],
        "research_scope_contract_counts": payload["research_scope_contract_counts"],
    }
    payload["archive_diagnostics_fingerprint"] = canonical_json_sha256(archive_diagnostics)
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
    if int(payload.get("schema_version") or 0) != RESEARCH_V2_CANDIDATE_SCHEMA_VERSION:
        raise ValueError("Research V2 candidate schema version drift；請重建 candidate")
    guard_fp = str(payload.get("historical_market_state_guard_fingerprint") or "").strip().lower()
    if str(payload.get("historical_market_state_guard_dataset") or "") != RESEARCH_V2_MARKET_STATE_GUARD_DATASET or len(guard_fp) != 64:
        raise ValueError("Research V2 candidate historical market-state guard identity 不完整")
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
    neutral_identity = str(payload.get("neutral_daily_pit_universe_identity_fingerprint") or "").strip().lower()
    if neutral_identity:
        neutral = load_market_data_v2_daily_pit_universe(
            root,
            provider_snapshot_fingerprint=archive.snapshot_fingerprint,
            required=True,
        )
        assert neutral is not None
        if neutral_identity != str(neutral.get("daily_pit_universe_identity_fingerprint") or ""):
            raise ValueError("Research V2 candidate neutral daily PIT universe identity drift")
        if str(payload.get("neutral_daily_pit_universe_file_sha256") or "") != str(
            neutral.get("daily_universe_file_sha256") or ""
        ):
            raise ValueError("Research V2 candidate neutral daily PIT universe physical provenance drift")
    projection = payload.get("required_source_projection")
    if not isinstance(projection, dict):
        raise ValueError("Research V2 candidate required source projection 缺失")
    if canonical_json_sha256(projection) != str(payload.get("required_source_projection_fingerprint") or ""):
        raise ValueError("Research V2 candidate required source projection fingerprint drift")
    proof_fp = str(payload.get("adjusted_price_revision_proof_fingerprint") or "").strip().lower()
    if len(proof_fp) != 64:
        raise ValueError("Research V2 candidate adjusted-price revision proof fingerprint invalid")
    proof = load_adjusted_price_revision_proof(root, proof_fingerprint=proof_fp, required=True)
    if str(proof.get("status") or "") != str(payload.get("adjusted_price_revision_proof_status") or ""):
        raise ValueError("Research V2 candidate adjusted-price revision proof status drift")
    if payload.get("adjusted_price_revision_proof") != proof:
        raise ValueError("Research V2 candidate persisted adjusted-price revision proof drift")
    archive_diagnostics = {
        "dataset_assessments": payload.get("dataset_assessments"),
        "dataset_assessment_counts": payload.get("dataset_assessment_counts"),
        "pit_review_contracts": payload.get("pit_review_contracts"),
        "pit_review_contract_counts": payload.get("pit_review_contract_counts"),
        "research_scope_contracts": payload.get("research_scope_contracts"),
        "research_scope_contract_counts": payload.get("research_scope_contract_counts"),
    }
    if str(payload.get("archive_diagnostics_fingerprint") or "") != canonical_json_sha256(archive_diagnostics):
        raise ValueError("Research V2 candidate supplemental archive diagnostics fingerprint drift")
    expected_adjusted_price_representation = adjusted_price_representation_contract_payload()
    expected_adjusted_price_representation_fingerprint = adjusted_price_representation_contract_fingerprint()
    if str(payload.get("adjusted_price_representation_contract_fingerprint") or "") != expected_adjusted_price_representation_fingerprint:
        raise ValueError("Research V2 candidate adjusted-price representation fingerprint drift")
    if payload.get("adjusted_price_representation_contract") != expected_adjusted_price_representation:
        raise ValueError("Research V2 candidate persisted adjusted-price representation contract drift")
    expected_scope_stats = validate_research_v2_dataset_scope_contracts()
    if str(payload.get("research_scope_contract_fingerprint") or "") != str(expected_scope_stats["contract_fingerprint"]):
        raise ValueError("Research V2 candidate required research scope contract fingerprint drift")
    if payload.get("required_dataset_scope") != expected_scope_stats["required_dataset_scope"]:
        raise ValueError("Research V2 candidate required dataset scope drift")
    date_audits = payload.get("dataset_date_audits")
    if not isinstance(date_audits, list):
        raise ValueError("Research V2 candidate dataset_date_audits 必須是 list")
    if str(payload.get("dataset_date_audit_fingerprint") or "") != canonical_json_sha256(date_audits):
        raise ValueError("Research V2 candidate dataset date-audit fingerprint drift")
    required_common = payload.get("required_common_complete")
    if not isinstance(required_common, dict):
        raise ValueError("Research V2 candidate required_common_complete 必須是 object")
    if set(required_common.get("participating_datasets") or []) != set(RESEARCH_V2_COMMON_COMPLETE_DATASETS):
        raise ValueError("Research V2 candidate required common-complete participant scope drift")
    if str(required_common.get("coverage_fingerprint") or "") != str(payload.get("required_common_complete_fingerprint") or ""):
        raise ValueError("Research V2 candidate required common-complete fingerprint drift")
    if required_common.get("common_complete_tail_start") != payload.get("required_common_complete_start_date"):
        raise ValueError("Research V2 candidate required common-complete start drift")
    if int(required_common.get("common_complete_tail_date_count") or 0) != int(payload.get("required_common_complete_tail_date_count") or 0):
        raise ValueError("Research V2 candidate required common-complete tail count drift")
    if required_common.get("common_complete_cutoff") != payload.get("research_common_complete_cutoff"):
        raise ValueError("Research V2 candidate required common-complete cutoff drift")
    if payload.get("research_common_complete_cutoff") not in {None, expected_required_cutoff}:
        raise ValueError("Research V2 candidate research_common_complete_cutoff 必須為 null 或 fixed required cutoff")
    scope_coverage_fp = str(payload.get("required_scope_coverage_fingerprint") or "")
    if len(scope_coverage_fp) != 64 or int(payload.get("required_scope_coverage_row_count") or 0) <= 0:
        raise ValueError("Research V2 candidate required-scope field completeness evidence 不完整")
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
    if payload.get("frozen_cutoff") is not None or bool(payload.get("promotion_authorized")):
        raise ValueError("Research V2 candidate layer 不得自行 promotion/freeze")
    return payload


def build_research_v2_freeze_candidate(
    project_root,
    *,
    snapshot_fingerprint: str | None = None,
    frame_reader: FrameReader | None = None,
    hash_fn: HashFn | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Build or REUSE one self-contained immutable freeze candidate.

    Freeze publication snapshots the source candidate manifest plus its derived
    daily-universe SQLite into the fingerprint-addressed freeze namespace.
    Later freeze/materialization/promotion validation therefore never depends on
    the mutable per-snapshot candidate slot remaining unchanged.
    """
    root = Path(project_root).resolve()
    candidate = load_research_v2_candidate(
        root,
        snapshot_fingerprint=snapshot_fingerprint,
        required=False,
    )
    if candidate is None:
        candidate = build_research_v2_candidate(
            root,
            snapshot_fingerprint=snapshot_fingerprint,
            frame_reader=frame_reader,
            hash_fn=hash_fn,
            now=now,
        )
    blockers = candidate.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError("Research V2 candidate blockers 必須是 list")
    if blockers:
        codes = [str(row.get("code") or "") for row in blockers if isinstance(row, dict)]
        raise RuntimeError(f"Research V2 freeze candidate 尚未 READY: blockers={codes}")

    snapshot_fp = str(candidate["provider_snapshot_fingerprint"])
    candidate_manifest_path = resolve_research_v2_candidate_manifest_path(root, snapshot_fp)
    if not candidate_manifest_path.is_file():
        raise FileNotFoundError("Research V2 source candidate manifest 不存在")
    candidate_manifest_sha = compute_file_sha256(candidate_manifest_path)
    candidate_universe_path = resolve_research_v2_daily_universe_path(root, snapshot_fp)
    if not candidate_universe_path.is_file():
        raise FileNotFoundError("Research V2 source candidate daily universe 不存在")
    candidate_universe_sha = compute_file_sha256(candidate_universe_path)
    if candidate_universe_sha != str(candidate["daily_universe_file_sha256"]):
        raise ValueError("Research V2 source candidate daily universe physical hash drift")

    required_common_payload = candidate.get("required_common_complete")
    if not isinstance(required_common_payload, dict):
        raise ValueError("Research V2 candidate required_common_complete 缺失")
    required_common_summary = ResearchV2RequiredCommonCompleteSummary(
        **{
            **required_common_payload,
            "participating_datasets": tuple(required_common_payload.get("participating_datasets") or ()),
        }
    )
    identity = build_research_v2_freeze_candidate_identity_payload(
        required_cutoff=str(candidate["required_cutoff"]),
        candidate_fingerprint=str(candidate["candidate_fingerprint"]),
        required_source_projection_fingerprint=str(candidate["required_source_projection_fingerprint"]),
        daily_universe_file_sha256=candidate_universe_sha,
        research_scope_contract_fingerprint=str(candidate["research_scope_contract_fingerprint"]),
        adjusted_price_representation_contract_fingerprint=str(
            candidate["adjusted_price_representation_contract_fingerprint"]
        ),
        adjusted_price_revision_proof_fingerprint=str(candidate["adjusted_price_revision_proof_fingerprint"]),
        required_common_complete_summary=required_common_summary,
    )
    freeze_fingerprint = canonical_json_sha256(identity)
    freeze_dir = resolve_research_v2_freeze_candidate_dir(root, freeze_fingerprint)
    manifest_path = resolve_research_v2_freeze_candidate_manifest_path(root, freeze_fingerprint)
    built_at = (now or datetime.now().astimezone()).astimezone().isoformat()
    payload = {
        **identity,
        "freeze_candidate_fingerprint": freeze_fingerprint,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "provider_snapshot_fingerprint": snapshot_fp,
        "provider_manifest_fingerprint": str(candidate["provider_manifest_fingerprint"]),
        "provider_as_of_date": str(candidate["provider_as_of_date"]),
        "built_at": built_at,
        "candidate_manifest_path": project_relative_display_path(candidate_manifest_path, project_root=root),
        "frozen_candidate_manifest_path": project_relative_display_path(
            resolve_research_v2_frozen_source_candidate_manifest_path(root, freeze_fingerprint),
            project_root=root,
        ),
        "frozen_daily_universe_path": project_relative_display_path(
            resolve_research_v2_frozen_daily_universe_path(root, freeze_fingerprint),
            project_root=root,
        ),
        "required_common_complete": required_common_payload,
        "provider_calls": 0,
    }
    if manifest_path.is_file():
        loaded = load_research_v2_freeze_candidate(
            root, freeze_candidate_fingerprint=freeze_fingerprint, required=True
        )
        return {
            **loaded,
            "manifest_path": project_relative_display_path(manifest_path, project_root=root),
            "provider_calls": 0,
            "reused": True,
        }
    if freeze_dir.exists():
        raise ValueError("Research V2 freeze immutable path 已存在但 manifest 缺失")

    parent = freeze_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{freeze_fingerprint}.", suffix=".tmp", dir=str(parent)))
    try:
        shutil.copyfile(candidate_manifest_path, stage_dir / "source_candidate_manifest.json")
        shutil.copyfile(candidate_universe_path, stage_dir / "daily_universe.sqlite3")
        atomic_write_json(stage_dir / manifest_path.name, payload)
        try:
            os.replace(stage_dir, freeze_dir)
        except FileExistsError:
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            loaded = load_research_v2_freeze_candidate(
                root, freeze_candidate_fingerprint=freeze_fingerprint, required=True
            )
            return {
                **loaded,
                "manifest_path": project_relative_display_path(manifest_path, project_root=root),
                "provider_calls": 0,
                "reused": True,
            }
        loaded = load_research_v2_freeze_candidate(
            root, freeze_candidate_fingerprint=freeze_fingerprint, required=True
        )
        return {
            **loaded,
            "manifest_path": project_relative_display_path(manifest_path, project_root=root),
            "provider_calls": 0,
            "reused": False,
        }
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def load_research_v2_freeze_candidate(
    project_root,
    *,
    freeze_candidate_fingerprint: str,
    required: bool = False,
) -> dict[str, object] | None:
    root = Path(project_root).resolve()
    path = resolve_research_v2_freeze_candidate_manifest_path(root, freeze_candidate_fingerprint)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Research V2 freeze candidate manifest 尚未建立")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("Research V2 freeze candidate manifest 必須是 object")
    identity = {key: payload.get(key) for key in RESEARCH_V2_FREEZE_CANDIDATE_IDENTITY_FIELDS}
    actual_fingerprint = canonical_json_sha256(identity)
    if actual_fingerprint != str(freeze_candidate_fingerprint).strip().lower():
        raise ValueError("Research V2 freeze candidate path fingerprint 不一致")
    if str(payload.get("freeze_candidate_fingerprint") or "") != actual_fingerprint:
        raise ValueError("Research V2 freeze candidate fingerprint 不一致")
    if str(payload.get("status") or "") != RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY:
        raise ValueError("Research V2 freeze candidate status 不合法")
    if bool(payload.get("promotion_authorized")) or bool(payload.get("active_research_generation_changed")):
        raise ValueError("Research V2 freeze candidate 不得自行 promotion/切換 active generation")
    if payload.get("research_common_complete_cutoff") != payload.get("required_cutoff"):
        raise ValueError("Research V2 freeze candidate common-complete cutoff 未覆蓋 required cutoff")
    if payload.get("frozen_cutoff") != payload.get("required_cutoff"):
        raise ValueError("Research V2 freeze candidate frozen cutoff 必須等於 fixed required cutoff")
    if set(payload.get("required_dataset_scope") or []) != set(
        validate_research_v2_dataset_scope_contracts()["required_dataset_scope"]
    ):
        raise ValueError("Research V2 freeze candidate required dataset scope drift")
    if set(payload.get("common_complete_dataset_scope") or []) != set(RESEARCH_V2_COMMON_COMPLETE_DATASETS):
        raise ValueError("Research V2 freeze candidate common-complete dataset scope drift")

    frozen_candidate_path = resolve_research_v2_frozen_source_candidate_manifest_path(root, actual_fingerprint)
    if not frozen_candidate_path.is_file():
        raise FileNotFoundError("Research V2 freeze source candidate snapshot 不存在")
    if compute_file_sha256(frozen_candidate_path) != str(payload.get("candidate_manifest_sha256") or ""):
        raise ValueError("Research V2 freeze source candidate manifest SHA256 drift")
    candidate = load_json_strict(frozen_candidate_path)
    if not isinstance(candidate, dict):
        raise ValueError("Research V2 frozen source candidate manifest 必須是 object")
    candidate_identity = {key: candidate.get(key) for key in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS}
    if canonical_json_sha256(candidate_identity) != str(payload.get("candidate_fingerprint") or ""):
        raise ValueError("Research V2 freeze source candidate fingerprint drift")
    if str(payload.get("required_source_projection_fingerprint") or "") != str(
        candidate.get("required_source_projection_fingerprint") or ""
    ):
        raise ValueError("Research V2 freeze candidate required source projection drift")
    if str(payload.get("adjusted_price_revision_proof_fingerprint") or "") != str(
        candidate.get("adjusted_price_revision_proof_fingerprint") or ""
    ):
        raise ValueError("Research V2 freeze candidate adjusted-price revision proof drift")
    if str(candidate.get("adjusted_price_revision_proof_status") or "") != ADJUSTED_PRICE_REVISION_STATUS_READY:
        raise ValueError("Research V2 freeze candidate requires READY adjusted-price revision proof")
    if str(payload.get("daily_universe_file_sha256") or "") != str(candidate.get("daily_universe_file_sha256") or ""):
        raise ValueError("Research V2 freeze candidate daily universe artifact drift")
    frozen_universe_path = resolve_research_v2_frozen_daily_universe_path(root, actual_fingerprint)
    if not frozen_universe_path.is_file():
        raise FileNotFoundError("Research V2 frozen daily universe 不存在")
    if compute_file_sha256(frozen_universe_path) != str(payload.get("daily_universe_file_sha256") or ""):
        raise ValueError("Research V2 frozen daily universe SHA256 drift")
    if str(payload.get("research_scope_contract_fingerprint") or "") != str(
        candidate.get("research_scope_contract_fingerprint") or ""
    ):
        raise ValueError("Research V2 freeze candidate scope contract drift")
    if str(payload.get("adjusted_price_representation_contract_fingerprint") or "") != str(
        candidate.get("adjusted_price_representation_contract_fingerprint") or ""
    ):
        raise ValueError("Research V2 freeze candidate adjusted-price representation drift")
    if payload.get("research_common_complete_cutoff") != candidate.get("research_common_complete_cutoff"):
        raise ValueError("Research V2 freeze candidate source common-complete cutoff drift")
    if candidate.get("blockers"):
        raise ValueError("Research V2 freeze candidate source candidate 仍有 blocker")
    required_common = payload.get("required_common_complete")
    if not isinstance(required_common, dict):
        raise ValueError("Research V2 freeze candidate required_common_complete 必須是 object")
    if str(required_common.get("coverage_fingerprint") or "") != str(
        payload.get("required_common_complete_fingerprint") or ""
    ):
        raise ValueError("Research V2 freeze candidate common-complete evidence drift")
    return payload


__all__ = [
    "ResearchV2ProviderView",
    "build_research_v2_candidate",
    "load_research_v2_candidate",
    "build_research_v2_freeze_candidate",
    "load_research_v2_freeze_candidate",
]
