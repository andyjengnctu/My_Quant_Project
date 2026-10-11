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
from functools import lru_cache
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Mapping

import pandas as pd

from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.data_utils import (
    OHLCV_SANITIZATION_CONTRACT_VERSION,
    discover_unique_csv_inputs,
    sanitize_ohlcv_dataframe,
)
from core.dataset_profiles import build_dataset_generation_identity, get_dataset_dir
from core.file_integrity import canonical_json_sha256, compute_file_sha256, is_sha256_hex, load_json_strict
from core.market_data_contract import FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.market_data_raw_quote import raw_ohlc_diagnostic_masks
from core.market_data_research_materialization import deterministic_compatibility_file_mtime_ns
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.research.market_data_v2 import ResearchV2ProviderView

RAW_PRICE_STRATEGY_DATASET_SCHEMA_VERSION = 2
RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID = "research_raw_unadjusted_controlled_ohlcv_v2"
RAW_PRICE_STRATEGY_PRICE_BASIS = "raw_unadjusted"
RAW_PRICE_QUOTE_POLICY_STRICT = "strict_numeric_v1"
RAW_PRICE_QUOTE_POLICY_OBSERVED = "observed_with_missing_quote_mask_v1"
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


def _source_identity(
    project_root: str | Path, *, quote_policy: str = RAW_PRICE_QUOTE_POLICY_STRICT,
) -> tuple[dict[str, object], ResearchV2ProviderView, Path]:
    if quote_policy not in {RAW_PRICE_QUOTE_POLICY_STRICT, RAW_PRICE_QUOTE_POLICY_OBSERVED}:
        raise ValueError(f"Unsupported raw source quote policy: {quote_policy}")
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
    if quote_policy == RAW_PRICE_QUOTE_POLICY_OBSERVED:
        # New immutable scientific input source. Strict/non-DL source identity is untouched.
        identity["contract_id"] = "research_raw_observed_quote_mask_v1"
        identity["generation_id"] = "research_raw_observed_quote_mask_v1"
        identity["quote_policy"] = quote_policy
    provider_fp = str(identity["provider_snapshot_fingerprint"])
    if not is_sha256_hex(provider_fp):
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



def _adjusted_control_required_raw_rows(
    raw_frame: pd.DataFrame, *, control_path: Path, ticker: str,
) -> pd.DataFrame:
    """Restrict the Raw quality gate to dates retained by the canonical adjusted loader.

    The physical Raw CSV still preserves the full control file's original date and
    Volume grid. Invalid Raw prices on dates already excluded from adjusted samples
    cannot change this experiment's feature windows or targets; required-date Raw
    defects must still fail closed. Both decisions use the same canonical sanitizer.
    """
    control = pd.read_csv(control_path, low_memory=False)
    try:
        adjusted, _stats = sanitize_ohlcv_dataframe(control, ticker, min_rows=1)
    except ValueError as exc:
        # With min_rows=1, an entirely invalid control source has no model dates.
        if str(exc) != f"{ticker} \u6e05\u6d17\u5f8c\u7121\u6709\u6548\u8cc7\u6599":
            raise
        return raw_frame.iloc[0:0].copy()
    required_dates = pd.DatetimeIndex(adjusted.index).normalize()
    raw_dates = pd.to_datetime(raw_frame["Date"], errors="coerce").dt.normalize()
    return raw_frame.loc[raw_dates.isin(required_dates)].copy()


def _raw_quote_diagnostics(frame: pd.DataFrame) -> dict[str, int]:
    """Project-wide raw-source diagnostics on an explicit controlled date grid.

    This does not decide whether a DL model, non-DL strategy or broker execution
    may use a given raw bar.  The same provider observations have different
    permitted downstream uses, determined by those consumer contracts.
    """
    return {
        name: int(mask.sum())
        for name, mask in raw_ohlc_diagnostic_masks(frame).items()
    }


def _raw_price_quality_failures(frame: pd.DataFrame, *, ticker: str) -> tuple[int, list[dict[str, object]]]:
    """Inspect Raw prices; retain unavailable-quote evidence, never repair it.

    The canonical sanitizer rejects all-zero prices for numerical model input,
    but zero OHLC may be an authentic provider no-quote record, rather than a
    corrupt row that can be recovered by re-downloading the same dataset.
    """
    try:
        clean, stats = sanitize_ohlcv_dataframe(frame, ticker, min_rows=1)
    except ValueError as exc:
        if "清洗後無有效資料" not in str(exc):
            raise
        removed = len(frame)
        clean_dates = pd.DatetimeIndex([])
        duplicate_count = 0
    else:
        removed = int(stats["invalid_row_count"]) + int(stats["duplicate_date_count"])
        if not removed:
            return 0, []
        clean_dates = pd.DatetimeIndex(clean.index).normalize()
        duplicate_count = int(stats["duplicate_date_count"])
    source_dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    rejected = ~source_dates.isin(clean_dates)
    if duplicate_count:
        rejected |= source_dates.duplicated(keep="last")
    raw_quote_masks = raw_ohlc_diagnostic_masks(frame)
    sample: list[dict[str, object]] = []
    for offset in frame.index[rejected][:8]:
        row = frame.loc[offset]
        date = source_dates.loc[offset]
        all_zero = bool(raw_quote_masks["all_zero_ohlc"].loc[offset])
        sample.append({
            "ticker": str(ticker),
            "date": date.strftime("%Y-%m-%d") if pd.notna(date) else str(row["Date"]),
            "ohlc": {field: str(row[field]) for field in ("Open", "High", "Low", "Close")},
            "quote_class": "ALL_ZERO_NO_PUBLISHED_QUOTE_POSSIBLE" if all_zero else "OTHER_INVALID_OHLC",
            "control_volume": str(row["Volume"]),
        })
    return removed, sample



@lru_cache(maxsize=4096)
def _cached_raw_price_quality(
    absolute_path: str, expected_content_sha256: str, sanitization_contract_version: int,
    ticker: str, control_path: str, control_inventory_fingerprint: str,
) -> tuple[int, list[dict[str, object]], dict[str, int]]:
    """Avoid re-parsing immutable, already-SHA-verified Raw files within one run.

    Callers check the content hash *before* this function.  A new file hash or
    canonical sanitizer version always invalidates the in-process cached proof.
    """
    del expected_content_sha256, sanitization_contract_version, control_inventory_fingerprint
    try:
        frame = pd.read_csv(absolute_path, low_memory=False)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"raw-price strategy dataset CSV無法檢查OHLCV: {Path(absolute_path).name}") from exc
    required = _adjusted_control_required_raw_rows(
        frame, control_path=Path(control_path), ticker=ticker,
    )
    invalid_rows, samples = _raw_price_quality_failures(required, ticker=ticker)
    return invalid_rows, samples, _raw_quote_diagnostics(required)


def _fail_on_raw_price_quality(issues: list[dict[str, object]], *, provider: str) -> None:
    if not issues:
        return
    total_rows = sum(int(issue["invalid_rows"]) for issue in issues)
    sample: list[dict[str, object]] = []
    for issue in issues:
        for item in issue["samples"]:
            if len(sample) < 24:
                sample.append(dict(item))
    zero_rows = sum(int(issue.get("quote_diagnostics", {}).get("all_zero_ohlc", 0)) for issue in issues)
    zero_volume = sum(int(issue.get("quote_diagnostics", {}).get("all_zero_control_volume_zero", 0)) for issue in issues)
    positive_volume = sum(int(issue.get("quote_diagnostics", {}).get("all_zero_control_volume_positive", 0)) for issue in issues)
    other_volume = sum(int(issue.get("quote_diagnostics", {}).get("all_zero_control_volume_other", 0)) for issue in issues)
    open_outside = sum(int(issue.get("quote_diagnostics", {}).get("nonzero_open_outside_high_low", 0)) for issue in issues)
    close_outside = sum(int(issue.get("quote_diagnostics", {}).get("nonzero_close_outside_high_low", 0)) for issue in issues)
    partial_nonpositive = sum(int(issue.get("quote_diagnostics", {}).get("nonzero_any_nonpositive_price", 0)) for issue in issues)
    raise ValueError(
        "RAW_SOURCE_QUALITY_BLOCKED: FinMind TaiwanStockPrice Raw OHLCV "
        f"has {total_rows} invalid rows on canonical adjusted retained dates "
        f"across {len(issues)} tickers; "
        f"all_zero_ohlc={zero_rows} (control_volume_zero={zero_volume}, "
        f"control_volume_positive={positive_volume}, control_volume_other={other_volume}); "
        f"other_invalid={total_rows - zero_rows}; "
        f"diagnostic_nonzero_open_outside_range={open_outside}, "
        f"close_outside_range={close_outside}, "
        f"any_nonpositive_price={partial_nonpositive} (categories may overlap); "
        f"sample={sample}; provider_snapshot={provider}; "
        "all-zero OHLC can mean no published market quote; emerging-market Open "
        "can be the previous-day average outside same-day High/Low; neither "
        "is automatically proof of corrupt provider data. "
        "A same-source re-download is not a demonstrated repair. Under the current "
        "controlled raw-price source contract, affected Research consumers (including "
        "non-DL raw-price studies and DL alternate-input studies) cannot use these "
        "bars as valid numeric prices. Any missing-quote representation, imputation, "
        "matched-date sample change or external price-source substitution requires "
        "separate scientific authorization and identity; no silent fallback or checkpoint reuse."
    )


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
        control_paths = {str(ticker): Path(raw_path) for ticker, raw_path in discovered}
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
        quality_issues: list[dict[str, object]] = []
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
            # Only strict numeric consumers require every retained quote to pass
            # the legacy sanitizer. Quote-aware materialization preserves source
            # observations and delegates per-bar availability to the shared mask.
            if source_identity.get("quote_policy", RAW_PRICE_QUOTE_POLICY_STRICT) == RAW_PRICE_QUOTE_POLICY_STRICT:
                required = _adjusted_control_required_raw_rows(
                    frame, control_path=control_paths[ticker], ticker=ticker,
                )
                invalid_rows, bad_samples = _raw_price_quality_failures(required, ticker=ticker)
                if invalid_rows:
                    quality_issues.append({
                        "ticker": ticker, "invalid_rows": invalid_rows, "samples": bad_samples,
                        "quote_diagnostics": _raw_quote_diagnostics(required),
                    })
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
        _fail_on_raw_price_quality(
            quality_issues, provider=str(source_identity["provider_snapshot_fingerprint"]),
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


def inspect_raw_price_strategy_dataset(
    project_root: str | Path, *, deep: bool = False,
    quote_policy: str = RAW_PRICE_QUOTE_POLICY_STRICT,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    try:
        identity, _view, control_dir = _source_identity(root, quote_policy=quote_policy)
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
        quality_issues: list[dict[str, object]] = []
        discovered, _duplicate_issues = discover_unique_csv_inputs(str(control_dir))
        control_paths = {str(ticker): Path(file_path) for ticker, file_path in discovered}
        for row in files:
            rel = str(row.get("relative_path") or "")
            path = dataset_dir / rel
            if not path.is_file():
                raise FileNotFoundError(f"raw-price strategy dataset CSV不存在: {rel}")
            if int(path.stat().st_size) != int(row.get("size_bytes") or -1):
                raise ValueError(f"raw-price strategy dataset CSV size drift: {rel}")
            if compute_file_sha256(path) != str(row.get("content_sha256") or ""):
                raise ValueError(f"raw-price strategy dataset CSV SHA256 drift: {rel}")
            ticker = str(row.get("ticker") or Path(rel).stem)
            control_path = control_paths.get(ticker)
            if control_path is None:
                raise ValueError(f"raw-price strategy control ticker missing: {ticker}")
            if quote_policy == RAW_PRICE_QUOTE_POLICY_STRICT:
                invalid_rows, bad_samples, quote_diagnostics = _cached_raw_price_quality(
                    str(path.resolve()), str(row["content_sha256"]),
                    int(OHLCV_SANITIZATION_CONTRACT_VERSION), ticker,
                    str(control_path.resolve()),
                    str(identity["control_dataset_inventory_fingerprint"]),
                )
                if invalid_rows:
                    quality_issues.append({
                        "ticker": ticker,
                        "invalid_rows": invalid_rows,
                        "samples": bad_samples,
                        "quote_diagnostics": quote_diagnostics,
                    })
        _fail_on_raw_price_quality(
            quality_issues, provider=str(identity["provider_snapshot_fingerprint"]),
        )
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


def ensure_raw_price_strategy_dataset(
    project_root: str | Path, *, quote_policy: str = RAW_PRICE_QUOTE_POLICY_STRICT,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    state = inspect_raw_price_strategy_dataset(root, deep=False, quote_policy=quote_policy)
    if bool(state.get("ready")):
        return inspect_raw_price_strategy_dataset(root, deep=True, quote_policy=quote_policy)

    identity, view, control_dir = _source_identity(root, quote_policy=quote_policy)
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
            "generation_id": str(identity["generation_id"]),
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
    return inspect_raw_price_strategy_dataset(root, deep=True, quote_policy=quote_policy)


__all__ = [
    "RAW_PRICE_STRATEGY_DATASET_CONTRACT_ID",
    "RAW_PRICE_STRATEGY_PRICE_BASIS",
    "RAW_PRICE_QUOTE_POLICY_STRICT",
    "RAW_PRICE_QUOTE_POLICY_OBSERVED",
    "ensure_raw_price_strategy_dataset",
    "inspect_raw_price_strategy_dataset",
    "raw_price_strategy_dataset_source_identity",
]
