"""Canonical content identity for the live Trading CSV dataset."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from core.data_utils import discover_unique_csv_inputs
from core.dataset_dates import resolve_latest_dataset_date
from core.file_integrity import compute_file_sha256


@dataclass(frozen=True)
class TradingDatasetMemberDateEvidence:
    exists: bool
    readable: bool
    dates: frozenset[str]
    last_date: str | None
    error: str | None = None


def inspect_trading_dataset_member_date_evidence(
    path: str | Path,
    *,
    through_date: object,
) -> TradingDatasetMemberDateEvidence:
    """Inspect one canonical Trading CSV date history without treating unreadable bytes as truth."""

    import pandas as pd

    csv_path = Path(path)
    if not csv_path.is_file():
        return TradingDatasetMemberDateEvidence(False, False, frozenset(), None, "missing csv")
    target_ts = pd.Timestamp(through_date).normalize()
    try:
        frame = pd.read_csv(csv_path, index_col=0)
        if frame.empty:
            raise ValueError("empty csv")
        parsed = pd.to_datetime(frame.index, errors="coerce")
        dates = {
            item.date().isoformat()
            for item in parsed
            if not pd.isna(item) and item.normalize() <= target_ts
        }
        if not dates:
            raise ValueError("no valid historical dates")
        return TradingDatasetMemberDateEvidence(True, True, frozenset(dates), max(dates), None)
    except (OSError, ValueError, TypeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return TradingDatasetMemberDateEvidence(
            True,
            False,
            frozenset(),
            None,
            f"{type(exc).__name__}: {exc}",
        )


def resolve_trading_dataset_member_latest_dates(
    data_dir: str | Path,
    tickers,
) -> dict[str, str]:
    """Resolve canonical per-ticker latest dates from the same CSV membership SSOT."""

    import pandas as pd

    root = Path(data_dir).resolve()
    csv_inputs, duplicate_issue_lines = discover_unique_csv_inputs(root)
    if duplicate_issue_lines:
        raise RuntimeError(
            "Trading dataset 存在同 ticker 重複 CSV，無法驗證 member freshness："
            + " | ".join(duplicate_issue_lines)
        )
    by_ticker = {str(ticker): Path(path) for ticker, path in csv_inputs}
    requested = tuple(dict.fromkeys(str(item).strip() for item in tickers if str(item).strip()))
    result: dict[str, str] = {}
    for ticker in requested:
        path = by_ticker.get(ticker)
        if path is None:
            raise FileNotFoundError(f"Trading dataset 缺少 ticker CSV: {ticker}")
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda col: str(col).strip().lower() in {"date", "time", "unnamed: 0"},
            )
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"Trading dataset ticker CSV 無法讀取日期: {ticker} | {type(exc).__name__}: {exc}"
            ) from exc
        if frame.empty or not list(frame.columns):
            raise RuntimeError(f"Trading dataset ticker CSV 沒有日期資料: {ticker}")
        column = next(
            (
                name
                for name in frame.columns
                if str(name).strip().lower() in {"date", "time", "unnamed: 0"}
            ),
            None,
        )
        if column is None:
            raise RuntimeError(f"Trading dataset ticker CSV 缺少 Date/Time index: {ticker}")
        parsed = pd.to_datetime(frame[column], errors="coerce").dropna()
        if parsed.empty:
            raise RuntimeError(f"Trading dataset ticker CSV 沒有合法日期: {ticker}")
        result[ticker] = parsed.max().normalize().strftime("%Y-%m-%d")
    return result


def assert_trading_dataset_tickers_current(
    data_dir: str | Path,
    *,
    tickers,
    market_date: object,
) -> dict[str, str]:
    """Fail closed unless every actionable ticker is current to ``market_date``."""

    target = str(market_date).strip()
    latest = resolve_trading_dataset_member_latest_dates(data_dir, tickers)
    stale = {ticker: value for ticker, value in latest.items() if value != target}
    if stale:
        sample = list(sorted(stale.items()))[:20]
        raise RuntimeError(
            "Trading dataset actionable universe freshness 不完整；"
            f"market_date={target} stale_count={len(stale)} sample={sample}"
        )
    return latest


def build_trading_dataset_fingerprint(data_dir: str | Path) -> dict[str, Any]:
    """Fingerprint every Trading CSV member by path, size and SHA256 content."""

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Trading dataset directory 不存在: {root}")
    csv_inputs, duplicate_issue_lines = discover_unique_csv_inputs(root)
    if duplicate_issue_lines:
        raise RuntimeError(
            "Trading dataset 存在同 ticker 重複 CSV，canonical consumer 與 dataset identity 不得分叉："
            + " | ".join(duplicate_issue_lines)
        )
    members = [(str(ticker), Path(path)) for ticker, path in csv_inputs]
    if not members:
        raise FileNotFoundError(f"Trading dataset 沒有 canonical CSV: {root}")

    members_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    total_bytes = 0
    for ticker, path in members:
        rel = path.relative_to(root).as_posix()
        size = int(path.stat().st_size)
        file_sha = compute_file_sha256(path)
        total_bytes += size
        members_digest.update(rel.encode("utf-8")); members_digest.update(b"\0")
        members_digest.update(str(size).encode("ascii")); members_digest.update(b"\0")
        content_digest.update(rel.encode("utf-8")); content_digest.update(b"\0")
        content_digest.update(file_sha.encode("ascii")); content_digest.update(b"\0")

    return {
        "csv_count": len(members),
        "csv_total_bytes": total_bytes,
        "csv_members_sha256": members_digest.hexdigest(),
        "csv_content_sha256": content_digest.hexdigest(),
        "fingerprint_algorithm": "sha256",
        "latest_data_date": str(resolve_latest_dataset_date(root)),
    }


def assert_trading_dataset_fingerprint_matches(
    data_dir: str | Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when live Trading CSV content diverges from its bound snapshot."""

    actual = build_trading_dataset_fingerprint(data_dir)
    fields = (
        "csv_count",
        "csv_total_bytes",
        "csv_members_sha256",
        "csv_content_sha256",
        "fingerprint_algorithm",
        "latest_data_date",
    )
    mismatches = [field for field in fields if actual.get(field) != expected.get(field)]
    if mismatches:
        raise RuntimeError(
            "Trading dataset content 已與 canonical market-data snapshot 不一致: "
            + ",".join(mismatches)
            + "；請重新執行「1 更新 Trading 資料」"
        )
    return actual


__all__ = [
    "TradingDatasetMemberDateEvidence",
    "inspect_trading_dataset_member_date_evidence",
    "build_trading_dataset_fingerprint",
    "assert_trading_dataset_fingerprint_matches",
    "resolve_trading_dataset_member_latest_dates",
    "assert_trading_dataset_tickers_current",
]
