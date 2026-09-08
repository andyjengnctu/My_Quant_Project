"""Build/load immutable Research V2 adjusted-price revision proof artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd

from core.console_report import project_relative_display_path
from core.dataset_profiles import DATASET_PROFILE_FULL, get_unpromoted_dataset_dir
from core.file_integrity import atomic_write_json, compute_file_sha256, load_json_strict
from core.market_data_adjusted_price_invariance import PROVIDER_PRICE_FIELDS
from core.market_data_adjusted_price_revision_proof import (
    ADJUSTED_PRICE_REVISION_PROOF_IDENTITY_FIELDS,
    ADJUSTED_PRICE_REVISION_STATUS_BLOCKED,
    ADJUSTED_PRICE_REVISION_STATUS_READY,
    adjusted_price_revision_proof_fingerprint,
    build_adjusted_price_revision_proof_identity_payload,
    validate_adjusted_price_revision_proof_payload,
)
from core.market_data_research_scope import RESEARCH_V2_ADJUSTED_PRICE_DATASET
from core.market_data_research_storage_contract import (
    resolve_research_v2_adjusted_price_proof_manifest_path,
)


def _digest_row(digest, row: Iterable[object]) -> None:
    digest.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    digest.update(b"\n")


def _as_decimal(value) -> Decimal | None:
    try:
        out = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not out.is_finite():
        return None
    return out


def _normalize_current(frame: pd.DataFrame, *, cutoff: str) -> dict[str, tuple[Decimal, Decimal, Decimal, Decimal]]:
    needed = {"date", "stock_id", *PROVIDER_PRICE_FIELDS}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"TaiwanStockPriceAdj revision proof 缺欄位: {sorted(missing)}")
    local = frame.loc[:, ["date", "open", "max", "min", "close"]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rows: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    for item in local.itertuples(index=False, name=None):
        date_value = str(item[0] or "")
        if not date_value or date_value > cutoff:
            continue
        values = tuple(_as_decimal(value) for value in item[1:])
        if any(value is None for value in values):
            continue
        normalized = values  # type: ignore[assignment]
        prior = rows.get(date_value)
        if prior is not None and prior != normalized:
            raise ValueError(f"TaiwanStockPriceAdj revision proof duplicate conflicting date: {date_value}")
        rows[date_value] = normalized
    return rows


def _normalize_legacy(frame: pd.DataFrame, *, cutoff: str) -> dict[str, tuple[Decimal, Decimal, Decimal, Decimal]]:
    aliases = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
    }
    missing = set(aliases).difference(frame.columns)
    if missing:
        raise ValueError(f"Research V1 adjusted-price revision reference 缺欄位: {sorted(missing)}")
    local = frame.loc[:, ["Date", "Open", "High", "Low", "Close"]].copy()
    local["Date"] = pd.to_datetime(local["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rows: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    for item in local.itertuples(index=False, name=None):
        date_value = str(item[0] or "")
        if not date_value or date_value > cutoff:
            continue
        values = tuple(_as_decimal(value) for value in item[1:])
        if any(value is None for value in values):
            continue
        normalized = values  # type: ignore[assignment]
        prior = rows.get(date_value)
        if prior is not None and prior != normalized:
            raise ValueError(f"Research V1 revision reference duplicate conflicting date: {date_value}")
        rows[date_value] = normalized
    return rows


def build_adjusted_price_revision_proof(
    project_root,
    *,
    provider_view,
    daily_universe_path: Path,
    required_cutoff: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Prove current PriceAdj is one positive scalar from frozen V1 per required stock.

    This function never computes adjusted prices.  It compares two FinMind-derived
    histories already persisted by the project and emits READY only for exact
    positive-scalar equivalence over every Research-V2 required stock-day.
    """
    root = Path(project_root).resolve()
    cutoff = str(required_cutoff)
    universe_path = Path(daily_universe_path)
    if not universe_path.is_file():
        raise FileNotFoundError("adjusted-price revision proof daily universe 不存在")
    universe_sha = compute_file_sha256(universe_path)
    legacy_dir = Path(get_unpromoted_dataset_dir(root, DATASET_PROFILE_FULL)).resolve()

    conn = sqlite3.connect(universe_path)
    try:
        stock_ids = tuple(
            str(row[0])
            for row in conn.execute("SELECT DISTINCT stock_id FROM daily_universe ORDER BY stock_id")
        )
        if not stock_ids:
            raise ValueError("adjusted-price revision proof required stock universe 不可為空")

        legacy_digest = hashlib.sha256()
        current_digest = hashlib.sha256()
        required_stock_days = 0
        proven_stock_days = 0
        proven_stocks = 0
        missing_legacy_stocks = 0
        missing_legacy_days = 0
        missing_current_days = 0
        mismatches = 0
        examples: list[dict[str, object]] = []

        with localcontext() as ctx:
            ctx.prec = 50
            for stock_id in stock_ids:
                required_dates = tuple(
                    str(row[0])
                    for row in conn.execute(
                        "SELECT date FROM daily_universe WHERE stock_id = ? AND date <= ? ORDER BY date",
                        (stock_id, cutoff),
                    )
                )
                if not required_dates:
                    continue
                required_stock_days += len(required_dates)

                current_frames = list(
                    provider_view.iter_dataset_scope_frames(
                        RESEARCH_V2_ADJUSTED_PRICE_DATASET,
                        columns=("date", "stock_id", *PROVIDER_PRICE_FIELDS),
                        data_id=stock_id,
                    )
                )
                if current_frames:
                    current_map = _normalize_current(pd.concat(current_frames, ignore_index=True), cutoff=cutoff)
                else:
                    current_map = {}

                legacy_path = legacy_dir / f"{stock_id}.csv"
                if not legacy_path.is_file():
                    missing_legacy_stocks += 1
                    missing_legacy_days += len(required_dates)
                    _digest_row(legacy_digest, ("MISSING_STOCK", stock_id, len(required_dates)))
                    if len(examples) < 20:
                        examples.append({"stock_id": stock_id, "reason": "legacy_v1_file_missing"})
                    for date_value in required_dates:
                        current_values = current_map.get(date_value)
                        if current_values is None:
                            missing_current_days += 1
                            _digest_row(current_digest, (stock_id, date_value, "MISSING"))
                        else:
                            _digest_row(current_digest, (stock_id, date_value, *(str(v) for v in current_values)))
                    continue

                legacy_map = _normalize_legacy(pd.read_csv(legacy_path), cutoff=cutoff)
                reference_pair: tuple[Decimal, Decimal] | None = None
                stock_ok = True
                stock_proven_days = 0

                for date_value in required_dates:
                    legacy_values = legacy_map.get(date_value)
                    current_values = current_map.get(date_value)
                    if legacy_values is None:
                        missing_legacy_days += 1
                        stock_ok = False
                        _digest_row(legacy_digest, (stock_id, date_value, "MISSING"))
                    else:
                        _digest_row(legacy_digest, (stock_id, date_value, *(str(v) for v in legacy_values)))
                    if current_values is None:
                        missing_current_days += 1
                        stock_ok = False
                        _digest_row(current_digest, (stock_id, date_value, "MISSING"))
                    else:
                        _digest_row(current_digest, (stock_id, date_value, *(str(v) for v in current_values)))
                    if legacy_values is None or current_values is None:
                        continue

                    day_ok = True
                    for legacy_value, current_value in zip(legacy_values, current_values):
                        if legacy_value <= 0 or current_value <= 0:
                            day_ok = False
                            break
                        if reference_pair is None:
                            reference_pair = (legacy_value, current_value)
                        ref_legacy, ref_current = reference_pair
                        if current_value * ref_legacy != legacy_value * ref_current:
                            day_ok = False
                            break
                    if day_ok:
                        stock_proven_days += 1
                    else:
                        mismatches += 1
                        stock_ok = False
                        if len(examples) < 20:
                            examples.append({"stock_id": stock_id, "date": date_value, "reason": "non_scalar_adjusted_price_revision"})

                if reference_pair is None:
                    stock_ok = False
                    if len(examples) < 20:
                        examples.append({"stock_id": stock_id, "reason": "no_positive_reference_pair"})
                proven_stock_days += stock_proven_days
                if stock_ok and stock_proven_days == len(required_dates):
                    proven_stocks += 1
    finally:
        conn.close()

    status = (
        ADJUSTED_PRICE_REVISION_STATUS_READY
        if (
            proven_stocks == len(stock_ids)
            and proven_stock_days == required_stock_days
            and missing_legacy_stocks == 0
            and missing_legacy_days == 0
            and missing_current_days == 0
            and mismatches == 0
        )
        else ADJUSTED_PRICE_REVISION_STATUS_BLOCKED
    )
    identity = build_adjusted_price_revision_proof_identity_payload(
        required_cutoff=cutoff,
        status=status,
        required_stock_count=len(stock_ids),
        proven_stock_count=proven_stocks,
        required_stock_day_count=required_stock_days,
        proven_stock_day_count=proven_stock_days,
        missing_legacy_stock_count=missing_legacy_stocks,
        missing_legacy_stock_day_count=missing_legacy_days,
        missing_current_stock_day_count=missing_current_days,
        non_scalar_mismatch_count=mismatches,
        legacy_required_source_fingerprint=legacy_digest.hexdigest(),
        current_required_source_fingerprint=current_digest.hexdigest(),
        daily_universe_file_sha256=universe_sha,
    )
    proof_fp = adjusted_price_revision_proof_fingerprint(identity)
    manifest_path = resolve_research_v2_adjusted_price_proof_manifest_path(root, proof_fp)
    payload = {
        **identity,
        "proof_fingerprint": proof_fp,
        "built_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "legacy_reference_dataset_dir": project_relative_display_path(legacy_dir, project_root=root),
        "mismatch_examples": examples,
        "provider_calls": 0,
    }
    if manifest_path.is_file():
        existing = load_json_strict(manifest_path)
        if not isinstance(existing, dict):
            raise ValueError("adjusted-price revision proof manifest 必須是 object")
        existing_identity = {key: existing.get(key) for key in ADJUSTED_PRICE_REVISION_PROOF_IDENTITY_FIELDS}
        if existing_identity != identity or str(existing.get("proof_fingerprint") or "") != proof_fp:
            raise ValueError("adjusted-price revision proof immutable identity collision")
        return {**existing, "manifest_path": project_relative_display_path(manifest_path, project_root=root), "reused": True}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, payload)
    return {**payload, "manifest_path": project_relative_display_path(manifest_path, project_root=root), "reused": False}


def load_adjusted_price_revision_proof(project_root, *, proof_fingerprint: str, required: bool = False) -> dict[str, object] | None:
    root = Path(project_root).resolve()
    path = resolve_research_v2_adjusted_price_proof_manifest_path(root, proof_fingerprint)
    if not path.is_file():
        if required:
            raise FileNotFoundError("adjusted-price revision proof manifest 不存在")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("adjusted-price revision proof manifest 必須是 object")
    identity = {key: payload.get(key) for key in ADJUSTED_PRICE_REVISION_PROOF_IDENTITY_FIELDS}
    actual = adjusted_price_revision_proof_fingerprint(identity)
    if actual != str(proof_fingerprint).strip().lower() or str(payload.get("proof_fingerprint") or "") != actual:
        raise ValueError("adjusted-price revision proof fingerprint drift")
    validate_adjusted_price_revision_proof_payload(identity)
    return dict(payload)


__all__ = ["build_adjusted_price_revision_proof", "load_adjusted_price_revision_proof"]
