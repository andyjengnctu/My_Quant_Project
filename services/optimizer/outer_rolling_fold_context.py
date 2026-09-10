from __future__ import annotations

import re

from services.optimizer.outer_rolling_formatting import (
    _canonical_date_text,
    _display_month_period,
    _infer_period_end,
    _infer_period_start,
    _oos_key_from_start_date,
    _period_label,
    _split_period_text,
    _timestamp_year_or_none,
)


def build_optimizer_seed_ensemble_fold_context(
    *,
    fold_idx: int,
    fold_count: int,
    selection_start_date: str,
    selection_end_date: str,
    oos_start_date: str,
    oos_end_date: str | None = None,
    selection_period: str | None = None,
    oos_period: str | None = None,
) -> dict:
    """Return the canonical seed-ensemble fold context for rolling and non-rolling.

    Non-rolling must use this with ``fold_count=1``.  Rolling uses the exact same
    schema with ``fold_count>1``.  Downstream progress events, result rows, and
    renderers rely on ``oos_year`` being the sortable numeric OOS key; period text
    belongs in ``oos_period`` only.
    """
    sel_start = _canonical_date_text(selection_start_date)
    sel_end = _canonical_date_text(selection_end_date)
    oos_start = _canonical_date_text(oos_start_date)
    oos_end = _canonical_date_text(oos_end_date or "latest")
    sel_period = str(selection_period or "").strip()
    if not sel_period and sel_start and sel_end:
        sel_period = f"{sel_start}~{sel_end}"
    oos_period_text = str(oos_period or "").strip()
    if not oos_period_text and oos_start:
        oos_period_text = f"{oos_start}~{oos_end or 'latest'}"
    oos_key = _oos_key_from_start_date(oos_start)
    return {
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "fold": f"{int(fold_idx)}/{int(fold_count)}",
        "selection_start_date": sel_start,
        "selection_end_date": sel_end,
        "selection_start": sel_start,
        "selection_end": sel_end,
        "selection_period": sel_period,
        "oos_start_date": oos_start,
        "oos_end_date": oos_end,
        "oos_year": int(oos_key),
        "oos_period": oos_period_text,
    }


def normalize_optimizer_seed_ensemble_fold_row(row: dict, *, context: dict | None = None) -> dict:
    """Normalize a rolling/non-rolling seed-ensemble result row before display.

    This is intentionally the only row schema adapter for live tables and final
    tables.  It prevents period strings such as ``2023-01-01~latest`` from being
    stored in ``oos_year`` while keeping the human-readable range in
    ``oos_period``.
    """
    source = dict(row or {})
    ctx = dict(context or {})
    merged = dict(ctx)
    merged.update(source)

    selection_start = _infer_period_start(merged, period_key="selection_period", start_key="selection_start_date") or str(ctx.get("selection_start") or "")
    selection_end = _infer_period_end(merged, period_key="selection_period", end_key="selection_end_date") or str(ctx.get("selection_end") or "")
    oos_start = _infer_period_start(merged, period_key="oos_period", start_key="oos_start_date")
    oos_end = _infer_period_end(merged, period_key="oos_period", end_key="oos_end_date")

    raw_oos_year = merged.get("oos_year")
    if not oos_start and isinstance(raw_oos_year, str) and "~" in raw_oos_year:
        start_from_year, end_from_year = _split_period_text(raw_oos_year)
        oos_start = _canonical_date_text(start_from_year)
        if not oos_end:
            oos_end = _canonical_date_text(end_from_year)
    if not oos_start and raw_oos_year not in (None, ""):
        raw = str(raw_oos_year).strip()
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 6:
            oos_start = f"{digits[:4]}-{digits[4:6]}-01"
        elif len(digits) >= 4:
            oos_start = f"{digits[:4]}-01-01"

    if not oos_end:
        oos_end = "latest" if str(merged.get("oos_period") or raw_oos_year or "").lower().endswith("latest") else ""
    fold_text = str(merged.get("fold") or "1/1")
    try:
        fold_idx = int(merged.get("fold_idx") or fold_text.split("/", 1)[0] or 1)
    except (TypeError, ValueError):
        fold_idx = 1
    try:
        fold_count = int(
            merged.get("fold_count")
            or (fold_text.split("/", 1)[1] if "/" in fold_text else ctx.get("fold_count", 1))
            or 1
        )
    except (TypeError, ValueError):
        fold_count = max(1, int(ctx.get("fold_count", 1) or 1))
    canonical = build_optimizer_seed_ensemble_fold_context(
        fold_idx=fold_idx,
        fold_count=fold_count,
        selection_start_date=selection_start,
        selection_end_date=selection_end,
        oos_start_date=oos_start,
        oos_end_date=oos_end or None,
        selection_period=merged.get("selection_period"),
        oos_period=merged.get("oos_period") if str(merged.get("oos_period") or "").strip() else None,
    )
    if str(canonical.get("oos_period") or "").strip() in {"", "latest"} and oos_start:
        canonical["oos_period"] = f"{oos_start}~{oos_end or 'latest'}"
    source.update(canonical)
    selection_start_year = _timestamp_year_or_none(source.get("selection_start_date"))
    if selection_start_year is not None:
        source["selection_start_year"] = selection_start_year
    selection_end_year = _timestamp_year_or_none(source.get("selection_end_date"))
    if selection_end_year is not None:
        source["selection_end_year"] = selection_end_year
    return source


def normalize_optimizer_seed_ensemble_fold_rows(rows: list[dict]) -> list[dict]:
    return [normalize_optimizer_seed_ensemble_fold_row(row) for row in list(rows or [])]


def optimizer_seed_ensemble_row_sort_key(row: dict) -> tuple[int, int]:
    normalized = normalize_optimizer_seed_ensemble_fold_row(row)
    return (int(normalized.get("oos_year", 0) or 0), int(normalized.get("fold_idx", 0) or 0))


def _fold_label(fold: dict) -> str:
    return str(fold.get("oos_period") or _period_label(fold.get("oos_start_date"), fold.get("oos_end_date")))


def _fold_selection_label(fold: dict) -> str:
    return str(fold.get("selection_period") or _period_label(fold.get("selection_start_date"), fold.get("selection_end_date")))


def _fold_label_display(fold: dict) -> str:
    return _display_month_period(_fold_label(fold))


def _fold_selection_label_display(fold: dict) -> str:
    return _display_month_period(_fold_selection_label(fold))

