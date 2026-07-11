"""Canonical breakout-quality outer Selection/OOS and optional inner-validation split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from core.walk_forward_policy import (
    build_optimizer_runtime_policy,
    build_walk_forward_policy_effective_snapshot,
    load_walk_forward_policy,
)
from filters.breakout_quality.contract import (
    LABEL_PASS,
    LABEL_REJECT,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_OUT_OF_SCOPE,
    OUTER_SPLIT_SELECTION,
    OUTER_SPLIT_VALUES,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_IGNORE,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    SELECTION_ROLE_VALUES,
    TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
)

KEY_COLUMNS = ("ticker", "date", "high_len")


def _normalize_iso_date(value, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"breakout quality split 日期不可空白: {field_name}")
    try:
        return pd.Timestamp(text).normalize().strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"breakout quality split 日期格式錯誤: {field_name}={value!r}") from exc


def _date_from_policy(
    policy: Mapping[str, object],
    date_key: str,
    year_key: str,
    *,
    end_of_year: bool,
) -> str | None:
    raw_date = str(policy.get(date_key) or "").strip()
    if raw_date:
        return _normalize_iso_date(raw_date, field_name=date_key)
    raw_year = policy.get(year_key)
    if raw_year is None:
        return None
    year = int(raw_year)
    suffix = "12-31" if end_of_year else "01-01"
    return f"{year:04d}-{suffix}"


def compute_outer_policy_fingerprint(payload: Mapping[str, object]) -> str:
    canonical_payload = {
        key: value
        for key, value in dict(payload).items()
        if key != "policy_fingerprint_sha256"
    }
    text = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_breakout_quality_outer_policy(
    project_root: str | Path,
    *,
    source_data_end_date: str,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Resolve the same single-fold OOS policy used by the optimizer."""
    root = str(Path(project_root).resolve())
    source_end = _normalize_iso_date(
        source_data_end_date,
        field_name="source_data_end_date",
    )
    base_policy = load_walk_forward_policy(root, environ=environ)
    runtime_policy = build_optimizer_runtime_policy(
        base_policy,
        "oos",
        latest_data_date=source_end,
    )
    snapshot = build_walk_forward_policy_effective_snapshot(runtime_policy)

    selection_start = _date_from_policy(
        snapshot,
        "selection_start_date",
        "selection_start_year",
        end_of_year=False,
    )
    selection_end = _date_from_policy(
        snapshot,
        "search_train_end_date",
        "search_train_end_year",
        end_of_year=True,
    )
    oos_start = _date_from_policy(
        snapshot,
        "oos_start_date",
        "oos_start_year",
        end_of_year=False,
    )
    configured_oos_end = _date_from_policy(
        snapshot,
        "oos_end_date",
        "oos_end_year",
        end_of_year=True,
    )
    if selection_start is None or selection_end is None or oos_start is None:
        raise ValueError(
            "breakout quality OOS 訓練需要既有 walk_forward_policy 提供 "
            "selection start/end 與 oos start"
        )
    effective_oos_end = min(configured_oos_end, source_end) if configured_oos_end else source_end
    selection_start_ts = pd.Timestamp(selection_start)
    selection_end_ts = pd.Timestamp(selection_end)
    oos_start_ts = pd.Timestamp(oos_start)
    oos_end_ts = pd.Timestamp(effective_oos_end)
    if selection_end_ts < selection_start_ts:
        raise ValueError("breakout quality selection_end 不可早於 selection_start")
    if oos_start_ts <= selection_end_ts:
        raise ValueError("breakout quality oos_start 必須晚於 selection_end")
    if oos_end_ts < oos_start_ts:
        raise ValueError(
            "breakout quality source data尚未覆蓋 OOS: "
            f"oos={oos_start}~{effective_oos_end}, source_end={source_end}"
        )

    payload = {
        "policy_source": "core.walk_forward_policy",
        "model_mode": "oos",
        "evaluation_scope": str(snapshot.get("evaluation_scope") or "oos_single_fold"),
        "selection_start_date": selection_start,
        "selection_end_date": selection_end,
        "oos_start_date": oos_start,
        "configured_oos_end_date": configured_oos_end,
        "effective_oos_end_date": effective_oos_end,
        "source_data_end_date": source_end,
        "walk_forward_policy": snapshot,
    }
    payload["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(payload)
    return payload


def normalize_split_assignment_keys(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"breakout quality split assignment 缺少 key 欄位: {missing}")
    normalized = frame.copy()
    normalized["ticker"] = normalized["ticker"].astype(str)
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    normalized["high_len"] = pd.to_numeric(
        normalized["high_len"],
        errors="raise",
    ).astype(int)
    duplicated = normalized.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicated.any():
        raise ValueError(
            "breakout quality split assignment key ticker/date/high_len 必須唯一；"
            f"duplicate_rows={int(duplicated.sum())}"
        )
    return normalized


def validate_split_assignment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS):
        raise ValueError(
            "breakout quality split assignment columns 必須精確等於 "
            f"{list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS)}；actual={list(frame.columns)}"
        )
    normalized = normalize_split_assignment_keys(frame)
    normalized["outer_split"] = normalized["outer_split"].astype(str)
    normalized["selection_role"] = normalized["selection_role"].astype(str)
    unknown_outer = sorted(set(normalized["outer_split"]) - set(OUTER_SPLIT_VALUES))
    unknown_role = sorted(set(normalized["selection_role"]) - set(SELECTION_ROLE_VALUES))
    if unknown_outer:
        raise ValueError(
            f"breakout quality split assignment outer_split 含未知值: {unknown_outer}"
        )
    if unknown_role:
        raise ValueError(
            f"breakout quality split assignment selection_role 含未知值: {unknown_role}"
        )
    invalid_role = normalized[
        (normalized["outer_split"] != OUTER_SPLIT_SELECTION)
        & (normalized["selection_role"] != SELECTION_ROLE_NOT_APPLICABLE)
    ]
    if not invalid_role.empty:
        raise ValueError("Selection 外的 split row，selection_role 必須為 not_applicable")
    return normalized[list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS)]


def _event_group_keys(events: pd.DataFrame) -> pd.Series:
    return events["ticker"].astype(str) + "\x1f" + events["date"].astype(str)


def _resolve_inner_validation_start(
    *,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    inner_validation_months: int,
) -> pd.Timestamp:
    months = int(inner_validation_months)
    if months < 1:
        raise ValueError("inner_validation_months 必須 >= 1")
    validation_start = (
        selection_end + pd.Timedelta(days=1) - pd.DateOffset(months=months)
    ).normalize()
    if not selection_start < validation_start <= selection_end:
        raise ValueError(
            "inner validation 期間必須完整落在 Selection 內，且前方仍須保留 train period: "
            f"selection={selection_start.date()}~{selection_end.date()}, "
            f"months={months}, validation_start={validation_start.date()}"
        )
    return validation_start


def build_selection_oos_split_assignments(
    events: pd.DataFrame,
    labels: Iterable[int],
    *,
    outer_policy: Mapping[str, object],
    use_inner_validation: bool = False,
    inner_validation_months: int = 24,
    early_stopping_enabled: bool = False,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Build one canonical split for fixed-epoch or inner-validation/full-refit training."""
    validation_enabled = bool(use_inner_validation)
    if bool(early_stopping_enabled) and not validation_enabled:
        raise ValueError("未開啟 inner validation 時不可啟用 early stopping")
    required = {"ticker", "date", "high_len", "label_eval_end_date"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(
            f"breakout quality split 缺少欄位 {missing}；請用新版 build_dataset 重建資料"
        )
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    if labels_arr.size != len(events):
        raise ValueError(
            "breakout quality split labels/events 長度不一致: "
            f"{labels_arr.size} != {len(events)}"
        )

    keyed = normalize_split_assignment_keys(events[["ticker", "date", "high_len"]])
    event_dates = pd.to_datetime(events["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        events["label_eval_end_date"],
        errors="raise",
    ).dt.normalize()
    selection_start = pd.Timestamp(
        _normalize_iso_date(
            outer_policy.get("selection_start_date"),
            field_name="selection_start_date",
        )
    )
    selection_end = pd.Timestamp(
        _normalize_iso_date(
            outer_policy.get("selection_end_date"),
            field_name="selection_end_date",
        )
    )
    oos_start = pd.Timestamp(
        _normalize_iso_date(
            outer_policy.get("oos_start_date"),
            field_name="oos_start_date",
        )
    )
    oos_end = pd.Timestamp(
        _normalize_iso_date(
            outer_policy.get("effective_oos_end_date"),
            field_name="effective_oos_end_date",
        )
    )
    if not selection_start <= selection_end < oos_start <= oos_end:
        raise ValueError(
            "breakout quality outer policy 日期順序必須為 "
            "selection_start <= selection_end < oos_start <= oos_end"
        )

    validation_start = None
    if validation_enabled:
        validation_start = _resolve_inner_validation_start(
            selection_start=selection_start,
            selection_end=selection_end,
            inner_validation_months=int(inner_validation_months),
        )

    valid_label = np.isin(labels_arr, [LABEL_REJECT, LABEL_PASS])
    selection_mask = (
        (event_dates >= selection_start) & (event_dates <= selection_end)
    ).to_numpy(dtype=bool)
    oos_mask = (
        (event_dates >= oos_start) & (event_dates <= oos_end)
    ).to_numpy(dtype=bool)
    label_before_oos = (label_end_dates < oos_start).to_numpy(dtype=bool)
    label_within_oos = (label_end_dates <= oos_end).to_numpy(dtype=bool)

    final_refit_mask = selection_mask & valid_label & label_before_oos
    selection_oos_embargo_mask = selection_mask & valid_label & ~label_before_oos
    selection_ignore_mask = selection_mask & ~valid_label
    oos_evaluable_mask = oos_mask & valid_label & label_within_oos
    oos_label_after_end_mask = oos_mask & valid_label & ~label_within_oos

    inner_train_mask = final_refit_mask.copy()
    inner_validation_mask = np.zeros(len(events), dtype=bool)
    inner_embargo_mask = np.zeros(len(events), dtype=bool)
    if validation_start is not None:
        before_validation = (event_dates < validation_start).to_numpy(dtype=bool)
        at_or_after_validation = (event_dates >= validation_start).to_numpy(dtype=bool)
        label_before_validation = (
            label_end_dates < validation_start
        ).to_numpy(dtype=bool)
        inner_train_mask = (
            final_refit_mask & before_validation & label_before_validation
        )
        inner_embargo_mask = (
            final_refit_mask & before_validation & ~label_before_validation
        )
        inner_validation_mask = final_refit_mask & at_or_after_validation

    outer_values = np.full(len(events), OUTER_SPLIT_OUT_OF_SCOPE, dtype=object)
    outer_values[selection_mask] = OUTER_SPLIT_SELECTION
    outer_values[oos_mask] = OUTER_SPLIT_OOS
    selection_roles = np.full(
        len(events),
        SELECTION_ROLE_NOT_APPLICABLE,
        dtype=object,
    )
    selection_roles[selection_ignore_mask] = SELECTION_ROLE_IGNORE
    selection_roles[inner_train_mask] = SELECTION_ROLE_TRAIN
    selection_roles[inner_validation_mask] = SELECTION_ROLE_VALIDATION
    selection_roles[inner_embargo_mask] = SELECTION_ROLE_INNER_EMBARGO
    selection_roles[selection_oos_embargo_mask] = SELECTION_ROLE_EMBARGO

    assignments = keyed.copy()
    assignments["outer_split"] = outer_values.astype(str)
    assignments["selection_role"] = selection_roles.astype(str)
    assignments = validate_split_assignment_frame(
        assignments[list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS)]
    )

    inner_train_idx = np.flatnonzero(inner_train_mask).astype(np.int64)
    inner_validation_idx = np.flatnonzero(inner_validation_mask).astype(np.int64)
    final_refit_idx = np.flatnonzero(final_refit_mask).astype(np.int64)
    oos_idx = np.flatnonzero(oos_evaluable_mask).astype(np.int64)
    if inner_train_idx.size == 0 or final_refit_idx.size == 0 or oos_idx.size == 0:
        raise ValueError(
            "breakout quality split 產生空集合: "
            f"inner_train={inner_train_idx.size}, final_refit={final_refit_idx.size}, "
            f"oos={oos_idx.size}"
        )
    if validation_enabled and inner_validation_idx.size == 0:
        raise ValueError("已開啟 inner validation，但 validation split 為空")

    group_keys = _event_group_keys(events)
    inner_train_groups = set(group_keys.iloc[inner_train_idx])
    inner_validation_groups = set(group_keys.iloc[inner_validation_idx])
    final_refit_groups = set(group_keys.iloc[final_refit_idx])
    oos_groups = set(group_keys.iloc[oos_idx])
    inner_train_dates = set(event_dates.iloc[inner_train_idx])
    inner_validation_dates = set(event_dates.iloc[inner_validation_idx])
    final_refit_dates = set(event_dates.iloc[final_refit_idx])
    oos_dates = set(event_dates.iloc[oos_idx])

    def _range(indices: np.ndarray, column: str) -> dict[str, str | None]:
        if indices.size == 0:
            return {"start": None, "end": None}
        values = pd.to_datetime(events.iloc[indices][column], errors="raise")
        return {
            "start": str(values.min().date()),
            "end": str(values.max().date()),
        }

    outer_counts = {
        name: int((assignments["outer_split"] == name).sum())
        for name in OUTER_SPLIT_VALUES
    }
    selection_role_counts = {
        name: int((assignments["selection_role"] == name).sum())
        for name in SELECTION_ROLE_VALUES
    }
    training_mode = (
        TRAINING_MODE_INNER_VALIDATION_FULL_REFIT
        if validation_enabled
        else TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION
    )
    report = {
        "strategy": (
            "walk_forward_outer_selection_oos_inner_validation_full_refit"
            if validation_enabled
            else "walk_forward_outer_selection_oos_fixed_epoch_full_selection"
        ),
        "training_mode": training_mode,
        "policy_source": str(
            outer_policy.get("policy_source") or "core.walk_forward_policy"
        ),
        "policy_fingerprint_sha256": str(
            outer_policy.get("policy_fingerprint_sha256") or ""
        ),
        "selection_start_date": str(selection_start.date()),
        "selection_end_date": str(selection_end.date()),
        "inner_validation_start_date": (
            str(validation_start.date()) if validation_start is not None else None
        ),
        "inner_validation_months": (
            int(inner_validation_months) if validation_enabled else None
        ),
        "oos_start_date": str(oos_start.date()),
        "oos_end_date": str(oos_end.date()),
        "selection_train_row_count": int(inner_train_idx.size),
        "inner_validation_row_count": int(inner_validation_idx.size),
        "inner_train_validation_embargo_row_count": int(inner_embargo_mask.sum()),
        "final_refit_row_count": int(final_refit_idx.size),
        "selection_oos_embargo_row_count": int(selection_oos_embargo_mask.sum()),
        "embargo_dropped_from_final_refit_row_count": int(
            selection_oos_embargo_mask.sum()
        ),
        "oos_evaluable_row_count": int(oos_idx.size),
        "oos_label_after_end_row_count": int(oos_label_after_end_mask.sum()),
        "outer_split_counts": outer_counts,
        "selection_role_counts": selection_role_counts,
        "selection_train_group_count": int(len(inner_train_groups)),
        "inner_validation_group_count": int(len(inner_validation_groups)),
        "final_refit_group_count": int(len(final_refit_groups)),
        "oos_group_count": int(len(oos_groups)),
        "inner_train_validation_overlap_group_count": int(
            len(inner_train_groups.intersection(inner_validation_groups))
        ),
        "inner_train_validation_overlap_event_date_count": int(
            len(inner_train_dates.intersection(inner_validation_dates))
        ),
        "overlap_group_count": int(
            len(final_refit_groups.intersection(oos_groups))
        ),
        "overlap_event_date_count": int(
            len(final_refit_dates.intersection(oos_dates))
        ),
        "selection_train_date_range": _range(inner_train_idx, "date"),
        "selection_train_label_end_date_range": _range(
            inner_train_idx,
            "label_eval_end_date",
        ),
        "inner_validation_date_range": _range(inner_validation_idx, "date"),
        "inner_validation_label_end_date_range": _range(
            inner_validation_idx,
            "label_eval_end_date",
        ),
        "final_refit_date_range": _range(final_refit_idx, "date"),
        "final_refit_label_end_date_range": _range(
            final_refit_idx,
            "label_eval_end_date",
        ),
        "oos_date_range": _range(oos_idx, "date"),
        "oos_label_end_date_range": _range(oos_idx, "label_eval_end_date"),
        "training_uses_all_eligible_selection_rows": True,
        "final_refit_includes_inner_embargo_rows": bool(validation_enabled),
        "inner_validation_used": validation_enabled,
        "early_stopping_used": bool(early_stopping_enabled),
    }
    return (
        assignments,
        inner_train_idx,
        inner_validation_idx,
        final_refit_idx,
        oos_idx,
        report,
    )


__all__ = [
    "KEY_COLUMNS",
    "build_selection_oos_split_assignments",
    "compute_outer_policy_fingerprint",
    "normalize_split_assignment_keys",
    "resolve_breakout_quality_outer_policy",
    "validate_split_assignment_frame",
]
