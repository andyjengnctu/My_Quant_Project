"""Canonical PIT cohort partitions and earliest legal score-date resolution.

Both continuous and binary PIT must use identical maturity, fold and sample
eligibility semantics.  This module owns their shared predicates and has no
optimizer, score producer, or model FIT dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.breakout_quality_registry import (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT
from filters.breakout_quality.point_in_time_schedule import build_point_in_time_fold_periods

AUTO_SCORE_START_VALUE = "auto"
_build_fold_periods = build_point_in_time_fold_periods

# Persisted evaluation-only Fold sidecar contract. FIT/epoch validation uses
# the supervised ids and remains unchanged. This scope is mandatory only for
# real execution-path Targets whose training labels can be NULL on a price-
# mature stock-day; an older target-gated sidecar is not a reusable SOP source.
STANDARD_PRICE_VALIDATION_SCORE_SCOPE = "independent_price_40d_matured"


def required_standard_validation_score_scope(experiment_profile: str) -> str | None:
    """Return a capability-derived score-only scope, never a model-ID switch."""
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH,
    )
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe

    recipe = get_continuous_ranker_execution_recipe(str(experiment_profile))
    return (
        STANDARD_PRICE_VALIDATION_SCORE_SCOPE
        if recipe.target_policy.materialization_mode
        == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH
        else None
    )


def validate_standard_validation_score_scope(manifest, *, experiment_profile: str) -> None:
    """Reject score-only evaluation sidecars predating the independent price scope.

    Check the published Fold *records*, not just the aggregate score hash.
    Missing coverage must trigger canonical checkpoint-only scoring, not an
    Audit-only refresh that silently republishes an incomparable validation.
    """
    required = required_standard_validation_score_scope(str(experiment_profile))
    if required is None:
        return
    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Rolling PIT independent-price validation missing Fold records")
    for fold in folds:
        if not isinstance(fold, dict) or fold.get("validation_score_scope") != required:
            fold_id = str(fold.get("fold_id") or "-") if isinstance(fold, dict) else "-"
            raise ValueError(
                f"Rolling PIT {fold_id} validation score scope stale/missing; "
                "score-only refresh from the verified fitted checkpoint is required; "
                "Audit-only refresh cannot repair this source"
            )

def build_training_scope_mask(bundle: ContinuousRankerDataBundle) -> np.ndarray:
    """Return the canonical target-eligible mask for this sample universe."""

    target_valid = np.asarray(bundle.target_valid, dtype=bool)
    if bundle.profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        return target_valid.copy()
    labels = bundle.group_table["label"].to_numpy(dtype=np.int64)
    if bundle.profile.training_label_scope == TRAINING_LABEL_SCOPE_PASS_ONLY:
        return target_valid & (labels == LABEL_PASS)
    if bundle.profile.training_label_scope == TRAINING_LABEL_SCOPE_ALL:
        return target_valid & np.isin(labels, [LABEL_REJECT, LABEL_PASS])
    raise ValueError(
        "不支援的continuous ranker training scope: "
        f"{bundle.profile.training_label_scope}"
    )

def _iso_timestamp(value: Any, *, field_name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(str(value)).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必須是合法日期: {value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"{field_name} 不可為NaT")
    return result


def _group_event_count(event_group_index: np.ndarray, group_ids: np.ndarray) -> int:
    return int(np.isin(event_group_index, np.asarray(group_ids, dtype=np.int64)).sum())


def _minimum_count_failures(settings, ids: dict[str, Any]) -> list[str]:
    required = {
        "inner_train": (len(ids["train_ids"]), settings.point_in_time_min_train_groups),
        "validation": (
            len(ids["validation_ids"]),
            settings.point_in_time_min_validation_groups,
        ),
        "score": (len(ids["score_ids"]), settings.point_in_time_min_score_groups),
    }
    return [
        f"{name}={actual}<{minimum}"
        for name, (actual, minimum) in required.items()
        if int(actual) < int(minimum)
    ]


def _candidate_month_starts(
    selection_start: pd.Timestamp, selection_end: pd.Timestamp
) -> list[pd.Timestamp]:
    first = selection_start.normalize()
    candidates = [first]
    next_month = (first + pd.offsets.MonthBegin(1)).normalize()
    while next_month <= selection_end:
        if next_month != candidates[-1]:
            candidates.append(next_month)
        next_month = (next_month + pd.offsets.MonthBegin(1)).normalize()
    return candidates


def _resolve_score_start(
    raw_value: Any,
    *,
    bundle,
    settings,
    selection_start: pd.Timestamp,
    score_end: pd.Timestamp,
    fold_months: int,
    validation_months: int,
    fold_anchor: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, dict[str, Any]]:
    raw_text = str(raw_value or "").strip().lower()
    if raw_text != AUTO_SCORE_START_VALUE:
        configured = _iso_timestamp(raw_value, field_name="score_start_date")
        return configured, {
            "mode": "configured",
            "configured_value": str(raw_value),
            "resolved_score_start": str(configured.date()),
        }

    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    scoped_target = np.asarray(build_training_scope_mask(bundle), dtype=bool)

    # ``auto`` may have to reject many early calendar months before enough legal
    # target history exists (MR-13I/J start only when historical Min ROOS risk
    # calibration is available).  Rebuilding four full-size masks through
    # ``_fold_group_ids`` for every rejected month made the CLI look hung on the
    # ~1M-row daily universe.  Precompute the immutable date arrays once and count
    # the first-fold partitions with search/small-window comparisons.  The formulas
    # below are exactly the same strict date/cutoff predicates as _fold_group_ids.
    all_date_ns = group_dates.to_numpy(dtype="datetime64[ns]").view("i8")
    sorted_all_date_ns = np.sort(all_date_ns)
    scoped_date_ns = all_date_ns[scoped_target]
    scoped_label_ns = (
        label_end_dates.to_numpy(dtype="datetime64[ns]").view("i8")[scoped_target]
    )
    nat_ns = np.datetime64("NaT", "ns").view("i8")
    valid_label_end = scoped_label_ns != nat_ns
    ready_ns = np.maximum(scoped_date_ns[valid_label_end], scoped_label_ns[valid_label_end])
    sorted_ready_ns = np.sort(ready_ns)
    scoped_order = np.argsort(scoped_date_ns, kind="mergesort")
    sorted_scoped_date_ns = scoped_date_ns[scoped_order]
    sorted_scoped_label_ns = scoped_label_ns[scoped_order]

    def first_fold_counts(candidate: pd.Timestamp) -> dict[str, int]:
        fold = _build_fold_periods(
            candidate,
            score_end,
            fold_months=int(fold_months),
            fold_anchor=fold_anchor,
        )[0]
        score_start_ns = np.int64(pd.Timestamp(candidate).value)
        score_end_ns = np.int64(pd.Timestamp(fold["score_end"]).value)
        validation_start = (
            pd.Timestamp(candidate) - pd.DateOffset(months=int(validation_months))
        ).normalize()
        validation_start_ns = np.int64(validation_start.value)

        train_count = int(
            np.searchsorted(sorted_ready_ns, validation_start_ns, side="left")
        )
        validation_left = int(
            np.searchsorted(
                sorted_scoped_date_ns, validation_start_ns, side="left"
            )
        )
        validation_right = int(
            np.searchsorted(sorted_scoped_date_ns, score_start_ns, side="left")
        )
        validation_label_ns = sorted_scoped_label_ns[
            validation_left:validation_right
        ]
        validation_count = int(
            (
                (validation_label_ns != nat_ns)
                & (validation_label_ns < score_start_ns)
            ).sum()
        )
        score_count = int(
            np.searchsorted(sorted_all_date_ns, score_end_ns, side="right")
            - np.searchsorted(sorted_all_date_ns, score_start_ns, side="left")
        )
        return {
            "inner_train": train_count,
            "validation": validation_count,
            "score": score_count,
        }

    rejected: list[dict[str, Any]] = []
    for candidate in _candidate_month_starts(selection_start, score_end):
        counts = first_fold_counts(candidate)
        required = {
            "inner_train": (
                counts["inner_train"],
                settings.point_in_time_min_train_groups,
            ),
            "validation": (
                counts["validation"],
                settings.point_in_time_min_validation_groups,
            ),
            "score": (counts["score"], settings.point_in_time_min_score_groups),
        }
        failures = [
            f"{name}={actual}<{minimum}"
            for name, (actual, minimum) in required.items()
            if int(actual) < int(minimum)
        ]
        if not failures:
            scoped_dates = group_dates[scoped_target]
            scoped_label_ends = label_end_dates[scoped_target]
            return candidate, {
                "mode": "auto_earliest_legal",
                "configured_value": AUTO_SCORE_START_VALUE,
                "resolved_score_start": str(candidate.date()),
                "selection_start": str(selection_start.date()),
                "selection_end": str(score_end.date()),
                "earliest_dataset_group_date": str(group_dates.min().date()),
                "earliest_scoped_target_group_date": (
                    str(scoped_dates.min().date()) if len(scoped_dates) else None
                ),
                "earliest_scoped_label_completion_date": (
                    str(scoped_label_ends.min().date()) if len(scoped_label_ends) else None
                ),
                "candidate_months_checked": int(len(rejected) + 1),
                "first_fold_group_counts": counts,
                "rejected_candidate_count": int(len(rejected)),
                "recent_rejected_candidates": rejected[-6:],
            }
        rejected.append(
            {
                "score_start": str(candidate.date()),
                "failures": failures,
            }
        )
    raise ValueError(
        "無法在可用歷史內找到符合目前PIT最小group契約的score start；"
        f"available={selection_start.date()}~{score_end.date()}, "
        f"last_failures={rejected[-1] if rejected else None}"
    )


def standard_price_validation_group_ids(bundle, *, validation_start, score_start) -> np.ndarray | None:
    """Evaluation-only 40D price eligibility, independent of supervised Target.

    ``None`` means this source uses the historical supervised-validation SOP
    population. The caller must never use these IDs for FIT/epoch selection.
    """
    if "standard_common_40d_price_end_date" not in bundle.group_table.columns:
        return None
    dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    price_end = pd.to_datetime(
        bundle.group_table["standard_common_40d_price_end_date"], errors="raise"
    ).dt.normalize()
    scope = (
        (dates >= pd.Timestamp(validation_start))
        & (dates < pd.Timestamp(score_start))
        & price_end.notna()
        & (price_end < pd.Timestamp(score_start))
    )
    return np.flatnonzero(scope.to_numpy(dtype=bool)).astype(np.int64)


def validate_standard_validation_sidecar_membership(
    expected_ids: np.ndarray, observed_ids: np.ndarray, *, fold_id: str,
) -> None:
    """Ensure signed Validation SCORE rows equal the canonical price-only cohort.

    An incomplete sidecar cannot be repaired by an Audit-only report refresh;
    the verified per-fold FIT must produce the missing PIT scores first.
    """
    expected = np.asarray(expected_ids, dtype=np.int64)
    observed = np.asarray(observed_ids, dtype=np.int64)
    if not np.array_equal(observed, expected):
        expected_set = set(expected)
        observed_set = set(observed)
        raise ValueError(
            f"Rolling Standard SOP {fold_id} validation price cohort differs "
            f"from the canonical score-only source: "
            f"expected={len(expected)}, scored={len(observed)}, "
            f"missing={len(expected_set - observed_set)}, "
            f"extra={len(observed_set - expected_set)}; "
            "re-score from the original verified FIT, never fill missing scores"
        )


def _fold_group_ids(
    bundle,
    fold: dict[str, Any],
    *,
    validation_months: int,
    train_window_months: int | None = None,
) -> dict[str, Any]:
    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    score_start = pd.Timestamp(fold["score_start"])
    score_end = pd.Timestamp(fold["score_end"])
    validation_start = (score_start - pd.DateOffset(months=int(validation_months))).normalize()
    history_start = (
        None
        if train_window_months is None
        else (score_start - pd.DateOffset(months=int(train_window_months))).normalize()
    )

    scoped_target = build_training_scope_mask(bundle)
    history_mask = np.ones(len(group_dates), dtype=bool)
    if history_start is not None:
        history_mask &= (group_dates >= history_start).to_numpy(dtype=bool)

    train_mask = (
        scoped_target
        & history_mask
        & (group_dates < validation_start).to_numpy(dtype=bool)
        & (label_end_dates < validation_start).to_numpy(dtype=bool)
    )
    validation_mask = (
        scoped_target
        & history_mask
        & (group_dates >= validation_start).to_numpy(dtype=bool)
        & (group_dates < score_start).to_numpy(dtype=bool)
        & (label_end_dates < score_start).to_numpy(dtype=bool)
    )
    final_mask = (
        scoped_target
        & history_mask
        & (group_dates < score_start).to_numpy(dtype=bool)
        & (label_end_dates < score_start).to_numpy(dtype=bool)
    )
    score_mask = (
        (group_dates >= score_start).to_numpy(dtype=bool)
        & (group_dates <= score_end).to_numpy(dtype=bool)
    )
    ids = {
        "train_ids": np.flatnonzero(train_mask).astype(np.int64),
        "validation_ids": np.flatnonzero(validation_mask).astype(np.int64),
        "final_ids": np.flatnonzero(final_mask).astype(np.int64),
        "score_ids": np.flatnonzero(score_mask).astype(np.int64),
        "validation_start": validation_start,
        "history_start": history_start,
    }
    # Evaluate price-mature rows independent of legal FIT/epoch supervision.
    evaluation_validation_ids = standard_price_validation_group_ids(
        bundle, validation_start=validation_start, score_start=score_start,
    )
    if evaluation_validation_ids is not None:
        ids["standard_validation_ids"] = evaluation_validation_ids
    if np.intersect1d(ids["train_ids"], ids["validation_ids"]).size:
        raise ValueError(f"{fold['fold_id']} train/validation group重疊")
    if np.intersect1d(ids["final_ids"], ids["score_ids"]).size:
        raise ValueError(f"{fold['fold_id']} final train/score group重疊")
    if len(ids["final_ids"]):
        cutoff = label_end_dates.iloc[ids["final_ids"]].max()
        if not cutoff < score_start:
            raise ValueError(
                f"{fold['fold_id']} model information cutoff未早於score start: "
                f"cutoff={cutoff.date()}, score_start={score_start.date()}"
            )
    return ids
