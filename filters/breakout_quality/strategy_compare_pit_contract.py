"""Canonical Strategy Compare Selection-PIT readiness contract.

This module owns the legality checks shared by preparation/status and training
completion.  Callers may change artifact namespace or seed, but must not copy the
PIT identity/period validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_ranking_contract,
)


def _normalize_contract_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Strategy Compare PIT artifact缺少日期")
    return text[:10]


def _is_auto_contract_date(value: Any) -> bool:
    return str(value or "").strip().lower() == "auto"


def validate_selection_pit_score_period(
    manifest: dict[str, Any],
    *,
    comparison_start: str | None,
    comparison_end: str | None,
) -> tuple[str, str]:
    """Validate configured/dynamic comparison bounds against persisted PIT truth."""

    period = dict(manifest.get("score_period") or {})
    actual_start = _normalize_contract_date(period.get("start"))
    actual_end = _normalize_contract_date(period.get("end"))

    if comparison_start not in (None, ""):
        if _is_auto_contract_date(comparison_start):
            resolution = dict(manifest.get("score_start_resolution") or {})
            resolved_start = _normalize_contract_date(
                resolution.get("resolved_score_start")
            )
            if (
                str(resolution.get("mode") or "") != "auto_earliest_legal"
                or actual_start != resolved_start
            ):
                raise ValueError(
                    "Strategy Compare PIT自動比較起始日解析不一致: "
                    f"actual={actual_start}, resolved={resolved_start}, "
                    f"mode={resolution.get('mode')!r}"
                )
        else:
            required_start = _normalize_contract_date(comparison_start)
            if actual_start != required_start:
                raise ValueError(
                    "Strategy Compare PIT比較起始日不一致: "
                    f"expected={required_start}, actual={actual_start}"
                )

    if comparison_end not in (None, ""):
        if _is_auto_contract_date(comparison_end):
            evaluation_policy = dict(manifest.get("evaluation_policy") or {})
            available_history = dict(manifest.get("available_history_period") or {})
            available_end = _normalize_contract_date(available_history.get("end"))
            resolution_mode = str(evaluation_policy.get("score_end_resolution") or "")
            if resolution_mode != "auto_available_end" or actual_end != available_end:
                raise ValueError(
                    "Strategy Compare PIT自動比較結束日解析不一致: "
                    f"actual={actual_end}, available={available_end}, "
                    f"mode={resolution_mode!r}"
                )
        else:
            required_end = _normalize_contract_date(comparison_end)
            if actual_end != required_end:
                raise ValueError(
                    "Strategy Compare PIT比較結束日不一致: "
                    f"expected={required_end}, actual={actual_end}"
                )
    return actual_start, actual_end


def load_validated_selection_pit_strategy_compare_contract(
    *,
    root: Path,
    source: Any,
    workflow: Any,
    seed: int,
    point_in_time_dir_override: Path | None = None,
    comparison_start: str | None = None,
    comparison_end: str | None = None,
):
    """Load one PIT bundle using the single current Strategy Compare READY contract."""

    contract = load_selection_point_in_time_ranking_contract(
        str(Path(root).resolve()),
        str(source.filter_id),
        str(source.model_architecture),
        str(source.experiment_profile),
        require_model_validation_pass=False,
        point_in_time_dir_override=point_in_time_dir_override,
    )
    if int(contract.seed) != int(seed):
        raise ValueError(
            "Strategy Compare PIT seed不一致: "
            f"expected={int(seed)}, actual={int(contract.seed)}"
        )
    manifest = dict(contract.manifest or {})
    expected_fold_months = int(
        source.point_in_time_fold_months
        if source.point_in_time_fold_months is not None
        else workflow.point_in_time_fold_months
    )
    if int(manifest.get("fold_months", -1) or -1) != expected_fold_months:
        raise ValueError("Strategy Compare PIT fold_months不一致")
    expected_single_block = bool(source.point_in_time_single_score_block)
    if bool(manifest.get("single_score_block", False)) != expected_single_block:
        raise ValueError("Strategy Compare PIT single_score_block不一致")
    expected_anchor = (
        None
        if source.point_in_time_fold_anchor_date in (None, "")
        else str(source.point_in_time_fold_anchor_date)
    )
    actual_anchor = str(manifest.get("fold_anchor_date") or "").strip() or None
    if expected_anchor is not None and actual_anchor != expected_anchor:
        raise ValueError("Strategy Compare PIT fold_anchor_date不一致")

    expected_start = comparison_start
    if expected_start in (None, ""):
        expected_start = source.point_in_time_score_start_date
    expected_end = comparison_end
    if expected_end in (None, ""):
        expected_end = source.point_in_time_score_end_date
    validate_selection_pit_score_period(
        manifest,
        comparison_start=(
            None if expected_start in (None, "") else str(expected_start)
        ),
        comparison_end=(None if expected_end in (None, "") else str(expected_end)),
    )

    folds = [dict(item or {}) for item in list(manifest.get("folds") or [])]
    if not folds:
        raise ValueError("Strategy Compare PIT manifest沒有folds")
    if any(int(item.get("selected_epoch", 0) or 0) < 1 for item in folds):
        raise ValueError("Strategy Compare PIT fold selected_epoch不合法")
    return contract


__all__ = [
    "load_validated_selection_pit_strategy_compare_contract",
    "validate_selection_pit_score_period",
]
