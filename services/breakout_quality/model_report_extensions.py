"""Canonical model-specific report-evidence composition for continuous DL models.

This module owns how persisted scientific evidence is normalized into reusable
Model-specific Extension payloads.  It does not render Markdown/console output and
never branches on MR identity.  Head applicability comes from the canonical
ContinuousRankerScoreOutputPolicy when available; evidence values remain the
persisted validation metrics produced by training/evaluation services.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.breakout_quality_runtime import get_continuous_ranker_training_policy
from core.research_report_contract import MODEL_HEAD_LEARNABILITY_SEMANTICS


# Evidence-source normalization is capability-based rather than model-ID based.
# Each entry maps one persisted evaluation family to semantic head IDs and the
# metric node inside that family.  New comparable head semantics should be added
# here once, alongside the canonical display registry in research_report_contract.
_HEAD_EVIDENCE_SOURCES: tuple[
    tuple[str, tuple[tuple[str, str], ...]], ...
] = (
    (
        "conditional_mfe_safety_evaluation",
        (("primary_mfe", "primary_mfe"), ("conditional_safety", "conditional_safety")),
    ),
    (
        "reverse_conditional_mfe_evaluation",
        (("raw_safety", "raw_safety"), ("conditional_mfe", "conditional_mfe")),
    ),
    (
        "safety_primary_evaluation",
        (("raw_safety", "raw_safety"), ("primary_target", "primary_target")),
    ),
    (
        "safety_raw_mfe_joint_min_evaluation",
        (("raw_safety", "raw_safety"), ("raw_mfe", "raw_mfe")),
    ),
    (
        "safety_raw_mfe_hmhs_evaluation",
        (("raw_safety", "raw_safety"), ("raw_mfe", "raw_mfe")),
    ),
    (
        "safety_raw_mfe_evaluation",
        (("raw_safety", "raw_safety"), ("raw_mfe", "raw_mfe")),
    ),
    (
        "hs_conditional_mfe_evaluation",
        (("raw_safety", "raw_safety"), ("conditional_mfe", "conditional_mfe_true_hs")),
    ),
    (
        "hs_priority_mfe_evaluation",
        (("raw_safety", "raw_safety"), ("hs_priority_mfe", "hs_priority_mfe")),
    ),
)

_HEAD_LABEL_TO_SEMANTIC = {
    "Raw Safety": "raw_safety",
    "Raw MFE": "raw_mfe",
    "Conditional MFE": "conditional_mfe",
    "Primary MFE": "primary_mfe",
    "Conditional Safety": "conditional_safety",
    "Economic Target": "primary_target",
    "HS-Priority MFE": "hs_priority_mfe",
}

_SCOPE_KEYS = ("oos", "breakout_candidate_oos")


def _scope_label(scope_key: str, *, oos_scope_label: str) -> str:
    if scope_key == "oos":
        return str(oos_scope_label)
    if scope_key == "breakout_candidate_oos":
        return "Breakout slice"
    raise KeyError(scope_key)


def _training_objective(payload: Mapping[str, Any]) -> str | None:
    source = dict(payload or {})
    training = dict(source.get("training") or {})
    objective = str(training.get("objective") or "").strip()
    if objective:
        return objective
    standard = dict(source.get("standard_model_sop") or {})
    standard_training = dict(standard.get("training") or {})
    objective = str(standard_training.get("objective") or "").strip()
    return objective or None


def expected_multi_head_semantics(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return semantic output-head IDs from the canonical training composition."""

    objective = _training_objective(payload)
    if not objective:
        return ()
    try:
        policy = get_continuous_ranker_training_policy(objective)
    except ValueError:
        return ()
    output = policy.score_output_policy
    if output is None or len(output.head_names) < 2:
        return ()
    supported = {item.semantic_id for item in MODEL_HEAD_LEARNABILITY_SEMANTICS}
    return tuple(str(name) for name in output.head_names if str(name) in supported)


def _metric_triplet(metric: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    row = dict(metric or {})
    return (
        row.get("mean_daily_spearman"),
        row.get("global_spearman_vs_raw_target"),
        row.get("pairwise_concordance"),
    )


def _head_metric_index(
    payload: Mapping[str, Any],
    *,
    oos_scope_label: str,
) -> dict[tuple[str, str], tuple[Any, Any, Any]]:
    source = dict(payload or {})
    result: dict[tuple[str, str], tuple[Any, Any, Any]] = {}

    # Current canonical evaluation payloads.
    for payload_key, head_sources in _HEAD_EVIDENCE_SOURCES:
        evaluation = dict(source.get(payload_key) or {})
        if not evaluation:
            continue
        for scope_key in _SCOPE_KEYS:
            scope = dict(evaluation.get(scope_key) or {})
            if not scope:
                continue
            label = _scope_label(scope_key, oos_scope_label=oos_scope_label)
            for semantic_id, metric_key in head_sources:
                metric = dict(scope.get(metric_key) or {})
                if not metric:
                    continue
                identity = (label, semantic_id)
                values = _metric_triplet(metric)
                if identity in result and result[identity] != values:
                    raise ValueError(
                        "Multi-head Learnability duplicate evidence conflict: "
                        f"{identity}"
                    )
                result[identity] = values

    # B374 compatibility: aggregated robustness may persist the old long-form
    # Standard head rows.  Normalize them into semantic IDs without retaining the
    # retired Standard-SOP presentation.
    persisted = source.get("standard_head_learnability_rows")
    if isinstance(persisted, list):
        for raw_row in persisted:
            row = dict(raw_row or {})
            split = str(row.get("split") or "").strip()
            semantic_id = _HEAD_LABEL_TO_SEMANTIC.get(str(row.get("head") or "").strip())
            if split not in {oos_scope_label, "Forward OOS", "Rolling OOS", "Breakout slice"}:
                continue
            if semantic_id is None:
                continue
            normalized_split = (
                oos_scope_label if split in {"Forward OOS", "Rolling OOS", oos_scope_label}
                else "Breakout slice"
            )
            identity = (normalized_split, semantic_id)
            values = (
                row.get("mean_daily_spearman"),
                row.get("global_spearman_vs_raw_target"),
                row.get("pairwise_concordance"),
            )
            if identity in result and result[identity] != values:
                raise ValueError(
                    "Multi-head Learnability legacy evidence conflict: "
                    f"{identity}"
                )
            result[identity] = values

    # Pre-B374 compatibility: old robustness comparison extension used the same
    # long-form Split/Head rows.
    for raw_extension in list(source.get("comparison_extensions") or []):
        extension = dict(raw_extension or {})
        if str(extension.get("id") or "") != "multi_head_learnability":
            continue
        for raw_row in list(extension.get("rows") or []):
            row = dict(raw_row or {})
            semantic_id = _HEAD_LABEL_TO_SEMANTIC.get(str(row.get("head") or "").strip())
            split = str(row.get("split") or "").strip()
            if semantic_id is None or not split:
                continue
            normalized_split = (
                oos_scope_label if split in {"Forward OOS", "Rolling OOS", oos_scope_label, "oos"}
                else "Breakout slice" if split in {"Breakout slice", "breakout_candidate_oos"}
                else split
            )
            if normalized_split not in {oos_scope_label, "Breakout slice"}:
                continue
            identity = (normalized_split, semantic_id)
            values = (
                row.get("mean_daily_spearman"),
                row.get("global_spearman_vs_raw_target"),
                row.get("pairwise_concordance"),
            )
            if identity in result and result[identity] != values:
                raise ValueError(
                    "Multi-head Learnability pre-B374 evidence conflict: "
                    f"{identity}"
                )
            result[identity] = values
    return result


def build_multi_head_learnability_extension(
    payload: Mapping[str, Any],
    *,
    oos_scope_label: str,
) -> dict[str, Any] | None:
    """Normalize semantic heads into one wide row per OOS/Breakout scope."""

    source = dict(payload or {})
    expected = expected_multi_head_semantics(source)
    index = _head_metric_index(source, oos_scope_label=oos_scope_label)
    observed = {semantic_id for _scope, semantic_id in index}
    if len(expected) < 2 and len(observed) < 2:
        return None

    semantic_ids = tuple(item.semantic_id for item in MODEL_HEAD_LEARNABILITY_SEMANTICS)
    rows: list[dict[str, Any]] = []
    for split in (oos_scope_label, "Breakout slice"):
        row: dict[str, Any] = {"split": split}
        for semantic in MODEL_HEAD_LEARNABILITY_SEMANTICS:
            daily_key, global_key, pair_key = semantic.metric_keys
            values = index.get((split, semantic.semantic_id))
            applicable = semantic.semantic_id in set(expected) or semantic.semantic_id in observed
            if values is None:
                sentinel = "MISSING" if applicable else None
                row[daily_key] = sentinel
                row[global_key] = sentinel
                row[pair_key] = sentinel
            else:
                row[daily_key], row[global_key], row[pair_key] = values
        rows.append(row)

    return {
        "id": "multi_head_learnability",
        "required_heads": list(expected),
        "rows": rows,
    }


def build_hs_conditional_quality_extension(
    payload: Mapping[str, Any],
    *,
    oos_scope_label: str,
) -> dict[str, Any] | None:
    """Build the compact HS qualification / Conditional-MFE decision surface."""

    hs_eval = dict(dict(payload or {}).get("hs_conditional_mfe_evaluation") or {})
    if not hs_eval:
        return None
    rows: list[dict[str, Any]] = []
    for scope_key in _SCOPE_KEYS:
        scope = dict(hs_eval.get(scope_key) or {})
        if not scope:
            continue
        gate = dict(scope.get("lexicographic_model_gate") or {})
        boundary = dict(scope.get("hs_qualification_boundary") or {})
        p45 = dict(boundary.get("p45_p55") or {})
        rows.append({
            "split": _scope_label(scope_key, oos_scope_label=oos_scope_label),
            "pred_hs_true_ls_pct": gate.get("predicted_hs_true_ls_pct"),
            "true_hs_recall_pct": gate.get("true_hs_recall_pct"),
            "p45_p55_pair": p45.get("pairwise_concordance"),
            "actual_hmhs_pct": gate.get("selected_hmhs_pct"),
            "actual_high_mfe_pct": gate.get("selected_high_mfe_pct"),
            "actual_mean_mfe_r": gate.get("selected_mean_favorable_r"),
            "mean_mfe_gap_r": gate.get("mean_favorable_r_gap_vs_true_hs_oracle"),
        })
    if not rows:
        return None
    return {"id": "hs_conditional_mfe_gate", "rows": rows}


def build_core_model_report_extensions(
    payload: Mapping[str, Any],
    *,
    oos_scope_label: str,
) -> list[dict[str, Any]]:
    """Return reusable core extensions in stable contract order."""

    result: list[dict[str, Any]] = []
    multi_head = build_multi_head_learnability_extension(
        payload, oos_scope_label=oos_scope_label
    )
    if multi_head is not None:
        result.append(multi_head)
    hs_quality = build_hs_conditional_quality_extension(
        payload, oos_scope_label=oos_scope_label
    )
    if hs_quality is not None:
        result.append(hs_quality)
    return result


def extension_missing_evidence(extension: Mapping[str, Any]) -> tuple[str, ...]:
    """Return canonical missing-evidence issues for a normalized extension view."""

    ext = dict(extension or {})
    extension_id = str(ext.get("id") or "")
    issues: list[str] = []
    if extension_id == "multi_head_learnability":
        required = tuple(str(value) for value in ext.get("required_heads") or ())
        for row in list(ext.get("rows") or []):
            split = str(dict(row).get("split") or "-")
            for semantic_id in required:
                semantic = next(
                    (item for item in MODEL_HEAD_LEARNABILITY_SEMANTICS if item.semantic_id == semantic_id),
                    None,
                )
                if semantic is None:
                    issues.append(f"unsupported head semantic:{semantic_id}")
                    continue
                for key in semantic.metric_keys:
                    value = dict(row).get(key)
                    if value in {None, "", "MISSING"}:
                        issues.append(f"{split}:{semantic_id}:{key} missing")
    return tuple(dict.fromkeys(issues))


__all__ = [
    "build_core_model_report_extensions",
    "build_hs_conditional_quality_extension",
    "build_multi_head_learnability_extension",
    "expected_multi_head_semantics",
    "extension_missing_evidence",
]
