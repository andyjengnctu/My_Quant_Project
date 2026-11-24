"""Admission capabilities for declared classifier outputs.

Legacy probability admission remains owned by positive_netpnl_contract.py.
Model-decision admission consumes a dated binary action without threshold fitting.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from core.positive_netpnl_contract import (
    POSITIVE_P_CLASS_COLUMN,
    POSITIVE_PROBABILITY_THRESHOLD,
    POSITIVE_SCORE_CONTRACT,
    requires_positive_score as _requires_dual_positive_score,
    positive_score_admitted as _dual_positive_score_admitted,
)

PRIMARY_CLASSIFIER_SOURCE_MODE = "primary_classifier"
MODEL_DECISION_SOURCE_MODE = "model_decision"


def uses_model_positive_decision(options: Mapping | None) -> bool:
    return str((options or {}).get("positive_source_mode") or "") == MODEL_DECISION_SOURCE_MODE



def uses_primary_positive_classifier(options: Mapping | None) -> bool:
    return str((options or {}).get("positive_source_mode") or "") == PRIMARY_CLASSIFIER_SOURCE_MODE


def requires_positive_score(options: Mapping | None) -> bool:
    """Declare and validate the unique Positive admission source topology."""
    options = dict(options or {})
    if uses_model_positive_decision(options):
        present = {str(k) for k in options if str(k).startswith("positive_")}
        required = {"positive_source_mode", "positive_dl_id", "positive_score_column"}
        resolved = {"positive_filter_id", "positive_model_architecture", "positive_experiment_profile",
                    "positive_score_source", "positive_score_path_override", "positive_score_manifest_path_override"}
        if (not required.issubset(present) or present - required - resolved
                or not str(options.get("positive_dl_id") or "").strip()
                or not str(options.get("positive_score_column") or "").strip()
                or options.get("selection_only") is not True):
            raise ValueError("Model decision requires one independent DL decision column and no probability threshold")
        return True
    if not uses_primary_positive_classifier(options):
        return _requires_dual_positive_score(options)
    allowed = {
        "positive_source_mode", "positive_score_column", "positive_probability_threshold",
    }
    present = {str(k) for k in options if str(k).startswith("positive_")}
    if present != allowed or options.get("primary_score_column") != POSITIVE_P_CLASS_COLUMN:
        raise ValueError("Positive-only PIT admission requires one primary binary score column, no rank/secondary source")
    threshold = options.get("positive_probability_threshold")
    if (options.get("positive_score_column") != POSITIVE_P_CLASS_COLUMN
            or type(threshold) not in (int, float)
            or not math.isfinite(float(threshold)) or threshold != POSITIVE_PROBABILITY_THRESHOLD):
        raise ValueError("Positive-only classifier requires the formal P(positive)>0.5 output")
    if options.get("selection_only") is not True:
        raise ValueError("Positive-only classifier must never alter capital or rank allocation")
    return True


def positive_score_admitted(row: Mapping) -> bool:
    """For primary-only, gate on valid PIT probability; never use it for rank."""
    options = row.get("breakout_quality_ranking_options") or {}
    if not isinstance(options, Mapping):
        raise ValueError("Ranking options must be a mapping")
    if uses_model_positive_decision(options):
        requires_positive_score(options)
        payload = row.get("breakout_quality_rank")
        if (not isinstance(payload, Mapping) or not payload.get("available")
                or not payload.get("positive_available")):
            return False
        rank_day = str(payload.get("score_date") or "")[:10]
        decision_day = str(payload.get("positive_score_date") or "")[:10]
        if not rank_day or rank_day != decision_day:
            return False
        # The shared score carrier retains its historical field name. Its
        # declared source column is the model's binary action, not probability.
        try:
            decision = float(payload.get("positive_probability"))
        except (ValueError, TypeError):
            return False
        return decision == 1.0
    if not uses_primary_positive_classifier(options):
        return _dual_positive_score_admitted(row)
    requires_positive_score(options)
    payload = row.get("breakout_quality_rank")
    if not isinstance(payload, Mapping) or not payload.get("available"):
        return False
    information_day = str(payload.get("score_date") or "")[:10]
    if not information_day:
        return False
    try:
        probability = float(payload.get("score"))
    except (ValueError, TypeError):
        return False
    return math.isfinite(probability) and 0.0 <= probability <= 1.0 and probability > POSITIVE_PROBABILITY_THRESHOLD


def admit_positive_score_rows(rows):
    return [row for row in list(rows or ()) if positive_score_admitted(row)]


def perfect_positive_arms_equivalent(arm: Mapping[str, Any], reference: Mapping[str, Any], *,
                                    sources: Mapping[str, Any], target_id: str,
                                    resolve_recipe: Callable[[str], Any]) -> bool:
    """Certify sign-perfect binary actions against sign-perfect probabilities.

    This applies ONLY to the user-approved hindsight intervention: mature
    positive NetPnL -> 1, nonpositive -> 0, unavailable -> no admission.
    It never equates learned probabilities/payoff mass, FITs or strategy runs.
    All ranking, teacher, entry, allocation and date/source semantics must
    otherwise match the already certified Oracle reference.
    Callers supply the canonical resolver so admission never imports configured
    model workflows that themselves depend on this low-level contract.
    """
    from dataclasses import asdict

    presentation = {"name", "description", "robustness_role", "replay_contract_version",
                    "dl_runtime_options"}
    if ({k: v for k, v in arm.items() if k not in presentation}
            != {k: v for k, v in reference.items() if k not in presentation}):
        return False
    policies, executions, input_contracts = [], [], []
    for selected in (arm, reference):
        options = dict(selected.get("dl_runtime_options") or {})
        if not requires_positive_score(options):
            return False
        source = sources.get(options.get("positive_dl_id"))
        if not source:
            return False
        recipe = resolve_recipe(source["experiment_profile"])
        policy = recipe.training_policy
        if (recipe.continuous_target_id != target_id
                or policy.target_builder != "netpnl_positive_binary"
                or policy.loss_handler != "categorical_cross_entropy"
                or policy.categorical_class_count != 2 or policy.categorical_primary_class != 1
                or policy.uses_pairwise_loss):
            return False
        if uses_model_positive_decision(options):
            if options.get("positive_score_column") != getattr(policy, "categorical_decision_column", None):
                return False
            policies.append("decision")
        else:
            if (getattr(policy, "categorical_decision_column", None) is not None
                    or options.get("positive_score_column") != POSITIVE_P_CLASS_COLUMN
                    or POSITIVE_P_CLASS_COLUMN not in policy.categorical_score_columns
                    or options.get("positive_probability_threshold") != 0.5):
                return False
            policies.append("probability")
        input_contracts.append({
            "source": {key: value for key, value in source.items()
                       if key not in {"description", "experiment_profile"}},
            "recipe": {key: asdict(recipe)[key] for key in (
                "model_architecture", "continuous_target_id", "training_label_scope",
                "training_sample_scope", "target_policy", "context_policy")},
        })
        executions.append({key: value for key, value in options.items()
                           if key not in {"positive_dl_id", "positive_score_column",
                                          "positive_source_mode", "positive_probability_threshold"}})
    return (policies == ["decision", "probability"] and
            input_contracts[0] == input_contracts[1] and executions[0] == executions[1])
