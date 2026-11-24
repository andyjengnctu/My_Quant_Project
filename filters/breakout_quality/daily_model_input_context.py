"""Bind optional daily model contexts to numeric inputs for existing rows.

Providers here consume a fixed stock/day index. Eligibility, targets, maturity
and replay stay in their canonical owners.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from filters.breakout_quality.models.architectures import ArchitectureDescriptor


@dataclass(frozen=True)
class DailyModelInputContext:
    features: tuple[str, ...]
    contract: Mapping[str, Any]
    materialize: Callable[..., np.ndarray]


def daily_context_metadata(
    *, model_input_context: DailyModelInputContext | None,
    risk_context_features: tuple[str, ...], use_risk_context: bool,
    predicted_context_column: str | None, use_predicted_scalar_context: bool,
    use_predicted_pair_weight_context: bool,
) -> dict[str, Any]:
    """One projection shared by the existing manifest and dataset summary."""
    return {
        **({"input_decision_context": dict(model_input_context.contract)}
           if model_input_context is not None else {}),
        "context_features": (
            list(model_input_context.features) if model_input_context is not None
            else list(risk_context_features) if use_risk_context
            else [predicted_context_column]
            if use_predicted_scalar_context and predicted_context_column is not None
            else []
        ),
        "pair_weight_context_features": (
            [predicted_context_column]
            if use_predicted_pair_weight_context and predicted_context_column is not None
            else []
        ),
    }


def bind_daily_model_input_context(
    architecture: ArchitectureDescriptor, *, trade_path_mode: bool, target_teacher,
) -> DailyModelInputContext | None:
    """Activate only explicit model-input capabilities."""
    if not architecture.has_capability("trade_decision_context"):
        return None
    if not trade_path_mode:
        raise ValueError("decision context requires a verified real-path teacher")
    if target_teacher is None:
        raise ValueError("decision context lacks the canonical target teacher")

    from filters.breakout_quality.trade_decision_context import (
        CONTEXT_CONTRACT, CONTEXT_FEATURES, build_trade_decision_context,
    )

    return DailyModelInputContext(
        features=CONTEXT_FEATURES,
        contract=CONTEXT_CONTRACT,
        materialize=partial(build_trade_decision_context, source=target_teacher),
    )
