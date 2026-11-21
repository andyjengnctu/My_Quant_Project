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


def bind_daily_model_input_context(
    architecture: ArchitectureDescriptor, *, trade_path_mode: bool, target_materializer,
) -> DailyModelInputContext | None:
    """Activate only explicit model-input capabilities."""
    if not architecture.has_capability("trade_decision_context"):
        return None
    if not trade_path_mode:
        raise ValueError("decision context requires a verified real-path teacher")
    source = getattr(target_materializer, "keywords", {}).get("source")
    if source is None:
        raise ValueError("decision context lacks the canonical target teacher")

    from filters.breakout_quality.trade_decision_context import (
        CONTEXT_CONTRACT, CONTEXT_FEATURES, build_trade_decision_context,
    )

    return DailyModelInputContext(
        features=CONTEXT_FEATURES,
        contract=CONTEXT_CONTRACT,
        materialize=partial(build_trade_decision_context, source=source),
    )
