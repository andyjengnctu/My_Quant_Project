"""Canonical optimizer cache keys.

This module is optimizer infrastructure, not a strategy rule.  It lets future
strategy families or ML models separate expensive signal/execution preparation
from cheap portfolio selection thresholds.
"""

from collections.abc import Iterable
from dataclasses import is_dataclass
from typing import Any

from config.training_policy import SELECTION_POLICY_PARAM_SPECS
from core.strategy_params import strategy_params_to_dict


SELECTION_POLICY_FIELDS = tuple(SELECTION_POLICY_PARAM_SPECS.keys())


def _normalize_cache_value(value: Any):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 12)
    return value


def canonical_param_items(params, *, omit_fields: Iterable[str] = ()):
    if isinstance(params, dict):
        payload = dict(params)
    elif is_dataclass(params):
        payload = strategy_params_to_dict(params)
    else:
        payload = {
            field_name: getattr(params, field_name)
            for field_name in dir(params)
            if not field_name.startswith("_") and not callable(getattr(params, field_name))
        }

    omitted = set(omit_fields)
    return tuple(
        (field_name, _normalize_cache_value(payload[field_name]))
        for field_name in sorted(payload)
        if field_name not in omitted
    )


def build_prep_cache_key(params):
    """Key for signal/execution artifacts.

    Selection-policy thresholds are intentionally omitted because they only
    decide whether an already-computed history record is usable in portfolio
    selection.  They must not force regeneration of signals or PIT trade paths.
    """

    return ("prep", canonical_param_items(params, omit_fields=SELECTION_POLICY_FIELDS))


def build_full_evaluation_cache_key(params, *, objective_mode, train_start_year, search_train_end_year, max_positions, enable_rotation):
    return (
        "evaluation",
        str(objective_mode),
        int(train_start_year),
        int(search_train_end_year),
        int(max_positions),
        bool(enable_rotation),
        canonical_param_items(params),
    )
