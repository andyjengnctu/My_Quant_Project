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
INFRASTRUCTURE_FIELDS = (
    "optimizer_max_workers",
)

# Ignore disabled feature-family knobs when building expensive prep keys.  Those
# fields do not affect generated signals/trade paths while their switch is off,
# so treating them as distinct only wastes optimizer prep.  Future strategy/ML
# adapters can extend this table without touching portfolio/backtest rules.
_CONDITIONAL_INACTIVE_FIELD_GROUPS = (
    ("use_bb", ("bb_len", "bb_mult")),
    ("use_kc", ("kc_len", "kc_mult")),
    ("use_vol", ("vol_short_len", "vol_long_len")),
)


def _normalize_cache_value(value: Any):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 12)
    return value


def _params_to_payload(params) -> dict:
    if isinstance(params, dict):
        return dict(params)
    if is_dataclass(params):
        return strategy_params_to_dict(params)
    return {
        field_name: getattr(params, field_name)
        for field_name in dir(params)
        if not field_name.startswith("_") and not callable(getattr(params, field_name))
    }


def _inactive_conditional_fields(payload: dict) -> tuple[str, ...]:
    inactive_fields = []
    for switch_field, dependent_fields in _CONDITIONAL_INACTIVE_FIELD_GROUPS:
        if switch_field in payload and not bool(payload.get(switch_field)):
            inactive_fields.extend(field for field in dependent_fields if field in payload)
    return tuple(inactive_fields)


def canonical_param_items(params, *, omit_fields: Iterable[str] = ()): 
    payload = _params_to_payload(params)
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
    selection.  Inactive feature-family fields and infrastructure knobs are
    also omitted because they do not affect the prepared artifacts.
    """

    payload = _params_to_payload(params)
    omitted_fields = (
        *SELECTION_POLICY_FIELDS,
        *INFRASTRUCTURE_FIELDS,
        *_inactive_conditional_fields(payload),
    )
    return ("prep", canonical_param_items(payload, omit_fields=omitted_fields))


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
