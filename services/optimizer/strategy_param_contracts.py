"""Pure parameter-search reference contracts shared by producers and readers.

These payloads are persisted scientific inputs. Keep their field presence and
canonical JSON unchanged when moving callers between execution workflows.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT
from core.file_integrity import canonical_json_sha256
from core.strategy_params import SELECTION_POLICY_PARAM_SPECS
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS


MIN_ROOS_SEARCH_FIELDS = (
    "high_len", "atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail",
)


def resolve_min_roos_search_fields(
    strategy_id: str, daily_search_profile: str | None = None,
) -> tuple[str, ...]:
    if strategy_id == "breakout":
        if daily_search_profile is not None:
            raise ValueError("Breakout Optimizer cannot select Daily search profile")
        return MIN_ROOS_SEARCH_FIELDS
    if strategy_id == "daily":
        from strategies.daily.search_space import resolve_daily_search_profile
        return tuple(resolve_daily_search_profile(daily_search_profile)["fields"])
    raise ValueError(f"未知Min Optimizer策略: {strategy_id!r}")


def build_min_roos_adaptation_contract(
    *, arm_id: str, training_dl_enabled: bool, strategy_id: str = "breakout",
    daily_search_profile: str | None = None,
) -> dict[str, Any]:
    payload = {
        "mode": "min_roos_training",
        "parameter_set": str(arm_id),
        "search_fields": list(resolve_min_roos_search_fields(strategy_id, daily_search_profile)),
        "fixed_rule_contract": "all_rule_filters_off",
        "training_dl_enabled": bool(training_dl_enabled),
    }
    if strategy_id == "daily":
        payload["entry_strategy"] = "Daily"
        payload["entry_universe"] = "unscored_all_adjacent_quote_stockdays_v1"
        if daily_search_profile is not None:
            from strategies.daily.search_space import resolve_daily_search_profile
            daily_spec = resolve_daily_search_profile(daily_search_profile)
            payload["daily_execution_contract"] = daily_spec["execution_contract"]
            payload["fixed_rule_contract"] = daily_spec.get(
                "fixed_rule_contract",
                "daily_common_filters_search_v2" if "common_filters" in daily_search_profile
                else "all_rule_filters_off",
            )
    return payload


def build_min_roos_schedule_contract(
    *, first_oos_date: str, last_oos_date: str,
    train_window_months: int, oos_months: int,
    strategy_id: str = "breakout", daily_search_profile: str | None = None,
) -> dict[str, Any]:
    """Resolve the one signed reference used by training and teacher inspection."""
    fields = resolve_min_roos_search_fields(strategy_id, daily_search_profile)
    first = pd.Timestamp(str(first_oos_date)).normalize()
    last = pd.Timestamp(str(last_oos_date)).normalize()
    if pd.isna(first) or pd.isna(last) or last < first:
        raise ValueError("Selection Min ROOS期間不合法")
    if int(train_window_months) < 1 or int(oos_months) < 1:
        raise ValueError("Selection Min ROOS rolling months必須>=1")
    meta = {
        "window_mode": "fixed",
        "first_oos_date": first.strftime("%Y-%m-%d"),
        "last_oos_date": last.strftime("%Y-%m-%d"),
        "train_window_months": int(train_window_months),
        "oos_horizon_months": int(oos_months),
    }
    reference = {
        "reference_type": "canonical_min_roos_fixed_defaults",
        "rolling_policy": meta,
        "search_fields": list(MIN_ROOS_SEARCH_FIELDS),
        "breakout_defaults": {key: spec["default"] for key, spec in BREAKOUT_PARAM_SPECS.items()},
        "selection_defaults": {key: spec["default"] for key, spec in SELECTION_POLICY_PARAM_SPECS.items()},
        "optimizer_fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
    }
    reference_type = "canonical_min_roos_fixed_defaults"
    if strategy_id == "daily":
        reference.update({
            "search_fields": list(fields),
            "entry_strategy": "Daily",
            "entry_universe": "unscored_all_adjacent_quote_stockdays_v1",
            "breakout_required": False,
        })
        if daily_search_profile is not None:
            from strategies.daily.search_space import resolve_daily_search_profile
            reference["daily_execution_contract"] = resolve_daily_search_profile(daily_search_profile)["execution_contract"]
        reference_type = "canonical_daily_min_roos_fixed_defaults"
    return {
        "path": None, "payload": {}, "meta": meta,
        "sha256": canonical_json_sha256(reference),
        "reference_type": reference_type, "reference_payload": reference,
    }
