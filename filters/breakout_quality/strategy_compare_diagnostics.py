"""Post-replay Strategy Compare candidate/selection diagnostics."""

from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from core.breakout_quality_registry import (
    get_breakout_quality_experiment_profile,
)
from core.breakout_quality_policy import (
    BREAKOUT_QUALITY_WORKFLOW_FILTER_ID,
    BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE,
)
from config.strategy_compare import (
    STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_ENABLED,
    STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_CUTOFF,
    STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_METHOD,
    STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PROFILE,
    STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PROFILE_IDS,
    STRATEGY_COMPARE_UPSIDE_REALIZATION_ADVERSE_BUCKET_EDGES_R,
    STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE,
    STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS,
)
from core.console_report import project_relative_display_path
from core.display_common import _display_width
from core.report_metrics import (
    R_ANALYSIS_GROUPED_SECTIONS,
    R_ANALYSIS_MERGED_METRICS,
    R_MODEL_PREDICTION_METRICS,
    R_SELECTION_TRANSLATION_METRICS,
    MFE_SAFETY_COMPARE_RESULT_METRICS,
    UPSIDE_SURVIVAL_BASE_METRICS,
    upside_survival_initial_stop_metric,
    RAnalysisMetricSpec,
)
from core.report_style import best_worst_signals, finite_number, styled_signal
from core.strategy_comparison import StrategyComparisonSettings

from core.exact_accounting import (
    calc_planned_initial_risk_from_prices_milli,
    milli_to_money,
)
from core.order_lot_policy import apply_board_lot_preferred_qty
from core.price_utils import calc_position_size

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
)
from filters.breakout_quality.daily_ranker_data import load_official_breakout_candidate_keys
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.mfe_safety_geometry import (
    attach_quadrants as attach_mfe_safety_quadrants,
    build_truth_geometry as build_mfe_safety_truth_geometry,
    distribution_for_keys as mfe_safety_distribution_for_keys,
    filter_period as filter_mfe_safety_period,
    normalize_date as normalize_geometry_date,
    normalize_ticker as normalize_geometry_ticker,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_continuous_ranker_oos_score_table,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_score_table,
    load_selection_point_in_time_score_table_from_path,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_SCORE_RANKING,
    comparison_labels,
)
from filters.breakout_quality.trade_attribution import (
    build_upside_realization_attribution,
    reconstruct_round_trips,
    upside_first_passage_columns,
)

STRATEGY_DIAGNOSTICS_SCHEMA_VERSION = 4
UPSIDE_REALIZATION_SCHEMA_VERSION = 2


def _upside_realization_contract() -> dict[str, Any]:
    return {
        "strategy_diagnostics_schema_version": STRATEGY_DIAGNOSTICS_SCHEMA_VERSION,
        "schema_version": UPSIDE_REALIZATION_SCHEMA_VERSION,
        "path_target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        "path_profile": STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE,
        "upside_r_thresholds": [float(value) for value in STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS],
        "adverse_bucket_edges_r": [float(value) for value in STRATEGY_COMPARE_UPSIDE_REALIZATION_ADVERSE_BUCKET_EDGES_R],
        "actual_stop_source": "canonical_completed_trade_exit_type",
        "actual_stop_stage_source": "canonical_trade_history_entry_and_exit_stop_prices",
        "path_component_source": "canonical_daily_profile_sample_provider",
        "peak_timing_source": "canonical_full_horizon_target_opportunity_date",
        "first_passage_timing_source": "canonical_daily_ohlcv_used_by_pure_mfe_sample_provider",
        "first_passage_threshold_basis": "score_event_close_plus_target_risk_budget_return_times_kR",
        "same_day_stop_peak_policy": "conservative_stop_before_peak",
        "same_day_risk_upside_policy": "conservative_risk_before_upside",
        "same_day_actual_stop_upside_policy": "conservative_stop_before_upside",
    }


def _upside_realization_contract_fingerprint() -> str:
    raw = json.dumps(
        _upside_realization_contract(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _pure_mfe_diagnostic_profile() -> str:
    """Resolve the explicit canonical owner of path diagnostics.

    A continuous target can be shared by multiple research profiles without those
    profiles owning the same diagnostic producer semantics.  MR-13P intentionally
    reuses the Pure-MFE primary target, so target-id uniqueness is not an ownership
    contract.  Strategy Compare therefore consumes the profile explicitly declared
    by its diagnostics configuration and validates that its target identity remains
    compatible.
    """
    profile_name = str(STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE)
    profile = get_breakout_quality_experiment_profile(profile_name)
    if str(profile.continuous_target_id or "") != DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID:
        raise ValueError(
            "Upside Realization path profile target identity不一致: "
            f"profile={profile_name}, target={profile.continuous_target_id!r}, "
            f"expected={DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID!r}"
        )
    return profile_name


@lru_cache(maxsize=4)
def _full_horizon_path_lookup_cached(
    project_root_text: str, filter_id: str, architecture: str
) -> pd.DataFrame:
    root = Path(project_root_text).resolve()
    # ``architecture`` is the active strategy arm architecture and is intentionally
    # not the owner of this offline path diagnostic.  Upside Survival is defined by
    # the explicit canonical Pure-MFE diagnostic profile.  Most historical arms use
    # inception_time_v1 so the two happened to match; MR-13AC uses a context
    # architecture, exposing the latent ownership bug.  Resolve the diagnostic
    # provider architecture from its own profile/default instead of the active arm.
    del architecture
    diagnostic_profile = _pure_mfe_diagnostic_profile()
    diagnostic_spec = get_breakout_quality_experiment_profile(diagnostic_profile)
    diagnostic_architecture = str(
        diagnostic_spec.model_architecture or BREAKOUT_QUALITY_MODEL_ARCHITECTURE
    )
    bundle = load_profile_continuous_ranker_data(
        filter_id=str(filter_id),
        model_architecture=diagnostic_architecture,
        experiment_profile=diagnostic_profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table.copy()
    required = {
        "ticker", "date", "target_adverse_r", "target_opportunity_bar",
        "target_first_risk_breach_bar", "target_opportunity_date",
        "target_first_risk_breach_date",
    }
    missing = sorted(required - set(groups.columns))
    if missing:
        raise ValueError(f"pure-MFE path lookup缺少欄位: {missing}")
    if len(groups) != len(bundle.raw_target) or len(groups) != len(bundle.target_valid):
        raise ValueError("pure-MFE path lookup group/target長度不一致")
    target_contract = dict(dict(bundle.target_manifest or {}).get("target_contract") or {})
    if str(target_contract.get("target_id") or "") != DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID:
        raise ValueError("pure-MFE canonical sample provider target identity不一致")

    # Daily-universal targets are canonicalized by the profile-aware sample provider itself;
    # unlike the older event-group target family, they do not own a second persisted
    # continuous-target component bundle.  Consume the provider's already-computed target
    # and path columns directly so this diagnostic stays on the same target SSOT without
    # inventing a duplicate physical artifact requirement.
    target_valid = pd.Series(bundle.target_valid, index=groups.index).astype(bool)
    mfe_r = pd.to_numeric(pd.Series(bundle.raw_target, index=groups.index), errors="coerce")
    adverse_r = pd.to_numeric(groups["target_adverse_r"], errors="coerce")
    opportunity_bar = pd.to_numeric(groups["target_opportunity_bar"], errors="coerce")
    first_risk_breach_bar = pd.to_numeric(
        groups["target_first_risk_breach_bar"], errors="coerce"
    )
    if bool((target_valid & ~mfe_r.map(math.isfinite)).any()):
        raise ValueError("pure-MFE canonical sample provider valid rows含非有限MFE")
    if bool((target_valid & ~adverse_r.map(math.isfinite)).any()):
        raise ValueError("pure-MFE canonical sample provider valid rows含非有限adverse")
    if bool((target_valid & (opportunity_bar < 1)).any()):
        raise ValueError("pure-MFE canonical sample provider valid rows的opportunity_bar必須>=1")

    risk_budget_return = float(target_contract.get("risk_budget_return") or 0.0)
    horizon_bars = int(target_contract.get("horizon_bars") or 0)
    if not math.isfinite(risk_budget_return) or risk_budget_return <= 0.0:
        raise ValueError("pure-MFE canonical sample provider缺少合法risk_budget_return")
    if horizon_bars <= 0:
        raise ValueError("pure-MFE canonical sample provider缺少合法horizon_bars")
    r_thresholds = tuple(
        sorted({float(value) for value in STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS})
    )
    return_thresholds = tuple(risk_budget_return * value for value in r_thresholds)
    feature_bank = bundle.feature_bank
    future_first_passage = getattr(feature_bank, "future_first_passage", None)
    if not callable(future_first_passage):
        raise ValueError("pure-MFE canonical daily feature bank不支援future first-passage materialization")
    passage_by_return = future_first_passage(
        pd.to_numeric(groups["group_index"], errors="raise").to_numpy(dtype="int64"),
        horizon_bars=horizon_bars,
        return_thresholds=return_thresholds,
    )

    lookup = pd.DataFrame({
        "ticker": groups["ticker"].fillna("").astype(str).str.strip(),
        "score_event_date": pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d"),
        "path_target_available": target_valid,
        "full_horizon_mfe_r": mfe_r,
        "full_horizon_adverse_to_peak_r": adverse_r,
        "full_horizon_opportunity_bar": opportunity_bar,
        "full_horizon_first_risk_breach_bar": first_risk_breach_bar,
        "full_horizon_opportunity_date": pd.to_datetime(
            groups["target_opportunity_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "full_horizon_first_risk_breach_date": pd.to_datetime(
            groups["target_first_risk_breach_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
    })
    for threshold_r, threshold_return in zip(r_thresholds, return_thresholds):
        first_bar_column, first_date_column = upside_first_passage_columns(threshold_r)
        passage_bar, passage_date = passage_by_return[threshold_return]
        lookup[first_bar_column] = pd.to_numeric(
            pd.Series(passage_bar, index=groups.index), errors="coerce"
        )
        lookup[first_date_column] = pd.to_datetime(
            pd.Series(passage_date, index=groups.index), errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        passage_valid = lookup[first_bar_column].ge(1) & lookup[first_date_column].notna()
        should_reach = target_valid & (mfe_r >= threshold_r + 1e-5)
        if bool((should_reach & ~passage_valid).any()):
            bad = groups.loc[should_reach & ~passage_valid].iloc[0]
            raise ValueError(
                "pure-MFE first-passage與MFE門檻不一致: "
                f"ticker={bad.get('ticker')}, date={bad.get('date')}, threshold={threshold_r:g}R"
            )
        impossible_reach = target_valid & (mfe_r < threshold_r - 1e-5) & passage_valid
        if bool(impossible_reach.any()):
            bad = groups.loc[impossible_reach].iloc[0]
            raise ValueError(
                "pure-MFE first-passage出現MFE未達門檻卻hit: "
                f"ticker={bad.get('ticker')}, date={bad.get('date')}, threshold={threshold_r:g}R"
            )
    if bool(lookup.duplicated(["ticker", "score_event_date"]).any()):
        raise ValueError("pure-MFE path lookup同ticker/date不唯一")
    return lookup.reset_index(drop=True)


def _join_full_horizon_path_diagnostics(
    selected: pd.DataFrame, *, project_root: Path, filter_id: str, architecture: str
) -> pd.DataFrame:
    frame = pd.DataFrame(selected).copy()
    if "score_event_date" not in frame.columns:
        raise ValueError("selected target diagnostics缺少score_event_date")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
    frame["score_event_date"] = pd.to_datetime(
        frame["score_event_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d").fillna("")
    path_columns = {
        "path_target_available", "full_horizon_mfe_r",
        "full_horizon_adverse_to_peak_r", "full_horizon_opportunity_bar",
        "full_horizon_first_risk_breach_bar", "full_horizon_opportunity_date",
        "full_horizon_first_risk_breach_date",
        *(
            column
            for threshold in STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS
            for column in upside_first_passage_columns(float(threshold))
        ),
    }
    frame = frame.drop(columns=[column for column in path_columns if column in frame.columns])
    lookup = _full_horizon_path_lookup_cached(
        str(Path(project_root).resolve()), str(filter_id), str(architecture)
    )
    return frame.merge(
        lookup, on=["ticker", "score_event_date"], how="left", validate="many_to_one"
    )


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _exact_planned_initial_risk(
    *,
    limit_price: float,
    stop_price: float,
    qty: int,
    params,
    ticker: str,
    security_profile,
    trade_date: str,
) -> float | None:
    if qty <= 0:
        return 0.0
    try:
        risk_milli = calc_planned_initial_risk_from_prices_milli(
            limit_price,
            stop_price,
            int(qty),
            params,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=trade_date,
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    return float(milli_to_money(risk_milli))


def _flatten_entry_execution_rows(replay_execution_rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    columns = [
        "execution_order", "ticker", "trade_date", "candidate_date", "signal_date", "entry_type",
        "candidate_qty", "chosen_qty", "filled_qty", "entry_filled",
        "limit_px", "init_sl", "entry_fill_price",
        "sizing_equity", "candidate_sizing_capital", "effective_entry_budget",
        "available_cash_before", "reserved_cost", "fixed_risk",
        "max_position_cap_pct", "max_qty", "core_qty", "qty_without_risk_cap",
        "qty_without_position_cap", "qty_without_risk_or_position_cap",
        "qty_after_max_qty_before_lot", "lot_rounding_qty_loss",
        "candidate_initial_risk", "chosen_initial_risk", "actual_initial_risk", "risk_budget",
        "candidate_risk_utilization", "chosen_risk_utilization", "actual_risk_utilization",
        "risk_cap_binding", "position_cap_binding", "risk_position_tie",
        "capital_binding", "max_qty_binding", "lot_rounding_binding",
        "entry_budget_cash_binding", "binding_signature",
    ]
    rows: list[dict[str, Any]] = []
    for raw in list(replay_execution_rows or []):
        item = dict(raw or {})
        if str(item.get("_event_type") or "") != "entry_execution":
            continue
        params = item.get("params_obj")
        if params is None:
            continue
        ticker = str(item.get("ticker") or "").strip()
        trade_date = str(item.get("trade_date") or "")
        limit_px = _finite_float(item.get("limit_px"))
        init_sl = _finite_float(item.get("init_sl"))
        sizing_capital = _finite_float(item.get("candidate_sizing_capital"))
        if sizing_capital is None:
            sizing_capital = _finite_float(item.get("sizing_equity"))
        candidate_qty = int(item.get("candidate_qty", 0) or 0)
        chosen_qty = int(item.get("chosen_qty", 0) or 0)
        filled_qty = int(item.get("filled_qty", 0) or 0)
        max_qty_raw = item.get("max_qty")
        try:
            max_qty = int(max_qty_raw) if max_qty_raw not in (None, "") else None
        except (TypeError, ValueError):
            max_qty = None

        core_qty = no_risk_qty = no_position_qty = no_both_qty = 0
        qty_after_max = 0
        lot_loss = 0
        risk_binding = position_binding = risk_position_tie = capital_binding = False
        max_qty_binding = lot_binding = cash_binding = False
        candidate_initial_risk = chosen_initial_risk = actual_initial_risk = risk_budget = None
        candidate_utilization = chosen_utilization = actual_utilization = None
        fixed_risk = _finite_float(getattr(params, "fixed_risk", None))
        max_position_cap_pct = _finite_float(getattr(params, "max_position_cap_pct", None))

        if limit_px is not None and init_sl is not None and sizing_capital is not None and sizing_capital > 0.0:
            core_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, float(params.fixed_risk), params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            no_risk_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, 1.0, params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            no_position_params = copy.copy(params)
            no_position_params.max_position_cap_pct = 1.0
            no_position_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, float(params.fixed_risk), no_position_params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            no_both_qty = int(calc_position_size(
                limit_px, init_sl, sizing_capital, 1.0, no_position_params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            ))
            risk_gain = no_risk_qty - core_qty
            position_gain = no_position_qty - core_qty
            both_gain = no_both_qty - core_qty
            risk_binding = risk_gain > 0 and position_gain <= 0
            position_binding = position_gain > 0 and risk_gain <= 0
            risk_position_tie = risk_gain <= 0 and position_gain <= 0 and both_gain > 0
            capital_binding = both_gain <= 0
            qty_after_max = min(core_qty, max_qty) if max_qty is not None and max_qty > 0 else core_qty
            max_qty_binding = bool(max_qty is not None and max_qty > 0 and qty_after_max < core_qty)
            lot_qty = int(apply_board_lot_preferred_qty(limit_px, qty_after_max, params)) if qty_after_max > 0 else 0
            lot_loss = max(0, int(qty_after_max) - int(lot_qty))
            lot_binding = lot_loss > 0
            candidate_initial_risk = _exact_planned_initial_risk(
                limit_price=limit_px, stop_price=init_sl, qty=candidate_qty, params=params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            )
            chosen_initial_risk = _exact_planned_initial_risk(
                limit_price=limit_px, stop_price=init_sl, qty=chosen_qty, params=params,
                ticker=ticker, security_profile=item.get("security_profile"), trade_date=trade_date,
            )
            risk_budget = float(sizing_capital * float(params.fixed_risk))
            if risk_budget > 0.0 and candidate_initial_risk is not None:
                candidate_utilization = float(candidate_initial_risk / risk_budget)
            if risk_budget > 0.0 and chosen_initial_risk is not None:
                chosen_utilization = float(chosen_initial_risk / risk_budget)

        actual_risk_milli = int(item.get("actual_initial_risk_total_milli", 0) or 0)
        if actual_risk_milli > 0:
            actual_initial_risk = float(milli_to_money(actual_risk_milli))
            if risk_budget is not None and risk_budget > 0.0:
                actual_utilization = float(actual_initial_risk / risk_budget)
        cash_binding = chosen_qty > 0 and candidate_qty > 0 and chosen_qty < candidate_qty

        signature: list[str] = []
        if risk_binding:
            signature.append("RISK_CAP")
        if position_binding:
            signature.append("POSITION_CAP")
        if risk_position_tie:
            signature.append("RISK_POSITION_TIE")
        if capital_binding:
            signature.append("CAPITAL")
        if max_qty_binding:
            signature.append("MAX_QTY")
        if lot_binding:
            signature.append("LOT_ROUNDING")
        if cash_binding:
            signature.append("ENTRY_BUDGET_CASH")
        if not signature:
            signature.append("NONE_DETECTED")

        rows.append({
            "execution_order": int(item.get("execution_order", len(rows)) or 0),
            "ticker": ticker,
            "trade_date": trade_date,
            "candidate_date": str(item.get("candidate_date") or ""),
            "signal_date": str(item.get("signal_date") or ""),
            "entry_type": str(item.get("type") or "normal"),
            "candidate_qty": candidate_qty,
            "chosen_qty": chosen_qty,
            "filled_qty": filled_qty,
            "entry_filled": bool(item.get("entry_filled", False)),
            "limit_px": limit_px,
            "init_sl": init_sl,
            "entry_fill_price": _finite_float(item.get("entry_fill_price")),
            "sizing_equity": _finite_float(item.get("sizing_equity")),
            "candidate_sizing_capital": sizing_capital,
            "effective_entry_budget": _finite_float(item.get("effective_entry_budget")),
            "available_cash_before": _finite_float(item.get("available_cash_before")),
            "reserved_cost": _finite_float(item.get("reserved_cost")),
            "fixed_risk": fixed_risk,
            "max_position_cap_pct": max_position_cap_pct,
            "max_qty": max_qty,
            "core_qty": core_qty,
            "qty_without_risk_cap": no_risk_qty,
            "qty_without_position_cap": no_position_qty,
            "qty_without_risk_or_position_cap": no_both_qty,
            "qty_after_max_qty_before_lot": qty_after_max,
            "lot_rounding_qty_loss": lot_loss,
            "candidate_initial_risk": candidate_initial_risk,
            "chosen_initial_risk": chosen_initial_risk,
            "actual_initial_risk": actual_initial_risk,
            "risk_budget": risk_budget,
            "candidate_risk_utilization": candidate_utilization,
            "chosen_risk_utilization": chosen_utilization,
            "actual_risk_utilization": actual_utilization,
            "risk_cap_binding": risk_binding,
            "position_cap_binding": position_binding,
            "risk_position_tie": risk_position_tie,
            "capital_binding": capital_binding,
            "max_qty_binding": max_qty_binding,
            "lot_rounding_binding": lot_binding,
            "entry_budget_cash_binding": cash_binding,
            "binding_signature": "+".join(signature),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(
        ["execution_order"], kind="mergesort"
    ).reset_index(drop=True)

def _flatten_candidate_replay_rows(replay_counts: dict[str, dict[str, Any]], field: str) -> pd.DataFrame:
    if field not in {"candidate_rows", "orderable_rows"}:
        raise ValueError(f"不支援的candidate replay field: {field}")
    rows: list[dict[str, Any]] = []
    for ticker in sorted(replay_counts):
        bucket = replay_counts.get(ticker) or {}
        for raw in list(bucket.get(field) or []):
            row = dict(raw or {})
            row["ticker"] = str(row.get("ticker") or ticker)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "candidate_date", "signal_date",
            "candidate_type", "entry_source", "is_orderable", "high_len",
            "ensemble_vote_count", "qty", "sort_value", "historical_ev",
            "historical_win_rate", "historical_trade_count",
            "breakout_quality_score", "breakout_quality_score_date",
        ])
    frame = pd.DataFrame(rows)
    for column in ("trade_date", "candidate_date", "signal_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return frame.sort_values(
        ["trade_date", "ticker", "signal_date", "candidate_type"],
        kind="mergesort",
    ).reset_index(drop=True)

def _flatten_selector_trace_rows(replay_selector_trace_rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Serialize transient max-DL selector stage membership without runtime objects."""

    columns = [
        "stage", "stage_rank", "ticker", "trade_date", "candidate_date", "signal_date",
        "candidate_type", "entry_source", "breakout_quality_score",
        "breakout_quality_score_date", "breakout_quality_score_available",
        "breakout_quality_expected_r_available", "breakout_quality_expected_r",
        "breakout_quality_daily_score_percentile", "breakout_quality_expected_r_calibration_cutoff",
        "breakout_quality_expected_excess_r_available", "breakout_quality_expected_excess_r",
        "breakout_quality_expected_excess_r_calibration_cutoff",
        "pre_market_order_limit", "direct_score_order_feasible", "repair_steps", "ascent_steps",
    ]
    rows: list[dict[str, Any]] = []
    for raw in list(replay_selector_trace_rows or []):
        item = dict(raw or {})
        if str(item.get("trace_kind") or "basket_stage") != "basket_stage":
            continue
        rows.append({
            "stage": str(item.get("stage") or ""),
            "stage_rank": int(item.get("stage_rank", 0) or 0),
            "ticker": str(item.get("ticker") or ""),
            "trade_date": str(item.get("trade_date") or ""),
            "candidate_date": str(item.get("candidate_date") or ""),
            "signal_date": str(item.get("signal_date") or ""),
            "candidate_type": str(item.get("candidate_type") or ""),
            "entry_source": str(item.get("entry_source") or ""),
            "breakout_quality_score": _finite_float(item.get("breakout_quality_score")),
            "breakout_quality_score_date": str(item.get("breakout_quality_score_date") or ""),
            "breakout_quality_score_available": bool(item.get("breakout_quality_score_available", False)),
            "breakout_quality_expected_r_available": bool(
                item.get("breakout_quality_expected_r_available", False)
            ),
            "breakout_quality_expected_r": _finite_float(item.get("breakout_quality_expected_r")),
            "breakout_quality_daily_score_percentile": _finite_float(
                item.get("breakout_quality_daily_score_percentile")
            ),
            "breakout_quality_expected_r_calibration_cutoff": str(
                item.get("breakout_quality_expected_r_calibration_cutoff") or ""
            ),
            "breakout_quality_expected_excess_r_available": bool(
                item.get("breakout_quality_expected_excess_r_available", False)
            ),
            "breakout_quality_expected_excess_r": _finite_float(
                item.get("breakout_quality_expected_excess_r")
            ),
            "breakout_quality_expected_excess_r_calibration_cutoff": str(
                item.get("breakout_quality_expected_excess_r_calibration_cutoff") or ""
            ),
            "pre_market_order_limit": (
                None if item.get("pre_market_order_limit") in (None, "")
                else int(item.get("pre_market_order_limit"))
            ),
            "direct_score_order_feasible": bool(item.get("direct_score_order_feasible", False)),
            "repair_steps": int(item.get("repair_steps", 0) or 0),
            "ascent_steps": int(item.get("ascent_steps", 0) or 0),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(
        ["trade_date", "stage", "stage_rank", "ticker"], kind="mergesort"
    ).reset_index(drop=True)


def _flatten_repair_mechanism_rows(
    replay_selector_trace_rows: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """Serialize scalable repair-search certificates from production selector trace."""

    summary_columns = [
        "trace_kind", "trade_date", "status", "classification",
        "target_count", "candidate_count", "reserve_floor_milli",
        "raw_selected_count", "raw_reserved_cost_milli", "raw_count_deficit",
        "raw_reserve_deficit_milli", "actual_repair_steps",
        "actual_repair_replacement_distance", "raw_score_sum",
        "repair_seed_score_sum", "final_score_sum",
        "repair_seed_score_loss_from_raw", "final_score_change_from_repair",
        "repair_search_evaluations", "ascent_search_evaluations",
        "ascent_local_optimum",
    ]
    candidate_columns = [
        "repair_role", "repair_step", "ticker", "candidate_date",
        "signal_date", "candidate_type", "entry_source", "breakout_quality_score",
        "breakout_quality_score_date", "before_selected_count", "after_selected_count",
        "before_reserved_cost_milli", "after_reserved_cost_milli",
        "before_count_deficit", "after_count_deficit",
        "before_reserve_deficit_milli", "after_reserve_deficit_milli", "after_feasible",
        "evaluated_swap_count", "progress_swap_count", "feasible_swap_count",
    ]
    columns = summary_columns + candidate_columns
    rows = []
    for raw in list(replay_selector_trace_rows or []):
        item = dict(raw or {})
        trace_kind = str(item.get("trace_kind") or "")
        if trace_kind not in {"repair_summary", "repair_swap"}:
            continue
        row = {column: None for column in columns}
        row.update({
            "trace_kind": trace_kind,
            "trade_date": str(item.get("trade_date") or ""),
            "status": str(item.get("status") or ""),
            "classification": str(item.get("classification") or ""),
            "repair_role": str(item.get("repair_role") or ""),
            "ticker": str(item.get("ticker") or ""),
            "candidate_date": str(item.get("candidate_date") or ""),
            "signal_date": str(item.get("signal_date") or ""),
            "candidate_type": str(item.get("candidate_type") or ""),
            "entry_source": str(item.get("entry_source") or ""),
            "breakout_quality_score": _finite_float(item.get("breakout_quality_score")),
            "breakout_quality_score_date": str(item.get("breakout_quality_score_date") or ""),
            "ascent_local_optimum": bool(item.get("ascent_local_optimum", False)) if trace_kind == "repair_summary" else None,
        })
        int_columns = (
            "target_count", "candidate_count", "reserve_floor_milli", "raw_selected_count",
            "raw_reserved_cost_milli", "raw_count_deficit", "raw_reserve_deficit_milli",
            "actual_repair_steps", "actual_repair_replacement_distance",
            "repair_search_evaluations", "ascent_search_evaluations",
            "repair_step", "before_selected_count", "after_selected_count",
            "before_reserved_cost_milli", "after_reserved_cost_milli",
            "before_count_deficit", "after_count_deficit",
            "before_reserve_deficit_milli", "after_reserve_deficit_milli",
            "evaluated_swap_count", "progress_swap_count", "feasible_swap_count",
        )
        for column in int_columns:
            value = item.get(column)
            row[column] = None if value in (None, "") else int(value)
        for column in (
            "raw_score_sum", "repair_seed_score_sum", "final_score_sum",
            "repair_seed_score_loss_from_raw", "final_score_change_from_repair",
        ):
            row[column] = _finite_float(item.get(column))
        if trace_kind == "repair_swap":
            row["after_feasible"] = bool(item.get("after_feasible", False))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(
        ["trade_date", "trace_kind", "repair_step", "repair_role", "ticker"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _flatten_selected_buy_rows(trade_history: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(trade_history).copy()
    columns = ["ticker", "trade_date", "signal_date", "type"]
    if frame.empty or not {"Date", "Ticker", "Type"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    buy_mask = frame["Type"].fillna("").astype(str).str.startswith("買進 (")
    out = frame.loc[buy_mask].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "ticker": out["Ticker"].fillna("").astype(str).str.strip(),
        "trade_date": pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "signal_date": pd.to_datetime(
            out.get("買訊日", pd.Series("", index=out.index)), errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "type": out["Type"].fillna("").astype(str),
    })
    return out.dropna(subset=["trade_date"]).sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)

def _selection_target_lookup(
    *, root: Path, filter_id: str, architecture: str, profile: str,
    score_path_override: str | None = None, manifest_path_override: str | None = None,
    score_column: str = "breakout_quality_score",
) -> pd.DataFrame:
    scores = (
        load_selection_point_in_time_score_table_from_path(
            str(score_path_override), manifest_path=str(manifest_path_override)
        )
        if score_path_override not in (None, "")
        else load_selection_point_in_time_score_table(
            str(root), filter_id, architecture, profile
        )
    ).reset_index()
    resolved_score_column = str(score_column or "breakout_quality_score").strip()
    if resolved_score_column not in scores.columns:
        raise ValueError(
            f"Selection PIT diagnostic score table缺少runtime欄位: {resolved_score_column}"
        )
    scores["__runtime_score__"] = pd.to_numeric(
        scores[resolved_score_column], errors="raise"
    ).astype(float)
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=architecture,
        experiment_profile=profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table[["group_index", "ticker", "date", "label"]].copy()
    groups["ticker"] = groups["ticker"].fillna("").astype(str).str.strip()
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d")
    groups["target_raw_r"] = bundle.raw_target
    groups["target_available"] = bundle.target_valid & pd.Series(bundle.raw_target).map(math.isfinite).to_numpy()
    joined = scores.merge(groups, on="group_index", how="left", validate="one_to_one", suffixes=("", "_dataset"))
    mismatch = (
        joined["ticker"] != joined["ticker_dataset"]
    ) | (
        joined["date"] != joined["date_dataset"]
    )
    if bool(mismatch.any()):
        raise ValueError("Selection PIT Score與Dataset group identity不一致")
    return joined[[
        "ticker", "date", "group_index", "__runtime_score__", "fold_id",
        "model_information_cutoff", "label", "target_raw_r", "target_available",
    ]].rename(columns={
        "date": "signal_date",
        "__runtime_score__": "breakout_quality_score",
    })


def _continuous_forward_target_lookup(
    *, root: Path, filter_id: str, architecture: str, profile: str,
    score_path_override: str | None = None,
    score_column: str = "model_score",
) -> pd.DataFrame:
    """Build a post-replay diagnostic lookup from frozen Forward-OOS scores.

    The Forward score artifact already embeds ``target_raw_r`` only for rows whose
    future target is evaluable. Runtime ranking consumes ``model_score`` only; this
    helper is called after replay and therefore cannot leak Future Target into
    candidate ordering, sizing, or execution.
    """

    scores = (
        load_continuous_ranker_oos_score_table_from_path(
            str(score_path_override), str(profile)
        )
        if score_path_override not in (None, "")
        else load_continuous_ranker_oos_score_table(
            str(root), str(filter_id), str(architecture), str(profile)
        )
    ).reset_index()
    resolved_score_column = str(score_column or "model_score").strip()
    required = {"ticker", "date", "group_index", resolved_score_column, "target_raw_r"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Forward-OOS diagnostic score table缺少欄位: {missing}")
    scores["ticker"] = scores["ticker"].fillna("").astype(str).str.strip()
    scores["date"] = pd.to_datetime(scores["date"], errors="raise").dt.strftime("%Y-%m-%d")
    scores["breakout_quality_score"] = pd.to_numeric(
        scores[resolved_score_column], errors="raise"
    ).astype(float)
    scores["target_raw_r"] = pd.to_numeric(scores["target_raw_r"], errors="coerce")
    scores["target_available"] = scores["target_raw_r"].map(math.isfinite)
    if bool(scores.duplicated(["ticker", "date"]).any()):
        raise ValueError("Forward-OOS diagnostic score同ticker/date必須唯一")
    return scores[[
        "ticker", "date", "group_index", "breakout_quality_score",
        "target_raw_r", "target_available",
    ]].rename(columns={"date": "signal_date"})

def _load_isolated_selection_pit_contract(
    *, score_path: str, manifest_path: str, filter_id: str, model_architecture: str,
    experiment_profile: str, expected_seed: int | None,
):
    score = Path(str(score_path)).resolve()
    manifest_file = Path(str(manifest_path)).resolve()
    for label, path in (("Selection PIT score", score), ("Selection PIT manifest", manifest_file)):
        if not path.is_file():
            raise FileNotFoundError(f"找不到isolated {label}: {path}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("isolated Selection PIT manifest根節點必須是object")
    if str(manifest.get("status") or "") != "BUILT":
        raise ValueError(f"isolated Selection PIT manifest尚未完成: {manifest.get('status')!r}")
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        actual = str(manifest.get(field) or "")
        if actual != expected:
            raise ValueError(
                f"isolated Selection PIT identity不一致: field={field}, expected={expected}, actual={actual}"
            )
    if expected_seed is not None and int(manifest.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            "isolated Selection PIT seed不一致: "
            f"expected={int(expected_seed)}, actual={manifest.get('seed')!r}"
        )
    lookahead = dict(manifest.get("lookahead_contract") or {})
    required_true = (
        "every_score_uses_model_not_trained_on_scored_event",
        "training_requires_label_eval_end_before_score_start",
    )
    required_false = (
        "oos_rows_or_target_statistics_used_for_training_or_epoch_selection",
        "future_target_in_score_table",
    )
    if any(lookahead.get(key) is not True for key in required_true) or any(
        lookahead.get(key) is not False for key in required_false
    ):
        raise ValueError("isolated Selection PIT lookahead contract不符合策略回放要求")
    score_record = dict((manifest.get("artifacts") or {}).get("scores") or {})
    if str(score_record.get("filename") or "") != score.name:
        raise ValueError("isolated Selection PIT score filename與manifest不一致")
    expected_hash = str(score_record.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(score).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("isolated Selection PIT score SHA256與manifest不一致")
    expected_size = int(score_record.get("size_bytes", -1))
    if expected_size != int(score.stat().st_size):
        raise ValueError("isolated Selection PIT score size與manifest不一致")
    table = load_selection_point_in_time_score_table_from_path(
        str(score), manifest_path=str(manifest_file)
    )
    available_from = str(table.attrs.get("available_from") or "")
    available_through = str(table.attrs.get("available_through") or "")
    if not available_from or not available_through or available_through < available_from:
        raise ValueError("isolated Selection PIT score period不合法")
    return SimpleNamespace(
        score_path=score,
        manifest_path=manifest_file,
        audit_path=None,
        manifest=manifest,
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        seed=int(manifest.get("seed", 0) or 0),
        available_from=available_from,
        available_through=available_through,
        model_validation_gate={
            "status": "ISOLATED_MULTI_SEED_ROBUSTNESS",
            "strategy_metrics_used": False,
            "future_target_used_for_runtime_sort": False,
        },
    )

def strategy_replay_score_event_frames(
    orderable: pd.DataFrame,
    selected: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Resolve replay rows to the canonical model score-event date.

    Continuation/re-entry rows may have a transaction signal date later than the
    model information date.  Strategy diagnostics therefore use the persisted
    ``breakout_quality_score_date`` when present and fall back to ``signal_date``
    only for legacy/direct breakout rows.  Audits must reuse the same mapping.
    """

    orderable_work = pd.DataFrame(orderable).copy()
    selected_work = None if selected is None else pd.DataFrame(selected).copy()
    frames_and_columns = [(orderable_work, ("trade_date", "signal_date"))]
    if selected_work is not None:
        frames_and_columns.append((selected_work, ("trade_date", "signal_date")))
    for frame, columns in frames_and_columns:
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column], errors="coerce"
                ).dt.strftime("%Y-%m-%d").fillna("")

    raw_score_dates = orderable_work.get(
        "breakout_quality_score_date",
        pd.Series("", index=orderable_work.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    parsed_score_dates = pd.to_datetime(raw_score_dates, errors="coerce")
    invalid_score_dates = raw_score_dates.ne("") & parsed_score_dates.isna()
    if bool(invalid_score_dates.any()):
        bad = orderable_work.loc[invalid_score_dates].iloc[0]
        bad_score_date = raw_score_dates.loc[invalid_score_dates].iloc[0]
        raise ValueError(
            "策略replay保存的Breakout Quality score_date無法解析: "
            f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
            f"signal_date={bad.get('signal_date')}, score_date={bad_score_date!r}"
        )
    normalized_score_dates = parsed_score_dates.dt.strftime("%Y-%m-%d").fillna("")
    orderable_work["score_event_date"] = normalized_score_dates.where(
        normalized_score_dates.ne(""), orderable_work.get("signal_date", "")
    )

    occurrence_keys = ["ticker", "trade_date", "signal_date"]
    missing_occurrence = [key for key in occurrence_keys if key not in orderable_work.columns]
    if missing_occurrence:
        raise ValueError(
            "策略replay orderable sidecar缺少occurrence欄位: "
            + ", ".join(missing_occurrence)
        )
    occurrence_score_event_counts = orderable_work.groupby(
        occurrence_keys, dropna=False
    )["score_event_date"].nunique(dropna=False)
    if bool((occurrence_score_event_counts > 1).any()):
        bad_key = occurrence_score_event_counts[occurrence_score_event_counts > 1].index[0]
        raise ValueError(
            "同一策略候選發生多個Breakout Quality score event date: "
            f"ticker={bad_key[0]}, trade_date={bad_key[1]}, signal_date={bad_key[2]}"
        )

    if selected_work is None:
        return orderable_work, None
    missing_selected = [key for key in occurrence_keys if key not in selected_work.columns]
    if missing_selected:
        raise ValueError(
            "策略replay selected sidecar缺少occurrence欄位: "
            + ", ".join(missing_selected)
        )
    occurrence_lookup = orderable_work[
        [*occurrence_keys, "score_event_date"]
    ].drop_duplicates(occurrence_keys, keep="first")
    selected_joined = selected_work.merge(
        occurrence_lookup,
        on=occurrence_keys,
        how="left",
        validate="many_to_one",
    )
    unresolved = selected_joined["score_event_date"].fillna("").astype(str).str.strip().eq("")
    if bool(unresolved.any()):
        bad = selected_joined.loc[unresolved].iloc[0]
        raise ValueError(
            "策略selected row無法對回orderable score event: "
            f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
            f"signal_date={bad.get('signal_date')}"
        )
    return orderable_work, selected_joined


def _strategy_selection_diagnostics(
    *, orderable: pd.DataFrame, selected: pd.DataFrame, lookup: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    orderable_work, selected_events = strategy_replay_score_event_frames(
        orderable, selected
    )
    if selected_events is None:
        raise ValueError("strategy selection diagnostics缺少selected replay rows")
    lookup_work = pd.DataFrame(lookup).copy()
    if "signal_date" in lookup_work.columns:
        lookup_work["signal_date"] = pd.to_datetime(
            lookup_work["signal_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("")

    lookup_work = lookup_work.rename(columns={
        "signal_date": "score_event_date",
        "breakout_quality_score": "pit_breakout_quality_score",
    })
    if bool(lookup_work.duplicated(["ticker", "score_event_date"]).any()):
        raise ValueError("Selection PIT diagnostic lookup同一ticker/score_event_date不唯一")

    orderable_joined = orderable_work.merge(
        lookup_work,
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )
    score_available = pd.to_numeric(
        orderable_joined.get(
            "pit_breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    runtime_score_available = pd.to_numeric(
        orderable_joined.get(
            "breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    declared_runtime_available = orderable_joined.get(
        "breakout_quality_score_available",
        pd.Series(False, index=orderable_joined.index),
    ).fillna(False).astype(bool)
    runtime_rows = declared_runtime_available | runtime_score_available
    if bool(runtime_rows.any()):
        expected = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "pit_breakout_quality_score"],
            errors="coerce",
        )
        actual = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "breakout_quality_score"],
            errors="coerce",
        )
        mismatch = (
            (~expected.map(math.isfinite))
            | (~actual.map(math.isfinite))
            | ((actual - expected).abs() > 1e-12)
        )
        if bool(mismatch.any()):
            bad_index = mismatch[mismatch].index[0]
            bad = orderable_joined.loc[bad_index]
            raise ValueError(
                "策略replay使用的Breakout Quality Score與PIT score table不一致: "
                f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
                f"signal_date={bad.get('signal_date')}, "
                f"score_event_date={bad.get('score_event_date')}, "
                f"runtime_score={actual.loc[bad_index]}, pit_score={expected.loc[bad_index]}"
            )

    target_available = orderable_joined.get(
        "target_available", pd.Series(False, index=orderable_joined.index)
    ).fillna(False).astype(bool)
    selected_joined = selected_events.merge(
        lookup_work,
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )

    day_rows = []
    valid_orderable = orderable_joined.loc[target_available].copy()
    for trade_date, day in valid_orderable.groupby("trade_date", sort=True):
        chosen = selected_joined[selected_joined["trade_date"] == trade_date]
        chosen_keys = set(zip(chosen["ticker"], chosen["signal_date"]))
        if not chosen_keys:
            continue
        day = day.drop_duplicates(["ticker", "signal_date"], keep="first").copy()
        day["target_percentile"] = day["target_raw_r"].rank(method="average", pct=True)
        selected_day = day[[
            (ticker, signal_date) in chosen_keys
            for ticker, signal_date in zip(day["ticker"], day["signal_date"])
        ]]
        if selected_day.empty:
            continue
        k = len(selected_day)
        top = day.nlargest(k, "target_raw_r", keep="first")
        top_keys = set(zip(top["ticker"], top["signal_date"]))
        retained = sum(key in top_keys for key in chosen_keys)
        day_rows.append({
            "trade_date": trade_date,
            "selected_count": k,
            "selected_target_mean_r": float(selected_day["target_raw_r"].mean()),
            "selected_target_percentile_mean": float(selected_day["target_percentile"].mean()),
            "target_top_k_retention": float(retained / k),
            "target_opportunity_gap_r": float(
                top["target_raw_r"].mean() - selected_day["target_raw_r"].mean()
            ),
        })
    daily = pd.DataFrame(day_rows)
    metrics = {
        "orderable_occurrences": int(len(orderable_joined)),
        "orderable_score_covered": int(score_available.sum()),
        "orderable_score_coverage_rate": float(score_available.mean()) if len(score_available) else None,
        "runtime_scored_orderable_occurrences": int(runtime_score_available.sum()),
        "runtime_score_identity_match": True,
        "orderable_target_covered": int(target_available.sum()),
        "orderable_target_coverage_rate": float(target_available.mean()) if len(target_available) else None,
        "selected_buy_rows": int(len(selected_joined)),
        "diagnostic_days": int(len(daily)),
        "selected_target_percentile_mean": float(daily["selected_target_percentile_mean"].mean()) if not daily.empty else None,
        "target_top_k_retention_mean": float(daily["target_top_k_retention"].mean()) if not daily.empty else None,
        "target_opportunity_gap_r_mean": float(daily["target_opportunity_gap_r"].mean()) if not daily.empty else None,
        "selected_target_mean_r": float(daily["selected_target_mean_r"].mean()) if not daily.empty else None,
        "future_target_used_for_runtime_sort": False,
    }
    return metrics, orderable_joined, selected_joined


def paired_trade_r_conversion_diagnostic(
    baseline_trade_history: pd.DataFrame,
    active_trade_history: pd.DataFrame,
    baseline_selected: pd.DataFrame,
    active_selected: pd.DataFrame,
) -> dict[str, Any]:
    """Measure Target-R -> realized-R conversion on the same exclusive trades.

    Both numerator and denominator use the canonical closed-trade partition
    (active-only versus baseline-only).  Future Target is joined only after
    replay using the existing selected-target sidecars.  This prevents trade
    count / fill differences from contaminating an efficiency ratio whose
    intent is to measure quality-edge realization.
    """

    baseline_trades = reconstruct_round_trips(
        pd.DataFrame(baseline_trade_history), scenario="baseline"
    )
    active_trades = reconstruct_round_trips(
        pd.DataFrame(active_trade_history), scenario="active"
    )
    baseline_keys = set(baseline_trades.get("match_key", pd.Series(dtype=str)).astype(str))
    active_keys = set(active_trades.get("match_key", pd.Series(dtype=str)).astype(str))
    baseline_only = baseline_trades.loc[
        baseline_trades["match_key"].astype(str).isin(baseline_keys - active_keys)
    ].copy()
    active_only = active_trades.loc[
        active_trades["match_key"].astype(str).isin(active_keys - baseline_keys)
    ].copy()

    join_keys = ["ticker", "trade_date", "signal_date"]
    required_target = [*join_keys, "target_raw_r", "target_available"]

    def normalize_targets(raw: pd.DataFrame, label: str) -> pd.DataFrame:
        frame = pd.DataFrame(raw).copy()
        missing = [column for column in required_target if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} selected target diagnostics缺欄位: {missing}")
        for column in ("trade_date", "signal_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
        if bool(frame.duplicated(join_keys).any()):
            bad = frame.loc[frame.duplicated(join_keys, keep=False), join_keys].iloc[0].to_dict()
            raise ValueError(f"{label} selected target diagnostics selection key不唯一: {bad}")
        frame["target_raw_r"] = pd.to_numeric(frame["target_raw_r"], errors="coerce")
        available = frame["target_available"]
        if available.dtype == bool:
            frame["target_available"] = available.fillna(False)
        else:
            frame["target_available"] = (
                available.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            )
        return frame[required_target]

    def join_targets(trades: pd.DataFrame, targets: pd.DataFrame, label: str) -> pd.DataFrame:
        work = pd.DataFrame(trades).copy()
        if work.empty:
            work["trade_date"] = pd.Series(dtype=str)
            work["target_raw_r"] = pd.Series(dtype=float)
            work["target_available"] = pd.Series(dtype=bool)
            return work
        work["ticker"] = work["ticker"].fillna("").astype(str).str.strip()
        work["trade_date"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        work["signal_date"] = pd.to_datetime(work["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        return work.merge(
            targets, on=join_keys, how="left", validate="many_to_one", suffixes=("", "_target")
        )

    baseline_target = normalize_targets(baseline_selected, "baseline")
    active_target = normalize_targets(active_selected, "active")
    baseline_only = join_targets(baseline_only, baseline_target, "baseline")
    active_only = join_targets(active_only, active_target, "active")
    all_exclusive = pd.concat([baseline_only, active_only], ignore_index=True)

    def target_covered(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        available = frame["target_available"].fillna(False).astype(bool)
        finite_target = pd.to_numeric(frame["target_raw_r"], errors="coerce").map(math.isfinite)
        finite_realized = pd.to_numeric(frame["r_multiple"], errors="coerce").map(math.isfinite)
        return frame.loc[available & finite_target & finite_realized].copy()

    baseline_covered = target_covered(baseline_only)
    active_covered = target_covered(active_only)
    covered_count = int(len(baseline_covered) + len(active_covered))
    coverage = (
        1.0 if all_exclusive.empty else float(covered_count / len(all_exclusive))
    )
    complete_target_coverage = int(len(all_exclusive)) == covered_count

    # RCE must compare Target-R and realized-R on the exact same observable trade subset.
    # Missing Future Target rows are excluded from both numerator and denominator rather
    # than forcing an all-or-nothing 100% coverage gate.  Coverage remains explicit
    # diagnostic evidence, so no hidden minimum-coverage magic threshold is introduced.
    baseline_covered_realized = pd.to_numeric(
        baseline_covered.get("r_multiple"), errors="coerce"
    ).dropna().astype(float)
    active_covered_realized = pd.to_numeric(
        active_covered.get("r_multiple"), errors="coerce"
    ).dropna().astype(float)
    baseline_all_realized = pd.to_numeric(
        baseline_only.get("r_multiple"), errors="coerce"
    ).dropna().astype(float)
    active_all_realized = pd.to_numeric(
        active_only.get("r_multiple"), errors="coerce"
    ).dropna().astype(float)

    realized_edge = None
    target_edge = None
    baseline_target_mean = None
    active_target_mean = None
    if len(baseline_covered) and len(active_covered):
        baseline_target_mean = float(
            pd.to_numeric(baseline_covered["target_raw_r"], errors="coerce").mean()
        )
        active_target_mean = float(
            pd.to_numeric(active_covered["target_raw_r"], errors="coerce").mean()
        )
        target_edge = float(active_target_mean - baseline_target_mean)
        realized_edge = float(
            active_covered_realized.mean() - baseline_covered_realized.mean()
        )
    rce = (
        None
        if target_edge is None or target_edge <= 1e-12 or realized_edge is None
        else float(realized_edge / target_edge)
    )
    baseline_total_r = float(baseline_all_realized.sum()) if len(baseline_all_realized) else 0.0
    active_total_r = float(active_all_realized.sum()) if len(active_all_realized) else 0.0
    return {
        "comparison_basis": "target_covered_exclusive_realized_trade_mean_r",
        "active_only_count": int(len(active_only)),
        "baseline_only_count": int(len(baseline_only)),
        "active_only_target_covered_count": int(len(active_covered)),
        "baseline_only_target_covered_count": int(len(baseline_covered)),
        "target_covered_exclusive_count": covered_count,
        "target_coverage_rate": coverage,
        "complete_target_coverage": bool(complete_target_coverage),
        "active_only_target_mean_r": active_target_mean,
        "baseline_only_target_mean_r": baseline_target_mean,
        "paired_target_selection_edge_r": target_edge,
        "active_only_realized_mean_r": (
            float(active_covered_realized.mean()) if len(active_covered_realized) else None
        ),
        "baseline_only_realized_mean_r": (
            float(baseline_covered_realized.mean()) if len(baseline_covered_realized) else None
        ),
        "paired_realized_selection_edge_r": realized_edge,
        "active_only_realized_total_r": active_total_r,
        "baseline_only_realized_total_r": baseline_total_r,
        "exclusive_selection_delta_r": float(active_total_r - baseline_total_r),
        "r_conversion_efficiency": rce,
        "future_target_used_for_runtime_sort": False,
    }



def backfill_pair_selection_diagnostics(
    payload: dict[str, Any],
    *,
    pair_dir: Path,
    project_root: Path,
    active_trades_filename: str = "score_ranking_trades.csv",
) -> bool:
    """Backfill post-replay Forward selection diagnostics from canonical sidecars.

    This is primarily for cache reuse after the Forward diagnostic contract was
    expanded. It never reruns portfolio replay and never feeds Future Target back
    into runtime ranking.
    """

    metadata = dict(payload.get("metadata") or {})
    if (
        str(metadata.get("comparison_mode") or "") != COMPARISON_MODE_SCORE_RANKING
        or str(metadata.get("score_source") or "") != SCORE_SOURCE_CONTINUOUS_RANKER_OOS
    ):
        return False
    existing = dict(payload.get("selection_diagnostics") or {})
    active_existing = dict(existing.get("score_ranking") or {})
    if finite_number(active_existing.get("orderable_score_coverage_rate")) is not None:
        return False

    score_path_raw = str(metadata.get("score_path") or "").strip()
    profile = str(metadata.get("experiment_profile") or "").strip()
    filter_id = str(metadata.get("filter_id") or "").strip()
    architecture = str(metadata.get("model_architecture") or "").strip()
    if not score_path_raw or not profile or not filter_id or not architecture:
        return False
    score_path = Path(score_path_raw)
    if not score_path.is_absolute():
        score_path = (Path(project_root) / score_path).resolve()
    if not score_path.is_file():
        return False

    baseline_orderable_path = Path(pair_dir) / "no_filter_orderable_candidates.csv"
    active_orderable_path = Path(pair_dir) / "score_ranking_orderable_candidates.csv"
    baseline_selected_path = Path(pair_dir) / "no_filter_selected_buys.csv"
    active_selected_path = Path(pair_dir) / "score_ranking_selected_buys.csv"
    baseline_trades_path = Path(pair_dir) / "no_filter_trades.csv"
    active_trades_path = Path(pair_dir) / str(active_trades_filename)
    required_paths = (
        baseline_orderable_path, active_orderable_path,
        baseline_selected_path, active_selected_path,
        baseline_trades_path, active_trades_path,
    )
    if not all(path.is_file() for path in required_paths):
        return False

    lookup = _continuous_forward_target_lookup(
        root=Path(project_root),
        filter_id=filter_id,
        architecture=architecture,
        profile=profile,
        score_path_override=str(score_path),
        score_column=str(
            (metadata.get("score_ranking_options") or {}).get("primary_score_column")
            or "model_score"
        ),
    )
    baseline_diag, baseline_orderable_joined, baseline_selected_joined = (
        _strategy_selection_diagnostics(
            orderable=pd.read_csv(baseline_orderable_path, encoding="utf-8-sig"),
            selected=pd.read_csv(baseline_selected_path, encoding="utf-8-sig"),
            lookup=lookup,
        )
    )
    active_diag, active_orderable_joined, active_selected_joined = (
        _strategy_selection_diagnostics(
            orderable=pd.read_csv(active_orderable_path, encoding="utf-8-sig"),
            selected=pd.read_csv(active_selected_path, encoding="utf-8-sig"),
            lookup=lookup,
        )
    )
    delta_keys = set(baseline_diag) | set(active_diag)
    diagnostic_delta = {}
    for key in delta_keys:
        left = finite_number(active_diag.get(key))
        right = finite_number(baseline_diag.get(key))
        diagnostic_delta[key] = None if left is None or right is None else left - right
    diagnostics = {
        "no_filter": baseline_diag,
        "score_ranking": active_diag,
        "score_ranking_minus_no_filter": diagnostic_delta,
        "selection_r_conversion": paired_trade_r_conversion_diagnostic(
            pd.read_csv(baseline_trades_path, encoding="utf-8-sig"),
            pd.read_csv(active_trades_path, encoding="utf-8-sig"),
            baseline_selected_joined,
            active_selected_joined,
        ),
        "future_target_join_stage": "post_replay_offline_diagnostic_only",
        "future_target_used_for_runtime_sort": False,
    }
    payload["selection_diagnostics"] = diagnostics

    baseline_orderable_joined.to_csv(
        Path(pair_dir) / "no_filter_orderable_target_diagnostics.csv",
        index=False, encoding="utf-8-sig",
    )
    active_orderable_joined.to_csv(
        Path(pair_dir) / "score_ranking_orderable_target_diagnostics.csv",
        index=False, encoding="utf-8-sig",
    )
    baseline_selected_joined.to_csv(
        Path(pair_dir) / "no_filter_selected_target_diagnostics.csv",
        index=False, encoding="utf-8-sig",
    )
    active_selected_joined.to_csv(
        Path(pair_dir) / "score_ranking_selected_target_diagnostics.csv",
        index=False, encoding="utf-8-sig",
    )
    return True

def backfill_pair_r_conversion_diagnostic(
    payload: dict[str, Any],
    *,
    pair_dir: Path,
    active_trades_filename: str = "score_ranking_trades.csv",
) -> bool:
    """Backfill realized-trade RCE from existing canonical pair sidecars on RUN/REUSE."""

    diagnostics = dict(payload.get("selection_diagnostics") or {})
    existing = dict(diagnostics.get("selection_r_conversion") or {})
    if finite_number(existing.get("r_conversion_efficiency")) is not None and str(
        existing.get("comparison_basis") or ""
    ) == "target_covered_exclusive_realized_trade_mean_r":
        return False
    baseline_target_path = Path(pair_dir) / "no_filter_selected_target_diagnostics.csv"
    active_target_path = Path(pair_dir) / "score_ranking_selected_target_diagnostics.csv"
    baseline_trades_path = Path(pair_dir) / "no_filter_trades.csv"
    active_trades_path = Path(pair_dir) / str(active_trades_filename)
    required_paths = (baseline_target_path, active_target_path, baseline_trades_path, active_trades_path)
    if not all(path.is_file() for path in required_paths):
        return False
    conversion = paired_trade_r_conversion_diagnostic(
        pd.read_csv(baseline_trades_path, encoding="utf-8-sig"),
        pd.read_csv(active_trades_path, encoding="utf-8-sig"),
        pd.read_csv(baseline_target_path, encoding="utf-8-sig"),
        pd.read_csv(active_target_path, encoding="utf-8-sig"),
    )
    diagnostics["selection_r_conversion"] = conversion
    payload["selection_diagnostics"] = diagnostics
    return True


def _upside_realization_is_current(payload: dict[str, Any]) -> bool:
    diagnostics = dict(payload.get("selection_diagnostics") or {})
    existing = dict(diagnostics.get("upside_realization") or {})
    return (
        str(existing.get("status") or "") == "READY"
        and int(existing.get("schema_version") or 0) == UPSIDE_REALIZATION_SCHEMA_VERSION
        and int(diagnostics.get("diagnostics_schema_version") or 0) == STRATEGY_DIAGNOSTICS_SCHEMA_VERSION
        and str(existing.get("contract_fingerprint") or "") == _upside_realization_contract_fingerprint()
    )


def _diagnostic_error_reason(exc: BaseException, *, project_root: Path) -> str:
    """Return a concise user-visible reason without leaking machine-specific root paths."""

    text = f"{type(exc).__name__}: {exc}"
    root_text = str(Path(project_root).resolve())
    variants = {root_text, root_text.replace("\\", "/")}
    for value in sorted((item for item in variants if item), key=len, reverse=True):
        text = text.replace(value, ".")
    return text.replace("\\", "/")


def pair_upside_realization_refresh_required(pair_dir: Path) -> bool:
    """Return whether a reusable score-ranking pair needs diagnostic/report backfill."""

    payload = _read_json(Path(pair_dir) / "strategy_comparison.json")
    if not isinstance(payload, dict):
        return True
    metadata = dict(payload.get("metadata") or {})
    if str(metadata.get("comparison_mode") or "") != COMPARISON_MODE_SCORE_RANKING:
        return False
    return not _upside_realization_is_current(payload)


def backfill_pair_upside_realization_diagnostic(
    payload: dict[str, Any],
    *,
    pair_dir: Path,
    project_root: Path,
    active_trades_filename: str = "score_ranking_trades.csv",
) -> bool:
    """Refresh read-only path-conversion diagnostics without replaying the strategy."""

    metadata = dict(payload.get("metadata") or {})
    if str(metadata.get("comparison_mode") or "") != COMPARISON_MODE_SCORE_RANKING:
        return False
    contract_fingerprint = _upside_realization_contract_fingerprint()
    diagnostics = dict(payload.get("selection_diagnostics") or {})
    if _upside_realization_is_current(payload):
        pair_metadata = dict(payload.get("metadata") or {})
        pair_metadata["diagnostics_schema_version"] = STRATEGY_DIAGNOSTICS_SCHEMA_VERSION
        pair_metadata["diagnostics_contract_fingerprint"] = contract_fingerprint
        payload["metadata"] = pair_metadata
        return False

    filter_id = str(metadata.get("filter_id") or "").strip()
    architecture = str(metadata.get("model_architecture") or "").strip()
    if not filter_id or not architecture:
        return False

    baseline_target_path = Path(pair_dir) / "no_filter_selected_target_diagnostics.csv"
    active_target_path = Path(pair_dir) / "score_ranking_selected_target_diagnostics.csv"
    baseline_trades_path = Path(pair_dir) / "no_filter_trades.csv"
    active_trades_path = Path(pair_dir) / str(active_trades_filename)
    required_paths = (
        baseline_target_path, active_target_path, baseline_trades_path, active_trades_path,
    )
    if not all(path.is_file() for path in required_paths):
        return False

    try:
        baseline_selected = _join_full_horizon_path_diagnostics(
            pd.read_csv(baseline_target_path, encoding="utf-8-sig"),
            project_root=Path(project_root),
            filter_id=filter_id,
            architecture=architecture,
        )
        active_selected = _join_full_horizon_path_diagnostics(
            pd.read_csv(active_target_path, encoding="utf-8-sig"),
            project_root=Path(project_root),
            filter_id=filter_id,
            architecture=architecture,
        )
        baseline_result = build_upside_realization_attribution(
            trade_history=pd.read_csv(baseline_trades_path, encoding="utf-8-sig"),
            selected_path_diagnostics=baseline_selected,
            upside_r_thresholds=tuple(STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS),
            adverse_bucket_edges_r=tuple(STRATEGY_COMPARE_UPSIDE_REALIZATION_ADVERSE_BUCKET_EDGES_R),
            scenario="no_filter",
        )
        active_result = build_upside_realization_attribution(
            trade_history=pd.read_csv(active_trades_path, encoding="utf-8-sig"),
            selected_path_diagnostics=active_selected,
            upside_r_thresholds=tuple(STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS),
            adverse_bucket_edges_r=tuple(STRATEGY_COMPARE_UPSIDE_REALIZATION_ADVERSE_BUCKET_EDGES_R),
            scenario="score_ranking",
        )
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        diagnostics["upside_realization"] = {
            "status": "UNAVAILABLE",
            "schema_version": UPSIDE_REALIZATION_SCHEMA_VERSION,
            "contract_fingerprint": contract_fingerprint,
            "contract": _upside_realization_contract(),
            "reason": _diagnostic_error_reason(exc, project_root=Path(project_root)),
        }
        diagnostics["diagnostics_schema_version"] = STRATEGY_DIAGNOSTICS_SCHEMA_VERSION
        payload["selection_diagnostics"] = diagnostics
        pair_metadata = dict(payload.get("metadata") or {})
        pair_metadata["diagnostics_schema_version"] = STRATEGY_DIAGNOSTICS_SCHEMA_VERSION
        pair_metadata["diagnostics_contract_fingerprint"] = contract_fingerprint
        payload["metadata"] = pair_metadata
        return True

    baseline_selected.to_csv(baseline_target_path, index=False, encoding="utf-8-sig")
    active_selected.to_csv(active_target_path, index=False, encoding="utf-8-sig")
    baseline_result["detail"].to_csv(
        Path(pair_dir) / "no_filter_upside_realization_trades.csv",
        index=False, encoding="utf-8-sig",
    )
    active_result["detail"].to_csv(
        Path(pair_dir) / "score_ranking_upside_realization_trades.csv",
        index=False, encoding="utf-8-sig",
    )
    diagnostics["upside_realization"] = {
        "status": "READY",
        "schema_version": UPSIDE_REALIZATION_SCHEMA_VERSION,
        "contract_fingerprint": contract_fingerprint,
        "contract": _upside_realization_contract(),
        "no_filter": baseline_result["summary"],
        "score_ranking": active_result["summary"],
        "score_ranking_joint_rows": active_result["joint_rows"],
        "future_target_join_stage": "post_replay_offline_diagnostic_only",
        "future_target_used_for_runtime_sort": False,
    }
    diagnostics["diagnostics_schema_version"] = STRATEGY_DIAGNOSTICS_SCHEMA_VERSION
    payload["selection_diagnostics"] = diagnostics
    pair_metadata = dict(payload.get("metadata") or {})
    pair_metadata["diagnostics_schema_version"] = STRATEGY_DIAGNOSTICS_SCHEMA_VERSION
    pair_metadata["diagnostics_contract_fingerprint"] = contract_fingerprint
    payload["metadata"] = pair_metadata
    return True


# Stable public aliases for read-only consumers.
flatten_candidate_replay_rows = _flatten_candidate_replay_rows

# Human-readable aggregate report reuse layer.
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_status_file(root: Path, status: dict[str, Any], dl_id: str, key: str) -> Path | None:
    row = (((status.get("dl_sources") or {}).get(dl_id) or {}).get("files") or {}).get(key) or {}
    raw = str(row.get("path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (root / path).resolve()


def _selection_pit_prediction_row(
    *, root: Path, status: dict[str, Any], dl_id: str, source: Any
) -> dict[str, Any]:
    path = _resolve_status_file(root, status, dl_id, "audit")
    payload = _read_json(path) if path is not None else None
    if payload is None:
        return {
            "dl_id": dl_id,
            "score_source": source.score_source,
            "scope": "-",
            "artifact": None if path is None else project_relative_display_path(path, project_root=root),
            "available": False,
        }
    contract = dict(payload.get("decision_contract") or {})
    scope = str(contract.get("primary_metric_scope") or "all_valid_target")
    metrics = dict((payload.get("metrics") or {}).get(scope) or {})
    direction = dict(payload.get("direction_summary") or {})
    return {
        "dl_id": dl_id,
        "score_source": source.score_source,
        "scope": str(contract.get("primary_metric_label") or scope),
        "continuous_target_id": str(payload.get("continuous_target_id") or ""),
        "global_spearman": finite_number(metrics.get("global_spearman")),
        "mean_daily_spearman": finite_number(metrics.get("mean_daily_spearman")),
        "pairwise_concordance": finite_number(metrics.get("pairwise_concordance")),
        "top_target_r": finite_number(metrics.get("top_decile_target_mean")),
        "bottom_target_r": finite_number(metrics.get("bottom_decile_target_mean")),
        "top_bottom_target_spread_r": finite_number(metrics.get("top_bottom_target_spread")),
        "valid_year_count": int(direction.get("valid_year_count") or 0),
        "positive_spearman_year_count": int(direction.get("positive_spearman_year_count") or 0),
        "positive_spearman_year_rate": finite_number(direction.get("positive_spearman_year_rate")),
        "artifact": project_relative_display_path(path, project_root=root),
        "available": True,
    }


def _continuous_prediction_row(
    *, root: Path, status: dict[str, Any], dl_id: str, source: Any
) -> dict[str, Any]:
    path = _resolve_status_file(root, status, dl_id, "report")
    payload = _read_json(path) if path is not None else None
    if payload is None:
        return {
            "dl_id": dl_id,
            "score_source": source.score_source,
            "scope": "Forward OOS",
            "artifact": None if path is None else project_relative_display_path(path, project_root=root),
            "available": False,
        }
    metrics = dict((payload.get("split_metrics") or {}).get("oos") or {})
    top = finite_number(metrics.get("top_score_decile_raw_target_mean"))
    bottom = finite_number(metrics.get("bottom_score_decile_raw_target_mean"))
    spread = None if top is None or bottom is None else top - bottom
    return {
        "dl_id": dl_id,
        "score_source": source.score_source,
        "scope": "Forward OOS all eligible stock-days",
        "continuous_target_id": str(payload.get("continuous_target_id") or ""),
        "global_spearman": finite_number(metrics.get("global_spearman_vs_raw_target")),
        "mean_daily_spearman": finite_number(metrics.get("mean_daily_spearman")),
        "pairwise_concordance": finite_number(metrics.get("pairwise_concordance")),
        "top_target_r": top,
        "bottom_target_r": bottom,
        "top_bottom_target_spread_r": spread,
        "valid_year_count": None,
        "positive_spearman_year_count": None,
        "positive_spearman_year_rate": None,
        "artifact": project_relative_display_path(path, project_root=root),
        "available": True,
    }


def _model_prediction_rows(
    *, root: Path, settings: StrategyComparisonSettings, status: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for arm in settings.enabled_arms:
        if not arm.dl_enabled or not arm.dl_id or arm.dl_id in seen:
            continue
        seen.add(arm.dl_id)
        source = settings.dl_sources[arm.dl_id]
        if source.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            row = _selection_pit_prediction_row(root=root, status=status, dl_id=arm.dl_id, source=source)
        elif source.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            row = _continuous_prediction_row(root=root, status=status, dl_id=arm.dl_id, source=source)
        else:
            row = {
                "dl_id": arm.dl_id,
                "score_source": source.score_source,
                "scope": "-",
                "artifact": None,
                "available": False,
            }
        rows.append(row)
    return rows


def _selection_translation_rows(
    *, settings: StrategyComparisonSettings, pair_payloads: dict[str, dict[str, Any]], scenarios: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_arm: dict[str, dict[str, Any]] = {}
    for pair in pair_payloads.values():
        _param_source, _rule_policy, _off_arm, on_arm = pair["arm_contract"]
        if on_arm is None:
            continue
        payload = dict(pair.get("payload") or {})
        diagnostics = dict(payload.get("selection_diagnostics") or {})
        if not diagnostics:
            continue
        mode = str((payload.get("metadata") or {}).get("comparison_mode") or "")
        active_name = comparison_labels(mode)["active_name"]
        baseline = dict(diagnostics.get("no_filter") or {})
        active = dict(diagnostics.get(active_name) or {})
        if not active:
            continue

        def delta(key: str) -> float | None:
            left = finite_number(active.get(key))
            right = finite_number(baseline.get(key))
            return None if left is None or right is None else left - right

        scenario = scenarios.get(on_arm.arm_id) or {}
        conversion = dict(diagnostics.get("selection_r_conversion") or {})
        target_selection_delta_r = finite_number(
            conversion.get("paired_target_selection_edge_r")
        )
        realized_selection_delta_r = finite_number(scenario.get("direct_selection_delta_r"))
        r_conversion_efficiency = finite_number(conversion.get("r_conversion_efficiency"))
        realized_selection_edge_r = finite_number(
            conversion.get("paired_realized_selection_edge_r")
        )
        by_arm[on_arm.arm_id] = {
            "arm_id": on_arm.arm_id,
            "name": on_arm.name,
            "dl_id": on_arm.dl_id,
            "score_coverage": finite_number(active.get("orderable_score_coverage_rate")),
            "paired_target_selection_edge_r": target_selection_delta_r,
            "paired_realized_selection_edge_r": realized_selection_edge_r,
            "r_conversion_efficiency": r_conversion_efficiency,
            "selected_target_mean_r": finite_number(active.get("selected_target_mean_r")),
            "selected_target_mean_r_delta": delta("selected_target_mean_r"),
            "selected_target_percentile": finite_number(active.get("selected_target_percentile_mean")),
            "selected_target_percentile_delta": delta("selected_target_percentile_mean"),
            "target_top_k_retention": finite_number(active.get("target_top_k_retention_mean")),
            "target_top_k_retention_delta": delta("target_top_k_retention_mean"),
            "target_opportunity_gap_r": finite_number(active.get("target_opportunity_gap_r_mean")),
            "target_opportunity_gap_r_delta": delta("target_opportunity_gap_r_mean"),
            "direct_selection_delta_r": finite_number(scenario.get("direct_selection_delta_r")),
        }
    return [by_arm[arm.arm_id] for arm in settings.enabled_arms if arm.arm_id in by_arm]



def _upside_realization_rows(
    *, settings: StrategyComparisonSettings, pair_payloads: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: dict[str, dict[str, Any]] = {}
    joint_rows: list[dict[str, Any]] = []
    unavailable_rows: list[dict[str, Any]] = []
    for pair in pair_payloads.values():
        _param_source, _rule_policy, _off_arm, on_arm = pair["arm_contract"]
        if on_arm is None or not bool(on_arm.dl_enabled):
            continue
        payload = dict(pair.get("payload") or {})
        diagnostics = dict(payload.get("selection_diagnostics") or {})
        realization = dict(diagnostics.get("upside_realization") or {})
        realization_status = str(realization.get("status") or "").strip()
        if realization_status != "READY":
            if realization_status:
                unavailable_rows.append({
                    "arm_id": on_arm.arm_id,
                    "name": on_arm.name,
                    "status": realization_status,
                    "reason": str(realization.get("reason") or "未提供原因"),
                })
            continue
        active = dict(realization.get("score_ranking") or {})
        if not active:
            continue
        summaries[on_arm.arm_id] = {
            "arm_id": on_arm.arm_id,
            "name": on_arm.name,
            **active,
        }
        for row in list(realization.get("score_ranking_joint_rows") or []):
            joint_rows.append({
                "arm_id": on_arm.arm_id,
                "name": on_arm.name,
                **dict(row or {}),
            })
    ordered = [
        summaries[arm.arm_id]
        for arm in settings.enabled_arms
        if arm.arm_id in summaries
    ]
    arm_order = {arm.arm_id: index for index, arm in enumerate(settings.enabled_arms)}
    joint_rows.sort(key=lambda row: (
        arm_order.get(str(row.get("arm_id") or ""), 10**9),
        str(row.get("mfe_bucket") or ""),
        str(row.get("adverse_bucket") or ""),
    ))
    unavailable_rows.sort(key=lambda row: arm_order.get(str(row.get("arm_id") or ""), 10**9))
    return ordered, joint_rows, unavailable_rows


def _execution_rows(
    *, settings: StrategyComparisonSettings, scenarios: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for arm in settings.enabled_arms:
        scenario = scenarios[arm.arm_id]
        rows.append({
            "arm_id": arm.arm_id,
            "name": arm.name,
            "avg_exposure_pct": finite_number(scenario.get("avg_exposure_pct")),
            "reserved_buy_fill_rate_pct": finite_number(scenario.get("reserved_buy_fill_rate_pct")),
            "avg_orderable_candidates": finite_number(scenario.get("avg_orderable_candidates")),
            "candidate_supply_gap_days": finite_number(scenario.get("candidate_supply_gap_days")),
            "underfilled_end_days": finite_number(scenario.get("underfilled_end_days")),
            "end_position_gap_slot_days": finite_number(scenario.get("end_position_gap_slot_days")),
        })
    return rows


def _r_analysis_rows(
    *,
    settings: StrategyComparisonSettings,
    scenarios: dict[str, dict[str, Any]],
    model_prediction: list[dict[str, Any]],
    selection_translation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join canonical model/selection diagnostics to canonical strategy R outcomes."""

    model_by_dl = {
        str(row.get("dl_id") or ""): dict(row)
        for row in model_prediction
        if str(row.get("dl_id") or "")
    }
    translation_by_arm = {
        str(row.get("arm_id") or ""): dict(row)
        for row in selection_translation
        if str(row.get("arm_id") or "")
    }
    rows: list[dict[str, Any]] = []
    for arm in settings.enabled_arms:
        scenario = dict(scenarios.get(arm.arm_id) or {})
        model = dict(model_by_dl.get(str(arm.dl_id or "")) or {})
        translation = dict(translation_by_arm.get(arm.arm_id) or {})
        rows.append({
            "arm_id": arm.arm_id,
            "name": arm.name,
            "dl_id": arm.dl_id,
            "continuous_target_id": str(model.get("continuous_target_id") or ""),
            "portfolio_avg_r": finite_number(scenario.get("portfolio_avg_r")),
            "portfolio_median_r": finite_number(scenario.get("portfolio_median_r")),
            "mean_daily_spearman": finite_number(model.get("mean_daily_spearman")),
            "global_spearman": finite_number(model.get("global_spearman")),
            "pairwise_concordance": finite_number(model.get("pairwise_concordance")),
            "top_target_r": finite_number(model.get("top_target_r")),
            "bottom_target_r": finite_number(model.get("bottom_target_r")),
            "top_bottom_target_spread_r": finite_number(model.get("top_bottom_target_spread_r")),
            "score_coverage": finite_number(translation.get("score_coverage")),
            "paired_target_selection_edge_r": finite_number(translation.get("paired_target_selection_edge_r")),
            "r_conversion_efficiency": finite_number(translation.get("r_conversion_efficiency")),
            "selected_target_mean_r": finite_number(translation.get("selected_target_mean_r")),
            "selected_target_mean_r_delta": finite_number(translation.get("selected_target_mean_r_delta")),
            "selected_target_percentile": finite_number(translation.get("selected_target_percentile")),
            "selected_target_percentile_delta": finite_number(translation.get("selected_target_percentile_delta")),
            "target_top_k_retention": finite_number(translation.get("target_top_k_retention")),
            "target_top_k_retention_delta": finite_number(translation.get("target_top_k_retention_delta")),
            "target_opportunity_gap_r": finite_number(translation.get("target_opportunity_gap_r")),
            "target_opportunity_gap_r_delta": finite_number(translation.get("target_opportunity_gap_r_delta")),
            "direct_selection_delta_r": finite_number(translation.get("direct_selection_delta_r")),
        })
    return rows



def _mfe_safety_pair_dir(payload: dict[str, Any], *, project_root: Path) -> Path:
    metadata = dict(payload.get("metadata") or {})
    raw = str(metadata.get("output_dir") or "").strip()
    if not raw:
        raise ValueError("Strategy Compare pair metadata缺少output_dir，無法建立MFE/Safety主報表")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def _filled_selection_geometry_keys(
    *,
    pair_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    orderable_path = Path(pair_dir) / f"{prefix}_orderable_candidates.csv"
    selected_path = Path(pair_dir) / f"{prefix}_selected_buys.csv"
    if not orderable_path.is_file() or not selected_path.is_file():
        raise ValueError(
            "Strategy Compare MFE/Safety主報表缺少既有selection sidecar: "
            f"{orderable_path.name} / {selected_path.name}"
        )
    orderable = pd.read_csv(orderable_path, encoding="utf-8-sig", low_memory=False)
    selected = pd.read_csv(selected_path, encoding="utf-8-sig", low_memory=False)
    _orderable_events, selected_events = strategy_replay_score_event_frames(orderable, selected)
    if selected_events is None or selected_events.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    keys = pd.DataFrame({
        "ticker": selected_events.get("ticker", pd.Series("", index=selected_events.index)).map(
            normalize_geometry_ticker
        ),
        "date": selected_events.get(
            "score_event_date", pd.Series("", index=selected_events.index)
        ).map(normalize_geometry_date),
    })
    keys = keys.loc[keys["ticker"].ne("") & keys["date"].ne("")]
    return keys.drop_duplicates(["ticker", "date"]).sort_values(
        ["date", "ticker"], kind="stable"
    ).reset_index(drop=True)


def _build_mfe_safety_main_report_geometry(
    *,
    project_root: Path,
    settings: StrategyComparisonSettings,
    pair_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_ENABLED:
        return {"status": "DISABLED"}
    root = Path(project_root).resolve()
    if settings.profile_id not in set(STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PROFILE_IDS):
        return {"status": "NOT_APPLICABLE", "profile_id": settings.profile_id}

    arm_keys: dict[str, pd.DataFrame] = {}
    periods: set[tuple[str, str]] = set()
    enabled_arm_ids = {item.arm_id for item in settings.enabled_arms}

    for pair in pair_payloads.values():
        _param_source, _rule_policy, off_arm, on_arm = pair["arm_contract"]
        payload = dict(pair["payload"])
        metadata = dict(payload.get("metadata") or {})
        period = dict(metadata.get("comparison_period") or {})
        start = normalize_geometry_date(period.get("start"))
        end = normalize_geometry_date(period.get("end"))
        if not start or not end:
            raise ValueError("Strategy Compare pair metadata缺少MFE/Safety comparison period")
        periods.add((start, end))
        pair_dir = _mfe_safety_pair_dir(payload, project_root=root)
        for arm, prefix in ((off_arm, "no_filter"), (on_arm, "score_ranking")):
            if arm is None or arm.arm_id not in enabled_arm_ids:
                continue
            keys = _filled_selection_geometry_keys(pair_dir=pair_dir, prefix=prefix)
            existing = arm_keys.get(arm.arm_id)
            if existing is None:
                arm_keys[arm.arm_id] = keys
            else:
                left = set(map(tuple, existing[["ticker", "date"]].to_numpy()))
                right = set(map(tuple, keys[["ticker", "date"]].to_numpy()))
                if left != right:
                    raise ValueError(
                        f"Strategy Compare重複baseline arm的Filled selection不一致: {arm.arm_id}"
                    )

    if len(periods) != 1:
        raise ValueError(
            "Strategy Compare MFE/Safety geometry period不一致: " + repr(sorted(periods))
        )
    missing_arms = [arm.arm_id for arm in settings.enabled_arms if arm.arm_id not in arm_keys]
    if missing_arms:
        raise ValueError(f"Strategy Compare MFE/Safety geometry缺少arm Filled selection: {missing_arms}")

    filter_id = BREAKOUT_QUALITY_WORKFLOW_FILTER_ID
    architecture = BREAKOUT_QUALITY_WORKFLOW_MODEL_ARCHITECTURE
    start, end = next(iter(periods))
    truth, source = build_mfe_safety_truth_geometry(
        project_root=root,
        filter_id=filter_id,
        model_architecture=architecture,
        provider_profile_id=STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PROFILE,
        mfe_target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        safety_target_id=DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
        percentile_method=STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_METHOD,
    )
    truth = attach_mfe_safety_quadrants(
        filter_mfe_safety_period(truth, start, end),
        cutoff=STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_CUTOFF,
    )
    if truth.empty:
        raise ValueError("Strategy Compare period內沒有MFE/Safety canonical truth")
    breakout_keys_raw = load_official_breakout_candidate_keys(
        filter_id,
        allow_stale_source=False,
    )
    breakout_keys = pd.DataFrame(
        [
            (normalize_geometry_ticker(ticker), normalize_geometry_date(date))
            for ticker, date in breakout_keys_raw
        ],
        columns=["ticker", "date"],
    )
    breakout_keys = breakout_keys.loc[
        breakout_keys["ticker"].ne("")
        & breakout_keys["date"].ge(start)
        & breakout_keys["date"].le(end)
    ].drop_duplicates(["ticker", "date"])
    return {
        "schema_version": 1,
        "status": "AVAILABLE",
        "cohort": "filled_buys",
        "comparison_period": {"start": start, "end": end},
        "percentile_cutoff": float(STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_CUTOFF),
        "percentile_method": STRATEGY_COMPARE_MFE_SAFETY_GEOMETRY_PERCENTILE_METHOD,
        "truth_source": source,
        "population": {
            "arm_id": "POP",
            "name": "Breakout candidate truth",
            **mfe_safety_distribution_for_keys(truth, breakout_keys, allow_empty=True),
        },
        "arms": {
            arm.arm_id: {
                "arm_id": arm.arm_id,
                "name": arm.name,
                **mfe_safety_distribution_for_keys(
                    truth, arm_keys[arm.arm_id], allow_empty=True
                ),
            }
            for arm in settings.enabled_arms
        },
        "contract": {
            "high_definition": "same-day percentile >= cutoff",
            "safety_definition": "same-day percentile of canonical -target_adverse_r",
            "cohort_definition": "canonical filled selected_buys resolved to model score_event_date",
            "population_definition": (
                "canonical breakout candidate ticker/date filtered from daily-universal truth; "
                "percentiles are not re-ranked inside breakout subset"
            ),
            "future_truth_used_for_runtime_sort": False,
        },
    }


def build_strategy_diagnostics(
    *,
    project_root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    pair_payloads: dict[str, dict[str, Any]],
    reference_arm_id: str | None = None,
) -> dict[str, Any]:
    """Build one reusable diagnostic payload from already-canonical artifacts."""

    root = Path(project_root).resolve()
    model_prediction = _model_prediction_rows(root=root, settings=settings, status=status)
    selection_translation = _selection_translation_rows(
        settings=settings, pair_payloads=pair_payloads, scenarios=scenarios
    )
    (
        upside_realization,
        upside_realization_joint,
        upside_realization_unavailable,
    ) = _upside_realization_rows(settings=settings, pair_payloads=pair_payloads)
    try:
        mfe_safety_geometry = _build_mfe_safety_main_report_geometry(
            project_root=root,
            settings=settings,
            pair_payloads=pair_payloads,
        )
    except (ValueError, OSError, UnicodeError) as exc:
        mfe_safety_geometry = {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
            "future_truth_used_for_runtime_sort": False,
        }
    return {
        "diagnostics_schema_version": STRATEGY_DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics_contract_fingerprint": _upside_realization_contract_fingerprint(),
        "reference_arm_id": str(reference_arm_id or ""),
        "model_prediction": model_prediction,
        "selection_translation": selection_translation,
        "r_analysis": _r_analysis_rows(
            settings=settings,
            scenarios=scenarios,
            model_prediction=model_prediction,
            selection_translation=selection_translation,
        ),
        "execution_conversion": _execution_rows(settings=settings, scenarios=scenarios),
        "upside_realization": upside_realization,
        "upside_realization_joint": upside_realization_joint,
        "upside_realization_unavailable": upside_realization_unavailable,
        "upside_realization_contract": _upside_realization_contract(),
        "mfe_safety_geometry": mfe_safety_geometry,
        "contract": {
            "model_metrics_source": "validated existing PIT audit / continuous ranker report",
            "strategy_metrics_source": "canonical Strategy Compare pair payloads",
            "path_components_source": "canonical daily profile sample provider",
            "diagnostic_target_formula_reimplementation": False,
            "canonical_provider_may_materialize_from_source_ohlcv": True,
        },
    }


def _fmt(value: Any, *, digits: int = 3, unit: str = "") -> str:
    numeric = finite_number(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}{unit}"


def _fmt_pct_fraction(value: Any) -> str:
    numeric = finite_number(value)
    return "-" if numeric is None else f"{numeric * 100.0:.2f}%"


def _value_with_signal(text: str, signal: str | None, *, target: str) -> str:
    if text == "-" or not signal:
        return text
    return styled_signal(text, signal, target=target)


def _format_r_analysis_value(value: Any, metric: RAnalysisMetricSpec) -> str:
    if metric.format_kind == "fraction_pct":
        return _fmt_pct_fraction(value)
    return _fmt(value, digits=metric.digits, unit=metric.unit)


def _pad_cell(text: str, width: int, *, align: str = "left") -> str:
    value = str(text)
    padding = max(0, int(width) - _display_width(value))
    if align == "right":
        return " " * padding + value
    if align == "center":
        left = padding // 2
        return " " * left + value + " " * (padding - left)
    return value + " " * padding


def _render_grouped_table(headers_top: list[str], headers_bottom: list[str], rows: list[list[str]]) -> str:
    column_count = len(headers_bottom)
    widths = []
    for index in range(column_count):
        values = [headers_top[index], headers_bottom[index], *(row[index] for row in rows)]
        widths.append(max(_display_width(value) for value in values))
    def join_line(values: list[str], *, centers: set[int] | None = None) -> str:
        centers = centers or set()
        return "  ".join(
            _pad_cell(value, widths[idx], align="center" if idx in centers else "left")
            for idx, value in enumerate(values)
        )
    separator = "  ".join("-" * width for width in widths)
    return "\n".join([join_line(headers_top, centers=set(range(column_count))), join_line(headers_bottom, centers=set(range(1, column_count))), separator, *[join_line(row) for row in rows]])


def render_mfe_safety_geometry_table(diagnostics: dict[str, Any], *, target: str = "plain") -> str:
    """Render filled-buy MFE×Safety geometry for the Strategy Compare main report."""

    geometry = dict(diagnostics.get("mfe_safety_geometry") or {})
    status = str(geometry.get("status") or "UNAVAILABLE")
    if status == "NOT_APPLICABLE":
        return "此 evaluation profile 不顯示 MFE×Safety 四象限。"
    if status != "AVAILABLE":
        reason = str(geometry.get("reason") or "canonical truth/selection sidecar unavailable")
        return "MFE×Safety 四象限暫不可用：" + reason
    rows = [dict(geometry.get("population") or {})]
    arms = dict(geometry.get("arms") or {})
    rows.extend(dict(value) for value in arms.values())
    if not rows:
        return "沒有可用的 MFE×Safety 四象限資料。"

    metrics = MFE_SAFETY_COMPARE_RESULT_METRICS
    top_headers = ["", ""] + ["Filled buys truth geometry" if i == 0 else "" for i, _ in enumerate(metrics)]
    bottom_headers = ["編號", "比較對象"] + [metric.label for metric in metrics]
    signals_by_metric: dict[str, dict[str, str]] = {}
    strategy_rows = [row for row in rows if str(row.get("arm_id")) != "POP"]
    for metric in metrics:
        if metric.preference == "neutral":
            signals_by_metric[metric.key] = {}
        else:
            signals_by_metric[metric.key] = best_worst_signals(
                {str(row.get("arm_id") or "-"): row.get(metric.key) for row in strategy_rows},
                preference=metric.preference,
            )
    body: list[list[str]] = []
    for row in rows:
        arm_id = str(row.get("arm_id") or "-")
        values = [arm_id, str(row.get("name") or "-")]
        for metric in metrics:
            value = row.get(metric.key)
            if metric.digits == 0 and metric.unit == "":
                numeric = finite_number(value)
                text = "-" if numeric is None else str(int(round(numeric)))
            else:
                text = _fmt(value, digits=metric.digits, unit=metric.unit)
            signal = signals_by_metric.get(metric.key, {}).get(arm_id)
            values.append(_value_with_signal(text, signal, target=target))
        body.append(values)
    table = _render_grouped_table(top_headers, bottom_headers, body)
    cutoff = finite_number(geometry.get("percentile_cutoff"))
    cutoff_text = "-" if cutoff is None else f"{cutoff:.2f}"
    note = (
        f"Cohort=實際 Filled buys；High 定義為 same-day percentile >= {cutoff_text}；"
        "Safety=canonical Low-Adverse (-target_adverse_r) percentile。"
    )
    return table + "\n" + note


def render_strategy_r_analysis_table(diagnostics: dict[str, Any], *, target: str = "plain") -> str:
    """Render canonical R diagnostics as one grouped arm-comparison table."""

    rows = list(diagnostics.get("r_analysis") or [])
    if not rows:
        return "沒有可用的R預測／轉化診斷。"
    metrics = R_ANALYSIS_MERGED_METRICS
    top_headers = ["分群", ""]
    bottom_headers = ["編號", "比較對象"]
    for group_label, group_metrics in R_ANALYSIS_GROUPED_SECTIONS:
        for index, metric in enumerate(group_metrics):
            top_headers.append(group_label if index == 0 else "")
            bottom_headers.append(metric.label)
    body: list[list[str]] = []
    target_dependent_keys = {
        metric.key
        for metric in (*R_MODEL_PREDICTION_METRICS, *R_SELECTION_TRANSLATION_METRICS)
        if metric.key != "score_coverage"
    }
    metric_signals: dict[str, dict[str, str]] = {}
    for metric in metrics:
        if metric.key not in target_dependent_keys:
            metric_signals[metric.key] = best_worst_signals(
                {str(row.get("arm_id") or "-"): row.get(metric.key) for row in rows},
                preference=metric.preference,
            )
            continue
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            target_id = str(row.get("continuous_target_id") or "").strip()
            if not target_id:
                continue
            grouped.setdefault(target_id, []).append(row)
        signals: dict[str, str] = {}
        for group_rows in grouped.values():
            if len(group_rows) < 2:
                continue
            signals.update(best_worst_signals(
                {str(row.get("arm_id") or "-"): row.get(metric.key) for row in group_rows},
                preference=metric.preference,
            ))
        metric_signals[metric.key] = signals
    for row in rows:
        arm_id = str(row.get("arm_id") or "-")
        values = [arm_id, str(row.get("name") or "-")]
        for metric in metrics:
            text = _format_r_analysis_value(row.get(metric.key), metric)
            signal = metric_signals.get(metric.key, {}).get(arm_id)
            values.append(_value_with_signal(text, signal, target=target))
        body.append(values)
    return _render_grouped_table(top_headers, bottom_headers, body)


def render_strategy_selection_quality_table(
    diagnostics: dict[str, Any],
    *,
    target: str = "plain",
) -> str:
    """Render the compact Strategy-SOP selection scorecard.

    Model learnability and RCE attribution intentionally stay outside this table:
    learnability belongs to Model SOP, while RCE belongs to reusable Selection
    Attribution when a conversion mechanism needs explanation.
    """

    rows = list(diagnostics.get("selection_translation") or [])
    if not rows:
        return "沒有可用的Selection Quality診斷。"
    metric_by_key = {metric.key: metric for metric in R_SELECTION_TRANSLATION_METRICS}
    metric_keys = (
        "selected_target_mean_r",
        "selected_target_percentile",
        "target_top_k_retention",
        "target_opportunity_gap_r",
    )
    signals: dict[str, dict[str, str]] = {}
    for key in metric_keys:
        metric = metric_by_key[key]
        signals[key] = best_worst_signals(
            {str(row.get("arm_id") or "-"): row.get(key) for row in rows},
            preference=metric.preference,
        )
    body: list[list[str]] = []
    for row in rows:
        arm_id = str(row.get("arm_id") or "-")
        values = [arm_id, str(row.get("name") or "-")]
        for key in metric_keys:
            metric = metric_by_key[key]
            text = _format_r_analysis_value(row.get(key), metric)
            values.append(_value_with_signal(text, signals[key].get(arm_id), target=target))
        body.append(values)
    headers = ["編號", "比較對象"] + [metric_by_key[key].label for key in metric_keys]
    return _render_grouped_table(["", "", "Selection Quality", "", "", ""], headers, body)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _count_rate(count: Any, rate: Any) -> str:
    try:
        count_value = int(count)
    except (TypeError, ValueError):
        return "-"
    rate_value = finite_number(rate)
    return str(count_value) if rate_value is None else f"{count_value} ({rate_value * 100.0:.2f}%)"


def _count_of_count_rate(count: Any, total: Any, rate: Any) -> str:
    try:
        count_value = int(count)
        total_value = int(total)
    except (TypeError, ValueError):
        return "-"
    rate_value = finite_number(rate)
    if rate_value is None:
        return f"{count_value}/{total_value}"
    return f"{count_value}/{total_value} ({rate_value * 100.0:.2f}%)"


def render_upside_survival_summary_table(
    diagnostics: dict[str, Any], *, target: str = "plain"
) -> str:
    """Render the compact decision surface for upside survival / first-passage."""

    rows = list(diagnostics.get("upside_realization") or [])
    unavailable = list(diagnostics.get("upside_realization_unavailable") or [])
    if not rows:
        if not unavailable:
            return "沒有可用的path-conversion診斷。"
        return "沒有可用的path-conversion診斷。\n" + "\n".join(
            f"- {row.get('arm_id') or '-'} {row.get('name') or '-'}："
            f"{row.get('status') or 'UNAVAILABLE'} | {row.get('reason') or '未提供原因'}"
            for row in unavailable
        )

    thresholds = [
        float(value)
        for value in dict(diagnostics.get("upside_realization_contract") or {}).get(
            "upside_r_thresholds", []
        )
    ]
    metrics = (
        *(upside_survival_initial_stop_metric(threshold) for threshold in thresholds),
        *UPSIDE_SURVIVAL_BASE_METRICS,
    )

    arm_values: dict[str, dict[str, float | None]] = {}
    for row in rows:
        arm_id = str(row.get("arm_id") or "-")
        threshold_payload = dict(row.get("thresholds") or {})
        values: dict[str, float | None] = {
            "full_horizon_mfe_mean_r": finite_number(row.get("full_horizon_mfe_mean_r")),
            "full_horizon_adverse_to_peak_mean_r": finite_number(
                row.get("full_horizon_adverse_to_peak_mean_r")
            ),
            "realized_mean_r": finite_number(row.get("realized_mean_r")),
        }
        for threshold in thresholds:
            item = dict(threshold_payload.get(f"{threshold:g}R") or {})
            values[f"initial_stop_before_{threshold:g}r_rate"] = finite_number(
                item.get("actual_initial_stop_before_first_upside_rate")
            )
        arm_values[arm_id] = values

    signals_by_metric = {
        metric.key: best_worst_signals(
            {arm_id: values.get(metric.key) for arm_id, values in arm_values.items()},
            preference=metric.preference,
        )
        for metric in metrics
    }

    headers = ["編號", "比較對象", *(metric.label for metric in metrics)]
    body: list[tuple[object, ...]] = []
    for row in rows:
        arm_id = str(row.get("arm_id") or "-")
        values = arm_values.get(arm_id, {})
        rendered: list[str] = []
        for metric in metrics:
            numeric = values.get(metric.key)
            if metric.key.startswith("initial_stop_before_"):
                text = "-" if numeric is None else f"{numeric * 100.0:.{metric.digits}f}{metric.unit}"
            else:
                text = _fmt(numeric, digits=metric.digits, unit=metric.unit)
            rendered.append(
                _value_with_signal(
                    text,
                    signals_by_metric.get(metric.key, {}).get(arm_id),
                    target=target,
                )
            )
        body.append((arm_id, str(row.get("name") or "-"), *rendered))

    widths = []
    for index, header in enumerate(headers):
        values = [str(header), *(str(row[index]) for row in body)]
        widths.append(max(_display_width(value) for value in values))
    separator = "  ".join("-" * width for width in widths)
    lines = [
        "  ".join(_pad_cell(str(header), widths[index]) for index, header in enumerate(headers)),
        separator,
    ]
    lines.extend(
        "  ".join(_pad_cell(str(value), widths[index]) for index, value in enumerate(row))
        for row in body
    )
    lines.extend([
        "",
        "註：+kR前初始Stop＝future horizon內確實首次到達+kR者中，實際策略在首次達標前／同日由初始Stop出場的比例；同日保守視為Stop先發生。",
    ])
    if unavailable:
        lines.extend(
            f"未產生 {row.get('arm_id') or '-'} {row.get('name') or '-'}："
            f"{row.get('status') or 'UNAVAILABLE'} | {row.get('reason') or '未提供原因'}"
            for row in unavailable
        )
    return "\n".join(lines)


def render_upside_realization_summary_table(
    diagnostics: dict[str, Any], *, target: str = "plain"
) -> str:
    """Render the core path-conversion diagnostics on the main Strategy Compare surface."""

    del target  # Neutral diagnostic table; detailed joint buckets remain in strategy_diagnostics.md.
    rows = list(diagnostics.get("upside_realization") or [])
    unavailable = list(diagnostics.get("upside_realization_unavailable") or [])
    if not rows:
        if not unavailable:
            return "沒有可用的path-conversion診斷。"
        return "沒有可用的path-conversion診斷。\n" + "\n".join(
            f"- {row.get('arm_id') or '-'} {row.get('name') or '-'}："
            f"{row.get('status') or 'UNAVAILABLE'} | {row.get('reason') or '未提供原因'}"
            for row in unavailable
        )
    thresholds = [
        float(value)
        for value in dict(diagnostics.get("upside_realization_contract") or {}).get(
            "upside_r_thresholds", []
        )
    ]
    headers = [
        "編號",
        "比較對象",
        "Path Coverage",
        "實際停損",
        "停損≤最終高點",
        *[f"MFE≥{value:g}R且先停損" for value in thresholds],
        "Full-MFE",
        "Adverse",
        "Realized",
    ]
    body: list[tuple[object, ...]] = []
    for row in rows:
        threshold_payload = dict(row.get("thresholds") or {})
        threshold_values = []
        for threshold in thresholds:
            item = dict(threshold_payload.get(f"{threshold:g}R") or {})
            threshold_values.append(
                _count_of_count_rate(
                    item.get("stop_before_later_peak_count"),
                    item.get("full_horizon_mfe_at_least_count"),
                    item.get("stop_before_later_peak_rate"),
                )
            )
        body.append(
            (
                str(row.get("arm_id") or "-"),
                str(row.get("name") or "-"),
                _fmt_pct_fraction(row.get("path_target_coverage_rate")),
                _count_rate(row.get("actual_stop_out_count"), row.get("actual_stop_out_rate")),
                _count_rate(
                    row.get("stop_before_later_peak_count"),
                    row.get("stop_before_later_peak_rate"),
                ),
                *threshold_values,
                _fmt(row.get("full_horizon_mfe_mean_r"), digits=2, unit="R"),
                _fmt(row.get("full_horizon_adverse_to_peak_mean_r"), digits=2, unit="R"),
                _fmt(row.get("realized_mean_r"), digits=2, unit="R"),
            )
        )
    widths = []
    for index, header in enumerate(headers):
        values = [str(header), *(str(row[index]) for row in body)]
        widths.append(max(_display_width(value) for value in values))
    separator = "  ".join("-" * width for width in widths)
    lines = [
        "  ".join(_pad_cell(str(header), widths[index]) for index, header in enumerate(headers)),
        separator,
    ]
    lines.extend(
        "  ".join(_pad_cell(str(value), widths[index]) for index, value in enumerate(row))
        for row in body
    )
    if unavailable:
        lines.append("")
        lines.extend(
            f"未產生 {row.get('arm_id') or '-'} {row.get('name') or '-'}："
            f"{row.get('status') or 'UNAVAILABLE'} | {row.get('reason') or '未提供原因'}"
            for row in unavailable
        )
    return "\n".join(lines)


def render_first_passage_summary_table(
    diagnostics: dict[str, Any], *, target: str = "plain"
) -> str:
    """Render exact first-passage conversion for each configured target-R threshold."""

    del target
    rows = list(diagnostics.get("upside_realization") or [])
    if not rows:
        return ""
    thresholds = [
        float(value)
        for value in dict(diagnostics.get("upside_realization_contract") or {}).get(
            "upside_r_thresholds", []
        )
    ]
    headers = [
        "編號", "比較對象", "門檻", "Future達標", "Risk≤首次達標",
        "首次達標<Risk", "實際Stop≤首次達標", "其中初始Stop", "其中拉高Stop",
    ]
    body: list[tuple[object, ...]] = []
    for row in rows:
        threshold_payload = dict(row.get("thresholds") or {})
        for threshold in thresholds:
            item = dict(threshold_payload.get(f"{threshold:g}R") or {})
            total = item.get("first_upside_reached_count")
            body.append((
                str(row.get("arm_id") or "-"),
                str(row.get("name") or "-"),
                f"+{threshold:g}R",
                str(int(total or 0)),
                _count_of_count_rate(
                    item.get("canonical_risk_before_first_upside_count"),
                    total,
                    item.get("canonical_risk_before_first_upside_rate"),
                ),
                _count_of_count_rate(
                    item.get("canonical_upside_before_risk_count"),
                    total,
                    item.get("canonical_upside_before_risk_rate"),
                ),
                _count_of_count_rate(
                    item.get("actual_stop_before_first_upside_count"),
                    total,
                    item.get("actual_stop_before_first_upside_rate"),
                ),
                _count_of_count_rate(
                    item.get("actual_initial_stop_before_first_upside_count"),
                    total,
                    item.get("actual_initial_stop_before_first_upside_rate"),
                ),
                _count_of_count_rate(
                    item.get("actual_raised_stop_before_first_upside_count"),
                    total,
                    item.get("actual_raised_stop_before_first_upside_rate"),
                ),
            ))
    widths = []
    for index, header in enumerate(headers):
        values = [str(header), *(str(row[index]) for row in body)]
        widths.append(max(_display_width(value) for value in values))
    separator = "  ".join("-" * width for width in widths)
    lines = [
        "First-Passage Realization（+kR＝Pure-MFE target R；同日保守視為risk／stop先發生）",
        "  ".join(_pad_cell(str(header), widths[index]) for index, header in enumerate(headers)),
        separator,
    ]
    lines.extend(
        "  ".join(_pad_cell(str(value), widths[index]) for index, value in enumerate(row))
        for row in body
    )
    return "\n".join(lines)


def render_first_passage_markdown(diagnostics: dict[str, Any]) -> str:
    rows = list(diagnostics.get("upside_realization") or [])
    if not rows:
        return ""
    thresholds = [
        float(value)
        for value in dict(diagnostics.get("upside_realization_contract") or {}).get(
            "upside_r_thresholds", []
        )
    ]
    headers = [
        "編號", "比較對象", "門檻", "Future達標", "Risk≤首次達標",
        "首次達標<Risk", "實際Stop≤首次達標", "其中初始Stop", "其中拉高Stop",
    ]
    lines = [
        "### First-Passage Realization",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        threshold_payload = dict(row.get("thresholds") or {})
        for threshold in thresholds:
            item = dict(threshold_payload.get(f"{threshold:g}R") or {})
            total = item.get("first_upside_reached_count")
            values = [
                str(row.get("arm_id") or "-"),
                str(row.get("name") or "-"),
                f"+{threshold:g}R",
                str(int(total or 0)),
                _count_of_count_rate(
                    item.get("canonical_risk_before_first_upside_count"), total,
                    item.get("canonical_risk_before_first_upside_rate"),
                ),
                _count_of_count_rate(
                    item.get("canonical_upside_before_risk_count"), total,
                    item.get("canonical_upside_before_risk_rate"),
                ),
                _count_of_count_rate(
                    item.get("actual_stop_before_first_upside_count"), total,
                    item.get("actual_stop_before_first_upside_rate"),
                ),
                _count_of_count_rate(
                    item.get("actual_initial_stop_before_first_upside_count"), total,
                    item.get("actual_initial_stop_before_first_upside_rate"),
                ),
                _count_of_count_rate(
                    item.get("actual_raised_stop_before_first_upside_count"), total,
                    item.get("actual_raised_stop_before_first_upside_rate"),
                ),
            ]
            lines.append("| " + " | ".join(_markdown_escape(value) for value in values) + " |")
    lines.extend([
        "",
        "> `+kR` 使用 Pure-MFE target 的 score-event close 與 target risk-budget R；不是策略實際 R_Multiple。",
        "> `Risk≤首次達標` 使用同一 canonical target risk barrier；同日 high/low 順序不可知時，依保守原則視為 risk 先發生。",
        "> `實際Stop≤首次達標` 表示該 DL-selected trade 的實際策略 stop exit 發生在 target-defined 第一次 +kR 當日或之前；並以 entry/exit trade-history stop price 拆成初始 stop 與已拉高 stop。",
        "",
    ])
    return "\n".join(lines)


def render_upside_realization_markdown(diagnostics: dict[str, Any]) -> str:
    rows = list(diagnostics.get("upside_realization") or [])
    unavailable = list(diagnostics.get("upside_realization_unavailable") or [])
    if not rows:
        lines = ["## Upside Realization / Stop-before-Upside", "", "沒有可用的path-conversion診斷。"]
        for row in unavailable:
            lines.append(
                f"- `{row.get('arm_id') or '-'}` {row.get('name') or '-'}："
                f"{row.get('status') or 'UNAVAILABLE'} | {row.get('reason') or '未提供原因'}"
            )
        lines.append("")
        return "\n".join(lines)
    thresholds = [
        float(value)
        for value in dict(diagnostics.get("upside_realization_contract") or {}).get(
            "upside_r_thresholds", []
        )
    ]
    headers = [
        "編號", "比較對象", "完成交易", "Path Coverage", "實際停損",
        "停損早於/同日最終高點",
    ] + [f"MFE≥{value:g}R 且停損早於高點" for value in thresholds] + [
        "Full-MFE均值", "到峰值Adverse均值", "Realized均值",
    ]
    lines = [
        "## Upside Realization / Stop-before-Upside",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        threshold_payload = dict(row.get("thresholds") or {})
        values = [
            str(row.get("arm_id") or "-"),
            str(row.get("name") or "-"),
            str(int(row.get("completed_trade_count") or 0)),
            _fmt_pct_fraction(row.get("path_target_coverage_rate")),
            _count_rate(row.get("actual_stop_out_count"), row.get("actual_stop_out_rate")),
            _count_rate(row.get("stop_before_later_peak_count"), row.get("stop_before_later_peak_rate")),
        ]
        for threshold in thresholds:
            item = dict(threshold_payload.get(f"{threshold:g}R") or {})
            values.append(_count_of_count_rate(
                item.get("stop_before_later_peak_count"),
                item.get("full_horizon_mfe_at_least_count"),
                item.get("stop_before_later_peak_rate"),
            ))
        values.extend([
            _fmt(row.get("full_horizon_mfe_mean_r"), digits=2, unit="R"),
            _fmt(row.get("full_horizon_adverse_to_peak_mean_r"), digits=2, unit="R"),
            _fmt(row.get("realized_mean_r"), digits=2, unit="R"),
        ])
        lines.append("| " + " | ".join(_markdown_escape(value) for value in values) + " |")
    if unavailable:
        lines.extend(["", "未產生的 arm："])
        for row in unavailable:
            lines.append(
                f"- `{row.get('arm_id') or '-'}` {row.get('name') or '-'}："
                f"{row.get('status') or 'UNAVAILABLE'} | {row.get('reason') or '未提供原因'}"
            )
    lines.extend([
        "",
        "> `停損早於/同日最終高點`：completed trade 實際以停損出場，且 canonical full-horizon pure-MFE 的最終高點日期在停損日之後或同日；同日依保守盤中順序視為尚未證明可先實現高點。",
        "> 上表仍保留既有 final-peak attribution；精確的「第一次到 +kR 前是否先 risk／stop」請以下方 First-Passage Realization 為準。",
        "",
    ])
    return "\n".join(lines)


def render_upside_realization_joint_markdown(diagnostics: dict[str, Any]) -> str:
    rows = list(diagnostics.get("upside_realization_joint") or [])
    if not rows:
        return ""
    headers = [
        "編號", "比較對象", "Full-MFE區間", "到峰值Adverse區間", "交易數",
        "停損率", "停損早於/同日最終高點率", "Realized均值",
    ]
    lines = [
        "## Full-MFE × Adverse-to-Peak 轉化",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("arm_id") or "-"),
            str(row.get("name") or "-"),
            str(row.get("mfe_bucket") or "-"),
            str(row.get("adverse_bucket") or "-"),
            str(int(row.get("trade_count") or 0)),
            _fmt_pct_fraction(row.get("stop_out_rate")),
            _fmt_pct_fraction(row.get("stop_before_later_peak_rate")),
            _fmt(row.get("realized_mean_r"), digits=2, unit="R"),
        ]
        lines.append("| " + " | ".join(_markdown_escape(value) for value in values) + " |")
    lines.append("")
    return "\n".join(lines)


def render_strategy_diagnostics_markdown(diagnostics: dict[str, Any]) -> str:
    """Persist canonical model/selection and path-conversion diagnostics."""

    sections = [
        "# Strategy Compare 間接指標\n",
        "## MFE × Safety 四象限（Filled buys）",
        render_mfe_safety_geometry_table(diagnostics, target="markdown").rstrip(),
        "",
        render_strategy_r_analysis_table(diagnostics, target="markdown").rstrip(),
        "",
        render_upside_realization_markdown(diagnostics).rstrip(),
        render_first_passage_markdown(diagnostics).rstrip(),
    ]
    joint = render_upside_realization_joint_markdown(diagnostics).rstrip()
    if joint:
        sections.extend(["", joint])
    return "\n".join(sections).rstrip() + "\n"
