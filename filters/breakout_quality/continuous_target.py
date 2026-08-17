"""Versioned continuous outcome targets for breakout-quality research."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.contract import BreakoutQualityLabelPolicy

CONTINUOUS_TARGET_SCHEMA_VERSION = 1
STRATEGY_ALIGNED_TARGET_ID = "strategy_aligned_opportunity_r_v1"
STRATEGY_ALIGNED_NO_TIME_TARGET_ID = "strategy_aligned_opportunity_no_time_r_v1"
DAILY_OPPORTUNITY_NO_TIME_TARGET_ID = "daily_opportunity_no_time_r_v1"
DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID = "daily_full_horizon_opportunity_r_v1"
DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID = "daily_full_horizon_pure_mfe_r_v1"

TARGET_RAW_FILENAME = "group_target_raw_r.npy"
TARGET_FAVORABLE_RETURN_FILENAME = "group_favorable_return.npy"
TARGET_ADVERSE_RETURN_FILENAME = "group_adverse_return_to_peak.npy"
TARGET_OPPORTUNITY_BAR_FILENAME = "group_opportunity_bar.npy"
TARGET_RISK_BREACH_BAR_FILENAME = "group_first_risk_breach_bar.npy"
TARGET_VALID_MASK_FILENAME = "group_target_valid_mask.npy"
TARGET_MANIFEST_FILENAME = "manifest.json"
TARGET_AUDIT_JSON_FILENAME = "continuous_target_audit.json"
TARGET_AUDIT_MARKDOWN_FILENAME = "continuous_target_audit.md"
TARGET_DAILY_CSV_FILENAME = "continuous_target_daily_rankability.csv"
TARGET_TRADE_MATCHES_CSV_FILENAME = "continuous_target_trade_matches.csv"


@dataclass(frozen=True)
class StrategyAlignedContinuousTargetSpec:
    """Fixed, non-tuned v1 target derived from the existing label policy."""

    target_id: str
    horizon_bars: int
    min_mfe_return: float
    max_adverse_return: float

    @classmethod
    def from_label_policy(
        cls,
        policy: BreakoutQualityLabelPolicy,
    ) -> "StrategyAlignedContinuousTargetSpec":
        return cls(
            target_id=STRATEGY_ALIGNED_TARGET_ID,
            horizon_bars=int(policy.label_horizon_bars),
            min_mfe_return=float(policy.min_mfe_return),
            max_adverse_return=float(policy.max_adverse_return),
        ).validated()

    def validated(self) -> "StrategyAlignedContinuousTargetSpec":
        if str(self.target_id) != STRATEGY_ALIGNED_TARGET_ID:
            raise ValueError(f"不支援的 continuous target id: {self.target_id}")
        if int(self.horizon_bars) < 2:
            raise ValueError("continuous target horizon_bars 必須 >= 2")
        if not math.isfinite(float(self.min_mfe_return)) or float(self.min_mfe_return) <= 0.0:
            raise ValueError("continuous target min_mfe_return 必須是有限正數")
        if not math.isfinite(float(self.max_adverse_return)) or not (-1.0 < float(self.max_adverse_return) < 0.0):
            raise ValueError("continuous target max_adverse_return 必須介於 -1 與 0")
        return self

    @property
    def risk_budget_return(self) -> float:
        return abs(float(self.max_adverse_return))

    @property
    def full_horizon_time_penalty_r(self) -> float:
        return float(self.min_mfe_return) / self.risk_budget_return

    def contract_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "target_id": str(self.target_id),
            "objective_family": "continuous_strategy_aligned_opportunity",
            "information_source": "fixed_future_high_low_path",
            "horizon_bars": int(self.horizon_bars),
            "risk_budget_return": float(self.risk_budget_return),
            "minimum_opportunity_return": float(self.min_mfe_return),
            "full_horizon_time_penalty_r": float(self.full_horizon_time_penalty_r),
            "formula": (
                "favorable_return_before_first_risk_breach / risk_budget_return "
                "- adverse_return_required_to_reach_that_peak / risk_budget_return "
                "- full_horizon_time_penalty_r * ((opportunity_bar - 1) / (horizon_bars - 1))"
            ),
            "peak_rule": "earliest maximum high before first risk-barrier touch",
            "risk_rule": "same-bar adverse-first; barrier-day high is excluded",
            "adverse_scope": "worst low from horizon start through selected peak bar",
            "no_safe_bar_rule": "target=-1R when the risk barrier is touched on the first bar",
            "normalization": "none",
            "clipping": "none",
            "split_derived_parameters": False,
            "oos_derived_parameters": False,
        }


@dataclass(frozen=True)
class StrategyAlignedContinuousTargetResult:
    valid: bool
    reason: str
    target_raw_r: float
    favorable_return: float
    adverse_return_to_peak: float
    opportunity_bar: int
    first_risk_breach_bar: int


def _invalid_result(reason: str) -> StrategyAlignedContinuousTargetResult:
    return StrategyAlignedContinuousTargetResult(
        valid=False,
        reason=str(reason),
        target_raw_r=math.nan,
        favorable_return=math.nan,
        adverse_return_to_peak=math.nan,
        opportunity_bar=-1,
        first_risk_breach_bar=-1,
    )


def strategy_aligned_target_from_cached_path(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    *,
    anchor_price: float,
    available_bars: int,
    spec: StrategyAlignedContinuousTargetSpec,
) -> StrategyAlignedContinuousTargetResult:
    """Build one conservative opportunity-R target without using split statistics."""

    spec = spec.validated()
    horizon = int(spec.horizon_bars)
    if int(available_bars) < horizon:
        return _invalid_result("insufficient_future")
    if not math.isfinite(float(anchor_price)) or float(anchor_price) <= 0.0:
        return _invalid_result("invalid_anchor")

    highs = np.asarray(high_prices[:horizon], dtype=np.float64)
    lows = np.asarray(low_prices[:horizon], dtype=np.float64)
    if highs.shape != lows.shape or highs.size != horizon:
        return _invalid_result("insufficient_future")
    valid = np.isfinite(highs) & np.isfinite(lows) & (highs > 0.0) & (lows > 0.0) & (highs >= lows)
    if not bool(np.all(valid)):
        return _invalid_result("invalid_future_bar")

    anchor = float(anchor_price)
    risk_budget = float(spec.risk_budget_return)
    risk_barrier_price = anchor * (1.0 - risk_budget)
    running_low = anchor
    best_favorable = -math.inf
    best_adverse = math.nan
    best_bar = -1
    first_risk_breach_bar = -1

    for bar_offset, (high_price, low_price) in enumerate(zip(highs, lows), start=1):
        low_value = float(low_price)
        high_value = float(high_price)
        if low_value <= risk_barrier_price:
            first_risk_breach_bar = int(bar_offset)
            break

        running_low = min(running_low, low_value)
        favorable_return = float(high_value / anchor - 1.0)
        if favorable_return > best_favorable:
            best_favorable = favorable_return
            best_adverse = max(0.0, float(1.0 - running_low / anchor))
            best_bar = int(bar_offset)

    if best_bar < 1:
        # Conservative adverse-first handling: an immediate barrier touch provides
        # no usable same-day high and consumes the full risk budget.
        favorable_return = 0.0
        adverse_return = risk_budget
        opportunity_bar = int(first_risk_breach_bar if first_risk_breach_bar > 0 else 1)
    else:
        favorable_return = float(best_favorable)
        adverse_return = float(best_adverse)
        opportunity_bar = int(best_bar)

    time_fraction = float(opportunity_bar - 1) / float(horizon - 1)
    favorable_r = favorable_return / risk_budget
    adverse_r = adverse_return / risk_budget
    time_penalty_r = float(spec.full_horizon_time_penalty_r) * time_fraction
    target_raw_r = favorable_r - adverse_r - time_penalty_r
    if not all(
        math.isfinite(value)
        for value in (target_raw_r, favorable_return, adverse_return, time_fraction)
    ):
        return _invalid_result("non_finite_target")

    return StrategyAlignedContinuousTargetResult(
        valid=True,
        reason="ok",
        target_raw_r=float(target_raw_r),
        favorable_return=float(favorable_return),
        adverse_return_to_peak=float(adverse_return),
        opportunity_bar=int(opportunity_bar),
        first_risk_breach_bar=int(first_risk_breach_bar),
    )


def daily_opportunity_no_time_target_from_cached_path(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    *,
    anchor_price: float,
    available_bars: int,
    spec: StrategyAlignedContinuousTargetSpec,
) -> StrategyAlignedContinuousTargetResult:
    """Build the strategy-independent daily opportunity target.

    The future-path mechanics are intentionally identical to the validated no-time
    target used by MR-12A/B/C.  The only semantic change is the sample universe:
    this function may be evaluated for any eligible ticker/date and does not require
    a breakout event, high_len, breakout level, candidate membership, or strategy
    execution decision.
    """

    source = strategy_aligned_target_from_cached_path(
        high_prices,
        low_prices,
        anchor_price=anchor_price,
        available_bars=available_bars,
        spec=spec,
    )
    if not source.valid:
        return source
    risk_budget = float(spec.risk_budget_return)
    target_raw_r = (
        float(source.favorable_return) / risk_budget
        - float(source.adverse_return_to_peak) / risk_budget
    )
    if not math.isfinite(target_raw_r):
        return _invalid_result("non_finite_target")
    return StrategyAlignedContinuousTargetResult(
        valid=True,
        reason="ok",
        target_raw_r=float(target_raw_r),
        favorable_return=float(source.favorable_return),
        adverse_return_to_peak=float(source.adverse_return_to_peak),
        opportunity_bar=int(source.opportunity_bar),
        first_risk_breach_bar=int(source.first_risk_breach_bar),
    )


def daily_full_horizon_opportunity_target_from_cached_path(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    *,
    anchor_price: float,
    available_bars: int,
    spec: StrategyAlignedContinuousTargetSpec,
) -> StrategyAlignedContinuousTargetResult:
    """Build the no-breach full-horizon opportunity target for MR-13H.

    Relative to ``daily_opportunity_no_time_r_v1``, the only scientific change is
    that a low crossing the canonical risk-budget return is diagnostic only and
    does not truncate the future high path.  The earliest maximum high over the
    full fixed horizon is selected and adverse excursion is still measured from
    horizon start through that selected peak bar.
    """

    spec = spec.validated()
    horizon = int(spec.horizon_bars)
    if int(available_bars) < horizon:
        return _invalid_result("insufficient_future")
    if not math.isfinite(float(anchor_price)) or float(anchor_price) <= 0.0:
        return _invalid_result("invalid_anchor")

    highs = np.asarray(high_prices[:horizon], dtype=np.float64)
    lows = np.asarray(low_prices[:horizon], dtype=np.float64)
    if highs.shape != lows.shape or highs.size != horizon:
        return _invalid_result("insufficient_future")
    valid = np.isfinite(highs) & np.isfinite(lows) & (highs > 0.0) & (lows > 0.0) & (highs >= lows)
    if not bool(np.all(valid)):
        return _invalid_result("invalid_future_bar")

    anchor = float(anchor_price)
    risk_budget = float(spec.risk_budget_return)
    risk_barrier_price = anchor * (1.0 - risk_budget)
    breach = np.flatnonzero(lows <= risk_barrier_price)
    first_risk_breach_bar = int(breach[0] + 1) if len(breach) else -1

    favorable_returns = highs / anchor - 1.0
    best_zero = int(np.argmax(favorable_returns))  # np.argmax preserves earliest tie.
    opportunity_bar = int(best_zero + 1)
    favorable_return = float(favorable_returns[best_zero])
    running_low = float(np.min(lows[: best_zero + 1]))
    adverse_return = max(0.0, float(1.0 - running_low / anchor))
    target_raw_r = favorable_return / risk_budget - adverse_return / risk_budget
    if not all(
        math.isfinite(value)
        for value in (target_raw_r, favorable_return, adverse_return)
    ):
        return _invalid_result("non_finite_target")

    return StrategyAlignedContinuousTargetResult(
        valid=True,
        reason="ok",
        target_raw_r=float(target_raw_r),
        favorable_return=float(favorable_return),
        adverse_return_to_peak=float(adverse_return),
        opportunity_bar=int(opportunity_bar),
        first_risk_breach_bar=int(first_risk_breach_bar),
    )


def daily_full_horizon_pure_mfe_target_from_cached_path(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    *,
    anchor_price: float,
    available_bars: int,
    spec: StrategyAlignedContinuousTargetSpec,
) -> StrategyAlignedContinuousTargetResult:
    """Build the MR-13K pure-MFE full-horizon target.

    Relative to MR-13H, the full 40-bar horizon, earliest maximum-high rule,
    risk-breach diagnostics and fixed canonical R scale are unchanged.  The
    only scientific change is that adverse excursion to the selected peak is
    diagnostic only and is not deducted from the target.
    """

    source = daily_full_horizon_opportunity_target_from_cached_path(
        high_prices,
        low_prices,
        anchor_price=anchor_price,
        available_bars=available_bars,
        spec=spec,
    )
    if not source.valid:
        return source
    risk_budget = float(spec.risk_budget_return)
    target_raw_r = float(source.favorable_return) / risk_budget
    if not math.isfinite(target_raw_r):
        return _invalid_result("non_finite_target")
    return StrategyAlignedContinuousTargetResult(
        valid=True,
        reason="ok",
        target_raw_r=float(target_raw_r),
        favorable_return=float(source.favorable_return),
        adverse_return_to_peak=float(source.adverse_return_to_peak),
        opportunity_bar=int(source.opportunity_bar),
        first_risk_breach_bar=int(source.first_risk_breach_bar),
    )


def build_daily_full_horizon_pure_mfe_contract(
    policy: BreakoutQualityLabelPolicy,
) -> dict[str, object]:
    """Return the MR-13K full-horizon pure-MFE target contract."""

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(policy)
    return {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "target_id": DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        "objective_family": "daily_cross_sectional_full_horizon_pure_mfe",
        "information_source": "fixed_future_high_low_path",
        "sample_scope": "daily_eligible_stock_days",
        "horizon_bars": int(spec.horizon_bars),
        "risk_budget_return": float(spec.risk_budget_return),
        "formula": "full_horizon_favorable_return / risk_budget_return",
        "peak_rule": "earliest maximum high across the complete fixed horizon",
        "risk_rule": "risk-barrier touch is diagnostic only and never truncates the future path",
        "adverse_scope": "diagnostic worst low from horizon start through selected peak bar; not deducted",
        "adverse_penalty_included": False,
        "no_safe_bar_rule": "not_applicable",
        "requires_breakout_event": False,
        "requires_high_len": False,
        "requires_breakout_level": False,
        "requires_strategy_candidate_membership": False,
        "time_penalty_included": False,
        "normalization": "fixed canonical risk-budget return used only as R scale",
        "clipping": "none",
        "split_derived_parameters": False,
        "oos_fitted_parameters": False,
    }


def build_daily_full_horizon_opportunity_contract(
    policy: BreakoutQualityLabelPolicy,
) -> dict[str, object]:
    """Return the MR-13H full-horizon no-breach target contract."""

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(policy)
    return {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "target_id": DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
        "objective_family": "daily_cross_sectional_full_horizon_opportunity_no_time",
        "information_source": "fixed_future_high_low_path",
        "sample_scope": "daily_eligible_stock_days",
        "horizon_bars": int(spec.horizon_bars),
        "risk_budget_return": float(spec.risk_budget_return),
        "formula": (
            "full_horizon_favorable_return / risk_budget_return "
            "- adverse_return_required_to_reach_that_peak / risk_budget_return"
        ),
        "peak_rule": "earliest maximum high across the complete fixed horizon",
        "risk_rule": (
            "risk-barrier touch is diagnostic only and never truncates the future path"
        ),
        "adverse_scope": "worst low from horizon start through selected peak bar",
        "no_safe_bar_rule": "not_applicable",
        "requires_breakout_event": False,
        "requires_high_len": False,
        "requires_breakout_level": False,
        "requires_strategy_candidate_membership": False,
        "time_penalty_included": False,
        "normalization": "fixed canonical risk-budget return used only as R scale",
        "clipping": "none",
        "split_derived_parameters": False,
        "oos_fitted_parameters": False,
    }


def build_daily_opportunity_no_time_contract(
    policy: BreakoutQualityLabelPolicy,
) -> dict[str, object]:
    """Return the fixed MR-13A target contract without breakout-event semantics."""

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(policy)
    return {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "target_id": DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
        "objective_family": "daily_cross_sectional_opportunity_no_time",
        "information_source": "fixed_future_high_low_path",
        "sample_scope": "daily_eligible_stock_days",
        "horizon_bars": int(spec.horizon_bars),
        "risk_budget_return": float(spec.risk_budget_return),
        "formula": (
            "favorable_return_before_first_risk_breach / risk_budget_return "
            "- adverse_return_required_to_reach_that_peak / risk_budget_return"
        ),
        "peak_rule": "earliest maximum high before first risk-barrier touch",
        "risk_rule": "same-bar adverse-first; barrier-day high is excluded",
        "adverse_scope": "worst low from horizon start through selected peak bar",
        "no_safe_bar_rule": "target=-1R when the risk barrier is touched on the first bar",
        "requires_breakout_event": False,
        "requires_high_len": False,
        "requires_breakout_level": False,
        "requires_strategy_candidate_membership": False,
        "time_penalty_included": False,
        "normalization": "none",
        "clipping": "none",
        "split_derived_parameters": False,
        "oos_fitted_parameters": False,
    }


def build_strategy_aligned_group_targets(
    group_anchor_prices: np.ndarray,
    future_high_prices: np.ndarray,
    future_low_prices: np.ndarray,
    future_available_bars: np.ndarray,
    *,
    spec: StrategyAlignedContinuousTargetSpec,
) -> dict[str, np.ndarray]:
    """Vector container around the single-group target contract."""

    anchors = np.asarray(group_anchor_prices, dtype=np.float64)
    highs = np.asarray(future_high_prices)
    lows = np.asarray(future_low_prices)
    available = np.asarray(future_available_bars, dtype=np.int64)
    if anchors.ndim != 1 or highs.ndim != 2 or lows.ndim != 2 or available.ndim != 1:
        raise ValueError("continuous target input shape 不合法")
    if highs.shape != lows.shape:
        raise ValueError("continuous target future high/low shape 不一致")
    group_count = int(len(anchors))
    if len(highs) != group_count or len(lows) != group_count or len(available) != group_count:
        raise ValueError("continuous target group input 長度不一致")
    if highs.shape[1] < int(spec.horizon_bars):
        raise ValueError("continuous target horizon 超過 future path cache")

    target = np.full(group_count, np.nan, dtype=np.float32)
    favorable = np.full(group_count, np.nan, dtype=np.float32)
    adverse = np.full(group_count, np.nan, dtype=np.float32)
    opportunity_bar = np.full(group_count, -1, dtype=np.int16)
    risk_breach_bar = np.full(group_count, -1, dtype=np.int16)
    valid_mask = np.zeros(group_count, dtype=np.bool_)

    for group_index in range(group_count):
        result = strategy_aligned_target_from_cached_path(
            highs[group_index],
            lows[group_index],
            anchor_price=float(anchors[group_index]),
            available_bars=int(available[group_index]),
            spec=spec,
        )
        if not result.valid:
            continue
        target[group_index] = np.float32(result.target_raw_r)
        favorable[group_index] = np.float32(result.favorable_return)
        adverse[group_index] = np.float32(result.adverse_return_to_peak)
        opportunity_bar[group_index] = np.int16(result.opportunity_bar)
        risk_breach_bar[group_index] = np.int16(result.first_risk_breach_bar)
        valid_mask[group_index] = True

    return {
        "target_raw_r": target,
        "favorable_return": favorable,
        "adverse_return_to_peak": adverse,
        "opportunity_bar": opportunity_bar,
        "first_risk_breach_bar": risk_breach_bar,
        "valid_mask": valid_mask,
    }



def build_strategy_aligned_no_time_group_targets(
    *,
    favorable_return: np.ndarray,
    adverse_return_to_peak: np.ndarray,
    opportunity_bar: np.ndarray,
    first_risk_breach_bar: np.ndarray,
    valid_mask: np.ndarray,
    risk_budget_return: float,
) -> dict[str, np.ndarray]:
    """Derive the fixed no-time target from validated 11A component arrays."""

    risk_budget = float(risk_budget_return)
    if not math.isfinite(risk_budget) or risk_budget <= 0.0:
        raise ValueError("no-time continuous target risk_budget_return必須是有限正數")

    favorable = np.asarray(favorable_return, dtype=np.float32)
    adverse = np.asarray(adverse_return_to_peak, dtype=np.float32)
    opportunity = np.asarray(opportunity_bar, dtype=np.int16)
    risk_breach = np.asarray(first_risk_breach_bar, dtype=np.int16)
    valid = np.asarray(valid_mask, dtype=bool)
    arrays = {
        "favorable_return": favorable,
        "adverse_return_to_peak": adverse,
        "opportunity_bar": opportunity,
        "first_risk_breach_bar": risk_breach,
        "valid_mask": valid,
    }
    shape = valid.shape
    if valid.ndim != 1:
        raise ValueError("no-time continuous target valid_mask必須是一維")
    for name, values in arrays.items():
        if np.asarray(values).ndim != 1 or np.asarray(values).shape != shape:
            raise ValueError(f"no-time continuous target component shape不一致: {name}")
    if bool(np.any(valid & (~np.isfinite(favorable) | ~np.isfinite(adverse)))):
        raise ValueError("no-time continuous target valid rows含非有限component")
    if bool(np.any(~valid & (np.isfinite(favorable) | np.isfinite(adverse)))):
        raise ValueError("no-time continuous target invalid rows的return component必須為NaN")
    if bool(np.any(valid & (opportunity < 1))) or bool(np.any(~valid & (opportunity != -1))):
        raise ValueError("no-time continuous target opportunity_bar sentinel不合法")
    if bool(np.any(valid & ((risk_breach == 0) | (risk_breach < -1)))) or bool(
        np.any(~valid & (risk_breach != -1))
    ):
        raise ValueError("no-time continuous target first_risk_breach_bar sentinel不合法")

    target = np.full(shape, np.nan, dtype=np.float32)
    target[valid] = (
        favorable[valid].astype(np.float64) / risk_budget
        - adverse[valid].astype(np.float64) / risk_budget
    ).astype(np.float32)
    if bool(np.any(valid & ~np.isfinite(target))):
        raise ValueError("no-time continuous target產生非有限值")
    return {
        "target_raw_r": target,
        "favorable_return": favorable.copy(),
        "adverse_return_to_peak": adverse.copy(),
        "opportunity_bar": opportunity.copy(),
        "first_risk_breach_bar": risk_breach.copy(),
        "valid_mask": valid.copy(),
    }


def build_strategy_aligned_no_time_contract(
    source_contract: dict[str, object],
) -> dict[str, object]:
    """Build the fixed no-time target contract from the strategy-aligned source contract."""

    if str(source_contract.get("target_id") or "") != STRATEGY_ALIGNED_TARGET_ID:
        raise ValueError("no-time source target contract必須是strategy-aligned v1")
    risk_budget = float(source_contract.get("risk_budget_return", math.nan))
    horizon = int(source_contract.get("horizon_bars", -1))
    if not math.isfinite(risk_budget) or risk_budget <= 0.0 or horizon < 2:
        raise ValueError("no-time source target contract的risk budget或horizon不合法")
    return {
        "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
        "target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        "source_target_id": STRATEGY_ALIGNED_TARGET_ID,
        "objective_family": "continuous_strategy_aligned_opportunity_no_time",
        "information_source": "validated_11a_component_arrays",
        "horizon_bars": horizon,
        "risk_budget_return": risk_budget,
        "minimum_opportunity_return": float(
            source_contract.get("minimum_opportunity_return", math.nan)
        ),
        "formula": (
            "favorable_return_before_first_risk_breach / risk_budget_return "
            "- adverse_return_required_to_reach_that_peak / risk_budget_return"
        ),
        "peak_rule": source_contract.get("peak_rule"),
        "risk_rule": source_contract.get("risk_rule"),
        "adverse_scope": source_contract.get("adverse_scope"),
        "no_safe_bar_rule": source_contract.get("no_safe_bar_rule"),
        "time_penalty_included": False,
        "normalization": "none",
        "clipping": "none",
        "split_derived_parameters": False,
        "oos_fitted_parameters": False,
        "prior_iterative_oos_hypothesis_informed": True,
        "current_audit_oos_rows_evaluated": False,
    }

def resolve_continuous_target_dir(
    project_root: str | Path,
    filter_id: str,
    *,
    target_id: str = STRATEGY_ALIGNED_TARGET_ID,
) -> Path:
    from filters.breakout_quality.paths import resolve_filter_output_dir

    return (
        resolve_filter_output_dir(project_root, filter_id=filter_id)
        / "continuous_targets"
        / str(target_id)
    )


def _artifact_record_identity(record: object) -> tuple[str, int, str] | None:
    if not isinstance(record, dict):
        return None
    try:
        return (
            str(record.get("filename") or ""),
            int(record.get("size_bytes", -1)),
            str(record.get("sha256") or "").lower(),
        )
    except (TypeError, ValueError):
        return None


def _validate_target_dataset_artifact_source(
    *,
    project_root: str | Path,
    filter_id: str,
    target_id: str,
    manifest: dict[str, object],
    expected_dataset_artifacts: dict[str, object],
) -> None:
    """Bind target arrays to the exact indexed Dataset artifacts that produced them."""

    source_records = manifest.get("dataset_artifact_source")
    if not isinstance(source_records, dict):
        source_target = manifest.get("source_target")
        if not isinstance(source_target, dict):
            raise ValueError("continuous target manifest缺少dataset artifact來源")
        source_target_id = str(source_target.get("target_id") or "").strip()
        source_manifest_record = source_target.get("manifest")
        if not source_target_id or not isinstance(source_manifest_record, dict):
            raise ValueError("continuous target manifest.source_target不完整")
        source_manifest_path = (
            resolve_continuous_target_dir(
                project_root,
                filter_id,
                target_id=source_target_id,
            )
            / TARGET_MANIFEST_FILENAME
        )
        expected_source_manifest_identity = _artifact_record_identity(source_manifest_record)
        if expected_source_manifest_identity is None or not source_manifest_path.is_file():
            raise ValueError("continuous target source manifest不存在或metadata不完整")
        actual_source_manifest_identity = (
            source_manifest_path.name,
            int(source_manifest_path.stat().st_size),
            compute_file_sha256(source_manifest_path).lower(),
        )
        if actual_source_manifest_identity != expected_source_manifest_identity:
            raise ValueError("continuous target source manifest已改變；請重建目前Target")
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("無法讀取continuous target source manifest") from exc
        if not isinstance(source_manifest, dict):
            raise ValueError("continuous target source manifest根節點必須是object")
        source_records = source_manifest.get("dataset_artifact_source")

    if not isinstance(source_records, dict) or not source_records:
        raise ValueError("continuous target manifest缺少可驗證的dataset artifact來源")
    for artifact_name, stored_record in source_records.items():
        current_record = expected_dataset_artifacts.get(str(artifact_name))
        stored_identity = _artifact_record_identity(stored_record)
        current_identity = _artifact_record_identity(current_record)
        if stored_identity is None or current_identity is None:
            raise ValueError(
                f"continuous target dataset artifact metadata不完整: {artifact_name}"
            )
        if stored_identity != current_identity:
            raise ValueError(
                f"continuous target與目前Dataset artifact不一致: {artifact_name}；請重建Target"
            )


def load_validated_continuous_target_manifest(
    project_root: str | Path,
    filter_id: str,
    *,
    target_id: str = STRATEGY_ALIGNED_TARGET_ID,
    expected_group_count: int | None = None,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate target identity, source binding and artifact metadata without loading arrays."""

    target_dir = resolve_continuous_target_dir(
        project_root,
        filter_id,
        target_id=target_id,
    )
    manifest_path = target_dir / TARGET_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到 continuous target manifest: {manifest_path}；"
            "請先由模型研究workflow建立目前設定的Continuous Target"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 continuous target manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("continuous target manifest 根節點必須是 object")
    if int(manifest.get("schema_version", -1)) != CONTINUOUS_TARGET_SCHEMA_VERSION:
        raise ValueError("continuous target schema version 不相容")
    if str(manifest.get("filter_id") or "").strip() != str(filter_id).strip():
        raise ValueError("continuous target manifest.filter_id 不一致")
    contract = manifest.get("target_contract")
    if not isinstance(contract, dict) or str(contract.get("target_id") or "") != str(target_id):
        raise ValueError("continuous target manifest.target_contract 不一致")
    if expected_group_count is not None and int(manifest.get("group_count", -1)) != int(expected_group_count):
        raise ValueError("continuous target group_count 與dataset不一致")
    if expected_dataset_policy is not None and manifest.get("dataset_policy") != expected_dataset_policy:
        raise ValueError("continuous target dataset_policy 與目前dataset不一致")
    if expected_dataset_artifacts is not None:
        _validate_target_dataset_artifact_source(
            project_root=project_root,
            filter_id=filter_id,
            target_id=target_id,
            manifest=manifest,
            expected_dataset_artifacts=expected_dataset_artifacts,
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("continuous target manifest缺少artifacts")
    required = {
        "target_raw_r": target_dir / TARGET_RAW_FILENAME,
        "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
    }
    for name, path in required.items():
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"continuous target artifacts缺少{name}")
        if str(record.get("filename") or "") != path.name or not path.is_file():
            raise ValueError(f"continuous target artifact不存在或filename不一致: {name}")
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"continuous target artifact size不一致: {name}")
        if str(record.get("sha256") or "").lower() != compute_file_sha256(path).lower():
            raise ValueError(f"continuous target artifact SHA256不一致: {name}")
    return manifest


def load_validated_continuous_target_arrays(
    project_root: str | Path,
    filter_id: str,
    *,
    target_id: str = STRATEGY_ALIGNED_TARGET_ID,
    expected_group_count: int | None = None,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Load target/valid arrays with strict manifest and hash validation."""

    manifest = load_validated_continuous_target_manifest(
        project_root,
        filter_id,
        target_id=target_id,
        expected_group_count=expected_group_count,
        expected_dataset_policy=expected_dataset_policy,
        expected_dataset_artifacts=expected_dataset_artifacts,
    )
    target_dir = resolve_continuous_target_dir(
        project_root,
        filter_id,
        target_id=target_id,
    )
    required = {
        "target_raw_r": target_dir / TARGET_RAW_FILENAME,
        "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
    }
    target = np.load(required["target_raw_r"], allow_pickle=False)
    valid_mask = np.load(required["valid_mask"], allow_pickle=False)
    if target.ndim != 1 or valid_mask.ndim != 1 or target.shape != valid_mask.shape:
        raise ValueError("continuous target array shape不合法")
    if expected_group_count is not None and len(target) != int(expected_group_count):
        raise ValueError("continuous target array長度與dataset不一致")
    valid = np.asarray(valid_mask, dtype=bool)
    values = np.asarray(target, dtype=np.float32)
    if bool(np.any(valid & ~np.isfinite(values))):
        raise ValueError("continuous target valid rows含NaN或infinite")
    if bool(np.any(~valid & np.isfinite(values))):
        raise ValueError("continuous target invalid rows必須為NaN")
    return manifest, values, valid


def load_validated_continuous_target_component_arrays(
    project_root: str | Path,
    filter_id: str,
    *,
    target_id: str = STRATEGY_ALIGNED_TARGET_ID,
    expected_group_count: int | None = None,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load all continuous-target component arrays with strict manifest validation."""

    manifest, target, valid_mask = load_validated_continuous_target_arrays(
        project_root,
        filter_id,
        target_id=target_id,
        expected_group_count=expected_group_count,
        expected_dataset_policy=expected_dataset_policy,
        expected_dataset_artifacts=expected_dataset_artifacts,
    )
    target_dir = resolve_continuous_target_dir(
        project_root,
        filter_id,
        target_id=target_id,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("continuous target manifest缺少artifacts")
    component_specs = {
        "favorable_return": TARGET_FAVORABLE_RETURN_FILENAME,
        "adverse_return_to_peak": TARGET_ADVERSE_RETURN_FILENAME,
        "opportunity_bar": TARGET_OPPORTUNITY_BAR_FILENAME,
        "first_risk_breach_bar": TARGET_RISK_BREACH_BAR_FILENAME,
    }
    arrays: dict[str, np.ndarray] = {
        "target_raw_r": np.asarray(target, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
    }
    for name, filename in component_specs.items():
        path = target_dir / filename
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"continuous target artifacts缺少{name}")
        if str(record.get("filename") or "") != filename or not path.is_file():
            raise ValueError(f"continuous target component不存在或filename不一致: {name}")
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"continuous target component size不一致: {name}")
        if str(record.get("sha256") or "").lower() != compute_file_sha256(path).lower():
            raise ValueError(f"continuous target component SHA256不一致: {name}")
        arrays[name] = np.load(path, allow_pickle=False)

    group_count = len(arrays["target_raw_r"])
    for name, values in arrays.items():
        if np.asarray(values).ndim != 1 or len(values) != group_count:
            raise ValueError(f"continuous target component shape不一致: {name}")
    valid = arrays["valid_mask"]
    for name in ("favorable_return", "adverse_return_to_peak"):
        values = np.asarray(arrays[name], dtype=np.float64)
        if bool(np.any(valid & ~np.isfinite(values))):
            raise ValueError(f"continuous target valid rows含非有限{name}")
        if bool(np.any(~valid & np.isfinite(values))):
            raise ValueError(f"continuous target invalid rows必須為NaN: {name}")
    opportunity = np.asarray(arrays["opportunity_bar"], dtype=np.int64)
    if bool(np.any(valid & (opportunity < 1))):
        raise ValueError("continuous target valid rows的opportunity_bar必須>=1")
    if bool(np.any(~valid & (opportunity != -1))):
        raise ValueError("continuous target invalid rows的opportunity_bar必須為-1")
    risk_breach = np.asarray(arrays["first_risk_breach_bar"], dtype=np.int64)
    if bool(np.any(valid & (risk_breach < -1))) or bool(np.any(valid & (risk_breach == 0))):
        raise ValueError("continuous target valid rows的first_risk_breach_bar只允許-1或>=1")
    if bool(np.any(~valid & (risk_breach != -1))):
        raise ValueError("continuous target invalid rows的first_risk_breach_bar必須為-1")
    return manifest, arrays


__all__ = [
    "CONTINUOUS_TARGET_SCHEMA_VERSION",
    "DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID",
    "DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID",
    "DAILY_OPPORTUNITY_NO_TIME_TARGET_ID",
    "STRATEGY_ALIGNED_TARGET_ID",
    "STRATEGY_ALIGNED_NO_TIME_TARGET_ID",
    "TARGET_ADVERSE_RETURN_FILENAME",
    "TARGET_AUDIT_JSON_FILENAME",
    "TARGET_AUDIT_MARKDOWN_FILENAME",
    "TARGET_DAILY_CSV_FILENAME",
    "TARGET_FAVORABLE_RETURN_FILENAME",
    "TARGET_MANIFEST_FILENAME",
    "TARGET_OPPORTUNITY_BAR_FILENAME",
    "TARGET_RAW_FILENAME",
    "TARGET_RISK_BREACH_BAR_FILENAME",
    "TARGET_TRADE_MATCHES_CSV_FILENAME",
    "TARGET_VALID_MASK_FILENAME",
    "StrategyAlignedContinuousTargetResult",
    "StrategyAlignedContinuousTargetSpec",
    "build_daily_full_horizon_opportunity_contract",
    "build_daily_full_horizon_pure_mfe_contract",
    "build_daily_opportunity_no_time_contract",
    "build_strategy_aligned_group_targets",
    "build_strategy_aligned_no_time_contract",
    "build_strategy_aligned_no_time_group_targets",
    "load_validated_continuous_target_manifest",
    "load_validated_continuous_target_arrays",
    "load_validated_continuous_target_component_arrays",
    "resolve_continuous_target_dir",
    "daily_full_horizon_opportunity_target_from_cached_path",
    "daily_full_horizon_pure_mfe_target_from_cached_path",
    "daily_opportunity_no_time_target_from_cached_path",
    "strategy_aligned_target_from_cached_path",
]
