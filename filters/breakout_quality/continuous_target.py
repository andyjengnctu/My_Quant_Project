"""Versioned continuous outcome targets for breakout-quality research."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from filters.breakout_quality.contract import BreakoutQualityLabelPolicy

CONTINUOUS_TARGET_SCHEMA_VERSION = 1
STRATEGY_ALIGNED_TARGET_ID = "strategy_aligned_opportunity_r_v1"

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


__all__ = [
    "CONTINUOUS_TARGET_SCHEMA_VERSION",
    "STRATEGY_ALIGNED_TARGET_ID",
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
    "build_strategy_aligned_group_targets",
    "resolve_continuous_target_dir",
    "strategy_aligned_target_from_cached_path",
]
