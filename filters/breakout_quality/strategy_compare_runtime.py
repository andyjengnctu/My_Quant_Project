"""Strategy Compare runtime grouping and arm execution-spec SSOT."""

from __future__ import annotations

from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_HARD_FILTER,
    StrategyComparisonArm,
    StrategyComparisonSettings,
    resolve_strategy_comparison_arm_param_policy,
)
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
)


def _arm_runtime_spec(arm: StrategyComparisonArm) -> dict[str, str]:
    mode = str(arm.dl_runtime_mode or "")
    if mode == STRATEGY_DL_RUNTIME_MODE_HARD_FILTER:
        return {
            "comparison_mode": COMPARISON_MODE_HARD_FILTER,
            "ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
            "active_key": "quality_filter",
            "yearly_key": "quality_filter_return_pct",
            "active_trades_filename": "quality_filter_trades.csv",
        }
    if mode in RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES:
        return {
            "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
            "ranking_policy": mode,
            "active_key": "score_ranking",
            "yearly_key": "score_ranking_return_pct",
            "active_trades_filename": "score_ranking_trades.csv",
        }
    raise ValueError(f"不支援的DL runtime mode: arm={arm.arm_id}, mode={mode!r}")


def _execution_pairs(
    settings: StrategyComparisonSettings,
) -> tuple[tuple[str, str, StrategyComparisonArm, StrategyComparisonArm], ...]:
    """Return one replay pair per enabled DL source in config order.

    A ``param_source`` / effective ``param_policy`` / ``rule_policy`` group owns
    one shared DL-off baseline.  This keeps base-finalist-best DL arms paired only
    with the matching base-finalist-best baseline even when additional DL-off
    parameter-policy references are displayed in the same Compare Suite.
    """
    grouped: dict[
        tuple[str, str, str],
        dict[str, StrategyComparisonArm | list[StrategyComparisonArm] | None],
    ] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for arm in settings.enabled_arms:
        param_policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
        key = (arm.param_source, param_policy, arm.rule_policy)
        if key not in grouped:
            grouped[key] = {"off": None, "on": []}
            ordered_keys.append(key)
        group = grouped[key]
        if not arm.dl_enabled:
            if group["off"] is not None:
                raise ValueError(
                    "啟用比較群組重複定義DL-off基準: "
                    f"{arm.param_source}/{param_policy}/{arm.rule_policy}"
                )
            group["off"] = arm
            continue
        on_arms = group["on"]
        if not isinstance(on_arms, list):
            raise TypeError("strategy comparison execution group contract錯誤")
        if not arm.dl_id:
            raise ValueError(f"DL-on arm缺少dl_id: {arm.arm_id}")
        if any(
            existing.dl_id == arm.dl_id
            and existing.dl_runtime_mode == arm.dl_runtime_mode
            and dict(existing.dl_runtime_options or {}) == dict(arm.dl_runtime_options or {})
            for existing in on_arms
        ):
            raise ValueError(
                "啟用比較群組重複定義相同DL source/runtime contract: "
                f"{arm.param_source}/{param_policy}/{arm.rule_policy}/{arm.dl_id}/{arm.dl_runtime_mode}/"
                f"{dict(arm.dl_runtime_options or {})!r}"
            )
        on_arms.append(arm)

    pairs: list[tuple[str, str, StrategyComparisonArm, StrategyComparisonArm]] = []
    for param_source, param_policy, rule_policy in ordered_keys:
        group = grouped[(param_source, param_policy, rule_policy)]
        off_arm = group["off"]
        on_arms = group["on"]
        if not isinstance(on_arms, list):
            raise TypeError("strategy comparison execution group contract錯誤")
        if not on_arms:
            continue
        if not isinstance(off_arm, StrategyComparisonArm):
            raise ValueError(
                "啟用DL比較群組缺少同parameter-policy的DL-off基準: "
                f"{param_source}/{param_policy}/{rule_policy}"
            )
        for on_arm in on_arms:
            pairs.append((param_source, rule_policy, off_arm, on_arm))
    return tuple(pairs)


def strategy_comparison_execution_pairs(
    settings: StrategyComparisonSettings,
) -> tuple[tuple[str, str, StrategyComparisonArm, StrategyComparisonArm], ...]:
    """Public read-only view of the canonical Strategy Compare execution groups.

    Consumers that need to resolve one displayed arm back to the pair artifacts must
    reuse this grouping contract instead of reconstructing param-policy/rule-policy
    membership independently.
    """

    return _execution_pairs(settings)

def _standalone_baseline_arms(
    settings: StrategyComparisonSettings,
) -> tuple[StrategyComparisonArm, ...]:
    enabled = tuple(settings.enabled_arms)
    output: list[StrategyComparisonArm] = []
    for arm in enabled:
        if arm.dl_enabled:
            continue
        arm_param_policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
        has_enabled_on = any(
            other.dl_enabled
            and other.param_source == arm.param_source
            and resolve_strategy_comparison_arm_param_policy(settings, other) == arm_param_policy
            and other.rule_policy == arm.rule_policy
            for other in enabled
        )
        if not has_enabled_on:
            output.append(arm)
    return tuple(output)
