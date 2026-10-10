"""Runtime helpers for execution-policy config."""

from config.execution_policy import (
    EXECUTION_POLICY_PARAM_SPECS,
    RUNTIME_PARAM_DEFAULTS,
)


def build_execution_policy_snapshot():
    return {
        field_name: spec["default"]
        for field_name, spec in EXECUTION_POLICY_PARAM_SPECS.items()
    }


def build_runtime_param_snapshot():
    return dict(RUNTIME_PARAM_DEFAULTS)


def build_yuanta_research_execution_overrides() -> dict[str, float]:
    """Current Research broker-cash costs, from the shared settlement policy.

    The post-trade public discount does not create an automatic receivable.
    """
    from config.broker_accounting import (
        BROKER_FEE_RATE,
        BROKER_MIN_FEE_TWD,
        BROKER_STOCK_TAX_RATE,
    )

    return {
        "buy_fee": float(BROKER_FEE_RATE),
        "sell_fee": float(BROKER_FEE_RATE),
        "tax_rate": float(BROKER_STOCK_TAX_RATE),
        "min_fee": float(BROKER_MIN_FEE_TWD),
    }
