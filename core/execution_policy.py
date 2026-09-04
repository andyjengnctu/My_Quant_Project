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
