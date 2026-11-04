"""Canonical execution-only policies for registered hindsight Oracle arms.

Price/MinTrade score-source planning and fingerprinting must never depend on
optional Portfolio admission semantics. Policy is verified against the single
canonical Audit Registry, not hard-coded into the common score producer.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.audit_registry import AUDIT_REUSABLE_REPORTS


class OracleExecutionPolicyError(ValueError):
    """An execution-only policy does not match its registered semantics."""


def execution_gate_policies(definition, plan: Mapping[str, Any]) -> dict[str, str]:
    """Reject any extra non-score execution gate under formal E13 Oracle.

    Future raw broker NetPnL may construct the signed perfect Positive *score*,
    but it must never be consulted again by the Portfolio candidate gate.
    """
    canonical = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]
    approved = {str(row["target"]): row for row in canonical["oracle_comparisons"]}
    selected = {str(row.get("target") or ""): row for row in
                definition.source.get("oracle_comparisons", ()) if row.get("enabled", True)}
    if set(selected) != set(plan["selectors"]):
        raise OracleExecutionPolicyError("selected Oracle target/selector contract differs")
    for target, entry in selected.items():
        if target not in approved:
            raise OracleExecutionPolicyError(f"unregistered Oracle execution target: {target}")
        if entry.get("positive_gate") is not None or approved[target].get("positive_gate") is not None:
            raise OracleExecutionPolicyError(
                f"E13 prohibits a raw-future execution gate: {target}")
    return {}
