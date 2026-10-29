"""Canonical execution-only policies for registered hindsight Oracle arms.

Price/MinTrade score-source planning and fingerprinting must never depend on
optional Portfolio admission semantics. Policy is verified against the single
canonical Audit Registry, not hard-coded into the common score producer.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.audit_registry import AUDIT_REUSABLE_REPORTS
from core.oracle_positive_gate import POSITIVE_FULLCYCLE_NETPNL_GATE


class OracleExecutionPolicyError(ValueError):
    """An execution-only policy does not match its registered semantics."""


def execution_gate_policies(definition, plan: Mapping[str, Any]) -> dict[str, str]:
    """Return target-specific approved gates, independent of score producers.

    A Direct full-cycle broker-ledger NetPnL rank can opt into the hindsight
    gate. Adding one such arm changes neither any producer function source nor
    any other arm's execution policy, score, or Replay identity.
    """
    canonical = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]
    approved = {str(row["target"]): row for row in canonical["oracle_comparisons"]}
    selected = {str(row.get("target") or ""): row for row in
                definition.source.get("oracle_comparisons", ()) if row.get("enabled", True)}
    if set(selected) != set(plan["selectors"]):
        raise OracleExecutionPolicyError("selected Oracle target/selector contract differs")
    gates = {}
    for target, entry in selected.items():
        gate = entry.get("positive_gate")
        precedent = approved.get(target)
        if precedent is None:
            raise OracleExecutionPolicyError(f"unregistered Oracle execution target: {target}")
        if gate != precedent.get("positive_gate"):
            raise OracleExecutionPolicyError(f"Oracle execution policy changed for registered target: {target}")
        if gate is None:
            continue
        score_source = plan["daily_sources"].get(target)
        recipe = plan["daily_recipes"].get(score_source)
        if (gate != POSITIVE_FULLCYCLE_NETPNL_GATE or
                entry.get("selector") != "direct" or
                plan["selectors"].get(target) != "direct" or
                entry.get("score_source") != score_source or
                recipe is None or recipe[0] != "daily_trade_path_net_pnl" or
                recipe[2] is not None):
            raise OracleExecutionPolicyError(
                f"NON-PIT admission requires canonical full-cycle MinTrade NetPnL Direct: {target}")
        gates[target] = gate
    return gates
