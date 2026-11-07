"""Research-side delegation of approved Oracle P16 prerequisites.

The Target Replay plan is owned by the read-only Oracle. After the single
Research [1] authorization, this coordinator asks the *canonical* Optimizer
to ensure only the P16 seeds already marked as missing. No separate P16 menu,
parameter fitting, or alternate Oracle Replay engine lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.audit_policy import get_reusable_audit_definitions
from services.audit.target_oracle_romd import inspect_oracle_parameter_prerequisites
from services.optimizer.strategy_param_service import (
    ensure_fixed_multiseed_robustness_strategy_parameter_artifact,
    resolve_fixed_multiseed_robustness_strategy_parameter_artifact,
)


def get_formal_oracle_definition(module_id: str):
    """Resolve the one registered Target validation definition."""
    definitions = tuple(
        item for item in get_reusable_audit_definitions(module_id, enabled_only=True)
        if item.report_type == "target_ground_truth_strategy_romd"
    )
    if len(definitions) != 1:
        raise RuntimeError("Target 驗證必須只有一個啟用的正式 Oracle 定義")
    return definitions[0]


def ensure_authorized_oracle_parameters(
    *, project_root: Path, definition, approved_plan: Mapping[str, Any],
) -> None:
    """Build only authorized missing canonical Optimizer seeds for each family.

    Distinct Min/Full parameter truth is inspected independently.  An invalid
    existing source stops ALL families before any Producer is started; missing
    sources are built by the SAME canonical Optimizer, never by this consumer.
    """
    from services.audit.target_oracle_romd import _oracle_strategy_scopes, _baseline_target
    root = Path(project_root).resolve()
    planned = approved_plan.get("parameter_plans")
    if not isinstance(planned, Mapping):
        raise RuntimeError("Oracle approval missing parameter-family source plans")
    inspections = {}
    for _scope_name, scoped in _oracle_strategy_scopes(definition):
        baseline = _baseline_target(scoped)
        current = inspect_oracle_parameter_prerequisites(scoped, project_root=root)
        expected = planned.get(baseline)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"Oracle approval has no {baseline} parameter plan")
        rows = current["rows"]
        seeds = tuple(int(row["seed"]) for row in rows)
        if (seeds != tuple(expected["seed_set"]) or
                dict(current["producer_kwargs"]) != dict(expected["producer_kwargs"])):
            raise RuntimeError(f"Oracle {baseline} benchmark/parameter source changed after approval")
        if baseline in inspections:
            earlier = inspections[baseline][0]
            if (earlier["producer_kwargs"] != current["producer_kwargs"] or
                    earlier["rows"] != current["rows"]):
                raise RuntimeError(f"Oracle {baseline} parallel scopes have unequal parameter identity")
            continue
        allowed = set(map(int, expected["missing"]))
        if not allowed.issubset(seeds):
            raise RuntimeError(f"Oracle {baseline} unauthorized optimizer seeds")
        for row in rows:
            seed = int(row["seed"])
            if not row["ready"] and seed not in allowed:
                raise RuntimeError(f"Oracle {baseline} seed={seed} 原為 REUSE 但參數來源已改變；停止")
            if not row["ready"] and row["action"] not in {"BUILD", "REPAIR"}:
                raise RuntimeError(f"Oracle {baseline} seed={seed} 身分衝突：{row['status']}，禁止覆寫")
        inspections[baseline] = (current, scoped, tuple(int(row["seed"]) for row in rows if int(row["seed"]) in allowed))
    if set(inspections) != set(planned):
        raise RuntimeError("Oracle parameter-family inventory changed after approval")
    for baseline, (inspection, scoped, missing) in inspections.items():
        for seed in missing:
            current = resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
                root, replication_seed=seed, **inspection["producer_kwargs"],
            )
            if current["ready"]:
                print(f"[{baseline} REUSE] seed={seed} canonical parameter truth already READY")
                continue
            if current["action"] not in {"BUILD", "REPAIR"}:
                raise RuntimeError(f"{baseline} seed={seed} unauthorized Producer action: {current['status']}")
            print(f"[{baseline} {current['action']}] seed={seed} | canonical Optimizer", flush=True)
            ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
                root, replication_seed=seed, **inspection["producer_kwargs"],
                resume_parameter_training=True, quiet=False,
            )
        final = inspect_oracle_parameter_prerequisites(scoped, project_root=root)
        if (final["producer_kwargs"] != inspection["producer_kwargs"] or
                tuple(int(row["seed"]) for row in final["rows"]) !=
                tuple(int(row["seed"]) for row in inspection["rows"])):
            raise RuntimeError(f"Oracle {baseline} source or seeds changed after Producer")
        if not all(row["ready"] for row in final["rows"]):
            raise RuntimeError(f"Oracle {baseline} parameter inputs are not READY after Producer")
        if missing:
            print(f"[{baseline} READY] missing parameters built and certified")
