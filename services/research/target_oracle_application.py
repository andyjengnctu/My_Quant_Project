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
    """Build P16 missing at approval; reject drift and never overwrite conflicts.

    The caller must already have shown and obtained consent for the Target x
    Seed Oracle plan. A seed absent from that plan cannot become an implicit
    Optimizer job; a source-contract change stops the work before fitting.
    """
    root = Path(project_root).resolve()
    inspection = inspect_oracle_parameter_prerequisites(definition, project_root=root)
    rows = inspection["rows"]
    seeds = tuple(int(row["seed"]) for row in rows)
    permitted = tuple(int(seed) for seed in approved_plan["missing_p16"])
    if (seeds != tuple(approved_plan["seed_set"])
            or dict(inspection["producer_kwargs"]) != dict(approved_plan["producer_kwargs"])):
        raise RuntimeError("Target 授權後 P16／Benchmark 身分變更，未執行 Optimizer 或 Replay")
    # Existing P16s were not authorized for rebuilding. They must remain
    # complete; only the originally missing seeds may need a Producer action.
    for row in rows:
        if not row["ready"] and int(row["seed"]) not in permitted:
            raise RuntimeError(f"seed={row['seed']} 原為 REUSE 但 P16 已改變，停止")
        if not row["ready"] and row["action"] not in {"BUILD", "REPAIR"}:
            raise RuntimeError(f"P16 seed={row['seed']} 身分衝突：{row['status']}，禁止覆寫")

    for seed in permitted:
        current = resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
            root, replication_seed=seed, **inspection["producer_kwargs"],
        )
        if current["ready"]:
            print(f"[P16 REUSE] seed={seed} 已由其他流程完成")
            continue
        if current["action"] not in {"BUILD", "REPAIR"}:
            raise RuntimeError(
                f"P16 seed={seed} status={current['status']}，非可直接建立的工件，停止")
        print(f"[P16 {current['action']}] seed={seed} | canonical Optimizer", flush=True)
        ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
            root, replication_seed=seed, **inspection["producer_kwargs"],
            resume_parameter_training=True, quiet=False,
        )
    final = inspect_oracle_parameter_prerequisites(definition, project_root=root)
    if (dict(final["producer_kwargs"]) != dict(inspection["producer_kwargs"])
            or tuple(int(row["seed"]) for row in final["rows"]) != seeds):
        raise RuntimeError("P16 Producer 執行後來源或 Seeds 身分改變，停止 Target Replay")
    if not all(row["ready"] for row in final["rows"]):
        pending = ", ".join(
            f"seed={row['seed']}:{row['status']}"
            for row in final["rows"] if not row["ready"]
        )
        raise RuntimeError("P16 Producer 完成後仍未 READY：" + pending)
    if permitted:
        print("[P16 READY] 缺少的 P16 已建立／核對；執行已授權的 Target Replay。")
