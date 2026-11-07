"""Read-only Research [1] prerequisite validation for Target Oracle.

Oracle consumes canonical Optimizer parameter truth but never builds it. A
missing Min/Full seed prevents Replay until the owning parameter workflow has
independently produced its legal artifact (PROJECT_SETTINGS E13).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.audit_policy import get_reusable_audit_definitions
from services.audit.target_oracle_romd import inspect_oracle_parameter_prerequisites


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
    """Verify, never build, all Min/Full inputs for an approved Oracle plan.

    Kept as the Research application's existing integration entry. Its only
    effect is to fail closed if any canonical parameter seed is missing,
    conflicting, or has changed since the displayed plan. The Oracle may
    authorize Portfolio Replay, never parameter optimization.
    """
    from services.audit.target_oracle_romd import _oracle_strategy_scopes, _baseline_target
    root = Path(project_root).resolve()
    planned = approved_plan.get("parameter_plans")
    if not isinstance(planned, Mapping):
        raise RuntimeError("Oracle 缺少正式參數家族檢查計畫")
    inspections = {}
    unavailable = []
    for _scope_name, scoped in _oracle_strategy_scopes(definition):
        baseline = _baseline_target(scoped)
        current = inspect_oracle_parameter_prerequisites(scoped, project_root=root)
        expected = planned.get(baseline)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"Oracle 缺少 {baseline} 參數來源檢查計畫")
        rows = tuple(current["rows"])
        seeds = tuple(int(row["seed"]) for row in rows)
        if (seeds != tuple(int(seed) for seed in expected["seed_set"]) or
                dict(current["producer_kwargs"]) != dict(expected["producer_kwargs"])):
            raise RuntimeError(f"Oracle {baseline} Benchmark／參數來源與預覽不一致，停止")
        if baseline in inspections:
            if inspections[baseline] != (current["producer_kwargs"], rows):
                raise RuntimeError(f"Oracle {baseline} 同家族來源不一致，停止")
            continue
        inspections[baseline] = (current["producer_kwargs"], rows)
        missing = tuple(int(row["seed"]) for row in rows if not row["ready"])
        if missing != tuple(int(seed) for seed in expected["missing"]):
            raise RuntimeError(f"Oracle {baseline} 參數 READY 狀態已改變，請重新預覽")
        for row in rows:
            if not row["ready"]:
                if row["action"] not in {"BUILD", "REPAIR"}:
                    raise RuntimeError(
                        f"Oracle {baseline} seed={row['seed']} 身分衝突：{row['status']}，禁止覆寫"
                    )
                unavailable.append(f"{baseline} seed={row['seed']} ({row['status']})")
    if set(inspections) != set(planned):
        raise RuntimeError("Oracle 參數家族清單與授權預覽不一致，停止")
    if unavailable:
        raise RuntimeError(
            "Oracle 前置策略參數尚未 READY；[1] 不執行 Optimizer，"
            "須由對應參數科學契約的正式 Producer 另行建立：" + "; ".join(unavailable)
        )
