"""Research [1] Target Oracle prerequisites and canonical producer delegation.

The Oracle is still a hypothetical-score consumer, never an Optimizer. Its
approved dependency plan may delegate missing, legally buildable parameters to
the sole Optimizer producer before any Target or Portfolio Replay is allowed.
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
    # Both the Summary and Oracle entry must reject an active additional-
    # parameter-family DL arm whose Target binding was omitted.  The roster
    # determines membership, while the Compare registry remains the single
    # owner of the arm's actual execution and parameter contracts.
    from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
    from core.audit_registry import validate_oracle_compare_family_coverage
    from core.strategy_compare_policy import get_strategy_compare_suite
    from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS, STRATEGY_COMPARE_PROFILES

    suite_id = STRATEGY_COMPARE_PROFILES[STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE]["suite_id"]
    validate_oracle_compare_family_coverage(
        definitions[0].source,
        active_arm_ids=get_strategy_compare_suite(suite_id)["arm_ids"],
        arm_registry=STRATEGY_COMPARE_ARMS,
    )
    return definitions[0]


def ensure_authorized_oracle_parameters(
    *, project_root: Path, definition, approved_plan: Mapping[str, Any],
) -> dict[str, int]:
    """Build only approved missing canonical-family seeds, then reattest every source.

    The displayed, read-only plan is the authorization boundary. Validate all
    parameter families first; never begin fitting if any seed is in conflict or
    the plan changed. Delegate only BUILD/REPAIR to the canonical Optimizer,
    then fail closed if any seed remains uncertified. Replay is a later step.
    """
    from services.audit.target_oracle_romd import _oracle_strategy_scopes, _baseline_target
    from services.optimizer.strategy_param_service import (
        ensure_fixed_multiseed_robustness_strategy_parameter_artifact,
    )

    root = Path(project_root).resolve()
    planned = approved_plan.get("parameter_plans")
    if not isinstance(planned, Mapping):
        raise RuntimeError("Oracle 缺少正式參數家族檢查計畫")

    inspections: dict[str, tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = {}
    pending: list[tuple[str, int, dict[str, Any]]] = []
    for _scope_name, scoped in _oracle_strategy_scopes(definition):
        baseline = _baseline_target(scoped)
        current = inspect_oracle_parameter_prerequisites(scoped, project_root=root)
        expected = planned.get(baseline)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"Oracle 缺少 {baseline} 參數來源檢查計畫")
        rows = tuple(dict(row) for row in current["rows"])
        seeds = tuple(int(row["seed"]) for row in rows)
        kwargs = dict(current["producer_kwargs"])
        if (seeds != tuple(int(seed) for seed in expected["seed_set"])
                or kwargs != dict(expected["producer_kwargs"])):
            raise RuntimeError(f"Oracle {baseline} Benchmark／參數來源與預覽不一致，停止")
        if baseline in inspections:
            if inspections[baseline] != (kwargs, rows):
                raise RuntimeError(f"Oracle {baseline} 同家族來源不一致，停止")
            continue
        missing = tuple(int(row["seed"]) for row in rows if not row["ready"])
        if missing != tuple(int(seed) for seed in expected["missing"]):
            raise RuntimeError(f"Oracle {baseline} 參數 READY 狀態已改變，請重新預覽")
        inspections[baseline] = (kwargs, rows)
        for row in rows:
            if row["ready"]:
                continue
            if row["action"] not in {"BUILD", "REPAIR"}:
                raise RuntimeError(
                    f"Oracle {baseline} seed={row['seed']} 身分衝突：{row['status']}，禁止覆寫"
                )
            pending.append((baseline, int(row["seed"]), kwargs))
    if set(inspections) != set(planned):
        raise RuntimeError("Oracle 參數家族清單與授權預覽不一致，停止")

    for i, (baseline, seed, kwargs) in enumerate(pending, 1):
        print(f"[Oracle 參數前置 {i}/{len(pending)}] {baseline} seed={seed} → canonical Optimizer")
        result = ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
            root, replication_seed=seed, **kwargs,
            force_rebuild=False, resume_parameter_training=True, quiet=False,
        )
        if str(result.get("action") or "") not in {"REUSE", "BUILD", "REBUILD", "REPAIR"}:
            raise RuntimeError(f"Oracle {baseline} seed={seed} Optimizer未完成：{result.get('action')}")
        print(f"[Oracle 參數前置 {i}/{len(pending)}] {baseline} seed={seed} {result['action']}")

    # A successful producer call is not itself READY evidence: re-read the
    # canonical signed artifact/manifest for each scope before starting Replay.
    checked: set[str] = set()
    for _scope_name, scoped in _oracle_strategy_scopes(definition):
        baseline = _baseline_target(scoped)
        if baseline in checked:
            continue
        checked.add(baseline)
        current = inspect_oracle_parameter_prerequisites(scoped, project_root=root)
        expected_kwargs, expected_rows = inspections[baseline]
        if (dict(current["producer_kwargs"]) != expected_kwargs or
                tuple(int(row["seed"]) for row in current["rows"]) !=
                tuple(int(row["seed"]) for row in expected_rows)):
            raise RuntimeError(f"Oracle {baseline} 補建後Benchmark／參數來源漂移，停止 Replay")
        incomplete = [row for row in current["rows"] if not row["ready"]]
        if incomplete:
            first = incomplete[0]
            raise RuntimeError(
                f"Oracle {baseline} seed={first['seed']} 尚未通過簽章READY：{first['status']}"
            )

    return {"reused": sum(int(row["ready"]) for _kwargs, rows in inspections.values() for row in rows),
            "built_or_repaired": len(pending)}
