"""Read-only resolver for canonical multi-arm strategy-comparison artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.audit import AuditDefinition


@dataclass(frozen=True)
class StrategyCompareArmArtifacts:
    arm_id: str
    run_dir: Path
    pair_dir: Path
    prefix: str
    summary: dict[str, Any]
    arm: dict[str, Any]

    @property
    def trades_path(self) -> Path:
        return self.pair_dir / f"{self.prefix}_trades.csv"

    @property
    def equity_path(self) -> Path:
        return self.pair_dir / f"{self.prefix}_equity.csv"

    @property
    def capacity_path(self) -> Path:
        return self.pair_dir / f"{self.prefix}_daily_capacity.csv"

    @property
    def selected_path(self) -> Path:
        return self.pair_dir / f"{self.prefix}_selected_buys.csv"

    @property
    def orderable_path(self) -> Path:
        return self.pair_dir / f"{self.prefix}_orderable_candidates.csv"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少JSON工件: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {path}")
    return payload


def resolve_strategy_compare_run(root: Path, definition: AuditDefinition) -> tuple[Path, dict[str, Any]]:
    if str(definition.source.get("kind") or "") != "strategy_compare":
        raise ValueError(f"{definition.audit_id}.source.kind必須是strategy_compare")
    run_setting = str(definition.source.get("run") or "").strip()
    if not run_setting:
        raise ValueError(f"{definition.audit_id}.source.run不可空白")
    if run_setting == "latest":
        manifest_path = root / "outputs" / "strategy_compare" / "latest" / "manifest.json"
        manifest = read_json(manifest_path)
        run_value = str(manifest.get("run_dir") or "").strip()
        if not run_value:
            raise ValueError("strategy_compare latest manifest缺少run_dir")
        run_dir = (root / Path(run_value)).resolve()
    else:
        run_dir = (root / Path(run_setting)).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Audit strategy_compare run必須位於專案root內") from exc
    result = read_json(run_dir / "strategy_comparison.json")
    if str(result.get("status") or "") != "COMPLETED":
        raise ValueError("strategy_compare結果尚未完成")
    return run_dir, result


def _runtime_prefix(arm: dict[str, Any]) -> str:
    if not bool(arm.get("dl_enabled")):
        return "no_filter"
    runtime = str(arm.get("dl_runtime_mode") or "")
    if runtime == "hard-filter":
        return "quality_filter"
    if runtime in {
        "resource-aware-binary",
        "resource-aware-binary-basket",
        "resource-aware-continuous",
    }:
        return "score_ranking"
    raise ValueError(f"不支援的strategy compare runtime: {runtime!r}")


def _pair_group_id(arm: dict[str, Any]) -> str:
    return (
        f"{str(arm.get('param_source') or '').strip()}__"
        f"{str(arm.get('rule_policy') or '').strip()}__"
        f"{str(arm.get('dl_id') or '').strip()}__"
        f"{str(arm.get('dl_runtime_mode') or '').strip().replace('-', '_')}"
    )


def resolve_arm_artifacts(
    *,
    run_dir: Path,
    result: dict[str, Any],
    arm_id: str,
    preferred_pair_arm_id: str | None = None,
) -> StrategyCompareArmArtifacts:
    settings = dict(result.get("settings") or {})
    arms = dict(settings.get("arms") or {})
    arm = arms.get(str(arm_id))
    if not isinstance(arm, dict):
        raise ValueError(f"strategy_compare結果不存在arm: {arm_id}")
    scenarios = dict(result.get("scenarios") or {})
    summary = scenarios.get(str(arm_id))
    if not isinstance(summary, dict):
        raise ValueError(f"strategy_compare結果缺少scenario summary: {arm_id}")

    if bool(arm.get("dl_enabled")):
        pair_arm = arm
    else:
        pair_arm = None
        if preferred_pair_arm_id:
            preferred = arms.get(str(preferred_pair_arm_id))
            if (
                isinstance(preferred, dict)
                and bool(preferred.get("dl_enabled"))
                and preferred.get("param_source") == arm.get("param_source")
                and preferred.get("rule_policy") == arm.get("rule_policy")
            ):
                pair_arm = preferred
        if pair_arm is None:
            for candidate in arms.values():
                if not isinstance(candidate, dict) or not bool(candidate.get("enabled", False)):
                    continue
                if not bool(candidate.get("dl_enabled")):
                    continue
                if (
                    candidate.get("param_source") == arm.get("param_source")
                    and candidate.get("rule_policy") == arm.get("rule_policy")
                ):
                    pair_arm = candidate
                    break
        if pair_arm is None:
            raise ValueError(f"找不到{arm_id}共用baseline對應的DL-on pair")

    pair_dir = run_dir / "pairs" / _pair_group_id(pair_arm)
    if not pair_dir.is_dir():
        raise FileNotFoundError(f"strategy_compare pair目錄不存在: {pair_dir}")
    return StrategyCompareArmArtifacts(
        arm_id=str(arm_id),
        run_dir=run_dir,
        pair_dir=pair_dir,
        prefix=_runtime_prefix(arm),
        summary=dict(summary),
        arm=dict(arm),
    )


__all__ = [
    "StrategyCompareArmArtifacts",
    "read_json",
    "resolve_arm_artifacts",
    "resolve_strategy_compare_run",
]
