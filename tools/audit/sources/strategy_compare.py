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


def read_json(path: Path, *, display_root: Path | None = None) -> dict[str, Any]:
    display_path = Path(path)
    if display_root is not None:
        try:
            display_path = Path(path).resolve().relative_to(Path(display_root).resolve())
        except ValueError:
            display_path = Path(path)
    display_text = display_path.as_posix()
    if not path.is_file():
        raise FileNotFoundError(f"缺少JSON工件: {display_text}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {display_text}")
    return payload


def _validated_completed_run(root: Path, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(run_dir).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Audit strategy_compare run必須位於專案root內") from exc
    result = read_json(resolved / "strategy_comparison.json", display_root=root)
    if str(result.get("status") or "") != "COMPLETED":
        raise ValueError("strategy_compare結果尚未完成")
    return resolved, result


def resolve_strategy_compare_run_selector(
    root: Path,
    selector: dict[str, Any],
    *,
    audit_id: str,
    required_arm_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(root).resolve()
    run_setting = str(selector.get("run") or "").strip()
    fingerprint = str(selector.get("config_fingerprint") or "").strip()
    if bool(run_setting) == bool(fingerprint):
        raise ValueError(
            f"{audit_id}.strategy_compare run selector必須二選一設定run或config_fingerprint"
        )
    if fingerprint:
        runs_root = root / "outputs" / "strategy_compare" / "runs"
        if not runs_root.is_dir():
            raise FileNotFoundError("缺少strategy_compare runs目錄")
        matches: list[tuple[Path, dict[str, Any]]] = []
        for candidate_dir in sorted(
            (path for path in runs_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        ):
            result_path = candidate_dir / "strategy_comparison.json"
            if not result_path.is_file():
                continue
            try:
                result = read_json(result_path, display_root=root)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if str(result.get("status") or "") != "COMPLETED":
                continue
            if str(result.get("config_fingerprint") or "").strip() != fingerprint:
                continue
            if required_arm_id:
                scenarios = dict(result.get("scenarios") or {})
                arms = dict(dict(result.get("settings") or {}).get("arms") or {})
                if required_arm_id not in scenarios or required_arm_id not in arms:
                    continue
            matches.append((candidate_dir.resolve(), result))
        if not matches:
            suffix = "" if not required_arm_id else f" 且包含arm={required_arm_id}"
            raise FileNotFoundError(
                f"找不到config_fingerprint={fingerprint}的已完成strategy_compare run{suffix}"
            )
        return _validated_completed_run(root, matches[0][0])

    if run_setting == "latest":
        manifest_path = root / "outputs" / "strategy_compare" / "latest" / "manifest.json"
        manifest = read_json(manifest_path, display_root=root)
        run_value = str(manifest.get("run_dir") or "").strip()
        if not run_value:
            raise ValueError("strategy_compare latest manifest缺少run_dir")
        run_dir = root / Path(run_value)
    else:
        run_dir = root / Path(run_setting)
    run_dir, result = _validated_completed_run(root, run_dir)
    if required_arm_id:
        scenarios = dict(result.get("scenarios") or {})
        arms = dict(dict(result.get("settings") or {}).get("arms") or {})
        if required_arm_id not in scenarios or required_arm_id not in arms:
            raise ValueError(f"strategy_compare run不存在arm: {required_arm_id}")
    return run_dir, result



def resolve_strategy_compare_profile_run_selector(
    root: Path,
    selector: dict[str, Any],
    *,
    audit_id: str,
    required_arm_ids: tuple[str, ...] = (),
) -> tuple[Path, dict[str, Any]]:
    """Resolve a completed Strategy Compare run within a configured profile namespace."""

    from config.strategy_compare import get_strategy_comparison_settings

    root = Path(root).resolve()
    profile_id = str(selector.get("profile_id") or "").strip()
    if not profile_id:
        raise ValueError(f"{audit_id}.source profile selector缺少profile_id")
    settings = get_strategy_comparison_settings(profile_id)
    run_setting = str(selector.get("run") or "").strip()
    fingerprint = str(selector.get("config_fingerprint") or "").strip()
    if bool(run_setting) == bool(fingerprint):
        raise ValueError(
            f"{audit_id}.strategy_compare profile run selector必須二選一設定run或config_fingerprint"
        )

    run_roots: list[Path] = []
    for raw in (settings.output_root, *settings.reuse_output_roots):
        candidate = (root / Path(str(raw)) / "runs").resolve()
        if candidate not in run_roots:
            run_roots.append(candidate)

    if fingerprint:
        matches: list[tuple[Path, dict[str, Any]]] = []
        for runs_root in run_roots:
            if not runs_root.is_dir():
                continue
            for candidate_dir in sorted(
                (path for path in runs_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            ):
                result_path = candidate_dir / "strategy_comparison.json"
                if not result_path.is_file():
                    continue
                try:
                    result = read_json(result_path, display_root=root)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    continue
                if str(result.get("status") or "") != "COMPLETED":
                    continue
                if str(result.get("config_fingerprint") or "").strip() != fingerprint:
                    continue
                scenarios = dict(result.get("scenarios") or {})
                arms = dict(dict(result.get("settings") or {}).get("arms") or {})
                if any(arm_id not in scenarios or arm_id not in arms for arm_id in required_arm_ids):
                    continue
                matches.append((candidate_dir.resolve(), result))
        if not matches:
            raise FileNotFoundError(
                f"找不到profile={profile_id}, config_fingerprint={fingerprint}的已完成strategy_compare run"
            )
        return _validated_completed_run(root, matches[0][0])

    if run_setting == "latest":
        latest_manifest = (root / Path(settings.output_root) / "latest" / "manifest.json").resolve()
        manifest = read_json(latest_manifest, display_root=root)
        run_value = str(manifest.get("run_dir") or "").strip()
        if not run_value:
            raise ValueError(f"strategy_compare profile={profile_id} latest manifest缺少run_dir")
        run_dir = root / Path(run_value)
    else:
        run_dir = root / Path(run_setting)
    run_dir, result = _validated_completed_run(root, run_dir)
    scenarios = dict(result.get("scenarios") or {})
    arms = dict(dict(result.get("settings") or {}).get("arms") or {})
    missing = [arm_id for arm_id in required_arm_ids if arm_id not in scenarios or arm_id not in arms]
    if missing:
        raise ValueError(
            f"strategy_compare profile={profile_id} run缺少arms: {','.join(missing)}"
        )
    return run_dir, result

def resolve_strategy_compare_run(root: Path, definition: AuditDefinition) -> tuple[Path, dict[str, Any]]:
    if str(definition.source.get("kind") or "") != "strategy_compare":
        raise ValueError(f"{definition.audit_id}.source.kind必須是strategy_compare")
    return resolve_strategy_compare_run_selector(
        root,
        dict(definition.source),
        audit_id=definition.audit_id,
    )


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
        "resource-aware-continuous-capital-preserving",
        "resource-aware-continuous-max-dl",
        "resource-aware-continuous-max-dl-feasible-ascent",
        "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard",
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
    "resolve_strategy_compare_run_selector",
    "resolve_strategy_compare_profile_run_selector",
]
