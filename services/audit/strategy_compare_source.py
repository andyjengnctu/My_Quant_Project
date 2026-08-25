"""Read-only canonical Strategy Compare source resolver shared by formal Audits."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from config.strategy_compare import get_strategy_comparison_settings
from core.path_utils import project_relative_display_path
from core.strategy_comparison import strategy_comparison_fingerprint
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_SCORE_RANKING,
)
from filters.breakout_quality.strategy_compare_runtime import (
    strategy_comparison_execution_pairs,
)


class AuditSourceBlockedError(RuntimeError):
    """Raised when canonical read-only Strategy Compare evidence is unavailable."""


@dataclass(frozen=True)
class StrategyCompareAuditSource:
    profile_id: str
    settings: Any
    result_dir: Path
    run_dir: Path
    result: Mapping[str, Any]
    manifest: Mapping[str, Any] | None
    config_fingerprint: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditSourceBlockedError(
            f"無法讀取正式JSON工件: {path.name} | {exc}"
        ) from exc


def resolve_latest_strategy_result_dir(project_root: Path, output_root_text: str) -> Path:
    root = Path(project_root).resolve()
    output_root = root / str(output_root_text)
    latest = output_root / "latest"
    if latest.is_dir() or latest.is_symlink():
        return latest.resolve()
    for pointer_name in ("latest.json", "latest.txt"):
        pointer = output_root / pointer_name
        if not pointer.exists():
            continue
        try:
            if pointer.suffix == ".json":
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                candidate_text = (
                    payload.get("run_dir")
                    or payload.get("path")
                    or payload.get("latest")
                    if isinstance(payload, Mapping)
                    else None
                )
            else:
                candidate_text = pointer.read_text(encoding="utf-8").strip()
        except (OSError, json.JSONDecodeError):
            candidate_text = None
        if candidate_text:
            candidate = Path(str(candidate_text))
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_dir():
                return candidate.resolve()
    runs_dir = output_root / "runs"
    runs = sorted(
        (path for path in runs_dir.glob("*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if runs:
        return runs[-1].resolve()
    raise AuditSourceBlockedError(
        "缺少Strategy Compare latest/runs結果: "
        f"{project_relative_display_path(output_root, project_root=root)}"
    )


def _collect_named_mappings(payload: Any, key_name: str) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) == key_name and isinstance(value, Mapping):
                found.append(value)
            found.extend(_collect_named_mappings(value, key_name))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_collect_named_mappings(value, key_name))
    return found


def _primary_strategy_fingerprints(payload: Any) -> set[str]:
    if not isinstance(payload, Mapping):
        return set()
    containers: list[Mapping[str, Any]] = [payload]
    for key in ("metadata", "run", "summary", "strategy_comparison"):
        child = payload.get(key)
        if isinstance(child, Mapping):
            containers.append(child)
    found: set[str] = set()
    for container in containers:
        for key in ("config_fingerprint", "strategy_comparison_fingerprint"):
            value = container.get(key)
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if len(text) == 12:
                    found.add(text)
    return found


def validate_strategy_result_identity(
    *,
    settings: Any,
    payloads: Sequence[Any],
    result_dir: Path,
    project_root: Path,
) -> str:
    root = Path(project_root).resolve()
    stored_fingerprints: set[str] = set()
    for payload in payloads:
        stored_fingerprints.update(_primary_strategy_fingerprints(payload))
    if not stored_fingerprints:
        raise AuditSourceBlockedError(
            "Strategy Compare latest缺少canonical config fingerprint，無法驗證artifact reuse identity: "
            f"{project_relative_display_path(result_dir, project_root=root)}"
        )
    if len(stored_fingerprints) != 1:
        raise AuditSourceBlockedError(
            f"Strategy Compare latest fingerprint互相衝突: {sorted(stored_fingerprints)}"
        )
    stored = next(iter(stored_fingerprints))

    expected_fingerprints = {strategy_comparison_fingerprint(settings)}
    artifact_identity_payloads: list[Mapping[str, Any]] = []
    for payload in payloads:
        artifact_identity_payloads.extend(
            _collect_named_mappings(payload, "artifact_identities")
        )
        artifact_identity_payloads.extend(
            _collect_named_mappings(payload, "resolved_artifact_identities")
        )
    for identities in artifact_identity_payloads:
        expected_fingerprints.add(
            strategy_comparison_fingerprint(settings, artifact_identities=identities)
        )
    if stored not in expected_fingerprints:
        raise AuditSourceBlockedError(
            "Strategy Compare latest與目前config canonical identity不相容；"
            f"stored={stored} expected={sorted(expected_fingerprints)}。"
            "Audit不得把stale result當成目前策略證據"
        )
    return stored


def _resolve_canonical_run_dir(
    *,
    project_root: Path,
    result_dir: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    config_fingerprint: str,
) -> Path:
    root = Path(project_root).resolve()
    run_dir_text = str((manifest or {}).get("run_dir") or "").strip()
    if not run_dir_text:
        if result_dir.name != "latest":
            return result_dir.resolve()
        raise AuditSourceBlockedError(
            "Strategy Compare latest manifest缺少canonical run_dir；"
            "無法安全解析pair row evidence"
        )
    run_dir = Path(run_dir_text)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise AuditSourceBlockedError(
            "Strategy Compare canonical run_dir不存在: "
            f"{project_relative_display_path(run_dir, project_root=root)}"
        )
    run_result_path = run_dir / "strategy_comparison.json"
    if not run_result_path.is_file():
        raise AuditSourceBlockedError(
            "Strategy Compare canonical run缺少strategy_comparison.json: "
            f"{project_relative_display_path(run_dir, project_root=root)}"
        )
    run_result = _read_json(run_result_path)
    if not isinstance(run_result, Mapping):
        raise AuditSourceBlockedError("Strategy Compare canonical run JSON格式無效")
    run_fingerprints = _primary_strategy_fingerprints(run_result)
    if run_fingerprints != {str(config_fingerprint)}:
        raise AuditSourceBlockedError(
            "Strategy Compare latest與canonical run fingerprint不一致；"
            f"latest={config_fingerprint}, run={sorted(run_fingerprints)}"
        )
    latest_pair_execution = result.get("pair_execution")
    run_pair_execution = run_result.get("pair_execution")
    if latest_pair_execution != run_pair_execution:
        raise AuditSourceBlockedError(
            "Strategy Compare latest與canonical run的pair_execution不一致"
        )
    return run_dir


def load_strategy_compare_source(
    project_root: Path,
    *,
    profile_id: str,
) -> StrategyCompareAuditSource:
    root = Path(project_root).resolve()
    profile_key = str(profile_id).strip()
    if not profile_key:
        raise ValueError("Strategy Compare profile_id不可空白")
    settings = get_strategy_comparison_settings(profile_key)
    result_dir = resolve_latest_strategy_result_dir(root, settings.output_root)
    result_path = result_dir / "strategy_comparison.json"
    if not result_path.is_file():
        raise AuditSourceBlockedError(
            "Strategy Compare latest缺少strategy_comparison.json: "
            f"{project_relative_display_path(result_dir, project_root=root)}"
        )
    result = _read_json(result_path)
    if not isinstance(result, Mapping):
        raise AuditSourceBlockedError("Strategy Compare strategy_comparison.json格式無效")
    manifest_path = result_dir / "manifest.json"
    manifest_raw = _read_json(manifest_path) if manifest_path.is_file() else None
    if manifest_raw is not None and not isinstance(manifest_raw, Mapping):
        raise AuditSourceBlockedError("Strategy Compare manifest.json格式無效")
    payloads = [result, *( [manifest_raw] if manifest_raw is not None else [] )]
    fingerprint = validate_strategy_result_identity(
        settings=settings,
        payloads=payloads,
        result_dir=result_dir,
        project_root=root,
    )
    run_dir = _resolve_canonical_run_dir(
        project_root=root,
        result_dir=result_dir,
        result=result,
        manifest=manifest_raw,
        config_fingerprint=fingerprint,
    )
    return StrategyCompareAuditSource(
        profile_id=profile_key,
        settings=settings,
        result_dir=result_dir,
        run_dir=run_dir,
        result=result,
        manifest=manifest_raw,
        config_fingerprint=fingerprint,
    )


def load_strategy_arm_replay_sidecars(
    project_root: Path,
    *,
    source: StrategyCompareAuditSource,
    arm_id: str,
) -> dict[str, Any]:
    """Load one arm's persisted Strategy Compare row evidence without replay.

    DL-on arms and standalone DL-off baselines have a direct ``pair_execution``
    entry.  A DL-off baseline that owns a shared execution group (for example C58)
    is instead embedded in each paired DL-on artifact and therefore may have no
    direct entry.  Resolve that topology through the canonical Strategy Compare
    execution-group SSOT, never by guessing pair-directory names.
    """

    root = Path(project_root).resolve()
    arm_key = str(arm_id).strip()
    pair_execution = source.result.get("pair_execution")
    if not isinstance(pair_execution, Mapping):
        raise AuditSourceBlockedError(
            f"{source.profile_id} Strategy Compare缺少pair_execution contract"
        )

    arm = next(
        (item for item in source.settings.enabled_arms if item.arm_id == arm_key),
        None,
    )
    if arm is None:
        raise AuditSourceBlockedError(
            f"{source.profile_id}/{arm_key} 不屬於目前Strategy Compare enabled arms"
        )

    execution = pair_execution.get(arm_key)
    execution_source_arm_id = arm_key
    paired_baseline = False
    if not isinstance(execution, Mapping):
        if arm.dl_enabled:
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{arm_key} Strategy Compare缺少pair_execution evidence"
            )
        paired_anchor: tuple[str, Mapping[str, Any] | None] | None = None
        for _param_source, _rule_policy, off_arm, on_arm in (
            strategy_comparison_execution_pairs(source.settings)
        ):
            if off_arm.arm_id != arm_key:
                continue
            candidate_execution = pair_execution.get(on_arm.arm_id)
            paired_anchor = (
                on_arm.arm_id,
                candidate_execution if isinstance(candidate_execution, Mapping) else None,
            )
            break
        if paired_anchor is None:
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{arm_key} 不屬於任何canonical Strategy Compare execution group"
            )
        execution_source_arm_id, execution = paired_anchor
        if not isinstance(execution, Mapping):
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{arm_key} canonical shared-baseline anchor "
                f"{execution_source_arm_id} 缺少pair_execution evidence"
            )
        paired_baseline = True

    def _resolve_pair_dir(
        mapping: Mapping[str, Any], *, mapping_arm_id: str
    ) -> Path:
        pair_dir_text = str(mapping.get("current_pair_dir") or "").strip()
        if not pair_dir_text:
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{mapping_arm_id} pair_execution缺少current_pair_dir"
            )
        candidate_dir = Path(pair_dir_text)
        if not candidate_dir.is_absolute():
            candidate_dir = root / candidate_dir
        candidate_dir = candidate_dir.resolve()
        try:
            candidate_dir.relative_to(source.run_dir.resolve())
        except ValueError as exc:
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{mapping_arm_id} current_pair_dir不屬於manifest指定的canonical run: "
                f"{project_relative_display_path(candidate_dir, project_root=root)}"
            ) from exc
        return candidate_dir

    pair_dir = _resolve_pair_dir(execution, mapping_arm_id=execution_source_arm_id)

    pair_result_path = pair_dir / "strategy_comparison.json"
    if not pair_result_path.is_file():
        raise AuditSourceBlockedError(
            f"{source.profile_id}/{arm_key} pair缺少strategy_comparison.json: "
            f"{project_relative_display_path(pair_dir, project_root=root)}"
        )
    pair_payload = _read_json(pair_result_path)
    if not isinstance(pair_payload, Mapping):
        raise AuditSourceBlockedError(
            f"{source.profile_id}/{arm_key} pair strategy_comparison.json格式無效"
        )
    metadata = pair_payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if not arm.dl_enabled:
        prefix = "no_filter"
    else:
        comparison_mode = str(metadata.get("comparison_mode") or "").strip()
        if comparison_mode != COMPARISON_MODE_SCORE_RANKING:
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{arm_key} Audit目前只接受"
                f"{COMPARISON_MODE_SCORE_RANKING} row evidence；"
                f"comparison_mode={comparison_mode!r}"
            )
        prefix = "score_ranking"
    orderable_path = pair_dir / f"{prefix}_orderable_candidates.csv"
    selected_path = pair_dir / f"{prefix}_selected_buys.csv"
    missing = [path.name for path in (orderable_path, selected_path) if not path.is_file()]
    if missing:
        raise AuditSourceBlockedError(
            f"{source.profile_id}/{arm_key} pair缺少既有row sidecar: {missing}; "
            "Audit不得重跑策略補資料"
        )
    try:
        orderable = pd.read_csv(orderable_path, encoding="utf-8-sig", low_memory=False)
        selected = pd.read_csv(selected_path, encoding="utf-8-sig", low_memory=False)
    except (OSError, ValueError, UnicodeError) as exc:
        raise AuditSourceBlockedError(
            f"{source.profile_id}/{arm_key} 無法讀取Strategy Compare row sidecar: {exc}"
        ) from exc
    return {
        "arm_id": arm_key,
        "pair_dir": pair_dir,
        "prefix": prefix,
        "pair_execution_source_arm_id": execution_source_arm_id,
        "paired_baseline_resolved": paired_baseline,
        "orderable": orderable,
        "selected": selected,
        "orderable_path": orderable_path,
        "selected_path": selected_path,
    }


def load_strategy_arm_pipeline_sidecars(
    project_root: Path,
    *,
    source: StrategyCompareAuditSource,
    arm_id: str,
) -> dict[str, Any]:
    """Load persisted score-ranking selector pipeline evidence for one DL-on arm.

    This is read-only and intentionally refuses to replay or reconstruct missing
    reservation diagnostics.  The pair directory is resolved through the same
    canonical arm→execution mapping as ``load_strategy_arm_replay_sidecars``.
    """

    evidence = load_strategy_arm_replay_sidecars(
        project_root, source=source, arm_id=arm_id
    )
    if evidence.get("prefix") != "score_ranking":
        raise AuditSourceBlockedError(
            f"{source.profile_id}/{arm_id} pipeline Audit只接受DL-on score-ranking arm"
        )
    pair_dir = Path(evidence["pair_dir"])
    required_names = {
        "daily_capacity": "score_ranking_daily_capacity.csv",
        "selector_trace": "score_ranking_selector_trace.csv",
        "execution": "score_ranking_execution.csv",
    }
    loaded: dict[str, Any] = dict(evidence)
    for key, filename in required_names.items():
        path = pair_dir / filename
        if not path.is_file():
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{arm_id} pair缺少既有pipeline sidecar: {filename}; "
                "Audit不得重跑策略補資料"
            )
        try:
            loaded[key] = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except (OSError, ValueError, UnicodeError) as exc:
            raise AuditSourceBlockedError(
                f"{source.profile_id}/{arm_id} 無法讀取{filename}: {exc}"
            ) from exc
        loaded[f"{key}_path"] = path
    return loaded


__all__ = [
    "AuditSourceBlockedError",
    "StrategyCompareAuditSource",
    "load_strategy_arm_pipeline_sidecars",
    "load_strategy_arm_replay_sidecars",
    "load_strategy_compare_source",
    "resolve_latest_strategy_result_dir",
    "validate_strategy_result_identity",
]
