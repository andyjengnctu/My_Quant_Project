"""Read-only canonical Strategy Compare source resolver shared by formal Audits."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.strategy_compare import get_strategy_comparison_settings
from core.path_utils import project_relative_display_path
from core.strategy_comparison import strategy_comparison_fingerprint


class AuditSourceBlockedError(RuntimeError):
    """Raised when canonical read-only Strategy Compare evidence is unavailable."""


@dataclass(frozen=True)
class StrategyCompareAuditSource:
    profile_id: str
    settings: Any
    result_dir: Path
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
    return StrategyCompareAuditSource(
        profile_id=profile_key,
        settings=settings,
        result_dir=result_dir,
        result=result,
        manifest=manifest_raw,
        config_fingerprint=fingerprint,
    )


__all__ = [
    "AuditSourceBlockedError",
    "StrategyCompareAuditSource",
    "load_strategy_compare_source",
    "resolve_latest_strategy_result_dir",
    "validate_strategy_result_identity",
]
