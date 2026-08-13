"""Read-only resolver for canonical multi-seed robustness artifacts.

This module owns Audit-side resolution and validation of completed robustness runs
and their verified compact attribution sources.  It deliberately contains no
training, replay, model, or strategy logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.strategy_compare import get_strategy_multi_seed_robustness_settings
from core.console_report import project_relative_display_path
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.strategy_multi_seed_robustness import (
    ATTRIBUTION_SOURCE_DIRNAME,
    ATTRIBUTION_SOURCE_SCHEMA_VERSION,
    LATEST_FILENAME,
    MANIFEST_FILENAME,
    SEED_RESULTS_FILENAME,
    SEED_YEARLY_RESULTS_FILENAME,
)


@dataclass(frozen=True)
class CompactRobustnessArmArtifacts:
    arm_id: str
    summary: dict[str, Any]
    trades_path: Path
    equity_path: Path
    capacity_path: Path
    selected_path: Path
    execution_path: Path


def read_json_object(path: Path, *, project_root: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        display = path.relative_to(root).as_posix()
    except ValueError:
        display = path.name
    if not path.is_file():
        raise FileNotFoundError(f"缺少JSON工件: {display}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {display}")
    return payload


def resolve_multi_seed_robustness_run(
    project_root: Path,
    robustness_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    cfg = get_strategy_multi_seed_robustness_settings(str(robustness_id))
    latest_path = (root / cfg.output_root / LATEST_FILENAME).resolve()
    latest = read_json_object(latest_path, project_root=root)
    summary_rel = str(latest.get("summary_path") or "").strip()
    if not summary_rel:
        raise ValueError("robustness latest缺少summary_path")
    summary_path = (root / summary_rel).resolve()
    try:
        summary_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("robustness summary必須位於專案root內") from exc
    summary = read_json_object(summary_path, project_root=root)
    run_root = summary_path.parent
    manifest = read_json_object(run_root / MANIFEST_FILENAME, project_root=root)
    if str(manifest.get("status") or "") != "COMPLETED":
        raise ValueError("robustness run尚未完成")
    contract = dict(manifest.get("contract") or summary.get("contract") or {})
    if str(contract.get("robustness_id") or "") != str(robustness_id):
        raise ValueError("robustness latest identity與Audit config不一致")
    return run_root, summary, contract


def load_multi_seed_results(run_root: Path) -> pd.DataFrame:
    path = Path(run_root) / SEED_RESULTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError("robustness缺少seed_results.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "arm_id", "name", "seed", "seed_order", "arm_order",
        "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
        "expected_value_r", "direct_selection_r",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("robustness seed_results缺少欄位: " + ", ".join(missing))
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["seed_order"] = pd.to_numeric(frame["seed_order"], errors="raise").astype(int)
    if frame.duplicated(["arm_id", "seed"], keep=False).any():
        raise ValueError("robustness seed_results存在重複arm/seed observation")
    if frame.duplicated(["arm_id", "seed_order"], keep=False).any():
        raise ValueError("robustness seed_results存在重複arm/seed_order observation")
    return frame


def load_multi_seed_yearly_results(run_root: Path) -> pd.DataFrame:
    path = Path(run_root) / SEED_YEARLY_RESULTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError("robustness缺少seed_yearly_returns.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "arm_id", "name", "seed", "seed_order", "year",
        "return_pct", "is_complete_year",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("robustness seed_yearly_returns缺少欄位: " + ", ".join(missing))
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["seed_order"] = pd.to_numeric(frame["seed_order"], errors="raise").astype(int)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if frame.duplicated(["arm_id", "seed", "year"], keep=False).any():
        raise ValueError("robustness seed_yearly_returns存在重複arm/seed/year observation")
    return frame


def resolve_verified_compact_unit(
    project_root: Path,
    run_root: Path,
    *,
    arm_id: str,
    seed: int,
    scientific_fingerprint: str,
) -> tuple[Path, dict[str, Any]]:
    root = Path(project_root).resolve()
    unit_dir = Path(run_root).resolve() / ATTRIBUTION_SOURCE_DIRNAME / f"{arm_id}__seed_{int(seed)}"
    payload = read_json_object(unit_dir / MANIFEST_FILENAME, project_root=root)
    if int(payload.get("schema_version") or 0) != ATTRIBUTION_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"compact attribution schema不一致: {arm_id}/seed={seed}")
    if str(payload.get("scientific_fingerprint") or "") != str(scientific_fingerprint):
        raise ValueError(f"compact attribution fingerprint不一致: {arm_id}/seed={seed}")
    if str(payload.get("arm_id") or "") != str(arm_id) or int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"compact attribution unit identity不一致: {arm_id}/seed={seed}")
    validation = dict(payload.get("scientific_observation_validation") or {})
    if str(validation.get("status") or "") != "VERIFIED":
        raise ValueError(f"compact attribution尚未通過scientific observation驗證: {arm_id}/seed={seed}")
    files = dict(payload.get("files") or {})
    for role in ("trades", "equity", "daily_capacity", "selected_buys", "execution"):
        item = dict(files.get(role) or {})
        rel = str(item.get("path") or "").strip()
        if not rel:
            raise ValueError(f"compact attribution缺少{role}: {arm_id}/seed={seed}")
        file_path = (root / rel).resolve()
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("compact attribution path必須位於專案root內") from exc
        if not file_path.is_file():
            raise FileNotFoundError(
                "compact attribution缺少檔案: "
                + project_relative_display_path(file_path, project_root=root)
            )
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if expected_sha and compute_file_sha256(file_path).lower() != expected_sha:
            raise ValueError(f"compact attribution SHA256不一致: {arm_id}/seed={seed}/{role}")
    return unit_dir, payload


def compact_artifacts_from_unit(
    project_root: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> CompactRobustnessArmArtifacts:
    root = Path(project_root).resolve()
    files = dict(manifest.get("files") or {})

    def path_for(role: str) -> Path:
        return (root / str(dict(files[role])["path"])).resolve()

    return CompactRobustnessArmArtifacts(
        arm_id=str(manifest["arm_id"]),
        summary=dict(summary),
        trades_path=path_for("trades"),
        equity_path=path_for("equity"),
        capacity_path=path_for("daily_capacity"),
        selected_path=path_for("selected_buys"),
        execution_path=path_for("execution"),
    )


def resolve_two_arm_multi_seed_source(
    project_root: Path,
    *,
    robustness_id: str,
    candidate_arm_id: str,
    comparator_arm_id: str,
    require_all_seeds: bool = True,
    include_yearly: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = str(candidate_arm_id).strip()
    comparator = str(comparator_arm_id).strip()
    if not robustness_id or not candidate or not comparator or candidate == comparator:
        raise ValueError("robustness/candidate/comparator設定不合法")

    run_root, robustness_summary, contract = resolve_multi_seed_robustness_run(
        root, str(robustness_id)
    )
    fingerprint = str(contract.get("fingerprint") or robustness_summary.get("fingerprint") or "")
    if not fingerprint:
        raise ValueError("robustness contract缺少scientific fingerprint")

    seed_frame = load_multi_seed_results(run_root)
    candidate_rows = seed_frame[seed_frame["arm_id"] == candidate].sort_values("seed_order")
    comparator_rows = seed_frame[seed_frame["arm_id"] == comparator].sort_values("seed_order")
    if candidate_rows.empty or comparator_rows.empty:
        raise ValueError("robustness結果缺少Audit candidate/comparator arm")

    candidate_seeds = list(zip(
        candidate_rows["seed_order"].astype(int),
        candidate_rows["seed"].astype(int),
    ))
    comparator_seeds = list(zip(
        comparator_rows["seed_order"].astype(int),
        comparator_rows["seed"].astype(int),
    ))
    if candidate_seeds != comparator_seeds:
        raise ValueError("robustness candidate/comparator resolved seeds不一致")

    expected_count = int(contract.get("seed_count") or 0)
    if expected_count < 2:
        raise ValueError("robustness contract seed_count必須>=2")
    if require_all_seeds and len(candidate_seeds) != expected_count:
        raise ValueError(
            f"Audit要求完整all-seed source: expected={expected_count}, actual={len(candidate_seeds)}"
        )
    resolved_seeds = [int(value) for value in list(contract.get("resolved_seeds") or [])]
    expected_seed_pairs = list(enumerate(resolved_seeds, start=1))
    if (
        len(resolved_seeds) != expected_count
        or (require_all_seeds and candidate_seeds != expected_seed_pairs)
    ):
        raise ValueError("robustness seed_results與scientific contract resolved_seeds不一致")

    units: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for _seed_order, seed in candidate_seeds:
        for arm_id in (candidate, comparator):
            units[(arm_id, int(seed))] = resolve_verified_compact_unit(
                root,
                run_root,
                arm_id=arm_id,
                seed=int(seed),
                scientific_fingerprint=fingerprint,
            )

    return {
        "robustness_id": str(robustness_id),
        "candidate_arm_id": candidate,
        "comparator_arm_id": comparator,
        "run_root": run_root,
        "summary": robustness_summary,
        "contract": contract,
        "fingerprint": fingerprint,
        "seed_frame": seed_frame,
        "seed_yearly_frame": (
            load_multi_seed_yearly_results(run_root)
            if include_yearly
            else pd.DataFrame()
        ),
        "seed_pairs": candidate_seeds,
        "units": units,
    }


__all__ = [
    "CompactRobustnessArmArtifacts",
    "compact_artifacts_from_unit",
    "load_multi_seed_results",
    "load_multi_seed_yearly_results",
    "read_json_object",
    "resolve_multi_seed_robustness_run",
    "resolve_two_arm_multi_seed_source",
    "resolve_verified_compact_unit",
]
