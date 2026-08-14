"""Shared read-only primitives for orderable/selector-stage alignment audits."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AuditDefinition
from config.strategy_compare import get_strategy_comparison_settings
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from tools.audit.sources.strategy_compare import (
    resolve_arm_artifacts,
    resolve_strategy_compare_profile_run_selector,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少Audit來源CSV: {path.name}")
    return pd.read_csv(path, low_memory=False)


def json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    return value


def _phase_definitions(definition: AuditDefinition) -> dict[str, dict[str, Any]]:
    source = dict(definition.source)
    if str(source.get("kind") or "") != "strategy_compare_cross_phase":
        raise ValueError(f"{definition.audit_id}.source.kind必須是strategy_compare_cross_phase")
    phases = dict(source.get("phases") or {})
    if set(phases) != {"selection_pit", "forward_oos"}:
        raise ValueError(f"{definition.audit_id}.source.phases必須恰為selection_pit/forward_oos")
    for phase_id, phase in phases.items():
        profile_id = str(phase.get("profile_id") or "").strip()
        run = str(phase.get("run") or "").strip()
        baseline_arm = str(phase.get("baseline_arm_id") or "").strip()
        candidate_arms = tuple(str(v).strip() for v in phase.get("candidate_arm_ids") or ())
        reference_arm = str(phase.get("reference_daily_arm_id") or "").strip()
        if profile_id != phase_id or run != "latest":
            raise ValueError(f"{definition.audit_id}.{phase_id}目前固定profile_id={phase_id}, run=latest")
        if not baseline_arm or len(candidate_arms) < 2 or len(set(candidate_arms)) != len(candidate_arms):
            raise ValueError(f"{definition.audit_id}.{phase_id} arm設定不合法")
        if reference_arm not in candidate_arms:
            raise ValueError(f"{definition.audit_id}.{phase_id} reference_daily_arm_id必須在candidate_arm_ids")
    return phases


def resolve_phase_source(root: Path, definition: AuditDefinition, phase_id: str) -> dict[str, Any]:
    phases = _phase_definitions(definition)
    phase = dict(phases[phase_id])
    required = (str(phase["baseline_arm_id"]), *tuple(phase["candidate_arm_ids"]))
    run_dir, result = resolve_strategy_compare_profile_run_selector(
        root,
        phase,
        audit_id=definition.audit_id,
        required_arm_ids=tuple(required),
    )
    baseline = resolve_arm_artifacts(
        run_dir=run_dir,
        result=result,
        arm_id=str(phase["baseline_arm_id"]),
        preferred_pair_arm_id=str(phase["candidate_arm_ids"][0]),
    )
    candidates = {
        arm_id: resolve_arm_artifacts(run_dir=run_dir, result=result, arm_id=arm_id)
        for arm_id in tuple(phase["candidate_arm_ids"])
    }
    settings = get_strategy_comparison_settings(phase_id)
    reference_arm_id = str(phase["reference_daily_arm_id"])
    reference_arm = settings.arms[reference_arm_id]
    if not reference_arm.dl_id:
        raise ValueError(f"{reference_arm_id}缺少DL source")
    reference_dl = settings.dl_sources[reference_arm.dl_id]
    return {
        "phase_id": phase_id,
        "run_dir": run_dir,
        "result": result,
        "baseline": baseline,
        "candidates": candidates,
        "reference_arm_id": reference_arm_id,
        "target_filter_id": reference_dl.filter_id,
        "target_architecture": reference_dl.model_architecture,
        "target_profile": reference_dl.experiment_profile,
    }


def load_common_daily_target(*, root: Path, filter_id: str, architecture: str, profile: str) -> tuple[pd.DataFrame, np.ndarray]:
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=architecture,
        experiment_profile=profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table[["ticker", "date", "group_index"]].copy()
    groups["ticker"] = groups["ticker"].fillna("").astype(str).str.strip()
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d")
    groups["common_target_raw_r"] = np.asarray(bundle.raw_target, dtype=np.float64)
    target_valid = np.asarray(bundle.target_valid, dtype=bool)
    groups.loc[~target_valid, "common_target_raw_r"] = np.nan
    if groups.duplicated(["ticker", "date"]).any():
        raise ValueError("common daily target同ticker/date不唯一")
    calendar = np.array(sorted(groups["date"].unique()), dtype="U10")
    return groups[["ticker", "date", "common_target_raw_r"]].set_index(["ticker", "date"]), calendar


def information_date_map(trade_dates: pd.Series, calendar: np.ndarray) -> pd.Series:
    values = pd.to_datetime(trade_dates, errors="raise").dt.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    positions = np.searchsorted(calendar, values, side="left") - 1
    mapped = np.full(len(values), "", dtype="U10")
    valid = positions >= 0
    mapped[valid] = calendar[positions[valid]]
    return pd.Series(mapped, index=trade_dates.index, dtype="object")


def annotate_orderable(orderable: pd.DataFrame, *, target_lookup: pd.DataFrame, target_calendar: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame(orderable).copy()
    required = {"ticker", "trade_date", "signal_date", "breakout_quality_score"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"orderable candidates缺少欄位: {missing}")
    rows["ticker"] = rows["ticker"].fillna("").astype(str).str.strip()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    rows["model_score"] = pd.to_numeric(rows["breakout_quality_score"], errors="coerce")
    rows["common_information_date"] = information_date_map(rows["trade_date"], target_calendar)
    lookup = target_lookup.reset_index().rename(columns={"date": "common_information_date"})
    rows = rows.merge(
        lookup,
        on=["ticker", "common_information_date"],
        how="left",
        validate="many_to_one",
    )
    rows["target_available"] = pd.to_numeric(rows["common_target_raw_r"], errors="coerce").map(math.isfinite)
    rows["score_available"] = pd.to_numeric(rows["model_score"], errors="coerce").map(math.isfinite)
    return rows
