"""Resolved Strategy Compare execution-plan boundary.

The preparation/reuse layers may progressively discover artifact provenance, but UI
rendering and replay execution must consume one finalized immutable plan.  This module
owns that boundary and the READY invariants; it does not own strategy mathematics,
artifact building, or completed-pair discovery.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from core.strategy_comparison import StrategyComparisonSettings, StrategyPreparationPlan


class _FrozenList(tuple):
    """Tagged immutable list so thawing preserves original list/tuple types."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}
    return deepcopy(value)


@dataclass(frozen=True)
class ResolvedContinuousScoreBinding:
    """Verified frozen-score artifact binding independent of profile membership."""

    score_path: Path
    sha256: str
    available_from: str
    available_through: str
    execution_start: str
    provenance_run_dir: Path
    provenance_pair_dir: Path
    path_source: str
    archived_path: str
    current_canonical_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "score_path": self.score_path,
            "sha256": self.sha256,
            "available_from": self.available_from,
            "available_through": self.available_through,
            "execution_start": self.execution_start,
            "provenance_run_dir": self.provenance_run_dir,
            "provenance_pair_dir": self.provenance_pair_dir,
            "path_source": self.path_source,
            "archived_path": self.archived_path,
            "current_canonical_path": self.current_canonical_path,
        }


@dataclass(frozen=True)
class ResolvedComparisonPlan:
    """Immutable, replay-ready view of one Strategy Compare planning result."""

    settings: StrategyComparisonSettings
    project_root: Path
    _status: Mapping[str, Any]

    @classmethod
    def from_status(
        cls,
        *,
        settings: StrategyComparisonSettings,
        project_root: Path,
        status: Mapping[str, Any],
    ) -> "ResolvedComparisonPlan":
        instance = cls(
            settings=settings,
            project_root=Path(project_root).resolve(),
            _status=_freeze(dict(status)),
        )
        instance.validate_contract()
        return instance

    @property
    def overall_status(self) -> str:
        return str(self._status.get("overall_status") or "")

    @property
    def config_fingerprint(self) -> str:
        return str(self._status.get("config_fingerprint") or "")

    @property
    def comparison_period(self) -> Mapping[str, Any]:
        raw = self._status.get("comparison_period")
        return raw if isinstance(raw, Mapping) else MappingProxyType({})

    @property
    def preparation_plan(self) -> StrategyPreparationPlan:
        plan = self._status.get("preparation_plan")
        if not isinstance(plan, StrategyPreparationPlan):
            raise RuntimeError("ResolvedComparisonPlan缺少StrategyPreparationPlan")
        return plan

    def status_dict(self) -> dict[str, Any]:
        """Return a detached mutable copy for legacy render/report call sites."""

        return _thaw(self._status)

    def validate_contract(self) -> None:
        """Enforce invariants that must hold before status can claim READY."""

        plan = self._status.get("preparation_plan")
        if not isinstance(plan, StrategyPreparationPlan):
            raise RuntimeError("Strategy Compare resolved plan缺少preparation_plan")
        overall_status = str(self._status.get("overall_status") or "")
        if overall_status != plan.overall_status:
            raise RuntimeError(
                "Strategy Compare resolved plan狀態分叉: "
                f"status={overall_status!r}, preparation={plan.overall_status!r}"
            )
        expected_ready = overall_status == "READY"
        if bool(self._status.get("comparison_ready")) != expected_ready:
            raise RuntimeError("Strategy Compare comparison_ready與overall_status不一致")
        if not str(self._status.get("config_fingerprint") or "").strip():
            raise RuntimeError("Strategy Compare resolved plan缺少config fingerprint")
        if overall_status != "READY":
            return

        period = self._status.get("comparison_period")
        if not isinstance(period, Mapping):
            raise RuntimeError("Strategy Compare READY但缺少comparison period")
        start = str(period.get("start") or "").strip()
        end = str(period.get("end") or "").strip()
        if not start or not end or start > end:
            raise RuntimeError("Strategy Compare READY但comparison period無效")

        resolved_parameter_paths = self._status.get("resolved_parameter_paths")
        if not isinstance(resolved_parameter_paths, Mapping):
            raise RuntimeError("Strategy Compare READY但缺少resolved parameter paths")
        resolved_arm_parameter_paths = self._status.get("resolved_arm_parameter_paths")
        if not isinstance(resolved_arm_parameter_paths, Mapping):
            raise RuntimeError("Strategy Compare READY但缺少resolved arm parameter paths")
        missing_params = sorted(
            arm.arm_id for arm in self.settings.enabled_arms
            if arm.arm_id not in resolved_arm_parameter_paths
        )
        if missing_params:
            raise RuntimeError(
                "Strategy Compare READY但缺少arm參數binding: " + ",".join(missing_params)
            )

        dl_rows = self._status.get("dl_sources")
        if not isinstance(dl_rows, Mapping):
            dl_rows = {}
        overrides = self._status.get("continuous_score_overrides")
        if not isinstance(overrides, Mapping):
            overrides = {}
        replay_cache = self._status.get("replay_cache")
        if not isinstance(replay_cache, Mapping):
            raise RuntimeError("Strategy Compare READY但缺少replay cache resolution")
        cached_pairs = replay_cache.get("pairs")
        if not isinstance(cached_pairs, Mapping):
            cached_pairs = {}

        unresolved_dl: list[str] = []
        for arm in self.settings.enabled_arms:
            if not arm.dl_enabled:
                continue
            dl_id = str(arm.dl_id or "").strip()
            source_row = dl_rows.get(dl_id)
            source_ready = isinstance(source_row, Mapping) and bool(source_row.get("ready"))
            override_ready = isinstance(overrides.get(dl_id), Mapping)
            pair_ready = isinstance(cached_pairs.get(arm.arm_id), Mapping)
            if not (source_ready or override_ready or pair_ready):
                unresolved_dl.append(f"{arm.arm_id}:{dl_id}")
        if unresolved_dl:
            raise RuntimeError(
                "Strategy Compare READY但DL artifact binding未解析: "
                + ",".join(sorted(unresolved_dl))
            )


__all__ = ["ResolvedComparisonPlan", "ResolvedContinuousScoreBinding"]
