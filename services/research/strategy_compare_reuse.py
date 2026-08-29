"""Completed-pair cache, archived provenance, and frozen-score reuse owner."""

from __future__ import annotations

from core.file_integrity import load_json_object_or_none as _read_json

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.console_report import project_relative_display_path
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    StrategyComparisonArm,
    StrategyComparisonSettings,
    StrategyPreparationAction,
    StrategyPreparationPlan,
    resolve_strategy_comparison_arm_param_policy,
    strategy_comparison_param_artifact_key,
    strategy_comparison_fingerprint,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_score_table_from_path,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_SCORE_RANKING,
    STRATEGY_COMPARE_SCHEMA_VERSION as STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
)
from filters.breakout_quality.strategy_compare_plan import ResolvedContinuousScoreBinding
from filters.breakout_quality.strategy_score_projection import (
    primary_score_column_for_source,
    runtime_score_projection_requirements,
)
from filters.breakout_quality.strategy_result_state import resolve_strategy_result_state
from services.research.strategy_compare_preparation import resolve_comparison_period
from filters.breakout_quality.strategy_compare_runtime import (
    _arm_runtime_spec,
    _execution_pairs,
    _standalone_baseline_arms,
)
from filters.breakout_quality.strategy_compare_sources import (
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    resolve_project_relative_path as _resolve_relative_path,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES

PAIR_CACHE_LEGACY_SCHEMA_VERSION = 1
PAIR_CACHE_SCHEMA_VERSION = 2

def _pair_group_id(
    *,
    param_source: str,
    rule_policy: str,
    dl_id: str,
    dl_runtime_mode: str,
    arm_id: str | None = None,
) -> str:
    base = (
        f"{param_source}__{rule_policy}__{dl_id}__"
        f"{str(dl_runtime_mode).replace('-', '_')}"
    )
    canonical_arm_id = str(arm_id or "").strip()
    return base if not canonical_arm_id else f"{base}__{canonical_arm_id}"


def _pair_group_id_candidates(
    *,
    param_source: str,
    rule_policy: str,
    dl_id: str,
    dl_runtime_mode: str,
    arm_id: str,
) -> tuple[str, ...]:
    current = _pair_group_id(
        param_source=param_source,
        rule_policy=rule_policy,
        dl_id=dl_id,
        dl_runtime_mode=dl_runtime_mode,
        arm_id=arm_id,
    )
    legacy = _pair_group_id(
        param_source=param_source,
        rule_policy=rule_policy,
        dl_id=dl_id,
        dl_runtime_mode=dl_runtime_mode,
    )
    return (current, legacy) if current != legacy else (current,)


def _stored_pair_group_id(
    *,
    pairs: dict[str, Any],
    param_source: str,
    rule_policy: str,
    dl_id: str,
    dl_runtime_mode: str,
    arm_id: str,
) -> str | None:
    for candidate in _pair_group_id_candidates(
        param_source=param_source,
        rule_policy=rule_policy,
        dl_id=dl_id,
        dl_runtime_mode=dl_runtime_mode,
        arm_id=arm_id,
    ):
        if isinstance(pairs.get(candidate), dict):
            return candidate
    return None

def _replay_arm_contract(raw: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "param_source": str(raw.get("param_source") or ""),
        "param_policy": raw.get("param_policy"),
        "rule_policy": str(raw.get("rule_policy") or ""),
        "dl_enabled": bool(raw.get("dl_enabled")),
        "dl_id": raw.get("dl_id"),
        "dl_runtime_mode": raw.get("dl_runtime_mode"),
    }
    runtime_options = dict(raw.get("dl_runtime_options") or {})
    if runtime_options:
        # 歷史arm沒有runtime options時維持既有pair fingerprint；
        # 只有真正新增的runtime option才進入cache identity。
        contract["dl_runtime_options"] = runtime_options
    return contract

def _selection_pit_projection_artifacts(
    *,
    settings_payload: dict[str, Any],
    artifact_identities: dict[str, Any],
    on_arm_payload: dict[str, Any],
) -> dict[str, Any] | None:
    requirements = runtime_score_projection_requirements(
        settings_payload=settings_payload,
        on_arm_payload=on_arm_payload,
    )
    if not requirements:
        return None
    dl_sources = dict(settings_payload.get("dl_sources") or {})
    projected: dict[str, Any] = {}
    for source_id, columns in requirements.items():
        source = dict(dl_sources.get(source_id) or {})
        if str(source.get("score_source") or "") != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            return None
        identity = dict(artifact_identities.get(f"dl:{source_id}:forward_scores") or {})
        projection_map = dict(identity.get("score_projection_sha256") or {})
        if not all(str(projection_map.get(column) or "").strip() for column in columns):
            return None
        projected[f"dl:{source_id}:forward_scores_projection"] = {
            "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            "columns": {
                column: str(projection_map[column]).strip().lower()
                for column in columns
            },
        }
    return projected


def _selection_pit_runtime_source_ids(
    *, settings_payload: dict[str, Any], on_arm_payload: dict[str, Any]
) -> set[str]:
    dl_sources = dict(settings_payload.get("dl_sources") or {})
    return {
        source_id
        for source_id in runtime_score_projection_requirements(
            settings_payload=settings_payload, on_arm_payload=on_arm_payload
        )
        if str(dict(dl_sources.get(source_id) or {}).get("score_source") or "")
        == SCORE_SOURCE_SELECTION_POINT_IN_TIME
    }


def _pair_cache_fingerprint_from_payload(
    *,
    settings_payload: dict[str, Any],
    artifact_identities: dict[str, Any],
    comparison_period: dict[str, Any],
    off_arm_payload: dict[str, Any],
    on_arm_payload: dict[str, Any],
    engine_schema_version: int,
    parameter_evaluation_sha256: str | None = None,
) -> str:
    param_source = str(on_arm_payload.get("param_source") or "")
    dl_id = str(on_arm_payload.get("dl_id") or "")
    parameter_sources = dict(settings_payload.get("parameter_sources") or {})
    dl_sources = dict(settings_payload.get("dl_sources") or {})
    param_payload = dict(parameter_sources.get(param_source) or {})
    dl_payload = dict(dl_sources.get(dl_id) or {})

    arm_param_policy = str(
        on_arm_payload.get("param_policy") or settings_payload.get("param_policy") or ""
    ).strip()
    param_artifact_key = strategy_comparison_param_artifact_key(param_source, arm_param_policy)
    artifact_keys: list[str] = []
    trained_with = str(param_payload.get("trained_with_dl_id") or "")
    if trained_with:
        artifact_keys.extend(
            f"dl:{trained_with}:{name}"
            for name in ("model", "manifest", "forward_scores")
        )
    if str(on_arm_payload.get("dl_runtime_mode") or "") == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT:
        arm_id = str(on_arm_payload.get("arm_id") or "")
        if arm_id:
            artifact_keys.append(f"runtime:{arm_id}:expected_r_calibration")
    if str(on_arm_payload.get("dl_runtime_mode") or "") in {
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    }:
        arm_id = str(on_arm_payload.get("arm_id") or "")
        if arm_id:
            artifact_keys.append(f"runtime:{arm_id}:expected_excess_r_calibration")
    if dl_id:
        dl_artifact_names = (
            ("manifest", "audit", "forward_scores")
            if str(dl_payload.get("score_source") or "") == "selection_point_in_time"
            else ("model", "manifest", "forward_scores")
        )
        artifact_keys.extend(
            f"dl:{dl_id}:{name}" for name in dl_artifact_names
        )
    safety_dl_id = ""
    safety_dl_payload = {}
    if str(on_arm_payload.get("dl_runtime_mode") or "") in {
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    }:
        safety_dl_id = str(
            dict(on_arm_payload.get("dl_runtime_options") or {}).get("safety_dl_id") or ""
        ).strip()
        safety_dl_payload = dict(dl_sources.get(safety_dl_id) or {})
        if safety_dl_id:
            safety_artifact_names = (
                ("manifest", "audit", "forward_scores")
                if str(safety_dl_payload.get("score_source") or "") == "selection_point_in_time"
                else ("model", "manifest", "forward_scores")
            )
            artifact_keys.extend(
                f"dl:{safety_dl_id}:{name}" for name in safety_artifact_names
            )
    projected_score_artifacts = _selection_pit_projection_artifacts(
        settings_payload=settings_payload,
        artifact_identities=artifact_identities,
        on_arm_payload=on_arm_payload,
    )
    if projected_score_artifacts is None:
        cache_schema_version = PAIR_CACHE_LEGACY_SCHEMA_VERSION
        selected_artifacts = {
            key: artifact_identities.get(key)
            for key in sorted(set(artifact_keys))
        }
    else:
        cache_schema_version = PAIR_CACHE_SCHEMA_VERSION
        runtime_selection_pit_ids = _selection_pit_runtime_source_ids(
            settings_payload=settings_payload, on_arm_payload=on_arm_payload
        )
        selected_artifacts = {
            key: artifact_identities.get(key)
            for key in sorted(set(artifact_keys))
            if not any(
                key in {
                    f"dl:{source_id}:manifest",
                    f"dl:{source_id}:audit",
                    f"dl:{source_id}:forward_scores",
                }
                for source_id in runtime_selection_pit_ids
            )
        }
        selected_artifacts.update(projected_score_artifacts)
    evaluation_param_sha = str(parameter_evaluation_sha256 or "").strip().lower()
    if not evaluation_param_sha:
        evaluation_param_sha = _artifact_identity_sha(artifact_identities.get(param_artifact_key))

    payload = {
        "cache_schema_version": cache_schema_version,
        "engine_schema_version": int(engine_schema_version),
        "dataset": settings_payload.get("dataset"),
        "comparison_period": dict(comparison_period or {}),
        "param_policy": arm_param_policy,
        "max_positions": settings_payload.get("max_positions"),
        "rotation": settings_payload.get("rotation"),
        "parameter_source": {
            "source_id": param_source,
            "path_template": param_payload.get("path_template"),
            "identity_manifest_path": param_payload.get("identity_manifest_path"),
            "trained_with_dl_id": param_payload.get("trained_with_dl_id"),
            # Result reuse is bound to the replay-effective evaluation view.
            # Physical publication lineage remains in preparation artifact identities.
            "evaluation_sha256": evaluation_param_sha,
        },
        "dl_source": {
            "dl_id": dl_id,
            "filter_id": dl_payload.get("filter_id"),
            "model_architecture": dl_payload.get("model_architecture"),
            "experiment_profile": dl_payload.get("experiment_profile"),
            "threshold": dl_payload.get("threshold"),
            "score_source": dl_payload.get("score_source"),
        },
        "safety_dl_source": (
            {
                "dl_id": safety_dl_id,
                "filter_id": safety_dl_payload.get("filter_id"),
                "model_architecture": safety_dl_payload.get("model_architecture"),
                "experiment_profile": safety_dl_payload.get("experiment_profile"),
                "threshold": safety_dl_payload.get("threshold"),
                "score_source": safety_dl_payload.get("score_source"),
            }
            if safety_dl_id else None
        ),
        "off_arm": _replay_arm_contract(off_arm_payload),
        "on_arm": _replay_arm_contract(on_arm_payload),
        "artifact_identities": selected_artifacts,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def _arm_parameter_evaluation_sha256(
    status: dict[str, Any], arm_id: str
) -> str:
    identity = dict(
        (status.get("resolved_arm_parameter_identities") or {}).get(str(arm_id)) or {}
    )
    return _artifact_identity_sha(identity)


def _current_pair_cache_fingerprint(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    off_arm: StrategyComparisonArm,
    on_arm: StrategyComparisonArm,
) -> str:
    return _pair_cache_fingerprint_from_payload(
        settings_payload=settings.as_dict(),
        artifact_identities=dict(status.get("artifact_identities") or {}),
        comparison_period=dict(status.get("comparison_period") or {}),
        off_arm_payload=off_arm.as_dict(),
        on_arm_payload=on_arm.as_dict(),
        engine_schema_version=STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
        parameter_evaluation_sha256=_arm_parameter_evaluation_sha256(
            status, on_arm.arm_id
        ),
    )

def _pair_cache_required_files(
    pair_dir: Path,
    *,
    on_arm: StrategyComparisonArm,
) -> tuple[Path, ...]:
    runtime_spec = _arm_runtime_spec(on_arm)
    required = [
        pair_dir / "strategy_comparison.json",
        pair_dir / "strategy_comparison.md",
        pair_dir / "yearly_returns_comparison.csv",
        pair_dir / "no_filter_equity.csv",
        pair_dir / "no_filter_trades.csv",
        pair_dir / runtime_spec["active_trades_filename"],
    ]
    if runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        required.extend([
            pair_dir / "score_ranking_equity.csv",
            pair_dir / "no_filter_daily_capacity.csv",
            pair_dir / "score_ranking_daily_capacity.csv",
            pair_dir / "no_filter_orderable_candidates.csv",
            pair_dir / "score_ranking_orderable_candidates.csv",
            pair_dir / "no_filter_selected_buys.csv",
            pair_dir / "score_ranking_selected_buys.csv",
            pair_dir / "score_ranking_execution.csv",
            pair_dir / "score_ranking_selector_trace.csv",
            pair_dir / "score_ranking_repair_search_certificate.csv",
        ])
    return tuple(required)

def _artifact_identity_sha(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("sha256") or "").strip().lower()

def _settings_core_matches_archived_pair(
    *,
    settings: StrategyComparisonSettings,
    stored_settings: dict[str, Any],
    on_arm: StrategyComparisonArm,
) -> bool:
    """Validate immutable replay semantics before using an archived comparator pair."""

    for field in ("dataset", "param_policy", "max_positions", "rotation"):
        if stored_settings.get(field) != settings.as_dict().get(field):
            return False

    stored_params = dict(stored_settings.get("parameter_sources") or {})
    stored_dl_sources = dict(stored_settings.get("dl_sources") or {})
    current_param = settings.parameter_sources[on_arm.param_source].as_dict()
    stored_param = dict(stored_params.get(on_arm.param_source) or {})
    if not stored_param:
        return False
    for field in ("path_template", "identity_manifest_path", "trained_with_dl_id"):
        if stored_param.get(field) != current_param.get(field):
            return False

    dl_id = str(on_arm.dl_id or "")
    if not dl_id or dl_id not in settings.dl_sources:
        return False
    current_dl = settings.dl_sources[dl_id].as_dict()
    stored_dl = dict(stored_dl_sources.get(dl_id) or {})
    if not stored_dl:
        return False
    for field in (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "threshold",
        "score_source",
    ):
        if stored_dl.get(field) != current_dl.get(field):
            return False
    return True

def _archived_pair_source_is_self_contained(
    *,
    run_artifact_identities: dict[str, Any],
    dl_id: str,
    score_source: str,
) -> bool:
    """Require immutable source provenance before reusing an archived completed pair."""

    required_names = (
        ("manifest", "audit", "forward_scores")
        if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
        else ("forward_scores",)
        if score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
        else ()
    )
    return bool(required_names) and all(
        bool(_artifact_identity_sha(run_artifact_identities.get(f"dl:{dl_id}:{name}")))
        for name in required_names
    )

def _comparison_runs_roots(*, root: Path, settings: StrategyComparisonSettings) -> tuple[Path, ...]:
    """Return current write root plus config-declared read-only legacy cache roots."""

    roots: list[Path] = []
    for raw in (settings.output_root, *settings.reuse_output_roots):
        candidate = _resolve_relative_path(root, str(raw)) / "runs"
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)

def _find_reusable_pair_with_archived_source(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    off_arm: StrategyComparisonArm,
    on_arm: StrategyComparisonArm,
) -> dict[str, Any] | None:
    """Reuse a completed pair when its old PIT source no longer exists locally.

    This path is intentionally narrower than normal cache reuse.  It is allowed only when
    the current parameter artifact is present and byte-identical to the completed run, the
    replay/config contracts still match, the historical run recorded complete PIT source
    identities, and every required pair output is still present.  No missing model/audit
    artifact is reconstructed or treated as READY.
    """

    if not settings.preparation.reuse_completed_results:
        return None
    dl_id = str(on_arm.dl_id or "")
    if not dl_id or dl_id not in settings.dl_sources:
        return None
    score_source = settings.dl_sources[dl_id].score_source
    if score_source not in {
        SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    }:
        return None
    dl_status = dict((status.get("dl_sources") or {}).get(dl_id) or {})
    if bool(dl_status.get("ready")):
        return None
    current_param_sha = _arm_parameter_evaluation_sha256(status, on_arm.arm_id)
    if not current_param_sha:
        return None

    comparison_period = dict(status.get("comparison_period") or {})
    if not comparison_period:
        return None
    runs_roots = tuple(
        path for path in _comparison_runs_roots(root=root, settings=settings)
        if path.is_dir()
    )
    if not runs_roots:
        return None
    current_group_id = _pair_group_id(
        param_source=on_arm.param_source,
        rule_policy=on_arm.rule_policy,
        dl_id=dl_id,
        dl_runtime_mode=str(on_arm.dl_runtime_mode or ""),
        arm_id=on_arm.arm_id,
    )
    expected_off = _replay_arm_contract(off_arm.as_dict())
    expected_on = _replay_arm_contract(on_arm.as_dict())

    run_dirs = sorted(
        (
            path
            for runs_root in runs_roots
            for path in runs_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        run_payload = _read_json(run_dir / "strategy_comparison.json")
        if not isinstance(run_payload, dict) or str(run_payload.get("status") or "") != "COMPLETED":
            continue
        stored_settings = dict(run_payload.get("settings") or {})
        if not _settings_core_matches_archived_pair(
            settings=settings,
            stored_settings=stored_settings,
            on_arm=on_arm,
        ):
            continue
        if dict(run_payload.get("comparison_period") or {}) != comparison_period:
            continue
        stored_arms = dict(stored_settings.get("arms") or {})
        stored_off = dict(stored_arms.get(off_arm.arm_id) or {})
        stored_on = dict(stored_arms.get(on_arm.arm_id) or {})
        if not stored_off or not stored_on:
            continue
        if _replay_arm_contract(stored_off) != expected_off or _replay_arm_contract(stored_on) != expected_on:
            continue

        run_artifacts = dict(run_payload.get("artifact_identities") or {})
        if not _archived_pair_source_is_self_contained(
            run_artifact_identities=run_artifacts,
            dl_id=dl_id,
            score_source=score_source,
        ):
            continue

        # Archived fallback is only a substitute for artifacts that are no longer
        # available locally.  It must never override a current artifact identity
        # that explicitly proves different bytes; otherwise a changed frozen score
        # could miss the normal cache and then be incorrectly resurrected here.
        current_artifacts = dict(status.get("artifact_identities") or {})
        source_identity_names = (
            ("manifest", "audit", "forward_scores")
            if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else ("forward_scores",)
        )
        current_identity_conflict = any(
            (current_sha := _artifact_identity_sha(
                current_artifacts.get(f"dl:{dl_id}:{name}")
            ))
            and current_sha
            != _artifact_identity_sha(run_artifacts.get(f"dl:{dl_id}:{name}"))
            for name in source_identity_names
        )
        if current_identity_conflict:
            continue

        pairs = dict(run_payload.get("pairs") or {})
        stored_group_id = _stored_pair_group_id(
            pairs=pairs,
            param_source=str(stored_on.get("param_source") or ""),
            rule_policy=str(stored_on.get("rule_policy") or ""),
            dl_id=str(stored_on.get("dl_id") or ""),
            dl_runtime_mode=str(stored_on.get("dl_runtime_mode") or ""),
            arm_id=on_arm.arm_id,
        )
        if stored_group_id is None:
            continue
        pair_payload = pairs.get(stored_group_id)
        if not isinstance(pair_payload, dict):
            continue
        metadata = dict(pair_payload.get("metadata") or {})
        if int(metadata.get("schema_version") or 0) != int(STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION):
            continue
        stored_param_sha = str(metadata.get("params_file_sha256") or "").strip().lower()
        if stored_param_sha != current_param_sha:
            continue
        pair_dir = run_dir / "pairs" / stored_group_id
        if not all(
            path.is_file()
            for path in _pair_cache_required_files(pair_dir, on_arm=on_arm)
        ):
            continue
        try:
            stored_fingerprint = _pair_cache_fingerprint_from_payload(
                settings_payload=stored_settings,
                artifact_identities=run_artifacts,
                comparison_period=comparison_period,
                off_arm_payload=stored_off,
                on_arm_payload=stored_on,
                engine_schema_version=STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
                parameter_evaluation_sha256=stored_param_sha,
            )
        except (TypeError, ValueError):
            continue
        return {
            "fingerprint": stored_fingerprint,
            "source_run_dir": run_dir,
            "source_pair_dir": pair_dir,
            "source_group_id": stored_group_id,
            "current_group_id": current_group_id,
            "source_artifact_mode": "archived_completed_pair",
            "archived_dl_id": dl_id,
        }
    return None

def _bool_values(series: pd.Series, *, fallback_non_null: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool, copy=False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    known = normalized.isin({"true", "false", "1", "0"})
    parsed = normalized.isin({"true", "1"})
    return np.where(known.to_numpy(), parsed.to_numpy(), fallback_non_null.to_numpy())


def _legacy_pair_score_projection_matches_current(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    run_payload: dict[str, Any],
    pair_dir: Path,
    off_arm: StrategyComparisonArm,
    on_arm: StrategyComparisonArm,
    stored_settings: dict[str, Any],
    stored_off: dict[str, Any],
    stored_on: dict[str, Any],
    pair_metadata: dict[str, Any],
) -> bool:
    """Migrate a legacy whole-file PIT cache only after replay-consumed scores match.

    Old completed pairs predate per-column projection hashes.  A newly added auxiliary
    PIT column may therefore change manifest/audit/file SHA while leaving an existing
    arm's actual inputs untouched.  Reuse is permitted only when the immutable replay
    contract still matches and every score/availability value recorded in the pair's
    active orderable sidecar is identical to the current PIT source for the columns the
    arm consumes.
    """

    if not _settings_core_matches_archived_pair(
        settings=settings, stored_settings=stored_settings, on_arm=on_arm
    ):
        return False
    if dict(run_payload.get("comparison_period") or {}) != dict(
        status.get("comparison_period") or {}
    ):
        return False
    if _replay_arm_contract(stored_off) != _replay_arm_contract(off_arm.as_dict()):
        return False
    if _replay_arm_contract(stored_on) != _replay_arm_contract(on_arm.as_dict()):
        return False
    current_param_sha = _arm_parameter_evaluation_sha256(status, on_arm.arm_id)
    stored_param_sha = str(pair_metadata.get("params_file_sha256") or "").strip().lower()
    if not current_param_sha or stored_param_sha != current_param_sha:
        return False

    requirements = runtime_score_projection_requirements(
        settings_payload=settings.as_dict(), on_arm_payload=on_arm.as_dict()
    )
    if not requirements:
        return False
    current_dl_sources = settings.dl_sources
    stored_dl_sources = dict(stored_settings.get("dl_sources") or {})
    source_fields = (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "threshold",
        "score_source",
    )
    for source_id in requirements:
        current_source = current_dl_sources.get(source_id)
        stored_source = dict(stored_dl_sources.get(source_id) or {})
        if current_source is None or not stored_source:
            return False
        current_source_payload = current_source.as_dict()
        if any(
            stored_source.get(field) != current_source_payload.get(field)
            for field in source_fields
        ):
            return False
        if current_source.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            return False

    sidecar_path = pair_dir / "score_ranking_orderable_candidates.csv"
    if not sidecar_path.is_file():
        return False
    try:
        sidecar = pd.read_csv(sidecar_path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if sidecar.empty or "ticker" not in sidecar.columns:
        return False
    sidecar = sidecar.copy()
    sidecar["ticker"] = sidecar["ticker"].fillna("").astype(str).str.strip()
    if bool((sidecar["ticker"] == "").any()):
        return False

    current_artifacts = dict(status.get("artifact_identities") or {})
    primary_dl_id = str(on_arm.dl_id or "").strip()
    for source_id, columns in requirements.items():
        artifact_identity = dict(
            current_artifacts.get(f"dl:{source_id}:forward_scores") or {}
        )
        raw_path = str(artifact_identity.get("path") or "").strip()
        if not raw_path:
            return False
        try:
            score_path = _resolve_relative_path(root, raw_path)
            table = load_selection_point_in_time_score_table_from_path(str(score_path))
        except (OSError, ValueError, KeyError, TypeError):
            return False

        source_payload = current_dl_sources[source_id].as_dict()
        primary_column = primary_score_column_for_source(
            str(source_payload.get("score_source") or "")
        )
        current_frame = table.reset_index()
        for score_column in columns:
            if score_column not in current_frame.columns:
                return False
            if source_id == primary_dl_id and score_column == primary_column:
                side_score_column = "breakout_quality_score"
                side_date_column = "breakout_quality_score_date"
                side_available_column = "breakout_quality_score_available"
            else:
                side_score_column = "breakout_quality_safety_score"
                side_date_column = "breakout_quality_safety_score_date"
                side_available_column = "breakout_quality_safety_score_available"
            if side_score_column not in sidecar.columns:
                return False
            date_values = (
                sidecar[side_date_column].fillna("").astype(str).str.strip()
                if side_date_column in sidecar.columns
                else pd.Series("", index=sidecar.index, dtype=object)
            )
            if "signal_date" in sidecar.columns:
                fallback_dates = sidecar["signal_date"].fillna("").astype(str).str.strip()
                date_values = date_values.mask(date_values == "", fallback_dates)
            if bool((date_values == "").any()):
                return False

            stored_scores = pd.to_numeric(sidecar[side_score_column], errors="coerce")
            stored_non_null = stored_scores.notna()
            if side_available_column in sidecar.columns:
                stored_available = _bool_values(
                    sidecar[side_available_column], fallback_non_null=stored_non_null
                )
            else:
                stored_available = stored_non_null.to_numpy(dtype=bool, copy=False)

            left = pd.DataFrame(
                {
                    "_row_id": np.arange(len(sidecar), dtype=np.int64),
                    "ticker": sidecar["ticker"].to_numpy(),
                    "date": date_values.to_numpy(),
                    "_stored_score": stored_scores.to_numpy(dtype=np.float64, copy=False),
                    "_stored_available": stored_available,
                }
            )
            right = current_frame[["ticker", "date", score_column]].rename(
                columns={score_column: "_current_score"}
            )
            merged = left.merge(
                right, on=["ticker", "date"], how="left", validate="many_to_one"
            ).sort_values("_row_id", kind="mergesort")
            current_available = merged["_current_score"].notna().to_numpy(dtype=bool)
            if not np.array_equal(
                merged["_stored_available"].to_numpy(dtype=bool), current_available
            ):
                return False
            mask = merged["_stored_available"].to_numpy(dtype=bool)
            if bool(mask.any()):
                stored_values = merged.loc[mask, "_stored_score"].to_numpy(dtype=np.float64)
                current_values = merged.loc[mask, "_current_score"].to_numpy(dtype=np.float64)
                if not np.array_equal(stored_values, current_values):
                    return False
    return True


def _find_reusable_pair(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    off_arm: StrategyComparisonArm,
    on_arm: StrategyComparisonArm,
) -> dict[str, Any] | None:
    if not settings.preparation.reuse_completed_results:
        return None
    runs_roots = tuple(
        path for path in _comparison_runs_roots(root=root, settings=settings)
        if path.is_dir()
    )
    if not runs_roots:
        return None
    expected = _current_pair_cache_fingerprint(
        settings=settings,
        status=status,
        off_arm=off_arm,
        on_arm=on_arm,
    )
    current_group_id = _pair_group_id(
        param_source=on_arm.param_source,
        rule_policy=on_arm.rule_policy,
        dl_id=str(on_arm.dl_id or ""),
        dl_runtime_mode=str(on_arm.dl_runtime_mode or ""),
        arm_id=on_arm.arm_id,
    )
    run_dirs = sorted(
        (
            path
            for runs_root in runs_roots
            for path in runs_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        run_payload = _read_json(run_dir / "strategy_comparison.json")
        if not isinstance(run_payload, dict):
            continue
        if str(run_payload.get("status") or "") != "COMPLETED":
            continue
        stored_settings = dict(run_payload.get("settings") or {})
        stored_arms = dict(stored_settings.get("arms") or {})
        stored_off = dict(stored_arms.get(off_arm.arm_id) or {})
        stored_on = dict(stored_arms.get(on_arm.arm_id) or {})
        if not stored_off or not stored_on:
            continue
        pairs = dict(run_payload.get("pairs") or {})
        stored_group_id = _stored_pair_group_id(
            pairs=pairs,
            param_source=str(stored_on.get("param_source") or ""),
            rule_policy=str(stored_on.get("rule_policy") or ""),
            dl_id=str(stored_on.get("dl_id") or ""),
            dl_runtime_mode=str(stored_on.get("dl_runtime_mode") or ""),
            arm_id=on_arm.arm_id,
        )
        if stored_group_id is None:
            continue
        pair_payload = pairs.get(stored_group_id)
        if not isinstance(pair_payload, dict):
            continue
        metadata = dict(pair_payload.get("metadata") or {})
        pair_dir = run_dir / "pairs" / stored_group_id
        if not all(
            path.is_file()
            for path in _pair_cache_required_files(pair_dir, on_arm=on_arm)
        ):
            continue
        try:
            actual = _pair_cache_fingerprint_from_payload(
                settings_payload=stored_settings,
                artifact_identities=dict(run_payload.get("artifact_identities") or {}),
                comparison_period=dict(run_payload.get("comparison_period") or {}),
                off_arm_payload=stored_off,
                on_arm_payload=stored_on,
                engine_schema_version=int(
                    metadata.get("schema_version")
                    or STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION
                ),
                parameter_evaluation_sha256=str(
                    metadata.get("params_file_sha256") or ""
                ),
            )
        except (TypeError, ValueError):
            continue
        source_artifact_mode = "projection_identity"
        if actual != expected:
            if not _legacy_pair_score_projection_matches_current(
                root=root,
                settings=settings,
                status=status,
                run_payload=run_payload,
                pair_dir=pair_dir,
                off_arm=off_arm,
                on_arm=on_arm,
                stored_settings=stored_settings,
                stored_off=stored_off,
                stored_on=stored_on,
                pair_metadata=metadata,
            ):
                continue
            source_artifact_mode = "legacy_pair_runtime_score_projection_verified"
        return {
            "fingerprint": expected,
            "source_run_dir": run_dir,
            "source_pair_dir": pair_dir,
            "source_group_id": stored_group_id,
            "current_group_id": current_group_id,
            "source_artifact_mode": source_artifact_mode,
        }
    return _find_reusable_pair_with_archived_source(
        root=root,
        settings=settings,
        status=status,
        off_arm=off_arm,
        on_arm=on_arm,
    )

def _collect_replay_cache_status(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    baseline_groups: dict[str, Any] = {}
    if status.get("comparison_period"):
        for param_source, rule_policy, off_arm, on_arm in _execution_pairs(settings):
            entry = _find_reusable_pair(
                root=root,
                settings=settings,
                status=status,
                off_arm=off_arm,
                on_arm=on_arm,
            )
            pairs[on_arm.arm_id] = entry
            if entry is not None:
                baseline_groups.setdefault(
                    f"{param_source}::{resolve_strategy_comparison_arm_param_policy(settings, off_arm)}::{rule_policy}",
                    {
                        "off_arm_id": off_arm.arm_id,
                        "source_pair_dir": entry["source_pair_dir"],
                    },
                )
        for off_arm in _standalone_baseline_arms(settings):
            source_pair_dir = _find_reusable_baseline_source(
                root=root, settings=settings, status=status, off_arm=off_arm
            )
            if source_pair_dir is not None:
                baseline_groups.setdefault(
                    f"{off_arm.param_source}::{resolve_strategy_comparison_arm_param_policy(settings, off_arm)}::{off_arm.rule_policy}",
                    {
                        "off_arm_id": off_arm.arm_id,
                        "source_pair_dir": source_pair_dir,
                    },
                )
    arm_states: dict[str, Any] = {}
    for arm in settings.enabled_arms:
        if arm.dl_enabled:
            evidence = pairs.get(arm.arm_id)
            source_dir = (
                evidence.get("source_pair_dir")
                if isinstance(evidence, dict)
                else None
            )
        else:
            group_key = (
                f"{arm.param_source}::"
                f"{resolve_strategy_comparison_arm_param_policy(settings, arm)}::"
                f"{arm.rule_policy}"
            )
            evidence = baseline_groups.get(group_key)
            source_dir = (
                evidence.get("source_pair_dir")
                if isinstance(evidence, dict)
                else None
            )
        arm_states[arm.arm_id] = resolve_strategy_result_state(
            verified_evidence=evidence if isinstance(evidence, dict) else None,
            source_dir=source_dir,
            missing_reason=f"{arm.arm_id}尚無identity一致的completed result",
        ).as_dict()
    return {
        "pairs": pairs,
        "baseline_groups": baseline_groups,
        "arm_states": arm_states,
    }

def _apply_completed_pair_dependency_waivers(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    replay_cache: dict[str, Any],
) -> dict[str, Any]:
    """Do not require DL source artifacts when every dependent arm reuses a completed pair."""

    pairs = dict(replay_cache.get("pairs") or {})
    waived_dl_ids: set[str] = set()
    for dl_id, source_row in dict(status.get("dl_sources") or {}).items():
        if bool(source_row.get("ready")):
            continue
        dependent_arms = tuple(
            arm
            for arm in settings.enabled_arms
            if arm.dl_enabled and str(arm.dl_id or "") == str(dl_id)
        )
        if not dependent_arms:
            continue
        if all(isinstance(pairs.get(arm.arm_id), dict) for arm in dependent_arms):
            waived_dl_ids.add(str(dl_id))
    if not waived_dl_ids:
        status["replay_cache"] = replay_cache
        return status

    plan: StrategyPreparationPlan = status["preparation_plan"]
    rewritten_actions: list[StrategyPreparationAction] = []
    for item in plan.actions:
        parts = item.artifact_key.split(":")
        dl_id = parts[1] if len(parts) >= 2 and parts[0] == "dl" else None
        if dl_id in waived_dl_ids:
            rewritten_actions.append(
                StrategyPreparationAction(
                    action_id=item.action_id,
                    artifact_key=item.artifact_key,
                    action="REUSE",
                    builder_type=None,
                    description=(
                        "本次使用此DL的arms全部重用identity一致的completed pair；"
                        "目前執行不需要模型／manifest／report／score source"
                    ),
                    path=item.path,
                    dependencies=item.dependencies,
                    producer_work_type="existing_artifact",
                    execution_priority=item.execution_priority,
                )
            )
        else:
            rewritten_actions.append(item)

    rewritten_plan = StrategyPreparationPlan.from_actions(rewritten_actions)
    status = dict(status)
    status["preparation_plan"] = rewritten_plan
    status["overall_status"] = rewritten_plan.overall_status
    status["comparison_ready"] = rewritten_plan.overall_status == "READY"
    status["replay_cache"] = replay_cache
    status["completed_pair_dependency_waivers"] = sorted(waived_dl_ids)
    for dl_id in waived_dl_ids:
        row = dict(status["dl_sources"][dl_id])
        row["status"] = "COMPLETED_PAIR_REUSE"
        row["required_for_current_execution"] = False
        files = {}
        for key, file_row in dict(row.get("files") or {}).items():
            updated = dict(file_row)
            updated["action"] = "REUSE"
            updated["required_for_current_execution"] = False
            files[key] = updated
        row["files"] = files
        status["dl_sources"][dl_id] = row
    return status

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()

def _resolve_pair_pinned_score_path(
    *,
    root: Path,
    raw_path: Any,
) -> Path | None:
    """Resolve an archived/current score path only when it remains inside project root."""

    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved

def _historical_continuous_score_provenance_entries(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    dl_id: str,
) -> tuple[dict[str, Any], ...]:
    """Find completed pairs that can prove one historical continuous-score artifact.

    Source provenance is independent of current profile membership.  A retired
    compatibility arm may therefore prove the exact frozen score used by a current
    arm, provided the DL identity and recorded score SHA remain identical.
    """

    source = settings.dl_sources.get(str(dl_id))
    if source is None or source.score_source != SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        return ()
    current_identity = source.as_dict()
    output: list[dict[str, Any]] = []
    seen: set[tuple[Path, str]] = set()
    run_dirs = sorted(
        (
            path
            for runs_root in _comparison_runs_roots(root=root, settings=settings)
            if runs_root.is_dir()
            for path in runs_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        payload = _read_json(run_dir / "strategy_comparison.json")
        if not isinstance(payload, dict) or str(payload.get("status") or "") != "COMPLETED":
            continue
        stored_settings = dict(payload.get("settings") or {})
        stored_source = dict((stored_settings.get("dl_sources") or {}).get(str(dl_id)) or {})
        if not stored_source:
            continue
        if any(
            stored_source.get(field) != current_identity.get(field)
            for field in (
                "filter_id",
                "model_architecture",
                "experiment_profile",
                "threshold",
                "score_source",
            )
        ):
            continue
        archived_identity = dict(
            (payload.get("artifact_identities") or {}).get(
                f"dl:{dl_id}:forward_scores"
            )
            or {}
        )
        if not _artifact_identity_sha(archived_identity):
            continue

        stored_arms = dict(stored_settings.get("arms") or {})
        pairs = dict(payload.get("pairs") or {})
        for arm_id, raw_arm in stored_arms.items():
            arm = dict(raw_arm or {})
            if not bool(arm.get("dl_enabled")) or str(arm.get("dl_id") or "") != str(dl_id):
                continue
            source_group_id = _stored_pair_group_id(
                pairs=pairs,
                param_source=str(arm.get("param_source") or ""),
                rule_policy=str(arm.get("rule_policy") or ""),
                dl_id=str(dl_id),
                dl_runtime_mode=str(arm.get("dl_runtime_mode") or ""),
                arm_id=str(arm_id),
            )
            if source_group_id is None:
                continue
            pair_payload = pairs.get(source_group_id)
            if not isinstance(pair_payload, dict):
                continue
            key = (run_dir, source_group_id)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "source_run_dir": run_dir,
                    "source_pair_dir": run_dir / "pairs" / source_group_id,
                    "source_group_id": source_group_id,
                    "provenance_arm_id": str(arm_id),
                    "source_artifact_mode": "historical_completed_pair_provenance",
                }
            )
    return tuple(output)

def _continuous_score_provenance_entries(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    replay_cache: dict[str, Any],
    dl_id: str,
) -> tuple[dict[str, Any], ...]:
    """Return completed-pair provenance independently of active-arm membership.

    Current profile membership determines artifact demand, not artifact existence.
    Any formally completed pair can prove the same frozen DL source when its archived
    identity and score SHA pass the resolver checks below.
    """

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in dict(replay_cache.get("pairs") or {}).values():
        if not isinstance(raw, dict):
            continue
        key = (
            str(raw.get("source_run_dir") or ""),
            str(raw.get("source_group_id") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        output.append(raw)
    for raw in _historical_continuous_score_provenance_entries(
        root=root,
        settings=settings,
        dl_id=str(dl_id),
    ):
        key = (
            str(raw.get("source_run_dir") or ""),
            str(raw.get("source_group_id") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        output.append(raw)
    return tuple(output)

def _resolve_completed_pair_continuous_score_binding(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    replay_cache: dict[str, Any],
    dl_id: str,
    diagnostics: list[str] | None = None,
) -> ResolvedContinuousScoreBinding | None:
    """Resolve one frozen OOS score file pinned by a reusable completed pair.

    This is deliberately score-only.  It never rebuilds a model and never derives a
    score universe from replay candidate/trade outputs.  The score CSV must still exist
    byte-for-byte and its SHA must match the completed pair provenance.

    Recovery checks the archived artifact-identity path first, then the current
    canonical path.  This keeps a completed pair reusable when a later config/path
    migration changes where the canonical loader looks, without weakening provenance.
    """

    def note(message: str) -> None:
        if diagnostics is not None:
            diagnostics.append(str(message))

    source = settings.dl_sources.get(str(dl_id))
    if source is None or source.score_source != SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        note("SOURCE_NOT_CONTINUOUS_OOS")
        return None
    provenance_entries = _continuous_score_provenance_entries(
        root=root,
        settings=settings,
        replay_cache=replay_cache,
        dl_id=str(dl_id),
    )
    if not provenance_entries:
        note("NO_COMPLETED_PAIR_PROVENANCE")
        return None

    source_row = dict((status.get("dl_sources") or {}).get(str(dl_id)) or {})
    score_row = dict((source_row.get("files") or {}).get("forward_scores") or {})
    current_score_display_path = str(score_row.get("path") or "").strip()

    current_identity = source.as_dict()
    current_period = dict(status.get("comparison_period") or {})
    current_start = str(current_period.get("start") or "")
    current_end = str(current_period.get("end") or "")

    for entry in provenance_entries:
        run_dir = Path(str(entry.get("source_run_dir") or "")).resolve()
        payload = _read_json(run_dir / "strategy_comparison.json")
        if not isinstance(payload, dict) or str(payload.get("status") or "") != "COMPLETED":
            note("COMPLETED_PAIR_REPORT_INVALID")
            continue
        stored_settings = dict(payload.get("settings") or {})
        stored_source = dict((stored_settings.get("dl_sources") or {}).get(str(dl_id)) or {})
        if not stored_source:
            note("COMPLETED_PAIR_DL_SOURCE_MISSING")
            continue
        if any(
            stored_source.get(field) != current_identity.get(field)
            for field in (
                "filter_id",
                "model_architecture",
                "experiment_profile",
                "threshold",
                "score_source",
            )
        ):
            note("COMPLETED_PAIR_DL_IDENTITY_MISMATCH")
            continue

        archived_identity = dict(
            (payload.get("artifact_identities") or {}).get(
                f"dl:{dl_id}:forward_scores"
            )
            or {}
        )
        archived_sha = _artifact_identity_sha(archived_identity)
        if not archived_sha:
            note("COMPLETED_PAIR_SCORE_SHA_MISSING")
            continue

        candidate_specs = (
            ("archived_artifact_identity", archived_identity.get("path")),
            ("current_canonical_path", current_score_display_path),
        )
        candidate_paths: list[tuple[str, Path]] = []
        seen_paths: set[Path] = set()
        for path_source, raw_path in candidate_specs:
            score_path = _resolve_pair_pinned_score_path(
                root=root,
                raw_path=raw_path,
            )
            if score_path is None or score_path in seen_paths:
                continue
            seen_paths.add(score_path)
            candidate_paths.append((path_source, score_path))
        if not candidate_paths:
            note("NO_PROJECT_SCOPED_SCORE_PATH")
            continue

        stored_period = dict(payload.get("comparison_period") or {})
        stored_start = str(stored_period.get("start") or "")
        stored_end = str(stored_period.get("end") or "")
        if not stored_start or not stored_end:
            note("COMPLETED_PAIR_PERIOD_MISSING")
            continue

        source_group_id = str(entry.get("source_group_id") or "").strip()
        pair_payload = dict((payload.get("pairs") or {}).get(source_group_id) or {})
        pair_metadata = dict(pair_payload.get("metadata") or {})
        pair_period = dict(pair_metadata.get("comparison_period") or {})
        archived_coverage = dict(pair_metadata.get("score_signal_coverage") or {})
        archived_required_start = str(archived_coverage.get("required_start") or "").strip()
        archived_first_scored_event = str(archived_coverage.get("first_scored_event") or "").strip()
        archived_available_through = str(archived_coverage.get("available_through") or "").strip()
        if not (
            source_group_id
            and pair_payload
            and archived_required_start
            and archived_first_scored_event
            and archived_available_through
        ):
            note("COMPLETED_PAIR_SCORE_COVERAGE_MISSING")
            continue
        if not (
            archived_required_start <= archived_first_scored_event <= archived_available_through
        ):
            note("COMPLETED_PAIR_SCORE_COVERAGE_INVALID")
            continue
        if pair_period and (
            str(pair_period.get("start") or "") != archived_required_start
            or str(pair_period.get("end") or "") != archived_available_through
        ):
            note("COMPLETED_PAIR_SCORE_COVERAGE_PERIOD_MISMATCH")
            continue
        if stored_start != archived_required_start or stored_end != archived_available_through:
            note("COMPLETED_RUN_SCORE_COVERAGE_PERIOD_MISMATCH")
            continue
        if current_start and current_start != archived_required_start:
            note("CURRENT_PERIOD_START_DIFFERS_FROM_COMPLETED_PAIR")
            continue
        if current_end and current_end != archived_available_through:
            note("CURRENT_PERIOD_END_DIFFERS_FROM_COMPLETED_PAIR")
            continue

        for path_source, score_path in candidate_paths:
            if not score_path.is_file():
                note(f"SCORE_FILE_MISSING:{path_source}")
                continue
            actual_sha = _file_sha256(score_path)
            if actual_sha != archived_sha:
                note(f"SCORE_SHA_MISMATCH:{path_source}")
                continue
            try:
                table = load_continuous_ranker_oos_score_table_from_path(
                    str(score_path),
                    str(source.experiment_profile),
                )
            except (OSError, ValueError, KeyError, TypeError):
                note(f"SCORE_TABLE_INVALID:{path_source}")
                continue
            available_from = str(table.attrs.get("available_from") or "")
            available_through = str(table.attrs.get("available_through") or "")
            if not available_from or not available_through:
                note(f"SCORE_PERIOD_METADATA_MISSING:{path_source}")
                continue
            if (
                available_from != archived_first_scored_event
                or available_through != archived_available_through
            ):
                note(f"SCORE_SIGNAL_COVERAGE_MISMATCH:{path_source}")
                continue
            note(f"REUSE_OK:{path_source}")
            return ResolvedContinuousScoreBinding(
                score_path=score_path,
                sha256=actual_sha,
                available_from=available_from,
                available_through=available_through,
                execution_start=archived_required_start,
                provenance_run_dir=run_dir,
                provenance_pair_dir=Path(
                    str(entry.get("source_pair_dir") or "")
                ).resolve(),
                path_source=path_source,
                archived_path=str(archived_identity.get("path") or ""),
                current_canonical_path=current_score_display_path,
            )
    return None

def _completed_pair_pinned_continuous_score(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    replay_cache: dict[str, Any],
    dl_id: str,
    diagnostics: list[str] | None = None,
) -> dict[str, Any] | None:
    """Compatibility projection of the typed frozen-score artifact resolver."""

    binding = _resolve_completed_pair_continuous_score_binding(
        root=root,
        settings=settings,
        status=status,
        replay_cache=replay_cache,
        dl_id=dl_id,
        diagnostics=diagnostics,
    )
    return None if binding is None else binding.as_dict()

def _apply_completed_pair_frozen_score_reuse(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    replay_cache: dict[str, Any],
) -> dict[str, Any]:
    """Allow a new continuous-score arm to reuse pair-pinned frozen scores.

    A completed comparison can prove the exact SHA of the frozen score artifact even
    when the historical model/report files have since been retired.  This waiver is
    valid only when at least one arm with the same DL source is cached, at least one
    arm still needs to RUN, and either the archived artifact-identity path or current
    canonical score path still exists byte-identically.
    """

    pairs = dict(replay_cache.get("pairs") or {})
    overrides: dict[str, Any] = {}
    reuse_diagnostics: dict[str, list[str]] = {}
    for dl_id, source_row in dict(status.get("dl_sources") or {}).items():
        if bool(source_row.get("ready")):
            continue
        source = settings.dl_sources.get(str(dl_id))
        if source is None or source.score_source != SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            continue
        dependent_arms = tuple(
            arm
            for arm in settings.enabled_arms
            if arm.dl_enabled and str(arm.dl_id or "") == str(dl_id)
        )
        if not dependent_arms:
            continue
        if all(isinstance(pairs.get(arm.arm_id), dict) for arm in dependent_arms):
            continue
        diagnostics: list[str] = []
        binding = _resolve_completed_pair_continuous_score_binding(
            root=root,
            settings=settings,
            status=status,
            replay_cache=replay_cache,
            dl_id=str(dl_id),
            diagnostics=diagnostics,
        )
        reuse_diagnostics[str(dl_id)] = diagnostics
        if binding is not None:
            overrides[str(dl_id)] = binding.as_dict()

    if not overrides and not reuse_diagnostics:
        return status

    plan: StrategyPreparationPlan = status["preparation_plan"]
    rewritten_actions: list[StrategyPreparationAction] = []
    for item in plan.actions:
        parts = item.artifact_key.split(":")
        dl_id = parts[1] if len(parts) >= 3 and parts[0] == "dl" else None
        artifact_name = parts[2] if len(parts) >= 3 and parts[0] == "dl" else None
        if dl_id in overrides and artifact_name in {"model", "manifest", "report"}:
            rewritten_actions.append(
                StrategyPreparationAction(
                    action_id=item.action_id,
                    artifact_key=item.artifact_key,
                    action="NOT_REQUIRED",
                    builder_type=None,
                    description=(
                        "新arm直接重用completed pair以SHA釘住的frozen Forward score；"
                        "目前策略執行不需要歷史模型／manifest／report"
                    ),
                    path=item.path,
                    dependencies=item.dependencies,
                    producer_work_type="existing_artifact",
                    execution_priority=item.execution_priority,
                )
            )
        elif dl_id in overrides and artifact_name == "forward_scores":
            source_info = overrides[dl_id]
            rewritten_actions.append(
                StrategyPreparationAction(
                    action_id=item.action_id,
                    artifact_key=item.artifact_key,
                    action="REUSE",
                    builder_type=None,
                    description=(
                        "重用既有completed pair以artifact identity path + SHA驗證的"
                        f"frozen OOS continuous scores（{source_info['path_source']}）"
                    ),
                    path=project_relative_display_path(
                        Path(source_info["score_path"]), project_root=root
                    ),
                    dependencies=(),
                    producer_work_type="existing_artifact",
                    execution_priority=item.execution_priority,
                )
            )
        elif dl_id in reuse_diagnostics and item.action == "BLOCKED":
            unique_codes = tuple(dict.fromkeys(reuse_diagnostics[dl_id]))
            diagnostic_text = ",".join(unique_codes[:6]) or "UNKNOWN"
            rewritten_actions.append(
                StrategyPreparationAction(
                    action_id=item.action_id,
                    artifact_key=item.artifact_key,
                    action=item.action,
                    builder_type=item.builder_type,
                    description=(
                        "completed pair存在，但frozen-score provenance recovery未通過："
                        f"{diagnostic_text}；"
                        "Strategy Compare不重訓；若archived/current score皆不可驗證，"
                        "請由模型訓練工作類型恢復正式frozen score"
                    ),
                    path=item.path,
                    dependencies=item.dependencies,
                    producer_work_type=item.producer_work_type,
                    execution_priority=item.execution_priority,
                )
            )
        else:
            rewritten_actions.append(item)

    rewritten_plan = StrategyPreparationPlan.from_actions(rewritten_actions)
    updated = dict(status)
    updated["preparation_plan"] = rewritten_plan
    updated["overall_status"] = rewritten_plan.overall_status
    updated["comparison_ready"] = rewritten_plan.overall_status == "READY"
    updated["continuous_score_reuse_diagnostics"] = {
        dl_id: list(values) for dl_id, values in reuse_diagnostics.items()
    }
    if not overrides:
        return updated

    updated["continuous_score_overrides"] = {
        dl_id: {
            "score_path": str(value["score_path"]),
            "sha256": value["sha256"],
            "available_from": value["available_from"],
            "available_through": value["available_through"],
            "execution_start": value["execution_start"],
            "provenance_run_dir": str(value["provenance_run_dir"]),
            "provenance_pair_dir": str(value["provenance_pair_dir"]),
            "path_source": value["path_source"],
            "archived_path": value["archived_path"],
            "current_canonical_path": value["current_canonical_path"],
        }
        for dl_id, value in overrides.items()
    }
    artifact_identities = dict(updated.get("artifact_identities") or {})
    for dl_id, value in overrides.items():
        row = dict(updated["dl_sources"][dl_id])
        row["ready"] = True
        row["status"] = "COMPLETED_PAIR_PINNED_FROZEN_SCORE"
        row["required_for_current_execution"] = True
        files = {}
        display_score_path = project_relative_display_path(
            Path(value["score_path"]), project_root=root
        )
        for key, file_row in dict(row.get("files") or {}).items():
            file_updated = dict(file_row)
            if key == "forward_scores":
                file_updated.update({
                    "ready": True,
                    "status": "READY_BY_COMPLETED_PAIR_SHA",
                    "action": "REUSE",
                    "sha256": value["sha256"],
                    "path": display_score_path,
                    "required_for_current_execution": True,
                    "path_source": value["path_source"],
                })
            else:
                file_updated.update({
                    "action": "NOT_REQUIRED",
                    "required_for_current_execution": False,
                })
            files[key] = file_updated
        row["files"] = files
        updated["dl_sources"][dl_id] = row
        artifact_identities[f"dl:{dl_id}:forward_scores"] = {
            "path": display_score_path,
            "sha256": value["sha256"],
            "status": "READY_BY_COMPLETED_PAIR_SHA",
            "path_source": value["path_source"],
        }
    updated["artifact_identities"] = artifact_identities

    # Preparation resolves the common period before completed-pair score recovery.
    # When every required continuous source is historical-only, that first pass has
    # no runtime period and legitimately returns ``None``.  Recovery has now restored
    # the exact execution coverage, so resolve the period again through the same SSOT
    # used by preparation instead of maintaining a second date-intersection rule here.
    if not dict(updated.get("comparison_period") or {}):
        recovered_runtime_periods = {
            str(dl_id): (
                str(value["execution_start"]),
                str(value["available_through"]),
            )
            for dl_id, value in overrides.items()
        }
        comparison_start, comparison_end, comparison_period_source = (
            resolve_comparison_period(
                settings=settings,
                runtime_periods=recovered_runtime_periods,
            )
        )
        if comparison_start is not None and comparison_end is not None:
            updated["comparison_period"] = {
                "start": comparison_start,
                "end": comparison_end,
            }
            updated["comparison_period_source"] = comparison_period_source

    updated["config_fingerprint"] = strategy_comparison_fingerprint(
        settings,
        artifact_identities=artifact_identities,
    )
    return updated

def _find_reusable_baseline_source(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    off_arm: StrategyComparisonArm,
) -> Path | None:
    if not settings.preparation.reuse_completed_results:
        return None
    comparison_period = dict(status.get("comparison_period") or {})
    if not comparison_period:
        return None
    expected_param_sha = _arm_parameter_evaluation_sha256(status, off_arm.arm_id)
    if not expected_param_sha:
        return None
    all_off = off_arm.rule_policy == "all_off"
    expected = {
        "dataset": settings.dataset,
        "params_file_sha256": expected_param_sha,
        "requested_param_policy": resolve_strategy_comparison_arm_param_policy(settings, off_arm),
        "param_evaluation_mode": str(
            settings.parameter_sources[off_arm.param_source].canonical_evaluation_mode or "rolling"
        ),
        "optional_entry_filter_policy": (
            OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            if all_off
            else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
        ),
        "shared_param_overrides": (
            dict(ALL_RULE_FILTERS_OFF_OVERRIDES) if all_off else {}
        ),
        "max_positions": int(settings.max_positions),
        "enable_rotation": settings.rotation == "on",
        "comparison_period": comparison_period,
    }
    run_dirs = sorted(
        (
            path
            for runs_root in _comparison_runs_roots(root=root, settings=settings)
            if runs_root.is_dir()
            for path in runs_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    required_names = (
        "strategy_comparison.json",
        "strategy_comparison.md",
        "yearly_returns_comparison.csv",
        "no_filter_equity.csv",
        "no_filter_trades.csv",
        "no_filter_daily_capacity.csv",
        "no_filter_orderable_candidates.csv",
    )
    for run_dir in run_dirs:
        pairs_root = run_dir / "pairs"
        if not pairs_root.is_dir():
            continue
        for pair_dir in sorted(
            (path for path in pairs_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            payload = _read_json(pair_dir / "strategy_comparison.json")
            if not isinstance(payload, dict) or not dict(payload.get("no_filter") or {}):
                continue
            metadata = dict(payload.get("metadata") or {})
            actual = {
                "dataset": str(metadata.get("dataset") or ""),
                "params_file_sha256": str(metadata.get("params_file_sha256") or ""),
                "requested_param_policy": str(metadata.get("requested_param_policy") or ""),
                "param_evaluation_mode": str(metadata.get("param_evaluation_mode") or ""),
                "optional_entry_filter_policy": str(
                    metadata.get("optional_entry_filter_policy") or ""
                ),
                "shared_param_overrides": dict(metadata.get("shared_param_overrides") or {}),
                "max_positions": int(metadata.get("max_positions") or 0),
                "enable_rotation": bool(metadata.get("enable_rotation")),
                "comparison_period": dict(metadata.get("comparison_period") or {}),
            }
            if actual != expected:
                continue
            if all((pair_dir / name).is_file() for name in required_names):
                return pair_dir
    return None
