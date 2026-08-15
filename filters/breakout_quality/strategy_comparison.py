"""Config-driven Breakout Quality strategy performance comparison orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from config.strategy_compare import get_strategy_comparison_settings
from core.runtime_utils import get_taipei_now
from core.strategy_comparison import (
    STRATEGY_DL_RUNTIME_MODE_HARD_FILTER,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    StrategyComparisonArm,
    StrategyComparisonSettings,
    StrategyDLSource,
    StrategyPreparationAction,
    StrategyPreparationPlan,
    strategy_comparison_fingerprint,
)
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.strategy_compare_contracts import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    STRATEGY_COMPARE_SCHEMA_VERSION as STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
)
from filters.breakout_quality.strategy_compare_sources import (
    read_json_object_or_none as _read_json,
    resolve_project_relative_path as _resolve_relative_path,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
)
from filters.breakout_quality.strategy_compare_reporting import (
    materialize_strategy_pair_readable_report,
    render_strategy_pair_simple_report,
)
from filters.breakout_quality.strategy_compare_engine import run_comparison
from filters.breakout_quality.strategy_compare_replay import run_standalone_baseline
from filters.breakout_quality.strategy_compare_plan import (
    ResolvedComparisonPlan,
    ResolvedContinuousScoreBinding,
)
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_continuous_ranker_oos_score_table_from_path,
)
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from filters.breakout_quality.strategy_rule_policies import (
    ALL_RULE_FILTERS_OFF_OVERRIDES,
)
from filters.breakout_quality.strategy_compare_preparation import (
    resolve_comparison_period,
    collect_artifact_status as collect_preparation_status,
    prepare_strategy_comparison_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 6


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_native(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


PAIR_CACHE_SCHEMA_VERSION = 1


def _pair_group_id(
    *,
    param_source: str,
    rule_policy: str,
    dl_id: str,
    dl_runtime_mode: str,
) -> str:
    return (
        f"{param_source}__{rule_policy}__{dl_id}__"
        f"{str(dl_runtime_mode).replace('-', '_')}"
    )


def _replay_arm_contract(raw: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "param_source": str(raw.get("param_source") or ""),
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


def _pair_cache_fingerprint_from_payload(
    *,
    settings_payload: dict[str, Any],
    artifact_identities: dict[str, Any],
    comparison_period: dict[str, Any],
    off_arm_payload: dict[str, Any],
    on_arm_payload: dict[str, Any],
    engine_schema_version: int,
) -> str:
    param_source = str(on_arm_payload.get("param_source") or "")
    dl_id = str(on_arm_payload.get("dl_id") or "")
    parameter_sources = dict(settings_payload.get("parameter_sources") or {})
    dl_sources = dict(settings_payload.get("dl_sources") or {})
    param_payload = dict(parameter_sources.get(param_source) or {})
    dl_payload = dict(dl_sources.get(dl_id) or {})

    artifact_keys = [f"param:{param_source}"]
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
    selected_artifacts = {
        key: artifact_identities.get(key)
        for key in sorted(set(artifact_keys))
    }

    payload = {
        "cache_schema_version": PAIR_CACHE_SCHEMA_VERSION,
        "engine_schema_version": int(engine_schema_version),
        "dataset": settings_payload.get("dataset"),
        "comparison_period": dict(comparison_period or {}),
        "param_policy": settings_payload.get("param_policy"),
        "max_positions": settings_payload.get("max_positions"),
        "rotation": settings_payload.get("rotation"),
        "parameter_source": {
            "source_id": param_source,
            "path_template": param_payload.get("path_template"),
            "identity_manifest_path": param_payload.get("identity_manifest_path"),
            "trained_with_dl_id": param_payload.get("trained_with_dl_id"),
        },
        "dl_source": {
            "dl_id": dl_id,
            "filter_id": dl_payload.get("filter_id"),
            "model_architecture": dl_payload.get("model_architecture"),
            "experiment_profile": dl_payload.get("experiment_profile"),
            "threshold": dl_payload.get("threshold"),
            "score_source": dl_payload.get("score_source"),
        },
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
    param_identity = dict(
        (status.get("artifact_identities") or {}).get(
            f"param:{on_arm.param_source}"
        )
        or {}
    )
    current_param_sha = _artifact_identity_sha(param_identity)
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
        stored_param_sha = _artifact_identity_sha(
            run_artifacts.get(f"param:{on_arm.param_source}")
        )
        if stored_param_sha != current_param_sha:
            continue
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

        stored_group_id = _pair_group_id(
            param_source=str(stored_on.get("param_source") or ""),
            rule_policy=str(stored_on.get("rule_policy") or ""),
            dl_id=str(stored_on.get("dl_id") or ""),
            dl_runtime_mode=str(stored_on.get("dl_runtime_mode") or ""),
        )
        pairs = dict(run_payload.get("pairs") or {})
        pair_payload = pairs.get(stored_group_id)
        if not isinstance(pair_payload, dict):
            continue
        metadata = dict(pair_payload.get("metadata") or {})
        if int(metadata.get("schema_version") or 0) != int(STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION):
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
        stored_group_id = _pair_group_id(
            param_source=str(stored_on.get("param_source") or ""),
            rule_policy=str(stored_on.get("rule_policy") or ""),
            dl_id=str(stored_on.get("dl_id") or ""),
            dl_runtime_mode=str(stored_on.get("dl_runtime_mode") or ""),
        )
        pairs = dict(run_payload.get("pairs") or {})
        pair_payload = pairs.get(stored_group_id)
        if not isinstance(pair_payload, dict):
            continue
        metadata = dict(pair_payload.get("metadata") or {})
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
            )
        except (TypeError, ValueError):
            continue
        if actual != expected:
            continue
        pair_dir = run_dir / "pairs" / stored_group_id
        if not all(
            path.is_file()
            for path in _pair_cache_required_files(pair_dir, on_arm=on_arm)
        ):
            continue
        return {
            "fingerprint": expected,
            "source_run_dir": run_dir,
            "source_pair_dir": pair_dir,
            "source_group_id": stored_group_id,
            "current_group_id": current_group_id,
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
                    f"{param_source}::{rule_policy}",
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
                    f"{off_arm.param_source}::{off_arm.rule_policy}",
                    {
                        "off_arm_id": off_arm.arm_id,
                        "source_pair_dir": source_pair_dir,
                    },
                )
    return {
        "pairs": pairs,
        "baseline_groups": baseline_groups,
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
            source_group_id = _pair_group_id(
                param_source=str(arm.get("param_source") or ""),
                rule_policy=str(arm.get("rule_policy") or ""),
                dl_id=str(dl_id),
                dl_runtime_mode=str(arm.get("dl_runtime_mode") or ""),
            )
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


def resolve_comparison_plan(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> ResolvedComparisonPlan:
    """Resolve preparation, reuse, provenance and period exactly once.

    UI rendering and replay execution consume this finalized plan.  Compatibility
    callers may still use :func:`collect_artifact_status`, which returns a detached
    status dictionary derived from the same plan.
    """

    root = Path(project_root).resolve()
    current = settings or get_strategy_comparison_settings()
    status = collect_preparation_status(project_root=root, settings=current)
    status["config_fingerprint"] = strategy_comparison_fingerprint(
        current,
        artifact_identities=status["artifact_identities"],
    )
    replay_cache = _collect_replay_cache_status(
        root=root,
        settings=current,
        status=status,
    )
    status = _apply_completed_pair_dependency_waivers(
        settings=current,
        status=status,
        replay_cache=replay_cache,
    )
    status = _apply_completed_pair_frozen_score_reuse(
        root=root,
        settings=current,
        status=status,
        replay_cache=replay_cache,
    )
    return ResolvedComparisonPlan.from_status(
        settings=current,
        project_root=root,
        status=status,
    )


def collect_artifact_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    """Compatibility view of the canonical resolved comparison plan."""

    return resolve_comparison_plan(
        project_root=project_root,
        settings=settings,
    ).status_dict()


def render_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
    status: dict[str, Any] | None = None,
) -> str:
    root = Path(project_root).resolve()
    current = settings or get_strategy_comparison_settings()
    current_status = status or collect_artifact_status(
        project_root=root,
        settings=current,
    )
    arm_rows = [
        (
            "ON" if arm.enabled else "OFF",
            arm.arm_id,
            arm.name,
            arm.param_source,
            arm.rule_policy,
            "DL-on" if arm.dl_enabled else "DL-off",
            arm.description,
        )
        for arm in current.enabled_arms
    ]
    contrast_rows = [
        (
            "ON" if item.enabled else "OFF",
            item.contrast_id,
            item.left,
            item.right,
            item.description,
        )
        for item in current.enabled_contrasts
    ]
    artifact_rows = []
    for source_id, row in current_status["parameters"].items():
        artifact_rows.append((f"param:{source_id}", row["status"], row["action"], row["path"]))
        if row.get("identity_manifest_path"):
            artifact_rows.append(
                (
                    f"param:{source_id}:identity",
                    row["identity_status"],
                    row["action"],
                    row["identity_manifest_path"],
                )
            )
    for dl_id, row in current_status["dl_sources"].items():
        for key, file_row in row["files"].items():
            artifact_rows.append(
                (f"dl:{dl_id}:{key}", file_row["status"], file_row["action"], file_row["path"])
            )
    return "\n\n".join(
        (
            render_title("策略績效比較設定與工件狀態"),
            render_key_values(
                (
                    ("設定檔", "config/strategy_compare.py"),
                    ("比較階段", f"{current.profile_label} ({current.profile_id})"),
                    ("Dataset", current.dataset),
                    (
                        "期間",
                        (
                            f"{current_status['comparison_period']['start']} ～ "
                            f"{current_status['comparison_period']['end']}"
                            if current_status.get("comparison_period")
                            else f"{current.start_date or 'artifact start'} ～ {current.end_date or 'artifact end'}"
                        ),
                    ),
                    ("Param policy", current.param_policy),
                    ("Max positions", current.max_positions),
                    ("Rotation", current.rotation),
                    ("Config fingerprint", current_status["config_fingerprint"]),
                    ("比較狀態", current_status["overall_status"]),
                    ("策略比較自動前置", "on" if current.preparation.auto_prepare else "off"),
                    ("模型權重前置", "模型訓練 → 準備策略比較所需模型工件"),
                )
            ),
            render_section("1. 比較對象"),
            render_table(
                ("開關", "編號", "名稱", "參數來源", "Rules", "DL", "用途"),
                arm_rows,
            ),
            render_section("2. 差異比較"),
            render_table(
                ("開關", "比較", "左側", "右側", "用途"),
                contrast_rows,
            ),
            render_section("3. 前置工件與預計動作"),
            render_table(("工件", "狀態", "預計動作", "路徑"), artifact_rows),
        )
    )


def render_execution_plan(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
) -> str:
    plan: StrategyPreparationPlan = status["preparation_plan"]
    ordered_actions = sorted(
        plan.actions,
        key=lambda item: (
            0 if item.artifact_key.startswith("param:") else 1,
            item.artifact_key,
        ),
    )
    rows = [
        (item.action, item.artifact_key, item.description)
        for item in ordered_actions
    ]
    replay_cache = dict(status.get("replay_cache") or {})
    cached_pairs = dict(replay_cache.get("pairs") or {})
    cached_baselines = dict(replay_cache.get("baseline_groups") or {})
    for arm in settings.enabled_arms:
        if arm.dl_enabled:
            action = "REUSE" if cached_pairs.get(arm.arm_id) is not None else "RUN"
            description = (
                f"{arm.name}｜重用已完成且identity一致的正式pair結果"
                if action == "REUSE"
                else arm.name
            )
        else:
            group_key = f"{arm.param_source}::{arm.rule_policy}"
            action = "REUSE" if cached_baselines.get(group_key) is not None else "RUN"
            description = (
                f"{arm.name}｜重用既有正式shared baseline"
                if action == "REUSE"
                else arm.name
            )
        rows.append((action, arm.arm_id, description))
    rows.extend(
        ("REPORT", item.contrast_id, item.description)
        for item in settings.enabled_contrasts
    )
    return "\n\n".join(
        (
            render_title("本次執行計畫"),
            render_key_values(
                (
                    ("整體狀態", plan.overall_status),
                    ("設定檔", "config/strategy_compare.py"),
                    ("比較階段", f"{settings.profile_label} ({settings.profile_id})"),
                    ("Config fingerprint", status["config_fingerprint"]),
                )
            ),
            render_table(("動作", "項目", "說明"), rows),
        )
    )


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any, *, unit: str = "", digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}{unit}"


def _arm_runtime_spec(arm: StrategyComparisonArm) -> dict[str, str]:
    mode = str(arm.dl_runtime_mode or "")
    if mode == STRATEGY_DL_RUNTIME_MODE_HARD_FILTER:
        return {
            "comparison_mode": COMPARISON_MODE_HARD_FILTER,
            "ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
            "active_key": "quality_filter",
            "yearly_key": "quality_filter_return_pct",
            "active_trades_filename": "quality_filter_trades.csv",
        }
    if mode in {
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    }:
        return {
            "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
            "ranking_policy": (
                BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET
                if mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET
                else BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY
            ),
            "active_key": "score_ranking",
            "yearly_key": "score_ranking_return_pct",
            "active_trades_filename": "score_ranking_trades.csv",
        }
    raise ValueError(f"不支援的DL runtime mode: arm={arm.arm_id}, mode={mode!r}")


def _exclusive_selection_r_from_trade_files(
    pair_dir: Path,
    *,
    active_trades_filename: str,
) -> float:
    baseline_path = pair_dir / "no_filter_trades.csv"
    active_path = pair_dir / active_trades_filename
    if not baseline_path.is_file() or not active_path.is_file():
        raise FileNotFoundError(
            "缺少直接選擇R所需交易工件: "
            f"{baseline_path.name}, {active_path.name}"
        )
    baseline = reconstruct_round_trips(
        pd.read_csv(baseline_path, encoding="utf-8-sig"),
        scenario="no_filter",
    )
    active = reconstruct_round_trips(
        pd.read_csv(active_path, encoding="utf-8-sig"),
        scenario="active",
    )
    left = baseline.set_index("match_key", drop=False) if not baseline.empty else baseline
    right = active.set_index("match_key", drop=False) if not active.empty else active
    left_keys = set(left.index.astype(str)) if not left.empty else set()
    right_keys = set(right.index.astype(str)) if not right.empty else set()
    left_only = left.loc[list(sorted(left_keys - right_keys))] if left_keys - right_keys else baseline.iloc[0:0]
    right_only = right.loc[list(sorted(right_keys - left_keys))] if right_keys - left_keys else active.iloc[0:0]
    left_r = pd.to_numeric(left_only.get("r_multiple"), errors="coerce").fillna(0.0).sum() if not left_only.empty else 0.0
    right_r = pd.to_numeric(right_only.get("r_multiple"), errors="coerce").fillna(0.0).sum() if not right_only.empty else 0.0
    return float(right_r - left_r)


def _load_direct_selection_r(
    pair_dir: Path,
    *,
    root: Path,
    active_trades_filename: str,
) -> float:
    path = pair_dir / "trade_attribution.json"
    if path.is_file():
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(
                "交易歸因工件格式無效: "
                + project_relative_display_path(path, project_root=root)
            )
        value = _metric(
            dict(payload.get("r_attribution") or {}),
            "exclusive_selection_delta_r",
        )
        if value is None:
            raise ValueError("交易歸因缺少exclusive_selection_delta_r")
        return float(value)
    return _exclusive_selection_r_from_trade_files(
        pair_dir,
        active_trades_filename=active_trades_filename,
    )


def _execution_pairs(
    settings: StrategyComparisonSettings,
) -> tuple[tuple[str, str, StrategyComparisonArm, StrategyComparisonArm], ...]:
    """Return one replay pair per enabled DL source in config order.

    A ``param_source`` / ``rule_policy`` group owns one shared DL-off baseline
    and may expose multiple DL-on arms.  Each DL-on arm is replayed against the
    same baseline; downstream aggregation verifies that repeated baseline
    summaries and yearly returns remain identical.
    """
    grouped: dict[
        tuple[str, str],
        dict[str, StrategyComparisonArm | list[StrategyComparisonArm] | None],
    ] = {}
    ordered_keys: list[tuple[str, str]] = []
    for arm in settings.enabled_arms:
        key = (arm.param_source, arm.rule_policy)
        if key not in grouped:
            grouped[key] = {"off": None, "on": []}
            ordered_keys.append(key)
        group = grouped[key]
        if not arm.dl_enabled:
            if group["off"] is not None:
                raise ValueError(
                    "啟用比較群組重複定義DL-off基準: "
                    f"{arm.param_source}/{arm.rule_policy}"
                )
            group["off"] = arm
            continue
        on_arms = group["on"]
        if not isinstance(on_arms, list):
            raise TypeError("strategy comparison execution group contract錯誤")
        if not arm.dl_id:
            raise ValueError(f"DL-on arm缺少dl_id: {arm.arm_id}")
        if any(
            existing.dl_id == arm.dl_id
            and existing.dl_runtime_mode == arm.dl_runtime_mode
            for existing in on_arms
        ):
            raise ValueError(
                "啟用比較群組重複定義相同DL source/runtime mode: "
                f"{arm.param_source}/{arm.rule_policy}/{arm.dl_id}/{arm.dl_runtime_mode}"
            )
        on_arms.append(arm)

    pairs: list[
        tuple[str, str, StrategyComparisonArm, StrategyComparisonArm]
    ] = []
    for param_source, rule_policy in ordered_keys:
        group = grouped[(param_source, rule_policy)]
        off_arm = group["off"]
        on_arms = group["on"]
        if not isinstance(off_arm, StrategyComparisonArm) or not isinstance(on_arms, list):
            raise ValueError(
                "啟用比較群組缺少共用DL-off基準: "
                f"{param_source}/{rule_policy}"
            )
        if not on_arms:
            # Standalone DL-off comparator由run_standalone_baseline處理；
            # 不需要為了engine pair contract而保留無研究價值的DL-on arm。
            continue
        for on_arm in on_arms:
            pairs.append((param_source, rule_policy, off_arm, on_arm))
    return tuple(pairs)


def _standalone_baseline_arms(
    settings: StrategyComparisonSettings,
) -> tuple[StrategyComparisonArm, ...]:
    enabled = tuple(settings.enabled_arms)
    output: list[StrategyComparisonArm] = []
    for arm in enabled:
        if arm.dl_enabled:
            continue
        has_enabled_on = any(
            other.dl_enabled
            and other.param_source == arm.param_source
            and other.rule_policy == arm.rule_policy
            for other in enabled
        )
        if not has_enabled_on:
            output.append(arm)
    return tuple(output)


def _standalone_baseline_group_id(arm: StrategyComparisonArm) -> str:
    return f"{arm.param_source}__{arm.rule_policy}__dl_off_baseline"


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
    param_identity = dict(
        (status.get("artifact_identities") or {}).get(
            f"param:{off_arm.param_source}"
        )
        or {}
    )
    expected_param_sha = _artifact_identity_sha(param_identity)
    if not expected_param_sha:
        return None
    all_off = off_arm.rule_policy == "all_off"
    expected = {
        "dataset": settings.dataset,
        "params_file_sha256": expected_param_sha,
        "requested_param_policy": settings.param_policy,
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


def _assert_same_shared_baseline(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    arm_id: str,
) -> None:
    ignored = {
        "arm_id",
        "param_source",
        "rule_policy",
        "dl_enabled",
        "dl_id",
        "dl_runtime_mode",
        "direct_selection_delta_r",
    }
    keys = {
        key
        for key in (set(existing) | set(candidate)) - ignored
        if not str(key).startswith("resource_aware_selector_timing_")
    }
    for key in sorted(keys):
        left = existing.get(key)
        right = candidate.get(key)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(
                    f"多DL比較的共用基準不一致: arm={arm_id}, key={key}, "
                    f"first={left}, repeated={right}"
                )
        elif left != right:
            raise ValueError(
                f"多DL比較的共用基準不一致: arm={arm_id}, key={key}, "
                f"first={left!r}, repeated={right!r}"
            )


def _scenario_payloads(
    pair_payloads: dict[str, dict[str, Any]],
    direct_r: dict[str, float],
    *,
    settings: StrategyComparisonSettings,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    enabled_ids = {arm.arm_id for arm in settings.enabled_arms}
    for group_id, pair in pair_payloads.items():
        param_source, rule_policy, off_arm, on_arm = pair["arm_contract"]
        if off_arm.arm_id in enabled_ids:
            baseline = {
                **dict(pair["payload"].get("no_filter") or {}),
                "arm_id": off_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": False,
                "dl_id": None,
                "dl_runtime_mode": None,
                "direct_selection_delta_r": 0.0,
            }
            existing = output.get(off_arm.arm_id)
            if existing is None:
                output[off_arm.arm_id] = baseline
            else:
                _assert_same_shared_baseline(
                    existing,
                    baseline,
                    arm_id=off_arm.arm_id,
                )
        if on_arm is None:
            continue
        if on_arm.arm_id in enabled_ids:
            runtime_spec = _arm_runtime_spec(on_arm)
            output[on_arm.arm_id] = {
                **dict(pair["payload"].get(runtime_spec["active_key"]) or {}),
                "arm_id": on_arm.arm_id,
                "param_source": param_source,
                "rule_policy": rule_policy,
                "dl_enabled": True,
                "dl_id": on_arm.dl_id,
                "dl_runtime_mode": on_arm.dl_runtime_mode,
                "direct_selection_delta_r": float(direct_r[group_id]),
            }
    return output


def _summary_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for arm in settings.enabled_arms:
        payload = scenarios[arm.arm_id]
        rows.append(
            (
                arm.arm_id,
                arm.name,
                _fmt(payload.get("total_return_pct"), unit="%"),
                _fmt(payload.get("max_drawdown_pct"), unit="%"),
                _fmt(payload.get("return_over_max_drawdown")),
                _fmt(payload.get("annual_return_pct"), unit="%"),
                _fmt(payload.get("expected_value_r"), unit=" R"),
                _fmt(payload.get("payoff_ratio")),
                _fmt(payload.get("avg_exposure_pct"), unit="%"),
                _fmt(payload.get("trade_count"), digits=0),
                _fmt(payload.get("direct_selection_delta_r"), unit=" R"),
            )
        )
    return render_table(
        (
            "編號",
            "比較對象",
            "報酬",
            "MDD",
            "RoMD",
            "年化",
            "EV",
            "Payoff",
            "曝險",
            "交易",
            "同參數DL選擇R",
        ),
        rows,
    )


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = _metric(left, key)
    b = _metric(right, key)
    return None if a is None or b is None else a - b


def _same_param_direct_selection_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float | None:
    """Return DL selection attribution only within one parameter/runtime universe.

    ``direct_selection_delta_r`` is defined against the DL-off baseline that
    shares the same ``param_source`` and ``rule_policy``.  Subtracting values
    across different parameter sources or rule policies mixes two different
    attribution universes and has no controlled physical interpretation.
    """
    if (
        left.get("param_source") != right.get("param_source")
        or left.get("rule_policy") != right.get("rule_policy")
    ):
        return None
    return _delta(left, right, "direct_selection_delta_r")


def _contrast_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for contrast in settings.enabled_contrasts:
        left = scenarios[contrast.left]
        right = scenarios[contrast.right]
        rows.append(
            (
                contrast.contrast_id,
                contrast.description,
                _fmt(_delta(left, right, "total_return_pct"), unit="pp"),
                _fmt(_delta(left, right, "max_drawdown_pct"), unit="pp"),
                _fmt(_delta(left, right, "return_over_max_drawdown")),
                _fmt(_delta(left, right, "annual_return_pct"), unit="pp"),
                _fmt(_delta(left, right, "expected_value_r"), unit=" R"),
                _fmt(_delta(left, right, "avg_exposure_pct"), unit="pp"),
                _fmt(_delta(left, right, "trade_count"), digits=0),
                _fmt(
                    _same_param_direct_selection_delta(left, right),
                    unit=" R",
                ),
            )
        )
    return render_table(
        (
            "比較",
            "用途",
            "Δ報酬",
            "ΔMDD",
            "ΔRoMD",
            "Δ年化",
            "ΔEV",
            "Δ曝險",
            "Δ交易",
            "Δ同參數DL選擇R",
        ),
        rows,
    )


def _fmt_money_milli(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    number = float(value) / 1000.0
    if not math.isfinite(number):
        return "-"
    return f"{number:,.0f}"


def _resource_aware_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode not in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_BINARY_BASKET,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
        }:
            continue
        payload = scenarios[arm.arm_id]
        rows.append((
            arm.arm_id,
            arm.name,
            _fmt(payload.get("resource_aware_dl_selection_days"), digits=0),
            _fmt(payload.get("resource_aware_capital_utilization_days"), digits=0),
            _fmt(payload.get("resource_aware_changed_days"), digits=0),
            _fmt(payload.get("resource_aware_promoted_pass_orders"), digits=0),
            _fmt_money_milli(payload.get("resource_aware_pass_reserved_gain_milli")),
            _fmt(payload.get("resource_aware_promoted_score_orders"), digits=0),
            _fmt(payload.get("resource_aware_selected_score_sum_gain"), digits=3),
            _fmt(payload.get("resource_aware_direct_score_order_days"), digits=0),
            _fmt(payload.get("resource_aware_selected_count_delta"), digits=0),
            _fmt_money_milli(payload.get("resource_aware_reserved_delta_milli")),
            _fmt(payload.get("resource_aware_preservation_violation_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_eligible_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_repair_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_fallback_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_seed_fallback_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_feasible_ascent_days"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_feasible_ascent_local_optimum_days"), digits=0),
            _fmt(payload.get("resource_aware_stale_score_guard_max_age_days"), unit="日", digits=0),
            _fmt(payload.get("resource_aware_stale_score_guard_triggered_days"), digits=0),
            _fmt(payload.get("resource_aware_stale_score_candidate_count"), digits=0),
            _fmt(payload.get("resource_aware_stale_score_guard_blocked_swaps"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_order_count_violation_days"), digits=0),
        ))
    if not rows:
        return "本次沒有啟用Resource-aware arm。"
    return render_table(
        (
            "編號",
            "比較對象",
            "DL選股日",
            "資金利用優先日",
            "實際改單日",
            "新增PASS單",
            "PASS預留資金增量",
            "Continuous新選入單",
            "Selected Score總和增量",
            "直接Score排序可行日",
            "預計選入差",
            "總預留資金增量",
            "資源保護違規日",
            "Max-DL可介入日",
            "Max-DL修復日",
            "Max-DL回退日",
            "Seed原為回退日",
            "Feasible-ascent改善日",
            "1-swap local optimum日",
            "Stale門檻",
            "Stale guard日",
            "Stale候選數",
            "Blocked swaps",
            "Max-DL K違規日",
        ),
        rows,
    )


def _selector_timing_table(
    scenarios: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    rows = []
    for arm in settings.enabled_arms:
        if arm.dl_runtime_mode not in {
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
        }:
            continue
        payload = scenarios[arm.arm_id]
        rows.append((
            arm.arm_id,
            arm.name,
            _fmt(payload.get("resource_aware_selector_timing_total_ms"), unit=" ms", digits=2),
            _fmt(payload.get("resource_aware_selector_timing_median_ms"), unit=" ms", digits=3),
            _fmt(payload.get("resource_aware_selector_timing_p95_ms"), unit=" ms", digits=3),
            _fmt(payload.get("resource_aware_selector_timing_max_ms"), unit=" ms", digits=3),
            _fmt(payload.get("resource_aware_max_dl_repair_evaluations"), digits=0),
            _fmt(payload.get("resource_aware_max_dl_feasible_ascent_evaluations"), digits=0),
            _fmt(payload.get("resource_aware_constrained_search_states"), digits=0),
            _fmt(payload.get("resource_aware_constrained_pruned_states"), digits=0),
            _fmt(payload.get("resource_aware_constrained_optimality_certified_days"), digits=0),
        ))
    if not rows:
        return "本次沒有啟用Max-DL selector arm。"
    return render_table(
        (
            "編號",
            "比較對象",
            "Selector總時間",
            "Median/日",
            "P95/日",
            "Max/日",
            "C17 repair eval",
            "Feasible-ascent eval",
            "Exact states",
            "Exact pruned",
            "Exact certified日",
        ),
        rows,
    )


def _yearly_table(
    pair_payloads: dict[str, dict[str, Any]],
    *,
    settings: StrategyComparisonSettings,
) -> str:
    enabled_ids = tuple(arm.arm_id for arm in settings.enabled_arms)
    by_id: dict[str, dict[int, float | None]] = {arm_id: {} for arm_id in enabled_ids}
    for pair in pair_payloads.values():
        _param_source, _rule_policy, off_arm, on_arm = pair["arm_contract"]
        for row in pair["payload"].get("yearly") or []:
            year = int(row["year"])
            if off_arm.arm_id in by_id:
                value = row.get("no_filter_return_pct")
                existing = by_id[off_arm.arm_id].get(year)
                if existing is not None and value is not None:
                    if not math.isclose(
                        float(existing), float(value), rel_tol=0.0, abs_tol=1e-10
                    ):
                        raise ValueError(
                            "多DL比較的共用基準年度報酬不一致: "
                            f"arm={off_arm.arm_id}, year={year}, "
                            f"first={existing}, repeated={value}"
                        )
                else:
                    by_id[off_arm.arm_id][year] = value
            if on_arm is not None and on_arm.arm_id in by_id:
                runtime_spec = _arm_runtime_spec(on_arm)
                by_id[on_arm.arm_id][year] = row.get(runtime_spec["yearly_key"])
    years = sorted({year for values in by_id.values() for year in values})
    rows = [
        (
            year,
            *(_fmt(by_id[arm_id].get(year), unit="%") for arm_id in enabled_ids),
        )
        for year in years
    ]
    return render_table(("年度", *enabled_ids), rows)


def _comparison_period(pair_payloads: dict[str, dict[str, Any]]) -> Any:
    periods = {
        json.dumps(
            dict(pair["payload"].get("metadata") or {}).get("comparison_period"),
            sort_keys=True,
        )
        for pair in pair_payloads.values()
    }
    if len(periods) != 1:
        raise ValueError(f"策略比較期間不一致: {periods}")
    return dict(next(iter(pair_payloads.values()))["payload"].get("metadata") or {}).get(
        "comparison_period"
    )


def _render_report(
    *,
    settings: StrategyComparisonSettings,
    status: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    pair_payloads: dict[str, dict[str, Any]],
) -> str:
    return "\n\n".join(
        (
            render_title("策略績效比較"),
            render_key_values(
                (
                    ("期間", _comparison_period(pair_payloads)),
                    ("Dataset", settings.dataset),
                    ("Param policy", settings.param_policy),
                    ("Max positions", settings.max_positions),
                    ("Rotation", settings.rotation),
                    ("Config fingerprint", status["config_fingerprint"]),
                    ("比較設定", "config/strategy_compare.py"),
                )
            ),
            render_section("1. 比較結果"),
            _summary_table(scenarios, settings=settings),
            render_section("2. 設定中的差異比較"),
            _contrast_table(scenarios, settings=settings),
            render_section("3. 年度結果"),
            _yearly_table(pair_payloads, settings=settings),
            render_section("4. Resource-aware盤前診斷"),
            _resource_aware_table(scenarios, settings=settings),
            render_section("5. Max-DL Selector計算時間"),
            _selector_timing_table(scenarios, settings=settings),
            render_section("6. 判讀原則"),
            (
                "以config中啟用的contrast逐項判讀；不得用單一年份改善取代"
                "全期RoMD、EV、同參數DL選擇R與年度穩定性。同參數DL選擇R只可在"
                "param_source與rule_policy皆相同的arms之間比較；跨參數contrast固定顯示-。"
                "比較流程不建立Label、不選模型也不訓練模型權重；可依config透過正式"
                "共用服務補建既有模型的forward-OOS scores與比較所需策略參數工件。"
                "Resource-aware Binary與舊Continuous沿用各自資源Gate；Max-DL Continuous則以Min ROOS"
                "預留單數與reserved-capital floor作硬限制，合法範圍內只最大化frozen DL score；"
                "Feasible-ascent只在相同K/R0合法集合內做best-improvement single-swap，不引入capital objective；"
                "不得新增資金利用Threshold。Binary arm看PASS資源配置，Continuous arm看selected score改善；各者都必須同時檢查"
                "總曝險、預留資金與策略績效，不能只看模型分數。"
            ),
        )
    ).rstrip() + "\n"


def _run_directory(
    *,
    root: Path,
    settings: StrategyComparisonSettings,
    fingerprint: str,
) -> tuple[Path, Path]:
    output_root = _resolve_relative_path(root, settings.output_root)
    enabled_ids = "-".join(arm.arm_id for arm in settings.enabled_arms)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"{timestamp}_{enabled_ids}_{fingerprint}"
    latest_dir = output_root / "latest"
    return run_dir, latest_dir


def run_strategy_comparison(
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
    status: dict[str, Any] | None = None,
    resolved_plan: ResolvedComparisonPlan | None = None,
    auto_prepare: bool = True,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = settings or get_strategy_comparison_settings()
    if status is not None and resolved_plan is not None:
        raise ValueError("status與resolved_plan不可同時提供")
    if resolved_plan is None:
        if status is None:
            resolved_plan = resolve_comparison_plan(project_root=root, settings=settings)
        else:
            resolved_plan = ResolvedComparisonPlan.from_status(
                settings=settings,
                project_root=root,
                status=status,
            )
    elif resolved_plan.settings.as_dict() != settings.as_dict():
        raise ValueError("resolved_plan與settings不一致")
    status = resolved_plan.status_dict()
    requested_fingerprint = resolved_plan.config_fingerprint
    requested_plan = resolved_plan.preparation_plan
    if status["overall_status"] == "BLOCKED":
        print("\n" + render_status(project_root=root, settings=settings, status=status))
        raise RuntimeError(
            "目前啟用比較缺少不可自動產生的上游工件；請依狀態頁使用正式模型入口處理。"
        )
    if status["overall_status"] == "PREPARABLE":
        if not auto_prepare:
            raise RuntimeError("目前工件可自動準備，但本次已停用auto_prepare")
        prepare_strategy_comparison_artifacts(
            project_root=root,
            settings=settings,
            status=status,
            status_refresher=lambda: collect_artifact_status(
                project_root=root, settings=settings
            ),
        )
        resolved_plan = resolve_comparison_plan(
            project_root=root,
            settings=settings,
        )
        status = resolved_plan.status_dict()
        status["requested_preparation_plan"] = requested_plan
        if resolved_plan.overall_status != "READY":
            raise RuntimeError("前置完成後正式比較狀態仍非READY")

    # READY has already passed ResolvedComparisonPlan.validate_contract().
    # Replay must consume this exact finalized plan instead of recollecting state.
    comparison_period = dict(status.get("comparison_period") or {})
    comparison_start = str(comparison_period.get("start") or "")
    comparison_end = str(comparison_period.get("end") or "")
    if not comparison_start or not comparison_end:
        raise RuntimeError("正式比較缺少已解析的共同comparison period")

    replay_cache = dict(status.get("replay_cache") or {})
    if not replay_cache:
        raise RuntimeError("ResolvedComparisonPlan缺少replay cache")

    run_dir, latest_dir = _run_directory(
        root=root,
        settings=settings,
        fingerprint=status["config_fingerprint"],
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    pair_payloads: dict[str, dict[str, Any]] = {}
    direct_r: dict[str, float] = {}
    pair_execution: dict[str, dict[str, Any]] = {}
    cached_pairs = dict(replay_cache.get("pairs") or {})
    cached_baseline_groups = dict(replay_cache.get("baseline_groups") or {})
    baseline_sources: dict[str, Path] = {
        str(group_key): Path(str(row["source_pair_dir"])).resolve()
        for group_key, row in cached_baseline_groups.items()
        if isinstance(row, dict) and row.get("source_pair_dir")
    }

    for off_arm in _standalone_baseline_arms(settings):
        group_key = f"{off_arm.param_source}::{off_arm.rule_policy}"
        group_id = _standalone_baseline_group_id(off_arm)
        pair_dir = run_dir / "pairs" / group_id
        baseline_reuse_source = (
            baseline_sources.get(group_key)
            if settings.preparation.reuse_shared_baseline
            else None
        )
        all_off = off_arm.rule_policy == "all_off"
        baseline_payload = run_standalone_baseline(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_parameter_paths"][off_arm.param_source]),
            param_policy=settings.param_policy,
            max_positions=settings.max_positions,
            enable_rotation=settings.rotation == "on",
            optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off
                else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            output_dir_override=pair_dir,
            comparison_start_date=comparison_start,
            comparison_end_date=comparison_end,
            quiet=quiet,
            shared_param_overrides=(
                ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None
            ),
            baseline_reuse_dir=baseline_reuse_source,
        )
        pair_payloads[group_id] = {
            "arm_contract": (
                off_arm.param_source, off_arm.rule_policy, off_arm, None
            ),
            "payload": baseline_payload,
        }
        pair_execution[off_arm.arm_id] = {
            "action": "REUSE" if baseline_reuse_source is not None else "RUN",
            "source_pair_dir": (
                None
                if baseline_reuse_source is None
                else project_relative_display_path(
                    baseline_reuse_source, project_root=root
                )
            ),
            "current_pair_dir": project_relative_display_path(
                pair_dir, project_root=root
            ),
            "standalone_dl_off_baseline": True,
        }
        baseline_sources[group_key] = pair_dir

    for param_source, rule_policy, off_arm, on_arm in _execution_pairs(settings):
        if not on_arm.dl_id:
            raise ValueError(f"DL-on arm缺少dl_id: {on_arm.arm_id}")
        dl: StrategyDLSource = settings.dl_sources[on_arm.dl_id]
        runtime_spec = _arm_runtime_spec(on_arm)
        group_id = _pair_group_id(
            param_source=param_source,
            rule_policy=rule_policy,
            dl_id=on_arm.dl_id,
            dl_runtime_mode=str(on_arm.dl_runtime_mode or ""),
        )
        pair_dir = run_dir / "pairs" / group_id
        baseline_group_key = f"{param_source}::{rule_policy}"
        cache_entry = cached_pairs.get(on_arm.arm_id)

        if isinstance(cache_entry, dict) and cache_entry.get("source_pair_dir"):
            source_pair_dir = Path(str(cache_entry["source_pair_dir"])).resolve()
            if pair_dir.exists():
                shutil.rmtree(pair_dir)
            shutil.copytree(source_pair_dir, pair_dir)
            pair_payload = _read_json(pair_dir / "strategy_comparison.json")
            if not isinstance(pair_payload, dict):
                raise RuntimeError(
                    "已命中Strategy Compare cache但pair JSON無法讀取: "
                    + project_relative_display_path(pair_dir, project_root=root)
                )
            pair_metadata = dict(pair_payload.get("metadata") or {})
            pair_metadata.update({
                "output_scope": "reused_pair_cache",
                "output_dir": project_relative_display_path(
                    pair_dir, project_root=root
                ),
                "cache_reused_from": project_relative_display_path(
                    source_pair_dir, project_root=root
                ),
            })
            pair_payload["metadata"] = pair_metadata
            _write_json(pair_dir / "strategy_comparison.json", pair_payload)
            materialize_strategy_pair_readable_report(
                pair_payload,
                output_dir=pair_dir,
            )
            pair_payloads[group_id] = {
                "arm_contract": (param_source, rule_policy, off_arm, on_arm),
                "payload": pair_payload,
            }
            direct_r[group_id] = _load_direct_selection_r(
                pair_dir,
                root=root,
                active_trades_filename=runtime_spec["active_trades_filename"],
            )
            baseline_sources[baseline_group_key] = pair_dir
            pair_execution[on_arm.arm_id] = {
                "action": "REUSE",
                "pair_fingerprint": cache_entry.get("fingerprint"),
                "source_pair_dir": project_relative_display_path(
                    source_pair_dir, project_root=root
                ),
                "current_pair_dir": project_relative_display_path(
                    pair_dir, project_root=root
                ),
                "shared_baseline_reused": True,
                "source_artifact_mode": cache_entry.get(
                    "source_artifact_mode", "current_artifacts"
                ),
            }
            if not quiet:
                print(
                    f"[{on_arm.arm_id}] REUSE 已完成正式pair："
                    + project_relative_display_path(
                        source_pair_dir,
                        project_root=root,
                    )
                )
                print("\n" + render_strategy_pair_simple_report(pair_payload))
                print_artifact_paths(
                    (("策略比較簡易報表", pair_dir / "strategy_comparison.md"),),
                    project_root=root,
                )
            continue

        all_off = rule_policy == "all_off"
        baseline_reuse_source = (
            baseline_sources.get(baseline_group_key)
            if settings.preparation.reuse_shared_baseline
            and runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else None
        )
        pair_payload = run_comparison(
            project_root=root,
            dataset=settings.dataset,
            params_path=str(status["resolved_parameter_paths"][param_source]),
            param_policy=settings.param_policy,
            max_positions=settings.max_positions,
            enable_rotation=settings.rotation == "on",
            fixed_risk=None,
            max_position_cap_pct=None,
            comparison_mode=runtime_spec["comparison_mode"],
            ranking_policy=runtime_spec["ranking_policy"],
            ranking_options=(
                {
                    **dict(on_arm.dl_runtime_options or {}),
                    **(
                        {
                            "expected_r_calibration_path": str(
                                (status.get("expected_r_calibrations") or {})[on_arm.arm_id]["lookup_path"]
                            )
                        }
                        if on_arm.dl_runtime_mode == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
                        else {
                            "expected_excess_r_calibration_path": str(
                                (status.get("expected_excess_r_calibrations") or {})[on_arm.arm_id]["lookup_path"]
                            )
                        }
                        if on_arm.dl_runtime_mode in {
                            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
                            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
                            STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
                        }
                        else {}
                    ),
                }
            ),
            optional_entry_filter_policy=(
                OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                if all_off
                else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
            ),
            filter_id=dl.filter_id,
            score_source=dl.score_source,
            model_architecture=dl.model_architecture,
            experiment_profile=dl.experiment_profile,
            threshold=dl.threshold,
            output_dir_override=pair_dir,
            comparison_start_date=comparison_start,
            comparison_end_date=comparison_end,
            quiet=quiet,
            shared_param_overrides=(
                ALL_RULE_FILTERS_OFF_OVERRIDES if all_off else None
            ),
            baseline_reuse_dir=baseline_reuse_source,
            continuous_score_path_override=(
                str(
                    (status.get("continuous_score_overrides") or {})[on_arm.dl_id][
                        "score_path"
                    ]
                )
                if (
                    dl.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
                    and on_arm.dl_id in (status.get("continuous_score_overrides") or {})
                )
                else None
            ),
            continuous_score_execution_start_override=(
                str(
                    (status.get("continuous_score_overrides") or {})[on_arm.dl_id][
                        "execution_start"
                    ]
                )
                if (
                    dl.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
                    and on_arm.dl_id in (status.get("continuous_score_overrides") or {})
                )
                else None
            ),
            capture_execution_diagnostics=(
                runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            ),
        )
        pair_payloads[group_id] = {
            "arm_contract": (param_source, rule_policy, off_arm, on_arm),
            "payload": pair_payload,
        }
        direct_r[group_id] = _load_direct_selection_r(
            pair_dir,
            root=root,
            active_trades_filename=runtime_spec["active_trades_filename"],
        )
        baseline_sources[baseline_group_key] = pair_dir
        pair_execution[on_arm.arm_id] = {
            "action": "RUN",
            "pair_fingerprint": _current_pair_cache_fingerprint(
                settings=settings,
                status=status,
                off_arm=off_arm,
                on_arm=on_arm,
            ),
            "source_pair_dir": None,
            "current_pair_dir": project_relative_display_path(
                pair_dir, project_root=root
            ),
            "shared_baseline_reused": baseline_reuse_source is not None,
            "shared_baseline_source": (
                None
                if baseline_reuse_source is None
                else project_relative_display_path(
                    baseline_reuse_source,
                    project_root=root,
                )
            ),
        }

    scenarios = _scenario_payloads(pair_payloads, direct_r, settings=settings)
    report = _render_report(
        settings=settings,
        status=status,
        scenarios=scenarios,
        pair_payloads=pair_payloads,
    )
    report_path = run_dir / "strategy_comparison.md"
    json_path = run_dir / "strategy_comparison.json"
    manifest_path = run_dir / "manifest.json"
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "COMPLETED",
        "created_at": get_taipei_now().isoformat(),
        "config_fingerprint": status["config_fingerprint"],
        "requested_config_fingerprint": requested_fingerprint,
        "settings": settings.as_dict(),
        "artifact_identities": status["artifact_identities"],
        "comparison_period": comparison_period,
        "comparison_period_source": status.get("comparison_period_source"),
        "requested_preparation_plan": requested_plan.as_dict(),
        "final_preparation_plan": status["preparation_plan"].as_dict(),
        "scenarios": scenarios,
        "contrasts": {
            item.contrast_id: {
                "left": item.left,
                "right": item.right,
                "description": item.description,
            }
            for item in settings.enabled_contrasts
        },
        "pairs": {
            key: value["payload"] for key, value in pair_payloads.items()
        },
        "pair_execution": pair_execution,
        "replay_reuse_policy": {
            "reuse_completed_results": bool(
                settings.preparation.reuse_completed_results
            ),
            "reuse_shared_baseline": bool(
                settings.preparation.reuse_shared_baseline
            ),
        },
    }
    _write_json(json_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "created_at": payload["created_at"],
            "config_path": "config/strategy_compare.py",
            "config_fingerprint": status["config_fingerprint"],
            "requested_config_fingerprint": requested_fingerprint,
            "enabled_arms": [arm.as_dict() for arm in settings.enabled_arms],
            "enabled_contrasts": [
                item.as_dict() for item in settings.enabled_contrasts
            ],
            "artifact_identities": status["artifact_identities"],
            "comparison_period": comparison_period,
            "comparison_period_source": status.get("comparison_period_source"),
            "preparation_policy": settings.preparation.as_dict(),
            "requested_preparation_plan": requested_plan.as_dict(),
            "final_preparation_plan": status["preparation_plan"].as_dict(),
            "pair_execution": pair_execution,
            "replay_reuse_policy": payload["replay_reuse_policy"],
            "run_dir": project_relative_display_path(run_dir, project_root=root),
        },
    )

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source, filename in (
        (report_path, "strategy_comparison.md"),
        (json_path, "strategy_comparison.json"),
        (manifest_path, "manifest.json"),
    ):
        shutil.copy2(source, latest_dir / filename)

    print("\n" + report)
    print_artifact_paths(
        (
            ("策略比較Markdown", report_path),
            ("策略比較JSON", json_path),
            ("執行Manifest", manifest_path),
            ("最新結果", latest_dir),
        ),
        project_root=root,
    )
    return payload


def show_strategy_comparison_status(
    *,
    project_root: Path = PROJECT_ROOT,
    settings: StrategyComparisonSettings | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    settings = settings or get_strategy_comparison_settings()
    status = collect_artifact_status(project_root=root, settings=settings)
    print("\n" + render_status(project_root=root, settings=settings, status=status))
    return status


__all__ = [
    "ResolvedComparisonPlan",
    "resolve_comparison_plan",
    "collect_artifact_status",
    "render_execution_plan",
    "render_status",
    "run_strategy_comparison",
    "show_strategy_comparison_status",
]
