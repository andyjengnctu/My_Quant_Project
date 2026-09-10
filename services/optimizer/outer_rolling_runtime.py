"""Outer-rolling runtime identity, resume, and failure guards.

This module owns runtime-context reproducibility checks, study identity
compatibility, resume DB identity/path resolution, and compact failure helpers.
It does not own fold construction, selection, replay, or scientific identity.
"""

from __future__ import annotations

import os

from core.runtime_utils import resolve_environment_flag as _env_flag


def _tail_text_file(path: str, *, max_lines: int = 8) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-int(max_lines):]).strip()
    except OSError:
        return ""


def _format_exception_summary(exc: BaseException) -> str:
    if exc is None:
        return "unknown"
    message = str(exc).strip()
    if "\n最後 fold log：" in message:
        message = message.split("\n最後 fold log：", 1)[0].strip()
    message = " ".join(message.split())
    if len(message) > 600:
        message = message[:597] + "..."
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def _optimizer_runtime_context_spec(session_spec: dict | None) -> dict:
    return dict(dict(session_spec or {}).get("runtime_context_spec") or {})


def _validate_optimizer_runtime_context(session, session_spec: dict | None) -> None:
    """Fail before search when a serialized runtime context cannot be reproduced."""
    spec = _optimizer_runtime_context_spec(session_spec)
    if not spec:
        return
    module_name = str(spec.get("module") or "")
    callable_name = str(spec.get("callable") or "")
    kwargs = dict(spec.get("kwargs") or {})
    try:
        with session.optimizer_runtime_context():
            if (
                module_name == "filters.breakout_quality.runtime"
                and callable_name == "breakout_quality_ranking_source_context"
            ):
                from filters.breakout_quality.runtime import (
                    get_breakout_quality_ranking_source_context,
                )

                actual = get_breakout_quality_ranking_source_context()
                expected = {
                    "score_source": str(kwargs.get("score_source") or ""),
                    "model_architecture": (
                        None
                        if kwargs.get("model_architecture") is None
                        else str(kwargs.get("model_architecture"))
                    ),
                    "experiment_profile": (
                        None
                        if kwargs.get("experiment_profile") is None
                        else str(kwargs.get("experiment_profile"))
                    ),
                }
                actual_payload = {
                    "score_source": str(actual.score_source),
                    "model_architecture": actual.model_architecture,
                    "experiment_profile": actual.experiment_profile,
                }
                if actual_payload != expected:
                    raise RuntimeError(
                        "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR: optimizer runtime context不一致："
                        f"actual={actual_payload}, expected={expected}"
                    )
            elif (
                module_name == "filters.breakout_quality.runtime"
                and callable_name == "breakout_quality_filter_source_context"
            ):
                from filters.breakout_quality.runtime import (
                    get_breakout_quality_filter_source_context,
                )

                actual = get_breakout_quality_filter_source_context()
                expected = {
                    "score_source": str(kwargs.get("score_source") or ""),
                    "manifest_path": str(kwargs.get("manifest_path") or ""),
                    "scores_path": str(kwargs.get("scores_path") or ""),
                }
                actual_payload = {
                    "score_source": str(actual.score_source),
                    "manifest_path": str(actual.manifest_path or ""),
                    "scores_path": str(actual.scores_path or ""),
                }
                if actual_payload != expected:
                    raise RuntimeError(
                        "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR: optimizer Binary PIT runtime context不一致："
                        f"actual={actual_payload}, expected={expected}"
                    )
    except Exception as exc:
        text = str(exc)
        if "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR" in text:
            raise
        raise RuntimeError(
            "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR: optimizer runtime context無法建立："
            f"module={module_name}, callable={callable_name}, error={type(exc).__name__}: {exc}"
        ) from exc


def _is_non_retryable_fold_failure(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    current = exc
    parts: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    text = " ".join(parts).lower()
    markers = (
        "non_retryable_runtime_identity_error",
        "找不到 breakout quality 正式 manifest",
        "breakout quality manifest experiment_profile",
        "找不到selection pit score",
        "找不到selection pit manifest",
        "找不到selection pit audit",
        "selection pit runtime identity不一致",
        "binary pit",
        "binary_point_in_time",
        "binary point-in-time",
        "同一 ticker／score_date 的 ensemble members score availability不一致",
        "同一 ticker 的 ensemble members score availability不一致",
        "同一 ticker 的 ensemble members score source不一致",
        "同一 ticker 的 ensemble members quality ranking 設定不一致",
        "同日 aggregated candidates 的 quality ranking 設定不一致",
        "啟用 quality ranking 的 ensemble 候選分數不是有限數值",
        "啟用 quality ranking 的 ensemble 候選缺少有效原始事件分數",
        "runtime context不一致",
        "runtime identity",
    )
    return any(marker.lower() in text for marker in markers)


def _outer_rolling_db_member_suffix(task: dict | None) -> str:
    payload = dict(task or {})
    if not bool(payload.get("seed_ensemble_member")):
        return ""
    member_index = int(payload.get("seed_ensemble_member_index", 0) or 0)
    seed = payload.get("optimizer_seed")
    seed_text = "none" if seed is None else str(int(seed))
    return f"_seed{member_index:02d}_{seed_text}"


def _outer_rolling_runtime_identity(task: dict | None) -> str:
    payload = dict(task or {})
    direct = str(payload.get("runtime_cache_identity") or "").strip()
    if direct:
        return direct
    session_spec = dict(payload.get("optimizer_session_spec") or {})
    return str(session_spec.get("runtime_cache_identity") or "").strip()


def _resolve_outer_rolling_db_file(
    *, output_dir: str, session_ts: str, oos_year: int, task: dict | None, environ
) -> tuple[str, bool]:
    db_dir = os.path.join(output_dir, "outer_rolling_oos", "db")
    os.makedirs(db_dir, exist_ok=True)
    member_suffix = _outer_rolling_db_member_suffix(task)
    resume_enabled = _env_flag(
        environ, "OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES", False
    )
    runtime_identity = _outer_rolling_runtime_identity(task)
    if resume_enabled and runtime_identity:
        stable_name = (
            f"outer_oos_runtime_{runtime_identity[:20]}_{int(oos_year)}"
            f"{member_suffix}.db"
        )
        stable_path = os.path.join(db_dir, stable_name)
        return stable_path, os.path.isfile(stable_path)
    new_path = os.path.join(
        db_dir, f"outer_oos_{session_ts}_{int(oos_year)}{member_suffix}.db"
    )
    return new_path, False


def _ensure_study_runtime_identity_compatible(study, session) -> None:
    if not hasattr(study, "user_attrs") or not hasattr(study, "set_user_attr"):
        return
    current_identity = str(getattr(session, "runtime_cache_identity", "") or "").strip()
    if not current_identity:
        return
    key = "optimizer_runtime_cache_identity"
    current_overrides = dict(
        getattr(session, "fixed_strategy_param_overrides", {}) or {}
    )
    existing_identity = str(
        dict(getattr(study, "user_attrs", {}) or {}).get(key) or ""
    ).strip()
    trials = list(getattr(study, "trials", []) or [])
    if existing_identity and existing_identity != current_identity and trials:
        raise RuntimeError(
            "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR: Optimizer study runtime identity不一致，"
            f"existing={existing_identity}, current={current_identity}"
        )
    if not existing_identity and trials:
        for trial in trials:
            trial_overrides = dict(
                (getattr(trial, "user_attrs", {}) or {}).get(
                    "fixed_strategy_param_overrides", {}
                )
                or {}
            )
            if trial_overrides != current_overrides:
                raise RuntimeError(
                    "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR: 既有Optimizer study缺少runtime identity，"
                    "且trial固定策略契約與目前設定不一致，禁止接續。"
                )
    study.set_user_attr(key, current_identity)
    study.set_user_attr(
        "optimizer_fixed_strategy_param_overrides", current_overrides
    )


def _remaining_optimizer_trials(study, requested_trials: int) -> tuple[int, int]:
    existing = len(list(getattr(study, "trials", []) or []))
    return existing, max(0, int(requested_trials) - int(existing))


def _apply_outer_rolling_process_environ(environ) -> None:
    """Expose explicit rolling runtime settings to spawned fold workers."""
    for raw_key, raw_value in dict(environ or {}).items():
        key = str(raw_key)
        if key == "V16_MODELS_DIR" or key.startswith("OPTIMIZER_"):
            os.environ[key] = str(raw_value)
