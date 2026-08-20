import inspect
import json
import math
import os
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_profiles import (
    DEFAULT_DATASET_PROFILE,
    normalize_dataset_profile_key,
    get_dataset_dir,
    get_dataset_profile_label,
    resolve_dataset_profile_from_cli_env,
    build_missing_dataset_dir_message,
    build_empty_dataset_dir_message,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_strategy_dashboard
from core.model_paths import resolve_models_dir, resolve_run_best_params_path
from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, get_process_pool_executor_kwargs, get_taipei_now, has_help_flag, resolve_cli_program_name, validate_cli_args, is_interactive_console, safe_prompt_choice, stdout_supports_inline_progress, write_inline_progress
from core.output_paths import build_output_dir
from core.walk_forward_policy import (
    build_optimizer_effective_policy_fingerprint,
    build_optimizer_runtime_policy,
    load_walk_forward_policy,
    normalize_optimizer_model_mode,
    normalize_optimizer_study_scope,
)
from core.active_param_ensemble import build_static_active_param_ensemble_payload, is_active_param_ensemble_payload
from core.raw_universe_contract import RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD, resolve_raw_universe_required_min_rows
from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot, generate_random_seed_ensemble, renumber_seed_ensemble_members
from config.training_policy import (
    DEFAULT_OPTIMIZER_MODEL_MODE,
    OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_FIXED_TP_PERCENT,
    OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED,
    OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE,
    OPTIMIZER_INNER_VALIDATE_MIN_SCORE,
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    TRADE_MODE_AUTO_PROMOTE_RUN_BEST,
    TRADE_MODE_CANDIDATE_SELECTOR,
    TRADE_MODE_RUN_BEST_SELECTOR,
    TRADE_PROMOTE_MIN_SCORE_DELTA,
    OPTIMIZER_PERSIST_STUDY_DB,
    is_optimizer_local_min_review_enabled,
    set_optimizer_runtime_model_mode,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE,
    OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
    OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
    OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
    resolve_optimizer_base_finalists_agree_min_agree,
    resolve_optimizer_local_finalists_agree_min_agree,
    resolve_optimizer_retention_finalists_agree_min_agree,
)
from config.execution_policy import DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION
from config.training_policy import OPTIMIZER_RANDOM_SEED_DEFAULT


from config.training_performance_policy import resolve_optimizer_random_seed_ensemble_parallel_backend_default, resolve_optimizer_random_seed_ensemble_parallel_workers_default

from tools.optimizer.study_utils import INVALID_TRIAL_VALUE
from tools.optimizer.score_display import format_optimizer_score_for_display
from tools.optimizer.session_factory import (
    build_optimizer_session,
    configure_optuna_logging,
    ensure_study_effective_policy_compatible as _ensure_study_effective_policy_compatible,
)

warnings.simplefilter("default")
warnings.filterwarnings("once", category=FutureWarning, module=r"optuna(\..*)?$")
warnings.filterwarnings("once", category=RuntimeWarning)



def _print_profile_summary_compatible(profile_recorder, *, emit_console: bool):
    print_summary = getattr(profile_recorder, "print_summary", None)
    if print_summary is None:
        return
    try:
        signature = inspect.signature(print_summary)
    except (TypeError, ValueError):
        print_summary()
        return

    params = signature.parameters.values()
    supports_emit_console = (
        "emit_console" in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    )
    if supports_emit_console:
        print_summary(emit_console=emit_console)
        return
    print_summary()

OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "ml_optimizer")
MODELS_DIR = resolve_models_dir(PROJECT_ROOT)
RUN_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "run_best_params.json")
CANDIDATE_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_best_params.json")
CANDIDATE_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_best_summary.json")
CANDIDATE_RETENTION_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_retention_best_params.json")
CANDIDATE_RETENTION_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_retention_best_summary.json")
CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "candidate_val_score_best_params.json")
CANDIDATE_VAL_SCORE_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "candidate_val_score_best_summary.json")
RUN_BEST_SUMMARY_PATH = os.path.join(MODELS_DIR, "run_best_summary.json")
DEFAULT_WALK_FORWARD_POLICY = load_walk_forward_policy(PROJECT_ROOT)
TRAIN_MAX_POSITIONS = DEFAULT_PORTFOLIO_MAX_POSITIONS
TRAIN_ENABLE_ROTATION = DEFAULT_PORTFOLIO_ROTATION == "on"
DEFAULT_OPTIMIZER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1))) if os.name == "nt" else min(6, max(1, (os.cpu_count() or 1) // 2))
ENABLE_OPTIMIZER_PROFILING = True
ENABLE_PROFILE_CONSOLE_PRINT = False
PROFILE_PRINT_EVERY_N_TRIALS = 1
START_BANNER_POLICY_LABEL = "Train/Test policy:"

COLORS = {
    "cyan": C_CYAN,
    "gray": C_GRAY,
    "green": C_GREEN,
    "red": C_RED,
    "reset": C_RESET,
    "yellow": C_YELLOW,
}


def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CANDIDATE_BEST_PARAMS_PATH), exist_ok=True)


def _write_json_file(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def _load_json_file_or_none(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _project_relative_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    try:
        return os.path.relpath(raw, PROJECT_ROOT)
    except ValueError:
        return os.path.basename(raw)


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalize_trade_selector(selector: str) -> str:
    raw = str(selector or "").strip().lower()
    aliases = {
        "base_best": "base_finalist_best",
        "base_finalist_best": "base_finalist_best",
        "local_best": "local_finalist_best",
        "local_finalist_best": "local_finalist_best",
        "retention_best": "retention_finalist_best",
        "retention_finalist_best": "retention_finalist_best",
        "base_agree": "base_finalists_agree",
        "finalists_agree": "base_finalists_agree",
        "base_finalist_agree": "base_finalists_agree",
        "base_finalists_agree": "base_finalists_agree",
        "local_agree": "local_finalists_agree",
        "local_finalist_agree": "local_finalists_agree",
        "local_finalists_agree": "local_finalists_agree",
        "retention_agree": "retention_finalists_agree",
        "retention_finalist_agree": "retention_finalists_agree",
        "retention_finalists_agree": "retention_finalists_agree",
    }
    return aliases.get(raw, raw or "retention")


def _selector_score_from_summary(summary: dict | None, selector: str):
    if not isinstance(summary, dict):
        return None
    normalized = _normalize_trade_selector(selector)
    if normalized in {"retention", "retention_finalists_agree", "retention_finalist_best"}:
        return summary.get("retention")
    if normalized in {"local", "local_finalists_agree", "local_finalist_best"}:
        return summary.get("local_min_score")
    return summary.get("base_score")


def _selector_score_is_valid(summary: dict | None, selector: str) -> bool:
    score = _selector_score_from_summary(summary, selector)
    return _is_finite_number(score) and float(score) != -9999.0


def _resolve_trade_candidate_selector() -> str:
    return _normalize_trade_selector(TRADE_MODE_CANDIDATE_SELECTOR)


def _resolve_trade_run_best_selector() -> str:
    return _normalize_trade_selector(TRADE_MODE_RUN_BEST_SELECTOR)


def _policy_fingerprint_from_summary(summary: dict | None) -> str:
    if not isinstance(summary, dict):
        return ""
    return str(summary.get("policy_fingerprint_sha256") or "").strip()


def _resolve_latest_dataset_date(data_dir: str):
    import pandas as pd
    from core.data_utils import discover_unique_csv_inputs

    latest = None
    csv_inputs, _duplicate_issue_lines = discover_unique_csv_inputs(data_dir)
    for csv_entry in csv_inputs:
        if isinstance(csv_entry, (tuple, list)) and len(csv_entry) >= 2:
            _ticker, csv_path = csv_entry[0], csv_entry[1]
        else:
            _ticker, csv_path = "", csv_entry
        try:
            df = pd.read_csv(csv_path, usecols=lambda col: str(col).strip().lower() in {"date", "time"})
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        columns = list(df.columns)
        if not columns:
            continue
        series = pd.to_datetime(df[columns[0]], errors="coerce").dropna()
        if series.empty:
            continue
        value = series.max().normalize()
        if latest is None or value > latest:
            latest = value
    if latest is None:
        raise ValueError("Trade mode 無法從資料集 CSV 解析最新交易日，請確認資料包含 Date/Time 欄位。")
    return latest.strftime("%Y-%m-%d")


def _embed_summary_in_params_file(params_path: str, summary: dict, *, remove_summary_sidecar: str = "") -> dict:
    payload = _load_json_file_or_none(params_path)
    if not isinstance(payload, dict):
        payload = {}
    payload["summary"] = dict(summary or {})
    _write_json_file(params_path, payload)
    if remove_summary_sidecar:
        try:
            if os.path.exists(remove_summary_sidecar):
                os.remove(remove_summary_sidecar)
        except OSError as exc:
            print(f"{C_YELLOW}⚠️ 無法移除舊 summary sidecar：{_project_relative_path(remove_summary_sidecar)}｜{type(exc).__name__}: {exc}{C_RESET}")
    return payload


def _extract_params_embedded_summary(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    return dict(summary) if isinstance(summary, dict) else None


def _load_params_summary_or_legacy_sidecar(params_path: str, legacy_summary_path: str) -> dict | None:
    payload = _load_json_file_or_none(params_path)
    embedded_summary = _extract_params_embedded_summary(payload)
    if embedded_summary is not None:
        return embedded_summary
    return _load_json_file_or_none(legacy_summary_path)


def _visible_policy_paramset_paths(policy_paramset_paths: dict) -> dict[str, str]:
    from tools.optimizer.outer_rolling_oos import get_optimizer_policy_output_label

    from tools.optimizer.outer_rolling_oos import get_optimizer_paramset_policy_names

    visible_names = tuple(get_optimizer_paramset_policy_names())
    return {
        get_optimizer_policy_output_label(name): str(policy_paramset_paths[name])
        for name in visible_names
        if name in dict(policy_paramset_paths or {})
    }


def _print_optimizer_output_files(title: str, entries: list[tuple[str, str]]) -> None:
    from tools.optimizer.outer_rolling_oos import print_optimizer_output_files

    print_optimizer_output_files(entries, title=title, project_root=PROJECT_ROOT, color=True)


def _find_finalist_entry(finalists, winner_trial):
    if winner_trial is None:
        return None
    for item in finalists:
        trial = item.get("trial")
        if trial is not None and int(trial.number) == int(winner_trial.number):
            return item
    return None


def _select_finalist_entry_by_selector(finalists, *, objective_mode: str, selector: str):
    from tools.optimizer.outer_rolling_oos import _build_policy_items

    normalized = _normalize_trade_selector(selector)
    policy_items = _build_policy_items(list(finalists or []), objective_mode=objective_mode)
    item = policy_items.get(normalized)
    if item is not None and item.get("trial") is not None:
        return item
    return None


def _build_best_summary_payload(*, winner_trial, finalist_entry, objective_mode: str, walk_forward_policy: dict, action_label: str, selection_rule: str, compare_only: bool = False):
    if winner_trial is None or finalist_entry is None:
        raise ValueError("缺少 winner_trial 或 finalist_entry，無法建立 summary")
    effective_search_train_end_year = int(
        winner_trial.user_attrs.get("search_train_end_year", walk_forward_policy.get("search_train_end_year", 0))
    )
    policy_contract = build_optimizer_effective_policy_fingerprint(walk_forward_policy)
    payload = {
        "trial_number": int(winner_trial.number) + 1,
        "mode": str(walk_forward_policy.get("model_mode", "")),
        "evaluation_scope": str(walk_forward_policy.get("evaluation_scope", "")),
        "policy_fingerprint_sha256": str(policy_contract["fingerprint_sha256"]),
        "policy_snapshot": policy_contract["snapshot"],
        "base_score": float(finalist_entry["base_score"]),
        "local_min_score": float(finalist_entry["local_min_score"]),
        "retention": float(finalist_entry["local_retention"]),
        "local_gate": bool(finalist_entry["gate_pass"]),
        "objective_mode": str(objective_mode),
        "train_start_year": int(walk_forward_policy.get("train_start_year", 0)),
        "search_train_end_year": int(effective_search_train_end_year),
        "selection_end_year": int(walk_forward_policy.get("search_train_end_year", effective_search_train_end_year)),
        "train_start_date": walk_forward_policy.get("train_start_date"),
        "search_train_end_date": walk_forward_policy.get("search_train_end_date"),
        "latest_data_date": walk_forward_policy.get("latest_data_date"),
        "train_window_months": walk_forward_policy.get("trade_train_window_months") or walk_forward_policy.get("train_window_months"),
        "oos_start_year": walk_forward_policy.get("oos_start_year"),
        "oos_start_date": walk_forward_policy.get("oos_start_date"),
        "oos_end_date": walk_forward_policy.get("oos_end_date"),
        "action": str(action_label),
        "selection_rule": str(selection_rule),
        "compare_only": bool(compare_only),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "local_min_review_mode": str(finalist_entry.get("local_min_review_mode", "exact")),
        "local_min_exact": bool(finalist_entry.get("local_min_exact", bool(is_optimizer_local_min_review_enabled()))),
        "inner_validate_anti_overfit_enabled": bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED),
        "inner_validate_min_score": float(OPTIMIZER_INNER_VALIDATE_MIN_SCORE),
        "inner_validate_max_rank_percentile": float(OPTIMIZER_INNER_VALIDATE_MAX_RANK_PERCENTILE),
        "dominant_year_dependency_anti_overfit_enabled": bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED),
        "created_at": get_taipei_now().isoformat(),
    }
    inner_validate_diagnostics = finalist_entry.get("inner_validate_diagnostics")
    if isinstance(inner_validate_diagnostics, dict):
        payload["inner_validate_diagnostics"] = inner_validate_diagnostics
        payload["inner_validate_score"] = float(inner_validate_diagnostics.get("inner_validate_score", 0.0))
        payload["inner_validate_gate"] = bool(inner_validate_diagnostics.get("inner_validate_gate", False))
        if inner_validate_diagnostics.get("inner_validate_rank") is not None:
            payload["inner_validate_rank"] = int(inner_validate_diagnostics["inner_validate_rank"])
        if inner_validate_diagnostics.get("inner_validate_rank_cutoff") is not None:
            payload["inner_validate_rank_cutoff"] = int(inner_validate_diagnostics["inner_validate_rank_cutoff"])
        if inner_validate_diagnostics.get("inner_validate_rank_total") is not None:
            payload["inner_validate_rank_total"] = int(inner_validate_diagnostics["inner_validate_rank_total"])
        if inner_validate_diagnostics.get("validate_year") is not None:
            payload["inner_validate_year"] = int(inner_validate_diagnostics["validate_year"])
        if inner_validate_diagnostics.get("inner_train_end_year") is not None:
            payload["inner_train_end_year"] = int(inner_validate_diagnostics["inner_train_end_year"])
    diagnostics = finalist_entry.get("dominant_year_dependency_diagnostics")
    if isinstance(diagnostics, dict):
        payload["dominant_year_dependency_diagnostics"] = diagnostics
        payload["dependency_warning"] = bool(diagnostics.get("dependency_warning", False))
    return payload


def _resolve_candidate_best_selection_rule():
    parts = ["max_local_min_score" if is_optimizer_local_min_review_enabled() else "max_base_score_local_min_disabled"]
    if bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
        parts.append("inner_validate_score_gt_0_and_rank_top_half")
    if bool(OPTIMIZER_DOMINANT_YEAR_DEPENDENCY_ANTI_OVERFIT_ENABLED):
        parts.append("intuitive_dominant_year_dependency_veto")
    return "_with_".join(parts)


def _summary_policy_signature(summary: dict | None):
    if not isinstance(summary, dict):
        return None
    return (
        str(summary.get("mode") or summary.get("selected_model_mode") or ""),
        str(summary.get("objective_mode", "")),
        summary.get("train_start_date") or summary.get("train_start_year"),
        summary.get("search_train_end_date") or summary.get("search_train_end_year"),
        summary.get("train_window_months"),
        summary.get("oos_start_date") or summary.get("oos_start_year"),
    )


def _format_summary_policy_signature(summary: dict | None) -> str:
    signature = _summary_policy_signature(summary)
    if signature is None:
        return "N/A"
    mode, objective_mode, train_start_date, search_train_end_date, train_window_months, oos_start_date = signature
    return (
        f"mode={mode or 'N/A'}"
        f", objective={objective_mode or 'N/A'}"
        f", train={train_start_date or 'N/A'}~{search_train_end_date or 'N/A'}"
        f", window_m={train_window_months if train_window_months is not None else 'N/A'}"
        f", oos_start={oos_start_date if oos_start_date is not None else 'N/A'}"
    )


def _summaries_have_compatible_policy(*, candidate_summary: dict, run_best_summary: dict | None) -> bool:
    if run_best_summary is None:
        return False
    candidate_fp = _policy_fingerprint_from_summary(candidate_summary)
    run_best_fp = _policy_fingerprint_from_summary(run_best_summary)
    if candidate_fp and run_best_fp:
        return candidate_fp == run_best_fp
    return _summary_policy_signature(candidate_summary) == _summary_policy_signature(run_best_summary)


def _score_from_summary_for_promote(summary: dict | None, selector: str | None = None):
    if not isinstance(summary, dict):
        return None
    selector_key = _normalize_trade_selector(selector or _resolve_trade_run_best_selector())
    if selector_key in {"base_finalist_best", "local_finalist_best", "retention_finalist_best", "base_finalists_agree", "local_finalists_agree", "retention_finalists_agree", "base", "local", "retention"}:
        try:
            return _selector_score_from_summary(summary, selector_key)
        except (TypeError, ValueError, KeyError):
            return None
    for key in ("selector_score", "local_min_score", "base_score"):
        value = summary.get(key)
        if _is_finite_number(value) and float(value) != float(INVALID_TRIAL_VALUE):
            return float(value)
    return None


def _should_promote_candidate(*, candidate_summary: dict, run_best_summary: dict | None, selector: str | None = None):
    """Compatibility helper for validation code; real promotion uses same-window replay.

    The old implementation allowed policy mismatch to auto-promote.  Trade mode
    now treats policy mismatch as candidate-only, because run_best can only be
    replaced after same-policy, same-window replay or when no run_best exists.
    """
    if not isinstance(candidate_summary, dict):
        return False, "candidate summary 缺失"
    if run_best_summary is None:
        return True, "run_best summary 不存在"
    if not _summaries_have_compatible_policy(candidate_summary=candidate_summary, run_best_summary=run_best_summary):
        return False, "effective policy 不相容，保留 candidate_best，不自動 promote run_best"
    candidate_score = _score_from_summary_for_promote(candidate_summary, selector=selector)
    run_best_score = _score_from_summary_for_promote(run_best_summary, selector=selector)
    if candidate_score is None:
        return False, "candidate selector score 無效"
    if run_best_score is None:
        return True, "run_best selector score 無效"
    delta = float(candidate_score) - float(run_best_score)
    if delta >= float(TRADE_PROMOTE_MIN_SCORE_DELTA):
        return True, f"candidate selector score delta={format_optimizer_score_for_display(delta, decimals=3)} 通過"
    return False, f"candidate selector score delta={format_optimizer_score_for_display(delta, decimals=3)} 未達門檻"


def _summary_is_trade_mode(summary: dict | None) -> bool:
    if not isinstance(summary, dict):
        return False
    mode = str(summary.get("mode") or summary.get("selected_model_mode") or "").strip().lower()
    try:
        return normalize_optimizer_model_mode(mode) == "trade"
    except ValueError:
        return False


def _load_candidate_params_payload_for_promote():
    candidate_params = _load_json_file_or_none(CANDIDATE_BEST_PARAMS_PATH)
    if candidate_params is None:
        print(f"{C_RED}❌ 找不到 candidate_best 參數檔: {_project_relative_path(CANDIDATE_BEST_PARAMS_PATH)}{C_RESET}", file=sys.stderr)
        return None
    if is_active_param_ensemble_payload(candidate_params):
        return candidate_params
    from core.params_io import load_params_from_json, params_to_json_dict
    return params_to_json_dict(load_params_from_json(CANDIDATE_BEST_PARAMS_PATH))


def _first_params_from_payload(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    if is_active_param_ensemble_payload(payload):
        members = payload.get("params_ensemble")
        if isinstance(members, list) and members and isinstance(members[0], dict):
            params = members[0].get("params")
            if isinstance(params, dict):
                return dict(params)
        by_date = payload.get("params_ensemble_by_effective_date")
        if isinstance(by_date, dict) and by_date:
            first_key = sorted(by_date.keys())[0]
            first_members = by_date.get(first_key)
            if isinstance(first_members, list) and first_members and isinstance(first_members[0], dict):
                params = first_members[0].get("params")
                if isinstance(params, dict):
                    return dict(params)
        return {}
    return dict(payload)


def _payload_for_trade_replay(payload: dict, *, selector: str) -> dict:
    if is_active_param_ensemble_payload(payload):
        replay_payload = dict(payload)
        replay_payload["selector"] = str(selector)
        return replay_payload
    return build_static_active_param_ensemble_payload(
        members=[{"member_index": 1, "params": dict(payload)}],
        selector=str(selector),
        created_at=get_taipei_now().isoformat(),
        meta={"source": "trade_promote_replay"},
    )


def _initial_capital_from_payload(payload: dict | None) -> float:
    params = _first_params_from_payload(payload)
    value = params.get("initial_capital", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _trade_window_from_policy(policy: dict) -> tuple[str, str]:
    start = str((policy or {}).get("train_start_date") or "").strip()
    end = str((policy or {}).get("search_train_end_date") or "").strip()
    if not start or not end:
        raise ValueError("Trade mode 缺少 train_start_date / search_train_end_date，無法 same-window replay。")
    return start, end


def _replay_payload_on_current_trade_window(session, payload: dict, *, selector: str) -> float:
    if session is None:
        raise ValueError("promote gate 需要目前 Trade session 才能重跑 same-window replay")
    from tools.optimizer.callbacks import _run_static_ensemble_dashboard_replay

    start_date, end_date = _trade_window_from_policy(getattr(session, "walk_forward_policy", {}) or {})
    replay_payload = _payload_for_trade_replay(dict(payload), selector=selector)
    session_contract_min_rows = resolve_raw_universe_required_min_rows(getattr(session, "walk_forward_policy", {}) or {})
    if session_contract_min_rows is not None and resolve_raw_universe_required_min_rows(replay_payload) is None:
        replay_payload[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(session_contract_min_rows)
    initial_capital = _initial_capital_from_payload(replay_payload)
    metrics, _benchmark_metrics, _range_text = _run_static_ensemble_dashboard_replay(
        session,
        replay_payload,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
    )
    score = metrics.get("pf_romd")
    if not _is_finite_number(score):
        raise ValueError("same-window replay score 無效")
    return float(score)


def _build_promote_result_payload(*, status: str, reason: str, selector: str, candidate_selector_score, candidate_replay_score=None, run_best_replay_score=None) -> dict:
    def _optional_float(value):
        return float(value) if _is_finite_number(value) else None

    candidate_selector_score_value = _optional_float(candidate_selector_score)
    candidate_replay_score_value = _optional_float(candidate_replay_score)
    run_best_replay_score_value = _optional_float(run_best_replay_score)
    delta = None
    if candidate_replay_score_value is not None and run_best_replay_score_value is not None:
        delta = float(candidate_replay_score_value) - float(run_best_replay_score_value)
    return {
        "promote_status": str(status),
        "promote_reason": str(reason),
        "candidate_selector": _resolve_trade_candidate_selector(),
        "run_best_selector": str(selector),
        "candidate_selector_score": candidate_selector_score_value,
        "candidate_trade_replay_score": candidate_replay_score_value,
        "run_best_replay_score": run_best_replay_score_value,
        "score_delta": delta,
        "required_score_delta": float(TRADE_PROMOTE_MIN_SCORE_DELTA),
        "run_best_replayed_on_current_window": run_best_replay_score is not None,
        "created_at": get_taipei_now().isoformat(),
    }


def _promote_candidate_to_run_best(*, session=None, emit_output: bool = True):
    candidate_params = _load_candidate_params_payload_for_promote()
    if candidate_params is None:
        return 1
    candidate_summary = _load_params_summary_or_legacy_sidecar(CANDIDATE_BEST_PARAMS_PATH, CANDIDATE_BEST_SUMMARY_PATH)
    if candidate_summary is None:
        print(f"{C_RED}❌ 找不到 candidate_best summary: {_project_relative_path(CANDIDATE_BEST_PARAMS_PATH)}{C_RESET}", file=sys.stderr)
        return 1
    selector = _resolve_trade_run_best_selector()
    candidate_selector_score = _selector_score_from_summary(candidate_summary, selector)
    run_best_payload = _load_json_file_or_none(RUN_BEST_PARAMS_PATH)
    run_best_summary = _load_params_summary_or_legacy_sidecar(RUN_BEST_PARAMS_PATH, RUN_BEST_SUMMARY_PATH)
    if not _summary_is_trade_mode(candidate_summary):
        result = _build_promote_result_payload(
            status="candidate_only",
            reason="candidate_not_from_trade_mode",
            selector=selector,
            candidate_selector_score=candidate_selector_score,
        )
        print(f"{C_YELLOW}ℹ️ run_best 未進版：candidate 不是 Trade mode 產物{C_RESET}")
        return 0
    if not _selector_score_is_valid(candidate_summary, selector):
        result = _build_promote_result_payload(
            status="candidate_only",
            reason="candidate_selector_score_invalid",
            selector=selector,
            candidate_selector_score=candidate_selector_score,
        )
        print(f"{C_YELLOW}ℹ️ run_best 未進版：candidate selector score 無效{C_RESET}")
        return 0

    try:
        candidate_replay_score = _replay_payload_on_current_trade_window(session, candidate_params, selector=selector)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"{C_YELLOW}ℹ️ run_best 未進版：candidate same-window replay 失敗｜{type(exc).__name__}: {exc}{C_RESET}")
        return 0

    if run_best_payload is None or run_best_summary is None:
        promoted_summary = dict(candidate_summary)
        promoted_summary["promoted_at"] = get_taipei_now().isoformat()
        promoted_summary["promote_gate"] = _build_promote_result_payload(
            status="promoted",
            reason="no_existing_run_best",
            selector=selector,
            candidate_selector_score=candidate_selector_score,
            candidate_replay_score=candidate_replay_score,
        )
        promoted_payload = dict(candidate_params) if isinstance(candidate_params, dict) else {}
        promoted_payload["summary"] = promoted_summary
        _write_json_file(RUN_BEST_PARAMS_PATH, promoted_payload)
        if bool(emit_output):
            _print_optimizer_output_files("✅ run_best 已進版", [("run_best", RUN_BEST_PARAMS_PATH)])
        return 0

    if not _summaries_have_compatible_policy(candidate_summary=candidate_summary, run_best_summary=run_best_summary):
        print(f"{C_YELLOW}ℹ️ run_best 未進版：policy fingerprint 不相容，保留 candidate_best{C_RESET}")
        return 0

    try:
        run_best_replay_score = _replay_payload_on_current_trade_window(session, run_best_payload, selector=selector)
    except (ValueError, RuntimeError, OSError) as exc:
        promoted_summary = dict(candidate_summary)
        promoted_summary["promoted_at"] = get_taipei_now().isoformat()
        promoted_summary["promote_gate"] = _build_promote_result_payload(
            status="promoted",
            reason=f"existing_run_best_replay_invalid:{type(exc).__name__}",
            selector=selector,
            candidate_selector_score=candidate_selector_score,
            candidate_replay_score=candidate_replay_score,
        )
        promoted_payload = dict(candidate_params) if isinstance(candidate_params, dict) else {}
        promoted_payload["summary"] = promoted_summary
        _write_json_file(RUN_BEST_PARAMS_PATH, promoted_payload)
        if bool(emit_output):
            _print_optimizer_output_files("✅ run_best 已進版", [("run_best", RUN_BEST_PARAMS_PATH)])
        return 0

    score_delta = float(candidate_replay_score) - float(run_best_replay_score)
    if score_delta < float(TRADE_PROMOTE_MIN_SCORE_DELTA):
        print(
            f"{C_YELLOW}ℹ️ run_best 未進版：same-window replay delta={format_optimizer_score_for_display(score_delta, decimals=3)} "
            f"< required={format_optimizer_score_for_display(TRADE_PROMOTE_MIN_SCORE_DELTA, decimals=3)}{C_RESET}"
        )
        return 0

    promoted_summary = dict(candidate_summary)
    promoted_summary["promoted_at"] = get_taipei_now().isoformat()
    promoted_summary["promote_gate"] = _build_promote_result_payload(
        status="promoted",
        reason="candidate_score_delta_passed",
        selector=selector,
        candidate_selector_score=candidate_selector_score,
        candidate_replay_score=candidate_replay_score,
        run_best_replay_score=run_best_replay_score,
    )
    promoted_payload = dict(candidate_params) if isinstance(candidate_params, dict) else {}
    promoted_payload["summary"] = promoted_summary
    _write_json_file(RUN_BEST_PARAMS_PATH, promoted_payload)
    try:
        if os.path.exists(RUN_BEST_SUMMARY_PATH):
            os.remove(RUN_BEST_SUMMARY_PATH)
    except OSError as exc:
        print(f"{C_YELLOW}⚠️ 無法移除舊 run_best summary sidecar：{_project_relative_path(RUN_BEST_SUMMARY_PATH)}｜{type(exc).__name__}: {exc}{C_RESET}")
    if bool(emit_output):
        _print_optimizer_output_files("✅ run_best 已進版", [("run_best", RUN_BEST_PARAMS_PATH)])
    return 0


def generate_walk_forward_report_from_payload(*, session, params_payload, dataset_label, db_file, best_trial_number=None, walk_forward_policy: dict):
    from core.params_io import build_params_from_mapping
    from core.strategy_params import build_runtime_param_raw_value
    from tools.optimizer.prep import prepare_trial_inputs
    from tools.optimizer.walk_forward import evaluate_walk_forward, write_walk_forward_report

    ai_params = build_params_from_mapping(params_payload)
    prep_executor_bundle = session.get_trial_prep_executor_bundle(
        build_runtime_param_raw_value(ai_params, "optimizer_max_workers")
    )
    prep_result = prepare_trial_inputs(
        raw_data_cache=session.raw_data_cache,
        params=ai_params,
        default_max_workers=session.default_max_workers,
        executor_bundle=prep_executor_bundle,
        static_fast_cache=session.static_fast_cache,
        static_master_dates=session.master_dates,
        include_trade_logs=False,
        include_pit_stats_index=True,
    )
    report = evaluate_walk_forward(
        all_dfs_fast=prep_result["all_dfs_fast"],
        all_trade_logs=prep_result["all_trade_logs"],
        sorted_dates=sorted(prep_result["master_dates"]),
        params=ai_params,
        max_positions=session.train_max_positions,
        enable_rotation=session.train_enable_rotation,
        benchmark_ticker="0050",
        train_start_year=int(walk_forward_policy["train_start_year"]),
        min_train_years=int(walk_forward_policy["min_train_years"]),
        oos_start_year=walk_forward_policy.get("oos_start_year"),
        train_start_date=walk_forward_policy.get("train_start_date"),
        oos_start_date=walk_forward_policy.get("oos_start_date"),
        oos_end_date=walk_forward_policy.get("oos_end_date"),
        pit_stats_index=prep_result.get("all_pit_stats_index"),
    )
    report_paths = write_walk_forward_report(
        output_dir=session.output_dir,
        params_payload=params_payload,
        dataset_label=dataset_label,
        report=report,
        best_trial_number=best_trial_number,
        source_db_path=db_file,
        session_ts=session.session_ts,
    )
    return report, report_paths


def generate_best_trial_walk_forward_report(*, session, best_trial, dataset_label, db_file, walk_forward_policy: dict):
    from tools.optimizer.study_utils import build_best_params_payload_from_trial

    params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=session.optimizer_fixed_tp_percent)
    report, report_paths = generate_walk_forward_report_from_payload(
        session=session,
        params_payload=params_payload,
        dataset_label=dataset_label,
        db_file=db_file,
        best_trial_number=int(best_trial.number) + 1,
        walk_forward_policy=walk_forward_policy,
    )
    return report, report_paths


def print_walk_forward_outputs(*, report, report_paths):
    return None


def finalize_best_trial_outputs(*, session, study, best_trial_resolver, dataset_label: str, db_file: str, walk_forward_policy: dict):
    from tools.optimizer.study_utils import is_qualified_trial_value

    best_trial = best_trial_resolver(study)
    if best_trial is None:
        print(f"{C_YELLOW}ℹ️ 目前尚無通過 local_min_score gate 的 winner，略過 OOS 報表輸出。{C_RESET}")
        return 0
    if not is_qualified_trial_value(best_trial.value):
        return 0

    report, report_paths = generate_best_trial_walk_forward_report(
        session=session,
        best_trial=best_trial,
        dataset_label=dataset_label,
        db_file=db_file,
        walk_forward_policy=walk_forward_policy,
    )
    print_walk_forward_outputs(report=report, report_paths=report_paths)
    return 0


def _export_selected_candidate_artifacts(
    *,
    study,
    finalists,
    selected_trial,
    params_path: str,
    summary_path: str,
    fixed_tp_percent: float,
    colors: dict,
    objective_mode: str,
    walk_forward_policy: dict,
    action_label: str,
    selection_rule: str,
    compare_only: bool,
    artifact_label: str,
    export_best_params_if_requested,
    is_qualified_trial_value,
):
    if selected_trial is None or not is_qualified_trial_value(selected_trial.value):
        print(f"{C_YELLOW}ℹ️ {artifact_label} 無可匯出 trial。{C_RESET}")
        return False
    export_status = export_best_params_if_requested(
        study,
        best_params_path=params_path,
        fixed_tp_percent=fixed_tp_percent,
        colors=colors,
        best_trial_resolver=(lambda _study, _selected_trial=selected_trial: _selected_trial),
        suppress_success_message=True,
    )
    if export_status != 0:
        return False
    finalist_entry = _find_finalist_entry(finalists, selected_trial)
    if finalist_entry is None:
        raise ValueError(f"找不到 {artifact_label} 的 finalist entry，無法建立 summary")
    candidate_summary = _build_best_summary_payload(
        winner_trial=selected_trial,
        finalist_entry=finalist_entry,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        action_label=action_label,
        selection_rule=selection_rule,
        compare_only=compare_only,
    )
    if os.path.abspath(summary_path) == os.path.abspath(CANDIDATE_BEST_SUMMARY_PATH):
        _embed_summary_in_params_file(params_path, candidate_summary, remove_summary_sidecar=summary_path)
    else:
        _embed_summary_in_params_file(params_path, candidate_summary)
        _write_json_file(summary_path, candidate_summary)
    compare_note = "（比較用，不參與 promote）" if compare_only else ""
    _print_optimizer_output_files(f"💾 {artifact_label}{compare_note} 已寫入", [(artifact_label, params_path)])
    return True


def _resolve_nonrolling_seed_ensemble_policy(*, seed_count: int | None = None):
    resolved_seed_count = OPTIMIZER_RANDOM_SEED_ENSEMBLE_SIZE if seed_count is None else int(seed_count)
    return build_seed_ensemble_policy_snapshot(
        enabled=OPTIMIZER_RANDOM_SEED_ENSEMBLE_ENABLED,
        seed_count=resolved_seed_count,
        min_agree=OPTIMIZER_RANDOM_SEED_ENSEMBLE_MIN_AGREE,
    )


def _is_base_finalists_agree_policy_name(policy_name: str | None) -> bool:
    return str(policy_name or "").strip() == "base_finalists_agree"


def _is_local_finalists_agree_policy_name(policy_name: str | None) -> bool:
    return str(policy_name or "").strip() == "local_finalists_agree"


def _is_retention_finalists_agree_policy_name(policy_name: str | None) -> bool:
    return str(policy_name or "").strip() == "retention_finalists_agree"


def _is_finalists_agree_policy_name(policy_name: str | None) -> bool:
    return str(policy_name or "").strip() in {"base_finalists_agree", "local_finalists_agree", "retention_finalists_agree"}


def _resolve_finalists_agree_min_agree_for_policy_name(policy_name: str, member_count: int) -> int:
    if _is_local_finalists_agree_policy_name(policy_name):
        return int(resolve_optimizer_local_finalists_agree_min_agree(member_count, OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE))
    if _is_retention_finalists_agree_policy_name(policy_name):
        return int(resolve_optimizer_retention_finalists_agree_min_agree(member_count, OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE))
    return int(resolve_optimizer_base_finalists_agree_min_agree(member_count, OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE))


def _finalists_agree_policy_metadata(policy_name: str) -> dict:
    if _is_local_finalists_agree_policy_name(policy_name):
        return {
            "selection_rule": "all_finalists_local_min_sum_best_seed_finalist_agree",
            "min_agree_requested_key": "local_agree_min_agree_requested",
            "min_agree_requested": OPTIMIZER_LOCAL_FINALISTS_AGREE_MIN_AGREE,
            "member_selection": "seed_with_max_all_finalists_local_min_sum",
        }
    if _is_retention_finalists_agree_policy_name(policy_name):
        return {
            "selection_rule": "all_finalists_retention_sum_best_seed_finalist_agree",
            "min_agree_requested_key": "retention_agree_min_agree_requested",
            "min_agree_requested": OPTIMIZER_RETENTION_FINALISTS_AGREE_MIN_AGREE,
            "member_selection": "seed_with_max_all_finalists_retention_sum",
        }
    return {
        "selection_rule": "all_finalists_base_score_sum_best_seed_finalist_agree",
        "min_agree_requested_key": "base_agree_min_agree_requested",
        "min_agree_requested": OPTIMIZER_BASE_FINALISTS_AGREE_MIN_AGREE,
        "member_selection": "seed_with_max_all_finalists_base_score_sum",
    }


def _resolve_nonrolling_policy_seed_ensemble_policy(*, policy_name: str, members: list[dict], seeds: list[int]) -> dict:
    if _is_finalists_agree_policy_name(policy_name):
        member_count = max(1, len(renumber_seed_ensemble_members(list(members or []))))
        metadata = _finalists_agree_policy_metadata(policy_name)
        policy = build_seed_ensemble_policy_snapshot(
            enabled=member_count > 1,
            seed_count=member_count,
            min_agree=_resolve_finalists_agree_min_agree_for_policy_name(policy_name, member_count),
        )
        policy["selection_rule"] = str(metadata["selection_rule"])
        policy[str(metadata["min_agree_requested_key"])] = metadata["min_agree_requested"]
        policy["member_selection"] = str(metadata["member_selection"])
        policy["intra_seed_agree"] = "selected_seed_finalists"
        policy["requested_random_seed_ensemble"] = _resolve_nonrolling_seed_ensemble_policy(seed_count=len(seeds))
        return policy
    return _resolve_nonrolling_seed_ensemble_policy(seed_count=len(seeds))


def _resolve_optimizer_session_ts(session, *, fallback_label: str = "") -> str:
    raw_session_ts = str(getattr(session, "session_ts", "") or "").strip()
    if raw_session_ts:
        return raw_session_ts
    suffix = str(fallback_label or "").strip()
    generated = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    return generated if not suffix else f"{generated}_{suffix}"


def _build_seed_ensemble_member(*, member_index: int, seed: int, best_trial, finalist_entry: dict, params_payload: dict) -> dict:
    member = {
        "member_index": int(member_index),
        "seed": int(seed),
        "selected_trial": int(best_trial.number) + 1,
        "params": dict(params_payload),
    }
    if isinstance(finalist_entry, dict):
        member["score"] = float(finalist_entry.get("base_score", 0.0))
        member["base_score"] = float(finalist_entry.get("base_score", 0.0))
        member["local_min_score"] = float(finalist_entry.get("local_min_score", 0.0))
        member["retention"] = float(finalist_entry.get("local_retention", 0.0))
        member["local_gate"] = bool(finalist_entry.get("gate_pass", False))
        member["local_min_review_enabled"] = bool(finalist_entry.get("local_min_review_enabled", is_optimizer_local_min_review_enabled()))
        member["local_min_review_mode"] = str(finalist_entry.get("local_min_review_mode", "exact"))
        member["local_min_exact"] = bool(finalist_entry.get("local_min_exact", bool(is_optimizer_local_min_review_enabled())))
    return member



def _format_nonrolling_result_number(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_nonrolling_system_score(value) -> str:
    return format_optimizer_score_for_display(value, decimals=3)


def _build_nonrolling_single_fold_period_context(walk_forward_policy: dict) -> dict:
    from tools.optimizer.outer_rolling_oos import build_optimizer_seed_ensemble_fold_context

    policy = dict(walk_forward_policy or {})
    normalized_mode = normalize_optimizer_model_mode(policy.get("model_mode", "oos"))
    evaluation_scope = str(policy.get("evaluation_scope") or "").strip().lower()
    is_train_only_scope = (
        normalized_mode in {"full", "trade"}
        or evaluation_scope in {"full_seed_ensemble", "trade_train_only"}
        or evaluation_scope.startswith("study_full")
    )

    selection_start_year = int(policy.get("selection_start_year", policy.get("train_start_year", 0)) or 0)
    selection_end_year = int(policy.get("search_train_end_year", selection_start_year) or selection_start_year)
    selection_start_date = str(policy.get("selection_start_date") or policy.get("train_start_date") or "").strip()
    selection_end_date = str(policy.get("search_train_end_date") or "").strip()
    if not selection_start_date and selection_start_year > 0:
        selection_start_date = f"{selection_start_year:04d}-01-01"
    if not selection_end_date and selection_end_year > 0:
        selection_end_date = f"{selection_end_year:04d}-12-31"

    if is_train_only_scope:
        context = build_optimizer_seed_ensemble_fold_context(
            fold_idx=1,
            fold_count=1,
            selection_start_date=selection_start_date,
            selection_end_date=selection_end_date,
            oos_start_date="",
            oos_end_date="",
            oos_period="",
        )
        context["model_mode"] = normalized_mode
        context["evaluation_scope"] = evaluation_scope or ("full_seed_ensemble" if normalized_mode == "full" else ("trade_train_only" if normalized_mode == "trade" else "study_full_single_seed"))
        context["oos_start_date"] = ""
        context["oos_end_date"] = ""
        context["oos_period"] = ""
        context["show_oos"] = False
        return context

    oos_start_year = int(policy.get("oos_start_year", selection_end_year + 1) or (selection_end_year + 1))
    oos_start_date = str(policy.get("oos_start_date") or "").strip()
    if not oos_start_date and oos_start_year > 0:
        oos_start_date = f"{oos_start_year:04d}-01-01"
    context = build_optimizer_seed_ensemble_fold_context(
        fold_idx=1,
        fold_count=1,
        selection_start_date=selection_start_date,
        selection_end_date=selection_end_date,
        oos_start_date=oos_start_date,
        oos_end_date=str(policy.get("oos_end_date") or "latest"),
    )
    context["model_mode"] = "oos"
    context["evaluation_scope"] = str(policy.get("evaluation_scope") or "oos_single_fold")
    context["show_oos"] = True
    return context


def _render_nonrolling_single_fold_progress_line(walk_forward_policy: dict, *, stage: str, status: str = "", completed: int = 0, total: int = 0, best_score=None, best_base_score=None, best_local_min_score=None, elapsed_sec=None) -> str:
    from tools.optimizer.outer_rolling_oos import render_optimizer_fold_progress_line

    context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
    return render_optimizer_fold_progress_line(
        **context,
        stage=stage,
        status=status,
        completed=completed,
        total=total,
        best_score=best_score,
        best_base_score=best_base_score,
        best_local_min_score=best_local_min_score,
        elapsed_sec=elapsed_sec,
    )


def _finalize_nonrolling_single_fold_inline_progress(progress_owner) -> None:
    if not bool(getattr(progress_owner, "nonrolling_fold_inline_progress_rendered", False)):
        return
    print(flush=True)
    progress_owner.nonrolling_fold_inline_progress_rendered = False
    progress_owner.nonrolling_fold_inline_progress_width = 0


def _print_nonrolling_single_fold_progress(walk_forward_policy: dict, *, stage: str, status: str = "", completed: int = 0, total: int = 0, best_score=None, best_base_score=None, best_local_min_score=None, elapsed_sec=None, inline: bool = False, progress_owner=None) -> None:
    line = _render_nonrolling_single_fold_progress_line(
        walk_forward_policy,
        stage=stage,
        status=status,
        completed=completed,
        total=total,
        best_score=best_score,
        best_base_score=best_base_score,
        best_local_min_score=best_local_min_score,
        elapsed_sec=elapsed_sec,
    )
    owner = progress_owner
    if inline and owner is not None and stdout_supports_inline_progress():
        previous_width = int(getattr(owner, "nonrolling_fold_inline_progress_width", 0) or 0)
        owner.nonrolling_fold_inline_progress_width = write_inline_progress(f"{C_GRAY}{line}{C_RESET}", previous_width=previous_width)
        owner.nonrolling_fold_inline_progress_rendered = True
        return
    if owner is not None:
        _finalize_nonrolling_single_fold_inline_progress(owner)
    print(f"{C_GRAY}{line}{C_RESET}", flush=True)


def _make_nonrolling_seed_trial_progress_callback(
    *,
    member_session,
    walk_forward_policy: dict,
    requested_trials: int,
    started_at: float,
    search_started_ts: float | None = None,
    status: str = "",
    inline: bool = True,
    emit_lock=None,
    throttle_trials: int = 1,
    throttle_seconds: float = 0.0,
    progress_board=None,
    seed_index: int = 0,
    seed: int = 0,
    seed_count: int = 0,
):
    state = {"last_completed": 0, "last_emitted_at": 0.0}

    def _should_emit(completed: int, now: float) -> bool:
        total = max(1, int(requested_trials))
        if completed <= 0:
            return False
        if completed >= total:
            return True
        if int(throttle_trials) <= 1:
            return True
        if completed - int(state.get("last_completed", 0) or 0) >= int(throttle_trials):
            return True
        if float(throttle_seconds) > 0 and now - float(state.get("last_emitted_at", 0.0) or 0.0) >= float(throttle_seconds):
            return True
        return False

    def _emit_progress(**kwargs) -> None:
        if progress_board is not None and int(seed_index or 0) > 0:
            progress = {
                "stage": kwargs.get("stage"),
                "status": kwargs.get("status"),
                "seed_ensemble_member_index": int(seed_index),
                "seed_ensemble_member_count": int(seed_count or 0),
                "seed": int(seed or 0),
                "completed": kwargs.get("completed"),
                "total": kwargs.get("total"),
                "best_score": kwargs.get("best_score"),
                "elapsed_sec": kwargs.get("elapsed_sec"),
            }
            for phase_key in ("search_started_ts", "search_last_done_ts"):
                if kwargs.get(phase_key) is not None:
                    progress[phase_key] = kwargs.get(phase_key)
            if emit_lock is None:
                progress_board.update(fold_idx=1, seed_index=int(seed_index), progress=progress)
                return
            with emit_lock:
                progress_board.update(fold_idx=1, seed_index=int(seed_index), progress=progress)
            return
        print_kwargs = dict(kwargs)
        for phase_key in ("search_started_ts", "search_last_done_ts"):
            print_kwargs.pop(phase_key, None)
        if emit_lock is None:
            _print_nonrolling_single_fold_progress(**print_kwargs)
            return
        with emit_lock:
            _print_nonrolling_single_fold_progress(**print_kwargs)

    def _callback(study, trial):
        completed = int(getattr(trial, "number", -1)) + 1
        now = time.perf_counter()
        if not _should_emit(completed, now):
            return
        best_trial = member_session.get_best_completed_trial_or_none(study)
        best_score = None if best_trial is None else best_trial.value
        _emit_progress(
            walk_forward_policy=walk_forward_policy,
            stage="OPTIMIZER_SEARCH",
            status=status,
            completed=completed,
            total=int(requested_trials),
            best_score=best_score,
            elapsed_sec=max(0.0, now - float(started_at)),
            search_started_ts=search_started_ts,
            search_last_done_ts=time.time(),
            inline=bool(inline),
            progress_owner=member_session if inline else None,
        )
        state["last_completed"] = completed
        state["last_emitted_at"] = now

    return _callback




def _emit_nonrolling_seed_process_progress_event(task: dict, *, stage: str, status: str = "", completed: int = 0, total: int = 0, best_score=None, best_base_score=None, best_local_min_score=None, elapsed_sec=None, **extra_payload) -> None:
    from tools.optimizer.outer_rolling_oos import write_optimizer_seed_progress_event

    context = dict((task or {}).get("progress_context") or {})
    member_index = int((task or {}).get("member_index", 0) or 0)
    member_count = int((task or {}).get("member_count", 0) or 0)
    seed = int((task or {}).get("seed", 0) or 0)
    payload = {
        "seed_ensemble_member_index": int(member_index),
        "seed_ensemble_member_count": int(member_count),
        "seed": int(seed),
        "status": str(status or ""),
        "elapsed_sec": elapsed_sec,
    }
    if completed is not None:
        payload["completed"] = int(completed or 0)
    if total is not None:
        payload["total"] = int(total or 0)
    if best_score is not None:
        payload["best_score"] = float(best_score)
    if best_base_score is not None:
        payload["best_base_score"] = float(best_base_score)
    if best_local_min_score is not None:
        payload["best_local_min_score"] = float(best_local_min_score)
    payload.update(dict(extra_payload or {}))
    write_optimizer_seed_progress_event(
        stage=str(stage),
        fold_idx=int(context.get("fold_idx", 1) or 1),
        fold_count=int(context.get("fold_count", 1) or 1),
        oos_year=int(context.get("oos_year", 0) or 0),
        selection_start=str(context.get("selection_start") or ""),
        selection_end=str(context.get("selection_end") or ""),
        **payload,
    )


def _make_nonrolling_seed_process_trial_progress_callback(*, task: dict, member_session, requested_trials: int, started_at: float, search_started_ts: float | None = None):
    state = {"last_completed": 0}

    def _callback(study, trial):
        completed = int(getattr(trial, "number", -1)) + 1
        if completed <= 0 or completed == int(state.get("last_completed", 0) or 0):
            return
        best_trial = member_session.get_best_completed_trial_or_none(study)
        best_score = None if best_trial is None else best_trial.value
        member_index = int((task or {}).get("member_index", 0) or 0)
        member_count = int((task or {}).get("member_count", 0) or 0)
        _emit_nonrolling_seed_process_progress_event(
            task,
            stage="OPTIMIZER_SEARCH",
            status=f"seed {member_index}/{member_count}",
            completed=int(completed),
            total=int(requested_trials),
            best_score=best_score,
            elapsed_sec=max(0.0, time.perf_counter() - float(started_at)),
            search_started_ts=search_started_ts,
            search_last_done_ts=time.time(),
        )
        state["last_completed"] = int(completed)

    return _callback


def _resolve_study_best_base_score(session, study):
    try:
        best_trial = session.get_best_completed_trial_or_none(study)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if best_trial is None:
        return None
    try:
        return float(best_trial.value)
    except (TypeError, ValueError):
        return None


def _score_from_policy_member(policy_members, policy_name: str, score_key: str):
    if not isinstance(policy_members, dict):
        return None
    member = policy_members.get(str(policy_name))
    if isinstance(member, list):
        member = member[0] if member else None
    if not isinstance(member, dict):
        return None
    try:
        return float(member.get(str(score_key)))
    except (TypeError, ValueError):
        return None


def _resolve_nonrolling_seed_done_best_base_score(policy_members, fallback_score=None):
    score = _score_from_policy_member(policy_members, "base_finalist_best", "base_score")
    return fallback_score if score is None else score


def _resolve_nonrolling_seed_done_best_local_min_score(policy_members, fallback_score=None):
    score = _score_from_policy_member(policy_members, "local_finalist_best", "local_min_score")
    return fallback_score if score is None else score


def _make_nonrolling_local_min_progress_context(walk_forward_policy: dict, *, best_base_score=None, completed_results=None, overall_start=None) -> dict:
    context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
    context["best_base_score"] = best_base_score
    context["completed_results"] = list(completed_results or [])
    if overall_start is not None:
        context["overall_start"] = overall_start
    return context


def _install_nonrolling_local_min_seed_board_progress(
    *,
    session,
    walk_forward_policy: dict,
    progress_board,
    progress_lock,
    member_index: int,
    member_count: int,
    seed: int,
    best_base_score=None,
    started_at: float | None = None,
    completed_trials: int = 0,
    search_started_ts: float | None = None,
    search_last_done_ts: float | None = None,
) -> bool:
    if progress_board is None:
        return False
    period_context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
    started_at = time.perf_counter() if started_at is None else float(started_at)
    local_started_ts = time.time()

    def _sink(event: dict) -> None:
        data = dict(event or {})
        progress = {
            "stage": "LOCAL_MIN_REVIEW",
            "seed_ensemble_member_index": int(member_index),
            "seed_ensemble_member_count": int(member_count),
            "seed": int(seed),
            "completed": int(completed_trials or 0),
            "selection_start": str(period_context.get("selection_start") or ""),
            "selection_end": str(period_context.get("selection_end") or ""),
            "oos_period": str(period_context.get("oos_period") or ""),
            "finalist_idx": int(data.get("finalist_idx", 0) or 0),
            "finalist_total": int(data.get("finalist_total", 0) or 0),
            "trial_number": data.get("trial_number"),
            "neighbor_done": int(data.get("neighbor_done", 0) or 0),
            "neighbor_total": int(data.get("neighbor_total", 0) or 0),
            "current": data.get("current"),
            "best": data.get("best"),
            "best_base_score": best_base_score,
            "status": str(data.get("status") or "RUN"),
            "elapsed_sec": max(0.0, time.perf_counter() - started_at),
            "search_started_ts": search_started_ts,
            "search_last_done_ts": search_last_done_ts,
            "local_min_started_ts": data.get("local_min_started_ts") or local_started_ts,
            "local_min_completed": int(data.get("local_min_completed", 0) or 0),
            "local_min_neighbor_completed": int(data.get("local_min_neighbor_completed", 0) or 0),
            "local_min_last_done_ts": data.get("local_min_last_done_ts"),
            "local_min_last_neighbor_done_ts": data.get("local_min_last_neighbor_done_ts") or data.get("local_min_last_done_ts"),
        }
        with progress_lock:
            progress_board.update(fold_idx=1, seed_index=int(member_index), progress=progress)

    session.outer_rolling_parallel_progress_sink = _sink
    session.local_min_progress_event_only = True
    return True


def _clear_nonrolling_local_min_progress_hooks(session) -> None:
    for attr_name in (
        "outer_rolling_parallel_progress_sink",
        "outer_rolling_local_progress_context",
        "local_min_progress_event_only",
    ):
        if hasattr(session, attr_name):
            delattr(session, attr_name)


def _run_nonrolling_seed_ensemble_member_process_task(task: dict) -> dict | None:
    log_path = str((task or {}).get("log_path") or "")

    def _execute() -> dict | None:
        from tools.optimizer.prep import load_all_raw_data as load_all_raw_data_func
        from tools.optimizer.runtime import create_optimizer_study, resolve_optimizer_single_fold_search_parallel_trials
        from tools.optimizer.robustness import print_local_min_score_finalist_review, print_local_min_score_winner_summary
        from tools.optimizer.session import close_study_storage
        from tools.optimizer.study_utils import build_best_params_payload_from_trial, is_qualified_trial_value

        walk_forward_policy = dict(task["walk_forward_policy"])
        set_optimizer_runtime_model_mode(walk_forward_policy.get("model_mode"))
        configure_optuna_logging()
        selected_data_dir = str(task["selected_data_dir"])
        optimizer_required_min_rows = int(task["optimizer_required_min_rows"])
        requested_trials = int(task["requested_trials"])
        seed = int(task["seed"])
        member_index = int(task["member_index"])
        member_count = int(task["member_count"])
        started_at = time.perf_counter()
        study = None
        member_session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        member_session.n_trials = int(requested_trials)
        member_session.run_action = "train"
        member_session.disable_milestone_dashboard = True
        member_session.disable_optimizer_status_line = True
        try:
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="START",
                status=f"seed {member_index}/{member_count}",
                elapsed_sec=0.0,
            )
            task_db_name = task.get("db_name")
            if isinstance(task_db_name, str) and task_db_name.strip().lower() in {"", "none", "null"}:
                task_db_name = None
            study = create_optimizer_study(task_db_name, seed=int(seed), sampler_kind="tpe")
            _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="RAW_DATA",
                status=f"seed {member_index}/{member_count} raw data",
                elapsed_sec=max(0.0, time.perf_counter() - started_at),
            )
            member_session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data_func,
                required_min_rows=optimizer_required_min_rows,
                verbose=False,
            )
            member_session.profile_recorder.init_output_files()
            member_session.profile_recorder.mark_run_started()
            search_started_perf = time.perf_counter()
            search_started_ts = time.time()
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="OPTIMIZER_SEARCH",
                status=f"seed {member_index}/{member_count}",
                completed=0,
                total=int(requested_trials),
                elapsed_sec=0.0,
                search_started_ts=search_started_ts,
            )
            search_parallel_trials = resolve_optimizer_single_fold_search_parallel_trials(task.get("environ") or os.environ, sampler_kind="tpe")
            study.optimize(
                member_session.objective,
                n_trials=int(requested_trials),
                n_jobs=int(search_parallel_trials),
                callbacks=[
                    member_session.monitoring_callback,
                    _make_nonrolling_seed_process_trial_progress_callback(
                        task=task,
                        member_session=member_session,
                        requested_trials=int(requested_trials),
                        started_at=search_started_perf,
                        search_started_ts=search_started_ts,
                    ),
                ],
            )
            search_done_ts = time.time()
            best_base_score = _resolve_study_best_base_score(member_session, study)
            local_started_at = time.perf_counter()
            local_started_ts = time.time()

            def _process_local_min_progress_sink(event: dict) -> None:
                data = dict(event or {})
                _emit_nonrolling_seed_process_progress_event(
                    task,
                    stage="LOCAL_MIN_REVIEW",
                    status=str(data.get("status") or "RUN"),
                    finalist_idx=int(data.get("finalist_idx", 0) or 0),
                    finalist_total=int(data.get("finalist_total", 0) or 0),
                    trial_number=data.get("trial_number"),
                    neighbor_done=int(data.get("neighbor_done", 0) or 0),
                    neighbor_total=int(data.get("neighbor_total", 0) or 0),
                    current=data.get("current"),
                    best=data.get("best"),
                    completed=int(requested_trials),
                    total=int(requested_trials),
                    best_base_score=best_base_score,
                    elapsed_sec=max(0.0, time.perf_counter() - local_started_at),
                    search_started_ts=search_started_ts,
                    search_last_done_ts=search_done_ts,
                    local_min_started_ts=data.get("local_min_started_ts") or local_started_ts,
                    local_min_completed=int(data.get("local_min_completed", 0) or 0),
                    local_min_neighbor_completed=int(data.get("local_min_neighbor_completed", 0) or 0),
                    local_min_last_done_ts=data.get("local_min_last_done_ts"),
                    local_min_last_neighbor_done_ts=data.get("local_min_last_neighbor_done_ts") or data.get("local_min_last_done_ts"),
                )

            member_session.outer_rolling_parallel_progress_sink = _process_local_min_progress_sink
            member_session.local_min_progress_event_only = True
            try:
                finalists, best_trial = print_local_min_score_finalist_review(
                    study,
                    session=member_session,
                    objective_mode=str(task.get("objective_mode") or "split_train_romd"),
                    colors=COLORS,
                    winner_trial=None,
                    emit_table=False,
                    show_progress=True,
                )
            finally:
                _clear_nonrolling_local_min_progress_hooks(member_session)
            if best_trial is None or not is_qualified_trial_value(best_trial.value):
                _emit_nonrolling_seed_process_progress_event(
                    task,
                    stage="DONE",
                    status=f"seed {member_index}/{member_count} skipped",
                    completed=int(requested_trials),
                    total=int(requested_trials),
                    elapsed_sec=max(0.0, time.perf_counter() - started_at),
                )
                return None
            print_local_min_score_winner_summary(winner_trial=best_trial, session=member_session, colors=COLORS)
            finalist_entry = _find_finalist_entry(finalists, best_trial)
            if finalist_entry is None:
                raise ValueError(f"seed={seed} 找不到 finalist entry，無法建立 ensemble member")
            params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
            member_payload = _build_seed_ensemble_member(
                member_index=member_index,
                seed=seed,
                best_trial=best_trial,
                finalist_entry=finalist_entry,
                params_payload=params_payload,
            )
            from tools.optimizer.outer_rolling_oos import build_optimizer_policy_members_from_finalists
            policy_members = build_optimizer_policy_members_from_finalists(
                finalists,
                objective_mode=str(task.get("objective_mode") or "split_train_romd"),
                member_index=int(member_index),
                seed=int(seed),
            )
            _emit_nonrolling_seed_process_progress_event(
                task,
                stage="DONE",
                status=f"seed {member_index}/{member_count} done",
                completed=int(requested_trials),
                total=int(requested_trials),
                best_base_score=_resolve_nonrolling_seed_done_best_base_score(policy_members, best_base_score),
                best_local_min_score=_resolve_nonrolling_seed_done_best_local_min_score(policy_members, member_payload.get("local_min_score")),
                elapsed_sec=max(0.0, time.perf_counter() - started_at),
            )
            return {"member": member_payload, "policy_members": policy_members}
        finally:
            member_session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    try:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as log_handle:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    return _execute()
        return _execute()
    except Exception as exc:
        if log_path:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
                traceback.print_exc(file=log_handle)
        raise RuntimeError(f"nonrolling seed member failed: seed={int((task or {}).get('seed', 0) or 0)} error={type(exc).__name__}: {exc}") from exc


def _print_static_seed_ensemble_result_table(*, members: list[dict], policy: dict, colors: dict) -> None:
    gray = colors.get("gray", "")
    green = colors.get("green", "")
    red = colors.get("red", "")
    yellow = colors.get("yellow", "")
    reset = colors.get("reset", "")
    member_count = len(members)
    min_agree = int(policy.get("min_agree", member_count or 1))
    title = "SEED ENSEMBLE RESULTS"
    header = (
        f"{'member':<8} | {'seed':>10} | {'trial':>8} | "
        f"{'base':>12} | {'local_min':>12} | {'retention':>10} | {'gate':>8} | {'result':>10}"
    )
    separator_width = max(len(title), len(header))
    print(f"{gray}{'-' * separator_width}{reset}")
    print(title)
    print(f"{gray}{'-' * separator_width}{reset}")
    print(header)
    print(f"{gray}{'-' * separator_width}{reset}")
    pass_count = 0
    for idx, item in enumerate(members, start=1):
        gate_pass = bool(item.get("local_gate", False))
        if gate_pass:
            pass_count += 1
        gate_text = "PASS" if gate_pass else "FAIL"
        gate_color = green if gate_pass else red
        result_text = "member"
        result_color = green if gate_pass else red
        print(
            f"#{int(item.get('member_index', idx)):<7} | "
            f"{int(item.get('seed', 0)):>10} | "
            f"#{int(item.get('selected_trial', 0)):>7} | "
            f"{_format_nonrolling_system_score(item.get('base_score')):>12} | "
            f"{_format_nonrolling_system_score(item.get('local_min_score')):>12} | "
            f"{_format_nonrolling_result_number(item.get('retention')):>10} | "
            f"{gate_color}{gate_text:>8}{reset} | "
            f"{result_color}{result_text:>10}{reset}"
        )
    base_scores = [float(item.get("base_score", 0.0)) for item in members]
    local_scores = [float(item.get("local_min_score", 0.0)) for item in members]
    retentions = [float(item.get("retention", 0.0)) for item in members]
    ensemble_pass = bool(member_count > 0 and pass_count >= min_agree)
    ensemble_color = green if ensemble_pass else red
    ensemble_result = "PASS" if ensemble_pass else "FAIL"
    print(f"{gray}{'=' * separator_width}{reset}")
    print(
        f"ENSEMBLE | N={member_count} | min_agree={min_agree} | "
        f"gate_pass={pass_count}/{member_count} | "
        f"base_min={_format_nonrolling_system_score(min(base_scores) if base_scores else None)} | "
        f"local_min_min={_format_nonrolling_system_score(min(local_scores) if local_scores else None)} | "
        f"retention_min={_format_nonrolling_result_number(min(retentions) if retentions else None)} | "
        f"result={ensemble_color}{ensemble_result}{reset}"
    )
    print(f"{yellow}static ensemble 已完成；Trade mode 產生 candidate_best/run_best，OOS mode 只產生 validation policy outputs。{reset}")


def _build_static_seed_ensemble_summary(*, members: list[dict], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int, policy_name: str = "") -> dict:
    local_scores = [float(item.get("local_min_score", 0.0)) for item in members]
    base_scores = [float(item.get("base_score", 0.0)) for item in members]
    retentions = [float(item.get("retention", 0.0)) for item in members]
    policy_contract = build_optimizer_effective_policy_fingerprint(walk_forward_policy)
    return {
        "schema_type": "optimizer_active_param_ensemble_summary",
        "mode": str(walk_forward_policy.get("model_mode", "")),
        "evaluation_scope": str(walk_forward_policy.get("evaluation_scope", "")),
        "policy_fingerprint_sha256": str(policy_contract["fingerprint_sha256"]),
        "policy_snapshot": policy_contract["snapshot"],
        "schema_version": 1,
        "action": "train",
        "selection_rule": "random_seed_ensemble_static_consensus",
        "objective_mode": str(objective_mode),
        "train_start_year": int(walk_forward_policy.get("train_start_year", 0)),
        "search_train_end_year": int(walk_forward_policy.get("search_train_end_year", 0)),
        "selection_end_year": int(walk_forward_policy.get("search_train_end_year", 0)),
        "train_start_date": walk_forward_policy.get("train_start_date"),
        "search_train_end_date": walk_forward_policy.get("search_train_end_date"),
        "latest_data_date": walk_forward_policy.get("latest_data_date"),
        "train_window_months": walk_forward_policy.get("trade_train_window_months") or walk_forward_policy.get("train_window_months"),
        "oos_start_year": walk_forward_policy.get("oos_start_year"),
        "oos_start_date": walk_forward_policy.get("oos_start_date"),
        "oos_end_date": walk_forward_policy.get("oos_end_date"),
        "dataset_label": str(dataset_label),
        "selected_model_mode": str(selected_model_mode),
        "trials_per_seed": int(trials_per_seed),
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "base_score": min(base_scores) if base_scores else 0.0,
        "local_min_score": min(local_scores) if local_scores else 0.0,
        "retention": min(retentions) if retentions else 0.0,
        "local_gate": all(bool(item.get("local_gate", False)) for item in members),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
        "member_metrics": [
            {
                "member_index": int(item.get("member_index", idx + 1)),
                "seed": int(item.get("seed", 0)),
                "selected_trial": int(item.get("selected_trial", 0)),
                "base_score": float(item.get("base_score", 0.0)),
                "local_min_score": float(item.get("local_min_score", 0.0)),
                "retention": float(item.get("retention", 0.0)),
                "local_gate": bool(item.get("local_gate", False)),
                "local_min_review_enabled": bool(item.get("local_min_review_enabled", is_optimizer_local_min_review_enabled())),
                "local_min_review_mode": str(item.get("local_min_review_mode", "exact")),
            }
            for idx, item in enumerate(members)
        ],
        "random_seed_ensemble": _resolve_nonrolling_policy_seed_ensemble_policy(policy_name=str(policy_name or ""), members=list(members or []), seeds=seeds),
        "created_at": get_taipei_now().isoformat(),
    }


def _build_static_seed_ensemble_meta(*, seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int, source: str, policy_name: str = "") -> dict:
    meta = {
        "source": str(source),
        "dataset_label": str(dataset_label),
        "selected_model_mode": str(selected_model_mode),
        "objective_mode": str(objective_mode),
        "trials_per_seed": int(trials_per_seed),
        "seeds": [int(seed) for seed in seeds],
        "walk_forward_policy": dict(walk_forward_policy),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
    }
    if policy_name:
        meta["policy"] = str(policy_name)
    return meta


def _build_static_seed_ensemble_policy_paramset_payload(*, policy_name: str, members: list[dict], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int) -> dict:
    normalized_members = renumber_seed_ensemble_members(list(members or []))
    requested_policy = _resolve_nonrolling_policy_seed_ensemble_policy(
        policy_name=str(policy_name),
        members=normalized_members,
        seeds=seeds,
    )
    payload = build_static_active_param_ensemble_payload(
        members=normalized_members,
        random_seed_ensemble=requested_policy,
        selector=str(policy_name),
        created_at=get_taipei_now().isoformat(),
        meta=_build_static_seed_ensemble_meta(
            seeds=seeds,
            objective_mode=objective_mode,
            walk_forward_policy=walk_forward_policy,
            dataset_label=dataset_label,
            selected_model_mode=selected_model_mode,
            trials_per_seed=trials_per_seed,
            source="nonrolling_random_seed_ensemble_policy_paramset",
            policy_name=str(policy_name),
        ),
        raw_universe_required_min_rows=resolve_raw_universe_required_min_rows(walk_forward_policy),
    )
    selection_start = str(walk_forward_policy.get("train_start_date") or f"{int(walk_forward_policy.get('train_start_year', 0) or 0):04d}-01-01")
    selection_end = str(walk_forward_policy.get("search_train_end_date") or f"{int(walk_forward_policy.get('search_train_end_year', 0) or 0):04d}-12-31")
    oos_start_raw = walk_forward_policy.get("oos_start_date")
    if not oos_start_raw and int(walk_forward_policy.get("oos_start_year", 0) or 0) > 0:
        oos_start_raw = f"{int(walk_forward_policy.get('oos_start_year', 0) or 0):04d}-01-01"
    oos_end_raw = walk_forward_policy.get("oos_end_date") or "latest"
    payload["summary"] = {
        "folds": 1,
        "mode": "static",
        "selector": str(policy_name),
        "selection_period": f"{selection_start}~{selection_end}",
        "oos_period": f"{oos_start_raw}~{oos_end_raw}" if oos_start_raw else "",
        "member_count": int(len(normalized_members)),
        "requested_seed_count": int(len(seeds)),
        "trials_per_seed": int(trials_per_seed),
        "local_min_review_enabled": bool(is_optimizer_local_min_review_enabled()),
    }
    return payload


def _remove_disabled_nonrolling_policy_paramset_files(*, mode: str, active_policy_names: set[str], all_policy_names: tuple[str, ...]) -> None:
    from core.strategy_param_artifacts import resolve_strategy_param_artifact_path

    normalized_mode = normalize_optimizer_model_mode(mode)
    active = {str(name) for name in set(active_policy_names or set())}
    for policy_name in tuple(str(name) for name in all_policy_names if str(name)):
        if policy_name in active:
            continue
        path = str(resolve_strategy_param_artifact_path(
            PROJECT_ROOT, family="full", evaluation_mode=normalized_mode, policy=policy_name
        ))
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f"{C_YELLOW}⚠️ 無法移除 disabled policy 檔：{_project_relative_path(path)}｜{type(exc).__name__}: {exc}{C_RESET}")


def _write_static_seed_ensemble_policy_paramsets(*, policy_members_by_policy: dict[str, list[dict]], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int, write_files: bool = True) -> tuple[dict[str, str], dict[str, dict]]:
    from tools.optimizer.outer_rolling_oos import (
        ALL_REPORT_POLICY_NAMES,
        get_optimizer_paramset_policy_names,
        get_optimizer_nonrolling_policy_paramset_filename,
        select_finalists_agree_members,
        select_finalist_best_members,
        _is_finalist_best_policy,
        _remove_stale_policy_paramset_files,
    )

    first_class_policy_names = tuple(get_optimizer_paramset_policy_names())
    replay_policy_names = first_class_policy_names
    _remove_stale_policy_paramset_files(MODELS_DIR)
    paths: dict[str, str] = {}
    payloads: dict[str, dict] = {}
    first_class_policy_set = set(first_class_policy_names)
    if bool(write_files):
        _remove_disabled_nonrolling_policy_paramset_files(
            mode=selected_model_mode,
            active_policy_names=first_class_policy_set,
            all_policy_names=tuple(ALL_REPORT_POLICY_NAMES),
        )
    for policy_name in replay_policy_names:
        members = sorted(
            list((policy_members_by_policy or {}).get(str(policy_name)) or []),
            key=lambda item: int(dict(item).get("member_index", 0) or 0),
        )
        if _is_finalist_best_policy(policy_name):
            members = select_finalist_best_members(members, policy_name=str(policy_name))
        elif _is_finalists_agree_policy_name(policy_name):
            members = select_finalists_agree_members(members, policy_name=str(policy_name))
        if not members:
            continue
        payload = _build_static_seed_ensemble_policy_paramset_payload(
            policy_name=str(policy_name),
            members=members,
            seeds=seeds,
            objective_mode=objective_mode,
            walk_forward_policy=walk_forward_policy,
            dataset_label=dataset_label,
            selected_model_mode=selected_model_mode,
            trials_per_seed=trials_per_seed,
        )
        payloads[str(policy_name)] = payload
        from core.strategy_param_artifacts import resolve_strategy_param_artifact_path

        path = str(resolve_strategy_param_artifact_path(
            PROJECT_ROOT,
            family="full",
            evaluation_mode=normalize_optimizer_model_mode(selected_model_mode),
            policy=str(policy_name),
        ))
        if str(policy_name) not in first_class_policy_set:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"{C_YELLOW}⚠️ 無法移除舊 threshold policy 檔：{_project_relative_path(path)}｜{type(exc).__name__}: {exc}{C_RESET}")
            continue
        if bool(write_files):
            _write_json_file(path, payload)
            paths[str(policy_name)] = path
    if bool(write_files) and paths:
        from services.optimizer.strategy_param_repository import refresh_strategy_parameter_manifest

        refresh_strategy_parameter_manifest(
            PROJECT_ROOT,
            family="full",
            evaluation_mode=normalize_optimizer_model_mode(selected_model_mode),
        )
    return paths, payloads


def _write_static_seed_ensemble_candidate(*, members: list[dict], seeds: list[int], objective_mode: str, walk_forward_policy: dict, dataset_label: str, selected_model_mode: str, trials_per_seed: int, policy_members_by_policy: dict[str, list[dict]] | None = None, write_candidate_best: bool = True, write_policy_files: bool = True) -> tuple[dict, dict, dict[str, dict]]:
    from tools.optimizer.outer_rolling_oos import select_finalist_best_members, select_finalists_agree_members, _is_finalist_best_policy

    candidate_selector = _resolve_trade_candidate_selector() if normalize_optimizer_model_mode(selected_model_mode) == "trade" else "candidate_best"
    candidate_members = list(members)
    if candidate_selector != "candidate_best":
        selector_members = list((policy_members_by_policy or {}).get(candidate_selector) or [])
        if selector_members:
            candidate_members = sorted(selector_members, key=lambda item: int(dict(item).get("member_index", 0) or 0))
            if _is_finalist_best_policy(candidate_selector):
                candidate_members = select_finalist_best_members(candidate_members, policy_name=str(candidate_selector))
            elif _is_finalists_agree_policy_name(candidate_selector):
                candidate_members = select_finalists_agree_members(candidate_members, policy_name=str(candidate_selector))
    candidate_members = renumber_seed_ensemble_members(candidate_members)
    policy = _resolve_nonrolling_policy_seed_ensemble_policy(
        policy_name=str(candidate_selector),
        members=candidate_members,
        seeds=seeds,
    )
    payload = build_static_active_param_ensemble_payload(
        members=candidate_members,
        random_seed_ensemble=policy,
        selector=str(candidate_selector),
        created_at=get_taipei_now().isoformat(),
        meta=_build_static_seed_ensemble_meta(
            seeds=seeds,
            objective_mode=objective_mode,
            walk_forward_policy=walk_forward_policy,
            dataset_label=dataset_label,
            selected_model_mode=selected_model_mode,
            trials_per_seed=trials_per_seed,
            source="nonrolling_random_seed_ensemble",
            policy_name=str(candidate_selector),
        ),
        raw_universe_required_min_rows=resolve_raw_universe_required_min_rows(walk_forward_policy),
    )
    policy_paramset_paths, policy_paramset_payloads = _write_static_seed_ensemble_policy_paramsets(
        policy_members_by_policy=dict(policy_members_by_policy or {}),
        seeds=seeds,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=trials_per_seed,
        write_files=bool(write_policy_files),
    )
    summary = _build_static_seed_ensemble_summary(
        members=candidate_members,
        seeds=seeds,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=trials_per_seed,
        policy_name=str(candidate_selector),
    )
    summary["selector"] = str(candidate_selector)
    summary["run_best_selector"] = _resolve_trade_run_best_selector()
    summary["policy_paramsets"] = dict(policy_paramset_paths)
    payload["summary"] = dict(summary)
    if bool(write_candidate_best):
        _write_json_file(CANDIDATE_BEST_PARAMS_PATH, payload)
        try:
            if os.path.exists(CANDIDATE_BEST_SUMMARY_PATH):
                os.remove(CANDIDATE_BEST_SUMMARY_PATH)
        except OSError as exc:
            print(f"{C_YELLOW}⚠️ 無法移除舊 candidate_best summary sidecar：{_project_relative_path(CANDIDATE_BEST_SUMMARY_PATH)}｜{type(exc).__name__}: {exc}{C_RESET}")
    return payload, summary, policy_paramset_payloads




def _resolve_trial_base_score(trial) -> float:
    attrs = getattr(trial, "user_attrs", {}) or {}
    value = attrs.get("base_score", getattr(trial, "value", INVALID_TRIAL_VALUE))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(INVALID_TRIAL_VALUE)


def _remove_study_full_non_base_policy_outputs() -> None:
    from tools.optimizer.outer_rolling_oos import (
        get_optimizer_nonrolling_policy_paramset_filename,
        get_optimizer_paramset_policy_names,
    )

    for policy_name in get_optimizer_paramset_policy_names():
        if str(policy_name) == "base":
            continue
        path = os.path.join(
            MODELS_DIR,
            get_optimizer_nonrolling_policy_paramset_filename(str(policy_name), mode="study"),
        )
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(
                f"{C_YELLOW}⚠️ Study-Full 無法移除非 base 舊檔："
                f"{_project_relative_path(path)}｜{type(exc).__name__}: {exc}{C_RESET}"
            )


def _finalize_single_seed_study_base_only_outputs(
    *,
    best_trial,
    optimizer_seed: int,
    objective_mode: str,
    walk_forward_policy: dict,
    dataset_label: str,
    selected_model_mode: str,
    trials_per_seed: int,
    build_best_params_payload_from_trial,
    dashboard_session=None,
) -> int:
    from tools.optimizer.outer_rolling_oos import get_optimizer_nonrolling_policy_paramset_filename

    base_score = _resolve_trial_base_score(best_trial)
    params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
    seed_value = int(optimizer_seed)
    member = {
        "member_index": 1,
        "seed": seed_value,
        "selected_trial": int(best_trial.number) + 1,
        "params": dict(params_payload),
        "score": float(base_score),
        "base_score": float(base_score),
    }
    base_payload = _build_static_seed_ensemble_policy_paramset_payload(
        policy_name="base",
        members=[member],
        seeds=[seed_value],
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=int(trials_per_seed),
    )
    if isinstance(base_payload.get("meta"), dict):
        base_payload["meta"].update({
            "base_only": True,
            "local_min_review_enabled": False,
            "local_min_review_skipped_reason": "Study-Full only exports base.",
        })
    base_payload.setdefault("summary", {})
    base_payload["summary"].update({
        "selected_policy": "base",
        "study_scope": "full",
        "base_only": True,
        "local_min_review_enabled": False,
        "selected_trial": int(best_trial.number) + 1,
        "seed": seed_value,
        "base_score": float(base_score),
    })
    base_path = os.path.join(
        MODELS_DIR,
        get_optimizer_nonrolling_policy_paramset_filename("base", mode=selected_model_mode),
    )
    _remove_study_full_non_base_policy_outputs()
    _write_json_file(base_path, base_payload)
    print(
        f"{C_GREEN}✅ Study-Full base 完成｜"
        f"trial=#{int(best_trial.number) + 1}｜base={format_optimizer_score_for_display(base_score, decimals=3)}{C_RESET}"
    )

    # AI註: Study-Full 訓練中的 milestone dashboard 來自 trial user_attrs，
    # 主要用於快速觀察進化中的候選；Workbench 則會讀取匯出的 base.json
    # 重新 replay。為避免最後輸出的 Optimizer 畫面與 Workbench 口徑分叉，
    # Study-Full 匯出後也立刻用同一份 base_payload 走 active-param ensemble replay。
    if dashboard_session is not None and getattr(dashboard_session, "raw_data_cache_data_dir", None):
        try:
            from tools.optimizer.static_ensemble_dashboard import print_optimizer_static_ensemble_console_dashboard

            print_optimizer_static_ensemble_console_dashboard(
                dashboard_session,
                ensemble_payload=base_payload,
                seeds=[seed_value],
                milestone_title="🏆 Study-Full base 匯出後 replay 詳細結果",
                title="Study-Full base 匯出後 replay 績效與風險對比表",
                force=True,
            )
        except Exception as exc:
            print(
                f"{C_YELLOW}⚠️ Study-Full base 匯出後 replay 顯示略過："
                f"{type(exc).__name__}: {exc}{C_RESET}"
            )

    _print_optimizer_output_files("💾 輸出檔案", [("base", base_path)])
    return 0





def _finalize_single_seed_study_outputs(
    *,
    session,
    finalists: list[dict],
    best_trial,
    optimizer_seed: int,
    objective_mode: str,
    walk_forward_policy: dict,
    dataset_label: str,
    selected_model_mode: str,
    trials_per_seed: int,
    build_best_params_payload_from_trial,
    elapsed_sec: float | None = None,
) -> int:
    from tools.optimizer.static_ensemble_dashboard import print_optimizer_static_ensemble_rolling_oos_table
    from tools.optimizer.outer_rolling_oos import build_optimizer_policy_members_from_finalists

    finalist_entry = _find_finalist_entry(finalists, best_trial)
    if finalist_entry is None:
        print(f"{C_YELLOW}ℹ️ Study mode 完成，但找不到 winner finalist entry，略過 validation policy output。{C_RESET}")
        return 0
    params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
    seed_value = int(optimizer_seed)
    member = _build_seed_ensemble_member(
        member_index=1,
        seed=seed_value,
        best_trial=best_trial,
        finalist_entry=finalist_entry,
        params_payload=params_payload,
    )
    policy_members = build_optimizer_policy_members_from_finalists(
        list(finalists or []),
        objective_mode=objective_mode,
        member_index=1,
        seed=seed_value,
    )
    policy_members_by_policy: dict[str, list[dict]] = {}
    for name, member_payload in dict(policy_members or {}).items():
        if isinstance(member_payload, list):
            policy_members_by_policy[str(name)] = [dict(item) for item in member_payload if isinstance(item, dict)]
        elif isinstance(member_payload, dict):
            policy_members_by_policy[str(name)] = [dict(member_payload)]
    ensemble_payload, ensemble_summary, policy_paramset_payloads = _write_static_seed_ensemble_candidate(
        members=[member],
        seeds=[seed_value],
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=int(trials_per_seed),
        policy_members_by_policy=policy_members_by_policy,
        write_candidate_best=False,
        write_policy_files=True,
    )
    print_optimizer_static_ensemble_rolling_oos_table(
        session,
        ensemble_payload=ensemble_payload,
        elapsed_sec=elapsed_sec,
        policy_paramsets=dict(policy_paramset_payloads or {}),
    )
    visible_policy_paths = _visible_policy_paramset_paths(dict((ensemble_summary or {}).get("policy_paramsets") or {}))
    _print_optimizer_output_files("💾 輸出檔案", list(visible_policy_paths.items()))
    return 0


def _tail_nonrolling_seed_log(path: str, *, max_lines: int = 24) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-int(max_lines):]).strip()
    except OSError:
        return ""


def _run_nonrolling_random_seed_ensemble_training(
    *,
    environ,
    selected_data_dir: str,
    dataset_profile_key: str,
    dataset_label: str,
    selected_model_mode: str,
    load_all_raw_data: bool,
    optimizer_required_min_rows: int,
    objective_mode: str,
    walk_forward_policy: dict,
    requested_trials: int,
    build_optimizer_session,
    create_optimizer_study,
    ensure_study_effective_policy_compatible,
    configure_optuna_logging,
    print_local_min_score_finalist_review,
    print_local_min_score_winner_summary,
    close_study_storage,
    is_qualified_trial_value,
    resolve_optimizer_single_fold_search_parallel_trials,
    build_best_params_payload_from_trial,
):
    if int(requested_trials) <= 0:
        return None
    if not os.path.isdir(selected_data_dir):
        raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
    policy = _resolve_nonrolling_seed_ensemble_policy()
    if not bool(policy.get("enabled", False)) or int(policy.get("seed_count", 1)) <= 1:
        return None

    seeds = generate_random_seed_ensemble(int(policy["seed_count"]))
    compact_display = True

    members: list[dict] = []
    policy_members_by_policy: dict[str, list[dict]] = {}
    dashboard_session = None
    ensemble_started_at = time.perf_counter()
    from tools.optimizer.outer_rolling_oos import _ResourceUsageSampler, _resolve_resource_sample_interval_sec
    resource_sampler = _ResourceUsageSampler(interval_sec=_resolve_resource_sample_interval_sec(environ))
    resource_sampler.start()
    parallel_workers = resolve_optimizer_random_seed_ensemble_parallel_workers_default(len(seeds))
    parallel_backend = resolve_optimizer_random_seed_ensemble_parallel_backend_default()
    process_parallel_enabled = bool(int(parallel_workers) > 1 and parallel_backend == "process")
    process_log_dir = os.path.join(OUTPUT_DIR, "seed_ensemble_logs", get_taipei_now().strftime("%Y%m%d_%H%M%S_%f"))
    process_log_paths: dict[int, str] = {}
    progress_lock = threading.Lock()
    progress_board = None
    def _collect_policy_members(policy_members: dict | None) -> None:
        for policy_name, member in dict(policy_members or {}).items():
            if isinstance(member, list):
                for item in member:
                    if isinstance(item, dict):
                        policy_members_by_policy.setdefault(str(policy_name), []).append(dict(item))
                continue
            if not isinstance(member, dict):
                continue
            policy_members_by_policy.setdefault(str(policy_name), []).append(dict(member))

    if compact_display:
        from tools.optimizer.outer_rolling_oos import OptimizerSeedEnsembleProgressBoard, format_optimizer_seed_ensemble_progress_header

        period_context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
        contexts = []
        for member_index, seed in enumerate(seeds, start=1):
            context = dict(period_context)
            context.update({
                "seed_index": int(member_index),
                "seed_count": int(len(seeds)),
                "seed": int(seed),
                "log_path": os.path.join(process_log_dir, f"seed_{int(member_index):02d}_{int(seed)}.log"),
                "selection_start": str(period_context.get("selection_start") or ""),
                "selection_end": str(period_context.get("selection_end") or ""),
                "oos_period": str(period_context.get("oos_period") or ""),
            })
            contexts.append(context)
        def _format_nonrolling_seed_header(board):
            completed = board.get_completed_fold_count() if hasattr(board, "get_completed_fold_count") else 0
            return format_optimizer_seed_ensemble_progress_header(
                folds=1,
                seeds=len(seeds),
                min_agree=int(policy["min_agree"]),
                parallel_workers=int(parallel_workers),
                backend=parallel_backend,
                completed_folds=int(completed),
                total_elapsed_sec=max(0.0, time.perf_counter() - float(ensemble_started_at)),
                completed_trials=board.get_completed_trial_count() if hasattr(board, "get_completed_trial_count") else 0,
                search_wall_elapsed_sec=board.get_search_wall_elapsed_sec() if hasattr(board, "get_search_wall_elapsed_sec") else None,
                completed_local_min_trials=board.get_completed_local_min_trial_count() if hasattr(board, "get_completed_local_min_trial_count") else 0,
                local_min_wall_elapsed_sec=board.get_local_min_wall_elapsed_sec() if hasattr(board, "get_local_min_wall_elapsed_sec") else None,
                completed_replays=board.get_completed_replay_count() if hasattr(board, "get_completed_replay_count") else 0,
                replay_wall_elapsed_sec=board.get_replay_wall_elapsed_sec() if hasattr(board, "get_replay_wall_elapsed_sec") else None,
            )

        progress_board = OptimizerSeedEnsembleProgressBoard(
            contexts,
            header_factory=_format_nonrolling_seed_header,
        )
        progress_board.render(force=True)
    else:
        from tools.optimizer.outer_rolling_oos import format_optimizer_seed_ensemble_progress_header
        print(f"{C_GRAY}{format_optimizer_seed_ensemble_progress_header(folds=1, seeds=len(seeds), min_agree=int(policy['min_agree']), parallel_workers=int(parallel_workers), backend=parallel_backend, completed_folds=0, total_elapsed_sec=0.0, completed_trials=0)}{C_RESET}")
    configure_optuna_logging()

    def _emit_seed_progress_for_member(member_index: int, seed: int, **kwargs) -> None:
        if compact_display and progress_board is not None:
            progress = dict(kwargs)
            progress.setdefault("seed_ensemble_member_index", int(member_index))
            progress.setdefault("seed_ensemble_member_count", int(len(seeds)))
            progress.setdefault("seed", int(seed))
            with progress_lock:
                progress_board.update(fold_idx=1, seed_index=int(member_index), progress=progress)
            return
        _print_nonrolling_single_fold_progress(walk_forward_policy, **kwargs)

    def _run_one_seed_member(member_index: int, seed: int) -> dict | None:
        member_session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        member_session.n_trials = int(requested_trials)
        member_session.run_action = "train"
        member_session.disable_milestone_dashboard = compact_display
        member_session.disable_optimizer_status_line = compact_display
        study = None
        try:
            if compact_display:
                _emit_seed_progress_for_member(member_index, int(seed),
                    stage="START",
                    status=f"seed {member_index}/{len(seeds)}",
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
            else:
                print(f"{C_CYAN}[seed {member_index}/{len(seeds)}] trials={int(requested_trials)}{C_RESET}")
            study = create_optimizer_study(None, seed=int(seed), sampler_kind="tpe")
            ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
            if compact_display:
                _emit_seed_progress_for_member(member_index, int(seed),
                    stage="RAW_DATA",
                    status=f"seed {member_index}/{len(seeds)} raw data",
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
            member_session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=False,
            )
            member_session.profile_recorder.init_output_files()
            member_session.profile_recorder.mark_run_started()
            search_parallel_trials = resolve_optimizer_single_fold_search_parallel_trials(environ, sampler_kind="tpe")
            search_callbacks = [member_session.monitoring_callback]
            # 多 seed 並行時避免多個 inline trial progress 同時爭用 stdout；仍保留 seed-level rolling 同源進度。
            emit_trial_progress = compact_display
            search_started_perf = time.perf_counter()
            search_started_ts = time.time()
            if emit_trial_progress:
                progress_status = f"seed {member_index}/{len(seeds)}" if int(parallel_workers) > 1 else ""
                _emit_seed_progress_for_member(
                    member_index,
                    int(seed),
                    stage="OPTIMIZER_SEARCH",
                    status=progress_status,
                    completed=0,
                    total=int(requested_trials),
                    elapsed_sec=0.0,
                    search_started_ts=search_started_ts,
                )
                search_callbacks.append(
                    _make_nonrolling_seed_trial_progress_callback(
                        member_session=member_session,
                        walk_forward_policy=walk_forward_policy,
                        requested_trials=int(requested_trials),
                        started_at=search_started_perf,
                        search_started_ts=search_started_ts,
                        status=progress_status,
                        inline=False if progress_board is not None else bool(int(parallel_workers) <= 1),
                        emit_lock=progress_lock,
                        throttle_trials=1,
                        throttle_seconds=0.0,
                        progress_board=progress_board,
                        seed_index=int(member_index),
                        seed=int(seed),
                        seed_count=int(len(seeds)),
                    )
                )
            study.optimize(
                member_session.objective,
                n_trials=int(requested_trials),
                n_jobs=int(search_parallel_trials),
                callbacks=search_callbacks,
            )
            search_done_ts = time.time()
            if compact_display:
                _finalize_nonrolling_single_fold_inline_progress(member_session)
            best_base_score = _resolve_study_best_base_score(member_session, study)
            local_progress_installed = False
            if compact_display and progress_board is not None:
                local_progress_installed = _install_nonrolling_local_min_seed_board_progress(
                    session=member_session,
                    walk_forward_policy=walk_forward_policy,
                    progress_board=progress_board,
                    progress_lock=progress_lock,
                    member_index=int(member_index),
                    member_count=int(len(seeds)),
                    seed=int(seed),
                    best_base_score=best_base_score,
                    started_at=ensemble_started_at,
                    completed_trials=int(requested_trials),
                    search_started_ts=search_started_ts,
                    search_last_done_ts=search_done_ts,
                )
            elif compact_display:
                member_session.outer_rolling_local_progress_context = _make_nonrolling_local_min_progress_context(
                    walk_forward_policy,
                    best_base_score=best_base_score,
                    overall_start=ensemble_started_at,
                )
            try:
                finalists, best_trial = print_local_min_score_finalist_review(
                    study,
                    session=member_session,
                    objective_mode=objective_mode,
                    colors=COLORS,
                    winner_trial=None,
                    emit_table=False,
                    show_progress=True,
                )
            finally:
                if compact_display:
                    _finalize_nonrolling_single_fold_inline_progress(member_session)
                if local_progress_installed or compact_display:
                    _clear_nonrolling_local_min_progress_hooks(member_session)
            if best_trial is None or not is_qualified_trial_value(best_trial.value):
                print(f"{C_YELLOW}ℹ️ seed {member_index}/{len(seeds)} 目前尚無通過 local_min_score gate 的 winner；本次不建立完整 N-seed ensemble member。{C_RESET}")
                return None
            print_local_min_score_winner_summary(
                winner_trial=best_trial,
                session=member_session,
                colors=COLORS,
            )
            finalist_entry = _find_finalist_entry(finalists, best_trial)
            if finalist_entry is None:
                raise ValueError(f"seed={int(seed)} 找不到 finalist entry，無法建立 ensemble member")
            params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT)
            member_payload = _build_seed_ensemble_member(
                member_index=member_index,
                seed=int(seed),
                best_trial=best_trial,
                finalist_entry=finalist_entry,
                params_payload=params_payload,
            )
            from tools.optimizer.outer_rolling_oos import build_optimizer_policy_members_from_finalists
            policy_members = build_optimizer_policy_members_from_finalists(
                finalists,
                objective_mode=objective_mode,
                member_index=int(member_index),
                seed=int(seed),
            )
            if compact_display:
                _emit_seed_progress_for_member(member_index, int(seed),
                    stage="DONE",
                    status=f"seed {member_index}/{len(seeds)} done",
                    completed=int(requested_trials),
                    total=int(requested_trials),
                    best_base_score=_resolve_nonrolling_seed_done_best_base_score(policy_members, best_base_score),
                    best_local_min_score=_resolve_nonrolling_seed_done_best_local_min_score(policy_members, member_payload.get("local_min_score")),
                    elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                )
            return {"member": member_payload, "session": member_session, "policy_members": policy_members}
        finally:
            member_session.close_trial_prep_executor()
            if study is not None:
                close_study_storage(study)

    def _refresh_process_progress_board(*, force: bool = False) -> None:
        if not compact_display or progress_board is None or not process_log_paths:
            return
        from tools.optimizer.outer_rolling_oos import read_optimizer_seed_progresses_from_log_paths

        latest = read_optimizer_seed_progresses_from_log_paths(process_log_paths.values())
        if not latest:
            return
        progress_map = {(1, int(member_index)): dict(progress) for member_index, progress in latest.items()}
        with progress_lock:
            progress_board.update_many(progress_map, force=force)

    if int(parallel_workers) <= 1:
        for member_index, seed in enumerate(seeds, start=1):
            result = _run_one_seed_member(member_index, int(seed))
            if result and result.get("member"):
                members.append(dict(result["member"]))
                _collect_policy_members(result.get("policy_members"))
                dashboard_session = result.get("session") or dashboard_session
    elif process_parallel_enabled:
        os.makedirs(process_log_dir, exist_ok=True)
        process_tasks = {}
        process_context = _build_nonrolling_single_fold_period_context(walk_forward_policy)
        for member_index, seed in enumerate(seeds, start=1):
            log_path = os.path.join(process_log_dir, f"seed_{int(member_index):02d}_{int(seed)}.log")
            process_log_paths[int(member_index)] = log_path
            process_tasks[int(member_index)] = {
                "member_index": int(member_index),
                "member_count": int(len(seeds)),
                "seed": int(seed),
                "walk_forward_policy": dict(walk_forward_policy),
                "selected_data_dir": str(selected_data_dir),
                "optimizer_required_min_rows": int(optimizer_required_min_rows),
                "objective_mode": str(objective_mode),
                "requested_trials": int(requested_trials),
                "db_name": None,
                "environ": dict(environ or {}),
                "log_path": log_path,
                "progress_context": {
                    "fold_idx": 1,
                    "fold_count": 1,
                    "oos_year": int(process_context.get("oos_year", 0) or 0),
                    "oos_period": str(process_context.get("oos_period") or ""),
                    "selection_start": str(process_context.get("selection_start") or ""),
                    "selection_end": str(process_context.get("selection_end") or ""),
                    "selection_period": str(process_context.get("selection_period") or ""),
                },
            }
        executor_kwargs, _start_method = get_process_pool_executor_kwargs()
        with ProcessPoolExecutor(max_workers=int(parallel_workers), **executor_kwargs) as executor:
            future_map = {
                executor.submit(_run_nonrolling_seed_ensemble_member_process_task, task): int(member_index)
                for member_index, task in process_tasks.items()
            }
            pending = set(future_map)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                _refresh_process_progress_board(force=bool(done))
                for future in done:
                    member_index = int(future_map[future])
                    try:
                        result = future.result()
                    except Exception as exc:
                        tail = _tail_nonrolling_seed_log(process_log_paths.get(member_index, ""), max_lines=30)
                        detail = f"\n最後 seed log：\n{tail}" if tail else ""
                        raise RuntimeError(f"nonrolling seed process failed: member={member_index}/{len(seeds)} error={type(exc).__name__}: {exc}{detail}") from exc
                    if result and result.get("member"):
                        members.append(dict(result["member"]))
                        _collect_policy_members(result.get("policy_members"))
                    elif compact_display:
                        _emit_seed_progress_for_member(member_index, int(seeds[member_index - 1]),
                            stage="DONE",
                            status=f"seed {member_index}/{len(seeds)} skipped",
                            elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                        )
            _refresh_process_progress_board(force=True)
        members.sort(key=lambda item: int(item.get("member_index", 0) or 0))
    else:
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            future_map = {
                executor.submit(_run_one_seed_member, int(member_index), int(seed)): (int(member_index), int(seed))
                for member_index, seed in enumerate(seeds, start=1)
            }
            pending = set(future_map)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    member_index, seed = future_map[future]
                    result = future.result()
                    if result and result.get("member"):
                        members.append(dict(result["member"]))
                        _collect_policy_members(result.get("policy_members"))
                        dashboard_session = result.get("session") or dashboard_session
                    elif compact_display:
                        _emit_seed_progress_for_member(member_index, int(seed),
                            stage="DONE",
                            status=f"seed {member_index}/{len(seeds)} skipped",
                            elapsed_sec=max(0.0, time.perf_counter() - ensemble_started_at),
                        )
        members.sort(key=lambda item: int(item.get("member_index", 0) or 0))

    if len(members) != len(seeds):
        resource_sampler.stop()
        if compact_display and progress_board is not None:
            progress_board.close()
        print(
            f"{C_YELLOW}ℹ️ 非 rolling random seed ensemble 未建立 candidate："
            f"可用 members={len(members)}/{len(seeds)}，不輸出不完整 ensemble。{C_RESET}"
        )
        return 0

    ensemble_payload, _ensemble_summary, _policy_paramset_payloads = _write_static_seed_ensemble_candidate(
        members=members,
        seeds=seeds,
        objective_mode=objective_mode,
        walk_forward_policy=walk_forward_policy,
        dataset_label=dataset_label,
        selected_model_mode=selected_model_mode,
        trials_per_seed=int(requested_trials),
        policy_members_by_policy=policy_members_by_policy,
        write_candidate_best=(normalize_optimizer_model_mode(selected_model_mode) == "trade"),
        # base/base_r/local/retention are selector-study artifacts and must remain
        # available in every non-rolling mode. Only candidate_best/run_best are
        # restricted to Trade mode.
        write_policy_files=True,
    )
    trade_mode = normalize_optimizer_model_mode(selected_model_mode) == "trade"
    if dashboard_session is None:
        dashboard_session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        dashboard_session.load_raw_data(
            selected_data_dir,
            load_all_raw_data=load_all_raw_data,
            required_min_rows=optimizer_required_min_rows,
            verbose=False,
        )
    from tools.optimizer.static_ensemble_dashboard import (
        build_optimizer_static_ensemble_single_fold_oos_row,
        print_optimizer_static_ensemble_console_dashboard,
        print_optimizer_static_ensemble_rolling_oos_table,
    )
    if compact_display and progress_board is not None:
        period_context = _build_nonrolling_single_fold_period_context(walk_forward_policy)

        def _nonrolling_replay_progress(event: dict) -> None:
            progress = dict(event or {})
            progress.setdefault("fold_idx", 1)
            progress.setdefault("fold_count", 1)
            progress.setdefault("oos_year", int(period_context.get("oos_year", 0) or 0))
            progress.setdefault("selection_start", str(period_context.get("selection_start") or ""))
            progress.setdefault("selection_end", str(period_context.get("selection_end") or ""))
            progress.setdefault("show_oos", bool(period_context.get("show_oos", True)))
            with progress_lock:
                progress_board.update_fold_progress(fold_idx=1, progress=progress)

        oos_row = build_optimizer_static_ensemble_single_fold_oos_row(
            dashboard_session,
            ensemble_payload=ensemble_payload,
            elapsed_sec=max(0.0, time.perf_counter() - float(ensemble_started_at)),
            policy_paramsets=dict(_policy_paramset_payloads or {}),
            progress_callback=_nonrolling_replay_progress,
        )
        with progress_lock:
            progress_board.update_result_row(oos_row, force=True)
            progress_board.close()
        if trade_mode:
            print_optimizer_static_ensemble_console_dashboard(
                dashboard_session,
                ensemble_payload=ensemble_payload,
                seeds=seeds,
                milestone_title="🏆 candidate_best 詳細結果",
                title="candidate_best 詳細結果表格",
                force=True,
            )
    else:
        print_optimizer_static_ensemble_rolling_oos_table(
            dashboard_session,
            ensemble_payload=ensemble_payload,
            elapsed_sec=max(0.0, time.perf_counter() - float(ensemble_started_at)),
            policy_paramsets=dict(_policy_paramset_payloads or {}),
        )
        if trade_mode:
            print_optimizer_static_ensemble_console_dashboard(
                dashboard_session,
                ensemble_payload=ensemble_payload,
                seeds=seeds,
                milestone_title="🏆 candidate_best 詳細結果",
                title="candidate_best 詳細結果表格",
                force=True,
            )
    resource_sampler.stop()
    from tools.optimizer.outer_rolling_oos import format_optimizer_final_performance_summary

    if progress_board is not None:
        completed_trials = progress_board.get_completed_trial_count()
        search_wall_elapsed_sec = progress_board.get_search_wall_elapsed_sec()
        completed_local_min_trials = progress_board.get_completed_local_min_trial_count()
        local_min_wall_elapsed_sec = progress_board.get_local_min_wall_elapsed_sec()
        completed_replays = progress_board.get_completed_replay_count()
        replay_wall_elapsed_sec = progress_board.get_replay_wall_elapsed_sec()
        completed_folds = progress_board.get_completed_fold_count()
    else:
        completed_trials = int(requested_trials) * int(len(seeds))
        search_wall_elapsed_sec = None
        completed_local_min_trials = 0
        local_min_wall_elapsed_sec = None
        completed_replays = 0
        replay_wall_elapsed_sec = None
        completed_folds = 1
    print(format_optimizer_final_performance_summary(
        folds=1,
        seeds=int(len(seeds)),
        min_agree=int(policy["min_agree"]),
        completed_folds=int(completed_folds or 1),
        total_elapsed_sec=max(0.0, time.perf_counter() - float(ensemble_started_at)),
        completed_trials=int(completed_trials or 0),
        search_wall_elapsed_sec=search_wall_elapsed_sec,
        completed_local_min_trials=int(completed_local_min_trials or 0),
        local_min_wall_elapsed_sec=local_min_wall_elapsed_sec,
        completed_replays=int(completed_replays or 0),
        replay_wall_elapsed_sec=replay_wall_elapsed_sec,
        resource_summary=resource_sampler.summary(),
        color=True,
    ))
    if normalize_optimizer_model_mode(selected_model_mode) == "trade":
        if bool(TRADE_MODE_AUTO_PROMOTE_RUN_BEST):
            promote_status = _promote_candidate_to_run_best(session=dashboard_session, emit_output=False)
            if promote_status != 0:
                return int(promote_status)
        output_entries = [("candidate_best", CANDIDATE_BEST_PARAMS_PATH)]
        if os.path.exists(RUN_BEST_PARAMS_PATH):
            output_entries.append(("run_best", RUN_BEST_PARAMS_PATH))
        _print_optimizer_output_files("💾 輸出檔案", output_entries)
        return 0
    visible_policy_paths = _visible_policy_paramset_paths(dict((_ensemble_summary or {}).get("policy_paramsets") or {}))
    _print_optimizer_output_files("💾 輸出檔案", list(visible_policy_paths.items()))
    return 0


def _extract_cli_value(argv, option_name: str):
    args = [] if argv is None else list(argv)
    for idx in range(1, len(args)):
        raw_arg = str(args[idx]).strip()
        if raw_arg == option_name and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if raw_arg.startswith(option_name + '='):
            return raw_arg.split('=', 1)[1].strip()
    return ''


def _has_cli_flag(argv, option_name: str) -> bool:
    args = [] if argv is None else list(argv)
    return any(str(arg).strip() == option_name for arg in args[1:])


def _mode_needs_latest_data_date(model_mode: str, study_scope: str | None = None) -> bool:
    normalized_mode = normalize_optimizer_model_mode(model_mode)
    if normalized_mode in {"full", "trade"}:
        return True
    if normalized_mode == "study" and normalize_optimizer_study_scope(study_scope) == "full":
        return True
    return False


def _resolve_optimizer_study_scope(argv, environ, *, selected_model_mode: str) -> str:
    if normalize_optimizer_model_mode(selected_model_mode) != "study":
        return ""
    cli_value = _extract_cli_value(argv, "--study-scope")
    if cli_value:
        return normalize_optimizer_study_scope(cli_value)
    env_value = str((environ or {}).get("V16_OPTIMIZER_STUDY_SCOPE", "") or "").strip()
    if env_value:
        return normalize_optimizer_study_scope(env_value)
    return normalize_optimizer_study_scope(None)


def _format_optimizer_model_mode_for_display(model_mode: str, walk_forward_policy: dict) -> str:
    normalized_mode = normalize_optimizer_model_mode(model_mode)
    if normalized_mode == "study":
        study_scope = normalize_optimizer_study_scope((walk_forward_policy or {}).get("study_scope"))
        return "Study-Full" if study_scope == "full" else "Study-OOS"
    if normalized_mode == "full":
        return "Full"
    return normalized_mode


def _is_study_full_runtime_mode(model_mode: str, walk_forward_policy: dict) -> bool:
    return (
        normalize_optimizer_model_mode(model_mode) == "study"
        and normalize_optimizer_study_scope((walk_forward_policy or {}).get("study_scope")) == "full"
    )


def _build_optimizer_study_db_file_path(*, output_dir: str, dataset_profile_key: str, study_scope: str) -> str:
    safe_dataset = normalize_dataset_profile_key(dataset_profile_key, default=DEFAULT_DATASET_PROFILE)
    safe_scope = normalize_optimizer_study_scope(study_scope)
    return os.path.join(output_dir, "study_db", f"optimizer_study_{safe_dataset}_{safe_scope}.db")


def _resolve_optimizer_db_file_for_mode(*, output_dir: str, dataset_profile_key: str, selected_model_mode: str, selected_study_scope: str, session_ts: str, timing_mode: bool) -> str:
    from tools.optimizer.benchmark import build_timing_db_file_path

    if bool(timing_mode):
        return build_timing_db_file_path(output_dir=output_dir, dataset_profile_key=dataset_profile_key, session_ts=session_ts)
    if normalize_optimizer_model_mode(selected_model_mode) == "study":
        return _build_optimizer_study_db_file_path(
            output_dir=output_dir,
            dataset_profile_key=dataset_profile_key,
            study_scope=selected_study_scope,
        )
    if bool(OPTIMIZER_PERSIST_STUDY_DB):
        return build_timing_db_file_path(output_dir=output_dir, dataset_profile_key=dataset_profile_key, session_ts=session_ts)
    return ""


def _normalize_study_db_action(raw_action: str) -> str:
    normalized = str(raw_action or "").strip().lower()
    if normalized in {"resume", "continue", "", "2"}:
        return "resume"
    if normalized in {"restart", "reset", "new", "start_over", "1"}:
        return "restart"
    raise ValueError(f"Study 記憶庫操作只接受 restart/resume，收到: {raw_action!r}")


def _apply_interactive_study_db_policy(*, selected_model_mode: str, db_file: str, colors: dict, study_db_action: str = "") -> None:
    if normalize_optimizer_model_mode(selected_model_mode) != "study":
        return
    if not db_file or not is_interactive_console():
        return
    if str(study_db_action or "").strip():
        action = _normalize_study_db_action(study_db_action)
    else:
        choice = safe_prompt_choice(
            "\n👉 Study 記憶庫：[Enter] 接續訓練  [1] 重頭開始 : ",
            "",
            ("", "1"),
            "Study 記憶庫操作選項",
        )
        action = "restart" if choice == "1" else "resume"
    if action == "resume":
        if os.path.exists(db_file):
            print(f"{colors['green']}🔁 Study mode 接續既有記憶庫：{_project_relative_path(db_file)}{colors['reset']}")
        else:
            print(f"{colors['yellow']}ℹ️ 找不到既有 Study 記憶庫，改用重頭開始：{_project_relative_path(db_file)}{colors['reset']}")
        return
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"{colors['red']}🗑️ Study mode 已刪除舊記憶，重頭開始：{_project_relative_path(db_file)}{colors['reset']}")
    else:
        print(f"{colors['gray']}🆕 Study mode 重頭開始：{_project_relative_path(db_file)}{colors['reset']}")


def _resolve_cli_run_request(argv):
    from core.runtime_utils import parse_int_strict
    from tools.optimizer.benchmark import OPTIMIZER_TIMING_MODE_DEFAULT_TRIALS

    timing_mode = _has_cli_flag(argv, '--timing')
    outer_oos_mode = _has_cli_flag(argv, '--outer-oos')
    cli_trials_raw = _extract_cli_value(argv, '--trials')
    if outer_oos_mode:
        return {
            'timing_mode': timing_mode,
            'n_trials': int(cli_trials_raw) if str(cli_trials_raw or '').strip() else (OPTIMIZER_TIMING_MODE_DEFAULT_TRIALS if timing_mode else 0),
            'action': 'outer_rolling_oos',
            'source': 'CLI:--outer-oos+--timing' if timing_mode else 'CLI:--outer-oos',
        }
    if cli_trials_raw:
        cli_trials = parse_int_strict(cli_trials_raw, 'CLI 參數 --trials', min_value=0)
        if timing_mode and cli_trials <= 0:
            raise ValueError('--timing 模式要求 --trials >= 1。')
        return {
            'timing_mode': timing_mode,
            'n_trials': int(cli_trials),
            'action': 'train' if int(cli_trials) > 0 else 'export_candidate',
            'source': 'CLI:--timing+--trials' if timing_mode else 'CLI:--trials',
        }
    if timing_mode:
        return {
            'timing_mode': True,
            'n_trials': 3,
            'action': 'train',
            'source': 'CLI:--timing',
        }
    return None


def _prompt_optimizer_model_mode(default_model: str):
    try:
        default_normalized = normalize_optimizer_model_mode(default_model)
    except ValueError:
        default_normalized = 'trade'
    return default_normalized, 'UI/MENU_DEFAULT'


def resolve_optimizer_model_mode(argv, environ, *, default_model: str = DEFAULT_OPTIMIZER_MODEL_MODE):
    cli_value = _extract_cli_value(argv, '--model')
    if cli_value:
        normalized = cli_value.strip().lower()
        source = 'CLI:--model'
    elif _has_cli_flag(argv, '--timing'):
        normalized = 'oos'
        source = 'TIMING_DEFAULT:oos'
    else:
        env_value = str((environ or {}).get('V16_OPTIMIZER_MODEL', '')).strip()
        if env_value:
            normalized = env_value.lower()
            source = 'ENV:V16_OPTIMIZER_MODEL'
        elif is_interactive_console():
            return _prompt_optimizer_model_mode(default_model)
        else:
            normalized = str(default_model).strip().lower() or 'trade'
            source = 'DEFAULT'
    try:
        normalized = normalize_optimizer_model_mode(normalized)
    except ValueError as exc:
        raise ValueError(f"optimizer 模式只接受 full、trade、oos 或 study，收到: {normalized}") from exc
    return normalized, source


def main(argv=None, environ=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset", "--model", "--study-scope", "--trials", "--outer-train-start", "--outer-first-oos", "--outer-last-oos", "--outer-first-oos-date", "--outer-last-oos-date", "--outer-window-mode", "--outer-train-window-years", "--outer-train-window-months", "--outer-oos-months"), flag_options=("--timing", "--outer-oos", "--yes"))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/optimizer/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full] [--model study|full|oos|trade] [--study-scope oos|full] [--trials N] [--timing] [--outer-oos] [--outer-window-mode fixed|expanding] [--outer-train-window-months N] [--outer-oos-months N]")
        print("說明: 預設 trade；full 為 seed ensemble 全期間訓練且無 OOS，且 local_min review 預設關閉；trade 以最新資料日往前固定訓練窗產生 candidate_best/run_best；oos 為 seed ensemble 單 fold validation；study 可選 Study-OOS 或 Study-Full，且維持單一隨機 seed study 輸出；--outer-oos 執行 rolling monthly OOS test。舊 --model split 仍相容為 oos。")
        return 0

    from core.data_utils import discover_unique_csv_inputs
    from strategies.breakout.search_space import get_breakout_optimizer_required_min_rows
    from tools.optimizer.prep import load_all_raw_data
    from tools.optimizer.runtime import (
        create_optimizer_study,
        ensure_export_only_db_not_empty,
        ensure_optimizer_db_usable,
        export_best_params_if_requested,
        maybe_print_history_best,
        print_resolved_trial_count,
        prompt_existing_db_policy,
        resolve_optimizer_single_fold_search_parallel_trials,
        resolve_training_session_export_policy,
        resolve_trial_count_or_exit,
    )
    from tools.optimizer.robustness import (
        build_local_min_score_best_trial_resolver,
        print_local_min_score_finalist_review,
        print_local_min_score_winner_summary,
        select_best_finalist_by_inner_validate_score,
        select_best_finalist_by_local_retention,
    )
    from tools.optimizer.session import close_study_storage
    from tools.optimizer.static_ensemble_dashboard import print_optimizer_static_ensemble_console_dashboard
    from tools.optimizer.study_utils import (
        build_best_params_payload_from_trial,
        build_optimizer_db_file_path,
        is_qualified_trial_value,
        resolve_optimizer_seed,
        resolve_optimizer_trial_count,
    )
    from tools.optimizer.benchmark import (
        build_timing_db_file_path,
        build_timing_summary,
        print_timing_summary,
        write_timing_summary,
    )

    loaded_policy = load_walk_forward_policy(PROJECT_ROOT)
    try:
        selected_model_mode, model_mode_source = resolve_optimizer_model_mode(argv, environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    set_optimizer_runtime_model_mode(selected_model_mode)

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(argv, environ, default=DEFAULT_DATASET_PROFILE)
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        dataset_label = get_dataset_profile_label(dataset_profile_key)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    try:
        selected_study_scope = _resolve_optimizer_study_scope(argv, environ, selected_model_mode=selected_model_mode)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    latest_data_date = None
    if _mode_needs_latest_data_date(selected_model_mode, selected_study_scope):
        try:
            latest_data_date = _resolve_latest_dataset_date(selected_data_dir)
        except ValueError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
    walk_forward_policy = build_optimizer_runtime_policy(
        loaded_policy,
        selected_model_mode,
        latest_data_date=latest_data_date,
        study_scope=selected_study_scope,
    )
    optimizer_required_min_rows = get_breakout_optimizer_required_min_rows()
    walk_forward_policy[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(optimizer_required_min_rows)
    objective_mode = str(walk_forward_policy.get('objective_mode', 'split_train_romd'))
    session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
    best_trial_resolver = build_local_min_score_best_trial_resolver(session=session, objective_mode=objective_mode)

    try:
        cli_run_request = _resolve_cli_run_request(argv)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    timing_mode = bool(cli_run_request and cli_run_request.get("timing_mode"))
    db_file = _resolve_optimizer_db_file_for_mode(
        output_dir=OUTPUT_DIR,
        dataset_profile_key=dataset_profile_key,
        selected_model_mode=selected_model_mode,
        selected_study_scope=selected_study_scope,
        session_ts=session.session_ts,
        timing_mode=timing_mode,
    )
    db_name = f"sqlite:///{db_file}" if db_file else None
    ensure_runtime_dirs()

    if cli_run_request is not None:
        session.n_trials = int(cli_run_request["n_trials"])
        session.run_action = str(cli_run_request.get("action", "train"))
        trial_source = str(cli_run_request.get("source", "CLI"))
        trial_count_exit = None
    else:
        trial_count_exit, trial_source = resolve_trial_count_or_exit(
            session,
            environ=environ,
            resolve_optimizer_trial_count=resolve_optimizer_trial_count,
            colors=COLORS,
        )
    if trial_count_exit is not None:
        return trial_count_exit

    requested_model_mode = str(getattr(session, "requested_model_mode", "") or "").strip().lower()
    if requested_model_mode:
        try:
            requested_model_mode = normalize_optimizer_model_mode(requested_model_mode)
        except ValueError:
            requested_model_mode = ""
    requested_study_scope = str(getattr(session, "requested_study_scope", "") or "").strip().lower()
    if requested_study_scope:
        try:
            requested_study_scope = normalize_optimizer_study_scope(requested_study_scope)
        except ValueError:
            requested_study_scope = ""
    target_model_mode = requested_model_mode if requested_model_mode in {"full", "oos", "study", "trade"} else selected_model_mode
    target_study_scope = ""
    if target_model_mode == "study":
        target_study_scope = requested_study_scope or selected_study_scope or normalize_optimizer_study_scope(None)
    needs_policy_rebuild = target_model_mode != selected_model_mode or target_study_scope != selected_study_scope
    if needs_policy_rebuild:
        requested_trials = int(getattr(session, "n_trials", 0) or 0)
        requested_action = str(getattr(session, "run_action", "train") or "train")
        requested_study_db_action = str(getattr(session, "requested_study_db_action", "") or "").strip().lower()
        selected_model_mode = target_model_mode
        selected_study_scope = target_study_scope
        set_optimizer_runtime_model_mode(selected_model_mode)
        latest_data_date = None
        if _mode_needs_latest_data_date(selected_model_mode, selected_study_scope):
            try:
                latest_data_date = _resolve_latest_dataset_date(selected_data_dir)
            except ValueError as exc:
                print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
                return 1
        walk_forward_policy = build_optimizer_runtime_policy(
            loaded_policy,
            selected_model_mode,
            latest_data_date=latest_data_date,
            study_scope=selected_study_scope,
        )
        walk_forward_policy[RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD] = int(optimizer_required_min_rows)
        objective_mode = str(walk_forward_policy.get('objective_mode', 'split_train_romd'))
        session = build_optimizer_session(walk_forward_policy=walk_forward_policy)
        session.n_trials = int(requested_trials)
        session.run_action = requested_action
        session.requested_model_mode = requested_model_mode
        if selected_model_mode == "study":
            session.requested_study_scope = selected_study_scope
            if requested_study_db_action:
                session.requested_study_db_action = requested_study_db_action
        best_trial_resolver = build_local_min_score_best_trial_resolver(session=session, objective_mode=objective_mode)
        db_file = _resolve_optimizer_db_file_for_mode(
            output_dir=OUTPUT_DIR,
            dataset_profile_key=dataset_profile_key,
            selected_model_mode=selected_model_mode,
            selected_study_scope=selected_study_scope,
            session_ts=session.session_ts,
            timing_mode=timing_mode,
        )
        db_name = f"sqlite:///{db_file}" if db_file else None

    if str(getattr(session, "run_action", "train")) == "outer_rolling_oos":
        from tools.optimizer.outer_rolling_oos import run_outer_rolling_oos
        try:
            optimizer_seed, seed_source = resolve_optimizer_seed(environ)
        except ValueError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        outer_timing_mode = bool(cli_run_request and cli_run_request.get("timing_mode"))
        if outer_timing_mode and optimizer_seed is None:
            optimizer_seed, seed_source = OPTIMIZER_RANDOM_SEED_DEFAULT, f'TIMING_DEFAULT:{OPTIMIZER_RANDOM_SEED_DEFAULT}'
        return run_outer_rolling_oos(
            argv=argv,
            environ=environ,
            project_root=PROJECT_ROOT,
            output_dir=OUTPUT_DIR,
            base_policy=loaded_policy,
            selected_data_dir=selected_data_dir,
            dataset_label=dataset_label,
            load_all_raw_data=load_all_raw_data,
            optimizer_required_min_rows=optimizer_required_min_rows,
            build_optimizer_session=build_optimizer_session,
            create_optimizer_study=create_optimizer_study,
            ensure_study_effective_policy_compatible=_ensure_study_effective_policy_compatible,
            configure_optuna_logging=configure_optuna_logging,
            optimizer_seed=optimizer_seed,
            default_trials=int(getattr(session, "n_trials", 0) or OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
            timing_mode=outer_timing_mode,
        )
    if str(getattr(session, "run_action", "train")) == "promote_candidate":
        ensure_runtime_dirs()
        try:
            if not os.path.isdir(selected_data_dir):
                raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
            session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=False,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        return _promote_candidate_to_run_best(session=session)

    try:
        optimizer_seed, seed_source = resolve_optimizer_seed(environ)
    except ValueError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    if timing_mode and optimizer_seed is None:
        optimizer_seed, seed_source = OPTIMIZER_RANDOM_SEED_DEFAULT, f'TIMING_DEFAULT:{OPTIMIZER_RANDOM_SEED_DEFAULT}'
    if selected_model_mode == "study" and optimizer_seed is None and int(getattr(session, "n_trials", 0) or 0) > 0:
        optimizer_seed = int(generate_random_seed_ensemble(1)[0])
        seed_source = "STUDY_RANDOM_SEED"

    if not timing_mode and selected_model_mode != "study" and str(getattr(session, "run_action", "train")) == "train":
        ensemble_result = _run_nonrolling_random_seed_ensemble_training(
            environ=environ,
            selected_data_dir=selected_data_dir,
            dataset_profile_key=dataset_profile_key,
            dataset_label=dataset_label,
            selected_model_mode=selected_model_mode,
            load_all_raw_data=load_all_raw_data,
            optimizer_required_min_rows=optimizer_required_min_rows,
            objective_mode=objective_mode,
            walk_forward_policy=walk_forward_policy,
            requested_trials=int(session.n_trials),
            build_optimizer_session=build_optimizer_session,
            create_optimizer_study=create_optimizer_study,
            ensure_study_effective_policy_compatible=_ensure_study_effective_policy_compatible,
            configure_optuna_logging=configure_optuna_logging,
            print_local_min_score_finalist_review=print_local_min_score_finalist_review,
            print_local_min_score_winner_summary=print_local_min_score_winner_summary,
            close_study_storage=close_study_storage,
            is_qualified_trial_value=is_qualified_trial_value,
            resolve_optimizer_single_fold_search_parallel_trials=resolve_optimizer_single_fold_search_parallel_trials,
            build_best_params_payload_from_trial=build_best_params_payload_from_trial,
        )
        if ensemble_result is not None:
            return int(ensemble_result)

    if session.n_trials == 0:
        if selected_model_mode == "study":
            if not db_file:
                print(f"{C_RED}❌ Study Mode 輸出 base.json 需要既有 study 記憶庫；目前未設定 db_file。{C_RESET}", file=sys.stderr)
                return 1
            if not os.path.exists(db_file):
                print(f"{C_RED}❌ 記憶庫不存在，無法輸出 base.json: {db_file}；請先用 Study Mode 訓練產生 study 記憶庫。{C_RESET}", file=sys.stderr)
                return 1
            study = None
            try:
                ensure_optimizer_db_usable(db_file)
                ensure_export_only_db_not_empty(db_file)
                study = create_optimizer_study(
                    db_name,
                    seed=(int(optimizer_seed) if optimizer_seed is not None else 0),
                    sampler_kind=("random" if timing_mode else "tpe"),
                )
                _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
                best_trial = session.get_best_completed_trial_or_none(study)
                if best_trial is None or not is_qualified_trial_value(best_trial.value):
                    print(f"{C_YELLOW}ℹ️ Study Mode 輸出模式完成，但目前尚無可匯出的 base trial。{C_RESET}")
                    return 0
                if not getattr(session, "raw_data_cache_data_dir", None):
                    try:
                        if not os.path.isdir(selected_data_dir):
                            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
                        session.load_raw_data(
                            selected_data_dir,
                            load_all_raw_data=load_all_raw_data,
                            required_min_rows=optimizer_required_min_rows,
                            verbose=False,
                        )
                    except (FileNotFoundError, RuntimeError, ValueError) as exc:
                        print(
                            f"{C_YELLOW}⚠️ Study Mode 輸出模式略過 base replay 顯示：{exc}{C_RESET}"
                        )
                return _finalize_single_seed_study_base_only_outputs(
                    best_trial=best_trial,
                    optimizer_seed=(int(optimizer_seed) if optimizer_seed is not None else 0),
                    objective_mode=objective_mode,
                    walk_forward_policy=walk_forward_policy,
                    dataset_label=dataset_label,
                    selected_model_mode=selected_model_mode,
                    trials_per_seed=0,
                    build_best_params_payload_from_trial=build_best_params_payload_from_trial,
                    dashboard_session=session,
                )
            except RuntimeError as exc:
                print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
                return 1
            finally:
                if study is not None:
                    close_study_storage(study)
        if selected_model_mode != "trade":
            mode_label = _format_optimizer_model_mode_for_display(selected_model_mode, walk_forward_policy)
            print(f"{C_YELLOW}ℹ️ {mode_label} 目前沒有可接續的單一 study 記憶庫輸出；請用 --trials N 重新產生結果。{C_RESET}")
            return 0
        if not db_file:
            print(f"{C_RED}❌ 目前預設使用 memory study，不保留長期 DB；export_candidate 不支援從硬碟接續匯出。請用 Trade mode --trials N 重新訓練產生 candidate_best。{C_RESET}", file=sys.stderr)
            return 1
        if not os.path.exists(db_file):
            print(f"{C_RED}❌ 記憶庫不存在，無法匯出: {db_file}；請用 Trade mode --trials N 重新訓練產生 candidate_best。{C_RESET}", file=sys.stderr)
            return 1
        try:
            ensure_optimizer_db_usable(db_file)
            ensure_export_only_db_not_empty(db_file)
            study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=("random" if timing_mode else "tpe"))
            _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
        except RuntimeError as exc:
            print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
            return 1
        try:
            if not os.path.isdir(selected_data_dir):
                raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
            csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
            if not csv_inputs:
                raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
            session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=False,
            )
            compact_display = True
            if compact_display:
                session.outer_rolling_local_progress_context = _make_nonrolling_local_min_progress_context(
                    walk_forward_policy,
                    best_base_score=_resolve_study_best_base_score(session, study),
                )
            try:
                finalists, best_trial = print_local_min_score_finalist_review(
                    study,
                    session=session,
                    objective_mode=objective_mode,
                    colors=COLORS,
                    winner_trial=None,
                    emit_table=True,
                    show_progress=True,
                )
            finally:
                if compact_display:
                    _finalize_nonrolling_single_fold_inline_progress(session)
                    _clear_nonrolling_local_min_progress_hooks(session)
            selector = _resolve_trade_candidate_selector()
            selected_entry = _select_finalist_entry_by_selector(finalists, objective_mode=objective_mode, selector=selector)
            selected_trial = None if selected_entry is None else selected_entry.get("trial")
            if selected_trial is None or not is_qualified_trial_value(selected_trial.value):
                print(f"{C_YELLOW}ℹ️ 匯出模式完成，但目前 selector={selector} 無可匯出 winner。{C_RESET}")
                return 0

            print_local_min_score_winner_summary(
                winner_trial=selected_trial,
                session=session,
                colors=COLORS,
            )
            exported = _export_selected_candidate_artifacts(
                study=study,
                finalists=finalists,
                selected_trial=selected_trial,
                params_path=CANDIDATE_BEST_PARAMS_PATH,
                summary_path=CANDIDATE_BEST_SUMMARY_PATH,
                fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                colors=COLORS,
                objective_mode=objective_mode,
                walk_forward_policy=walk_forward_policy,
                action_label="export_candidate",
                selection_rule=f"trade_selector:{selector}",
                compare_only=False,
                artifact_label="candidate_best",
                export_best_params_if_requested=export_best_params_if_requested,
                is_qualified_trial_value=is_qualified_trial_value,
            )
            if not exported:
                return 1
            if bool(TRADE_MODE_AUTO_PROMOTE_RUN_BEST):
                promote_status = _promote_candidate_to_run_best(session=session, emit_output=False)
                if promote_status != 0:
                    return int(promote_status)
            output_entries = [("candidate_best", CANDIDATE_BEST_PARAMS_PATH)]
            if os.path.exists(RUN_BEST_PARAMS_PATH):
                output_entries.append(("run_best", RUN_BEST_PARAMS_PATH))
            _print_optimizer_output_files("💾 輸出檔案", output_entries)
            return 0
        finally:
            session.close_trial_prep_executor()
            close_study_storage(study)

    try:
        if not os.path.isdir(selected_data_dir):
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
    except FileNotFoundError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
    if not csv_inputs:
        print(f"{C_RED}❌ {build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir)}{C_RESET}", file=sys.stderr)
        return 1

    try:
        if db_file and selected_model_mode != "study" and os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
    except RuntimeError as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    configure_optuna_logging()
    if not timing_mode:
        print_resolved_trial_count(session, trial_source=trial_source, colors=COLORS)
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端投資組合 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    selection_start_year = int(walk_forward_policy.get('selection_start_year', walk_forward_policy['train_start_year']))
    search_train_end_year = int(walk_forward_policy['search_train_end_year'])
    oos_start_year = walk_forward_policy.get('oos_start_year')
    is_study_full_mode = _is_study_full_runtime_mode(selected_model_mode, walk_forward_policy)
    if selected_model_mode in {'oos', 'study'} and not is_study_full_mode:
        inner_scope_text = ""
        if bool(OPTIMIZER_INNER_VALIDATE_ANTI_OVERFIT_ENABLED):
            inner_validate_year = int(search_train_end_year)
            inner_train_end_year = int(inner_validate_year) - 1
            inner_scope_text = f" | inner_train={selection_start_year}~{inner_train_end_year} | inner_val={inner_validate_year}"
        scope_text = (
            f"selection={selection_start_year}~{search_train_end_year}{inner_scope_text} | "
            f"oos={oos_start_year if oos_start_year is not None else search_train_end_year + 1}~latest"
        )
    else:
        train_start_date = walk_forward_policy.get("train_start_date") or str(selection_start_year)
        train_end_date = walk_forward_policy.get("search_train_end_date") or str(search_train_end_year)
        latest_text = walk_forward_policy.get("latest_data_date") or train_end_date
        scope_text = f"selection={train_start_date}~{train_end_date} | oos=disabled | latest={latest_text}"
    inline_override_fields = list(walk_forward_policy.get("inline_override_fields", []) or [])
    override_text = "" if not inline_override_fields else f" | override={','.join(inline_override_fields)}"
    display_model_mode = _format_optimizer_model_mode_for_display(selected_model_mode, walk_forward_policy)
    print(
        f"{C_GRAY}📌 設定｜資料集={dataset_label}｜模式={display_model_mode}{override_text}｜"
        f"{scope_text}｜trials={session.n_trials}{C_RESET}"
    )
    seed_text = str(optimizer_seed) if optimizer_seed is not None else "未設定"
    print(f"{C_GRAY}🎲 Optimizer seed: {seed_text} | 來源: {seed_source}{C_RESET}")

    try:
        if not timing_mode and selected_model_mode == "study":
            _apply_interactive_study_db_policy(
                selected_model_mode=selected_model_mode,
                db_file=db_file,
                colors=COLORS,
                study_db_action=str(getattr(session, "requested_study_db_action", "") or ""),
            )
        elif not timing_mode:
            prompt_existing_db_policy(db_file, COLORS)
        if db_file:
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
        if os.path.exists(db_file):
            ensure_optimizer_db_usable(db_file)
        sampler_kind = "random" if timing_mode else "tpe"
        study = create_optimizer_study(db_name, seed=optimizer_seed, sampler_kind=sampler_kind)
        _ensure_study_effective_policy_compatible(study=study, walk_forward_policy=walk_forward_policy)
    except (ValueError, RuntimeError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1

    session.timing_mode = timing_mode
    session.disable_milestone_dashboard = False if selected_model_mode == "study" else True

    overall_started_at = time.perf_counter()
    raw_data_load_sec = 0.0
    startup_total_sec = 0.0
    optimize_wall_sec = 0.0
    try:
        raw_data_load_started_at = time.perf_counter()
        session.load_raw_data(
                selected_data_dir,
                load_all_raw_data=load_all_raw_data,
                required_min_rows=optimizer_required_min_rows,
                verbose=False,
            )
        raw_data_load_sec = max(0.0, time.perf_counter() - raw_data_load_started_at)

        maybe_print_history_best(
            study,
            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
            train_enable_rotation=TRAIN_ENABLE_ROTATION,
            train_max_positions=TRAIN_MAX_POSITIONS,
            colors=COLORS,
            best_trial_resolver=session.get_best_completed_trial_or_none,
            session=session,
        )

        session.profile_recorder.init_output_files()

        startup_total_sec = max(0.0, time.perf_counter() - overall_started_at)
        print(f"{C_CYAN}⏱️ 前置完成：raw_data_load={raw_data_load_sec:.3f}s | startup_total={startup_total_sec:.3f}s | 即將進入第 1 輪{C_RESET}")
        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        session.profile_recorder.mark_run_started()
        optimize_started_at = time.perf_counter()
        training_interrupted = False
        try:
            search_parallel_trials = resolve_optimizer_single_fold_search_parallel_trials(
                environ,
                sampler_kind=sampler_kind,
            )
            study.optimize(
                session.objective,
                n_trials=session.n_trials,
                n_jobs=int(search_parallel_trials),
                callbacks=[session.monitoring_callback],
            )
        except KeyboardInterrupt:
            training_interrupted = True
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        optimize_wall_sec = max(0.0, time.perf_counter() - optimize_started_at)
        print()
        _print_profile_summary_compatible(session.profile_recorder, emit_console=not timing_mode)
        session.print_optimizer_prep_summary()
        should_export, export_policy = resolve_training_session_export_policy(
            requested_n_trials=session.n_trials,
            completed_session_trials=session.current_session_trial,
            interrupted=training_interrupted,
        )
        if timing_mode:
            timing_payload = build_timing_summary(
                session=session,
                dataset_label=dataset_label,
                dataset_profile_key=dataset_profile_key,
                selected_model_mode=selected_model_mode,
                db_file=db_file,
                raw_data_load_sec=raw_data_load_sec,
                startup_total_sec=startup_total_sec,
                optimize_wall_sec=optimize_wall_sec,
                total_wall_sec=max(0.0, time.perf_counter() - overall_started_at),
                optimizer_seed=optimizer_seed,
                timing_sampler_kind=sampler_kind,
            )
            timing_summary_path = write_timing_summary(
                output_dir=OUTPUT_DIR,
                session_ts=session.session_ts,
                payload=timing_payload,
            )
            print_timing_summary(payload=timing_payload)
        elif should_export:
            if _is_study_full_runtime_mode(selected_model_mode, walk_forward_policy):
                best_trial = session.get_best_completed_trial_or_none(study)
                if best_trial is not None and is_qualified_trial_value(best_trial.value):
                    status = _finalize_single_seed_study_base_only_outputs(
                        best_trial=best_trial,
                        optimizer_seed=int(optimizer_seed),
                        objective_mode=objective_mode,
                        walk_forward_policy=walk_forward_policy,
                        dataset_label=dataset_label,
                        selected_model_mode=selected_model_mode,
                        trials_per_seed=int(session.n_trials),
                        build_best_params_payload_from_trial=build_best_params_payload_from_trial,
                        dashboard_session=session,
                    )
                    if status != 0:
                        return int(status)
                else:
                    print(f"{C_YELLOW}ℹ️ Study-Full 訓練完成，但目前尚無可匯出的 base trial。{C_RESET}")
            else:
                compact_display = True
                if compact_display:
                    session.outer_rolling_local_progress_context = _make_nonrolling_local_min_progress_context(
                        walk_forward_policy,
                        best_base_score=_resolve_study_best_base_score(session, study),
                        overall_start=overall_started_at,
                    )
                try:
                    finalists, best_trial = print_local_min_score_finalist_review(
                        study,
                        session=session,
                        objective_mode=objective_mode,
                        colors=COLORS,
                        winner_trial=None,
                        emit_table=True,
                        show_progress=True,
                    )
                finally:
                    if compact_display:
                        _finalize_nonrolling_single_fold_inline_progress(session)
                        _clear_nonrolling_local_min_progress_hooks(session)
                if best_trial is not None and is_qualified_trial_value(best_trial.value):
                    if selected_model_mode != 'trade':
                        print_local_min_score_winner_summary(
                            winner_trial=best_trial,
                            session=session,
                            colors=COLORS,
                        )
                    if selected_model_mode == 'trade':
                        selector = _resolve_trade_candidate_selector()
                        selected_entry = _select_finalist_entry_by_selector(finalists, objective_mode=objective_mode, selector=selector)
                        selected_trial = None if selected_entry is None else selected_entry.get("trial")
                        if selected_trial is None or not is_qualified_trial_value(selected_trial.value):
                            print(f"{C_YELLOW}ℹ️ Trade mode 完成，但 selector={selector} 無可匯出的 candidate。{C_RESET}")
                            return 0
                        print_local_min_score_winner_summary(
                            winner_trial=selected_trial,
                            session=session,
                            colors=COLORS,
                        )
                        exported = _export_selected_candidate_artifacts(
                            study=study,
                            finalists=finalists,
                            selected_trial=selected_trial,
                            params_path=CANDIDATE_BEST_PARAMS_PATH,
                            summary_path=CANDIDATE_BEST_SUMMARY_PATH,
                            fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
                            colors=COLORS,
                            objective_mode=objective_mode,
                            walk_forward_policy=walk_forward_policy,
                            action_label="train",
                            selection_rule=f"trade_selector:{selector}",
                            compare_only=False,
                            artifact_label="candidate_best",
                            export_best_params_if_requested=export_best_params_if_requested,
                            is_qualified_trial_value=is_qualified_trial_value,
                        )
                        if not exported:
                            return 1
                        if bool(TRADE_MODE_AUTO_PROMOTE_RUN_BEST):
                            promote_status = _promote_candidate_to_run_best(session=session, emit_output=False)
                            if promote_status != 0:
                                return int(promote_status)
                        output_entries = [("candidate_best", CANDIDATE_BEST_PARAMS_PATH)]
                        if os.path.exists(RUN_BEST_PARAMS_PATH):
                            output_entries.append(("run_best", RUN_BEST_PARAMS_PATH))
                        _print_optimizer_output_files("💾 輸出檔案", output_entries)
                    elif selected_model_mode == 'study':
                        status = _finalize_single_seed_study_outputs(
                            session=session,
                            finalists=finalists,
                            best_trial=best_trial,
                            optimizer_seed=int(optimizer_seed),
                            objective_mode=objective_mode,
                            walk_forward_policy=walk_forward_policy,
                            dataset_label=dataset_label,
                            selected_model_mode=selected_model_mode,
                            trials_per_seed=int(session.n_trials),
                            build_best_params_payload_from_trial=build_best_params_payload_from_trial,
                            elapsed_sec=max(0.0, time.perf_counter() - optimize_started_at),
                        )
                        if status != 0:
                            return int(status)
                    elif selected_model_mode in {'full', 'oos'}:
                        finalize_best_trial_outputs(
                            session=session,
                            study=study,
                            best_trial_resolver=(lambda _study, _best_trial=best_trial: _best_trial),
                            dataset_label=dataset_label,
                            db_file=db_file,
                            walk_forward_policy=walk_forward_policy,
                        )
                else:
                    print(f"{C_YELLOW}ℹ️ 訓練完成，但目前尚無通過 local_min_score gate 的 winner。{C_RESET}")
        elif export_policy == "interrupted_before_target":
            print(
                f"{C_YELLOW}ℹ️ 本輪由使用者中斷，已完成 {session.current_session_trial}/{session.n_trials}；"
                f"不自動覆寫 candidate_best 或 run_best。{C_RESET}"
            )
        print(f"\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    finally:
        session.close_trial_prep_executor()
        close_study_storage(study)


if __name__ == "__main__":
    run_cli_entrypoint(main)
