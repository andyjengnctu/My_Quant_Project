import time

import pandas as pd

from core.active_param_ensemble import get_active_param_ensemble_policy, load_json_file
from core.config import SCORE_CALC_METHOD, SCORE_NUMERATOR_METHOD, format_system_score_for_display
from core.display_common import C_GRAY, C_RESET, get_p
from core.runtime_utils import stdout_supports_inline_progress, write_inline_progress
from core.strategy_dashboard import print_optimizer_trial_console_dashboard
from tools.optimizer.callbacks import (
    _build_first_zone_rows,
    _build_global_strategy_text,
    _build_hard_gate_lines,
    _build_oos_metrics_from_report,
    _build_search_train_dates_for_session,
    _build_training_param_lines,
    _build_trial_params_object,
    _get_reference_console_cache,
    _latest_data_end_text,
    _optimizer_train_score_display_label,
    _policy_date,
    _resolve_session_model_mode,
    _run_static_ensemble_dashboard_replay,
    _safe_float,
    _safe_int,
)


def _static_replay_progress_message(message: str) -> str:
    return f"{C_GRAY}⏳ {str(message).strip()}{C_RESET}"


def _emit_static_replay_progress(progress_state: dict | None, message: str, *, finish: bool = False) -> None:
    if progress_state is None:
        return
    text = _static_replay_progress_message(message)
    if stdout_supports_inline_progress():
        previous_width = int(progress_state.get("width", 0) or 0)
        progress_state["width"] = write_inline_progress(text, previous_width=previous_width)
        if finish:
            print(flush=True)
            progress_state["width"] = 0
    else:
        last = str(progress_state.get("last", ""))
        if text != last or finish:
            print(text, flush=True)
            progress_state["last"] = text


def _finish_static_replay_progress(progress_state: dict | None) -> None:
    if progress_state is None:
        return
    if stdout_supports_inline_progress() and int(progress_state.get("width", 0) or 0) > 0:
        print(flush=True)
    progress_state["width"] = 0




def _build_static_policy_oos_row_from_active_metrics(metrics: dict, *, benchmark_score: float) -> dict:
    candidate_score = _safe_float(metrics.get("score", 0.0))
    return {
        "available": True,
        "rank_1_trial": None,
        "rank_1_oos": float(candidate_score),
        "rank_1_return_pct": _safe_float(metrics.get("return_pct", 0.0)),
        "rank_1_mdd_pct": _safe_float(metrics.get("mdd_pct", 0.0)),
        "rank_1_trades": _safe_int(metrics.get("trade_count", 0)),
        "benchmark_0050_gap": float(candidate_score) - float(benchmark_score),
        "benchmark_return_pct": _safe_float(metrics.get("benchmark_return_pct", 0.0)),
        "benchmark_mdd_pct": _safe_float(metrics.get("benchmark_mdd_pct", 0.0)),
        "benchmark_oos_score": _safe_float(metrics.get("benchmark_oos_score", 0.0)),
    }


def _load_static_policy_paramset_payloads(policy_paramsets: dict | None) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for policy_name, source in dict(policy_paramsets or {}).items():
        if isinstance(source, dict):
            payloads[str(policy_name)] = dict(source)
            continue
        path = str(source or "").strip()
        if not path:
            continue
        try:
            payload = load_json_file(path)
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            payloads[str(policy_name)] = payload
    return payloads


def _build_static_policy_rows_from_paramsets(
    session,
    *,
    policy_paramsets: dict | None,
    start_date,
    end_date,
    initial_capital: float,
    progress_state: dict | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_emit=None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    _ = (initial_capital, progress_offset, progress_total)
    payloads = _load_static_policy_paramset_payloads(policy_paramsets)
    if not payloads:
        return {}, {}
    data_dir = getattr(session, "raw_data_cache_data_dir", None)
    if not data_dir:
        raise ValueError("session 尚未載入 data_dir，無法建立 ensemble policy replay")
    from tools.optimizer.outer_rolling_oos import (
        _build_policy_replay_context,
        _run_policy_replay_tasks,
        _stable_policy_replay_signature,
        is_optimizer_policy_replay_dedup_by_signature_enabled_default,
    )

    replay_context = _build_policy_replay_context(
        selected_data_dir=str(data_dir),
        oos_year=int(pd.Timestamp(start_date).year) if start_date is not None else int(session.train_start_year),
        oos_start_date=str(start_date)[:10],
        oos_end_date=str(end_date)[:10] if end_date is not None else str(_latest_data_end_text(session))[:10],
        max_positions=int(session.train_max_positions),
        enable_rotation=bool(session.train_enable_rotation),
    )
    dedup_enabled = is_optimizer_policy_replay_dedup_by_signature_enabled_default()
    unique_jobs_by_signature: dict[str, dict] = {}
    policy_jobs_by_name: dict[str, dict] = {}
    for policy_name, payload in payloads.items():
        signature = _stable_policy_replay_signature(payload) if dedup_enabled else f"{policy_name}:{_stable_policy_replay_signature(payload)}"
        job = policy_jobs_by_name[str(policy_name)] = {
            "signature": signature,
            "policy_name": str(policy_name),
            "policy_names": [str(policy_name)],
            "payload": dict(payload),
            "replay_context": replay_context,
        }
        if signature in unique_jobs_by_signature:
            unique_jobs_by_signature[signature].setdefault("policy_names", []).append(str(policy_name))
            continue
        unique_jobs_by_signature[signature] = dict(job)

    metrics_by_signature = _run_policy_replay_tasks(
        list(unique_jobs_by_signature.values()),
        progress_emit=progress_emit,
    )
    if not callable(progress_emit):
        _emit_static_replay_progress(progress_state, "seed ensemble policy replay | done", finish=True)
    raw_metrics_by_policy: dict[str, dict] = {
        name: dict(metrics_by_signature.get(str(job.get("signature"))) or {})
        for name, job in policy_jobs_by_name.items()
    }
    benchmark_source = next((metrics for metrics in raw_metrics_by_policy.values() if metrics.get("benchmark_oos_score") is not None), {})
    benchmark_score = _safe_float(benchmark_source.get("benchmark_oos_score", 0.0))
    rows: dict[str, dict] = {}
    for policy_name, metrics in raw_metrics_by_policy.items():
        if not metrics:
            rows[str(policy_name)] = {
                "available": False,
                "rank_1_oos": 0.0,
                "unavailable_reason": "policy_paramset_replay_failed",
            }
            continue
        rows[str(policy_name)] = _build_static_policy_oos_row_from_active_metrics(metrics, benchmark_score=benchmark_score)
    return rows, raw_metrics_by_policy


def build_optimizer_static_ensemble_single_fold_oos_row(session, *, ensemble_payload: dict, elapsed_sec: float | None = None, policy_paramsets: dict | None = None, progress_callback=None) -> dict:
    """Build the single-fold row used by the shared rolling-OOS table renderer."""
    policy = get_active_param_ensemble_policy(ensemble_payload)
    schedule = ensemble_payload.get("params_ensemble") or []
    first_member = schedule[0] if schedule else {}
    primary_params = _build_trial_params_object(first_member.get("params") or {})
    initial_capital = _safe_float(get_p(primary_params, "initial_capital", 0.0))
    model_mode = _resolve_session_model_mode(session)
    search_train_dates = _build_search_train_dates_for_session(session)
    selection_start = search_train_dates[0] if search_train_dates else _policy_date(session, "train_start_date")
    selection_end = search_train_dates[-1] if search_train_dates else _policy_date(session, "search_train_end_date")
    oos_start_date = None
    oos_end_date = None
    if model_mode == "oos":
        oos_start_date = _policy_date(session, "oos_start_date")
        if oos_start_date is None and session.walk_forward_policy.get("oos_start_year") is not None:
            oos_start_date = f"{int(session.walk_forward_policy['oos_start_year'])}-01-01"
        oos_end_date = _policy_date(session, "oos_end_date")
    replay_start_date = oos_start_date if model_mode == "oos" and oos_start_date else selection_start
    replay_end_date = oos_end_date if model_mode == "oos" and oos_start_date else selection_end
    replay_policy_paramsets = policy_paramsets or ensemble_payload.get("policy_paramsets")
    replay_payloads = _load_static_policy_paramset_payloads(replay_policy_paramsets)
    progress_state: dict | None = None if callable(progress_callback) else {}
    replay_started_perf = time.perf_counter()
    replay_started_ts = time.time()
    replay_last_done_ts: float | None = None

    def _emit_replay(policy_name: str, *, done: int, total: int | None = None, status: str = "RUN") -> None:
        nonlocal replay_last_done_ts
        done_value = int(done or 0)
        total_value = int(total if total is not None else len(replay_payloads))
        if done_value > 0:
            replay_last_done_ts = time.time()
        if callable(progress_callback):
            progress_callback({
                "stage": "ENSEMBLE_REPLAY",
                "status": str(status or "RUN"),
                "policy": str(policy_name),
                "replay_done": int(done_value),
                "replay_total": int(total_value),
                "replay_started_ts": float(replay_started_ts),
                "replay_last_done_ts": float(replay_last_done_ts) if replay_last_done_ts is not None else None,
                "elapsed_sec": max(0.0, time.perf_counter() - replay_started_perf),
            })
        else:
            step = max(1, min(max(1, total_value), done_value if done_value > 0 else 1))
            _emit_static_replay_progress(progress_state, f"seed ensemble policy replay | {step}/{max(1, total_value)} | {policy_name}")

    policy_rows, raw_policy_metrics = _build_static_policy_rows_from_paramsets(
        session,
        policy_paramsets=replay_payloads,
        start_date=replay_start_date,
        end_date=replay_end_date,
        initial_capital=initial_capital,
        progress_state=progress_state,
        progress_emit=_emit_replay if callable(progress_callback) else None,
    )
    available_policy_rows = [dict(row) for row in policy_rows.values() if bool(dict(row).get("available", False))]
    candidate_score = max((_safe_float(row.get("rank_1_oos", 0.0)) for row in available_policy_rows), default=0.0)
    benchmark_score = 0.0
    benchmark_return_pct = 0.0
    benchmark_mdd_pct = 0.0
    for metrics in raw_policy_metrics.values():
        if metrics:
            benchmark_score = _safe_float(metrics.get("benchmark_oos_score", 0.0))
            benchmark_return_pct = _safe_float(metrics.get("benchmark_return_pct", 0.0))
            benchmark_mdd_pct = _safe_float(metrics.get("benchmark_mdd_pct", 0.0))
            break
    range_start = str(oos_start_date or "")[:10] if model_mode == "oos" else ""
    range_end = str(oos_end_date or "")[:10] if model_mode == "oos" and oos_end_date else ""
    from tools.optimizer.outer_rolling_oos import (
        BASE_RETENTION_COMPARISON_POLICY_NAMES,
        REPORT_POLICY_NAMES,
        build_optimizer_seed_ensemble_fold_context,
        normalize_optimizer_seed_ensemble_fold_row,
    )

    fold_context = build_optimizer_seed_ensemble_fold_context(
        fold_idx=1,
        fold_count=1,
        selection_start_date=str(selection_start or "")[:10],
        selection_end_date=str(selection_end or "")[:10],
        oos_start_date=range_start,
        oos_end_date=range_end or ("latest" if model_mode == "oos" else ""),
        oos_period="" if model_mode != "oos" else None,
    )
    row = dict(fold_context)
    row.update({
        "best_finalist_oos_score": float(candidate_score),
        "benchmark_oos_score": float(benchmark_score),
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_mdd_pct": float(benchmark_mdd_pct),
        "elapsed_sec": elapsed_sec,
        "random_seed_ensemble": dict(policy),
    })

    unavailable_policy = {
        "available": False,
        "rank_1_oos": 0.0,
        "unavailable_reason": "nonrolling_policy_paramset_not_available",
    }
    for policy_name in tuple(REPORT_POLICY_NAMES) + tuple(BASE_RETENTION_COMPARISON_POLICY_NAMES):
        if policy_name in policy_rows:
            row[str(policy_name)] = dict(policy_rows[policy_name])
        else:
            row[str(policy_name)] = dict(unavailable_policy)
    return normalize_optimizer_seed_ensemble_fold_row(row)

def print_optimizer_static_ensemble_rolling_oos_table(session, *, ensemble_payload: dict, elapsed_sec: float | None = None, policy_paramsets: dict | None = None, progress_callback=None) -> dict | None:
    """Print non-rolling ensemble summary through the exact rolling-OOS table renderer."""
    from tools.optimizer.outer_rolling_oos import optimizer_seed_ensemble_table_titles, render_optimizer_results_tables

    row = build_optimizer_static_ensemble_single_fold_oos_row(
        session,
        ensemble_payload=ensemble_payload,
        elapsed_sec=elapsed_sec,
        policy_paramsets=policy_paramsets,
        progress_callback=progress_callback,
    )
    main_title, retention_title = optimizer_seed_ensemble_table_titles()
    table_text = render_optimizer_results_tables(
        [row],
        color=True,
        include_chain=False,
        include_oos_avg=False,
        main_table_title=main_title,
        retention_table_title=retention_title,
    )
    if table_text:
        print("\n" + table_text)


def print_optimizer_static_ensemble_console_dashboard(
    session,
    *,
    ensemble_payload: dict,
    seeds: list[int],
    milestone_title: str = "🏆 ENSEMBLE 訓練結果",
    title: str = "ENSEMBLE 績效與風險對比表",
    force: bool = False,
):
    if not bool(force):
        return {"payload_sec": 0.0, "render_sec": 0.0}
    payload_started_at = time.perf_counter()
    progress_state: dict = {}
    _emit_static_replay_progress(progress_state, "seed ensemble dashboard replay | train")
    policy = get_active_param_ensemble_policy(ensemble_payload)
    schedule = ensemble_payload.get("params_ensemble") or []
    first_member = schedule[0] if schedule else {}
    primary_params = _build_trial_params_object(first_member.get("params") or {})
    initial_capital = _safe_float(get_p(primary_params, "initial_capital", 0.0))
    model_mode = _resolve_session_model_mode(session)
    search_train_dates = _build_search_train_dates_for_session(session)
    train_start_date = search_train_dates[0] if search_train_dates else _policy_date(session, "train_start_date")
    train_end_date = search_train_dates[-1] if search_train_dates else _policy_date(session, "search_train_end_date")
    candidate_train_metrics, benchmark_train_metrics, train_range_text = _run_static_ensemble_dashboard_replay(
        session,
        ensemble_payload,
        start_date=train_start_date,
        end_date=train_end_date,
        initial_capital=initial_capital,
    )
    reference_cache = _get_reference_console_cache(session)
    train_rows = _build_first_zone_rows(
        candidate_metrics=candidate_train_metrics,
        reference_metrics=reference_cache,
        benchmark_metrics=benchmark_train_metrics,
    )
    test_title = None
    test_rows = None
    latest_data_end = _latest_data_end_text(session)
    if model_mode == "oos":
        oos_start_date = _policy_date(session, "oos_start_date")
        if oos_start_date is None and session.walk_forward_policy.get("oos_start_year") is not None:
            oos_start_date = f"{int(session.walk_forward_policy['oos_start_year'])}-01-01"
        if oos_start_date:
            _emit_static_replay_progress(progress_state, "seed ensemble dashboard replay | OOS")
            candidate_test_metrics, benchmark_test_metrics, oos_range_text = _run_static_ensemble_dashboard_replay(
                session,
                ensemble_payload,
                start_date=oos_start_date,
                end_date=_policy_date(session, "oos_end_date"),
                initial_capital=initial_capital,
            )
            reference_test_metrics = None
            if reference_cache and reference_cache.get("wf_report"):
                reference_test_metrics, _unused_bm, _unused_range = _build_oos_metrics_from_report(
                    report=reference_cache.get("wf_report"),
                    initial_capital=initial_capital,
                )
            elif reference_cache and isinstance(reference_cache.get("oos_metrics"), dict):
                reference_test_metrics = dict(reference_cache.get("oos_metrics") or {})
            test_title = f"【OOS 驗證績效摘要｜{oos_range_text}｜資料終點：{latest_data_end}】"
            test_rows = _build_first_zone_rows(
                candidate_metrics=candidate_test_metrics,
                reference_metrics=reference_test_metrics,
                benchmark_metrics=benchmark_test_metrics,
            )
    seed_text = ",".join(str(int(seed)) for seed in seeds)
    params_lines = [
        f"ENSEMBLE：N={int(policy['seed_count'])}｜min_agree={int(policy['min_agree'])}｜seeds={seed_text}",
        *[f"代表 member#1｜{line}" if idx == 0 else line for idx, line in enumerate(_build_training_param_lines(primary_params, entry_trade_counts=candidate_train_metrics))],
    ]
    payload_elapsed = max(0.0, time.perf_counter() - payload_started_at)
    _emit_static_replay_progress(progress_state, "seed ensemble dashboard replay | render")
    render_started_at = time.perf_counter()
    print_optimizer_trial_console_dashboard(
        title=title,
        milestone_title=milestone_title,
        global_strategy_text=_build_global_strategy_text(),
        mode_display="關閉明牌（穩定鎖倉）" if not session.train_enable_rotation else "啟用 (汰弱換強)",
        max_pos=session.train_max_positions,
        model_mode=model_mode,
        objective_mode=str(session.objective_mode),
        score_calc_method=SCORE_CALC_METHOD,
        score_numerator_method=SCORE_NUMERATOR_METHOD,
        system_score_display=f"{format_system_score_for_display(candidate_train_metrics.get('pf_romd', 0.0), decimals=3)}（{_optimizer_train_score_display_label()}／ENSEMBLE）",
        training_title=f"【訓練期間績效對比｜{train_range_text}】",
        training_rows=train_rows,
        testing_title=test_title,
        testing_rows=test_rows,
        upgrade_rows=None,
        compare_rows=None,
        params_lines=params_lines,
        hard_gate_lines=_build_hard_gate_lines(),
    )
    _finish_static_replay_progress(progress_state)
    return {
        "payload_sec": float(payload_elapsed),
        "render_sec": float(max(0.0, time.perf_counter() - render_started_at)),
    }

