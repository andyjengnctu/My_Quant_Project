import json
import os
import sqlite3
import sys

import numpy as np

from core.console_report import render_menu_item
from core.log_utils import format_exception_summary

from core.training_performance import resolve_optimizer_single_fold_search_parallel_trials_default

from core.display import print_strategy_dashboard
from services.optimizer.callbacks import print_optimizer_trial_milestone_dashboard
from core.runtime_utils import is_interactive_console, safe_prompt_choice
from services.optimizer.study_utils import (
    build_best_params_payload_from_trial,
    build_optimizer_trial_params,
    get_best_completed_trial_or_none,
    is_qualified_trial_value,
    list_completed_study_trials,
)


def prompt_existing_db_policy(db_file, colors):
    if not db_file or not os.path.exists(db_file):
        return
    if not is_interactive_console():
        return
    choice = safe_prompt_choice(
        "\n👉 Portfolio 記憶庫："
        + render_menu_item(1, "接續訓練", default=True)
        + "  "
        + render_menu_item(2, "刪除重來")
        + " : ",
        "1",
        ("1", "2"),
        "記憶庫操作選項",
    )
    if choice == "2":
        os.remove(db_file)
        print(f"{colors['red']}🗑️ 已刪除舊記憶。{colors['reset']}")


OPTIMIZER_LEGACY_STUDY_NAME = "portfolio_optimization_overnight"
OPTIMIZER_DETERMINISTIC_RESUME_STUDY_NAME = "portfolio_optimization_overnight_resume_safe_v1"
OPTIMIZER_EXECUTION_EVIDENCE_CONTRACT = "optimizer_resume_determinism_v1"
OPTIMIZER_CANONICAL_SAMPLER_KIND = "tpe"
_OPTIMIZER_SAMPLER_PRE_CHECKPOINT_KEY = "optimizer_sampler_rng_pre_v1"
_OPTIMIZER_SAMPLER_POST_CHECKPOINT_KEY = "optimizer_sampler_rng_post_v1"
_OPTIMIZER_SAMPLER_CHECKPOINT_SCHEMA = "optimizer_sampler_rng_checkpoint_v1"


def resolve_optimizer_sampler_kind() -> str:
    """Return the canonical production sampler used by Research, Trading, and timing."""
    return OPTIMIZER_CANONICAL_SAMPLER_KIND


def _deterministic_optimizer_study_name(*, sampler_kind: str, optuna_version: str) -> str:
    safe_version = "".join(
        character if character.isalnum() else "_" for character in str(optuna_version or "unknown")
    ).strip("_") or "unknown"
    sampler_key = str(sampler_kind or "tpe").strip().lower() or "tpe"
    return f"{OPTIMIZER_DETERMINISTIC_RESUME_STUDY_NAME}_{sampler_key}_optuna_{safe_version}"


def _serialize_numpy_random_state(state) -> dict:
    return {
        "bit_generator": str(state[0]),
        "keys": [int(value) for value in np.asarray(state[1], dtype=np.uint32).tolist()],
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _deserialize_numpy_random_state(payload: dict):
    data = dict(payload or {})
    return (
        str(data["bit_generator"]),
        np.asarray(list(data["keys"]), dtype=np.uint32),
        int(data["pos"]),
        int(data["has_gauss"]),
        float(data["cached_gaussian"]),
    )


def _sampler_rng_snapshot(sampler, sampler_kind: str) -> dict:
    sampler_key = str(sampler_kind or "tpe").strip().lower()
    primary_rng = getattr(getattr(sampler, "_rng", None), "rng", None)
    if primary_rng is None:
        raise RuntimeError("Optimizer sampler缺少可持久化RNG state")
    payload = {"primary": _serialize_numpy_random_state(primary_rng.get_state())}
    if sampler_key == "tpe":
        random_sampler = getattr(sampler, "_random_sampler", None)
        random_rng = getattr(getattr(random_sampler, "_rng", None), "rng", None)
        if random_rng is None:
            raise RuntimeError("Optimizer TPE sampler缺少startup RandomSampler RNG state")
        payload["startup_random"] = _serialize_numpy_random_state(random_rng.get_state())
    return payload


def _restore_sampler_rng_snapshot(sampler, sampler_kind: str, payload: dict) -> None:
    sampler_key = str(sampler_kind or "tpe").strip().lower()
    data = dict(payload or {})
    primary_rng = getattr(getattr(sampler, "_rng", None), "rng", None)
    if primary_rng is None or "primary" not in data:
        raise RuntimeError("Optimizer sampler checkpoint缺少primary RNG state")
    primary_rng.set_state(_deserialize_numpy_random_state(data["primary"]))
    if sampler_key == "tpe":
        random_sampler = getattr(sampler, "_random_sampler", None)
        random_rng = getattr(getattr(random_sampler, "_rng", None), "rng", None)
        if random_rng is None or "startup_random" not in data:
            raise RuntimeError("Optimizer TPE sampler checkpoint缺少startup RandomSampler RNG state")
        random_rng.set_state(_deserialize_numpy_random_state(data["startup_random"]))


def _study_internal_attrs(study) -> dict:
    return dict(study._storage.get_study_system_attrs(study._study_id) or {})


def _set_study_internal_attr(study, key: str, value) -> None:
    study._storage.set_study_system_attr(study._study_id, str(key), value)


def _build_sampler_checkpoint(*, sampler, sampler_kind: str, seed: int, trial_number: int, phase: str, optuna_version: str) -> dict:
    return {
        "schema": _OPTIMIZER_SAMPLER_CHECKPOINT_SCHEMA,
        "sampler_kind": str(sampler_kind),
        "seed": int(seed),
        "trial_number": int(trial_number),
        "phase": str(phase),
        "optuna_version": str(optuna_version),
        "rng": _sampler_rng_snapshot(sampler, sampler_kind),
    }


def _persist_sampler_checkpoint(*, study, sampler, sampler_kind: str, seed: int, trial_number: int, phase: str, optuna_version: str) -> None:
    key = (
        _OPTIMIZER_SAMPLER_PRE_CHECKPOINT_KEY
        if str(phase) == "pre"
        else _OPTIMIZER_SAMPLER_POST_CHECKPOINT_KEY
    )
    _set_study_internal_attr(
        study,
        key,
        _build_sampler_checkpoint(
            sampler=sampler,
            sampler_kind=sampler_kind,
            seed=int(seed),
            trial_number=int(trial_number),
            phase=str(phase),
            optuna_version=str(optuna_version),
        ),
    )


def _checkpointing_sampler(optuna, *, seed: int | None, sampler_kind: str):
    sampler_key = str(sampler_kind or "tpe").strip().lower()
    if seed is None:
        if sampler_key == "random":
            return optuna.samplers.RandomSampler(seed=None)
        return optuna.samplers.TPESampler()

    resolved_seed = int(seed)
    optuna_version = str(getattr(optuna, "__version__", ""))
    if sampler_key == "random":
        class _CheckpointingRandomSampler(optuna.samplers.RandomSampler):
            def before_trial(self, study, trial):
                super().before_trial(study, trial)
                _persist_sampler_checkpoint(
                    study=study, sampler=self, sampler_kind="random", seed=resolved_seed,
                    trial_number=int(trial.number), phase="pre", optuna_version=optuna_version,
                )

            def after_trial(self, study, trial, state, values):
                super().after_trial(study, trial, state, values)
                _persist_sampler_checkpoint(
                    study=study, sampler=self, sampler_kind="random", seed=resolved_seed,
                    trial_number=int(trial.number), phase="post", optuna_version=optuna_version,
                )

        return _CheckpointingRandomSampler(seed=resolved_seed)

    class _CheckpointingTPESampler(optuna.samplers.TPESampler):
        def before_trial(self, study, trial):
            super().before_trial(study, trial)
            _persist_sampler_checkpoint(
                study=study, sampler=self, sampler_kind="tpe", seed=resolved_seed,
                trial_number=int(trial.number), phase="pre", optuna_version=optuna_version,
            )

        def after_trial(self, study, trial, state, values):
            super().after_trial(study, trial, state, values)
            _persist_sampler_checkpoint(
                study=study, sampler=self, sampler_kind="tpe", seed=resolved_seed,
                trial_number=int(trial.number), phase="post", optuna_version=optuna_version,
            )

    return _CheckpointingTPESampler(seed=resolved_seed)


def _plain_seeded_sampler(optuna, *, seed: int, sampler_kind: str):
    sampler_key = str(sampler_kind or "tpe").strip().lower()
    if sampler_key == "random":
        return optuna.samplers.RandomSampler(seed=int(seed))
    return optuna.samplers.TPESampler(seed=int(seed))


def _checkpoint_compatible(checkpoint: dict, *, sampler_kind: str, seed: int, optuna_version: str, phase: str) -> bool:
    payload = dict(checkpoint or {})
    return (
        payload.get("schema") == _OPTIMIZER_SAMPLER_CHECKPOINT_SCHEMA
        and str(payload.get("sampler_kind") or "") == str(sampler_kind)
        and int(payload.get("seed", -1)) == int(seed)
        and str(payload.get("optuna_version") or "") == str(optuna_version)
        and str(payload.get("phase") or "") == str(phase)
        and isinstance(payload.get("rng"), dict)
    )


def _trial_params_match(generated: dict, expected: dict) -> bool:
    if list(generated) != list(expected):
        return False
    for key in generated:
        left = generated[key]
        right = expected[key]
        if isinstance(left, float) or isinstance(right, float):
            try:
                if float(left) != float(right):
                    return False
            except (TypeError, ValueError):
                return False
        elif left != right:
            return False
    return True


def _advance_sampler_over_running_trial(optuna, *, sampler, sampler_kind: str, seed: int, checkpoint: dict, study, running_trial) -> None:
    replay_sampler = _plain_seeded_sampler(optuna, seed=int(seed), sampler_kind=sampler_kind)
    _restore_sampler_rng_snapshot(replay_sampler, sampler_kind, dict(checkpoint.get("rng") or {}))
    directions = list(getattr(study, "directions", []) or [])
    if len(directions) == 1:
        replay_study = optuna.create_study(direction=directions[0], sampler=replay_sampler)
    else:
        replay_study = optuna.create_study(directions=directions, sampler=replay_sampler)
    for prior in list(getattr(study, "trials", []) or []):
        if int(prior.number) >= int(running_trial.number):
            break
        if not prior.state.is_finished():
            raise RuntimeError("Optimizer resume發現running trial之前仍有未完成trial，禁止非決定性接續")
        replay_study.add_trial(prior)
    replay_trial = replay_study.ask()
    if int(replay_trial.number) != int(running_trial.number):
        raise RuntimeError("Optimizer resume trial numbering無法重建，禁止非決定性接續")
    generated = {}
    for name, distribution in dict(running_trial.distributions or {}).items():
        generated[str(name)] = replay_trial._suggest(str(name), distribution)
    if not _trial_params_match(generated, dict(running_trial.params or {})):
        raise RuntimeError("Optimizer resume無法重建中斷trial的sampler state，禁止非決定性接續")
    _restore_sampler_rng_snapshot(
        sampler, sampler_kind, _sampler_rng_snapshot(replay_sampler, sampler_kind)
    )


def _restore_seeded_sampler_resume_state(optuna, *, study, sampler, sampler_kind: str, seed: int) -> None:
    trials = list(getattr(study, "trials", []) or [])
    if not trials:
        return
    attrs = _study_internal_attrs(study)
    optuna_version = str(getattr(optuna, "__version__", ""))
    post = dict(attrs.get(_OPTIMIZER_SAMPLER_POST_CHECKPOINT_KEY) or {})
    pre = dict(attrs.get(_OPTIMIZER_SAMPLER_PRE_CHECKPOINT_KEY) or {})
    running = [trial for trial in trials if not trial.state.is_finished()]
    if len(running) > 1:
        raise RuntimeError("Optimizer resume發現多個未完成trial；deterministic TPE只允許單一trial序列")

    if running:
        current = running[0]
        trial_number = int(current.number)
        if _checkpoint_compatible(
            post, sampler_kind=sampler_kind, seed=int(seed), optuna_version=optuna_version, phase="post"
        ) and int(post.get("trial_number", -1)) == trial_number:
            _restore_sampler_rng_snapshot(sampler, sampler_kind, dict(post["rng"]))
        elif _checkpoint_compatible(
            pre, sampler_kind=sampler_kind, seed=int(seed), optuna_version=optuna_version, phase="pre"
        ) and int(pre.get("trial_number", -1)) == trial_number:
            _advance_sampler_over_running_trial(
                optuna, sampler=sampler, sampler_kind=sampler_kind, seed=int(seed),
                checkpoint=pre, study=study, running_trial=current,
            )
        elif not dict(current.params or {}):
            previous = [trial for trial in trials if trial.state.is_finished()]
            if previous and _checkpoint_compatible(
                post, sampler_kind=sampler_kind, seed=int(seed), optuna_version=optuna_version, phase="post"
            ) and int(post.get("trial_number", -1)) == int(previous[-1].number):
                _restore_sampler_rng_snapshot(sampler, sampler_kind, dict(post["rng"]))
            else:
                raise RuntimeError("Optimizer resume缺少可恢復的sampler checkpoint")
        else:
            raise RuntimeError("Optimizer resume缺少中斷trial的sampler checkpoint")
        study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
        trials = list(getattr(study, "trials", []) or [])
        attrs = _study_internal_attrs(study)
        post = dict(attrs.get(_OPTIMIZER_SAMPLER_POST_CHECKPOINT_KEY) or {})

    last_trial = trials[-1]
    if not last_trial.state.is_finished():
        raise RuntimeError("Optimizer resume仍存在未完成trial")
    post_matches_last = (
        _checkpoint_compatible(
            post,
            sampler_kind=sampler_kind,
            seed=int(seed),
            optuna_version=optuna_version,
            phase="post",
        )
        and int(post.get("trial_number", -1)) == int(last_trial.number)
    )
    if not post_matches_last:
        pre_matches_last = (
            _checkpoint_compatible(
                pre,
                sampler_kind=sampler_kind,
                seed=int(seed),
                optuna_version=optuna_version,
                phase="pre",
            )
            and int(pre.get("trial_number", -1)) == int(last_trial.number)
        )
        if not pre_matches_last or not dict(last_trial.params or {}):
            raise RuntimeError("Optimizer resume缺少與最後完成trial一致的sampler checkpoint")
        _advance_sampler_over_running_trial(
            optuna,
            sampler=sampler,
            sampler_kind=sampler_kind,
            seed=int(seed),
            checkpoint=pre,
            study=study,
            running_trial=last_trial,
        )
        _persist_sampler_checkpoint(
            study=study,
            sampler=sampler,
            sampler_kind=sampler_kind,
            seed=int(seed),
            trial_number=int(last_trial.number),
            phase="post",
            optuna_version=optuna_version,
        )
        attrs = _study_internal_attrs(study)
        post = dict(attrs.get(_OPTIMIZER_SAMPLER_POST_CHECKPOINT_KEY) or {})
    _restore_sampler_rng_snapshot(sampler, sampler_kind, dict(post["rng"]))


def create_optimizer_study(db_name, *, seed=None, sampler_kind="tpe"):
    try:
        import optuna
        from sqlalchemy.exc import SQLAlchemyError

        sampler_key = str(sampler_kind or "tpe").strip().lower()
        deterministic_resume = seed is not None and db_name is not None
        sampler = (
            _checkpointing_sampler(optuna, seed=seed, sampler_kind=sampler_key)
            if deterministic_resume
            else (
                _plain_seeded_sampler(optuna, seed=int(seed), sampler_kind=sampler_key)
                if seed is not None
                else _checkpointing_sampler(optuna, seed=None, sampler_kind=sampler_key)
            )
        )
        create_kwargs = {
            "study_name": (
                _deterministic_optimizer_study_name(
                    sampler_kind=sampler_key,
                    optuna_version=str(getattr(optuna, "__version__", "")),
                )
                if deterministic_resume
                else OPTIMIZER_LEGACY_STUDY_NAME
            ),
            "direction": "maximize",
            "sampler": sampler,
        }
        if db_name is not None:
            create_kwargs["storage"] = db_name
            create_kwargs["load_if_exists"] = True

        study = optuna.create_study(**create_kwargs)
        if deterministic_resume and db_name is not None:
            attrs = _study_internal_attrs(study)
            persisted_checkpoint = dict(
                attrs.get(_OPTIMIZER_SAMPLER_POST_CHECKPOINT_KEY)
                or attrs.get(_OPTIMIZER_SAMPLER_PRE_CHECKPOINT_KEY)
                or {}
            )
            persisted_seed = persisted_checkpoint.get("seed")
            if list(getattr(study, "trials", []) or []) and persisted_seed is not None:
                resolved_persisted_seed = int(persisted_seed)
                if resolved_persisted_seed != int(seed):
                    sampler = _checkpointing_sampler(
                        optuna, seed=resolved_persisted_seed, sampler_kind=sampler_key
                    )
                    study.sampler = sampler
                    seed = resolved_persisted_seed
            _restore_seeded_sampler_resume_state(
                optuna, study=study, sampler=sampler, sampler_kind=sampler_key, seed=int(seed)
            )
        return study
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Optimizer 記憶庫開啟失敗: {format_exception_summary(exc, include_traceback=False)}") from exc
    except (sqlite3.Error, SQLAlchemyError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫開啟失敗: {format_exception_summary(exc, include_traceback=False)}") from exc


def _env_value_for_optimizer_runtime(environ, name: str, default: str) -> str:
    if isinstance(environ, dict) and environ.get(name) is not None:
        value = environ.get(name)
    else:
        value = os.environ.get(name, default)
    if value is None or str(value).strip() == "":
        return str(default)
    return str(value).strip()


def resolve_optimizer_single_fold_search_parallel_trials(environ=None, *, sampler_kind: str) -> int:
    """解析 rolling / 非 rolling 共用的單一 search unit Optuna trial 併發數。"""
    default_trials = resolve_optimizer_single_fold_search_parallel_trials_default()
    raw_value = _env_value_for_optimizer_runtime(
        environ,
        "OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS",
        str(default_trials),
    )
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        resolved = int(default_trials)
    resolved = max(1, min(16, resolved))
    sampler_text = str(sampler_kind or "").strip().lower()
    if sampler_text == "tpe":
        # TPE is history-dependent. Parallel trials can observe different completed-history
        # frontiers and therefore change the scientific trial sequence. Keep intra-study
        # TPE strictly sequential; parallelism belongs at the independent fold/seed level.
        return 1
    return int(resolved)

def ensure_optimizer_db_usable(db_file):
    if not os.path.exists(db_file):
        return

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA schema_version;").fetchone()
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫檔案損壞或不可讀: {format_exception_summary(exc, include_traceback=False)}") from exc
    finally:
        if conn is not None:
            conn.close()


def ensure_export_only_db_not_empty(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        has_any_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
        ).fetchone()
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise RuntimeError(f"Optimizer 記憶庫檔案損壞或不可讀: {format_exception_summary(exc, include_traceback=False)}") from exc
    finally:
        if conn is not None:
            conn.close()

    if not has_any_table:
        raise RuntimeError("記憶庫為空，無法匯出")


def print_best_trial_dashboard(trial, *, fixed_tp_percent, train_enable_rotation, train_max_positions, colors):
    attrs = trial.user_attrs
    params = build_optimizer_trial_params(trial.params, attrs, fixed_tp_percent=fixed_tp_percent)
    mode_display = "啟用 (汰弱換強)" if train_enable_rotation else "關閉 (穩定鎖倉)"
    print_strategy_dashboard(
        params=params,
        title="績效與風險對比表",
        mode_display=mode_display,
        max_pos=train_max_positions,
        trades=attrs["pf_trades"],
        missed_b=attrs.get("missed_buys", 0),
        missed_s=attrs.get("missed_sells", 0),
        final_eq=attrs["final_equity"],
        avg_exp=attrs["avg_exposure"],
        max_exp=attrs.get("max_exposure", None),
        sys_ret=attrs["pf_return"],
        bm_ret=attrs["bm_return"],
        sys_mdd=attrs["pf_mdd"],
        bm_mdd=attrs["bm_mdd"],
        win_rate=attrs["win_rate"],
        payoff=attrs["pf_payoff"],
        ev=attrs["pf_ev"],
        r_sq=attrs.get("r_squared", 0.0),
        m_win_rate=attrs.get("m_win_rate", 0.0),
        bm_r_sq=attrs.get("bm_r_squared", 0.0),
        bm_m_win_rate=attrs.get("bm_m_win_rate", 0.0),
        normal_trades=attrs.get("normal_trades", attrs["pf_trades"]),
        extended_trades=attrs.get("extended_trades", 0),
        reentry_trades=attrs.get("reentry_trades", 0),
        annual_trades=attrs.get("annual_trades", 0.0),
        reserved_buy_fill_rate=attrs.get("reserved_buy_fill_rate", 0.0),
        annual_return_pct=attrs.get("annual_return_pct", 0.0),
        bm_annual_return_pct=attrs.get("bm_annual_return_pct", 0.0),
        min_full_year_return_pct=attrs.get("min_full_year_return_pct", 0.0),
        min_month_return_pct=attrs.get("min_month_return_pct", 0.0),
        min_quarter_return_pct=attrs.get("min_quarter_return_pct", 0.0),
        portfolio_total_r=attrs.get("pf_total_r", 0.0),
        portfolio_median_r=attrs.get("pf_median_r", 0.0),
        score_total_r=attrs.get("score_total_r", attrs.get("single_stock_total_r", 0.0)),
        score_median_r=attrs.get("score_median_r", attrs.get("single_stock_median_r", 0.0)),
        bm_min_full_year_return_pct=attrs.get("bm_min_full_year_return_pct", 0.0),
        bm_min_month_return_pct=attrs.get("bm_min_month_return_pct", 0.0),
        bm_min_quarter_return_pct=attrs.get("bm_min_quarter_return_pct", 0.0),
        params_section_title="目前 trial 參數（非 final artifact ensemble）",
    )
    print(
        f"{colors['gray']}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | "
        f"年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | "
        f"保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | "
        f"完整年度數: {attrs.get('full_year_count', 0)} | "
        f"最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}% | "
        f"完整季度數: {attrs.get('full_quarter_count', 0)} | "
        f"最差完整季度: {attrs.get('min_quarter_return_pct', 0.0):.2f}% | "
        f"完整月度數: {attrs.get('full_month_count', 0)} | "
        f"最差完整月度: {attrs.get('min_month_return_pct', 0.0):.2f}%{colors['reset']}"
    )


def maybe_print_history_best(study, *, fixed_tp_percent, train_enable_rotation, train_max_positions, colors, best_trial_resolver=None, session=None):
    if len(study.trials) <= 0:
        return
    print(f"\n{colors['green']}✅ 已累積 {len(study.trials)} 次經驗。{colors['reset']}")
    resolver = get_best_completed_trial_or_none if best_trial_resolver is None else best_trial_resolver
    best_trial = resolver(study)
    if best_trial is None:
        print(f"{colors['gray']}ℹ️ 記憶庫目前尚無已完成 trial，略過歷史最佳儀表板還原。{colors['reset']}")
        return
    if not is_qualified_trial_value(best_trial.value):
        return
    if session is not None and bool(getattr(session, "disable_milestone_dashboard", False)):
        print(f"{colors['gray']}ℹ️ 非 rolling 訓練結果表格已關閉，略過歷史最佳儀表板還原。{colors['reset']}")
        return
    print(f"\n{colors['cyan']}📜 【歷史突破紀錄還原】{colors['reset']}")
    if session is not None:
        print_optimizer_trial_milestone_dashboard(
            session,
            best_trial,
            title="績效與風險對比表",
            milestone_title=f"🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試)",
        )
        return
    print(f"{colors['red']}🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試){colors['reset']}")
    print_best_trial_dashboard(
        best_trial,
        fixed_tp_percent=fixed_tp_percent,
        train_enable_rotation=train_enable_rotation,
        train_max_positions=train_max_positions,
        colors=colors,
    )


def export_best_params_if_requested(study, *, best_params_path, fixed_tp_percent, colors, best_trial_resolver=None, artifact_label=None, suppress_success_message: bool = False):
    if len(study.trials) == 0:
        print(f"{colors['red']}❌ 記憶庫為空，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1
    completed_trials = list_completed_study_trials(study)
    if not completed_trials:
        print(f"{colors['red']}❌ 目前記憶庫中尚無已完成紀錄，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1

    resolver = get_best_completed_trial_or_none if best_trial_resolver is None else best_trial_resolver
    best_trial = resolver(study)
    if best_trial is None:
        print(f"{colors['red']}❌ 目前記憶庫中尚無通過 local_min_score gate 的最佳 completed trial，無法匯出。{colors['reset']}", file=sys.stderr)
        return 1
    if is_qualified_trial_value(best_trial.value):
        best_params_payload = build_best_params_payload_from_trial(best_trial, fixed_tp_percent=fixed_tp_percent)
        with open(best_params_path, "w", encoding="utf-8") as handle:
            json.dump(best_params_payload, handle, indent=4, ensure_ascii=False)
        if not suppress_success_message:
            print(f"\n{colors['green']}💾 匯出成功！已從記憶庫提取最強參數！{colors['reset']}\n")
        return 0

    print(f"{colors['red']}❌ 目前記憶庫中尚無及格的紀錄，無法匯出。{colors['reset']}", file=sys.stderr)
    return 1


def resolve_training_session_export_policy(*, requested_n_trials, completed_session_trials, interrupted):
    requested_trials = int(requested_n_trials)
    completed_trials = int(completed_session_trials)
    if requested_trials <= 0:
        return True, "export_only"
    if completed_trials >= requested_trials:
        return True, "target_reached"
    if interrupted:
        return False, "interrupted_before_target"
    return False, "target_not_reached"


def resolve_run_request_or_exit(*, environ, resolve_optimizer_run_request, colors):
    try:
        return None, resolve_optimizer_run_request(environ)
    except ValueError as exc:
        print(f"{colors['red']}❌ {exc}{colors['reset']}", file=sys.stderr)
        return 1, None


def resolve_trial_count_or_exit(session, *, environ, resolve_optimizer_trial_count=None, colors, resolve_optimizer_run_request=None):
    if resolve_optimizer_run_request is None:
        from services.optimizer.study_utils import resolve_optimizer_run_request as _resolve_optimizer_run_request
        resolve_optimizer_run_request = _resolve_optimizer_run_request
    exit_code, request = resolve_run_request_or_exit(
        environ=environ,
        resolve_optimizer_run_request=resolve_optimizer_run_request,
        colors=colors,
    )
    if exit_code is not None:
        return exit_code, None
    session.n_trials = int(request["n_trials"])
    session.run_action = str(request.get("action", "train"))
    requested_model_mode = str(request.get("model_mode", "") or "").strip().lower()
    if requested_model_mode:
        session.requested_model_mode = requested_model_mode
    requested_study_scope = str(request.get("study_scope", "") or "").strip().lower()
    if requested_study_scope:
        session.requested_study_scope = requested_study_scope
    requested_study_db_action = str(request.get("study_db_action", "") or "").strip().lower()
    if requested_study_db_action:
        session.requested_study_db_action = requested_study_db_action
    return None, str(request.get("source", "unknown"))


def print_resolved_run_request(*, n_trials, action, source, colors):
    action_labels = {
        "train": f"訓練 {int(n_trials)} 次",
        "export_candidate": "輸出參數",
        "promote_candidate": "promote candidate",
        "outer_rolling_oos": "outer rolling monthly OOS test",
    }
    print(f"{colors['gray']}🎯 Optimizer 動作: {action_labels.get(str(action), str(action))} | 來源: {source}{colors['reset']}")


def print_resolved_trial_count(session, *, trial_source, colors):
    action = str(getattr(session, "run_action", "train"))
    print_resolved_run_request(
        n_trials=int(getattr(session, "n_trials", 0)),
        action=action,
        source=trial_source,
        colors=colors,
    )
