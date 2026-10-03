"""Config-driven Research strategy-parameter producer application.

This module is a thin orchestrator.  Scientific target selection lives in
``config/strategy_params.py``; Optimizer services remain the sole producers of
parameter truth.  Stable menu choices express work types only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from config.execution_policy import (
    DEFAULT_FIXED_RISK,
    DEFAULT_MAX_POSITION_CAP_PCT,
    DEFAULT_PORTFOLIO_MAX_POSITIONS,
    DEFAULT_PORTFOLIO_ROTATION,
)
from config.training_policy import (
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OPTIMIZER_RANDOM_SEED_DEFAULT,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)
from core.console_report import render_menu_item, render_table
from core.strategy_param_policy import (
    StrategyParamTrainingProfile,
    get_active_strategy_param_training_profile,
    get_strategy_param_fitting_seed_protocol,
    get_strategy_param_validation_protocol,
)
from core.training_policy import get_robustness_benchmark_policy_snapshot
from services.research.producer_execution import (
    PRODUCER_ACTION_CANCEL,
    PRODUCER_ACTION_FORCE_REBUILD,
    PRODUCER_ACTION_NORMAL,
    PRODUCER_ACTION_VIEW_REPORT,
    prompt_producer_execution_action,
)
from services.optimizer.strategy_param_service import (
    ensure_fixed_multiseed_strategy_parameter_artifact,
    ensure_fixed_multiseed_robustness_strategy_parameter_artifact,
    ensure_robustness_benchmark_strategy_parameter_artifact,
    inspect_fixed_multiseed_strategy_parameter_artifact,
    inspect_fixed_multiseed_robustness_strategy_parameter_artifact,
    ensure_strategy_parameter_artifact,
    ensure_validation_budget_robustness_strategy_parameter_artifact,
    ensure_validation_budget_strategy_parameter_artifact,
    inspect_robustness_benchmark_strategy_parameter_artifact,
    inspect_strategy_parameter_artifact,
    inspect_validation_budget_robustness_strategy_parameter_artifact,
    inspect_validation_budget_strategy_parameter_artifact,
)


@dataclass(frozen=True)
class StrategyParamWorkType:
    work_id: str
    label: str
    robustness: bool
    rolling: bool


WORK_TYPES: tuple[StrategyParamWorkType, ...] = (
    StrategyParamWorkType("forward_oos", "Forward OOS 參數訓練", False, False),
    StrategyParamWorkType("rolling_oos", "Rolling OOS 參數訓練", False, True),
    StrategyParamWorkType(
        "forward_oos_robustness", "Forward OOS Robustness 參數訓練", True, False
    ),
    StrategyParamWorkType(
        "rolling_oos_robustness", "Rolling OOS Robustness 參數訓練", True, True
    ),
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _required_schedule_end(profile: StrategyParamTrainingProfile, work: StrategyParamWorkType) -> str:
    return profile.coverage_end_date if work.rolling else profile.schedule_start_date


def _validation_protocol(profile: StrategyParamTrainingProfile):
    if not profile.validation_enabled:
        return None
    protocol_id = str(profile.validation_protocol or "").strip()
    protocol = get_strategy_param_validation_protocol(protocol_id)
    if protocol.mode not in {"fixed_checkpoints", "adaptive_plateau"}:
        raise ValueError(
            "目前Optimizer尚未註冊此validation capability handler: "
            f"{protocol.protocol_id}"
        )
    return protocol


def _fitting_seed_protocol(profile: StrategyParamTrainingProfile):
    return get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)


def _single_state(profile: StrategyParamTrainingProfile, *, required_end: str) -> dict:
    protocol = _validation_protocol(profile)
    seed_protocol = _fitting_seed_protocol(profile)
    if protocol is not None and seed_protocol.enabled:
        raise ValueError("目前尚未註冊multi-seed + validation的共同producer；此能力屬後續研究輪次")
    if seed_protocol.enabled:
        return inspect_fixed_multiseed_strategy_parameter_artifact(
            PROJECT_ROOT,
            training_profile_id=profile.profile_id,
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
    if protocol is not None:
        return inspect_validation_budget_strategy_parameter_artifact(
            PROJECT_ROOT,
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=profile.parameter_set_identity,
            validation_protocol_id=protocol.protocol_id,
        )
    return inspect_strategy_parameter_artifact(
        PROJECT_ROOT,
        family=profile.filter_profile,
        evaluation_mode="rolling",
        policy=profile.parameter_policy,
        comparison_end_date=required_end,
        dataset=profile.dataset,
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
    )


def _ensure_single(
    profile: StrategyParamTrainingProfile, *, required_end: str, force_rebuild: bool = False
) -> dict:
    protocol = _validation_protocol(profile)
    seed_protocol = _fitting_seed_protocol(profile)
    if protocol is not None and seed_protocol.enabled:
        raise ValueError("目前尚未註冊multi-seed + validation的共同producer；此能力屬後續研究輪次")
    if seed_protocol.enabled:
        return ensure_fixed_multiseed_strategy_parameter_artifact(
            PROJECT_ROOT,
            training_profile_id=profile.profile_id,
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            force_rebuild=bool(force_rebuild),
            resume_parameter_training=not bool(force_rebuild),
        )
    if protocol is not None:
        return ensure_validation_budget_strategy_parameter_artifact(
            PROJECT_ROOT,
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=profile.parameter_set_identity,
            validation_protocol_id=protocol.protocol_id,
            force_rebuild=bool(force_rebuild),
            resume_parameter_training=not bool(force_rebuild),
        )
    return ensure_strategy_parameter_artifact(
        PROJECT_ROOT,
        family=profile.filter_profile,
        evaluation_mode="rolling",
        policy=profile.parameter_policy,
        comparison_end_date=required_end,
        dataset=profile.dataset,
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        force_rebuild=bool(force_rebuild),
        resume_parameter_training=not bool(force_rebuild),
    )


def _benchmark_state(
    profile: StrategyParamTrainingProfile, *, seed: int, required_end: str
) -> dict:
    benchmark = get_robustness_benchmark_policy_snapshot()
    seed_protocol = _fitting_seed_protocol(profile)
    if seed_protocol.enabled:
        return inspect_fixed_multiseed_robustness_strategy_parameter_artifact(
            PROJECT_ROOT,
            training_profile_id=profile.profile_id,
            benchmark_id=str(benchmark["benchmark_id"]),
            replication_seed=int(seed),
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        )
    protocol = _validation_protocol(profile)
    if protocol is not None:
        return inspect_validation_budget_robustness_strategy_parameter_artifact(
            PROJECT_ROOT,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=int(seed),
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            parameter_set_identity=profile.parameter_set_identity,
            validation_protocol_id=protocol.protocol_id,
        )
    return inspect_robustness_benchmark_strategy_parameter_artifact(
        PROJECT_ROOT,
        benchmark_id=str(benchmark["benchmark_id"]),
        seed=int(seed),
        family=profile.filter_profile,
        evaluation_mode="rolling",
        policy=profile.parameter_policy,
        comparison_end_date=required_end,
        dataset=profile.dataset,
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
    )


def _ensure_benchmark(
    profile: StrategyParamTrainingProfile, *, seed: int, required_end: str,
    force_rebuild: bool = False,
) -> dict:
    benchmark = get_robustness_benchmark_policy_snapshot()
    seed_protocol = _fitting_seed_protocol(profile)
    if seed_protocol.enabled:
        return ensure_fixed_multiseed_robustness_strategy_parameter_artifact(
            PROJECT_ROOT,
            training_profile_id=profile.profile_id,
            benchmark_id=str(benchmark["benchmark_id"]),
            replication_seed=int(seed),
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            force_rebuild=bool(force_rebuild),
            resume_parameter_training=not bool(force_rebuild),
            quiet=False,
        )
    protocol = _validation_protocol(profile)
    if protocol is not None:
        return ensure_validation_budget_robustness_strategy_parameter_artifact(
            PROJECT_ROOT,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=int(seed),
            policy=profile.parameter_policy,
            comparison_end_date=required_end,
            dataset=profile.dataset,
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            resume_parameter_training=True,
            quiet=False,
            parameter_set_identity=profile.parameter_set_identity,
            validation_protocol_id=protocol.protocol_id,
            force_rebuild=bool(force_rebuild),
        )
    return ensure_robustness_benchmark_strategy_parameter_artifact(
        PROJECT_ROOT,
        benchmark_id=str(benchmark["benchmark_id"]),
        seed=int(seed),
        family=profile.filter_profile,
        evaluation_mode="rolling",
        policy=profile.parameter_policy,
        comparison_end_date=required_end,
        dataset=profile.dataset,
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        quiet=False,
        force_rebuild=bool(force_rebuild),
        resume_parameter_training=not bool(force_rebuild),
    )


def _render_profile_summary(profile: StrategyParamTrainingProfile) -> str:
    protocol = _validation_protocol(profile)
    seed_protocol = _fitting_seed_protocol(profile)
    rows = [
        ("Training Profile", profile.profile_id),
        ("Strategy", profile.strategy_id),
        ("Filter Profile", profile.filter_profile),
        ("Param Policy", profile.parameter_policy),
        ("Validation", "ON" if protocol is not None else "OFF"),
        ("Parameter Set", profile.parameter_set_identity),
        ("Dataset", profile.dataset),
        ("Fitting Seeds", str(seed_protocol.seed_count)),
        ("Seed Mode", seed_protocol.seed_mode),
        ("Trials / Seed", str(seed_protocol.trials_per_seed)),
        ("Schedule Start", profile.schedule_start_date),
        ("Coverage End", profile.coverage_end_date),
    ]
    if protocol is not None:
        rows.extend(
            (
                ("Validation Protocol", protocol.protocol_id),
                ("Validation Holdout", f"{protocol.holdout_months} months"),
            )
        )
        if protocol.mode == "fixed_checkpoints":
            rows.append(
                ("Budget Checkpoints", " / ".join(str(v) for v in protocol.checkpoints))
            )
        else:
            if protocol.observation_start_trial is not None and protocol.min_trials_before_stop is not None:
                rows.extend(
                    (
                        ("Observation Start Trial", str(int(protocol.observation_start_trial))),
                        ("Min Trials Before Stop", str(int(protocol.min_trials_before_stop))),
                    )
                )
            else:
                rows.append(("Legacy Validation Warmup", str(int(protocol.warmup_trials or 0))))
            rows.extend(
                (
                    ("Plateau Patience", str(int(protocol.patience_trials or 0))),
                    ("Observation Interval", str(int(protocol.observation_interval or 0))),
                    ("Min Delta", str(float(protocol.min_delta or 0.0))),
                )
            )
    if seed_protocol.enabled:
        rows.append(("Fixed Seed Bundle", ", ".join(str(v) for v in seed_protocol.resolved_seeds)))
    return render_table(("設定", "目前值"), rows)


def _build_plan(profile: StrategyParamTrainingProfile, work: StrategyParamWorkType) -> dict:
    required_end = _required_schedule_end(profile, work)
    benchmark = get_robustness_benchmark_policy_snapshot()
    seed_protocol = _fitting_seed_protocol(profile)
    rows: list[dict] = []
    if work.robustness:
        for seed in tuple(int(v) for v in benchmark["resolved_seeds"]):
            state = _benchmark_state(profile, seed=seed, required_end=required_end)
            rows.append(
                {
                    "action": str(state.get("action") or "BUILD"),
                    "item": (
                        f"Replicate {seed} → {seed_protocol.seed_count}-seed fitting bundle"
                        if seed_protocol.enabled else f"Seed {seed}"
                    ),
                    "status": str(state.get("status") or ""),
                    "path": _relative(state.get("path") or state.get("target") or ""),
                    "seed": seed,
                }
            )
    else:
        state = _single_state(profile, required_end=required_end)
        item = (
            f"Fixed fitting seeds {seed_protocol.seed_count}"
            if seed_protocol.enabled
            else f"Canonical seed {OPTIMIZER_RANDOM_SEED_DEFAULT}"
        )
        rows.append(
            {
                "action": str(state.get("action") or "BUILD"),
                "item": item,
                "status": str(state.get("status") or ""),
                "path": _relative(state.get("path") or ""),
                "seed": None,
            }
        )
    ready = all(row["action"] == "REUSE" for row in rows)
    return {
        "work": work,
        "profile": profile,
        "required_end": required_end,
        "benchmark": benchmark,
        "rows": rows,
        "status": "READY" if ready else "PREPARABLE",
    }


def render_strategy_param_plan(plan: dict) -> str:
    profile: StrategyParamTrainingProfile = plan["profile"]
    work: StrategyParamWorkType = plan["work"]
    protocol = _validation_protocol(profile)
    seed_protocol = _fitting_seed_protocol(profile)
    lines = [
        "====================================================================================================",
        f" Strategy Parameter Training | {work.label}",
        "====================================================================================================",
        "",
        f"整體狀態            ：{plan['status']}",
        f"Training Profile     ：{profile.profile_id}",
        f"Filter Profile       ：{profile.filter_profile}",
        f"Param Policy         ：{profile.parameter_policy}",
        f"Validation           ：{'ON' if protocol is not None else 'OFF'}",
        f"Parameter Set        ：{profile.parameter_set_identity}",
        f"Required coverage end：{plan['required_end']}",
    ]
    if protocol is not None:
        lines.extend(
            (
                f"Selection window      ：{OUTER_ROLLING_TRAIN_WINDOW_MONTHS}m",
                f"Inner validation      ：{int(protocol.holdout_months)}m",
            )
        )
        if work.rolling:
            lines.extend(
                (
                    f"Parameter cadence     ：{OUTER_ROLLING_OOS_HORIZON_MONTHS}m",
                    f"Schedule coverage     ：{profile.schedule_start_date} ~ {plan['required_end']}",
                )
            )
        else:
            lines.extend(
                (
                    f"Parameter effective   ：{profile.schedule_start_date}",
                    f"Final Forward OOS     ：{profile.schedule_start_date} ~ {profile.coverage_end_date}",
                )
            )
    else:
        lines.append(
            f"Train/OOS            ：{OUTER_ROLLING_TRAIN_WINDOW_MONTHS}m / {OUTER_ROLLING_OOS_HORIZON_MONTHS}m"
        )
    lines.extend(
        (
            f"Fitting seeds        ：{seed_protocol.seed_count}",
            f"Seed mode            ：{seed_protocol.seed_mode}",
            f"Trials/seed          ：{seed_protocol.trials_per_seed}",
        )
    )
    if work.robustness:
        benchmark = dict(plan["benchmark"])
        lines.extend(
            (
                f"Benchmark ID         ：{benchmark['benchmark_id']}",
                f"Benchmark seeds      ：{len(benchmark['resolved_seeds'])}",
            )
        )
    lines.extend(("", render_table(
        ("動作", "項目", "說明"),
        [
            (
                row["action"],
                row["item"],
                row["status"] or row["path"] or "canonical Optimizer parameter artifact",
            )
            for row in plan["rows"]
        ],
    )))
    return "\n".join(lines)


def _load_training_evidence(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def render_strategy_param_training_report(plan: dict) -> str:
    """Render persisted canonical training evidence without rerunning Optimizer."""

    lines = [
        "====================================================================================================",
        f" Strategy Parameter Training Report | {plan['work'].label}",
        "====================================================================================================",
    ]
    rendered_any = False
    for row in plan["rows"]:
        payload = _load_training_evidence(row.get("path") or "")
        if payload is None:
            lines.extend(("", f"[{row['item']}] 尚無可讀 canonical parameter artifact。"))
            continue
        folds = list(payload.get("folds") or [])
        adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
        lines.extend((
            "",
            f"[{row['item']}]",
            f"Parameter Set ：{adaptation.get('parameter_set') or plan['profile'].parameter_set_identity}",
            f"Protocol      ：{adaptation.get('optimizer_search_protocol') or '-'}",
            f"Artifact      ：{row.get('path') or '-'}",
        ))
        if not folds:
            lines.append("Training protocol evidence：此工件沒有 fold-level validation evidence。")
            continue
        for index, fold in enumerate(folds, start=1):
            protocol = dict((fold or {}).get("training_protocol") or {})
            if not protocol:
                continue
            rendered_any = True
            lines.extend((
                "",
                f"Fold {index} | OOS {fold.get('oos_start_date') or '-'} ～ {fold.get('oos_end_date') or '-'}",
                f"Validation    ：{protocol.get('validation_start_date') or '-'} ～ {protocol.get('validation_end_date') or '-'}",
                f"Selected N*  ：{protocol.get('selected_trials') or '-'}",
                f"Selected Score：{('-' if protocol.get('selected_validation_score') is None else f"{float(protocol.get('selected_validation_score')):.4f}")}",
                f"OOS feedback ：{bool(protocol.get('oos_feedback_used', False))}",
            ))
            checkpoints = list(protocol.get("checkpoint_results") or [])
            if checkpoints:
                lines.append(render_table(
                    ("Trials", "Available", "Validation Score", "RoMD", "Return", "MDD", "Members"),
                    [
                        (
                            str(item.get("trials") or "-"),
                            "Y" if bool(item.get("available")) else "N",
                            "-" if item.get("validation_score") is None else f"{float(item['validation_score']):.4f}",
                            "-" if item.get("plain_romd_score") is None else f"{float(item['plain_romd_score']):.2f}",
                            "-" if item.get("return_pct") is None else f"{float(item['return_pct']):.2f}%",
                            "-" if item.get("mdd_pct") is None else f"{float(item['mdd_pct']):.2f}%",
                            str(item.get("member_count") or 0),
                        )
                        for item in checkpoints
                    ],
                ))
            observations = list(protocol.get("observation_results") or [])
            if observations:
                lines.extend((
                    f"Stop           ：{protocol.get('stop_trial') or '-'} / {protocol.get('stop_reason') or '-'}",
                    f"Last Improve   ：{protocol.get('last_improvement_trial') or '-'}",
                    f"Observations   ：{int(protocol.get('observation_count') or len(observations))}",
                    f"Validation Replay / Reuse：{int(protocol.get('actual_validation_replays') or 0)} / {int(protocol.get('reused_validation_observations') or 0)}",
                    f"Validation Wall：{float(protocol.get('validation_wall_sec') or 0.0):.2f}s",
                ))
                lines.append(render_table(
                    ("Trials", "Source", "Improved", "Validation Score", "Members"),
                    [
                        (
                            str(item.get("trials") or "-"),
                            str(item.get("validation_source") or "-"),
                            "Y" if bool(item.get("improved")) else "N",
                            "-" if item.get("validation_score") is None else f"{float(item['validation_score']):.4f}",
                            str(item.get("member_count") or 0),
                        )
                        for item in observations
                    ],
                ))
    if not rendered_any:
        lines.extend(("", "目前沒有可顯示的 fold-level validation checkpoint evidence。"))
    return "\n".join(lines)


def _execute_plan(plan: dict, *, force_rebuild: bool = False) -> dict:
    profile: StrategyParamTrainingProfile = plan["profile"]
    work: StrategyParamWorkType = plan["work"]
    required_end = str(plan["required_end"])
    if work.robustness:
        for row in plan["rows"]:
            if row["action"] == "REUSE" and not force_rebuild:
                print(f"[REUSE] {row['item']}")
                continue
            action = "FORCE_REBUILD" if force_rebuild else row["action"]
            print(f"[{action}] {row['item']}")
            _ensure_benchmark(
                profile, seed=int(row["seed"]), required_end=required_end,
                force_rebuild=bool(force_rebuild),
            )
    else:
        row = plan["rows"][0]
        action = "FORCE_REBUILD" if force_rebuild else row["action"]
        print(f"[{action}] {row['item']}")
        _ensure_single(
            profile, required_end=required_end, force_rebuild=bool(force_rebuild)
        )
    final_plan = _build_plan(profile, work)
    if final_plan["status"] != "READY":
        pending = ", ".join(
            f"{row['item']}:{row['action']}" for row in final_plan["rows"] if row["action"] != "REUSE"
        )
        raise RuntimeError(f"Strategy parameter producer完成後仍未READY: {pending}")
    print("\n[DONE] Strategy parameter training prerequisites READY")
    print(
        render_table(
            ("項目", "狀態", "工件"),
            [(row["item"], "READY", row["path"]) for row in final_plan["rows"]],
        )
    )
    return final_plan


def _run_work_type(work: StrategyParamWorkType) -> int:
    profile = get_active_strategy_param_training_profile()
    plan = _build_plan(profile, work)
    print("\n" + render_strategy_param_plan(plan))
    if plan["status"] == "BLOCKED":
        print("\n目前此work type尚未具備合法producer contract；不執行。")
        return 0
    can_view = any(bool(row.get("path")) and (PROJECT_ROOT / row["path"]).is_file() for row in plan["rows"])
    choice = prompt_producer_execution_action(
        ready=plan["status"] == "READY",
        can_view_report=can_view,
    )
    if choice.action == PRODUCER_ACTION_CANCEL:
        return 0
    if choice.action == PRODUCER_ACTION_VIEW_REPORT:
        print("\n" + render_strategy_param_training_report(plan))
        return 0
    if choice.action == PRODUCER_ACTION_FORCE_REBUILD:
        _execute_plan(plan, force_rebuild=True)
        refreshed = _build_plan(profile, work)
        print("\n" + render_strategy_param_training_report(refreshed))
        return 0
    if choice.action == PRODUCER_ACTION_NORMAL:
        _execute_plan(plan, force_rebuild=False)
        return 0
    raise RuntimeError(f"未知 producer execution action: {choice.action}")


def show_strategy_param_status() -> None:
    profile = get_active_strategy_param_training_profile()
    print(_render_profile_summary(profile))
    rows = []
    for work in WORK_TYPES:
        plan = _build_plan(profile, work)
        reusable = sum(1 for row in plan["rows"] if row["action"] == "REUSE")
        rows.append((work.label, plan["status"], f"{reusable}/{len(plan['rows'])} READY"))
    print("\n" + render_table(("工作類型", "狀態", "工件"), rows))


def run_strategy_param_training_menu(*, advanced_handler: Callable[[], int]) -> int:
    while True:
        profile = get_active_strategy_param_training_profile()
        print("\n====================================================================================================")
        print(" 策略參數最佳化")
        print("====================================================================================================")
        protocol = _validation_protocol(profile)
        seed_protocol = _fitting_seed_protocol(profile)
        print(f"Training Profile ：{profile.profile_id}")
        print(f"Filter Profile   ：{profile.filter_profile}")
        print(f"Param Policy     ：{profile.parameter_policy}")
        print(f"Validation       ：{'ON | ' + protocol.protocol_id if protocol else 'OFF'}")
        print(f"Parameter Set    ：{profile.parameter_set_identity}")
        print(f"Fitting Seeds    ：{seed_protocol.seed_count} | {seed_protocol.seed_mode} | {seed_protocol.trials_per_seed} trials/seed")
        print()
        for index, work in enumerate(WORK_TYPES, start=1):
            print(render_menu_item(index, work.label, default=(index == 1)))
        print(render_menu_item(5, "進階／單次 Optimizer"))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice in {"1", "2", "3", "4"}:
            _run_work_type(WORK_TYPES[int(choice) - 1])
        elif choice == "5":
            advanced_handler()
        else:
            print("選項無效，請按 Enter 或輸入 0～5。")


__all__ = [
    "WORK_TYPES",
    "render_strategy_param_plan",
    "render_strategy_param_training_report",
    "run_strategy_param_training_menu",
    "show_strategy_param_status",
]
