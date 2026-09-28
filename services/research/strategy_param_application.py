"""Config-driven Research strategy-parameter producer application.

This module is a thin orchestrator.  Scientific target selection lives in
``config/strategy_params.py``; Optimizer services remain the sole producers of
parameter truth.  Stable menu choices express work types only.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    get_strategy_param_validation_protocol,
)
from core.training_policy import get_robustness_benchmark_policy_snapshot
from services.optimizer.strategy_param_service import (
    ensure_robustness_benchmark_strategy_parameter_artifact,
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
    if protocol.protocol_id != "validation_budget_refit_v1":
        raise ValueError(
            "目前Optimizer尚未註冊此validation capability handler: "
            f"{protocol.protocol_id}"
        )
    return protocol


def _single_state(profile: StrategyParamTrainingProfile, *, required_end: str) -> dict:
    protocol = _validation_protocol(profile)
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


def _ensure_single(profile: StrategyParamTrainingProfile, *, required_end: str) -> dict:
    protocol = _validation_protocol(profile)
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
    )


def _benchmark_state(
    profile: StrategyParamTrainingProfile, *, seed: int, required_end: str
) -> dict:
    benchmark = get_robustness_benchmark_policy_snapshot()
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
    profile: StrategyParamTrainingProfile, *, seed: int, required_end: str
) -> dict:
    benchmark = get_robustness_benchmark_policy_snapshot()
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
    )


def _render_profile_summary(profile: StrategyParamTrainingProfile) -> str:
    protocol = _validation_protocol(profile)
    rows = [
        ("Training Profile", profile.profile_id),
        ("Strategy", profile.strategy_id),
        ("Filter Profile", profile.filter_profile),
        ("Param Policy", profile.parameter_policy),
        ("Validation", "ON" if protocol is not None else "OFF"),
        ("Parameter Set", profile.parameter_set_identity),
        ("Dataset", profile.dataset),
        ("Canonical Seed", str(OPTIMIZER_RANDOM_SEED_DEFAULT)),
        ("Schedule Start", profile.schedule_start_date),
        ("Coverage End", profile.coverage_end_date),
    ]
    if protocol is not None:
        rows.extend(
            (
                ("Validation Protocol", protocol.protocol_id),
                ("Validation Holdout", f"{protocol.holdout_months} months"),
                ("Budget Checkpoints", " / ".join(str(v) for v in protocol.checkpoints)),
            )
        )
    return render_table(("設定", "目前值"), rows)


def _build_plan(profile: StrategyParamTrainingProfile, work: StrategyParamWorkType) -> dict:
    required_end = _required_schedule_end(profile, work)
    benchmark = get_robustness_benchmark_policy_snapshot()
    rows: list[dict] = []
    if work.robustness:
        for seed in tuple(int(v) for v in benchmark["resolved_seeds"]):
            state = _benchmark_state(profile, seed=seed, required_end=required_end)
            rows.append(
                {
                    "action": str(state.get("action") or "BUILD"),
                    "item": f"Seed {seed}",
                    "status": str(state.get("status") or ""),
                    "path": _relative(state.get("path") or state.get("target") or ""),
                    "seed": seed,
                }
            )
    else:
        state = _single_state(profile, required_end=required_end)
        rows.append(
            {
                "action": str(state.get("action") or "BUILD"),
                "item": f"Canonical seed {OPTIMIZER_RANDOM_SEED_DEFAULT}",
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
        f"Train/OOS            ：{OUTER_ROLLING_TRAIN_WINDOW_MONTHS}m / {OUTER_ROLLING_OOS_HORIZON_MONTHS}m",
        f"Trials/fold          ：{OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT}",
    ]
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


def _execute_plan(plan: dict) -> dict:
    profile: StrategyParamTrainingProfile = plan["profile"]
    work: StrategyParamWorkType = plan["work"]
    required_end = str(plan["required_end"])
    if work.robustness:
        for row in plan["rows"]:
            if row["action"] == "REUSE":
                print(f"[REUSE] {row['item']}")
                continue
            print(f"[{row['action']}] {row['item']}")
            _ensure_benchmark(profile, seed=int(row["seed"]), required_end=required_end)
    else:
        row = plan["rows"][0]
        print(f"[{row['action']}] {row['item']}")
        _ensure_single(profile, required_end=required_end)
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
    if plan["status"] == "READY":
        print("\n目前所需策略參數工件已全部 READY；本次不需執行 Optimizer。")
        return 0
    try:
        choice = input("👉 按 Enter 執行全部必要 BUILD／REBUILD／RESUME；輸入 0 返回：").strip().lower()
    except EOFError:
        print("\n輸入已結束，本次不執行。")
        return 0
    if choice in {"0", "q", "quit", "exit"}:
        print("已取消本次執行。")
        return 0
    if choice not in {"", "1"}:
        print("輸入無效，本次不執行。")
        return 0
    _execute_plan(plan)
    return 0


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
        print(f"Training Profile ：{profile.profile_id}")
        print(f"Filter Profile   ：{profile.filter_profile}")
        print(f"Param Policy     ：{profile.parameter_policy}")
        print(f"Validation       ：{'ON | ' + protocol.protocol_id if protocol else 'OFF'}")
        print(f"Parameter Set    ：{profile.parameter_set_identity}")
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
    "run_strategy_param_training_menu",
    "show_strategy_param_status",
]
