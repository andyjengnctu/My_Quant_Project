"""Independent PIT Daily Min/P19 4D NetPnL teacher (Rank and Positive).

This is not the historical Breakout Full/P17 CZ/DA source, nor TrueDaily Min
CU/CT. Only the canonical Optimizer may build parameters; the two model heads
consume identical matured daily-stockday NetPnL from the same signed teacher.

P19's approved parameter protocol is single Optimizer seed 42. Different DL
FIT seeds share that one economic teacher; they never silently choose other
optimizer seeds or reuse a Min/Breakout parameter source. Historical 2014-2020
parameters are independently Daily Min optimized on past-only data.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from functools import partial
from pathlib import Path

import pandas as pd

from config.research import RESEARCH_SINGLE_SEED
from config.execution_policy import (
    DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT,
    DEFAULT_PORTFOLIO_MAX_POSITIONS, DEFAULT_PORTFOLIO_ROTATION,
)
from core.file_integrity import canonical_json_sha256, load_json_strict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.strategy_param_artifacts import compute_strategy_param_file_sha256
from filters.breakout_quality.daily_min_p19_target import (
    verify_daily_min_p19_params, daily_min_p19_execution_sha256,
)
from filters.breakout_quality.daily_trade40_target import (
    build_formal_min_execution_schedule, resolve_formal_min_param_records,
)
from filters.breakout_quality.strategy_compare_sources import (
    PARAM_POLICY_BASE_FINALIST_BEST, load_strategy_param_evaluation_view,
)
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_BASE_FILTER_ID,
    TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
    TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
)
from services.breakout_quality.daily_min_p19_source_identity import SOURCE_ID
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked, FormalMinTrade40PITSource,
)
from services.breakout_quality.formal_mintrade40_training_source import (
    _current_p16_seed_artifact, inspect_formal_mintrade40_p16_teacher,
)
from services.optimizer.strategy_param_service import _canonical_schedule_build_contract
from services.optimizer.optimizer_workspace_recovery import (
    nested_optimizer_workspace_candidate, inspect_nested_optimizer_workspace,
    adopt_nested_optimizer_workspace, verify_optimizer_runtime_preflight,
)
from services.optimizer.strategy_param_training import (
    prepare_selection_historical_p2_params,
    _build_min_roos_schedule_contract, build_min_roos_fold_overrides,
    _runtime_contract,
)
from strategies.daily.search_space import resolve_daily_search_profile

HISTORY_ID = "DAILY_MIN_P19_LABEL_HISTORY_V1"
PROFILE = "min_atr_limit_v2"
BASELINE_ARM = "C166"
TEACHER_SEED = int(RESEARCH_SINGLE_SEED)
ENTRY_POLICY = "next_exchange_session_daily_min_p19_atr_limit_v1"


def historical_params_path(project_root: str | Path) -> Path:
    return (Path(project_root).resolve() / "outputs" / "optimizer"
            / "daily_min_p19_label_history" / f"seed_{TEACHER_SEED}"
            / "p2_dl_off_trained" / "active_params" / "roos_base_best.json")


def _expected_historical_preflight(root: Path) -> dict[str, object]:
    """Derive the expected signed preflight with the canonical Optimizer owner.

    Merely checking a receipt against its own claimed hash is insufficient:
    neither a changed P19 search protocol nor a changed source dataset may
    silently turn a stale historical teacher into a current FIT source.
    """
    policy = _canonical_schedule_build_contract(family="min")
    search = resolve_daily_search_profile(PROFILE)
    schedule = _build_min_roos_schedule_contract(
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(policy["train_window_months"]),
        oos_months=int(policy["oos_horizon_months"]),
    )
    reference = {
        **schedule["reference_payload"],
        "search_fields": list(search["fields"]), "entry_strategy": "Daily",
        "entry_universe": "unscored_all_adjacent_quote_stockdays_v1",
        "breakout_required": False,
        "daily_execution_contract": search["execution_contract"],
    }
    schedule = {
        **schedule, "reference_type": "canonical_daily_min_roos_fixed_defaults",
        "reference_payload": reference, "sha256": canonical_json_sha256(reference),
    }
    args = SimpleNamespace(
        dataset="full", trials_per_fold=int(policy["trials_per_fold"]),
        strategy_id="daily", daily_search_profile=PROFILE,
        filter_id=TRADE_PATH_BASE_FILTER_ID,
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        canonical_strategy_param_family="", optimizer_search_protocol="",
    )
    overrides = build_min_roos_fold_overrides(
        baseline_contract=schedule, args=args, training_dl_enabled=False,
    )
    return _runtime_contract(
        root=root, args=args, baseline_contract=schedule,
        fold_overrides=overrides, model_artifact=None,
        arm_id=HISTORY_ID, training_dl_enabled=False, binary_pit=None,
    )


def _validated_records(path: Path, *, historical: bool, project_root: str | Path):
    if not path.is_file():
        raise FormalMinTrade40PITBlocked(f"Missing signed P19 parameter artifact: {path.name}")
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise FormalMinTrade40PITBlocked("P19 parameter source must be a JSON object")
    if historical:
        adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
        search = resolve_daily_search_profile(PROFILE)
        if (adaptation.get("mode") != "min_roos_training"
                or adaptation.get("parameter_set") != HISTORY_ID
                or adaptation.get("training_dl_enabled") is not False
                or adaptation.get("entry_strategy") != "Daily"
                or adaptation.get("fixed_rule_contract") != "all_rule_filters_off"
                or adaptation.get("daily_execution_contract") != search["execution_contract"]
                or tuple(adaptation.get("search_fields") or ()) != tuple(search["fields"])):
            raise FormalMinTrade40PITBlocked("Historical P19 teacher is not independently Daily Min 4D")
        expected = _expected_historical_preflight(Path(project_root).resolve())
        preflight = verify_optimizer_runtime_preflight(
            path.parent.parent / "rolling_preflight.json",
            expected_runtime_identity_sha256=expected["runtime_identity_sha256"],
        )
        summary = load_json_strict(path.parent.parent / "rolling_optimizer_summary.json")
        if not isinstance(summary, dict):
            raise FormalMinTrade40PITBlocked("Missing signed historical P19 Optimizer evidence")
        if (preflight.get("runtime_identity_sha256") != summary.get("runtime_identity_sha256")
                or preflight.get("runtime_identity_sha256") != expected["runtime_identity_sha256"]
                or summary.get("status") != "COMPLETED"
                or preflight.get("arm_id") != HISTORY_ID
                or preflight.get("training_dl_enabled") is not False):
            raise FormalMinTrade40PITBlocked("Historical P19 provenance/preflight mismatch")
    records = resolve_formal_min_param_records("rolling_active_param_ensemble", payload)
    for row in records:
        members = tuple(row.get("members") or ())
        if len(members) != 1 or members[0].get("seed") != TEACHER_SEED:
            raise FormalMinTrade40PITBlocked("P19 parameter teacher seed/member mismatch")
        params = row["params_obj"]
        verify_daily_min_p19_params(params)
        if (bool(params.use_breakout_buy) or bool(params.use_breakout_quality_filter)
                or bool(params.use_breakout_quality_ranking)
                or build_portfolio_params_signature(params) != row["params_signature"]):
            raise FormalMinTrade40PITBlocked("P19 teacher changed Daily Min execution/score semantics")
    return payload, records


def inspect_history(*, project_root: str | Path, seed: int) -> dict[str, object]:
    root = Path(project_root).resolve()
    path = historical_params_path(root)
    if seed <= 0:
        raise ValueError("DL seed must be positive")
    try:
        nested_candidate = nested_optimizer_workspace_candidate(path.parent.parent)
    except ValueError as exc:
        return {"path": path, "action": "BLOCKED", "reason": f"P19 nested history invalid: {exc}"}
    if path.is_file():
        if nested_candidate is not None:
            return {"path": path, "action": "BLOCKED",
                    "reason": "Canonical and nested P19 Optimizer artifacts coexist"}
        try:
            _validated_records(path, historical=True, project_root=root)
            return {"path": path, "action": "REUSE", "reason": "Verified P19 4D historical teacher"}
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return {"path": path, "action": "BLOCKED", "reason": f"P19 history invalid: {exc}"}
    # A prior P19 run accidentally supplied the arm directory as the Optimizer
    # work root, causing an extra p2_dl_off_trained/ level. Inspect it without
    # side effects; only the canonical Optimizer owner may relocate it on RUN.
    try:
        if nested_candidate is not None:
            expected = _expected_historical_preflight(root)
            nested = inspect_nested_optimizer_workspace(
                canonical_arm_dir=path.parent.parent,
                expected_runtime_identity_sha256=expected["runtime_identity_sha256"],
            )
            if nested is None:
                raise FormalMinTrade40PITBlocked("Nested Optimizer workspace disappeared")
            legacy_params = nested / "active_params" / path.name
            if legacy_params.is_file():
                _validated_records(legacy_params, historical=True, project_root=root)
            else:
                summary_path = nested / "rolling_optimizer_summary.json"
                if summary_path.is_file():
                    summary = load_json_strict(summary_path)
                    if not isinstance(summary, dict) or summary.get("status") == "COMPLETED":
                        raise FormalMinTrade40PITBlocked(
                            "Completed nested Optimizer summary missing signed parameters"
                        )
            return {"path": path, "action": "RESUME",
                    "reason": "Verified nested P19 Optimizer workspace; relocate without refitting completed folds"}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return {"path": path, "action": "BLOCKED", "reason": f"P19 nested history invalid: {exc}"}
    preflight = path.parent.parent / "rolling_preflight.json"
    if preflight.is_file():
        try:
            expected = _expected_historical_preflight(root)
            verify_optimizer_runtime_preflight(
                preflight, expected_runtime_identity_sha256=expected["runtime_identity_sha256"],
            )
            summary_path = path.parent.parent / "rolling_optimizer_summary.json"
            if summary_path.is_file():
                summary = load_json_strict(summary_path)
                if not isinstance(summary, dict) or summary.get("status") == "COMPLETED":
                    raise FormalMinTrade40PITBlocked(
                        "Completed P19 Optimizer summary missing signed parameters"
                    )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return {"path": path, "action": "BLOCKED",
                    "reason": f"Interrupted P19 history incompatible: {exc}"}
    return {"path": path, "action": "RESUME" if preflight.is_file() else "BUILD",
            "reason": "Independent P19 historical Daily Min Optimizer teacher required"}

def execute_history(*, project_root: str | Path, seed: int):
    root = Path(project_root).resolve()
    state = inspect_history(project_root=root, seed=seed)
    if state["action"] == "BLOCKED":
        raise FormalMinTrade40PITBlocked(str(state["reason"]))
    if state["action"] == "REUSE":
        return Path(state["path"])
    path = historical_params_path(root)
    # The read-only plan identified a valid legacy nested workspace. Normalize
    # through the Optimizer's own recovery owner and re-check signed readiness.
    if nested_optimizer_workspace_candidate(path.parent.parent) is not None:
        expected = _expected_historical_preflight(root)
        adopted = adopt_nested_optimizer_workspace(
            canonical_arm_dir=path.parent.parent,
            expected_runtime_identity_sha256=expected["runtime_identity_sha256"],
        )
    else:
        adopted = False
    if adopted:
        state = inspect_history(project_root=root, seed=seed)
        if state["action"] == "BLOCKED":
            raise FormalMinTrade40PITBlocked(str(state["reason"]))
        if state["action"] == "REUSE":
            return path
    policy = _canonical_schedule_build_contract(family="min")
    result = prepare_selection_historical_p2_params(
        project_root=root, dataset="full", param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        trials_per_fold=int(policy["trials_per_fold"]),
        max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
        rotation=str(DEFAULT_PORTFOLIO_ROTATION),
        fixed_risk=float(DEFAULT_FIXED_RISK),
        max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
        optimizer_seed=TEACHER_SEED, resume_parameter_training=True,
        recover_completed_strategy_compare=False,
        first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        train_window_months=int(policy["train_window_months"]),
        oos_months=int(policy["oos_horizon_months"]),
        output_relative_dir=path.parent.parent.parent,
        parameter_set_identity=HISTORY_ID,
        strategy_id="daily", daily_search_profile=PROFILE,
        build_context_label="Historical Daily Min/P19 PIT label teacher",
    )
    if Path(result["params_path"]).resolve() != path.resolve():
        raise FormalMinTrade40PITBlocked("Canonical optimizer published unexpected P19 teacher")
    _validated_records(path, historical=True, project_root=root)
    return path


def inspect_current(*, project_root: str | Path, seed: int) -> dict[str, object]:
    # P19's single-seed optimizer is explicitly independent of DL FIT seed.
    return inspect_formal_mintrade40_p16_teacher(
        project_root=project_root, seed=TEACHER_SEED, baseline_arm=BASELINE_ARM,
    )


def execute_current(*, project_root: str | Path, seed: int):
    return _current_p16_seed_artifact(
        Path(project_root).resolve(), TEACHER_SEED,
        authorized_build=True, baseline_arm=BASELINE_ARM,
    )


def load_training_source(*, project_root: str | Path, seed: int,
                         authorized_build: bool = False) -> FormalMinTrade40PITSource:
    root = Path(project_root).resolve()
    if type(seed) is not int or seed <= 0:
        raise FormalMinTrade40PITBlocked("P19 model FIT needs an explicit DL seed")
    current_path, start, end, _policy = _current_p16_seed_artifact(
        root, TEACHER_SEED, authorized_build=authorized_build,
        baseline_arm=BASELINE_ARM,
    )
    if authorized_build:
        execute_history(project_root=root, seed=seed)
    history_path = historical_params_path(root)
    history, historical_records = _validated_records(history_path, historical=True, project_root=root)
    current, current_sha = load_strategy_param_evaluation_view(
        current_path, evaluation_mode="rolling", start_date=start, end_date=end,
    )
    if current.get("kind") != "rolling_active_param_ensemble":
        raise FormalMinTrade40PITBlocked("P19 requires canonical single-member Rolling effective parameters")
    current_records = resolve_formal_min_param_records(current["kind"], current["payload"])
    for row in current_records:
        verify_daily_min_p19_params(row["params_obj"])
        if any(member.get("seed") != TEACHER_SEED
               for member in tuple(row.get("members") or ())):
            raise FormalMinTrade40PITBlocked("P19 current teacher seed identity mismatch")
    from core.strategy_param_artifacts import compute_strategy_param_file_sha256
    hist_sha = compute_strategy_param_file_sha256(history_path)
    history_periods, _ = build_formal_min_execution_schedule(
        historical_records, seed=TEACHER_SEED, param_sha256=hist_sha,
    )
    current_periods, _ = build_formal_min_execution_schedule(
        current_records, seed=TEACHER_SEED, param_sha256=current_sha,
    )
    if (not history_periods or not current_periods
            or current_periods[0].start != pd.Timestamp(start).normalize()
            or history_periods[-1].end is None
            or history_periods[-1].end + pd.Timedelta(days=1) != current_periods[0].start):
        raise FormalMinTrade40PITBlocked("P19 historical/current PIT parameter periods not contiguous")
    # The genuine Optimizer fold end is the last day legally known before
    # the next effective date. A future-effective parameter cannot label past rows.
    def publications(payload, periods):
        folds = tuple(payload.get("folds") or ())
        by_start = {pd.Timestamp(f.get("effective_start") or f["oos_start_date"]).normalize(): f
                    for f in folds}
        if len(by_start) != len(folds) or len(folds) != len(periods):
            raise FormalMinTrade40PITBlocked("P19 Optimizer fold evidence incomplete")
        pub = []
        for period in periods:
            fold = by_start.get(period.start)
            if fold is None:
                raise FormalMinTrade40PITBlocked("P19 effective fold absent")
            selected = pd.Timestamp(fold["selection_end_date"]).normalize()
            if selected != period.start - pd.Timedelta(days=1):
                raise FormalMinTrade40PITBlocked("P19 training end after effective start")
            pub.append(str(selected.date()))
        return tuple(pub)
    published = publications(history, history_periods) + publications(current["payload"], current_periods)
    return FormalMinTrade40PITSource(
        seed=TEACHER_SEED, source_id=SOURCE_ID,
        param_sha256=canonical_json_sha256({
            "contract": "daily_min_p19_historical_plus_current_v1",
            "teacher_seed": TEACHER_SEED, "history_sha256": hist_sha,
            "current_sha256": current_sha, "search_profile": PROFILE,
        }),
        schedule=tuple(replace(period, source_id=SOURCE_ID)
                       for period in (*history_periods, *current_periods)),
        publication_dates=published,
        oos_start_date=str(current_periods[0].start.date()),
    )


def resolve_loader_kwargs(*, project_root: str | Path, seed: int,
                          authorized_build: bool = False) -> dict[str, object]:
    from services.breakout_quality.daily_min_p19_pit_source import (
        materialize_daily_min_p19_pit_labels,
    )
    root = Path(project_root).resolve()
    source = load_training_source(
        project_root=root, seed=seed, authorized_build=authorized_build,
    )
    return {
        "target_materializer": partial(
            materialize_daily_min_p19_pit_labels,
            source=source, expected_seed=TEACHER_SEED,
            cache_project_root=str(root),
        ),
        "target_lineage": {
            "contract": "daily_min_p19_historical_plus_current_v1",
            "target_kind": "daily_min_p19_seed42_fullcycle_netpnl",
            "entry_policy": ENTRY_POLICY, "horizon_mode": "full_trade_cycle",
            "value_field": "raw_net_pnl", "seed": seed,
            "execution_implementation_sha256": daily_min_p19_execution_sha256(),
            "optimizer_teacher_seed": TEACHER_SEED,
            "maturity_policy": "natural_exit_or_research_terminal_closeout_pit_v1",
            "source_sha256": source.param_sha256,
            "first_effective_date": str(source.schedule[0].start.date()),
            "last_source_period_start": str(source.schedule[-1].start.date()),
            "oos_start_date": source.oos_start_date,
            "teacher_periods": [
                {"start": str(p.start.date()),
                 "end": None if p.end is None else str(p.end.date()),
                 "params_signature": p.params_signature, "published": published}
                for p, published in zip(source.schedule, source.publication_dates)
            ],
        },
    }

__all__ = ["HISTORY_ID", "SOURCE_ID", "ENTRY_POLICY", "historical_params_path",
           "inspect_history", "execute_history", "inspect_current", "execute_current",
           "load_training_source", "resolve_loader_kwargs"]
